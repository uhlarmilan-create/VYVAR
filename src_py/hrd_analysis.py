"""Hertzsprung–Russell diagram helpers from MASTERSTAR field catalog + local Gaia DR3 SQLite."""

from __future__ import annotations

import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gaia_catalog_id import normalize_gaia_source_id, normalize_gaia_source_id_series, read_vyvar_csv
from infolog import log_event

logger = logging.getLogger(__name__)

# HRD extreme-object thresholds (Stage 2) — Pecaut & Mamajek (2013, ApJS 208, 9) intrinsic-color
# scale; Gaia BP-RP breakpoints from their updated online table (2024 revision).
# GSP-Phot teff/logg provenance: Andrae et al. (2023, A&A 674, A27); incomplete at color extremes.
HRD_BP_RP_VERY_HOT = -0.3  # Pecaut & Mamajek: early-O/B intrinsic BP-RP floor (Gaia era)
HRD_BP_RP_HOT_LUM = -0.1
HRD_BP_RP_VERY_COOL = 3.0  # late-M / carbon-star regime (Pecaut & Mamajek sequence tail)
HRD_BP_RP_WD = 0.3
HRD_BP_RP_GIANT = 1.5
HRD_TEFF_VERY_HOT = 25000.0
HRD_TEFF_VERY_COOL = 3000.0
HRD_LOGG_GIANT = 3.5
HRD_LOGG_SUPERGIANT = 1.5
HRD_ABS_MAG_LUMINOUS = 0.0
HRD_ABS_MAG_WD = 10.0

HRD_EMPTY_FIELD_MSG = "No extreme objects in this field (all stars near the main sequence)"
HRD_DETAILS_DISTANCE_CAVEAT = (
    "Distances use naive inverse parallax (1000/ω, mas); not corrected for Gaia DR3 zero-point "
    "or Lutz-Kelker bias — see Bailer-Jones et al. (2021)."
)
HRD_CAPTION = (
    "HRD positions use Gaia DR3 catalog values (BP-RP, M_G). Differences between filter sessions "
    "reflect which stars DAO detected in that band (detection selection), not the measured photometry. "
    "Gray points use apparent G (no reliable distance) and appear fainter than the absolute-magnitude "
    "sequence by their distance modulus."
)

HRD_PARALLAX_MIN_MAS_DEFAULT = 0.15
HRD_PARALLAX_SNR_MIN_DEFAULT = 5.0


def hrd_nss_category_enabled(cfg: Any | None) -> bool:
    """True when Gaia NSS binary category is active in HRD Stage-1/2."""
    if cfg is None:
        return False
    return bool(getattr(cfg, "hrd_nss_category_enabled", False))


def hrd_parallax_params_from_cfg(cfg: Any | None) -> tuple[float, float]:
    """Resolve HRD parallax gate from config (defaults: 0.15 mas floor, SNR 5)."""
    pmin = HRD_PARALLAX_MIN_MAS_DEFAULT
    snr = HRD_PARALLAX_SNR_MIN_DEFAULT
    if cfg is not None:
        try:
            pmin = float(getattr(cfg, "hrd_parallax_min_mas", pmin))
        except (TypeError, ValueError):
            pmin = HRD_PARALLAX_MIN_MAS_DEFAULT
        try:
            snr = float(getattr(cfg, "hrd_parallax_snr_min", snr))
        except (TypeError, ValueError):
            snr = HRD_PARALLAX_SNR_MIN_DEFAULT
        pmin = max(0.0, min(10.0, pmin))
        snr = max(1.0, min(20.0, snr))
    return pmin, snr


def is_hrd_parallax_reliable(
    parallax_mas: float | None,
    parallax_over_error: float | None,
    *,
    parallax_min_mas: float = HRD_PARALLAX_MIN_MAS_DEFAULT,
    parallax_snr_min: float = HRD_PARALLAX_SNR_MIN_DEFAULT,
) -> bool:
    """True when parallax passes the HRD M_G reliability gate."""
    try:
        p = float(parallax_mas)
        snr = float(parallax_over_error)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(p) and math.isfinite(snr)):
        return False
    return p >= float(parallax_min_mas) and snr >= float(parallax_snr_min)

_LABEL_VERY_HOT = "Very hot (O/B candidate)"
_LABEL_HOT_LUM = "Hot luminous (WR/OB? candidate)"
_LABEL_VERY_COOL = "Very cool (late-M/C candidate)"
_LABEL_WD = "White dwarf candidate"
_LABEL_RG = "Red giant"
_LABEL_RSG = "Red supergiant"
_LABEL_BINARY = "Binary candidate (Gaia NSS)"

# Stage-2 priority (luminosity-informed; most specific wins).
_LABEL_PRIORITY: tuple[str, ...] = (
    _LABEL_WD,
    _LABEL_RSG,
    _LABEL_RG,
    _LABEL_HOT_LUM,
    _LABEL_VERY_HOT,
    _LABEL_VERY_COOL,
    _LABEL_BINARY,
)

_SIMBAD_OTYPE_PREFIXES = ("WR", "WD", "C", "S", "LP", "EB", "RG", "AGB", "PN", "HV")

_LUM_CLASS_RE = re.compile(r"(IAB|IB|IA|I|II|III|IV|V)\s*$", re.IGNORECASE)
_LUM_CLASS_EMBED_RE = re.compile(r"\d(?:\.\d)?(IAB|IB|IA|I|II|III|IV|V)", re.IGNORECASE)
_IDENT_TIERS = ("confirmed", "likely", "candidate")

# Display names for confirmed-tier labels (drop "candidate").
_CONFIRMED_NAME: dict[str, str] = {
    _LABEL_WD: "White dwarf",
    _LABEL_RSG: "Red supergiant",
    _LABEL_RG: "Red giant",
    _LABEL_VERY_HOT: "Very hot (O/B)",
    _LABEL_VERY_COOL: "Very cool (late-M/C)",
    _LABEL_HOT_LUM: "Hot luminous (WR/OB?)",
    _LABEL_BINARY: "Binary (Gaia NSS)",
}


def hrd_dsc_confirm_prob_from_cfg(cfg: Any | None) -> float:
    """Resolve HRD DSC 'likely' probability floor (default 0.90, clamp 0.5..1.0)."""
    prob = 0.90
    if cfg is not None:
        try:
            prob = float(getattr(cfg, "hrd_dsc_confirm_prob", prob))
        except (TypeError, ValueError):
            prob = 0.90
    return max(0.5, min(1.0, float(prob)))


def _sp_normalized(sp: str | None) -> str:
    return str(sp or "").upper().replace(" ", "")


def _parse_luminosity_class(sp_type: str | None) -> str | None:
    if not sp_type or not str(sp_type).strip():
        return None
    s = _sp_normalized(sp_type)
    m = _LUM_CLASS_RE.search(s)
    if m:
        return m.group(1).upper()
    m2 = _LUM_CLASS_EMBED_RE.search(s)
    if m2:
        return m2.group(1).upper()
    return None


def _effective_logg(row: pd.Series) -> tuple[float | None, str]:
    """Gaia logg wins; else SIMBAD MK luminosity class substitute for Stage-2 only."""
    logg_gaia = _f(row.get("logg_gspphot"))
    if logg_gaia is not None:
        return logg_gaia, "gaia"
    lc = _parse_luminosity_class(str(row.get("simbad_sp_type") or ""))
    if lc is None:
        return None, "n/a"
    if lc in ("I", "IA", "IAB", "IB", "II"):
        return 1.0, "simbad_lumclass"
    if lc == "III":
        return 3.0, "simbad_lumclass"
    if lc in ("IV", "V"):
        return 4.5, "simbad_lumclass"
    return None, "n/a"


def _is_wd_sp(sp: str | None) -> bool:
    s = _sp_normalized(sp)
    return len(s) >= 2 and s[0] == "D" and s[1] in "ABCQXZO"


def _is_kmc_s_type(sp: str | None) -> bool:
    s = _sp_normalized(sp)
    return bool(s) and s[0] in "KMCS"


def _is_m5_plus_or_cs(sp: str | None) -> bool:
    s = _sp_normalized(sp)
    if not s:
        return False
    if s.startswith("C") or s.startswith("S"):
        return True
    m = re.match(r"M(\d+)", s)
    return bool(m and int(m.group(1)) >= 5)


def _is_hot_sp(sp: str | None) -> bool:
    s = _sp_normalized(sp)
    if s.startswith("O"):
        return True
    m = re.match(r"B([0-9])", s)
    return bool(m and int(m.group(1)) <= 2)


def _is_wr_sp(sp: str | None) -> bool:
    s = _sp_normalized(sp)
    return s.startswith(("WN", "WC", "WO"))


def _otype_confirmed(otype: str | None) -> str | None:
    ot = str(otype or "").strip()
    if not ot or ot.endswith("?"):
        return None
    return ot


def _match_literature_confirmed(base_label: str, otype: str | None, sp_type: str | None) -> str | None:
    """Return ident_detail when literature confirms the Stage-2 category."""
    ot = _otype_confirmed(otype)
    sp = str(sp_type or "").strip()
    ot_up = (ot or "").upper().replace(" ", "")
    sp_n = _sp_normalized(sp)
    lc = _parse_luminosity_class(sp)

    if base_label == _LABEL_WD:
        if ot_up.startswith("WD"):
            return f"{sp or ot}, SIMBAD"
        if _is_wd_sp(sp):
            return f"{sp}, SIMBAD"
    elif base_label == _LABEL_RSG:
        if ot and ("S*R" in ot_up or ot_up == "S*R"):
            return f"{sp or ot}, SIMBAD"
        if lc in ("I", "IA", "IAB", "IB") and _is_kmc_s_type(sp):
            return f"{sp}, SIMBAD"
    elif base_label == _LABEL_RG:
        if ot_up.startswith("RG"):
            return f"{sp or ot}, SIMBAD"
        if lc in ("II", "III") and sp_n and sp_n[0] in "KM":
            return f"{sp}, SIMBAD"
    elif base_label == _LABEL_VERY_COOL:
        if ot:
            ot_clean = ot_up.replace(" ", "")
            if ot_clean.startswith(("LP", "MI")):
                return f"{sp or ot}, SIMBAD"
            if ot_clean.startswith("C*") or (ot_clean.startswith("C") and "C*" not in ot_clean[:2]):
                return f"{sp or ot}, SIMBAD"
            if ot_clean.startswith("S*") and not any(x in ot_clean for x in ("S*R", "S*B")):
                return f"{sp or ot}, SIMBAD"
        if _is_m5_plus_or_cs(sp):
            return f"{sp}, SIMBAD"
    elif base_label == _LABEL_VERY_HOT:
        if ot_up.startswith("BE") or "S*B" in ot_up:
            return f"{sp or ot}, SIMBAD"
        if _is_hot_sp(sp):
            return f"{sp}, SIMBAD"
    elif base_label == _LABEL_HOT_LUM:
        if ot_up.startswith("WR"):
            return "SIMBAD"
        if _is_wr_sp(sp):
            return f"{sp}, SIMBAD"
    elif base_label == _LABEL_BINARY:
        if ot and ("*" in ot or "BIN" in ot_up or ot_up.startswith("EB")):
            return f"{sp or ot}, SIMBAD"
    return None


def _dsc_likely_detail(row: pd.Series, base_label: str, threshold: float) -> str | None:
    if base_label != _LABEL_WD:
        return None
    p = _f(row.get("classprob_dsc_combmod_whitedwarf"))
    if p is not None and p >= float(threshold):
        return f"likely, DSC p={p:.2f}"
    return None


def _render_ident_label(base_label: str, tier: str, ident_detail: str) -> str:
    if tier == "confirmed":
        if base_label == _LABEL_HOT_LUM:
            return "Wolf-Rayet (SIMBAD)"
        short = _CONFIRMED_NAME.get(base_label, base_label.replace(" candidate", ""))
        if ident_detail:
            return f"{short} ({ident_detail})"
        return short
    if tier == "likely" and base_label == _LABEL_WD:
        return f"White dwarf ({ident_detail})"
    return base_label


def _finalize_ident(
    row: pd.Series,
    base_label: str,
    *,
    dsc_threshold: float,
    enrichment_active: bool,
) -> tuple[str, str, str, str]:
    """Return (tier, display_category, ident_detail, logg_source)."""
    _, logg_source = _effective_logg(row)
    if not enrichment_active:
        return "candidate", base_label, "", logg_source

    lit = _match_literature_confirmed(
        base_label, row.get("simbad_otype"), row.get("simbad_sp_type")
    )
    if lit:
        return (
            "confirmed",
            _render_ident_label(base_label, "confirmed", lit),
            lit,
            logg_source,
        )

    dsc = _dsc_likely_detail(row, base_label, dsc_threshold)
    if dsc:
        return (
            "likely",
            _render_ident_label(base_label, "likely", dsc),
            dsc,
            logg_source,
        )

    detail = ""
    ot = str(row.get("simbad_otype") or "").strip()
    if ot.endswith("?"):
        detail = f"SIMBAD {ot} (uncertain)"
    return "candidate", base_label, detail, logg_source


def _f(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return None
    try:
        x = float(val)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _gaia_table_name(conn: sqlite3.Connection) -> str:
    """Prefer ``gaia_dr3`` (VYVAR default) over ``gaia_source`` if both exist."""
    for t in ("gaia_dr3", "gaia_source"):
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
            (t,),
        ).fetchone()
        if row:
            return t
    return "gaia_dr3"


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info('{table}');")
    return {str(r[1]).strip().lower() for r in cur.fetchall()}


def _fetch_gaia_columns_by_source_id(
    gaia_db_path: Path,
    catalog_ids: list[str],
    want: list[str],
) -> pd.DataFrame:
    """Batch-fetch optional Gaia columns keyed by source_id (as str)."""
    if not gaia_db_path.is_file() or not want:
        return pd.DataFrame()
    ids_u: list[int] = []
    seen: set[int] = set()
    for raw in catalog_ids:
        s = str(raw or "").strip()
        if not s or not s.isdigit():
            continue
        try:
            sid = int(s)
        except (TypeError, ValueError, OverflowError):
            continue
        if sid in seen:
            continue
        seen.add(sid)
        ids_u.append(sid)
    if not ids_u:
        return pd.DataFrame()

    conn = sqlite3.connect(str(gaia_db_path))
    conn.row_factory = sqlite3.Row
    try:
        table = _gaia_table_name(conn)
        cols_db = _table_columns(conn, table)
        if "source_id" not in cols_db:
            return pd.DataFrame()

        parts: list[str] = ["source_id"]
        seen_sel: set[str] = {"source_id"}

        for w in want:
            wl = (w or "").strip().lower()
            if not wl or wl == "parallax_over_error":
                continue
            if wl in cols_db and wl not in seen_sel:
                parts.append(wl)
                seen_sel.add(wl)
                continue
            if wl == "phot_g_mean_mag" and "g_mag" in cols_db and "g_mag" not in seen_sel:
                parts.append("g_mag AS phot_g_mean_mag")
                seen_sel.add("g_mag")
                continue

        if "parallax_over_error" in {x.lower() for x in want}:
            if (
                "parallax" in cols_db
                and "parallax_error" in cols_db
                and "parallax_over_error" not in seen_sel
            ):
                parts.append("(parallax / NULLIF(parallax_error, 0)) AS parallax_over_error")
                seen_sel.add("parallax_over_error")

        sel = ", ".join(parts)
        out_rows: list[dict[str, Any]] = []
        bs = 500
        for i0 in range(0, len(ids_u), bs):
            chunk = ids_u[i0 : i0 + bs]
            ph = ",".join("?" * len(chunk))
            q = f"SELECT {sel} FROM {table} WHERE source_id IN ({ph});"
            for row in conn.execute(q, chunk):
                d = {key: row[key] for key in row.keys()}
                sid0 = d.get("source_id")
                if sid0 is None:
                    continue
                d["catalog_id"] = str(int(sid0))
                out_rows.append(d)
        if not out_rows:
            return pd.DataFrame()
        return pd.DataFrame(out_rows)
    finally:
        conn.close()


def _spectral_class(row: pd.Series) -> str:
    t = _f(row.get("teff_gspphot"))
    g = _f(row.get("logg_gspphot"))
    bprp = _f(row.get("bp_rp"))
    if t is None and bprp is None:
        return ""
    if t is not None:
        if t >= 30000:
            return "O–B"
        if t >= 10000:
            return "B–A"
        if t >= 7500:
            return "A–F"
        if t >= 6000:
            return "F–G"
        if t >= 5200:
            return "G"
        if t >= 3700:
            return "K"
        if t >= 2400:
            return "M"
        return "cool"
    if bprp is not None:
        if bprp < 0.0:
            return "hot (BP−RP)"
        if bprp < 0.5:
            return "early"
        if bprp < 1.2:
            return "solar-type"
        if bprp < 2.0:
            return "late"
        return "very cool"
    if g is not None and g < 3.5:
        return "evolved (log g)"
    return ""


def _is_reliable(row: pd.Series) -> bool:
    return bool(row.get("hrd_reliable"))


def _nss_is_binary(row: pd.Series) -> bool:
    try:
        return int(float(row.get("non_single_star", 0))) == 1
    except (TypeError, ValueError):
        return False


def _simbad_suffix(otype: Any) -> str:
    ot = str(otype or "").strip()
    if not ot:
        return ""
    up = ot.upper()
    for pref in _SIMBAD_OTYPE_PREFIXES:
        if up.startswith(pref) or pref in up.split("*")[0]:
            return f" (SIMBAD: {ot})"
    return ""


def _stage2_labels(row: pd.Series, *, nss_enabled: bool = False) -> list[str]:
    """Return all Stage-2 category labels satisfied by one star (before priority pick)."""
    bp_rp = _f(row.get("bp_rp"))
    logg, _ = _effective_logg(row)
    abs_g = _f(row.get("abs_mag_g"))
    teff = _f(row.get("teff_gspphot"))
    reliable = _is_reliable(row)
    labels: list[str] = []

    if reliable and bp_rp is not None and abs_g is not None:
        if bp_rp < HRD_BP_RP_WD and abs_g > HRD_ABS_MAG_WD:
            labels.append(_LABEL_WD)
        if bp_rp < HRD_BP_RP_HOT_LUM and abs_g < HRD_ABS_MAG_LUMINOUS:
            labels.append(_LABEL_HOT_LUM)

    if teff is not None and teff >= HRD_TEFF_VERY_HOT:
        labels.append(_LABEL_VERY_HOT)
    elif teff is None and bp_rp is not None and bp_rp <= HRD_BP_RP_VERY_HOT:
        labels.append(_LABEL_VERY_HOT)

    is_giant_branch = (
        bp_rp is not None
        and logg is not None
        and bp_rp > HRD_BP_RP_GIANT
        and logg < HRD_LOGG_GIANT
    )
    if is_giant_branch:
        if logg < HRD_LOGG_SUPERGIANT:
            labels.append(_LABEL_RSG)
        else:
            labels.append(_LABEL_RG)

    if not is_giant_branch:
        if teff is not None and teff <= HRD_TEFF_VERY_COOL:
            labels.append(_LABEL_VERY_COOL)
        elif bp_rp is not None and bp_rp >= HRD_BP_RP_VERY_COOL:
            labels.append(_LABEL_VERY_COOL)

    if nss_enabled and _nss_is_binary(row):
        labels.append(_LABEL_BINARY)

    return labels


def _pick_base_label(labels: list[str]) -> str:
    if not labels:
        return ""
    chosen = ""
    best = len(_LABEL_PRIORITY) + 1
    for lab in labels:
        try:
            idx = _LABEL_PRIORITY.index(lab)
        except ValueError:
            continue
        if idx < best:
            best = idx
            chosen = lab
    return chosen


def _pick_label(labels: list[str], simbad_otype: Any = None) -> str:
    chosen = _pick_base_label(labels)
    if not chosen:
        return ""
    suffix = _simbad_suffix(simbad_otype)
    return f"{chosen}{suffix}" if chosen else ""


def _classify_star(row: pd.Series) -> str:
    return _pick_label(_stage2_labels(row), row.get("simbad_otype"))


def build_hrd_dataframe(
    masterstars_csv: Path,
    gaia_db_path: Path,
    *,
    parallax_min_mas: float = HRD_PARALLAX_MIN_MAS_DEFAULT,
    parallax_snr_min: float = HRD_PARALLAX_SNR_MIN_DEFAULT,
) -> pd.DataFrame:
    """Build HRD dataframe from masterstars CSV + optional Gaia SQLite enrichment."""
    ms = read_vyvar_csv(masterstars_csv, low_memory=False)
    if ms.empty:
        return ms
    # Keep only stars with DAO detection (dao_flux or flux > 0)
    # dao_flux column name varies — try both
    _flux_col = None
    for _candidate in ("dao_flux", "flux", "peak_dao"):
        if _candidate in ms.columns:
            _numeric = pd.to_numeric(ms[_candidate], errors="coerce")
            if (_numeric > 0).any():
                _flux_col = _candidate
                break

    if _flux_col is not None:
        _before = len(ms)
        ms = ms[pd.to_numeric(ms[_flux_col], errors="coerce") > 0].copy()
        _after = len(ms)
        if _before != _after:
            log_event(
                f"HRD: filtered to DAO-detected stars only "
                f"({_after}/{_before} via {_flux_col})"
            )
    else:
        log_event("HRD: no flux column found — showing all masterstar rows")

    ms["catalog_id"] = normalize_gaia_source_id_series(ms["catalog_id"])
    if "phot_g_mean_mag" not in ms.columns and "g_mag" in ms.columns:
        ms["phot_g_mean_mag"] = pd.to_numeric(ms["g_mag"], errors="coerce")

    needed = ["teff_gspphot", "logg_gspphot", "parallax", "parallax_over_error", "non_single_star"]
    missing = [c for c in needed if c not in ms.columns]
    # CSV často obsahuje prázdne stĺpce z matchu — potom ``missing`` je prázdne a merge by zahodil
    # stĺpce z Gaia (starý dup_cols). Doplň fetch aj pre stĺpce bez jedinej platnej hodnoty.
    for c in ("teff_gspphot", "logg_gspphot", "non_single_star"):
        if c in ms.columns and c not in missing:
            num = pd.to_numeric(ms[c], errors="coerce")
            if int(num.notna().sum()) == 0:
                missing.append(c)
    if missing and gaia_db_path.is_file():
        gdf = _fetch_gaia_columns_by_source_id(
            Path(gaia_db_path),
            ms["catalog_id"].tolist(),
            missing + (["parallax_error"] if "parallax_over_error" in missing else []),
        )
        if not gdf.empty and "catalog_id" in gdf.columns:
            gdf = gdf.drop(columns=[c for c in ("source_id",) if c in gdf.columns], errors="ignore")
            # Povoliť prepis hodnôt z SQLite namiesto zahodenia duplicitných stĺpcov z gdf.
            overlap = [c for c in gdf.columns if c != "catalog_id" and c in ms.columns]
            if overlap:
                ms = ms.drop(columns=overlap, errors="ignore")
            ms = ms.merge(gdf, on="catalog_id", how="left", suffixes=("", "_gaia"))

    if "parallax_over_error" not in ms.columns and "parallax" in ms.columns and "parallax_error" in ms.columns:
        p = pd.to_numeric(ms["parallax"], errors="coerce")
        pe = pd.to_numeric(ms["parallax_error"], errors="coerce")
        ms["parallax_over_error"] = np.where((pe > 0) & np.isfinite(p) & np.isfinite(pe), p / pe, np.nan)

    p = pd.to_numeric(ms.get("parallax"), errors="coerce")
    snr = pd.to_numeric(ms.get("parallax_over_error"), errors="coerce")
    g = pd.to_numeric(ms.get("phot_g_mean_mag"), errors="coerce")
    # Parallax reliability: SNR primary filter (Bailer-Jones et al. 2021, AJ 161, 147); mas floor
    # excludes only the near-zero regime where 1/parallax breaks down (not a distance-quality cut).
    # DR3 zero-point bias is negligible at SNR>=5 for ~0.4 mas (Lindegren et al. 2021, A&A 649, A4).
    # No zero-point correction or Bayesian distances here (diagnostic HRD only).
    ok = (
        np.isfinite(p)
        & np.isfinite(snr)
        & np.isfinite(g)
        & (p >= float(parallax_min_mas))
        & (snr >= float(parallax_snr_min))
    )
    dist_pc = np.where(ok, 1000.0 / p, np.nan)
    abs_mag = np.where(ok, g + 5.0 - 5.0 * np.log10(dist_pc), np.nan)
    ms["abs_mag_g"] = abs_mag
    ms["hrd_reliable"] = np.isfinite(abs_mag)
    ms["spectral_class"] = ms.apply(_spectral_class, axis=1)
    ms["interesting_label"] = ms.apply(_classify_star, axis=1)
    return ms


def _fmt(val: Any, spec: str, *, na: bool = False) -> str:
    missing = "N/A" if na else "\u2014"
    if val is None or (isinstance(val, str) and not str(val).strip()):
        return missing
    try:
        f = float(val)
        return format(f, spec) if math.isfinite(f) else missing
    except (TypeError, ValueError):
        return missing


def _fmt_raw_str(val: Any, *, table_na: bool = False) -> str:
    s = str(val or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return "N/A" if table_na else "\u2014"
    return s


def format_hrd_dsc_wd_p(val: Any) -> str:
    """Format Gaia DSC white-dwarf probability (~2 significant figures)."""
    f = _f(val)
    if f is None:
        return "N/A"
    if f == 0.0:
        return "0.0"
    return f"{f:.2g}"


def format_hrd_sexagesimal(ra_deg: Any, dec_deg: Any) -> str:
    """RA/Dec sexagesimal (J2000); fall back to decimal degrees on error."""
    try:
        ra = float(ra_deg)
        dec = float(dec_deg)
        if not (math.isfinite(ra) and math.isfinite(dec)):
            raise ValueError("non-finite coordinates")
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        ra_s = coord.ra.to_string(unit=u.hour, sep=":", precision=1, pad=True)
        dec_s = coord.dec.to_string(sep=":", alwayssign=True, precision=1, pad=True)
        return f"{ra_s} {dec_s}"
    except Exception:  # noqa: BLE001
        return f"{ra_deg} {dec_deg}"


def build_hrd_detail_line(row: dict[str, Any] | pd.Series) -> str:
    """Assemble the pipe-separated detail line for PDF/UI (Part 2 spec)."""
    rr = dict(row)
    parts: list[str] = []

    ra = rr.get("ra_deg")
    dec = rr.get("dec_deg")
    if ra not in (None, "", "N/A", "\u2014") and dec not in (None, "", "N/A", "\u2014"):
        sex = str(rr.get("ra_dec_sex") or "").strip()
        if not sex:
            sex = format_hrd_sexagesimal(ra, dec)
        parts.append(f"RA/Dec (J2000): {sex}")

    for key, label in (
        ("mag_g", "G"),
        ("abs_mag_g", "M_G"),
        ("bp_rp", "BP-RP"),
    ):
        val = rr.get(key)
        if val not in (None, "", "N/A", "\u2014"):
            parts.append(f"{label}={val}")

    dist = rr.get("dist_pc")
    if dist not in (None, "", "N/A", "\u2014"):
        plx = rr.get("parallax_mas", "N/A")
        snr = rr.get("parallax_snr", "N/A")
        parts.append(f"dist={dist} pc (plx {plx} mas, SNR {snr})")

    teff = rr.get("teff", "N/A")
    teff_src = rr.get("teff_source", "n/a")
    parts.append(f"Teff={teff} K ({teff_src})")
    logg = rr.get("logg", "N/A")
    logg_src = rr.get("logg_source", "n/a")
    parts.append(f"logg={logg} ({logg_src})")

    spt = rr.get("sp_type_raw")
    if spt not in (None, "", "N/A", "\u2014"):
        parts.append(f"SpT={spt}")
    otype = rr.get("otype_raw")
    if otype not in (None, "", "N/A", "\u2014"):
        parts.append(f"otype={otype}")

    dsc = rr.get("dsc_wd_p")
    if dsc not in (None, "", "N/A", "\u2014"):
        parts.append(f"DSC WD p={dsc}")

    x_px = rr.get("x_px")
    y_px = rr.get("y_px")
    if x_px not in (None, "", "N/A", "\u2014") and y_px not in (None, "", "N/A", "\u2014"):
        parts.append(f"pix=({x_px}, {y_px})")

    return " | ".join(parts)


def hrd_detail_header_name(row: dict[str, Any] | pd.Series) -> str:
    main_id = str(dict(row).get("simbad_main_id") or "").strip()
    if main_id:
        return main_id
    cid = str(dict(row).get("catalog_id") or "").strip()
    return f"Gaia DR3 {cid}" if cid else "Unknown"


def _make_row(
    row: pd.Series,
    category: str,
    *,
    ident: str = "candidate",
    ident_detail: str = "",
    table_na: bool = False,
) -> dict[str, Any]:
    simbad_id = str(row.get("simbad_main_id") or "").strip()
    simbad_otype = str(row.get("simbad_otype") or "").strip()
    simbad_disp = simbad_id
    if simbad_otype:
        simbad_disp = f"{simbad_id} ({simbad_otype})" if simbad_id else simbad_otype
    if ident_detail and ident != "confirmed":
        simbad_disp = f"{simbad_disp}; {ident_detail}" if simbad_disp else ident_detail
    src = str(row.get("enrich_source") or "local").strip() or "local"
    logg_src = str(row.get("_logg_source") or "n/a").strip() or "n/a"
    teff_src = "gaia" if _f(row.get("teff_gspphot")) is not None else "n/a"
    reliable = bool(row.get("hrd_reliable"))
    parallax = _f(row.get("parallax"))
    parallax_snr = _f(row.get("parallax_over_error"))
    if reliable and parallax is not None and parallax > 0:
        dist_pc_val = 1000.0 / parallax
        dist_pc = _fmt(dist_pc_val, ".1f", na=table_na)
        parallax_mas = _fmt(parallax, ".2f", na=table_na)
        parallax_snr_s = _fmt(parallax_snr, ".2f", na=table_na)
    else:
        dist_pc = "N/A" if table_na else "\u2014"
        parallax_mas = "N/A" if table_na else "\u2014"
        parallax_snr_s = "N/A" if table_na else "\u2014"
    ra_raw = row.get("ra_deg")
    dec_raw = row.get("dec_deg")
    ra_dec_sex = ""
    if _f(ra_raw) is not None and _f(dec_raw) is not None:
        ra_dec_sex = format_hrd_sexagesimal(ra_raw, dec_raw)
    dsc_wd_p = format_hrd_dsc_wd_p(row.get("classprob_dsc_combmod_whitedwarf"))
    return {
        "catalog_id": str(row.get("catalog_id", "")),
        "category": category,
        "ident": ident if ident in _IDENT_TIERS else "candidate",
        "ident_detail": ident_detail,
        "simbad_id": simbad_disp,
        "simbad_main_id": simbad_id,
        "mag_g": _fmt(row.get("phot_g_mean_mag"), ".2f", na=table_na),
        "abs_mag_g": _fmt(row.get("abs_mag_g"), ".2f", na=table_na),
        "bp_rp": _fmt(row.get("bp_rp"), ".3f", na=table_na),
        "teff": _fmt(row.get("teff_gspphot"), ".0f", na=table_na),
        "logg": _fmt(row.get("logg_gspphot"), ".2f", na=table_na),
        "src": src if src else "N/A",
        "logg_source": logg_src,
        "teff_source": teff_src,
        "dist_pc": dist_pc,
        "parallax_mas": parallax_mas,
        "parallax_snr": parallax_snr_s,
        "sp_type_raw": _fmt_raw_str(row.get("simbad_sp_type"), table_na=table_na),
        "otype_raw": _fmt_raw_str(row.get("simbad_otype"), table_na=table_na),
        "dsc_wd_p": dsc_wd_p,
        "ra_deg": _fmt(ra_raw, ".4f"),
        "dec_deg": _fmt(dec_raw, ".4f"),
        "ra_dec_sex": ra_dec_sex,
        "x_px": _fmt(row.get("x"), ".0f"),
        "y_px": _fmt(row.get("y"), ".0f"),
        "_empty_field": False,
    }


def _empty_field_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "catalog_id": "",
                "category": HRD_EMPTY_FIELD_MSG,
                "ident": "",
                "ident_detail": "",
                "simbad_id": "",
                "simbad_main_id": "",
                "dist_pc": "",
                "parallax_mas": "",
                "parallax_snr": "",
                "sp_type_raw": "",
                "otype_raw": "",
                "dsc_wd_p": "",
                "teff_source": "",
                "ra_dec_sex": "",
                "logg_source": "",
                "mag_g": "",
                "abs_mag_g": "",
                "bp_rp": "",
                "teff": "",
                "logg": "",
                "src": "",
                "_empty_field": True,
            }
        ]
    )


def _nss_mask(df: pd.DataFrame) -> pd.Series:
    if "non_single_star" in df.columns:
        return pd.to_numeric(df["non_single_star"], errors="coerce").fillna(0).astype(int) == 1
    return pd.Series(False, index=df.index)


def _stage1_net_masks(df: pd.DataFrame, *, nss_enabled: bool = False) -> dict[str, pd.Series]:
    """Stage-1 candidate nets (not mutually exclusive)."""
    bp = pd.to_numeric(df.get("bp_rp"), errors="coerce")
    abs_g = pd.to_numeric(df.get("abs_mag_g"), errors="coerce")
    reliable = df.get("hrd_reliable", pd.Series(False, index=df.index)).astype(bool)
    nets: dict[str, pd.Series] = {
        "blue": bp <= -0.1,
        "red": bp >= 2.5,
        "wd": reliable & (bp < 0.5) & (abs_g > 9.0),
        "luminous": reliable & (abs_g < -2.0),
    }
    if nss_enabled:
        nets["nss"] = _nss_mask(df)
    return nets


def _physics_net_mask(df: pd.DataFrame, *, nss_enabled: bool = False) -> pd.Series:
    nets = _stage1_net_masks(df, nss_enabled=nss_enabled)
    return nets["blue"] | nets["red"] | nets["wd"] | nets["luminous"]


def _stage1_candidate_mask(df: pd.DataFrame, *, nss_enabled: bool = False) -> pd.Series:
    nets = _stage1_net_masks(df, nss_enabled=nss_enabled)
    mask = nets["blue"] | nets["red"] | nets["wd"] | nets["luminous"]
    if nss_enabled:
        mask = mask | nets["nss"]
    return mask


def _add_extremity_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["bp_rp"] = pd.to_numeric(work.get("bp_rp"), errors="coerce")
    work["abs_mag_g"] = pd.to_numeric(work.get("abs_mag_g"), errors="coerce")
    med = float(work["bp_rp"].median()) if work["bp_rp"].notna().any() else 0.0
    work["_ext_bp"] = (work["bp_rp"] - med).abs()
    work["_ext_abs"] = work["abs_mag_g"].abs()
    return work


def _shrink_net_reservations(alloc: dict[str, int], budget: int) -> dict[str, int]:
    """Round-robin shrink per-net reservations until sum <= budget."""
    nets = list(alloc.keys())
    while sum(alloc.values()) > budget:
        changed = False
        for net in nets:
            if sum(alloc.values()) <= budget:
                break
            if alloc.get(net, 0) > 0:
                alloc[net] -= 1
                changed = True
        if not changed:
            break
    return alloc


def _select_stage1_candidates(
    df: pd.DataFrame,
    max_candidates: int,
    *,
    min_per_net: int = 4,
    nss_enabled: bool = False,
) -> pd.DataFrame:
    """Stage-1 candidate pick with per-net reserved slots then global extremity fill.

    Each net (blue, red, WD-box, luminous, NSS) receives up to ``min_per_net`` guaranteed
    slots in the enrich budget, ranked within the net (luminous: ascending ``abs_mag_g``;
    NSS: input order). NSS reservations are applied last among nets. If total reservations
    exceed ``max_candidates``, shrink round-robin (blue -> red -> wd -> luminous -> nss)
    until the sum fits, then fill any remaining budget from global |bp_rp - median| ranking.
    """
    if df.empty:
        return df
    work = _add_extremity_columns(df)
    mask = _stage1_candidate_mask(work, nss_enabled=nss_enabled)
    cand = work.loc[mask].copy()
    if cand.empty:
        return cand

    cap = max(1, int(max_candidates))
    reserve = max(0, int(min_per_net))
    net_masks = _stage1_net_masks(cand, nss_enabled=nss_enabled)
    net_order: tuple[str, ...] = ("blue", "red", "wd", "luminous")
    if nss_enabled:
        net_order = net_order + ("nss",)

    alloc = {net: min(reserve, int(net_masks[net].sum())) for net in net_order}
    if sum(alloc.values()) > cap:
        alloc = _shrink_net_reservations(dict(alloc), cap)

    selected_idx: set[Any] = set()
    parts: list[pd.DataFrame] = []

    def _take(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0 or pool.empty:
            return pool.iloc[0:0]
        return pool.head(n)

    for net in net_order:
        n = alloc.get(net, 0)
        if n <= 0:
            continue
        pool = cand.loc[net_masks[net] & ~cand.index.isin(selected_idx)].copy()
        if net == "luminous":
            pool = pool.sort_values("abs_mag_g", ascending=True, na_position="last")
        elif net == "nss":
            pool = pool.sort_index()
        else:
            pool = pool.sort_values(["_ext_bp", "_ext_abs"], ascending=[False, False])
        pick = _take(pool, n)
        if not pick.empty:
            selected_idx.update(pick.index)
            parts.append(pick)

    remaining = cap - sum(len(p) for p in parts)
    if remaining > 0:
        pool = cand.loc[~cand.index.isin(selected_idx)].sort_values(
            ["_ext_bp", "_ext_abs"], ascending=[False, False]
        )
        parts.append(_take(pool, remaining))

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.drop(columns=["_ext_bp", "_ext_abs"], errors="ignore")


def _category_base(label: str) -> str:
    return str(label or "").split(" (SIMBAD")[0].strip()


def _display_to_base_label(display: str) -> str:
    """Map rendered category string back to Stage-2 base label (for cap / annotation colors)."""
    s = str(display or "").strip()
    if s.startswith("Wolf-Rayet"):
        return _LABEL_HOT_LUM
    for base in _LABEL_PRIORITY:
        short = _CONFIRMED_NAME.get(base, "")
        if short and s.startswith(short):
            return base
        stem = base.split(" (")[0]
        if s.startswith(stem):
            return base
    return _category_base(s)


def _apply_category_cap(
    classified: list[tuple[pd.Series, str, str, str, str, str]],
    max_per_category: int,
) -> list[dict[str, Any]]:
    """Keep at most max_per_category rows per Stage-2 base label (extremity order preserved)."""
    cap = max(1, int(max_per_category))
    counts: dict[str, int] = {}
    results: list[dict[str, Any]] = []
    for row, base_label, display, tier, ident_detail, logg_src in classified:
        if counts.get(base_label, 0) >= cap:
            continue
        counts[base_label] = counts.get(base_label, 0) + 1
        row_out = row.copy()
        row_out["_logg_source"] = logg_src
        results.append(
            _make_row(
                row_out,
                display,
                ident=tier,
                ident_detail=ident_detail,
                table_na=True,
            )
        )
    return results


def get_top_interesting_stars(
    hrd_df: pd.DataFrame,
    *,
    cfg: Any | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Return extreme-object table rows for PDF/UI (Stage 1 net -> enrich -> Stage 2 classify)."""
    if hrd_df is None or hrd_df.empty:
        return pd.DataFrame()

    max_candidates = 20
    max_per_category = 3
    min_per_net = 4
    enrich_enabled = True
    simbad_enabled = True
    dsc_threshold = hrd_dsc_confirm_prob_from_cfg(cfg)
    if cfg is not None:
        try:
            max_candidates = int(getattr(cfg, "hrd_enrich_max_candidates", max_candidates))
        except (TypeError, ValueError):
            max_candidates = 20
        max_candidates = max(1, min(100, int(max_candidates)))
        try:
            max_per_category = int(getattr(cfg, "hrd_max_per_category", max_per_category))
        except (TypeError, ValueError):
            max_per_category = 3
        max_per_category = max(1, min(20, int(max_per_category)))
        try:
            min_per_net = int(getattr(cfg, "hrd_min_per_net", min_per_net))
        except (TypeError, ValueError):
            min_per_net = 4
        min_per_net = max(0, min(20, int(min_per_net)))
        enrich_enabled = bool(getattr(cfg, "hrd_online_enrich_enabled", True))
        simbad_enabled = bool(getattr(cfg, "hrd_simbad_enrich_enabled", True))

    nss_enabled = hrd_nss_category_enabled(cfg)
    candidates = _select_stage1_candidates(
        hrd_df, max_candidates, min_per_net=min_per_net, nss_enabled=nss_enabled
    )
    if candidates.empty:
        return _empty_field_table()

    if enrich_enabled or simbad_enabled:
        from hrd_enrich import enrich_candidates  # noqa: PLC0415

        candidates = enrich_candidates(
            candidates,
            cache_path,
            enabled=enrich_enabled,
            simbad_enabled=simbad_enabled,
            timeout_s=float(getattr(cfg, "hrd_enrich_tap_timeout_s", 20.0)),
        )

    enrichment_active = bool(enrich_enabled or simbad_enabled)
    scored = _add_extremity_columns(candidates)
    scored = scored.sort_values(["_ext_bp", "_ext_abs"], ascending=[False, False])
    classified: list[tuple[pd.Series, str, str, str, str, str]] = []
    seen: set[str] = set()
    for _, row in scored.iterrows():
        cid = str(row.get("catalog_id", "")).strip()
        if cid in seen:
            continue
        base = _pick_base_label(_stage2_labels(row, nss_enabled=nss_enabled))
        if not base:
            continue
        tier, display, ident_detail, logg_src = _finalize_ident(
            row,
            base,
            dsc_threshold=dsc_threshold,
            enrichment_active=enrichment_active,
        )
        seen.add(cid)
        classified.append((row, base, display, tier, ident_detail, logg_src))

    results = _apply_category_cap(classified, max_per_category)

    if not results:
        return _empty_field_table()
    return pd.DataFrame(results)


def plot_hrd_matplotlib(
    hrd_df: pd.DataFrame,
    top_stars: pd.DataFrame,
    *,
    output_path: Path | None = None,
    obs_group: str = "",
) -> Path:
    """Render HRD scatter to PNG (matplotlib Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    reliable = hrd_df[hrd_df["hrd_reliable"] == True].copy()  # noqa: E712
    unreliable = hrd_df[hrd_df["hrd_reliable"] != True].copy()

    if not reliable.empty and reliable["bp_rp"].notna().any():
        bp_rp_vals = reliable["bp_rp"].fillna(1.0).clip(-0.5, 4.0)
        norm = plt.Normalize(-0.5, 4.0)
        colors = cm.RdYlBu_r(norm(bp_rp_vals.to_numpy(dtype=float)))
        ax.scatter(
            reliable["bp_rp"],
            reliable["abs_mag_g"],
            c=colors,
            s=8,
            alpha=0.6,
            linewidths=0,
            label=f"Reliable ({len(reliable)})",
            zorder=2,
        )

    unreliable_plot = unreliable.copy()
    if not unreliable_plot.empty:
        unreliable_plot["phot_g_mean_mag"] = pd.to_numeric(
            unreliable_plot.get("phot_g_mean_mag"), errors="coerce"
        )
        unreliable_plot["bp_rp"] = pd.to_numeric(unreliable_plot.get("bp_rp"), errors="coerce")
        unreliable_plot = unreliable_plot[
            np.isfinite(unreliable_plot["phot_g_mean_mag"]) & np.isfinite(unreliable_plot["bp_rp"])
        ]
    if not unreliable_plot.empty:
        ax.scatter(
            unreliable_plot["bp_rp"],
            unreliable_plot["phot_g_mean_mag"],
            c="gray",
            s=3,
            alpha=0.2,
            linewidths=0,
            label=f"No/low-quality parallax (apparent G) ({len(unreliable_plot)})",
            zorder=1,
        )

    if top_stars is not None and not top_stars.empty:
        if "_empty_field" in top_stars.columns:
            plot_stars = top_stars[~top_stars["_empty_field"].astype(bool)]
        else:
            plot_stars = top_stars
        for _, star in plot_stars.iterrows():
            cid = normalize_gaia_source_id(star.get("catalog_id", ""))
            match = hrd_df[hrd_df["catalog_id"] == cid]
            if match.empty:
                continue
            r = match.iloc[0]
            bprp = _f(r.get("bp_rp"))
            if bool(r.get("hrd_reliable")):
                y_val = _f(r.get("abs_mag_g"))
            else:
                y_val = _f(r.get("phot_g_mean_mag"))
            if bprp is None or y_val is None:
                continue
            ax.scatter(
                bprp,
                y_val,
                s=80,
                zorder=5,
                edgecolors="white",
                linewidths=0.8,
                facecolors="none",
            )
            lab = str(star.get("category", "")).strip()
            if len(lab) > 16:
                lab = lab[:14] + "\u2026"
            ax.annotate(
                lab,
                (bprp, y_val),
                fontsize=6,
                color="white",
                xytext=(5, 3),
                textcoords="offset points",
                zorder=6,
            )

    ax.invert_yaxis()
    ax.set_xlabel("BP − RP  [mag]", color="white", fontsize=11)
    ax.set_ylabel("M$_G$ / G  [mag]", color="white", fontsize=11)
    title = f"Field HRD -- {obs_group}" if str(obs_group or "").strip() else "Field Hertzsprung\u2013Russell diagram"
    ax.set_title(title, color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", framealpha=0.8, loc="upper left")

    sm = cm.ScalarMappable(norm=plt.Normalize(-0.5, 4.0), cmap=cm.RdYlBu_r)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("BP − RP", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    n_rel = int(hrd_df["hrd_reliable"].sum()) if "hrd_reliable" in hrd_df.columns else 0
    ax.text(
        0.02,
        0.02,
        f"N = {len(hrd_df)} stars  |  reliable: {n_rel}",
        transform=ax.transAxes,
        color="gray",
        fontsize=8,
    )
    plt.tight_layout()
    outp = Path(output_path) if output_path is not None else Path("hrd_field.png")
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outp), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return outp


def resolve_clean_field_image_path(
    platesolve_dir: Path,
    photometry_dir: Path,
    *,
    allow_field_map: bool = True,
) -> Path | None:
    """Prefer clean masterstar PNGs; ``field_map*.png`` only when allowed (not for HRD annotate)."""
    ps = Path(platesolve_dir)
    pt = Path(photometry_dir)
    for p in (ps / "masterstar_best.png", ps / "masterstar.png", pt.parent / "masterstar_best.png"):
        if p.is_file():
            return p
    if allow_field_map:
        for p in sorted(pt.parent.rglob("field_map*.png")):
            if p.is_file():
                return p
    return None


def _resolve_masterstar_fits_path(platesolve_dir: Path | None) -> Path | None:
    if platesolve_dir is None:
        return None
    ps = Path(platesolve_dir)
    candidates: list[Path] = []
    for name in ("MASTERSTAR.fits", "masterstar.fits"):
        p = ps / name
        if p.is_file():
            candidates.append(p)
    candidates.extend(sorted(ps.glob("MASTERSTAR*.fits")))
    seen: set[Path] = set()
    for fits_path in candidates:
        fp = fits_path.resolve()
        if fp in seen:
            continue
        seen.add(fp)
        return fp
    return None


def _draft_dir_from_photometry(photometry_dir: Path) -> Path:
    """``.../Drafts/draft_XXX/platesolve/<obs>/photometry`` -> draft root."""
    return Path(photometry_dir).resolve().parent.parent.parent


def _obs_group_from_photometry(photometry_dir: Path) -> str:
    return str(Path(photometry_dir).resolve().parent.name)


def _masterstar_frame_score(df: pd.DataFrame) -> pd.Series:
    """Same spirit as ``photometry_report._compute_masterstar_score`` / UI dashboard (higher = better)."""
    score = pd.Series(0.0, index=df.index)
    if df.empty:
        return score

    def _norm_inverse(s: pd.Series) -> pd.Series:
        mn, mx = float(s.min()), float(s.max())
        if not (np.isfinite(mn) and np.isfinite(mx) and mx > mn):
            return pd.Series(1.0, index=s.index)
        return 1.0 - (s - mn) / (mx - mn)

    def _norm_direct(s: pd.Series) -> pd.Series:
        mn, mx = float(s.min()), float(s.max())
        if not (np.isfinite(mn) and np.isfinite(mx) and mx > mn):
            return pd.Series(1.0, index=s.index)
        return (s - mn) / (mx - mn)

    fwhm = pd.to_numeric(df.get("FWHM_PX"), errors="coerce")
    elong = pd.to_numeric(df.get("ELONGATION"), errors="coerce")
    stars = pd.to_numeric(df.get("STAR_COUNT"), errors="coerce")
    sky = pd.to_numeric(df.get("SKY_LEVEL"), errors="coerce")

    if fwhm.notna().sum() >= 2:
        score += 0.45 * _norm_inverse(fwhm.fillna(fwhm.max()))
    if elong.notna().sum() >= 2:
        score += 0.30 * _norm_inverse(elong.fillna(elong.max()))
    if stars.notna().sum() >= 2:
        score += 0.15 * _norm_direct(stars.fillna(stars.min()))
    if sky.notna().sum() >= 2:
        score += 0.10 * _norm_inverse(sky.fillna(sky.max()))
    return score


def fits_first_image_to_png(
    fits_path: Path,
    png_path: Path,
    *,
    lo_pct: float = 5.0,
    hi_pct: float = 99.5,
) -> bool:
    """Render primary HDU to 8-bit RGB PNG (percentile stretch)."""
    try:
        from astropy.io import fits
        from PIL import Image

        with fits.open(fits_path, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
        if data.size == 0:
            return False
        ok = np.isfinite(data)
        if not ok.any():
            return False
        lo = float(np.nanpercentile(data[ok], float(lo_pct)))
        hi = float(np.nanpercentile(data[ok], float(hi_pct)))
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            lo = float(np.nanmin(data[ok]))
            hi = float(np.nanmax(data[ok]))
        if hi <= lo:
            hi = lo + 1e-6
        scaled = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        gray = (scaled * 255.0).astype(np.uint8)
        gray[~ok] = 0
        img = Image.fromarray(gray, mode="L").convert("RGB")
        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(png_path))
        return png_path.is_file()
    except Exception:  # noqa: BLE001
        logger.exception("fits_first_image_to_png failed for %s", fits_path)
        return False


def ensure_clean_field_background_png(
    platesolve_dir: Path,
    photometry_dir: Path,
    *,
    cache_dir: Path | None = None,
) -> tuple[Path | None, bool]:
    """
    Resolve a clean field snapshot for HRD annotations.

    Returns ``(path, is_fits_derived)``. ``is_fits_derived`` is True only for PNGs
    rendered 1:1 from a science FITS via ``fits_first_image_to_png``.
    """
    hit = resolve_clean_field_image_path(platesolve_dir, photometry_dir, allow_field_map=False)
    if hit is not None:
        return hit, False

    pt = Path(photometry_dir)
    cache = Path(cache_dir or (pt / "_hrd_cache"))
    cache.mkdir(parents=True, exist_ok=True)
    out_png = cache / "hrd_field_from_fits.png"

    ms_fits = _resolve_masterstar_fits_path(Path(platesolve_dir))
    if ms_fits is not None and fits_first_image_to_png(ms_fits, out_png):
        return out_png, True

    try:
        draft_dir = _draft_dir_from_photometry(pt)
        obs_group = _obs_group_from_photometry(pt)
        from pipeline import find_qc_metrics_csv

        qc_csv = find_qc_metrics_csv(draft_dir, app_config=None)
        if qc_csv is None:
            logger.info(
                "HRD: no clean PNG and no qc_metrics.csv under %s (calibrated/ or processed/)",
                draft_dir,
            )
            return None, False
        dfq = pd.read_csv(qc_csv, low_memory=False)
        if dfq.empty or "dst" not in dfq.columns:
            return None, False
        m = dfq["dst"].astype(str).str.contains(str(obs_group), regex=False)
        dfq = dfq.loc[m].copy()
        if dfq.empty:
            return None, False
        dfq["FWHM_PX"] = pd.to_numeric(dfq.get("fwhm_px"), errors="coerce")
        dfq["ELONGATION"] = pd.to_numeric(dfq.get("elongation"), errors="coerce")
        if "n_stars_detected" in dfq.columns:
            dfq["STAR_COUNT"] = pd.to_numeric(dfq.get("n_stars_detected"), errors="coerce")
        else:
            dfq["STAR_COUNT"] = pd.to_numeric(dfq.get("n_sources"), errors="coerce")
        dfq["SKY_LEVEL"] = pd.to_numeric(dfq.get("bg_median"), errors="coerce")
        dfq["_fits_path"] = dfq["dst"].map(lambda s: Path(str(s).strip()))
        dfq["_score"] = _masterstar_frame_score(dfq)
        best = dfq.sort_values("_score", ascending=False).head(3)
        for _, row in best.iterrows():
            fp = row["_fits_path"]
            if isinstance(fp, Path) and fp.is_file() and fits_first_image_to_png(fp, out_png):
                return out_png, True
    except Exception:  # noqa: BLE001
        logger.exception("ensure_clean_field_background_png failed")

    return None, False


def _resolve_masterstar_naxis(platesolve_dir: Path | None) -> tuple[int, int] | None:
    """Read MASTERSTAR FITS NAXIS1/NAXIS2 from platesolve directory."""
    fits_path = _resolve_masterstar_fits_path(platesolve_dir)
    if fits_path is None:
        return None
    try:
        from astropy.io import fits

        with fits.open(str(fits_path), memmap=False) as hdul:
            hdr = hdul[0].header
            n1 = int(hdr.get("NAXIS1") or 0)
            n2 = int(hdr.get("NAXIS2") or 0)
        if n1 > 0 and n2 > 0:
            return n1, n2
    except Exception:  # noqa: BLE001
        logger.debug("Could not read NAXIS from %s", fits_path, exc_info=True)
    return None


def _field_png_is_fits_derived(png_path: Path) -> bool:
    return Path(png_path).name == "hrd_field_from_fits.png"


def field_annotation_pixel_scale(
    platesolve_dir: Path | None,
    png_w: int,
    png_h: int,
    *,
    png_from_fits: bool,
) -> tuple[float, float, bool]:
    """Return (scale_x, scale_y, ok_to_draw). Never draw when ok_to_draw is False."""
    naxis = _resolve_masterstar_naxis(platesolve_dir)
    if naxis is not None:
        nx, ny = naxis
        if nx > 0 and ny > 0:
            return png_w / float(nx), png_h / float(ny), True
    if png_from_fits:
        return 1.0, 1.0, True
    log_event(
        "HRD annotation skipped: background PNG size unknown vs MASTERSTAR FITS "
        "(no MASTERSTAR*.fits and PNG not FITS-derived)"
    )
    return 1.0, 1.0, False


def _category_short_label(category: str, *, nss_enabled: bool) -> str:
    base = _category_base(category)
    short_map = {
        _LABEL_WD: "WD",
        _LABEL_RSG: "RSG",
        _LABEL_RG: "RG",
        _LABEL_HOT_LUM: "HOT-LUM",
        _LABEL_VERY_HOT: "HOT",
        _LABEL_VERY_COOL: "COOL",
        _LABEL_BINARY: "NSS",
    }
    if base == _LABEL_BINARY and not nss_enabled:
        return "?"
    return short_map.get(base, base[:8] if base else "?")


def _annotation_font(size: int):
    from PIL import ImageFont

    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate_field_image(
    field_image_path: str | Path,
    top_stars: pd.DataFrame,
    hrd_df: pd.DataFrame,
    *,
    platesolve_dir: Path | None = None,
    output_path: Path | None = None,
    nss_category_enabled: bool = False,
    png_from_fits: bool = False,
) -> Path | None:
    """Mark interesting stars on a field PNG/JPEG. Returns None when alignment is unknown."""
    from PIL import Image, ImageDraw

    field_image_path = Path(field_image_path)
    if not field_image_path.is_file():
        return None
    if not png_from_fits:
        png_from_fits = _field_png_is_fits_derived(field_image_path)

    img = Image.open(str(field_image_path)).convert("RGB")
    draw = ImageDraw.Draw(img)

    category_colors: dict[str, tuple[int, int, int]] = {
        _LABEL_VERY_HOT: (80, 80, 255),
        _LABEL_HOT_LUM: (0, 200, 255),
        _LABEL_VERY_COOL: (255, 80, 80),
        _LABEL_WD: (220, 220, 255),
        _LABEL_RG: (255, 140, 0),
        _LABEL_RSG: (255, 100, 0),
        _LABEL_BINARY: (0, 255, 100),
    }

    w, h = img.size
    sx, sy, ok = field_annotation_pixel_scale(
        platesolve_dir, w, h, png_from_fits=png_from_fits
    )
    if not ok:
        return None

    radius = max(12, int(round(w / 150)))
    font_size = max(10, int(round(w / 120)))
    font = _annotation_font(font_size)
    line_gap = max(2, font_size // 4)

    out_path = Path(output_path) if output_path is not None else field_image_path.parent / "hrd_field_annotated.png"
    if top_stars is None or top_stars.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        return out_path

    used_categories: set[str] = set()

    for _, star in top_stars.iterrows():
        cid = normalize_gaia_source_id(star.get("catalog_id", ""))
        match = hrd_df[hrd_df["catalog_id"] == cid]
        if match.empty:
            continue
        r = match.iloc[0]
        x_raw = _f(r.get("x"))
        y_raw = _f(r.get("y"))
        if x_raw is None or y_raw is None:
            continue
        x = x_raw * sx
        y = y_raw * sy
        if not (radius < x < w - radius and radius < y < h - radius):
            continue
        cat_raw = str(star.get("category", "")).strip()
        base = _display_to_base_label(cat_raw)
        color = category_colors.get(base)
        if color is None:
            for key, col in category_colors.items():
                if cat_raw.startswith(key.split(" (")[0]):
                    color = col
                    break
        if color is None:
            color = (255, 255, 0)
        if base:
            used_categories.add(base)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=color,
            width=max(2, radius // 9),
        )
        label = _category_short_label(cat_raw, nss_enabled=nss_category_enabled)
        simbad_id = str(r.get("simbad_main_id") or star.get("simbad_id") or "").strip()
        if simbad_id and "(" in simbad_id:
            simbad_id = simbad_id.split("(")[0].strip()
        lines = [label]
        if simbad_id:
            lines.append(simbad_id[:24])
        ty = y - radius - 2
        for line in lines:
            draw.text((x + radius + 3, ty), line, fill=color, font=font)
            ty += font_size + line_gap

    if used_categories:
        legend_x, legend_y = 8, 8
        pad = 4
        entries = sorted(
            used_categories,
            key=lambda lab: _LABEL_PRIORITY.index(lab) if lab in _LABEL_PRIORITY else 99,
        )
        if nss_category_enabled and _LABEL_BINARY not in entries:
            pass
        lh = font_size + line_gap
        box_h = pad * 2 + lh * len(entries)
        box_w = max(80, int(w * 0.12))
        draw.rectangle(
            [legend_x, legend_y, legend_x + box_w, legend_y + box_h],
            fill=(24, 24, 24),
            outline=(180, 180, 180),
        )
        for i, lab in enumerate(entries):
            col = category_colors.get(lab, (255, 255, 0))
            yy = legend_y + pad + i * lh
            dot_r = max(4, font_size // 3)
            draw.ellipse(
                [legend_x + pad, yy, legend_x + pad + dot_r * 2, yy + dot_r * 2],
                fill=col,
                outline=col,
            )
            draw.text(
                (legend_x + pad + dot_r * 2 + 4, yy - 1),
                _category_short_label(lab, nss_enabled=nss_category_enabled),
                fill=(240, 240, 240),
                font=font,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    return out_path
