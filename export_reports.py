from __future__ import annotations

import json
import math
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from astropy.time import Time

from config import AppConfig
from citations import (
    build_run_citation_context,
    emit_export_citation_lines,
    emit_varastro_method_summary_lines,
    load_pipeline_meta,
)
from gaia_catalog_id import normalize_gaia_source_id
from report_methods import (
    aavso_export_path,
    active_report_methods,
    lc_csv_path,
    software_method_label,
    varastro_export_path,
)
from check_star_kmag import (
    kmag_values_for_export,
    resolve_proc_csv_dir,
    select_check_star,
)
from photometry_core import parse_comp_quality_json_map, _resolve_plate_scale_arcsec_per_px

# Gaia ID musí byť str — float64 stráca cifry
_GAIA_ID_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}

# Single source for export headers (AAVSO #SOFTWARE + VarAstro Software line).
VYVAR_SOFTWARE_VERSION = "VYVAR 1.0"


def _aavso_software_header_line(software_version: str, export_method: str) -> str:
    """Format AAVSO ``#SOFTWARE`` from the canonical version string."""
    ver = str(software_version or VYVAR_SOFTWARE_VERSION).strip() or VYVAR_SOFTWARE_VERSION
    soft_id = ver.replace(" ", "/") if "/" not in ver else ver
    return (
        f"#SOFTWARE={soft_id} ({software_method_label(export_method)} photometry; "
        "Broeg 2005 ensemble)\n"
    )


def _resolve_export_arcsec_per_px(
    photometry_dir: Path,
    cfg: AppConfig,
) -> float | None:
    """Derive plate scale (arcsec/px) for export headers — derive-or-None (no magic 1.3).

    Priority: ``pipeline_meta.json`` ``plate_scale_arcsec_px`` (Phase 2A session);
    then WCS/CD from sibling ``MASTERSTAR.fits`` via ``_resolve_plate_scale_arcsec_per_px``.
    """
    meta = load_pipeline_meta(photometry_dir)
    ps = meta.get("plate_scale_arcsec_px")
    if ps is not None:
        try:
            v = float(ps)
            if math.isfinite(v) and v > 0:
                return float(v)
        except (TypeError, ValueError):
            pass
    masterstar = photometry_dir.parent / "MASTERSTAR.fits"
    if masterstar.is_file():
        try:
            v = _resolve_plate_scale_arcsec_per_px(cfg, masterstar)
            if v is not None and math.isfinite(float(v)) and float(v) > 0:
                return float(v)
        except Exception:  # noqa: BLE001
            pass
    return None


def _resolved_site_from_meta(output_dir: Path | str | None) -> dict[str, Any] | None:
    """Read the per-draft resolved observer site from ``pipeline_meta.json``.

    Phase 2A persists ``observer_location`` (param_resolver: draft ID_LOCATION ->
    header SITELAT -> flagged config) with lat/lon/alt/source. Exported coordinates
    must match the site used for BJD/airmass, so we prefer this over ``cfg`` (which
    tracks the *last* session and can drift when reprocessing an old draft).
    """
    if output_dir is None:
        return None
    try:
        pm = Path(output_dir) / "pipeline_meta.json"
        if not pm.is_file():
            return None
        meta = json.loads(pm.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    loc = meta.get("observer_location") if isinstance(meta, dict) else None
    if not isinstance(loc, dict):
        return None
    try:
        lat = float(loc.get("lat"))
        lon = float(loc.get("lon"))
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)) or (lat == 0.0 and lon == 0.0):
        return None
    try:
        alt = float(loc.get("alt_m", 0.0) or 0.0)
    except (TypeError, ValueError):
        alt = 0.0
    return {
        "lat": lat,
        "lon": lon,
        "alt_m": alt,
        "name": str(loc.get("name", "") or "").strip(),
        "source": str(loc.get("source", "resolved") or "resolved"),
    }


def _site_coords(
    cfg: AppConfig | None, site: dict[str, Any] | None
) -> tuple[float, float, float, str] | None:
    """(lat, lon, alt_m, name) from the resolved per-draft site, else config."""
    if site is not None:
        return (
            float(site["lat"]),
            float(site["lon"]),
            float(site.get("alt_m", 0.0) or 0.0),
            str(site.get("name", "") or "").strip(),
        )
    if cfg is None:
        return None
    try:
        lat = float(getattr(cfg, "observer_lat", 0.0) or 0.0)
        lon = float(getattr(cfg, "observer_lon", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)) or (lat == 0.0 and lon == 0.0):
        return None
    alt = float(getattr(cfg, "observer_alt_m", 0.0) or 0.0)
    name = str(getattr(cfg, "observer_location_name", "") or "").strip()
    return lat, lon, alt, name


def _observer_location_configured(cfg: AppConfig | None) -> bool:
    if cfg is None:
        return False
    try:
        lat = float(getattr(cfg, "observer_lat", 0.0) or 0.0)
        lon = float(getattr(cfg, "observer_lon", 0.0) or 0.0)
        return math.isfinite(lat) and math.isfinite(lon) and (lat != 0.0 or lon != 0.0)
    except (TypeError, ValueError):
        return False


def _append_aavso_observer_location_lines(
    lines: list[str], cfg: AppConfig | None, site: dict[str, Any] | None = None
) -> None:
    coords = _site_coords(cfg, site)
    if coords is None:
        return
    lat, lon, alt, _name = coords
    lines.append(f"#LATITUDE={lat:.4f}\n")
    lines.append(f"#LONGITUDE={lon:.4f}\n")
    lines.append(f"#ELEVATION={alt:.0f}\n")


def _append_varastro_site_line(
    lines: list[str], cfg: AppConfig | None, site: dict[str, Any] | None = None
) -> None:
    coords = _site_coords(cfg, site)
    if coords is None:
        return
    lat, lon, alt, name = coords
    if not name:
        name = "Observer site"
    lines.append(f"# Site: {name} ({lat:.4f}°N, {lon:.4f}°E, {alt:.0f} m)\n")


def _vyvar_export_citation_lines(
    cfg: AppConfig | None = None,
    *,
    run_ctx: Any = None,
    photometry_dir: Path | None = None,
    target_row: pd.Series | None = None,
    targets_df: pd.DataFrame | None = None,
    lc_method: str | None = None,
) -> list[str]:
    """Comment header block for AAVSO / VAR.ASTRO text exports (CITATIONS.bib)."""
    if run_ctx is None:
        meta = load_pipeline_meta(photometry_dir)
        targets = targets_df
        if targets is None and target_row is not None:
            targets = pd.DataFrame([target_row])
        run_ctx = build_run_citation_context(
            cfg,
            pipeline_meta=meta,
            targets_df=targets,
            lc_method=lc_method,
        )
    return emit_export_citation_lines(run_ctx)


def _varastro_alg_lines(
    fresh_cfg: AppConfig | None,
    *,
    run_ctx: Any = None,
    photometry_dir: Path | None = None,
    lc_method: str | None = None,
) -> list[str]:
    """VAR.ASTRO per-run algorithm summary (CITATIONS.bib)."""
    if run_ctx is None:
        run_ctx = build_run_citation_context(
            fresh_cfg,
            pipeline_meta=load_pipeline_meta(photometry_dir),
            lc_method=lc_method,
        )
    return emit_varastro_method_summary_lines(run_ctx)


def _safe_filename(name: str) -> str:
    """Sanitizuj VSX meno pre filesystem. 'BO CVn' → 'BO_CVn'."""
    s = str(name).strip()
    s = re.sub(r"[^\w\s\-\+]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s or "unknown"


def _bjd_to_datestr_yyyymmdd(bjd_tdb: float) -> str:
    """BJD(TDB) (JD-like float) -> calendar date string YYYYMMDD (UTC-like display)."""
    try:
        t = Time(float(bjd_tdb), format="jd", scale="tdb")
        # For file naming we want just a stable date; use UTC conversion for readability.
        dt = t.utc.datetime
        return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"
    except Exception:  # noqa: BLE001
        return "unknown"


def _fmt_opt_num(v: Any, fmt: str, *, na: str = "na") -> str:
    try:
        f = float(pd.to_numeric(v, errors="coerce"))
    except Exception:  # noqa: BLE001
        return na
    return (format(f, fmt) if math.isfinite(f) else na)


def _aavso_gs11_notes_suffix(summary_row: pd.Series, cfg: AppConfig | None) -> str:
    """Append GS11 dilution tag to AAVSO NOTES when blend is significant."""
    if cfg is None or not bool(getattr(cfg, "gs11_dilution_enabled", False)):
        return ""
    try:
        d = float(pd.to_numeric(summary_row.get("dilution_factor", 1.0), errors="coerce"))
    except (TypeError, ValueError):
        return ""
    if math.isfinite(d) and d < 0.99:
        return f"|GS11:D={d:.3f}"
    return ""


def _fmt_opt_int(v: Any, *, na: str = "na") -> str:
    try:
        f = float(pd.to_numeric(v, errors="coerce"))
    except Exception:  # noqa: BLE001
        return na
    if not math.isfinite(f):
        return na
    try:
        return str(int(f))
    except Exception:  # noqa: BLE001
        return na


# Default AAVSO observer placeholder when config is unset (must trigger export/validator warning).
_AAVSO_OBSCODE_PLACEHOLDER = "UMIA"

# Built-in filter/setup name → AAVSO Extended FILT code (uppercase keys).
# User overrides via ``AppConfig.aavso_filter_map`` (merged on lookup).
_AAVSO_FILTER_BUILTIN: dict[str, str] = {
    "U": "U",
    "B": "B",
    "V": "V",
    "R": "R",
    "I": "I",
    "RJ": "RJ",
    "IJ": "IJ",
    "RC": "Rc",
    "IC": "Ic",
    "SU": "SU",
    "SG": "SG",
    "SR": "SR",
    "SI": "SI",
    "SZ": "SZ",
    "CV": "CV",
    "CR": "CR",
    "TB": "TB",
    "TG": "TG",
    "TR": "TR",
    "J": "J",
    "H": "H",
    "K": "K",
    "Y": "Y",
    "L": "CV",
    "LUM": "CV",
    "LUMINANCE": "CV",
    "NOFILTER": "CV",
    "NO_FILTER": "CV",
    "CLEAR": "CV",
}


def _filter_lookup_key(name: str) -> str:
    s = str(name or "").strip().upper()
    return re.sub(r"[\s\-]+", "_", s)


def _resolve_aavso_filter(
    filter_name: str,
    cfg: AppConfig | None = None,
) -> tuple[str, str | None]:
    """Map setup/filter name to AAVSO FILT code.

    Returns ``(code, warning_message)``. Unrecognized filters emit ``UNKN`` and a
    non-empty warning — never silently default to CV.
    """
    raw = str(filter_name or "").strip()
    if not raw:
        return "UNKN", "FILT empty — review before AAVSO submit"

    key = _filter_lookup_key(raw)
    merged: dict[str, str] = dict(_AAVSO_FILTER_BUILTIN)
    if cfg is not None:
        umap = getattr(cfg, "aavso_filter_map", None) or {}
        for uk, uv in umap.items():
            nk = _filter_lookup_key(str(uk))
            if nk and str(uv).strip():
                merged[nk] = str(uv).strip()

    if key in merged:
        return merged[key], None

    key_compact = key.replace("_", "")
    if "NOFILTER" in key_compact or key_compact == "CLEAR" or key.startswith("LUM"):
        return "CV", None

    return (
        "UNKN",
        f"FILT unrecognized: '{raw}' — map via aavso_filter_map or review before AAVSO submit",
    )


def _guess_setup_info_from_obs_group(obs_group: str) -> tuple[str, str | None, str | None]:
    """Best-effort: parse 'NoFilter_60_2' -> (filter, exptime, binning)."""
    parts = [p for p in str(obs_group or "").strip().split("_") if p]
    if not parts:
        return "", None, None
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return "_".join(parts[:-2]), parts[-2], parts[-1]
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), parts[-1], None
    return str(obs_group or ""), None, None


# Zakrytové typy podľa VSX dokumentácie
_ECLIPSING_TOKENS = frozenset(
    {
        "E",  # Generic eclipsing
        "EA",  # Algol (Beta Persei)
        "EB",  # Beta Lyrae
        "EW",  # W Ursae Majoris
        "EP",  # Planetary transits
        "E-DO",  # Disk occultation
        "ELL",  # Ellipsoidal (no eclipse but VAR.ASTRO má záujem)
        "EC",  # Contact binaries — ASAS survey typ
        "ED",  # Detached eclipsing — ASAS survey typ
        "ESD",  # Semi-detached eclipsing — ASAS survey typ
    }
)


def _token_is_eclipsing(token: str) -> bool:
    """Skontroluj či jeden VSX token (bez špeciálnych znakov) je zakrytový."""
    t = str(token or "").strip().rstrip(":").upper()
    if not t:
        return False
    if t in _ECLIPSING_TOKENS:
        return True
    if t.startswith("E-"):
        return True
    return False


def _is_eclipsing(vsx_type: str) -> bool:
    """Vráti True ak VSX typ obsahuje zakrytovú komponentu (VSX konvencie)."""
    if not vsx_type or not isinstance(vsx_type, str):
        return False

    vsx_type = vsx_type.strip()
    if not vsx_type:
        return False

    # Pipe | → OR/neistota: len ak VŠETKY alternatívy sú zakrytové
    if "|" in vsx_type:
        alternatives = [a.strip() for a in vsx_type.split("|") if a.strip()]
        if not alternatives:
            return False
        for alt in alternatives:
            main = alt.split("/")[0].split("+")[0].strip().rstrip(":")
            if not _token_is_eclipsing(main):
                return False
        return True

    # Plus + → AND: ak KTORÝKOĽVEK komponent je zakrytový → True
    plus_parts = vsx_type.split("+")
    for part in plus_parts:
        part = part.strip()
        if not part:
            continue
        main_token = part.split("/")[0].strip()
        if _token_is_eclipsing(main_token):
            return True

    return False


def _test_is_eclipsing() -> None:
    cases = [
        ("EW", True, "W UMa — základný typ"),
        ("EA", True, "Algol — základný typ"),
        ("EB", True, "Beta Lyrae — základný typ"),
        ("E", True, "Generic eclipsing"),
        ("EP", True, "Planetary transit"),
        ("ELL", True, "Ellipsoidal"),
        ("EC", True, "ASAS contact binary"),
        ("ED", True, "ASAS detached"),
        ("ESD", True, "ASAS semi-detached"),
        ("E:", True, "Uncertain eclipsing — colon"),
        ("EW:", True, "Uncertain EW — colon"),
        ("EA/SD", True, "EA s subTypom SD"),
        ("EA/DM", True, "EA s subTypom DM"),
        ("EA/RS", True, "EA s RS aktivitou"),
        ("EW+DSCT", True, "EW + delta Scuti pulsácia"),
        ("EA+EA", True, "Dvojitý EA systém"),
        ("EB+ROT", True, "EB + rotačná variabilita"),
        ("EA|EB", True, "Neistota medzi EA a EB — oba zakrytové"),
        ("ELL|DSCT", False, "Neistota ELL alebo DSCT — DSCT nie je zakrytová"),
        ("EA|RRAB", False, "Neistota EA alebo RRAB — RRAB nie zakrytová"),
        ("RRAB", False, "RR Lyrae — pulsujúca"),
        ("ROT", False, "Rotujúca"),
        ("SR", False, "Semi-regular"),
        ("DSCT|GDOR|SXPHE", False, "Pulsujúce typy"),
        ("TTS/ROT", False, "T Tauri s rotáciou — nie zakrytová"),
        ("DPV/ELL", False, "DPV hlavný typ — nie zakrytový"),
        ("L", False, "Slow irregular"),
        ("M", False, "Mira"),
        ("", False, "Prázdny typ"),
        ("VAR", False, "Unspecified variable"),
        ("RRAB/BL", False, "RR Lyrae Blazhko — nie zakrytová"),
        ("E-DO", True, "Disk occultation"),
        ("LB:", False, "Uncertain slow irregular"),
    ]

    errors = []
    for vsx_type, expected, desc in cases:
        result = _is_eclipsing(vsx_type)
        status = "OK" if result == expected else "FAIL"
        if result != expected:
            errors.append(f"  FAIL: '{vsx_type}' -> {result} (expected {expected}) | {desc}")
        msg = f"  [{status}] '{vsx_type}' -> {result} | {desc}"
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "backslashreplace").decode("ascii"))

    if errors:
        try:
            print(f"\n{len(errors)} FAILED:")
        except UnicodeEncodeError:
            print(f"\n{len(errors)} FAILED:".encode("ascii", "backslashreplace").decode("ascii"))
        for e in errors:
            try:
                print(e)
            except UnicodeEncodeError:
                print(e.encode("ascii", "backslashreplace").decode("ascii"))
    else:
        print(f"\nAll {len(cases)} tests passed!")


def _select_check_star(
    comp_df: pd.DataFrame,
    *,
    ensemble_ids: set[str] | None = None,
    n_comp_min: int = 3,
    cfg: AppConfig | None = None,
) -> pd.Series | None:
    return select_check_star(
        comp_df,
        ensemble_ids=ensemble_ids or set(),
        n_comp_min=n_comp_min,
        cfg=cfg,
    )


def _copy_field_image(
    lc_dir: Path,
    catalog_id: str,
    dst_dir: Path,
    fname: str,
    obs_date: str,
) -> str | None:
    """Skopíruj field map PNG do varastro adresára."""
    candidates = [
        lc_dir / f"field_map_{catalog_id}.png",
        lc_dir / "field_map.png",
        lc_dir.parent / "field_map.png",
    ]
    for src in candidates:
        try:
            if src.exists():
                dst_name = f"{fname}_{obs_date}_field.png"
                dst = Path(dst_dir) / dst_name
                try:
                    shutil.copy2(src, dst)
                    return dst_name
                except Exception as exc:  # noqa: BLE001
                    logging.warning("[EXPORT] Field image copy failed: %s", exc)
                    return None
        except Exception:  # noqa: BLE001
            continue
    return None


def _comp_quality_map_for_export(raw: dict[str, Any]) -> dict[str, str]:
    """``catalog_id`` → ``good`` / ``suspect`` (excluded comps omitted)."""
    out: dict[str, str] = {}
    for k, v in parse_comp_quality_json_map(raw).items():
        nk = str(normalize_gaia_source_id(k) or "").strip()
        if not nk:
            continue
        q = str(v.get("quality", "")).strip().lower()
        if q == "excluded":
            continue
        if q in ("good", "suspect"):
            out[nk] = q
    return out


def _normalize_comp_df_export_columns(
    comp_df: pd.DataFrame,
    comp_quality_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Map Phase-1 CSV names (``comp_rms``, ``comp_weight``) to export columns (``p2p_rms``, ``w_rel``)."""
    df = comp_df.copy()
    if df.empty:
        return df
    if "p2p_rms" not in df.columns and "comp_rms" in df.columns:
        df["p2p_rms"] = df["comp_rms"]
    if "w_rel" not in df.columns and "comp_weight" in df.columns:
        from photometry_core import apply_comp_w_rel_for_display

        df = apply_comp_w_rel_for_display(df, comp_quality_map)
    return df


def _export_comp_status_label(row: pd.Series, quality_map: dict[str, str] | None) -> str:
    """COMP TABLE status: good / suspect / rejected; fallback ``comp_quality`` map alebo ``unknown``."""
    for col in ("status", "stav", "quality"):
        if col not in row.index:
            continue
        v = row.get(col)
        try:
            if v is None or (isinstance(v, float) and not math.isfinite(float(v))):
                continue
        except Exception:  # noqa: BLE001
            continue
        s = str(v).strip().lower()
        if not s or s in ("?", "nan", "none"):
            continue
        if s == "excluded":
            return "rejected"
        if s in ("good", "suspect", "rejected"):
            return s
    if quality_map:
        k = str(normalize_gaia_source_id(row.get("catalog_id")) or "").strip()
        q = str(quality_map.get(k, "") or "").strip().lower()
        if q == "excluded":
            return "rejected"
        if q in ("good", "suspect"):
            return q
    return "unknown"


def _format_varastro_comp_table(
    comp_df: pd.DataFrame,
    *,
    comp_quality_map: dict[str, str] | None = None,
) -> str:
    header = (
        "# Nr  CatalogId             Mag    BP-RP  dBPRP  tier_color  "
        "p2p_RMS  w_rel  tier status\n"
    )
    if comp_df is None or comp_df.empty:
        return header
    lines = [header]
    df = _normalize_comp_df_export_columns(comp_df, comp_quality_map)
    export_row_n = 0
    for _, row in df.iterrows():
        cid_key = str(normalize_gaia_source_id(row.get("catalog_id")) or "").strip()
        if comp_quality_map and cid_key:
            q_skip = str(comp_quality_map.get(cid_key, "") or "").strip().lower()
            if q_skip == "excluded":
                continue
        export_row_n += 1
        cid = str(row.get("catalog_id", "") or "")[:19].ljust(19)
        mag = pd.to_numeric(row.get("mag", float("nan")), errors="coerce")
        bprp = pd.to_numeric(row.get("bp_rp", float("nan")), errors="coerce")
        dbprp = pd.to_numeric(row.get("delta_bprp_abs", float("nan")), errors="coerce")
        p2p = pd.to_numeric(row.get("p2p_rms", float("nan")), errors="coerce")
        w_rel = pd.to_numeric(row.get("w_rel", float("nan")), errors="coerce")
        tier = pd.to_numeric(row.get("tier", row.get("comp_tier", 4)), errors="coerce")
        st = _export_comp_status_label(row, comp_quality_map)

        mag_s = f"{float(mag):.3f}" if math.isfinite(float(mag)) else "  —  "
        bprp_s = f"{float(bprp):.3f}" if math.isfinite(float(bprp)) else "  —  "
        dbprp_s = f"{float(dbprp):.3f}" if math.isfinite(float(dbprp)) else "  —  "
        p2p_s = f"{float(p2p):.4f}" if math.isfinite(float(p2p)) else "  —  "
        wrel_s = f"{float(w_rel):.3f}" if math.isfinite(float(w_rel)) else "  —  "
        try:
            tier_i = int(tier) if math.isfinite(float(tier)) else 4
        except Exception:  # noqa: BLE001
            tier_i = 4
        st_s = str(st)[:10].ljust(10)

        tier_cs = str(row.get("color_tier_src", "") or "")[:12].ljust(12)
        lines.append(
            f"# C{export_row_n:02d} {cid} {mag_s}  {bprp_s}  {dbprp_s}  {tier_cs} {p2p_s}  "
            f"{wrel_s}  {tier_i}  {st_s}\n"
        )
    return "".join(lines)


def _select_export_lc_rows(lc_df: pd.DataFrame) -> pd.DataFrame:
    """Rows suitable for AAVSO/VAR.ASTRO export (canonical ``mag_calib_final`` when present)."""
    if lc_df is None or lc_df.empty:
        return pd.DataFrame()
    work = lc_df.copy()
    mag_col = "mag_calib"
    if "mag_calib_final" in work.columns:
        mag_col = "mag_calib_final"
    elif "mag_calib_ac" in work.columns:
        ac_ok = (
            work["ac_ok"].astype(bool)
            if "ac_ok" in work.columns
            else pd.Series(False, index=work.index)
        )
        mac = pd.to_numeric(work["mag_calib_ac"], errors="coerce")
        use_ac = ac_ok & mac.notna() & np.isfinite(mac.to_numpy(dtype=float))
        if use_ac.any():
            work = work.copy()
            work["_export_mag"] = pd.to_numeric(work["mag_calib"], errors="coerce")
            work.loc[use_ac, "_export_mag"] = mac.loc[use_ac]
            mag_col = "_export_mag"
    mag = pd.to_numeric(work.get(mag_col), errors="coerce")
    bjd = pd.to_numeric(work.get("bjd"), errors="coerce")
    finite = mag.notna() & np.isfinite(mag.to_numpy(dtype=float)) & bjd.notna() & np.isfinite(
        bjd.to_numpy(dtype=float)
    )
    if "flag" in work.columns:
        fl = work["flag"].astype(str).str.strip().str.lower()
        good = fl.isin(("normal", "")) | fl.isna()
        bad = fl.isin(("no_data", "saturated", "edge_fail", "nondetection"))
        mask = finite & (good | ~bad)
        out = work.loc[mask].copy()
        if out.empty and finite.any():
            out = work.loc[finite].copy()
    else:
        out = work.loc[finite].copy()
    if mag_col == "_export_mag" and "mag_calib" in out.columns:
        out = out.copy()
        out["mag_calib"] = out["_export_mag"]
    elif mag_col == "mag_calib_final" and "mag_calib" in out.columns:
        out = out.copy()
        out["mag_calib"] = pd.to_numeric(out["mag_calib_final"], errors="coerce")
    return out


def export_lightcurve_reports(
    output_dir: Path,
    target_row: pd.Series,
    lc_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    summary_row: pd.Series,
    *,
    observer_code: str = "",
    observer_name: str = "Unknown Observer",
    comp_quality_map: dict[str, str] | None = None,
    arcsec_per_px: float | None = None,
    software_version: str | None = None,
    cfg: AppConfig | None = None,
    lc_dir: Path | None = None,
    obs_group: str = "",
    targets_df: pd.DataFrame | None = None,
    run_citation_ctx: Any = None,
    export_method: str = "aperture",
    active_methods: list[str] | None = None,
    proc_csv_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    """Generuje AAVSO a VAR.ASTRO súbory pre jeden target."""
    fresh_cfg = cfg or AppConfig()
    if not str(observer_code).strip() and fresh_cfg is not None:
        observer_code = str(getattr(fresh_cfg, "observer_code", "") or "")
    if observer_name == "Unknown Observer" and fresh_cfg is not None:
        _on = str(getattr(fresh_cfg, "observer_name", "") or "").strip()
        if _on:
            observer_name = _on

    out_base = Path(output_dir)
    (out_base / "aavso").mkdir(parents=True, exist_ok=True)
    (out_base / "varastro").mkdir(parents=True, exist_ok=True)

    _phot_dir = out_base.parent
    _sw_ver = str(software_version or VYVAR_SOFTWARE_VERSION).strip() or VYVAR_SOFTWARE_VERSION
    arcsec_per_px = _resolve_export_arcsec_per_px(_phot_dir, fresh_cfg)

    # Per-draft resolved observer site (param_resolver, persisted by Phase 2A) so the
    # exported #LATITUDE / VAR.ASTRO site matches the location used for BJD/airmass.
    _resolved_site = _resolved_site_from_meta(out_base)

    vsx_name = str(target_row.get("vsx_name", "") or "").strip() or "unknown"
    safe = _safe_filename(vsx_name)

    # Use exportable LC points (normal frames; canonical mag_calib_final when present).
    lc0 = lc_df.copy() if lc_df is not None else pd.DataFrame()
    lc_normal = _select_export_lc_rows(lc0)

    if lc_normal.empty:
        logging.info(
            "[EXPORT] Skip %s — no exportable LC points (flags/mag empty)",
            str(vsx_name),
        )
        return {}

    bjd_first = pd.to_numeric(lc_normal.iloc[0].get("bjd", float("nan")), errors="coerce")
    date_tag = _bjd_to_datestr_yyyymmdd(float(bjd_first)) if math.isfinite(float(bjd_first)) else "unknown"

    _export_method = str(export_method or "aperture").strip().lower()
    _active_methods = list(active_methods or active_report_methods(fresh_cfg))
    aavso_path = aavso_export_path(out_base, safe, date_tag, _export_method, active_methods=_active_methods)
    var_path = varastro_export_path(out_base, safe, date_tag, _export_method, active_methods=_active_methods)

    # Setup/filter info (best effort).
    obs_group_resolved = str(obs_group or "").strip()
    if not obs_group_resolved:
        obs_group_resolved = str(
            summary_row.get("obs_group", "") or summary_row.get("setup", "") or ""
        ).strip()
    if not obs_group_resolved:
        obs_group_resolved = str(summary_row.get("obs_group_name", "") or "").strip()
    if not obs_group_resolved:
        try:
            setup_dir = out_base.parent.parent
            if setup_dir.name and setup_dir.name.lower() not in ("platesolve", "photometry"):
                obs_group_resolved = setup_dir.name
        except Exception:  # noqa: BLE001
            pass
    setup_filter_raw, exptime_s, binning = _guess_setup_info_from_obs_group(obs_group_resolved)
    aavso_filter, filt_warn = _resolve_aavso_filter(setup_filter_raw, fresh_cfg)

    # Target catalog info.
    target_cid = str(target_row.get("catalog_id", "") or "").strip()
    t_mag = pd.to_numeric(target_row.get("mag", float("nan")), errors="coerce")
    t_bprp = pd.to_numeric(target_row.get("bp_rp", float("nan")), errors="coerce")

    # Check star (exclude Phase-2A ensemble members when comp_quality sidecar exists).
    from check_star_kmag import resolve_ensemble_ids_for_check  # noqa: PLC0415

    _lc_dir_pre = Path(lc_dir) if lc_dir is not None else (out_base.parent / "lightcurves")
    _ens_ids = resolve_ensemble_ids_for_check(
        target_cid,
        comp_df,
        lc_dir=_lc_dir_pre,
        comp_quality_map=comp_quality_map,
        cfg=fresh_cfg,
    )
    check_row = _select_check_star(
        comp_df,
        ensemble_ids=_ens_ids,
        n_comp_min=3,
        cfg=fresh_cfg,
    )
    check_cid = str(check_row.get("catalog_id")) if check_row is not None else "na"
    _lc_dir = Path(lc_dir) if lc_dir is not None else (out_base.parent / "lightcurves")
    _phot_dir = out_base.parent
    _proc_dir = resolve_proc_csv_dir(_phot_dir, obs_group_resolved)
    kmag_values, _kmag_mode = kmag_values_for_export(
        check_row,
        comp_df,
        lc_normal,
        target_cid=target_cid,
        lc_dir=_lc_dir,
        proc_dir=_proc_dir,
        comp_quality_map=comp_quality_map,
        cfg=fresh_cfg,
        export_method=_export_method,
        proc_csv_cache=proc_csv_cache,
    )
    logging.debug("[EXPORT] KMAG mode=%s for %s", _kmag_mode, str(vsx_name))

    _lc_method = _export_method
    if "method" in lc_normal.columns and _export_method == "aperture":
        _mvals = lc_normal["method"].astype(str).str.strip().str.lower()
        _mvals = _mvals[_mvals != ""]
        if not _mvals.empty:
            try:
                _lc_method = str(_mvals.mode().iloc[0])
            except Exception:  # noqa: BLE001
                _lc_method = str(_mvals.iloc[0])

    # Notes: comp Gaia IDs (first line only).
    comp_ids = []
    if comp_df is not None and (not comp_df.empty) and "catalog_id" in comp_df.columns:
        for _, crow in comp_df.iterrows():
            v2 = str(crow.get("catalog_id", "") or "").strip()
            if not v2:
                continue
            ck = str(normalize_gaia_source_id(v2) or "").strip()
            if comp_quality_map and ck:
                if str(comp_quality_map.get(ck, "") or "").strip().lower() == "excluded":
                    continue
            # Safe: Gaia ID truncated to 18 chars intentionally for AAVSO notes field length limit.
            comp_ids.append(v2[:18])
    notes_first = (f"GaiaDR3:{'|'.join(comp_ids)}")[:100] if comp_ids else "na"
    _gs11_note = _aavso_gs11_notes_suffix(summary_row, fresh_cfg)
    if _gs11_note:
        notes_first = (notes_first + _gs11_note)[:100] if notes_first != "na" else _gs11_note.strip("|")[:100]
    _meth_note = f"meth={_lc_method}|"
    if notes_first != "na":
        notes_first = (_meth_note + notes_first)[:100]
    else:
        notes_first = _meth_note.strip("|")[:100]

    _trust = str(summary_row.get("trust", "") or "").strip().upper()
    _trust_reason = str(summary_row.get("trust_reason", "") or "").strip()
    if _trust:
        from trust_flag_core import format_export_trust_note  # noqa: PLC0415

        _remain = max(0, 100 - len(notes_first) - 1)
        if _remain >= 8:
            _tnote = format_export_trust_note(_trust, _trust_reason, max_len=_remain)
            if _tnote:
                notes_first = (notes_first + "|" + _tnote)[:100]

    # --- AAVSO Extended ---
    _cite_kw: dict[str, Any] = {
        "photometry_dir": _phot_dir,
        "target_row": target_row,
        "lc_method": _export_method,
    }
    if run_citation_ctx is not None:
        _cite_kw["run_ctx"] = run_citation_ctx
    elif targets_df is not None:
        _cite_kw["targets_df"] = targets_df
    if _export_method in ("psf", "adaptive"):
        _cite_kw["run_ctx"] = build_run_citation_context(
            fresh_cfg,
            pipeline_meta=load_pipeline_meta(_phot_dir),
            targets_df=targets_df,
            lc_method=_export_method,
        )
    a_lines: list[str] = []
    a_lines.extend(_vyvar_export_citation_lines(fresh_cfg, **_cite_kw))
    if filt_warn:
        a_lines.append(f"#WARNING={filt_warn}\n")
        logging.warning("[EXPORT] %s (%s)", filt_warn, str(vsx_name))
    a_lines.append("#TYPE=Extended\n")
    _obc = str(observer_code).strip()
    if not _obc:
        a_lines.append(
            "#WARNING=OBSCODE is not set — configure observer_code before AAVSO submit\n"
        )
        logging.warning(
            "[EXPORT] OBSCODE is empty for %s — set observer_code in config before submit",
            str(vsx_name),
        )
    elif _obc.upper() == _AAVSO_OBSCODE_PLACEHOLDER:
        a_lines.append(
            f"#WARNING=OBSCODE is default placeholder {_AAVSO_OBSCODE_PLACEHOLDER} "
            "— set your AAVSO observer code in config\n"
        )
        logging.warning(
            "[EXPORT] OBSCODE is default placeholder %s for %s",
            _AAVSO_OBSCODE_PLACEHOLDER,
            str(vsx_name),
        )
    a_lines.append(f"#OBSCODE={_obc}\n")
    _append_aavso_observer_location_lines(a_lines, fresh_cfg, _resolved_site)
    a_lines.append(_aavso_software_header_line(_sw_ver, _export_method))
    a_lines.append("#DELIM=,\n")
    a_lines.append("#DATE=BJD\n")
    a_lines.append("#OBSTYPE=CCD\n")
    a_lines.append("#\n")

    starid = vsx_name[:30]
    for row_pos, (_, row) in enumerate(lc_normal.iterrows()):
        bjd = pd.to_numeric(row.get("bjd", float("nan")), errors="coerce")
        mag_calib = pd.to_numeric(row.get("mag_calib", float("nan")), errors="coerce")
        err = pd.to_numeric(row.get("err", float("nan")), errors="coerce")
        am = pd.to_numeric(row.get("airmass", float("nan")), errors="coerce")
        if not (math.isfinite(float(bjd)) and math.isfinite(float(mag_calib))):
            continue

        date_s = f"{float(bjd):.6f}"
        mag_s = f"{float(mag_calib):.3f}"
        err_s = f"{float(err):.3f}" if math.isfinite(float(err)) else "na"
        am_s = f"{float(am):.3f}" if math.isfinite(float(am)) else "na"
        notes_s = notes_first if row_pos == 0 else "na"
        kmag_str = kmag_values[row_pos] if row_pos < len(kmag_values) else "na"

        # STARID,DATE,MAG,MAGERR,FILTER,TRANS,MTYPE,CNAME,CMAG,KNAME,KMAG,AMASS,GROUP,CHART,NOTES
        a_lines.append(
            ",".join(
                [
                    starid,
                    date_s,
                    mag_s,
                    err_s,
                    aavso_filter,
                    "NO",
                    "STD",
                    "ENSEMBLE",
                    "na",
                    (check_cid if check_cid else "na"),
                    kmag_str,
                    am_s,
                    "na",
                    "na",
                    notes_s,
                ]
            )
            + "\n"
        )
    aavso_path.write_text("".join(a_lines), encoding="utf-8")
    logging.info("[EXPORT] AAVSO: %s", str(aavso_path.name))

    # --- VAR.ASTRO.CZ ---
    vsx_type = str(target_row.get("vsx_type", "") or "").strip()
    if not _is_eclipsing(vsx_type):
        logging.info("[EXPORT] Skip varastro %s — nie zakrytova (%s)", str(vsx_name), str(vsx_type))
        return {"aavso": aavso_path}

    v_lines: list[str] = []
    v_lines.extend(_vyvar_export_citation_lines(fresh_cfg, **_cite_kw))
    v_lines.append("# VYVAR — Differential Ensemble Photometry\n")
    v_lines.append(f"# Software: {_sw_ver} | Observer: {observer_name}\n")
    _append_varastro_site_line(v_lines, fresh_cfg, _resolved_site)
    if obs_group_resolved:
        v_lines.append(f"# Setup: {obs_group_resolved}\n")
    v_lines.append("#\n")
    v_lines.append("# TIME SYSTEM: BJD(TDB)\n")

    bjd_hjd_median = float("nan")
    if "hjd" in lc_normal.columns and "bjd" in lc_normal.columns:
        try:
            off = (pd.to_numeric(lc_normal["bjd"], errors="coerce") - pd.to_numeric(lc_normal["hjd"], errors="coerce")).dropna()
            if len(off) > 0:
                bjd_hjd_median = float(off.median())
        except Exception:  # noqa: BLE001
            bjd_hjd_median = float("nan")
    if math.isfinite(bjd_hjd_median):
        v_lines.append(f"#   BJD-HJD approx (median): {bjd_hjd_median:.6f}\n")
    v_lines.append("#\n")
    v_lines.extend(_varastro_alg_lines(fresh_cfg, photometry_dir=_phot_dir, lc_method=_export_method))
    _cfg_tier = fresh_cfg
    _t1 = float(getattr(_cfg_tier, "comp_tier1_bprp_limit", 0.25) or 0.25)
    _t2 = float(getattr(_cfg_tier, "comp_tier2_bprp_limit", 0.48) or 0.48)
    _t3 = float(getattr(_cfg_tier, "comp_tier3_bprp_limit", 0.79) or 0.79)
    v_lines.append(
        f"#   Tier system (|ΔBP-RP| Gaia): "
        f"T1≤{_t1:.2f}(w=1.00) T2≤{_t2:.2f}(w=0.85) T3≤{_t3:.2f}(w=0.50) T4>{_t3:.2f}(w=0.25)\n"
    )
    v_lines.append("# Color system: Gaia BP-RP\n")
    if setup_filter_raw or exptime_s or binning:
        v_lines.append(
            f"#   Filter: {setup_filter_raw or 'CV'} | Exp: {exptime_s or 'na'}s | Bin: {binning or 'na'}\n"
        )
    ap_px = pd.to_numeric(summary_row.get("aperture_px", float("nan")), errors="coerce")
    fwhm_px = pd.to_numeric(summary_row.get("fwhm_px", float("nan")), errors="coerce")
    n_frames = pd.to_numeric(summary_row.get("n_frames", float("nan")), errors="coerce")
    n_good_comp = pd.to_numeric(summary_row.get("n_good_comp", float("nan")), errors="coerce")
    lc_rms = pd.to_numeric(summary_row.get("lc_rms", float("nan")), errors="coerce")
    if math.isfinite(float(ap_px)) and arcsec_per_px is not None and math.isfinite(float(arcsec_per_px)):
        v_lines.append(
            f"#   Aperture: {float(ap_px):.2f}px ({float(ap_px) * float(arcsec_per_px):.2f}arcsec)\n"
        )
    if math.isfinite(float(fwhm_px)):
        v_lines.append(f"#   FWHM: {float(fwhm_px):.2f}px\n")
    if math.isfinite(float(n_frames)) or math.isfinite(float(n_good_comp)) or math.isfinite(float(lc_rms)):
        v_lines.append(
            f"#   n_frames: {_fmt_opt_int(n_frames)} | "
            f"n_good_comp: {_fmt_opt_int(n_good_comp)} | "
            f"lc_rms: {_fmt_opt_num(lc_rms, '.4f')}\n"
        )
    _trust = str(summary_row.get("trust", "") or "").strip().upper()
    _trust_reason = str(summary_row.get("trust_reason", "") or "").strip()
    if _trust:
        from trust_flag_core import format_varastro_trust_comment  # noqa: PLC0415

        v_lines.append(format_varastro_trust_comment(_trust, _trust_reason))
    v_lines.append("#\n")

    v_lines.append(f"# VAR Name: {vsx_name} | Type: {vsx_type}\n")
    v_lines.append(f"#   Catalog: GaiaDR3 | CatalogId: {target_cid or 'na'}\n")
    v_lines.append(
        f"#   CatalogMag: {_fmt_opt_num(t_mag, '.3f')} | "
        f"BP-RP: {_fmt_opt_num(t_bprp, '.3f')}\n"
    )
    v_lines.append("#\n")
    v_lines.append("# COMP TABLE:\n")
    v_lines.append(
        _format_varastro_comp_table(
            comp_df,
            comp_quality_map=comp_quality_map,
        )
    )
    v_lines.append("#\n")
    if check_row is not None:
        chk_mag = pd.to_numeric(check_row.get("mag", float("nan")), errors="coerce")
        chk_p2p = pd.to_numeric(check_row.get("p2p_rms", float("nan")), errors="coerce")
        v_lines.append(
            f"# CHK CatalogId: {check_cid} | Mag: {_fmt_opt_num(chk_mag, '.3f')} | "
            f"p2p_rms: {_fmt_opt_num(chk_p2p, '.4f')}\n"
        )
    else:
        v_lines.append("# CHK CatalogId: na\n")
    v_lines.append("#\n")

    # Field image copy (if available).
    try:
        # output_dir points to .../photometry/lightcurves_reports
        # field maps live next to lightcurves, i.e. .../photometry/lightcurves/
        _lc_dir = out_base.parent / "lightcurves"
        field_img = _copy_field_image(_lc_dir, str(target_cid or ""), (out_base / "varastro"), str(safe), str(date_tag))
    except Exception:  # noqa: BLE001
        field_img = None
    if field_img:
        v_lines.append(f"# FIELD IMAGE: {field_img}\n")
    else:
        v_lines.append("# FIELD IMAGE: not available\n")
    v_lines.append("#\n")

    v_lines.append("# BJD(TDB)       delta_mag  err    mag_calib\n")
    for _, row in lc_normal.iterrows():
        bjd = pd.to_numeric(row.get("bjd", float("nan")), errors="coerce")
        dmag = pd.to_numeric(row.get("delta_mag", float("nan")), errors="coerce")
        err = pd.to_numeric(row.get("err", float("nan")), errors="coerce")
        mag_cal = pd.to_numeric(row.get("mag_calib", float("nan")), errors="coerce")
        if not math.isfinite(float(bjd)):
            continue
        v_lines.append(f"{float(bjd):.6f}   {_fmt_opt_num(dmag, '.4f'):>7}   {_fmt_opt_num(err, '.4f'):>7}   {_fmt_opt_num(mag_cal, '.4f'):>7}\n")

    var_path.write_text("".join(v_lines), encoding="utf-8")
    logging.info("[EXPORT] VAR.ASTRO: %s", str(var_path.name))

    return {"aavso": aavso_path, "varastro": var_path}


def export_all_method_lightcurve_reports(
    output_dir: Path,
    target_row: pd.Series,
    *,
    lc_dir: Path,
    target_cid: str,
    comp_df: pd.DataFrame,
    summary_row: pd.Series,
    cfg: AppConfig | None = None,
    proc_csv_cache: dict[str, pd.DataFrame] | None = None,
    **kwargs: Any,
) -> dict[str, dict[str, Path]]:
    """Export AAVSO + VarAstro for each active photometry method."""
    fresh_cfg = cfg or AppConfig()
    _lc_dir = Path(lc_dir)
    _have_psf_files = any(_lc_dir.glob("lightcurve_*_psf.csv")) or any(
        _lc_dir.glob("lightcurve_*_adaptive.csv")
    )
    _methods = active_report_methods(
        fresh_cfg,
        have_psf_cols=_have_psf_files
        or bool(getattr(fresh_cfg, "psf_photometry_enabled", False))
        or bool(getattr(fresh_cfg, "psf_adaptive_enabled", False)),
    )
    out: dict[str, dict[str, Path]] = {}
    _proc_cache = proc_csv_cache if proc_csv_cache is not None else {}
    for method in _methods:
        lc_path = lc_csv_path(_lc_dir, target_cid, method)
        if not lc_path.is_file():
            continue
        try:
            lc_df = pd.read_csv(lc_path, low_memory=False)
        except Exception:  # noqa: BLE001
            continue
        if lc_df.empty:
            continue
        try:
            paths = export_lightcurve_reports(
                output_dir,
                target_row,
                lc_df,
                comp_df,
                summary_row,
                cfg=fresh_cfg,
                lc_dir=_lc_dir,
                export_method=method,
                active_methods=_methods,
                proc_csv_cache=_proc_cache,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("[EXPORT] %s method %s: %s", target_cid, method, exc)
            continue
        if paths:
            out[method] = paths
    return out

