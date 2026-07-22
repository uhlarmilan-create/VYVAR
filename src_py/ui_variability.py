from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from catalog_crossmatch import check_candidate_in_catalogs
from config import AppConfig
from gaia_catalog_id import normalize_gaia_source_id, read_vyvar_csv
from tess_verify import TessResult, run_tess_analysis
from utils import resolve_draft_dir_path
from variability_detector import compute_rms_variability, compute_vdi, load_field_flux_matrix

LOGGER = logging.getLogger(__name__)


if TYPE_CHECKING:
    from pipeline import AstroPipeline


def _detect_obs_groups(draft_dir: Path) -> list[str]:
    root = Path(draft_dir) / "detrended_aligned" / "lights"
    if not root.is_dir():
        return []
    out: list[str] = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and any(d.glob("proc_*.csv")):
            out.append(d.name)
    return out


@st.cache_data(show_spinner=False)
def _cached_load_matrix(
    per_frame_dir_s: str,
    flux_col: str,
    min_frames_frac: float,
    cfg_dict: dict[str, Any],
    proc_store_nframes: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    # PERF-6: reuse ProcFrameStore from session if available (proc_store_nframes busts cache)
    csv_cache_arg = None
    if proc_store_nframes > 0:
        try:
            _store = getattr(st.session_state, "proc_frame_store", None)
            if _store is not None:
                csv_cache_arg = _store
        except Exception:  # noqa: BLE001
            # EXC-0540: T3 -- UI diagnostic/plot only (if _store is not None: / csv_cache_arg = _store / except Excep... (EXCEPT-BULK 2026-07-08)
            pass

    return load_field_flux_matrix(
        Path(per_frame_dir_s),
        flux_col=flux_col,
        min_frames_frac=min_frames_frac,
        config=cfg_dict,
        csv_cache=csv_cache_arg,
    )


def _read_comp_catalog_ids(platesolve_dir: Path) -> list[str]:
    p = Path(platesolve_dir) / "comparison_stars.csv"
    if not p.exists():
        return []
    try:
        df = read_vyvar_csv(p, low_memory=False)
    except Exception:  # noqa: BLE001
        # EXC-0541: T3 -- UI diagnostic/plot only (try: / df = read_vyvar_csv(p, low_memory=False) / except Excep... (EXCEPT-BULK 2026-07-08)
        return []
    for col in ("catalog_id", "name"):
        if col in df.columns:
            vals = df[col].dropna().astype(str).tolist()
            return vals
    return []


def _vizier_link(ra: float, dec: float) -> str:
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return ""
    vizier_url = (
        f"https://vizier.cds.unistra.fr/viz-bin/VizieR-3"
        f"?-c={ra:.6f}%20{dec:.6f}&-c.rs=10"
        f"&-source=B/vsx/vsx,II/366,J/ApJS/249/18/table2"
        f",I/358/varisum,J/AJ/156/241/table4"
        f",J/AJ/147/119/table1,J/AJ/155/39/Variables"
        f",J/A%2BA/598/A108/tablea11,J/ApJS/258/16/tess-ebs"
    )
    return vizier_url


def _raw_lightcurve_from_frames(per_frame_dir: Path, catalog_id: str, flux_col: str) -> pd.DataFrame:
    frames = sorted(Path(per_frame_dir).glob("proc_*.csv"))
    rows: list[dict[str, Any]] = []
    for p in frames:
        try:
            df = read_vyvar_csv(
                p,
                usecols=["catalog_id", flux_col, "bjd_tdb_mid"],
                low_memory=False,
            )
        except Exception:  # noqa: BLE001
            # EXC-0542: T3 -- UI diagnostic/plot only (low_memory=False, / ) / except Exception:  # noqa: BLE001 / co... (EXCEPT-BULK 2026-07-08)
            continue
        _want = normalize_gaia_source_id(catalog_id)
        df["_cid"] = df["catalog_id"].map(normalize_gaia_source_id)
        sub = df[df["_cid"] == _want] if _want else df.iloc[0:0]
        if sub.empty:
            continue
        # Use first match
        r = sub.iloc[0]
        flux = float(pd.to_numeric(r.get(flux_col), errors="coerce"))
        bjd = float(pd.to_numeric(r.get("bjd_tdb_mid"), errors="coerce"))
        if not (np.isfinite(flux) and flux > 0 and np.isfinite(bjd)):
            continue
        mag_inst = -2.5 * float(np.log10(flux))
        rows.append({"bjd_tdb_mid": bjd, "mag_inst": mag_inst})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("bjd_tdb_mid").reset_index(drop=True)
    return out


def _find_active_targets_csv(draft_dir: Path | str | None) -> Path | None:
    """First ``platesolve/*/photometry/active_targets.csv`` under draft (same order as crossmatch_runner)."""
    if draft_dir is None:
        return None
    ps = Path(draft_dir) / "platesolve"
    if not ps.is_dir():
        return None
    matches = sorted(ps.glob("*/photometry/active_targets.csv"))
    return matches[0] if matches else None


def _variability_results_coordinate_table(var_results: Any) -> pd.DataFrame | None:
    """Flatten ``var_results`` session dict to one table with ``catalog_id`` and sky coordinates.

    ``rms_df`` is listed first so ``drop_duplicates(subset=['catalog_id'], keep='first')`` keeps
    full metadata rows (``vdi_df`` has no ``ra_deg``/``dec_deg``).
    """
    if var_results is None:
        return None
    if isinstance(var_results, pd.DataFrame):
        return var_results if not var_results.empty else None
    if not isinstance(var_results, dict):
        return None
    frames: list[pd.DataFrame] = []
    for key in ("rms_df", "vdi_df", "combined_df"):
        df_ = var_results.get(key)
        if isinstance(df_, pd.DataFrame) and not df_.empty and "catalog_id" in df_.columns:
            frames.append(df_.copy())
    if not frames:
        return None
    comb = pd.concat(frames, ignore_index=True)
    comb["_cid_norm"] = comb["catalog_id"].astype(str).str.strip()
    comb = comb[comb["_cid_norm"].str.len() > 0].copy()
    comb = comb.drop_duplicates(subset=["_cid_norm"], keep="first").drop(columns=["_cid_norm"], errors="ignore")

    rename_map: dict[str, str] = {}
    if "ra_deg" not in comb.columns:
        for alt in ("RAJ2000", "RA", "ra"):
            if alt in comb.columns:
                rename_map[alt] = "ra_deg"
                break
    if "dec_deg" not in comb.columns:
        for alt in ("DEJ2000", "DE", "DEC", "dec"):
            if alt in comb.columns:
                rename_map[alt] = "dec_deg"
                break
    if rename_map:
        comb = comb.rename(columns=rename_map)
    return comb


def _get_candidate_row(
    var_results: Any,
    cid: str,
    *,
    draft_dir: Path | str | None = None,
    platesolve_dir: Path | str | None = None,
) -> dict | None:
    """Vrati dict s ra, dec, mag pre cid z var_results."""
    if var_results is None:
        return None
    try:
        df = _variability_results_coordinate_table(var_results)
        if df is None or df.empty:
            return None

        id_col = "catalog_id" if "catalog_id" in df.columns else None
        if id_col is None:
            id_col = next(
                (c for c in df.columns if "catalog_id" in str(c).lower() or str(c) == "id"),
                None,
            )
        if id_col is None:
            return None

        row = df[df[id_col].astype(str).str.strip() == str(cid).strip()]
        if row.empty:
            return None
        r = row.iloc[0]

        ra = None
        dec = None
        for rc in ("ra", "RAJ2000", "RA", "ra_deg"):
            if rc in r.index and pd.notna(r[rc]):
                ra = float(r[rc])
                break
        for dc in ("dec", "DEJ2000", "DE", "dec_deg"):
            if dc in r.index and pd.notna(r[dc]):
                dec = float(r[dc])
                break

        if ra is None or dec is None:
            return None

        mag = None
        for mc in ("mag", "Vmag", "mag_median"):
            if mc in r.index and pd.notna(r[mc]):
                mag = float(r[mc])
                break

        # TODO-16: prefer active_targets.csv coords (WCS-verified; mirrors crossmatch_runner)
        try:
            _at_path: Path | None = None
            if platesolve_dir is not None:
                _at_cand = Path(platesolve_dir) / "photometry" / "active_targets.csv"
                if _at_cand.is_file():
                    _at_path = _at_cand
            if _at_path is None:
                _dd = draft_dir
                if _dd is None:
                    try:
                        _pd = st.session_state.get("var_photometry_dir")
                        if _pd:
                            _at_cand = Path(str(_pd)) / "active_targets.csv"
                            if _at_cand.is_file():
                                _at_path = _at_cand
                    except Exception:  # noqa: BLE001
                        # EXC-0543: T3 -- UI diagnostic/plot only (if _at_cand.is_file(): / _at_path = _at_cand / except Exceptio... (EXCEPT-BULK 2026-07-08)
                        pass
                else:
                    _at_path = _find_active_targets_csv(_dd)
            if _at_path is not None and _at_path.is_file():
                _at_df = read_vyvar_csv(_at_path, low_memory=False)
                if "catalog_id" in _at_df.columns:
                    _at_row = _at_df[
                        _at_df["catalog_id"].astype(str).str.strip() == str(cid).strip()
                    ]
                    if not _at_row.empty:
                        _at_ra = pd.to_numeric(_at_row.iloc[0].get("ra_deg"), errors="coerce")
                        _at_dec = pd.to_numeric(_at_row.iloc[0].get("dec_deg"), errors="coerce")
                        if pd.notna(_at_ra) and pd.notna(_at_dec):
                            ra = float(_at_ra)
                            dec = float(_at_dec)
        except Exception:  # noqa: BLE001
            # EXC-0544: T3 -- UI diagnostic/plot only (ra = float(_at_ra) / dec = float(_at_dec) / except Exception: ... (EXCEPT-BULK 2026-07-08)
            pass

        return {"ra": ra, "dec": dec, "mag": mag}
    except Exception:  # noqa: BLE001
        # EXC-0545: T3 -- UI diagnostic/plot only (return {'ra': ra, 'dec': dec, 'mag': mag} / except Exception: ... (EXCEPT-BULK 2026-07-08)
        return None


def _katalogy_positive_lines(text: str) -> list[str]:
    """Lines that indicate a positive catalog match (VSX / ASAS-SN / Gaia var, ...)."""
    out: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[telescope]"):
            continue
        if "ziadny zaznam" in line.lower():
            continue
        if "no match" in line.lower():
            continue
        out.append(line)
    return out


def _should_trigger_tess(bullets_text: str) -> bool:
    """Auto-TESS len pre kandidatov bez pozitivneho katalogoveho matchu."""
    return not bool(_katalogy_positive_lines(bullets_text))


def _katalogy_display(bullets_text: str) -> str:
    """Katalogy stlpec: doplni vizualnu znacku ak sa spusti auto-TESS."""
    base = str(bullets_text).strip() if bullets_text is not None else ""
    if not base:
        base = "-"
    if _should_trigger_tess(base):
        suf = "\n[telescope] TESS auto"
        if "[telescope] TESS auto" not in base:
            return base + suf
    return base


def _katalogy_column_name(columns: Any) -> str | None:
    for c in columns:
        cl = str(c).strip().lower().replace("o", "o")
        if cl in ("katalogy", "katalog"):
            return str(c)
    return None


def load_katalogy_map_from_disk(draft_dir: Path, obs_group: str) -> dict[str, str]:
    """``catalog_id`` -> katalogy text from ``variability_candidates.csv`` (pipeline ground truth)."""
    out: dict[str, str] = {}
    paths = [
        Path(draft_dir) / "platesolve" / str(obs_group) / "variability_candidates.csv",
        Path(draft_dir) / "platesolve" / str(obs_group) / "photometry" / "variability_candidates.csv",
    ]
    for p in paths:
        if not p.is_file():
            continue
        try:
            df = read_vyvar_csv(p, low_memory=False)
        except Exception:  # noqa: BLE001
            # EXC-0546: T3 -- UI diagnostic/plot only (try: / df = read_vyvar_csv(p, low_memory=False) / except Excep... (EXCEPT-BULK 2026-07-08)
            continue
        if df.empty or "catalog_id" not in df.columns:
            continue
        col = _katalogy_column_name(df.columns)
        if not col:
            continue
        for _, row in df.iterrows():
            cid = str(row.get("catalog_id", "") or "").strip()
            if not cid:
                continue
            txt = str(row.get(col, "") or "").strip()
            if txt and txt.lower() not in ("-", "nan", "none"):
                out[cid] = txt
        if out:
            return out
    return out


def load_tess_eligible_candidate_ids_from_disk(draft_dir: Path, obs_group: str) -> frozenset[str]:
    """Kandidati z CSV bez known VSX/Gaia flag a bez pozitivneho katalogoveho matchu (auto-TESS)."""
    ids: set[str] = set()
    paths = [
        Path(draft_dir) / "platesolve" / str(obs_group) / "photometry" / "variability_candidates.csv",
        Path(draft_dir) / "platesolve" / str(obs_group) / "variability_candidates.csv",
    ]
    for p in paths:
        if not p.is_file():
            continue
        try:
            df = read_vyvar_csv(p, low_memory=False)
        except Exception:  # noqa: BLE001
            # EXC-0547: T3 -- UI diagnostic/plot only (try: / df = read_vyvar_csv(p, low_memory=False) / except Excep... (EXCEPT-BULK 2026-07-08)
            continue
        if df.empty or "catalog_id" not in df.columns:
            continue
        kat_col = _katalogy_column_name(df.columns)
        for _, row in df.iterrows():
            cid = str(row.get("catalog_id", "") or "").strip()
            if not cid:
                continue
            if "vsx_known_variable" in df.columns:
                try:
                    if bool(pd.to_numeric(row.get("vsx_known_variable"), errors="coerce")):
                        continue
                except Exception:  # noqa: BLE001
                    # EXC-0548: T3 -- UI diagnostic/plot only (if bool(pd.to_numeric(row.get('vsx_known_variable'), errors='c... (EXCEPT-BULK 2026-07-08)
                    pass
            if "vsx_match" in df.columns:
                try:
                    if bool(pd.to_numeric(row.get("vsx_match"), errors="coerce")):
                        continue
                except Exception:  # noqa: BLE001
                    # EXC-0549: T3 -- UI diagnostic/plot only (if bool(pd.to_numeric(row.get('vsx_match'), errors='coerce')):... (EXCEPT-BULK 2026-07-08)
                    pass
            if "gaia_dr3_variable_catalog" in df.columns:
                try:
                    if bool(pd.to_numeric(row.get("gaia_dr3_variable_catalog"), errors="coerce")):
                        continue
                except Exception:  # noqa: BLE001
                    # EXC-0550: T3 -- UI diagnostic/plot only (if bool(pd.to_numeric(row.get('gaia_dr3_variable_catalog'), er... (EXCEPT-BULK 2026-07-08)
                    pass
            if kat_col and kat_col in df.columns:
                if not _should_trigger_tess(str(row.get(kat_col, "") or "")):
                    continue
            ids.add(cid)
        if ids:
            return frozenset(ids)
    return frozenset()


def tess_catalog_ids_for_auto_run(
    draft_dir: Path,
    obs_group: str,
    memory_catalog_ids: list[str],
) -> list[str]:
    """Auto-TESS: disk CSV intersect memory, len bez known variable / katalogoveho matchu."""
    eligible = load_tess_eligible_candidate_ids_from_disk(draft_dir, obs_group)
    if not eligible:
        return []
    mem = [str(c).strip() for c in memory_catalog_ids if str(c).strip()]
    if mem:
        return [c for c in mem if c in eligible]
    return sorted(eligible)


def _merge_katalogy_maps(
    bullets_map: dict[str, str],
    disk_map: dict[str, str],
) -> dict[str, str]:
    """Disk CSV wins over in-memory session bullets (may be stale after pipeline crossmatch)."""
    merged = {str(k): str(v) for k, v in (bullets_map or {}).items()}
    for cid, txt in (disk_map or {}).items():
        merged[str(cid).strip()] = str(txt).strip()
    return merged


def _edge_ok_from_masterstar(
    masterstar_fits: Path,
    stars_df: pd.DataFrame,
    cfg_dict: dict[str, Any],
) -> pd.Series:
    """
    Per-star edge safety (annulus-aware, best-effort).

    Uses MASTERSTAR dimensions and margin ~= outer annulus radius in px.
    If FITS/header missing, returns True for all rows.
    """
    if stars_df is None or stars_df.empty:
        return pd.Series(dtype=bool)
    if not masterstar_fits.exists():
        return pd.Series(True, index=stars_df.index)

    try:
        from astropy.io import fits as astrofits
    except Exception:  # noqa: BLE001
        # EXC-0551: T3 -- UI diagnostic/plot only (try: / from astropy.io import fits as astrofits / except Excep... (EXCEPT-BULK 2026-07-08)
        return pd.Series(True, index=stars_df.index)

    nx = ny = None
    fwhm_px = float("nan")
    try:
        with astrofits.open(masterstar_fits, memmap=False) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data
        try:
            fwhm_px = float(hdr.get("VY_FWHM", float("nan")))
        except Exception:  # noqa: BLE001
            fwhm_px = float("nan")
        try:
            if data is not None and hasattr(data, "shape") and len(data.shape) >= 2:
                ny, nx = int(data.shape[-2]), int(data.shape[-1])
        except Exception:  # noqa: BLE001
            nx = ny = None
    except Exception:  # noqa: BLE001
        nx = ny = None

    # Base margin (same spirit as phase01 interior margin)
    try:
        base_margin = float(cfg_dict.get("phase01_chip_interior_margin_px", 100))
    except Exception:  # noqa: BLE001
        base_margin = 100.0

    # Annulus-aware margin: outer annulus radius in px + small guard
    try:
        ann_outer_fwhm = float(cfg_dict.get("annulus_outer_fwhm", 9.0))
    except Exception:  # noqa: BLE001
        ann_outer_fwhm = 9.0
    ann_margin = float(ann_outer_fwhm) * float(fwhm_px) + 5.0 if np.isfinite(fwhm_px) else float("nan")

    margin = float(base_margin)
    if np.isfinite(ann_margin):
        margin = max(float(margin), float(ann_margin))

    x = pd.to_numeric(stars_df.get("x"), errors="coerce")
    y = pd.to_numeric(stars_df.get("y"), errors="coerce")
    ok = np.isfinite(x) & np.isfinite(y)
    if nx is not None and ny is not None and nx > 0 and ny > 0 and np.isfinite(margin) and margin >= 0:
        ok = ok & (x >= margin) & (x <= float(nx) - margin) & (y >= margin) & (y <= float(ny) - margin)

    return ok.fillna(False).astype(bool)


def count_edge_safe_combined_candidates(
    rms_df: pd.DataFrame,
    vdi_df: pd.DataFrame,
    platesolve_dir: Path,
    cfg_dict: dict[str, Any],
) -> int:
    """Pocet kandidatov (RMS|VDI) bez VSX a s edge_ok - rovnaka logika ako v dashboarde."""
    if rms_df is None or rms_df.empty:
        return 0
    results_df = rms_df.copy()
    if vdi_df is not None and not vdi_df.empty:
        results_df = results_df.merge(
            vdi_df[["catalog_id", "vdi_score", "vdi_z_score", "is_variable_candidate"]],
            on="catalog_id",
            how="left",
            suffixes=("_rms", "_vdi"),
        )
        results_df = results_df.rename(columns={"is_variable_candidate": "is_variable_candidate_vdi"})
    else:
        results_df["vdi_score"] = np.nan
        results_df["vdi_z_score"] = np.nan
        results_df["is_variable_candidate_vdi"] = False
    if "is_variable_candidate" in results_df.columns and "is_variable_candidate_rms" not in results_df.columns:
        results_df = results_df.rename(columns={"is_variable_candidate": "is_variable_candidate_rms"})
    results_df["is_variable_candidate_rms"] = results_df["is_variable_candidate_rms"].fillna(False).astype(bool)
    results_df["is_variable_candidate_vdi"] = results_df["is_variable_candidate_vdi"].fillna(False).astype(bool)
    results_df["is_candidate_combined"] = (
        results_df["is_variable_candidate_rms"] | results_df["is_variable_candidate_vdi"]
    )
    work = results_df.copy()
    work["is_candidate_combined"] = work["is_candidate_combined"].fillna(False).astype(bool)
    work["vsx_known_variable"] = work["vsx_known_variable"].fillna(False).astype(bool)
    work["gaia_dr3_variable_catalog"] = work["gaia_dr3_variable_catalog"].fillna(False).astype(bool)
    masterstar_fits = platesolve_dir / "MASTERSTAR.fits"
    edge_ok = _edge_ok_from_masterstar(masterstar_fits, work, cfg_dict)
    work["edge_ok"] = edge_ok.reindex(work.index).fillna(False).astype(bool)
    cand_mask = work["is_candidate_combined"] & ~work["vsx_known_variable"] & work["edge_ok"]
    return int(cand_mask.sum())


def run_variability_detection_session(
    *,
    cfg: "AppConfig",
    draft_dir: Path,
    obs_group: str,
    flux_col: str,
    min_frames_pct: int,
    sigma_thr: float,
    mag_limit: float,
) -> tuple[dict[str, Any], int, tuple[str, str, int, float, float]]:
    """
    Nacita maticu fluxov, RMS + VDI (vzdy obe). Nevola Streamlit API.
    Vracia (results dict ako var_results, pocet edge-safe kandidatov, _var_run_sig).
    """
    cfg_dict = cfg.to_dict()
    per_frame_dir = draft_dir / "detrended_aligned" / "lights" / str(obs_group)
    platesolve_dir = draft_dir / "platesolve" / str(obs_group)
    cfg_run = dict(cfg_dict)
    cfg_run["variability_min_frames_frac"] = float(min_frames_pct) / 100.0
    _store_nframes = 0
    try:
        _store = getattr(st.session_state, "proc_frame_store", None)
        if _store is not None:
            _store_nframes = len(_store)
    except Exception:  # noqa: BLE001
        # EXC-0552: T3 -- UI diagnostic/plot only (if _store is not None: / _store_nframes = len(_store) / except... (EXCEPT-BULK 2026-07-08)
        pass
    fm, meta, _bjd = _cached_load_matrix(
        str(per_frame_dir),
        flux_col,
        float(min_frames_pct) / 100.0,
        cfg_run,
        _store_nframes,
    )
    _at_path = platesolve_dir / "photometry" / "active_targets.csv"
    if _at_path.is_file() and not meta.empty:
        try:
            _at_zf = read_vyvar_csv(_at_path, low_memory=False)
            if "zone_flag" in _at_zf.columns and "catalog_id" in _at_zf.columns:
                _zf_map = (
                    _at_zf.drop_duplicates("catalog_id", keep="first")
                    .set_index("catalog_id")["zone_flag"]
                    .astype(str)
                )
                meta = meta.copy()
                meta["zone_flag"] = meta.index.astype(str).map(_zf_map)
        except Exception:  # noqa: BLE001
            # EXC-0553: T3 -- UI diagnostic/plot only (meta = meta.copy() / meta['zone_flag'] = meta.index.astype(str... (EXCEPT-BULK 2026-07-08)
            pass
    comp_ids = _read_comp_catalog_ids(platesolve_dir)
    comp_rms_map: dict[str, float] = {}
    comp_csv = platesolve_dir / "photometry" / "comparison_stars_per_target.csv"
    if comp_csv.exists():
        try:
            comp_df = read_vyvar_csv(comp_csv, low_memory=False)
            if "catalog_id" in comp_df.columns and "comp_rms" in comp_df.columns:
                for _, row in comp_df.iterrows():
                    cid = str(row.get("catalog_id", "")).strip()
                    rms = float(pd.to_numeric(row.get("comp_rms"), errors="coerce"))
                    if cid and np.isfinite(rms) and rms > 1e-4:
                        if cid not in comp_rms_map or rms < comp_rms_map[cid]:
                            comp_rms_map[cid] = float(rms)
        except Exception:  # noqa: BLE001
            comp_rms_map = {}
    results: dict[str, Any] = {
        "flux_matrix": fm,
        "metadata": meta,
        "comp_ids": comp_ids,
        "obs_group": str(obs_group),
        "flux_col": flux_col,
        "comp_rms_map": comp_rms_map,
    }
    cfg_run = dict(cfg_dict)
    cfg_run["variability_sigma_threshold"] = float(sigma_thr)
    cfg_run["variability_mag_limit"] = float(mag_limit)
    cfg_run["variability_min_frames_frac"] = float(min_frames_pct) / 100.0
    rms_df = compute_rms_variability(
        fm,
        meta,
        comp_ids,
        sigma_threshold=float(sigma_thr),
        vsx_targets_csv=(platesolve_dir / "variable_targets.csv"),
        config=cfg_run,
        comp_rms_map=comp_rms_map,
    )
    cfg_run = dict(cfg_dict)
    cfg_run["variability_min_frames"] = int(cfg_run.get("variability_min_frames", 30))
    vdi_df = compute_vdi(fm, meta, min_frames=30, config=cfg_run)
    results["vdi_df"] = vdi_df

    # Ensure detection_method is present in the stored rms_df (used by PDF cover table).
    try:
        rms_df2 = rms_df.copy()
        if "detection_method" not in rms_df2.columns:
            rms_df2["detection_method"] = "-"
        # RMS candidate flag (compute_rms_variability uses is_variable_candidate)
        if "is_variable_candidate" in rms_df2.columns:
            m_rms = rms_df2["is_variable_candidate"].fillna(False).astype(bool)
            rms_df2.loc[m_rms, "detection_method"] = "RMS"
        # If VDI available, mark combined candidates
        if isinstance(vdi_df, pd.DataFrame) and (not vdi_df.empty) and ("catalog_id" in vdi_df.columns):
            if "is_variable_candidate" in vdi_df.columns:
                vdi_ids = set(
                    str(x).strip()
                    for x in vdi_df.loc[
                        vdi_df["is_variable_candidate"].fillna(False).astype(bool), "catalog_id"
                    ]
                    .astype(str)
                    .tolist()
                    if str(x).strip()
                )
                if vdi_ids and "catalog_id" in rms_df2.columns:
                    m_vdi = rms_df2["catalog_id"].astype(str).isin(vdi_ids)
                    # upgrade method label
                    m_both = m_vdi & (rms_df2["detection_method"].astype(str) == "RMS")
                    rms_df2.loc[m_vdi & ~m_both, "detection_method"] = "VDI"
                    rms_df2.loc[m_both, "detection_method"] = "RMS+VDI"
    except Exception:  # noqa: BLE001
        rms_df2 = rms_df

    results["rms_df"] = rms_df2
    var_sig = (str(obs_group), str(flux_col), int(min_frames_pct), float(sigma_thr), float(mag_limit))
    n_cand = count_edge_safe_combined_candidates(rms_df2, vdi_df, platesolve_dir, cfg_dict)
    return results, n_cand, var_sig


@st.cache_data(ttl=600, show_spinner=False)
def _render_field_image_with_candidate(
    masterstar_fits_path_s: str,
    *,
    x: float,
    y: float,
    label: str,
    percentile_lo: float = 5.0,
    percentile_hi: float = 99.5,
) -> bytes | None:
    """Render MASTERSTAR FITS as PNG and mark candidate at (x,y)."""
    try:
        from astropy.io import fits as astrofits
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
    except Exception:  # noqa: BLE001
        # EXC-0554: T3 -- UI diagnostic/plot only (import matplotlib.pyplot as plt / from io import BytesIO / exc... (EXCEPT-BULK 2026-07-08)
        return None

    p = Path(masterstar_fits_path_s)
    if not p.exists():
        return None

    try:
        with astrofits.open(p, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:  # noqa: BLE001
        # EXC-0555: T3 -- UI diagnostic/plot only (with astrofits.open(p, memmap=False) as hdul: / data = np.asar... (EXCEPT-BULK 2026-07-08)
        return None

    if data.size == 0:
        return None

    ok = np.isfinite(data)
    if not ok.any():
        return None
    try:
        vmin = float(np.percentile(data[ok], float(percentile_lo)))
        vmax = float(np.percentile(data[ok], float(percentile_hi)))
    except Exception:  # noqa: BLE001
        vmin, vmax = float("nan"), float("nan")

    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=140)
    ax.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=vmin if np.isfinite(vmin) else None,
        vmax=vmax if np.isfinite(vmax) else None,
        aspect="equal",
    )
    ax.scatter([float(x)], [float(y)], s=140, facecolors="none", edgecolors="#ff3333", linewidths=2.0)
    ax.scatter([float(x)], [float(y)], s=18, c="#ff3333", alpha=0.95)
    ax.text(float(x) + 18, float(y), str(label)[:24], color="#ff3333", fontsize=9, va="center")
    ax.set_title("Star field (MASTERSTAR) - selected candidate", fontsize=11)
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@st.cache_data(ttl=600, show_spinner=False)
def _render_field_image_with_candidates(
    masterstar_fits_path_s: str,
    cand_xy_label: tuple[tuple[float, float, str], ...],
    *,
    vsx_xy_label: tuple[tuple[float, float, str], ...] | None = None,
    selected_xy: tuple[float, float] | None = None,
    selected_label: str = "",
    percentile_lo: float = 5.0,
    percentile_hi: float = 99.5,
) -> bytes | None:
    """Render MASTERSTAR FITS as PNG: optional Known VSX (orange), candidates (red), selection (yellow)."""
    try:
        from astropy.io import fits as astrofits
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
    except Exception:  # noqa: BLE001
        # EXC-0556: T3 -- UI diagnostic/plot only (import matplotlib.pyplot as plt / from io import BytesIO / exc... (EXCEPT-BULK 2026-07-08)
        return None

    p = Path(masterstar_fits_path_s)
    if not p.exists():
        return None

    try:
        with astrofits.open(p, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:  # noqa: BLE001
        # EXC-0557: T3 -- UI diagnostic/plot only (with astrofits.open(p, memmap=False) as hdul: / data = np.asar... (EXCEPT-BULK 2026-07-08)
        return None

    if data.size == 0:
        return None

    ok = np.isfinite(data)
    if not ok.any():
        return None
    try:
        vmin = float(np.percentile(data[ok], float(percentile_lo)))
        vmax = float(np.percentile(data[ok], float(percentile_hi)))
    except Exception:  # noqa: BLE001
        vmin, vmax = float("nan"), float("nan")

    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=140)
    ax.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=vmin if np.isfinite(vmin) else None,
        vmax=vmax if np.isfinite(vmax) else None,
        aspect="equal",
    )

    # Known VSX in field (orange) - drawn under candidate markers
    if vsx_xy_label:
        xs_v = [float(t[0]) for t in vsx_xy_label]
        ys_v = [float(t[1]) for t in vsx_xy_label]
        ax.scatter(xs_v, ys_v, s=100, facecolors="none", edgecolors="#f39c12", linewidths=1.6, alpha=0.95)
        ax.scatter(xs_v, ys_v, s=14, c="#f39c12", alpha=0.9)
        for (vx, vy, vlab) in vsx_xy_label[:35]:
            try:
                ax.text(float(vx) + 14, float(vy), str(vlab)[:18], color="#f39c12", fontsize=7, va="center")
            except Exception:  # noqa: BLE001
                # EXC-0558: T3 -- UI diagnostic/plot only (try: / ax.text(float(vx) + 14, float(vy), str(vlab)[:18], colo... (EXCEPT-BULK 2026-07-08)
                continue

    # Candidates: red circles
    if cand_xy_label:
        xs = [float(t[0]) for t in cand_xy_label]
        ys = [float(t[1]) for t in cand_xy_label]
        ax.scatter(xs, ys, s=110, facecolors="none", edgecolors="#ff3333", linewidths=1.8, alpha=0.95)

        for (cx, cy, lab) in cand_xy_label[:25]:
            try:
                ax.text(float(cx) + 16, float(cy), str(lab)[:18], color="#ff3333", fontsize=7, va="center")
            except Exception:  # noqa: BLE001
                # EXC-0559: T3 -- UI diagnostic/plot only (try: / ax.text(float(cx) + 16, float(cy), str(lab)[:18], color... (EXCEPT-BULK 2026-07-08)
                continue

    # Selected candidate highlight (yellow)
    if selected_xy is not None:
        try:
            sx, sy = float(selected_xy[0]), float(selected_xy[1])
            if np.isfinite(sx) and np.isfinite(sy):
                ax.scatter([sx], [sy], s=190, facecolors="none", edgecolors="#ffd54a", linewidths=2.6)
                ax.scatter([sx], [sy], s=26, c="#ffd54a", alpha=0.95)
                if selected_label:
                    ax.text(sx + 16, sy, str(selected_label)[:22], color="#ffd54a", fontsize=9, va="center")
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[VARIABILITY] Matplotlib overlay failed (non-critical): %s", exc)

    ax.set_title(
        "Star field (MASTERSTAR) - red=candidates, orange=Known VSX, yellow=selected",
        fontsize=10,
    )
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _variability_crossmatch_dialog_body() -> None:
    cid = str(st.session_state.get("var_cm_cid", "") or "")
    ra = st.session_state.get("var_cm_ra")
    dec = st.session_state.get("var_cm_dec")
    mag = st.session_state.get("var_cm_mag")
    st.markdown(f"**catalog_id:** `{cid}`")
    if not (isinstance(ra, (int, float)) and isinstance(dec, (int, float)) and np.isfinite(ra) and np.isfinite(dec)):
        st.error("Invalid RA/Dec for crossmatch.")
        return
    mag_f = float(mag) if isinstance(mag, (int, float)) and np.isfinite(mag) else None
    _vsx_dlg = str(getattr(AppConfig(), "vsx_local_db_path", "") or "").strip() or None
    with st.spinner("Searching catalogs (up to 30 s)..."):
        res = check_candidate_in_catalogs(
            float(ra), float(dec), mag=mag_f, radius_arcsec=10.0, vsx_local_db_path=_vsx_dlg
        )
    st.session_state["crossmatch_result"] = res
    bullets_map = st.session_state.setdefault("var_catalog_bullets", {})
    bullets_map[str(cid)] = "\n".join(res.catalog_summary_bullets())

    tab1, tab2, tab3 = st.tabs(["Catalogs", "TESS", "Export"])
    with tab1:
        catalog_order = [
            "SIMBAD",
            "VSX",
            "ASAS-SN",
            "ZTF",
            "Gaia varisum",
            "ATLAS",
            "CSS",
            "KELT",
            "VSBS",
            "TESS-EB",
        ]
        for cat in catalog_order:
            lst = res.matches.get(cat, [])
            err = res.errors.get(cat)
            if err:
                st.markdown(f":gray[**{cat}** - error: {err}]")
            elif not lst:
                st.markdown(f":gray[**{cat}** - no match]")
            else:
                with st.expander(f"{cat} - {len(lst)} record(s)", expanded=True):
                    for m in lst:
                        st.markdown(
                            f"**{m.name}**  \n"
                            f"type: {m.var_type or '-'} . P: {m.period} . amp: {m.amplitude} . "
                            f"dr: {m.delta_r} . epoch: {m.epoch} . mag: {m.mag}"
                        )
        bp = res.best_period()
        if bp is not None and np.isfinite(bp):
            st.metric("Best period (VSX->ASAS-SN->ZTF priority...)", f"{bp:.6g} d")
        else:
            st.metric("Best period (VSX->ASAS-SN->ZTF priority...)", "-")
    with tab2:
        candidate_catalog_id = str(cid)
        candidate_ra = float(ra)
        candidate_dec = float(dec)
        candidate_mag = mag_f
        _tess_cfg_dialog = AppConfig()
        _tess_allowed = bool(getattr(_tess_cfg_dialog, "tess_enabled", False))
        if not _tess_allowed:
            st.info(
                "[TESS] TessCut download (lightkurve) is **disabled** - `tess_enabled`: false in `config.json` "
                "(VYVAR project root). To enable: set `\"tess_enabled\": true` and restart Streamlit."
            )
        tess_store: dict[str, TessResult] = st.session_state.setdefault("tess_results", {})
        tess_result = tess_store.get(candidate_catalog_id)

        if tess_result is None:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.info(
                    "TESS typically provides 1-20+ sectors depending on ecliptic latitude and observing span. "
                    "Analysis downloads FFI cutouts (TessCut), subtracts background, cleans the LC, finds period "
                    "(Lomb-Scargle in day windows or catalog hint) and saves CSV + PNG per sector."
                )
            with c2:
                if st.button(
                    "Run TESS analysis",
                    type="primary",
                    key=f"var_tess_run_{candidate_catalog_id}",
                    disabled=not _tess_allowed,
                    help=(
                        "TESS temporarily disabled in config.json (tess_enabled)."
                        if not _tess_allowed
                        else None
                    ),
                ):
                    cm = st.session_state.get("crossmatch_result")
                    period_hint = None
                    if cm is not None and hasattr(cm, "best_period"):
                        try:
                            period_hint = cm.best_period()
                        except Exception:  # noqa: BLE001
                            period_hint = None
                    photometry_dir = str(st.session_state.get("var_photometry_dir") or "").strip()
                    if not photometry_dir:
                        st.error("Missing photometry path (var_photometry_dir).")
                    else:
                        with st.spinner("Downloading TESS data..."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            def progress_callback(message: str, value: float) -> None:
                                status_text.text(str(message))
                                progress_bar.progress(float(min(1.0, max(0.0, value))))

                            result = run_tess_analysis(
                                catalog_id=candidate_catalog_id,
                                ra=candidate_ra,
                                dec=candidate_dec,
                                mag=candidate_mag,
                                photometry_dir=photometry_dir,
                                period_hint=period_hint,
                                progress_callback=progress_callback,
                                cfg=_tess_cfg_dialog,
                            )
                        st.session_state["tess_results"][candidate_catalog_id] = result
                        st.rerun()
        else:
            if tess_result.error_global:
                st.error(tess_result.error_global)
                if st.button(
                    "Try again",
                    key=f"var_tess_retry_{candidate_catalog_id}",
                    disabled=not _tess_allowed,
                    help=("Enable tess_enabled in config.json first." if not _tess_allowed else None),
                ):
                    st.session_state["tess_results"].pop(candidate_catalog_id, None)
                    st.rerun()
            else:
                col_info, col_rerun = st.columns([4, 1])
                with col_info:
                    st.caption(
                        f"Analyzed: {tess_result.total_sectors_found} "
                        f"sectors . {tess_result.total_sectors_ok} OK"
                    )
                with col_rerun:
                    if st.button(
                        "Re-run",
                        key=f"tess_rerun_{candidate_catalog_id}",
                        help=(
                            "Enable tess_enabled in config.json first."
                            if not _tess_allowed
                            else "Delete cached result and re-run TESS analysis"
                        ),
                        type="secondary",
                        disabled=not _tess_allowed,
                    ):
                        st.session_state["tess_results"].pop(candidate_catalog_id, None)
                        st.session_state.get("accepted_period_msg", {}).pop(candidate_catalog_id, None)
                        st.rerun()

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sectors found", str(tess_result.total_sectors_found))
                m2.metric("Sectors OK", str(tess_result.total_sectors_ok))
                bp = tess_result.best_period()
                m3.metric("Period P", f"{bp:.6f} d" if bp is not None and np.isfinite(bp) else "-")
                p2c = tess_result.period_2p_consensus
                m4.metric("Period 2P", f"{p2c:.6f} d" if p2c is not None and np.isfinite(p2c) else "-")

                _rel = getattr(tess_result, "period_reliability", "unknown")
                _rel_reason = getattr(tess_result, "period_reliability_reason", "")
                _badge = {
                    "reliable": "[green] reliable",
                    "uncertain": "[yellow] uncertain",
                    "noise": "[red] noise / no period",
                }.get(_rel, "o unknown")
                st.caption(f"Period reliability: {_badge}  -  {_rel_reason}")

                ok_sectors = [s for s in tess_result.sectors if s.error is None]
                if not ok_sectors:
                    st.warning("No sector without errors.")
                else:
                    sector_ids = [s.sector for s in ok_sectors]
                    pick = st.radio(
                        "Sector",
                        options=sector_ids,
                        format_func=lambda s: f"Sector {s}",
                        horizontal=True,
                        key=f"tess_sector_pick_{candidate_catalog_id}",
                    )
                    sector = next((s for s in ok_sectors if int(s.sector) == int(pick)), ok_sectors[0])
                    st.session_state["tess_selected_sector"] = int(pick)

                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric(
                        "L-S",
                        f"{float(sector.period_found):.6f} d"
                        if sector.period_found is not None and np.isfinite(sector.period_found)
                        else "-",
                    )
                    col2.metric(
                        "PDM",
                        f"{float(getattr(sector, 'period_pdm', None)):.6f} d"
                        if getattr(sector, "period_pdm", None) is not None and np.isfinite(float(sector.period_pdm))
                        else "-",
                    )
                    col3.metric(
                        "BLS",
                        f"{float(getattr(sector, 'period_bls', None)):.6f} d"
                        if getattr(sector, "period_bls", None) is not None and np.isfinite(float(sector.period_bls))
                        else "-",
                    )
                    col4.metric(
                        "Consensus *",
                        f"{float(getattr(sector, 'period_consensus', None)):.6f} d"
                        if getattr(sector, "period_consensus", None) is not None
                        and np.isfinite(float(sector.period_consensus))
                        else "-",
                        delta=str(getattr(sector, "period_method_used", "lomb-scargle") or "lomb-scargle"),
                        delta_color="off",
                    )
                    col5.metric(
                        "ANOVA",
                        f"{float(getattr(sector, 'period_anova', None)):.6f} d"
                        if getattr(sector, "period_anova", None) is not None
                        and np.isfinite(float(getattr(sector, "period_anova", None)))
                        else "-",
                    )

                    amp = getattr(sector, "amplitude_ppt", None)
                    snr_v = getattr(sector, "snr", None)
                    npts = getattr(sector, "n_points", None)
                    parts = []
                    if amp is not None:
                        parts.append(f"Amplitude: **{amp:.1f} ppt**")
                    if snr_v is not None:
                        parts.append(f"SNR: **{snr_v:.1f}**")
                    if npts:
                        parts.append(f"LC points: **{npts}**")
                    if parts:
                        st.caption("  .  ".join(parts))

                    if sector.lc_raw_path and Path(sector.lc_raw_path).exists():
                        try:
                            df_lc = pd.read_csv(sector.lc_raw_path)
                        except Exception:  # noqa: BLE001
                            df_lc = pd.DataFrame()
                        tcol = "time" if "time" in df_lc.columns else df_lc.columns[0]
                        fcol = "flux" if "flux" in df_lc.columns else df_lc.columns[1]
                        tt = pd.to_numeric(df_lc[tcol], errors="coerce")
                        ff = pd.to_numeric(df_lc[fcol], errors="coerce")
                        ok = np.isfinite(tt) & np.isfinite(ff)
                        figp = go.Figure(
                            data=[
                                go.Scatter(
                                    x=tt[ok],
                                    y=ff[ok],
                                    mode="markers",
                                    marker=dict(size=3, color="#1D9E75"),
                                )
                            ]
                        )
                        figp.update_layout(
                            title=f"{candidate_catalog_id} | Sector {sector.sector} | raw LC",
                            xaxis_title="BJD",
                            yaxis_title="Flux [e-/s]",
                            height=300,
                            margin=dict(l=40, r=20, t=40, b=40),
                        )
                        st.plotly_chart(figp, width="stretch")
                    else:
                        st.warning("Missing light curve CSV for this sector.")

                    col_p, col_2p = st.columns(2)
                    pf = sector.period_found
                    with col_p:
                        if pf is not None and np.isfinite(pf) and sector.plot_phased_p_path and Path(sector.plot_phased_p_path).exists():
                            st.caption(f"Phased P = {float(pf):.6f} d")
                            st.image(str(sector.plot_phased_p_path))
                        else:
                            st.caption("Phased P - unavailable")
                    with col_2p:
                        if pf is not None and np.isfinite(pf) and sector.plot_phased_2p_path and Path(sector.plot_phased_2p_path).exists():
                            st.caption(f"Phased 2P = {float(pf) * 2.0:.6f} d")
                            st.image(str(sector.plot_phased_2p_path))
                        else:
                            st.caption("Phased 2P - unavailable")
                    blend_p = getattr(sector, "blend_check_path", None)
                    if blend_p and Path(str(blend_p)).is_file():
                        st.image(str(blend_p), caption="Blend check - TESS vs Gaia sky")
                    st.caption(
                        "Compare P vs 2P - for EA/EB binaries, 2P is correct if you see two unequal minima"
                    )

                    _acc_msg = st.session_state.get("accepted_period_msg", {})
                    if _acc_msg.get(candidate_catalog_id):
                        st.success(_acc_msg[candidate_catalog_id])

                    acc = st.session_state.setdefault("accepted_period", {})
                    col_accept, col_custom = st.columns(2)
                    with col_accept:
                        b1, b2 = st.columns(2)
                        with b1:
                            if bp is not None and np.isfinite(bp):
                                if st.button(f"Use P = {float(bp):.6f} d", key=f"tess_use_p_{candidate_catalog_id}"):
                                    acc[candidate_catalog_id] = float(bp)
                                    st.session_state.setdefault("accepted_period_msg", {})[
                                        candidate_catalog_id
                                    ] = f"Saved period P = {float(bp):.6f} d for {candidate_catalog_id}."
                        with b2:
                            if p2c is not None and np.isfinite(p2c):
                                if st.button(f"Use 2P = {float(p2c):.6f} d", key=f"tess_use_2p_{candidate_catalog_id}"):
                                    acc[candidate_catalog_id] = float(p2c)
                                    st.session_state.setdefault("accepted_period_msg", {})[
                                        candidate_catalog_id
                                    ] = f"Saved period 2P = {float(p2c):.6f} d for {candidate_catalog_id}."
                    with col_custom:
                        cust = st.number_input(
                            "Custom period (d)",
                            min_value=0.0,
                            value=float(bp) if bp is not None and np.isfinite(bp) else 0.0,
                            step=0.0001,
                            format="%.6f",
                            key=f"tess_custom_num_{candidate_catalog_id}",
                        )
                        if st.button("Use custom", key=f"tess_use_custom_{candidate_catalog_id}"):
                            if cust > 0:
                                acc[candidate_catalog_id] = float(cust)
                                st.session_state.setdefault("accepted_period_msg", {})[
                                    candidate_catalog_id
                                ] = f"Saved custom period = {float(cust):.6f} d for {candidate_catalog_id}."
                            else:
                                st.warning("Enter a positive period.")
    with tab3:
        st.info("Export LC data - will be implemented after TESS analysis")


_variability_crossmatch_dialog = (
    st.dialog("Candidate - catalog crossmatch")(_variability_crossmatch_dialog_body)
    if hasattr(st, "dialog")
    else _variability_crossmatch_dialog_body
)


def render_variability_dashboard(
    pipeline: "AstroPipeline",
    cfg: AppConfig,
    *,
    draft_id: int | None = None,
    draft_dir_override: Path | None = None,
) -> None:
    import pandas as pd  # noqa: PLC0415

    st.header("[search] Variability Detection")
    st.session_state.setdefault("tess_results", {})
    st.session_state.setdefault("accepted_period", {})
    st.session_state.setdefault("accepted_period_msg", {})
    st.session_state.setdefault("var_analysis_done", False)
    st.session_state.setdefault("var_analysis_timestamp", None)
    st.session_state.setdefault("pdf_ready", False)
    st.session_state.setdefault("var_candidate_count_autorun", 0)

    # Draft resolution consistent with other dashboards (Aperture Photometry).
    if draft_id is None and draft_dir_override is None:
        st.info("Load a draft first.")
        return
    draft_dir = resolve_draft_dir_path(
        draft_dir_override, draft_id, cfg.archive_root
    )
    if draft_dir is None:
        st.info("Load a draft first.")
        return

    if st.session_state.get("var_analysis_done"):
        _vts = str(st.session_state.get("var_analysis_timestamp") or "-")
        _nc = int(st.session_state.get("var_candidate_count_autorun", 0))
        st.success(
            f"Analysis complete: {_vts} | Candidates: {_nc} | Started automatically after Aperture Photometry"
        )
    elif not st.session_state.get("var_analysis_done"):
        _vr = st.session_state.get("var_results") or {}
        _rms_m = _vr.get("rms_df")
        if isinstance(_rms_m, pd.DataFrame) and not _rms_m.empty:
            st.info("Results from manual analysis - run Aperture Photometry for auto-update")

    with st.expander("Debug", expanded=False):
        st.write(f"draft_dir: {draft_dir}")
        try:
            st.write(f"pipeline.draft_dir: {getattr(pipeline, 'draft_dir', None)}")
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[VARIABILITY] Dashboard init/session restore failed (non-critical): %s", exc)

    obs_groups = _detect_obs_groups(draft_dir)
    if not obs_groups:
        st.warning("No `proc_*.csv` found in detrended_aligned/lights/.")
        return

    obs_group = st.selectbox("Setup:", options=obs_groups, key="var_obs_group")

    flux_source = st.radio("Flux source:", options=["dao_flux", "psf_flux"], horizontal=True, key="var_flux_source")
    cfg_dict = cfg.to_dict()
    sigma_thr = st.slider(
        "Sigma threshold (RMS score):",
        min_value=2.0,
        max_value=4.0,
        value=float(cfg_dict.get("variability_sigma_threshold", 2.3)),
        step=0.1,
        key="var_sigma_thr",
        help="Higher = fewer candidates (stricter RMS threshold). Default 2.3 from config.json.",
    )
    mag_limit = float(cfg_dict.get("variability_mag_limit", 14.5) or 14.5)
    st.caption(
        f"Variability mag cutoff: {mag_limit:.1f} mag (Settings -> `variability_mag_limit`)"
    )
    min_frames_pct = 100  # vsetky framy (bez slidera)

    per_frame_dir = draft_dir / "detrended_aligned" / "lights" / str(obs_group)
    platesolve_dir = draft_dir / "platesolve" / str(obs_group)
    _phot_dir = platesolve_dir / "photometry"
    _phot_dir.mkdir(parents=True, exist_ok=True)
    st.session_state["var_photometry_dir"] = str(_phot_dir.resolve())
    with st.expander("Debug paths", expanded=False):
        st.write(f"per_frame_dir: {per_frame_dir} (exists={per_frame_dir.exists()})")
        st.write(f"platesolve_dir: {platesolve_dir} (exists={platesolve_dir.exists()})")

    _var_sig = (str(obs_group), str(flux_source), int(min_frames_pct), float(sigma_thr), float(mag_limit))
    if st.session_state.get("_var_run_sig") != _var_sig:
        try:
            with st.spinner("Loading flux matrix and computing RMS + VDI..."):
                results, _n_cand_unused, sig = run_variability_detection_session(
                    cfg=cfg,
                    draft_dir=draft_dir,
                    obs_group=str(obs_group),
                    flux_col=str(flux_source),
                    min_frames_pct=int(min_frames_pct),
                    sigma_thr=float(sigma_thr),
                    mag_limit=float(mag_limit),
                )
            st.session_state["var_results"] = results
            st.session_state["_var_run_sig"] = sig
            st.session_state.pop("crossmatch_auto_done", None)
            try:
                from photometry_core import auto_export_variability_candidates_csv

                _ms_fits = platesolve_dir / "MASTERSTAR.fits"
                _comp_csv = _phot_dir / "comparison_stars_per_target.csv"
                if not _comp_csv.is_file():
                    _comp_csv = platesolve_dir / "comparison_stars.csv"
                _exported = auto_export_variability_candidates_csv(
                    masterstar_fits_path=_ms_fits,
                    comparison_stars_csv=_comp_csv if _comp_csv.is_file() else None,
                    per_frame_csv_dir=per_frame_dir,
                    output_dir=_phot_dir,
                    cfg=cfg,
                )
                if _exported is not None and Path(_exported).is_file():
                    _legacy_csv = draft_dir / "platesolve" / str(obs_group) / "variability_candidates.csv"
                    _legacy_csv.parent.mkdir(parents=True, exist_ok=True)
                    import shutil

                    shutil.copy2(str(_exported), str(_legacy_csv))
                    logging.info(
                        "[VARIABILITY] Auto-exported variability_candidates.csv -> %s (%d rows mirror)",
                        str(_legacy_csv),
                        len(read_vyvar_csv(_exported)),
                    )
            except Exception as _var_export_exc:  # noqa: BLE001
                # EXC-0560: T3 -- UI diagnostic/plot only (len(read_vyvar_csv(_exported)), / ) / except Exception as _var... (EXCEPT-BULK 2026-07-08)
                logging.warning("[VARIABILITY] Auto-export variability_candidates.csv failed: %s", _var_export_exc)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Analysis error: {exc}")
            logging.exception("Variability analysis failed")

    res = st.session_state.get("var_results") or {}
    rms_df: pd.DataFrame = res.get("rms_df") if isinstance(res.get("rms_df"), pd.DataFrame) else pd.DataFrame()
    vdi_df: pd.DataFrame = res.get("vdi_df") if isinstance(res.get("vdi_df"), pd.DataFrame) else pd.DataFrame()

    if rms_df.empty:
        st.session_state["var_candidates"] = []

    if not rms_df.empty:
        _vsx_local_cm = str(getattr(cfg, "vsx_local_db_path", "") or "").strip() or None
        # Merge RMS + VDI (if available)
        results_df = rms_df.copy()
        if not vdi_df.empty:
            results_df = results_df.merge(
                vdi_df[["catalog_id", "vdi_score", "vdi_z_score", "is_variable_candidate"]],
                on="catalog_id",
                how="left",
                suffixes=("_rms", "_vdi"),
            )
            # Rename the VDI candidate flag to avoid confusion
            results_df = results_df.rename(columns={"is_variable_candidate": "is_variable_candidate_vdi"})
        else:
            results_df["vdi_score"] = np.nan
            results_df["vdi_z_score"] = np.nan
            results_df["is_variable_candidate_vdi"] = False

        # RMS candidate flag rename
        if "is_variable_candidate" in results_df.columns and "is_variable_candidate_rms" not in results_df.columns:
            results_df = results_df.rename(columns={"is_variable_candidate": "is_variable_candidate_rms"})

        results_df["is_variable_candidate_rms"] = results_df["is_variable_candidate_rms"].fillna(False).astype(bool)
        results_df["is_variable_candidate_vdi"] = results_df["is_variable_candidate_vdi"].fillna(False).astype(bool)
        results_df["is_candidate_combined"] = (
            results_df["is_variable_candidate_rms"] | results_df["is_variable_candidate_vdi"]
        )
        results_df["detection_method"] = "-"
        results_df.loc[results_df["is_variable_candidate_rms"], "detection_method"] = "RMS"
        results_df.loc[results_df["is_variable_candidate_vdi"], "detection_method"] = "VDI"
        results_df.loc[
            results_df["is_variable_candidate_rms"] & results_df["is_variable_candidate_vdi"],
            "detection_method",
        ] = "RMS+VDI"

        st.subheader("Hockey stick (RMS)")
        work = results_df.copy()
        work["mag"] = pd.to_numeric(work["mag"], errors="coerce")
        work["rms_pct"] = pd.to_numeric(work["rms_pct"], errors="coerce")
        work["expected_rms_pct"] = pd.to_numeric(work["expected_rms_pct"], errors="coerce")
        work["variability_score"] = pd.to_numeric(work["variability_score"], errors="coerce")
        work["vsx_known_variable"] = work["vsx_known_variable"].fillna(False).astype(bool)
        work["gaia_dr3_variable_catalog"] = work["gaia_dr3_variable_catalog"].fillna(False).astype(bool)

        # Per-star edge filter (annulus-aware) - avoid false candidates near chip border
        masterstar_fits = platesolve_dir / "MASTERSTAR.fits"
        edge_ok = _edge_ok_from_masterstar(masterstar_fits, work, cfg_dict)
        work["edge_ok"] = edge_ok.reindex(work.index).fillna(False).astype(bool)

        comp_mask = (~work["vsx_known_variable"]) & (work["is_variable_candidate_rms"] == False)
        cand_mask = (work["is_candidate_combined"] == True) & (~work["vsx_known_variable"]) & (work["edge_ok"] == True)
        vsx_mask = work["vsx_known_variable"]
        gaia_mask = work["gaia_dr3_variable_catalog"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=work.loc[comp_mask, "mag"],
                y=work.loc[comp_mask, "rms_pct"],
                mode="markers",
                name="Stable (COMP)",
                marker=dict(color="#2ecc71", size=6, opacity=0.5),
                hovertemplate="cid=%{customdata}<br>mag=%{x:.3f}<br>rms=%{y:.3f}%<extra></extra>",
                customdata=work.loc[comp_mask, "catalog_id"],
            )
        )
        # expected curve (sorted by mag)
        curve = work[["mag", "expected_rms_pct"]].dropna().sort_values("mag")
        if not curve.empty:
            fig.add_trace(
                go.Scatter(
                    x=curve["mag"],
                    y=curve["expected_rms_pct"],
                    mode="lines",
                    name="Expected noise",
                    line=dict(color="#888888", width=2),
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=work.loc[cand_mask, "mag"],
                y=work.loc[cand_mask, "rms_pct"],
                mode="markers",
                name="Candidates",
                marker=dict(color="#e74c3c", size=9, opacity=0.9),
                hovertemplate="cid=%{customdata}<br>mag=%{x:.3f}<br>rms=%{y:.3f}%<br>score=%{text:.2f}<extra></extra>",
                customdata=work.loc[cand_mask, "catalog_id"],
                text=work.loc[cand_mask, "variability_score"],
            )
        )
        vsx_data = work.loc[vsx_mask].copy()
        if not vsx_data.empty:
            if "vsx_name" not in vsx_data.columns:
                vsx_data["vsx_name"] = np.nan
            if "vsx_type" not in vsx_data.columns:
                vsx_data["vsx_type"] = ""
            vsx_data["vsx_name_display"] = vsx_data["vsx_name"].fillna(
                vsx_data["catalog_id"].astype(str)
            )
            fig.add_trace(
                go.Scatter(
                    x=vsx_data["mag"],
                    y=vsx_data["rms_pct"],
                    mode="markers",
                    name="Known VSX",
                    marker=dict(color="#f39c12", symbol="x", size=10, opacity=0.9),
                    customdata=vsx_data[["vsx_name_display", "vsx_type", "catalog_id"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Type: %{customdata[1]}<br>"
                        "mag: %{x:.2f}<br>"
                        "rms: %{y:.2f}%<br>"
                        "ID: %{customdata[2]}<br>"
                        "<extra>Known VSX</extra>"
                    ),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=work.loc[gaia_mask, "mag"],
                y=work.loc[gaia_mask, "rms_pct"],
                mode="markers",
                name="Gaia variable",
                marker=dict(color="#3498db", symbol="square", size=9, opacity=0.9),
                customdata=work.loc[gaia_mask, "catalog_id"],
            )
        )

        fig.update_yaxes(type="log", title="rms_pct (%)")
        fig.update_xaxes(title="mag (Gaia G)")
        fig.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

        st.subheader("Results")
        n_all = int(len(work))
        n_rms_candidates = int((work["is_variable_candidate_rms"] & (~work["vsx_known_variable"])).sum())
        n_vdi_candidates = int((work["is_variable_candidate_vdi"] & (~work["vsx_known_variable"])).sum())
        n_combined = int(cand_mask.sum())
        n_vsx = int(vsx_mask.sum())
        n_gaia = int(gaia_mask.sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Stars analyzed",
            f"{n_all}",
            help="Total stars analyzed by the variability detector (RMS + VDI).",
        )
        m2.metric(
            "RMS candidates",
            f"{n_rms_candidates}",
            help=(
                "Stars flagged by the RMS hockey-stick test, excluding known VSX. "
                "The combined RMS/VDI export count is the 'Combined' metric."
            ),
        )
        m3.metric(
            "VDI candidates",
            f"{n_vdi_candidates}",
            help="Stars with high VDI score (variability detection index - LC shape analysis). Excludes known VSX stars.",
        )
        m4.metric(
            "Combined",
            f"{n_combined}",
            help="Candidates meeting RMS or VDI criteria, excluding known VSX and off-chip edge (edge_ok). Exported to variability_candidates.csv.",
        )
        st.caption(
            f"Known VSX: {n_vsx} - detected as variable but already in catalog | "
            f"Gaia variable: {n_gaia} - flagged in Gaia DR3 variable catalog"
        )

        # Keep only edge-safe candidates in the candidate table
        cand = work.loc[cand_mask].copy()
        _bullets = st.session_state.setdefault("var_catalog_bullets", {})
        _cur_cids = {str(x).strip() for x in cand["catalog_id"].astype(str).tolist() if str(x).strip()}
        for _old in set(_bullets.keys()) - _cur_cids:
            _bullets.pop(str(_old), None)
        st.session_state["var_catalog_bullets"] = _bullets

        st.session_state["var_candidates"] = [str(x).strip() for x in cand["catalog_id"].astype(str).tolist() if str(x).strip()]
        cand["Vizier"] = [
            _vizier_link(float(pd.to_numeric(r.get("ra_deg"), errors="coerce")), float(pd.to_numeric(r.get("dec_deg"), errors="coerce")))
            for _, r in cand.iterrows()
        ]
        bullets_map: dict[str, str] = st.session_state.setdefault("var_catalog_bullets", {})
        disk_katalogy = load_katalogy_map_from_disk(draft_dir, str(obs_group))
        bullets_map = _merge_katalogy_maps(bullets_map, disk_katalogy)
        st.session_state["var_catalog_bullets"] = bullets_map
        cand["katalogy"] = cand["catalog_id"].astype(str).map(
            lambda cid: _katalogy_display(bullets_map.get(str(cid), "-"))
        )

        show_cols = [
            "catalog_id",
            "mag",
            "bp_rp",
            "rms_pct",
            "smoothness_ratio",
            "vdi_score",
            "vdi_z_score",
            "detection_method",
            "variability_score",
            "katalogy",
            "zone",
            "Vizier",
        ]
        candidates_df = cand[show_cols].copy()
        # Formatting
        if "smoothness_ratio" in candidates_df.columns:
            candidates_df["smoothness_ratio"] = pd.to_numeric(candidates_df["smoothness_ratio"], errors="coerce").round(2)
        if "vdi_score" in candidates_df.columns:
            candidates_df["vdi_score"] = pd.to_numeric(candidates_df["vdi_score"], errors="coerce").round(3)
        if "vdi_z_score" in candidates_df.columns:
            candidates_df["vdi_z_score"] = pd.to_numeric(candidates_df["vdi_z_score"], errors="coerce").round(2)

        sel_export = set(str(x) for x in (st.session_state.get("selected_for_export") or []))
        candidates_df["export"] = candidates_df["catalog_id"].astype(str).isin(sel_export)

        st.markdown(
            "**Candidates** - check `export` to add to `variable_targets.csv`; "
            "pick a candidate in the list and click **Open crossmatch**."
        )
        disabled_cols = [c for c in candidates_df.columns if c != "export"]
        edited_cand = st.data_editor(
            candidates_df,
            column_config={
                "export": st.column_config.CheckboxColumn("export", default=False, help="Add to variable_targets.csv"),
                "katalogy": st.column_config.TextColumn("Catalogs", width="large"),
            },
            disabled=disabled_cols,
            width="stretch",
            hide_index=True,
            key="var_candidates_editor",
        )
        if "export" in edited_cand.columns:
            _em = edited_cand["export"].fillna(False).astype(bool)
            st.session_state["selected_for_export"] = edited_cand.loc[_em, "catalog_id"].astype(str).tolist()
        else:
            st.session_state["selected_for_export"] = []

        if st.toggle("Debug bullets_map", value=False):
            st.write(st.session_state.get("var_catalog_bullets", {}))

        # -- Auto crossmatch --------------------------
        candidates = st.session_state.get("var_candidates", [])
        bullets_map = st.session_state.get("var_catalog_bullets", {})

        # Fallback: if var_candidates missing, try deriving from var_results (best-effort).
        if not candidates:
            vr = st.session_state.get("var_results")
            try:
                import pandas as pd  # noqa: PLC0415

                df0 = None
                if isinstance(vr, dict):
                    df0 = vr.get("rms_df") if isinstance(vr.get("rms_df"), pd.DataFrame) else None
                if isinstance(df0, pd.DataFrame) and (not df0.empty) and "catalog_id" in df0.columns:
                    st.session_state["var_candidates"] = [str(x) for x in df0["catalog_id"].astype(str).tolist()]
                    candidates = st.session_state.get("var_candidates", [])
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("[VARIABILITY] TESS auto-branch guard (non-critical): %s", exc)

        missing = [
            c for c in candidates if str(c) not in {str(k) for k in bullets_map}
        ]

        if missing and "crossmatch_auto_done" not in st.session_state:
            pb = st.progress(0, text="Running catalog crossmatch...")
            status = st.empty()

            for i, cid in enumerate(missing):
                status.caption(f"Crossmatch {i+1}/{len(missing)}: {str(cid)[:18]}...")
                row = _get_candidate_row(
                    st.session_state.get("var_results"),
                    cid,
                    draft_dir=draft_dir,
                    platesolve_dir=platesolve_dir,
                )
                if row:
                    try:
                        cr = check_candidate_in_catalogs(
                            ra=float(row["ra"]),
                            dec=float(row["dec"]),
                            mag=row.get("mag"),
                            radius_arcsec=10.0,
                            vsx_local_db_path=_vsx_local_cm,
                        )
                        b = cr.catalog_summary_bullets()
                        bullets_map[cid] = "\n".join(b) if b else "-"
                        xr = st.session_state.setdefault("var_crossmatch_results", {})
                        xr[cid] = cr
                    except Exception as exc:  # noqa: BLE001
                        bullets_map[cid] = f"Error: {exc}"
                else:
                    bullets_map[cid] = "-"

                pb.progress((i + 1) / len(missing))

            pb.empty()
            status.empty()
            st.session_state["var_catalog_bullets"] = bullets_map
            st.session_state["crossmatch_auto_done"] = True
            st.rerun()

        st.caption(
            f"DEBUG: crossmatch_auto_done={st.session_state.get('crossmatch_auto_done')} | "
            f"var_photometry_dir={st.session_state.get('var_photometry_dir')} | "
            f"tess_enabled={getattr(cfg, 'tess_enabled', False)}"
        )

        # -- Auto TESS trigger ------------------------
        # crossmatch_auto_done is set True, then st.rerun(), so this block runs on the next cycle.
        if st.session_state.get("crossmatch_auto_done"):
            tess_results = st.session_state.get("tess_results", {})
            photometry_dir = st.session_state.get("var_photometry_dir")
            if not photometry_dir:
                st.warning(
                    "! Auto-TESS: var_photometry_dir is not set - open Variability with a valid draft."
                )
                logging.getLogger("pipeline").warning("[TESS] auto-TESS preskoceny - var_photometry_dir chyba")

            if not bool(getattr(cfg, "tess_enabled", False)):
                if not st.session_state.get("tess_auto_done"):
                    logging.getLogger("pipeline").info(
                        "[TESS] preskocene - tess_enabled=False (Variability auto vetva)"
                    )
                    st.session_state["tess_auto_done"] = True
                    st.rerun()
            else:
                _memory_cids = [
                    str(x).strip() for x in cand["catalog_id"].astype(str).tolist() if str(x).strip()
                ]
                _cid_rows = tess_catalog_ids_for_auto_run(
                    draft_dir, str(obs_group), _memory_cids
                )
                logging.getLogger("pipeline").info(
                    "[TESS] Auto-run eligible: %d candidates (from variability_candidates.csv)",
                    len(_cid_rows),
                )
                if _memory_cids and not _cid_rows:
                    logging.getLogger("pipeline").info(
                        "[TESS] auto-TESS preskoceny - ziadny kandidat v variability_candidates.csv "
                        "(alebo CSV chyba)"
                    )
                _done_tess = {str(k) for k in (tess_results or {})}
                _tess_photo = Path(str(photometry_dir)) if photometry_dir else None

                def _tess_result_json_on_disk(scid: str) -> bool:
                    if _tess_photo is None:
                        return False
                    return (_tess_photo / "_tess" / str(scid) / "result.json").is_file()

                to_tess = [
                    cid
                    for cid in _cid_rows
                    if cid not in _done_tess
                    and _should_trigger_tess(bullets_map.get(cid, "-"))
                    and photometry_dir
                    and not _tess_result_json_on_disk(cid)
                ]

                if not to_tess:
                    if photometry_dir:
                        if not _cid_rows:
                            logging.getLogger("pipeline").info(
                                "[TESS] auto-TESS preskoceny - ziadny kandidat bez katalogoveho "
                                "match v variability_candidates.csv"
                            )
                        else:
                            logging.getLogger("pipeline").info(
                                "[TESS] auto-TESS - vsetci opravneni kandidati uz spracovani "
                                "(%d v CSV, %d hotovych)",
                                len(_cid_rows),
                                len(_done_tess),
                            )
                    st.session_state["tess_auto_done"] = True
                else:
                    logging.getLogger("pipeline").info(
                        f"[TESS] auto-TESS start - {len(to_tess)} kandidatov: {to_tess}"
                    )
                    st.session_state.setdefault("tess_auto_done", False)

                    # Process exactly one candidate per rerun.
                    cid = to_tess[0]
                    remaining = to_tess[1:]

                    _need_tess = [
                        c
                        for c in _cid_rows
                        if _should_trigger_tess(bullets_map.get(c, "-"))
                        and not _tess_result_json_on_disk(c)
                    ]
                    total_tess = len(_need_tess)
                    done = len([c for c in _need_tess if str(c) in _done_tess])
                    st.info(
                        f"[telescope] TESS analysis: {done}/{total_tess} done - processing {str(cid)[:16]}..."
                    )

                    row = _get_candidate_row(
                        st.session_state.get("var_results"),
                        cid,
                        draft_dir=draft_dir,
                        platesolve_dir=platesolve_dir,
                    )
                    if row:
                        cr = st.session_state.get("var_crossmatch_results", {}).get(cid)
                        period_hint = None
                        if cr is not None and hasattr(cr, "best_period"):
                            try:
                                period_hint = cr.best_period()
                            except Exception:  # noqa: BLE001
                                period_hint = None

                        with st.spinner(f"TESS: {str(cid)[:16]}..."):
                            try:
                                tres = run_tess_analysis(
                                    catalog_id=str(cid),
                                    ra=float(row["ra"]),
                                    dec=float(row["dec"]),
                                    mag=row.get("mag"),
                                    photometry_dir=st.session_state.get("var_photometry_dir"),
                                    period_hint=period_hint,
                                    cfg=cfg,
                                )
                                tess_results[cid] = tres
                                logging.getLogger("pipeline").info(
                                    f"[TESS] {cid} - sektory: {tres.total_sectors_found}, "
                                    f"period: {tres.period_consensus}, error: {tres.error_global}"
                                )
                            except Exception as exc:  # noqa: BLE001
                                # Store marker so auto loop doesn't retry this cid.
                                tess_results[cid] = None
                                st.warning(f"TESS failed for {str(cid)[:16]}: {exc}")
                    else:
                        tess_results[cid] = None

                    st.session_state["tess_results"] = tess_results

                    if remaining:
                        st.rerun()
                    else:
                        st.session_state["tess_auto_done"] = True
                        st.rerun()

        coord_by_cid: dict[str, dict[str, float | None]] = {}
        for _, r in cand.iterrows():
            cid = str(r.get("catalog_id", ""))
            if not cid:
                continue
            coord_by_cid[cid] = {
                "ra": float(pd.to_numeric(r.get("ra_deg"), errors="coerce")),
                "dec": float(pd.to_numeric(r.get("dec_deg"), errors="coerce")),
                "mag": float(pd.to_numeric(r.get("mag"), errors="coerce")),
            }
        st.session_state["var_cm_coord_by_cid"] = coord_by_cid

        def _cm_format(cid: str) -> str:
            if not cid:
                return "(no candidates)"
            sub = candidates_df[candidates_df["catalog_id"].astype(str) == str(cid)]
            if sub.empty:
                return str(cid)
            r = sub.iloc[0]
            return (
                f"{cid} . mag={float(pd.to_numeric(r.get('mag'), errors='coerce')):.2f} . "
                f"rms={float(pd.to_numeric(r.get('rms_pct'), errors='coerce')):.1f}%"
            )

        cm_cids = [str(x) for x in candidates_df["catalog_id"].astype(str).tolist() if str(x)]
        if cm_cids:
            cx1, cx2 = st.columns([3, 1])
            with cx1:
                # Bez on_change: pri zmene widgetu na inej zalozke (napr. VSX checkbox v MASTERSTAR QA)
                # Streamlit rerun sposobil spustanie callbacku a otvaranie tohto dialogu omylom.
                st.selectbox(
                    "Select candidate for catalog crossmatch (modal):",
                    options=cm_cids,
                    format_func=_cm_format,
                    key="var_cm_pick_id",
                )
            with cx2:
                st.caption("")  # align button vertically
                if st.button("Open crossmatch", key="var_cm_open_btn", type="secondary"):
                    cid0 = str(st.session_state.get("var_cm_pick_id", cm_cids[0]) or cm_cids[0])
                    cr0 = coord_by_cid.get(cid0, {})
                    st.session_state["var_cm_cid"] = cid0
                    st.session_state["var_cm_ra"] = cr0.get("ra")
                    st.session_state["var_cm_dec"] = cr0.get("dec")
                    st.session_state["var_cm_mag"] = cr0.get("mag")
                    st.session_state["var_cm_open_requested"] = True
        else:
            st.caption("No candidates for crossmatch.")

        # Dialog len po explicitnom kliknuti 'Otvorit crossmatch' (nie pri zmene selectboxu / rerune z ineho tabu).
        # Jeden dialog na rerun: inak StreamlitDuplicateElementId (dva volania st.dialog v tom istom behu).
        if st.session_state.pop("var_cm_open_requested", False):
            _variability_crossmatch_dialog()

        colA, colB = st.columns(2)
        with colA:
            if st.button("[inbox] Export candidates CSV", key="var_export"):
                out_csv = draft_dir / "platesolve" / str(obs_group) / "variability_candidates.csv"
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                candidates_df.drop(columns=["export"], errors="ignore").to_csv(out_csv, index=False)
                st.success(f"Saved: {out_csv}")
        with colB:
            st.caption("")

        if st.button("+ Add selected to variable_targets.csv", key="var_add_to_var2"):
            selected_ids = list(st.session_state.get("selected_for_export") or [])
            if not selected_ids:
                st.warning("No stars selected (export column).")
            else:
                vt_path = platesolve_dir / "variable_targets.csv"
                if vt_path.exists():
                    vt_df = read_vyvar_csv(vt_path, low_memory=False)
                else:
                    vt_df = pd.DataFrame()

                if "catalog_id" in vt_df.columns:
                    try:
                        from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

                        existing_ids = set(
                            vt_df["catalog_id"].map(normalize_gaia_source_id).astype(str).str.strip().tolist()
                        )
                        existing_ids.discard("")
                    except Exception:  # noqa: BLE001
                        existing_ids = set(vt_df["catalog_id"].astype(str).str.strip().tolist())
                        existing_ids.discard("")
                else:
                    existing_ids = set()

                n_added = 0
                new_rows: list[dict[str, Any]] = []
                for cid in selected_ids:
                    try:
                        from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

                        cid_key = str(normalize_gaia_source_id(cid)).strip()
                    except Exception:  # noqa: BLE001
                        cid_key = str(cid).strip()
                    if not cid_key:
                        continue
                    if cid_key in existing_ids:
                        continue
                    sub = results_df[results_df["catalog_id"].astype(str).map(lambda x: str(x).strip()) == cid_key]
                    if sub.empty:
                        continue
                    row = sub.iloc[0]
                    new_rows.append(
                        {
                            "catalog_id": cid_key,
                            "name": str(cid_key),
                            "vsx_name": row.get("vsx_name", ""),
                            "vsx_type": row.get("vsx_type", "CAND"),
                            "ra_deg": row.get("ra_deg", ""),
                            "dec_deg": row.get("dec_deg", ""),
                            "x": row.get("x", ""),
                            "y": row.get("y", ""),
                            "mag": row.get("mag", ""),
                            "zone": row.get("zone", "linear"),
                            "priority": 2,
                            "notes": (
                                f"VarDetect: RMS={float(pd.to_numeric(row.get('rms_pct'), errors='coerce')):.1f}% "
                                f"smooth={float(pd.to_numeric(row.get('smoothness_ratio'), errors='coerce')):.2f} "
                                f"method={row.get('detection_method','-')}"
                            ),
                            "gaia_match_source": "variability_detection",
                        }
                    )
                    n_added += 1

                if new_rows:
                    new_df = pd.DataFrame(new_rows)
                    vt_df = pd.concat([vt_df, new_df], ignore_index=True)
                    try:
                        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

                        if "catalog_id" in vt_df.columns:
                            vt_df = vt_df.copy()
                            vt_df["catalog_id"] = normalize_gaia_source_id_series(vt_df["catalog_id"])
                    except Exception as exc:  # noqa: BLE001
                        from except_fix_counters import get_except_fix_counters

                        get_except_fix_counters().variability_gaia_id_norm_skip += 1
                        LOGGER.error(
                            "[VARIABILITY] Gaia ID normalization skipped before to_csv; "
                            "IDs written UNNORMALIZED (float-rounding hazard): %s",
                            exc,
                        )
                    vt_df.to_csv(vt_path, index=False)
                    st.success(
                        f"[OK] Added {n_added} stars to variable_targets.csv\n"
                        f"Run Aperture Photometry for full calibration."
                    )
                else:
                    st.info("All selected stars are already in variable_targets.csv")

        # ---- Candidate pick (shared for map + light curve) ----
        candidate_options = {
            f"{row.get('vsx_name', str(row.catalog_id)) or str(row.catalog_id)} "
            f"(mag={float(pd.to_numeric(row.mag, errors='coerce')):.2f}, "
            f"rms={float(pd.to_numeric(row.rms_pct, errors='coerce')):.1f}%, "
            f"{row.get('detection_method','-')})": str(row.catalog_id)
            for _, row in candidates_df.iterrows()
        }
        selected_cid = ""
        selected_label_lc = ""
        if candidate_options:
            selected_label_lc = st.selectbox(
                "Select candidate:",
                options=list(candidate_options.keys()),
                key="var_lc_select2",
            )
            selected_cid = str(candidate_options.get(selected_label_lc, ""))
        elif not candidates_df.empty and "catalog_id" in candidates_df.columns:
            selected_cid = str(candidates_df["catalog_id"].iloc[0])

        st.subheader("[chart] Candidate light curve")
        if selected_cid:

            @st.cache_data(ttl=300, show_spinner=False)
            def load_candidate_lc(per_frame_dir_s: str, catalog_id: str, flux_col_in: str = "dao_flux") -> pd.DataFrame:
                # Lokalny import: @st.cache_data na vnorenej funkcii neviaze spolahlivo closure na import z modulu.
                from gaia_catalog_id import normalize_gaia_source_id as _norm_cid  # noqa: PLC0415

                records: list[dict[str, Any]] = []
                for csv in sorted(Path(per_frame_dir_s).glob("proc_*.csv")):
                    try:
                        df = read_vyvar_csv(
                            csv,
                            usecols=["catalog_id", flux_col_in, "bjd_tdb_mid", "airmass"],
                            low_memory=False,
                        )
                    except Exception:  # noqa: BLE001
                        continue
                    _want = _norm_cid(catalog_id)
                    if not _want:
                        continue
                    df["_cid"] = df["catalog_id"].map(_norm_cid)
                    row = df[df["_cid"] == _want]
                    if row.empty:
                        continue
                    r0 = row.iloc[0]
                    flux = float(pd.to_numeric(r0.get(flux_col_in), errors="coerce"))
                    if not (np.isfinite(flux) and flux > 0):
                        continue
                    records.append(
                        {
                            "bjd": float(pd.to_numeric(r0.get("bjd_tdb_mid"), errors="coerce")),
                            "mag_inst": -2.5 * float(np.log10(flux)),
                            "airmass": float(pd.to_numeric(r0.get("airmass"), errors="coerce")),
                        }
                    )
                if not records:
                    return pd.DataFrame()
                return pd.DataFrame(records).sort_values("bjd").reset_index(drop=True)

            lc_df = load_candidate_lc(str(per_frame_dir), str(selected_cid), flux_source)
            if len(lc_df) > 0:
                fig2 = go.Figure()
                bjd_rel = lc_df["bjd"] - lc_df["bjd"].iloc[0]
                fig2.add_trace(
                    go.Scatter(
                        x=bjd_rel,
                        y=lc_df["mag_inst"],
                        mode="markers",
                        marker=dict(size=5, color="steelblue", opacity=0.8),
                        name="mag_inst",
                        hovertemplate="BJD+%{x:.4f}<br>mag=%{y:.3f}<extra></extra>",
                    )
                )
                fig2.update_layout(
                    title=f"Raw light curve - {selected_label_lc or str(selected_cid)}",
                    xaxis_title="BJD - BJD0",
                    yaxis_title="mag_inst (uncalibrated)",
                    yaxis_autorange="reversed",
                    height=350,
                    margin=dict(l=50, r=20, t=50, b=40),
                    hovermode="closest",
                )
                st.plotly_chart(fig2, width="stretch")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("N frames", f"{len(lc_df)}")
                c2.metric("RMS", f"{lc_df.mag_inst.std():.3f} mag")
                c3.metric("Amplitude", f"{(lc_df.mag_inst.max() - lc_df.mag_inst.min()):.3f} mag")
                c4.metric("Median mag", f"{lc_df.mag_inst.median():.3f}")
            else:
                st.warning("Light curve not available for this star.")
        else:
            st.info("No candidates available to show a light curve.")

        # ---- Field map ----
        st.subheader("[map] Star field (marked candidates)")
        try:
            _field_map_df = results_df[
                results_df["catalog_id"].astype(str) == str(selected_cid)
            ]
            if _field_map_df.empty:
                st.warning("No field map data available for this target.")
                cx, cy = float("nan"), float("nan")
                selected_label_map = str(selected_cid)
            else:
                selected_row = _field_map_df.iloc[0]
                cx = float(pd.to_numeric(selected_row.get("x"), errors="coerce"))
                cy = float(pd.to_numeric(selected_row.get("y"), errors="coerce"))
                _sel_name = str(selected_row.get("vsx_name", "") or "").strip()
                selected_label_map = _sel_name if _sel_name else str(selected_cid)
        except Exception:  # noqa: BLE001
            cx, cy = float("nan"), float("nan")
            selected_label_map = str(selected_cid)
        masterstar_fits = platesolve_dir / "MASTERSTAR.fits"
        if masterstar_fits.exists():
            cand_rows = work.loc[cand_mask].copy()
            cand_rows["x"] = pd.to_numeric(cand_rows.get("x"), errors="coerce")
            cand_rows["y"] = pd.to_numeric(cand_rows.get("y"), errors="coerce")
            cand_rows = cand_rows[np.isfinite(cand_rows["x"]) & np.isfinite(cand_rows["y"])].copy()
            cand_xy_label: list[tuple[float, float, str]] = []
            for _, rr in cand_rows.iterrows():
                lab = str(rr.get("vsx_name", "") or rr.get("catalog_id", ""))
                if not lab:
                    lab = str(rr.get("catalog_id", ""))
                cand_xy_label.append((float(rr["x"]), float(rr["y"]), lab))

            vsx_xy_list: list[tuple[float, float, str]] = []
            vsx_draw = work.loc[vsx_mask].copy()
            if not vsx_draw.empty:
                vsx_draw["x"] = pd.to_numeric(vsx_draw.get("x"), errors="coerce")
                vsx_draw["y"] = pd.to_numeric(vsx_draw.get("y"), errors="coerce")
                vsx_draw = vsx_draw[np.isfinite(vsx_draw["x"]) & np.isfinite(vsx_draw["y"])].copy()
                for _, vr in vsx_draw.iterrows():
                    vnm = str(vr.get("vsx_name", "") or "").strip()
                    if not vnm:
                        vnm = str(vr.get("catalog_id", ""))
                    vsx_xy_list.append((float(vr["x"]), float(vr["y"]), vnm))

            cand_tup = tuple(cand_xy_label)
            vsx_tup = tuple(vsx_xy_list) if vsx_xy_list else None

            png_bytes = _render_field_image_with_candidates(
                str(masterstar_fits),
                cand_tup,
                vsx_xy_label=vsx_tup,
                selected_xy=(float(cx), float(cy)) if (np.isfinite(cx) and np.isfinite(cy)) else None,
                selected_label=str(selected_label_map),
            )
            if png_bytes:
                st.image(png_bytes, width="stretch")
                st.caption("[red] Candidates (new)   [orange] Known VSX   [yellow] Selected candidate")
            else:
                st.info("Could not render field from MASTERSTAR.fits (missing astropy/matplotlib?).")
        else:
            st.info("Field not available (missing `MASTERSTAR.fits`).")

