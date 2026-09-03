"""Moved from pipeline.py (CONSOLIDATE-01E1). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from astropy.io import fits
from typing import Any, Callable
import math
import pandas as pd
from config import AppConfig
from database import VyvarDatabase
from utils import iter_fits_paths_recursive as _iter_fits_recursive

from pipeline import (
    LOGGER,
    _archive_raw_to_calibrated_light,
    _path_segments_forbidden_for_masterstar_physical_source,
    _sort_masterstar_paths_by_fwhm,
    _vyvar_open_database,
    draft_median_pointing_icrs_deg,
    resolve_masterstar_input_root,
    sync_obs_files_drift_arcmin_for_draft,
)

def _quality_inspection_dao_metrics(fp: Path) -> dict[str, Any]:
    """Fast DAOStarFinder + moment FWHM on brightest sources; sky median; star count."""
    import numpy as np

    out0: dict[str, Any] = {
        "fwhm_mean": None,
        "sky_background": None,
        "star_count": 0,
        "inspection_jd": None,
    }
    fp = Path(fp)
    if not fp.is_file():
        return {**out0, "error": "missing_file"}
    try:
        with fits.open(fp, memmap=True) as hdul:
            hdr = hdul[0].header
            data = np.asarray(hdul[0].data, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        return {**out0, "error": str(exc)}
    from pipeline_calibrate import _quality_inspection_dao_metrics_array  # noqa: PLC0415
    return _quality_inspection_dao_metrics_array(data, hdr)


def _estimate_fov_deg_from_fits_path(fp: Path) -> float | None:
    p = Path(fp)
    if not p.is_file():
        return None
    try:
        with fits.open(p, memmap=False) as hdul:
            from pipeline_calibrate import _estimate_fov_deg_from_header  # noqa: PLC0415
            return _estimate_fov_deg_from_header(hdul[0].header)
    except Exception:  # noqa: BLE001
        return None


def _obs_fwhm_basename_map_from_db(db: VyvarDatabase, draft_id: int) -> dict[str, float]:
    """Map ``basename.casefold()`` -> FWHM from ``manifest files[]`` for draft lights (last row wins per name)."""
    out: dict[str, float] = {}
    for row in db.fetch_draft_light_rows_for_quality(int(draft_id)):
        try:
            fv = row.get("FWHM")
            if fv is None:
                continue
            v = float(fv)
            if not math.isfinite(v) or v <= 0.5 or v >= 80.0:
                continue
            bn = Path(str(row.get("FILE_PATH") or "")).name.casefold()
            if bn:
                out[bn] = float(v)
                if bn.startswith("proc_"):
                    out.setdefault(bn[5:], float(v))
                else:
                    out.setdefault(f"proc_{bn}", float(v))
        except (TypeError, ValueError):
            continue
    return out


def _resolve_light_fits_for_quality_inspection(archive: Path, raw_fp: Path | str) -> Path | None:
    """Prefer calibrated counterpart; else existing raw path under archive."""
    m = _archive_raw_to_calibrated_light(archive, raw_fp)
    if m is not None:
        return m[0]
    p = Path(raw_fp)
    if p.is_file():
        return p
    ap = Path(archive).expanduser()
    p2 = ap / p
    if p2.is_file():
        return p2
    return None

def run_quality_analysis(
    *,
    db: VyvarDatabase,
    draft_id: int,
    archive_path: Path | str,
    progress_cb: Callable[[int, int, str], None] | None = None,
    roundness_reject_above: float | None = None,
) -> dict[str, Any]:
    """Per-draft light: DAO metrics -> ``manifest files[]``; FWHM x1.5 and optional DAO roundness auto-reject.

    When Streamlit is available, updates ``st.session_state`` keys ``fwhm_threshold``, ``center_ra``,
    ``center_de`` from computed medians (same as RAM QC path).

    The Quality Dashboard assigns ``frame_index`` 1...N in the same order as
    :meth:`VyvarDatabase.fetch_draft_light_rows_for_quality` (stable table <-> plot alignment).

    After metrics, :func:`sync_obs_files_drift_arcmin_for_draft` fills ``DRIFT`` / ``DRIFT_DRA`` / ``DRIFT_DDE``.
    """
    import numpy as np

    _rn = 1.25 if roundness_reject_above is None else float(roundness_reject_above)
    rlim_active = math.isfinite(_rn) and _rn > 0.0

    ap = Path(archive_path).expanduser()
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    n = len(rows)
    rid_to_scan: dict[int, int] = {}
    for _r in rows:
        try:
            rid_to_scan[int(_r["ID"])] = int(_r.get("ID_SCANNING") or 0)
        except Exception:  # noqa: BLE001
            continue
    fwhm_by_id: dict[int, float] = {}
    roundness_by_id: dict[int, float] = {}
    errors: list[str] = []
    fov_sample_deg: float | None = None

    for i, row in enumerate(rows, start=1):
        rid = int(row["ID"])
        raw = Path(str(row.get("FILE_PATH") or ""))
        tgt = _resolve_light_fits_for_quality_inspection(ap, raw)
        if tgt is None:
            errors.append(f"missing {raw.name}")
            db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"Skip missing {raw.name}")
            continue
        m = _quality_inspection_dao_metrics(tgt)
        if fov_sample_deg is None and not m.get("error"):
            fov_sample_deg = _estimate_fov_deg_from_fits_path(tgt)
        if m.get("error"):
            errors.append(f"{tgt.name}: {m['error']}")
        _rm = m.get("roundness_mean")
        _rm_db = float(_rm) if _rm is not None and math.isfinite(float(_rm)) and float(_rm) >= 0.0 else None
        _el = m.get("elongation_mean")
        _el_db = float(_el) if _el is not None and math.isfinite(float(_el)) and float(_el) > 0.0 else None
        db.update_obs_file_quality_by_id(int(draft_id), rid,
            fwhm=m.get("fwhm_mean"),
            sky_level=m.get("sky_background"),
            star_count=int(m.get("star_count") or 0),
            rejected_auto=0,
            inspection_jd=m.get("inspection_jd"),
            ra_deg=m.get("ra_deg"),
            de_deg=m.get("de_deg"),
            exptime_sec=m.get("exposure_sec"),
            roundness_mean=_rm_db,
            elongation_mean=_el_db,
        )
        fv = m.get("fwhm_mean")
        if fv is not None and math.isfinite(float(fv)):
            fwhm_by_id[rid] = float(fv)
        if _rm_db is not None:
            roundness_by_id[rid] = float(_rm_db)
        if progress_cb is not None:
            progress_cb(i, n, f"Quality {tgt.name}")

    vals = [v for v in fwhm_by_id.values() if math.isfinite(v) and v > 0]
    med: float | None
    if vals:
        med = float(np.median(np.asarray(vals, dtype=np.float64)))
        if not math.isfinite(med) or med <= 0:
            med = None
    else:
        med = None

    thr = med * 1.5 if med is not None else None
    light_rows2 = db.fetch_draft_light_rows_for_quality(int(draft_id))
    auto_n = 0
    for row in light_rows2:
        rid = int(row["ID"])
        rej = 0
        if med is not None and thr is not None:
            fv = fwhm_by_id.get(rid)
            if fv is not None and math.isfinite(float(fv)) and float(fv) > thr:
                rej = 1
        if rlim_active:
            rv = roundness_by_id.get(rid)
            if rv is not None and math.isfinite(float(rv)) and float(rv) > _rn:
                rej = 1
        if rej:
            auto_n += 1
        db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=rej)

    med_ra, med_de = draft_median_pointing_icrs_deg(db, int(draft_id))
    sync_obs_files_drift_arcmin_for_draft(db, int(draft_id), ref_ra_deg=med_ra, ref_de_deg=med_de)
    _dl_suggest = 5.0
    if fov_sample_deg is not None and math.isfinite(float(fov_sample_deg)) and float(fov_sample_deg) > 0:
        _dl_suggest = max(0.5, min(180.0, 0.1 * float(fov_sample_deg) * 60.0))
    result: dict[str, Any] = {
        "draft_id": int(draft_id),
        "n_lights": n,
        "n_successful_fwhm": int(len(fwhm_by_id)),
        "median_fwhm": med,
        "median_ra_deg": med_ra,
        "median_de_deg": med_de,
        "auto_rejected": int(auto_n),
        "errors": errors,
        "suggested_drift_limit_arcmin": float(_dl_suggest),
    }
    try:
        import streamlit as st

        _upd: dict[str, Any] = {}
        if med is not None and math.isfinite(float(med)) and float(med) > 0:
            _upd["fwhm_threshold"] = float(med)
        if med_ra is not None and math.isfinite(float(med_ra)):
            _upd["center_ra"] = float(med_ra)
            _upd["cur_draft_ra"] = float(med_ra)
        if med_de is not None and math.isfinite(float(med_de)):
            _upd["center_de"] = float(med_de)
            _upd["cur_draft_de"] = float(med_de)
        _upd["drift_limit_arcmin"] = float(_dl_suggest)
        st.session_state.update(_upd)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)
    return result

def list_best_processed_light_paths_for_masterstar(
    archive_path: Path | str | None,
    *,
    setup_name: str | None = None,
    draft_id: int | None = None,
    app_config: AppConfig | None = None,
    take_n: int = 5,
) -> list[Path]:
    """Najlepsie (najnizsi FWHM) FITS pod ``processed/lights`` - na UI tabulku a vyber pre MASTERSTAR."""
    if archive_path is None:
        return []
    ap = Path(archive_path).expanduser()
    if ap.name.casefold() == "non_calibrated":
        ap = ap.parent
    if not ap.is_dir():
        return []
    root = resolve_masterstar_input_root(
        ap,
        setup_name=setup_name,
        app_config=app_config,
        draft_id=draft_id,
    )
    if root is None or not root.exists():
        return []
    from draft_provenance import is_pre_calibrated_draft

    pre_cal = is_pre_calibrated_draft(ap, draft_id=draft_id)
    _forbid_kw = {"pre_calibrated": pre_cal}
    files = [
        fp
        for fp in _iter_fits_recursive(root)
        if not _path_segments_forbidden_for_masterstar_physical_source(fp, **_forbid_kw)
    ]
    if not files:
        return []
    tn = max(2, min(5, int(take_n)))
    _fb: dict[str, float] = {}
    if draft_id is not None:
        _dbc = _vyvar_open_database(app_config or AppConfig())
        if _dbc is not None:
            try:
                _fb = _obs_fwhm_basename_map_from_db(_dbc, int(draft_id))
            except Exception:  # noqa: BLE001
                # EXC-0305: T4 -- DB conn.close failure after FWHM map fetch is ignored once ranked paths are computed. (EXCEPT-BULK 2026-07-08)
                _fb = {}
            finally:
                try:
                    _dbc.conn.close()
                except Exception:  # noqa: BLE001
                    pass
    ranked = _sort_masterstar_paths_by_fwhm(files, fwhm_by_basename=_fb or None)
    return ranked[:tn]

def resolve_masterstars_metadata_csv(platesolve_dir: Path | str) -> Path | None:
    """Return masterstars metadata CSV in a platesolve setup dir (full_match preferred)."""
    ps = Path(platesolve_dir)
    for name in ("masterstars_full_match.csv", "masterstars.csv"):
        p = ps / name
        if p.is_file():
            return p
    return None

def preprocess_sky_summary_from_df(df: pd.DataFrame) -> dict[str, Any]:
    """Extract sky-surface guard counters stored on a preprocess result dataframe."""
    raw = getattr(df, "attrs", {}).get("preprocess_sky_summary") or {}
    return {
        "sky_surface_skip_count": int(raw.get("sky_surface_skip_count") or 0),
        "sky_surface_force_reapply": bool(raw.get("sky_surface_force_reapply")),
    }
