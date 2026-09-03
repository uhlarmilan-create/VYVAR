"""Moved from pipeline.py (CONSOLIDATE-01E5). Facade re-exports these names.

AstroPipeline stays in pipeline.py (C-C). Spawn-worker globals, initializer,
and `_calibrate_batch_process_one` live here so `global` binds one namespace.
"""
from __future__ import annotations

import contextlib
import itertools
import logging
import math
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from config import AppConfig
from database import VyvarDatabase
from calibration import (
    CALIBRATION_LIBRARY_NATIVE_BINNING,
    filter_light_paths_for_calibration_db,
    get_processed_master,
)
from cal_diag import (
    CalDiagGateResult,
    CalDiagSession,
    apply_cal_diag_headers,
    convention_to_dark_mode,
    dark_np_for_cal_diag,
    gate_result_for_frame,
    is_obs_group_aborted,
    passthrough_cal_diag_headers,
    run_cal_diag_pregate,
    write_cal_diag_json,
)
from cal_stage import _header_has_vy_skysf
from fits_meta import (
    _safe_filter_token,
    _summarize_lights_binning_from_headers,
    _valid_bayerpat_from_header,
    extract_fits_metadata,
    fits_metadata_from_primary_header,
    log_lights_binning_from_headers_preflight,
    observation_group_key_from_metadata,
)
from fits_suffixes import FITS_SUFFIXES_LOWER
from infolog import log_event, log_exception
from plain_stats import plain_mean_med_std
from utils import (
    DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    fits_binning_xy_from_header,
    fits_header_has_celestial_wcs,
    iter_fits_paths_recursive as _iter_fits_recursive,
)
from vyvar_alignment_frame import _as_fits_float32_image
from vyvar_platesolver import pointing_hint_from_header as _pointing_hint_from_header

# Same named logger as pipeline.LOGGER (logging.getLogger singleton). Avoids
# pipeline -> pipeline_calibrate -> pipeline at module load (spawn children
# import this module first).
LOGGER = logging.getLogger("pipeline")

def _vyvar_calibrate_multiprocessing_enabled() -> bool:
    """Parallel calibration uses ``spawn`` workers (errors are easy to miss). Set ``VYVAR_CALIBRATE_MP=1`` to enable."""
    v = os.environ.get("VYVAR_CALIBRATE_MP", "").strip().lower()
    return v in ("1", "true", "yes")


def _cfg_calibration_library_native_binning(cfg: Any) -> int | None:
    """Config ``calibration_library_native_binning``: ``None`` = read ``XBINNING`` from each master FITS."""
    raw = cfg.calibration_library_native_binning
    if raw is None:
        return None
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return int(CALIBRATION_LIBRARY_NATIVE_BINNING)


def _obs_group_key_from_light_path(src: Path) -> str:
    with fits.open(src, memmap=False) as hdul:
        return observation_group_key_from_metadata(
            fits_metadata_from_primary_header(hdul[0].header)
        )


def _light_binning_from_path(src: Path) -> int:
    with fits.open(src, memmap=False) as hdul:
        bx, _ = fits_binning_xy_from_header(hdul[0].header)
    return int(bx)


def _archive_root_from_lights_root(lights_root: Path) -> Path | None:
    p = Path(lights_root).resolve()
    if p.name.casefold() != "lights":
        return None
    parent = p.parent
    if parent.name.casefold() in ("raw", "non_calibrated"):
        return parent.parent
    return None


def _resolve_dark_path_for_light(
    *,
    src: Path,
    obs_group_key: str,
    master_dark_path: Path | None,
    master_dark_by_obs_key: dict[str, str | Path | None] | None,
) -> Path | None:
    md_use = master_dark_path
    if master_dark_by_obs_key:
        _alt = master_dark_by_obs_key.get(obs_group_key)
        if _alt is not None and str(_alt).strip() != "":
            _pa = Path(_alt)
            if _pa.is_file():
                md_use = _pa
    if md_use is not None and Path(md_use).is_file():
        return Path(md_use)
    return None


def _saturation_adu_for_cal_diag(
    hdr: fits.Header,
    *,
    db: VyvarDatabase | None,
    equipment_id: int | None,
) -> float | None:
    eq_sat: float | None = None
    if db is not None and equipment_id is not None:
        try:
            eq_sat = db.get_equipment_saturation_adu(int(equipment_id))
        except Exception:  # noqa: BLE001
            eq_sat = None
    lim, _src = _effective_saturation_limit(
        hdr,
        fallback_adu=None,
        equipment_saturate_adu=eq_sat,
    )
    return lim


def _cal_diag_session_from_export(blob: dict[str, Any] | None) -> CalDiagSession:
    session = CalDiagSession()
    if not blob:
        return session
    for k, raw in (blob.get("keys") or {}).items():
        try:
            session.gate_results[str(k)] = CalDiagGateResult(**raw)
        except TypeError:
            continue
    session.aborted_groups = set(str(x) for x in (blob.get("aborted_groups") or []))
    return session


def _cal_diag_export_for_workers(session: CalDiagSession) -> dict[str, Any] | None:
    if not session.gate_results and not session.aborted_groups:
        return None
    return session.json_export()


# ``_calibrate_one_light_*``: explicit ``None`` = read master FITS; omit param = library default (1x1).
_CALIB_MASTER_NB_UNSET = object()


def _log_calibration_io_preflight(
    *,
    calibrated_root: Path,
    master_dark_path: Path | None,
    masterflat_by_filter: dict[str, Path | None],
) -> None:
    """Log resolved master paths and best-effort write access to ``calibrated_root``."""
    try:
        calibrated_root.mkdir(parents=True, exist_ok=True)
        probe = calibrated_root / ".vyvar_write_probe"
        probe.write_text("ok", encoding="ascii")
        probe.unlink(missing_ok=True)
        log_event(f"Kalibracia - zapis OK: {calibrated_root.resolve()}")
    except OSError as exc:
        log_event(f"Kalibracia - CHYBA PRAVO ZAPISU (calibrated): {calibrated_root.resolve()} -> {exc}")

    if master_dark_path is not None:
        md = Path(master_dark_path)
        md_r = md.resolve()
        ok = md_r.is_file()
        log_event(f"MasterDark: {md_r} (exists={ok})")
        if not ok:
            log_event("MasterDark: subor neexistuje - dark sa neaplikuje (flat-only / copy-only podla nastavenia).")

    for fk, fp in sorted((masterflat_by_filter or {}).items(), key=lambda x: str(x[0])):
        if fp is None:
            log_event(f"MasterFlat[{fk!r}]: (ziadna cesta)")
            continue
        p = Path(fp)
        pr = p.resolve()
        ok = pr.is_file()
        log_event(f"MasterFlat[{fk!r}]: {pr} (exists={ok})")
        if not ok:
            log_event(f"MasterFlat[{fk!r}]: subor neexistuje - pre tento filter sa flat neaplikuje.")


def _pipeline_ui_error(msg: str) -> None:
    """Log always; mirror text to footer during a running job, then ``st.error``."""
    log_event(msg)
    try:
        import streamlit as st

        fs = st.session_state.get("vyvar_footer_state")
        if isinstance(fs, dict) and fs.get("running"):
            st.session_state["vyvar_footer_state"] = {**fs, "status_detail": str(msg)[:800]}
            _fn = st.session_state.get("vyvar_ui_rerender_footer")
            if callable(_fn):
                _fn()
        st.error(msg)
    except Exception:  # noqa: BLE001
        pass


def _match_and_crop_pair(a: "np.ndarray", b: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Crop 2D arrays to common smallest shape (top-left)."""
    import numpy as np

    a2 = np.asarray(a)
    b2 = np.asarray(b)
    if a2.ndim != 2 or b2.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got shapes {a2.shape} and {b2.shape}")
    h = min(a2.shape[0], b2.shape[0])
    w = min(a2.shape[1], b2.shape[1])
    return a2[:h, :w], b2[:h, :w]


def _iter_light_fits(lights_root: Path) -> list[Path]:
    """Collect lights FITS under lights_root (including nested subdirs, e.g. ``filter_exp_binning``)."""
    return _iter_fits_recursive(lights_root)


def _pick_light_for_metadata_diagnostic(paths: list[Path]) -> Path:
    """Representative light for infolog diagnostics: predominant header binning, not sort order."""
    if not paths:
        raise ValueError("empty paths")
    summary = _summarize_lights_binning_from_headers(paths)
    counts: dict[tuple[int, int], int] = summary.get("counts") or {}
    if not counts:
        return paths[0]
    best_key = max(counts.items(), key=lambda kv: kv[1])[0]
    samples: dict[tuple[int, int], Path] = summary.get("samples") or {}
    return samples.get(best_key) or paths[0]


def _filter_light_paths_maybe(
    files: list[Path],
    only_paths: Sequence[Path | str] | None,
) -> list[Path]:
    """If ``only_paths`` is set, keep only those members (resolved path match, casefold on Windows)."""
    if only_paths is None:
        return files

    def _norm(p: Path) -> str:
        try:
            return str(p.resolve()).casefold()
        except OSError:
            return str(p).casefold()

    by_norm: dict[str, Path] = {}
    for fp in files:
        by_norm.setdefault(_norm(fp), fp)

    ordered: list[Path] = []
    seen: set[str] = set()
    for x in only_paths:
        px = Path(x)
        k = _norm(px)
        if k in seen:
            continue
        hit = by_norm.get(k)
        if hit is not None:
            ordered.append(hit)
            seen.add(k)
            continue
        for fp in files:
            fk = _norm(fp)
            if fk in seen:
                continue
            try:
                if os.path.samefile(fp, px):
                    ordered.append(fp)
                    seen.add(fk)
                    break
            except OSError:
                continue
    return ordered


def _resolve_draft_light_raw_path(archive: Path, file_path: Path | str) -> Path | None:
    """Resolve raw light FITS on disk for in-RAM calibration (``manifest files[].FILE_PATH``)."""
    ap = Path(archive).expanduser()
    p = Path(str(file_path))
    if p.is_file():
        return p
    q = ap / p
    if q.is_file():
        return q
    name = p.name
    for sub in (ap / "non_calibrated" / "lights", ap / "Raw" / "lights"):
        cand = sub / name
        if cand.is_file():
            return cand
        # Setup subfolder (e.g. Raw/lights/NoFilter_60_2/BO_CVn_Light_001.fits)
        try:
            for hit in sub.rglob(name):
                if hit.is_file():
                    return hit
        except OSError:
            pass
    return None


def norm_fits_path_key(path: Path | str) -> str:
    """Normalized absolute path key for qc_metrics / allowlist joins (casefold)."""
    try:
        return str(Path(path).resolve()).casefold()
    except OSError:
        return str(path).casefold()


def _archive_preprocess_lights_root(
    ap: Path | str,
    *,
    app_config: AppConfig | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> Path:
    """Lights root for alignment (draft lights root or legacy ``detrended/lights``)."""
    from draft_provenance import draft_archive_root, resolve_draft_lights_root

    root = draft_archive_root(Path(ap).expanduser())
    lights = resolve_draft_lights_root(root, draft_id=draft_id, db=db)
    for cand in (lights, root / "detrended" / "lights"):
        if cand.is_dir() and _iter_fits_recursive(cand):
            return cand
    return lights


def _inspection_jd_from_header(hdr: fits.Header) -> float | None:
    """Julian Date (UTC) for scatter time axis."""
    for k in ("MJD-OBS", "MJD_OBS"):
        v = hdr.get(k)
        if v is not None:
            try:
                mjd = float(v)
                if math.isfinite(mjd):
                    return mjd + 2400000.5
            except (TypeError, ValueError):
                continue
    for k in ("JD-OBS", "JD_OBS", "JD"):
        v = hdr.get(k)
        if v is not None:
            try:
                jd = float(v)
                if math.isfinite(jd):
                    return jd
            except (TypeError, ValueError):
                continue
    date = hdr.get("DATE-OBS")
    if date:
        tim = hdr.get("TIME-OBS", hdr.get("TIME", "00:00:00"))
        d_s = str(date).strip()
        tim_s = str(tim).strip() if tim is not None else "00:00:00"
        if "T" in d_s:
            base = d_s.split("T", 1)[0]
            iso = f"{base}T{tim_s}"
        else:
            iso = f"{d_s}T{tim_s}"
        try:
            t = Time(iso, format="isot", scale="utc")
            return float(t.jd)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0283] DATE-OBS/Time parse failure leaves inspection_jd NULL in manifest files[] QC updates.: %s', exc)
            try:
                t = Time(d_s, scale="utc")
                return float(t.jd)
            except Exception:  # noqa: BLE001
                pass
    return None


def _exposure_sec_from_header(hdr: fits.Header) -> float | None:
    """Exposure duration in seconds from common FITS keywords."""
    for k in ("EXPTIME", "EXPOSURE", "EXPOSURE0", "EXP_TIME", "EXPTIM"):
        v = hdr.get(k)
        if v is None:
            continue
        try:
            sec = float(v)
            if math.isfinite(sec) and sec >= 0.0:
                return sec
        except (TypeError, ValueError):
            continue
    return None


def _dao_star_table_mean_roundness(tbl: Any) -> float | None:
    """Mean ``hypot(|roundness1|, |roundness2|)`` over detected sources (DAOStarFinder table)."""
    import numpy as np

    if tbl is None or len(tbl) == 0:
        return None
    try:
        if "roundness1" not in tbl.colnames or "roundness2" not in tbl.colnames:
            return None
        r1 = np.asarray(tbl["roundness1"], dtype=np.float64)
        r2 = np.asarray(tbl["roundness2"], dtype=np.float64)
        per = np.hypot(np.abs(r1), np.abs(r2))
        ok = np.isfinite(per)
        if not np.any(ok):
            return None
        return float(np.mean(per[ok]))
    except Exception:  # noqa: BLE001
        return None


def _quality_inspection_dao_metrics_array(
    data: "np.ndarray",
    hdr: fits.Header,
) -> dict[str, Any]:
    """Same as :func:`_quality_inspection_dao_metrics` but on an in-memory calibrated image.

    FWHM is the median moment-FWHM over many star-like detections (see
    :func:`_robust_frame_fwhm_median`); not a single detection and not
    segmentation islands (which track cosmics/hot pixels after CR removal).
    """
    import numpy as np

    out: dict[str, Any] = {
        "fwhm_mean": None,
        "sky_background": None,
        "star_count": 0,
        "inspection_jd": _inspection_jd_from_header(hdr),
        "exposure_sec": _exposure_sec_from_header(hdr),
        "roundness_mean": None,
        "elongation_mean": None,
    }
    _pra, _pde, _ = _pointing_hint_from_header(hdr)
    out["ra_deg"] = float(_pra) if _pra is not None and math.isfinite(float(_pra)) else None
    out["de_deg"] = float(_pde) if _pde is not None and math.isfinite(float(_pde)) else None
    try:
        arr = np.asarray(data, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        return {**out, "error": str(exc)}

    crop = _qc_center_crop_for_stars(arr)
    finite = np.isfinite(crop)
    if not np.any(finite):
        return out
    _, med, std = plain_mean_med_std(crop[finite], sigma=3.0, maxiters=5)
    std = float(std)
    if not math.isfinite(std) or std <= 0:
        return out
    out["sky_background"] = float(med) if np.isfinite(med) else None

    rob = _robust_frame_fwhm_median(
        arr,
        max_sources=120,
        min_keep=12,
        use_center_crop=True,
    )
    if rob.get("fwhm_px") is None:
        rob = _robust_frame_fwhm_median(
            arr,
            max_sources=120,
            min_keep=5,
            use_center_crop=True,
        )
    out["star_count"] = int(rob.get("n_stars_detected") or 0)
    out["elongation_mean"] = rob.get("elongation")
    if rob.get("fwhm_px") is not None:
        out["fwhm_mean"] = float(rob["fwhm_px"])
    # roundness_mean: keep best-effort from a quick DAO table when available
    try:
        from photutils.detection import DAOStarFinder

        img2 = np.asarray(crop - float(med), dtype=np.float32)
        img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.nanmedian(img2)) < 0:
            # Only for roundness helper; prefer tail test when near zero.
            pos = float(np.count_nonzero(img2 > (4.0 * std)))
            neg = float(np.count_nonzero(img2 < (-4.0 * std)))
            if neg > (pos * 2.0) and neg > 50:
                img2 = -img2
        fwhm_guess = float(max(3.0, min(12.0, _estimate_dao_fwhm_guess(img2, std))))
        daofind = DAOStarFinder(
            fwhm=fwhm_guess,
            threshold=5.0 * std,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = daofind(img2)
        out["roundness_mean"] = _dao_star_table_mean_roundness(tbl)
    except Exception:  # noqa: BLE001
        pass
    return out


def draft_median_pointing_icrs_deg(db: VyvarDatabase, draft_id: int) -> tuple[float | None, float | None]:
    """Median ``RA`` / ``DE`` from draft light rows (degrees ICRS), for preprocess hints when headers lack coords."""
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    ras: list[float] = []
    des: list[float] = []
    for row in rows:
        try:
            ra = row.get("RA")
            de_val = row.get("DE")
            if ra is None or de_val is None:
                continue
            raf = float(ra)
            dec_f = float(de_val)
            if math.isfinite(raf) and math.isfinite(dec_f):
                ras.append(raf)
                des.append(dec_f)
        except (TypeError, ValueError):
            continue
    if not ras:
        return None, None
    import statistics

    return float(statistics.median(ras)), float(statistics.median(des))


def _estimate_fov_deg_from_header(hdr: fits.Header) -> float | None:
    """Rough field diameter in degrees from WCS pixel scale x ``NAXIS*`` (fallback when CDELT missing)."""
    try:
        n1 = int(hdr.get("NAXIS1", 0) or 0)
        n2 = int(hdr.get("NAXIS2", 0) or 0)
        if n1 <= 0 or n2 <= 0:
            return None
        d1 = hdr.get("CDELT1")
        d2 = hdr.get("CDELT2")
        if d1 is not None and d2 is not None:
            a, b = abs(float(d1)), abs(float(d2))
            if math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0:
                return float(math.hypot(a * n1, b * n2))
        c11 = hdr.get("CD1_1")
        c12 = hdr.get("CD1_2")
        c21 = hdr.get("CD2_1")
        c22 = hdr.get("CD2_2")
        if None not in (c11, c12, c21, c22):
            a11, a12 = float(c11), float(c12)
            a21, a22 = float(c21), float(c22)
            if all(math.isfinite(x) for x in (a11, a12, a21, a22)):
                wx = abs(a11) * n1 + abs(a12) * n2
                hy = abs(a21) * n1 + abs(a22) * n2
                if wx > 0 and hy > 0:
                    return float(math.hypot(wx, hy))
    except (TypeError, ValueError):
        return None
    return None


def sync_obs_files_drift_arcmin_for_draft(
    db: VyvarDatabase,
    draft_id: int,
    *,
    ref_ra_deg: float | None,
    ref_de_deg: float | None,
) -> int:
    """Fill ``manifest files[].DRIFT`` (arcmin), ``DRIFT_DRA`` / ``DRIFT_DDE`` (deg plane offsets vs median). Clears when ref missing."""
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    n = 0
    if ref_ra_deg is None or ref_de_deg is None:
        for row in rows:
            db.update_obs_file_quality_by_id(int(draft_id), int(row["ID"]), clear_drift=True)
            n += 1
        return n
    try:
        rref0, dref0 = float(ref_ra_deg), float(ref_de_deg)
    except (TypeError, ValueError):
        for row in rows:
            db.update_obs_file_quality_by_id(int(draft_id), int(row["ID"]), clear_drift=True)
            n += 1
        return n
    if not (math.isfinite(rref0) and math.isfinite(dref0)):
        for row in rows:
            db.update_obs_file_quality_by_id(int(draft_id), int(row["ID"]), clear_drift=True)
            n += 1
        return n
    rref, dref = rref0, dref0
    for row in rows:
        rid = int(row["ID"])
        try:
            ra = row.get("RA")
            de = row.get("DE")
            if ra is None or de is None:
                db.update_obs_file_quality_by_id(int(draft_id), rid, clear_drift=True)
                n += 1
                continue
            raf, def_ = float(ra), float(de)
            if not (math.isfinite(raf) and math.isfinite(def_)):
                db.update_obs_file_quality_by_id(int(draft_id), rid, clear_drift=True)
                n += 1
                continue
            dra_deg = ((raf - rref) + 180.0) % 360.0 - 180.0
            dde_deg = def_ - dref
            dra_plane = dra_deg * math.cos(math.radians(dref))
            d_arc = math.hypot(dra_plane, dde_deg) * 60.0
            if not math.isfinite(d_arc):
                db.update_obs_file_quality_by_id(int(draft_id), rid, clear_drift=True)
            else:
                db.update_obs_file_quality_by_id(int(draft_id), 
                    rid,
                    drift_arcmin=float(d_arc),
                    drift_dra_deg=float(dra_plane),
                    drift_dde_deg=float(dde_deg),
                )
            n += 1
        except (TypeError, ValueError):
            db.update_obs_file_quality_by_id(int(draft_id), rid, clear_drift=True)
            n += 1
    return n


def _perf10_lookup_qc(
    perf10_qc_results: dict[str, dict[str, Any]],
    archive: Path,
    file_path: str,
) -> dict[str, Any] | None:
    """Match ``perf10_qc_results`` keyed by resolved raw FITS path."""
    raw_fp = _resolve_draft_light_raw_path(archive, file_path)
    if raw_fp is None:
        return None
    for key in (str(raw_fp.resolve()), str(raw_fp)):
        qc = perf10_qc_results.get(key)
        if isinstance(qc, dict) and qc:
            return qc
    return None


def apply_perf10_dao_qc_to_obs_files(
    *,
    db: VyvarDatabase,
    draft_id: int,
    archive_path: Path | str,
    perf10_qc_results: dict[str, dict[str, Any]],
    roundness_reject_above: float | None = 1.25,
) -> dict[str, Any]:
    """Write calibration-time DAO QC to ``manifest files[]`` (same columns as RAM QC step 5)."""
    import numpy as np

    ap = Path(archive_path).expanduser()
    _rn = 1.25 if roundness_reject_above is None else float(roundness_reject_above)
    rlim_active = math.isfinite(_rn) and _rn > 0.0

    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    fwhm_by_id: dict[int, float] = {}
    roundness_by_id: dict[int, float] = {}
    n_updated = 0
    errors: list[str] = []

    for row in rows:
        rid = int(row["ID"])
        qc = _perf10_lookup_qc(perf10_qc_results, ap, str(row.get("FILE_PATH") or ""))
        if qc is None or qc.get("error"):
            continue
        try:
            _rm = qc.get("roundness_mean")
            _rm_db = (
                float(_rm)
                if _rm is not None and math.isfinite(float(_rm)) and float(_rm) >= 0.0
                else None
            )
            _el = qc.get("elongation_mean")
            _el_db = (
                float(_el)
                if _el is not None and math.isfinite(float(_el)) and float(_el) > 0.0
                else None
            )
            db.update_obs_file_quality_by_id(int(draft_id), rid,
                fwhm=qc.get("fwhm_mean"),
                sky_level=qc.get("sky_background"),
                star_count=int(qc.get("star_count") or 0),
                rejected_auto=0,
                inspection_jd=qc.get("inspection_jd"),
                ra_deg=qc.get("ra_deg"),
                de_deg=qc.get("de_deg"),
                exptime_sec=qc.get("exposure_sec"),
                roundness_mean=_rm_db,
                elongation_mean=_el_db,
            )
            n_updated += 1
            fv = qc.get("fwhm_mean")
            if fv is not None and math.isfinite(float(fv)):
                fwhm_by_id[rid] = float(fv)
            if _rm_db is not None:
                roundness_by_id[rid] = float(_rm_db)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{row.get('FILE_PATH')}: {exc}")

    logging.info("[PERF-10] Updated manifest files[] QC for %d frames during calibration", n_updated)

    vals = [v for v in fwhm_by_id.values() if math.isfinite(v) and v > 0]
    med: float | None
    if vals:
        med = float(np.median(np.asarray(vals, dtype=np.float64)))
        if not math.isfinite(med) or med <= 0:
            med = None
    else:
        med = None

    thr = med * 1.5 if med is not None else None
    auto_n = 0
    for row in rows:
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
    db._try_refresh_draft_manifest(int(draft_id))
    _dl_suggest = 5.0
    return {
        "draft_id": int(draft_id),
        "n_lights": len(rows),
        "n_successful_fwhm": int(len(fwhm_by_id)),
        "median_fwhm": med,
        "median_ra_deg": med_ra,
        "median_de_deg": med_de,
        "auto_rejected": int(auto_n),
        "errors": errors,
        "perf10_n_updated": int(n_updated),
        "suggested_drift_limit_arcmin": float(_dl_suggest),
    }


def run_draft_ram_calibration_qc_to_obs_files(
    *,
    db: VyvarDatabase,
    draft_id: int,
    archive_path: Path | str,
    master_dark_path: Path | None,
    masterflat_by_filter: dict[str, Path | None],
    masterflat_by_obs_key: dict[str, str | Path | None] | None = None,
    master_dark_by_obs_key: dict[str, str | Path | None] | None = None,
    equipment_id: int | None = None,
    pipeline_config: AppConfig | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    roundness_reject_above: float | None = None,
) -> dict[str, Any]:
    """Calibrate each draft light **in RAM only**, DAO metrics -> ``manifest files[]``; FWHM x1.5 and optional roundness reject.

    No calibrated FITS are written. Uses the same master selection rules as :func:`calibrate_lights_to_calibrated`.
    After QC, :func:`sync_obs_files_drift_arcmin_for_draft` writes ``DRIFT`` / ``DRIFT_DRA`` / ``DRIFT_DDE``.
    """
    import numpy as np

    _rn = 1.25 if roundness_reject_above is None else float(roundness_reject_above)
    rlim_active = math.isfinite(_rn) and _rn > 0.0

    ap = Path(archive_path).expanduser()
    cfg = pipeline_config or AppConfig()
    db_cal = _db_for_calibration_tasks(None)

    mf_merged: dict[str, Path | None] = {}
    for k, v in (masterflat_by_filter or {}).items():
        mf_merged[str(k)] = None if v is None else Path(v)
    for k, v in (masterflat_by_obs_key or {}).items():
        mf_merged[str(k)] = None if v is None or str(v).strip() == "" else Path(v)

    md_pre: Any = None
    md_path_ok: Path | None = None
    if master_dark_path is not None and Path(master_dark_path).exists():
        md_path_ok = Path(master_dark_path)
        with fits.open(md_path_ok, memmap=False) as hdul:
            md_pre = np.array(hdul[0].data, dtype=np.float32, copy=True)

    dark_cache: dict[str, Any] = {}
    _native_b = _cfg_calibration_library_native_binning(cfg)

    flat_cache: dict[str, Any] = {}
    flat_median_scale: dict[str, float] = {}
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    n = len(rows)
    fwhm_by_id: dict[int, float] = {}
    roundness_by_id: dict[int, float] = {}
    errors: list[str] = []
    fov_sample_deg: float | None = None

    cal_diag_session = CalDiagSession()
    _pregate_paths: list[Path] = []
    for row in rows:
        _rf = _resolve_draft_light_raw_path(ap, str(row.get("FILE_PATH") or ""))
        if _rf is not None:
            _pregate_paths.append(_rf)
    if _pregate_paths:
        cal_diag_session = run_cal_diag_pregate(
            _pregate_paths,
            obs_group_key_from_path=_obs_group_key_from_light_path,
            resolve_dark_path=lambda fp, og, lb: _resolve_dark_path_for_light(
                src=fp,
                obs_group_key=og,
                master_dark_path=md_path_ok,
                master_dark_by_obs_key=master_dark_by_obs_key,
            ),
            light_binning_from_path=_light_binning_from_path,
            master_binning=_native_b,
            match_and_crop_pair=_match_and_crop_pair,
            saturation_for_light=lambda fp: _saturation_adu_for_cal_diag(
                fits.getheader(fp, 0),
                db=db_cal,
                equipment_id=equipment_id,
            ),
            ui_error=_pipeline_ui_error,
        )
    write_cal_diag_json(ap, cal_diag_session)

    for i, row in enumerate(rows, start=1):
        rid = int(row["ID"])
        raw_fp = _resolve_draft_light_raw_path(ap, str(row.get("FILE_PATH") or ""))
        if raw_fp is None:
            errors.append(f"missing raw {row.get('FILE_PATH')}")
            db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"Skip missing {Path(str(row.get('FILE_PATH') or '')).name}")
            continue

        try:
            with fits.open(raw_fp, memmap=False) as hdul:
                hdr0 = hdul[0].header
                _ok = observation_group_key_from_metadata(fits_metadata_from_primary_header(hdr0))
                _light_shape = (int(hdul[0].data.shape[0]), int(hdul[0].data.shape[1]))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{raw_fp.name}: header {exc}")
            db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"Header fail {raw_fp.name}")
            continue

        if is_obs_group_aborted(cal_diag_session, _ok):
            db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"CAL-DIAG skip {raw_fp.name}")
            continue

        md_use = md_path_ok
        md_np_use = None
        light_bx, _ = fits_binning_xy_from_header(hdr0)
        if master_dark_by_obs_key:
            _alt = master_dark_by_obs_key.get(_ok)
            if _alt is not None and str(_alt).strip() != "":
                _pa = Path(_alt)
                if _pa.is_file():
                    md_use = _pa
        gr = gate_result_for_frame(
            cal_diag_session,
            obs_group_key=_ok,
            dark_path=md_use,
            light_binning=light_bx,
        )
        if md_use is not None and md_use.is_file():
            if (
                md_pre is not None
                and md_path_ok is not None
                and md_use.resolve() == md_path_ok.resolve()
                and _native_b is not None
                and _native_b == light_bx
                and (gr is None or convention_to_dark_mode(gr.convention) == "sum")
            ):
                md_np_use = md_pre
            else:
                md_np_use = dark_np_for_cal_diag(
                    cal_diag_session,
                    master_binning=_native_b,
                    dark_path=md_use,
                    light_binning=light_bx,
                    light_shape=_light_shape,
                    light_filename=raw_fp.name,
                    gate_result=gr,
                )

        try:
            data, hdr, _ud, _uf = _calibrate_one_light_apply_masters_in_ram(
                src=raw_fp,
                master_dark_path=md_use,
                masterflat_by_filter=mf_merged,
                flat_cache=flat_cache,
                flat_median_scale=flat_median_scale,
                md_data_preload=md_np_use,
                db=db_cal,
                id_equipments=equipment_id,
                calibration_master_native_binning=_native_b,
                cal_diag_gate_result=gr,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{raw_fp.name}: {exc}")
            db.update_obs_file_quality_by_id(int(draft_id), rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"Cal fail {raw_fp.name}")
            continue

        m = _quality_inspection_dao_metrics_array(data, hdr)
        if fov_sample_deg is None:
            fov_sample_deg = _estimate_fov_deg_from_header(hdr)
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
            progress_cb(i, n, f"QC RAM {raw_fp.name}")

    vals = [v for v in fwhm_by_id.values() if math.isfinite(v) and v > 0]
    med: float | None
    if vals:
        med = float(np.median(np.asarray(vals, dtype=np.float64)))
        if not math.isfinite(med) or med <= 0:
            med = None
    else:
        med = None

    thr = med * 1.5 if med is not None else None
    light_rows_ram = db.fetch_draft_light_rows_for_quality(int(draft_id))
    auto_n = 0
    for row in light_rows_ram:
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
    _dl_suggest_ram = 5.0
    if fov_sample_deg is not None and math.isfinite(float(fov_sample_deg)) and float(fov_sample_deg) > 0:
        _dl_suggest_ram = max(0.5, min(180.0, 0.1 * float(fov_sample_deg) * 60.0))
    result = {
        "draft_id": int(draft_id),
        "n_lights": n,
        "n_successful_fwhm": int(len(fwhm_by_id)),
        "median_fwhm": med,
        "median_ra_deg": med_ra,
        "median_de_deg": med_de,
        "auto_rejected": int(auto_n),
        "errors": errors,
        "suggested_drift_limit_arcmin": float(_dl_suggest_ram),
    }
    try:
        rid_to_scan_local: dict[int, int] = locals().get("rid_to_scan", {})  # type: ignore[assignment]
        by_scan: dict[int, dict[str, Any]] = {}
        for rid, sid in rid_to_scan_local.items():
            if sid <= 0:
                continue
            rec = by_scan.setdefault(int(sid), {"n_rows": 0, "fwhm_vals": [], "round_vals": []})
            rec["n_rows"] = int(rec["n_rows"]) + 1
            fv = fwhm_by_id.get(rid)
            if fv is not None and math.isfinite(float(fv)):
                rec["fwhm_vals"].append(float(fv))
            rv = roundness_by_id.get(rid)
            if rv is not None and math.isfinite(float(rv)):
                rec["round_vals"].append(float(rv))
        if by_scan:
            result["by_scanning"] = {
                str(sid): {
                    "n_rows": int(v["n_rows"]),
                    "median_fwhm": (
                        float(np.median(np.asarray(v["fwhm_vals"], dtype=np.float64)))
                        if v["fwhm_vals"]
                        else None
                    ),
                    "median_roundness": (
                        float(np.median(np.asarray(v["round_vals"], dtype=np.float64)))
                        if v["round_vals"]
                        else None
                    ),
                }
                for sid, v in sorted(by_scan.items())
            }
    except Exception:  # noqa: BLE001
        pass
    try:
        import streamlit as st

        _upd2: dict[str, Any] = {}
        if med is not None and math.isfinite(float(med)) and float(med) > 0:
            _upd2["fwhm_threshold"] = float(med)
        if med_ra is not None and math.isfinite(float(med_ra)):
            _upd2["center_ra"] = float(med_ra)
            _upd2["cur_draft_ra"] = float(med_ra)
        if med_de is not None and math.isfinite(float(med_de)):
            _upd2["center_de"] = float(med_de)
            _upd2["cur_draft_de"] = float(med_de)
        _upd2["drift_limit_arcmin"] = float(_dl_suggest_ram)
        st.session_state.update(_upd2)
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)
    db._try_refresh_draft_manifest(int(draft_id))
    return result


def format_memory_bytes(n: float | int) -> str:
    """Human-readable binary size (KiB = 1024 B)."""
    try:
        x = float(n)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(x) or x <= 0:
        return "0 B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{n} B"


def _fits_primary_pixel_count(header: fits.Header) -> int:
    """Pixel count of primary image HDU (product of NAXIS*)."""
    try:
        naxis = int(header.get("NAXIS", 0) or 0)
    except (TypeError, ValueError):
        return 0
    if naxis < 1:
        return 0
    prod = 1
    for i in range(1, naxis + 1):
        try:
            prod *= int(header.get(f"NAXIS{i}", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return int(prod)


def _available_system_ram_bytes() -> tuple[int | None, str]:
    """Best-effort free/available RAM (``psutil`` if installed)."""
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available), "psutil"
    except Exception:  # noqa: BLE001
        return None, "unknown"


def estimate_memory_from_fits_headers(
    paths: list[Path],
    *,
    sample_headers: int = 48,
) -> dict[str, Any]:
    """Read only FITS primary headers; estimate float32 array size (no decompression of data)."""
    n = len(paths)
    pixels: list[int] = []
    step = max(1, n // max(1, min(sample_headers, n))) if n else 1
    for i in range(0, n, step):
        if len(pixels) >= sample_headers:
            break
        fp = paths[i]
        try:
            hdr = fits.getheader(fp, ext=0, ignore_missing_end=True, memmap=False)
            pixels.append(_fits_primary_pixel_count(hdr))
        except Exception:  # noqa: BLE001
            pixels.append(0)
    pixels = [p for p in pixels if p > 0]
    import numpy as np

    med = int(np.median(pixels)) if pixels else 0
    mx = int(np.max(pixels)) if pixels else 0
    bytes_med = med * 4
    bytes_max = mx * 4
    return {
        "n_files": n,
        "primary_pixels_median": med,
        "primary_pixels_max": mx,
        "bytes_float32_median_frame": bytes_med,
        "bytes_float32_max_frame": bytes_max,
    }


def estimate_archive_memory_profile(archive_path: str | Path) -> dict[str, Any]:
    """Rough RAM hints for QC analyze and optional platesolve ``ram_align_and_catalog`` (header-only scan).

    **QC analyze** is sequential: peak is a fewx the largest single frame (QC temps).

    **RAM handoff** (after detrending) holds ~one float32 copy per successfully aligned frame in memory
    before flush; add a margin for align-time temporaries (reference + source + aligned).
    """
    ap = Path(archive_path)
    avail, avail_src = _available_system_ram_bytes()
    out: dict[str, Any] = {
        "archive_path": str(ap),
        "available_ram_bytes": avail,
        "available_ram_human": format_memory_bytes(avail) if avail is not None else "nezname (nainstaluj ``psutil``)",
        "available_ram_source": avail_src,
        "qc_analyze": None,
        "platesolve_ram_handoff": None,
        "notes_sk": (
            "Odhad z hlaviciek FITS (NAXIS*), predpoklad prace vo float32. Skutocna spotreba zavisi od OS, "
            "Streamlit a dalsich kniznic. Po preprocess este pribudnu docasne polia pri detrendingu."
        ),
    }

    cal = ap / "calibrated" / "lights"
    try:
        from draft_provenance import resolve_draft_lights_root

        _lights_mem = resolve_draft_lights_root(ap)
        if _lights_mem.is_dir():
            cal = _lights_mem
    except Exception:  # noqa: BLE001
        pass
    if cal.is_dir():
        cfiles = _iter_light_fits(cal)
        st = estimate_memory_from_fits_headers(cfiles)
        # Working set: jeden snimok nacitany + QC (hruby faktor)
        qc_factor = 6.0
        peak_qc = int(float(st["bytes_float32_max_frame"]) * qc_factor)
        out["qc_analyze"] = {
            **st,
            "estimated_peak_bytes_sequential": peak_qc,
            "estimated_peak_human": format_memory_bytes(peak_qc),
            "explanation_sk": (
                f"Sekvencne spracovanie ~{st['n_files']} snimok; spicka RAM ~ {qc_factor:.0f}x najvacsi snimok "
                f"({format_memory_bytes(st['bytes_float32_max_frame'])} float32) pri analyze."
            ),
        }

    det = _archive_preprocess_lights_root(ap)
    if det.is_dir():
        dfiles = _iter_fits_recursive(det)
        st2 = estimate_memory_from_fits_headers(dfiles)
        n = int(st2["n_files"])
        med_b = int(st2["bytes_float32_median_frame"])
        max_b = int(st2["bytes_float32_max_frame"])
        # Buffer: jedna kopia zarovnaneho float32 na uspesny snimok (horny odhad = vsetky vstupy zarovnane)
        buffer_est = med_b * max(0, n)
        # Pocas astroalign: ref + src + aligned chvilu naraz
        align_spike = max_b * 3
        total_conservative = buffer_est + align_spike
        out["platesolve_ram_handoff"] = {
            **st2,
            "estimated_aligned_buffer_bytes": buffer_est,
            "estimated_aligned_buffer_human": format_memory_bytes(buffer_est),
            "estimated_align_spike_bytes": align_spike,
            "estimated_align_spike_human": format_memory_bytes(align_spike),
            "estimated_total_conservative_bytes": total_conservative,
            "estimated_total_conservative_human": format_memory_bytes(total_conservative),
            "explanation_sk": (
                f"Rezim 'zarovnanie + katalog v RAM': drzi ~{n} snimok x ~{format_memory_bytes(med_b)} "
                f"(median) + kratkodoba spicka pri zarovnani. Ak je to viac nez volna RAM, vypni RAM handoff."
            ),
        }

    if avail is not None and out.get("platesolve_ram_handoff"):
        tot = int(out["platesolve_ram_handoff"]["estimated_total_conservative_bytes"])
        out["platesolve_ram_handoff"]["estimate_below_available_ram"] = bool(tot <= avail)
        out["platesolve_ram_handoff"]["available_vs_estimated_ratio"] = float(avail) / float(tot) if tot > 0 else None

    if avail is not None and out.get("qc_analyze"):
        pq = int(out["qc_analyze"]["estimated_peak_bytes_sequential"])
        out["qc_analyze"]["estimate_below_available_ram"] = bool(pq <= avail)

    return out


def _log_calibration_metadata_diagnostic(filename: str, metadata: dict[str, Any]) -> None:
    log_event("--- DIAGNOSTIKA METADAT PRE KALIBRACIU ---")
    log_event(f"Subor: {filename}")
    log_event(f"FOCAL (z DB/FITS): {metadata.get('focal_length')} mm")
    log_event(f"PIXEL_SIZE (surovy): {metadata.get('pixel_size_raw')} um")
    _n1 = metadata.get("naxis1")
    _n2 = metadata.get("naxis2")
    if _n1 or _n2:
        log_event(f"NAXIS: {_n1}x{_n2}")
    _rbx = metadata.get("fits_xbinning_raw")
    _rby = metadata.get("fits_ybinning_raw")
    if _rbx is not None or _rby is not None:
        log_event(f"BINNING (FITS raw): XBINNING={_rbx!r} YBINNING={_rby!r}")
    log_event(f"BINNING (X/Y): {metadata.get('binning')}x{metadata.get('binning_y')}")
    log_event(f"EFEKTIVNY PIXEL (pre vypocet): {metadata.get('pixel_um')} um")
    log_event("------------------------------------------")


def _has_valid_wcs(header: fits.Header) -> bool:
    return fits_header_has_celestial_wcs(header)


def _saturate_limit_adu_from_header(hdr: fits.Header) -> float | None:
    """Return saturation / linearity ceiling in image units (ADU, e-, ...) if present in header."""
    import math

    for key in ("SATURATE", "MAXLIN", "ESATUR", "LINLIMIT", "MAXADU"):
        if key not in hdr:
            continue
        try:
            v = float(hdr[key])
            if math.isfinite(v) and v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _infer_sat_limit_from_bitpix(hdr: fits.Header) -> float | None:
    """Infer linearity ceiling from FITS integer layout (e.g. unsigned 16-bit -> 65535)."""
    import math

    try:
        bitpix = int(hdr.get("BITPIX", 0))
    except (TypeError, ValueError):
        return None
    bzero = float(hdr.get("BZERO", 0.0))
    bscale = float(hdr.get("BSCALE", 1.0))
    if not math.isfinite(bzero) or not math.isfinite(bscale) or bscale <= 0:
        return None
    if bitpix == 16:
        # Unsigned 16-bit (common): physical 0...65535 stored with BZERO=32768
        if abs(bzero - 32768.0) < 1.0 and abs(bscale - 1.0) < 1e-9:
            return 65535.0
        # Native signed 16-bit
        if abs(bzero) < 1e-6 and abs(bscale - 1.0) < 1e-9:
            return 32767.0
    return None


def _effective_saturation_limit(
    hdr: fits.Header,
    *,
    fallback_adu: float | None,
    equipment_saturate_adu: float | None = None,
) -> tuple[float, str]:
    """Resolve saturation ceiling: header keywords -> ``EQUIPMENTS`` / caller ``equipment_saturate_adu`` ->
    ``DATAMAX`` / ``MAXPIX`` -> BITPIX guess -> optional ``fallback_adu`` -> GAIN-DOMAIN-01
    container clip (INV-SAT-LIMIT; never None).
    """
    import math

    from pipeline import SAT_LIMIT_CONTAINER_CLIP_ADU  # noqa: PLC0415

    lim = _saturate_limit_adu_from_header(hdr)
    if lim is not None:
        return lim, "header_keyword"

    if equipment_saturate_adu is not None:
        fe = float(equipment_saturate_adu)
        if math.isfinite(fe) and fe > 0:
            return fe, "equipment_db"

    for dk in ("DATAMAX", "MAXPIX"):
        if dk not in hdr:
            continue
        try:
            v = float(hdr[dk])
            if math.isfinite(v) and v > 0:
                return v, f"header_{dk.lower()}"
        except (TypeError, ValueError):
            continue

    lim2 = _infer_sat_limit_from_bitpix(hdr)
    if lim2 is not None:
        return lim2, "bitpix"

    if fallback_adu is not None:
        fa = float(fallback_adu)
        if math.isfinite(fa) and fa > 0:
            return fa, "config_fallback"

    # INV-SAT-LIMIT: never return None. MASTERSTAR stacks are float (BITPIX -32),
    # EQUIPMENTS.SATURATE_ADU may be NULL (QHY294MM migration), headers often omit
    # SATURATE/MAXLIN. Silent None made peak>limit comparisons False for the whole catalog.
    logging.warning(
        "[INV-SAT-LIMIT] saturation clip unresolved (header/equipment/BITPIX none); "
        "using GAIN-DOMAIN-01 container clip %.0f ADU",
        SAT_LIMIT_CONTAINER_CLIP_ADU,
    )
    return float(SAT_LIMIT_CONTAINER_CLIP_ADU), "conservative_default_container_clip_65535"


def _vyvar_parallel_use_processes() -> bool:
    """CPU-heavy parallel steps default to subprocesses. Set ``VYVAR_PARALLEL_BACKEND=thread`` for threads."""
    v = (os.environ.get("VYVAR_PARALLEL_BACKEND") or "process").strip().lower()
    return v not in ("thread", "threads")


@contextlib.contextmanager
def _vyvar_parallel_pool(max_workers: int):
    if _vyvar_parallel_use_processes():
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            yield ex
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            yield ex


def _db_for_calibration_tasks(
    qc_opt: dict[str, Any] | None,
) -> VyvarDatabase | None:
    """Open DB once per worker / sequential pass when post-calibrate QC needs it."""
    q = qc_opt or {}
    p: str | None = None
    if q.get("enabled") and q.get("db_path"):
        p = str(q["db_path"])
    if not p:
        return None
    try:
        return VyvarDatabase(Path(p))
    except Exception:  # noqa: BLE001
        return None


def _qc_pack_from_config(
    cfg: AppConfig,
    *,
    draft_id: int | None,
    observation_id: str | None,
) -> dict[str, Any]:
    """Post-calibration QC limits + DB linkage (``manifest files[]`` update by raw light path)."""
    en = bool(cfg.qc_after_calibrate_enabled)
    _dao = float(cfg.qc_dao_detection_sigma)
    if not math.isfinite(_dao) or _dao <= 0:
        _dao = 5.0
    id_equipments: int | None = None
    if draft_id is not None and str(cfg.database_path).strip():
        try:
            _dbp = Path(cfg.database_path)
            if _dbp.is_file():
                _vdb = VyvarDatabase(_dbp)
                id_equipments = _vdb.get_draft_equipment_id(int(draft_id))
        except Exception:  # noqa: BLE001
            id_equipments = None
    return {
        "enabled": en,
        "max_hfr": float(cfg.qc_max_hfr),
        "max_hfr_fwhm_ratio": getattr(cfg, "qc_max_hfr_fwhm_ratio", None),
        "min_stars": int(cfg.qc_min_stars),
        "max_bg_rms": cfg.qc_max_background_rms,
        "dao_detection_sigma": _dao,
        "db_path": str(Path(cfg.database_path).resolve()) if en else None,
        "draft_id": draft_id,
        "observation_id": observation_id,
        "id_equipments": id_equipments,
        "dao_qc_in_calibrate": bool(cfg.dao_qc_in_calibrate),
    }


def _qc_center_crop_for_stars(data: "np.ndarray", max_side: int = 1000) -> "np.ndarray":
    """Central crop for star metrics when the frame is larger than ``max_side``."""
    import numpy as np

    a = np.asarray(data, dtype=np.float32)
    if a.ndim != 2:
        return a
    h, w = int(a.shape[0]), int(a.shape[1])
    if h <= max_side and w <= max_side:
        return a
    cy, cx = h // 2, w // 2
    hs, ws = max_side // 2, max_side // 2
    y0, y1 = max(0, cy - hs), min(h, cy + hs)
    x0, x1 = max(0, cx - ws), min(w, cx + ws)
    return np.asarray(a[y0:y1, x0:x1], dtype=np.float32)


def _half_flux_radius_in_cutout(cut: "np.ndarray", xc: float, yc: float) -> float:
    import numpy as np

    cut = np.asarray(cut, dtype=np.float64)
    h, w = cut.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2).ravel()
    pix = np.maximum(cut.ravel(), 0.0)
    order = np.argsort(r)
    r = r[order]
    pix = pix[order]
    tot = float(np.sum(pix))
    if tot <= 0 or not math.isfinite(tot):
        return float("nan")
    cum = np.cumsum(pix)
    idx = int(np.searchsorted(cum, 0.5 * tot))
    idx = min(max(idx, 0), cum.size - 1)
    return float(r[idx])


def _mean_hfr_bright_stars_dao(
    crop: "np.ndarray",
    *,
    max_stars: int = 50,
    dao_detection_sigma: float = 5.0,
) -> tuple[float | None, int, float | None]:
    """Median half-flux radius [px] on up to ``max_stars`` brightest DAO sources.

    Returns (HFR, n_detected, fwhm_guess_px).
    """
    import numpy as np
    from photutils.detection import DAOStarFinder

    img = np.asarray(crop, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return None, 0, None
    _, med, std = plain_mean_med_std(img[finite], sigma=3.0, maxiters=5)
    std = float(std)
    if not math.isfinite(std) or std <= 0:
        return None, 0, None
    img2 = np.asarray(img - float(med), dtype=np.float32)
    img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.nanmedian(img2)) < 0:
        img2 = -img2
    fwhm_guess = _estimate_dao_fwhm_guess(img2, std)
    sig0 = float(dao_detection_sigma) if math.isfinite(float(dao_detection_sigma)) and float(dao_detection_sigma) > 0 else 5.0
    daofind = DAOStarFinder(
        fwhm=float(fwhm_guess),
        threshold=sig0 * std,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    tbl = daofind(img2)
    if tbl is None or len(tbl) == 0:
        daofind = DAOStarFinder(
            fwhm=float(fwhm_guess),
            threshold=max(3.5, 0.7 * sig0) * std,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = daofind(img2)
    if tbl is None or len(tbl) == 0:
        return None, 0, float(fwhm_guess)
    n_det = int(len(tbl))
    tbl.sort("flux")
    tbl = tbl[::-1]
    h, w = img2.shape
    half = 10
    hfrs: list[float] = []
    for i in range(min(max_stars, len(tbl))):
        x0 = float(tbl["x_centroid"][i])
        y0 = float(tbl["y_centroid"][i])
        xi, yi = int(round(x0)), int(round(y0))
        y1, y2 = max(0, yi - half), min(h, yi + half + 1)
        x1, x2 = max(0, xi - half), min(w, xi + half + 1)
        sl = img2[y1:y2, x1:x2]
        if sl.shape[0] < 5 or sl.shape[1] < 5:
            continue
        hfr = _half_flux_radius_in_cutout(sl, x0 - x1, y0 - y1)
        if math.isfinite(hfr) and 0.2 < hfr < 50.0:
            hfrs.append(hfr)
    if not hfrs:
        return None, n_det, float(fwhm_guess)
    return float(np.nanmedian(np.asarray(hfrs, dtype=np.float64))), n_det, float(fwhm_guess)


def _post_calibration_qc_eval(
    data: "np.ndarray",
    *,
    limits: dict[str, Any],
    light_basename: str = "",
) -> dict[str, Any]:
    """Sky stats on full frame; star HFR/count on central crop if frame is large."""
    import numpy as np

    img = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        out = {
            "qc_passed": False,
            "hfr": None,
            "n_stars": 0,
            "sky_mean": None,
            "sky_median": None,
            "sky_rms": None,
            "reject_reasons": ["no finite pixels"],
        }
        if light_basename:
            log_event(f"Frame {light_basename} REJECTED (no finite pixels)")
        return out

    sky_mean, sky_med, sky_rms = plain_mean_med_std(img[finite], sigma=3.0, maxiters=5)
    sky_mean = float(sky_mean)
    sky_med = float(sky_med)
    sky_rms = float(sky_rms)

    crop = _qc_center_crop_for_stars(img, 1000)
    _ds = float(limits.get("dao_detection_sigma", 5.0))
    if not math.isfinite(_ds) or _ds <= 0:
        _ds = 5.0
    hfr_m, n_star, fwhm_guess = _mean_hfr_bright_stars_dao(crop, max_stars=50, dao_detection_sigma=_ds)

    from unit_resolver import resolve_hfr_limit_px

    class _QcHfrCfg:
        qc_max_hfr = float(limits.get("max_hfr", 5.0))
        qc_max_hfr_fwhm_ratio = limits.get("max_hfr_fwhm_ratio")

    max_h = resolve_hfr_limit_px(_QcHfrCfg(), fwhm_px=fwhm_guess)
    min_star = int(limits.get("min_stars", 10))
    max_rms = limits.get("max_bg_rms")

    reasons: list[str] = []
    ok = True
    if hfr_m is None or not math.isfinite(hfr_m):
        ok = False
        reasons.append("HFR unavailable")
    elif hfr_m > max_h:
        ok = False
        reasons.append(f"HFR: {hfr_m:.2f} > limit {max_h:.2f}")
    if n_star < min_star:
        ok = False
        reasons.append(f"stars: {n_star} < min {min_star}")
    if max_rms is not None and math.isfinite(float(max_rms)) and math.isfinite(sky_rms):
        if sky_rms > float(max_rms):
            ok = False
            reasons.append(f"background RMS: {sky_rms:.4g} > limit {float(max_rms):.4g}")

    if not ok and light_basename:
        if len(reasons) == 1 and reasons[0].startswith("HFR:"):
            log_event(f"Frame {light_basename} REJECTED ({reasons[0]})")
        else:
            log_event(f"Frame {light_basename} REJECTED ({'; '.join(reasons)})")

    return {
        "qc_passed": ok,
        "hfr": hfr_m,
        "n_stars": int(n_star),
        "sky_mean": sky_mean,
        "sky_median": sky_med,
        "sky_rms": sky_rms,
        "reject_reasons": reasons,
    }


def _strip_raw_linearity_header_keywords(hdr: fits.Header) -> None:
    """Remove FITS keys that describe the **raw** detector linearity range.

    After ``(light - dark) / flat`` the pixel scale is no longer raw ADU; keeping SATURATE/DATAMAX from the
    light frame makes viewers and automated limits disagree with the actual array values.
    """
    for key in (
        "SATURATE",
        "MAXLIN",
        "ESATUR",
        "LINLIMIT",
        "MAXADU",
        "DATAMAX",
        "MAXPIX",
    ):
        if key in hdr:
            try:
                del hdr[key]
            except KeyError:
                pass


def _vy_calib_status_numeric(flags: str) -> int:
    """Map ``VY_CFLAG`` to CALIB_STATUS-like 0/1/2 (reference: full / partial / raw)."""
    f = (flags or "").upper()
    if f == "DF":
        return 2
    if f == "D":
        return 1
    return 0


def _hdr_vy_cflag_str(hdr: fits.Header) -> str:
    raw = hdr.get("VY_CFLAG")
    if isinstance(raw, tuple):
        return str(raw[0]).strip().upper() or "P"
    if raw is None:
        return "P"
    return str(raw).strip().upper() or "P"


def _calibration_flags(
    *,
    used_dark: bool,
    used_flat: bool,
    passthrough: bool,
    flat_skipped_no_dark: bool = False,
) -> str:
    """Build ``VY_CFLAG`` (D=dark, F=flat, DF=full, FS=flat skipped without dark, P=passthrough/raw)."""
    if passthrough:
        return "P"
    if flat_skipped_no_dark:
        return "FS"
    out = ""
    if used_dark:
        out += "D"
    if used_flat:
        out += "F"
    return out or "P"


def _calibration_type_from_flags(flags: str) -> str:
    f = (flags or "").upper()
    if f == "P":
        return "PASSTHROUGH"
    if f == "DF":
        return "DARK+FLAT"
    if f == "D":
        return "DARK_ONLY"
    if f == "F":
        return "FLAT_ONLY"
    if f == "FS":
        return "RAW_FLAT_SKIPPED"
    return "PASSTHROUGH"


def _calibrate_one_light_apply_masters_in_ram(
    *,
    src: Path,
    master_dark_path: Path | None,
    masterflat_by_filter: dict[str, Path | None],
    flat_norm_floor: float = 0.15,
    flat_cache: dict[str, Any] | None = None,
    flat_median_scale: dict[str, float] | None = None,
    md_data_preload: Any = None,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    calibration_master_native_binning: int | None | object = _CALIB_MASTER_NB_UNSET,
    cal_diag_gate_result: CalDiagGateResult | None = None,
) -> tuple[Any, fits.Header, bool, bool]:
    """Apply dark/flat in RAM; return ``(data_float32, header, used_dark, used_flat)`` (no disk write).

    ``calibration_master_native_binning`` defaults to :data:`calibration.CALIBRATION_LIBRARY_NATIVE_BINNING`
    (CalibrationLibrary stores native masters; resample in RAM to match light ``XBINNING``).
    Pass ``None`` explicitly to read ``XBINNING`` from each master FITS (``calibration_library_native_binning: null`` in config).
    """
    import numpy as np

    if calibration_master_native_binning is _CALIB_MASTER_NB_UNSET:
        _mb_lib = int(CALIBRATION_LIBRARY_NATIVE_BINNING)
    elif calibration_master_native_binning is None:
        _mb_lib = None
    else:
        _mb_lib = max(1, int(calibration_master_native_binning))

    fc: dict[str, Any] = flat_cache if flat_cache is not None else {}
    fms: dict[str, float] = flat_median_scale if flat_median_scale is not None else {}

    with fits.open(src, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)

    light_bx, light_by = fits_binning_xy_from_header(hdr)
    hdr["VY_CLBX"] = (int(light_bx), "Light XBINNING used for master matching / resampling")
    hdr["VY_CLBY"] = (int(light_by), "Light YBINNING (diagnostic)")
    if _mb_lib is None:
        hdr["VY_MBNC"] = (
            -1,
            "VYVAR: native XBINNING read from each CalibrationLibrary master FITS (XBINNING)",
        )
    else:
        hdr["VY_MBNC"] = (
            int(_mb_lib),
            "VYVAR: assumed native XBINNING of CalibrationLibrary master before resample to light",
        )

    md_data: np.ndarray | None = md_data_preload
    if _mb_lib is None or (md_data is not None and md_data.shape != data.shape):
        md_data = None
    if md_data is not None and master_dark_path is not None and master_dark_path.exists():
        if _mb_lib != light_bx:
            md_data = None

    if md_data is None and master_dark_path is not None and master_dark_path.exists():
        _drm = "sum"
        if cal_diag_gate_result is not None:
            _drm = convention_to_dark_mode(cal_diag_gate_result.convention)
        pm = get_processed_master(
            master_dark_path,
            light_bx,
            kind="dark",
            master_binning=_mb_lib,
            light_shape=data.shape,
            light_filename=src.name,
            dark_resample_mode=_drm,
        )
        if pm.resampled:
            log_event(
                f"Calibration: library master native {pm.master_binning}x{pm.master_binning} -> "
                f"resampled (RAM) to {light_bx}x{light_bx} for Light [{src.name}]"
            )
        md_data = pm.data

    used_dark = False
    used_flat = False

    if md_data is not None:
        data2, md2 = _match_and_crop_pair(data, md_data)
        data = data2 - md2
        used_dark = True

    flt = _safe_filter_token(str(hdr.get("FILTER") or hdr.get("FILT") or "NoFilter"))
    _obs_k = observation_group_key_from_metadata(fits_metadata_from_primary_header(hdr))
    hdr["VY_OBSG"] = (_obs_k, "VYVAR observation group: FILTER|EXPTIME|XBINNING")
    mf_path = None
    if masterflat_by_filter:
        c_obs = masterflat_by_filter.get(_obs_k)
        if c_obs is not None:
            p_obs = Path(c_obs)
            if p_obs.is_file():
                mf_path = p_obs
    if mf_path is None and masterflat_by_filter:
        c_f = masterflat_by_filter.get(flt) or masterflat_by_filter.get("NoFilter")
        if c_f is not None:
            p_f = Path(c_f)
            if p_f.is_file():
                mf_path = p_f
    if mf_path is not None and mf_path.exists() and used_dark:
        key = f"{flt}|{mf_path!s}|lb{light_bx}"
        if key not in fc:
            pmf = get_processed_master(
                mf_path,
                light_bx,
                kind="flat",
                master_binning=_mb_lib,
                light_shape=data.shape,
                light_filename=src.name,
                db=db,
                id_equipments=id_equipments,
            )
            if pmf.resampled:
                log_event(
                    f"Calibration: library flat native {pmf.master_binning}x{pmf.master_binning} -> "
                    f"resampled (RAM) to {light_bx}x{light_bx} for Light [{src.name}]"
                )
            flat = pmf.data
            flat = np.where(np.isfinite(flat) & (flat > 0), flat, 1.0).astype(np.float32)
            if pmf.flat_normalized_at_calibrate:
                fm = pmf.flat_median_adu_before_norm
                if fm is None or not np.isfinite(fm) or fm <= 0:
                    fm = float(np.nanmedian(flat))
                    if not np.isfinite(fm) or fm <= 0:
                        fm = 1.0
            else:
                fm = float(np.nanmedian(flat))
                if not np.isfinite(fm) or fm <= 0:
                    fm = 1.0
                flat = (flat / fm).astype(np.float32)
            flat = np.maximum(flat, float(flat_norm_floor)).astype(np.float32)
            fc[key] = flat
            fms[key] = fm
        flat_arr = fc[key]
        data2, flat2 = _match_and_crop_pair(data, flat_arr)
        data = data2 / flat2
        used_flat = True
    flat_skipped_no_dark = bool(mf_path is not None and mf_path.exists() and not used_dark)
    if flat_skipped_no_dark:
        log_event("Flat skipped because no Dark/Bias was subtracted to avoid over-correction.")

    if used_dark or used_flat:
        if used_flat or not np.all(np.isfinite(data)):
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    hdr["VYVARCAL"] = (True, "VYVAR calibrated output")
    hdr["VY_DARK"] = (bool(used_dark), "MasterDark applied")
    hdr["VY_FLAT"] = (bool(used_flat), "MasterFlat applied")
    _flags = _calibration_flags(
        used_dark=bool(used_dark),
        used_flat=bool(used_flat),
        passthrough=False,
        flat_skipped_no_dark=flat_skipped_no_dark,
    )
    hdr["VY_CFLAG"] = (
        _flags,
        "VYVAR flags: DF=full, D=partial, FS=flat skipped (no dark), P=passthrough/raw",
    )
    hdr["VY_CALIB"] = (_calibration_type_from_flags(_flags), "Calibration mode")
    hdr["VY_CALST"] = (
        int(_vy_calib_status_numeric(_flags)),
        "VYVAR CALIB_STATUS: 2=full DF, 1=partial D, 0=raw (incl. FS passthrough)",
    )
    if _flags == "FS":
        hdr["VY_WARN"] = (
            True,
            "Flat available but not applied: subtract MasterDark first (over-correction risk).",
        )
    if master_dark_path is not None:
        hdr["VY_MDP"] = (str(master_dark_path.name)[:68], "MasterDark filename")
    if mf_path is not None:
        hdr["VY_MFP"] = (str(Path(mf_path).name)[:68], "MasterFlat filename")
    if used_flat:
        try:
            key_m = f"{flt}|{mf_path!s}|lb{light_bx}"
            hdr["VY_FLATM"] = (
                float(fms[key_m]),
                "Median ADU of master flat at target resample before normalize-to-1 (legacy: before pipeline division)",
            )
            hdr["VY_FLFL"] = (
                float(flat_norm_floor),
                "Min normalized flat before division (limits local gain from flat only)",
            )
        except KeyError:
            pass

    if used_dark or used_flat:
        _strip_raw_linearity_header_keywords(hdr)
        hdr.add_history(
            "VYVAR: cleared raw SATURATE/DATAMAX/... "
            "(pixels = (light-dark)/median-norm flat; not raw ADU)."
        )

    apply_cal_diag_headers(hdr, cal_diag_gate_result)

    return data, hdr, used_dark, used_flat


def _sync_obs_calibration_state_with_retry(
    db: VyvarDatabase | None,
    *,
    raw_light_path: Path,
    draft_id: int | None,
    observation_id: str | None,
    is_calibrated: int,
    calib_type: str,
    calib_flags: str,
    stats: dict[str, Any] | None = None,
) -> bool:
    """Sync manifest files[] cal state after successful calibrate; retry once then ERROR + count."""
    if db is None:
        return True
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            db.update_obs_file_calibration_state_by_raw_light_path(
                raw_light_path,
                draft_id=draft_id,
                observation_id=observation_id,
                is_calibrated=is_calibrated,
                calib_type=calib_type,
                calib_flags=calib_flags,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0426] [CAL-DIAG] `_quality_inspection_dao_metrics_array` in calibrate path logs WARNING and o...: %s', exc)
            last_exc = exc
            if attempt == 0:
                LOGGER.warning(
                    "CAL-DIAG: manifest files[] cal sync retry after failure (%s): %s",
                    raw_light_path.name,
                    exc,
                )
    if last_exc is not None:
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().calibrate_db_sync_fail += 1
        LOGGER.error(
            "CAL-DIAG: manifest files[] cal sync failed after calibrate (%s): %s",
            raw_light_path.name,
            last_exc,
        )
        if stats is not None:
            stats["cal_db_sync_failures"] = int(stats.get("cal_db_sync_failures", 0)) + 1
            errs = stats.setdefault("cal_db_sync_errors", [])
            if isinstance(errs, list):
                errs.append({"file": raw_light_path.name, "error": str(last_exc)})
    return False


def _calibrate_one_light_disk(
    *,
    src: Path,
    dst: Path,
    master_dark_path: Path | None,
    masterflat_by_filter: dict[str, Path | None],
    flat_norm_floor: float = 0.15,
    flat_cache: dict[str, Any] | None = None,
    flat_median_scale: dict[str, float] | None = None,
    md_data_preload: Any = None,
    db: VyvarDatabase | None = None,
    qc_pack: dict[str, Any] | None = None,
    calibration_master_native_binning: int | None | object = _CALIB_MASTER_NB_UNSET,
    cal_diag_gate_result: CalDiagGateResult | None = None,
) -> tuple[bool, bool, dict[str, Any] | None, str, dict[str, Any] | None]:
    """Apply master dark / flat to one light FITS and write ``dst``.

    Uses ``fits.open(..., memmap=False)`` and ``with`` blocks so BZERO/BSCALE frames load reliably and
    file handles close after each read (arrays are copied into RAM before processing).

    Returns ``(used_dark, used_flat, qc_summary, vy_cflag, perf10_qc)`` where ``qc_summary`` is set when
    post-calibration QC ran; ``perf10_qc`` holds DAO inspection metrics when ``dao_qc_in_calibrate``;
    ``vy_cflag`` matches ``VY_CFLAG`` written to the FITS (see calibration decision table).
    """
    import numpy as np

    dst.parent.mkdir(parents=True, exist_ok=True)

    _id_eq = None
    if qc_pack is not None:
        try:
            _raw = qc_pack.get("id_equipments")
            _id_eq = int(_raw) if _raw is not None else None
        except (TypeError, ValueError):
            _id_eq = None
    data, hdr, used_dark, used_flat = _calibrate_one_light_apply_masters_in_ram(
        src=src,
        master_dark_path=master_dark_path,
        masterflat_by_filter=masterflat_by_filter,
        flat_norm_floor=flat_norm_floor,
        flat_cache=flat_cache,
        flat_median_scale=flat_median_scale,
        md_data_preload=md_data_preload,
        db=db,
        id_equipments=_id_eq,
        calibration_master_native_binning=calibration_master_native_binning,
        cal_diag_gate_result=cal_diag_gate_result,
    )

    qc_summary: dict[str, Any] | None = None
    _osc_mosaic = _valid_bayerpat_from_header(hdr) is not None
    if (used_dark or used_flat) and qc_pack and qc_pack.get("enabled") and not _osc_mosaic:
        limits = {
            "max_hfr": float(qc_pack.get("max_hfr", 5.0)),
            "min_stars": int(qc_pack.get("min_stars", 10)),
            "max_bg_rms": qc_pack.get("max_bg_rms"),
            "dao_detection_sigma": float(qc_pack.get("dao_detection_sigma", 5.0)),
        }
        qc_summary = _post_calibration_qc_eval(
            np.asarray(data, dtype=np.float32),
            limits=limits,
            light_basename=src.name,
        )
        hdr["VYQCPASS"] = (bool(qc_summary["qc_passed"]), "Post-calibration QC pass")
        hfrv = qc_summary.get("hfr")
        if hfrv is not None and math.isfinite(float(hfrv)):
            hdr["VY_QCHFR"] = (float(hfrv), "QC median HFR [px]")
        hdr["VY_QCNS"] = (int(qc_summary["n_stars"]), "QC DAO detections (central crop if large)")
        sm = qc_summary.get("sky_median")
        if sm is not None and math.isfinite(float(sm)):
            hdr["VY_QCBG"] = (float(sm), "QC sigma-clipped sky median")
        sr = qc_summary.get("sky_rms")
        if sr is not None and math.isfinite(float(sr)):
            hdr["VY_QCRMS"] = (float(sr), "QC sigma-clipped sky RMS")
        if qc_pack.get("draft_id") is not None or (
            qc_pack.get("observation_id") not in (None, "")
        ):
            try:
                db_q = db if db is not None else _db_for_calibration_tasks(qc_pack)
                if db_q is not None:
                    db_q.update_obs_file_qc_by_raw_light_path(
                        src,
                        draft_id=qc_pack.get("draft_id"),
                        observation_id=qc_pack.get("observation_id"),
                        qc_hfr=qc_summary.get("hfr"),
                        qc_stars=int(qc_summary["n_stars"]),
                        qc_background=qc_summary.get("sky_median"),
                        qc_bg_rms=qc_summary.get("sky_rms"),
                        qc_passed=bool(qc_summary["qc_passed"]),
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("manifest files[] QC update failed: %s", exc)

    perf10_qc: dict[str, Any] | None = None
    if (
        not _osc_mosaic
        and qc_pack
        and bool(qc_pack.get("dao_qc_in_calibrate"))
    ):
        try:
            perf10_qc = _quality_inspection_dao_metrics_array(
                np.asarray(data, dtype=np.float32),
                hdr,
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("[PERF-10] DAO QC in calibrate failed for %s: %s", src.name, exc)

    from cal_stage import stamp_cal_stage_headers  # noqa: PLC0415

    cal_stage_token = "PURE"
    cal_datasum = stamp_cal_stage_headers(hdr, data, stage=cal_stage_token)
    fits.writeto(dst, _as_fits_float32_image(data), header=hdr, overwrite=True)
    if qc_pack is not None and qc_pack.get("draft_id") is not None:
        try:
            db_cs = db if db is not None else _db_for_calibration_tasks(qc_pack)
            if db_cs is not None:
                db_cs.update_obs_file_cal_stage_by_raw_light_path(
                    src,
                    draft_id=int(qc_pack["draft_id"]),
                    observation_id=qc_pack.get("observation_id"),
                    cal_stage=cal_stage_token,
                    cal_datasum=cal_datasum,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("manifest cal_stage sync failed after calibrate: %s", exc)
    return used_dark, used_flat, qc_summary, _hdr_vy_cflag_str(hdr), perf10_qc


_cal_batch_flat_cache: dict[str, Any] | None = None
_cal_batch_flat_median: dict[str, float] | None = None
_cal_batch_md_preload: Any = None
_cal_batch_native_binning: int | None = 1
_cal_batch_cal_diag: CalDiagSession | None = None


def _init_calibrate_batch_worker(
    _md_s: str | None,
    native_b: int | None,
    cal_diag_blob: dict[str, Any] | None,
) -> None:
    """Per-subprocess caches; ``native_binning`` = CalibrationLibrary master convention (``None`` = read FITS)."""
    global _cal_batch_flat_cache, _cal_batch_flat_median, _cal_batch_md_preload
    global _cal_batch_native_binning, _cal_batch_cal_diag
    _ = _md_s  # path reserved for future worker-side dark preload
    _cal_batch_flat_cache = {}
    _cal_batch_flat_median = {}
    _cal_batch_md_preload = None
    _cal_batch_cal_diag = _cal_diag_session_from_export(cal_diag_blob)
    if native_b is None:
        _cal_batch_native_binning = None
    else:
        try:
            _cal_batch_native_binning = max(1, int(native_b))
        except (TypeError, ValueError):
            _cal_batch_native_binning = int(CALIBRATION_LIBRARY_NATIVE_BINNING)


def _calibrate_batch_process_one(
    item: tuple[
        str,
        str,
        str | None,
        dict[str, str | None],
        dict[str, Any] | None,
        dict[str, Any] | None,
    ]
    | tuple[str, str, str | None, dict[str, str | None], dict[str, Any] | None]
    | tuple[str, str, str | None, dict[str, str | None]],
) -> dict[str, Any]:
    """Picklable worker: calibrate one light; returns ``dst`` path on success."""
    global _cal_batch_flat_cache, _cal_batch_flat_median, _cal_batch_md_preload
    global _cal_batch_native_binning, _cal_batch_cal_diag
    qc_opt: dict[str, Any] | None = None
    if len(item) == 4:
        src_s, dst_s, md_s, mf_map = item  # type: ignore[misc]
    else:
        src_s, dst_s, md_s, mf_map, qc_opt = item  # type: ignore[misc]
    fc = _cal_batch_flat_cache
    fm = _cal_batch_flat_median
    if fc is None or fm is None:
        fc, fm = {}, {}
    src_p = Path(src_s)
    dst_p = Path(dst_s)
    try:
        _ok = _obs_group_key_from_light_path(src_p)
        if _cal_batch_cal_diag is not None and is_obs_group_aborted(_cal_batch_cal_diag, _ok):
            if dst_p.exists():
                try:
                    dst_p.unlink()
                except OSError:
                    pass
            return {
                "src": src_s,
                "dst": dst_s,
                "ok": True,
                "error": None,
                "qc_summary": None,
                "perf10_qc": None,
                "traceback": None,
                "vy_cflag": "P",
                "skipped": True,
            }
        md_use: Path | None = Path(md_s) if md_s else None
        _md_obs = (qc_opt or {}).get("master_dark_by_obs_key") or {}
        if _md_obs:
            _alt = _md_obs.get(_ok)
            if _alt is not None and str(_alt).strip() != "":
                _pa = Path(str(_alt))
                if _pa.is_file():
                    md_use = _pa
        light_bx = _light_binning_from_path(src_p)
        gr = None
        if _cal_batch_cal_diag is not None:
            gr = gate_result_for_frame(
                _cal_batch_cal_diag,
                obs_group_key=_ok,
                dark_path=md_use,
                light_binning=light_bx,
            )
        md_np = _cal_batch_md_preload
        if md_use is not None and md_use.is_file():
            with fits.open(src_p, memmap=False) as hdul:
                lshape = (int(hdul[0].data.shape[0]), int(hdul[0].data.shape[1]))
            md_np = dark_np_for_cal_diag(
                _cal_batch_cal_diag or CalDiagSession(),
                master_binning=_cal_batch_native_binning,
                dark_path=md_use,
                light_binning=light_bx,
                light_shape=lshape,
                light_filename=src_p.name,
                gate_result=gr,
            )
        mf: dict[str, Path | None] = {str(k): Path(v) if v else None for k, v in mf_map.items()}
        db_w = _db_for_calibration_tasks(qc_opt)
        _ud, _uf, qc_sum, _cf, perf10_qc = _calibrate_one_light_disk(
            src=src_p,
            dst=dst_p,
            master_dark_path=md_use,
            masterflat_by_filter=mf,
            flat_cache=fc,
            flat_median_scale=fm,
            md_data_preload=md_np,
            db=db_w,
            qc_pack=qc_opt,
            calibration_master_native_binning=_cal_batch_native_binning,
            cal_diag_gate_result=gr,
        )
        return {
            "src": src_s,
            "dst": dst_s,
            "ok": True,
            "error": None,
            "qc_summary": qc_sum,
            "perf10_qc": perf10_qc,
            "traceback": None,
            "vy_cflag": _cf,
        }
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0428] [CAL-DIAG] Passthrough reuse path: `update_obs_file_calibration_state_by_raw_light_path...: %s', exc)
        tb = traceback.format_exc()
        LOGGER.error("calibrate_batch worker: %s -> %s\n%s", src_s, exc, tb)
        try:
            log_exception(f"CHYBA WORKERA: {Path(src_s).name}", exc)
        except Exception:  # noqa: BLE001
            pass
        return {
            "src": src_s,
            "dst": None,
            "ok": False,
            "error": str(exc),
            "qc_summary": None,
            "traceback": tb,
        }


def _has_usable_master_dark(path: Path | None) -> bool:
    return bool(path is not None and Path(path).is_file())


def _passthrough_lights_to_calibrated(
    *,
    lights_root: Path,
    calibrated_root: Path,
    progress_cb: "callable | None" = None,
    database_path: Path | None = None,
    draft_id: int | None = None,
    observation_id: str | None = None,
) -> dict[str, Any]:
    """Passthrough mode: copy raw lights to calibrated and mark FITS header."""
    files = _iter_light_fits(lights_root)
    total = len(files)
    calibrated_root.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "processed": 0,
        "used_dark": 0,
        "used_flat": 0,
        "copied_only": 0,
        "errors": 0,
        "calibrate_workers": 1,
        "qc_evaluated": 0,
        "qc_rejected": 0,
        "passthrough_mode": True,
        "passthrough_existing_reused": 0,
    }
    db_pt: VyvarDatabase | None = None
    try:
        if database_path is not None and Path(database_path).is_file():
            db_pt = VyvarDatabase(Path(database_path))
    except Exception:  # noqa: BLE001
        db_pt = None
    for i, src in enumerate(files, start=1):
        rel = src.relative_to(lights_root)
        dst = calibrated_root / rel
        if progress_cb is not None:
            progress_cb(i, total, f"Passthrough {src.name}")
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                # Reuse previous calibrated output; do not fail when rerunning.
                stats["processed"] += 1
                stats["copied_only"] += 1
                stats["passthrough_existing_reused"] += 1
                if db_pt is not None:
                    try:
                        db_pt.update_obs_file_calibration_state_by_raw_light_path(
                            src,
                            draft_id=draft_id,
                            observation_id=observation_id,
                            is_calibrated=0,
                            calib_type="PASSTHROUGH",
                            calib_flags="P",
                        )
                    except Exception:  # noqa: BLE001
                        pass
                continue
            with fits.open(src, memmap=False) as hdul:
                hdr = hdul[0].header.copy()
                data = np.asarray(hdul[0].data, dtype=np.float32)
            hdr["VYVARCAL"] = (True, "VYVAR calibrated output")
            hdr["VY_DARK"] = (False, "MasterDark applied")
            hdr["VY_FLAT"] = (False, "MasterFlat applied")
            hdr["VY_CFLAG"] = ("P", "VYVAR flags: DF=full, D=partial, FS=flat skipped, P=passthrough/raw")
            hdr["VY_CALIB"] = ("PASSTHROUGH", "Calibration mode")
            hdr["VY_CALST"] = (0, "VYVAR CALIB_STATUS: 2=full DF, 1=partial D, 0=raw (passthrough)")
            hdr.add_history("No calibration frames applied.")
            passthrough_cal_diag_headers(hdr)
            from cal_stage import stamp_cal_stage_headers  # noqa: PLC0415

            cal_stage_token = "PASSTHROUGH"
            cal_datasum = stamp_cal_stage_headers(hdr, data, stage=cal_stage_token)
            fits.writeto(dst, _as_fits_float32_image(data), header=hdr, overwrite=True)
            stats["processed"] += 1
            stats["copied_only"] += 1
            if db_pt is not None:
                try:
                    db_pt.update_obs_file_calibration_state_by_raw_light_path(
                        src,
                        draft_id=draft_id,
                        observation_id=observation_id,
                        is_calibrated=0,
                        calib_type="PASSTHROUGH",
                        calib_flags="P",
                    )
                    db_pt.update_obs_file_cal_stage_by_raw_light_path(
                        src,
                        draft_id=draft_id,
                        observation_id=observation_id,
                        cal_stage=cal_stage_token,
                        cal_datasum=cal_datasum,
                    )
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            stats["errors"] += 1
    try:
        if db_pt is not None:
            db_pt.conn.close()
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0431] `filter_light_paths_for_calibration_db` failure logs and skips IS_REJECTED filter, pote...: %s', exc)
        pass
    return stats


def calibrate_lights_to_calibrated(
    *,
    lights_root: Path,
    calibrated_root: Path,
    master_dark_path: Path | None,
    masterflat_by_filter: dict[str, Path | None],
    progress_cb: "callable | None" = None,
    pipeline_config: AppConfig | None = None,
    equipment_id: int | None = None,
    max_workers: int | None = None,
    draft_id: int | None = None,
    observation_id: str | None = None,
    masterflat_by_obs_key: dict[str, str | Path | None] | None = None,
    master_dark_by_obs_key: dict[str, str | Path | None] | None = None,
) -> dict[str, Any]:
    """Apply available masters to lights and write into calibrated_root.

    Works in all modes:
    - dark+flat
    - only dark
    - none (passthrough copy when dark is unavailable)

    MasterFlat is **median-normalized to ~1.0** before division so calibrated values stay in the same order
    of magnitude as (light - dark) and extreme spikes from tiny flat pixels are avoided.

    After normalization, each flat pixel is **clamped below by a small positive floor** so dust donuts and
    dead columns cannot divide the image by near-zero (which would create nonsense ADU far above the
    detector range in viewers).

    Pixels are **not** expected to match raw ADU counts: calibration is ``(L - D) / F_norm`` with
    ``F_norm = flat / median(flat)``. Raw linearity keywords (SATURATE, DATAMAX, ...) are dropped when any
    calibration is applied so headers stay consistent with the stored float image.

    Parallelism uses ``max_workers`` or auto ``pipeline_config.qc_preprocess_workers`` only when environment
    variable ``VYVAR_CALIBRATE_MP`` is set to ``1``/``true`` (default is sequential for clearer tracebacks).
    """
    import numpy as np

    cfg = pipeline_config or AppConfig()
    qc_pack = _qc_pack_from_config(cfg, draft_id=draft_id, observation_id=observation_id)
    nw = max_workers if max_workers is not None else int(cfg.qc_preprocess_workers)
    nw = max(1, min(32, int(nw)))
    if not _vyvar_calibrate_multiprocessing_enabled():
        nw = 1
    if master_dark_by_obs_key:
        nw = 1

    mf_merged: dict[str, Path | None] = {}
    for k, v in (masterflat_by_filter or {}).items():
        mf_merged[str(k)] = None if v is None else Path(v)
    for k, v in (masterflat_by_obs_key or {}).items():
        mf_merged[str(k)] = None if v is None or str(v).strip() == "" else Path(v)
    masterflat_by_filter = mf_merged

    calibrated_root.mkdir(parents=True, exist_ok=True)
    _log_calibration_io_preflight(
        calibrated_root=calibrated_root,
        master_dark_path=master_dark_path,
        masterflat_by_filter=masterflat_by_filter,
    )

    _has_dark_from_obs = any(
        Path(v).is_file() for v in (master_dark_by_obs_key or {}).values() if v is not None and str(v).strip() != ""
    )
    _has_dark_any = _has_usable_master_dark(master_dark_path) or bool(_has_dark_from_obs)
    # Dark-first policy: if no usable dark exists, keep pipeline alive in passthrough mode.
    if not _has_dark_any:
        log_event(
            "Calibration Passthrough: missing MasterDark -> "
            "copy Raw/lights to calibrated/lights with VY_CALIB=PASSTHROUGH."
        )
        return _passthrough_lights_to_calibrated(
            lights_root=lights_root,
            calibrated_root=calibrated_root,
            progress_cb=progress_cb,
            database_path=Path(cfg.database_path) if str(cfg.database_path).strip() else None,
            draft_id=draft_id,
            observation_id=observation_id,
        )

    md_pre: Any = None
    md_path_ok: Path | None = None
    md_init_str: str | None = None
    if master_dark_path is not None and master_dark_path.exists():
        md_path_ok = master_dark_path
        md_init_str = str(master_dark_path.resolve())
        with fits.open(master_dark_path, memmap=False) as hdul:
            md_pre = np.array(hdul[0].data, dtype=np.float32, copy=True)

    _native_b = _cfg_calibration_library_native_binning(cfg)

    mf_serial: dict[str, str | None] = {}
    for k, v in (masterflat_by_filter or {}).items():
        if v is None:
            mf_serial[str(k)] = None
        else:
            mf_serial[str(k)] = str(Path(v).resolve())

    flat_cache: dict[str, Any] = {}
    flat_median_scale: dict[str, float] = {}
    stats: dict[str, Any] = {
        "processed": 0,
        "used_dark": 0,
        "used_flat": 0,
        "copied_only": 0,
        "errors": 0,
        "calibrate_workers": 1,
        "qc_evaluated": 0,
        "qc_rejected": 0,
        "applied_focal_length": None,
        "applied_pixel_size": None,
        "perf10_qc_results": {},
        "cal_db_sync_failures": 0,
        "cal_db_sync_errors": [],
    }
    perf10_qc_results: dict[str, dict[str, Any]] = stats["perf10_qc_results"]

    files = _iter_light_fits(lights_root)
    _n_before_obs_filter = len(files)
    if cfg.database_path:
        try:
            _dbp = Path(cfg.database_path)
            if _dbp.is_file():
                files = filter_light_paths_for_calibration_db(
                    files,
                    database_path=_dbp,
                    draft_id=draft_id,
                    observation_id=observation_id,
                )
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0432] [CAL-DIAG] First-light metadata diagnostic (`extract_fits_metadata` for focal/pixel log...: %s', exc)
            log_event(f"manifest files[] IS_REJECTED filter skipped (error): {exc}")
    total = len(files)
    if total < _n_before_obs_filter:
        log_event(
            f"Kalibracia - vylucene {_n_before_obs_filter - total} suborov podla manifest files[] (IS_REJECTED=1 alebo mimo DB)"
        )

    _suffix_note = ", ".join(sorted(FITS_SUFFIXES_LOWER))
    _log_bits = [
        f"lights_root={lights_root.resolve()}",
        f"disk_fits_count={total} (suffixes {_suffix_note}, case-insensitive via path.suffix.casefold)",
    ]
    if observation_id or draft_id is not None:
        try:
            _dbc = VyvarDatabase(Path(cfg.database_path))
            if observation_id:
                _n_o = _dbc.count_obs_files_for_observation(str(observation_id))
                _log_bits.append(
                    f"manifest files[] count (obs={observation_id!r}) -> {_n_o} rows"
                )
            if draft_id is not None:
                _n_d = _dbc.count_obs_files_for_draft(int(draft_id))
                _log_bits.append(
                    f"manifest files[] count (draft_id={int(draft_id)}) -> {_n_d} rows"
                )
        except Exception as exc:  # noqa: BLE001
            _log_bits.append(f"manifest files[] count failed: {exc}")
    log_event("Kalibracia - vstupne subory: " + " | ".join(_log_bits))

    if total > 0:
        log_lights_binning_from_headers_preflight(files, context="Kalibracia")
        _diag_light = _pick_light_for_metadata_diagnostic(files)
        try:
            _db_cal_meta = VyvarDatabase(Path(cfg.database_path))
            try:
                _meta0 = extract_fits_metadata(
                    _diag_light,
                    db=_db_cal_meta,
                    app_config=cfg,
                    id_equipment=equipment_id,
                    draft_id=draft_id,
                )
                _log_calibration_metadata_diagnostic(_diag_light.name, _meta0)
                stats["applied_focal_length"] = _meta0.get("focal_length")
                stats["applied_pixel_size"] = _meta0.get("pixel_um")
            finally:
                _db_cal_meta.conn.close()
        except Exception as exc:  # noqa: BLE001
            log_event(f"DIAGNOSTIKA KALIBRACIE: metadata prveho suboru zlyhali: {exc!s}")

    db_main = _db_for_calibration_tasks(qc_pack)

    def _refresh_manifest_after_cal_qc() -> None:
        if draft_id is not None and db_main is not None:
            try:
                db_main._try_refresh_draft_manifest(int(draft_id))
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("draft_manifest refresh after calibrate QC: %s", exc)

    cal_diag_session = CalDiagSession()

    def _resolve_dark_for_pregate(fp: Path, og: str, lb: int) -> Path | None:
        return _resolve_dark_path_for_light(
            src=fp,
            obs_group_key=og,
            master_dark_path=md_path_ok,
            master_dark_by_obs_key=master_dark_by_obs_key,
        )

    def _sat_for_pregate(fp: Path) -> float | None:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
        return _saturation_adu_for_cal_diag(hdr, db=db_main, equipment_id=equipment_id)

    if total > 0:
        cal_diag_session = run_cal_diag_pregate(
            files,
            obs_group_key_from_path=_obs_group_key_from_light_path,
            resolve_dark_path=_resolve_dark_for_pregate,
            light_binning_from_path=_light_binning_from_path,
            master_binning=_native_b,
            match_and_crop_pair=_match_and_crop_pair,
            saturation_for_light=_sat_for_pregate,
            ui_error=_pipeline_ui_error,
        )
    stats["cal_diag_aborted_groups"] = len(cal_diag_session.aborted_groups)
    cal_diag_worker_blob = _cal_diag_export_for_workers(cal_diag_session)

    dark_cache: dict[str, Any] = {}

    def _one_sequential(i: int, src: Path, dst: Path) -> None:
        nonlocal stats
        if progress_cb is not None:
            progress_cb(i, total, f"Calibrating {src.name}")
        md_use = md_path_ok
        md_np_use = None
        light_bx = 1
        with fits.open(src, memmap=False) as hdul:
            hdr_l = hdul[0].header
            _ok = observation_group_key_from_metadata(fits_metadata_from_primary_header(hdr_l))
            light_bx, _ = fits_binning_xy_from_header(hdr_l)
            _light_shape = (int(hdul[0].data.shape[0]), int(hdul[0].data.shape[1]))
        if is_obs_group_aborted(cal_diag_session, _ok):
            if dst.exists():
                try:
                    dst.unlink()
                except OSError:
                    pass
            return
        if master_dark_by_obs_key:
            _alt = master_dark_by_obs_key.get(_ok)
            if _alt is not None and str(_alt).strip() != "":
                _pa = Path(_alt)
                if _pa.is_file():
                    md_use = _pa
        gr = gate_result_for_frame(
            cal_diag_session,
            obs_group_key=_ok,
            dark_path=md_use,
            light_binning=light_bx,
        )
        if md_use is not None and md_use.is_file():
            if (
                md_pre is not None
                and md_path_ok is not None
                and md_use.resolve() == md_path_ok.resolve()
                and _native_b is not None
                and _native_b == light_bx
                and (gr is None or convention_to_dark_mode(gr.convention) == "sum")
            ):
                md_np_use = md_pre
            else:
                md_np_use = dark_np_for_cal_diag(
                    cal_diag_session,
                    master_binning=_native_b,
                    dark_path=md_use,
                    light_binning=light_bx,
                    light_shape=_light_shape,
                    light_filename=src.name,
                    gate_result=gr,
                )
        used_dark, used_flat, qc_sum, _flags, perf10_qc = _calibrate_one_light_disk(
            src=src,
            dst=dst,
            master_dark_path=md_use,
            masterflat_by_filter=masterflat_by_filter,
            flat_cache=flat_cache,
            flat_median_scale=flat_median_scale,
            md_data_preload=md_np_use,
            db=db_main,
            qc_pack=qc_pack,
            calibration_master_native_binning=_native_b,
            cal_diag_gate_result=gr,
        )
        if isinstance(perf10_qc, dict) and perf10_qc and not perf10_qc.get("error"):
            perf10_qc_results[str(src.resolve())] = perf10_qc
        _sync_obs_calibration_state_with_retry(
            db_main,
            raw_light_path=src,
            draft_id=draft_id,
            observation_id=observation_id,
            is_calibrated=1 if "D" in _flags else 0,
            calib_type=_calibration_type_from_flags(_flags),
            calib_flags=_flags,
            stats=stats,
        )
        stats["processed"] += 1
        if used_dark:
            stats["used_dark"] += 1
        if used_flat:
            stats["used_flat"] += 1
        if not used_dark and not used_flat:
            stats["copied_only"] += 1
        if qc_sum is not None:
            stats["qc_evaluated"] += 1
            if not bool(qc_sum.get("qc_passed", True)):
                stats["qc_rejected"] += 1

    if _vyvar_calibrate_multiprocessing_enabled() and nw > 1 and total > 1:
        items: list[
            tuple[
                str,
                str,
                str | None,
                dict[str, str | None],
                dict[str, Any] | None,
            ]
        ] = []
        for src in files:
            rel = src.relative_to(lights_root)
            dst = calibrated_root / rel
            _qc_mp = dict(qc_pack) if qc_pack else {}
            if master_dark_by_obs_key:
                _qc_mp["master_dark_by_obs_key"] = {
                    str(k): str(Path(v).resolve()) if v is not None and str(v).strip() != "" else None
                    for k, v in master_dark_by_obs_key.items()
                }
            items.append(
                (
                    str(src.resolve()),
                    str(dst.resolve()),
                    md_init_str,
                    mf_serial,
                    _qc_mp,
                )
            )
        stats["calibrate_workers"] = min(nw, total)
        ctx = multiprocessing.get_context("spawn")
        rows: list[dict[str, Any] | None] = [None] * total
        try:
            with ProcessPoolExecutor(
                max_workers=stats["calibrate_workers"],
                mp_context=ctx,
                initializer=_init_calibrate_batch_worker,
                initargs=(md_init_str, _native_b, cal_diag_worker_blob),
            ) as ex:
                future_map = {
                    ex.submit(_calibrate_batch_process_one, it): idx for idx, it in enumerate(items)
                }
                done = 0
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    rows[idx] = fut.result()
                    done += 1
                    if progress_cb is not None:
                        src_name = Path(items[idx][0]).name
                        progress_cb(done, total, f"Calibrating batch {done}/{total} ({src_name})")
        except Exception as exc:  # noqa: BLE001
            # EXC-0434: T2 -- [CAL-DIAG] MP calibrate batch: same manifest files[] calibration-state update `pass` after wor... (EXCEPT-BULK-2 2026-07-08)
            _tb_pool = traceback.format_exc()
            LOGGER.error("Kalibracia (parallel): pool zlyhal, fallback na sekvencny rezim: %s\n%s", exc, _tb_pool)
            log_exception("CHYBA POOLU KALIBRACIE", exc)
            stats["errors"] = 0
            stats["processed"] = 0
            stats["used_dark"] = 0
            stats["used_flat"] = 0
            stats["copied_only"] = 0
            stats["qc_evaluated"] = 0
            stats["qc_rejected"] = 0
            stats["calibrate_workers"] = 1
            for i, src in enumerate(files, start=1):
                rel = src.relative_to(lights_root)
                dst = calibrated_root / rel
                try:
                    _one_sequential(i, src, dst)
                except Exception as exc2:  # noqa: BLE001
                    _tb2 = traceback.format_exc()
                    LOGGER.error("Kalibracia: subor %s: %s\n%s", src, exc2, _tb2)
                    log_exception(f"CHYBA KALIBRACIE: {src.name}", exc2)
                    stats["errors"] += 1
            _arch = _archive_root_from_lights_root(lights_root)
            if _arch is not None:
                write_cal_diag_json(_arch, cal_diag_session)
            _refresh_manifest_after_cal_qc()
            return stats

        for idx, r in enumerate(rows):
            if r is None or not r.get("ok"):
                stats["errors"] += 1
                if stats["errors"] == 1:
                    _ename = Path(items[idx][0]).name if idx < len(items) else "?"
                    _emsg = (r or {}).get("error") if isinstance(r, dict) else None
                    LOGGER.error(
                        "Kalibracia: worker zlyhal pre %s: %s",
                        _ename,
                        _emsg or r,
                    )
                    if isinstance(r, dict) and r.get("traceback"):
                        log_event(f"CHYBA WORKERA: {_ename}: {r.get('error', '')}")
                        log_event(str(r["traceback"]))
                    elif isinstance(r, dict) and r.get("error"):
                        log_event(f"CHYBA WORKERA: {_ename}: {r.get('error')}")
                    else:
                        log_event(f"CHYBA WORKERA: {_ename}: {_emsg or 'no traceback in result'}")
                continue
            stats["processed"] += 1
            qcs = r.get("qc_summary")
            if qcs is not None:
                stats["qc_evaluated"] += 1
                if not bool(qcs.get("qc_passed", True)):
                    stats["qc_rejected"] += 1
            try:
                with fits.open(Path(items[idx][1]), memmap=False) as hh:
                    h0 = hh[0].header
                    _flags = _hdr_vy_cflag_str(h0)
                    ud = bool(h0.get("VY_DARK", False))
                    uf = bool(h0.get("VY_FLAT", False))
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0435] `_estimate_fwhm_from_image` broad `except` returns hardcoded `3.0` px; QC FWHM reject a...: %s', exc)
                ud = uf = False
                _flags = "P"
            _sync_obs_calibration_state_with_retry(
                db_main,
                raw_light_path=Path(items[idx][0]),
                draft_id=draft_id,
                observation_id=observation_id,
                is_calibrated=1 if "D" in _flags else 0,
                calib_type=_calibration_type_from_flags(_flags),
                calib_flags=_flags,
                stats=stats,
            )
            if ud:
                stats["used_dark"] += 1
            if uf:
                stats["used_flat"] += 1
            if not ud and not uf:
                stats["copied_only"] += 1
            p10 = r.get("perf10_qc") if isinstance(r, dict) else None
            if isinstance(p10, dict) and p10 and not p10.get("error"):
                perf10_qc_results[str(Path(items[idx][0]).resolve())] = p10
        _arch = _archive_root_from_lights_root(lights_root)
        if _arch is not None:
            write_cal_diag_json(_arch, cal_diag_session)
        _refresh_manifest_after_cal_qc()
        return stats

    _seq_tb_logged = False
    for i, src in enumerate(files, start=1):
        rel = src.relative_to(lights_root)
        dst = calibrated_root / rel
        try:
            _one_sequential(i, src, dst)
        except Exception as exc:  # noqa: BLE001
            _tb_seq = traceback.format_exc()
            LOGGER.error("Kalibracia: subor %s: %s\n%s", src, exc, _tb_seq)
            if not _seq_tb_logged:
                log_exception(f"CHYBA KALIBRACIE: {src.name}", exc)
                _seq_tb_logged = True
            else:
                log_event(f"CHYBA KALIBRACIE: {src.name}: {exc!s}")
            stats["errors"] += 1
            continue

    _arch = _archive_root_from_lights_root(lights_root)
    if _arch is not None:
        write_cal_diag_json(_arch, cal_diag_session)

    _refresh_manifest_after_cal_qc()
    return stats


def _estimate_dao_fwhm_guess(img2: "np.ndarray", std: float) -> float:
    """DAOStarFinder kernel FWHM hint (pixels).

    Returns a star-scale default. Do not derive this from segmentation SourceCatalog
    after CR cleaning was removed -- that path preferred hot pixels / cosmics (~1-2 px)
    and pulled the kernel (and later FWHM) unphysically low.
    """
    _ = (img2, std)
    return 4.5


def _moment_fwhm_elong_peak_at(
    img2: "np.ndarray",
    x0: float,
    y0: float,
    *,
    half: int = 7,
) -> tuple[float | None, float | None, float, float, float]:
    """Moment FWHM/elongation at ``(x0,y0)`` on a background-subtracted image.

    Returns ``(fwhm_px, elongation, peak, flux_sum, concentration)`` where
    ``concentration = peak / flux_sum`` (hot pixels / CRs are near 1).
    """
    import numpy as np

    h, w = img2.shape
    xi = int(round(float(x0)))
    yi = int(round(float(y0)))
    y1 = max(0, yi - half)
    y2 = min(h, yi + half + 1)
    x1 = max(0, xi - half)
    x2 = min(w, xi + half + 1)
    cut = np.asarray(img2[y1:y2, x1:x2], dtype=np.float32)
    if cut.size < 25:
        return None, None, 0.0, 0.0, 1.0
    cut_pos = np.where(cut > 0, cut, 0.0).astype(np.float32)
    flux_sum = float(np.sum(cut_pos))
    peak = float(np.max(cut_pos)) if cut_pos.size else 0.0
    if not math.isfinite(flux_sum) or flux_sum <= 0 or not math.isfinite(peak) or peak <= 0:
        return None, None, peak, flux_sum, 1.0
    conc = float(peak / flux_sum)
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    cx = float(np.sum(xx * cut_pos) / flux_sum)
    cy = float(np.sum(yy * cut_pos) / flux_sum)
    dx = xx - cx
    dy = yy - cy
    mxx = float(np.sum((dx * dx) * cut_pos) / flux_sum)
    myy = float(np.sum((dy * dy) * cut_pos) / flux_sum)
    mxy = float(np.sum((dx * dy) * cut_pos) / flux_sum)
    tr = mxx + myy
    det = mxx * myy - mxy * mxy
    disc = tr * tr - 4.0 * det
    if disc < 0:
        return None, None, peak, flux_sum, conc
    l1 = 0.5 * (tr + float(np.sqrt(disc)))
    l2 = 0.5 * (tr - float(np.sqrt(disc)))
    if l1 <= 0 or l2 <= 0:
        return None, None, peak, flux_sum, conc
    sig1 = float(np.sqrt(l1))
    sig2 = float(np.sqrt(l2))
    fwhm = 2.355 * 0.5 * (sig1 + sig2)
    elong = (sig1 / sig2) if sig2 > 0 else float("nan")
    if not (math.isfinite(fwhm) and 0.2 < fwhm < 50):
        return None, None, peak, flux_sum, conc
    if not (math.isfinite(elong) and 0.5 < elong < 20):
        return float(fwhm), None, peak, flux_sum, conc
    return float(fwhm), float(elong), peak, flux_sum, conc


def _robust_frame_fwhm_median(
    data: "np.ndarray",
    *,
    max_sources: int = 120,
    min_keep: int = 12,
    use_center_crop: bool = True,
    sat_adu: float | None = None,
    isol_px: float = 10.0,
    max_concentration: float = 0.22,
    elong_lo: float = 0.75,
    elong_hi: float = 1.55,
) -> dict[str, Any]:
    """Per-frame FWHM = median moment-FWHM over many star-like detections.

    Membership selection only (no sigma-clip of the FWHM sample; Milan 2026-08-12):
    - DAO detections on a background-subtracted (optional center-crop) image
    - Unsaturated (peak < ``sat_adu``)
    - Extended / anti-CR: ``peak/sum <= max_concentration`` (rejects hot pixels / cosmics)
    - Roundish: moment elongation in ``[elong_lo, elong_hi]``
    - Isolated: no brighter accepted neighbor within ``isol_px``
    - Aggregate: median of remaining FWHMs (requires ``min_keep``)

    This is measurement aggregation across stars; it does not alter science pixels.
    """
    import numpy as np
    from photutils.detection import DAOStarFinder

    out: dict[str, Any] = {
        "fwhm_px": None,
        "elongation": None,
        "n_sources": 0,
        "n_stars_detected": 0,
        "n_fwhm_sample": 0,
    }
    img = np.asarray(data, dtype=np.float32)
    if img.ndim != 2:
        return out
    work = _qc_center_crop_for_stars(img) if use_center_crop else img
    finite = np.isfinite(work)
    if not np.any(finite):
        return out
    _, med, std = plain_mean_med_std(work[finite])
    std_f = float(std)
    if not math.isfinite(std_f) or std_f <= 0:
        return out
    img2 = np.asarray(work - float(med), dtype=np.float32)
    img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
    # Polarity: only flip when the negative tail clearly dominates (stars as dips).
    # Do NOT flip on a near-zero median after background subtraction -- float noise
    # around 0 would invert a normal star field and DAO finds almost nothing.
    if math.isfinite(std_f) and std_f > 0:
        pos = float(np.count_nonzero(img2 > (4.0 * std_f)))
        neg = float(np.count_nonzero(img2 < (-4.0 * std_f)))
        if neg > (pos * 2.0) and neg > 100:
            img2 = -img2

    # DAO kernel must not be tuned to CR-scale (~1-2 px) blobs.
    fwhm_guess = float(max(3.0, min(12.0, _estimate_dao_fwhm_guess(img2, std_f))))
    sat = float(sat_adu) if sat_adu is not None and math.isfinite(float(sat_adu)) else 50000.0
    sat = max(1000.0, sat)

    tbl = None
    for thr_k in (5.0, 3.5, 2.5):
        daofind = DAOStarFinder(
            fwhm=float(fwhm_guess),
            threshold=float(thr_k) * std_f,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = daofind(img2)
        if tbl is not None and len(tbl) >= max(8, min_keep):
            break
    if tbl is None or len(tbl) == 0:
        return out
    out["n_stars_detected"] = int(len(tbl))
    tbl.sort("flux")
    tbl = tbl[::-1]

    n_scan = int(min(len(tbl), max(max_sources * 3, max_sources)))
    cand: list[tuple[float, float, float, float, float]] = []
    for i in range(n_scan):
        x0 = float(tbl["x_centroid"][i])
        y0 = float(tbl["y_centroid"][i])
        flux = float(tbl["flux"][i])
        fwhm, elong, peak, _flux_sum, conc = _moment_fwhm_elong_peak_at(img2, x0, y0, half=7)
        if fwhm is None or elong is None:
            continue
        if not math.isfinite(peak) or peak >= sat:
            continue
        if not math.isfinite(conc) or conc > float(max_concentration):
            continue
        if not (float(elong_lo) <= float(elong) <= float(elong_hi)):
            continue
        cand.append((flux, x0, y0, float(fwhm), float(elong)))

    accepted_fwhm: list[float] = []
    accepted_elong: list[float] = []
    accepted_xy: list[tuple[float, float]] = []
    isol2 = float(isol_px) * float(isol_px)
    for _flux, x0, y0, fwhm, elong in cand:
        if len(accepted_fwhm) >= int(max_sources):
            break
        ok_iso = True
        for ax, ay in accepted_xy:
            if (x0 - ax) * (x0 - ax) + (y0 - ay) * (y0 - ay) < isol2:
                ok_iso = False
                break
        if not ok_iso:
            continue
        accepted_fwhm.append(fwhm)
        accepted_elong.append(elong)
        accepted_xy.append((x0, y0))

    out["n_fwhm_sample"] = int(len(accepted_fwhm))
    out["n_sources"] = int(len(accepted_fwhm))
    if len(accepted_fwhm) < int(min_keep):
        return out
    arr_f = np.asarray(accepted_fwhm, dtype=np.float64)
    arr_e = np.asarray(accepted_elong, dtype=np.float64)
    out["fwhm_px"] = float(np.median(arr_f))
    out["elongation"] = float(np.median(arr_e))
    return out


def _qc_fwhm_elongation(
    data: "np.ndarray",
    *,
    max_sources: int = 200,
) -> dict[str, Any]:
    """Estimate average FWHM and elongation (best-effort).

    Robust path (2026-08-12): median moment-FWHM over many star-like DAO detections
    (bright, unsaturated, isolated, extended). Segmentation SourceCatalog is NOT used
    for FWHM -- after CR cleaning was removed it preferentially measured hot pixels /
    cosmics (~1-2 px) and poisoned ``VY_FWHM`` / aperture sizing.

    Returns ``n_stars_detected``: approximate total star-like detections on the frame.
    ``n_sources``: subset used for the FWHM/elongation median.
    """
    _ = max_sources
    rob = _robust_frame_fwhm_median(
        data,
        max_sources=120,
        min_keep=12,
        use_center_crop=True,
    )
    if rob.get("fwhm_px") is not None:
        return {
            "fwhm_px": rob.get("fwhm_px"),
            "elongation": rob.get("elongation"),
            "n_sources": int(rob.get("n_sources") or 0),
            "n_stars_detected": int(rob.get("n_stars_detected") or 0),
        }
    rob2 = _robust_frame_fwhm_median(
        data,
        max_sources=120,
        min_keep=5,
        use_center_crop=True,
    )
    return {
        "fwhm_px": rob2.get("fwhm_px"),
        "elongation": rob2.get("elongation"),
        "n_sources": int(rob2.get("n_sources") or 0),
        "n_stars_detected": int(rob2.get("n_stars_detected") or 0),
    }

def _vyvar_parallel_worker_count(app_config: AppConfig | None = None) -> int:
    """Jednotny pocet workerov pre QC, preprocess, combined, alignment seed, per-frame CSV seed, calibrate MP.

    Prednost: ``VYVAR_PARALLEL_WORKERS`` -> (ak su obe) minimum z legacy ``VYVAR_QC_PREPROCESS_WORKERS`` a
    ``VYVAR_PER_FRAME_CSV_WORKERS`` -> ``app_config.qc_preprocess_workers`` (uz zjednotene) -> host auto z ``config``.
    """
    u = os.environ.get("VYVAR_PARALLEL_WORKERS")
    if u is not None and str(u).strip() != "":
        try:
            return max(1, min(32, int(str(u).strip())))
        except ValueError:
            pass
    legacy: list[int] = []
    for key in ("VYVAR_QC_PREPROCESS_WORKERS", "VYVAR_PER_FRAME_CSV_WORKERS"):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip() != "":
            try:
                legacy.append(max(1, min(32, int(str(raw).strip()))))
            except ValueError:
                pass
    if legacy:
        return int(min(legacy))
    if app_config is not None:
        try:
            return max(1, min(32, int(app_config.qc_preprocess_workers)))
        except (TypeError, ValueError):
            pass
    from config import load_config_json, recommended_vyvar_parallel_workers, resolve_data_root

    _install = Path(__file__).resolve().parent.parent
    data = load_config_json(resolve_data_root(_install))
    try:
        res_gb = float(data.get("per_frame_mp_reserve_ram_gb", 1.5))
        if not math.isfinite(res_gb) or res_gb < 0:
            res_gb = 1.5
    except (TypeError, ValueError):
        res_gb = 1.5
    return int(recommended_vyvar_parallel_workers(reserve_ram_gb=res_gb))


def _vyvar_qc_preprocess_workers() -> int:
    """Parallel workers for analyze / preprocess / combined (see :func:`_vyvar_parallel_worker_count`)."""
    return _vyvar_parallel_worker_count(None)


def _fit_subtract_preprocess_sky_surface(
    data: Any,
    *,
    order: int,
    fwhm_px: float | None = None,
    subsample_step: int = 4,
    calm_adu: float = 100.0,
) -> tuple[Any, dict[str, Any]]:
    """Fit and subtract a 2D polynomial sky correction (shared calibrated->processed preprocess).

  Star-masked fit on calm background pixels (|cal-median| < ``calm_adu``).
  Fits order-N to ``work - bg_median`` on the fit set, then subtracts the fitted surface from
  the frame (429-class gradient removal; full surface including constant term).
  No sigma-clip on fit samples (zero-clipping policy 2026-08-12).
    """
    import numpy as np
    from photutils.detection import DAOStarFinder

    arr = np.asarray(data, dtype=np.float32)
    order_i = int(order)
    if order_i <= 0:
        return arr.copy(), {"sky_surface_order": 0, "sky_surface_applied": False}

    order_i = min(2, max(1, order_i))
    h, w = arr.shape
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
    work = np.where(finite, arr, fill)

    mask = np.ones((h, w), dtype=bool)
    margin = 40
    if h > 2 * margin and w > 2 * margin:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False

    fwhm_eff = max(
        1.2,
        float(fwhm_px) if fwhm_px is not None and math.isfinite(float(fwhm_px)) else 2.5,
    )
    _, med, std = plain_mean_med_std(work)
    data0 = np.nan_to_num((work - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    thr = max(3.0 * float(std), 1e-6)
    finder = DAOStarFinder(
        fwhm=float(fwhm_eff),
        threshold=float(thr),
        n_brightest=5000,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    tbl = finder(data0)
    stamp_r = int(max(4, round(3.5 * fwhm_eff)))
    if tbl is not None and len(tbl) > 0:
        r2 = stamp_r * stamp_r
        for row in tbl:
            cy = int(round(float(row["y_centroid"])))
            cx = int(round(float(row["x_centroid"])))
            if not (0 <= cy < h and 0 <= cx < w):
                continue
            y0, y1 = max(0, cy - stamp_r), min(h, cy + stamp_r + 1)
            x0, x1 = max(0, cx - stamp_r), min(w, cx + stamp_r + 1)
            yy_l, xx_l = np.ogrid[y0:y1, x0:x1]
            local_excl = (yy_l - cy) ** 2 + (xx_l - cx) ** 2 <= r2
            mask[y0:y1, x0:x1] &= ~local_excl

    bg_median, _, _ = plain_mean_med_std(work, mask=~mask)
    calm_thr = max(5.0, float(calm_adu))
    fit_mask = mask & (np.abs(work - float(bg_median)) < calm_thr)

    step = max(1, int(subsample_step))
    yy_s, xx_s = np.mgrid[0:h:step, 0:w:step]
    z_s = (work[::step, ::step] - float(bg_median)).astype(np.float64)
    m_s = fit_mask[::step, ::step]
    use_mask = m_s & np.isfinite(z_s)
    min_coef = (order_i + 1) * (order_i + 2) // 2
    if int(np.count_nonzero(use_mask)) < min_coef + 10:
        return arr.copy(), {
            "sky_surface_order": order_i,
            "sky_surface_applied": False,
            "sky_surface_skip_reason": "insufficient_fit_pixels",
        }

    z_samples = z_s[use_mask]
    x_fit = xx_s[use_mask].astype(np.float64)
    y_fit = yy_s[use_mask].astype(np.float64)
    z_fit = z_samples
    if z_fit.size < min_coef + 5:
        return arr.copy(), {
            "sky_surface_order": order_i,
            "sky_surface_applied": False,
            "sky_surface_skip_reason": "insufficient_fit_pixels",
        }

    cols: list[np.ndarray] = []
    for i in range(order_i + 1):
        for j in range(order_i + 1 - i):
            cols.append((x_fit**i) * (y_fit**j))
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), z_fit, rcond=None)

    yy_f, xx_f = np.mgrid[0:h, 0:w]
    cols_f: list[np.ndarray] = []
    x_flat = xx_f.ravel().astype(np.float64)
    y_flat = yy_f.ravel().astype(np.float64)
    for i in range(order_i + 1):
        for j in range(order_i + 1 - i):
            cols_f.append((x_flat**i) * (y_flat**j))
    surf = (np.column_stack(cols_f) @ coef).reshape(h, w).astype(np.float32)

    out = (work - surf).astype(np.float32)
    out = np.where(finite, out, np.nan).astype(np.float32)
    # INV-FLAT-01: residual large-scale flatness after full-surface subtract (WARN band).
    _flat_p99 = float("nan")
    try:
        from invariants_runtime import residual_large_scale_p99_adu  # noqa: PLC0415

        _flat_p99 = float(residual_large_scale_p99_adu(out))
    except Exception:  # noqa: BLE001
        _flat_p99 = float("nan")
    return out, {
        "sky_surface_order": order_i,
        "sky_surface_applied": True,
        "sky_surface_p2p_adu": float(np.nanmax(surf) - np.nanmin(surf)),
        "sky_surface_median_adu": float(np.nanmedian(surf)),
        "sky_surface_n_fit_pixels": int(z_fit.size),
        "sky_surface_fwhm_px": float(fwhm_eff),
        "sky_surface_bg_median_adu": float(bg_median),
        "sky_surface_calm_adu": float(calm_thr),
        "residual_flatness_p99_adu": _flat_p99,
    }


def run_osc_channel_extraction_for_archive(
    *,
    calibrated_lights_root: Path,
    db: VyvarDatabase,
    equipment_id: int,
    app_config: AppConfig | None = None,
    progress_cb: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Extract OSC CFA mosaics into four channel obs-groups; QC+sky-surface per channel."""
    from osc_extract import (
        OSC_CHANNELS,
        channel_obs_group_folder,
        extract_one_light_to_channels,
        is_channel_obs_group_folder,
        is_osc_bayermask,
        iter_mosaic_light_fits,
    )

    cfg = app_config or AppConfig()
    bayermask = db.get_equipment_bayermask(int(equipment_id))
    if not is_osc_bayermask(bayermask):
        return {"skipped": True, "reason": "mono equipment"}

    gain_raw, rn_raw = db.get_equipment_cosmic_params(int(equipment_id))
    gain_e = float(gain_raw) if gain_raw is not None and gain_raw > 0 else 1.0
    rn_e = float(rn_raw) if rn_raw is not None and rn_raw >= 0 else 5.0
    osc_bin = int(cfg.osc_channel_binning)
    root = Path(calibrated_lights_root)
    out: dict[str, Any] = {
        "skipped": False,
        "bayermask": bayermask,
        "osc_bin": osc_bin,
        "groups": [],
        "n_extracted": 0,
        "n_mosaic_removed": 0,
    }

    group_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and not is_channel_obs_group_folder(d.name)],
        key=lambda p: p.name.casefold(),
    )
    total_files = sum(len(iter_mosaic_light_fits(g)) for g in group_dirs)
    done = 0
    for group_dir in group_dirs:
        mosaic_files = iter_mosaic_light_fits(group_dir)
        if not mosaic_files:
            continue
        base_name = group_dir.name
        out_dirs = {
            ch: root / channel_obs_group_folder(base_name, ch) for ch in OSC_CHANNELS
        }
        grp_stat: dict[str, Any] = {"base": base_name, "channels": list(OSC_CHANNELS), "n": 0}
        for fp in mosaic_files:
            done += 1
            if progress_cb is not None:
                progress_cb(done, max(total_files, 1), f"OSC extract {fp.name}")
            extract_one_light_to_channels(
                fp,
                out_dirs=out_dirs,
                bayermask=str(bayermask),
                osc_bin=osc_bin,
                gain_e_per_adu=gain_e,
                read_noise_e=rn_e,
            )
            try:
                fp.unlink()
                out["n_mosaic_removed"] += 1
            except OSError as exc:
                LOGGER.warning("[OSC] failed to remove mosaic %s: %s", fp, exc)
            grp_stat["n"] += 1
            out["n_extracted"] += 1
        for ch in OSC_CHANNELS:
            _qc_enrich_calibrated_in_place(
                out_dirs[ch],
                app_config=cfg,
                progress_cb=progress_cb,
            )
        try:
            from osc_align import merge_osc_qc_metrics_at_lights_root, replicate_qc_verdict_from_one_rggb

            replicate_qc_verdict_from_one_rggb(lights_root=root, base_name=base_name)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[OSC] QC verdict replication failed for %s: %s", base_name, exc)
        try:
            if not any(group_dir.iterdir()):
                group_dir.rmdir()
        except OSError:
            pass
        out["groups"].append(grp_stat)
    try:
        from osc_align import merge_osc_qc_metrics_at_lights_root

        merged_qc = merge_osc_qc_metrics_at_lights_root(root)
        if merged_qc is not None:
            out["qc_metrics_csv"] = str(merged_qc)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[OSC] qc_metrics merge at lights root failed: %s", exc)
    log_event(
        f"OSC extraction: {out['n_extracted']} mosaic(s) -> {len(out['groups'])} base group(s); "
        f"channels={','.join(OSC_CHANNELS)} bin={osc_bin}"
    )
    return out


class SkySurfaceOrderConflictError(RuntimeError):
    """``VY_SKYSF`` present but ``VYSKYORD`` differs from the requested preprocess order."""


def _header_vyskyord(hdr: fits.Header) -> int | None:
    if "VYSKYORD" not in hdr:
        return None
    try:
        return int(hdr["VYSKYORD"])
    except (TypeError, ValueError):
        return None


def _decide_preprocess_sky_action(
    hdr: fits.Header,
    *,
    sky_order: int,
    force_reapply: bool,
) -> str:
    """Return ``apply`` or ``skip``. Raises :class:`SkySurfaceOrderConflictError` on order mismatch."""
    if sky_order <= 0:
        return "skip"
    if force_reapply:
        return "apply"
    if not _header_has_vy_skysf(hdr):
        return "apply"
    existing = _header_vyskyord(hdr)
    if existing is None:
        return "apply"
    if existing == sky_order:
        return "skip"
    raise SkySurfaceOrderConflictError(
        f"VYSKYORD={existing} but preprocess requests order={sky_order}; "
        "recalibration from raw is required to change sky-surface order."
    )


def _qc_enrich_one_frame(
    fp_str: str,
    *,
    sky_order: int,
    force_reapply: bool,
    prefilter_status: str | None,
    target_ra: float | None,
    target_dec: float | None,
    inject_pointing_only_if_missing: bool,
) -> dict[str, Any]:
    """Process one calibrated light frame in-place (picklable worker for parallel QC)."""
    import numpy as np

    from cal_stage import (  # noqa: PLC0415
        compute_skysf_apply_stage,
        stamp_cal_stage_headers,
        verify_fits_datasum,
    )

    fp = Path(fp_str)
    try:
        with fits.open(fp, memmap=False) as hdul:
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            hdr = hdul[0].header.copy()

        sky_stats: dict[str, Any] = {}
        cal_stage_token: str | None = None
        cal_datasum: str | None = None
        cal_pstbg: float | None = None
        skypass: int | None = None
        is_mosaic = _valid_bayerpat_from_header(hdr) is not None and not hdr.get("VY_CHANNEL")
        if sky_order > 0 and not is_mosaic:
            action = _decide_preprocess_sky_action(
                hdr, sky_order=sky_order, force_reapply=force_reapply
            )
            if action == "apply":
                _force_flag = bool(force_reapply and _header_has_vy_skysf(hdr))
                if _force_flag:
                    LOGGER.warning(
                        "preprocess_sky_surface_force_reapply: re-subtracting sky surface on %s",
                        fp.name,
                    )
                data, sky_stats = _fit_subtract_preprocess_sky_surface(data, order=sky_order)
                if _force_flag:
                    sky_stats["sky_surface_force_reapply"] = True
                cal_stage_token, skypass = compute_skysf_apply_stage(
                    hdr,
                    sky_order=sky_order,
                    force_reapply=_force_flag,
                )
                cal_pstbg = float(np.nanmedian(data)) if data.size else None
            else:
                sky_stats = {
                    "sky_surface_applied": False,
                    "sky_surface_skipped": True,
                    "sky_surface_order": sky_order,
                }
                LOGGER.info(
                    "Sky-surface subtract skipped on %s (VY_SKYSF present, VYSKYORD=%s)",
                    fp.name,
                    sky_order,
                )

        qc = _qc_fwhm_elongation(data)
        fwhm = float(qc.get("fwhm_px")) if qc.get("fwhm_px") is not None else float("nan")
        elong = float(qc.get("elongation")) if qc.get("elongation") is not None else float("nan")
        n_stars = int(qc.get("n_stars_detected") or 0)
        status = prefilter_status if prefilter_status else "ok"

        with fits.open(fp, mode="update") as hdul:
            hdul[0].data = _as_fits_float32_image(data)
            hdr = hdul[0].header
            if sky_stats.get("sky_surface_applied"):
                hdr["VY_SKYSF"] = (True, "Sky-surface subtract applied")
                hdr["VYSKYORD"] = (
                    int(sky_stats.get("sky_surface_order") or sky_order),
                    "Preprocess sky-surface polynomial order",
                )
                p2p = sky_stats.get("sky_surface_p2p_adu")
                if p2p is not None and math.isfinite(float(p2p)):
                    hdr["VYSKYP2P"] = (
                        round(float(p2p), 4),
                        "Sky surface peak-to-peak ADU",
                    )
                if cal_stage_token:
                    cal_datasum = stamp_cal_stage_headers(
                        hdr,
                        data,
                        stage=cal_stage_token,
                        pstbg=cal_pstbg,
                        skypass=skypass,
                    )
                    if not verify_fits_datasum(data, cal_datasum):
                        raise RuntimeError(
                            f"INV-CAL-02: VY_CALDATASUM self-check failed on {fp.name}"
                        )
            if math.isfinite(fwhm):
                hdr["VY_FWHM"] = (round(fwhm, 4), "Estimated FWHM [pix]")
            if math.isfinite(elong):
                hdr["VY_ELONG"] = (round(elong, 4), "Estimated elongation (a/b)")
            hdr["VY_NSTAR"] = (n_stars, "Approx. star detections (QC)")
            hdr["VY_QC"] = (status, "QC status")
            # Drop legacy CR provenance if present (L.A.Cosmic removed 2026-08-12).
            if "VY_COSM" in hdr:
                del hdr["VY_COSM"]
            if "VY_COSMNPX" in hdr:
                del hdr["VY_COSMNPX"]
            hdr["VYVARPR"] = (True, "VYVAR pre-processed output")
            if target_ra is not None and target_dec is not None:
                ira = float(target_ra)
                idec = float(target_dec)
                if math.isfinite(ira) and math.isfinite(idec):
                    ex_ra, ex_dec, _ = _pointing_hint_from_header(hdr)
                    do_inject = (not bool(inject_pointing_only_if_missing)) or (
                        ex_ra is None or ex_dec is None
                    )
                    if do_inject:
                        hdr["VYTARGRA"] = (ira, "VYVAR plate-solve hint RA [deg] ICRS")
                        hdr["VYTARGDE"] = (idec, "VYVAR plate-solve hint Dec [deg] ICRS")
                        hdr.add_history("VYVAR: VYTARGRA/VYTARGDE for plate solving (QC in-place)")
            hdul.flush()

        out_row: dict[str, Any] = {
            "src": str(fp),
            "dst": str(fp),
            "status": status,
            "fwhm_px": fwhm,
            "elongation": elong,
            "n_sources": qc.get("n_sources"),
            "n_stars_detected": n_stars,
            "bg_median": float(np.nanmedian(data)) if data.size else None,
            **sky_stats,
        }
        if cal_stage_token and cal_datasum:
            out_row["cal_stage"] = cal_stage_token
            out_row["cal_datasum"] = cal_datasum
            if cal_pstbg is not None:
                out_row["cal_pstbg"] = cal_pstbg
        return out_row
    except SkySurfaceOrderConflictError:
        raise
    except Exception as exc:  # noqa: BLE001
        return {
            "src": str(fp),
            "dst": str(fp),
            "status": f"error: {exc}",
            "fwhm_px": float("nan"),
            "elongation": float("nan"),
            "n_sources": None,
            "n_stars_detected": 0,
            "bg_median": None,
        }


def _infer_raw_light_path_for_calibrated(cal_fp: Path) -> Path | None:
    """Map ``calibrated/lights/...`` frame to matching ``Raw/lights/...`` if present."""
    cal_fp = Path(cal_fp).resolve()
    marker = f"{os.sep}calibrated{os.sep}lights{os.sep}"
    s = str(cal_fp)
    idx = s.find(marker)
    if idx < 0:
        return None
    draft_root = Path(s[:idx])
    rel = Path(s[idx + len(marker) :])
    raw = draft_root / "Raw" / "lights" / rel
    return raw if raw.is_file() else None


def _sync_manifest_cal_stage_from_qc_row(
    row: dict[str, Any],
    *,
    db: VyvarDatabase | None,
    draft_id: int | None,
) -> None:
    if db is None or draft_id is None:
        return
    stage = row.get("cal_stage")
    datasum = row.get("cal_datasum")
    if not stage or not datasum:
        return
    raw = _infer_raw_light_path_for_calibrated(Path(str(row.get("src") or row.get("dst") or "")))
    if raw is None:
        return
    try:
        db.update_obs_file_cal_stage_by_raw_light_path(
            raw,
            draft_id=int(draft_id),
            cal_stage=str(stage),
            cal_datasum=str(datasum),
            cal_pstbg=row.get("cal_pstbg"),
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("manifest cal_stage sync failed after preprocess: %s", exc)


def _qc_enrich_calibrated_in_place(
    calibrated_root: Path,
    *,
    app_config: AppConfig | None = None,
    fwhm_reject_limit: float | None = None,
    elong_reject_limit: float | None = None,
    target_ra: float | None = None,
    target_dec: float | None = None,
    inject_pointing_only_if_missing: bool = True,
    only_paths: Sequence[Path | str] | None = None,
    prefilter_rejected: Mapping[Path | str, str] | None = None,
    progress_cb: Callable[..., None] | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> dict[str, Any]:
    """Lightweight QC pass that writes VY_* headers directly onto calibrated FITS files.

    In-place preprocess QC for ``preprocess_calibrated_to_processed`` (skip-only; no ``processed/`` copy tree).
    Does NOT copy files. Does NOT do temporal sigma clipping.
    Writes: VY_FWHM, VY_ELONG, VY_QC, VY_NSTAR, VYVARPR (and optional VYTARG*) to each FITS.

    Skip-mode frame **selection authority** is the DB DAO-FWHM prefilter passed via
    ``prefilter_rejected`` (status ``rejected_prefilter_*``). Segmentation FWHM /
    elongation are measured and recorded as diagnostics only; they do not drive reject
    statuses here. Alignment gating reads ``qc_metrics.csv`` (QC-01).

    Returns a dict with ``results`` rows compatible with the preprocess QC table.
    """
    import numpy as np

    _ = fwhm_reject_limit, elong_reject_limit  # skip-mode: diagnostics only
    cfg = app_config or AppConfig()
    sky_order = int(cfg.preprocess_sky_surface_order)
    force_reapply = bool(getattr(cfg, "preprocess_sky_surface_force_reapply", False))

    calibrated_root = Path(calibrated_root)
    _prefilter: dict[str, str] = {}
    if prefilter_rejected:
        for fp, reason in prefilter_rejected.items():
            _prefilter[norm_fits_path_key(fp)] = str(reason)

    if prefilter_rejected is not None:
        fits_paths = _iter_light_fits(calibrated_root)
    else:
        fits_paths = _filter_light_paths_maybe(_iter_light_fits(calibrated_root), only_paths)
    results: list[dict[str, Any]] = []
    total = len(fits_paths)
    n_workers = _vyvar_qc_preprocess_workers() if total > 1 else 1
    if n_workers > 1:
        LOGGER.info(
            "QC in-place: parallel_workers=%s (%s frames; qc_preprocess_workers)",
            n_workers,
            total,
        )

    if n_workers > 1 and total > 1:
        with _vyvar_parallel_pool(n_workers) as ex:
            futs = {}
            for fp in fits_paths:
                futs[ex.submit(
                    _qc_enrich_one_frame,
                    str(fp),
                    sky_order=sky_order,
                    force_reapply=force_reapply,
                    prefilter_status=_prefilter.get(norm_fits_path_key(fp)),
                    target_ra=target_ra,
                    target_dec=target_dec,
                    inject_pointing_only_if_missing=inject_pointing_only_if_missing,
                )] = fp
            done = 0
            for fut in as_completed(futs):
                fp = futs[fut]
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"QC in-place {fp.name}")
                try:
                    row = fut.result()
                except SkySurfaceOrderConflictError as exc:
                    raise RuntimeError(f"QC in-place failed for {fp.name}: {exc}") from exc
                results.append(row)
                _sync_manifest_cal_stage_from_qc_row(row, db=db, draft_id=draft_id)
                if str(row.get("status", "")).startswith("error"):
                    log_event(f"QC in-place failed for {fp.name}: {row.get('status')}")
                else:
                    log_event(
                        f"QC in-place {fp.name}: {row.get('status')} "
                        f"FWHM={float(row.get('fwhm_px', float('nan'))):.2f} "
                        f"elong={float(row.get('elongation', float('nan'))):.2f}"
                    )
        results.sort(key=lambda r: str(r.get("src", "")))
    else:
        for i, fp in enumerate(fits_paths, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"QC in-place {fp.name}")
            row = _qc_enrich_one_frame(
                str(fp),
                sky_order=sky_order,
                force_reapply=force_reapply,
                prefilter_status=_prefilter.get(norm_fits_path_key(fp)),
                target_ra=target_ra,
                target_dec=target_dec,
                inject_pointing_only_if_missing=inject_pointing_only_if_missing,
            )
            results.append(row)
            _sync_manifest_cal_stage_from_qc_row(row, db=db, draft_id=draft_id)
            if str(row.get("status", "")).startswith("error"):
                log_event(f"QC in-place failed for {fp.name}: {row.get('status')}")
            else:
                log_event(
                    f"QC in-place {fp.name}: {row.get('status')} "
                    f"FWHM={float(row.get('fwhm_px', float('nan'))):.2f} "
                    f"elong={float(row.get('elongation', float('nan'))):.2f}"
                )

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_rejected = sum(1 for r in results if str(r.get("status", "")).startswith("rejected"))
    sky_surface_skip_count = sum(1 for r in results if r.get("sky_surface_skipped"))
    sky_surface_force_reapply = force_reapply and any(
        r.get("sky_surface_force_reapply") for r in results
    )
    log_event(
        f"QC in-place: {n_ok} ok, {n_rejected} rejected, "
        f"{len(results) - n_ok - n_rejected} errors"
    )
    if sky_surface_skip_count:
        log_event(
            f"QC in-place: sky-surface subtract skipped on {sky_surface_skip_count} frame(s) "
            "(VY_SKYSF guard)"
        )
    if sky_surface_force_reapply:
        log_event(
            "QC in-place: preprocess_sky_surface_force_reapply=True (sky-surface guard bypassed)"
        )

    try:
        from invariants_runtime import check_preprocess_large_small_ratio  # noqa: PLC0415
        from invariants_runtime import inv_check  # noqa: PLC0415

        _prep_inv_meta: dict[str, Any] = {"invariants": []}
        _sample_by_group: dict[str, Path] = {}
        for _r in results:
            if str(_r.get("status") or "") != "ok":
                continue
            _src = Path(str(_r.get("src") or ""))
            if not _src.is_file():
                continue
            _grp = str(_src.parent)
            if _grp not in _sample_by_group:
                _sample_by_group[_grp] = _src
        for _grp, _fp in _sample_by_group.items():
            with fits.open(_fp, memmap=False) as _hdul:
                _frame = np.asarray(_hdul[0].data, dtype=np.float32)
            _ok_prep, _det_prep, _ratio_prep = check_preprocess_large_small_ratio(_frame)
            inv_check(
                _prep_inv_meta,
                "INV-PREP-01",
                _ok_prep,
                policy="WARN",
                detail=f"{Path(_grp).name}: {_det_prep}",
            )
            if math.isfinite(_ratio_prep):
                _prep_guard_msg = (
                    f"INV-PREP-01 Preprocess gradient guard ({Path(_grp).name}): {_det_prep}"
                )
                LOGGER.info(_prep_guard_msg)
                log_event(_prep_guard_msg)
    except Exception as _prep_inv_exc:  # noqa: BLE001
        LOGGER.debug("[INV-PREP-01] skipped: %s", _prep_inv_exc)

    _qc_df = pd.DataFrame(results)
    _qc_csv = calibrated_root / "qc_metrics.csv"
    try:
        _qc_df.to_csv(_qc_csv, index=False)
        log_event(f"qc_metrics.csv written to {_qc_csv}")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PIPELINE] qc_metrics.csv write failed: %s", exc)

    return {
        "n_processed": len(results),
        "n_ok": n_ok,
        "n_rejected": n_rejected,
        "sky_surface_skip_count": sky_surface_skip_count,
        "sky_surface_force_reapply": sky_surface_force_reapply,
        "results": results,
        "qc_root": str(calibrated_root),
        "qc_csv": str(_qc_csv),
    }


def scan_calibrated_lights_pointing(
    calibrated_root: Path | str,
    *,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Summarize celestial WCS vs object-style pointing keywords under ``calibrated_root`` (lights).

    If ``max_files`` is None, all light FITS under the tree are scanned.
    """
    from astropy.wcs import WCS

    root = Path(calibrated_root)
    files = _iter_light_fits(root)
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    rows: list[dict[str, Any]] = []
    n_wcs = 0
    n_hint = 0
    n_missing = 0

    def _wcs_center_from_header(h: fits.Header) -> tuple[float | None, float | None]:
        try:
            if not _has_valid_wcs(h):
                return None, None
            nax1 = int(h.get("NAXIS1") or 0)
            nax2 = int(h.get("NAXIS2") or 0)
            if nax1 <= 0 or nax2 <= 0:
                return None, None
            w = WCS(h, relax=True)
            cx = 0.5 * float(nax1)
            cy = 0.5 * float(nax2)
            ra, dec = w.celestial.all_pix2world([cx], [cy], 0)
            ra0 = float(ra[0]) % 360.0
            de0 = float(dec[0])
            if math.isfinite(ra0) and math.isfinite(de0) and (-90.0 <= de0 <= 90.0):
                return ra0, de0
        except Exception:  # noqa: BLE001
            # EXC-0442: T2 -- JD parse helper returns `None`; WCS change-point logic excludes frame from segment list. (EXCEPT-BULK-2 2026-07-08)
            return None, None
        return None, None

    def _jd_from_header(h: fits.Header) -> float | None:
        try:
            meta = fits_metadata_from_primary_header(h)
            jd = meta.get("jd_start")
            if jd is None:
                return None
            jd_f = float(jd)
            if math.isfinite(jd_f) and jd_f > 0:
                return jd_f
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0443] Multi-pointing segment detector returns `None` on any internal error; caller treats as ...: %s', exc)
            return None
        return None

    def _detect_segments_from_wcs_centers(
        _rows: list[dict[str, Any]],
        *,
        break_arcmin: float = 10.0,
        min_segment_size: int = 12,
    ) -> dict[str, Any] | None:
        try:
            import numpy as np
            from astropy.coordinates import SkyCoord
            import astropy.units as u

            pts: list[tuple[int, float, float, float]] = []
            for idx, r in enumerate(_rows):
                ra = r.get("wcs_center_ra_deg")
                de = r.get("wcs_center_dec_deg")
                jd = r.get("jd")
                if ra is None or de is None or jd is None:
                    continue
                ra_f = float(ra)
                de_f = float(de)
                jd_f = float(jd)
                if math.isfinite(ra_f) and math.isfinite(de_f) and math.isfinite(jd_f) and jd_f > 0:
                    pts.append((idx, jd_f, ra_f, de_f))
            if len(pts) < max(20, 2 * int(min_segment_size)):
                return None
            pts = sorted(pts, key=lambda t: t[1])
            coords = [SkyCoord(ra=ra * u.deg, dec=de * u.deg, frame="icrs") for _, _, ra, de in pts]
            break_positions: list[int] = []
            seps_arcmin: list[float] = []
            for i in range(1, len(coords)):
                s = float(coords[i - 1].separation(coords[i]).arcminute)
                seps_arcmin.append(s)
                if math.isfinite(s) and s >= float(break_arcmin):
                    break_positions.append(i)
            if not break_positions:
                return None
            cuts = [0] + break_positions + [len(pts)]
            segments: list[dict[str, Any]] = []
            for a, b in itertools.pairwise(cuts):
                seg = pts[a:b]
                if len(seg) < int(min_segment_size):
                    continue
                ras = np.asarray([p[2] for p in seg], dtype=np.float64)
                des = np.asarray([p[3] for p in seg], dtype=np.float64)
                ang = np.deg2rad(ras)
                x = np.nanmedian(np.cos(ang))
                y = np.nanmedian(np.sin(ang))
                ra_med = float((math.degrees(math.atan2(y, x)) + 360.0) % 360.0)
                de_med = float(np.nanmedian(des))
                segments.append(
                    {
                        "segment_id": int(len(segments)),
                        "n": int(len(seg)),
                        "jd_min": float(min(p[1] for p in seg)),
                        "jd_max": float(max(p[1] for p in seg)),
                        "median_ra_deg": ra_med,
                        "median_dec_deg": de_med,
                        "member_row_indices": [int(p[0]) for p in seg],
                    }
                )
            if len(segments) < 2:
                return None
            return {
                "detected": True,
                "method": "wcs_center_change_point",
                "break_arcmin": float(break_arcmin),
                "min_segment_size": int(min_segment_size),
                "n_points_used": int(len(pts)),
                "breaks_on_sorted_points": [int(x) for x in break_positions],
                "segments": segments,
                "sep_arcmin_max": float(max(seps_arcmin)) if seps_arcmin else None,
            }
        except Exception:  # noqa: BLE001
            return None
    for fp in files:
        with fits.open(fp, memmap=False) as hdul:
            h = hdul[0].header
        hwcs = _has_valid_wcs(h)
        ha, hd, hs = _pointing_hint_from_header(h)
        wra, wde = _wcs_center_from_header(h)
        jd = _jd_from_header(h)
        da, dd, ds = ha, hd, hs
        if hwcs:
            n_wcs += 1
        elif ha is not None and hd is not None:
            n_hint += 1
        else:
            n_missing += 1
        rows.append(
            {
                "file": fp.name,
                "has_celestial_wcs": hwcs,
                "hint_ra_deg": ha,
                "hint_dec_deg": hd,
                "hint_source": hs,
                "wcs_center_ra_deg": wra,
                "wcs_center_dec_deg": wde,
                "jd": jd,
                "display_ra_deg": da,
                "display_dec_deg": dd,
                "display_source": ds,
            }
        )
    seg = _detect_segments_from_wcs_centers(rows)
    return {
        "calibrated_root": str(root.resolve()),
        "n_files_scanned": len(rows),
        "n_has_celestial_wcs": n_wcs,
        "n_has_object_hint_no_wcs": n_hint,
        "n_no_pointing_hint": n_missing,
        "rows": rows,
        "pointing_segments": seg,
    }


