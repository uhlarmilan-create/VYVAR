"""Moved from pipeline.py (CONSOLIDATE-01E5). Facade re-exports these names.

AstroPipeline stays in pipeline.py (C-C). Spawn-worker globals, initializer,
and `_calibrate_batch_process_one` live here so `global` binds one namespace.
"""
from __future__ import annotations

import itertools
import logging
import math
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from config import AppConfig, resolve_detection_threshold_sigma
from database import VyvarDatabase
from calibration import (
    filter_light_paths_for_calibration_db,
    get_processed_master,
)
from cal_diag import (
    CalDiagGateResult,
    CalDiagSession,
    apply_cal_diag_headers,
    convention_to_dark_mode,
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
from infolog import log_event
from pipeline_constants import (
    N_STARS_DIAG_K,
    SAT_LIMIT_CONTAINER_CLIP_ADU,
    _N_STARS_DIAG_MAD_TO_SIGMA,
)
from plain_stats import plain_mean_med_std
from utils import (
    set_float_image_data_in_place,
    DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    fits_binning_xy_from_header,
    fits_header_has_celestial_wcs,
    iter_fits_paths_recursive as _iter_fits_recursive,
)
from vyvar_alignment_frame import _as_fits_float32_image
from fits_meta import pointing_hint_from_header as _pointing_hint_from_header

# Same named logger as pipeline.LOGGER (logging.getLogger singleton). Avoids
# pipeline -> pipeline_calibrate -> pipeline at module load (spawn children
# import this module first).
LOGGER = logging.getLogger("pipeline")

#: DAOStarFinder kernel FWHM hint (px) for the post-calibration QC: a star-scale default.
#: Not derived from segmentation (that preferred hot pixels / cosmics and pulled it low).
_DAO_FWHM_GUESS_PX = 4.5

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
        log_event(f"Calibration - write OK: {calibrated_root.resolve()}")
    except OSError as exc:
        log_event(f"Calibration - WRITE PERMISSION ERROR (calibrated): {calibrated_root.resolve()} -> {exc}")

    if master_dark_path is not None:
        md = Path(master_dark_path)
        md_r = md.resolve()
        ok = md_r.is_file()
        log_event(f"MasterDark: {md_r} (exists={ok})")
        if not ok:
            log_event("MasterDark: file does not exist - dark is not applied (flat-only / copy-only per settings).")

    for fk, fp in sorted((masterflat_by_filter or {}).items(), key=lambda x: str(x[0])):
        if fp is None:
            log_event(f"MasterFlat[{fk!r}]: (no path)")
            continue
        p = Path(fp)
        pr = p.resolve()
        ok = pr.is_file()
        log_event(f"MasterFlat[{fk!r}]: {pr} (exists={ok})")
        if not ok:
            log_event(f"MasterFlat[{fk!r}]: file does not exist - no flat is applied for this filter.")


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


def _quality_inspection_dao_metrics_array(
    data: "np.ndarray",
    hdr: fits.Header,
) -> dict[str, Any]:
    """Per-frame QC metrics on an in-memory calibrated image (QC-METRICS-2026-10).

    Measured on the central crop by :func:`frame_metrics.measure_frame`: pixel noise from
    neighbour differences, FWHM0 from half-maximum radii, DAO detection at a fixed 5 sigma,
    sky outside a star mask, FWHM / elongation from elliptical Gaussian fits, and a star
    list (x, y, aperture flux) for the transparency step in ``frame_qc``.
    """
    import numpy as np

    from frame_metrics import measure_frame  # noqa: PLC0415

    out: dict[str, Any] = {
        "fwhm_mean": None,
        "sky_background": None,
        "star_count": 0,
        "inspection_jd": _inspection_jd_from_header(hdr),
        "exposure_sec": _exposure_sec_from_header(hdr),
        "elongation_mean": None,
        "noise_sigma": None,
        "qc_stars": [],
        "qc_shape": None,
    }
    _pra, _pde, _ = _pointing_hint_from_header(hdr)
    out["ra_deg"] = float(_pra) if _pra is not None and math.isfinite(float(_pra)) else None
    out["de_deg"] = float(_pde) if _pde is not None and math.isfinite(float(_pde)) else None
    try:
        arr = np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        return {**out, "error": str(exc)}
    m = measure_frame(_qc_center_crop_for_stars(arr))
    out.update({k: m.get(k) for k in (
        "fwhm_mean", "sky_background", "elongation_mean", "noise_sigma", "qc_stars", "qc_shape", "fwhm0", "n_fit",
    )})
    out["star_count"] = int(m.get("star_count") or 0)
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


def _take_qc_star_list(lists: dict[str, Any], file_path: str, qc: dict[str, Any]) -> None:
    """Collect the star list (x, y, flux) of the QC crop for the transparency step (frame_qc)."""
    stars = qc.pop("qc_stars", None)
    if file_path and stars:
        lists[file_path] = (stars, qc.get("qc_shape"))


def _save_qc_star_lists(ap: Path, lists: dict[str, Any]) -> None:
    """One draft-level file ``qc/qc_stars.json`` (OUTPUT-REVIEW-2026-10)."""
    try:
        from frame_qc import save_qc_stars  # noqa: PLC0415

        save_qc_stars(ap, lists)
    except OSError as exc:
        logging.warning("[QC] star lists not saved: %s", exc)


def apply_perf10_dao_qc_to_obs_files(
    *,
    db: VyvarDatabase,
    draft_id: int,
    archive_path: Path | str,
    perf10_qc_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Write calibration-time DAO QC to ``manifest files[]`` (same columns as RAM QC step 5)."""
    import numpy as np

    ap = Path(archive_path).expanduser()

    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    fwhm_by_id: dict[int, float] = {}
    n_updated = 0
    errors: list[str] = []
    _qc_star_lists: dict[str, Any] = {}

    for row in rows:
        rid = int(row["ID"])
        qc = _perf10_lookup_qc(perf10_qc_results, ap, str(row.get("FILE_PATH") or ""))
        if qc is None or qc.get("error"):
            continue
        try:
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
                elongation_mean=_el_db,
                noise_sigma=qc.get("noise_sigma"),
            )
            _take_qc_star_list(_qc_star_lists, str(row.get("FILE_PATH") or ""), qc)
            n_updated += 1
            fv = qc.get("fwhm_mean")
            if fv is not None and math.isfinite(float(fv)):
                fwhm_by_id[rid] = float(fv)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{row.get('FILE_PATH')}: {exc}")

    _save_qc_star_lists(ap, _qc_star_lists)
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
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Calibrate each draft light **in RAM only**, DAO metrics -> ``manifest files[]``; FWHM x1.5 reject.

    No calibrated FITS are written. Uses the same master selection rules as :func:`calibrate_lights_to_calibrated`.
    After QC, :func:`sync_obs_files_drift_arcmin_for_draft` writes ``DRIFT`` / ``DRIFT_DRA`` / ``DRIFT_DDE``.
    """
    import numpy as np

    ap = Path(archive_path).expanduser()
    db_cal = _db_for_calibration_tasks(None)

    mf_merged: dict[str, Path | None] = {}
    for k, v in (masterflat_by_filter or {}).items():
        mf_merged[str(k)] = None if v is None else Path(v)
    for k, v in (masterflat_by_obs_key or {}).items():
        mf_merged[str(k)] = None if v is None or str(v).strip() == "" else Path(v)

    md_path_ok: Path | None = None
    if master_dark_path is not None and Path(master_dark_path).exists():
        md_path_ok = Path(master_dark_path)

    _native_b = None  # CAL-2026-10: master binning from each master header

    flat_cache: dict[str, Any] = {}
    flat_median_scale: dict[str, float] = {}
    dark_cache: dict[tuple[Any, ...], Any] = {}
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    n = len(rows)
    fwhm_by_id: dict[int, float] = {}
    errors: list[str] = []
    fov_sample_deg: float | None = None
    _qc_star_lists: dict[str, Any] = {}

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

        try:
            data, hdr, _ud, _uf = _calibrate_one_light_apply_masters_in_ram(
                src=raw_fp,
                master_dark_path=md_use,
                masterflat_by_filter=mf_merged,
                flat_cache=flat_cache,
                flat_median_scale=flat_median_scale,
                dark_cache=dark_cache,
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
        _take_qc_star_list(_qc_star_lists, str(row.get("FILE_PATH") or ""), m)
        if fov_sample_deg is None:
            fov_sample_deg = _estimate_fov_deg_from_header(hdr)
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
            elongation_mean=_el_db,
            noise_sigma=m.get("noise_sigma"),
        )
        fv = m.get("fwhm_mean")
        if fv is not None and math.isfinite(float(fv)):
            fwhm_by_id[rid] = float(fv)
        if progress_cb is not None:
            progress_cb(i, n, f"QC RAM {raw_fp.name}")

    _save_qc_star_lists(ap, _qc_star_lists)
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


def _log_calibration_metadata_diagnostic(filename: str, metadata: dict[str, Any]) -> None:
    log_event("--- CALIBRATION METADATA DIAGNOSTICS ---")
    log_event(f"File: {filename}")
    log_event(f"FOCAL (from DB/FITS): {metadata.get('focal_length')} mm")
    log_event(f"PIXEL_SIZE (raw): {metadata.get('pixel_size_raw')} um")
    _n1 = metadata.get("naxis1")
    _n2 = metadata.get("naxis2")
    if _n1 or _n2:
        log_event(f"NAXIS: {_n1}x{_n2}")
    _rbx = metadata.get("fits_xbinning_raw")
    _rby = metadata.get("fits_ybinning_raw")
    if _rbx is not None or _rby is not None:
        log_event(f"BINNING (FITS raw): XBINNING={_rbx!r} YBINNING={_rby!r}")
    log_event(f"BINNING (X/Y): {metadata.get('binning')}x{metadata.get('binning_y')}")
    log_event(f"EFFECTIVE PIXEL (for computation): {metadata.get('pixel_um')} um")
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
    # EQUIPMENTS.SATURATE_ADU may be NULL, headers often omit
    # SATURATE/MAXLIN. Silent None made peak>limit comparisons False for the whole catalog.
    logging.warning(
        "[INV-SAT-LIMIT] saturation clip unresolved (header/equipment/BITPIX none); "
        "using GAIN-DOMAIN-01 container clip %.0f ADU",
        SAT_LIMIT_CONTAINER_CLIP_ADU,
    )
    return float(SAT_LIMIT_CONTAINER_CLIP_ADU), "conservative_default_container_clip_65535"


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
    """Post-calibration QC settings + DB linkage (``manifest files[]`` update by raw light path)."""
    en = bool(cfg.qc_after_calibrate_enabled)
    _dao = resolve_detection_threshold_sigma(cfg.detection_threshold_sigma)
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
        "dao_detection_sigma": _dao,
        "db_path": str(Path(cfg.database_path).resolve()) if en else None,
        "draft_id": draft_id,
        "observation_id": observation_id,
        "id_equipments": id_equipments,
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
    _, med, std = plain_mean_med_std(img[finite])
    std = float(std)
    if not math.isfinite(std) or std <= 0:
        return None, 0, None
    img2 = np.asarray(img - float(med), dtype=np.float32)
    img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.nanmedian(img2)) < 0:
        img2 = -img2
    fwhm_guess = _DAO_FWHM_GUESS_PX
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
    dao_detection_sigma: float,
) -> dict[str, Any]:
    """Sky stats on full frame; star HFR/count on central crop if frame is large (measurements only,
    no pass/fail limits - CFG-REVIEW-2026-10)."""
    import numpy as np

    img = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return {"hfr": None, "n_stars": 0, "sky_mean": None, "sky_median": None, "sky_rms": None}

    sky_mean, sky_med, sky_rms = plain_mean_med_std(img[finite])
    sky_mean = float(sky_mean)
    sky_med = float(sky_med)
    sky_rms = float(sky_rms)

    crop = _qc_center_crop_for_stars(img, 1000)
    _ds = float(dao_detection_sigma)
    if not math.isfinite(_ds) or _ds <= 0:
        _ds = 5.0
    hfr_m, n_star, _fwhm_guess = _mean_hfr_bright_stars_dao(crop, max_stars=50, dao_detection_sigma=_ds)

    return {
        "hfr": hfr_m,
        "n_stars": int(n_star),
        "sky_mean": sky_mean,
        "sky_median": sky_med,
        "sky_rms": sky_rms,
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
    dark_cache: dict[tuple[Any, ...], Any] | None = None,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    calibration_master_native_binning: int | None | object = _CALIB_MASTER_NB_UNSET,
    cal_diag_gate_result: CalDiagGateResult | None = None,
    flat_offsets: dict[str, float] | None = None,
) -> tuple[Any, fits.Header, bool, bool]:
    """Apply dark/flat in RAM; return ``(data_float32, header, used_dark, used_flat)`` (no disk write).

    ``calibration_master_native_binning`` defaults to :data:`calibration.CALIBRATION_LIBRARY_NATIVE_BINNING`
    (CalibrationLibrary stores native masters; resample in RAM to match light ``XBINNING``).
    Pass ``None`` to read ``XBINNING`` from each master FITS.

    ``dark_cache`` (SPEED-2026-10): the processed master dark per (path, binning, shape, resample
    mode) is read and resampled once per run instead of once per frame (same array).
    """
    import numpy as np

    if calibration_master_native_binning is _CALIB_MASTER_NB_UNSET:
        _mb_lib = None  # CAL-2026-10: master binning from each master header
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
    hdr["VY_CLBX"] = (int(light_bx), "Light XBINNING for master matching/resample")
    hdr["VY_CLBY"] = (int(light_by), "Light YBINNING (diagnostic)")
    if _mb_lib is None:
        hdr["VY_MBNC"] = (
            -1,
            "Native XBINNING of the library masters",
        )
    else:
        hdr["VY_MBNC"] = (
            int(_mb_lib),
            "Assumed native XBINNING of library masters",
        )

    md_data: np.ndarray | None = None
    if master_dark_path is not None and master_dark_path.exists():
        _drm = "sum"
        if cal_diag_gate_result is not None:
            _drm = convention_to_dark_mode(cal_diag_gate_result.convention)
        _dk = (str(Path(master_dark_path).resolve()), int(light_bx), tuple(data.shape), str(_drm), _mb_lib)
        pm = dark_cache.get(_dk) if dark_cache is not None else None
        if pm is None:
            pm = get_processed_master(
                master_dark_path,
                light_bx,
                kind="dark",
                master_binning=_mb_lib,
                light_shape=data.shape,
                light_filename=src.name,
                dark_resample_mode=_drm,
            )
            if dark_cache is not None:
                dark_cache[_dk] = pm
            if pm.resampled:
                log_event(
                    f"Calibration: library master native {pm.master_binning}x{pm.master_binning} -> "
                    f"resampled (RAM) to {light_bx}x{light_bx} (first Light [{src.name}])"
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
    hdr["VY_OBSG"] = (_obs_k, "Observation group FILTER|EXPTIME|XBINNING")
    mf_path = None
    if masterflat_by_filter:
        c_obs = masterflat_by_filter.get(_obs_k)
        if c_obs is not None:
            p_obs = Path(c_obs)
            if p_obs.is_file():
                mf_path = p_obs
    # CAL-2026-10: when the plan has decided this observation group (key present, even
    # with None = no valid flat), never fall back to another flat. Without a plan entry,
    # only an exact filter match is allowed - never the "NoFilter" flat on a filtered LIGHT.
    if mf_path is None and masterflat_by_filter and _obs_k not in masterflat_by_filter:
        c_f = masterflat_by_filter.get(flt)
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
                flat_offset_adu=(flat_offsets or {}).get(str(Path(mf_path).resolve())),
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
        log_event("Flat skipped because no Dark was subtracted (offset would remain in the LIGHT).")

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
        "DF full, D dark only, FS no flat, P raw",
    )
    hdr["VY_CALIB"] = (_calibration_type_from_flags(_flags), "Calibration mode")
    hdr["VY_CALST"] = (
        int(_vy_calib_status_numeric(_flags)),
        "Calib status: 2 DF, 1 dark only, 0 raw",
    )
    if _flags == "FS":
        hdr["VY_WARN"] = (
            True,
            "Flat not applied: no dark (over-correction)",
        )
    if master_dark_path is not None:
        # LOG-NOISE-2026-10: a long file name leaves no room for a comment on the 80-character card
        hdr["VY_MDP"] = str(master_dark_path.name)[:68]
    if mf_path is not None:
        hdr["VY_MFP"] = str(Path(mf_path).name)[:68]
    if used_flat:
        try:
            key_m = f"{flt}|{mf_path!s}|lb{light_bx}"
            hdr["VY_FLATM"] = (
                float(fms[key_m]),
                "Median ADU of master flat before normalising",
            )
            hdr["VY_FLFL"] = (
                float(flat_norm_floor),
                "Min normalised flat value before division",
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
    dark_cache: dict[tuple[Any, ...], Any] | None = None,
    db: VyvarDatabase | None = None,
    qc_pack: dict[str, Any] | None = None,
    calibration_master_native_binning: int | None | object = _CALIB_MASTER_NB_UNSET,
    cal_diag_gate_result: CalDiagGateResult | None = None,
    db_updates: list[tuple[str, dict[str, Any]]] | None = None,
    flat_offsets: dict[str, float] | None = None,
) -> tuple[bool, bool, dict[str, Any] | None, str, dict[str, Any] | None]:
    """Apply master dark / flat to one light FITS and write ``dst``.

    Uses ``fits.open(..., memmap=False)`` and ``with`` blocks so BZERO/BSCALE frames load reliably and
    file handles close after each read (arrays are copied into RAM before processing).

    Returns ``(used_dark, used_flat, qc_summary, vy_cflag, perf10_qc)`` where ``qc_summary`` is set when
    post-calibration QC ran; ``perf10_qc`` holds the DAO inspection metrics (PERF-10);
    ``vy_cflag`` matches ``VY_CFLAG`` written to the FITS (see calibration decision table).

    ``db_updates`` (SPEED-2026-10): when a list is given, the manifest / DB updates of this frame are
    appended to it (``(kind, kwargs)``, applied by :func:`_apply_cal_frame_db_updates` in the run
    process) instead of written here - worker processes never write the draft manifest.
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
        dark_cache=dark_cache,
        db=db,
        id_equipments=_id_eq,
        calibration_master_native_binning=calibration_master_native_binning,
        cal_diag_gate_result=cal_diag_gate_result,
        flat_offsets=flat_offsets,
    )

    qc_summary: dict[str, Any] | None = None
    _osc_mosaic = _valid_bayerpat_from_header(hdr) is not None
    if (used_dark or used_flat) and qc_pack and qc_pack.get("enabled") and not _osc_mosaic:
        qc_summary = _post_calibration_qc_eval(
            np.asarray(data, dtype=np.float32),
            dao_detection_sigma=float(qc_pack.get("dao_detection_sigma", 5.0)),
        )
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
            _upd_qc = {
                "draft_id": qc_pack.get("draft_id"),
                "qc_hfr": qc_summary.get("hfr"),
                "qc_stars": int(qc_summary["n_stars"]),
                "qc_background": qc_summary.get("sky_median"),
                "qc_bg_rms": qc_summary.get("sky_rms"),
            }
            if db_updates is not None:
                db_updates.append(("qc", _upd_qc))
            else:
                try:
                    db_q = db if db is not None else _db_for_calibration_tasks(qc_pack)
                    if db_q is not None:
                        db_q.update_obs_file_qc_by_raw_light_path(src, **_upd_qc)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("manifest files[] QC update failed: %s", exc)

    perf10_qc: dict[str, Any] | None = None
    if not _osc_mosaic and qc_pack:
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
        _upd_cs = {"draft_id": int(qc_pack["draft_id"]), "cal_stage": cal_stage_token, "cal_datasum": cal_datasum}
        if db_updates is not None:
            db_updates.append(("cal_stage", _upd_cs))
        else:
            try:
                db_cs = db if db is not None else _db_for_calibration_tasks(qc_pack)
                if db_cs is not None:
                    db_cs.update_obs_file_cal_stage_by_raw_light_path(src, **_upd_cs)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("manifest cal_stage sync failed after calibrate: %s", exc)
    return used_dark, used_flat, qc_summary, _hdr_vy_cflag_str(hdr), perf10_qc


def _apply_cal_frame_db_updates(
    db: VyvarDatabase | None,
    qc_pack: dict[str, Any] | None,
    src: Path,
    updates: list[tuple[str, dict[str, Any]]],
) -> None:
    """Write one frame's deferred manifest / DB updates (same calls and order as inline)."""
    for kind, kw in updates:
        try:
            dbx = db if db is not None else _db_for_calibration_tasks(qc_pack)
            if dbx is None:
                continue
            if kind == "qc":
                dbx.update_obs_file_qc_by_raw_light_path(src, **kw)
            elif kind == "cal_stage":
                dbx.update_obs_file_cal_stage_by_raw_light_path(src, **kw)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("manifest files[] %s update failed: %s", kind, exc)


_CAL_FRAME_STATE: dict[str, Any] = {}


def _init_calibrate_frame_worker(ctx: dict[str, Any]) -> None:
    """Per-process state for :func:`_calibrate_frame_task` (SPEED-2026-10).

    ``ctx``: ``master_dark_path``, ``master_dark_by_obs_key``, ``masterflat_by_filter`` (str paths),
    ``qc_pack``, ``native_b`` and either the live ``session`` (run process) or ``cal_diag_blob``
    (worker process). Flat and dark masters are cached per process.
    """
    _CAL_FRAME_STATE.clear()
    _CAL_FRAME_STATE.update(ctx)
    if _CAL_FRAME_STATE.get("session") is None:
        _CAL_FRAME_STATE["session"] = _cal_diag_session_from_export(ctx.get("cal_diag_blob"))
    _CAL_FRAME_STATE["flat_cache"] = {}
    _CAL_FRAME_STATE["flat_median"] = {}
    _CAL_FRAME_STATE["dark_cache"] = {}


def _calibrate_frame_task(item: tuple[str, str]) -> dict[str, Any]:
    """Calibrate one LIGHT (the one path for any number of workers).

    Returns ``ok``, ``skipped``, ``used_dark``, ``used_flat``, ``qc_summary``, ``vy_cflag``,
    ``perf10_qc``, ``db_updates`` and the infolog lines (``events``); the run process applies the
    DB updates and writes the lines in frame order.
    """
    from infolog import capture_events  # noqa: PLC0415

    src_s, dst_s = item
    st = _CAL_FRAME_STATE
    src = Path(src_s)
    dst = Path(dst_s)
    with capture_events() as events:
        try:
            with fits.open(src, memmap=False) as hdul:
                hdr_l = hdul[0].header
                _ok = observation_group_key_from_metadata(fits_metadata_from_primary_header(hdr_l))
                light_bx, _ = fits_binning_xy_from_header(hdr_l)
            session = st.get("session")
            if session is not None and is_obs_group_aborted(session, _ok):
                if dst.exists():
                    try:
                        dst.unlink()
                    except OSError:
                        pass
                return {"ok": True, "skipped": True, "events": list(events)}
            md_s = st.get("master_dark_path")
            md_use: Path | None = Path(md_s) if md_s else None
            _alt = (st.get("master_dark_by_obs_key") or {}).get(_ok)
            if _alt is not None and str(_alt).strip() != "" and Path(str(_alt)).is_file():
                md_use = Path(str(_alt))
            gr = None
            if session is not None:
                gr = gate_result_for_frame(session, obs_group_key=_ok, dark_path=md_use, light_binning=light_bx)
            mf = {str(k): (Path(v) if v else None) for k, v in (st.get("masterflat_by_filter") or {}).items()}
            updates: list[tuple[str, dict[str, Any]]] = []
            used_dark, used_flat, qc_sum, flags, perf10_qc = _calibrate_one_light_disk(
                src=src,
                dst=dst,
                master_dark_path=md_use,
                masterflat_by_filter=mf,
                flat_cache=st["flat_cache"],
                flat_median_scale=st["flat_median"],
                dark_cache=st["dark_cache"],
                db=st.get("db"),
                qc_pack=st.get("qc_pack"),
                calibration_master_native_binning=st.get("native_b"),
                cal_diag_gate_result=gr,
                db_updates=updates,
                flat_offsets=st.get("flat_offsets"),
            )
            return {
                "ok": True,
                "skipped": False,
                "used_dark": bool(used_dark),
                "used_flat": bool(used_flat),
                "qc_summary": qc_sum,
                "vy_cflag": flags,
                "perf10_qc": perf10_qc,
                "db_updates": updates,
                "events": list(events),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "exc_type": type(exc).__name__,
                    "traceback": traceback.format_exc(), "events": list(events)}


def _flat_offsets_for_masters(paths: Any, db: VyvarDatabase | None) -> dict[str, float]:
    """Offset to remove from each FLAT master (resolved path -> ADU), looked up once in the run
    process with the run's database (same lookup ``get_processed_master`` makes)."""
    from calibration import flat_offset_for_master  # noqa: PLC0415

    out: dict[str, float] = {}
    for p in paths:
        if not p:
            continue
        key = str(Path(p).resolve())
        if key in out or not Path(key).is_file():
            continue
        try:
            out[key] = flat_offset_for_master(key, db)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[CAL] flat offset lookup failed for %s: %s", key, exc)
    return out


def _has_usable_master_dark(path: Path | None) -> bool:
    return bool(path is not None and Path(path).is_file())


def _passthrough_lights_to_calibrated(
    *,
    lights_root: Path,
    calibrated_root: Path,
    progress_cb: "callable | None" = None,
    database_path: Path | None = None,
    draft_id: int | None = None,
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
            hdr["VY_CFLAG"] = ("P", "DF full, D dark only, FS no flat, P raw")
            hdr["VY_CALIB"] = ("PASSTHROUGH", "Calibration mode")
            hdr["VY_CALST"] = (0, "Calib status: 2 DF, 1 dark only, 0 raw")
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
                        is_calibrated=0,
                        calib_type="PASSTHROUGH",
                        calib_flags="P",
                    )
                    db_pt.update_obs_file_cal_stage_by_raw_light_path(
                        src,
                        draft_id=draft_id,
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

    Parallelism (SPEED-2026-10): ``max_workers`` or the pipeline's worker count, capped by free RAM; the
    same per-frame function runs in every worker and the run process alone writes the draft manifest.
    """
    cfg = pipeline_config or AppConfig()
    qc_pack = _qc_pack_from_config(cfg, draft_id=draft_id, observation_id=observation_id)

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
        )

    md_path_ok: Path | None = None
    md_init_str: str | None = None
    if master_dark_path is not None and master_dark_path.exists():
        md_path_ok = master_dark_path
        md_init_str = str(master_dark_path.resolve())

    _native_b = None  # CAL-2026-10: master binning from each master header

    mf_serial: dict[str, str | None] = {}
    for k, v in (masterflat_by_filter or {}).items():
        if v is None:
            mf_serial[str(k)] = None
        else:
            mf_serial[str(k)] = str(Path(v).resolve())

    stats: dict[str, Any] = {
        "processed": 0,
        "used_dark": 0,
        "used_flat": 0,
        "copied_only": 0,
        "errors": 0,
        "calibrate_workers": 1,
        "qc_evaluated": 0,
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
            f"Calibration - excluded {_n_before_obs_filter - total} files per manifest files[] (IS_REJECTED=1 or not in DB)"
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
    log_event("Calibration - input files: " +" | ".join(_log_bits))

    if total > 0:
        log_lights_binning_from_headers_preflight(files, context="Calibration")
        _diag_light = _pick_light_for_metadata_diagnostic(files)
        try:
            _db_cal_meta = VyvarDatabase(Path(cfg.database_path))
            try:
                _meta0 = extract_fits_metadata(
                    _diag_light,
                    db=_db_cal_meta,
                    id_equipment=equipment_id,
                    draft_id=draft_id,
                )
                _log_calibration_metadata_diagnostic(_diag_light.name, _meta0)
                stats["applied_focal_length"] = _meta0.get("focal_length")
                stats["applied_pixel_size"] = _meta0.get("pixel_um")
            finally:
                _db_cal_meta.conn.close()
        except Exception as exc:  # noqa: BLE001
            log_event(f"CALIBRATION DIAGNOSTICS: metadata of the first file failed: {exc!s}")

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
        )
    stats["cal_diag_aborted_groups"] = len(cal_diag_session.aborted_groups)

    # SPEED-2026-10: one per-frame function for any number of workers; the run process writes the
    # manifest / DB updates and the infolog lines in frame order (workers never touch the manifest).
    pairs = [(src, calibrated_root / src.relative_to(lights_root)) for src in files]
    ctx: dict[str, Any] = {
        "master_dark_path": md_init_str,
        "master_dark_by_obs_key": {
            str(k): (str(Path(v).resolve()) if v is not None and str(v).strip() != "" else None)
            for k, v in (master_dark_by_obs_key or {}).items()
        },
        "masterflat_by_filter": mf_serial,
        "qc_pack": qc_pack,
        "native_b": _native_b,
        "flat_offsets": _flat_offsets_for_masters(mf_serial.values(), db_main),
    }
    from parallel_frames import frame_bytes_of_fits, map_frames, stage_workers  # noqa: PLC0415

    nw = stage_workers(total, frame_bytes_of_fits(files[0]), stage="calibration", cpu_workers=max_workers) if files else 1
    stats["calibrate_workers"] = int(nw)
    if nw > 1:
        log_event(f"Calibration: {nw} parallel processes ({total} frames)")
    _ctx_w = dict(ctx, cal_diag_blob=_cal_diag_export_for_workers(cal_diag_session)) if nw > 1 else None
    _results = map_frames(
        _calibrate_frame_task,
        [(str(a), str(b)) for a, b in pairs],
        workers=nw,
        initializer=_init_calibrate_frame_worker,
        ctx=_ctx_w,
        serial_ctx=dict(ctx, session=cal_diag_session, db=db_main),
        stage="calibration",
    )
    _seq_tb_logged = False
    try:
        for i, ((src, _dst), res) in enumerate(zip(pairs, _results, strict=True), start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"Calibrating {src.name}")
            from infolog import replay_events  # noqa: PLC0415

            replay_events(res.get("events"))
            if not res.get("ok"):
                LOGGER.error("Calibration: file %s: %s\n%s", src, res.get("error"), res.get("traceback", ""))
                if not _seq_tb_logged:
                    log_event(f"CALIBRATION ERROR: {src.name}: {res.get('exc_type', '')}: {res.get('error')}")
                    log_event(str(res.get("traceback", "")))
                    _seq_tb_logged = True
                else:
                    log_event(f"CALIBRATION ERROR: {src.name}: {res.get('error')}")
                stats["errors"] += 1
                continue
            if res.get("skipped"):
                continue
            _apply_cal_frame_db_updates(db_main, qc_pack, src, res.get("db_updates") or [])
            perf10_qc = res.get("perf10_qc")
            if isinstance(perf10_qc, dict) and perf10_qc and not perf10_qc.get("error"):
                perf10_qc_results[str(src.resolve())] = perf10_qc
            _flags = str(res.get("vy_cflag") or "P")
            _sync_obs_calibration_state_with_retry(
                db_main,
                raw_light_path=src,
                draft_id=draft_id,
                is_calibrated=1 if "D" in _flags else 0,
                calib_type=_calibration_type_from_flags(_flags),
                calib_flags=_flags,
                stats=stats,
            )
            stats["processed"] += 1
            if res.get("used_dark"):
                stats["used_dark"] += 1
            if res.get("used_flat"):
                stats["used_flat"] += 1
            if not res.get("used_dark") and not res.get("used_flat"):
                stats["copied_only"] += 1
            if res.get("qc_summary") is not None:
                stats["qc_evaluated"] += 1
    finally:
        _results.close()
        _CAL_FRAME_STATE.clear()

    _arch = _archive_root_from_lights_root(lights_root)
    if _arch is not None:
        write_cal_diag_json(_arch, cal_diag_session)

    _refresh_manifest_after_cal_qc()
    return stats




def _qc_fwhm_elongation(
    data: "np.ndarray",
) -> dict[str, Any]:
    """FWHM / elongation for ``VY_FWHM`` (FWHM-ONE-2026-10).

    Same measurement as frame QC (:func:`frame_metrics.measure_frame` on the
    central crop: elliptical Gaussian fits), so ``VY_FWHM`` in every header is
    the one VYVAR FWHM. Returns ``n_stars_detected`` (DAO detections at 5 sigma
    on the crop) and ``n_sources`` (stars used in the fit).
    """
    from frame_metrics import measure_frame  # noqa: PLC0415

    m = measure_frame(_qc_center_crop_for_stars(np.asarray(data, dtype=np.float32)))
    return {
        "fwhm_px": m.get("fwhm_mean"),
        "elongation": m.get("elongation_mean"),
        "n_sources": int(m.get("n_fit") or 0),
        "n_stars_detected": int(m.get("star_count") or 0),
    }

def _poly_surface_fit(work: Any, fit_mask: Any, order: int, step: int, bg: float) -> Any | None:
    """Least-squares 2D polynomial (order 1-2) to ``work - bg`` on ``fit_mask`` (subsampled by ``step``);
    the full-frame surface, or None with too few samples. Plain least squares, no rejection."""
    import numpy as np

    h, w = work.shape
    yy_s, xx_s = np.mgrid[0:h:step, 0:w:step]
    z_s = (work[::step, ::step] - float(bg)).astype(np.float64)
    use = fit_mask[::step, ::step] & np.isfinite(z_s)
    min_coef = (order + 1) * (order + 2) // 2
    if int(np.count_nonzero(use)) < min_coef + 10:
        return None
    x_fit, y_fit, z_fit = xx_s[use].astype(np.float64), yy_s[use].astype(np.float64), z_s[use]
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    coef, *_ = np.linalg.lstsq(np.column_stack([(x_fit**i) * (y_fit**j) for i, j in terms]), z_fit, rcond=None)
    yy_f, xx_f = np.mgrid[0:h, 0:w]
    x_flat, y_flat = xx_f.ravel().astype(np.float64), yy_f.ravel().astype(np.float64)
    surf = sum(c * (x_flat**i) * (y_flat**j) for c, (i, j) in zip(coef, terms, strict=True))
    return np.asarray(surf).reshape(h, w).astype(np.float32), int(z_fit.size)


#: SKY-MASK-2026-10 source mask (photutils background guide, ``make_source_mask``): pixels of the
#: first-pass residual above SKY_MASK_NSIGMA x the sky MAD sigma in groups of >= SKY_MASK_NPIXELS
#: connected pixels are sources; the mask is dilated by one FWHM so their wings are excluded too.
SKY_MASK_NSIGMA = 2.0
SKY_MASK_NPIXELS = 5


def _fit_subtract_preprocess_sky_surface(
    data: Any,
    *,
    order: int,
    fwhm_px: float | None = None,
    subsample_step: int = 4,
) -> tuple[Any, dict[str, Any]]:
    """Fit and subtract a 2D polynomial sky correction (shared calibrated->processed preprocess).

    SKY-MASK-2026-10 (Milan 2026-10-26, D6): background pixels are found by masking sources, not by
    a cut in ADU around the median (the former +-100 ADU "calm" cut removed the strong part of a real
    gradient from the fit: 19-37 % of it stayed in the frame, and the cut depended on the rig).
    Pass 1: polynomial on pixels outside the DAO star stamps. Pass 2: sources of the pass-1 residual
    (segmentation, ``SKY_MASK_NSIGMA`` x MAD sigma, >= ``SKY_MASK_NPIXELS`` pixels, dilated by one
    FWHM) are added to the mask and the polynomial is fitted again; that surface is subtracted.
    No sigma-clip on the fit samples (zero-clipping policy 2026-08-12); the photutils guide calls
    masking the better way to exclude sources from a background estimate.
    """
    import numpy as np
    from photutils.detection import DAOStarFinder
    from photutils.segmentation import detect_sources
    from scipy.ndimage import binary_dilation

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
    mask &= finite

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
    step = max(1, int(subsample_step))
    skip = {"sky_surface_order": order_i, "sky_surface_applied": False,
            "sky_surface_skip_reason": "insufficient_fit_pixels"}
    pass1 = _poly_surface_fit(work, mask, order_i, step, float(bg_median))
    if pass1 is None:
        return arr.copy(), skip
    resid = work - float(bg_median) - pass1[0]
    sky_res = resid[mask]
    sig = 1.4826 * float(np.median(np.abs(sky_res - np.median(sky_res)))) if sky_res.size else 0.0
    n_src_px = 0
    if sig > 0 and math.isfinite(sig):
        seg = detect_sources(np.where(finite, resid, 0.0), SKY_MASK_NSIGMA * sig, npixels=SKY_MASK_NPIXELS)
        if seg is not None:
            r_d = max(1, int(round(fwhm_eff)))
            yy_d, xx_d = np.ogrid[-r_d:r_d + 1, -r_d:r_d + 1]
            src = binary_dilation(seg.data > 0, structure=(yy_d**2 + xx_d**2) <= r_d * r_d)
            n_src_px = int(np.count_nonzero(src & mask))
            mask &= ~src
    pass2 = _poly_surface_fit(work, mask, order_i, step, float(bg_median))
    if pass2 is None:
        return arr.copy(), skip
    # the surface is fitted to work - bg_median: subtracting it flattens the sky and keeps its level
    # (the sky photons stay in the frame for the CCD-equation error), as before
    surf = pass2[0]

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
        "sky_surface_n_fit_pixels": int(pass2[1]),
        "sky_surface_fwhm_px": float(fwhm_eff),
        "sky_surface_bg_median_adu": float(bg_median),
        "sky_surface_source_mask_px": n_src_px,
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

    # CAL-2026-10 K8: never guess gain / read noise. CAMERA-PROFILE-2026-10: resolved per mosaic frame
    # (header e-/ADU, else the camera profile at its gain setting and binning). Unknown -> keywords
    # not written; downstream resolves them or reports them missing.
    from astropy.io import fits as _fits_osc  # noqa: PLC0415

    from camera_profile import profile_from_connection  # noqa: PLC0415
    from param_resolver import resolve_gain, resolve_read_noise  # noqa: PLC0415

    _osc_profile = profile_from_connection(db.conn, int(equipment_id))

    def _osc_gain_rn(fp: Path) -> tuple[float, float]:
        try:
            _h = _fits_osc.getheader(str(fp), 0)
        except OSError:
            _h = None
        g = resolve_gain(_h, equipment_id=int(equipment_id), profile=_osc_profile)
        r = resolve_read_noise(_h, equipment_id=int(equipment_id), profile=_osc_profile)
        ge = float(g.value) if g.ok and g.value is not None else float("nan")
        re_ = float(r.value) if r.ok and r.value is not None else float("nan")
        if not (math.isfinite(ge) and math.isfinite(re_)):
            log_event(
                f"OSC extract: camera {equipment_id} gain/read noise unknown for {fp.name} - "
                "EGAIN/VYRDNOIS not written to channel frames."
            )
        return ge, re_
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
            gain_e, rn_e = _osc_gain_rn(fp)
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
        from osc_align import replicate_qc_verdict_from_one_rggb

        replicate_qc_verdict_from_one_rggb(lights_root=root, base_name=base_name)
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
) -> str:
    """Return ``apply`` or ``skip``. Raises :class:`SkySurfaceOrderConflictError` on order mismatch."""
    if sky_order <= 0:
        return "skip"
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


def _qc_enrich_task(item: tuple[str, str | None, int, float | None, float | None, bool]) -> Any:
    """Pool task of the in-place QC: one frame; a sky-surface order conflict is returned, not
    raised, so the main process stops with the frame's name."""
    fp_str, status, sky_order, ra, dec, only_if_missing = item
    try:
        return _qc_enrich_one_frame(
            fp_str,
            sky_order=sky_order,
            prefilter_status=status,
            target_ra=ra,
            target_dec=dec,
            inject_pointing_only_if_missing=only_if_missing,
        )
    except SkySurfaceOrderConflictError as exc:
        return exc


def _qc_enrich_one_frame(
    fp_str: str,
    *,
    sky_order: int,
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
        is_mosaic = _valid_bayerpat_from_header(hdr) is not None and not hdr.get("VYCHANNL")
        if sky_order > 0 and not is_mosaic:
            action = _decide_preprocess_sky_action(hdr, sky_order=sky_order)
            if action == "apply":
                data, sky_stats = _fit_subtract_preprocess_sky_surface(data, order=sky_order)
                cal_stage_token, skypass = compute_skysf_apply_stage(sky_order=sky_order)
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
            # FITS-SCALE-2026-10: no BZERO of the integer original on the float data
            set_float_image_data_in_place(hdul[0], _as_fits_float32_image(data))
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
                            f"INV-CAL-02: VYCALSUM self-check failed on {fp.name}"
                        )
            if math.isfinite(fwhm):
                hdr["VY_FWHM"] = (round(fwhm, 4), "Estimated FWHM [pix]")
            if math.isfinite(elong):
                hdr["VY_ELONG"] = (round(elong, 4), "Estimated elongation (a/b)")
            hdr["VY_NSTAR"] = (n_stars, "Approx. star detections (QC)")
            hdr["VY_QC"] = (status, "QC status")
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


#: qc_metrics.csv columns kept in memory only: n_sources (n_stars_detected is the count read),
#: cal_pstbg (= bg_median; synced to the DB from the row), the sky-surface FWHM / calm threshold
#: (configuration constants).
QC_METRICS_NOT_WRITTEN: tuple[str, ...] = ("n_sources", "cal_pstbg", "sky_surface_fwhm_px", "sky_surface_source_mask_px")


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

    cfg = app_config or AppConfig()
    sky_order = int(cfg.preprocess_sky_surface_order)

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
    from parallel_frames import frame_bytes_of_fits, map_frames, stage_workers  # noqa: PLC0415

    n_workers = stage_workers(total, frame_bytes_of_fits(fits_paths[0]), stage="QC in-place") if total > 1 else 1
    tasks = [
        (str(fp), _prefilter.get(norm_fits_path_key(fp)), sky_order, target_ra, target_dec,
         bool(inject_pointing_only_if_missing))
        for fp in fits_paths
    ]
    rows = map_frames(_qc_enrich_task, tasks, workers=n_workers, stage="QC in-place")
    for i, (fp, row) in enumerate(zip(fits_paths, rows, strict=True), start=1):
        if progress_cb is not None:
            progress_cb(i, total, f"QC in-place {fp.name}")
        if isinstance(row, SkySurfaceOrderConflictError):
            row.add_note(f"QC in-place: frame {fp.name}")
            raise row
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

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    n_rejected = sum(1 for r in results if str(r.get("status", "")).startswith("rejected"))
    sky_surface_skip_count = sum(1 for r in results if r.get("sky_surface_skipped"))
    log_event(
        f"QC in-place: {n_ok} ok, {n_rejected} rejected, "
        f"{len(results) - n_ok - n_rejected} errors"
    )
    if sky_surface_skip_count:
        log_event(
            f"QC in-place: sky-surface subtract skipped on {sky_surface_skip_count} frame(s) "
            "(VY_SKYSF guard)"
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
        # OUTPUT-REVIEW-2026-10 step 6: in-memory only (DB sync / config constants / duplicates)
        _qc_df.drop(columns=[c for c in QC_METRICS_NOT_WRITTEN if c in _qc_df.columns]).to_csv(_qc_csv, index=False)
        log_event(f"qc_metrics.csv written to {_qc_csv}")
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[PIPELINE] qc_metrics.csv write failed: %s", exc)
    emit_n_stars_diagnostic(_qc_df)

    return {
        "n_processed": len(results),
        "n_ok": n_ok,
        "n_rejected": n_rejected,
        "sky_surface_skip_count": sky_surface_skip_count,
        "results": results,
        "qc_root": str(calibrated_root),
        "qc_csv": str(_qc_csv),
    }


def _n_stars_diag_frame_label(src: Any) -> str:
    stem = Path(str(src or "")).stem
    idx = stem.rfind("Light_")
    if idx >= 0:
        return stem[idx:]
    return stem or "unknown"


def _n_stars_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def plan_n_stars_diagnostic(
    df: pd.DataFrame | None,
    *,
    k: float | None = None,
) -> dict[str, Any]:
    """Bounds and outlier list for n_stars_detected; no I/O. D-NSTARS-DIAG-01."""
    kk = float(N_STARS_DIAG_K if k is None else k)
    empty: dict[str, Any] = {
        "k": kk,
        "n_ok": 0,
        "median": None,
        "mad": 0.0,
        "sigma_mad": 0.0,
        "lo": None,
        "hi": None,
        "n_low": 0,
        "n_high": 0,
        "frames": [],
        "bounds_undefined": True,
        "used_all_rows_mad": False,
    }
    if df is None or df.empty or "n_stars_detected" not in df.columns:
        return empty
    work = df.copy()
    work["_n"] = pd.to_numeric(work["n_stars_detected"], errors="coerce")
    if "status" in work.columns:
        ok = work[work["status"].astype(str) == "ok"]
    else:
        ok = work
    ok_n = ok["_n"].to_numpy(dtype=np.float64)
    ok_n = ok_n[np.isfinite(ok_n)]
    empty["n_ok"] = int(ok_n.size)
    if ok_n.size == 0:
        return empty
    med = float(np.median(ok_n))
    mad = _n_stars_mad(ok_n)
    used_all = False
    if mad == 0.0:
        mad = _n_stars_mad(work["_n"].to_numpy(dtype=np.float64))
        used_all = True
    if mad == 0.0:
        empty["median"] = med
        empty["used_all_rows_mad"] = used_all
        return empty
    sig = float(mad * _N_STARS_DIAG_MAD_TO_SIGMA)
    lo = med - kk * sig
    hi = med + kk * sig
    frames: list[dict[str, Any]] = []
    src_col = "src" if "src" in ok.columns else ("dst" if "dst" in ok.columns else None)
    for _, row in ok.iterrows():
        nv = row["_n"]
        if not math.isfinite(float(nv)):
            continue
        n_i = float(nv)
        if n_i < lo or n_i > hi:
            label = _n_stars_diag_frame_label(row[src_col] if src_col else "")
            side = "low" if n_i < lo else "high"
            frames.append({"frame": label, "n": n_i, "side": side})
    n_low = sum(1 for f in frames if f["side"] == "low")
    n_high = sum(1 for f in frames if f["side"] == "high")
    return {
        "k": kk,
        "n_ok": int(ok_n.size),
        "median": med,
        "mad": float(mad),
        "sigma_mad": sig,
        "lo": lo,
        "hi": hi,
        "n_low": int(n_low),
        "n_high": int(n_high),
        "frames": frames,
        "bounds_undefined": False,
        "used_all_rows_mad": used_all,
    }


def emit_n_stars_diagnostic(df: pd.DataFrame | None) -> dict[str, Any]:
    """Log n_stars outliers; never changes status, headers, or files."""
    plan = plan_n_stars_diagnostic(df)
    if plan["bounds_undefined"]:
        LOGGER.warning(
            "[FRAME-QC] n_stars diagnostic: MAD=0; bounds undefined; no per-frame warnings"
        )
        return plan
    lo = float(plan["lo"])
    hi = float(plan["hi"])
    med = float(plan["median"])
    sig = float(plan["sigma_mad"])
    kk = float(plan["k"])
    for rec in plan["frames"]:
        LOGGER.warning(
            "[FRAME-QC] n_stars diagnostic: %s n=%s outside [%.4g, %.4g] "
            "(median %.4g, sigma_MAD %.4g, k=%.1f); frame kept",
            rec["frame"],
            int(rec["n"]) if float(rec["n"]).is_integer() else rec["n"],
            lo,
            hi,
            med,
            sig,
            kk,
        )
    LOGGER.warning(
        "[FRAME-QC] n_stars diagnostic: n_low=%d n_high=%d "
        "(median %.4g, sigma_MAD %.4g, k=%.1f); frames kept",
        int(plan["n_low"]),
        int(plan["n_high"]),
        med,
        sig,
        kk,
    )
    return plan


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


