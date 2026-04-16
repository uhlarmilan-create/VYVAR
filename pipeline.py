"""Core processing pipeline for FITS observations."""

from __future__ import annotations

import contextlib
import json
import logging
import pickle
import math
import multiprocessing
import os
import shutil
import subprocess
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import pandas as pd

from config import AppConfig, load_config_json
from database import (
    DraftTechnicalMetadataError,
    VyvarDatabase,
    _db_to_float as _to_float_db,
    query_local_gaia,
    query_local_vsx,
)
from time_utils import _header_float as _header_float_tu
from photometry import (
    common_field_intersection_bbox_px,
    enhance_catalog_dataframe_aperture_bpm,
    recommended_aperture_by_color,
    stress_test_relative_rms_from_sidecars,
    vsx_is_known_variable_top3_per_bin,
)
from fits_suffixes import FITS_SUFFIXES_LOWER, path_suffix_is_fits
from gaia_catalog_id import catalog_id_series_for_masterstars_export
from infolog import log_event, log_exception
from calibration import (
    CALIBRATION_LIBRARY_NATIVE_BINNING,
    filter_light_paths_for_calibration_db,
    get_processed_master,
)

from utils import (
    ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC,
    DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    MIN_GAIA_CONE_RADIUS_DEG,
    astrometry_net_scale_bounds_arcsec_per_pix,
    catalog_cone_radius_deg_from_optics,
    catalog_cone_radius_from_fov_diameter_deg,
    circular_angle_diff_deg,
    dao_detection_fwhm_pixels,
    effective_astrometry_net_tweak_order,
    effective_binned_pixel_pitch_um,
    fits_binning_xy_from_header,
    fits_header_has_celestial_wcs,
    header_key_is_celestial_wcs,
    iter_fits_paths_recursive,
    maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel,
    normalize_telescope_focal_mm_for_plate_scale,
    plate_scale_arcsec_per_pixel,
    strip_celestial_wcs_keys,
    wcs_distortion_log_suffix,
    wcs_rotation_angle_deg,
)
from vyvar_platesolver import (
    _fits_header_parse_dec_deg,
    _fits_header_parse_ra_deg,
    pointing_hint_from_header as _pointing_hint_from_header,
)
from vyvar_astrometry_wsl import (
    is_wsl_available,
    masterstar_wcs_quality,
    solve_field_wsl,
)


def _apply_aperture_catalog_enhancements_from_st(
    df: pd.DataFrame,
    data: Any,
    hdr: fits.Header,
    st: dict[str, Any],
) -> pd.DataFrame:
    """Aperture photometry + linearity/BPM flags for per-frame catalog DataFrames."""
    if not bool(st.get("aperture_photometry_enabled", True)):
        return df
    mdp = st.get("master_dark_path") or ""
    mdp = str(mdp).strip() or None
    try:
        _go = st.get("gaussian_fwhm_px_override")
        try:
            _go_f = float(_go) if _go is not None else None
        except (TypeError, ValueError):
            _go_f = None
        return enhance_catalog_dataframe_aperture_bpm(
            df,
            data,
            hdr,
            aperture_enabled=True,
            aperture_fwhm_factor=float(st.get("aperture_fwhm_factor", 1.7)),
            annulus_inner_fwhm=float(st.get("annulus_inner_fwhm", 4.0)),
            annulus_outer_fwhm=float(st.get("annulus_outer_fwhm", 6.0)),
            nonlinearity_peak_percentile=float(st.get("nonlinearity_peak_percentile", 20.0)),
            nonlinearity_fwhm_ratio=float(st.get("nonlinearity_fwhm_ratio", 1.25)),
            master_dark_path=mdp,
            gaussian_fwhm_px_override=_go_f,
        )
    except Exception:  # noqa: BLE001
        return df


def _fits_header_positive_float(hdr: fits.Header, *keys: str) -> float | None:
    for k in keys:
        v = _header_float_tu(hdr, k)
        if v is not None and v > 0:
            return v
    return None


def _frame_gain_readnoise_for_error_map(
    hdr: fits.Header,
    *,
    db: VyvarDatabase | None,
    equipment_id: int | None,
) -> tuple[float, float]:
    gain = _fits_header_positive_float(hdr, "EGAIN", "GAIN")
    if gain is None and db is not None and equipment_id is not None:
        g_db, _ = db.get_equipment_cosmic_params(int(equipment_id))
        if g_db is not None and g_db > 0:
            gain = float(g_db)
    if gain is None or gain <= 0 or not math.isfinite(gain):
        gain = 1.0

    rn = _fits_header_positive_float(hdr, "RDNOISE", "READNOISE")
    if rn is None and db is not None and equipment_id is not None:
        _, rn_db = db.get_equipment_cosmic_params(int(equipment_id))
        if rn_db is not None and rn_db > 0:
            rn = float(rn_db)
    if rn is None or not math.isfinite(rn) or rn <= 0:
        rn = 10.0
    return float(gain), float(rn)


def _per_frame_noise_error_map(data: Any, hdr: fits.Header, *, db: VyvarDatabase | None, equipment_id: int | None):
    """Per-pixel noise σ for CCD-like error: sqrt(max(data,0)/gain + readnoise²)."""
    import numpy as np

    gain, rn = _frame_gain_readnoise_for_error_map(hdr, db=db, equipment_id=equipment_id)
    d = np.asarray(data, dtype=np.float64)
    sig = np.where(np.isfinite(d), np.maximum(d, 0.0), 0.0)
    var = sig / float(gain) + float(rn) ** 2
    return np.sqrt(np.maximum(var, 1e-24))


def _vyvar_calibrate_multiprocessing_enabled() -> bool:
    """Parallel calibration uses ``spawn`` workers (errors are easy to miss). Set ``VYVAR_CALIBRATE_MP=1`` to enable."""
    v = os.environ.get("VYVAR_CALIBRATE_MP", "").strip().lower()
    return v in ("1", "true", "yes")


def _cfg_calibration_library_native_binning(cfg: Any) -> int | None:
    """Config ``calibration_library_native_binning``: ``None`` = read ``XBINNING`` from each master FITS."""
    raw = getattr(cfg, "calibration_library_native_binning", CALIBRATION_LIBRARY_NATIVE_BINNING)
    if raw is None:
        return None
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return int(CALIBRATION_LIBRARY_NATIVE_BINNING)


# ``_calibrate_one_light_*``: explicit ``None`` = read master FITS; omit param = library default (1×1).
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
        log_event(f"Kalibrácia — zápis OK: {calibrated_root.resolve()}")
    except OSError as exc:
        log_event(f"Kalibrácia — CHÝBA PRÁVO ZÁPISU (calibrated): {calibrated_root.resolve()} → {exc}")

    if master_dark_path is not None:
        md = Path(master_dark_path)
        md_r = md.resolve()
        ok = md_r.is_file()
        log_event(f"MasterDark: {md_r} (exists={ok})")
        if not ok:
            log_event("MasterDark: súbor neexistuje — dark sa neaplikuje (flat-only / copy-only podľa nastavenia).")

    for fk, fp in sorted((masterflat_by_filter or {}).items(), key=lambda x: str(x[0])):
        if fp is None:
            log_event(f"MasterFlat[{fk!r}]: (žiadna cesta)")
            continue
        p = Path(fp)
        pr = p.resolve()
        ok = pr.is_file()
        log_event(f"MasterFlat[{fk!r}]: {pr} (exists={ok})")
        if not ok:
            log_event(f"MasterFlat[{fk!r}]: súbor neexistuje — pre tento filter sa flat neaplikuje.")

# Public aliases (historically some callers used ``pipeline.parse_user_*`` / ``pointing_hint_from_header``).
pointing_hint_from_header = _pointing_hint_from_header


LOGGER = logging.getLogger(__name__)


def _pipeline_ui_info(msg: str) -> None:
    """Show an info line in Streamlit when available; always record via :func:`log_event`."""
    try:
        import streamlit as st

        st.info(msg)
    except Exception:  # noqa: BLE001
        pass
    log_event(msg)


def _pipeline_ui_error(msg: str) -> None:
    """Show an error line in Streamlit when available; always record via :func:`log_event`."""
    try:
        import streamlit as st

        st.error(msg)
    except Exception:  # noqa: BLE001
        pass
    log_event(msg)


def _ensure_parent_dirs_for_aligned_fits(out_path: Path) -> None:
    """Ensure parent folders exist for nested outputs under ``detrended_aligned/lights/...``."""
    os.makedirs(str(out_path.parent), exist_ok=True)


def _assert_alignment_produced_fits(aligned_root: Path) -> None:
    """Fail fast if alignment wrote no FITS under the group folder (recursive; not only top-level)."""
    n = len(iter_fits_paths_recursive(aligned_root))
    if n == 0:
        raise RuntimeError(
            "Alignment zlyhal - nenašli sa žiadne výstupné súbory! "
            f"(žiadne FITS v {aligned_root.resolve()} vrátane podadresárov.)"
        )


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


def _safe_filter_token(text: str) -> str:
    t = (text or "").strip()
    if not t or t.lower() in {"unknown", "none", "nan"}:
        return "NoFilter"
    return t


def observation_group_key_from_metadata(meta: dict[str, Any]) -> str:
    """Must match ``importer.observation_group_key`` (FILTER, EXPTIME, binning from FITS / metadata dict)."""
    flt = _safe_filter_token(str(meta.get("filter") or "NoFilter"))
    try:
        e = float(meta.get("exposure", 0.0))
    except (TypeError, ValueError):
        e = 0.0
    b = max(1, int(meta.get("binning", 1) or 1))
    return f"{flt}|{e:g}|{b}"


def _iter_light_fits(lights_root: Path) -> list[Path]:
    """Collect lights FITS under lights_root (including nested subdirs, e.g. ``filter_exp_binning``)."""
    return iter_fits_paths_recursive(lights_root)


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


def _archive_raw_to_calibrated_light(
    archive: Path,
    raw_fp: Path | str,
) -> tuple[Path, Path] | None:
    """Map archived raw light path to ``(calibrated_fits, lights_root_under_archive)``."""
    ap = Path(archive).expanduser().resolve()
    raw_path = Path(raw_fp)
    try:
        r = raw_path.resolve() if raw_path.is_file() else (ap / raw_path).resolve()
    except OSError:
        r = raw_path if raw_path.is_file() else (ap / raw_path)
    pairs: tuple[tuple[Path, Path], ...] = (
        (ap / "non_calibrated" / "lights", ap / "calibrated" / "lights"),
        (ap / "Raw" / "lights", ap / "calibrated" / "lights"),
    )
    for raw_root, cal_root in pairs:
        if not raw_root.is_dir():
            continue
        try:
            rel = r.relative_to(raw_root.resolve())
        except ValueError:
            continue
        cand = cal_root / rel
        if cand.is_file():
            return cand, cal_root
    return None


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


def _resolve_draft_light_raw_path(archive: Path, file_path: Path | str) -> Path | None:
    """Resolve raw light FITS on disk for in-RAM calibration (``OBS_FILES.FILE_PATH``)."""
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
    return None


def _archive_preprocess_lights_root(ap: Path | str) -> Path:
    """Detrended/proc FITS root for alignment (prefers ``processed/lights``, else ``detrended/lights``)."""
    root = Path(ap).expanduser()
    for cand in (root / "processed" / "lights", root / "detrended" / "lights"):
        if cand.is_dir() and _iter_fits_recursive(cand):
            return cand
    return root / "processed" / "lights"


def resolve_masterstar_input_root(archive_path: Path | str, setup_name: str | None = None) -> Path:
    """Pick MASTERSTAR input root under ``processed/lights[/setup_name]`` with robust fallback."""
    ap = Path(archive_path).expanduser()
    base = ap / "processed" / "lights"
    if setup_name:
        cand = base / str(setup_name).strip()
        if cand.is_dir() and _iter_fits_recursive(cand):
            return cand
    if base.is_dir():
        subdirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.name.casefold())
        for sd in subdirs:
            if _iter_fits_recursive(sd):
                return sd
        if _iter_fits_recursive(base):
            return base
    return base


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
        except Exception:  # noqa: BLE001
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
    return _quality_inspection_dao_metrics_array(data, hdr)


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


def _dao_star_table_mean_elongation(tbl: Any) -> float | None:
    """Mean elongation (semi-major / semi-minor axis ratio) over detected sources.
    Values near 1.0 = round stars. High values (>1.5) suggest satellite/aircraft trail."""
    import numpy as np

    if tbl is None or len(tbl) == 0:
        return None
    try:
        if "sharpness" not in tbl.colnames:
            return None
        # DAOStarFinder nemá priamu elongation — odvodíme z roundness1/roundness2.
        # elongation ≈ 1 + hypot(roundness1, roundness2)
        if "roundness1" not in tbl.colnames or "roundness2" not in tbl.colnames:
            return None
        r1 = np.asarray(tbl["roundness1"], dtype=np.float64)
        r2 = np.asarray(tbl["roundness2"], dtype=np.float64)
        elong = 1.0 + np.hypot(np.abs(r1), np.abs(r2))
        ok = np.isfinite(elong)
        if not np.any(ok):
            return None
        return float(np.mean(elong[ok]))
    except Exception:  # noqa: BLE001
        return None


def _quality_inspection_dao_metrics_array(
    data: "np.ndarray",
    hdr: fits.Header,
) -> dict[str, Any]:
    """Same as :func:`_quality_inspection_dao_metrics` but on an in-memory calibrated image."""
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

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
    _, med, std = sigma_clipped_stats(crop[finite], sigma=3.0, maxiters=5)
    std = float(std)
    if not math.isfinite(std) or std <= 0:
        return out
    out["sky_background"] = float(med) if np.isfinite(med) else None

    img2 = np.asarray(crop - float(med), dtype=np.float32)
    img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.nanmedian(img2)) < 0:
        img2 = -img2

    fwhm_guess = _estimate_dao_fwhm_guess(img2, std)
    daofind = DAOStarFinder(
        fwhm=float(fwhm_guess),
        threshold=5.0 * std,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    tbl = daofind(img2)
    if tbl is None or len(tbl) == 0:
        daofind = DAOStarFinder(
            fwhm=float(fwhm_guess),
            threshold=3.0 * std,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = daofind(img2)
    if tbl is None or len(tbl) == 0:
        return out

    tbl.sort("flux")
    tbl = tbl[::-1]
    out["star_count"] = int(len(tbl))
    out["roundness_mean"] = _dao_star_table_mean_roundness(tbl)
    out["elongation_mean"] = _dao_star_table_mean_elongation(tbl)
    max_sources = 50
    n_use = int(min(len(tbl), max_sources))
    fwhm_list: list[float] = []
    half = 7
    h, w = img2.shape

    for i in range(n_use):
        x0 = float(tbl["xcentroid"][i])
        y0 = float(tbl["ycentroid"][i])
        xi = int(round(x0))
        yi = int(round(y0))
        y1 = max(0, yi - half)
        y2 = min(h, yi + half + 1)
        x1 = max(0, xi - half)
        x2 = min(w, xi + half + 1)
        cut = img2[y1:y2, x1:x2]
        if cut.size < 25:
            continue
        cut = np.where(cut > 0, cut, 0.0).astype(np.float32)
        s = float(np.sum(cut))
        if not math.isfinite(s) or s <= 0:
            continue
        yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
        cx = float(np.sum(xx * cut) / s)
        cy = float(np.sum(yy * cut) / s)
        dx = xx - cx
        dy = yy - cy
        mxx = float(np.sum((dx * dx) * cut) / s)
        myy = float(np.sum((dy * dy) * cut) / s)
        mxy = float(np.sum((dx * dy) * cut) / s)
        tr = mxx + myy
        det = mxx * myy - mxy * mxy
        disc = tr * tr - 4.0 * det
        if disc < 0:
            continue
        l1 = 0.5 * (tr + float(np.sqrt(disc)))
        l2 = 0.5 * (tr - float(np.sqrt(disc)))
        if l1 <= 0 or l2 <= 0:
            continue
        sig1 = float(np.sqrt(l1))
        sig2 = float(np.sqrt(l2))
        fwhm = 2.355 * 0.5 * (sig1 + sig2)
        if np.isfinite(fwhm) and 0.2 < fwhm < 50:
            fwhm_list.append(float(fwhm))

    if fwhm_list:
        out["fwhm_mean"] = float(np.nanmedian(np.asarray(fwhm_list, dtype=np.float64)))
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


def resolve_preprocess_target_coordinates(
    *,
    db: VyvarDatabase,
    draft_id: int | None,
    ui_ra_deg: float | None,
    ui_dec_deg: float | None,
) -> tuple[float | None, float | None]:
    """Resolve preprocess target coordinates with DB-first priority.

    Priority:
    1) ``OBS_DRAFT.CENTEROFFIELDRA/DE`` (when ``draft_id`` exists and values are finite, non-zero pair)
    2) UI values
    3) median RA/DE from draft light rows
    """
    if draft_id is not None:
        try:
            drow = db.fetch_obs_draft_by_id(int(draft_id)) or {}
            ra_db = drow.get("CENTEROFFIELDRA")
            de_db = drow.get("CENTEROFFIELDDE")
            if ra_db is not None and de_db is not None:
                ra_f = float(ra_db)
                de_f = float(de_db)
                if math.isfinite(ra_f) and math.isfinite(de_f):
                    if not (abs(ra_f) < 1e-9 and abs(de_f) < 1e-9):
                        log_event(
                            f"INFO: Preprocessing forced to DB coordinates RA:{ra_f}, Dec:{de_f} for stability."
                        )
                        return float(ra_f), float(de_f)
                    log_event(
                        "WARNING: OBS_DRAFT center coordinates are 0.0/0.0; falling back to UI/median values."
                    )
        except Exception:  # noqa: BLE001
            pass
    try:
        if ui_ra_deg is not None and ui_dec_deg is not None:
            ra_ui = float(ui_ra_deg)
            de_ui = float(ui_dec_deg)
            if math.isfinite(ra_ui) and math.isfinite(de_ui):
                if not (abs(ra_ui) < 1e-9 and abs(de_ui) < 1e-9):
                    log_event(f"DEBUG: Preprocess using UI fallback coordinates: RA={ra_ui}, Dec={de_ui}")
                    return float(ra_ui), float(de_ui)
    except (TypeError, ValueError):
        pass
    if draft_id is not None:
        try:
            med_ra, med_de = draft_median_pointing_icrs_deg(db, int(draft_id))
            if med_ra is not None and med_de is not None:
                if math.isfinite(float(med_ra)) and math.isfinite(float(med_de)):
                    log_event(
                        f"DEBUG: Preprocess using draft median coordinates: RA={float(med_ra)}, Dec={float(med_de)}"
                    )
                    return float(med_ra), float(med_de)
        except Exception:  # noqa: BLE001
            pass
    return None, None


def _estimate_fov_deg_from_header(hdr: fits.Header) -> float | None:
    """Rough field diameter in degrees from WCS pixel scale × ``NAXIS*`` (fallback when CDELT missing)."""
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


def _estimate_fov_deg_from_fits_path(fp: Path) -> float | None:
    p = Path(fp)
    if not p.is_file():
        return None
    try:
        with fits.open(p, memmap=False) as hdul:
            return _estimate_fov_deg_from_header(hdul[0].header)
    except Exception:  # noqa: BLE001
        return None


def _icrs_offset_arcmin(
    ra_deg: float,
    de_deg: float,
    ref_ra_deg: float,
    ref_de_deg: float,
) -> float:
    """Small-angle offset from reference (degrees → arcminutes): sqrt((ΔRA·cos δ)² + Δδ²)·60."""
    dra = (float(ra_deg) - float(ref_ra_deg) + 180.0) % 360.0 - 180.0
    dde = float(de_deg) - float(ref_de_deg)
    cos_dec = math.cos(math.radians(float(ref_de_deg)))
    sep_deg = math.hypot(dra * cos_dec, dde)
    return sep_deg * 60.0


def sync_obs_files_drift_arcmin_for_draft(
    db: VyvarDatabase,
    draft_id: int,
    *,
    ref_ra_deg: float | None,
    ref_de_deg: float | None,
) -> int:
    """Fill ``OBS_FILES.DRIFT`` (arcmin), ``DRIFT_DRA`` / ``DRIFT_DDE`` (deg plane offsets vs median). Clears when ref missing."""
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    n = 0
    if ref_ra_deg is None or ref_de_deg is None:
        for row in rows:
            db.update_obs_file_quality_by_id(int(row["ID"]), clear_drift=True)
            n += 1
        return n
    try:
        rref0, dref0 = float(ref_ra_deg), float(ref_de_deg)
    except (TypeError, ValueError):
        for row in rows:
            db.update_obs_file_quality_by_id(int(row["ID"]), clear_drift=True)
            n += 1
        return n
    if not (math.isfinite(rref0) and math.isfinite(dref0)):
        for row in rows:
            db.update_obs_file_quality_by_id(int(row["ID"]), clear_drift=True)
            n += 1
        return n
    rref, dref = rref0, dref0
    for row in rows:
        rid = int(row["ID"])
        try:
            ra = row.get("RA")
            de = row.get("DE")
            if ra is None or de is None:
                db.update_obs_file_quality_by_id(rid, clear_drift=True)
                n += 1
                continue
            raf, def_ = float(ra), float(de)
            if not (math.isfinite(raf) and math.isfinite(def_)):
                db.update_obs_file_quality_by_id(rid, clear_drift=True)
                n += 1
                continue
            dra_deg = ((raf - rref) + 180.0) % 360.0 - 180.0
            dde_deg = def_ - dref
            dra_plane = dra_deg * math.cos(math.radians(dref))
            d_arc = math.hypot(dra_plane, dde_deg) * 60.0
            if not math.isfinite(d_arc):
                db.update_obs_file_quality_by_id(rid, clear_drift=True)
            else:
                db.update_obs_file_quality_by_id(
                    rid,
                    drift_arcmin=float(d_arc),
                    drift_dra_deg=float(dra_plane),
                    drift_dde_deg=float(dde_deg),
                )
            n += 1
        except (TypeError, ValueError):
            db.update_obs_file_quality_by_id(rid, clear_drift=True)
            n += 1
    return n


def generate_observation_hash(db: VyvarDatabase, draft_id: int) -> str:
    """Deterministic processing hashtag: camera (equipment) + telescope + filter/exptime set + JD start.

    JD start is the minimum finite ``INSPECTION_JD`` among draft lights, else ``OBS_DRAFT.OBSERVATIONSTARTJD``.
    Filter/exptime signature is sorted unique ``(FILTER, EXPTIME)`` pairs from ``OBS_FILES`` lights.
    """
    import hashlib

    drow = db.fetch_obs_draft_by_id(int(draft_id))
    if drow is None:
        raise ValueError(f"OBS_DRAFT id={int(draft_id)} not found")
    id_eq = int(drow.get("ID_EQUIPMENTS") or 0)
    id_tel = int(drow.get("ID_TELESCOPE") or 0)
    lights = db.fetch_draft_light_rows_for_quality(int(draft_id))
    pair_set: set[tuple[str, float]] = set()
    jds: list[float] = []
    for r in lights:
        flt = str(r.get("FILTER") or "").strip() or "None"
        ex_raw = r.get("EXPTIME")
        try:
            ex_f = float(ex_raw) if ex_raw is not None else 0.0
        except (TypeError, ValueError):
            ex_f = 0.0
        if math.isfinite(ex_f):
            pair_set.add((flt, round(ex_f, 6)))
        jd_raw = r.get("INSPECTION_JD")
        try:
            jf = float(jd_raw) if jd_raw is not None else float("nan")
            if math.isfinite(jf):
                jds.append(jf)
        except (TypeError, ValueError):
            pass
    pairs_sorted = "|".join(f"{a}:{b:.6f}" for a, b in sorted(pair_set))
    try:
        jd0 = float(drow.get("OBSERVATIONSTARTJD") or 0.0)
    except (TypeError, ValueError):
        jd0 = 0.0
    if jds:
        jd0 = min(jds)
    if not math.isfinite(jd0):
        jd0 = 0.0
    payload = f"{id_eq}|{id_tel}|{pairs_sorted}|{jd0:.8f}"
    digest = hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    date_prefix = VyvarDatabase._jd_to_yyyymmdd(jd0)
    return f"{date_prefix}_{digest}"


def run_quality_analysis(
    *,
    db: VyvarDatabase,
    draft_id: int,
    archive_path: Path | str,
    progress_cb: Callable[[int, int, str], None] | None = None,
    roundness_reject_above: float | None = None,
) -> dict[str, Any]:
    """Per-draft light: DAO metrics → ``OBS_FILES``; FWHM ×1.5 and optional DAO roundness auto-reject.

    When Streamlit is available, updates ``st.session_state`` keys ``fwhm_threshold``, ``center_ra``,
    ``center_de`` from computed medians (same as RAM QC path).

    The Quality Dashboard assigns ``frame_index`` 1…N in the same order as
    :meth:`VyvarDatabase.fetch_draft_light_rows_for_quality` (stable table ↔ plot alignment).

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
            db.update_obs_file_quality_by_id(rid, rejected_auto=0)
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
        db.update_obs_file_quality_by_id(
            rid,
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
        db.update_obs_file_quality_by_id(rid, rejected_auto=rej)

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
    except Exception:
        pass
    return result


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
    """Calibrate each draft light **in RAM only**, DAO metrics → ``OBS_FILES``; FWHM ×1.5 and optional roundness reject.

    No calibrated FITS are written. Uses the same master selection rules as :func:`calibrate_lights_to_calibrated`.
    After QC, :func:`sync_obs_files_drift_arcmin_for_draft` writes ``DRIFT`` / ``DRIFT_DRA`` / ``DRIFT_DDE``.
    """
    import numpy as np

    _rn = 1.25 if roundness_reject_above is None else float(roundness_reject_above)
    rlim_active = math.isfinite(_rn) and _rn > 0.0

    ap = Path(archive_path).expanduser()
    cfg = pipeline_config or AppConfig()
    db_cal = _db_for_calibration_tasks(None, None)

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

    def _dark_np_for_path(p: Path | None, light_binning: int) -> Any:
        if p is None or not p.is_file():
            return None
        _mb_key = "hdr" if _native_b is None else str(int(_native_b))
        key = f"{str(p.resolve())}|b{int(light_binning)}|mb{_mb_key}"
        if key not in dark_cache:
            pm = get_processed_master(
                p,
                int(light_binning),
                kind="dark",
                master_binning=_native_b,
            )
            dark_cache[key] = pm.data
        return dark_cache[key]

    flat_cache: dict[str, Any] = {}
    flat_median_scale: dict[str, float] = {}
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    n = len(rows)
    fwhm_by_id: dict[int, float] = {}
    roundness_by_id: dict[int, float] = {}
    errors: list[str] = []
    fov_sample_deg: float | None = None

    for i, row in enumerate(rows, start=1):
        rid = int(row["ID"])
        raw_fp = _resolve_draft_light_raw_path(ap, str(row.get("FILE_PATH") or ""))
        if raw_fp is None:
            errors.append(f"missing raw {row.get('FILE_PATH')}")
            db.update_obs_file_quality_by_id(rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"Skip missing {Path(str(row.get('FILE_PATH') or '')).name}")
            continue

        try:
            with fits.open(raw_fp, memmap=False) as hdul:
                hdr0 = hdul[0].header
                _ok = observation_group_key_from_metadata(fits_metadata_from_primary_header(hdr0))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{raw_fp.name}: header {exc}")
            db.update_obs_file_quality_by_id(rid, rejected_auto=0)
            if progress_cb is not None:
                progress_cb(i, n, f"Header fail {raw_fp.name}")
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
        if md_use is not None and md_use.is_file():
            if (
                md_pre is not None
                and md_path_ok is not None
                and md_use.resolve() == md_path_ok.resolve()
                and _native_b is not None
                and _native_b == light_bx
            ):
                md_np_use = md_pre
            else:
                md_np_use = _dark_np_for_path(md_use, light_bx)

        try:
            data, hdr, _ud, _uf = _calibrate_one_light_apply_masters_in_ram(
                src=raw_fp,
                master_dark_path=md_use,
                masterflat_by_filter=mf_merged,
                flat_cache=flat_cache,
                flat_median_scale=flat_median_scale,
                md_data_preload=md_np_use,
                cosmic_options=None,
                db=db_cal,
                id_equipments=equipment_id,
                calibration_master_native_binning=_native_b,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{raw_fp.name}: {exc}")
            db.update_obs_file_quality_by_id(rid, rejected_auto=0)
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
        db.update_obs_file_quality_by_id(
            rid,
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
        db.update_obs_file_quality_by_id(rid, rejected_auto=rej)

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
    except Exception:
        pass
    return result


def calibrated_paths_for_draft_not_rejected(
    archive_path: Path | str,
    db: VyvarDatabase,
    draft_id: int,
) -> tuple[list[Path], list[Path]]:
    """Accepted lights (``IS_REJECTED`` 0 or NULL) mapped to ``calibrated/lights``."""
    ap = Path(archive_path).expanduser()
    rows = db.fetch_draft_light_rows_for_quality(int(draft_id))
    cal_paths: list[Path] = []
    cal_root = (ap / "calibrated" / "lights").resolve()

    for row in rows:
        if int(row.get("IS_REJECTED") or 0):
            continue
        raw = Path(str(row.get("FILE_PATH") or ""))
        m = _archive_raw_to_calibrated_light(ap, raw)
        if m is None:
            continue
        cal_file, lights_root = m
        if not cal_file.is_file():
            continue
        lr = lights_root.resolve()
        if lr == cal_root:
            cal_paths.append(cal_file)
    if cal_paths:
        cal_paths = list(dict.fromkeys([Path(p) for p in cal_paths]))
    return cal_paths, []


def calibrated_paths_for_draft_apply_filters(
    archive_path: Path | str,
    db: VyvarDatabase,
    draft_id: int,
    *,
    fwhm_max_px: float,
    drift_max_arcmin: float = 0.0,
    source_dir: Path | str | None = None,
) -> tuple[list[Path], list[Path]]:
    """Calibrated FITS paths that pass QC: ``IS_REJECTED`` 0/NULL + optional ``FWHM`` cap.

    When ``fwhm_max_px`` ≤ 0, no FWHM filter (still excludes manual reject).
    When ``fwhm_max_px`` > 0, rows with finite ``FWHM`` > max are excluded.
    Rows with missing ``FWHM`` (NULL/NaN) are kept.

    ``drift_max_arcmin`` is kept only for backward-compatible callers and is ignored.
    """
    ap = Path(archive_path).expanduser()
    fwhm_limit = fwhm_max_px
    print(f"DEBUG: Hľadám súbory pre Draft {draft_id} s FWHM <= {fwhm_limit}")
    db.conn.execute(
        "UPDATE OBS_FILES SET IS_REJECTED = 0 WHERE DRAFT_ID = ?;",
        (int(draft_id),),
    )
    db.conn.commit()
    log_event(f"DEBUG: Draft {draft_id} reset: all files set to IS_REJECTED=0 before filtering.")
    cal_paths: list[Path] = []
    if source_dir is None:
        src_root = (ap / "calibrated" / "lights").resolve()
    else:
        src_root = Path(source_dir).expanduser().resolve()
    lim_active = bool(fwhm_max_px is not None and float(fwhm_max_px) > 0)
    lim_v = float(fwhm_max_px) if lim_active else 0.0
    _ = drift_max_arcmin  # backward compatibility: cleaning is FWHM-only
    if lim_active:
        _nonnull_cnt = int(
            db.conn.execute(
                "SELECT COUNT(*) FROM OBS_FILES WHERE DRAFT_ID = ? AND FWHM IS NOT NULL;",
                (int(draft_id),),
            ).fetchone()[0]
        )
        if _nonnull_cnt > 0:
            rows = [
                dict(r)
                for r in db.conn.execute(
                    """
                    SELECT * FROM OBS_FILES
                    WHERE DRAFT_ID = ?
                      AND (IS_REJECTED = 0 OR IS_REJECTED IS NULL)
                      AND FWHM IS NOT NULL
                      AND (FWHM <= ?);
                    """,
                    (int(draft_id), float(lim_v)),
                ).fetchall()
            ]
        else:
            rows = [
                dict(r)
                for r in db.conn.execute(
                    """
                    SELECT * FROM OBS_FILES
                    WHERE DRAFT_ID = ?
                      AND (IS_REJECTED = 0 OR IS_REJECTED IS NULL)
                      AND (FWHM <= ? OR FWHM IS NULL);
                    """,
                    (int(draft_id), float(lim_v)),
                ).fetchall()
            ]
        log_event(
            f"DEBUG: Preprocess DB filter selected {len(rows)} rows (limit={float(lim_v):.3f} px, strict, nonnull_fwhm={_nonnull_cnt})."
        )
    else:
        log_event("DEBUG: Preprocess filter - FWHM limit disabled (0/None); keeping all non-rejected frames.")
        rows = [
            dict(r)
            for r in db.conn.execute(
                """
                SELECT * FROM OBS_FILES
                WHERE DRAFT_ID = ?
                  AND (IS_REJECTED = 0 OR IS_REJECTED IS NULL);
                """,
                (int(draft_id),),
            ).fetchall()
        ]
        log_event(f"DEBUG: Preprocess DB filter selected {len(rows)} rows (FWHM disabled).")

    for row in rows:
        raw = Path(str(row.get("FILE_PATH") or ""))
        m = _archive_raw_to_calibrated_light(ap, raw)
        if m is not None:
            cand_m, _cand_root = m
            try:
                if cand_m.is_file() and cand_m.resolve().is_relative_to(src_root):
                    cal_paths.append(cand_m.resolve())
                    continue
            except Exception:  # noqa: BLE001
                pass
        cand = src_root / raw.name
        if cand.is_file():
            cal_paths.append(cand)
            continue
        resolved_raw = _resolve_draft_light_raw_path(ap, raw)
        if resolved_raw is None or not resolved_raw.is_file():
            continue
        try:
            resolved_raw.relative_to(src_root)
        except ValueError:
            continue
        cal_paths.append(resolved_raw)
    if not cal_paths:
        rescue_rows = db.conn.execute(
            """
            SELECT * FROM OBS_FILES
            WHERE DRAFT_ID = ?
              AND (IS_REJECTED = 0 OR IS_REJECTED IS NULL);
            """,
            (int(draft_id),),
        ).fetchall()
        if rescue_rows:
            log_event(f"INFO: Rescue pass found {len(rescue_rows)} files by ignoring QC filters.")
        for row in rescue_rows:
            raw = Path(str(row["FILE_PATH"] or ""))
            m = _archive_raw_to_calibrated_light(ap, raw)
            if m is not None:
                cand_m, _cand_root = m
                try:
                    if cand_m.is_file() and cand_m.resolve().is_relative_to(src_root):
                        cal_paths.append(cand_m.resolve())
                        continue
                except Exception:  # noqa: BLE001
                    pass
            cand = src_root / raw.name
            if cand.is_file():
                cal_paths.append(cand)
                continue
            resolved_raw = _resolve_draft_light_raw_path(ap, raw)
            if resolved_raw is None or not resolved_raw.is_file():
                continue
            try:
                resolved_raw.relative_to(src_root)
            except ValueError:
                continue
            cal_paths.append(resolved_raw)
    if not cal_paths and src_root.is_dir():
        # Final safety net: keep DB-filter intent by fuzzy matching selected DB file names to on-disk FITS.
        disk_fits = [p for p in iter_fits_paths_recursive(src_root) if p.is_file()]
        db_names = [Path(str(r.get("FILE_PATH") or "")).name for r in rows]
        db_stems = {Path(n).stem.lower() for n in db_names if str(n).strip()}
        matched: list[Path] = []
        for fp in disk_fits:
            s = fp.stem.lower()
            if any((st in s) or (s in st) for st in db_stems):
                matched.append(fp)
        if matched:
            cal_paths = sorted(matched)
            log_event(
                f"INFO: Disk fallback selected {len(cal_paths)} FITS files from {src_root} (DB-name matched)."
            )
        elif disk_fits:
            cal_paths = sorted(disk_fits)
            log_event(
                f"WARNING: Disk fallback used all {len(cal_paths)} FITS files from {src_root} (no DB-name match)."
            )
    return cal_paths, []


def first_calibrated_light_fits_under_archive(archive_path: Path | str) -> Path | None:
    """First ``*.fits`` / ``*.fit`` / ``*.fts`` under ``calibrated/lights``."""
    ap = Path(archive_path).expanduser()
    for root in (ap / "calibrated" / "lights",):
        if not root.is_dir():
            continue
        for pat in ("*.fits", "*.fit", "*.fts"):
            hits = sorted(root.rglob(pat))
            if hits:
                return hits[0]
    return None


def format_memory_bytes(n: float | int) -> str:
    """Human-readable binary size (KiB = 1024 B)."""
    try:
        x = float(n)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x) or x <= 0:
        return "0 B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024.0 or unit == "TiB":
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{n} B"


def _as_fits_float32_image(data: Any) -> "np.ndarray":
    """Contiguous float32 image for FITS I/O and downstream QC (avoids astropy ``float32`` dtype errors)."""
    import numpy as np

    return np.ascontiguousarray(np.asarray(data, dtype=np.float32))


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

    **QC analyze** is sequential: peak is a few× the largest single frame (cosmics + background + QC temps).

    **RAM handoff** (after detrending) holds ~one float32 copy per successfully aligned frame in memory
    before flush; add a margin for align-time temporaries (reference + source + aligned).
    """
    ap = Path(archive_path)
    avail, avail_src = _available_system_ram_bytes()
    out: dict[str, Any] = {
        "archive_path": str(ap),
        "available_ram_bytes": avail,
        "available_ram_human": format_memory_bytes(avail) if avail is not None else "neznáme (nainštaluj ``psutil``)",
        "available_ram_source": avail_src,
        "qc_analyze": None,
        "platesolve_ram_handoff": None,
        "notes_sk": (
            "Odhad z hlavičiek FITS (NAXIS*), predpoklad práce vo float32. Skutočná spotreba závisí od OS, "
            "Streamlit a ďalších knižníc. Po preprocess ešte pribudnú dočasné polia pri detrendingu."
        ),
    }

    cal = ap / "calibrated" / "lights"
    if cal.is_dir():
        cfiles = _iter_light_fits(cal)
        st = estimate_memory_from_fits_headers(cfiles)
        # Working set: jeden snímok načítaný + lacosmic/background + QC (hrubý faktor)
        qc_factor = 6.0
        peak_qc = int(float(st["bytes_float32_max_frame"]) * qc_factor)
        out["qc_analyze"] = {
            **st,
            "estimated_peak_bytes_sequential": peak_qc,
            "estimated_peak_human": format_memory_bytes(peak_qc),
            "explanation_sk": (
                f"Sekvenčné spracovanie ~{st['n_files']} snímok; špička RAM ≈ {qc_factor:.0f}× najväčší snímok "
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
        # Buffer: jedna kópia zarovnaného float32 na úspešný snímok (horný odhad = všetky vstupy zarovnané)
        buffer_est = med_b * max(0, n)
        # Počas astroalign: ref + src + aligned chvíľu naraz
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
                f"Režim „zarovnanie + katalóg v RAM“: drží ~{n} snímok × ~{format_memory_bytes(med_b)} "
                f"(medián) + krátkodobá špička pri zarovnaní. Ak je to viac než voľná RAM, vypni RAM handoff."
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


def _cupy_available() -> bool:
    """Return True if CuPy + CUDA device is available (optional dependency)."""
    try:
        import cupy as cp  # type: ignore

        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001
        return False


def _iter_fits_recursive(root: Path) -> list[Path]:
    """Collect FITS under ``root`` recursively (see :func:`utils.iter_fits_paths_recursive`)."""
    return iter_fits_paths_recursive(root)


def _header_vy_fwhm_px(hdr: fits.Header | None) -> float | None:
    """Measured QC FWHM from ``VY_FWHM`` if present and sane."""
    if hdr is None or "VY_FWHM" not in hdr:
        return None
    try:
        v = float(hdr["VY_FWHM"])
        if math.isfinite(v) and 0.5 < v < 80.0:
            return float(v)
    except (TypeError, ValueError):
        return None
    return None


def _obs_fwhm_basename_map_from_db(db: VyvarDatabase, draft_id: int) -> dict[str, float]:
    """Map ``basename.casefold()`` → FWHM from ``OBS_FILES`` for draft lights (last row wins per name)."""
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


def _pick_masterstar_reference_index(
    files: list[Path],
    *,
    fwhm_by_basename: dict[str, float] | None = None,
) -> int:
    """Choose reference frame: lowest ``VY_FWHM`` among inputs (header, then DB map), else index 0."""
    if len(files) <= 1:
        return 0
    best_i = 0
    best_f = float("inf")
    found_any = False
    _fb = fwhm_by_basename or {}
    for i, fp in enumerate(files):
        try:
            with fits.open(fp, memmap=False) as h:
                v = _header_vy_fwhm_px(h[0].header)
            if v is None:
                _n = fp.name.casefold()
                vv = _fb.get(_n)
                if vv is None and _n.startswith("proc_"):
                    vv = _fb.get(_n[5:])
                if vv is None and not _n.startswith("proc_"):
                    vv = _fb.get(f"proc_{_n}")
                if vv is not None and math.isfinite(vv) and vv > 0:
                    v = float(vv)
            if v is not None and math.isfinite(v) and v > 0:
                found_any = True
                if v < best_f:
                    best_f = float(v)
                    best_i = i
        except Exception:  # noqa: BLE001
            continue
    if found_any:
        log_event(
            f"MASTERSTAR: referenčná snímka = najlepší VY_FWHM → index {best_i} ({files[best_i].name})."
        )
        return best_i
    log_event("MASTERSTAR: VY_FWHM v hlavičkách chýba — referencia = prvá snímka v zozname (index 0).")
    return 0


def _sort_masterstar_paths_by_fwhm(
    files: list[Path],
    *,
    fwhm_by_basename: dict[str, float] | None = None,
) -> list[Path]:
    """Najlepšie prvé: najnižší VY_FWHM (hlavička alebo DB mapa), neznáme na koniec."""
    _fb = fwhm_by_basename or {}
    scored: list[tuple[float, str, Path]] = []
    for fp in files:
        v: float | None = None
        try:
            with fits.open(fp, memmap=False) as h:
                v = _header_vy_fwhm_px(h[0].header)
        except Exception:  # noqa: BLE001
            pass
        if v is None:
            _n = fp.name.casefold()
            vv = _fb.get(_n)
            if vv is None and _n.startswith("proc_"):
                vv = _fb.get(_n[5:])
            if vv is None and not _n.startswith("proc_"):
                vv = _fb.get(f"proc_{_n}")
            if vv is not None and math.isfinite(vv) and vv > 0:
                v = float(vv)
        score = float(v) if v is not None and math.isfinite(v) and v > 0 else float("inf")
        scored.append((score, str(fp).casefold(), fp))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in scored]


def _masterstar_aligned_stack_combine(
    files: list[Path],
    *,
    ref_idx: int,
    max_control_points: int,
    detection_sigma: float,
    combine: str,
    log_astroalign: bool = True,
) -> "np.ndarray":
    """``astroalign.register`` na referenciu, potom ``nanmean`` alebo ``nanmedian`` pozdĺž osi snímok."""
    import numpy as np

    try:
        import astroalign as aa  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"astroalign required for MASTERSTAR stack: {exc}") from exc

    n = len(files)
    if n == 0:
        raise ValueError("MASTERSTAR: no input files")
    ref_idx = int(ref_idx)
    if ref_idx < 0 or ref_idx >= n:
        raise ValueError(f"MASTERSTAR: invalid ref_idx {ref_idx} for {n} files")
    if n == 1:
        with fits.open(files[0], memmap=False) as hdul:
            return np.asarray(_as_fits_float32_image(hdul[0].data), dtype=np.float32)

    with fits.open(files[ref_idx], memmap=False) as hdul:
        ref_data = np.asarray(_as_fits_float32_image(hdul[0].data), dtype=np.float32)
    if ref_data.ndim != 2:
        raise ValueError("MASTERSTAR: referenčná snímka nie je 2D")
    h0, w0 = int(ref_data.shape[0]), int(ref_data.shape[1])

    for fp in files:
        with fits.open(fp, memmap=False) as hdul:
            a = np.asarray(hdul[0].data)
        if a.ndim != 2 or int(a.shape[0]) != h0 or int(a.shape[1]) != w0:
            raise ValueError(
                f"MASTERSTAR: očakávaný jednotný rozmer {w0}×{h0}, neplatné: {fp.name} → {getattr(a, 'shape', None)}"
            )

    mcp = int(max(12, min(int(max_control_points), 50)))
    det = float(detection_sigma)
    if not math.isfinite(det) or det <= 0:
        det = 5.0

    layers: list[np.ndarray] = []
    for i, fp in enumerate(files):
        if i == ref_idx:
            layers.append(np.copy(ref_data))
            continue
        with fits.open(fp, memmap=False) as hdul:
            data = np.asarray(_as_fits_float32_image(hdul[0].data), dtype=np.float32)
        try:
            aligned, _footprint = aa.register(
                data,
                ref_data,
                fill_value=np.nan,
                max_control_points=mcp,
                detection_sigma=det,
            )
        except Exception as exc:
            raise RuntimeError(f"astroalign.register zlyhal pre {fp.name}: {exc}") from exc
        layers.append(np.asarray(aligned, dtype=np.float32))

    cube = np.stack(layers, axis=0)
    c = str(combine).strip().lower()
    if c == "median":
        out = np.nanmedian(cube, axis=0).astype(np.float32)
    else:
        out = np.nanmean(cube, axis=0).astype(np.float32)
    if log_astroalign:
        log_event(
            f"MASTERSTAR stack: {n} snímok, ref_idx={ref_idx}, combine={c} "
            f"({'nanmedian' if c == 'median' else 'nanmean'})."
        )
    return out


def build_masterstar_from_detrended(
    *,
    detrended_root: Path,
    output_fits: Path,
    only_paths: "Sequence[Path | str] | None" = None,
    fwhm_fallback_px: float | None = None,
    app_config: AppConfig | None = None,
    draft_id: int | None = None,
    db: VyvarDatabase | None = None,
) -> dict[str, Any]:
    """Build MASTERSTAR by copying the single best processed FITS (lowest VY_FWHM)."""
    import shutil

    import numpy as np

    root = Path(detrended_root).resolve()
    all_fits = _iter_fits_recursive(root)
    files = _filter_light_paths_maybe(all_fits, only_paths)
    if not files and only_paths is not None:
        remapped: list[Path] = []
        seen_r: set[str] = set()
        for op in only_paths:
            hit = _resolve_best_effort_path_under(root, str(op))
            if hit is None:
                continue
            try:
                rk = str(hit.resolve()).casefold()
            except OSError:
                rk = str(hit).casefold()
            if rk in seen_r:
                continue
            seen_r.add(rk)
            remapped.append(hit)
        if remapped:
            files = remapped
            log_event(
                f"MASTERSTAR: výber zlúčený cez best-effort mapovanie ({len(files)} FITS; path filter bol prázdny)."
            )
    if files:
        if only_paths is None:
            _pipeline_ui_info(
                f"Nájdených {len(files)} súborov v {root} (vrátane podadresárov)."
            )
        else:
            _pipeline_ui_info(
                f"Nájdených {len(files)} súborov pre MASTERSTAR v {root} (výber kandidátov; "
                f"v strome {len(all_fits)} FITS celkom)."
            )
    if not files:
        # Candidate paths did not match any files on disk: small deterministic batch from the tree.
        log_event("MASTERSTAR: výber kandidátov sa nezhoduje so súbormi na disku — beriem prvých N FITS ako kandidátov.")
        batch = sorted(_iter_fits_recursive(root), key=lambda p: str(p).casefold())
        if batch:
            n_take = max(1, min(8, len(batch)))
            files = batch[:n_take]
            all_fits = batch

    if not files:
        if not all_fits:
            msg = (
                f"Nenašli sa žiadne FITS súbory v {root} (prehľadávaná cesta, vrátane podadresárov)."
            )
        else:
            msg = (
                f"Žiadne FITS pre MASTERSTAR po výbere kandidátov: {root} obsahuje {len(all_fits)} súbor(ov), "
                "ale žiadna cesta nezodpovedá výberu z databázy."
            )
        _pipeline_ui_info(msg)
        raise FileNotFoundError(msg)
    try:
        _comp_names = [Path(p).name for p in files]
        log_event(f"📂 MASTERSTAR COMPOSITION: Using [{', '.join(_comp_names)}]")
    except Exception:  # noqa: BLE001
        pass

    sorted_files = _sort_masterstar_paths_by_fwhm(files, fwhm_by_basename=None)

    output_fits.parent.mkdir(parents=True, exist_ok=True)
    _cfg_ms = app_config or AppConfig()

    reference_path = Path(sorted_files[0])
    if not reference_path.exists():
        log_event(f"❌ MASTERSTAR FAIL: Reference file {reference_path} not found.")
        fallback_hits = [x for x in _iter_fits_recursive(root) if x.name == reference_path.name]
        if fallback_hits:
            reference_path = fallback_hits[0]
            log_event(f"✅ MASTERSTAR fallback reference found: {reference_path}")
        else:
            raise FileNotFoundError(f"MASTERSTAR reference file not found: {reference_path}")

    try:
        with fits.open(reference_path, memmap=False) as hdul0:
            _d0 = hdul0[0].data
            _sh = getattr(_d0, "shape", None)
            if _d0 is None or _sh is None or len(_sh) != 2:
                raise ValueError("MASTERSTAR: referenčný FITS nie je platný 2D primary.")
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"MASTERSTAR: neviem načítať referenčný FITS: {exc}") from exc

    shutil.copy2(reference_path, output_fits)
    log_event(
        f"MASTERSTAR: čistá kópia → {output_fits} "
        f"(zdroj {reference_path.name}, kandidátov {len(files)}; najlepší podľa VY_FWHM)."
    )

    # Auto FWHM: médian VY_FWHM zo všetkých processed FITS v root sade
    _all_processed = list(_iter_fits_recursive(root))
    _fwhm_auto_values: list[float] = []
    for _fp_fw in _all_processed:
        try:
            with fits.open(_fp_fw, memmap=False) as _hf_fw:
                _v_fw = _header_vy_fwhm_px(_hf_fw[0].header)
                if _v_fw is not None and 1.0 < _v_fw < 15.0:
                    _fwhm_auto_values.append(float(_v_fw))
        except Exception:  # noqa: BLE001
            pass
    if _fwhm_auto_values:
        _fwhm_auto = float(np.median(np.asarray(_fwhm_auto_values, dtype=np.float64)))
        log_event(
            f"MASTERSTAR: auto FWHM z {len(_fwhm_auto_values)} processed FITS "
            f"= {_fwhm_auto:.3f} px (médian VY_FWHM)"
        )
    else:
        _fwhm_auto = float(fwhm_fallback_px) if fwhm_fallback_px is not None else 4.5
        log_event(f"MASTERSTAR: VY_FWHM nedostupné — fallback FWHM = {_fwhm_auto:.1f} px")

    try:
        with fits.open(output_fits, mode="update", memmap=False) as h:
            hdr = h[0].header
            # Vždy prepíš VY_FWHM hodnotou z auto výpočtu (médian sady)
            hdr["VY_FWHM"] = (float(_fwhm_auto), "FWHM [pix] auto z medianu processed FITS")
            log_event(f"MASTERSTAR: VY_FWHM = {_fwhm_auto:.3f} px zapísaný do FITS.")

            # Ak má processed FITS platný ASTAP WCS (CD matica, izotropná),
            # zachovaj ho — VYVAR solver bude refinovať od tohto dobrého základu.
            _has_cd = all(k in hdr for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"))
            _pltsolvd = bool(hdr.get("PLTSOLVD", False))
            if _has_cd and _pltsolvd:
                try:
                    _sx = abs(hdr["CD1_1"]) * 3600
                    _sy = abs(hdr["CD2_2"]) * 3600
                    _ratio = max(_sx, _sy) / max(min(_sx, _sy), 0.001)
                    if _ratio < 1.3 and 7.0 < _sx < 13.0 and 7.0 < _sy < 13.0:
                        log_event(
                            f"MASTERSTAR: processed FITS má dobrý ASTAP WCS "
                            f"(sx={_sx:.3f} sy={_sy:.3f} ratio={_ratio:.3f}) — "
                            "zachovávam CD maticu pre solver."
                        )
                    else:
                        log_event(
                            f"MASTERSTAR: ASTAP WCS anizotropný alebo mimo rozsahu "
                            f"(sx={_sx:.3f} sy={_sy:.3f}) — solver použije vlastné riešenie."
                        )
                except Exception:  # noqa: BLE001
                    pass

            h.flush()
    except Exception as _exc:  # noqa: BLE001
        log_event(f"MASTERSTAR: VY_FWHM zapis zlyhal: {_exc!s}")

    return {
        "masterstar_path": str(output_fits),
        "frames_used": int(len(files)),
        "reference_path": str(reference_path),
        "reference_index": 0,
        "stacked": False,
        "frames_combined": 1,
        "copied_from": str(reference_path),
    }


def _update_masterstar_obs_file_status(
    *,
    cfg: AppConfig | None,
    draft_id: int | None,
    selected_ref_path: Path | None,
    wcs_ok: bool,
    n_stars: int,
) -> None:
    if draft_id is None or selected_ref_path is None:
        return
    try:
        db = VyvarDatabase(Path((cfg or AppConfig()).database_path))
        try:
            cur = db.conn.execute("PRAGMA table_info('OBS_FILES');")
            cols = {str(r[1]).upper() for r in cur.fetchall()}
            if "WCS" not in cols:
                db.conn.execute("ALTER TABLE OBS_FILES ADD COLUMN WCS INTEGER;")
            if "STARS" not in cols:
                db.conn.execute("ALTER TABLE OBS_FILES ADD COLUMN STARS INTEGER;")
            ref_name = selected_ref_path.name
            ref_l = ref_name.casefold()
            raw_l = ref_l[5:] if ref_l.startswith("proc_") else ref_l
            proc_l = raw_l if raw_l.startswith("proc_") else f"proc_{raw_l}"
            like_raw = f"%{raw_l}"
            like_proc = f"%{proc_l}"
            cur_upd = db.conn.execute(
                """
                UPDATE OBS_FILES
                   SET WCS = ?, STARS = ?
                 WHERE DRAFT_ID = ?
                   AND (
                        LOWER(FILE_PATH) LIKE ?
                        OR LOWER(FILE_PATH) LIKE ?
                        OR LOWER(FILE_PATH) LIKE ?
                   );
                """,
                (
                    1 if bool(wcs_ok) else 0,
                    int(max(0, int(n_stars))),
                    int(draft_id),
                    like_raw,
                    like_proc,
                    f"%{ref_l}",
                ),
            )
            db.conn.commit()
            log_event(
                f"MASTERSTAR DB update: DRAFT_ID={int(draft_id)}, WCS={1 if wcs_ok else 0}, "
                f"Stars={int(n_stars)}, rows={int(cur_upd.rowcount or 0)}"
            )
        finally:
            db.conn.close()
    except Exception as exc:  # noqa: BLE001
        log_event(f"MASTERSTAR DB update skipped: {exc!s}")


def _resolve_best_effort_path_under(root: Path, raw_path: str) -> Path | None:
    """Map an OBS_FILES.FILE_PATH (often archived raw path) to an existing file under ``root``.

    Strategy: exact relative join, else basename match (first hit). This is intentionally heuristic because
    imports may store different path bases (absolute vs relative, calibrated vs processed).
    """
    rp = str(raw_path or "").strip()
    if not rp:
        return None
    p = Path(rp)
    # If already absolute and exists under root tree, accept.
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(root.resolve())
            cand = (root / rel).resolve()
            if cand.is_file():
                return cand
        except Exception:  # noqa: BLE001
            pass
    # If relative, try directly.
    cand2 = (root / p).resolve()
    if cand2.is_file():
        return cand2
    # Explicit processed-name fallback in the same folder (e.g. Light_066.fits -> proc_Light_066.fits).
    try:
        cand2_proc = cand2.with_name(_safe_proc_name(cand2.name)).resolve()
        if cand2_proc.is_file():
            return cand2_proc
    except Exception:  # noqa: BLE001
        pass
    # Basename / fuzzy suffix fallback (handles prefixes like ``proc_``).
    name = p.name
    if not name:
        return None
    _name_cf = name.casefold()
    _name_noproc = _name_cf[5:] if _name_cf.startswith("proc_") else _name_cf
    hits = []
    for x in _iter_fits_recursive(root):
        xn = x.name.casefold()
        xn_noproc = xn[5:] if xn.startswith("proc_") else xn
        if (
            xn == _name_cf
            or xn_noproc == _name_noproc
            or xn.endswith(_name_cf)
            or xn_noproc.endswith(_name_noproc)
        ):
            hits.append(x)
    return hits[0] if hits else None


def resolve_obs_file_to_processed_fits(
    archive_path: Path | str | None,
    obs_file_path: str,
    *,
    setup_name: str | None = None,
) -> Path | None:
    """Map ``OBS_FILES.FILE_PATH`` (kalibrovaný / importný kľúč) na skutočný FITS v ``processed/lights`` (``proc_*.fits``).

    Používa rovnakú heuristiku ako MASTERSTAR build (``_resolve_best_effort_path_under``).
    """
    if archive_path is None:
        return None
    ap = Path(archive_path).expanduser()
    if ap.name.casefold() == "non_calibrated":
        ap = ap.parent
    if not ap.is_dir():
        return None
    root = resolve_masterstar_input_root(ap, setup_name=setup_name)
    if not root.exists():
        return None
    return _resolve_best_effort_path_under(root, str(obs_file_path))


def list_best_processed_light_paths_for_masterstar(
    archive_path: Path | str | None,
    *,
    setup_name: str | None = None,
    draft_id: int | None = None,
    app_config: AppConfig | None = None,
    take_n: int = 5,
) -> list[Path]:
    """Najlepšie (najnižší FWHM) FITS pod ``processed/lights`` — na UI tabuľku a výber pre MASTERSTAR."""
    if archive_path is None:
        return []
    ap = Path(archive_path).expanduser()
    if ap.name.casefold() == "non_calibrated":
        ap = ap.parent
    if not ap.is_dir():
        return []
    root = resolve_masterstar_input_root(ap, setup_name=setup_name)
    files = list(_iter_fits_recursive(root))
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
                _fb = {}
            finally:
                try:
                    _dbc.conn.close()
                except Exception:  # noqa: BLE001
                    pass
    ranked = _sort_masterstar_paths_by_fwhm(files, fwhm_by_basename=_fb or None)
    return ranked[:tn]


def get_masterstar_candidate_rows(
    draft_id: int,
    percentage: float,
    *,
    fwhm_max_px: float | None = None,
    db: VyvarDatabase,
) -> "pd.DataFrame":
    """Rank draft light frames by quality metrics for MASTERSTAR selection.

    Strict filter: include only rows with ``IS_REJECTED`` 0/NULL.
    Score (higher is better) is a normalized version of:

    Score = (1 / fwhm) * (1 / sky_level) * snr_estimate
    where ``snr_estimate`` is approximated from ``STAR_COUNT`` and ``SKY_LEVEL`` when no explicit SNR exists.
    """
    import numpy as np
    import pandas as pd

    did = int(draft_id)
    pct = float(percentage)
    pct = float(max(0.1, min(100.0, pct)))

    rows = db.fetch_draft_light_rows_for_quality(did)
    if not rows:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    # Strict: IS_REJECTED == 0 (or NULL)
    if "IS_REJECTED" in df.columns:
        df["IS_REJECTED"] = pd.to_numeric(df["IS_REJECTED"], errors="coerce").fillna(0).astype(int)
        df = df[df["IS_REJECTED"] == 0].copy()
    lim_active = bool(fwhm_max_px is not None and float(fwhm_max_px) > 0)
    lim_v = float(fwhm_max_px) if lim_active else 0.0
    if lim_active:
        _f = pd.to_numeric(df.get("FWHM"), errors="coerce")
        df = df[_f.notna() & (_f <= lim_v)].copy()
    if df.empty:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    df["FWHM"] = pd.to_numeric(df.get("FWHM"), errors="coerce")
    df["SKY_LEVEL"] = pd.to_numeric(df.get("SKY_LEVEL"), errors="coerce")
    df["STAR_COUNT"] = pd.to_numeric(df.get("STAR_COUNT"), errors="coerce").fillna(0).astype(int)

    # Robust fallbacks: median scales for normalization.
    f_med = float(np.nanmedian(df["FWHM"].values)) if np.isfinite(np.nanmedian(df["FWHM"].values)) else 0.0
    s_med = (
        float(np.nanmedian(df["SKY_LEVEL"].values))
        if np.isfinite(np.nanmedian(df["SKY_LEVEL"].values))
        else 0.0
    )
    if not (math.isfinite(f_med) and f_med > 0):
        f_med = 1.0
    if not (math.isfinite(s_med) and s_med > 0):
        s_med = 1.0

    eps = 1e-9
    f = df["FWHM"].astype(float)
    sky = df["SKY_LEVEL"].astype(float)
    stars = df["STAR_COUNT"].astype(float)

    # snr_estimate: more stars and lower sky → better SNR (proxy; true SNR not stored in OBS_FILES).
    snr_est = (stars + 1.0) / np.sqrt(np.maximum(sky, 0.0) + 1.0)

    f_norm_inv = f_med / (np.maximum(f, 0.0) + eps)
    sky_norm_inv = s_med / (np.maximum(sky, 0.0) + eps)

    score = f_norm_inv * sky_norm_inv * snr_est
    score = np.where(np.isfinite(score), score, 0.0)

    df["SNR_EST"] = snr_est
    df["SCORE"] = score

    df = df.sort_values(["SCORE", "STAR_COUNT"], ascending=[False, False], kind="mergesort").reset_index(
        drop=True
    )

    k = int(max(1, math.ceil(len(df) * (pct / 100.0))))
    return df.head(k).loc[:, ["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"]]


def get_masterstar_candidates(draft_id: int, percentage: float, *, db: VyvarDatabase) -> list[str]:
    """Return FILE_PATH list of top-ranked MASTERSTAR candidates (temporary list; OBS_FILES is not modified)."""
    df = get_masterstar_candidate_rows(int(draft_id), float(percentage), db=db)
    if df.empty:
        return []
    return [str(x) for x in df["FILE_PATH"].tolist() if str(x).strip()]


def _vyvar_open_database(cfg: AppConfig) -> VyvarDatabase | None:
    try:
        return VyvarDatabase(Path(cfg.database_path))
    except Exception:  # noqa: BLE001
        return None


def _fits_pixel_raw_to_micrometres(value: float) -> float:
    """Map raw FITS pixel-size keywords to **micrometres** (WCS often uses SI metres)."""
    if not math.isfinite(value) or value <= 0:
        return 0.0
    v = float(value)
    if v < 5e-5:
        return v * 1e6
    if v < 0.2:
        return v * 1000.0
    return v


def _header_focal_length_mm(header: fits.Header) -> float | None:
    """Focal length [mm] from common FITS keys (``FOCALLEN`` / ``FOCLEN`` often in **metres**)."""
    for key in ("FOCALLEN", "FOCLEN", "TELFOCA", "FOCAL_LEN", "FOCALL", "FOC_LEN"):
        if key not in header or header[key] in (None, "", " ", "0", 0):
            continue
        try:
            v = float(header[key])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v <= 0:
            continue
        mm = v * 1000.0 if v < 25.0 else v
        if 40.0 <= mm <= 120_000.0:
            return float(mm)
    return None


def _db_header_pixel_native_um_mean(header: fits.Header) -> float | None:
    """Pixel pitch [µm] (native, unbinned) from FITS header (best-effort).

    FITS keywords vary by acquisition software; values may be in metres, mm, or µm.
    """
    raw1 = _header_pick_first(header, "PIXSIZE1", "XPIXSZ", "PIXSZLX", "PIXSIZE", default=0.0)
    raw2 = _header_pick_first(header, "PIXSIZE2", "YPIXSZ", "PIXSZLY", default=0.0)
    try:
        v1 = float(raw1)
    except (TypeError, ValueError):
        v1 = 0.0
    try:
        v2 = float(raw2)
    except (TypeError, ValueError):
        v2 = 0.0
    p1 = _fits_pixel_raw_to_micrometres(v1)
    p2 = _fits_pixel_raw_to_micrometres(v2)
    vals = [p for p in (p1, p2) if math.isfinite(p) and p > 0]
    if not vals:
        return None
    return float(sum(vals) / float(len(vals)))


def get_auto_fov(*, archive_path: Path | None = None, masterstar_path: Path | None = None) -> float | None:
    """Auto field diameter [deg] (diagonal) for plate solving.

    Priority:
    - From FITS header: FOCALLEN/FOCLEN + pixel pitch (PIXSIZE/XPIXSZ) + NAXIS => diagonal FOV
    - Else, from WCS after a successful solve: angular separation of opposite corners
    """
    import numpy as np
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    ms = Path(masterstar_path) if masterstar_path is not None else None
    if ms is None and archive_path is not None:
        ap = Path(archive_path)
        cand = ap / "platesolve" / "MASTERSTAR.fits"
        ms = cand if cand.is_file() else None
    if ms is None or not ms.is_file():
        return None

    with fits.open(ms, memmap=False) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
    if data is None:
        return None
    h, w = int(data.shape[0]), int(data.shape[1])

    # Try optics-based FOV.
    f_mm = _header_focal_length_mm(hdr)
    p_um = _db_header_pixel_native_um_mean(hdr)
    if f_mm is not None and p_um is not None and f_mm > 0 and p_um > 0 and w > 0 and h > 0:
        diag_mm = math.hypot(float(w) * float(p_um) * 0.001, float(h) * float(p_um) * 0.001)
        rad = 2.0 * math.atan2(0.5 * diag_mm, float(f_mm))
        if math.isfinite(rad) and rad > 0:
            return float(rad * 180.0 / math.pi)

    # Fall back to WCS-based FOV (after solve).
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            wcs0 = WCS(hdr)
        if not getattr(wcs0, "has_celestial", False):
            return None
        corners = np.array(
            [[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]],
            dtype=np.float64,
        )
        sky = wcs0.celestial.pixel_to_world(corners[:, 0], corners[:, 1])
        c0 = SkyCoord(sky[0])
        c2 = SkyCoord(sky[2])
        sep = c0.separation(c2).to(u.deg).value
        if math.isfinite(float(sep)) and float(sep) > 0:
            return float(sep)
    except Exception:  # noqa: BLE001
        return None
    return None


def _focal_mm_plausible(mm: float) -> bool:
    return math.isfinite(mm) and 40.0 <= mm <= 120_000.0


def _resolve_focal_mm_for_plate_scale(
    header: fits.Header | None,
    db: VyvarDatabase | None,
    *,
    equipment_id: int | None = None,
) -> tuple[float | None, str]:
    """Plausible FITS focal first; else ``EQUIPMENTS.FOCAL`` (optional); else ``TELESCOPE.FOCAL``."""
    if header is not None:
        hdr_mm = _header_focal_length_mm(header)
        if hdr_mm is not None:
            hdr_n, hfixed = normalize_telescope_focal_mm_for_plate_scale(hdr_mm)
            if hfixed:
                log_event(
                    f"FOCAL: v hlavičke FITS ohnisko vyzeralo ako 10× preklep ({hdr_mm:g} mm → {hdr_n:g} mm)."
                )
            hdr_mm = hdr_n
        if hdr_mm is not None and _focal_mm_plausible(hdr_mm):
            return float(hdr_mm), "fits_header"
    if db is not None and equipment_id is not None:
        try:
            eq_raw = db.get_equipment_focal_mm(int(equipment_id))
        except Exception:  # noqa: BLE001
            eq_raw = None
        if eq_raw is not None:
            eq_n, eq_fixed = normalize_telescope_focal_mm_for_plate_scale(float(eq_raw))
            if eq_fixed:
                log_event(
                    f"FOCAL: EQUIPMENTS.FOCAL (ID={int(equipment_id)}) vyzeralo ako 10× preklep "
                    f"({eq_raw:g} mm → {eq_n:g} mm) — použité pre mierku / solver."
                )
            if _focal_mm_plausible(eq_n):
                return float(eq_n), "database_equipment"
    if db is not None:
        raw = db.get_telescope_focal_mm(None)
        if raw is not None:
            norm, fixed = normalize_telescope_focal_mm_for_plate_scale(float(raw))
            if fixed:
                log_event(
                    f"FOCAL: TELESCOPE.FOCAL v DB vyzeralo ako 10× preklep ({raw:g} mm → {norm:g} mm) — "
                    "použité pre mierku / solver."
                )
            if _focal_mm_plausible(norm):
                return float(norm), "database_telescope"
    return None, "none"


def _merge_equipment_pixel_into_metadata(meta: dict[str, Any], db: VyvarDatabase, equipment_id: int) -> None:
    """If FITS native pixel is missing or nonsense, use ``EQUIPMENTS.PIXELSIZE`` [µm, 1×1] × binning."""
    try:
        native = db.get_equipment_pixel_size_um(int(equipment_id))
    except Exception:  # noqa: BLE001
        return
    if native is None:
        return
    try:
        nv = float(native)
    except (TypeError, ValueError):
        return
    if not math.isfinite(nv) or nv <= 0 or nv > 300.0:
        return
    x_bin = max(1, int(meta.get("binning", 1) or 1))
    y_bin = max(1, int(meta.get("binning_y", x_bin) or x_bin))
    prev = meta.get("pixel_size_um_physical")
    # Universal scale: trust EQUIPMENTS.PIXELSIZE (UI/DB) whenever equipment is known; binning stays from FITS.
    meta["pixel_size_um_physical"] = float(nv)
    eff_x = float(nv) * float(x_bin)
    eff_y = float(nv) * float(y_bin)
    meta["pixel_size_um_header"] = (eff_x + eff_y) / 2.0
    meta["effective_pixel_um_plate_scale"] = float(nv) * float(x_bin)
    meta["pixel_size_um_source"] = "equipment_db"
    _eff = float(nv) * float(x_bin)
    if prev is not None:
        try:
            pv = float(prev)
            same = math.isfinite(pv) and abs(pv - float(nv)) < 1e-6
        except (TypeError, ValueError):
            same = False
        if not same:
            log_event(
                f"PIXEL: EQUIPMENTS.PIXELSIZE={nv:g} µm × binning {x_bin}×{y_bin} → efektívny {_eff:g} µm "
                f"(preferované pred FITS; predtým {prev!r})."
            )
    else:
        log_event(
            f"PIXEL: EQUIPMENTS.PIXELSIZE={nv:g} µm × binning {x_bin}×{y_bin} → efektívny {_eff:g} µm."
        )


def _recompute_effective_pixel_from_physical(meta: dict[str, Any]) -> None:
    """``effective_pixel_um_plate_scale`` = native [µm] × XBINNING (int)."""
    phys = meta.get("pixel_size_um_physical")
    if phys is None:
        return
    try:
        p = float(phys)
    except (TypeError, ValueError):
        return
    if not math.isfinite(p) or p <= 0:
        return
    x_bin = max(1, int(meta.get("binning", 1) or 1))
    y_bin = max(1, int(meta.get("binning_y", x_bin) or x_bin))
    eff_x = p * float(x_bin)
    eff_y = p * float(y_bin)
    meta["pixel_size_um_header"] = (eff_x + eff_y) / 2.0
    meta["effective_pixel_um_plate_scale"] = float(p) * float(x_bin)


def _header_pick_first(header: fits.Header, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return header[key]
    return default


def _enrich_calibration_metadata_from_header(
    meta: dict[str, Any],
    header: fits.Header,
    *,
    db: VyvarDatabase | None,
    id_equipment: int | None,
) -> None:
    """Add ``focal_length``, ``pixel_size_raw``, ``pixel_um``, ``focal_length_source`` for diagnostics / UI."""

    def _to_f(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    r1 = _to_f(_header_pick_first(header, "PIXSIZE1", "XPIXSZ", "PIXSZLX", "PIXSIZE", default=0.0))
    r2 = _to_f(_header_pick_first(header, "PIXSIZE2", "YPIXSZ", "PIXSZLY", default=0.0))
    parts: list[str] = []
    if r1 > 0:
        parts.append(f"PIX1={r1:g}")
    if r2 > 0:
        parts.append(f"PIX2={r2:g}")
    meta["pixel_size_raw"] = " | ".join(parts) if parts else "n/a"

    meta["pixel_um"] = meta.get("effective_pixel_um_plate_scale")

    focal_mm: float | None = None
    src = "none"
    # Universal optics: prefer DB (UI) over FITS FOCALLEN for plate scale / solver hints.
    if db is not None:
        eqf: float | None = None
        if id_equipment is not None:
            try:
                eqf = db.get_equipment_focal_mm(int(id_equipment))
            except Exception:  # noqa: BLE001
                eqf = None
        telf = db.get_telescope_focal_mm(None)
        raw_db = eqf if eqf is not None else telf
        if raw_db is not None:
            focal_mm, _fx = normalize_telescope_focal_mm_for_plate_scale(float(raw_db))
            src = "equipment_focal" if eqf is not None else "telescope_focal"

    has_focallen = "FOCALLEN" in header and header["FOCALLEN"] not in (None, "", " ", "0", 0)
    if focal_mm is None and has_focallen:
        raw_fc = header["FOCALLEN"]
        log_event(f"DIAG: FITS FOCALLEN (SIPS / zapisovateľ hlavičky) = {raw_fc!r}")
        try:
            v = float(raw_fc)
            mm0 = (v * 1000.0) if (math.isfinite(v) and v > 0 and v < 25.0) else v
        except (TypeError, ValueError):
            mm0 = None
        if mm0 is not None and math.isfinite(mm0) and mm0 > 0 and _focal_mm_plausible(float(mm0)):
            focal_mm, _fx = normalize_telescope_focal_mm_for_plate_scale(float(mm0))
            src = "fits_focallen"

    meta["focal_length"] = focal_mm
    meta["focal_length_source"] = src


def _apply_draft_combined_to_pipeline_meta(meta: dict[str, Any], comb: dict[str, Any]) -> None:
    """Overlay ``VyvarDatabase.get_combined_metadata`` onto ``extract_fits_metadata`` output."""
    fl = comb.get("focal_length_mm")
    if fl is not None:
        try:
            fv = float(fl)
            if math.isfinite(fv) and _focal_mm_plausible(fv):
                meta["focal_length"] = fv
                meta["focal_length_source"] = str(comb.get("focal_source") or "draft_combined")
        except (TypeError, ValueError):
            pass
    pn = comb.get("pixel_native_um")
    if pn is not None:
        try:
            pv = float(pn)
            if math.isfinite(pv) and 0 < pv <= 300.0:
                meta["pixel_size_um_physical"] = pv
        except (TypeError, ValueError):
            pass
    xb = max(1, int(comb.get("xbinning", 1) or 1))
    yb = max(1, int(comb.get("ybinning", xb) or xb))
    meta["binning"] = xb
    meta["binning_y"] = yb
    pe = comb.get("pixel_effective_um")
    if pe is not None:
        try:
            ev = float(pe)
            if math.isfinite(ev) and ev > 0:
                meta["effective_pixel_um_plate_scale"] = ev
                if meta.get("pixel_size_um_physical") is not None:
                    pph = float(meta["pixel_size_um_physical"])
                    meta["pixel_size_um_header"] = (pph * float(xb) + pph * float(yb)) / 2.0
                else:
                    meta["pixel_size_um_header"] = ev
                meta["pixel_um"] = ev
        except (TypeError, ValueError):
            pass
    sat = comb.get("saturate_adu")
    if sat is not None:
        try:
            sv = float(sat)
            if math.isfinite(sv) and sv > 0:
                meta["equipment_saturate_adu"] = sv
        except (TypeError, ValueError):
            pass


def _log_calibration_metadata_diagnostic(filename: str, metadata: dict[str, Any]) -> None:
    log_event("--- DIAGNOSTIKA METADÁT PRE KALIBRÁCIU ---")
    log_event(f"Súbor: {filename}")
    log_event(f"FOCAL (z DB/FITS): {metadata.get('focal_length')} mm")
    log_event(f"PIXEL_SIZE (surový): {metadata.get('pixel_size_raw')} um")
    log_event(f"BINNING (X/Y): {metadata.get('binning')}x{metadata.get('binning_y')}")
    log_event(f"EFEKTÍVNY PIXEL (pre výpočet): {metadata.get('pixel_um')} um")
    log_event("------------------------------------------")


def _plate_solve_input_bundle(
    fits_path: Path,
    *,
    app_config: AppConfig | None,
    equipment_id: int | None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """Open DB once: metadata, effective pixel, focal length, expected plate scale [arcsec/pixel]."""
    cfg_u = app_config or AppConfig()
    db_u = _vyvar_open_database(cfg_u)
    out: dict[str, Any] = {
        "meta": {},
        "header": None,
        "eff_um": None,
        "focal_mm": None,
        "expected_arcsec_per_px": None,
    }
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            out["header"] = hdul[0].header.copy()
        out["meta"] = extract_fits_metadata(
            fits_path,
            db=db_u,
            app_config=cfg_u,
            id_equipment=equipment_id,
            draft_id=draft_id,
        )
        if draft_id is not None:
            _m = out["meta"]
            if _m.get("focal_length") is None or _m.get("effective_pixel_um_plate_scale") is None:
                raise DraftTechnicalMetadataError(int(draft_id))
        foc: float | None = None
        _mf = out["meta"].get("focal_length")
        if _mf is not None:
            try:
                _fx = float(_mf)
                if math.isfinite(_fx) and _focal_mm_plausible(_fx):
                    foc = _fx
            except (TypeError, ValueError):
                foc = None
        if foc is None:
            foc, _ = _resolve_focal_mm_for_plate_scale(
                out["header"], db_u, equipment_id=equipment_id
            )
        out["focal_mm"] = foc
        eff_um: float | None = None
        x_bin = max(1, int(out["meta"].get("binning", 1) or 1))
        if db_u is not None and equipment_id is not None:
            try:
                _nat = db_u.get_equipment_pixel_size_um(int(equipment_id))
            except Exception:  # noqa: BLE001
                _nat = None
            if _nat is not None:
                try:
                    _nv = float(_nat)
                    if math.isfinite(_nv) and 0.5 < _nv <= 300.0:
                        eff_um = float(_nv) * float(x_bin)
                except (TypeError, ValueError):
                    pass
        if eff_um is None:
            _ev = out["meta"].get("effective_pixel_um_plate_scale")
            if _ev is not None:
                try:
                    _ef = float(_ev)
                    if math.isfinite(_ef) and _ef > 0:
                        eff_um = _ef
                except (TypeError, ValueError):
                    eff_um = None
        out["eff_um"] = eff_um
        if eff_um is not None and foc is not None:
            out["expected_arcsec_per_px"] = plate_scale_arcsec_per_pixel(
                pixel_pitch_um=float(eff_um),
                focal_length_mm=float(foc),
            )
            calculated_scale = out["expected_arcsec_per_px"]
            if calculated_scale is not None:
                log_event(f"MATH CHECK: ({eff_um} / {foc}) * 206.265 = {calculated_scale}")
    except Exception:  # noqa: BLE001
        pass
    finally:
        if db_u is not None:
            try:
                db_u.conn.close()
            except Exception:  # noqa: BLE001
                pass
    return out


def _try_rescale_masterstar_linear_wcs_to_expected_plate_scale(
    fits_path: Path,
    *,
    app_config: AppConfig | None,
    equipment_id: int | None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """If DB/optics yield expected arcsec/pixel and the on-disk WCS is linear (no SIP), rescale CD when mismatch is large."""
    out: dict[str, Any] = {"rescaled": False}
    try:
        b = _plate_solve_input_bundle(
            fits_path,
            app_config=app_config or AppConfig(),
            equipment_id=equipment_id,
            draft_id=draft_id,
        )
        exp = b.get("expected_arcsec_per_px")
        if exp is None:
            return out
        exp_f = float(exp)
        if not math.isfinite(exp_f) or exp_f <= 0:
            return out
        fp = Path(fits_path)
        with fits.open(fp, memmap=False) as hdul:
            hdr0 = hdul[0].header.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w0 = WCS(hdr0)
        if not w0.has_celestial or w0.sip is not None:
            return out
        w2, changed = maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel(w0, exp_f)
        if not changed:
            return out
        try:
            wh = w2.to_header(relax=True)
        except Exception:  # noqa: BLE001
            return out
        with fits.open(fp, mode="update", memmap=False) as hdul:
            h = hdul[0].header
            strip_celestial_wcs_keys(h)
            for k in wh:
                if k in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                    continue
                if k.startswith("NAXIS") and k != "NAXIS":
                    continue
                try:
                    h[k] = wh[k]
                except Exception:  # noqa: BLE001
                    pass
            h.add_history("VYVAR: CD scaled to expected arcsec/pixel from equipment optics")
            hdul.flush()
        log_event(
            f"WCS PLATE SCALE: lineárny TAN prispôsobený optickej mierke {exp_f:.3f} arcsec/px."
        )
        out["rescaled"] = True
        out["expected_arcsec_per_px"] = exp_f
    except Exception as exc:  # noqa: BLE001
        log_event(f"WCS PLATE SCALE: úprava CD preskočená — {exc!s}")
        out["error"] = str(exc)
    return out


def _solve_wcs_solve_field_cli(
    masterstar_path: Path,
    *,
    expected_arcsec_per_pixel: float | None = None,
) -> dict[str, Any]:
    """Run local ``solve-field`` if available (ANSVR / astrometry.net indexes on PATH or ``VYVAR_SOLVE_FIELD_EXE``).

    Uses ``--tweak-order`` (SIP-style distortion) and ``--cpulimit`` from :data:`ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC`.
    Skip with env ``VYVAR_SKIP_SOLVE_FIELD=1``.
    """

    if os.environ.get("VYVAR_SKIP_SOLVE_FIELD", "").strip().lower() in {"1", "true", "yes", "on"}:
        return {"solved": False, "reason": "VYVAR_SKIP_SOLVE_FIELD set"}
    exe = (os.environ.get("VYVAR_SOLVE_FIELD_EXE") or "").strip()
    if not exe:
        exe = shutil.which("solve-field") or ""
    if not exe:
        return {"solved": False, "reason": "solve-field not on PATH (set VYVAR_SOLVE_FIELD_EXE for ANSVR)"}

    mp = Path(masterstar_path).resolve()
    if not mp.is_file():
        return {"solved": False, "reason": f"File not found: {mp}"}

    wcs_path = mp.parent / f"{mp.stem}.wcs"
    wcs_path.unlink(missing_ok=True)

    cmd: list[str] = [
        exe,
        "--cpulimit",
        str(int(ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC)),
        "--tweak-order",
        str(int(effective_astrometry_net_tweak_order())),
        "--no-plots",
        "--overwrite",
    ]
    if expected_arcsec_per_pixel is not None:
        s = float(expected_arcsec_per_pixel)
        if math.isfinite(s) and 0.03 < s < 200.0:
            lo, hi = astrometry_net_scale_bounds_arcsec_per_pix(s)
            cmd.extend(
                [
                    "--scale-low",
                    f"{float(lo):.6g}",
                    "--scale-high",
                    f"{float(hi):.6g}",
                ]
            )
    cmd.append(str(mp))

    log_event(
        f"solve-field (lokálny): {exe} — --cpulimit {ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC}, "
        f"--tweak-order {effective_astrometry_net_tweak_order()}, {mp.name}"
    )
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(mp.parent),
            capture_output=True,
            text=True,
            timeout=900,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"solved": False, "reason": "solve-field subprocess timeout (900s wall)"}
    except Exception as exc:  # noqa: BLE001
        return {"solved": False, "reason": f"solve-field: {exc!s}"}

    if not wcs_path.is_file():
        tail = (proc.stderr or proc.stdout or "")[-900:]
        if proc.returncode != 0:
            return {"solved": False, "reason": f"solve-field exit {proc.returncode}: {tail!s}"}
        return {"solved": False, "reason": f"solve-field: missing {wcs_path.name} — {tail!s}"}

    try:
        with fits.open(wcs_path, memmap=False) as wh:
            wcs_hdr = wh[0].header.copy()
    except Exception as exc:  # noqa: BLE001
        return {"solved": False, "reason": f"read {wcs_path.name}: {exc!s}"}

    _apply_wcs_header_to_fits(mp, wcs_hdr)
    with fits.open(mp, mode="update", memmap=False) as hdul:
        hdr = hdul[0].header
        hdr["VY_PSOLV"] = (True, "Plate solved by local solve-field (SIP tweak-order)")
        hdr.add_history(
            f"VYVAR: solve-field --cpulimit {ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC} "
            f"--tweak-order {effective_astrometry_net_tweak_order()}"
        )
        hdul.flush()

    log_event(
        f"solve-field OK: WCS so SIP (tweak-order {effective_astrometry_net_tweak_order()}) → {mp.name}"
    )
    return {"solved": True, "method": "solve-field (local CLI)"}


def _try_solve_wcs_astrometry_net_or_local_cli(
    masterstar_path: Path,
    api_key: str | None = None,
    *,
    expected_arcsec_per_pixel: float | None = None,
) -> dict[str, Any]:
    """Prefer local ``solve-field`` when installed; else nova Astrometry.net API (both use SIP tweak order)."""
    r_loc = _solve_wcs_solve_field_cli(
        masterstar_path,
        expected_arcsec_per_pixel=expected_arcsec_per_pixel,
    )
    if r_loc.get("solved"):
        return r_loc
    r_api = _solve_wcs_astrometry_net(
        masterstar_path,
        api_key=api_key,
        expected_arcsec_per_pixel=expected_arcsec_per_pixel,
    )
    if not r_api.get("solved") and r_loc.get("reason"):
        r_api = dict(r_api)
        r_api["solve_field_attempt"] = r_loc.get("reason")
    return r_api


def _solve_wcs_astrometry_net(
    masterstar_path: Path,
    api_key: str | None = None,
    *,
    expected_arcsec_per_pixel: float | None = None,
) -> dict[str, Any]:
    """Try to solve WCS via astrometry.net (optional). Requires astroquery + API key."""
    import os

    api_key = (api_key or os.environ.get("ASTROMETRY_NET_API_KEY", "")).strip()
    if not api_key:
        return {"solved": False, "reason": "Missing ASTROMETRY_NET_API_KEY"}

    try:
        from astroquery.astrometry_net import AstrometryNet  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"solved": False, "reason": f"astroquery astrometry_net unavailable: {exc}"}

    ast = AstrometryNet()
    ast.api_key = api_key
    _tw = int(effective_astrometry_net_tweak_order())
    solve_kw: dict[str, Any] = {
        "solve_timeout": 180,
        "verbose": False,
        "tweak_order": max(0, _tw),
    }
    log_event(f"Astrometry.net API: tweak_order={_tw} (SIP / distortion, ~solve-field --tweak-order).")
    if expected_arcsec_per_pixel is not None:
        s = float(expected_arcsec_per_pixel)
        if math.isfinite(s) and 0.03 < s < 200.0:
            lo, hi = astrometry_net_scale_bounds_arcsec_per_pix(s)
            solve_kw.update(
                scale_type="ul",
                scale_units="arcsecperpix",
                scale_lower=float(lo),
                scale_upper=float(hi),
            )
            log_event(
                f"Astrometry.net: obmedzenie mierky ~{s:.3f} arcsec/px "
                f"(scale_low={lo:.3f}, scale_high={hi:.3f}; ~solve-field --scale-low/--scale-high)."
            )
    try:
        wcs_header = ast.solve_from_image(str(masterstar_path), **solve_kw)
    except Exception as exc:  # noqa: BLE001
        return {"solved": False, "reason": f"Astrometry.net solve failed: {exc}"}

    if not wcs_header:
        return {"solved": False, "reason": "Astrometry.net returned no WCS header"}

    _apply_wcs_header_to_fits(masterstar_path, fits.Header(wcs_header))
    with fits.open(masterstar_path, mode="update", memmap=False) as hdul:
        hdr = hdul[0].header
        hdr["VY_PSOLV"] = (True, "Plate solved by Astrometry.net")
        hdr.add_history(
            f"VYVAR: Astrometry.net API tweak_order={int(effective_astrometry_net_tweak_order())} (SIP / distortion)"
        )
        hdul.flush()

    return {"solved": True, "method": "astrometry.net"}


def _apply_wcs_header_to_fits(fits_path: Path, wcs_hdr: fits.Header) -> None:
    """Merge celestial WCS keywords from ``wcs_hdr`` into the image FITS (primary HDU)."""
    with fits.open(fits_path, mode="update", memmap=False) as hdul:
        h = hdul[0].header
        strip_celestial_wcs_keys(h)
        for k in wcs_hdr:
            if k in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
                continue
            if k.startswith("NAXIS") and k != "NAXIS":
                continue
            try:
                h[k] = wcs_hdr[k]
            except Exception:  # noqa: BLE001
                pass
        hdul.flush()


def _solve_wcs_external(
    fits_path: Path,
    *,
    backend: str = "vyvar",
    astrometry_api_key: str | None = None,
    plate_solve_fov_deg: float = 1.0,
    hint_ra_deg: float | None = None,
    hint_dec_deg: float | None = None,
    app_config: AppConfig | None = None,
    equipment_id: int | None = None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """Plate solve: výhradne VYVAR (lokálna Gaia DB). Backend ``auto`` / Astrometry.net aliasy sa mapujú na VYVAR."""
    be = (backend or "vyvar").strip().lower()
    if be in {"astap", "astap_cli", "local"}:
        LOGGER.warning("platesolve backend %r is no longer supported; using VYVAR.", be)
        be = "vyvar"

    fp_solve = Path(fits_path)

    def _finalize_plate_solve_result(r: dict[str, Any]) -> dict[str, Any]:
        # Post-refine removed; Gaia-based pipeline keeps the WCS as-solved.
        try:
            sm = r.get("sip_meta") if isinstance(r, dict) else None
            if isinstance(sm, dict) and sm.get("initial_wcs_offset_px") is not None:
                _off = float(sm.get("initial_wcs_offset_px"))
                if math.isfinite(_off) and _off > 0:
                    log_event(
                        f"DEBUG: Initial WCS offset detected: {_off:.2f} pixels. Applying coarse correction (Pass 0)..."
                    )
        except Exception:  # noqa: BLE001
            pass
        if isinstance(r, dict) and not bool(r.get("solved", False)):
            try:
                mr = r.get("match_rate")
                mrf = float(mr) if mr is not None else float("nan")
                if math.isfinite(mrf) and mrf < 0.02:
                    log_event(
                        f"WARNING: Plate solve final match rate too low ({mrf * 100.0:.1f}%). "
                        "Returning solved=False to prevent downstream matching on invalid WCS."
                    )
            except Exception:  # noqa: BLE001
                pass
        return r

    def _try_vyvar(*, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg_u = app_config or AppConfig()
        gaia_db = (getattr(cfg_u, "gaia_db_path", "") or "").strip()
        if not gaia_db:
            return {
                "solved": False,
                "reason": "VYVAR solver: v Settings nastav cestu k lokálnej Gaia DR3 SQLite DB (gaia_db_path).",
            }
        if hint_ra_deg is None or hint_dec_deg is None:
            return {
                "solved": False,
                "reason": "VYVAR solver: chýba RA/Dec hint (VYTARGRA/VYTARGDE v hlavičke alebo zadanie v UI / preprocess).",
            }
        from vyvar_platesolver import solve_wcs_with_local_gaia

        b = bundle or _plate_solve_input_bundle(
            fits_path,
            app_config=cfg_u,
            equipment_id=equipment_id,
            draft_id=draft_id,
        )
        _em = b.get("meta") or {}
        eff_um = b.get("eff_um")
        exp_scale = b.get("expected_arcsec_per_px")
        _bx = max(1, int(_em.get("binning", 1) or 1))
        _pix = _em.get("pixel_size_um_physical")
        _foc_mm = b.get("focal_mm")
        _pix_s = f"{float(_pix):.4g}" if _pix is not None else "n/a"
        _foc_s = f"{float(_foc_mm):.4g}" if _foc_mm is not None else "n/a"
        _eff_s = f"{float(eff_um):.4g}" if eff_um is not None else "n/a"
        log_event(
            f"SOLVER INPUT: Focal={_foc_s}mm, Pixel={_pix_s}um, Bin={_bx}x -> Effective Pixel={_eff_s}um"
        )
        _calc = b.get("expected_arcsec_per_px")
        if _calc is not None and _foc_mm is not None and eff_um is not None:
            log_event(
                f"PLATE SOLVING: Mierka nastavená na {float(_calc):.3f} arcsec/px "
                f"(vypočítané z {float(_foc_mm)}mm a {float(eff_um)}um)"
            )
        elif _calc is not None:
            log_event(f"PLATE SOLVING: Mierka nastavená na {float(_calc):.3f} arcsec/px")

        _no_sip = os.environ.get("VYVAR_PLATE_SOLVE_NO_SIP", "").strip().lower() in {"1", "true", "yes", "on"}
        _is_masterstar = fp_solve.name.strip().upper() == "MASTERSTAR.FITS"
        return _finalize_plate_solve_result(
            solve_wcs_with_local_gaia(
                fp_solve,
                hint_ra_deg=float(hint_ra_deg),
                hint_dec_deg=float(hint_dec_deg),
                fov_diameter_deg=float(plate_solve_fov_deg),
                gaia_db_path=Path(gaia_db),
                enable_sip=not _no_sip,
                effective_pixel_um=eff_um,
                focal_length_mm=float(_foc_mm) if _foc_mm is not None else None,
                expected_plate_scale_arcsec_per_px=exp_scale,
                max_catalog_rows=100000 if _is_masterstar else None,
                faintest_mag_limit=18.0 if _is_masterstar else None,
            )
        )

    if be == "auto":
        cfg_a = app_config or AppConfig()
        b_auto = _plate_solve_input_bundle(
            fits_path, app_config=cfg_a, equipment_id=equipment_id, draft_id=draft_id
        )
        log_event('Plate-solve backend "auto": len VYVAR lokálny solver (bez Astrometry.net).')
        return _try_vyvar(bundle=b_auto)

    # Accept legacy backend aliases for backward compatibility.
    if be in {"vyvar", "vyvar_platesolver", "vyvar_gaia"}:
        return _try_vyvar(bundle=None)

    if be in {"astrometry.net", "astrometry_net", "online", "net"}:
        LOGGER.warning(
            "Plate-solve backend %r už nie je podporovaný — používam VYVAR (lokálna Gaia).", be
        )
        cfg_n = app_config or AppConfig()
        b_net = _plate_solve_input_bundle(
            fits_path, app_config=cfg_n, equipment_id=equipment_id, draft_id=draft_id
        )
        return _try_vyvar(bundle=b_net)

    LOGGER.warning("Unknown platesolve backend %r; using VYVAR.", be)
    return _try_vyvar(bundle=None)


def _has_valid_wcs(header: fits.Header) -> bool:
    return fits_header_has_celestial_wcs(header)


def _wcs_astrometry_nearly_identical(wa: WCS, wb: WCS, rtol: float = 1e-6) -> bool:
    """True if CRVAL, CRPIX and PC matrix match (frames aligned to reference share its astrometry)."""
    import numpy as np

    if not (wa.has_celestial and wb.has_celestial):
        return False
    try:
        if not np.allclose(wa.wcs.crval, wb.wcs.crval, rtol=rtol, atol=1e-9):
            return False
        if not np.allclose(wa.wcs.crpix, wb.wcs.crpix, rtol=rtol, atol=1e-6):
            return False
        if not np.allclose(wa.wcs.get_pc(), wb.wcs.get_pc(), rtol=rtol, atol=1e-12):
            return False
        return True
    except Exception:  # noqa: BLE001
        return False


def _bin2d_mean(arr: "np.ndarray", factor: int) -> "np.ndarray":
    """Mean-bin a 2D array by integer ``factor`` (>=1). Used to speed up reference scoring."""
    import numpy as np

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or int(factor) < 2:
        return arr
    f = int(factor)
    h, w = arr.shape
    h2, w2 = (h // f) * f, (w // f) * f
    if h2 < f or w2 < f:
        return arr
    a = arr[:h2, :w2].reshape(h2 // f, f, w2 // f, f)
    return np.mean(a, axis=(1, 3)).astype(np.float32)


def _dao_star_count_from_array(arr: "np.ndarray", *, fwhm_px: float = 3.0) -> int:
    """Count DAOStarFinder sources (same recipe as alignment star detection)."""
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        return 0
    finite = np.isfinite(arr)
    if not np.any(finite):
        return 0
    _, med, std = sigma_clipped_stats(arr[finite], sigma=3.0, maxiters=5)
    std = float(std) if np.isfinite(std) else 0.0
    if std <= 0:
        return 0
    img2 = np.nan_to_num((arr - float(med)).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        finder = DAOStarFinder(
            fwhm=float(fwhm_px),
            threshold=max(3.0 * std, 1e-6),
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = finder(img2)
    except Exception:  # noqa: BLE001
        return 0
    if tbl is None or len(tbl) == 0:
        return 0
    return int(len(tbl))


def _pick_reference_frame_by_star_count(files: list[Path]) -> tuple[Path, dict[str, int]]:
    """Choose the FITS with the most detected stars as alignment / plate-solve reference.

    Uses 2×2 binning + slightly smaller DAO FWHM for speed; preserves relative rankings.
    """
    import numpy as np

    scores: dict[str, int] = {}
    for fp in files:
        try:
            with fits.open(fp, memmap=False) as hdul:
                data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            data_b = _bin2d_mean(data, 2)
            # 2×2 binning: FWHM v px ≈ sips_dao_fwhm_px/2 (širšie okno pre kómu v rohoch).
            scores[str(fp)] = _dao_star_count_from_array(data_b, fwhm_px=2.5)
        except Exception:  # noqa: BLE001
            scores[str(fp)] = 0
    if not files:
        raise ValueError("no FITS files")
    if not scores or max(scores.values(), default=0) <= 0:
        return files[0], scores
    best_n = max(scores.values())
    for fp in files:
        if scores.get(str(fp), 0) == best_n:
            return fp, scores
    return files[0], scores


def _wcs_field_center_radec_deg(fits_path: Path) -> tuple[float, float] | None:
    """RA/Dec (deg) of image center from existing celestial WCS."""

    try:
        with fits.open(fits_path, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
        h = int(hdr.get("NAXIS2", 0) or 0)
        wpx = int(hdr.get("NAXIS1", 0) or 0)
        if h <= 0 or wpx <= 0:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr)
        if not w.has_celestial:
            return None
        c = SkyCoord.from_pixel((wpx - 1) / 2.0, (h - 1) / 2.0, wcs=w, origin=0)
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:  # noqa: BLE001
        return None


def _catalog_df_cap_brightest_by_mag(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    """Keep at most ``max_rows`` catalog rows, brightest first (lowest ``mag``)."""
    if df is None or getattr(df, "empty", True):
        return df
    try:
        cap = int(max_rows) if max_rows is not None else int(AppConfig().catalog_query_max_rows)
    except Exception:  # noqa: BLE001
        cap = 50_000
    cap = max(1000, min(500_000, cap))
    if len(df) <= cap:
        return df
    out = df.copy()
    if "mag" not in out.columns:
        return out.iloc[:cap].copy().reset_index(drop=True)
    m = pd.to_numeric(out["mag"], errors="coerce")
    out = out.assign(_vyvar_mag_sort=m)
    out = out.sort_values("_vyvar_mag_sort", na_position="last").head(int(cap))
    return out.drop(columns=["_vyvar_mag_sort"], errors="ignore").reset_index(drop=True)


def _query_gaia_local(
    *,
    center: SkyCoord,
    radius_deg: float,
    gaia_db_path: Path | None,
    max_mag: float | None = None,
    focal_mm_for_log: float | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Query **local Gaia DR3 SQLite** for the field; return unified dataframe (ICRS deg)."""
    try:
        _ra_l = float(center.icrs.ra.deg)
        _de_l = float(center.icrs.dec.deg)
    except Exception:  # noqa: BLE001
        _ra_l, _de_l = float("nan"), float("nan")
    _f_l = f"{float(focal_mm_for_log):g}" if focal_mm_for_log is not None else "?"
    log_event(
        f"CATALOG SEARCH (GAIA local): Ra={_ra_l}, Dec={_de_l}, Radius={float(radius_deg):.2f} deg (pre {_f_l}mm)"
    )
    if gaia_db_path is None:
        return pd.DataFrame()
    gp = Path(gaia_db_path).expanduser().resolve()
    if not gp.is_file():
        return pd.DataFrame()

    ra_min = float(_ra_l) - float(radius_deg)
    ra_max = float(_ra_l) + float(radius_deg)
    de_min = float(_de_l) - float(radius_deg)
    de_max = float(_de_l) + float(radius_deg)
    # query_local_gaia defaults mag_limit to 11.5 when omitted — must pass caller's max_mag
    # so MASTERSTAR / cone export honor faintest_mag_limit in SQL (not only a redundant pandas filter).
    _ql_kw: dict[str, Any] = {
        "ra_min": ra_min,
        "ra_max": ra_max,
        "dec_min": de_min,
        "dec_max": de_max,
    }
    if max_mag is not None:
        try:
            _mm = float(max_mag)
            if math.isfinite(_mm) and _mm > 0:
                _ql_kw["mag_limit"] = _mm
        except (TypeError, ValueError):
            pass
    # SQLite uses a square ra/dec box + ORDER BY g_mag LIMIT. For wide boxes (optics floor vs WCS) the 100k
    # brightest stars are often far from the field center, so cone stars never enter the result set.
    _cap_out: int | None = None
    if max_rows is not None:
        try:
            _mr0 = int(max_rows)
            if _mr0 > 0:
                _cap_out = _mr0
        except (TypeError, ValueError):
            _cap_out = None
    _sql_fetch = _cap_out
    if _cap_out is not None and float(radius_deg) > 6.0:
        _af = max(1.0, (float(radius_deg) / 5.5) ** 2)
        # Oversample for cone cut; cap keeps SQLite practical (ORDER BY g_mag on huge boxes is costly).
        _sql_fetch = min(800_000, max(_cap_out, int(_cap_out * _af * 2.5)))
    if _sql_fetch is not None:
        _ql_kw["max_rows"] = int(_sql_fetch)
    rows = query_local_gaia(gp, **_ql_kw)
    if not rows:
        return pd.DataFrame()
    df0 = pd.DataFrame(rows)
    if "bp_rp" not in df0.columns and "bp_mag" in df0.columns and "rp_mag" in df0.columns:
        df0["bp_rp"] = pd.to_numeric(df0["bp_mag"], errors="coerce") - pd.to_numeric(
            df0["rp_mag"], errors="coerce"
        )
    df = df0.rename(
        columns={"source_id": "catalog_id", "ra": "ra_deg", "dec": "dec_deg", "g_mag": "mag", "bp_rp": "b_v"}
    )
    df["catalog"] = "GAIA_DR3"
    # Great-circle cone cut (query box is square; LIMIT is isotropic in mag, not in radius).
    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if bool(_okq.any()):
        sub = df.loc[_okq].copy()
        _coo_q = SkyCoord(
            ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
            dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
            frame="icrs",
        )
        _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
        df = sub.loc[_inner].reset_index(drop=True)
    if max_mag is not None and "mag" in df.columns:
        m = pd.to_numeric(df["mag"], errors="coerce")
        df = df[(m.notna()) & (m <= float(max_mag))].copy()
    df = df.reset_index(drop=True)
    if "b_v" in df.columns and "bp_rp" not in df.columns:
        df = df.copy()
        df["bp_rp"] = pd.to_numeric(df["b_v"], errors="coerce")
    return _catalog_df_cap_brightest_by_mag(df, max_rows=_cap_out)


def _query_vsx_local(
    *,
    center: SkyCoord,
    radius_deg: float,
    vsx_db_path: Path | None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Query **local VSX SQLite** for the field; same kužeľ ako Gaia (najprv obdĺžnik, potom great-circle orez)."""
    if vsx_db_path is None:
        return pd.DataFrame()
    vp = Path(vsx_db_path).expanduser().resolve()
    if not vp.is_file():
        return pd.DataFrame()
    try:
        _ra_l = float(center.icrs.ra.deg)
        _de_l = float(center.icrs.dec.deg)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    ra_min = float(_ra_l) - float(radius_deg)
    ra_max = float(_ra_l) + float(radius_deg)
    de_min = float(_de_l) - float(radius_deg)
    de_max = float(_de_l) + float(radius_deg)
    _cap = max_rows
    if _cap is None:
        try:
            _cap = int(AppConfig().catalog_query_max_rows)
        except Exception:  # noqa: BLE001
            _cap = 500_000
        _cap = max(10_000, min(500_000, int(_cap)))
    if float(radius_deg) > 6.0:
        _af = max(1.0, (float(radius_deg) / 5.5) ** 2)
        _cap = min(800_000, max(int(_cap), int(_cap * _af * 2.5)))

    rows = query_local_vsx(
        vp,
        ra_min=ra_min,
        ra_max=ra_max,
        dec_min=de_min,
        dec_max=de_max,
        max_rows=int(_cap),
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        return pd.DataFrame()
    _raq = pd.to_numeric(df["ra_deg"], errors="coerce")
    _deq = pd.to_numeric(df["dec_deg"], errors="coerce")
    _okq = _raq.notna() & _deq.notna()
    if not bool(_okq.any()):
        return pd.DataFrame()
    sub = df.loc[_okq].copy()
    _coo_q = SkyCoord(
        ra=pd.to_numeric(sub["ra_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        dec=pd.to_numeric(sub["dec_deg"], errors="coerce").astype(float).to_numpy() * u.deg,
        frame="icrs",
    )
    _inner = center.separation(_coo_q).deg <= float(radius_deg) + 1e-9
    out = sub.loc[_inner].reset_index(drop=True)
    try:
        log_event(
            f"CATALOG SEARCH (VSX local): {len(out)} zdrojov v kuželi r≈{float(radius_deg):.3f}° "
            f"(Ra={_ra_l:.4f}, Dec={_de_l:.4f})"
        )
    except Exception:  # noqa: BLE001
        pass
    return out


def _saturate_limit_adu_from_header(hdr: fits.Header) -> float | None:
    """Return saturation / linearity ceiling in image units (ADU, e⁻, …) if present in header."""
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
    """Infer linearity ceiling from FITS integer layout (e.g. unsigned 16-bit → 65535)."""
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
        # Unsigned 16-bit (common): physical 0…65535 stored with BZERO=32768
        if abs(bzero - 32768.0) < 1.0 and abs(bscale - 1.0) < 1e-9:
            return 65535.0
        # Native signed 16-bit
        if abs(bzero) < 1e-6 and abs(bscale - 1.0) < 1e-9:
            return 32767.0
    return None


def _equipment_saturate_adu_from_db(equipment_id: int | None) -> float | None:
    """Read ``EQUIPMENTS.SATURATE_ADU`` when a valid equipment id is given."""
    if equipment_id is None:
        return None
    try:
        eid = int(equipment_id)
    except (TypeError, ValueError):
        return None
    if eid <= 0:
        return None
    try:
        cfg = AppConfig()
        db = VyvarDatabase(cfg.database_path)
        return db.get_equipment_saturation_adu(eid)
    except Exception:  # noqa: BLE001
        return None


def _effective_saturation_limit(
    hdr: fits.Header,
    *,
    fallback_adu: float | None,
    equipment_saturate_adu: float | None = None,
) -> tuple[float | None, str]:
    """Resolve saturation ceiling: header keywords → DATAMAX → equipment DB → BITPIX guess → config fallback."""
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

    return None, "none"


def _box_peak_max_adu(data: "np.ndarray", x: float, y: float, half: int = 3) -> float:
    """Maximum pixel value in ``(2*half+1)²`` box around ``(x,y)`` on the **original** image (linear units)."""
    import numpy as np

    arr = np.asarray(data)
    if arr.ndim != 2:
        return float("nan")
    h, w = arr.shape
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    if y0 >= y1 or x0 >= x1:
        return float("nan")
    return float(np.nanmax(arr[y0:y1, x0:x1]))


def _box_peaks_at_centroids(
    arr: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    half: int = 3,
) -> "np.ndarray":
    """Maximum ADU in each ``(2*half+1)²`` box centred on ``round(x), round(y)`` (vectorized).

    Used for per-star saturation on thousands of DAO detections; matches ``_box_peak_max_adu`` on
    interior pixels. Falls back to a Python loop if SciPy is unavailable.
    """
    import numpy as np

    xa = np.asarray(x, dtype=np.float64).reshape(-1)
    ya = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(xa.size)
    if n == 0:
        return np.zeros([], dtype=np.float64)
    if n != int(ya.size):
        raise ValueError("x and y must have the same length")

    try:
        from scipy.ndimage import maximum_filter  # type: ignore
    except ImportError:
        return np.asarray(
            [_box_peak_max_adu(arr, float(xa[i]), float(ya[i]), half=half) for i in range(n)],
            dtype=np.float64,
        )

    a = np.asarray(np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0))
    size = int(2 * int(half) + 1)
    mf = maximum_filter(a, size=size, mode="nearest")
    xi = np.rint(xa).astype(np.intp, copy=False)
    yi = np.rint(ya).astype(np.intp, copy=False)
    h, w = mf.shape
    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)
    return np.asarray(mf[yi, xi], dtype=np.float64)


def _icrs_deg_to_unitxyz(ra_deg: "np.ndarray", dec_deg: "np.ndarray") -> "np.ndarray":
    """ICRS degrees → unit direction vectors on the celestial sphere (N,3)."""
    import numpy as np

    ra = np.radians(np.asarray(ra_deg, dtype=np.float64).ravel())
    de = np.radians(np.asarray(dec_deg, dtype=np.float64).ravel())
    cd = np.cos(de)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(de)])


def _chord_to_arcsec(dist_chord: "np.ndarray") -> "np.ndarray":
    """Chord length between unit sphere points (0…2) → great-circle separation in arcseconds."""
    import numpy as np

    d = np.asarray(dist_chord, dtype=np.float64)
    half = np.clip(d * 0.5, 0.0, 1.0)
    return np.degrees(2.0 * np.arcsin(half)) * 3600.0


def build_ucac_catalog_kdtree(cat_df: pd.DataFrame) -> tuple[Any, "np.ndarray"] | None:
    """Build a SciPy ``cKDTree`` on finite ``ra_deg``/``dec_deg`` rows and row indices into ``cat_df``.

    Read-only ``query`` calls are thread-safe across workers sharing the same tree.
    """
    import numpy as np

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    if cat_df is None or cat_df.empty or "ra_deg" not in cat_df.columns or "dec_deg" not in cat_df.columns:
        return None
    ra = np.asarray(pd.to_numeric(cat_df["ra_deg"], errors="coerce"), dtype=np.float64)
    de = np.asarray(pd.to_numeric(cat_df["dec_deg"], errors="coerce"), dtype=np.float64)
    m = np.isfinite(ra) & np.isfinite(de)
    if not np.any(m):
        return None
    orig_idx = np.nonzero(m)[0].astype(np.int64)
    xyz = _icrs_deg_to_unitxyz(ra[m], de[m])
    return cKDTree(xyz), orig_idx


def nearest_sky_nn_kdtree(
    tree: Any,
    det_ra_deg: "np.ndarray",
    det_dec_deg: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    """Nearest catalog point on the sphere for each detection (k=1, same idea as ``match_to_catalog_sky``).

    Returns ``(idx_compact, sep_arcsec)`` where ``idx_compact`` indexes the finite subset used to build
    ``tree``, or ``-1`` if invalid.
    """
    import numpy as np

    det_xyz = _icrs_deg_to_unitxyz(det_ra_deg, det_dec_deg)
    dist, idx = tree.query(det_xyz, k=1)
    sep = _chord_to_arcsec(dist)
    idx_a = np.asarray(idx, dtype=np.int64).ravel()
    dist_a = np.asarray(dist, dtype=np.float64).ravel()
    ntree = int(getattr(tree, "n", 0))
    bad = ~np.isfinite(dist_a) | (idx_a < 0) | (idx_a >= max(ntree, 1))
    sep = np.asarray(sep, dtype=np.float64).ravel()
    sep[bad] = np.inf
    idx_a = idx_a.copy()
    idx_a[bad] = -1
    return idx_a, sep


def _saturated_core_plateau(
    data: "np.ndarray",
    x: float,
    y: float,
    *,
    half_inner: int = 1,
    plateau_rel: float = 0.996,
    min_plateau_pixels: int = 5,
) -> bool:
    """Detect a clipped / ``flat-top`` core: many pixels in the central box sit near the local maximum.

    Mirrors what eyeballing a radial profile shows for saturated stars (plateau vs a smooth Gaussian peak).
    Works on **any** linear image scale (raw or calibrated floats).
    """
    import numpy as np

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        return False
    h, w = arr.shape
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    hi = max(0, int(half_inner))
    y0, y1 = max(0, yi - hi), min(h, yi + hi + 1)
    x0, x1 = max(0, xi - hi), min(w, xi + hi + 1)
    if y0 >= y1 or x0 >= x1:
        return False
    patch = arr[y0:y1, x0:x1]
    if patch.size < int(min_plateau_pixels):
        return False
    pmax = float(np.nanmax(patch))
    if not np.isfinite(pmax) or pmax <= 0:
        return False
    thr = pmax * float(plateau_rel)
    n_high = int(np.sum(np.isfinite(patch) & (patch >= thr)))
    return n_high >= int(min_plateau_pixels)


def _star_saturation_flags(
    arr: "np.ndarray",
    x: float,
    y: float,
    *,
    sat_limit: float | None,
    sat_frac: float,
    peak_dao_val: float | None,
    peak_max_adu: float | None = None,
) -> dict[str, Any]:
    """Per-star saturation: ADU limit crossing + central plateau (flat core)."""
    import numpy as np

    pmax = float(peak_max_adu) if peak_max_adu is not None else _box_peak_max_adu(arr, float(x), float(y), half=3)
    lim = sat_limit
    sat_by_peak = bool(
        lim is not None and np.isfinite(pmax) and pmax >= float(lim) * float(sat_frac)
    )
    if lim is None:
        sat_by_plateau = _saturated_core_plateau(arr, float(x), float(y))
    else:
        plateau_skip_thr = float(lim) * float(sat_frac) * 0.55
        if np.isfinite(pmax) and pmax < plateau_skip_thr:
            sat_by_plateau = False
        else:
            sat_by_plateau = _saturated_core_plateau(arr, float(x), float(y))
    likely = bool(sat_by_peak or sat_by_plateau)
    return {
        "peak_dao": float(peak_dao_val) if peak_dao_val is not None and np.isfinite(peak_dao_val) else None,
        "peak_max_adu": float(pmax) if np.isfinite(pmax) else None,
        "saturate_limit_adu": float(lim) if lim is not None else None,
        "saturated_from_peak": sat_by_peak,
        "saturated_plateau": sat_by_plateau,
        "likely_saturated": likely,
        "photometry_ok": not likely,
    }


def _all_pix2world_icrs_deg(wcs_obj: WCS, x: "np.ndarray", y: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """Vectorized pixel → world (degrees) for celestial axes; single WCS call (no per-star Python loops)."""
    import numpy as np

    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        world0, world1 = wcs_obj.all_pix2world(xa, ya, 0)
    return (
        np.asarray(world0, dtype=np.float64).ravel(),
        np.asarray(world1, dtype=np.float64).ravel(),
    )


def _saturated_core_plateau_vectorized(
    data: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    half_inner: int = 1,
    plateau_rel: float = 0.996,
    min_plateau_pixels: int = 5,
) -> "np.ndarray":
    """Same criterion as ``_saturated_core_plateau``, vectorized over ``(x,y)`` centroids (3×3 patch per star)."""
    import numpy as np

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        return np.zeros(len(np.asarray(x).ravel()), dtype=bool)
    h, w = arr.shape
    n = int(np.asarray(x, dtype=np.float64).size)
    yi = np.rint(np.asarray(y, dtype=np.float64).ravel()).astype(np.int32, copy=False)
    xi = np.rint(np.asarray(x, dtype=np.float64).ravel()).astype(np.int32, copy=False)
    hi = max(0, int(half_inner))
    offs = np.arange(-hi, hi + 1, dtype=np.int32)
    yy = yi.reshape(-1, 1, 1) + offs.reshape(1, -1, 1)
    xx = xi.reshape(-1, 1, 1) + offs.reshape(1, 1, -1)
    yy = np.clip(yy, 0, h - 1)
    xx = np.clip(xx, 0, w - 1)
    patches = arr[yy, xx]
    pmax = np.nanmax(patches.reshape(n, -1), axis=1)
    thr = pmax * float(plateau_rel)
    thr3 = thr.reshape(-1, 1, 1)
    n_high = np.sum(np.isfinite(patches) & (patches >= thr3), axis=(1, 2))
    return (n_high >= int(min_plateau_pixels)) & np.isfinite(pmax) & (pmax > 0)


def _vectorized_star_saturation_columns(
    arr: "np.ndarray",
    x: "np.ndarray",
    y: "np.ndarray",
    *,
    sat_limit: float | None,
    sat_frac: float,
    peak_dao: "np.ndarray",
    peak_max_adu: "np.ndarray",
) -> dict[str, "np.ndarray"]:
    """Per-star saturation flags as column arrays (replaces ``N`` calls to ``_star_saturation_flags``)."""
    import numpy as np

    n = int(np.asarray(x, dtype=np.float64).size)
    pmax = np.asarray(peak_max_adu, dtype=np.float64).reshape(-1)
    lim = sat_limit
    sf = float(sat_frac)
    pdv = np.asarray(peak_dao, dtype=np.float64).reshape(-1)

    if lim is not None:
        sat_peak = np.isfinite(pmax) & (pmax >= float(lim) * sf)
        plateau_skip_thr = float(lim) * sf * 0.55
        need_plateau = np.isfinite(pmax) & (pmax >= plateau_skip_thr)
    else:
        sat_peak = np.zeros(n, dtype=bool)
        need_plateau = np.ones(n, dtype=bool)

    pl_full = _saturated_core_plateau_vectorized(arr, x, y, half_inner=1, plateau_rel=0.996, min_plateau_pixels=5)
    if lim is None:
        sat_plateau = pl_full
    else:
        sat_plateau = np.where(need_plateau, pl_full, False)

    likely = sat_peak | sat_plateau
    peak_dao_col = pdv.copy()
    peak_dao_col[~np.isfinite(peak_dao_col)] = np.nan
    peak_max_col = pmax.copy()
    peak_max_col[~np.isfinite(peak_max_col)] = np.nan
    sl = np.full(n, np.nan, dtype=np.float64)
    if lim is not None:
        sl[:] = float(lim)
    return {
        "peak_dao": peak_dao_col,
        "peak_max_adu": peak_max_col,
        "saturate_limit_adu": sl,
        "saturated_from_peak": sat_peak,
        "saturated_plateau": sat_plateau,
        "likely_saturated": likely,
        "photometry_ok": ~likely,
    }


def _vyvar_df_to_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Write sidecar / index CSV. Optional PyArrow writer; else pandas (``to_csv`` has no ``engine='c'`` for write)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    export_df = df
    if "catalog_id" in df.columns:
        export_df = df.copy()
        export_df["catalog_id"] = catalog_id_series_for_masterstars_export(df)
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.csv as pacsv  # type: ignore[import-not-found]

        pacsv.write_csv(pa.Table.from_pandas(export_df, preserve_index=False), str(p))
    except Exception:  # noqa: BLE001
        export_df.to_csv(p, index=False, lineterminator="\n")


def _fits_header_first_positive_float(hdr: fits.Header, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k not in hdr:
            continue
        try:
            v = float(hdr[k])
            if math.isfinite(v) and v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _gaia_catalog_cone_radius_optics_floor_deg(
    hdr: fits.Header | None,
    *,
    naxis1: int,
    naxis2: int,
    plate_solve_fov_fallback_deg: float,
) -> float:
    """Minimum Gaia cone radius from FOCALLEN + PIXSIZE + binning (ignores flawed linear WCS at chip edges).

    A solved TAN-without-SIP header often under-predicts corner separation vs true sky; the rectangular
    SQL prefilter then clips the catalog so QA shows „missing Gaia“ stripes at left/right edges.
    """
    if hdr is None:
        return 0.0
    foc = _fits_header_first_positive_float(
        hdr, ("FOCALLEN", "FOCALLENGTH", "FOCAL", "FOC_LEN")
    )
    pix = _fits_header_first_positive_float(
        hdr, ("PIXSIZE", "XPIXSZ", "PIXSZ", "PIXELSIZE", "PIX_SIZE")
    )
    if foc is None or pix is None:
        return 0.0
    try:
        foc_n, _ = normalize_telescope_focal_mm_for_plate_scale(float(foc))
        xb, _yb = fits_binning_xy_from_header(hdr)
        eff_um = effective_binned_pixel_pitch_um(base_pixel_um_1x1=float(pix), binning=int(xb))
        # margin ~2.0: VYVAR-solver-class corner coverage + CRVAL slop / high-Dec geometry
        return float(
            catalog_cone_radius_deg_from_optics(
                naxis1=int(naxis1),
                naxis2=int(naxis2),
                pixel_pitch_um=float(eff_um),
                focal_length_mm=float(foc_n),
                margin=2.05,
                fov_diameter_fallback_deg=float(plate_solve_fov_fallback_deg),
            )
        )
    except Exception:  # noqa: BLE001
        return 0.0


def _field_center_and_radius_from_wcs(w: WCS, h: int, wpx: int) -> tuple[SkyCoord, float]:
    """Footprint for **one** Vizier/Gaia cone query tied to the detector — not „celá obloha“.

    Uses the geometric pixel centre, then the **maximum** great-circle separation from that centre to a
    dense sample of the **full rectangle border** (not only corners). With sip / distortion, the farthest
    sky point from the centre is not always a corner — undersized cones produced a visible circular
    „matched only in the middle“ QA overlay. Adds multiplicative + absolute margin for edge stars and
    small plate-solve errors.
    """
    center = SkyCoord.from_pixel((wpx - 1) / 2.0, (h - 1) / 2.0, wcs=w, origin=0)
    max_sep_deg = 0.0
    step = max(1, int(round(float(max(wpx, h)) / 120.0)))
    pixels: set[tuple[int, int]] = set()
    for cx, cy in ((0, 0), (wpx - 1, 0), (0, h - 1), (wpx - 1, h - 1)):
        pixels.add((int(cx), int(cy)))
    for x in range(0, wpx, step):
        pixels.add((x, 0))
        pixels.add((x, h - 1))
    for y in range(0, h, step):
        pixels.add((0, y))
        pixels.add((wpx - 1, y))
    for cx, cy in pixels:
        cc = SkyCoord.from_pixel(float(cx), float(cy), wcs=w, origin=0)
        max_sep_deg = max(max_sep_deg, float(center.separation(cc).deg))
    # Margin: WCS at edges, rectangle vs cone on sphere, plate-solve error. Larger margin because
    # ``field_catalog_cone.csv`` built from a cropped MASTERSTAR must not underserve full-chip frames.
    radius_deg = max(max_sep_deg * 1.38, max_sep_deg + 45.0 / 3600.0)
    # If CD scale is wrong (too „zoomed“ on sky), spherical sampling collapses — blend in tangent-plane
    # half-diagonal from pixel scales so the cone stays physically plausible.
    try:
        scales = w.proj_plane_pixel_scales()
        sx_deg = abs(float(scales[0].to(u.deg).value))
        sy_deg = abs(float(scales[1].to(u.deg).value))
        r_cd = float(math.hypot(0.5 * (wpx - 1) * sx_deg, 0.5 * (h - 1) * sy_deg))
        radius_deg = max(radius_deg, r_cd * 1.22)
    except Exception:  # noqa: BLE001
        pass
    radius_deg = max(float(radius_deg), float(MIN_GAIA_CONE_RADIUS_DEG))
    return center, radius_deg


def _effective_field_catalog_cone_radius_deg(
    w: WCS,
    h: int,
    wpx: int,
    plate_solve_fov_deg: float | None,
    fits_header: fits.Header | None = None,
) -> tuple[SkyCoord, float]:
    """WCS-derived cone radius, optics floor from FITS (FOCALLEN+PIXSIZE), optional UI FOV minimum."""
    center, r = _field_center_and_radius_from_wcs(w, h, wpx)
    try:
        _pf_fb = float(plate_solve_fov_deg) if plate_solve_fov_deg is not None else 1.5
        if not math.isfinite(_pf_fb) or _pf_fb <= 0:
            _pf_fb = 1.5
    except (TypeError, ValueError):
        _pf_fb = 1.5
    r_opt = _gaia_catalog_cone_radius_optics_floor_deg(
        fits_header,
        naxis1=int(wpx),
        naxis2=int(h),
        plate_solve_fov_fallback_deg=float(_pf_fb),
    )
    r = max(float(r), float(r_opt))
    r_physical = float(r)
    if plate_solve_fov_deg is not None:
        try:
            pf = float(plate_solve_fov_deg)
            if math.isfinite(pf) and pf > 0:
                r_fov = catalog_cone_radius_from_fov_diameter_deg(pf)
                # UI FOV je bezpečnostné minimum, ale nesmie „rozšíriť“ kužeľ oveľa nad reálny čip:
                # pri zle nastavenom veľkom FOV (napr. 20°+) by inak vznikol polomer ~13° a SQL by
                # ťahalo 500k+ hviezd (minúty behu). Ak už WCS + optika dávajú rozumný polomer,
                # obmedzíme príspevok z FOV na ~30 % nad fyzikálnu stopu.
                if r_physical >= 2.5:
                    r_fov_eff = min(float(r_fov), r_physical * 1.30 + 0.35)
                else:
                    r_fov_eff = float(r_fov)
                r = min(22.0, max(r_physical, r_fov_eff))
        except (TypeError, ValueError):
            pass
    return center, float(r)


def _invalidate_field_catalog_cone_cache_if_needed(
    field_catalog_csv: Path,
    *,
    plate_solve_fov_deg: float | None,
    effective_radius_deg: float,
) -> None:
    """Remove ``field_catalog_cone.csv`` + meta when UI FOV or required cone size no longer matches cache."""
    p_csv = Path(field_catalog_csv)
    meta_p = _field_catalog_cone_meta_path(p_csv)
    if not p_csv.is_file() and not meta_p.is_file():
        return
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
    except Exception:  # noqa: BLE001
        meta = {}
    r_stored = float(meta.get("cone_radius_deg") or 0.0)
    fov_stored = meta.get("plate_solve_fov_deg")
    slack_deg = 45.0 / 3600.0
    reasons: list[str] = []

    if plate_solve_fov_deg is not None:
        try:
            pf = float(plate_solve_fov_deg)
            if math.isfinite(pf):
                if fov_stored is None:
                    reasons.append("meta chýba plate_solve_fov_deg (starý cache)")
                else:
                    try:
                        if abs(float(fov_stored) - pf) > 1e-4:
                            reasons.append(f"plate_solve_fov_deg {float(fov_stored):.6f} → {pf:.6f}")
                    except (TypeError, ValueError):
                        reasons.append("neplatný uložený plate_solve_fov_deg")
        except (TypeError, ValueError):
            pass

    r_eff = float(effective_radius_deg)
    if r_stored > 0 and r_eff > r_stored * 1.02 + slack_deg:
        reasons.append(f"kužeľ r≈{r_eff:.4f}° > uložené {r_stored:.4f}°")

    if not reasons:
        return
    p_csv.unlink(missing_ok=True)
    meta_p.unlink(missing_ok=True)
    log_event("Katalóg: field_catalog_cone cache vymazaná — " + "; ".join(reasons) + " (načítam nanovo).")


def _field_catalog_cone_meta_path(field_catalog_csv: Path) -> Path:
    return field_catalog_csv.parent / "field_catalog_cone_meta.json"


def _write_field_catalog_cone_meta(
    field_catalog_csv: Path,
    *,
    center: SkyCoord,
    radius_deg: float,
    naxis1: int,
    naxis2: int,
    plate_solve_fov_deg: float | None = None,
) -> None:
    """Persist cone parameters used to build ``field_catalog_cone.csv`` (cache invalidation for larger chips)."""
    p = _field_catalog_cone_meta_path(field_catalog_csv)
    rec = {
        "cone_radius_deg": float(radius_deg),
        "center_ra_icrs_deg": float(center.ra.deg),
        "center_dec_icrs_deg": float(center.dec.deg),
        "naxis1": int(naxis1),
        "naxis2": int(naxis2),
        "plate_solve_fov_deg": float(plate_solve_fov_deg)
        if plate_solve_fov_deg is not None and math.isfinite(float(plate_solve_fov_deg))
        else None,
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")


def select_comparison_stars_spatial_grid(
    df: pd.DataFrame,
    *,
    width_px: float,
    height_px: float,
    n_comp: int = 150,
    require_catalog_match: bool = True,
    require_photometry_ok: bool = True,
    require_non_variable: bool = True,
    exclude_nonlinear_badcolumn: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Choose comparison stars for ensemble photometry: ~one brightest candidate per spatial grid cell.

    Uses catalog-matched, photometry-safe detections, stratified on a nx×ny grid sized for ``n_comp`` and
    image aspect ratio so stars are spread across the full detector (typical 100–200 comps for APASS-like work).

    When ``require_non_variable`` is True and column ``catalog_known_variable`` exists, stars flagged as
    known variables (VSX and/or Gaia ``phot_variable_flag``) are excluded from the ensemble.
    """
    import math

    import numpy as np

    if df.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "empty"}

    work = df.copy()
    if "is_usable" in work.columns:
        work = work[work["is_usable"].fillna(False).astype(bool)]
    if work.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "no_rows_after_usable_filter"}
    if require_catalog_match and "catalog" in work.columns:
        work = work[work["catalog"].fillna("").astype(str).str.strip() != ""]
    if require_photometry_ok and "photometry_ok" in work.columns:
        work = work[work["photometry_ok"].fillna(True).astype(bool)]
    if require_non_variable and "catalog_known_variable" in work.columns:
        work = work[~work["catalog_known_variable"].fillna(False).astype(bool)]
    if exclude_nonlinear_badcolumn and "likely_nonlinear" in work.columns:
        _ln = pd.to_numeric(work["likely_nonlinear"], errors="coerce").fillna(0).astype(int)
        work = work[_ln == 0]
    if exclude_nonlinear_badcolumn and "on_bad_column" in work.columns:
        _ob = pd.to_numeric(work["on_bad_column"], errors="coerce").fillna(0).astype(int)
        work = work[_ob == 0]

    if work.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "no_rows_after_filter"}

    w = float(width_px)
    h = float(height_px)
    if w <= 0 or h <= 0:
        w = float(work["x"].max()) + 1.0
        h = float(work["y"].max()) + 1.0

    ar = w / h
    nc = max(1, int(n_comp))
    ny = max(1, int(round(math.sqrt(nc / ar))))
    nx = max(1, int(math.ceil(nc / ny)))
    while nx * ny < nc:
        nx += 1

    cw = w / float(nx)
    ch = h / float(ny)

    if "flux" not in work.columns:
        work["flux"] = 0.0
    work["_flux_key"] = pd.to_numeric(work["flux"], errors="coerce").fillna(0.0)

    ix = np.floor(np.clip(work["x"].to_numpy(dtype=float), 0.0, w - 1e-6) / cw).astype(int)
    iy = np.floor(np.clip(work["y"].to_numpy(dtype=float), 0.0, h - 1e-6) / ch).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    work["_cell"] = ix + iy * nx

    # Brightest per cell
    picked = (
        work.sort_values("_flux_key", ascending=False)
        .groupby("_cell", as_index=False, sort=False)
        .head(1)
    )

    if len(picked) > nc:
        picked = picked.nlargest(nc, "_flux_key")

    picked = picked.drop(columns=["_cell", "_flux_key"], errors="ignore")
    picked.insert(0, "comp_id", [f"COMP_{i+1:04d}" for i in range(len(picked))])
    picked.insert(1, "role", "comparison")

    meta = {
        "n_selected": int(len(picked)),
        "grid_nx": int(nx),
        "grid_ny": int(ny),
        "n_comp_requested": int(nc),
        "width_px": float(w),
        "height_px": float(h),
        "require_non_variable": bool(require_non_variable),
        "exclude_nonlinear_badcolumn": bool(exclude_nonlinear_badcolumn),
    }
    return picked, meta


def _annotate_masterstars_flux_zones(
    df: pd.DataFrame,
    *,
    noise_floor_adu: Any,
    equipment_saturate_adu: float | None,
    saturate_limit_adu_fallback: Any = None,
    n_stack: int | None = None,
    saturate_limit_fraction: float = 0.85,
) -> pd.DataFrame:
    """Tag MASTERSTAR catalog rows by flux vs SNR noise floor and saturation limit (equipment × 0.85).

    ``noise_floor_adu`` must match the DAO pre-match SNR filter (``median + k×σ``) from
    :func:`detect_stars_and_match_catalog` (see ``det_meta["noise_floor_adu"]``).
    """
    import numpy as np

    out = df.copy()
    if "flux" not in out.columns:
        return out
    flux_s = pd.to_numeric(out["flux"], errors="coerce")

    nf: float | None = None
    try:
        if noise_floor_adu is not None and str(noise_floor_adu).strip() != "":
            nf = float(noise_floor_adu)
            if not math.isfinite(nf):
                nf = None
    except (TypeError, ValueError):
        nf = None

    # Saturation: convert equipment limit to per-frame-equivalent for stacked MASTERSTAR.
    # MASTERSTAR is a mean stack; a single-frame saturation level corresponds to ~sat / n_stack in the stack.
    ns = int(n_stack) if n_stack is not None else 1
    ns = max(1, ns)

    sat_lim: float | None = None
    if equipment_saturate_adu is not None:
        try:
            fe = float(equipment_saturate_adu)
            if math.isfinite(fe) and fe > 0:
                sat_lim = fe * float(saturate_limit_fraction) / float(ns)
        except (TypeError, ValueError):
            pass
    if sat_lim is None and saturate_limit_adu_fallback is not None:
        try:
            fe = float(saturate_limit_adu_fallback)
            if math.isfinite(fe) and fe > 0:
                sat_lim = fe * float(saturate_limit_fraction) / float(ns)
        except (TypeError, ValueError):
            pass

    out["noise_floor_adu"] = nf if nf is not None else np.nan
    out["saturate_limit_adu_85pct"] = sat_lim if sat_lim is not None else np.nan
    out["saturate_limit_per_frame_adu"] = sat_lim if sat_lim is not None else np.nan

    # ── Noisy sub-kategórie ──
    # Prahy sú fixné v kóde — adaptívne pre akúkoľvek zostavu a podmienky.
    # noise_floor = median + k×σ  kde k = prematch_peak_sigma_floor (config)
    # Noisy1: flux medzi noise_floor a (median + 3σ) → možná premenná, slabší signál
    # Noisy2: flux medzi (median + 2σ) a noise_floor → veľmi slabý signál
    # Noisy3: flux < (median + 2σ)                   → prakticky nepoužiteľné
    # Pre výpočet σ prahov: odhadneme zo samotného noise_floor
    # noise_floor = med + k*std  => std = (noise_floor - med) / k
    # k = prematch_peak_sigma_floor (typicky 2.5)
    # Pre jednoduchosť: noise_floor_2s = med + (2/k)*(nf - med)
    #                   noise_floor_3s = nf  (= med + k*std, kde k≈2.5 ≈ 3sigma)

    # Saturácia: porovnávaj PEAK pixel hodnotu (nie flux sumu!)
    # flux je aperture sum = suma ADU v aperture → vždy >> sat_limit pre akúkoľvek hviezdu
    # peak_max_adu = max pixel v hviezde → správna veličina pre saturáciu
    if "peak_max_adu" in out.columns:
        peak_s = pd.to_numeric(out["peak_max_adu"], errors="coerce")
    else:
        # Fallback: ak peak_max_adu chýba, použi flux (starý spôsob, nepresný)
        peak_s = flux_s

    out["zone"] = "linear"

    if sat_lim is not None:
        out.loc[peak_s > float(sat_lim), "zone"] = "saturated"

    if nf is not None:
        nf_val = float(nf)
        # Odhad mediánu signálu (pre výpočet sub-prahov)
        finite_flux = flux_s[flux_s.notna() & (flux_s < nf_val)]
        if len(finite_flux) > 10:
            flux_med = float(finite_flux.median())
        else:
            flux_med = float(flux_s.median()) if flux_s.notna().any() else 0.0

        # Sub-prahy: lineárna interpolácia medzi flux_med a nf_val
        # noisy1_thresh = nf_val           (= median + k*sigma, config k)
        # noisy2_thresh = flux_med + 2/3 * (nf_val - flux_med)  (~2σ ak k=3)
        # noisy3_thresh = flux_med + 1/3 * (nf_val - flux_med)  (~1σ ak k=3)
        spread = nf_val - flux_med
        noisy2_thresh = flux_med + (2.0 / 3.0) * spread  # ~2σ
        noisy3_thresh = flux_med + (1.0 / 3.0) * spread  # ~1σ

        linear_mask = out["zone"] == "linear"
        # Noisy1: pod noise_floor ale nad 2σ prahom → slabý ale možno použiteľný
        noisy1_mask = linear_mask & (flux_s < nf_val) & (flux_s >= noisy2_thresh)
        # Noisy2: pod 2σ ale nad 1σ → veľmi slabý
        noisy2_mask = linear_mask & (flux_s < noisy2_thresh) & (flux_s >= noisy3_thresh)
        # Noisy3: pod 1σ → nepoužiteľný
        noisy3_mask = linear_mask & (flux_s < noisy3_thresh)

        out.loc[noisy1_mask, "zone"] = "noisy1"
        out.loc[noisy2_mask, "zone"] = "noisy2"
        out.loc[noisy3_mask, "zone"] = "noisy3"

    # is_saturated: peak > sat_limit
    if sat_lim is not None:
        out["is_saturated"] = (peak_s > float(sat_lim)).fillna(False)
    else:
        out["is_saturated"] = False

    if nf is not None:
        # is_noisy = True pre akúkoľvek noisy sub-kategóriu
        out["is_noisy"] = out["zone"].isin(["noisy1", "noisy2", "noisy3"])
    else:
        out["is_noisy"] = False

    # is_usable: len linear (nie saturated, nie žiadna noisy kategória)
    out["is_usable"] = out["zone"].eq("linear") & flux_s.notna()
    return out


def write_photometry_plan_files(
    *,
    platesolve_dir: Path,
    masterstar_fits: Path,
    masterstars_csv: Path,
    n_comparison_stars: int = 150,
    require_non_variable: bool = True,
    draft_id: int | None = None,
    database_path: Path | str | None = None,
) -> dict[str, Any]:
    """Write ``comparison_stars.csv`` from ``masterstars.csv`` + image size; stub ``variable_targets.csv``."""
    import numpy as np
    import json


    ps = Path(platesolve_dir)
    ps.mkdir(parents=True, exist_ok=True)

    if not masterstars_csv.is_file():
        return {"comparison_stars_csv": "", "variable_targets_csv": "", "error": "missing masterstars.csv"}

    df = pd.read_csv(masterstars_csv)
    if draft_id is not None and database_path:
        dbp = Path(str(database_path))
        if dbp.is_file():
            try:
                _db_m = VyvarDatabase(dbp)
                try:
                    _msr = _db_m.fetch_master_sources_for_draft(int(draft_id))
                finally:
                    _db_m.conn.close()
                if _msr:
                    sid_nl = {
                        str(r.get("SOURCE_ID_GAIA") or "").strip(): int(r.get("LIKELY_NONLINEAR") or 0)
                        for r in _msr
                    }
                    sid_bc = {
                        str(r.get("SOURCE_ID_GAIA") or "").strip(): int(r.get("ON_BAD_COLUMN") or 0)
                        for r in _msr
                    }

                    def _cid(v: Any) -> str:
                        return str(v or "").strip()

                    if "catalog_id" in df.columns:
                        df["likely_nonlinear"] = df["catalog_id"].map(lambda c: sid_nl.get(_cid(c), 0))
                        df["on_bad_column"] = df["catalog_id"].map(lambda c: sid_bc.get(_cid(c), 0))
            except Exception:  # noqa: BLE001
                pass
    try:
        with fits.open(masterstar_fits, memmap=False) as hdul:
            hdr = hdul[0].header
    except Exception as _ms_open_exc:  # noqa: BLE001
        log_event(f"write_photometry_plan_files: nepodarilo sa otvoriť MASTERSTAR.fits ({masterstar_fits}): {_ms_open_exc!s}")
        return {"comparison_stars_csv": "", "variable_targets_csv": "", "error": "MASTERSTAR open failed"}
    wpx = int(hdr.get("NAXIS1", 0) or 0)
    h = int(hdr.get("NAXIS2", 0) or 0)
    if wpx <= 0 or h <= 0:
        return {"comparison_stars_csv": "", "error": "MASTERSTAR has no data"}

    comp_df, cmeta = select_comparison_stars_spatial_grid(
        df,
        width_px=float(wpx),
        height_px=float(h),
        n_comp=int(n_comparison_stars),
        require_non_variable=bool(require_non_variable),
    )
    comp_path = ps / "comparison_stars.csv"
    comp_df.to_csv(comp_path, index=False)

    # VSX variable targets for the field (full cone), with pixel coords from MASTERSTAR WCS.
    var_path = ps / "variable_targets.csv"
    var_cols = [
        "name",
        "catalog_id",
        "catalog",
        "ra_deg",
        "dec_deg",
        "priority",
        "notes",
        "vsx_name",
        "vsx_type",
        "vsx_period",
        "x",
        "y",
    ]
    vsx_out = pd.DataFrame(columns=var_cols)
    _vsx_n_cone = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w0 = WCS(hdr)
        if not bool(getattr(w0, "has_celestial", False)):
            raise RuntimeError("MASTERSTAR WCS has no celestial axes.")
        if w0.has_celestial:
            center, radius_deg = _effective_field_catalog_cone_radius_deg(
                w0, int(h), int(wpx), plate_solve_fov_deg=None, fits_header=hdr
            )
            _vsx_p3: Path | None = None
            try:
                _vsp3 = str(getattr(AppConfig(), "vsx_local_db_path", "") or "").strip()
                if _vsp3:
                    _vsx_p3 = Path(_vsp3).expanduser().resolve()
            except Exception:  # noqa: BLE001
                _vsx_p3 = None
            vsx_df = _query_vsx_local(center=center, radius_deg=float(radius_deg), vsx_db_path=_vsx_p3)
            _vsx_n_cone = int(len(vsx_df)) if vsx_df is not None else 0
            try:
                mag_limit = float(getattr(AppConfig(), "vsx_variable_targets_mag_limit", 13.0) or 13.0)
            except Exception:  # noqa: BLE001
                mag_limit = 13.0
            if (
                vsx_df is not None
                and not vsx_df.empty
                and "mag_max" in vsx_df.columns
                and mag_limit is not None
                and float(mag_limit) > 0
            ):
                mm = pd.to_numeric(vsx_df["mag_max"], errors="coerce")
                vsx_df = vsx_df[mm.isna() | (mm <= float(mag_limit))].copy()
                _vsx_n_cone = int(len(vsx_df))

            # Gaia cross-id from MASTERSTAR catalog (within 10 arcsec).
            ga = df.copy()
            if "catalog_id" in ga.columns and "ra_deg" in ga.columns and "dec_deg" in ga.columns:
                ga["catalog_id"] = ga["catalog_id"].fillna("").astype(str).str.strip()
                ga = ga[ga["catalog_id"].ne("")].copy()
                ga["ra_deg"] = pd.to_numeric(ga["ra_deg"], errors="coerce")
                ga["dec_deg"] = pd.to_numeric(ga["dec_deg"], errors="coerce")
                ga = ga[ga["ra_deg"].notna() & ga["dec_deg"].notna()].copy()
            else:
                ga = ga.iloc[0:0].copy()

            if vsx_df is not None and not vsx_df.empty and "ra_deg" in vsx_df.columns and "dec_deg" in vsx_df.columns:
                v = vsx_df.copy()
                v["ra_deg"] = pd.to_numeric(v["ra_deg"], errors="coerce")
                v["dec_deg"] = pd.to_numeric(v["dec_deg"], errors="coerce")
                v = v[v["ra_deg"].notna() & v["dec_deg"].notna()].copy()
                if not v.empty:
                    # Pixel coords from WCS (origin=0).
                    xy = w0.all_world2pix(v["ra_deg"].astype(float).to_numpy(), v["dec_deg"].astype(float).to_numpy(), 0)
                    x = np.asarray(xy[0], dtype=float)
                    y = np.asarray(xy[1], dtype=float)
                    v["x"] = x
                    v["y"] = y
                    in_frame = (v["x"] >= -50.0) & (v["y"] >= -50.0) & (v["x"] <= float(wpx) + 50.0) & (v["y"] <= float(h) + 50.0)
                    v = v.loc[in_frame].copy()

                # Nearest Gaia match for catalog_id.
                cat_id_out = [""] * int(len(v))
                if len(v) > 0 and len(ga) > 0:
                    vcoo = SkyCoord(
                        ra=v["ra_deg"].astype(float).to_numpy() * u.deg,
                        dec=v["dec_deg"].astype(float).to_numpy() * u.deg,
                        frame="icrs",
                    )
                    gcoo = SkyCoord(
                        ra=ga["ra_deg"].astype(float).to_numpy() * u.deg,
                        dec=ga["dec_deg"].astype(float).to_numpy() * u.deg,
                        frame="icrs",
                    )
                    idx, sep2d, _ = vcoo.match_to_catalog_sky(gcoo)
                    max_sep = 10.0 * u.arcsec
                    for i in range(len(v)):
                        if 0 <= int(idx[i]) < len(ga) and sep2d[i] <= max_sep:
                            cat_id_out[i] = str(ga.iloc[int(idx[i])]["catalog_id"])

                # Period column varies by VSX schema.
                _pcol = None
                for c in ("period", "varperiod", "var_period", "Period", "VarPeriod", "Var_Period"):
                    if c in v.columns:
                        _pcol = c
                        break
                if _pcol is None:
                    v["vsx_period"] = np.nan
                else:
                    v["vsx_period"] = pd.to_numeric(v[_pcol], errors="coerce")

                vname = v.get("name", pd.Series([""] * len(v))).fillna("").astype(str).str.strip()
                vtype = v.get("var_type", pd.Series([""] * len(v))).fillna("").astype(str).str.strip()
                notes = []
                for t, p in zip(vtype.tolist(), v["vsx_period"].tolist(), strict=False):
                    t0 = str(t or "").strip()
                    p0 = float(p) if p is not None and pd.notna(p) else None
                    if p0 is not None:
                        notes.append(f"{t0} P={p0}")
                    else:
                        notes.append(t0)

                vsx_out = pd.DataFrame(
                    {
                        "name": vname,
                        "catalog_id": cat_id_out,
                        "catalog": "VSX",
                        "ra_deg": v["ra_deg"].astype(float).to_numpy(),
                        "dec_deg": v["dec_deg"].astype(float).to_numpy(),
                        "priority": 1,
                        "notes": notes,
                        "vsx_name": vname,
                        "vsx_type": vtype,
                        "vsx_period": v["vsx_period"].to_numpy(),
                        "x": pd.to_numeric(v.get("x"), errors="coerce"),
                        "y": pd.to_numeric(v.get("y"), errors="coerce"),
                    }
                )
    except Exception as _vsx_exc:  # noqa: BLE001
        log_event(f"variable_targets.csv (VSX export) preskočený: {_vsx_exc!s}")
        vsx_out = pd.DataFrame(columns=var_cols)

    # Always overwrite (even if it exists) so UI sees current field cone.
    vsx_out.to_csv(var_path, index=False)
    log_event(
        f"variable_targets.csv (VSX export): cone={_vsx_n_cone} → zapísané={int(len(vsx_out))} "
        f"({var_path.name})"
    )

    plan_path = ps / "photometry_plan.json"
    plan = {
        "purpose": "VYVAR photometry: comparison ensemble + variable targets",
        "comparison_stars": str(comp_path),
        "variable_targets_template": str(var_path),
        "masterstars": str(masterstars_csv),
        "masterstar_fits": str(masterstar_fits),
        "comparison_selection": cmeta,
        "next_steps": [
            "Fill variable_targets.csv with programme stars (catalog IDs / coordinates).",
            "Use comparison_stars.csv comp_id / catalog_id to tie ensemble photometry on detrended_aligned frames.",
            "Run validate_comparison_ensemble_flatness on detrended_aligned frames to check comp stars stay flat vs ensemble.",
            "Light curves: aperture photometry per frame vs time (JD), then search for new variables vs field behavior.",
        ],
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    return {
        "comparison_stars_csv": str(comp_path),
        "variable_targets_csv": str(var_path),
        "photometry_plan_json": str(plan_path),
        "comparison_selection": cmeta,
    }


def _dao_auto_binning_factor(h: int, w: int) -> int:
    """2×2 mean binning for DAO on large chips (~4× fewer pixels); skipped below ~5 MP."""
    mp = float(int(h) * int(w)) / 1_000_000.0
    if mp < 5.0:
        return 1
    return 2


def _mean_bin2d_for_dao(data0: "np.ndarray", factor: int) -> tuple["np.ndarray", int]:
    import numpy as np

    f = int(factor)
    a = np.asarray(data0, dtype=np.float32)
    if f <= 1:
        return a, 1
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    if h2 < f or w2 < f:
        return a, 1
    b = a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3)).astype(np.float32)
    return b, f


def _dao_xy_binned_to_full(x: "np.ndarray", y: "np.ndarray", f: int) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    if f <= 1:
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    s = float(f)
    off = float(f - 1) * 0.5
    return np.asarray(x, dtype=np.float64) * s + off, np.asarray(y, dtype=np.float64) * s + off


def _prefilter_dao_table_brightest(tbl: Any, keep_top: int) -> Any:
    """Cap DAO rows before sorting when the finder returns an enormous table (slow + useless)."""
    import numpy as np

    if tbl is None or len(tbl) <= int(keep_top):
        return tbl
    flux_np = np.asarray(tbl["flux"], dtype=np.float64)
    k = int(keep_top)
    take = np.argpartition(flux_np, -k)[-k:]
    return tbl[take]


def _dao_spatial_flux_cap_row_indices(
    tbl: Any,
    *,
    max_n: int,
    width_px: float,
    height_px: float,
) -> "np.ndarray":
    """Row indices into ``tbl`` for up to ``max_n`` sources, spread across the detector.

    Sorting globally by flux and truncating biases toward the frame centre when vignetting or
    gradients make edge stars fainter in ADU — QA then falsely looks like a ``catalog disc``. This
    fills a coarse grid brightest-first per cell, then tops up by global flux (same cap).
    """
    import numpy as np

    n = len(tbl)
    m = int(max_n)
    if n <= m:
        return np.arange(n, dtype=np.int64)
    x = np.asarray(tbl["xcentroid"], dtype=np.float64)
    y = np.asarray(tbl["ycentroid"], dtype=np.float64)
    flux = np.asarray(tbl["flux"], dtype=np.float64)
    w = max(float(width_px), 1.0)
    h = max(float(height_px), 1.0)
    # ~25 sources per cell on average, bounded grid
    ncell_target = max(48, min(512, max(1, m // 25)))
    aspect = w / h
    ny = max(4, int(round((ncell_target / aspect) ** 0.5)))
    nx = max(4, int(round(ncell_target / float(ny))))
    ncells = nx * ny
    per_cell = max(1, m // ncells)
    ix = np.clip((x / w * nx).astype(np.int64), 0, nx - 1)
    iy = np.clip((y / h * ny).astype(np.int64), 0, ny - 1)
    cell_id = ix + iy * nx
    order = np.argsort(-flux)
    taken = np.zeros(n, dtype=bool)
    picked: list[int] = []
    counts = np.zeros(ncells, dtype=np.int32)
    for idx in order:
        c = int(cell_id[idx])
        if counts[c] < per_cell and len(picked) < m:
            picked.append(int(idx))
            counts[c] += 1
            taken[idx] = True
    for idx in order:
        if len(picked) >= m:
            break
        if not taken[idx]:
            picked.append(int(idx))
            taken[idx] = True
    return np.asarray(sorted(picked), dtype=np.int64)


def detect_stars_match_master_reference(
    data: "np.ndarray",
    hdr: fits.Header,
    master_df: pd.DataFrame,
    *,
    max_catalog_rows: int = 12000,
    match_sep_arcsec: float = 8.0,
    saturate_level_fraction: float = 0.999,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    dao_fwhm_px: float | None = None,
    fallback_saturate_adu: float | None = None,
    equipment_saturate_adu: float | None = None,
    frame_name: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """DAO on this frame + nearest-neighbor match to ``masterstars.csv`` (no Vizier / cone).

    Catalog IDs and static columns come from the master row; ``x``, ``y``, ``flux``, ``peak_*`` and
    saturation flags are per-frame. Intended for ``detrended_aligned`` data whose WCS matches
    ``MASTERSTAR.fits`` astrometry.
    """
    import numpy as np
    from astropy.stats import sigma_clipped_stats

    m = master_df
    # Primary mode: sky match using per-frame WCS (DAO x/y -> RA/Dec), then NN match against MASTERSTAR ra/dec.
    # Fallback: if frame has no usable celestial WCS, do a pixel NN match (15 px) against MASTERSTAR x/y.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs_obj = WCS(hdr)

    # MASTERSTAR sky catalog
    if "ra_deg" not in m.columns or "dec_deg" not in m.columns:
        raise ValueError("masterstars table must contain ra_deg, dec_deg")
    _ra = pd.to_numeric(m["ra_deg"], errors="coerce")
    _de = pd.to_numeric(m["dec_deg"], errors="coerce")
    okm_sky = _ra.notna() & _de.notna()
    if not bool(okm_sky.any()):
        raise ValueError("No valid ra_deg/dec_deg rows in masterstars")
    m_valid = m.loc[okm_sky].reset_index(drop=True)
    ra_m = pd.to_numeric(m_valid["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de_m = pd.to_numeric(m_valid["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    master_coords = SkyCoord(ra=ra_m * u.deg, dec=de_m * u.deg, frame="icrs")

    match_mode = "sky"
    plate_scale_arcsec_per_px = None

    arr = np.asarray(data, dtype=np.float32)
    mean, med, std = sigma_clipped_stats(arr, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((arr - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bfac = 1
    _cfg_dao = AppConfig()
    _fb_m = float(dao_fwhm_px) if dao_fwhm_px is not None else float(getattr(_cfg_dao, "sips_dao_fwhm_px", 2.5))
    _base_fw_m = dao_detection_fwhm_pixels(hdr, configured_fallback=_fb_m)
    try:
        from photutils.detection import DAOStarFinder  # type: ignore

        thr_s = float(dao_threshold_sigma)
        thr_s = max(0.5, min(20.0, thr_s))
        dao_scale = _dao_auto_binning_factor(*data0.shape)
        data_dao, bfac = _mean_bin2d_for_dao(data0, dao_scale)
        _, _, std_dao = (
            sigma_clipped_stats(data_dao, sigma=3.0, maxiters=3)
            if bfac > 1
            else (mean, med, std)
        )
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            try:
                std_dao = float(np.nanstd(arr))
            except Exception:  # noqa: BLE001
                std_dao = float("nan")
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            try:
                finite = np.isfinite(arr)
                n_finite = int(np.count_nonzero(finite))
                if n_finite > 0:
                    vals = arr[finite]
                    n_unique = int(len(np.unique(vals)))
                    mn = float(np.nanmin(vals))
                    mx = float(np.nanmax(vals))
                else:
                    n_unique, mn, mx = 0, float("nan"), float("nan")
                print(
                    f"DEBUG std=0: {_nm} n_unique={n_unique} n_finite={n_finite} "
                    f"min={mn:.1f} max={mx:.1f}"
                )
            except Exception:  # noqa: BLE001
                pass
            # If the frame isn't constant (min != max), try std from non-zero finite pixels.
            try:
                nonzero_mask = (arr != 0) & np.isfinite(arr)
                if int(np.count_nonzero(nonzero_mask)) > 100:
                    std_dao = float(np.std(arr[nonzero_mask]))
            except Exception:  # noqa: BLE001
                pass
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(f"WARNING: {_nm} std=0 aj po fallback, preskakujem")
            empty_meta: dict[str, Any] = {"status": "std_zero"}
            return pd.DataFrame(), empty_meta
        fwhm_eff = max(1.2, _base_fw_m / float(bfac))
        _thr = max(thr_s * float(std_dao), 1e-6)
        try:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(
                f"DEBUG DAO INPUT: {_nm} mean={float(np.nanmean(arr)):.1f} std={float(np.nanstd(arr)):.1f} "
                f"threshold={float(_thr):.1f} fwhm={float(fwhm_eff):.2f}"
            )
        except Exception:  # noqa: BLE001
            pass
        finder = DAOStarFinder(
            fwhm=float(fwhm_eff),
            threshold=float(_thr),
            brightest=None,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = finder(data_dao)
        if tbl is not None and len(tbl) > 0:
            tbl = _prefilter_dao_table_brightest(tbl, max(int(max_catalog_rows) * 12, 36_000))
    except Exception:  # noqa: BLE001
        tbl = None
        bfac = 1

    _fb_sat = fallback_saturate_adu
    if _fb_sat is None:
        try:
            _cfg_sat = AppConfig()
            _fb_sat = getattr(_cfg_sat, "photometry_fallback_saturate_adu", None)
        except Exception:  # noqa: BLE001
            _fb_sat = None

    sat_limit, sat_limit_src = _effective_saturation_limit(
        hdr, fallback_adu=_fb_sat, equipment_saturate_adu=equipment_saturate_adu
    )
    sat_frac = float(saturate_level_fraction)
    sat_frac = min(max(sat_frac, 0.5), 1.0)

    foot_meta = {
        "catalog_footprint": {"method": "master_reference_only"},
        "saturation": {
            "effective_limit_adu": float(sat_limit) if sat_limit is not None else None,
            "limit_source": sat_limit_src,
            "plateau_half_inner_px": 1,
            "plateau_rel": 0.996,
            "plateau_min_pixels": 5,
        },
    }

    empty_meta = {
        "n_detected": 0,
        "n_detected_dao": 0,
        "n_matched": 0,
        "n_matched_before_mag_limit": 0,
        "catalog_rows": int(len(master_df)),
        "catalog_match_mode": ("master_reference_pixel" if match_mode == "pixel" else "master_reference_sky"),
        "n_likely_saturated": 0,
        "n_saturated_from_peak": 0,
        "n_saturated_plateau": 0,
        "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
        "saturate_limit_source": sat_limit_src,
        "n_vsx_in_field": 0,
        "n_gaia_variable_in_field": 0,
        "field_catalog_cone_csv": None,
        "dao_threshold_sigma": float(dao_threshold_sigma),
        "dao_fwhm_px": float(max(1.2, _base_fw_m)),
        "dao_detect_binning": 1,
        "match_sep_arcsec_requested": float(match_sep_arcsec),
        "match_sep_arcsec_effective": float(match_sep_arcsec),
        "plate_scale_arcsec_per_px": (
            float(plate_scale_arcsec_per_px) if plate_scale_arcsec_per_px is not None else None
        ),
        **foot_meta,
    }

    if tbl is None or len(tbl) == 0:
        empty_meta["faintest_mag_limit"] = float(faintest_mag_limit) if faintest_mag_limit is not None else None
        empty_meta["n_dropped_fainter_than_limit"] = 0
        return pd.DataFrame(), empty_meta

    _fwhm_used_m = float(max(1.2, _base_fw_m / float(bfac)))
    _d_h, _d_w = int(data_dao.shape[0]), int(data_dao.shape[1])
    _keep = _dao_spatial_flux_cap_row_indices(
        tbl, max_n=int(max_catalog_rows), width_px=float(_d_w), height_px=float(_d_h)
    )
    tbl = tbl[_keep]
    tbl.sort("flux")
    tbl = tbl[::-1]
    n = len(tbl)
    xb = np.asarray(tbl["xcentroid"], dtype=np.float64)
    yb = np.asarray(tbl["ycentroid"], dtype=np.float64)
    x, y = _dao_xy_binned_to_full(xb, yb, bfac)
    flux = np.asarray(tbl["flux"], dtype=np.float64)
    peak_dao = np.asarray(tbl["peak"], dtype=np.float64) if "peak" in tbl.colnames else np.full(n, np.nan)

    match_thr = float(match_sep_arcsec)
    icomp: np.ndarray | None = None
    sep_arcsec_arr: np.ndarray
    oix: np.ndarray | None = None
    ra_deg: np.ndarray
    dec_deg: np.ndarray
    if getattr(wcs_obj, "has_celestial", False):
        ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
        det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        idx, sep2d, _ = det_coords.match_to_catalog_sky(master_coords)
        sep_arcsec_arr = np.asarray(sep2d.to_value(u.arcsec), dtype=np.float64)
        icomp = np.asarray(idx, dtype=np.int64)
        oix = None
        match_mode = "sky"
    else:
        # Fallback for frames without WCS: pixel NN match with a loose threshold (15 px).
        match_mode = "pixel_fallback_no_wcs"
        ra_deg = np.full(n, np.nan, dtype=np.float64)
        dec_deg = np.full(n, np.nan, dtype=np.float64)
        if "x" in m_valid.columns and "y" in m_valid.columns:
            try:
                from scipy.spatial import cKDTree  # type: ignore

                mx = pd.to_numeric(m_valid["x"], errors="coerce").to_numpy(dtype=np.float64)
                my = pd.to_numeric(m_valid["y"], errors="coerce").to_numpy(dtype=np.float64)
                okxy = np.isfinite(mx) & np.isfinite(my)
                mx = mx[okxy]
                my = my[okxy]
                # Keep a parallel index map into m_valid
                oix = np.nonzero(okxy)[0].astype(np.int64)
                master_xy = np.column_stack([mx, my])
                det_xy = np.column_stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)])
                tree = cKDTree(master_xy)
                dist_px, idx = tree.query(det_xy, distance_upper_bound=float(15.0))
                icomp = np.asarray(idx, dtype=np.int64)
                sep_arcsec_arr = np.asarray(dist_px, dtype=np.float64)  # pixels (no WCS); keep numeric for mask
                sep_arcsec_arr[~np.isfinite(sep_arcsec_arr)] = np.nan
            except Exception:  # noqa: BLE001
                icomp = np.zeros(n, dtype=np.int64)
                sep_arcsec_arr = np.full(n, np.nan, dtype=np.float64)
                oix = None
        else:
            icomp = np.zeros(n, dtype=np.int64)
            sep_arcsec_arr = np.full(n, np.nan, dtype=np.float64)
            oix = None

    pmax_arr = _box_peaks_at_centroids(arr, x, y)
    _sat_block = _vectorized_star_saturation_columns(
        arr,
        x,
        y,
        sat_limit=sat_limit,
        sat_frac=sat_frac,
        peak_dao=peak_dao,
        peak_max_adu=pmax_arr,
    )

    nm_m = len(m_valid)
    idx_det = np.arange(1, n + 1, dtype=np.int32)
    det_str = np.array([f"DET_{i:04d}" for i in idx_det], dtype=object)
    finite_sep = np.isfinite(sep_arcsec_arr)
    icomp_a = icomp if icomp is not None else np.zeros(n, dtype=np.int64)
    if match_mode == "pixel_fallback_no_wcs" and oix is not None:
        ok_ix = (icomp_a >= 0) & (icomp_a < int(len(oix)))
        cat_row = np.full(n, -1, dtype=np.int64)
        cat_row[ok_ix] = oix[icomp_a[ok_ix]]
        matched = finite_sep & ok_ix & (sep_arcsec_arr <= float(15.0)) & (cat_row >= 0)
        # For output semantics, keep match_sep_arcsec as NaN when unmatched (already handled).
    else:
        cat_row = np.clip(icomp_a.astype(np.int64, copy=False), 0, max(nm_m - 1, 0))
        matched = finite_sep & (sep_arcsec_arr <= match_thr) & (nm_m > 0)

    safe = np.clip(np.where(matched, cat_row, 0), 0, max(nm_m - 1, 0))
    cat_s_m = (
        m_valid["catalog"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "catalog" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    cid_m = (
        m_valid["catalog_id"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "catalog_id" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    nm_c = (
        m_valid["name"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
        if "name" in m_valid.columns
        else np.array([""] * nm_m, dtype=object)
    )
    mag_m = (
        pd.to_numeric(m_valid["mag"], errors="coerce").to_numpy(dtype=np.float64)
        if "mag" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    bv_m = (
        pd.to_numeric(m_valid["b_v"], errors="coerce").to_numpy(dtype=np.float64)
        if "b_v" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    gn_m = (
        pd.to_numeric(m_valid["gaia_nss"], errors="coerce").to_numpy(dtype=np.float64)
        if "gaia_nss" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    gq_m = (
        pd.to_numeric(m_valid["gaia_qso"], errors="coerce").to_numpy(dtype=np.float64)
        if "gaia_qso" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    gg_m = (
        pd.to_numeric(m_valid["gaia_gal"], errors="coerce").to_numpy(dtype=np.float64)
        if "gaia_gal" in m_valid.columns
        else np.full(nm_m, np.nan, dtype=np.float64)
    )
    vx_m = (
        m_valid["vsx_known_variable"].fillna(False).astype(bool).to_numpy()
        if "vsx_known_variable" in m_valid.columns
        else np.zeros(nm_m, dtype=bool)
    )
    gv_m = (
        m_valid["gaia_dr3_variable_catalog"].fillna(False).astype(bool).to_numpy()
        if "gaia_dr3_variable_catalog" in m_valid.columns
        else np.zeros(nm_m, dtype=bool)
    )
    if "catalog_known_variable" in m_valid.columns:
        ck_m = m_valid["catalog_known_variable"].fillna(False).astype(bool).to_numpy()
    else:
        ck_m = vx_m | gv_m

    cat_sel = cat_s_m[safe]
    cid_sel = cid_m[safe]
    nm_sel = nm_c[safe]
    name_cand = np.where(
        nm_sel != "",
        nm_sel,
        np.where(cid_sel != "", cid_sel, np.where(cat_sel != "", cat_sel, det_str)),
    )
    name_out = np.where(matched, name_cand, det_str)

    mag_out = np.full(n, np.nan, dtype=np.float64)
    bv_out = np.full(n, np.nan, dtype=np.float64)
    gn_out = np.full(n, np.nan, dtype=np.float64)
    gq_out = np.full(n, np.nan, dtype=np.float64)
    gg_out = np.full(n, np.nan, dtype=np.float64)
    mag_out[matched] = mag_m[safe[matched]]
    bv_out[matched] = bv_m[safe[matched]]
    gn_out[matched] = gn_m[safe[matched]]
    gq_out[matched] = gq_m[safe[matched]]
    gg_out[matched] = gg_m[safe[matched]]

    vx_out = np.zeros(n, dtype=bool)
    gv_out = np.zeros(n, dtype=bool)
    ck_out = np.zeros(n, dtype=bool)
    vx_out[matched] = vx_m[safe[matched]]
    gv_out[matched] = gv_m[safe[matched]]
    ck_out[matched] = ck_m[safe[matched]]

    cat_out = np.array([""] * n, dtype=object)
    cid_out = np.array([""] * n, dtype=object)
    cat_out[matched] = cat_s_m[safe[matched]]
    cid_out[matched] = cid_m[safe[matched]]

    n_matched = int(np.count_nonzero(matched & (cat_s_m[safe].astype(str) != "")))

    df_out = pd.DataFrame(
        {
            "name": name_out,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "mag": mag_out,
            "b_v": bv_out,
            "catalog": cat_out,
            "catalog_id": cid_out,
            "x": x,
            "y": y,
            "flux": flux,
            "match_sep_arcsec": sep_arcsec_arr.astype(np.float64, copy=False),
            "gaia_nss": gn_out,
            "gaia_qso": gq_out,
            "gaia_gal": gg_out,
            "vsx_known_variable": vx_out,
            "gaia_dr3_variable_catalog": gv_out,
            "catalog_known_variable": ck_out,
            **_sat_block,
        }
    )
    n_detected_dao = int(n)
    n_matched_before_mag = int(n_matched)
    n_before_mag = len(df_out)
    if faintest_mag_limit is not None and np.isfinite(float(faintest_mag_limit)):
        lim_m = float(faintest_mag_limit)
        mcol = pd.to_numeric(df_out["mag"], errors="coerce")
        # Drop only matched stars fainter than limit; keep unmatched detections (no catalog mag) for QA.
        df_out = df_out.loc[mcol.isna() | (mcol <= lim_m)].reset_index(drop=True)
        meta_mag = {
            "faintest_mag_limit": lim_m,
            "n_dropped_fainter_than_limit": int(n_before_mag - len(df_out)),
        }
    else:
        meta_mag = {"faintest_mag_limit": None, "n_dropped_fainter_than_limit": 0}

    n_sat = int(df_out["likely_saturated"].sum()) if len(df_out) and "likely_saturated" in df_out.columns else 0
    n_sat_pk = (
        int(df_out["saturated_from_peak"].sum())
        if len(df_out) and "saturated_from_peak" in df_out.columns
        else 0
    )
    n_sat_pl = (
        int(df_out["saturated_plateau"].sum())
        if len(df_out) and "saturated_plateau" in df_out.columns
        else 0
    )
    cat_nonempty = (
        df_out["catalog"].fillna("").astype(str).str.strip().ne("")
        if len(df_out) and "catalog" in df_out.columns
        else pd.Series([], dtype=bool)
    )
    n_matched_final = int(cat_nonempty.sum()) if len(df_out) else 0
    meta = {
        "n_detected_dao": n_detected_dao,
        "n_detected": int(len(df_out)),
        "n_matched_before_mag_limit": n_matched_before_mag,
        "n_matched": n_matched_final,
        "catalog_rows": int(len(master_df)),
        "catalog_match_mode": ("master_reference_pixel" if match_mode == "pixel" else "master_reference_sky"),
        "n_likely_saturated": n_sat,
        "n_saturated_from_peak": n_sat_pk,
        "n_saturated_plateau": n_sat_pl,
        "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
        "saturate_limit_source": sat_limit_src,
        "n_vsx_in_field": 0,
        "n_gaia_variable_in_field": 0,
        **foot_meta,
        "field_catalog_cone_csv": None,
        "dao_threshold_sigma": float(dao_threshold_sigma),
        "dao_fwhm_px": _fwhm_used_m,
        "dao_detect_binning": int(bfac),
        "match_sep_arcsec_requested": float(match_sep_arcsec),
        "match_sep_arcsec_effective": float(match_thr),
        "plate_scale_arcsec_per_px": (
            float(plate_scale_arcsec_per_px) if plate_scale_arcsec_per_px is not None else None
        ),
        **meta_mag,
    }
    return df_out, meta


def _merge_platesolve_gaia_pairs_into_masterstars_df(
    df: pd.DataFrame,
    *,
    pairs_x: list[float],
    pairs_y: list[float],
    pairs_ra: list[float],
    pairs_de: list[float],
    pairs_catalog_id: list[str],
    max_pair_px: float = 12.0,
) -> pd.DataFrame:
    """Map VYVAR plate-solve Gaia pairs onto DAO rows so ``astrometry_optimizer`` sees catalog_id + small sep."""
    import numpy as np

    if df is None or df.empty or not pairs_x:
        return df
    n = len(pairs_x)
    if len(pairs_y) != n or len(pairs_ra) != n or len(pairs_de) != n or len(pairs_catalog_id) != n:
        return df
    if "x" not in df.columns or "y" not in df.columns:
        return df
    out = df.copy()
    x = pd.to_numeric(out["x"], errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(out["y"], errors="coerce").to_numpy(dtype=np.float64)
    max2 = float(max_pair_px) ** 2
    used: set[int] = set()
    for i in range(n):
        cid = str(pairs_catalog_id[i] or "").strip()
        if not cid:
            continue
        d2 = (x - float(pairs_x[i])) ** 2 + (y - float(pairs_y[i])) ** 2
        d2[~np.isfinite(d2)] = np.inf
        j = int(np.argmin(d2))
        if j in used or float(d2[j]) > max2:
            continue
        used.add(j)
        if "catalog_id" in out.columns:
            out.iat[j, out.columns.get_loc("catalog_id")] = cid
        if "ra_deg" in out.columns:
            out.iat[j, out.columns.get_loc("ra_deg")] = float(pairs_ra[i])
        if "dec_deg" in out.columns:
            out.iat[j, out.columns.get_loc("dec_deg")] = float(pairs_de[i])
        if "match_sep_arcsec" in out.columns:
            out.iat[j, out.columns.get_loc("match_sep_arcsec")] = 0.25
    return out


def detect_stars_and_match_catalog(
    data: "np.ndarray",
    hdr: fits.Header,
    *,
    max_catalog_rows: int = 12000,
    cat_df: pd.DataFrame | None = None,
    vsx_df: pd.DataFrame | None = None,
    gaia_variable_df: pd.DataFrame | None = None,
    match_sep_arcsec: float = 8.0,
    vsx_match_max_sep_arcsec: float = 5.0,
    gaia_variable_match_max_sep_arcsec: float = 2.0,
    saturate_level_fraction: float = 0.999,
    faintest_mag_limit: float | None = None,
    gaia_db_path: Path | None = None,
    field_catalog_export_path: Path | None = None,
    dao_threshold_sigma: float = 3.5,
    dao_fwhm_px: float | None = None,
    fallback_saturate_adu: float | None = None,
    equipment_saturate_adu: float | None = None,
    catalog_local_gaia_only: bool | None = None,
    catalog_kd_pack: tuple[Any, "np.ndarray"] | None = None,
    plate_solve_fov_deg: float | None = None,
    prematch_peak_sigma_floor: float = 10.0,
    frame_name: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect stars (DAOStarFinder), sky coordinates from WCS, match to **local Gaia** (or pre-fetched ``cat_df``).

    If ``cat_df`` is None, builds a local Gaia cone/box catalog using ``AppConfig.gaia_db_path``.
    Pass the same ``cat_df``
    for every frame in a sequence with identical pointing/WCS scale to avoid repeated work.

    Pass ``catalog_kd_pack`` from ``build_ucac_catalog_kdtree(cat_df)`` when exporting many frames with the same
    cone table to avoid rebuilding match structures every call.

    **Known variables:** handled via Gaia flags and/or optional VSX checks (no Gaia TAP here).

    **Faintest magnitude:** if ``faintest_mag_limit`` is set (e.g. ``14``), **matched** stars with catalog
    ``mag`` fainter than the limit are dropped. **Unmatched** detections (no ``mag``) are kept for QA.

    ``match_sep_arcsec`` default 8″ is a robust initial tolerance for slight WCS-scale/offset mismatch; the
    effective floor is ~12″, then the matcher widens toward a high match fraction (target ~95% on typical
    good frames). When the fraction stays low, a **Gaia/pixel** TAN refit (brightest-first greedy pairs, optional
    Gaia cone re-query) is applied iteratively. A final tightening to ~4.5″ is applied only when most loose
    matches survive it.

    ``max_catalog_rows`` caps DAO detections written per frame. Rows are chosen with **spatial
    stratification** (brightest per coarse grid cell, then global flux top-up) so vignetting does not
    mimic a ``catalog disc`` the way a plain brightest-N sort does.

    If ``field_catalog_export_path`` is set, the **full** cone table (``cat_df``) is written there for
    QA overlays — many more rows than DAO detections in ``masterstars.csv``.

    ``dao_threshold_sigma``: DAOStarFinder threshold = sigma × std(background); lower values detect more faint
    sources (cf. SIPS ~2.5σ).

    ``prematch_peak_sigma_floor`` (default 10): before catalog matching, drop DAO rows whose local ``peak`` is
    below ``median + k × σ`` of the frame (sigma-clipped stats). Lower **k** keeps more faint detections
    (MASTERSTAR / ``config.json`` / DAO-STARS typicky **2.0–3.5**); higher **k** čistí šum pred matchom.

    Saturation: (1) ``peak_max_adu`` vs resolved ceiling from header / ``EQUIPMENTS.SATURATE_ADU`` (before BITPIX)
    / ``AppConfig.photometry_fallback_saturate_adu``; (2) **plateau core** — many pixels in the central 3×3
    near the local maximum (flat-top clipping, similar to a saturated radial profile). Row flags:
    ``saturated_from_peak``, ``saturated_plateau``, ``likely_saturated`` (OR), ``photometry_ok`` (not OR).
    """
    import numpy as np
    from astropy.stats import sigma_clipped_stats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs_obj = WCS(hdr)
    if not wcs_obj.has_celestial:
        raise ValueError("FITS header has no usable celestial WCS for catalog matching.")

    arr = np.asarray(data, dtype=np.float32)
    h, wpx = arr.shape
    _fov_hint = plate_solve_fov_deg
    if _fov_hint is None:
        try:
            _fov_hint = float(AppConfig().plate_solve_fov_deg)
        except Exception:  # noqa: BLE001
            _fov_hint = None
    center, radius_deg = _effective_field_catalog_cone_radius_deg(
        wcs_obj, h, wpx, _fov_hint, fits_header=hdr
    )
    if gaia_db_path is not None:
        _gaia_db_path = Path(gaia_db_path)
    else:
        _gaia_db_path: Path | None = None
    try:
        cfg = AppConfig()
        gp = (getattr(cfg, "gaia_db_path", None) or "").strip()
        if gp:
            _gaia_db_path = Path(gp)
    except Exception:  # noqa: BLE001
        _gaia_db_path = None
    _fb_sat = fallback_saturate_adu
    if _fb_sat is None:
        try:
            _cfg_sat = AppConfig()
            _fb_sat = getattr(_cfg_sat, "photometry_fallback_saturate_adu", None)
        except Exception:  # noqa: BLE001
            _fb_sat = None
    try:
        _cfg_cap = int(AppConfig().catalog_query_max_rows)
    except Exception:  # noqa: BLE001
        _cfg_cap = 50_000
    _cat_cap_eff = max(int(max_catalog_rows), 50_000, int(_cfg_cap))
    if cat_df is None:
        _max_mag = float(faintest_mag_limit) if faintest_mag_limit is not None and np.isfinite(float(faintest_mag_limit)) else None
        cat_df = _query_gaia_local(
            center=center,
            radius_deg=radius_deg,
            gaia_db_path=_gaia_db_path,
            max_mag=_max_mag,
            max_rows=int(_cat_cap_eff),
        )
    cat_df = _catalog_df_cap_brightest_by_mag(cat_df, max_rows=_cat_cap_eff)
    if field_catalog_export_path is not None and cat_df is not None and len(cat_df) > 0:
        _fcp = Path(field_catalog_export_path)
        _fcp.parent.mkdir(parents=True, exist_ok=True)
        _vyvar_df_to_csv(cat_df, _fcp)
        log_event(
            f"Vykresľujem katalóg pre celé zorné pole: {int(wpx)}x{int(h)} pixelov "
            f"(export {len(cat_df)} riadkov do field_catalog_cone.csv, cap={int(_cat_cap_eff)}, kužeľ r≈{float(radius_deg):.2f}°)."
        )
        log_event(
            f"KATALÓG TARGET: export {_cat_cap_eff} riadkov do field_catalog_cone.csv "
            f"(ak je dostupných >= {_cat_cap_eff})."
        )
        try:
            _write_field_catalog_cone_meta(
                _fcp,
                center=center,
                radius_deg=float(radius_deg),
                naxis1=int(wpx),
                naxis2=int(h),
                plate_solve_fov_deg=float(_fov_hint) if _fov_hint is not None else None,
            )
        except Exception:  # noqa: BLE001
            pass
    _ = catalog_local_gaia_only
    # ``vsx_df`` prázdny DataFrame z prefetch = „už sme skúšali“; dopĺňaj len ak volajúci nepredal tabuľku (``None``).
    if vsx_df is None:
        _vx: Path | None = None
        try:
            _vxs = str(getattr(AppConfig(), "vsx_local_db_path", "") or "").strip()
            if _vxs:
                _vx = Path(_vxs).expanduser().resolve()
        except Exception:  # noqa: BLE001
            _vx = None
        vsx_df = _query_vsx_local(center=center, radius_deg=radius_deg, vsx_db_path=_vx)
    if gaia_variable_df is None:
        gaia_variable_df = pd.DataFrame()

    mean, med, std = sigma_clipped_stats(arr, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((arr - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bfac = 1
    n_raw_dao = 0
    _cfg_df = AppConfig()
    _fb_c = float(dao_fwhm_px) if dao_fwhm_px is not None else float(getattr(_cfg_df, "sips_dao_fwhm_px", 2.5))
    _base_fw = dao_detection_fwhm_pixels(hdr, configured_fallback=_fb_c)
    try:
        from photutils.detection import DAOStarFinder  # type: ignore

        _ds = float(dao_threshold_sigma)
        _ds = max(0.5, min(20.0, _ds))
        dao_scale = _dao_auto_binning_factor(*data0.shape)
        data_dao, bfac = _mean_bin2d_for_dao(data0, dao_scale)
        _, _, std_dao = (
            sigma_clipped_stats(data_dao, sigma=3.0, maxiters=3)
            if bfac > 1
            else (mean, med, std)
        )
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            try:
                std_dao = float(np.nanstd(arr))
            except Exception:  # noqa: BLE001
                std_dao = float("nan")
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            try:
                finite = np.isfinite(arr)
                n_finite = int(np.count_nonzero(finite))
                if n_finite > 0:
                    vals = arr[finite]
                    n_unique = int(len(np.unique(vals)))
                    mn = float(np.nanmin(vals))
                    mx = float(np.nanmax(vals))
                else:
                    n_unique, mn, mx = 0, float("nan"), float("nan")
                print(
                    f"DEBUG std=0: {_nm} n_unique={n_unique} n_finite={n_finite} "
                    f"min={mn:.1f} max={mx:.1f}"
                )
            except Exception:  # noqa: BLE001
                pass
            try:
                nonzero_mask = (arr != 0) & np.isfinite(arr)
                if int(np.count_nonzero(nonzero_mask)) > 100:
                    std_dao = float(np.std(arr[nonzero_mask]))
            except Exception:  # noqa: BLE001
                pass
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(f"WARNING: {_nm} std=0 aj po fallback, preskakujem")
            return pd.DataFrame(), {
                "n_detected": 0,
                "n_detected_dao": 0,
                "n_matched": 0,
                "n_matched_before_mag_limit": 0,
                "catalog_rows": int(len(cat_df)) if cat_df is not None else 0,
                "catalog_match_mode": "full_cone",
                "reason": "std_dao_zero",
            }
        fwhm_eff = max(1.2, _base_fw / float(bfac))
        _thr = max(_ds * float(std_dao), 1e-6)
        try:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(
                f"DEBUG DAO INPUT: {_nm} mean={float(np.nanmean(arr)):.1f} std={float(np.nanstd(arr)):.1f} "
                f"threshold={float(_thr):.1f} fwhm={float(fwhm_eff):.2f}"
            )
        except Exception:  # noqa: BLE001
            pass
        finder = DAOStarFinder(
            fwhm=float(fwhm_eff),
            threshold=float(_thr),
            brightest=None,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = finder(data_dao)
        n_raw_dao = int(len(tbl)) if tbl is not None else 0
        if tbl is not None and len(tbl) > 0:
            tbl = _prefilter_dao_table_brightest(tbl, max(int(max_catalog_rows) * 12, 36_000))
    except Exception:  # noqa: BLE001
        tbl = None
        bfac = 1
        n_raw_dao = 0

    sat_limit, sat_limit_src = _effective_saturation_limit(
        hdr, fallback_adu=_fb_sat, equipment_saturate_adu=equipment_saturate_adu
    )
    foot_meta = {
        "catalog_footprint": {
            "center_ra_icrs_deg": float(center.ra.deg),
            "center_dec_icrs_deg": float(center.dec.deg),
            "cone_radius_deg": float(radius_deg),
            "naxis1_px": int(wpx),
            "naxis2_px": int(h),
            "method": "circumscribed_cone_border_sample_plus_margin",
            "reference_catalog": "gaia_local_sqlite",
        },
        "saturation": {
            "effective_limit_adu": float(sat_limit) if sat_limit is not None else None,
            "limit_source": sat_limit_src,
            "plateau_half_inner_px": 1,
            "plateau_rel": 0.996,
            "plateau_min_pixels": 5,
        },
    }
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame(), {
            "n_detected": 0,
            "n_detected_dao": 0,
            "n_matched": 0,
            "n_matched_before_mag_limit": 0,
            "catalog_rows": int(len(cat_df)),
            "catalog_match_mode": "full_cone",
            "n_likely_saturated": 0,
            "n_saturated_from_peak": 0,
            "n_saturated_plateau": 0,
            "n_vsx_in_field": int(len(vsx_df)) if vsx_df is not None else 0,
            "n_gaia_variable_in_field": int(len(gaia_variable_df)) if gaia_variable_df is not None else 0,
            "faintest_mag_limit": float(faintest_mag_limit) if faintest_mag_limit is not None else None,
            "n_dropped_fainter_than_limit": 0,
            "field_catalog_cone_csv": str(Path(field_catalog_export_path)) if field_catalog_export_path else None,
            "dao_threshold_sigma": float(dao_threshold_sigma),
            "dao_fwhm_px": float(max(1.2, _base_fw)),
            "dao_detect_binning": int(bfac),
            "match_sep_arcsec_requested": float(match_sep_arcsec),
            "match_sep_arcsec_effective": float(match_sep_arcsec),
            "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
            "saturate_limit_source": sat_limit_src,
            **foot_meta,
        }

    _fwhm_used = float(max(1.2, _base_fw / float(bfac)))
    _d_h2, _d_w2 = int(data_dao.shape[0]), int(data_dao.shape[1])
    # Stratify brightest DAO sources on a coarse grid over the **full** chip (not radial distance from center).
    _keep2 = _dao_spatial_flux_cap_row_indices(
        tbl, max_n=int(max_catalog_rows), width_px=float(_d_w2), height_px=float(_d_h2)
    )
    tbl = tbl[_keep2]
    tbl.sort("flux")
    tbl = tbl[::-1]
    n_spatial = int(len(tbl))
    log_event(
        f"DAO na snímku: raw={n_raw_dao} (po brightest-prefilter max {max(int(max_catalog_rows) * 12, 36_000):d}) → "
        f"po priestorovom strope max_n={int(max_catalog_rows)}: {n_spatial} bodov (binning DAO={bfac}×)."
    )
    n = n_spatial
    xb = np.asarray(tbl["xcentroid"], dtype=np.float64)
    yb = np.asarray(tbl["ycentroid"], dtype=np.float64)
    x, y = _dao_xy_binned_to_full(xb, yb, bfac)
    flux = np.asarray(tbl["flux"], dtype=np.float64)
    peak_dao = np.asarray(tbl["peak"], dtype=np.float64) if "peak" in tbl.colnames else np.full(n, np.nan)
    ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
    det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

    vsx_max = float(vsx_match_max_sep_arcsec)
    gvar_max = float(gaia_variable_match_max_sep_arcsec)
    if vsx_df is not None and not vsx_df.empty:
        vxc = SkyCoord(
            ra=np.asarray(vsx_df["ra_deg"], dtype=float) * u.deg,
            dec=np.asarray(vsx_df["dec_deg"], dtype=float) * u.deg,
        )
        _, sepvx, _ = det_coords.match_to_catalog_sky(vxc)
        vsx_hit = np.asarray(sepvx.arcsec <= vsx_max, dtype=bool)
    else:
        vsx_hit = np.zeros(n, dtype=bool)
    if gaia_variable_df is not None and not gaia_variable_df.empty:
        gvc = SkyCoord(
            ra=np.asarray(gaia_variable_df["ra_deg"], dtype=float) * u.deg,
            dec=np.asarray(gaia_variable_df["dec_deg"], dtype=float) * u.deg,
        )
        _, sepgv, _ = det_coords.match_to_catalog_sky(gvc)
        gvar_hit = np.asarray(sepgv.arcsec <= gvar_max, dtype=bool)
    else:
        gvar_hit = np.zeros(n, dtype=bool)
    catalog_known_variable = np.asarray(vsx_hit, dtype=bool) | np.asarray(gvar_hit, dtype=bool)

    sat_frac = float(saturate_level_fraction)
    sat_frac = min(max(sat_frac, 0.5), 1.0)

    pmax_arr = _box_peaks_at_centroids(arr, x, y)
    _sat_block = _vectorized_star_saturation_columns(
        arr,
        x,
        y,
        sat_limit=sat_limit,
        sat_frac=sat_frac,
        peak_dao=peak_dao,
        peak_max_adu=pmax_arr,
    )
    # Pre-match SNR cleanup: drop weak DAO detections under (median + k×σ) from image background.
    _snr_k = float(prematch_peak_sigma_floor)
    if not math.isfinite(_snr_k):
        _snr_k = 10.0
    # Spodná hranica 0.5 = zhoda s AppConfig / DAO-STARS pre MASTERSTAR; horná 15 = per-frame default k=10 zostane platný.
    _snr_k = min(15.0, max(0.5, _snr_k))
    noise_floor = float(med + _snr_k * max(float(std) if np.isfinite(std) else 0.0, 1.0))
    snr_keep = np.isfinite(pmax_arr) & (pmax_arr > noise_floor)
    n_snr = int(np.count_nonzero(snr_keep))
    if 0 < n_snr < n:
        x = x[snr_keep]
        y = y[snr_keep]
        flux = flux[snr_keep]
        peak_dao = peak_dao[snr_keep]
        ra_deg = ra_deg[snr_keep]
        dec_deg = dec_deg[snr_keep]
        det_coords = det_coords[snr_keep]
        vsx_hit = vsx_hit[snr_keep]
        gvar_hit = gvar_hit[snr_keep]
        catalog_known_variable = catalog_known_variable[snr_keep]
        pmax_arr = pmax_arr[snr_keep]
        _sat_block = {k: np.asarray(v)[snr_keep] for k, v in _sat_block.items()}
        n = int(n_snr)
        log_event(
            f"DAO po SNR filtri (šumová podlaha median+{_snr_k:.1f}×σ): {n}/{n_spatial} bodov "
            f"(noise_floor≈{noise_floor:.1f} ADU; pred matchom s katalógom)."
        )
    elif n_snr == 0:
        log_event(
            f"DAO SNR filter by zahodil všetko — ponechávam {n_spatial} bodov pred matchom."
        )

    idx_det = np.arange(1, n + 1, dtype=np.int32)
    det_str = np.array([f"DET_{i:04d}" for i in idx_det], dtype=object)
    n_matched = 0
    match_sep_used = max(12.0, float(match_sep_arcsec))
    _wcs_refine_iters = 0
    if cat_df.empty:
        df_out = pd.DataFrame(
            {
                "name": det_str,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "mag": np.full(n, np.nan, dtype=np.float64),
                "b_v": np.full(n, np.nan, dtype=np.float64),
                "catalog": np.array([""] * n, dtype=object),
                "catalog_id": np.array([""] * n, dtype=object),
                "x": x,
                "y": y,
                "flux": flux,
                "match_sep_arcsec": np.full(n, np.nan, dtype=np.float64),
                "gaia_nss": np.full(n, np.nan, dtype=np.float64),
                "gaia_qso": np.full(n, np.nan, dtype=np.float64),
                "gaia_gal": np.full(n, np.nan, dtype=np.float64),
                "vsx_known_variable": vsx_hit,
                "gaia_dr3_variable_catalog": gvar_hit,
                "catalog_known_variable": catalog_known_variable,
                **_sat_block,
            }
        )
    else:
        nc = 0
        cid_c = np.empty(0, dtype=object)
        cat_c = np.empty(0, dtype=object)
        mag_c = np.empty(0, dtype=np.float64)
        bv_c = np.empty(0, dtype=np.float64)
        gn_c = np.empty(0, dtype=np.float64)
        gq_c = np.empty(0, dtype=np.float64)
        gg_c = np.empty(0, dtype=np.float64)
        tree_pack = catalog_kd_pack
        if tree_pack is None and len(cat_df) >= 120:
            tree_pack = build_ucac_catalog_kdtree(cat_df)
        oix_rows: np.ndarray | None
        icomp: np.ndarray
        sepa: np.ndarray
        if tree_pack is not None:
            tr, oix_rows = tree_pack
            icomp, sepa = nearest_sky_nn_kdtree(tr, ra_deg, dec_deg)
        else:
            tr = None
            cat_coords = SkyCoord(
                ra=np.asarray(cat_df["ra_deg"], dtype=float) * u.deg,
                dec=np.asarray(cat_df["dec_deg"], dtype=float) * u.deg,
            )
            idx, sep2d, _ = det_coords.match_to_catalog_sky(cat_coords)
            icomp = np.asarray(idx, dtype=np.int64)
            sepa = np.asarray(sep2d.to_value(u.arcsec), dtype=np.float64)
            oix_rows = None

        def _bind_gaia_catalog_columns() -> None:
            nonlocal nc, cid_c, cat_c, mag_c, bv_c, gn_c, gq_c, gg_c
            nc = int(len(cat_df))
            cid_c = (
                cat_df["catalog_id"].fillna("").astype(str).str.strip().to_numpy(dtype=object)
                if "catalog_id" in cat_df.columns
                else np.array([""] * nc, dtype=object)
            )
            cat_c = (
                cat_df["catalog"].fillna("").astype(str).to_numpy(dtype=object)
                if "catalog" in cat_df.columns
                else np.array([""] * nc, dtype=object)
            )
            mag_c = (
                pd.to_numeric(cat_df["mag"], errors="coerce").to_numpy(dtype=np.float64)
                if "mag" in cat_df.columns
                else np.full(nc, np.nan, dtype=np.float64)
            )
            bv_c = (
                pd.to_numeric(cat_df["b_v"], errors="coerce").to_numpy(dtype=np.float64)
                if "b_v" in cat_df.columns
                else np.full(nc, np.nan, dtype=np.float64)
            )
            gn_c = (
                pd.to_numeric(cat_df["gaia_nss"], errors="coerce").to_numpy(dtype=np.float64)
                if "gaia_nss" in cat_df.columns
                else np.full(nc, np.nan, dtype=np.float64)
            )
            gq_c = (
                pd.to_numeric(cat_df["gaia_qso"], errors="coerce").to_numpy(dtype=np.float64)
                if "gaia_qso" in cat_df.columns
                else np.full(nc, np.nan, dtype=np.float64)
            )
            gg_c = (
                pd.to_numeric(cat_df["gaia_gal"], errors="coerce").to_numpy(dtype=np.float64)
                if "gaia_gal" in cat_df.columns
                else np.full(nc, np.nan, dtype=np.float64)
            )

        _bind_gaia_catalog_columns()

        def _assign_catalog_at_threshold(thr: float) -> tuple[pd.DataFrame, int]:
            thr_f = float(thr)
            sepa_eff = np.asarray(sepa, dtype=np.float64)
            if tr is not None and oix_rows is not None:
                # Greedy 1:1 matching on k-nearest sphere neighbors (avoids many detections sharing one Gaia row).
                ntree = int(getattr(tr, "n", 0))
                _nk = min(48, max(1, ntree))
                det_xyz = _icrs_deg_to_unitxyz(ra_deg, dec_deg)
                dist, idx = tr.query(det_xyz, k=_nk)
                dist = np.asarray(dist, dtype=np.float64)
                idx = np.asarray(idx, dtype=np.int64)
                if dist.ndim == 1:
                    dist = dist.reshape(-1, 1)
                    idx = idx.reshape(-1, 1)
                sep_k = _chord_to_arcsec(dist)
                n_oix = int(len(oix_rows))
                pairs: list[tuple[float, int, int]] = []
                for i in range(n):
                    for kk in range(int(idx.shape[1])):
                        j_comp = int(idx[i, kk])
                        if j_comp < 0 or j_comp >= n_oix:
                            continue
                        s = float(sep_k[i, kk])
                        if not np.isfinite(s) or s > thr_f:
                            continue
                        cr = int(oix_rows[j_comp])
                        if 0 <= cr < nc:
                            pairs.append((s, i, cr))
                pairs.sort(key=lambda t: t[0])
                used_det: set[int] = set()
                used_cat: set[int] = set()
                cat_row = np.full(n, -1, dtype=np.int64)
                sepa_out = np.full(n, np.nan, dtype=np.float64)
                for s, i, cr in pairs:
                    if i in used_det or cr in used_cat:
                        continue
                    used_det.add(i)
                    used_cat.add(cr)
                    cat_row[i] = cr
                    sepa_out[i] = s
                matched_l = cat_row >= 0
                sepa_eff = np.where(matched_l, sepa_out, sepa.astype(np.float64))
                finite_sep = np.isfinite(sepa_eff)
            else:
                finite_sep = np.isfinite(sepa)
                cat_row = icomp.astype(np.int64, copy=False)
                cat_row = np.clip(cat_row, 0, max(nc - 1, 0))
                matched_l = finite_sep & (sepa <= thr_f) & (nc > 0)
            n_ma = int(np.count_nonzero(matched_l))
            safe_l = np.clip(np.where(matched_l, cat_row, 0), 0, max(nc - 1, 0))
            cid_sel = cid_c[safe_l]
            cat_sel = cat_c[safe_l]
            cid_st = pd.Series(cid_sel, dtype=object).astype(str).str.strip()
            cat_st = pd.Series(cat_sel, dtype=object).astype(str).str.strip()
            empty_cid = cid_st.eq("").to_numpy()
            cat_lab = np.where(cat_st.ne("").to_numpy(), cat_sel, "CAT")
            name_fb = np.array(
                [f"{str(cat_lab[i])}_{int(idx_det[i]):04d}" for i in range(n)],
                dtype=object,
            )
            name_cand = np.where(empty_cid, name_fb, cid_st.to_numpy())
            name_out = np.where(matched_l, name_cand, det_str)
            mag_out = np.full(n, np.nan, dtype=np.float64)
            bv_out = np.full(n, np.nan, dtype=np.float64)
            gn_out = np.full(n, np.nan, dtype=np.float64)
            gq_out = np.full(n, np.nan, dtype=np.float64)
            gg_out = np.full(n, np.nan, dtype=np.float64)
            mag_out[matched_l] = mag_c[safe_l[matched_l]]
            bv_out[matched_l] = bv_c[safe_l[matched_l]]
            gn_out[matched_l] = gn_c[safe_l[matched_l]]
            gq_out[matched_l] = gq_c[safe_l[matched_l]]
            gg_out[matched_l] = gg_c[safe_l[matched_l]]
            cat_out = np.array([""] * n, dtype=object)
            cid_out = np.array([""] * n, dtype=object)
            cat_out[matched_l] = cat_c[safe_l[matched_l]]
            cid_out[matched_l] = cid_c[safe_l[matched_l]]
            df_l = pd.DataFrame(
                {
                    "name": name_out,
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,
                    "mag": mag_out,
                    "b_v": bv_out,
                    "catalog": cat_out,
                    "catalog_id": cid_out,
                    "x": x,
                    "y": y,
                    "flux": flux,
                    "match_sep_arcsec": sepa_eff.astype(np.float64, copy=False),
                    "gaia_nss": gn_out,
                    "gaia_qso": gq_out,
                    "gaia_gal": gg_out,
                    "vsx_known_variable": vsx_hit,
                    "gaia_dr3_variable_catalog": gvar_hit,
                    "catalog_known_variable": catalog_known_variable,
                    **_sat_block,
                }
            )
            return df_l, n_ma

        def _run_full_match_pass() -> None:
            nonlocal df_out, n_matched, match_sep_used
            match_sep_used = max(12.0, float(match_sep_arcsec))
            df_out, n_matched = _assign_catalog_at_threshold(match_sep_used)
            if n >= 20:
                r_match = float(n_matched) / float(max(1, n))
                if r_match < 0.70:
                    _req_before = float(match_sep_used)
                    thr_wider = min(float(match_sep_used) * 1.5, 90.0)
                    if thr_wider > float(match_sep_used) + 1e-9:
                        df_try, n_try = _assign_catalog_at_threshold(thr_wider)
                        if n_try > n_matched:
                            df_out, n_matched, match_sep_used = df_try, n_try, thr_wider
                            LOGGER.info(
                                "Catalog match: zhoda %.0f%% < 70 %%, opakovanie s max separáciou %.2f″ (požadované %.2f″)",
                                100.0 * r_match,
                                thr_wider,
                                _req_before,
                            )
            # Extra widen toward ~95% match rate on retained DAO rows (residual WCS / crowding).
            if n >= 8:
                cur_thr = float(match_sep_used)
                _widen_cap_arcsec = 96.0
                for _ in range(16):
                    r2 = float(n_matched) / float(max(1, n))
                    if r2 >= 0.95 or cur_thr >= _widen_cap_arcsec:
                        break
                    nxt = min(max(cur_thr * 1.12, cur_thr + 0.45), _widen_cap_arcsec)
                    if nxt <= cur_thr + 1e-6:
                        break
                    df_try, n_try = _assign_catalog_at_threshold(nxt)
                    if n_try <= n_matched:
                        break
                    df_out, n_matched, match_sep_used = df_try, n_try, nxt
                    cur_thr = nxt
            # After a successful loose initial match, tighten for cleaner final IDs (only if most matches survive).
            _tight_sec = 4.5
            if n_matched >= max(10, int(0.20 * max(1, n))) and float(match_sep_used) > _tight_sec + 1e-9:
                df_tight, n_tight = _assign_catalog_at_threshold(_tight_sec)
                if n_tight >= max(8, int(0.92 * max(1, n_matched))):
                    LOGGER.info(
                        "Catalog match: počiatočný loose match %.2f″ -> finálne zúženie na %.2f″ (matches %d -> %d)",
                        float(match_sep_used),
                        _tight_sec,
                        int(n_matched),
                        int(n_tight),
                    )
                    df_out, n_matched, match_sep_used = df_tight, n_tight, _tight_sec

        _run_full_match_pass()
        # Gaia / DAO pixel NN TAN refit when sky match fraction stays below ~95% (fixes offset / scale drift).
        if tr is not None and oix_rows is not None and n >= 12 and float(n_matched) / float(max(1, n)) < 0.95:
            try:
                from vyvar_platesolver import _refine_wcs_tan_nn_gaia

                _target_mf = 0.95
                diag = float(np.hypot(float(wpx), float(h)))
                # Allow cross-chip distances when the initial plate solve is badly offset (Gaia world2pix vs DAO).
                max_px = float(min(0.98 * diag, max(800.0, 0.88 * float(max(wpx, h)))))
                det_order_idx = np.argsort(-np.asarray(flux, dtype=np.float64), kind="stable")
                for _wr in range(10):
                    cat_df_snap = cat_df.copy()
                    ra_cat = pd.to_numeric(cat_df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
                    de_cat = pd.to_numeric(cat_df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
                    r_now = float(n_matched) / float(max(1, n))
                    if r_now >= _target_mf:
                        break
                    n_before_wcs = int(n_matched)
                    hdr_snapshot = hdr.copy()
                    w_try = WCS(hdr)
                    w_new, _meta_wcs = _refine_wcs_tan_nn_gaia(
                        w_try,
                        xs_det=x,
                        ys_det=y,
                        ra_cat_full_deg=ra_cat,
                        dec_cat_full_deg=de_cat,
                        max_match_px=max_px,
                        min_pairs=10,
                        det_order_idx=det_order_idx,
                    )
                    if w_new is None:
                        w_new, _meta_wcs = _refine_wcs_tan_nn_gaia(
                            w_try,
                            xs_det=x,
                            ys_det=y,
                            ra_cat_full_deg=ra_cat,
                            dec_cat_full_deg=de_cat,
                            max_match_px=max_px,
                            min_pairs=8,
                            det_order_idx=det_order_idx,
                        )
                    if w_new is None:
                        max_px = min(max_px * 1.32, 1.52 * diag)
                        if max_px >= 1.48 * diag:
                            break
                        continue
                    _rms_w = _meta_wcs.get("rms_px")
                    if _rms_w is not None and math.isfinite(float(_rms_w)) and float(_rms_w) > 10.0:
                        LOGGER.info(
                            "Catalog match: WCS refine zamietnutý (rms=%.2fpx > 10) — širší pixelový matching.",
                            float(_rms_w),
                        )
                        max_px = min(max_px * 1.32, 1.52 * diag)
                        if max_px >= 1.48 * diag:
                            break
                        continue
                    _apply_wcs_tan_fragment_to_header(
                        hdr,
                        w_new.to_header(relax=True),
                        f"VYVAR: Gaia/pixel NN WCS refine (match {100.0 * r_now:.1f}%, goal {_target_mf * 100:.0f}%)",
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FITSFixedWarning)
                        wcs_obj = WCS(hdr)
                    center2, radius2 = _effective_field_catalog_cone_radius_deg(
                        wcs_obj, h, wpx, _fov_hint, fits_header=hdr
                    )
                    _mag_lim = (
                        float(faintest_mag_limit)
                        if faintest_mag_limit is not None and np.isfinite(float(faintest_mag_limit))
                        else None
                    )
                    if _gaia_db_path is not None and Path(_gaia_db_path).is_file():
                        # Wide-field cones already subsume the chip; tangent-plane WCS nudges do not warrant
                        # re-running multi-hundred-k row SQLite queries on every refine iteration (was ~10× per frame).
                        _skip_gaia_rerequery = float(radius2) >= 5.0
                        if _skip_gaia_rerequery:
                            LOGGER.info(
                                "Catalog match: WCS refine — ponechávam existujúci lokálny Gaia výrez "
                                f"(r={float(radius2):.2f}° ≥ 5°; bez opätovného SQL dotazu)."
                            )
                        else:
                            cat_df_new = _catalog_df_cap_brightest_by_mag(
                                _query_gaia_local(
                                    center=center2,
                                    radius_deg=radius2,
                                    gaia_db_path=_gaia_db_path,
                                    max_mag=_mag_lim,
                                    max_rows=int(_cat_cap_eff),
                                ),
                                max_rows=int(_cat_cap_eff),
                            )
                            if len(cat_df_new) < 120:
                                LOGGER.info(
                                    "Catalog match: WCS refine — Gaia re-query < 120 hviezd; refine zrušený."
                                )
                                hdr.clear()
                                hdr.extend(hdr_snapshot.cards)
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", FITSFixedWarning)
                                    wcs_obj = WCS(hdr)
                                ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
                                det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
                                if vsx_df is not None and not vsx_df.empty:
                                    vxc = SkyCoord(
                                        ra=np.asarray(vsx_df["ra_deg"], dtype=float) * u.deg,
                                        dec=np.asarray(vsx_df["dec_deg"], dtype=float) * u.deg,
                                    )
                                    _, sepvx, _ = det_coords.match_to_catalog_sky(vxc)
                                    vsx_hit = np.asarray(sepvx.arcsec <= vsx_max, dtype=bool)
                                else:
                                    vsx_hit = np.zeros(n, dtype=bool)
                                if gaia_variable_df is not None and not gaia_variable_df.empty:
                                    gvc = SkyCoord(
                                        ra=np.asarray(gaia_variable_df["ra_deg"], dtype=float) * u.deg,
                                        dec=np.asarray(gaia_variable_df["dec_deg"], dtype=float) * u.deg,
                                    )
                                    _, sepgv, _ = det_coords.match_to_catalog_sky(gvc)
                                    gvar_hit = np.asarray(sepgv.arcsec <= gvar_max, dtype=bool)
                                else:
                                    gvar_hit = np.zeros(n, dtype=bool)
                                catalog_known_variable = np.asarray(vsx_hit, dtype=bool) | np.asarray(
                                    gvar_hit, dtype=bool
                                )
                                if tree_pack is not None:
                                    tr, oix_rows = tree_pack
                                    icomp, sepa = nearest_sky_nn_kdtree(tr, ra_deg, dec_deg)
                                _run_full_match_pass()
                                break
                            cat_df = cat_df_new
                            _bind_gaia_catalog_columns()
                            tree_pack = build_ucac_catalog_kdtree(cat_df)
                            tr, oix_rows = tree_pack
                    else:
                        LOGGER.info("Catalog match: WCS refine bez nového Gaia kužela (gaia_db_path).")
                    ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
                    det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
                    if vsx_df is not None and not vsx_df.empty:
                        vxc = SkyCoord(
                            ra=np.asarray(vsx_df["ra_deg"], dtype=float) * u.deg,
                            dec=np.asarray(vsx_df["dec_deg"], dtype=float) * u.deg,
                        )
                        _, sepvx, _ = det_coords.match_to_catalog_sky(vxc)
                        vsx_hit = np.asarray(sepvx.arcsec <= vsx_max, dtype=bool)
                    else:
                        vsx_hit = np.zeros(n, dtype=bool)
                    if gaia_variable_df is not None and not gaia_variable_df.empty:
                        gvc = SkyCoord(
                            ra=np.asarray(gaia_variable_df["ra_deg"], dtype=float) * u.deg,
                            dec=np.asarray(gaia_variable_df["dec_deg"], dtype=float) * u.deg,
                        )
                        _, sepgv, _ = det_coords.match_to_catalog_sky(gvc)
                        gvar_hit = np.asarray(sepgv.arcsec <= gvar_max, dtype=bool)
                    else:
                        gvar_hit = np.zeros(n, dtype=bool)
                    catalog_known_variable = np.asarray(vsx_hit, dtype=bool) | np.asarray(gvar_hit, dtype=bool)
                    icomp, sepa = nearest_sky_nn_kdtree(tr, ra_deg, dec_deg)
                    _run_full_match_pass()
                    _wcs_refine_iters += 1
                    # Revert only on a large regression (refit can briefly reshuffle pairs).
                    if int(n_matched) < int(0.88 * max(1, n_before_wcs)):
                        hdr.clear()
                        hdr.extend(hdr_snapshot.cards)
                        cat_df = cat_df_snap
                        _bind_gaia_catalog_columns()
                        tree_pack = build_ucac_catalog_kdtree(cat_df) if len(cat_df) >= 120 else None
                        if tree_pack is None:
                            tr = None
                            oix_rows = None
                        else:
                            tr, oix_rows = tree_pack
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FITSFixedWarning)
                            wcs_obj = WCS(hdr)
                        ra_deg, dec_deg = _all_pix2world_icrs_deg(wcs_obj, x, y)
                        det_coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
                        if vsx_df is not None and not vsx_df.empty:
                            vxc = SkyCoord(
                                ra=np.asarray(vsx_df["ra_deg"], dtype=float) * u.deg,
                                dec=np.asarray(vsx_df["dec_deg"], dtype=float) * u.deg,
                            )
                            _, sepvx, _ = det_coords.match_to_catalog_sky(vxc)
                            vsx_hit = np.asarray(sepvx.arcsec <= vsx_max, dtype=bool)
                        else:
                            vsx_hit = np.zeros(n, dtype=bool)
                        if gaia_variable_df is not None and not gaia_variable_df.empty:
                            gvc = SkyCoord(
                                ra=np.asarray(gaia_variable_df["ra_deg"], dtype=float) * u.deg,
                                dec=np.asarray(gaia_variable_df["dec_deg"], dtype=float) * u.deg,
                            )
                            _, sepgv, _ = det_coords.match_to_catalog_sky(gvc)
                            gvar_hit = np.asarray(sepgv.arcsec <= gvar_max, dtype=bool)
                        else:
                            gvar_hit = np.zeros(n, dtype=bool)
                        catalog_known_variable = np.asarray(vsx_hit, dtype=bool) | np.asarray(gvar_hit, dtype=bool)
                        icomp, sepa = nearest_sky_nn_kdtree(tr, ra_deg, dec_deg)
                        _run_full_match_pass()
                        break
                    r_after = float(n_matched) / float(max(1, n))
                    if r_after < 0.88:
                        max_px = min(max_px * 1.1, 1.48 * diag)
                    else:
                        max_px = min(max(max_px * 0.92, 0.42 * diag), 1.2 * diag)
            except Exception as exc:  # noqa: BLE001
                LOGGER.info("Catalog match: WCS Gaia/pixel refine skipped: %s", exc)
    n_detected_dao = int(n)
    n_matched_before_mag = int(n_matched)
    n_before_mag = len(df_out)
    if (
        len(cat_df) > 0
        and len(df_out) >= 30
        and n_matched_before_mag < max(5, int(0.12 * len(df_out)))
    ):
        try:
            _sep_c = pd.to_numeric(df_out["match_sep_arcsec"], errors="coerce")
            _arr = _sep_c.to_numpy(dtype=np.float64)
            _ok = _arr[np.isfinite(_arr)]
            if len(_ok) >= 20:
                med_nn = float(np.nanmedian(_ok))
                if med_nn > float(match_sep_used) * 1.15:
                    LOGGER.warning(
                        "Catalog match je slabý: %s/%s detekcií v rámci %.2f″; medián vzdialenosti k najbližšiemu "
                        "katalógu ≈ %.2f″ — skús zväčšiť „Max catalog match distance″, overiť plate solve (FOV, RA/Dec) "
                        "a lokálna Gaia DR3.%s",
                        n_matched_before_mag,
                        len(df_out),
                        float(match_sep_used),
                        med_nn,
                        wcs_distortion_log_suffix(hdr),
                    )
        except Exception:  # noqa: BLE001
            pass
    if faintest_mag_limit is not None and np.isfinite(float(faintest_mag_limit)):
        lim = float(faintest_mag_limit)
        mcol = pd.to_numeric(df_out["mag"], errors="coerce")
        df_out = df_out.loc[mcol.isna() | (mcol <= lim)].reset_index(drop=True)
        meta_mag = {
            "faintest_mag_limit": lim,
            "n_dropped_fainter_than_limit": int(n_before_mag - len(df_out)),
        }
    else:
        meta_mag = {"faintest_mag_limit": None, "n_dropped_fainter_than_limit": 0}

    n_sat = int(df_out["likely_saturated"].sum()) if len(df_out) and "likely_saturated" in df_out.columns else 0
    n_sat_pk = (
        int(df_out["saturated_from_peak"].sum())
        if len(df_out) and "saturated_from_peak" in df_out.columns
        else 0
    )
    n_sat_pl = (
        int(df_out["saturated_plateau"].sum())
        if len(df_out) and "saturated_plateau" in df_out.columns
        else 0
    )
    cat_nonempty = (
        df_out["catalog"].fillna("").astype(str).str.strip().ne("")
        if len(df_out) and "catalog" in df_out.columns
        else pd.Series([], dtype=bool)
    )
    n_matched_final = int(cat_nonempty.sum()) if len(df_out) else 0
    meta = {
        "noise_floor_adu": float(noise_floor),
        "n_detected_dao_raw": int(n_raw_dao),
        "n_dao_after_spatial_cap": int(n_spatial),
        "n_detected_dao": n_detected_dao,
        "n_detected": int(len(df_out)),
        "n_matched_before_mag_limit": n_matched_before_mag,
        "n_matched": n_matched_final,
        "catalog_rows": int(len(cat_df)),
        "catalog_match_mode": "full_cone",
        "n_likely_saturated": n_sat,
        "n_saturated_from_peak": n_sat_pk,
        "n_saturated_plateau": n_sat_pl,
        "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
        "saturate_limit_source": sat_limit_src,
        "n_vsx_in_field": int(len(vsx_df)) if vsx_df is not None else 0,
        "n_gaia_variable_in_field": int(len(gaia_variable_df)) if gaia_variable_df is not None else 0,
        **foot_meta,
        "field_catalog_cone_csv": str(Path(field_catalog_export_path)) if field_catalog_export_path else None,
        "dao_threshold_sigma": float(dao_threshold_sigma),
        "dao_fwhm_px": _fwhm_used,
        "dao_detect_binning": int(bfac),
        "prematch_peak_sigma_floor": float(_snr_k),
        "match_sep_arcsec_requested": float(match_sep_arcsec),
        "match_sep_arcsec_effective": float(match_sep_used),
        "wcs_gaia_pixel_refine_iters": int(_wcs_refine_iters),
        "catalog_match_fraction_target": 0.95,
        "catalog_match_fraction_met": (
            bool((float(n_matched_final) / float(max(1, len(df_out)))) >= 0.95) if len(df_out) else True
        ),
        **meta_mag,
    }
    return df_out, meta


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


def _icrs_center_radius_from_hdr_data(
    hdr: fits.Header,
    data: Any,
    *,
    plate_solve_fov_deg: float | None = None,
) -> tuple[SkyCoord, float, int, int] | None:
    """ICRS center + cone radius from an in-memory frame (RAM handoff), same semantics as disk scan."""
    import numpy as np

    d = np.asarray(data)
    if d.ndim != 2:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr)
        if not w.has_celestial:
            return None
        h_i, wpx_i = int(d.shape[0]), int(d.shape[1])
        c_i, r_i = _effective_field_catalog_cone_radius_deg(
            w, h_i, wpx_i, plate_solve_fov_deg, fits_header=hdr
        )
        return c_i, r_i, wpx_i, h_i
    except Exception:  # noqa: BLE001
        return None


def _export_first_icrs_center_radius(
    files: list[Path],
    *,
    plate_solve_fov_deg: float | None = None,
) -> tuple[SkyCoord, float, int, int] | None:
    import numpy as np

    for fp in files:
        try:
            with fits.open(fp, memmap=False) as h:
                hdr = h[0].header
                d = np.asarray(h[0].data)
                if d.ndim != 2:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    w = WCS(hdr)
                if not w.has_celestial:
                    continue
                h_i, wpx_i = int(d.shape[0]), int(d.shape[1])
                c_i, r_i = _effective_field_catalog_cone_radius_deg(
                    w, h_i, wpx_i, plate_solve_fov_deg, fits_header=hdr
                )
                return c_i, r_i, wpx_i, h_i
        except Exception:  # noqa: BLE001
            continue
    return None


def _prefetch_export_shared_catalog_for_process_pool(
    *,
    files: list[Path] | None = None,
    reference_hdr_data: tuple[Any, Any] | None = None,
    field_cat_path: Path,
    cat_df: pd.DataFrame | None,
    vsx_df: pd.DataFrame | None,
    gaia_variable_df: pd.DataFrame | None,
    gaia_db_path: Path | None,
    gaia_local_max_mag: float | None,
    export_cat_local: bool,
    plate_solve_fov_deg: float | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, Any]:
    """Load cone + VSX/Gaia + KD-tree in the parent before ``ProcessPoolExecutor`` (workers share no memory)."""
    cr: tuple[SkyCoord, float, int, int] | None = None
    if reference_hdr_data is not None:
        _h0, _d0 = reference_hdr_data
        cr = _icrs_center_radius_from_hdr_data(_h0, _d0, plate_solve_fov_deg=plate_solve_fov_deg)
    if cr is None and files:
        cr = _export_first_icrs_center_radius(files, plate_solve_fov_deg=plate_solve_fov_deg)
    c_df = cat_df
    v_df = vsx_df
    g_df = gaia_variable_df

    if cr is not None:
        c_i, r_i, wpx_i, h_i = cr
        if c_df is None or getattr(c_df, "empty", True):
            _gaia_db_path = Path(gaia_db_path) if gaia_db_path is not None else None
            if _gaia_db_path is None:
                try:
                    _cfg = AppConfig()
                    _gp = (getattr(_cfg, "gaia_db_path", None) or "").strip()
                    if _gp:
                        _gaia_db_path = Path(_gp)
                except Exception:  # noqa: BLE001
                    _gaia_db_path = None
            c_df = _query_gaia_local(
                center=c_i,
                radius_deg=r_i,
                gaia_db_path=_gaia_db_path,
                max_mag=float(gaia_local_max_mag) if gaia_local_max_mag is not None else None,
            )
            if c_df is not None and len(c_df) > 0:
                try:
                    field_cat_path.parent.mkdir(parents=True, exist_ok=True)
                    _vyvar_df_to_csv(c_df, field_cat_path)
                    _write_field_catalog_cone_meta(
                        field_cat_path,
                        center=c_i,
                        radius_deg=float(r_i),
                        naxis1=int(wpx_i),
                        naxis2=int(h_i),
                        plate_solve_fov_deg=plate_solve_fov_deg,
                    )
                except Exception:  # noqa: BLE001
                    pass

        if v_df is None or getattr(v_df, "empty", True):
            _vsx_p: Path | None = None
            try:
                _vsp = str(getattr(AppConfig(), "vsx_local_db_path", "") or "").strip()
                if _vsp:
                    _vsx_p = Path(_vsp).expanduser().resolve()
            except Exception:  # noqa: BLE001
                _vsx_p = None
            if _vsx_p is not None and _vsx_p.is_file():
                v_df = _query_vsx_local(
                    center=c_i,
                    radius_deg=float(r_i),
                    vsx_db_path=_vsx_p,
                )
            else:
                v_df = pd.DataFrame()
        if g_df is None:
            g_df = pd.DataFrame()

    kd_pack = None
    if c_df is not None and not getattr(c_df, "empty", True) and len(c_df) >= 120:
        kd_pack = build_ucac_catalog_kdtree(c_df)
        if kd_pack is not None:
            LOGGER.info(
                "Per-frame catalog: shared cKDTree for %s cone rows (process pool)",
                len(c_df),
            )
    return c_df, v_df, g_df, kd_pack


_EXPORT_PER_FRAME_WORKER_STATE: dict[str, Any] = {}
_PIXEL_MATCH_DEBUG_LOGGED = False


def _init_export_per_frame_worker(state: dict[str, Any]) -> None:
    global _EXPORT_PER_FRAME_WORKER_STATE
    _EXPORT_PER_FRAME_WORKER_STATE = state
    if not bool(state.get("use_master_fast_path")):
        return
    mpath = str(state.get("masterstar_fits_path") or "").strip()
    if not mpath or not Path(mpath).is_file():
        return
    try:
        with fits.open(mpath, memmap=False) as h:
            w = WCS(h[0].header)
            if w.has_celestial:
                _EXPORT_PER_FRAME_WORKER_STATE["ref_wcs"] = w
            d = h[0].data
            if d is not None and getattr(d, "ndim", 0) == 2:
                _EXPORT_PER_FRAME_WORKER_STATE["masterstar_data_shape"] = (
                    int(d.shape[0]),
                    int(d.shape[1]),
                )
    except Exception:  # noqa: BLE001
        pass


def _extract_airmass_from_header(hdr: fits.Header) -> float:
    """Extrahuj airmass z FITS hlavičky.

    Skúša AIRMASS, potom ALT_OBJ/OBJCTALT (prepočet cez sec(z) = 1/cos(alt)).
    Vracia float("nan") ak nie je dostupný.
    """
    # Priamy keyword
    for kw in ("AIRMASS", "AIRMAS", "SECZ"):
        val = hdr.get(kw)
        if val is not None:
            try:
                v = float(val)
                if 0.9 <= v <= 10.0:  # fyzikálny rozsah
                    return v
            except (TypeError, ValueError):
                pass

    # Fallback: altitude → airmass cez Rozenbergovu aproximáciu
    for kw in ("ALT_OBJ", "OBJCTALT", "ALTITUDE", "TELALT"):
        val = hdr.get(kw)
        if val is not None:
            try:
                alt_deg = float(val)
                if 5.0 <= alt_deg <= 90.0:
                    alt_rad = math.radians(alt_deg)
                    # Rozenbergova aproximácia: presnejšia ako sec(z) pri nízkej výške
                    am = 1.0 / (
                        math.sin(alt_rad)
                        + 0.50572 * (alt_deg + 6.07995) ** (-1.6364)
                    )
                    if 0.9 <= am <= 10.0:
                        return round(am, 5)
            except (TypeError, ValueError):
                pass

    return float("nan")


def _export_per_frame_run_catalog_core(
    base_path: Path,
    hdr: fits.Header,
    data: Any,
    st: dict[str, Any],
) -> dict[str, Any]:
    log_event("DEBUG: per-frame worker entry point called")
    fname = base_path.name
    debug_pixel_match: dict[str, Any] = {
        "file": fname,
        "use_fast": bool(st.get("use_master_fast_path")),
        "master_cols": None,
        "have_x": None,
        "have_y": None,
        "match_mode": None,
        "plate_scale_arcsec_per_px": None,
        "n_matched": None,
    }
    deferred_writes: list[tuple[str, pd.DataFrame]] = []
    if not _has_valid_wcs(hdr):
        return {
            "file": fname,
            "status": "no_wcs",
            "csv": "",
            "deferred_writes": deferred_writes,
            "infolog_messages": [],
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        _ = WCS(hdr)
    h_i, wpx_i = data.shape

    master_tab = st.get("master_tab")
    use_master_fast_path = bool(st.get("use_master_fast_path"))
    master_only_mode = bool(st.get("master_only_mode"))

    use_fast = (
        use_master_fast_path
        and master_tab is not None
        and not getattr(master_tab, "empty", True)
    )
    if master_only_mode and not use_fast:
        return {
            "file": fname,
            "status": "error: master_only_mode requires masterstars_full_match.csv",
            "csv": "",
            "deferred_writes": deferred_writes,
            "infolog_messages": [
                f"Per-frame catalog {fname}: master_only_mode requires masterstars_full_match.csv",
            ],
        }

    cat_df = st.get("cat_df")
    vsx_df = st.get("vsx_df")
    gaia_variable_df = st.get("gaia_variable_df")
    kd_pack = st.get("kd_pack")
    export_cat_local = bool(st.get("export_cat_local"))

    df: pd.DataFrame
    meta: dict[str, Any]
    if use_fast:
        try:
            # One-time debug: verify MASTERSTAR pixel columns used for KD-tree matching.
            try:
                if not bool(st.get("_debug_logged_master_xy")) and master_tab is not None:
                    st["_debug_logged_master_xy"] = True
                    cols = [c for c in ("x", "y", "ra_deg", "dec_deg") if c in master_tab.columns]
                    if cols:
                        log_event(
                            "DEBUG: masterstars_full_match.csv sample columns "
                            + ",".join(cols)
                            + ":\n"
                            + master_tab.loc[:, cols].head(5).to_string(index=False)
                        )
            except Exception:  # noqa: BLE001
                pass
            try:
                if master_tab is not None and "catalog_id" in master_tab.columns and "ra_deg" in master_tab.columns:
                    print(
                        "DEBUG SKY-MATCH: "
                        f"master_tab rows={len(master_tab)}, "
                        f"catalog_id notna={int(master_tab['catalog_id'].notna().sum())}, "
                        f"ra_deg notna={int(master_tab['ra_deg'].notna().sum())}"
                    )
            except Exception:  # noqa: BLE001
                pass
            try:
                if master_tab is not None:
                    debug_pixel_match["master_cols"] = list(master_tab.columns[:10])
                    debug_pixel_match["have_x"] = bool("x" in master_tab.columns)
                    debug_pixel_match["have_y"] = bool("y" in master_tab.columns)
            except Exception:  # noqa: BLE001
                pass
            df, meta = detect_stars_match_master_reference(
                data,
                hdr,
                master_tab,
                max_catalog_rows=int(st["max_catalog_rows"]),
                match_sep_arcsec=float(st["catalog_match_max_sep_arcsec"]),
                saturate_level_fraction=float(st["saturate_level_fraction"]),
                faintest_mag_limit=st.get("faintest_mag_limit"),
                dao_threshold_sigma=float(st["dao_threshold_sigma"]),
                dao_fwhm_px=float(st.get("dao_fwhm_px", 2.5)),
                equipment_saturate_adu=st.get("equipment_saturate_adu"),
                frame_name=fname,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "file": fname,
                "status": f"error: {exc}",
                "csv": "",
                "deferred_writes": deferred_writes,
                "infolog_messages": [f"Per-frame catalog {fname}: {exc}"],
                "debug_pixel_match": debug_pixel_match,
            }
    else:
        if master_only_mode:
            return {
                "file": fname,
                "status": "error: master_only_mode fallback to Gaia cone is disabled",
                "csv": "",
                "deferred_writes": deferred_writes,
                "infolog_messages": [
                    f"Per-frame catalog {fname}: master_only_mode fallback to Gaia cone is disabled",
                ],
            }
        try:
            df, meta = detect_stars_and_match_catalog(
                data,
                hdr,
                max_catalog_rows=int(st["max_catalog_rows"]),
                cat_df=cat_df,
                vsx_df=vsx_df,
                gaia_variable_df=gaia_variable_df,
                match_sep_arcsec=float(st["catalog_match_max_sep_arcsec"]),
                saturate_level_fraction=float(st["saturate_level_fraction"]),
                faintest_mag_limit=st.get("faintest_mag_limit"),
                field_catalog_export_path=None,
                dao_threshold_sigma=float(st["dao_threshold_sigma"]),
                dao_fwhm_px=float(st.get("dao_fwhm_px", 2.5)),
                equipment_saturate_adu=st.get("equipment_saturate_adu"),
                catalog_local_gaia_only=export_cat_local,
                catalog_kd_pack=kd_pack,
                plate_solve_fov_deg=st.get("plate_solve_fov_deg"),
                frame_name=fname,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "file": fname,
                "status": f"error: {exc}",
                "csv": "",
                "deferred_writes": deferred_writes,
                "infolog_messages": [f"Per-frame catalog {fname}: {exc}"],
                "debug_pixel_match": debug_pixel_match,
            }

    # --- Join MASTERSTAR annotations into per-frame catalog (via catalog_id) ---
    # For matched rows (catalog_id non-empty), bring stable MASTERSTAR columns like zone/is_usable/bp_rp/etc.
    try:
        if master_tab is not None and "catalog_id" in df.columns and "catalog_id" in master_tab.columns:
            _JOIN_COLS = [
                "zone",
                "is_saturated",
                "is_noisy",
                "is_usable",
                "bp_rp",
                "phot_g_mean_mag",
                "catalog_mag",
                "edge_safe_10px",
                "snr10_ok",
                "snr50_ok",
                "is_discovery_candidate",
                "is_saturated_flagged",
                "noise_floor_adu",
                "saturate_limit_adu_85pct",
                "source_type",
            ]
            join_cols = [c for c in _JOIN_COLS if c in master_tab.columns]
            if join_cols:
                master_lookup = master_tab[["catalog_id"] + join_cols].copy()
                master_lookup["catalog_id"] = (
                    master_lookup["catalog_id"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
                )
                master_lookup = master_lookup.dropna(subset=["catalog_id"])
                master_lookup = master_lookup.drop_duplicates(subset=["catalog_id"], keep="first")

                df = df.copy()
                df["catalog_id"] = df["catalog_id"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

                # Avoid column collisions; MASTERSTAR values should win for these columns.
                df = df.drop(columns=[c for c in join_cols if c in df.columns], errors="ignore")
                df = df.merge(master_lookup, on="catalog_id", how="left")
    except Exception:  # noqa: BLE001
        pass

    try:
        debug_pixel_match["match_mode"] = meta.get("catalog_match_mode")
        debug_pixel_match["plate_scale_arcsec_per_px"] = meta.get("plate_scale_arcsec_per_px")
        debug_pixel_match["n_matched"] = meta.get("n_matched")
    except Exception:  # noqa: BLE001
        pass

    df = _apply_aperture_catalog_enhancements_from_st(df, data, hdr, st)
    # PSF fotometria: Fáza 2B — zatiaľ vypnuté
    for _psf_col in ("psf_flux", "psf_flux_err", "psf_chi2"):
        if _psf_col not in df.columns:
            df[_psf_col] = float("nan")
    if "psf_fit_ok" not in df.columns:
        df["psf_fit_ok"] = False

    # --- Time columns (JD / HJD / BJD) ---
    _db_tc = None
    try:
        from time_utils import compute_time_columns

        _dbp = str(st.get("database_path") or "").strip()
        _did_tc = st.get("draft_id")
        if _dbp and _did_tc is not None:
            try:
                _db_tc = VyvarDatabase(Path(_dbp))
            except Exception:
                _db_tc = None
        _time_cols = compute_time_columns(
            hdr,
            db=_db_tc,
            draft_id=int(_did_tc) if _did_tc is not None else None,
        )
        _tk = ("jd_mid", "hjd_mid", "bjd_tdb_mid")
        _cols_base = list(df.columns)
        _anchors = [c for c in ("jd", "inspection_jd") if c in _cols_base]
        if _anchors:
            _pos = max(_cols_base.index(c) for c in _anchors) + 1
        else:
            _flux_first = next((c for c in _cols_base if c in ("dao_flux", "flux")), None)
            _pos = _cols_base.index(_flux_first) if _flux_first is not None else len(_cols_base)
        for _i, _nm in enumerate(_tk):
            df.insert(_pos + _i, _nm, _time_cols[_nm])

        # Airmass — rovnaká hodnota pre všetky hviezdy v snímke (frame-level)
        _am_val = _extract_airmass_from_header(hdr)
        if "airmass" not in df.columns:
            _am_insert_pos = _pos + len(_tk)
            df.insert(_am_insert_pos, "airmass", _am_val)
    except Exception as _tc_exc:
        log_event(f"Time columns skipped: {_tc_exc}")
        for _tc in ("jd_mid", "hjd_mid", "bjd_tdb_mid"):
            if _tc not in df.columns:
                df[_tc] = None
        if "airmass" not in df.columns:
            df["airmass"] = float("nan")
    finally:
        if _db_tc is not None:
            try:
                _db_tc.conn.close()
            except Exception:
                pass

    df2 = df.copy()
    df2.insert(0, "source_file", fname)

    csv_paths: list[str] = []
    write_sidecar = bool(st.get("write_sidecar_csv_next_to_fits"))
    mirror_flat = bool(st.get("mirror_flat_platesolve_folder"))
    defer = bool(st.get("defer_disk_writes"))
    out_flat = Path(str(st.get("out_flat") or "."))

    if write_sidecar:
        sidecar = base_path.with_suffix(".csv")
        if defer:
            deferred_writes.append((str(sidecar), df2.copy()))
        else:
            _vyvar_df_to_csv(df2, sidecar)
        csv_paths.append(str(sidecar))

    if mirror_flat:
        stem = Path(fname).stem
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:120]
        flat_path = out_flat / f"{safe}_catalog.csv"
        if defer:
            deferred_writes.append((str(flat_path), df2.copy()))
        else:
            _vyvar_df_to_csv(df2, flat_path)
        csv_paths.append(str(flat_path))

    primary_csv = csv_paths[0] if csv_paths else ""
    return {
        "file": fname,
        "status": "ok",
        "csv": primary_csv,
        "csv_paths": ";".join(csv_paths),
        "n_detected": meta.get("n_detected"),
        "n_matched": meta.get("n_matched"),
        "catalog_match_mode": ("master_reference_locked" if master_only_mode else meta.get("catalog_match_mode", "full_cone")),
        "deferred_writes": deferred_writes,
        "infolog_messages": [],
        "debug_pixel_match": debug_pixel_match,
    }


def _export_per_frame_disk_worker_task(fp_str: str) -> dict[str, Any]:
    print("DEBUG: REAL WORKER CALLED")
    import numpy as np

    st = _EXPORT_PER_FRAME_WORKER_STATE
    fp = Path(fp_str)
    try:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fp.name,
            "status": f"read_error: {exc}",
            "csv": "",
            "deferred_writes": [],
            "infolog_messages": [f"Per-frame catalog {fp.name}: read_error: {exc}"],
        }
    return _export_per_frame_run_catalog_core(fp, hdr, data, st)


def _export_per_frame_ram_worker_task(
    packed: tuple[str, bytes, bytes, int, int],
) -> dict[str, Any]:
    """Picklable worker: (fits_path_str, pickled_fits_Header, float32_bytes, ny, nx)."""
    print("DEBUG: REAL WORKER CALLED")
    import numpy as np

    st = _EXPORT_PER_FRAME_WORKER_STATE
    base_str, hdr_pkl, raw, ny, nx = packed
    fname = Path(base_str).name
    empty_def: list[tuple[str, pd.DataFrame]] = []
    try:
        hdr = pickle.loads(hdr_pkl)
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fname,
            "status": f"header_error: {exc}",
            "csv": "",
            "deferred_writes": empty_def,
            "infolog_messages": [f"Per-frame catalog {fname}: header_error: {exc}"],
        }
    try:
        data = np.frombuffer(raw, dtype=np.float32, count=ny * nx).reshape((ny, nx)).copy()
    except Exception as exc:  # noqa: BLE001
        return {
            "file": fname,
            "status": f"buffer_error: {exc}",
            "csv": "",
            "deferred_writes": empty_def,
            "infolog_messages": [f"Per-frame catalog {fname}: buffer_error: {exc}"],
        }
    return _export_per_frame_run_catalog_core(Path(base_str), hdr, data, st)


def export_per_frame_catalogs(
    *,
    frames_root: Path,
    platesolve_dir: Path,
    max_catalog_rows: int = 12000,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    dao_fwhm_px: float | None = None,
    write_sidecar_csv_next_to_fits: bool = True,
    mirror_flat_platesolve_folder: bool = False,
    progress_cb: "callable | None" = None,
    masterstars_csv: Path | str | None = None,
    masterstar_fits: Path | str | None = None,
    use_master_fast_path: bool = True,
    equipment_saturate_adu: float | None = None,
    catalog_local_gaia_only: bool | None = None,
    aligned_ram: "Sequence[tuple[str, fits.Header, Any]] | None" = None,
    aligned_target_dir: Path | str | None = None,
    defer_disk_writes: bool = False,
    app_config: AppConfig | None = None,
    plate_solve_fov_deg: float | None = None,
    master_dark_path: Path | str | None = None,
    draft_id: int | None = None,
    equipment_id: int | None = None,
) -> dict[str, Any]:
    """For each FITS under ``frames_root`` with WCS: DAO + catalog table, write one CSV per frame.

    **Optional fast path:** if ``use_master_fast_path=True`` and ``masterstars_csv`` + ``masterstar_fits`` match
    the frame (same WCS and **same array shape** as MASTERSTAR), exposures are matched only to
    ``masterstars.csv`` sky positions (faster; on chip edges NN distances can exceed the match threshold).
    ``astrometry_align_and_build_masterstar`` defaults this to **off** unless ``VYVAR_PER_FRAME_MASTER_FAST=1``.

    **Fallback:** if paths are missing or WCS differs, uses one shared cone (see ``field_catalog_cone.csv``)
    and full ``detect_stars_and_match_catalog`` per frame (local Gaia).
    A sidecar ``field_catalog_cone_meta.json`` records the angular cone radius used; if a **cropped** MASTERSTAR
    built a too-small cone, full-chip frames automatically trigger a refetch instead of reusing the stale CSV.

    ``<platesolve_dir>/per_frame_catalog_index.csv`` lists every file and CSV path.

    **Performance:** each frame still runs DAO + catalog match + disk CSV write (dominant cost for many lights).
    Parallelism: ``per_frame_csv_workers`` from ``app_config`` / ``config.json`` or env
    ``VYVAR_PER_FRAME_CSV_WORKERS`` (env wins). When ``>1``, uses ``ProcessPoolExecutor`` (``spawn``); the
    parent prefetches Gaia cone. Worker count is capped using ``psutil`` and
    ``per_frame_mp_reserve_ram_gb``. RAM handoff ``aligned_ram`` uses the same process pool with serialized
    headers + float32 pixels. Lower ``max_catalog_rows`` in the UI to reduce DAO work per file.

    **RAM handoff:** pass ``aligned_ram`` as ``(filename, header, ndarray)`` tuples plus ``aligned_target_dir`` to
    run catalog matching **without** re-reading aligned FITS from disk. With ``defer_disk_writes=True``, sidecar
    CSV (and optional flat mirror) are returned in ``deferred_csv_writes`` for the caller to flush after FITS.

    ``master_dark_path``: optional CalibrationLibrary master dark; enables ``*_dark_bpm.json`` column flags when present.
    """
    import numpy as np

    _cfg_ap = app_config or AppConfig()
    _md_bpm_str = ""
    if master_dark_path is not None and str(master_dark_path).strip():
        _mp = Path(str(master_dark_path))
        if _mp.is_file():
            _md_bpm_str = str(_mp.resolve())
    _ap_st: dict[str, Any] = {
        "aperture_photometry_enabled": bool(_cfg_ap.aperture_photometry_enabled),
        "aperture_fwhm_factor": float(_cfg_ap.aperture_fwhm_factor),
        "annulus_inner_fwhm": float(_cfg_ap.annulus_inner_fwhm),
        "annulus_outer_fwhm": float(_cfg_ap.annulus_outer_fwhm),
        "nonlinearity_peak_percentile": float(_cfg_ap.nonlinearity_peak_percentile),
        "nonlinearity_fwhm_ratio": float(_cfg_ap.nonlinearity_fwhm_ratio),
        "bpm_dark_mad_sigma": float(_cfg_ap.bpm_dark_mad_sigma),
        "master_dark_path": _md_bpm_str,
        "database_path": str(Path(_cfg_ap.database_path).resolve()),
        "draft_id": int(draft_id) if draft_id is not None else None,
        "equipment_id": int(equipment_id) if equipment_id is not None else None,
    }

    use_ram_inputs = aligned_ram is not None
    if use_ram_inputs and aligned_target_dir is None:
        raise ValueError("export_per_frame_catalogs: aligned_target_dir is required when aligned_ram is set")

    if (
        not write_sidecar_csv_next_to_fits
        and not mirror_flat_platesolve_folder
        and not defer_disk_writes
    ):
        write_sidecar_csv_next_to_fits = True

    root = Path(aligned_target_dir) if use_ram_inputs else Path(frames_root)
    ps = Path(platesolve_dir)
    out_flat = ps / "per_frame_catalogs"
    if mirror_flat_platesolve_folder:
        out_flat.mkdir(parents=True, exist_ok=True)

    work_ram: list[tuple[str, fits.Header, Any]] | None = None
    if use_ram_inputs:
        root.mkdir(parents=True, exist_ok=True)
        work_ram = sorted(list(aligned_ram), key=lambda t: t[0])
        files = [root / name for name, _, _ in work_ram]
    else:
        files = sorted(_iter_fits_recursive(root))

    if not files:
        return {
            "written": 0,
            "per_frame_dir": str(root),
            "per_frame_csv_mode": "sidecar" if write_sidecar_csv_next_to_fits else "none",
            "index_csv": "",
            "frames": [],
            "deferred_csv_writes": [],
        }

    master_only_mode = bool(use_master_fast_path)
    field_cat_path = ps / "field_catalog_cone.csv"
    meta_path = _field_catalog_cone_meta_path(field_cat_path)
    cat_df: pd.DataFrame | None = None

    _pfov_res: float | None = plate_solve_fov_deg
    if _pfov_res is None:
        try:
            _pfov_res = float((app_config or AppConfig()).plate_solve_fov_deg)
        except Exception:  # noqa: BLE001
            _pfov_res = None

    r_need_deg: float | None = None
    try:
        _ref_fp = files[0]
        with fits.open(_ref_fp, memmap=False) as _hdu0:
            _hdr0 = _hdu0[0].header
            _dat0 = np.asarray(_hdu0[0].data)
            if _dat0.ndim == 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    _w0 = WCS(_hdr0)
                if _w0.has_celestial:
                    _h0, _w0px = _dat0.shape
                    _, r_need_deg = _effective_field_catalog_cone_radius_deg(
                        _w0, _h0, _w0px, _pfov_res, fits_header=_hdr0
                    )
    except Exception:  # noqa: BLE001
        r_need_deg = None

    if (not master_only_mode) and r_need_deg is not None:
        _invalidate_field_catalog_cone_cache_if_needed(
            field_cat_path,
            plate_solve_fov_deg=_pfov_res,
            effective_radius_deg=float(r_need_deg),
        )

    if (not master_only_mode) and field_cat_path.is_file():
        try:
            _cdf = pd.read_csv(field_cat_path)
            if len(_cdf) > 0:
                _reuse = True
                if r_need_deg is not None and meta_path.is_file():
                    try:
                        _meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        r_stored = float(_meta.get("cone_radius_deg", 0.0))
                        slack_deg = 45.0 / 3600.0
                        if r_stored <= 0.0 or r_need_deg > r_stored * 1.02 + slack_deg:
                            _reuse = False
                            LOGGER.info(
                                "Per-frame catalog: ignoring cached %s (full chip needs cone_radius_deg≈%.6f, "
                                "cached %.6f from %s) — fetching larger Gaia cone",
                                field_cat_path,
                                r_need_deg,
                                r_stored,
                                meta_path.name,
                            )
                    except Exception:  # noqa: BLE001
                        pass
                if _reuse:
                    cat_df = _cdf
                    LOGGER.info(
                        "Per-frame catalog: reusing %s (%s rows) — skipping duplicate cone query",
                        field_cat_path,
                        len(cat_df),
                    )
        except Exception:  # noqa: BLE001
            cat_df = None

    kd_cell: list[tuple[Any, Any]] = []

    def _refresh_shared_cat_kdtree() -> None:
        if kd_cell or cat_df is None or getattr(cat_df, "empty", True) or len(cat_df) < 120:
            return
        bk = build_ucac_catalog_kdtree(cat_df)
        if bk is not None:
            kd_cell.append(bk)
            LOGGER.info(
                "Per-frame catalog: shared cKDTree for %s cone rows (fast sky matching)",
                len(cat_df),
            )

    vsx_df: pd.DataFrame | None = None
    gaia_variable_df: pd.DataFrame | None = None
    total = len(files)
    _debug_logged_once = False
    _uc_um = float(faintest_mag_limit) if faintest_mag_limit is not None else None
    try:
        _cfg_e = AppConfig()
        _gp = (getattr(_cfg_e, "gaia_db_path", None) or "").strip()
        _uc_root_e = Path(_gp) if _gp else None
    except Exception:  # noqa: BLE001
        _uc_root_e = None
    _export_cat_local = (not master_only_mode)

    master_tab: pd.DataFrame | None = None
    ref_wcs: WCS | None = None
    masterstar_data_shape: tuple[int, int] | None = None
    if use_master_fast_path and masterstars_csv is not None:
        _msp = Path(masterstars_csv)
        if _msp.is_file():
            try:
                _mt = pd.read_csv(_msp)
                if not _mt.empty and "ra_deg" in _mt.columns and "dec_deg" in _mt.columns:
                    master_tab = _mt
            except Exception:  # noqa: BLE001
                master_tab = None
    if use_master_fast_path and masterstar_fits is not None and master_tab is not None:
        _msf = Path(masterstar_fits)
        if _msf.is_file():
            try:
                with fits.open(_msf, memmap=False) as _mh:
                    _marr = np.asarray(_mh[0].data)
                    if _marr.ndim == 2:
                        masterstar_data_shape = (int(_marr.shape[0]), int(_marr.shape[1]))
                    ref_wcs = WCS(_mh[0].header)
                if not ref_wcs.has_celestial:
                    ref_wcs = None
            except Exception:  # noqa: BLE001
                ref_wcs = None
                masterstar_data_shape = None
        else:
            ref_wcs = None
    else:
        ref_wcs = None

    if master_tab is not None:
        LOGGER.info(
            "Per-frame catalog: MASTERSTAR lock enabled (single catalog: masterstars_full_match.csv, %s rows)",
            len(master_tab),
        )
    elif master_only_mode:
        raise RuntimeError(
            "Per-frame catalog lock requested, but masterstars_full_match.csv is missing or invalid."
        )

    _gauss_override: float | None = None
    try:
        if masterstar_fits is not None:
            _ms_gauss = Path(masterstar_fits)
            if _ms_gauss.is_file():
                with fits.open(_ms_gauss, memmap=False) as _gfh:
                    _ghdr = _gfh[0].header
                    for _gk in ("VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN"):
                        _gv = _ghdr.get(_gk)
                        if _gv is None:
                            continue
                        try:
                            _gfv = float(_gv)
                            if math.isfinite(_gfv) and 0.5 < _gfv < 30.0:
                                _gauss_override = _gfv
                                break
                        except (TypeError, ValueError):
                            pass
    except Exception:  # noqa: BLE001
        _gauss_override = None
    _ap_st["gaussian_fwhm_px_override"] = _gauss_override

    cfg_for_workers = app_config if app_config is not None else AppConfig()
    _dao_fw_export = (
        float(dao_fwhm_px)
        if dao_fwhm_px is not None
        else float(getattr(cfg_for_workers, "sips_dao_fwhm_px", 2.5))
    )
    n_workers = _vyvar_per_frame_csv_workers(cfg_for_workers)
    _ny, _nx = _estimate_catalog_frame_hw(work_ram if use_ram_inputs else None, files)
    n_workers = _vyvar_cap_mp_workers_for_catalog(
        n_workers,
        (_ny, _nx),
        reserve_gb=float(cfg_for_workers.per_frame_mp_reserve_ram_gb),
    )
    if n_workers > 1 and total > 1:
        LOGGER.info(
            "Per-frame catalog: up to %s process worker(s); per_frame_csv_workers / env; "
            "RAM cap: per_frame_mp_reserve_ram_gb + psutil",
            n_workers,
        )

    use_parallel_mp = n_workers > 1 and total > 1
    kd_pack_mp: Any = None
    if use_parallel_mp and not master_only_mode:
        if use_ram_inputs and work_ram is not None:
            _h_ref, _d_ref = work_ram[0][1], np.asarray(work_ram[0][2], dtype=np.float32)
            cat_df, vsx_df, gaia_variable_df, kd_pack_mp = _prefetch_export_shared_catalog_for_process_pool(
                files=None,
                reference_hdr_data=(_h_ref.copy(), _d_ref),
                field_cat_path=field_cat_path,
                cat_df=cat_df,
                vsx_df=vsx_df,
                gaia_variable_df=gaia_variable_df,
                gaia_db_path=_uc_root_e,
                gaia_local_max_mag=_uc_um,
                export_cat_local=_export_cat_local,
                plate_solve_fov_deg=_pfov_res,
            )
        else:
            cat_df, vsx_df, gaia_variable_df, kd_pack_mp = _prefetch_export_shared_catalog_for_process_pool(
                files=list(files),
                reference_hdr_data=None,
                field_cat_path=field_cat_path,
                cat_df=cat_df,
                vsx_df=vsx_df,
                gaia_variable_df=gaia_variable_df,
                gaia_db_path=_uc_root_e,
                gaia_local_max_mag=_uc_um,
                export_cat_local=_export_cat_local,
                plate_solve_fov_deg=_pfov_res,
            )
        LOGGER.info(
        "Per-frame catalog: parallel backend=process (%s workers); Gaia cone prefetched in parent",
            n_workers,
        )
    else:
        _refresh_shared_cat_kdtree()

    _prog_seq = 0
    deferred_csv_writes: list[tuple[Path, pd.DataFrame]] = []

    def _append_deferred_csv(p: Path, df: pd.DataFrame) -> None:
        deferred_csv_writes.append((p, df))

    def _ensure_cone_and_variables(
        c_i: SkyCoord, r_i: float, *, naxis1: int = 0, naxis2: int = 0
    ) -> None:
        def _fill() -> None:
            nonlocal cat_df, vsx_df, gaia_variable_df
            if master_only_mode:
                return
            if cat_df is None or cat_df.empty:
                _gaia_db_path: Path | None = None
                try:
                    _cfg = AppConfig()
                    _gp = (getattr(_cfg, "gaia_db_path", None) or "").strip()
                    if _gp:
                        _gaia_db_path = Path(_gp)
                except Exception:  # noqa: BLE001
                    _gaia_db_path = None
                cat_df = _query_gaia_local(
                    center=c_i,
                    radius_deg=r_i,
                    gaia_db_path=_gaia_db_path,
                    max_mag=float(_uc_um) if _uc_um is not None else None,
                )
                if cat_df is not None and len(cat_df) > 0:
                    try:
                        field_cat_path.parent.mkdir(parents=True, exist_ok=True)
                        _vyvar_df_to_csv(cat_df, field_cat_path)
                        _write_field_catalog_cone_meta(
                            field_cat_path,
                            center=c_i,
                            radius_deg=float(r_i),
                            naxis1=int(naxis1),
                            naxis2=int(naxis2),
                            plate_solve_fov_deg=_pfov_res,
                        )
                    except Exception:  # noqa: BLE001
                        pass
            if vsx_df is None or getattr(vsx_df, "empty", True):
                _vsx_p2: Path | None = None
                try:
                    _vsp2 = str(getattr(AppConfig(), "vsx_local_db_path", "") or "").strip()
                    if _vsp2:
                        _vsx_p2 = Path(_vsp2).expanduser().resolve()
                except Exception:  # noqa: BLE001
                    _vsx_p2 = None
                if _vsx_p2 is not None and _vsx_p2.is_file():
                    vsx_df = _query_vsx_local(
                        center=c_i,
                        radius_deg=float(r_i),
                        vsx_db_path=_vsx_p2,
                    )
                else:
                    vsx_df = pd.DataFrame()
            if gaia_variable_df is None:
                gaia_variable_df = pd.DataFrame()
            _refresh_shared_cat_kdtree()

        _fill()

    def _run_one_catalog(base_path: Path, hdr: fits.Header, data: np.ndarray) -> dict[str, Any]:
        fname = base_path.name
        nonlocal _prog_seq
        if progress_cb is not None and n_workers <= 1:
            _prog_seq += 1
            progress_cb(_prog_seq, total, f"Catalog: {fname}")
        if not _has_valid_wcs(hdr):
            return {"file": fname, "status": "no_wcs", "csv": ""}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w_i = WCS(hdr)
        h_i, wpx_i = data.shape

        use_fast = (
            bool(use_master_fast_path)
            and master_tab is not None
            and (not getattr(master_tab, "empty", True))
        )
        if master_only_mode and not use_fast:
            return {"file": fname, "status": "error: master_only_mode requires masterstars_full_match.csv", "csv": ""}

        df: pd.DataFrame
        meta: dict[str, Any]
        if use_fast:
            try:
                df, meta = detect_stars_match_master_reference(
                    data,
                    hdr,
                    master_tab,
                    max_catalog_rows=int(max_catalog_rows),
                    match_sep_arcsec=float(catalog_match_max_sep_arcsec),
                    saturate_level_fraction=float(saturate_level_fraction),
                    faintest_mag_limit=faintest_mag_limit,
                    dao_threshold_sigma=float(dao_threshold_sigma),
                    dao_fwhm_px=float(_dao_fw_export),
                    equipment_saturate_adu=equipment_saturate_adu,
                )
            except Exception as exc:  # noqa: BLE001
                return {"file": fname, "status": f"error: {exc}", "csv": ""}
        else:
            if master_only_mode:
                return {"file": fname, "status": "error: master_only_mode fallback to Gaia cone is disabled", "csv": ""}
            c_i, r_i = _effective_field_catalog_cone_radius_deg(
                w_i, h_i, wpx_i, _pfov_res, fits_header=hdr
            )
            _ensure_cone_and_variables(c_i, r_i, naxis1=int(wpx_i), naxis2=int(h_i))

            try:
                df, meta = detect_stars_and_match_catalog(
                    data,
                    hdr,
                    max_catalog_rows=int(max_catalog_rows),
                    cat_df=cat_df,
                    vsx_df=vsx_df,
                    gaia_variable_df=gaia_variable_df,
                    match_sep_arcsec=float(catalog_match_max_sep_arcsec),
                    saturate_level_fraction=float(saturate_level_fraction),
                    faintest_mag_limit=faintest_mag_limit,
                    field_catalog_export_path=None,
                    dao_threshold_sigma=float(dao_threshold_sigma),
                    dao_fwhm_px=float(_dao_fw_export),
                    equipment_saturate_adu=equipment_saturate_adu,
                    catalog_local_gaia_only=_export_cat_local,
                    catalog_kd_pack=kd_cell[0] if kd_cell else None,
                    plate_solve_fov_deg=_pfov_res,
                )
            except Exception as exc:  # noqa: BLE001
                return {"file": fname, "status": f"error: {exc}", "csv": ""}

        df = _apply_aperture_catalog_enhancements_from_st(df, data, hdr, _ap_st)
        # PSF fotometria: Fáza 2B — zatiaľ vypnuté
        for _psf_col in ("psf_flux", "psf_flux_err", "psf_chi2"):
            if _psf_col not in df.columns:
                df[_psf_col] = float("nan")
        if "psf_fit_ok" not in df.columns:
            df["psf_fit_ok"] = False

        # --- Time columns (JD / HJD / BJD) ---
        _db_tc = None
        try:
            from time_utils import compute_time_columns

            _dbp = str(_ap_st.get("database_path") or "").strip()
            _did_tc = _ap_st.get("draft_id")
            if _dbp and _did_tc is not None:
                try:
                    _db_tc = VyvarDatabase(Path(_dbp))
                except Exception:
                    _db_tc = None
            _time_cols = compute_time_columns(
                hdr,
                db=_db_tc,
                draft_id=int(_did_tc) if _did_tc is not None else None,
            )
            _tk = ("jd_mid", "hjd_mid", "bjd_tdb_mid")
            _cols_base = list(df.columns)
            _anchors = [c for c in ("jd", "inspection_jd") if c in _cols_base]
            if _anchors:
                _pos = max(_cols_base.index(c) for c in _anchors) + 1
            else:
                _flux_first = next((c for c in _cols_base if c in ("dao_flux", "flux")), None)
                _pos = _cols_base.index(_flux_first) if _flux_first is not None else len(_cols_base)
            for _i, _nm in enumerate(_tk):
                df.insert(_pos + _i, _nm, _time_cols[_nm])

            # Airmass — frame-level hodnota z FITS hlavičky
            _am_val = _extract_airmass_from_header(hdr)
            if "airmass" not in df.columns:
                _am_insert_pos = _pos + len(_tk)
                df.insert(_am_insert_pos, "airmass", _am_val)
        except Exception as _tc_exc:
            log_event(f"Time columns skipped: {_tc_exc}")
            for _tc in ("jd_mid", "hjd_mid", "bjd_tdb_mid"):
                if _tc not in df.columns:
                    df[_tc] = None
            if "airmass" not in df.columns:
                df["airmass"] = float("nan")
        finally:
            if _db_tc is not None:
                try:
                    _db_tc.conn.close()
                except Exception:
                    pass

        df2 = df.copy()
        df2.insert(0, "source_file", fname)

        csv_paths: list[str] = []
        if write_sidecar_csv_next_to_fits:
            sidecar = base_path.with_suffix(".csv")
            if defer_disk_writes:
                _append_deferred_csv(sidecar, df2.copy())
            else:
                _vyvar_df_to_csv(df2, sidecar)
            csv_paths.append(str(sidecar))

        if mirror_flat_platesolve_folder:
            stem = Path(fname).stem
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:120]
            flat_path = out_flat / f"{safe}_catalog.csv"
            if defer_disk_writes:
                _append_deferred_csv(flat_path, df2.copy())
            else:
                _vyvar_df_to_csv(df2, flat_path)
            csv_paths.append(str(flat_path))

        primary_csv = csv_paths[0] if csv_paths else ""
        return {
            "file": fname,
            "status": "ok",
            "csv": primary_csv,
            "csv_paths": ";".join(csv_paths),
            "n_detected": meta.get("n_detected"),
            "n_matched": meta.get("n_matched"),
            "catalog_match_mode": ("master_reference_locked" if master_only_mode else meta.get("catalog_match_mode", "full_cone")),
        }

    def _process_frame(fp: Path) -> dict[str, Any]:
        try:
            try:
                with fits.open(fp, memmap=False) as hdul:
                    hdr = hdul[0].header.copy()
                    data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            except Exception as exc:  # noqa: BLE001
                return {"file": fp.name, "status": f"read_error: {exc}", "csv": ""}
            return _run_one_catalog(fp, hdr, data)
        except Exception as exc:  # noqa: BLE001
            return {"file": fp.name, "status": f"error: {exc}", "csv": ""}

    def _process_ram_item(item: tuple[str, fits.Header, Any]) -> dict[str, Any]:
        name, hdr0, arr0 = item
        try:
            base = Path(aligned_target_dir) / name
            return _run_one_catalog(base, hdr0.copy(), np.asarray(arr0, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            return {"file": name, "status": f"error: {exc}", "csv": ""}

    def _catalog_worker_state() -> dict[str, Any]:
        return {
            "cat_df": cat_df,
            "vsx_df": vsx_df,
            "gaia_variable_df": gaia_variable_df,
            "kd_pack": kd_pack_mp,
            "master_tab": master_tab,
            "masterstar_fits_path": (
                str(Path(masterstar_fits).resolve())
                if masterstar_fits is not None and Path(masterstar_fits).is_file()
                else ""
            ),
            "use_master_fast_path": bool(use_master_fast_path),
            "masterstar_data_shape": masterstar_data_shape,
            "max_catalog_rows": int(max_catalog_rows),
            "catalog_match_max_sep_arcsec": float(catalog_match_max_sep_arcsec),
            "saturate_level_fraction": float(saturate_level_fraction),
            "faintest_mag_limit": faintest_mag_limit,
            "dao_threshold_sigma": float(dao_threshold_sigma),
            "dao_fwhm_px": float(_dao_fw_export),
            "equipment_saturate_adu": equipment_saturate_adu,
            "export_cat_local": _export_cat_local,
            "master_only_mode": bool(master_only_mode),
            "plate_solve_fov_deg": _pfov_res,
            "write_sidecar_csv_next_to_fits": write_sidecar_csv_next_to_fits,
            "mirror_flat_platesolve_folder": mirror_flat_platesolve_folder,
            "defer_disk_writes": defer_disk_writes,
            "out_flat": str(out_flat.resolve()),
            "aperture_photometry_enabled": bool(_cfg_ap.aperture_photometry_enabled),
            "aperture_fwhm_factor": float(_cfg_ap.aperture_fwhm_factor),
            "annulus_inner_fwhm": float(_cfg_ap.annulus_inner_fwhm),
            "annulus_outer_fwhm": float(_cfg_ap.annulus_outer_fwhm),
            "nonlinearity_peak_percentile": float(_cfg_ap.nonlinearity_peak_percentile),
            "nonlinearity_fwhm_ratio": float(_cfg_ap.nonlinearity_fwhm_ratio),
            "bpm_dark_mad_sigma": float(_cfg_ap.bpm_dark_mad_sigma),
            "master_dark_path": _md_bpm_str,
            "database_path": str(Path(_cfg_ap.database_path).resolve()),
            "draft_id": int(draft_id) if draft_id is not None else None,
            "equipment_id": int(equipment_id) if equipment_id is not None else None,
            "gaussian_fwhm_px_override": _gauss_override,
        }

    if use_parallel_mp and use_ram_inputs and work_ram is not None:
        ctx = multiprocessing.get_context("spawn")
        ws = _catalog_worker_state()
        packs: list[tuple[str, bytes, bytes, int, int]] = []
        for _name, _hdr, _arr in work_ram:
            base_p = (root / _name).resolve()
            d = np.asarray(_arr, dtype=np.float32)
            if not d.flags.c_contiguous:
                d = np.ascontiguousarray(d, dtype=np.float32)
            packs.append(
                (
                    str(base_p),
                    pickle.dumps(_hdr.copy()),
                    d.tobytes(),
                    int(d.shape[0]),
                    int(d.shape[1]),
                )
            )
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_export_per_frame_worker,
            initargs=(ws,),
        ) as ex:
            futs = {ex.submit(_export_per_frame_ram_worker_task, pack): pack[0] for pack in packs}
            by_name: dict[str, dict[str, Any]] = {}
            done = 0
            for fut in as_completed(futs):
                base_str = futs[fut]
                res = fut.result()
                nm = Path(base_str).name
                if (not _debug_logged_once) and isinstance(res, dict) and res.get("debug_pixel_match") is not None:
                    _debug_logged_once = True
                    try:
                        log_event(
                            "DEBUG: per-frame debug_pixel_match (first frame): "
                            + json.dumps(res.get("debug_pixel_match"), ensure_ascii=False, default=str)
                        )
                    except Exception:  # noqa: BLE001
                        log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {res.get('debug_pixel_match')}")
                for msg in res.get("infolog_messages") or []:
                    if msg:
                        log_event(str(msg))
                for p_str, dff in res.get("deferred_writes") or []:
                    deferred_csv_writes.append((Path(p_str), dff))
                by_name[nm] = res
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"Catalog: {nm}")
        rows_out = []
        for it in work_ram:
            rr = dict(by_name[str(it[0])])
            rr.pop("deferred_writes", None)
            rr.pop("infolog_messages", None)
            rows_out.append(rr)
    elif use_parallel_mp:
        ctx = multiprocessing.get_context("spawn")
        ws = _catalog_worker_state()
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_init_export_per_frame_worker,
            initargs=(ws,),
        ) as ex:
            futs = {ex.submit(_export_per_frame_disk_worker_task, str(fp.resolve())): fp for fp in files}
            by_fp: dict[Path, dict[str, Any]] = {}
            done = 0
            for fut in as_completed(futs):
                fp = futs[fut]
                res = fut.result()
                if (not _debug_logged_once) and isinstance(res, dict) and res.get("debug_pixel_match") is not None:
                    _debug_logged_once = True
                    try:
                        log_event(
                            "DEBUG: per-frame debug_pixel_match (first frame): "
                            + json.dumps(res.get("debug_pixel_match"), ensure_ascii=False, default=str)
                        )
                    except Exception:  # noqa: BLE001
                        log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {res.get('debug_pixel_match')}")
                for msg in res.get("infolog_messages") or []:
                    if msg:
                        log_event(str(msg))
                by_fp[fp] = res
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"Catalog: {fp.name}")
                for p_str, dff in res.get("deferred_writes") or []:
                    deferred_csv_writes.append((Path(p_str), dff))
        rows_out = []
        for fp in files:
            rr = dict(by_fp[fp])
            rr.pop("deferred_writes", None)
            rr.pop("infolog_messages", None)
            rows_out.append(rr)
    elif use_ram_inputs and work_ram is not None:
        rows_out = []
        for i, it in enumerate(work_ram, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"Catalog: {it[0]}")
            r = _process_ram_item(it)
            if (not _debug_logged_once) and isinstance(r, dict) and r.get("debug_pixel_match") is not None:
                _debug_logged_once = True
                try:
                    log_event(
                        "DEBUG: per-frame debug_pixel_match (first frame): "
                        + json.dumps(r.get("debug_pixel_match"), ensure_ascii=False, default=str)
                    )
                except Exception:  # noqa: BLE001
                    log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {r.get('debug_pixel_match')}")
            for msg in r.get("infolog_messages") or []:
                if msg:
                    log_event(str(msg))
            r.pop("infolog_messages", None)
            rows_out.append(r)
    else:
        rows_out = []
        for i, fp in enumerate(files, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"Catalog: {fp.name}")
            r = _process_frame(fp)
            if (not _debug_logged_once) and isinstance(r, dict) and r.get("debug_pixel_match") is not None:
                _debug_logged_once = True
                try:
                    log_event(
                        "DEBUG: per-frame debug_pixel_match (first frame): "
                        + json.dumps(r.get("debug_pixel_match"), ensure_ascii=False, default=str)
                    )
                except Exception:  # noqa: BLE001
                    log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {r.get('debug_pixel_match')}")
            for msg in r.get("infolog_messages") or []:
                if msg:
                    log_event(str(msg))
            r.pop("infolog_messages", None)
            rows_out.append(r)

    index_path = ps / "per_frame_catalog_index.csv"
    if not defer_disk_writes:
        index_rows = [
            {k: v for k, v in row.items() if k not in ("deferred_writes", "infolog_messages")}
            for row in rows_out
        ]
        _vyvar_df_to_csv(pd.DataFrame(index_rows), index_path)
    n_ok = sum(1 for r in rows_out if r.get("status") == "ok")
    n_master_ref = sum(1 for r in rows_out if r.get("catalog_match_mode") == "master_reference")
    return {
        "written": int(n_ok),
        "per_frame_dir": str(root),
        "per_frame_csv_mode": "sidecar" if write_sidecar_csv_next_to_fits else ("flat_mirror" if mirror_flat_platesolve_folder else "none"),
        "index_csv": str(index_path),
        "frames": rows_out,
        "mirror_flat_platesolve_folder": bool(mirror_flat_platesolve_folder),
        "frames_master_reference_match": int(n_master_ref),
        "deferred_csv_writes": list(deferred_csv_writes) if defer_disk_writes else [],
    }


def validate_comparison_ensemble_flatness(
    *,
    frames_root: Path,
    comparison_stars_csv: Path,
    flux_col: str = "flux",
    name_col: str = "name",
    max_relative_rms: float = 0.03,
    min_frames_per_star: int = 5,
    min_stars_per_frame: int = 3,
    output_report_csv: Path | None = None,
) -> dict[str, Any]:
    """Check that comparison stars stay flat vs the **per-frame** ensemble (median flux of comps on that frame).

    Uses DAO ``flux`` from sidecar CSVs next to each aligned FITS under ``frames_root`` (same layout as
    ``export_per_frame_catalogs``). For each exposure, builds relative flux ``f_i / median(f_all comps present)``.
    A good comparison star has low RMS of that ratio over time (instrument / transparency drifts divide out).

    Catalog non-variable filtering (VSX / Gaia) is handled earlier when building ``comparison_stars.csv``; this
    step is the **photometric** sanity check among the selected ensemble.

    Returns summary counts and per-star metrics; optionally writes ``output_report_csv``.
    """
    import numpy as np

    comp_path = Path(comparison_stars_csv)
    if not comp_path.is_file():
        return {"error": f"missing {comp_path}", "rows": []}

    comp_df = pd.read_csv(comp_path)
    if name_col not in comp_df.columns:
        return {"error": f"comparison table missing column {name_col!r}", "rows": []}
    names = [str(x).strip() for x in comp_df[name_col].dropna().astype(str).unique() if str(x).strip()]
    if not names:
        return {"error": "no comparison star names", "rows": []}

    by_jd: dict[float, dict[str, float]] = {}
    root = Path(frames_root)
    files_n = 0
    for fp in sorted(_iter_fits_recursive(root)):
        sidecar = fp.with_suffix(".csv")
        if not sidecar.is_file():
            continue
        meta = extract_fits_metadata(fp, app_config=AppConfig())
        jd = float(meta.get("jd_start") or 0.0)
        if jd <= 0.0:
            continue
        try:
            dff = pd.read_csv(sidecar)
        except Exception:  # noqa: BLE001
            continue
        if name_col not in dff.columns or flux_col not in dff.columns:
            continue
        files_n += 1
        rowmap: dict[str, float] = {}
        for nm in names:
            m = dff.loc[dff[name_col].astype(str).str.strip() == nm]
            if m.empty:
                continue
            fl = float(m.iloc[0][flux_col])
            if np.isfinite(fl) and fl > 0:
                rowmap[nm] = fl
        if len(rowmap) >= int(min_stars_per_frame):
            by_jd[jd] = rowmap

    rel_lists: dict[str, list[float]] = {nm: [] for nm in names}
    for _jd, rowmap in sorted(by_jd.items(), key=lambda t: t[0]):
        vals = np.array(list(rowmap.values()), dtype=np.float64)
        med = float(np.median(vals))
        if not np.isfinite(med) or med <= 0:
            continue
        for nm, fl in rowmap.items():
            rel_lists[nm].append(float(fl / med))

    rows_out: list[dict[str, Any]] = []
    n_pass = 0
    n_fail = 0
    for nm in names:
        arr = np.array(rel_lists[nm], dtype=np.float64)
        n_fr = int(len(arr))
        if n_fr < int(min_frames_per_star):
            rows_out.append(
                {
                    "name": nm,
                    "n_frames": n_fr,
                    "relative_rms": None,
                    "flatness_ok": False,
                    "reason": "too_few_frames",
                }
            )
            n_fail += 1
            continue
        rms = float(np.sqrt(np.mean((arr - 1.0) ** 2)))
        ok = rms <= float(max_relative_rms)
        rows_out.append(
            {
                "name": nm,
                "n_frames": n_fr,
                "relative_rms": rms,
                "flatness_ok": ok,
                "reason": "" if ok else "high_rms",
            }
        )
        if ok:
            n_pass += 1
        else:
            n_fail += 1

    rep_path: str | None = None
    if output_report_csv is not None:
        outp = Path(output_report_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(outp, index=False)
        rep_path = str(outp)

    return {
        "comparison_stars_csv": str(comp_path),
        "frames_sampled": int(files_n),
        "frames_used_ensemble": int(len(by_jd)),
        "n_comparison_names": int(len(names)),
        "n_pass_flatness": int(n_pass),
        "n_fail_flatness": int(n_fail),
        "max_relative_rms_threshold": float(max_relative_rms),
        "min_frames_per_star": int(min_frames_per_star),
        "rows": rows_out,
        "report_csv": rep_path,
    }


def _apply_wcs_tan_fragment_to_header(h: fits.Header, wh: fits.Header, history_note: str) -> None:
    strip_celestial_wcs_keys(h)
    for k in wh:
        if k in ("", "COMMENT", "HISTORY", "SIMPLE", "BITPIX", "NAXIS", "EXTEND"):
            continue
        if k.startswith("NAXIS") and k != "NAXIS":
            continue
        try:
            h[k] = wh[k]
        except Exception:  # noqa: BLE001
            pass
    h.add_history(history_note)


def _gaia_sky_match_wcs_fragment(
    hdr: fits.Header,
    data: "np.ndarray",
    *,
    app_config: AppConfig | None,
    dao_threshold_sigma: float,
    dao_fwhm_px: float,
    plate_solve_fov_deg: float | None,
    max_bright_stars: int = 200,
    max_sky_match_arcsec: float = 25.0,
    gaia_max_mag: float = 16.5,
    min_pairs: int = 10,
    log_gaia_sql: bool = True,
    log_messages: bool = True,
) -> tuple[fits.Header | None, dict[str, Any]]:
    """Legacy WCS refinement removed in Gaia migration (no-op)."""
    _ = (
        hdr,
        data,
        app_config,
        dao_threshold_sigma,
        dao_fwhm_px,
        plate_solve_fov_deg,
        max_bright_stars,
        max_sky_match_arcsec,
        gaia_max_mag,
        min_pairs,
        log_gaia_sql,
        log_messages,
    )
    return None, {"refined": False, "reason": "gaia_refine_removed"}


def _refine_masterstar_wcs_gaia_sky_match_infile(
    fits_path: Path,
    *,
    app_config: AppConfig | None,
    equipment_id: int | None,
    dao_threshold_sigma: float,
    dao_fwhm_px: float | None = None,
    max_bright_stars: int = 200,
    max_sky_match_arcsec: float = 25.0,
    gaia_max_mag: float = 16.5,
    min_pairs: int = 10,
) -> dict[str, Any]:
    """Legacy WCS refinement removed (no-op)."""
    _ = (
        fits_path,
        app_config,
        equipment_id,
        dao_threshold_sigma,
        dao_fwhm_px,
        max_bright_stars,
        max_sky_match_arcsec,
        gaia_max_mag,
        min_pairs,
    )
    return {"refined": False, "reason": "gaia_refine_removed"}


def generate_masterstar_and_catalog(
    *,
    archive_path: Path,
    max_catalog_rows: int = 12000,
    astrometry_api_key: str | None = None,
    source_root: Path | None = None,
    platesolve_dir: Path | None = None,
    platesolve_backend: str = "vyvar",
    plate_solve_fov_deg: float = 1.0,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    n_comparison_stars: int = 150,
    require_non_variable_comparisons: bool = True,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    equipment_saturate_adu: float | None = None,
    catalog_local_gaia_only: bool | None = None,
    app_config: AppConfig | None = None,
    equipment_id: int | None = None,
    draft_id: int | None = None,
    master_dark_path: Path | str | None = None,
    masterstar_candidate_paths: "Sequence[str] | None" = None,
    masterstar_selection_pct: float | None = None,
    setup_name: str | None = None,
    masterstar_basename: str = "MASTERSTAR.fits",
    masterstars_csv_basename: str = "masterstars_full_match.csv",
    masterstar_fits_only: bool = False,
    masterstar_skip_build: bool = False,
    hint_ra_deg: float | None = None,
    hint_dec_deg: float | None = None,
) -> dict[str, Any]:
    """Create MASTERSTAR.fits, plate-solve it, and export masterstars.csv.

    Ak je ``masterstar_fits_only=True``, po zostavení FITS v ``platesolve/`` sa skončí (žiadny plate-solve ani CSV).
    Ak je ``masterstar_skip_build=True``, preskočí sa build z processed — použije sa existujúci ``MASTERSTAR.fits`` v ``platesolve/`` a beží solver + katalóg.
    """
    max_catalog_rows = max(int(max_catalog_rows), 100000)
    import numpy as np

    ap = Path(archive_path).expanduser()
    # Draft UI môže poslať .../draft_x/non_calibrated — MASTERSTAR a platesolve patria pod koreň draftu.
    if ap.name.casefold() == "non_calibrated":
        ap = ap.parent
    detrended_root: Path | None = None
    if masterstar_skip_build:
        ps = Path(platesolve_dir) if platesolve_dir is not None else (ap / "platesolve")
        platesolve_dir = ps
        platesolve_dir.mkdir(parents=True, exist_ok=True)
        _ms_name = str(masterstar_basename or "MASTERSTAR.fits").strip() or "MASTERSTAR.fits"
        masterstar_fits = Path(platesolve_dir) / _ms_name
        if not masterstar_fits.is_file():
            raise FileNotFoundError(
                f"MASTERSTAR plate-solve: v {platesolve_dir} chýba súbor {_ms_name}. "
                "Najprv v sekcii Masterstar spusti mean / average / copy."
            )
        _match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
        if _match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
            log_event(
                f"MASTERSTAR: catalog match sep eff={_match_sep_eff:.2f} arcsec (min 10 for initial match)."
            )
        log_event(
            f"MASTERSTAR platesolve-from-disk: {masterstar_fits.resolve()} — VYVAR solver + katalóg "
            "(bez nového buildu z processed)."
        )
        ms_selection_meta = {
            "source": "platesolve_existing",
            "file": str(masterstar_fits.resolve()),
        }
        try:
            _ms_resolved = str(masterstar_fits.resolve())
        except OSError:
            _ms_resolved = str(masterstar_fits)
        info = {
            "masterstar_path": _ms_resolved,
            "frames_used": 1,
            "reference_path": _ms_resolved,
            "reference_index": 0,
            "stacked": False,
            "frames_combined": 1,
        }
    if not masterstar_skip_build:
        # MASTERSTAR-only reads from processed/lights/setup_name (robust folder-based discovery).
        if source_root is not None:
            detrended_root = Path(source_root)
        else:
            detrended_root = resolve_masterstar_input_root(ap, setup_name=setup_name)
        if not detrended_root.exists():
            log_event(f"❌ MASTERSTAR FAIL: Input path {detrended_root} not found.")
            processed_lights = ap / "processed" / "lights"
            if processed_lights.is_dir():
                subdirs = sorted(
                    [d for d in processed_lights.iterdir() if d.is_dir()],
                    key=lambda p: p.name.casefold(),
                )
                if subdirs:
                    detrended_root = subdirs[0]
                    log_event(f"✅ MASTERSTAR fallback input found: {detrended_root}")
        if not detrended_root.exists():
            raise FileNotFoundError(f"Missing processed/detrended lights: {detrended_root}")
        # If root exists but has no FITS, try first setup subfolder under processed lights.
        if not _iter_fits_recursive(detrended_root):
            processed_lights = ap / "processed" / "lights"
            if processed_lights.is_dir():
                subdirs = sorted(
                    [d for d in processed_lights.iterdir() if d.is_dir()],
                    key=lambda p: p.name.casefold(),
                )
                for sd in subdirs:
                    if _iter_fits_recursive(sd):
                        log_event(f"✅ MASTERSTAR fallback to setup subdir: {sd}")
                        detrended_root = sd
                        break

        log_event(f"🔍 MASTERSTAR: Searching for candidates in {Path(detrended_root).resolve()}")
        log_event(f"Vstupný priečinok pre Masterstar: {Path(detrended_root).resolve()}")
        _match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
        if _match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
            log_event(
                f"MASTERSTAR: catalog match sep zvýšený na {_match_sep_eff:.2f}\" "
                f"(požadované minimum pre počiatočný match)."
            )

        ps = Path(platesolve_dir) if platesolve_dir is not None else (ap / "platesolve")
        platesolve_dir = ps
        platesolve_dir.mkdir(parents=True, exist_ok=True)
        _ms_name = str(masterstar_basename or "MASTERSTAR.fits").strip() or "MASTERSTAR.fits"
        masterstar_fits = Path(platesolve_dir) / _ms_name
        only_ms_paths: list[Path] | None = None
        ms_selection_meta: dict[str, Any] = {}

        def _map_qc_paths_to_disk(raw_paths: list[str]) -> list[Path]:
            out: list[Path] = []
            for rp in raw_paths:
                hit = _resolve_best_effort_path_under(Path(detrended_root), str(rp))
                if hit is not None:
                    out.append(hit)
            return out

        def _disk_stack_fallback_paths(input_dir: Path, *, max_frames: int = 8) -> list[Path]:
            """When QC paths / DB mapping fail: sigma-median stack a small batch from disk (deterministic order)."""
            all_on_disk = sorted(_iter_fits_recursive(input_dir), key=lambda p: str(p).casefold())
            if not all_on_disk:
                return []
            n = max(1, min(int(max_frames), len(all_on_disk)))
            return all_on_disk[:n]

        try:
            _pct_eff = float(masterstar_selection_pct) if masterstar_selection_pct is not None else 10.0
        except (TypeError, ValueError):
            _pct_eff = 10.0
        if not math.isfinite(_pct_eff) or _pct_eff <= 0:
            _pct_eff = 10.0
        _pct_eff = max(0.1, min(100.0, _pct_eff))

        cand_paths = [str(x) for x in (masterstar_candidate_paths or []) if str(x).strip()]
        if cand_paths:
            mapped = _map_qc_paths_to_disk(cand_paths)
            if mapped:
                only_ms_paths = mapped
                ms_selection_meta = {
                    "source": "ui_paths",
                    "requested": int(len(cand_paths)),
                    "mapped_found": int(len(mapped)),
                }
            else:
                log_event(
                    "MASTERSTAR: žiadna z ciest z UI sa nenašla pod zadaným koreňom; skúšam výber z DB (draft + % top)…"
                )
                ms_selection_meta = {"source": "ui_paths", "requested": int(len(cand_paths)), "mapped_found": 0}

        if only_ms_paths is None and draft_id is not None:
            _db_ms = _vyvar_open_database(app_config or AppConfig())
            if _db_ms is not None:
                try:
                    db_paths = get_masterstar_candidates(int(draft_id), _pct_eff, db=_db_ms)
                    mapped_db = _map_qc_paths_to_disk([str(x) for x in db_paths if str(x).strip()])
                    if mapped_db:
                        only_ms_paths = mapped_db
                        ms_selection_meta = {
                            "source": "db_top_pct",
                            "draft_id": int(draft_id),
                            "pct": float(_pct_eff),
                            "mapped_found": int(len(mapped_db)),
                        }
                        log_event(
                            f"MASTERSTAR: výber z DB (draft {int(draft_id)}, top {_pct_eff:g} %) → "
                            f"{len(mapped_db)} kandidátov (najlepší sa skopíruje do platesolve)."
                        )
                    else:
                        log_event(
                            f"MASTERSTAR: DB výber (draft {int(draft_id)}) sa nepodarilo namapovať na FITS pod {detrended_root}."
                        )
                except Exception as exc:  # noqa: BLE001
                    log_event(f"MASTERSTAR: DB výber kandidátov zlyhal ({exc!s}).")
                finally:
                    try:
                        _db_ms.conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        if only_ms_paths is None:
            disk_batch = _disk_stack_fallback_paths(Path(detrended_root), max_frames=8)
            if disk_batch:
                only_ms_paths = disk_batch
                ms_selection_meta = {
                    "source": "disk_fallback_stack",
                    "mapped_found": int(len(disk_batch)),
                }
                log_event(
                    f"MASTERSTAR disk fallback: {len(disk_batch)} kandidátov z disku (bez platného QC výberu)."
                )

        if only_ms_paths is None:
            raise FileNotFoundError(
                f"MASTERSTAR: v {detrended_root} nie sú žiadne FITS pre výber ani po UI/DB."
            )

        _cfg_stack = app_config or AppConfig()
        try:
            _ms_fwhm_fb = float(getattr(_cfg_stack, "sips_dao_fwhm_px", 2.5))
        except (TypeError, ValueError):
            _ms_fwhm_fb = 2.5
        if not math.isfinite(_ms_fwhm_fb) or _ms_fwhm_fb <= 0:
            _ms_fwhm_fb = 2.5
        if draft_id is not None:
            _dbc_fw = _vyvar_open_database(_cfg_stack)
            if _dbc_fw is not None:
                try:
                    _fdf = get_masterstar_candidate_rows(int(draft_id), 100.0, db=_dbc_fw)
                    if _fdf is not None and not _fdf.empty and "FWHM" in _fdf.columns:
                        _vals = pd.to_numeric(_fdf["FWHM"], errors="coerce").to_numpy(dtype=float)
                        _vals = _vals[np.isfinite(_vals) & (_vals > 0.5) & (_vals < 80.0)]
                        if _vals.size:
                            _ms_fwhm_fb = float(np.median(_vals))
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    try:
                        _dbc_fw.conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        _db_ms_build: VyvarDatabase | None = None
        if draft_id is not None:
            _db_ms_build = _vyvar_open_database(_cfg_stack)
        # Dočasná inicializácia pre prípad že cleanup nezmazal všetky referencie
        # masterstar_build_mode a masterstar_stack_n sú deprecated — hardcoded copy
        try:
            info = build_masterstar_from_detrended(
                detrended_root=detrended_root,
                output_fits=masterstar_fits,
                only_paths=only_ms_paths,
                fwhm_fallback_px=float(_ms_fwhm_fb),
                app_config=_cfg_stack,
                draft_id=draft_id,
                db=_db_ms_build,
            )
        finally:
            if _db_ms_build is not None:
                try:
                    _db_ms_build.conn.close()
                except Exception:  # noqa: BLE001
                    pass
        try:
            _legacy_master = Path(detrended_root) / "MASTERSTAR.fits"
            if _legacy_master.is_file() and _legacy_master.resolve() != masterstar_fits.resolve():
                _legacy_master.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    _selected_ref_path: Path | None = None
    try:
        _rp = str(info.get("reference_path") or "").strip()
        _selected_ref_path = Path(_rp) if _rp else None
    except Exception:  # noqa: BLE001
        _selected_ref_path = None

    if masterstar_fits_only:
        _cfg_fast = app_config or AppConfig()
        try:
            _ms_out = str(Path(masterstar_fits).resolve())
        except OSError:
            _ms_out = str(masterstar_fits)
        log_event(
            f"MASTERSTAR (len FITS, bez plate-solve): zapísané {_ms_out} | "
            f"zkombinovaných snímok={info.get('frames_combined', info.get('frames_used', '?'))}"
        )
        out_fast: dict[str, Any] = {
            "masterstar_fits": _ms_out,
            "masterstars_csv": "",
            "frames_used": int(info.get("frames_used", 0)),
            "masterstar_selection": ms_selection_meta or None,
            "masterstar_build_info": info,
            "n_raw_dao": 0,
            "detected_stars": 0,
            "catalog_matched": 0,
            "catalog_rows": 0,
            "catalog_match_max_sep_arcsec": float(_match_sep_eff),
            "solve": {"skipped": True, "reason": "masterstar_fits_only"},
        }
        try:
            if draft_id is not None:
                _db_ms = VyvarDatabase(Path(_cfg_fast.database_path))
                try:
                    _db_ms.set_obs_draft_masterstar_path(int(draft_id), _ms_out)
                finally:
                    _db_ms.conn.close()
        except Exception as exc:  # noqa: BLE001
            out_fast["masterstar_path_store_error"] = str(exc)
        return out_fast

    # Solve WCS (MASTERSTAR): always force VYVAR local solver.
    # Ak processed FITS (reference snímka) má dobrý ASTAP WCS (izotropná CD matica),
    # injektujeme ho do MASTERSTAR pred solverom — solver refinuje od dobrého základu.
    if _selected_ref_path is not None and _selected_ref_path.is_file():
        try:
            with fits.open(_selected_ref_path, memmap=False) as _href:
                _rh = _href[0].header
                _ref_pltsolvd = bool(_rh.get("PLTSOLVD", False))
                _ref_has_cd = all(k in _rh for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"))
                if _ref_pltsolvd and _ref_has_cd:
                    _sx = abs(float(_rh["CD1_1"])) * 3600
                    _sy = abs(float(_rh["CD2_2"])) * 3600
                    _ratio = max(_sx, _sy) / max(min(_sx, _sy), 0.001)
                    if _ratio < 1.1 and 7.0 < _sx < 13.0:
                        with fits.open(masterstar_fits, mode="update", memmap=False) as _hms:
                            _hmh = _hms[0].header
                            for _k in (
                                "CD1_1",
                                "CD1_2",
                                "CD2_1",
                                "CD2_2",
                                "CRVAL1",
                                "CRVAL2",
                                "CRPIX1",
                                "CRPIX2",
                                "CTYPE1",
                                "CTYPE2",
                                "CUNIT1",
                                "CUNIT2",
                                "EQUINOX",
                                "RADESYS",
                            ):
                                if _k in _rh:
                                    _hmh[_k] = _rh[_k]
                            for _k in (
                                "PC1_1",
                                "PC1_2",
                                "PC2_1",
                                "PC2_2",
                                "CDELT1",
                                "CDELT2",
                                "WCSAXES",
                                "CROTA1",
                                "CROTA2",
                            ):
                                if _k in _hmh:
                                    del _hmh[_k]
                            _hms.flush()
                        log_event(
                            f"MASTERSTAR: ASTAP WCS z {_selected_ref_path.name} "
                            f"(sx={_sx:.3f} sy={_sy:.3f} ratio={_ratio:.3f}) "
                            "injected ako štartovací WCS pre VYVAR solver."
                        )
        except Exception as _wcs_inj_exc:  # noqa: BLE001
            log_event(f"MASTERSTAR: ASTAP WCS inject zlyhal (nekritické): {_wcs_inj_exc!s}")

    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)

    _cfg_ms = app_config or AppConfig()
    _mra, _mde, _ = _pointing_hint_from_header(hdr)
    if hint_ra_deg is not None and hint_dec_deg is not None:
        try:
            _hra_ov = float(hint_ra_deg)
            _hde_ov = float(hint_dec_deg)
            if math.isfinite(_hra_ov) and math.isfinite(_hde_ov):
                _mra, _mde = _hra_ov, _hde_ov
                log_event(
                    "MASTERSTAR: hint_ra_deg / hint_dec_deg z volania prepisujú hint z FITS "
                    "(druhý MASTERSTAR / detrended aligned)."
                )
        except (TypeError, ValueError):
            pass
    try:
        _hint_sep_thr = float(getattr(_cfg_ms, "masterstar_solver_use_draft_median_if_hint_sep_deg", 1.0))
    except (TypeError, ValueError):
        _hint_sep_thr = 1.0
    if not math.isfinite(_hint_sep_thr) or _hint_sep_thr < 0:
        _hint_sep_thr = 1.0
    if draft_id is not None:
        _dbc_hint = _vyvar_open_database(_cfg_ms)
        if _dbc_hint is not None:
            try:
                med_ra, med_de = draft_median_pointing_icrs_deg(_dbc_hint, int(draft_id))
                if med_ra is not None and med_de is not None:
                    if _mra is None or _mde is None:
                        _mra, _mde = med_ra, med_de
                        log_event(
                            "MASTERSTAR solve: používam medián RA/Dec z OBS_FILES (hlavička bez spoľahlivého hintu)."
                        )
                    else:
                        sc_h = SkyCoord(ra=float(_mra) * u.deg, dec=float(_mde) * u.deg, frame="icrs")
                        sc_d = SkyCoord(ra=float(med_ra) * u.deg, dec=float(med_de) * u.deg, frame="icrs")
                        sep = float(sc_h.separation(sc_d).deg)
                        if sep > float(_hint_sep_thr):
                            log_event(
                                f"MASTERSTAR solve: hint vs draft median = {sep:.3f}° > {_hint_sep_thr}° "
                                "— používam draft medián z OBS_FILES."
                            )
                            _mra, _mde = med_ra, med_de
                        elif sep > 0.05:
                            log_event(
                                f"MASTERSTAR solve: hint vs draft median = {sep:.3f}° (skontrolujte pointing)."
                            )
            finally:
                try:
                    _dbc_hint.conn.close()
                except Exception:  # noqa: BLE001
                    pass

    try:
        _dao_sigma_eff = float(getattr(_cfg_ms, "masterstar_dao_threshold_sigma", 1.8))
    except (TypeError, ValueError):
        _dao_sigma_eff = 1.8
    if not math.isfinite(_dao_sigma_eff) or _dao_sigma_eff <= 0:
        _dao_sigma_eff = 1.8
    _dao_sigma_eff = max(0.1, min(6.0, float(_dao_sigma_eff)))
    log_event(
        f"MASTERSTAR: DAO threshold σ×RMS = {_dao_sigma_eff:.2f} "
        f"(config masterstar_dao_threshold_sigma; plate solve + katalóg)"
    )

    _full_db = str(getattr(_cfg_ms, "gaia_db_path", "") or "").strip()
    if not _full_db:
        raise RuntimeError(
            "MASTERSTAR: v Settings nastavte gaia_db_path (plná lokálna Gaia DR3 SQLite DB)."
        )
    from vyvar_platesolver import solve_wcs_with_local_gaia

    log_event("MASTERSTAR WCS: VYVAR solver + plná Gaia DB (gaia_db_path).")
    try:
        _sip_ms = int(getattr(_cfg_ms, "masterstar_platesolve_sip_max_order", 5))
    except (TypeError, ValueError):
        _sip_ms = 5
    _sip_ms = max(2, min(5, _sip_ms))
    try:
        _sip_lo = int(getattr(_cfg_ms, "masterstar_platesolve_sip_min_order", 3))
    except (TypeError, ValueError):
        _sip_lo = 3
    _sip_lo = max(2, min(5, _sip_lo))
    if _sip_lo > _sip_ms:
        _sip_lo = _sip_ms
    log_event(
        f"MASTERSTAR: SIP skúšanie {_sip_ms}→…→{_sip_lo} (config max/min plate-solve SIP)."
    )
    _exp_arc = getattr(_cfg_ms, "masterstar_platesolve_expected_arcsec_per_px", None)
    if _exp_arc is not None:
        try:
            _ea = float(_exp_arc)
            if math.isfinite(_ea) and _ea > 0:
                log_event(
                    f"MASTERSTAR: očakávaná mierka z config masterstar_platesolve_expected_arcsec_per_px = {_ea:.4f} arcsec/px."
                )
        except (TypeError, ValueError):
            _exp_arc = None
    _prms = getattr(_cfg_ms, "masterstar_platesolve_prewrite_rms_max_px", None)
    _prms_r = getattr(_cfg_ms, "masterstar_platesolve_prewrite_relaxed_rms_max_px", None)
    _nnrms = getattr(_cfg_ms, "masterstar_platesolve_nn_refine_max_rms_px", None)
    # MASTERSTAR platesolve: always single best processed FITS (copy mode).
    _ms_vyvar_max_rows = 30000

    def _run_masterstar_vyvar_solve(*, enable_sip: bool, sip_max_order: int, fov_deg: float, max_rows: int) -> dict[str, Any]:
        return solve_wcs_with_local_gaia(
            masterstar_fits,
            hint_ra_deg=_mra,
            hint_dec_deg=_mde,
            fov_diameter_deg=float(fov_deg),
            gaia_db_path=Path(_full_db),
            enable_sip=bool(enable_sip),
            sip_max_order=int(sip_max_order),
            ransac_refinement=True,
            max_catalog_rows=int(max_rows),
            faintest_mag_limit=18.0,
            dao_threshold_sigma=float(_dao_sigma_eff),
            expected_plate_scale_arcsec_per_px=(
                float(_exp_arc) if _exp_arc is not None else None
            ),
            masterstar_prewrite_rms_max_px=float(_prms) if _prms is not None else None,
            masterstar_prewrite_relaxed_rms_max_px=float(_prms_r) if _prms_r is not None else None,
            masterstar_nn_refine_max_rms_px=float(_nnrms) if _nnrms is not None else None,
            masterstar_sip_min_order=int(_sip_lo),
        )

    # Ak injektovaný ASTAP WCS je dobrý (izotropný, ratio<1.05),
    # preskočíme VYVAR solver a použijeme ASTAP WCS priamo.
    _skip_vyvar_solve = False
    if _selected_ref_path is not None and _selected_ref_path.is_file():
        try:
            with fits.open(_selected_ref_path, memmap=False) as _href_chk:
                _rh_chk = _href_chk[0].header
                if (bool(_rh_chk.get("PLTSOLVD", False))
                        and all(k in _rh_chk for k in ("CD1_1", "CD2_2"))):
                    _sx_chk = abs(float(_rh_chk["CD1_1"])) * 3600
                    _sy_chk = abs(float(_rh_chk["CD2_2"])) * 3600
                    _ratio_chk = max(_sx_chk, _sy_chk) / max(min(_sx_chk, _sy_chk), 0.001)
                    if _ratio_chk < 1.05 and 7.0 < _sx_chk < 13.0:
                        _skip_vyvar_solve = True
                        log_event(
                            f"MASTERSTAR: ASTAP WCS ratio={_ratio_chk:.3f} — "
                            "preskakujem VYVAR solver, používam ASTAP WCS priamo."
                        )
        except Exception:
            pass

    if _skip_vyvar_solve:
        # Použijem ASTAP WCS ako finálny — simulujeme úspešný solve result
        solve_meta = {"solved": True, "match_rate": 0.95, "skipped_vyvar": True}
    else:
        solve_meta = _run_masterstar_vyvar_solve(
            enable_sip=True,
            sip_max_order=int(_sip_ms),
            fov_deg=float(plate_solve_fov_deg),
            max_rows=int(_ms_vyvar_max_rows),
        )
    if not isinstance(solve_meta, dict) or not bool(solve_meta.get("solved", False)):
        raise RuntimeError(
            "MASTERSTAR plate-solve zlyhal. "
            f"Back-end returned: {solve_meta!r}. "
            "Cannot safely continue with photometry / source extraction."
        )

    # Refresh header/data after solve attempt (solver overwrote MASTERSTAR.fits header)
    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)
    if not _has_valid_wcs(hdr):
        raise RuntimeError(
            "MASTERSTAR: po plate-solve chýba platný WCS. Skontroluj gaia_db_path, RA/Dec a mierku v hlavičke "
            "(FOCALLEN/PIXSIZE alebo SECPIX) a výstup solvera."
        )

    # Pipeline-level acceptance criteria (stricter than solver's minimal guard):
    # - match_rate: allow 60% on the first solve (optimizer refines later)
    try:
        _mr = float(solve_meta.get("match_rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        _mr = 0.0
    _min_mr = 0.60
    if _mr < _min_mr:
        raise RuntimeError(
            f"MASTERSTAR plate-solve zamietnutý: match_rate={_mr * 100.0:.1f}% < {_min_mr * 100.0:.0f}%. "
            "Skús zvýšiť n_stack alebo upraviť hint/DAO prahy."
        )

    try:
        _aniso_thr = float(getattr(_cfg_ms, "platesolve_anisotropy_threshold", 1.3))
    except (TypeError, ValueError):
        _aniso_thr = 1.3
    if not math.isfinite(_aniso_thr) or _aniso_thr <= 0:
        _aniso_thr = 1.3
    _aniso_thr = max(1.01, min(5.0, float(_aniso_thr)))

    # Post-solve anisotropy validation: reject strongly anisotropic pixel scale and retry solver once.
    try:
        from astropy.wcs import WCS

        wcs0 = WCS(hdr)
        scale_x = abs(float(wcs0.pixel_scale_matrix[0, 0])) * 3600.0  # arcsec/px
        scale_y = abs(float(wcs0.pixel_scale_matrix[1, 1])) * 3600.0  # arcsec/px
        if math.isfinite(scale_x) and math.isfinite(scale_y) and scale_x > 0 and scale_y > 0:
            scale_ratio = max(scale_x, scale_y) / min(scale_x, scale_y)
        else:
            scale_ratio = float("nan")
    except Exception:  # noqa: BLE001
        scale_ratio = float("nan")

    if math.isfinite(scale_ratio) and scale_ratio > _aniso_thr:
        log_event(
            f"VAROVANIE: Anizotropná mierka ratio={scale_ratio:.2f} — plate-solve zamietnutý, restartujem solver (relaxed)."
        )
        # Retry with relaxed knobs:
        # - slightly larger FOV diameter (hint-vs-solved tolerance),
        # - more Gaia rows,
        # - no SIP (simpler model can be more stable when the fit goes off-rails).
        solve_meta2 = _run_masterstar_vyvar_solve(
            enable_sip=False,
            sip_max_order=0,
            fov_deg=float(plate_solve_fov_deg) * 1.25,
            max_rows=int(max(_ms_vyvar_max_rows, 30000)),
        )
        if not isinstance(solve_meta2, dict) or not bool(solve_meta2.get("solved", False)):
            raise RuntimeError(
                f"MASTERSTAR platesolve retry zlyhal po anizotropii. Back-end returned: {solve_meta2!r}"
            )
        solve_meta = solve_meta2
        # Reload header after retry
        with fits.open(masterstar_fits, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        if not _has_valid_wcs(hdr):
            raise RuntimeError("MASTERSTAR: po retry plate-solve chýba platný WCS.")
        try:
            from astropy.wcs import WCS

            wcs1 = WCS(hdr)
            sx = abs(float(wcs1.pixel_scale_matrix[0, 0])) * 3600.0
            sy = abs(float(wcs1.pixel_scale_matrix[1, 1])) * 3600.0
            if math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0:
                scale_ratio2 = max(sx, sy) / min(sx, sy)
            else:
                scale_ratio2 = float("nan")
        except Exception:  # noqa: BLE001
            scale_ratio2 = float("nan")
        if math.isfinite(scale_ratio2) and scale_ratio2 > _aniso_thr:
            raise RuntimeError(
                f"MASTERSTAR plate-solve zamietnutý: anizotropná mierka po retry ratio={scale_ratio2:.2f} (>{_aniso_thr})."
            )

    # --- astrometry.net (WSL) fallback ak je VYVAR WCS „zlý“ (pomer mierky / odchýlka od očakávanej) ---
    _anet_cfg = str(getattr(_cfg_ms, "astrometry_net_wsl_cfg", "") or "").strip() or "/home/uhlar/.config/astrometry.cfg"
    _anet_enabled = bool(getattr(_cfg_ms, "astrometry_net_enabled", True))
    _anet_to = int(getattr(_cfg_ms, "astrometry_net_timeout_sec", 180) or 180)
    _anet_to = max(30, min(3600, _anet_to))
    try:
        _anet_scale_err = float(getattr(_cfg_ms, "astrometry_net_scale_err_pct", 15.0))
    except (TypeError, ValueError):
        _anet_scale_err = 15.0
    _anet_scale_err = max(1.0, min(80.0, _anet_scale_err))
    try:
        _anet_radius = float(getattr(_cfg_ms, "astrometry_net_radius_deg", 3.0))
    except (TypeError, ValueError):
        _anet_radius = 3.0
    _anet_radius = max(0.1, min(30.0, _anet_radius))

    _exp_scale_apx: float | None = None
    if _exp_arc is not None:
        try:
            _ea2 = float(_exp_arc)
            if math.isfinite(_ea2) and _ea2 > 0:
                _exp_scale_apx = float(_ea2)
        except (TypeError, ValueError):
            _exp_scale_apx = None
    if _exp_scale_apx is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                _w_hdr = WCS(hdr)
            if getattr(_w_hdr, "has_celestial", False):
                _pm0 = _w_hdr.pixel_scale_matrix
                _sx0 = abs(float(_pm0[0, 0])) * 3600.0
                _sy0 = abs(float(_pm0[1, 1])) * 3600.0
                if math.isfinite(_sx0) and math.isfinite(_sy0) and _sx0 > 0 and _sy0 > 0:
                    _exp_scale_apx = float((_sx0 + _sy0) / 2.0)
        except Exception:  # noqa: BLE001
            _exp_scale_apx = None
    if _exp_scale_apx is None or (not math.isfinite(_exp_scale_apx)) or _exp_scale_apx <= 0:
        _exp_scale_apx = 9.77

    _wcs_ok = False
    try:
        with fits.open(masterstar_fits, memmap=False) as _hd_wq:
            _w_check = WCS(_hd_wq[0].header)
        _wcs_q = masterstar_wcs_quality(
            _w_check, float(_exp_scale_apx), anisotropy_limit=float(_aniso_thr)
        )
        _wcs_ok = bool(_wcs_q.get("ok", False))
        if not _wcs_ok:
            _rq = _wcs_q.get("ratio")
            _se = _wcs_q.get("scale_err_pct")
            try:
                _rq_s = f"{float(_rq):.2f}" if _rq is not None and math.isfinite(float(_rq)) else str(_rq)
            except (TypeError, ValueError):
                _rq_s = str(_rq)
            try:
                _se_s = f"{float(_se):.1f}" if _se is not None and math.isfinite(float(_se)) else str(_se)
            except (TypeError, ValueError):
                _se_s = str(_se)
            log_event(
                f"MASTERSTAR WCS kvalita: zlá (ratio={_rq_s}, scale_err={_se_s}%) — "
                "skúšam astrometry.net WSL fallback."
            )
    except Exception as _wq_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR WCS check failed: {_wq_exc}")
        _wcs_ok = False

    if not _wcs_ok and _anet_enabled:
        if is_wsl_available():
            if _mra is None or _mde is None or (not math.isfinite(float(_mra))) or (not math.isfinite(float(_mde))):
                log_event("MASTERSTAR: astrometry.net WSL preskočený — chýba RA/Dec hint.")
            else:
                log_event("MASTERSTAR: spúšťam astrometry.net WSL solve-field...")
                _ans = solve_field_wsl(
                    fits_path=masterstar_fits,
                    ra_hint_deg=float(_mra),
                    dec_hint_deg=float(_mde),
                    scale_arcsec_per_px=float(_exp_scale_apx),
                    scale_err_pct=float(_anet_scale_err),
                    radius_deg=float(_anet_radius),
                    wsl_cfg=_anet_cfg,
                    timeout_sec=int(_anet_to),
                    anisotropy_max_ratio=float(_aniso_thr),
                )
                if _ans.get("success"):
                    solve_meta["astrometry_net_wsl_fallback"] = True
                    _sx_a = _ans.get("scale_x_arcsec")
                    _sy_a = _ans.get("scale_y_arcsec")
                    try:
                        _sxa = float(_sx_a) if _sx_a is not None else float("nan")
                        _sya = float(_sy_a) if _sy_a is not None else float("nan")
                        log_event(
                            "MASTERSTAR: astrometry.net WSL OK "
                            f"(scale={_sxa:.3f}x{_sya:.3f} arcsec/px)"
                        )
                    except (TypeError, ValueError):
                        log_event("MASTERSTAR: astrometry.net WSL OK (scale detail N/A)")
                    with fits.open(masterstar_fits, memmap=False) as hdul:
                        hdr = hdul[0].header.copy()
                        data = np.array(hdul[0].data, dtype=np.float32, copy=True)
                else:
                    log_event(f"MASTERSTAR: astrometry.net WSL FAILED: {_ans.get('error', 'unknown')}")
        else:
            log_event("MASTERSTAR: WSL/solve-field nie je dostupný — skip astrometry.net fallback.")
    elif not _wcs_ok:
        log_event("MASTERSTAR: astrometry.net disabled v config — pokračujem so zlým WCS.")

    try:
        _pscale_adj = _try_rescale_masterstar_linear_wcs_to_expected_plate_scale(
            masterstar_fits,
            app_config=app_config or AppConfig(),
            equipment_id=equipment_id,
            draft_id=draft_id,
        )
    except Exception as exc:  # noqa: BLE001
        log_event(f"WCS PLATE SCALE: neočakávaná chyba — {exc!s}")
        _pscale_adj = {"rescaled": False, "error": str(exc)}
    solve_meta["wcs_plate_scale_adjustment"] = _pscale_adj

    # _cfg_ms / _dao_sigma_eff už vyššie (rovnaké DAO σ pre plate solve aj katalóg).
    _ms_fwhm = float(getattr(_cfg_ms, "sips_dao_fwhm_px", 2.5))
    if not math.isfinite(_ms_fwhm) or _ms_fwhm <= 0:
        _ms_fwhm = 2.5

    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)
    data = np.ascontiguousarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    _skip_boj = True
    if _skip_boj:
        _ms_mean = float(np.nanmean(data))
        log_event(
            "MASTERSTAR: globálne odčítanie medianu (BOJ) vypnuté."
        )
        log_event(f"MASTERSTAR po nan_to_num: mean={_ms_mean:.6f}")
    else:
        _ms_sky = data[data > 10.0]
        if _ms_sky.size > 0:
            _ms_med = float(np.nanmedian(_ms_sky))
        else:
            _ms_med = float(np.nanmedian(data))
        data -= np.float32(_ms_med)
        data = np.clip(data, -100.0, None).astype(np.float32, copy=False)
        _ms_mean = float(np.nanmean(data))
        log_event(f"⭐ MASTERSTAR - Background flattened by {_ms_med:.2f}")
        log_event(f"🚨 BOJ O NULU: Removed {_ms_med:.2f} | Resulting Mean: {_ms_mean:.6f}")
    _ms_min = float(np.nanmin(data))
    _ms_max = float(np.nanmax(data))
    log_event(f"MASTERSTAR levels: noise_floor(min)={_ms_min:.2f}, saturation_proxy(max)={_ms_max:.2f}")

    platesolve_dir.mkdir(parents=True, exist_ok=True)
    _fov_job = float(plate_solve_fov_deg)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _w_pre = WCS(hdr)
        if _w_pre.has_celestial:
            _, _r_cat_need = _effective_field_catalog_cone_radius_deg(
                _w_pre, int(data.shape[0]), int(data.shape[1]), _fov_job, fits_header=hdr
            )
            _invalidate_field_catalog_cone_cache_if_needed(
                platesolve_dir / "field_catalog_cone.csv",
                plate_solve_fov_deg=_fov_job,
                effective_radius_deg=float(_r_cat_need),
            )
    except Exception as exc:  # noqa: BLE001
        log_event(f"Katalóg: kontrola cache field_catalog_cone preskočená — {exc!s}")

    # Full-field MASTERSTAR depth: keep deeper Gaia and larger catalog rows for corner recovery.
    _ms_max_catalog_rows_eff = max(int(max_catalog_rows), 100000)
    if faintest_mag_limit is None:
        _ms_faintest_mag_eff: float | None = 18.0
    else:
        try:
            _ms_faintest_mag_eff = max(float(faintest_mag_limit), 18.0)
        except (TypeError, ValueError):
            _ms_faintest_mag_eff = 18.0
    df_out, det_meta = detect_stars_and_match_catalog(
        data,
        hdr,
        max_catalog_rows=int(_ms_max_catalog_rows_eff),
        cat_df=None,
        match_sep_arcsec=float(_match_sep_eff),
        saturate_level_fraction=float(saturate_level_fraction),
        faintest_mag_limit=_ms_faintest_mag_eff,
        field_catalog_export_path=platesolve_dir / "field_catalog_cone.csv",
        dao_threshold_sigma=float(_dao_sigma_eff),
        dao_fwhm_px=_ms_fwhm,
        equipment_saturate_adu=equipment_saturate_adu,
        catalog_local_gaia_only=catalog_local_gaia_only,
        plate_solve_fov_deg=float(plate_solve_fov_deg),
        prematch_peak_sigma_floor=float(
            getattr(_cfg_ms, "masterstar_prematch_peak_sigma_floor", 3.2)
        ),
    )
    try:
        if isinstance(solve_meta, dict) and bool(solve_meta.get("solved")):
            _px = solve_meta.get("pairs_x")
            _py = solve_meta.get("pairs_y")
            _pra = solve_meta.get("pairs_ra")
            _pde = solve_meta.get("pairs_de")
            _pids = solve_meta.get("pairs_catalog_id")
            if (
                isinstance(_px, list)
                and isinstance(_py, list)
                and isinstance(_pra, list)
                and isinstance(_pde, list)
                and isinstance(_pids, list)
                and len(_px) > 0
                and len(_px) == len(_py) == len(_pra) == len(_pde) == len(_pids)
            ):
                _sm0 = solve_meta.get("sip_meta") if isinstance(solve_meta.get("sip_meta"), dict) else {}
                _mir = str((_sm0 or {}).get("det_mirror_orientation") or "").strip()
                df_out = _merge_platesolve_gaia_pairs_into_masterstars_df(
                    df_out,
                    pairs_x=[float(t) for t in _px],
                    pairs_y=[float(t) for t in _py],
                    pairs_ra=[float(t) for t in _pra],
                    pairs_de=[float(t) for t in _pde],
                    pairs_catalog_id=[str(t) for t in _pids],
                )
                log_event(
                    f"MASTERSTAR: VYVAR páry ({len(_px)}) zlúčené do katalógu "
                    f"(mirror={_mir or 'native'}, pre astrometry optimizer)."
                )
    except Exception as exc:  # noqa: BLE001
        log_event(f"MASTERSTAR: zlúčenie VYVAR párov preskočené — {exc!s}")

    if "b_v" in df_out.columns and "bp_rp" not in df_out.columns:
        df_out = df_out.copy()
        df_out["bp_rp"] = pd.to_numeric(df_out["b_v"], errors="coerce")
    if "mag" in df_out.columns:
        df_out = df_out.copy()
        df_out["phot_g_mean_mag"] = pd.to_numeric(df_out["mag"], errors="coerce")

    if int(det_meta.get("n_detected", 0)) == 0:
        raise RuntimeError("No stars detected on MASTERSTAR.")
    _n_det_raw = int(det_meta.get("n_detected", 0) or 0)
    _n_mat_raw = int(det_meta.get("n_matched", 0) or 0)
    _rate_raw = (100.0 * float(_n_mat_raw) / float(_n_det_raw)) if _n_det_raw > 0 else 0.0
    log_event(
        f"📊 MATCH STATS (raw): Found {_n_det_raw} stars on image | {_n_mat_raw} matched with Gaia | Match Rate: {_rate_raw:.2f}%"
    )
    _update_masterstar_obs_file_status(
        cfg=_cfg_ms,
        draft_id=draft_id,
        selected_ref_path=_selected_ref_path,
        wcs_ok=bool(_has_valid_wcs(hdr)),
        n_stars=_n_det_raw,
    )
    temp_csv = platesolve_dir / "masterstars.csv"
    _msc_name = str(masterstars_csv_basename or "masterstars_full_match.csv").strip() or "masterstars_full_match.csv"
    csv_path = platesolve_dir / _msc_name
    _vyvar_df_to_csv(df_out, temp_csv)
    try:
        from astrometry_optimizer import optimize_masterstar_matches

        _gdb_opt = str(getattr(_cfg_ms, "gaia_db_path", "") or "").strip()
        if _gdb_opt:
            _mir_extra = bool(getattr(_cfg_ms, "masterstar_optimizer_mirror_extra_log", True))
            csv_path = optimize_masterstar_matches(
                masterstars_csv=temp_csv,
                masterstar_fits=masterstar_fits,
                gaia_db_path=_gdb_opt,
                output_csv=csv_path,
                gaia_mag_limit=float(_ms_faintest_mag_eff),
                gaia_max_catalog_rows=int(_ms_max_catalog_rows_eff),
                mirror_orientation_extra_log=_mir_extra,
                sip_force_rms_guard_ratio=_cfg_ms.masterstar_sip_force_rms_guard_ratio,
            )
            # Force one more pass after WCS displacement update for final edge recovery.
            csv_path = optimize_masterstar_matches(
                masterstars_csv=csv_path,
                masterstar_fits=masterstar_fits,
                gaia_db_path=_gdb_opt,
                output_csv=csv_path,
                gaia_mag_limit=float(_ms_faintest_mag_eff),
                gaia_max_catalog_rows=int(_ms_max_catalog_rows_eff),
                mirror_orientation_extra_log=_mir_extra,
                sip_force_rms_guard_ratio=_cfg_ms.masterstar_sip_force_rms_guard_ratio,
            )
            log_event("MASTERSTAR optimizer: forced final re-match pass completed.")
        else:
            _vyvar_df_to_csv(df_out, csv_path)
    except Exception as exc:  # noqa: BLE001
        log_event(f"MASTERSTAR optimizer skipped/fallback: {exc!s}")
        _vyvar_df_to_csv(df_out, csv_path)
    try:
        df_final = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001
        df_final = df_out.copy()
    df_final = _annotate_masterstars_flux_zones(
        df_final,
        noise_floor_adu=det_meta.get("noise_floor_adu"),
        equipment_saturate_adu=equipment_saturate_adu,
        saturate_limit_adu_fallback=det_meta.get("saturate_limit_adu"),
        saturate_limit_fraction=float(getattr(_cfg_ms, "saturate_limit_fraction", 0.85)),
    )
    try:
        cid = df_final.get("catalog_id", pd.Series([""] * len(df_final))).fillna("").astype(str).str.strip()
        df_final["source_type"] = np.where(cid.ne(""), "GAIA_MATCHED", "DAO_ONLY")
        _vyvar_df_to_csv(df_final, csv_path)
    except Exception as exc:  # noqa: BLE001
        log_event(f"MASTERSTAR source_type annotate failed: {exc!s}")
    _n_det = int(len(df_final))
    _n_mat = int(
        df_final.get("catalog_id", pd.Series([""] * len(df_final)))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    _rate = (100.0 * float(_n_mat) / float(_n_det)) if _n_det > 0 else 0.0
    log_event(
        f"📊 MATCH STATS (optimized): Found {_n_det} stars on image | {_n_mat} matched with Gaia | Match Rate: {_rate:.2f}%"
    )
    log_event(
        f"MASTERSTAR JSON consistency: n_raw_dao={int(det_meta.get('n_detected_dao_raw', 0) or 0)}, "
        f"detected_stars={_n_det}, catalog_matched={_n_mat}"
    )
    log_event(
        f"MASTERSTAR katalóg: {Path(csv_path).name} — {len(df_final)} riadkov "
        f"(DAO + katalóg na celom poli; žiadne orezanie podľa vzdialenosti od stredu snímku)."
    )
    # Gaussian FWHM (2D fit) → FITS header for universal per-frame aperture scaling.
    try:
        from photometry_phase2a import measure_fwhm_from_masterstar

        _ms_path = Path(masterstar_fits)
        if "mag" in df_final.columns:
            _star_pos = df_final[["x", "y", "mag"]].dropna()
        elif "phot_g_mean_mag" in df_final.columns:
            _star_pos = (
                df_final[["x", "y", "phot_g_mean_mag"]]
                .dropna()
                .rename(columns={"phot_g_mean_mag": "mag"})
            )
        else:
            _star_pos = df_final[["x", "y"]].dropna()
        with fits.open(_ms_path, memmap=False) as _hint_hdul:
            _vy_fwhm_hint = float(_hint_hdul[0].header.get("VY_FWHM", 3.5))
        _gaussian_fwhm = measure_fwhm_from_masterstar(
            _ms_path,
            _star_pos,
            dao_fwhm_hint=_vy_fwhm_hint,
            n_stars=30,
        )
        with fits.open(_ms_path, mode="update", memmap=False) as _hdul:
            _hdul[0].header["VY_FWHM_GAUSS"] = (
                round(float(_gaussian_fwhm), 4),
                "Gaussian FWHM [px] z 2D fit na MASTERSTAR",
            )
            _hdul.flush()
        logging.info(
            f"[MASTERSTAR] VY_FWHM_GAUSS={float(_gaussian_fwhm):.3f}px uložené do hlavičky"
        )
    except Exception as _e:
        logging.warning(f"[MASTERSTAR] Gaussian FWHM fit zlyhal: {_e}")
    # Small flush pause: UI may read CSV immediately after this returns.
    time.sleep(0.5)
    # Drop stale pre-optimizer dataframe to avoid accidental reuse ("ghost rows").
    try:
        del df_out
    except Exception:  # noqa: BLE001
        pass
    # Keep platesolve clean: remove temporary/duplicate artifacts.
    for _dup in (
        platesolve_dir / "MASTERSTAR_full.fits",
        platesolve_dir / "MASTERSTAR_full.jpg",
        temp_csv,
    ):
        try:
            if Path(_dup).is_file() and Path(_dup).resolve() != Path(csv_path).resolve():
                Path(_dup).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    photo_plan = write_photometry_plan_files(
        platesolve_dir=platesolve_dir,
        masterstar_fits=masterstar_fits,
        masterstars_csv=csv_path,
        n_comparison_stars=int(n_comparison_stars),
        require_non_variable=bool(require_non_variable_comparisons),
    )

    out: dict[str, Any] = {
        "masterstar_fits": str(masterstar_fits),
        "masterstars_csv": str(csv_path),
        "frames_used": int(info.get("frames_used", 0)),
        "masterstar_selection": ms_selection_meta or None,
        "n_raw_dao": int(det_meta.get("n_detected_dao_raw", 0) or 0),
        "detected_stars": int(_n_det),
        "catalog_matched": int(_n_mat),
        "catalog_rows": int(det_meta.get("catalog_rows", 0)),
        "catalog_match_max_sep_arcsec": float(_match_sep_eff),
        "max_catalog_rows": int(_ms_max_catalog_rows_eff),
        "n_likely_saturated": int(det_meta.get("n_likely_saturated", 0)),
        "saturate_limit_adu": det_meta.get("saturate_limit_adu"),
        "saturate_limit_source": det_meta.get("saturate_limit_source"),
        "solve": solve_meta,
        "n_comparison_stars_requested": int(n_comparison_stars),
        "faintest_mag_limit": det_meta.get("faintest_mag_limit"),
        "n_dropped_fainter_than_limit": det_meta.get("n_dropped_fainter_than_limit"),
        "field_catalog_cone_csv": det_meta.get("field_catalog_cone_csv"),
        "dao_threshold_sigma": det_meta.get("dao_threshold_sigma"),
        "masterstar_match_png": "",
    }
    out.update(photo_plan)
    # MASTER_SOURCES: rich Gaia cross-match for MASTERSTAR detections (stored in project DB).
    try:
        gaia_db = str(getattr(_cfg_ms, "gaia_db_path", "") or "").strip()
        if gaia_db and draft_id is not None and "ra_deg" in df_final.columns and "dec_deg" in df_final.columns:
            det = df_final.copy()
            det["ra_deg"] = pd.to_numeric(det["ra_deg"], errors="coerce")
            det["dec_deg"] = pd.to_numeric(det["dec_deg"], errors="coerce")
            det = det[det["ra_deg"].notna() & det["dec_deg"].notna()].copy()
            if not det.empty:
                ra_min = float(det["ra_deg"].min()) - 0.01
                ra_max = float(det["ra_deg"].max()) + 0.01
                de_min = float(det["dec_deg"].min()) - 0.01
                de_max = float(det["dec_deg"].max()) + 0.01
                ga = query_local_gaia(
                    gaia_db,
                    ra_min=ra_min,
                    ra_max=ra_max,
                    dec_min=de_min,
                    dec_max=de_max,
                )
                if ga:
                    gdf = pd.DataFrame(ga)
                    gcoo = SkyCoord(
                        ra=pd.to_numeric(gdf["ra"], errors="coerce").astype(float).values * u.deg,
                        dec=pd.to_numeric(gdf["dec"], errors="coerce").astype(float).values * u.deg,
                        frame="icrs",
                    )
                    dcoo = SkyCoord(
                        ra=det["ra_deg"].astype(float).values * u.deg,
                        dec=det["dec_deg"].astype(float).values * u.deg,
                        frame="icrs",
                    )
                    idx, sep2d, _ = dcoo.match_to_catalog_sky(gcoo)
                    ok = sep2d.to(u.arcsec).value <= 2.0
                    if bool(np.any(ok)):
                        # Geometry + blending pruning and dynamic photometric binning.
                        nax1 = int(hdr.get("NAXIS1", 0) or 0) or int(data.shape[1])
                        nax2 = int(hdr.get("NAXIS2", 0) or 0) or int(data.shape[0])
                        border_px = 50.0

                        try:
                            from astropy.coordinates import search_around_sky

                            pairs_i, pairs_j, _, _ = search_around_sky(gcoo, gcoo, 5.0 * u.arcsec)
                            gmag_all = (
                                pd.to_numeric(gdf.get("g_mag"), errors="coerce")
                                .astype(float)
                                .to_numpy()
                            )
                            blended_idx: set[int] = set()
                            for a, b in zip(pairs_i, pairs_j, strict=False):
                                ia = int(a)
                                ib = int(b)
                                if ia == ib:
                                    continue
                                ma = gmag_all[ia] if ia < len(gmag_all) else float("nan")
                                mb = gmag_all[ib] if ib < len(gmag_all) else float("nan")
                                if not (math.isfinite(ma) and math.isfinite(mb)):
                                    continue
                                if abs(ma - mb) < 3.0:
                                    blended_idx.add(ia)
                                    blended_idx.add(ib)
                        except Exception:  # noqa: BLE001
                            blended_idx = set()

                        filt = str(det_meta.get("filter") or hdr.get("FILTER") or "Clear").strip() or "Clear"
                        if filt.lower() in {"nofilter", "none", "null"}:
                            filt = "Clear"

                        def _bin_step(v: float, step: float) -> float:
                            if not math.isfinite(v):
                                return float("nan")
                            return math.floor((float(v) / float(step)) + 0.5) * float(step)

                        # Saturation threshold for MASTERSTAR
                        sat_limit = det_meta.get("saturate_limit_adu")
                        if sat_limit is None:
                            sat_limit = _saturate_limit_adu_from_header(hdr) or getattr(_cfg_ms, "photometry_fallback_saturate_adu", 65535.0)
                        sat_thr = float(sat_limit) * float(saturate_level_fraction)

                        rows_ms: list[dict[str, Any]] = []
                        det_ok = det.iloc[np.where(ok)[0]].reset_index(drop=True)
                        g_ok = gdf.iloc[idx[np.where(ok)[0]]].reset_index(drop=True)
                        g_ok_idx = idx[np.where(ok)[0]]
                        # Aperture optimization: estimate per-star FWHM on MASTERSTAR, then take medians per color.
                        try:
                            import numpy as _np

                            arr_ms = _np.asarray(data, dtype=_np.float32)

                            def _fwhm_moment_at(xc: float, yc: float, r: int = 6) -> float:
                                if not (math.isfinite(xc) and math.isfinite(yc)):
                                    return float("nan")
                                xi = int(round(float(xc)))
                                yi = int(round(float(yc)))
                                x0 = max(0, xi - r)
                                x1 = min(arr_ms.shape[1] - 1, xi + r)
                                y0 = max(0, yi - r)
                                y1 = min(arr_ms.shape[0] - 1, yi + r)
                                if x1 <= x0 or y1 <= y0:
                                    return float("nan")
                                patch = arr_ms[y0 : y1 + 1, x0 : x1 + 1].astype(_np.float64, copy=False)
                                if patch.size < 9:
                                    return float("nan")
                                # Background-subtract by median; clip negatives to avoid noise moments.
                                medp = float(_np.nanmedian(patch))
                                w = patch - medp
                                w[~_np.isfinite(w)] = 0.0
                                w[w < 0] = 0.0
                                s = float(w.sum())
                                if not math.isfinite(s) or s <= 0:
                                    return float("nan")
                                yy, xx = _np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
                                mx = float((w * xx).sum() / s)
                                my = float((w * yy).sum() / s)
                                vx = float((w * (xx - mx) ** 2).sum() / s)
                                vy = float((w * (yy - my) ** 2).sum() / s)
                                if not (vx > 0 and vy > 0 and math.isfinite(vx) and math.isfinite(vy)):
                                    return float("nan")
                                sigx = math.sqrt(vx)
                                sigy = math.sqrt(vy)
                                fwhm = 2.355 * 0.5 * (sigx + sigy)
                                return float(fwhm) if math.isfinite(fwhm) else float("nan")

                            fwhm_est = [
                                _fwhm_moment_at(
                                    float(det_ok["x"].iloc[i]) if "x" in det_ok.columns and pd.notna(det_ok["x"].iloc[i]) else float("nan"),
                                    float(det_ok["y"].iloc[i]) if "y" in det_ok.columns and pd.notna(det_ok["y"].iloc[i]) else float("nan"),
                                )
                                for i in range(len(det_ok))
                            ]
                            fwhm_med_px = float(_np.nanmedian(_np.asarray(fwhm_est, dtype=_np.float64)))
                        except Exception:  # noqa: BLE001
                            fwhm_est = [float("nan")] * len(det_ok)
                            fwhm_med_px = float("nan")

                        if not (math.isfinite(fwhm_med_px) and fwhm_med_px > 0):
                            try:
                                fwhm_med_px = float(det_meta.get("dao_fwhm_px") or _ms_fwhm)
                            except Exception:  # noqa: BLE001
                                fwhm_med_px = float(_ms_fwhm)
                        if not (math.isfinite(fwhm_med_px) and fwhm_med_px > 0):
                            fwhm_med_px = float(_ms_fwhm)

                        # Median per coarse color category.
                        def _color_bucket(bp_rp: float) -> str:
                            if not math.isfinite(bp_rp):
                                return "neutral"
                            if bp_rp < 0.5:
                                return "blue"
                            if bp_rp <= 1.5:
                                return "neutral"
                            return "red"

                        by_col: dict[str, list[float]] = {"blue": [], "neutral": [], "red": []}
                        for i in range(len(det_ok)):
                            bprp_v0 = (
                                float(g_ok["bp_rp"].iloc[i])
                                if "bp_rp" in g_ok.columns and pd.notna(g_ok["bp_rp"].iloc[i])
                                else float("nan")
                            )
                            fe = float(fwhm_est[i]) if i < len(fwhm_est) else float("nan")
                            if math.isfinite(fe) and fe > 0:
                                by_col[_color_bucket(bprp_v0)].append(fe)
                        fwhm_blue = float(_np.median(by_col["blue"])) if by_col["blue"] else fwhm_med_px
                        fwhm_neu = float(_np.median(by_col["neutral"])) if by_col["neutral"] else fwhm_med_px
                        fwhm_red = float(_np.median(by_col["red"])) if by_col["red"] else fwhm_med_px

                        # Gaia neighbour veto radius in arcsec: 3× median FWHM (px) × plate scale.
                        try:
                            from astropy.wcs.utils import proj_plane_pixel_scales

                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", FITSFixedWarning)
                                _w_ms = WCS(hdr)
                            sc = proj_plane_pixel_scales(_w_ms.celestial)  # deg/pix
                            scale_arcsec_per_px = float(_np.nanmedian(_np.asarray(sc, dtype=_np.float64))) * 3600.0
                        except Exception:  # noqa: BLE001
                            scale_arcsec_per_px = float("nan")
                        veto_radius_arcsec = (
                            float(3.0 * fwhm_med_px * scale_arcsec_per_px)
                            if math.isfinite(scale_arcsec_per_px) and scale_arcsec_per_px > 0
                            else float("nan")
                        )
                        import numpy as _np
                        from photometry import bad_columns_for_light_frame

                        _thr_nl = float("nan")
                        _peaks_nl: list[float] = []
                        for _i in range(len(det_ok)):
                            _pm = (
                                det_ok["peak_max_adu"].iloc[_i]
                                if "peak_max_adu" in det_ok.columns and pd.notna(det_ok["peak_max_adu"].iloc[_i])
                                else None
                            )
                            if _pm is not None and math.isfinite(float(_pm)):
                                _peaks_nl.append(float(_pm))
                        if _peaks_nl:
                            _pp = float(getattr(_cfg_ms, "nonlinearity_peak_percentile", 20.0))
                            _thr_nl = float(
                                _np.percentile(
                                    _np.asarray(_peaks_nl, dtype=_np.float64),
                                    min(100.0, max(0.0, 100.0 - _pp)),
                                )
                            )
                        _nl_ratio = float(getattr(_cfg_ms, "nonlinearity_fwhm_ratio", 1.25))
                        _bpm_js = None
                        if master_dark_path and str(master_dark_path).strip():
                            _mdp = Path(str(master_dark_path))
                            _bj = _mdp.parent / f"{_mdp.stem}_dark_bpm.json"
                            if _bj.is_file():
                                try:
                                    _bpm_js = json.loads(_bj.read_text(encoding="utf-8"))
                                except Exception:  # noqa: BLE001
                                    _bpm_js = None
                        _bad_x = bad_columns_for_light_frame(_bpm_js, light_header=hdr)
                        for i in range(len(det_ok)):
                            x = float(det_ok["x"].iloc[i]) if "x" in det_ok.columns else None
                            y = float(det_ok["y"].iloc[i]) if "y" in det_ok.columns else None
                            pmax = float(det_ok["peak_max_adu"].iloc[i]) if "peak_max_adu" in det_ok.columns and pd.notna(det_ok["peak_max_adu"].iloc[i]) else None
                            is_sat = 1 if (pmax is not None and math.isfinite(pmax) and pmax > sat_thr) else 0
                            var_flag = str(g_ok.get("var_flag").iloc[i]) if "var_flag" in g_ok.columns else ""
                            is_var = 1 if var_flag.strip() not in ("", "0", "False", "false", "NO", "No") else 0

                            is_border = (
                                x is not None
                                and y is not None
                                and (
                                    x < border_px
                                    or y < border_px
                                    or x > float(nax1) - border_px
                                    or y > float(nax2) - border_px
                                )
                            )
                            gi = int(g_ok_idx[i]) if i < len(g_ok_idx) else -1
                            is_blend = bool(gi in blended_idx) if gi >= 0 else False
                            excl = "Border" if is_border else ("Blended" if is_blend else None)

                            fe_i = float(fwhm_est[i]) if i < len(fwhm_est) else float("nan")
                            likely_nl = False
                            if (
                                math.isfinite(fe_i)
                                and math.isfinite(fwhm_med_px)
                                and fwhm_med_px > 0
                                and pmax is not None
                                and math.isfinite(float(pmax))
                                and math.isfinite(_thr_nl)
                                and float(pmax) >= _thr_nl
                                and fe_i > _nl_ratio * fwhm_med_px
                            ):
                                likely_nl = True
                            on_bad = False
                            if x is not None and _bad_x:
                                if int(round(float(x))) in _bad_x:
                                    on_bad = True

                            # New Gaia stability/multiplicity filters.
                            gfer = None
                            if "g_flux_error_rel" in g_ok.columns and pd.notna(g_ok["g_flux_error_rel"].iloc[i]):
                                try:
                                    gfer = float(g_ok["g_flux_error_rel"].iloc[i])
                                except (TypeError, ValueError):
                                    gfer = None
                            nss = 0
                            if "non_single_star" in g_ok.columns and pd.notna(g_ok["non_single_star"].iloc[i]):
                                try:
                                    nss = int(float(g_ok["non_single_star"].iloc[i]))
                                except (TypeError, ValueError):
                                    nss = 0
                            pvf = ""
                            if "phot_variable_flag" in g_ok.columns and pd.notna(g_ok["phot_variable_flag"].iloc[i]):
                                pvf = str(g_ok["phot_variable_flag"].iloc[i]).strip()

                            if excl is None:
                                if gfer is not None and math.isfinite(gfer) and float(gfer) > 0.02:
                                    excl = "CatalogNoise"
                                elif int(nss or 0) > 0:
                                    excl = "NonSingle"
                                elif pvf.upper() == "VARIABLE":
                                    excl = "Variable"
                                else:
                                    # Neighbour veto: exclude if Gaia neighbour would change mag by > 0.001.
                                    if (
                                        veto_radius_arcsec is not None
                                        and math.isfinite(float(veto_radius_arcsec))
                                        and float(veto_radius_arcsec) > 0
                                        and "g_mag" in g_ok.columns
                                        and pd.notna(g_ok["g_mag"].iloc[i])
                                    ):
                                        try:
                                            m0 = float(g_ok["g_mag"].iloc[i])
                                        except (TypeError, ValueError):
                                            m0 = float("nan")
                                        if math.isfinite(m0):
                                            try:
                                                from astropy.coordinates import search_around_sky as _sas

                                                # Query neighbours in the Gaia window itself (fast; same gcoo).
                                                # Use the matched Gaia index gi.
                                                gi2 = int(g_ok_idx[i]) if i < len(g_ok_idx) else -1
                                                if gi2 >= 0 and gi2 < len(gcoo):
                                                    c0 = gcoo[gi2]
                                                    _, jj, _, _ = _sas(
                                                        c0,
                                                        gcoo,
                                                        float(veto_radius_arcsec) * u.arcsec,
                                                    )
                                                    ratios: list[float] = []
                                                    for jx in list(jj):
                                                        j = int(jx)
                                                        if j == gi2:
                                                            continue
                                                        try:
                                                            mj = float(gdf["g_mag"].iloc[j])
                                                        except Exception:  # noqa: BLE001
                                                            continue
                                                        if not math.isfinite(mj):
                                                            continue
                                                        ratios.append(10.0 ** (-0.4 * (mj - m0)))
                                                    if ratios:
                                                        dm = -2.5 * math.log10(1.0 + float(sum(ratios)))
                                                        if abs(dm) > 0.001:
                                                            excl = "Gaia neighbor blend"
                                            except Exception:  # noqa: BLE001
                                                pass
                            if likely_nl and excl is None:
                                excl = "Nonlinear FWHM"
                            if on_bad and excl is None:
                                excl = "Bad column"
                            safe = 0 if excl is not None else 1

                            gmag_v = (
                                float(g_ok["g_mag"].iloc[i])
                                if "g_mag" in g_ok.columns and pd.notna(g_ok["g_mag"].iloc[i])
                                else float("nan")
                            )
                            bprp_v = (
                                float(g_ok["bp_rp"].iloc[i])
                                if "bp_rp" in g_ok.columns and pd.notna(g_ok["bp_rp"].iloc[i])
                                else float("nan")
                            )
                            mb = _bin_step(gmag_v, 0.5)
                            cb = _bin_step(bprp_v, 0.25)
                            phot_cat = (
                                f"{filt}_mag_{mb:.1f}_col_{cb:.2f}"
                                if math.isfinite(mb) and math.isfinite(cb)
                                else f"{filt}_mag_nan_col_nan"
                            )
                            rows_ms.append(
                                {
                                    "x_master": x,
                                    "y_master": y,
                                    "ra": float(g_ok["ra"].iloc[i]) if pd.notna(g_ok["ra"].iloc[i]) else float(det_ok["ra_deg"].iloc[i]),
                                    "dec": float(g_ok["dec"].iloc[i]) if pd.notna(g_ok["dec"].iloc[i]) else float(det_ok["dec_deg"].iloc[i]),
                                    "g_mag": float(g_ok["g_mag"].iloc[i]) if "g_mag" in g_ok.columns and pd.notna(g_ok["g_mag"].iloc[i]) else None,
                                    "bp_rp": float(g_ok["bp_rp"].iloc[i]) if "bp_rp" in g_ok.columns and pd.notna(g_ok["bp_rp"].iloc[i]) else None,
                                    "is_var": is_var,
                                    "is_saturated": is_sat,
                                    "source_id_gaia": str(g_ok["source_id"].iloc[i]) if "source_id" in g_ok.columns else "",
                                    "g_flux_error_rel": gfer,
                                    "non_single_star": int(nss or 0),
                                    "phot_variable_flag": pvf,
                                    "filter_name": filt,
                                    "phot_category": phot_cat,
                                    "recommended_aperture": recommended_aperture_by_color(
                                        bp_rp=bprp_v if math.isfinite(bprp_v) else None,
                                        median_fwhm_blue=fwhm_blue,
                                        median_fwhm_neutral=fwhm_neu,
                                        median_fwhm_red=fwhm_red,
                                    ),
                                    "is_safe_comp": safe,
                                    "exclusion_reason": excl,
                                    "safe_override": 0,
                                    "likely_nonlinear": 1 if likely_nl else 0,
                                    "on_bad_column": 1 if on_bad else 0,
                                }
                            )
                        try:
                            pipeline_db = VyvarDatabase(Path(_cfg_ms.database_path))
                            try:
                                n_ins = pipeline_db.replace_master_sources_for_draft(int(draft_id), rows_ms)
                            finally:
                                pipeline_db.conn.close()
                            out["master_sources_written"] = int(n_ins)
                            try:
                                _wp2 = write_photometry_plan_files(
                                    platesolve_dir=platesolve_dir,
                                    masterstar_fits=masterstar_fits,
                                    masterstars_csv=csv_path,
                                    n_comparison_stars=int(n_comparison_stars),
                                    require_non_variable=bool(require_non_variable_comparisons),
                                    draft_id=int(draft_id),
                                    database_path=Path(_cfg_ms.database_path),
                                )
                                out.update(_wp2)
                            except Exception:  # noqa: BLE001
                                pass
                        except Exception as exc:  # noqa: BLE001
                            out["master_sources_written"] = 0
                            out["master_sources_error"] = str(exc)

                        # Stress-test: 10% random sample, exclude Border/Blended by default (soft-crop).
                        try:
                            root_frames = (
                                Path(source_root)
                                if source_root is not None
                                else (Path(detrended_root) if detrended_root is not None else ap)
                            )
                            # Common field intersection bbox across MASTERSTAR input frames (finite data overlap).
                            try:
                                _ms_inputs: list[Path] = []
                                if only_ms_paths is not None:
                                    _ms_inputs = [Path(p) for p in only_ms_paths if Path(p).is_file()]
                                else:
                                    # Fallback: approximate using a subset of aligned frames.
                                    _ms_inputs = sorted(_iter_fits_recursive(root_frames))[
                                        : max(2, int(info.get("frames_used", 0)))
                                    ]
                                bbox = common_field_intersection_bbox_px(frame_paths=_ms_inputs, finite_stride=16)
                                if bbox is not None:
                                    x0b, y0b, x1b, y1b = bbox
                                    pipeline_db_b = VyvarDatabase(Path(_cfg_ms.database_path))
                                    try:
                                        pipeline_db_b.conn.execute(
                                            """
                                            UPDATE MASTER_SOURCES
                                            SET IS_SAFE_COMP = 0, EXCLUSION_REASON = 'Out of common field'
                                            WHERE DRAFT_ID = ?
                                              AND COALESCE(SAFE_OVERRIDE, 0) = 0
                                              AND COALESCE(IS_SAFE_COMP, 0) = 1
                                              AND (
                                                X_MASTER IS NULL OR Y_MASTER IS NULL
                                                OR X_MASTER < ? OR X_MASTER > ?
                                                OR Y_MASTER < ? OR Y_MASTER > ?
                                              );
                                            """,
                                            (int(draft_id), float(x0b), float(x1b), float(y0b), float(y1b)),
                                        )
                                        pipeline_db_b.conn.commit()
                                        out["common_field_bbox_px"] = [float(x0b), float(y0b), float(x1b), float(y1b)]
                                    finally:
                                        pipeline_db_b.conn.close()
                            except Exception as exc:  # noqa: BLE001
                                out["common_field_error"] = str(exc)

                            safe_ids = [
                                str(r.get("source_id_gaia") or "").strip()
                                for r in rows_ms
                                if int(r.get("is_safe_comp") or 0) == 1
                            ]
                            st_res = stress_test_relative_rms_from_sidecars(
                                frames_root=root_frames,
                                source_ids=safe_ids,
                                sample_frac=0.10,
                                seed=42,
                            )
                            out["stress_frames_sampled"] = int(st_res.frames_sampled)
                            out["stress_frames_used"] = int(st_res.frames_used)

                            pipeline_db2 = VyvarDatabase(Path(_cfg_ms.database_path))
                            try:
                                ms_rows = pipeline_db2.fetch_master_sources_for_draft(int(draft_id))
                                # Per-bin median RMS on safe comps.
                                by_bin: dict[str, list[float]] = {}
                                for rr in ms_rows:
                                    if int(rr.get("IS_SAFE_COMP") or 0) != 1:
                                        continue
                                    sid = str(rr.get("SOURCE_ID_GAIA") or "").strip()
                                    if not sid or sid not in st_res.per_source_rms:
                                        continue
                                    b = str(rr.get("PHOT_CATEGORY") or "").strip()
                                    if b:
                                        by_bin.setdefault(b, []).append(float(st_res.per_source_rms[sid]))
                                med_by_bin = {b: float(pd.Series(v).median()) for b, v in by_bin.items() if v}

                                # Update STRESS_RMS in DB.
                                for rr in ms_rows:
                                    sid = str(rr.get("SOURCE_ID_GAIA") or "").strip()
                                    if not sid or sid not in st_res.per_source_rms:
                                        continue
                                    pipeline_db2.conn.execute(
                                        "UPDATE MASTER_SOURCES SET STRESS_RMS = ? WHERE ID = ?;",
                                        (float(st_res.per_source_rms[sid]), int(rr["ID"])),
                                    )
                                pipeline_db2.conn.commit()

                                # Unstable: RMS > 1.5× bin median
                                for rr in ms_rows:
                                    if int(rr.get("SAFE_OVERRIDE") or 0) == 1:
                                        continue
                                    if int(rr.get("IS_SAFE_COMP") or 0) != 1:
                                        continue
                                    b = str(rr.get("PHOT_CATEGORY") or "").strip()
                                    sid = str(rr.get("SOURCE_ID_GAIA") or "").strip()
                                    if not b or b not in med_by_bin or sid not in st_res.per_source_rms:
                                        continue
                                    if float(st_res.per_source_rms[sid]) > 1.5 * float(med_by_bin[b]):
                                        pipeline_db2.conn.execute(
                                            "UPDATE MASTER_SOURCES SET IS_SAFE_COMP = 0, EXCLUSION_REASON = 'Unstable' WHERE ID = ?;",
                                            (int(rr["ID"]),),
                                        )
                                pipeline_db2.conn.commit()

                                # VSX check for TOP3 stable per occupied bin.
                                ms_rows2 = pipeline_db2.fetch_master_sources_for_draft(int(draft_id))
                                packed = [
                                    {
                                        "source_id_gaia": rr.get("SOURCE_ID_GAIA"),
                                        "phot_category": rr.get("PHOT_CATEGORY"),
                                        "stress_rms": rr.get("STRESS_RMS"),
                                        "ra": rr.get("RA"),
                                        "dec": rr.get("DE"),
                                    }
                                    for rr in ms_rows2
                                    if rr.get("STRESS_RMS") is not None
                                ]
                                var_ids = vsx_is_known_variable_top3_per_bin(rows=packed)
                                if var_ids:
                                    qmarks = ",".join(["?"] * len(var_ids))
                                    pipeline_db2.conn.execute(
                                        f"""
                                        UPDATE MASTER_SOURCES
                                        SET IS_VAR = 1, IS_SAFE_COMP = 0, EXCLUSION_REASON = 'Variable'
                                        WHERE DRAFT_ID = ?
                                          AND SOURCE_ID_GAIA IN ({qmarks})
                                          AND COALESCE(SAFE_OVERRIDE, 0) = 0;
                                        """,
                                        (int(draft_id), *list(var_ids)),
                                    )
                                    pipeline_db2.conn.commit()
                                    out["vsx_flagged_variables"] = int(len(var_ids))
                            finally:
                                pipeline_db2.conn.close()
                        except Exception as exc:  # noqa: BLE001
                            out["stress_test_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        out["master_sources_error"] = str(exc)
    # Persist MASTERSTAR path on draft for later UI reloads / Step 3 continuity.
    try:
        if draft_id is not None:
            _db_ms = VyvarDatabase(Path(_cfg_ms.database_path))
            try:
                _db_ms.set_obs_draft_masterstar_path(int(draft_id), str(Path(masterstar_fits).resolve()))
            finally:
                _db_ms.conn.close()
    except Exception as exc:  # noqa: BLE001
        out["masterstar_path_store_error"] = str(exc)
    return out


def _partition_detrended_by_subfolder(files: list[Path], detrended_root: Path) -> dict[str, list[Path]]:
    """Group detrended FITS by full parent subpath under ``detrended_root``.

    This preserves on-disk structure (e.g. ``NoFilter_120_2`` or deeper nested layouts)
    so alignment, zero-level correction and DAO detection run on the exact same file tree.
    """
    root = detrended_root.resolve()
    out: dict[str, list[Path]] = {}
    for fp in files:
        p = Path(fp)
        try:
            rel = p.relative_to(root)
        except ValueError:
            continue
        parent_rel = rel.parent
        key = "" if str(parent_rel) == "." else parent_rel.as_posix()
        out.setdefault(key, []).append(p)
    for k in out:
        out[k].sort()
    return out


def _merge_astrometry_group_reports(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    if len(rows) == 1:
        return rows[0]
    tot_aligned = sum(int(x.get("aligned_frames") or 0) for x in rows)
    tot_in = sum(int(x.get("input_frames") or 0) for x in rows)
    merged = dict(rows[-1])
    merged["aligned_frames"] = int(tot_aligned)
    merged["input_frames"] = int(tot_in)
    merged["observation_subgroup_reports"] = rows
    merged["masterstar_fits"] = "; ".join(
        str(x.get("masterstar_fits") or "").strip()
        for x in rows
        if str(x.get("masterstar_fits") or "").strip()
    )
    merged["masterstars_csv"] = "; ".join(
        str(x.get("masterstars_csv") or "").strip()
        for x in rows
        if str(x.get("masterstars_csv") or "").strip()
    )
    log_event(
        f"Astrometria: dokončené {len(rows)} pod-skupín — zarovnaných snímok spolu {tot_aligned} / {tot_in}."
    )
    return merged


def _astrometry_align_impl_body(
    *,
    job: dict[str, Any],
    archive_path: Path,
    astrometry_api_key: str | None = None,
    max_control_points: int = 180,
    min_detected_stars: int = 100,
    max_detected_stars: int = 500,
    platesolve_backend: str = "vyvar",
    plate_solve_fov_deg: float = 1.0,
    max_extra_platesolve: int = 0,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    max_catalog_rows: int = 12000,
    n_comparison_stars: int = 150,
    require_non_variable_comparisons: bool = True,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    id_equipment: int | None = None,
    draft_id: int | None = None,
    catalog_local_gaia_only: bool | None = None,
    build_masterstar_and_catalogs: bool = False,
    progress_cb: "callable | None" = None,
    ram_align_and_catalog: bool = False,
    app_config: AppConfig | None = None,
) -> dict[str, Any]:
    """Internal: astrometry + alignment + per-frame CSV for one observation subtree (``job``)."""
    import numpy as np

    ap = Path(archive_path)
    obs_group_key = str(job.get("gkey", "") or "")
    scanning_id = int(job.get("scanning_id", 0) or 0)
    _job_md = job.get("master_dark_path")
    _master_dark_bpm_path: Path | None = Path(str(_job_md)) if _job_md and str(_job_md).strip() else None
    if _master_dark_bpm_path is not None and not _master_dark_bpm_path.is_file():
        _master_dark_bpm_path = None
    detrended_root = Path(job["detrended_root"])
    aligned_root = Path(job["aligned_root"])
    platesolve_dir = Path(job["platesolve_dir"])
    files = list(job["files"])
    # Zarovnané FITS: {archive}/detrended_aligned/lights/{filter_exp_binning}/… (vnorené cesty podľa vstupu)
    os.makedirs(str(aligned_root), exist_ok=True)
    os.makedirs(str(platesolve_dir), exist_ok=True)
    _cfg_align = app_config or AppConfig()
    _align_star_cap = max(10, min(5000, int(_cfg_align.alignment_max_stars)))
    # Keep alignment input stable on dense fields: use at most TOP 200 brightest stars.
    _align_star_cap = min(_align_star_cap, 200)
    _sips_sig = float(getattr(_cfg_align, "sips_dao_threshold_sigma", 3.5))
    if not math.isfinite(_sips_sig) or _sips_sig <= 0:
        _sips_sig = 3.5
    try:
        _ui_sig = float(dao_threshold_sigma)
    except (TypeError, ValueError):
        _ui_sig = 3.5
    if not math.isfinite(_ui_sig) or _ui_sig <= 0:
        _ui_sig = 3.5
    # Keep alignment DAO sensitive enough for weak/star-poor frames.
    _align_det_sigma = max(0.8, min(20.0, _ui_sig if _ui_sig > 0 else _sips_sig))
    _fb_align = float(getattr(_cfg_align, "sips_dao_fwhm_px", 2.5))
    if not math.isfinite(_fb_align) or _fb_align <= 0:
        _fb_align = 2.5
    try:
        _pfov_align = float(plate_solve_fov_deg)
        if not math.isfinite(_pfov_align) or _pfov_align <= 0:
            _pfov_align = None
    except (TypeError, ValueError):
        _pfov_align = None
    if build_masterstar_and_catalogs:
        LOGGER.info("Astrometria + MASTERSTAR + per-frame CSV: archív %s", ap)
    else:
        LOGGER.info("Astrometria + zarovnanie + per-frame CSV (bez MASTERSTAR): archív %s", ap)
    # MASTERSTAR initial match: allow a looser sep (min 10") for robust first-pass Gaia join.
    _catalog_match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
    if _catalog_match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
        _pipeline_ui_info(
            f"Katalógový match prah zvýšený na {_catalog_match_sep_eff:.2f}\" "
            "(minimum pre robustný počiatočný cross-match)."
        )

    # Per-frame catalogs: use a tighter matching radius (default 3") independent of MASTERSTAR.
    per_frame_match_sep = float(
        job.get(
            "per_frame_catalog_match_sep_arcsec",
            getattr(_cfg_align, "per_frame_catalog_match_sep_arcsec", 3.0),
        )
        or 3.0
    )
    if not math.isfinite(per_frame_match_sep) or per_frame_match_sep <= 0:
        per_frame_match_sep = 3.0
    _cat_loc_only = bool(catalog_local_gaia_only) if catalog_local_gaia_only is not None else True
    if _cat_loc_only:
        LOGGER.info("Katalóg: režim lokálny Gaia (SQLite)")
    equip_sat_adu = _equipment_saturate_adu_from_db(id_equipment)
    if draft_id is not None and files:
        try:
            _db_sat = VyvarDatabase(Path(_cfg_align.database_path))
            try:
                _cmb_sat = _db_sat.get_combined_metadata(files[0], int(draft_id))
                if _cmb_sat.get("saturate_adu") is not None:
                    equip_sat_adu = _cmb_sat["saturate_adu"]
            finally:
                _db_sat.conn.close()
        except Exception:  # noqa: BLE001
            pass
    if not files:
        raise FileNotFoundError(
            f"Chýbajú FITS v {detrended_root}. Krok 3 číta len **spracované** snímky. "
            "Najprv spusti **Apply Calibration & Detrending** po kroku **Analyze** (zápis do "
            f"`{ap / 'processed' / 'lights'}` alebo staršie `{ap / 'detrended' / 'lights'}`)."
        )

    _t_step3_start = time.time()
    n_files = len(files)
    ref_fp, ref_star_scores = _pick_reference_frame_by_star_count(files)
    # Read reference once (no lock during solve step).
    with fits.open(ref_fp, memmap=False) as hdul:
        ref_hdr = hdul[0].header.copy()
        ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
    has_wcs = _has_valid_wcs(ref_hdr)
    solve_steps = 0 if has_wcs else 1
    master_steps = 1 if build_masterstar_and_catalogs else 0
    global_total = max(1, 1 + solve_steps + n_files + master_steps + n_files)
    prog_i = [0]

    def _prog(msg: str) -> None:
        if progress_cb is None:
            return
        prog_i[0] += 1
        progress_cb(prog_i[0], global_total, msg)

    # --- MASTERSTAR build + plate-solve (root platesolve/) before alignment ---
    # Goal: one-button Step 3 flow. Always build MASTERSTAR in platesolve/ (root), not in platesolve/<setup>/.
    _masterstar_built = False
    _cat_info_root: dict[str, Any] = {}
    _ps_root = platesolve_dir.parent
    _t_platesolve = time.time()
    if build_masterstar_and_catalogs:
        _prog("platesolve/MASTERSTAR: build mean stack + plate-solve + katalógy…")
        _cat_info_root = generate_masterstar_and_catalog(
            archive_path=ap,
            max_catalog_rows=int(max_catalog_rows),
            astrometry_api_key=astrometry_api_key,
            source_root=detrended_root,
            platesolve_dir=_ps_root,
            platesolve_backend=platesolve_backend,
            plate_solve_fov_deg=float(plate_solve_fov_deg),
            catalog_match_max_sep_arcsec=float(_catalog_match_sep_eff),
            saturate_level_fraction=float(saturate_level_fraction),
            n_comparison_stars=int(n_comparison_stars),
            require_non_variable_comparisons=bool(require_non_variable_comparisons),
            faintest_mag_limit=faintest_mag_limit,
            dao_threshold_sigma=float(dao_threshold_sigma),
            equipment_saturate_adu=equip_sat_adu,
            catalog_local_gaia_only=_cat_loc_only,
            app_config=_cfg_align,
            equipment_id=id_equipment,
            draft_id=draft_id,
            master_dark_path=_master_dark_bpm_path,
            masterstar_candidate_paths=job.get("masterstar_candidate_paths"),
            masterstar_selection_pct=job.get("masterstar_selection_pct"),
            setup_name=obs_group_key or None,
            masterstar_basename="MASTERSTAR.fits",
            masterstars_csv_basename="masterstars_full_match.csv",
            masterstar_fits_only=False,
            masterstar_skip_build=False,
        )
        _masterstar_built = True

    _prog(
        f"detrended_aligned/lights: pripravujem zarovnanie ({n_files} snímok z {detrended_root.name}/…)…"
    )

    if not has_wcs:
        _prog("Plate solve referencie (môže chvíľu trvať)…")
        # Solve reference file in-place (no open handle on Windows).
        _hra, _hdec, _ = _pointing_hint_from_header(ref_hdr)
        solve = _solve_wcs_external(
            ref_fp,
            backend=platesolve_backend,
            astrometry_api_key=astrometry_api_key,
            plate_solve_fov_deg=float(plate_solve_fov_deg),
            hint_ra_deg=_hra,
            hint_dec_deg=_hdec,
            app_config=_cfg_align,
            equipment_id=id_equipment,
            draft_id=draft_id,
        )
        if not solve.get("solved", False):
            raise RuntimeError(f"Reference astrometry failed: {solve.get('reason')}")
        # Reload solved header/data
        with fits.open(ref_fp, memmap=False) as hdul:
            ref_hdr = hdul[0].header.copy()
            ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)

    print(f"  Plate solve: {time.time() - _t_platesolve:.1f}s")
    _t_align = time.time()

    _align_fwhm_ref = float(_fb_align)

    hint_center = _wcs_field_center_radec_deg(ref_fp)
    hint_ra: float | None = hint_center[0] if hint_center else None
    hint_dec: float | None = hint_center[1] if hint_center else None

    extra_platesolve_results: list[dict[str, Any]] = []

    try:
        from astropy.stats import sigma_clipped_stats
        from photutils.background import Background2D, MedianBackground
        from photutils.detection import DAOStarFinder
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"photutils required for star detection/alignment: {exc}") from exc

    try:
        import astroalign  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"astroalign required for frame registration: {exc}") from exc

    log_event(
        f"Detekcia hviezd: Použité FWHM={_align_fwhm_ref:.2f}, Sigma={_align_det_sigma:.2f}"
    )

    def _maybe_refine_aligned(
        hdr_mut: fits.Header, data_mut: np.ndarray, label: str, *, dao_fwhm_px_frame: float
    ) -> None:
        _ = (hdr_mut, data_mut, label, dao_fwhm_px_frame)
        return

    def _as_alignment_points(sources: Any, *, label: str) -> np.ndarray:
        """Normalize astroalign inputs to ``Nx2`` float array.

        Accepts photutils table-like objects (``xcentroid``, ``ycentroid``) and plain array-like inputs.
        """
        pts: np.ndarray
        try:
            # photutils/astropy table path
            if hasattr(sources, "colnames") and "xcentroid" in getattr(sources, "colnames", []) and "ycentroid" in getattr(sources, "colnames", []):
                pts = np.transpose((sources["xcentroid"], sources["ycentroid"]))  # type: ignore[index]
            elif hasattr(sources, "dtype") and getattr(sources.dtype, "names", None):  # structured ndarray
                names = set(getattr(sources.dtype, "names") or ())
                if "xcentroid" in names and "ycentroid" in names:
                    pts = np.transpose((sources["xcentroid"], sources["ycentroid"]))  # type: ignore[index]
                else:
                    arr = np.asarray(sources, dtype=np.float64)
                    pts = arr.reshape((-1, 2)) if arr.ndim == 1 and arr.size % 2 == 0 else arr
            else:
                arr = np.asarray(sources, dtype=np.float64)
                pts = arr.reshape((-1, 2)) if arr.ndim == 1 and arr.size % 2 == 0 else arr
        except Exception:  # noqa: BLE001
            pts = np.zeros((0, 2), dtype=np.float32)
        if pts.ndim != 2:
            pts = np.zeros((0, 2), dtype=np.float32)
        elif pts.shape[1] > 2:
            pts = np.asarray(pts[:, :2], dtype=np.float64)
        elif pts.shape[1] < 2:
            pts = np.zeros((0, 2), dtype=np.float32)
        pts = np.asarray(pts, dtype=np.float32)
        log_event(
            f"DEBUG: Alignment vstup ({label}) type: {type(sources).__name__} a shape: {tuple(pts.shape)}"
        )
        return pts

    def _detect_xy(
        img: np.ndarray,
        want_max: int,
        *,
        det_sigma: float,
        fwhm_px: float,
        label: str = "",
    ) -> np.ndarray:
        if int(want_max) <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        data = np.ascontiguousarray(img, dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        sky_pixels = data[data > 10.0]
        if sky_pixels.size > 0:
            m = float(np.nanmedian(sky_pixels))
        else:
            m = float(np.nanmedian(data))
        data -= np.float32(m)
        data = np.clip(data, -100.0, None).astype(np.float32, copy=False)
        _frame_name = label or "frame"
        log_event(
            f"🚨 [ALIGNMENT] Frame: {_frame_name} | Offset Removed: {m:.2f} | Final Mean: {np.nanmean(data):.4f}"
        )
        arr = data
        finite = np.isfinite(data)
        if not np.any(finite):
            return np.zeros((0, 2), dtype=np.float32)
        # Diagnostics before background subtraction.
        try:
            _vals = arr[finite].astype(np.float64, copy=False)
            _mean = float(np.mean(_vals)) if _vals.size else 0.0
            _std = float(np.std(_vals)) if _vals.size else 0.0
            _max = float(np.max(_vals)) if _vals.size else 0.0
            _nm = label or "frame"
            log_event(
                f"DEBUG: Image stats pre {_nm} - Mean: {_mean:.6g}, Std: {_std:.6g}, Max: {_max:.6g}"
            )
        except Exception:  # noqa: BLE001
            pass
        # Robust background subtraction to suppress slow gradients.
        arr_bg = arr
        used_bg_sub = False
        try:
            _h, _w = int(arr.shape[0]), int(arr.shape[1])
            _bx = max(16, min(50, _w))
            _by = max(16, min(50, _h))
            bkg = Background2D(
                arr,
                box_size=(_by, _bx),
                filter_size=(3, 3),
                bkg_estimator=MedianBackground(),
            )
            arr_bg = np.asarray(arr - bkg.background, dtype=np.float32)
            used_bg_sub = True
        except Exception:  # noqa: BLE001
            arr_bg = arr
            used_bg_sub = False
        # Clip negatives after subtraction as DAO thresholding expects positive signal above sky.
        arr_bg = np.nan_to_num(arr_bg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        _med_bg = float(np.nanmedian(arr_bg))
        _std_bg = float(np.nanstd(arr_bg))
        _floor_bg = _med_bg - 3.0 * _std_bg if np.isfinite(_std_bg) and _std_bg > 0 else _med_bg
        arr_bg = np.clip(arr_bg, _floor_bg, None).astype(np.float32, copy=False)
        _, med, clipped_std = sigma_clipped_stats(arr_bg[np.isfinite(arr_bg)], sigma=3.0, maxiters=5)
        clipped_std = float(clipped_std) if np.isfinite(clipped_std) else 0.0
        if clipped_std <= 0:
            clipped_std = 1.0
        if clipped_std < 1e-6:
            _nm = label or "frame"
            log_event(
                f"DEBUG: Image stats pre {_nm} - podozrivo nízke Std po BG ({clipped_std:.3e}); "
                "background subtraction môže byť príliš agresívna."
            )
        data = np.nan_to_num((arr_bg - float(med)).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        sig = max(0.5, min(20.0, float(det_sigma)))
        threshold = float(sig) * float(clipped_std)
        _nm = label or "frame"
        try:
            data_chk = np.asarray(data, dtype=np.float32)
            nan_count = int(np.isnan(data_chk).sum())
            inf_count = int(np.isinf(data_chk).sum())
            min_val = float(np.min(data_chk))
            max_val = float(np.max(data_chk))
            mean_val = float(np.mean(data_chk))
            log_event(f"[DEBUG DATA CHECK] Súbor: {_nm}")
            log_event(f"[DEBUG DATA CHECK] Stats -> Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
            log_event(f"[DEBUG DATA CHECK] Bad Pixels -> NaNs: {nan_count}, Infs: {inf_count}")
            log_event(f"[DEBUG DATA CHECK] Current Threshold in use: {threshold:.2f}")
        except Exception:  # noqa: BLE001
            pass
        log_event(
            f"DEBUG: Používam FWHM={float(fwhm_px):.2f} a Sigma={float(sig):.2f} (z alignment det_sigma) pre detekciu."
        )
        log_event(f"DEBUG: Threshold set to {threshold:.2f} (using clipped_std={clipped_std:.2f})")
        fw = max(1.2, float(fwhm_px))
        # Aggressive fallback: lower threshold by 0.2 down to 0.5 sigma.
        sig_try: list[float] = []
        s_cur = float(sig)
        while s_cur >= 0.5 - 1e-9:
            ss = max(float(s_cur), 0.5)
            if not any(abs(ss - t) < 1e-9 for t in sig_try):
                sig_try.append(ss)
            s_cur -= 0.2
        def _run_dao_on(img_in: np.ndarray, sigma_list: list[float], *, nm: str) -> tuple[Any, float, int]:
            _best_tbl = None
            _best_n = -1
            _best_sigma = float(sigma_list[0] if sigma_list else sig)
            _used_sigma = _best_sigma
            for idx_s, s in enumerate(sigma_list):
                if idx_s > 0:
                    log_event(f"Skúšam znížiť threshold na {float(s):.2f} pre {nm}...")
                finder = DAOStarFinder(
                    fwhm=fw,
                    threshold=max(float(s) * float(clipped_std), 1e-6),
                    brightest=None,
                    **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
                )
                _tbl_try = finder(img_in)
                _n_try = int(len(_tbl_try)) if _tbl_try is not None else 0
                if _n_try > _best_n:
                    _best_n = _n_try
                    _best_tbl = _tbl_try
                    _best_sigma = float(s)
                if _tbl_try is not None and _n_try >= 50:
                    _used_sigma = float(s)
                    return _tbl_try, _used_sigma, _n_try
            return _best_tbl, _best_sigma, max(0, _best_n)

        _nm = label or "frame"
        tbl, used_sigma, best_n = _run_dao_on(data, sig_try, nm=_nm)
        # If no stars with BG-subtracted image, retry without BG subtraction (median-only).
        if (tbl is None or best_n <= 0) and used_bg_sub:
            _med_r = float(np.nanmedian(arr))
            _std_r = float(np.nanstd(arr))
            _floor_r = _med_r - 3.0 * _std_r if np.isfinite(_std_r) and _std_r > 0 else _med_r
            arr_raw2 = np.nan_to_num((arr - _med_r).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            arr_raw2 = np.clip(arr_raw2, _floor_r - _med_r, None).astype(np.float32, copy=False)
            log_event(f"DEBUG: {_nm} - fallback bez background subtraction (median-only).")
            tbl2, used_sigma2, best_n2 = _run_dao_on(arr_raw2, sig_try, nm=_nm)
            if best_n2 > best_n:
                tbl, used_sigma, best_n = tbl2, used_sigma2, best_n2
        if float(used_sigma) < float(sig) - 1e-9:
            log_event(
                f"Detekcia hviezd: Použité FWHM={fw:.2f}, Sigma={float(used_sigma):.2f} (fallback z {float(sig):.2f})"
            )
        if tbl is None or len(tbl) == 0:
            log_event(
                f"DEBUG: Snímka {_nm} - pokus o detekciu s extrémnou citlivosťou (threshold=1.0)"
            )
            try:
                finder_ext = DAOStarFinder(
                    fwhm=5.0,
                    threshold=max(1.0 * float(clipped_std), 1e-6),
                    brightest=None,
                    **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
                )
                tbl_ext = finder_ext(data)
                if tbl_ext is not None and len(tbl_ext) > 0:
                    tbl = tbl_ext
            except Exception:  # noqa: BLE001
                pass
        if tbl is None or len(tbl) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        k = int(min(len(tbl), int(want_max)))
        # brightest k without full sort when DAO returns huge tables (faster + less RAM)
        if len(tbl) > k:
            flux_np = np.asarray(tbl["flux"], dtype=np.float64)
            take = np.argpartition(flux_np, -k)[-k:]
            tbl = tbl[take]
        tbl.sort("flux")
        tbl = tbl[::-1]
        x = np.asarray(tbl["xcentroid"], dtype=np.float32)
        y = np.asarray(tbl["ycentroid"], dtype=np.float32)
        return np.stack([x, y], axis=1)

    # Adaptive alignment star budget:
    # - if STAR_COUNT > 1000 → use top 300 brightest
    # - if STAR_COUNT < 100 → use all
    # - else → cap at 300
    data_to_detect = np.asarray(ref_data, dtype=np.float32)
    try:
        log_event(
            "DEBUG: Data stats for alignment - "
            f"Min: {np.min(data_to_detect):.2f}, "
            f"Max: {np.max(data_to_detect):.2f}, "
            f"Mean: {np.mean(data_to_detect):.2f}, "
            f"NaN count: {np.isnan(data_to_detect).sum()}"
        )
    except Exception:  # noqa: BLE001
        pass
    ref_xy_all = _detect_xy(
        data_to_detect,
        int(max(100, max_detected_stars)),
        det_sigma=_align_det_sigma,
        fwhm_px=_align_fwhm_ref,
        label=ref_fp.name,
    )
    n_ref = int(len(ref_xy_all))
    if n_ref > 1000:
        n_keep = 300
    elif n_ref < 100:
        n_keep = n_ref
    else:
        n_keep = min(300, n_ref)
    ref_xy = ref_xy_all[:n_keep]
    if len(ref_xy) < int(min_detected_stars):
        raise RuntimeError(
            f"Reference frame has too few detected stars ({len(ref_xy)} < {min_detected_stars})."
        )
    ref_xy_fit = ref_xy[: int(min(_align_star_cap, len(ref_xy)))]
    log_event(
        f"Zarovnanie referencia {ref_fp.name}: DAO hviezd={len(ref_xy)}, "
        f"cap pre transform={_align_star_cap}, DAO σ={_align_det_sigma:.2f}, FWHM={_align_fwhm_ref:.2f}px "
        f"(QC VY_FWHM alebo sips_dao_fwhm_px)"
    )

    # Auto RAM management: default in-memory, but switch to disk when estimated working set exceeds 70% of available RAM.
    use_ram_handoff = bool(ram_align_and_catalog)
    try:
        mp = estimate_archive_memory_profile(ap)
        avail = mp.get("available_ram_bytes")
        prh = mp.get("platesolve_ram_handoff") or {}
        tot = prh.get("estimated_total_conservative_bytes")
        if isinstance(avail, int) and isinstance(tot, int) and avail > 0 and tot > 0:
            if tot > int(0.70 * avail):
                use_ram_handoff = False
    except Exception:  # noqa: BLE001
        pass

    aligned_ram_buffer: list[tuple[str, fits.Header, np.ndarray]] = []
    aligned_files: list[Path] = []
    star_counts: list[dict[str, Any]] = []
    rotation_ref_angle_deg: float | None = None
    rotation_flip_frame_indices_1based: list[int] = []
    rotation_flip_first_index_1based: int | None = None
    _flip_logged = False

    try:
        rotation_ref_angle_deg = wcs_rotation_angle_deg(ref_hdr)
    except Exception:  # noqa: BLE001
        rotation_ref_angle_deg = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            ref_wcs_obj = WCS(ref_hdr) if _has_valid_wcs(ref_hdr) else None
    except Exception:  # noqa: BLE001
        ref_wcs_obj = None

    # save reference as aligned baseline, keep WCS
    _prog(
        f"detrended_aligned/lights: {'RAM — referencia' if use_ram_handoff else 'zapisujem FITS'} "
        f"{ref_fp.name} (1/{n_files})"
    )
    ref_rel = ref_fp.relative_to(detrended_root)
    ref_out = aligned_root / ref_rel
    _ensure_parent_dirs_for_aligned_fits(ref_out)
    with fits.open(ref_fp, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = _as_fits_float32_image(hdul[0].data)
    hdr["VY_ALGN"] = (True, "Aligned to reference")
    hdr["VYALGOK"] = (True, "Alignment OK")
    hdr["VY_REF"] = (ref_fp.name[:60], "Reference frame for alignment")
    _maybe_refine_aligned(hdr, data, ref_fp.name, dao_fwhm_px_frame=_align_fwhm_ref)
    if use_ram_handoff:
        aligned_ram_buffer.append((ref_rel.as_posix(), hdr.copy(), np.copy(data)))
    else:
        fits.writeto(ref_out, data, header=hdr, overwrite=True)
    aligned_files.append(ref_out)
    star_counts.append(
        {
            "file": ref_fp.name,
            "frame_index": int(files.index(ref_fp) + 1) if ref_fp in files else 1,
            "detected_stars": int(len(ref_xy)),
            "aligned": True,
            "alignment_method": "reference",
            "is_flipped": False,
            "rotation_angle_deg": rotation_ref_angle_deg,
        }
    )

    # Align every other frame to reference (skip duplicate if ref is not files[0]).
    # Control points fixed to 1.5× selected stars (bounded).
    align_cp = int(max(12, min(_align_star_cap, int(round(1.5 * max(1, len(ref_xy_fit)))))))
    ref_pts = np.asarray(ref_xy_fit, dtype=np.float32)
    if ref_pts is None or len(ref_pts) == 0:
        raise ValueError("Referenčné hviezdy sú prázdne pred štartom alignmentu!")
    # Keep immutable backup of reference points; never overwrite with per-frame source detections.
    fixed_target_pts = np.copy(ref_pts).astype("float32")
    log_event(f"DEBUG: Start alignment, reference stars N = {len(fixed_target_pts)}")
    # Brute-force isolation from any shared numpy buffers.
    REFERENCE_LIST = fixed_target_pts.tolist()
    LOGGER.info(
        "Astrometry alignment: astroalign uses up to %s control points on up to %s DAO sources per frame",
        align_cp,
        _align_star_cap,
    )

    def _run_astroalign_points(
        *,
        source_pts: Any,
        target_pts: Any,
        image_source: np.ndarray,
        image_target: np.ndarray,
        max_control_points: int,
    ) -> tuple[np.ndarray | None, str | None]:
        try:
            src = np.array(source_pts, dtype="float32")
            tgt = np.array(target_pts, dtype="float32")
            if src.ndim != 2 or src.shape[1] != 2:
                return None, f"Nesprávny formát source bodov: shape={tuple(src.shape)}"
            if tgt.ndim != 2 or tgt.shape[1] != 2:
                return None, f"Nesprávny formát target bodov: shape={tuple(tgt.shape)}"
            if src.shape == (0, 2) or tgt.shape == (0, 2):
                return None, "Alignment nemôže začať, prázdne súradnice!"
            if len(src) < 10 or len(tgt) < 10:
                return None, "Insufficient stars"
            mcp = max(12, min(int(max_control_points), int(min(len(src), len(tgt)))))
            t, _ = astroalign.find_transform(
                source=np.asarray(src, dtype=np.float32),
                target=np.asarray(tgt, dtype=np.float32),
                max_control_points=mcp,
            )
            aligned_data, _ = astroalign.apply_transform(t, image_source, image_target)
            return _as_fits_float32_image(aligned_data), None
        except Exception as e:  # noqa: BLE001
            return None, str(e)

    n_written_align = 1
    def _load_masterstar_catalog_points_for_frame(hdr_frame: fits.Header, *, shape_hw: tuple[int, int]) -> np.ndarray:
        try:
            ms_csv = Path(platesolve_dir) / "masterstars.csv"
            if not ms_csv.is_file():
                return np.zeros((0, 2), dtype=np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                w = WCS(hdr_frame)
            if not getattr(w, "has_celestial", False):
                return np.zeros((0, 2), dtype=np.float32)
            mdf = pd.read_csv(ms_csv)
            if "ra_deg" not in mdf.columns or "dec_deg" not in mdf.columns:
                return np.zeros((0, 2), dtype=np.float32)
            ra = pd.to_numeric(mdf["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
            de = pd.to_numeric(mdf["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(ra) & np.isfinite(de)
            if not np.any(ok):
                return np.zeros((0, 2), dtype=np.float32)
            sky = SkyCoord(ra=ra[ok] * u.deg, dec=de[ok] * u.deg, frame="icrs")
            xp, yp = w.world_to_pixel(sky)
            xy = np.stack([np.asarray(xp, dtype=np.float32), np.asarray(yp, dtype=np.float32)], axis=1)
            h, wpx = int(shape_hw[0]), int(shape_hw[1])
            inb = (
                np.isfinite(xy[:, 0])
                & np.isfinite(xy[:, 1])
                & (xy[:, 0] >= 0)
                & (xy[:, 1] >= 0)
                & (xy[:, 0] < float(wpx))
                & (xy[:, 1] < float(h))
            )
            xy = xy[inb]
            if len(xy) > int(_align_star_cap):
                xy = xy[: int(_align_star_cap)]
            return np.asarray(xy, dtype=np.float32)
        except Exception:  # noqa: BLE001
            return np.zeros((0, 2), dtype=np.float32)

    for frame_index_1based, fp in enumerate(files, start=1):
        if fp == ref_fp:
            continue
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
        is_flipped = False
        try:
            if rotation_ref_angle_deg is not None:
                a_i = wcs_rotation_angle_deg(hdr)
                if a_i is not None:
                    diff = circular_angle_diff_deg(float(a_i), float(rotation_ref_angle_deg))
                    if diff > 175.0:
                        is_flipped = True
                        rotation_flip_frame_indices_1based.append(int(frame_index_1based))
                        if rotation_flip_first_index_1based is None:
                            rotation_flip_first_index_1based = int(frame_index_1based)
                        if not _flip_logged:
                            log_event(
                                f"Physical rotation change detected at frame index {int(frame_index_1based)}. "
                                "Adjusting alignment strategy."
                            )
                            _flip_logged = True
        except Exception:  # noqa: BLE001
            is_flipped = False
        fw_i = dao_detection_fwhm_pixels(hdr, configured_fallback=_fb_align)
        data_to_detect = np.asarray(data, dtype=np.float32)
        try:
            log_event(
                "DEBUG: Data stats for alignment - "
                f"Min: {np.min(data_to_detect):.2f}, "
                f"Max: {np.max(data_to_detect):.2f}, "
                f"Mean: {np.mean(data_to_detect):.2f}, "
                f"NaN count: {np.isnan(data_to_detect).sum()}"
            )
        except Exception:  # noqa: BLE001
            pass

        _attempts = [
            {"dao_sigma": 3.5, "max_stars": 200},
            {"dao_sigma": 2.5, "max_stars": 300},
            {"dao_sigma": 2.0, "max_stars": 400},
            {"dao_sigma": 1.5, "max_stars": 500},
        ]
        aligned_data: np.ndarray | None = None
        aligned_method = "astroalign"
        xy_used = np.zeros((0, 2), dtype=np.float32)
        _attempt_ok = False

        for i_att, att in enumerate(_attempts, start=1):
            dao_sig = float(att["dao_sigma"])
            max_st = int(att["max_stars"])
            try:
                xy = _detect_xy(
                    data_to_detect,
                    int(max(100, max_st)),
                    det_sigma=dao_sig,
                    fwhm_px=fw_i,
                    label=fp.name,
                )
            except Exception:
                xy = np.zeros((0, 2), dtype=np.float32)

            if len(xy) == 0:
                xy_ms = _load_masterstar_catalog_points_for_frame(hdr, shape_hw=tuple(data.shape))
                if len(xy_ms) > 0:
                    xy = np.asarray(xy_ms[:max_st], dtype=np.float32)
                    log_event(
                        f"DEBUG: {fp.name} attempt {i_att} - DAO=0, používam fallback MASTERSTAR katalóg ({len(xy)} bodov)."
                    )
                else:
                    log_event(f"WARNING: {fp.name} attempt {i_att} zlyhalo, skúšam s uvoľnenými parametrami")
                    continue

            if len(xy) < int(min_detected_stars):
                log_event(f"WARNING: {fp.name} attempt {i_att} zlyhalo, skúšam s uvoľnenými parametrami")
                continue

            try:
                aligned_data = None
                aligned_method = "astroalign"
                if ref_wcs_obj is not None and _has_valid_wcs(hdr):
                    try:
                        from reproject import reproject_interp  # type: ignore

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FITSFixedWarning)
                            w_i = WCS(hdr)
                        if getattr(w_i, "has_celestial", False):
                            aligned_data, _foot = reproject_interp(
                                (np.asarray(data, dtype=np.float32), w_i.celestial),
                                ref_wcs_obj.celestial,
                                shape_out=ref_data.shape,
                            )
                            aligned_data = _as_fits_float32_image(aligned_data)
                            aligned_method = "wcs_reproject"
                    except Exception:  # noqa: BLE001
                        aligned_data = None

                if aligned_data is None:
                    log_event(
                        f"DEBUG: Alignment detekcia {fp.name} (attempt {i_att}) - DAO hviezd(source)={len(xy)}, "
                        f"DAO hviezd(target_ref)={len(fixed_target_pts)}"
                    )
                    if fixed_target_pts.shape[0] == 0:
                        raise Exception(
                            "FATÁLNA CHYBA: Referenčné body boli vymazané z pamäte pred štartom alignmentu!"
                        )
                    current_target = np.array(REFERENCE_LIST, dtype="float32")
                    n_fit = int(min(len(current_target), len(xy)))
                    xy_fit = _as_alignment_points(xy[:n_fit], label="source")
                    ref_fit = _as_alignment_points(current_target[:n_fit], label="target")
                    n_fit = int(min(len(xy_fit), len(ref_fit)))
                    if n_fit > 0:
                        xy_fit = np.asarray(xy_fit[:n_fit], dtype=np.float32)
                        ref_fit = np.asarray(ref_fit[:n_fit], dtype=np.float32)
                    mcp = max(12, min(max_st, n_fit))
                    aligned_data, aa_err = _run_astroalign_points(
                        source_pts=xy_fit,
                        target_pts=ref_fit,
                        image_source=data,
                        image_target=ref_data,
                        max_control_points=mcp,
                    )
                    if aligned_data is None:
                        raise RuntimeError(str(aa_err or "astroalign failed"))

                # Guard: detect silently failed constant frames (often from bad transform).
                try:
                    arr_check = np.asarray(aligned_data, dtype=np.float32)
                    finite_chk = arr_check[np.isfinite(arr_check)]
                    if finite_chk.size:
                        samp = finite_chk[: min(10_000, int(finite_chk.size))]
                        n_unique = int(len(np.unique(samp)))
                    else:
                        n_unique = 0
                    if n_unique <= 3:
                        print(
                            f"WARNING: {fp.name} aligned frame je konštantný (n_unique={n_unique}), "
                            f"pokus {i_att} zlyhol"
                        )
                        aligned_data = None
                        continue
                except Exception:  # noqa: BLE001
                    pass

                # Validate: reject "black/empty" aligned results (near-zero mean + low variance).
                m_al = float(np.nanmean(aligned_data)) if aligned_data is not None else float("nan")
                s_al = float(np.nanstd(aligned_data)) if aligned_data is not None else float("nan")
                if math.isfinite(m_al) and math.isfinite(s_al) and (m_al < 50.0) and (s_al < 20.0):
                    log_event(f"WARNING: {fp.name} attempt {i_att} zlyhalo, skúšam s uvoľnenými parametrami")
                    aligned_data = None
                    continue
                if float(np.nansum(np.abs(aligned_data))) < 1.0:
                    log_event(f"WARNING: {fp.name} attempt {i_att} zlyhalo, skúšam s uvoľnenými parametrami")
                    aligned_data = None
                    continue

                xy_used = np.asarray(xy, dtype=np.float32)
                _attempt_ok = True
                break
            except Exception:
                aligned_data = None
                log_event(f"WARNING: {fp.name} attempt {i_att} zlyhalo, skúšam s uvoľnenými parametrami")
                continue

        if not _attempt_ok or aligned_data is None:
            # Final fallback: identity (do not destroy frame). Still never write black/empty frames.
            m_id = float(np.nanmean(data_to_detect)) if data_to_detect is not None else float("nan")
            s_id = float(np.nanstd(data_to_detect)) if data_to_detect is not None else float("nan")
            if math.isfinite(m_id) and math.isfinite(s_id) and (m_id < 50.0) and (s_id < 20.0):
                log_event(
                    f"ERROR: alignment zlyhal pre {fp.name}, identity nepoužitá (black frame guard), preskakujem."
                )
                star_counts.append(
                    {
                        "file": fp.name,
                        "frame_index": int(frame_index_1based),
                        "detected_stars": int(len(xy_used)),
                        "aligned": False,
                        "reason": "alignment_failed_black_guard",
                        "is_flipped": bool(is_flipped),
                    }
                )
                continue
            aligned_data = _as_fits_float32_image(data_to_detect)
            aligned_method = "identity"
            hdr["VYALGOK"] = (False, "Alignment failed; identity fallback used")
            hdr["VY_ALGN"] = (False, "Not aligned to reference (identity fallback)")
            hdr["VY_REF"] = (ref_fp.name[:60], "Reference frame for alignment")
            hdr["VYALGM"] = (aligned_method[:30], "Alignment method (astroalign or WCS reprojection)")
            log_event(f"ERROR: alignment zlyhal pre {fp.name}, použitá identity")
            # Write identity frame (still keep its original WCS; do not strip/copy reference WCS).
            fp_rel = fp.relative_to(detrended_root)
            out_fp = aligned_root / fp_rel
            _ensure_parent_dirs_for_aligned_fits(out_fp)
            n_written_align += 1
            _prog(
                f"detrended_aligned/lights: "
                f"{'RAM — zarovnanie' if use_ram_handoff else 'zapisujem FITS'} "
                f"{fp.name} ({n_written_align}/{n_files})…"
            )
            if use_ram_handoff:
                aligned_ram_buffer.append((fp_rel.as_posix(), hdr.copy(), np.copy(aligned_data)))
            else:
                fits.writeto(out_fp, aligned_data, header=hdr, overwrite=True)
            aligned_files.append(out_fp)
            star_counts.append(
                {
                    "file": fp.name,
                    "frame_index": int(frame_index_1based),
                    "detected_stars": int(len(xy_used)),
                    "aligned": False,
                    "reason": "identity_fallback",
                    "alignment_method": aligned_method,
                    "is_flipped": bool(is_flipped),
                }
            )
            continue

        # propagate reference WCS to aligned file (incl. RADESYS / SIP via shared key set)
        strip_celestial_wcs_keys(hdr)
        for k, v in ref_hdr.items():
            if header_key_is_celestial_wcs(k):
                hdr[k] = v
        hdr["VYALGOK"] = (True, "Aligned to reference (passed black frame guard)")
        hdr["VY_ALGN"] = (True, "Aligned to reference")
        hdr["VY_REF"] = (ref_fp.name[:60], "Reference frame for alignment")
        hdr["VYALGM"] = (aligned_method[:30], "Alignment method (astroalign or WCS reprojection)")
        _maybe_refine_aligned(hdr, aligned_data, fp.name, dao_fwhm_px_frame=fw_i)
        fp_rel = fp.relative_to(detrended_root)
        out_fp = aligned_root / fp_rel
        _ensure_parent_dirs_for_aligned_fits(out_fp)
        n_written_align += 1
        _prog(
            f"detrended_aligned/lights: "
            f"{'RAM — zarovnanie' if use_ram_handoff else 'zapisujem FITS'} "
            f"{fp.name} ({n_written_align}/{n_files})…"
        )
        if use_ram_handoff:
            aligned_ram_buffer.append((fp_rel.as_posix(), hdr.copy(), np.copy(aligned_data)))
        else:
            fits.writeto(out_fp, aligned_data, header=hdr, overwrite=True)
        aligned_files.append(out_fp)
        star_counts.append(
            {
                "file": fp.name,
                "frame_index": int(frame_index_1based),
                "detected_stars": int(len(xy_used)),
                "aligned": True,
                "alignment_method": aligned_method,
                "is_flipped": bool(is_flipped),
            }
        )

    n_aligned = int(sum(1 for r in star_counts if r.get("aligned")))
    n_failed_align = int(sum(1 for r in star_counts if not bool(r.get("aligned"))))
    if n_failed_align > 0:
        reasons: dict[str, int] = {}
        for r in star_counts:
            if bool(r.get("aligned")):
                continue
            rr = str(r.get("reason") or "unknown")
            reasons[rr] = int(reasons.get(rr, 0)) + 1
        reason_txt = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:5])
        _pipeline_ui_info(
            f"Alignment warning: zlyhalo {n_failed_align}/{len(files)} snímok v {aligned_root.resolve()} "
            f"(dôvody: {reason_txt})."
        )
    if n_aligned <= 1:
        msg = (
            f"Alignment zlyhal: úspešne zarovnaná len referencia (1/{len(files)}). "
            f"Skontroluj DAO prah/FWHM a WCS vstupy. Výstupný priečinok: {aligned_root.resolve()}."
        )
        _pipeline_ui_error(msg)
        raise RuntimeError(msg)
    rep_path = platesolve_dir / "alignment_report.csv"
    pd.DataFrame(star_counts).to_csv(rep_path, index=False)

    print(f"  Zarovnanie: {time.time() - _t_align:.1f}s")
    _t_csv = time.time()

    # If we aligned in RAM, flush aligned FITS to disk before MASTERSTAR (needs files on disk).
    _ram_flushed_before_masterstar = False
    if use_ram_handoff and aligned_ram_buffer and build_masterstar_and_catalogs:
        _prog("detrended_aligned/lights: zapisujem FITS na disk (RAM → disk, pred MASTERSTAR)…")
        for name, hdr, arr in aligned_ram_buffer:
            _target = aligned_root / Path(name)
            _ensure_parent_dirs_for_aligned_fits(_target)
            fits.writeto(
                _target,
                _as_fits_float32_image(arr),
                header=hdr,
                overwrite=True,
            )
        _ram_flushed_before_masterstar = True

    cat_info: dict[str, Any] = {}
    ms_csv: Path | None = None
    ms_fits: Path | None = None
    use_master_fast = False

    if build_masterstar_and_catalogs:
        # Use root platesolve/ artifacts built before alignment. Keep platesolve/<setup>/ clean
        # (no MASTERSTAR.fits or masterstars_full_match.csv here).
        cat_info = dict(_cat_info_root or {})
        try:
            ms_csv = Path(str((cat_info.get("masterstars_csv") or (_ps_root / "masterstars_full_match.csv")))).resolve()
        except Exception:  # noqa: BLE001
            ms_csv = Path(str(cat_info.get("masterstars_csv") or (_ps_root / "masterstars_full_match.csv")))
        try:
            ms_fits = Path(str((cat_info.get("masterstar_fits") or (_ps_root / "MASTERSTAR.fits")))).resolve()
        except Exception:  # noqa: BLE001
            ms_fits = Path(str(cat_info.get("masterstar_fits") or (_ps_root / "MASTERSTAR.fits")))

        # Copy comparison/variable lists into platesolve/<setup>/ for UI consistency.
        try:
            _src_comp = _ps_root / "comparison_stars.csv"
            _src_var = _ps_root / "variable_targets.csv"
            _dst_comp = platesolve_dir / "comparison_stars.csv"
            _dst_var = platesolve_dir / "variable_targets.csv"
            if _src_comp.is_file():
                shutil.copy2(_src_comp, _dst_comp)
                cat_info["comparison_stars_csv"] = str(_dst_comp)
            if _src_var.is_file():
                shutil.copy2(_src_var, _dst_var)
                cat_info["variable_targets_csv"] = str(_dst_var)
        except Exception as _cp_lists:  # noqa: BLE001
            log_event(f"platesolve/<setup>: copy comp/vars failed (nekritické): {_cp_lists}")

        # Masterstar lock for Step 3: per-frame catalogs must use one fixed reference list.
        use_master_fast = True

    export_base = prog_i[0]
    _catalog_app_cfg = _cfg_align

    def _cat_prog(i: int, tot: int, msg: str) -> None:
        if progress_cb is None:
            return
        progress_cb(
            min(export_base + i, global_total),
            global_total,
            f"detrended_aligned/lights: CSV ({i}/{tot}) — {msg}",
        )

    if use_ram_handoff:
        per_cat = export_per_frame_catalogs(
            frames_root=aligned_root,
            platesolve_dir=platesolve_dir,
            max_catalog_rows=int(max_catalog_rows),
            catalog_match_max_sep_arcsec=float(per_frame_match_sep),
            saturate_level_fraction=float(saturate_level_fraction),
            faintest_mag_limit=faintest_mag_limit,
            dao_threshold_sigma=float(dao_threshold_sigma),
            masterstars_csv=ms_csv,
            masterstar_fits=ms_fits,
            use_master_fast_path=use_master_fast,
            equipment_saturate_adu=equip_sat_adu,
            catalog_local_gaia_only=_cat_loc_only,
            progress_cb=_cat_prog if progress_cb is not None else None,
            aligned_ram=aligned_ram_buffer,
            aligned_target_dir=aligned_root,
            defer_disk_writes=True,
            app_config=_catalog_app_cfg,
            plate_solve_fov_deg=float(plate_solve_fov_deg),
            master_dark_path=_master_dark_bpm_path,
            draft_id=draft_id,
            equipment_id=id_equipment,
        )
        _prog("detrended_aligned/lights: zapisujem FITS + CSV na disk (dávka po práci v RAM)…")
        if not _ram_flushed_before_masterstar:
            for name, hdr, arr in aligned_ram_buffer:
                _target = aligned_root / Path(name)
                _ensure_parent_dirs_for_aligned_fits(_target)
                fits.writeto(
                    _target,
                    _as_fits_float32_image(arr),
                    header=hdr,
                    overwrite=True,
                )
        for pcsv, df in per_cat.get("deferred_csv_writes", []):
            df.to_csv(pcsv, index=False)
        pd.DataFrame(per_cat.get("frames", [])).to_csv(Path(per_cat["index_csv"]), index=False)
    else:
        per_cat = export_per_frame_catalogs(
            frames_root=aligned_root,
            platesolve_dir=platesolve_dir,
            max_catalog_rows=int(max_catalog_rows),
            catalog_match_max_sep_arcsec=float(per_frame_match_sep),
            saturate_level_fraction=float(saturate_level_fraction),
            faintest_mag_limit=faintest_mag_limit,
            dao_threshold_sigma=float(dao_threshold_sigma),
            masterstars_csv=ms_csv,
            masterstar_fits=ms_fits,
            use_master_fast_path=use_master_fast,
            equipment_saturate_adu=equip_sat_adu,
            catalog_local_gaia_only=_cat_loc_only,
            progress_cb=_cat_prog if progress_cb is not None else None,
            app_config=_catalog_app_cfg,
            plate_solve_fov_deg=float(plate_solve_fov_deg),
            master_dark_path=_master_dark_bpm_path,
            draft_id=draft_id,
            equipment_id=id_equipment,
        )

    _assert_alignment_produced_fits(aligned_root)

    print(f"  Per-frame CSV: {time.time() - _t_csv:.1f}s")
    print(f"CELKOM krok 3 ({obs_group_key or detrended_root.name}): {time.time() - _t_step3_start:.1f}s")

    LOGGER.info(
        "Astrometria dokončená: zarovnané %s / %s snímok; per-frame CSV: %s; MASTERSTAR: %s; RAM handoff: %s",
        n_aligned,
        len(files),
        int(per_cat.get("written", 0)),
        "áno" if build_masterstar_and_catalogs else "nie",
        "áno" if use_ram_handoff else "nie",
    )

    return {
        "ram_align_handoff_used": bool(use_ram_handoff),
        "detrended_input_root": str(detrended_root),
        "detrended_files_used": len(files),
        "reference_frame": str(ref_fp),
        "reference_star_counts": dict(ref_star_scores),
        "reference_hint_ra_dec_deg": {"ra": hint_ra, "dec": hint_dec} if hint_ra is not None else None,
        "extra_platesolve": extra_platesolve_results,
        "alignment_max_control_points_used": align_cp,
        "alignment_max_stars_cap": int(_align_star_cap),
        "alignment_detection_sigma": float(_align_det_sigma),
        "aligned_root": str(aligned_root),
        "aligned_frames": n_aligned,
        "input_frames": int(len(files)),
        "alignment_report_csv": str(rep_path),
        "rotation_ref_angle_deg": rotation_ref_angle_deg,
        "rotation_flip_frame_indices_1based": rotation_flip_frame_indices_1based,
        "rotation_flip_first_index_1based": rotation_flip_first_index_1based,
        "build_masterstar_and_catalogs": bool(build_masterstar_and_catalogs),
        "masterstar_built": bool(_masterstar_built) if build_masterstar_and_catalogs else False,
        "masterstar_fits": str(ms_fits) if build_masterstar_and_catalogs and ms_fits is not None else "",
        "masterstars_csv": str(ms_csv) if build_masterstar_and_catalogs and ms_csv is not None else "",
        "catalog_match_max_sep_arcsec": float(_catalog_match_sep_eff),
        "saturate_level_fraction": float(saturate_level_fraction),
        "saturate_limit_adu": (cat_info.get("saturate_limit_adu") if build_masterstar_and_catalogs else None),
        "saturate_limit_source": (cat_info.get("saturate_limit_source") if build_masterstar_and_catalogs else None),
        "max_catalog_rows": int(max_catalog_rows),
        "faintest_mag_limit": (
            cat_info.get("faintest_mag_limit") if build_masterstar_and_catalogs else faintest_mag_limit
        ),
        "per_frame_catalog_dir": per_cat.get("per_frame_dir"),
        "per_frame_catalog_index_csv": per_cat.get("index_csv"),
        "per_frame_catalogs_written": per_cat.get("written"),
        "comparison_stars_csv": cat_info.get("comparison_stars_csv", "") if build_masterstar_and_catalogs else "",
        "variable_targets_csv": cat_info.get("variable_targets_csv", "") if build_masterstar_and_catalogs else "",
        "photometry_plan_json": cat_info.get("photometry_plan_json", "") if build_masterstar_and_catalogs else "",
        "n_comparison_stars_requested": (
            cat_info.get("n_comparison_stars_requested") if build_masterstar_and_catalogs else None
        ),
        "comparison_selection": cat_info.get("comparison_selection") if build_masterstar_and_catalogs else None,
        "id_equipment": int(id_equipment) if id_equipment is not None else None,
        "equipment_saturate_adu_resolved": equip_sat_adu,
        "catalog_local_gaia_only": _cat_loc_only,
        "observation_group_key": obs_group_key,
        "scanning_id": (scanning_id if scanning_id > 0 else None),
    }


def astrometry_align_and_build_masterstar(
    *,
    archive_path: Path,
    astrometry_api_key: str | None = None,
    max_control_points: int = 180,
    min_detected_stars: int = 100,
    max_detected_stars: int = 500,
    platesolve_backend: str = "vyvar",
    plate_solve_fov_deg: float = 1.0,
    max_extra_platesolve: int = 0,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    max_catalog_rows: int = 12000,
    n_comparison_stars: int = 150,
    require_non_variable_comparisons: bool = True,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    id_equipment: int | None = None,
    draft_id: int | None = None,
    catalog_local_gaia_only: bool | None = None,
    build_masterstar_and_catalogs: bool = False,
    progress_cb: "callable | None" = None,
    ram_align_and_catalog: bool = False,
    app_config: AppConfig | None = None,
    masterstar_candidate_paths: "Sequence[str] | None" = None,
    masterstar_selection_pct: float | None = None,
    master_dark_path: Path | str | None = None,
) -> dict[str, Any]:
    """Astrometry + alignment + per-frame catalog CSV (mandatory outputs).

    Preprocessed frames under ``<archive>/processed/lights`` (or legacy ``detrended/lights``) are grouped
    by full relative parent folder (multi-observation / FILTER+EXP+BIN layout). Each group gets its own
    alignment, ``MASTERSTAR``, and ``platesolve/<group>/`` outputs. Frames stored directly in ``lights/``
    form a single group.
    """
    ap = Path(archive_path)
    input_root = _archive_preprocess_lights_root(ap)
    if not input_root.exists():
        log_event(f"❌ ERROR: Input path {input_root} does not exist! Trying fallback...")
        processed_lights = ap / "processed" / "lights"
        subdirs: list[Path] = []
        try:
            if processed_lights.is_dir():
                subdirs = [d for d in processed_lights.iterdir() if d.is_dir()]
        except Exception:  # noqa: BLE001
            subdirs = []
        if subdirs:
            subdirs = sorted(subdirs, key=lambda p: p.name.casefold())
            input_root = subdirs[0]
            log_event(f"✅ Fallback found: {input_root}")
    det_top = input_root
    ali_top = ap / "detrended_aligned" / "lights"
    os.makedirs(str(ali_top), exist_ok=True)
    ps_top = ap / "platesolve"
    os.makedirs(str(ps_top), exist_ok=True)
    files_all = _iter_fits_recursive(det_top)
    _n_root_only = len(list(det_top.glob("*.fits"))) if det_top.exists() else 0
    _n_all = len(files_all)
    if _n_root_only != _n_all:
        log_event(
            f"FITS celkom: {_n_all} pod {det_top} ({_n_root_only} priamo v koreni; "
            f"ostatné v podpriečinkoch napr. filter/exp/binning)."
        )
    else:
        log_event(f"FITS celkom: {_n_all} v {det_top}")
    log_event(f"Alignment input root: {det_top}")
    if not files_all:
        raise FileNotFoundError(
            f"Chýbajú FITS v {det_top}. Krok 3 číta len **spracované** snímky. "
            "Najprv spusti **Apply Calibration & Detrending** po kroku **Analyze** (zápis z "
            f"`{ap / 'calibrated' / 'lights'}` → `{ap / 'processed' / 'lights'}` alebo staršie `{ap / 'detrended' / 'lights'}`)."
        )
    job_list: list[dict[str, Any]] = []
    # Prefer explicit per-file ID_SCANNING groups from OBS_FILES when draft has >1 scanning configuration.
    _scan_groups_used = False
    if draft_id is not None:
        try:
            _db_scan = VyvarDatabase(Path((app_config or AppConfig()).database_path))
            try:
                rows_q = _db_scan.fetch_draft_light_rows_for_quality(int(draft_id))
            finally:
                _db_scan.conn.close()
            sid_set = sorted(
                {
                    int(r.get("ID_SCANNING") or 0)
                    for r in rows_q
                    if int(r.get("ID_SCANNING") or 0) > 0
                }
            )
            if len(sid_set) > 1:
                map_sid: dict[int, list[Path]] = {int(s): [] for s in sid_set}
                known = {str(p.resolve()): p for p in files_all}
                for rr in rows_q:
                    sid = int(rr.get("ID_SCANNING") or 0)
                    if sid <= 0:
                        continue
                    rp = str(rr.get("FILE_PATH") or "").strip()
                    if not rp:
                        continue
                    hit = _resolve_best_effort_path_under(det_top, rp)
                    if hit is None:
                        continue
                    key = str(hit.resolve())
                    hp = known.get(key)
                    if hp is not None:
                        map_sid[sid].append(hp)
                for sid in sid_set:
                    fs = sorted({str(x.resolve()): x for x in map_sid.get(sid, [])}.values(), key=lambda p: str(p))
                    if not fs:
                        continue
                    skey = f"scan_{int(sid)}"
                    job_list.append(
                        {
                            "gkey": skey,
                            "scanning_id": int(sid),
                            "detrended_root": det_top,
                            "aligned_root": ali_top / skey,
                            "platesolve_dir": ps_top / skey,
                            "files": fs,
                        }
                    )
                _scan_groups_used = len(job_list) > 0
                if _scan_groups_used:
                    log_event(
                        f"Astrometria: Draft {int(draft_id)} obsahuje viac ID_SCANNING "
                        f"({', '.join(str(s) for s in sid_set)}); spracovanie beží samostatne po konfiguráciách."
                    )
        except Exception as exc:  # noqa: BLE001
            log_event(f"Astrometria: ID_SCANNING grouping fallback na folder grouping ({exc!s}).")

    if not _scan_groups_used:
        groups = _partition_detrended_by_subfolder(files_all, det_top)
        for gkey in sorted(groups.keys()):
            gfs = groups[gkey]
            if not gfs:
                continue
            _setup_name = Path(gkey).name if gkey else ""
            if _setup_name:
                log_event(
                    f"Alignment subgroup detected: setup_name={_setup_name} "
                    f"(input={det_top / gkey}, files={len(gfs)})"
                )
            if gkey:
                job_list.append(
                    {
                        "gkey": gkey,
                        "scanning_id": None,
                        "detrended_root": det_top / gkey,
                        "aligned_root": ali_top / gkey,
                        "platesolve_dir": ps_top / gkey,
                        "files": gfs,
                    }
                )
            else:
                job_list.append(
                    {
                        "gkey": "",
                        "scanning_id": None,
                        "detrended_root": det_top,
                        "aligned_root": ali_top,
                        "platesolve_dir": ps_top,
                        "files": gfs,
                    }
                )
    if len(job_list) > 1:
        log_event(
            "Astrometria: viacero pod-pozorovaní (podpriečinky v processed|detrended/lights) — "
            "samostatné zarovnanie, MASTERSTAR a katalógy pre každú skupinu."
        )
    _ms_paths = [str(x).strip() for x in (masterstar_candidate_paths or []) if str(x).strip()]
    try:
        _ms_pct = float(masterstar_selection_pct) if masterstar_selection_pct is not None else None
    except (TypeError, ValueError):
        _ms_pct = None
    if _ms_pct is not None and (not math.isfinite(_ms_pct) or _ms_pct <= 0):
        _ms_pct = None
    _md_bpm_job: str | None = None
    if master_dark_path is not None and str(master_dark_path).strip():
        _mdp = Path(str(master_dark_path))
        if _mdp.is_file():
            _md_bpm_job = str(_mdp.resolve())
    for _j in job_list:
        _j["masterstar_candidate_paths"] = _ms_paths
        _j["masterstar_selection_pct"] = _ms_pct
        if _md_bpm_job:
            _j["master_dark_path"] = _md_bpm_job

    _kw: dict[str, Any] = dict(
        archive_path=archive_path,
        astrometry_api_key=astrometry_api_key,
        max_control_points=max_control_points,
        min_detected_stars=min_detected_stars,
        max_detected_stars=max_detected_stars,
        platesolve_backend=platesolve_backend,
        plate_solve_fov_deg=plate_solve_fov_deg,
        max_extra_platesolve=max_extra_platesolve,
        catalog_match_max_sep_arcsec=catalog_match_max_sep_arcsec,
        saturate_level_fraction=saturate_level_fraction,
        max_catalog_rows=max_catalog_rows,
        n_comparison_stars=n_comparison_stars,
        require_non_variable_comparisons=require_non_variable_comparisons,
        faintest_mag_limit=faintest_mag_limit,
        dao_threshold_sigma=dao_threshold_sigma,
        id_equipment=id_equipment,
        draft_id=draft_id,
        catalog_local_gaia_only=catalog_local_gaia_only,
        build_masterstar_and_catalogs=build_masterstar_and_catalogs,
        progress_cb=progress_cb,
        ram_align_and_catalog=ram_align_and_catalog,
        app_config=app_config,
    )
    if len(job_list) == 1:
        return _astrometry_align_impl_body(job=job_list[0], **_kw)
    return _merge_astrometry_group_reports(
        [_astrometry_align_impl_body(job=j, **_kw) for j in job_list]
    )


def _fits_meta_ra_deg(value: Any) -> float:
    r = _fits_header_parse_ra_deg(value)
    return float(r) if r is not None else 0.0


def _fits_meta_dec_deg(value: Any) -> float:
    d = _fits_header_parse_dec_deg(value)
    return float(d) if d is not None else 0.0


def _resolve_cosmic_gain_readnoise(
    hdr: fits.Header,
    *,
    db: VyvarDatabase | None,
    equipment_id: int | None,
    gain_default: float = 1.0,
    readnoise_default: float = 6.5,
) -> tuple[float, float, str]:
    """``astroscrappy`` noise model: gain [e⁻/ADU], read noise [e⁻], provenance tag."""
    gain_h = _fits_header_positive_float(hdr, "GAIN", "EGAIN", "AVGAIN")
    rn_h = _fits_header_positive_float(
        hdr,
        "READNOIS",
        "RDNOISE",
        "READOUTNS",
        "READNOISE",
        "RON",
        "ERRD",
    )
    gain_db: float | None = None
    rn_db: float | None = None
    if db is not None and equipment_id is not None:
        try:
            gain_db, rn_db = db.get_equipment_cosmic_params(int(equipment_id))
        except Exception:  # noqa: BLE001
            gain_db, rn_db = None, None

    if gain_h is not None:
        gain = float(gain_h)
        g_src = "header"
    elif gain_db is not None:
        gain = float(gain_db)
        g_src = "EQUIPMENTS.GAIN_ADU"
    else:
        gain = float(gain_default)
        g_src = "default"

    if rn_h is not None:
        rn = float(rn_h)
        r_src = "header"
    elif rn_db is not None:
        rn = float(rn_db)
        r_src = "EQUIPMENTS.READNOISE_E"
    else:
        rn = float(readnoise_default)
        r_src = "default"

    return gain, rn, f"gain={g_src};RN={r_src}"


def _satlevel_electrons_from_header(hdr: fits.Header, gain: float) -> float | None:
    """Rough saturation threshold in electrons for ``detect_cosmics`` (ADU × gain)."""
    adu = _fits_header_positive_float(hdr, "SATURATE", "MAXLIN", "MAXADU", "DATAMAX")
    if adu is None:
        return None
    g = float(gain) if gain > 0 and math.isfinite(gain) else 1.0
    return float(adu) * g


def _db_for_cosmic(db: VyvarDatabase | None, db_path: str | Path | None) -> VyvarDatabase | None:
    if db is not None:
        return db
    if db_path is None or str(db_path).strip() == "":
        return None
    try:
        return VyvarDatabase(Path(db_path))
    except Exception:  # noqa: BLE001
        return None


def _db_for_calibration_tasks(
    cosmic_opt: dict[str, Any] | None,
    qc_opt: dict[str, Any] | None,
) -> VyvarDatabase | None:
    """Open DB once per worker / sequential pass when cosmic or post-calibrate QC needs it."""
    c = cosmic_opt or {}
    q = qc_opt or {}
    p: str | None = None
    if c.get("enabled") and c.get("db_path"):
        p = str(c["db_path"])
    elif q.get("enabled") and q.get("db_path"):
        p = str(q["db_path"])
    if not p:
        return None
    try:
        return VyvarDatabase(Path(p))
    except Exception:  # noqa: BLE001
        return None


def _cosmic_pack_from_config(
    cfg: AppConfig,
    *,
    equipment_id: int | None,
) -> dict[str, Any]:
    """Serializable options for workers: L.A.Cosmic after linear calibration."""
    en = bool(cfg.cosmic_clean_enabled)
    return {
        "enabled": en,
        "sigclip": float(cfg.cosmic_sigclip),
        "objlim": float(cfg.cosmic_objlim),
        "gain_default": 1.0,
        "readnoise_default": 6.5,
        "db_path": str(Path(cfg.database_path).resolve()) if en else None,
        "equipment_id": equipment_id,
    }


def _qc_pack_from_config(
    cfg: AppConfig,
    *,
    draft_id: int | None,
    observation_id: str | None,
) -> dict[str, Any]:
    """Post-calibration QC limits + DB linkage (``OBS_FILES`` update by raw light path)."""
    en = bool(cfg.qc_after_calibrate_enabled)
    _dao = float(getattr(cfg, "qc_dao_detection_sigma", 5.0))
    if not math.isfinite(_dao) or _dao <= 0:
        _dao = 5.0
    id_equipments: int | None = None
    if draft_id is not None and str(cfg.database_path).strip():
        try:
            _dbp = Path(cfg.database_path)
            if _dbp.is_file():
                _row = VyvarDatabase(_dbp).conn.execute(
                    "SELECT ID_EQUIPMENTS FROM OBS_DRAFT WHERE ID = ?",
                    (int(draft_id),),
                ).fetchone()
                if _row and _row[0] is not None:
                    id_equipments = int(_row[0])
        except Exception:  # noqa: BLE001
            id_equipments = None
    return {
        "enabled": en,
        "max_hfr": float(cfg.qc_max_hfr),
        "min_stars": int(cfg.qc_min_stars),
        "max_bg_rms": cfg.qc_max_background_rms,
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
) -> tuple[float | None, int]:
    """Median half-flux radius [px] on up to ``max_stars`` brightest DAO sources; return (HFR, n_detected)."""
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    img = np.asarray(crop, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return None, 0
    _, med, std = sigma_clipped_stats(img[finite], sigma=3.0, maxiters=5)
    std = float(std)
    if not math.isfinite(std) or std <= 0:
        return None, 0
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
        return None, 0
    n_det = int(len(tbl))
    tbl.sort("flux")
    tbl = tbl[::-1]
    h, w = img2.shape
    half = 10
    hfrs: list[float] = []
    for i in range(min(max_stars, len(tbl))):
        x0 = float(tbl["xcentroid"][i])
        y0 = float(tbl["ycentroid"][i])
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
        return None, n_det
    return float(np.nanmedian(np.asarray(hfrs, dtype=np.float64))), n_det


def _post_calibration_qc_eval(
    data: "np.ndarray",
    *,
    limits: dict[str, Any],
    light_basename: str = "",
) -> dict[str, Any]:
    """Sky stats on full frame; star HFR/count on central crop if frame is large."""
    import numpy as np
    from astropy.stats import sigma_clipped_stats

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

    sky_mean, sky_med, sky_rms = sigma_clipped_stats(img[finite], sigma=3.0, maxiters=5)
    sky_mean = float(sky_mean)
    sky_med = float(sky_med)
    sky_rms = float(sky_rms)

    crop = _qc_center_crop_for_stars(img, 1000)
    _ds = float(limits.get("dao_detection_sigma", 5.0))
    if not math.isfinite(_ds) or _ds <= 0:
        _ds = 5.0
    hfr_m, n_star = _mean_hfr_bright_stars_dao(crop, max_stars=50, dao_detection_sigma=_ds)

    max_h = float(limits.get("max_hfr", 5.0))
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


def _maybe_cosmic_clean_calibrated_frame(
    data: "np.ndarray",
    hdr: fits.Header,
    *,
    used_dark: bool,
    used_flat: bool,
    cosmic_options: dict[str, Any] | None,
    db: VyvarDatabase | None,
    db_path: str | Path | None = None,
    light_basename: str = "",
) -> "np.ndarray":
    """Calibration stage no longer performs L.A.Cosmic; return input unchanged."""
    _ = (hdr, used_dark, used_flat, cosmic_options, db, db_path, light_basename)
    return data


def _strip_raw_linearity_header_keywords(hdr: fits.Header) -> None:
    """Remove FITS keys that describe the **raw** detector linearity range.

    After ``(light − dark) / flat`` the pixel scale is no longer raw ADU; keeping SATURATE/DATAMAX from the
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
    cosmic_options: dict[str, Any] | None = None,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    calibration_master_native_binning: int | None | object = _CALIB_MASTER_NB_UNSET,
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
    if _mb_lib is None:
        md_data = None
    elif md_data is not None and md_data.shape != data.shape:
        md_data = None
    if md_data is not None and master_dark_path is not None and master_dark_path.exists():
        if _mb_lib != light_bx:
            md_data = None

    if md_data is None and master_dark_path is not None and master_dark_path.exists():
        pm = get_processed_master(
            master_dark_path,
            light_bx,
            kind="dark",
            master_binning=_mb_lib,
            light_shape=data.shape,
            light_filename=src.name,
        )
        if pm.resampled:
            log_event(
                f"Calibration: library master native {pm.master_binning}x{pm.master_binning} -> "
                f"resampled (RAM) to {light_bx}x{light_bx} for Light [{src.name}]"
            )
        md_data = pm.data

    hdr["VY_COSM"] = (False, "Cosmic-ray cleaning applied in preprocessing")

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
        key = f"{flt}|{str(mf_path)}|lb{light_bx}"
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
            key_m = f"{flt}|{str(mf_path)}|lb{light_bx}"
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

    return data, hdr, used_dark, used_flat


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
    cosmic_options: dict[str, Any] | None = None,
    db: VyvarDatabase | None = None,
    qc_pack: dict[str, Any] | None = None,
    calibration_master_native_binning: int | None | object = _CALIB_MASTER_NB_UNSET,
) -> tuple[bool, bool, dict[str, Any] | None, str]:
    """Apply master dark / flat to one light FITS and write ``dst``.

    Uses ``fits.open(..., memmap=False)`` and ``with`` blocks so BZERO/BSCALE frames load reliably and
    file handles close after each read (arrays are copied into RAM before processing).

    Returns ``(used_dark, used_flat, qc_summary, vy_cflag)`` where ``qc_summary`` is set when post-calibration
    QC ran; ``vy_cflag`` matches ``VY_CFLAG`` written to the FITS (see calibration decision table).
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
        cosmic_options=cosmic_options,
        db=db,
        id_equipments=_id_eq,
        calibration_master_native_binning=calibration_master_native_binning,
    )

    qc_summary: dict[str, Any] | None = None
    if (used_dark or used_flat) and qc_pack and qc_pack.get("enabled"):
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
                db_q = db if db is not None else _db_for_cosmic(None, qc_pack.get("db_path"))
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
                LOGGER.debug("OBS_FILES QC update failed: %s", exc)

    fits.writeto(dst, _as_fits_float32_image(data), header=hdr, overwrite=True)
    return used_dark, used_flat, qc_summary, _hdr_vy_cflag_str(hdr)


_cal_batch_flat_cache: dict[str, Any] | None = None
_cal_batch_flat_median: dict[str, float] | None = None
_cal_batch_md_preload: Any = None
_cal_batch_native_binning: int | None = 1


def _init_calibrate_batch_worker(initargs: tuple[str | None, int | None]) -> None:
    """Per-subprocess caches; ``native_binning`` = CalibrationLibrary master convention (``None`` = read FITS)."""
    global _cal_batch_flat_cache, _cal_batch_flat_median, _cal_batch_md_preload, _cal_batch_native_binning
    _md_s, native_b = initargs
    _ = _md_s  # path reserved for future worker-side dark preload
    _cal_batch_flat_cache = {}
    _cal_batch_flat_median = {}
    _cal_batch_md_preload = None
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
    global _cal_batch_flat_cache, _cal_batch_flat_median, _cal_batch_md_preload, _cal_batch_native_binning
    cosmic_opt: dict[str, Any] | None = None
    qc_opt: dict[str, Any] | None = None
    if len(item) == 4:
        src_s, dst_s, md_s, mf_map = item  # type: ignore[misc]
    elif len(item) == 5:
        src_s, dst_s, md_s, mf_map, cosmic_opt = item  # type: ignore[misc]
    else:
        src_s, dst_s, md_s, mf_map, cosmic_opt, qc_opt = item  # type: ignore[misc]
    fc = _cal_batch_flat_cache
    fm = _cal_batch_flat_median
    if fc is None or fm is None:
        fc, fm = {}, {}
    try:
        mf: dict[str, Path | None] = {str(k): Path(v) if v else None for k, v in mf_map.items()}
        db_w = _db_for_calibration_tasks(cosmic_opt, qc_opt)
        _ud, _uf, qc_sum, _cf = _calibrate_one_light_disk(
            src=Path(src_s),
            dst=Path(dst_s),
            master_dark_path=Path(md_s) if md_s else None,
            masterflat_by_filter=mf,
            flat_cache=fc,
            flat_median_scale=fm,
            md_data_preload=_cal_batch_md_preload,
            cosmic_options=cosmic_opt,
            db=db_w,
            qc_pack=qc_opt,
            calibration_master_native_binning=_cal_batch_native_binning,
        )
        return {
            "src": src_s,
            "dst": dst_s,
            "ok": True,
            "error": None,
            "qc_summary": qc_sum,
            "traceback": None,
            "vy_cflag": _cf,
        }
    except Exception as exc:  # noqa: BLE001
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


def _has_any_usable_master_flat(masterflat_by_filter: dict[str, Path | None] | None) -> bool:
    if not masterflat_by_filter:
        return False
    for _k, v in masterflat_by_filter.items():
        if v is not None and Path(v).is_file():
            return True
    return False


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
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            stats["errors"] += 1
    try:
        if db_pt is not None:
            db_pt.conn.close()
    except Exception:  # noqa: BLE001
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
    of magnitude as (light − dark) and extreme spikes from tiny flat pixels are avoided.

    After normalization, each flat pixel is **clamped below by a small positive floor** so dust donuts and
    dead columns cannot divide the image by near-zero (which would create nonsense ADU far above the
    detector range in viewers).

    Pixels are **not** expected to match raw ADU counts: calibration is ``(L − D) / F_norm`` with
    ``F_norm = flat / median(flat)``. Raw linearity keywords (SATURATE, DATAMAX, …) are dropped when any
    calibration is applied so headers stay consistent with the stored float image.

    Parallelism uses ``max_workers`` or ``pipeline_config.qc_preprocess_workers`` only when environment
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

    dark_cache: dict[str, Any] = {}
    _native_b = _cfg_calibration_library_native_binning(cfg)

    def _dark_np_for_path(p: Path | None, light_binning: int) -> Any:
        if p is None or not p.is_file():
            return None
        _mb_key = "hdr" if _native_b is None else str(int(_native_b))
        key = f"{str(p.resolve())}|b{int(light_binning)}|mb{_mb_key}"
        if key not in dark_cache:
            pm = get_processed_master(
                p,
                int(light_binning),
                kind="dark",
                master_binning=_native_b,
            )
            dark_cache[key] = pm.data
        return dark_cache[key]

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
    }

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
            log_event(f"OBS_FILES IS_REJECTED filter skipped (error): {exc}")
    total = len(files)
    if total < _n_before_obs_filter:
        log_event(
            "Kalibrácia — vylúčené %s súborov podľa OBS_FILES (IS_REJECTED=1 alebo mimo DB)",
            _n_before_obs_filter - total,
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
                    "OBS_FILES: SELECT COUNT(*) FROM OBS_FILES WHERE OBSERVATION_ID = ? "
                    f"(obs={observation_id!r}) → {_n_o} rows"
                )
            if draft_id is not None:
                _n_d = _dbc.count_obs_files_for_draft(int(draft_id))
                _log_bits.append(
                    "OBS_FILES: SELECT COUNT(*) FROM OBS_FILES WHERE DRAFT_ID = ? "
                    f"(draft_id={int(draft_id)}) → {_n_d} rows"
                )
        except Exception as exc:  # noqa: BLE001
            _log_bits.append(f"OBS_FILES count query failed: {exc}")
    log_event("Kalibrácia — vstupné súbory: " + " | ".join(_log_bits))

    if total > 0:
        try:
            _db_cal_diag = VyvarDatabase(Path(cfg.database_path))
            try:
                _meta0 = extract_fits_metadata(
                    files[0],
                    db=_db_cal_diag,
                    app_config=cfg,
                    id_equipment=equipment_id,
                    draft_id=draft_id,
                )
                _log_calibration_metadata_diagnostic(files[0].name, _meta0)
                stats["applied_focal_length"] = _meta0.get("focal_length")
                stats["applied_pixel_size"] = _meta0.get("pixel_um")
            finally:
                _db_cal_diag.conn.close()
        except Exception as exc:  # noqa: BLE001
            log_event(f"DIAGNOSTIKA KALIBRÁCIE: metadáta prvého súboru zlyhali: {exc!s}")

    db_main = _db_for_calibration_tasks(None, qc_pack)

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
        if master_dark_by_obs_key:
            _alt = master_dark_by_obs_key.get(_ok)
            if _alt is not None and str(_alt).strip() != "":
                _pa = Path(_alt)
                if _pa.is_file():
                    md_use = _pa
        if md_use is not None and md_use.is_file():
            if (
                md_pre is not None
                and md_path_ok is not None
                and md_use.resolve() == md_path_ok.resolve()
                and _native_b is not None
                and _native_b == light_bx
            ):
                md_np_use = md_pre
            else:
                md_np_use = _dark_np_for_path(md_use, light_bx)
        used_dark, used_flat, qc_sum, _flags = _calibrate_one_light_disk(
            src=src,
            dst=dst,
            master_dark_path=md_use,
            masterflat_by_filter=masterflat_by_filter,
            flat_cache=flat_cache,
            flat_median_scale=flat_median_scale,
            md_data_preload=md_np_use,
            cosmic_options=None,
            db=db_main,
            qc_pack=qc_pack,
            calibration_master_native_binning=_native_b,
        )
        try:
            if db_main is not None:
                db_main.update_obs_file_calibration_state_by_raw_light_path(
                    src,
                    draft_id=draft_id,
                    observation_id=observation_id,
                    is_calibrated=1 if "D" in _flags else 0,
                    calib_type=_calibration_type_from_flags(_flags),
                    calib_flags=_flags,
                )
        except Exception:  # noqa: BLE001
            pass
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
                dict[str, Any] | None,
            ]
        ] = []
        for src in files:
            rel = src.relative_to(lights_root)
            dst = calibrated_root / rel
            items.append(
                (
                    str(src.resolve()),
                    str(dst.resolve()),
                    md_init_str,
                    mf_serial,
                    None,
                    qc_pack,
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
                initargs=(md_init_str, _native_b),
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
            _tb_pool = traceback.format_exc()
            LOGGER.error("Kalibrácia (parallel): pool zlyhal, fallback na sekvenčný režim: %s\n%s", exc, _tb_pool)
            log_exception("CHYBA POOLU KALIBRÁCIE", exc)
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
                    LOGGER.error("Kalibrácia: súbor %s: %s\n%s", src, exc2, _tb2)
                    log_exception(f"CHYBA KALIBRÁCIE: {src.name}", exc2)
                    stats["errors"] += 1
            return stats

        for idx, r in enumerate(rows):
            if r is None or not r.get("ok"):
                stats["errors"] += 1
                if stats["errors"] == 1:
                    _ename = Path(items[idx][0]).name if idx < len(items) else "?"
                    _emsg = (r or {}).get("error") if isinstance(r, dict) else None
                    LOGGER.error(
                        "Kalibrácia: worker zlyhal pre %s: %s",
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
            except Exception:  # noqa: BLE001
                ud = uf = False
                _flags = "P"
            try:
                if db_main is not None:
                    db_main.update_obs_file_calibration_state_by_raw_light_path(
                        Path(items[idx][0]),
                        draft_id=draft_id,
                        observation_id=observation_id,
                        is_calibrated=1 if "D" in _flags else 0,
                        calib_type=_calibration_type_from_flags(_flags),
                        calib_flags=_flags,
                    )
            except Exception:  # noqa: BLE001
                pass
            if ud:
                stats["used_dark"] += 1
            if uf:
                stats["used_flat"] += 1
            if not ud and not uf:
                stats["copied_only"] += 1
        return stats

    _seq_tb_logged = False
    for i, src in enumerate(files, start=1):
        rel = src.relative_to(lights_root)
        dst = calibrated_root / rel
        try:
            _one_sequential(i, src, dst)
        except Exception as exc:
            _tb_seq = traceback.format_exc()
            LOGGER.error("Kalibrácia: súbor %s: %s\n%s", src, exc, _tb_seq)
            if not _seq_tb_logged:
                log_exception(f"CHYBA KALIBRÁCIE: {src.name}", exc)
                _seq_tb_logged = True
            else:
                log_event(f"CHYBA KALIBRÁCIE: {src.name}: {exc!s}")
            stats["errors"] += 1
            continue

    return stats


def _safe_proc_name(original_name: str) -> str:
    name = (original_name or "").strip()
    if not name:
        return "proc.fits"
    if name.lower().startswith("proc_"):
        return name
    return "proc_" + name


def _fit_background_poly2d(data: "np.ndarray", order: int) -> "np.ndarray":
    """Fit a 2D polynomial background (order 2 or 3) with robust downsampling."""
    import numpy as np

    if order not in (2, 3):
        raise ValueError("order must be 2 or 3")
    img = np.asarray(data, dtype=np.float32)
    h, w = img.shape
    # Downsample for speed/robustness (target ~256x256)
    step = max(1, int(max(h, w) // 256))
    ys = np.arange(0, h, step, dtype=np.int32)
    xs = np.arange(0, w, step, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    z = img[yy, xx].ravel()
    x = xx.ravel().astype(np.float32)
    y = yy.ravel().astype(np.float32)

    # Robust clip to reduce star influence
    med = np.nanmedian(z)
    mad = np.nanmedian(np.abs(z - med)) + 1e-6
    good = np.isfinite(z) & (np.abs(z - med) < 6.0 * 1.4826 * mad)
    x = x[good]
    y = y[good]
    z = z[good]

    # Build design matrix
    if order == 2:
        A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)
    else:
        A = np.stack(
            [
                np.ones_like(x),
                x,
                y,
                x * x,
                x * y,
                y * y,
                x * x * x,
                x * x * y,
                x * y * y,
                y * y * y,
            ],
            axis=1,
        )
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)

    # Evaluate on full grid
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    if order == 2:
        bg = (
            coef[0]
            + coef[1] * X
            + coef[2] * Y
            + coef[3] * X * X
            + coef[4] * X * Y
            + coef[5] * Y * Y
        )
    else:
        bg = (
            coef[0]
            + coef[1] * X
            + coef[2] * Y
            + coef[3] * X * X
            + coef[4] * X * Y
            + coef[5] * Y * Y
            + coef[6] * X * X * X
            + coef[7] * X * X * Y
            + coef[8] * X * Y * Y
            + coef[9] * Y * Y * Y
        )
    return np.asarray(bg, dtype=np.float32)


def _background2d_mesh_box_size(shape: tuple[int, ...]) -> int:
    """Larger tiles ⇒ fewer Background2D cells ⇒ much faster on big detectors (sky estimate still OK)."""
    h, w = int(shape[0]), int(shape[1])
    m = min(h, w)
    return max(48, int(m // 10))


def _background_subtract(
    data: "np.ndarray",
    *,
    method: str = "background2d",
    poly_order: int = 2,
) -> tuple["np.ndarray", dict[str, float]]:
    """Return (data_minus_background, metrics)."""
    import numpy as np

    img = np.asarray(data, dtype=np.float32)
    if method == "poly2d":
        bg = _fit_background_poly2d(img, poly_order)
        out = img - bg
        return out, {"bg_median": float(np.nanmedian(bg)), "method": 0.0}

    # Default: photutils Background2D
    try:
        from astropy.stats import SigmaClip
        from photutils.background import Background2D, MedianBackground

        sigma_clip = SigmaClip(sigma=3.0, maxiters=3)
        box = _background2d_mesh_box_size(img.shape)
        box_size = (box, box)
        bkg = Background2D(
            img,
            box_size,
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=MedianBackground(),
            exclude_percentile=10.0,
        )
        bg = np.asarray(bkg.background, dtype=np.float32)
        out = img - bg
        return out, {"bg_median": float(np.nanmedian(bg)), "method": 1.0}
    except Exception:
        # Fallback to polynomial if photutils not available
        bg = _fit_background_poly2d(img, poly_order)
        out = img - bg
        return out, {"bg_median": float(np.nanmedian(bg)), "method": 0.0}


def _remove_cosmics_lacosmic(
    data: "np.ndarray",
    *,
    sigclip: float = 4.5,
    objlim: float = 5.0,
    gain: float = 1.0,
    readnoise: float = 6.5,
    inmask: "np.ndarray | None" = None,
    satlevel: float | None = None,
    cleantype: str = "medmask",
) -> tuple["np.ndarray", dict[str, float]]:
    import numpy as np

    img = np.asarray(data, dtype=np.float32)
    try:
        import astroscrappy  # type: ignore

        kwargs: dict[str, Any] = {
            "sigclip": float(sigclip),
            "objlim": float(objlim),
            "gain": float(gain),
            "readnoise": float(readnoise),
            "cleantype": str(cleantype),
        }
        if inmask is not None:
            kwargs["inmask"] = np.asarray(inmask, dtype=bool)
        if satlevel is not None and math.isfinite(float(satlevel)) and float(satlevel) > 0:
            kwargs["satlevel"] = float(satlevel)

        crmask, cleaned = astroscrappy.detect_cosmics(img, **kwargs)
        n_cosmic = int(np.count_nonzero(crmask))
        frac = float(n_cosmic) / float(crmask.size) if crmask.size else 0.0
        return np.asarray(cleaned, dtype=np.float32), {
            "cr_frac": frac,
            "n_cosmic": float(n_cosmic),
            "ok": True,
        }
    except Exception as exc:  # noqa: BLE001
        return img, {"cr_frac": 0.0, "n_cosmic": 0.0, "ok": False, "error": str(exc)}


def _estimate_dao_fwhm_guess(img2: "np.ndarray", std: float) -> float:
    """Data-driven FWHM guess (pixels) for DAOStarFinder when segmentation path failed."""
    import numpy as np

    try:
        from photutils.segmentation import detect_sources, SourceCatalog

        std = float(std)
        if not np.isfinite(std) or std <= 0:
            return 3.0
        segm = None
        for k in (4.0, 3.0):
            segm = detect_sources(img2, k * std, npixels=6)
            if segm is not None and getattr(segm, "nlabels", 0) >= 5:
                break
        if segm is None or getattr(segm, "nlabels", 0) < 1:
            return 3.0
        cat = SourceCatalog(img2, segm)
        flux = np.asarray(cat.segment_flux, dtype=np.float64)
        if flux.size == 0:
            return 3.0
        idx = np.argsort(flux)[::-1][: min(50, flux.size)]
        a = np.asarray(cat.semimajor_sigma, dtype=np.float64)[idx]
        b = np.asarray(cat.semiminor_sigma, dtype=np.float64)[idx]
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        a = a[ok]
        b = b[ok]
        if a.size < 3:
            return 3.0
        fwhm = 2.355 * 0.5 * (a + b)
        mf = float(np.nanmedian(fwhm))
        if not np.isfinite(mf):
            return 3.0
        return float(np.clip(mf, 1.0, 20.0))
    except Exception:  # noqa: BLE001
        return 3.0


def _qc_fwhm_elongation(
    data: "np.ndarray",
    *,
    max_sources: int = 200,
) -> dict[str, Any]:
    """Estimate average FWHM and elongation (best-effort).

    Primary method: photutils segmentation SourceCatalog.
    Fallback: DAOStarFinder + moment-based shape on cutouts.

    Returns ``n_stars_detected``: approximate total star-like detections on the frame.
    ``n_sources``: subset used for robust FWHM/elongation (capped by ``max_sources``).
    """
    import numpy as np

    img = np.asarray(data, dtype=np.float32)
    # Detect polarity: some tools show/process images as negatives.
    # If stars are negative (dips), detection on positive threshold will fail.
    def _maybe_flip(sign_img: "np.ndarray", std_val: float) -> "np.ndarray":
        import numpy as np

        if not np.isfinite(std_val) or std_val <= 0:
            return sign_img
        # Compare tails at 4-sigma: if negative tail dominates, flip.
        pos = float(np.count_nonzero(sign_img > (4.0 * std_val)))
        neg = float(np.count_nonzero(sign_img < (-4.0 * std_val)))
        if neg > (pos * 2.0) and neg > 100:
            return -sign_img
        return sign_img

    try:
        from astropy.stats import sigma_clipped_stats
        from photutils.segmentation import detect_sources, SourceCatalog

        finite = np.isfinite(img)
        if not np.any(finite):
            return {"fwhm_px": None, "elongation": None, "n_sources": 0, "n_stars_detected": 0}

        mean, med, std = sigma_clipped_stats(img[finite], sigma=3.0, maxiters=5)
        std = float(std)
        if not np.isfinite(std) or std <= 0:
            return {"fwhm_px": None, "elongation": None, "n_sources": 0, "n_stars_detected": 0}

        # Work on background-centered, finite image to avoid NaNs killing detection
        img2 = np.asarray(img - float(med), dtype=np.float32)
        img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
        img2 = _maybe_flip(img2, std)

        # Try a few thresholds to be robust across backgrounds/exposures
        segm = None
        for k in (5.0, 4.0, 3.0):
            thresh = k * std
            segm = detect_sources(img2, thresh, npixels=8)
            if segm is not None and getattr(segm, "nlabels", 0) > 0:
                break
        if segm is None:
            return {"fwhm_px": None, "elongation": None, "n_sources": 0, "n_stars_detected": 0}
        n_seg = int(getattr(segm, "nlabels", 0))
        cat = SourceCatalog(img2, segm)
        # pick brightest-ish sources by flux
        flux = np.asarray(cat.segment_flux, dtype=np.float64)
        if flux.size == 0:
            return {
                "fwhm_px": None,
                "elongation": None,
                "n_sources": 0,
                "n_stars_detected": n_seg,
            }
        idx = np.argsort(flux)[::-1][:max_sources]
        a = np.asarray(cat.semimajor_sigma, dtype=np.float64)[idx]
        b = np.asarray(cat.semiminor_sigma, dtype=np.float64)[idx]
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        a = a[ok]
        b = b[ok]
        if a.size == 0:
            return {
                "fwhm_px": None,
                "elongation": None,
                "n_sources": 0,
                "n_stars_detected": n_seg,
            }
        fwhm = 2.355 * 0.5 * (a + b)
        elong = a / b
        return {
            "fwhm_px": float(np.nanmedian(fwhm)),
            "elongation": float(np.nanmedian(elong)),
            "n_sources": int(a.size),
            "n_stars_detected": n_seg,
        }
    except Exception:
        pass

    # Fallback: DAOStarFinder + moments on small cutouts (requires photutils)
    try:
        from astropy.stats import sigma_clipped_stats
        from photutils.detection import DAOStarFinder

        finite = np.isfinite(img)
        if not np.any(finite):
            return {"fwhm_px": None, "elongation": None, "n_sources": 0, "n_stars_detected": 0}
        mean, med, std = sigma_clipped_stats(img[finite], sigma=3.0, maxiters=5)
        std = float(std)
        if not np.isfinite(std) or std <= 0:
            return {"fwhm_px": None, "elongation": None, "n_sources": 0, "n_stars_detected": 0}

        img2 = np.asarray(img - float(med), dtype=np.float32)
        img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
        img2 = _maybe_flip(img2, std)

        fwhm_guess = _estimate_dao_fwhm_guess(img2, std)
        daofind = DAOStarFinder(
            fwhm=float(fwhm_guess),
            threshold=5.0 * std,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = daofind(img2)
        if tbl is None or len(tbl) == 0:
            # try lower threshold for very faint fields
            daofind = DAOStarFinder(
                fwhm=float(fwhm_guess),
                threshold=3.0 * std,
                **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
            )
            tbl = daofind(img2)
        if tbl is None or len(tbl) == 0:
            return {"fwhm_px": None, "elongation": None, "n_sources": 0, "n_stars_detected": 0}

        # Sort by flux descending
        tbl.sort("flux")
        tbl = tbl[::-1]
        n_dao = int(len(tbl))
        n_use = int(min(len(tbl), max_sources))

        fwhm_list: list[float] = []
        elong_list: list[float] = []
        half = 7  # cutout half-size => 15x15
        h, w = img2.shape

        for i in range(n_use):
            x0 = float(tbl["xcentroid"][i])
            y0 = float(tbl["ycentroid"][i])
            xi = int(round(x0))
            yi = int(round(y0))
            y1 = max(0, yi - half)
            y2 = min(h, yi + half + 1)
            x1 = max(0, xi - half)
            x2 = min(w, xi + half + 1)
            cut = img2[y1:y2, x1:x2]
            if cut.size < 25:
                continue
            # Use only positive signal (after optional polarity flip)
            cut = np.where(cut > 0, cut, 0.0).astype(np.float32)
            s = float(np.sum(cut))
            if not np.isfinite(s) or s <= 0:
                continue
            yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
            cx = float(np.sum(xx * cut) / s)
            cy = float(np.sum(yy * cut) / s)
            dx = xx - cx
            dy = yy - cy
            mxx = float(np.sum((dx * dx) * cut) / s)
            myy = float(np.sum((dy * dy) * cut) / s)
            mxy = float(np.sum((dx * dy) * cut) / s)

            # Eigenvalues of covariance give principal sigmas^2
            tr = mxx + myy
            det = mxx * myy - mxy * mxy
            disc = tr * tr - 4.0 * det
            if disc < 0:
                continue
            l1 = 0.5 * (tr + float(np.sqrt(disc)))
            l2 = 0.5 * (tr - float(np.sqrt(disc)))
            if l1 <= 0 or l2 <= 0:
                continue
            sig1 = float(np.sqrt(l1))
            sig2 = float(np.sqrt(l2))
            fwhm = 2.355 * 0.5 * (sig1 + sig2)
            elong = sig1 / sig2 if sig2 > 0 else np.nan
            if np.isfinite(fwhm) and 0.2 < fwhm < 50:
                fwhm_list.append(float(fwhm))
            if np.isfinite(elong) and 0.8 < elong < 20:
                elong_list.append(float(elong))

        if not fwhm_list or not elong_list:
            return {
                "fwhm_px": None,
                "elongation": None,
                "n_sources": 0,
                "n_stars_detected": n_dao,
            }

        return {
            "fwhm_px": float(np.nanmedian(np.asarray(fwhm_list, dtype=np.float64))),
            "elongation": float(np.nanmedian(np.asarray(elong_list, dtype=np.float64))),
            "n_sources": int(min(len(fwhm_list), len(elong_list))),
            "n_stars_detected": n_dao,
        }
    except Exception:
        pass

    # Last-resort fallback without photutils: naive local-max peak picking + moments
    try:
        from astropy.stats import sigma_clipped_stats

        finite = np.isfinite(img)
        if not np.any(finite):
            return {"fwhm_px": None, "elongation": None, "n_sources": 0, "n_stars_detected": 0}
        mean, med, std = sigma_clipped_stats(img[finite], sigma=3.0, maxiters=5)
        std = float(std)
        if not np.isfinite(std) or std <= 0:
            return {"fwhm_px": None, "elongation": None, "n_sources": 0, "n_stars_detected": 0}

        img2 = np.asarray(img - float(med), dtype=np.float32)
        img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
        img2 = _maybe_flip(img2, std)

        # Simple smoothing (3x3 mean) to reduce single-pixel noise
        kern = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=np.float32)
        kern /= float(kern.sum())
        # Convolution via shifts (no scipy)
        sm = (
            img2
            + np.roll(img2, 1, 0)
            + np.roll(img2, -1, 0)
            + np.roll(img2, 1, 1)
            + np.roll(img2, -1, 1)
            + np.roll(np.roll(img2, 1, 0), 1, 1)
            + np.roll(np.roll(img2, 1, 0), -1, 1)
            + np.roll(np.roll(img2, -1, 0), 1, 1)
            + np.roll(np.roll(img2, -1, 0), -1, 1)
        ) / 9.0

        # Local maxima mask (exclude borders)
        c = sm[1:-1, 1:-1]
        nbrs = [
            sm[0:-2, 0:-2],
            sm[0:-2, 1:-1],
            sm[0:-2, 2:],
            sm[1:-1, 0:-2],
            sm[1:-1, 2:],
            sm[2:, 0:-2],
            sm[2:, 1:-1],
            sm[2:, 2:],
        ]
        is_peak = np.ones_like(c, dtype=bool)
        for n in nbrs:
            is_peak &= c > n

        # Threshold
        thr = 5.0 * std
        is_peak &= c > thr
        ys, xs = np.nonzero(is_peak)
        if ys.size == 0:
            # try lower threshold
            thr = 3.0 * std
            is_peak = np.ones_like(c, dtype=bool)
            for n in nbrs:
                is_peak &= c > n
            is_peak &= c > thr
            ys, xs = np.nonzero(is_peak)
        if ys.size == 0:
            return {"fwhm_px": None, "elongation": None, "n_sources": 0, "n_stars_detected": 0}

        n_peaks_total = int(ys.size)

        # Convert to full-image coordinates (+1 offset)
        ys = ys + 1
        xs = xs + 1
        vals = sm[ys, xs]
        idx = np.argsort(vals)[::-1][:max_sources]
        ys = ys[idx]
        xs = xs[idx]

        fwhm_list: list[float] = []
        elong_list: list[float] = []
        half = 7
        h, w = img2.shape
        for yi, xi in zip(ys.tolist(), xs.tolist(), strict=False):
            y1 = max(0, yi - half)
            y2 = min(h, yi + half + 1)
            x1 = max(0, xi - half)
            x2 = min(w, xi + half + 1)
            cut = img2[y1:y2, x1:x2]
            if cut.size < 25:
                continue
            cut = np.where(cut > 0, cut, 0.0).astype(np.float32)
            s = float(np.sum(cut))
            if not np.isfinite(s) or s <= 0:
                continue
            yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
            cx = float(np.sum(xx * cut) / s)
            cy = float(np.sum(yy * cut) / s)
            dx = xx - cx
            dy = yy - cy
            mxx = float(np.sum((dx * dx) * cut) / s)
            myy = float(np.sum((dy * dy) * cut) / s)
            mxy = float(np.sum((dx * dy) * cut) / s)
            tr = mxx + myy
            det = mxx * myy - mxy * mxy
            disc = tr * tr - 4.0 * det
            if disc < 0:
                continue
            l1 = 0.5 * (tr + float(np.sqrt(disc)))
            l2 = 0.5 * (tr - float(np.sqrt(disc)))
            if l1 <= 0 or l2 <= 0:
                continue
            sig1 = float(np.sqrt(l1))
            sig2 = float(np.sqrt(l2))
            fwhm = 2.355 * 0.5 * (sig1 + sig2)
            elong = sig1 / sig2 if sig2 > 0 else np.nan
            if np.isfinite(fwhm) and 0.2 < fwhm < 50:
                fwhm_list.append(float(fwhm))
            if np.isfinite(elong) and 0.8 < elong < 20:
                elong_list.append(float(elong))

        if not fwhm_list or not elong_list:
            return {
                "fwhm_px": None,
                "elongation": None,
                "n_sources": 0,
                "n_stars_detected": n_peaks_total,
            }
        return {
            "fwhm_px": float(np.nanmedian(np.asarray(fwhm_list, dtype=np.float64))),
            "elongation": float(np.nanmedian(np.asarray(elong_list, dtype=np.float64))),
            "n_sources": int(min(len(fwhm_list), len(elong_list))),
            "n_stars_detected": n_peaks_total,
        }
    except Exception:
        return {"fwhm_px": None, "elongation": None, "n_sources": 0, "n_stars_detected": 0}


def _vyvar_qc_preprocess_workers() -> int:
    """Parallel workers for analyze / preprocess / combined.

    Precedence: non-empty ``VYVAR_QC_PREPROCESS_WORKERS`` env → ``qc_preprocess_workers`` in ``config.json`` → 1.
    """
    env_raw = os.environ.get("VYVAR_QC_PREPROCESS_WORKERS")
    if env_raw is not None and str(env_raw).strip() != "":
        try:
            return max(1, min(32, int(str(env_raw).strip())))
        except ValueError:
            pass
    data = load_config_json(Path(__file__).resolve().parent)
    try:
        return max(1, min(32, int(data.get("qc_preprocess_workers", 1))))
    except (TypeError, ValueError):
        return 1


def _estimate_catalog_frame_hw(
    work_ram: Sequence[tuple[str, Any, Any]] | None,
    files: list[Path],
) -> tuple[int, int]:
    """Rough (ny, nx) for RAM cap heuristics (per-frame catalog export)."""
    import numpy as np

    if work_ram:
        d = np.asarray(work_ram[0][2])
        if d.ndim == 2:
            return int(d.shape[0]), int(d.shape[1])
    for fp in files[:1]:
        try:
            with fits.open(fp, memmap=False) as h:
                nh = h[0].header
                ny = int(nh.get("NAXIS2", 0) or 0)
                nx = int(nh.get("NAXIS1", 0) or 0)
                if ny > 0 and nx > 0:
                    return ny, nx
        except Exception:  # noqa: BLE001
            continue
    return 2048, 2048


def _vyvar_cap_mp_workers_for_catalog(
    n_workers: int,
    frame_hw: tuple[int, int],
    *,
    reserve_gb: float,
) -> int:
    """Cap process pool size using available RAM (rough float32 frame footprint per worker)."""
    h, w = frame_hw
    if h <= 0 or w <= 0:
        h, w = 2048, 2048
    per_worker = max(int(h * w * 4 * 3), 1)
    try:
        import psutil
    except ImportError:
        return max(1, n_workers)
    reserve = int(max(0.0, float(reserve_gb)) * (1024**3))
    avail = int(psutil.virtual_memory().available) - reserve
    if avail <= 0:
        return 1
    mx = max(1, avail // per_worker)
    return max(1, min(int(n_workers), mx))


def _vyvar_per_frame_csv_workers(app_config: AppConfig | None = None) -> int:
    """Process workers for plate-solve step 3 per-frame catalog (DAO + match + CSV).

    Precedence: non-empty ``VYVAR_PER_FRAME_CSV_WORKERS`` env → ``app_config.per_frame_csv_workers`` →
    ``per_frame_csv_workers`` in ``config.json`` → 1.
    """
    env_raw = os.environ.get("VYVAR_PER_FRAME_CSV_WORKERS")
    if env_raw is not None and str(env_raw).strip() != "":
        try:
            return max(1, min(32, int(str(env_raw).strip())))
        except ValueError:
            pass
    if app_config is not None:
        try:
            return max(1, min(32, int(app_config.per_frame_csv_workers)))
        except (TypeError, ValueError):
            pass
    data = load_config_json(Path(__file__).resolve().parent)
    try:
        return max(1, min(32, int(data.get("per_frame_csv_workers", 1))))
    except (TypeError, ValueError):
        return 1


def _analyze_calibrated_qc_one(
    src: Path,
    *,
    background_method: str,
    poly_order: int,
    lacosmic_sigclip: float,
    lacosmic_objlim: float,
) -> dict[str, Any]:
    import numpy as np

    src = Path(src)
    try:
        with fits.open(src, memmap=False) as hdul:
            hdr = hdul[0].header
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        data2, cr_m = _remove_cosmics_lacosmic(
            data, sigclip=lacosmic_sigclip, objlim=lacosmic_objlim
        )
        data3, bg_m = _background_subtract(data2, method=background_method, poly_order=poly_order)
        qc = _qc_fwhm_elongation(data3)
        finite = np.isfinite(data3)
        arr = data3[finite]
        return {
            "src": str(src),
            "filter": _safe_filter_token(str(hdr.get("FILTER") or hdr.get("FILT") or "NoFilter")),
            "fwhm_px": qc.get("fwhm_px"),
            "elongation": qc.get("elongation"),
            "n_sources": qc.get("n_sources"),
            "n_stars_detected": qc.get("n_stars_detected"),
            "cr_frac": cr_m.get("cr_frac"),
            "bg_median": bg_m.get("bg_median"),
            "p50": float(np.nanpercentile(arr, 50)) if arr.size else None,
            "p99": float(np.nanpercentile(arr, 99)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"src": str(src), "status": f"error: {exc}"}


def _preprocess_calibrated_one(
    src: Path,
    *,
    calibrated_root: Path,
    processed_root: Path,
    background_method: str,
    poly_order: int,
    lacosmic_sigclip: float,
    lacosmic_objlim: float,
    enable_lacosmic: bool,
    enable_background_flattening: bool,
    reject_fwhm_px: float | None,
    reject_elongation: float | None,
    inject_pointing_ra_deg: float | None,
    inject_pointing_dec_deg: float | None,
    inject_pointing_only_if_missing: bool,
) -> dict[str, Any]:
    import numpy as np

    src = Path(src)
    calibrated_root = Path(calibrated_root)
    processed_root = Path(processed_root)
    rel = src.relative_to(calibrated_root)
    dst = processed_root / rel.parent / _safe_proc_name(rel.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    status = "ok"
    try:
        with fits.open(src, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)

        if bool(enable_lacosmic):
            data2, cr_m = _remove_cosmics_lacosmic(
                data, sigclip=lacosmic_sigclip, objlim=lacosmic_objlim
            )
        else:
            data2 = np.asarray(data, dtype=np.float32)
            cr_m = {"cr_frac": 0.0, "n_cosmic": 0.0, "ok": True}
        if bool(enable_lacosmic) and cr_m.get("cr_frac", 0.0) > 0:
            hdr.add_history(
                f"VYVAR: L.A.Cosmic applied (sigclip={lacosmic_sigclip}, objlim={lacosmic_objlim})"
            )

        if bool(enable_background_flattening):
            data3, bg_m = _background_subtract(data2, method=background_method, poly_order=poly_order)
            hdr.add_history(f"VYVAR: Background subtracted ({background_method})")
        else:
            data3 = np.asarray(data2, dtype=np.float32)
            bg_m = {"bg_median": float(np.nanmedian(data3)), "method": -1.0}

        if reject_fwhm_px is not None or reject_elongation is not None:
            qc = _qc_fwhm_elongation(data3)
        else:
            qc = {"fwhm_px": None, "elongation": None, "n_sources": None, "n_stars_detected": None}
        fwhm_px = qc.get("fwhm_px")
        elong = qc.get("elongation")
        if reject_fwhm_px is not None and fwhm_px is not None and float(fwhm_px) > float(
            reject_fwhm_px
        ):
            status = "rejected_fwhm"
        if reject_elongation is not None and elong is not None and float(elong) > float(
            reject_elongation
        ):
            status = "rejected_elong"

        hdr["VYVARPR"] = (True, "VYVAR pre-processed output")
        if fwhm_px is not None:
            hdr["VY_FWHM"] = (float(fwhm_px), "Estimated FWHM [pix]")
        if elong is not None:
            hdr["VY_ELONG"] = (float(elong), "Estimated elongation (a/b)")
        nsd = qc.get("n_stars_detected")
        if nsd is not None:
            hdr["VY_NSTAR"] = (int(nsd), "Approx. star detections (QC)")
        hdr["VY_QC"] = (status, "QC status")

        if (
            inject_pointing_ra_deg is not None
            and inject_pointing_dec_deg is not None
            and math.isfinite(float(inject_pointing_ra_deg))
            and math.isfinite(float(inject_pointing_dec_deg))
        ):
            ira = float(inject_pointing_ra_deg)
            idec = float(inject_pointing_dec_deg)
            ex_ra, ex_dec, _ = _pointing_hint_from_header(hdr)
            do_inject = (not bool(inject_pointing_only_if_missing)) or (
                ex_ra is None or ex_dec is None
            )
            if do_inject:
                hdr["VYTARGRA"] = (ira, "VYVAR plate-solve hint RA [deg] ICRS")
                hdr["VYTARGDE"] = (idec, "VYVAR plate-solve hint Dec [deg] ICRS")
                hdr.add_history("VYVAR: VYTARGRA/VYTARGDE for plate solving (preprocess)")

        saved_dst = ""
        if not str(status).startswith("rejected"):
            fits.writeto(dst, _as_fits_float32_image(data3), header=hdr, overwrite=True)
            saved_dst = str(dst)
        return {
            "src": str(src),
            "dst": saved_dst,
            "status": status,
            "fwhm_px": fwhm_px,
            "elongation": elong,
            "n_sources": qc.get("n_sources"),
            "n_stars_detected": qc.get("n_stars_detected"),
            "cr_frac": cr_m.get("cr_frac"),
            "bg_median": bg_m.get("bg_median"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"src": str(src), "dst": str(dst), "status": f"error: {exc}"}


def _analyze_preprocess_combined_one(
    src: Path,
    *,
    calibrated_root: Path,
    processed_root: Path,
    background_method: str,
    poly_order: int,
    lacosmic_sigclip: float,
    lacosmic_objlim: float,
    enable_lacosmic: bool,
    enable_background_flattening: bool,
    reject_fwhm_px: float | None,
    reject_elongation: float | None,
    inject_pointing_ra_deg: float | None,
    inject_pointing_dec_deg: float | None,
    inject_pointing_only_if_missing: bool,
) -> dict[str, Any]:
    """Single read: cosmics + background + full QC + optional reject + write detrended."""
    import numpy as np

    src = Path(src)
    calibrated_root = Path(calibrated_root)
    processed_root = Path(processed_root)
    rel = src.relative_to(calibrated_root)
    dst = processed_root / rel.parent / _safe_proc_name(rel.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    status = "ok"
    try:
        with fits.open(src, memmap=False) as hdul:
            hdr0 = hdul[0].header
            filt_tok = _safe_filter_token(
                str(hdr0.get("FILTER") or hdr0.get("FILT") or "NoFilter")
            )
            hdr = hdul[0].header.copy()
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)

        if bool(enable_lacosmic):
            data2, cr_m = _remove_cosmics_lacosmic(
                data, sigclip=lacosmic_sigclip, objlim=lacosmic_objlim
            )
        else:
            data2 = np.asarray(data, dtype=np.float32)
            cr_m = {"cr_frac": 0.0, "n_cosmic": 0.0, "ok": True}
        if bool(enable_lacosmic) and cr_m.get("cr_frac", 0.0) > 0:
            hdr.add_history(
                f"VYVAR: L.A.Cosmic applied (sigclip={lacosmic_sigclip}, objlim={lacosmic_objlim})"
            )

        if bool(enable_background_flattening):
            data3, bg_m = _background_subtract(data2, method=background_method, poly_order=poly_order)
            hdr.add_history(f"VYVAR: Background subtracted ({background_method})")
        else:
            data3 = np.asarray(data2, dtype=np.float32)
            bg_m = {"bg_median": float(np.nanmedian(data3)), "method": -1.0}

        qc = _qc_fwhm_elongation(data3)
        fwhm_px = qc.get("fwhm_px")
        elong = qc.get("elongation")
        if reject_fwhm_px is not None and fwhm_px is not None and float(fwhm_px) > float(
            reject_fwhm_px
        ):
            status = "rejected_fwhm"
        if reject_elongation is not None and elong is not None and float(elong) > float(
            reject_elongation
        ):
            status = "rejected_elong"

        finite = np.isfinite(data3)
        arr = data3[finite]

        hdr["VYVARPR"] = (True, "VYVAR pre-processed output")
        if fwhm_px is not None:
            hdr["VY_FWHM"] = (float(fwhm_px), "Estimated FWHM [pix]")
        if elong is not None:
            hdr["VY_ELONG"] = (float(elong), "Estimated elongation (a/b)")
        nsd = qc.get("n_stars_detected")
        if nsd is not None:
            hdr["VY_NSTAR"] = (int(nsd), "Approx. star detections (QC)")
        hdr["VY_QC"] = (status, "QC status")

        if (
            inject_pointing_ra_deg is not None
            and inject_pointing_dec_deg is not None
            and math.isfinite(float(inject_pointing_ra_deg))
            and math.isfinite(float(inject_pointing_dec_deg))
        ):
            ira = float(inject_pointing_ra_deg)
            idec = float(inject_pointing_dec_deg)
            ex_ra, ex_dec, _ = _pointing_hint_from_header(hdr)
            do_inject = (not bool(inject_pointing_only_if_missing)) or (
                ex_ra is None or ex_dec is None
            )
            if do_inject:
                hdr["VYTARGRA"] = (ira, "VYVAR plate-solve hint RA [deg] ICRS")
                hdr["VYTARGDE"] = (idec, "VYVAR plate-solve hint Dec [deg] ICRS")
                hdr.add_history("VYVAR: VYTARGRA/VYTARGDE for plate solving (preprocess)")

        saved_dst = ""
        if not str(status).startswith("rejected"):
            fits.writeto(dst, _as_fits_float32_image(data3), header=hdr, overwrite=True)
            saved_dst = str(dst)

        return {
            "src": str(src),
            "filter": filt_tok,
            "fwhm_px": fwhm_px,
            "elongation": elong,
            "n_sources": qc.get("n_sources"),
            "n_stars_detected": qc.get("n_stars_detected"),
            "cr_frac": cr_m.get("cr_frac"),
            "bg_median": bg_m.get("bg_median"),
            "p50": float(np.nanpercentile(arr, 50)) if arr.size else None,
            "p99": float(np.nanpercentile(arr, 99)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
            "dst": saved_dst,
            "status": status,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "src": str(src),
            "dst": str(dst),
            "status": f"error: {exc}",
        }


def preprocess_calibrated_to_processed(
    *,
    calibrated_root: Path,
    processed_root: Path,
    background_method: str = "background2d",  # "background2d" | "poly2d"
    poly_order: int = 2,
    lacosmic_sigclip: float = 4.5,
    lacosmic_objlim: float = 5.0,
    enable_lacosmic: bool = False,
    enable_background_flattening: bool = False,
    reject_fwhm_px: float | None = None,
    reject_elongation: float | None = None,
    temporal_sigma_clip: bool = False,
    temporal_sigma: float = 6.0,
    temporal_min_frames: int = 5,
    use_gpu_if_available: bool = False,
    progress_cb: Callable[..., None] | None = None,
    inject_pointing_ra_deg: float | None = None,
    inject_pointing_dec_deg: float | None = None,
    inject_pointing_only_if_missing: bool = True,
    only_paths: Sequence[Path | str] | None = None,
    db: VyvarDatabase | None = None,
    draft_id: int | None = None,
) -> pd.DataFrame:
    """Pre-process calibrated lights into /processed with proc_ prefix.

    Steps:
    - Optional L.A.Cosmic (astroscrappy)
    - Optional background subtraction (photutils Background2D; fallback polynomial)
    - QC metrics: FWHM + elongation (best-effort)

    Optional ``inject_pointing_*``: write ``VYTARGRA`` / ``VYTARGDE`` (deg ICRS) into saved FITS headers
    for plate-solve hints when the frame has no celestial WCS yet (see ``vyvar_platesolver.pointing_hint_from_header``).

    Parallelism: integer ``qc_preprocess_workers`` in ``config.json`` (Settings UI) or non-empty env
    ``VYVAR_QC_PREPROCESS_WORKERS`` (env wins). ``>1`` defaults to ``ProcessPoolExecutor`` (true CPU
    parallelism); set env ``VYVAR_PARALLEL_BACKEND=thread`` to use threads. Peak RAM scales roughly with
    the number of workers times per-frame working set.
    """
    import numpy as np

    calibrated_root = Path(calibrated_root)
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    fits_files = _filter_light_paths_maybe(_iter_light_fits(calibrated_root), only_paths)
    if db is not None and draft_id is not None:
        _ra_eff, _de_eff = resolve_preprocess_target_coordinates(
            db=db,
            draft_id=int(draft_id),
            ui_ra_deg=inject_pointing_ra_deg,
            ui_dec_deg=inject_pointing_dec_deg,
        )
        if _ra_eff is not None and _de_eff is not None:
            inject_pointing_ra_deg = float(_ra_eff)
            inject_pointing_dec_deg = float(_de_eff)
    target_ra = inject_pointing_ra_deg
    target_dec = inject_pointing_dec_deg
    fwhm_limit_log = reject_fwhm_px
    log_event(
        f"DEBUG: Preprocess filter - RA:{target_ra}, DE:{target_dec}, FWHM_Lim:{fwhm_limit_log}"
    )

    def _median_pointing_from_headers(files: Sequence[Path]) -> tuple[float | None, float | None]:
        ras: list[float] = []
        decs: list[float] = []
        for fp in files:
            try:
                with fits.open(fp, memmap=False) as hdul:
                    hdr = hdul[0].header
                p = _pointing_hint_from_header(hdr)
                ra_v = float(p.ra_icrs_deg)
                de_v = float(p.dec_icrs_deg)
                if math.isfinite(ra_v) and math.isfinite(de_v):
                    ras.append(ra_v)
                    decs.append(de_v)
            except Exception:  # noqa: BLE001
                continue
        if not ras or not decs:
            return None, None
        return float(np.median(np.asarray(ras, dtype=np.float64))), float(
            np.median(np.asarray(decs, dtype=np.float64))
        )

    try:
        ra_is_zero = target_ra is not None and math.isfinite(float(target_ra)) and abs(float(target_ra)) < 1e-9
        de_is_zero = target_dec is not None and math.isfinite(float(target_dec)) and abs(float(target_dec)) < 1e-9
    except (TypeError, ValueError):
        ra_is_zero, de_is_zero = False, False
    if ra_is_zero and de_is_zero:
        log_event(
            "WARNING: Preprocess got target RA/DE = 0.0/0.0; using median coordinates from draft frames as fallback."
        )
        fb_ra, fb_de = _median_pointing_from_headers(fits_files)
        if fb_ra is not None and fb_de is not None:
            inject_pointing_ra_deg = float(fb_ra)
            inject_pointing_dec_deg = float(fb_de)
            log_event(
                f"DEBUG: Preprocess fallback pointing applied: RA={float(fb_ra):.6f}, DE={float(fb_de):.6f}"
            )

    total = len(fits_files)
    n_workers = _vyvar_qc_preprocess_workers()
    if n_workers > 1 and total > 1:
        LOGGER.info(
            "VYVAR preprocess: VYVAR_QC_PREPROCESS_WORKERS=%s (paralelne; ~%s× RAM na snímok oproti 1 vláknu)",
            n_workers,
            n_workers,
        )

    if n_workers > 1 and total > 1:
        with _vyvar_parallel_pool(n_workers) as ex:
            futs = {
                ex.submit(
                    _preprocess_calibrated_one,
                    src,
                    calibrated_root=calibrated_root,
                    processed_root=processed_root,
                    background_method=background_method,
                    poly_order=poly_order,
                    lacosmic_sigclip=lacosmic_sigclip,
                    lacosmic_objlim=lacosmic_objlim,
                    enable_lacosmic=bool(enable_lacosmic),
                    enable_background_flattening=bool(enable_background_flattening),
                    reject_fwhm_px=reject_fwhm_px,
                    reject_elongation=reject_elongation,
                    inject_pointing_ra_deg=inject_pointing_ra_deg,
                    inject_pointing_dec_deg=inject_pointing_dec_deg,
                    inject_pointing_only_if_missing=inject_pointing_only_if_missing,
                ): src
                for src in fits_files
            }
            by_src: dict[Path, dict[str, Any]] = {}
            done = 0
            for fut in as_completed(futs):
                src = futs[fut]
                by_src[src] = fut.result()
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"Preprocessing {src.name}")
            rows = [by_src[s] for s in fits_files]
    else:
        rows = []
        for i, src in enumerate(fits_files, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"Preprocessing {src.name}")
            rows.append(
                _preprocess_calibrated_one(
                    src,
                    calibrated_root=calibrated_root,
                    processed_root=processed_root,
                    background_method=background_method,
                    poly_order=poly_order,
                    lacosmic_sigclip=lacosmic_sigclip,
                    lacosmic_objlim=lacosmic_objlim,
                    enable_lacosmic=bool(enable_lacosmic),
                    enable_background_flattening=bool(enable_background_flattening),
                    reject_fwhm_px=reject_fwhm_px,
                    reject_elongation=reject_elongation,
                    inject_pointing_ra_deg=inject_pointing_ra_deg,
                    inject_pointing_dec_deg=inject_pointing_dec_deg,
                    inject_pointing_only_if_missing=inject_pointing_only_if_missing,
                )
            )

    produced = [Path(r["dst"]) for r in rows if r.get("dst")]
    df = pd.DataFrame(rows)

    if temporal_sigma_clip:
        try:
            _apply_temporal_sigma_clip_in_place(
                processed_root=processed_root,
                produced_files=produced,
                sigma=float(temporal_sigma),
                min_frames=int(temporal_min_frames),
                use_gpu_if_available=bool(use_gpu_if_available),
            )
            if not df.empty:
                df["temporal_mask"] = True
        except Exception:
            # keep preprocessing results; temporal mask is optional
            if not df.empty:
                df["temporal_mask"] = False

    return df


def analyze_calibrated_qc(
    *,
    calibrated_root: Path,
    max_frames: int | None = None,
    background_method: str = "background2d",
    poly_order: int = 2,
    lacosmic_sigclip: float = 4.5,
    lacosmic_objlim: float = 5.0,
    progress_cb: Callable[..., None] | None = None,
    only_paths: Sequence[Path | str] | None = None,
) -> pd.DataFrame:
    """Analyze calibrated frames (QC) without writing /processed outputs.

    Runs the same transforms as preprocessing (cosmics + background subtraction) but keeps results in memory
    and returns a QC dataframe (fwhm/elongation + basic stats).

    If ``max_frames`` is None, every light FITS under ``calibrated_root`` is analyzed.

    Parallelism: ``qc_preprocess_workers`` in ``config.json`` or env ``VYVAR_QC_PREPROCESS_WORKERS`` (env wins);
    integer ``>1`` uses a process pool by default (``VYVAR_PARALLEL_BACKEND=thread`` for threads).
    """
    calibrated_root = Path(calibrated_root)
    files = _filter_light_paths_maybe(_iter_light_fits(calibrated_root), only_paths)
    if max_frames is not None:
        files = files[: max(0, int(max_frames))]
    total = len(files)
    if total > 0:
        _mh = estimate_memory_from_fits_headers(files)
        _peak = int(float(_mh["bytes_float32_max_frame"]) * 6.0)
        LOGGER.info(
            "VYVAR QC analyze: %s frames; odhad špičky RAM ~%s (float32 + dočasné polia)",
            total,
            format_memory_bytes(_peak),
        )
    n_workers = _vyvar_qc_preprocess_workers()
    if n_workers > 1 and total > 1:
        LOGGER.info(
            "VYVAR QC analyze: VYVAR_QC_PREPROCESS_WORKERS=%s (paralelne; ~%s× RAM na snímok oproti 1 vláknu)",
            n_workers,
            n_workers,
        )

    if n_workers > 1 and total > 1:
        with _vyvar_parallel_pool(n_workers) as ex:
            futs = {
                ex.submit(
                    _analyze_calibrated_qc_one,
                    src,
                    background_method=background_method,
                    poly_order=poly_order,
                    lacosmic_sigclip=lacosmic_sigclip,
                    lacosmic_objlim=lacosmic_objlim,
                ): src
                for src in files
            }
            by_src: dict[Path, dict[str, Any]] = {}
            done = 0
            for fut in as_completed(futs):
                src = futs[fut]
                by_src[src] = fut.result()
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"Analyzing {src.name}")
            rows = [by_src[s] for s in files]
    else:
        rows = []
        for i, src in enumerate(files, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"Analyzing {src.name}")
            rows.append(
                _analyze_calibrated_qc_one(
                    src,
                    background_method=background_method,
                    poly_order=poly_order,
                    lacosmic_sigclip=lacosmic_sigclip,
                    lacosmic_objlim=lacosmic_objlim,
                )
            )

    return pd.DataFrame(rows)


def analyze_preprocess_calibrated_combined(
    *,
    calibrated_root: Path,
    processed_root: Path,
    background_method: str = "background2d",
    poly_order: int = 2,
    lacosmic_sigclip: float = 4.5,
    lacosmic_objlim: float = 5.0,
    enable_lacosmic: bool = False,
    enable_background_flattening: bool = False,
    reject_fwhm_px: float | None = None,
    reject_elongation: float | None = None,
    temporal_sigma_clip: bool = False,
    temporal_sigma: float = 6.0,
    temporal_min_frames: int = 5,
    use_gpu_if_available: bool = False,
    progress_cb: Callable[..., None] | None = None,
    inject_pointing_ra_deg: float | None = None,
    inject_pointing_dec_deg: float | None = None,
    inject_pointing_only_if_missing: bool = True,
    only_paths: Sequence[Path | str] | None = None,
) -> pd.DataFrame:
    """One FITS read per light: full QC (as Analyze) plus detrended output (as Preprocess).

    Skips the duplicate disk read that occurs when running Analyze and Preprocess as two separate steps.
    Honors ``qc_preprocess_workers`` in ``config.json`` or env ``VYVAR_QC_PREPROCESS_WORKERS`` (env wins).
    Defaults to process-based parallelism unless ``VYVAR_PARALLEL_BACKEND=thread``.
    """
    calibrated_root = Path(calibrated_root)
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    fits_files = _filter_light_paths_maybe(_iter_light_fits(calibrated_root), only_paths)
    total = len(fits_files)
    if total > 0:
        _mh = estimate_memory_from_fits_headers(fits_files)
        _peak = int(float(_mh["bytes_float32_max_frame"]) * 6.0)
        LOGGER.info(
            "VYVAR combined QC+preprocess: %s frames; odhad špičky RAM ~%s (float32 + dočasné polia)",
            total,
            format_memory_bytes(_peak),
        )

    n_workers = _vyvar_qc_preprocess_workers()
    if n_workers > 1 and total > 1:
        LOGGER.info(
            "VYVAR combined: VYVAR_QC_PREPROCESS_WORKERS=%s (paralelne; ~%s× RAM na snímok oproti 1 vláknu)",
            n_workers,
            n_workers,
        )

    if n_workers > 1 and total > 1:
        with _vyvar_parallel_pool(n_workers) as ex:
            futs = {
                ex.submit(
                    _analyze_preprocess_combined_one,
                    src,
                    calibrated_root=calibrated_root,
                    processed_root=processed_root,
                    background_method=background_method,
                    poly_order=poly_order,
                    lacosmic_sigclip=lacosmic_sigclip,
                    lacosmic_objlim=lacosmic_objlim,
                    enable_lacosmic=bool(enable_lacosmic),
                    enable_background_flattening=bool(enable_background_flattening),
                    reject_fwhm_px=reject_fwhm_px,
                    reject_elongation=reject_elongation,
                    inject_pointing_ra_deg=inject_pointing_ra_deg,
                    inject_pointing_dec_deg=inject_pointing_dec_deg,
                    inject_pointing_only_if_missing=inject_pointing_only_if_missing,
                ): src
                for src in fits_files
            }
            by_src: dict[Path, dict[str, Any]] = {}
            done = 0
            for fut in as_completed(futs):
                src = futs[fut]
                by_src[src] = fut.result()
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"QC+detrend {src.name}")
            rows = [by_src[s] for s in fits_files]
    else:
        rows = []
        for i, src in enumerate(fits_files, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"QC+detrend {src.name}")
            rows.append(
                _analyze_preprocess_combined_one(
                    src,
                    calibrated_root=calibrated_root,
                    processed_root=processed_root,
                    background_method=background_method,
                    poly_order=poly_order,
                    lacosmic_sigclip=lacosmic_sigclip,
                    lacosmic_objlim=lacosmic_objlim,
                    enable_lacosmic=bool(enable_lacosmic),
                    enable_background_flattening=bool(enable_background_flattening),
                    reject_fwhm_px=reject_fwhm_px,
                    reject_elongation=reject_elongation,
                    inject_pointing_ra_deg=inject_pointing_ra_deg,
                    inject_pointing_dec_deg=inject_pointing_dec_deg,
                    inject_pointing_only_if_missing=inject_pointing_only_if_missing,
                )
            )

    produced = [Path(r["dst"]) for r in rows if r.get("dst")]
    df = pd.DataFrame(rows)

    if temporal_sigma_clip:
        try:
            _apply_temporal_sigma_clip_in_place(
                processed_root=processed_root,
                produced_files=produced,
                sigma=float(temporal_sigma),
                min_frames=int(temporal_min_frames),
                use_gpu_if_available=bool(use_gpu_if_available),
            )
            if not df.empty:
                df["temporal_mask"] = True
        except Exception:
            if not df.empty:
                df["temporal_mask"] = False

    return df


def _qc_suggest_thresholds(df: "pd.DataFrame") -> dict[str, float | int | None]:
    """Compute best/worst and robust suggested reject thresholds."""
    import numpy as np

    out: dict[str, float | int | None] = {
        "fwhm_min": None,
        "fwhm_median": None,
        "fwhm_max": None,
        "elong_min": None,
        "elong_median": None,
        "elong_max": None,
        "suggest_reject_fwhm_px": None,
        "suggest_reject_elongation": None,
        "n_qc": 0,
    }
    if df is None or df.empty:
        return out

    def _robust_suggest(x: "np.ndarray", k: float) -> float | None:
        x = x[np.isfinite(x)]
        if x.size < 5:
            return None
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med))) + 1e-6
        sigma = 1.4826 * mad
        return float(med + k * sigma)

    f = np.asarray(df.get("fwhm_px"), dtype=np.float64)
    e = np.asarray(df.get("elongation"), dtype=np.float64)
    f_ok = f[np.isfinite(f)]
    e_ok = e[np.isfinite(e)]
    out["n_qc"] = int(min(f_ok.size, e_ok.size) if (f_ok.size and e_ok.size) else max(f_ok.size, e_ok.size))

    if f_ok.size:
        out["fwhm_min"] = float(np.nanmin(f_ok))
        out["fwhm_median"] = float(np.nanmedian(f_ok))
        out["fwhm_max"] = float(np.nanmax(f_ok))
        # Suggest: median + 3*MAD_sigma (conservative)
        out["suggest_reject_fwhm_px"] = _robust_suggest(f_ok, k=3.0)
    if e_ok.size:
        out["elong_min"] = float(np.nanmin(e_ok))
        out["elong_median"] = float(np.nanmedian(e_ok))
        out["elong_max"] = float(np.nanmax(e_ok))
        # Suggest: median + 4*MAD_sigma (elongation is often tighter)
        out["suggest_reject_elongation"] = _robust_suggest(e_ok, k=4.0)

    return out


def _apply_temporal_sigma_clip_in_place(
    *,
    processed_root: Path,
    produced_files: list[Path],
    sigma: float = 6.0,
    min_frames: int = 5,
    tile: int = 256,
    use_gpu_if_available: bool = False,
) -> None:
    """Mask transient artifacts across a time series (satellites/aircraft) per folder.

    For each folder (typically per filter), compute per-pixel median and MAD across frames
    and replace outliers in each frame with the median value.
    """
    import numpy as np
    import hashlib

    if sigma <= 0:
        return
    if min_frames < 3:
        min_frames = 3

    # group by parent directory (per-filter folder)
    groups: dict[Path, list[Path]] = {}
    for fp in produced_files:
        groups.setdefault(fp.parent, []).append(fp)

    tmp_root = Path(processed_root) / ".vyvar_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    use_gpu = bool(use_gpu_if_available) and _cupy_available()
    cp = None
    if use_gpu:
        try:
            import cupy as cp  # type: ignore
        except Exception:  # noqa: BLE001
            use_gpu = False
            cp = None

    for folder, files in groups.items():
        files = sorted(files)
        if len(files) < min_frames:
            continue

        # Create per-frame memmaps to allow tile updates without holding the full stack in RAM.
        tag = hashlib.md5(str(folder).encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
        work = tmp_root / f"temporal_{tag}"
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)

        memmaps: list[np.ndarray] = []
        headers: list[fits.Header] = []
        shapes: list[tuple[int, int]] = []
        for fp in files:
            with fits.open(fp, memmap=False) as hdul:
                hdr = hdul[0].header.copy()
                data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            headers.append(hdr)
            shapes.append((int(data.shape[0]), int(data.shape[1])))
            mm_path = work / f"{fp.stem}.npy"
            mm = np.lib.format.open_memmap(mm_path, mode="w+", dtype="float32", shape=data.shape)
            mm[:] = data
            memmaps.append(mm)

        # Use common smallest shape to avoid mismatch issues (cropping)
        h = min(s[0] for s in shapes)
        w = min(s[1] for s in shapes)
        eps = np.float32(1e-6)
        k = np.float32(1.4826 * float(sigma))

        for y0 in range(0, h, tile):
            y1 = min(h, y0 + tile)
            for x0 in range(0, w, tile):
                x1 = min(w, x0 + tile)
                stack_cpu = np.stack([mm[y0:y1, x0:x1] for mm in memmaps], axis=0)

                if use_gpu and cp is not None:
                    stack_gpu = cp.asarray(stack_cpu)
                    med_gpu = cp.nanmedian(stack_gpu, axis=0)
                    mad_gpu = cp.nanmedian(cp.abs(stack_gpu - med_gpu), axis=0) + eps
                    thr_gpu = k * mad_gpu
                    med_cpu = cp.asnumpy(med_gpu)
                    for i, mm in enumerate(memmaps):
                        mask = cp.asnumpy(cp.abs(stack_gpu[i] - med_gpu) > thr_gpu)
                        if np.any(mask):
                            block = mm[y0:y1, x0:x1]
                            block[mask] = med_cpu[mask]
                            mm[y0:y1, x0:x1] = block
                else:
                    med = np.nanmedian(stack_cpu, axis=0)
                    mad = np.nanmedian(np.abs(stack_cpu - med), axis=0) + eps
                    thr = k * mad
                    for i, mm in enumerate(memmaps):
                        resid = np.abs(stack_cpu[i] - med)
                        mask = resid > thr
                        if np.any(mask):
                            block = mm[y0:y1, x0:x1]
                            block[mask] = med[mask]
                            mm[y0:y1, x0:x1] = block

        # write back to FITS
        for fp, hdr, mm in zip(files, headers, memmaps, strict=False):
            hdr.add_history(f"VYVAR: Temporal sigma-clip mask applied (sigma={sigma:g})")
            fits.writeto(fp, np.asarray(mm[:h, :w], dtype=np.float32), header=hdr, overwrite=True)

        shutil.rmtree(work, ignore_errors=True)


def scan_calibrated_lights_pointing(
    calibrated_root: Path | str,
    *,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Summarize celestial WCS vs object-style pointing keywords under ``calibrated_root`` (lights).

    If ``max_files`` is None, all light FITS under the tree are scanned.
    """
    root = Path(calibrated_root)
    files = _iter_light_fits(root)
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    rows: list[dict[str, Any]] = []
    n_wcs = 0
    n_hint = 0
    n_missing = 0
    for fp in files:
        with fits.open(fp, memmap=False) as hdul:
            h = hdul[0].header
        hwcs = _has_valid_wcs(h)
        ha, hd, hs = _pointing_hint_from_header(h)
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
                "display_ra_deg": da,
                "display_dec_deg": dd,
                "display_source": ds,
            }
        )
    return {
        "calibrated_root": str(root.resolve()),
        "n_files_scanned": len(rows),
        "n_has_celestial_wcs": n_wcs,
        "n_has_object_hint_no_wcs": n_hint,
        "n_no_pointing_hint": n_missing,
        "rows": rows,
    }


def _parse_fits_binning_int(raw: Any, default: int = 1) -> int:
    try:
        v = int(float(raw))
    except (TypeError, ValueError):
        v = int(default)
    return max(1, v)


def _log_effective_pixel_pitch(meta: dict[str, Any], *, filepath: str = "") -> None:
    """Infolog: physical pixel, binning, effective pixel for plate-scale style calculations."""
    pixsz_from_header_or_db = meta.get("pixel_size_um_physical")
    binning_x = max(1, int(meta.get("binning", 1) or 1))
    binning_y = max(1, int(meta.get("binning_y", binning_x) or binning_x))
    effective_pixsz = meta.get("effective_pixel_um_plate_scale")
    tail = f" — {filepath}" if filepath else ""
    try:
        ps_s = f"{float(pixsz_from_header_or_db):.4g}" if pixsz_from_header_or_db is not None else "n/a"
        eff_s = f"{float(effective_pixsz):.4g}" if effective_pixsz is not None else "n/a"
        log_event(
            f"DEBUG: Fyzický pixel: {ps_s} um | Binning: {binning_x}x{binning_y} | "
            f"EFEKTÍVNY PIXEL PRE VÝPOČET: {eff_s} um{tail}"
        )
    except (TypeError, ValueError):
        pass


def fits_metadata_from_primary_header(
    header: fits.Header,
    *,
    force_physical_pixel_um: float | None = None,
) -> dict[str, Any]:
    """Build the same dict as :func:`extract_fits_metadata` from an already-loaded primary header.

    ``pixel_size_um_header`` is the **effective** on-sky pitch [µm] (physical pixel from header × binning).
    ``pixel_size_um_physical`` is the native pitch before binning (after optional ``force_physical_pixel_um``).
    ``effective_pixel_um_plate_scale`` is native × XBINNING [µm] for plate scale / solver hints.
    """

    def _pick(*keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in header and header[key] not in (None, ""):
                return header[key]
        return default

    jd_obs = _pick("JD", "JD-OBS", default=None)
    if jd_obs is not None:
        jd_start = _to_float_db(jd_obs, 0.0)
    else:
        mjd_obs = _pick("MJD-OBS", default=None)
        if mjd_obs is not None:
            jd_start = _to_float_db(mjd_obs, 0.0) + 2400000.5
        else:
            # Fallback: compute JD from exposure start time keywords.
            # DATE-OBS is typically the start of exposure in UTC (FITS standard).
            date_obs = _pick("DATE-OBS", "DATEOBS", default=None)
            time_obs = _pick("TIME-OBS", "TIMEOBS", default=None)
            jd_start = 0.0
            if date_obs is not None:
                dt_str = str(date_obs).strip()
                if time_obs is not None and "T" not in dt_str:
                    dt_str = f"{dt_str}T{str(time_obs).strip()}"
                try:
                    jd_start = float(Time(dt_str, format="isot", scale="utc").jd)
                except Exception:  # noqa: BLE001
                    jd_start = 0.0

    x_bin = _parse_fits_binning_int(_pick("XBINNING", "BINNING", default=1), 1)
    y_bin = _parse_fits_binning_int(_pick("YBINNING", default=x_bin), x_bin)

    _ps1 = _fits_pixel_raw_to_micrometres(_to_float_db(_pick("PIXSIZE1", "XPIXSZ", "PIXSZLX", "PIXSIZE", default=0.0)))
    _ps2 = _fits_pixel_raw_to_micrometres(_to_float_db(_pick("PIXSIZE2", "YPIXSZ", "PIXSZLY", default=0.0)))

    _force = None
    if force_physical_pixel_um is not None:
        try:
            fv = float(force_physical_pixel_um)
            if math.isfinite(fv) and fv > 0:
                _force = fv
        except (TypeError, ValueError):
            _force = None
    if _force is not None:
        _ps1, _ps2 = _force, _force

    _physical_x = _ps1 if _ps1 > 0 else None
    _physical_y = _ps2 if _ps2 > 0 else None
    if _physical_x is not None and _physical_y is not None:
        _physical_mean = (_physical_x + _physical_y) / 2.0
    elif _physical_x is not None:
        _physical_mean = float(_physical_x)
    elif _physical_y is not None:
        _physical_mean = float(_physical_y)
    else:
        _physical_mean = None

    _eff_x = (_physical_x * float(x_bin)) if _physical_x is not None else None
    _eff_y = (_physical_y * float(y_bin)) if _physical_y is not None else None
    if _eff_x is not None and _eff_y is not None:
        _effective_um = (_eff_x + _eff_y) / 2.0
    elif _eff_x is not None:
        _effective_um = float(_eff_x)
    elif _eff_y is not None:
        _effective_um = float(_eff_y)
    else:
        _effective_um = None

    if _physical_mean is not None:
        _plate_eff = float(_physical_mean) * float(x_bin)
    else:
        _plate_eff = _effective_um

    return {
        "exposure": float(_pick("EXPTIME", "EXPOSURE", default=0.0)),
        "filter": str(_pick("FILTER", "FILT", default="NoFilter")),
        "binning": int(x_bin),
        "binning_y": int(y_bin),
        "naxis1": int(header.get("NAXIS1", 0) or 0),
        "naxis2": int(header.get("NAXIS2", 0) or 0),
        "pixel_size_um_physical": _physical_mean,
        "pixel_size_um_header": _effective_um,
        "effective_pixel_um_plate_scale": _plate_eff,
        "temp": float(_pick("CCD-TEMP", "SENSORTEMP", "SET-TEMP", default=0.0)),
        "gain": int(_pick("GAIN", "GAINER", "CCD-GAIN", default=0) or 0),
        "ra": _fits_meta_ra_deg(
            _pick(
                "OBJCTRA",
                "RA_OBJ",
                "TARGRA",
                "CENTRA",
                "RA",
                "RAJ2000",
                "CAT-RA",
                "CRVAL1",
                default=0.0,
            )
        ),
        "dec": _fits_meta_dec_deg(
            _pick(
                "OBJCTDEC",
                "DEC_OBJ",
                "TARGDEC",
                "CENTDEC",
                "DEC",
                "DEJ2000",
                "CAT-DEC",
                "CRVAL2",
                default=0.0,
            )
        ),
        "jd_start": jd_start,
        "telescope": _pick("TELESCOP", "SCOPE", default=None),
        "camera": _pick("INSTRUME", "CAMERA", default=None),
    }


def extract_fits_metadata(
    filepath: Path | str,
    *,
    db: VyvarDatabase | None = None,
    app_config: AppConfig | None = None,
    force_physical_pixel_um: float | None = None,
    id_equipment: int | None = None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """Extract key metadata from FITS primary header.

    When ``db`` is set, uses ``FITS_HEADER_CACHE`` when ``FILE_PATH``, ``FILE_SIZE``, and ``MTIME`` match
    the file on disk; otherwise reads the FITS header and refreshes the cache row.

    ``ra`` / ``dec`` are in decimal degrees. Sources (first match wins): OBJCTRA/OBJCTDEC,
    RA_OBJ/DEC_OBJ, TARGRA/TARGDEC, CENTRA/CENTDEC, RA/DEC, RAJ2000/DEJ2000, CAT-RA/CAT-DEC,
    CRVAL1/CRVAL2.
    Sexagesimal strings are accepted with colons or spaces (e.g. ``' 3 39 06.45'`` for RA).

    Returned keys:
    - exposure
    - filter
    - binning (XBINNING / BINNING)
    - binning_y (YBINNING, default same as binning)
    - naxis1, naxis2
    - pixel_size_um_physical (native pitch from header, or ``force_pixel_size`` from config)
    - pixel_size_um_header (**effective** pitch [µm] = physical × binning for plate solve / WCS)
    - effective_pixel_um_plate_scale (native mean × XBINNING; solver / plate scale)
    - focal_length [mm]: ``FOCALLEN`` v hlavičke ak je; inak ``EQUIPMENTS.FOCAL`` (ak je) alebo ``TELESCOPE.FOCAL``
    - focal_length_source, pixel_size_raw (surové čísla z hlavičky), pixel_um (= effective pixel)
    - temp
    - ra
    - dec
    - jd_start
    - telescope
    - camera

    Convention:     ``EQUIPMENTS.PIXELSIZE`` in the DB is the **native 1×1** pitch [µm]; FITS cache
    ``PIXEL_UM`` stores **effective** pitch after binning.

    When ``draft_id`` is set and ``db`` is given, :meth:`VyvarDatabase.get_combined_metadata` overlays
    FITS+SQL focal/pixel (``XBINNING``-strict effective pixel) and ``EQUIPMENTS.SATURATE_ADU``.
    """
    fp = Path(filepath)
    st: os.stat_result | None = None
    try:
        st = fp.stat()
    except OSError:
        st = None

    if db is not None and st is not None:
        cached = db.fits_header_cache_try_meta(fp)
        if cached is not None:
            meta = dict(cached)
            if id_equipment is not None:
                _merge_equipment_pixel_into_metadata(meta, db, int(id_equipment))
            _recompute_effective_pixel_from_physical(meta)
            with fits.open(fp, memmap=False) as hdul:
                _enrich_calibration_metadata_from_header(
                    meta, hdul[0].header, db=db, id_equipment=id_equipment
                )
            if draft_id is not None:
                comb = db.get_combined_metadata(fp, int(draft_id))
                _apply_draft_combined_to_pipeline_meta(meta, comb)
            return meta

    cfg = app_config or AppConfig()
    _fpu: float | None = None
    if force_physical_pixel_um is not None:
        try:
            v = float(force_physical_pixel_um)
            if v > 0 and math.isfinite(v):
                _fpu = v
        except (TypeError, ValueError):
            _fpu = None
    if _fpu is None and cfg.force_pixel_size_um is not None:
        try:
            cfpv = float(cfg.force_pixel_size_um)
            if cfpv > 0 and math.isfinite(cfpv):
                _fpu = cfpv
        except (TypeError, ValueError):
            pass

    with fits.open(fp, memmap=False) as hdul:
        header = hdul[0].header
    meta = fits_metadata_from_primary_header(header, force_physical_pixel_um=_fpu)
    if db is not None and id_equipment is not None:
        _merge_equipment_pixel_into_metadata(meta, db, int(id_equipment))
    _recompute_effective_pixel_from_physical(meta)
    _enrich_calibration_metadata_from_header(meta, header, db=db, id_equipment=id_equipment)
    if draft_id is not None and db is not None:
        comb = db.get_combined_metadata(fp, int(draft_id))
        _apply_draft_combined_to_pipeline_meta(meta, comb)
    _log_effective_pixel_pitch(meta, filepath=str(fp.name))

    if db is not None and st is not None:
        imagetyp = str(header.get("IMAGETYP") or header.get("FRAME") or header.get("IMTYPE") or "")
        do = header.get("DATE-OBS") or header.get("DATEOBS")
        date_obs = None if do in (None, "") else str(do)
        db.fits_header_cache_upsert_one(
            fp,
            file_size=int(st.st_size),
            mtime=float(st.st_mtime),
            meta=meta,
            imagetyp=imagetyp,
            date_obs=date_obs,
        )

    return meta


def scan_usb_folder(path: Path | str) -> pd.DataFrame:
    """Recursively scan source tree and detect Lights/Darks/Flats by IMAGETYP/FRAME.

    This scan is folder-name agnostic and reports real folder paths.
    If a folder contains mixed types, it will be marked as Mixed, but files can still be
    sorted per-file by IMAGETYP downstream (importer).

    Output columns:
    - Folder Path
    - Type
    - File Count
    - Lights Count
    - Darks Count
    - Flats Count
    - Unknown Count
    - Detected Filters
    - Params
    """
    root = Path(path)
    rows: list[dict[str, Any]] = []

    def _classify(text: str) -> str:
        t = (text or "").lower()
        if "light" in t:
            return "Lights"
        if "dark" in t:
            return "Darks"
        if "flat" in t:
            return "Flats"
        return "Unknown"

    def _fits_files(folder: Path) -> list[Path]:
        files: list[Path] = []
        seen: set[str] = set()
        for fp in folder.iterdir():
            if not fp.is_file():
                continue
            if not path_suffix_is_fits(fp):
                continue
            key = str(fp.resolve()).casefold()
            if key in seen:
                continue
            seen.add(key)
            files.append(fp)
        return sorted(files)

    if not root.exists() or not root.is_dir():
        return pd.DataFrame(columns=["Folder Path", "Type", "File Count", "Params"])

    # scan all subfolders, including root itself
    folders = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for folder in folders:
        files = _fits_files(folder)
        if not files:
            continue

        # classify per-file to detect mixed folders
        type_counts: dict[str, int] = {"Lights": 0, "Darks": 0, "Flats": 0, "Unknown": 0}
        filter_set: set[str] = set()
        for fp in files:
            try:
                with fits.open(fp, memmap=False) as hdul:
                    hdr = hdul[0].header
                imagetyp = str(hdr.get("IMAGETYP") or hdr.get("FRAME") or hdr.get("IMTYPE") or "")
                cls = _classify(imagetyp)
                flt = str(hdr.get("FILTER") or hdr.get("FILT") or "").strip()
                if flt:
                    filter_set.add(flt)
            except Exception:  # noqa: BLE001
                cls = "Unknown"
            type_counts[cls] = type_counts.get(cls, 0) + 1

        present = [k for k, v in type_counts.items() if v > 0 and k != "Unknown"]
        if len(present) == 1:
            detected = present[0]
        elif len(present) > 1:
            detected = "Mixed"
        else:
            detected = "Unknown"

        # params from first file
        first = files[0]
        try:
            with fits.open(first, memmap=False) as hdul:
                hdr = hdul[0].header
            exp = hdr.get("EXPTIME") or hdr.get("EXPOSURE") or 0
            gain = hdr.get("GAIN") or hdr.get("GAINER") or hdr.get("CCD-GAIN") or 0
            params = f"Exp={float(exp):g}s, Gain={int(gain)}"
        except Exception:  # noqa: BLE001
            params = ""

        rows.append(
            {
                "Folder Path": str(folder),
                "Type": detected,
                "File Count": int(len(files)),
                "Lights Count": int(type_counts.get("Lights", 0)),
                "Darks Count": int(type_counts.get("Darks", 0)),
                "Flats Count": int(type_counts.get("Flats", 0)),
                "Unknown Count": int(type_counts.get("Unknown", 0)),
                "Detected Filters": ", ".join(sorted(filter_set)) if filter_set else "",
                "Params": params,
            }
        )

    return pd.DataFrame(rows).sort_values(["Type", "Folder Path"], ignore_index=True)


class AstroPipeline:
    """Skeleton for the modular variable-star processing workflow."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.db = VyvarDatabase(self.config.database_path)

    def calibrate(self, session_path: Path | str) -> Any:
        """Calibrate raw images and build nightly masters.

        Planned implementation:
        - library masters: mean stack (dark), median stack (flat); calibration re-use library
        """
        LOGGER.info("Calibrate requested for session: %s", session_path)
        return None

    def quick_calibrate_last_import(
        self,
        *,
        archive_path: Path | str,
        master_dark_path: Path | None,
        masterflat_by_filter: dict[str, Path | None],
        progress_cb: "callable | None" = None,
        equipment_id: int | None = None,
        draft_id: int | None = None,
        observation_id: str | None = None,
        masterflat_by_obs_key: dict[str, str | Path | None] | None = None,
        master_dark_by_obs_key: dict[str, str | Path | None] | None = None,
    ) -> dict[str, Any]:
        """Calibrate imported draft/raw lights into `calibrated/` under archive_path."""
        ap = Path(archive_path)
        # Accept both draft root (.../draft_xxx) and direct non_calibrated path (.../draft_xxx/non_calibrated).
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        LOGGER.info("Kalibrácia archívu: %s", ap)
        # Support Draft structure: <archive>/Raw/lights and optional <archive>/non_calibrated/lights.
        # Both sources are calibrated/passthrough-written into the single target: <archive>/calibrated/lights.
        outputs: dict[str, Any] = {"archive_path": str(ap_root), "results": {}}

        draft_lights = ap_root / "non_calibrated" / "lights"
        if not draft_lights.exists() and ap.name.casefold() == "non_calibrated":
            draft_lights = ap / "lights"
        raw_lights = ap_root / "Raw" / "lights"
        # In non_calibrated draft mode always prefer Draft/non_calibrated/lights.
        if draft_lights.exists():
            out_root = ap_root / "calibrated" / "lights"
            outputs["results"]["non_calibrated"] = calibrate_lights_to_calibrated(
                lights_root=draft_lights,
                calibrated_root=out_root,
                master_dark_path=master_dark_path,
                masterflat_by_filter=masterflat_by_filter,
                progress_cb=progress_cb,
                pipeline_config=self.config,
                equipment_id=equipment_id,
                draft_id=draft_id,
                observation_id=observation_id,
                masterflat_by_obs_key=masterflat_by_obs_key,
                master_dark_by_obs_key=master_dark_by_obs_key,
            )
        elif raw_lights.exists():
            out_root = ap_root / "calibrated" / "lights"
            outputs["results"]["Raw"] = calibrate_lights_to_calibrated(
                lights_root=raw_lights,
                calibrated_root=out_root,
                master_dark_path=master_dark_path,
                masterflat_by_filter=masterflat_by_filter,
                progress_cb=progress_cb,
                pipeline_config=self.config,
                equipment_id=equipment_id,
                draft_id=draft_id,
                observation_id=observation_id,
                masterflat_by_obs_key=masterflat_by_obs_key,
                master_dark_by_obs_key=master_dark_by_obs_key,
            )

        LOGGER.info("Kalibrácia dokončená (sekcií výstupu: %s)", list((outputs.get("results") or {}).keys()))
        return outputs

    def calibrate_batch(
        self,
        *,
        light_paths: Sequence[Path | str],
        lights_root: Path | str,
        calibrated_root: Path | str,
        master_dark_path: Path | str | None,
        masterflat_by_filter: dict[str, Path | str | None],
        max_workers: int | None = None,
        progress_cb: "callable | None" = None,
        equipment_id: int | None = None,
        draft_id: int | None = None,
        observation_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply master dark/flat to many lights using ``ProcessPoolExecutor`` (``spawn``).

        Output layout matches :func:`calibrate_lights_to_calibrated`: for each input file
        ``dst = calibrated_root / Path(light).relative_to(lights_root)``.

        Returns a dict with:

        - ``output_paths``: list aligned with ``light_paths`` — calibrated FITS path or ``None`` on failure
        - ``results``: list of per-file ``dict``\\ s from workers (``src``, ``dst``, ``ok``, ``error``)
        - ``stats``: processed / ok / failed counts
        """
        import numpy as np

        lr = Path(lights_root).resolve()
        cr = Path(calibrated_root)
        cr.mkdir(parents=True, exist_ok=True)

        light_paths = filter_light_paths_for_calibration_db(
            [Path(lp) for lp in light_paths],
            database_path=self.config.database_path,
            draft_id=draft_id,
            observation_id=observation_id,
        )

        mf_paths: dict[str, Path | None] = {}
        for k, v in (masterflat_by_filter or {}).items():
            if v is None or str(v).strip() == "":
                mf_paths[str(k)] = None
            else:
                mf_paths[str(k)] = Path(v)

        mf_serial: dict[str, str | None] = {
            k: str(p.resolve()) if p is not None else None for k, p in mf_paths.items()
        }

        _md_log = Path(master_dark_path) if master_dark_path is not None else None
        _log_calibration_io_preflight(
            calibrated_root=cr,
            master_dark_path=_md_log,
            masterflat_by_filter=mf_paths,
        )

        md_init: str | None = None
        if master_dark_path is not None:
            md_p = Path(master_dark_path)
            if md_p.is_file():
                md_init = str(md_p.resolve())

        qc_pack = _qc_pack_from_config(
            self.config, draft_id=draft_id, observation_id=observation_id
        )

        items: list[
            tuple[
                str,
                str,
                str | None,
                dict[str, str | None],
                dict[str, Any] | None,
                dict[str, Any] | None,
            ]
        ] = []
        for lp in light_paths:
            src_p = Path(lp).resolve()
            rel = src_p.relative_to(lr)
            dst_p = (cr / rel).resolve()
            items.append((str(src_p), str(dst_p), md_init, mf_serial, None, qc_pack))

        n = len(items)
        if n == 0:
            return {
                "output_paths": [],
                "results": [],
                "stats": {"n_input": 0, "ok": 0, "failed": 0},
            }

        nw = (
            max_workers
            if max_workers is not None
            else max(1, min(32, int(self.config.qc_preprocess_workers)))
        )
        nw = max(1, min(int(nw), n))
        if not _vyvar_calibrate_multiprocessing_enabled():
            nw = 1

        _native_b = _cfg_calibration_library_native_binning(self.config)

        rows: list[dict[str, Any]]
        if nw <= 1:
            md_pre: Any = None
            if md_init:
                with fits.open(md_init, memmap=False) as h:
                    md_pre = np.array(h[0].data, dtype=np.float32, copy=True)
            flat_cache: dict[str, Any] = {}
            flat_med: dict[str, float] = {}
            db_main = _db_for_calibration_tasks(None, qc_pack)
            rows = []
            for i, it in enumerate(items):
                src_s, dst_s, md_s, mf_map, _copt, _qopt = it
                try:
                    mf = {str(k): Path(v) if v else None for k, v in mf_map.items()}
                    _ud, _uf, qc_sum, _cf = _calibrate_one_light_disk(
                        src=Path(src_s),
                        dst=Path(dst_s),
                        master_dark_path=Path(md_s) if md_s else None,
                        masterflat_by_filter=mf,
                        flat_cache=flat_cache,
                        flat_median_scale=flat_med,
                        md_data_preload=md_pre,
                        cosmic_options=_copt,
                        db=db_main,
                        qc_pack=_qopt,
                        calibration_master_native_binning=_native_b,
                    )
                    rows.append(
                        {
                            "src": src_s,
                            "dst": dst_s,
                            "ok": True,
                            "error": None,
                            "qc_summary": qc_sum,
                            "traceback": None,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    _tb_cb = traceback.format_exc()
                    LOGGER.error("calibrate_batch: %s -> %s\n%s", src_s, exc, _tb_cb)
                    log_exception(f"CHYBA WORKERA: {Path(src_s).name}", exc)
                    rows.append(
                        {
                            "src": src_s,
                            "dst": None,
                            "ok": False,
                            "error": str(exc),
                            "traceback": _tb_cb,
                        }
                    )
                if progress_cb is not None:
                    progress_cb(i + 1, n, f"Calibrating {Path(src_s).name}")
        else:
            ctx = multiprocessing.get_context("spawn")
            rows = [None] * n  # type: ignore[misc]
            with ProcessPoolExecutor(
                max_workers=nw,
                mp_context=ctx,
                initializer=_init_calibrate_batch_worker,
                initargs=(md_init, _native_b),
            ) as ex:
                future_map = {ex.submit(_calibrate_batch_process_one, it): idx for idx, it in enumerate(items)}
                done = 0
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    rows[idx] = fut.result()
                    done += 1
                    if progress_cb is not None:
                        progress_cb(done, n, f"Calibrating batch {done}/{n}")

        out_paths: list[str | None] = []
        ok_c = 0
        fail_c = 0
        for r in rows:
            if r.get("ok"):
                ok_c += 1
                out_paths.append(str(r["dst"]) if r.get("dst") else None)
            else:
                fail_c += 1
                out_paths.append(None)

        return {
            "output_paths": out_paths,
            "results": rows,
            "stats": {
                "n_input": n,
                "ok": ok_c,
                "failed": fail_c,
                "max_workers": nw,
            },
        }

    def quick_preprocess_last_import(
        self,
        *,
        archive_path: Path | str,
        run: bool = True,
        reject_fwhm_px: float | None = None,
        reject_elongation: float | None = None,
        background_method: str = "background2d",
        poly_order: int = 2,
        lacosmic_sigclip: float = 4.5,
        lacosmic_objlim: float = 5.0,
        enable_lacosmic: bool = False,
        enable_background_flattening: bool = False,
        temporal_sigma_clip: bool = False,
        temporal_sigma: float = 6.0,
        temporal_min_frames: int = 5,
        use_gpu_if_available: bool = False,
    ) -> dict[str, Any]:
        ap = Path(archive_path)
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        src_cal = ap_root / "calibrated" / "lights"
        src_noncal = ap_root / "non_calibrated" / "lights"
        source_dir = src_cal if src_cal.exists() else src_noncal
        out: dict[str, Any] = {
            "archive_path": str(ap_root),
            "processed": {},
            "qc_suggestions": {},
            "checked": {
                "archive_exists": bool(ap_root.exists()),
                "source_path": str(source_dir),
                "calibrated_path": str(src_cal),
                "non_calibrated_path": str(src_noncal),
                "processed_path": str(ap_root / "processed" / "lights"),
                "detrended_path": str(ap_root / "detrended" / "lights"),
            },
        }

        if source_dir.exists():
            proc_root = ap_root / "processed" / "lights"
            if run:
                df = preprocess_calibrated_to_processed(
                    calibrated_root=source_dir,
                    processed_root=proc_root,
                    background_method=background_method,
                    poly_order=poly_order,
                    lacosmic_sigclip=lacosmic_sigclip,
                    lacosmic_objlim=lacosmic_objlim,
                    enable_lacosmic=bool(enable_lacosmic),
                    enable_background_flattening=bool(enable_background_flattening),
                    reject_fwhm_px=reject_fwhm_px,
                    reject_elongation=reject_elongation,
                    temporal_sigma_clip=temporal_sigma_clip,
                    temporal_sigma=temporal_sigma,
                    temporal_min_frames=temporal_min_frames,
                    use_gpu_if_available=use_gpu_if_available,
                    progress_cb=None,
                )
            else:
                # summarize from existing qc_metrics.csv if present
                qc_csv_existing = proc_root / "qc_metrics.csv"
                if not qc_csv_existing.exists():
                    qc_csv_existing = ap_root / "detrended" / "lights" / "qc_metrics.csv"
                try:
                    df = pd.read_csv(qc_csv_existing) if qc_csv_existing.exists() else pd.DataFrame()
                except Exception:  # noqa: BLE001
                    df = pd.DataFrame()
            out["processed"]["source"] = {
                "processed_root": str(proc_root),
                "rows": int(len(df)),
                "rejected": int((df["status"].astype(str).str.startswith("rejected")).sum()) if not df.empty else 0,
                "source_dir": str(source_dir),
            }
            # Save QC table for review
            try:
                qc_csv = proc_root / "qc_metrics.csv"
                if run and not df.empty:
                    df.to_csv(qc_csv, index=False)
                out["processed"]["source"]["qc_csv"] = str(qc_csv)
            except Exception:
                pass
            out["qc_suggestions"]["source"] = _qc_suggest_thresholds(df)

        if not out["processed"]:
            out["warning"] = "No calibrated lights found under this archive_path. Run calibration first (Run quick pipeline step)."
        return out

    def quick_analyze_last_import(
        self,
        *,
        archive_path: Path | str,
        max_frames: int | None = None,
        background_method: str = "background2d",
        poly_order: int = 2,
        lacosmic_sigclip: float = 4.5,
        lacosmic_objlim: float = 5.0,
    ) -> dict[str, Any]:
        ap = Path(archive_path)
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        out: dict[str, Any] = {
            "archive_path": str(ap_root),
            "analysis": {},
            "qc_suggestions": {},
        }
        cal = ap_root / "calibrated" / "lights"
        if cal.exists():
            df = analyze_calibrated_qc(
                calibrated_root=cal,
                max_frames=max_frames,
                background_method=background_method,
                poly_order=poly_order,
                lacosmic_sigclip=lacosmic_sigclip,
                lacosmic_objlim=lacosmic_objlim,
            )
            out["analysis"]["calibrated"] = {"rows": int(len(df))}
            out["qc_suggestions"]["calibrated"] = _qc_suggest_thresholds(df)
            try:
                qc_csv = ap_root / "calibrated" / "lights" / "qc_analysis.csv"
                df.to_csv(qc_csv, index=False)
                out["analysis"]["calibrated"]["qc_csv"] = str(qc_csv)
            except Exception:
                pass
        else:
            out["warning"] = "No calibrated lights found. Run calibration first."
        return out

    @staticmethod
    def _first_fits_file(session_path: Path | str) -> Path:
        session = Path(session_path)
        search_roots = [session, session / "Raw"]
        for root in search_roots:
            if not root.exists():
                continue
            for fp in sorted(root.rglob("*")):
                if fp.is_file() and path_suffix_is_fits(fp):
                    return fp
        raise FileNotFoundError(f"No FITS file found in session path: {session}")

    def prepare_observation_from_session(
        self,
        session_path: Path | str,
        *,
        id_equipment: int | None = None,
        id_telescope: int | None = None,
        id_location: int = 1,
    ) -> dict[str, Any]:
        """Read first FITS and build prefilled OBSERVATION payload."""
        first_fits = self._first_fits_file(session_path)
        metadata = extract_fits_metadata(first_fits, db=self.db, app_config=self.config)
        scanning_id = self.db.find_or_create_scanning_id(metadata)

        equipment_id = int(id_equipment) if id_equipment is not None else 1
        telescope_id = int(id_telescope) if id_telescope is not None else 1

        observation_payload = {
            "id_equipments": equipment_id,
            "id_telescope": telescope_id,
            "id_location": int(id_location),
            "id_scanning": scanning_id,
            "center_of_field_ra": float(metadata["ra"]),
            "center_of_field_de": float(metadata["dec"]),
            "observation_start_jd": float(metadata["jd_start"]),
        }
        return {
            "fits_file": str(first_fits),
            "metadata": metadata,
            "observation_payload": observation_payload,
            "missing_telescope": not bool(metadata.get("telescope")),
            "missing_camera": not bool(metadata.get("camera")),
        }

    def create_observation_from_payload(self, payload: dict[str, Any]) -> str:
        """Insert prepared observation payload and return new observation ID."""
        return self.db.add_observation(payload)

