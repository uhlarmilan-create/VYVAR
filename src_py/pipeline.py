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
from typing import Any, Callable, Mapping, Sequence

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import pandas as pd

from config import AppConfig, load_config_json
from catalog_match_trust import export_catalog_match_mode_from_internal
import vyvar_alignment_frame  # A-durable: fresh-attr MP func lookup at dispatch (reload-safe)
from vyvar_alignment_frame import (
    _alignment_compute_one_frame,
    _alignment_detect_xy,
    _as_fits_float32_image,
    _astrometry_align_mp_init,
    _astrometry_align_mp_task,
)
from database import (
    DraftTechnicalMetadataError,
    VyvarDatabase,
    _db_header_pixel_native_um_mean,
    _db_to_float as _to_float_db,
    query_local_gaia,
    query_local_gaia_by_source_ids,
    query_local_exoplanet,
    query_local_vsx,
)
from time_utils import _header_float as _header_float_tu
from photometry import (
    common_field_intersection_bbox_px,
    compute_fwhm_gaussian_for_aperture_catalog,
    enhance_catalog_dataframe_aperture_bpm,
    recommended_aperture_by_color,
    stress_test_relative_rms_from_sidecars,
    vsx_is_known_variable_top3_per_bin,
)
from fits_suffixes import FITS_SUFFIXES_LOWER, path_suffix_is_fits
from gaia_catalog_id import (
    catalog_id_series_for_masterstars_export,
    normalize_gaia_source_id,
    read_vyvar_csv,
)
from infolog import log_event, log_exception, log_milestone
from optics_selection import resolve_optics_ids_for_platesolve
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
from photometry_core import (
    _fwhm_moment_at,
    merge_photometry_pipeline_meta,
    stamp_masterstar_snr_columns,
)
from proc_frame_store import proc_csv_path_for_aligned_fits

from dao_reconcile import compute_gaia_dao_reconcile, reconcile_to_pipeline_meta, resolve_effective_match_depth
from masterstar_context import header_core_fwhm_px
from plain_stats import plain_mean_med_std, sky_mad_sigma_adu

from utils import (
    ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC,
    DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    MIN_GAIA_CONE_RADIUS_DEG,
    astrometry_net_scale_bounds_arcsec_per_pix,
    catalog_cone_radius_deg_from_optics,
    catalog_cone_radius_from_fov_diameter_deg,
    dao_detection_fwhm_pixels,
    effective_astrometry_net_tweak_order,
    effective_binned_pixel_pitch_um,
    fits_binning_xy_from_header,
    fits_header_has_celestial_wcs,
    iter_fits_paths_recursive as _iter_fits_recursive,
    masterstar_wcs_quality,
    maybe_rescale_linear_wcs_cd_to_target_arcsec_per_pixel,
    normalize_telescope_focal_mm_for_plate_scale,
    per_frame_catalog_match_sep_arcsec_for_scale,
    plate_scale_arcsec_per_pixel,
    plate_solve_fov_deg_diagonal_from_scale,
    strip_celestial_wcs_keys,
    strip_vendor_platesolve_metadata,
    wcs_distortion_log_suffix,
    wcs_rotation_angle_deg,
)
from vyvar_platesolver import (
    _fits_header_parse_dec_deg,
    _fits_header_parse_ra_deg,
    pointing_hint_from_header as _pointing_hint_from_header,
)
import itertools

from fits_meta import (  # noqa: F401
    _safe_filter_token,
    observation_group_key_from_metadata,
    _summarize_lights_binning_from_headers,
    log_lights_binning_from_headers_preflight,
    generate_observation_hash,
    _fits_pixel_raw_to_micrometres,
    _focal_mm_plausible,
    _merge_equipment_pixel_into_metadata,
    _recompute_effective_pixel_from_physical,
    _header_pick_first,
    _enrich_calibration_metadata_from_header,
    _apply_draft_combined_to_pipeline_meta,
    _fits_meta_ra_deg,
    _fits_meta_dec_deg,
    _parse_fits_binning_int,
    _log_effective_pixel_pitch,
    fits_metadata_from_primary_header,
    _valid_bayerpat_from_header,
    extract_fits_metadata,
    scan_usb_folder,
)

def _frame_gain_readnoise_for_error_map(
    hdr: fits.Header,
    *,
    db: VyvarDatabase | None,
    equipment_id: int | None,
) -> tuple[float, float]:
    """Gain / read-noise for the per-frame error map.

    Unified resolution (param_resolver): gain header-first (e-/ADU or index-mapped) ->
    DB -> config; read noise DB-first. Matches Phase 2A photometric-error path.
    """
    from param_resolver import resolve_gain, resolve_read_noise  # noqa: PLC0415

    g_res = resolve_gain(hdr, db=db, equipment_id=equipment_id)
    rn_res = resolve_read_noise(hdr, db=db, equipment_id=equipment_id)
    gain = float(g_res.value) if g_res.ok else 1.0
    rn = float(rn_res.value) if rn_res.ok else 10.0
    return gain, rn


def _per_frame_noise_error_map(data: Any, hdr: fits.Header, *, db: VyvarDatabase | None, equipment_id: int | None):
    """Per-pixel noise sigma for CCD-like error: sqrt(max(data,0)/gain + readnoise^2)."""
    import numpy as np

    gain, rn = _frame_gain_readnoise_for_error_map(hdr, db=db, equipment_id=equipment_id)
    d = np.asarray(data, dtype=np.float64)
    sig = np.where(np.isfinite(d), np.maximum(d, 0.0), 0.0)
    var = sig / float(gain) + float(rn) ** 2
    return np.sqrt(np.maximum(var, 1e-24))


# Public aliases (historically some callers used ``pipeline.parse_user_*`` / ``pointing_hint_from_header``).
pointing_hint_from_header = _pointing_hint_from_header


LOGGER = logging.getLogger(__name__)

_SKY_ADU_FALLBACK = 1581.6                             # was cfg.sky_adu_fallback (1581.6)
_MASTERSTAR_SOLVER_USE_DRAFT_MEDIAN_IF_HINT_SEP_DEG = 1.0  # was cfg.masterstar_solver_use_draft_median_if_hint_sep_deg (1.0)
_MASTERSTAR_OPTIMIZER_MIRROR_EXTRA_LOG = True         # was cfg.masterstar_optimizer_mirror_extra_log (True)
_MASTERSTAR_PLATESOLVE_PREWRITE_RMS_MAX_PX = 30.0     # was cfg.masterstar_platesolve_prewrite_rms_max_px (30.0)
_MASTERSTAR_PLATESOLVE_PREWRITE_RELAXED_RMS_MAX_PX = 35.0  # was cfg.masterstar_platesolve_prewrite_relaxed_rms_max_px (35.0)
_MASTERSTAR_PLATESOLVE_NN_REFINE_MAX_RMS_PX = None    # was cfg.masterstar_platesolve_nn_refine_max_rms_px (None)
_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO = 1.15          # was cfg.masterstar_sip_force_rms_guard_ratio (1.15)
_PLATESOLVE_ANISOTROPY_THRESHOLD = 1.3                # was cfg.platesolve_anisotropy_threshold (1.3)


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


def _estimate_fov_deg_from_fits_path(fp: Path) -> float | None:
    p = Path(fp)
    if not p.is_file():
        return None
    try:
        with fits.open(p, memmap=False) as hdul:
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


def get_auto_fov(
    *,
    archive_path: Path | None = None,
    masterstar_path: Path | None = None,
    database_path: Path | str | None = None,
    equipment_id: int | None = None,
    draft_id: int | None = None,
) -> float | None:
    """Auto field diameter [deg] (diagonal) for plate solving.

    Priority:
    - Header optics (focal + pixel) or DB plate-scale x NAXIS diagonal
    - Else WCS corners (after a successful solve)
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

    _dbp = str(database_path or "").strip()
    if not _dbp:
        try:
            _dbp = str(AppConfig().database_path)
        except Exception:  # noqa: BLE001
            _dbp = ""
    _hint = resolve_plate_solve_fov_deg_hint(
        hdr, h, w, database_path=_dbp or None, equipment_id=equipment_id, draft_id=draft_id
    )
    if _hint is not None and math.isfinite(_hint) and _hint > 0:
        return float(_hint)

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
        f"solve-field (lokalny): {exe} - --cpulimit {ASTROMETRY_SOLVE_FIELD_CPULIMIT_SEC}, "
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
        logging.error('[EXC-0319] Generated .wcs read failure returns solved=False without applying any WCS to the image.: %s', exc)
        return {"solved": False, "reason": f"solve-field: {exc!s}"}

    if not wcs_path.is_file():
        tail = (proc.stderr or proc.stdout or "")[-900:]
        if proc.returncode != 0:
            return {"solved": False, "reason": f"solve-field exit {proc.returncode}: {tail!s}"}
        return {"solved": False, "reason": f"solve-field: missing {wcs_path.name} - {tail!s}"}

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
        f"solve-field OK: WCS so SIP (tweak-order {effective_astrometry_net_tweak_order()}) -> {mp.name}"
    )
    return {"solved": True, "method": "solve-field (local CLI)"}



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


_EXO_HOST_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "exo_host_obj_id",
    "exo_host_name",
    "exo_cat_source",
    "exo_disposition",
    "exo_match_sep_arcsec",
)


# SAT-LIMIT-01 / GAIN-DOMAIN-01: 16-bit FITS container clip (pile-up at 65535, not 65532).
SAT_LIMIT_CONTAINER_CLIP_ADU = 65535.0
# Peak-test fraction when the linearity knee is unmeasured (D1-2 / SAT-LIMIT-01 B3).
SAT_LIMIT_NO_KNEE_FRAC = 0.80
# Provenance string for the INV-SAT-LIMIT peak-test (catalog zone + per-frame clean).
SAT_LIMIT_PEAK_TEST_SOURCE = (
    f"INV-SAT-LIMIT peak-test {SAT_LIMIT_NO_KNEE_FRAC:.2f}x "
    f"container_clip_{SAT_LIMIT_CONTAINER_CLIP_ADU:.0f}"
)


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
    likely = bool(sat_by_peak)
    return {
        "peak_dao": float(peak_dao_val) if peak_dao_val is not None and np.isfinite(peak_dao_val) else None,
        "peak_max_adu": float(pmax) if np.isfinite(pmax) else None,
        "saturate_limit_adu": float(lim) if lim is not None else None,
        "saturated_from_peak": sat_by_peak,
        "saturated_plateau": sat_by_plateau,
        "likely_saturated": likely,
        "photometry_ok": not likely,
    }


from masterstar_gaia_accounting import _dao_xy_binned_to_full  # noqa: E402,F401


from masterstar_gaia_accounting import _dao_full_to_binned_xy  # noqa: E402,F401


from masterstar_gaia_accounting import _dao_pass2_annulus_stats  # noqa: E402,F401


def detect_stars_and_match_catalog(
    data: "np.ndarray",
    hdr: fits.Header,
    *,
    max_catalog_rows: int = 12000,
    cat_df: pd.DataFrame | None = None,
    vsx_df: pd.DataFrame | None = None,
    exo_df: pd.DataFrame | None = None,
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
    fov_database_path: Path | str | None = None,
    fov_equipment_id: int | None = None,
    fov_draft_id: int | None = None,
    prematch_peak_sigma_floor: float = 10.0,
    prematch_exempt_pass2: bool = True,
    frame_name: str = "",
    dao_fwhm_bypass_header: bool = False,
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

    ``match_sep_arcsec`` requested value is recorded; D1 sets the one-pass
    effective radius to max(12 arcsec, 3 x FWHM_dao_px x plate_scale).
    ``solve_rms_px`` is stamped as a diagnostic and does not enter the radius.
    There is no match-rate widening. A final tightening to ~4.5 arcsec is
    applied only when most loose matches survive it. Low match rate remains a
    WARN.

    ``max_catalog_rows`` caps DAO detections written per frame. Rows are chosen with **spatial
    stratification** (brightest per coarse grid cell, then global flux top-up) so vignetting does not
    mimic a ``catalog disc`` the way a plain brightest-N sort does.

    If ``field_catalog_export_path`` is set, the **full** cone table (``cat_df``) is written there for
    QA overlays - many more rows than DAO detections in ``masterstars.csv``.

    ``dao_threshold_sigma``: DAOStarFinder threshold = sigma x std(background); lower values detect more faint
    sources (cf. SIPS ~2.5sigma).

    ``prematch_peak_sigma_floor`` (default 10): before catalog matching, drop **pass-1** DAO rows whose local
    ``peak`` is below ``sky_median + k x sky_mad_sigma`` (SNR-GATE-01). Pass-2 recoveries are exempt when
    ``prematch_exempt_pass2`` is True (local annulus test already applied). Lower **k** keeps more faint pass-1
    detections (MASTERSTAR / ``config.json`` / DAO-STARS typicky **1.8-3.5**).

    Saturation: (1) ``peak_max_adu`` vs resolved ceiling from FITS keywords / ``EQUIPMENTS.SATURATE_ADU`` (before BITPIX);
    (2) **plateau core** - many pixels in the central 3x3
    near the local maximum (flat-top clipping, similar to a saturated radial profile). Row flags:
    ``saturated_from_peak``, ``saturated_plateau``, ``likely_saturated`` (OR), ``photometry_ok`` (not OR).
    """
    import numpy as np

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
            _fov_hint = resolve_plate_solve_fov_deg_hint(
                hdr,
                int(h),
                int(wpx),
                database_path=fov_database_path,
                equipment_id=fov_equipment_id,
                draft_id=fov_draft_id,
            )
        except Exception:  # noqa: BLE001
            _fov_hint = None
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
        gp = (cfg.gaia_db_path or "").strip()
        if gp:
            _gaia_db_path = Path(gp)
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0360] field_catalog_cone_meta.json write failure leaves stale cone-radius metadata for cache ...: %s', exc)
        _gaia_db_path = None
    _fb_sat = fallback_saturate_adu
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
            f"Vykreslujem katalog pre cele zorne pole: {int(wpx)}x{int(h)} pixelov "
            f"(export {len(cat_df)} riadkov do field_catalog_cone.csv, cap={int(_cat_cap_eff)}, kuzel r~{float(radius_deg):.2f} deg)."
        )
        log_event(
            f"KATALOG TARGET: export {_cat_cap_eff} riadkov do field_catalog_cone.csv "
            f"(ak je dostupnych >= {_cat_cap_eff})."
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
    # ``vsx_df`` prazdny DataFrame z prefetch = 'uz sme skusali'; doplnaj len ak volajuci nepredal tabulku (``None``).
    if vsx_df is None:
        _vx: Path | None = None
        try:
            _vxs = str(cfg.vsx_local_db_path or "").strip()
            if _vxs:
                _vx = Path(_vxs).expanduser().resolve()
        except Exception:  # noqa: BLE001
            _vx = None
        vsx_df = _query_vsx_local(center=center, radius_deg=radius_deg, vsx_db_path=_vx)
    exo_annotation_active = False
    exo_max = 3.0
    try:
        exo_max = float(cfg.exoplanet_match_max_sep_arcsec)
        if not math.isfinite(exo_max):
            exo_max = 3.0
    except Exception:  # noqa: BLE001
        exo_max = 3.0
    exo_max = max(0.5, min(30.0, float(exo_max)))
    _exo_path: Path | None = None
    try:
        _exs = str(cfg.exoplanet_local_db_path or "").strip()
        if _exs:
            _exo_path = Path(_exs).expanduser().resolve()
    except Exception:  # noqa: BLE001
        _exo_path = None
    if _exo_path is not None and _exo_path.is_file():
        exo_annotation_active = True
    if exo_df is None and exo_annotation_active:
        exo_df = _query_exoplanet_local(
            center=center,
            radius_deg=radius_deg,
            exoplanet_db_path=_exo_path,
        )
    elif not exo_annotation_active:
        exo_df = pd.DataFrame()
    exo_ann: dict[str, Any] = {}
    if gaia_variable_df is None:
        gaia_variable_df = pd.DataFrame()

    mean, med, std = plain_mean_med_std(arr, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((arr - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bfac = 1
    n_raw_dao = 0
    _cfg_df = AppConfig()
    _fb_c = float(dao_fwhm_px) if dao_fwhm_px is not None else float(_cfg_df.sips_dao_fwhm_px)
    if dao_fwhm_bypass_header and dao_fwhm_px is not None:
        try:
            _dao_arg = float(dao_fwhm_px)
            _base_fw = max(1.2, min(20.0, _dao_arg)) if math.isfinite(_dao_arg) else dao_detection_fwhm_pixels(
                hdr, configured_fallback=_fb_c
            )
        except (TypeError, ValueError):
            _base_fw = dao_detection_fwhm_pixels(hdr, configured_fallback=_fb_c)
    else:
        _base_fw = dao_detection_fwhm_pixels(hdr, configured_fallback=_fb_c)
    _dao_n_equiv_used: float | None = None
    _derived_tol: Any = None
    try:
        from photutils.detection import DAOStarFinder  # type: ignore

        _ds = float(dao_threshold_sigma)
        _ds = max(0.5, min(20.0, _ds))
        dao_scale = _dao_auto_binning_factor(*data0.shape)
        data_dao, bfac = _mean_bin2d_for_dao(data0, dao_scale)
        fwhm_eff = max(1.2, _base_fw / float(bfac))
        from masterstar_gaia_accounting import (  # noqa: PLC0415
            Pass2AcceptParams,
            dao_pass2_born_owned_rows,
            dedup_pass1_spatial,
            estimate_star_masked_sky_sigma,
            merge_dao_pass1_pass2_born_owned,
            star_mask_from_gaia_xy,
        )

        _gaia_chip_det = _gaia_chip_xy_from_catalog(cat_df, wcs_obj, wpx=int(wpx), h=int(h))
        _gx_sm = (
            pd.to_numeric(_gaia_chip_det["x_gaia"], errors="coerce").to_numpy(dtype=np.float64)
            if len(_gaia_chip_det)
            else np.asarray([], dtype=np.float64)
        )
        _gy_sm = (
            pd.to_numeric(_gaia_chip_det["y_gaia"], errors="coerce").to_numpy(dtype=np.float64)
            if len(_gaia_chip_det)
            else np.asarray([], dtype=np.float64)
        )
        _smask = star_mask_from_gaia_xy(_gx_sm, _gy_sm, wpx=int(wpx), h=int(h), fwhm_px=float(_base_fw))
        sky_sig, _sky_med_det = estimate_star_masked_sky_sigma(data0, star_mask=_smask)
        rms_conv, _dao_rel_err = _dao_convolved_background_rms_adu(data_dao, fwhm_px=fwhm_eff)
        sigma_pp_diag = _dao_noise_sigma_adu(arr, bfac=bfac, fallback_std=float(std), data_dao=data_dao)
        std_dao = sky_sig
        if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
            std_dao = rms_conv
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
        _thr_sigma = float(getattr(_cfg_df, "masterstar_dao_threshold_sigma", _ds))
        if not math.isfinite(_thr_sigma) or _thr_sigma <= 0:
            _thr_sigma = float(_ds)
        _thr = max(_thr_sigma * float(std_dao), 1e-6)
        _dao_n_equiv_used = float(_thr_sigma)
        # Adaptive threshold monitoring: match-rate check runs after first catalog match pass (below).
        try:
            _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
            print(
                f"DEBUG DAO INPUT: {_nm} mean={float(np.nanmean(arr)):.1f} std={float(np.nanstd(arr)):.1f} "
                f"threshold={float(_thr):.1f} sky_sigma={float(std_dao):.2f} n_sigma={float(_thr_sigma):.2f} "
                f"rms_conv_diag={float(rms_conv):.2f} sigma_pp_diag={float(sigma_pp_diag):.2f} "
                f"fwhm={float(fwhm_eff):.2f}"
            )
        except Exception:  # noqa: BLE001
            pass
        finder = DAOStarFinder(
            fwhm=float(fwhm_eff),
            threshold=float(_thr),
            scale_threshold=False,
            n_brightest=None,
            **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
        )
        tbl = finder(data_dao)
        n_raw_dao = int(len(tbl)) if tbl is not None else 0
        if tbl is not None and len(tbl) > 0:
            tbl = _prefilter_dao_table_brightest(tbl, max(int(max_catalog_rows) * 12, 36_000))
            from astropy.table import Table as _AstropyTable

            tbl = _AstropyTable(tbl, copy=True)
            tbl["vy_dao_pass"] = np.ones(len(tbl), dtype=np.int16)
        n_pass1_dao = int(len(tbl)) if tbl is not None else 0
        try:
            _dedup_px = float(_cfg_df.masterstar_dao_pass1_dedup_px)
        except (TypeError, ValueError):
            _dedup_px = 0.75
        if tbl is not None and len(tbl) > 0:
            tbl = dedup_pass1_spatial(tbl, sep_px=max(0.25, min(2.0, _dedup_px)))
        try:
            _sigma_p2_cfg = float(_cfg_df.masterstar_dao_pass2_sigma)
        except (TypeError, ValueError):
            _sigma_p2_cfg = 4.0
        try:
            _depth_p2 = float(_cfg_df.masterstar_gaia_census_target_depth_g)
        except (TypeError, ValueError):
            _depth_p2 = 15.0
        try:
            _edge_p2 = float(_cfg_df.masterstar_gaia_census_edge_margin_px)
        except (TypeError, ValueError):
            _edge_p2 = 10.0
        _match_r_coarse_px = _catalog_match_radius_px(
            wcs_obj, match_sep_arcsec=float(match_sep_arcsec), wpx=int(wpx), h=int(h)
        )
        _dao_x_p1 = np.asarray([], dtype=np.float64)
        _dao_y_p1 = np.asarray([], dtype=np.float64)
        if tbl is not None and len(tbl) > 0:
            xb = np.asarray(tbl["x_centroid"], dtype=np.float64)
            yb = np.asarray(tbl["y_centroid"], dtype=np.float64)
            _dao_x_p1, _dao_y_p1 = _dao_xy_binned_to_full(xb, yb, int(bfac))
        from dao_gaia_calibration import (  # noqa: PLC0415
            compute_pass1_astrometric_residuals_px,
            derive_tolerances_from_residuals,
            plate_scale_arcsec_per_px_from_wcs_nan,
        )

        _gg_sm = (
            pd.to_numeric(_gaia_chip_det.get("g_mag"), errors="coerce").to_numpy(dtype=np.float64)
            if len(_gaia_chip_det) and "g_mag" in _gaia_chip_det.columns
            else (
                pd.to_numeric(_gaia_chip_det.get("mag"), errors="coerce").to_numpy(dtype=np.float64)
                if len(_gaia_chip_det) and "mag" in _gaia_chip_det.columns
                else np.asarray([], dtype=np.float64)
            )
        )
        _res_dr = compute_pass1_astrometric_residuals_px(
            _dao_x_p1,
            _dao_y_p1,
            _gx_sm,
            _gy_sm,
            coarse_match_px=float(_match_r_coarse_px),
        )
        _derived_tol = derive_tolerances_from_residuals(
            _res_dr,
            np.asarray([], dtype=np.float64),
            fwhm_px=float(max(1.2, _base_fw)),
            plate_scale_arcsec_per_px=plate_scale_arcsec_per_px_from_wcs_nan(wcs_obj),
            pass1_sigma=float(_thr_sigma),
            pass2_sigma=float(_sigma_p2_cfg),
            match_k=float(getattr(_cfg_df, "masterstar_dao_match_radius_k", 1.7)),
            centroid_floor_px=float(getattr(_cfg_df, "masterstar_dao_centroid_qa_floor_px", 1.0)),
            centroid_cap_px=float(getattr(_cfg_df, "masterstar_dao_centroid_qa_cap_px", 3.0)),
        )
        _match_r_px_p2 = float(_derived_tol.match_radius_px)
        p2_params = Pass2AcceptParams(
            sigma=max(1.5, min(20.0, float(_sigma_p2_cfg))),
            center_tol_px=max(0.5, min(10.0, float(_derived_tol.pass2_center_tol_px))),
            fwhm_px=float(max(1.2, _base_fw)),
        )
        _pass2_rows, _n_unmatched_gaia, _n_pass2_raw, _amb_p2 = dao_pass2_born_owned_rows(
            data0,
            tbl,
            gaia_chip=_gaia_chip_det,
            bfac=int(bfac),
            fwhm_px=float(max(1.2, _base_fw)),
            pass2_params=p2_params,
            target_depth_g=float(_depth_p2),
            edge_margin_px=float(_edge_p2),
            match_r_px=float(_match_r_px_p2),
            wpx=int(wpx),
            h=int(h),
        )
        tbl = merge_dao_pass1_pass2_born_owned(
            tbl, _pass2_rows, bfac=int(bfac), gaia_chip=_gaia_chip_det
        )
        n_merged_dao = int(len(tbl)) if tbl is not None else 0
        LOGGER.info(
            "[DAO pass 1] %d detections, %d Gaia unmatched",
            int(n_pass1_dao),
            int(_n_unmatched_gaia),
        )
        LOGGER.info(
            "[DAO pass 2] %d additional detections from %d targeted positions",
            int(_n_pass2_raw),
            int(_n_unmatched_gaia),
        )
        LOGGER.info("[DAO total] %d detections after merge", int(n_merged_dao))
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
            "n_exo_hosts_in_field": (
                int(len(exo_df)) if exo_annotation_active and exo_df is not None else 0
            ),
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
        f"DAO na snimku: raw={n_raw_dao} (po brightest-prefilter max {max(int(max_catalog_rows) * 12, 36_000):d}) -> "
        f"po priestorovom strope max_n={int(max_catalog_rows)}: {n_spatial} bodov (binning DAO={bfac}x)."
    )
    n = n_spatial
    xb = np.asarray(tbl["x_centroid"], dtype=np.float64)
    yb = np.asarray(tbl["y_centroid"], dtype=np.float64)
    x, y = _dao_xy_binned_to_full(xb, yb, bfac)
    flux = np.asarray(tbl["flux"], dtype=np.float64)
    peak_dao = np.asarray(tbl["peak"], dtype=np.float64) if "peak" in tbl.colnames else np.full(n, np.nan)
    if "vy_seed_catalog_id" in tbl.colnames:
        vy_seed_cid = np.asarray(tbl["vy_seed_catalog_id"], dtype=object)
    else:
        vy_seed_cid = np.array([""] * n, dtype=object)
    if "vy_ambiguous_owner" in tbl.colnames:
        vy_amb_owner = np.asarray(tbl["vy_ambiguous_owner"], dtype=bool)
    else:
        vy_amb_owner = np.zeros(n, dtype=bool)
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
    if exo_annotation_active:
        exo_ann, _exo_warns = _exo_host_annotation_arrays(
            det_coords,
            exo_df if exo_df is not None else pd.DataFrame(),
            exo_max,
            frame_name=frame_name,
        )

    sat_frac = float(saturate_level_fraction)
    sat_frac = min(max(sat_frac, 0.5), 1.0)

    pmax_arr = _box_peaks_at_centroids(arr, x, y)
    _frame_max_adu = float(np.nanmax(arr))
    _empirical_clip_adu = _detect_empirical_clip_level_adu(arr)
    _peak_sat_lim = _resolve_peak_saturation_limit_adu(
        camera_sat_limit_adu=sat_limit,
        saturate_fraction=sat_frac,
        sky_median_adu=float(med),
        frame_max_adu=_frame_max_adu,
        empirical_clip_adu=_empirical_clip_adu,
    )
    _sat_block = _vectorized_star_saturation_columns(
        arr,
        x,
        y,
        sat_limit=_peak_sat_lim,
        sat_frac=sat_frac,
        peak_dao=peak_dao,
        peak_max_adu=pmax_arr,
    )
    _sat_csv, n_sat_pk, n_sat_pl = _proc_sat_block_for_csv(_sat_block)
    # Prematch peak gate (SNR-GATE-01):
    # - noise scale = sky MAD on pixels <= median (not full-frame sample std / scene variance)
    # - pass-2 recoveries already passed a local annulus test; exempt them from this global cut
    _snr_k = float(prematch_peak_sigma_floor)
    if not math.isfinite(_snr_k):
        _snr_k = 10.0
    # Spodna hranica 0.5 = zhoda s AppConfig / DAO-STARS pre MASTERSTAR; horna 15 = per-frame default k=10 zostane platny.
    _snr_k = min(15.0, max(0.5, _snr_k))
    _sky_med_gate, _sky_sig_gate = sky_mad_sigma_adu(arr)
    if not (math.isfinite(_sky_sig_gate) and float(_sky_sig_gate) > 0):
        _sky_sig_gate = float(std) if np.isfinite(std) else 1.0
    if not math.isfinite(_sky_med_gate):
        _sky_med_gate = float(med)
    _bg_sigma_adu = max(float(_sky_sig_gate), 1.0)
    noise_floor = float(float(_sky_med_gate) + _snr_k * _bg_sigma_adu)
    if "vy_dao_pass" in tbl.colnames:
        _dao_pass = np.asarray(tbl["vy_dao_pass"], dtype=np.int16)
        if int(_dao_pass.size) != int(n):
            # Spatial / other pre-filters may have shortened peak arrays; fall back to pass-1.
            _dao_pass = np.ones(n, dtype=np.int16)
    else:
        _dao_pass = np.ones(n, dtype=np.int16)
    _is_pass2 = _dao_pass == 2
    if bool(prematch_exempt_pass2):
        snr_keep = _is_pass2 | (np.isfinite(pmax_arr) & (pmax_arr > noise_floor))
    else:
        snr_keep = np.isfinite(pmax_arr) & (pmax_arr > noise_floor)
    n_snr = int(np.count_nonzero(snr_keep))
    n_gate_drop = int(np.count_nonzero(~snr_keep))
    n_pass2_kept = int(np.count_nonzero(snr_keep & _is_pass2))
    _exempt_tag = "pass2 exempt" if bool(prematch_exempt_pass2) else "pass2 gated"
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
        if exo_annotation_active:
            exo_ann = _slice_exo_annotation(exo_ann, snr_keep)
        pmax_arr = pmax_arr[snr_keep]
        _sat_block = {k: np.asarray(v)[snr_keep] for k, v in _sat_block.items()}
        _sat_csv, n_sat_pk, n_sat_pl = _proc_sat_block_for_csv(_sat_block)
        vy_seed_cid = vy_seed_cid[snr_keep]
        vy_amb_owner = vy_amb_owner[snr_keep]
        _dao_pass = _dao_pass[snr_keep]
        n = int(n_snr)
        log_event(
            f"DAO po SNR filtri (sky_mad median+{_snr_k:.1f}xsigma; {_exempt_tag}): {n}/{n_spatial} bodov "
            f"(noise_floor~{noise_floor:.1f} ADU sky_sig~{_bg_sigma_adu:.1f}; "
            f"pass2_kept={n_pass2_kept}; dropped={n_gate_drop}; pred matchom s katalogom)."
        )
    elif n_snr == 0:
        log_event(
            f"DAO SNR filter by zahodil vsetko - ponechavam {n_spatial} bodov pred matchom."
        )

    idx_det = np.arange(1, n + 1, dtype=np.int32)
    det_str = np.array([f"DET_{i:04d}" for i in idx_det], dtype=object)
    n_matched = 0
    match_sep_used = max(12.0, float(match_sep_arcsec))
    _match_sep_formula_inputs: dict[str, Any] = {}
    _wcs_refine_iters = 0
    from wcs_invertibility import empty_identity_gate_acc

    _identity_gate_acc = empty_identity_gate_acc()
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
                "match_sep_arcsec": np.full(n, np.nan, dtype=np.float64),
                "x": x,
                "y": y,
                "flux": flux,
                "vsx_known_variable": vsx_hit,
                "gaia_dr3_variable_catalog": gvar_hit,
                **(exo_ann if exo_annotation_active else {}),
                **_sat_csv,
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
                # Born-owned pass2: pre-lock seed catalog_id (no greedy reassignment).
                _cid_to_cr = {
                    str(cid_c[_k]).strip(): _k for _k in range(nc) if str(cid_c[_k]).strip()
                }
                for i in range(n):
                    if int(_dao_pass[i]) != 2:
                        continue
                    sc = str(vy_seed_cid[i] if i < len(vy_seed_cid) else "").strip()
                    if not sc:
                        continue
                    cr_b = _cid_to_cr.get(sc)
                    if cr_b is None or cr_b in used_cat or i in used_det:
                        continue
                    used_det.add(i)
                    used_cat.add(cr_b)
                    cat_row[i] = int(cr_b)
                    sepa_out[i] = 0.0
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
                [f"{cat_lab[i]!s}_{int(idx_det[i]):04d}" for i in range(n)],
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
                    "match_sep_arcsec": np.where(matched_l, sepa_eff, np.nan),
                    "x": x,
                    "y": y,
                    "flux": flux,
                    "vsx_known_variable": vsx_hit,
                    "gaia_dr3_variable_catalog": gvar_hit,
                    **(exo_ann if exo_annotation_active else {}),
                    **_sat_csv,
                }
            )
            return df_l, n_ma

        def _apply_post_match_identity_gate() -> None:
            nonlocal df_out, n_matched, _identity_gate_acc
            try:
                from wcs_invertibility import (
                    accumulate_identity_gate,
                    apply_post_match_identity_gate_df,
                    gaia_radec_map_from_table,
                )

                _fwhm_gate = float(_fwhm_used)
                if not math.isfinite(_fwhm_gate) or _fwhm_gate <= 0:
                    _fwhm_gate = 3.5
                _gmap = gaia_radec_map_from_table(cat_df)
                _det_fb = None
                if len(df_out) == int(len(det_str)):
                    import pandas as _pd

                    _det_fb = _pd.Series(det_str, index=df_out.index)
                df_out, _idc = apply_post_match_identity_gate_df(
                    df_out,
                    wcs_obj,
                    gaia_ra_dec_by_cid=_gmap,
                    fwhm_px=_fwhm_gate,
                    log_fn=log_event,
                    det_fallback_names=_det_fb,
                )
                n_matched = int(
                    df_out.get("catalog_id", pd.Series([""] * len(df_out)))
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .sum()
                )
                _identity_gate_acc = accumulate_identity_gate(_identity_gate_acc, _idc, n_matched)
            except Exception as _idg_exc:  # noqa: BLE001
                log_event(f"post_match_identity_gate skipped: {_idg_exc!s}")

        def _run_full_match_pass() -> None:
            nonlocal df_out, n_matched, match_sep_used, _match_sep_formula_inputs
            from dao_gaia_calibration import (
                catalog_match_radius_d1_arcsec,
                plate_scale_arcsec_per_px_from_wcs_nan,
                solve_rms_px_from_fits_header,
            )

            _ps_match = float(plate_scale_arcsec_per_px_from_wcs_nan(wcs_obj))
            _rms_match = solve_rms_px_from_fits_header(hdr)
            match_sep_used, _d1_inputs = catalog_match_radius_d1_arcsec(
                solve_rms_px=_rms_match,
                fwhm_dao_px=float(_fwhm_used),
                plate_scale_arcsec_per_px=_ps_match,
                floor_arcsec=12.0,
            )
            LOGGER.info(
                "Catalog match: radius = max(12, 3 x FWHM_dao=%.3f px x scale=%.4f arcsec/px) "
                "-> %.2f arcsec (formula=%.2f; solve_rms=%.3f px diagnostic only)",
                float(_d1_inputs["fwhm_dao_px"]),
                float(_d1_inputs["plate_scale_arcsec_per_px"] or float("nan")),
                float(match_sep_used),
                float(_d1_inputs["formula_arcsec"] or float("nan")),
                float(_d1_inputs["solve_rms_px"] or float("nan")),
            )
            df_out, n_matched = _assign_catalog_at_threshold(match_sep_used)
            # After a successful loose initial match, tighten for cleaner final IDs (only if most matches survive).
            _tight_sec = 4.5
            if n_matched >= max(10, int(0.20 * max(1, n))) and float(match_sep_used) > _tight_sec + 1e-9:
                df_tight, n_tight = _assign_catalog_at_threshold(_tight_sec)
                if n_tight >= max(8, int(0.92 * max(1, n_matched))):
                    LOGGER.info(
                        "Catalog match: pociatocny loose match %.2f arcsec -> finalne zuzenie na %.2f arcsec (matches %d -> %d)",
                        float(match_sep_used),
                        _tight_sec,
                        int(n_matched),
                        int(n_tight),
                    )
                    df_out, n_matched, match_sep_used = df_tight, n_tight, _tight_sec
            _apply_post_match_identity_gate()
            _match_sep_formula_inputs = dict(_d1_inputs)

        _run_full_match_pass()
        if len(df_out) == int(n):
            df_out["vy_dao_pass"] = np.asarray(_dao_pass, dtype=np.int16)
            df_out["ambiguous_owner"] = np.asarray(vy_amb_owner, dtype=bool)
        if n >= 8:
            _dao_match_rate = float(n_matched) / float(max(1, n))
            if _dao_match_rate < 0.88:
                LOGGER.warning(
                    "[DAO] Match rate %.1f%% below 88%% threshold, "
                    "consider lowering masterstar_dao_threshold_sigma in config.json",
                    100.0 * _dao_match_rate,
                )
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
                            "Catalog match: WCS refine zamietnuty (rms=%.2fpx > 10) - sirsi pixelovy matching.",
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
                        # re-running multi-hundred-k row SQLite queries on every refine iteration (was ~10x per frame).
                        _skip_gaia_rerequery = float(radius2) >= 5.0
                        if _skip_gaia_rerequery:
                            LOGGER.info(
                                "Catalog match: WCS refine - ponechavam existujuci lokalny Gaia vyrez "
                                f"(r={float(radius2):.2f} deg >= 5 deg; bez opatovneho SQL dotazu)."
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
                                    "Catalog match: WCS refine - Gaia re-query < 120 hviezd; refine zruseny."
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
                                if exo_annotation_active:
                                    exo_ann, _ = _exo_host_annotation_arrays(
                                        det_coords,
                                        exo_df if exo_df is not None else pd.DataFrame(),
                                        exo_max,
                                        frame_name=frame_name,
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
                        LOGGER.info("Catalog match: WCS refine bez noveho Gaia kuzela (gaia_db_path).")
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
                    if exo_annotation_active:
                        exo_ann, _ = _exo_host_annotation_arrays(
                            det_coords,
                            exo_df if exo_df is not None else pd.DataFrame(),
                            exo_max,
                            frame_name=frame_name,
                        )
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
                        if exo_annotation_active:
                            exo_ann, _ = _exo_host_annotation_arrays(
                                det_coords,
                                exo_df if exo_df is not None else pd.DataFrame(),
                                exo_max,
                                frame_name=frame_name,
                            )
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
                        "Catalog match je slaby: %s/%s detekcii v ramci %.2f arcsec; median vzdialenosti k najblizsiemu "
                        "katalogu ~ %.2f arcsec - skus zvacsit 'Max catalog match distance arcsec, overit plate solve (FOV, RA/Dec) "
                        "a lokalna Gaia DR3.%s",
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
    cat_nonempty = (
        df_out["catalog"].fillna("").astype(str).str.strip().ne("")
        if len(df_out) and "catalog" in df_out.columns
        else pd.Series([], dtype=bool)
    )
    n_matched_final = int(cat_nonempty.sum()) if len(df_out) else 0
    _ps_idg: float | None = None
    try:
        from dao_gaia_calibration import plate_scale_arcsec_per_px_from_wcs_nan as _ps_idg_fn

        _ps_try = float(_ps_idg_fn(wcs_obj))
        if math.isfinite(_ps_try) and _ps_try > 0:
            _ps_idg = _ps_try
    except Exception:  # noqa: BLE001
        _ps_idg = None
    meta = {
        "noise_floor_adu": float(noise_floor),
        "sky_median_adu": float(_sky_med_gate),
        "bg_sigma_adu": float(_bg_sigma_adu),
        "bg_sigma_estimator": "sky_mad_le_median",
        "prematch_pass2_exempt": bool(prematch_exempt_pass2),
        "frame_max_adu": float(_frame_max_adu),
        "empirical_clip_adu": _empirical_clip_adu,
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
        "n_exo_hosts_in_field": int(len(exo_df)) if exo_annotation_active and exo_df is not None else 0,
        "n_gaia_variable_in_field": int(len(gaia_variable_df)) if gaia_variable_df is not None else 0,
        **foot_meta,
        "field_catalog_cone_csv": str(Path(field_catalog_export_path)) if field_catalog_export_path else None,
        "dao_threshold_sigma": float(dao_threshold_sigma),
        "dao_detection_n_equiv": (
            float(_dao_n_equiv_used) if _dao_n_equiv_used is not None and math.isfinite(_dao_n_equiv_used) else None
        ),
        "dao_fwhm_px": _fwhm_used,
        "dao_detect_binning": int(bfac),
        "prematch_peak_sigma_floor": float(_snr_k),
        "match_sep_arcsec_requested": float(match_sep_arcsec),
        "match_sep_arcsec_effective": float(match_sep_used),
        "match_sep_formula_inputs": dict(_match_sep_formula_inputs),
        "wcs_gaia_pixel_refine_iters": int(_wcs_refine_iters),
        "catalog_match_fraction_target": 0.95,
        "catalog_match_fraction_met": (
            bool((float(n_matched_final) / float(max(1, len(df_out)))) >= 0.95) if len(df_out) else True
        ),
        "dao_gaia_derived_tol": (
            _derived_tol.to_dict() if _derived_tol is not None else None
        ),
        "identity_gate": {
            **dict(_identity_gate_acc),
            "fwhm_px": float(_fwhm_used),
            "plate_scale_arcsec_per_px": (
                float(_ps_idg) if _ps_idg is not None and math.isfinite(float(_ps_idg)) else None
            ),
            "fail_threshold_px": float(3.0 * float(_fwhm_used)),
            "fail_threshold_arcsec": (
                float(3.0 * float(_fwhm_used) * float(_ps_idg))
                if _ps_idg is not None and math.isfinite(float(_ps_idg))
                else None
            ),
        },
        **meta_mag,
    }
    _catalog_rows = int(meta.get("catalog_rows", 0))
    if _catalog_rows > 0:
        if "catalog_id" in df_out.columns:
            _cid_u = (
                df_out["catalog_id"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            _n_gaia_detected = int(_cid_u[_cid_u != ""].nunique())
        else:
            _n_gaia_detected = int(n_matched_final)
        _gaia_dao_rate = 100.0 * float(_n_gaia_detected) / float(_catalog_rows)
        LOGGER.info(
            "[DAO] Gaia->DAO completeness (raw): "
            "%d/%d Gaia stars detected (%.1f%%) "
            "| catalog_only (undetected): %d",
            _n_gaia_detected,
            _catalog_rows,
            _gaia_dao_rate,
            _catalog_rows - _n_gaia_detected,
        )
        meta["gaia_dao_completeness_raw_pct"] = round(_gaia_dao_rate, 2)
        meta["n_gaia_detected"] = int(_n_gaia_detected)
        meta["n_gaia_undetected"] = int(_catalog_rows - _n_gaia_detected)
        try:
            _plate_recon = None
            if getattr(wcs_obj, "has_celestial", False):
                try:
                    from astropy.wcs.utils import proj_plane_pixel_scales

                    _plate_recon = float(np.mean(proj_plane_pixel_scales(wcs_obj) * 3600.0))
                except Exception:  # noqa: BLE001
                    pass
            if _gaia_db_path is not None and getattr(wcs_obj, "has_celestial", False):
                _max_mag_recon = (
                    float(faintest_mag_limit)
                    if faintest_mag_limit is not None and np.isfinite(float(faintest_mag_limit))
                    else 18.0
                )
                _recon = compute_gaia_dao_reconcile(
                    df_out,
                    gaia_db_path=_gaia_db_path,
                    wcs=wcs_obj,
                    naxis1=int(wpx),
                    naxis2=int(h),
                    fwhm_px=float(_fwhm_used),
                    plate_scale_arcsec=_plate_recon,
                    mag_limit=_max_mag_recon,
                    match_sep_arcsec=float(match_sep_used),
                    cone_df=cat_df,
                )
                _md = resolve_effective_match_depth(meta, is_masterstar=False)
                _recon.update(_md)
                meta.update(reconcile_to_pipeline_meta(_recon))
                LOGGER.info(
                    "[DAO] Gaia->DAO reconcile: completeness_50=%.1f%% (matched=%d missed=%d "
                    "off_frame=%d below_limit=%d blended=%d) G_lim_50=%.2f fit=%s",
                    float(meta.get("gaia_dao_completeness_pct") or 0.0),
                    int(meta.get("n_gaia_matched") or 0),
                    int(meta.get("n_gaia_missed") or 0),
                    int(meta.get("n_gaia_off_frame") or 0),
                    int(meta.get("n_gaia_below_limit") or 0),
                    int(meta.get("n_gaia_blended") or 0),
                    float(meta.get("g_lim_50") or meta.get("g_lim_est") or 0.0),
                    str(meta.get("fit_method") or "?"),
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[DAO] Gaia reconcile decomposition failed: %s", exc)
            meta["gaia_dao_completeness_pct"] = round(_gaia_dao_rate, 2)
        _corr = meta.get("gaia_dao_completeness_pct")
        if _corr is not None and float(_corr) < 80.0:
            LOGGER.warning(
                "[DAO] Gaia->DAO corrected completeness LOW: %.1f%% (%d genuinely-missed in-frame)",
                float(_corr),
                int(meta.get("n_gaia_missed") or 0),
            )
    else:
        LOGGER.debug("[DAO] catalog_rows not available - Gaia->DAO skip")
    df_out = _proc_rename_det_names_to_catalog_id(df_out)
    try:
        _gain_ms = float(getattr(_cfg_df, "gain", 1.0) or 1.0)
    except (TypeError, ValueError):
        _gain_ms = 1.0
    if not math.isfinite(_gain_ms) or _gain_ms <= 0:
        _gain_ms = 1.0
    try:
        _rn_ms = float(getattr(_cfg_df, "read_noise", 10.0) or 10.0)
    except (TypeError, ValueError):
        _rn_ms = 10.0
    try:
        _ap_fac = float(getattr(_cfg_df, "aperture_fwhm_factor", 1.9) or 1.9)
    except (TypeError, ValueError):
        _ap_fac = 1.9
    try:
        _ann_in = float(getattr(_cfg_df, "annulus_inner_fwhm", 4.75) or 4.75)
    except (TypeError, ValueError):
        _ann_in = 4.75
    try:
        _ann_out = float(getattr(_cfg_df, "annulus_outer_fwhm", 9.0) or 9.0)
    except (TypeError, ValueError):
        _ann_out = 9.0
    df_out = stamp_masterstar_snr_columns(
        df_out,
        image=arr,
        fwhm_dao_px=float(_fwhm_used),
        bg_sigma_adu=float(_bg_sigma_adu),
        gain=_gain_ms,
        read_noise=_rn_ms,
        aperture_fwhm_factor=_ap_fac,
        annulus_inner_fwhm=_ann_in,
        annulus_outer_fwhm=_ann_out,
    )
    return df_out, meta


_PIXEL_MATCH_DEBUG_LOGGED = False


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
    full_catalog_export: bool = False,
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
    Parallelism: jednotny pocet z ``app_config`` / env (``VYVAR_PARALLEL_WORKERS`` alebo legacy env, pozri
    :func:`_vyvar_parallel_worker_count`). When ``>1``, uses ``ProcessPoolExecutor`` (``spawn``); the
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

    ps = Path(platesolve_dir) if platesolve_dir is not None else None
    _qc_fwhm_by_name: dict[str, float] = {}
    _fwhm_night_median_px: float | None = None
    try:
        from aperture_policy import load_qc_fwhm_map, normalize_aperture_policy_mode  # noqa: PLC0415

        _qc_root = None
        if ps is not None:
            try:
                _qc_root = Path(ps).resolve().parents[1]
            except IndexError:
                _qc_root = None
        _qc_csv = find_qc_metrics_csv(_qc_root) if _qc_root is not None else None
        _qc_fwhm_by_name, _fwhm_night_median_px = load_qc_fwhm_map(_qc_csv)
        _ap_policy_mode = normalize_aperture_policy_mode(
            getattr(_cfg_ap, "aperture_policy_mode", "f_fixed_night")
        )
    except Exception as _qc_exc:  # noqa: BLE001
        LOGGER.warning("[APERTURE-01] QC FWHM map not loaded: %s", _qc_exc)
        _ap_policy_mode = "f_fixed_night"

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
        "aperture_correction_enabled": bool(_cfg_ap.aperture_correction_enabled),
        "aperture_fwhm_factor_small": float(_cfg_ap.aperture_snr_sizing.get("small", 1.5)),
        "aperture_fwhm_factor_large": float(_cfg_ap.aperture_snr_sizing.get("large", 4.0)),
        "aperture_policy_mode": str(_ap_policy_mode),
        "fwhm_night_median_px": _fwhm_night_median_px,
        "qc_fwhm_by_name": dict(_qc_fwhm_by_name),
        "platesolve_dir": str(ps.resolve()) if ps is not None else "",
        "cog_aperture_correction_enabled": bool(_cfg_ap.cog_aperture_correction_enabled),
        "cog_ref_fwhm": float(_cfg_ap.cog_ref_fwhm),
        "cog_min_stars": int(_cfg_ap.cog_min_stars),
        "cog_isolation_fwhm": float(_cfg_ap.cog_isolation_fwhm),
        "cog_snr_min": float(_cfg_ap.cog_snr_min),
        "cog_sat_frac": float(_cfg_ap.cog_sat_frac),
        "cog_ladder_step_px": float(_cfg_ap.cog_ladder_step_px),
        "cog_ladder_step_fwhm": getattr(_cfg_ap, "cog_ladder_step_fwhm", None),
        "cog_ac_factor_max": float(_cfg_ap.cog_ac_factor_max),
        "gain": float(_cfg_ap.gain),
        "read_noise": float(_cfg_ap.read_noise),
        "err_background_mode": "empirical",
        "err_empty_apertures_n": int(_cfg_ap.err_empty_apertures_n),
        "err_empty_apertures_min": int(_cfg_ap.err_empty_apertures_min),
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
    if ps is None:
        ps = Path(platesolve_dir)
    _ap_st.update(_export_catalog_psf_st_fields(_cfg_ap, ps))
    _epsf_science_meta: dict[str, Any] | None = None
    if bool(_cfg_ap.psf_photometry_enabled) and bool(_ap_st.get("_run_epsf")):
        from epsf_science_set import build_epsf_science_set

        _sci = build_epsf_science_set(ps)
        if not _sci.catalog_ids:
            raise ValueError(
                "ePSF science set is empty"
                + (f": {_sci.empty_reason}" if _sci.empty_reason else "")
                + "; refusing silent fallback to full LC pool."
            )
        _epsf_science_meta = _sci.to_meta_dict()
        _ap_st["epsf_science_set_meta"] = _epsf_science_meta
    out_flat = ps / "per_frame_catalogs"
    if mirror_flat_platesolve_folder:
        out_flat.mkdir(parents=True, exist_ok=True)

    work_ram: list[tuple[str, fits.Header, Any]] | None = None
    if use_ram_inputs:
        root.mkdir(parents=True, exist_ok=True)
        work_ram = sorted(list(aligned_ram), key=lambda t: t[0])
        files = [root / name for name, _, _ in work_ram]
    else:
        if full_catalog_export:
            files = sorted(_iter_fits_recursive(root))
        else:
            from epsf_frame_accounting import list_epsf_science_light_fits

            files = list_epsf_science_light_fits(root)

    _frame_index_by_name = {Path(f).name: i for i, f in enumerate(files)}
    _ap_st["epsf_frame_index_by_name"] = _frame_index_by_name

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

    _pfov_res: float | None = None
    try:
        _pf0 = float(plate_solve_fov_deg) if plate_solve_fov_deg is not None else float("nan")
        if math.isfinite(_pf0) and _pf0 > 0:
            _pfov_res = _pf0
    except (TypeError, ValueError):
        _pfov_res = None
    if _pfov_res is None and files:
        try:
            _rf0 = files[0]
            with fits.open(_rf0, memmap=False) as _h0:
                _hd0 = _h0[0].header.copy()
                _ar0 = np.asarray(_h0[0].data)
            if _ar0.ndim == 2:
                _pfov_res = resolve_plate_solve_fov_deg_hint(
                    _hd0,
                    int(_ar0.shape[0]),
                    int(_ar0.shape[1]),
                    database_path=_cfg_ap.database_path,
                    equipment_id=int(equipment_id) if equipment_id is not None else None,
                    draft_id=int(draft_id) if draft_id is not None else None,
                )
        except Exception:  # noqa: BLE001
            _pfov_res = None
    if _pfov_res is None:
        try:
            _pfov_res = float(_cfg_ap.plate_solve_fov_deg)
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
            _hdr_fc = pd.read_csv(field_cat_path, nrows=0)
            _dtype_fc: dict[str, type] = {}
            if "catalog_id" in _hdr_fc.columns:
                _dtype_fc["catalog_id"] = str
            _kw_fc: dict[str, Any] = {}
            if _dtype_fc:
                _kw_fc["dtype"] = _dtype_fc
            _cdf = pd.read_csv(field_cat_path, **_kw_fc)
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
                                "Per-frame catalog: ignoring cached %s (full chip needs cone_radius_deg~%.6f, "
                                "cached %.6f from %s) - fetching larger Gaia cone",
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
                        "Per-frame catalog: reusing %s (%s rows) - skipping duplicate cone query",
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
        _gp = (_cfg_e.gaia_db_path or "").strip()
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
                _hdr_ms = pd.read_csv(_msp, nrows=0)
                _dtype_ms: dict[str, type] = {}
                if "catalog_id" in _hdr_ms.columns:
                    _dtype_ms["catalog_id"] = str
                if "name" in _hdr_ms.columns:
                    _dtype_ms["name"] = str
                _kw_ms: dict[str, Any] = {}
                if _dtype_ms:
                    _kw_ms["dtype"] = _dtype_ms
                _mt = pd.read_csv(_msp, **_kw_ms)
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

    _sat_diag_ctx: Any | None = None
    _sat_diag_archive: str = ""
    _ref_ra_deg: float | None = None
    _ref_dec_deg: float | None = None
    _drift_ref_catalog_id: str | None = None
    try:
        from sat_diag import (  # noqa: PLC0415
            draft_archive_from_platesolve,
            run_sat_diag,
        )

        _arch = draft_archive_from_platesolve(ps)
        if _arch is not None:
            _sat_diag_archive = str(_arch)
            _eq_sat = equipment_saturate_adu
            if _eq_sat is None and equipment_id is not None:
                _eq_sat = _equipment_saturate_adu_from_db(int(equipment_id))
            _ref_hdr = fits.Header()
            if files:
                with fits.open(files[0], memmap=False) as _rh:
                    _ref_hdr = _rh[0].header
            _sat_diag_ctx = run_sat_diag(_arch, equipment_adu=_eq_sat, hdr=_ref_hdr)
            if _sat_diag_ctx.sat_adu is not None:
                equipment_saturate_adu = float(_sat_diag_ctx.sat_adu)
            from sat_diag import resolve_drift_ref_sky_deg  # noqa: PLC0415

            _frame_hint = Path(files[0]).name if files else None
            _ref_ra_deg, _ref_dec_deg, _drift_ref_catalog_id = resolve_drift_ref_sky_deg(
                ps, frame_name_hint=_frame_hint
            )
            if _ref_ra_deg is None and master_tab is not None and not getattr(master_tab, "empty", True):
                if "ra_deg" in master_tab.columns and "dec_deg" in master_tab.columns:
                    _mra = pd.to_numeric(master_tab["ra_deg"], errors="coerce")
                    _mde = pd.to_numeric(master_tab["dec_deg"], errors="coerce")
                    _flux_col = "flux" if "flux" in master_tab.columns else None
                    if _flux_col:
                        _ord = pd.to_numeric(master_tab[_flux_col], errors="coerce").fillna(0)
                        _j = int(_ord.idxmax())
                    else:
                        _j = 0
                    if math.isfinite(float(_mra.iloc[_j])) and math.isfinite(float(_mde.iloc[_j])):
                        _ref_ra_deg = float(_mra.iloc[_j])
                        _ref_dec_deg = float(_mde.iloc[_j])
            if _ref_ra_deg is not None:
                LOGGER.info(
                    "[SAT-DIAG] drift reference sky (%.5f, %.5f) from %s",
                    _ref_ra_deg,
                    _ref_dec_deg,
                    _frame_hint or "platesolve",
                )
            LOGGER.info(
                "[SAT-DIAG] sat_adu=%s source=%s lin_adu=%s (archive %s)",
                _sat_diag_ctx.sat_adu,
                _sat_diag_ctx.sat_source,
                _sat_diag_ctx.lin_adu,
                _sat_diag_archive,
            )
    except Exception as _sd_exc:  # noqa: BLE001
        LOGGER.warning("[SAT-DIAG] init skipped: %s", _sd_exc)

    _gauss_override: float | None = None
    try:
        if masterstar_fits is not None:
            _ms_gauss = Path(masterstar_fits)
            if _ms_gauss.is_file():
                with fits.open(_ms_gauss, memmap=False) as _gfh:
                    _ghdr = _gfh[0].header
                    # PRIORITA 1: VY_FWHM_GAUSS - 2D Gaussian fit, closest to SExtractor
                    for _gk in ("VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN"):
                        _vg = _ghdr.get(_gk)
                        if _vg is None:
                            continue
                        try:
                            _vgf = float(_vg)
                            if 1.0 <= _vgf <= 15.0:
                                _gauss_override = _vgf
                                LOGGER.debug(
                                    "[FWHM] gaussian_override from %s: %.3f px",
                                    _gk,
                                    _gauss_override,
                                )
                                break
                        except (TypeError, ValueError):
                            pass

                    # PRIORITA 2: VY_FWHM x 0.667 fallback
                    if _gauss_override is None:
                        _vy = _ghdr.get("VY_FWHM")
                        if _vy is not None:
                            try:
                                _vyf = float(_vy)
                                if 1.0 <= _vyf <= 15.0:
                                    _gauss_override = _vyf * (1.0 / 1.5)
                                    LOGGER.debug(
                                        "[FWHM] gaussian_override from VY_FWHMx0.667: %.3f px",
                                        _gauss_override,
                                    )
                            except (TypeError, ValueError):
                                pass
    except Exception:  # noqa: BLE001
        _gauss_override = None
    _ap_st["gaussian_fwhm_px_override"] = _gauss_override
    if _gauss_override is not None:
        log_event(
            f"[PHOT] gaussian_fwhm_px_override = {float(_gauss_override):.4f}px "
            "(z VY_FWHM alebo VY_FWHM_GAUSS)"
        )
    else:
        log_event("[PHOT] gaussian_fwhm_px_override = None -> fallback na momentx0.619 per frame")

    cfg_for_workers = app_config if app_config is not None else AppConfig()
    _dao_fw_export = (
        float(dao_fwhm_px)
        if dao_fwhm_px is not None
        else float(cfg_for_workers.sips_dao_fwhm_px)
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
            "Per-frame catalog: up to %s process worker(s); jednotny parallel count + RAM cap (psutil); "
            "env VYVAR_PARALLEL_WORKERS / legacy",
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
                    _gp = (_cfg_ap.gaia_db_path or "").strip()
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
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.debug("[PIPE] cone/variables CSV write failed: %s", exc)
                        pass
            if vsx_df is None or getattr(vsx_df, "empty", True):
                _vsx_p2: Path | None = None
                try:
                    _vsp2 = str(_cfg_ap.vsx_local_db_path or "").strip()
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
                _raw_arr = None
                _raw_hdr = None
                if _sat_diag_ctx is not None and _sat_diag_archive:
                    _raw_p = _resolve_draft_light_raw_path(Path(_sat_diag_archive), fname)
                    if _raw_p is not None and _raw_p.is_file():
                        try:
                            from sat_diag import image_adu_array  # noqa: PLC0415

                            with fits.open(_raw_p, memmap=False) as _rhd:
                                if int(_rhd[0].header.get("BITPIX", 0)) >= 0:
                                    _raw_arr = image_adu_array(_rhd[0])
                                    _raw_hdr = _rhd[0].header.copy()
                        except Exception:  # noqa: BLE001
                            _raw_arr = None
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
                    frame_name=fname,
                    sat_diag_ctx=_sat_diag_ctx,
                    raw_data=_raw_arr,
                    raw_hdr=_raw_hdr,
                    ref_ra_deg=_ref_ra_deg,
                    ref_dec_deg=_ref_dec_deg,
                    drift_ref_catalog_id=_drift_ref_catalog_id,
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
                    fov_database_path=_cfg_ap.database_path,
                    fov_equipment_id=int(equipment_id) if equipment_id is not None else None,
                    fov_draft_id=int(draft_id) if draft_id is not None else None,
                )
            except Exception as exc:  # noqa: BLE001
                return {"file": fname, "status": f"error: {exc}", "csv": ""}

        _exo_st = {
            "exoplanet_local_db_path": str(_cfg_ap.exoplanet_local_db_path or ""),
            "exoplanet_match_max_sep_arcsec": float(_cfg_ap.exoplanet_match_max_sep_arcsec),
            "plate_solve_fov_deg": _pfov_res,
            "database_path": str(Path(_cfg_ap.database_path).resolve()),
            "equipment_id": equipment_id,
            "draft_id": draft_id,
        }
        df = _apply_exo_host_columns_to_proc_df(df, hdr, (h_i, wpx_i), _exo_st, frame_name=fname)

        _before_dao = len(df)
        df = _proc_drop_unmatched_dao_rows(df)
        LOGGER.debug("[TODO-13] catalog-only pre-filter (detect): %d -> %d rows", _before_dao, len(df))

        _run_aperture = bool(_ap_st.get("_run_aperture", True))
        _run_epsf = bool(_ap_st.get("_run_epsf", False))
        if _run_aperture:
            _ap_st["current_frame_name"] = fname
            df = _apply_aperture_catalog_enhancements_from_st(df, data, hdr, _ap_st)
        _psf_on = bool(_ap_st.get("psf_photometry_enabled", False))
        _epsf_ids = (
            _epsf_fit_catalog_ids(ps, psf_photometry_enabled=_psf_on)
            if ps is not None
            else None
        )
        LOGGER.info(
            "[ePSF] _run_one_catalog %s: platesolve_dir=%r n_psf_ids=%s lc_set=%s",
            fname,
            str(ps.resolve()) if ps is not None else None,
            len(_epsf_ids) if _epsf_ids is not None else "ALL",
            _psf_on,
        )
        if not _run_aperture and _run_epsf:
            # PSF-only mode: psf_flux promoted to primary.
            pass
        df = _fill_psf_catalog_columns(df, data, hdr, _ap_st, target_ids=_epsf_ids)

        # --- Time columns (JD / HJD / BJD) ---
        _db_tc = None
        try:
            from time_utils import compute_time_columns

            _dbp = str(_ap_st.get("database_path") or "").strip()
            _did_tc = _ap_st.get("draft_id")
            if _dbp and _did_tc is not None:
                try:
                    _db_tc = VyvarDatabase(Path(_dbp))
                except Exception:  # noqa: BLE001
                    _db_tc = None
            _time_cols = compute_time_columns(
                hdr,
                db=_db_tc,
                draft_id=int(_did_tc) if _did_tc is not None else None,
                cfg=_cfg_ap,
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

            # Airmass - frame-level hodnota z FITS hlavicky
            _am_val = _extract_airmass_from_header(
                hdr,
                cfg=_cfg_ap,
                db=_db_tc,
                draft_id=int(_did_tc) if _did_tc is not None else None,
            )
            if "airmass" not in df.columns:
                _am_insert_pos = _pos + len(_tk)
                df.insert(_am_insert_pos, "airmass", _am_val)
        except Exception as _tc_exc:  # noqa: BLE001
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
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)

        df2 = df.copy()
        df2.insert(0, "source_file", fname)

        _before_dedupe = len(df2)
        df2 = _proc_deduplicate_matched_catalog_rows(df2)
        if len(df2) != _before_dedupe:
            LOGGER.debug(
                "[PROC] per-frame catalog dedupe: %d -> %d rows (%s)",
                _before_dedupe,
                len(df2),
                fname,
            )
        _before_cat = len(df2)
        df2 = _proc_catalog_keep_matched_rows_only(df2)
        LOGGER.debug("[TODO-13] catalog-only filter: %d -> %d rows", _before_cat, len(df2))

        csv_paths: list[str] = []
        if write_sidecar_csv_next_to_fits:
            sidecar = proc_csv_path_for_aligned_fits(base_path)
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
            "raw_peaks_used": bool(meta.get("raw_peaks_used")),
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
            "forced_photometry_enabled": bool(
                getattr(cfg_for_workers, "forced_photometry_enabled", True)
            ),
            "forced_photometry_centroid_bound_fwhm": float(
                getattr(cfg_for_workers, "forced_photometry_centroid_bound_fwhm", 2.5) or 2.5
            ),
            "forced_photometry_margin_px": float(
                getattr(cfg_for_workers, "forced_photometry_margin_px", 0.0) or 0.0
            ),
            "equipment_saturate_adu": equipment_saturate_adu,
            "sat_diag_ctx_dict": (
                _sat_diag_ctx.to_json_dict() if _sat_diag_ctx is not None else None
            ),
            "sat_diag_archive": _sat_diag_archive,
            "sat_diag_ref_ra": _ref_ra_deg,
            "sat_diag_ref_dec": _ref_dec_deg,
            "sat_diag_ref_catalog_id": _drift_ref_catalog_id,
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
            "aperture_correction_enabled": bool(_cfg_ap.aperture_correction_enabled),
            "aperture_fwhm_factor_small": float(_cfg_ap.aperture_snr_sizing.get("small", 1.5)),
            "aperture_fwhm_factor_large": float(_cfg_ap.aperture_snr_sizing.get("large", 4.0)),
            "aperture_policy_mode": str(_ap_policy_mode),
            "fwhm_night_median_px": _fwhm_night_median_px,
            "qc_fwhm_by_name": dict(_qc_fwhm_by_name),
            "platesolve_dir": str(ps.resolve()),
            "observer_lat": float(_cfg_ap.observer_lat),
            "observer_lon": float(_cfg_ap.observer_lon),
            "observer_alt_m": float(_cfg_ap.observer_alt_m),
            "exoplanet_local_db_path": str(_cfg_ap.exoplanet_local_db_path or ""),
            "exoplanet_match_max_sep_arcsec": float(_cfg_ap.exoplanet_match_max_sep_arcsec),
            **_export_catalog_psf_st_fields(_cfg_ap, ps),
            "epsf_frame_index_by_name": dict(_frame_index_by_name),
            "epsf_science_set_meta": _epsf_science_meta,
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
    _hybrid_stats: dict[str, Any] = {}
    if not defer_disk_writes:
        _hybrid_stats = _finalize_hybrid_bkg_fallback_sidecar(
            root,
            err_background_mode="empirical",
            write_sidecar=bool(write_sidecar_csv_next_to_fits),
            gain=float(_ap_st.get("gain", _cfg_ap.gain)),
            read_noise=float(_ap_st.get("read_noise", _cfg_ap.read_noise)),
            setup_label=str(root.name),
        )
    if _sat_diag_ctx is not None and _sat_diag_archive:
        try:
            from sat_diag import commit_sat_diag_provenance  # noqa: PLC0415

            _placed_raw = any(bool(r.get("raw_peaks_used")) for r in rows_out)
            commit_sat_diag_provenance(
                _sat_diag_ctx,
                _sat_diag_archive,
                placed_aperture_used=_placed_raw,
            )
        except Exception as _sd_write_exc:  # noqa: BLE001
            LOGGER.warning("[SAT-DIAG] provenance commit skipped: %s", _sd_write_exc)
    _epsf_job_summary: dict[str, Any] | None = None
    if bool(_cfg_ap.psf_photometry_enabled) and bool(_ap_st.get("_run_epsf")):
        from epsf_frame_accounting import finalize_epsf_frame_job

        _psf_recs = [
            r["psf_frame_record"]
            for r in rows_out
            if isinstance(r.get("psf_frame_record"), dict)
        ]
        if _psf_recs:
            _epsf_job_summary = finalize_epsf_frame_job(
                _psf_recs,
                platesolve_dir=ps,
                science_set_meta=_epsf_science_meta,
            )
    return {
        "written": int(n_ok),
        "per_frame_dir": str(root),
        "per_frame_csv_mode": "sidecar" if write_sidecar_csv_next_to_fits else ("flat_mirror" if mirror_flat_platesolve_folder else "none"),
        "index_csv": str(index_path),
        "frames": rows_out,
        "mirror_flat_platesolve_folder": bool(mirror_flat_platesolve_folder),
        "frames_master_reference_match": int(n_master_ref),
        "deferred_csv_writes": list(deferred_csv_writes) if defer_disk_writes else [],
        "hybrid_bkg_fallback": _hybrid_stats,
        "epsf_job_summary": _epsf_job_summary,
    }




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
    n_comparison_stars: int = 0,
    require_non_variable_comparisons: bool = True,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    equipment_saturate_adu: float | None = None,
    catalog_local_gaia_only: bool | None = None,
    app_config: AppConfig | None = None,
    equipment_id: int | None = None,
    draft_id: int | None = None,
    telescope_id: int | None = None,
    master_dark_path: Path | str | None = None,
    masterstar_candidate_paths: "Sequence[str] | None" = None,
    masterstar_selection_pct: float | None = None,
    setup_name: str | None = None,
    masterstar_basename: str = "MASTERSTAR.fits",
    masterstars_csv_basename: str = "masterstars_full_match.csv",
    masterstar_fits_only: bool = False,
    masterstar_skip_build: bool = False,
    masterstar_platesolve_only: bool = False,
    masterstar_platesolve_skip_solve: bool = False,
    hint_ra_deg: float | None = None,
    hint_dec_deg: float | None = None,
) -> dict[str, Any]:
    """Create MASTERSTAR.fits, plate-solve it, and export masterstars.csv.

    Ak je ``masterstar_fits_only=True``, po zostaveni FITS v ``platesolve/`` sa skonci (ziadny plate-solve ani CSV).
    Ak je ``masterstar_skip_build=True``, preskoci sa build z processed - pouzije sa existujuci ``MASTERSTAR.fits`` v ``platesolve/`` a bezi solver + katalog.
    Ak je ``masterstar_platesolve_only=True``, po uspesnom plate-solve a uprave mierky WCS sa skonci (bez DAO CSV, ``masterstars_full_match.csv``, fotometrickeho planu a zapisu MASTER_SOURCES).
    """
    max_catalog_rows = max(int(max_catalog_rows), 100000)
    import numpy as np

    ap = Path(archive_path).expanduser()
    # Draft UI moze poslat .../draft_x/non_calibrated - MASTERSTAR a platesolve patria pod koren draftu.
    if ap.name.casefold() == "non_calibrated":
        ap = ap.parent
    if equipment_saturate_adu is None:
        _sd_clip = _sat_adu_from_draft_sat_diag(ap)
        if _sd_clip is not None:
            equipment_saturate_adu = _sd_clip
            logging.info(
                "[INV-SAT-LIMIT] EQUIPMENTS.SATURATE_ADU missing; using sat_diag.json sat_adu=%.0f",
                _sd_clip,
            )
    detrended_root: Path | None = None
    if masterstar_skip_build:
        ps = Path(platesolve_dir) if platesolve_dir is not None else (ap / "platesolve")
        platesolve_dir = ps
        platesolve_dir.mkdir(parents=True, exist_ok=True)
        _ms_name = str(masterstar_basename or "MASTERSTAR.fits").strip() or "MASTERSTAR.fits"
        masterstar_fits = Path(platesolve_dir) / _ms_name
        if not masterstar_fits.is_file():
            raise FileNotFoundError(
                f"MASTERSTAR plate-solve: v {platesolve_dir} chyba subor {_ms_name}. "
                "Najprv spusti **MAKE MASTERSTAR** na archive alebo vytvor MASTERSTAR inak (FITS QA -> referencny snimok)."
            )
        _match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
        if _match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
            log_event(
                f"MASTERSTAR: catalog match sep eff={_match_sep_eff:.2f} arcsec (min 10 for initial match)."
            )
        log_event(
            f"MASTERSTAR platesolve-from-disk: {masterstar_fits.resolve()} - VYVAR solver + katalog "
            "(bez noveho buildu z processed)."
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
            detrended_root = resolve_masterstar_input_root(
                ap,
                setup_name=setup_name,
                app_config=app_config,
                draft_id=draft_id,
            )
            if detrended_root is None:
                raise FileNotFoundError(
                    f"MASTERSTAR input root for setup {str(setup_name)!r} not found under "
                    f"{ap / 'processed' / 'lights'} (refusing cross-group fallback)."
                )
        if not detrended_root.exists():
            if setup_name:
                raise FileNotFoundError(
                    f"MASTERSTAR input root for setup {str(setup_name)!r} not found: {detrended_root}"
                )
            log_event(f"[X] MASTERSTAR FAIL: Input path {detrended_root} not found.")
            processed_lights = ap / "processed" / "lights"
            if processed_lights.is_dir():
                subdirs = sorted(
                    [d for d in processed_lights.iterdir() if d.is_dir()],
                    key=lambda p: p.name.casefold(),
                )
                if subdirs:
                    detrended_root = subdirs[0]
                    log_event(f"[OK] MASTERSTAR fallback input found: {detrended_root}")
        if not detrended_root.exists():
            raise FileNotFoundError(f"Missing processed/detrended lights: {detrended_root}")
        # If root exists but has no FITS, try first setup subfolder under processed lights (single-group only).
        if not setup_name and not _iter_fits_recursive(detrended_root):
            processed_lights = ap / "processed" / "lights"
            if processed_lights.is_dir():
                subdirs = sorted(
                    [d for d in processed_lights.iterdir() if d.is_dir()],
                    key=lambda p: p.name.casefold(),
                )
                for sd in subdirs:
                    if _iter_fits_recursive(sd):
                        log_event(f"[OK] MASTERSTAR fallback to setup subdir: {sd}")
                        detrended_root = sd
                        break

        log_event(f"[search] MASTERSTAR: Searching for candidates in {Path(detrended_root).resolve()}")
        log_event(f"Vstupny priecinok pre Masterstar: {Path(detrended_root).resolve()}")
        from draft_provenance import is_pre_calibrated_draft

        _pre_cal_ms = is_pre_calibrated_draft(ap, draft_id=draft_id)
        if _pre_cal_ms:
            log_event(
                "MASTERSTAR: pre-calibrated draft - candidates resolved directly under "
                f"{Path(detrended_root).resolve()} (no processed/calibrated remap)."
            )
        _match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
        if _match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
            log_event(
                f"MASTERSTAR: catalog match sep zvyseny na {_match_sep_eff:.2f}\" "
                f"(pozadovane minimum pre pociatocny match)."
            )

        ps = Path(platesolve_dir) if platesolve_dir is not None else (ap / "platesolve")
        platesolve_dir = ps
        platesolve_dir.mkdir(parents=True, exist_ok=True)
        _ms_name = str(masterstar_basename or "MASTERSTAR.fits").strip() or "MASTERSTAR.fits"
        masterstar_fits = Path(platesolve_dir) / _ms_name
        only_ms_paths: list[Path] | None = None
        ms_selection_meta: dict[str, Any] = {}
        #: When True, ``masterstar_candidate_paths`` mapped to disk - do not append unrelated FITS
        #: for "best-of-N" pool (that would override a deliberate single-frame pick in the UI).
        explicit_ui_masterstar_paths = False

        def _map_qc_paths_to_disk(raw_paths: list[str]) -> list[Path]:
            """Map UI / DB paths onto draft lights FITS under ``detrended_root``.

            Pre-calibrated: match by basename under ``non_calibrated/lights/<setup>/``.
            VYVAR-calibrated: prefer ``processed/lights/.../proc_*.fits`` via remap helpers.
            """

            def _mapped_hit_ok(hit: Path) -> bool:
                if not hit.is_file() or _path_segments_forbidden_for_masterstar_physical_source(
                    hit, pre_calibrated=_pre_cal_ms
                ):
                    return False
                if _pre_cal_ms:
                    return _path_is_under_tree(Path(detrended_root), hit)
                pl = ap / "processed" / "lights"
                if pl.is_dir():
                    try:
                        hit.resolve().relative_to(pl.resolve())
                        return True
                    except ValueError:
                        return False
                return _path_is_under_tree(Path(detrended_root), hit)

            out: list[Path] = []
            for rp in raw_paths:
                s = str(rp).strip()
                if not s:
                    continue
                hit = _resolve_best_effort_path_under(
                    Path(detrended_root),
                    s,
                    pre_calibrated=_pre_cal_ms,
                )
                if hit is not None and _mapped_hit_ok(hit):
                    out.append(hit)
                    continue
                if _pre_cal_ms:
                    continue
                try:
                    hit2 = resolve_obs_file_to_processed_fits(
                        ap,
                        s,
                        setup_name=setup_name,
                        app_config=app_config,
                        draft_id=draft_id,
                    )
                except Exception:  # noqa: BLE001
                    hit2 = None
                if hit2 is not None and _mapped_hit_ok(hit2):
                    out.append(hit2)
            return out

        def _disk_stack_fallback_paths(input_dir: Path, *, max_frames: int = 8) -> list[Path]:
            """When QC paths / DB mapping fail: pick best frames from disk (deterministic order)."""
            all_on_disk = sorted(
                (
                    fp
                    for fp in _iter_fits_recursive(input_dir)
                    if _path_is_under_tree(input_dir, fp)
                    and not _path_segments_forbidden_for_masterstar_physical_source(
                        fp, pre_calibrated=_pre_cal_ms
                    )
                ),
                key=lambda p: str(p).casefold(),
            )
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
                explicit_ui_masterstar_paths = True
                ms_selection_meta = {
                    "source": "ui_paths",
                    "requested": int(len(cand_paths)),
                    "mapped_found": int(len(mapped)),
                    "explicit_ui_lock": True,
                }
            else:
                raise FileNotFoundError(
                    "MASTERSTAR: z UI/job prisli explicitne cesty k referencnemu snimku, ale ziadna sa nenasla "
                    f"ako ``processed/lights/.../proc_*.fits`` (koren vyberu: {Path(detrended_root).resolve()}). "
                    "Skontroluj preprocess, archiv a vyber vo FITS QA (potvrd znovu po **Create Archive & Do Calibration**). "
                    f"Pozadovane ({len(cand_paths)}): " + "; ".join(cand_paths[:6]) + (" ..." if len(cand_paths) > 6 else "")
                )

        _multi_obs = draft_is_multi_group_obs(ap)
        if only_ms_paths is None and draft_id is not None and not (_multi_obs and setup_name):
            _db_ms = _vyvar_open_database(app_config or AppConfig())
            if _db_ms is not None:
                try:
                    # FITS QA 'Potvrdit vyber MASTERSTAR' -> ``get_obs_draft_masterstar_source_path``
                    # (``draft manifest.MASTERSTAR_PATH`` = zdrojovy frame, nie hotovy ``MASTERSTAR.fits``).
                    # Musi ist pred automatickym top-% z ``manifest files[]``, inak sa pouzivatelsky vyber prepise.
                    _src = _db_ms.get_obs_draft_masterstar_source_path(int(draft_id))
                    if _src and str(_src).strip():
                        mapped_src = _map_qc_paths_to_disk([str(_src).strip()])
                        if mapped_src:
                            only_ms_paths = mapped_src
                            explicit_ui_masterstar_paths = True
                            ms_selection_meta = {
                                "source": "db_masterstar_source_path",
                                "draft_id": int(draft_id),
                                "mapped_found": int(len(mapped_src)),
                                "explicit_ui_lock": True,
                            }
                            log_event(
                                f"MASTERSTAR: FITS QA vyber (DB source, draft {int(draft_id)}) -> {mapped_src[0].name}"
                            )
                    if only_ms_paths is None:
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
                                f"MASTERSTAR: vyber z DB (draft {int(draft_id)}, top {_pct_eff:g} %) -> "
                                f"{len(mapped_db)} kandidatov (najlepsi sa skopiruje do platesolve)."
                            )
                        else:
                            log_event(
                                f"MASTERSTAR: DB vyber (draft {int(draft_id)}) sa nepodarilo namapovat na FITS pod {detrended_root}."
                            )
                except Exception as exc:  # noqa: BLE001
                    # EXC-0394: T4 -- `_dbc_fw.conn.close()` cleanup only; no radiometry or frame data touched. (EXCEPT-BULK 2026-07-08)
                    logging.error('[EXC-0393] DB FWHM median fetch `pass` leaves `_ms_fwhm_fb` at config default instead of draft QC ...: %s', exc)
                    log_event(f"MASTERSTAR: DB vyber kandidatov zlyhal ({exc!s}).")
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
                    f"MASTERSTAR disk fallback: {len(disk_batch)} kandidatov z disku (bez platneho QC vyberu)."
                )

        if only_ms_paths is None:
            raise FileNotFoundError(
                f"MASTERSTAR: v {detrended_root} nie su ziadne FITS pre vyber ani po UI/DB."
            )

        _cfg_stack = app_config or AppConfig()
        try:
            _ms_fwhm_fb = float(_cfg_stack.sips_dao_fwhm_px)
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

        # Build MASTERSTAR with best-of-N fallback: try a few top candidates if build/selection is brittle.
        try:
            _best_n = int(float(_cfg_stack.masterstar_best_of_n))
        except (TypeError, ValueError):
            _best_n = 10
        _best_n = max(1, min(25, int(_best_n)))
        _cand_all = [Path(p) for p in (only_ms_paths or []) if Path(p).is_file()]
        # If UI/DB mapping yields too few candidates, expand from disk for best-of-N robustness -
        # but never when the user explicitly passed ``masterstar_candidate_paths`` (would replace e.g.
        # a single chosen frame with unrelated lights and pick lowest VY_FWHM among them).
        try:
            if not explicit_ui_masterstar_paths and len(_cand_all) < max(2, _best_n):
                _disk_more = _disk_stack_fallback_paths(Path(detrended_root), max_frames=max(8, _best_n * 2))
                for p in _disk_more:
                    if p not in _cand_all and p.is_file():
                        _cand_all.append(p)
        except Exception:  # noqa: BLE001
            pass
        if not _cand_all:
            raise FileNotFoundError(f"MASTERSTAR: v {detrended_root} nie su ziadne FITS pre vyber.")
        _cand_singletons = _cand_all[:_best_n]

        _db_ms_build: VyvarDatabase | None = None
        if draft_id is not None:
            _db_ms_build = _vyvar_open_database(_cfg_stack)
        try:
            last_exc: Exception | None = None
            info = {}
            # Try pool first (as before), then single best-of-N frames.
            attempt_lists: list[tuple[str, list[Path]]] = [("pool", _cand_all)]
            for i, p in enumerate(_cand_singletons, start=1):
                attempt_lists.append((f"single_{i:02d}_of_{len(_cand_singletons):02d}", [p]))
            for label, paths_try in attempt_lists:
                try:
                    log_event(f"MASTERSTAR build attempt: {label} (n={len(paths_try)})")
                    info = build_masterstar_from_detrended(
                        detrended_root=detrended_root,
                        output_fits=masterstar_fits,
                        only_paths=paths_try,
                        fwhm_fallback_px=float(_ms_fwhm_fb),
                        app_config=_cfg_stack,
                        draft_id=draft_id,
                        db=_db_ms_build,
                        pre_calibrated=_pre_cal_ms,
                    )
                    # Update selection metadata for traceability.
                    ms_selection_meta = dict(ms_selection_meta or {})
                    ms_selection_meta["best_of_n"] = int(_best_n)
                    ms_selection_meta["build_attempt"] = str(label)
                    ms_selection_meta["build_only_paths"] = [str(p.name) for p in paths_try]
                    _bfpx = info.get("best_frame_fwhm_px")
                    if _bfpx is not None:
                        try:
                            _bfv = float(_bfpx)
                            if math.isfinite(_bfv) and 0.5 < _bfv < 80.0:
                                ms_selection_meta["best_frame_fwhm_px"] = float(_bfv)
                        except (TypeError, ValueError):
                            pass
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    log_event(f"MASTERSTAR build attempt failed: {label}: {exc!s}")
                    continue
            if last_exc is not None:
                raise last_exc
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
            f"MASTERSTAR (len FITS, bez plate-solve): zapisane {_ms_out} | "
            f"zkombinovanych snimok={info.get('frames_combined', info.get('frames_used', '?'))}"
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
                    _db_ms.set_obs_draft_masterstar_fits_path(int(draft_id), _ms_out)
                finally:
                    _db_ms.conn.close()
        except Exception as exc:  # noqa: BLE001
            out_fast["masterstar_path_store_error"] = str(exc)
        return out_fast

    # Solve WCS (MASTERSTAR): vyhradne VYVAR lokalny Gaia solver (ziadny ASTAP / astrometry.net).

    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)

    _cfg_ms = app_config or AppConfig()

    try:
        _dao_sigma_eff = float(_cfg_ms.masterstar_dao_threshold_sigma)
    except (TypeError, ValueError):
        _dao_sigma_eff = 1.8
    if not math.isfinite(_dao_sigma_eff) or _dao_sigma_eff <= 0:
        _dao_sigma_eff = 1.8
    _dao_sigma_eff = max(0.1, min(6.0, float(_dao_sigma_eff)))
    log_event(
        f"MASTERSTAR: DAO threshold sigmaxRMS = {_dao_sigma_eff:.2f} "
        f"(config masterstar_dao_threshold_sigma; plate solve + katalog)"
    )

    _full_db = str(_cfg_ms.gaia_db_path or "").strip()
    if not _full_db:
        raise RuntimeError(
            "MASTERSTAR: v Settings nastavte gaia_db_path (plna lokalna Gaia DR3 SQLite DB)."
        )
    from vyvar_platesolver import solve_wcs_with_local_gaia

    log_event("MASTERSTAR WCS: VYVAR solver + plna Gaia DB (gaia_db_path).")
    try:
        _sip_ms = int(_cfg_ms.masterstar_platesolve_sip_max_order)
    except (TypeError, ValueError):
        _sip_ms = 5
    _sip_ms = max(2, min(5, _sip_ms))
    try:
        _sip_lo = int(_cfg_ms.masterstar_platesolve_sip_min_order)
    except (TypeError, ValueError):
        _sip_lo = 3
    _sip_lo = max(2, min(5, _sip_lo))
    if _sip_lo > _sip_ms:
        _sip_lo = _sip_ms
    log_event(
        f"MASTERSTAR: SIP skusanie {_sip_ms}->...->{_sip_lo} (config max/min plate-solve SIP)."
    )
    try:
        _xb_ms, _yb_ms = fits_binning_xy_from_header(hdr)
        _bin_ms = max(1, int(_xb_ms), int(_yb_ms))
    except Exception:  # noqa: BLE001
        _bin_ms = 1

    _auto_scale_ms: float | None = None
    _db_scale = _vyvar_open_database(_cfg_ms)
    _eq_ms: int | None = int(equipment_id) if equipment_id is not None else None
    _tel_ms: int | None = None
    if _db_scale is not None:
        try:
            _eq_ms, _tel_ms = resolve_optics_ids_for_platesolve(
                _db_scale, draft_id, equipment_id=equipment_id, telescope_id=telescope_id
            )
            _auto_scale_ms = compute_plate_scale_from_db(
                _eq_ms, _tel_ms, _db_scale.conn, binning=_bin_ms
            )
        except Exception:  # noqa: BLE001
            _auto_scale_ms = None
        finally:
            try:
                _db_scale.conn.close()
            except Exception:  # noqa: BLE001
                pass

    if _auto_scale_ms is not None:
        log_event(
            f"INFO: Plate scale z DB (Equipment+Telescope): {_auto_scale_ms:.4f} arcsec/px"
        )
    else:
        log_event(
            "WARNING: Plate scale z DB nedostupna - solver odvodi mierku z FITS alebo None"
        )

    _plate_scale_ms = _auto_scale_ms or None
    # Pull more complete optics hints (focal + effective pixel) from DB/FITS.
    # This is critical when FITS headers lack FOCALLEN/PIXSIZE and the solver would otherwise
    # overestimate FOV / cone radius and fail triangle matching.
    _bundle = _plate_solve_input_bundle(
        Path(masterstar_fits),
        app_config=_cfg_ms,
        equipment_id=_eq_ms,
        draft_id=int(draft_id) if draft_id is not None else None,
        telescope_id=_tel_ms if _tel_ms is not None else telescope_id,
    )
    _eff_um = _bundle.get("eff_um")
    _foc_mm = _bundle.get("focal_mm")
    _expected_bundle = _bundle.get("expected_arcsec_per_px")
    # D1/S6: a FITS/config/UI scale must not overwrite Equipment+Telescope DB
    # scale. On 520 g_60_4 the bundle used 200 mm / 15.511 "/px (Zeiss-wide
    # default) while the AZ800 row is 0.566 "/px; the triangle filter then
    # rejected every match. First auto-scale from DB wins.
    try:
        _bundle_scale = (
            float(_expected_bundle)
            if _expected_bundle is not None
            and math.isfinite(float(_expected_bundle))
            and float(_expected_bundle) > 0
            else None
        )
    except (TypeError, ValueError):
        _bundle_scale = None
    if _auto_scale_ms is not None:
        _plate_scale_ms = float(_auto_scale_ms)
        if (
            _bundle_scale is not None
            and abs(float(_bundle_scale) - float(_auto_scale_ms)) / float(_auto_scale_ms) > 0.05
        ):
            log_event(
                f"WARNING: MASTERSTAR plate-scale from FITS/config/UI "
                f"({_bundle_scale:.4f} arcsec/px) disagrees with DB Equipment+Telescope "
                f"({_auto_scale_ms:.4f} arcsec/px) - keeping DB scale for the triangle filter."
            )
    elif _bundle_scale is not None:
        _plate_scale_ms = float(_bundle_scale)

    _skip_independent_solve = bool(masterstar_platesolve_skip_solve) or (
        str(hdr.get("VY_CRT", "")).strip().lower() == "sibling_recovered"
        and _has_valid_wcs(hdr)
    )
    solve_meta: dict[str, Any] = {}
    if _skip_independent_solve:
        log_event(
            "MASTERSTAR: sibling-recovered WCS on disk - skipping independent Pass-1 plate-solve."
        )
        try:
            _vy_sodd = int(hdr.get("VY_SODD", 0) or 0)
        except (TypeError, ValueError):
            _vy_sodd = 0
        solve_meta = {
            "solved": True,
            "method": "sibling_recovered",
            "match_rate": 1.0,
            "sip_meta": {
                "masterstar_verified": True,
                "route": "sibling_recovered",
                "n_matched_tight": _vy_sodd,
            },
        }

    if not _skip_independent_solve:

        _mra, _mde, _ = _pointing_hint_from_header(hdr)
        if hint_ra_deg is not None and hint_dec_deg is not None:
            try:
                _hra_ov = float(hint_ra_deg)
                _hde_ov = float(hint_dec_deg)
                if math.isfinite(_hra_ov) and math.isfinite(_hde_ov):
                    _mra, _mde = _hra_ov, _hde_ov
                    log_event(
                        "MASTERSTAR: hint_ra_deg / hint_dec_deg z volania prepisuju hint z FITS "
                        "(druhy MASTERSTAR / detrended aligned)."
                    )
            except (TypeError, ValueError):
                pass
        try:
            _hint_sep_thr = float(_MASTERSTAR_SOLVER_USE_DRAFT_MEDIAN_IF_HINT_SEP_DEG)
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
                                "MASTERSTAR solve: pouzivam median RA/Dec z manifest files[] (hlavicka bez spolahliveho hintu)."
                            )
                        else:
                            sc_h = SkyCoord(ra=float(_mra) * u.deg, dec=float(_mde) * u.deg, frame="icrs")
                            sc_d = SkyCoord(ra=float(med_ra) * u.deg, dec=float(med_de) * u.deg, frame="icrs")
                            sep = float(sc_h.separation(sc_d).deg)
                            if sep > float(_hint_sep_thr):
                                log_event(
                                    f"MASTERSTAR solve: hint vs draft median = {sep:.3f} deg > {_hint_sep_thr} deg "
                                    "- pouzivam draft median z manifest files[]."
                                )
                                _mra, _mde = med_ra, med_de
                            elif sep > 0.05:
                                log_event(
                                    f"MASTERSTAR solve: hint vs draft median = {sep:.3f} deg (skontrolujte pointing)."
                                )
                finally:
                    try:
                        _dbc_hint.conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        _fov_ms_solve = resolve_plate_solve_fov_deg_hint(
            hdr,
            int(data.shape[0]),
            int(data.shape[1]),
            database_path=_cfg_ms.database_path,
            equipment_id=_eq_ms,
            draft_id=int(draft_id) if draft_id is not None else None,
        )
        if _fov_ms_solve is None:
            try:
                _pf_ms = float(plate_solve_fov_deg)
                if math.isfinite(_pf_ms) and _pf_ms > 0:
                    _fov_ms_solve = _pf_ms
            except (TypeError, ValueError):
                pass
        if _fov_ms_solve is None:
            _fov_ms_solve = float(_cfg_ms.plate_solve_fov_deg)
        _prms = _MASTERSTAR_PLATESOLVE_PREWRITE_RMS_MAX_PX
        _prms_r = _MASTERSTAR_PLATESOLVE_PREWRITE_RELAXED_RMS_MAX_PX
        _nnrms = _MASTERSTAR_PLATESOLVE_NN_REFINE_MAX_RMS_PX
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
                effective_pixel_um=float(_eff_um) if _eff_um is not None else None,
                focal_length_mm=float(_foc_mm) if _foc_mm is not None else None,
                expected_plate_scale_arcsec_per_px=(
                    float(_plate_scale_ms) if _plate_scale_ms is not None else None
                ),
                masterstar_prewrite_rms_max_px=float(_prms) if _prms is not None else None,
                masterstar_prewrite_relaxed_rms_max_px=float(_prms_r) if _prms_r is not None else None,
                masterstar_nn_refine_max_rms_px=float(_nnrms) if _nnrms is not None else None,
                masterstar_sip_min_order=int(_sip_lo),
                app_config=_cfg_ms,
                solver_use_cone_for_sip=True,
                solver_fits_header_hint_sep_escape=True,
                solver_legacy_masterstar_mirror_sweep=True,
                solver_apply_roworder_yflip=False,
            )

        solve_meta = _run_masterstar_vyvar_solve(
            enable_sip=True,
            sip_max_order=int(_sip_ms),
            fov_deg=float(_fov_ms_solve),
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
                "MASTERSTAR: po plate-solve chyba platny WCS. Skontroluj gaia_db_path, RA/Dec a mierku v hlavicke "
                "(FOCALLEN/PIXSIZE alebo SECPIX) a vystup solvera."
            )

        # Pipeline-level acceptance criteria (stricter than solver's minimal guard):
        # - match_rate: allow 60% on the first solve (optimizer refines later)
        try:
            _mr = float(solve_meta.get("match_rate", 0.0) or 0.0)
        except (TypeError, ValueError):
            _mr = 0.0
        from vyvar_platesolver import MASTERSTAR_PLATESOLVE_MIN_MATCH_RATE

        _min_mr = float(MASTERSTAR_PLATESOLVE_MIN_MATCH_RATE)
        if _mr < _min_mr:
            raise RuntimeError(
                f"MASTERSTAR plate-solve zamietnuty: match_rate={_mr * 100.0:.1f}% < {_min_mr * 100.0:.0f}%. "
                "Skus zvysit n_stack alebo upravit hint/DAO prahy."
            )

        try:
            _aniso_thr = float(_PLATESOLVE_ANISOTROPY_THRESHOLD)
        except (TypeError, ValueError):
            _aniso_thr = 1.3
        if not math.isfinite(_aniso_thr) or _aniso_thr <= 0:
            _aniso_thr = 1.3
        _aniso_thr = max(1.01, min(5.0, float(_aniso_thr)))

        # Post-solve anisotropy validation: reject strongly anisotropic pixel scale and retry solver once.
        try:
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
                f"VAROVANIE: Anizotropna mierka ratio={scale_ratio:.2f} - plate-solve zamietnuty, restartujem solver (relaxed)."
            )
            # Retry with relaxed knobs:
            # - slightly larger FOV diameter (hint-vs-solved tolerance),
            # - more Gaia rows,
            # - no SIP (simpler model can be more stable when the fit goes off-rails).
            solve_meta2 = _run_masterstar_vyvar_solve(
                enable_sip=False,
                sip_max_order=0,
                fov_deg=float(_fov_ms_solve) * 1.25,
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
                raise RuntimeError("MASTERSTAR: po retry plate-solve chyba platny WCS.")
            try:
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
                    f"MASTERSTAR plate-solve zamietnuty: anizotropna mierka po retry ratio={scale_ratio2:.2f} (>{_aniso_thr})."
                )

    _exp_scale_apx: float | None = None
    if _plate_scale_ms is not None:
        try:
            _ea2 = float(_plate_scale_ms)
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
        # derive-or-None (DR6 pattern): all principled sources exhausted - do not guess.
        _exp_scale_apx = None
        log_event(
            "WARNING: MASTERSTAR plate scale not derivable (DB/FITS/WCS exhausted) - "
            "expected scale unknown; VY_PLTS will not be written."
        )

    _wcs_ok = False
    try:
        with fits.open(masterstar_fits, memmap=False) as _hd_wq:
            _w_check = WCS(_hd_wq[0].header)
        if _exp_scale_apx is None:
            log_event(
                "MASTERSTAR WCS quality: expected plate scale unknown "
                "(no DB/FITS/WCS/config) - check skipped."
            )
            _wcs_q = None
        else:
            _wcs_q = masterstar_wcs_quality(
                _w_check, float(_exp_scale_apx), anisotropy_limit=float(_aniso_thr)
            )
            _wcs_ok = bool(_wcs_q.get("ok", False))
        if _wcs_q is not None and not _wcs_ok:
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
                f"MASTERSTAR WCS kvalita: zla (ratio={_rq_s}, scale_err={_se_s}%) - "
                "pokracujem bez externeho plate-solve (ocakava sa FITS metadata / buduci blind solver)."
            )
    except Exception as _wq_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR WCS check failed: {_wq_exc}")
        _wcs_ok = False

    try:
        _pscale_adj = _try_rescale_masterstar_linear_wcs_to_expected_plate_scale(
            masterstar_fits,
            app_config=app_config or AppConfig(),
            equipment_id=equipment_id,
            draft_id=draft_id,
        )
    except Exception as exc:  # noqa: BLE001
        log_event(f"WCS PLATE SCALE: neocakavana chyba - {exc!s}")
        _pscale_adj = {"rescaled": False, "error": str(exc)}
    solve_meta["wcs_plate_scale_adjustment"] = _pscale_adj

    # Write calibrated plate scale to MASTERSTAR header
    _vy_plts = None
    try:
        if isinstance(solve_meta, dict):
            _vy_plts = solve_meta.get("plate_scale_arcsec_px")
        if _vy_plts is None and isinstance(_pscale_adj, dict):
            _vy_plts = _pscale_adj.get("new_scale_arcsec_per_px") or _pscale_adj.get(
                "expected_arcsec_per_px"
            )
        if _vy_plts is None:
            _vy_plts = _exp_scale_apx
    except Exception:  # noqa: BLE001
        _vy_plts = None

    if _vy_plts is not None:
        try:
            _vy_plts_f = float(_vy_plts)
            if math.isfinite(_vy_plts_f) and _vy_plts_f > 0:
                with fits.open(masterstar_fits, mode="update") as hdul:
                    hdul[0].header["VY_PLTS"] = (
                        _vy_plts_f,
                        "VYVAR plate scale arcsec/px",
                    )
                    hdul.flush()
                log_event(f"VY_PLTS={_vy_plts_f:.4f} written to MASTERSTAR.fits")
        except Exception as exc:  # noqa: BLE001
            log_event(f"Could not write VY_PLTS to MASTERSTAR: {exc}")
    else:
        log_event(
            "WARNING: MASTERSTAR VY_PLTS not written - plate scale not derivable "
            "(derive-or-None; no rig/global constant written to header)."
        )

    try:
        from wcs_invertibility import evaluate_wcs_roundtrip

        _nax1 = int(hdr.get("NAXIS1") or (data.shape[1] if data.ndim >= 2 else 0))
        _nax2 = int(hdr.get("NAXIS2") or (data.shape[0] if data.ndim >= 1 else 0))
        if _has_valid_wcs(hdr) and _nax1 > 0 and _nax2 > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                _w_rt = WCS(hdr)
            _rt0 = evaluate_wcs_roundtrip(_w_rt, naxis1=_nax1, naxis2=_nax2)
            if isinstance(solve_meta, dict):
                solve_meta["wcs_roundtrip_p99_px"] = _rt0.get("wcs_roundtrip_p99_px")
                solve_meta["wcs_roundtrip_pass"] = bool(_rt0.get("pass"))
            if not _rt0.get("pass"):
                log_event(
                    f"WARNING: MASTERSTAR initial-solve WCS round-trip p99="
                    f"{_rt0.get('wcs_roundtrip_p99_px'):.4f}px (threshold "
                    f"{_rt0.get('p99_threshold_px')}px) - provenance flag set; continuing."
                )
            else:
                log_event(
                    f"MASTERSTAR WCS round-trip PASS (initial solve): p99="
                    f"{_rt0.get('wcs_roundtrip_p99_px'):.4f}px"
                )
    except Exception as _rt_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR WCS round-trip check skipped: {_rt_exc!s}")

    if masterstar_platesolve_only:
        _cfg_early = app_config or AppConfig()
        try:
            _ms_out_early = str(Path(masterstar_fits).resolve())
        except OSError:
            _ms_out_early = str(masterstar_fits)
        log_event(
            f"ONLY MASTER (test): plate-solve + uprava mierky WCS hotove -> {_ms_out_early} "
            "(preskakujem DAO export, masterstars CSV, fotometricky plan, MASTER_SOURCES)."
        )
        out_ps: dict[str, Any] = {
            "masterstar_fits": _ms_out_early,
            "masterstars_csv": "",
            "frames_used": int(info.get("frames_used", 0)),
            "masterstar_selection": ms_selection_meta or None,
            "masterstar_build_info": info,
            "n_raw_dao": 0,
            "detected_stars": 0,
            "catalog_matched": 0,
            "catalog_rows": 0,
            "catalog_match_max_sep_arcsec": float(_match_sep_eff),
            "solve": solve_meta,
            "masterstar_platesolve_only": True,
            "comparison_stars_csv": "",
            "variable_targets_csv": "",
            "photometry_plan_json": "",
        }
        try:
            if draft_id is not None:
                _db_early = VyvarDatabase(Path(_cfg_early.database_path))
                try:
                    _db_early.set_obs_draft_masterstar_fits_path(int(draft_id), _ms_out_early)
                finally:
                    _db_early.conn.close()
        except Exception as exc:  # noqa: BLE001
            out_ps["masterstar_path_store_error"] = str(exc)
        return out_ps

    # _cfg_ms / _dao_sigma_eff uz vyssie (rovnake DAO sigma pre plate solve aj katalog).
    _ms_fwhm = float(_cfg_ms.sips_dao_fwhm_px)
    if not math.isfinite(_ms_fwhm) or _ms_fwhm <= 0:
        _ms_fwhm = 2.5
    _ms_meta = ms_selection_meta if isinstance(ms_selection_meta, dict) else {}
    _best_fwhm = _ms_meta.get("best_frame_fwhm_px")
    try:
        _best_fwhm_f = float(_best_fwhm) if _best_fwhm is not None else float("nan")
    except (TypeError, ValueError):
        _best_fwhm_f = float("nan")
    _use_best_frame_fwhm = bool(_cfg_ms.masterstar_use_best_frame_fwhm)
    if (
        _use_best_frame_fwhm
        and math.isfinite(_best_fwhm_f)
        and 1.2 <= _best_fwhm_f <= 20.0
    ):
        dao_fwhm_px_for_ms = float(_best_fwhm_f)
        _dao_fwhm_bypass_hdr = True
        log_event(
            f"MASTERSTAR DAO: dao_fwhm_px={dao_fwhm_px_for_ms:.3f} from best_frame_fwhm_px "
            f"(header VY_FWHM median ignored)"
        )
    else:
        dao_fwhm_px_for_ms = float(_ms_fwhm)
        _dao_fwhm_bypass_hdr = False
        log_event(
            f"MASTERSTAR DAO: dao_fwhm_px={dao_fwhm_px_for_ms:.3f} from sips_dao_fwhm_px / header VY_FWHM"
        )

    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)
    data = np.ascontiguousarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # Global median (BOJ) background subtraction is intentionally OFF (handled downstream).
    _ms_mean = float(np.nanmean(data))
    log_event("MASTERSTAR: globalne odcitanie medianu (BOJ) vypnute.")
    log_event(f"MASTERSTAR po nan_to_num: mean={_ms_mean:.6f}")
    _ms_min = float(np.nanmin(data))
    _ms_max = float(np.nanmax(data))
    log_event(f"MASTERSTAR levels: noise_floor(min)={_ms_min:.2f}, saturation_proxy(max)={_ms_max:.2f}")

    if platesolve_dir is None:
        raise ValueError("generate_masterstar_and_catalog: platesolve_dir is required (got None).")
    platesolve_dir.mkdir(parents=True, exist_ok=True)
    _fov_job: float | None = None
    try:
        _fj = float(plate_solve_fov_deg)
        if math.isfinite(_fj) and _fj > 0:
            _fov_job = _fj
    except (TypeError, ValueError):
        _fov_job = None
    if _fov_job is None:
        _fov_job = resolve_plate_solve_fov_deg_hint(
            hdr,
            int(data.shape[0]),
            int(data.shape[1]),
            database_path=_cfg_ms.database_path,
            equipment_id=int(equipment_id) if equipment_id is not None else None,
            draft_id=int(draft_id) if draft_id is not None else None,
        )
    if _fov_job is None:
        _fov_job = float(_cfg_ms.plate_solve_fov_deg)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _w_pre = WCS(hdr)
        if _w_pre.has_celestial:
            _, _r_cat_need = _effective_field_catalog_cone_radius_deg(
                _w_pre, int(data.shape[0]), int(data.shape[1]), float(_fov_job), fits_header=hdr
            )
            _invalidate_field_catalog_cone_cache_if_needed(
                platesolve_dir / "field_catalog_cone.csv",
                plate_solve_fov_deg=float(_fov_job),
                effective_radius_deg=float(_r_cat_need),
            )
    except Exception as exc:  # noqa: BLE001
        log_event(f"Katalog: kontrola cache field_catalog_cone preskocena - {exc!s}")

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
        dao_fwhm_px=dao_fwhm_px_for_ms,
        equipment_saturate_adu=equipment_saturate_adu,
        catalog_local_gaia_only=catalog_local_gaia_only,
        plate_solve_fov_deg=float(_fov_job),
        fov_database_path=_cfg_ms.database_path,
        fov_equipment_id=int(equipment_id) if equipment_id is not None else None,
        fov_draft_id=int(draft_id) if draft_id is not None else None,
        prematch_peak_sigma_floor=float(
            _cfg_ms.masterstar_prematch_peak_sigma_floor
        ),
        dao_fwhm_bypass_header=bool(_dao_fwhm_bypass_hdr),
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
                    f"MASTERSTAR: VYVAR pary ({len(_px)}) zlucene do katalogu "
                    f"(mirror={_mir or 'native'}, pre astrometry optimizer)."
                )
    except Exception as exc:  # noqa: BLE001
        log_event(f"MASTERSTAR: zlucenie VYVAR parov preskocene - {exc!s}")

    _fwhm_dao = float(det_meta.get("dao_fwhm_px") or 0.0)
    if not math.isfinite(_fwhm_dao) or _fwhm_dao <= 0:
        _fwhm_dao = float((det_meta.get("identity_gate") or {}).get("fwhm_px") or 1.25)

    try:
        from wcs_invertibility import (
            accumulate_identity_gate,
            apply_post_match_identity_gate_df,
            gaia_radec_map_from_table,
        )

        _gmap_ms: dict[str, tuple[float, float]] = {}
        _cone_p = Path(platesolve_dir) / "field_catalog_cone.csv"
        if _cone_p.is_file():
            _cone_df = pd.read_csv(_cone_p, low_memory=False, dtype={"catalog_id": str, "source_id": str})
            _gmap_ms.update(gaia_radec_map_from_table(_cone_df))
        if isinstance(solve_meta, dict):
            _pids = solve_meta.get("pairs_catalog_id") or []
            _pra = solve_meta.get("pairs_ra") or []
            _pde = solve_meta.get("pairs_de") or []
            from gaia_catalog_id import normalize_gaia_source_id as _norm_gid

            for _i, _pid in enumerate(_pids):
                _k = _norm_gid(str(_pid))
                if _k and _i < len(_pra) and _i < len(_pde):
                    _gmap_ms[_k] = (float(_pra[_i]), float(_pde[_i]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _w_gate = WCS(hdr)
        df_out, _idc_merge = apply_post_match_identity_gate_df(
            df_out,
            _w_gate,
            gaia_ra_dec_by_cid=_gmap_ms,
            fwhm_px=_fwhm_dao,
            log_fn=log_event,
        )
        _acc = dict(det_meta.get("identity_gate") or {})
        _n_out = int(
            df_out.get("catalog_id", pd.Series([""] * len(df_out)))
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )
        det_meta["identity_gate"] = accumulate_identity_gate(_acc, _idc_merge, _n_out)
        det_meta["identity_gate"]["fwhm_px"] = float(_fwhm_dao)
    except Exception as _mg_exc:  # noqa: BLE001
        log_event(f"post_match_identity_gate (post-merge) skipped: {_mg_exc!s}")

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
    _cat_rows = int(det_meta.get("catalog_rows", 0) or 0)
    if "catalog_id" in df_out.columns:
        _cid_raw = df_out["catalog_id"].fillna("").astype(str).str.strip()
        _n_gaia_det_raw = int(_cid_raw[_cid_raw != ""].nunique())
    else:
        _n_gaia_det_raw = int(_n_mat_raw)
    _gaia_rate_raw = (100.0 * float(_n_gaia_det_raw) / float(_cat_rows)) if _cat_rows > 0 else 0.0
    log_event(
        f"[chart] MATCH STATS (raw): Found {_n_det_raw} stars on image | {_n_mat_raw} matched with Gaia | "
        f"Match Rate: {_rate_raw:.2f}% | Gaia->DAO: {_gaia_rate_raw:.2f}% ({_n_gaia_det_raw}/{_cat_rows})"
    )
    if _cat_rows > 0:
        LOGGER.info(
            "[MASTERSTAR] Gaia->DAO completeness: "
            "%d/%d (%.1f%%) | catalog_only: %d",
            _n_gaia_det_raw,
            _cat_rows,
            _gaia_rate_raw,
            _cat_rows - _n_gaia_det_raw,
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
    _opt_stats_last: dict[str, Any] = {}
    try:
        from astrometry_optimizer import optimize_masterstar_matches

        _gdb_opt = str(_cfg_ms.gaia_db_path or "").strip()
        if _gdb_opt:
            _mir_extra = bool(_MASTERSTAR_OPTIMIZER_MIRROR_EXTRA_LOG)
            _idg_n_out = int((det_meta.get("identity_gate") or {}).get("n_matched_out") or 0)
            try:
                with fits.open(masterstar_fits, mode="update", memmap=False) as _hf_dao:
                    _hf_dao[0].header["VY_FWHM_DAO"] = (
                        float(_fwhm_dao),
                        "DAO-domain FWHM [pix] for identity gate",
                    )
                    _hf_dao.flush()
            except Exception as _dao_hdr_exc:  # noqa: BLE001
                log_event(f"MASTERSTAR VY_FWHM_DAO stamp skipped: {_dao_hdr_exc!s}")
            csv_path = optimize_masterstar_matches(
                masterstars_csv=temp_csv,
                masterstar_fits=masterstar_fits,
                gaia_db_path=_gdb_opt,
                output_csv=csv_path,
                gaia_mag_limit=float(_ms_faintest_mag_eff),
                gaia_max_catalog_rows=int(_ms_max_catalog_rows_eff),
                mirror_orientation_extra_log=_mir_extra,
                sip_force_rms_guard_ratio=_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO,
                fwhm_dao_px=float(_fwhm_dao),
                identity_gate_n_out=_idg_n_out,
                stats_out=_opt_stats_last,
            )
            # Force one more pass after WCS displacement update for final edge recovery.
            # Identity-count contract is first-entry only; rematch may add honest pairs.
            csv_path = optimize_masterstar_matches(
                masterstars_csv=csv_path,
                masterstar_fits=masterstar_fits,
                gaia_db_path=_gdb_opt,
                output_csv=csv_path,
                gaia_mag_limit=float(_ms_faintest_mag_eff),
                gaia_max_catalog_rows=int(_ms_max_catalog_rows_eff),
                mirror_orientation_extra_log=_mir_extra,
                sip_force_rms_guard_ratio=_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO,
                fwhm_dao_px=float(_fwhm_dao),
                identity_gate_n_out=None,
                stats_out=_opt_stats_last,
            )
            log_event("MASTERSTAR optimizer: forced final re-match pass completed.")
            # Final safety: repair any residual precision-loss IDs in masterstars_full_match.csv via Gaia RA/DEC lookup.
            try:
                from repair_catalog_ids import repair_csv_catalog_ids_from_gaia_db  # noqa: PLC0415

                rep = repair_csv_catalog_ids_from_gaia_db(
                    csv_path=Path(csv_path),
                    gaia_db_path=Path(_gdb_opt),
                    id_col="catalog_id",
                    backup=True,
                    max_sep_arcsec=2.0,
                    log_fn=log_event,
                    skip_unmatched_placeholders=True,
                )
                if int(rep.get("repaired") or 0) > 0:
                    log_event(
                        f"MASTERSTAR repair: repaired={rep.get('repaired')} warnings={rep.get('warnings')} ({Path(csv_path).name})"
                    )
            except Exception as _rep_exc:  # noqa: BLE001
                log_event(f"MASTERSTAR repair skipped: {_rep_exc!s}")
        else:
            _vyvar_df_to_csv(df_out, csv_path)
    except Exception as exc:  # noqa: BLE001
        from invariants_runtime import InvariantViolation as _InvMatchId  # noqa: PLC0415

        if isinstance(exc, _InvMatchId):
            raise
        log_event(f"MASTERSTAR optimizer skipped/fallback: {exc!s}")
        _vyvar_df_to_csv(df_out, csv_path)
    try:
        # Critical: keep Gaia IDs as strings (avoid float/scientific precision loss).
        df_final = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str, "name": str})
        # Preserve DAO pass provenance through astrometry_optimizer CSV round-trip.
        for _prov_col in ("vy_dao_pass", "ambiguous_owner"):
            if _prov_col not in df_out.columns:
                continue
            _cid_out = df_out.get("catalog_id", pd.Series([""] * len(df_out))).map(
                lambda c: str(c).strip()
            )
            _map = df_out.assign(_cid=_cid_out).drop_duplicates("_cid", keep="last").set_index("_cid")[
                _prov_col
            ]
            _cid_fin = df_final.get("catalog_id", pd.Series([""] * len(df_final))).map(
                lambda c: str(c).strip()
            )
            df_final[_prov_col] = _cid_fin.map(_map)
            if _prov_col == "vy_dao_pass":
                df_final[_prov_col] = pd.to_numeric(df_final[_prov_col], errors="coerce").fillna(1)
            elif _prov_col == "ambiguous_owner":
                df_final[_prov_col] = df_final[_prov_col].fillna(False).astype(bool)
        if len(df_final) == len(df_out):
            for _idcol in ("vy_identity_gate", "gaia_dao_resid_px"):
                if _idcol in df_out.columns and _idcol not in df_final.columns:
                    df_final[_idcol] = df_out[_idcol].to_numpy()
    except Exception as _df_final_exc:  # noqa: BLE001
        log_event(
            f"MASTERSTAR: re-read of {Path(csv_path).name} failed ({_df_final_exc!s}); "
            "using in-memory df_out and re-asserting catalog_id/name as str."
        )
        df_final = df_out.copy()
        # df_out.copy() can carry catalog_id/name as non-string dtypes -> re-assert to avoid
        # reintroducing float/scientific precision loss on Gaia IDs downstream.
        for _idcol in ("catalog_id", "name"):
            if _idcol in df_final.columns:
                df_final[_idcol] = df_final[_idcol].astype(str)
    _wcs_rt_p99: float | None = None
    _wcs_rt_pass: bool | None = None
    _identity_qa: dict[str, Any] = {}
    try:
        from wcs_invertibility import (
            evaluate_matched_world2pix_identity_px,
            evaluate_wcs_roundtrip,
            finalize_masterstar_sky_coords,
        )

        with fits.open(masterstar_fits, memmap=False) as _hf:
            _hdr_fin = _hf[0].header
            _w_fin = WCS(_hdr_fin)
        df_final = finalize_masterstar_sky_coords(
            df_final,
            _w_fin,
            gaia_db_path=str(_cfg_ms.gaia_db_path or ""),
            log_fn=log_event,
        )
        _nax1f = int(_hdr_fin.get("NAXIS1") or 0)
        _nax2f = int(_hdr_fin.get("NAXIS2") or 0)
        _rt_fin = evaluate_wcs_roundtrip(_w_fin, naxis1=_nax1f, naxis2=_nax2f)
        _wcs_rt_p99 = _rt_fin.get("wcs_roundtrip_p99_px")
        _wcs_rt_pass = bool(_rt_fin.get("pass"))
        _identity_qa = evaluate_matched_world2pix_identity_px(
            df_final,
            _w_fin,
            gaia_db_path=str(_cfg_ms.gaia_db_path or ""),
            log_fn=log_event,
        )
        _p95 = _identity_qa.get("matched_world2pix_identity_p95_px")
        try:
            _p95f = float(_p95) if _p95 is not None else float("nan")
        except (TypeError, ValueError):
            _p95f = float("nan")
        # Standing series WARN (Anchor #3 / draft_435 baseline p95~1.54 px): soft threshold only.
        # INV-WCS-01: same band, recorded into pipeline_meta invariants at merge below.
        _IDENTITY_P95_WARN_PX = 2.0
        if math.isfinite(_p95f) and _p95f > _IDENTITY_P95_WARN_PX:
            logging.warning(
                "[IDENTITY-QA] matched_world2pix_identity_p95_px=%.3f exceeds WARN threshold %.1f px "
                "(series baseline draft_435 p95~1.54; no FAIL)",
                _p95f,
                _IDENTITY_P95_WARN_PX,
            )
            log_event(
                f"IDENTITY-QA WARN: p95={_p95f:.3f} px > {_IDENTITY_P95_WARN_PX:.1f} px threshold"
            )
        try:
            from invariants_runtime import check_wcs_identity_p95  # noqa: PLC0415
            from invariants_runtime import inv_check  # noqa: PLC0415

            _ok_w, _det_w = check_wcs_identity_p95(_p95f if math.isfinite(_p95f) else None)
            _inv_meta_wcs: dict = {"invariants": []}
            inv_check(_inv_meta_wcs, "INV-WCS-01", _ok_w, policy="WARN", detail=_det_w)
            _identity_qa = dict(_identity_qa or {})
            _identity_qa["_inv_wcs_01"] = _inv_meta_wcs.get("invariants") or []
        except Exception as _inv_wcs_exc:  # noqa: BLE001
            logging.debug("[INV-WCS-01] record skipped: %s", _inv_wcs_exc)
    except Exception as _fin_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR coordinate finalization / round-trip QA skipped: {_fin_exc!s}")
        _wcs_rt_p99 = None
        _wcs_rt_pass = None
        _identity_qa = {}
    # DAO-GAIA-ERA-01 M1: expand detection table to catalog-derived membership before zones/enrich.
    # INV-MS-EXPAND-01: when cone+WCS exist, expand must succeed or raise (no silent skip).
    _chip_ms: pd.DataFrame | None = None
    _membership_expand_meta: dict[str, Any] = {}
    _cone_gaia_pre = Path(platesolve_dir) / "field_catalog_cone.csv"
    _wcs_ok_pre = bool(_has_valid_wcs(hdr))
    if _cone_gaia_pre.is_file() and _wcs_ok_pre:
        from masterstar_gaia_accounting import (  # noqa: PLC0415
            expand_detection_to_catalog_membership,
            gaia_on_chip_from_cone,
        )
        from astropy.wcs import WCS as _WCS_expand  # noqa: PLC0415

        LOGGER.info(
            "[M1] catalog membership expand: cone=%s wcs_ok=%s n_ms_in=%d",
            True,
            True,
            int(len(df_final)),
        )
        _cone_df_pre = read_vyvar_csv(_cone_gaia_pre, low_memory=False, dtype={"catalog_id": str})
        _nax1_pre = int(hdr.get("NAXIS1") or 0)
        _nax2_pre = int(hdr.get("NAXIS2") or 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _wcs_pre = _WCS_expand(hdr)
        _ra_pre = pd.to_numeric(_cone_df_pre["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _de_pre = pd.to_numeric(_cone_df_pre["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _ok_pre = np.isfinite(_ra_pre) & np.isfinite(_de_pre)
        _gx_pre, _gy_pre = _wcs_pre.world_to_pixel_values(_ra_pre[_ok_pre], _de_pre[_ok_pre])
        _chip_ms = gaia_on_chip_from_cone(
            _cone_df_pre, gx=_gx_pre, gy=_gy_pre, ok_mask=_ok_pre, wpx=_nax1_pre, h=_nax2_pre
        )
        _membership_depth_g = float(
            getattr(_cfg_ms, "masterstar_gaia_census_target_depth_g", None) or 15.0
        )
        df_final, _membership_expand_meta = expand_detection_to_catalog_membership(
            df_final,
            _chip_ms,
            membership_depth_g=_membership_depth_g,
            wpx=_nax1_pre,
            h=_nax2_pre,
        )
        det_meta["catalog_derived_membership"] = dict(_membership_expand_meta)
        LOGGER.info(
            "[M1] catalog-derived membership: +%d Gaia rows (depth G<=%.1f), n_out=%d",
            int(_membership_expand_meta.get("n_catalog_rows_added", 0)),
            _membership_depth_g,
            int(_membership_expand_meta.get("n_rows_out", len(df_final))),
        )
        log_event(
            "MASTERSTAR catalog-derived membership: "
            f"+{int(_membership_expand_meta.get('n_catalog_rows_added', 0))} Gaia rows "
            f"(depth G<={_membership_depth_g:.1f}), "
            f"n_out={int(_membership_expand_meta.get('n_rows_out', len(df_final)))}"
        )
    elif _cone_gaia_pre.is_file() or _wcs_ok_pre:
        raise RuntimeError(
            "INV-MS-EXPAND-01: catalog membership expand blocked "
            f"(cone={_cone_gaia_pre.is_file()} wcs_ok={_wcs_ok_pre})"
        )
    # VSX stamp deferred until after write_photometry_plan_files (VT CSV created there).
    df_final = _annotate_masterstars_flux_zones(
        df_final,
        noise_floor_adu=det_meta.get("noise_floor_adu"),
        equipment_saturate_adu=equipment_saturate_adu,
        saturate_limit_adu_fallback=det_meta.get("saturate_limit_adu"),
        saturate_limit_fraction=float(_cfg_ms.saturate_limit_fraction),
        sigma_px=det_meta.get("bg_sigma_adu"),
        sky_median_adu=det_meta.get("sky_median_adu"),
        prematch_peak_sigma_floor=det_meta.get("prematch_peak_sigma_floor"),
        frame_max_adu=det_meta.get("frame_max_adu"),
        empirical_clip_adu=det_meta.get("empirical_clip_adu"),
        dao_detection_n_equiv=(
            det_meta.get("dao_detection_n_equiv")
            if det_meta.get("dao_detection_n_equiv") is not None
            else float(_cfg_ms.dao_detection_n_equiv)
        ),
    )
    _dao_class_meta: dict[str, Any] = {}
    _recon_ms: dict[str, Any] | None = None
    try:
        cid = df_final.get("catalog_id", pd.Series([""] * len(df_final))).fillna("").astype(str).str.strip()
        df_final["source_type"] = np.where(cid.ne(""), "GAIA_MATCHED", "DAO_ONLY")
        from masterstar_gaia_accounting import (  # noqa: PLC0415
            enrich_masterstar_gaia_complete,
            gaia_on_chip_from_cone,
            write_gaia_census_and_verify,
        )
        from astropy.wcs import WCS as _WCS_enrich  # noqa: PLC0415

        _cone_gaia = Path(platesolve_dir) / "field_catalog_cone.csv"
        if not _cone_gaia.is_file():
            raise RuntimeError("MASTERSTAR Gaia-complete enrich: missing field_catalog_cone.csv")
        if not _has_valid_wcs(hdr):
            raise RuntimeError("MASTERSTAR Gaia-complete enrich: MASTERSTAR WCS missing/invalid")
        _cone_df = read_vyvar_csv(_cone_gaia, low_memory=False, dtype={"catalog_id": str})
        _nax1_g = int(hdr.get("NAXIS1") or 0)
        _nax2_g = int(hdr.get("NAXIS2") or 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _wcs_g = _WCS_enrich(hdr)
        _ra_g = pd.to_numeric(_cone_df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _de_g = pd.to_numeric(_cone_df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _ok_g = np.isfinite(_ra_g) & np.isfinite(_de_g)
        _gx, _gy = _wcs_g.world_to_pixel_values(_ra_g[_ok_g], _de_g[_ok_g])
        _chip = (
            _chip_ms
            if _chip_ms is not None and len(_chip_ms)
            else gaia_on_chip_from_cone(
                _cone_df, gx=_gx, gy=_gy, ok_mask=_ok_g, wpx=_nax1_g, h=_nax2_g
            )
        )
        with fits.open(masterstar_fits, memmap=False) as _hg:
            _raw_g = np.asarray(_hg[0].data, dtype=np.float32)
        _, _med_g, _ = plain_mean_med_std(_raw_g, sigma=3.0, maxiters=3)
        _data0_g = np.nan_to_num(
            (_raw_g - _med_g).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        _fwhm_g = float(det_meta.get("dao_fwhm_px") or header_core_fwhm_px(hdr) or 3.5)
        _membership_depth_g = float(
            getattr(_cfg_ms, "masterstar_gaia_census_target_depth_g", None) or 15.0
        )
        _census_depth_g = 17.5  # M1-amend: census accounting depth; G 15-17.5 census-only
        df_final, _gaia_census, _gaia_meta = enrich_masterstar_gaia_complete(
            df_final,
            data0=_data0_g,
            gaia_on_chip=_chip,
            cfg=_cfg_ms,
            wpx=_nax1_g,
            h=_nax2_g,
            fwhm_px=_fwhm_g,
            target_depth_g=_census_depth_g,
            sat_limit_adu=det_meta.get("saturate_limit_adu"),
            identity_lock_only=False,
            catalog_derived_membership=bool(_membership_expand_meta),
            tolerance_overrides=det_meta.get("dao_gaia_derived_tol"),
        )
        _census_inv = write_gaia_census_and_verify(
            _gaia_census,
            n_on_chip=len(_chip),
            census_path=Path(platesolve_dir) / "gaia_source_state_census.csv",
        )
        try:
            from dao_gaia_calibration import (  # noqa: PLC0415
                build_calibration_certificate,
                write_calibration_certificate,
            )

            _setup_nm = str(setup_name or platesolve_dir.name or "MASTERSTAR")
            _tol_d = det_meta.get("dao_gaia_derived_tol") or {}
            _dao_x_f = pd.to_numeric(df_final.get("x"), errors="coerce").to_numpy(dtype=np.float64)
            _dao_y_f = pd.to_numeric(df_final.get("y"), errors="coerce").to_numpy(dtype=np.float64)
            _cert = build_calibration_certificate(
                setup=_setup_nm,
                wcs_obj=_wcs_g,
                data0=_data0_g,
                dao_x=_dao_x_f,
                dao_y=_dao_y_f,
                gaia_x=pd.to_numeric(_chip.get("x_gaia"), errors="coerce").to_numpy(dtype=np.float64),
                gaia_y=pd.to_numeric(_chip.get("y_gaia"), errors="coerce").to_numpy(dtype=np.float64),
                gaia_g=pd.to_numeric(_chip.get("g_mag"), errors="coerce").to_numpy(dtype=np.float64),
                fwhm_px=float(_fwhm_g),
                pass1_sigma=float(_tol_d.get("pass1_sigma") or _cfg_ms.masterstar_dao_threshold_sigma),
                pass2_sigma=float(_tol_d.get("pass2_sigma") or _cfg_ms.masterstar_dao_pass2_sigma),
                seed_snr_min=float(_cfg_ms.masterstar_forced_seed_snr_min),
                target_depth_g=float(_census_depth_g),
                edge_margin_px=float(_cfg_ms.masterstar_gaia_census_edge_margin_px),
                cfg=_cfg_ms,
                ms_df=df_final,
                census_df=_gaia_census,
                repo_root=Path(__file__).resolve().parent.parent,
            )
            _cert_path = write_calibration_certificate(
                _cert, Path(platesolve_dir), fail_closed=True
            )
            if _membership_expand_meta:
                from masterstar_gaia_accounting import verify_ms_expand_guard  # noqa: PLC0415

                _ok_exp, _det_exp = verify_ms_expand_guard(
                    _membership_expand_meta,
                    census_path=Path(platesolve_dir) / "gaia_source_state_census.csv",
                    cert_path=_cert_path,
                )
                if not _ok_exp:
                    from invariants_runtime import InvariantViolation  # noqa: PLC0415

                    raise InvariantViolation("INV-MS-EXPAND-01", _det_exp)
                log_event(f"INV-MS-EXPAND-01 PASS: {_det_exp}")
            det_meta["dao_gaia_calibration"] = _cert.to_dict()
            det_meta["dao_gaia_calibration_path"] = str(_cert_path)
            log_event(
                f"DAO-Gaia calibration certificate {_cert.status}: "
                f"match_r={_cert.derived.match_radius_px:.1f}px "
                f"centroid={_cert.derived.pass2_center_tol_px:.1f}px "
                f"empty-sky det={_cert.empty_sky.inv_det} seed={_cert.empty_sky.inv_seed}"
            )
        except Exception as _cal_exc:  # noqa: BLE001
            from invariants_runtime import InvariantViolation  # noqa: PLC0415

            if isinstance(_cal_exc, InvariantViolation):
                raise
            if _membership_expand_meta:
                raise RuntimeError(
                    f"INV-MS-EXPAND-01: certificate write failed: {_cal_exc!s}"
                ) from _cal_exc
            log_event(f"DAO-Gaia calibration certificate skipped: {_cal_exc!s}")
        det_meta["gaia_census_meta"] = _gaia_meta
        det_meta["gaia_census_invariants"] = _census_inv.get("invariants") or []
        log_event(
            f"MASTERSTAR Gaia census: {len(_chip)} on-chip, "
            f"forced_seed={_gaia_meta.get('n_forced_seed', 0)}, "
            f"leftover_promotions={_gaia_meta.get('n_leftover_promotions', 0)}, "
            f"INV-MS-CENSUS-01 {_census_inv.get('detail')}"
        )
        _gdb_fill = str(_cfg_ms.gaia_db_path or "").strip()
        df_final, _n_bp_fill, _n_bp_miss = _fill_masterstars_gaia_matched_bp_rp_from_local_db(
            df_final,
            gaia_db_path=_gdb_fill,
        )
        if _n_bp_miss > 0:
            log_event(f"masterstars bp_rp fallback: {_n_bp_fill}/{_n_bp_miss} doplnenych z Gaia DB")
        _fleming_sigma: float | None = None
        if _gdb_fill:
            try:
                from dao_reconcile import (  # noqa: PLC0415
                    annotate_dao_only_magnitude_classes,
                    compute_gaia_dao_reconcile,
                    fit_fleming_completeness,
                    format_dao_only_census_log,
                    resolve_effective_match_depth,
                )

                _cone_df_cls = None
                _cone_csv_cls = Path(platesolve_dir) / "field_catalog_cone.csv"
                if _cone_csv_cls.is_file():
                    _cone_df_cls = read_vyvar_csv(_cone_csv_cls, low_memory=False, dtype={"catalog_id": str})
                _fwhm_cls = float(det_meta.get("dao_fwhm_px") or 0.0)
                if not (_fwhm_cls > 0.0):
                    _fwhm_cls = float(header_core_fwhm_px(hdr) or 3.5)
                _md_cls = resolve_effective_match_depth(det_meta, is_masterstar=True)
                _cone_lim_cls: float | None = None
                try:
                    _raw_lim = det_meta.get("faintest_mag_limit")
                    if _raw_lim is not None and math.isfinite(float(_raw_lim)):
                        _cone_lim_cls = float(_raw_lim)
                except (TypeError, ValueError):
                    _cone_lim_cls = None
                _noise_cls = det_meta.get("noise_floor_adu")
                _nax1_cls = int(hdr.get("NAXIS1") or 0)
                _nax2_cls = int(hdr.get("NAXIS2") or 0)
                _wcs_cls = None
                _plate_cls = None
                if _has_valid_wcs(hdr) and _nax1_cls > 0 and _nax2_cls > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FITSFixedWarning)
                        _wcs_cls = WCS(hdr)
                    try:
                        from astropy.wcs.utils import proj_plane_pixel_scales

                        _plate_cls = float(np.mean(proj_plane_pixel_scales(_wcs_cls) * 3600.0))
                    except Exception:  # noqa: BLE001
                        pass
                if _wcs_cls is not None:
                    _recon_ms = compute_gaia_dao_reconcile(
                        df_final,
                        gaia_db_path=_gdb_fill,
                        wcs=_wcs_cls,
                        naxis1=_nax1_cls,
                        naxis2=_nax2_cls,
                        fwhm_px=_fwhm_cls,
                        plate_scale_arcsec=_plate_cls,
                        mag_limit=float(det_meta.get("faintest_mag_limit") or 18.0),
                        match_sep_arcsec=float(
                            det_meta.get("match_sep_arcsec_effective")
                            or det_meta.get("match_sep_arcsec_requested")
                            or 8.0
                        ),
                        cone_df=_cone_df_cls,
                    )
                    _recon_ms.update(_md_cls)
                    _ff = fit_fleming_completeness(_recon_ms.get("completeness_curve") or [])
                    _fleming_sigma = _ff.sigma_mag
                df_final, _dao_class_meta = annotate_dao_only_magnitude_classes(
                    df_final,
                    gaia_db_path=_gdb_fill,
                    effective_match_depth=_md_cls.get("match_depth"),
                    cone_query_mag_limit=_cone_lim_cls,
                    fleming_sigma_mag=_fleming_sigma,
                    frame_noise_adu=_noise_cls,
                )
                if _recon_ms is not None:
                    _recon_ms["dao_only_class_meta"] = _dao_class_meta
                _ms_info_msg = format_dao_only_census_log(_dao_class_meta, n_total=len(df_final))
                LOGGER.info(_ms_info_msg)
                log_event(_ms_info_msg)
            except Exception as _ms_census_exc:  # noqa: BLE001
                LOGGER.debug("[MASTERSTAR-DAO-CENSUS] skipped: %s", _ms_census_exc)
                try:
                    from invariants_runtime import dao_only_fraction_from_masterstars  # noqa: PLC0415

                    _frac_ms = float(dao_only_fraction_from_masterstars(df_final))
                    _n_dao_ms = int(round(_frac_ms * float(len(df_final))))
                    _ms_info_msg = (
                        f"MASTERSTAR DAO_ONLY census: {_n_dao_ms}/{len(df_final)} "
                        f"(fraction={_frac_ms:.3f}) -- informational, not a gate"
                    )
                    LOGGER.info(_ms_info_msg)
                    log_event(_ms_info_msg)
                except Exception:  # noqa: BLE001
                    pass
        else:
            try:
                from invariants_runtime import dao_only_fraction_from_masterstars  # noqa: PLC0415

                _frac_ms = float(dao_only_fraction_from_masterstars(df_final))
                _n_dao_ms = int(round(_frac_ms * float(len(df_final))))
                _ms_info_msg = (
                    f"MASTERSTAR DAO_ONLY census: {_n_dao_ms}/{len(df_final)} "
                    f"(fraction={_frac_ms:.3f}) -- informational, not a gate"
                )
                LOGGER.info(_ms_info_msg)
                log_event(_ms_info_msg)
            except Exception as _ms_census_exc:  # noqa: BLE001
                LOGGER.debug("[MASTERSTAR-DAO-CENSUS] skipped: %s", _ms_census_exc)
    except Exception as exc:  # noqa: BLE001
        from invariants_runtime import InvariantViolation  # noqa: PLC0415

        if isinstance(exc, InvariantViolation):
            raise
        _cone_gaia_fail = Path(platesolve_dir) / "field_catalog_cone.csv"
        if _cone_gaia_fail.is_file() and _has_valid_wcs(hdr):
            raise RuntimeError(
                f"MASTERSTAR Gaia-complete enrich failed on production path: {exc!s}"
            ) from exc
        LOGGER.exception("[M1] MASTERSTAR source_type annotate failed: %s", exc)
        log_event(f"MASTERSTAR source_type annotate failed: {exc!s}")
    _vyvar_df_to_csv(df_final, csv_path)
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
    _cat_rows_opt = int(det_meta.get("catalog_rows", 0) or 0)
    if "catalog_id" in df_final.columns:
        _cid_opt = df_final["catalog_id"].fillna("").astype(str).str.strip()
        _n_gaia_det_opt = int(_cid_opt[_cid_opt != ""].nunique())
    else:
        _n_gaia_det_opt = int(_n_mat)
    _gaia_rate_opt = (100.0 * float(_n_gaia_det_opt) / float(_cat_rows_opt)) if _cat_rows_opt > 0 else 0.0
    log_event(
        f"[chart] MATCH STATS (optimized): Found {_n_det} stars on image | {_n_mat} matched with Gaia | "
        f"Match Rate: {_rate:.2f}% | Gaia->DAO: {_gaia_rate_opt:.2f}% ({_n_gaia_det_opt}/{_cat_rows_opt})"
    )
    if _cat_rows_opt > 0:
        LOGGER.info(
            "[MASTERSTAR] Gaia->DAO completeness: "
            "%d/%d (%.1f%%) | catalog_only: %d",
            _n_gaia_det_opt,
            _cat_rows_opt,
            _gaia_rate_opt,
            _cat_rows_opt - _n_gaia_det_opt,
        )
    log_event(
        f"MASTERSTAR JSON consistency: n_raw_dao={int(det_meta.get('n_detected_dao_raw', 0) or 0)}, "
        f"detected_stars={_n_det}, catalog_matched={_n_mat}, "
        f"gaia_dao_completeness_pct={round(_gaia_rate_opt, 2) if _cat_rows_opt > 0 else None}, "
        f"n_gaia_undetected={(_cat_rows_opt - _n_gaia_det_opt) if _cat_rows_opt > 0 else None}"
    )
    # TODO-25: persist to pipeline_meta.json so UI can read single source of truth
    if _cat_rows_opt > 0:
        _meta_patch: dict[str, Any] = {
            "gaia_dao_completeness_raw_pct": round(float(_gaia_rate_opt), 2),
            "catalog_rows": int(_cat_rows_opt),
            "n_gaia_detected": int(_n_gaia_det_opt),
            "n_gaia_undetected": int(_cat_rows_opt - _n_gaia_det_opt),
        }
        _idg_stamp = dict(det_meta.get("identity_gate") or {})
        _gmeta = det_meta.get("gaia_census_meta") or {}
        if isinstance(_gmeta, dict) and "n_lock_geometry_reject" in _gmeta:
            _idg_stamp["n_lock_geometry_reject"] = int(_gmeta.get("n_lock_geometry_reject") or 0)
        if _idg_stamp:
            _meta_patch["identity_gate"] = _idg_stamp
        try:
            from dao_gaia_calibration import effective_tol_stamps  # noqa: PLC0415

            _meta_patch["dao_gaia_tol"] = effective_tol_stamps(
                det_meta.get("dao_gaia_derived_tol")
                if isinstance(det_meta.get("dao_gaia_derived_tol"), dict)
                else None,
                _cfg_ms,
                fwhm_px=float(det_meta.get("dao_fwhm_px") or _idg_stamp.get("fwhm_px") or 3.5),
                census_meta=_gmeta if isinstance(_gmeta, dict) else None,
            )
        except Exception:  # noqa: BLE001
            pass
        for _mk in (
            "match_sep_arcsec_requested",
            "match_sep_arcsec_effective",
            "match_sep_formula_inputs",
            "wcs_gaia_pixel_refine_iters",
        ):
            if det_meta.get(_mk) is not None:
                _meta_patch[_mk] = det_meta.get(_mk)
        if _opt_stats_last:
            _meta_patch["optimizer_refit"] = dict(_opt_stats_last)
        if _wcs_rt_p99 is not None:
            _meta_patch["wcs_roundtrip_p99_px"] = float(_wcs_rt_p99)
            _meta_patch["wcs_roundtrip_pass"] = bool(_wcs_rt_pass)
        if _identity_qa:
            _inv_wcs_recs = _identity_qa.pop("_inv_wcs_01", None)
            _meta_patch.update(_identity_qa)
            if _inv_wcs_recs:
                _meta_patch.setdefault("invariants", [])
                if isinstance(_meta_patch["invariants"], list):
                    _meta_patch["invariants"].extend(list(_inv_wcs_recs))
        try:
            if _recon_ms is not None:
                _meta_patch.update(reconcile_to_pipeline_meta(_recon_ms))
            elif _dao_class_meta:
                from dao_reconcile import dao_only_class_meta_flat  # noqa: PLC0415

                _meta_patch.update(dao_only_class_meta_flat(_dao_class_meta))
            else:
                _cone_df = None
                _cone_csv = Path(platesolve_dir) / "field_catalog_cone.csv"
                if _cone_csv.is_file():
                    _cone_df = read_vyvar_csv(_cone_csv, low_memory=False, dtype={"catalog_id": str})
                _fwhm_recon = float(det_meta.get("dao_fwhm_px") or 0.0)
                if not (_fwhm_recon > 0.0):
                    _fwhm_recon = float(header_core_fwhm_px(hdr) or 3.5)
                _wcs_recon = None
                _plate_recon = None
                _nax1 = int(hdr.get("NAXIS1") or 0)
                _nax2 = int(hdr.get("NAXIS2") or 0)
                if _has_valid_wcs(hdr) and _nax1 > 0 and _nax2 > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FITSFixedWarning)
                        _wcs_recon = WCS(hdr)
                    try:
                        from astropy.wcs.utils import proj_plane_pixel_scales

                        _plate_recon = float(np.mean(proj_plane_pixel_scales(_wcs_recon) * 3600.0))
                    except Exception:  # noqa: BLE001
                        pass
                _gdb_recon = str(_cfg_ms.gaia_db_path or "").strip()
                _faintest_recon = float(det_meta.get("faintest_mag_limit") or 18.0)
                _match_sep_recon = float(
                    det_meta.get("match_sep_arcsec_effective")
                    or det_meta.get("match_sep_arcsec_requested")
                    or 8.0
                )
                if _wcs_recon is not None and _gdb_recon:
                    _recon = compute_gaia_dao_reconcile(
                        df_final,
                        gaia_db_path=_gdb_recon,
                        wcs=_wcs_recon,
                        naxis1=_nax1,
                        naxis2=_nax2,
                        fwhm_px=_fwhm_recon,
                        plate_scale_arcsec=_plate_recon,
                        mag_limit=_faintest_recon,
                        match_sep_arcsec=_match_sep_recon,
                        cone_df=_cone_df,
                    )
                    _md = resolve_effective_match_depth(det_meta, is_masterstar=True)
                    _recon.update(_md)
                    _meta_patch.update(reconcile_to_pipeline_meta(_recon))
        except Exception as exc:  # noqa: BLE001
            log_event(f"MASTERSTAR Gaia reconcile decomposition skipped: {exc!s}")
            _meta_patch["gaia_dao_completeness_pct"] = round(float(_gaia_rate_opt), 2)
        merge_photometry_pipeline_meta(
            Path(platesolve_dir) / "photometry",
            _meta_patch,
            _cfg_ms,
            entry_point="generate_masterstar_and_catalog",
        )
        # INV-DAG-01: masterstar stage stamp (cold-start OK if earlier stages absent).
        try:
            from invariants_runtime import stamp_stage_on_disk  # noqa: PLC0415

            stamp_stage_on_disk(
                Path(platesolve_dir) / "photometry",
                "masterstar",
                enforce_upstream=True,
            )
        except Exception as _dag_exc:  # noqa: BLE001
            logging.debug("[INV-DAG-01] masterstar stamp skipped: %s", _dag_exc)
    log_event(
        f"MASTERSTAR katalog: {Path(csv_path).name} - {len(df_final)} riadkov "
        f"(DAO + katalog na celom poli; ziadne orezanie podla vzdialenosti od stredu snimku)."
    )
    # Gaussian FWHM (2D fit) -> hlavicka; VY_FWHM je DAO odhad, nie moment FWHM - nepouzivaj 0.619.
    masterstars_df = df_final
    if (
        masterstars_df is None
        or len(masterstars_df) == 0
        or "x" not in masterstars_df.columns
        or "y" not in masterstars_df.columns
    ):
        masterstars_df = df_out
    try:
        from photometry_phase2a import measure_fwhm_from_masterstar

        _ms_path = Path(masterstar_fits)
        if "mag" in masterstars_df.columns:
            _star_pos = masterstars_df[["x", "y", "mag"]].dropna().head(50)
        elif "phot_g_mean_mag" in masterstars_df.columns:
            _star_pos = (
                masterstars_df[["x", "y", "phot_g_mean_mag"]]
                .dropna()
                .rename(columns={"phot_g_mean_mag": "mag"})
                .head(50)
            )
        else:
            _star_pos = masterstars_df[["x", "y"]].dropna().head(50)
        with fits.open(_ms_path, memmap=False) as _hint_hdul:
            _vy_hint = _hint_hdul[0].header.get("VY_FWHM", 3.5)
            _vy_fwhm_hint = float(_vy_hint) if _vy_hint is not None else 3.5
        _gaussian_fwhm = measure_fwhm_from_masterstar(
            _ms_path,
            _star_pos,
            dao_fwhm_hint=_vy_fwhm_hint,
            n_stars=30,
        )
        with fits.open(_ms_path, mode="update", memmap=False) as _hdul:
            _hdul[0].header["VY_FWHM_GAUSS"] = (
                round(float(_gaussian_fwhm), 4),
                "Gaussian FWHM px (2D fit)",
            )
            _n_raw_dao_hdr = int(det_meta.get("n_detected_dao_raw", 0) or 0)
            _hdul[0].header["VY_NDAO"] = (
                _n_raw_dao_hdr,
                "VYVAR: raw DAO detections on MASTERSTAR (stars/Mpx density)",
            )
            _hdul.flush()
        logging.info(
            f"[MASTERSTAR] VY_FWHM_GAUSS={float(_gaussian_fwhm):.3f}px ulozene do hlavicky (2D fit)"
        )
    except Exception as e:  # noqa: BLE001
        logging.error('[EXC-0409] Cross-setup `comparison_stars.csv` sync failure leaves B/V/R setups with inconsistent c...: %s', e)
        log_event(f"[ERROR] VY_FWHM_GAUSS fit ZLYHAL: {e}\n{traceback.format_exc()}")
    try:
        _ms_path_tag = Path(masterstar_fits)
        _n_raw_dao_hdr = int(det_meta.get("n_detected_dao_raw", 0) or 0)
        with fits.open(_ms_path_tag, mode="update", memmap=False) as _hdul_tag:
            if "VY_NDAO" not in _hdul_tag[0].header:
                _hdul_tag[0].header["VY_NDAO"] = (
                    _n_raw_dao_hdr,
                    "VYVAR: raw DAO detections on MASTERSTAR (stars/Mpx density)",
                )
                _hdul_tag.flush()
    except Exception:  # noqa: BLE001
        pass
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
    # F-431 / C1: stamp AFTER VT exists (write_photometry_plan_files).
    try:
        _vt_stamp_path = platesolve_dir / "variable_targets.csv"
        if _vt_stamp_path.is_file():
            from photometry_core import stamp_vsx_known_variable_on_masterstars  # noqa: PLC0415

            _ms_for_stamp = pd.read_csv(
                csv_path, low_memory=False, dtype={"catalog_id": str, "name": str}
            )
            _vt_stamp_df = pd.read_csv(_vt_stamp_path, low_memory=False, dtype={"catalog_id": str})
            _ms_for_stamp, _vsx_stamp_final = stamp_vsx_known_variable_on_masterstars(
                _ms_for_stamp,
                _vt_stamp_df,
                log_fn=log_event,
            )
            _vyvar_df_to_csv(_ms_for_stamp, csv_path)
            df_final = _ms_for_stamp
            log_event(
                f"MASTERSTAR VSX catalog_id stamp (post-VT): "
                f"id_join={_vsx_stamp_final.get('id_join')} "
                f"positional_fallback={_vsx_stamp_final.get('positional_fallback')}"
            )
        else:
            log_event("MASTERSTAR VSX stamp skipped: variable_targets.csv missing after photometry plan.")
    except Exception as _vsx_final_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR VSX catalog_id stamp (post-VT) skipped: {_vsx_final_exc!s}")
    # Multi-filter support: keep comparison stars consistent across platesolve/<setup>/ folders.
    try:
        _sync_comparison_stars_across_setups(Path(platesolve_dir).parent)
    except Exception as _sync_exc:  # noqa: BLE001
        log_event(
            f"MASTERSTAR: comparison-star cross-setup sync failed ({_sync_exc!s}); "
            "B/V/R comp sets may be inconsistent across setups."
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
        "catalog_derived_membership": det_meta.get("catalog_derived_membership"),
        "dao_threshold_sigma": det_meta.get("dao_threshold_sigma"),
        "masterstar_match_png": "",
    }
    out.update(photo_plan)
    # Enrichment columns for masterstars_full_match.csv (formerly MASTER_SOURCES DB).
    try:
        gaia_db = str(_cfg_ms.gaia_db_path or "").strip()
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
                    mag_limit=None,
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

                        # Saturation threshold for MASTERSTAR (FITS + EQUIPMENTS; no global config fallback)
                        sat_limit = det_meta.get("saturate_limit_adu")
                        if sat_limit is None:
                            _eq_sat_ms = equipment_saturate_adu
                            if _eq_sat_ms is None and equipment_id is not None:
                                _eq_sat_ms = _equipment_saturate_adu_from_db(equipment_id)
                            sat_limit, _ = _effective_saturation_limit(
                                hdr,
                                fallback_adu=None,
                                equipment_saturate_adu=_eq_sat_ms,
                            )
                        if (
                            sat_limit is not None
                            and math.isfinite(float(sat_limit))
                            and float(sat_limit) > 0
                        ):
                            sat_thr = float(sat_limit) * float(saturate_level_fraction)
                        else:
                            # INV-SAT-LIMIT: never admit against +inf.
                            sat_thr = float(SAT_LIMIT_CONTAINER_CLIP_ADU) * float(SAT_LIMIT_NO_KNEE_FRAC)
                            logging.warning(
                                "[INV-SAT-LIMIT] MASTERSTAR sat_thr unresolved; "
                                "using peak-test %.1f ADU (0.80 x container clip)",
                                sat_thr,
                            )

                        rows_ms: list[dict[str, Any]] = []
                        det_ok = det.iloc[np.where(ok)[0]].reset_index(drop=True)
                        g_ok = gdf.iloc[idx[np.where(ok)[0]]].reset_index(drop=True)
                        g_ok_idx = idx[np.where(ok)[0]]
                        # Aperture optimization: estimate per-star FWHM on MASTERSTAR, then take medians per color.
                        try:
                            import numpy as _np

                            arr_ms = _np.asarray(data, dtype=_np.float32)

                            fwhm_est = [
                                _fwhm_moment_at(
                                    arr_ms,
                                    float(det_ok["x"].iloc[i]) if "x" in det_ok.columns and pd.notna(det_ok["x"].iloc[i]) else float("nan"),
                                    float(det_ok["y"].iloc[i]) if "y" in det_ok.columns and pd.notna(det_ok["y"].iloc[i]) else float("nan"),
                                    half=6,
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

                        # Gaia neighbour veto radius in arcsec: 3x median FWHM (px) x plate scale.
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
                        # photometry.py is legacy; use the merged core module.
                        from photometry_core import bad_columns_for_light_frame

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
                            _pp = float(_cfg_ms.nonlinearity_peak_percentile)
                            _thr_nl = float(
                                _np.percentile(
                                    _np.asarray(_peaks_nl, dtype=_np.float64),
                                    min(100.0, max(0.0, 100.0 - _pp)),
                                )
                            )
                        _nl_ratio = float(_cfg_ms.nonlinearity_fwhm_ratio)
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
                                elif int(nss) > 0:
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
                                    "non_single_star": int(nss),
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
                            from masterstars_enrichment import (  # noqa: PLC0415
                                apply_common_field_bbox_exclusion,
                                apply_stress_rms_to_rows_ms,
                                apply_vsx_variable_flags,
                                merge_enrichment_into_masterstars_df,
                            )

                            df_final = merge_enrichment_into_masterstars_df(df_final, rows_ms)
                            _vyvar_df_to_csv(df_final, csv_path)
                            out["masterstars_enrichment_written"] = int(len(rows_ms))
                            try:
                                _wp2 = write_photometry_plan_files(
                                    platesolve_dir=platesolve_dir,
                                    masterstar_fits=masterstar_fits,
                                    masterstars_csv=csv_path,
                                    n_comparison_stars=int(n_comparison_stars),
                                    require_non_variable=bool(require_non_variable_comparisons),
                                    draft_id=int(draft_id),
                                )
                                out.update(_wp2)
                            except Exception as _wp2_exc:  # noqa: BLE001
                                log_event(
                                    f"MASTERSTAR: enriched photometry-plan rewrite failed ({_wp2_exc!s}); "
                                    "keeping the prior photometry plan."
                                )
                        except Exception as exc:  # noqa: BLE001
                            out["masterstars_enrichment_error"] = str(exc)

                        # Stress-test: 10% random sample, exclude Border/Blended by default (soft-crop).
                        try:
                            from masterstars_enrichment import merge_enrichment_into_masterstars_df  # noqa: PLC0415

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
                                    apply_common_field_bbox_exclusion(
                                        rows_ms,
                                        x0=float(x0b),
                                        x1=float(x1b),
                                        y0=float(y0b),
                                        y1=float(y1b),
                                    )
                                    df_final = merge_enrichment_into_masterstars_df(df_final, rows_ms)
                                    _vyvar_df_to_csv(df_final, csv_path)
                                    out["common_field_bbox_px"] = [float(x0b), float(y0b), float(x1b), float(y1b)]
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

                            by_bin: dict[str, list[float]] = {}
                            for rr in rows_ms:
                                if int(rr.get("is_safe_comp") or 0) != 1:
                                    continue
                                sid = str(rr.get("source_id_gaia") or "").strip()
                                if not sid or sid not in st_res.per_source_rms:
                                    continue
                                b = str(rr.get("phot_category") or "").strip()
                                if b:
                                    by_bin.setdefault(b, []).append(float(st_res.per_source_rms[sid]))
                            med_by_bin = {b: float(pd.Series(v).median()) for b, v in by_bin.items() if v}
                            apply_stress_rms_to_rows_ms(rows_ms, st_res.per_source_rms, med_by_bin)

                            packed = [
                                {
                                    "source_id_gaia": rr.get("source_id_gaia"),
                                    "phot_category": rr.get("phot_category"),
                                    "stress_rms": rr.get("stress_rms"),
                                    "ra": rr.get("ra"),
                                    "dec": rr.get("dec"),
                                }
                                for rr in rows_ms
                                if rr.get("stress_rms") is not None
                            ]
                            var_ids = vsx_is_known_variable_top3_per_bin(rows=packed)
                            if var_ids:
                                apply_vsx_variable_flags(rows_ms, set(var_ids))
                                out["vsx_flagged_variables"] = int(len(var_ids))

                            df_final = merge_enrichment_into_masterstars_df(df_final, rows_ms)
                            _vyvar_df_to_csv(df_final, csv_path)
                        except Exception as exc:  # noqa: BLE001
                            out["stress_test_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        out["masterstars_enrichment_error"] = str(exc)
    # Persist MASTERSTAR path on draft for later UI reloads / Step 3 continuity.
    try:
        if draft_id is not None:
            _db_ms = VyvarDatabase(Path(_cfg_ms.database_path))
            try:
                _db_ms.set_obs_draft_masterstar_fits_path(int(draft_id), str(Path(masterstar_fits).resolve()))
            finally:
                _db_ms.conn.close()
    except Exception as exc:  # noqa: BLE001
        out["masterstar_path_store_error"] = str(exc)
    return out


def _astrometry_align_impl_body(
    *,
    job: dict[str, Any],
    archive_path: Path,
    astrometry_api_key: str | None = None,
    max_control_points: int = 80,
    min_detected_stars: int = 100,
    max_detected_stars: int = 500,
    platesolve_backend: str = "vyvar",
    plate_solve_fov_deg: float = 1.0,
    max_extra_platesolve: int = 0,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    max_catalog_rows: int = 12000,
    n_comparison_stars: int = 0,
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
    sibling_recovery_use_masterstar: bool = False,
    osc_registration_handoff: dict[str, Any] | None = None,
    osc_write_registration_handoff: bool = False,
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
    # Zarovnane FITS: {archive}/detrended_aligned/lights/{filter_exp_binning}/... (vnorene cesty podla vstupu)
    os.makedirs(str(aligned_root), exist_ok=True)
    os.makedirs(str(platesolve_dir), exist_ok=True)
    _cfg_align = app_config or AppConfig()
    _align_star_cap = max(10, min(5000, int(_cfg_align.alignment_max_stars)))
    # Keep alignment input stable on dense fields: use at most TOP 200 brightest stars.
    _align_star_cap = min(_align_star_cap, 200)
    _sips_sig = float(_cfg_align.sips_dao_threshold_sigma)
    if not math.isfinite(_sips_sig) or _sips_sig <= 0:
        _sips_sig = 3.5
    _cfg_align_sig = float(_cfg_align.alignment_detection_sigma)
    if not math.isfinite(_cfg_align_sig) or _cfg_align_sig <= 0:
        _cfg_align_sig = _sips_sig
    try:
        _ui_sig = float(dao_threshold_sigma)
    except (TypeError, ValueError):
        _ui_sig = 0.0
    if not math.isfinite(_ui_sig) or _ui_sig <= 0:
        _ui_sig = 0.0
    # Session override > Settings alignment_detection_sigma > sips_dao_threshold_sigma.
    _align_det_sigma = max(
        0.8,
        min(20.0, _ui_sig if _ui_sig > 0 else _cfg_align_sig),
    )
    _fb_align = float(_cfg_align.sips_dao_fwhm_px)
    if not math.isfinite(_fb_align) or _fb_align <= 0:
        _fb_align = 2.5
    _pfov_align: float | None = None
    if build_masterstar_and_catalogs:
        LOGGER.info("Astrometria + MASTERSTAR + per-frame CSV: archiv %s", ap)
    else:
        LOGGER.info("Astrometria + zarovnanie + per-frame CSV (bez MASTERSTAR): archiv %s", ap)
    # MASTERSTAR initial match: allow a looser sep (min 10") for robust first-pass Gaia join.
    _catalog_match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
    if _catalog_match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
        _pipeline_ui_info(
            f"Katalogovy match prah zvyseny na {_catalog_match_sep_eff:.2f}\" "
            "(minimum pre robustny pociatocny cross-match)."
        )

    _cat_loc_only = bool(catalog_local_gaia_only) if catalog_local_gaia_only is not None else True
    if _cat_loc_only:
        LOGGER.info("Katalog: rezim lokalny Gaia (SQLite)")
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
            # EXC-0414: T4 -- `_db_pf.conn.close()` after plate-scale lookup for per-frame match sep; cleanup only. (EXCEPT-BULK-2 2026-07-08)
            pass
    if not files:
        raise FileNotFoundError(
            f"Chybaju FITS v {detrended_root}. Plate solve cita len **spracovane** snimky. "
            "Najprv spusti **MAKE MASTERSTAR** po kroku **Analyze** (zapis do "
            f"`{ap / 'processed' / 'lights'}` alebo starsie `{ap / 'detrended' / 'lights'}`)."
        )

    _t_step3_start = time.time()
    n_files = len(files)
    ref_fp, ref_star_scores = _pick_reference_frame_by_star_count(files)
    if osc_registration_handoff is not None:
        ref_name = str(osc_registration_handoff.get("reference_file") or "")
        for _rf in files:
            if _rf.name == ref_name:
                ref_fp = _rf
                break
    # Read reference once (no lock during solve step).
    with fits.open(ref_fp, memmap=False) as hdul:
        ref_hdr = hdul[0].header.copy()
        ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
    _rh, _rw = int(ref_data.shape[0]), int(ref_data.shape[1])
    try:
        _pf_try = float(plate_solve_fov_deg)
        if math.isfinite(_pf_try) and _pf_try > 0:
            _pfov_align = _pf_try
    except (TypeError, ValueError):
        _pfov_align = None
    if _pfov_align is None:
        _pfov_align = resolve_plate_solve_fov_deg_hint(
            ref_hdr,
            _rh,
            _rw,
            database_path=_cfg_align.database_path,
            equipment_id=int(id_equipment) if id_equipment is not None else None,
            draft_id=int(draft_id) if draft_id is not None else None,
        )
    if _pfov_align is None:
        _pfov_align = float(_cfg_align.plate_solve_fov_deg)

    _scale_pf: float | None = None
    _db_pf = _vyvar_open_database(_cfg_align)
    if _db_pf is not None:
        try:
            _eq_pf, _tel_pf = resolve_optics_ids_for_platesolve(
                _db_pf, draft_id, equipment_id=id_equipment
            )
            _xb_pf, _yb_pf = fits_binning_xy_from_header(ref_hdr)
            _bin_pf = max(1, int(_xb_pf), int(_yb_pf))
            _scale_pf = compute_plate_scale_from_db(_eq_pf, _tel_pf, _db_pf.conn, binning=_bin_pf)
        except Exception:  # noqa: BLE001
            _scale_pf = None
        finally:
            try:
                _db_pf.conn.close()
            except Exception:  # noqa: BLE001
                pass
    _j_psep = job.get("per_frame_catalog_match_sep_arcsec")
    if _j_psep is not None:
        try:
            per_frame_match_sep = float(_j_psep)
        except (TypeError, ValueError):
            per_frame_match_sep = per_frame_catalog_match_sep_arcsec_for_scale(_scale_pf)
    else:
        per_frame_match_sep = per_frame_catalog_match_sep_arcsec_for_scale(_scale_pf)
    if not math.isfinite(per_frame_match_sep) or per_frame_match_sep <= 0:
        per_frame_match_sep = per_frame_catalog_match_sep_arcsec_for_scale(_scale_pf)

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

    # --- MASTERSTAR build + plate-solve (per-setup platesolve/) before alignment ---
    # IMPORTANT (multi-filter): each setup must have its own MASTERSTAR + catalogs, otherwise
    # R/V/B runs overwrite each other (MASTERSTAR.fits, masterstars_full_match.csv, VY_MIRR, ...)
    # and reference/per-frame astrometry becomes unstable.
    _masterstar_built = False
    _cat_info_root: dict[str, Any] = {}
    _ps_root = platesolve_dir
    _t_platesolve = time.time()
    if build_masterstar_and_catalogs:
        _prog("platesolve/MASTERSTAR: referencny snimok + plate-solve + katalogy...")
        _cat_info_root = generate_masterstar_and_catalog(
            archive_path=ap,
            max_catalog_rows=int(max_catalog_rows),
            astrometry_api_key=astrometry_api_key,
            source_root=detrended_root,
            platesolve_dir=_ps_root,
            platesolve_backend=platesolve_backend,
            plate_solve_fov_deg=float(_pfov_align),
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

        # Prefer MASTERSTAR as the canonical alignment reference when available.
        # This guarantees that:
        # - the output pixel grid matches MASTERSTAR (no WCS/data grid mismatch),
        # - per-frame matching against masterstars_full_match.csv works reliably.
        try:
            _ms_fp = _cat_info_root.get("masterstar_fits") if isinstance(_cat_info_root, dict) else None
            if _ms_fp:
                _ms_path = Path(str(_ms_fp)).resolve()
                if _ms_path.is_file():
                    with fits.open(_ms_path, memmap=False) as hdul:
                        ref_hdr = hdul[0].header.copy()
                        ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
                    ref_fp = _ms_path
                    has_wcs = _has_valid_wcs(ref_hdr)
                    log_event(f"INFO: Alignment reference set to MASTERSTAR: {ref_fp.name}")
        except Exception as _ms_ref_exc:  # noqa: BLE001
            # EXC-0416: T1 -- Copying MASTERSTAR WCS onto reference FITS fails silently; aligned products can carry a... (EXCEPT-BULK-2 2026-07-08)
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().masterstar_ref_swap_fail += 1
            LOGGER.error("Using MASTERSTAR as alignment reference failed: %s", _ms_ref_exc)

    _prog(
        f"detrended_aligned/lights: pripravujem zarovnanie ({n_files} snimok z {detrended_root.name}/...)..."
    )

    # If MASTERSTAR was built for this setup and has a valid WCS, prefer it as the canonical
    # WCS for detrended_aligned products. Some frames already carry a WCS that can be offset by
    # arcminutes from MASTERSTAR; using it would break per-frame Gaia matching (master_reference_sky).
    if build_masterstar_and_catalogs and isinstance(_cat_info_root, dict):
        try:
            _ms_fp = _cat_info_root.get("masterstar_fits")
            if _ms_fp:
                _ms_path = Path(str(_ms_fp)).resolve()
                if _ms_path.is_file():
                    with fits.open(_ms_path, memmap=False) as _ms_hdul:
                        _ms_hdr = _ms_hdul[0].header.copy()
                    if _has_valid_wcs(_ms_hdr):
                        _apply_wcs_header_to_fits(ref_fp, _ms_hdr)
                        with fits.open(ref_fp, memmap=False) as hdul:
                            ref_hdr = hdul[0].header.copy()
                            ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
                        has_wcs = True
                        log_event(
                            f"INFO: Reference WCS prevzate z MASTERSTAR ({_ms_path.name}) - pouzijem MASTERSTAR WCS pre alignment aj per-frame match."
                        )
        except Exception as _wcs_copy_exc:  # noqa: BLE001
            try:
                log_event(f"DEBUG: Reference WCS copy from MASTERSTAR failed: {_wcs_copy_exc}")
            except Exception:  # noqa: BLE001
                pass

    if (
        not build_masterstar_and_catalogs
        and sibling_recovery_use_masterstar
        and not _masterstar_built
    ):
        try:
            _ms_path = (platesolve_dir / "MASTERSTAR.fits").resolve()
            if _ms_path.is_file():
                with fits.open(_ms_path, memmap=False) as hdul:
                    _ms_hdr = hdul[0].header.copy()
                    ref_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
                if _has_valid_wcs(_ms_hdr):
                    ref_fp = _ms_path
                    ref_hdr = _ms_hdr
                    has_wcs = True
                    _cat_info_root = {
                        "masterstar_fits": str(_ms_path),
                        "masterstars_csv": str(platesolve_dir / "masterstars_full_match.csv"),
                    }
                    log_event(
                        f"INFO: Sibling-recovery alignment using existing MASTERSTAR: {_ms_path.name}"
                    )
        except Exception as _sib_ms_exc:  # noqa: BLE001
            # EXC-0418: T3 -- DEBUG min/max/mean/NaN stats logging `pass`; alignment detection proceeds regardless. (EXCEPT-BULK-2 2026-07-08)
            log_event(f"DEBUG: Sibling-recovery MASTERSTAR load failed: {_sib_ms_exc}")

    if not has_wcs:
        _prog("Plate solve referencie (moze chvilu trvat)...")

    if not has_wcs:
        # Solve reference file in-place (no open handle on Windows).
        _hra, _hdec, _ = _pointing_hint_from_header(ref_hdr)
        solve = _solve_wcs_external(
            ref_fp,
            backend=platesolve_backend,
            astrometry_api_key=astrometry_api_key,
            plate_solve_fov_deg=float(_pfov_align),
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

    # Use the same FWHM rule as per-frame alignment (VY_FWHM / header), not only ``sips_dao_fwhm_px``.
    # A fixed ~2.5 px kernel on the reference while sources use ~5 px yields different brightest-N
    # orderings -> bogus point pairs -> astroalign "triangles exhausted" and identity/no_wcs cascades.
    _raw_ref_fw = dao_detection_fwhm_pixels(ref_hdr, configured_fallback=_fb_align)
    try:
        _fwv = float(_raw_ref_fw) if _raw_ref_fw is not None else float("nan")
    except (TypeError, ValueError):
        _fwv = float("nan")
    _align_fwhm_ref = float(_fwv) if math.isfinite(_fwv) and _fwv > 0 else float(_fb_align)

    hint_center = _wcs_field_center_radec_deg(ref_fp)
    hint_ra: float | None = hint_center[0] if hint_center else None
    hint_dec: float | None = hint_center[1] if hint_center else None

    extra_platesolve_results: list[dict[str, Any]] = []

    try:
        pass  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0419] `estimate_archive_memory_profile` failure leaves `use_ram_handoff=True`; may exhaust RA...: %s', exc)
        raise RuntimeError(f"astroalign required for frame registration: {exc}") from exc

    log_event(
        f"Detekcia hviezd: Pouzite FWHM={_align_fwhm_ref:.2f}, Sigma={_align_det_sigma:.2f}"
    )

    def _maybe_refine_aligned(
        hdr_mut: fits.Header, data_mut: np.ndarray, label: str, *, dao_fwhm_px_frame: float
    ) -> None:
        _ = (hdr_mut, data_mut, label, dao_fwhm_px_frame)
        return

    # Adaptive alignment star budget:
    # - if STAR_COUNT > 1000 -> use top 300 brightest
    # - if STAR_COUNT < 100 -> use all
    # - else -> cap at 300
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
    ref_xy_all = _alignment_detect_xy(
        data_to_detect,
        int(max(100, max_detected_stars)),
        det_sigma=_align_det_sigma,
        fwhm_px=_align_fwhm_ref,
        label=ref_fp.name,
        log_sink=None,
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
        f"cap pre transform={_align_star_cap}, DAO sigma={_align_det_sigma:.2f}, FWHM={_align_fwhm_ref:.2f}px "
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
        # EXC-0420: T1 -- `_vy_fwhm_header_value` returns `None` on read error; MASTERSTAR match heuristic falls ... (EXCEPT-BULK-2 2026-07-08)
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
        # EXC-0421: T2 -- `_aligned_masterstar_matches_platesolve` stat/read failure returns `False`, allowing al... (EXCEPT-BULK-2 2026-07-08)
        rotation_ref_angle_deg = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            ref_wcs_obj = WCS(ref_hdr) if _has_valid_wcs(ref_hdr) else None
    except Exception:  # noqa: BLE001
        ref_wcs_obj = None

    # save reference as aligned baseline, keep WCS
    _prog(
        f"detrended_aligned/lights: {'RAM - referencia' if use_ram_handoff else 'zapisujem FITS'} "
        f"{ref_fp.name} (1/{n_files})"
    )
    try:
        ref_rel = ref_fp.relative_to(detrended_root)
    except Exception:  # noqa: BLE001
        # Reference can live outside detrended_root (e.g. MASTERSTAR in platesolve/...).
        # In that case, store it at the aligned root top-level.
        ref_rel = Path(ref_fp.name)
    ref_out = aligned_root / ref_rel
    _ensure_parent_dirs_for_aligned_fits(ref_out)

    def _vy_fwhm_header_value(path: Path) -> float | None:
        try:
            with fits.open(path, memmap=False) as _h:
                _v = _h[0].header.get("VY_FWHM")
            if _v is None:
                return None
            vv = float(_v)
            return float(vv) if math.isfinite(vv) and vv > 0 else None
        except Exception:  # noqa: BLE001
            return None

    def _aligned_masterstar_matches_platesolve(*, platesolve_masterstar: Path, aligned_masterstar: Path) -> bool:
        """Heuristic match: size+mtime OR size+VY_FWHM.

        This is intentionally lightweight to prevent unintended overwrites of the aligned MASTERSTAR baseline.
        """
        try:
            if not platesolve_masterstar.is_file() or not aligned_masterstar.is_file():
                return False
            s_src = platesolve_masterstar.stat()
            s_dst = aligned_masterstar.stat()
            if int(s_src.st_size) != int(s_dst.st_size):
                return False
            # Preferred fast path: preserved mtime via shutil.copy2
            if int(s_src.st_mtime) == int(s_dst.st_mtime):
                return True
        except Exception:  # noqa: BLE001
            return False
        # Header fallback: VY_FWHM should be stable for a given MASTERSTAR (copied+solved).
        v_src = _vy_fwhm_header_value(platesolve_masterstar)
        v_dst = _vy_fwhm_header_value(aligned_masterstar)
        if v_src is None or v_dst is None:
            return False
        return abs(float(v_src) - float(v_dst)) <= 1e-9

    def _ensure_aligned_masterstar_copy(*, platesolve_masterstar: Path, aligned_masterstar: Path) -> None:
        import shutil  # local import: narrow scope

        if aligned_masterstar.is_file() and _aligned_masterstar_matches_platesolve(
            platesolve_masterstar=platesolve_masterstar, aligned_masterstar=aligned_masterstar
        ):
            log_event(
                "[MASTERSTAR] aligned ref already exists and matches platesolve source - skipping write"
            )
            return
        log_event("[MASTERSTAR] copying platesolve MASTERSTAR -> detrended_aligned ref")
        shutil.copy2(platesolve_masterstar, aligned_masterstar)

    _ps_masterstar = (platesolve_dir / "MASTERSTAR.fits")
    _is_platesolve_masterstar_ref = False
    try:
        _is_platesolve_masterstar_ref = (
            ref_fp.name.strip().casefold() == "masterstar.fits"
            and _ps_masterstar.is_file()
            and ref_fp.resolve() == _ps_masterstar.resolve()
            and ref_out.name.strip().casefold() == "masterstar.fits"
        )
    except Exception:  # noqa: BLE001
        _is_platesolve_masterstar_ref = False

    if _is_platesolve_masterstar_ref:
        _ensure_aligned_masterstar_copy(
            platesolve_masterstar=_ps_masterstar.resolve(),
            aligned_masterstar=ref_out,
        )
    else:
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
    # Astroalign control-point cap from cfg (decoupled from detection ladder max_stars).
    align_cp = int(max(12, min(500, int(_cfg_align.alignment_max_control_points))))
    ref_pts = np.asarray(ref_xy_fit, dtype=np.float32)
    if ref_pts is None or len(ref_pts) == 0:
        raise ValueError("Referencne hviezdy su prazdne pred startom alignmentu!")
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

    n_written_align = 1
    n_align_workers = _vyvar_parallel_worker_count(_cfg_align)
    align_tasks: list[tuple[str, int]] = []
    for frame_index_1based, fp in enumerate(files, start=1):
        if fp == ref_fp:
            continue
        align_tasks.append((str(fp.resolve()), int(frame_index_1based)))

    _align_ctx: dict[str, Any] = {
        "ref_data": ref_data,
        "ref_hdr": ref_hdr.copy(),
        "ref_fp_name": ref_fp.name,
        "fixed_target_pts": np.copy(fixed_target_pts).astype(np.float32, copy=False),
        "reference_list": list(REFERENCE_LIST),
        "has_ref_wcs": ref_wcs_obj is not None,
        "platesolve_dir": str(platesolve_dir),
        "align_star_cap": int(_align_star_cap),
        "max_control_points": int(align_cp),
        "min_detected_stars": int(min_detected_stars),
        "max_detected_stars": int(max_detected_stars),
        "fb_align": float(_fb_align),
        "rotation_ref_angle_deg": rotation_ref_angle_deg,
    }
    _osc_registration_capture: dict[str, dict[str, Any]] = {}

    def _flush_one_alignment(res: dict[str, Any]) -> None:
        nonlocal n_written_align, _flip_logged, rotation_flip_first_index_1based
        idx = int(res["frame_index_1based"])
        fp = Path(res["fp"])
        if bool(res.get("is_flipped", False)):
            rotation_flip_frame_indices_1based.append(idx)
            if rotation_flip_first_index_1based is None:
                rotation_flip_first_index_1based = idx
            if not _flip_logged:
                log_event(
                    f"Physical rotation change detected at frame index {idx}. "
                    "Adjusting alignment strategy."
                )
                _flip_logged = True
        if res.get("kind") == "failed_skip":
            star_counts.append(res["star_count"])
            return
        hdr_out = res["hdr"]
        aligned_data = res["aligned_data"]
        fw_i = float(res["fw_i"])
        _maybe_refine_aligned(hdr_out, aligned_data, fp.name, dao_fwhm_px_frame=fw_i)
        try:
            fp_rel = fp.relative_to(detrended_root)
        except Exception:  # noqa: BLE001
            fp_rel = Path(fp.name)
        out_fp = aligned_root / fp_rel
        _ensure_parent_dirs_for_aligned_fits(out_fp)
        n_written_align += 1
        _prog(
            f"detrended_aligned/lights: "
            f"{'RAM - zarovnanie' if use_ram_handoff else 'zapisujem FITS'} "
            f"{fp.name} ({n_written_align}/{n_files})..."
        )
        if use_ram_handoff:
            aligned_ram_buffer.append((fp_rel.as_posix(), hdr_out.copy(), np.copy(aligned_data)))
        else:
            fits.writeto(out_fp, aligned_data, header=hdr_out, overwrite=True)
        aligned_files.append(out_fp)
        star_counts.append(res["star_count"])
        rh = res.get("registration_handoff")
        if isinstance(rh, dict) and res.get("star_count", {}).get("file"):
            _osc_registration_capture[str(res["star_count"]["file"])] = dict(rh)

    if osc_registration_handoff is not None:
        from osc_align import apply_registration_handoff_to_frame

        handoff_frames = dict(osc_registration_handoff.get("frames") or {})
        ref_name = str(osc_registration_handoff.get("reference_file") or ref_fp.name)
        for fp in files:
            if fp == ref_fp and fp.name != ref_name:
                continue
            if fp.name == ref_name:
                continue
            entry = handoff_frames.get(fp.name) or {}
            if entry and not bool(entry.get("aligned", True)):
                star_counts.append(
                    {
                        "file": fp.name,
                        "frame_index": int(files.index(fp) + 1) if fp in files else 0,
                        "detected_stars": 0,
                        "aligned": False,
                        "reason": "donor_not_aligned",
                        "alignment_method": "osc_handoff_skip",
                        "is_flipped": False,
                    }
                )
                continue
            with fits.open(fp, memmap=False) as hdul:
                raw_hdr = hdul[0].header.copy()
                raw_data = _as_fits_float32_image(hdul[0].data).astype(np.float32, copy=False)
            aligned_data, hdr_out, method = apply_registration_handoff_to_frame(
                frame_path=fp,
                frame_data=raw_data,
                frame_hdr=raw_hdr,
                ref_data=ref_data,
                ref_hdr=ref_hdr,
                handoff_entry=entry,
            )
            _flush_one_alignment(
                {
                    "kind": "aligned",
                    "fp": str(fp.resolve()),
                    "frame_index_1based": int(files.index(fp) + 1) if fp in files else 0,
                    "is_flipped": False,
                    "hdr": hdr_out,
                    "aligned_data": aligned_data,
                    "aligned_method": method,
                    "fw_i": float(_fb_align),
                    "star_count": {
                        "file": fp.name,
                        "frame_index": int(files.index(fp) + 1) if fp in files else 0,
                        "detected_stars": 0,
                        "aligned": True,
                        "alignment_method": method,
                        "is_flipped": False,
                    },
                }
            )
    elif n_align_workers > 1 and len(align_tasks) > 1:
        _mp_ctx: dict[str, Any] = {
            "ref_data": np.ascontiguousarray(np.copy(_align_ctx["ref_data"])),
            "ref_hdr": _align_ctx["ref_hdr"].copy(),
            "ref_fp_name": _align_ctx["ref_fp_name"],
            "fixed_target_pts": np.copy(_align_ctx["fixed_target_pts"]).astype(np.float32, copy=False),
            "reference_list": list(_align_ctx["reference_list"]),
            "has_ref_wcs": bool(_align_ctx["has_ref_wcs"]),
            "platesolve_dir": str(_align_ctx["platesolve_dir"]),
            "align_star_cap": int(_align_ctx["align_star_cap"]),
            "max_control_points": int(_align_ctx["max_control_points"]),
            "min_detected_stars": int(_align_ctx["min_detected_stars"]),
            "max_detected_stars": int(_align_ctx["max_detected_stars"]),
            "fb_align": float(_align_ctx["fb_align"]),
            "rotation_ref_angle_deg": _align_ctx["rotation_ref_angle_deg"],
        }
        def _run_alignment_single_process() -> None:
            for fp_s, idx in align_tasks:
                res = _alignment_compute_one_frame(Path(fp_s), int(idx), _align_ctx, None)
                _flush_one_alignment(res)

        # A-durable: resolve the MP init/task by FRESH module attribute at call time, so the
        # objects handed to the spawn pool are exactly what sys.modules resolves - even if the
        # Streamlit file-watcher reloaded vyvar_alignment_frame after pipeline.py was imported
        # (the import-time `from ... import` binding would otherwise go stale -> PicklingError).
        _mp_init = vyvar_alignment_frame._astrometry_align_mp_init
        _mp_task = vyvar_alignment_frame._astrometry_align_mp_task
        try:
            with ProcessPoolExecutor(
                max_workers=n_align_workers,
                initializer=_mp_init,
                initargs=(_mp_ctx,),
            ) as pool:
                raw_list = list(pool.map(_mp_task, align_tasks, chunksize=1))
            for res in raw_list:
                for ln in res.get("log_events", ()):
                    log_event(ln)
                res_flush = {k: v for k, v in res.items() if k != "log_events"}
                _flush_one_alignment(res_flush)
        except pickle.PicklingError as _pkl_err:
            # Robust fallback: if the spawn pool cannot pickle the MP funcs (e.g. mid-run module
            # reload defeats the fresh-attr lookup), run alignment single-process instead of
            # aborting. Photometry is byte-identical to the MP path (same per-frame compute).
            # PicklingError is raised at task-submission (pickling the worker funcs) before any
            # result is flushed, so no partial per-frame state exists here - run single-process.
            _pipeline_ui_info(
                f"Alignment: multiprocessing dispatch failed to pickle worker functions "
                f"({_pkl_err}); falling back to single-process alignment."
            )
            _run_alignment_single_process()
    else:
        for fp_s, idx in align_tasks:
            res = _alignment_compute_one_frame(Path(fp_s), int(idx), _align_ctx, None)
            _flush_one_alignment(res)

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
            f"Alignment warning: zlyhalo {n_failed_align}/{len(files)} snimok v {aligned_root.resolve()} "
            f"(dovody: {reason_txt})."
        )
    if n_aligned <= 1:
        msg = (
            f"Alignment zlyhal: uspesne zarovnana len referencia (1/{len(files)}). "
            f"Skontroluj DAO prah/FWHM a WCS vstupy. Vystupny priecinok: {aligned_root.resolve()}."
        )
        _pipeline_ui_error(msg)
        raise RuntimeError(msg)
    rep_path = platesolve_dir / "alignment_report.csv"
    pd.DataFrame(star_counts).to_csv(rep_path, index=False)
    if osc_write_registration_handoff or bool(job.get("osc_write_registration_handoff")):
        from osc_align import write_registration_handoff

        write_registration_handoff(
            platesolve_dir,
            reference_file=str(ref_fp.name),
            frames=_osc_registration_capture,
        )

    print(f"  Zarovnanie: {time.time() - _t_align:.1f}s")
    _t_csv = time.time()

    # If we aligned in RAM, flush aligned FITS to disk before MASTERSTAR (needs files on disk).
    _ram_flushed_before_masterstar = False
    if use_ram_handoff and aligned_ram_buffer and build_masterstar_and_catalogs:
        _prog("detrended_aligned/lights: zapisujem FITS na disk (RAM -> disk, pred MASTERSTAR)...")
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
        _aligned_file_list = sorted(aligned_root.glob("proc_*.fits"))
        LOGGER.info(f"[BORDER] RAM flush done - {len(_aligned_file_list)} aligned frames on disk")

    cat_info: dict[str, Any] = {}
    ms_csv: Path | None = None
    ms_fits: Path | None = None
    use_master_fast = False

    if build_masterstar_and_catalogs:
        # Use per-setup platesolve/ artifacts built before alignment.
        cat_info = dict(_cat_info_root or {})
        try:
            ms_csv = Path(str((cat_info.get("masterstars_csv") or (_ps_root / "masterstars_full_match.csv")))).resolve()
        except Exception:  # noqa: BLE001
            # EXC-0422: T3 -- Optional `variable_targets_csv` path probe `pass`; UI metadata field may be absent. (EXCEPT-BULK-2 2026-07-08)
            ms_csv = Path(str(cat_info.get("masterstars_csv") or (_ps_root / "masterstars_full_match.csv")))
        try:
            ms_fits = Path(str((cat_info.get("masterstar_fits") or (_ps_root / "MASTERSTAR.fits")))).resolve()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0423] Post-RAM-flush DB-aware photometry plan rewrite `pass`es; border-safe bbox may use stal...: %s', exc)
            ms_fits = Path(str(cat_info.get("masterstar_fits") or (_ps_root / "MASTERSTAR.fits")))

        # comparison_stars.csv / variable_targets.csv are produced in this setup directory already.
        try:
            _comp = platesolve_dir / "comparison_stars.csv"
            _var = platesolve_dir / "variable_targets.csv"
            if _comp.is_file():
                cat_info["comparison_stars_csv"] = str(_comp)
            if _var.is_file():
                cat_info["variable_targets_csv"] = str(_var)
        except Exception:  # noqa: BLE001
            pass

        # Masterstar lock for Step 3: per-frame catalogs must use one fixed reference list.
        use_master_fast = True

        # Recompute photometry plan after alignment so border-safe bbox uses aligned frames.
        try:
            _aligned_disk: list[Path] = []
            if use_ram_handoff and aligned_ram_buffer and _ram_flushed_before_masterstar:
                _aligned_disk = list(_aligned_file_list)
            else:
                try:
                    _aligned_disk = sorted(aligned_root.glob("proc_*.fits"))
                except Exception:  # noqa: BLE001
                    _aligned_disk = []
            _has_aligned = bool(_aligned_disk) or bool(use_ram_handoff and aligned_ram_buffer)
            if build_masterstar_and_catalogs and _has_aligned:
                _wp_aligned = write_photometry_plan_files(
                    platesolve_dir=platesolve_dir,
                    masterstar_fits=ms_fits or (platesolve_dir / "MASTERSTAR.fits"),
                    masterstars_csv=ms_csv or (platesolve_dir / "masterstars_full_match.csv"),
                    n_comparison_stars=int(n_comparison_stars),
                    require_non_variable=bool(require_non_variable_comparisons),
                    draft_id=int(draft_id) if draft_id is not None else None,
                    database_path=(
                        Path(str(_cfg_align.database_path))
                        if str(_cfg_align.database_path or "").strip()
                        else None
                    ),
                    aligned_files=_aligned_disk if _aligned_disk else None,
                    aligned_ram_frames=aligned_ram_buffer if use_ram_handoff and aligned_ram_buffer else None,
                    require_safe_bbox=True,
                )
                cat_info.update(_wp_aligned or {})
        except RuntimeError as _wp_exc:
            log_event(f"[BORDER] Post-alignment photometry plan rewrite failed: {_wp_exc!s}")
            if "[BORDER]" in str(_wp_exc):
                raise
        except Exception as _wp_exc:  # noqa: BLE001
            log_event(f"[BORDER] Post-alignment photometry plan rewrite failed: {_wp_exc!s}")

    export_base = prog_i[0]
    _catalog_app_cfg = _cfg_align
    _, _run_epsf = _photometry_mode_run_flags(
        _catalog_app_cfg,
        platesolve_dir=platesolve_dir,
    )

    # TODO-8: Build ePSF model after MASTERSTAR (Phase 2B prep)
    if _run_epsf:
        try:
            if draft_id is not None and str(_catalog_app_cfg.database_path or "").strip():
                _ms_for_epsf = ms_fits
                if _ms_for_epsf is None or not Path(_ms_for_epsf).is_file():
                    _ms_try = platesolve_dir / "MASTERSTAR.fits"
                    if _ms_try.is_file():
                        _ms_for_epsf = _ms_try
                _ms_csv_epsf = ms_csv
                if _ms_csv_epsf is None or not Path(_ms_csv_epsf).is_file():
                    _ms_csv_try = platesolve_dir / "masterstars_full_match.csv"
                    if _ms_csv_try.is_file():
                        _ms_csv_epsf = _ms_csv_try
                if (
                    _ms_for_epsf is not None
                    and _ms_csv_epsf is not None
                    and Path(_ms_for_epsf).is_file()
                    and Path(_ms_csv_epsf).is_file()
                ):
                    from psf_photometry import build_epsf_model

                    _db_epsf = VyvarDatabase(Path(str(_catalog_app_cfg.database_path)))
                    try:
                        _epsf_path = build_epsf_model(
                            masterstar_fits_path=Path(_ms_for_epsf),
                            masterstars_csv_path=Path(_ms_csv_epsf),
                            db=_db_epsf,
                            draft_id=int(draft_id),
                            # TODO-PSF-PHASE2: moffat_centroids not yet available at MASTERSTAR
                            # build time - requires per-frame Moffat run first then aggregate
                            # centroids. Implement in next session.
                        )
                        LOGGER.info("[ePSF] Model built: %s", _epsf_path)
                    finally:
                        _db_epsf.conn.close()
        except Exception as _e:  # noqa: BLE001
            # EXC-0424: T1 -- `normalize_gaia_source_id_series` failure `pass`es before deferred CSV write; per-frame... (EXCEPT-BULK-2 2026-07-08)
            LOGGER.warning("[ePSF] build_epsf_model failed (non-fatal): %s", _e)

    def _cat_prog(i: int, tot: int, msg: str) -> None:
        if progress_cb is None:
            return
        progress_cb(
            min(export_base + i, global_total),
            global_total,
            f"detrended_aligned/lights: CSV ({i}/{tot}) - {msg}",
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
            plate_solve_fov_deg=float(_pfov_align),
            master_dark_path=_master_dark_bpm_path,
            draft_id=draft_id,
            equipment_id=id_equipment,
            full_catalog_export=True,
        )
        _prog("detrended_aligned/lights: zapisujem FITS + CSV na disk (davka po praci v RAM)...")
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
            try:
                from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

                if isinstance(df, pd.DataFrame) and "catalog_id" in df.columns:
                    df = df.copy()
                    df["catalog_id"] = normalize_gaia_source_id_series(df["catalog_id"])
            except Exception:  # noqa: BLE001
                pass
            df.to_csv(pcsv, index=False)
        pd.DataFrame(per_cat.get("frames", [])).to_csv(Path(per_cat["index_csv"]), index=False)
        if per_cat.get("deferred_csv_writes"):
            _hybrid_ram = _finalize_hybrid_bkg_fallback_sidecar(
                aligned_root,
                err_background_mode="empirical",
                write_sidecar=True,
                gain=float(_catalog_app_cfg.gain),
                read_noise=float(_catalog_app_cfg.read_noise),
                setup_label=str(aligned_root.name),
            )
            if _hybrid_ram:
                per_cat["hybrid_bkg_fallback"] = _hybrid_ram
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
            plate_solve_fov_deg=float(_pfov_align),
            master_dark_path=_master_dark_bpm_path,
            draft_id=draft_id,
            equipment_id=id_equipment,
            full_catalog_export=True,
        )

    _assert_alignment_produced_fits(aligned_root)

    print(f"  Per-frame CSV: {time.time() - _t_csv:.1f}s")
    print(f"CELKOM krok 3 ({obs_group_key or detrended_root.name}): {time.time() - _t_step3_start:.1f}s")

    LOGGER.info(
        "Astrometria dokoncena: zarovnane %s / %s snimok; per-frame CSV: %s; MASTERSTAR: %s; RAM handoff: %s",
        n_aligned,
        len(files),
        int(per_cat.get("written", 0)),
        "ano" if build_masterstar_and_catalogs else "nie",
        "ano" if use_ram_handoff else "nie",
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


def _analyze_calibrated_qc_one(src: Path) -> dict[str, Any]:
    import numpy as np

    src = Path(src)
    try:
        with fits.open(src, memmap=False) as hdul:
            hdr = hdul[0].header
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        qc = _qc_fwhm_elongation(data)
        finite = np.isfinite(data)
        arr = data[finite]
        return {
            "src": str(src),
            "filter": _safe_filter_token(str(hdr.get("FILTER") or hdr.get("FILT") or "NoFilter")),
            "fwhm_px": qc.get("fwhm_px"),
            "elongation": qc.get("elongation"),
            "n_sources": qc.get("n_sources"),
            "n_stars_detected": qc.get("n_stars_detected"),
            "bg_median": float(np.nanmedian(arr)) if arr.size else None,
            "p50": float(np.nanpercentile(arr, 50)) if arr.size else None,
            "p99": float(np.nanpercentile(arr, 99)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
        }
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0439] [SILENT-DROP] analyze calibrated QC one-frame helper failed: %s', exc)
        return {"src": str(src), "status": f"error: {exc}"}


from cal_stage import _header_has_vy_skysf  # noqa: E402,F401  # re-export; survivor in cal_stage.py


def analyze_calibrated_qc(
    *,
    calibrated_root: Path,
    max_frames: int | None = None,
    progress_cb: Callable[..., None] | None = None,
    only_paths: Sequence[Path | str] | None = None,
) -> pd.DataFrame:
    """Analyze calibrated frames (QC) without writing /processed outputs.

    Runs FWHM/elongation QC on calibrated data in memory and returns a QC dataframe.

    If ``max_frames`` is None, every light FITS under ``calibrated_root`` is analyzed.

    Parallelism: jednotny pocet workerov (auto CPU/RAM alebo env, pozri :func:`_vyvar_parallel_worker_count`);
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
            "VYVAR QC analyze: %s frames; odhad spicky RAM ~%s (float32 + docasne polia)",
            total,
            format_memory_bytes(_peak),
        )
    n_workers = _vyvar_qc_preprocess_workers()
    if n_workers > 1 and total > 1:
        LOGGER.info(
            "VYVAR QC analyze: parallel_workers=%s (paralelne; ~%sx RAM na snimok oproti 1 vlaknu)",
            n_workers,
            n_workers,
        )

    if n_workers > 1 and total > 1:
        with _vyvar_parallel_pool(n_workers) as ex:
            futs = {
                ex.submit(_analyze_calibrated_qc_one, src): src
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
            rows.append(_analyze_calibrated_qc_one(src))

    return pd.DataFrame(rows)


class AstroPipeline:
    """Skeleton for the modular variable-star processing workflow."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.db = VyvarDatabase(self.config.database_path)
        self.db._archive_root_override = Path(self.config.archive_root)

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
        roundness_reject_above: float | None = 1.25,
    ) -> dict[str, Any]:
        """Calibrate imported draft/raw lights into `calibrated/` under archive_path."""
        ap = Path(archive_path)
        # Accept both draft root (.../draft_xxx) and direct non_calibrated path (.../draft_xxx/non_calibrated).
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        LOGGER.info("Kalibracia archivu: %s", ap)
        # Support Draft structure: <archive>/Raw/lights and optional <archive>/non_calibrated/lights.
        # Both sources are calibrated/passthrough-written into the single target: <archive>/calibrated/lights.
        outputs: dict[str, Any] = {"archive_path": str(ap_root), "results": {}, "perf10_qsum": {}}
        merged_perf10: dict[str, dict[str, Any]] = {}

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

        for _sec_stats in (outputs.get("results") or {}).values():
            if isinstance(_sec_stats, dict):
                _p10 = _sec_stats.get("perf10_qc_results")
                if isinstance(_p10, dict):
                    merged_perf10.update(_p10)

        if (
            bool(self.config.dao_qc_in_calibrate)
            and merged_perf10
            and draft_id is not None
        ):
            outputs["perf10_qsum"] = apply_perf10_dao_qc_to_obs_files(
                db=self.db,
                draft_id=int(draft_id),
                archive_path=ap_root,
                perf10_qc_results=merged_perf10,
                roundness_reject_above=roundness_reject_above,
            )

        LOGGER.info("Kalibracia dokoncena (sekcii vystupu: %s)", list((outputs.get("results") or {}).keys()))
        if equipment_id is not None:
            cal_root = ap_root / "calibrated" / "lights"
            if cal_root.is_dir():
                osc_out = run_osc_channel_extraction_for_archive(
                    calibrated_lights_root=cal_root,
                    db=self.db,
                    equipment_id=int(equipment_id),
                    app_config=self.config,
                    progress_cb=progress_cb,
                )
                outputs["osc_extraction"] = osc_out
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

        - ``output_paths``: list aligned with ``light_paths`` - calibrated FITS path or ``None`` on failure
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
            ]
        ] = []
        for lp in light_paths:
            src_p = Path(lp).resolve()
            rel = src_p.relative_to(lr)
            dst_p = (cr / rel).resolve()
            items.append((str(src_p), str(dst_p), md_init, mf_serial, qc_pack))

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

        cal_diag_session = CalDiagSession()
        db_cal = _db_for_calibration_tasks(qc_pack)
        _lpaths = [Path(it[0]) for it in items]
        if _lpaths and md_init:
            cal_diag_session = run_cal_diag_pregate(
                _lpaths,
                obs_group_key_from_path=_obs_group_key_from_light_path,
                resolve_dark_path=lambda fp, og, lb: Path(md_init) if md_init else None,
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
        cal_diag_worker_blob = _cal_diag_export_for_workers(cal_diag_session)

        rows: list[dict[str, Any]]
        if nw <= 1:
            md_pre: Any = None
            if md_init:
                with fits.open(md_init, memmap=False) as h:
                    md_pre = np.array(h[0].data, dtype=np.float32, copy=True)
            flat_cache: dict[str, Any] = {}
            flat_med: dict[str, float] = {}
            db_main = _db_for_calibration_tasks(qc_pack)
            rows = []
            for i, it in enumerate(items):
                src_s, dst_s, md_s, mf_map, _qopt = it
                src_p = Path(src_s)
                dst_p = Path(dst_s)
                try:
                    _ok = _obs_group_key_from_light_path(src_p)
                    if is_obs_group_aborted(cal_diag_session, _ok):
                        if dst_p.exists():
                            dst_p.unlink(missing_ok=True)
                        rows.append(
                            {
                                "src": src_s,
                                "dst": dst_s,
                                "ok": True,
                                "skipped": True,
                                "error": None,
                                "qc_summary": None,
                                "traceback": None,
                            }
                        )
                        if progress_cb is not None:
                            progress_cb(i + 1, n, f"CAL-DIAG skip {src_p.name}")
                        continue
                    light_bx = _light_binning_from_path(src_p)
                    gr = gate_result_for_frame(
                        cal_diag_session,
                        obs_group_key=_ok,
                        dark_path=Path(md_s) if md_s else None,
                        light_binning=light_bx,
                    )
                    md_np = md_pre
                    if md_s:
                        with fits.open(src_p, memmap=False) as hdul:
                            lshape = (int(hdul[0].data.shape[0]), int(hdul[0].data.shape[1]))
                        md_np = dark_np_for_cal_diag(
                            cal_diag_session,
                            master_binning=_native_b,
                            dark_path=Path(md_s),
                            light_binning=light_bx,
                            light_shape=lshape,
                            light_filename=src_p.name,
                            gate_result=gr,
                        )
                    mf = {str(k): Path(v) if v else None for k, v in mf_map.items()}
                    _ud, _uf, qc_sum, _cf, _p10 = _calibrate_one_light_disk(
                        src=src_p,
                        dst=dst_p,
                        master_dark_path=Path(md_s) if md_s else None,
                        masterflat_by_filter=mf,
                        flat_cache=flat_cache,
                        flat_median_scale=flat_med,
                        md_data_preload=md_np,
                        db=db_main,
                        qc_pack=_qopt,
                        calibration_master_native_binning=_native_b,
                        cal_diag_gate_result=gr,
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
                initargs=(md_init, _native_b, cal_diag_worker_blob),
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
            if run:
                df = preprocess_calibrated_to_processed(
                    calibrated_root=source_dir,
                    processed_root=source_dir,
                    reject_fwhm_px=reject_fwhm_px,
                    reject_elongation=reject_elongation,
                    use_gpu_if_available=use_gpu_if_available,
                    progress_cb=None,
                    app_config=self.config,
                )
            else:
                qc_csv_existing = source_dir / "qc_metrics.csv"
                if not qc_csv_existing.exists():
                    qc_csv_existing = ap_root / "detrended" / "lights" / "qc_metrics.csv"
                try:
                    df = pd.read_csv(qc_csv_existing) if qc_csv_existing.exists() else pd.DataFrame()
                except Exception:  # noqa: BLE001
                    df = pd.DataFrame()
            out["processed"]["source"] = {
                "lights_root": str(source_dir),
                "rows": int(len(df)),
                "rejected": int((df["status"].astype(str).str.startswith("rejected")).sum()) if not df.empty else 0,
                "source_dir": str(source_dir),
            }
            try:
                qc_csv = source_dir / "qc_metrics.csv"
                out["processed"]["source"]["qc_csv"] = str(qc_csv)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)
            out["qc_suggestions"]["source"] = _qc_suggest_thresholds(df)

        if not out["processed"]:
            out["warning"] = "No calibrated lights found under this archive_path. Run calibration first (Run quick pipeline step)."
        return out

    def quick_analyze_last_import(
        self,
        *,
        archive_path: Path | str,
        max_frames: int | None = None,
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
            )
            out["analysis"]["calibrated"] = {"rows": int(len(df))}
            out["qc_suggestions"]["calibrated"] = _qc_suggest_thresholds(df)
            try:
                qc_csv = ap_root / "calibrated" / "lights" / "qc_analysis.csv"
                df.to_csv(qc_csv, index=False)
                out["analysis"]["calibrated"]["qc_csv"] = str(qc_csv)
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0444] `fetch_draft_light_rows_for_quality` failure logs WARNING and returns empty jump result...: %s', exc)
                LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)
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
        scanning_id = self.db.derive_scanning_id(metadata)

        if id_equipment is None or id_telescope is None:
            raise ValueError(
                "prepare_observation_from_session: id_equipment a id_telescope su povinne "
                "(vyberte kameru a dalekohlad v Session Upload)."
            )
        equipment_id = int(id_equipment)
        telescope_id = int(id_telescope)

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

from pipeline_calibrate import (  # noqa: E402,F401
    SkySurfaceOrderConflictError,
    _CALIB_MASTER_NB_UNSET,
    _archive_preprocess_lights_root,
    _archive_root_from_lights_root,
    _available_system_ram_bytes,
    _cal_diag_export_for_workers,
    _cal_diag_session_from_export,
    _calibrate_batch_process_one,
    _calibrate_one_light_apply_masters_in_ram,
    _calibrate_one_light_disk,
    _calibration_flags,
    _calibration_type_from_flags,
    _cfg_calibration_library_native_binning,
    _dao_star_table_mean_roundness,
    _db_for_calibration_tasks,
    _decide_preprocess_sky_action,
    _effective_saturation_limit,
    _estimate_dao_fwhm_guess,
    _estimate_fov_deg_from_header,
    _exposure_sec_from_header,
    _filter_light_paths_maybe,
    _fit_subtract_preprocess_sky_surface,
    _fits_primary_pixel_count,
    _half_flux_radius_in_cutout,
    _has_usable_master_dark,
    _has_valid_wcs,
    _hdr_vy_cflag_str,
    _header_vyskyord,
    _infer_raw_light_path_for_calibrated,
    _infer_sat_limit_from_bitpix,
    _init_calibrate_batch_worker,
    _inspection_jd_from_header,
    _iter_light_fits,
    _light_binning_from_path,
    _log_calibration_io_preflight,
    _log_calibration_metadata_diagnostic,
    _match_and_crop_pair,
    _mean_hfr_bright_stars_dao,
    _moment_fwhm_elong_peak_at,
    _obs_group_key_from_light_path,
    _passthrough_lights_to_calibrated,
    _perf10_lookup_qc,
    _pick_light_for_metadata_diagnostic,
    _pipeline_ui_error,
    _post_calibration_qc_eval,
    _qc_center_crop_for_stars,
    _qc_enrich_calibrated_in_place,
    _qc_enrich_one_frame,
    _qc_fwhm_elongation,
    _qc_pack_from_config,
    _quality_inspection_dao_metrics_array,
    _resolve_dark_path_for_light,
    _resolve_draft_light_raw_path,
    _robust_frame_fwhm_median,
    _saturate_limit_adu_from_header,
    _saturation_adu_for_cal_diag,
    _strip_raw_linearity_header_keywords,
    _sync_manifest_cal_stage_from_qc_row,
    _sync_obs_calibration_state_with_retry,
    _vy_calib_status_numeric,
    _vyvar_calibrate_multiprocessing_enabled,
    _vyvar_parallel_pool,
    _vyvar_parallel_use_processes,
    _vyvar_parallel_worker_count,
    _vyvar_qc_preprocess_workers,
    apply_perf10_dao_qc_to_obs_files,
    calibrate_lights_to_calibrated,
    draft_median_pointing_icrs_deg,
    estimate_archive_memory_profile,
    estimate_memory_from_fits_headers,
    format_memory_bytes,
    norm_fits_path_key,
    run_draft_ram_calibration_qc_to_obs_files,
    run_osc_channel_extraction_for_archive,
    scan_calibrated_lights_pointing,
    sync_obs_files_drift_arcmin_for_draft,
)
import pipeline_calibrate as _pipeline_calibrate  # noqa: E402

# Call-time follow so monkeypatch.setattr(pipeline, "_fit_subtract_preprocess_sky_surface", ...)
# still reaches _qc_enrich_one_frame (moved).
_pipeline_calibrate._fit_subtract_preprocess_sky_surface = (
    lambda *a, **k: _fit_subtract_preprocess_sky_surface(*a, **k)
)


from pipeline_preprocess import (  # noqa: E402,F401
    _archive_raw_to_calibrated_light,
    _load_raw_for_frame,
    _load_raw_hdr_for_frame,
    _partition_detrended_by_subfolder,
    _qc_suggest_thresholds,
    build_prefilter_rejected_map,
    calibrated_paths_for_draft_apply_filters,
    filter_files_by_qc_metrics_allowlist,
    load_qc_metrics_status_by_path,
    preprocess_calibrated_to_processed,
    qc_enrich_calibrated_lights_in_place,
    resolve_obs_file_to_processed_fits,
    resolve_preprocess_target_coordinates,
)
import pipeline_preprocess as _pipeline_preprocess  # noqa: E402


from pipeline_astrometry import (  # noqa: E402,F401
    _EPSF_SKIP_LOGGED,
    _VYVAR_TIME_JD_CSV_COLS,
    _apply_wcs_header_to_fits,
    _assert_alignment_produced_fits,
    _catalog_match_radius_px,
    _dao_targeted_pass2_unmatched_gaia,
    _ensure_parent_dirs_for_aligned_fits,
    _equipment_saturate_adu_from_db,
    _export_catalog_psf_st_fields,
    _field_jump_empty_result,
    _fill_masterstars_gaia_matched_bp_rp_from_local_db,
    _finite_positive_adu,
    _header_focal_length_mm,
    _header_vy_fwhm_px,
    _merge_astrometry_group_reports,
    _merge_dao_pass1_pass2_tables,
    _merge_platesolve_gaia_pairs_into_masterstars_df,
    _merge_vsx_exoplanet_variable_targets,
    _pass2_sibling_wcs_recovery,
    _path_is_under_tree,
    _path_segments_forbidden_for_masterstar_physical_source,
    _photometry_mode_run_flags,
    _pick_preferred_masterstar_basename_hit,
    _pipeline_ui_info,
    _plate_solve_input_bundle,
    _query_vsx_local_frame_bbox,
    _resolve_best_effort_path_under,
    _resolve_focal_mm_for_plate_scale,
    _run_osc_multi_group_alignment,
    _safe_proc_name,
    _sat_adu_from_draft_sat_diag,
    _solve_wcs_external,
    _sort_masterstar_paths_by_fwhm,
    _strip_external_platesolve_header,
    _sync_comparison_stars_across_setups,
    _try_rescale_masterstar_linear_wcs_to_expected_plate_scale,
    _update_masterstar_obs_file_status,
    _vyvar_df_round_time_jd_for_csv,
    _vyvar_df_to_csv,
    _vyvar_open_database,
    _vyvar_per_frame_csv_workers,
    _wcs_field_center_radec_deg,
    astrometry_align_and_build_masterstar,
    build_masterstar_from_detrended,
    compute_plate_scale_from_db,
    detect_field_jumps,
    draft_is_multi_group_obs,
    draft_obs_group_count,
    get_masterstar_candidate_rows,
    get_masterstar_candidates,
    resolve_masterstar_input_root,
    resolve_plate_solve_fov_deg_hint,
    select_comparison_stars_spatial_grid,
    write_photometry_plan_files,
)
import pipeline_astrometry as _pipeline_astrometry  # noqa: E402

# Call-time follow so monkeypatch.setattr(pipeline, "_plate_solve_input_bundle", ...)
# still reaches in-module callers after the move.
_pipeline_astrometry._plate_solve_input_bundle = (
    lambda *a, **k: _plate_solve_input_bundle(*a, **k)
)


from pipeline_catalog import (  # noqa: E402,F401
    _BATCH_E_N_EQUIV_LOGGED,
    _EXPORT_PER_FRAME_WORKER_STATE,
    _MASTERSTAR_ZONE_LOG_ONCE,
    _MOFFAT_CHI2_LIMIT,
    _airmass_from_altitude_deg,
    _all_pix2world_icrs_deg,
    _annotate_masterstars_flux_zones,
    _apply_aperture_catalog_enhancements_from_st,
    _apply_dao_centroid_wcs_guard,
    _apply_exo_host_columns_to_proc_df,
    _apply_wcs_tan_fragment_to_header,
    _bin2d_mean,
    _box_peak_max_adu,
    _box_peaks_at_centroids,
    _build_exoplanet_promotion_rows_from_masterstars,
    _catalog_df_cap_brightest_by_mag,
    _cfg_from_export_worker_state,
    _chord_to_arcsec,
    _compute_airmass_from_altaz,
    _dao_auto_binning_factor,
    _dao_convolved_background_rms_adu,
    _dao_detection_threshold_adu,
    _dao_noise_sigma_adu,
    _dao_spatial_flux_cap_row_indices,
    _dao_star_count_from_array,
    _detect_empirical_clip_level_adu,
    _effective_field_catalog_cone_radius_deg,
    _epsf_fit_catalog_ids,
    _epsf_target_catalog_ids,
    _estimate_catalog_frame_hw,
    _exo_host_annotation_arrays,
    _export_first_icrs_center_radius,
    _export_per_frame_disk_worker_task,
    _export_per_frame_ram_worker_task,
    _export_per_frame_run_catalog_core,
    _extract_airmass_from_header,
    _field_catalog_cone_meta_path,
    _field_center_and_radius_from_wcs,
    _fill_psf_catalog_columns,
    _finalize_hybrid_bkg_fallback_sidecar,
    _fits_header_first_positive_float,
    _fits_header_vy_algn_aligned,
    _gaia_catalog_cone_radius_optics_floor_deg,
    _gaia_chip_xy_from_catalog,
    _icrs_center_radius_from_hdr_data,
    _icrs_deg_to_unitxyz,
    _init_export_per_frame_worker,
    _invalidate_field_catalog_cone_cache_if_needed,
    _lock_matched_centroids_to_master_grid,
    _masterstar_zone_linear_threshold,
    _masterstar_zone_log_once,
    _mean_bin2d_for_dao,
    _pick_reference_frame_by_star_count,
    _pixel_noise_sigma_pp_adu,
    _prefetch_export_shared_catalog_for_process_pool,
    _prefilter_dao_table_brightest,
    _proc_catalog_keep_matched_rows_only,
    _proc_deduplicate_matched_catalog_rows,
    _proc_drop_unmatched_dao_rows,
    _proc_rename_det_names_to_catalog_id,
    _proc_sat_block_for_csv,
    _query_exoplanet_local,
    _query_gaia_local,
    _query_vsx_local,
    _resolve_masterstar_bg_sigma_adu,
    _resolve_peak_saturation_limit_adu,
    _sat_ctx_from_worker,
    _saturated_core_plateau_vectorized,
    _slice_exo_annotation,
    _vectorized_star_saturation_columns,
    _vyvar_cap_mp_workers_for_catalog,
    _write_field_catalog_cone_meta,
    build_ucac_catalog_kdtree,
    detect_stars_match_master_reference,
    find_qc_metrics_csv,
    inv_sat_limit_peak_test_adu,
    nearest_sky_nn_kdtree,
)
import pipeline_catalog as _pipeline_catalog  # noqa: E402

from pipeline_ui_helpers import (  # noqa: E402,F401
    _resolve_light_fits_for_quality_inspection,
    run_quality_analysis,
    list_best_processed_light_paths_for_masterstar,
    resolve_masterstars_metadata_csv,
    preprocess_sky_summary_from_df,
)

from pipeline_gate_helpers import validate_comparison_ensemble_flatness  # noqa: E402,F401
import pipeline_gate_helpers as _pipeline_gate_helpers

# Call-time follow so monkeypatch.setattr("pipeline.extract_fits_metadata", ...)
# still reaches validate_comparison_ensemble_flatness (moved; risk_register patch-string).
_pipeline_gate_helpers.extract_fits_metadata = (
    lambda *a, **k: extract_fits_metadata(*a, **k)
)

from epsf_hooks import (  # noqa: E402,F401
    _add_catalog_ids_from_csv,
    _epsf_lc_catalog_ids,
)
