"""Photometry core - zluceny modul (photometry + photometry_phase2a)."""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import random
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import AbstractSet, Any, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from astropy.io import fits as astrofits

from stats_core import _flux_to_mag
from comp_pool_rms import attach_comp_rms_to_pool_rows, compute_global_pool_rms_map
from comp_rms_loo import (
    COMP_RMS_FRAMES_BASIS,
    COMP_RMS_LOO_PHOTON_K_DEFAULT,
    LN10_OVER_2P5,
    compute_loo_mag_rms_map,
)
from proc_frame_store import (
    PROC_CSV_GLOB,
    PROC_STORE_COLS,
    ProcFrameStore,
    is_masterstar_proc_name,
    proc_csv_path_for_aligned_fits,
)
from config import (
    AppConfig,
    DENSITY_OVERRIDES,
    apply_crowding_overrides,
    apply_density_overrides,
    classify_field_density,
    compute_field_density,
    resolve_comp_sparse_fallback_enabled,
)
from database import query_local_gaia, query_local_gaia_by_source_ids
from gaia_catalog_id import (
    GAIA_PROC_CSV_READ_DTYPE,
    masterstar_row_gaia_key,
    normalize_gaia_source_id,
    read_vyvar_csv,
)
from infolog import log_event
from plain_stats import plain_mean_med_std

from catalog_match_trust import is_wcs_untrusted_catalog_match_mode, normalize_catalog_match_mode
from jd_axis_format import jd_axis_title, jd_series_relative
from utils import iter_fits_paths_recursive as _iter_fits_recursive
from unit_resolver import (
    phase01_chip_interior_margin_px as _resolve_chip_interior_margin_px,
    phase01_comparison_isolation_radius_px as _resolve_isolation_radius_px,
    resolve_max_dist_fallback_deg,
    resolve_px_from_arcsec,
    resolve_px_from_fwhm_factor,
    plate_scale_arcsec_per_px_from_header,
    sips_dao_fwhm_px as _resolve_sips_dao_fwhm_px,
)

LOGGER = logging.getLogger(__name__)

_MAD_CONSISTENCY = 0.6745  # normalizacny faktor MAD -> sigma ekvivalent

# Explicit annulus sky (ADU/px) for Howell err; ``noise_floor_adu`` remains detection-floor legacy.
SKY_ADU_PER_PX_ANNULUS_COL = "sky_adu_per_px_annulus"
SKY_SURFACE_BG_MEDIAN_ADU_COL = "sky_surface_bg_median_adu"

# F-BINGAIN-1: empirical background noise (empty-aperture scatter) + provenance.
SIGMA_BKG_AP_COL = "sigma_bkg_ap"
ERR_BKG_SOURCE_COL = "err_bkg_source"
ERR_BKG_MODE_EMPIRICAL = "empirical"
ERR_BKG_MODE_HOWELL = "howell"
ERR_BKG_SOURCE_EMPIRICAL = "empirical"
ERR_BKG_SOURCE_HOWELL_FALLBACK = "howell_fallback"
ERR_BKG_SOURCE_HOWELL_SCALED = "howell_scaled"
BKG_SCALE_R_CLAMP_LO = 0.05
BKG_SCALE_R_CLAMP_HI = 2.0





# Per-target LC time provenance (F-BJD-1): labels BJD recompute path, does not alter time values.
TIME_BASE_COL = "time_base"
TIME_BASE_BJD_TDB = "BJD_TDB"
TIME_BASE_JD_FALLBACK = "JD_FALLBACK"










# Comp tier: Gaia BP-RP outside this band -> unreliable vs field comps (use B-V fallback).
_BPRP_VALID_MIN = 0.1
_BPRP_VALID_MAX = 3.5

# Gaia ID (`catalog_id`, VSX / masterstars `name`) musi byt str - float64 straca cifry
_GAIA_ID_DTYPE: dict[str, type] = dict(GAIA_PROC_CSV_READ_DTYPE)




_COMP_QUALITY_JSON_META_KEYS = frozenset(
    {
        "selected_tier",
        "tier4_warning",
        "n_tier1",
        "n_tier2",
        "n_tier3",
        "n_tier4",
        "aperture_correction",
        "qa_degraded",
        "qa_degraded_reason",
    }
)









# ---------------------------------------------------------------------------
# Pomocne funkcie
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# KROK 1: Globalna fixna apertura z PSF FWHM (MASTERSTAR VY_FWHM alebo fit)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# KROK 2: Aperturna fotometria per snimka - medianovy sky
# ---------------------------------------------------------------------------




def _clamp_err_empty_apertures_min(n: int) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = 16
    return max(1, min(256, v))




def _robust_scatter_mad(values: np.ndarray) -> float:
    """Sigma-clipped MAD scatter (Labbe et al. 2003 empty-aperture convention)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 0:
        return 0.0
    return mad / _MAD_CONSISTENCY


def _build_star_exclusion_mask(
    shape: tuple[int, ...],
    star_x: np.ndarray,
    star_y: np.ndarray,
    exclusion_radius_px: float,
    edge_margin_px: float,
) -> np.ndarray:
    """Boolean mask: True where empty apertures must not be placed."""
    ny, nx = int(shape[0]), int(shape[1])
    blocked = np.zeros((ny, nx), dtype=bool)
    em = int(math.ceil(max(0.0, float(edge_margin_px))))
    if em > 0:
        blocked[:em, :] = True
        blocked[-em:, :] = True
        blocked[:, :em] = True
        blocked[:, -em:] = True
    ex_r = float(exclusion_radius_px)
    if ex_r <= 0:
        return blocked
    ex_r2 = ex_r * ex_r
    xs = np.asarray(star_x, dtype=np.float64)
    ys = np.asarray(star_y, dtype=np.float64)
    ok = np.isfinite(xs) & np.isfinite(ys)
    # Canonical order: mask is OR-commutative, but keep draw/debug paths order-stable.
    order = np.lexsort((xs[ok], ys[ok]))
    xs_s = xs[ok][order]
    ys_s = ys[ok][order]
    for xi, yi in zip(xs_s, ys_s):
        x0 = max(0, int(math.floor(float(xi) - ex_r)) - 1)
        x1 = min(nx, int(math.ceil(float(xi) + ex_r)) + 2)
        y0 = max(0, int(math.floor(float(yi) - ex_r)) - 1)
        y1 = min(ny, int(math.ceil(float(yi) + ex_r)) + 2)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (xx - float(xi)) ** 2 + (yy - float(yi)) ** 2
        blocked[y0:y1, x0:x1] |= dist2 <= ex_r2
    return blocked


def _canonicalize_star_xy(
    star_x: np.ndarray,
    star_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Sort finite (x,y) pairs and return (xs, ys, sha256 hex of canonical list)."""
    import hashlib

    xs = np.asarray(star_x, dtype=np.float64).ravel()
    ys = np.asarray(star_y, dtype=np.float64).ravel()
    n = min(xs.size, ys.size)
    xs, ys = xs[:n], ys[:n]
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    if xs.size == 0:
        return xs, ys, hashlib.sha256(b"").hexdigest()
    order = np.lexsort((xs, ys))
    xs, ys = xs[order], ys[order]
    # Deduplicate exact duplicates after sort (stable membership).
    if xs.size > 1:
        keep = np.empty(xs.size, dtype=bool)
        keep[0] = True
        keep[1:] = (xs[1:] != xs[:-1]) | (ys[1:] != ys[:-1])
        xs, ys = xs[keep], ys[keep]
    blob = np.column_stack([xs, ys]).astype("<f8", copy=False).tobytes()
    return xs, ys, hashlib.sha256(blob).hexdigest()


def _labbe_debug_dump_enabled() -> bool:
    import os

    return str(os.environ.get("VYVAR_LABBE_DEBUG_DUMP", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _labbe_debug_dump_path() -> Path:
    import os

    raw = str(os.environ.get("VYVAR_LABBE_DEBUG_DUMP_PATH", "")).strip()
    if raw:
        return Path(raw)
    return Path("tmp") / "labbe_debug_dump.jsonl"


def _labbe_append_debug_record(record: dict[str, Any]) -> None:
    if not _labbe_debug_dump_enabled():
        return
    path = _labbe_debug_dump_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except OSError:
        pass












































LAST_EXCLUDED_TARGETS: pd.DataFrame = pd.DataFrame(
    columns=["name", "vsx_name", "vsx_type", "ra_deg", "dec_deg", "mag", "reason"]
)































# ---------------------------------------------------------------------------
# ALG-3: Temporal binning of comparison ensemble (MNRAS 2023)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# ALG-5: PyTICS iterative comp intercalibration (RASTI 2026)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# KROK 3: Stability check porovnavaciek (Abbeho p2p scatter + MAD)
# ---------------------------------------------------------------------------

# Observed-band / catalog mag before broad Gaia G for SNR-optimal aperture sizing.
_APERTURE_SIZING_MAG_COLS: tuple[str, ...] = (
    "mag",
    "catalog_mag",
    "lc_median_mag",
    "phot_g_mean_mag",
)











# ---------------------------------------------------------------------------
# KROK 4: Ensemble normalizacia
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# Color term (BP-RP) - globalny shift na noc
# ---------------------------------------------------------------------------





































_CT_PROTOTYPE_CSV_FIELDS: tuple[str, ...] = (
    "catalog_id",
    "vsx_name",
    "obs_group",
    "n_comp_used",
    "c1",
    "c1_stderr",
    "stderr_ratio",
    "target_bp_rp",
    "comp_med_bp_rp",
    "ct_corr",
    "cat_inst_scatter",
    "cat_inst_scatter_resid",
    "gate_would_pass",
)







# ---------------------------------------------------------------------------
# KROK 5: Outlier detekcia
# ---------------------------------------------------------------------------
















# ---------------------------------------------------------------------------
# KROK 6: Vystup - lightcurve CSV
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# KROK 6: Vystup - PNG grafy
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# Hlavny wrapper - run_phase2a
# ---------------------------------------------------------------------------


_EDGE_FILTER_NOTE_OK = ""
_EDGE_FILTER_NOTE_FAILED = "EDGE-UNFILTERED: edge safety check failed"












# PFS-SEMANTICS-01: never rescue these skip_reason values (TARGET-DEPTH-02 outranks PFS).
PFS_NEVER_RESCUE_REASONS = frozenset({"zone_noise", "below_target_depth"})
PFS_SATURATION_SKIP_REASONS = frozenset(
    {
        "zone_flag",
        "saturovany ciel",
        "per_frame_saturation",
        "likely_saturated",
        "saturated",
    }
)




















_LC_QUALITY_FLAGS: tuple[str, ...] = (
    "good",
    "noisy",
    "noisy_moon",
    "short_baseline",
    "no_data",
    "saturated",
)














_GIT_PROVENANCE_WARNED = False
# src_py/photometry_core.py -> repo root is parent.parent (git cwd + porcelain path base).
_REPO_ROOT_FOR_PROVENANCE = Path(__file__).resolve().parent.parent
































_ADAPTIVE_BLEND_CACHE: dict[str, dict[str, BlendMapEntry]] = {}






from mag_constants import MAG_ERR_SCALE

_PSF_ERR_MAG_SCALE = MAG_ERR_SCALE












def _get_lc_star_method(cid: str, all_frames: pd.DataFrame, star_method: str) -> np.ndarray:
    """Inst mag for one star using a fixed method for all frames (NaN if PSF missing)."""
    sub = all_frames[all_frames["catalog_id"] == cid]
    if sub.empty:
        return np.array([], dtype=float)
    if str(star_method).strip().lower() != "psf":
        return _get_lc(cid, all_frames)
    return _get_lc_psf_strict(cid, all_frames)





































# ======================================================================
# photometry.py (zlucene do photometry_core)
# ======================================================================

from utils import (
    fits_binning_xy_from_header,
    plate_scale_arcsec_per_pixel,
    plate_solve_fov_deg_diagonal_from_scale,
)











# Stlpce nacitavane z per-frame CSV pre bootstrap (78 % uspora pamate)
_PHASE_USECOLS_PERFRAME: list[str] = [
    "name",
    "catalog_id",
    "bjd_tdb_mid",
    "flux",
    "dao_flux",
    "noise_floor_adu",
    "sky_adu_per_px_annulus",
    "aperture_r_px",
    "is_usable",
    "is_saturated",
    "is_noisy",
    "snr50_ok",
    "vsx_known_variable",
    "likely_saturated",
]
















def _finite_pixel_bbox_from_array(
    data: "np.ndarray",
    *,
    finite_stride: int,
) -> tuple[float, float, float, float] | None:
    """Per-frame finite-pixel bbox from a 2-D array (strided sampling)."""
    import numpy as np

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        return None
    stride = max(1, int(finite_stride))
    samp = arr[::stride, ::stride]
    fin = np.isfinite(samp)
    if not bool(fin.any()):
        return None
    ys, xs = np.where(fin)
    x0 = float(xs.min() * stride)
    y0 = float(ys.min() * stride)
    x1 = float(min(arr.shape[1] - 1, xs.max() * stride + (stride - 1)))
    y1 = float(min(arr.shape[0] - 1, ys.max() * stride + (stride - 1)))
    return (x0, y0, x1, y1)


def _intersection_bbox_from_frame_bboxes(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if len(bboxes) < 2:
        return None
    x0_i, y0_i = 0.0, 0.0
    x1_i, y1_i = float("inf"), float("inf")
    for x0, y0, x1, y1 in bboxes:
        x0_i = max(x0_i, x0)
        y0_i = max(y0_i, y0)
        x1_i = min(x1_i, x1)
        y1_i = min(y1_i, y1)
    if not (math.isfinite(x0_i) and math.isfinite(y0_i) and math.isfinite(x1_i) and math.isfinite(y1_i)):
        return None
    if x1_i <= x0_i or y1_i <= y0_i:
        return None
    return (x0_i, y0_i, x1_i, y1_i)












def compute_auto_fwhm_limit(
    fwhm_values: np.ndarray | Sequence[float],
    k: float = 1.5,
) -> dict[str, Any]:
    """
    Vypocita automaticky FWHM limit pomocou MAD statistiky.

    Vracia dict:
        median_fwhm, mad, sigma_mad, auto_limit, k, n_total, n_kept, n_cut
    (``auto_limit`` moze byt ``None`` pri prilis malo bodoch.)
    """
    arr = np.asarray(fwhm_values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < 3:
        return {
            "median_fwhm": None,
            "mad": None,
            "sigma_mad": None,
            "auto_limit": None,
            "k": float(k),
            "n_total": int(len(arr)),
            "n_kept": int(len(arr)),
            "n_cut": 0,
        }
    median_f = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median_f)))
    sigma_mad = mad * 1.4826
    auto_limit = median_f + float(k) * sigma_mad
    n_kept = int(np.sum(arr <= auto_limit))
    return {
        "median_fwhm": round(median_f, 3),
        "mad": round(mad, 4),
        "sigma_mad": round(sigma_mad, 4),
        "auto_limit": round(float(auto_limit), 3),
        "k": float(k),
        "n_total": int(len(arr)),
        "n_kept": n_kept,
        "n_cut": int(len(arr) - n_kept),
    }






def _aperture_flux_sky_batch(
    d: np.ndarray,
    pos: np.ndarray,
    r_ap: float | np.ndarray,
    r_in: float | np.ndarray,
    r_out: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sky-subtracted circular aperture sums (photutils ``method='exact'``).

    Shared by production per-star photometry and the scatter/CoG radius ladder
    (IMPL-04). When all stars share one radius, uses one batch aperture call.
    """
    from photutils.aperture import CircularAnnulus, CircularAperture
    from photutils.aperture import aperture_photometry as _aphot

    pos = np.asarray(pos, dtype=np.float64)
    n = int(pos.shape[0])
    flux_arr = np.full(n, np.nan, dtype=np.float64)
    sky_pp_arr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return flux_arr, sky_pp_arr

    r_ap_arr = np.full(n, float(r_ap), dtype=np.float64) if np.isscalar(r_ap) else np.asarray(r_ap, dtype=np.float64)
    r_in_arr = np.full(n, float(r_in), dtype=np.float64) if np.isscalar(r_in) else np.asarray(r_in, dtype=np.float64)
    r_out_arr = np.full(n, float(r_out), dtype=np.float64) if np.isscalar(r_out) else np.asarray(r_out, dtype=np.float64)

    # Uniform radius: one batch exact sum.
    if (
        np.all(np.isfinite(r_ap_arr))
        and np.all(r_ap_arr > 0)
        and float(np.nanmax(r_ap_arr) - np.nanmin(r_ap_arr)) < 1e-9
        and np.all(np.isfinite(r_in_arr))
        and np.all(np.isfinite(r_out_arr))
        and float(np.nanmax(r_in_arr) - np.nanmin(r_in_arr)) < 1e-9
        and float(np.nanmax(r_out_arr) - np.nanmin(r_out_arr)) < 1e-9
        and float(r_out_arr[0]) > float(r_in_arr[0]) > 0
    ):
        r0 = float(r_ap_arr[0])
        rin = float(r_in_arr[0])
        rout = float(r_out_arr[0])
        try:
            ap = CircularAperture(pos, r=r0)
            phot = _aphot(d, ap, method="exact")
            sums = np.asarray(phot["aperture_sum"], dtype=np.float64)
            area = float(ap.area)
            an = CircularAnnulus(pos, r_in=rin, r_out=rout)
            masks = an.to_mask(method="center")
            if not isinstance(masks, (list, tuple)):
                masks = [masks]
            for i, m in enumerate(masks):
                try:
                    ann_img = m.to_image(d.shape)
                    sky_pp_arr[i] = _sky_pp_from_annulus_image(d, ann_img)
                except Exception:  # noqa: BLE001
                    sky_pp_arr[i] = float("nan")
            flux_arr = sums - sky_pp_arr * area
            return flux_arr, sky_pp_arr
        except Exception as exc:  # noqa: BLE001
            logging.debug("[PHOT] batch exact aperture failed, falling back per-star: %s", exc)

    # Heterogeneous radii: per-star exact (photutils 2.3 scalar-r constraint).
    return _aperture_flux_sky_per_star(d, pos, r_ap_arr, r_in_arr, r_out_arr)




def compute_per_frame_cog_correction(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dao_flux: np.ndarray,
    aperture_r_px: np.ndarray,
    sky_pp: np.ndarray,
    *,
    fwhm_px: float,
    peak_max_adu: np.ndarray | None = None,
    sat_limit_adu: np.ndarray | None = None,
    ref_fwhm: float = 4.5,
    ladder_step_px: float = 0.5,
    min_stars: int = 8,
    isolation_fwhm: float = 6.0,
    snr_min: float = 50.0,
    sat_frac: float = 0.85,
    gain: float = 1.0,
    read_noise: float = 10.0,
    ac_factor_max: float = 5.0,
    max_stars: int = 60,
    fallback_ee: tuple[np.ndarray, np.ndarray] | None = None,
    ladder_outer_factor: float = 1.0,
) -> dict[str, Any]:
    """Per-frame curve-of-growth (encircled-energy) aperture correction.

    Builds an EE(r) curve (normalised to ``ref_fwhm x FWHM``) from bright, isolated,
    unsaturated, high-SNR stars and returns a per-star multiplicative correction
    ``ac_factor = 1 / EE(r_star)`` that puts every star on the common ref-radius
    enclosed-flux scale (removing the per-star SNR-radius differential bias).

    ``ladder_outer_factor`` > 1 extends the ladder past the normalisation radius so a
    real flatness check can compare EE(outer) to EE(ref) (IMPL-02). Normalisation
    remains at ``ref_r``, not at the last ladder point.

    Returns dict: ``ac_factor`` (len n, >=1.0), ``cog_ok``, ``n_cog``, ``ref_r_px``,
    ``ee_radii``, ``ee_curve``, ``flatness_outer_over_norm``. When fewer than
    ``min_stars`` COG stars are found and no ``fallback_ee`` is given, ``cog_ok=False``
    and every ``ac_factor=1.0``.
    """
    from photutils.aperture import CircularAperture
    from photutils.aperture import aperture_photometry as _aphot

    n = int(len(x))
    out: dict[str, Any] = {
        "ac_factor": np.ones(n, dtype=np.float64),
        "cog_ok": False,
        "n_cog": 0,
        "ref_r_px": float(ref_fwhm) * float(fwhm_px) if math.isfinite(fwhm_px) else float("nan"),
        "ee_radii": None,
        "ee_curve": None,
        "flatness_outer_over_norm": float("nan"),
        "ladder_outer_r_px": float("nan"),
    }
    if n == 0 or not (math.isfinite(fwhm_px) and fwhm_px > 0):
        return out

    d = np.asarray(data, dtype=np.float64)
    if np.any(~np.isfinite(d)):
        fill = float(np.nanmedian(d)) if np.any(np.isfinite(d)) else 0.0
        d = np.where(np.isfinite(d), d, fill)
    height, width = d.shape

    ref_r = float(ref_fwhm) * float(fwhm_px)
    try:
        outer_factor = float(ladder_outer_factor)
    except (TypeError, ValueError):
        outer_factor = 1.0
    if not math.isfinite(outer_factor) or outer_factor < 1.0:
        outer_factor = 1.0
    outer_r = float(outer_factor) * ref_r
    # Isolation uses the configured multiple of FWHM (do not force >= ref_r).
    iso_r = float(isolation_fwhm) * float(fwhm_px)
    step = float(ladder_step_px) if math.isfinite(ladder_step_px) and ladder_step_px > 0 else 0.5
    radii = np.arange(step, outer_r + 1e-6, step, dtype=np.float64)
    if radii.size == 0 or radii[-1] < outer_r - 1e-6:
        radii = np.append(radii, outer_r)
    # Ensure both normalisation and outer radii are exact ladder points.
    if not np.any(np.isclose(radii, ref_r, rtol=0.0, atol=1e-6)):
        radii = np.sort(np.append(radii, ref_r))
    if not np.any(np.isclose(radii, outer_r, rtol=0.0, atol=1e-6)):
        radii = np.sort(np.append(radii, outer_r))
    i_norm = int(np.argmin(np.abs(radii - ref_r)))
    radii[i_norm] = ref_r
    out["ladder_outer_r_px"] = float(radii[-1])

    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    flux = np.asarray(dao_flux, dtype=np.float64)
    rap = np.asarray(aperture_r_px, dtype=np.float64)
    skp = np.asarray(sky_pp, dtype=np.float64)

    finite_xy = np.isfinite(xx) & np.isfinite(yy)
    nn = np.full(n, np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        pts = np.column_stack([xx[finite_xy], yy[finite_xy]])
        if pts.shape[0] >= 2:
            tree = cKDTree(pts)
            dist, _ = tree.query(pts, k=2)
            nn[finite_xy] = dist[:, 1]
    except (ImportError, AttributeError, ValueError, TypeError):
        logging.debug("[COG] cKDTree unavailable - isolation check skipped")

    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    rn = float(read_noise) if math.isfinite(read_noise) and read_noise >= 0 else 10.0
    area_rap = math.pi * np.square(np.where(np.isfinite(rap) & (rap > 0), rap, np.nan))
    var = flux / g + np.maximum(0.0, skp) / g * area_rap + (rn / g) ** 2 * area_rap
    snr = np.where((flux > 0) & np.isfinite(var) & (var > 0), flux / np.sqrt(var), 0.0)

    if peak_max_adu is not None and sat_limit_adu is not None:
        pk = np.asarray(peak_max_adu, dtype=np.float64)
        sl = np.asarray(sat_limit_adu, dtype=np.float64)
        unsat = ~(np.isfinite(pk) & np.isfinite(sl) & (pk > float(sat_frac) * sl))
    else:
        unsat = np.ones(n, dtype=bool)

    margin = float(radii[-1]) + 1.0
    in_bounds = (xx > margin) & (xx < (width - margin)) & (yy > margin) & (yy < (height - margin))

    sel = (
        finite_xy
        & (flux > 0)
        & np.isfinite(skp)
        & unsat
        & in_bounds
        & (nn > iso_r)
        & (snr >= float(snr_min))
    )
    cog_idx = np.where(sel)[0]
    # Cap to the highest-SNR subset - a robust median EE needs only a few dozen stars.
    if int(max_stars) > 0 and cog_idx.size > int(max_stars):
        order = np.argsort(snr[cog_idx])[::-1][: int(max_stars)]
        cog_idx = cog_idx[order]

    fracs: list[np.ndarray] = []
    for i in cog_idx:
        try:
            xy = [(float(xx[i]), float(yy[i]))]
            sums = np.array(
                [
                    float(_aphot(d, CircularAperture(xy, r=float(rr)), method="exact")["aperture_sum"][0])
                    for rr in radii
                ],
                dtype=np.float64,
            )
            ee = sums - float(skp[i]) * math.pi * np.square(radii)
            ref_val = float(ee[i_norm])
            if math.isfinite(ref_val) and ref_val > 0:
                fr = ee / ref_val
                if np.all(np.isfinite(fr)):
                    fracs.append(fr)
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0190] One bright star skipped in COG encircled-energy curve - aperture COG correction biased: %s', exc)
            continue

    n_cog = len(fracs)
    out["n_cog"] = n_cog
    ee_radii = radii
    ee_curve: np.ndarray | None = None
    if n_cog >= int(min_stars):
        ee_curve = np.median(np.vstack(fracs), axis=0)
        ee_curve = np.clip(ee_curve, 1e-3, None)
        # Renormalise so EE(ref)=1 exactly after the median (outer may exceed 1 slightly).
        norm_v = float(ee_curve[i_norm])
        if math.isfinite(norm_v) and norm_v > 0:
            ee_curve = ee_curve / norm_v
        ee_curve[i_norm] = 1.0
        out["cog_ok"] = True
        out["flatness_outer_over_norm"] = float(ee_curve[-1])
    elif fallback_ee is not None:
        ee_radii, ee_curve = fallback_ee
        ee_radii = np.asarray(ee_radii, dtype=np.float64)
        ee_curve = np.asarray(ee_curve, dtype=np.float64)
        out["cog_ok"] = False  # fallback used; flag as not-fresh
    else:
        return out  # too few COG stars and no fallback -> no correction (ac_factor=1)

    out["ee_radii"] = ee_radii
    out["ee_curve"] = ee_curve

    ee_at = np.interp(np.clip(rap, ee_radii[0], ee_radii[-1]), ee_radii, ee_curve)
    acf = np.where((ee_at > 0) & np.isfinite(ee_at), 1.0 / ee_at, 1.0)
    acf = np.clip(acf, 1.0, float(ac_factor_max))
    acf = np.where(np.isfinite(rap) & (rap > 0), acf, 1.0)
    out["ac_factor"] = acf
    return out
















































def run_full_photometry_pipeline(
    *,
    masterstar_fits_path: Path,
    variable_targets_csv: Path,
    masterstars_csv: Path,
    per_frame_csv_dir: Path,
    detrended_aligned_dir: Path,
    output_dir: Path,
    cfg: AppConfig | None = None,
    db: Any = None,
    draft_id: int | None = None,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Jedno-krokovy wrapper: Faza 0+1 + Faza 2A ako jeden celok.

    UI to pouziva ako jednu akciu 'RUN Aperture Photometry' pre dany obs_group.
    """
    _cfg = cfg or AppConfig()

    ensure_full_variable_targets_if_presel_stub(
        variable_targets_csv=Path(variable_targets_csv),
        masterstars_csv=Path(masterstars_csv),
        masterstar_fits=Path(masterstar_fits_path),
        cfg=_cfg,
        draft_id=draft_id,
    )

    def _p(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    # FWHM: prefer header (VY_FWHM_GAUSS/VY_FWHM), inak default z configu.
    fwhm_px = _resolve_sips_dao_fwhm_px(_cfg, fwhm_px=None)
    _ms_header_shared: Any | None = None
    _ms_path_shared = Path(masterstar_fits_path)
    if _ms_path_shared.is_file():
        try:
            from astropy.io import fits as astrofits  # noqa: PLC0415

            with astrofits.open(_ms_path_shared, memmap=False) as _hdul:
                _ms_header_shared = _hdul[0].header.copy()
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0220] Shared MASTERSTAR header cache load fails - repeated FITS opens (perf), not science num...: %s', exc)
            logging.warning("[PERF-2] Cannot open MASTERSTAR.fits for header: %s", exc)
    if _ms_header_shared is not None:
        try:
            # Prefer night seeing (VY_FWHM) over Gaussian-core (VY_FWHM_GAUSS) for
            # Phase-1 isolation / blend geometry. Core FWHM under-states the CoG
            # 3-FWHM single-source radius (COMP-ASSIGN-03 / A-1).
            for key in ("VY_FWHM", "VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN"):
                v = _ms_header_shared.get(key)
                if v is None:
                    continue
                fv = float(v)
                if 0.5 < fv < 30.0:
                    fwhm_px = fv
                    break
        except Exception as exc:  # noqa: BLE001
            logging.error('[EXC-0221] VY_FWHM/VY_FWHM_GAUSS header parse fails - pipeline uses default/config FWHM for phase0+1: %s', exc)
            pass

    # -- FAZA 0+1 --
    _p("Faza 0+1: select targets + comparison stars...")
    _plate_scale = _get_plate_scale_from_cfg(
        _cfg,
        db=db,
        draft_id=draft_id,
        fits_path=Path(masterstar_fits_path),
        ms_header=_ms_header_shared,
    )
    if _plate_scale is None:
        _plate_scale = _read_plate_scale_from_fits_path(
            Path(masterstar_fits_path),
            ms_header=_ms_header_shared,
        )
        if _plate_scale is not None and math.isfinite(float(_plate_scale)) and float(_plate_scale) > 0:
            logging.info(
                "[FOV] plate_scale from MASTERSTAR.fits header -> %.4f arcsec/px",
                float(_plate_scale),
            )
    _fw_pipe, _fh_pipe, _frame_hw_src = _resolve_frame_hw_px_from_masterstar(
        Path(masterstar_fits_path),
        frame_w_px=int(_cfg.frame_width_px),
        frame_h_px=int(_cfg.frame_height_px),
        db=db,
        draft_id=draft_id,
    )
    if _frame_hw_src != "caller_default":
        logging.info(
            "[PHASE 0+1] Pipeline frame dimensions %dx%d px from %s",
            int(_fw_pipe),
            int(_fh_pipe),
            _frame_hw_src,
        )
    _pt_mags = _cfg.phase01_tier_mags()
    p01 = run_phase0_and_phase1(
        variable_targets_csv=Path(variable_targets_csv),
        masterstars_csv=Path(masterstars_csv),
        per_frame_csv_dir=Path(per_frame_csv_dir),
        output_dir=Path(output_dir),
        fwhm_px=float(fwhm_px),
        frame_w_px=int(_fw_pipe),
        frame_h_px=int(_fh_pipe),
        chip_interior_margin_px=_resolve_chip_interior_margin_px(_cfg, arcsec_per_px=_plate_scale),
        plate_scale_arcsec_px=_plate_scale,
        max_dist_deg=_compute_fov_max_dist(
            frame_w_px=int(_fw_pipe),
            frame_h_px=int(_fh_pipe),
            plate_scale=_plate_scale,
            fov_fraction=float(_cfg.phase01_comparison_fov_fraction),
            fallback_deg=resolve_max_dist_fallback_deg(
                _cfg,
                frame_w_px=int(_fw_pipe),
                frame_h_px=int(_fh_pipe),
                plate_scale_arcsec_px=_plate_scale,
            ),
        ),
        max_mag_diff=float(_cfg.phase01_comparison_max_mag_diff),
        comp_max_delta_bprp=float(_cfg.comp_max_delta_bprp),
        max_mag_diff_t1=float(_pt_mags[0]),
        max_mag_diff_t2=float(_pt_mags[1]),
        max_mag_diff_t3=float(_pt_mags[2]),
        max_mag_diff_t4=float(_pt_mags[3]),
        n_comp_min=int(_cfg.phase01_comparison_n_comp_min),
        n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
        max_comp_rms=float(_cfg.phase01_comparison_max_comp_rms),
        min_dist_arcsec=float(_cfg.phase01_comparison_min_dist_arcsec),
        min_frames_frac=float(_cfg.phase01_comparison_min_frames_frac),
        rms_outlier_sigma=3.0,
        exclude_gaia_nss=bool(_cfg.phase01_comparison_exclude_gaia_nss),
        exclude_gaia_extobj=bool(_cfg.phase01_comparison_exclude_gaia_extobj),
        mag_bright_threshold=float(_cfg.phase01_comparison_mag_bright_threshold),
        max_mag_diff_bright_floor=float(
            _cfg.phase01_comparison_max_mag_diff_bright_floor or 0.0
        ),
        max_psf_chi2=float(_cfg.phase01_comparison_max_psf_chi2),
        max_fwhm_factor=float(_cfg.phase01_comparison_max_fwhm_factor),
        isolation_radius_px=_resolve_isolation_radius_px(_cfg, arcsec_per_px=_plate_scale),
        flux_col=_cfg.phase01_flux_col,
        cfg=_cfg,
        progress_cb=progress_cb,
        draft_id=draft_id,
        db=db,
    )

    active_targets_csv = Path(str(p01.get("active_targets_csv") or ""))
    comparison_stars_csv = Path(str(p01.get("comparison_stars_csv") or ""))
    n_active = int(p01.get("n_active_targets") or 0)
    if n_active <= 0:
        return {
            "phase01": p01,
            "phase2a": None,
            "output_dir": str(Path(output_dir)),
            "zero_targets": True,
            "n_active_targets": 0,
        }
    if not active_targets_csv.is_file() or not comparison_stars_csv.is_file():
        return {
            "phase01": p01,
            "phase2a": None,
            "output_dir": str(Path(output_dir)),
            "error": "Faza 0+1 nevygenerovala active_targets/comparison_stars CSV.",
        }

    # INV-DAG-01: phase01 stamp after successful Phase 0+1.
    try:
        from invariants_runtime import stamp_stage_on_disk  # noqa: PLC0415

        stamp_stage_on_disk(Path(output_dir), "phase01", enforce_upstream=True)
    except Exception as _dag_p01_exc:  # noqa: BLE001
        logging.debug("[INV-DAG-01] phase01 stamp skipped: %s", _dag_p01_exc)

    # -- FAZA 2A --
    _p("Faza 2A: aperture photometry + lightcurves...")
    _cfg2a = p01.get("cfg_effective_for_photometry") or _cfg
    p2a = run_phase2a(
        masterstar_fits_path=Path(masterstar_fits_path),
        active_targets_csv=active_targets_csv,
        comparison_stars_csv=comparison_stars_csv,
        per_frame_csv_dir=Path(per_frame_csv_dir),
        detrended_aligned_dir=Path(detrended_aligned_dir),
        output_dir=Path(output_dir),
        fwhm_px=float(fwhm_px),
        annulus_inner_fwhm=float(_cfg.annulus_inner_fwhm),
        annulus_outer_fwhm=float(_cfg.annulus_outer_fwhm),
        cfg=_cfg2a,
        progress_cb=progress_cb,
        db=db,
        draft_id=draft_id,
        proc_frame_store=p01.get("proc_store"),
    )

    sysrem_result: dict[str, Any] | None = None
    if bool(_cfg.sysrem_enabled):
        _p("SysRem: removing systematic trends...")
        _sysrem_lc_dir = Path(output_dir) / "lightcurves"
        sysrem_result = run_sysrem_field(
            _sysrem_lc_dir,
            n_iter=int(_cfg.sysrem_n_iter),
        )
        logging.info(
            "[SysRem] %d stars | RMS improvement %.1f%% (%d iter)",
            int(sysrem_result.get("n_stars", 0)),
            float(sysrem_result.get("rms_improvement_pct", float("nan"))),
            int(sysrem_result.get("n_iter", 0)),
        )

    return {
        "phase01": p01,
        "phase2a": p2a,
        "sysrem": sysrem_result,
        "output_dir": str(Path(output_dir)),
        "proc_frame_store": p01.get("proc_store"),
    }




__all__ = [
    # photometry (legacy)
    "StressTestResult",
    "_get_lc_adaptive",
    "apply_reporting_postprocess",
    "check_comparison_stability",
    "common_field_intersection_bbox_px",
    "compute_aperture_correction",
    "compute_fwhm_gaussian_for_aperture_catalog",
    "compute_lc_rms_ooe",
    "compute_mag_calib_final",
    "compute_optimal_apertures",
    "detect_outliers",
    "empirical_feature_mask_mag",
    "enhance_catalog_dataframe_aperture_bpm",
    "ensemble_normalize",
    "ensure_full_variable_targets_if_presel_stub",
    "load_epsf_metrics_for_draft",
    # photometry_phase2a (legacy)
    "measure_fwhm_from_masterstar",
    "pytics_iterative_weights",
    "read_flux_from_csv",
    "recommended_aperture_by_color",
    "resolve_apply_color_term",
    "run_full_photometry_pipeline",
    "run_phase0_and_phase1",
    "run_phase2a",
    "run_sysrem_field",
    "save_cutout_png",
    "save_field_map_png",
    "save_lightcurve_csv",
    "save_lightcurve_png",
    "save_target_field_map_png",
    "select_active_targets",
    "select_comparison_stars_per_target",
    "stress_test_relative_rms_from_sidecars",
    "vsx_is_known_variable_top3_per_bin",
]

from photometry_ui_helpers import (  # noqa: E402,F401
    resolve_lc_time_base,
    lc_time_axis_short_label,
)

from photometry_shared import (  # noqa: E402,F401
    _safe_polyfit,
    _normalize_gaia_id,
    finalize_hybrid_bkg_fallback_proc_dir,
    stamp_masterstar_snr_columns,
    _target_display_name,
    stamp_vsx_known_variable_on_masterstars,
    build_gs11_summary,
    _get_lc_adaptive,
    _get_plate_scale_from_cfg,
    _resolve_plate_scale_arcsec_per_px,
    _cd_matrix_scale_arcsec_per_px,
    _read_plate_scale_from_fits_path,
    _angular_distance_deg,
    StressTestResult,
    stress_test_relative_rms_from_sidecars,
    vsx_is_known_variable_top3_per_bin,
    common_field_intersection_bbox_px_from_arrays,
    common_field_intersection_bbox_px,
    recommended_aperture_by_color,
    bad_columns_for_light_frame,
    _fwhm_moment_at,
    compute_fwhm_gaussian_for_aperture_catalog,
    enhance_catalog_dataframe_aperture_bpm,
)

import sys as _sys_e4
_p2a_mod = _sys_e4.modules.get("photometry_phase2a")
if _p2a_mod is None or getattr(_p2a_mod, "run_phase2a", None) is not None:
    from photometry_phase2a import (  # noqa: E402,F401
    parse_comp_quality_json_map,
    _build_csv_lookup,
    _lookup_star_in_csv,
    _sat_limit_peak_adu,
    _mad_sigma_or_std_floor,
    measure_fwhm_from_masterstar,
    compute_optimal_apertures,
    _howell_variance_adu2,
    _photometric_error,
    _photometric_error_with_bkg_mode,
    _phase2a_proc_column_requirements,
    _phase2a_cache_columns,
    _phase2a_empirical_sigma_bkg_ap,
    _sky_pp_for_photometric_error,
    _resolve_phase2a_equipment_id,
    _draft_dir_from_phase2a_paths,
    _require_comparison_stars_per_target_schema,
    _median_sky_from_phase2a_csv_cache,
    _measured_aperture_from_proc_cache,
    _resolve_photometric_aperture_px_for_gs11,
    read_flux_from_csv,
    compute_aperture_correction,
    fit_color_term_c1,
    should_apply_color_term,
    _obs_group_filter_key,
    resolve_apply_color_term,
    _ColorTermGroupFit,
    _group_comp_mag_inst_from_flux_matrix,
    _group_comp_mag_inst_from_proc_csvs,
    _comp_maps_from_comparison_stars_csv,
    _phase2a_attempt_k2_night_fit,
    _compute_group_color_term_fit,
    _ensure_group_comp_pool_csv,
    _target_row_is_vsx_known_variable,
    empirical_feature_mask_mag,
    detect_outliers,
    apply_reporting_postprocess,
    democratic_detrend_lc,
    save_field_map_png,
    _edge_ok_from_masterstar_pipeline,
    resolve_variable_targets_csv,
    auto_export_variability_candidates_csv,
    _phase2a_coerce_skip_photometry,
    build_rms_mag_model,
    expected_rms_from_model,
    classify_lc_quality,
    build_lc_quality_summary,
    _phase2a_write_summary,
    _phase2a_observer_location_dict,
    _sky_surface_meta_from_qc,
    _phase2a_resolve_field_center_ra_dec,
    _phase2a_collect_session_jd_values,
    _Phase2AState,
    _build_phase2a_dynamic_params,
    _phase2a_compute_lunar_context,
    _preserve_nondetection_flags_helper,
    _proc_stem,
    _compute_frame_align_residuals,
    _record_align_residuals_to_report,
    _frame_align_residual_gate_select,
    _propagate_phase2a_skip_reason_to_active,
    _phase2a_finalize_exports,
        run_phase2a,
    )

from photometry_lightcurve import (  # noqa: E402,F401
    _ac_summary_fields,
    _phase2a_empty_comp_summary_row,
    _phase2a_skip_empty_comps_target,
    _coerce_bool_cell,
    _frame_has_usable_cog,
    evaluate_cog_night_apcorr_gate,
    temporal_bin_comp_lc,
    pytics_iterative_weights,
    _common_mode_detrend_comp_lc,
    _comp_lc_frame_ensemble_residual,
    compute_lc_rms_ooe,
    check_comparison_stability,
    ensemble_normalize,
    _ensemble_scatter_by_source_file,
    _combine_err_with_ensemble_scatter_keyed,
    _err_budget_components_keyed,
    _exclude_err_scatter_unmatched_epochs,
    ct_ensemble_reference_maps,
    apply_color_term,
    _check_color_term_extrapolation,
    _ct_prototype_enabled,
    _color_term_cat_inst_scatter_pair,
    _append_ct_prototype_row,
    savgol_detrend_lc,
    compute_mag_calib_final,
    save_lightcurve_csv,
    save_lightcurve_png,
    save_cutout_png,
    save_target_field_map_png,
    pfs_rescue_eligible,
    _keep_recorded_skip_reason,
    decide_target_saturation_policy,
    _per_frame_sat_flags_for_catalog_id,
    _resolve_pfs_peak_test,
    apply_per_frame_saturation_to_active_targets,
    _fits_header_facts,
    _build_phase2a_resolved_facts,
    BlendMapEntry,
    _load_blend_worklist,
    _load_adaptive_blend_map,
    _route_lc_per_frame_err,
    _get_lc,
    _get_comp_bjd_series,
    compute_lc_flux_method,
    _recompute_bjd_hjd_with_status,
    run_sysrem_field,
)

import photometry_shared as _photometry_shared_e4c  # noqa: E402
_photometry_shared_e4c._coerce_bool_cell = _coerce_bool_cell


from photometry_gate_helpers import (  # noqa: E402,F401
    _sigma_bkg_r_key,
    _assert_inv_err_sigma_acct_01,
    comp_quality_quality_strings,
    _clamp_err_empty_apertures_n,
    _normalize_err_background_mode,
    _labbe_content_seed_from_header,
    measure_empty_aperture_sigma_bkg,
    estimate_star_free_per_pixel_variance_adu2,
    _howell_bkg_variance_adu2,
    _clamp_bkg_scale_r,
    bkg_scale_ratio_empirical_over_howell,
    compute_setup_bkg_scale_r,
    scaled_sigma_bkg_ap_from_howell,
    measure_growth_curve_ee,
    _phase2a_star_mag_lookup,
    discover_aligned_science_fits,
    _median_bkg_var_from_aligned_frames,
    _estimate_annulus_sky_pp,
    _annulus_sky_subtracted_flux,
    _resolve_star_flux_method,
    _frame_quality_gate_select,
    _recompute_bjd_hjd_per_target,
    photometer_check_star_production_path,
    _compute_fov_max_dist,
    _sky_pp_from_annulus_image,
    _aperture_flux_sky_per_star,
)

import photometry_shared as _photometry_shared  # noqa: E402
_photometry_shared._assert_inv_err_sigma_acct_01 = _assert_inv_err_sigma_acct_01
_photometry_shared._clamp_err_empty_apertures_n = _clamp_err_empty_apertures_n
_photometry_shared._labbe_content_seed_from_header = _labbe_content_seed_from_header
_photometry_shared._sigma_bkg_r_key = _sigma_bkg_r_key
_photometry_shared._sky_pp_from_annulus_image = _sky_pp_from_annulus_image
_photometry_shared.bkg_scale_ratio_empirical_over_howell = (
    bkg_scale_ratio_empirical_over_howell
)
_photometry_shared.compute_setup_bkg_scale_r = compute_setup_bkg_scale_r
_photometry_shared.measure_empty_aperture_sigma_bkg = measure_empty_aperture_sigma_bkg
_photometry_shared.scaled_sigma_bkg_ap_from_howell = scaled_sigma_bkg_ap_from_howell

from photometry_exports import (  # noqa: E402,F401
    lc_has_finite_airmass,
    apply_comp_w_rel_for_display,
    ensemble_member_ids,
    _get_lc_psf_strict,
    _get_lc_adaptive_per_star,
)

from epsf_hooks import load_epsf_metrics_for_draft  # noqa: E402,F401

from photometry_provenance import (  # noqa: E402,F401
    _is_import_relevant_py_path,
    _porcelain_status_by_path,
    classify_git_dirty_paths,
    _resolve_git_provenance,
    _json_safe_snapshot_value,
    _complete_config_snapshot,
    _build_pipeline_provenance_block,
    merge_photometry_pipeline_meta,
)

import photometry_provenance as _photometry_provenance  # noqa: E402
_photometry_provenance._resolve_git_provenance = (
    lambda *a, **k: _resolve_git_provenance(*a, **k)
)

from photometry_comp import (  # noqa: E402,F401
    _sid_int,
    _enrich_comp_bp_rp,
    _ensure_active_target_display_names,
    _variable_targets_looks_like_ct_presel_stub,
    ensure_full_variable_targets_if_presel_stub,
    _normalize_id_value,
    _normalize_id_series,
    _bool_col,
    _phase0_effective_frame_hw_px,
    _active_target_zone_flag,
    _auto_repair_catalog_ids,
    _enrich_active_targets_bp_rp,
    _resolve_frame_hw_px_from_masterstar,
    _read_field_density_inputs,
    _refresh_variable_targets_xy,
    _attach_predicted_dilution_report,
    select_active_targets,
    _batch_enrich_targets_bp_rp_from_gaia_db,
    _enrich_target_bp_rp_from_gaia_db,
    _bprp_tier_ladder_for_selection,
    _select_comps_by_rms_then_color,
    _select_comps_by_color_then_rms,
    _select_comps_tiered,
    build_global_comp_pool,
    _dedupe_comp_pool_by_gaia_key,
    _warn_zero_compstars_edge,
    _count_gate_passing_comps,
    select_comparison_stars_per_target,
    _write_suspected_variables,
)

import functools
import sys
import photometry_comp as _photometry_comp  # noqa: E402
_photometry_comp._enrich_active_targets_bp_rp = (
    lambda *a, **k: _enrich_active_targets_bp_rp(*a, **k)
)
_photometry_comp._ensure_active_target_display_names = (
    lambda *a, **k: _ensure_active_target_display_names(*a, **k)
)
_e3_select_active_targets = _photometry_comp.select_active_targets

@functools.wraps(_e3_select_active_targets)
def select_active_targets(*a, **k):
    out = _e3_select_active_targets(*a, **k)
    global LAST_EXCLUDED_TARGETS
    LAST_EXCLUDED_TARGETS = _photometry_comp.LAST_EXCLUDED_TARGETS
    _p01 = sys.modules.get("phase01_run")
    if _p01 is not None:
        _p01.LAST_EXCLUDED_TARGETS = LAST_EXCLUDED_TARGETS
    return out

_photometry_comp.select_active_targets = select_active_targets

from phase01_run import run_phase0_and_phase1  # noqa: E402,F401
