"""Moved from pipeline.py (CONSOLIDATE-01E6b). Facade re-exports this name.

detect_stars_and_match_catalog: DAOStarFinder + WCS + local Gaia match.
pipeline.py re-exports this name; external callers untouched.
"""
from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import astropy.units as u
import pandas as pd

from plain_stats import plain_mean_med_std, sky_mad_sigma_adu
from utils import (
    DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    dao_detection_fwhm_pixels,
)
from masterstar_gaia_accounting import _dao_xy_binned_to_full
from config import AppConfig
from infolog import log_event
from pipeline_constants import (
    DAO_FWHM_FALLBACK_PX,
    EXOPLANET_MATCH_MAX_SEP_ARCSEC,
    LINEARITY_FRAC,
    MS_DAO_CENTROID_QA_CAP_PX,
    MS_DAO_CENTROID_QA_FLOOR_PX,
    MS_DAO_MATCH_RADIUS_K,
    MS_DAO_PASS1_DEDUP_PX,
    MS_DAO_PASS2_SIGMA,
    MS_GAIA_CENSUS_EDGE_MARGIN_PX,
)
from photometry_shared import stamp_masterstar_snr_columns
from pipeline_calibrate import _effective_saturation_limit
from proc_frame_store import _vyvar_df_to_csv
from pipeline_catalog import (
    _all_pix2world_icrs_deg,
    _box_peaks_at_centroids,
    _chord_to_arcsec,
    _dao_auto_binning_factor,
    _dao_convolved_background_rms_adu,
    _dao_spatial_flux_cap_row_indices,
    _detect_empirical_clip_level_adu,
    _effective_field_catalog_cone_radius_deg,
    _exo_host_annotation_arrays,
    _gaia_chip_xy_from_catalog,
    _icrs_deg_to_unitxyz,
    _bin2d_mean,
    _prefilter_dao_table_brightest,
    _proc_rename_det_names_to_catalog_id,
    _proc_sat_block_for_csv,
    _query_exoplanet_local,
    _query_gaia_local,
    _query_vsx_local,
    _resolve_peak_saturation_limit_adu,
    _slice_exo_annotation,
    _vectorized_star_saturation_columns,
    _write_field_catalog_cone_meta,
    build_ucac_catalog_kdtree,
)
from pipeline_constants import CATALOG_MATCH_SEP_ARCSEC, MASTERSTAR_MAX_DAO_ROWS  # noqa: E402

LOGGER = logging.getLogger("pipeline")


#: IDENTITY-2026-10: a Gaia star more than this fainter than the detected flux implies is no
#: identity candidate (2 mag keeps large-amplitude variables near maximum).
IDENTITY_MAX_FAINTER_MAG = 2.0


def _flux_g_zero_point(df_ident: pd.DataFrame, det_flux: Any, use: Any = None) -> float:
    """Frame zero point G = ZP - 2.5 log10(DAO flux) from the identified stars (robust median);
    ``use`` limits it to one flux scale (pass-1 DAO) and non-variables. NaN with fewer than 20."""
    import numpy as np

    try:
        f = np.asarray(det_flux, dtype=np.float64)
        g = pd.to_numeric(df_ident.get("mag"), errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(f) & (f > 0) & np.isfinite(g)
        if use is not None:
            ok &= np.asarray(use, dtype=bool)
        if int(ok.sum()) < 20:
            return float("nan")
        return float(np.median(g[ok] + 2.5 * np.log10(f[ok])))
    except Exception:  # noqa: BLE001
        return float("nan")


def detection_on_chip(x: Any, y: Any, *, wpx: int, h: int) -> "np.ndarray":
    """A detection is a star on the chip: DAO edge fits can put a centroid outside the image
    (532: x = -0.4 got a G 17.3 identity and its forced position broke the aperture photometry
    of every frame). The DAO flux sign is NOT a criterion: it is summed against the global sky,
    so real faint stars on a brighter local background have DAO flux <= 0 (532: 1 671 such
    detections, 95 % with positive aperture flux, 867 with good identities)."""
    import numpy as np

    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    return (
        np.isfinite(xa) & np.isfinite(ya) & (xa >= 0.0) & (ya >= 0.0)
        & (xa <= float(wpx) - 1.0) & (ya <= float(h) - 1.0)
    )


def _dm_detection_noise(
    *,
    _base_fw,
    arr,
    bfac,
    cat_df,
    data0,
    data_dao,
    frame_name,
    fwhm_eff,
    h,
    hdr,
    wcs_obj,
    wpx,
):
    """Pixel noise of the frame for the DAO threshold (masked sky, aligned footprint)."""
    import numpy as np  # noqa: PLC0415

    from masterstar_gaia_accounting import (  # noqa: PLC0415
        estimate_star_masked_sky_sigma,
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
    # DAO-NOISE-2026-10: threshold basis = noise of the convolved (binned) detection
    # image, the same as the per-frame detection; the unbinned star-masked sigma
    # was on a different scale (effective ~6.8 sigma instead of the nominal N).
    std_dao = rms_conv
    if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
        std_dao = sky_sig
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
            LOGGER.debug("DAO std=0: %s n_unique=%d n_finite=%d min=%.1f max=%.1f",
                         _nm, n_unique, n_finite, mn, mx)
        except Exception:  # noqa: BLE001
            pass
        try:
            nonzero_mask = (arr != 0) & np.isfinite(arr)
            if int(np.count_nonzero(nonzero_mask)) > 100:
                std_dao = float(np.std(arr[nonzero_mask]))
        except Exception:  # noqa: BLE001
            pass
    return _gaia_chip_det, _gx_sm, _gy_sm, rms_conv, std_dao


def _dm_dao_pass1(
    *,
    _dao_false,
    _dao_flux_noise,
    _thr,
    _thr_sigma,
    arr,
    bfac,
    data_dao,
    frame_name,
    fwhm_eff,
    hdr,
    n_raw_dao,
    rms_conv,
    std_dao,
):
    """DAO pass 1 (DAOStarFinder at the threshold sigma) and its false-detection fraction."""
    from photutils.detection import DAOStarFinder  # noqa: PLC0415
    from masterstar_gaia_accounting import dedup_pass1_spatial  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    _dbg_log = logging.getLogger(__name__)
    if _dbg_log.isEnabledFor(logging.DEBUG):
        # G04-22 / G05-07: the full-frame mean/std are computed only when DEBUG is on.
        _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
        _dbg_log.debug(
            f"DAO input: {_nm} mean={float(np.nanmean(arr)):.1f} std={float(np.nanstd(arr)):.1f} "
            f"threshold={float(_thr):.1f} sky_sigma={float(std_dao):.2f} n_sigma={float(_thr_sigma):.2f} "
            f"rms_conv_diag={float(rms_conv):.2f} fwhm={float(fwhm_eff):.2f}"
        )
    finder = DAOStarFinder(
        fwhm=float(fwhm_eff),
        threshold=float(_thr),
        scale_threshold=False,
        n_brightest=None,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    tbl = finder(data_dao)
    n_raw_dao = int(len(tbl)) if tbl is not None else 0
    # DAO-FALSE-2026-10: false fraction of this detection run (negated-image test).
    from pipeline_catalog import DAO_FALSE_FRACTION_WARN, dao_false_detection_fraction  # noqa: PLC0415

    _dao_false = dao_false_detection_fraction(
        data_dao, fwhm_px=float(fwhm_eff), threshold_adu=float(_thr),
        star_xy=(np.c_[np.asarray(tbl["x_centroid"]), np.asarray(tbl["y_centroid"])] if n_raw_dao else []),
    )
    _ff = float(_dao_false["false_fraction"])
    # Sky noise of a DAO flux = sum of the detection image over the kernel footprint
    # (photutils DAOStarFinder: flux = sum(cutout x kernel.mask)).
    from photutils.detection.core import _StarFinderKernel  # noqa: PLC0415

    _kmask_n = int(_StarFinderKernel(fwhm=float(fwhm_eff), sigma_radius=1.5).mask.sum())
    _sig_det = sky_mad_sigma_adu(data_dao)[1]
    if math.isfinite(_sig_det) and _sig_det > 0 and _kmask_n > 0:
        _dao_flux_noise = float(_sig_det * math.sqrt(_kmask_n))
    log_event(
        f"DAO false detections (negated image): ~{_dao_false['n_false_estimate']} / {n_raw_dao} = "
        f"{100.0 * _ff:.1f} % at {float(_thr_sigma):.2f} sigma"
        + (f" - above {100.0 * DAO_FALSE_FRACTION_WARN:.0f} %: the frame has non-Gaussian noise "
           "(hot / cold pixels, gradients) or the threshold is too low" if _ff > DAO_FALSE_FRACTION_WARN else "")
    )
    if tbl is not None and len(tbl) > 0:
        tbl = _prefilter_dao_table_brightest(tbl, max(int(MASTERSTAR_MAX_DAO_ROWS) * 12, 36_000))
        from astropy.table import Table as _AstropyTable

        tbl = _AstropyTable(tbl, copy=True)
        tbl["vy_dao_pass"] = np.ones(len(tbl), dtype=np.int16)
    n_pass1_dao = int(len(tbl)) if tbl is not None else 0
    if tbl is not None and len(tbl) > 0:
        tbl = dedup_pass1_spatial(tbl, sep_px=MS_DAO_PASS1_DEDUP_PX)
    _sigma_p2_cfg = MS_DAO_PASS2_SIGMA
    _edge_p2 = MS_GAIA_CENSUS_EDGE_MARGIN_PX
    _dao_x_p1 = np.asarray([], dtype=np.float64)
    _dao_y_p1 = np.asarray([], dtype=np.float64)
    if tbl is not None and len(tbl) > 0:
        xb = np.asarray(tbl["x_centroid"], dtype=np.float64)
        yb = np.asarray(tbl["y_centroid"], dtype=np.float64)
        _dao_x_p1, _dao_y_p1 = _dao_xy_binned_to_full(xb, yb, int(bfac))
    # FRAME-DEPTH-2026-10: faint limit of this frame from its own pass-1 detections; it sets
    # the depth of pass-2, the census and the forced seeds (no static magnitude parameter).
    return _dao_false, _dao_flux_noise, _dao_x_p1, _dao_y_p1, _edge_p2, _sigma_p2_cfg, n_pass1_dao, n_raw_dao, tbl


def _dm_frame_depth(
    *,
    _base_fw,
    _cov_depth,
    _dao_x_p1,
    _dao_y_p1,
    _frame_depth,
    _gaia_chip_det,
    _gx_sm,
    _gy_sm,
    _match_depth_g,
    _thr,
    cat_df,
    h,
    tbl,
    wpx,
):
    """Frame depth (FRAME-DEPTH): completeness of pass-1 detections against the Gaia stars on the chip."""
    from frame_depth import measure_frame_depth  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    _g_cat = (
        pd.to_numeric(_gaia_chip_det.get("g_mag"), errors="coerce").to_numpy(dtype=np.float64)
        if _gaia_chip_det is not None and not _gaia_chip_det.empty
        else np.asarray([], dtype=np.float64)
    )
    _cat_depth = (
        float(_cov_depth)
        if _cov_depth is not None
        else (
            float(np.nanmax(pd.to_numeric(cat_df["mag"], errors="coerce")))
            if cat_df is not None and len(cat_df) and "mag" in cat_df.columns
            else float("nan")
        )
    )
    _frame_depth = measure_frame_depth(
        det_x=_dao_x_p1,
        det_y=_dao_y_p1,
        det_peak=(np.asarray(tbl["peak"], dtype=np.float64) if tbl is not None and len(tbl) else []),
        gaia_x=_gx_sm,
        gaia_y=_gy_sm,
        gaia_g=_g_cat,
        match_r_px=float(max(1.2, _base_fw)),
        threshold_adu=float(_thr),
        wpx=int(wpx),
        h=int(h),
        db_depth_g=_cat_depth if math.isfinite(_cat_depth) else None,
    )
    _depth_p2 = (
        float(_frame_depth["g_lim"])
        if _frame_depth.get("g_lim") is not None
        else (_cat_depth if math.isfinite(_cat_depth) else float("inf"))
    )
    _match_depth_g = float(_depth_p2)
    # The matching phase sees the catalogue to the matching depth only, so its tolerances,
    # pass-2 seeds and identities do not depend on how deep the DB is beyond the frame limit
    # (532: DB to 17.5 vs DB to 13 + top-up gave 267 vs 177 pass-2 recoveries before).
    _in_md = np.isfinite(_g_cat) & (_g_cat <= _match_depth_g + 1e-6)
    _gaia_chip_m = (
        _gaia_chip_det.loc[_in_md].reset_index(drop=True)
        if _gaia_chip_det is not None and len(_gaia_chip_det) == len(_in_md)
        else _gaia_chip_det
    )
    _gx_m = _gx_sm[_in_md] if len(_gx_sm) == len(_in_md) else _gx_sm
    _gy_m = _gy_sm[_in_md] if len(_gy_sm) == len(_in_md) else _gy_sm
    log_event(
        f"FRAME-DEPTH: G50={_frame_depth.get('g50')} sigma={_frame_depth.get('sigma_mag')} "
        f"({_frame_depth.get('method')}) -> frame limit G={_frame_depth.get('g_lim_frame')}, "
        f"catalogue depth G={_frame_depth.get('db_depth_g')}, matching depth G={_depth_p2:.2f}"
        + (" (limited by the catalogue)" if _frame_depth.get("limited_by_catalogue") else "")
    )
    return _depth_p2, _frame_depth, _gaia_chip_m, _gx_m, _gy_m, _match_depth_g


def _dm_pass2(
    *,
    _base_fw,
    _dao_x_p1,
    _dao_y_p1,
    _depth_p2,
    _derived_tol,
    _edge_p2,
    _gaia_chip_m,
    _gx_m,
    _gy_m,
    _shape_meta,
    _sigma_p2_cfg,
    _thr_sigma,
    bfac,
    data0,
    h,
    n_pass1_dao,
    tbl,
    wcs_obj,
    wpx,
):
    """Match tolerances from the pass-1 residuals and DAO pass 2 at the Gaia positions."""
    from masterstar_gaia_accounting import Pass2AcceptParams  # noqa: PLC0415
    from masterstar_gaia_accounting import dao_pass2_born_owned_rows  # noqa: PLC0415
    from masterstar_gaia_accounting import merge_dao_pass1_pass2_born_owned  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from dao_gaia_calibration import (  # noqa: PLC0415
        compute_pass1_astrometric_residuals_px,
        derive_tolerances_from_residuals,
        plate_scale_arcsec_per_px_from_wcs_nan,
    )

    # Residuals of the identities (detection <-> Gaia within the identity radius, catalogue to
    # the matching depth). A wider "coarse" radius let chance neighbours into the p95: 532 gave
    # p95 3.4 px with the DB to 17.5 and 6.3 px to the frame limit (match radius 5.5 / 10.5 px
    # for FWHM 1.5 px), and the pass-2 / census tolerances followed the DB depth.
    _r_ident_px = float(max(1.0, _base_fw))
    _res_dr = compute_pass1_astrometric_residuals_px(
        _dao_x_p1,
        _dao_y_p1,
        _gx_m,
        _gy_m,
        coarse_match_px=_r_ident_px,
    )
    _derived_tol = derive_tolerances_from_residuals(
        _res_dr,
        np.asarray([], dtype=np.float64),
        fwhm_px=float(max(1.2, _base_fw)),
        plate_scale_arcsec_per_px=plate_scale_arcsec_per_px_from_wcs_nan(wcs_obj),
        pass1_sigma=float(_thr_sigma),
        pass2_sigma=float(_sigma_p2_cfg),
        match_k=MS_DAO_MATCH_RADIUS_K,
        centroid_floor_px=MS_DAO_CENTROID_QA_FLOOR_PX,
        centroid_cap_px=MS_DAO_CENTROID_QA_CAP_PX,
    )
    # Pass-2 seeds: Gaia stars with no pass-1 detection closer than the pass-2 centre tolerance
    # plus the identity radius - nearer, pass-2 only finds that pass-1 detection again
    # (identity radius alone: 419 of 789 pass-2 rows on 532 were pass-1 detections).
    _match_r_px_p2 = float(_derived_tol.pass2_center_tol_px) + _r_ident_px
    from masterstar_gaia_accounting import derive_point_source_q_floor  # noqa: PLC0415

    _shape_meta = derive_point_source_q_floor(
        data0, _dao_x_p1, _dao_y_p1, _gx_m, _gy_m, match_r_px=_r_ident_px
    )
    log_event(
        f"POINT-SOURCE-SHAPE-01: q_floor={_shape_meta.get('q_floor')} "
        f"q_p50={_shape_meta.get('q_p50')} n_ref={_shape_meta.get('n_ref')}"
    )
    p2_params = Pass2AcceptParams(
        sigma=max(1.5, min(20.0, float(_sigma_p2_cfg))),
        center_tol_px=max(0.5, min(10.0, float(_derived_tol.pass2_center_tol_px))),
        fwhm_px=float(max(1.2, _base_fw)),
        shape_q_floor=_shape_meta.get("q_floor"),
    )
    _pass2_rows, _n_unmatched_gaia, _n_pass2_raw, _amb_p2 = dao_pass2_born_owned_rows(
        data0,
        tbl,
        gaia_chip=_gaia_chip_m,
        bfac=int(bfac),
        pass2_params=p2_params,
        target_depth_g=float(_depth_p2),
        edge_margin_px=float(_edge_p2),
        match_r_px=float(_match_r_px_p2),
        wpx=int(wpx),
        h=int(h),
    )
    tbl = merge_dao_pass1_pass2_born_owned(
        tbl, _pass2_rows, gaia_chip=_gaia_chip_m
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
    return _derived_tol, _shape_meta, tbl


def _dm_saturation_and_footprint(*, _fb_sat, center, equipment_saturate_adu, h, hdr, radius_deg, wpx):
    """Effective saturation limit and the footprint meta of the frame."""
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
    return foot_meta, sat_limit, sat_limit_src



def _n_gaia_variable(cat_df: pd.DataFrame) -> int:
    """Gaia DR3 stars flagged VARIABLE (phot_variable_flag, DB column var_flag) in the catalogue cone."""
    if cat_df is None or "var_flag" not in getattr(cat_df, "columns", []):
        return 0
    return int(cat_df["var_flag"].fillna("").astype(str).str.strip().str.upper().eq("VARIABLE").sum())

def _identity_candidate_pairs(
    sep_k: "np.ndarray", idx: "np.ndarray", oix_rows: "np.ndarray", nc: int, thr: float
) -> list[tuple[float, int, int]]:
    """Candidate (separation, detection, catalogue row) pairs within ``thr`` arcsec, sorted by
    separation; equal separations keep the row-major (detection, neighbour) order.

    G04-21: vectorised form of the former n x k Python loop; the stable argsort reproduces
    ``pairs.sort(key=sep)`` over the row-major insertion order exactly."""
    import numpy as np  # noqa: PLC0415

    sep = np.asarray(sep_k, dtype=np.float64)
    jj = np.asarray(idx, dtype=np.int64)
    n_oix = int(len(oix_rows))
    ok = (jj >= 0) & (jj < n_oix) & np.isfinite(sep) & (sep <= float(thr))
    ii, kk = np.nonzero(ok)  # row-major order
    cr = np.asarray(oix_rows, dtype=np.int64)[jj[ii, kk]]
    keep = (cr >= 0) & (cr < int(nc))
    ii, kk, cr = ii[keep], kk[keep], cr[keep]
    s = sep[ii, kk]
    order = np.argsort(s, kind="stable")
    return list(zip(s[order].tolist(), ii[order].tolist(), cr[order].tolist(), strict=True))


def _dm_assign_identities(
    *,
    _base_fw,
    _dao_pass,
    _identity_gate_acc,
    _match_depth_g,
    _match_sep_formula_inputs,
    _sat_csv,
    cat_df,
    dec_deg,
    det_str,
    exo_ann,
    exo_annotation_active,
    flux,
    hdr,
    idx_det,
    match_sep_used,
    n,
    n_matched,
    ra_deg,
    vsx_hit,
    vy_amb_owner,
    vy_seed_cid,
    wcs_obj,
    x,
    y,
):
    """Catalogue identities of the detections (1:1 greedy match, identity gate) -> the detection table."""
    import numpy as np  # noqa: PLC0415

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
                "gaia_dr3_variable_catalog": np.zeros(n, dtype=bool),
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
        var_c = np.empty(0, dtype=bool)
        # One identity path for every catalogue size (before: < 120 Gaia rows - sparse or narrow
        # fields - fell back to plain nearest neighbour, without 1:1 or the brightness rule).
        tree_pack = build_ucac_catalog_kdtree(cat_df)
        oix_rows: np.ndarray | None
        if tree_pack is not None:
            tr, oix_rows = tree_pack
        else:  # no finite catalogue position -> no identities
            tr = None
            oix_rows = None

        def _bind_gaia_catalog_columns() -> None:
            nonlocal nc, cid_c, cat_c, mag_c, bv_c, var_c
            nc = int(len(cat_df))
            # GAIA-VAR-ID-2026-10 (code review 2 G04-01): Gaia DR3 phot_variable_flag of the catalogue
            # row itself (local DB column var_flag), joined by identity - not by a separate position list
            var_c = (
                cat_df["var_flag"].fillna("").astype(str).str.strip().str.upper().eq("VARIABLE").to_numpy(dtype=bool)
                if "var_flag" in cat_df.columns
                else np.zeros(nc, dtype=bool)
            )
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

        _bind_gaia_catalog_columns()
        # G04-21: catalog_id -> catalogue row once per match (was rebuilt on every threshold call);
        # the last row of a duplicated id wins, as before
        _cid_to_cr = {str(cid_c[_k]).strip(): _k for _k in range(nc) if str(cid_c[_k]).strip()}

        def _assign_catalog_at_threshold(thr: float, g_zp: float = float("nan")) -> tuple[pd.DataFrame, int]:
            thr_f = float(thr)
            sepa_eff = np.full(n, np.nan, dtype=np.float64)
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
                pairs = _identity_candidate_pairs(sep_k, idx, oix_rows, nc, thr_f)
                used_det: set[int] = set()
                used_cat: set[int] = set()
                cat_row = np.full(n, -1, dtype=np.int64)
                sepa_out = np.full(n, np.nan, dtype=np.float64)
                # Born-owned pass2: pre-lock seed catalog_id (no greedy reassignment).
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
                # IDENTITY-2026-10: brightness-aware 1:1 assignment. Detections are taken from the
                # brightest down; each takes the BRIGHTEST unclaimed Gaia star within the identity
                # radius (it dominates the detected flux), ties by separation. Nearest-first let a
                # faint catalogue neighbour take the identity of the star DAO actually detected.
                _cands: dict[int, list[tuple[float, float, int]]] = {}
                _fl = np.asarray(flux, dtype=np.float64)
                for s_p, i_p, cr_p in pairs:
                    _mg = float(mag_c[cr_p]) if math.isfinite(float(mag_c[cr_p])) else 99.0
                    # a known variable (VSX at the detection or a Gaia DR3 VARIABLE catalogue star) is exempt
                    _known_var = bool(vsx_hit[i_p]) or bool(var_c[cr_p])
                    if _mg > _match_depth_g + 1e-6 and not _known_var:
                        # FRAME-DEPTH-2026-10: no identity with a star fainter than the frame can
                        # detect (532, DB to G 17.5: 335 detections took a G 16-17.5 star by chance
                        # at frame limit 15.57; the result depended on how deep the DB was).
                        continue
                    if (math.isfinite(float(g_zp)) and math.isfinite(_fl[i_p]) and _fl[i_p] > 0
                            and not _known_var):
                        # the catalogue star must be able to make the detected flux: not more than
                        # IDENTITY_MAX_FAINTER_MAG fainter than the flux says. Known variables (VSX /
                        # Gaia variable at this position) are exempt: a Mira near maximum can be far
                        # brighter than its Gaia mean G.
                        if _mg - (float(g_zp) - 2.5 * math.log10(float(_fl[i_p]))) > IDENTITY_MAX_FAINTER_MAG:
                            continue
                    _cands.setdefault(int(i_p), []).append((_mg, float(s_p), int(cr_p)))
                _order = sorted(_cands.keys(), key=lambda i_o: (-(_fl[i_o] if math.isfinite(_fl[i_o]) else -np.inf), i_o))
                for i_p in _order:
                    if i_p in used_det:
                        continue
                    for _mg, s_p, cr_p in sorted(_cands[i_p]):
                        if cr_p in used_cat:
                            continue
                        used_det.add(i_p)
                        used_cat.add(cr_p)
                        cat_row[i_p] = cr_p
                        sepa_out[i_p] = s_p
                        break
                matched_l = cat_row >= 0
                sepa_eff = sepa_out
            else:
                cat_row = np.full(n, -1, dtype=np.int64)
                matched_l = np.zeros(n, dtype=bool)
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
            mag_out[matched_l] = mag_c[safe_l[matched_l]]
            bv_out[matched_l] = bv_c[safe_l[matched_l]]
            cat_out = np.array([""] * n, dtype=object)
            cid_out = np.array([""] * n, dtype=object)
            cat_out[matched_l] = cat_c[safe_l[matched_l]]
            cid_out[matched_l] = cid_c[safe_l[matched_l]]
            gvar_out = np.zeros(n, dtype=bool)
            gvar_out[matched_l] = var_c[safe_l[matched_l]]
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
                    "gaia_dr3_variable_catalog": gvar_out,
                    **(exo_ann if exo_annotation_active else {}),
                    **_sat_csv,
                }
            )
            return df_l, n_ma

        def _apply_post_match_identity_gate() -> None:
            nonlocal df_out, n_matched, _identity_gate_acc
            from wcs_invertibility import (
                accumulate_identity_gate,
                apply_post_match_identity_gate_df,
                gaia_radec_map_from_table,
            )

            # x/y are full-resolution pixels -> the full-resolution FWHM (DAO may run 2x binned)
            _fwhm_gate = float(max(1.2, _base_fw))
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
        def _run_full_match_pass() -> None:
            nonlocal df_out, n_matched, match_sep_used, _match_sep_formula_inputs
            from dao_gaia_calibration import (
                catalog_identity_radius_arcsec,
                plate_scale_arcsec_per_px_from_wcs_nan,
                solve_rms_px_from_fits_header,
            )

            _ps_match = float(plate_scale_arcsec_per_px_from_wcs_nan(wcs_obj))
            match_sep_used, _d1_inputs = catalog_identity_radius_arcsec(
                fwhm_dao_px=float(max(1.2, _base_fw)),  # full-resolution px (x/y are full-res)
                plate_scale_arcsec_per_px=_ps_match,
                solve_rms_px=solve_rms_px_from_fits_header(hdr),
            )
            if not math.isfinite(float(match_sep_used)):
                raise RuntimeError("Catalog match: plate scale from WCS unavailable - identity radius undefined")
            df_out, n_matched = _assign_catalog_at_threshold(match_sep_used)
            # Brightness consistency: the flux -> G zero point of this frame (robust median over the
            # identities) rejects a catalogue star that is far too faint to make the detected flux
            # (532: 328 detections took a G 16-17.5 neighbour, flux 2 mag brighter; spurious or
            # secondary detections around bright stars). Second pass with that constraint.
            _g_zp = _flux_g_zero_point(
                df_out,
                flux,
                use=(np.asarray(_dao_pass) == 1) & ~np.asarray(vsx_hit, dtype=bool)
                & ~df_out["gaia_dr3_variable_catalog"].to_numpy(dtype=bool),
            )
            if math.isfinite(_g_zp):
                df_out, n_matched = _assign_catalog_at_threshold(match_sep_used, g_zp=_g_zp)
            _d1_inputs["flux_g_zero_point"] = float(_g_zp) if math.isfinite(_g_zp) else None
            log_event(
                f"Catalog match (IDENTITY-2026-10): radius max(1 px, FWHM_dao={float(_d1_inputs['fwhm_dao_px']):.3f} px) "
                f"= {float(_d1_inputs['radius_px']):.2f} px = {float(match_sep_used):.2f} arcsec "
                f"(scale {float(_ps_match):.4f} arcsec/px); brightest Gaia star per detection, at most "
                f"{IDENTITY_MAX_FAINTER_MAG:.1f} mag fainter than its flux (ZP {float(_g_zp):.3f}): "
                f"{int(n_matched)} of {int(n)} detections identified"
            )
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
                    "consider lowering detection_threshold_sigma in config.json",
                    100.0 * _dao_match_rate,
                )
    n_detected_dao = int(n)

    n_sat = int(df_out["likely_saturated"].sum()) if len(df_out) and "likely_saturated" in df_out.columns else 0
    return _identity_gate_acc, _match_sep_formula_inputs, df_out, match_sep_used, n_detected_dao, n_sat


def _dm_meta_and_snr(
    *,
    _base_fw,
    _bg_sigma_adu,
    _dao_false,
    _dao_flux_noise,
    _dao_n_equiv_used,
    _derived_tol,
    _empirical_clip_adu,
    _frame_depth,
    _frame_max_adu,
    _identity_gate_acc,
    _match_sep_formula_inputs,
    _shape_meta,
    _sky_med_gate,
    _thr_sigma,
    arr,
    bfac,
    cat_df,
    cfg,
    df_out,
    exo_annotation_active,
    exo_df,
    field_catalog_export_path,
    foot_meta,
    hdr,
    match_sep_used,
    n_detected_dao,
    n_raw_dao,
    n_sat,
    n_sat_pk,
    n_sat_pl,
    n_spatial,
    photometric_gain,
    sat_limit,
    sat_limit_src,
    vsx_df,
    wcs_obj,
):
    """Detection meta (counts, tolerances, depth, identity gate) and the S/N columns of the table."""
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
        "sky_median_adu": float(_sky_med_gate),
        "bg_sigma_adu": float(_bg_sigma_adu),
        "bg_sigma_estimator": "1.4826 x MAD of all pixels (plain_stats.sky_mad_sigma_adu)",
        "frame_max_adu": float(_frame_max_adu),
        "empirical_clip_adu": _empirical_clip_adu,
        "n_detected_dao_raw": int(n_raw_dao),
        "n_dao_after_spatial_cap": int(n_spatial),
        "n_detected_dao": n_detected_dao,
        "n_detected": int(len(df_out)),
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
        "n_gaia_variable_in_field": _n_gaia_variable(cat_df),
        **foot_meta,
        "field_catalog_cone_csv": str(Path(field_catalog_export_path)) if field_catalog_export_path else None,
        "dao_threshold_sigma": float(_thr_sigma),
        "dao_detection_n_equiv": (
            float(_dao_n_equiv_used) if _dao_n_equiv_used is not None and math.isfinite(_dao_n_equiv_used) else None
        ),
        # full-resolution FWHM (x/y, apertures, gates); DAO may detect on a binned copy
        "dao_fwhm_px": float(max(1.2, _base_fw)),
        "dao_detect_binning": int(bfac),
        "dao_false_detection": dict(_dao_false),
        "dao_flux_noise_adu": _dao_flux_noise,
        "frame_depth": dict(_frame_depth),
        "match_sep_arcsec_requested": float(CATALOG_MATCH_SEP_ARCSEC),
        "match_sep_arcsec_effective": float(match_sep_used),
        "match_sep_formula_inputs": dict(_match_sep_formula_inputs),
        "catalog_match_fraction_target": 0.95,
        "catalog_match_fraction_met": (
            bool((float(n_matched_final) / float(max(1, len(df_out)))) >= 0.95) if len(df_out) else True
        ),
        "dao_gaia_derived_tol": (
            {
                **_derived_tol.to_dict(),
                "point_source_q_floor": (_shape_meta or {}).get("q_floor"),
            }
            if _derived_tol is not None
            else None
        ),
        "point_source_shape": _shape_meta,
        "identity_gate": {
            **dict(_identity_gate_acc),
            "fwhm_px": float(max(1.2, _base_fw)),
            "plate_scale_arcsec_per_px": (
                float(_ps_idg) if _ps_idg is not None and math.isfinite(float(_ps_idg)) else None
            ),
            "fail_threshold_px": float(3.0 * float(max(1.2, _base_fw))),
            "fail_threshold_arcsec": (
                float(3.0 * float(max(1.2, _base_fw)) * float(_ps_idg))
                if _ps_idg is not None and math.isfinite(float(_ps_idg))
                else None
            ),
        },
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
        # The footprint reconciliation (completeness decomposition) runs once, on the final
        # MASTERSTAR catalogue in masterstar_build (FRAME-DEPTH-2026-10: one depth, one report).
    else:
        LOGGER.debug("[DAO] catalog_rows not available - Gaia->DAO skip")
    df_out = _proc_rename_det_names_to_catalog_id(df_out)
    # GAIN-ONE-2026-10: the run's gain (rig_photometry.photometric_gain_read_noise), not the raw
    # config default 1.0; config aperture / annulus without divergent fallbacks
    _gain_ms = float(photometric_gain) if photometric_gain is not None else float("nan")
    if not (math.isfinite(_gain_ms) and _gain_ms > 0):
        from rig_photometry import photometric_gain_read_noise  # noqa: PLC0415

        _gain_ms = float(photometric_gain_read_noise(hdr, cfg, None).gain)
    _ap_fac = float(cfg.aperture_fwhm_factor)
    _ann_in = float(cfg.annulus_inner_fwhm)
    _ann_out = float(cfg.annulus_outer_fwhm)
    df_out = stamp_masterstar_snr_columns(
        df_out,
        image=arr,
        fwhm_dao_px=float(max(1.2, _base_fw)),  # aperture on the full-resolution image
        bg_sigma_adu=float(_bg_sigma_adu),
        gain=_gain_ms,
        aperture_fwhm_factor=_ap_fac,
        annulus_inner_fwhm=_ann_in,
        annulus_outer_fwhm=_ann_out,
    )
    return df_out, meta


def detect_stars_and_match_catalog(
    data: "np.ndarray",
    hdr: fits.Header,
    *,
    vsx_df: pd.DataFrame | None = None,
    exo_df: pd.DataFrame | None = None,
    vsx_match_max_sep_arcsec: float = 5.0,
    gaia_db_path: Path | None = None,
    field_catalog_export_path: Path | None = None,
    dao_fwhm_px: float | None = None,
    fallback_saturate_adu: float | None = None,
    equipment_saturate_adu: float | None = None,
    frame_name: str = "",
    dao_fwhm_bypass_header: bool = False,
    measured_saturate_adu: float | None = None,
    photometric_gain: float | None = None,
    cfg: AppConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect stars (DAOStarFinder), sky coordinates from WCS, match to the **local Gaia** DB.

    ``cfg`` is the run configuration (CONFIG-ONE-2026-10): DAO threshold, aperture / annulus factors and
    the catalogue paths come from it, never from a fresh read of config.json.

    Used for MASTERSTAR (WCS verified by astrosolve). The local Gaia catalogue of the solved
    footprint is read from ``gaia_db_path`` (argument; config as fallback), every star the DB holds
    there; its completeness depth comes from the DB coverage record (gaia_coverage).

    **Known variables:** handled via Gaia flags and/or optional VSX checks (no Gaia TAP here).

    **Depth (FRAME-DEPTH-2026-10):** no magnitude parameter; the faint limit of the frame is measured
    after pass-1 (``frame_depth``) and capped by the catalogue depth of the field.

    Identity (IDENTITY-2026-10): radius max(1 px, FWHM); each detection (brightest first) takes
    the brightest unclaimed Gaia star in the radius that is at most 2 mag fainter than its flux
    implies (known variables exempt); then the post-match identity gate. Tolerances
    (pass-2 seeds, census lock) derive from the identity residuals; ``CATALOG_MATCH_SEP_ARCSEC`` is only
    recorded in the meta (requested value).

    ``MASTERSTAR_MAX_DAO_ROWS`` caps the DAO detections kept. Rows are chosen with **spatial
    stratification** (brightest per coarse grid cell, then global flux top-up) so vignetting does not
    mimic a ``catalog disc`` the way a plain brightest-N sort does.

    If ``field_catalog_export_path`` is set, the **full** cone table (``cat_df``) is written there for
    QA overlays - many more rows than DAO detections in ``masterstars.csv``.

    DAOStarFinder threshold = ``cfg.detection_threshold_sigma`` x std(background).

    False detections: measured on the negated image (``det_meta["dao_false_detection"]``);
    ``det_meta["dao_flux_noise_adu"]`` is the sky noise of a DAO ``flux`` (sum over the kernel
    footprint of the detection image): sigma_pixel(detection image) x sqrt(footprint pixels).

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
    center, radius_deg = _effective_field_catalog_cone_radius_deg(wcs_obj, h, wpx)
    # The caller's gaia_db_path wins; the run's configured DB is the fallback.
    _gaia_db_path: Path | None = Path(gaia_db_path) if gaia_db_path is not None else None
    if _gaia_db_path is None:
        gp = (cfg.gaia_db_path or "").strip()
        _gaia_db_path = Path(gp) if gp else None
    _fb_sat = fallback_saturate_adu
    cat_df = _query_gaia_local(center=center, radius_deg=radius_deg, gaia_db_path=_gaia_db_path)
    # Catalogue depth of this field: the coverage record of the main DB (gaia_coverage).
    _cov_depth: float | None = None
    if _gaia_db_path is not None and Path(_gaia_db_path).is_file():
        from gaia_coverage import covered_depth_g, field_box  # noqa: PLC0415

        _cov_depth = covered_depth_g(_gaia_db_path, field_box(center, radius_deg))
    if field_catalog_export_path is not None and cat_df is not None and len(cat_df) > 0:
        _fcp = Path(field_catalog_export_path)
        _fcp.parent.mkdir(parents=True, exist_ok=True)
        _vyvar_df_to_csv(cat_df, _fcp)
        log_event(
            f"Field catalog: {int(wpx)}x{int(h)} px, {len(cat_df)} Gaia stars in cone r~{float(radius_deg):.2f} deg "
            f"-> field_catalog_cone.csv (no cropping)."
        )
        try:
            _write_field_catalog_cone_meta(
                _fcp,
                center=center,
                radius_deg=float(radius_deg),
                naxis1=int(wpx),
                naxis2=int(h),
            )
        except Exception:  # noqa: BLE001
            pass
    # ``vsx_df`` empty DataFrame from prefetch = 'already tried'; fill in only if the caller passed no table (``None``).
    if vsx_df is None:
        _vx: Path | None = None
        try:
            _vxs = str(cfg.objects_db_path or "").strip()
            if _vxs:
                _vx = Path(_vxs).expanduser().resolve()
        except Exception:  # noqa: BLE001
            _vx = None
        vsx_df = _query_vsx_local(center=center, radius_deg=radius_deg, vsx_db_path=_vx)
    exo_annotation_active = False
    exo_max = EXOPLANET_MATCH_MAX_SEP_ARCSEC
    _exo_path: Path | None = None
    try:
        _exs = str(cfg.objects_db_path or "").strip()
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

    mean, med, std = plain_mean_med_std(arr)
    data0 = np.nan_to_num((arr - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    bfac = 1
    n_raw_dao = 0
    _dao_false: dict[str, Any] = {}
    _dao_flux_noise: float | None = None
    cfg = cfg
    _fb_c = float(dao_fwhm_px) if dao_fwhm_px is not None else DAO_FWHM_FALLBACK_PX
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
    _shape_meta: dict[str, Any] | None = None
    _frame_depth: dict[str, Any] = {}
    # Matching depth of this frame (FRAME-DEPTH-2026-10); identities only to it (set after pass-1).
    _match_depth_g = float("inf")

    dao_scale = _dao_auto_binning_factor(*data0.shape)
    data_dao, bfac = _bin2d_mean(data0, dao_scale)
    fwhm_eff = max(1.2, _base_fw / float(bfac))
    _gaia_chip_det, _gx_sm, _gy_sm, rms_conv, std_dao = _dm_detection_noise(
        _base_fw=_base_fw,
        arr=arr,
        bfac=bfac,
        cat_df=cat_df,
        data0=data0,
        data_dao=data_dao,
        frame_name=frame_name,
        fwhm_eff=fwhm_eff,
        h=h,
        hdr=hdr,
        wcs_obj=wcs_obj,
        wpx=wpx,
    )
    if std_dao is None or (not np.isfinite(float(std_dao))) or float(std_dao) <= 0:
        _nm = str(frame_name or hdr.get("FILENAME") or "").strip() or "frame"
        LOGGER.warning("%s: DAO std=0 even after fallback, frame skipped", _nm)
        return pd.DataFrame(), {
            "n_detected": 0,
            "n_detected_dao": 0,
            "n_matched": 0,
            "catalog_rows": int(len(cat_df)) if cat_df is not None else 0,
            "catalog_match_mode": "full_cone",
            "reason": "std_dao_zero",
        }
    _thr_sigma = float(cfg.detection_threshold_sigma)
    _thr = max(_thr_sigma * float(std_dao), 1e-6)
    _dao_n_equiv_used = float(_thr_sigma)
    # Adaptive threshold monitoring: match-rate check runs after first catalog match pass (below).
    _dao_false, _dao_flux_noise, _dao_x_p1, _dao_y_p1, _edge_p2, _sigma_p2_cfg, n_pass1_dao, n_raw_dao, tbl = _dm_dao_pass1(
        _dao_false=_dao_false,
        _dao_flux_noise=_dao_flux_noise,
        _thr=_thr,
        _thr_sigma=_thr_sigma,
        arr=arr,
        bfac=bfac,
        data_dao=data_dao,
        frame_name=frame_name,
        fwhm_eff=fwhm_eff,
        hdr=hdr,
        n_raw_dao=n_raw_dao,
        rms_conv=rms_conv,
        std_dao=std_dao,
    )

    _depth_p2, _frame_depth, _gaia_chip_m, _gx_m, _gy_m, _match_depth_g = _dm_frame_depth(
        _base_fw=_base_fw,
        _cov_depth=_cov_depth,
        _dao_x_p1=_dao_x_p1,
        _dao_y_p1=_dao_y_p1,
        _frame_depth=_frame_depth,
        _gaia_chip_det=_gaia_chip_det,
        _gx_sm=_gx_sm,
        _gy_sm=_gy_sm,
        _match_depth_g=_match_depth_g,
        _thr=_thr,
        cat_df=cat_df,
        h=h,
        tbl=tbl,
        wpx=wpx,
    )
    _derived_tol, _shape_meta, tbl = _dm_pass2(
        _base_fw=_base_fw,
        _dao_x_p1=_dao_x_p1,
        _dao_y_p1=_dao_y_p1,
        _depth_p2=_depth_p2,
        _derived_tol=_derived_tol,
        _edge_p2=_edge_p2,
        _gaia_chip_m=_gaia_chip_m,
        _gx_m=_gx_m,
        _gy_m=_gy_m,
        _shape_meta=_shape_meta,
        _sigma_p2_cfg=_sigma_p2_cfg,
        _thr_sigma=_thr_sigma,
        bfac=bfac,
        data0=data0,
        h=h,
        n_pass1_dao=n_pass1_dao,
        tbl=tbl,
        wcs_obj=wcs_obj,
        wpx=wpx,
    )
    foot_meta, sat_limit, sat_limit_src = _dm_saturation_and_footprint(
        _fb_sat=_fb_sat,
        center=center,
        equipment_saturate_adu=equipment_saturate_adu,
        h=h,
        hdr=hdr,
        radius_deg=radius_deg,
        wpx=wpx,
    )
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame(), {
            "n_detected": 0,
            "n_detected_dao": 0,
            "n_matched": 0,
            "catalog_rows": int(len(cat_df)),
            "catalog_match_mode": "full_cone",
            "n_likely_saturated": 0,
            "n_saturated_from_peak": 0,
            "n_saturated_plateau": 0,
            "n_vsx_in_field": int(len(vsx_df)) if vsx_df is not None else 0,
            "n_exo_hosts_in_field": (
                int(len(exo_df)) if exo_annotation_active and exo_df is not None else 0
            ),
            "n_gaia_variable_in_field": _n_gaia_variable(cat_df),
            "field_catalog_cone_csv": str(Path(field_catalog_export_path)) if field_catalog_export_path else None,
            "dao_threshold_sigma": float(_thr_sigma),
            "dao_fwhm_px": float(max(1.2, _base_fw)),
            "dao_detect_binning": int(bfac),
            "frame_depth": dict(_frame_depth),
            "match_sep_arcsec_requested": float(CATALOG_MATCH_SEP_ARCSEC),
            "match_sep_arcsec_effective": float(CATALOG_MATCH_SEP_ARCSEC),
            "saturate_limit_adu": float(sat_limit) if sat_limit is not None else None,
            "saturate_limit_source": sat_limit_src,
            **foot_meta,
        }

    _d_h2, _d_w2 = int(data_dao.shape[0]), int(data_dao.shape[1])
    # Stratify brightest DAO sources on a coarse grid over the **full** chip (not radial distance from center).
    _keep2 = _dao_spatial_flux_cap_row_indices(
        tbl, max_n=int(MASTERSTAR_MAX_DAO_ROWS), width_px=float(_d_w2), height_px=float(_d_h2)
    )
    tbl = tbl[_keep2]
    tbl.sort("flux")
    tbl = tbl[::-1]
    n_spatial = int(len(tbl))
    log_event(
        f"DAO on frame: raw={n_raw_dao} (after brightest-prefilter max {max(int(MASTERSTAR_MAX_DAO_ROWS) * 12, 36_000):d}) -> "
        f"after spatial cap max_n={int(MASTERSTAR_MAX_DAO_ROWS)}: {n_spatial} points (DAO binning={bfac}x)."
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
    if vsx_df is not None and not vsx_df.empty:
        vxc = SkyCoord(
            ra=np.asarray(vsx_df["ra_deg"], dtype=float) * u.deg,
            dec=np.asarray(vsx_df["dec_deg"], dtype=float) * u.deg,
        )
        _, sepvx, _ = det_coords.match_to_catalog_sky(vxc)
        vsx_hit = np.asarray(sepvx.arcsec <= vsx_max, dtype=bool)
    else:
        vsx_hit = np.zeros(n, dtype=bool)
    if exo_annotation_active:
        exo_ann, _exo_warns = _exo_host_annotation_arrays(
            det_coords,
            exo_df if exo_df is not None else pd.DataFrame(),
            exo_max,
            frame_name=frame_name,
        )

    sat_frac = float(LINEARITY_FRAC)

    pmax_arr = _box_peaks_at_centroids(arr, x, y)
    _frame_max_adu = float(np.nanmax(arr))
    _empirical_clip_adu = _detect_empirical_clip_level_adu(arr)
    _peak_sat_lim = _resolve_peak_saturation_limit_adu(
        camera_sat_limit_adu=sat_limit,
        saturate_fraction=sat_frac,
        sky_median_adu=float(med),
        frame_max_adu=_frame_max_adu,
        empirical_clip_adu=_empirical_clip_adu,
        measured_sat_adu=measured_saturate_adu,
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
    # Sky level and pixel noise of the frame (zones, DAO-only classification). The former
    # prematch peak gate (masterstar_prematch_peak_sigma_floor) is removed (DAO-FALSE-2026-10):
    # at the DAO threshold it dropped 1 of 1957 detections (530) - the detection is the gate.
    _sky_med_gate, _sky_sig_gate = sky_mad_sigma_adu(arr)
    if not (math.isfinite(_sky_sig_gate) and float(_sky_sig_gate) > 0):
        _sky_sig_gate = float(std) if np.isfinite(std) else 1.0
    if not math.isfinite(_sky_med_gate):
        _sky_med_gate = float(med)
    _bg_sigma_adu = max(float(_sky_sig_gate), 1.0)
    if "vy_dao_pass" in tbl.colnames:
        _dao_pass = np.asarray(tbl["vy_dao_pass"], dtype=np.int16)
        if int(_dao_pass.size) != int(n):
            # Spatial / other pre-filters may have shortened peak arrays; fall back to pass-1.
            _dao_pass = np.ones(n, dtype=np.int16)
    else:
        _dao_pass = np.ones(n, dtype=np.int16)
    on_chip = detection_on_chip(x, y, wpx=int(wpx), h=int(h))
    n_on = int(np.count_nonzero(on_chip))
    if 0 < n_on < n:
        x = x[on_chip]
        y = y[on_chip]
        flux = flux[on_chip]
        peak_dao = peak_dao[on_chip]
        ra_deg = ra_deg[on_chip]
        dec_deg = dec_deg[on_chip]
        det_coords = det_coords[on_chip]
        vsx_hit = vsx_hit[on_chip]
        if exo_annotation_active:
            exo_ann = _slice_exo_annotation(exo_ann, on_chip)
        pmax_arr = pmax_arr[on_chip]
        _sat_block = {k: np.asarray(v)[on_chip] for k, v in _sat_block.items()}
        _sat_csv, n_sat_pk, n_sat_pl = _proc_sat_block_for_csv(_sat_block)
        vy_seed_cid = vy_seed_cid[on_chip]
        vy_amb_owner = vy_amb_owner[on_chip]
        _dao_pass = _dao_pass[on_chip]
        n = n_on
        log_event(f"DAO detections on the chip: {n}/{n_spatial} (sky_sig~{_bg_sigma_adu:.1f} ADU).")

    idx_det = np.arange(1, n + 1, dtype=np.int32)
    det_str = np.array([f"DET_{i:04d}" for i in idx_det], dtype=object)
    n_matched = 0
    match_sep_used = float(CATALOG_MATCH_SEP_ARCSEC)  # replaced by the identity radius once matching runs
    _match_sep_formula_inputs: dict[str, Any] = {}
    from wcs_invertibility import empty_identity_gate_acc

    _identity_gate_acc = empty_identity_gate_acc()
    _identity_gate_acc, _match_sep_formula_inputs, df_out, match_sep_used, n_detected_dao, n_sat = _dm_assign_identities(
        _base_fw=_base_fw,
        _dao_pass=_dao_pass,
        _identity_gate_acc=_identity_gate_acc,
        _match_depth_g=_match_depth_g,
        _match_sep_formula_inputs=_match_sep_formula_inputs,
        _sat_csv=_sat_csv,
        cat_df=cat_df,
        dec_deg=dec_deg,
        det_str=det_str,
        exo_ann=exo_ann,
        exo_annotation_active=exo_annotation_active,
        flux=flux,
        hdr=hdr,
        idx_det=idx_det,
        match_sep_used=match_sep_used,
        n=n,
        n_matched=n_matched,
        ra_deg=ra_deg,
        vsx_hit=vsx_hit,
        vy_amb_owner=vy_amb_owner,
        vy_seed_cid=vy_seed_cid,
        wcs_obj=wcs_obj,
        x=x,
        y=y,
    )
    df_out, meta = _dm_meta_and_snr(
        _base_fw=_base_fw,
        _bg_sigma_adu=_bg_sigma_adu,
        _dao_false=_dao_false,
        _dao_flux_noise=_dao_flux_noise,
        _dao_n_equiv_used=_dao_n_equiv_used,
        _derived_tol=_derived_tol,
        _empirical_clip_adu=_empirical_clip_adu,
        _frame_depth=_frame_depth,
        _frame_max_adu=_frame_max_adu,
        _identity_gate_acc=_identity_gate_acc,
        _match_sep_formula_inputs=_match_sep_formula_inputs,
        _shape_meta=_shape_meta,
        _sky_med_gate=_sky_med_gate,
        _thr_sigma=_thr_sigma,
        arr=arr,
        bfac=bfac,
        cat_df=cat_df,
        cfg=cfg,
        df_out=df_out,
        exo_annotation_active=exo_annotation_active,
        exo_df=exo_df,
        field_catalog_export_path=field_catalog_export_path,
        foot_meta=foot_meta,
        hdr=hdr,
        match_sep_used=match_sep_used,
        n_detected_dao=n_detected_dao,
        n_raw_dao=n_raw_dao,
        n_sat=n_sat,
        n_sat_pk=n_sat_pk,
        n_sat_pl=n_sat_pl,
        n_spatial=n_spatial,
        photometric_gain=photometric_gain,
        sat_limit=sat_limit,
        sat_limit_src=sat_limit_src,
        vsx_df=vsx_df,
        wcs_obj=wcs_obj,
    )
    return df_out, meta
