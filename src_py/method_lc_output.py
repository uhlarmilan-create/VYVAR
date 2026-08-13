"""Write alternate-method lightcurve CSVs (PSF / adaptive) for method-keyed reports."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from report_methods import lc_csv_path

LOGGER = logging.getLogger(__name__)


@dataclass
class MethodLcWriteContext:
    method: str
    target_cid: str
    comp_ids: list[str]
    all_frames: pd.DataFrame
    lc_dir: Path
    cfg: Any
    stability_sigma: float
    outlier_sigma: float
    comp_catalog_mag: dict[str, float]
    comp_rms_map: dict[str, float]
    comp_tier_map: dict[str, int]
    tier_weights: dict[int, float]
    target_row: Any
    state: Any
    apertures_px: dict[str, float]
    ac_result: dict[str, Any] | None
    comp_bp_rp: dict[str, float]
    target_bp_rp: float
    bjd: np.ndarray
    hjd: np.ndarray
    jd: np.ndarray
    airmass_arr: np.ndarray
    flip_arr: np.ndarray
    err: np.ndarray
    ap_arr: np.ndarray
    src_files: list[str]
    sat_flags: np.ndarray
    target_frames: pd.DataFrame
    lunar_phase_pct: float
    lunar_separation_deg: float
    lunar_risk: str
    time_base: str = "BJD_TDB"


def _build_flux(
    ctx: MethodLcWriteContext,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from photometry_core import (  # noqa: PLC0415
        _get_lc,
        _get_lc_adaptive_per_star,
        _get_lc_psf_strict,
    )

    m = str(ctx.method).strip().lower()
    if m == "psf":
        tlc = _get_lc_psf_strict(ctx.target_cid, ctx.all_frames)
        clc = {cid: _get_lc_psf_strict(cid, ctx.all_frames) for cid in ctx.comp_ids}
    elif m == "adaptive":
        tlc = _get_lc_adaptive_per_star(ctx.target_cid, ctx.all_frames)
        clc = {cid: _get_lc_adaptive_per_star(cid, ctx.all_frames) for cid in ctx.comp_ids}
    else:
        tlc = _get_lc(ctx.target_cid, ctx.all_frames)
        clc = {cid: _get_lc(cid, ctx.all_frames) for cid in ctx.comp_ids}
    return tlc, clc


def save_method_variant_lightcurve(ctx: MethodLcWriteContext) -> Path | None:
    """Run ensemble pipeline for one flux method and write suffixed LC CSV."""
    from photometry_core import (  # noqa: PLC0415
        _get_comp_bjd_series,
        apply_color_term,
        apply_reporting_postprocess,
        check_comparison_stability,
        democratic_detrend_lc,
        ensemble_normalize,
        fit_color_term_c1,
        pytics_iterative_weights,
        save_lightcurve_csv,
        savgol_detrend_lc,
        should_apply_color_term,
        temporal_bin_comp_lc,
    )
    from dilution import apply_target_dilution_to_mag_calib, compute_dilution_factor  # noqa: PLC0415

    method = str(ctx.method).strip().lower()
    if method not in ("psf", "adaptive"):
        return None

    target_lc, comp_lc = _build_flux(ctx)
    _cfg = ctx.cfg

    comp_lc = temporal_bin_comp_lc(
        comp_lc=comp_lc,
        comp_quality={},
        all_frames=ctx.all_frames,
        window=int(_cfg.temporal_bin_window),
        enabled=bool(_cfg.temporal_binning_enabled),
    )
    comp_bjd = {cid: _get_comp_bjd_series(cid, ctx.all_frames) for cid in ctx.comp_ids}
    comp_quality = check_comparison_stability(
        comp_lc,
        comp_rms_map=ctx.comp_rms_map,
        comp_bjd=comp_bjd,
        n_comp_min=3,
        outlier_sigma=ctx.stability_sigma,
        max_comp_p2p=float(_cfg.phase01_comparison_max_comp_rms),
        max_comp_slope_mmag_hr=float(_cfg.comp_max_slope_mmag_hr),
        comp_slope_significance_k=float(getattr(_cfg, "comp_slope_significance_k", 3.0)),
        common_mode_detrend=True,
    )
    comp_rms_map = pytics_iterative_weights(
        comp_lc=comp_lc,
        comp_quality=comp_quality,
        comp_rms_map=dict(ctx.comp_rms_map),
        n_iter=int(_cfg.pytics_n_iter),
        enabled=bool(_cfg.pytics_enabled),
    )
    mag_calib, delta_mag, _ = ensemble_normalize(
        target_lc,
        comp_lc,
        ctx.comp_catalog_mag,
        comp_quality,
        comp_rms_map=comp_rms_map,
        comp_tier_map=ctx.comp_tier_map,
        tier_weights=ctx.tier_weights,
        n_comp_min=3,
        n_comp_max=int(_cfg.phase01_comparison_n_comp_max),
    )

    _dilution_result: dict[str, Any] = {
        "dilution_factor": 1.0,
        "dilution_delta_mag": 0.0,
        "n_neighbors": 0,
    }
    if bool(_cfg.gs11_dilution_enabled) and ctx.state.gaia_db_path:
        from dilution import _normalize_exclude_source_id  # noqa: PLC0415

        try:
            _target_ra = float(
                pd.to_numeric(ctx.target_row.get("ra_deg", ctx.target_row.get("ra", float("nan"))), errors="coerce")
            )
            _target_dec = float(
                pd.to_numeric(ctx.target_row.get("dec_deg", ctx.target_row.get("dec", float("nan"))), errors="coerce")
            )
        except (TypeError, ValueError):
            _target_ra = _target_dec = float("nan")
        _target_g_mag = float("nan")
        for _gk in ("mag", "phot_g_mean_mag", "catalog_mag"):
            try:
                _gv = float(pd.to_numeric(ctx.target_row.get(_gk, float("nan")), errors="coerce"))
            except (TypeError, ValueError):
                _gv = float("nan")
            if math.isfinite(_gv):
                _target_g_mag = _gv
                break
        _ap_cfg = float(_cfg.gs11_dilution_aperture_arcsec)
        if math.isfinite(_ap_cfg) and _ap_cfg > 0:
            _ap_arcsec = _ap_cfg
        else:
            _ap_px = float(ctx.apertures_px.get(ctx.target_cid, 3.0))
            _ap_arcsec = float(_ap_px) * float(ctx.state.plate_scale_arcsec)
        _cid_int = None
        try:
            _cid_int = _normalize_exclude_source_id(ctx.target_cid)
        except Exception:  # noqa: BLE001
            _cid_int = None
        _dilution_result = compute_dilution_factor(
            _target_ra,
            _target_dec,
            _target_g_mag,
            _ap_arcsec,
            str(ctx.state.gaia_db_path),
            catalog_id=_cid_int,
            mag_limit_delta=float(_cfg.gs11_dilution_mag_limit_delta),
        )
        mag_calib, _dilution_result = apply_target_dilution_to_mag_calib(
            mag_calib,
            _dilution_result,
            _cfg,
            target_cid=str(ctx.target_cid),
        )

    ac_ok = bool(ctx.ac_result.get("ok", False)) if isinstance(ctx.ac_result, dict) else False
    delta_m_corr = ctx.ac_result.get("delta_m_corr", None) if isinstance(ctx.ac_result, dict) else None
    if ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
        mag_calib_ac = mag_calib + float(delta_m_corr)
    else:
        mag_calib_ac = np.full_like(mag_calib, float("nan"))

    target_bp_rp = float(ctx.target_bp_rp)
    c1 = 0.0
    ct_corr = 0.0
    bp_rp_comp_med = float("nan")
    ct_n_comp = 0
    ct_ok = False
    mag_calib_ct = mag_calib.copy()
    _c1_stderr = float("nan")
    from k2_extinction import (  # noqa: PLC0415
        K2Source,
        apply_k2_per_frame,
        apply_k2_to_comp_mag_inst,
        bp_rp_comp_median,
        resolve_k2_bprp_value,
    )

    _obs_group = str(getattr(ctx.state, "obs_group", "") or "")
    _k2_val, _k2_src = resolve_k2_bprp_value(_cfg, _obs_group)
    _bp_med_k2 = bp_rp_comp_median(ctx.comp_bp_rp, comp_quality)
    if (
        _k2_src in (K2Source.LITERATURE_DEFAULT, K2Source.NIGHT_FIT)
        and math.isfinite(float(_k2_val))
        and math.isfinite(_bp_med_k2)
    ):
        comp_lc = apply_k2_to_comp_mag_inst(
            comp_lc,
            ctx.comp_bp_rp,
            comp_quality,
            ctx.airmass_arr,
            float(_k2_val),
            _bp_med_k2,
            k2_source=_k2_src,
        )
    k2_value_lc = float("nan")
    k2_colour_ref = float("nan")
    k2_source_rows = [K2Source.NONE.value] * len(mag_calib)
    apply_ct = False
    if ctx.comp_bp_rp:
        c1, _c1_stderr, ct_n_comp = fit_color_term_c1(
            comp_lc,
            ctx.comp_catalog_mag,
            ctx.comp_bp_rp,
            comp_quality,
            min_comp=5,
            sigma_clip_sigma=3.0,
        )
        apply_ct, _ct_reason = should_apply_color_term(
            obs_group=_obs_group,
            c1=c1,
            c1_stderr=_c1_stderr,
            n_comp=ct_n_comp,
            min_comp_for_ct=int(_cfg.phase01_ct_min_comp),
        )
    if (
        _k2_src in (K2Source.LITERATURE_DEFAULT, K2Source.NIGHT_FIT)
        and math.isfinite(float(_k2_val))
        and math.isfinite(_bp_med_k2)
    ):
        mag_calib, _, k2_source_rows = apply_k2_per_frame(
            mag_calib,
            ctx.airmass_arr,
            object_bp_rp=float(target_bp_rp),
            bp_rp_comp_med=_bp_med_k2,
            k2_value=float(_k2_val),
            k2_source=_k2_src,
        )
        k2_value_lc = float(_k2_val)
        k2_colour_ref = _bp_med_k2
    if ctx.comp_bp_rp and apply_ct:
        mag_calib_ct, ct_corr, bp_rp_comp_med = apply_color_term(
            mag_calib,
            target_bp_rp,
            ctx.comp_bp_rp,
            comp_quality,
            c1,
        )
        ct_ok = (
            bool(math.isfinite(float(target_bp_rp)))
            and float(c1) != 0.0
            and math.isfinite(float(bp_rp_comp_med))
        )

    if "flag" in ctx.target_frames.columns:
        _raw_tf = ctx.target_frames["flag"].astype(str).str.strip().str.lower().reset_index(drop=True)
    else:
        _raw_tf = pd.Series(["__none__"] * len(mag_calib))
    base_flags: list[str] = []
    for i in range(len(mag_calib)):
        if bool(ctx.sat_flags[i]):
            base_flags.append("saturated")
        elif i < len(_raw_tf) and str(_raw_tf.iloc[i]) == "nondetection":
            base_flags.append("nondetection")
        elif math.isfinite(mag_calib[i]):
            base_flags.append("normal")
        else:
            base_flags.append("no_data")

    mag_calib_raw, mag_calib, mag_calib_ct, mag_calib_ac, out_flags = apply_reporting_postprocess(
        mag_calib,
        mag_calib_ct,
        target_row=ctx.target_row,
        target_name=str(ctx.target_row.get("vsx_name", ctx.target_cid)),
        sat_flags=ctx.sat_flags,
        target_frames=ctx.target_frames,
        outlier_sigma=ctx.outlier_sigma,
        ct_ok=bool(ct_ok),
        ac_ok=bool(ac_ok),
        delta_m_corr=(float(delta_m_corr) if delta_m_corr is not None else None),
        cfg=_cfg,
    )

    if bool(_cfg.savgol_detrend_enabled):
        mag_calib = savgol_detrend_lc(
            mag_calib=mag_calib,
            bjd=ctx.bjd,
            flags=list(out_flags),
            window_frac=float(_cfg.savgol_window_frac),
            polyorder=int(_cfg.savgol_polyorder),
            enabled=True,
        )
        if ac_ok and delta_m_corr is not None and np.isfinite(float(delta_m_corr)):
            mag_calib_ac = mag_calib + float(delta_m_corr)

    _mag_democratic = None
    _err_inflation = None
    if bool(_cfg.democratic_detrend_enabled):
        _mag_democratic, _err_inflation = democratic_detrend_lc(
            mag_calib=mag_calib,
            bjd=ctx.bjd,
            airmass=ctx.airmass_arr,
            flags=list(out_flags),
            window_frac=float(_cfg.democratic_sg_window_frac),
            enabled=True,
        )

    out_path = lc_csv_path(ctx.lc_dir, ctx.target_cid, method)
    save_lightcurve_csv(
        out_path,
        ctx.bjd,
        ctx.hjd,
        ctx.jd,
        ctx.airmass_arr,
        ctx.flip_arr,
        target_lc,
        mag_calib_raw,
        mag_calib,
        np.asarray(mag_calib_ct, dtype=np.float64),
        mag_calib_ac,
        delta_mag,
        ctx.err,
        ctx.ap_arr,
        out_flags,
        ctx.src_files,
        ct_correction=(float(ct_corr) if bool(ct_ok) else float("nan")),
        ct_c1=(float(c1) if bool(ct_ok) else float("nan")),
        ct_bp_rp_target=(float(target_bp_rp) if bool(ct_ok) else float("nan")),
        ct_bp_rp_comp_med=(float(bp_rp_comp_med) if bool(ct_ok) else float("nan")),
        ct_n_comp=(int(ct_n_comp) if bool(ct_ok) else None),
        ct_ok=bool(ct_ok),
        k2_source=k2_source_rows,
        k2_value=(float(k2_value_lc) if math.isfinite(float(k2_value_lc)) else float("nan")),
        k2_colour_ref=(float(k2_colour_ref) if math.isfinite(float(k2_colour_ref)) else float("nan")),
        ac_result=(ctx.ac_result if isinstance(ctx.ac_result, dict) else None),
        mag_democratic=_mag_democratic,
        err_inflation=_err_inflation,
        lunar_phase_pct=float(ctx.lunar_phase_pct),
        lunar_separation_deg=float(ctx.lunar_separation_deg),
        lunar_risk=str(ctx.lunar_risk),
        dilution_factor=float(_dilution_result.get("dilution_factor", 1.0)),
        method=method,
        time_base=str(ctx.time_base),
    )
    LOGGER.info("[METHOD-LC] %s %s -> %s", method, ctx.target_cid, out_path.name)
    return out_path
