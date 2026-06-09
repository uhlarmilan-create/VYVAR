"""V3d empirical mid-mag bias decomposition (T1-T4) on real psf_photometry_stars.

Harness-only; seed 367; no production changes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from psf_photometry import (
    _MASTERSTAR_EPSF_META_NAME,
    _fit_shape_for_cutout,
    _resolve_psf_fit_sky,
    psf_photometry_stars,
)
from tests.validation.gen_frame import moffat_stamp
from tests.validation.v3d_fine_scale import (
    MOFFAT_BETA,
    V3dFineConfig,
    _error_map,
    _extract_cutout,
    _median_bias_pct,
    _rng_for,
    aperture_correction_factor,
    build_epsf_training_frame,
    calibrate_psf_aperture_correction,
    mag_to_flux,
    write_epsf_artifacts,
)

MID_MAG = 15
T3_FIT_SHAPES: tuple[int, ...] = (7, 9, 11, 13, 15, 17, 21, 25, 31)
T4_NOISE_SCALES: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def build_isolated_frame_noiseless(true_flux: float, cfg: V3dFineConfig) -> np.ndarray:
    """Moffat + sky pedestal; no Poisson or read noise."""
    img = moffat_stamp(
        cfg.stamp_c,
        cfg.stamp_c,
        true_flux,
        cfg.fwhm_px,
        MOFFAT_BETA,
        ny=cfg.stamp_n,
        nx=cfg.stamp_n,
    )
    img += float(cfg.sky_adu)
    return np.asarray(img, dtype=np.float64)


def build_isolated_frame_noise_scaled(
    true_flux: float,
    rng: np.random.Generator,
    cfg: V3dFineConfig,
    *,
    noise_scale: float,
) -> np.ndarray:
    """Noisy frame with read noise and Poisson scaled by noise_scale (0 = noiseless)."""
    img = build_isolated_frame_noiseless(true_flux, cfg)
    if noise_scale <= 0.0:
        return img
    el = np.clip(img * cfg.gain, 0.0, None)
    img = rng.poisson(el).astype(np.float64) / cfg.gain
    rn = float(cfg.read_noise_e) / float(cfg.gain) * float(noise_scale)
    if rn > 0:
        img += rng.normal(0.0, rn, size=img.shape)
    return img


def _shared_assets(
    cfg: V3dFineConfig,
    work_dir: Path,
) -> tuple[Path, float, dict[str, Any]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    rng_epsf = np.random.default_rng(cfg.rng_seed)
    epsf_frame, epsf_cat = build_epsf_training_frame(rng_epsf, cfg)
    epsf_path = write_epsf_artifacts(work_dir, epsf_frame, epsf_cat, cfg)
    psf_ac, psf_ac_n = calibrate_psf_aperture_correction(
        epsf_path,
        cfg,
        np.random.default_rng(cfg.rng_seed + 1),
        apcor=aperture_correction_factor(cfg),
    )
    meta = json.loads((work_dir / _MASTERSTAR_EPSF_META_NAME).read_text(encoding="ascii"))
    return epsf_path, float(psf_ac), meta


def _measure_psf_fluxes(
    frame: np.ndarray,
    epsf_path: Path,
    cfg: V3dFineConfig,
    true_flux: float,
    *,
    psf_ac: float,
    error: np.ndarray | None = None,
) -> dict[str, float]:
    pos = pd.DataFrame(
        [
            {
                "catalog_id": "inj_0001",
                "name": "target",
                "x": float(cfg.stamp_c),
                "y": float(cfg.stamp_c),
            }
        ]
    )
    hdr = fits.Header()
    hdr["GAIN"] = float(cfg.gain)
    hdr["RDNOISE"] = float(cfg.read_noise_e)
    err = error if error is not None else _error_map(frame, cfg)
    df = psf_photometry_stars(
        np.asarray(frame, dtype=np.float64),
        hdr,
        pos,
        epsf_path,
        cutout_size=cfg.cutout_size(),
        error=err,
        use_iterative=True,
        apply_aperture_correction=False,
        grouper_enabled=False,
        quality_fallback_enabled=False,
    )
    row = df.iloc[0]
    pre = float(row.get("psf_flux", float("nan")))
    post = pre * float(psf_ac)
    tf = float(true_flux)
    return {
        "pre_ac_flux": pre,
        "post_ac_flux": post,
        "pre_ac_bias_pct": (pre / tf - 1.0) * 100.0 if tf > 0 and math.isfinite(pre) else float("nan"),
        "post_ac_bias_pct": (post / tf - 1.0) * 100.0 if tf > 0 and math.isfinite(post) else float("nan"),
        "psf_sky_method": str(row.get("psf_sky_method", "")),
    }


def _bias_vs_mag_table(
    cfg: V3dFineConfig,
    epsf_path: Path,
    psf_ac: float,
    frame_builder: Callable[[float, np.random.Generator], np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mag in cfg.mags:
        tf = mag_to_flux(mag, cfg.zp)
        pre_ratios: list[float] = []
        post_ratios: list[float] = []
        for ireal in range(cfg.n_real):
            rng = _rng_for(mag, ireal, base=cfg.rng_seed)
            frame = frame_builder(tf, rng)
            m = _measure_psf_fluxes(frame, epsf_path, cfg, tf, psf_ac=psf_ac)
            if math.isfinite(m["pre_ac_bias_pct"]):
                pre_ratios.append(1.0 + m["pre_ac_bias_pct"] / 100.0)
            if math.isfinite(m["post_ac_bias_pct"]):
                post_ratios.append(1.0 + m["post_ac_bias_pct"] / 100.0)
        pre_arr = np.array(pre_ratios, dtype=float)
        post_arr = np.array(post_ratios, dtype=float)
        rows.append(
            {
                "mag": mag,
                "pre_ac_bias_pct": _median_bias_pct(pre_arr) if pre_arr.size else float("nan"),
                "post_ac_bias_pct": _median_bias_pct(post_arr) if post_arr.size else float("nan"),
                "n": cfg.n_real,
            }
        )
    return rows


def run_t1_noiseless(
    cfg: V3dFineConfig,
    epsf_path: Path,
    psf_ac: float,
) -> dict[str, Any]:
    def _builder(tf: float, _rng: np.random.Generator) -> np.ndarray:
        return build_isolated_frame_noiseless(tf, cfg)

    table = _bias_vs_mag_table(cfg, epsf_path, psf_ac, _builder)
    mid_post = [r["post_ac_bias_pct"] for r in table if 14 <= r["mag"] <= 16]
    mid_max = max(abs(v) for v in mid_post) if mid_post else float("nan")
    vanishes = math.isfinite(mid_max) and mid_max < 1.5
    return {
        "test": "T1_noiseless",
        "table": table,
        "mid_mag_post_ac_max_abs_pct": float(mid_max),
        "bias_vanishes": bool(vanishes),
        "branch": "T4_noise_driven" if vanishes else "T2_T3_deterministic",
    }


def run_t2_normalization(
    cfg: V3dFineConfig,
    epsf_path: Path,
    meta: dict[str, Any],
    *,
    mag: int = MID_MAG,
) -> dict[str, Any]:
    from photutils.psf import ImagePSF, PSFPhotometry

    true_flux = mag_to_flux(mag, cfg.zp)
    frame = build_isolated_frame_noiseless(true_flux, cfg)
    m = _measure_psf_fluxes(frame, epsf_path, cfg, true_flux, psf_ac=1.0, error=None)
    reported = float(m["pre_ac_flux"])

    osamp = int(meta.get("oversampling", 2))
    fwhm_meta = float(meta.get("fwhm_px", cfg.fwhm_px))
    psf_data = np.asarray(fits.getdata(epsf_path), dtype=np.float64)
    epsf_norm_sum = float(np.sum(psf_data)) / float(osamp**2)

    cut, xc, yc = _extract_cutout(frame, cfg)
    sky, _ = _resolve_psf_fit_sky(frame, cut, float(cfg.stamp_c), float(cfg.stamp_c), fwhm_px=fwhm_meta)
    cut_sub = cut - sky
    fit_shape = _fit_shape_for_cutout(cfg.cutout_size(), fwhm_px=fwhm_meta)
    model = ImagePSF(psf_data, oversampling=osamp)
    phot = PSFPhotometry(model, fit_shape=fit_shape, progress_bar=False)
    flux_guess = float(np.nansum(np.clip(cut_sub, 0.0, None)))
    init = Table([[xc], [yc], [flux_guess]], names=("x_0", "y_0", "flux_0"))
    res = phot(data=cut_sub, init_params=init)
    flux_fit = float(res["flux_fit"][0])

    y_ix, x_ix = np.mgrid[0 : cfg.stamp_n, 0 : cfg.stamp_n]
    model_stamp = model.evaluate(
        x_ix.astype(np.float64),
        y_ix.astype(np.float64),
        np.full((cfg.stamp_n, cfg.stamp_n), flux_fit, dtype=np.float64),
        np.full((cfg.stamp_n, cfg.stamp_n), float(cfg.stamp_c), dtype=np.float64),
        np.full((cfg.stamp_n, cfg.stamp_n), float(cfg.stamp_c), dtype=np.float64),
    )
    model_stamp_sum = float(np.nansum(model_stamp))

    return {
        "test": "T2_normalization",
        "mag": mag,
        "true_flux": true_flux,
        "reported_psf_flux": reported,
        "fit_flux_direct": flux_fit,
        "model_stamp_sum": model_stamp_sum,
        "epsf_norm_sum_over_osamp2": epsf_norm_sum,
        "ratio_reported_to_truth": reported / true_flux,
        "ratio_fit_to_truth": flux_fit / true_flux,
        "ratio_model_sum_to_truth": model_stamp_sum / true_flux,
        "ratio_epsf_norm_to_truth": epsf_norm_sum / true_flux,
        "reported_over_fit": reported / flux_fit if flux_fit > 0 else float("nan"),
        "fit_shape": list(fit_shape),
        "oversampling": osamp,
    }


def _fit_with_shape(
    frame: np.ndarray,
    epsf_path: Path,
    cfg: V3dFineConfig,
    meta: dict[str, Any],
    true_flux: float,
    fit_shape: tuple[int, int],
    *,
    fix_position: bool = False,
    error: np.ndarray | None = None,
    sky_per_px: float | None = None,
) -> float:
    from photutils.psf import ImagePSF, PSFPhotometry

    osamp = int(meta.get("oversampling", 2))
    fwhm_meta = float(meta.get("fwhm_px", cfg.fwhm_px))
    psf_data = np.asarray(fits.getdata(epsf_path), dtype=np.float64)
    cut, xc, yc = _extract_cutout(frame, cfg)
    if sky_per_px is not None:
        sky = float(sky_per_px)
    else:
        sky, _ = _resolve_psf_fit_sky(frame, cut, float(cfg.stamp_c), float(cfg.stamp_c), fwhm_px=fwhm_meta)
    cut_sub = cut - sky
    model = ImagePSF(psf_data, oversampling=osamp)
    phot = PSFPhotometry(model, fit_shape=fit_shape, progress_bar=False)
    if fix_position:
        phot.psf_model.x_0.fixed = True
        phot.psf_model.y_0.fixed = True
    flux_guess = float(np.nansum(np.clip(cut_sub, 0.0, None)))
    init = Table([[xc], [yc], [flux_guess]], names=("x_0", "y_0", "flux_0"))
    if error is not None:
        half = cfg.cutout_size() // 2
        c = cfg.stamp_c
        err_cut = error[c - half : c + half + 1, c - half : c + half + 1]
    else:
        err_cut = None
    res = phot(data=cut_sub, init_params=init, error=err_cut)
    return float(res["flux_fit"][0])


def run_t3_fit_shape_sweep(
    cfg: V3dFineConfig,
    epsf_path: Path,
    meta: dict[str, Any],
    *,
    mag: int = MID_MAG,
) -> dict[str, Any]:
    true_flux = mag_to_flux(mag, cfg.zp)
    frame = build_isolated_frame_noiseless(true_flux, cfg)
    prod_shape = _fit_shape_for_cutout(cfg.cutout_size(), fwhm_px=float(meta.get("fwhm_px", cfg.fwhm_px)))
    rows: list[dict[str, Any]] = []
    for fs in T3_FIT_SHAPES:
        shape = (int(fs), int(fs))
        flux = _fit_with_shape(frame, epsf_path, cfg, meta, true_flux, shape)
        rows.append(
            {
                "fit_shape": fs,
                "flux_fit": flux,
                "bias_pct": (flux / true_flux - 1.0) * 100.0,
                "is_production_default": shape == prod_shape,
            }
        )
    biases = [r["bias_pct"] for r in rows]
    spread = float(max(biases) - min(biases)) if biases else float("nan")
    return {
        "test": "T3_fit_shape",
        "mag": mag,
        "production_fit_shape": list(prod_shape),
        "table": rows,
        "bias_spread_pct": spread,
        "truncation_sensitive": math.isfinite(spread) and spread > 1.0,
    }


def run_t4_noise_characterization(
    cfg: V3dFineConfig,
    epsf_path: Path,
    psf_ac: float,
    meta: dict[str, Any],
    *,
    mag: int = MID_MAG,
) -> dict[str, Any]:
    true_flux = mag_to_flux(mag, cfg.zp)
    prod_shape = _fit_shape_for_cutout(cfg.cutout_size(), fwhm_px=float(meta.get("fwhm_px", cfg.fwhm_px)))

    noise_rows: list[dict[str, Any]] = []
    for scale in T4_NOISE_SCALES:
        post_biases: list[float] = []
        for ireal in range(cfg.n_real):
            rng = _rng_for(mag, ireal, base=cfg.rng_seed)
            frame = build_isolated_frame_noise_scaled(true_flux, rng, cfg, noise_scale=scale)
            m = _measure_psf_fluxes(frame, epsf_path, cfg, true_flux, psf_ac=psf_ac)
            if math.isfinite(m["post_ac_bias_pct"]):
                post_biases.append(m["post_ac_bias_pct"])
        med = float(np.median(post_biases)) if post_biases else float("nan")
        noise_rows.append({"noise_scale": scale, "post_ac_bias_pct": med})

    frame_nom = build_isolated_frame_noise_scaled(
        true_flux, np.random.default_rng(cfg.rng_seed), cfg, noise_scale=1.0
    )
    err_nom = _error_map(frame_nom, cfg)
    frame_nl = build_isolated_frame_noiseless(true_flux, cfg)

    flux_free = _fit_with_shape(frame_nom, epsf_path, cfg, meta, true_flux, prod_shape, error=err_nom)
    flux_fixed = _fit_with_shape(
        frame_nom, epsf_path, cfg, meta, true_flux, prod_shape, fix_position=True, error=err_nom
    )
    flux_free_nl = _fit_with_shape(frame_nl, epsf_path, cfg, meta, true_flux, prod_shape, error=None)
    flux_fixed_nl = _fit_with_shape(
        frame_nl, epsf_path, cfg, meta, true_flux, prod_shape, fix_position=True, error=None
    )

    # Weighting: production passes error= when available (inverse-variance in photutils).
    pos = pd.DataFrame([{"catalog_id": "t", "name": "t", "x": float(cfg.stamp_c), "y": float(cfg.stamp_c)}])
    df_w = psf_photometry_stars(
        frame_nom,
        fits.Header(),
        pos,
        epsf_path,
        cutout_size=cfg.cutout_size(),
        error=err_nom,
        apply_aperture_correction=False,
        grouper_enabled=False,
        quality_fallback_enabled=False,
    )
    df_uw = psf_photometry_stars(
        frame_nom,
        fits.Header(),
        pos,
        epsf_path,
        cutout_size=cfg.cutout_size(),
        error=None,
        apply_aperture_correction=False,
        grouper_enabled=False,
        quality_fallback_enabled=False,
    )

    return {
        "test": "T4_noise_characterization",
        "mag": mag,
        "noise_scale_table": noise_rows,
        "position_anchor": {
            "noisy_free_bias_pct": (flux_free / true_flux - 1.0) * 100.0,
            "noisy_fixed_bias_pct": (flux_fixed / true_flux - 1.0) * 100.0,
            "noiseless_free_bias_pct": (flux_free_nl / true_flux - 1.0) * 100.0,
            "noiseless_fixed_bias_pct": (flux_fixed_nl / true_flux - 1.0) * 100.0,
        },
        "weighting": {
            "production_mode": "inverse_variance when error map passed to photutils PSFPhotometry",
            "weighted_flux": float(df_w.iloc[0]["psf_flux"]),
            "unweighted_flux": float(df_uw.iloc[0]["psf_flux"]),
            "weighted_bias_pct": (float(df_w.iloc[0]["psf_flux"]) / true_flux - 1.0) * 100.0,
            "unweighted_bias_pct": (float(df_uw.iloc[0]["psf_flux"]) / true_flux - 1.0) * 100.0,
        },
    }


def _identify_cause(result: dict[str, Any]) -> dict[str, str]:
    t1 = result["t1"]
    t2 = result.get("t2")
    t3 = result.get("t3")
    t4 = result.get("t4")

    if t1["bias_vanishes"]:
        pos = t4["position_anchor"] if t4 else {}
        w = t4["weighting"] if t4 else {}
        delta_pos = abs(pos.get("noisy_free_bias_pct", 0) - pos.get("noisy_fixed_bias_pct", 0))
        delta_w = abs(w.get("weighted_bias_pct", 0) - w.get("unweighted_bias_pct", 0))
        if delta_pos > 1.0:
            cause = "noise_driven_free_position"
            fix = "Anchor PSF fit position to catalog/WCS at fine scale when astrometry is trusted."
        elif delta_w > 1.0:
            cause = "noise_driven_weighting"
            fix = "Review inverse-variance weighting in psf_photometry_stars error map at low SNR."
        else:
            cause = "noise_driven_general"
            fix = "Characterize SNR-dependent bias; consider position anchor and/or noise bias correction."
        return {
            "identified_cause": cause,
            "supported_by": "T1 bias vanishes noiseless; T4 characterization",
            "proposed_fix": fix,
            "implement_fix": "separate task after review of T4 tables",
        }

    # Deterministic branch
    prod_row = next((r for r in (t3 or {}).get("table", []) if r.get("is_production_default")), None)
    large_row = max((t3 or {}).get("table", []), key=lambda r: r["fit_shape"], default=None)

    if t3 and t3.get("truncation_sensitive"):
        cause = "deterministic_fit_shape_truncation"
        prod_b = float(prod_row["bias_pct"]) if prod_row else float("nan")
        large_b = float(large_row["bias_pct"]) if large_row else float("nan")
        fix = (
            f"Enlarge fit_shape beyond production {t3['production_fit_shape']}. "
            f"T3 noiseless mag {t3['mag']}: bias {prod_b:+.1f}% at default shape vs "
            f"{large_b:+.1f}% at fit_shape={large_row['fit_shape'] if large_row else 'n/a'} "
            f"(spread {t3['bias_spread_pct']:.1f}%). "
            "Truncated wings leave flux outside the fit window; a single AC cannot remove "
            "mag-dependent pre-AC ratio drift when the recovered fraction varies."
        )
    elif t2 and abs(t2["ratio_reported_to_truth"] - t2["ratio_fit_to_truth"]) > 0.02:
        cause = "deterministic_reported_vs_fit_mismatch"
        fix = (
            "Align psf_photometry_stars output with direct PSFPhotometry flux "
            f"(reported/fit={t2['reported_over_fit']:.4f} at mag {t2['mag']})."
        )
    elif t2 and abs(t2["ratio_reported_to_truth"] - 1.0) > 0.02:
        cause = "deterministic_epsf_profile_mismatch"
        fix = (
            f"ePSF unit normalization is correct (sum/osamp^2={t2['epsf_norm_sum_over_osamp2']:.4f}) "
            f"and reported~fit ({t2['reported_over_fit']:.4f}), but noiseless recovery/truth="
            f"{t2['ratio_reported_to_truth']:.4f}. Reconcile ePSF build profile with injected "
            "Moffat (V3d mismatch ratio ~1.05) and/or enlarge fit_shape."
        )
    else:
        cause = "deterministic_unresolved"
        fix = "Bias persists noiseless; inspect iterative PSF path and ePSF QC."

    supported = []
    if t1 and not t1["bias_vanishes"]:
        supported.append("T1 bias persists noiseless")
    if t2:
        supported.append(f"T2 ratio reported/truth={t2['ratio_reported_to_truth']:.4f}")
    if t3:
        supported.append(f"T3 fit_shape spread={t3['bias_spread_pct']:.2f}%")
    if t2:
        supported.append(f"T2 reported/fit={t2['reported_over_fit']:.4f}, epsf_norm={t2['epsf_norm_sum_over_osamp2']:.4f}")

    return {
        "identified_cause": cause,
        "supported_by": "; ".join(supported),
        "proposed_fix": fix,
        "implement_fix": "separate task -- single targeted fix from identified cause",
    }


def run_fallback_truth_sky_noiseless(
    cfg: V3dFineConfig,
    epsf_path: Path,
    psf_ac: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Noiseless mag sweep: production annulus sky vs truth sky forced in harness fit."""
    prod_shape = _fit_shape_for_cutout(cfg.cutout_size(), fwhm_px=float(meta.get("fwhm_px", cfg.fwhm_px)))
    rows: list[dict[str, Any]] = []
    for mag in cfg.mags:
        if mag > 17:
            continue
        tf = mag_to_flux(mag, cfg.zp)
        frame = build_isolated_frame_noiseless(tf, cfg)
        ann = _measure_psf_fluxes(frame, epsf_path, cfg, tf, psf_ac=psf_ac, error=None)
        flux_truth = _fit_with_shape(
            frame,
            epsf_path,
            cfg,
            meta,
            tf,
            prod_shape,
            error=None,
            sky_per_px=float(cfg.sky_adu),
        )
        post_truth = flux_truth * psf_ac
        rows.append(
            {
                "mag": mag,
                "annulus_pre_ac_pct": ann["pre_ac_bias_pct"],
                "annulus_post_ac_pct": ann["post_ac_bias_pct"],
                "truth_pre_ac_pct": (flux_truth / tf - 1.0) * 100.0,
                "truth_post_ac_pct": (post_truth / tf - 1.0) * 100.0,
            }
        )
    ann_drift = float(rows[-1]["annulus_post_ac_pct"] - rows[0]["annulus_post_ac_pct"]) if rows else float("nan")
    truth_drift = float(rows[-1]["truth_post_ac_pct"] - rows[0]["truth_post_ac_pct"]) if rows else float("nan")
    mid_ann = float(np.mean([r["annulus_post_ac_pct"] for r in rows if 14 <= r["mag"] <= 16]))
    mid_truth = float(np.mean([r["truth_post_ac_pct"] for r in rows if 14 <= r["mag"] <= 16]))
    return {
        "test": "fallback_truth_sky_noiseless",
        "table": rows,
        "annulus_post_ac_drift_mag16_minus_12_pp": ann_drift,
        "truth_post_ac_drift_mag16_minus_12_pp": truth_drift,
        "mid_mag_annulus_post_ac_mean_pct": mid_ann,
        "mid_mag_truth_post_ac_mean_pct": mid_truth,
        "drift_vanishes_with_truth_sky": abs(truth_drift) < 1.5,
    }


def run_v3d_bias_decomposition_v2(
    cfg: V3dFineConfig | None = None,
    *,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = cfg or V3dFineConfig()
    work_dir = Path(work_dir or Path(__file__).resolve().parent / "data" / "tier_v3d" / "_work_v2")
    epsf_path, psf_ac, meta = _shared_assets(cfg, work_dir)

    t1 = run_t1_noiseless(cfg, epsf_path, psf_ac)
    t2 = t3 = t4 = None
    if t1["branch"] == "T2_T3_deterministic":
        t2 = run_t2_normalization(cfg, epsf_path, meta)
        t3 = run_t3_fit_shape_sweep(cfg, epsf_path, meta)
    else:
        t4 = run_t4_noise_characterization(cfg, epsf_path, psf_ac, meta)

    result = {
        "config": {
            "rng_seed": cfg.rng_seed,
            "n_real": cfg.n_real,
            "mags": list(cfg.mags),
        },
        "psf_ac_factor": psf_ac,
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "t4": t4,
    }
    result["decision"] = _identify_cause(result)
    return result


def write_v3d_bias_decomposition_v2_report(
    out_dir: Path,
    result: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if result is None:
        result = run_v3d_bias_decomposition_v2(work_dir=out_dir / "_work_v2")

    jp = out_dir / "v3d_bias_decomposition_v2.json"
    with open(jp, "w", encoding="ascii") as f:
        json.dump(result, f, indent=2)

    t1 = result["t1"]
    dec = result["decision"]
    lines = [
        "# V3d PSF mid-mag bias -- empirical decomposition v2",
        "",
        "Real ``psf_photometry_stars`` on V3d fine-scale setup (seed 367). Harness-only.",
        "",
        f"PSF AC factor: **{result.get('psf_ac_factor', float('nan')):.6f}**",
        "",
        "## Decision",
        "",
        f"- **Identified cause:** {dec.get('identified_cause')}",
        f"- **Supported by:** {dec.get('supported_by')}",
        f"- **Proposed fix (not implemented):** {dec.get('proposed_fix')}",
        "",
        "## T1 -- Noiseless (decisive split)",
        "",
        f"- Mid-mag (14-16) max |post-AC bias|: **{t1.get('mid_mag_post_ac_max_abs_pct', float('nan')):.2f}%**",
        f"- Bias vanishes noiseless: **{t1.get('bias_vanishes')}** -> branch **{t1.get('branch')}**",
        "",
        "| mag | pre-AC bias % | post-AC bias % |",
        "|----:|--------------:|---------------:|",
    ]
    for row in t1.get("table", []):
        lines.append(
            f"| {row['mag']} | {row['pre_ac_bias_pct']:+.3f} | {row['post_ac_bias_pct']:+.3f} |"
        )

    t2 = result.get("t2")
    if t2:
        lines.extend(
            [
                "",
                "## T2 -- Normalization / integration (noiseless mid-mag)",
                "",
                f"Mag **{t2['mag']}**, true flux **{t2['true_flux']:.6g}**",
                "",
                "| quantity | value | ratio to truth |",
                "|:---------|------:|---------------:|",
                f"| reported psf_photometry_stars flux | {t2['reported_psf_flux']:.6g} | {t2['ratio_reported_to_truth']:.6f} |",
                f"| direct PSFPhotometry flux_fit | {t2['fit_flux_direct']:.6g} | {t2['ratio_fit_to_truth']:.6f} |",
                f"| fitted model sum (full stamp) | {t2['model_stamp_sum']:.6g} | {t2['ratio_model_sum_to_truth']:.6f} |",
                f"| ePSF sum / oversampling^2 | {t2['epsf_norm_sum_over_osamp2']:.6f} | {t2['ratio_epsf_norm_to_truth']:.6f} |",
                "",
                f"reported / fit_flux: **{t2['reported_over_fit']:.6f}**; production fit_shape: {t2['fit_shape']}",
            ]
        )

    t3 = result.get("t3")
    if t3:
        lines.extend(
            [
                "",
                f"## T3 -- fit_shape sweep (noiseless mag {t3['mag']})",
                "",
                f"Production fit_shape: **{t3['production_fit_shape']}**; bias spread: **{t3['bias_spread_pct']:.3f}%**; "
                f"truncation-sensitive: **{t3['truncation_sensitive']}**",
                "",
                "| fit_shape | bias % | prod default |",
                "|----------:|-------:|:------------:|",
            ]
        )
        for row in t3.get("table", []):
            mark = "Y" if row.get("is_production_default") else ""
            lines.append(f"| {row['fit_shape']} | {row['bias_pct']:+.3f} | {mark} |")

    t4 = result.get("t4")
    if t4:
        lines.extend(
            [
                "",
                f"## T4 -- Noise characterization (mag {t4['mag']})",
                "",
                "### bias vs noise scale (post-AC)",
                "",
                "| noise_scale | post-AC bias % |",
                "|------------:|---------------:|",
            ]
        )
        for row in t4.get("noise_scale_table", []):
            lines.append(f"| {row['noise_scale']:.2f} | {row['post_ac_bias_pct']:+.3f} |")
        pa = t4.get("position_anchor", {})
        lines.extend(
            [
                "",
                "### Position: free vs fixed (pre-AC bias %)",
                "",
                f"- noisy free: {pa.get('noisy_free_bias_pct', float('nan')):+.3f}%",
                f"- noisy fixed: {pa.get('noisy_fixed_bias_pct', float('nan')):+.3f}%",
                f"- noiseless free: {pa.get('noiseless_free_bias_pct', float('nan')):+.3f}%",
                f"- noiseless fixed: {pa.get('noiseless_fixed_bias_pct', float('nan')):+.3f}%",
                "",
                "### Weighting",
                "",
                f"- mode: {t4['weighting'].get('production_mode')}",
                f"- weighted bias: {t4['weighting'].get('weighted_bias_pct', float('nan')):+.3f}%",
                f"- unweighted bias: {t4['weighting'].get('unweighted_bias_pct', float('nan')):+.3f}%",
            ]
        )

    lines.append("")
    mp = out_dir / "v3d_bias_decomposition_v2.md"
    mp.write_text("\n".join(lines), encoding="ascii")
    return mp
