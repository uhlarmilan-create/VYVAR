"""Noiseless V3d sweep for PSF sky estimators A/B/C (harness-only)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table
from photutils.psf import ImagePSF, PSFPhotometry

from psf_photometry import (
    _MASTERSTAR_EPSF_META_NAME,
    _annulus_sky_per_px_custom,
    _annulus_sky_per_px_full_frame,
    _fit_shape_for_cutout,
    _residual_annulus_sky_per_px,
    _subtract_psf_models,
)
from tests.validation.v3d_bias_decomposition_v2 import build_isolated_frame_noiseless
from tests.validation.v3d_fine_scale import (
    V3dFineConfig,
    _extract_cutout,
    calibrate_psf_aperture_correction,
    mag_to_flux,
    write_epsf_artifacts,
    build_epsf_training_frame,
    aperture_correction_factor,
)


def _fit_post_ac_bias_pct(
    frame: np.ndarray,
    epsf_path: Path,
    cfg: V3dFineConfig,
    meta: dict[str, Any],
    true_flux: float,
    psf_ac: float,
    *,
    sky_per_px: float,
    sky_method: str,
) -> dict[str, Any]:
    osamp = int(meta.get("oversampling", 2))
    fwhm_meta = float(meta.get("fwhm_px", cfg.fwhm_px))
    psf_data = np.asarray(fits.getdata(epsf_path), dtype=np.float64)
    cut, xc, yc = _extract_cutout(frame, cfg)
    cut_sub = cut - float(sky_per_px)
    fit_shape = _fit_shape_for_cutout(cfg.cutout_size(), fwhm_px=fwhm_meta)
    model = ImagePSF(psf_data, oversampling=osamp)
    phot = PSFPhotometry(model, fit_shape=fit_shape, progress_bar=False)
    flux_guess = float(np.nansum(np.clip(cut_sub, 0.0, None)))
    init = Table([[xc], [yc], [flux_guess]], names=("x_0", "y_0", "flux_0"))
    res = phot(data=cut_sub, init_params=init)
    pre = float(res["flux_fit"][0])
    post = pre * psf_ac
    return {
        "pre_ac_bias_pct": (pre / true_flux - 1.0) * 100.0,
        "post_ac_bias_pct": (post / true_flux - 1.0) * 100.0,
        "sky_method": sky_method,
    }


def _sky_option_a(
    frame: np.ndarray,
    cfg: V3dFineConfig,
    meta: dict[str, Any],
    *,
    inner_fwhm: float,
    outer_fwhm: float,
) -> tuple[float, str]:
    fwhm = float(meta.get("fwhm_px", cfg.fwhm_px))
    return _annulus_sky_per_px_custom(
        frame,
        float(cfg.stamp_c),
        float(cfg.stamp_c),
        fwhm_px=fwhm,
        inner_fwhm=inner_fwhm,
        outer_fwhm=outer_fwhm,
    )


def _sky_option_b(
    frame: np.ndarray,
    epsf_path: Path,
    cfg: V3dFineConfig,
    meta: dict[str, Any],
    true_flux: float,
) -> tuple[float, str]:
    """Model wing subtraction: annulus median after subtracting ePSF wing from preliminary fit."""
    osamp = int(meta.get("oversampling", 2))
    fwhm_meta = float(meta.get("fwhm_px", cfg.fwhm_px))
    psf_data = np.asarray(fits.getdata(epsf_path), dtype=np.float64)
    model = ImagePSF(psf_data, oversampling=osamp)
    sky0, _ = _annulus_sky_per_px_full_frame(
        frame, float(cfg.stamp_c), float(cfg.stamp_c), fwhm_px=fwhm_meta
    )
    cut, xc, yc = _extract_cutout(frame, cfg)
    cut_sub = cut - sky0
    fit_shape = _fit_shape_for_cutout(cfg.cutout_size(), fwhm_px=fwhm_meta)
    phot = PSFPhotometry(model, fit_shape=fit_shape, progress_bar=False)
    flux_guess = float(np.nansum(np.clip(cut_sub, 0.0, None)))
    init = Table([[xc], [yc], [flux_guess]], names=("x_0", "y_0", "flux_0"))
    res = phot(data=cut_sub, init_params=init)
    flux1 = float(res["flux_fit"][0])
    x1 = int(cfg.stamp_c - cfg.cutout_size() // 2)
    y1 = int(cfg.stamp_c - cfg.cutout_size() // 2)
    xf = float(res["x_fit"][0]) + x1
    yf = float(res["y_fit"][0]) + y1
    residual = _subtract_psf_models(frame, model, [(xf, yf, flux1)])
    from psf_photometry import _psf_annulus_radii_px, _annulus_median_per_px

    _, r_in, r_out = _psf_annulus_radii_px(fwhm_meta)
    sky = _annulus_median_per_px(
        residual, float(cfg.stamp_c), float(cfg.stamp_c), r_in=r_in, r_out=r_out
    )
    if math.isfinite(sky):
        return sky, "model_wing_sub"
    return sky0, "annulus_local"


def _sky_option_c(
    frame: np.ndarray,
    epsf_path: Path,
    cfg: V3dFineConfig,
    meta: dict[str, Any],
) -> tuple[float, str]:
    """Residual annulus after 1 iterative refine (production candidate)."""
    osamp = int(meta.get("oversampling", 2))
    fwhm_meta = float(meta.get("fwhm_px", cfg.fwhm_px))
    psf_data = np.asarray(fits.getdata(epsf_path), dtype=np.float64)
    model = ImagePSF(psf_data, oversampling=osamp)
    sky0, _ = _annulus_sky_per_px_full_frame(
        frame, float(cfg.stamp_c), float(cfg.stamp_c), fwhm_px=fwhm_meta
    )
    cut, xc, yc = _extract_cutout(frame, cfg)
    cut_sub = cut - sky0
    fit_shape = _fit_shape_for_cutout(cfg.cutout_size(), fwhm_px=fwhm_meta)
    phot = PSFPhotometry(model, fit_shape=fit_shape, progress_bar=False)
    flux_guess = float(np.nansum(np.clip(cut_sub, 0.0, None)))
    init = Table([[xc], [yc], [flux_guess]], names=("x_0", "y_0", "flux_0"))
    res = phot(data=cut_sub, init_params=init)
    flux1 = float(res["flux_fit"][0])
    x1 = int(cfg.stamp_c - cfg.cutout_size() // 2)
    y1 = int(cfg.stamp_c - cfg.cutout_size() // 2)
    xf = float(res["x_fit"][0]) + x1
    yf = float(res["y_fit"][0]) + y1
    sky1, meth = _residual_annulus_sky_per_px(
        frame,
        float(cfg.stamp_c),
        float(cfg.stamp_c),
        fwhm_px=fwhm_meta,
        psf_model=model,
        sources=[(xf, yf, flux1)],
    )
    if meth == "residual_annulus" and math.isfinite(sky1):
        cut_sub2 = cut - sky1
        flux_guess2 = float(np.nansum(np.clip(cut_sub2, 0.0, None)))
        init2 = Table([[xc], [yc], [flux_guess2]], names=("x_0", "y_0", "flux_0"))
        res2 = phot(data=cut_sub2, init_params=init2)
        flux2 = float(res2["flux_fit"][0])
        xf2 = float(res2["x_fit"][0]) + x1
        yf2 = float(res2["y_fit"][0]) + y1
        sky2, meth2 = _residual_annulus_sky_per_px(
            frame,
            float(cfg.stamp_c),
            float(cfg.stamp_c),
            fwhm_px=fwhm_meta,
            psf_model=model,
            sources=[(xf2, yf2, flux2)],
        )
        if meth2 == "residual_annulus" and math.isfinite(sky2):
            return sky2, meth2
        return sky1, meth
    return sky0, "annulus_local"


def run_sky_option_sweep(
    cfg: V3dFineConfig | None = None,
    *,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    cfg = cfg or V3dFineConfig()
    work_dir = Path(work_dir or Path(__file__).resolve().parent / "data" / "tier_v3d" / "_work_sky_sweep")
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.rng_seed)
    epsf_frame, epsf_cat = build_epsf_training_frame(rng, cfg)
    epsf_path = write_epsf_artifacts(work_dir, epsf_frame, epsf_cat, cfg)
    psf_ac, _ = calibrate_psf_aperture_correction(
        epsf_path, cfg, np.random.default_rng(cfg.rng_seed + 1), apcor=aperture_correction_factor(cfg)
    )
    meta = __import__("json").loads((work_dir / _MASTERSTAR_EPSF_META_NAME).read_text(encoding="ascii"))

    option_a_specs = [
        (4.75, 9.0),
        (9.0, 13.0),
        (12.0, 16.0),
        (15.0, 18.0),
        (18.0, 21.0),
    ]
    variants: dict[str, Any] = {
        "baseline_annulus": {"inner": 4.75, "outer": 9.0, "kind": "A_baseline"},
    }
    for inn, out in option_a_specs:
        if inn == 4.75 and out == 9.0:
            continue
        variants[f"A_rin{inn}_rout{out}"] = {"inner": inn, "outer": out, "kind": "A"}
    variants["B_model_wing_sub"] = {"kind": "B"}
    variants["C_residual_annulus"] = {"kind": "C"}

    out: dict[str, Any] = {"variants": {}, "psf_ac": psf_ac}
    for vname, vspec in variants.items():
        rows: list[dict[str, Any]] = []
        for mag in cfg.mags:
            if mag > 17:
                continue
            tf = mag_to_flux(mag, cfg.zp)
            frame = build_isolated_frame_noiseless(tf, cfg)
            kind = vspec["kind"]
            if kind.startswith("A"):
                sky, sm = _sky_option_a(
                    frame, cfg, meta, inner_fwhm=float(vspec["inner"]), outer_fwhm=float(vspec["outer"])
                )
            elif kind == "B":
                sky, sm = _sky_option_b(frame, epsf_path, cfg, meta, tf)
            else:
                sky, sm = _sky_option_c(frame, epsf_path, cfg, meta)
            m = _fit_post_ac_bias_pct(frame, epsf_path, cfg, meta, tf, psf_ac, sky_per_px=sky, sky_method=sm)
            rows.append({"mag": mag, **m})
        drift = float(rows[-1]["post_ac_bias_pct"] - rows[0]["post_ac_bias_pct"]) if rows else float("nan")
        mid = float(np.mean([r["post_ac_bias_pct"] for r in rows if 14 <= r["mag"] <= 16]))
        out["variants"][vname] = {
            "spec": vspec,
            "table": rows,
            "post_ac_drift_mag16_minus_12_pp": drift,
            "mid_mag_post_ac_mean_pct": mid,
        }
    return out
