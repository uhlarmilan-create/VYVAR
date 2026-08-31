"""NEIGHBOR-SUB: joint-fit contaminating neighbour(s), subtract model, aperture target residual.

Gated OFF in production (``psf_neighbor_sub_enabled``). Step 2b will wire into per-frame
measurement; step 2 uses A9 validation scoring only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Moffat2D

from config import AppConfig
from photometry_core import _annulus_sky_subtracted_flux

_DEFAULT_NN_CONTAM_DMAG = 2.5
_DEFAULT_FLUX_ZP = 25.0


def _flux_to_mag_with_zp(flux: float, zp: float) -> float:
    """Instrumental mag with a zero-point; non-positive flux returns +inf (not nan)."""
    if not math.isfinite(flux) or flux <= 0.0:
        return float("inf")
    return float(zp) - 2.5 * math.log10(float(flux))


def _aperture_area(r_ap: float) -> float:
    return math.pi * float(r_ap) ** 2


def _aperture_noise_adu(
    stamp: np.ndarray,
    r_ap: float,
    *,
    gain_e_per_adu: float = 1.5,
    read_noise_e: float = 9.0,
) -> float:
    """Poisson + read noise in circular aperture (1-sigma ADU)."""
    sky = _stamp_sky_median(stamp)
    per_pix = math.sqrt(max(float(sky) / float(gain_e_per_adu), 0.0) + (float(read_noise_e) / float(gain_e_per_adu)) ** 2)
    return per_pix * math.sqrt(_aperture_area(r_ap))


@dataclass
class NeighborSubResult:
    target_flux: float
    plain_target_flux: float
    neighbor_subtracted: bool
    refused: bool
    refuse_reason: str
    n_neighbors_subtracted: int
    subtracted_neighbor_flux: float
    joint_fit_chi2: float
    residual_rms: float
    fit_condition: float
    target_x_fit: float
    target_y_fit: float


def _moffat_gamma(fwhm_px: float, beta: float) -> float:
    return float(fwhm_px) / (2.0 * math.sqrt(2.0 ** (1.0 / float(beta)) - 1.0))


def _stamp_sky_median(stamp: np.ndarray, margin: int = 3) -> float:
    h, w = stamp.shape
    if h <= 2 * margin or w <= 2 * margin:
        return float(np.nanmedian(stamp))
    border = np.concatenate(
        [
            stamp[:margin, :].ravel(),
            stamp[-margin:, :].ravel(),
            stamp[margin:-margin, :margin].ravel(),
            stamp[margin:-margin, -margin:].ravel(),
        ]
    )
    border = border[np.isfinite(border)]
    return float(np.median(border)) if border.size else float(np.nanmedian(stamp))


def _joint_moffat_fit_subtract(
    stamp: np.ndarray,
    *,
    target_xy: tuple[float, float],
    neighbour_xys: list[tuple[float, float]],
    fwhm_px: float,
    fit_beta: float,
    centroid_bound_fwhm: float = 1.0,
    r_ap: float | None = None,
    r_in: float | None = None,
    r_out: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Joint Moffat fit; subtract ONLY neighbour component(s) from stamp copy."""
    img = np.asarray(stamp, dtype=np.float64)
    h, w = img.shape
    sky = _stamp_sky_median(img)
    sub = img - sky
    yy, xx = np.mgrid[0:h, 0:w]
    gamma = _moffat_gamma(fwhm_px, fit_beta)
    models: list[Moffat2D] = []
    roles: list[str] = []

    tx, ty = float(target_xy[0]), float(target_xy[1])
    amp_t = float(max(sub[int(round(ty)), int(round(tx))], 1.0))
    mt = Moffat2D(amplitude=amp_t, x_0=tx, y_0=ty, gamma=gamma, alpha=float(fit_beta))
    mt.x_0.bounds = (tx - centroid_bound_fwhm * fwhm_px, tx + centroid_bound_fwhm * fwhm_px)
    mt.y_0.bounds = (ty - centroid_bound_fwhm * fwhm_px, ty + centroid_bound_fwhm * fwhm_px)
    mt.alpha.fixed = True
    mt.gamma.fixed = True
    models.append(mt)
    roles.append("target")

    for nx, ny in neighbour_xys:
        amp_n = float(max(sub[int(round(ny)), int(round(nx))], 1.0))
        mn = Moffat2D(amplitude=amp_n, x_0=float(nx), y_0=float(ny), gamma=gamma, alpha=float(fit_beta))
        mn.x_0.bounds = (nx - centroid_bound_fwhm * fwhm_px, nx + centroid_bound_fwhm * fwhm_px)
        mn.y_0.bounds = (ny - centroid_bound_fwhm * fwhm_px, ny + centroid_bound_fwhm * fwhm_px)
        mn.alpha.fixed = True
        mn.gamma.fixed = True
        models.append(mn)
        roles.append("neighbour")

    compound = models[0]
    for m in models[1:]:
        compound = compound + m

    fitter = LevMarLSQFitter()
    try:
        fitted = fitter(compound, xx, yy, sub, maxiter=400)
    except Exception as exc:  # noqa: BLE001
        # EXC-0445: ? -- intent unclear (try: / fitted = fitter(compound, xx, yy, sub, maxiter=400) / except Exc... (EXCEPT-BULK 2026-07-08)
        return img.copy(), {
            "ok": False,
            "error": str(exc),
            "joint_fit_chi2": float("nan"),
            "residual_rms": float("nan"),
            "fit_condition": float("inf"),
            "subtracted_neighbor_flux": 0.0,
            "target_x_fit": tx,
            "target_y_fit": ty,
            "n_neighbors_subtracted": 0,
        }

    neigh_model = np.zeros_like(sub, dtype=np.float64)
    n_sub = 0
    sub_flux = 0.0
    neighbor_fit_fluxes: list[float] = []
    neigh_idx = 0
    for i, role in enumerate(roles):
        if role != "neighbour":
            continue
        comp_i = fitted[i](xx, yy)
        neigh_model += comp_i
        n_sub += 1
        sub_flux += float(np.sum(comp_i))
        if r_ap is not None and r_in is not None and r_out is not None and neigh_idx < len(neighbour_xys):
            nx, ny = neighbour_xys[neigh_idx]
            nfit, _, _ = _annulus_sky_subtracted_flux(
                comp_i + sky, float(nx), float(ny), float(r_ap), float(r_in), float(r_out)
            )
            neighbor_fit_fluxes.append(float(nfit))
        neigh_idx += 1

    neigh_only = neigh_model
    residual_stamp = img - neigh_only
    resid_patch = residual_stamp - sky
    mask = (xx - tx) ** 2 + (yy - ty) ** 2 <= (1.5 * fwhm_px) ** 2
    rvals = resid_patch[mask]
    rvals = rvals[np.isfinite(rvals)]
    residual_rms = float(np.sqrt(np.mean(rvals**2))) if rvals.size else float("nan")
    dof = max(1, int(mask.sum()) - 2)
    chi2 = float(np.sum(rvals**2) / dof) if rvals.size else float("nan")

    t_amp = float(fitted[0].amplitude.value)
    n_amp = float(fitted[1].amplitude.value) if len(models) > 1 else float("nan")
    fit_cond = float(abs(t_amp) / max(abs(n_amp), 1e-9)) if math.isfinite(n_amp) else float("inf")

    return residual_stamp, {
        "ok": True,
        "joint_fit_chi2": chi2,
        "residual_rms": residual_rms,
        "fit_condition": fit_cond,
        "subtracted_neighbor_flux": sub_flux,
        "target_x_fit": float(fitted[0].x_0.value),
        "target_y_fit": float(fitted[0].y_0.value),
        "target_amplitude": t_amp,
        "n_neighbors_subtracted": n_sub,
        "neighbor_fit_fluxes": neighbor_fit_fluxes,
    }


def neighbor_sub_target_flux(
    stamp: np.ndarray,
    *,
    target_xy: tuple[float, float],
    neighbour_xys: list[tuple[float, float]],
    fwhm_px: float,
    r_ap: float,
    r_in: float,
    r_out: float,
    delta_mag_nn: float | None = None,
    nn_dist_fwhm: float | None = None,
    target_mag: float | None = None,
    nn_mag: float | None = None,
    flux_zp: float = _DEFAULT_FLUX_ZP,
    fit_beta: float = 2.5,
    cfg: AppConfig | None = None,
) -> NeighborSubResult:
    """Joint-fit subtract neighbour(s), aperture target on residual. Fail-safe to plain aperture."""
    cfg = cfg or AppConfig()
    tx, ty = float(target_xy[0]), float(target_xy[1])
    plain_flux, _, _ = _annulus_sky_subtracted_flux(stamp, tx, ty, r_ap, r_in, r_out)
    plain_flux = float(plain_flux)

    contam_dmag = float(getattr(cfg, "neighbor_sub_nn_contam_dmag", _DEFAULT_NN_CONTAM_DMAG))
    is_contam = (delta_mag_nn is None) or (
        math.isfinite(float(delta_mag_nn)) and float(delta_mag_nn) <= contam_dmag
    )
    if not neighbour_xys or not is_contam:
        return NeighborSubResult(
            target_flux=plain_flux,
            plain_target_flux=plain_flux,
            neighbor_subtracted=False,
            refused=False,
            refuse_reason="no_contaminant",
            n_neighbors_subtracted=0,
            subtracted_neighbor_flux=0.0,
            joint_fit_chi2=float("nan"),
            residual_rms=float("nan"),
            fit_condition=float("nan"),
            target_x_fit=tx,
            target_y_fit=ty,
        )

    if not bool(getattr(cfg, "psf_neighbor_sub_enabled", False)):
        return NeighborSubResult(
            target_flux=plain_flux,
            plain_target_flux=plain_flux,
            neighbor_subtracted=False,
            refused=False,
            refuse_reason="disabled",
            n_neighbors_subtracted=0,
            subtracted_neighbor_flux=0.0,
            joint_fit_chi2=float("nan"),
            residual_rms=float("nan"),
            fit_condition=float("nan"),
            target_x_fit=tx,
            target_y_fit=ty,
        )

    sep_floor = float(getattr(cfg, "neighbor_sub_refuse_sep_fwhm", 0.8))
    if nn_dist_fwhm is not None and math.isfinite(nn_dist_fwhm) and nn_dist_fwhm <= sep_floor:
        return NeighborSubResult(
            target_flux=plain_flux,
            plain_target_flux=plain_flux,
            neighbor_subtracted=False,
            refused=True,
            refuse_reason="sep_floor",
            n_neighbors_subtracted=0,
            subtracted_neighbor_flux=0.0,
            joint_fit_chi2=float("nan"),
            residual_rms=float("nan"),
            fit_condition=float("nan"),
            target_x_fit=tx,
            target_y_fit=ty,
        )

    regime_dmag = float(getattr(cfg, "neighbor_sub_regime_dmag_min", 2.5))
    regime_sep = float(getattr(cfg, "neighbor_sub_regime_sep_max", 1.1))
    if (
        delta_mag_nn is not None
        and math.isfinite(float(delta_mag_nn))
        and float(delta_mag_nn) <= -regime_dmag
        and nn_dist_fwhm is not None
        and math.isfinite(nn_dist_fwhm)
        and nn_dist_fwhm <= regime_sep
    ):
        return NeighborSubResult(
            target_flux=plain_flux,
            plain_target_flux=plain_flux,
            neighbor_subtracted=False,
            refused=True,
            refuse_reason="bright_close_regime",
            n_neighbors_subtracted=0,
            subtracted_neighbor_flux=0.0,
            joint_fit_chi2=float("nan"),
            residual_rms=float("nan"),
            fit_condition=float("nan"),
            target_x_fit=tx,
            target_y_fit=ty,
        )

    centroid_max = float(getattr(cfg, "neighbor_sub_centroid_max_fwhm", 1.0))
    residual, meta = _joint_moffat_fit_subtract(
        stamp,
        target_xy=(tx, ty),
        neighbour_xys=neighbour_xys,
        fwhm_px=float(fwhm_px),
        fit_beta=float(fit_beta),
        centroid_bound_fwhm=centroid_max,
        r_ap=float(r_ap),
        r_in=float(r_in),
        r_out=float(r_out),
    )

    if not meta.get("ok"):
        return NeighborSubResult(
            target_flux=plain_flux,
            plain_target_flux=plain_flux,
            neighbor_subtracted=False,
            refused=True,
            refuse_reason="fit_failed",
            n_neighbors_subtracted=0,
            subtracted_neighbor_flux=0.0,
            joint_fit_chi2=float("nan"),
            residual_rms=float("nan"),
            fit_condition=float("inf"),
            target_x_fit=tx,
            target_y_fit=ty,
        )

    chi2_max = float(getattr(cfg, "neighbor_sub_chi2_max", 120.0))
    rms_max = float(getattr(cfg, "neighbor_sub_residual_rms_max", 150.0))
    chi2 = float(meta.get("joint_fit_chi2", float("nan")))
    rms = float(meta.get("residual_rms", float("nan")))
    txf = float(meta.get("target_x_fit", tx))
    tyf = float(meta.get("target_y_fit", ty))
    shift = math.hypot(txf - tx, tyf - ty)
    t_amp = float(meta.get("target_amplitude", 0.0))

    clean_flux, _, _ = _annulus_sky_subtracted_flux(residual, tx, ty, r_ap, r_in, r_out)
    clean_flux = float(clean_flux)

    max_nn_over = float(getattr(cfg, "neighbor_sub_max_neighbor_overmag", 0.3))
    max_t_under = float(getattr(cfg, "neighbor_sub_max_target_undermag", 0.5))
    min_snr = float(getattr(cfg, "neighbor_sub_min_recovered_snr", 5.0))
    zp = float(flux_zp)
    neighbor_fit_fluxes: list[float] = list(meta.get("neighbor_fit_fluxes") or [])
    noise_est = _aperture_noise_adu(stamp, r_ap)
    recovered_snr = clean_flux / noise_est if noise_est > 0 else float("nan")

    refuse_reason = ""
    if shift > centroid_max * float(fwhm_px):
        refuse_reason = "centroid_shift"
    if not refuse_reason and t_amp <= 0.0:
        refuse_reason = "target_amp_ill_conditioned"
    if not refuse_reason and clean_flux <= 0.0:
        refuse_reason = "nonphysical_flux"
    if not refuse_reason and math.isfinite(recovered_snr) and recovered_snr < min_snr:
        refuse_reason = "low_recovered_snr"
    if not refuse_reason and nn_mag is not None and math.isfinite(float(nn_mag)) and neighbor_fit_fluxes:
        for nfit in neighbor_fit_fluxes:
            fit_nn_mag = _flux_to_mag_with_zp(nfit, zp)
            if math.isfinite(fit_nn_mag) and fit_nn_mag < float(nn_mag) - max_nn_over:
                refuse_reason = "neighbor_overfit"
                break
    if (
        not refuse_reason
        and target_mag is not None
        and math.isfinite(float(target_mag))
        and clean_flux > 0.0
    ):
        rec_mag = _flux_to_mag_with_zp(clean_flux, zp)
        if math.isfinite(rec_mag) and rec_mag > float(target_mag) + max_t_under:
            refuse_reason = "target_undershoot"
    if (
        not refuse_reason
        and target_mag is not None
        and math.isfinite(float(target_mag))
        and plain_flux > 0
    ):
        expected_flux = 10.0 ** (-0.4 * (float(target_mag) - zp))
        mild_contam = plain_flux < expected_flux * 1.35
        if mild_contam and clean_flux < plain_flux * 0.95:
            refuse_reason = "subtract_harmed"
    if not refuse_reason and is_contam and plain_flux > 0 and clean_flux >= plain_flux * 0.97:
        refuse_reason = "no_improvement"
    if (
        not refuse_reason
        and math.isfinite(chi2)
        and chi2 > chi2_max
        and is_contam
        and clean_flux >= plain_flux * 0.9
    ):
        refuse_reason = "chi2_high"
    if (
        not refuse_reason
        and math.isfinite(rms)
        and rms > rms_max
        and is_contam
        and clean_flux >= plain_flux * 0.9
    ):
        refuse_reason = "residual_rms_high"

    if refuse_reason:
        return NeighborSubResult(
            target_flux=plain_flux,
            plain_target_flux=plain_flux,
            neighbor_subtracted=False,
            refused=True,
            refuse_reason=refuse_reason,
            n_neighbors_subtracted=0,
            subtracted_neighbor_flux=float(meta.get("subtracted_neighbor_flux", 0.0)),
            joint_fit_chi2=chi2,
            residual_rms=rms,
            fit_condition=float(meta.get("fit_condition", float("nan"))),
            target_x_fit=txf,
            target_y_fit=tyf,
        )
    return NeighborSubResult(
        target_flux=float(clean_flux),
        plain_target_flux=plain_flux,
        neighbor_subtracted=True,
        refused=False,
        refuse_reason="",
        n_neighbors_subtracted=int(meta.get("n_neighbors_subtracted", 0)),
        subtracted_neighbor_flux=float(meta.get("subtracted_neighbor_flux", 0.0)),
        joint_fit_chi2=chi2,
        residual_rms=rms,
        fit_condition=float(meta.get("fit_condition", float("nan"))),
        target_x_fit=txf,
        target_y_fit=tyf,
    )
