"""Per-point error model of the light curves and its systematic floor (sigma_sys).

Production error model (relative-flux domain for LC ``err``, :func:`combine_production_err_rel`):

    err^2 = err_photon^2 + err_ensemble^2 + err_scint^2 + sigma_sys^2

* err_photon: CCD equation of the target (source, sky, read noise; Merline & Howell 1995);
* err_ensemble: error of the Broeg-weighted comparison ensemble on the frame, sqrt(sum wn_j^2 (p_jf^2 + x_j^2))
  over the present comps (``photometry_lightcurve.ensemble_error_per_frame``, Collier Cameron et al. 2006 eq. 4:
  each comp's own per-frame noise model p_jf plus its night excess x_j, ENSEMBLE-ERR-CC06-2026-10);
* err_scint: scintillation at the epoch airmass (Young 1967 / Osborn et al. 2015; telescope diameter, site
  altitude, exposure; 0 when the rig parameters are missing, SCINT-NO-RIG-2026-10);
* sigma_sys: the night floor measured on the comparison stars (SIGMA-SYS-NIGHT / -MODEL), else the per-rig value.

LC columns: err_photon, err_sem_rel, err_scint_rel, err_sigma_sys_rel. See dev/results/specs/VYVAR_SIGMA_FLOOR_SPEC.md.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from mag_constants import MAG_ERR_SCALE

_LOGGED_UNFLOORED: set[str] = set()


def mag_sigma_to_rel(sigma_mag: float) -> float:
    if not math.isfinite(sigma_mag) or sigma_mag <= 0:
        return 0.0
    return float(sigma_mag) / MAG_ERR_SCALE


def rel_sigma_to_mag(sigma_rel: float) -> float:
    if not math.isfinite(sigma_rel) or sigma_rel <= 0:
        return 0.0
    return float(sigma_rel) * MAG_ERR_SCALE


#: G08-32: the terms combine_production_err_rel adds (pipeline_meta dynamic_params.sigma_floor)
ERR_MODEL_TERMS = ("photon", "ensemble", "scintillation", "systematic_floor")
ERR_MODEL_TEXT = ("err^2 = err_photon^2 + err_ensemble^2 + err_scint^2 + sigma_sys^2 (relative flux; "
                  "err_ensemble = sqrt(sum wn_j^2 (p_jf^2 + x_j^2)) of the present comparison stars, "
                  "Collier Cameron et al. 2006 eq. 4)")


def combine_production_err_rel(
    err_photon_rel: float,
    sem_mag: float,
    *,
    sigma_sys_mag: float = 0.0,
    sigma_scint_mag: float = 0.0,
) -> float:
    """Quadrature combine photon, ensemble SEM, scintillation, and per-rig floor (rel-flux domain)."""
    terms = 0.0
    if math.isfinite(err_photon_rel) and err_photon_rel > 0:
        terms += err_photon_rel * err_photon_rel
    sem_rel = mag_sigma_to_rel(sem_mag)
    if sem_rel > 0:
        terms += sem_rel * sem_rel
    scint_rel = mag_sigma_to_rel(sigma_scint_mag)
    if scint_rel > 0:
        terms += scint_rel * scint_rel
    sys_rel = mag_sigma_to_rel(sigma_sys_mag)
    if sys_rel > 0:
        terms += sys_rel * sys_rel
    if terms <= 0:
        return float("nan")
    return math.sqrt(terms)


def combine_production_err_mag(
    err_photon_rel: float,
    sem_mag: float,
    *,
    sigma_sys_mag: float = 0.0,
    sigma_scint_mag: float = 0.0,
) -> float:
    """Magnitude-domain sigma from production composition."""
    rel = combine_production_err_rel(
        err_photon_rel,
        sem_mag,
        sigma_sys_mag=sigma_sys_mag,
        sigma_scint_mag=sigma_scint_mag,
    )
    return rel_sigma_to_mag(rel)


def scintillation_mag_per_epoch(
    *,
    telescope_diameter_m: float,
    airmass: float,
    exposure_s: float,
    altitude_m: float,
    c_y: float = 1.5,
) -> float:
    """Per-epoch scintillation sigma in magnitudes (Young/Osborn via sigma_budget)."""
    from sigma_budget import relative_flux_err_to_mag_sigma, scintillation_sigma  # noqa: PLC0415

    rel = scintillation_sigma(
        telescope_diameter_m=telescope_diameter_m,
        airmass=airmass,
        exposure_s=exposure_s,
        altitude_m=altitude_m,
        c_y=c_y,
    )
    if not math.isfinite(rel) or rel <= 0:
        return 0.0
    return float(relative_flux_err_to_mag_sigma(rel))


def frame_scintillation_mag(rig: Any, airmass: float) -> float:
    """Scintillation [mag] of one frame at its airmass with the run's rig parameters (``rig``:
    ``sigma_budget.RigScintillationParams``); 0 where the airmass is unknown or below 1. The one per-frame
    value of the light-curve error, the night floor (``aperture_photometry._frame_scintillation_mag``) and the
    comparison ceiling (``comp_rms_loo.scint_mag_err_map``, COMP-CEIL-NOISEMODEL-2026-10)."""
    am = float(airmass)
    if not (math.isfinite(am) and am >= 1.0):
        return 0.0
    return scintillation_mag_per_epoch(
        telescope_diameter_m=rig.telescope_diameter_m, airmass=am, exposure_s=rig.exposure_s,
        altitude_m=rig.altitude_m, c_y=rig.c_y,
    )


def resolve_sigma_sys_mag(
    equipment_id: int | None,
    cfg: Any | None,
    *,
    rig_label: str = "",
) -> float:
    """Per-camera white floor in mag (EQUIPMENTS.SIGMA_SYS_MAG); explicit 0.0 with a log line when unset.

    WIDE-ERR-03 S3: an unset floor must not be silent; do not invent a constant.
    """
    from rig_photometry import rig_photometry  # noqa: PLC0415

    key = "unknown" if equipment_id is None else str(int(equipment_id))
    ssm = rig_photometry(cfg, equipment_id).sigma_sys_mag
    present = ssm is not None
    val = float(ssm) if present else 0.0
    log_key = f"{key}:{'set' if present else 'default0'}"
    if log_key not in _LOGGED_UNFLOORED:
        _LOGGED_UNFLOORED.add(log_key)
        if present:
            logging.info(
                "[SIGMA-FLOOR] equipment_id=%s (%s): sigma_sys_mag=%.6g mag (EQUIPMENTS.SIGMA_SYS_MAG)",
                key,
                rig_label or "rig",
                val,
            )
        else:
            logging.info(
                "[SIGMA-FLOOR] equipment_id=%s (%s): EQUIPMENTS.SIGMA_SYS_MAG unset - "
                "explicit default 0.0 mag (WIDE-ERR-03; residual floor owned by err calibration)",
                key,
                rig_label or "rig",
            )
    return float(val)


#: G08-03: scintillation share of the predicted comp noise above which the floor is not separable from the
#: scintillation model (C_Y quartile spread ~ +-30 %, Osborn et al. 2015 Table 1) - a WARNING, not a gate
SCINT_SHARE_WARN = 0.5


def _robust_pull_scale(z: Any) -> float:
    """1.4826 MAD of the pooled pulls (no clipping)."""
    import numpy as np

    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if z.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(z - np.median(z))))


def night_sigma_sys_from_comps(
    flux_by_cid: dict[str, Any],
    sky_by_cid: dict[str, Any],
    fwhm_by_frame: Any,
    *,
    k: float,
    gain: float,
    read_noise: float,
    n_sky_by_cid: dict[str, Any],
    max_photon_mag: float = 0.01,
    min_comps: int = 5,
    scint_mag_by_frame: Any = None,
) -> dict[str, Any]:
    """SIGMA-SYS-NIGHT-2026-10 (Milan 2026-10-12): the night's systematic floor measured on the
    comparison stars instead of a fixed config constant.

    Each comp (normalised to its own median) is compared with the Broeg-weighted mean of the OTHER comps
    present on the frame (leave-one-out; weights ``ensemble_weights.broeg_weights``, the ensemble's
    weights): r_jf. SIGMA-SYS-MODEL-2026-10 (D3): the predicted variance holds every known term once - the
    comp's photon + sky + read noise (``ccd_noise.ccd_variance_adu2`` with its real number of sky pixels per
    frame, ``n_sky_by_cid``; gain and read noise required, ValueError), its scintillation at the frame airmass
    (``scint_mag_by_frame``, the value the LC error adds) and the noise of its reference,
    sum_k u_jkf^2 (photon_k^2 + scint^2), u = the normalised weights of the others (the same deconvolution as
    the ensemble error, ENSEMBLE-ERR-CC06-2026-10). SIGMA-SYS-TOTAL-2026-10 (O1): the residual is the total
    (not point-to-point) one, so the floor holds the night's slow systematics.

    SIGMA-SYS-POOL-2026-10 (G08-03): the floor f is the ONE value for all qualifying comps (own median photon
    error below ``max_photon_mag``) for which the pooled robust pull over all (comp, frame) is 1:
    1.4826 MAD(r_jf / sqrt(pred_jf + f^2 (1 + sum_k u_jkf^2))) = 1 (robust analogue of Collier Cameron et al.
    2006 eq. 5; the reference carries the others' floor too, hence 1 + sum u^2 = 1 + 1/n for equal weights;
    no clipping). f = 0 when the pull is <= 1 without a floor. Returns {"sigma_sys_mag", "n_comps",
    "pull_comps" (pooled pull at f), "scint_share" (median scint^2 / predicted variance without the floor),
    "warning" (scint_share > ``SCINT_SHARE_WARN``: floor not separable from the scintillation model), ...};
    sigma_sys_mag is NaN when fewer than ``min_comps`` comps qualify.
    """
    import numpy as np
    from scipy.optimize import brentq  # noqa: PLC0415

    from ccd_noise import ccd_variance_adu2, check_gain_read_noise  # noqa: PLC0415
    from ensemble_weights import broeg_weights  # noqa: PLC0415

    check_gain_read_noise(gain, read_noise)
    ids = [c for c in flux_by_cid if np.isfinite(np.asarray(flux_by_cid[c], float)).sum() >= 10]
    out: dict[str, Any] = {"sigma_sys_mag": float("nan"), "n_comps": 0,
                           "method": "comps LOO (Broeg-weighted reference) total scatter: pooled robust pull = 1 "
                                     "over photon + scintillation + reference + floor"}
    if len(ids) < int(min_comps):
        out["reason"] = f"only {len(ids)} comps with >= 10 frames"
        return out
    F = np.vstack([np.asarray(flux_by_cid[c], float) for c in ids])
    S = np.vstack([np.asarray(sky_by_cid.get(c, np.full(F.shape[1], np.nan)), float) for c in ids])
    NS = np.vstack([np.asarray(n_sky_by_cid.get(c, np.full(F.shape[1], np.nan)), float) for c in ids])
    fw = np.asarray(fwhm_by_frame, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = -2.5 * np.log10(np.where(F > 0, F, np.nan))
        M = M - np.nanmedian(M, axis=1, keepdims=True)
        r_ap = float(k) * fw
        area = math.pi * r_ap**2
        var = ccd_variance_adu2(F, S, area, NS, gain=gain, read_noise=read_noise)
        E2 = (1.0857362 * np.sqrt(var) / F) ** 2  # photon + sky + read noise per comp and frame [mag^2]
        E2 = np.where(np.isfinite(M) & (F > 0), E2, np.nan)
    sc = np.zeros(F.shape[1]) if scint_mag_by_frame is None else np.nan_to_num(np.asarray(scint_mag_by_frame, float))
    sc2 = sc * sc
    fin = np.isfinite(M)
    w, _ = broeg_weights(M)
    wf = w[:, None] * fin
    tot_w = wf.sum(axis=0)
    tot_wm = (wf * np.where(fin, M, 0.0)).sum(axis=0)
    own_var = np.where(fin, np.nan_to_num(E2) + sc2[None, :], 0.0)  # every comp of the frame: photon + scint
    tot_w2v = (wf * wf * own_var).sum(axis=0)
    tot_w2 = (wf * wf).sum(axis=0)
    R, P0, U2, SC, ph = [], [], [], [], []
    for i in range(len(ids)):
        ow = tot_w - wf[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            ref = np.where(ow > 0, (tot_wm - wf[i] * np.nan_to_num(M[i])) / ow, np.nan)
            u2 = np.where(ow > 0, (tot_w2 - wf[i] ** 2) / (ow * ow), np.nan)  # sum_k u_ik^2
            ref_var = np.where(ow > 0, (tot_w2v - wf[i] ** 2 * own_var[i]) / (ow * ow), np.nan)
        r = M[i] - ref
        pred = E2[i] + sc2 + ref_var
        ok = np.isfinite(r) & np.isfinite(pred) & np.isfinite(u2)
        if int(ok.sum()) < 10:
            continue
        R.append(r[ok]); P0.append(pred[ok]); U2.append(u2[ok]); SC.append(sc2[ok])
        ph.append(float(np.sqrt(np.median(E2[i][ok]))))
    ph_a = np.asarray(ph)
    sel = np.flatnonzero(ph_a < float(max_photon_mag))
    out["n_comps_total"] = int(len(ph_a))
    if int(sel.size) < int(min_comps):
        out["reason"] = f"only {int(sel.size)} comps with photon error < {max_photon_mag} mag"
        return out
    r_all = np.concatenate([R[i] for i in sel])
    p_all = np.concatenate([P0[i] for i in sel])
    u_all = np.concatenate([U2[i] for i in sel])
    s_all = np.concatenate([SC[i] for i in sel])

    def _pull(f: float) -> float:
        return _robust_pull_scale(r_all / np.sqrt(p_all + f * f * (1.0 + u_all)))

    pull0 = _pull(0.0)
    f = 0.0
    if pull0 > 1.0:
        hi = _robust_pull_scale(r_all) + 1e-6  # the floor alone at the total scatter: pull < 1
        while _pull(hi) >= 1.0:
            hi *= 2.0
        f = float(brentq(lambda x: _pull(x) - 1.0, 0.0, hi, xtol=1e-7))
    scint_share = float(np.median(s_all / p_all)) if s_all.size else float("nan")
    out.update(sigma_sys_mag=f, n_comps=int(sel.size), median_photon_mag=float(np.median(ph_a[sel])),
               pull_comps=round(_pull(f), 4), pull_comps_no_floor=round(pull0, 4),
               scint_share=round(scint_share, 4), warning="")
    if math.isfinite(scint_share) and scint_share > SCINT_SHARE_WARN:
        out["warning"] = (f"scintillation is {100 * scint_share:.0f} % of the predicted comp noise: floor not "
                          "separable from the scintillation model (C_Y uncertain)")
        logging.warning("[SIGMA-SYS] %s", out["warning"])
    return out
