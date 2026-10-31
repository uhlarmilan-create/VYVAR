"""Per-point error model of the light curves and its systematic floor (sigma_sys).

Production error model (relative-flux domain for LC ``err``, :func:`combine_production_err_rel`):

    err^2 = err_photon^2 + err_ensemble^2 + err_scint^2 + sigma_sys^2

* err_photon: CCD equation of the target (source, sky, read noise; Merline & Howell 1995);
* err_ensemble: error of the weighted comparison ensemble on the frame, sqrt(1 / sum 1/sigma_j^2) over the
  present comps (``photometry_lightcurve.ensemble_normalize``, Broeg weights) - no c4 / small-sample factor;
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
                  "err_ensemble = sqrt(1/sum 1/sigma_j^2) of the present comparison stars)")


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

    For each comp: leave-one-out differential light curve against the equal-weight ensemble of
    the other comps (each normalised to its median), its robust scatter over the night
    s = 1.4826 MAD(r - median r), and its predicted photon + sky + read noise e (the same CCD
    equation as the LC ``err``). SIGMA-SYS-TOTAL-2026-10 (Milan 2026-10-28, phase-1 item O1 option C):
    s is the total scatter, not the point-to-point scatter, so the floor holds the night's slow
    systematics (Brno 534 check stars: p2p / err 0.85-1.0 but total / err 1.3-1.9) and the errors
    describe the repeatability over the night, as the check-star scatter does (AAVSO practice). The floor is
    sqrt(median(s^2 - e^2)) over comps whose own photon error is below ``max_photon_mag``
    (bright comps, where a systematic term is measurable), never negative.
    Everett & Howell (2001): 2-5 mmag typical. CCD-ONE-2026-10: the photon error is
    ``ccd_noise.ccd_variance_adu2`` with each comp's real number of sky pixels per frame (``n_sky_by_cid``);
    gain and read noise are required (ValueError). Returns {"sigma_sys_mag", "n_comps", ...};
    sigma_sys_mag is NaN when fewer than ``min_comps`` comps qualify.

    SIGMA-SYS-MODEL-2026-10 (Milan 2026-10-26, D3): the predicted variance e^2 holds every known
    term once - the comp's photon + sky + read noise, its scintillation (``scint_mag_by_frame``,
    the same per-epoch value the LC error adds) and the noise of its reference, the mean of the
    other n comps: (sum over the others of photon^2 + scintillation^2) / n^2. The floor is what
    is left above that model, so the LC error, which adds scintillation and the ensemble error
    itself, does not count them twice (Burdanov et al. 2014; Everett & Howell 2001).
    """
    import numpy as np

    from ccd_noise import ccd_variance_adu2, check_gain_read_noise  # noqa: PLC0415

    check_gain_read_noise(gain, read_noise)
    ids = [c for c in flux_by_cid if np.isfinite(np.asarray(flux_by_cid[c], float)).sum() >= 10]
    out: dict[str, Any] = {"sigma_sys_mag": float("nan"), "n_comps": 0,
                           "method": "comps LOO total scatter - (photon + scintillation + reference)"}
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
        tot = np.nansum(M, axis=0)
        cnt = np.isfinite(M).sum(axis=0)
        r_ap = float(k) * fw
        area = math.pi * r_ap**2
        var = ccd_variance_adu2(F, S, area, NS, gain=gain, read_noise=read_noise)
        E2 = (1.0857362 * np.sqrt(var) / F) ** 2  # photon + sky + read noise per comp and frame [mag^2]
        E2 = np.where(np.isfinite(M) & (F > 0), E2, np.nan)
    sc = np.zeros(F.shape[1]) if scint_mag_by_frame is None else np.nan_to_num(np.asarray(scint_mag_by_frame, float))
    sc2 = sc * sc
    tot_var = np.nansum(E2 + sc2, axis=0)  # every comp of the frame: photon + scintillation
    excess, ph = [], []
    for i in range(len(ids)):
        m = M[i]
        n_oth = np.maximum(cnt - np.isfinite(m), 1)
        others = (tot - np.nan_to_num(m)) / n_oth
        r = m - others
        ok = np.isfinite(r)
        if int(ok.sum()) < 10:
            continue
        rr = r[ok]
        s = 1.4826 * float(np.median(np.abs(rr - np.median(rr))))
        own = np.nan_to_num(E2[i])
        ref_var = (tot_var - own - sc2) / (n_oth * n_oth)
        pred = E2[i] + sc2 + ref_var
        e_med = float(np.sqrt(np.nanmedian(E2[i][ok]))) if np.isfinite(E2[i][ok]).any() else float("nan")
        pred_med = float(np.nanmedian(pred[ok])) if np.isfinite(pred[ok]).any() else float("nan")
        if math.isfinite(s) and math.isfinite(e_med) and math.isfinite(pred_med):
            excess.append(s * s - pred_med)
            ph.append(e_med)
    ph_a, ex_a = np.asarray(ph), np.asarray(excess)
    sel = ph_a < float(max_photon_mag)
    out["n_comps_total"] = int(len(ph_a))
    if int(sel.sum()) < int(min_comps):
        out["reason"] = f"only {int(sel.sum())} comps with photon error < {max_photon_mag} mag"
        return out
    med = float(np.median(ex_a[sel]))
    out.update(sigma_sys_mag=math.sqrt(med) if med > 0 else 0.0, n_comps=int(sel.sum()),
               median_photon_mag=float(np.median(ph_a[sel])))
    return out
