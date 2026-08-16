"""Per-rig systematic floor (sigma_sys) and small-sample SEM correction (c4).

Production error model (relative-flux domain for LC ``err``):

    err_total^2 = err_photon_bkg^2 + sem_ens_rel^2 + sigma_sys_rel^2

See dev/results/specs/VYVAR_SIGMA_FLOOR_SPEC.md.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from mag_constants import MAG_ERR_SCALE

_LOGGED_UNFLOORED: set[str] = set()


def c4_small_sample(n: int) -> float:
    """Unbiased std scale factor c4(n) = E[s_ddof1] / sigma for normal data.

    c4(n) = sqrt(2/(n-1)) * Gamma(n/2) / Gamma((n-1)/2).

    Requires n >= 2. Unit-tested against literature constants to 1e-4.

    For large n, ``math.gamma`` overflows (OverflowError: math range error on
    Windows when Gamma(x) exceeds ~1e308, x>~171). Use ``math.lgamma`` ratio so
    COMP-ADMIT-03 full membership (n_comps >> 200) does not crash Phase 2A.
    """
    if n < 2:
        return float("nan")
    if n == 2:
        return math.sqrt(2.0) * math.gamma(1.0) / math.gamma(0.5)
    half = 0.5 * float(n)
    half_m1 = 0.5 * float(n - 1)
    # Prefer lgamma for all n>=3: numerically stable and overflow-safe.
    try:
        log_ratio = math.lgamma(half) - math.lgamma(half_m1)
        return math.sqrt(2.0 / float(n - 1)) * math.exp(log_ratio)
    except (OverflowError, ValueError):
        # Asymptotic: c4 -> 1 as n -> inf (correction ~ 1/(4n)).
        return float(1.0 - 1.0 / (4.0 * float(n)))


def ensemble_sem_mag_from_residuals(residuals: list[float] | Any) -> float:
    """Honeycutt ensemble SEM (mag) with c4 small-sample correction."""
    if len(residuals) < 2:
        return 0.0
    arr = [float(x) for x in residuals if math.isfinite(float(x))]
    n = len(arr)
    if n < 2:
        return 0.0
    c4 = c4_small_sample(n)
    if not math.isfinite(c4) or c4 <= 0:
        return float("nan")
    std_ddof1 = float(math.sqrt(sum((x - sum(arr) / n) ** 2 for x in arr) / (n - 1)))
    return std_ddof1 / c4 / math.sqrt(n)


def ensemble_sem_mag_from_residuals_weighted(
    residuals: list[float] | Any,
    weights: list[float] | Any,
) -> float:
    """Reliability-weighted ensemble SEM (mag); reduces to unweighted when w equal.

    WIDE-ERR-03 / SEM-WEIGHT-01: same residuals as production
    ``(m_inst - median_night)``, weights ``w=1/sigma_eff^2`` matching mag_calib ZP.

        mu = sum(w x)/sum(w)
        V1 = sum(w), V2 = sum(w^2), N_eff = V1^2/V2
        s_w^2 = sum(w (x-mu)^2)/(V1 - V2/V1)
        SEM = s_w / c4(round(N_eff)) / sqrt(N_eff)
    """
    xs: list[float] = []
    ws: list[float] = []
    for x, w in zip(residuals, weights, strict=False):
        xf = float(x)
        wf = float(w)
        if math.isfinite(xf) and math.isfinite(wf) and wf > 0:
            xs.append(xf)
            ws.append(wf)
    n = len(xs)
    if n < 2:
        return 0.0
    # Equal weights -> exact unweighted path
    if max(ws) - min(ws) <= 1e-15 * max(ws):
        return ensemble_sem_mag_from_residuals(xs)
    v1 = sum(ws)
    v2 = sum(w * w for w in ws)
    if v1 <= 0 or v2 <= 0:
        return float("nan")
    n_eff = (v1 * v1) / v2
    mu = sum(w * x for w, x in zip(ws, xs, strict=True)) / v1
    denom = v1 - (v2 / v1)
    if denom <= 0:
        return float("nan")
    s2 = sum(w * (x - mu) ** 2 for w, x in zip(ws, xs, strict=True)) / denom
    if s2 < 0:
        return float("nan")
    s = math.sqrt(s2)
    n_eff_r = max(2, int(round(n_eff)))
    c4 = c4_small_sample(n_eff_r)
    if not math.isfinite(c4) or c4 <= 0:
        return float("nan")
    return s / c4 / math.sqrt(n_eff)


def mag_sigma_to_rel(sigma_mag: float) -> float:
    if not math.isfinite(sigma_mag) or sigma_mag <= 0:
        return 0.0
    return float(sigma_mag) / MAG_ERR_SCALE


def rel_sigma_to_mag(sigma_rel: float) -> float:
    if not math.isfinite(sigma_rel) or sigma_rel <= 0:
        return 0.0
    return float(sigma_rel) * MAG_ERR_SCALE


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


def resolve_sigma_sys_mag(
    equipment_id: int | None,
    cfg: Any | None,
    *,
    rig_label: str = "",
) -> float:
    """Per-rig white floor in mag; explicit 0.0 with a log line when unset.

    WIDE-ERR-03 S3: missing map keys (e.g. equipment 1 while only ``\"4\"`` is set)
    must not be silent. Interim for wide-rig equipment 1 is sys=0 WITH this log;
    Stage 5 calibration owns any residual floor (do not invent a constant).
    """
    if equipment_id is None:
        key = "unknown"
        val = 0.0
        present = False
    else:
        key = str(int(equipment_id))
        raw_map = getattr(cfg, "sigma_sys_mag", None) if cfg is not None else None
        val = 0.0
        present = False
        if isinstance(raw_map, dict) and key in raw_map:
            present = True
            try:
                v = float(raw_map.get(key))
                if math.isfinite(v) and v >= 0:
                    val = v
            except (TypeError, ValueError):
                val = 0.0
    log_key = f"{key}:{'set' if present else 'default0'}"
    if log_key not in _LOGGED_UNFLOORED:
        _LOGGED_UNFLOORED.add(log_key)
        if present:
            logging.info(
                "[SIGMA-FLOOR] equipment_id=%s (%s): sigma_sys_mag=%.6g mag (config map)",
                key,
                rig_label or "rig",
                val,
            )
        else:
            logging.info(
                "[SIGMA-FLOOR] equipment_id=%s (%s): sigma_sys_mag unset in config map - "
                "explicit default 0.0 mag (WIDE-ERR-03; residual floor owned by err calibration)",
                key,
                rig_label or "rig",
            )
    return float(val)


def pzq_fit_sigma_r(
    mags: Any,
    *,
    bin_sizes: tuple[int, ...] = (2, 4, 8),
) -> dict[str, Any]:
    """PZQ (2006) binned RMS diagnostic: sigma_N^2 = sigma_w^2/N + sigma_r^2."""
    import numpy as np

    m = np.asarray(mags, dtype=np.float64)
    ok = np.isfinite(m)
    m = m[ok]
    n = int(m.size)
    if n < max(bin_sizes) * 2:
        return {"n_epochs": n, "sigma_w": float("nan"), "sigma_r": float("nan"), "bins": []}
    ref = float(np.median(m))
    resid = m - ref
    sigma_1 = float(np.std(resid, ddof=1))
    bins_out: list[dict[str, Any]] = []
    xs: list[float] = []
    ys: list[float] = []
    for nb in bin_sizes:
        n_bin = n // nb
        if n_bin < 2:
            continue
        trimmed = resid[: n_bin * nb].reshape(n_bin, nb)
        bin_means = np.mean(trimmed, axis=1)
        sigma_n = float(np.std(bin_means, ddof=1))
        white_expect = sigma_1 / math.sqrt(nb) if sigma_1 > 0 else float("nan")
        bins_out.append(
            {
                "N": int(nb),
                "sigma_N": sigma_n,
                "sigma_white_expect": white_expect,
                "n_bins": int(n_bin),
            }
        )
        xs.append(1.0 / float(nb))
        ys.append(sigma_n * sigma_n)
    if len(xs) < 2:
        return {
            "n_epochs": n,
            "sigma_1": sigma_1,
            "sigma_w": float("nan"),
            "sigma_r": float("nan"),
            "bins": bins_out,
        }
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    A = np.column_stack([x_arr, np.ones(len(x_arr))])
    beta, *_ = np.linalg.lstsq(A, y_arr, rcond=None)
    sigma_w_sq = max(0.0, float(beta[0]))
    sigma_r_sq = max(0.0, float(beta[1]))
    return {
        "n_epochs": n,
        "sigma_1": sigma_1,
        "sigma_w": math.sqrt(sigma_w_sq),
        "sigma_r": math.sqrt(sigma_r_sq),
        "bins": bins_out,
    }
