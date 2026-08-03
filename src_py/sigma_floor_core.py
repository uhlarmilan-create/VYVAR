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
    """
    if n < 2:
        return float("nan")
    if n == 2:
        return math.sqrt(2.0) * math.gamma(1.0) / math.gamma(0.5)
    half = 0.5 * float(n)
    half_m1 = 0.5 * float(n - 1)
    return math.sqrt(2.0 / float(n - 1)) * math.gamma(half) / math.gamma(half_m1)


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
    """Per-rig white floor in mag; 0.0 fail-safe when unknown (one-time INFO log)."""
    if equipment_id is None:
        key = "unknown"
        val = 0.0
    else:
        key = str(int(equipment_id))
        raw_map = getattr(cfg, "sigma_sys_mag", None) if cfg is not None else None
        val = 0.0
        if isinstance(raw_map, dict):
            try:
                v = float(raw_map.get(key, 0.0))
                if math.isfinite(v) and v > 0:
                    val = v
            except (TypeError, ValueError):
                val = 0.0
    if val <= 0 and key not in _LOGGED_UNFLOORED:
        _LOGGED_UNFLOORED.add(key)
        logging.info(
            "[SIGMA-FLOOR] equipment_id=%s (%s): no sigma_sys_mag configured - floor=0",
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
