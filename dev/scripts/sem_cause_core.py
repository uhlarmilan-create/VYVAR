"""Pure helpers for SIGMA-SEM-CAUSE diagnostics (sandbox, no production imports)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_MAG_SCALE = 2.5 / math.log(10.0)


def lag1_autocorrelation(series: np.ndarray) -> float:
    """Lag-1 autocorrelation; NaN if undefined."""
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return float("nan")
    x0 = x[:-1] - float(np.mean(x[:-1]))
    x1 = x[1:] - float(np.mean(x[1:]))
    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(x1 * x1)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x0 * x1) / denom)


def trend_fraction(
    y: np.ndarray,
    x: np.ndarray,
    *,
    deg: int = 1,
) -> tuple[float, float, np.ndarray]:
    """Return (var_trend/var_total, r_squared, fitted_trend).

    Uses polyfit on finite pairs; var_total is variance of y (ddof=1).
    """
    yv = np.asarray(y, dtype=np.float64)
    xv = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(yv) & np.isfinite(xv)
    yf = yv[ok]
    xf = xv[ok]
    if yf.size < max(3, deg + 1):
        return float("nan"), float("nan"), np.full_like(yv, np.nan)
    coef = np.polyfit(xf, yf, deg)
    trend = np.polyval(coef, xv)
    trend_f = trend[ok]
    resid = yf - trend_f
    var_total = float(np.var(yf, ddof=1)) if yf.size >= 2 else float("nan")
    var_resid = float(np.var(resid, ddof=1)) if resid.size >= 2 else float("nan")
    if not math.isfinite(var_total) or var_total <= 0:
        frac = float("nan")
    else:
        frac = max(0.0, min(1.0, 1.0 - var_resid / var_total))
    ss_tot = float(np.sum((yf - float(np.mean(yf))) ** 2))
    ss_res = float(np.sum(resid**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return frac, r2, trend


def per_frame_sem_from_residuals(
    residuals_by_frame: list[list[float]],
) -> np.ndarray:
    """Production SEM: std(comp_resid, ddof=1) / sqrt(n) per frame."""
    out = np.full(len(residuals_by_frame), np.nan, dtype=np.float64)
    for i, res in enumerate(residuals_by_frame):
        arr = np.asarray([float(r) for r in res if math.isfinite(float(r))], dtype=np.float64)
        if arr.size >= 2:
            out[i] = float(np.std(arr, ddof=1) / math.sqrt(arr.size))
        elif arr.size == 1:
            out[i] = 0.0
    return out


def flux_sum_ensemble_mag(mags: np.ndarray) -> float:
    """AIJ/Honeycutt flux-sum ensemble (-2.5 log10 sum 10^-0.4 m)."""
    m = np.asarray(mags, dtype=np.float64)
    m = m[np.isfinite(m)]
    if m.size == 0:
        return float("nan")
    fluxes = 10.0 ** (-0.4 * m)
    s = float(np.sum(fluxes))
    if s > 0:
        return float(-2.5 * math.log10(s))
    return float(np.median(m))


def split_half_zp_sem(
    comp_mags_frame: dict[str, float],
    *,
    n_splits: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    """Per-frame split-half ZP noise estimate and scaling factor used.

    Returns (empirical_sem_mag, sqrt_n_half_over_n_full) where empirical_sem_mag is
    the median over random splits of std(zp_half_diff) * sqrt(n / n_half), with
    zp_half_diff = (ens_A - ens_B) / 2 and ens from flux-sum ensemble per half.

    Split-half convention: each half has n_half = floor(n/2) comps; independent
    half-ensemble means differ with variance ~ 2*(sigma^2/n_half); for difference
    D = mean_A - mean_B, Var(D) ~ 2*sigma^2/n_half, so sigma/sqrt(n) ~ |D|/sqrt(2).
    Using (ens_A-ens_B)/2 gives SEM ~ |zp_half_diff| when n_half = n/2; we take the
    median |zp_half_diff| over splits and scale by sqrt(n / n_half) to express as
    full-n ensemble SEM equivalent (Honeycutt std/sqrt(n) units).
    """
    ids = [c for c, v in comp_mags_frame.items() if math.isfinite(v)]
    n = len(ids)
    if n < 4:
        return float("nan"), float("nan")
    n_half = max(2, n // 2)
    scale = math.sqrt(n / n_half)
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    mags = {c: comp_mags_frame[c] for c in ids}
    for _ in range(max(1, n_splits)):
        perm = rng.permutation(ids)
        ha = perm[:n_half]
        hb = perm[n_half : n_half + n_half]
        if len(hb) < 2:
            continue
        ma = np.asarray([mags[c] for c in ha], dtype=float)
        mb = np.asarray([mags[c] for c in hb], dtype=float)
        ea = flux_sum_ensemble_mag(ma)
        eb = flux_sum_ensemble_mag(mb)
        if math.isfinite(ea) and math.isfinite(eb):
            diffs.append(abs(ea - eb) / 2.0)
    if not diffs:
        return float("nan"), scale
    return float(np.median(diffs) * scale), scale


def recompose_err_mag(
    err_phot_mag: np.ndarray,
    sem_mag: np.ndarray,
) -> np.ndarray:
    """Quadrature recombination in magnitude domain (offline diagnostic)."""
    p = np.asarray(err_phot_mag, dtype=np.float64)
    s = np.asarray(sem_mag, dtype=np.float64)
    n = max(p.size, s.size)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        pv = float(p[i]) if i < p.size else float("nan")
        sv = float(s[i]) if i < s.size else float("nan")
        if not math.isfinite(pv) or pv < 0:
            continue
        sv_eff = sv if math.isfinite(sv) and sv > 0 else 0.0
        out[i] = math.sqrt(pv * pv + sv_eff * sv_eff)
    return out


def chi2_dof_from_mags_sigmas(mags: np.ndarray, sig_mag: np.ndarray) -> tuple[float, int, float]:
    """Reduced chi2/dof for constant source."""
    m = np.asarray(mags, dtype=np.float64)
    s = np.asarray(sig_mag, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(s) & (s > 0)
    m = m[ok]
    s = s[ok]
    n = int(m.size)
    if n < 3:
        return float("nan"), max(0, n - 1), float("nan")
    ref = float(np.mean(m))
    chi2 = float(np.sum(((m - ref) / s) ** 2))
    dof = n - 1
    return chi2, dof, chi2 / dof if dof > 0 else float("nan")


def distribution_stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "median": None, "p25": None, "p75": None, "mean": None, "values": []}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
        "mean": float(np.mean(arr)),
        "values": arr.tolist(),
    }


def rel_to_mag_err(err_rel: np.ndarray) -> np.ndarray:
    return _MAG_SCALE * np.asarray(err_rel, dtype=np.float64)


def mag_to_rel_err(err_mag: np.ndarray) -> np.ndarray:
    return np.asarray(err_mag, dtype=np.float64) / _MAG_SCALE
