"""Broeg, Fernandez & Neuhaeuser (2005) ensemble weights and the leave-one-out comp scatter.

The one implementation (G08-36) used by the light-curve ensemble (``photometry_lightcurve``) and by the night
floor (``sigma_floor_core``, SIGMA-SYS-POOL-2026-10); a low module (numpy only) so neither imports the other.
"""
from __future__ import annotations

import numpy as np

BROEG_TOL = 1e-5
BROEG_MAX_ITER = 200


def _loo_sigma(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Leave-one-out scatter of each comparison star: sample std (ddof=1, >= 3 frames) of comp j
    against the ``w``-weighted ensemble of the OTHER comps (only comp j when it is the only one).
    ``y`` is (n_comp, n_frames) with NaN = missing. G08-36: the one LOO sigma - used inside the Broeg
    iteration and, with the final weights, for the ensemble inverse variances."""
    y = np.asarray(y, dtype=np.float64)
    n = int(y.shape[0])
    fin = np.isfinite(y)
    y0 = np.where(fin, y, 0.0)
    wf = np.asarray(w, dtype=np.float64)[:, None] * fin
    tot_w = wf.sum(axis=0)
    tot_wy = (wf * y0).sum(axis=0)
    sig = np.full(n, np.nan)
    for j in range(n):
        if n >= 2:
            ow = tot_w - wf[j]
            ens = np.where(ow > 0, (tot_wy - wf[j] * y0[j]) / np.where(ow > 0, ow, 1.0), np.nan)
            r = y[j] - ens
        else:
            r = y[j]
        v = r[np.isfinite(r)]
        sig[j] = float(np.std(v, ddof=1)) if v.size >= 3 else np.nan
    return sig


def broeg_weights(
    y: np.ndarray,
    *,
    tol: float = BROEG_TOL,
    max_iter: int = BROEG_MAX_ITER,
) -> tuple[np.ndarray, int]:
    """Broeg, Fernandez & Neuhaeuser (2005) weights for comparison stars.

    ``y`` is (n_comp, n_frames) of normalised instrumental magnitudes (each comp minus its
    own median; NaN = missing). Each weight is ``1/sigma_j^2`` where ``sigma_j`` is the
    standard deviation of comp j against the weighted ensemble of the OTHER comps
    (leave-one-out); iterated until the largest relative weight change is < ``tol``.
    No clipping: a variable comparison star simply ends with a small weight.
    Returns (weights normalised to sum 1, n_iterations).
    """
    y = np.asarray(y, dtype=np.float64)
    n = int(y.shape[0])
    if n == 0:
        return np.zeros(0), 0
    if n == 1:
        return np.ones(1), 0
    fin = np.isfinite(y)

    def _sig(r: np.ndarray, m: np.ndarray) -> float:
        v = r[m]
        return float(np.std(v, ddof=1)) if v.size >= 3 else float("nan")

    w = np.array([_sig(y[j], fin[j]) for j in range(n)], dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), 1.0 / (w * w), np.nan)
    if not np.isfinite(w).any():
        return np.full(n, 1.0 / n), 0
    w = np.where(np.isfinite(w), w, np.nanmin(w))
    it = 0
    for it in range(1, int(max_iter) + 1):
        sj = _loo_sigma(y, w)
        new = np.where(np.isfinite(sj) & (sj > 0), 1.0 / (sj * sj), np.nan)
        if not np.isfinite(new).any():
            break
        new = np.where(np.isfinite(new), new, np.nanmin(new))
        rel = np.max(np.abs(new - w) / np.maximum(w, 1e-300))
        w = new
        if rel < float(tol):
            break
    return w / float(np.sum(w)), int(it)
