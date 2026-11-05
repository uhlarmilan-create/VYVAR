# -*- coding: ascii -*-
"""Colour term of the instrumental band (CT-2026-10).

Standard transformation (Landolt / Stetson style) for differential photometry with
comparison stars whose catalogue magnitudes are in the observed band (Gaia DR3 ->
g/r/i/V/B/R, :mod:`gaia_bands`):

    m_band - m_inst = Z + P(C),   C = Gaia BP-RP,   P(C_ref) = 0

P is a polynomial whose order (1..3) is chosen from the data by BIC, with at least
``MIN_STARS_PER_COEF`` stars per coefficient (the same rule as the SIP order in plate
solving). The fit is robust (soft-L1 loss, scale = 1.4826 MAD of the residuals) -
variable or blended stars lose weight but no star is rejected (no clipping).

For a target measured against an ensemble with zero-point weights w_j the colour
correction is

    dm = P(C_target) - sum_j w_j P(C_j) / sum_j w_j

(for a non-linear P this is not P(C_target) - P(weighted mean colour)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from stats_core import MAD_TO_SIGMA

MAX_ORDER = 3
MIN_STARS_PER_COEF = 10
#: An order is accepted only when it lowers BIC by more than this (Kass & Raftery 1995:
#: Delta BIC > 10 = very strong evidence).
BIC_MIN_GAIN = 10.0


@dataclass
class ColourTermFit:
    order: int
    coeffs: list[float]          # P(C) = sum_k coeffs[k] * (C - c_ref)^(k+1), k = 0..order-1
    c_ref: float
    zero_point: float
    rms: float
    n: int
    c_min: float
    c_max: float
    bic: dict[int, float] = field(default_factory=dict)
    #: G08-46: covariance of [zero_point, coeffs...] from the robust fit (least_squares Jacobian, robust scale)
    cov: list[list[float]] | None = None

    def poly(self, c: np.ndarray | float) -> np.ndarray | float:
        x = np.asarray(c, dtype=np.float64) - self.c_ref
        out = np.zeros_like(x)
        for k, a in enumerate(self.coeffs):
            out = out + a * x ** (k + 1)
        return out if out.ndim else float(out)

    def correction(self, c_target: float, c_comps: list[float], weights: list[float]) -> float:
        """P(C_target) - weighted mean of P(C_comp) (ensemble zero-point weights)."""
        cc = np.asarray(c_comps, dtype=np.float64)
        ww = np.asarray(weights, dtype=np.float64)
        ok = np.isfinite(cc) & np.isfinite(ww) & (ww > 0)
        if not (math.isfinite(float(c_target)) and ok.any()):
            return float("nan")
        return float(self.poly(float(c_target))) - float(np.sum(ww[ok] * self.poly(cc[ok])) / np.sum(ww[ok]))

    def slope(self, c: float) -> float:
        """P'(C) = sum_k (k+1) coeffs[k] (C - c_ref)^k."""
        x = float(c) - self.c_ref
        return float(sum((k + 1) * a * x ** k for k, a in enumerate(self.coeffs)))

    def coeff_stderr(self) -> list[float]:
        """Standard errors of ``coeffs`` (NaN without a covariance)."""
        if self.cov is None:
            return [float("nan")] * len(self.coeffs)
        d = np.diag(np.asarray(self.cov, dtype=np.float64))[1:]
        return [float(math.sqrt(v)) if v >= 0 else float("nan") for v in d]

    def correction_sigma(self, c_target: float, c_comps: list[float], weights: list[float]) -> float:
        """G08-46: uncertainty of ``correction`` from the coefficient covariance, sqrt(g^T Cov g) with
        g_k = x_t^k - sum_j wn_j x_j^k (x = C - c_ref, k = 1..order; the zero point cancels). The same
        fitted coefficients serve every epoch, so this is a zero-point term of the light curve, not a
        per-point error. NaN without a covariance or outside the inputs of ``correction``."""
        if self.cov is None or not self.coeffs:
            return float("nan")
        cc = np.asarray(c_comps, dtype=np.float64)
        ww = np.asarray(weights, dtype=np.float64)
        ok = np.isfinite(cc) & np.isfinite(ww) & (ww > 0)
        if not (math.isfinite(float(c_target)) and ok.any()):
            return float("nan")
        wn = ww[ok] / np.sum(ww[ok])
        xt = float(c_target) - self.c_ref
        xj = cc[ok] - self.c_ref
        g = np.array([xt ** k - float(np.sum(wn * xj ** k)) for k in range(1, len(self.coeffs) + 1)])
        cov_a = np.asarray(self.cov, dtype=np.float64)[1:, 1:]
        return float(math.sqrt(max(float(g @ cov_a @ g), 0.0)))


def _robust_poly(x: np.ndarray, y: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft-L1 polynomial fit (order incl. constant). Returns (coeffs low->high, residuals, covariance).

    G08-46: covariance of the M-estimator (Huber 1981, sandwich form) at the solution,
    Cov = B^-1 M B^-1 n / (n - p), B = A^T diag(psi'(r)) A, M = A^T diag(psi(r)^2) A, with the soft-L1
    influence psi(r) = r / sqrt(1 + (r/f)^2) at the fit scale f (1.4826 MAD of the start residuals).
    The loss-scaled least_squares Jacobian with the robust scale, s^2 (J^T J)^-1, overestimates the
    bootstrap scatter of the coefficients by 25-45 % on Gaussian data (soft-L1 down-weights 1-sigma
    residuals); the sandwich agrees with the bootstrap within 5 % (dev/tests/test_step7_comp_2026_10.py)."""
    from scipy.optimize import least_squares  # noqa: PLC0415

    A = np.vander(x, order + 1, increasing=True)
    p0, *_ = np.linalg.lstsq(A, y, rcond=None)
    r0 = y - A @ p0
    scale = MAD_TO_SIGMA * float(np.median(np.abs(r0 - np.median(r0))))
    if not (scale > 0):
        return p0, r0, np.zeros((order + 1, order + 1))
    res = least_squares(lambda p: A @ p - y, p0, loss="soft_l1", f_scale=scale)
    p = res.x
    r = y - A @ p
    z = (r / scale) ** 2
    psi = r / np.sqrt(1.0 + z)
    dpsi = (1.0 + z) ** -1.5
    b_inv = np.linalg.pinv(A.T @ (dpsi[:, None] * A))
    m = A.T @ ((psi * psi)[:, None] * A)
    n, k = A.shape
    cov = b_inv @ m @ b_inv * (float(n) / float(n - k)) if n > k else np.full((k, k), np.nan)
    return p, r, cov


def fit_colour_term(colour: np.ndarray, delta: np.ndarray, *, max_order: int = MAX_ORDER) -> ColourTermFit | None:
    """Fit delta = m_band_cat - m_inst against colour; order by BIC (0 = no colour term)."""
    c = np.asarray(colour, dtype=np.float64)
    d = np.asarray(delta, dtype=np.float64)
    ok = np.isfinite(c) & np.isfinite(d)
    c, d = c[ok], d[ok]
    n = int(c.size)
    if n < 2 * MIN_STARS_PER_COEF:
        return None
    c_ref = float(np.median(c))
    x = c - c_ref
    bic: dict[int, float] = {}
    fits: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for order in range(0, max_order + 1):
        if n < MIN_STARS_PER_COEF * (order + 1):
            break
        p, r, cv = _robust_poly(x, d, order)
        s = MAD_TO_SIGMA * float(np.median(np.abs(r - np.median(r))))
        if not (s > 0):
            continue
        # Gaussian likelihood with the robust scale; k = order + 1 parameters
        bic[order] = float(n * math.log(s * s) + (order + 1) * math.log(n))
        fits[order] = (p, r, cv)
    if 0 not in bic:
        return None
    best = 0
    for order in sorted(bic):
        if order > best and bic[order] < bic[best] - BIC_MIN_GAIN:
            best = order
    p, r, cv = fits[best]
    return ColourTermFit(
        order=int(best),
        coeffs=[float(v) for v in p[1:]],
        c_ref=c_ref,
        zero_point=float(p[0]),
        rms=float(MAD_TO_SIGMA * np.median(np.abs(r - np.median(r)))),
        n=n,
        c_min=float(np.min(c)),
        c_max=float(np.max(c)),
        bic=bic,
        cov=[[float(v) for v in row] for row in np.asarray(cv, dtype=np.float64)],
    )
