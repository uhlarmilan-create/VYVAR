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


def _robust_poly(x: np.ndarray, y: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Soft-L1 polynomial fit (order incl. constant). Returns (coeffs low->high, residuals)."""
    from scipy.optimize import least_squares  # noqa: PLC0415

    A = np.vander(x, order + 1, increasing=True)
    p0, *_ = np.linalg.lstsq(A, y, rcond=None)
    r0 = y - A @ p0
    scale = 1.4826 * float(np.median(np.abs(r0 - np.median(r0))))
    if not (scale > 0):
        return p0, r0
    res = least_squares(lambda p: A @ p - y, p0, loss="soft_l1", f_scale=scale)
    p = res.x
    return p, y - A @ p


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
    fits: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for order in range(0, max_order + 1):
        if n < MIN_STARS_PER_COEF * (order + 1):
            break
        p, r = _robust_poly(x, d, order)
        s = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        if not (s > 0):
            continue
        # Gaussian likelihood with the robust scale; k = order + 1 parameters
        bic[order] = float(n * math.log(s * s) + (order + 1) * math.log(n))
        fits[order] = (p, r)
    if 0 not in bic:
        return None
    best = 0
    for order in sorted(bic):
        if order > best and bic[order] < bic[best] - BIC_MIN_GAIN:
            best = order
    p, r = fits[best]
    return ColourTermFit(
        order=int(best),
        coeffs=[float(v) for v in p[1:]],
        c_ref=c_ref,
        zero_point=float(p[0]),
        rms=float(1.4826 * np.median(np.abs(r - np.median(r)))),
        n=n,
        c_min=float(np.min(c)),
        c_max=float(np.max(c)),
        bic=bic,
    )
