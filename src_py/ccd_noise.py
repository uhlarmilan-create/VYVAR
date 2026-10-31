"""The CCD equation: variance of a sky-subtracted aperture flux (CCD-ONE-2026-10, G06-20 / G08-28).

One implementation for every stage (per-frame proc error fallback, hybrid background scaling, light-curve
error at the light-curve aperture, night systematic floor, field variability search, transient search,
k* S/N). Leaf module: imports only numpy / math.

Merline & Howell (1995, Exp. Astron. 6, 163), eq. 1; Howell (2006, Handbook of CCD Astronomy, sec. 4.4):

    var = F/g + A (1 + c A / n_sky) (S/g + (RN/g)^2)          [ADU^2]

F: sky-subtracted source flux [ADU] (source photons only, so F < 0 counts as 0); S: sky per pixel [ADU];
A: aperture area [px]; n_sky: number of sky pixels the sky estimate used (the real count, after the
source mask and the footprint cut); g: gain [e-/ADU]; RN: read noise [e-]. The term A^2/n_sky (times the
per-pixel variance) is the error of the sky estimate, which is subtracted A times.

c is the variance of the sky *estimator* relative to the mean of n_sky pixels. VYVAR's sky is a plain
median (``sky_estimation.annulus_sky_medians``); the asymptotic variance of the sample median of n
normal deviates is (pi/2) sigma^2 / n (Kendall & Stuart 1977, The Advanced Theory of Statistics, vol. 1,
4th ed., ch. 10: var(median) = 1 / (4 n f(m)^2), with f(m) = 1 / (sigma sqrt(2 pi)) for a normal
distribution), so c = pi/2 for a median and 1 for a mean. The sky of a CCD pixel at the levels VYVAR
measures (tens to thousands of e-) is Poisson + Gaussian read noise, i.e. normal to good approximation.

Not included, because the call sites do not know them: dark-current shot noise of dark-subtracted
pixels and the ADC quantisation term g^2 sigma_f^2 of Merline & Howell (both below 1 % of the per-pixel
variance for cooled cameras at the read noise / gain of VYVAR rigs).

No defaults for gain, read noise, area or n_sky: a missing gain / read noise is an error (ValueError);
a non-finite or non-positive area / n_sky (a star without a sky estimate) gives NaN for that value.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

#: variance of the sky estimator relative to the mean of n_sky pixels (see module docstring)
SKY_ESTIMATOR_VARIANCE_FACTOR: dict[str, float] = {"mean": 1.0, "median": math.pi / 2.0}


def check_gain_read_noise(gain: Any, read_noise: Any) -> tuple[float, float]:
    try:
        g = float(gain)
        rn = float(read_noise)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CCD equation: gain / read noise not numeric (gain={gain!r}, read_noise={read_noise!r})") from exc
    if not (math.isfinite(g) and g > 0):
        raise ValueError(f"CCD equation: gain must be finite and > 0 e-/ADU (got {gain!r})")
    if not (math.isfinite(rn) and rn >= 0):
        raise ValueError(f"CCD equation: read noise must be finite and >= 0 e- (got {read_noise!r})")
    return g, rn


def _out(v: np.ndarray, scalar: bool) -> Any:
    return float(v) if scalar else v


def ccd_sky_variance_adu2(
    sky_pp: Any,
    area: Any,
    n_sky: Any,
    *,
    gain: float,
    read_noise: float,
    sky_estimator: str = "median",
) -> Any:
    """Sky + read-noise part of the CCD equation [ADU^2]: A (1 + c A / n_sky) (S/g + (RN/g)^2).

    Scalars in -> float out; arrays in -> ndarray out (element-wise, broadcast). Negative sky counts as 0
    (no sky photons). NaN where area is not finite and > 0, n_sky is not > 0 (NaN), or sky is NaN.
    """
    g, rn = check_gain_read_noise(gain, read_noise)
    try:
        c = SKY_ESTIMATOR_VARIANCE_FACTOR[str(sky_estimator)]
    except KeyError as exc:
        raise ValueError(f"CCD equation: unknown sky estimator {sky_estimator!r}") from exc
    scalar = all(np.ndim(v) == 0 for v in (sky_pp, area, n_sky))
    S = np.asarray(sky_pp, dtype=np.float64)
    A = np.asarray(area, dtype=np.float64)
    N = np.asarray(n_sky, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        A_ok = np.where(np.isfinite(A) & (A > 0), A, np.nan)
        N_ok = np.where(N > 0, N, np.nan)  # inf: a sky known exactly (no sky-estimation term)
        var = A_ok * (1.0 + c * A_ok / N_ok) * (np.maximum(S, 0.0) / g + (rn / g) ** 2)
    return _out(var, scalar)


def ccd_variance_adu2(
    flux: Any,
    sky_pp: Any,
    area: Any,
    n_sky: Any,
    *,
    gain: float,
    read_noise: float,
    sky_estimator: str = "median",
) -> Any:
    """Total variance of a sky-subtracted aperture flux [ADU^2] (Merline & Howell 1995, see module doc).

    Scalars in -> float out; arrays in -> ndarray out. A flux <= 0 adds no source photon noise (the
    variance is then that of the sky and read noise alone); NaN flux -> NaN.
    """
    bkg = ccd_sky_variance_adu2(sky_pp, area, n_sky, gain=gain, read_noise=read_noise, sky_estimator=sky_estimator)
    scalar = np.ndim(bkg) == 0 and np.ndim(flux) == 0
    F = np.asarray(flux, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        var = np.maximum(F, 0.0) / float(gain) + np.asarray(bkg, dtype=np.float64)
    return _out(var, scalar)


def ccd_rel_error(
    flux: Any,
    sky_pp: Any,
    area: Any,
    n_sky: Any,
    *,
    gain: float,
    read_noise: float,
    sky_estimator: str = "median",
) -> Any:
    """sqrt(ccd_variance_adu2) / F (relative flux error); NaN where F <= 0 or not finite."""
    var = ccd_variance_adu2(flux, sky_pp, area, n_sky, gain=gain, read_noise=read_noise, sky_estimator=sky_estimator)
    scalar = np.ndim(var) == 0
    F = np.asarray(flux, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(np.isfinite(F) & (F > 0), np.sqrt(np.asarray(var, dtype=np.float64)) / F, np.nan)
    return _out(e, scalar)
