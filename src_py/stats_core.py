"""Shared scalar stats helpers (flux-to-mag, MAD variants live at call sites).

Bodies here are the bit-identical survivors of CONSOLIDATE-01C C-2 merges.
Do not add a second MAD flavour under the same name.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _flux_to_mag(flux: float) -> float:
    """Instrumental magnitude from flux; non-finite or non-positive flux returns nan."""
    if not math.isfinite(flux) or flux <= 0:
        return float("nan")
    return -2.5 * math.log10(flux)


def flux_to_mag(flux: Any) -> np.ndarray:
    """Vectorised :func:`_flux_to_mag` (float64): -2.5 log10(flux), NaN where the flux is not finite or
    <= 0 (G06-21: the one array conversion)."""
    f = np.asarray(flux, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


# G02-39 / G09-25 (step 5): the ONE boolean vocabulary for CSV / JSON / config flags. Numbers (and numeric
# strings such as "1.0" / "0.0", which a bool column with blanks becomes after a CSV round trip) are True when
# != 0; words below; None, NaN, "" and anything else are unknown (``parse_bool`` -> None).
_TRUE_WORDS = frozenset({"true", "t", "yes", "y", "on"})
_FALSE_WORDS = frozenset({"false", "f", "no", "n", "off"})


def parse_bool(raw: Any) -> bool | None:
    """Tri-state flag parser: True / False, or None when the value is unknown (None, NaN, "", unrecognised)."""
    if isinstance(raw, (bool, np.bool_)):
        return bool(raw)
    if raw is None:
        return None
    if isinstance(raw, (int, float, np.integer, np.floating)):
        v = float(raw)
        return None if math.isnan(v) else v != 0.0
    t = str(raw).strip().lower()
    if t in _TRUE_WORDS:
        return True
    if t in _FALSE_WORDS:
        return False
    try:
        v = float(t)
    except ValueError:
        return None
    return None if math.isnan(v) else v != 0.0


def coerce_bool(raw: Any, default: bool = False) -> bool:
    """Two-state flag: :func:`parse_bool`, unknown -> ``default``."""
    b = parse_bool(raw)
    return bool(default) if b is None else b


def coerce_bool_series(series: Any, default: bool = False) -> Any:
    """Vector form of :func:`coerce_bool` on a pandas Series: a bool Series on the same index.

    Replaces ``.fillna(x).astype(bool)`` (reads the text "False" as True) and the per-module
    ``astype(str).isin(...)`` vocabularies. Identical to ``fillna(default).astype(bool)`` on bool / numeric /
    bool-with-NaN columns."""
    import pandas as pd  # noqa: PLC0415

    s = series if isinstance(series, pd.Series) else pd.Series(series)
    if pd.api.types.is_bool_dtype(s.dtype):
        return s.fillna(bool(default)).astype(bool) if s.hasnans else s.astype(bool)
    if pd.api.types.is_numeric_dtype(s.dtype):
        v = s.astype("float64")
        return (v != 0.0).where(v.notna(), bool(default)).astype(bool)
    return s.map(lambda v: coerce_bool(v, default)).astype(bool)


def robust_p2p_scatter(values: Any, *, min_points: int) -> float:
    """Robust point-to-point scatter of one series in time order: 1.4826 MAD(diff) / sqrt(2), no clipping.

    The von Neumann (1941) / Abbe successive-difference estimator of the white-noise sigma (insensitive to slow
    trends), with the standard deviation of the differences replaced by its MAD estimate (Rousseeuw & Croux 1993:
    breakdown 50 %), so one outlier frame does not inflate it (G08-55, COMP-P2P-ROBUST-2026-10). Non-finite
    values are dropped before differencing; NaN with fewer than ``min_points`` finite values.
    """
    m = np.asarray(values, dtype=np.float64)
    m = m[np.isfinite(m)]
    if m.size < int(min_points) or m.size < 2:
        return float("nan")
    d = np.diff(m)
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0))


#: NOISE-FIT-ONE-2026-10: start point and bounds of ``fit_scatter_noise_model`` (photon scale a, floor [mag]);
#: the values the field variability search has used since its noise model was introduced
NOISE_FIT_X0: tuple[float, float] = (1.2, 0.005)
NOISE_FIT_BOUNDS: tuple[tuple[float, float], tuple[float, float]] = ((0.2, 0.0), (10.0, 0.2))


def fit_scatter_noise_model(s: Any, e: Any, *, extra: Any = None) -> tuple[float, float, bool]:
    """The one field noise-model fit (Sokolovsky et al. 2017 style): ``s = sqrt((a e)^2 [+ extra^2] + floor^2)``.

    ``s``: each star's measured scatter [mag]; ``e``: its predicted (photon / CCD-equation) error [mag];
    ``extra``: an optional fixed, unscaled per-star term [mag] (scintillation for the comparison ceiling).
    Robust least squares on the log scatter (scipy ``least_squares``, soft_l1 loss, f_scale 0.1 in ln s), so a
    few variables or broken stars do not pull the fit; no clipping. Returns ``(a, floor, success)``.
    Shared by ``variability_field`` (no ``extra``, bit-identical to its former inline fit) and the comparison
    ceiling (``comp_rms_loo.fit_comp_ceiling_model``, COMP-CEIL-NOISEMODEL-2026-10).
    """
    from scipy import optimize  # noqa: PLC0415

    s = np.asarray(s, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    if extra is None:
        def _res(p: np.ndarray) -> np.ndarray:
            a, fl = p
            return np.log(s) - 0.5 * np.log((a * e) ** 2 + fl**2)
    else:
        x2 = np.asarray(extra, dtype=np.float64) ** 2

        def _res(p: np.ndarray) -> np.ndarray:
            a, fl = p
            return np.log(s) - 0.5 * np.log((a * e) ** 2 + x2 + fl**2)

    fit = optimize.least_squares(
        _res, list(NOISE_FIT_X0), loss="soft_l1", f_scale=0.1,
        bounds=(list(NOISE_FIT_BOUNDS[0]), list(NOISE_FIT_BOUNDS[1])),
    )
    return float(fit.x[0]), float(fit.x[1]), bool(fit.success)
