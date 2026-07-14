"""Pure helpers for WIDE-SLOPE-NOISE report-only analysis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats

from k2_cohort_core import (
    benjamini_hochberg_fdr,
    check_k2_internal_consistency,
    photon_weighted_airmass_slope,
    weighted_linear_regression,
)


def analytic_slope_se(
    airmass: np.ndarray,
    err_mag: np.ndarray,
) -> dict[str, float]:
    """Propagated WLS slope SE from per-epoch err: sqrt(1 / sum(w * (x - xbar_w)^2))."""
    x = np.asarray(airmass, dtype=np.float64)
    e = np.asarray(err_mag, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(e) & (e > 0)
    x, e = x[ok], e[ok]
    n = int(len(x))
    if n < 2:
        nan = float("nan")
        return {
            "se_analytic": nan,
            "sxx_w": nan,
            "sd_x": nan,
            "n": n,
            "median_err": nan,
            "hand_formula_se": nan,
            "xbar_w": nan,
        }
    w = 1.0 / (e * e)
    sw = float(w.sum())
    xbar = float(np.sum(w * x) / sw)
    sxx = float(np.sum(w * (x - xbar) ** 2))
    se = math.sqrt(1.0 / sxx) if sxx > 0 else float("nan")
    sd_x = float(np.std(x, ddof=1))
    med_err = float(np.median(e))
    hand = med_err / (math.sqrt(n) * sd_x) if sd_x > 0 else float("nan")
    return {
        "se_analytic": se,
        "sxx_w": sxx,
        "sd_x": sd_x,
        "n": n,
        "median_err": med_err,
        "hand_formula_se": hand,
        "xbar_w": xbar,
    }


def brightness_tertile_slices(mags: np.ndarray) -> list[tuple[str, float, float]]:
    """Return (label, mag_lo, mag_hi) with lower mag_g = brighter (astronomy convention)."""
    t1, t2 = np.quantile(mags, [1.0 / 3.0, 2.0 / 3.0])
    return [
        ("bright", float("-inf"), float(t1)),
        ("mid", float(t1), float(t2)),
        ("faint", float(t2), float("inf")),
    ]


def brightness_tertile_slices_legacy_inverted(mags: np.ndarray) -> list[tuple[str, float, float]]:
    """Pre-WSN-FIX inverted labels (bright/faint swapped); audit comparison only."""
    t1, t2 = np.quantile(mags, [1.0 / 3.0, 2.0 / 3.0])
    return [
        ("bright", float(t2), float("inf")),
        ("mid", float(t1), float(t2)),
        ("faint", float("-inf"), float(t1)),
    ]


def slope_se_audit_steps(
    mags: np.ndarray,
    airmass: np.ndarray,
    err_mag: np.ndarray,
    *,
    bootstrap_draws: int = 1000,
    seed: int = 0,
    min_airmass_range: float = 0.15,
) -> dict[str, Any]:
    """Step-by-step SE audit for one star (P1 worked example)."""
    fit = photon_weighted_airmass_slope(
        mags, airmass, err_mag, min_airmass_range=min_airmass_range,
    )
    analytic = analytic_slope_se(airmass, err_mag)
    se_b = bootstrap_slope_se(
        mags, airmass, err_mag,
        n_draws=bootstrap_draws, seed=seed, min_airmass_range=min_airmass_range,
    )
    pair = slope_se_pair(
        mags, airmass, err_mag,
        bootstrap_draws=bootstrap_draws, seed=seed, min_airmass_range=min_airmass_range,
    )
    return {
        "N": analytic["n"],
        "SD_X": analytic["sd_x"],
        "median_err_epoch": analytic["median_err"],
        "xbar_w": analytic["xbar_w"],
        "sxx_w": analytic["sxx_w"],
        "se_hand_formula": analytic["hand_formula_se"],
        "se_analytic_propagated": analytic["se_analytic"],
        "se_wls_residual": float(fit.get("b_X_se", float("nan"))),
        "se_bootstrap": se_b,
        "se_use": pair["se_use"],
        "b_X": pair["b_X"],
    }


def bootstrap_slope_se(
    mags: np.ndarray,
    airmass: np.ndarray,
    err_mag: np.ndarray,
    *,
    n_draws: int = 1000,
    seed: int = 0,
    min_airmass_range: float = 0.15,
) -> float:
    """Bootstrap SE of per-star b_X from epoch resampling."""
    m = np.asarray(mags, dtype=np.float64)
    x = np.asarray(airmass, dtype=np.float64)
    e = np.asarray(err_mag, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(x) & np.isfinite(e) & (e > 0)
    m, x, e = m[ok], x[ok], e[ok]
    n = len(m)
    if n < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    for _ in range(int(n_draws)):
        idx = rng.integers(0, n, size=n)
        fit = photon_weighted_airmass_slope(
            m[idx], x[idx], e[idx], min_airmass_range=min_airmass_range,
        )
        bx = float(fit.get("b_X", float("nan")))
        if math.isfinite(bx):
            slopes.append(bx)
    if len(slopes) < 10:
        return float("nan")
    return float(np.std(slopes, ddof=1))


def slope_se_pair(
    mags: np.ndarray,
    airmass: np.ndarray,
    err_mag: np.ndarray,
    *,
    bootstrap_draws: int = 1000,
    seed: int = 0,
    min_airmass_range: float = 0.15,
) -> dict[str, float]:
    """Per-star slope SE from propagated LC err (a) and epoch bootstrap (b)."""
    fit = photon_weighted_airmass_slope(
        mags, airmass, err_mag, min_airmass_range=min_airmass_range,
    )
    analytic = analytic_slope_se(airmass, err_mag)
    se_a = float(analytic["se_analytic"])
    se_wls = float(fit.get("b_X_se", float("nan")))
    se_b = bootstrap_slope_se(
        mags, airmass, err_mag,
        n_draws=bootstrap_draws, seed=seed, min_airmass_range=min_airmass_range,
    )
    se_use = se_a
    if math.isfinite(se_b) and (not math.isfinite(se_a) or se_b > se_a):
        se_use = se_b
    return {
        "b_X": float(fit.get("b_X", float("nan"))),
        "se_propagated": se_a,
        "se_wls_residual": se_wls,
        "se_bootstrap": se_b,
        "se_use": se_use,
        "n_epochs": int(fit.get("n_epochs", 0) or 0),
        "sd_x": float(analytic["sd_x"]),
        "median_err": float(analytic["median_err"]),
        "hand_formula_se": float(analytic["hand_formula_se"]),
    }


def _bootstrap_excess_ci(
    bx: np.ndarray,
    se: np.ndarray,
    *,
    n_draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap 16-84% CI on excess variance SD_obs^2 - median(SE^2)."""
    rng = np.random.default_rng(seed)
    n = len(bx)
    if n < 3:
        nan = float("nan")
        return nan, nan, nan
    obs = float(np.var(bx, ddof=1))
    med_se2 = float(np.median(se * se))
    excess = obs - med_se2
    draws: list[float] = []
    for _ in range(int(n_draws)):
        idx = rng.integers(0, n, size=n)
        b = bx[idx]
        s = se[idx]
        if len(b) < 2:
            continue
        draws.append(float(np.var(b, ddof=1) - np.median(s * s)))
    if not draws:
        return excess, float("nan"), float("nan")
    lo, hi = np.quantile(draws, [0.16, 0.84])
    return excess, float(lo), float(hi)


def excess_variance_by_tertile(
    stars: list[dict[str, Any]],
    *,
    mag_key: str = "mag_g",
    bx_key: str = "b_X",
    se_key: str = "se_use",
    n_bootstrap: int = 2000,
    seed: int = 0,
    legacy_inverted_labels: bool = False,
) -> list[dict[str, Any]]:
    """H0 noise floor: excess variance per brightness tertile."""
    rows: list[dict[str, Any]] = []
    mags = [
        float(s[mag_key]) for s in stars
        if s.get(mag_key) is not None and math.isfinite(float(s[mag_key]))
        and s.get(bx_key) is not None and math.isfinite(float(s[bx_key]))
        and s.get(se_key) is not None and math.isfinite(float(s[se_key])) and float(s[se_key]) > 0
    ]
    if len(mags) < 6:
        return rows
    mag_arr = np.asarray(mags, dtype=np.float64)
    slices_fn = (
        brightness_tertile_slices_legacy_inverted
        if legacy_inverted_labels
        else brightness_tertile_slices
    )
    for label, lo, hi in slices_fn(mag_arr):
        sub = [
            s for s in stars
            if s.get(mag_key) is not None and lo <= float(s[mag_key]) < hi
            and s.get(bx_key) is not None and math.isfinite(float(s[bx_key]))
            and s.get(se_key) is not None and math.isfinite(float(s[se_key])) and float(s[se_key]) > 0
        ]
        if len(sub) < 3:
            continue
        bx = np.asarray([float(s[bx_key]) for s in sub], dtype=np.float64)
        se = np.asarray([float(s[se_key]) for s in sub], dtype=np.float64)
        sd_obs = float(np.std(bx, ddof=1))
        med_se = float(np.median(se))
        med_se2 = float(np.median(se * se))
        excess, ci_lo, ci_hi = _bootstrap_excess_ci(
            bx, se, n_draws=n_bootstrap, seed=seed + hash(label) % 10000,
        )
        noise_dominated = bool(ci_lo <= 0.0 <= ci_hi)
        sub_mags = [float(s[mag_key]) for s in sub]
        rows.append({
            "tertile": label,
            "n": len(sub),
            "mag_lo": lo if math.isfinite(lo) else None,
            "mag_hi": hi if math.isfinite(hi) else None,
            "mag_min": float(np.min(sub_mags)),
            "mag_max": float(np.max(sub_mags)),
            "sd_obs": sd_obs,
            "median_se": med_se,
            "median_se2": med_se2,
            "excess_variance": excess,
            "excess_ci_lo": ci_lo,
            "excess_ci_hi": ci_hi,
            "noise_dominated": noise_dominated,
        })
    return rows


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(y) & np.isfinite(x)
    if int(ok.sum()) < 2:
        return np.full_like(y, np.nan, dtype=np.float64)
    fit = weighted_linear_regression(x[ok], y[ok], np.ones(int(ok.sum())))
    out = np.full_like(y, np.nan, dtype=np.float64)
    out[ok] = y[ok] - fit["intercept"] - fit["slope"] * x[ok]
    return out


def star_drift_metrics(
    x: np.ndarray,
    y: np.ndarray,
    airmass: np.ndarray,
) -> dict[str, float]:
    """Field drift path and its correlation with airmass (H1 inputs)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    am = np.asarray(airmass, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(am)
    if int(ok.sum()) < 3:
        nan = float("nan")
        return {
            "x_med": nan, "y_med": nan, "r2_norm": nan,
            "drift_span_px": nan, "drift_x_corr": nan, "corr_x_am": nan, "corr_y_am": nan,
        }
    x, y, am = x[ok], y[ok], am[ok]
    x_med = float(np.median(x))
    y_med = float(np.median(y))
    r2 = (x - x_med) ** 2 + (y - y_med) ** 2
    r2_norm = float(np.median(r2))
    dx = float(x[-1] - x[0])
    dy = float(y[-1] - y[0])
    drift_span = float(math.hypot(dx, dy))
    pos_scalar = x + y
    drift_x_corr = float(np.corrcoef(pos_scalar, am)[0, 1]) if np.std(pos_scalar) > 0 and np.std(am) > 0 else float("nan")
    corr_x = float(np.corrcoef(x, am)[0, 1]) if np.std(x) > 0 else float("nan")
    corr_y = float(np.corrcoef(y, am)[0, 1]) if np.std(y) > 0 else float("nan")
    return {
        "x_med": x_med,
        "y_med": y_med,
        "r2_norm": r2_norm,
        "drift_span_px": drift_span,
        "drift_x_corr": drift_x_corr,
        "corr_x_am": corr_x,
        "corr_y_am": corr_y,
    }


def apply_affine_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    a: float,
    b: float,
    tx: float,
    c: float,
    d: float,
    ty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply 2D affine x' = a*x + b*y + tx, y' = c*x + d*y + ty."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xp = a * x + b * y + tx
    yp = c * x + d * y + ty
    return xp, yp


def invert_affine_2x3(
    a: float,
    b: float,
    tx: float,
    c: float,
    d: float,
    ty: float,
) -> tuple[float, float, float, float, float, float]:
    """Invert a 2x3 forward affine; returns (a_i, b_i, tx_i, c_i, d_i, ty_i)."""
    det = a * d - b * c
    if abs(det) < 1e-15:
        raise ValueError("singular affine transform")
    a_i = d / det
    b_i = -b / det
    c_i = -c / det
    d_i = a / det
    tx_i = -(a_i * tx + b_i * ty)
    ty_i = -(c_i * tx + d_i * ty)
    return a_i, b_i, tx_i, c_i, d_i, ty_i


def centroid_cutout_detector(
    image: np.ndarray,
    seed_x: float,
    seed_y: float,
    *,
    half: int = 16,
    fwhm: float = 3.0,
) -> tuple[float, float]:
    """Refine star position in detector coordinates via local DAO cutout."""
    from astropy.stats import sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    data = np.asarray(image, dtype=np.float64)
    h, w = data.shape
    ix = int(round(seed_x))
    iy = int(round(seed_y))
    x0 = max(0, ix - half)
    x1 = min(w, ix + half + 1)
    y0 = max(0, iy - half)
    y1 = min(h, iy + half + 1)
    cut = data[y0:y1, x0:x1]
    _, med, std = sigma_clipped_stats(cut, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=max(5.0 * float(std), 1e-6))
    tbl = finder(cut - med)
    if tbl is None or len(tbl) == 0:
        return float(seed_x), float(seed_y)
    xs = np.asarray(tbl["xcentroid"], dtype=np.float64) + x0
    ys = np.asarray(tbl["ycentroid"], dtype=np.float64) + y0
    d2 = (xs - seed_x) ** 2 + (ys - seed_y) ** 2
    j = int(np.argmin(d2))
    return float(xs[j]), float(ys[j])


def track_detector_positions(
    images: list[np.ndarray],
    seed_x: float,
    seed_y: float,
    *,
    half: int = 16,
    fwhm: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Chain cutout centroids across detector-frame images (pre-alignment lights)."""
    xs: list[float] = []
    ys: list[float] = []
    cx, cy = float(seed_x), float(seed_y)
    for img in images:
        cx, cy = centroid_cutout_detector(img, cx, cy, half=half, fwhm=fwhm)
        xs.append(cx)
        ys.append(cy)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def fwhm_sensitivity(
    mags: np.ndarray,
    fwhm: np.ndarray,
    airmass: np.ndarray,
) -> dict[str, float]:
    """Partial d(mag)/d(FWHM) at fixed airmass (H2)."""
    m = np.asarray(mags, dtype=np.float64)
    f = np.asarray(fwhm, dtype=np.float64)
    x = np.asarray(airmass, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(f) & np.isfinite(x) & (f > 0)
    if int(ok.sum()) < 4:
        nan = float("nan")
        return {"fwhm_sens": nan, "aperture_over_fwhm": nan, "corr_fwhm_am": nan}
    m, f, x = m[ok], f[ok], x[ok]
    m_r = _residualize(m, x)
    f_r = _residualize(f, x)
    ok2 = np.isfinite(m_r) & np.isfinite(f_r)
    if int(ok2.sum()) < 3:
        nan = float("nan")
        return {"fwhm_sens": nan, "aperture_over_fwhm": nan, "corr_fwhm_am": nan}
    fit = weighted_linear_regression(f_r[ok2], m_r[ok2], np.ones(int(ok2.sum())))
    corr_fa = float(np.corrcoef(f, x)[0, 1]) if np.std(f) > 0 and np.std(x) > 0 else float("nan")
    return {
        "fwhm_sens": float(fit["slope"]),
        "aperture_over_fwhm": float("nan"),  # filled by caller when aperture available
        "corr_fwhm_am": corr_fa,
    }


def attainable_flat_drift_slope(
    drift_span_px: float,
    drift_x_corr: float,
    *,
    eps_flat: float,
) -> float:
    """Max |b_X| from flat fractional error eps across drift span (H1 effect size)."""
    if not all(math.isfinite(v) for v in (drift_span_px, drift_x_corr, eps_flat)):
        return float("nan")
    return float(abs(eps_flat) * drift_span_px * abs(drift_x_corr))


def attainable_fwhm_slope(
    fwhm_sens: float,
    fwhm_range: float,
    corr_fwhm_am: float,
) -> float:
    """Order-of-magnitude |b_X| from FWHM-airmass coupling (H2 effect size)."""
    if not all(math.isfinite(v) for v in (fwhm_sens, fwhm_range, corr_fwhm_am)):
        return float("nan")
    return float(abs(fwhm_sens) * fwhm_range * abs(corr_fwhm_am))


def physical_effect_size_table(
    stars: list[dict[str, Any]],
    *,
    eps_flat_values: tuple[float, ...] = (0.003, 0.01),
    drift_span_key: str = "det_drift_span_px",
    drift_corr_key: str = "det_drift_x_corr",
) -> dict[str, Any]:
    """P3: pre-registered attainable b_X contributions from measured inputs."""
    spans = [
        float(s[drift_span_key]) for s in stars
        if s.get(drift_span_key) is not None and math.isfinite(float(s[drift_span_key]))
    ]
    if not spans:
        spans = [
            float(s["drift_span_px"]) for s in stars
            if math.isfinite(float(s.get("drift_span_px", float("nan"))))
        ]
        drift_span_key = "drift_span_px"
    corrs = [
        abs(float(s[drift_corr_key])) for s in stars
        if s.get(drift_corr_key) is not None and math.isfinite(float(s[drift_corr_key]))
    ]
    if not corrs:
        corrs = [
            abs(float(s["drift_x_corr"])) for s in stars
            if math.isfinite(float(s.get("drift_x_corr", float("nan"))))
        ]
        drift_corr_key = "drift_x_corr"
    fwhm_sens = [abs(float(s["fwhm_sens"])) for s in stars if math.isfinite(float(s.get("fwhm_sens", float("nan"))))]
    fwhm_rng = [
        float(np.nanmax(s.get("_fwhm_epochs", [])) - np.nanmin(s.get("_fwhm_epochs", [])))
        for s in stars
        if s.get("_fwhm_epochs") is not None and len(s["_fwhm_epochs"]) >= 2
    ]
    if not fwhm_rng:
        fwhm_rng = [
            float(s["fwhm_range"]) for s in stars
            if s.get("fwhm_range") is not None and math.isfinite(float(s["fwhm_range"]))
        ]
    fwhm_rng = [v for v in fwhm_rng if math.isfinite(v)]
    corr_fa = [abs(float(s["corr_fwhm_am"])) for s in stars if math.isfinite(float(s.get("corr_fwhm_am", float("nan"))))]
    med_se = float(np.median([
        float(s["se_use"]) for s in stars
        if s.get("se_use") is not None and math.isfinite(float(s["se_use"])) and float(s["se_use"]) > 0
    ])) if stars else float("nan")

    drift_span_p90 = float(np.percentile(spans, 90)) if spans else float("nan")
    drift_corr_p90 = float(np.percentile(corrs, 90)) if corrs else float("nan")
    fwhm_sens_p90 = float(np.percentile(fwhm_sens, 90)) if fwhm_sens else float("nan")
    fwhm_rng_p90 = float(np.percentile(fwhm_rng, 90)) if fwhm_rng else float("nan")
    corr_fa_p90 = float(np.percentile(corr_fa, 90)) if corr_fa else float("nan")

    flat_rows = []
    for eps in eps_flat_values:
        att = attainable_flat_drift_slope(drift_span_p90, drift_corr_p90, eps_flat=eps)
        flat_rows.append({
            "eps_flat": eps,
            "attainable_bX_p90": att,
            "testable": bool(math.isfinite(att) and math.isfinite(med_se) and att > med_se),
        })

    att_fwhm = attainable_fwhm_slope(fwhm_sens_p90, fwhm_rng_p90, corr_fa_p90)
    return {
        "drift_span_key": drift_span_key,
        "drift_corr_key": drift_corr_key,
        "measurement_floor_median_se": med_se,
        "H1_drift_span_px_p90": drift_span_p90,
        "H1_drift_x_corr_abs_p90": drift_corr_p90,
        "H1_flat_scenarios": flat_rows,
        "H2_fwhm_sens_abs_p90": fwhm_sens_p90,
        "H2_fwhm_range_p90": fwhm_rng_p90,
        "H2_corr_fwhm_am_abs_p90": corr_fa_p90,
        "H2_attainable_bX_p90": att_fwhm,
        "H2_testable": bool(math.isfinite(att_fwhm) and math.isfinite(med_se) and att_fwhm > med_se),
        "H4_colour_bound_mag_airmass": 0.031,
    }


def _wls_multivariate(design: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    """Weighted least squares with intercept in design[:,0]."""
    X = np.asarray(design, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(w) & (w > 0)
    ok &= np.all(np.isfinite(X), axis=1)
    if int(ok.sum()) < X.shape[1] + 1:
        return {"coef": None, "rss": float("nan"), "n": int(ok.sum())}
    X, y, w = X[ok], y[ok], w[ok]
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    yhat = X @ coef
    resid = y - yhat
    rss = float(np.sum(w * resid * resid))
    return {"coef": coef, "rss": rss, "n": int(len(y)), "yhat": yhat, "resid": resid}


def _design_matrix(stars: list[dict[str, Any]], term_groups: dict[str, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Build design matrix from star dicts; drop all-missing columns."""
    all_terms: list[str] = []
    for terms in term_groups.values():
        all_terms.extend(terms)
    present_terms: list[str] = []
    for t in all_terms:
        vals = [
            float(s[t]) for s in stars
            if s.get(t) is not None and math.isfinite(float(s[t]))
        ]
        if vals:
            present_terms.append(t)
    rows = []
    for s in stars:
        row = [1.0]
        for t in present_terms:
            v = s.get(t)
            row.append(float(v) if v is not None and math.isfinite(float(v)) else float("nan"))
        rows.append(row)
    return np.asarray(rows, dtype=np.float64), ["intercept"] + present_terms


def _cross_validated_r2(
    usable: list[dict[str, Any]],
    design: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    k: int = 10,
    seed: int = 0,
) -> float:
    """Out-of-sample R^2 from k-fold CV on the full WLS model."""
    n = len(usable)
    n_params = int(np.asarray(design).shape[1])
    if n < max(15, n_params * 3):
        return float("nan")
    k_fold = min(5, max(2, n // (n_params + 2)))
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k_fold)
    y_true: list[float] = []
    y_pred: list[float] = []
    for i in range(k_fold):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k_fold) if j != i])
        fit = _wls_multivariate(design[train_idx], y[train_idx], w[train_idx])
        if fit["coef"] is None:
            continue
        pred = design[test_idx] @ fit["coef"]
        y_true.extend(y[test_idx].tolist())
        y_pred.extend(pred.tolist())
    if len(y_true) < 5:
        return float("nan")
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def variance_decomposition_regression(
    stars: list[dict[str, Any]],
    *,
    y_key: str = "b_X",
    se_key: str = "se_use",
    term_groups: dict[str, list[str]] | None = None,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """P2 weighted regression: Type-II partial SS shares from the full model."""
    if term_groups is None:
        term_groups = {
            "colour": ["colour_offset_signed"],
            "spatial": ["x_med", "y_med", "r2_norm"],
            "drift_aligned": ["drift_x_corr"],
            "drift_detector": ["det_drift_x_corr", "det_drift_span_px"],
            "fwhm": ["fwhm_sens", "aperture_over_fwhm"],
            "mag": ["mag_g"],
        }

    usable = [
        s for s in stars
        if s.get(y_key) is not None and math.isfinite(float(s[y_key]))
        and s.get(se_key) is not None and math.isfinite(float(s[se_key])) and float(s[se_key]) > 0
    ]
    if len(usable) < 10:
        return {"n": len(usable), "error": "insufficient stars"}

    for s in usable:
        xm = float(s.get("x_med", float("nan")))
        ym = float(s.get("y_med", float("nan")))
        if math.isfinite(xm) and math.isfinite(ym):
            s.setdefault("x_med_sq", xm * xm)
            s.setdefault("y_med_sq", ym * ym)
            s.setdefault("xy_med", xm * ym)

    design_full, names = _design_matrix(usable, term_groups)
    y = np.asarray([float(s[y_key]) for s in usable], dtype=np.float64)
    se = np.asarray([float(s[se_key]) for s in usable], dtype=np.float64)
    w = 1.0 / (se * se)
    ybar = float(np.average(y, weights=w))
    tss = float(np.sum(w * (y - ybar) ** 2))
    if tss <= 0:
        return {"n": len(usable), "error": "zero total sum of squares"}

    fit_full = _wls_multivariate(design_full, y, w)
    if fit_full["coef"] is None:
        return {"n": len(usable), "error": "fit failed"}
    coef_map = {names[i]: float(fit_full["coef"][i]) for i in range(len(names))}
    rss_unscaled = float(fit_full["rss"])
    dof = max(len(usable) - len(names), 1)
    chi2_red = rss_unscaled / dof
    scale = math.sqrt(chi2_red) if chi2_red > 0 else 1.0
    w_scaled = w / (scale * scale)
    fit_scaled = _wls_multivariate(design_full, y, w_scaled)
    if fit_scaled["coef"] is None:
        return {"n": len(usable), "error": "scaled fit failed"}
    coef_scaled_map = {names[i]: float(fit_scaled["coef"][i]) for i in range(len(names))}
    rss_full = float(fit_scaled["rss"])

    all_terms = [t for ts in term_groups.values() for t in ts if t in names]
    group_shares: dict[str, Any] = {}
    p_values: list[float] = []
    p_labels: list[str] = []
    for gname, terms in term_groups.items():
        terms_present = [t for t in terms if t in names]
        if not terms_present:
            group_shares[gname] = {
                "partial_delta_rss": float("nan"),
                "share_of_total_ss": float("nan"),
                "f_p_value": float("nan"),
                "note": "all terms missing",
            }
            p_values.append(float("nan"))
            p_labels.append(gname)
            continue
        keep = [t for t in all_terms if t not in terms_present]
        keep_idx = [0] + [names.index(t) for t in keep if t in names]
        fit_red = _wls_multivariate(design_full[:, keep_idx], y, w_scaled)
        rss_red = float(fit_red["rss"])
        delta = max(0.0, rss_red - rss_full)
        share = delta / tss if tss > 0 else float("nan")
        df_num = len(terms_present)
        df_den = max(len(usable) - len(names), 1)
        if df_num > 0 and rss_full > 0 and delta > 0:
            f_stat = (delta / df_num) / (rss_full / df_den)
            p_val = float(1.0 - stats.f.cdf(f_stat, df_num, df_den))
        else:
            p_val = float("nan")
        group_shares[gname] = {
            "partial_delta_rss": delta,
            "share_of_total_ss": share,
            "f_p_value": p_val,
        }
        p_values.append(p_val)
        p_labels.append(gname)

    fdr = benjamini_hochberg_fdr(p_values, q=0.05)
    for label, adj in zip(p_labels, fdr, strict=True):
        group_shares[label]["q_value"] = adj["q_value"]
        group_shares[label]["reject_fdr"] = adj["reject"]

    co_vals = [
        float(s["colour_offset_signed"]) for s in usable
        if s.get("colour_offset_signed") is not None and math.isfinite(float(s["colour_offset_signed"]))
    ]
    colour_span = float(np.percentile(co_vals, 95) - np.percentile(co_vals, 5)) if len(co_vals) >= 5 else float("nan")
    k2_slope = coef_scaled_map.get("colour_offset_signed", float("nan"))
    colour_slope_spread = (
        abs(k2_slope * colour_span)
        if math.isfinite(k2_slope) and math.isfinite(colour_span) else float("nan")
    )
    h4_consistent = bool(math.isfinite(colour_slope_spread) and colour_slope_spread <= 0.031)

    rng = np.random.default_rng(seed)
    boot_shares: dict[str, list[float]] = {g: [] for g in term_groups}
    n = len(usable)
    for _ in range(int(n_bootstrap)):
        idx_b = rng.integers(0, n, size=n)
        sub = [usable[i] for i in idx_b]
        for s in sub:
            xm = float(s.get("x_med", float("nan")))
            ym = float(s.get("y_med", float("nan")))
            if math.isfinite(xm) and math.isfinite(ym):
                s["x_med_sq"] = xm * xm
                s["y_med_sq"] = ym * ym
                s["xy_med"] = xm * ym
        d_b, n_b = _design_matrix(sub, term_groups)
        y_b = np.asarray([float(s[y_key]) for s in sub], dtype=np.float64)
        w_b = 1.0 / (np.asarray([float(s[se_key]) for s in sub], dtype=np.float64) ** 2)
        ybar_b = float(np.average(y_b, weights=w_b))
        tss_b = float(np.sum(w_b * (y_b - ybar_b) ** 2))
        if tss_b <= 0:
            continue
        fit_f = _wls_multivariate(d_b, y_b, w_b)
        rss_f = float(fit_f["rss"])
        for gname, terms in term_groups.items():
            terms_present = [t for t in terms if t in n_b]
            if not terms_present:
                continue
            keep = [t for t in all_terms if t not in terms_present]
            keep_idx = [0] + [n_b.index(t) for t in keep if t in n_b]
            fit_r = _wls_multivariate(d_b[:, keep_idx], y_b, w_b)
            boot_shares[gname].append(max(0.0, (float(fit_r["rss"]) - rss_f) / tss_b))

    for gname, draws in boot_shares.items():
        if draws:
            lo, hi = np.quantile(draws, [0.025, 0.975])
            group_shares[gname]["share_bootstrap_ci_lo"] = float(lo)
            group_shares[gname]["share_bootstrap_ci_hi"] = float(hi)

    cv_r2 = _cross_validated_r2(usable, design_full, y, w_scaled, k=10, seed=seed + 1)
    cv_group_r2: dict[str, float] = {}
    for gname, terms in term_groups.items():
        keep = [t for t in all_terms if t not in terms]
        keep_idx = [0] + [names.index(t) for t in keep if t in names]
        d_sub = design_full[:, keep_idx]
        cv_group_r2[gname] = _cross_validated_r2(usable, d_sub, y, w_scaled, k=10, seed=seed + 2 + hash(gname) % 1000)

    rho_colour = float("nan")
    pairs = [
        (float(s["colour_offset_signed"]), float(s[y_key]))
        for s in usable
        if s.get("colour_offset_signed") is not None and math.isfinite(float(s["colour_offset_signed"]))
    ]
    if len(pairs) >= 3:
        cx, cy = zip(*pairs, strict=False)
        rho_colour = float(stats.spearmanr(cx, cy).statistic)
    warnings = check_k2_internal_consistency(k2_slope, float(np.median(se)), rho_colour)
    if not h4_consistent:
        warnings.append(
            f"H4_INCONSISTENT: colour slope spread {colour_slope_spread:.4f} mag/airmass "
            "exceeds K2 bound 0.031; do not interpret colour partial SS as physical k''."
        )

    return {
        "n": len(usable),
        "coef": coef_map,
        "coef_overdisp_scaled": coef_scaled_map,
        "chi2_red": chi2_red,
        "overdispersion_scale": scale,
        "rss_full": rss_full,
        "tss": tss,
        "group_shares": group_shares,
        "cv_r2_full": cv_r2,
        "cv_r2_by_group": cv_group_r2,
        "colour_span_p5_p95": colour_span,
        "colour_slope_spread": colour_slope_spread,
        "h4_consistent": h4_consistent,
        "warnings": warnings,
    }


def predicted_group_component(
    coef: dict[str, float],
    star: dict[str, Any],
    term_keys: list[str],
) -> float:
    pred = 0.0
    for key in term_keys:
        c = coef.get(key, float("nan"))
        v = star.get(key)
        if c is not None and v is not None and math.isfinite(float(c)) and math.isfinite(float(v)):
            pred += float(c) * float(v)
    return pred


def rms_predicted_mmag(
    coef: dict[str, float],
    stars: list[dict[str, Any]],
    term_keys: list[str],
) -> float:
    """P4(a): RMS of fitted term-group contribution to b_X (mag/airmass -> mmag scale)."""
    vals = [predicted_group_component(coef, s, term_keys) for s in stars]
    arr = [v for v in vals if math.isfinite(v)]
    if len(arr) < 2:
        return float("nan")
    return float(np.std(arr, ddof=1) * 1000.0)


def p4_noise_consistency_check(
    rms_fitted_mmag: float,
    *,
    sigma_r_ref_mmag: float = 5.5,
    max_ratio: float = 2.0,
) -> dict[str, Any]:
    """P4(a): order-of-magnitude check fitted correlated noise vs PZQ sigma_r."""
    if not math.isfinite(rms_fitted_mmag) or not math.isfinite(sigma_r_ref_mmag) or sigma_r_ref_mmag <= 0:
        return {
            "passed": False,
            "rms_fitted_mmag": rms_fitted_mmag,
            "sigma_r_ref_mmag": sigma_r_ref_mmag,
            "ratio": float("nan"),
            "detail": "non-finite inputs",
        }
    ratio = float(rms_fitted_mmag / sigma_r_ref_mmag)
    passed = ratio <= max_ratio
    detail = "PASS" if passed else f"FAIL: fitted RMS {rms_fitted_mmag:.1f} mmag >> sigma_r {sigma_r_ref_mmag:.1f} mmag"
    return {
        "passed": passed,
        "rms_fitted_mmag": rms_fitted_mmag,
        "sigma_r_ref_mmag": sigma_r_ref_mmag,
        "ratio": ratio,
        "detail": detail,
    }


def univariate_hypothesis_scan(
    stars: list[dict[str, Any]],
    *,
    y_key: str = "b_X",
) -> dict[str, Any]:
    """Exploratory Spearman rho and unweighted R^2 for each hypothesis proxy vs b_X."""
    probes = {
        "colour": "colour_offset_signed",
        "drift_x_corr": "drift_x_corr",
        "fwhm_sens": "fwhm_sens",
        "aperture_over_fwhm": "aperture_over_fwhm",
        "mag_g": "mag_g",
        "r2_norm": "r2_norm",
        "drift_span_px": "drift_span_px",
    }
    ys = [
        float(s[y_key]) for s in stars
        if s.get(y_key) is not None and math.isfinite(float(s[y_key]))
    ]
    out: dict[str, Any] = {"n": len(ys)}
    if len(ys) < 5:
        return out
    y_arr = np.asarray(ys, dtype=np.float64)
    y_var = float(np.var(y_arr, ddof=1))
    for label, key in probes.items():
        pairs = [
            (float(s[key]), float(s[y_key]))
            for s in stars
            if s.get(key) is not None and s.get(y_key) is not None
            and math.isfinite(float(s[key])) and math.isfinite(float(s[y_key]))
        ]
        if len(pairs) < 5:
            out[label] = {"n": len(pairs), "rho": float("nan"), "r2_linear": float("nan")}
            continue
        xs, ys_p = zip(*pairs, strict=False)
        rho = float(stats.spearmanr(xs, ys_p).statistic)
        fit = weighted_linear_regression(
            np.asarray(xs, dtype=np.float64),
            np.asarray(ys_p, dtype=np.float64),
            np.ones(len(xs)),
        )
        yhat = fit["intercept"] + fit["slope"] * np.asarray(xs, dtype=np.float64)
        ss_res = float(np.sum((np.asarray(ys_p) - yhat) ** 2))
        ss_tot = float(np.sum((np.asarray(ys_p) - float(np.mean(ys_p))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out[label] = {"n": len(pairs), "rho": rho, "r2_linear": r2, "share_of_bX_var": r2 if y_var > 0 else float("nan")}
    return out


def pre_registered_outcome(
    tertiles: list[dict[str, Any]],
    decomposition: dict[str, Any],
    *,
    dominant_share_threshold: float = 0.50,
) -> dict[str, Any]:
    """P5 verdict from excess tertiles and regression decomposition."""
    any_excess = any(not t.get("noise_dominated") for t in tertiles)
    if not any_excess:
        return {"verdict": "H0_CLOSED", "detail": "All tertiles noise-dominated; mystery closed."}

    groups = decomposition.get("group_shares") or {}
    eligible = {
        g: info for g, info in groups.items()
        if g != "colour" or decomposition.get("h4_consistent", False)
    }
    dominant = [
        g for g, info in eligible.items()
        if float(info.get("share_of_total_ss", 0) or 0) >= dominant_share_threshold
        and info.get("reject_fdr")
    ]
    if dominant:
        return {
            "verdict": "DOMINANT_SOURCE",
            "groups": dominant,
            "detail": f"Group(s) {dominant} explain >= {dominant_share_threshold:.0%} SS with q<=0.05.",
        }

    if not decomposition.get("h4_consistent", True):
        colour_share = float(groups.get("colour", {}).get("share_of_total_ss", 0) or 0)
        if colour_share >= dominant_share_threshold:
            return {
                "verdict": "EXCESS_UNATTRIBUTED",
                "groups_partial": [g for g, info in groups.items() if float(info.get("share_of_total_ss", 0) or 0) >= 0.10],
                "detail": (
                    "Colour partial SS inflated but H4 slope spread exceeds 0.031 bound; "
                    "no physical dominant source named."
                ),
            }

    return {
        "verdict": "EXCESS_UNATTRIBUTED",
        "groups_partial": [
            g for g, info in groups.items()
            if float(info.get("share_of_total_ss", 0) or 0) >= 0.10
        ],
        "detail": "Excess real but no eligible group >= 50% with FDR significance.",
    }
