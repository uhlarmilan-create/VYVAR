"""Pure helpers for K2-COHORT report-only analysis (FDR, slopes, power)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats


def benjamini_hochberg_fdr(p_values: list[float], *, q: float = 0.05) -> list[dict[str, float | bool]]:
    """Benjamini-Hochberg FDR control across a family of p-values.

    Returns list aligned to input with keys p, q_value, reject (bool).
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if math.isfinite(float(p))]
    if not indexed:
        return [{"p": float("nan"), "q_value": float("nan"), "reject": False} for _ in p_values]
    indexed.sort(key=lambda t: t[1])
    k = len(indexed)
    q_vals: dict[int, float] = {}
    prev_q = 1.0
    for rank in range(k, 0, -1):
        i, p = indexed[rank - 1]
        raw = p * m / rank
        adj = min(prev_q, raw)
        prev_q = adj
        q_vals[i] = adj
    out: list[dict[str, float | bool]] = []
    for i, p in enumerate(p_values):
        pf = float(p)
        if not math.isfinite(pf):
            out.append({"p": pf, "q_value": float("nan"), "reject": False})
            continue
        qv = float(q_vals.get(i, 1.0))
        out.append({"p": pf, "q_value": qv, "reject": bool(qv <= q)})
    return out


def spearman_min_n_for_power(
    *,
    rho_alt: float = 0.4,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Approximate minimum n for Spearman rank correlation power (two-sided).

    Uses Fisher z transform approximation for rank correlation (standard practice).
    """
    if rho_alt <= 0 or rho_alt >= 1:
        return 9999
    z_rho = 0.5 * math.log((1.0 + rho_alt) / (1.0 - rho_alt))
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / z_rho) ** 2 + 3.0
    return int(math.ceil(n))


def spearman_power_at_n(
    n: int,
    *,
    rho_alt: float = 0.4,
    alpha: float = 0.05,
) -> float:
    """Approximate two-sided Spearman power at given n."""
    if n < 3 or rho_alt <= 0:
        return 0.0
    z_rho = 0.5 * math.log((1.0 + rho_alt) / (1.0 - rho_alt))
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    se = math.sqrt(1.0 / (n - 3.0)) if n > 3 else float("inf")
    if not math.isfinite(se) or se <= 0:
        return 0.0
    z_effect = z_rho / se
    return float(1.0 - stats.norm.cdf(z_alpha - z_effect) + stats.norm.cdf(-z_alpha - z_effect))


def weighted_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> dict[str, float]:
    """Weighted least-squares y = intercept + slope * x."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if int(ok.sum()) < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "slope_se": float("nan"),
            "intercept_se": float("nan"),
            "n": int(ok.sum()),
        }
    x, y, w = x[ok], y[ok], w[ok]
    sw = float(w.sum())
    xbar = float(np.sum(w * x) / sw)
    ybar = float(np.sum(w * y) / sw)
    sxx = float(np.sum(w * (x - xbar) ** 2))
    if sxx <= 0:
        return {
            "slope": float("nan"),
            "intercept": float(ybar),
            "slope_se": float("nan"),
            "intercept_se": float("nan"),
            "n": int(len(x)),
        }
    sxy = float(np.sum(w * (x - xbar) * (y - ybar)))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    resid = y - intercept - slope * x
    dof = max(len(x) - 2, 1)
    sigma2 = float(np.sum(w * resid**2) / (sw * dof / len(x)))
    slope_se = math.sqrt(sigma2 / sxx) if sxx > 0 else float("nan")
    intercept_se = math.sqrt(sigma2 * (1.0 / sw + xbar * xbar / sxx)) if sxx > 0 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_se": float(slope_se),
        "intercept_se": float(intercept_se),
        "n": int(len(x)),
    }


def photon_weighted_airmass_slope(
    mags: np.ndarray,
    airmass: np.ndarray,
    err_mag: np.ndarray,
    *,
    min_airmass_range: float = 0.15,
) -> dict[str, float]:
    """Robust airmass slope on mean-detrended mags with photon weights 1/err^2."""
    m = np.asarray(mags, dtype=np.float64)
    x = np.asarray(airmass, dtype=np.float64)
    e = np.asarray(err_mag, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(x) & np.isfinite(e) & (e > 0)
    if int(ok.sum()) < 3:
        return {
            "b_X": float("nan"),
            "b_X_se": float("nan"),
            "n_epochs": int(ok.sum()),
            "airmass_range": float("nan"),
            "excluded_lever_arm": True,
        }
    m, x, e = m[ok], x[ok], e[ok]
    am_range = float(x.max() - x.min())
    if am_range < min_airmass_range:
        return {
            "b_X": float("nan"),
            "b_X_se": float("nan"),
            "n_epochs": int(len(m)),
            "airmass_range": am_range,
            "excluded_lever_arm": True,
        }
    m_det = m - float(np.mean(m))
    w = 1.0 / (e * e)
    fit = weighted_linear_regression(x, m_det, w)
    return {
        "b_X": fit["slope"],
        "b_X_se": fit["slope_se"],
        "n_epochs": int(len(m)),
        "airmass_range": am_range,
        "excluded_lever_arm": False,
    }


def lag1_autocorrelation(series: np.ndarray) -> float:
    """Lag-1 autocorrelation of a finite series (pairwise complete)."""
    y = np.asarray(series, dtype=np.float64)
    ok = np.isfinite(y)
    y = y[ok]
    if y.size < 3:
        return float("nan")
    y0 = y[:-1]
    y1 = y[1:]
    if float(np.std(y0)) <= 0 or float(np.std(y1)) <= 0:
        return float("nan")
    return float(np.corrcoef(y0, y1)[0, 1])


def expected_k2_sign(band: str) -> float | None:
    """Expected sign of rho(b_X vs signed colour offset) from literature k2 in BP-RP domain.

    Returns +1, -1, or None (no prediction / k2 off).
    """
    b = band.strip().upper()
    if b in ("CLEAR", "UNFILTERED", "NOFILTER"):
        return None
    if b in ("I", "IC", "Z", "ZP"):
        return 1.0
    if b in ("B", "V", "R", "RC", "G", "GP"):
        return -1.0 if b != "V" else None  # V literature ~0; weak sign
    return -1.0  # default negative for standard filters (g, r, B, R)


def k2_eff_ci95(
    k2_eff: float,
    k2_se: float,
    *,
    z: float = 1.96,
) -> dict[str, float]:
    """Two-sided 95% CI for WLS k2_eff [mag / airmass / mag_colour]."""
    eff = float(k2_eff)
    se = float(k2_se)
    if not (math.isfinite(eff) and math.isfinite(se) and se >= 0):
        return {
            "k2_eff": eff,
            "k2_eff_se": se,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "ci_half_width": float("nan"),
        }
    half = z * se
    return {
        "k2_eff": eff,
        "k2_eff_se": se,
        "ci_lo": eff - half,
        "ci_hi": eff + half,
        "ci_half_width": half,
    }


def colour_offset_percentiles(
    stars: list[dict[str, Any]],
    *,
    key: str = "colour_offset_signed",
) -> dict[str, float | int]:
    """p5/p95 of signed colour offsets in a cohort cell."""
    vals = [
        float(s[key])
        for s in stars
        if s.get(key) is not None and math.isfinite(float(s[key]))
    ]
    if not vals:
        return {"n": 0, "p5": float("nan"), "p95": float("nan"), "span": float("nan")}
    arr = np.asarray(vals, dtype=np.float64)
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    return {"n": len(vals), "p5": p5, "p95": p95, "span": p95 - p5}


def extract_cell_report_stats(cell: dict[str, Any]) -> dict[str, Any]:
    """Report-only statistics from one cohort cell (no re-analysis)."""
    t1 = cell.get("t1") or {}
    k2 = float(t1.get("k2_eff_mag_per_airmass_per_colour", float("nan")))
    se = float(t1.get("k2_eff_se", float("nan")))
    ci = k2_eff_ci95(k2, se)
    stars = [s for s in cell.get("stars", []) if not s.get("t1_lever_excluded")]
    colour = colour_offset_percentiles(stars)
    bx = [
        float(s["b_X"])
        for s in stars
        if s.get("b_X") is not None and math.isfinite(float(s["b_X"]))
    ]
    bx_std = float(np.std(np.asarray(bx, dtype=np.float64))) if len(bx) >= 2 else float("nan")
    span = float(colour.get("span", float("nan")))
    max_slope_spread = abs(k2) * span if math.isfinite(k2) and math.isfinite(span) else float("nan")
    sp = t1.get("spearman") or {}
    t1_fdr = cell.get("t1_fdr") or {}
    t2_fdr = cell.get("t2_fdr") or {}
    return {
        "cell_key": cell.get("cell_key"),
        "n_t1": int(t1.get("n_stars_t1", 0) or 0),
        "k2_eff": ci["k2_eff"],
        "k2_eff_se": ci["k2_eff_se"],
        "ci_lo": ci["ci_lo"],
        "ci_hi": ci["ci_hi"],
        "ci_half_width": ci["ci_half_width"],
        "sensitivity_exclude_abs_k2_gt": ci["ci_half_width"],
        "colour_p5": colour["p5"],
        "colour_p95": colour["p95"],
        "colour_span": colour["span"],
        "b_X_std": bx_std,
        "max_slope_spread_at_k2_eff": max_slope_spread,
        "rho_t1": float(sp.get("rho", float("nan"))),
        "q_t1": float(t1_fdr.get("q_value", float("nan"))),
        "rho_t2": float(t2_fdr.get("rho", float("nan"))),
        "q_t2": float(t2_fdr.get("q_value", float("nan"))),
        "power_rho0.4": float(cell.get("spearman_power_rho0.4", float("nan"))),
    }


def k2_priority_verdict(
    cells: list[dict[str, Any]],
    *,
    fdr_q: float = 0.05,
    rho_up_threshold: float = 0.3,
    rho_power: float = 0.4,
    alpha: float = 0.05,
    power_target: float = 0.8,
    min_cell_n: int = 10,
) -> dict[str, Any]:
    """Apply pre-registered UP/DOWN/UNCHANGED rule to per-cell test results.

    DOWN requires every tested (rig, band) cell to be null and each to have >= power_target
    (verbatim frozen rule: "each cell had >= 80% power").
    """
    min_n_power = spearman_min_n_for_power(rho_alt=rho_power, alpha=alpha, power=power_target)
    up_hits: list[str] = []
    tested: list[str] = []
    tested_null: list[str] = []
    tested_power_adequate: list[str] = []
    for cell in cells:
        key = cell.get("cell_key", "")
        if cell.get("excluded"):
            continue
        n = int(cell.get("n_t1", 0) or 0)
        if n < min_cell_n:
            cell["status"] = "excluded_for_power"
            continue
        power_frac = spearman_power_at_n(n, rho_alt=rho_power, alpha=alpha)
        cell["spearman_power_rho0.4"] = power_frac
        cell["power_adequate"] = bool(power_frac >= power_target)
        t1 = cell.get("t1_fdr") or {}
        t2 = cell.get("t2_fdr") or {}
        for test_name, block, is_t1 in (
            ("T1", t1, True),
            ("T2", t2, False),
        ):
            rho = float(block.get("rho", float("nan")))
            qv = float(block.get("q_value", float("nan")))
            exp_sign = block.get("expected_sign")
            sign_ok = True
            if is_t1 and exp_sign is not None and math.isfinite(rho):
                sign_ok = (rho * float(exp_sign)) > 0
            sig = (
                math.isfinite(rho)
                and abs(rho) >= rho_up_threshold
                and math.isfinite(qv)
                and qv <= fdr_q
                and (not is_t1 or sign_ok)
            )
            if is_t1 and sig:
                up_hits.append(f"{key}:{test_name}")
        t1_null = not (t1.get("reject") and abs(float(t1.get("rho", 0))) >= rho_up_threshold)
        t2_null = not t2.get("reject")
        both_null = t1_null and t2_null
        tested.append(key)
        if both_null:
            tested_null.append(key)
        if cell["power_adequate"]:
            tested_power_adequate.append(key)
    if up_hits:
        verdict = "UP"
    elif (
        tested
        and len(tested_null) == len(tested)
        and len(tested_power_adequate) == len(tested)
    ):
        verdict = "DOWN"
    else:
        verdict = "UNCHANGED"
    return {
        "verdict": verdict,
        "min_n_for_80pct_power_rho0.4": min_n_power,
        "up_hits": up_hits,
        "tested_cells": tested,
        "tested_null_cells": tested_null,
        "tested_power_adequate_cells": tested_power_adequate,
        # Legacy keys retained for summary JSON readers.
        "down_eligible_cells": tested_power_adequate,
        "down_null_cells": [k for k in tested_null if k in tested_power_adequate],
    }
