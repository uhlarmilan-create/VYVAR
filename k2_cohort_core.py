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


def k2_priority_verdict(
    cells: list[dict[str, Any]],
    *,
    fdr_q: float = 0.05,
    rho_up_threshold: float = 0.3,
    rho_power: float = 0.4,
    alpha: float = 0.05,
    power_target: float = 0.8,
) -> dict[str, Any]:
    """Apply pre-registered UP/DOWN/UNCHANGED rule to per-cell test results."""
    min_n_power = spearman_min_n_for_power(rho_alt=rho_power, alpha=alpha, power=power_target)
    up_hits: list[str] = []
    down_eligible: list[str] = []
    down_null: list[str] = []
    for cell in cells:
        key = cell.get("cell_key", "")
        if cell.get("excluded"):
            continue
        n = int(cell.get("n_t1", 0) or 0)
        if n < 10:
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
        if cell["power_adequate"]:
            down_eligible.append(key)
            if both_null:
                down_null.append(key)
    if up_hits:
        verdict = "UP"
    elif down_eligible and len(down_null) == len(down_eligible) and down_eligible:
        verdict = "DOWN"
    else:
        verdict = "UNCHANGED"
    return {
        "verdict": verdict,
        "min_n_for_80pct_power_rho0.4": min_n_power,
        "up_hits": up_hits,
        "down_eligible_cells": down_eligible,
        "down_null_cells": down_null,
    }
