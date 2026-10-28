"""Faint limit of a MASTERSTAR from its own data (FRAME-DEPTH-2026-10, Milan 2026-10-14).

No static magnitude parameter. Two independent measurements from the frame:

* **Completeness fit** - fraction of on-chip Gaia stars with their own pass-1 DAO detection within
  the identity radius (brightest-first 1:1), per 0.5 mag G bin, corrected for chance coincidences
  ``p = 1 - exp(-rho pi r^2)`` (rho = detection density), fitted with the Fleming et al. (1995)
  erf: C(G) = 0.5 (1 + erf((G50 - G) / (sqrt 2 sigma))). Frame limit G_lim = G50 + 3 sigma
  (completeness ~0.1 %).
* **Detector estimate** (used when the catalogue ends before the completeness drop, i.e. the fit is
  censored by the DB depth): a point source is detected when its peak reaches the DAO threshold.
  G = ZP_peak - 2.5 log10(peak) from the identified stars gives G50 ~ ZP_peak - 2.5 log10(threshold);
  sigma = robust scatter of that relation for the fainter half of the stars.

The usable depth for the matching phase is min(G_lim, catalogue depth); G_lim itself is kept so a
deeper catalogue can be fetched for the field (gaia_coverage.ensure_field_depth). This is the one
depth of VYVAR: G50, G90 = G50 - 1.28 sigma and G_lim feed the matching, the census, the
reconciliation report and the crowding diagnostic.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from pipeline_constants import Z_90  # LAYER-2026-10: moved there

#: G_lim = G50 + N_SIGMA * sigma (C = 0.13 %): beyond it the frame holds practically no detections.
N_SIGMA_LIMIT = 3.0
_BIN_MAG = 0.5
_MIN_STARS_ZP = 20


def _completeness_bins(g: np.ndarray, matched: np.ndarray, p_chance: float) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    if g.size == 0:
        return bins
    lo = math.floor(float(np.min(g)) / _BIN_MAG) * _BIN_MAG
    hi = math.ceil(float(np.max(g)) / _BIN_MAG) * _BIN_MAG
    edge = lo
    while edge < hi + 1e-9:
        m = (g >= edge) & (g < edge + _BIN_MAG)
        n = int(m.sum())
        if n > 0:
            f_raw = float(matched[m].sum()) / float(n)
            f = (f_raw - p_chance) / (1.0 - p_chance) if p_chance < 1.0 else f_raw
            bins.append({
                "bin_lo": round(edge, 3), "bin_hi": round(edge + _BIN_MAG, 3),
                "bin_center": round(edge + 0.5 * _BIN_MAG, 3), "n_ref": n,
                "n_matched": int(matched[m].sum()), "completeness_frac": round(min(1.0, max(0.0, f)), 4),
            })
        edge += _BIN_MAG
    return bins


def measure_frame_depth(
    *,
    det_x: Any,
    det_y: Any,
    det_peak: Any,
    gaia_x: Any,
    gaia_y: Any,
    gaia_g: Any,
    match_r_px: float,
    threshold_adu: float,
    wpx: int,
    h: int,
    db_depth_g: float | None,
    sat_peak_adu: float | None = None,
) -> dict[str, Any]:
    """Frame faint limit from pass-1 DAO detections vs on-chip Gaia (see module docstring)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    from dao_reconcile import fit_fleming_completeness  # noqa: PLC0415

    dx = np.asarray(det_x, dtype=np.float64)
    dy = np.asarray(det_y, dtype=np.float64)
    dp = np.asarray(det_peak, dtype=np.float64)
    gx = np.asarray(gaia_x, dtype=np.float64)
    gy = np.asarray(gaia_y, dtype=np.float64)
    gg = np.asarray(gaia_g, dtype=np.float64)
    okd = np.isfinite(dx) & np.isfinite(dy)
    okg = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(gg)
    dx, dy, dp = dx[okd], dy[okd], dp[okd]
    gx, gy, gg = gx[okg], gy[okg], gg[okg]
    r = max(1.0, float(match_r_px))
    out: dict[str, Any] = {
        "g_lim": None, "g_lim_frame": None, "g50": None, "sigma_mag": None, "method": "none",
        "censored": False, "db_depth_g": float(db_depth_g) if db_depth_g else None,
        "n_gaia_on_chip": int(gg.size), "n_det_pass1": int(dx.size), "match_r_px": r,
    }
    if gg.size < _MIN_STARS_ZP or dx.size < _MIN_STARS_ZP:
        out["method"] = "too_few_stars"
        return out

    # Brightest-first 1:1 (the identity rule): a Gaia star claims the nearest free detection
    # within r; a faint neighbour of an already claimed detection stays unmatched (it is not
    # detected on its own) and never lends its G to the bright star's peak.
    tree = cKDTree(np.column_stack([dx, dy]))
    cand = tree.query_ball_point(np.column_stack([gx, gy]), r=r)
    taken = np.zeros(dx.size, dtype=bool)
    i_nn = np.full(gg.size, -1, dtype=np.int64)
    for j in np.argsort(gg, kind="stable"):
        best, best_d = -1, np.inf
        for i in cand[j]:
            if not taken[i]:
                d = (dx[i] - gx[j]) ** 2 + (dy[i] - gy[j]) ** 2
                if d < best_d:
                    best, best_d = i, d
        if best >= 0:
            taken[best] = True
            i_nn[j] = best
    matched = i_nn >= 0
    rho = float(dx.size) / float(max(1, int(wpx) * int(h)))
    p_chance = 1.0 - math.exp(-rho * math.pi * r * r)
    # Catalogue depth of this field (gaia_coverage); deeper stars of a neighbouring top-up are a
    # partial sample of the chip and stay out of the completeness curve.
    g_cat = float(np.max(gg))
    depth = (
        float(db_depth_g)
        if db_depth_g is not None and math.isfinite(float(db_depth_g)) and float(db_depth_g) > 0
        else g_cat
    )
    in_depth = gg <= depth + 1e-6
    bins = _completeness_bins(gg[in_depth], matched[in_depth], p_chance)
    fit = fit_fleming_completeness(bins)
    out.update({"p_chance": round(p_chance, 5), "completeness_bins": bins, "fit_method": fit.fit_method})
    if fit.fit_method == "fleming1995_erf" and fit.g_lim_50 is not None and fit.sigma_mag is not None:
        out["g50_fit"] = round(float(fit.g_lim_50), 3)
        out["sigma_fit"] = round(float(fit.sigma_mag), 3)

    # Detector estimate: G = ZP_peak - 2.5 log10(peak) over identified, unsaturated stars.
    ip = i_nn[matched]
    pk = dp[ip]
    g_m = gg[matched]
    ok = np.isfinite(pk) & (pk > 0)
    if sat_peak_adu is not None and math.isfinite(float(sat_peak_adu)) and float(sat_peak_adu) > 0:
        ok &= pk < float(sat_peak_adu)
    if int(ok.sum()) >= _MIN_STARS_ZP and threshold_adu > 0:
        res = g_m[ok] + 2.5 * np.log10(pk[ok])
        zp = float(np.median(res))
        faint = g_m[ok] >= np.median(g_m[ok])
        rf = res[faint]
        sig = float(1.4826 * np.median(np.abs(rf - np.median(rf)))) if rf.size >= 10 else float("nan")
        g50_est = zp - 2.5 * math.log10(float(threshold_adu))
        out.update({"zp_peak": round(zp, 3), "g50_detector": round(g50_est, 3),
                    "sigma_detector": round(sig, 3) if math.isfinite(sig) else None})

    # The fit is trusted when its limit lies inside the catalogue (the drop is in the data).
    if "g50_fit" in out and out["g50_fit"] + N_SIGMA_LIMIT * out["sigma_fit"] <= min(depth, g_cat):
        g50, sig, method = out["g50_fit"], out["sigma_fit"], "completeness_fit"
    elif "g50_detector" in out and out.get("sigma_detector"):
        g50, sig, method = out["g50_detector"], float(out["sigma_detector"]), "detector_estimate"
        out["censored"] = True
    elif "g50_fit" in out:
        g50, sig, method = out["g50_fit"], out["sigma_fit"], "completeness_fit_censored"
        out["censored"] = True
    else:
        out["method"] = "undetermined"
        return out
    g_lim_frame = float(g50) + N_SIGMA_LIMIT * float(sig)
    out.update({
        "g50": round(float(g50), 3), "sigma_mag": round(float(sig), 3), "method": method,
        "g90": round(float(g50) - Z_90 * float(sig), 3),
        "g_lim_frame": round(g_lim_frame, 3), "g_lim": round(min(g_lim_frame, depth), 3),
        "limited_by_catalogue": bool(g_lim_frame > depth),
    })
    return out
