#!/usr/bin/env python3
"""Step 1k: decompose F(12) slope; test production path (K1-K4).

Diagnostic only. No production change.

Usage:
  python dev/tools/closure_step1k_decompose_f12_slope.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --step1b-json tmp/closure_step1b_results.json \\
    --cache tmp/closure_step1f_ee_cache.npz \\
    --out tmp/closure_step1k_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from astropy.io import fits
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from scipy import stats

from closure_step1b_differential_aperture import _load_all_proc
from closure_step1c_differential_aperture import PROXY_R_AP
from closure_step1e_differential_aperture import COG_DR, COG_RMAX
from closure_step1j_test_f12_normalisation import (
    FW_PROD,
    GEOM_HARNESS,
    _measure_cog,
    _prod_annulus,
)

warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)

A12 = math.pi * 12.0**2  # px^2 at r=12
R_AP = PROXY_R_AP


def _fit_log_flux_vs_g(
    log_f: np.ndarray,
    g: np.ndarray,
    bprp: np.ndarray | None = None,
) -> dict[str, float]:
    m = np.isfinite(log_f) & np.isfinite(g)
    if bprp is not None:
        m &= np.isfinite(bprp)
    if m.sum() < 5:
        return {"n": int(m.sum()), "insufficient": True}
    y = log_f[m]
    if bprp is None:
        slope, intercept, r, p, se = stats.linregress(g[m], y)
        return {
            "n": int(m.sum()),
            "slope_G": float(slope),
            "slope_G_se": float(se),
            "intercept": float(intercept),
            "r_squared": float(r**2),
        }
    X = np.column_stack([np.ones(m.sum()), g[m], bprp[m]])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(m.sum() - 3, 1)
    s2 = float(np.sum(resid**2) / dof)
    cov = s2 * np.linalg.inv(X.T @ X)
    return {
        "n": int(m.sum()),
        "intercept": float(coef[0]),
        "slope_G": float(coef[1]),
        "slope_G_se": float(math.sqrt(cov[1, 1])),
        "slope_BPRP": float(coef[2]),
        "slope_BPRP_se": float(math.sqrt(cov[2, 2])),
        "r_squared": float(1.0 - np.sum(resid**2) / np.sum((y - np.mean(y)) ** 2)),
    }


def _same_mag_ratios(
    star_med_f: dict[str, float],
    meta: dict[str, dict[str, float]],
    dg: float = 0.05,
) -> dict[str, Any]:
    ratios = []
    for a, b in combinations(star_med_f.keys(), 2):
        if a not in meta or b not in meta:
            continue
        ga, gb = meta[a]["phot_g"], meta[b]["phot_g"]
        if abs(ga - gb) <= dg:
            ratios.append(star_med_f[a] / star_med_f[b])
    return {
        "n_pairs": len(ratios),
        "median": float(np.median(ratios)) if ratios else float("nan"),
        "p95": float(np.percentile(ratios, 95)) if ratios else float("nan"),
        "max": float(np.max(ratios)) if ratios else float("nan"),
    }


def _star_medians(rows: list[dict[str, Any]], key: str, phot_g: dict[str, float]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for r in rows:
        if key not in r or not math.isfinite(r[key]) or r[key] <= 0:
            continue
        acc.setdefault(r["star_id"], []).append(float(r[key]))
    return {sid: float(np.median(v)) for sid, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--step1b-json", type=Path, default=Path("tmp/closure_step1b_results.json"))
    ap.add_argument("--cache", type=Path, default=Path("tmp/closure_step1f_ee_cache.npz"))
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1k_diagnostics.json"))
    args = ap.parse_args()

    draft = args.draft.resolve()
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    csvs, catalog_full, n_frames = _load_all_proc(lights)

    with args.step1b_json.open(encoding="utf-8") as f:
        s1b = json.load(f)
    star_ids: list[str] = s1b["part_a"]["star_ids"]

    meta: dict[str, dict[str, float]] = {}
    for sid in star_ids:
        row = catalog_full[catalog_full["catalog_id"].astype(str) == sid]
        if row.empty:
            continue
        r0 = row.iloc[0]
        meta[sid] = {
            "phot_g": float(r0["phot_g"]),
            "bp_rp": float("nan"),
        }
    # bp_rp from proc CSV (not in catalogue table loaded by step1b)
    for proc in csvs[:1]:
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        for sid in star_ids:
            pr = df.loc[df["catalog_id"] == sid]
            if not pr.empty and "bp_rp" in pr.columns:
                v = float(pr.iloc[0]["bp_rp"])
                if math.isfinite(v):
                    meta[sid]["bp_rp"] = v

    npz = np.load(args.cache, allow_pickle=True)
    ee_cache = npz["ee_cache"].item()
    aperture_by_frame = npz["aperture_by_frame"].item()

    harness_rows: list[dict[str, Any]] = []
    prod_rows: list[dict[str, Any]] = []

    g1152, g1153 = "1497368849430107904", "1497091703781835776"

    for fi, proc in enumerate(csvs):
        fn = proc.name
        fits_path = lights / fn.replace(".csv", ".fits")
        df = pd.read_csv(proc, dtype={"catalog_id": str})

        with fits.open(fits_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float64)
            ny, nx = data.shape
            global_sky = float(np.median(data))

        for sid in star_ids:
            if sid not in ee_cache.get(fi, {}):
                continue
            proc_row = df.loc[df["catalog_id"] == sid]
            if proc_row.empty:
                continue
            proc_row = proc_row.iloc[0]
            cached = ee_cache[fi][sid]
            xc, yc = float(cached["centroid_x"]), float(cached["centroid_y"])
            m = _measure_cog(data, xc, yc, *GEOM_HARNESS)
            if m is None:
                continue
            sky_off = m["sky_ann"] - global_sky
            additive_corr = sky_off * A12
            f12_add = m["F_12"] + additive_corr

            harness_rows.append({
                "star_id": sid,
                "frame": fn,
                "frame_idx": fi,
                "phot_g": meta[sid]["phot_g"],
                "bp_rp": meta[sid]["bp_rp"],
                "F_12": m["F_12"],
                "F_12_additive_corrected": f12_add,
                "sky_ann": m["sky_ann"],
                "sky_ann_minus_global": sky_off,
                "additive_correction_adu": additive_corr,
                "peak_max_adu": float(proc_row.get("peak_max_adu", float("nan"))),
                "is_saturated": bool(proc_row.get("is_saturated", False)),
                "likely_saturated": bool(proc_row.get("likely_saturated", False)),
                "saturate_limit_85": float(proc_row.get("saturate_limit_adu_85pct", float("nan"))),
                "x": xc, "y": yc,
                "nx": nx, "ny": ny,
                "dist_edge_px": min(xc, yc, nx - 1 - xc, ny - 1 - yc),
            })

            # K3 production
            flux = float(proc_row.get("flux", float("nan")))
            mag = float(proc_row.get("mag", proc_row.get("catalog_mag", float("nan"))))
            if not math.isfinite(mag):
                mag = meta[sid]["phot_g"]
            ok = str(proc_row.get("photometry_ok", "true")).lower() in ("true", "1", "yes")
            usable = str(proc_row.get("is_usable", "true")).lower() in ("true", "1", "yes")
            if math.isfinite(flux) and flux > 0 and ok and usable:
                prod_rows.append({
                    "star_id": sid,
                    "frame": fn,
                    "phot_g": meta[sid]["phot_g"],
                    "mag": mag,
                    "bp_rp": float(proc_row.get("bp_rp", meta[sid]["bp_rp"])),
                    "flux": flux,
                    "dao_flux": float(proc_row.get("dao_flux", float("nan"))),
                    "sky_adu_per_px_annulus": float(proc_row.get("sky_adu_per_px_annulus", float("nan"))),
                    "aperture_r_px": float(proc_row.get("aperture_r_px", float("nan"))),
                    "sky_annulus_r_out_px": float(proc_row.get("sky_annulus_r_out_px", float("nan"))),
                    "peak_max_adu": float(proc_row.get("peak_max_adu", float("nan"))),
                    "is_saturated": bool(proc_row.get("is_saturated", False)),
                    "likely_saturated": bool(proc_row.get("likely_saturated", False)),
                })

    # K1 harness
    logf = np.log10(np.array([r["F_12"] for r in harness_rows]))
    g = np.array([r["phot_g"] for r in harness_rows])
    bprp = np.array([r["bp_rp"] for r in harness_rows])
    peak = np.array([r["peak_max_adu"] for r in harness_rows])

    fit_g = _fit_log_flux_vs_g(logf, g)
    fit_gb = _fit_log_flux_vs_g(logf, g, bprp)

    resid_g = logf - (fit_g["slope_G"] * g + fit_g["intercept"])
    pearson_peak = float(np.corrcoef(resid_g, peak)[0, 1]) if len(resid_g) >= 5 else float("nan")

    # bin peak ADU
    bins = [(0, 2000), (2000, 4000), (4000, 8000), (8000, 20000), (20000, 1e9)]
    peak_bins = []
    for lo, hi in bins:
        m = (peak >= lo) & (peak < hi)
        if m.sum() >= 3:
            peak_bins.append({
                "peak_lo": lo, "peak_hi": hi, "n": int(m.sum()),
                "mean_residual_dex": float(np.mean(resid_g[m])),
                "median_residual_dex": float(np.median(resid_g[m])),
            })

    med_g = float(np.median(g))
    bright = g >= med_g
    faint = g < med_g
    fit_bright = _fit_log_flux_vs_g(logf[bright], g[bright])
    fit_faint = _fit_log_flux_vs_g(logf[faint], g[faint])

    brightest = sorted(meta.items(), key=lambda t: t[1]["phot_g"])[:5]
    sat_report = []
    for sid, _ in brightest:
        rows = [r for r in harness_rows if r["star_id"] == sid]
        if rows:
            r0 = rows[0]
            sat_report.append({
                "star_id": sid, "phot_g": meta[sid]["phot_g"],
                "is_saturated": r0["is_saturated"],
                "likely_saturated": r0["likely_saturated"],
                "saturate_limit_85": r0["saturate_limit_85"],
                "peak_max_adu_median": float(np.median([r["peak_max_adu"] for r in rows])),
            })

    # per-star medians for colour joint fit (BP-RP is constant per star)
    med_f12 = _star_medians(harness_rows, "F_12", meta)
    med_f12_add = _star_medians(harness_rows, "F_12_additive_corrected", meta)
    common = sorted(set(med_f12) & set(med_f12_add))
    star_g = {sid: meta[sid]["phot_g"] for sid in common}
    log_med = np.log10(np.array([med_f12[s] for s in common]))
    log_med_add = np.log10(np.array([med_f12_add[s] for s in common]))
    g_med = np.array([star_g[s] for s in common])
    bprp_med = np.array([meta[s]["bp_rp"] for s in common])

    k1 = {
        "fit_G_only_star_frames": fit_g,
        "fit_G_and_BPRP_star_frames": fit_gb,
        "fit_G_only_per_star_median": _fit_log_flux_vs_g(log_med, g_med),
        "fit_G_and_BPRP_per_star_median": _fit_log_flux_vs_g(log_med, g_med, bprp_med),
        "expected_slope_G": -0.4,
        "pearson_residual_vs_peak_adu": pearson_peak,
        "residual_binned_by_peak_adu": peak_bins,
        "fit_bright_half": fit_bright,
        "fit_faint_half": fit_faint,
        "brightest_stars_saturation": sat_report,
    }

    # K2 additive / multiplicative decomposition
    fit_med_g = _fit_log_flux_vs_g(log_med, g_med)
    fit_med_add = _fit_log_flux_vs_g(log_med_add, g_med)

    pair_raw = _same_mag_ratios(med_f12, meta)
    pair_add = _same_mag_ratios(med_f12_add, meta)

    if g1152 in med_f12 and g1153 in med_f12:
        pair_1152_1153 = {
            "F12_G11.52": med_f12[g1152],
            "F12_G11.53": med_f12[g1153],
            "ratio_raw": med_f12[g1152] / med_f12[g1153],
            "ratio_after_additive": med_f12_add[g1152] / med_f12_add[g1153],
            "sky_ann_diff_adu_per_px": float(np.median([r["sky_ann_minus_global"] for r in harness_rows if r["star_id"] == g1152])
                                            - np.median([r["sky_ann_minus_global"] for r in harness_rows if r["star_id"] == g1153])),
            "implied_sky_adu_per_px_from_F12_diff": (med_f12[g1152] - med_f12[g1153]) / A12,
        }
    else:
        pair_1152_1153 = {}

    k2 = {
        "A12_px2": A12,
        "slope_per_star_median_F12_raw": fit_med_g,
        "slope_per_star_median_F12_after_additive": fit_med_add,
        "same_mag_pair_ratios_raw": pair_raw,
        "same_mag_pair_ratios_after_additive": pair_add,
        "G11.52_vs_G11.53": pair_1152_1153,
    }

    # K3 production
    if prod_rows:
        pf = np.log10(np.array([r["flux"] for r in prod_rows]))
        pm = np.array([r["mag"] for r in prod_rows])
        pb = np.array([r["bp_rp"] for r in prod_rows])
        pp = np.array([r["peak_max_adu"] for r in prod_rows])
        fit_p_g = _fit_log_flux_vs_g(pf, pm)
        fit_p_gb = _fit_log_flux_vs_g(pf, pm, pb)
        resid_p = pf - (fit_p_g["slope_G"] * pm + fit_p_g["intercept"])
        pearson_p_peak = float(np.corrcoef(resid_p, pp)[0, 1]) if len(resid_p) >= 5 else float("nan")
        k3 = {
            "n_star_frames": len(prod_rows),
            "fit_mag_only": fit_p_g,
            "fit_mag_and_BPRP": fit_p_gb,
            "expected_slope": -0.4,
            "pearson_residual_vs_peak_adu": pearson_p_peak,
            "residual_binned_by_peak_adu": [],
        }
        for lo, hi in bins:
            m = (pp >= lo) & (pp < hi)
            if m.sum() >= 3:
                k3["residual_binned_by_peak_adu"].append({
                    "peak_lo": lo, "peak_hi": hi, "n": int(m.sum()),
                    "mean_residual_dex": float(np.mean(resid_p[m])),
                })
        # dao_flux duplicate
        dao_ok = [r for r in prod_rows if math.isfinite(r["dao_flux"]) and r["dao_flux"] > 0]
        if len(dao_ok) >= 5:
            k3["fit_dao_flux_mag_only"] = _fit_log_flux_vs_g(
                np.log10(np.array([r["dao_flux"] for r in dao_ok])),
                np.array([r["mag"] for r in dao_ok]),
            )
    else:
        k3 = {"insufficient": True}

    # K4 G 12.59 edge analysis
    sid_1259 = "1499238946911605504"
    edge_rows = [r for r in harness_rows if r["star_id"] == sid_1259]
    rin_p, rout_p = _prod_annulus(R_AP)
    k4_edge = {
        "star_id": sid_1259,
        "phot_g": meta.get(sid_1259, {}).get("phot_g"),
        "n_frames_in_cache": len(edge_rows),
        "median_x": float(np.median([r["x"] for r in edge_rows])) if edge_rows else float("nan"),
        "median_y": float(np.median([r["y"] for r in edge_rows])) if edge_rows else float("nan"),
        "median_dist_edge_px": float(np.median([r["dist_edge_px"] for r in edge_rows])) if edge_rows else float("nan"),
        "min_dist_edge_px": float(np.min([r["dist_edge_px"] for r in edge_rows])) if edge_rows else float("nan"),
        "frame_shape_median": [int(np.median([r["nx"] for r in edge_rows])), int(np.median([r["ny"] for r in edge_rows]))] if edge_rows else [],
        "production_annulus_r_in_r_out": [rin_p, rout_p],
        "harness_annulus_r_in_r_out": list(GEOM_HARNESS),
        "narrow_annulus_r_in_r_out": [12.0, 20.0],
        "required_half_extent_prod": rout_p + 12.0 + 2.0,
        "note": "COG fails when centroid +/- (max(COG_RADII)+r_out+2) exceeds frame bounds",
    }
    # count prod annulus failures
    n_fail_prod = 0
    for fi, proc in enumerate(csvs):
        if sid_1259 not in ee_cache.get(fi, {}):
            continue
        with fits.open(lights / proc.name.replace(".csv", ".fits"), memmap=False) as hdul:
            data = hdul[0].data
            ny, nx = data.shape
        c = ee_cache[fi][sid_1259]
        xc, yc = float(c["centroid_x"]), float(c["centroid_y"])
        max_r = max(float(np.max(np.arange(COG_DR, COG_RMAX + COG_DR / 2, COG_DR))), rout_p) + 2.0
        if xc - max_r < 0 or yc - max_r < 0 or xc + max_r >= nx or yc + max_r >= ny:
            n_fail_prod += 1
    k4_edge["n_frames_prod_annulus_oob"] = n_fail_prod

    out = {"K1": k1, "K2": k2, "K3": k3, "K4_edge_G12_59": k4_edge}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    print("K1 slope G:", fit_g.get("slope_G"), "with BPRP per-star:", k1["fit_G_and_BPRP_per_star_median"].get("slope_G"))
    print("K1 after additive slope:", fit_med_add.get("slope_G"))
    if prod_rows:
        print("K3 prod slope:", k3["fit_mag_only"].get("slope_G"), "with BPRP:", k3["fit_mag_and_BPRP"].get("slope_G"))


if __name__ == "__main__":
    main()
