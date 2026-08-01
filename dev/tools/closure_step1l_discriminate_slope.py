#!/usr/bin/env python3
"""Step 1l: discriminate slope mechanism; confirm D5-2 (L1-L4).

Diagnostic only. No production change.

Usage:
  python dev/tools/closure_step1l_discriminate_slope.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --step1b-json tmp/closure_step1b_results.json \\
    --cache tmp/closure_step1f_ee_cache.npz \\
    --out tmp/closure_step1l_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats

from closure_step1b_differential_aperture import _load_all_proc
from closure_step1j_test_f12_normalisation import GEOM_HARNESS, _measure_cog

warnings.filterwarnings("ignore", category=fits.verify.VerifyWarning)


def _fit_y_vs_x(
    y: np.ndarray,
    *xs: np.ndarray,
    names: list[str] | None = None,
) -> dict[str, Any]:
    m = np.isfinite(y)
    for x in xs:
        m &= np.isfinite(x)
    if m.sum() < len(xs) + 3:
        return {"n": int(m.sum()), "insufficient": True}
    yf = y[m]
    if not xs:
        return {"n": int(m.sum())}
    X = np.column_stack([np.ones(m.sum())] + [x[m] for x in xs])
    coef, _, _, _ = np.linalg.lstsq(X, yf, rcond=None)
    resid = yf - X @ coef
    dof = max(m.sum() - X.shape[1], 1)
    s2 = float(np.sum(resid**2) / dof)
    cov = s2 * np.linalg.lstsq(X.T @ X, np.eye(X.shape[1]), rcond=None)[0]
    labels = ["intercept"] + (names or [f"x{i}" for i in range(len(xs))])
    out: dict[str, Any] = {"n": int(m.sum()), "r_squared": float(1 - np.sum(resid**2) / np.sum((yf - np.mean(yf)) ** 2))}
    for i, lab in enumerate(labels):
        out[f"coef_{lab}"] = float(coef[i])
        out[f"se_{lab}"] = float(math.sqrt(cov[i, i]))
    if len(xs) == 1:
        out["slope"] = float(coef[1])
        out["slope_se"] = float(math.sqrt(cov[1, 1]))
    return out


def _g_only_residual(log_f: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    fit = _fit_y_vs_x(log_f, g, names=["G"])
    m = np.isfinite(log_f) & np.isfinite(g)
    slope = fit.get("coef_G", float("nan"))
    intercept = fit.get("coef_intercept", float("nan"))
    resid = np.full_like(log_f, np.nan, dtype=np.float64)
    resid[m] = log_f[m] - (slope * g[m] + intercept)
    return resid, fit


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 5:
        return float("nan")

    def _resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        sl, ic, _, _, _ = stats.linregress(b, a)
        return a - (sl * b + ic)

    xm, ym, zm = x[m], y[m], z[m]
    return float(np.corrcoef(_resid(xm, zm), _resid(ym, zm))[0, 1])


def _bin_residual(
    resid: np.ndarray,
    bin_var: np.ndarray,
    edges: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = np.isfinite(resid) & np.isfinite(bin_var) & (bin_var >= lo) & (bin_var < hi)
        if m.sum() >= 3:
            out.append({
                "lo": lo, "hi": hi, "n": int(m.sum()),
                "mean_residual_dex": float(np.mean(resid[m])),
                "median_residual_dex": float(np.median(resid[m])),
            })
    return out


def _l1_block(
    label: str,
    log_f: np.ndarray,
    g: np.ndarray,
    r50: np.ndarray,
    peak: np.ndarray,
    fwhm_est: np.ndarray,
    vy_fwhm: np.ndarray,
) -> dict[str, Any]:
    resid, fit = _g_only_residual(log_f, g)
    m = np.isfinite(resid)

    def _corr(a: np.ndarray) -> dict[str, float]:
        mm = m & np.isfinite(a)
        if mm.sum() < 5:
            return {"pearson": float("nan"), "spearman": float("nan"), "n": int(mm.sum())}
        return {
            "n": int(mm.sum()),
            "pearson": float(np.corrcoef(resid[mm], a[mm])[0, 1]),
            "spearman": float(stats.spearmanr(resid[mm], a[mm]).correlation),
        }

    r50_edges = [1.60, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 2.05]
    peak_edges = [0, 2000, 4000, 8000, 20000, 1e9]

    mm = m & np.isfinite(peak) & np.isfinite(r50)
    return {
        "label": label,
        "fit_G_only": fit,
        "corr_residual_r50": _corr(r50),
        "corr_residual_fwhm_estimate_px": _corr(fwhm_est),
        "corr_residual_VY_FWHM": _corr(vy_fwhm),
        "corr_residual_peak_adu": _corr(peak),
        "partial_corr_residual_peak_given_r50": float(_partial_corr(resid[mm], peak[mm], r50[mm])) if mm.sum() >= 5 else float("nan"),
        "residual_binned_by_r50": _bin_residual(resid, r50, r50_edges),
        "residual_binned_by_peak_adu": _bin_residual(resid, peak, peak_edges),
    }


def _l2_block(
    log_f: np.ndarray,
    g: np.ndarray,
    bprp: np.ndarray,
) -> dict[str, Any]:
    med_g = float(np.nanmedian(g))
    bright = g >= med_g
    faint = g < med_g

    def _half(mask: np.ndarray, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "n": int(mask.sum()),
            "fit_G_only": _fit_y_vs_x(log_f[mask], g[mask], names=["G"]),
            "fit_G_and_BPRP": _fit_y_vs_x(log_f[mask], g[mask], bprp[mask], names=["G", "BPRP"]),
            "pearson_G_BPRP": float(np.corrcoef(g[mask & np.isfinite(bprp)], bprp[mask & np.isfinite(bprp)])[0, 1])
            if (mask & np.isfinite(bprp)).sum() >= 5 else float("nan"),
        }

    m = np.isfinite(log_f) & np.isfinite(g) & np.isfinite(bprp)
    return {
        "median_G_split": med_g,
        "full_sample_pearson_G_BPRP": float(np.corrcoef(g[m], bprp[m])[0, 1]) if m.sum() >= 5 else float("nan"),
        "faint_half": _half(faint, "faint"),
        "bright_half": _half(bright, "bright"),
        "full_sample": {
            "fit_G_only": _fit_y_vs_x(log_f, g, names=["G"]),
            "fit_G_and_BPRP": _fit_y_vs_x(log_f, g, bprp, names=["G", "BPRP"]),
        },
    }


def _l3_block(
    log_f: np.ndarray,
    g: np.ndarray,
    aperture_r: np.ndarray,
    sky_pp: np.ndarray,
    sky_area: np.ndarray,
    flux_large: np.ndarray | None = None,
    flux_small: np.ndarray | None = None,
) -> dict[str, Any]:
    sky_sub = sky_pp * sky_area
    out: dict[str, Any] = {
        "fit_G_only": _fit_y_vs_x(log_f, g, names=["G"]),
        "fit_G_and_aperture_r_px": _fit_y_vs_x(log_f, g, aperture_r, names=["G", "aperture_r_px"]),
        "fit_G_and_sky_subtracted_adu": _fit_y_vs_x(log_f, g, sky_sub, names=["G", "sky_sub_adu"]),
        "fit_G_aperture_and_sky": _fit_y_vs_x(log_f, g, aperture_r, sky_sub, names=["G", "aperture_r_px", "sky_sub_adu"]),
    }
    if flux_large is not None:
        lf = np.log10(np.where(flux_large > 0, flux_large, np.nan))
        out["flux_large_fit_G_only"] = _fit_y_vs_x(lf, g, names=["G"])
    if flux_small is not None:
        ls = np.log10(np.where(flux_small > 0, flux_small, np.nan))
        out["flux_small_fit_G_only"] = _fit_y_vs_x(ls, g, names=["G"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--step1b-json", type=Path, default=Path("tmp/closure_step1b_results.json"))
    ap.add_argument("--cache", type=Path, default=Path("tmp/closure_step1f_ee_cache.npz"))
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1l_diagnostics.json"))
    args = ap.parse_args()

    draft = args.draft.resolve()
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    csvs, catalog_full, _ = _load_all_proc(lights)

    with args.step1b_json.open(encoding="utf-8") as f:
        star_ids: list[str] = json.load(f)["part_a"]["star_ids"]

    meta: dict[str, dict[str, float]] = {}
    for sid in star_ids:
        row = catalog_full[catalog_full["catalog_id"].astype(str) == sid]
        if row.empty:
            continue
        meta[sid] = {"phot_g": float(row.iloc[0]["phot_g"]), "bp_rp": float("nan")}

    npz = np.load(args.cache, allow_pickle=True)
    ee_cache = npz["ee_cache"].item()
    r50_series = np.array(npz["r50_series"], dtype=np.float64)
    vy_series = np.array(npz["vy_series"], dtype=np.float64)

    harness_log_f: list[float] = []
    harness_g: list[float] = []
    harness_r50: list[float] = []
    harness_peak: list[float] = []
    harness_fwhm: list[float] = []
    harness_vy: list[float] = []
    harness_bprp: list[float] = []

    prod_log_f: list[float] = []
    prod_g: list[float] = []
    prod_r50: list[float] = []
    prod_peak: list[float] = []
    prod_fwhm: list[float] = []
    prod_vy: list[float] = []
    prod_bprp: list[float] = []
    prod_ap: list[float] = []
    prod_sky_pp: list[float] = []
    prod_sky_area: list[float] = []
    prod_flux_large: list[float] = []
    prod_flux_small: list[float] = []

    for fi, proc in enumerate(csvs):
        fn = proc.name
        fits_path = lights / fn.replace(".csv", ".fits")
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        r50_f = float(r50_series[fi])
        vy_f = float(vy_series[fi])

        with fits.open(fits_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float64)

        for sid in star_ids:
            if sid not in ee_cache.get(fi, {}):
                continue
            pr = df.loc[df["catalog_id"] == sid]
            if pr.empty:
                continue
            pr = pr.iloc[0]
            if sid in meta and math.isfinite(float(pr.get("bp_rp", float("nan")))):
                meta[sid]["bp_rp"] = float(pr["bp_rp"])

            cached = ee_cache[fi][sid]
            xc, yc = float(cached["centroid_x"]), float(cached["centroid_y"])
            m = _measure_cog(data, xc, yc, *GEOM_HARNESS)
            if m is not None and m["F_12"] > 0:
                harness_log_f.append(math.log10(m["F_12"]))
                harness_g.append(meta[sid]["phot_g"])
                harness_r50.append(r50_f)
                harness_peak.append(float(pr.get("peak_max_adu", float("nan"))))
                harness_fwhm.append(float(pr.get("fwhm_estimate_px", float("nan"))))
                harness_vy.append(vy_f)
                harness_bprp.append(meta[sid].get("bp_rp", float("nan")))

            flux = float(pr.get("flux", float("nan")))
            mag = float(pr.get("mag", meta[sid]["phot_g"]))
            ok = str(pr.get("photometry_ok", "true")).lower() in ("true", "1", "yes")
            usable = str(pr.get("is_usable", "true")).lower() in ("true", "1", "yes")
            if not (math.isfinite(flux) and flux > 0 and ok and usable):
                continue
            ap_r = float(pr.get("aperture_r_px", float("nan")))
            sky_pp = float(pr.get("sky_adu_per_px_annulus", float("nan")))
            sky_out = float(pr.get("sky_annulus_r_out_px", float("nan")))
            sky_in = max(ap_r + 0.5, 4.75 * 2.395)
            sky_area = math.pi * (sky_out**2 - sky_in**2) if math.isfinite(sky_out) else float("nan")

            prod_log_f.append(math.log10(flux))
            prod_g.append(mag)
            prod_r50.append(r50_f)
            prod_peak.append(float(pr.get("peak_max_adu", float("nan"))))
            prod_fwhm.append(float(pr.get("fwhm_estimate_px", float("nan"))))
            prod_vy.append(vy_f)
            prod_bprp.append(float(pr.get("bp_rp", meta[sid].get("bp_rp", float("nan")))))
            prod_ap.append(ap_r)
            prod_sky_pp.append(sky_pp)
            prod_sky_area.append(sky_area)
            fl = float(pr.get("flux_large", float("nan")))
            fs = float(pr.get("flux_small", float("nan")))
            prod_flux_large.append(fl if math.isfinite(fl) and fl > 0 else float("nan"))
            prod_flux_small.append(fs if math.isfinite(fs) and fs > 0 else float("nan"))

    def _arr(x: list[float]) -> np.ndarray:
        return np.array(x, dtype=np.float64)

    h_log, h_g = _arr(harness_log_f), _arr(harness_g)
    p_log, p_g = _arr(prod_log_f), _arr(prod_g)

    l1 = {
        "harness_F12": _l1_block("harness_F12", h_log, h_g, _arr(harness_r50), _arr(harness_peak),
                                  _arr(harness_fwhm), _arr(harness_vy)),
        "production_flux": _l1_block("production_flux", p_log, p_g, _arr(prod_r50), _arr(prod_peak),
                                       _arr(prod_fwhm), _arr(prod_vy)),
    }

    l2 = {
        "harness_F12": _l2_block(h_log, h_g, _arr(harness_bprp)),
        "production_flux": _l2_block(p_log, p_g, _arr(prod_bprp)),
    }

    l3 = _l3_block(
        p_log, p_g, _arr(prod_ap), _arr(prod_sky_pp), _arr(prod_sky_area),
        _arr(prod_flux_large), _arr(prod_flux_small),
    )

    # outcome helpers
    pr_r50_h = l1["harness_F12"]["corr_residual_r50"]["pearson"]
    pr_r50_p = l1["production_flux"]["corr_residual_r50"]["pearson"]
    slope_ap = l3["fit_G_and_aperture_r_px"].get("coef_G", float("nan"))

    out = {"L1": l1, "L2": l2, "L3": l3, "summary": {
        "harness_pearson_residual_r50": pr_r50_h,
        "production_pearson_residual_r50": pr_r50_p,
        "production_slope_G_after_aperture_control": slope_ap,
        "production_slope_G_only": l3["fit_G_only"].get("coef_G", float("nan")),
    }}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    print("L1 prod pearson(residual,r50):", pr_r50_p)
    print("L3 prod slope G after aperture:", slope_ap)


if __name__ == "__main__":
    main()
