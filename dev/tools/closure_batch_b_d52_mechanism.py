#!/usr/bin/env python3
"""Batch B: settle D5-2 / D1-2 mechanism (B1 sky, B2 non-linearity).

Diagnostic only. No production change.

Usage:
  python dev/tools/closure_batch_b_d52_mechanism.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --step1b-json tmp/closure_step1b_results.json \\
    --cache tmp/closure_step1f_ee_cache.npz \\
    --out tmp/closure_batch_b_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

from closure_step1b_differential_aperture import TABLE_FWHM, _load_all_proc
from pipeline import _fit_subtract_preprocess_sky_surface

warnings.filterwarnings("ignore")

R_LARGE = 4.0 * TABLE_FWHM  # 9.58 px
SKY_IN = max(R_LARGE + 0.5, 4.75 * TABLE_FWHM)
SKY_OUT = max(SKY_IN + 0.5, 9.0 * TABLE_FWHM)
SANITY_SLOPE_LO = -0.48
SANITY_SLOPE_HI = -0.32


def _fit_slope(y: np.ndarray, g: np.ndarray) -> dict[str, Any]:
    m = np.isfinite(y) & np.isfinite(g)
    if m.sum() < 5 or float(np.std(g[m])) < 1e-9:
        return {"n": int(m.sum()), "insufficient": True}
    slope, intercept, r, _, se = stats.linregress(g[m], y[m])
    ok = SANITY_SLOPE_LO <= slope <= SANITY_SLOPE_HI
    return {
        "n": int(m.sum()),
        "slope": float(slope),
        "slope_se": float(se),
        "intercept": float(intercept),
        "r_squared": float(r**2),
        "sanity_pass": bool(ok),
    }


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 8:
        return float("nan")
    x, y, z = x[m], y[m], z[m]
    rxy = float(np.corrcoef(x, y)[0, 1])
    rxz = float(np.corrcoef(x, z)[0, 1])
    ryz = float(np.corrcoef(y, z)[0, 1])
    den = math.sqrt(max(1e-12, (1 - rxz**2) * (1 - ryz**2)))
    return float((rxy - rxz * ryz) / den)


def _ap_flux(data: np.ndarray, x: float, y: float) -> float | None:
    h, w = data.shape
    if x - SKY_OUT < 0 or y - SKY_OUT < 0 or x + SKY_OUT >= w or y + SKY_OUT >= h:
        return None
    ann = CircularAnnulus([(x, y)], r_in=SKY_IN, r_out=SKY_OUT)
    sky = float(aperture_photometry(data, ann)["aperture_sum"][0] / ann.area)
    ap = CircularAperture([(x, y)], r=R_LARGE)
    raw = float(aperture_photometry(data, ap)["aperture_sum"][0])
    fl = raw - sky * ap.area
    return fl if fl > 0 else None


def _surface_map(data_cal: np.ndarray) -> np.ndarray:
    proc, st = _fit_subtract_preprocess_sky_surface(data_cal.copy(), order=2, fwhm_px=TABLE_FWHM)
    if not st.get("sky_surface_applied"):
        return np.zeros_like(data_cal, dtype=np.float64)
    return data_cal.astype(np.float64) - proc.astype(np.float64)


def _surface_in_ap(surf: np.ndarray, x: float, y: float) -> float:
    yy, xx = np.ogrid[0 : surf.shape[0], 0 : surf.shape[1]]
    m = (xx - x) ** 2 + (yy - y) ** 2 <= R_LARGE**2
    return float(np.sum(surf[m]))


def run_b1(draft: Path, csvs: list[Path], star_ids: list[str], ee_cache: dict) -> dict[str, Any]:
    """Before/after sky on aligned grid at proc CSV x,y."""
    before_log: list[float] = []
    after_log: list[float] = []
    before_add_log: list[float] = []
    g_list: list[float] = []
    n_void = 0

    for fi, proc in enumerate(csvs):
        stem = proc.name.replace(".csv", ".fits")
        ali_p = draft / "detrended_aligned/lights/NoFilter_60_2" / stem
        cal_p = draft / "calibrated/lights/NoFilter_60_2" / stem.replace("proc_", "")
        if not ali_p.is_file() or not cal_p.is_file():
            continue
        data_ali = fits.getdata(ali_p).astype(np.float64)
        data_cal = fits.getdata(cal_p).astype(np.float64)
        surf = _surface_map(data_cal)
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        for sid in star_ids:
            if sid not in ee_cache.get(fi, {}):
                continue
            pr = df.loc[df["catalog_id"] == sid]
            if pr.empty:
                continue
            pr = pr.iloc[0]
            if str(pr.get("photometry_ok", "")).lower() not in ("true", "1", "yes"):
                continue
            if str(pr.get("is_usable", "")).lower() not in ("true", "1", "yes"):
                continue
            mag = float(pr["mag"])
            x, y = float(pr["x"]), float(pr["y"])
            fa = _ap_flux(data_ali, x, y)
            fb = _ap_flux(data_cal, x, y)
            if fa is None or fb is None:
                n_void += 1
                continue
            add = _surface_in_ap(surf, x, y)
            fb_add = fb + add
            if fb_add <= 0:
                n_void += 1
                continue
            g_list.append(mag)
            after_log.append(math.log10(fa))
            before_log.append(math.log10(fb))
            before_add_log.append(math.log10(fb_add))

    g = np.array(g_list)
    out = {
        "method": "fixed 9.58 px at proc CSV x,y; before=calibrated same pixels; after=detrended_aligned; before_add=cal+surface_in_aperture",
        "n_star_frames": len(g_list),
        "n_void_apertures": n_void,
        "slope_before_cal": _fit_slope(np.array(before_log), g),
        "slope_after_aligned": _fit_slope(np.array(after_log), g),
        "slope_before_addback": _fit_slope(np.array(before_add_log), g),
        "median_delta_log10_after_minus_before": float(np.median(np.array(after_log) - np.array(before_log))),
        "median_delta_log10_addback_minus_after": float(np.median(np.array(before_add_log) - np.array(after_log))),
    }
    before_ok = out["slope_before_cal"].get("sanity_pass") or out["slope_before_addback"].get("sanity_pass")
    out["b1_valid"] = bool(before_ok)
    if not before_ok:
        out["b1_verdict"] = "VOID (before-subtraction slope not near -0.4; sanity gate failed)"
    elif out["slope_after_aligned"].get("slope", 0) > out["slope_before_addback"].get("slope", -99) + 0.02:
        out["b1_verdict"] = "sky contributes (after shallower than before)"
    else:
        out["b1_verdict"] = "sky does not explain compression (before and after similar)"
    return out


def run_b2(rows: list[dict[str, float]]) -> dict[str, Any]:
    d = pd.DataFrame(rows)
    faint = d[d["G"] > 10.0]
    ref = _fit_slope(faint["logF"].to_numpy(), faint["G"].to_numpy())
    if ref.get("insufficient"):
        return {"insufficient": True}
    ic, sl = ref["intercept"], ref["slope"]
    d = d.copy()
    d["pred"] = ic + sl * d["G"]
    d["deficit"] = d["pred"] - d["logF"]

    bright = d[d["G"] < 9.0]
    bright1011 = d[d["G"] < 10.11]

    def _block(sub: pd.DataFrame, label: str) -> dict[str, Any]:
        if sub.empty:
            return {"label": label, "n": 0}
        m = sub.dropna(subset=["peak", "r50"])
        ols = stats.linregress(m["peak"], m["deficit"]) if len(m) >= 5 else None
        return {
            "label": label,
            "n": int(len(sub)),
            "mean_deficit_dex": float(sub["deficit"].mean()),
            "pearson_deficit_peak": float(sub["deficit"].corr(sub["peak"])) if sub["peak"].notna().sum() >= 5 else float("nan"),
            "pearson_deficit_r50": float(sub["deficit"].corr(sub["r50"])) if sub["r50"].notna().sum() >= 5 else float("nan"),
            "pearson_deficit_sky": float(sub["deficit"].corr(sub["sky"])) if sub["sky"].notna().sum() >= 5 else float("nan"),
            "partial_deficit_peak_given_r50": _partial_corr(
                sub["deficit"].to_numpy(), sub["peak"].to_numpy(), sub["r50"].to_numpy()
            ),
            "ols_deficit_on_peak_slope_per_adu": float(ols.slope) if ols else float("nan"),
        }

    peak_bins = []
    edges = [0, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 55000]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (bright["peak"] >= lo) & (bright["peak"] < hi)
        if m.sum() >= 3:
            peak_bins.append({
                "peak_lo": lo,
                "peak_hi": hi,
                "n": int(m.sum()),
                "mean_deficit_dex": float(bright.loc[m, "deficit"].mean()),
                "mean_G": float(bright.loc[m, "G"].mean()),
            })

    pc = _partial_corr(
        bright1011["deficit"].to_numpy(),
        bright1011["peak"].to_numpy(),
        bright1011["r50"].to_numpy(),
    )
    nl = pc > 0.4
    return {
        "faint_reference_G_gt_10": ref,
        "bright_G_lt_9": _block(bright, "G_lt_9"),
        "bright_G_lt_10_11": _block(bright1011, "G_lt_10_11"),
        "deficit_binned_by_peak_G_lt_9": peak_bins,
        "partial_deficit_peak_given_r50_G_lt_10_11": float(pc),
        "b2_nl_pre_registered": bool(nl),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--step1b-json", type=Path, default=Path("tmp/closure_step1b_results.json"))
    ap.add_argument("--cache", type=Path, default=Path("tmp/closure_step1f_ee_cache.npz"))
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_batch_b_diagnostics.json"))
    args = ap.parse_args()

    draft = args.draft.resolve()
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    csvs, _, _ = _load_all_proc(lights)
    with args.step1b_json.open(encoding="utf-8") as f:
        star_ids: list[str] = json.load(f)["part_a"]["star_ids"]
    npz = np.load(args.cache, allow_pickle=True)
    ee_cache = npz["ee_cache"].item()
    r50_series = np.array(npz["r50_series"], dtype=np.float64)

    rows: list[dict[str, float]] = []
    for fi, proc in enumerate(csvs):
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        r50 = float(r50_series[fi])
        for sid in star_ids:
            if sid not in ee_cache.get(fi, {}):
                continue
            pr = df.loc[df["catalog_id"] == sid]
            if pr.empty:
                continue
            pr = pr.iloc[0]
            if str(pr.get("photometry_ok", "")).lower() not in ("true", "1", "yes"):
                continue
            if str(pr.get("is_usable", "")).lower() not in ("true", "1", "yes"):
                continue
            fl = float(pr.get("flux_large", float("nan")))
            if not (math.isfinite(fl) and fl > 0):
                continue
            rows.append({
                "G": float(pr["mag"]),
                "logF": math.log10(fl),
                "peak": float(pr.get("peak_max_adu", float("nan"))),
                "r50": r50,
                "sky": float(pr.get("sky_adu_per_px_annulus", float("nan"))),
            })

    b1 = run_b1(draft, csvs, star_ids, ee_cache)
    b2 = run_b2(rows)

    pc = b2.get("partial_deficit_peak_given_r50_G_lt_10_11", float("nan"))
    b1_valid = b1.get("b1_valid", False)
    b2_nl = b2.get("b2_nl_pre_registered", False)

    if b2_nl and (not b1_valid or "does not explain" in b1.get("b1_verdict", "")):
        outcome = "B-nl"
    elif b1_valid and "sky contributes" in b1.get("b1_verdict", ""):
        outcome = "B-sky" if not b2_nl else "B-both"
    elif b2_nl and b1_valid:
        outcome = "B-both"
    elif not b2_nl and (not b1_valid or "does not explain" in b1.get("b1_verdict", "")):
        outcome = "B-open"
    else:
        outcome = "B-open"

    out = {"outcome": outcome, "B1": b1, "B2": b2}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    print("Outcome", outcome)
    print("B1 before", b1["slope_before_cal"].get("slope"), "after", b1["slope_after_aligned"].get("slope"))
    print("B2 partial peak|r50", pc)


if __name__ == "__main__":
    main()
