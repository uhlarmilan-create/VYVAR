#!/usr/bin/env python3
"""Batch B-revised M1: D5-2 mechanism on production proc CSV columns only."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closure_step1b_differential_aperture import _load_all_proc


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 8:
        return float("nan")
    x, y, z = x[m], y[m], z[m]
    rxy = float(np.corrcoef(x, y)[0, 1])
    rxz = float(np.corrcoef(x, z)[0, 1])
    ryz = float(np.corrcoef(y, z)[0, 1])
    den = math.sqrt(max(1e-12, (1 - rxz**2) * (1 - ryz**2)))
    return float((rxy - rxz * ryz) / den)


def main() -> None:
    draft = Path("Archive/Drafts/draft_000435_snapshot_skysurface_20260716")
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    csvs, _, _ = _load_all_proc(lights)
    with Path("tmp/closure_step1b_results.json").open(encoding="utf-8") as f:
        sids = json.load(f)["part_a"]["star_ids"]
    npz = np.load("tmp/closure_step1f_ee_cache.npz", allow_pickle=True)
    ee = npz["ee_cache"].item()
    r50s = np.array(npz["r50_series"], dtype=np.float64)

    rows = []
    for fi, proc in enumerate(csvs):
        df = pd.read_csv(proc, dtype={"catalog_id": str})
        r50 = float(r50s[fi])
        for sid in sids:
            if sid not in ee.get(fi, {}):
                continue
            pr = df.loc[df["catalog_id"] == sid]
            if pr.empty:
                continue
            pr = pr.iloc[0]
            if str(pr.get("photometry_ok", "")).lower() not in ("true", "1", "yes"):
                continue
            if str(pr.get("is_usable", "")).lower() not in ("true", "1", "yes"):
                continue
            flux = float(pr["flux"])
            if not (math.isfinite(flux) and flux > 0):
                continue
            rows.append(
                {
                    "mag": float(pr["mag"]),
                    "logF": math.log10(flux),
                    "logFL": math.log10(float(pr["flux_large"])) if float(pr["flux_large"]) > 0 else float("nan"),
                    "peak": float(pr["peak_max_adu"]),
                    "r50": r50,
                    "sat_lim": float(pr.get("saturate_limit_adu_85pct", 55705)),
                    "is_sat": bool(pr.get("is_saturated", False)),
                    "likely_sat": bool(pr.get("likely_saturated", False)),
                    "sid": sid,
                }
            )

    d = pd.DataFrame(rows)
    out: dict = {"n_star_frames": len(d)}

    out["bins_1mag"] = []
    for lo in range(8, 13):
        sub = d[(d["mag"] >= lo) & (d["mag"] < lo + 1)]
        entry = {"G_lo": lo, "G_hi": lo + 1, "n": int(len(sub))}
        if len(sub) >= 5 and sub["mag"].std() > 1e-9:
            fit = stats.linregress(sub["mag"], sub["logF"])
            entry["slope_flux"] = float(fit.slope)
            entry["slope_se"] = float(fit.stderr)
        out["bins_1mag"].append(entry)

    out["G8_9_bin"] = {
        "slope_flux": out["bins_1mag"][0].get("slope_flux"),
        "slope_deficit_vs_expected_0p4": 0.4 + out["bins_1mag"][0].get("slope_flux", 0),
    }

    # 0.5 mag bins
    bins_half = []
    lo = math.floor(d["mag"].min() * 2) / 2
    hi_max = math.ceil(d["mag"].max() * 2) / 2
    while lo < hi_max:
        hi = lo + 0.5
        sub = d[(d["mag"] >= lo) & (d["mag"] < hi)]
        entry = {"G_lo": lo, "G_hi": hi, "n": int(len(sub))}
        if len(sub) >= 5 and sub["mag"].std() > 1e-9:
            fit = stats.linregress(sub["mag"], sub["logF"])
            entry["slope_flux"] = float(fit.slope)
            entry["slope_se"] = float(fit.stderr)
        bins_half.append(entry)
        lo = hi
    out["bins_0p5mag"] = bins_half

    ref = d[(d["mag"] >= 10.0) & (d["mag"] < 13.0)]
    fit_ref = stats.linregress(ref["mag"], ref["logF"])
    out["ref_G10_13"] = {
        "n": int(len(ref)),
        "slope": float(fit_ref.slope),
        "slope_se": float(fit_ref.stderr),
    }

    ic04 = float(np.median(ref["logF"] + 0.4 * ref["mag"]))
    d["pred04"] = ic04 - 0.4 * d["mag"]
    d["deficit"] = d["pred04"] - d["logF"]

    bright = d[d["mag"] < 9.0].copy()
    out["G_lt_9"] = {
        "n": int(len(bright)),
        "mean_deficit_dex": float(bright["deficit"].mean()),
        "pearson_deficit_peak": float(bright["deficit"].corr(bright["peak"])),
        "pearson_deficit_r50": float(bright["deficit"].corr(bright["r50"])),
        "partial_deficit_peak_given_r50": partial_corr(
            bright["deficit"].to_numpy(), bright["peak"].to_numpy(), bright["r50"].to_numpy()
        ),
    }

    peak_bins = []
    edges = [0, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 56000]
    for i in range(len(edges) - 1):
        m = (bright["peak"] >= edges[i]) & (bright["peak"] < edges[i + 1])
        if m.sum() >= 3:
            peak_bins.append(
                {
                    "peak_lo": edges[i],
                    "peak_hi": edges[i + 1],
                    "n": int(m.sum()),
                    "mean_deficit_dex": float(bright.loc[m, "deficit"].mean()),
                    "mean_peak_adu": float(bright.loc[m, "peak"].mean()),
                }
            )
    out["deficit_by_peak_G_lt_9"] = peak_bins

    stars = []
    for sid, grp in bright.groupby("sid"):
        stars.append(
            {
                "star_id": sid,
                "G": float(grp["mag"].iloc[0]),
                "peak_max": float(grp["peak"].max()),
                "peak_median": float(grp["peak"].median()),
                "pct_full_well": float(100 * grp["peak"].max() / grp["sat_lim"].iloc[0]),
                "n_frames_gt_70pct": int((grp["peak"] > 0.7 * grp["sat_lim"]).sum()),
                "n_frames": int(len(grp)),
                "is_saturated_any": bool(grp["is_sat"].any()),
                "likely_saturated_any": bool(grp["likely_sat"].any()),
            }
        )
    out["G_lt_9_stars"] = sorted(stars, key=lambda t: t["G"])

    # knee: first bin where mean deficit > 0.05 dex above lowest bin baseline
    if peak_bins:
        base = min(b["mean_deficit_dex"] for b in peak_bins)
        knee = None
        for b in peak_bins:
            if b["mean_deficit_dex"] - base > 0.05:
                knee = b["peak_lo"]
                break
        out["knee_adu_estimate"] = knee

    out_path = Path("tmp/closure_batch_b_revised_m1.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
