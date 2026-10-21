#!/usr/bin/env python3
"""Step 1h: diagnose magnitude dependence of delta_ap (Step 1g proxies).

Diagnostic only - no re-measurement, no config change.

Usage:
  python dev/tools/closure_step1h_diagnose_magnitude_dependence.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --step1b-json tmp/closure_step1b_results.json \\
    --step1g-json tmp/closure_step1g_results.json \\
    --cache tmp/closure_step1f_ee_cache.npz \\
    --out tmp/closure_step1h_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from closure_step1c_differential_aperture import PROXY_R_AP, _comp_subsets, _delta_ap_mmag
from closure_step1b_differential_aperture import _ee_at_radius, _load_all_proc
from closure_step1e_differential_aperture import _delta_ap_series, _robust_delta_stats

R_AP = PROXY_R_AP
NEIGHBOR_MAX_PX = 8.0
DG_MAX = 5.0


def _moffat_ee(r: float, r50: float, beta: float = 3.0) -> float:
    """EE(r) / EE(12 px) using the fixture Moffat parameterisation."""
    alpha = r50 / math.sqrt(2.0 ** (1.0 / (beta - 1.0)) - 1.0)
    ee_r = 1.0 - (1.0 + (r / alpha) ** 2) ** (1.0 - beta)
    ee_12 = 1.0 - (1.0 + (12.0 / alpha) ** 2) ** (1.0 - beta)
    return ee_r / ee_12 if ee_12 > 0 else float("nan")


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson correlation of x and y after removing linear dependence on z."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 5:
        return float("nan")
    xv, yv, zv = x[m], y[m], z[m]
    zx = np.polyval(np.polyfit(zv, xv, 1), zv)
    zy = np.polyval(np.polyfit(zv, yv, 1), zv)
    rx, ry = xv - zx, yv - zy
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _load_frame_meta(lights: Path, frame_names: list[str]) -> dict[str, dict[str, float]]:
    meta: dict[str, dict[str, float]] = {}
    for fn in frame_names:
        proc = lights / fn
        if not proc.is_file():
            continue
        df = pd.read_csv(proc, nrows=1)
        row = df.iloc[0]
        meta[fn] = {
            "airmass": float(row.get("airmass", float("nan"))),
            "sky_adu_per_px_annulus": float(row.get("sky_adu_per_px_annulus", float("nan"))),
        }
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--step1b-json", type=Path, default=Path("tmp/closure_step1b_results.json"))
    ap.add_argument("--step1g-json", type=Path, default=Path("tmp/closure_step1g_results.json"))
    ap.add_argument("--cache", type=Path, default=Path("tmp/closure_step1f_ee_cache.npz"))
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1h_diagnostics.json"))
    args = ap.parse_args()

    draft = args.draft.resolve()
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    csvs, catalog_full, n_frames = _load_all_proc(lights)

    with args.step1b_json.open(encoding="utf-8") as f:
        s1b = json.load(f)
    with args.step1g_json.open(encoding="utf-8") as f:
        s1g = json.load(f)

    star_ids: list[str] = s1b["part_a"]["star_ids"]
    proxies: list[str] = s1g["proxies"]
    catalog = catalog_full[catalog_full["catalog_id"].isin(star_ids)].copy()
    comp_subs = _comp_subsets(catalog, star_ids, exclude=set(proxies))
    clist_g89 = comp_subs["G8_9"]

    npz = np.load(args.cache, allow_pickle=True)
    ee_cache = npz["ee_cache"].item()
    aperture_by_frame = npz["aperture_by_frame"].item()
    frame_names = list(npz["frame_names"])
    r50_series = list(npz["r50_series"])
    sky_med = list(npz["sky_med"])
    vy_series = list(npz["vy_series"])

    drops_by_frame = s1g.get("drops", {}).get("by_frame", {})
    admissible_count = [
        float(drops_by_frame.get(fn, {}).get("kept", float("nan"))) for fn in frame_names
    ] if drops_by_frame else [float("nan")] * n_frames
    adm_arr = np.array(admissible_count[:n_frames], dtype=np.float64)
    frame_idx_arr = np.arange(n_frames, dtype=np.float64)
    frame_meta = _load_frame_meta(lights, frame_names)
    airmass_arr = np.array([frame_meta.get(fn, {}).get("airmass", float("nan")) for fn in frame_names])

    r50_arr = np.array(r50_series, dtype=np.float64)
    sky_arr = np.array(sky_med, dtype=np.float64)

    # positions from median across frames
    positions: dict[str, tuple[float, float, float]] = {}
    for cid in star_ids:
        xs, ys, gs = [], [], []
        for proc in csvs:
            df = pd.read_csv(proc, dtype={"catalog_id": str})
            row = df.loc[df["catalog_id"] == cid]
            if row.empty:
                continue
            xs.append(float(row.iloc[0]["x"]))
            ys.append(float(row.iloc[0]["y"]))
            if cid in catalog["catalog_id"].values:
                g = catalog.loc[catalog["catalog_id"] == cid, "phot_g"].iloc[0]
                gs.append(float(g))
        if xs:
            positions[cid] = (float(np.median(xs)), float(np.median(ys)), float(np.median(gs)) if gs else float("nan"))

    out: dict[str, Any] = {"proxies": proxies, "n_frames": n_frames, "D1": {}, "D2": {}, "D3": {}, "D4": {}, "D5": {}}

    # Build per-proxy delta_ap and EE series (G8_9)
    delta_series: dict[str, np.ndarray] = {}
    ee_t_series: dict[str, np.ndarray] = {}
    ee_c_med_series: dict[str, np.ndarray] = {}

    # fixed denominator: median EE per comp star over frames at its aperture
    comp_fixed_ee: dict[str, float] = {}
    for cid in clist_g89:
        ees = []
        for fi in range(n_frames):
            if cid not in ee_cache[fi]:
                continue
            rap = aperture_by_frame[fi].get(cid, float("nan"))
            ee = _ee_at_radius(ee_cache[fi][cid]["radii"], ee_cache[fi][cid]["ee"], rap)
            if math.isfinite(ee):
                ees.append(ee)
        comp_fixed_ee[cid] = float(np.median(ees)) if ees else float("nan")

    denom_med_per_frame = []
    denom_std_per_frame = []
    for fi in range(n_frames):
        eec = []
        for cid in clist_g89:
            if cid not in ee_cache[fi]:
                continue
            rap = aperture_by_frame[fi].get(cid, float("nan"))
            eec.append(_ee_at_radius(ee_cache[fi][cid]["radii"], ee_cache[fi][cid]["ee"], rap))
        denom_med_per_frame.append(float(np.median(eec)) if eec else float("nan"))
        denom_std_per_frame.append(float(np.std(eec)) if len(eec) > 1 else float("nan"))
    denom_med_arr = np.array(denom_med_per_frame)
    denom_std_arr = np.array(denom_std_per_frame)

    for pid in proxies:
        d = _delta_ap_series(
            pid, clist_g89, ee_cache=ee_cache, aperture_by_frame=aperture_by_frame,
            n_frames=n_frames, target_r_override=R_AP,
        )
        delta_series[pid] = d
        et, ec = [], []
        for fi in range(n_frames):
            if pid not in ee_cache[fi]:
                et.append(float("nan"))
                ec.append(float("nan"))
                continue
            ee_t = _ee_at_radius(ee_cache[fi][pid]["radii"], ee_cache[fi][pid]["ee"], R_AP)
            eec = []
            for cid in clist_g89:
                if cid not in ee_cache[fi]:
                    continue
                rap = aperture_by_frame[fi].get(cid, float("nan"))
                eec.append(_ee_at_radius(ee_cache[fi][cid]["radii"], ee_cache[fi][cid]["ee"], rap))
            et.append(ee_t)
            ec.append(float(np.median(eec)) if eec else float("nan"))
        ee_t_series[pid] = np.array(et)
        ee_c_med_series[pid] = np.array(ec)

    def _frame_char(fi: int) -> dict[str, float]:
        fn = frame_names[fi]
        return {
            "frame": fn,
            "r50_px": float(r50_arr[fi]),
            "sky_adu_per_px": float(sky_arr[fi]),
            "vy_fwhm": float(vy_series[fi]) if fi < len(vy_series) else float("nan"),
            "airmass": float(airmass_arr[fi]),
            "admissible_stars_kept": float(adm_arr[fi]),
        }

    # Step 1g reproduction check (H3)
    s1g_repro: dict[str, Any] = {}
    for pid in proxies:
        d = delta_series[pid]
        st = _robust_delta_stats(d, r50_arr, sky_arr, frame_names, excluded_frames=set())
        g89 = s1g.get("M1", {}).get("proxies", {}).get(pid, {}).get("G8_9", {})
        s1g_repro[pid] = {
            "recomputed_p95_p5_mmag": st["range_p95_p5_mmag"],
            "step1g_p95_p5_mmag": g89.get("range_p95_p5_mmag"),
            "delta_mmag": (
                st["range_p95_p5_mmag"] - g89["range_p95_p5_mmag"]
                if g89.get("range_p95_p5_mmag") is not None else float("nan")
            ),
        }
    out["H3_step1g_reproduction"] = s1g_repro

    # D1
    for pid in proxies:
        d = delta_series[pid]
        v = d[np.isfinite(d)]
        p5, p95 = np.percentile(v, 5), np.percentile(v, 95)
        p16, p84 = np.percentile(v, 16), np.percentile(v, 84)
        mad = float(np.median(np.abs(v - np.median(v))))
        # trim 5 most extreme from median
        resid = np.abs(v - np.median(v))
        keep = np.argsort(resid)[:-5] if v.size > 10 else np.arange(v.size)
        v_trim = v[keep]
        imax = int(np.nanargmax(d))
        imin = int(np.nanargmin(d))
        out["D1"][pid] = {
            "phot_g": positions.get(pid, (0, 0, float("nan")))[2],
            "n_frames_finite": int(v.size),
            "p95_p5_mmag": float(p95 - p5),
            "p84_p16_mmag": float(p84 - p16),
            "mad_mmag": mad,
            "p95_p5_after_trim5_extreme_mmag": float(np.percentile(v_trim, 95) - np.percentile(v_trim, 5)) if v_trim.size >= 5 else float("nan"),
            "n_beyond_p95": int(np.sum(v > p95)),
            "n_below_p5": int(np.sum(v < p5)),
            "max_frame": _frame_char(imax),
            "min_frame": _frame_char(imin),
            "delta_ap_at_max_mmag": float(d[imax]),
            "delta_ap_at_min_mmag": float(d[imin]),
        }

    # D2
    r50_lo, r50_hi = float(np.nanmin(r50_arr)), float(np.nanmax(r50_arr))
    ee_phys_lo = _moffat_ee(R_AP, r50_lo)
    ee_phys_hi = _moffat_ee(R_AP, r50_hi)
    out["D2"]["physics_ref"] = {
        "r50_min_px": r50_lo,
        "r50_max_px": r50_hi,
        "EE_1.916_at_r50_min": ee_phys_lo,
        "EE_1.916_at_r50_max": ee_phys_hi,
        "delta_EE_span": ee_phys_hi - ee_phys_lo,
    }
    ee_c_shared = np.array(denom_med_per_frame)
    for pid in proxies:
        et = ee_t_series[pid]
        m = np.isfinite(et) & np.isfinite(r50_arr)
        et_v = et[m]
        out["D2"][pid] = {
            "EE_target_p95_p5": float(np.percentile(et_v, 95) - np.percentile(et_v, 5)) if et_v.size else float("nan"),
            "EE_target_p84_p16": float(np.percentile(et_v, 84) - np.percentile(et_v, 16)) if et_v.size else float("nan"),
            "EE_target_min": float(np.min(et_v)) if et_v.size else float("nan"),
            "EE_target_max": float(np.max(et_v)) if et_v.size else float("nan"),
            "pearson_EE_target_vs_r50": float(np.corrcoef(r50_arr[m], et[m])[0, 1]) if m.sum() >= 5 else float("nan"),
            "median_comp_EE_p95_p5": float(
                np.percentile(ee_c_shared[np.isfinite(ee_c_shared)], 95)
                - np.percentile(ee_c_shared[np.isfinite(ee_c_shared)], 5)
            ),
            "delta_ap_p95_p5_mmag": out["D1"][pid]["p95_p5_mmag"],
        }

    # D3 neighbours within 8 px
    def neighbours(cid: str) -> list[dict[str, Any]]:
        if cid not in positions:
            return []
        x0, y0, g0 = positions[cid]
        nbs = []
        for oid, (x, y, g) in positions.items():
            if oid == cid:
                continue
            if not math.isfinite(g0) or not math.isfinite(g):
                continue
            if abs(g - g0) >= DG_MAX:
                continue
            sep = math.hypot(x - x0, y - y0)
            if sep <= NEIGHBOR_MAX_PX:
                nbs.append({"id": oid, "sep_px": sep, "dG": g - g0, "phot_g": g})
        return sorted(nbs, key=lambda t: t["sep_px"])

    # flux proxy from catalog G (relative)
    def rel_flux(g: float) -> float:
        return 10.0 ** (-0.4 * (g - 8.0))

    contam: dict[str, Any] = {}
    for pid in proxies:
        nbs = neighbours(pid)
        g0 = positions[pid][2]
        f0 = rel_flux(g0)
        frac_sum = 0.0
        for nb in nbs:
            f1 = rel_flux(nb["phot_g"])
            # rough flux fraction at 1.916 px if neighbour at sep contributes uniformly
            sep = nb["sep_px"]
            if sep < R_AP:
                frac_sum += f1 / (f0 + f1)
        contam[pid] = {
            "phot_g": g0,
            "n_neighbours_8px_dG5": len(nbs),
            "neighbours": nbs,
            "estimated_contam_flux_fraction": frac_sum,
        }
    out["D3"]["per_proxy"] = contam
    out["D3"]["per_comp_G89"] = {cid: {"phot_g": positions[cid][2], "neighbours": neighbours(cid)} for cid in clist_g89}

    # correlate range with contam
    ranges = [out["D1"][p]["p95_p5_mmag"] for p in proxies]
    fracs = [contam[p]["estimated_contam_flux_fraction"] for p in proxies]
    gs = [contam[p]["phot_g"] for p in proxies]
    out["D3"]["correlation"] = {
        "pearson_range_vs_contam_frac": float(np.corrcoef(ranges, fracs)[0, 1]) if len(ranges) >= 3 else float("nan"),
        "pearson_range_vs_G": float(np.corrcoef(ranges, gs)[0, 1]) if len(ranges) >= 3 else float("nan"),
        "spearman_range_vs_G": float(stats.spearmanr(ranges, gs).statistic) if len(ranges) >= 3 else float("nan"),
    }

    # D4
    mat = np.column_stack([delta_series[p] for p in proxies])
    m_all = np.all(np.isfinite(mat), axis=1)
    mat_c = mat[m_all]
    cross = np.corrcoef(mat_c.T) if mat_c.shape[0] >= 5 else None

    d4_proxy: dict[str, Any] = {}
    for pid in proxies:
        d = delta_series[pid]
        st = _robust_delta_stats(d, r50_arr, sky_arr, frame_names, excluded_frames=set())
        m = np.isfinite(d)
        slope, intercept = np.polyfit(r50_arr[m], d[m], 1)
        span = float(np.nanmax(r50_arr) - np.nanmin(r50_arr))
        resid = d[m] - (slope * r50_arr[m] + intercept)
        d4_proxy[pid] = {
            "slope_mmag_per_r50": float(slope),
            "slope_times_span_mmag": float(slope * span),
            "p95_p5_mmag": out["D1"][pid]["p95_p5_mmag"],
            "slope_equals_p95_p5": abs(float(slope * span) - out["D1"][pid]["p95_p5_mmag"]) < 0.5,
            "two_point_min_max_r50_mmag": st["two_point_min_max_r50_diff_mmag"],
            "pearson_r50": st["pearson_r50"],
            "pearson_sky": st["pearson_sky"],
            "partial_corr_r50_given_denom": _partial_corr(d, r50_arr, denom_med_arr),
            "partial_corr_sky": _partial_corr(d, sky_arr, r50_arr),
            "partial_corr_airmass": _partial_corr(d, airmass_arr, r50_arr),
            "partial_corr_frame_index": _partial_corr(d, frame_idx_arr, r50_arr),
            "partial_corr_admissible_count": _partial_corr(d, adm_arr, r50_arr),
            "pearson_vs_denom_ee": float(np.corrcoef(d[m], denom_med_arr[m])[0, 1]) if m.sum() >= 5 else float("nan"),
            "fit_residual_rms_mmag": st["slope_fit_residual_rms_mmag"],
            "step1g_report_pearson_r50": s1g.get("M1", {}).get("proxies", {}).get(pid, {}).get("G8_9", {}).get("pearson_r50"),
        }
    out["D4"]["per_proxy"] = d4_proxy
    out["D4"]["n_frames_all_proxies_finite"] = int(m_all.sum())
    if cross is not None:
        out["D4"]["cross_proxy_correlation"] = {
            "matrix": cross.tolist(),
            "labels": proxies,
            "mean_off_diagonal": float(np.mean(cross[np.triu_indices(len(proxies), k=1)])),
            "min_off_diagonal": float(np.min(cross[np.triu_indices(len(proxies), k=1)])),
        }
    out["D4"]["explanation"] = (
        "slope_times_span approximates p95-p5 only when delta_ap is nearly linear in r50_frame "
        "with small residual scatter (one proxy here). Pearson vs r50 is NOT identical across "
        "proxies: fainter proxies show weaker r50 correlation and larger fit residuals. "
        "The Step 1g markdown table incorrectly listed slope*span=p95-p5 and Pearson=0.54 for "
        "four proxies; JSON on disk shows otherwise (see step1g_report_pearson_r50 fields)."
    )

    # D5 fixed denominator
    d5: dict[str, Any] = {
        "denom_per_frame_median_ee_std_mean": float(np.nanmean(denom_std_arr)),
        "denom_median_ee_p95_p5": float(
            np.percentile(denom_med_arr[np.isfinite(denom_med_arr)], 95)
            - np.percentile(denom_med_arr[np.isfinite(denom_med_arr)], 5)
        ),
        "denom_median_ee_pearson_r50": float(
            np.corrcoef(r50_arr[np.isfinite(denom_med_arr)], denom_med_arr[np.isfinite(denom_med_arr)])[0, 1]
        ) if np.isfinite(denom_med_arr).sum() >= 5 else float("nan"),
        "comp_fixed_ee_G89": {cid: comp_fixed_ee[cid] for cid in clist_g89},
    }
    for pid in proxies:
        d_fix = []
        d_orig = []
        for fi in range(n_frames):
            if pid not in ee_cache[fi]:
                continue
            ee_t = _ee_at_radius(ee_cache[fi][pid]["radii"], ee_cache[fi][pid]["ee"], R_AP)
            fixed_d = np.median([comp_fixed_ee[c] for c in clist_g89 if math.isfinite(comp_fixed_ee.get(c, float("nan")))])
            d_fix.append(_delta_ap_mmag(ee_t, [fixed_d]))
            d_orig.append(delta_series[pid][fi])
        d_fix = np.array(d_fix)
        d_orig = np.array(d_orig)
        v = d_orig[np.isfinite(d_orig)]
        vf = d_fix[np.isfinite(d_fix)]
        d5[pid] = {
            "p95_p5_original_mmag": float(np.percentile(v, 95) - np.percentile(v, 5)),
            "p95_p5_fixed_denom_mmag": float(np.percentile(vf, 95) - np.percentile(vf, 5)),
            "fraction_of_range_in_numerator": float(
                (np.percentile(vf, 95) - np.percentile(vf, 5))
                / max(np.percentile(v, 95) - np.percentile(v, 5), 1e-6)
            ),
        }
    out["D5"] = d5

    # Outcome hint
    ee_spread_by_g = [out["D2"][p]["EE_target_p95_p5"] for p in proxies]
    out["summary"] = {
        "EE_target_spread_increases_with_G": ee_spread_by_g,
        "denom_ee_frame_std_mean": float(np.nanmean(denom_std_arr)),
        "denom_ee_p95_p5": float(np.percentile(denom_med_arr[np.isfinite(denom_med_arr)], 95) - np.percentile(denom_med_arr[np.isfinite(denom_med_arr)], 5)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
