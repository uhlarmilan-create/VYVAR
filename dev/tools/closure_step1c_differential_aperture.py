#!/usr/bin/env python3
"""Closure Step 1c: repair delta_ap computation + self-tests T1-T4.

Reuses Step 1b star set and COG functions; does NOT re-run A.1/A.2 selection or A.3 fits
unless --rebuild-cache is passed without an existing cache.

Usage:
  python dev/tools/closure_step1c_differential_aperture.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --step1b-json tmp/closure_step1b_results.json \\
    --out tmp/closure_step1c_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow import of sibling harness when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats

# Reuse Step 1b COG / selection helpers (Part A stands)
from closure_step1b_differential_aperture import (  # noqa: E402
    COG_DR,
    COG_RMAX,
    FOCUS_ID,
    TABLE_FWHM,
    _curve_of_growth,
    _ee_at_radius,
    _growth_curve_flat_ok,
    _load_all_proc,
    _lookup_table_r,
    _mag_for_aperture,
    _pick_fixed_stars,
    _r_at_ee,
    _snr_table_for_frame,
    _arcsec_per_px,
)

PROXY_R_AP = 1.916
T4_RATIO_LO = 5.0
T4_RATIO_HI = 15.0
GATE_MM = 10.0  # 10 mmag differential gate (Step 1b/1c pre-register)


def _delta_ap_mmag(ee_target: float, ee_comps: list[float]) -> float:
    """Differential aperture metric in millimagnitudes (not magnitudes)."""
    med_c = float(np.median(ee_comps))
    if ee_target <= 0 or med_c <= 0:
        return float("nan")
    return -2.5 * math.log10(ee_target / med_c) * 1000.0


def _monotone_ee(radii: np.ndarray, ee: np.ndarray) -> np.ndarray:
    """Cumulative maximum + clip to [0, 1] for admissible interpolation."""
    ee = np.asarray(ee, dtype=np.float64)
    ee = np.maximum.accumulate(ee)
    return np.clip(ee, 0.0, 1.0)


def _moffat_ee(r: float, r50: float, beta: float = 3.0) -> float:
    """Enclosed fraction at radius r for Moffat with r50 = radius at EE=0.5."""
    # Solve gamma from r50: EE(r50)=0.5 -> (1+(r50/g)^2)^(-beta)=0.5
    g = r50 / math.sqrt(2.0 ** (1.0 / beta) - 1.0)
    ee_r = (1.0 + (r / g) ** 2) ** (-beta)
    ee_inf = 1.0  # normalize at infinity; use ee(r)/ee(12) approx
    ee_12 = (1.0 + (12.0 / g) ** 2) ** (-beta)
    return float(np.clip(ee_r / ee_12, 0.0, 1.0))


def _build_ee_cache(
    draft: Path,
    star_ids: list[str],
    *,
    csvs: list[Path],
    lights: Path,
) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[int, dict[str, float]], list[str], list[float], list[float]]:
    radii = np.arange(COG_DR, COG_RMAX + COG_DR / 2, COG_DR)
    ee_cache: dict[int, dict[str, dict[str, Any]]] = {}
    aperture_by_frame: dict[int, dict[str, float]] = {}
    frame_names: list[str] = []
    r50_series: list[float] = []
    sky_med: list[float] = []
    vy_series: list[float] = []

    for fi, proc in enumerate(csvs):
        fits_path = lights / proc.name.replace(".csv", ".fits")
        df = pd.read_csv(proc, dtype={"catalog_id": str}).set_index("catalog_id")
        with fits.open(fits_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float64)
            vy = float(hdul[0].header.get("VY_FWHM", float("nan")))
        frame_names.append(proc.name)
        vy_series.append(vy)
        ee_cache[fi] = {}
        aperture_by_frame[fi] = {}
        r50_stars: list[float] = []
        skies: list[float] = []
        for cid in star_ids:
            if cid not in df.index:
                continue
            row = df.loc[cid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            x, y = float(row["x"]), float(row["y"])
            aperture_by_frame[fi][cid] = float(row.get("aperture_r_px", float("nan")))
            if "sky_adu_per_px_annulus" in row.index:
                sv = float(row["sky_adu_per_px_annulus"])
                if math.isfinite(sv):
                    skies.append(sv)
            cog = _curve_of_growth(data, x, y, radii=radii)
            if cog is None:
                continue
            ee_mono = _monotone_ee(cog["radii"], cog["ee"])
            qc_ok = _growth_curve_flat_ok(ee_mono, cog["radii"])
            ee_cache[fi][cid] = {
                "radii": cog["radii"],
                "ee": ee_mono,
                "ee_raw": cog["ee"],
                "qc_ok": qc_ok,
                "r50": _r_at_ee(cog["radii"], ee_mono, 0.5),
            }
            if math.isfinite(ee_cache[fi][cid]["r50"]):
                r50_stars.append(ee_cache[fi][cid]["r50"])
        r50_series.append(float(np.median(r50_stars)) if r50_stars else float("nan"))
        sky_med.append(float(np.median(skies)) if skies else float("nan"))
    return ee_cache, aperture_by_frame, frame_names, r50_series, sky_med, vy_series


def _comp_subsets(catalog: pd.DataFrame, ids: list[str], *, exclude: set[str]) -> dict[str, list[str]]:
    sub: dict[str, list[str]] = {"G8_9": [], "G9_11": [], "G_gt_11": []}
    for cid in ids:
        if cid in exclude:
            continue
        row = catalog.loc[catalog["catalog_id"] == cid]
        if row.empty:
            continue
        g = float(row.iloc[0]["phot_g"])
        if not math.isfinite(g):
            continue
        if 8.0 <= g < 9.0:
            sub["G8_9"].append(cid)
        elif 9.0 <= g < 11.0:
            sub["G9_11"].append(cid)
        elif g >= 11.0:
            sub["G_gt_11"].append(cid)
    return sub


def _delta_ap_series(
    target_id: str,
    comp_list: list[str],
    *,
    ee_cache: dict[int, dict[str, dict[str, Any]]],
    aperture_by_frame: dict[int, dict[str, float]],
    n_frames: int,
    target_r_override: float | None = None,
) -> np.ndarray:
    deltas = []
    for fi in range(n_frames):
        if target_id not in ee_cache[fi]:
            deltas.append(float("nan"))
            continue
        rap_t = PROXY_R_AP if target_r_override is not None else aperture_by_frame[fi].get(target_id, float("nan"))
        if target_r_override is not None:
            rap_t = target_r_override
        ee_t = _ee_at_radius(
            ee_cache[fi][target_id]["radii"],
            ee_cache[fi][target_id]["ee"],
            rap_t,
        )
        ee_c: list[float] = []
        for cid in comp_list:
            if cid not in ee_cache[fi]:
                continue
            rap_c = aperture_by_frame[fi].get(cid, float("nan"))
            ee_c.append(
                _ee_at_radius(
                    ee_cache[fi][cid]["radii"],
                    ee_cache[fi][cid]["ee"],
                    rap_c,
                )
            )
        if not ee_c or not math.isfinite(ee_t) or ee_t <= 0:
            deltas.append(float("nan"))
            continue
        med_c = float(np.median(ee_c))
        if med_c <= 0:
            deltas.append(float("nan"))
            continue
        deltas.append(_delta_ap_mmag(ee_t, ee_c))
    return np.array(deltas, dtype=np.float64)


def _summarize_delta(
    d: np.ndarray,
    r50_arr: np.ndarray,
    sky_arr: np.ndarray,
    frame_names: list[str],
) -> dict[str, Any]:
    valid = d[np.isfinite(d)]
    idx_best = int(np.nanargmin(r50_arr))
    idx_worst = int(np.nanargmax(r50_arr))
    m_r = np.isfinite(d) & np.isfinite(r50_arr)
    m_s = np.isfinite(d) & np.isfinite(sky_arr)
    slope = float(np.polyfit(r50_arr[m_r], d[m_r], 1)[0]) if m_r.sum() >= 5 else float("nan")
    return {
        "range_best_worst_mmag": float(abs(d[idx_worst] - d[idx_best]))
        if np.isfinite(d[idx_worst]) and np.isfinite(d[idx_best])
        else float("nan"),
        "min_r50_frame": frame_names[idx_best],
        "max_r50_frame": frame_names[idx_worst],
        "slope_mmag_per_r50": slope,
        "pearson_r50": float(np.corrcoef(r50_arr[m_r], d[m_r])[0, 1]) if m_r.sum() >= 5 else float("nan"),
        "spearman_r50": float(stats.spearmanr(r50_arr[m_r], d[m_r]).statistic) if m_r.sum() >= 5 else float("nan"),
        "pearson_sky": float(np.corrcoef(sky_arr[m_s], d[m_s])[0, 1]) if m_s.sum() >= 5 else float("nan"),
        "median_mmag": float(np.median(valid)) if valid.size else float("nan"),
    }


def _delta_ap_frozen_k(
    target_id: str,
    comp_list: list[str],
    *,
    ee_cache: dict[int, dict[str, dict[str, Any]]],
    aperture_by_frame: dict[int, dict[str, float]],
    r50_arr: np.ndarray,
    scale_arr: np.ndarray,
    n_frames: int,
    k_ref: float,
    target_r_override: float | None = None,
) -> np.ndarray:
    """r_i(f) = k_i * scale(f); EE from star i's own curve."""
    k_i: dict[str, float] = {}
    for cid in [target_id, *comp_list]:
        # k_i = r_ap / scale_ref where scale_ref = median r50
        rap_vals = [aperture_by_frame[fi].get(cid, float("nan")) for fi in range(n_frames)]
        rap_med = float(np.nanmedian(rap_vals))
        k_i[cid] = rap_med / k_ref if k_ref > 0 and math.isfinite(rap_med) else float("nan")
    if target_r_override is not None:
        k_i[target_id] = target_r_override / k_ref

    deltas = []
    for fi in range(n_frames):
        sc = scale_arr[fi]
        if not math.isfinite(sc) or target_id not in ee_cache[fi]:
            deltas.append(float("nan"))
            continue
        rap_t = k_i[target_id] * sc
        ee_t = _ee_at_radius(ee_cache[fi][target_id]["radii"], ee_cache[fi][target_id]["ee"], rap_t)
        ee_c = []
        for cid in comp_list:
            if cid not in ee_cache[fi]:
                continue
            rap_c = k_i.get(cid, float("nan")) * sc
            ee_c.append(_ee_at_radius(ee_cache[fi][cid]["radii"], ee_cache[fi][cid]["ee"], rap_c))
        if not ee_c or not math.isfinite(ee_t) or ee_t <= 0:
            deltas.append(float("nan"))
            continue
        med_c = float(np.median(ee_c))
        if med_c <= 0:
            deltas.append(float("nan"))
            continue
        deltas.append(_delta_ap_mmag(ee_t, ee_c))
    return np.array(deltas, dtype=np.float64)


def _estimator_stats(r50_arr: np.ndarray, est: np.ndarray, name: str) -> dict[str, float]:
    m = np.isfinite(r50_arr) & np.isfinite(est)
    x, y = r50_arr[m], est[m]
    if x.size < 5:
        return {"name": name, "n": int(x.size)}
    slope = float(np.sum(x * y) / np.sum(x * x))
    resid = y - slope * x
    dr_x = (float(np.nanmax(x)) - float(np.nanmin(x))) / float(np.nanmedian(x))
    dr_y = (float(np.nanmax(y)) - float(np.nanmin(y))) / float(np.nanmedian(y))
    sp = float(stats.spearmanr(x, y).statistic)
    return {
        "name": name,
        "slope_origin": slope,
        "frac_scatter_over_slope": _mad(resid) / abs(slope) if slope else float("nan"),
        "dynamic_range_ratio": dr_y / dr_x if dr_x > 0 else float("nan"),
        "spearman": sp,
        "n": int(x.size),
    }


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))))


def _validate_reference_fixture() -> dict[str, Any]:
    """Gate: harness _delta_ap_mmag must match closure_a1_reference_fixture.py (L2 photutils)."""
    from closure_a1_reference_fixture import (  # noqa: PLC0415
        BETA,
        COG_RADII,
        R50_GRID,
        R_TARGET,
        SKY_ADU,
        SUBSETS,
        TOTAL_FLUX,
        TOL_MMAG,
        alpha_from_r50,
        build_expected,
        ee_at,
        ee_curve_photutils,
        render_moffat,
    )

    exp = build_expected()
    errors: dict[str, list[float]] = {k: [] for k in SUBSETS}
    for r50 in R50_GRID:
        a = alpha_from_r50(r50, BETA)
        img = render_moffat((161, 161), 80.37, 80.62, a, BETA, TOTAL_FLUX, SKY_ADU)
        ee = ee_curve_photutils(img, 80.37, 80.62)
        et = ee_at(COG_RADII, ee, R_TARGET)
        key = f"{r50:.2f}"
        for subset, radii in SUBSETS.items():
            ec = [ee_at(COG_RADII, ee, r) for r in radii]
            got = _delta_ap_mmag(et, ec)
            want = exp["table"][key][subset]
            errors[subset].append(abs(got - want))
    max_err = max(v for vals in errors.values() for v in vals)
    return {
        "pass": max_err <= TOL_MMAG,
        "max_abs_error_mmag": max_err,
        "expected_t4_ratio": exp["t4_ratio"],
        "expected_ranges": exp["range_over_span"],
        "per_subset_max_err": {k: max(v) for k, v in errors.items()},
    }


def run_self_tests(
    ee_cache: dict[int, dict[str, dict[str, Any]]],
    aperture_by_frame: dict[int, dict[str, float]],
    r50_arr: np.ndarray,
    star_ids: list[str],
    comp_subs: dict[str, list[str]],
    n_frames: int,
    catalog: pd.DataFrame,
) -> dict[str, Any]:
    tests: dict[str, Any] = {}
    radii = np.arange(COG_DR, COG_RMAX + COG_DR / 2, COG_DR)

    # T1 equal-radius null
    proxy = _pick_proxies(catalog, star_ids, ee_cache, n_frames)
    tid = proxy[0] if proxy else star_ids[0]
    clist = comp_subs["G8_9"][:3] or [s for s in star_ids if s != tid][:3]

    # single star self: must be exactly 0
    d_self = []
    for fi in range(n_frames):
        if tid not in ee_cache[fi]:
            d_self.append(float("nan"))
            continue
        ee = _ee_at_radius(ee_cache[fi][tid]["radii"], ee_cache[fi][tid]["ee"], PROXY_R_AP)
        d_self.append(_delta_ap_mmag(ee, [ee]))
    t1_self_max = float(np.nanmax(np.abs(np.array(d_self))))

    d_eq = _delta_ap_series(
        tid, clist, ee_cache=ee_cache, aperture_by_frame=aperture_by_frame, n_frames=n_frames,
        target_r_override=PROXY_R_AP,
    )
    # force comps to same radius
    ap_fixed = {fi: {cid: PROXY_R_AP for cid in clist} for fi in range(n_frames)}
    d_eq2 = _delta_ap_series(
        tid, clist, ee_cache=ee_cache, aperture_by_frame=ap_fixed, n_frames=n_frames,
        target_r_override=PROXY_R_AP,
    )
    tests["T1"] = {
        "self_compare_max_abs_mmag": t1_self_max,
        "self_pass": t1_self_max < 1e-12,
        "ensemble_equal_r_max_abs_mmag": float(np.nanmax(np.abs(d_eq2))),
        "ensemble_pass": float(np.nanmax(np.abs(d_eq2))) < 0.01,
        "ensemble_star_spread_mmag": float(np.nanmax(np.abs(d_eq))),
        "note": "non-zero ensemble residual measures star-to-star profile spread at equal r",
    }

    # T2 synthetic Moffat
    t2_errs = []
    scales = np.linspace(1.45, 1.98, 12)
    for si, sc in enumerate(scales):
        for label, clist_t2 in comp_subs.items():
            if not clist_t2:
                continue
            rap_t = PROXY_R_AP
            rap_cs = [float(np.nanmedian([aperture_by_frame[fi].get(c, float("nan")) for fi in range(n_frames)])) for c in clist_t2[:3]]
            ee_t = _moffat_ee(rap_t, sc)
            ee_cs = [_moffat_ee(r, sc) for r in rap_cs]
            analytic = _delta_ap_mmag(ee_t, ee_cs)
            # build synthetic cache one frame
            syn: dict[str, dict[str, Any]] = {}
            for cid in [tid, *clist_t2[:3]]:
                rap = rap_t if cid == tid else float(np.nanmedian([aperture_by_frame[fi].get(cid, float("nan")) for fi in range(n_frames)]))
                ee_curve = np.array([_moffat_ee(r, sc) for r in radii])
                syn[cid] = {"radii": radii, "ee": ee_curve, "qc_ok": True, "r50": sc}
            syn_cache = {0: syn}
            syn_ap = {0: {}}
            for cid in syn:
                syn_ap[0][cid] = PROXY_R_AP if cid == tid else float(np.nanmedian([aperture_by_frame[fi].get(cid, float("nan")) for fi in range(n_frames)]))
            rec = _delta_ap_series(tid, clist_t2[:3], ee_cache=syn_cache, aperture_by_frame=syn_ap, n_frames=1, target_r_override=PROXY_R_AP)
            err = abs(float(rec[0]) - analytic) if np.isfinite(rec[0]) else float("nan")
            t2_errs.append(err)
    tests["T2"] = {
        "max_abs_error_mmag": float(np.nanmax(t2_errs)),
        "pass": float(np.nanmax(t2_errs)) < 0.5,
        "n_cases": len(t2_errs),
    }

    # T3 proportional-scaling identity (synthetic curves - implementation gate)
    k_ref = float(np.nanmedian(r50_arr))
    syn_cache: dict[int, dict[str, dict[str, Any]]] = {}
    syn_ap: dict[int, dict[str, float]] = {}
    ref_fi = int(np.nanargmin(np.abs(r50_arr - k_ref)))
    clist_t3 = comp_subs["G8_9"][:3]
    stars_t3 = [tid, *clist_t3]
    rap_map = {tid: PROXY_R_AP}
    for cid in clist_t3:
        rap_map[cid] = float(np.nanmedian([aperture_by_frame[fi].get(cid, float("nan")) for fi in range(n_frames)]))
    for fi in range(n_frames):
        syn_cache[fi] = {}
        syn_ap[fi] = {}
        sc = r50_arr[fi]
        for cid in stars_t3:
            rap = rap_map[cid]
            ee_curve = np.array([_moffat_ee(r, sc, beta=3.0) for r in radii])
            syn_cache[fi][cid] = {"radii": radii, "ee": ee_curve, "qc_ok": True, "r50": sc}
            syn_ap[fi][cid] = rap
    d_t3_syn = _delta_ap_frozen_k(
        tid, clist_t3, ee_cache=syn_cache, aperture_by_frame=syn_ap,
        r50_arr=r50_arr, scale_arr=r50_arr, n_frames=n_frames, k_ref=k_ref,
        target_r_override=PROXY_R_AP,
    )
    tests["T3"] = {
        "synthetic_range_mmag": float(np.nanmax(d_t3_syn) - np.nanmin(d_t3_syn)) if np.isfinite(d_t3_syn).any() else float("nan"),
        "synthetic_max_abs_mmag": float(np.max(np.abs(d_t3_syn))) if np.isfinite(d_t3_syn).all() else float("nan"),
        "pass": float(np.max(np.abs(d_t3_syn))) < 1e-6 if np.isfinite(d_t3_syn).all() else False,
        "real_data_range_mmag": None,  # filled below if needed
    }

    return tests


def _pick_proxies(
    catalog: pd.DataFrame,
    star_ids: list[str],
    ee_cache: dict[int, dict[str, dict[str, Any]]],
    n_frames: int,
) -> list[str]:
    """Clean stars G 12-15 passing QC on all frames."""
    cands = []
    for cid in star_ids:
        if cid == FOCUS_ID:
            continue
        row = catalog.loc[catalog["catalog_id"] == cid]
        if row.empty:
            continue
        g = float(row.iloc[0]["phot_g"])
        if not (12.0 <= g <= 15.5):
            continue
        ok_all = all(
            cid in ee_cache[fi] and ee_cache[fi][cid].get("qc_ok", False)
            for fi in range(n_frames)
        )
        if ok_all:
            cands.append((g, cid))
    cands.sort(key=lambda t: t[0])
    return [c for _, c in cands[:5]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=False)
    ap.add_argument("--step1b-json", type=Path, default=Path("tmp/closure_step1b_results.json"))
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1c_results.json"))
    ap.add_argument("--cache", type=Path, default=Path("tmp/closure_step1c_ee_cache.npz"))
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--fixture-check", action="store_true", help="validate against closure_a1_reference_fixture.py and exit")
    args = ap.parse_args()

    if args.fixture_check:
        res = _validate_reference_fixture()
        print(json.dumps(res, indent=2))
        raise SystemExit(0 if res["pass"] else 1)

    if args.draft is None:
        ap.error("--draft is required unless --fixture-check")
    t0 = time.perf_counter()
    draft = args.draft.resolve()
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    csvs, catalog_full, n_frames = _load_all_proc(lights)

    with args.step1b_json.open(encoding="utf-8") as f:
        s1b = json.load(f)
    star_ids: list[str] = s1b["part_a"]["star_ids"]
    catalog = catalog_full[catalog_full["catalog_id"].isin(star_ids)].copy()

    if args.cache.is_file() and not args.rebuild_cache:
        npz = np.load(args.cache, allow_pickle=True)
        ee_cache = npz["ee_cache"].item()
        aperture_by_frame = npz["aperture_by_frame"].item()
        frame_names = list(npz["frame_names"])
        r50_series = list(npz["r50_series"])
        sky_med = list(npz["sky_med"])
        vy_series = list(npz["vy_series"])
    else:
        ee_cache, aperture_by_frame, frame_names, r50_series, sky_med, vy_series = _build_ee_cache(
            draft, star_ids, csvs=csvs, lights=lights
        )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.cache,
            ee_cache=ee_cache,
            aperture_by_frame=aperture_by_frame,
            frame_names=np.array(frame_names),
            r50_series=np.array(r50_series),
            sky_med=np.array(sky_med),
            vy_series=np.array(vy_series),
        )

    r50_arr = np.array(r50_series, dtype=np.float64)
    sky_arr = np.array(sky_med, dtype=np.float64)
    vy_arr = np.array(vy_series, dtype=np.float64)

    # moment median for production estimator
    moment_med = []
    for fi, proc in enumerate(csvs):
        df = pd.read_csv(proc, dtype={"catalog_id": str}).set_index("catalog_id")
        m = []
        for cid in star_ids:
            if cid not in df.index:
                continue
            row = df.loc[cid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            v = float(row.get("fwhm_estimate_px", float("nan")))
            if math.isfinite(v):
                m.append(v)
        moment_med.append(float(np.median(m)) if m else float("nan"))
    moment_arr = np.array(moment_med, dtype=np.float64)

    comp_subs = _comp_subsets(catalog, star_ids, exclude=set())
    proxies = _pick_proxies(catalog, star_ids, ee_cache, n_frames)

    # Self-tests first
    tests = run_self_tests(ee_cache, aperture_by_frame, r50_arr, star_ids, comp_subs, n_frames, catalog)
    tests["F_fixture"] = _validate_reference_fixture()

    # Recompute B.3 for proxies + real target
    b3: dict[str, Any] = {"proxies": {}, "real_target": {}}
    for pid in proxies:
        b3["proxies"][pid] = {}
        for label, clist in comp_subs.items():
            d = _delta_ap_series(
                pid, clist, ee_cache=ee_cache, aperture_by_frame=aperture_by_frame,
                n_frames=n_frames, target_r_override=PROXY_R_AP,
            )
            b3["proxies"][pid][label] = {
                **_summarize_delta(d, r50_arr, sky_arr, frame_names),
                "phot_g": float(catalog.loc[catalog["catalog_id"] == pid, "phot_g"].iloc[0]),
            }
    for label, clist in comp_subs.items():
        d = _delta_ap_series(
            FOCUS_ID, clist, ee_cache=ee_cache, aperture_by_frame=aperture_by_frame,
            n_frames=n_frames,
        )
        b3["real_target"][label] = {
            **_summarize_delta(d, r50_arr, sky_arr, frame_names),
            "qc_failed": True,
        }

    # T4 ratio on best proxy (widest G8-9 range expected)
    t4_ratios = []
    for pid in proxies:
        r89 = abs(b3["proxies"][pid]["G8_9"]["range_best_worst_mmag"])
        r11 = abs(b3["proxies"][pid]["G_gt_11"]["range_best_worst_mmag"])
        if r11 > 1e-6:
            t4_ratios.append(r89 / r11)
        elif r89 > 0:
            t4_ratios.append(float("inf"))
    t4_ratio = float(np.median(t4_ratios)) if t4_ratios else float("nan")
    tests["T4"] = {
        "ratio_G89_over_Ggt11_median": t4_ratio,
        "pass": T4_RATIO_LO <= t4_ratio <= T4_RATIO_HI if math.isfinite(t4_ratio) else False,
        "per_proxy_ratios": t4_ratios,
    }

    k_ref = float(np.nanmedian(r50_arr))
    b5: dict[str, Any] = {}
    for pid in proxies[:3]:
        b5[pid] = {}
        clist = comp_subs["G8_9"] or comp_subs["G9_11"]
        d_t3 = _delta_ap_frozen_k(
            pid, clist, ee_cache=ee_cache, aperture_by_frame=aperture_by_frame,
            r50_arr=r50_arr, scale_arr=r50_arr, n_frames=n_frames, k_ref=k_ref,
            target_r_override=PROXY_R_AP,
        )
        d_prod = _delta_ap_frozen_k(
            pid, clist, ee_cache=ee_cache, aperture_by_frame=aperture_by_frame,
            r50_arr=r50_arr, scale_arr=vy_arr, n_frames=n_frames, k_ref=float(np.nanmedian(vy_arr)),
            target_r_override=PROXY_R_AP,
        )
        b5[pid] = {
            "scale_r50_tautology_range_mmag": float(np.nanmax(d_t3) - np.nanmin(d_t3)),
            "scale_r50_tautology_max_abs_mmag": float(np.nanmax(np.abs(d_t3))),
            "scale_VY_FWHM_production_range_mmag": float(np.nanmax(d_prod) - np.nanmin(d_prod)),
        }

    # B.6 reopt
    snr_path = draft / "aperture_snr_table.json"
    with snr_path.open(encoding="utf-8") as f:
        snr_disk = json.load(f)
    b6: dict[str, Any] = {}
    pid = proxies[0] if proxies else star_ids[0]
    for label, clist in comp_subs.items():
        dre = []
        for fi in range(n_frames):
            fw = vy_arr[fi] if math.isfinite(vy_arr[fi]) else TABLE_FWHM
            sky = sky_arr[fi] if math.isfinite(sky_arr[fi]) else float(snr_disk["sky_adu_per_px"])
            tbl = _snr_table_for_frame(fw, sky, gain=float(snr_disk["gain"]), rn=float(snr_disk["read_noise"]), zero_point=25.0)
            df = pd.read_csv(csvs[fi], dtype={"catalog_id": str})
            row_t = df.loc[df["catalog_id"] == pid].iloc[0]
            rap_t = _lookup_table_r(_mag_for_aperture(row_t), tbl)
            rap_t = PROXY_R_AP  # proxy forced r for A-1 question
            ee_t = _ee_at_radius(ee_cache[fi][pid]["radii"], ee_cache[fi][pid]["ee"], rap_t)
            ee_c = []
            for cid in clist:
                if cid not in ee_cache[fi]:
                    continue
                row_c = df.loc[df["catalog_id"] == cid].iloc[0]
                rap = _lookup_table_r(_mag_for_aperture(row_c), tbl)
                ee_c.append(_ee_at_radius(ee_cache[fi][cid]["radii"], ee_cache[fi][cid]["ee"], rap))
            if ee_c and math.isfinite(ee_t) and ee_t > 0:
                med_c = float(np.median(ee_c))
                dre.append(_delta_ap_mmag(ee_t, ee_c))
            else:
                dre.append(float("nan"))
        dre = np.array(dre, dtype=np.float64)
        m = np.isfinite(dre) & np.isfinite(sky_arr)
        b6[label] = {
            "range_mmag": float(np.nanmax(dre) - np.nanmin(dre)) if np.isfinite(dre).any() else float("nan"),
            "corr_sky": float(np.corrcoef(sky_arr[m], dre[m])[0, 1]) if m.sum() >= 5 else float("nan"),
        }

    # D estimator ranking
    est_rank = [
        _estimator_stats(r50_arr, vy_arr, "VY_FWHM"),
        _estimator_stats(r50_arr, moment_arr, "moment_median"),
    ]

    # A.6 diagnostic EE values
    a6 = {}
    comp_g89 = comp_subs["G8_9"][0] if comp_subs["G8_9"] else None
    med_fi = int(np.nanargmin(np.abs(r50_arr - np.nanmedian(r50_arr))))
    for fn, fi in [
        ("007_min_r50", int(np.nanargmin(r50_arr))),
        ("048_max_r50", int(np.nanargmax(r50_arr))),
        ("median_r50", med_fi),
    ]:
        a6[fn] = {}
        for sid in [FOCUS_ID, comp_g89]:
            if sid and sid in ee_cache[fi]:
                rap = PROXY_R_AP if sid != FOCUS_ID else aperture_by_frame[fi].get(sid, PROXY_R_AP)
                if sid == comp_g89:
                    rap = aperture_by_frame[fi].get(sid, float("nan"))
                ee = _ee_at_radius(ee_cache[fi][sid]["radii"], ee_cache[fi][sid]["ee"], rap)
                a6[fn][sid] = {"ee_at_rap": ee, "rap": rap, "qc_ok": ee_cache[fi][sid].get("qc_ok")}

    # Audit answers (pre-fix semantics documented)
    audit = {
        "A1_join": "catalog_id via df.set_index; aperture_by_frame[fi][cid] from row.get(aperture_r_px) lines 477-484 step1b",
        "A2_comp_radius": "comp own aperture: aperture_by_frame[fi].get(cid) line 584-590 step1b",
        "A3_normalisation": "per star per frame: norm = arr[-1] in _curve_of_growth line 156-157",
        "A4_curve_pooling": "per-star ee_cache[fi][cid], NOT pooled median",
        "A5_scale_frame_bug": "scale_adj = moment_median rescaled to TABLE_FWHM=2.395; k_i = r_ap/2.395 mixed r50 with FWHM units",
        "located_defect": "Focus target non-monotonic COG (EE>1, r50 nonsense) used as B.3 numerator despite focus_in_qc=false; monotone fix + proxy decoupling",
        "A6": a6,
    }

    # Part D.5 counts from step1b + growth recount
    n_growth_ok = sum(1 for cid in catalog["catalog_id"] if all(
        cid in ee_cache[fi] and ee_cache[fi][cid].get("qc_ok") for fi in range(n_frames)
    ))

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wall_sec": time.perf_counter() - t0,
        "draft": str(draft),
        "harness_audit": audit,
        "self_tests": tests,
        "proxies": proxies,
        "b3_delta_ap": b3,
        "b5_frozen_k": b5,
        "b6_reopt": b6,
        "estimator_ranking": est_rank,
        "r50_frame": {"min": float(np.nanmin(r50_arr)), "median": float(np.nanmedian(r50_arr)), "max": float(np.nanmax(r50_arr))},
        "isolation_counts": {
            "n_eligible": int(s1b["part_a"]["n_eligible"]),
            "n_isolated_angular": int(s1b["part_a"]["n_isolated_angular"]),
            "n_growth_ok_sample_frame": int(s1b["part_a"]["n_growth_ok"]),
            "n_pass_all_sample_frame": int(s1b["part_a"]["n_pass_all"]),
            "n_qc_ok_all_frames": n_growth_ok,
            "n_old_admitted_new_rejects": int(s1b["part_a"]["n_old_admitted_new_rejects"]),
        },
        "option_iv_r50_sizing": {
            "note": "evaluate only",
            "cache_build_sec": time.perf_counter() - t0,
            "stars_per_frame": len(star_ids),
            "frames": n_frames,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")
    print(f"T1 pass {tests['T1']['self_pass']} T2 {tests['T2']['pass']} T3 {tests['T3']['pass']} T4 {tests.get('T4',{}).get('pass')} F {tests['F_fixture']['pass']}")


if __name__ == "__main__":
    main()
