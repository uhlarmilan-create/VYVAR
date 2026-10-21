#!/usr/bin/env python3
"""Closure Step 1e: repair measurement method (photutils exact COG, Gaussian centroid),
robust ranges, reject bad curves; gates G6/G7; re-measure delta_ap.

Repairs only -- no production code change, no new star set, no anchor re-cut.

Usage:
  python dev/tools/closure_step1e_differential_aperture.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --step1b-json tmp/closure_step1b_results.json \\
    --out tmp/closure_step1e_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import modeling
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from scipy import stats

from closure_step1b_differential_aperture import (  # noqa: E402
    COG_DR,
    COG_RMAX,
    FOCUS_ID,
    TABLE_FWHM,
    _ee_at_radius,
    _load_all_proc,
    _lookup_table_r,
    _mag_for_aperture,
    _r_at_ee,
    _snr_table_for_frame,
)
from closure_step1c_differential_aperture import (  # noqa: E402
    GATE_MM,
    PROXY_R_AP,
    T4_RATIO_HI,
    T4_RATIO_LO,
    _comp_subsets,
    _delta_ap_frozen_k,
    _delta_ap_mmag,
    _estimator_stats,
    _pick_proxies,
)

# Match closure_a1_reference_fixture.py L2 exactly (R1)
ANNULUS_IN = 25.0
ANNULUS_OUT = 45.0
NORM_RADIUS = 12.0
COG_RADII = np.arange(COG_DR, COG_RMAX + COG_DR / 2, COG_DR)
FIT_BOX = 15
G6_MAX_RATIO = 1.25
G7_TOL_MMAG = 5.0
FRAME_DROP_FRAC = 0.20
MONO_TOL = 1e-4
EE1_TOL = 1e-6


def _curve_of_growth_photutils(
    data: np.ndarray,
    xc: float,
    yc: float,
    *,
    radii: np.ndarray = COG_RADII,
) -> dict[str, Any] | None:
    """R1: photutils exact apertures at sub-pixel centre (fixture L2)."""
    h, w = data.shape
    max_r = float(np.max(radii)) + ANNULUS_OUT + 2.0
    if xc - max_r < 0 or yc - max_r < 0 or xc + max_r >= w or yc + max_r >= h:
        return None
    ann = CircularAnnulus([(xc, yc)], r_in=ANNULUS_IN, r_out=ANNULUS_OUT)
    sky_pp = float(aperture_photometry(data, ann)["aperture_sum"][0] / ann.area)
    flux: list[float] = []
    for r in radii:
        ap = CircularAperture([(xc, yc)], r=float(r))
        s = float(aperture_photometry(data, ap)["aperture_sum"][0])
        flux.append(s - sky_pp * ap.area)
    arr = np.asarray(flux, dtype=np.float64)
    if arr[-1] <= 0:
        return None
    ee = arr / arr[-1]
    return {"radii": radii, "ee": ee, "flux": arr}


def _cog_admissible(radii: np.ndarray, ee: np.ndarray) -> tuple[bool, str]:
    """R4: reject non-monotonic or EE>1 inside normalisation radius."""
    if ee.size < 2:
        return False, "too_few_points"
    if np.any(ee > 1.0 + EE1_TOL):
        return False, "ee_gt_1"
    if np.any(np.diff(ee) < -MONO_TOL):
        return False, "non_monotonic"
    if ee[-1] <= 0:
        return False, "bad_norm"
    return True, "ok"


def _gaussian_centroid_or_none(
    data: np.ndarray,
    x_proc: float,
    y_proc: float,
    *,
    fwhm_hint: float,
    box: int = FIT_BOX,
) -> tuple[float, float] | None:
    """R2: centroid from converged Gaussian2D+Const2D; None if fit fails."""
    h, w = data.shape
    xc_i, yc_i = int(round(x_proc)), int(round(y_proc))
    if not (box <= xc_i < w - box and box <= yc_i < h - box):
        return None
    cut = data[yc_i - box : yc_i + box + 1, xc_i - box : xc_i + box + 1].astype(np.float64)
    yy, xx = np.mgrid[0 : cut.shape[0], 0 : cut.shape[1]]
    bg = float(np.median(cut))
    amp = max(float(np.max(cut) - bg), 1.0)
    fitter = modeling.fitting.TRFLSQFitter()
    c0 = modeling.models.Const2D(amplitude=bg)
    g0 = modeling.models.Gaussian2D(
        amplitude=amp,
        x_mean=float(box),
        y_mean=float(box),
        x_stddev=fwhm_hint / 2.355,
        y_stddev=fwhm_hint / 2.355,
        theta=0.0,
    )
    g0.x_mean.bounds = (box - 2, box + 2)
    g0.y_mean.bounds = (box - 2, box + 2)
    g0.x_stddev.bounds = (0.4, 15.0)
    g0.y_stddev.bounds = (0.4, 15.0)
    g0.theta.fixed = True
    try:
        mg = fitter(g0 + c0, xx, yy, cut, maxiter=200)
        sx = float(mg.x_stddev_0.value)
        sy = float(mg.y_stddev_0.value)
        fwhm_g = 0.5 * (2.355 * sx + 2.355 * sy)
        if not (math.isfinite(fwhm_g) and 0.5 < fwhm_g < 20):
            return None
        xi = xc_i - box + float(mg.x_mean_0.value)
        yi = yc_i - box + float(mg.y_mean_0.value)
        if not (math.isfinite(xi) and math.isfinite(yi)):
            return None
        return xi, yi
    except Exception:  # noqa: BLE001
        return None


def _build_ee_cache_step1e(
    draft: Path,
    star_ids: list[str],
    *,
    csvs: list[Path],
    lights: Path,
) -> tuple[
    dict[int, dict[str, dict[str, Any]]],
    dict[int, dict[str, float]],
    list[str],
    list[float],
    list[float],
    list[float],
    dict[str, Any],
]:
    """Build EE cache with R1-R4; track dropped star-frames and excluded frames."""
    ee_cache: dict[int, dict[str, dict[str, Any]]] = {}
    aperture_by_frame: dict[int, dict[str, float]] = {}
    frame_names: list[str] = []
    r50_series: list[float] = []
    sky_med: list[float] = []
    vy_series: list[float] = []
    drops: dict[str, Any] = {
        "fit_fail": 0,
        "cog_fail": 0,
        "validation_fail": 0,
        "by_reason": {},
        "by_star": {},
        "by_frame": {},
        "excluded_frames": [],
    }

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
        n_attempted = 0
        n_dropped_frame = 0

        for cid in star_ids:
            if cid not in df.index:
                continue
            row = df.loc[cid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            n_attempted += 1
            x_proc, y_proc = float(row["x"]), float(row["y"])
            aperture_by_frame[fi][cid] = float(row.get("aperture_r_px", float("nan")))
            if "sky_adu_per_px_annulus" in row.index:
                sv = float(row["sky_adu_per_px_annulus"])
                if math.isfinite(sv):
                    skies.append(sv)
            fwhm_hint = float(row.get("fwhm_estimate_px", TABLE_FWHM))
            if not math.isfinite(fwhm_hint):
                fwhm_hint = TABLE_FWHM

            cen = _gaussian_centroid_or_none(data, x_proc, y_proc, fwhm_hint=fwhm_hint)
            if cen is None:
                drops["fit_fail"] += 1
                n_dropped_frame += 1
                drops["by_star"].setdefault(cid, {"fit_fail": 0, "cog_fail": 0, "validation_fail": 0})
                drops["by_star"][cid]["fit_fail"] += 1
                continue
            xi, yi = cen
            cog = _curve_of_growth_photutils(data, xi, yi)
            if cog is None:
                drops["cog_fail"] += 1
                n_dropped_frame += 1
                drops["by_star"].setdefault(cid, {"fit_fail": 0, "cog_fail": 0, "validation_fail": 0})
                drops["by_star"][cid]["cog_fail"] += 1
                continue
            ok, reason = _cog_admissible(cog["radii"], cog["ee"])
            if not ok:
                drops["validation_fail"] += 1
                n_dropped_frame += 1
                drops["by_reason"][reason] = drops["by_reason"].get(reason, 0) + 1
                drops["by_star"].setdefault(cid, {"fit_fail": 0, "cog_fail": 0, "validation_fail": 0})
                drops["by_star"][cid]["validation_fail"] += 1
                continue
            ee_cache[fi][cid] = {
                "radii": cog["radii"],
                "ee": cog["ee"],
                "centroid_x": xi,
                "centroid_y": yi,
                "r50": _r_at_ee(cog["radii"], cog["ee"], 0.5),
                "qc_ok": True,
            }
            if math.isfinite(ee_cache[fi][cid]["r50"]):
                r50_stars.append(ee_cache[fi][cid]["r50"])

        drops["by_frame"][proc.name] = {
            "attempted": n_attempted,
            "dropped": n_dropped_frame,
            "kept": len(ee_cache[fi]),
        }
        if n_attempted > 0 and n_dropped_frame / n_attempted > FRAME_DROP_FRAC:
            drops["excluded_frames"].append(proc.name)

        r50_series.append(float(np.median(r50_stars)) if r50_stars else float("nan"))
        sky_med.append(float(np.median(skies)) if skies else float("nan"))

    return ee_cache, aperture_by_frame, frame_names, r50_series, sky_med, vy_series, drops


def _robust_delta_stats(
    d: np.ndarray,
    r50_arr: np.ndarray,
    sky_arr: np.ndarray,
    frame_names: list[str],
    *,
    excluded_frames: set[str],
) -> dict[str, Any]:
    """R3: slope*span and p95-p5; two-point diff labelled separately only."""
    mask = np.isfinite(d)
    for i, fn in enumerate(frame_names):
        if fn in excluded_frames:
            mask[i] = False
    d_use = d.copy()
    d_use[~mask] = float("nan")

    valid = d_use[np.isfinite(d_use)]
    idx_best = int(np.nanargmin(r50_arr))
    idx_worst = int(np.nanargmax(r50_arr))
    m_r = np.isfinite(d_use) & np.isfinite(r50_arr)
    m_s = np.isfinite(d_use) & np.isfinite(sky_arr)

    slope = intercept = float("nan")
    slope_span = float("nan")
    fit_residual_rms = float("nan")
    if m_r.sum() >= 5:
        slope, intercept = np.polyfit(r50_arr[m_r], d_use[m_r], 1)
        resid = d_use[m_r] - (slope * r50_arr[m_r] + intercept)
        fit_residual_rms = float(np.sqrt(np.mean(resid**2)))
        span = float(np.nanmax(r50_arr) - np.nanmin(r50_arr))
        slope_span = float(slope * span)

    p95_p5 = float(np.percentile(valid, 95) - np.percentile(valid, 5)) if valid.size >= 5 else float("nan")

    twopoint = float("nan")
    if np.isfinite(d_use[idx_worst]) and np.isfinite(d_use[idx_best]):
        twopoint = float(abs(d_use[idx_worst] - d_use[idx_best]))

    return {
        "range_p95_p5_mmag": p95_p5,
        "slope_times_r50_span_mmag": slope_span,
        "slope_mmag_per_r50": float(slope) if math.isfinite(slope) else float("nan"),
        "slope_fit_residual_rms_mmag": fit_residual_rms,
        "two_point_min_max_r50_diff_mmag": twopoint,
        "two_point_note": "NOT a range; min-r50 vs max-r50 frame only",
        "min_r50_frame": frame_names[idx_best],
        "max_r50_frame": frame_names[idx_worst],
        "pearson_r50": float(np.corrcoef(r50_arr[m_r], d_use[m_r])[0, 1]) if m_r.sum() >= 5 else float("nan"),
        "spearman_r50": float(stats.spearmanr(r50_arr[m_r], d_use[m_r]).statistic) if m_r.sum() >= 5 else float("nan"),
        "pearson_sky": float(np.corrcoef(sky_arr[m_s], d_use[m_s])[0, 1]) if m_s.sum() >= 5 else float("nan"),
        "median_mmag": float(np.median(valid)) if valid.size else float("nan"),
        "n_frames_used": int(m_r.sum()),
    }


def _delta_ap_series(
    target_id: str,
    comp_list: list[str],
    *,
    ee_cache: dict[int, dict[str, dict[str, Any]]],
    aperture_by_frame: dict[int, dict[str, float]],
    n_frames: int,
    target_r_override: float | None = None,
    excluded_frames: set[str] | None = None,
    frame_names: list[str] | None = None,
) -> np.ndarray:
    ex = excluded_frames or set()
    deltas = []
    for fi in range(n_frames):
        if frame_names and frame_names[fi] in ex:
            deltas.append(float("nan"))
            continue
        if target_id not in ee_cache[fi]:
            deltas.append(float("nan"))
            continue
        rap_t = PROXY_R_AP if target_r_override is not None else aperture_by_frame[fi].get(target_id, float("nan"))
        if target_r_override is not None:
            rap_t = target_r_override
        ee_t = _ee_at_radius(ee_cache[fi][target_id]["radii"], ee_cache[fi][target_id]["ee"], rap_t)
        ee_c: list[float] = []
        for cid in comp_list:
            if cid not in ee_cache[fi]:
                continue
            rap_c = aperture_by_frame[fi].get(cid, float("nan"))
            ee_c.append(_ee_at_radius(ee_cache[fi][cid]["radii"], ee_cache[fi][cid]["ee"], rap_c))
        if not ee_c or not math.isfinite(ee_t) or ee_t <= 0:
            deltas.append(float("nan"))
            continue
        deltas.append(_delta_ap_mmag(ee_t, ee_c))
    return np.array(deltas, dtype=np.float64)


def _gate_g7(
    cog_fn: Callable[..., dict[str, Any] | None],
) -> dict[str, Any]:
    """G7: harness COG on fixture synthetic images vs L2 expected table."""
    from closure_a1_reference_fixture import (  # noqa: PLC0415
        BETA,
        COG_RADII as FIX_RADII,
        R50_GRID,
        R_TARGET,
        SKY_ADU,
        SUBSETS,
        TOTAL_FLUX,
        alpha_from_r50,
        build_expected,
        ee_at,
        render_moffat,
    )

    exp = build_expected()
    errors: dict[str, list[float]] = {k: [] for k in SUBSETS}
    for r50 in R50_GRID:
        a = alpha_from_r50(r50, BETA)
        img = render_moffat((161, 161), 80.37, 80.62, a, BETA, TOTAL_FLUX, SKY_ADU)
        cog = cog_fn(img, 80.37, 80.62, radii=FIX_RADII)
        if cog is None:
            return {"pass": False, "error": "cog_none", "max_abs_error_mmag": float("inf")}
        ee = cog["ee"]
        et = ee_at(FIX_RADII, ee, R_TARGET)
        key = f"{r50:.2f}"
        for subset, radii in SUBSETS.items():
            ec = [ee_at(FIX_RADII, ee, r) for r in radii]
            got = _delta_ap_mmag(et, ec)
            want = exp["table"][key][subset]
            errors[subset].append(abs(got - want))
    max_err = max(v for vals in errors.values() for v in vals)
    return {
        "pass": max_err <= G7_TOL_MMAG,
        "max_abs_error_mmag": max_err,
        "per_subset_max_err": {k: max(v) for k, v in errors.items()},
        "expected_ranges": exp["range_over_span"],
    }


def _gate_g4_jitter(
    cog_fn: Callable[..., dict[str, Any] | None],
    *,
    label: str,
) -> dict[str, Any]:
    """Position jitter of a COG method at R_TARGET on fixture (G4-style)."""
    from closure_a1_reference_fixture import (  # noqa: PLC0415
        BETA,
        COG_RADII as FIX_RADII,
        R_TARGET,
        SKY_ADU,
        TOTAL_FLUX,
        alpha_from_r50,
        ee_at,
        ee_curve_photutils,
        render_moffat,
    )

    a = alpha_from_r50(1.87, BETA)
    truth, measured = [], []
    for dx, dy in [(0.0, 0.0), (0.25, 0.25), (0.5, 0.0), (0.5, 0.5), (-0.4, 0.3)]:
        xc, yc = 80.0 + dx, 80.0 + dy
        img = render_moffat((161, 161), xc, yc, a, BETA, TOTAL_FLUX, SKY_ADU)
        truth.append(ee_at(FIX_RADII, ee_curve_photutils(img, xc, yc), R_TARGET))
        cog = cog_fn(img, xc, yc, radii=FIX_RADII)
        if cog is None:
            measured.append(float("nan"))
        else:
            measured.append(ee_at(FIX_RADII, cog["ee"], R_TARGET))
    m_arr = np.array(measured, dtype=np.float64)
    t_arr = np.array(truth, dtype=np.float64)
    bias_pct = 100.0 * (float(np.mean(m_arr)) - float(np.mean(t_arr))) / float(np.mean(t_arr))
    jitter = abs(2.5 * math.log10(float(np.max(m_arr)) / float(np.min(m_arr)))) * 1000.0
    return {"label": label, "bias_pct_of_ee": bias_pct, "position_jitter_mmag": jitter}


def _gate_g6(
    proxies: list[str],
    comp_subs: dict[str, list[str]],
    ee_cache: dict[int, dict[str, dict[str, Any]]],
    aperture_by_frame: dict[int, dict[str, float]],
    r50_arr: np.ndarray,
    sky_arr: np.ndarray,
    frame_names: list[str],
    excluded_frames: set[str],
    n_frames: int,
) -> dict[str, Any]:
    """G6: five proxies must agree within 25% on p95-p5 range per sub-ensemble."""
    label_map = {"G8_9": "G_8_9", "G9_11": "G_9_11", "G_gt_11": "G_gt_11"}
    results: dict[str, Any] = {"pass": True, "sub_ensembles": {}}
    for label in ("G8_9", "G9_11", "G_gt_11"):
        clist = comp_subs[label]
        ranges: dict[str, float] = {}
        for pid in proxies:
            d = _delta_ap_series(
                pid, clist, ee_cache=ee_cache, aperture_by_frame=aperture_by_frame,
                n_frames=n_frames, target_r_override=PROXY_R_AP,
                excluded_frames=excluded_frames, frame_names=frame_names,
            )
            st = _robust_delta_stats(d, r50_arr, sky_arr, frame_names, excluded_frames=excluded_frames)
            ranges[pid] = st["range_p95_p5_mmag"]
        vals = [v for v in ranges.values() if math.isfinite(v) and v > 0]
        if len(vals) < 2:
            ratio = float("nan")
            ok = False
        else:
            ratio = float(max(vals) / min(vals))
            ok = ratio <= G6_MAX_RATIO
        results["sub_ensembles"][label_map[label]] = {
            "per_proxy_p95_p5_mmag": ranges,
            "max_min_ratio": ratio,
            "pass": ok,
        }
        if not ok:
            results["pass"] = False
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=False)
    ap.add_argument("--step1b-json", type=Path, default=Path("tmp/closure_step1b_results.json"))
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1e_results.json"))
    ap.add_argument("--cache", type=Path, default=Path("tmp/closure_step1e_ee_cache.npz"))
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--gate-only", action="store_true", help="run G6/G7/G4 jitter only (needs cache)")
    args = ap.parse_args()

    t0 = time.perf_counter()

    # G4 before (integer centre) and G7/G4 after (photutils harness)
    from closure_a1_reference_fixture import ee_curve_integer_centre  # noqa: PLC0415

    def _integer_cog(img, xc, yc, radii=COG_RADII):
        ee = ee_curve_integer_centre(img, xc, yc, radii=radii)
        return {"radii": radii, "ee": ee}

    g4_before = _gate_g4_jitter(_integer_cog, label="integer_centre_L3")
    g7 = _gate_g7(_curve_of_growth_photutils)
    g4_after = _gate_g4_jitter(_curve_of_growth_photutils, label="photutils_harness_R1")

    if args.gate_only:
        print(json.dumps({"G7": g7, "G4_before": g4_before, "G4_after": g4_after}, indent=2))
        raise SystemExit(0 if g7["pass"] else 1)

    if args.draft is None:
        ap.error("--draft is required unless --gate-only")

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
        drops = npz["drops"].item() if "drops" in npz else {}
    else:
        ee_cache, aperture_by_frame, frame_names, r50_series, sky_med, vy_series, drops = _build_ee_cache_step1e(
            draft, star_ids, csvs=csvs, lights=lights,
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
            drops=drops,
        )

    excluded_frames = set(drops.get("excluded_frames", []))
    r50_arr = np.array(r50_series, dtype=np.float64)
    sky_arr = np.array(sky_med, dtype=np.float64)
    vy_arr = np.array(vy_series, dtype=np.float64)

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

    # Fixed proxy IDs from Step 1c/1d (star set stands; do not re-pick on repaired qc_ok)
    fixed_proxies: list[str] = []
    s1c_path = Path("tmp/closure_step1c_mmag_results.json")
    if s1c_path.is_file():
        with s1c_path.open(encoding="utf-8") as f:
            fixed_proxies = json.load(f).get("proxies", [])
    if not fixed_proxies:
        fixed_proxies = _pick_proxies(catalog, star_ids, ee_cache, n_frames)
    proxies = fixed_proxies

    g6 = _gate_g6(
        proxies, comp_subs, ee_cache, aperture_by_frame, r50_arr, sky_arr,
        frame_names, excluded_frames, n_frames,
    )

    gates_pass = g6["pass"] and g7["pass"]

    # M1-M4 only meaningful if gates pass; still compute for audit JSON
    m1: dict[str, Any] = {"proxies": {}, "real_target": {}}
    for pid in proxies:
        m1["proxies"][pid] = {"phot_g": float(catalog.loc[catalog["catalog_id"] == pid, "phot_g"].iloc[0])}
        for label, clist in comp_subs.items():
            d = _delta_ap_series(
                pid, clist, ee_cache=ee_cache, aperture_by_frame=aperture_by_frame,
                n_frames=n_frames, target_r_override=PROXY_R_AP,
                excluded_frames=excluded_frames, frame_names=frame_names,
            )
            m1["proxies"][pid][label] = _robust_delta_stats(
                d, r50_arr, sky_arr, frame_names, excluded_frames=excluded_frames,
            )
    for label, clist in comp_subs.items():
        d = _delta_ap_series(
            FOCUS_ID, clist, ee_cache=ee_cache, aperture_by_frame=aperture_by_frame,
            n_frames=n_frames, excluded_frames=excluded_frames, frame_names=frame_names,
        )
        m1["real_target"][label] = {
            **_robust_delta_stats(d, r50_arr, sky_arr, frame_names, excluded_frames=excluded_frames),
            "qc_failed": True,
        }

    # T4 median across proxies (pre-registered)
    t4_ratios = []
    for pid in proxies:
        r89 = m1["proxies"][pid]["G8_9"]["range_p95_p5_mmag"]
        r11 = m1["proxies"][pid]["G_gt_11"]["range_p95_p5_mmag"]
        if math.isfinite(r89) and math.isfinite(r11) and r11 > 1e-6:
            t4_ratios.append(r89 / r11)
    t4_median = float(np.median(t4_ratios)) if t4_ratios else float("nan")

    # M2 consolidated (median across proxies of p95-p5)
    m2: dict[str, Any] = {}
    for label, fixture_key in [("G8_9", "G_8_9"), ("G9_11", "G_9_11"), ("G_gt_11", "G_gt_11")]:
        vals = [
            m1["proxies"][p][label]["range_p95_p5_mmag"]
            for p in proxies
            if p in m1["proxies"] and math.isfinite(m1["proxies"][p][label]["range_p95_p5_mmag"])
        ]
        if vals:
            med = float(np.median(vals))
            spread = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            m2[label] = {
                "median_p95_p5_mmag": med,
                "iqr_mmag": spread,
                "report": f"{med:.1f} +/- {spread:.1f} mmag",
                "fixture_expectation_mmag": g7["expected_ranges"][fixture_key],
            }
        else:
            m2[label] = {"median_p95_p5_mmag": float("nan")}

    k_ref = float(np.nanmedian(r50_arr))
    m4_b5: dict[str, Any] = {}
    for pid in proxies[:3]:
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
        m4_b5[pid] = {
            "scale_r50_tautology_range_mmag": float(np.nanmax(d_t3) - np.nanmin(d_t3)),
            "scale_r50_tautology_max_abs_mmag": float(np.nanmax(np.abs(d_t3))),
            "scale_VY_FWHM_production_range_mmag": float(np.nanmax(d_prod) - np.nanmin(d_prod)),
        }

    snr_path = draft / "aperture_snr_table.json"
    with snr_path.open(encoding="utf-8") as f:
        snr_disk = json.load(f)
    m4_b6: dict[str, Any] = {}
    pid0 = proxies[0] if proxies else star_ids[0]
    for label, clist in comp_subs.items():
        dre = []
        for fi in range(n_frames):
            if frame_names[fi] in excluded_frames:
                dre.append(float("nan"))
                continue
            fw = vy_arr[fi] if math.isfinite(vy_arr[fi]) else TABLE_FWHM
            sky = sky_arr[fi] if math.isfinite(sky_arr[fi]) else float(snr_disk["sky_adu_per_px"])
            tbl = _snr_table_for_frame(fw, sky, gain=float(snr_disk["gain"]), rn=float(snr_disk["read_noise"]), zero_point=25.0)
            df = pd.read_csv(csvs[fi], dtype={"catalog_id": str})
            if pid0 not in ee_cache[fi]:
                dre.append(float("nan"))
                continue
            ee_t = _ee_at_radius(ee_cache[fi][pid0]["radii"], ee_cache[fi][pid0]["ee"], PROXY_R_AP)
            ee_c = []
            for cid in clist:
                if cid not in ee_cache[fi]:
                    continue
                row_c = df.loc[df["catalog_id"] == cid].iloc[0]
                rap = _lookup_table_r(_mag_for_aperture(row_c), tbl)
                ee_c.append(_ee_at_radius(ee_cache[fi][cid]["radii"], ee_cache[fi][cid]["ee"], rap))
            if ee_c and math.isfinite(ee_t) and ee_t > 0:
                dre.append(_delta_ap_mmag(ee_t, ee_c))
            else:
                dre.append(float("nan"))
        dre = np.array(dre, dtype=np.float64)
        m = np.isfinite(dre) & np.isfinite(sky_arr)
        m4_b6[label] = {
            "range_p95_p5_mmag": float(np.percentile(dre[np.isfinite(dre)], 95) - np.percentile(dre[np.isfinite(dre)], 5))
            if np.isfinite(dre).sum() >= 5 else float("nan"),
            "corr_sky": float(np.corrcoef(sky_arr[m], dre[m])[0, 1]) if m.sum() >= 5 else float("nan"),
        }

    repairs = {
        "R1": "closure_step1e_differential_aperture.py::_curve_of_growth_photutils L69-L93 photutils CircularAperture exact",
        "R2": "closure_step1e_differential_aperture.py::_gaussian_centroid_or_none L96-L145 Gaussian2D+Const2D centroid",
        "R3": "closure_step1e_differential_aperture.py::_robust_delta_stats L248-L295 p95-p5 + slope*span",
        "R4": "closure_step1e_differential_aperture.py::_cog_admissible L96-L107 reject non-monotonic/EE>1; drops tracked",
    }

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wall_sec": time.perf_counter() - t0,
        "draft": str(draft),
        "repairs": repairs,
        "drops": drops,
        "gates": {
            "G6_proxy_consistency": g6,
            "G7_measurement_equivalence": g7,
            "G4_integer_centre_jitter_mmag": g4_before["position_jitter_mmag"],
            "G4_photutils_harness_jitter_mmag": g4_after["position_jitter_mmag"],
            "gates_pass": gates_pass,
        },
        "T4": {
            "median_ratio_G89_over_Ggt11": t4_median,
            "per_proxy_ratios": t4_ratios,
            "pass": T4_RATIO_LO <= t4_median <= T4_RATIO_HI if math.isfinite(t4_median) else False,
            "fixture_expectation": 9.74,
        },
        "proxies": proxies,
        "M1_per_proxy": m1,
        "M2_consolidated": m2 if gates_pass else {"blocked": "G6 or G7 failed"},
        "M3_real_target": m1["real_target"],
        "M4_B5_frozen_k": m4_b5,
        "M4_B6_reopt": m4_b6,
        "estimator_ranking": [
            _estimator_stats(r50_arr, vy_arr, "VY_FWHM"),
            _estimator_stats(r50_arr, moment_arr, "moment_median"),
        ],
        "excluded_frames": list(excluded_frames),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"G6 {'PASS' if g6['pass'] else 'FAIL'}  G7 {'PASS' if g7['pass'] else 'FAIL'}")
    print(f"G4 jitter: before {g4_before['position_jitter_mmag']:.1f} mmag  after {g4_after['position_jitter_mmag']:.1f} mmag")
    if gates_pass:
        print(f"M2 G8_9: {m2['G8_9'].get('report', 'n/a')}")
    else:
        print("No decisive number (gate failure)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
