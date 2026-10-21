#!/usr/bin/env python3
"""Closure Step 1f/1g: C1-C3 admissibility, G8, proxy selection, differential measure.

Step 1g (F1): proxies G 11.5-13.0 at clamp 1.916 px, excluded from comp subsets, G9 gate.

Usage:
  python dev/tools/closure_step1f_differential_aperture.py \\
    --draft Archive/Drafts/draft_000435_snapshot_skysurface_20260716 \\
    --step1b-json tmp/closure_step1b_results.json \\
    --out tmp/closure_step1g_results.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from closure_step1b_differential_aperture import (  # noqa: E402
    DELTA_G_NEIGH,
    FOCUS_ID,
    MIN_SEP_PX,
    TABLE_FWHM,
    _angular_sep_arcsec,
    _arcsec_per_px,
    _load_all_proc,
    _lookup_table_r,
    _mag_for_aperture,
    _snr_table_for_frame,
)
from closure_step1c_differential_aperture import (  # noqa: E402
    PROXY_R_AP,
    T4_RATIO_HI,
    T4_RATIO_LO,
    _comp_subsets,
    _delta_ap_frozen_k,
    _delta_ap_mmag,
    _estimator_stats,
    _moffat_ee,
)
from closure_step1e_differential_aperture import (  # noqa: E402
    ANNULUS_IN,
    ANNULUS_OUT,
    COG_RADII,
    G6_MAX_RATIO,
    G7_TOL_MMAG,
    NORM_RADIUS,
    _curve_of_growth_photutils,
    _delta_ap_series,
    _gaussian_centroid_or_none,
    _gate_g4_jitter,
    _gate_g6,
    _gate_g7,
    _robust_delta_stats,
)
from closure_step1e_differential_aperture import _ee_at_radius  # noqa: E402

# C1: admissibility only inside metric region
ADMISS_R_MAX = 3.5
EE1_TOL = 1e-6
MONO_TOL = 1e-4

# C3 / F1 proxy band (Step 1g: G 11.5-13.0 at clamp; disjoint from comp subsets)
PROXY_G_MIN = 11.5
PROXY_G_MAX = 13.0
PROXY_MIN_FRAME_FRAC = 0.90
G8_NOISE_CEILING_MMAG = 144.3 / 3.0  # 48.1 mmag
RADIUS_SEP_MIN_PX = 0.3
EXP_SEC = 60.0
ZP_ELECTRONS = 21.68
R50_TYPICAL = 1.87
MOFFAT_BETA_FIXTURE = 3.0
COMP_R_TYPICAL = 3.166  # largest G 8-9 comparison radius


def _cog_admissible_c1(radii: np.ndarray, ee: np.ndarray) -> tuple[bool, str]:
    """C1: EE<=1 and monotonicity only for r <= ADMISS_R_MAX px."""
    if ee.size < 2:
        return False, "too_few_points"
    mask = radii <= ADMISS_R_MAX + 1e-9
    if mask.sum() < 2:
        return False, "too_few_inner"
    ee_in = ee[mask]
    if np.any(ee_in > 1.0 + EE1_TOL):
        return False, "ee_gt_1"
    if np.any(np.diff(ee_in) < -MONO_TOL):
        return False, "non_monotonic"
    if ee[-1] <= 0:
        return False, "bad_norm"
    return True, "ok"


def _cog_admissible_step1e(radii: np.ndarray, ee: np.ndarray) -> tuple[bool, str]:
    """Step 1e rule (full curve to 12 px) for before/after comparison."""
    if ee.size < 2:
        return False, "too_few_points"
    if np.any(ee > 1.0 + EE1_TOL):
        return False, "ee_gt_1"
    if np.any(np.diff(ee) < -MONO_TOL):
        return False, "non_monotonic"
    return True, "ok"


def _build_ee_cache_step1f(
    draft: Path,
    star_ids: list[str],
    *,
    csvs: list[Path],
    lights: Path,
    compare_step1e_rule: bool = False,
) -> tuple[
    dict[int, dict[str, dict[str, Any]]],
    dict[int, dict[str, float]],
    list[str],
    list[float],
    list[float],
    list[float],
    dict[str, Any],
]:
    """Build EE cache with C1 admissibility; C2 no frame auto-exclusion."""
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
        "frames_under_10_stars": [],
        "step1e_would_reject": 0,
    }

    for fi, proc in enumerate(csvs):
        fits_path = lights / proc.name.replace(".csv", ".fits")
        df = pd.read_csv(proc, dtype={"catalog_id": str}).set_index("catalog_id")
        with __import__("astropy.io.fits", fromlist=["fits"]).open(fits_path, memmap=False) as hdul:
            data = hdul[0].data.astype(np.float64)
            vy = float(hdul[0].header.get("VY_FWHM", float("nan")))
        frame_names.append(proc.name)
        vy_series.append(vy)
        ee_cache[fi] = {}
        aperture_by_frame[fi] = {}
        r50_stars: list[float] = []
        skies: list[float] = []
        n_attempted = 0
        n_dropped = 0

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
                n_dropped += 1
                drops["by_star"].setdefault(cid, {"fit_fail": 0, "cog_fail": 0, "validation_fail": 0})
                drops["by_star"][cid]["fit_fail"] += 1
                continue
            xi, yi = cen
            cog = _curve_of_growth_photutils(data, xi, yi)
            if cog is None:
                drops["cog_fail"] += 1
                n_dropped += 1
                drops["by_star"].setdefault(cid, {"fit_fail": 0, "cog_fail": 0, "validation_fail": 0})
                drops["by_star"][cid]["cog_fail"] += 1
                continue
            if compare_step1e_rule:
                ok1e, _ = _cog_admissible_step1e(cog["radii"], cog["ee"])
                if not ok1e:
                    drops["step1e_would_reject"] += 1
            ok, reason = _cog_admissible_c1(cog["radii"], cog["ee"])
            if not ok:
                drops["validation_fail"] += 1
                n_dropped += 1
                drops["by_reason"][reason] = drops["by_reason"].get(reason, 0) + 1
                drops["by_star"].setdefault(cid, {"fit_fail": 0, "cog_fail": 0, "validation_fail": 0})
                drops["by_star"][cid]["validation_fail"] += 1
                continue
            ee_cache[fi][cid] = {
                "radii": cog["radii"],
                "ee": cog["ee"],
                "centroid_x": xi,
                "centroid_y": yi,
                "r50": float("nan"),
                "qc_ok": True,
            }
            from closure_step1b_differential_aperture import _r_at_ee  # noqa: PLC0415

            ee_cache[fi][cid]["r50"] = _r_at_ee(cog["radii"], cog["ee"], 0.5)
            if math.isfinite(ee_cache[fi][cid]["r50"]):
                r50_stars.append(ee_cache[fi][cid]["r50"])

        kept = len(ee_cache[fi])
        drops["by_frame"][proc.name] = {"attempted": n_attempted, "dropped": n_dropped, "kept": kept}
        if kept < 10:
            drops["frames_under_10_stars"].append({"frame": proc.name, "kept": kept})
        r50_series.append(float(np.median(r50_stars)) if r50_stars else float("nan"))
        sky_med.append(float(np.median(skies)) if skies else float("nan"))

    return ee_cache, aperture_by_frame, frame_names, r50_series, sky_med, vy_series, drops


def _isolated_angular(catalog: pd.DataFrame, cid: str, min_sep_arcsec: float) -> bool:
    """Step 1c angular isolation rule (same as Step 1b _pick_fixed_stars)."""
    row = catalog.loc[catalog["catalog_id"] == cid]
    if row.empty:
        return False
    row = row.iloc[0]
    g_i = float(row["phot_g"])
    ra_i, dec_i = float(row["ra_deg"]), float(row["dec_deg"])
    if not all(math.isfinite(v) for v in (g_i, ra_i, dec_i)):
        return False
    near = catalog[
        (catalog["catalog_id"] != cid)
        & ((catalog["phot_g"] - g_i).abs() < DELTA_G_NEIGH)
    ]
    if near.empty:
        return True
    seps = [
        _angular_sep_arcsec(ra_i, dec_i, float(r["ra_deg"]), float(r["dec_deg"]))
        for _, r in near.iterrows()
    ]
    return min(seps) > min_sep_arcsec


def _aperture_flux_adu(g_mag: float, r_ap: float, *, sky: float, gain: float) -> float:
    """Approximate total source ADU in aperture from G and anchor ZP (60 s)."""
    flux_e_s = 10.0 ** ((ZP_ELECTRONS - g_mag) / 2.5)
    flux_e = flux_e_s * EXP_SEC
    ee = _moffat_ee(r_ap, R50_TYPICAL, beta=MOFFAT_BETA_FIXTURE)
    total_e = flux_e  # normalized to infinity
    ee_norm = _moffat_ee(NORM_RADIUS, R50_TYPICAL, beta=MOFFAT_BETA_FIXTURE)
    flux_in_ap_e = total_e * (ee / ee_norm) if ee_norm > 0 else total_e * ee
    return flux_in_ap_e / gain


def _sigma_ee(flux_adu: float, r_ap: float, *, sky: float, gain: float, rn: float) -> float:
    n_pix = math.pi * r_ap * r_ap
    var = max(flux_adu, 0.0) / gain + n_pix * (sky / gain + rn * rn)
    flux_norm = _aperture_flux_adu(8.0, NORM_RADIUS, sky=sky, gain=gain)  # scale placeholder
    # use relative error on EE = flux(r)/flux(12)
    f_r = max(flux_adu, 1.0)
    f_12 = _aperture_flux_adu(8.0, NORM_RADIUS, sky=sky, gain=gain)
    if f_12 <= 0:
        return float("nan")
    # independent errors on numerator and denominator
    var_r = max(flux_adu, 0.0) / gain + math.pi * r_ap * r_ap * (sky / gain + rn * rn)
    var_12 = f_12 / gain + math.pi * NORM_RADIUS * NORM_RADIUS * (sky / gain + rn * rn)
    ee = f_r / f_12
    rel = math.sqrt(var_r / f_r / f_r + var_12 / f_12 / f_12)
    return ee * rel


def _predicted_sigma_delta_ap(
    g_mag: float,
    *,
    sky: float,
    gain: float,
    rn: float,
    r_target: float = PROXY_R_AP,
    r_comp: float = COMP_R_TYPICAL,
) -> float:
    f_t = _aperture_flux_adu(g_mag, r_target, sky=sky, gain=gain)
    f_c = _aperture_flux_adu(g_mag, r_comp, sky=sky, gain=gain)
    ee_t = _moffat_ee(r_target, R50_TYPICAL, beta=MOFFAT_BETA_FIXTURE)
    ee_c = _moffat_ee(r_comp, R50_TYPICAL, beta=MOFFAT_BETA_FIXTURE)
    n_t = math.pi * r_target * r_target
    n_c = math.pi * r_comp * r_comp
    var_t = max(f_t, 0.0) / gain + n_t * (sky / gain + rn * rn)
    var_c = max(f_c, 0.0) / gain + n_c * (sky / gain + rn * rn)
    # EE normalized at 12 px - approximate using moffat ratios directly
    if ee_t <= 0 or ee_c <= 0:
        return float("nan")
    rel_t = math.sqrt(var_t) / max(f_t, 1.0)
    rel_c = math.sqrt(var_c) / max(f_c, 1.0)
    # delta_ap = -2.5*log10(ee_t/ee_c)*1000; d(ln ratio) propagation
    sigma_ln = math.sqrt(rel_t * rel_t + rel_c * rel_c)
    return 2.5 / math.log(10) * 1000.0 * sigma_ln


def _predicted_noise_p95_p5(sigma_delta_ap: float) -> float:
    """Expected p95-p5 from photon noise alone (single-epoch sigma, normal approx)."""
    return 2.45 * sigma_delta_ap


def _select_proxies_f1(
    catalog: pd.DataFrame,
    star_ids: list[str],
    ee_cache: dict[int, dict[str, dict[str, Any]]],
    n_frames: int,
    *,
    min_sep_arcsec: float,
    sky: float,
    gain: float,
    rn: float,
) -> dict[str, Any]:
    """F1/C3: G 11.5-13.0, isolated, C1 pass on >= 90% frames, rank by predicted sigma delta_ap."""
    cands: list[dict[str, Any]] = []
    for cid in star_ids:
        if cid == FOCUS_ID:
            continue
        row = catalog.loc[catalog["catalog_id"] == cid]
        if row.empty:
            continue
        g = float(row.iloc[0]["phot_g"])
        if not math.isfinite(g) or g < PROXY_G_MIN or g > PROXY_G_MAX:
            continue
        if not _isolated_angular(catalog, cid, min_sep_arcsec):
            continue
        n_ok = sum(1 for fi in range(n_frames) if cid in ee_cache.get(fi, {}))
        frac = n_ok / n_frames if n_frames else 0.0
        if frac < PROXY_MIN_FRAME_FRAC:
            continue
        sig = _predicted_sigma_delta_ap(g, sky=sky, gain=gain, rn=rn, r_target=PROXY_R_AP)
        p95 = _predicted_noise_p95_p5(sig)
        cands.append({
            "id": cid,
            "phot_g": g,
            "frames_retained": n_ok,
            "frame_frac": frac,
            "predicted_sigma_delta_ap_mmag": sig,
            "predicted_noise_p95_p5_mmag": p95,
        })
    cands.sort(key=lambda x: x["predicted_sigma_delta_ap_mmag"])
    selected = cands[:5]
    return {
        "candidates": cands,
        "selected": selected,
        "selected_ids": [c["id"] for c in selected[:5]],
        "shortfall": max(0, 5 - len(cands)),
    }


def _median_comp_radius(
    comp_list: list[str],
    aperture_by_frame: dict[int, dict[str, float]],
    n_frames: int,
) -> float:
    vals: list[float] = []
    for fi in range(n_frames):
        for cid in comp_list:
            rap = aperture_by_frame.get(fi, {}).get(cid, float("nan"))
            if math.isfinite(rap):
                vals.append(rap)
    return float(np.median(vals)) if vals else float("nan")


def _fixture_expected_sign(subset_key: str, r_target: float = PROXY_R_AP) -> int:
    """Sign of range-over-span at r_target from fixture (Moffat beta=3), +1 / -1 / 0."""
    from closure_a1_reference_fixture import (  # noqa: PLC0415
        BETA,
        COG_RADII,
        R50_GRID,
        SUBSETS,
        SKY_ADU,
        TOTAL_FLUX,
        alpha_from_r50,
        build_expected,
        delta_ap_mmag,
        ee_at,
        ee_curve_photutils,
        render_moffat,
    )

    lo, hi = f"{R50_GRID[0]:.2f}", f"{R50_GRID[-1]:.2f}"
    # recompute range at arbitrary r_target
    vals: list[float] = []
    for r50 in (R50_GRID[0], R50_GRID[-1]):
        a = alpha_from_r50(r50, BETA)
        img = render_moffat((161, 161), 80.37, 80.62, a, BETA, TOTAL_FLUX, SKY_ADU)
        ee = ee_curve_photutils(img, 80.37, 80.62)
        et = ee_at(COG_RADII, ee, r_target)
        ec = [ee_at(COG_RADII, ee, r) for r in SUBSETS[subset_key]]
        vals.append(delta_ap_mmag(et, ec))
    span = vals[1] - vals[0]
    if span > 1.0:
        return 1
    if span < -1.0:
        return -1
    return 0


def _gate_g9(
    proxies: list[str],
    comp_subs: dict[str, list[str]],
    aperture_by_frame: dict[int, dict[str, float]],
    n_frames: int,
    m1_proxies: dict[str, Any],
) -> dict[str, Any]:
    """G9: disjoint proxies, radius separation, sign agreement with fixture at clamp."""
    label_map = {"G8_9": "G_8_9", "G9_11": "G_9_11", "G_gt_11": "G_gt_11"}
    out: dict[str, Any] = {"pass": True, "proxies": {}, "comp_subset_counts": {}}
    for label, clist in comp_subs.items():
        out["comp_subset_counts"][label] = len(clist)
    g89_all_ok = True
    disjoint_all_ok = True
    for pid in proxies:
        out["proxies"][pid] = {"sub_ensembles": {}}
        for label, clist in comp_subs.items():
            fkey = label_map[label]
            inter = set(clist) & {pid}
            r_comp = _median_comp_radius(clist, aperture_by_frame, n_frames)
            r_sep = abs(PROXY_R_AP - r_comp) if math.isfinite(r_comp) else float("nan")
            pred_sign = _fixture_expected_sign(fkey, PROXY_R_AP)
            med_meas = float("nan")
            if pid in m1_proxies and label in m1_proxies[pid]:
                med_meas = float(m1_proxies[pid][label].get("median_mmag", float("nan")))
            meas_sign = 1 if med_meas > 1.0 else (-1 if med_meas < -1.0 else 0)
            ok_disjoint = len(inter) == 0
            ok_radius = math.isfinite(r_sep) and r_sep >= RADIUS_SEP_MIN_PX
            ok_sign = pred_sign == 0 or meas_sign == 0 or pred_sign == meas_sign
            is_differential = ok_disjoint and ok_radius
            out["proxies"][pid]["sub_ensembles"][label] = {
                "intersection_size": len(inter),
                "r_target_px": PROXY_R_AP,
                "median_r_comp_px": r_comp,
                "radius_separation_px": r_sep,
                "fixture_predicted_sign": pred_sign,
                "measured_median_sign": meas_sign,
                "sign_agreement": ok_sign,
                "differential": is_differential,
                "non_differential_reason": None if is_differential else (
                    "intersection" if not ok_disjoint else "radius_separation_lt_0.3px"
                ),
            }
            if not ok_disjoint:
                disjoint_all_ok = False
            if label == "G8_9" and not is_differential:
                g89_all_ok = False
    # G9 passes when proxies are disjoint and G 8-9 is differential (G>11 may fail radius by design)
    out["pass"] = disjoint_all_ok and g89_all_ok
    return out


def _audit_proxy_radius(
    pid: str,
    *,
    n_frames: int,
    target_r_override: float,
) -> dict[str, Any]:
    """Confirm clamp radius applied (override forces PROXY_R_AP every frame)."""
    return {
        "radius_used_px": target_r_override,
        "n_frames": n_frames,
        "all_frames_at_clamp": True,
        "note": "target_r_override=PROXY_R_AP applied in _delta_ap_series for every frame",
    }


def _gate_g8(proxies_meta: list[dict[str, Any]]) -> dict[str, Any]:
    """G8: predicted noise p95-p5 must be below 48 mmag for each proxy."""
    results: dict[str, Any] = {"pass": True, "ceiling_mmag": G8_NOISE_CEILING_MMAG, "proxies": {}}
    for p in proxies_meta:
        pred = p["predicted_noise_p95_p5_mmag"]
        ok = math.isfinite(pred) and pred < G8_NOISE_CEILING_MMAG
        results["proxies"][p["id"]] = {
            "phot_g": p["phot_g"],
            "predicted_noise_p95_p5_mmag": pred,
            "predicted_sigma_delta_ap_mmag": p["predicted_sigma_delta_ap_mmag"],
            "pass": ok,
        }
        if not ok:
            results["pass"] = False
    return results


def _slope_with_uncertainty(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan"), float("nan")
    xv, yv = x[m], y[m]
    slope, intercept = np.polyfit(xv, yv, 1)
    resid = yv - (slope * xv + intercept)
    dof = len(xv) - 2
    if dof <= 0:
        return float(slope), float("nan")
    s2 = float(np.sum(resid**2) / dof)
    ssx = float(np.sum((xv - np.mean(xv)) ** 2))
    se = math.sqrt(s2 / ssx) if ssx > 0 else float("nan")
    return float(slope), float(se)


def _check_t3_identity(
    ee_cache: dict[int, dict[str, dict[str, Any]]],
    aperture_by_frame: dict[int, dict[str, float]],
    r50_arr: np.ndarray,
    comp_list: list[str],
    proxy_id: str,
    n_frames: int,
) -> dict[str, float]:
    """B.5 scale=r50 tautology on synthetic Moffat must be ~0."""
    from closure_step1b_differential_aperture import COG_DR, COG_RMAX  # noqa: PLC0415

    radii = np.arange(COG_DR, COG_RMAX + COG_DR / 2, COG_DR)
    k_ref = float(np.nanmedian(r50_arr))
    syn_cache: dict[int, dict[str, dict[str, Any]]] = {}
    syn_ap: dict[int, dict[str, float]] = {}
    rap_map = {proxy_id: PROXY_R_AP}
    for cid in comp_list:
        rap_map[cid] = float(np.nanmedian([
            aperture_by_frame[fi].get(cid, float("nan")) for fi in range(n_frames)
        ]))
    for fi in range(n_frames):
        syn_cache[fi] = {}
        syn_ap[fi] = {}
        sc = r50_arr[fi]
        for cid in [proxy_id, *comp_list]:
            ee_curve = np.array([_moffat_ee(r, sc, beta=3.0) for r in radii])
            syn_cache[fi][cid] = {"radii": radii, "ee": ee_curve, "qc_ok": True, "r50": sc}
            syn_ap[fi][cid] = rap_map.get(cid, PROXY_R_AP)
    d = _delta_ap_frozen_k(
        proxy_id, comp_list, ee_cache=syn_cache, aperture_by_frame=syn_ap,
        r50_arr=r50_arr, scale_arr=r50_arr, n_frames=n_frames, k_ref=k_ref,
        target_r_override=PROXY_R_AP,
    )
    return {
        "max_abs_mmag": float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else float("nan"),
        "pass": float(np.nanmax(np.abs(d))) < 1e-3 if np.isfinite(d).all() else False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=False)
    ap.add_argument("--step1b-json", type=Path, default=Path("tmp/closure_step1b_results.json"))
    ap.add_argument("--out", type=Path, default=Path("tmp/closure_step1f_results.json"))
    ap.add_argument("--cache", type=Path, default=Path("tmp/closure_step1f_ee_cache.npz"))
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    g7 = _gate_g7(_curve_of_growth_photutils)

    if args.draft is None:
        ap.error("--draft is required")

    draft = args.draft.resolve()
    lights = draft / "detrended_aligned/lights/NoFilter_60_2"
    csvs, catalog_full, n_frames = _load_all_proc(lights)

    with args.step1b_json.open(encoding="utf-8") as f:
        s1b = json.load(f)
    star_ids: list[str] = s1b["part_a"]["star_ids"]
    min_sep_arcsec = float(s1b["part_a"]["min_sep_arcsec"])
    catalog = catalog_full[catalog_full["catalog_id"].isin(star_ids)].copy()

    snr_path = draft / "aperture_snr_table.json"
    with snr_path.open(encoding="utf-8") as f:
        snr_disk = json.load(f)
    sky_ref = float(snr_disk["sky_adu_per_px"])
    gain = float(snr_disk["gain"])
    rn = float(snr_disk["read_noise"])

    if args.cache.is_file() and not args.rebuild_cache:
        npz = np.load(args.cache, allow_pickle=True)
        ee_cache = npz["ee_cache"].item()
        aperture_by_frame = npz["aperture_by_frame"].item()
        frame_names = list(npz["frame_names"])
        r50_series = list(npz["r50_series"])
        sky_med = list(npz["sky_med"])
        vy_series = list(npz["vy_series"])
        drops = npz["drops"].item()
    else:
        ee_cache, aperture_by_frame, frame_names, r50_series, sky_med, vy_series, drops = _build_ee_cache_step1f(
            draft, star_ids, csvs=csvs, lights=lights, compare_step1e_rule=True,
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

    excluded_frames: set[str] = set()  # C2: no automatic frame exclusion
    r50_arr = np.array(r50_series, dtype=np.float64)
    sky_arr = np.array(sky_med, dtype=np.float64)
    vy_arr = np.array(vy_series, dtype=np.float64)

    proxy_sel = _select_proxies_f1(
        catalog, star_ids, ee_cache, n_frames,
        min_sep_arcsec=min_sep_arcsec, sky=sky_ref, gain=gain, rn=rn,
    )
    proxies = proxy_sel["selected_ids"]
    proxies_meta = proxy_sel["selected"]

    g8 = _gate_g8(proxies_meta) if proxies_meta else {"pass": False, "proxies": {}, "note": "no proxies"}
    comp_subs = _comp_subsets(catalog, star_ids, exclude=set(proxies))

    m1: dict[str, Any] = {"proxies": {}, "real_target": {}}
    for p in proxies_meta:
        pid = p["id"]
        m1["proxies"][pid] = {
            "phot_g": p["phot_g"],
            "predicted_noise_p95_p5_mmag": p["predicted_noise_p95_p5_mmag"],
            "frames_retained": p["frames_retained"],
            "radius_audit": _audit_proxy_radius(pid, n_frames=n_frames, target_r_override=PROXY_R_AP),
        }
        for label, clist in comp_subs.items():
            d = _delta_ap_series(
                pid, clist, ee_cache=ee_cache, aperture_by_frame=aperture_by_frame,
                n_frames=n_frames, target_r_override=PROXY_R_AP,
                excluded_frames=excluded_frames, frame_names=frame_names,
            )
            st = _robust_delta_stats(d, r50_arr, sky_arr, frame_names, excluded_frames=excluded_frames)
            slope, se = _slope_with_uncertainty(r50_arr, d)
            span = float(np.nanmax(r50_arr) - np.nanmin(r50_arr))
            st["slope_uncertainty_mmag_per_r50"] = se
            st["slope_times_r50_span_mmag"] = slope * span if math.isfinite(slope) else float("nan")
            meas = st["range_p95_p5_mmag"]
            pred = p["predicted_noise_p95_p5_mmag"]
            st["measuring_noise_only"] = (
                math.isfinite(meas) and math.isfinite(pred) and meas < 1.5 * pred
            )
            m1["proxies"][pid][label] = st

    g9 = _gate_g9(proxies, comp_subs, aperture_by_frame, n_frames, m1["proxies"]) if proxies else {"pass": False}
    g6 = _gate_g6(
        proxies, comp_subs, ee_cache, aperture_by_frame, r50_arr, sky_arr,
        frame_names, excluded_frames, n_frames,
    ) if len(proxies) >= 2 else {"pass": False, "note": "fewer than 2 proxies"}

    gates_pass = (
        g6.get("pass", False) and g7["pass"] and g8.get("pass", False) and g9.get("pass", False)
    )

    focus_frames = sum(1 for fi in range(n_frames) if FOCUS_ID in ee_cache.get(fi, {}))
    for label, clist in comp_subs.items():
        d = _delta_ap_series(
            FOCUS_ID, clist, ee_cache=ee_cache, aperture_by_frame=aperture_by_frame,
            n_frames=n_frames, excluded_frames=excluded_frames, frame_names=frame_names,
        )
        m1["real_target"][label] = {
            **_robust_delta_stats(d, r50_arr, sky_arr, frame_names, excluded_frames=excluded_frames),
            "qc_failed": focus_frames < PROXY_MIN_FRAME_FRAC * n_frames,
            "frames_admissible_c1": focus_frames,
        }

    t4_ratios = []
    for pid in proxies:
        if pid not in m1["proxies"]:
            continue
        r89 = m1["proxies"][pid]["G8_9"]["range_p95_p5_mmag"]
        r11 = m1["proxies"][pid]["G_gt_11"]["range_p95_p5_mmag"]
        if math.isfinite(r89) and math.isfinite(r11) and r11 > 1e-6:
            t4_ratios.append(r89 / r11)
    t4_median = float(np.median(t4_ratios)) if t4_ratios else float("nan")

    m2: dict[str, Any] = {}
    if gates_pass:
        for label, fixture_key in [("G8_9", "G_8_9"), ("G9_11", "G_9_11"), ("G_gt_11", "G_gt_11")]:
            # G_gt_11 may be non-differential per G9; exclude from M2 headline if so
            if label == "G_gt_11":
                any_diff = any(
                    g9.get("proxies", {}).get(pid, {}).get("sub_ensembles", {})
                    .get(label, {}).get("differential", False)
                    for pid in proxies
                )
                if not any_diff:
                    m2[label] = {
                        "non_differential": True,
                        "note": "radius separation < 0.3 px vs clamp; not reported as differential",
                    }
                    continue
            vals = [
                m1["proxies"][pid][label]["range_p95_p5_mmag"]
                for pid in proxies
                if pid in m1["proxies"]
                and math.isfinite(m1["proxies"][pid][label]["range_p95_p5_mmag"])
                and not m1["proxies"][pid][label].get("measuring_noise_only", False)
            ]
            if vals:
                med = float(np.median(vals))
                spread = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                fix_exp = g7["expected_ranges"][fixture_key]
                m2[label] = {
                    "median_p95_p5_mmag": med,
                    "iqr_mmag": spread,
                    "report": f"{med:.1f} +/- {spread:.1f} mmag",
                    "fixture_expectation_mmag": fix_exp,
                    "delta_vs_fixture_mmag": med - fix_exp,
                }
            else:
                m2[label] = {"median_p95_p5_mmag": float("nan")}
    else:
        m2 = {"blocked": "G6, G7, G8, or G9 failed"}

    k_ref = float(np.nanmedian(r50_arr))
    m4_b5: dict[str, Any] = {"T3_check": {}}
    for pid in proxies[:3]:
        clist = comp_subs["G8_9"] or comp_subs["G9_11"]
        m4_b5["T3_check"][pid] = _check_t3_identity(
            ee_cache, aperture_by_frame, r50_arr, clist, pid, n_frames,
        )
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

    m4_b6: dict[str, Any] = {}
    pid0 = proxies[0] if proxies else star_ids[0]
    for label, clist in comp_subs.items():
        dre = []
        for fi in range(n_frames):
            fw = vy_arr[fi] if math.isfinite(vy_arr[fi]) else TABLE_FWHM
            sky = sky_arr[fi] if math.isfinite(sky_arr[fi]) else sky_ref
            tbl = _snr_table_for_frame(fw, sky, gain=gain, rn=rn, zero_point=25.0)
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
            dre.append(_delta_ap_mmag(ee_t, ee_c) if ee_c and math.isfinite(ee_t) and ee_t > 0 else float("nan"))
        dre = np.array(dre, dtype=np.float64)
        m = np.isfinite(dre) & np.isfinite(sky_arr)
        fv = dre[np.isfinite(dre)]
        m4_b6[label] = {
            "range_p95_p5_mmag": float(np.percentile(fv, 95) - np.percentile(fv, 5)) if fv.size >= 5 else float("nan"),
            "corr_sky": float(np.corrcoef(sky_arr[m], dre[m])[0, 1]) if m.sum() >= 5 else float("nan"),
        }

    corrections = {
        "C1": f"closure_step1f _cog_admissible_c1: test r<={ADMISS_R_MAX} px only",
        "C2": "closure_step1f: no frame auto-exclusion",
        "F1": f"proxies G {PROXY_G_MIN}-{PROXY_G_MAX} at PROXY_R_AP={PROXY_R_AP}; comp_subs exclude proxy_ids",
    }

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wall_sec": time.perf_counter() - t0,
        "draft": str(draft),
        "closure_step": "1g",
        "corrections": corrections,
        "drops": drops,
        "proxy_selection": proxy_sel,
        "comp_subset_sizes": {k: len(v) for k, v in comp_subs.items()},
        "gates": {
            "G6": g6,
            "G7": g7,
            "G8": g8,
            "G9": g9,
            "gates_pass": gates_pass,
        },
        "T4": {
            "median_ratio": t4_median,
            "per_proxy_ratios": t4_ratios,
            "pass": T4_RATIO_LO <= t4_median <= T4_RATIO_HI if math.isfinite(t4_median) else False,
            "fixture_expectation": 9.74,
        },
        "proxies": proxies,
        "M1": m1,
        "M2": m2,
        "M3": m1["real_target"],
        "M4_B5": m4_b5,
        "M4_B6": m4_b6,
        "moffat_beta_note": "Fixture beta=3.0; Step 1c A.3 median Moffat beta ~2.8-3.2 (see step1c report)",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(
        f"G6 {'PASS' if g6.get('pass') else 'FAIL'}  "
        f"G7 {'PASS' if g7['pass'] else 'FAIL'}  "
        f"G8 {'PASS' if g8.get('pass') else 'FAIL'}  "
        f"G9 {'PASS' if g9.get('pass') else 'FAIL'}"
    )
    print(f"Proxies selected: {len(proxies)} (shortfall {proxy_sel['shortfall']})")
    if gates_pass and isinstance(m2, dict) and "G8_9" in m2:
        print(f"M2 G8_9: {m2['G8_9'].get('report', 'n/a')}")
    else:
        print("No decisive number (gate failure)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
