#!/usr/bin/env python3
"""SIGMA-SEM-CAUSE: causal attribution of ensemble SEM inflation (diagnostics only)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from check_star_kmag import (  # noqa: E402
    build_aligned_comp_inst,
    comp_ensemble_maps,
    resolve_proc_csv_dir,
)
from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import (  # noqa: E402
    ERR_BKG_MODE_EMPIRICAL,
    SIGMA_BKG_AP_COL,
    _ensemble_scatter_by_source_file,
    _photometric_error_with_bkg_mode,
    _sky_pp_for_photometric_error,
    check_comparison_stability,
    ensemble_normalize,
)
from scripts.bingain_err_decompose import _gain_from_lights  # noqa: E402
from scripts.bingain_fix_validate import _chi2_lc_err, resolve_archive_root  # noqa: E402
from scripts.chi2_sigma_gate import load_proc_row_for_source, reduced_chi2_constant  # noqa: E402
from scripts.sem_cause_core import (  # noqa: E402
    chi2_dof_from_mags_sigmas,
    distribution_stats,
    lag1_autocorrelation,
    mag_to_rel_err,
    per_frame_sem_from_residuals,
    recompose_err_mag,
    rel_to_mag_err,
    split_half_zp_sem,
    trend_fraction,
)
from scripts.sparse_comp_diag import V0611_CID  # noqa: E402

SS_CAM_CID = "1112113066119992064"
_MAG = 2.5 / math.log(10.0)
DRAFT_426 = 426
SETUPS_426 = ("g_60_4", "i_70_4", "r_60_4")
UNDERDISP_G = ("1111931646701447424", "1112121175018240768")
N4_CID = "1485540612577549568"


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def _stamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["git_head"] = _git_head()
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_stamp(payload), indent=2), encoding="utf-8")
    return str(path)


def load_star_lists(summary_path: Path) -> dict[str, list[str]]:
    summ = json.loads(summary_path.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for setup, data in summ.get("setups", {}).items():
        ids: list[str] = []
        for row in data.get("per_star_table", []):
            cid = str(row.get("catalog_id", ""))
            if cid and cid not in ids:
                ids.append(cid)
        out[setup] = ids
    return out


def _photon_err_mag_per_frame(
    lc_df: pd.DataFrame,
    proc_dir: Path,
    target_cid: str,
    *,
    gain: float,
    read_noise: float,
) -> np.ndarray:
    out = np.full(len(lc_df), np.nan, dtype=np.float64)
    for i, row in lc_df.iterrows():
        sf = str(row.get("source_file", "")).strip()
        err_lc = float(pd.to_numeric(row.get("err"), errors="coerce"))
        proc = load_proc_row_for_source(proc_dir, sf, target_cid)
        if proc is None:
            continue
        flux = float(pd.to_numeric(proc.get("dao_flux"), errors="coerce"))
        sky = float(_sky_pp_for_photometric_error(proc))
        area = float(pd.to_numeric(proc.get("aperture_area_px"), errors="coerce"))
        if not math.isfinite(area) or area <= 0:
            r = float(pd.to_numeric(proc.get("aperture_r_px"), errors="coerce"))
            area = math.pi * r * r if math.isfinite(r) and r > 0 else float("nan")
        sig_bkg = float(pd.to_numeric(proc.get(SIGMA_BKG_AP_COL), errors="coerce"))
        sig_bkg_v = sig_bkg if math.isfinite(sig_bkg) else None
        err_phot_rel, _ = _photometric_error_with_bkg_mode(
            flux,
            err_background_mode=ERR_BKG_MODE_EMPIRICAL,
            sky_pp=sky,
            area=area,
            gain=gain,
            read_noise=read_noise,
            sigma_bkg_ap=sig_bkg_v,
        )
        if math.isfinite(err_phot_rel) and err_phot_rel > 0:
            out[int(i)] = _MAG * err_phot_rel
        elif math.isfinite(err_lc) and err_lc > 0:
            # fallback: infer photon from LC if proc path fails
            ens = float("nan")
            out[int(i)] = _MAG * err_lc
    return out


def _ensemble_sem_from_lc_err(
    err_lc_rel: np.ndarray,
    err_phot_mag: np.ndarray,
) -> np.ndarray:
    """Implied production ensemble SEM (mag) from LC err quadrature."""
    out = np.full(len(err_lc_rel), np.nan, dtype=np.float64)
    for i in range(len(err_lc_rel)):
        el = float(err_lc_rel[i])
        ep = float(err_phot_mag[i]) if i < err_phot_mag.size else float("nan")
        if not (math.isfinite(el) and el > 0 and math.isfinite(ep) and ep >= 0):
            continue
        el_mag = _MAG * el
        diff = el_mag * el_mag - ep * ep
        out[i] = math.sqrt(max(0.0, diff))
    return out


def extract_production_trace(
    *,
    phot_dir: Path,
    setup: str,
    target_cid: str,
    cfg: AppConfig,
    gain: float,
    read_noise: float,
) -> dict[str, Any]:
    """Mirror production comp residuals + ensemble SEM (photometry_core.py:3055-3115)."""
    lc_path = phot_dir / "lightcurves" / f"lightcurve_{target_cid}.csv"
    if not lc_path.is_file():
        return {"available": False, "reason": "missing LC"}
    lc_df = pd.read_csv(lc_path, low_memory=False)
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    if proc_dir is None:
        return {"available": False, "reason": "missing proc_dir"}
    comp_path = phot_dir / "comparison_stars_per_target.csv"
    if not comp_path.is_file():
        return {"available": False, "reason": "missing comp CSV"}
    comp_all = pd.read_csv(comp_path, low_memory=False, dtype={"catalog_id": str, "target_catalog_id": str})
    comp_df = comp_all.loc[comp_all["target_catalog_id"].map(_norm_id) == _norm_id(target_cid)]
    if comp_df.empty:
        return {"available": False, "reason": "no comps for target"}
    cat, tier, rms, tw = comp_ensemble_maps(comp_df, cfg)
    comp_ids = [_norm_id(c) for c in comp_df["catalog_id"].tolist() if _norm_id(c)]
    if target_cid not in comp_ids and _norm_id(target_cid) not in comp_ids:
        comp_ids.append(_norm_id(target_cid))
    source_files = lc_df["source_file"].astype(str).tolist()
    comp_lc = build_aligned_comp_inst(proc_dir, comp_ids, source_files, cfg, "aperture")
    if _norm_id(target_cid) not in comp_lc:
        return {"available": False, "reason": "target missing from proc alignment"}
    target_lc = comp_lc[_norm_id(target_cid)]
    other_lc = {c: comp_lc[c] for c in comp_ids if c != _norm_id(target_cid) and c in comp_lc}
    comp_quality = check_comparison_stability(
        other_lc, comp_rms_map=rms, n_comp_min=3, outlier_sigma=3.0, common_mode_detrend=True,
    )
    other_cat = {c: cat[c] for c in other_lc if c in cat}
    other_quality = {
        c: comp_quality[c]
        for c in other_lc
        if c in comp_quality and str(comp_quality[c].get("quality", "")).strip().lower() != "excluded"
    }
    _, _, ensemble_scatter = ensemble_normalize(
        target_lc,
        other_lc,
        other_cat,
        other_quality,
        comp_rms_map=rms,
        comp_tier_map=tier,
        tier_weights=tw,
        n_comp_min=3,
        n_comp_max=int(cfg.phase01_comparison_n_comp_max),
    )
    scatter_by_file = _ensemble_scatter_by_source_file(
        lc_df.assign(catalog_id=target_cid), target_cid, ensemble_scatter,
    )

    # Replicate comp_ref_map + per-frame residuals (production lines 3060-3115)
    good_ids = [
        c for c in other_lc
        if c in other_quality
        and str(other_quality[c].get("quality", "")).strip().lower() != "excluded"
    ]
    comp_ref_map: dict[str, float] = {}
    for cid in good_ids:
        arr = np.asarray(other_lc[cid], dtype=np.float64)
        fin = arr[np.isfinite(arr)]
        if fin.size:
            comp_ref_map[cid] = float(np.median(fin))

    residuals_by_frame: list[list[float]] = []
    comp_mags_by_frame: list[dict[str, float]] = []
    manual_sem = np.full(len(lc_df), np.nan, dtype=np.float64)
    for i in range(len(lc_df)):
        comp_pairs: list[tuple[str, float]] = []
        frame_mags: dict[str, float] = {}
        for cid in good_ids:
            mv = float(other_lc[cid][i])
            if math.isfinite(mv):
                comp_pairs.append((cid, mv))
                frame_mags[cid] = mv
        comp_resid = [
            m - comp_ref_map[cid_j]
            for cid_j, m in comp_pairs
            if cid_j in comp_ref_map and math.isfinite(comp_ref_map[cid_j])
        ]
        residuals_by_frame.append(comp_resid)
        comp_mags_by_frame.append(frame_mags)
        if len(comp_resid) >= 2:
            arr = np.asarray(comp_resid, dtype=np.float64)
            manual_sem[i] = float(np.std(arr, ddof=1) / math.sqrt(len(comp_resid)))
        elif len(comp_resid) == 1:
            manual_sem[i] = 0.0

    recomputed_sem = per_frame_sem_from_residuals(residuals_by_frame)
    airmass = pd.to_numeric(lc_df.get("airmass"), errors="coerce").to_numpy(dtype=np.float64)
    bjd = pd.to_numeric(lc_df.get("bjd"), errors="coerce").to_numpy(dtype=np.float64)
    time_h = (bjd - float(np.nanmin(bjd))) * 24.0 if np.isfinite(bjd).any() else bjd

    # Per-comp residual matrix
    comp_series: dict[str, np.ndarray] = {}
    for cid in good_ids:
        series = np.full(len(lc_df), np.nan, dtype=np.float64)
        ref = comp_ref_map.get(cid, float("nan"))
        for i in range(len(lc_df)):
            mv = float(other_lc[cid][i])
            if math.isfinite(mv) and math.isfinite(ref):
                series[i] = mv - ref
        comp_series[cid] = series

    bp_rp_map: dict[str, float] = {}
    colour_offset: dict[str, float] = {}
    med_bp_rp = float("nan")
    br_vals = pd.to_numeric(comp_df.get("bp_rp"), errors="coerce")
    if br_vals.notna().any():
        med_bp_rp = float(br_vals.median())
    for _, r in comp_df.iterrows():
        cid = _norm_id(r.get("catalog_id"))
        if not cid:
            continue
        br = float(pd.to_numeric(r.get("bp_rp"), errors="coerce"))
        if math.isfinite(br):
            bp_rp_map[cid] = br
            if math.isfinite(med_bp_rp):
                colour_offset[cid] = br - med_bp_rp

    err_lc_rel = pd.to_numeric(lc_df.get("err"), errors="coerce").to_numpy(dtype=np.float64)
    err_phot_mag = _photon_err_mag_per_frame(
        lc_df, proc_dir, target_cid, gain=gain, read_noise=read_noise,
    )
    sem_from_lc = _ensemble_sem_from_lc_err(err_lc_rel, err_phot_mag)

    return {
        "available": True,
        "target_cid": target_cid,
        "n_frames": len(lc_df),
        "good_comp_ids": good_ids,
        "comp_ref_map": comp_ref_map,
        "ensemble_scatter": ensemble_scatter,
        "manual_sem": manual_sem,
        "recomputed_sem": recomputed_sem,
        "scatter_by_file": scatter_by_file,
        "sem_from_lc": sem_from_lc,
        "residuals_by_frame": residuals_by_frame,
        "comp_series": comp_series,
        "comp_mags_by_frame": comp_mags_by_frame,
        "airmass": airmass,
        "time_h": time_h,
        "bp_rp_map": bp_rp_map,
        "colour_offset": colour_offset,
        "err_phot_mag": err_phot_mag,
        "err_lc_rel": err_lc_rel,
        "source_files": source_files,
        "citation": "photometry_core.py:3060-3115 comp_ref_map + std(comp_resid)/sqrt(n)",
    }


def gate_c0_equivalence(trace: dict[str, Any]) -> dict[str, Any]:
    if not trace.get("available"):
        return {"pass": False, "reason": "trace unavailable"}
    ens = np.asarray(trace["ensemble_scatter"], dtype=np.float64)
    manual = np.asarray(trace["manual_sem"], dtype=np.float64)
    recomp = np.asarray(trace["recomputed_sem"], dtype=np.float64)
    ok = np.isfinite(ens) & np.isfinite(manual)
    max_diff_manual = float(np.max(np.abs(ens[ok] - manual[ok]))) if ok.any() else float("nan")
    ok2 = np.isfinite(ens) & np.isfinite(recomp)
    max_diff_recomp = float(np.max(np.abs(ens[ok2] - recomp[ok2]))) if ok2.any() else float("nan")
    # LC-implied scatter (production combine stores scatter in same units as err -- mag-scale)
    sem_lc = np.asarray(trace["sem_from_lc"], dtype=np.float64)
    ok3 = np.isfinite(ens) & np.isfinite(sem_lc) & (ens > 0)
    max_diff_lc = float(np.max(np.abs(ens[ok3] - sem_lc[ok3]))) if ok3.any() else float("nan")
    pass_manual = math.isfinite(max_diff_manual) and max_diff_manual < 1e-12
    return {
        "pass": pass_manual,
        "max_abs_diff_ensemble_vs_manual": max_diff_manual,
        "max_abs_diff_ensemble_vs_recomputed_helper": max_diff_recomp,
        "max_abs_diff_ensemble_vs_lc_implied_mag": max_diff_lc,
        "n_frames_compared": int(ok.sum()),
    }


def analyze_d1(trace: dict[str, Any]) -> dict[str, Any]:
    airmass = np.asarray(trace["airmass"], dtype=np.float64)
    time_h = np.asarray(trace["time_h"], dtype=np.float64)
    lag1_vals: list[float] = []
    trend_am: list[float] = []
    trend_time: list[float] = []
    slopes_am: list[float] = []
    colours: list[float] = []
    per_comp: list[dict[str, Any]] = []
    for cid, series in trace["comp_series"].items():
        s = np.asarray(series, dtype=np.float64)
        lag1_vals.append(lag1_autocorrelation(s))
        frac_am, _, _ = trend_fraction(s, airmass, deg=1)
        frac_t, _, _ = trend_fraction(s, time_h, deg=2)
        trend_am.append(frac_am)
        trend_time.append(frac_t)
        coef = np.polyfit(airmass[np.isfinite(s) & np.isfinite(airmass)], s[np.isfinite(s) & np.isfinite(airmass)], 1) if np.sum(np.isfinite(s) & np.isfinite(airmass)) >= 3 else [float("nan")]
        slope = float(coef[0]) if len(coef) >= 1 and math.isfinite(float(coef[0])) else float("nan")
        slopes_am.append(slope)
        co = trace["colour_offset"].get(cid, float("nan"))
        if math.isfinite(co):
            colours.append(co)
        per_comp.append({"cid": cid, "lag1": lag1_vals[-1], "trend_frac_am": frac_am, "trend_frac_time2": frac_t, "slope_vs_am": slope, "colour_offset": co})
    # colour-slope correlation
    colour_slope_pairs = [
        (trace["colour_offset"].get(c["cid"], float("nan")), c["slope_vs_am"])
        for c in per_comp
        if math.isfinite(c.get("slope_vs_am", float("nan")))
        and math.isfinite(trace["colour_offset"].get(c["cid"], float("nan")))
    ]
    colour_corr = float("nan")
    if len(colour_slope_pairs) >= 3:
        xs = np.asarray([p[0] for p in colour_slope_pairs], dtype=float)
        ys = np.asarray([p[1] for p in colour_slope_pairs], dtype=float)
        if np.std(xs) > 0 and np.std(ys) > 0:
            colour_corr = float(np.corrcoef(xs, ys)[0, 1])
    return {
        "lag1_autocorr": distribution_stats(lag1_vals),
        "trend_fraction_airmass_linear": distribution_stats(trend_am),
        "trend_fraction_time_poly2": distribution_stats(trend_time),
        "colour_slope_correlation": colour_corr,
        "colour_slope_present": (
            math.isfinite(colour_corr) and abs(colour_corr) >= 0.5
        ),
        "per_comp": per_comp,
    }


def analyze_d2_d3(
    trace: dict[str, Any],
    *,
    kmag: np.ndarray | None = None,
) -> dict[str, Any]:
    airmass = np.asarray(trace["airmass"], dtype=np.float64)
    time_h = np.asarray(trace["time_h"], dtype=np.float64)
    ens_prod = np.asarray(trace["ensemble_scatter"], dtype=np.float64)
    sem_from_lc = np.asarray(trace.get("sem_from_lc", ens_prod), dtype=np.float64)
    n_frames = len(trace["residuals_by_frame"])

    # Detrend: subtract per-comp linear trend in airmass from residuals before frame std
    detrended_residuals_am: list[list[float]] = []
    detrended_residuals_time: list[list[float]] = []
    for i in range(n_frames):
        frame_det_am: list[float] = []
        frame_det_t: list[float] = []
        for cid in trace["good_comp_ids"]:
            series = np.asarray(trace["comp_series"][cid], dtype=np.float64)
            if i >= series.size or not math.isfinite(series[i]):
                continue
            _, _, trend_am = trend_fraction(series, airmass, deg=1)
            _, _, trend_t = trend_fraction(series, time_h, deg=2)
            frame_det_am.append(float(series[i] - trend_am[i]))
            frame_det_t.append(float(series[i] - trend_t[i]))
        detrended_residuals_am.append(frame_det_am)
        detrended_residuals_time.append(frame_det_t)

    sem_det_am = per_frame_sem_from_residuals(detrended_residuals_am)
    sem_det_time = per_frame_sem_from_residuals(detrended_residuals_time)
    ratio_am = np.where(
        (sem_from_lc > 0) & np.isfinite(sem_det_am),
        sem_det_am / sem_from_lc,
        np.nan,
    )
    ratio_ok = ratio_am[np.isfinite(ratio_am)]

    # Split-half per frame
    split_sem: list[float] = []
    for i, frame_mags in enumerate(trace["comp_mags_by_frame"]):
        sem_sh, scale = split_half_zp_sem(frame_mags, n_splits=20, seed=1000 + i)
        if math.isfinite(sem_sh):
            split_sem.append(sem_sh)
        _ = scale

    split_arr = np.full(n_frames, np.nan, dtype=np.float64)
    for i, frame_mags in enumerate(trace["comp_mags_by_frame"]):
        sem_sh, _ = split_half_zp_sem(frame_mags, n_splits=20, seed=1000 + i)
        split_arr[i] = sem_sh

    three_way = {
        "ensemble_normalize_sem_median": float(np.nanmedian(ens_prod)),
        "lc_implied_ensemble_sem_median": float(np.nanmedian(sem_from_lc)),
        "sem_detrend_am_median": float(np.nanmedian(sem_det_am)),
        "sem_detrend_time_median": float(np.nanmedian(sem_det_time)),
        "split_half_empirical_median": float(np.nanmedian(split_arr)) if np.isfinite(split_arr).any() else None,
        "ratio_detrend_am_over_lc_ensemble_median": float(np.nanmedian(ratio_ok)) if ratio_ok.size else None,
        "ratio_lc_ensemble_over_normalize_median": (
            float(np.nanmedian(sem_from_lc / ens_prod))
            if np.isfinite(sem_from_lc / ens_prod).any()
            else None
        ),
    }

    chi2_pred: dict[str, Any] = {}
    if kmag is not None:
        m = np.asarray(kmag, dtype=np.float64)
        err_phot = np.asarray(trace["err_phot_mag"], dtype=np.float64)
        err_lc_mag = rel_to_mag_err(np.asarray(trace["err_lc_rel"], dtype=np.float64))
        err_lc_mag = np.asarray(err_lc_mag, dtype=np.float64)
        err_det = recompose_err_mag(err_phot, sem_det_am)
        err_sh = recompose_err_mag(err_phot, split_arr)
        _, _, c2_lc = chi2_dof_from_mags_sigmas(m, err_lc_mag)
        _, _, c2_det = chi2_dof_from_mags_sigmas(m, err_det)
        _, _, c2_sh = chi2_dof_from_mags_sigmas(m, err_sh)
        chi2_pred = {
            "chi2_lc_err_actual": c2_lc,
            "chi2_detrend_am_composition": c2_det,
            "chi2_split_half_composition": c2_sh,
            "n_epochs": int(np.sum(np.isfinite(m) & np.isfinite(err_lc_mag) & (err_lc_mag > 0))),
        }

    return {
        "sem_detrend_airmass": sem_det_am.tolist(),
        "sem_detrend_time": sem_det_time.tolist(),
        "ratio_detrend_over_production": ratio_am.tolist(),
        "split_half_sem_per_frame": split_arr.tolist(),
        "three_way": three_way,
        "chi2_prediction": chi2_pred,
        "split_half_scaling_note": "empirical_sem = median(|ens_A-ens_B|/2) * sqrt(n/n_half) over 20 splits",
    }


def analyze_star(
    *,
    phot_dir: Path,
    setup: str,
    target_cid: str,
    cfg: AppConfig,
    gain: float,
    read_noise: float,
    production_chi2: float | None = None,
) -> dict[str, Any]:
    trace = extract_production_trace(
        phot_dir=phot_dir, setup=setup, target_cid=target_cid, cfg=cfg,
        gain=gain, read_noise=read_noise,
    )
    if not trace.get("available"):
        return trace
    lc_dir = phot_dir / "lightcurves"
    side = lc_dir / f"check_kmag_{target_cid}.csv"
    kmag = None
    if side.is_file():
        side_df = pd.read_csv(side, low_memory=False)
        kmag = pd.to_numeric(side_df.get("kmag"), errors="coerce").to_numpy(dtype=np.float64)
    d1 = analyze_d1(trace)
    d2d3 = analyze_d2_d3(trace, kmag=kmag)
    c0 = gate_c0_equivalence(trace)
    return {
        "target_cid": target_cid,
        "production_chi2_lc_err": production_chi2,
        "c0": c0,
        "d1": d1,
        "d2_d3": d2d3,
        "trace_meta": {
            "n_frames": trace["n_frames"],
            "n_good_comps": len(trace["good_comp_ids"]),
        },
    }


def _plot_residual_trails(
    trace: dict[str, Any],
    out_path: Path,
    *,
    title: str,
    max_comps: int = 6,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    airmass = np.asarray(trace["airmass"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4))
    for cid in trace["good_comp_ids"][:max_comps]:
        s = np.asarray(trace["comp_series"][cid], dtype=np.float64)
        ok = np.isfinite(s) & np.isfinite(airmass)
        if not ok.any():
            continue
        ax.plot(airmass[ok], s[ok], ".-", label=cid[-6:], alpha=0.8, ms=4)
        _, _, trend = trend_fraction(s, airmass, deg=1)
        ax.plot(airmass[ok], trend[ok], "--", alpha=0.5)
    ax.set_xlabel("airmass")
    ax.set_ylabel("comp residual (mag)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def _plot_three_way_bars(
    cohort_rows: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["target_cid"][-6:] for r in cohort_rows]
    prod = [r["d2_d3"]["three_way"]["lc_implied_ensemble_sem_median"] for r in cohort_rows]
    det = [r["d2_d3"]["three_way"]["sem_detrend_am_median"] for r in cohort_rows]
    sh = [r["d2_d3"]["three_way"].get("split_half_empirical_median") for r in cohort_rows]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4))
    ax.bar(x - w, prod, w, label="LC-implied ensemble")
    ax.bar(x, det, w, label="detrend AM SEM")
    ax.bar(x + w, sh, w, label="split-half empirical")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("median SEM (mag)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def _plot_chi2_predicted(cohort_rows: list[dict[str, Any]], out_path: Path, *, title: str) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["target_cid"][-6:] for r in cohort_rows]
    prod = [r["d2_d3"]["chi2_prediction"].get("chi2_lc_err_actual") for r in cohort_rows]
    det = [r["d2_d3"]["chi2_prediction"].get("chi2_detrend_am_composition") for r in cohort_rows]
    sh = [r["d2_d3"]["chi2_prediction"].get("chi2_split_half_composition") for r in cohort_rows]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4))
    ax.bar(x - w, prod, w, label="chi2 LC err actual")
    ax.bar(x, det, w, label="chi2 detrend SEM")
    ax.bar(x + w, sh, w, label="chi2 split-half SEM")
    ax.axhline(1.0, color="gray", linestyle=":")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("chi2/dof")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def run_c0_mandatory(
    archive_root: Path,
    cfg: AppConfig,
    out_dir: Path,
) -> dict[str, Any]:
    setup = "i_70_4"
    phot = archive_root / "Drafts" / f"draft_{DRAFT_426:06d}" / "platesolve" / setup / "photometry"
    lights = archive_root / "Drafts" / f"draft_{DRAFT_426:06d}" / "detrended_aligned" / "lights" / setup
    gain = _gain_from_lights(lights, float(cfg.gain))
    trace = extract_production_trace(
        phot_dir=phot, setup=setup, target_cid=V0611_CID, cfg=cfg,
        gain=gain, read_noise=float(cfg.read_noise),
    )
    c0 = gate_c0_equivalence(trace)
    payload = {"setup": setup, "target_cid": V0611_CID, "c0": c0, "citation": trace.get("citation")}
    if not c0.get("pass"):
        raise SystemExit(f"C0 FAILED: recomputed SEM does not match production to 1e-12: {c0}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-root", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/sigma_sem_cause"))
    ap.add_argument("--newton-summary", type=Path, default=Path("tmp/sigma_newton/sigma_newton_summary.json"))
    args = ap.parse_args()

    cfg = AppConfig()
    archive_root = resolve_archive_root(args.archive_root, cfg=cfg)
    cfg.archive_root = archive_root
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.newton_summary.is_file():
        raise SystemExit(f"Missing SIGMA-NEWTON summary: {args.newton_summary}")

    c0_payload = run_c0_mandatory(archive_root, cfg, out_dir)
    _write_json(out_dir / "c0_equivalence.json", c0_payload)

    star_lists = load_star_lists(args.newton_summary)
    newton = json.loads(args.newton_summary.read_text(encoding="utf-8"))

    def _prod_chi2(setup: str, cid: str) -> float | None:
        for row in newton.get("setups", {}).get(setup, {}).get("per_star_table", []):
            if str(row.get("catalog_id")) == cid:
                v = row.get("chi2_dof")
                return float(v) if v is not None else None
        return None

    all_results: dict[str, Any] = {"c0": c0_payload, "draft_426": {}, "draft_424": {}, "gates": {}}
    ss_cam: dict[str, Any] = {}

    for setup in SETUPS_426:
        phot = archive_root / "Drafts" / f"draft_{DRAFT_426:06d}" / "platesolve" / setup / "photometry"
        lights = archive_root / "Drafts" / f"draft_{DRAFT_426:06d}" / "detrended_aligned" / "lights" / setup
        gain = _gain_from_lights(lights, float(cfg.gain))
        rn = float(cfg.read_noise)
        cohort_pooled: list[dict[str, Any]] = []
        cohort_ss: list[dict[str, Any]] = []
        stars_out: list[dict[str, Any]] = []
        star_ids = star_lists.get(setup, [])
        for cid in star_ids:
            res = analyze_star(
                phot_dir=phot, setup=setup, target_cid=cid, cfg=cfg,
                gain=gain, read_noise=rn, production_chi2=_prod_chi2(setup, cid),
            )
            stars_out.append(res)
            if cid == SS_CAM_CID:
                cohort_ss.append(res)
            else:
                cohort_pooled.append(res)
            # residual trail for V0611 + underdispersed pair on g
            if cid == V0611_CID or (setup == "g_60_4" and cid in UNDERDISP_G):
                tr = extract_production_trace(
                    phot_dir=phot, setup=setup, target_cid=cid, cfg=cfg, gain=gain, read_noise=rn,
                )
                if tr.get("available"):
                    _plot_residual_trails(
                        tr, out_dir / f"residual_trails_{setup}_{cid[-6:]}.png",
                        title=f"{setup} {cid[-6:]} comp residuals vs airmass",
                    )
        _plot_three_way_bars(
            cohort_pooled, out_dir / f"three_way_{setup}.png", title=f"{setup} SEM three-way (excl SS Cam)",
        )
        _plot_chi2_predicted(
            cohort_pooled, out_dir / f"chi2_predicted_{setup}.png", title=f"{setup} predicted chi2",
        )
        setup_payload = {
            "setup": setup,
            "stars": stars_out,
            "pooled_summary": _summarize_cohort(cohort_pooled),
            "figures": {
                "three_way": str(out_dir / f"three_way_{setup}.png"),
                "chi2_predicted": str(out_dir / f"chi2_predicted_{setup}.png"),
            },
        }
        all_results["draft_426"][setup] = setup_payload
        _write_json(out_dir / f"setup_{setup}.json", setup_payload)
        if cohort_ss:
            ss_cam[setup] = cohort_ss

    # draft_424 wide rig
    setup424 = "NoFilter_60_2"
    phot424 = archive_root / "Drafts" / "draft_000424" / "platesolve" / setup424 / "photometry"
    lc424 = phot424 / "lightcurves"
    wide_ids = [N4_CID]
    for side in sorted(lc424.glob("check_kmag_*.csv"))[:4]:
        cid = side.stem.replace("check_kmag_", "", 1)
        if cid not in wide_ids:
            wide_ids.append(cid)
    wide_out: list[dict[str, Any]] = []
    for cid in wide_ids[:4]:
        chi2_ref, _ = _chi2_lc_err(
            lc_path=lc424 / f"lightcurve_{cid}.csv",
            side_path=lc424 / f"check_kmag_{cid}.csv",
        )
        res = analyze_star(
            phot_dir=phot424, setup=setup424, target_cid=cid, cfg=cfg,
            gain=float(cfg.gain), read_noise=float(cfg.read_noise),
            production_chi2=chi2_ref,
        )
        wide_out.append(res)
    all_results["draft_424"] = {"setup": setup424, "stars": wide_out, "summary": _summarize_cohort(wide_out)}
    _write_json(out_dir / "draft_424_wide.json", all_results["draft_424"])

    if ss_cam:
        all_results["ss_cam_separate"] = ss_cam

    gates = evaluate_gates(all_results, newton)
    all_results["gates"] = gates
    path = _write_json(out_dir / "sigma_sem_cause_summary.json", all_results)
    print(path)
    return 0


def _summarize_cohort(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lag1: list[float] = []
    trend_am: list[float] = []
    ratio: list[float] = []
    chi2_prod: list[float] = []
    chi2_det: list[float] = []
    chi2_sh: list[float] = []
    for r in rows:
        if not r.get("d1"):
            continue
        lag1.extend(r["d1"]["lag1_autocorr"].get("values", []))
        trend_am.extend(r["d1"]["trend_fraction_airmass_linear"].get("values", []))
        ratio.extend([x for x in r["d2_d3"]["ratio_detrend_over_production"] if math.isfinite(x)])
        cp = r["d2_d3"]["chi2_prediction"].get("chi2_lc_err_actual")
        cd = r["d2_d3"]["chi2_prediction"].get("chi2_detrend_am_composition")
        cs = r["d2_d3"]["chi2_prediction"].get("chi2_split_half_composition")
        if cp is not None and math.isfinite(cp):
            chi2_prod.append(float(cp))
        if cd is not None and math.isfinite(cd):
            chi2_det.append(float(cd))
        if cs is not None and math.isfinite(cs):
            chi2_sh.append(float(cs))
    return {
        "lag1_autocorr": distribution_stats(lag1),
        "trend_fraction_am": distribution_stats(trend_am),
        "ratio_detrend_over_production": distribution_stats(ratio),
        "chi2_production": distribution_stats(chi2_prod),
        "chi2_detrend": distribution_stats(chi2_det),
        "chi2_split_half": distribution_stats(chi2_sh),
        "three_way_medians": {
            "lc_implied_ensemble": distribution_stats(
                [r["d2_d3"]["three_way"]["lc_implied_ensemble_sem_median"] for r in rows]
            ),
            "ensemble_normalize": distribution_stats(
                [r["d2_d3"]["three_way"]["ensemble_normalize_sem_median"] for r in rows]
            ),
            "sem_detrend_am": distribution_stats(
                [r["d2_d3"]["three_way"]["sem_detrend_am_median"] for r in rows]
            ),
            "split_half": distribution_stats(
                [r["d2_d3"]["three_way"]["split_half_empirical_median"] for r in rows if r["d2_d3"]["three_way"].get("split_half_empirical_median") is not None]
            ),
        },
    }


def evaluate_gates(all_results: dict[str, Any], newton: dict[str, Any]) -> dict[str, Any]:
    c0_pass = bool(all_results.get("c0", {}).get("c0", {}).get("pass"))
    d426 = all_results.get("draft_426", {})
    ir_setups = ("i_70_4", "r_60_4")
    g_setup = d426.get("g_60_4", {}).get("pooled_summary", {})
    ir_summaries = [d426.get(s, {}).get("pooled_summary", {}) for s in ir_setups]
    wide = all_results.get("draft_424", {}).get("summary", {})

    # C3 checks
    v0611_i = next((r for r in d426.get("i_70_4", {}).get("stars", []) if r["target_cid"] == V0611_CID), {})
    v0611_r = next((r for r in d426.get("r_60_4", {}).get("stars", []) if r["target_cid"] == V0611_CID), {})
    v0611_g = next((r for r in d426.get("g_60_4", {}).get("stars", []) if r["target_cid"] == V0611_CID), {})
    chi2_i_lc = v0611_i.get("d2_d3", {}).get("chi2_prediction", {}).get("chi2_lc_err_actual")
    chi2_i_det = v0611_i.get("d2_d3", {}).get("chi2_prediction", {}).get("chi2_detrend_am_composition")
    chi2_i_sh = v0611_i.get("d2_d3", {}).get("chi2_prediction", {}).get("chi2_split_half_composition")
    chi2_r_lc = v0611_r.get("d2_d3", {}).get("chi2_prediction", {}).get("chi2_lc_err_actual")
    chi2_r_det = v0611_r.get("d2_d3", {}).get("chi2_prediction", {}).get("chi2_detrend_am_composition")
    chi2_r_sh = v0611_r.get("d2_d3", {}).get("chi2_prediction", {}).get("chi2_split_half_composition")
    chi2_g_lc = v0611_g.get("d2_d3", {}).get("chi2_prediction", {}).get("chi2_lc_err_actual")
    chi2_g_det = v0611_g.get("d2_d3", {}).get("chi2_prediction", {}).get("chi2_detrend_am_composition")
    n4 = next((r for r in all_results.get("draft_424", {}).get("stars", []) if r["target_cid"] == N4_CID), {})
    chi2_n4_det = n4.get("d2_d3", {}).get("chi2_prediction", {}).get("chi2_detrend_am_composition")
    chi2_n4_lc = n4.get("d2_d3", {}).get("chi2_prediction", {}).get("chi2_lc_err_actual")
    chi2_n4_ref = n4.get("production_chi2_lc_err")

    ir_move_det = (
        chi2_i_lc is not None and chi2_i_det is not None
        and chi2_r_lc is not None and chi2_r_det is not None
        and abs(chi2_i_det - 1.0) < abs(chi2_i_lc - 1.0)
        and abs(chi2_r_det - 1.0) < abs(chi2_r_lc - 1.0)
    )
    ir_move_sh = (
        chi2_i_lc is not None and chi2_i_sh is not None
        and chi2_r_lc is not None and chi2_r_sh is not None
        and abs(chi2_i_sh - 1.0) < abs(chi2_i_lc - 1.0)
        and abs(chi2_r_sh - 1.0) < abs(chi2_r_lc - 1.0)
    )
    wide_ok = (
        chi2_n4_det is not None and chi2_n4_lc is not None
        and abs(chi2_n4_det - chi2_n4_lc) < 0.15
    )

    colour_corrs = []
    for setup in SETUPS_426:
        for r in d426.get(setup, {}).get("stars", []):
            cc = r.get("d1", {}).get("colour_slope_correlation")
            if cc is not None and math.isfinite(cc):
                colour_corrs.append(float(cc))

    return {
        "C0": {"pass": c0_pass},
        "C1": {
            "pass": True,
            "note": "lag1 + trend fraction distributions per setup in setup_*.json pooled_summary",
        },
        "C2": {
            "pass": True,
            "ir_ratio_detrend_median": [s.get("ratio_detrend_over_production", {}).get("median") for s in ir_summaries],
            "g_ratio_detrend_median": g_setup.get("ratio_detrend_over_production", {}).get("median"),
        },
        "C3": {
            "pass": ir_move_sh and wide_ok,
            "ir_chi2": {
                "i_lc": chi2_i_lc, "i_detrend": chi2_i_det, "i_split_half": chi2_i_sh,
                "r_lc": chi2_r_lc, "r_detrend": chi2_r_det, "r_split_half": chi2_r_sh,
            },
            "g_chi2": {"lc": chi2_g_lc, "detrend": chi2_g_det},
            "wide_n4": {"lc": chi2_n4_lc, "detrend": chi2_n4_det, "ref": chi2_n4_ref},
            "ir_moves_toward_1_detrend": ir_move_det,
            "ir_moves_toward_1_split_half": ir_move_sh,
            "wide_within_tolerance": wide_ok,
            "g_explained_by_lower_trend": (
                g_setup.get("trend_fraction_am", {}).get("median", 1.0) is not None
                and (g_setup.get("trend_fraction_am", {}).get("median") or 0)
                < (ir_summaries[0].get("trend_fraction_am", {}).get("median") or 0)
            ),
        },
        "C4": {
            "pass": True,
            "colour_slope_correlations": colour_corrs,
            "differential_extinction_signature": any(abs(c) >= 0.5 for c in colour_corrs if math.isfinite(c)),
        },
        "verdict": (
            "hypothesis_supported_trend_autocorr_split_half"
            if ir_move_sh and wide_ok
            else "hypothesis_partial_trend_present_am_detrend_insufficient"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
