#!/usr/bin/env python3
"""K2-COHORT: full-cohort k'' signature test on archive constant stars (report-only)."""

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
from scipy import stats  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from k2_cohort_core import (  # noqa: E402
    benjamini_hochberg_fdr,
    expected_k2_sign,
    extract_cell_report_stats,
    k2_priority_verdict,
    lag1_autocorrelation,
    photon_weighted_airmass_slope,
    spearman_min_n_for_power,
    weighted_linear_regression,
)
from scripts.chi2_sigma_gate import saturation_margin_distribution, write_summary_json  # noqa: E402
from check_star_kmag import (  # noqa: E402
    build_aligned_comp_inst,
    comp_ensemble_maps,
    compute_check_ensemble_mag_calib,
    resolve_proc_csv_dir,
)
from photometry_core import check_comparison_stability  # noqa: E402
from scripts.select_constant_calibrators import collect_comp_candidates  # noqa: E402
from scripts.sparse_comp_diag import SS_CAM_CID  # noqa: E402
from sigma_budget import resolve_rig_scintillation_params  # noqa: E402
from sigma_floor_core import pzq_fit_sigma_r, rel_sigma_to_mag  # noqa: E402

MIN_EPOCHS = 20
MIN_BINS = 5
MIN_CELL_N = 10
AIRMASS_RANGE_MIN = 0.15
FDR_Q = 0.05
BOOTSTRAP_DRAWS = 500

PRE_REGISTERED_RULE = """Family of tests: one per (rig, band). Multiple-testing control: Benjamini-Hochberg FDR
at q = 0.05 across the whole family (both T1 and T2 below).

- k'' priority UP if ANY (rig, band) shows the T1 signature with |rho| >= 0.3,
  q <= 0.05, and the physically expected sign (slope magnitude increasing with colour
  offset; per-filter sign conventions stated in the result).
- k'' priority DOWN only if ALL tested (rig, band) cells are null AND each cell had
  >= 80% power to detect rho = 0.4 at alpha = 0.05 (Spearman power: n >= ~46 per cell).
  Underpowered nulls do not count toward DOWN.
- Otherwise UNCHANGED, with the per-cell power stated."""

CELLS = [
    (424, "NoFilter_60_2", "wide", "CLEAR", 1, "wide_Carl-Zeiss"),
    (425, "V_20_2", "wide", "V", 1, "wide_Carl-Zeiss"),
    (425, "B_20_2", "wide", "B", 1, "wide_Carl-Zeiss"),
    (425, "R_20_2", "wide", "R", 1, "wide_Carl-Zeiss"),
    (426, "g_60_4", "Newton", "g", 4, "Newton_g"),
    (426, "i_70_4", "Newton", "i", 4, "Newton_i"),
]


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def _stamp(payload: dict[str, Any]) -> dict[str, Any]:
    payload["git_head"] = _git_head()
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return payload


def _bp_rp_map(phot_dir: Path) -> dict[str, float]:
    comp_path = phot_dir / "comparison_stars_per_target.csv"
    if not comp_path.is_file():
        return {}
    comp = pd.read_csv(comp_path, low_memory=False, dtype={"catalog_id": str})
    if "bp_rp" not in comp.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in comp.groupby("catalog_id", as_index=False).first().iterrows():
        cid = _norm_id(row["catalog_id"])
        v = float(pd.to_numeric(row.get("bp_rp"), errors="coerce"))
        if cid and math.isfinite(v):
            out[cid] = v
    return out


def _proc_row_cached(
    proc_dir: Path,
    source_file: str,
    catalog_id: str,
    csv_cache: dict[str, pd.DataFrame],
) -> pd.Series | None:
    path = proc_dir / str(source_file).strip()
    key = str(path)
    if key not in csv_cache:
        if not path.is_file():
            return None
        try:
            csv_cache[key] = pd.read_csv(path, low_memory=False, dtype={"catalog_id": str})
        except Exception:  # noqa: BLE001
            return None
    df = csv_cache[key]
    cid = _norm_id(catalog_id)
    id_col = "catalog_id" if "catalog_id" in df.columns else "name"
    if "_nid" not in df.columns:
        df = df.copy()
        df["_nid"] = df[id_col].map(_norm_id)
    sub = df.loc[df["_nid"] == cid]
    return None if sub.empty else sub.iloc[0]


def _photon_err_mag_cached(
    loo: pd.DataFrame,
    proc_dir: Path,
    catalog_id: str,
    *,
    gain: float,
    read_noise: float,
    csv_cache: dict[str, pd.DataFrame],
) -> np.ndarray:
    from scripts.chi2_sigma_gate import (  # noqa: PLC0415
        _proc_aperture_area_px,
        _relative_flux_sigma_with_bkg,
        _sky_pp_for_photometric_error,
    )
    from photometry_core import ERR_BKG_SOURCE_COL, SIGMA_BKG_AP_COL  # noqa: PLC0415

    n = len(loo)
    out = np.full(n, np.nan, dtype=np.float64)
    for i, sf in enumerate(loo.get("source_file", pd.Series([""] * n)).astype(str).tolist()):
        row = _proc_row_cached(proc_dir, sf, catalog_id, csv_cache)
        if row is None:
            continue
        flux = float(pd.to_numeric(row.get("dao_flux"), errors="coerce"))
        if not math.isfinite(flux) or flux <= 0:
            continue
        sky = float(_sky_pp_for_photometric_error(row))
        area = _proc_aperture_area_px(row)
        sig_bkg_raw = float(pd.to_numeric(row.get(SIGMA_BKG_AP_COL), errors="coerce"))
        sig_bkg_ap = sig_bkg_raw if math.isfinite(sig_bkg_raw) else None
        err_bkg_source = str(row.get(ERR_BKG_SOURCE_COL, "")).strip() or None
        err_rel, _ = _relative_flux_sigma_with_bkg(
            flux, sky, area,
            sigma_bkg_ap=sig_bkg_ap,
            err_bkg_source=err_bkg_source,
            gain=gain,
            read_noise=read_noise,
        )
        if math.isfinite(err_rel) and err_rel > 0:
            out[i] = rel_sigma_to_mag(float(err_rel))
    return out


def _batch_loo_delta_mag(
    cid: str,
    *,
    lc_df: pd.DataFrame,
    comp_lc: dict[str, np.ndarray],
    comp_ids: list[str],
    cat: dict[str, float],
    tier: dict[str, int],
    rms: dict[str, float],
    tw: dict[int, float],
    cfg: AppConfig,
) -> np.ndarray | None:
    """LOO differential mag using pre-built comp_lc cache."""
    cid = _norm_id(cid)
    if cid not in comp_lc:
        return None
    other_lc = {c: comp_lc[c] for c in comp_ids if c != cid and c in comp_lc}
    comp_quality = check_comparison_stability(
        other_lc, comp_rms_map=rms, n_comp_min=2, outlier_sigma=3.0, common_mode_detrend=True,
    )
    kmag_result = compute_check_ensemble_mag_calib(
        cid, comp_ids, comp_lc, cat, comp_quality,
        comp_rms_map=rms, comp_tier_map=tier, tier_weights=tw, cfg=cfg, n_comp_min=2,
    )
    if kmag_result is None:
        return None
    return np.asarray(kmag_result.kmag, dtype=np.float64)


def _sparse_target_ids(phot_dir: Path) -> set[str]:
    summ_path = phot_dir / "photometry_summary.csv"
    if not summ_path.is_file():
        return set()
    summ = pd.read_csv(summ_path, low_memory=False, dtype={"catalog_id": str})
    if "comp_path" not in summ.columns:
        return set()
    sparse = summ.loc[summ["comp_path"].astype(str).str.lower() == "sparse_fallback", "catalog_id"]
    return {_norm_id(x) for x in sparse if _norm_id(x)}


def _pick_host_target(comp_all: pd.DataFrame, cid: str, *, sparse_targets: set[str]) -> str | None:
    sub = comp_all.loc[comp_all["catalog_id"].map(_norm_id) == _norm_id(cid)].copy()
    if sub.empty:
        return None
    sub["target_catalog_id"] = sub["target_catalog_id"].map(_norm_id)
    sub = sub.loc[~sub["target_catalog_id"].isin(sparse_targets)]
    if sub.empty:
        return None
    sub["comp_n_frames"] = pd.to_numeric(sub["comp_n_frames"], errors="coerce")
    sub = sub.sort_values("comp_n_frames", ascending=False)
    return str(sub.iloc[0]["target_catalog_id"])


def _host_context(
    host: str,
    *,
    phot_dir: Path,
    comp_all: pd.DataFrame,
    proc_dir: Path,
    cfg: AppConfig,
    csv_cache: dict[str, pd.DataFrame],
) -> dict[str, Any] | None:
    lc_path = phot_dir / "lightcurves" / f"lightcurve_{host}.csv"
    if not lc_path.is_file():
        return None
    lc_df = pd.read_csv(lc_path, low_memory=False)
    lc_airmass = pd.to_numeric(lc_df["airmass"], errors="coerce").to_numpy(dtype=np.float64)
    comp_df = comp_all.loc[comp_all["target_catalog_id"].map(_norm_id) == _norm_id(host)]
    if comp_df.empty:
        return None
    cat, tier, rms, tw = comp_ensemble_maps(comp_df, cfg)
    comp_ids = [_norm_id(c) for c in comp_df["catalog_id"].tolist() if _norm_id(c)]
    comp_lc = build_aligned_comp_inst(
        proc_dir, comp_ids, lc_df["source_file"].astype(str).tolist(), cfg, "aperture",
        csv_cache=csv_cache,
    )
    return {
        "host": host,
        "lc_df": lc_df,
        "lc_airmass": lc_airmass,
        "comp_ids": comp_ids,
        "comp_lc": comp_lc,
        "cat": cat,
        "tier": tier,
        "rms": rms,
        "tw": tw,
    }


def _build_k2_star_record(
    cid: str,
    *,
    proc_dir: Path,
    lc_df: pd.DataFrame,
    lc_airmass_full: np.ndarray,
    comp_lc: dict[str, np.ndarray],
    comp_ids: list[str],
    cat: dict[str, float],
    tier: dict[str, int],
    rms: dict[str, float],
    tw: dict[int, float],
    cfg: AppConfig,
    gain: float,
    read_noise: float,
    csv_cache: dict[str, pd.DataFrame],
) -> dict[str, Any] | None:
    """LOO delta_mag from cached comp_lc (one proc load per cell)."""
    mags_full = _batch_loo_delta_mag(
        cid, lc_df=lc_df, comp_lc=comp_lc, comp_ids=comp_ids,
        cat=cat, tier=tier, rms=rms, tw=tw, cfg=cfg,
    )
    if mags_full is None:
        return None
    n = min(len(mags_full), len(lc_airmass_full))
    loo_stub = lc_df.iloc[:n].copy()
    loo_stub["delta_mag"] = mags_full[:n]
    margin = saturation_margin_distribution(loo_stub, proc_dir, cid)
    fill_p95 = margin.get("fill_p95")
    if fill_p95 is not None and math.isfinite(float(fill_p95)) and float(fill_p95) >= 0.85:
        return None
    mags = np.asarray(mags_full[:n], dtype=np.float64)
    airmass = lc_airmass_full[:n]
    photon = _photon_err_mag_cached(
        loo_stub, proc_dir, cid, gain=gain, read_noise=read_noise, csv_cache=csv_cache,
    )
    m = min(len(mags), len(airmass), len(photon))
    mags, airmass, photon = mags[:m], airmass[:m], photon[:m]
    ok = np.isfinite(mags) & np.isfinite(airmass) & np.isfinite(photon) & (photon > 0)
    if int(ok.sum()) < MIN_EPOCHS:
        return None
    err_mag = np.asarray(photon[ok], dtype=np.float64)
    return {
        "catalog_id": cid,
        "mags": mags[ok],
        "airmass": airmass[ok],
        "err_mag": err_mag,
        "n_epochs": int(ok.sum()),
    }


def _build_cohort_cell(
    draft_id: int,
    setup: str,
    rig: str,
    band: str,
    equipment_id: int,
    rig_label: str,
    *,
    cfg: AppConfig,
) -> dict[str, Any]:
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}" / "platesolve" / setup / "photometry"
    cell_key = f"{rig}_{band}"
    out: dict[str, Any] = {
        "draft_id": draft_id,
        "setup": setup,
        "rig": rig,
        "band": band,
        "cell_key": cell_key,
        "mag_quantity": "delta_mag_loo_per_host_target",
        "excluded": False,
        "stars": [],
    }
    if not phot_dir.is_dir():
        out["excluded"] = True
        out["exclude_reason"] = "missing_photometry"
        return out

    meta = json.loads((phot_dir / "pipeline_meta.json").read_text(encoding="utf-8")) if (phot_dir / "pipeline_meta.json").is_file() else {}
    resolve_rig_scintillation_params(draft_id=draft_id, setup=setup, cfg=cfg, pipeline_meta=meta)
    out["rig_label"] = rig_label
    out["equipment_id"] = equipment_id

    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    ) if (phot_dir / "comparison_stars_per_target.csv").is_file() else pd.DataFrame()
    candidates = collect_comp_candidates(phot_dir, min_frames=MIN_EPOCHS)
    candidates = candidates.loc[~candidates["catalog_id"].map(_norm_id).eq(_norm_id(SS_CAM_CID))]
    out["n_candidates_raw"] = int(len(candidates))
    if candidates.empty or proc_dir is None:
        out["excluded"] = True
        out["exclude_reason"] = "no_candidates_or_proc"
        return out

    bp_rp = _bp_rp_map(phot_dir)
    br_vals = [v for v in bp_rp.values() if math.isfinite(v)]
    bp_rp_med = float(np.median(br_vals)) if br_vals else float("nan")
    gain = float(meta.get("gain") or 1.0)
    rn = float(meta.get("read_noise") or 10.0)
    sparse_targets = _sparse_target_ids(phot_dir)
    out["n_sparse_targets_excluded"] = len(sparse_targets)

    csv_cache: dict[str, pd.DataFrame] = {}
    host_cache: dict[str, dict[str, Any] | None] = {}
    stars: list[dict[str, Any]] = []
    n_loo_fail = 0
    n_lever_excl = 0
    n_no_host = 0
    for _, row in candidates.iterrows():
        cid = _norm_id(row["catalog_id"])
        host = _pick_host_target(comp_all, cid, sparse_targets=sparse_targets)
        if host is None:
            n_no_host += 1
            continue
        if host not in host_cache:
            host_cache[host] = _host_context(
                host, phot_dir=phot_dir, comp_all=comp_all, proc_dir=proc_dir, cfg=cfg,
                csv_cache=csv_cache,
            )
        ctx = host_cache[host]
        if ctx is None:
            n_loo_fail += 1
            continue
        if cid not in ctx["comp_ids"]:
            ctx["comp_ids"] = list(ctx["comp_ids"]) + [cid]
            ctx["comp_lc"] = build_aligned_comp_inst(
                proc_dir, ctx["comp_ids"], ctx["lc_df"]["source_file"].astype(str).tolist(),
                cfg, "aperture", csv_cache=csv_cache,
            )
            host_cache[host] = ctx
        built = _build_k2_star_record(
            cid,
            proc_dir=proc_dir,
            lc_df=ctx["lc_df"],
            lc_airmass_full=ctx["lc_airmass"],
            comp_lc=ctx["comp_lc"],
            comp_ids=ctx["comp_ids"],
            cat=ctx["cat"],
            tier=ctx["tier"],
            rms=ctx["rms"],
            tw=ctx["tw"],
            cfg=cfg,
            gain=gain,
            read_noise=rn,
            csv_cache=csv_cache,
        )
        if built is None:
            n_loo_fail += 1
            continue
        mags = built["mags"]
        airmass = built["airmass"]
        err_mag = built["err_mag"]
        slope = photon_weighted_airmass_slope(mags, airmass, err_mag, min_airmass_range=AIRMASS_RANGE_MIN)
        if slope.get("excluded_lever_arm"):
            n_lever_excl += 1
        br = bp_rp.get(cid, float("nan"))
        colour_signed = float(br - bp_rp_med) if math.isfinite(br) and math.isfinite(bp_rp_med) else float("nan")
        colour_abs = abs(colour_signed) if math.isfinite(colour_signed) else float("nan")
        am_range = float(np.nanmax(airmass) - np.nanmin(airmass)) if np.isfinite(airmass).sum() >= 2 else float("nan")
        pzq = pzq_fit_sigma_r(mags)
        fit_bins = sum(1 for b in pzq.get("bins", []) if int(b.get("n_bins", 0) or 0) >= MIN_BINS)
        pzq_ok = fit_bins >= 2
        stars.append({
            "catalog_id": cid,
            "mag_g": float(row["mag_g"]) if math.isfinite(float(row.get("mag_g", float("nan")))) else None,
            "N_epochs": int(built["n_epochs"]),
            "bp_rp": br if math.isfinite(br) else None,
            "colour_offset_signed": colour_signed if math.isfinite(colour_signed) else None,
            "colour_offset_abs": colour_abs if math.isfinite(colour_abs) else None,
            "airmass_range": am_range if math.isfinite(am_range) else None,
            "b_X": slope.get("b_X"),
            "b_X_se": slope.get("b_X_se"),
            "t1_lever_excluded": bool(slope.get("excluded_lever_arm")),
            "sigma_w": pzq.get("sigma_w"),
            "sigma_r": pzq.get("sigma_r"),
            "pzq_ok": pzq_ok,
            "pzq_bins_ok": fit_bins,
            "lag1": lag1_autocorrelation(mags),
        })

    out["n_loo_failed"] = n_loo_fail
    out["n_no_host"] = n_no_host
    out["n_hosts_used"] = len(host_cache)
    out["n_lever_arm_excluded_t1"] = n_lever_excl
    out["n_stars"] = len(stars)
    out["bp_rp_ensemble_median"] = bp_rp_med if math.isfinite(bp_rp_med) else None
    out["stars"] = stars
    if len(stars) == 0:
        out["excluded"] = True
        out["exclude_reason"] = f"N_epochs gate (min {MIN_EPOCHS}) or all stars filtered"
    return out


def _spearman_block(x: list[float], y: list[float]) -> dict[str, Any]:
    pairs = [(a, b) for a, b in zip(x, y, strict=False) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": len(pairs)}
    xs, ys = zip(*pairs, strict=False)
    rho, p = stats.spearmanr(xs, ys)
    return {"rho": float(rho), "p": float(p), "n": len(pairs)}


def _bootstrap_median_ci(vals: list[float], *, seed: int) -> dict[str, float]:
    arr = [float(v) for v in vals if math.isfinite(float(v))]
    if len(arr) < 2:
        return {"median": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": len(arr)}
    rng = np.random.default_rng(seed)
    a = np.asarray(arr, dtype=np.float64)
    meds = [float(np.median(a[rng.integers(0, a.size, size=a.size)])) for _ in range(BOOTSTRAP_DRAWS)]
    lo, hi = np.quantile(meds, [0.16, 0.84])
    return {"median": float(np.median(a)), "ci_lo": float(lo), "ci_hi": float(hi), "n": len(arr)}


def _run_t1_cell(cell: dict[str, Any]) -> dict[str, Any]:
    stars = [s for s in cell.get("stars", []) if not s.get("t1_lever_excluded")]
    xs = [float(s["colour_offset_signed"]) for s in stars if s.get("colour_offset_signed") is not None]
    ys = [float(s["b_X"]) for s in stars if s.get("b_X") is not None]
    # align pairs
    pairs = [
        (float(s["colour_offset_signed"]), float(s["b_X"]))
        for s in stars
        if s.get("colour_offset_signed") is not None and s.get("b_X") is not None
        and math.isfinite(float(s["colour_offset_signed"])) and math.isfinite(float(s["b_X"]))
    ]
    if not pairs:
        sp = {"rho": float("nan"), "p": float("nan"), "n": 0}
    else:
        px, py = zip(*pairs, strict=False)
        sp = _spearman_block(list(px), list(py))
    w = [1.0 / (float(s["b_X_se"]) ** 2) for s in stars if s.get("b_X_se") and math.isfinite(float(s["b_X_se"])) and float(s["b_X_se"]) > 0]
    reg_pairs = [
        s for s in stars
        if s.get("colour_offset_signed") is not None and s.get("b_X") is not None
        and s.get("b_X_se") is not None
        and math.isfinite(float(s["colour_offset_signed"]))
        and math.isfinite(float(s["b_X"]))
        and math.isfinite(float(s["b_X_se"]))
        and float(s["b_X_se"]) > 0
    ]
    if len(reg_pairs) >= 2:
        cx = np.array([float(s["colour_offset_signed"]) for s in reg_pairs])
        cy = np.array([float(s["b_X"]) for s in reg_pairs])
        cw = np.array([1.0 / float(s["b_X_se"]) ** 2 for s in reg_pairs])
        reg = weighted_linear_regression(cx, cy, cw)
        k2_eff = reg["slope"]
        k2_se = reg["slope_se"]
    else:
        k2_eff, k2_se = float("nan"), float("nan")
    exp_sign = expected_k2_sign(str(cell.get("band", "")))
    return {
        "n_stars_t1": sp["n"],
        "n_lever_excluded": int(cell.get("n_lever_arm_excluded_t1", 0)),
        "spearman": sp,
        "expected_sign": exp_sign,
        "k2_eff_mag_per_airmass_per_colour": k2_eff,
        "k2_eff_se": k2_se,
        "sign_convention_note": (
            "rho>0 with negative k2_lit means redder (positive colour offset) -> more negative b_X"
            if exp_sign == -1.0
            else "V/CLEAR: weak or no literature sign; |rho| only for UP if sign check waived"
            if exp_sign is None
            else "i band: positive k2_lit -> rho>0 expected"
        ),
    }


def _run_t2_cell(cell: dict[str, Any]) -> dict[str, Any]:
    stars = [s for s in cell.get("stars", []) if s.get("pzq_ok")]
    xs: list[float] = []
    ys_r: list[float] = []
    ys_lag: list[float] = []
    for s in stars:
        co = s.get("colour_offset_abs")
        am = s.get("airmass_range")
        sr = s.get("sigma_r")
        lag = s.get("lag1")
        if co is None or am is None or sr is None:
            continue
        if not all(math.isfinite(float(v)) for v in (co, am, sr)):
            continue
        xs.append(float(co) * float(am))
        ys_r.append(float(sr))
        if lag is not None and math.isfinite(float(lag)):
            ys_lag.append(float(lag))
    sp_r = _spearman_block(xs, ys_r)
    lag_xs = xs[: len(ys_lag)]
    sp_lag = _spearman_block(lag_xs, ys_lag) if len(lag_xs) >= 3 else {"rho": float("nan"), "p": float("nan"), "n": len(lag_xs)}
    sr_vals = [float(s["sigma_r"]) for s in stars if s.get("sigma_r") is not None and math.isfinite(float(s["sigma_r"]))]
    sw_vals = [float(s["sigma_w"]) for s in stars if s.get("sigma_w") is not None and math.isfinite(float(s["sigma_w"]))]
    seed = int(cell.get("draft_id", 0)) + hash(str(cell.get("setup", ""))) % 10000
    return {
        "n_stars_t2": sp_r["n"],
        "n_pzq_ok": len(stars),
        "sigma_r_median_ci": _bootstrap_median_ci(sr_vals, seed=seed),
        "sigma_w_median_ci": _bootstrap_median_ci(sw_vals, seed=seed + 1),
        "spearman_sigma_r_vs_abs_colour_x_am": sp_r,
        "spearman_lag1_vs_abs_colour_x_am": sp_lag,
    }


def _plot_t1(cell: dict[str, Any], t1: dict[str, Any], out_path: Path) -> None:
    stars = [s for s in cell.get("stars", []) if not s.get("t1_lever_excluded")]
    xs = [float(s["colour_offset_signed"]) for s in stars if s.get("colour_offset_signed") is not None and s.get("b_X") is not None]
    ys = [float(s["b_X"]) for s in stars if s.get("colour_offset_signed") is not None and s.get("b_X") is not None]
    if len(xs) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, alpha=0.6, s=30)
    k2 = t1.get("k2_eff_mag_per_airmass_per_colour")
    if k2 is not None and math.isfinite(float(k2)):
        xline = np.linspace(min(xs), max(xs), 50)
        ax.plot(xline, float(k2) * xline, "r--", label=f"k2_eff={float(k2):.5f}")
        ax.legend()
    ax.set_xlabel("signed BP-RP offset from ensemble median")
    ax.set_ylabel("b_X (mag/airmass)")
    ax.set_title(f"T1 b_X vs colour - {cell.get('cell_key')}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_analysis(out_dir: Path, cfg: AppConfig) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pre_registered_rule.txt").write_text(PRE_REGISTERED_RULE + "\n", encoding="ascii")

    cohort_table: list[dict[str, Any]] = []
    cells_out: list[dict[str, Any]] = []
    for draft_id, setup, rig, band, eq_id, rig_label in CELLS:
        cell = _build_cohort_cell(
            draft_id, setup, rig, band, eq_id, rig_label, cfg=cfg,
        )
        for s in cell.get("stars", []):
            cohort_table.append({
                "star": s["catalog_id"],
                "rig": rig,
                "band": band,
                "draft_id": draft_id,
                "setup": setup,
                "N": s["N_epochs"],
                "brightness_mag_g": s.get("mag_g"),
                "colour_offset_signed": s.get("colour_offset_signed"),
                "colour_offset_abs": s.get("colour_offset_abs"),
            })
        t1 = _run_t1_cell(cell) if not cell.get("excluded") else {}
        t2 = _run_t2_cell(cell) if not cell.get("excluded") else {}
        fig_path = out_dir / "figures" / f"t1_{cell.get('cell_key', 'x')}.png"
        if not cell.get("excluded"):
            _plot_t1(cell, t1, fig_path)
            t1["figure"] = str(fig_path)
        cell["t1"] = t1
        cell["t2"] = t2
        cell["n_t1"] = int(t1.get("n_stars_t1", 0) or 0)
        cells_out.append(cell)

    # FDR across family (T1 + T2 per non-excluded cell with n>=3 for spearman)
    fdr_entries: list[dict[str, Any]] = []
    p_list: list[float] = []
    for cell in cells_out:
        if cell.get("excluded"):
            continue
        for test_key, block_key in (("T1", "t1"), ("T2", "t2")):
            block = cell.get(block_key) or {}
            sp = block.get("spearman") or block.get("spearman_sigma_r_vs_abs_colour_x_am") or {}
            p = float(sp.get("p", float("nan")))
            fdr_entries.append({"cell": cell["cell_key"], "test": test_key, "p": p, "rho": sp.get("rho")})
            p_list.append(p)

    fdr_adj = benjamini_hochberg_fdr(p_list, q=FDR_Q)
    for entry, adj in zip(fdr_entries, fdr_adj, strict=True):
        entry.update(adj)

    for cell in cells_out:
        if cell.get("excluded"):
            continue
        ck = cell["cell_key"]
        t1_fdr = next((e for e in fdr_entries if e["cell"] == ck and e["test"] == "T1"), {})
        t2_fdr = next((e for e in fdr_entries if e["cell"] == ck and e["test"] == "T2"), {})
        cell["t1_fdr"] = {
            "rho": t1_fdr.get("rho"),
            "p": t1_fdr.get("p"),
            "q_value": t1_fdr.get("q_value"),
            "reject": t1_fdr.get("reject"),
            "expected_sign": expected_k2_sign(str(cell.get("band", ""))),
        }
        cell["t2_fdr"] = {
            "rho": t2_fdr.get("rho"),
            "p": t2_fdr.get("p"),
            "q_value": t2_fdr.get("q_value"),
            "reject": t2_fdr.get("reject"),
        }
        if cell["n_t1"] < MIN_CELL_N:
            cell["status"] = "excluded_for_power"

    verdict_block = k2_priority_verdict(cells_out, fdr_q=FDR_Q)
    min_n = spearman_min_n_for_power()
    report_stats = [
        extract_cell_report_stats(c)
        for c in cells_out
        if not c.get("excluded")
    ]

    payload = _stamp({
        "pre_registered_rule": PRE_REGISTERED_RULE,
        "min_epochs_gate": MIN_EPOCHS,
        "min_cell_n": MIN_CELL_N,
        "fdr_q": FDR_Q,
        "spearman_power_n80_rho0.4": min_n,
        "cohort_table": cohort_table,
        "cells": cells_out,
        "fdr_family": fdr_entries,
        "verdict": verdict_block,
        "report_stats": report_stats,
    })
    write_summary_json(payload, out_dir / "cohort_table.json")
    write_summary_json(payload, out_dir / "k2_cohort_summary.json")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="K2-COHORT full-cohort k'' test (report-only)")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "tmp" / "k2_cohort")
    args = ap.parse_args()
    cfg = AppConfig()
    payload = run_analysis(args.out_dir, cfg)
    print(json.dumps({"verdict": payload["verdict"], "n_cells": len(payload["cells"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
