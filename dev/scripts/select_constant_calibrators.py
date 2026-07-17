#!/usr/bin/env python3
"""Filter comp-selection population to verified-constant calibrators; run chi2 harness."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from check_star_kmag import (  # noqa: E402
    build_aligned_comp_inst,
    comp_ensemble_maps,
    compute_check_ensemble_mag_calib,
    resolve_proc_csv_dir,
)
from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photometry_core import check_comparison_stability, ensemble_member_ids, parse_comp_quality_json_map  # noqa: E402
from scripts.chi2_sigma_gate import (  # noqa: E402
    Chi2StarResult,
    compute_production_ensemble_scatter,
    evaluate_lc_chi2_variants,
    fit_f_resid_ensemble,
    fit_f_resid_sigma_floor_ensemble,
    plot_chi2_vs_g,
    saturation_margin_distribution,
    sigma_arrays_from_lc_and_proc,
    write_summary_json,
)
from sigma_budget import (  # noqa: E402
    SIGMA_VARIANT_HOWELL_SCINT_FRESID,
    SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR,
    SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE,
    resolve_rig_scintillation_params,
)

G93_CALIBRATOR = "1497674651102612992"
SATURATION_FLAG_FILL = 0.85


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def collect_comp_candidates(
    phot_dir: Path,
    *,
    min_frames: int,
    max_comp_rms: float = 0.1,
    max_contamination: float = 0.15,
) -> pd.DataFrame:
    comp_path = phot_dir / "comparison_stars_per_target.csv"
    if not comp_path.is_file():
        return pd.DataFrame()
    comp = pd.read_csv(comp_path, low_memory=False, dtype={"catalog_id": str, "target_catalog_id": str})
    agg = (
        comp.groupby("catalog_id", as_index=False)
        .agg(
            mag=("mag", "first"),
            phot_g_mean_mag=("phot_g_mean_mag", "first"),
            comp_n_frames=("comp_n_frames", "max"),
            comp_rms=("comp_rms", "min"),
            vsx_known_variable=("vsx_known_variable", "first"),
            gaia_dr3_variable_catalog=("gaia_dr3_variable_catalog", "first"),
            contamination_idx=("contamination_idx", "min"),
        )
        .copy()
    )
    agg["catalog_id"] = agg["catalog_id"].map(_norm_id)
    agg = agg.loc[agg["catalog_id"].astype(str).str.len().gt(0)]
    agg = agg.loc[~agg["vsx_known_variable"].fillna(False).astype(bool)]
    agg = agg.loc[~agg["gaia_dr3_variable_catalog"].fillna(False).astype(bool)]
    agg["comp_rms"] = pd.to_numeric(agg["comp_rms"], errors="coerce")
    agg["comp_n_frames"] = pd.to_numeric(agg["comp_n_frames"], errors="coerce")
    agg["contamination_idx"] = pd.to_numeric(agg["contamination_idx"], errors="coerce")
    agg = agg.loc[agg["comp_rms"].le(max_comp_rms)]
    agg = agg.loc[agg["contamination_idx"].isna() | agg["contamination_idx"].le(max_contamination)]
    agg = agg.loc[agg["comp_n_frames"].ge(float(min_frames))]
    gcol = "phot_g_mean_mag" if "phot_g_mean_mag" in agg.columns else "mag"
    agg["mag_g"] = pd.to_numeric(agg[gcol], errors="coerce")
    return agg.sort_values(["mag_g", "comp_rms"]).reset_index(drop=True)


def pick_g_coverage(calibrators: pd.DataFrame, *, aim: int = 8) -> pd.DataFrame:
    if calibrators.empty or len(calibrators) <= aim:
        return calibrators
    mags = calibrators["mag_g"].to_numpy(dtype=float)
    ok = np.isfinite(mags)
    qs = np.quantile(mags[ok], np.linspace(0.05, 0.95, aim))
    picked: list[int] = []
    for q in qs:
        d = np.abs(mags - q)
        d[~ok] = np.inf
        j = int(np.argmin(d))
        if j not in picked:
            picked.append(j)
    for i in range(len(calibrators)):
        if len(picked) >= aim:
            break
        if i not in picked:
            picked.append(i)
    return calibrators.iloc[picked[:aim]].reset_index(drop=True)


def pick_anchor_target(phot_dir: Path, comp_all: pd.DataFrame) -> str | None:
    summ_path = phot_dir / "photometry_summary.csv"
    if not summ_path.is_file():
        return None
    summ = pd.read_csv(summ_path, low_memory=False, dtype={"catalog_id": str})
    green = summ.loc[summ.get("trust", pd.Series(dtype=str)).astype(str).str.upper() == "GREEN"].copy()
    green["catalog_id"] = green["catalog_id"].map(_norm_id)
    counts = comp_all.groupby("target_catalog_id").size()
    best, best_n = None, -1
    for cid in green["catalog_id"]:
        n = int(counts.get(cid, 0))
        if n > best_n:
            best_n = n
            best = cid
    return best


def build_loo_differential_lc(
    calibrator_id: str,
    *,
    phot_dir: Path,
    setup: str,
    anchor_target: str,
    cfg: AppConfig,
) -> pd.DataFrame | None:
    lc_dir = phot_dir / "lightcurves"
    lc_path = lc_dir / f"lightcurve_{anchor_target}.csv"
    if not lc_path.is_file():
        return None
    lc_df = pd.read_csv(lc_path, low_memory=False)
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp_df = comp_all.loc[comp_all["target_catalog_id"].map(_norm_id) == _norm_id(anchor_target)]
    if comp_df.empty:
        return None
    cat, tier, rms, tw = comp_ensemble_maps(comp_df, cfg)
    cq_path = lc_dir / f"comp_quality_{anchor_target}.json"
    comp_quality_full: dict[str, dict] = {}
    if cq_path.is_file():
        comp_quality_full = parse_comp_quality_json_map(json.loads(cq_path.read_text(encoding="utf-8")))
    comp_ids = [_norm_id(c) for c in comp_df["catalog_id"].tolist() if _norm_id(c)]
    cid = _norm_id(calibrator_id)
    if cid not in comp_ids:
        comp_ids.append(cid)
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    if proc_dir is None:
        return None
    comp_lc = build_aligned_comp_inst(
        proc_dir, comp_ids, lc_df["source_file"].astype(str).tolist(), cfg, "aperture",
    )
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
    out = lc_df.iloc[: len(kmag_result.kmag)].copy()
    out["delta_mag"] = np.asarray(kmag_result.kmag, dtype=float)
    return out


def compute_loo_production_ensemble_scatter(
    calibrator_id: str,
    *,
    phot_dir: Path,
    setup: str,
    anchor_target: str,
    cfg: AppConfig,
) -> np.ndarray | None:
    """Path (b): Honeycutt ensemble SEM for LOO calibrator via production ``ensemble_normalize``."""
    lc_dir = phot_dir / "lightcurves"
    lc_path = lc_dir / f"lightcurve_{anchor_target}.csv"
    if not lc_path.is_file():
        return None
    lc_df = pd.read_csv(lc_path, low_memory=False)
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp_df = comp_all.loc[comp_all["target_catalog_id"].map(_norm_id) == _norm_id(anchor_target)]
    if comp_df.empty:
        return None
    cat, tier, rms, tw = comp_ensemble_maps(comp_df, cfg)
    cq_path = lc_dir / f"comp_quality_{anchor_target}.json"
    comp_quality_full: dict[str, dict] = {}
    if cq_path.is_file():
        comp_quality_full = parse_comp_quality_json_map(json.loads(cq_path.read_text(encoding="utf-8")))
    comp_ids = [_norm_id(c) for c in comp_df["catalog_id"].tolist() if _norm_id(c)]
    cid = _norm_id(calibrator_id)
    if cid not in comp_ids:
        comp_ids.append(cid)
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    if proc_dir is None or cid not in comp_ids:
        return None
    comp_lc = build_aligned_comp_inst(
        proc_dir, comp_ids, lc_df["source_file"].astype(str).tolist(), cfg, "aperture",
    )
    if cid not in comp_lc:
        return None
    other_ids = [c for c in comp_ids if c != cid and c in comp_lc]
    if len(other_ids) < 3:
        return None
    other_lc = {c: comp_lc[c] for c in other_ids}
    comp_quality = check_comparison_stability(
        other_lc, comp_rms_map=rms, n_comp_min=3, outlier_sigma=3.0, common_mode_detrend=True,
    )
    other_cat = {c: cat[c] for c in other_ids if c in cat}
    other_quality = {
        c: comp_quality[c]
        for c in other_ids
        if c in comp_quality and str(comp_quality[c].get("quality", "")).strip().lower() != "excluded"
    }
    if len(other_quality) < 3:
        return None
    return compute_production_ensemble_scatter(
        comp_lc[cid],
        other_lc,
        other_cat,
        other_quality,
        comp_rms_map=rms,
        comp_tier_map=tier,
        tier_weights=tw,
        n_comp_min=3,
        n_comp_max=int(cfg.phase01_comparison_n_comp_max),
    )


def draft_trust_summary(phot_dir: Path) -> dict[str, Any]:
    summ_path = phot_dir / "photometry_summary.csv"
    if not summ_path.is_file():
        return {"available": False}
    summ = pd.read_csv(summ_path, low_memory=False)
    trust_col = summ.get("trust", pd.Series(dtype=str)).astype(str)
    counts = trust_col.value_counts(dropna=False).to_dict()
    return {
        "available": True,
        "trust_value_counts": {str(k): int(v) for k, v in counts.items()},
        "n_rows": int(len(summ)),
        "n_green": int((trust_col.str.upper() == "GREEN").sum()),
    }


def _run_chi2_for_calibrators(
    calibrators: pd.DataFrame,
    *,
    phot_dir: Path,
    setup: str,
    anchor: str,
    proc_dir: Path,
    rig: Any,
    cfg: AppConfig,
    f_resid: float,
    sigma_floor_mag: float,
    f_resid_e: float,
    sigma_floor_mag_e: float,
    production_scatter_by_cid: dict[str, np.ndarray],
    exclude_ids: set[str] | None = None,
) -> tuple[list[Chi2StarResult], list[dict[str, Any]]]:
    chi2_objs: list[Chi2StarResult] = []
    rows_out: list[dict[str, Any]] = []
    fmap = {
        SIGMA_VARIANT_HOWELL_SCINT_FRESID: f_resid,
        SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR: f_resid,
        SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE: f_resid_e,
    }
    fmap_floor = {
        SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR: sigma_floor_mag,
        SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE: sigma_floor_mag_e,
    }
    for _, row in calibrators.iterrows():
        cid = _norm_id(row["catalog_id"])
        if exclude_ids and cid in exclude_ids:
            continue
        loo = build_loo_differential_lc(cid, phot_dir=phot_dir, setup=setup, anchor_target=anchor, cfg=cfg)
        if loo is None:
            continue
        prod_scatter = production_scatter_by_cid.get(cid)
        mags, variants, _, _, _ = sigma_arrays_from_lc_and_proc(
            loo,
            proc_dir,
            cid,
            rig_params=rig,
            f_resid=f_resid,
            sigma_floor_mag=sigma_floor_mag,
            production_ensemble_scatter=prod_scatter,
        )
        # Rebuild variant (e) with e-specific fit parameters
        mags_e, variants_e, _, _, sem_meta = sigma_arrays_from_lc_and_proc(
            loo,
            proc_dir,
            cid,
            rig_params=rig,
            f_resid=f_resid_e,
            sigma_floor_mag=sigma_floor_mag_e,
            production_ensemble_scatter=prod_scatter,
        )
        variants[SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE] = variants_e[
            SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE
        ]
        bjd = pd.to_numeric(loo["bjd"], errors="coerce").to_numpy(dtype=np.float64)
        res = evaluate_lc_chi2_variants(
            mags,
            variants,
            catalog_id=cid,
            mag_g=float(row["mag_g"]) if math.isfinite(float(row["mag_g"])) else None,
            bjd=bjd,
            f_resid_map=fmap,
            sigma_floor_map=fmap_floor,
        )
        chi2_objs.extend(res)
        for r in res:
            row_dict = r.to_dict()
            if r.variant == SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE:
                row_dict["ensemble_sem_meta"] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in sem_meta.items()
                    if k != "ensemble_sem_primary"
                }
                sem_p = sem_meta.get("ensemble_sem_primary")
                if isinstance(sem_p, np.ndarray):
                    fin = sem_p[np.isfinite(sem_p)]
                    row_dict["ensemble_sem_median_mag"] = float(np.median(fin)) if fin.size else None
                    row_dict["ensemble_sem_p95_mag"] = float(np.quantile(fin, 0.95)) if fin.size else None
            rows_out.append(row_dict)
    return chi2_objs, rows_out


def run_setup(draft_id: int, setup: str, *, cfg: AppConfig, min_frames: int, out_dir: Path) -> dict[str, Any]:
    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}" / "platesolve" / setup / "photometry"
    meta = json.loads((phot_dir / "pipeline_meta.json").read_text(encoding="utf-8")) if (phot_dir / "pipeline_meta.json").is_file() else {}
    rig = resolve_rig_scintillation_params(draft_id=draft_id, setup=setup, cfg=cfg, pipeline_meta=meta)
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    ) if (phot_dir / "comparison_stars_per_target.csv").is_file() else pd.DataFrame()

    eff_min = min_frames
    candidates = collect_comp_candidates(phot_dir, min_frames=eff_min)
    frame_gate_relaxed = False
    if candidates.empty and min_frames > 100:
        eff_min = 120 if draft_id == 424 else 10
        candidates = collect_comp_candidates(phot_dir, min_frames=eff_min)
        frame_gate_relaxed = True

    calibrators = pick_g_coverage(candidates, aim=8)
    anchor = pick_anchor_target(phot_dir, comp_all)
    out: dict[str, Any] = {
        "draft_id": draft_id,
        "setup": setup,
        "rig": rig.to_dict(),
        "anchor_target": anchor,
        "code_path": "check_star_kmag.compute_check_ensemble_mag_calib",
        "min_frames_requested": min_frames,
        "min_frames_effective": eff_min,
        "frame_gate_relaxed": frame_gate_relaxed,
        "trust_summary": draft_trust_summary(phot_dir),
        "calibrators": [],
        "chi2_results": [],
        "chi2_results_excl_g93": [],
        "variant_iqr": {},
    }
    if calibrators.empty or anchor is None or proc_dir is None:
        out["note"] = "no calibrators or anchor/proc missing"
        return out

    ensemble_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    ensemble_inputs_e: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    production_scatter_by_cid: dict[str, np.ndarray] = {}
    clamp_fracs: list[float] = []
    agreements: list[dict[str, Any]] = []
    loo_by_cid: dict[str, pd.DataFrame] = {}
    for _, row in calibrators.iterrows():
        cid = _norm_id(row["catalog_id"])
        loo = build_loo_differential_lc(cid, phot_dir=phot_dir, setup=setup, anchor_target=anchor, cfg=cfg)
        if loo is None:
            continue
        loo_by_cid[cid] = loo
        prod_scatter = compute_loo_production_ensemble_scatter(
            cid, phot_dir=phot_dir, setup=setup, anchor_target=anchor, cfg=cfg,
        )
        if prod_scatter is not None:
            production_scatter_by_cid[cid] = prod_scatter
        mags, _, sh, ss, sem_meta = sigma_arrays_from_lc_and_proc(
            loo,
            proc_dir,
            cid,
            rig_params=rig,
            production_ensemble_scatter=prod_scatter,
        )
        sem_primary = sem_meta["ensemble_sem_primary"]
        ensemble_inputs.append((mags, sh, ss))
        ensemble_inputs_e.append((mags, sh, ss, sem_primary))
        if math.isfinite(float(sem_meta.get("ensemble_sem_clamp_fraction", float("nan")))):
            clamp_fracs.append(float(sem_meta["ensemble_sem_clamp_fraction"]))
        agreements.append(dict(sem_meta.get("ensemble_sem_agreement") or {}))
        cal_entry = {
            "catalog_id": cid,
            "mag_g": float(row["mag_g"]) if math.isfinite(float(row["mag_g"])) else None,
            "n_frames": int(row["comp_n_frames"]),
            "comp_rms": float(row["comp_rms"]),
            "ensemble_sem_clamp_fraction": sem_meta.get("ensemble_sem_clamp_fraction"),
            "ensemble_sem_agreement": sem_meta.get("ensemble_sem_agreement"),
        }
        sem_fin = np.asarray(sem_primary, dtype=float)
        sem_fin = sem_fin[np.isfinite(sem_fin)]
        if sem_fin.size:
            cal_entry["ensemble_sem_median_mag"] = float(np.median(sem_fin))
            cal_entry["ensemble_sem_p95_mag"] = float(np.quantile(sem_fin, 0.95))
        if cid == _norm_id(G93_CALIBRATOR):
            cal_entry["saturation_margin"] = saturation_margin_distribution(loo, proc_dir, cid)
        out["calibrators"].append(cal_entry)

    f_resid, med_c2d, spread = fit_f_resid_ensemble(ensemble_inputs)
    out["f_resid_fit"] = {"f_resid": f_resid, "median_chi2_dof": med_c2d, "chi2_dof_iqr": spread}
    joint_d = fit_f_resid_sigma_floor_ensemble(ensemble_inputs, include_ensemble=False)
    joint_e = fit_f_resid_sigma_floor_ensemble(ensemble_inputs_e, include_ensemble=True)
    out["joint_fit"] = joint_d.to_dict()
    out["joint_fit_ensemble"] = joint_e.to_dict()
    out["sigma_floor_mag"] = joint_d.sigma_floor_mag
    out["sigma_floor_mag_mm"] = joint_d.sigma_floor_mag * 1000.0
    out["sigma_floor_mag_ensemble_mm"] = joint_e.sigma_floor_mag * 1000.0
    out["ensemble_sem_clamp_fraction_median"] = (
        float(np.median(clamp_fracs)) if clamp_fracs else None
    )
    agree_medians = [a.get("median_abs_diff") for a in agreements if a.get("median_abs_diff") is not None]
    agree_p95s = [a.get("p95_abs_diff") for a in agreements if a.get("p95_abs_diff") is not None]
    out["ensemble_sem_agreement_pooled"] = {
        "median_abs_diff": float(np.median(agree_medians)) if agree_medians else None,
        "p95_abs_diff": float(np.max(agree_p95s)) if agree_p95s else None,
        "n_calibrators": int(len(agree_medians)),
    }
    floor_collapsed = joint_e.sigma_floor_mag <= 0.005
    out["prediction_verdict"] = "floor_collapsed" if floor_collapsed else "floor_did_not_collapse"
    out["prediction_sigma_floor_e_mmag"] = joint_e.sigma_floor_mag * 1000.0

    sat_flag = False
    for c in out["calibrators"]:
        sm = c.get("saturation_margin") or {}
        if sm.get("fill_p95") is not None and float(sm["fill_p95"]) >= SATURATION_FLAG_FILL:
            sat_flag = True
        if sm.get("fill_max") is not None and float(sm["fill_max"]) >= 1.0:
            sat_flag = True
    out["g93_saturation_flagged"] = sat_flag

    chi2_objs, chi2_rows = _run_chi2_for_calibrators(
        calibrators,
        phot_dir=phot_dir,
        setup=setup,
        anchor=anchor,
        proc_dir=proc_dir,
        rig=rig,
        cfg=cfg,
        f_resid=joint_d.f_resid,
        sigma_floor_mag=joint_d.sigma_floor_mag,
        f_resid_e=joint_e.f_resid,
        sigma_floor_mag_e=joint_e.sigma_floor_mag,
        production_scatter_by_cid=production_scatter_by_cid,
    )
    out["chi2_results"] = chi2_rows
    if sat_flag:
        _, chi2_excl = _run_chi2_for_calibrators(
            calibrators,
            phot_dir=phot_dir,
            setup=setup,
            anchor=anchor,
            proc_dir=proc_dir,
            rig=rig,
            cfg=cfg,
            f_resid=joint_d.f_resid,
            sigma_floor_mag=joint_d.sigma_floor_mag,
            f_resid_e=joint_e.f_resid,
            sigma_floor_mag_e=joint_e.sigma_floor_mag,
            production_scatter_by_cid=production_scatter_by_cid,
            exclude_ids={_norm_id(G93_CALIBRATOR)},
        )
        out["chi2_results_excl_g93"] = chi2_excl

    variant_medians: dict[str, float] = {}
    for variant in sorted({r.variant for r in chi2_objs}):
        sub = [r.chi2_dof for r in chi2_objs if r.variant == variant and math.isfinite(r.chi2_dof)]
        if sub:
            variant_medians[variant] = float(np.median(sub))
        if len(sub) >= 2:
            out["variant_iqr"][variant] = float(np.subtract(*np.percentile(sub, [75, 25])))
    out["variant_median_chi2_dof"] = variant_medians

    if chi2_objs:
        out["chi2_plot"] = plot_chi2_vs_g(
            chi2_objs, out_dir / f"chi2_vs_g_draft{draft_id:06d}_{setup}.png",
            title=f"draft_{draft_id:06d}/{setup}",
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafts", default="424")
    ap.add_argument("--min-frames", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/sigma_budget"))
    args = ap.parse_args()
    cfg = AppConfig()
    out_dir = Path(args.out_dir)
    setups = []
    for d_s in args.drafts.split(","):
        d = int(d_s.strip())
        ps = Path(cfg.archive_root) / "Drafts" / f"draft_{d:06d}" / "platesolve"
        for sd in sorted(ps.iterdir()) if ps.is_dir() else []:
            if sd.is_dir():
                setups.append(run_setup(d, sd.name, cfg=cfg, min_frames=args.min_frames, out_dir=out_dir))
    path = write_summary_json({"setups": setups}, out_dir / "calibrator_chi2_summary.json")
    # Side-question: draft_425 trust distribution (K2 validation draft)
    trust_425: list[dict[str, Any]] = []
    for setup_name in ("V_20_2", "B_20_2", "R_20_2"):
        phot_dir = Path(cfg.archive_root) / "Drafts" / "draft_000425" / "platesolve" / setup_name / "photometry"
        trust_425.append({"draft_id": 425, "setup": setup_name, **draft_trust_summary(phot_dir)})
    write_summary_json({"draft_425_trust": trust_425}, out_dir / "draft_425_trust.json")
    print(path)


if __name__ == "__main__":
    main()
