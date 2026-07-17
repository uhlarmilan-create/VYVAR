#!/usr/bin/env python3
"""Sparse-field comp_rms decomposition and check-star reliability diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from check_star_kmag import build_aligned_comp_inst, resolve_proc_csv_dir  # noqa: E402
from config import AppConfig  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from scripts.chi2_sigma_gate import (  # noqa: E402
    bootstrap_chi2_dof_ci,
    evaluate_lc_chi2_variants,
    reduced_chi2_constant,
    sigma_arrays_from_lc_and_proc,
    write_summary_json,
)
from scripts.select_constant_calibrators import compute_loo_production_ensemble_scatter  # noqa: E402
from sigma_budget import resolve_rig_scintillation_params  # noqa: E402

SS_CAM_CID = "1112113066119992064"
V0611_CID = "1112127291051695744"


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def two_way_comp_decomposition(comp_matrix: np.ndarray) -> dict[str, float]:
    """Decompose star x frame comp mags: inter-star offset vs temporal residual."""
    m = np.asarray(comp_matrix, dtype=np.float64)
    ok = np.isfinite(m)
    if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < 2:
        return {
            "inter_star_rms": float("nan"),
            "temporal_rms": float("nan"),
            "headline_rms": float("nan"),
            "ratio_temporal_to_headline": float("nan"),
        }
    star_means = np.nanmean(m, axis=0)
    frame_means = np.nanmean(m, axis=1)
    grand = np.nanmean(m)
    resid = m - star_means - frame_means[:, np.newaxis] + grand
    inter = float(np.nanstd(star_means))
    temporal = float(np.nanstd(resid))
    headline = float(np.nanstd(m))
    ratio = temporal / headline if math.isfinite(headline) and headline > 0 else float("nan")
    return {
        "inter_star_rms": inter,
        "temporal_rms": temporal,
        "headline_rms": headline,
        "ratio_temporal_to_headline": ratio,
    }


def build_comp_matrix(
    comp_ids: list[str],
    source_files: list[str],
    proc_dir: Path,
    cfg: AppConfig,
) -> np.ndarray:
    comp_lc = build_aligned_comp_inst(proc_dir, comp_ids, source_files, cfg, "aperture")
    n_frames = len(source_files)
    mat = np.full((n_frames, len(comp_ids)), np.nan, dtype=np.float64)
    for j, cid in enumerate(comp_ids):
        if cid in comp_lc:
            mat[:, j] = comp_lc[cid][:n_frames]
    return mat


def bootstrap_scatter_mag_ci(
    mags: np.ndarray,
    *,
    n_boot: int = 400,
    seed: int = 0,
    alpha: float = 0.16,
) -> tuple[float | None, float | None]:
    """Bootstrap CI on sample std (ddof=1) of magnitude series."""
    m = np.asarray(mags, dtype=np.float64)
    m = m[np.isfinite(m)]
    n = int(m.size)
    if n < 5 or n_boot <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = float(np.std(m[idx], ddof=1))
        if math.isfinite(s):
            vals.append(s)
    if len(vals) < 10:
        return None, None
    arr = np.sort(np.asarray(vals, dtype=float))
    return float(np.quantile(arr, alpha)), float(np.quantile(arr, 1.0 - alpha))


def _load_draft424_joint_fits(summary_path: Path) -> tuple[float, float, float, float]:
    """Return (f_resid_d, sigma_floor_d, f_resid_e, sigma_floor_e) from calibrator summary."""
    defaults = (0.74, 0.0105, 0.0, 0.0065)
    if not summary_path.is_file():
        return defaults
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        setups = payload.get("setups") or []
        setup = next((s for s in setups if int(s.get("draft_id", -1)) == 424), None)
        if not setup:
            return defaults
        jd = setup.get("joint_fit") or {}
        je = setup.get("joint_fit_ensemble") or {}
        return (
            float(jd.get("f_resid", defaults[0])),
            float(jd.get("sigma_floor_mag", defaults[1])),
            float(je.get("f_resid", defaults[2])),
            float(je.get("sigma_floor_mag", defaults[3])),
        )
    except Exception:  # noqa: BLE001
        return defaults


def _check_star_chi2_rows(
    *,
    phot_dir: Path,
    setup: str,
    target_cid: str,
    lc_df: pd.DataFrame,
    side_df: pd.DataFrame,
    proc_dir: Path,
    rig: Any,
    cfg: AppConfig,
    f_resid_d: float = 0.74,
    sigma_floor_d: float = 0.0105,
    f_resid_e: float = 0.0,
    sigma_floor_e: float = 0.0065,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chk_cid = str(side_df["check_catalog_id"].iloc[0]) if "check_catalog_id" in side_df.columns else ""
    if not chk_cid:
        return [], {"available": False}
    work = side_df.copy()
    work["delta_mag"] = pd.to_numeric(work["kmag"], errors="coerce")
    work["source_file"] = lc_df["source_file"].astype(str).iloc[: len(work)].tolist()
    work["airmass"] = pd.to_numeric(lc_df["airmass"], errors="coerce").iloc[: len(work)].tolist()
    if "err" not in work.columns and "err" in lc_df.columns:
        work["err"] = pd.to_numeric(lc_df["err"], errors="coerce").iloc[: len(work)].tolist()
    prod_scatter = compute_loo_production_ensemble_scatter(
        chk_cid,
        phot_dir=phot_dir,
        setup=setup,
        anchor_target=target_cid,
        cfg=cfg,
    )
    mags_d, variants_d, _, _, _sem_meta_d = sigma_arrays_from_lc_and_proc(
        work,
        proc_dir,
        chk_cid,
        rig_params=rig,
        f_resid=f_resid_d,
        sigma_floor_mag=sigma_floor_d,
        production_ensemble_scatter=prod_scatter,
    )
    _, variants_e, _, _, sem_meta_e = sigma_arrays_from_lc_and_proc(
        work,
        proc_dir,
        chk_cid,
        rig_params=rig,
        f_resid=f_resid_e,
        sigma_floor_mag=sigma_floor_e,
        production_ensemble_scatter=prod_scatter,
    )
    variants_d.update(
        {k: variants_e[k] for k in variants_e if k.endswith("_ensemble")},
    )
    bjd = pd.to_numeric(lc_df.get("bjd"), errors="coerce").to_numpy(dtype=np.float64)
    rows = [
        r.to_dict()
        for r in evaluate_lc_chi2_variants(
            mags_d,
            variants_d,
            catalog_id=chk_cid,
            mag_g=None,
            bjd=bjd,
        )
    ]
    sem_p = sem_meta_e.get("ensemble_sem_primary")
    sem_summary: dict[str, Any] = {
        "check_catalog_id": chk_cid,
        "ensemble_sem_clamp_fraction": sem_meta_e.get("ensemble_sem_clamp_fraction"),
        "ensemble_sem_agreement": sem_meta_e.get("ensemble_sem_agreement"),
    }
    if isinstance(sem_p, np.ndarray):
        fin = sem_p[np.isfinite(sem_p)]
        if fin.size:
            sem_summary["ensemble_sem_median_mag"] = float(np.median(fin))
            sem_summary["ensemble_sem_p95_mag"] = float(np.quantile(fin, 0.95))
            sem_summary["ensemble_sem_max_mag"] = float(np.max(fin))
    return rows, sem_summary


def check_scatter_from_sidecar(lc_dir: Path, target_cid: str) -> dict[str, Any]:
    side = lc_dir / f"check_kmag_{target_cid}.csv"
    if not side.is_file():
        return {"scatter_mag": float("nan"), "n_frames": 0}
    df = pd.read_csv(side, low_memory=False)
    kmag = pd.to_numeric(df.get("kmag"), errors="coerce").dropna()
    if kmag.empty:
        return {"scatter_mag": float("nan"), "n_frames": 0}
    arr = kmag.to_numpy(dtype=float)
    scatter = float(np.std(arr, ddof=1))
    sig_naive = np.full_like(arr, scatter if math.isfinite(scatter) and scatter > 0 else np.nan)
    _, _, c2d, _ = reduced_chi2_constant(arr, sig_naive)
    c2_lo, c2_hi = bootstrap_chi2_dof_ci(arr, sig_naive)
    sc_lo, sc_hi = bootstrap_scatter_mag_ci(arr)
    return {
        "scatter_mag": scatter,
        "n_frames": int(len(arr)),
        "chi2_dof_naive": c2d,
        "chi2_ci_lo": c2_lo,
        "chi2_ci_hi": c2_hi,
        "scatter_mag_ci_lo": sc_lo,
        "scatter_mag_ci_hi": sc_hi,
    }


def analyze_setup(
    draft_id: int,
    setup: str,
    target_cid: str,
    *,
    cfg: AppConfig,
    joint_path: Path | None = None,
) -> dict[str, Any]:
    phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}" / "platesolve" / setup / "photometry"
    lc_dir = phot_dir / "lightcurves"
    lc_path = lc_dir / f"lightcurve_{target_cid}.csv"
    comp_all = pd.read_csv(
        phot_dir / "comparison_stars_per_target.csv",
        low_memory=False,
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp_df = comp_all.loc[comp_all["target_catalog_id"].map(_norm_id) == _norm_id(target_cid)]
    proc_dir = resolve_proc_csv_dir(phot_dir, setup)
    out: dict[str, Any] = {
        "draft_id": draft_id,
        "setup": setup,
        "target_catalog_id": target_cid,
        "available": lc_path.is_file() and proc_dir is not None,
    }
    if not out["available"]:
        return out
    lc_df = pd.read_csv(lc_path, low_memory=False)
    comp_ids = [_norm_id(c) for c in comp_df["catalog_id"].tolist() if _norm_id(c)]
    mat = build_comp_matrix(comp_ids, lc_df["source_file"].astype(str).tolist(), proc_dir, cfg)
    decomp = two_way_comp_decomposition(mat)
    out["decomposition"] = decomp
    # field-wide comp_rms headline from comp CSV (sparse path)
    if not comp_df.empty and "comp_rms_fieldwide" in comp_df.columns:
        fw = pd.to_numeric(comp_df["comp_rms_fieldwide"], errors="coerce").dropna()
        out["comp_rms_fieldwide_median"] = float(fw.median()) if not fw.empty else None
    fw2 = pd.to_numeric(comp_df.get("comp_rms"), errors="coerce").dropna()
    out["comp_rms_per_target_median"] = float(fw2.median()) if not fw2.empty else None
    chk = check_scatter_from_sidecar(lc_dir, target_cid)
    out["check_star"] = chk
    if chk.get("scatter_mag") and decomp.get("headline_rms"):
        out["cancellation_factor"] = (
            float(chk["scatter_mag"]) / float(decomp["headline_rms"])
            if float(decomp["headline_rms"]) > 0
            else None
        )
    meta = json.loads((phot_dir / "pipeline_meta.json").read_text(encoding="utf-8")) if (phot_dir / "pipeline_meta.json").is_file() else {}
    rig = resolve_rig_scintillation_params(draft_id=draft_id, setup=setup, cfg=cfg, pipeline_meta=meta)
    f_resid_d, sigma_floor_d, f_resid_e, sigma_floor_e = _load_draft424_joint_fits(
        joint_path or Path("tmp/sigma_budget/calibrator_chi2_summary.json"),
    )
    side = lc_dir / f"check_kmag_{target_cid}.csv"
    if side.is_file() and proc_dir is not None:
        side_df = pd.read_csv(side, low_memory=False)
        rows, sem_summary = _check_star_chi2_rows(
            phot_dir=phot_dir,
            setup=setup,
            target_cid=target_cid,
            lc_df=lc_df,
            side_df=side_df,
            proc_dir=proc_dir,
            rig=rig,
            cfg=cfg,
            f_resid_d=f_resid_d,
            sigma_floor_d=sigma_floor_d,
            f_resid_e=f_resid_e,
            sigma_floor_e=sigma_floor_e,
        )
        out["check_chi2"] = rows
        out["check_ensemble_sem"] = sem_summary
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/sigma_budget"))
    ap.add_argument(
        "--chi2-only",
        action="store_true",
        help="Recompute check-star chi2 rows only (skip decomposition)",
    )
    args = ap.parse_args()
    cfg = AppConfig()
    joint_path = Path(args.out_dir) / "calibrator_chi2_summary.json"
    f_resid_d, sigma_floor_d, f_resid_e, sigma_floor_e = _load_draft424_joint_fits(joint_path)
    cases = [
        (426, "g_60_4", SS_CAM_CID),
        (426, "r_60_4", SS_CAM_CID),
        (426, "i_70_4", SS_CAM_CID),
        (426, "g_60_4", V0611_CID),
    ]
    if args.chi2_only:
        results = []
        for d, s, t in cases:
            phot_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{d:06d}" / "platesolve" / s / "photometry"
            lc_dir = phot_dir / "lightcurves"
            lc_path = lc_dir / f"lightcurve_{t}.csv"
            proc_dir = resolve_proc_csv_dir(phot_dir, s)
            meta = json.loads((phot_dir / "pipeline_meta.json").read_text(encoding="utf-8")) if (phot_dir / "pipeline_meta.json").is_file() else {}
            rig = resolve_rig_scintillation_params(draft_id=d, setup=s, cfg=cfg, pipeline_meta=meta)
            entry: dict[str, Any] = {"draft_id": d, "setup": s, "target_catalog_id": t, "chi2_only": True}
            side = lc_dir / f"check_kmag_{t}.csv"
            if not side.is_file() or proc_dir is None or not lc_path.is_file():
                entry["available"] = False
            else:
                lc_df = pd.read_csv(lc_path, low_memory=False)
                side_df = pd.read_csv(side, low_memory=False)
                rows, sem_summary = _check_star_chi2_rows(
                    phot_dir=phot_dir,
                    setup=s,
                    target_cid=t,
                    lc_df=lc_df,
                    side_df=side_df,
                    proc_dir=proc_dir,
                    rig=rig,
                    cfg=cfg,
                    f_resid_d=f_resid_d,
                    sigma_floor_d=sigma_floor_d,
                    f_resid_e=f_resid_e,
                    sigma_floor_e=sigma_floor_e,
                )
                entry["available"] = True
                entry["rig"] = rig.to_dict()
                entry["joint_fit_d"] = {"f_resid": f_resid_d, "sigma_floor_mag": sigma_floor_d}
                entry["joint_fit_e"] = {"f_resid": f_resid_e, "sigma_floor_mag": sigma_floor_e}
                entry["check_chi2"] = rows
                entry["check_ensemble_sem"] = sem_summary
            results.append(entry)
    else:
        results = [analyze_setup(d, s, t, cfg=cfg, joint_path=joint_path) for d, s, t in cases]
    path = write_summary_json({"cases": results}, args.out_dir / "sparse_comp_diag.json")
    print(path)


if __name__ == "__main__":
    main()
