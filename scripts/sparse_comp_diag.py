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

_ROOT = Path(__file__).resolve().parent.parent
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
    side = lc_dir / f"check_kmag_{target_cid}.csv"
    if side.is_file():
        side_df = pd.read_csv(side, low_memory=False)
        side_df["delta_mag"] = pd.to_numeric(side_df["kmag"], errors="coerce")
        side_df["source_file"] = lc_df["source_file"].astype(str).iloc[: len(side_df)].tolist()
        side_df["airmass"] = pd.to_numeric(lc_df["airmass"], errors="coerce").iloc[: len(side_df)].tolist()
        chk_cid = str(side_df["check_catalog_id"].iloc[0]) if "check_catalog_id" in side_df.columns else ""
        if chk_cid and proc_dir is not None:
            mags, variants, _, _ = sigma_arrays_from_lc_and_proc(side_df, proc_dir, chk_cid, rig_params=rig)
            bjd = pd.to_numeric(lc_df.get("bjd"), errors="coerce").to_numpy(dtype=np.float64)
            out["check_chi2"] = [r.to_dict() for r in evaluate_lc_chi2_variants(
                mags, variants, catalog_id=chk_cid, mag_g=None, bjd=bjd,
            )]
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
                side_df["delta_mag"] = pd.to_numeric(side_df["kmag"], errors="coerce")
                side_df["source_file"] = lc_df["source_file"].astype(str).iloc[: len(side_df)].tolist()
                side_df["airmass"] = pd.to_numeric(lc_df["airmass"], errors="coerce").iloc[: len(side_df)].tolist()
                chk_cid = str(side_df["check_catalog_id"].iloc[0]) if "check_catalog_id" in side_df.columns else ""
                mags, variants, _, _ = sigma_arrays_from_lc_and_proc(side_df, proc_dir, chk_cid, rig_params=rig)
                bjd = pd.to_numeric(lc_df.get("bjd"), errors="coerce").to_numpy(dtype=np.float64)
                entry["available"] = True
                entry["rig"] = rig.to_dict()
                entry["check_chi2"] = [r.to_dict() for r in evaluate_lc_chi2_variants(
                    mags, variants, catalog_id=chk_cid, mag_g=None, bjd=bjd,
                )]
            results.append(entry)
    else:
        results = [analyze_setup(d, s, t, cfg=cfg) for d, s, t in cases]
    path = write_summary_json({"cases": results}, args.out_dir / "sparse_comp_diag.json")
    print(path)


if __name__ == "__main__":
    main()
