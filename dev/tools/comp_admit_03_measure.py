"""COMP-ADMIT-03: measure pool sizes and weight coeffs on drafts 435/512/513."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

from comp_pool_noise import analyze_draft_comp_pool  # noqa: E402
from comp_weights import resolve_comp_weight_coeffs, weights_table, CompWeightCoeffs  # noqa: E402
from config import AppConfig  # noqa: E402
from photometry_core import build_global_comp_pool  # noqa: E402


def _draft_paths(n: int) -> tuple[Path, Path, Path]:
    d = REPO / f"Archive/Drafts/draft_{n:06d}"
    base = d / "platesolve/NoFilter_60_2"
    proc = d / "detrended_aligned/lights/NoFilter_60_2"
    return d, base, proc


def run_one(draft_id: int) -> dict:
    draft, base, proc = _draft_paths(draft_id)
    ms = pd.read_csv(base / "masterstars_full_match.csv", low_memory=False)
    paths = sorted(proc.glob("proc_*.csv"))
    vt = pd.read_csv(base / "variable_targets.csv", low_memory=False)
    vt_ids = set(vt["catalog_id"].astype(str)) if "catalog_id" in vt.columns else set()
    cfg = AppConfig()
    cfg.comp_pool_derived_admission = True
    fw = int(pd.to_numeric(ms["x"], errors="coerce").max()) + 50
    fh = int(pd.to_numeric(ms["y"], errors="coerce").max()) + 50
    art = REPO / f"tmp/comp_admit_03_d{draft_id}"
    art.mkdir(parents=True, exist_ok=True)
    pool = build_global_comp_pool(
        masterstars_df=ms,
        per_frame_csv_paths=paths,
        csv_cache={},
        variable_target_catalog_ids=vt_ids,
        safe_bbox=None,
        chip_fw=fw,
        chip_fh=fh,
        chip_interior_margin_px=20,
        max_comp_rms=0.1,
        cfg=cfg,
        admission_artifact_dir=art,
    )
    analysis = analyze_draft_comp_pool(
        proc,
        draft_id=draft_id,
        setup="NoFilter_60_2",
        gain=float(getattr(cfg, "gain", 1.0) or 1.0),
        read_noise_e=float(getattr(cfg, "read_noise", 0.0) or 0.0),
    )
    n_sum = int(analysis.get("n_stars", 0) or 0)
    n_adm = int(analysis.get("n_admitted", 0) or 0)
    coeffs = resolve_comp_weight_coeffs(
        k2_bprp=None,
        airmass_span=0.0,
        r_deg=None,
        residual_scatter_mag=None,
    )
    comp_csv = base / "photometry/comparison_stars_per_target.csv"
    pool_ids = sorted(pool["catalog_id"].astype(str).tolist()) if "catalog_id" in pool.columns else []
    bo_ids = []
    fw_n = 0
    if comp_csv.is_file():
        cdf = pd.read_csv(comp_csv, low_memory=False)
        tcol = "target_catalog_id" if "target_catalog_id" in cdf.columns else None
        ncol = next((c for c in ("target_name", "vsx_name", "target") if c in cdf.columns), None)
        if tcol and "catalog_id" in cdf.columns:
            bo = cdf[cdf[tcol].astype(str) == "1498613634033133184"]
            bo_ids = sorted(bo["catalog_id"].astype(str).unique().tolist())
        if ncol and "catalog_id" in cdf.columns:
            fw = cdf[cdf[ncol].astype(str).str.contains("FW", case=False, na=False)]
            fw_n = int(fw["catalog_id"].nunique()) if not fw.empty else 0
    return {
        "draft_id": draft_id,
        "n_masterstars": int(len(ms)),
        "n_proc": len(paths),
        "n_summarized": n_sum,
        "n_admitted_decisions": n_adm,
        "n_global_pool": int(len(pool)),
        "pool_empty": bool(pool.empty),
        "coeffs": {
            "c_col": coeffs.c_col_mag_per_bprp,
            "c_dist": coeffs.c_dist_mag_per_deg,
            "c_col_source": coeffs.c_col_source,
            "c_dist_source": coeffs.c_dist_source,
            "notes": list(coeffs.notes),
        },
        "bo_cvn_archived_comps": bo_ids,
        "bo_cvn_in_new_pool": [c for c in bo_ids if c in set(pool_ids)],
        "fw_cvn_archived_n_comps": fw_n,
    }


def main() -> None:
    out = {str(d): run_one(d) for d in (512, 513, 435)}
    # BO intersection 512 vs 513
    a = set(out["512"].get("bo_cvn_in_new_pool") or [])
    b = set(out["513"].get("bo_cvn_in_new_pool") or [])
    out["bo_cvn_pool_intersection_512_513"] = sorted(a & b)
    out["bo_cvn_pool_union_512_513"] = sorted(a | b)
    path = REPO / "dev/results/COMP_ADMIT_03_measurements.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
