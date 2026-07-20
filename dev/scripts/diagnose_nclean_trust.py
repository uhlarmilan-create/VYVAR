#!/usr/bin/env python3
"""Diagnostic: n_clean=0 / trust RED - regression vs draft-specific (read-only)."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from comp_qa_core import compute_comp_qa, load_proc_pivot  # noqa: E402
from config import AppConfig  # noqa: E402
from proc_frame_store import PROC_CSV_GLOB, list_proc_csvs  # noqa: E402
from trust_flag_core import (  # noqa: E402
    comp_thresholds_from_config,
    run_trust_flag_for_photometry_dir,
)

OUT_DIR = _ROOT / "tmp" / "diagnostic_nclean_trust"
DRAFT_366 = _ROOT / "Archive" / "Drafts" / "draft_000366"
DRAFT_380 = _ROOT / "Archive" / "Drafts" / "draft_000380"
SETUP_366 = "NoFilter_60_2"
SETUP_380 = "B_20_2"


def _phot_paths(draft: Path, setup: str) -> tuple[Path, Path]:
    ps = draft / "platesolve" / setup
    return ps / "photometry", draft / "detrended_aligned" / "lights" / setup


def _backup_summary(phot_dir: Path) -> Path | None:
    src = phot_dir / "photometry_summary.csv"
    if not src.is_file():
        return None
    bak = phot_dir / "photometry_summary.csv.bak_diagnostic"
    if not bak.is_file():
        shutil.copy2(src, bak)
    return bak


def _restore_summary(phot_dir: Path, bak: Path | None) -> None:
    if bak and bak.is_file():
        shutil.copy2(bak, phot_dir / "photometry_summary.csv")


def _original_snapshot(phot_dir: Path) -> dict:
    sm = phot_dir / "photometry_summary.csv"
    if not sm.is_file():
        return {"error": "no summary"}
    df = pd.read_csv(sm, dtype=str, low_memory=False)
    nc = pd.to_numeric(df.get("n_clean"), errors="coerce")
    trust = df.get("trust", pd.Series(dtype=str)).astype(str).str.upper()
    nf = pd.to_numeric(df.get("n_frames"), errors="coerce")
    ng = pd.to_numeric(df.get("n_good_comp"), errors="coerce")
    return {
        "n_targets": len(df),
        "n_clean_nan": int(nc.isna().sum()),
        "n_clean_eq0": int((nc.fillna(-1) == 0).sum()),
        "n_clean_gt0": int((nc.fillna(0) > 0).sum()),
        "n_clean_median": float(nc.median()) if nc.notna().any() else None,
        "n_clean_max": float(nc.max()) if nc.notna().any() else None,
        "trust_GREEN": int((trust == "GREEN").sum()),
        "trust_YELLOW": int((trust == "YELLOW").sum()),
        "trust_RED": int((trust == "RED").sum()),
        "n_frames_median": float(nf.median()) if nf.notna().any() else None,
        "n_good_comp_median": float(ng.median()) if ng.notna().any() else None,
    }


def _rerun_comp_qa_trust(*, phot_dir: Path, proc_dir: Path, cfg: AppConfig) -> dict:
    bak = _backup_summary(phot_dir)
    # copy summary to temp work - mutate in place then we'll restore
    work_sm = phot_dir / "photometry_summary.csv"
    df_before = pd.read_csv(work_sm, dtype=str, low_memory=False) if work_sm.is_file() else pd.DataFrame()

    qa = compute_comp_qa(
        photometry_dir=phot_dir,
        proc_dir=proc_dir,
        min_comps=int(cfg.phase01_comparison_n_comp_min),
        max_comps=int(cfg.phase01_comparison_n_comp_max),
    )
    from comp_qa_core import write_comp_qa_artifacts  # noqa: PLC0415

    write_comp_qa_artifacts(qa, photometry_dir=phot_dir, update_summary=True)
    trust = run_trust_flag_for_photometry_dir(
        photometry_dir=phot_dir, cfg=cfg, update_summary=True
    )
    df_after = pd.read_csv(work_sm, dtype=str, low_memory=False)
    snap = _original_snapshot(phot_dir)
    snap["qa_stats"] = qa.get("stats", {})
    snap["qa_n_targets_in_result"] = len(qa.get("per_target", {}))
    snap["qa_per_comp_rows"] = len(qa.get("per_comp_rows", []))
    snap["trust_stats"] = trust.get("stats", {})
    _restore_summary(phot_dir, bak)
    return snap


def _proc_glob_diag(proc_dir: Path) -> dict:
    all_proc = list_proc_csvs(proc_dir)
    light = [p for p in all_proc if "_Light_" in p.name]
    return {
        "proc_dir": str(proc_dir),
        "n_proc_light_glob": len(light),
        "n_proc_all_glob": len(all_proc),
        "sample_names": [p.name for p in all_proc[:3]],
    }


def _dump_380_comp_qa(*, phot_dir: Path, proc_dir: Path, cfg: AppConfig, out_csv: Path) -> dict:
    qa = compute_comp_qa(
        photometry_dir=phot_dir,
        proc_dir=proc_dir,
        min_comps=int(cfg.phase01_comparison_n_comp_min),
        max_comps=int(cfg.phase01_comparison_n_comp_max),
    )
    rows = qa.get("per_comp_rows", [])
    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    else:
        out_csv.write_text("EMPTY - no per_comp_rows (load_proc_pivot found no frames)\n", encoding="utf-8")

    # sample targets
    pt = qa.get("per_target", {})
    sample_targets = list(pt.keys())[:5]
    target_samples = {tid: pt[tid] for tid in sample_targets}

    # direct load_proc_pivot test on first target from comparison_stars_per_target
    cpt = phot_dir / "comparison_stars_per_target.csv"
    pivot_diag = {}
    if cpt.is_file():
        comps = pd.read_csv(cpt, dtype=str)
        if not comps.empty:
            tid = str(comps.iloc[0]["target_catalog_id"]).strip()
            comp_ids = comps[comps["target_catalog_id"] == tid]["catalog_id"].astype(str).tolist()
            ids = set(comp_ids) | {tid}
            fw, td = load_proc_pivot(proc_dir, ids)
            pivot_diag = {
                "test_target": tid,
                "n_comp_ids": len(comp_ids),
                "pivot_empty": fw.empty,
                "pivot_shape": list(fw.shape) if not fw.empty else None,
                "pivot_columns_sample": list(fw.columns[:5]) if not fw.empty else [],
            }

    return {
        "qa_stats": qa.get("stats", {}),
        "n_targets_in_qa": len(pt),
        "n_per_comp_rows": len(rows),
        "target_samples": target_samples,
        "pivot_diag": pivot_diag,
        "glob": _proc_glob_diag(proc_dir),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    th = comp_thresholds_from_config(cfg)

    phot366, proc366 = _phot_paths(DRAFT_366, SETUP_366)
    phot380, proc380 = _phot_paths(DRAFT_380, SETUP_380)

    report: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "trust_thresholds": {"min_comps": th.min_comps, "max_comps": th.max_comps, "strong": th.strong},
        "load_proc_pivot_pattern": PROC_CSV_GLOB,
    }

    report["draft_366_original"] = _original_snapshot(phot366)
    report["draft_366_glob"] = _proc_glob_diag(proc366)
    report["draft_366_rerun"] = _rerun_comp_qa_trust(phot_dir=phot366, proc_dir=proc366, cfg=cfg)

    report["draft_380_original"] = _original_snapshot(phot380)
    report["draft_380_glob"] = _proc_glob_diag(proc380)
    report["draft_380_rerun"] = _rerun_comp_qa_trust(phot_dir=phot380, proc_dir=proc380, cfg=cfg)

    dump_csv = OUT_DIR / "comp_qa_per_comp_draft380_B_20_2.csv"
    report["draft_380_comp_dump"] = _dump_380_comp_qa(
        phot_dir=phot380, proc_dir=proc380, cfg=cfg, out_csv=dump_csv
    )

    # compare frames
    sm366 = pd.read_csv(phot366 / "photometry_summary.csv", dtype=str, low_memory=False)
    sm380 = pd.read_csv(phot380 / "photometry_summary.csv", dtype=str, low_memory=False)
    report["comparison_366_vs_380"] = {
        "366_n_frames_median": float(pd.to_numeric(sm366.get("n_frames"), errors="coerce").median()),
        "380_n_frames_median": float(pd.to_numeric(sm380.get("n_frames"), errors="coerce").median()),
        "366_calibrated": "calibrated/lights (Jirny V842 Her)",
        "380_pre_calibrated": "non_calibrated/lights Chi_and_H",
        "366_proc_light_glob_hits": report["draft_366_glob"]["n_proc_light_glob"],
        "380_proc_light_glob_hits": report["draft_380_glob"]["n_proc_light_glob"],
        "366_comp_qa_targets_computed": report["draft_366_rerun"]["qa_n_targets_in_result"],
        "380_comp_qa_targets_computed": report["draft_380_rerun"]["qa_n_targets_in_result"],
    }

    out_json = OUT_DIR / "diagnose_nclean_trust_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
