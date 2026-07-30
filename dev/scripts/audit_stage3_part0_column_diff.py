#!/usr/bin/env python3
"""Column-level photometry diff: fresh run vs anchor snapshot (Stage 3 Part 0)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src_py", REPO / "dev", REPO):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tests.photometry_sha import (  # noqa: E402
    PHOTOMETRY_PROVENANCE_COLS,
    PHOTOMETRY_QC_COLS_LC,
    PHOTOMETRY_SCIENCE_COLS_LC,
    PHOTOMETRY_TIME_COLS,
    TOL_SCIENCE,
    TOL_TIME_D,
    _compare_lc_science,
    _lc_map,
    compute_photometry_sha,
)


def _classify_column(col: str) -> str:
    c = col.lower()
    if col in PHOTOMETRY_PROVENANCE_COLS:
        return "provenance"
    if col in PHOTOMETRY_QC_COLS_LC:
        return "qc"
    if c in PHOTOMETRY_TIME_COLS:
        return "time"
    if c in PHOTOMETRY_SCIENCE_COLS_LC or c.startswith("mag") or c.startswith("flux"):
        return "science"
    return "other"


def compare_roots(run_root: Path, snap_root: Path, setup: str) -> dict:
    run_root = Path(run_root)
    snap_root = Path(snap_root)
    run_lc = _lc_map(run_root, setup)
    snap_lc = _lc_map(snap_root, setup)
    common = sorted(set(run_lc) & set(snap_lc))
    col_stats: dict[str, dict] = defaultdict(
        lambda: {"n_targets_with_delta": 0, "max_delta": 0.0, "targets_sample": []}
    )
    science_fail = 0
    time_fail = 0
    missing_run = sorted(set(snap_lc) - set(run_lc))
    missing_snap = sorted(set(run_lc) - set(snap_lc))
    per_target: list[dict] = []

    for tid in common:
        cmp = _compare_lc_science(run_lc[tid], snap_lc[tid])
        per_target.append({"target_id": tid, **cmp})
        if not cmp.get("science_ok", True):
            science_fail += 1
        if not cmp.get("time_ok", True):
            time_fail += 1
        for col, delta in (cmp.get("max_delta") or {}).items():
            st = col_stats[col]
            st["n_targets_with_delta"] += 1
            st["max_delta"] = max(float(st["max_delta"]), float(delta))
            if len(st["targets_sample"]) < 5:
                st["targets_sample"].append({"target_id": tid, "max_delta": float(delta)})

    explained_p10 = {
        "mag_inst",
        "mag_calib_raw",
        "mag_calib",
        "mag_calib_ct",
        "mag_calib_ac",
        "mag_calib_final",
        "delta_mag",
        "mag_democratic",
    }
    explained_dao = {"comparison_stars_per_target", "active_targets", "masterstars_full_match"}

    column_report = []
    for col, st in sorted(col_stats.items()):
        kind = _classify_column(col)
        tol = TOL_TIME_D if col.lower() in PHOTOMETRY_TIME_COLS else TOL_SCIENCE
        exceeds = float(st["max_delta"]) > tol if kind in ("science", "time") else False
        note = "unexplained"
        if col in explained_p10:
            note = "likely P-10 sky-surface (photometry path)"
        elif kind == "qc":
            note = "QC/metadata (non-science SHA if excluded)"
        elif kind == "provenance":
            note = "provenance"
        column_report.append(
            {
                "column": col,
                "kind": kind,
                "n_targets_with_delta": st["n_targets_with_delta"],
                "max_delta": st["max_delta"],
                "exceeds_tol": exceeds,
                "explanation": note,
            }
        )

    run_core, run_core_n = compute_photometry_sha(run_root, include_comp_qa=False)
    run_ext, run_ext_n = compute_photometry_sha(run_root, include_comp_qa=True)
    snap_core, snap_core_n = compute_photometry_sha(snap_root, include_comp_qa=False)
    snap_ext, snap_ext_n = compute_photometry_sha(snap_root, include_comp_qa=True)

    return {
        "setup": setup,
        "n_lc_common": len(common),
        "n_lc_missing_run": len(missing_run),
        "n_lc_missing_snap": len(missing_snap),
        "science_fail_targets": science_fail,
        "time_fail_targets": time_fail,
        "sha": {
            "run_core": run_core,
            "run_core_n": run_core_n,
            "run_ext": run_ext,
            "run_ext_n": run_ext_n,
            "snap_core": snap_core,
            "snap_core_n": snap_core_n,
            "snap_ext": snap_ext,
            "snap_ext_n": snap_ext_n,
        },
        "columns": column_report,
        "missing_run_sample": missing_run[:10],
        "missing_snap_sample": missing_snap[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True, help="Fresh pipeline output draft root")
    ap.add_argument("--snap-root", type=Path, required=True, help="Anchor snapshot draft root")
    ap.add_argument("--setup", default="NoFilter_60_2")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    payload = compare_roots(args.run_root, args.snap_root, args.setup)
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
