#!/usr/bin/env python3
"""Audit Stage 3 Part 0d: delta-tail forensics (read-only analysis harness)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "scripts"))
import _bootstrap  # noqa: E402,F401

sys.path.insert(0, str(_bootstrap.REPO_ROOT / "src_py"))

from config import AppConfig  # noqa: E402
from photometry_core import _resolve_git_provenance  # noqa: E402

SETUP = "NoFilter_60_2"
REBUILD_DRAFT = 499
ANCHOR_SNAPSHOT = "draft_000435_snapshot_skysurface_20260716"
FOCUS_TARGET = "1498135552633294976"


def _pair_positional(adf: pd.DataFrame, rdf: pd.DataFrame) -> pd.DataFrame:
    n = min(len(adf), len(rdf))
    dm = pd.to_numeric(rdf["mag_calib_final"].iloc[:n], errors="coerce") - pd.to_numeric(
        adf["mag_calib_final"].iloc[:n], errors="coerce"
    )
    return pd.DataFrame({"delta_mag": dm})


def _pair_source_file(adf: pd.DataFrame, rdf: pd.DataFrame) -> pd.DataFrame:
    m = adf.merge(rdf, on="source_file", suffixes=("_an", "_rb"))
    m["delta_mag"] = pd.to_numeric(m["mag_calib_final_rb"], errors="coerce") - pd.to_numeric(
        m["mag_calib_final_an"], errors="coerce"
    )
    return m


def _comp_sets(comp_df: pd.DataFrame, target_cid: str) -> set[str]:
    sub = comp_df.loc[comp_df["target_catalog_id"].astype(str).str.strip() == str(target_cid)]
    return set(sub["catalog_id"].astype(str).str.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=REPO / "tmp" / "audit_stage3_part0d_results.json")
    args = parser.parse_args()

    cfg = AppConfig()
    rebuild = Path(cfg.archive_root) / "Drafts" / f"draft_{REBUILD_DRAFT:06d}"
    anchor = Path(cfg.archive_root) / "Drafts" / ANCHOR_SNAPSHOT
    an_ps = anchor / "platesolve" / SETUP
    rb_ps = rebuild / "platesolve" / SETUP
    an_lc = an_ps / "photometry" / "lightcurves"
    rb_lc = rb_ps / "photometry" / "lightcurves"

    gh, dirty, _ = _resolve_git_provenance()
    common = sorted(
        {p.stem.replace("lightcurve_", "") for p in an_lc.glob("lightcurve_*.csv")}
        & {p.stem.replace("lightcurve_", "") for p in rb_lc.glob("lightcurve_*.csv")}
    )

    ca = pd.read_csv(an_ps / "photometry" / "comparison_stars_per_target.csv", low_memory=False, dtype={"catalog_id": str, "target_catalog_id": str})
    cr = pd.read_csv(rb_ps / "photometry" / "comparison_stars_per_target.csv", low_memory=False, dtype={"catalog_id": str, "target_catalog_id": str})

    pos_rows: list[dict[str, Any]] = []
    sf_rows: list[dict[str, Any]] = []
    for cid in common:
        adf = pd.read_csv(an_lc / f"lightcurve_{cid}.csv", low_memory=False)
        rdf = pd.read_csv(rb_lc / f"lightcurve_{cid}.csv", low_memory=False)
        dp = _pair_positional(adf, rdf)["delta_mag"].abs()
        ds = _pair_source_file(adf, rdf)["delta_mag"].abs()
        ens = _comp_sets(ca, cid) != _comp_sets(cr, cid)
        if len(dp):
            pos_rows.append({"cid": cid, "p95": float(dp.quantile(0.95)), "max": float(dp.max()), "ensemble_changed": ens})
        if len(ds):
            sf_rows.append({"cid": cid, "p95": float(ds.quantile(0.95)), "max": float(ds.max()), "ensemble_changed": ens})

    pos_df = pd.DataFrame(pos_rows)
    sf_df = pd.DataFrame(sf_rows)

    # Focus target detail
    adf = pd.read_csv(an_lc / f"lightcurve_{FOCUS_TARGET}.csv", low_memory=False)
    rdf = pd.read_csv(rb_lc / f"lightcurve_{FOCUS_TARGET}.csv", low_memory=False)
    n = min(len(adf), len(rdf))
    order_match = int((adf["source_file"].iloc[:n].values == rdf["source_file"].iloc[:n].values).sum())

    out: dict[str, Any] = {
        "provenance": {"git_hash": gh, "git_dirty": dirty},
        "a1_pairing": {
            "method_0c": "positional index iloc[:min(len)]",
            "anchor_lc_rows_focus": int(len(adf)),
            "rebuild_lc_rows_focus": int(len(rdf)),
            "common_source_files_focus": int(len(set(adf["source_file"]) & set(rdf["source_file"]))),
            "positional_order_match_focus": order_match,
            "part_0c_delta_table_valid": False,
        },
        "cohort_positional": {
            "median_p95": float(pos_df["p95"].median()),
            "max_p95": float(pos_df["p95"].max()),
            "worst_cid": str(pos_df.loc[pos_df["p95"].idxmax(), "cid"]),
        },
        "cohort_source_file": {
            "median_p95": float(sf_df["p95"].median()),
            "max_p95": float(sf_df["p95"].max()),
            "worst_cid": str(sf_df.loc[sf_df["p95"].idxmax(), "cid"]),
            "count_p95_gt_0.1": int((sf_df["p95"] > 0.1).sum()),
            "count_p95_gt_0.5": int((sf_df["p95"] > 0.5).sum()),
            "count_p95_gt_1.0": int((sf_df["p95"] > 1.0).sum()),
            "ensemble_changed_n": int(sf_df["ensemble_changed"].sum()),
        },
        "focus_target": FOCUS_TARGET,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
