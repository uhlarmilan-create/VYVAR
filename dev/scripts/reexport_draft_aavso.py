#!/usr/bin/env python3
"""Re-run AAVSO/VarAstro export from existing Phase 2A LC CSVs (no photometry recompute)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_PROJECT_ROOT = _bootstrap.REPO_ROOT
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from config import AppConfig
from citations import build_run_citation_context, load_pipeline_meta
from export_reports import export_all_method_lightcurve_reports
from gaia_catalog_id import normalize_gaia_source_id
from photometry_core import parse_comp_quality_json_map


def _norm_id(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-export lightcurve reports for a draft setup")
    parser.add_argument("--draft", type=int, required=True)
    parser.add_argument("--setup", type=str, default="NoFilter_60_2")
    args = parser.parse_args()

    root = _PROJECT_ROOT
    cfg = AppConfig()
    phot = root / "Archive" / "Drafts" / f"draft_{args.draft:06d}" / "platesolve" / args.setup / "photometry"
    if not phot.is_dir():
        print(f"FATAL: photometry dir not found: {phot}", file=sys.stderr)
        return 1

    lc_dir = phot / "lightcurves"
    reports_dir = phot / "lightcurves_reports"
    (reports_dir / "aavso").mkdir(parents=True, exist_ok=True)

    at_df = pd.read_csv(phot / "active_targets.csv", dtype={"catalog_id": str})
    sum_df = pd.read_csv(phot / "photometry_summary.csv", dtype={"catalog_id": str})
    comp_all = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )

    sum_by: dict[str, pd.Series] = {}
    for _, r in sum_df.iterrows():
        cid = str(r.get("catalog_id") or "").strip()
        if cid:
            sum_by[cid] = r

    comp_index: dict[str, pd.DataFrame] = {}
    for tid, sub in comp_all.groupby("target_catalog_id"):
        comp_index[_norm_id(tid)] = sub.copy()

    run_cite = build_run_citation_context(
        cfg,
        pipeline_meta=load_pipeline_meta(phot),
        targets_df=at_df,
    )

    proc_cache: dict[str, pd.DataFrame] = {}
    n_ok = 0
    for _, trow in at_df.iterrows():
        target_cid = _norm_id(trow.get("catalog_id", ""))
        if not target_cid:
            continue
        lc_csv = lc_dir / f"lightcurve_{target_cid}.csv"
        if not lc_csv.is_file():
            continue
        comp_target = comp_index.get(target_cid, pd.DataFrame()).copy()
        srow = sum_by.get(target_cid, pd.Series(dtype=object))
        cq_path = lc_dir / f"comp_quality_{target_cid}.json"
        comp_qmap: dict[str, str] = {}
        if cq_path.is_file():
            raw = json.loads(cq_path.read_text(encoding="utf-8"))
            for qk, qv in parse_comp_quality_json_map(raw).items():
                nk = _norm_id(qk)
                q2 = str(qv.get("quality", "")).strip().lower()
                if q2 != "excluded":
                    comp_qmap[nk] = q2
        paths = export_all_method_lightcurve_reports(
            reports_dir,
            trow,
            lc_dir=lc_dir,
            target_cid=target_cid,
            comp_df=comp_target,
            summary_row=srow,
            observer_code=str(cfg.observer_code or ""),
            observer_name=str(cfg.observer_name or "Unknown Observer"),
            comp_quality_map=comp_qmap or None,
            arcsec_per_px=float(cfg.export_arcsec_per_px),
            software_version="VYVAR 1.0",
            cfg=cfg,
            obs_group=args.setup,
            targets_df=at_df,
            run_citation_ctx=run_cite,
            proc_csv_cache=proc_cache,
        )
        if paths:
            n_ok += 1

    print(f"Re-exported {n_ok} targets -> {reports_dir / 'aavso'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
