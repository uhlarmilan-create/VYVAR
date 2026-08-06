#!/usr/bin/env python3
"""Offline DAO_ONLY magnitude classification on stored masterstars catalogues (A-6/A-6b)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from config import AppConfig  # noqa: E402
from dao_reconcile import (  # noqa: E402
    annotate_dao_only_magnitude_classes,
    fit_fleming_completeness,
    format_dao_only_census_log,
    resolve_effective_match_depth,
)
from gaia_catalog_id import read_vyvar_csv  # noqa: E402

DRAFTS = (
    ("draft_000501", "V_60_2"),
    ("draft_000435_snapshot_skysurface_20260716", "NoFilter_60_2"),
    ("draft_000500", "NoFilter_60_2"),
)

A6_COUNTS = {
    "draft_000501/V_60_2": {
        "artifact_negative": 142,
        "below_catalogue": 525,
        "unconfirmed_bright": 14,
        "indeterminate": 15,
    },
    "draft_000435_snapshot_skysurface_20260716/NoFilter_60_2": {
        "artifact_negative": 8,
        "below_catalogue": 0,
        "unconfirmed_bright": 98,
        "indeterminate": 3,
    },
    "draft_000500/NoFilter_60_2": {
        "artifact_negative": 48,
        "below_catalogue": 8,
        "unconfirmed_bright": 496,
        "indeterminate": 9,
    },
}


def classify_csv(csv_path: Path, gaia_db_path: str) -> dict:
    df = read_vyvar_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
    if "source_type" not in df.columns:
        cid = df.get("catalog_id", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        df["source_type"] = pd.Series(["DAO_ONLY"] * len(df)).where(cid.eq(""), "GAIA_MATCHED")
    pm_path = csv_path.parent / "photometry" / "pipeline_meta.json"
    det_meta: dict = {}
    fleming_sigma = None
    match_depth = None
    cone_lim = None
    noise = None
    if pm_path.is_file():
        try:
            pm = json.loads(pm_path.read_text(encoding="utf-8"))
            det_meta = {
                "faintest_mag_limit": pm.get("faintest_mag_limit"),
                "provenance": pm.get("provenance"),
            }
            md = resolve_effective_match_depth(det_meta, is_masterstar=True)
            match_depth = md.get("match_depth")
            if det_meta.get("faintest_mag_limit") is not None:
                cone_lim = float(det_meta["faintest_mag_limit"])
            noise = pm.get("noise_floor_adu")
            curve = pm.get("completeness_curve") or []
            if curve:
                fleming_sigma = fit_fleming_completeness(curve).sigma_mag
        except Exception:  # noqa: BLE001
            pass
    _df, meta = annotate_dao_only_magnitude_classes(
        df,
        gaia_db_path=gaia_db_path,
        effective_match_depth=match_depth,
        cone_query_mag_limit=cone_lim,
        fleming_sigma_mag=fleming_sigma,
        frame_noise_adu=noise,
    )
    meta["census_log"] = format_dao_only_census_log(meta, n_total=len(df))
    meta["csv_path"] = str(csv_path)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="A-6b offline DAO_ONLY classification")
    ap.add_argument("--out", type=Path, default=ROOT / "tmp" / "a6b_offline_counts.json")
    args = ap.parse_args()
    cfg = AppConfig()
    gaia_db = str(cfg.gaia_db_path or "").strip()
    if not gaia_db:
        raise SystemExit("gaia_db_path not configured")
    archive = Path(cfg.archive_root) / "Drafts"
    results: dict[str, dict] = {}
    for draft, setup in DRAFTS:
        csv_path = archive / draft / "platesolve" / setup / "masterstars_full_match.csv"
        key = f"{draft}/{setup}"
        if not csv_path.is_file():
            results[key] = {"error": f"missing {csv_path}"}
            continue
        results[key] = classify_csv(csv_path, gaia_db)
        a6 = A6_COUNTS.get(key, {})
        a6b = results[key].get("counts") or {}
        if a6:
            results[key]["delta_vs_a6"] = {
                "artifact_negative": int(a6b.get("artifact_negative", 0)) - int(a6.get("artifact_negative", 0)),
                "unmatched_in_range_vs_unconfirmed_bright": int(a6b.get("unmatched_in_range", 0))
                - int(a6.get("unconfirmed_bright", 0)),
                "beyond_catalogue_vs_below_catalogue": int(a6b.get("beyond_catalogue", 0))
                - int(a6.get("below_catalogue", 0)),
                "ambiguous_depth_new": int(a6b.get("ambiguous_depth", 0)),
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for key, meta in results.items():
        if "error" in meta:
            print(f"{key}: {meta['error']}")
            continue
        counts = meta.get("counts") or {}
        print(
            f"{key}: depth={meta.get('confirmable_depth_g')} winner={meta.get('confirmable_depth_winner')} "
            f"sigma_g_med={meta.get('sigma_g_row_median')} counts={counts}"
        )
        print(f"  {meta.get('census_log')}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
