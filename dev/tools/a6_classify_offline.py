#!/usr/bin/env python3
"""Offline DAO_ONLY magnitude classification on stored masterstars catalogues (A-6)."""

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
)
from gaia_catalog_id import read_vyvar_csv  # noqa: E402

DRAFTS = (
    ("draft_000501", "V_60_2"),
    ("draft_000435_snapshot_skysurface_20260716", "NoFilter_60_2"),
    ("draft_000500", "NoFilter_60_2"),
)


def classify_csv(csv_path: Path, gaia_db_path: str) -> dict:
    df = read_vyvar_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
    if "source_type" not in df.columns:
        cid = df.get("catalog_id", pd.Series([""] * len(df))).fillna("").astype(str).str.strip()
        df["source_type"] = pd.Series(["DAO_ONLY"] * len(df)).where(cid.eq(""), "GAIA_MATCHED")
    fleming_sigma = None
    pm_path = csv_path.parent / "photometry" / "pipeline_meta.json"
    if pm_path.is_file():
        try:
            pm = json.loads(pm_path.read_text(encoding="utf-8"))
            curve = pm.get("completeness_curve") or []
            if curve:
                fleming_sigma = fit_fleming_completeness(curve).sigma_mag
        except Exception:  # noqa: BLE001
            pass
    _df, meta = annotate_dao_only_magnitude_classes(
        df,
        gaia_db_path=gaia_db_path,
        fleming_sigma_mag=fleming_sigma,
    )
    meta["census_log"] = format_dao_only_census_log(meta, n_total=len(df))
    meta["csv_path"] = str(csv_path)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="A-6 offline DAO_ONLY classification")
    ap.add_argument("--out", type=Path, default=ROOT / "tmp" / "a6_offline_counts.json")
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for key, meta in results.items():
        if "error" in meta:
            print(f"{key}: {meta['error']}")
            continue
        counts = meta.get("counts") or {}
        print(f"{key}: n_dao={meta.get('n_dao_only')} counts={counts}")
        print(f"  {meta.get('census_log')}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
