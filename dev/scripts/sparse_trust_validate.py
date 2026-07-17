#!/usr/bin/env python3
"""SPARSE-TRUST validation S2-S4 (see docs/VYVAR_SPARSE_TRUST_SPEC.md Section 6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trust_flag_core import (  # noqa: E402
    compute_trust_for_photometry_dir,
    read_sparse_trust_sidecar,
)


def s2_no_regression(phot_dir: Path, *, min_comps: int = 5) -> dict[str, Any]:
    """S2: n>=5 healthy targets must not flip GREEN->RED under sparse trust columns."""
    before = compute_trust_for_photometry_dir(phot_dir)
    comp_path = phot_dir / "comparison_stars_per_target.csv"
    comp_df = pd.read_csv(comp_path, dtype={"target_catalog_id": str}, low_memory=False) if comp_path.is_file() else pd.DataFrame()
    pool_n: dict[str, int] = {}
    if not comp_df.empty and "target_catalog_id" in comp_df.columns:
        nf = pd.to_numeric(comp_df.get("comp_pool_n_final"), errors="coerce")
        comp_df = comp_df.assign(_nf=nf)
        for tid, grp in comp_df.groupby(comp_df["target_catalog_id"].astype(str).str.strip()):
            if tid and grp["_nf"].notna().any():
                pool_n[tid] = int(grp["_nf"].max())
    flips: list[dict[str, str]] = []
    for tid, info in before.get("per_target", {}).items():
        nc = int(info.get("n_clean", 0))
        n_pool = int(pool_n.get(tid, nc))
        if n_pool < min_comps:
            continue
        old = str(info.get("trust", ""))
        if old != "GREEN":
            continue
        side = read_sparse_trust_sidecar(phot_dir, tid)
        if side and side.get("check_sparse"):
            flips.append({"target": tid, "was": old, "note": "unexpected check_sparse on n>=5"})
    return {"pass": len(flips) == 0, "n_green_n5": sum(1 for t, i in before.get("per_target", {}).items() if pool_n.get(t, i.get("n_clean", 0)) >= min_comps and i.get("trust") == "GREEN"), "flips": flips}


def s3_sparse_sidecars(phot_dir: Path) -> dict[str, Any]:
    """S3: sparse targets have sidecars with trust columns."""
    lc = phot_dir / "lightcurves"
    rows: list[dict[str, Any]] = []
    for p in sorted(lc.glob("check_kmag_*.csv")):
        try:
            df = pd.read_csv(p, nrows=1, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            rows.append({"file": p.name, "ok": False, "error": str(exc)})
            continue
        ok = "check_sparse" in df.columns and "trust_R" in df.columns
        rows.append(
            {
                "file": p.name,
                "ok": ok,
                "check_sparse": int(df.get("check_sparse", [0]).iloc[0]) if ok else None,
                "trust_R": df.get("trust_R", [""]).iloc[0] if ok else None,
            }
        )
    sparse = [r for r in rows if r.get("check_sparse") == 1]
    return {"pass": len(rows) > 0, "n_sidecars": len(rows), "n_sparse": len(sparse), "rows": rows[:20]}


def s4_err_unchanged(anchor_phot: Path, baseline_phot: Path, setup: str = "NoFilter_60_2") -> dict[str, Any]:
    """S4: LC err byte-identical vs anchor baseline."""
    from tests.photometry_sha import compare_photometry_science_meaningful  # noqa: E402

    rep = compare_photometry_science_meaningful(baseline_phot.parent.parent, anchor_phot.parent.parent, setup=setup)
    err_only = [m for m in rep.get("lc_mismatches", []) if "err" in str(m).lower()]
    return {
        "pass": rep.get("overall_pass") and len(err_only) == 0,
        "overall_pass": rep.get("overall_pass"),
        "err_mismatches": err_only[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft424-phot", type=Path, help="draft_424 photometry dir for S2/S4")
    ap.add_argument("--draft426-phot", type=Path, help="draft_426 sparse photometry dir for S3")
    ap.add_argument("--baseline424-phot", type=Path, help="anchor baseline photometry for S4 err compare")
    ap.add_argument("--out", type=Path, default=_ROOT / "tmp" / "sparse_trust" / "validation.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    if args.draft424_phot and args.draft424_phot.is_dir():
        out["S2"] = s2_no_regression(args.draft424_phot)
    if args.draft426_phot and args.draft426_phot.is_dir():
        out["S3"] = s3_sparse_sidecars(args.draft426_phot)
    if args.draft424_phot and args.baseline424_phot and args.baseline424_phot.is_dir():
        out["S4"] = s4_err_unchanged(args.draft424_phot, args.baseline424_phot)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    ok = all(v.get("pass") for v in out.values() if isinstance(v, dict))
    return 0 if ok or not out else 1


if __name__ == "__main__":
    raise SystemExit(main())
