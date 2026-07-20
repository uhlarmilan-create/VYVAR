#!/usr/bin/env python3
"""Validate proximity tie-break on draft_000365 (comp set diff + LC impact)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402

DRAFT = 365
SETUP = "NoFilter_60_2"
PHOT = _ROOT / "Archive" / "Drafts" / f"draft_{DRAFT:06d}" / "platesolve" / SETUP / "photometry"
BEFORE = PHOT / "comparison_stars_per_target.pre_proximity.csv"
AFTER = PHOT / "comparison_stars_per_target.csv"
LC = PHOT / "lightcurves"


def _norm(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def _load_sets(path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(path, dtype={"catalog_id": str, "target_catalog_id": str})
    out: dict[str, set[str]] = {}
    for tid, grp in df.groupby(df["target_catalog_id"].astype(str)):
        out[_norm(tid)] = {_norm(c) for c in grp["catalog_id"] if _norm(c)}
    return out


def main() -> int:
    if not BEFORE.is_file() or not AFTER.is_file():
        print("FATAL: missing before/after comp CSV", file=sys.stderr)
        return 1

    before = _load_sets(BEFORE)
    after = _load_sets(AFTER)
    targets = sorted(set(before) & set(after))
    changed: list[dict] = []
    for tid in targets:
        b, a = before[tid], after[tid]
        if b != a:
            changed.append(
                {
                    "target": tid,
                    "n_before": len(b),
                    "n_after": len(a),
                    "n_swapped": len(b ^ a),
                    "only_before": sorted(b - a),
                    "only_after": sorted(a - b),
                }
            )

    print(f"targets_common={len(targets)} changed={len(changed)}")
    swaps = [c["n_swapped"] for c in changed]
    if swaps:
        print(f"comps_swapped_per_changed_target: min={min(swaps)} med={int(np.median(swaps))} max={max(swaps)}")

    summ_path = PHOT / "photometry_summary.csv"
    summ_before = PHOT / "photometry_summary.pre_proximity.csv"
    lc_deltas: list[float] = []
    mag_deltas: list[float] = []
    if summ_before.is_file() and summ_path.is_file():
        sb = pd.read_csv(summ_before, dtype={"catalog_id": str})
        sa = pd.read_csv(summ_path, dtype={"catalog_id": str})
        sb["_n"] = sb["catalog_id"].map(_norm)
        sa["_n"] = sa["catalog_id"].map(_norm)
        merged = sb.merge(sa, on="_n", suffixes=("_b", "_a"), how="inner")
        changed_ids = {_norm(c["target"]) for c in changed}
        sub = merged[merged["_n"].isin(changed_ids)]
        if "lc_rms" in merged.columns or "lc_rms_b" in merged.columns:
            rb = pd.to_numeric(
                sub.get("lc_rms_b", sub.get("lc_rms")), errors="coerce"
            )
            ra = pd.to_numeric(
                sub.get("lc_rms_a", sub.get("lc_rms")), errors="coerce"
            )
            d = (ra - rb).abs()
            lc_deltas = d[d.notna()].tolist()
        # median |Deltamag| from LC files for changed targets
        for tid in changed_ids:
            lc_b = LC / f"lightcurve_{tid}.csv"
            if not lc_b.is_file():
                continue
            # no separate before LC - skip unless we saved them
    else:
        print("note: save photometry_summary.pre_proximity.csv before phase01 for lc_rms diff")

    # LC rms from comp-quality / summary after only - compare pre/post summary if available
    if lc_deltas:
        print(
            f"|Deltalc_rms| changed targets: max={max(lc_deltas):.6f} "
            f"median={float(np.median(lc_deltas)):.6f} n={len(lc_deltas)}"
        )

    for c in changed[:8]:
        print(json.dumps(c, ensure_ascii=False))
    if len(changed) > 8:
        print(f"... +{len(changed) - 8} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
