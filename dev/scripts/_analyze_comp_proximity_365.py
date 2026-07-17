#!/usr/bin/env python3
"""Analyze comp-set changes after proximity tie-break (draft 000365)."""
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

PHOT = _ROOT / "Archive/Drafts/draft_000365/platesolve/NoFilter_60_2/photometry"
BEFORE = PHOT / "comparison_stars_per_target.pre_proximity.csv"
AFTER = PHOT / "comparison_stars_per_target.csv"
SUM_B = PHOT / "photometry_summary.pre_proximity.csv"
SUM_A = PHOT / "photometry_summary.csv"


def _norm(x: object) -> str:
    return str(normalize_gaia_source_id(x) or "").strip()


def main() -> int:
    bdf = pd.read_csv(BEFORE, dtype={"catalog_id": str, "target_catalog_id": str})
    adf = pd.read_csv(AFTER, dtype={"catalog_id": str, "target_catalog_id": str})
    bdf["_t"] = bdf["target_catalog_id"].map(_norm)
    bdf["_c"] = bdf["catalog_id"].map(_norm)
    adf["_t"] = adf["target_catalog_id"].map(_norm)
    adf["_c"] = adf["catalog_id"].map(_norm)

    changed_targets = 0
    swap_counts: list[int] = []
    tier_rms_ok = 0
    tier_rms_fail = 0
    bin_w = 0.001

    for tid in sorted(bdf["_t"].unique()):
        if not tid:
            continue
        b_ids = set(bdf.loc[bdf["_t"] == tid, "_c"]) - {""}
        a_ids = set(adf.loc[adf["_t"] == tid, "_c"]) - {""}
        if b_ids == a_ids:
            continue
        changed_targets += 1
        swap_counts.append(len(b_ids ^ a_ids))
        removed = b_ids - a_ids
        added = a_ids - b_ids
        brow = bdf[(bdf["_t"] == tid) & (bdf["_c"].isin(removed))]
        arow = adf[(adf["_t"] == tid) & (adf["_c"].isin(added))]
        for _, br in brow.iterrows():
            brms = float(pd.to_numeric(br.get("comp_rms"), errors="coerce"))
            btier = int(pd.to_numeric(br.get("comp_tier"), errors="coerce") or 4)
            bbin = round(brms / bin_w) * bin_w if np.isfinite(brms) else float("nan")
            matched = False
            for _, ar in arow.iterrows():
                arms = float(pd.to_numeric(ar.get("comp_rms"), errors="coerce"))
                atier = int(pd.to_numeric(ar.get("comp_tier"), errors="coerce") or 4)
                abin = round(arms / bin_w) * bin_w if np.isfinite(arms) else float("nan")
                if btier == atier and bbin == abin:
                    matched = True
                    break
            if matched:
                tier_rms_ok += 1
            else:
                tier_rms_fail += 1

    print(f"changed_targets={changed_targets} / {bdf['_t'].nunique()}")
    if swap_counts:
        print(
            f"comps_swapped_per_target min={min(swap_counts)} "
            f"median={int(np.median(swap_counts))} max={max(swap_counts)}"
        )
    print(f"removed_comp_swap_pairs_same_tier_rms_bin={tier_rms_ok} other={tier_rms_fail}")

    if SUM_B.is_file() and SUM_A.is_file():
        sb = pd.read_csv(SUM_B, dtype={"catalog_id": str})
        sa = pd.read_csv(SUM_A, dtype={"catalog_id": str})
        sb["_n"] = sb["catalog_id"].map(_norm)
        sa["_n"] = sa["catalog_id"].map(_norm)
        m = sb.merge(sa, on="_n", suffixes=("_b", "_a"))
        if "lc_rms_b" in m.columns:
            d = (pd.to_numeric(m["lc_rms_a"], errors="coerce") - pd.to_numeric(m["lc_rms_b"], errors="coerce")).abs()
            d = d[d.notna()]
            print(
                f"lc_rms_abs_delta_all_targets max={float(d.max()):.6f} "
                f"median={float(d.median()):.6f} mean={float(d.mean()):.6f}"
            )
            changed_t = set()
            for tid in bdf["_t"].unique():
                b_ids = set(bdf.loc[bdf["_t"] == tid, "_c"])
                a_ids = set(adf.loc[adf["_t"] == tid, "_c"])
                if b_ids != a_ids:
                    changed_t.add(tid)
            sub = m[m["_n"].isin(changed_t)]
            if len(sub):
                d2 = (
                    pd.to_numeric(sub["lc_rms_a"], errors="coerce")
                    - pd.to_numeric(sub["lc_rms_b"], errors="coerce")
                ).abs()
                d2 = d2[d2.notna()]
                print(
                    f"lc_rms_abs_delta_changed_comp_targets max={float(d2.max()):.6f} "
                    f"median={float(d2.median()):.6f} n={len(d2)}"
                )

    # order-only check on unchanged membership targets
    order_only = 0
    for tid in bdf["_t"].unique():
        bsub = bdf[bdf["_t"] == tid].sort_values("_c")["_c"].tolist()
        asub = adf[adf["_t"] == tid].sort_values("_c")["_c"].tolist()
        if bsub == asub and not bdf[bdf["_t"] == tid].equals(adf[adf["_t"] == tid]):
            order_only += 1
    print(f"same_membership_different_row_order_targets={order_only}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
