#!/usr/bin/env python3
"""CLI for per-target trust flags (uses draft photometry_summary.csv)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from trust_flag_core import run_trust_flag_for_photometry_dir

ROOT = Path(__file__).resolve().parent.parent  # src_py -> repo root (tmp/ output defaults)


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-target trust flag from photometry_summary")
    ap.add_argument(
        "--photometry-dir",
        type=Path,
        default=ROOT
        / "Archive"
        / "Drafts"
        / "draft_000365"
        / "platesolve"
        / "NoFilter_60_2"
        / "photometry",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "tmp" / "xval_out" / "trust_per_target.csv")
    args = ap.parse_args()

    result = run_trust_flag_for_photometry_dir(
        photometry_dir=args.photometry_dir,
        update_summary=False,
    )
    rows = []
    for tid, info in sorted(result.get("per_target", {}).items()):
        rows.append(
            {
                "catalog_id": tid,
                "vsx_name": info.get("vsx_name", ""),
                "trust": info.get("trust"),
                "n_clean": info.get("n_clean"),
                "lc_quality": info.get("lc_quality"),
                "check_scatter": info.get("check_scatter") or "",
                "reason": info.get("trust_reason"),
            }
        )
    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"\nWrote {args.out} ({len(out_df)} targets)")
    for level in ("GREEN", "YELLOW", "RED"):
        sub = out_df[out_df["trust"] == level]
        print(f"  {level}: {len(sub)}")
    red = out_df[out_df["trust"] == "RED"]
    if not red.empty:
        print("\nRED targets:")
        for _, r in red.iterrows():
            print(f"  {r['vsx_name']} ({str(r['catalog_id'])[-6:]}): {r['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
