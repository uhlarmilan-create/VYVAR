"""Summarize draft-level ct_prototype.csv from VYVAR_CT_PROTOTYPE=1 Phase 2A runs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _draft_dir(draft: int) -> Path:
    return _ROOT / "Archive" / "Drafts" / f"draft_{draft:06d}"


def summarize(ct_path: Path) -> int:
    if not ct_path.is_file():
        print(f"Missing: {ct_path}")
        return 1
    df = pd.read_csv(ct_path)
    n = len(df)
    c1 = pd.to_numeric(df["c1"], errors="coerce")
    ct = pd.to_numeric(df["ct_corr"], errors="coerce").abs()
    sc = pd.to_numeric(df["cat_inst_scatter"], errors="coerce")
    scr = pd.to_numeric(df["cat_inst_scatter_resid"], errors="coerce")
    gate = df["gate_would_pass"].astype(str).str.lower().isin(("true", "1", "yes"))

    print(f"rows={n}  path={ct_path}")
    print("\n1) c1 distribution")
    print(f"   median={c1.median():.4f}  IQR=[{c1.quantile(0.25):.4f}, {c1.quantile(0.75):.4f}]")
    print(f"   min={c1.min():.4f}  max={c1.max():.4f}")

    print("\n2) |ct_corr|")
    print(f"   median={ct.median():.4f}  p90={ct.quantile(0.9):.4f}")
    for thr in (0.02, 0.05, 0.10):
        print(f"   |ct_corr|>{thr:.2f}: {(ct > thr).sum()}")

    print("\n3) scatter reduction (median)")
    print(f"   cat_inst_scatter={sc.median():.4f}  cat_inst_scatter_resid={scr.median():.4f}")
    print(f"   delta={sc.median() - scr.median():.4f} mag")

    print("\n4) gate_would_pass")
    print(f"   pass={int(gate.sum())}/{n} ({100.0 * gate.mean():.1f}%)")

    print("\n5) top 5 |ct_corr|")
    top = df.assign(_abs=ct).sort_values("_abs", ascending=False).head(5)
    cols = ["vsx_name", "target_bp_rp", "comp_med_bp_rp", "ct_corr", "_abs"]
    print(top[cols].to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=int, required=True, help="Draft number, e.g. 366")
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Override path to ct_prototype.csv",
    )
    args = ap.parse_args()
    ct_path = args.csv or (_draft_dir(args.draft) / "ct_prototype.csv")
    return summarize(ct_path)


if __name__ == "__main__":
    raise SystemExit(main())
