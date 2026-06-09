"""CLI: V3d empirical bias decomposition v2 (T1-T4)."""
from __future__ import annotations

import argparse
from pathlib import Path

from tests.validation.v3d_bias_decomposition_v2 import (
    run_v3d_bias_decomposition_v2,
    write_v3d_bias_decomposition_v2_report,
)
from tests.validation.v3d_fine_scale import V3dFineConfig

DATA_DIR = Path(__file__).resolve().parent / "data" / "tier_v3d"


def main() -> None:
    ap = argparse.ArgumentParser(description="V3d empirical PSF bias decomposition v2")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--n-real", type=int, default=30)
    args = ap.parse_args()
    cfg = V3dFineConfig(n_real=max(5, int(args.n_real)))
    out = Path(args.out)
    result = run_v3d_bias_decomposition_v2(cfg, work_dir=out / "_work_v2")
    mp = write_v3d_bias_decomposition_v2_report(out, result)
    dec = result.get("decision", {})
    print(f"branch: {result['t1'].get('branch')}")
    print(f"cause: {dec.get('identified_cause')}")
    print(f"md: {mp}")


if __name__ == "__main__":
    main()
