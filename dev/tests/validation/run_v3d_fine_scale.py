"""CLI: V3d fine-scale PSF-vs-aperture-vs-truth validation."""
from __future__ import annotations

import argparse
from pathlib import Path

from tests.validation.v3d_fine_scale import V3dFineConfig, run_v3d_fine_scale, write_v3d_report

DATA_DIR = Path(__file__).resolve().parent / "data" / "tier_v3d"


def main() -> None:
    ap = argparse.ArgumentParser(description="V3d fine-scale PSF validation")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--n-real", type=int, default=30)
    args = ap.parse_args()
    cfg = V3dFineConfig(n_real=max(5, int(args.n_real)))
    out = Path(args.out)
    result = run_v3d_fine_scale(cfg, work_dir=out / "_work")
    jp, mp = write_v3d_report(out, result)
    print(f"status: {result.get('status')}")
    print(f"json: {jp}")
    print(f"md:   {mp}")


if __name__ == "__main__":
    main()
