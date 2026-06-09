"""CLI: V3e ePSF FWHM QC estimator validation (EPSF-1)."""
from __future__ import annotations

import argparse
from pathlib import Path

from tests.validation.v3e_epsf_fwhm import run_v3e_epsf_fwhm, write_v3e_report

DATA_DIR = Path(__file__).resolve().parent / "data" / "tier_v3e"


def main() -> None:
    ap = argparse.ArgumentParser(description="V3e ePSF FWHM QC estimator validation")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    kwargs = {}
    if args.seed is not None:
        kwargs["rng_seed"] = int(args.seed)
    result = run_v3e_epsf_fwhm(**kwargs)
    jp, mp = write_v3e_report(Path(args.out), result)
    print(f"status: {result.get('status')}")
    for row in result.get("cases", []):
        print(
            f"  {row['name']}: OLD ratio={row['ratio_old']:.4f} "
            f"NEW ratio={row['ratio_new']:.4f}"
        )
    print(f"json: {jp}")
    print(f"md:   {mp}")


if __name__ == "__main__":
    main()
