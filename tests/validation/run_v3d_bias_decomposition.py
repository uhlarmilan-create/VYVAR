"""CLI: V3d PSF bias decomposition (pre-AC vs post-AC, background sensitivity)."""
from __future__ import annotations

import argparse
from pathlib import Path

from tests.validation.v3d_fine_scale import (
    V3dFineConfig,
    run_v3d_bias_decomposition,
    write_bias_decomposition_report,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "tier_v3d"


def main() -> None:
    ap = argparse.ArgumentParser(description="V3d PSF bias decomposition diagnostic")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--n-real", type=int, default=30)
    args = ap.parse_args()
    cfg = V3dFineConfig(n_real=max(5, int(args.n_real)))
    out = Path(args.out)
    result = run_v3d_bias_decomposition(cfg, work_dir=out / "_work_bias")
    jp, mp = write_bias_decomposition_report(out, result)
    diag = result.get("diagnosis", {})
    print(f"localized_cause: {diag.get('localized_cause')}")
    print(f"psf_ac_factor: {result.get('psf_aperture_correction_factor')}")
    print(f"json: {jp}")
    print(f"md:   {mp}")


if __name__ == "__main__":
    main()
