"""CLI: A9 PSF-mismatch diagnostic (current vs realistic) for step 2b gate."""
from __future__ import annotations

import argparse
from pathlib import Path

from tests.validation.a9_core import write_mismatch_diagnostic_report

DATA_DIR = Path(__file__).resolve().parent / "data" / "tier_a9"


def main() -> None:
    ap = argparse.ArgumentParser(description="A9 mismatch diagnostic (analysis-only)")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    jp, mp = write_mismatch_diagnostic_report(Path(args.out))
    print(f"diagnostic json: {jp}")
    print(f"diagnostic md:   {mp}")


if __name__ == "__main__":
    main()
