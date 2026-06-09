"""CLI: draft 367 fine-scale ePSF mismatch + A9 NEIGHBOR-SUB yield diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

from tests.validation.a9_core import write_draft367_report

DATA_DIR = Path(__file__).resolve().parent / "data" / "tier_a9"


def main() -> None:
    ap = argparse.ArgumentParser(description="A9 draft 367 fine-scale diagnostic")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    jp, mp = write_draft367_report(Path(args.out))
    print(f"diagnostic json: {jp}")
    print(f"diagnostic md:   {mp}")


if __name__ == "__main__":
    main()
