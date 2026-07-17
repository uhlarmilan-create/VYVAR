"""CLI: generate A9 truth + contamination envelope report."""
from __future__ import annotations

import argparse
from pathlib import Path

from tests.validation.a9_core import write_envelope_report
from tests.validation.gen_a9 import write_a9_truth

DATA_DIR = Path(__file__).resolve().parent / "data" / "tier_a9"


def main() -> None:
    ap = argparse.ArgumentParser(description="A9 NEIGHBOR-SUB acceptance envelope")
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    ap.add_argument("--truth-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    tp = write_a9_truth(out)
    print(f"truth: {tp}")
    if args.truth_only:
        return
    jp, mp, ok = write_envelope_report(out)
    print(f"envelope json: {jp}")
    print(f"envelope md:   {mp}")
    print(f"self_check: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
