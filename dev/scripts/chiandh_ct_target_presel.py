#!/usr/bin/env python3
"""CT target presel for h & chi Persei B/V/R (photometric Johnson-Cousins)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import m67_ct_target_presel as _base  # noqa: E402

CT_FILTERS = ("B", "V", "R")


def presel_draft(draft_id: int, **kwargs):
    return _base.presel_draft(draft_id, filters=CT_FILTERS, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=int, required=True)
    ap.add_argument("--n-in-range", type=int, default=30)
    ap.add_argument("--n-red-giants", type=int, default=5)
    args = ap.parse_args()
    reps = presel_draft(
        args.draft,
        n_in_range=int(args.n_in_range),
        n_red_giants=int(args.n_red_giants),
    )
    for r in reps:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
