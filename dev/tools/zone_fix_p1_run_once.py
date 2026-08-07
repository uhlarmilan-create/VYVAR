# -*- coding: ascii -*-
"""Single P1 headless run in a fresh interpreter (for A/B)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SRC = ROOT / "src_py"
_DEV = ROOT / "dev"
_TESTS = ROOT / "dev" / "tests"
for p in (_SRC, _DEV, _TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import AppConfig  # noqa: E402
from photometry_sha import compute_photometry_sha  # noqa: E402
from test_invariants_p1_golden import _mini_root, _p1_headless_chain, _wipe_photometry  # noqa: E402


def main() -> None:
    cfg = AppConfig()
    mini = _mini_root(cfg)
    _wipe_photometry(mini)
    t0 = time.time()
    _p1_headless_chain(mini)
    core, nc = compute_photometry_sha(mini, include_comp_qa=False)
    print(f"core_sha={core}")
    print(f"core_n={nc}")
    print(f"elapsed={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
