#!/usr/bin/env python3
"""Thin wrapper — use ``ct_bgr_summary.py --draft`` for new summaries."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))
import ct_bgr_summary  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(ct_bgr_summary.main())
