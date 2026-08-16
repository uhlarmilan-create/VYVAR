#!/usr/bin/env python3
"""Thin wrapper: ensure S4 stars JSON exists for S5."""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "dev" / "tools" / "wide_err_03_s4_remeasure.py"), run_name="__main__")
