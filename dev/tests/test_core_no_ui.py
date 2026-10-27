"""CORE-NO-UI-2026-10: the pipeline never imports Streamlit; GUI state lives in ui_* / app."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src_py"
GUI = re.compile(r"^(ui_.*|app)\.py$")


def test_no_core_module_imports_streamlit() -> None:
    pat = re.compile(r"^\s*(import streamlit|from streamlit)", re.M)
    offenders = [p.name for p in sorted(SRC.glob("*.py")) if not GUI.match(p.name) and pat.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_no_core_module_imports_a_gui_module() -> None:
    pat = re.compile(r"^\s*(from (ui_\w+|app) import|import (ui_\w+|app)\b)", re.M)
    offenders = [p.name for p in sorted(SRC.glob("*.py")) if not GUI.match(p.name) and pat.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_night_run_loads_without_streamlit() -> None:
    code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); import pipeline, night_run; "
            "assert 'streamlit' not in sys.modules, sorted(m for m in sys.modules if m.startswith('ui_'))")
    subprocess.run([sys.executable, "-c", code], check=True)
