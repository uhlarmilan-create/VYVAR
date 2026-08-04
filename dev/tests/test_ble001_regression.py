"""Regression guard: unmarked broad except (BLE001 / E722) must not grow."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def test_ruff_ble001_e722_clean() -> None:
    """CI / pre-commit parity: ruff --select BLE001,E722 must pass on production tree."""
    if importlib.util.find_spec("ruff") is None:
        pytest.skip("ruff not installed")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            ".",
            "--select",
            "BLE001",
            "E722",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    combined = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0 and "No module named ruff" in combined:
        pytest.skip("ruff not installed")
    assert proc.returncode == 0, (
        "Unmarked broad-except violations (add noqa: BLE001 or narrow the handler):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
