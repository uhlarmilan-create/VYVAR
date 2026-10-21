"""DRAFT-514-TRIAGE: preflight log must retain real traceback."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from run_preflight_log import write_run_preflight_error_log  # noqa: E402


def test_preflight_log_uses_exception_traceback(tmp_path: Path) -> None:
    try:
        math.gamma(500.0)
    except OverflowError as exc:
        path = write_run_preflight_error_log(tmp_path, step="test", exc=exc)
        text = path.read_text(encoding="utf-8")
    assert "OverflowError" in text
    assert "math range error" in text
    assert "traceback:" in text
    assert "NoneType: None" not in text
    assert "gamma" in text or "sigma_floor" in text or 'File "' in text
