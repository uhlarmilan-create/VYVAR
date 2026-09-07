"""Shared pytest configuration."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: draft_000366 comp_qa integration (minutes per case)",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """Vendor tree is not a pytest package; smoke test lives in dev/tests."""
    _ = config
    parts = set(Path(collection_path).parts)
    return "xval_pythonphot" in parts and "vendor" in parts
