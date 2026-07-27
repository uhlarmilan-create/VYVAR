"""Shared pytest configuration."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))


class _PreferSourcePyFinder:
    """Load ``src_py/*.py`` when a stale compiled ``.pyd`` shadows it (dev/test only)."""

    def find_spec(self, fullname: str, path: object | None, target: object | None = None):
        if path is not None:
            return None
        py = _SRC_PY / f"{fullname}.py"
        if not py.is_file():
            return None
        return importlib.util.spec_from_file_location(fullname, py, submodule_search_locations=[str(_SRC_PY)])


if not any(isinstance(f, _PreferSourcePyFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _PreferSourcePyFinder())

# Drop stale compiled modules if a root-level .pyd shadowed src_py during an earlier import.
for _mod in ("pipeline", "database", "night_run", "photometry_core"):
    if _mod in sys.modules and str(getattr(sys.modules[_mod], "__file__", "") or "").endswith(".pyd"):
        del sys.modules[_mod]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: draft_000366 comp_qa integration (minutes per case)",
    )
