# -*- coding: ascii -*-
"""Multiprocessing spawn check: compiled modules pickle by reference on Windows."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CYTHON_RELEASE = REPO_ROOT / "dev" / "tools" / "cython_release"


def _module_is_compiled(name: str) -> bool:
    src_py = str(REPO_ROOT / "src_py")
    if src_py not in sys.path:
        sys.path.insert(0, src_py)
    mod = importlib.import_module(name)
    path = str(getattr(mod, "__file__", "") or "")
    return path.endswith((".pyd", ".so"))


@pytest.mark.skipif(sys.platform != "win32", reason="spawn semantics primary on Windows")
def test_mp_spawn_loads_compiled_photometry_modules() -> None:
    if not (_module_is_compiled("photometry_core") and _module_is_compiled("comp_selection_per_target")):
        pytest.skip("compiled .pyd/.so not present (interpreted dev path)")
    sys.path.insert(0, str(CYTHON_RELEASE))
    from verify_mp import verify

    result = verify()
    assert result["comp_compiled"] is True
    assert result["pc_compiled"] is True
