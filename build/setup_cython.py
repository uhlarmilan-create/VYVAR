#!/usr/bin/env python3
"""Cython build shim for VYVAR release (delegates to dev/tools/cython_release/).

Usage (from repo root):
    python build/setup_cython.py build
    python build/setup_cython.py clean

Legacy setuptools argv also supported:
    python build/setup_cython.py build_ext --inplace

Compiled .pyd/.so land beside matching .py under src_py/ and shadow imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CYTHON_RELEASE = REPO_ROOT / "dev" / "tools" / "cython_release"
sys.path.insert(0, str(CYTHON_RELEASE))

from build_release import (  # noqa: E402
    COMPILER_DIRECTIVES,
    REQUIRED_COMPILER_DIRECTIVES,
    _assert_pinned_flags,
    run_build,
    run_clean,
)
from module_list import derive_module_lists, module_list  # noqa: E402

# Backward-compatible names for spike tests and docs.
MODULE_LIST = module_list()


def main() -> None:
    _assert_pinned_flags()
    argv = sys.argv[1:]
    if not argv or argv[0] in ("build", "build_ext"):
        run_build()
        return
    if argv[0] == "clean":
        run_clean()
        return
    run_build()


if __name__ == "__main__":
    main()
