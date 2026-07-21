#!/usr/bin/env python3
"""Cython build script for VYVAR release bundle (pure-Python mode, zero source edits).

Usage (from repo root):
    python build/setup_cython.py build_ext --inplace
    python build/setup_cython.py clean --all

Compiled .pyd/.so land beside the matching .py under src_py/ and shadow imports.
C intermediates and temp build tree go under build/_cython_out/.

Requires: Cython 3.x, setuptools, platform C compiler (MSVC on Windows, gcc on Linux).
"""
from __future__ import annotations

import sys
from pathlib import Path

from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

# Record at spike time: Cython 3.2.8
CYTHON_SPIKE_VERSION = "3.2.8"

# Strip docstrings from compiled binaries (Options, not a compiler_directive in Cython 3.x).
Options.docstrings = False

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PY = REPO_ROOT / "src_py"
BUILD_DIR = REPO_ROOT / "build" / "_cython_out"

# Spike module list: photometry_core + top PY-LOOP candidates from profiling.
# photometry_core: Cython translate STOP (undeclared _get_lc_psf_strict).
# BUILDABLE_MODULE_LIST: modules that translate without source edits (partial spike).
BUILDABLE_MODULE_LIST: list[str] = [
    "comp_selection_per_target",
    "photometry_phase2a",
]

MODULE_LIST: list[str] = [
    "photometry_core",
    *BUILDABLE_MODULE_LIST,
]


def _active_modules() -> list[str]:
    env = str(__import__("os").environ.get("CYTHON_MODULES", "")).strip()
    if env.lower() == "buildable":
        return list(BUILDABLE_MODULE_LIST)
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    return list(MODULE_LIST)

COMPILER_DIRECTIVES = {
    "language_level": "3",
    "embedsignature": False,
}


def _extensions() -> list[Extension]:
    exts: list[Extension] = []
    for name in _active_modules():
        src = SRC_PY / f"{name}.py"
        if not src.is_file():
            print(f"WARNING: missing source {src}", file=sys.stderr)
            continue
        exts.append(Extension(name, [str(src)]))
    return exts


def main() -> None:
    setup(
        name="vyvar_cython",
        ext_modules=cythonize(
            _extensions(),
            compiler_directives=COMPILER_DIRECTIVES,
            build_dir=str(BUILD_DIR),
        ),
        script_args=sys.argv[1:],
    )


if __name__ == "__main__":
    main()
