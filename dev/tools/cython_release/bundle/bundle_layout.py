# -*- coding: ascii -*-
"""Release bundle layout helpers and assertions."""
from __future__ import annotations

import os
import sys
from pathlib import Path

BUNDLE_DIR = Path(__file__).resolve().parent
CYTHON_RELEASE = BUNDLE_DIR.parent
REPO_ROOT = CYTHON_RELEASE.parents[2]
SRC_PY = REPO_ROOT / "src_py"
_DEFAULT_DIST = REPO_ROOT / "tmp" / "cython_release" / "bundle" / "dist"
DIST_DIR = Path(os.environ.get("VYVAR_BUNDLE_DIST", str(_DEFAULT_DIST)))

sys.path.insert(0, str(CYTHON_RELEASE))
from module_list import derive_module_lists, is_ui_layer  # noqa: E402


def ui_py_names() -> list[str]:
    names: list[str] = []
    for p in sorted(SRC_PY.glob("*.py")):
        if is_ui_layer(p.stem) or p.stem == "app":
            names.append(p.name)
    return names


def compiled_module_stems() -> list[str]:
    mods, _ = derive_module_lists()
    return mods


def assert_no_compiled_py_sources(bundle_src_py: Path, compiled: list[str]) -> None:
    violations: list[str] = []
    for name in compiled:
        py = bundle_src_py / f"{name}.py"
        if py.is_file():
            violations.append(str(py.relative_to(bundle_src_py.parent)))
    if violations:
        raise SystemExit(
            "Bundle assertion failed: .py present for compiled modules:\n  "
            + "\n  ".join(violations)
        )


def bundle_name(tag: str, platform_key: str) -> str:
    return f"VYVAR-{tag}-{platform_key}"
