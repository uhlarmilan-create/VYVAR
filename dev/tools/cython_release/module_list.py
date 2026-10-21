# -*- coding: ascii -*-
"""Single source of truth for CYTHON-RELEASE-1 MODULE_LIST derivation (S2)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_PY = REPO_ROOT / "src_py"

UI_APP_REASON = (
    "UI layer (S1): Streamlit entry; stays interpreted for reload/tracebacks/no compute benefit"
)
UI_PREFIX_REASON = (
    "UI layer (S1): ui_* stays interpreted for Streamlit reload/tracebacks"
)

# Explicit exclusions beyond ui_* / app.py. One-line reason per entry required.
EXPLICIT_EXCLUDE: dict[str, str] = {}


def all_src_py_stems() -> list[str]:
    return sorted(p.stem for p in SRC_PY.glob("*.py"))


def is_ui_layer(name: str) -> bool:
    return name == "app" or name.startswith("ui_")


def derive_module_lists() -> tuple[list[str], list[tuple[str, str]]]:
    """Return (MODULE_LIST sorted, exclusions as (name, reason) sorted)."""
    included: list[str] = []
    excluded: list[tuple[str, str]] = []
    for name in all_src_py_stems():
        if name == "app":
            excluded.append((name, UI_APP_REASON))
            continue
        if name.startswith("ui_"):
            excluded.append((name, UI_PREFIX_REASON))
            continue
        if name in EXPLICIT_EXCLUDE:
            excluded.append((name, EXPLICIT_EXCLUDE[name]))
            continue
        src = SRC_PY / f"{name}.py"
        if not src.is_file():
            excluded.append((name, "missing source file"))
            continue
        included.append(name)
    return sorted(included), sorted(excluded, key=lambda x: x[0])


def module_list() -> list[str]:
    mods, _ = derive_module_lists()
    return mods
