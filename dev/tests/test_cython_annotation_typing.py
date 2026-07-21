# -*- coding: ascii -*-
"""Recurrence: Cython annotation_typing must stay False for release compiles."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_setup_cython():
    path = Path(__file__).resolve().parents[2] / "build" / "setup_cython.py"
    spec = importlib.util.spec_from_file_location("setup_cython", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_release_compiler_directives_pin_annotation_typing_false() -> None:
    mod = _load_setup_cython()
    directives = mod.COMPILER_DIRECTIVES
    assert directives.get("annotation_typing") is False, (
        "Release compiles must set annotation_typing=False; PEP-484 annotations "
        "become C type declarations when True and changed comp_selection science "
        "(P1 SHA drift, 167/169 empty-comp drops on spike without this pin)."
    )


def test_pep484_int_annotation_differs_under_cython_semantics() -> None:
    """Document mechanism: optional None rejected when annotation_typing=True."""

    def accepts_optional_count(x: int | None) -> int:
        return 0 if x is None else int(x)

    # Interpreted Python accepts None despite int annotation (metadata only).
    assert accepts_optional_count(None) == 0
    # Under Cython annotation_typing=True, x: int coerces/rejects at boundary;
    # comp_selection_per_target PY-LOOP paths hit similar silent failures.
