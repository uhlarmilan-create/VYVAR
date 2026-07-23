# -*- coding: ascii -*-
"""Recurrence: Cython release compile config pins (full MODULE_LIST, S3)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CYTHON_RELEASE = REPO_ROOT / "dev" / "tools" / "cython_release"


def _load_setup_cython():
    path = REPO_ROOT / "build" / "setup_cython.py"
    spec = importlib.util.spec_from_file_location("setup_cython", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_module_list():
    sys.path.insert(0, str(CYTHON_RELEASE))
    from module_list import derive_module_lists, module_list

    return derive_module_lists, module_list


def test_release_compiler_directives_pin_annotation_typing_false() -> None:
    mod = _load_setup_cython()
    directives = mod.COMPILER_DIRECTIVES
    assert directives.get("annotation_typing") is False, (
        "Release compiles must set annotation_typing=False; PEP-484 annotations "
        "become C type declarations when True and changed comp_selection science "
        "(P1 SHA drift, 167/169 empty-comp drops on spike without this pin)."
    )
    assert directives.get("language_level") == "3"
    assert directives.get("embedsignature") is False


def test_release_module_list_covers_all_non_ui_src_py() -> None:
    derive_module_lists, module_list_fn = _load_module_list()
    included, excluded = derive_module_lists()
    assert included == module_list_fn()
    assert len(included) >= 80
    ui = [n for n, _ in excluded if n == "app" or n.startswith("ui_")]
    assert len(ui) >= 14
    for name in included:
        assert not name.startswith("ui_")
        assert name != "app"
        assert (REPO_ROOT / "src_py" / f"{name}.py").is_file()


def test_build_release_refuses_flag_drift(monkeypatch) -> None:
    sys.path.insert(0, str(CYTHON_RELEASE))
    from Cython.Compiler import Options

    import build_release as br

    monkeypatch.setattr(br, "COMPILER_DIRECTIVES", {"annotation_typing": True})
    try:
        br._assert_pinned_flags()
        raise AssertionError("expected SystemExit on annotation_typing drift")
    except SystemExit:
        pass
    monkeypatch.setattr(br, "COMPILER_DIRECTIVES", dict(br.REQUIRED_COMPILER_DIRECTIVES))
    monkeypatch.setattr(Options, "docstrings", True)
    try:
        br._assert_pinned_flags()
        raise AssertionError("expected SystemExit on Options.docstrings drift")
    except SystemExit:
        pass


def test_pep484_int_annotation_differs_under_cython_semantics() -> None:
    """Document mechanism: optional None rejected when annotation_typing=True."""

    def accepts_optional_count(x: int | None) -> int:
        return 0 if x is None else int(x)

    assert accepts_optional_count(None) == 0
