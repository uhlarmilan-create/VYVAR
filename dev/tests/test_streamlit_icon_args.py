# -*- coding: ascii -*-
"""Guard Streamlit status widgets: icon= must be valid or omitted."""

from __future__ import annotations

import ast
import re
from pathlib import Path

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
_WIDGETS = ("st.warning", "st.info", "st.error", "st.success")
_MATERIAL = re.compile(r"^:material/[^:]+:$")


def _icon_arg_ok(value: ast.expr) -> bool:
    if isinstance(value, ast.Constant):
        if value.value is None:
            return True
        if isinstance(value.value, str):
            s = value.value
            if _MATERIAL.match(s):
                return True
            if len(s) == 1 and ord(s) > 127:
                return True
            return False
    return False


def test_streamlit_status_icon_args_are_valid() -> None:
    violations: list[str] = []
    for py in sorted(_SRC_PY.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            qual = f"{getattr(func.value, 'id', '')}.{func.attr}" if isinstance(func.value, ast.Name) else ""
            if qual not in _WIDGETS:
                continue
            for kw in node.keywords:
                if kw.arg != "icon":
                    continue
                if not _icon_arg_ok(kw.value):
                    violations.append(f"{py.relative_to(_SRC_PY.parent)}:{node.lineno} icon={ast.unparse(kw.value)!r}")
    assert not violations, "Invalid Streamlit icon= arguments:\n" + "\n".join(violations)
