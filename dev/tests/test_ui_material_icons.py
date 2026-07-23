# -*- coding: ascii -*-
"""Repo-wide guard: every :material/...: literal must pass pinned Streamlit validation."""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PY = REPO_ROOT / "src_py"
PINNED_STREAMLIT_TAG = "1.60.0"
_MATERIAL_ICON_LITERAL = re.compile(r"""(?P<q>['"])(?P<icon>:material/[a-z0-9_]+:)(?P=q)""")
_PINNED_ICONS: set[str] | None = None


def _load_pinned_material_icons() -> set[str]:
    global _PINNED_ICONS
    if _PINNED_ICONS is not None:
        return _PINNED_ICONS
    url = (
        "https://raw.githubusercontent.com/streamlit/streamlit/"
        f"{PINNED_STREAMLIT_TAG}/lib/streamlit/material_icon_names.py"
    )
    text = urllib.request.urlopen(url, timeout=30).read().decode("ascii")
    match = re.search(r"ALL_MATERIAL_ICONS\s*=\s*\{([^}]+)\}", text, re.S)
    assert match is not None, "failed to parse pinned Streamlit material icon set"
    _PINNED_ICONS = set(re.findall(r'"([^"]+)"', match.group(1)))
    return _PINNED_ICONS


def _pinned_validate_material_icon(maybe_material_icon: str) -> str:
    """Mirror ``streamlit.string_util.validate_material_icon`` for the bundle pin."""
    try:
        from streamlit.string_util import validate_material_icon

        return validate_material_icon(maybe_material_icon)
    except ImportError:
        pass

    icon_regex = r"^\s*:(.+)\/(.+):\s*$"
    icon_match = re.match(icon_regex, maybe_material_icon)
    if not icon_match:
        raise ValueError(f"invalid material icon shortcode: {maybe_material_icon!r}")
    pack_name, icon_name = icon_match.groups()
    icons = _load_pinned_material_icons()
    if pack_name != "material" or not icon_name or icon_name not in icons:
        raise ValueError(f"unknown material icon: {maybe_material_icon!r}")
    return f":{pack_name}/{icon_name}:"


def _iter_material_icon_literals() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for path in sorted(SRC_PY.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in _MATERIAL_ICON_LITERAL.finditer(text):
            found.append((str(path.relative_to(REPO_ROOT)), match.group("icon")))
    return found


def test_repo_material_icon_literals_validate_against_pinned_streamlit() -> None:
    literals = _iter_material_icon_literals()
    assert literals, "expected at least one :material/...: literal under src_py/"
    failures: list[str] = []
    for rel_path, icon in literals:
        try:
            _pinned_validate_material_icon(icon)
        except ValueError as exc:
            failures.append(f"{rel_path}: {icon} ({exc})")
    assert not failures, "invalid material icons:\n" + "\n".join(failures)


def test_telescope_icon_is_not_used() -> None:
    icons = _load_pinned_material_icons()
    assert "telescope" not in icons
