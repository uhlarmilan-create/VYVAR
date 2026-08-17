# -*- coding: ascii -*-
"""Repo-wide guard: every :material/...: literal must pass pinned Streamlit validation.

NET-TEST-01: icon names are vendored from the locally installed streamlit
package. No network fetch in this module.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PY = REPO_ROOT / "src_py"
VENDOR_ICON_LIST = (
    REPO_ROOT / "dev" / "tests" / "data" / "streamlit_material_icon_names.txt"
)
_MATERIAL_ICON_LITERAL = re.compile(r"""(?P<q>['"])(?P<icon>:material/[a-z0-9_]+:)(?P=q)""")
_PINNED_ICONS: set[str] | None = None
_SANITY_KNOWN_NAMES = ("search", "settings", "science", "help")


def _load_pinned_material_icons() -> set[str]:
    global _PINNED_ICONS
    if _PINNED_ICONS is not None:
        return _PINNED_ICONS
    assert VENDOR_ICON_LIST.is_file(), f"missing vendored icon list {VENDOR_ICON_LIST}"
    names: set[str] = set()
    for raw in VENDOR_ICON_LIST.read_text(encoding="ascii").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line)
    _PINNED_ICONS = names
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


def test_vendored_material_icon_list_sanity() -> None:
    icons = _load_pinned_material_icons()
    assert len(icons) >= 1000, f"vendored icon list too small: n={len(icons)}"
    missing = [n for n in _SANITY_KNOWN_NAMES if n not in icons]
    assert not missing, f"vendored icon list missing known names: {missing}"


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
    src = "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(SRC_PY.rglob("*.py"))
    )
    assert ":material/telescope:" not in src
