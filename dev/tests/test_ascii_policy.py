# -*- coding: ascii -*-
"""ENCODING-POLICY guard: tracked text files must be ASCII-only (byte < 0x80).

Allowlist is empty by default. Justified exceptions only, with reason strings.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS = REPO_ROOT / "dev" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from ascii_migrate import SPECIAL_NAMES, _tracked_text_files  # noqa: E402

# path -> reason. Keep EMPTY unless Milan approves a durable exception.
ASCII_POLICY_ALLOWLIST: dict[str, str] = {}


def test_ascii_policy_allowlist_is_empty_or_justified() -> None:
    for path, reason in ASCII_POLICY_ALLOWLIST.items():
        assert reason and reason.strip(), f"allowlist {path!r} needs a reason"
        assert (REPO_ROOT / path).is_file(), f"allowlist path missing: {path}"


def test_tracked_text_files_are_ascii() -> None:
    offenders: list[str] = []
    for rel in _tracked_text_files(REPO_ROOT):
        key = rel.as_posix()
        if key in ASCII_POLICY_ALLOWLIST:
            continue
        data = (REPO_ROOT / rel).read_bytes()
        if any(b >= 0x80 for b in data):
            # locate first offender byte for a short diagnostic
            idx = next(i for i, b in enumerate(data) if b >= 0x80)
            offenders.append(f"{key}: first_non_ascii_at={idx} byte=0x{data[idx]:02x}")
    assert offenders == [], (
        "ENCODING-POLICY: non-ASCII bytes in tracked text files "
        f"({len(offenders)}). Run: python dev/tools/ascii_migrate.py\n"
        + "\n".join(offenders[:40])
    )


def test_walk_covers_license_and_gitattributes() -> None:
    names = {p.name for p in _tracked_text_files(REPO_ROOT)}
    for required in SPECIAL_NAMES:
        assert required in names or any(
            p.name == required for p in _tracked_text_files(REPO_ROOT)
        ), f"walk must include {required}"
