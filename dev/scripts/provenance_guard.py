#!/usr/bin/env python3
"""Refuse unstamped pipeline_meta for harness/baseline runs (PROVENANCE-GUARD)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_GIT_HASH_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)


class ProvenanceGuardError(SystemExit):
    """Raised when unstamped draft photometry is used without override."""


def read_pipeline_meta(photometry_dir: Path) -> dict[str, Any]:
    path = Path(photometry_dir) / "pipeline_meta.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def provenance_block(meta: dict[str, Any]) -> dict[str, Any] | None:
    prov = meta.get("provenance")
    return prov if isinstance(prov, dict) else None


def parseable_git_hash(prov: dict[str, Any] | None) -> str | None:
    if not prov:
        return None
    gh = prov.get("git_hash")
    if gh is None:
        return None
    s = str(gh).strip()
    if not s or s.lower() in ("null", "none", "unknown"):
        return None
    return s if _GIT_HASH_RE.match(s) else None


def is_stamped(photometry_dir: Path) -> bool:
    return parseable_git_hash(provenance_block(read_pipeline_meta(photometry_dir))) is not None


def assert_stamped(
    photometry_dir: Path,
    *,
    draft_id: int | None = None,
    setup: str = "",
    allow_unstamped: bool = False,
) -> dict[str, Any]:
    """Return provenance dict; refuse or warn per allow_unstamped."""
    phot = Path(photometry_dir)
    meta = read_pipeline_meta(phot)
    prov = provenance_block(meta)
    gh = parseable_git_hash(prov)
    draft_s = f"draft_{int(draft_id):06d}" if draft_id is not None else "?"
    setup_s = str(setup or phot.parent.name)
    if gh is not None:
        return {
            "stamped": True,
            "provenance_unstamped": False,
            "git_hash": gh,
            "stamped_at_utc": prov.get("stamped_at_utc") if prov else None,
            "entry_point": prov.get("entry_point") if prov else meta.get("entry_point"),
        }
    msg = (
        f"PROVENANCE-GUARD REFUSE: {draft_s} setup {setup_s} lacks pipeline_meta provenance "
        f"(git_hash). Path: {phot / 'pipeline_meta.json'}. "
        f"Pass --allow-unstamped to override (records provenance_unstamped=true)."
    )
    if allow_unstamped:
        return {
            "stamped": False,
            "provenance_unstamped": True,
            "warning": msg,
            "git_hash": None,
            "stamped_at_utc": None,
            "entry_point": None,
        }
    raise ProvenanceGuardError(msg)


def stamp_output_meta(payload: dict[str, Any], guard_result: dict[str, Any]) -> dict[str, Any]:
    """Merge guard flags into harness output JSON."""
    out = dict(payload)
    out["provenance_guard"] = {
        "stamped": bool(guard_result.get("stamped")),
        "provenance_unstamped": bool(guard_result.get("provenance_unstamped")),
        "git_hash": guard_result.get("git_hash"),
        "warning": guard_result.get("warning"),
    }
    if guard_result.get("provenance_unstamped"):
        out["provenance_unstamped"] = True
    return out


def add_allow_unstamped_arg(parser: Any) -> None:
    parser.add_argument(
        "--allow-unstamped",
        action="store_true",
        help="Allow pipeline_meta without provenance block (WARNING; marks outputs unstamped).",
    )
