# -*- coding: ascii -*-
"""Where a processed field (a VYVAR run of one night) lives, from any path inside it.

PROCESSED-FIELD-NAME-2026-10 (Milan 2026-10-24): a run is stored in Archive/Drafts/<TARGET>_<YYYYMMDD>_<NNN>
(legacy: draft_NNNNNN). The folder name carries no id; the run's id is the ``draft_id`` of its
``draft_manifest.json``. These helpers replace parsing ``draft_<digits>`` out of folder names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

MANIFEST_NAME = "draft_manifest.json"


_LEGACY_FIELD_RE = re.compile(r"draft_\d+(_snapshot.*)?", re.IGNORECASE)


def processed_field_root(path: Path | str | None) -> Path | None:
    """The run folder containing ``path``: ``path`` itself or the nearest parent with ``draft_manifest.json``;
    without a manifest, the nearest legacy ``draft_NNNNNN`` (or ``draft_NNNNNN_snapshot...``) folder.

    G12-19: the ONE "field root from a path" rule, for calibrated and pre-calibrated fields alike (a
    pre-calibrated field has ``non_calibrated/`` and no ``calibrated/``; the rule never looks at either)."""
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    chain: list[Path] = []
    for cand in (p, *p.parents):
        if (cand / MANIFEST_NAME).is_file():
            return cand
        if cand.name == "Drafts":
            break
        chain.append(cand)
    for cand in chain:
        if _LEGACY_FIELD_RE.fullmatch(cand.name):
            return cand
    return None


def processed_field_id(path: Path | str | None) -> int | None:
    """The run id of the folder containing ``path``: its manifest's ``draft_id``; a legacy
    ``draft_NNNNNN`` folder without a readable manifest by its name."""
    if path is None:
        return None
    root = processed_field_root(path)
    if root is not None:
        try:
            return int(json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8")).get("draft_id"))
        except Exception:  # noqa: BLE001
            pass
    p = Path(path).expanduser().resolve()
    for cand in (p, *p.parents):
        m = re.fullmatch(r"draft_(\d+)", cand.name, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None
