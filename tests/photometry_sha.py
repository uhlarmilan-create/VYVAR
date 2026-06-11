"""Numeric photometry SHA helpers (Chi_and_H zaloha anchor; draft-independent).

Recorded values (2026-06-11, draft_000386, confirmed draft_000387): core 203254fd... (2806),
full 95a5515a... (4285). Re-verify via regeneration recipe in VYVAR_STATE.md /
VYVAR_CHIANDH_BASELINE_RUNBOOK.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# Core photometry only (LC + Phase-2A comp_quality + comparison pool).
PHOTOMETRY_SHA_CORE = (
    "203254fd75ea5874f5986eac3f478260c2e7e5a9c2636bfecf2b31244cfb09ba"
)
PHOTOMETRY_SHA_CORE_PREFIX = "203254fd"

# Full reference: core fileset + comp_qa sidecars (draft_000386, 4285 files).
PHOTOMETRY_SHA_BASELINE = (
    "95a5515a6c15a473b6fcd29d3afe0c3b78d88a2da434f8a1c03f28dbe2783c24"
)
PHOTOMETRY_SHA_BASELINE_PREFIX = "95a5515a"

_SHA_PATTERNS_CORE = (
    "**/photometry/**/lightcurve_*.csv",
    "**/photometry/**/comp_quality_*.json",
    "**/platesolve/**/comparison_stars_per_target.csv",
)
_SHA_PATTERN_COMP_QA = "**/photometry/**/lightcurves/comp_qa_*.json"


def photometry_sha_files(
    draft_root: Path,
    *,
    include_comp_qa: bool = False,
) -> list[Path]:
    draft_root = Path(draft_root)
    patterns = list(_SHA_PATTERNS_CORE)
    if include_comp_qa:
        patterns.append(_SHA_PATTERN_COMP_QA)
    files: set[Path] = set()
    for pat in patterns:
        files.update(draft_root.glob(pat))
    return sorted(files)


def compute_photometry_sha(
    draft_root: Path,
    *,
    include_comp_qa: bool = False,
) -> tuple[str, int]:
    files = photometry_sha_files(draft_root, include_comp_qa=include_comp_qa)
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(draft_root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest(), len(files)
