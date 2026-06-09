"""Numeric photometry SHA helpers (draft_000366 reference set)."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Core photometry only (LC + Phase-2A comp_quality + comparison pool).
PHOTOMETRY_SHA_CORE = (
    "770966c36fd7e7da925466cbe746b0eb09a7f69fced191fb62a15d8cbbb8574a"
)
PHOTOMETRY_SHA_CORE_PREFIX = "770966c3"

# Post-CQ-C reference: core fileset + comp_qa sidecars (draft_000366, 426 files).
PHOTOMETRY_SHA_BASELINE = (
    "edbd97e7f61c7dc1868eac12322c10cd62a46023c5f2bdac7ff94e97876360a3"
)
PHOTOMETRY_SHA_BASELINE_PREFIX = "edbd97e7"

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
