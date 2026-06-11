"""Byte-identity SHA guards when ephemeral draft_000386 is present (optional local check).

Anchor gate is draft-independent: compare compute_photometry_sha output to PHOTOMETRY_SHA_* constants.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.photometry_sha import (
    PHOTOMETRY_SHA_BASELINE,
    PHOTOMETRY_SHA_CORE,
    compute_photometry_sha,
)

_DRAFT = Path(__file__).resolve().parents[1] / "Archive" / "Drafts" / "draft_000386"

pytestmark = pytest.mark.skipif(
    not (_DRAFT / "platesolve").is_dir(),
    reason="draft_000386 not present",
)


def test_draft_386_core_photometry_sha():
    """LC + comp_quality + comparison pool (excludes comp_qa, lc_quality, trust)."""
    sha, n = compute_photometry_sha(_DRAFT)
    assert n == 2806
    assert sha == PHOTOMETRY_SHA_CORE


def test_draft_386_extended_photometry_sha():
    """Full anchor includes comp_qa sidecars."""
    sha, n = compute_photometry_sha(_DRAFT, include_comp_qa=True)
    assert n == 4285
    assert sha == PHOTOMETRY_SHA_BASELINE
