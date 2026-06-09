"""Byte-identity SHA guards for draft_000366 reference tree."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.photometry_sha import (
    PHOTOMETRY_SHA_BASELINE,
    PHOTOMETRY_SHA_CORE,
    compute_photometry_sha,
)

_DRAFT = Path(__file__).resolve().parents[1] / "Archive" / "Drafts" / "draft_000366"

pytestmark = pytest.mark.skipif(
    not (_DRAFT / "platesolve").is_dir(),
    reason="draft_000366 not present",
)


def test_draft_366_core_photometry_sha_unchanged_by_cq_c():
    """LC + comp_quality + comparison pool unchanged (CQ-C touches comp_qa only)."""
    sha, n = compute_photometry_sha(_DRAFT)
    assert n == 283
    assert sha == PHOTOMETRY_SHA_CORE


def test_draft_366_extended_photometry_sha_post_cq_c():
    """Reference baseline includes comp_qa sidecars after fix-once locus re-baseline."""
    sha, n = compute_photometry_sha(_DRAFT, include_comp_qa=True)
    assert n == 426
    assert sha == PHOTOMETRY_SHA_BASELINE
