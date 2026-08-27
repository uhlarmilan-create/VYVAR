"""Byte-identity SHA guards when local anchor trees are present (optional).

Locked anchor: compare re-cut output to PHOTOMETRY_SHA_* (2026-06-11 lock).
Historical archive `draft_000387` remains at PHOTOMETRY_SHA_*_PRE_SPARSE_FB.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.photometry_sha import (
    PHOTOMETRY_SHA_BASELINE,
    PHOTOMETRY_SHA_BASELINE_PRE_SPARSE_FB,
    PHOTOMETRY_SHA_CORE,
    PHOTOMETRY_SHA_CORE_PRE_SPARSE_FB,
    compare_photometry_science_meaningful,
    compute_photometry_sha,
)

_ROOT = Path(__file__).resolve().parents[2]
_DRAFT387 = _ROOT / "Archive" / "Drafts" / "draft_000387"
_RECUT = _ROOT / "tmp" / "rebaseline_387_sparse_fb_cut1"


@pytest.mark.skipif(
    not (_DRAFT387 / "platesolve").is_dir(),
    reason="draft_000387 archive not present",
)
def test_draft_387_historical_core_photometry_sha():
    """Frozen archive matches pre-sparse-fallback SHA."""
    sha, n = compute_photometry_sha(_DRAFT387, strip_provenance=False)
    assert n == 2806
    assert sha == PHOTOMETRY_SHA_CORE_PRE_SPARSE_FB


@pytest.mark.skipif(
    not (_DRAFT387 / "platesolve").is_dir(),
    reason="draft_000387 archive not present",
)
def test_draft_387_historical_extended_photometry_sha():
    sha, n = compute_photometry_sha(_DRAFT387, include_comp_qa=True, strip_provenance=False)
    assert n == 4285
    assert sha == PHOTOMETRY_SHA_BASELINE_PRE_SPARSE_FB


@pytest.mark.skipif(
    not (_RECUT / "platesolve").is_dir(),
    reason="rebaseline cut1 not present",
)
def test_rebaseline_cut1_core_photometry_sha():
    sha, n = compute_photometry_sha(_RECUT, strip_provenance=False)
    assert n == 2806
    assert sha == PHOTOMETRY_SHA_CORE


@pytest.mark.skipif(
    not (_RECUT / "platesolve").is_dir(),
    reason="rebaseline cut1 not present",
)
def test_rebaseline_cut1_extended_photometry_sha():
    sha, n = compute_photometry_sha(_RECUT, include_comp_qa=True, strip_provenance=False)
    assert n == 4285
    assert sha == PHOTOMETRY_SHA_BASELINE


@pytest.mark.skipif(
    not ((_DRAFT387 / "platesolve").is_dir() and (_RECUT / "platesolve").is_dir()),
    reason="draft_000387 and rebaseline cut1 required",
)
def test_rebaseline_science_meaningful_vs_historical():
    rep = compare_photometry_science_meaningful(_DRAFT387, _RECUT)
    assert rep["summary"]["benign"] is True
    assert rep["summary"]["science_failures"] == 0
