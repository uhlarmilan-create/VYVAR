#!/usr/bin/env python3
"""INVARIANTS P1 seed: draft_435 double-photometry SHA + census-band smoke (slow).

Marked slow; skipped unless VYVAR_INVARIANTS_P1=1 or --run-invariants-p1.
Uses in-Archive snapshot as golden; does not require UI path in this seed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from config import AppConfig
from tests.photometry_sha import compute_photometry_sha

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "draft_000435_snapshot_skysurface_20260716"
EXPECTED_CORE = "3d26f4692ac81fc52db6ef9f70b148f9f7c56a5bb5e84e637339c4883ba47a96"
EXPECTED_EXT = "6420f1daa53a0d5d0a92bfd1ab30eba68e2ab88be8fe5f4c68048a5463054ac8"


def _enabled() -> bool:
    return str(os.environ.get("VYVAR_INVARIANTS_P1", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


pytestmark = pytest.mark.skipif(not _enabled(), reason="set VYVAR_INVARIANTS_P1=1 to run P1 golden")


def test_p1_snapshot_sha_matches_registered() -> None:
    cfg = AppConfig()
    snap = Path(cfg.archive_root) / "Drafts" / SNAPSHOT
    assert snap.is_dir(), f"missing snapshot {snap}"
    core, nc = compute_photometry_sha(snap, include_comp_qa=False)
    ext, ne = compute_photometry_sha(snap, include_comp_qa=True)
    assert core == EXPECTED_CORE
    assert ext == EXPECTED_EXT
    assert nc == 333
    assert ne == 499


def test_p1_census_fingerprint_in_meta() -> None:
    cfg = AppConfig()
    meta_path = (
        Path(cfg.archive_root)
        / "Drafts"
        / SNAPSHOT
        / "platesolve"
        / "NoFilter_60_2"
        / "photometry"
        / "pipeline_meta.json"
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("matched_world2pix_identity_n") == 2842
    p95 = float(meta.get("matched_world2pix_identity_p95_px"))
    assert 1.0 < p95 < 2.0
    assert int(meta.get("sky_surface_order") or 0) == 2
