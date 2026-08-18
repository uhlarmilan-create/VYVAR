#!/usr/bin/env python3
"""INVARIANTS P1 seed: frozen 516 snapshot SHA + census-band smoke (slow).

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

ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = ROOT / "dev" / "validation" / "VYVAR_VALIDATION_LEDGER.json"
SNAPSHOT = "draft_000516_snapshot_cleanrebuild_20260818"
EXPECTED_CORE = "477dc8cfc292ed63910ecca6ea1dacfda279fee2850422229739a5cf7db90956"
EXPECTED_EXT = "f71e07226893a6b07e24999927bad0da8c16e6407656fc97ee02e0d57494be5d"


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
    assert nc == 97
    assert ne == 145


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
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    anchor = next(i for i in ledger["items"] if i.get("id") == "VL-ANCHOR-WCSINV")
    fp = anchor["census_fingerprint"]
    dyn = meta.get("dynamic_params") or {}
    assert int(dyn.get("n_stars_dao") or 0) == fp["n_raw_dao"]
    lc = meta.get("lc_quality_summary") or {}
    assert int(lc.get("total") or 0) == fp["targets"]
    p95 = float(anchor.get("identity_p95_baseline_px"))
    assert 1.0 < p95 < 2.0
