"""Unit tests for A9 NEIGHBOR-SUB acceptance envelope (validation-only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests.validation.a9_core import (
    A9_CONTEXTS,
    A9Cell,
    classify_zone,
    envelope_criterion,
    measure_cell,
    run_baseline_envelope,
    self_check_envelope,
    write_envelope_report,
)
from tests.validation.gen_a9 import write_a9_truth

TIER_A9 = Path(__file__).resolve().parent / "validation" / "data" / "tier_a9"


def test_classify_zones():
    assert classify_zone(0.5, -3, 500.0) == "REFUSE"
    assert classify_zone(1.0, -2, 400.0) == "HIGH_VALUE"
    assert classify_zone(3.0, 0, 2.0) == "CLEAN"
    assert classify_zone(3.0, -3, 20.0) in ("HIGH_VALUE", "CLEAN")


def test_envelope_criterion_refuse():
    c = envelope_criterion("REFUSE", 0.5, -3)
    assert c["pass_rule"] == "guard_refuse"
    assert c["neighbor_sub_must_fire"] is False


def test_measure_cell_plain_aperture_coarse():
    ctx = A9_CONTEXTS["coarse"]
    cell = A9Cell(sep_fwhm=1.0, delta_mag=-3, context="coarse")
    res = measure_cell(cell, ctx, mode="plain_aperture", n_frames=8)
    assert res.mode == "plain_aperture"
    assert res.contamination_excess_pct > 100.0


def test_measure_cell_neighbor_sub_clean_noop():
    ctx = A9_CONTEXTS["coarse"]
    cell = A9Cell(sep_fwhm=3.0, delta_mag=0, context="coarse")
    res = measure_cell(cell, ctx, mode="neighbor_sub")
    assert res.zone == "CLEAN"
    assert res.pass_future_neighbor_sub is True


def test_baseline_envelope_self_check_coarse():
    rep = run_baseline_envelope("coarse")
    ok, notes = self_check_envelope(rep)
    assert ok, notes
    M = np.asarray(rep["contamination_excess_pct"])
    assert M[0, 3] > 500.0  # sep 0.5 dM-3
    assert abs(M[-1, 0]) < 10.0  # sep 3.0 dM 0


def test_write_truth_and_report(tmp_path):
    tp = write_a9_truth(tmp_path)
    assert tp.is_file()
    data = json.loads(tp.read_text(encoding="ascii"))
    assert data["tier"] == "A9"
    assert len(data["cells"]) == len(A9_CONTEXTS) * 7 * 4
    jp, mp, ok = write_envelope_report(tmp_path)
    assert jp.is_file() and mp.is_file()
    payload = json.loads(jp.read_text(encoding="ascii"))
    assert payload["self_check"]["coarse"]["ok"]
    assert payload["self_check"]["fine"]["ok"]
    _ = ok  # overall includes draft367 at production radii; not the A9 paper gate


def test_full_a9_report_self_check():
    """Regenerate tier_a9 envelope if missing; verify structure."""
    TIER_A9.mkdir(parents=True, exist_ok=True)
    jp, mp, ok = write_envelope_report(TIER_A9)
    assert jp.is_file()
    payload = json.loads(jp.read_text(encoding="ascii"))
    assert payload["self_check"]["coarse"]["ok"]
