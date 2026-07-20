"""Tests for comp_quality JSON parse/write helpers."""

from __future__ import annotations

import numpy as np

from photometry_core import (
    check_comparison_stability,
    comp_quality_quality_strings,
    parse_comp_quality_json_map,
)


def test_parse_comp_quality_flat_and_structured():
    raw = {
        "111": "good",
        "222": {"quality": "suspect", "note": "isolated_bin"},
        "333": {"quality": "excluded", "note": "few_frames"},
        "selected_tier": "TIER1",
        "aperture_correction": {"ok": True, "reason": "ok"},
    }
    m = parse_comp_quality_json_map(raw)
    assert m["111"] == {"quality": "good", "note": ""}
    assert m["222"]["quality"] == "suspect"
    assert m["222"]["note"] == "isolated_bin"
    assert m["333"]["quality"] == "excluded"
    assert "selected_tier" not in m
    assert comp_quality_quality_strings(m)["222"] == "suspect"


def test_stability_outlier_note_on_exclude():
    """One noisy comp among stable comps -> excluded with outlier note."""
    rng = np.random.default_rng(1)
    n = 40
    bjd = np.linspace(2459000.0, 2459000.4, n)
    comp_lc = {}
    comp_bjd = {}
    for i in range(5):
        cid = f"c{i}"
        comp_lc[cid] = 12.0 + 0.002 * rng.standard_normal(n)
        comp_bjd[cid] = bjd.copy()
    comp_lc["c4"] = 12.0 + 0.08 * rng.standard_normal(n)
    comp_bjd["c4"] = bjd.copy()
    q = check_comparison_stability(
        comp_lc,
        comp_bjd=comp_bjd,
        n_comp_min=3,
        outlier_sigma=3.0,
        max_comp_slope_mmag_hr=99.0,
    )
    assert q["c4"]["quality"] == "excluded"
    assert "outlier" in q["c4"].get("note", "")
    assert "p2p=" in q["c4"]["note"]


def test_stability_few_frames_note():
    comp_lc = {"x": np.array([1.0, 2.0])}
    q = check_comparison_stability(comp_lc, max_comp_slope_mmag_hr=0)
    assert q["x"]["quality"] == "excluded"
    assert q["x"]["note"] == "few_frames"
