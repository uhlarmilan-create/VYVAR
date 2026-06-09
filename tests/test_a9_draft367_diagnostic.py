"""A9 draft 367 fine-scale NEIGHBOR-SUB diagnostic (analysis-only)."""
from __future__ import annotations

import json
from pathlib import Path

from tests.validation.a9_core import (
    A9_CONTEXTS,
    DRAFT367_MISMATCH_RATIO,
    draft367_a9_verdict,
    draft367_combined_decision,
    psf_variant_spec,
    run_draft367_neighbor_sub_diagnostic,
    write_draft367_report,
)

_MOCK_AUDIT = {
    "draft_id": 367,
    "setup": "Red_180_2",
    "ratio_moffat_vs_stars": DRAFT367_MISMATCH_RATIO,
    "plate_scale_arcsec": 0.3889,
    "vy_fwhm_gauss": 6.0203,
    "fwhm_moffat_native": 5.3925,
    "fwhm_stars_native": 5.396,
    "fwhm_moffat_arcsec": 2.0971,
}


def test_draft367_variant_anchors_measured_mismatch():
    spec = psf_variant_spec("draft367")
    ctx = A9_CONTEXTS["draft367"]
    ratios = spec.model_over_star_fwhm(ctx)
    assert 0.98 <= ratios["target"] <= 1.02
    assert spec.inject_ellip < 0.05


def test_draft367_decision_verdict_edge_case():
    assert draft367_a9_verdict(
        {"fail_silent_count": 1, "high_value_pass_recover_rate": 0.83, "mismatch_ratio": 0.999}
    ) == "A9_EDGE_FAIL_SILENT"
    assert draft367_a9_verdict(
        {"fail_silent_count": 0, "high_value_pass_recover_rate": 0.60, "mismatch_ratio": 1.10}
    ) == "MISMATCH_STILL_HIGH"
    assert draft367_combined_decision("A9_PASS", "SPARSE") == "VALIDATED_FINE_SCALE_IDLE"
    assert draft367_combined_decision("A9_PASS", "PROCEED_2B_CANDIDATE") == "PROCEED_2B_CANDIDATE"


def test_draft367_neighbor_sub_diagnostic_structure():
    diag = run_draft367_neighbor_sub_diagnostic(epsf_audit=_MOCK_AUDIT)
    ns = diag["neighbor_sub"]
    assert diag["mismatch_ratio"] == DRAFT367_MISMATCH_RATIO
    assert "fail_silent_count" in ns
    assert ns["high_value_pass_recover_rate"] is not None
    assert ns["high_value_pass_recover_rate"] >= 0.75
    assert ns["fail_silent_count"] == 0
    assert len(ns["cells"]) == 28
    assert diag["a9_verdict"] == "A9_PASS"
    assert diag["decision"] in (
        "VALIDATED_FINE_SCALE_IDLE",
        "PROCEED_2B_CANDIDATE",
        "BLOCK_2B_GUARDS",
    )


def test_write_draft367_report(tmp_path):
    jp, mp = write_draft367_report(tmp_path, epsf_audit=_MOCK_AUDIT)
    assert jp.is_file() and mp.is_file()
    text = mp.read_text(encoding="ascii")
    assert "0.9994" in text
    assert "Part 2" in text
    payload = json.loads(jp.read_text(encoding="ascii"))
    assert payload["context"] == "draft367"
