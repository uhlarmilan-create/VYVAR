"""A9 PSF-mismatch diagnostic (analysis-only; step 2b gate)."""
from __future__ import annotations

from pathlib import Path

from tests.validation.a9_core import (
    A9_CONTEXTS,
    classify_cell_outcome,
    psf_variant_spec,
    run_mismatch_diagnostic,
    write_mismatch_diagnostic_report,
)

TIER_A9 = Path(__file__).resolve().parent / "validation" / "data" / "tier_a9"


def test_realistic_variant_anchors_epsf_audit():
    spec = psf_variant_spec("realistic")
    ctx = A9_CONTEXTS["coarse"]
    ratios = spec.model_over_star_fwhm(ctx)
    assert 1.05 <= ratios["target"] <= 1.12
    assert spec.inject_ellip >= 0.05


def test_legacy_mismatch_is_stress_not_audit_direction():
    spec = psf_variant_spec("mismatch")
    ctx = A9_CONTEXTS["coarse"]
    ratios = spec.model_over_star_fwhm(ctx)
    assert ratios["neighbour"] < 1.0  # model narrower than inject neighbour (inverted vs field)
    assert spec.fit_beta != spec.inject_beta


def test_classify_outcome_refuse_zone():
    assert (
        classify_cell_outcome(
            "REFUSE", 500.0, 500.0, refused=True, subtracted=False, criterion={}
        )
        == "PASS-REFUSE"
    )
    assert (
        classify_cell_outcome(
            "REFUSE", 500.0, 50.0, refused=False, subtracted=True, criterion={}
        )
        == "FAIL-REFUSE-MISS"
    )


def test_mismatch_diagnostic_structure():
    diag = run_mismatch_diagnostic("coarse", variants=("realistic",))
    real = diag["variants"]["realistic"]
    assert "fail_silent_count" in real
    assert "verdict" in real
    assert real["fail_silent_count"] <= 1
    assert len(real["cells"]) == 28


def test_write_mismatch_diagnostic_report(tmp_path):
    jp, mp = write_mismatch_diagnostic_report(tmp_path)
    assert jp.is_file() and mp.is_file()
    assert "realistic" in mp.read_text(encoding="ascii")
