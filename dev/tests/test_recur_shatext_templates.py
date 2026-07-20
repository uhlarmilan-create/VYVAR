"""Recurrence guard: prose templates in SHA-covered photometry sidecars (ANCHOR-RECUT-SIGMA-NOTES).

2026-07-20: ENCODING-POLICY (ecbae90) folded VSX slope-exclusion notes from Unicode sigma
(U+03C3) to ASCII 'sigma' in comp_quality_*.json; 19 anchor files drifted while all
166 lightcurve_*.csv stayed byte-identical. P1 mini never emits slope notes (coverage hole).

Any edit to these templates is a conscious SHA/text change - update this test and run --full.
"""
from __future__ import annotations

from pathlib import Path

# Canonical templates from photometry_core.check_comparison_stability slope branch.
_SLOPE_NOTE_EXCLUDED = "slope={slope_mmag_hr:.1f} mmag/hr ({slope_sig:.1f}sigma)"
_SLOPE_NOTE_SUSPECT_KEPT = "slope={slope_mmag_hr:.1f} mmag/hr ({slope_sig:.1f}sigma, kept: n_good<min)"


def _format_slope_note_excluded(slope_mmag_hr: float, slope_sig: float) -> str:
    return _SLOPE_NOTE_EXCLUDED.format(slope_mmag_hr=slope_mmag_hr, slope_sig=slope_sig)


def _format_slope_note_suspect_kept(slope_mmag_hr: float, slope_sig: float) -> str:
    return _SLOPE_NOTE_SUSPECT_KEPT.format(slope_mmag_hr=slope_mmag_hr, slope_sig=slope_sig)


def test_comp_quality_slope_note_templates_ascii_sigma() -> None:
    """Slope exclusion notes use ASCII 'sigma', never U+03C3 (ENCODING-POLICY)."""
    ex = _format_slope_note_excluded(6.4, 3.6)
    sus = _format_slope_note_suspect_kept(6.4, 3.6)
    for note in (ex, sus):
        assert "sigma" in note
        assert "\u03c3" not in note
        assert "\\" not in note
    assert ex == "slope=6.4 mmag/hr (3.6sigma)"
    assert sus == "slope=6.4 mmag/hr (3.6sigma, kept: n_good<min)"


def test_comp_quality_slope_note_templates_match_photometry_core_source() -> None:
    """Pinned strings must stay in sync with photometry_core slope note f-strings."""
    src = (Path(__file__).resolve().parents[2] / "src_py" / "photometry_core.py").read_text(
        encoding="utf-8"
    )
    assert 'f"slope={slope_mmag_hr:.1f} mmag/hr ({slope_sig:.1f}sigma)"' in src
    assert (
        'f"slope={slope_mmag_hr:.1f} mmag/hr ({slope_sig:.1f}sigma, kept: n_good<min)"' in src
    )
    assert "mmag/hr (%.1fsigma)" in src
    assert "\u03c3" not in src[src.index("slope_mmag_hr") : src.index("slope_mmag_hr") + 800]


def test_comp_quality_aperture_correction_reason_literals() -> None:
    """Stable reason tokens written into comp_quality JSON (SHA-covered extended set)."""
    src = (Path(__file__).resolve().parents[2] / "src_py" / "photometry_core.py").read_text(
        encoding="utf-8"
    )
    for token in ("insufficient_ref_stars", "disabled"):
        assert token in src
