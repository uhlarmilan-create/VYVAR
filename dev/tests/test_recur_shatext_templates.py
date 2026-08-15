"""Recurrence guard: prose templates in SHA-covered photometry sidecars (ANCHOR-RECUT-SIGMA-NOTES).

2026-07-20: ENCODING-POLICY (ecbae90) folded VSX slope-exclusion notes from Unicode sigma
(U+03C3) to ASCII 'sigma' in comp_quality_*.json; 19 anchor files drifted while all
166 lightcurve_*.csv stayed byte-identical. P1 mini never emits slope notes (coverage hole).

COMP-ADMIT-03 (2026-08-15): slope no longer ejects members; note marks suspect kept.

Any edit to these templates is a conscious SHA/text change - update this test and run --full.
"""
from __future__ import annotations

from pathlib import Path

_SLOPE_NOTE_KEPT = (
    "slope={slope_mmag_hr:.1f} mmag/hr ({slope_sig:.1f}sigma; kept COMP-ADMIT-03)"
)


def _format_slope_note_kept(slope_mmag_hr: float, slope_sig: float) -> str:
    return _SLOPE_NOTE_KEPT.format(slope_mmag_hr=slope_mmag_hr, slope_sig=slope_sig)


def test_comp_quality_slope_note_templates_ascii_sigma() -> None:
    """Slope notes use ASCII 'sigma', never U+03C3 (ENCODING-POLICY)."""
    note = _format_slope_note_kept(6.4, 3.6)
    assert "sigma" in note
    assert "\u03c3" not in note
    assert "\\" not in note
    assert note == "slope=6.4 mmag/hr (3.6sigma; kept COMP-ADMIT-03)"


def test_comp_quality_slope_note_templates_match_photometry_core_source() -> None:
    """Pinned strings must stay in sync with photometry_core slope note f-strings."""
    src = (Path(__file__).resolve().parents[2] / "src_py" / "photometry_core.py").read_text(
        encoding="utf-8"
    )
    assert (
        'f"slope={slope_mmag_hr:.1f} mmag/hr ({slope_sig:.1f}sigma; kept COMP-ADMIT-03)"' in src
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
