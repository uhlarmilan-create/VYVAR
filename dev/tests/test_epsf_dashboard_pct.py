"""Regression test for ePSF dashboard ProgressColumn percent format (EPSF-VALID-02 F1)."""

from __future__ import annotations


def test_progress_column_pct_format_is_explicit_percent() -> None:
    """Stored pct_psf_ok is 0-100; ProgressColumn must use %.1f%% not implicit scale."""
    import inspect

    import ui_epsf_dashboard as mod

    src = inspect.getsource(mod._render_epsf_dashboard_body)
    assert 'format="%.1f%%"' in src
    assert "ProgressColumn" in src
    assert "psf_ac_policy" in src
