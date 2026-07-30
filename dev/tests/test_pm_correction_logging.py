"""I-12: PM correction must not silently no-op when pmra/pmdec are absent."""

from __future__ import annotations

import logging

from vyvar_platesolver import _apply_pm_to_gaia_rows


def test_pm_missing_columns_logs_once(caplog) -> None:
    import vyvar_platesolver as vps

    vps._PM_COLUMNS_UNAVAILABLE_LOGGED = False
    rows = [{"ra": 120.0, "dec": 45.0, "source_id": "1"}]
    with caplog.at_level(logging.WARNING):
        out, n = _apply_pm_to_gaia_rows(rows, obs_year=2026.0)
    assert n == 0
    assert out[0]["ra"] == 120.0
    pm_logs = [r for r in caplog.records if "[PM]" in r.message]
    assert len(pm_logs) == 1
    assert "pmra/pmdec" in pm_logs[0].message
