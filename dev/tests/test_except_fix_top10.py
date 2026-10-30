"""Unit tests for EXCEPT-FIX-1 TOP-10 terminal failure behavior."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from except_fix_counters import get_except_fix_counters, reset_except_fix_counters


def test_exc0132_annulus_invalid_returns_nan_flux(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from photometry_core import _annulus_sky_subtracted_flux
    from photutils.aperture import CircularAnnulus

    class _EmptyMask:
        def get_values(self, _d):
            return np.array([], dtype=np.float64)

    monkeypatch.setattr(
        CircularAnnulus,
        "to_mask",
        lambda self, method="center": [_EmptyMask()],
    )

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    data = np.ones((32, 32), dtype=np.float64) * 100.0
    flux, sky, peak = _annulus_sky_subtracted_flux(data, 16.0, 16.0, r_ap=3.0, r_in=5.0, r_out=8.0)
    assert not np.isfinite(flux)
    assert not np.isfinite(sky)
    assert get_except_fix_counters().sky_annulus_invalid >= 1
    assert any("annulus sky invalid" in r.message for r in caplog.records)








def test_require_comparison_stars_per_target_schema_rejects_pool(tmp_path: Path) -> None:
    import pandas as pd

    from photometry_core import _require_comparison_stars_per_target_schema

    pool = tmp_path / "comparison_stars.csv"
    pd.DataFrame(
        {
            "comp_id": ["1"],
            "role": ["comp"],
            "catalog_id": ["1499883638682689408"],
            "x": [100.0],
            "y": [200.0],
        }
    ).to_csv(pool, index=False)
    df = pd.read_csv(pool, dtype={"catalog_id": str})
    with pytest.raises(ValueError, match="comparison_stars_per_target"):
        _require_comparison_stars_per_target_schema(df, pool)


def test_aperture_empty_comp_drop_row_and_counter(caplog: pytest.LogCaptureFixture) -> None:
    from photometry_core import _aperture_skip_empty_comps_target

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    rows = _aperture_skip_empty_comps_target(
        target_cid="123",
        target_name="T1",
        zone_flag="linear",
        summary_rows=[],
    )
    assert len(rows) == 1
    assert rows[0]["catalog_id"] == "123"
    assert rows[0]["n_frames"] == 0
    assert get_except_fix_counters().aperture_empty_comp_drop == 1
    assert any("no comp stars" in r.message for r in caplog.records)
