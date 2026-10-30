"""BAND-2026-10: Gaia DR3 G + BP-RP -> observed band (DR3 docs 5.5.1, Tables 5.8 / 5.9 / 5.10)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

from gaia_bands import band_for_filter, band_mag_from_gaia  # noqa: E402


def test_table58_values():
    x = 1.0
    g, s = band_mag_from_gaia(12.0, x, "g")
    assert abs(g - (12.0 - (0.2199 - 0.6365 - 0.1548 + 0.0064))) < 1e-12 and s == 0.0745
    r, _ = band_mag_from_gaia(12.0, x, "r")
    assert abs(r - (12.0 - (-0.09837 + 0.08592 + 0.1907 - 0.1701 + 0.02263))) < 1e-12
    v, _ = band_mag_from_gaia(12.0, x, "V")
    assert abs(v - (12.0 - (-0.02704 + 0.01424 - 0.2156 + 0.01426))) < 1e-12


def test_validity_ranges_no_extrapolation():
    assert math.isnan(band_mag_from_gaia(12.0, 2.5, "g")[0])  # > 2.0: only M giants (note g)
    assert math.isnan(band_mag_from_gaia(12.0, 0.2, "g")[0])  # < 0.3
    assert math.isfinite(band_mag_from_gaia(12.0, 0.1, "r")[0])
    assert math.isnan(band_mag_from_gaia(12.0, float("nan"), "r")[0])
    assert band_mag_from_gaia(12.0, float("nan"), "G") == (12.0, 0.0)


def test_filter_names_case_sensitive():
    assert [band_for_filter(f) for f in ("g", "r", "i", "R", "Rc", "V", "B", "Clear", "L")] == [
        "g", "r", "i", "R", "R", "V", "B", "G", "G"]


def test_v_colour_limit_is_table_5_10() -> None:
    """GAIA-JC-RANGE-2026-10 (Milan 2026-10-29): V valid for -0.5 < BP-RP < 5.0, no G magnitude limit."""
    import math

    from gaia_bands import band_mag_from_gaia

    assert math.isfinite(band_mag_from_gaia(14.0, 4.99, "V")[0])
    assert math.isnan(band_mag_from_gaia(14.0, 5.05, "V")[0])
    for g in (6.0, 18.5):  # no magnitude range in Table 5.10
        assert math.isfinite(band_mag_from_gaia(g, 1.0, "V")[0])
