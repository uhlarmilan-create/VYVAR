"""Tests for lunar_context.py."""

from __future__ import annotations

import math

import pytest

from lunar_context import (
    _lunar_risk_from_metrics,
    get_jd_midpoint,
    get_lunar_context,
)


def test_lunar_context_known_date() -> None:
    result = get_lunar_context(
        jd_mid=2460812.0,
        ra_field=198.0,
        dec_field=47.0,
        lat=50.07,
        lon=14.42,
        alt_m=355.5,
    )
    assert result["lunar_phase_pct"] > 80.0
    sep = result["lunar_separation_deg"]
    assert math.isfinite(sep) and 0.0 <= sep <= 180.0
    assert math.isfinite(result["lunar_altitude_deg"])
    assert result["lunar_risk"] in {"LOW", "MEDIUM", "HIGH"}


def test_lunar_risk_below_horizon() -> None:
    risk, _ = _lunar_risk_from_metrics(50.0, 10.0, -5.0)
    assert risk == "LOW"


def test_lunar_risk_new_moon() -> None:
    risk, _ = _lunar_risk_from_metrics(5.0, 10.0, 30.0)
    assert risk == "LOW"


def test_lunar_risk_high() -> None:
    risk, _ = _lunar_risk_from_metrics(80.0, 15.0, 30.0)
    assert risk == "HIGH"


def test_separation_is_topocentric() -> None:
    """REPORT-AUDIT-2026-10: a field at the Moon's apparent position is ~0 deg from the Moon (the
    barycentric direction used before was ~1 deg off for a topocentric Moon)."""
    site = dict(lat=50.11, lon=14.70, alt_m=300.0)
    r0 = get_lunar_context(2461162.0, 0.0, 0.0, **site)
    r1 = get_lunar_context(2461162.0, r0["moon_ra_deg"], r0["moon_dec_deg"], **site)
    assert r1["lunar_separation_deg"] < 0.05


def test_lunar_age_waning() -> None:
    """Age runs 0..29.5 d; a waning Moon (after full) is older than 14.8 d, not mirrored."""
    site = dict(lat=50.11, lon=14.70, alt_m=300.0)
    waxing = get_lunar_context(2461300.4, 200.0, 40.0, **site)["lunar_age_days"]
    waning = get_lunar_context(2461315.4, 200.0, 40.0, **site)["lunar_age_days"]
    assert 3.0 < waxing < 9.0
    assert 17.0 < waning < 24.0


def test_jd_midpoint() -> None:
    mid = get_jd_midpoint([2460000.0, 2460001.0, float("nan")])
    assert mid == pytest.approx(2460000.5)
