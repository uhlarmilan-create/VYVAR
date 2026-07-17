"""Tests for lunar_context.py."""

from __future__ import annotations

import math

import pytest

from lunar_context import (
    _angular_separation,
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


def test_angular_separation_known() -> None:
    sep = _angular_separation(0.0, 0.0, 0.0, 90.0)
    assert abs(sep - 90.0) < 1e-6


def test_jd_midpoint() -> None:
    mid = get_jd_midpoint([2460000.0, 2460001.0, float("nan")])
    assert mid == pytest.approx(2460000.5)
