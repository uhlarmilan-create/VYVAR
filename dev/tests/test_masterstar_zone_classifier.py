# -*- coding: ascii -*-
"""Unit tests for scale-invariant MASTERSTAR flux-zone classification."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))

from pipeline import (  # noqa: E402
    _annotate_masterstars_flux_zones,
    _MASTERSTAR_ZONE_LOG_ONCE,
)


@pytest.fixture(autouse=True)
def _clear_zone_log_once() -> None:
    _MASTERSTAR_ZONE_LOG_ONCE.clear()
    yield
    _MASTERSTAR_ZONE_LOG_ONCE.clear()


def _mini_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flux": [800.0, 120.0, 45000.0],
            "peak_dao": [45.0, 8.0, 50000.0],
            "peak_max_adu": [50.0, 10.0, 50000.0],
        }
    )


def test_peak_dao_missing_marks_unknown_not_flux_guess() -> None:
    df = _mini_df()
    df.loc[1, "peak_dao"] = float("nan")
    out = _annotate_masterstars_flux_zones(
        df,
        noise_floor_adu=2105.9,
        equipment_saturate_adu=65535.0,
        zone_mode="peak_significance",
        sigma_px=83.85,
        zone_sigma_linear=3.5,
        zone_sigma_noisy1=2.5,
        zone_sigma_noisy2=1.5,
    )
    assert out.loc[1, "zone"] == "unknown"
    assert not bool(out.loc[1, "is_usable"])


def test_sigma_unresolvable_leaves_empty_zone() -> None:
    df = _mini_df()
    out = _annotate_masterstars_flux_zones(
        df,
        noise_floor_adu=2105.9,
        equipment_saturate_adu=65535.0,
        zone_mode="peak_significance",
        sigma_px=None,
        sky_median_adu=None,
        prematch_peak_sigma_floor=None,
    )
    assert (out["zone"] == "").all()
    assert not out["is_usable"].any()


def test_legacy_path_unchanged_on_fixture_shape() -> None:
    fixture = Path(__file__).resolve().parents[1] / (
        "results/context/session_20260727/draft_452_masterstars_full_match.csv"
    )
    ms = pd.read_csv(fixture, low_memory=False)
    expected = ms["zone"].copy()
    sample = ms.drop(columns=["zone", "is_usable", "is_noisy", "is_saturated"], errors="ignore")
    nf = float(ms["noise_floor_adu"].iloc[0])
    sat = float(ms["saturate_limit_adu_85pct"].iloc[0])
    out = _annotate_masterstars_flux_zones(
        sample,
        noise_floor_adu=nf,
        equipment_saturate_adu=sat / 0.85,
        saturate_limit_adu_fallback=sat,
        zone_mode="legacy",
    )
    pd.testing.assert_series_equal(out["zone"], expected, check_names=False)


def test_pedestal_invariant_under_peak_significance() -> None:
    df = _mini_df()
    sigma = 10.0
    kw = dict(
        equipment_saturate_adu=65535.0,
        zone_mode="peak_significance",
        sigma_px=sigma,
        sky_median_adu=1955.0,
        prematch_peak_sigma_floor=1.8,
        zone_sigma_linear=3.5,
        zone_sigma_noisy1=2.5,
        zone_sigma_noisy2=1.5,
    )
    nf_low = 1955.0 + 1.8 * sigma
    nf_high = 33487.0 + 1.8 * sigma
    z_low = _annotate_masterstars_flux_zones(df, noise_floor_adu=nf_low, **kw)["zone"].tolist()
    z_high = _annotate_masterstars_flux_zones(df, noise_floor_adu=nf_high, **kw)["zone"].tolist()
    assert z_low == z_high


def test_pedestal_changes_legacy_zones() -> None:
    df = pd.DataFrame(
        {
            "flux": [2000.0],
            "peak_dao": [45.0],
            "peak_max_adu": [50.0],
        }
    )
    kw = dict(
        equipment_saturate_adu=65535.0,
        zone_mode="legacy",
    )
    nf_low = 1955.0 + 1.8 * 10.0
    nf_high = 33487.0 + 1.8 * 10.0
    z_low = _annotate_masterstars_flux_zones(df, noise_floor_adu=nf_low, **kw)["zone"].iloc[0]
    z_high = _annotate_masterstars_flux_zones(df, noise_floor_adu=nf_high, **kw)["zone"].iloc[0]
    assert z_low == "linear"
    assert z_high == "noisy3"
