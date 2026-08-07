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
    _MASTERSTAR_ZONE_SIGMA_STEP,
    _annotate_masterstars_flux_zones,
    _masterstar_zone_sigma_thresholds,
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
            "flux": [800.0, 120.0, 8000.0],
            "peak_dao": [45.0, 8.0, 45.0],
            "peak_max_adu": [50.0, 10.0, 55.0],
        }
    )


def test_zone_thresholds_derive_from_dao_detection_n_equiv() -> None:
    n_equiv = 3.78
    t1, t2, t3 = _masterstar_zone_sigma_thresholds(n_equiv)
    assert t1 == n_equiv
    assert t2 == n_equiv - _MASTERSTAR_ZONE_SIGMA_STEP
    assert t3 == n_equiv - 2.0 * _MASTERSTAR_ZONE_SIGMA_STEP


def test_peak_dao_missing_marks_unknown_not_flux_guess() -> None:
    df = _mini_df()
    df.loc[1, "peak_dao"] = float("nan")
    out = _annotate_masterstars_flux_zones(
        df,
        noise_floor_adu=2105.9,
        equipment_saturate_adu=65535.0,
        sigma_px=83.85,
        dao_detection_n_equiv=3.78,
    )
    assert out.loc[1, "zone"] == "unknown"
    assert not bool(out.loc[1, "is_usable"])


def test_sigma_unresolvable_leaves_empty_zone() -> None:
    df = _mini_df()
    out = _annotate_masterstars_flux_zones(
        df,
        noise_floor_adu=2105.9,
        equipment_saturate_adu=65535.0,
        sigma_px=None,
        sky_median_adu=None,
        prematch_peak_sigma_floor=None,
        dao_detection_n_equiv=3.78,
    )
    assert (out["zone"] == "").all()
    assert not out["is_usable"].any()


def test_pedestal_invariant_zones() -> None:
    df = _mini_df()
    sigma = 10.0
    kw = dict(
        equipment_saturate_adu=65535.0,
        sigma_px=sigma,
        sky_median_adu=1955.0,
        prematch_peak_sigma_floor=1.8,
        frame_max_adu=5000.0,
        dao_detection_n_equiv=3.78,
    )
    nf_low = 1955.0 + 1.8 * sigma
    nf_high = 33487.0 + 1.8 * sigma
    z_low = _annotate_masterstars_flux_zones(df, noise_floor_adu=nf_low, **kw)["zone"].tolist()
    z_high = _annotate_masterstars_flux_zones(df, noise_floor_adu=nf_high, **kw)["zone"].tolist()
    assert z_low == z_high


def test_precalibrated_field_skips_camera_peak_saturation() -> None:
    df = pd.DataFrame(
        {
            "flux": [500.0],
            "peak_dao": [45.0],
            "peak_max_adu": [60000.0],
        }
    )
    out = _annotate_masterstars_flux_zones(
        df,
        noise_floor_adu=33493.9,
        equipment_saturate_adu=65535.0,
        sigma_px=10.0,
        sky_median_adu=33487.0,
        prematch_peak_sigma_floor=1.8,
        frame_max_adu=98232.0,
        empirical_clip_adu=None,
        dao_detection_n_equiv=3.78,
    )
    assert out.loc[0, "zone"] != "saturated"
    assert not bool(out.loc[0, "is_saturated"])
