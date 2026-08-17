# -*- coding: ascii -*-
"""Unit tests for scale-invariant MASTERSTAR flux-zone classification."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))

from pipeline import (  # noqa: E402
    _annotate_masterstars_flux_zones,
    _masterstar_zone_linear_threshold,
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


def test_zone_linear_threshold_from_dao_detection_n_equiv() -> None:
    n_equiv = 3.78
    t1 = _masterstar_zone_linear_threshold(n_equiv)
    assert t1 == n_equiv


def test_sub_linear_stars_marked_noise() -> None:
    df = pd.DataFrame(
        {
            "flux": [800.0, 120.0],
            "peak_dao": [45.0, 8.0],
            "peak_max_adu": [50.0, 10.0],
        }
    )
    out = _annotate_masterstars_flux_zones(
        df,
        noise_floor_adu=2105.9,
        equipment_saturate_adu=65535.0,
        sigma_px=10.0,
        sky_median_adu=1955.0,
        prematch_peak_sigma_floor=1.8,
        dao_detection_n_equiv=3.78,
    )
    assert out.loc[0, "zone"] == "linear"
    assert out.loc[1, "zone"] == "noise"
    assert bool(out.loc[1, "is_noisy"])


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


def test_pandas_nan_limit_is_silently_false() -> None:
    """Fire-proof of the historical hole: peak > NaN never flags saturation."""
    peak = pd.Series([65000.0])
    assert not bool((peak > float("nan")).iloc[0])


def test_inv_sat_limit_unresolved_clip_does_not_silently_admit() -> None:
    """INV-SAT-LIMIT: limit=NaN and peak=65000 must flag saturated after the fix."""
    df = pd.DataFrame(
        {
            "flux": [8000.0],
            "peak_dao": [45.0],
            "peak_max_adu": [65000.0],
        }
    )
    out = _annotate_masterstars_flux_zones(
        df,
        noise_floor_adu=2105.9,
        equipment_saturate_adu=None,
        saturate_limit_adu_fallback=None,
        sigma_px=10.0,
        sky_median_adu=1955.0,
        prematch_peak_sigma_floor=1.8,
        dao_detection_n_equiv=3.78,
    )
    assert math.isfinite(float(out.loc[0, "saturate_limit_adu"]))
    assert math.isfinite(float(out.loc[0, "saturate_limit_adu_85pct"]))
    assert bool(out.loc[0, "is_saturated"])
    assert str(out.loc[0, "zone"]) == "saturated"


def test_inv_sat_limit_nan_equipment_value_does_not_silently_admit() -> None:
    df = pd.DataFrame(
        {
            "flux": [8000.0],
            "peak_dao": [45.0],
            "peak_max_adu": [65000.0],
        }
    )
    out = _annotate_masterstars_flux_zones(
        df,
        noise_floor_adu=2105.9,
        equipment_saturate_adu=float("nan"),
        saturate_limit_adu_fallback=float("nan"),
        sigma_px=10.0,
        sky_median_adu=1955.0,
        prematch_peak_sigma_floor=1.8,
        dao_detection_n_equiv=3.78,
    )
    assert bool(out.loc[0, "is_saturated"])


def test_effective_saturation_limit_never_none() -> None:
    from astropy.io import fits

    from pipeline import SAT_LIMIT_CONTAINER_CLIP_ADU, _effective_saturation_limit

    hdr = fits.Header()
    hdr["BITPIX"] = -32
    lim, src = _effective_saturation_limit(hdr, fallback_adu=None, equipment_saturate_adu=None)
    assert lim == SAT_LIMIT_CONTAINER_CLIP_ADU
    assert src == "conservative_default_container_clip_65535"


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


def test_masterstar_stack_few_percent_overshoot_still_flags_peak() -> None:
    """515-like: float stack max 68429 vs clip 65535 must not disable the gate."""
    df = pd.DataFrame(
        {
            "flux": [8000.0],
            "peak_dao": [45.0],
            "peak_max_adu": [65000.0],
        }
    )
    out = _annotate_masterstars_flux_zones(
        df,
        noise_floor_adu=2105.9,
        equipment_saturate_adu=65535.0,
        sigma_px=10.0,
        sky_median_adu=1401.0,
        prematch_peak_sigma_floor=1.8,
        frame_max_adu=68429.0,
        dao_detection_n_equiv=3.78,
    )
    assert bool(out.loc[0, "is_saturated"])
    assert str(out.loc[0, "zone"]) == "saturated"
