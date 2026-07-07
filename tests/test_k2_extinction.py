"""Unit tests for second-order extinction (k2_extinction)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from config import AppConfig
from k2_extinction import (
    K2Source,
    SLOPE_GR_PER_BPRP,
    apply_k2_per_frame,
    computed_k2_bprp_for_token,
    filter_token_from_obs_group,
    resolve_k2_bprp_value,
)


def test_g_k2_bprp_from_smith_jordi() -> None:
    expected = -0.016 * SLOPE_GR_PER_BPRP
    got = computed_k2_bprp_for_token("G")
    assert got is not None
    assert got == pytest.approx(expected, rel=1e-6)


def test_v_k2_zero() -> None:
    assert computed_k2_bprp_for_token("V") == 0.0


def test_osc_blue_k2_none() -> None:
    assert computed_k2_bprp_for_token("BLUE") is None


def test_nofilter_k2_none() -> None:
    cfg = AppConfig()
    cfg.k2_mode = "literature"
    val, src = resolve_k2_bprp_value(cfg, "NoFilter_60_2")
    assert src is K2Source.NONE
    assert not math.isfinite(val)


def test_g_band_literature_default() -> None:
    cfg = AppConfig()
    cfg.k2_mode = "literature"
    val, src = resolve_k2_bprp_value(cfg, "g_60_2")
    assert src is K2Source.LITERATURE_DEFAULT
    assert val == pytest.approx(-0.016 * SLOPE_GR_PER_BPRP, rel=1e-6)


def test_k2_config_override() -> None:
    cfg = AppConfig()
    cfg.k2_mode = "literature"
    cfg.k2_defaults_bprp = {"g": -0.99}
    val, src = resolve_k2_bprp_value(cfg, "g_60_2")
    assert src is K2Source.LITERATURE_DEFAULT
    assert val == pytest.approx(-0.99)


def test_k2_off_mode() -> None:
    cfg = AppConfig()
    cfg.k2_mode = "off"
    val, src = resolve_k2_bprp_value(cfg, "B_20_2")
    assert src is K2Source.NONE


def test_apply_k2_per_frame_analytic() -> None:
    mag = np.array([10.0, 10.0])
    am = np.array([1.0, 2.0])
    k2 = -0.014
    bp_t, bp_med = 1.2, 0.8
    out, delta, src = apply_k2_per_frame(
        mag,
        am,
        object_bp_rp=bp_t,
        bp_rp_comp_med=bp_med,
        k2_value=k2,
        k2_source=K2Source.LITERATURE_DEFAULT,
    )
    d_c = bp_t - bp_med
    assert out[0] == pytest.approx(10.0 - k2 * d_c * 1.0)
    assert out[1] == pytest.approx(10.0 - k2 * d_c * 2.0)
    assert all(s == K2Source.LITERATURE_DEFAULT.value for s in src)


def test_apply_k2_missing_airmass_skips_row() -> None:
    mag = np.array([10.0, 10.0])
    am = np.array([1.0, float("nan")])
    out, _, src = apply_k2_per_frame(
        mag,
        am,
        object_bp_rp=1.0,
        bp_rp_comp_med=0.5,
        k2_value=-0.014,
        k2_source=K2Source.LITERATURE_DEFAULT,
    )
    assert out[0] != 10.0
    assert out[1] == 10.0
    assert src[0] == K2Source.LITERATURE_DEFAULT.value
    assert src[1] == K2Source.NONE.value


def test_filter_token_from_obs_group() -> None:
    assert filter_token_from_obs_group("g_60_2") == "g"
    assert filter_token_from_obs_group("R_20_2") == "R"
    assert filter_token_from_obs_group("CV_20_2") == "CV"


def test_sloan_r_literature_johnson_r_zero() -> None:
    assert computed_k2_bprp_for_token("r") == pytest.approx(-0.004 * SLOPE_GR_PER_BPRP, rel=1e-6)
    assert computed_k2_bprp_for_token("R") == 0.0
