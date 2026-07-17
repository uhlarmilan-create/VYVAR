"""F-BJD-1 Stage D: time_base provenance on per-target BJD recompute."""

from __future__ import annotations

import numpy as np
import pytest

from config import AppConfig
from photometry_core import (
    TIME_BASE_BJD_TDB,
    TIME_BASE_JD_FALLBACK,
    _recompute_bjd_hjd_per_target,
    _recompute_bjd_hjd_with_status,
)


def _jirnt_cfg() -> AppConfig:
    c = AppConfig()
    c.observer_lat = 49.4041
    c.observer_lon = 16.8785
    c.observer_alt_m = 275.0
    return c


def test_time_base_bjd_tdb_success_path():
    cfg = _jirnt_cfg()
    jd = np.array([2461154.316555035], dtype=float)
    bjd, hjd, time_base = _recompute_bjd_hjd_with_status(
        jd,
        207.4983,
        39.4037,
        cfg,
    )
    assert time_base == TIME_BASE_BJD_TDB
    assert not np.array_equal(bjd, jd)
    assert not np.array_equal(hjd, jd)
    assert np.isfinite(bjd[0]) and np.isfinite(hjd[0])


def test_time_base_jd_fallback_invalid_coords():
    cfg = _jirnt_cfg()
    jd = np.array([2461154.31, 2461154.32], dtype=float)
    bjd, hjd, time_base = _recompute_bjd_hjd_with_status(
        jd,
        float("nan"),
        40.0,
        cfg,
    )
    assert time_base == TIME_BASE_JD_FALLBACK
    np.testing.assert_array_equal(bjd, jd)
    np.testing.assert_array_equal(hjd, jd)


def test_time_base_jd_fallback_observer_zero():
    cfg = AppConfig()
    cfg.observer_lat = 0.0
    cfg.observer_lon = 0.0
    jd = np.array([2461154.31, 2461154.32], dtype=float)
    bjd, hjd, time_base = _recompute_bjd_hjd_with_status(jd, 200.0, 40.0, cfg)
    assert time_base == TIME_BASE_JD_FALLBACK
    np.testing.assert_array_equal(bjd, jd)
    np.testing.assert_array_equal(hjd, jd)


def test_time_base_jd_fallback_astropy_failure(monkeypatch):
    cfg = _jirnt_cfg()
    jd = np.array([2461154.316555035], dtype=float)

    def _boom(*args, **kwargs):
        raise RuntimeError("forced astropy failure")

    monkeypatch.setattr("astropy.time.Time", _boom)
    bjd, hjd, time_base = _recompute_bjd_hjd_with_status(jd, 207.4983, 39.4037, cfg)
    assert time_base == TIME_BASE_JD_FALLBACK
    np.testing.assert_array_equal(bjd, jd)
    np.testing.assert_array_equal(hjd, jd)


def test_recompute_bjd_hjd_per_target_wrapper_unchanged():
    """Sandbox/test 2-tuple callers remain valid."""
    cfg = _jirnt_cfg()
    jd = np.array([2461154.316555035], dtype=float)
    bjd, hjd = _recompute_bjd_hjd_per_target(jd, 207.4983, 39.4037, cfg)
    assert bjd.shape == jd.shape
    assert hjd.shape == jd.shape
    assert not np.array_equal(bjd, jd)


def test_save_lightcurve_csv_writes_time_base(tmp_path):
    from photometry_core import save_lightcurve_csv

    n = 2
    bjd = np.array([2461154.1, 2461154.2], dtype=float)
    out = tmp_path / "lightcurve_test.csv"
    save_lightcurve_csv(
        out,
        bjd,
        bjd,
        bjd,
        np.full(n, 1.2),
        np.zeros(n, dtype=bool),
        np.full(n, 12.0),
        np.full(n, 12.0),
        np.full(n, 12.0),
        None,
        None,
        np.zeros(n),
        np.full(n, 0.01),
        np.full(n, 3.5),
        ["normal"] * n,
        ["f1.fits", "f2.fits"],
        time_base=TIME_BASE_JD_FALLBACK,
    )
    import pandas as pd

    df = pd.read_csv(out)
    assert "time_base" in df.columns
    assert (df["time_base"] == TIME_BASE_JD_FALLBACK).all()
