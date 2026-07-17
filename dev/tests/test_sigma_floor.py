"""Tests for PROD-SIGMA-FLOOR: c4, err combine, floor resolution."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mag_constants import MAG_ERR_SCALE
from sigma_floor_core import (
    c4_small_sample,
    combine_production_err_mag,
    combine_production_err_rel,
    ensemble_sem_mag_from_residuals,
    resolve_sigma_sys_mag,
)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (2, 0.7979),
        (3, 0.8862),
        (4, 0.9213),
        (5, 0.9400),
        (6, 0.9515),
        (7, 0.9594),
        (8, 0.9650),
    ],
)
def test_c4_small_sample_literature_constants(n: int, expected: float) -> None:
    assert abs(c4_small_sample(n) - expected) < 1e-4


def test_c4_small_sample_n_lt_2_is_nan() -> None:
    assert math.isnan(c4_small_sample(1))


def test_ensemble_sem_c4_increases_vs_uncorrected() -> None:
    resid = [0.01, -0.02, 0.015, -0.005]
    n = len(resid)
    std = float(np.std(resid, ddof=1))
    uncorrected = std / math.sqrt(n)
    corrected = ensemble_sem_mag_from_residuals(resid)
    assert corrected > uncorrected


def test_combine_production_err_rel_domain() -> None:
    phot = 0.01
    sem_mag = 0.005
    floor_mag = 0.003
    rel = combine_production_err_rel(phot, sem_mag, sigma_sys_mag=floor_mag)
    sem_rel = sem_mag / MAG_ERR_SCALE
    floor_rel = floor_mag / MAG_ERR_SCALE
    expected = math.sqrt(phot * phot + sem_rel * sem_rel + floor_rel * floor_rel)
    assert abs(rel - expected) < 1e-12


def test_combine_production_err_mag_roundtrip() -> None:
    phot = 0.02
    sem = 0.004
    floor = 0.006
    mag = combine_production_err_mag(phot, sem, sigma_sys_mag=floor)
    rel = combine_production_err_rel(phot, sem, sigma_sys_mag=floor)
    assert abs(mag - rel * MAG_ERR_SCALE) < 1e-12


def test_resolve_sigma_sys_default_zero() -> None:
    class _Cfg:
        sigma_sys_mag = {}

    assert resolve_sigma_sys_mag(99, _Cfg()) == 0.0


def test_resolve_sigma_sys_from_config() -> None:
    class _Cfg:
        sigma_sys_mag = {"4": 0.005}

    assert resolve_sigma_sys_mag(4, _Cfg()) == pytest.approx(0.005)


def test_save_lightcurve_sigma_sys_mag_column(tmp_path: Path) -> None:
    from photometry_core import save_lightcurve_csv

    n = 3
    bjd = np.arange(n, dtype=float)
    mag = np.full(n, 12.0)
    save_lightcurve_csv(
        tmp_path / "lc.csv",
        bjd,
        bjd,
        bjd,
        np.ones(n),
        None,
        mag,
        mag,
        mag,
        mag,
        mag,
        np.zeros(n),
        np.full(n, 0.01),
        np.full(n, 3.0),
        ["normal"] * n,
        ["a.csv"] * n,
        sigma_sys_mag=0.018,
    )
    df = pd.read_csv(tmp_path / "lc.csv")
    assert "sigma_sys_mag" in df.columns
    assert float(df["sigma_sys_mag"].iloc[0]) == 0.018


def test_floor_add_after_sem_in_quadrature() -> None:
    from photometry_core import _combine_err_with_ensemble_scatter_keyed

    err = np.array([0.01, 0.012])
    scatter = {"a.csv": 0.004, "b.csv": 0.003}
    out, unmatched = _combine_err_with_ensemble_scatter_keyed(
        err, ["a.csv", "b.csv"], scatter, sigma_sys_mag=0.005,
    )
    assert not unmatched.any()
    for i, sf in enumerate(["a.csv", "b.csv"]):
        expected = combine_production_err_rel(
            float(err[i]), scatter[sf], sigma_sys_mag=0.005,
        )
        assert out[i] == pytest.approx(expected, rel=1e-9)
