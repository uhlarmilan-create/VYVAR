"""Tests for sigma A4 attribution regressors (synthetic injected signals)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.sigma_floor_attribution import (
    ols_multivariate,
    ols_through_origin,
    ols_with_intercept,
    phase_harmonic_design,
    residuals_from_delta_mag,
    variance_explained,
)


def test_residuals_from_delta_mag_zero_mean():
    m = np.array([1.0, 1.01, 0.99, 1.02], dtype=float)
    r = residuals_from_delta_mag(m)
    assert np.nanmean(r) == pytest.approx(0.0, abs=1e-12)


def test_k2_regressor_recovers_injected_slope():
    rng = np.random.default_rng(0)
    n = 200
    xdc = rng.uniform(0.0, 0.4, size=n)
    true_k2 = -0.035
    noise = rng.normal(0.0, 0.002, size=n)
    y = true_k2 * xdc + noise
    slope, yhat = ols_through_origin(y, xdc)
    assert slope == pytest.approx(true_k2, rel=0.15)
    assert variance_explained(y, yhat) > 0.5


def test_phase_regressor_recovers_injected_signal():
    rng = np.random.default_rng(1)
    n = 240
    fx = rng.uniform(0, 1, size=n)
    fy = rng.uniform(0, 1, size=n)
    X = phase_harmonic_design(fx, fy)
    beta_true = np.array([0.004, -0.003, 0.002, 0.001], dtype=float)
    y = X @ beta_true + rng.normal(0.0, 0.0015, size=n)
    beta_hat, yhat = ols_multivariate(y, X)
    assert beta_hat.shape == (4,)
    for b_true, b_est in zip(beta_true, beta_hat, strict=True):
        assert b_est == pytest.approx(b_true, abs=0.003)
    assert variance_explained(y, yhat) > 0.4


def test_x_linear_control_recovers_slope():
    rng = np.random.default_rng(2)
    x = rng.uniform(1.0, 2.0, size=120)
    true_slope = 0.008
    y = true_slope * (x - 1.5) + rng.normal(0, 0.002, size=120)
    _, slope, yhat = ols_with_intercept(y, x)
    assert slope == pytest.approx(true_slope, abs=0.004)
    assert variance_explained(y, yhat) > 0.2


def test_time_linear_control_recovers_drift():
    rng = np.random.default_rng(3)
    t = np.linspace(0, 1, 100)
    drift = 0.005
    y = drift * t + rng.normal(0, 0.001, size=100)
    _, slope, yhat = ols_with_intercept(y, t)
    assert slope == pytest.approx(drift, abs=0.003)
    assert variance_explained(y, yhat) > 0.3
