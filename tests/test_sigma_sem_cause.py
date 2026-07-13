"""Tests for SIGMA-SEM-CAUSE pure helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.sem_cause_core import (
    chi2_dof_from_mags_sigmas,
    lag1_autocorrelation,
    per_frame_sem_from_residuals,
    split_half_zp_sem,
    trend_fraction,
)


def test_lag1_white_noise_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    assert abs(lag1_autocorrelation(x)) < 0.15


def test_lag1_linear_trend_positive():
    x = np.linspace(0, 1, 50) + 0.01 * np.sin(np.linspace(0, 3, 50))
    assert lag1_autocorrelation(x) > 0.9


def test_per_frame_sem_matches_production_formula():
    residuals = [[0.01, -0.01, 0.02], [0.005, -0.005]]
    sem = per_frame_sem_from_residuals(residuals)
    arr = np.asarray([0.01, -0.01, 0.02], dtype=float)
    expected = float(np.std(arr, ddof=1) / math.sqrt(3))
    assert sem[0] == pytest.approx(expected)


def test_trend_fraction_high_on_ramp():
    x = np.linspace(1, 2, 30)
    y = 2.0 * x + 0.01 * np.random.default_rng(1).normal(size=30)
    frac, _, _ = trend_fraction(y, x, deg=1)
    assert frac > 0.95


def test_split_half_scales_with_n():
    mags = {f"c{i}": 10.0 + 0.01 * i for i in range(8)}
    sem, scale = split_half_zp_sem(mags, n_splits=30, seed=2)
    assert math.isfinite(sem)
    assert scale == pytest.approx(math.sqrt(8 / 4))


def test_chi2_hand_constant():
    mags = np.array([1.0, 1.01, 0.99])
    sig = np.full(3, 0.01)
    _, _, c2 = chi2_dof_from_mags_sigmas(mags, sig)
    resid = mags - np.mean(mags)
    hand = float(np.sum((resid / 0.01) ** 2) / 2)
    assert c2 == pytest.approx(hand)
