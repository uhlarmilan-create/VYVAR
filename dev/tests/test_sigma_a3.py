"""Tests for sigma A3 ensemble SEM extraction and variant (e)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.chi2_sigma_gate import (
    dual_ensemble_sem_arrays,
    ensemble_sem_agreement_stats,
    ensemble_sem_from_lc,
    select_primary_ensemble_sem,
)
from sigma_budget import (
    SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE,
    combine_sigma_mag_quadrature,
)


def test_ensemble_sem_from_lc_clamp_fraction():
    err_rel = np.array([0.02, 0.015, 0.01], dtype=float)
    phot_mag = np.array([0.012, 0.012, 0.015], dtype=float)
    sem, clamp_frac = ensemble_sem_from_lc(err_rel, phot_mag)
    assert clamp_frac == pytest.approx(1 / 3)
    assert sem[0] > 0
    assert sem[1] > 0
    assert sem[2] == 0.0


def test_combine_sigma_mag_quadrature_with_ensemble():
    total = combine_sigma_mag_quadrature(
        0.01,
        0.005,
        sigma_floor_mag=0.003,
        ensemble_sem_mag=0.004,
    )
    expected = math.sqrt(0.01**2 + 0.005**2 + 0.003**2 + 0.004**2)
    assert total == pytest.approx(expected)


def test_ensemble_sem_agreement_on_synthetic():
    sem_a = np.array([0.01, 0.02, 0.015], dtype=float)
    sem_b = np.array([0.0105, 0.019, 0.016], dtype=float)
    stats = ensemble_sem_agreement_stats(sem_a, sem_b)
    assert stats["n_compared"] == 3
    assert stats["median_abs_diff"] == pytest.approx(0.001, abs=0.002)
    assert stats["p95_abs_diff"] is not None
    assert stats["p95_abs_diff"] < 0.005


def test_select_primary_ensemble_sem_prefers_production():
    sem_a = np.array([0.02, 0.03, np.nan], dtype=float)
    sem_b = np.array([0.011, np.nan, 0.02], dtype=float)
    out = select_primary_ensemble_sem(sem_a, sem_b)
    assert out[0] == pytest.approx(0.011)
    assert out[1] == pytest.approx(0.03)
    assert out[2] == pytest.approx(0.02)


def test_dual_ensemble_sem_arrays_synthetic():
    import pandas as pd

    err_rel = np.array([0.02, 0.015], dtype=float)
    phot_mag = np.array([0.01, 0.01], dtype=float)
    lc_df = pd.DataFrame({"err": err_rel})
    prod = np.array([0.011, 0.012], dtype=float)
    primary, sem_lc, sem_prod, clamp_frac, agreement = dual_ensemble_sem_arrays(
        lc_df, phot_mag, production_scatter=prod,
    )
    assert clamp_frac == 0.0
    assert agreement["n_compared"] == 2
    assert np.allclose(primary, prod)
    assert SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE.endswith("_ensemble")
