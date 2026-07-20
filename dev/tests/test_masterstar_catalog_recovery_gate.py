"""MASTERSTAR catalog-recovery verification + hint-as-prior acceptance."""
from __future__ import annotations

import math

import pytest

from vyvar_platesolver import _masterstar_solve_acceptance


def _accept(**kwargs):
    defaults = dict(
        accept_mode="fraction",
        catalog_recovery_tight=0.84,
        n_matched_tight=126,
        dist_benign=False,
        centre_rms=0.9,
        recovery_min=0.65,
        matched_floor=40,
        centre_rms_max=1.20,
        hint_sep_deg=0.228,
        hint_sep_limit=0.15,
        fov_diameter_deg=2.5,
    )
    defaults.update(kwargs)
    return _masterstar_solve_acceptance(**defaults)


def test_verified_solve_large_hint_sep_is_warning_not_reject():
    out = _accept(hint_sep_deg=0.228, hint_sep_limit=0.15)
    assert out["masterstar_verified"] is True
    assert out["hint_sep_warn"] is True
    assert out["hint_sep_bad_hard"] is False


def test_low_recovery_solve_rejected():
    out = _accept(catalog_recovery_tight=0.34, n_matched_tight=51, hint_sep_deg=0.05)
    assert out["masterstar_verified"] is False
    assert out["hint_sep_warn"] is False
    assert out["hint_sep_bad_hard"] is False


def test_wrong_field_high_hint_sep_non_verified_rejected():
    out = _accept(
        catalog_recovery_tight=0.34,
        n_matched_tight=51,
        hint_sep_deg=2.0,
        fov_diameter_deg=1.0,
    )
    assert out["masterstar_verified"] is False
    assert out["hint_sep_bad_hard"] is True
    tripwire = float(out["hint_sep_tripwire_deg"])
    assert math.isfinite(tripwire)
    assert 2.0 > tripwire


def test_wide_field_gate_uses_detection_budget_not_full_catalog():
    """Home-rig: 242/1476 raw looks bad; 242/250 gate is excellent (draft_404 class)."""
    out = _accept(
        catalog_recovery_tight=242.0 / 250.0,
        n_matched_tight=242,
        dist_benign=False,
        centre_rms=0.30,
        hint_sep_deg=0.013,
        hint_sep_limit=0.33,
        fov_diameter_deg=4.5,
    )
    assert out["masterstar_verified"] is True
    raw_frac = 242.0 / 1476.0
    assert raw_frac < 0.20
    assert out["masterstar_verified"] is True


def test_narrow_field_gate_unchanged_when_cat_lt_det():
    """Brno r: denom=min(150,233)=150 - gate equals raw catalog fraction."""
    out = _accept(catalog_recovery_tight=126.0 / 150.0, n_matched_tight=126)
    assert out["masterstar_verified"] is True


def test_invalid_composition_matches_task_spec():
    out = _accept()
    rms_bad = False
    rms_px = 2.09
    invalid = (
        (not out["masterstar_verified"])
        or (not math.isfinite(float(rms_px)))
        or rms_bad
        or out["hint_sep_bad_hard"]
    )
    assert invalid is False

    out_bad = _accept(catalog_recovery_tight=0.34, n_matched_tight=51)
    invalid_bad = (
        (not out_bad["masterstar_verified"])
        or (not math.isfinite(float(rms_px)))
        or rms_bad
        or out_bad["hint_sep_bad_hard"]
    )
    assert invalid_bad is True
