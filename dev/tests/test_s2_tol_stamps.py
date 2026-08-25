# -*- coding: ascii -*-
"""S2: pipeline_meta stamps effective DAO-Gaia tols vs config defaults."""

from __future__ import annotations

from config import AppConfig
from dao_gaia_calibration import effective_tol_stamps


def test_effective_tol_stamp_equals_override() -> None:
    cfg = AppConfig()
    derived = {
        "lock_pair_tol_px": 2.5,
        "pass2_center_tol_px": 2.5,
        "match_radius_px": 2.5,
        "lock_leftover_radius_px": 2.5,
    }
    out = effective_tol_stamps(
        derived, cfg, fwhm_px=5.19465, census_meta={"identity_fail_px": 15.58395}
    )
    assert out["lock_pair_tol_px"] == 2.5
    assert out["pass2_center_tol_px"] == 2.5
    assert out["match_radius_px"] == 2.5
    assert out["identity_fail_px"] == 15.58395
    assert out["lock_pair_tol_px_config_default"] == float(cfg.masterstar_lock_pair_tol_px)
    assert out["pass2_center_tol_px_config_default"] == float(cfg.masterstar_dao_pass2_center_tol_px)
    assert out["lock_pair_tol_px"] != out["lock_pair_tol_px_config_default"]


def test_effective_tol_stamp_falls_back_to_config() -> None:
    cfg = AppConfig()
    out = effective_tol_stamps(None, cfg, fwhm_px=4.0)
    assert out["lock_pair_tol_px"] == float(cfg.masterstar_lock_pair_tol_px)
    assert out["identity_fail_px"] == 12.0
