# -*- coding: ascii -*-
"""Tests for unit_resolver.py (Task D1)."""
from __future__ import annotations

from config import AppConfig
from unit_resolver import (
    blind_verify_match_tol_px,
    cog_ladder_step_px,
    hrd_color_bg_box_px,
    masterstar_centre_rms_max_px,
    masterstar_sibling_rms_max_px,
    phase01_chip_interior_margin_px,
    phase01_comparison_isolation_radius_px,
    reset_unit_resolver_logs,
    resolve_hfr_limit_px,
    resolve_max_dist_fallback_deg,
    resolve_px_from_arcsec,
    resolve_px_from_fwhm_factor,
    sips_dao_fwhm_px,
)


def test_none_uses_legacy_px_verbatim() -> None:
    reset_unit_resolver_logs()
    cfg = AppConfig()
    assert phase01_comparison_isolation_radius_px(cfg, arcsec_per_px=2.6) == 25.0
    assert blind_verify_match_tol_px(cfg, arcsec_per_px=1.3) == 2.5
    assert cog_ladder_step_px(cfg, fwhm_px=4.0) == 0.5
    assert sips_dao_fwhm_px(cfg, fwhm_px=3.0) == 2.5


def test_arcsec_conversion_at_known_scale() -> None:
    reset_unit_resolver_logs()
    cfg = AppConfig()
    cfg.phase01_comparison_isolation_radius_arcsec = 60.0
    # Newton bin2 ~1.30 arcsec/px -> 60/1.3 ~ 46.15 px
    got = phase01_comparison_isolation_radius_px(cfg, arcsec_per_px=1.30)
    assert abs(got - 60.0 / 1.30) < 1e-9


def test_fwhm_factor_conversion() -> None:
    reset_unit_resolver_logs()
    cfg = AppConfig()
    cfg.sips_dao_fwhm_fwhm_factor = 1.5
    assert sips_dao_fwhm_px(cfg, fwhm_px=4.0) == 6.0


def test_resolve_px_from_arcsec_helpers() -> None:
    assert resolve_px_from_arcsec(None, 25.0, 2.0, param_name="t") == 25.0
    assert resolve_px_from_arcsec(10.0, 25.0, 2.0, param_name="t") == 5.0


def test_resolve_px_from_fwhm_helpers() -> None:
    assert resolve_px_from_fwhm_factor(None, 2.5, 4.0, param_name="t") == 2.5
    assert resolve_px_from_fwhm_factor(1.25, 2.5, 4.0, param_name="t") == 5.0


def test_chip_margin_arcsec() -> None:
    reset_unit_resolver_logs()
    cfg = AppConfig()
    cfg.phase01_chip_interior_margin_arcsec = 13.0
    got = phase01_chip_interior_margin_px(cfg, arcsec_per_px=1.30)
    assert got == int(round(13.0 / 1.30))


def test_max_dist_fov_frac_fallback() -> None:
    reset_unit_resolver_logs()
    cfg = AppConfig()
    assert resolve_max_dist_fallback_deg(cfg, frame_w_px=1000, frame_h_px=1000, plate_scale_arcsec_px=None) == 1.5
    cfg.phase01_comparison_max_dist_fov_frac = 0.5
    # scale 1.0 arcsec/px, 1000x1000 -> diag deg from utils
    got = resolve_max_dist_fallback_deg(cfg, frame_w_px=1000, frame_h_px=1000, plate_scale_arcsec_px=1.0)
    assert got > 0.0
    assert got != 1.5


def test_hrd_color_bg_box_legacy_and_arcsec() -> None:
    reset_unit_resolver_logs()
    cfg = AppConfig()
    assert hrd_color_bg_box_px(cfg, arcsec_per_px=1.3) == 96
    cfg.hrd_color_bg_box_arcsec = 130.0
    assert hrd_color_bg_box_px(cfg, arcsec_per_px=1.3) == 100


def test_rms_gates_arcsec() -> None:
    reset_unit_resolver_logs()
    cfg = AppConfig()
    cfg.masterstar_centre_rms_max_arcsec = 1.56
    assert abs(masterstar_centre_rms_max_px(cfg, arcsec_per_px=1.3) - 1.2) < 1e-9
    cfg.masterstar_sibling_rms_max_arcsec = 2.6
    assert abs(masterstar_sibling_rms_max_px(cfg, arcsec_per_px=1.3) - 2.0) < 1e-9


def test_qc_hfr_fwhm_ratio() -> None:
    reset_unit_resolver_logs()
    cfg = AppConfig()
    assert resolve_hfr_limit_px(cfg, fwhm_px=4.0) == 5.0
    cfg.qc_max_hfr_fwhm_ratio = 1.25
    assert resolve_hfr_limit_px(cfg, fwhm_px=4.0) == 5.0
