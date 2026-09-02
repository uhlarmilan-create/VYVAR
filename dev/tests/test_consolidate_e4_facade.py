"""CONSOLIDATE-01E4: moved defs remain reachable through the facade."""

from __future__ import annotations

import photometry
import photometry_core
import photometry_lightcurve
import photometry_phase2a


PHOTOMETRY_E4_PHASE2A: tuple[str, ...] = (
    "run_phase2a",
    "measure_fwhm_from_masterstar",
    "read_flux_from_csv",
    "compute_aperture_correction",
    "_Phase2AState",
    "_ColorTermGroupFit",
    "parse_comp_quality_json_map",
    "democratic_detrend_lc",
    "detect_outliers",
    "apply_reporting_postprocess",
    "auto_export_variability_candidates_csv",
    "fit_color_term_c1",
    "should_apply_color_term",
    "resolve_apply_color_term",
    "save_field_map_png",
    "_photometric_error",
    "_sky_pp_for_photometric_error",
)


def test_e4_phase2a_facade_getattr() -> None:
    for name in PHOTOMETRY_E4_PHASE2A:
        obj = getattr(photometry_core, name)
        assert getattr(photometry_phase2a, name) is obj, name
        assert obj.__module__ == "photometry_phase2a", name


def test_e4_star_import_phase2a_names() -> None:
    for name in ("run_phase2a", "measure_fwhm_from_masterstar", "read_flux_from_csv"):
        assert hasattr(photometry, name), name
        assert getattr(photometry, name) is getattr(photometry_core, name)


def test_e4_normalize_gaia_id_still_on_phase2a() -> None:
    """V1 stub re-exported _normalize_gaia_id; ui_aperture_photometry imports it."""
    from photometry_shared import _normalize_gaia_id as shared_norm

    assert photometry_phase2a._normalize_gaia_id is shared_norm


def test_e4_c_d_full_entry_stays() -> None:
    assert photometry_core.run_full_photometry_pipeline.__module__ == "photometry_core"


PHOTOMETRY_E4_LC: tuple[str, ...] = (
    "ensemble_normalize",
    "compute_mag_calib_final",
    "save_lightcurve_csv",
    "apply_color_term",
    "_coerce_bool_cell",
    "BlendMapEntry",
    "_get_lc",
    "_route_lc_per_frame_err",
    "_recompute_bjd_hjd_with_status",
    "run_sysrem_field",
)


def test_e4_lightcurve_facade_getattr() -> None:
    for name in PHOTOMETRY_E4_LC:
        obj = getattr(photometry_core, name)
        assert getattr(photometry_lightcurve, name) is obj, name
        assert obj.__module__ == "photometry_lightcurve", name


def test_e4_state_facade() -> None:
    import phase2a_state

    obj = photometry_core._phase2a_prepare_shared_state
    assert obj.__module__ == "phase2a_state"
    assert obj is phase2a_state._phase2a_prepare_shared_state


def test_e4_target_facade() -> None:
    import phase2a_target

    obj = photometry_core._phase2a_process_one_target
    assert obj.__module__ == "phase2a_target"
    assert obj is phase2a_target._phase2a_process_one_target


def test_e4_spatial_grid_stayed_in_pipeline() -> None:
    import pipeline

    assert pipeline.select_comparison_stars_spatial_grid.__module__ == "pipeline"
