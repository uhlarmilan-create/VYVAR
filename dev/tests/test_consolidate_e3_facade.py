"""CONSOLIDATE-01E3: moved defs remain reachable through the facade."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import photometry
import photometry_comp
import photometry_core


PHOTOMETRY_E3_COMP: tuple[str, ...] = (
    "select_comparison_stars_per_target",
    "select_active_targets",
    "build_global_comp_pool",
    "_select_comps_by_rms_then_color",
    "_write_suspected_variables",
    "_enrich_comp_bp_rp",
    "_enrich_active_targets_bp_rp",
    "_select_comps_tiered",
    "_enrich_target_bp_rp_from_gaia_db",
    "_batch_enrich_targets_bp_rp_from_gaia_db",
    "_refresh_variable_targets_xy",
    "_read_field_density_inputs",
    "_resolve_frame_hw_px_from_masterstar",
    "ensure_full_variable_targets_if_presel_stub",
    "_auto_repair_catalog_ids",
    "_warn_zero_compstars_edge",
    "_count_gate_passing_comps",
    "_attach_predicted_dilution_report",
    "_phase0_effective_frame_hw_px",
    "_select_comps_by_color_then_rms",
    "_dedupe_comp_pool_by_gaia_key",
    "_bprp_tier_ladder_for_selection",
    "_variable_targets_looks_like_ct_presel_stub",
    "_active_target_zone_flag",
    "_ensure_active_target_display_names",
    "_normalize_id_value",
    "_sid_int",
    "_bool_col",
    "_normalize_id_series",
)


def test_e3_comp_facade_getattr() -> None:
    for name in PHOTOMETRY_E3_COMP:
        obj = getattr(photometry_core, name)
        assert callable(obj), name
        assert obj.__module__ == "photometry_comp", name


def test_e3_star_import_comp_all_names() -> None:
    for name in (
        "ensure_full_variable_targets_if_presel_stub",
        "select_active_targets",
        "select_comparison_stars_per_target",
        "run_phase0_and_phase1",
    ):
        assert hasattr(photometry, name), name
        assert getattr(photometry, name) is getattr(photometry_core, name)


def test_e3_normalize_id_value_not_alias() -> None:
    """E2 removed the dead alias; live def moved with this bucket."""
    assert photometry_core._normalize_id_value is photometry_comp._normalize_id_value
    assert photometry_core._normalize_id_value.__name__ == "_normalize_id_value"
    assert photometry_core._normalize_id_value("1234.0") == "1234"


def test_e3_ensure_full_vt_smoke_tmp_draft(tmp_path: Path) -> None:
    """Non-stub VT must no-op; path resolution unchanged (pure move)."""
    vt = tmp_path / "variable_targets.csv"
    ms = tmp_path / "masterstars_full_match.csv"
    fits = tmp_path / "MASTERSTAR.fits"
    pd.DataFrame({"catalog_id": ["1"], "name": ["a"]}).to_csv(vt, index=False)
    pd.DataFrame({"catalog_id": ["1"], "name": ["a"]}).to_csv(ms, index=False)
    fits.write_bytes(b"")
    assert (
        photometry_core.ensure_full_variable_targets_if_presel_stub(
            variable_targets_csv=vt,
            masterstars_csv=ms,
            masterstar_fits=fits,
        )
        is False
    )
    assert vt.read_text(encoding="utf-8").startswith("catalog_id")


def test_e3_spatial_grid_stays_in_pipeline() -> None:
    """E3: not in photometry_comp. E6a C2 moved it to pipeline_astrometry."""
    import pipeline

    assert pipeline.select_comparison_stars_spatial_grid.__module__ == "pipeline_astrometry"


def test_e3_phase01_run_facade() -> None:
    import phase01_run

    assert photometry_core.run_phase0_and_phase1.__module__ == "phase01_run"
    assert photometry_core.run_phase0_and_phase1 is phase01_run.run_phase0_and_phase1
    assert photometry.run_phase0_and_phase1 is photometry_core.run_phase0_and_phase1
