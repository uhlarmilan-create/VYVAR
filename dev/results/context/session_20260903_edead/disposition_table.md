# E-DEAD disposition table (27 names)

Measured on b44c82f, then applied. Ref classes: REAL / GLUE / TEST-ROOT /
TXT-DUMP / DOCSTRING / SUBSTRING.

| name | home on b44c82f | strongest ref | class | disposition | commit |
| --- | --- | --- | --- | --- | --- |
| `_median_bkg_var_adu2_per_px_from_proc_cache` | photometry_core | none outside the def | (none) | DELETE | 89a34c7 |
| `_star_mag_for_aperture_sizing` | photometry_core | none outside the def (SNR-table relic; `_APERTURE_SIZING_MAG_COLS` kept) | (none) | DELETE | 89a34c7 |
| `_is_broadband_photometric_filter` | photometry_core | `band_classify.py:250` comment "legacy heuristic" | DOCSTRING | DELETE | 89a34c7 |
| `get_auto_fov` | pipeline | none executable; prior audit listed TRULY-DEAD | (none) | DELETE | 980c57f |
| `_solve_wcs_solve_field_cli` | pipeline | `dev/scripts/vyvar_audit.txt` / `solver_audit.txt` | TXT-DUMP | DELETE | 980c57f |
| `_solve_wcs_astrometry_net` | pipeline | same TXT-DUMP pair (`_solve_wcs_external` is a different name and stays) | TXT-DUMP | DELETE | 980c57f |
| `_per_frame_noise_error_map` | pipeline | zero external refs | (none) | DELETE | 980c57f |
| `_frame_gain_readnoise_for_error_map` | pipeline | sole caller was `_per_frame_noise_error_map` (pipeline.py:173) | (transitive) | DELETE | 980c57f |
| `_saturated_core_plateau` | pipeline | `pipeline_catalog.py` docstring of `_saturated_core_plateau_vectorized` | DOCSTRING | DELETE | 980c57f |
| `_star_saturation_flags` | pipeline | `pipeline_catalog.py` docstring of `_vectorized_star_saturation_columns`; internal call to plateau | DOCSTRING | DELETE | 980c57f |
| `analyze_calibrated_qc` | pipeline | `AstroPipeline` in pipeline.py | REAL | STAY | (none) |
| `_analyze_calibrated_qc_one` | pipeline | sole caller `analyze_calibrated_qc` | REAL | STAY | (none) |
| `compute_per_frame_cog_correction` | photometry_core | `photometry_shared.enhance_catalog_dataframe_aperture_bpm` when `cog_params` set; also `measure_growth_curve_ee` | REAL | R-MOVE photometry_shared | 0b37a7a |
| `_aperture_flux_sky_batch` | photometry_core | `photometry_shared.stamp_masterstar_snr_columns` | REAL | R-MOVE photometry_shared | 0b37a7a |
| `_finite_pixel_bbox_from_array` | photometry_core | `photometry_shared.common_field_intersection_bbox_px*` | REAL | R-MOVE photometry_shared | 0b37a7a |
| `_intersection_bbox_from_frame_bboxes` | photometry_core | same bbox pair | REAL | R-MOVE photometry_shared | 0b37a7a |
| `_build_star_exclusion_mask` | photometry_core | `photometry_gate_helpers.measure_empty_aperture_sigma_bkg` / `estimate_star_free_per_pixel_variance_adu2` | REAL | R-MOVE photometry_gate_helpers | ce0eae2 |
| `_canonicalize_star_xy` | photometry_core | same empty-aperture path | REAL | R-MOVE photometry_gate_helpers | ce0eae2 |
| `_robust_scatter_mad` | photometry_core | same empty-aperture path (test_comp_pool_noise_s1 is a different name) | REAL | R-MOVE photometry_gate_helpers | ce0eae2 |
| `_clamp_err_empty_apertures_min` | photometry_core | `measure_empty_aperture_sigma_bkg` + `enhance_catalog_dataframe_aperture_bpm` | REAL | R-MOVE photometry_gate_helpers | ce0eae2 |
| `_labbe_append_debug_record` | photometry_core | `measure_empty_aperture_sigma_bkg` | REAL | R-MOVE photometry_gate_helpers | ce0eae2 |
| `_labbe_debug_dump_enabled` | photometry_core | only `_labbe_append_debug_record` | REAL (transitive) | R-MOVE photometry_gate_helpers | ce0eae2 |
| `_labbe_debug_dump_path` | photometry_core | only `_labbe_append_debug_record` | REAL (transitive) | R-MOVE photometry_gate_helpers | ce0eae2 |
| `_get_lc_star_method` | photometry_core | `photometry_exports._get_lc_adaptive_per_star` | REAL | R-MOVE photometry_exports | 89fc0b1 |
| `_quality_inspection_dao_metrics` | pipeline | `pipeline_ui_helpers.run_quality_analysis` | REAL | R-MOVE pipeline_ui_helpers | 8006c88 |
| `_estimate_fov_deg_from_fits_path` | pipeline | `pipeline_ui_helpers` | REAL | R-MOVE pipeline_ui_helpers | 8006c88 |
| `_obs_fwhm_basename_map_from_db` | pipeline | `pipeline_ui_helpers` | REAL | R-MOVE pipeline_ui_helpers | 8006c88 |

## Why deleted refs did not count

Per deleted def, strongest hit and why it is not a root:

- `_median_bkg_var_adu2_per_px_from_proc_cache`: inventory JSON / E0 lists only. Not a call.
- `_star_mag_for_aperture_sizing`: audit ledger + old diagnostic snippets (`mag3 = ...`) in docs/results, not an executable path. Live SNR modules were product-deleted at CONSOLIDATE-01.
- `_is_broadband_photometric_filter`: comment in `band_classify.py` (reworded on delete) + JOURNAL/ledger mentions. No call.
- `get_auto_fov`: prior dead-code audits; no caller in src_py or tests.
- `_solve_wcs_solve_field_cli` / `_solve_wcs_astrometry_net`: names appear in `dev/scripts/*.txt` captured solver/vyvar audit stdout. Those files are not executed. `_solve_wcs_external` in `pipeline_astrometry.py` is a different symbol.
- `_per_frame_noise_error_map`: no external caller. After it is gone, `_frame_gain_readnoise_for_error_map` has no remaining caller (fixed-point).
- `_saturated_core_plateau` / `_star_saturation_flags`: docstrings of the vectorized replacements (reworded on delete). Internal plateau<-flags call dies with the pair.

Header `from photometry_core import ...` lines in shared/gate_helpers are GLUE. Call sites inside those modules are REAL.
