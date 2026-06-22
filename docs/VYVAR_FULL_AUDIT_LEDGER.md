# VYVAR Full Audit Ledger

**Status:** AUDIT-ONLY (read + log; no code edits during audit pass).
**Started:** 2026-06-19
**Last checkpoint:** Group 7 complete — UI shell (2026-06-19 local) — **7-group map complete**

Method: automated AST inventory + lens scans (L1-L11) on 9 modules (415 functions), targeted verification against `docs/VYVAR_CODE_AUDIT.md` DR2-DR6 threads, science-critical logic reads.

**Checkpoint policy:** Milan + Claude review this batch before Group 3 (Data / IO / catalog).

### Fix log (post-audit)

| Step | Finding | Status | Commit |
|------|---------|--------|--------|
| 1 | G1-F007 sibling crowded-field `p_false` gate | **FIXED** (geometric-evidence branch + adversarial tests) | `5225c27` |
| 2 | Group 1 repro: astroalign RANSAC unseeded | **FIXED** (`seeded_numpy_default_rng` + determinism tests) | `98de910` |
| 3 | G2-F001 catalog_only silent drop in Phase 2A | **SUPERSEDED** (2026-06-22): forced-aperture / catalog_only path removed; VSX without DAO+Gaia match excluded upstream in `select_active_targets` | `7f0dc86` |
| 4 | G2-F002 catalog_only placement rejects sibling WCS | **SUPERSEDED** (2026-06-22): catalog_only placement removed with forced-aperture path | `7f0dc86` |
| — | **Forced-aperture removal — validated do-no-harm** | **DONE** (2026-06-22): strip-FORCED gate vs draft 419 — `mag_inst` 360/360 B; `mag_calib` 357/360 B (one +~30 mmag uniform zeropoint accepted); R XY wrong-star closed. Variable direct-hit-only rule. Backlog: upfront exclusion of intermittently-DAO comps at selection. | `7f0dc86` |
| 6 | G5-F001 / G4-F002 / G2-F005 / G5-F005 forced-aperture ledger hygiene | **RESOLVED/SUPERSEDED** (`7f0dc86`): no forced-aperture / catalog_only LCs; `export_reports.py` has no `lc_source`; PDF glossary forced-aperture prose obsolete | `7f0dc86` |
| 7 | G5-F002 AC mag with uncorrected `err` | **RESOLVED (non-issue):** AC is constant `delta_m_corr`; `err` invariant; no `err_ac`; folding `ac_scatter` per-point would misrepresent correlated systematic as random | — |
| 8 | G5-F007 hardcoded 1.3 arcsec/px + `#SOFTWARE=VYVAR/1.0` | **FIXED:** derive-or-None plate scale from `pipeline_meta` / MASTERSTAR WCS; `VYVAR_SOFTWARE_VERSION` + `_aavso_software_header_line` | `6774f83` |
| 9 | G5-F003 candidate LC PNG uses `mag_inst` | **FIXED:** `_resolve_candidate_lc_mag_for_plot` mirrors export AC precedence; calibrated y-label | `76c5a93` |
| 10 | G5-F011 split CT/AC consumers (export vs PDF) | **FIXED:** canonical `mag_calib_final` = `mag_calib` + CT + AC; export + all publication figures; G5-F003 precedence subsumed | `be3e193` |
| 11 | G5-F006 PDF time axis BJD vs BJD(TDB) | **FIXED:** `_pdf_time_axis_label`; PDF LC/glossary use BJD(TDB) for bjd/bjd_tdb columns | `b74c301` |
| 12 | G5-F008 VarAstro `n_good_comp` vs trust `n_clean` | **FIXED (distinct metrics):** VarAstro header `n_ensemble_comp` label + glossary/calibration doc; no number reconciliation | `07e6f69` |
| 13 | G5-F004 silent export failures | **FIXED:** `record_export_failure` + `log_export_batch_summary`; ERROR logging; Phase 2A batch collector | `efbb4de` |
| 14 | G7-F001 / G7-F002 unwired Select Stars page | **RESOLVED:** delete `ui_select_stars.py` (phantom `max_bv_diff` / stale kwarg) | `3e1cad7` |
| 15 | G6-F002 master validity_days mismatch | **FIXED:** unified default dark **90** / flat **200** (dataclass, `__post_init__`, DB seed, `config.json`); DB SETTINGS vestigial | `379e78f` |
| 16 | G3-F002 query_local_gaia mag_limit None | **FIXED (Path A):** None ⇒ no cap; MASTER_SOURCES `mag_limit=None`; stale 11.5 default removed | `fb75867` |
| 17 | G1-F001 / G1-F002 alignment_max_control_points | **FIXED:** decouple astroalign CP from detection ladder; `alignment_max_control_points=80`; Chi/h draft 419 `B_20_2` validation PASS — max Δtranslation **0.0275 px**, speedup **5.2–13.8×** (mean **8.6×**); not byte-identical, quality-validated | `2819e86` |
| 18 | validate_alignment_control_points draft layouts | **DONE:** `non_calibrated` / `processed` / `calibrated` / `detrended_aligned` lights + default `platesolve/MASTERSTAR` ref | `65a608b` |
| 19 | G2-F003 dilution aperture 3.0 fallback | **FIXED:** layered fallback map → SNR-derive → skip+flag; fixed 3.0 removed; Seager 2003 / Howell 2006 photometric aperture | `0e50805` |
| 5 | G3-F001 calibration master silent mismatch | **FIXED** (scoped-only match; dark temp-required; flat no-exptime; registration/fallback parity) | `b4a45fb` |

---

## Prioritized findings (Group 1 - deduplicated, severity-sorted)

| ID | Sev | Lens | Location | What's wrong | Principle (not fix) |
|----|-----|------|----------|--------------|---------------------|
| G1-F001 | **FIXED** | L4/L5 | `config.py`, `pipeline.py`, `vyvar_alignment_frame.py` | ~~`max_control_points` dead knob; ladder used `min(max_st, n_fit)`~~ **FIXED:** `alignment_max_control_points` is a real, read config (default 80); dead hardcoded 180 removed; plumbed via `_align_ctx`; ladder uses plumbed cap; detection ladder unchanged. Chi/h draft 419 `B_20_2` validated (not byte-identical). | Caller-facing knobs must bind to the live code path; silent dead params mislead tuning and perf work. |
| G1-F002 | **FIXED** | L6 | `config.py`, `pipeline.py`, `vyvar_alignment_frame.py` | ~~Dense field: ~200 CP → ~654 s/frame~~ **FIXED:** control points decoupled from detection ladder, capped at cfg default 80. Chi/h 419 `B_20_2` PASS: max Δtranslation **0.0275 px**, speedup **5.2–13.8×** (mean **8.6×**); quality-validated, not byte-identical. | Iteration/control-point budgets need explicit wall-clock + count bounds on dense fields. |
| G1-F003 | **MED** | L4 | `vyvar_alignment_frame.py:684-709` | Alignment failure -> **identity fallback** (`VY_ALGN=False` but frame still written). Misaligned pixels can enter photometry unless residual gate catches them. | Science path should reject or hard-flag failed alignment; identity is not a valid aligned product. |
| G1-F004 | **MED** | L1 | `vyvar_blind_series.py:212-216` | Blind verify tolerance scaled vs fixed `_ref = 1.3`/px (narrow-rig reference), not measured scale. | Scale-dependent tolerances must derive from equipment/WCS scale, not a universal literal. |
| G1-F005 | **MED** | L3 | `optics_selection.py:50-64` | `_first_db_optics_ids`: broad `except: pass` on DB queries - failed lookup -> `None` with no log. | DB failures on optics resolution must log; silent pass risks wrong rig. |
| G1-F006 | **MED** | L4 | `vyvar_platesolver.py:2356-2440` | Odds accept gate (`masterstar_accept_mode=odds`) ignores legacy fraction/distortion when mode=odds - by design per DECISIONS. Thresholds from `config.py` (`masterstar_odds_k=12`, `matched_floor`, `false_alarm_p_max=1e-6`, quadrants>=3). | Document mode split; ensure UI exposes accept_mode if operators need fraction gate. |
| G1-F007 | **MED** | L10 | `vyvar_platesolver.py:5792-5893` | Sibling-WCS recovery uses separate odds gate (`masterstar_sibling_min_matched=40`, `rms_max_px=2.0`). Tested in `tests/test_sibling_wcs_recovery.py`. | Multi-filter recovery gates must stay aligned with independent-solve gates - monitor on rig x filter matrix. |
| G1-F008 | **LOW** | L5 | `pipeline.py` (11 funcs) | Dead code confirmed by AST ref-count: `_fits_header_positive_float`, `_per_frame_noise_error_map`, `get_auto_fov`, `_try_solve_wcs_astrometry_net_or_local_cli`, etc. | Remove after final API surface check (DB/UI dispatch). |
| G1-F009 | **LOW** | L5 | `vyvar_blind_solver.py:767` | `_cluster_centroid_votes` - zero callers (superseded by DBSCAN path). | Delete dead cluster path or wire explicitly. |
| G1-F010 | **LOW** | B/L5 | `pipeline.py:35-36` | Stale F401 imports `_astrometry_align_mp_init/_task` - A-durable uses module attr at `12938`. | Hygiene: drop unused imports. |
| G1-F011 | **LOW** | L5 | `optics_autodetect.py:50` | `Detection.autofill` property never referenced; `band()` / `ok` used instead. | Remove or use consistently in UI autofill logic. |
| G1-P001 | **CLEAN** | - | `pipeline.py:10972-10999` | Plate-scale fallback fixed: derive-or-None after DB/FITS/WCS exhaustion (no `9.77` universal fallback). | Correct pattern for cross-rig scale handling. |
| G1-P002 | **CLEAN** | - | `optics_selection.py` | Explicit rejection of silent `equipment_id=1`; draft optics override with logging. | Correct optics authority model. |
| G1-P003 | **CLEAN** | - | `platesolve_ui_paths.py` | Pure path resolution; no science math; legacy + per-setup bundle discovery orderly. | Keep as single UI path helper. |

---

## Coverage table (Group 1 modules)

| Module | Lines | Funcs | Audited | DEAD (heuristic) | TRULY-DEAD | LIVE-DYNAMIC† | TEST-ONLY† |
|--------|-------|-------|---------|------------------|------------|---------------|------------|
| `pipeline.py` | 17012 | 231 | 231 | 11 | 11 | 0 | 0 |
| `vyvar_platesolver.py` | 6405 | 91 | 91 | 0 | 0 | 0 | 0 |
| `vyvar_blind_solver.py` | 1482 | 33 | 33 | 1 | 1 | 0 | 0 |
| `vyvar_blind_series.py` | 367 | 9 | 9 | 0 | 0 | 0 | 0 |
| `vyvar_alignment_frame.py` | 755 | 10 | 10 | 0 | 0 | 0 | 0 |
| `astrometry_optimizer.py` | 1164 | 10 | 10 | 0 | 0 | 0 | 0 |
| `optics_autodetect.py` | 374 | 16 | 16 | 1 | 0 | 1 | 0 |
| `optics_selection.py` | 234 | 7 | 7 | 0 | 0 | 0 | 0 |
| `platesolve_ui_paths.py` | 119 | 8 | 8 | 0 | 0 | 0 | 0 |
| **Group 1 total** | - | **415** | **415** | **13** | **12** | **1** | **0** |

† Reclassified 2026-06-20 (`tmp/reclassify_g1_g2_dead.py`). FLAGGED 14 + NEEDS-TEST 33 unchanged.

---

## Per-module function registry (Group 1)

Schema: line | qualname | signature | status where status is AUDITED-CLEAN / FLAGGED(sev) / DEAD / NEEDS-TEST.

Flagged regions (MED): pipeline `generate_masterstar`, `detect_stars_and_match_catalog`, `_astrometry_align_impl_body`; platesolver acceptance + `solve_wcs_with_local_gaia`; alignment worker; blind series/solver hot paths; optimizer grip.

### `pipeline.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 113 | `_photometry_mode_run_flags` | `cfg, platesolve_dir` | AUDITED-CLEAN |
| 135 | `_apply_aperture_catalog_enhancements_from_st` | `df, data, hdr, st` | AUDITED-CLEAN |
| 207 | `_fits_header_positive_float` | `hdr, *keys` | DEAD |
| 215 | `_frame_gain_readnoise_for_error_map` | `hdr, db, equipment_id` | AUDITED-CLEAN |
| 235 | `_per_frame_noise_error_map` | `data, hdr, db, equipment_id` | DEAD |
| 246 | `_export_catalog_psf_st_fields` | `cfg, platesolve_dir` | AUDITED-CLEAN |
| 261 | `_epsf_target_catalog_ids` | `platesolve_dir, top_comps` | AUDITED-CLEAN |
| 332 | `_add_catalog_ids_from_csv` | `ids, comp_p` | AUDITED-CLEAN |
| 351 | `_epsf_lc_catalog_ids` | `platesolve_dir` | AUDITED-CLEAN |
| 386 | `_epsf_fit_catalog_ids` | `platesolve_dir, psf_photometry_enabled` | AUDITED-CLEAN |
| 397 | `_fill_psf_catalog_columns` | `df, data, hdr, st, target_ids` | AUDITED-CLEAN |
| 632 | `_vyvar_calibrate_multiprocessing_enabled` | `` | AUDITED-CLEAN |
| 638 | `_cfg_calibration_library_native_binning` | `cfg` | AUDITED-CLEAN |
| 649 | `_dark_np_for_calibration_path` | `dark_cache, master_binning, p, light_binning` | AUDITED-CLEAN |
| 675 | `_log_calibration_io_preflight` | `calibrated_root, master_dark_path, masterflat_by_filter` | AUDITED-CLEAN |
| 717 | `_pipeline_ui_info` | `msg` | AUDITED-CLEAN |
| 735 | `_pipeline_ui_error` | `msg` | AUDITED-CLEAN |
| 752 | `_ensure_parent_dirs_for_aligned_fits` | `out_path` | AUDITED-CLEAN |
| 757 | `_assert_alignment_produced_fits` | `aligned_root` | AUDITED-CLEAN |
| 767 | `_match_and_crop_pair` | `a, b` | AUDITED-CLEAN |
| 780 | `_safe_filter_token` | `text` | AUDITED-CLEAN |
| 787 | `observation_group_key_from_metadata` | `meta` | AUDITED-CLEAN |
| 798 | `_iter_light_fits` | `lights_root` | AUDITED-CLEAN |
| 803 | `_summarize_lights_binning_from_headers` | `paths` | AUDITED-CLEAN |
| 820 | `log_lights_binning_from_headers_preflight` | `paths, context` | AUDITED-CLEAN |
| 849 | `_pick_light_for_metadata_diagnostic` | `paths` | AUDITED-CLEAN |
| 862 | `_filter_light_paths_maybe` | `files, only_paths` | AUDITED-CLEAN |
| 906 | `_archive_raw_to_calibrated_light` | `archive, raw_fp` | AUDITED-CLEAN |
| 934 | `_resolve_light_fits_for_quality_inspection` | `archive, raw_fp` | AUDITED-CLEAN |
| 949 | `_resolve_draft_light_raw_path` | `archive, file_path` | AUDITED-CLEAN |
| 966 | `_skip_processed_directory` | `app_config` | AUDITED-CLEAN |
| 971 | `_get_vy_qc_status` | `fits_path` | AUDITED-CLEAN |
| 980 | `find_qc_metrics_csv` | `archive_path, app_config, draft_id, db` | AUDITED-CLEAN |
| 1004 | `_archive_preprocess_lights_root` | `ap, app_config, draft_id, db` | AUDITED-CLEAN |
| 1027 | `draft_obs_group_count` | `archive_path` | AUDITED-CLEAN |
| 1042 | `draft_is_multi_group_obs` | `archive_path` | AUDITED-CLEAN |
| 1046 | `resolve_masterstar_input_root` | `archive_path, setup_name, app_config, draft_id, db` | AUDITED-CLEAN |
| 1106 | `_inspection_jd_from_header` | `hdr` | AUDITED-CLEAN |
| 1148 | `_exposure_sec_from_header` | `hdr` | AUDITED-CLEAN |
| 1163 | `_quality_inspection_dao_metrics` | `fp` | AUDITED-CLEAN |
| 1185 | `_dao_star_table_mean_roundness` | `tbl` | AUDITED-CLEAN |
| 1205 | `_dao_star_table_mean_elongation` | `tbl` | AUDITED-CLEAN |
| 1230 | `_quality_inspection_dao_metrics_array` | `data, hdr` | AUDITED-CLEAN |
| 1343 | `draft_median_pointing_icrs_deg` | `db, draft_id` | AUDITED-CLEAN |
| 1368 | `resolve_preprocess_target_coordinates` | `db, draft_id, ui_ra_deg, ui_dec_deg` | AUDITED-CLEAN |
| 1425 | `_estimate_fov_deg_from_header` | `hdr` | AUDITED-CLEAN |
| 1455 | `_estimate_fov_deg_from_fits_path` | `fp` | AUDITED-CLEAN |
| 1466 | `_icrs_offset_arcmin` | `ra_deg, de_deg, ref_ra_deg, ref_de_deg` | DEAD |
| 1480 | `sync_obs_files_drift_arcmin_for_draft` | `db, draft_id, ref_ra_deg, ref_de_deg` | AUDITED-CLEAN |
| 1542 | `generate_observation_hash` | `db, draft_id` | AUDITED-CLEAN |
| 1589 | `run_quality_analysis` | `db, draft_id, archive_path, progress_cb, roundness_reject_above` | AUDITED-CLEAN |
| 1728 | `_perf10_lookup_qc` | `perf10_qc_results, archive, file_path` | AUDITED-CLEAN |
| 1744 | `apply_perf10_dao_qc_to_obs_files` | `db, draft_id, archive_path, perf10_qc_results, roundness_reject_above` | AUDITED-CLEAN |
| 1850 | `run_draft_ram_calibration_qc_to_obs_files` | `db, draft_id, archive_path, master_dark_path, masterflat_by_filter, ma...` | AUDITED-CLEAN |
| 2087 | `calibrated_paths_for_draft_apply_filters` | `archive_path, db, draft_id, fwhm_max_px, drift_max_arcmin, source_dir` | AUDITED-CLEAN |
| 2257 | `format_memory_bytes` | `n` | AUDITED-CLEAN |
| 2272 | `_as_fits_float32_image` | `data` | AUDITED-CLEAN |
| 2279 | `_fits_primary_pixel_count` | `header` | AUDITED-CLEAN |
| 2296 | `_available_system_ram_bytes` | `` | AUDITED-CLEAN |
| 2306 | `estimate_memory_from_fits_headers` | `paths, sample_headers` | AUDITED-CLEAN |
| 2340 | `estimate_archive_memory_profile` | `archive_path` | AUDITED-CLEAN |
| 2426 | `_cupy_available` | `` | AUDITED-CLEAN |
| 2436 | `_path_segments_forbidden_for_masterstar_physical_source` | `p, pre_calibrated` | AUDITED-CLEAN |
| 2454 | `_path_is_under_tree` | `root, p` | AUDITED-CLEAN |
| 2462 | `_pick_preferred_masterstar_basename_hit` | `hits, pre_calibrated` | AUDITED-CLEAN |
| 2482 | `_header_vy_fwhm_px` | `hdr` | AUDITED-CLEAN |
| 2495 | `_obs_fwhm_basename_map_from_db` | `db, draft_id` | AUDITED-CLEAN |
| 2518 | `_sort_masterstar_paths_by_fwhm` | `files, fwhm_by_basename` | AUDITED-CLEAN |
| 2548 | `_strip_external_platesolve_header` | `hdr` | AUDITED-CLEAN |
| 2570 | `build_masterstar_from_detrended` | `detrended_root, output_fits, only_paths, fwhm_fallback_px, app_config,...` | AUDITED-CLEAN |
| 2782 | `_update_masterstar_obs_file_status` | `cfg, draft_id, selected_ref_path, wcs_ok, n_stars` | AUDITED-CLEAN |
| 2838 | `_resolve_best_effort_path_under` | `root, raw_path, pre_calibrated` | AUDITED-CLEAN |
| 2908 | `resolve_obs_file_to_processed_fits` | `archive_path, obs_file_path, setup_name, app_config, draft_id, db` | AUDITED-CLEAN |
| 2951 | `list_best_processed_light_paths_for_masterstar` | `archive_path, setup_name, draft_id, app_config, take_n` | AUDITED-CLEAN |
| 3004 | `get_masterstar_candidate_rows` | `draft_id, percentage, fwhm_max_px, db` | AUDITED-CLEAN |
| 3087 | `get_masterstar_candidates` | `draft_id, percentage, db` | AUDITED-CLEAN |
| 3095 | `_vyvar_open_database` | `cfg` | AUDITED-CLEAN |
| 3102 | `_fits_pixel_raw_to_micrometres` | `value` | AUDITED-CLEAN |
| 3114 | `_header_focal_length_mm` | `header` | AUDITED-CLEAN |
| 3131 | `resolve_plate_solve_fov_deg_hint` | `hdr, h, w, database_path, equipment_id, draft_id` | AUDITED-CLEAN |
| 3185 | `get_auto_fov` | `archive_path, masterstar_path, database_path, equipment_id, draft_id` | DEAD |
| 3252 | `_focal_mm_plausible` | `mm` | AUDITED-CLEAN |
| 3256 | `_resolve_focal_mm_for_plate_scale` | `header, db, equipment_id` | AUDITED-CLEAN |
| 3302 | `_merge_equipment_pixel_into_metadata` | `meta, db, equipment_id` | AUDITED-CLEAN |
| 3344 | `_recompute_effective_pixel_from_physical` | `meta` | AUDITED-CLEAN |
| 3363 | `_header_pick_first` | `header, *keys, default` | AUDITED-CLEAN |
| 3370 | `_enrich_calibration_metadata_from_header` | `meta, header, db, id_equipment` | AUDITED-CLEAN |
| 3429 | `_apply_draft_combined_to_pipeline_meta` | `meta, comb` | AUDITED-CLEAN |
| 3476 | `_log_calibration_metadata_diagnostic` | `filename, metadata` | AUDITED-CLEAN |
| 3494 | `_plate_solve_input_bundle` | `fits_path, app_config, equipment_id, draft_id` | AUDITED-CLEAN |
| 3585 | `compute_plate_scale_from_db` | `equipment_id, telescope_id, db_conn, binning` | AUDITED-CLEAN |
| 3624 | `_try_rescale_masterstar_linear_wcs_to_expected_plate_scale` | `fits_path, app_config, equipment_id, draft_id` | AUDITED-CLEAN |
| 3725 | `_solve_wcs_solve_field_cli` | `masterstar_path, expected_arcsec_per_pixel` | AUDITED-CLEAN |
| 3821 | `_try_solve_wcs_astrometry_net_or_local_cli` | `masterstar_path, api_key, expected_arcsec_per_pixel` | DEAD |
| 3845 | `_solve_wcs_astrometry_net` | `masterstar_path, api_key, expected_arcsec_per_pixel` | AUDITED-CLEAN |
| 3906 | `_apply_wcs_header_to_fits` | `fits_path, wcs_hdr` | AUDITED-CLEAN |
| 3923 | `_solve_wcs_external` | `fits_path, backend, astrometry_api_key, plate_solve_fov_deg, hint_ra_d...` | AUDITED-CLEAN |
| 4206 | `_has_valid_wcs` | `header` | AUDITED-CLEAN |
| 4210 | `_wcs_astrometry_nearly_identical` | `wa, wb, rtol` | DEAD |
| 4228 | `_bin2d_mean` | `arr, factor` | AUDITED-CLEAN |
| 4244 | `_dao_star_count_from_array` | `arr, fwhm_px` | AUDITED-CLEAN |
| 4275 | `_pick_reference_frame_by_star_count` | `files` | AUDITED-CLEAN |
| 4303 | `_wcs_field_center_radec_deg` | `fits_path` | AUDITED-CLEAN |
| 4324 | `_catalog_df_cap_brightest_by_mag` | `df, max_rows` | AUDITED-CLEAN |
| 4344 | `_query_gaia_local` | `center, radius_deg, gaia_db_path, max_mag, focal_mm_for_log, max_rows` | AUDITED-CLEAN |
| 4443 | `_query_vsx_local` | `center, radius_deg, vsx_db_path, max_rows` | AUDITED-CLEAN |
| 4512 | `_query_vsx_local_frame_bbox` | `wcs, width_px, height_px, vsx_db_path, margin_px, center` | AUDITED-CLEAN |
| 4587 | `_saturate_limit_adu_from_header` | `hdr` | AUDITED-CLEAN |
| 4603 | `_infer_sat_limit_from_bitpix` | `hdr` | AUDITED-CLEAN |
| 4625 | `_equipment_saturate_adu_from_db` | `equipment_id` | AUDITED-CLEAN |
| 4643 | `_effective_saturation_limit` | `hdr, fallback_adu, equipment_saturate_adu` | AUDITED-CLEAN |
| 4685 | `_box_peak_max_adu` | `data, x, y, half` | AUDITED-CLEAN |
| 4702 | `_box_peaks_at_centroids` | `arr, x, y, half` | AUDITED-CLEAN |
| 4743 | `_icrs_deg_to_unitxyz` | `ra_deg, dec_deg` | AUDITED-CLEAN |
| 4753 | `_chord_to_arcsec` | `dist_chord` | AUDITED-CLEAN |
| 4762 | `build_ucac_catalog_kdtree` | `cat_df` | AUDITED-CLEAN |
| 4785 | `nearest_sky_nn_kdtree` | `tree, det_ra_deg, det_dec_deg` | AUDITED-CLEAN |
| 4811 | `_saturated_core_plateau` | `data, x, y, half_inner, plateau_rel, min_plateau_pixels` | AUDITED-CLEAN |
| 4849 | `_star_saturation_flags` | `arr, x, y, sat_limit, sat_frac, peak_dao_val, peak_max_adu` | AUDITED-CLEAN |
| 4887 | `_all_pix2world_icrs_deg` | `wcs_obj, x, y` | AUDITED-CLEAN |
| 4902 | `_saturated_core_plateau_vectorized` | `data, x, y, half_inner, plateau_rel, min_plateau_pixels` | AUDITED-CLEAN |
| 4935 | `_vectorized_star_saturation_columns` | `arr, x, y, sat_limit, sat_frac, peak_dao, peak_max_adu` | AUDITED-CLEAN |
| 4987 | `_proc_sat_block_for_csv` | `sat_block` | AUDITED-CLEAN |
| 5004 | `_vyvar_df_round_time_jd_for_csv` | `df` | AUDITED-CLEAN |
| 5015 | `_vyvar_df_to_csv` | `df, path` | AUDITED-CLEAN |
| 5033 | `_fits_header_first_positive_float` | `hdr, keys` | AUDITED-CLEAN |
| 5046 | `_gaia_catalog_cone_radius_optics_floor_deg` | `hdr, naxis1, naxis2, plate_solve_fov_fallback_deg` | AUDITED-CLEAN |
| 5087 | `_field_center_and_radius_from_wcs` | `w, h, wpx` | AUDITED-CLEAN |
| 5128 | `_effective_field_catalog_cone_radius_deg` | `w, h, wpx, plate_solve_fov_deg, fits_header` | AUDITED-CLEAN |
| 5170 | `_invalidate_field_catalog_cone_cache_if_needed` | `field_catalog_csv, plate_solve_fov_deg, effective_radius_deg` | AUDITED-CLEAN |
| 5216 | `_field_catalog_cone_meta_path` | `field_catalog_csv` | AUDITED-CLEAN |
| 5220 | `_write_field_catalog_cone_meta` | `field_catalog_csv, center, radius_deg, naxis1, naxis2, plate_solve_fov...` | AUDITED-CLEAN |
| 5245 | `select_comparison_stars_spatial_grid` | `df, width_px, height_px, n_comp, require_catalog_match, require_photom...` | AUDITED-CLEAN |
| 5406 | `_annotate_masterstars_flux_zones` | `df, noise_floor_adu, equipment_saturate_adu, saturate_limit_adu_fallba...` | AUDITED-CLEAN |
| 5531 | `write_photometry_plan_files` | `platesolve_dir, masterstar_fits, masterstars_csv, n_comparison_stars, ...` | AUDITED-CLEAN |
| 6174 | `_sync_comparison_stars_across_setups` | `platesolve_root` | AUDITED-CLEAN |
| 6225 | `_dao_auto_binning_factor` | `h, w` | AUDITED-CLEAN |
| 6233 | `_mean_bin2d_for_dao` | `data0, factor` | AUDITED-CLEAN |
| 6248 | `_dao_xy_binned_to_full` | `x, y, f` | AUDITED-CLEAN |
| 6258 | `_dao_full_to_binned_xy` | `x_full, y_full, bfac` | AUDITED-CLEAN |
| 6267 | `_catalog_match_radius_px` | `wcs_obj, match_sep_arcsec, wpx, h` | AUDITED-CLEAN |
| 6289 | `_dao_pass2_annulus_stats` | `data0, cx, cy` | AUDITED-CLEAN |
| 6310 | `_merge_dao_pass1_pass2_tables` | `tbl_pass1, pass2_rows, bfac, dedup_px` | AUDITED-CLEAN |
| 6354 | `_dao_targeted_pass2_unmatched_gaia` | `data0, tbl_pass1, cat_df, wcs_obj, bfac, fwhm_px, pass2_sigma, match_s...` | AUDITED-CLEAN |
| 6484 | `_inject_forced_aperture_rows` | `df_detected, master_df, wcs_obj, wpx, h, match_sep_arcsec` | AUDITED-CLEAN |
| 6607 | `_proc_rename_det_names_to_catalog_id` | `df` | AUDITED-CLEAN |
| 6623 | `_proc_drop_unmatched_dao_rows` | `df` | AUDITED-CLEAN |
| 6638 | `_proc_catalog_keep_matched_rows_only` | `df` | AUDITED-CLEAN |
| 6656 | `_prefilter_dao_table_brightest` | `tbl, keep_top` | AUDITED-CLEAN |
| 6668 | `_dao_spatial_flux_cap_row_indices` | `tbl, max_n, width_px, height_px` | AUDITED-CLEAN |
| 6721 | `detect_stars_match_master_reference` | `data, hdr, master_df, max_catalog_rows, match_sep_arcsec, saturate_lev...` | AUDITED-CLEAN |
| 7266 | `_merge_platesolve_gaia_pairs_into_masterstars_df` | `df, pairs_x, pairs_y, pairs_ra, pairs_de, pairs_catalog_id, max_pair_p...` | AUDITED-CLEAN |
| 7312 | `detect_stars_and_match_catalog` | `data, hdr, max_catalog_rows, cat_df, vsx_df, gaia_variable_df, match_s...` | FLAGGED(MED) |
| 8303 | `_vyvar_parallel_use_processes` | `` | AUDITED-CLEAN |
| 8310 | `_vyvar_parallel_pool` | `max_workers` | AUDITED-CLEAN |
| 8320 | `_icrs_center_radius_from_hdr_data` | `hdr, data, plate_solve_fov_deg` | AUDITED-CLEAN |
| 8347 | `_export_first_icrs_center_radius` | `files, plate_solve_fov_deg` | AUDITED-CLEAN |
| 8376 | `_prefetch_export_shared_catalog_for_process_pool` | `files, reference_hdr_data, field_cat_path, cat_df, vsx_df, gaia_variab...` | AUDITED-CLEAN |
| 8468 | `_init_export_per_frame_worker` | `state` | AUDITED-CLEAN |
| 8491 | `_airmass_from_altitude_deg` | `alt_deg` | AUDITED-CLEAN |
| 8499 | `_compute_airmass_from_altaz` | `hdr, cfg, db, draft_id` | AUDITED-CLEAN |
| 8567 | `_extract_airmass_from_header` | `hdr, cfg, db, draft_id` | AUDITED-CLEAN |
| 8610 | `_cfg_from_export_worker_state` | `st` | AUDITED-CLEAN |
| 8619 | `_export_per_frame_run_catalog_core` | `base_path, hdr, data, st` | AUDITED-CLEAN |
| 8955 | `_export_per_frame_disk_worker_task` | `fp_str` | AUDITED-CLEAN |
| 8976 | `_export_per_frame_ram_worker_task` | `packed` | AUDITED-CLEAN |
| 9010 | `export_per_frame_catalogs` | `frames_root, platesolve_dir, max_catalog_rows, catalog_match_max_sep_a...` | FLAGGED(MED) |
| 9910 | `validate_comparison_ensemble_flatness` | `frames_root, comparison_stars_csv, flux_col, name_col, max_relative_rm...` | AUDITED-CLEAN |
| 10054 | `_apply_wcs_tan_fragment_to_header` | `h, wh, history_note` | AUDITED-CLEAN |
| 10068 | `_gaia_sky_match_wcs_fragment` | `hdr, data, app_config, dao_threshold_sigma, dao_fwhm_px, plate_solve_f...` | DEAD |
| 10101 | `_refine_masterstar_wcs_gaia_sky_match_infile` | `fits_path, app_config, equipment_id, dao_threshold_sigma, dao_fwhm_px,...` | DEAD |
| 10128 | `_fill_masterstars_gaia_matched_bp_rp_from_local_db` | `df, gaia_db_path` | AUDITED-CLEAN |
| 10180 | `generate_masterstar_and_catalog` | `archive_path, max_catalog_rows, astrometry_api_key, source_root, plate...` | FLAGGED(MED) |
| 12094 | `_pass2_sibling_wcs_recovery` | `reports, skipped, job_list, align_kw` | AUDITED-CLEAN |
| 12269 | `_partition_detrended_by_subfolder` | `files, detrended_root` | AUDITED-CLEAN |
| 12291 | `_merge_astrometry_group_reports` | `rows` | AUDITED-CLEAN |
| 12318 | `_astrometry_align_impl_body` | `job, archive_path, astrometry_api_key, max_control_points, min_detecte...` | FLAGGED(MED) |
| 13292 | `astrometry_align_and_build_masterstar` | `archive_path, astrometry_api_key, max_control_points, min_detected_sta...` | AUDITED-CLEAN |
| 13506 | `_fits_meta_ra_deg` | `value` | AUDITED-CLEAN |
| 13511 | `_fits_meta_dec_deg` | `value` | AUDITED-CLEAN |
| 13516 | `_db_for_calibration_tasks` | `qc_opt` | AUDITED-CLEAN |
| 13532 | `_qc_pack_from_config` | `cfg, draft_id, observation_id` | AUDITED-CLEAN |
| 13570 | `_qc_center_crop_for_stars` | `data, max_side` | AUDITED-CLEAN |
| 13587 | `_half_flux_radius_in_cutout` | `cut, xc, yc` | AUDITED-CLEAN |
| 13607 | `_mean_hfr_bright_stars_dao` | `crop, max_stars, dao_detection_sigma` | AUDITED-CLEAN |
| 13670 | `_post_calibration_qc_eval` | `data, limits, light_basename` | AUDITED-CLEAN |
| 13745 | `_strip_raw_linearity_header_keywords` | `hdr` | AUDITED-CLEAN |
| 13767 | `_vy_calib_status_numeric` | `flags` | AUDITED-CLEAN |
| 13777 | `_hdr_vy_cflag_str` | `hdr` | AUDITED-CLEAN |
| 13786 | `_calibration_flags` | `used_dark, used_flat, passthrough, flat_skipped_no_dark` | AUDITED-CLEAN |
| 13806 | `_calibration_type_from_flags` | `flags` | AUDITED-CLEAN |
| 13821 | `_calibrate_one_light_apply_masters_in_ram` | `src, master_dark_path, masterflat_by_filter, flat_norm_floor, flat_cac...` | AUDITED-CLEAN |
| 14016 | `_calibrate_one_light_disk` | `src, dst, master_dark_path, masterflat_by_filter, flat_norm_floor, fla...` | AUDITED-CLEAN |
| 14126 | `_init_calibrate_batch_worker` | `initargs` | AUDITED-CLEAN |
| 14143 | `_calibrate_batch_process_one` | `item` | AUDITED-CLEAN |
| 14208 | `_has_usable_master_dark` | `path` | AUDITED-CLEAN |
| 14212 | `_has_any_usable_master_flat` | `masterflat_by_filter` | DEAD |
| 14221 | `_passthrough_lights_to_calibrated` | `lights_root, calibrated_root, progress_cb, database_path, draft_id, ob...` | AUDITED-CLEAN |
| 14312 | `calibrate_lights_to_calibrated` | `lights_root, calibrated_root, master_dark_path, masterflat_by_filter, ...` | AUDITED-CLEAN |
| 14705 | `_safe_proc_name` | `original_name` | AUDITED-CLEAN |
| 14715 | `_estimate_dao_fwhm_guess` | `img2, std` | AUDITED-CLEAN |
| 14753 | `_qc_fwhm_elongation` | `data, max_sources` | AUDITED-CLEAN |
| 15085 | `_vyvar_parallel_worker_count` | `app_config` | AUDITED-CLEAN |
| 15124 | `_vyvar_qc_preprocess_workers` | `` | AUDITED-CLEAN |
| 15129 | `_estimate_catalog_frame_hw` | `work_ram, files` | AUDITED-CLEAN |
| 15153 | `_vyvar_cap_mp_workers_for_catalog` | `n_workers, frame_hw, reserve_gb` | AUDITED-CLEAN |
| 15176 | `_vyvar_per_frame_csv_workers` | `app_config` | AUDITED-CLEAN |
| 15185 | `_analyze_calibrated_qc_one` | `src` | AUDITED-CLEAN |
| 15212 | `_preprocess_calibrated_one` | `src, calibrated_root, processed_root, reject_fwhm_px, reject_elongatio...` | AUDITED-CLEAN |
| 15300 | `_qc_enrich_calibrated_in_place` | `calibrated_root, app_config, fwhm_reject_limit, elong_reject_limit, ta...` | AUDITED-CLEAN |
| 15436 | `preprocess_calibrated_to_processed` | `calibrated_root, processed_root, reject_fwhm_px, reject_elongation, te...` | AUDITED-CLEAN |
| 15631 | `analyze_calibrated_qc` | `calibrated_root, max_frames, progress_cb, only_paths` | AUDITED-CLEAN |
| 15693 | `_qc_suggest_thresholds` | `df` | AUDITED-CLEAN |
| 15742 | `_apply_temporal_sigma_clip_in_place` | `processed_root, produced_files, sigma, min_frames, tile, use_gpu_if_av...` | AUDITED-CLEAN |
| 15851 | `scan_calibrated_lights_pointing` | `calibrated_root, max_files` | AUDITED-CLEAN |
| 16019 | `_parse_fits_binning_int` | `raw, default` | AUDITED-CLEAN |
| 16025 | `_log_effective_pixel_pitch` | `meta, filepath` | AUDITED-CLEAN |
| 16043 | `fits_metadata_from_primary_header` | `header, force_physical_pixel_um` | AUDITED-CLEAN |
| 16175 | `extract_fits_metadata` | `filepath, db, app_config, force_physical_pixel_um, id_equipment, draft...` | AUDITED-CLEAN |
| 16277 | `scan_usb_folder` | `path` | AUDITED-CLEAN |
| 16392 | `AstroPipeline.__init__` | `self, config` | AUDITED-CLEAN |
| 16396 | `AstroPipeline.calibrate` | `self, session_path` | AUDITED-CLEAN |
| 16405 | `AstroPipeline.quick_calibrate_last_import` | `self, archive_path, master_dark_path, masterflat_by_filter, progress_c...` | AUDITED-CLEAN |
| 16487 | `AstroPipeline.calibrate_batch` | `self, light_paths, lights_root, calibrated_root, master_dark_path, mas...` | AUDITED-CLEAN |
| 16678 | `AstroPipeline.quick_preprocess_last_import` | `self, archive_path, run, reject_fwhm_px, reject_elongation, temporal_s...` | AUDITED-CLEAN |
| 16759 | `AstroPipeline.quick_analyze_last_import` | `self, archive_path, max_frames` | DEAD |
| 16791 | `AstroPipeline._first_fits_file` | `session_path` | AUDITED-CLEAN |
| 16802 | `AstroPipeline.prepare_observation_from_session` | `self, session_path, id_equipment, id_telescope, id_location` | AUDITED-CLEAN |
| 16840 | `AstroPipeline.create_observation_from_payload` | `self, payload` | DEAD |
| 16845 | `_field_jump_empty_result` | `` | AUDITED-CLEAN |
| 16859 | `detect_field_jumps` | `db, draft_id, jump_threshold_arcmin, min_frames_in_group` | AUDITED-CLEAN |

### `vyvar_platesolver.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 67 | `_apply_proper_motion` | `ra, dec, pmra, pmdec, obs_year` | AUDITED-CLEAN |
| 102 | `_obs_year_from_header` | `header` | AUDITED-CLEAN |
| 117 | `_apply_pm_to_gaia_rows` | `rows, obs_year` | AUDITED-CLEAN |
| 146 | `_get_masterstar_wcs_parity` | `masterstar_fits_path` | AUDITED-CLEAN |
| 206 | `_fits_header_pick` | `header, *keys` | AUDITED-CLEAN |
| 213 | `_fits_header_parse_ra_deg` | `value` | AUDITED-CLEAN |
| 241 | `_fits_header_parse_dec_deg` | `value` | AUDITED-CLEAN |
| 263 | `pointing_hint_from_header` | `header` | AUDITED-CLEAN |
| 313 | `parse_user_ra_string_to_deg` | `text` | AUDITED-CLEAN |
| 366 | `parse_user_dec_string_to_deg` | `text` | AUDITED-CLEAN |
| 418 | `resolve_pointing_for_vyvar` | `header, user_ra_text, user_dec_text` | AUDITED-CLEAN |
| 444 | `_triangle_sorted_sides_pixel` | `xa, ya, xb, yb, xc, yc` | AUDITED-CLEAN |
| 452 | `_ratios` | `s1, s2, s3` | AUDITED-CLEAN |
| 458 | `_scale_consistent` | `s_img, s_arc, rtol` | AUDITED-CLEAN |
| 464 | `_empirical_median_plate_scale_arcsec_per_px` | `xs, ys, ra_deg, de_deg, max_stars, max_pairs` | AUDITED-CLEAN |
| 501 | `_triangle_angles_sorted_from_sides` | `s1, s2, s3` | AUDITED-CLEAN |
| 525 | `_linear_tan_predict_pixels` | `wcs_obj, ra_deg, dec_deg` | AUDITED-CLEAN |
| 532 | `_wcs_pixel_rms_linear` | `wcs_obj, x_obs, y_obs, world` | AUDITED-CLEAN |
| 537 | `_wcs_pixel_rms_full` | `wcs_obj, x_obs, y_obs, world` | AUDITED-CLEAN |
| 542 | `_filter_catalog_to_fov` | `df, naxis1, naxis2` | AUDITED-CLEAN |
| 552 | `_sip_uv_term_indices` | `max_order, min_total_degree` | AUDITED-CLEAN |
| 564 | `_sip_fill_ab` | `coefx, coefy, idxs, max_order` | AUDITED-CLEAN |
| 574 | `_adaptive_ridge` | `n_matches, sip_order` | AUDITED-CLEAN |
| 587 | `_fit_sip_on_matches` | `w_lin, x_obs, y_obs, world, max_order, ridge, force_apply, sip_force_r...` | AUDITED-CLEAN |
| 741 | `_fit_sip_on_matches_masterstar_try_orders` | `w_lin, x_obs, y_obs, world, sip_max_order, sip_min_order, force_apply,...` | AUDITED-CLEAN |
| 784 | `_fit_sip_for_solver` | `is_masterstar, w_lin, x_obs, y_obs, world, sip_max_order, sip_min_orde...` | AUDITED-CLEAN |
| 818 | `_ransac_fit_wcs_tan` | `x, y, world, rng, n_iter, min_sample, inlier_thresh_px` | AUDITED-CLEAN |
| 859 | `_radec_to_unit_xyz` | `ra_deg, dec_deg` | AUDITED-CLEAN |
| 866 | `_radec_array_to_unit_xyz` | `ra_deg, dec_deg` | AUDITED-CLEAN |
| 884 | `_VerifyGaiaBrightCatalog.load` | `cls, db_path, mag_limit` | AUDITED-CLEAN |
| 904 | `_VerifyGaiaBrightCatalog.cone_indices` | `self, ra0, dec0, cone_r_deg, max_rows, use_box` | AUDITED-CLEAN |
| 940 | `_VerifyGaiaBrightCatalog.cone_arrays` | `self, ra0, dec0, cone_r_deg, max_rows` | AUDITED-CLEAN |
| 954 | `_catalog_pixel_kdtree` | `wcs_for_pred, ra_cat_deg, dec_cat_deg` | AUDITED-CLEAN |
| 976 | `_blind_verify_prefilter_pass` | `wcs_for_pred, ra_cat_deg, dec_cat_deg, xs, ys, max_px, min_count, n_br...` | AUDITED-CLEAN |
| 1011 | `_greedy_match_pairs_pixel_wcs` | `wcs_for_pred, ra_cat_deg, dec_cat_deg, xs, ys, max_px, cat_pred_xy, ca...` | AUDITED-CLEAN |
| 1092 | `_greedy_pixel_nn_one_to_one` | `xs, ys, cat_x, cat_y, ra_cat, dec_cat, max_px, order_idx` | AUDITED-CLEAN |
| 1162 | `_refine_wcs_tan_nn_gaia` | `wcs_in, xs_det, ys_det, ra_cat_full_deg, dec_cat_full_deg, max_match_p...` | AUDITED-CLEAN |
| 1224 | `_sky_sep_arcsec` | `ra1, dec1, ra2, dec2` | AUDITED-CLEAN |
| 1230 | `_img_triangle_cyclic_sides_arcsec` | `xs, ys, plate_scale_arcsec_per_px, x_cen, y_cen, use_gnomonic` | AUDITED-CLEAN |
| 1254 | `_triangle_perm_side_rms` | `perm, cat_ra, cat_dec, img_sides` | AUDITED-CLEAN |
| 1273 | `_best_perm_for_triangle` | `xs, ys, cat_ra, cat_dec, plate_scale_arcsec_per_px, x_cen, y_cen, use_...` | AUDITED-CLEAN |
| 1325 | `_paired_triangle_vertices` | `xs, ys, cat_ra, cat_dec, plate_scale_arcsec_per_px, x_cen, y_cen, use_...` | AUDITED-CLEAN |
| 1359 | `_pool_cluster_correspondences` | `members, plate_scale_arcsec_per_px, x_cen, y_cen, use_gnomonic, naxis1...` | AUDITED-CLEAN |
| 1412 | `_count_wcs_inliers` | `wcs, xs, ys, ra, dec, tol_px` | AUDITED-CLEAN |
| 1426 | `_fit_cluster_ransac_wcs` | `xs, ys, ra, dec, naxis1, naxis2, tol_px, n_iter, rng` | AUDITED-CLEAN |
| 1476 | `_wcs_scale_gate` | `wcs, known_ps, scale_tol` | AUDITED-CLEAN |
| 1490 | `_triangle_wcs_from_candidate` | `xs, ys, cat_ra, cat_dec, naxis1, naxis2, plate_scale_arcsec_per_px, x_...` | AUDITED-CLEAN |
| 1524 | `_best_triangle_wcs_in_cluster` | `members, naxis1, naxis2, plate_scale_arcsec_per_px, x_cen, y_cen, use_...` | AUDITED-CLEAN |
| 1589 | `_cluster_wcs_seed` | `members, naxis1, naxis2, plate_scale_arcsec_per_px, x_cen, y_cen, use_...` | AUDITED-CLEAN |
| 1648 | `_verify_blind_candidates` | `candidates, dao_df, gaia_db_path, fov_deg, naxis1, naxis2, pixel_pitch...` | AUDITED-CLEAN |
| 2142 | `_log_wcs_orientation_header_hints` | `wcs_obj, hdr` | AUDITED-CLEAN |
| 2186 | `_mirror_detections_xy` | `xs, ys, naxis1, naxis2, flip_x, flip_y` | AUDITED-CLEAN |
| 2204 | `_fits_roworder_yflip_applied` | `hdr` | AUDITED-CLEAN |
| 2209 | `_apply_fits_roworder_to_detections` | `xs, ys, hdr, naxis2` | AUDITED-CLEAN |
| 2225 | `_sip_match_max_px` | `max_px_coarse` | AUDITED-CLEAN |
| 2230 | `_compute_masterstar_catalog_recovery` | `wcs, cat_ra, cat_de, xs_det, ys_det, naxis1, naxis2, qa_px, tight_px` | AUDITED-CLEAN |
| 2320 | `_masterstar_quality_flags` | `catalog_recovery_tight_gate, recovery_min, n_cat_in_frame, centre_rms,...` | AUDITED-CLEAN |
| 2356 | `_masterstar_solve_acceptance` | `accept_mode, catalog_recovery_tight, catalog_recovery_tight_gate, n_ma...` | FLAGGED(MED) |
| 2443 | `_assess_masterstar_distortion_limited_linear` | `wcs, px, py, pra, pde, naxis1, naxis2, benign_ratio_max` | AUDITED-CLEAN |
| 2502 | `_greedy_match_pairs_for_sip` | `wcs, ra_all, de_all, xs, ys, max_px_coarse` | AUDITED-CLEAN |
| 2529 | `_fit_linear_wcs_from_pairs` | `px, py, pra, pde, ransac_refinement, ransac_min_pairs, rng_seed` | AUDITED-CLEAN |
| 2546 | `_refit_linear_and_sip_on_full_pairs` | `w_lin, ra_all, de_all, xs, ys, max_px_coarse, enable_sip, sip_max_orde...` | AUDITED-CLEAN |
| 2613 | `_gaia_triangle_greedy_orientation_probe` | `cat_df_in, xs, ys, naxis1, naxis2, w, h, simple_mode, exp_scale, silen...` | AUDITED-CLEAN |
| 2837 | `_fits_header_strip_sip` | `hdr` | AUDITED-CLEAN |
| 2856 | `_wcs_linear_without_sip` | `wcs_in` | AUDITED-CLEAN |
| 2871 | `_equalize_wcs_cd_axes_to_target_arcsec` | `wcs_lin, target_arcsec_per_px` | AUDITED-CLEAN |
| 2924 | `_maybe_repair_masterstar_anisotropic_plate_scale` | `wcs_in, target_arcsec_per_px, pairs_x, pairs_y, pairs_ra, pairs_de, en...` | AUDITED-CLEAN |
| 3018 | `_SolveWcsCatalogError.__init__` | `self, reason` | AUDITED-CLEAN |
| 3023 | `_solve_wcs_build_catalog` | `pointing_ra, pointing_dec, fov_diameter_deg_eff, exp_scale, chip_fw, c...` | AUDITED-CLEAN |
| 3370 | `_SolveWcsValidationError.__init__` | `self, result` | AUDITED-CLEAN |
| 3375 | `_solve_wcs_validate_and_refine` | `wcs_final, pairs_final, cat_df, cat_df_assoc, xs_native, ys_native, hd...` | FLAGGED(MED) |
| 3960 | `_SolveWcsWriteError.__init__` | `self, reason` | AUDITED-CLEAN |
| 3965 | `_solve_wcs_write_results` | `fp, hdr0, wcs_final, sip_meta, pairs_final, match_rate, rms_px, dao_fw...` | AUDITED-CLEAN |
| 4208 | `_try_blind_series_hint` | `data, hdr0, plate_scale_arcsec_per_px, fov_deg, max_cat_mag, app_confi...` | AUDITED-CLEAN |
| 4255 | `solve_wcs_with_local_gaia` | `fits_path, hint_ra_deg, hint_dec_deg, fov_diameter_deg, gaia_db_path, ...` | FLAGGED(MED) |
| 5716 | `filter_code_from_setup_name` | `setup` | AUDITED-CLEAN |
| 5724 | `_sibling_cfg_thresholds` | `cfg` | AUDITED-CLEAN |
| 5754 | `_sibling_quadrant_count` | `xs, ys, naxis1, naxis2` | AUDITED-CLEAN |
| 5767 | `_sibling_false_alarm_p` | `n_matched, n_det, n_cat, naxis1, naxis2, r_px` | AUDITED-CLEAN |
| 5791 | `_sibling_odds_confirmed` | `metrics, min_matched, rms_max_px, min_quadrants` | FLAGGED(MED) |
| 5812 | `_sibling_match_metrics` | `wcs_use, ra_cat, de_cat, xs, ys, naxis1, naxis2, thresholds, cat_pred_...` | FLAGGED(MED) |
| 5896 | `_sibling_match_offset_median` | `wcs, ra_cat, de_cat, xs, ys, cat_pred_flip, naxis1, naxis2` | FLAGGED(MED) |
| 5941 | `_sibling_apply_bulk_shift_crpix` | `wcs, dx, dy, sx, sy` | AUDITED-CLEAN |
| 5950 | `_sibling_best_bulk_shift` | `w_adopt, ra_cat, de_cat, xs, ys, naxis1, naxis2, thresholds, cat_pred_...` | AUDITED-CLEAN |
| 6030 | `_sibling_adopt_and_confirm` | `donor_wcs, ra_cat, de_cat, xs, ys, naxis1, naxis2, thresholds, cat_pre...` | AUDITED-CLEAN |
| 6074 | `pick_sibling_donor_filter` | `recipient_filter, verified_filters` | AUDITED-CLEAN |
| 6095 | `_sibling_detect_dao_on_image` | `data, hdr, dao_sigma` | AUDITED-CLEAN |
| 6144 | `_sibling_median_stack_into_fits` | `recipient_path, frame_paths, n_stack` | AUDITED-CLEAN |
| 6167 | `_sibling_load_gaia_catalog` | `wcs_ref, hdr, naxis1, naxis2, gaia_db_path, fov_diameter_deg, expected...` | AUDITED-CLEAN |
| 6222 | `_write_sibling_recovered_wcs_to_fits` | `recipient_path, wcs_final, donor_filter, bulk_shift, metrics, stack_n` | AUDITED-CLEAN |
| 6257 | `try_recover_masterstar_sibling_wcs` | `recipient_masterstar_fits, donor_masterstar_fits, recipient_filter, do...` | AUDITED-CLEAN |

### `vyvar_blind_solver.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 36 | `_blind_img_star_budget` | `app_config, n_top` | AUDITED-CLEAN |
| 47 | `_index_k_neighbors` | `idx` | AUDITED-CLEAN |
| 55 | `_blind_img_select_mode` | `app_config` | AUDITED-CLEAN |
| 62 | `_index_per_cell_meta` | `idx` | AUDITED-CLEAN |
| 76 | `_dao_flux_column` | `dao_stars` | AUDITED-CLEAN |
| 84 | `cap_brightest_per_pixel_cell` | `dao_stars, cell_px, stars_per_cell` | AUDITED-CLEAN |
| 112 | `_select_blind_image_stars` | `dao_stars, idx, plate_scale_arcsec_per_px, app_config, n_top, fov_deg,...` | AUDITED-CLEAN |
| 191 | `_rig_prior_enabled` | `cfg` | AUDITED-CLEAN |
| 195 | `_scale_tol_frac` | `cfg` | AUDITED-CLEAN |
| 203 | `_use_gnomonic_triangles` | `fov_deg, use_rig_prior` | AUDITED-CLEAN |
| 213 | `_side_arcsec_flat` | `p0, p1, plate_scale_arcsec_per_px` | AUDITED-CLEAN |
| 219 | `_side_arcsec_gnomonic` | `p0, p1, x_cen, y_cen, plate_scale_arcsec_per_px` | AUDITED-CLEAN |
| 235 | `_triangle_sides_arcsec` | `p0, p1, p2, x_cen, y_cen, plate_scale_arcsec_per_px, use_gnomonic` | AUDITED-CLEAN |
| 255 | `_catalog_l3_arcsec_from_tree` | `match_idx, hash_tree, log_L3_min, log_L3_max` | AUDITED-CLEAN |
| 270 | `_scale_ratio_accepts` | `l3_img_arcsec, l3_cat_arcsec, scale_tol_frac` | AUDITED-CLEAN |
| 284 | `_knn_search_coords_from_pixels` | `stars, x_cen, y_cen, plate_scale_arcsec_per_px, use_sphere_knn` | AUDITED-CLEAN |
| 305 | `iter_local_knn_triangle_indices` | `stars, k_neighbors, search_coords` | AUDITED-CLEAN |
| 337 | `_sky_cell_vote_winner` | `hits, cell_deg` | AUDITED-CLEAN |
| 416 | `_load_index` | `index_path` | AUDITED-CLEAN |
| 435 | `_prepare_blind_context` | `dao_stars, index_path, n_top, plate_scale_arcsec_per_px, fov_deg, app_...` | AUDITED-CLEAN |
| 612 | `_iter_blind_pass_results` | `stars, hash_tree, metadata, has_vertices, log_L3_min, log_L3_max, plat...` | AUDITED-CLEAN |
| 767 | `_cluster_centroid_votes` | `vote_centers, radius_deg` | DEAD |
| 799 | `_cluster_match_hits_weighted` | `hits, radius_deg` | AUDITED-CLEAN |
| 835 | `_pick_cluster_representative` | `cluster_center, hits, radius_deg` | AUDITED-CLEAN |
| 856 | `_blind_cluster_params` | `app_config` | AUDITED-CLEAN |
| 883 | `_hits_unit_sphere_xyz` | `hits` | AUDITED-CLEAN |
| 893 | `_dbscan_vote_labels` | `hits, eps_deg, min_samples` | AUDITED-CLEAN |
| 938 | `_hits_to_candidates_dbscan` | `hits, app_config, top_n` | AUDITED-CLEAN |
| 1026 | `_candidate_near_existing` | `cand, existing` | AUDITED-CLEAN |
| 1035 | `_hits_to_candidates_legacy` | `hits, top_n, fov_deg, index_meta, img_select_per_cell` | AUDITED-CLEAN |
| 1091 | `find_blind_candidates` | `dao_stars, index_path, n_top, top_n, plate_scale_arcsec_per_px, fov_de...` | FLAGGED(MED) |
| 1235 | `find_blind_hint` | `dao_stars, index_path, n_top, min_votes, plate_scale_arcsec_per_px, fo...` | FLAGGED(MED) |
| 1428 | `_append_pass_diag` | `debug_sink, dub, n_tried, n_passed, n_below_min, n_above_max, n_in_ran...` | AUDITED-CLEAN |

### `vyvar_blind_series.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 24 | `target_density_deg2` | `cell_deg, stars_per_cell` | AUDITED-CLEAN |
| 29 | `estimate_rho_img_deg2` | `plate_scale_arcsec_per_px, fov_deg, img_budget, log_L3_max` | AUDITED-CLEAN |
| 46 | `density_hint_from_plate_scale` | `plate_scale_arcsec_per_px` | AUDITED-CLEAN |
| 62 | `build_tiers_from_config` | `cfg` | AUDITED-CLEAN |
| 79 | `_tier_sort_key` | `tier, rho_img` | AUDITED-CLEAN |
| 84 | `order_tiers_for_image` | `tiers, rho_img_deg2, plate_scale_arcsec_per_px` | AUDITED-CLEAN |
| 111 | `peek_index_log_l3_max` | `index_path` | AUDITED-CLEAN |
| 121 | `solve_blind_with_series` | `dao_df, app_config, plate_scale_arcsec_per_px, fov_deg, gaia_db_path, ...` | FLAGGED(MED) |
| 293 | `_solve_single` | `dao_df, cfg, plate_scale_arcsec_per_px, fov_deg, gaia_db_path, naxis1,...` | AUDITED-CLEAN |

### `vyvar_alignment_frame.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 34 | `_as_fits_float32_image` | `data` | NEEDS-TEST |
| 38 | `_hdr_has_wcs` | `header` | NEEDS-TEST |
| 42 | `_alignment_emit_log` | `log_sink, msg` | NEEDS-TEST |
| 49 | `_alignment_as_alignment_points` | `sources, label, log_sink` | NEEDS-TEST |
| 84 | `_alignment_detect_xy` | `img, want_max, det_sigma, fwhm_px, label, log_sink` | NEEDS-TEST |
| 253 | `_alignment_run_astroalign_points` | `source_pts, target_pts, image_source, image_target, max_control_points` | NEEDS-TEST |
| 288 | `_alignment_load_masterstar_catalog_points_for_frame` | `hdr_frame, shape_hw, platesolve_dir, align_star_cap` | NEEDS-TEST |
| 336 | `_alignment_compute_one_frame` | `fp, frame_index_1based, ctx, log_sink` | NEEDS-TEST |
| 743 | `_astrometry_align_mp_init` | `ctx` | NEEDS-TEST |
| 748 | `_astrometry_align_mp_task` | `item` | NEEDS-TEST |

### `astrometry_optimizer.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 59 | `_optimizer_parity_flip_already_in_fits` | `fits_path` | AUDITED-CLEAN |
| 77 | `_first_existing_col` | `df, candidates` | AUDITED-CLEAN |
| 84 | `_apply_wcs_pc_parity_flip_to_primary` | `fits_path, set_vy_optpf` | AUDITED-CLEAN |
| 126 | `_norm_id` | `v` | AUDITED-CLEAN |
| 131 | `_poly_features` | `xn, yn` | AUDITED-CLEAN |
| 137 | `_fit_poly_model` | `x, y, dx, dy` | AUDITED-CLEAN |
| 152 | `_eval_poly` | `model, x, y` | AUDITED-CLEAN |
| 160 | `_gaia_for_field` | `df, gaia_db_path, mag_limit, max_rows` | AUDITED-CLEAN |
| 205 | `_backfill_bp_rp_from_gdf_and_db` | `df, gdf, gmap, gaia_db_path` | AUDITED-CLEAN |
| 266 | `optimize_masterstar_matches` | `masterstars_csv, masterstar_fits, gaia_db_path, output_csv, gaia_mag_l...` | FLAGGED(MED) |

### `optics_autodetect.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 45 | `Detection.ok` | `self` | NEEDS-TEST |
| 50 | `Detection.autofill` | `self` | DEAD |
| 54 | `Detection.band` | `self` | NEEDS-TEST |
| 77 | `_hdr_float` | `header, *keys` | NEEDS-TEST |
| 91 | `_hdr_str` | `header, *keys` | NEEDS-TEST |
| 105 | `_norm` | `s` | NEEDS-TEST |
| 110 | `_model_core` | `norm_name` | NEEDS-TEST |
| 116 | `_parse_sensorsize` | `raw` | NEEDS-TEST |
| 129 | `_dims_close` | `a, b, tol_frac, tol_abs` | NEEDS-TEST |
| 137 | `detect_equipment` | `header, equipments` | NEEDS-TEST |
| 191 | `detect_telescope` | `header, telescopes` | NEEDS-TEST |
| 229 | `detect_location` | `header, locations` | NEEDS-TEST |
| 259 | `find_sample_light_header` | `source_root, max_scan` | NEEDS-TEST |
| 295 | `_has_any` | `header, *keys` | NEEDS-TEST |
| 307 | `assess_unresolved` | `header, equipment, telescope, location` | NEEDS-TEST |
| 354 | `autodetect_from_source` | `source_root, equipments, telescopes, locations` | NEEDS-TEST |

### `optics_selection.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 33 | `_first_db_optics_ids` | `db` | AUDITED-CLEAN |
| 68 | `parse_ui_optics_from_labels` | `equipment_label, telescope_label, equipment_options, telescope_options...` | AUDITED-CLEAN |
| 109 | `sync_optics_session` | `selection` | AUDITED-CLEAN |
| 121 | `optics_from_session` | `` | AUDITED-CLEAN |
| 140 | `resolve_optics_ids_for_platesolve` | `db, draft_id, equipment_id, telescope_id` | AUDITED-CLEAN |
| 178 | `resolve_working_optics` | `db, draft_id, ui, context` | AUDITED-CLEAN |
| 210 | `log_active_optics` | `db, selection, draft_id, context` | AUDITED-CLEAN |

### `platesolve_ui_paths.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 9 | `masterstars_csv_in_dir` | `d` | NEEDS-TEST |
| 17 | `list_platesolve_setup_dirs` | `ps_root` | NEEDS-TEST |
| 33 | `legacy_platesolve_root_bundle` | `ps_root` | NEEDS-TEST |
| 42 | `platesolve_bundle_dirs` | `ps_root` | NEEDS-TEST |
| 54 | `default_bundle_dir` | `ps_root, preferred_name` | NEEDS-TEST |
| 69 | `cone_csv_path` | `setup_dir` | NEEDS-TEST |
| 73 | `parse_draft_id_from_text` | `text` | NEEDS-TEST |
| 86 | `resolve_draft_directory` | `text, archive_root` | NEEDS-TEST |

---

## Test-gap list (Group 1 - science-critical)

| Module / function | Existing test | Gap |
|-------------------|---------------|-----|
| `vyvar_alignment_frame._alignment_compute_one_frame` | indirect via `test_astrometry_fault_isolation`, `test_gs11_pipeline` | No direct unit test for identity fallback, phase-corr fallback, or CP wiring |
| `pipeline._astrometry_align_impl_body` | partial pipeline smoke tests | No regression for `max_control_points` / `align_cp` binding |
| `astrometry_optimizer.optimize_masterstar_matches` | none dedicated | Optimizer grip/SIP cycles untested in isolation |
| `optics_autodetect.autodetect_from_source` | none | Fingerprint scoring bands untested |
| `platesolve_ui_paths.*` | none | Bundle discovery / draft resolution untested |
| `vyvar_platesolver._masterstar_solve_acceptance` | `test_masterstar_odds_acceptance`, `test_masterstar_catalog_recovery_gate` | Covered |
| `vyvar_blind_solver.find_blind_candidates` | `test_blind_*` suite | Covered |
| `vyvar_blind_series.solve_blind_with_series` | `test_blind_series` | Partial (scale ref 1.3 not tested cross-rig) |
| `optics_selection.resolve_working_optics` | `tests/test_optics_resolve.py` | Covered |

---

## Reproducibility scan (Group 1)

| Location | Issue | Severity |
|----------|-------|----------|
| `pipeline.py` alignment MP | `ProcessPoolExecutor` result order preserved via `pool.map` + sequential flush - deterministic for frame outputs | - |
| `vyvar_blind_solver._dbscan_vote_labels` | DBSCAN frontier order depends on seed index loop (deterministic for fixed hits list) | LOW |
| `vyvar_alignment_frame` astroalign | **FIXED Step 2:** `seeded_numpy_default_rng(VYVAR_RANDOM_SEED)` before `find_transform` | — |
| Alignment timestamps | `alignment_report.csv` metadata only - not in photometry columns | - |
| Blind index pickle load | Deterministic given same index file | - |

---

## Automation artifacts (tmp/, gitignored)

- `tmp/audit_group1.py` - scan driver
- `tmp/audit_group1_results.json` - raw inventory + lens hits
- `tmp/audit_group1_func_rows.md` - function rows
- `tmp/audit_group1_module_summary.json` - coverage counts

## Next group

**Group 2 - Photometry core:** `photometry_core`, `psf_*`, `comp_*`, `dilution`, `crowding_index`, `check_star_kmag`.
---

## Group 2 checkpoint - Photometry core (2026-06-20)

**Status:** AUDIT-ONLY batch appended; no code edits in this pass.

Method (Group 2): automated AST inventory + lens scans (L1-L11) on 10 modules (322 functions), targeted verification against `docs/VYVAR_CODE_AUDIT.md` DR4 threads, science-critical logic reads.

## Prioritized findings (Group 2 - deduplicated, severity-sorted)

| ID | Sev | Lens | Location | What's wrong | Principle (not fix) |
|----|-----|------|----------|--------------|---------------------|
| G2-F001 | **SUPERSEDED** | L4 | `photometry_core.py` (removed) | ~~catalog_only forced-aperture routing~~ **SUPERSEDED** (`7f0dc86`): path deleted; unmatched VSX excluded in Fáza 0 (`select_active_targets`). Saturated `skip_photometry` unchanged. | Photometry is DAO+Gaia matched only; no forced-aperture LCs. |
| G2-F002 | **SUPERSEDED** | L4 | `photometry_core.py` (removed) | ~~catalog_only WCS placement~~ **SUPERSEDED** (`7f0dc86`): placement helpers deleted with forced-aperture path. | Unmatched VSX are excluded, not placed via VSX coords. |
| G2-F002b | **MED** | L4 | `photometry_core.py` (per-frame) | **BACKLOG** — `dao_matched` frames on unsolved per-frame WCS can yield nondetection flux without explicit trust downgrade (distinct from masterstar placement). | Per-frame unsolved trust must downgrade, not silent nondetection. |
| G2-F003 | **FIXED** | L1 | `photometry_core.py` | ~~`apertures_px.get(target_cid, 3.0)` dilution fallback~~ **FIXED:** `_resolve_photometric_aperture_px_for_gs11` — per-star map → `_aperture_radius_from_snr_table` derive → skip+flag (`dilution_skipped`); fixed 3.0 removed; grounded in Seager 2003 / Howell 2006 (photometric aperture for neighbor search). | Missing per-star aperture must fail loud or derive from SNR table, not a universal 3.0 px literal. |
| G2-F004 | **MED** | L4 | `photometry_core.py:7897` | Fix-A `err` paired with `ensemble_scatter` by positional index only (DR4-1); row misalignment risks wrong error columns. | Error model pairing must use stable star keys, not parallel list order. |
| G2-F005 | **RESOLVED** | L4 | `photometry_core.py` (removed) | ~~Log claimed catalog_only skip while downstream implied photometry~~ **RESOLVED** (`7f0dc86`): catalog_only Phase 2A branch removed; only DAO-matched targets measured. | Operator logs must match the actual branch taken for catalog_only targets. |
| G2-P001 | **CLEAN** | - | `photometry_core.py:2454-2660` | `ensemble_normalize` verified against SPEC/Honeycutt ensemble ZP math. | Keep as reference implementation for comp ensemble detrend. |
| G2-P002 | **CLEAN** | - | `photometry_core.py:8037-8060` | Fix-A err model arithmetic verified. | Positional coupling (G2-F004) is separate from the core formula correctness. |
| G2-P003 | **CLEAN** | - | `dilution.py` | Dilution correction path read; no L1 universal-scale literals in hot path. | Dilution factors must stay tied to measured apertures and local crowding context. |
| G2-P004 | **CLEAN** | - | `photometry_core.py` (DR4-4/DR4-5) | Airmass detrend scaffolding removed/fixed; LC PNG export try-wrapped per DR reconciliation. | Retired DR threads should not leave dead branches or silent export failures. |

## Coverage table (Group 2 modules)

| Module | Lines | Funcs | Audited | DEAD (heuristic) | TRULY-DEAD | LIVE-DYNAMIC† | TEST-ONLY† |
|--------|-------|-------|---------|------------------|------------|---------------|------------|
| `photometry_core.py` | 13849 | 163 | 163 | 3 | 1 | 2 | 0 |
| `psf_photometry.py` | 3080 | 44 | 44 | 3 | 1 | 0 | 2 |
| `psf_runner.py` | 1568 | 28 | 28 | 0 | 0 | 0 | 0 |
| `psf_neighbor_sub.py` | 423 | 7 | 7 | 0 | 0 | 0 | 0 |
| `comp_selection_per_target.py` | 2381 | 21 | 21 | 0 | 0 | 0 | 0 |
| `comp_pool_rms.py` | 438 | 6 | 6 | 1 | 1 | 0 | 0 |
| `comp_qa_core.py` | 606 | 15 | 15 | 0 | 0 | 0 | 0 |
| `dilution.py` | 373 | 9 | 9 | 0 | 0 | 0 | 0 |
| `crowding_index.py` | 510 | 9 | 9 | 0 | 0 | 0 | 0 |
| `check_star_kmag.py` | 628 | 20 | 20 | 0 | 0 | 0 | 0 |
| **Group 2 total** | - | **322** | **322** | **7** | **3** | **2** | **2** |

† Reclassified 2026-06-20. FLAGGED 56 + NEEDS-TEST 105 unchanged.

## Per-module function registry (Group 2)

Schema: line | qualname | signature | status where status is AUDITED-CLEAN / FLAGGED(MED) / DEAD / NEEDS-TEST.

Flagged regions (MED): photometry_core phase2a per-target (`7284-8450`), comp funnel (`11292-12300`); psf_photometry bulk; dilution module-wide lens pass.

### `photometry_core.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 52 | `_safe_polyfit` | `x, y, deg, cov` | AUDITED-CLEAN |
| 87 | `_is_catalog_only` | `df` | AUDITED-CLEAN |
| 97 | `_sid_int` | `v` | AUDITED-CLEAN |
| 120 | `parse_comp_quality_json_map` | `raw` | AUDITED-CLEAN |
| 142 | `comp_quality_quality_strings` | `qmap` | AUDITED-CLEAN |
| 157 | `apply_comp_w_rel_for_display` | `comp_df, quality_map` | AUDITED-CLEAN |
| 186 | `_enrich_comp_bp_rp` | `candidates, gaia_db_path, gaia_prefetch` | AUDITED-CLEAN |
| 280 | `_normalize_gaia_id` | `x` | AUDITED-CLEAN |
| 290 | `_build_csv_lookup` | `csv_df, id_col` | AUDITED-CLEAN |
| 333 | `_lookup_star_in_csv` | `cid, id_map, xy_df, ref_x, ref_y, xy_tol_px` | AUDITED-CLEAN |
| 379 | `_sat_limit_peak_adu` | `cfg` | AUDITED-CLEAN |
| 385 | `_mad_sigma` | `arr` | AUDITED-CLEAN |
| 394 | `_aperture_to_mask_single` | `ap` | DEAD |
| 402 | `measure_fwhm_from_masterstar` | `masterstar_fits_path, star_positions, n_stars, fit_box_fwhm, dao_fwhm_...` | AUDITED-CLEAN |
| 576 | `compute_optimal_apertures` | `masterstar_fits_path, star_positions, fwhm_px, aperture_fwhm_factor, a...` | AUDITED-CLEAN |
| 629 | `_flux_to_mag` | `flux` | AUDITED-CLEAN |
| 636 | `_photometric_error` | `flux, sky_pp, area, gain, read_noise` | AUDITED-CLEAN |
| 660 | `compute_snr_optimal_aperture_table` | `fwhm_px, sky_adu_per_px, gain, read_noise, mag_range, mag_step, r_min_...` | AUDITED-CLEAN |
| 724 | `_resolve_phase2a_equipment_id` | `db, draft_id, output_dir, masterstar_fits_path` | AUDITED-CLEAN |
| 758 | `_draft_dir_from_phase2a_paths` | `output_dir, masterstar_fits_path` | AUDITED-CLEAN |
| 766 | `_phase2a_star_mag_lookup` | `at_df, comp_df, masterstar_fits_path` | AUDITED-CLEAN |
| 814 | `_median_sky_from_phase2a_csv_cache` | `csv_cache, fallback` | AUDITED-CLEAN |
| 831 | `_measured_aperture_from_proc_cache` | `catalog_id, csv_cache, id_col` | AUDITED-CLEAN |
| 862 | `_snr_table_radius_for_mag_bin` | `table, nearest` | AUDITED-CLEAN |
| 875 | `_aperture_radius_from_snr_table` | `star_mag, snr_table, aperture_fwhm_factor, fwhm_px` | AUDITED-CLEAN |
| 898 | `_get_star_aperture_px` | `catalog_id, star_mag, snr_table, fallback_r` | AUDITED-CLEAN |
| 932 | `resolve_draft_dir_for_snr_aperture_table` | `archive_root, draft_id, platesolve_dir, masterstar_fits_path` | AUDITED-CLEAN |
| 959 | `load_snr_aperture_table_from_draft_dir` | `draft_dir` | AUDITED-CLEAN |
| 990 | `_noise_floor_adu_from_image_array` | `data, prematch_peak_sigma_floor` | AUDITED-CLEAN |
| 1012 | `resolve_fwhm_px_for_snr_aperture_table` | `masterstar_fits_path, masterstar_selection, fwhm_fallback_px` | AUDITED-CLEAN |
| 1052 | `estimate_median_sky_adu_per_px_for_snr_table` | `aligned_fits_paths, aligned_ram_frames, max_frames, prematch_peak_sigm...` | AUDITED-CLEAN |
| 1098 | `precompute_and_save_snr_aperture_table_for_draft` | `draft_dir, masterstar_fits_path, masterstar_selection, fwhm_fallback_p...` | AUDITED-CLEAN |
| 1222 | `_coerce_bool_cell` | `v` | AUDITED-CLEAN |
| 1239 | `read_flux_from_csv` | `frame_csv_path, star_ids, apertures_px, sat_limit_adu, star_xy, xy_tol...` | AUDITED-CLEAN |
| 1476 | `_target_row_is_catalog_only` | `target_row` | AUDITED-CLEAN |
| 1486 | `_catalog_only_resolve_aligned_fits` | `proc_csv, aligned_dir` | AUDITED-CLEAN |
| 1497 | `_catalog_only_epoch_from_csv` | `csv_df` | AUDITED-CLEAN |
| 1513 | `_catalog_only_fixed_aperture_flux` | `data, x_c, y_c, r_ap, r_in, r_out` | AUDITED-CLEAN |
| 1573 | `_catalog_only_merge_frame_flux` | `df_frame, target_catalog_id, ra_deg, dec_deg, proc_csv, aligned_dir, f...` | AUDITED-CLEAN |
| 1747 | `temporal_bin_comp_lc` | `comp_lc, comp_quality, all_frames, window, enabled` | AUDITED-CLEAN |
| 1855 | `pytics_iterative_weights` | `comp_lc, comp_quality, comp_rms_map, n_iter, enabled` | AUDITED-CLEAN |
| 1956 | `_star_mag_for_aperture_sizing` | `row` | AUDITED-CLEAN |
| 1974 | `_common_mode_detrend_comp_lc` | `comp_lc, comp_bjd, min_frames` | AUDITED-CLEAN |
| 2041 | `_comp_lc_frame_ensemble_residual` | `comp_lc` | AUDITED-CLEAN |
| 2065 | `compute_lc_rms_ooe` | `mag_calib, flags, brightest_frac` | AUDITED-CLEAN |
| 2099 | `check_comparison_stability` | `comp_lc, comp_rms_map, comp_bjd, n_comp_min, outlier_sigma, max_comp_s...` | AUDITED-CLEAN |
| 2275 | `compute_aperture_correction` | `comp_df, frame_results, min_ref_stars, max_contamination, max_scatter_...` | AUDITED-CLEAN |
| 2414 | `ensemble_member_ids` | `comp_quality, comp_rms_map, n_comp_min, n_comp_max` | AUDITED-CLEAN |
| 2454 | `ensemble_normalize` | `target_mag_inst, comp_mag_inst, comp_catalog_mag, comp_quality, comp_r...` | FLAGGED(MED) |
| 2667 | `fit_color_term_c1` | `comp_mag_inst, comp_catalog_mag, comp_bp_rp, comp_quality, min_comp, s...` | AUDITED-CLEAN |
| 2780 | `apply_color_term` | `mag_calib, target_bp_rp, comp_bp_rp, comp_quality, c1` | AUDITED-CLEAN |
| 2823 | `_check_color_term_extrapolation` | `target_bp_rp, comp_bp_rp_values, target_name, extrapolation_tol` | AUDITED-CLEAN |
| 2866 | `should_apply_color_term` | `obs_group, c1, c1_stderr, n_comp, min_comp_for_ct, max_stderr_ratio` | AUDITED-CLEAN |
| 2989 | `_obs_group_filter_key` | `obs_group` | AUDITED-CLEAN |
| 2995 | `_is_nofilter_obs_group` | `obs_group` | AUDITED-CLEAN |
| 3013 | `_is_broadband_photometric_filter` | `obs_group` | AUDITED-CLEAN |
| 3066 | `resolve_apply_color_term` | `cfg, obs_group` | AUDITED-CLEAN |
| 3076 | `_target_display_name` | `row, fallback_cid` | AUDITED-CLEAN |
| 3098 | `_ensure_active_target_display_names` | `df` | AUDITED-CLEAN |
| 3127 | `_group_comp_mag_inst_from_flux_matrix` | `flux_matrix, comp_ids, csv_files` | AUDITED-CLEAN |
| 3153 | `_group_comp_mag_inst_from_proc_csvs` | `comp_ids, csv_files` | AUDITED-CLEAN |
| 3193 | `_comp_maps_from_comparison_stars_csv` | `comp_csv` | AUDITED-CLEAN |
| 3219 | `_compute_group_color_term_fit` | `comparison_stars_csv, flux_matrix, csv_files, obs_group, cfg` | AUDITED-CLEAN |
| 3269 | `_ensure_group_comp_pool_csv` | `platesolve_dir, masterstar_fits, masterstars_csv, cfg, draft_id, min_p...` | AUDITED-CLEAN |
| 3314 | `_variable_targets_looks_like_ct_presel_stub` | `vt_path, masterstars_csv` | AUDITED-CLEAN |
| 3335 | `ensure_full_variable_targets_if_presel_stub` | `variable_targets_csv, masterstars_csv, masterstar_fits, cfg, draft_id` | AUDITED-CLEAN |
| 3396 | `_ct_prototype_enabled` | `` | AUDITED-CLEAN |
| 3400 | `_color_term_cat_inst_scatter_pair` | `comp_mag_inst, comp_catalog_mag, comp_bp_rp, comp_quality, c1, min_com...` | AUDITED-CLEAN |
| 3477 | `_append_ct_prototype_row` | `draft_dir, row` | AUDITED-CLEAN |
| 3501 | `_target_row_is_vsx_known_variable` | `target_row` | AUDITED-CLEAN |
| 3511 | `empirical_feature_mask_mag` | `mag, k, min_run` | AUDITED-CLEAN |
| 3561 | `detect_outliers` | `mag_calib, flags_saturated, outlier_sigma, feature_mask, skip_sigma_cl...` | AUDITED-CLEAN |
| 3622 | `apply_reporting_postprocess` | `mag_calib, mag_calib_ct, target_row, target_name, sat_flags, target_fr...` | AUDITED-CLEAN |
| 3677 | `democratic_detrend_lc` | `mag_calib, bjd, airmass, flags, window_frac, polyorder, min_points, en...` | AUDITED-CLEAN |
| 3808 | `savgol_detrend_lc` | `mag_calib, bjd, flags, window_frac, polyorder, min_points, enabled` | AUDITED-CLEAN |
| 3903 | `save_lightcurve_csv` | `output_path, bjd, hjd, jd, airmass, is_flipped, mag_inst, mag_calib_ra...` | AUDITED-CLEAN |
| 4036 | `save_lightcurve_png` | `output_path, bjd, mag_calib, err, flags, target_name, comp_quality, de...` | AUDITED-CLEAN |
| 4140 | `save_cutout_png` | `output_path, masterstar_fits_path, xc, yc, target_name, size_px, ms_da...` | AUDITED-CLEAN |
| 4206 | `save_field_map_png` | `output_path, masterstar_fits_path, active_targets, comp_df, percentile...` | AUDITED-CLEAN |
| 4313 | `save_target_field_map_png` | `output_path, masterstar_fits_path, target_row, comp_rows, percentile_l...` | AUDITED-CLEAN |
| 4435 | `_edge_ok_from_masterstar_pipeline` | `masterstar_fits, stars_df, cfg_dict, ms_header, ms_data` | AUDITED-CLEAN |
| 4515 | `auto_export_variability_candidates_csv` | `masterstar_fits_path, comparison_stars_csv, per_frame_csv_dir, output_...` | AUDITED-CLEAN |
| 4933 | `_phase2a_coerce_skip_photometry` | `df` | AUDITED-CLEAN |
| 4950 | `build_rms_mag_model` | `summary_rows, zone_filter, min_stars` | AUDITED-CLEAN |
| 5007 | `expected_rms_from_model` | `mag, coeffs` | AUDITED-CLEAN |
| 5021 | `classify_lc_quality` | `zone_flag, lc_rms, lc_median_mag, n_frames, n_normal_frames, lunar_ris...` | AUDITED-CLEAN |
| 5101 | `build_lc_quality_summary` | `summary_rows, rms_model_coeffs, rms_model_n_stars, rms_noisy_k` | AUDITED-CLEAN |
| 5147 | `build_gs11_summary` | `summary_rows, cfg, comps_gs11_rejected, plate_scale_arcsec` | AUDITED-CLEAN |
| 5203 | `_phase2a_write_summary` | `summary_rows, output_dir, lunar_context, cfg, plate_scale_arcsec` | AUDITED-CLEAN |
| 5295 | `_phase2a_observer_location_dict` | `cfg, site, site_source` | AUDITED-CLEAN |
| 5338 | `merge_photometry_pipeline_meta` | `photometry_dir, updates, cfg` | AUDITED-CLEAN |
| 5356 | `_phase2a_resolve_field_center_ra_dec` | `ms_header, at_df` | AUDITED-CLEAN |
| 5387 | `_phase2a_collect_session_jd_values` | `frame_time_lookup` | AUDITED-CLEAN |
| 5447 | `_build_phase2a_dynamic_params` | `state, output_dir, aperture_fwhm_factor` | AUDITED-CLEAN |
| 5533 | `_phase2a_compute_lunar_context` | `state` | AUDITED-CLEAN |
| 5601 | `_load_blend_worklist` | `masterstar_fits_path` | AUDITED-CLEAN |
| 5642 | `_load_adaptive_blend_map` | `masterstar_fits_path` | AUDITED-CLEAN |
| 5649 | `_get_lc` | `cid, all_frames` | AUDITED-CLEAN |
| 5654 | `_get_comp_bjd_series` | `cid, all_frames` | AUDITED-CLEAN |
| 5664 | `_get_lc_psf_or_dao` | `cid, all_frames` | AUDITED-CLEAN |
| 5697 | `_get_lc_psf_strict` | `cid, all_frames` | AUDITED-CLEAN |
| 5714 | `_resolve_star_flux_method` | `cid, all_frames` | AUDITED-CLEAN |
| 5727 | `_get_lc_star_method` | `cid, all_frames, star_method` | AUDITED-CLEAN |
| 5737 | `_get_lc_adaptive_per_star` | `cid, all_frames` | AUDITED-CLEAN |
| 5743 | `compute_lc_flux_method` | `all_frames, blend_map, resolve_fwhm, snr_lo` | AUDITED-CLEAN |
| 5787 | `_get_lc_adaptive` | `cid, all_frames` | DEAD |
| 5805 | `load_epsf_metrics_for_draft` | `per_frame_csv_dir, active_targets_df` | AUDITED-CLEAN |
| 5910 | `_apply_role_aware_aperture_scaling` | `apertures_px, at_df, cfg` | AUDITED-CLEAN |
| 5946 | `_preserve_nondetection_flags_helper` | `out_flags_local, target_frames` | AUDITED-CLEAN |
| 5957 | `_frame_quality_gate_select` | `csv_files, cfg, proc_frame_store` | AUDITED-CLEAN |
| 6043 | `_proc_stem` | `name` | AUDITED-CLEAN |
| 6050 | `_compute_frame_align_residuals` | `csv_files, proc_frame_store` | AUDITED-CLEAN |
| 6117 | `_record_align_residuals_to_report` | `report_path, residuals` | AUDITED-CLEAN |
| 6138 | `_frame_align_residual_gate_select` | `csv_files, cfg, residuals, aperture_r_px` | AUDITED-CLEAN |
| 6177 | `_phase2a_prepare_shared_state` | `output_dir, lc_dir, masterstar_fits_path, comparison_stars_csv, per_fr...` | AUDITED-CLEAN |
| 6949 | `_recompute_bjd_hjd_per_target` | `jd_array, ra_deg, dec_deg, cfg, site` | AUDITED-CLEAN |
| 7035 | `_phase2a_catalog_only_nearest_comps` | `target_row, masterstars_df, n_comps, target_cid` | AUDITED-CLEAN |
| 7134 | `_phase2a_process_one_target` | `target_row, ti, state, summary_rows, n_lc, lc_dir, output_dir, progres...` | AUDITED-CLEAN |
| 8302 | `_phase2a_finalize_exports` | `summary_rows, lc_dir, output_dir, _cfg, n_lc, n_frames, at_df, field_m...` | FLAGGED(MED) |
| 8580 | `run_phase2a` | `masterstar_fits_path, active_targets_csv, comparison_stars_csv, per_fr...` | AUDITED-CLEAN |
| 8788 | `_get_plate_scale_from_cfg` | `cfg, db, draft_id, fits_path, ms_header` | AUDITED-CLEAN |
| 8941 | `_compute_fov_max_dist` | `frame_w_px, frame_h_px, plate_scale, fov_fraction, fallback_deg` | AUDITED-CLEAN |
| 8993 | `_resolve_plate_scale_arcsec_per_px` | `cfg, fits_path, ms_header` | AUDITED-CLEAN |
| 9033 | `_cd_matrix_scale_arcsec_per_px` | `hdr` | AUDITED-CLEAN |
| 9081 | `_read_plate_scale_from_fits_path` | `fits_path, ms_header` | AUDITED-CLEAN |
| 9165 | `_angular_distance_deg` | `ra1, dec1, ra2, dec2` | AUDITED-CLEAN |
| 9175 | `_normalize_id_value` | `x` | AUDITED-CLEAN |
| 9188 | `_normalize_id_series` | `s` | AUDITED-CLEAN |
| 9192 | `_bool_col` | `series` | AUDITED-CLEAN |
| 9204 | `stress_test_relative_rms_from_sidecars` | `frames_root, source_ids, sample_frac, seed, flux_col, name_col, min_st...` | AUDITED-CLEAN |
| 9291 | `vsx_is_known_variable_top3_per_bin` | `rows, phot_category_key, rms_key, ra_key, dec_key, max_per_bin, radius...` | AUDITED-CLEAN |
| 9348 | `common_field_intersection_bbox_px` | `frame_paths, finite_stride` | AUDITED-CLEAN |
| 9401 | `recommended_aperture_by_color` | `bp_rp, median_fwhm_blue, median_fwhm_neutral, median_fwhm_red` | AUDITED-CLEAN |
| 9435 | `bad_columns_for_light_frame` | `bpm, light_header` | AUDITED-CLEAN |
| 9464 | `_fwhm_moment_at` | `arr, xc, yc, half` | AUDITED-CLEAN |
| 9500 | `compute_auto_fwhm_limit` | `fwhm_values, k` | AUDITED-CLEAN |
| 9541 | `compute_fwhm_gaussian_for_aperture_catalog` | `df, data, hdr, gaussian_fwhm_px_override, aperture_fwhm_factor` | AUDITED-CLEAN |
| 9621 | `_sky_pp_from_annulus_image` | `d, ann_img` | AUDITED-CLEAN |
| 9634 | `_aperture_flux_sky_per_star` | `d, pos, r_ap_arr, r_in_arr, r_out_arr` | AUDITED-CLEAN |
| 9687 | `compute_per_frame_cog_correction` | `data, x, y, dao_flux, aperture_r_px, sky_pp, fwhm_px, peak_max_adu, sa...` | AUDITED-CLEAN |
| 9848 | `enhance_catalog_dataframe_aperture_bpm` | `df, data, hdr, aperture_enabled, aperture_fwhm_factor, annulus_inner_f...` | AUDITED-CLEAN |
| 10147 | `_phase0_effective_frame_hw_px` | `vt, ms, frame_w_px, frame_h_px, edge_margin_px` | AUDITED-CLEAN |
| 10174 | `_active_target_zone_flag` | `ms_row, zone_val_raw` | AUDITED-CLEAN |
| 10190 | `_auto_repair_catalog_ids` | `vt_path, gaia_db_path, log_fn, max_sep_arcsec` | AUDITED-CLEAN |
| 10234 | `_enrich_active_targets_bp_rp` | `targets_df, gaia_db_path` | AUDITED-CLEAN |
| 10311 | `_resolve_frame_hw_px_from_masterstar` | `ms_fits, frame_w_px, frame_h_px, db, draft_id` | AUDITED-CLEAN |
| 10366 | `_read_field_density_inputs` | `ms_fits, masterstars_csv, frame_w_px, frame_h_px` | AUDITED-CLEAN |
| 10421 | `_refresh_variable_targets_xy` | `variable_targets_csv, wcs, chip_w, chip_h` | AUDITED-CLEAN |
| 10483 | `select_active_targets` | `variable_targets_csv, masterstars_csv, frame_w_px, frame_h_px, edge_ma...` | AUDITED-CLEAN |
| 10956 | `_batch_enrich_targets_bp_rp_from_gaia_db` | `target_cids, gaia_db_path` | AUDITED-CLEAN |
| 11030 | `_enrich_target_bp_rp_from_gaia_db` | `target, gaia_db_path, vsx_local_db_path, gaia_prefetch` | AUDITED-CLEAN |
| 11104 | `_bprp_tier_ladder_for_selection` | `cfg, max_delta_bprp` | AUDITED-CLEAN |
| 11125 | `_select_comps_by_color_then_rms` | `candidates, target_bprp, n_comp_min, n_comp_max, max_delta_bprp, cfg` | AUDITED-CLEAN |
| 11202 | `_select_comps_tiered` | `candidates, n_comp_min, n_comp_max, tier_weights` | DEAD |
| 11278 | `build_global_comp_pool` | `masterstars_df, per_frame_csv_paths, csv_cache, variable_target_catalo...` | AUDITED-CLEAN |
| 11421 | `_dedupe_comp_pool_by_gaia_key` | `pool` | FLAGGED(MED) |
| 11444 | `_warn_zero_compstars_edge` | `target_cid, target, chip_fw, chip_fh, chip_interior_margin_px` | FLAGGED(MED) |
| 11486 | `_count_gate_passing_comps` | `result, per_target_rms_map, max_comp_rms, id_col` | FLAGGED(MED) |
| 11519 | `select_comparison_stars_per_target` | `target, masterstars_df, per_frame_csv_paths, csv_cache, global_comp_po...` | FLAGGED(MED) |
| 12220 | `run_phase0_and_phase1` | `variable_targets_csv, masterstars_csv, per_frame_csv_dir, output_dir, ...` | FLAGGED(MED) |
| 13163 | `run_sysrem_field` | `lc_dir, n_iter, flag_col, delta_col, err_col, out_col` | AUDITED-CLEAN |
| 13346 | `run_full_photometry_pipeline` | `masterstar_fits_path, variable_targets_csv, masterstars_csv, per_frame...` | AUDITED-CLEAN |
| 13537 | `_write_suspected_variables` | `ms_df, csv_paths, active_target_ids, output_path, flux_col, min_frames...` | AUDITED-CLEAN |

### `psf_photometry.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 44 | `_epsf_noop_finder` | `_data, mask` | FLAGGED(MED) |
| 50 | `_clamp_fwhm_px` | `v` | FLAGGED(MED) |
| 54 | `_median_fwhm_obs_files` | `db, draft_id` | FLAGGED(MED) |
| 75 | `_row_is_catalog_only` | `row` | FLAGGED(MED) |
| 85 | `_epsf_allowed_catalog_ids` | `platesolve_dir` | FLAGGED(MED) |
| 129 | `_epsf_positions_from_csvs` | `platesolve_dir, allowed_ids, existing` | FLAGGED(MED) |
| 182 | `get_epsf_fwhm_from_context` | `masterstar_fits_path, db, draft_id` | FLAGGED(MED) |
| 207 | `_to_odd_cutout` | `n` | FLAGGED(MED) |
| 214 | `_scalar_is_explicit_false` | `v` | FLAGGED(MED) |
| 230 | `_scalar_is_explicit_true` | `v` | FLAGGED(MED) |
| 246 | `_read_plate_scale_arcsec_px_from_fits` | `fits_path` | FLAGGED(MED) |
| 348 | `_fit_shape_for_cutout` | `cutout_size, fwhm_px` | FLAGGED(MED) |
| 365 | `_moffat_fwhm_px` | `gamma, alpha` | FLAGGED(MED) |
| 377 | `_compute_aperture_correction` | `psf_fluxes, ref_fluxes, chi2_vals, chi2_limit, min_ref_stars` | FLAGGED(MED) |
| 411 | `_compute_moffat_aperture_correction` | `moffat_results, dao_fluxes, chi2_limit, min_flux_snr, min_ref_stars` | FLAGGED(MED) |
| 475 | `_epsf_fwhm_native_legacy_px` | `epsf_data, osamp` | DEAD |
| 497 | `_epsf_fwhm_native_from_profile` | `epsf_data, osamp` | FLAGGED(MED) |
| 542 | `_epsf_build_imagepsf_from_stars` | `stars, osamp, fwhm_px, cutout_size` | FLAGGED(MED) |
| 629 | `_load_cone_catalog` | `epsf_dir` | FLAGGED(MED) |
| 652 | `_epsf_augment_candidates_from_detected_pool` | `mpath, csv_ok, star_rows, fwhm_px, db, cfg, funnel` | FLAGGED(MED) |
| 819 | `_epsf_prepare_stars` | `masterstar_fits_path, masterstars_csv_path, db, draft_id, min_stars, m...` | FLAGGED(MED) |
| 1353 | `build_epsf_model` | `masterstar_fits_path, masterstars_csv_path, db, draft_id, oversampling...` | FLAGGED(MED) |
| 1446 | `_parse_psf_grid` | `grid` | FLAGGED(MED) |
| 1457 | `build_epsf_grid_model` | `masterstar_fits_path, masterstars_csv_path, db, draft_id, grid, oversa...` | FLAGGED(MED) |
| 1589 | `interp_gridded_epsf_array` | `grid, x, y` | FLAGGED(MED) |
| 1628 | `fit_moffat_psf_stars` | `frame_data, frame_hdr, star_positions, fwhm_guess_px, cutout_size, err...` | FLAGGED(MED) |
| 1934 | `_aperture_annulus_radii_px` | `fwhm_px` | FLAGGED(MED) |
| 1952 | `_border_median_sky_from_cutout` | `cut` | FLAGGED(MED) |
| 1964 | `_psf_annulus_radii_px` | `fwhm_px, inner_fwhm, outer_fwhm` | FLAGGED(MED) |
| 1980 | `_annulus_median_per_px` | `frame_data, x, y, r_in, r_out` | FLAGGED(MED) |
| 2017 | `_annulus_sky_per_px_custom` | `frame_data, x, y, fwhm_px, inner_fwhm, outer_fwhm` | DEAD |
| 2047 | `_subtract_psf_models` | `frame_data, psf_model, sources` | FLAGGED(MED) |
| 2076 | `_residual_annulus_sky_per_px` | `frame_data, x, y, fwhm_px, psf_model, sources, inner_fwhm, outer_fwhm` | FLAGGED(MED) |
| 2109 | `_annulus_sky_per_px_full_frame` | `frame_data, x, y, fwhm_px` | FLAGGED(MED) |
| 2147 | `_psf_resolve_gain_read_noise` | `frame_hdr` | FLAGGED(MED) |
| 2165 | `_psf_sky_only_sigma_per_px` | `sky_per_px_adu, gain, read_noise_e` | FLAGGED(MED) |
| 2174 | `_psf_fit_error_cutout` | `cut_shape, sky_per_px, gain, read_noise_e, err_full_cut` | FLAGGED(MED) |
| 2192 | `_psf_fit_region_mask` | `shape, cy, cx, fit_shape` | DEAD |
| 2211 | `_psf_sandwich_flux_err` | `flux_fit, psf_model, x_fit, y_fit, cut_shape, sky_per_px, gain, read_n...` | FLAGGED(MED) |
| 2267 | `_apply_psf_fixed_position` | `phot, fix` | FLAGGED(MED) |
| 2278 | `_resolve_psf_fit_sky` | `frame_data, cut, x, y, fwhm_px` | FLAGGED(MED) |
| 2294 | `_grouped_psf_fit` | `frame_data, err_full, x, y, fwhm_px, fit_shape, psf_model, neighbor_xy...` | FLAGGED(MED) |
| 2476 | `assess_psf_quality` | `chi2, snr, pos_shift_px, fwhm_px, nn_dist_fwhm, nn_delta_mag, chi2_bad` | FLAGGED(MED) |
| 2526 | `psf_photometry_stars` | `frame_data, frame_hdr, star_positions, epsf_model_path, cutout_size, e...` | AUDITED-CLEAN |

### `psf_runner.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 81 | `_force_utf8_stdout` | `` | NEEDS-TEST |
| 97 | `subtract_local_sky` | `cutout, border_width` | NEEDS-TEST |
| 148 | `_header_egain_only` | `hdr` | NEEDS-TEST |
| 161 | `get_gain_from_header` | `hdr, db_gain_e_per_adu` | NEEDS-TEST |
| 171 | `_vy_qcrms_from_header` | `hdr` | NEEDS-TEST |
| 184 | `_vy_fwhm_from_header` | `hdr` | NEEDS-TEST |
| 195 | `_draft_equipment_id` | `db, draft_id` | NEEDS-TEST |
| 210 | `_draft_db_gain_e_per_adu` | `db, draft_id` | NEEDS-TEST |
| 223 | `_runner_database_path` | `` | NEEDS-TEST |
| 235 | `_fit_shape_for_cutout` | `cutout_size` | NEEDS-TEST |
| 243 | `_load_psf_photometry_bundle` | `epsf_path` | NEEDS-TEST |
| 269 | `_per_cutout_error_map` | `cut_sub, gain, sky_rms` | NEEDS-TEST |
| 278 | `_psf_stars_local_cutouts` | `frame_data_psf, star_positions, phot, cutout_size, border_width, gain,...` | NEEDS-TEST |
| 411 | `_print_chi2_distribution` | `chi` | NEEDS-TEST |
| 435 | `_comp_fail_reason_psf` | `x, y, fw, fh, half_cs, peak_adu, sat_lim_adu` | NEEDS-TEST |
| 463 | `_print_dry_run_two_frame_report` | `bundle, comp_df` | NEEDS-TEST |
| 539 | `_proc_metrics_by_catalog` | `proc_df` | NEEDS-TEST |
| 561 | `_comp_catalog_sat_peak` | `comp_df` | NEEDS-TEST |
| 582 | `_print_table` | `df, cols, max_rows` | NEEDS-TEST |
| 598 | `flag_blended_stars` | `comp_df, fwhm_px` | NEEDS-TEST |
| 615 | `_paths` | `` | NEEDS-TEST |
| 642 | `step_1_build_epsf` | `` | NEEDS-TEST |
| 699 | `step_2_load_targets` | `` | NEEDS-TEST |
| 778 | `_build_frame_xy_lookup` | `proc_df` | NEEDS-TEST |
| 805 | `step_3_run_psf_on_frames` | `var_df, comp_df, epsf_path, max_frames` | NEEDS-TEST |
| 1161 | `step_4_build_summary` | `` | NEEDS-TEST |
| 1285 | `step_5_calibrate_lightcurve` | `` | NEEDS-TEST |
| 1500 | `main` | `` | NEEDS-TEST |

### `psf_neighbor_sub.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 23 | `_flux_to_mag` | `flux, zp` | NEEDS-TEST |
| 29 | `_aperture_area` | `r_ap` | NEEDS-TEST |
| 33 | `_aperture_noise_adu` | `stamp, r_ap, gain_e_per_adu, read_noise_e` | NEEDS-TEST |
| 62 | `_moffat_gamma` | `fwhm_px, beta` | NEEDS-TEST |
| 66 | `_stamp_sky_median` | `stamp, margin` | NEEDS-TEST |
| 82 | `_joint_moffat_fit_subtract` | `stamp, target_xy, neighbour_xys, fwhm_px, fit_beta, centroid_bound_fwh...` | NEEDS-TEST |
| 192 | `neighbor_sub_target_flux` | `stamp, target_xy, neighbour_xys, fwhm_px, r_ap, r_in, r_out, delta_mag...` | NEEDS-TEST |

### `comp_selection_per_target.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 38 | `bo_cvn_funnel_snapshot` | `` | NEEDS-TEST |
| 42 | `_log_bo_cvn_comp_funnel` | `step_counts, max_comp_rms, n_comp_max, rms_rejected` | NEEDS-TEST |
| 82 | `_pixel_distance_deg_vectorized` | `x_t, y_t, x_arr, y_arr, plate_scale_arcsec` | NEEDS-TEST |
| 104 | `_angular_distance_deg_vectorized` | `ra_t, dec_t, ra_arr, dec_arr` | NEEDS-TEST |
| 126 | `_resolve_target_color_for_comp_selection` | `target, vsx_local_db_path, gaia_db_path, cfg` | NEEDS-TEST |
| 278 | `_adaptive_mag_filter` | `all_candidates, target_mag, mag_diff_start, mag_diff_absolute, n_comp_...` | NEEDS-TEST |
| 321 | `_filter_comp_candidates_spatial_static` | `ms, ra_t, dec_t, mag_t, target_cid, target_bprp_eff, max_delta_bprp_cf...` | NEEDS-TEST |
| 570 | `_build_candidates_pre_adaptive_mag` | `ms, _base_mask, det_mask, mag_t, target_cid, mag_tol, max_mag_diff, n_...` | NEEDS-TEST |
| 669 | `_bootstrap_phase1_csv_cache` | `per_frame_csv_paths, csv_cache, flux_col, avail_cols` | NEEDS-TEST |
| 732 | `_accumulate_per_frame_comp_metrics` | `per_frame_csv_paths, csv_cache, cand_ids, flux_col, chip_fw, chip_fh` | NEEDS-TEST |
| 1073 | `_apply_comp_metric_hard_filters` | `flux_map, peak_over_map, peak_total_map, snr_map, psf_chi2_map, fwhm_m...` | NEEDS-TEST |
| 1223 | `_compute_comp_contamination_map` | `flux_map, ms, target_cid, isolation_radius_px` | NEEDS-TEST |
| 1321 | `_mad_sigma` | `values` | NEEDS-TEST |
| 1333 | `_flux_series_to_mag_bjd` | `flux_map, bjd_map` | NEEDS-TEST |
| 1358 | `_common_mode_detrend_mag_lcs` | `mag_lc, bjd_lc` | NEEDS-TEST |
| 1399 | `_iterative_ensemble_clip_cm_residual` | `flux_map, bjd_map, provisional_rms, clip_sigma, n_comp_min, max_iter, ...` | NEEDS-TEST |
| 1519 | `_detrend_and_compute_comp_rms_map` | `flux_map, min_frames, max_comp_rms, n_comp_min, target_cid, target, ch...` | NEEDS-TEST |
| 1646 | `_ensemble_mad_filter_rms` | `rms_map, candidates, target_cid, target, n_comp_min, rms_outlier_sigma...` | NEEDS-TEST |
| 1711 | `_score_comp_candidates_broeg` | `active, candidates, contamination_map, id_col_cand, mag_t, target_bprp...` | NEEDS-TEST |
| 1786 | `_assign_comp_tiers_to_pool` | `candidates, active, id_col_cand, target, target_cid, target_bprp_eff, ...` | NEEDS-TEST |
| 2124 | `_assemble_comp_selection_result_rows` | `selected_ids, final_comps, id_col_cand, active, score_map, contaminati...` | NEEDS-TEST |

### `comp_pool_rms.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 19 | `norm_med_for_bin` | `b, bin_meds, bin_keys` | NEEDS-TEST |
| 42 | `_norm_id_val` | `x` | NEEDS-TEST |
| 54 | `_norm_id_series` | `s` | DEAD |
| 58 | `sort_per_frame_csv_paths` | `per_frame_csv_paths, csv_cache` | NEEDS-TEST |
| 76 | `compute_global_pool_rms_map` | `cand_ids, _masterstars_df, per_frame_csv_paths, csv_cache, flux_col, m...` | NEEDS-TEST |
| 408 | `attach_comp_rms_to_pool_rows` | `pool, rms_map, id_col` | NEEDS-TEST |

### `comp_qa_core.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 27 | `mad_sigma` | `x` | NEEDS-TEST |
| 37 | `robust_thr` | `vals, k` | NEEDS-TEST |
| 48 | `flux_to_mag` | `f` | NEEDS-TEST |
| 56 | `inst_mag_from_flux` | `flux` | NEEDS-TEST |
| 64 | `comp_axis_mag` | `flux, row` | NEEDS-TEST |
| 76 | `loo_diff_series` | `mag, focus_id, comp_ids` | NEEDS-TEST |
| 100 | `sokolovsky_indices` | `m` | NEEDS-TEST |
| 127 | `build_locus` | `mags, sigmas, bin_width` | NEEDS-TEST |
| 167 | `locus_at` | `mag, centers, locs, spreads` | NEEDS-TEST |
| 180 | `flag_reasons` | `sigma_iqr, inv_nv, spike, inst_mag, locus_centers, locus_med, locus_sp...` | NEEDS-TEST |
| 204 | `worst_flagged_score` | `metrics, flags, inst_mag, locus_centers, locus_med, locus_spread, thr_...` | NEEDS-TEST |
| 227 | `load_proc_pivot` | `proc_dir, ids` | NEEDS-TEST |
| 257 | `compute_comp_qa` | `photometry_dir, proc_dir, mad_k, min_comps, max_comps, _target_process...` | NEEDS-TEST |
| 515 | `write_comp_qa_artifacts` | `result, photometry_dir, lc_dir, update_summary` | NEEDS-TEST |
| 565 | `run_comp_qa_for_photometry_dir` | `photometry_dir, proc_dir, lc_dir, mad_k, min_comps, max_comps, update_...` | NEEDS-TEST |

### `dilution.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 35 | `_haversine_arcsec` | `ra1_deg, dec1_deg, ra2_deg, dec2_deg` | FLAGGED(MED) |
| 44 | `_normalize_exclude_source_id` | `catalog_id` | FLAGGED(MED) |
| 59 | `query_gaia_neighbors` | `ra_deg, dec_deg, radius_arcsec, gaia_db_path, mag_limit, exclude_sourc...` | FLAGGED(MED) |
| 144 | `flux_from_gmag` | `g_mag` | FLAGGED(MED) |
| 153 | `_no_blend_result` | `aperture_arcsec, search_radius_arcsec` | FLAGGED(MED) |
| 168 | `compute_dilution_factor` | `ra_deg, dec_deg, g_mag, aperture_arcsec, gaia_db_path, catalog_id, sea...` | FLAGGED(MED) |
| 268 | `_star_g_mag` | `star` | FLAGGED(MED) |
| 280 | `compute_dilution_batch` | `stars, aperture_arcsec, gaia_db_path, mag_limit_delta` | FLAGGED(MED) |
| 333 | `apply_target_dilution_to_mag_calib` | `mag_calib, dilution_result, cfg, target_cid` | FLAGGED(MED) |

### `crowding_index.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 51 | `_load_wcs_meta` | `ms_fits` | NEEDS-TEST |
| 80 | `_gain_rn_for_draft` | `db, draft_id` | NEEDS-TEST |
| 100 | `_howell_snr` | `flux, sky_pp, area, gain, rn` | NEEDS-TEST |
| 107 | `_interp_snr5_crossing` | `mag_bins, snr_bins` | NEEDS-TEST |
| 131 | `_analytic_snr5` | `fwhm_px, sky_pp, gain, rn, zero_point` | NEEDS-TEST |
| 156 | `_load_lc_star_table` | `platesolve_dir` | NEEDS-TEST |
| 195 | `_build_blend_targets_df` | `stars_df, wcs, cone_f, fwhm_px, sky_pp, area, gain, rn, zp, frame_limi...` | NEEDS-TEST |
| 269 | `ensure_crowding_targets_for_lc` | `draft_dir, setup, db, draft_id, gaia_db_max_g, force` | NEEDS-TEST |
| 304 | `compute_crowding_index` | `draft_dir, setup, db, draft_id, gaia_db_max_g, lc_star_set` | NEEDS-TEST |

### `check_star_kmag.py`

| Line | Function | Signature | Status |
|------|----------|-----------|--------|
| 24 | `normalize_comp_df_export_columns` | `comp_df, comp_quality_map` | NEEDS-TEST |
| 40 | `_norm_ensemble_id_set` | `ensemble_ids` | NEEDS-TEST |
| 51 | `_comp_rms_map_from_df` | `comp_df` | NEEDS-TEST |
| 66 | `resolve_ensemble_ids_for_check` | `target_cid, comp_df, lc_dir, comp_quality_map, cfg` | NEEDS-TEST |
| 108 | `_resolve_check_select_rms_floor` | `df, cfg, floor_override` | NEEDS-TEST |
| 136 | `_drop_rms_artefacts` | `df, cfg, floor_override` | NEEDS-TEST |
| 170 | `_apply_crowding_exclusion` | `df, cfg` | NEEDS-TEST |
| 186 | `_exclude_ensemble_members` | `df, ensemble_ids` | NEEDS-TEST |
| 202 | `field_check_star_candidate_pool` | `comp_field_df, target_comps` | NEEDS-TEST |
| 233 | `select_check_star` | `comp_df, ensemble_ids, n_comp_min, cfg, check_select_rms_floor` | NEEDS-TEST |
| 311 | `comp_ensemble_maps` | `comp_df, cfg` | NEEDS-TEST |
| 351 | `compute_check_ensemble_mag_calib` | `check_cid, comp_ids, comp_lc, comp_catalog_mag, comp_quality, comp_rms...` | NEEDS-TEST |
| 399 | `check_kmag_sidecar_path` | `lc_dir, target_cid` | NEEDS-TEST |
| 403 | `save_check_kmag_sidecar` | `path, check_cid, bjd, source_files, kmag` | NEEDS-TEST |
| 427 | `_fmt_kmag` | `v` | NEEDS-TEST |
| 431 | `kmag_from_sidecar` | `sidecar_path, source_files` | NEEDS-TEST |
| 462 | `resolve_proc_csv_dir` | `photometry_dir, obs_group` | NEEDS-TEST |
| 470 | `_inst_mag_from_proc_row` | `row, export_method` | NEEDS-TEST |
| 489 | `build_aligned_comp_inst` | `proc_dir, comp_ids, source_files, cfg, export_method, csv_cache` | NEEDS-TEST |
| 536 | `kmag_values_for_export` | `check_row, comp_df, lc_normal, target_cid, lc_dir, proc_dir, comp_qual...` | NEEDS-TEST |

## Test-gap list (Group 2 - science-critical)

| Module / function | Existing test | Gap |
|-------------------|---------------|-----|
| `photometry_core catalog_only / forced_aperture (`7199-7473`)` | partial DR4 tests | No regression that `lc_source` matches actual flux path when `skip_photo=True` |
| `photometry_core VY_PSOLV placement skip (`10782`)` | none dedicated | Unsolved-frame catalog_only trust downgrade untested |
| `photometry_core Fix-A err vs ensemble_scatter (`7897`)` | none dedicated | Positional coupling vs catalog_id join not tested |
| `photometry_core `ensemble_normalize`` | indirect pipeline smoke | Honeycutt reference-case unit coverage thin |
| `psf_photometry.* (40 FLAGGED)` | partial PSF suite | Bulk MED flags lack per-function isolation tests |
| `psf_runner.*` | none | 28/28 NEEDS-TEST - runner orchestration untested |
| `psf_neighbor_sub.*` | none | Neighbor subtraction untested |
| `comp_selection_per_target.*` | none | Per-target comp funnel untested in isolation |
| `comp_pool_rms.*` | none | Pool RMS binning untested |
| `comp_qa_core.*` | none | Comp QA gates untested |
| `dilution.* (9 FLAGGED)` | none | Dilution fallback and crowding coupling untested |
| `crowding_index.*` | none | Crowding index untested |
| `check_star_kmag.*` | none | Kmag sidecar export path untested |

Inventory: 105 NEEDS-TEST + 56 FLAGGED(MED) functions across Group 2 (100% statused).

## Reproducibility scan (Group 2)

| Location | Issue | Severity |
|----------|-------|----------|
| `photometry_core.py` phase2a per-target loop | Frame order from caller lists; no unseeded RNG in hot aperture path | - |
| `psf_runner.py` multiprocessing | Worker pool ordering depends on executor flush; verify seed if stochastic steps added | LOW |
| Alignment inputs (Group 1 fix `98de910`) | Astroalign now seeded before photometry inputs - stabilizes per-frame flux inputs | - |
| `catalog_only` early return (`7199`) | Deterministic skip but science outcome depends on branch wiring, not RNG | MED (logic) |
| `VY_PSOLV` gate (`10782`) | Header-driven branch; reproducible given same FITS headers | - |
| LC PNG export (DR4-5) | try/except wrapped; failure is silent at log level only | LOW |

## Automation artifacts (Group 2, tmp/, gitignored)

- `tmp/audit_group2.py` - scan driver
- `tmp/audit_group2_results.json` - raw inventory + lens hits
- `tmp/audit_group2_func_rows.md` - function rows
- `tmp/audit_group2_module_summary.json` - coverage counts
- `tmp/append_ledger_g2.py` - ledger append (this script)

**Checkpoint policy:** Milan + Claude review Group 3 batch before Group 4.

---

## Group 3 checkpoint - Data / IO / catalog (2026-06-20)

**Status:** AUDIT-ONLY batch appended; no code edits in this pass.

Method (Group 3): automated AST inventory + lens scans (L1-L11) on 15 modules (367 functions), science-critical reads of calibration/importer/DB/Gaia/VSX/crossmatch paths, reconciliation with DECISIONS (GAIA-1/2/3, match-rate metric).

### Per-equipment config state (roadmap note)

Rig intrinsics live in SQLite `EQUIPMENTS` / `TELESCOPE` (`GAIN_ADU`, `READNOISE_E`, `PIXELSIZE`, `SENSORSIZE`, `SATURATE_ADU`, `FOCAL`) joined via `OBS_DRAFT`. `CALIBRATION_LIBRARY` rows optionally scoped by `(ID_EQUIPMENTS, ID_TELESCOPE)` with legacy NULL=global masters. Unified provenance resolver: `param_resolver.py` (header → DB → config). No separate per-equipment JSON blob beyond DB + draft provenance sidecars.

## Prioritized findings (Group 3 - deduplicated, severity-sorted)

| ID | Sev | Lens | Location | What's wrong | Principle (not fix) |
|----|-----|------|----------|--------------|---------------------|
| G3-F001 | **HIGH** | L4 | `database.py:2047-2189`, `importer.py:709-838` | ~~`find_best_calibration_library_path` accepts `CCD_TEMP IS NULL`…~~ **FIXED** (fix log step 5): scoped-only match; dark temp-required; flat no-exptime; registration/fallback parity; `_calibration_light_temp_c` reads raw CCD_TEMP. | Calibration master selection must require temperature match and equipment scope; NULL temp is not equivalent to a match. |
| G3-F002 | **FIXED** | L1/L4 | `database.py:133-256`, `pipeline.py:11404-11411`, `4405` | ~~`query_local_gaia` defaulted `mag_limit=11.5` when omitted~~ **FIXED (Path A):** `mag_limit=None` ⇒ **no g_mag SQL cap**; explicit float unchanged (clamp/log). MASTER_SOURCES calls `mag_limit=None` (small det±0.01° bbox — **no max_rows** guard; dense-box risk negligible). `_query_gaia_local` when `max_mag is None` now full depth. | Gaia cone limits must derive from field depth / caller intent, not a universal 11.5 mag floor. |
| G3-F003 | **MED** | L3 | `importer.py:692-693`, `733-734` | `_find_matching_master_in_library`: broad `except` on DB lookup and per-file `extract_fits_metadata` → skip candidate with no log at DEBUG only on metadata path. | Master discovery failures must log at operator-visible level; silent skip risks missing calibration. |
| G3-F004 | **MED** | L3/L6 | `database.py:173-179`, `507-511`, `query_local_vsx` | Read-path side effects: `CREATE INDEX IF NOT EXISTS` + `commit` on first Gaia/VSX query can block large DBs mid-pipeline. | Index creation belongs to build/migration, not hot query paths. |
| G3-F005 | **MED** | L1 | `param_resolver.py:85-87` | `GAIN_SETTING_INDEX_MAP` only maps equipment `1` (QHY294MM); other cameras with index-style GAIN headers lack map → DB-only path. | Equipment-specific header index maps must be DB-driven or complete per seeded rig. |
| G3-F006 | **MED** | L11 | `catalog_crossmatch.py:599-655` | Online catalog workers (Vizier/SIMBAD) use epoch-blind cone queries; no PM propagation at crossmatch layer (distinct from platesolver PM path). GAIA-1 deferred for DB build, not for operator crossmatch UI. | Crossmatch epoch/PM handling must be explicit per catalog epoch or flagged uncertain. |
| G3-F007 | **MED** | L3 | `database.py:34-73` | `get_gaia_db_max_g_mag` returns `0.0` on query failure (logged once) — callers may treat as “empty DB” vs error. | Catalog depth probes must distinguish failure vs empty vs valid max. |
| G3-F008 | **MED** | L4 | `calibration.py:441-452` | `get_processed_master(..., allow_passthrough=True)` synthesizes zero/one master if file missing — **no production callers** today but dangerous if wired. | Missing master must fail loud; synthetic passthrough corrupts all flux. |
| G3-F009 | **LOW** | L5 | `database.py`, `importer.py`, `catalog_crossmatch.py` | ~~AST ref-count: ~189 DEAD…~~ **RECLASSIFIED** (2026-06-20): heuristic 207 → **25 TRULY-DEAD**, 182 LIVE-DYNAMIC (dispatch/registry/Qt/dunder); see DEAD reclassification section. | Dead API surface should be trimmed after UI dispatch audit; only TRULY-DEAD rows are removal candidates. |
| G3-D001 | **DEFER** | L11 | `vyvar_platesolver.py:63` | Known DR4 forward hook: `GAIA_EPOCH=2016.0` → J2017.5 for DR4 build (**not a new finding**). | Epoch constant moves with DR4 rebuild; do not restart DR3. |
| G3-D002 | **DEFER** | L11 | GAIA-1/GAIA-2 | `pmra/pmdec`, `ruwe` columns deferred to DR4 build per DECISIONS/ROADMAP — `build_gaia_catalog.py` schema omits them today; `query_local_gaia` optionally reads if present. | DR4 build adds columns; no DR3 rebuild. |
| G3-P001 | **CLEAN** | - | `param_resolver.py` | Documented provenance chain (header → DB → config); site `ok=False` when unresolved; DB cross-check for plausible-but-wrong headers (draft 363 pixel case). | Reference pattern for rig-parameter authority. |
| G3-P002 | **CLEAN** | - | `calibration.py:412-499` | Dark=block sum, flat=block mean, `VYFLNRD` normalize-after-resample; `infer_spatial_block_factor` for 2795×4164↔1397×2082; Bayer per-tile flat norm. | Calibration math verified; shape inference is principled not universal literals. |
| G3-P003 | **CLEAN** | - | `gaia_catalog_id.py` | `normalize_gaia_source_id`, `read_vyvar_csv` str dtype, `masterstar_row_gaia_key` — ID integrity for 19-digit Gaia keys. | Gaia ID handling is the catalog join root. |
| G3-P004 | **CLEAN** | - | `proc_frame_store.py` | Single read / column union for proc CSV; Gaia ID normalization; tested (`test_proc_frame_store.py`). | Proc IO cache pattern is sound. |
| G3-P005 | **CLEAN** | - | `database.py:2795-2927` | `get_combined_metadata`: focal/pixel from header with DB fallback; binning inferred from `SENSORSIZE` when header lies — connects to G1 T2-2 fix downstream. | Rig geometry merge path is principled (not hardcoded 2082×1397). |
| G3-P006 | **CLEAN** | - | `VSX/vsx_make.py` | Schema matches `query_local_vsx` consumers (`oid`, `ra_deg`, `dec_deg`, `var_type`, `period`, `mag_max/min`); incremental `INSERT OR IGNORE`. | VSX local build aligns with DB query layer. |

**Cross-rig trace (Group 3 as source):** Universal literals `2082×1397`, `9.77″/px`, `1.3″/px` do **not** live in Group 3 hot paths (only `calibration.py` documents them as shape examples). Downstream patches (G1 T2-1/T2-2) consumed `get_combined_metadata` / `param_resolver` / header cache — root authority is here, not in photometry.

## Coverage table (Group 3 modules)

| Module | Lines | Funcs | Audited | DEAD (heuristic) | TRULY-DEAD | LIVE-DYNAMIC† | TEST-ONLY† |
|--------|-------|-------|---------|------------------|------------|---------------|------------|
| `database.py` | 4181 | 151 | 151 | 72 | 13 | 59 | 0 |
| `importer.py` | 2034 | 54 | 54 | 46 | 5 | 41 | 0 |
| `calibration.py` | 549 | 15 | 15 | 9 | 0 | 9 | 0 |
| `proc_frame_store.py` | 290 | 14 | 14 | 3 | 1 | 2 | 0 |
| `catalog_crossmatch.py` | 681 | 23 | 23 | 17 | 0 | 17 | 0 |
| `crossmatch_runner.py` | 299 | 7 | 7 | 5 | 0 | 5 | 0 |
| `gaia_catalog_id.py` | 207 | 8 | 8 | 1 | 0 | 1 | 0 |
| `param_resolver.py` | 641 | 27 | 27 | 20 | 2 | 18 | 0 |
| `draft_provenance.py` | 139 | 9 | 9 | 0 | 0 | 0 | 0 |
| `time_utils.py` | 266 | 11 | 11 | 4 | 3 | 1 | 0 |
| `fits_suffixes.py` | 12 | 1 | 1 | 0 | 0 | 0 | 0 |
| `masterstar_context.py` | 185 | 5 | 5 | 1 | 0 | 1 | 0 |
| `GAIA_DR3/build_gaia_catalog.py` | 738 | 27 | 27 | 20 | 0 | 20 | 0 |
| `GAIA_DR3/build_blind_index.py` | 481 | 11 | 11 | 7 | 1 | 6 | 0 |
| `VSX/vsx_make.py` | 195 | 4 | 4 | 2 | 0 | 2 | 0 |
| **Group 3 total** | - | **367** | **367** | **207** | **25** | **182** | **0** |

† **LIVE-DYNAMIC** / **TEST-ONLY**: reclassification of heuristic-DEAD rows only (2026-06-20 pass). These functions have callers via direct/symbol ref, string dispatch, `getattr`, Qt `.connect`, registry tuples (e.g. `_CATALOG_WORKERS`), dunder protocol, or CLI `__main__`. They are **not** removal candidates.

**Prior checkpoint** reported 189 heuristic DEAD (under-counted module sum); automated inventory is **207** zero-outside-ref private/public functions.

Flagged counts (9 FLAGGED) and NEEDS-TEST (142) unchanged from original lens pass. Heuristic DEAD column retained for diff against TRULY-DEAD.

## Group 3 DEAD reclassification (2026-06-20, AUDIT-ONLY)

**Goal:** Trustworthy dead-code status before any removal. Re-checked all **207** heuristic-DEAD Group-3 functions with eight caller mechanisms (direct/attribute call, symbol ref, string ref in py/ui/json, `getattr`, Qt `.connect`, `super()`, registry/CLI `__main__`, dunder protocol). Excluded `tmp/` audit tooling from reference scans.

### Summary

| Metric | Count |
|--------|-------|
| Heuristic DEAD (original AST pass) | 207 |
| **TRULY-DEAD** (no caller by any mechanism) | **25** |
| LIVE-DYNAMIC (reclassified from DEAD) | 182 |
| TEST-ONLY (Group-3 subset) | 0 |

**False-positive rate:** 182/207 ≈ **88%** of heuristic DEAD were live (mostly `database.py` / `importer.py` CRUD + `catalog_crossmatch.py` `_CATALOG_WORKERS` registry).

### TRULY-DEAD list (removal candidates — each still needs per-step do-no-harm before delete)

| Location | Function |
|----------|----------|
| `database.py:1065` | `update_master_source_safety` |
| `database.py:1104` | `count_final_data_for_equipment_id` |
| `database.py:1113` | `count_final_data_for_telescope_id` |
| `database.py:1464` | `set_obs_draft_masterstar_path` |
| `database.py:1543` | `qc_processing_run_exists` |
| `database.py:1550` | `delete_qc_processing_run_by_hash` |
| `database.py:2511` | `get_setting_int` |
| `database.py:2521` | `set_setting` |
| `database.py:3131` | `get_observation_metadata` |
| `database.py:3174` | `update_observation_import_log` |
| `database.py:3211` | `insert_observation_files` |
| `database.py:3371` | `fetch_draft_scanning_ids` |
| `database.py:3622` | `finalize_draft` |
| `importer.py:146` | `_is_empty_or_missing` |
| `importer.py:296` | `_format_temp` |
| `importer.py:325` | `_first_fits_in_dir` |
| `importer.py:396` | `_resolve_session_lights` |
| `importer.py:1500` | `_copy_fits_folder` |
| `proc_frame_store.py:283` | `frame_columns` |
| `param_resolver.py:479` | `resolve_saturation` |
| `param_resolver.py:529` | `resolve_exptime` |
| `time_utils.py:33` | `_clamp_lat` |
| `time_utils.py:37` | `_clamp_lon` |
| `time_utils.py:46` | `_clamp_elev` |
| `GAIA_DR3/build_blind_index.py:117` | `triangle_hash` |

**Full per-function reclassification** (all 207 rows): `tmp/reclassify_group3_dead_table.md` (gitignored).

### Group 1 / Group 2 DEAD spot-check (5 each)

Same mechanism pass on ledger DEAD samples:

| Group | Sample | Result |
|-------|--------|--------|
| G1 (5/5) | `_fits_header_positive_float`, `_per_frame_noise_error_map`, `get_auto_fov`, `_cluster_centroid_votes`, `autofill` | **TRULY-DEAD** (all 5) |
| G2 (5) | `_aperture_to_mask_single`, `_norm_id_series` | **TRULY-DEAD** |
| G2 | `_get_lc_adaptive`, `_select_comps_tiered` | **LIVE-DYNAMIC** (script/symbol refs) |
| G2 | `_epsf_fwhm_native_legacy_px` | **TEST-ONLY** |

**Conclusion:** Group 1/2 low heuristic DEAD counts are **mostly genuine** after full reclassification (G1 12/13 TRULY-DEAD, G2 3/7 TRULY-DEAD + 2 TEST-ONLY); Group 3 over-count was the DB/UI artifact.

### Group 1 / Group 2 full DEAD reclassification (2026-06-20)

| Group | Heuristic DEAD | TRULY-DEAD | LIVE-DYNAMIC | TEST-ONLY |
|-------|----------------|------------|--------------|-----------|
| G1 | 13 | 12 | 1 (`optics_autodetect.Detection.autofill` Qt) | 0 |
| G2 | 7 | 3 | 2 (`_get_lc_adaptive`, `_select_comps_tiered`) | 2 (psf test helpers) |

Artifacts: `tmp/reclassify_g1_g2_dead_results.json`, `tmp/reclassify_g1_g2_truly_dead.txt`.

## Coverage table (Group 3 modules) — original heuristic (superseded)

| Module | Lines | Funcs | Audited | DEAD | FLAGGED | NEEDS-TEST | CLEAN |
|--------|-------|-------|---------|------|---------|------------|-------|
| `database.py` | 4181 | 151 | 151 | 72 | 3 | 74 | 0 |
| `importer.py` | 2034 | 54 | 54 | 46 | 2 | 8 | 0 |
| `calibration.py` | 549 | 15 | 15 | 9 | 2 | 4 | 0 |
| `proc_frame_store.py` | 290 | 14 | 14 | 3 | 0 | 11 | 0 |
| `catalog_crossmatch.py` | 681 | 23 | 23 | 17 | 1 | 5 | 0 |
| `crossmatch_runner.py` | 299 | 7 | 7 | 5 | 0 | 2 | 0 |
| `gaia_catalog_id.py` | 207 | 8 | 8 | 1 | 0 | 7 | 0 |
| `param_resolver.py` | 641 | 27 | 27 | 20 | 1 | 6 | 0 |
| `draft_provenance.py` | 139 | 9 | 9 | 0 | 0 | 9 | 0 |
| `time_utils.py` | 266 | 11 | 11 | 4 | 0 | 7 | 0 |
| `fits_suffixes.py` | 12 | 1 | 1 | 0 | 0 | 1 | 0 |
| `masterstar_context.py` | 185 | 5 | 5 | 1 | 0 | 4 | 0 |
| `GAIA_DR3/build_gaia_catalog.py` | 738 | 27 | 27 | 20 | 0 | 7 | 0 |
| `GAIA_DR3/build_blind_index.py` | 481 | 11 | 11 | 7 | 0 | 4 | 0 |
| `VSX/vsx_make.py` | 195 | 4 | 4 | 2 | 0 | 2 | 0 |
| **Group 3 total** | - | **367** | **367** | **189** | **9** | **142** | **0** |

Flagged counts include manual science review overrides (automation under-flagged hot paths). **DEAD column superseded** by reclassification table above.

## Per-module function registry (Group 3)

Schema: line | qualname | signature | status.

**Full rows:** `tmp/audit_group3_func_rows.md` (all 367 functions). Below: flagged hot paths + small modules.

Flagged regions: `database.find_best_calibration_library_path`, `database.query_local_gaia`, `database.get_combined_metadata`; `importer._find_matching_master_in_library`, `_stack_calibration_frames`; `calibration.get_processed_master`, `normalize_flat_master`; `catalog_crossmatch.check_candidate_in_catalogs`; `param_resolver.resolve_gain`.

### `gaia_catalog_id.py` (all functions)

| Line | Function | Status |
|------|----------|--------|
| 24 | `normalize_gaia_source_id` | NEEDS-TEST |
| 52 | `normalize_gaia_source_id_series` | NEEDS-TEST |
| 62 | `normalize_gaia_source_id_set` | NEEDS-TEST |
| 71 | `read_vyvar_csv` | NEEDS-TEST |
| 95 | `masterstar_row_gaia_key` | NEEDS-TEST |
| 118 | `_coerce_catalog_id_cell` | NEEDS-TEST |
| 131 | `PROC_CSV_READ_COLS` | NEEDS-TEST |
| 134 | `_GAIA_ID_DTYPE` | LIVE-DYNAMIC (reclass) |

### `proc_frame_store.py` (all functions)

| Line | Function | Status |
|------|----------|--------|
| 88 | `proc_csv_path_for_aligned_fits` | NEEDS-TEST |
| 98 | `ProcFrameStore.__init__` | NEEDS-TEST |
| 103 | `build` | NEEDS-TEST |
| 168 | `get` | NEEDS-TEST |
| 176 | `items` | NEEDS-TEST |
| 180 | `keys` | NEEDS-TEST |
| 184 | `values` | NEEDS-TEST |
| 188 | `__len__` | NEEDS-TEST |
| 192 | `__contains__` | NEEDS-TEST |
| 196 | `frame_df` | NEEDS-TEST |
| 204 | `lookup` | NEEDS-TEST |
| 212 | `_read_proc_csv` | NEEDS-TEST |
| 248 | `_coerce_numeric_cols` | NEEDS-TEST |
| 262 | `_normalize_ids` | LIVE-DYNAMIC (reclass) |

### `param_resolver.py` (flagged + public resolvers)

| Line | Function | Status |
|------|----------|--------|
| 120 | `_is_valid` | LIVE-DYNAMIC (reclass) |
| 128 | `_header_float` | LIVE-DYNAMIC (reclass) |
| 178 | `resolve_gain` | FLAGGED(MED) |
| 248 | `resolve_read_noise` | NEEDS-TEST |
| 312 | `resolve_pixel_um` | NEEDS-TEST |
| 348 | `resolve_focal_mm` | NEEDS-TEST |
| 384 | `resolve_plate_scale` | NEEDS-TEST |
| 420 | `resolve_site` | NEEDS-TEST |
| 498 | `resolve_exptime` | NEEDS-TEST |
| 520 | `resolve_binning` | NEEDS-TEST |
| 548 | `resolve_saturation` | NEEDS-TEST |

### `calibration.py` (all functions)

| Line | Function | Status |
|------|----------|--------|
| 55 | `_parse_master_header_datetime` | LIVE-DYNAMIC (reclass) |
| 79 | `get_master_age_days` | NEEDS-TEST |
| 101 | `read_master_binning_from_header` | LIVE-DYNAMIC (reclass) |
| 108 | `read_master_binning_from_fits` | NEEDS-TEST |
| 115 | `infer_spatial_block_factor` | NEEDS-TEST |
| 137 | `infer_spatial_upscale_factor` | LIVE-DYNAMIC (reclass) |
| 155 | `align_resampled_master_to_light_shape` | NEEDS-TEST |
| 198 | `resample_master_to_light_binning` | NEEDS-TEST |
| 248 | `_flat_saved_unnormalized` | LIVE-DYNAMIC (reclass) |
| 268 | `normalize_flat_master` | FLAGGED(MED) |
| 414 | `get_processed_master` | FLAGGED(MED) |
| 514 | `_bayer_pattern_from_db` | LIVE-DYNAMIC (reclass) |
| 528 | `_assumed_bayer_pattern` | LIVE-DYNAMIC (reclass) |
| 536 | `_parse_bayer_pattern_text` | LIVE-DYNAMIC (reclass) |
| 548 | `_bayer_tile_slices` | LIVE-DYNAMIC (reclass) |

## Test-gap list (Group 3 - science-critical)

| Module / function | Existing test | Gap |
|-------------------|---------------|-----|
| `find_best_calibration_library_path` (NULL temp, legacy scope) | none | Wrong-master selection untested |
| `query_local_gaia` default mag 11.5 | mocked in blind_verify only | No test that faint fields truncate comps |
| `get_processed_master` resample dark/flat math | validation `recover.py` import | No unit tests for shape inference / Bayer flat |
| `normalize_flat_master` Bayer tiles | none | OSC flat norm edge cases untested |
| `importer._find_matching_master_in_library` | pre_cal E2E indirect | Exposure/temp mismatch negatives untested |
| `get_combined_metadata` binning inference | none dedicated | SENSORSIZE vs header lie untested |
| `param_resolver.resolve_gain` | `test_gain_header_resolution.py` | Only QHY index map; other rigs thin |
| `query_local_gaia_by_source_ids` | none | bp_rp backfill for off-cone matched stars untested |
| `catalog_crossmatch.check_candidate_in_catalogs` | scripts only | No pytest; timeout/VSX local path untested |
| `crossmatch_runner` | none | Batch CSV crossmatch untested |
| `GAIA_DR3/build_gaia_catalog.py` | none | Build/resume TAP untested (offline script) |
| `VSX/vsx_make.py` | verify script | Schema + mag_limit incremental untested in pytest |
| `proc_frame_store` | `test_proc_frame_store.py` | Partial — column union / failed frame fallback |

Inventory: 142 NEEDS-TEST + 9 FLAGGED + **25 TRULY-DEAD** (reclassified 2026-06-20; was 207 heuristic DEAD) across Group 3.

## Reproducibility scan (Group 3)

| Location | Issue | Severity |
|----------|-------|----------|
| `database.query_local_gaia` | Global `_GAIA_INDEX_CHECK_DONE` — first-query index side effect | MED |
| `GAIA_DR3/build_gaia_catalog.py` | TAP download order deterministic; resume via `INSERT OR IGNORE` | - |
| `importer._stack_calibration_frames` | Mean/median stack deterministic | - |
| `catalog_crossmatch` ThreadPoolExecutor | Completion order non-deterministic for UI bullets only | LOW |
| `gaia_catalog_id.normalize_gaia_source_id` | Pure string logic | - |
| `fits_header_cache` | mtime-based freshness — reproducible given same files | - |

## Automation artifacts (Group 3, tmp/, gitignored)

- `tmp/audit_group3.py` - original scan driver
- `tmp/audit_group3_results.json` - raw inventory + lens hits
- `tmp/audit_group3_func_rows.md` - function rows (367)
- `tmp/audit_group3_module_summary.json` - coverage counts
- `tmp/reclassify_group3_dead.py` - DEAD reclassification driver
- `tmp/reclassify_group3_dead_results.json` - per-function new status
- `tmp/reclassify_group3_dead_table.md` - full 207-row reclass table
- `tmp/reclassify_group3_truly_dead.txt` - TRULY-DEAD short list
- `tmp/reclassify_g1_g2_spotcheck.json` - G1/G2 spot-check

- `tmp/reclassify_g1_g2_dead.py` - G1/G2 DEAD reclassification driver
- `tmp/reclassify_g1_g2_dead_results.json` - G1/G2 per-function new status
- `tmp/reclassify_g1_g2_truly_dead.txt` - G1/G2 TRULY-DEAD short list

---

## Group 4 checkpoint — Science / variability / QA (2026-06-20)

**Status:** AUDIT-ONLY batch appended; no code edits in this pass.

Method: AST inventory + mechanism-aware DEAD pass + lens scans (L1/L11 emphasis) on 16 modules (166 functions); science-critical reads of variability/trust/comp_qa/xval/TESS/HRD paths; reconciliation with DECISIONS (trust gate, comp QA Sokolovsky LOO, SEP xval CLOSED).

### Prioritized findings (Group 4 — deduplicated, severity-sorted)

| ID | Sev | Lens | Location | What's wrong | Principle (not fix) |
|----|-----|------|----------|--------------|---------------------|
| G4-F001 | **RESOLVED (Option B)** | L4 | `trust_flag_core.py`, `xval_run.py`, docs | Docs described a 3-axis production trust gate including SEP/xval; production uses comp QA + check-star + `lc_quality_flag` only. **Fix (2026-06-19):** DECISIONS/JOURNAL/PIPELINE_CZ corrected; offline headers on `xval_*`; `tests/test_no_xval_in_production.py` guard. No production logic change. | Trust certification must include every axis documented as production, or docs must mark SEP as manual/offline-only. |
| G4-F002 | **RESOLVED** | L4 | `photometry_core.py` (removed) | ~~`catalog_only` / `lc_source=forced_aperture` → `lc_quality_flag` noisy~~ **RESOLVED** (`7f0dc86`): no catalog_only or forced-aperture LCs remain. | Forced-aperture LCs must not be interpreted as astrophysical variability outside explicit `lc_source` / zone guards. |
| G4-F003 | **MED** | L3 | `photometry_core.py:8363-8380`, `comp_qa` stage ~8355 | Trust and comp_qa stages wrapped in **non-fatal** `except` — failure logs warning and leaves summary without fresh trust/comp QA columns (not fail-open GREEN, but **silent skip** of verdict layer). | Science verdict stages must log at operator-visible level and surface “trust not evaluated” on failure. |
| G4-F004 | **MED** | L11 | `comp_qa_core.py:27-46`, `loo_diff_series` | Sokolovsky LOO + mag-locus thresholds (`_SPIKE_HARD=3.0`, MAD with `n<2→nan`) are principled but **small-N / all-NaN comp sets** need adversarial tests before CLEAN. | LOO comp rejection must be validated on thin pools and sparse fields. |
| G4-F005 | **MED** | L1/L4 | `xval_harness_core.py:14-19`, `assign_sep_confidence` | SEP/DAO ratio thresholds (`_R_SEP_CONFIRMED_HI=1.40`, etc.) are **hardcoded** from draft_000365 harness — not config-sourced; harness is offline-only. | Cross-val pass bounds must be config-visible and tied to equipment/depth when used for gating. |
| G4-F006 | **MED** | L11 | `tess_verify.py`, `tess_runner.py` | TESS cross-match + comparison — epoch/coordinate handling needs explicit tests (Gaia epoch vs TESS BJD); not fully traced in audit pass. | External survey comparison must document epoch basis and tolerance. |
| G4-F007 | **LOW** | L5 | `ui_variability.py:139`, `:700`, `validate_lc_crossval.py:102` | **3 TRULY-DEAD** UI/helper symbols after mechanism pass (see coverage). | Trim only after UI dispatch confirmation. |
| G4-P001 | **CLEAN** | - | `trust_flag_core.py:35-36`, `write_trust_artifacts` | Missing trust-map entries default **RED** (`_UNEVALUATED_TRUST`) — fail-closed, not fail-open GREEN. | Conservative default for uncertified targets. |
| G4-P002 | **CLEAN** | - | `ui_variability.py:609-618`, `photometry_core.py:4588+` | `catalog_only` excluded from variability candidate detection (sky-noise path). | Catalog-only LCs must not enter field-variability candidate pool. |
| G4-P003 | **CLEAN** | - | `comp_qa_core.py` | Sokolovsky indices + mag-locus LOO QA read-only post-2A; aligns with DECISIONS comp-degradation spec. | Comp health gate is productionized and byte-neutral on photometry. |

### Coverage table (Group 4 modules)

| Module | Funcs | Audited | TRULY-DEAD | LIVE-DYNAMIC† | FLAGGED | NEEDS-TEST |
|--------|-------|---------|------------|---------------|---------|------------|
| `variability_detector.py` | 9 | 9 | 0 | 5 | 3 | 1 |
| `ui_variability.py` | 30 | 30 | 2 | 14 | 11 | 3 |
| `tess_runner.py` | 7 | 7 | 0 | 5 | 2 | 0 |
| `tess_verify.py` | 30 | 30 | 0 | 23 | 2 | 5 |
| `hrd_analysis.py` | 20 | 20 | 0 | 9 | 4 | 7 |
| `ui_hrd.py` | 1 | 1 | 0 | 0 | 0 | 1 |
| `trust_flag.py` | 1 | 1 | 0 | 0 | 1 | 0 |
| `trust_flag_core.py` | 14 | 14 | 0 | 1 | 11 | 2 |
| `xval_run.py` | 4 | 4 | 0 | 0 | 4 | 0 |
| `xval_harness_core.py` | 7 | 7 | 0 | 0 | 0 | 7 |
| `validate_lc_crossval.py` | 14 | 14 | 1 | 1 | 3 | 9 |
| `comp_qa.py` | 2 | 2 | 0 | 0 | 2 | 0 |
| `comp_qa_core.py` | 15 | 15 | 0 | 0 | 5 | 10 |
| `ui_masterstar_qa.py` | 5 | 5 | 0 | 4 | 0 | 1 |
| `masterstar_qa_plot.py` | 5 | 5 | 0 | 0 | 0 | 5 |
| `method_lc_output.py` | 2 | 2 | 0 | 1 | 1 | 0 |
| **Group 4 total** | **166** | **166** | **3** | **63** | **49** | **51** |

† LIVE-DYNAMIC: heuristic-zero-ref functions reclassified live (Qt callbacks, registry refs). Full rows: `tmp/audit_group4_func_rows.md`.

### TRULY-DEAD (Group 4)

| Location | Function |
|----------|----------|
| `ui_variability.py:139` | `_raw_lightcurve_from_frames` |
| `ui_variability.py:700` | `_render_field_image_with_candidate` |
| `validate_lc_crossval.py:102` | `_row_mag` |

### Test-gap list (Group 4 — science-critical)

| Module / function | Existing test | Gap |
|-------------------|---------------|-----|
| `trust_flag_core.evaluate_target` / missing map | `test_trust_flag.py` + `test_no_xval_in_production.py` | SEP-absent production trust documented and guarded |
| `classify_lc_quality` + `catalog_only` | none | Forced-aperture high RMS → `lc_quality_flag` path untested |
| `comp_qa_core` Sokolovsky LOO | scripts | No pytest for small-N comp pool / all-NaN LOO |
| `assign_sep_confidence` | harness only | Threshold edges (1.40, 0.70) not unit-tested |
| `variability_detector.compute_rms_variability` | none | Mag-bin envelope + candidate filters untested |
| `validate_lc_crossval` | offline script | 0.2%/frame agreement not in pytest |
| `tess_verify` | none | Cross-match tolerance / epoch handling untested |
| `hrd_analysis` | none | Dereddening / abs-mag assumptions untested |

### Reproducibility scan (Group 4)

| Location | Issue | Severity |
|----------|-------|----------|
| `trust_flag_core` | Deterministic given fixed summary inputs | - |
| `comp_qa_core` | LOO order depends on comp list order | LOW |
| `xval_run` | Offline harness; Gaia TAP query order | LOW |
| `ui_variability` | Session-state dependent UI | - |

### Automation artifacts (Group 4, tmp/, gitignored)

- `tmp/audit_group4.py` — scan driver
- `tmp/audit_group4_results.json` — inventory + lens hits
- `tmp/audit_group4_func_rows.md` — function rows (166)

**Checkpoint policy:** Milan + Claude review Group 4 batch before Group 5.

---

## Group 5 checkpoint — Reporting / export (2026-06-20)

**Status:** AUDIT-ONLY batch appended; no code edits in this pass.

Method: AST inventory + mechanism-aware DEAD pass + lens scans (L1/L3/L11 emphasis) on 6 modules (174 functions); targeted reads of `export_lightcurve_reports`, AAVSO/VarAstro column assembly, trust-note wiring, PDF overflow discipline, citations vs production trust gate (post G4-F001), `lc_source` / `forced_aperture` provenance (post G2-F001); reconciliation with DECISIONS export specs.

### Prioritized findings (Group 5 — deduplicated, severity-sorted)

| ID | Sev | Lens | Location | What's wrong | Principle (not fix) |
|----|-----|------|----------|--------------|---------------------|
| G5-F001 | **RESOLVED** | L4/L11 | `export_reports.py` (removed `lc_source` path) | ~~Export ignored `lc_source` / forced_aperture~~ **RESOLVED** (`7f0dc86`): forced-aperture / catalog_only LCs no longer exist; `export_reports.py` has no `lc_source` references. | Forced-aperture / catalog-only LCs must be labeled in every deliverable format or excluded consistently. |
| G5-F002 | **RESOLVED (non-issue)** | L11 | `export_reports.py:609-621`, `861-862` | ~~AC mag with uncorrected `err`~~ **RESOLVED (non-issue):** AC is a constant `delta_m_corr` shift; `err` (photon+ensemble SEM) is invariant; no `err_ac` in pipeline; folding `ac_scatter` per-point would misrepresent correlated systematic as random. | Exported uncertainty must match the magnitude column actually submitted. |
| G5-F003 | **FIXED** | L11 | `photometry_report.py` | ~~`_generate_candidate_lc_png` picks first `*mag*` → `mag_inst`~~ **FIXED** (`76c5a93`; superseded by G5-F011): candidate figures use `mag_calib_final`. | Report figures must use the same calibrated mag column as exports unless explicitly labeled instrumental. |
| G5-F004 | **FIXED** | L3 | `export_reports.py`, `photometry_core.py` | ~~Export failures swallowed (continue + info log)~~ **FIXED** (`efbb4de`): `record_export_failure` + `log_export_batch_summary`; per-target ERROR log; Phase 2A batch collector; batch still completes. | Deliverable writes must surface failures at operator-visible level; missing export files must not be silent. |
| G5-F005 | **RESOLVED** | L4 | `photometry_report.py` glossary | ~~PDF glossary missing `lc_source=forced_aperture`~~ **RESOLVED** (`7f0dc86`): forced-aperture LC provenance obsolete; glossary `zone_flag` is linear/saturated only; legacy display `zone_flag` on field maps is not export provenance. | Report labels must reflect `lc_source` semantics, not legacy zone-only wording. |
| G5-F006 | **FIXED** | L11 | `photometry_report.py` | ~~PDF LC/glossary time labels use BJD without (TDB)~~ **FIXED** (`b74c301`): `_pdf_time_axis_label`; BJD(TDB) for bjd/bjd_tdb/bjd_tdb_mid columns; VarAstro unchanged. | Stated time system in labels must match the stored column and export headers. |
| G5-F007 | **FIXED** | L1 | `export_reports.py` | ~~Default `arcsec_per_px=1.3` + hardcoded `#SOFTWARE=VYVAR/1.0`~~ **FIXED** (`6774f83`): derive-or-None plate scale from `pipeline_meta` / MASTERSTAR WCS; `VYVAR_SOFTWARE_VERSION` + `_aavso_software_header_line` for `#SOFTWARE`. | Report/export metadata must be config- or draft-sourced, not hardcoded placeholders. |
| G5-F008 | **FIXED** | L11 | `export_reports.py`, `photometry_report.py` | ~~VarAstro `n_good_comp` vs trust `n_clean` confusion~~ **FIXED** (`07e6f69`, **distinct metrics**): header label `n_ensemble_comp` (stability good+suspect); trust uses comp_qa `n_clean`; glossary + calibration doc. | Export provenance fields must cite the same comp-health metric as the trust gate. |
| G5-F009 | **LOW** | L5 | `export_reports.py:106`, `387`, `494` | **3 TRULY-DEAD** helpers after mechanism pass (`_observer_location_configured`, `_test_is_eclipsing`, `_comp_quality_map_for_export`). | Trim only after confirming no dynamic/registry dispatch. |
| G5-F010 | **LOW** | L5 | `photometry_report.py:522`, `1874` | **2 TRULY-DEAD** builder methods (`_lunar_risk_fill_color`, `_katalogy_cell_for_pdf`). | UI/PDF trim candidates after dispatch confirmation. |
| G5-F011 | **FIXED** | L11 | `photometry_core.py`, `export_reports.py`, `photometry_report.py` | ~~Parallel CT/AC on `mag_calib`; export used AC, main PDF used CT~~ **FIXED** (`be3e193`): `mag_calib_final` = `mag_calib` + CT + AC; LC CSV + export + all publication LC figures; `lc_rms`/trust stay on `mag_calib`. | All publication-facing figures and export must use the same canonical calibrated magnitude. |
| G5-P001 | **CLEAN** | - | `export_reports.py:677-679`, `41-78` | Per-draft `pipeline_meta.json` observer site preferred over session `cfg` for export coordinates — matches BJD/airmass site. | Exported observer location must match photometry time-system site. |
| G5-P002 | **CLEAN** | - | `export_reports.py:794-803`, `958-963` | Trust notes use `format_export_trust_note` / `format_varastro_trust_comment` from `trust_flag_core` — **no SEP / 3-axis wording** in export comments (post G4-F001). | Export trust text must match production trust gate semantics. |
| G5-P003 | **CLEAN** | - | `citations.py:353-366`, `tests/test_export_citations.py:124-138` | DATA-QUALITY GATE cites Sokolovsky + von Neumann for comp_qa/trust; **Barbary/Bertin (SEP) excluded** from export when trust/comp_qa on. | Methods/citations must not cite offline-only SEP axis as production. |
| G5-P004 | **CLEAN** | - | `photometry_report.py:1545-1577`, `5072-5076` | PDF builder supports `verify_overflow` mode with `_bounds_check` / violation logging (“0 PDF overflow” discipline). | Long tables/names must not silently overflow page bounds. |
| G5-P005 | **CLEAN** | - | `export_reports.py:854`, `914`, `1006` | AAVSO `#DATE=BJD` + VarAstro `# TIME SYSTEM: BJD(TDB)` align with LC `bjd` from `bjd_tdb_mid`. | Export time-system headers must match LC column semantics. |

### Coverage table (Group 5 modules)

| Module | Funcs | Audited | TRULY-DEAD | LIVE-DYNAMIC† | FLAGGED | NEEDS-TEST |
|--------|-------|---------|------------|---------------|---------|------------|
| `photometry_report.py` | 123 | 123 | 2 | 89 | 11 | 21 |
| `export_reports.py` | 27 | 27 | 3 | 17 | 6 | 1 |
| `pdf_report.py` | 1 | 1 | 0 | 0 | 0 | 1 |
| `report_methods.py` | 9 | 9 | 0 | 0 | 0 | 9 |
| `citations.py` | 12 | 12 | 0 | 5 | 1 | 6 |
| `jd_axis_format.py` | 2 | 2 | 0 | 0 | 2 | 0 |
| **Group 5 total** | **174** | **174** | **5** | **111** | **20** | **38** |

† LIVE-DYNAMIC: heuristic-zero-ref functions reclassified live (Qt/reportlab callbacks, registry refs). Full rows: `tmp/audit_group5_func_rows.md`.

### TRULY-DEAD (Group 5)

| Location | Function |
|----------|----------|
| `photometry_report.py:522` | `_PhotometryReportBuilder._lunar_risk_fill_color` |
| `photometry_report.py:1874` | `_PhotometryReportBuilder._katalogy_cell_for_pdf` |
| `export_reports.py:106` | `_observer_location_configured` |
| `export_reports.py:387` | `_test_is_eclipsing` |
| `export_reports.py:494` | `_comp_quality_map_for_export` |

### Test-gap list (Group 5 — deliverable-critical)

| Module / function | Existing test | Gap |
|-------------------|---------------|-----|
| `export_lightcurve_reports` / AAVSO row assembly | `test_gs11_pipeline` (notes suffix only) | No pytest round-trip: full LC row mag/err/filter/BJD/airmass column order vs AAVSO Extended spec |
| `_select_export_lc_rows` + `mag_calib_ac` | none | AC-on export must pair `mag_calib_ac` with correct uncertainty — untested |
| `catalog_only` / `forced_aperture` export | `test_phase2a_catalog_only_routing` (summary only) | No test that AAVSO/VarAstro labels or excludes forced-aperture LCs |
| RED-trust export row | `test_trust_flag` (formatters only) | No test that `trust=` appears in AAVSO NOTES for RED targets with correct reason text |
| `_generate_candidate_lc_png` | none | Candidate LC figure uses `mag_inst` — no test asserting calibrated mag |
| `_is_eclipsing` / VarAstro gate | inline `_test_is_eclipsing` (dead) | Eclipsing-type filter not in pytest |
| `pdf_report.generate_report` | `test_report_methods` (paths only) | No PDF byte/layout test; overflow verify not in CI |
| `report_methods` path helpers | `test_report_methods.py` | Path naming only — no export content |
| `jd_axis_format` | none | Display offset logic untested (low risk — not export column path) |
| `emit_export_citation_lines` | `test_export_citations.py` | Citation presence only — not full VarAstro body reconciliation |

### Reproducibility scan (Group 5)

| Location | Issue | Severity |
|----------|-------|----------|
| `export_reports._bjd_to_datestr_yyyymmdd` | BJD(TDB) → UTC calendar for filename date tag; edge-of-night boundary could shift date tag vs observer night | LOW |
| `export_reports` file naming | `safe_name` + first-LC `date_tag` — deterministic given LC order | - |
| `photometry_report` PDF | Reportlab paragraph wrap — deterministic given data | - |
| `citations.build_run_citation_context` | Config/meta driven — reproducible given same draft meta | - |
| `jd_axis_format` | Pure numeric offset for display | - |

### Per-module function registry (Group 5)

| Location | Function | Status |
|----------|----------|--------|
| `photometry_report.py:49` | `_norm_cid` | NEEDS-TEST |
| `photometry_report.py:63` | `_register_pdf_unicode_fonts` | NEEDS-TEST |
| `photometry_report.py:99` | `gs11_report_lines` | NEEDS-TEST |
| `photometry_report.py:5048` | `generate_photometry_report` | NEEDS-TEST |
| `photometry_report.py:5150` | `generate_all_method_photometry_reports` | NEEDS-TEST |
| `photometry_report.py:134` | `_PhotometryReportBuilder.__init__` | NEEDS-TEST |
| `photometry_report.py:397` | `_PhotometryReportBuilder._vsx_type_sort_rank` | NEEDS-TEST |
| `photometry_report.py:413` | `_PhotometryReportBuilder._try_load_variability_from_csv` | NEEDS-TEST |
| `photometry_report.py:464` | `_PhotometryReportBuilder._obs_date_str` | NEEDS-TEST |
| `photometry_report.py:481` | `_PhotometryReportBuilder._metric_color` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:490` | `_PhotometryReportBuilder._format_lc_count_display` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:512` | `_PhotometryReportBuilder._lunar_risk_fill_color` | TRULY-DEAD |
| `photometry_report.py:522` | `_PhotometryReportBuilder._draw_observing_conditions_section` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:710` | `_PhotometryReportBuilder._draw_gs11_dilution_section` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:723` | `_PhotometryReportBuilder._build_comp_pool_cover_rows` | NEEDS-TEST |
| `photometry_report.py:787` | `_PhotometryReportBuilder._load_variability_candidates_by_cid` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:808` | `_PhotometryReportBuilder._resolve_observer_identity` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:835` | `_PhotometryReportBuilder._resolve_plate_scale_arcsec` | FLAGGED(MED) |
| `photometry_report.py:865` | `_PhotometryReportBuilder._resolve_equipment_summary` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:889` | `_PhotometryReportBuilder._build_night_qc_summary` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:945` | `_PhotometryReportBuilder._build_target_lc_stats` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:982` | `_PhotometryReportBuilder._resolve_check_kname` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1002` | `_PhotometryReportBuilder._check_star_report_for` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1054` | `_PhotometryReportBuilder._ground_variability_line` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1086` | `_PhotometryReportBuilder._variability_edge_filter_note` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1113` | `_PhotometryReportBuilder._variability_cover_rows` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1129` | `_PhotometryReportBuilder._compress_image_for_pdf` | LIVE-DYNAMIC (attribute call, symbol ref) |
| `photometry_report.py:1196` | `_PhotometryReportBuilder._compress_png_bytes_for_pdf` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1236` | `_PhotometryReportBuilder._prepare_jpeg` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1269` | `_PhotometryReportBuilder._plot_lightcurve_to_jpeg` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1338` | `_PhotometryReportBuilder._robust_rms_148mad` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1347` | `_PhotometryReportBuilder._load_lc_xy_from_csv` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1376` | `_PhotometryReportBuilder._overlay_lc_cache_fresh` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1387` | `_PhotometryReportBuilder._plot_lightcurve_overlay_to_jpeg` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1490` | `_PhotometryReportBuilder._resolve_primary_lc_image` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1520` | `_PhotometryReportBuilder._page_footer` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1531` | `_PhotometryReportBuilder._layout_y_floor` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1535` | `_PhotometryReportBuilder._record_overflow` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1540` | `_PhotometryReportBuilder.overflow_violation_count` | NEEDS-TEST |
| `photometry_report.py:1543` | `_PhotometryReportBuilder._bounds_check` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1554` | `_PhotometryReportBuilder._layout_page_break` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1560` | `_PhotometryReportBuilder._layout_ensure_space` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1566` | `_PhotometryReportBuilder._get_para_style` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1590` | `_PhotometryReportBuilder._pdf_escape` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1593` | `_PhotometryReportBuilder._pdf_break_long` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1599` | `_PhotometryReportBuilder._pdf_id_display` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1609` | `_PhotometryReportBuilder._para_row_height` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1616` | `_PhotometryReportBuilder._draw_paragraph_block` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1639` | `_PhotometryReportBuilder._draw_flow_lines` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1670` | `_PhotometryReportBuilder._variability_cover_metrics` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1725` | `_PhotometryReportBuilder._draw_image_fit` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1760` | `_PhotometryReportBuilder._draw_kv_table_section` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1843` | `_PhotometryReportBuilder._sanitize_katalogy_pdf_line` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1850` | `_PhotometryReportBuilder._katalogy_positive_lines` | NEEDS-TEST |
| `photometry_report.py:1864` | `_PhotometryReportBuilder._katalogy_cell_for_pdf` | TRULY-DEAD |
| `photometry_report.py:1875` | `_PhotometryReportBuilder._katalogy_row_has_positive` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:1877` | `_PhotometryReportBuilder._draw_hockey_stick_png` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:2189` | `_PhotometryReportBuilder._report_cover_page` | NEEDS-TEST |
| `photometry_report.py:2268` | `_PhotometryReportBuilder._report_observation_summary` | NEEDS-TEST |
| `photometry_report.py:2526` | `_PhotometryReportBuilder._draft_id_from_dirname` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:2537` | `_PhotometryReportBuilder._load_obs_files_for_obs` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:2596` | `_PhotometryReportBuilder._load_qc_metrics_for_obs` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:2624` | `_PhotometryReportBuilder._compute_masterstar_score` | NEEDS-TEST |
| `photometry_report.py:2655` | `_PhotometryReportBuilder._qa_fwhm_limit_px` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:2698` | `_PhotometryReportBuilder._qc_row_by_frame_index` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:2705` | `_PhotometryReportBuilder._qc_row_by_file_hint` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:2736` | `_PhotometryReportBuilder._masterstar_from_candidates_csv` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:2776` | `_PhotometryReportBuilder._match_qc_row_by_vy_header_metrics` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:2811` | `_PhotometryReportBuilder._resolve_masterstar_used_frame` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:2858` | `_PhotometryReportBuilder._report_fits_qa` | FLAGGED(MED) |
| `photometry_report.py:2978` | `_PhotometryReportBuilder._format_comp_catalog_id` | FLAGGED(MED) |
| `photometry_report.py:2997` | `_PhotometryReportBuilder._proc_csv_dir` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3006` | `_PhotometryReportBuilder._rms_p2p_from_quality_note` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3015` | `_PhotometryReportBuilder._comp_rms_p2p_map_from_proc` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3057` | `_PhotometryReportBuilder._comp_rms_p2p_map_for_target` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3090` | `_PhotometryReportBuilder._comp_rows_for_target` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3227` | `_PhotometryReportBuilder._should_trigger_tess_report` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3251` | `_PhotometryReportBuilder._get_candidate_row_pdf` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3290` | `_PhotometryReportBuilder._generate_candidate_lc_png` | LIVE-DYNAMIC (attribute call, symbol ref) |
| `photometry_report.py:3332` | `_PhotometryReportBuilder._draw_candidate_detail_page` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3383` | `_PhotometryReportBuilder._is_sparse_star_data` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3389` | `_PhotometryReportBuilder._draw_compact_star_block` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3431` | `_PhotometryReportBuilder._report_per_star_compact_page` | NEEDS-TEST |
| `photometry_report.py:3443` | `_PhotometryReportBuilder._draw_catalog_crossmatch_block` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3479` | `_PhotometryReportBuilder._draw_aperture_correction_block` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:3567` | `_PhotometryReportBuilder._report_per_star_page` | FLAGGED(MED) |
| `photometry_report.py:3896` | `_PhotometryReportBuilder._report_summary_table` | NEEDS-TEST |
| `photometry_report.py:4091` | `_PhotometryReportBuilder._report_psf_summary_section` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:4191` | `_PhotometryReportBuilder._report_hrd_page` | NEEDS-TEST |
| `photometry_report.py:4281` | `_PhotometryReportBuilder._report_field_map` | NEEDS-TEST |
| `photometry_report.py:4324` | `_PhotometryReportBuilder._find_hockey_stick_disk_png` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:4332` | `_PhotometryReportBuilder._report_hockey_stick` | NEEDS-TEST |
| `photometry_report.py:4372` | `_PhotometryReportBuilder._col_pick` | LIVE-DYNAMIC (attribute call) |
| `photometry_report.py:4377` | `_PhotometryReportBuilder._report_candidates_table` | FLAGGED(MED) |
| `photometry_report.py:4565` | `_PhotometryReportBuilder._report_tess_section` | FLAGGED(MED) |
| `photometry_report.py:4833` | `_PhotometryReportBuilder._report_abbreviations` | FLAGGED(MED) |
| `photometry_report.py:4900` | `_PhotometryReportBuilder.build_pdf` | NEEDS-TEST |
| `photometry_report.py:1158` | `_to_rgb_white_bg` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:1207` | `_to_rgb_white_bg` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:1892` | `_legacy_simple_plot` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:2073` | `_known_vsx_row` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:2310` | `_cover_obs_condition_rows` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:2629` | `_norm_inverse` | NEEDS-TEST |
| `photometry_report.py:2635` | `_norm_direct` | NEEDS-TEST |
| `photometry_report.py:3126` | `_fmt` | NEEDS-TEST |
| `photometry_report.py:3144` | `_cid_short` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:3897` | `_cell_txt` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:3900` | `_zone_row_fill` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4024` | `_row_height` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4381` | `_empty_candidates_page` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4404` | `_short` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4434` | `_katalogy_paragraph_source_lines` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4449` | `_kat_cell` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4459` | `_kat_row_h_pts` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4570` | `_vsx_display_for_cid` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4580` | `_sector_sort_key` | LIVE-DYNAMIC (symbol ref) |
| `photometry_report.py:4587` | `_rel_color` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4597` | `_fmt_metric` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4690` | `_fmt_period_cell` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4694` | `_tess_blend_tail_h` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4701` | `_sector_block_h` | LIVE-DYNAMIC (direct call, symbol ref) |
| `photometry_report.py:4706` | `_tess_period_analysis_table` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:41` | `_resolved_site_from_meta` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:81` | `_site_coords` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:106` | `_observer_location_configured` | TRULY-DEAD |
| `export_reports.py:117` | `_append_aavso_observer_location_lines` | FLAGGED(MED) |
| `export_reports.py:129` | `_append_varastro_site_line` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:141` | `_vyvar_export_citation_lines` | FLAGGED(MED) |
| `export_reports.py:165` | `_varastro_alg_lines` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:182` | `_safe_filename` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:190` | `_bjd_to_datestr_yyyymmdd` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:201` | `_fmt_opt_num` | FLAGGED(MED) |
| `export_reports.py:209` | `_aavso_gs11_notes_suffix` | FLAGGED(MED) |
| `export_reports.py:222` | `_fmt_opt_int` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:273` | `_filter_lookup_key` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:278` | `_resolve_aavso_filter` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:313` | `_guess_setup_info_from_obs_group` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:342` | `_token_is_eclipsing` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:354` | `_is_eclipsing` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:387` | `_test_is_eclipsing` | TRULY-DEAD |
| `export_reports.py:450` | `_select_check_star` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:465` | `_copy_field_image` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:494` | `_comp_quality_map_for_export` | TRULY-DEAD |
| `export_reports.py:509` | `_normalize_comp_df_export_columns` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:526` | `_export_comp_status_label` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:554` | `_format_varastro_comp_table` | LIVE-DYNAMIC (direct call, symbol ref) |
| `export_reports.py:603` | `_select_export_lc_rows` | FLAGGED(MED) |
| `export_reports.py:643` | `export_lightcurve_reports` | FLAGGED(MED) |
| `export_reports.py:1022` | `export_all_method_lightcurve_reports` | NEEDS-TEST |
| `pdf_report.py:15` | `generate_report` | NEEDS-TEST |
| `report_methods.py:16` | `have_psf_frame_columns` | NEEDS-TEST |
| `report_methods.py:26` | `active_report_methods` | NEEDS-TEST |
| `report_methods.py:42` | `multi_method_reports_active` | NEEDS-TEST |
| `report_methods.py:46` | `lc_csv_path` | NEEDS-TEST |
| `report_methods.py:55` | `aavso_export_path` | NEEDS-TEST |
| `report_methods.py:71` | `varastro_export_path` | NEEDS-TEST |
| `report_methods.py:86` | `pdf_report_path` | NEEDS-TEST |
| `report_methods.py:105` | `report_title` | NEEDS-TEST |
| `report_methods.py:113` | `software_method_label` | NEEDS-TEST |
| `citations.py:28` | `_strip_bib_value` | LIVE-DYNAMIC (direct call, symbol ref) |
| `citations.py:70` | `load_citations_bib` | NEEDS-TEST |
| `citations.py:98` | `citation_line` | NEEDS-TEST |
| `citations.py:134` | `_vsx_db_configured` | LIVE-DYNAMIC (direct call, symbol ref) |
| `citations.py:141` | `_targets_use_vsx_names` | LIVE-DYNAMIC (direct call, symbol ref) |
| `citations.py:162` | `_lc_method_implies_psf` | LIVE-DYNAMIC (direct call, symbol ref) |
| `citations.py:167` | `build_run_citation_context` | FLAGGED(MED) |
| `citations.py:236` | `load_pipeline_meta` | NEEDS-TEST |
| `citations.py:248` | `_sections_for_context` | LIVE-DYNAMIC (direct call, symbol ref) |
| `citations.py:382` | `emit_export_citation_lines` | NEEDS-TEST |
| `citations.py:404` | `emit_pdf_methods_sections` | NEEDS-TEST |
| `citations.py:412` | `emit_varastro_method_summary_lines` | NEEDS-TEST |
| `jd_axis_format.py:16` | `jd_series_relative` | FLAGGED(MED) |
| `jd_axis_format.py:34` | `jd_axis_title` | FLAGGED(MED) |

### Automation artifacts (Group 5, tmp/, gitignored)

- `tmp/audit_group5.py` — scan driver
- `tmp/audit_group5_results.json` — inventory + lens hits
- `tmp/audit_group5_func_rows.md` — function rows (174)

**Checkpoint policy:** Milan + Claude review Group 5 batch before Group 6.

---

## Group 6 checkpoint — Config / orchestration / utils (2026-06-19)

**Module set (12 modules, 110 functions, ~4.9k LOC audited):**

| Module | Role |
|--------|------|
| `config.py` | `AppConfig` dataclass + JSON load (`__post_init__`), density/crowding override helpers |
| `utils.py` | WCS/plate-scale, FITS discovery, session paths, `seeded_numpy_default_rng` |
| `night_run.py` | Headless night pipeline orchestration |
| `infolog.py` | In-process log ring buffer + logging handler |
| `vyvar_ui_status.py` | Footer status strings / BV column visibility helper |
| `inspect_drafts.py` | CLI draft summary (plate scale hints) |
| `lunar_context.py` | Lunar phase / separation for reports |
| `orchestrator/vyvar_orchestrator.py` | Claude↔Cursor file bridge (dev) |
| `scripts/_build_vyvar_params.py` | `VYVAR_PARAMS.md` registry generator |
| `simulate_night_run.py` | Night-run simulation CLI |
| `run_crowding_index.py` | Crowding index CLI |
| `run_smoothness_report.py` | Smoothness report CLI |

**Excluded (audited in prior groups):** `param_resolver.py`, `time_utils.py`, `draft_provenance.py`, `fits_suffixes.py`, `masterstar_context.py` (Group 3); `app.py` + `ui_*.py` shells (Group 7 scope — scanned for L7 parity only).

Method: AST inventory + lens scans (L1–L11) on 12 modules (110 functions), `__post_init__` science read, cross-check `docs/VYVAR_PARAMS.md` + live `config.json` vs `AppConfig` (script `tmp/audit_group6_parity.py`).

### Prioritized findings (Group 6 — deduplicated, severity-sorted)

| ID | Sev | Lens | Location | What's wrong | Principle (not fix) |
|----|-----|------|----------|--------------|---------------------|
| G6-F001 | **HIGH** | L1/L2 | `config.py:439-441`, `540-541`, `211`, `1695-1702` | **Set-1 rig literals** as global defaults: `plate_scale_arcsec_per_px=1.3`, `phase01_plate_scale_arcsec_per_px=1.3`, `frame_width_px=2082`, `frame_height_px=1397`, `export_arcsec_per_px=1.3`. Wide-field (9.77″/px) requires explicit `config.json` overrides (TODO-MULTISET). Export path now derive-or-None (`G5-F007`) but config defaults still imply fine rig. | Cross-rig constants must not masquerade as universal defaults; derive-or-None or per-rig profiles. |
| G6-F002 | **FIXED** | L4/L7 | `config.py`, `database.py`, `config.json` | ~~Calibration validity default mismatch (80/524 vs 60/200)~~ **FIXED:** single source **90/200** (dataclass, `__post_init__` fallback, DB seed, tracked json). **Verify:** DB `SETTINGS` seed is **vestigial** — `get_setting_int` has **zero callers**; live path is `cfg.masterdark_validity_days` / `cfg.masterflat_validity_days` → `importer.py`. | Default in dataclass, `__post_init__`, PARAMS, and UI must agree. |
| G6-F003 | **HIGH** | L7 | `ui_select_stars.py:443`, `531`; `config.py` (no field) | UI reads **`cfg.phase01_comparison_max_bv_diff`** — **not an `AppConfig` field** (`hasattr` False). Comp-selection display / `max_bv_diff=` call will **AttributeError** when that path runs on a stock config. | Every UI `cfg.*` access must map to a declared config field or safe resolver. |
| G6-F004 | **HIGH** | L7 | `ui_aperture_photometry.py:1657`, `ui_select_stars.py:618`; `config.py` (no field) | UI uses **`getattr(cfg, "phase01_use_bprp_primary", True)`** — not on `AppConfig`; always defaults True; cannot be persisted or tuned via config/PARAMS registry. | Hidden UI behavior via getattr bypass breaks config↔UI parity. |
| G6-F005 | **MED** | L3 | `config.py:19-26` | `load_config_json` returns `{}` on `JSONDecodeError` with **no log** — corrupt `config.json` looks like “no overrides”. | Config load failures must log at operator-visible level. |
| G6-F006 | **MED** | L3 | `config.py:1045-1046` | Observer location DB hydration (`get_observer_location_by_id`) on broad **`except: pass`** — bad DB/path leaves stale lat/lon silently. | DB failures on site resolution must log. |
| G6-F007 | **MED** | L3/L4 | `config.py:666-667`, `1148-1149`, `2291-2292` | Silent **`pass`** on calibration temp tolerance parse, aperture tier parse, annulus crowding clamp — invalid JSON values retain prior silently. | Invalid config values should log + clamp explicitly. |
| G6-F008 | **MED** | L1/L11 | `config.py:74-75`, `utils.py:408-417` | `recommended_vyvar_parallel_workers` RAM estimate hardcodes **2048×2048** frames; `per_frame_catalog_match_sep_arcsec_for_scale` uses **20″** fallback when scale unknown. | Worker/RAM and match tolerances should use cfg geometry or derive-or-None. |
| G6-F009 | **MED** | L7 | `config.json` key `phase2a_variable_xy_fallback_mag_tol`; not in `AppConfig` | Orphan JSON key — edits have **no effect** on pipeline. | Every `config.json` key must bind to `AppConfig` or be removed. |
| G6-F010 | **MED** | L3 | `infolog.py:35-37`, `night_run.py:329-336` | Infolog `log_event` swallows all exceptions; `night_run` pending DAO FWHM/threshold parse failures **`pass`** without log. | Orchestration glue must not hide operator-actionable parse failures. |
| G6-F011 | **MED** | L7 | `docs/VYVAR_PARAMS.md` summary | Registry reports **34 config-only (no UI)** keys marked `exposed \| no` — drift bucket (observer site fields, blind cluster tuning, neighbor-sub PSF knobs, etc.). Not all are intentional-hidden; PARAMS generator marks `no` when no `ui*.py` string match. | Config-only knobs need explicit intentional-hidden vs drift classification. |
| G6-F012 | **LOW** | L5 | `inspect_drafts.py:46-109` | Diagnostic CLI: multiple **`except: pass`** on FITS/header reads (acceptable for CLI probe, but errors invisible). | Even diagnostics should print skip reason. |
| G6-P001 | **CLEAN** | - | `utils.py:24-48`, `VYVAR_RANDOM_SEED=42` | `seeded_numpy_default_rng` patches astroalign nondeterminism — aligns with Group 1 fix `98de910`. | Reproducibility hooks belong in shared utils. |
| G6-P002 | **CLEAN** | - | `utils.py:382-396`, `477-516` | `plate_scale_arcsec_per_pixel` + `catalog_cone_radius_deg_from_optics` derive cone from optics with `MIN_GAIA_CONE_RADIUS_DEG` floor — correct derive-or-fallback pattern. | Scale-dependent geometry should derive from optics/WCS. |
| G6-P003 | **CLEAN** | - | `scripts/_build_vyvar_params.py` | Automated PARAMS registry from `config.py` + `ui*.py` scan — supports L7 parity workflow. | Registry tooling should stay in sync with `AppConfig` fields. |
| G6-P004 | **CLEAN** | - | `config.py:645-646` | `plate_solve_fov_deg` intentionally **not** read from JSON — resolved from FITS+DB per comment (matches DECISIONS FOV authority). | FOV hints must not be stale JSON literals. |

### Config ↔ UI parity mismatch list (L7 — concrete)

**Registry snapshot** (`tmp/audit_group6_parity.py`, 2026-06-19): `AppConfig` fields **278** · `config.json` keys **252** · UI `cfg.*` refs **95** (16 `ui*.py` files) · `VYVAR_PARAMS.md` keys **259**.

**Summary (from `VYVAR_PARAMS.md`):** 82 exposed · 136 intentionally-hidden · **34 config-only (`exposed \| no`)** · **2 UI strings without `AppConfig` field**.

#### A — UI references without `AppConfig` field (reverse drift)

| Symbol | UI location | Risk |
|--------|-------------|------|
| `phase01_comparison_max_bv_diff` | `ui_select_stars.py:443`, `531` | **AttributeError** on direct `cfg.` access |
| `phase01_use_bprp_primary` | `ui_aperture_photometry.py:1657`, `ui_select_stars.py:618` | Silent default via `getattr(..., True)` only |

#### B — `config.json` key not on `AppConfig`

| Key | Note |
|-----|------|
| `phase2a_variable_xy_fallback_mag_tol` | Present in repo `config.json`; no loader field |

#### C — Dataclass vs `__post_init__` vs PARAMS default mismatches (selected)

| Key | Dataclass default | `__post_init__` if JSON missing | PARAMS / UI |
|-----|-------------------|----------------------------------|-------------|
| `masterdark_validity_days` | 80 | **60** | 80 (UI yes) |
| `masterflat_validity_days` | 524 | **200** | 524 (UI yes) |
| `comp_iterative_clip_enabled` | False | loaded from JSON (often True) | PARAMS notes both |
| Observer block (`observer_lat/lon/…`) | Jirny-ish literals | DB/json session values | PARAMS shows dataclass vs json drift |

#### D — Config-only keys (`VYVAR_PARAMS.md` `exposed \| no`, 33 rows)

`aavso_filter_map`, `observer_alt_m`, `observer_code`, `observer_lat`, `observer_location_id`, `observer_location_name`, `observer_lon`, `observer_name`, `apply_color_term`, `blind_cluster_coherence_cap`, `blind_cluster_eps_deg`, `blind_cluster_min_samples`, `blind_cluster_min_votes`, `blind_cluster_vote_span`, `blind_img_select_mode`, `blind_img_star_budget`, `blind_scale_tol_frac`, `blind_use_rig_prior`, `check_select_rms_floor`, `neighbor_sub_centroid_max_fwhm`, `neighbor_sub_chi2_max`, `neighbor_sub_max_neighbor_overmag`, `neighbor_sub_max_target_undermag`, `neighbor_sub_min_recovered_snr`, `neighbor_sub_nn_contam_dmag`, `neighbor_sub_refuse_sep_fwhm`, `neighbor_sub_regime_dmag_min`, `neighbor_sub_regime_sep_max`, `neighbor_sub_residual_rms_max`, `psf_neighbor_sub_enabled`, `blind_index_select_mode`, `catalog_query_max_rows`, `dao_qc_in_calibrate`.

(Plus `export_arcsec_per_px` marked intentionally-hidden in PARAMS — superseded in export code by derive-or-None but config default remains 1.3.)

### Coverage table (Group 6 modules)

| Module | Lines | Funcs | Audited | DEAD (heuristic) | TRULY-DEAD | LIVE-DYNAMIC† | FLAGGED | NEEDS-TEST |
|--------|-------|-------|---------|------------------|------------|---------------|---------|------------|
| `config.py` | 2294 | 19‡ | 19 | 0 | 0 | 0 | 9 | 10 |
| `utils.py` | 736 | 34 | 34 | 0 | 0 | 2§ | 6 | 14 |
| `night_run.py` | 1128 | 19 | 19 | 0 | 0 | 2§ | 4 | 9 |
| `infolog.py` | 138 | 10 | 10 | 0 | 0 | 0 | 2 | 5 |
| `vyvar_ui_status.py` | 90 | 4 | 4 | 0 | 0 | 0 | 0 | 1 |
| `inspect_drafts.py` | 154 | 4 | 4 | 0 | 0 | 0 | 1 | 1 |
| `lunar_context.py` | 139 | 6 | 6 | 0 | 0 | 0 | 0 | 4 |
| `orchestrator/vyvar_orchestrator.py` | 206 | 6 | 6 | 0 | 0 | 0 | 0 | 1 |
| `scripts/_build_vyvar_params.py` | 578 | 4 | 4 | 0 | 0 | 0 | 0 | 2 |
| `simulate_night_run.py` | 164 | 2 | 2 | 0 | 0 | 0 | 0 | 1 |
| `run_crowding_index.py` | 86 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| `run_smoothness_report.py` | 61 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| **Group 6 total** | - | **110** | **110** | **0** | **0** | **4** | **22** | **49** |

‡ `AppConfig.__post_init__` (~1.6k lines) is the primary load surface — counted as part of `config.py` audit, not separate func row.

§ LIVE-DYNAMIC: nested closures in `resolve_draft_dir` / `night_run` progress callbacks — heuristic zero-ref, live via parent.

**Mechanism-aware DEAD:** Raw AST heuristic DEAD **0/110** — no removal candidates; config uses `_f01`/`_i01` helpers and density override tables (registry/dispatch). Do **not** apply Group-3-style 88% DEAD heuristic here.

### TRULY-DEAD (Group 6)

None confirmed. CLI `main()` entries and orchestrator helpers are entrypoints or dev tooling.

### Test-gap list (Group 6 — science-critical)

| Module / function | Existing test | Gap |
|-------------------|---------------|-----|
| `config.load_config_json` / corrupt JSON | — | `JSONDecodeError` → `{}` silent |
| `AppConfig.__post_init__` round-trip | partial via integration | validity-day defaults 60/200 vs dataclass 80/524 |
| `apply_density_overrides` / `apply_crowding_overrides` | — | additive override clamps |
| `recommended_vyvar_parallel_workers` | — | RAM cap vs cfg frame size |
| `utils.plate_scale_arcsec_per_pixel` | indirect | unit edge cases |
| `utils.per_frame_catalog_match_sep_arcsec_for_scale` | — | 20″ fallback when scale None |
| `utils.seeded_numpy_default_rng` | Group 1 alignment tests | direct unit test optional |
| `night_run.run_night_pipeline` error paths | smoke scripts | headless failure surfacing |
| `infolog.save_infolog_to_disk` | — | disk write failure path |
| `lunar_context.get_lunar_context` | `tests/test_lunar_context.py` | partial coverage |

Inventory: **49 NEEDS-TEST** + **22 FLAGGED** across Group 6 (100% statused at module level).

### Reproducibility scan (Group 6)

| Area | Finding |
|------|---------|
| RNG | `utils.VYVAR_RANDOM_SEED=42`; `seeded_numpy_default_rng` for astroalign |
| Parallelism | `recommended_vyvar_parallel_workers` uses `os.cpu_count()` + **psutil** free RAM — run-dependent worker cap |
| Config authority | `AppConfig()` always merges `config.json` (session observer site, equipment-tuned overrides) — not a pure dataclass default |
| Orchestrator | `vyvar_orchestrator.py` local timestamps; no science RNG |
| Night run | Delegates to `pipeline` / Group 1 reproducibility fixes |

### Automation artifacts (Group 6, tmp/, gitignored)

- `tmp/audit_group6_parity.py` — config↔UI↔json parity driver
- `tmp/audit_group6_parity.json` — full parity dump (189 keys with no `ui*.py` ref)

**Checkpoint policy:** Milan + Claude review Group 6 batch before Group 7.

---

## Group 7 checkpoint — UI shell (2026-06-19)

**Module set (14 modules, 144 functions, ~9.8k LOC audited):**

| Module | Funcs | Lines | Notes |
|--------|-------|-------|-------|
| `app.py` | 29 | 2733 | Main Streamlit shell, pipeline pending orchestration |
| `ui_aperture_photometry.py` | 25 | 1800 | Phase 2A UI, LC plots, report triggers |
| `ui_quality_dashboard.py` | 14 | 1072 | QC / analyze workflow |
| `ui_select_stars.py` | ~~13~~ **deleted** | unwired legacy page removed 2026-06-22 |
| `ui_calibration_library.py` | 12 | 415 | Calibration library browser |
| `ui_calibration.py` | 10 | 306 | Calibration import UI |
| `ui_photometry_quality.py` | 10 | 380 | Photometry QC views |
| `ui_epsf_dashboard.py` | 7 | 374 | ePSF dashboard |
| `ui_finalization.py` | 7 | 506 | Export / finalization |
| `ui_database_explorer.py` | 6 | 375 | DB explorer |
| `ui_dao_stars.py` | 4 | 370 | DAO / MASTERSTAR controls |
| `ui_components.py` | 3 | 212 | Shared MASTERSTAR candidate widget |
| `ui_photometry.py` | 2 | 108 | Photometry tab router |
| `ui_settings.py` | 2 | 1019 | Settings save (`render_settings` body is one large function) |

**Excluded (audited in Group 4):** `ui_variability.py` (30), `ui_hrd.py` (1), `ui_masterstar_qa.py` (5). **Excluded (Group 6):** `vyvar_ui_status.py`.

Method: AST inventory + lens scans (L1–L11) on 14 modules (144 functions), Phase B verification of G6-F003/F004 from UI source, parity cross-check vs `tmp/audit_group6_parity.json` + `VYVAR_PARAMS.md`.

### Prioritized findings (Group 7 — deduplicated, severity-sorted)

| ID | Sev | Lens | Location | What's wrong | Principle (not fix) |
|----|-----|------|----------|--------------|---------------------|
| G7-F001 | **RESOLVED** | L7 | ~~`ui_select_stars.py`~~ **removed** (`refactor` dead unwired page) | ~~`cfg.phase01_comparison_max_bv_diff` AttributeError~~ **RESOLVED:** unwired `ui_select_stars.py` deleted; phantom field no longer referenced repo-wide. | Every `cfg.*` in UI must be a declared field or guarded accessor. |
| G7-F002 | **RESOLVED** | L8/L7 | ~~`ui_select_stars.py:531`~~ **removed** | ~~Stale `max_bv_diff=` kwarg to `run_phase0_and_phase1`~~ **RESOLVED:** deleted with unwired Select Stars page. | UI must call core with the live signature; dead kwargs hide broken comp-color paths. |
| G7-F003 | **MED** | L7 | `ui_aperture_photometry.py:1657` | **`phase01_use_bprp_primary`** via `getattr(cfg, …, True)` only — **no crash**, but **non-persistable** (not on `AppConfig`, not in PARAMS). **STILL OPEN** — defer to config↔UI parity fix-pass. | getattr defaults bypass config registry and mislead operators. |
| G7-F004 | **MED** | L3 | `app.py` (15+ `except: pass` lines), `ui_aperture_photometry.py`, `ui_quality_dashboard.py` | Broad **`except: pass`** / silent exception branches in action handlers (archive hash, PDF/report triggers, QC paths) — failures can present as **button did nothing**. | UI actions must surface errors to Infolog / `st.error`. |
| G7-F005 | **MED** | L4 | `ui_settings.py:645-700` | Frame-quality / align-residual gates use **`getattr(..., False)`** defaults — **fail-open OFF by design** (byte-identical when unset); documented in PARAMS Round-2 B.2. Not a bug; monitor that toggles stay wired to `cfg` save path (`973-976`). | Safety gates default OFF only when explicitly documented. |
| G7-F006 | **MED** | L7 | `VYVAR_PARAMS.md`, Group 6 parity | **34 config-only (`exposed \| no`)** keys: **intentionally hidden** for blind-cluster tuning, neighbor-sub PSF, `dao_qc_in_calibrate`, observer export block (session `config.json` values). **Not accidental unexposed** for most; observer site uses DB location picker (`observer_location_id` in Settings) while lat/lon/name live in json without dedicated widgets. | Hidden keys need intentional-hidden classification vs drift. |
| G7-F007 | **MED** | L8 | `ui_aperture_photometry.py:236+` | `_enrich_summary_with_zone_flags` / LC column picks (`mag_calib` vs `mag_calib_raw`) in UI — **display-layer** only (no ensemble math); risk is **label drift** vs export `mag_calib_final` (G5-F011), not duplicated photometry core. | Display columns must match publication canonical names. |
| G7-F008 | **LOW** | L3 | `ui_components.py:103-104`, `ui_aperture_photometry.py:77-78` | Silent passes on archive-path probe / JD tick helper (`try: pass` stub). Low operator impact. | Even minor UI helpers should log skip reason. |
| G7-P001 | **CLEAN** | - | `ui_settings.py:973-1010` | Settings save writes frame-quality + align-residual + comp knobs back to `cfg` + `to_json()` — live round-trip for exposed controls. | Settings must persist to `AppConfig` + json. |
| G7-P002 | **CLEAN** | - | `app.py` pipeline pending handlers | Photometry/platesolve/preprocess delegate to `pipeline` / `photometry_core` — **no** `ensemble_normalize` / `savgol` in UI shell (L8 scan clean on `app.py`). | UI orchestrates; core computes. |
| G7-P003 | **CLEAN** | - | `app.py` L1 scan | **No** hardcoded `1.3` / `9.77` / `2082` rig literals in UI shell (rig values come from `cfg` or DB). | UI must not embed cross-rig literals. |

### Config ↔ UI parity disposition (resolves G6-F003 / G6-F004)

| Symbol | Access pattern | UI verdict | Severity |
|--------|----------------|------------|----------|
| `phase01_comparison_max_bv_diff` | ~~`ui_select_stars.py`~~ **removed** | **RESOLVED** (G7-F001) — no repo references |
| `phase01_use_bprp_primary` | `ui_aperture_photometry.py:1657` | Silent default via `getattr(..., True)` only — **STILL OPEN** (G7-F003) |
| `max_bv_diff=` kwarg | ~~`ui_select_stars.py`~~ **removed** | **RESOLVED** (G7-F002) |
| 34 PARAMS `no` keys | No `ui*.py` string match | Mostly **intentionally-hidden** dev/json knobs; observer block is **session json + DB location id** | Documented (G7-F006) |

False positives from regex scan (`cfg.to_json`, `cfg.ensure_base_dirs`) are **methods** on `AppConfig`, not missing fields.

### Coverage table (Group 7 modules)

| Module | Funcs | Audited | TRULY-DEAD | LIVE-DYNAMIC† | FLAGGED | NEEDS-TEST |
|--------|-------|---------|------------|---------------|---------|------------|
| `app.py` | 29 | 29 | 0 | 4 | 8 | 12 |
| `ui_aperture_photometry.py` | 25 | 25 | 0 | 3 | 6 | 10 |
| `ui_quality_dashboard.py` | 14 | 14 | 0 | 2 | 5 | 8 |
| `ui_select_stars.py` | — | — | — | — | — | — | **deleted** (2026-06-22) |
| `ui_calibration_library.py` | 12 | 12 | 0 | 0 | 1 | 4 |
| `ui_calibration.py` | 10 | 10 | 0 | 0 | 0 | 3 |
| `ui_photometry_quality.py` | 10 | 10 | 0 | 1 | 2 | 4 |
| `ui_epsf_dashboard.py` | 7 | 7 | 0 | 0 | 1 | 3 |
| `ui_finalization.py` | 7 | 7 | 0 | 1 | 1 | 3 |
| `ui_database_explorer.py` | 6 | 6 | 0 | 0 | 0 | 3 |
| `ui_dao_stars.py` | 4 | 4 | 0 | 0 | 1 | 2 |
| `ui_components.py` | 3 | 3 | 0 | 0 | 1 | 1 |
| `ui_photometry.py` | 2 | 2 | 0 | 0 | 0 | 2 |
| `ui_settings.py` | 2 | 2 | 0 | 0 | 1 | 3 |
| **Group 7 total** | **131** | **131** | **0** | **11** | **26** | **48** |

† LIVE-DYNAMIC: Streamlit callbacks, nested render closures, `@st.cache_data` wrappers — heuristic zero-ref, not removal candidates.

**Mechanism-aware DEAD:** **0 TRULY-DEAD** / 144 (Group 4 already logged 2 dead symbols in `ui_variability.py`).

### Test-gap list (Group 7 — UI-critical)

| Area | Gap |
|------|-----|
| `ui_settings` save → `config.json` round-trip | partial manual; no automated test |
| `ui_select_stars` Phase 0+1 launch | ~~broken~~ | **removed** — unwired page deleted |
| Frame-quality gate toggles | default OFF regression (byte-identical) |
| `app.py` pending preprocess/platesolve error surfacing | headless parity with `night_run` |
| QC dashboard analyze flow | exception → user-visible error |

Inventory: **52 NEEDS-TEST** + **29 FLAGGED** across Group 7.

### Reproducibility scan (Group 7)

| Area | Finding |
|------|---------|
| Session state | `st.session_state` keys (`vyvar_draft_dir_override`, variability caches) — run-order / draft dependent |
| Draft override | Effective paths from session override vs `cfg.archive_root` |
| No UI RNG | Science RNG remains in core/utils (Group 1/6) |

### Automation artifacts (Group 7, tmp/, gitignored)

- `tmp/audit_group7_inventory.py` — scan driver
- `tmp/audit_group7_results.json` — per-module lens + cfg ref dump

**Checkpoint policy:** Group 7 closes the systematic map — G1-F001/F002 closed after Chi/h validation PASS.

---

## 7-group map complete (2026-06-19)

| Group | Scope | Funcs audited | HIGH findings (open at map close) |
|-------|--------|---------------|-----------------------------------|
| 1 | Alignment / platesolve / optics | 415 | G1-F003 (+ MED alignment identity fallback); G1-F001/F002 **FIXED** |
| 2 | Photometry core | 322 | G2-F004 (+ G2-F002b backlog); G2-F003 **FIXED** |
| 3 | Data / IO / catalog | 367 | G3-F002 (+ G3-F001 **FIXED**) |
| 4 | Science / variability / QA | 166 | G4-F001 trust (partially **FIXED**) |
| 5 | Reporting / export | 174 | G5-F004 **FIXED**; remaining LOW dead helpers |
| 6 | Config / orchestration / utils | 110 | G6-F001–F004 config/parity |
| 7 | UI shell | ~~144~~ **131** | G7-F001/F002 **RESOLVED** (Select Stars removed); G7-F003 open |
| **Total** | **7 groups** | **1728** | Fix-pass queue starts after Claude review |

---

## Post-map fix-pass queue (not started)

Per roadmap after map close: G1-F001/F002 (alignment caps), G3-F002, G6 config items (validity defaults, `max_bv_diff` / `phase01_use_bprp_primary`, orphan json keys).
