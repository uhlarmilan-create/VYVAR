# VYVAR -- Config <-> UI parameter registry

<!-- GENERATED FILE -- DO NOT EDIT BY HAND. -->
Regenerate with `python tools/gen_params_md.py`. Hand edits will be overwritten.
Source: `validation/params_registry.json` (editorial metadata) + `dataclasses.fields(AppConfig)` (defaults and types, from code).
Human-readable guide: `VYVAR_CONFIG_GUIDE_EN.md` / `VYVAR_CONFIG_GUIDE_CZ.md` (per-parameter plain-language explanations, hand-authored).
In-depth handbook: `VYVAR_PARAMETER_HANDBOOK_CZ.pdf` (Czech; per-parameter reasoning, ranges, math and literature; regenerate with `python dev/tools/docs_pdf/build_parameter_handbook.py`).

_Generated 2026-10-21T20:08:37Z at git HEAD 7c5dfb9._

## Summary

- Entries: 127
- Tier: basic 13, advanced 39, expert 75
- Kind: static 119, derived 0, resolved 8
- Widget: auto 107, custom 15, hidden 5
- Owner: db_static 5, config_runtime 108, fits_dynamic 5, internal 9
- Scope: universal 100, rig 14, site 9, session 4
- Scope key: none 100, rig 7, rig_band 2, rig_sampling 5, site 9, frame 4
- Rig triage group: a 8, b 3, c 3

Columns: key, default, range, tier, kind, owner, scope, scope_key, scope_group, widget, label. `kind=resolved` means the runtime value can be auto-derived/overridden by the pipeline (the configured value is the base/fallback). `owner` is the storage-and-ownership axis: `db_static` (DB reference tables), `config_runtime` (user-tuned config.json), `fits_dynamic` (resolved from FITS/WCS at run time), `internal` (plumbing). `widget=custom` keys keep their hand-built UI; `widget=hidden` keys are plumbing not surfaced in the generated dashboard.

## observer

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `aavso_filter_map` | {} | - | expert | static | config_runtime | site | site | n/a | auto | AAVSO filter names |
| `observer_alt_m` | 275.0 | - | basic | static | db_static | site | site | n/a | auto | Observer Alt M |
| `observer_code` |  | - | basic | static | config_runtime | site | site | n/a | auto | AAVSO observer code |
| `observer_lat` | 50.1121658 | - | basic | static | db_static | site | site | n/a | auto | Observer Lat |
| `observer_location_id` | 0 | - | expert | static | db_static | site | site | n/a | hidden | Observing site (run time) |
| `observer_location_name` |  | - | basic | static | db_static | site | site | n/a | auto | Observer Location Name |
| `observer_lon` | 14.6982547 | - | basic | static | db_static | site | site | n/a | auto | Observer Lon |
| `observer_name` | Unknown Observer | - | basic | static | config_runtime | site | site | n/a | auto | Observer name |

## paths

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `archive_root` | (resolved at runtime) | - | expert | static | internal | universal | none | n/a | auto | Archive |
| `astrometry_index_dir` |  | - | advanced | static | config_runtime | universal | none | n/a | auto | Blind plate-solve index |
| `calibration_library_root` | (resolved at runtime) | - | expert | static | internal | universal | none | n/a | auto | Calibration library |
| `database_path` | (resolved at runtime) | - | expert | static | internal | universal | none | n/a | auto | App database |
| `gaia_db_path` |  | - | expert | static | internal | universal | none | n/a | auto | Gaia DR3 catalog |
| `objects_db_path` | catalogs/vyvar_objects.db | - | expert | static | internal | universal | none | n/a | auto | Objects catalog (VSX + exoplanets) |
| `project_root` | (git toplevel) | - | expert | static | internal | universal | none | n/a | hidden | Project Root |

## calibration

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `calibration_master_ccd_temp_tolerance_c` | 0.5 | 0 .. 10 | expert | static | config_runtime | rig | rig | a | auto | Dark temperature tolerance |
| `masterdark_validity_days` | 90 | 1 .. 3650 | basic | static | config_runtime | rig | rig | c | auto | Master dark valid for |
| `masterflat_validity_days` | 200 | 1 .. 3650 | basic | static | config_runtime | rig | rig | c | auto | Master flat valid for |

## qc

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `auto_fwhm_enabled` | True | - | advanced | static | config_runtime | universal | none | n/a | auto | Night FWHM limit from the data |
| `auto_fwhm_k_factor` | 3.0 | 1 .. 4 | advanced | static | config_runtime | universal | none | n/a | auto | Night FWHM limit k |
| `lc_outlier_frame_qc_n_sigma` | 5.0 | 3 .. 10 | advanced | static | config_runtime | universal | none | n/a | auto | Bad-frame threshold |
| `osc_channel_binning` | 2 | 1 .. 4 | advanced | static | config_runtime | rig | rig | a | auto | OSC channel binning |
| `preprocess_sky_surface_force_reapply` | False | - | expert | static | config_runtime | universal | none | n/a | auto | Re-subtract sky surface |
| `preprocess_sky_surface_order` | 2 | 0 .. 2 | expert | static | config_runtime | universal | none | n/a | auto | Sky background surface order |
| `qc_after_calibrate_enabled` | True | - | basic | static | config_runtime | universal | none | n/a | auto | Quality check after calibration |

## alignment

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `alignment_max_control_points` | 80 | 12 .. 500 | expert | static | config_runtime | rig | rig_sampling | c | auto | Alignment control points |
| `alignment_max_stars` | 160 | 10 .. 5000 | advanced | static | config_runtime | universal | none | n/a | auto | Stars used for alignment |

## detection

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `dao_centroid_max_shift_fwhm` | 1.0 | 0.1 .. 5 | expert | static | config_runtime | universal | none | n/a | auto | Centroid shift limit |
| `detection_threshold_sigma` | 4.5 | 2 .. 20 | advanced | static | config_runtime | rig | rig_sampling | a | auto | Star detection threshold |
| `exoplanet_match_max_sep_arcsec` | 3.0 | 0.5 .. 30 | expert | static | config_runtime | universal | none | n/a | auto | Exoplanet host match radius |
| `frame_height_px` | 1397 | - | expert | resolved | internal | session | frame | n/a | hidden | Frame Height PX |
| `frame_width_px` | 2082 | - | expert | resolved | internal | session | frame | n/a | hidden | Frame Width PX |
| `masterstar_best_of_n` | 10 | 1 .. 25 | expert | static | config_runtime | universal | none | n/a | auto | MASTERSTAR candidates |
| `masterstar_dao_centroid_qa_cap_px` | 3.0 | - | expert | static | config_runtime | universal | none | n/a | custom | Masterstar DAO Centroid QA Cap PX |
| `masterstar_dao_centroid_qa_floor_px` | 1.0 | - | expert | static | config_runtime | universal | none | n/a | custom | Masterstar DAO Centroid QA Floor PX |
| `masterstar_dao_empty_sky_false_accept_max` | 0.01 | 0 .. 0.05 | expert | static | config_runtime | universal | none | n/a | custom | Masterstar DAO Empty Sky False Accept Max |
| `masterstar_dao_empty_sky_target_n` | 2200 | 500 .. 5000 | expert | static | config_runtime | universal | none | n/a | custom | Masterstar DAO Empty Sky Target N |
| `masterstar_dao_match_radius_k` | 1.7 | 0.5 .. 3 | expert | static | config_runtime | universal | none | n/a | custom | Masterstar DAO Match Radius K |
| `masterstar_dao_pass1_dedup_px` | 0.75 | 0.1 .. 2 | expert | static | config_runtime | universal | none | n/a | custom | Masterstar DAO Pass1 Dedup PX |
| `masterstar_dao_pass2_center_tol_px` | 2.0 | 0.5 .. 5 | expert | static | config_runtime | universal | none | n/a | custom | Masterstar DAO Pass2 Center Tol PX |
| `masterstar_dao_pass2_sigma` | 4.0 | - | expert | static | config_runtime | universal | none | n/a | custom | Masterstar DAO Pass2 Sigma |
| `masterstar_forced_seed_centroid_max_px` | 2.0 | 0.5 .. 5 | expert | static | config_runtime | universal | none | n/a | custom | Masterstar Forced Seed Centroid Max PX |
| `masterstar_forced_seed_comp_pool_enabled` | False | - | expert | static | config_runtime | universal | none | n/a | custom | Masterstar Forced Seed Comp Pool Enabled |
| `masterstar_forced_seed_snr_min` | 4.0 | 1 .. 10 | expert | static | config_runtime | universal | none | n/a | custom | Masterstar Forced Seed SNR Min |
| `masterstar_gaia_census_edge_margin_px` | 10.0 | 1 .. 30 | expert | static | config_runtime | universal | none | n/a | custom | Masterstar Gaia Census Edge Margin PX |
| `masterstar_lock_leftover_radius_px` | 3.0 | 0.5 .. 10 | expert | static | config_runtime | universal | none | n/a | custom | Masterstar Lock Leftover Radius PX |
| `masterstar_lock_pair_tol_px` | 3.0 | 0.5 .. 10 | expert | static | config_runtime | universal | none | n/a | custom | Masterstar Lock Pair Tol PX |
| `masterstar_use_best_frame_fwhm` | True | - | expert | static | config_runtime | universal | none | n/a | custom | Masterstar Use Best Frame FWHM |
| `phase01_chip_interior_margin_px` | 50 | 0 .. 500 | advanced | static | config_runtime | rig | rig_sampling | b | auto | Chip edge margin |
| `phase01_plate_scale_arcsec_per_px` | 1.3 | 0 .. 30 | expert | resolved | fits_dynamic | session | frame | n/a | auto | Phase01 Plate Scale Arcsec Per PX |
| `plate_scale_arcsec_per_px` | 1.3 | 0.1 .. 30 | expert | resolved | fits_dynamic | session | frame | n/a | auto | Plate Scale Arcsec Per PX |
| `plate_solve_fov_deg` | 1.0 | - | expert | resolved | fits_dynamic | rig | rig | a | auto | Plate Solve FOV Deg |
| `sips_dao_fwhm_px` | 2.5 | 1 .. 8 | advanced | static | config_runtime | rig | rig_sampling | b | auto | Fallback FWHM |
| `vsx_out_of_scope_types` | [] | - | basic | static | config_runtime | universal | none | n/a | auto | VSX types not measured |

## photometry

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `annulus_inner_fwhm` | 2.7 | 1 .. 10 | advanced | static | config_runtime | universal | none | n/a | auto | Sky annulus inner radius |
| `annulus_outer_fwhm` | 5.2 | 1.5 .. 12 | advanced | static | config_runtime | universal | none | n/a | auto | Sky annulus outer radius |
| `aperture_fwhm_factor` | 1.35 | 0.25 .. 6 | advanced | static | config_runtime | universal | none | n/a | auto | Aperture radius |
| `epsf_auto_run` | False | - | basic | static | config_runtime | universal | none | n/a | auto | Run ePSF in every night run (beta) |
| `epsf_export_auto` | True | - | basic | static | config_runtime | universal | none | n/a | auto | Write ePSF light-curve files |
| `epsf_min_stars` | 30 | 5 .. 500 | expert | static | config_runtime | universal | none | n/a | auto | ePSF minimum stars |
| `epsf_targets` | ALL | - | basic | static | config_runtime | universal | none | n/a | auto | ePSF targets |
| `err_empty_apertures_min` | 16 | 4 .. 1000 | expert | static | config_runtime | universal | none | n/a | auto | Empty apertures needed |
| `err_empty_apertures_n` | 64 | 8 .. 2000 | expert | static | config_runtime | universal | none | n/a | auto | Empty apertures per frame |
| `forced_photometry_enabled` | True | - | expert | static | config_runtime | universal | none | n/a | auto | Forced photometry |
| `forced_photometry_margin_px` | 0.0 | 0 .. 200 | expert | static | config_runtime | universal | none | n/a | auto | Forced photometry edge margin |
| `gain` | 1.0 | - | expert | resolved | fits_dynamic | rig | rig | a | auto | Gain |
| `gs11_dilution_aperture_arcsec` | 0.0 | 0 .. 120 | expert | static | config_runtime | universal | none | n/a | auto | Neighbour light aperture |
| `gs11_dilution_enabled` | False | - | expert | static | config_runtime | universal | none | n/a | auto | Neighbour light estimate |
| `gs11_dilution_mag_limit_delta` | 5.0 | 0.5 .. 15 | expert | static | config_runtime | universal | none | n/a | auto | Neighbour light depth |
| `gs11_target_min_dilution` | 0.5 | 0.01 .. 1 | expert | static | config_runtime | universal | none | n/a | auto | Blended target limit |
| `lc_high_err_nsigma` | 5.0 | 3 .. 10 | advanced | static | config_runtime | universal | none | n/a | auto | High-error point flag |
| `lc_outlier_adjacent_sigma` | 3.0 | 1 .. 8 | advanced | static | config_runtime | universal | none | n/a | auto | Outlier neighbour protection |
| `lc_outlier_enabled` | True | - | advanced | static | config_runtime | universal | none | n/a | auto | Flag isolated outliers |
| `lc_outlier_evidence_n_sigma` | 5.0 | 3 .. 10 | advanced | static | config_runtime | universal | none | n/a | auto | Outlier image evidence |
| `lc_outlier_n_sigma` | 5.0 | 3 .. 10 | advanced | static | config_runtime | universal | none | n/a | auto | Outlier threshold |
| `per_frame_sat_min_clean_frac` | 0.5 | 0.1 .. 1 | advanced | static | config_runtime | universal | none | n/a | auto | Unsaturated frames needed |
| `photon_transfer_ci_max_width_factor` | 3.0 | 1 .. 10 | expert | static | config_runtime | universal | none | n/a | auto | Measured gain acceptance |
| `psf_chi2_locus_nsigma` | 5.0 | 1 .. 20 | advanced | static | config_runtime | universal | none | n/a | auto | ePSF fit quality limit |
| `read_noise` | 10.0 | - | expert | resolved | fits_dynamic | rig | rig | a | auto | Read Noise |
| `save_lightcurve_png` | False | - | advanced | static | config_runtime | universal | none | n/a | auto | Save light-curve PNG files |

## comp_selection

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `comp_catalog_offset_max_mag` | 0.2 | 0 .. 2 | advanced | static | config_runtime | universal | none | n/a | auto | Comparison catalogue offset limit |
| `comp_color_tiers` | [{'bprp': 0.15}, {'bprp': 0.3}, {'bprp': 0.55}] | - | advanced | static | config_runtime | universal | none | n/a | auto | Colour classes |
| `comp_contamination_penalty_k` | 3.0 | 0 .. 20 | expert | static | config_runtime | universal | none | n/a | auto | Contamination weight penalty |
| `comp_isolation_fwhm` | 3.0 | 1 .. 12 | expert | static | config_runtime | universal | none | n/a | auto | Comparison isolation |
| `comp_max_delta_bprp` | 0.79 | 0 .. 5 | advanced | static | config_runtime | universal | none | n/a | auto | Largest colour difference |
| `comp_max_slope_mmag_hr` | 5.0 | 0 .. 500 | expert | static | config_runtime | universal | none | n/a | auto | Comparison drift limit |
| `comp_pool_derived_admission` | True | - | expert | static | config_runtime | universal | none | n/a | auto | Drop only known variables |
| `comp_rms_loo_photon_k` | 5.0 | 1 .. 50 | advanced | static | config_runtime | universal | none | n/a | auto | Comparison scatter vs photon noise |
| `comp_select_rms_floor` | 1e-06 | 0 .. 0.01 | expert | static | config_runtime | universal | none | n/a | auto | Comparison scatter floor |
| `comp_slope_significance_k` | 3.0 | 0 .. 10 | advanced | static | config_runtime | universal | none | n/a | auto | Comparison drift significance |
| `comparison_stars_pool_n` | 0 | 0 .. 50000 | expert | static | config_runtime | universal | none | n/a | auto | Comparison pool size |
| `phase01_comparison_isolation_radius_px` | 25.0 | 1 .. 200 | expert | static | config_runtime | rig | rig_sampling | b | auto | Contamination radius |
| `phase01_comparison_max_comp_rms` | 0.1 | 0.01 .. 0.5 | advanced | static | config_runtime | universal | none | n/a | auto | Comparison scatter limit |
| `phase01_comparison_min_dist_arcsec` | 60.0 | 0 .. 600 | advanced | static | config_runtime | universal | none | n/a | auto | Comparison minimum distance |
| `phase01_comparison_min_frames_frac` | 0.2 | 0.05 .. 0.95 | advanced | static | config_runtime | universal | none | n/a | auto | Comparison frame coverage |
| `phase01_comparison_n_comp_max` | 8 | 3 .. 20 | advanced | static | config_runtime | universal | none | n/a | auto | Most comparison stars |
| `phase01_comparison_n_comp_min` | 3 | 2 .. 12 | advanced | static | config_runtime | universal | none | n/a | auto | Fewest comparison stars |
| `phase01_ct_extrapolation_tol` | 0.0 | 0 .. 1 | expert | static | config_runtime | universal | none | n/a | auto | Colour term extrapolation |
| `phase01_ct_min_comp` | 7 | 2 .. 30 | expert | static | config_runtime | universal | none | n/a | auto | Colour term minimum stars |
| `snr_min` | 10.0 | 3 .. 1000 | advanced | static | config_runtime | universal | none | n/a | auto | Minimum SNR |

## trust

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `check_select_rms_floor` | 0.0001 | 0 .. 0.01 | expert | static | config_runtime | universal | none | n/a | auto | Check star scatter floor |
| `check_star_min_epochs` | 5 | 3 .. 50 | advanced | static | config_runtime | universal | none | n/a | auto | Check star minimum points |
| `comp_trust_min_comps` | 5 | 3 .. 20 | advanced | static | config_runtime | universal | none | n/a | auto | Comparison stars for green |
| `lc_quality_min_frames` | 20 | 3 .. 500 | advanced | static | config_runtime | universal | none | n/a | auto | Light curve minimum frames |
| `lc_quality_min_normal_frac` | 0.5 | 0.1 .. 1 | advanced | static | config_runtime | universal | none | n/a | auto | Light curve normal points |
| `lc_quality_short_min_frames` | 3 | 2 .. 100 | advanced | static | config_runtime | universal | none | n/a | auto | Short series minimum frames |
| `sparse_trust_T_green` | 1.5 | 0 .. 10 | expert | static | config_runtime | universal | none | n/a | auto | Sparse field green limit |
| `sparse_trust_T_red` | 4.0 | 0 .. 20 | expert | static | config_runtime | universal | none | n/a | auto | Sparse field red limit |
| `sparse_trust_X2_RED` | 0.0004 | 0 .. 0.01 | expert | static | config_runtime | universal | none | n/a | auto | Sparse field excess variance limit |
| `trust_flag_enabled` | True | - | expert | static | config_runtime | universal | none | n/a | auto | Trust flag |

## extinction

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `apply_color_term` | auto | - | expert | static | config_runtime | rig | rig_band | a | auto | Colour term |
| `k2_defaults_bprp` | {} | - | expert | static | config_runtime | rig | rig_band | a | auto | Own k2 per band |
| `k2_mode` | literature | - | advanced | static | config_runtime | site | site | n/a | auto | Second-order extinction |

## reports

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `hrd_dsc_confirm_prob` | 0.9 | 0.5 .. 1 | expert | static | config_runtime | universal | none | n/a | auto | HRD class probability |
| `hrd_enrich_max_candidates` | 20 | 1 .. 100 | expert | static | config_runtime | universal | none | n/a | auto | HRD online lookups |
| `hrd_enrich_tap_timeout_s` | 60.0 | 5 .. 120 | expert | static | config_runtime | universal | none | n/a | auto | HRD lookup timeout |
| `hrd_max_per_category` | 3 | 1 .. 20 | expert | static | config_runtime | universal | none | n/a | auto | HRD highlights per category |
| `hrd_min_per_net` | 4 | 0 .. 20 | expert | static | config_runtime | universal | none | n/a | auto | HRD highlights per search |
| `hrd_nss_category_enabled` | False | - | expert | static | config_runtime | universal | none | n/a | auto | HRD binaries category |
| `hrd_online_enrich_enabled` | True | - | expert | static | config_runtime | universal | none | n/a | auto | HRD online details |
| `hrd_parallax_min_mas` | 0.15 | 0 .. 10 | expert | static | config_runtime | universal | none | n/a | auto | HRD minimum parallax |
| `hrd_parallax_snr_min` | 5.0 | 1 .. 20 | expert | static | config_runtime | universal | none | n/a | auto | HRD parallax SNR |
| `hrd_simbad_enrich_enabled` | True | - | expert | static | config_runtime | universal | none | n/a | auto | HRD SIMBAD types |

## export

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `tess_contam_max` | 0.1 | 0 .. 1 | advanced | static | config_runtime | universal | none | n/a | auto | TESS contamination limit |
| `tess_enabled` | False | - | advanced | static | config_runtime | universal | none | n/a | auto | TESS check of new candidates |

## system

| key | default | range | tier | kind | owner | scope | scope_key | scope_group | widget | label |
|-----|---------|-------|------|------|-------|-------|-----------|-------------|--------|-------|
| `per_frame_mp_reserve_ram_gb` | 1.5 | 0 .. 64 | expert | static | config_runtime | universal | none | n/a | auto | RAM kept free per worker |
| `qc_preprocess_workers` | 1 | - | expert | resolved | internal | universal | none | n/a | hidden | QC Preprocess Workers |

