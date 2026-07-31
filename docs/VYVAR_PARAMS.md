# VYVAR -- Config <-> UI parameter registry

<!-- GENERATED FILE -- DO NOT EDIT BY HAND. -->
Regenerate with `python tools/gen_params_md.py`. Hand edits will be overwritten.
Source: `validation/params_registry.json` (editorial metadata) + `dataclasses.fields(AppConfig)` (defaults and types, from code).
Human-readable guide: `VYVAR_CONFIG_GUIDE_EN.md` / `VYVAR_CONFIG_GUIDE_CZ.md` (per-parameter plain-language explanations, hand-authored).
In-depth handbook: `VYVAR_PARAMETER_HANDBOOK_CZ.pdf` (Czech; per-parameter reasoning, ranges, math and literature; regenerate with `python dev/tools/docs_pdf/build_parameter_handbook.py`).

_Generated 2026-07-28T06:38:37Z at git HEAD 5dd34e5._

## Summary

- Entries: 270
- Tier: basic 13, advanced 70, expert 187
- Kind: static 252, derived 0, resolved 18
- Widget: auto 112, custom 143, hidden 15
- Owner: db_static 9, config_runtime 242, fits_dynamic 6, internal 13

Columns: key, default, range, tier, kind, owner, widget, label. `kind=resolved` means the runtime value can be auto-derived/overridden by the pipeline (the configured value is the base/fallback). `owner` is the storage-and-ownership axis: `db_static` (DB reference tables), `config_runtime` (user-tuned config.json), `fits_dynamic` (resolved from FITS/WCS at run time), `internal` (plumbing). `widget=custom` keys keep their hand-built UI; `widget=hidden` keys are plumbing not surfaced in the generated dashboard.

## observer

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `aavso_filter_map` | {} | - | expert | static | db_static | auto | AAVSO Filter Map |
| `aavso_observer_code` | UMIA | - | basic | static | db_static | auto | AAVSO Observer Code |
| `observer_alt_m` | 275.0 | - | basic | static | db_static | auto | Observer Alt M |
| `observer_code` |  | - | basic | static | db_static | auto | Observer Code |
| `observer_lat` | 50.1121658 | - | basic | static | db_static | auto | Observer Lat |
| `observer_location_id` | 0 | - | basic | static | db_static | auto | Observer Location ID |
| `observer_location_name` |  | - | basic | static | db_static | auto | Observer Location Name |
| `observer_lon` | 14.6982547 | - | basic | static | db_static | auto | Observer Lon |
| `observer_name` | Unknown Observer | - | basic | static | db_static | auto | Observer Name |

## paths

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `archive_root` | (resolved at runtime) | - | expert | static | internal | hidden | Archive Root |
| `blind_index_fine_path` |  | - | expert | static | internal | hidden | Blind Index Fine Path |
| `blind_index_path` |  | - | expert | static | internal | hidden | Blind Index Path |
| `blind_index_select_mode` | auto | - | expert | static | config_runtime | hidden | Blind Index Select Mode |
| `blind_index_wide_path` |  | - | expert | static | internal | hidden | Blind Index Wide Path |
| `calibration_library_root` | (resolved at runtime) | - | expert | static | internal | hidden | Calibration Library Root |
| `database_path` | (resolved at runtime) | - | expert | static | internal | hidden | Database Path |
| `exoplanet_local_db_path` | exoplanets/vyvar_exoplanet_local.db | - | expert | static | internal | hidden | Exoplanet Local Db Path |
| `gaia_db_path` |  | - | expert | static | internal | hidden | Gaia Db Path |
| `project_root` | VYVAR | - | expert | static | internal | hidden | Project Root |
| `vsx_local_db_path` |  | - | expert | static | internal | hidden | VSX Local Db Path |

## calibration

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `bpm_dark_mad_sigma` | 5.0 | 2 .. 12 | expert | static | config_runtime | auto | Bpm Dark Mad Sigma |
| `cal_diag_autocorrect_enabled` | True | - | expert | static | config_runtime | auto | Cal Diag Autocorrect Enabled |
| `cal_diag_gate_enabled` | True | - | advanced | static | config_runtime | auto | Cal Diag Gate Enabled |
| `cal_diag_hard_sigma` | 5.0 | 3 .. 10 | expert | static | config_runtime | auto | Cal Diag Hard Sigma |
| `cal_diag_rel_tol` | 0.02 | 0 .. 0.2 | expert | static | config_runtime | auto | Cal Diag Rel Tol |
| `cal_diag_sat_warn_frac` | 0.9 | 0.5 .. 1 | expert | static | config_runtime | auto | Cal Diag Sat Warn Frac |
| `calibration_library_native_binning` | 1 | 1 .. 16 | advanced | static | config_runtime | auto | Calibration Library Native Binning |
| `calibration_master_ccd_temp_tolerance_c` | 0.5 | - | expert | static | config_runtime | auto | Calibration Master CCD TEMP Tolerance C |
| `dao_qc_in_calibrate` | True | - | expert | static | config_runtime | auto | DAO QC In Calibrate |
| `masterdark_validity_days` | 90 | - | basic | static | config_runtime | auto | Masterdark Validity Days |
| `masterflat_validity_days` | 200 | - | basic | static | config_runtime | auto | Masterflat Validity Days |

## qc

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `auto_fwhm_enabled` | True | - | advanced | static | config_runtime | custom | Auto FWHM Enabled |
| `auto_fwhm_k_factor` | 1.5 | - | advanced | static | config_runtime | custom | Auto FWHM K Factor |
| `auto_fwhm_k_max` | 4.0 | - | advanced | static | config_runtime | custom | Auto FWHM K Max |
| `auto_fwhm_k_min` | 1.0 | - | advanced | static | config_runtime | custom | Auto FWHM K Min |
| `frame_align_residual_gate_enabled` | False | - | expert | static | config_runtime | auto | Frame Align Residual Gate Enabled |
| `frame_align_residual_max_frac` | 0.25 | 0.05 .. 1 | expert | static | config_runtime | auto | Frame Align Residual Max Frac |
| `frame_align_residual_min_keep_frames` | 10 | 3 .. 100000 | expert | static | config_runtime | auto | Frame Align Residual Min Keep Frames |
| `frame_quality_fwhm_factor` | 1.0 | 0.8 .. 3 | expert | static | config_runtime | auto | Frame Quality FWHM Factor |
| `frame_quality_gate_enabled` | False | - | expert | static | config_runtime | auto | Frame Quality Gate Enabled |
| `frame_quality_min_keep_frames` | 10 | 3 .. 100000 | expert | static | config_runtime | auto | Frame Quality Min Keep Frames |
| `frame_quality_ratio_k` | 5.0 | 2 .. 20 | expert | static | config_runtime | auto | Frame Quality Ratio K |
| `osc_channel_binning` | 2 | 1 .. 4 | advanced | static | config_runtime | auto | OSC Channel Binning |
| `preprocess_sky_surface_order` | 2 | 0 .. 2 | expert | static | config_runtime | auto | Preprocess Sky Surface Order |
| `qc_after_calibrate_enabled` | True | - | basic | static | config_runtime | auto | QC After Calibrate Enabled |
| `qc_dao_detection_sigma` | 5.0 | - | expert | static | config_runtime | auto | QC DAO Detection Sigma |
| `qc_elong_limit` | 1.8 | - | expert | static | config_runtime | auto | QC Elong Limit |
| `qc_fwhm_limit` | 8.0 | - | expert | static | config_runtime | auto | QC FWHM Limit |
| `qc_max_background_rms` | None | - | expert | static | config_runtime | auto | QC Max Background RMS |
| `qc_max_hfr` | 5.0 | - | advanced | static | config_runtime | auto | QC Max HFR |
| `qc_min_stars` | 10 | - | advanced | static | config_runtime | auto | QC Min Stars |

## alignment

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `alignment_detection_sigma` | 5.0 | - | advanced | static | config_runtime | custom | Alignment Detection Sigma |
| `alignment_max_control_points` | 80 | - | expert | static | config_runtime | custom | Alignment Max Control Points |
| `alignment_max_stars` | 160 | 10 .. 5000 | advanced | static | config_runtime | custom | Alignment Max Stars |

## detection

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `blind_img_select_mode` | per_cell | - | expert | static | config_runtime | custom | Blind Img Select Mode |
| `blind_img_star_budget` | 80 | - | expert | static | config_runtime | custom | Blind Img Star Budget |
| `blind_use_rig_prior` | True | - | expert | static | config_runtime | custom | Blind Use Rig Prior |
| `blind_verify_early_accept` | 30 | - | expert | static | config_runtime | custom | Blind Verify Early Accept |
| `blind_verify_early_floor` | 0 | - | expert | static | config_runtime | custom | Blind Verify Early Floor |
| `blind_verify_early_fraction` | 0.2 | 0 .. 0.95 | expert | static | config_runtime | custom | Blind Verify Early Fraction |
| `blind_verify_enabled` | True | - | expert | static | config_runtime | custom | Blind Verify Enabled |
| `blind_verify_inmemory_catalog` | True | - | expert | static | config_runtime | custom | Blind Verify Inmemory Catalog |
| `blind_verify_match_tol_px` | 2.5 | 0.5 .. 20 | expert | static | config_runtime | custom | Blind Verify Match Tol PX |
| `blind_verify_min_fraction` | 0.15 | 0.05 .. 0.95 | expert | static | config_runtime | custom | Blind Verify Min Fraction |
| `blind_verify_min_matches` | 12 | - | expert | static | config_runtime | custom | Blind Verify Min Matches |
| `blind_verify_top_n` | 15 | - | expert | static | config_runtime | custom | Blind Verify Top N |
| `catalog_query_max_rows` | 15000 | 1000 .. 500000 | expert | static | config_runtime | auto | Catalog Query Max Rows |
| `crowding_blend_tighten_threshold` | 0.04 | 0 .. 1 | expert | static | config_runtime | custom | Crowding Blend Tighten Threshold |
| `crowding_classifier_enabled` | False | - | expert | static | config_runtime | custom | Crowding Classifier Enabled |
| `crowding_comp_availability_loosen_count` | 500.0 | 0 .. 1000000 | expert | static | config_runtime | custom | Crowding Comp Availability Loosen Count |
| `crowding_tighten_min_fwhm_px` | 3.0 | 0 .. 30 | expert | static | config_runtime | custom | Crowding Tighten Min FWHM PX |
| `debug_platesolver` | False | - | expert | static | config_runtime | auto | Debug Platesolver |
| `epsf_min_stars` | 30 | - | expert | static | config_runtime | custom | EPSF Min Stars |
| `exoplanet_match_max_sep_arcsec` | 3.0 | 0.5 .. 30 | expert | static | config_runtime | auto | Exoplanet Match Max Sep Arcsec |
| `field_density_adaptive_enabled` | True | - | expert | static | config_runtime | custom | Field Density Adaptive Enabled |
| `field_density_dense_threshold` | 1000.0 | - | expert | static | config_runtime | custom | Field Density Dense Threshold |
| `field_density_sparse_threshold` | 300.0 | 1 .. 50000 | expert | static | config_runtime | custom | Field Density Sparse Threshold |
| `frame_height_px` | 1397 | - | expert | resolved | internal | hidden | Frame Height PX |
| `frame_width_px` | 2082 | - | expert | resolved | internal | hidden | Frame Width PX |
| `masterstar_accept_mode` | odds | - | expert | static | config_runtime | custom | Masterstar Accept Mode |
| `masterstar_best_of_n` | 10 | 1 .. 25 | expert | static | config_runtime | custom | Masterstar Best Of N |
| `masterstar_catalog_recovery_min` | 0.65 | 0.4 .. 0.95 | advanced | static | config_runtime | custom | Masterstar Catalog Recovery Min |
| `masterstar_centre_rms_max_px` | 1.2 | 0.5 .. 5 | advanced | static | config_runtime | custom | Masterstar Centre RMS Max PX |
| `masterstar_dao_pass2_sigma` | 1.9 | - | expert | static | config_runtime | custom | Masterstar DAO Pass2 Sigma |
| `masterstar_dao_threshold_sigma` | 2.1 | 0.1 .. 6 | advanced | static | config_runtime | custom | Masterstar DAO Threshold Sigma |
| `masterstar_detection_cap_adaptive` | True | - | expert | static | config_runtime | custom | Masterstar Detection Cap Adaptive |
| `masterstar_detection_cap_k` | 0.08 | 0.01 .. 1 | expert | static | config_runtime | custom | Masterstar Detection Cap K |
| `masterstar_detection_cap_max` | 800 | - | expert | static | config_runtime | custom | Masterstar Detection Cap Max |
| `masterstar_detection_cap_min` | 250 | - | expert | static | config_runtime | custom | Masterstar Detection Cap Min |
| `masterstar_distortion_benign_ratio_max` | 3.2 | 2 .. 5 | advanced | static | config_runtime | custom | Masterstar Distortion Benign Ratio Max |
| `masterstar_min_matched_floor` | 40 | - | advanced | static | config_runtime | custom | Masterstar Min Matched Floor |
| `masterstar_platesolve_sip_max_order` | 4 | - | advanced | static | config_runtime | custom | Masterstar Platesolve SIP Max Order |
| `masterstar_platesolve_sip_min_order` | 3 | - | advanced | static | config_runtime | custom | Masterstar Platesolve SIP Min Order |
| `masterstar_prematch_peak_sigma_floor` | 1.8 | 0.5 .. 6 | advanced | static | config_runtime | custom | Masterstar Prematch Peak Sigma Floor |
| `masterstar_quality_crowded_n_cat_min` | 800 | - | expert | static | config_runtime | custom | Masterstar Quality Crowded N Cat Min |
| `masterstar_sibling_min_matched` | 40 | - | advanced | static | config_runtime | custom | Masterstar Sibling Min Matched |
| `masterstar_sibling_min_quadrants` | 3 | - | advanced | static | config_runtime | custom | Masterstar Sibling Min Quadrants |
| `masterstar_sibling_recovery_enabled` | True | - | advanced | static | config_runtime | custom | Masterstar Sibling Recovery Enabled |
| `masterstar_sibling_rms_max_px` | 2.0 | 0.5 .. 10 | advanced | static | config_runtime | custom | Masterstar Sibling RMS Max PX |
| `masterstar_sibling_stack_n` | 10 | - | advanced | static | config_runtime | custom | Masterstar Sibling Stack N |
| `masterstar_use_best_frame_fwhm` | True | - | expert | static | config_runtime | custom | Masterstar Use Best Frame FWHM |
| `phase01_chip_interior_margin_px` | 50 | - | advanced | static | config_runtime | auto | Phase01 Chip Interior Margin PX |
| `phase01_plate_scale_arcsec_per_px` | 1.3 | 0 .. 30 | expert | resolved | fits_dynamic | auto | Phase01 Plate Scale Arcsec Per PX |
| `plate_scale_arcsec_per_px` | 1.3 | 0.1 .. 30 | expert | resolved | fits_dynamic | auto | Plate Scale Arcsec Per PX |
| `plate_solve_fov_deg` | 1.0 | - | expert | resolved | fits_dynamic | auto | Plate Solve FOV Deg |
| `saturate_limit_fraction` | 0.85 | - | expert | static | config_runtime | auto | Saturate Limit Fraction |
| `sips_dao_fwhm_px` | 2.5 | 1 .. 8 | advanced | static | config_runtime | custom | Sips DAO FWHM PX |
| `sips_dao_threshold_sigma` | 3.5 | - | advanced | static | config_runtime | custom | Sips DAO Threshold Sigma |
| `variability_clip_ratio_min` | 0.8 | - | expert | static | config_runtime | auto | Variability Clip Ratio Min |
| `variability_comp_floor_factor` | 1.5 | - | expert | static | config_runtime | auto | Variability Comp Floor Factor |
| `variability_mag_limit` | 14.5 | - | expert | static | config_runtime | auto | Variability Mag Limit |
| `variability_min_amplitude_mag` | 0.01 | - | expert | static | config_runtime | auto | Variability Min Amplitude Mag |
| `variability_min_frames` | 30 | - | expert | static | config_runtime | auto | Variability Min Frames |
| `variability_min_frames_frac` | 0.5 | 0.05 .. 0.99 | expert | static | config_runtime | auto | Variability Min Frames Frac |
| `variability_min_points_rms` | 20 | - | expert | static | config_runtime | auto | Variability Min Points RMS |
| `variability_min_rms_pct` | 1.5 | - | expert | static | config_runtime | auto | Variability Min RMS Pct |
| `variability_p85_filter` | 85 | - | expert | static | config_runtime | auto | Variability P85 Filter |
| `variability_sigma_clip` | 5.0 | - | expert | static | config_runtime | auto | Variability Sigma Clip |
| `variability_sigma_threshold` | 2.3 | - | expert | static | config_runtime | auto | Variability Sigma Threshold |
| `variability_slope_floor` | 0.02 | - | expert | static | config_runtime | auto | Variability Slope Floor |
| `variability_smoothness_max` | 0.8 | - | expert | static | config_runtime | auto | Variability Smoothness Max |
| `variability_vdi_z_threshold` | 3.0 | - | expert | static | config_runtime | auto | Variability VDI Z Threshold |
| `verify_mag_limit` | 14.0 | 8 .. 18 | expert | static | config_runtime | auto | Verify Mag Limit |
| `vsx_out_of_scope_types` | [] | - | basic | static | config_runtime | auto | VSX Out Of Scope Types |

## photometry

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `annulus_inner_fwhm` | 4.75 | 1 .. 10 | advanced | resolved | config_runtime | auto | Annulus Inner FWHM |
| `annulus_outer_fwhm` | 9.0 | 1.5 .. 12 | advanced | resolved | config_runtime | auto | Annulus Outer FWHM |
| `aperture_comp_factor` | 1.1 | 0.25 .. 3 | advanced | static | config_runtime | auto | Aperture Comp Factor |
| `aperture_correction_enabled` | True | - | expert | static | config_runtime | auto | Aperture Correction Enabled |
| `aperture_correction_max_contamination` | 0.15 | 0 .. 2 | expert | static | config_runtime | auto | Aperture Correction Max Contamination |
| `aperture_correction_max_scatter_mag` | 0.03 | 0 .. 2 | expert | static | config_runtime | auto | Aperture Correction Max Scatter Mag |
| `aperture_correction_min_ref_stars` | 3 | 1 .. 50 | expert | static | config_runtime | auto | Aperture Correction Min Ref Stars |
| `aperture_fwhm_factor` | 1.9 | 0.5 .. 6 | advanced | resolved | config_runtime | auto | Aperture FWHM Factor |
| `aperture_photometry_enabled` | True | - | advanced | static | config_runtime | auto | Aperture Photometry Enabled |
| `aperture_snr_sizing` | {"large": 4.0, "small": 1.5} | - | expert | resolved | config_runtime | auto | Aperture SNR Sizing |
| `aperture_variable_factor` | 1.0 | 0.25 .. 3 | advanced | static | config_runtime | auto | Aperture Variable Factor |
| `cog_ac_factor_max` | 5.0 | - | expert | static | config_runtime | custom | COG Ac Factor Max |
| `cog_aperture_correction_enabled` | False | - | expert | static | config_runtime | custom | COG Aperture Correction Enabled |
| `cog_isolation_fwhm` | 6.0 | - | expert | static | config_runtime | custom | COG Isolation FWHM |
| `cog_ladder_step_px` | 0.5 | - | expert | static | config_runtime | custom | COG Ladder Step PX |
| `cog_min_stars` | 8 | 1 .. 500 | expert | static | config_runtime | custom | COG Min Stars |
| `cog_ref_fwhm` | 4.5 | 1.5 .. 10 | expert | static | config_runtime | custom | COG Ref FWHM |
| `cog_sat_frac` | 0.85 | - | expert | static | config_runtime | custom | COG Sat Frac |
| `cog_snr_min` | 50.0 | - | expert | static | config_runtime | custom | COG SNR Min |
| `democratic_detrend_enabled` | False | - | advanced | static | config_runtime | custom | Democratic Detrend Enabled |
| `democratic_sg_window_frac` | 0.5 | 0.05 .. 0.95 | expert | static | config_runtime | custom | Democratic SG Window Frac |
| `err_background_mode` | empirical | - | expert | static | config_runtime | auto | Err Background Mode |
| `err_empty_apertures_min` | 16 | - | expert | static | config_runtime | auto | Err Empty Apertures Min |
| `err_empty_apertures_n` | 64 | - | expert | static | config_runtime | auto | Err Empty Apertures N |
| `gain` | 1.0 | - | expert | resolved | fits_dynamic | auto | Gain |
| `gs11_comp_max_dilution` | 0.9 | 0.01 .. 1 | expert | static | config_runtime | custom | GS11 Comp Max Dilution |
| `gs11_comp_suspect_dilution` | 0.98 | 0.01 .. 1 | expert | static | config_runtime | custom | GS11 Comp Suspect Dilution |
| `gs11_dilution_aperture_arcsec` | 0.0 | 0 .. 120 | expert | static | config_runtime | custom | GS11 Dilution Aperture Arcsec |
| `gs11_dilution_enabled` | False | - | expert | static | config_runtime | custom | GS11 Dilution Enabled |
| `gs11_dilution_mag_limit_delta` | 5.0 | 0.5 .. 15 | expert | static | config_runtime | custom | GS11 Dilution Mag Limit Delta |
| `gs11_target_min_dilution` | 0.5 | 0.01 .. 1 | expert | static | config_runtime | custom | GS11 Target Min Dilution |
| `neighbor_sub_centroid_max_fwhm` | 1.0 | - | expert | static | config_runtime | custom | Neighbor Sub Centroid Max FWHM |
| `neighbor_sub_chi2_max` | 120.0 | - | expert | static | config_runtime | custom | Neighbor Sub Chi2 Max |
| `neighbor_sub_max_neighbor_overmag` | 0.3 | - | expert | static | config_runtime | custom | Neighbor Sub Max Neighbor Overmag |
| `neighbor_sub_max_target_undermag` | 0.2 | - | expert | static | config_runtime | custom | Neighbor Sub Max Target Undermag |
| `neighbor_sub_min_recovered_snr` | 5.0 | - | expert | static | config_runtime | custom | Neighbor Sub Min Recovered SNR |
| `neighbor_sub_nn_contam_dmag` | 2.5 | - | expert | static | config_runtime | custom | Neighbor Sub NN Contam Dmag |
| `neighbor_sub_refuse_sep_fwhm` | 0.8 | - | expert | static | config_runtime | custom | Neighbor Sub Refuse Sep FWHM |
| `neighbor_sub_regime_dmag_min` | 2.5 | - | expert | static | config_runtime | custom | Neighbor Sub Regime Dmag Min |
| `neighbor_sub_regime_sep_max` | 1.1 | - | expert | static | config_runtime | custom | Neighbor Sub Regime Sep Max |
| `neighbor_sub_residual_rms_max` | 150.0 | - | expert | static | config_runtime | custom | Neighbor Sub Residual RMS Max |
| `nonlinearity_fwhm_ratio` | 1.25 | 1.01 .. 3 | advanced | static | config_runtime | auto | Nonlinearity FWHM Ratio |
| `nonlinearity_peak_percentile` | 20.0 | 0 .. 50 | advanced | static | config_runtime | auto | Nonlinearity Peak Percentile |
| `per_frame_sat_min_clean_frac` | 0.5 | 0.1 .. 1 | advanced | static | config_runtime | custom | Per Frame Sat Min Clean Frac |
| `per_frame_saturation_enabled` | False | - | advanced | static | config_runtime | custom | Per Frame Saturation Enabled |
| `phase2a_airmass_before_outlier` | False | - | expert | static | config_runtime | auto | Phase2a Airmass Before Outlier |
| `photometry_mode` | both | - | basic | static | config_runtime | auto | Photometry Mode |
| `psf_adaptive_enabled` | False | - | expert | static | config_runtime | custom | PSF Adaptive Enabled |
| `psf_adaptive_resolve_fwhm` | 2.0 | - | expert | static | config_runtime | custom | PSF Adaptive Resolve FWHM |
| `psf_adaptive_snr_lo` | 15.0 | - | expert | static | config_runtime | custom | PSF Adaptive SNR Lo |
| `psf_chi2_threshold` | 50.0 | - | advanced | static | config_runtime | custom | PSF Chi2 Threshold |
| `psf_group_sep_fwhm` | 1.5 | - | expert | static | config_runtime | custom | PSF Group Sep FWHM |
| `psf_grouper_enabled` | False | - | expert | static | config_runtime | custom | PSF Grouper Enabled |
| `psf_neighbor_include_fwhm` | 3.0 | - | expert | static | config_runtime | custom | PSF Neighbor Include FWHM |
| `psf_neighbor_sub_enabled` | False | - | expert | static | config_runtime | custom | PSF Neighbor Sub Enabled |
| `psf_photometry_enabled` | False | - | advanced | static | config_runtime | custom | PSF Photometry Enabled |
| `psf_quality_fallback_enabled` | True | - | expert | static | config_runtime | custom | PSF Quality Fallback Enabled |
| `psf_spatial_enabled` | False | - | expert | static | config_runtime | custom | PSF Spatial Enabled |
| `psf_spatial_grid` | 3x3 | - | expert | static | config_runtime | custom | PSF Spatial Grid |
| `psf_spatial_min_stars_per_cell` | 25 | - | expert | static | config_runtime | custom | PSF Spatial Min Stars Per Cell |
| `psf_spatial_order` | 0 | 0 .. 2 | expert | static | config_runtime | custom | PSF Spatial Order |
| `pytics_enabled` | True | - | advanced | static | config_runtime | custom | PYTICS Enabled |
| `pytics_n_iter` | 5 | 1 .. 20 | expert | static | config_runtime | custom | PYTICS N Iter |
| `read_noise` | 10.0 | - | expert | resolved | fits_dynamic | auto | Read Noise |
| `save_lightcurve_png` | False | - | advanced | static | config_runtime | auto | Save Lightcurve Png |
| `savgol_detrend_enabled` | False | - | advanced | static | config_runtime | custom | Savgol Detrend Enabled |
| `savgol_polyorder` | 2 | - | expert | static | config_runtime | custom | Savgol Polyorder |
| `savgol_window_frac` | 0.5 | - | expert | static | config_runtime | custom | Savgol Window Frac |
| `sigma_sys_mag` | {} | - | expert | static | config_runtime | auto | Sigma Sys Mag |
| `sysrem_enabled` | False | - | expert | static | config_runtime | custom | SYSREM Enabled |
| `sysrem_n_iter` | 3 | - | expert | static | config_runtime | custom | SYSREM N Iter |
| `temporal_bin_window` | 0 | 0 .. 51 | expert | static | config_runtime | custom | Temporal Bin Window |
| `temporal_binning_enabled` | False | - | advanced | static | config_runtime | custom | Temporal Binning Enabled |

## comp_selection

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `comp_clip_sigma` | 5.0 | 3 .. 10 | advanced | static | config_runtime | auto | Comp Clip Sigma |
| `comp_color_tiers` | [{'bprp': 0.15, 'w': 1.0}, {'bprp': 0.3, 'w': 0.85}, {'bprp': 0.55, 'w': 0.5}, {'bprp': 1.1, 'w': 0.25}] | - | advanced | static | config_runtime | auto | Comp Color Tiers |
| `comp_contamination_penalty_k` | 3.0 | 0 .. 20 | expert | static | config_runtime | auto | Comp Contamination Penalty K |
| `comp_iterative_clip_enabled` | False | - | advanced | static | config_runtime | auto | Comp Iterative Clip Enabled |
| `comp_max_delta_bprp` | 0.79 | 0 .. 5 | advanced | resolved | config_runtime | auto | Comp Max Delta BPRP |
| `comp_max_slope_mmag_hr` | 5.0 | 0 .. 500 | expert | static | config_runtime | auto | Comp Max Slope Mmag Hr |
| `comp_select_rms_floor` | 1e-06 | - | expert | static | config_runtime | auto | Comp Select RMS Floor |
| `comp_slope_significance_k` | 3.0 | 0 .. 10 | advanced | static | config_runtime | auto | Comp Slope Significance K |
| `comp_sparse_fallback_enabled` | True | - | advanced | static | config_runtime | auto | Comp Sparse Fallback Enabled |
| `comp_sparse_fallback_min` | 0 | - | advanced | static | config_runtime | auto | Comp Sparse Fallback Min |
| `global_comp_pool_enabled` | True | - | expert | static | config_runtime | auto | Global Comp Pool Enabled |
| `phase01_comparison_exclude_gaia_extobj` | True | - | advanced | static | config_runtime | custom | Phase01 Comparison Exclude Gaia Extobj |
| `phase01_comparison_exclude_gaia_nss` | True | - | advanced | static | config_runtime | custom | Phase01 Comparison Exclude Gaia NSS |
| `phase01_comparison_fov_fraction` | 0.75 | - | expert | static | config_runtime | custom | Phase01 Comparison FOV Fraction |
| `phase01_comparison_isolation_radius_px` | 25.0 | 1 .. 200 | expert | static | config_runtime | custom | Phase01 Comparison Isolation Radius PX |
| `phase01_comparison_mag_bright_threshold` | 12.75 | 6 .. 18 | advanced | static | config_runtime | custom | Phase01 Comparison Mag Bright Threshold |
| `phase01_comparison_max_comp_rms` | 0.1 | 0.01 .. 0.5 | advanced | resolved | config_runtime | custom | Phase01 Comparison Max Comp RMS |
| `phase01_comparison_max_dist_deg` | 1.5 | 0.05 .. 10 | advanced | resolved | config_runtime | custom | Phase01 Comparison Max Dist Deg |
| `phase01_comparison_max_fwhm_factor` | 1.5 | 0.5 .. 5 | expert | static | config_runtime | custom | Phase01 Comparison Max FWHM Factor |
| `phase01_comparison_max_mag_diff` | 1.5 | 0.05 .. 5 | advanced | resolved | config_runtime | custom | Phase01 Comparison Max Mag Diff |
| `phase01_comparison_max_mag_diff_absolute` | 3.0 | 1 .. 10 | advanced | static | config_runtime | custom | Phase01 Comparison Max Mag Diff Absolute |
| `phase01_comparison_max_mag_diff_bright_floor` | 1.5 | 0 .. 4 | advanced | static | config_runtime | custom | Phase01 Comparison Max Mag Diff Bright Floor |
| `phase01_comparison_max_psf_chi2` | 50.0 | 1 .. 500 | expert | static | config_runtime | custom | Phase01 Comparison Max PSF Chi2 |
| `phase01_comparison_min_dist_arcsec` | 60.0 | 0 .. 600 | advanced | resolved | config_runtime | custom | Phase01 Comparison Min Dist Arcsec |
| `phase01_comparison_min_frames_frac` | 0.2 | 0.05 .. 0.95 | advanced | static | config_runtime | custom | Phase01 Comparison Min Frames Frac |
| `phase01_comparison_n_comp_max` | 8 | - | advanced | static | config_runtime | custom | Phase01 Comparison N Comp Max |
| `phase01_comparison_n_comp_min` | 3 | - | advanced | resolved | config_runtime | custom | Phase01 Comparison N Comp Min |
| `phase01_comparison_rms_outlier_sigma` | 3.0 | 1 .. 10 | expert | static | config_runtime | custom | Phase01 Comparison RMS Outlier Sigma |
| `phase01_ct_extrapolation_tol` | 0.0 | - | expert | static | config_runtime | auto | Phase01 CT Extrapolation Tol |
| `phase01_ct_min_comp` | 7 | 2 .. 30 | expert | static | config_runtime | auto | Phase01 CT Min Comp |
| `phase01_flux_col` | dao_flux | - | expert | static | config_runtime | auto | Phase01 Flux Col |
| `phase01_tiers` | [0.5, 1.0, 1.5, 2.0] | - | expert | static | config_runtime | auto | Phase01 Tiers |
| `phase01_use_bprp_primary` | True | - | expert | static | config_runtime | auto | Phase01 Use BPRP Primary |

## trust

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `check_select_rms_floor` | 0.0001 | 0 .. 0.01 | expert | static | config_runtime | auto | Check Select RMS Floor |
| `check_star_min_epochs` | 5 | - | advanced | static | config_runtime | auto | Check Star Min Epochs |
| `comp_qa_enabled` | True | - | expert | static | config_runtime | auto | Comp Qa Enabled |
| `comp_trust_min_comps` | 5 | - | advanced | static | config_runtime | auto | Comp Trust Min Comps |
| `lc_quality_min_frames` | 20 | - | advanced | static | config_runtime | auto | Lc Quality Min Frames |
| `lc_quality_min_normal_frac` | 0.5 | 0.1 .. 1 | advanced | static | config_runtime | auto | Lc Quality Min Normal Frac |
| `lc_quality_short_min_frames` | 3 | - | advanced | static | config_runtime | auto | Lc Quality Short Min Frames |
| `sparse_trust_T_green` | 1.5 | - | expert | static | config_runtime | auto | Sparse Trust T Green |
| `sparse_trust_T_red` | 4.0 | - | expert | static | config_runtime | auto | Sparse Trust T Red |
| `sparse_trust_X2_RED` | 0.0004 | - | expert | static | config_runtime | auto | Sparse Trust X2 RED |
| `trust_flag_enabled` | True | - | expert | static | config_runtime | auto | Trust Flag Enabled |

## extinction

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `apply_color_term` | off | - | expert | static | config_runtime | auto | Apply Color Term |
| `k2_ceiling` | 0.1 | - | expert | static | config_runtime | custom | K2 Ceiling |
| `k2_defaults_bprp` | {} | - | expert | static | config_runtime | custom | K2 Defaults BPRP |
| `k2_fit_consistency_sigma` | 2.0 | - | expert | static | config_runtime | custom | K2 Fit Consistency Sigma |
| `k2_fit_enabled` | False | - | expert | static | config_runtime | custom | K2 Fit Enabled |
| `k2_fit_lit_factor` | 4.0 | - | expert | static | config_runtime | custom | K2 Fit Lit Factor |
| `k2_fit_min_detectability` | 3.0 | - | expert | static | config_runtime | custom | K2 Fit Min Detectability |
| `k2_mode` | literature | - | advanced | static | config_runtime | custom | K2 Mode |

## reports

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `hrd_color_bg_box_px` | 96 | 32 .. 512 | expert | static | config_runtime | custom | HRD Color Bg Box PX |
| `hrd_color_chroma_boost` | 2.2 | 1 .. 3 | expert | static | config_runtime | custom | HRD Color Chroma Boost |
| `hrd_color_chroma_snr` | 3.0 | 0 .. 20 | expert | static | config_runtime | custom | HRD Color Chroma SNR |
| `hrd_color_field_enabled` | True | - | expert | static | config_runtime | custom | HRD Color Field Enabled |
| `hrd_color_highlight_mode` | soft | - | expert | static | config_runtime | custom | HRD Color Highlight Mode |
| `hrd_color_saturation` | 0.85 | 0 .. 1 | expert | static | config_runtime | custom | HRD Color Saturation |
| `hrd_color_white_point` | field_median | - | expert | static | config_runtime | custom | HRD Color White Point |
| `hrd_dsc_confirm_prob` | 0.9 | 0.5 .. 1 | expert | static | config_runtime | custom | HRD DSC Confirm Prob |
| `hrd_enrich_max_candidates` | 20 | 1 .. 100 | expert | static | config_runtime | custom | HRD Enrich Max Candidates |
| `hrd_enrich_tap_timeout_s` | 20.0 | 5 .. 120 | expert | static | config_runtime | custom | HRD Enrich Tap Timeout S |
| `hrd_max_per_category` | 3 | 1 .. 20 | expert | static | config_runtime | custom | HRD Max Per Category |
| `hrd_min_per_net` | 4 | 0 .. 20 | expert | static | config_runtime | custom | HRD Min Per Net |
| `hrd_nss_category_enabled` | False | - | expert | static | config_runtime | custom | HRD NSS Category Enabled |
| `hrd_online_enrich_enabled` | True | - | expert | static | config_runtime | custom | HRD Online Enrich Enabled |
| `hrd_parallax_min_mas` | 0.15 | 0 .. 10 | expert | static | config_runtime | custom | HRD Parallax Min Mas |
| `hrd_parallax_snr_min` | 5.0 | 1 .. 20 | expert | static | config_runtime | custom | HRD Parallax SNR Min |
| `hrd_simbad_enrich_enabled` | True | - | expert | static | config_runtime | custom | HRD Simbad Enrich Enabled |

## export

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `export_arcsec_per_px` | 1.3 | - | expert | static | fits_dynamic | auto | Export Arcsec Per PX |
| `tess_enabled` | False | - | advanced | static | config_runtime | auto | TESS Enabled |

## system

| key | default | range | tier | kind | owner | widget | label |
|-----|---------|-------|------|------|-------|--------|-------|
| `per_frame_mp_reserve_ram_gb` | 1.5 | - | expert | static | config_runtime | hidden | Per Frame MP Reserve RAM Gb |
| `qc_preprocess_workers` | 1 | - | expert | resolved | internal | hidden | QC Preprocess Workers |

---

## Parameter budget notes (closure Step 1, 2026-07-31)

| key | budget status | evidence |
|-----|---------------|----------|
| `aperture_snr_sizing` | **DEAD on science path** | `precompute_and_save_snr_aperture_table_for_draft` never passes bounds; defaults `r_min_fwhm=0.8`, `r_max_fwhm=2.5` used (`CURSOR_RESULT_closure_step1.md`, S1) |

**Flow doc note (Step 1b, V7):** `build_flow_doc.py:391` correctly documents hardcoded
`r_min=0.8 x FWHM` .. `r_max=2.5 x FWHM`; `flow_doc_facts.py:60` tracks
`compute_snr_optimal_aperture_table`. The orphan parameter is `aperture_snr_sizing` in config,
not the flow-doc SNR sweep text. PDF regen not required for closure measurement.

