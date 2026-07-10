# VYVAR — Config ↔ UI parameter registry

Generated **2026-06-02** from `config.py`, `config.json`, and `ui*.py`.
Registry required by `docs/VYVAR_PROCESS.md` Definition of Done §4.

**Legend — exposed:** `yes` = Settings or tool UI widget; `intentionally-hidden` =
dev/gated flag documented here (edit via `config.json`); `no` = drift (config-only,
no UI yet).

**Summary:** 82 exposed · 137 intentionally-hidden · 34 config-only (no UI) · 0 UI references without config key

---

## Recently-added flags (audit focus)

| key | default | clamp | UI | exposed | notes |
|-----|---------|-------|-----|---------|-------|
| `comp_qa_enabled` | True | — | — | **intentionally-hidden** | OK in config.json |
| `trust_flag_enabled` | True | — | — | **intentionally-hidden** | OK in config.json |
| `phase01_comparison_proximity_tiebreak` | False | — | — | **intentionally-hidden** | OK in config.json |
| `phase01_comparison_rms_bin_mag` | 0.001 | 0.0001 … 0.05 | — | **intentionally-hidden** | OK in config.json |
| `cog_aperture_correction_enabled` | False | — | — | **intentionally-hidden** | OK in config.json |
| `crowding_classifier_enabled` | False | — | — | **intentionally-hidden** | OK in config.json |
| `psf_adaptive_enabled` | False | — | — | **intentionally-hidden** | OK in config.json |
| `psf_grouper_enabled` | False | — | — | **intentionally-hidden** | OK in config.json |
| `psf_spatial_enabled` | False | — | — | **intentionally-hidden** | OK in config.json |
| `skip_processed_directory` | False | — | — | **intentionally-hidden** | OK in config.json |
| `qc_fwhm_limit` | 8.0 | — | — | **intentionally-hidden** | OK in config.json |
| `qc_elong_limit` | 1.8 | — | — | **intentionally-hidden** | OK in config.json |
| `frame_quality_gate_enabled` | False | — | ui_settings.py (Photometry → Data quality) | **yes** | Round-2 B.2; default OFF → byte-identical |
| `frame_quality_ratio_k` | 5.0 | 2.0 … 20.0 | ui_settings.py (Photometry → Data quality) | **yes** | robust z-cut on per-frame flux_large/flux |
| `frame_quality_fwhm_factor` | 1.0 | 0.8 … 3.0 | ui_settings.py (Photometry → Data quality) | **yes** | guard: reject only if FWHM ≥ factor×median |
| `frame_quality_min_keep_frames` | 10 | 3 … 100000 | ui_settings.py (Photometry → Data quality) | **yes** | safety floor: skip gate below this |
| `frame_align_residual_gate_enabled` | False | — | ui_settings.py (Photometry → Data quality) | **yes** | Fix B; default OFF → byte-identical; residual column always recorded |
| `frame_align_residual_max_frac` | 0.25 | 0.05 … 1.0 | ui_settings.py (Photometry → Data quality) | **yes** | reject if residual > frac×science aperture radius (rig-agnostic, not fixed px) |
| `frame_align_residual_min_keep_frames` | 10 | 3 … 100000 | ui_settings.py (Photometry → Data quality) | **yes** | safety floor: skip gate below this |

---

## Alignment

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `alignment_detection_sigma` | 5.0 | — | ui_settings.py:79, ui_settings.py:389, ui_settings.py:392 | yes |
| `alignment_max_stars` | 160 | 10 … 5000 | ui_settings.py:79, ui_settings.py:375, ui_settings.py:378 | yes |

## Calibration / BPM

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `bpm_dark_mad_sigma` | 5.0 | 2.0 … 12.0 | — | intentionally-hidden |

## Calibration validity

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `masterdark_validity_days` | 80 | — | ui_settings.py:286, ui_settings.py:869 | yes |
| `masterflat_validity_days` | 524 | — | ui_settings.py:299, ui_settings.py:870 | yes |

## CAL-DIAG (calibrate-time radiometry gate)

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `cal_diag_gate_enabled` | True | — | ui_settings.py (Calibration tab) | yes |
| `cal_diag_autocorrect_enabled` | True | — | config.json only | intentionally-hidden |
| `cal_diag_rel_tol` | 0.02 | 0.0 … 0.2 | config.json only | intentionally-hidden |
| `cal_diag_hard_sigma` | 5.0 | 3.0 … 10.0 | config.json only | intentionally-hidden |
| `cal_diag_sat_warn_frac` | 0.90 | 0.5 … 1.0 | config.json only | intentionally-hidden |

## Crowding classifier

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `crowding_blend_tighten_threshold` | 0.04 | 0.0 … 1.0 | — | intentionally-hidden |
| `crowding_classifier_enabled` | False | — | — | intentionally-hidden |
| `crowding_comp_availability_loosen_count` | 500.0 | 0.0 … 1_000_000.0 | — | intentionally-hidden |
| `crowding_tighten_min_fwhm_px` | 3.0 | 0.0 … 30.0 | — | intentionally-hidden |

## Field density

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `field_density_adaptive_enabled` | True | — | — | intentionally-hidden |
| `field_density_dense_threshold` | 1000.0 | float(self.field_density_sparse_threshold) + 1.0 … 100_000.0 | — | intentionally-hidden |
| `field_density_sparse_threshold` | 300.0 | 1.0 … 50_000.0 | — | intentionally-hidden |

## GS11 dilution

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `gs11_comp_max_dilution` | 0.9 | 0.01 … 1.0 | — | intentionally-hidden |
| `gs11_comp_suspect_dilution` | 0.98 | 0.01 … 1.0 | — | intentionally-hidden |
| `gs11_dilution_aperture_arcsec` | 0.0 | 0.0 … 120.0 | — | intentionally-hidden |
| `gs11_dilution_enabled` | False | — | — | intentionally-hidden |
| `gs11_dilution_mag_limit_delta` | 5.0 | 0.5 … 15.0 | — | intentionally-hidden |
| `gs11_target_min_dilution` | 0.5 | 0.01 … 1.0 | — | intentionally-hidden |

## MASTERSTAR / SIPS DAO

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `sips_dao_fwhm_px` | 2.5 | 1.0 … 8.0 | ui_epsf_dashboard.py:334 | yes |
| `sips_dao_threshold_sigma` | 3.5 | — | ui_epsf_dashboard.py:336 | yes |

## MASTERSTAR / plate-solve

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `blind_verify_early_accept` | 30 | — | — | intentionally-hidden |
| `blind_verify_early_floor` | 0 | — | — | intentionally-hidden |
| `blind_verify_early_fraction` | 0.2 | 0.0 … 0.95 | — | intentionally-hidden |
| `blind_verify_enabled` | True | — | — | intentionally-hidden |
| `blind_verify_inmemory_catalog` | True | — | — | intentionally-hidden |
| `blind_verify_match_tol_px` | 2.5 | 0.5 … 20.0 | — | intentionally-hidden |
| `blind_verify_min_fraction` | 0.15 | 0.05 … 0.95 | — | intentionally-hidden |
| `blind_verify_min_matches` | 12 | — | — | intentionally-hidden |
| `blind_verify_top_n` | 15 | — | — | intentionally-hidden |
| `debug_platesolver` | False | — | — | intentionally-hidden |
| `masterstar_accept_mode` | 'odds' | — | — | intentionally-hidden |
| `masterstar_best_of_n` | 10 | 1 … 25 | — | intentionally-hidden |
| `masterstar_catalog_recovery_min` | 0.65 | 0.40 … 0.95 | ui_dao_stars.py:154, ui_dao_stars.py:159, ui_dao_stars.py:351 | yes |
| `masterstar_centre_rms_max_px` | 1.2 | 0.5 … 5.0 | ui_dao_stars.py:156, ui_dao_stars.py:177, ui_dao_stars.py:353 | yes |
| `masterstar_dao_pass2_sigma` | 1.9 | — | — | intentionally-hidden ⚠ not in config.json |
| `masterstar_dao_threshold_sigma` | 2.1 | 0.1 … 6.0 | ui_dao_stars.py:79, ui_dao_stars.py:98, ui_dao_stars.py:136 | yes |
| `masterstar_detection_cap_adaptive` | True | — | — | intentionally-hidden |
| `masterstar_detection_cap_k` | 0.08 | 0.01 … 1.0 | — | intentionally-hidden |
| `masterstar_detection_cap_max` | 800 | — | — | intentionally-hidden |
| `masterstar_detection_cap_min` | 250 | — | — | intentionally-hidden |
| `masterstar_distortion_benign_ratio_max` | 3.2 | 2.0 … 5.0 | ui_dao_stars.py:157, ui_dao_stars.py:186, ui_dao_stars.py:354 | yes |
| `masterstar_false_alarm_p_max` | 1e-06 | 1e-12 … 1.0 | — | intentionally-hidden |
| `masterstar_log_astroalign` | True | — | — | intentionally-hidden |
| `masterstar_min_matched_floor` | 40 | — | ui_dao_stars.py:155, ui_dao_stars.py:168, ui_dao_stars.py:352 | yes |
| `masterstar_odds_k` | 12.0 | 1.0 … 100.0 | — | intentionally-hidden |
| `masterstar_odds_match_floor` | 30 | — | — | intentionally-hidden |
| `masterstar_odds_min_quadrants` | 3 | — | — | intentionally-hidden |
| `masterstar_optimizer_mirror_extra_log` | True | — | — | intentionally-hidden |
| `masterstar_platesolve_nn_refine_max_rms_px` | None | — | — | intentionally-hidden |
| `masterstar_platesolve_prewrite_relaxed_rms_max_px` | 35.0 | — | — | intentionally-hidden |
| `masterstar_platesolve_prewrite_rms_max_px` | 30.0 | — | — | intentionally-hidden |
| `masterstar_platesolve_sip_max_order` | 4 | — | ui_dao_stars.py:311, ui_dao_stars.py:316, ui_dao_stars.py:349 | yes |
| `masterstar_platesolve_sip_min_order` | 3 | — | ui_dao_stars.py:312, ui_dao_stars.py:323, ui_dao_stars.py:350 | yes |
| `masterstar_prematch_peak_sigma_floor` | 1.8 | 0.5 … 6.0 | ui_dao_stars.py:80, ui_dao_stars.py:97, ui_dao_stars.py:115 | yes |
| `masterstar_quality_crowded_n_cat_min` | 800 | — | — | intentionally-hidden |
| `masterstar_sibling_min_matched` | 40 | — | ui_dao_stars.py:207, ui_dao_stars.py:218, ui_dao_stars.py:356 | yes |
| `masterstar_sibling_min_quadrants` | 3 | — | ui_dao_stars.py:209, ui_dao_stars.py:234, ui_dao_stars.py:358 | yes |
| `masterstar_sibling_recovery_enabled` | True | — | ui_dao_stars.py:206, ui_dao_stars.py:212, ui_dao_stars.py:355 | yes |
| `masterstar_sibling_rms_max_px` | 2.0 | 0.5 … 10.0 | ui_dao_stars.py:208, ui_dao_stars.py:226, ui_dao_stars.py:357 | yes |
| `masterstar_sibling_stack_n` | 10 | — | ui_dao_stars.py:210, ui_dao_stars.py:242, ui_dao_stars.py:359 | yes |
| `masterstar_sip_force_rms_guard_ratio` | 1.15 | — | — | intentionally-hidden |
| `masterstar_solver_use_draft_median_if_hint_sep_deg` | 1.0 | 0.0 … 180.0 | — | intentionally-hidden |
| `masterstar_use_best_frame_fwhm` | True | — | — | intentionally-hidden ⚠ not in config.json |
| `platesolve_anisotropy_threshold` | 1.3 | 1.01 … 5.0 | — | intentionally-hidden |

## Observer / export

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `aavso_filter_map` | {} | — | — | no |
| `aavso_observer_code` | 'UMIA' | — | — | intentionally-hidden |
| `export_arcsec_per_px` | 1.3 | — | — | intentionally-hidden |
| `observer_alt_m` | 1412.0 (dataclass 275.0) | — | — | no |
| `observer_code` | 'UMIA' (dataclass '') | — | — | no |
| `observer_lat` | -29.039083 (dataclass 50.1121658) | — | — | no |
| `observer_location_id` | 6 (dataclass 2) | — | — | no |
| `observer_location_name` | 'Boyden - JAR' (dataclass '') | — | — | no |
| `observer_lon` | 26.40394 (dataclass 14.6982547) | — | — | no |
| `observer_name` | 'Milan Uhlar' (dataclass 'Unknown Observer') | — | — | no |

## Other

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `apply_color_term` | 'off' | — | — | no |
| `k2_mode` | 'literature' | — | ui_settings.py (Second-order extinction) | yes |
| `k2_defaults_bprp` | {} | — | — | no |
| `blind_cluster_coherence_cap` | 25 | — | — | no |
| `blind_cluster_eps_deg` | 1.0 | 0.1 … 5.0 | — | no |
| `blind_cluster_min_samples` | 3 | — | — | no |
| `blind_cluster_min_votes` | 4 | — | — | no |
| `blind_cluster_vote_span` | 12 | — | — | no |
| `blind_img_select_mode` | 'per_cell' | — | — | no |
| `blind_img_star_budget` | 80 | — | — | no |
| `blind_prefilter_min` | 4 | — | — | intentionally-hidden |
| `blind_scale_tol_frac` | 0.1 | 0.02 … 0.50 | — | no |
| `blind_use_rig_prior` | True | — | — | no |
| `check_select_rms_floor` | 0.0001 | 0.0 … 0.01 | — | no |
| `check_star_min_epochs` | 5 | — | ui_settings.py:631, ui_settings.py:905 | yes |
| `comp_clip_sigma` | 5.0 | 3.0 … 10.0 | ui_settings.py:792, ui_settings.py:795, ui_settings.py:930 | yes |
| `comp_iterative_clip_enabled` | True (dataclass False) | — | ui_settings.py:927 | yes |
| `comp_slope_significance_k` | 3.0 | 0.0 … 10.0 | ui_settings.py:562, ui_settings.py:565, ui_settings.py:570 | yes |
| `comp_sparse_fallback_enabled` | True | — | ui_settings.py:775, ui_settings.py:776, ui_settings.py:926 | yes |
| `comp_sparse_fallback_min` | 0 | — | ui_settings.py:783, ui_settings.py:786, ui_settings.py:929 | yes |
| `comp_trust_min_comps` | 5 | — | ui_settings.py:621, ui_settings.py:624, ui_settings.py:902 | yes |
| `lc_quality_min_frames` | 20 | — | ui_settings.py:602, ui_settings.py:899, ui_settings.py:900 | yes |
| `lc_quality_min_normal_frac` | 0.5 | 0.1 … 1.0 | ui_settings.py:616, ui_settings.py:901 | yes |
| `lc_quality_short_min_frames` | 3 | — | ui_settings.py:609, ui_settings.py:898, ui_settings.py:899 | yes |
| `neighbor_sub_centroid_max_fwhm` | 1.0 | — | — | no |
| `neighbor_sub_chi2_max` | 120.0 | — | — | no |
| `neighbor_sub_max_neighbor_overmag` | 0.3 | — | — | no |
| `neighbor_sub_max_target_undermag` | 0.2 | — | — | no |
| `neighbor_sub_min_recovered_snr` | 5.0 | — | — | no |
| `neighbor_sub_nn_contam_dmag` | 2.5 | — | — | no |
| `neighbor_sub_refuse_sep_fwhm` | 0.8 | — | — | no |
| `neighbor_sub_regime_dmag_min` | 2.5 | — | — | no |
| `neighbor_sub_regime_sep_max` | 1.1 | — | — | no |
| `neighbor_sub_residual_rms_max` | 150.0 | — | — | no |
| `verify_mag_limit` | 14.0 | 8.0 … 18.0 | — | intentionally-hidden |

## PSF photometry

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `epsf_min_stars` | 30 | — | — | intentionally-hidden |
| `moffat_chi2_limit` | 50.0 | — | — | intentionally-hidden |
| `psf_adaptive_enabled` | False | — | — | intentionally-hidden |
| `psf_adaptive_resolve_fwhm` | 2.0 | — | — | intentionally-hidden |
| `psf_adaptive_snr_lo` | 15.0 | — | — | intentionally-hidden |
| `psf_chi2_threshold` | 50.0 | — | ui_epsf_dashboard.py:128 | yes |
| `psf_group_sep_fwhm` | 1.5 | — | — | intentionally-hidden |
| `psf_grouper_enabled` | False | — | — | intentionally-hidden |
| `psf_neighbor_include_fwhm` | 3.0 | — | — | intentionally-hidden |
| `psf_neighbor_sub_enabled` | False | — | — | no |
| `psf_photometry_enabled` | False | — | ui_epsf_dashboard.py:260, ui_photometry.py:84, ui_photometry.py:89 | yes |
| `psf_quality_fallback_enabled` | True | — | — | intentionally-hidden |
| `psf_spatial_enabled` | False | — | — | intentionally-hidden |
| `psf_spatial_grid` | '3x3' | — | — | intentionally-hidden |
| `psf_spatial_min_stars_per_cell` | 25 | — | — | intentionally-hidden |
| `psf_spatial_order` | 0 | 0 … 2 | — | intentionally-hidden |

## Parallelism

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `per_frame_mp_reserve_ram_gb` | 1.5 | — | — | intentionally-hidden |

## Paths / calibration library

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `calibration_library_native_binning` | 1 | 1 … 16 | ui_settings.py:310, ui_settings.py:317, ui_settings.py:322 | yes |

## Paths / catalogs

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `blind_index_fine_path` | 'C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\gaia_triangles_fine.pkl' (dataclass '') | — | ui_settings.py:158, ui_settings.py:873, ui_settings.py:875 | yes |
| `blind_index_path` | '' → runtime 'C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\gaia_triangles_fine.pkl' | — | ui_settings.py:875 | yes ⚠ not in config.json |
| `blind_index_select_mode` | 'auto' | — | — | no |
| `blind_index_wide_path` | 'C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\gaia_triangles_wide.pkl' (dataclass '') | — | ui_settings.py:193, ui_settings.py:874 | yes |
| `catalog_query_max_rows` | 15000 | 1000 … 500_000 | — | no |
| `gaia_db_path` | 'C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\vyvar_gaia_dr3.db' (dataclass '') | — | ui_hrd.py:49, ui_settings.py:123, ui_settings.py:872 | yes |
| `hrd_online_enrich_enabled` | True | — | hrd_enrich.py, hrd_analysis.py | intentionally-hidden |
| `hrd_simbad_enrich_enabled` | True | — | hrd_enrich.py, hrd_analysis.py | intentionally-hidden |
| `hrd_enrich_max_candidates` | 20 | 1 … 100 | hrd_analysis.py | intentionally-hidden |
| `hrd_parallax_min_mas` | 0.15 | 0.0 … 10.0 | hrd_analysis.py | intentionally-hidden |
| `hrd_parallax_snr_min` | 5.0 | 1.0 … 20.0 | hrd_analysis.py | intentionally-hidden |
| `hrd_max_per_category` | 3 | 1 … 20 | hrd_analysis.py | intentionally-hidden |
| `hrd_min_per_net` | 4 | 0 … 20 | hrd_analysis.py | intentionally-hidden |
| `hrd_nss_category_enabled` | False | — | hrd_analysis.py | intentionally-hidden |
| `hrd_dsc_confirm_prob` | 0.90 | 0.5 … 1.0 | hrd_analysis.py | intentionally-hidden |
| `vsx_local_db_path` | 'C:\\ASTRO\\python\\VYVAR\\VSX\\vyvar_vsx_local_v2.db' (dataclass '') | — | ui_aperture_photometry.py:1416, ui_masterstar_qa.py:595, ui_settings.py:230 | yes |
| `exoplanet_local_db_path` | `exoplanets/vyvar_exoplanet_local.db` (dataclass default) | — | ui_settings.py, pipeline.py `detect_stars_and_match_catalog` | yes |
| `exoplanet_match_max_sep_arcsec` | 3.0 | 0.5 … 30.0 | pipeline.py `detect_stars_and_match_catalog` | intentionally-hidden |
| `vsx_variable_targets_mag_limit` | 14.5 | — | ui_aperture_photometry.py:1358, ui_settings.py:269, ui_settings.py:274 | yes |

## Phase 0+1

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `phase01_chip_interior_margin_px` | 50 | — | ui_select_stars.py:435, ui_select_stars.py:436, ui_select_stars.py:539 | yes |
| `phase01_ct_extrapolation_tol` | 0.0 | — | — | intentionally-hidden |
| `phase01_ct_min_comp` | 7 | 2 … 30 | — | intentionally-hidden |
| `phase01_flux_col` | 'dao_flux' | — | — | intentionally-hidden |
| `phase01_use_bprp_primary` | True | — | ui_aperture_photometry.py:1661 | intentionally-hidden | OK in config.json |
| `phase01_match_radius_arcsec` | 10.0 | 3.0 … 30.0 | — | intentionally-hidden |
| `phase01_plate_scale_arcsec_per_px` | 1.3 | 0.0 … 30.0 | — | intentionally-hidden |
| `phase01_tier1_mag` | 0.5 | — | — | intentionally-hidden ⚠ not in config.json |
| `phase01_tier2_mag` | 1.0 | — | — | intentionally-hidden ⚠ not in config.json |
| `phase01_tier3_mag` | 1.5 | — | — | intentionally-hidden ⚠ not in config.json |
| `phase01_tier4_mag` | 2.0 | — | — | intentionally-hidden ⚠ not in config.json |

## Phase 0+1 comp selection

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `comp_contamination_penalty_k` | 3.0 | 0.0 … 20.0 | — | intentionally-hidden |
| `comp_max_delta_bprp` | 0.79 | 0.0 … 5.0 | ui_settings.py:723, ui_settings.py:916 | yes |
| `comp_max_slope_mmag_hr` | 5.0 | 0.0 … 500.0 | — | intentionally-hidden |
| `comp_tier1_bprp_limit` | 0.15 | 0.02 … 5.0 | ui_settings.py:694, ui_settings.py:912 | yes |
| `comp_tier1_weight` | 1.0 | 0.01 … 1.00 | ui_settings.py:728, ui_settings.py:731, ui_settings.py:917 | yes |
| `comp_tier2_bprp_limit` | 0.3 | 0.05 … 5.0 | ui_settings.py:702, ui_settings.py:913 | yes |
| `comp_tier2_weight` | 0.85 | 0.01 … 1.00 | ui_settings.py:735, ui_settings.py:738, ui_settings.py:918 | yes |
| `comp_tier3_bprp_limit` | 0.55 | 0.05 … 5.0 | ui_settings.py:709, ui_settings.py:914 | yes |
| `comp_tier3_weight` | 0.5 | 0.01 … 1.00 | ui_settings.py:742, ui_settings.py:745, ui_settings.py:919 | yes |
| `comp_tier4_bprp_limit` | 1.1 | 0.05 … 5.0 | ui_settings.py:716, ui_settings.py:915 | yes |
| `comp_tier4_weight` | 0.25 | 0.01 … 1.00 | ui_settings.py:749, ui_settings.py:752, ui_settings.py:920 | yes |
| `global_comp_pool_enabled` | True | — | — | intentionally-hidden |
| `phase01_comparison_exclude_gaia_extobj` | True | — | ui_select_stars.py:449, ui_select_stars.py:538, ui_settings.py:824 | yes |
| `phase01_comparison_exclude_gaia_nss` | True | — | ui_select_stars.py:448, ui_select_stars.py:537, ui_settings.py:820 | yes |
| `phase01_comparison_fov_fraction` | 0.75 | — | — | intentionally-hidden ⚠ not in config.json |
| `phase01_comparison_isolation_radius_px` | 25.0 | 1.0 … 200.0 | — | intentionally-hidden |
| `phase01_comparison_mag_bright_threshold` | 12.75 | 6.0 … 18.0 | ui_select_stars.py:441, ui_select_stars.py:529, ui_settings.py:663 | yes |
| `phase01_comparison_max_comp_rms` | 0.1 | 0.01 … 0.5 | ui_select_stars.py:445, ui_select_stars.py:534, ui_settings.py:768 | yes |
| `phase01_comparison_max_dist_deg` | 1.5 | 0.05 … 10.0 | ui_dao_stars.py:81, ui_dao_stars.py:293, ui_dao_stars.py:301 | yes |
| `phase01_comparison_max_fwhm_factor` | 1.5 | 0.5 … 5.0 | — | intentionally-hidden |
| `phase01_comparison_max_mag_diff` | 1.5 | 0.05 … 5.0 | ui_select_stars.py:440, ui_select_stars.py:528, ui_settings.py:80 | yes |
| `phase01_comparison_max_mag_diff_absolute` | 3.0 | 1.0 … 10.0 | ui_settings.py:680, ui_settings.py:911 | yes |
| `phase01_comparison_max_mag_diff_bright_floor` | 1.5 | 0.0 … 4.0 | ui_select_stars.py:442, ui_select_stars.py:530, ui_settings.py:670 | yes |
| `phase01_comparison_max_psf_chi2` | 50.0 | 1.0 … 500.0 | — | intentionally-hidden |
| `phase01_comparison_min_dist_arcsec` | 60.0 | 0.0 … 600.0 | ui_select_stars.py:446, ui_select_stars.py:535, ui_settings.py:800 | yes |
| `phase01_comparison_min_frames_frac` | 0.2 | 0.05 … 0.95 | ui_quality_dashboard.py:852, ui_select_stars.py:447, ui_select_stars.py:536 | yes |
| `phase01_comparison_n_comp_max` | 8 | — | ui_select_stars.py:444, ui_select_stars.py:533, ui_settings.py:762 | yes |
| `phase01_comparison_n_comp_min` | 3 | — | ui_select_stars.py:444, ui_select_stars.py:532, ui_settings.py:756 | yes |
| `phase01_comparison_proximity_tiebreak` | False | — | — | intentionally-hidden |
| `phase01_comparison_rms_bin_mag` | 0.001 | 0.0001 … 0.05 | — | intentionally-hidden |
| `phase01_comparison_rms_outlier_sigma` | 3.0 | 1.0 … 10.0 | — | intentionally-hidden |

## Phase 2A

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `phase2a_airmass_before_outlier` | False | — | — | intentionally-hidden |

## Phase 2A detrend / ALG

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `democratic_detrend_enabled` | False | — | ui_settings.py:558, ui_settings.py:896 | yes |
| `democratic_sg_window_frac` | 0.5 | 0.05 … 0.95 | — | intentionally-hidden |
| `pytics_enabled` | True | — | ui_settings.py:547, ui_settings.py:894 | yes |
| `pytics_n_iter` | 5 | 1 … 20 | — | intentionally-hidden |
| `savgol_detrend_enabled` | False | — | ui_settings.py:553, ui_settings.py:895 | yes |
| `savgol_polyorder` | 2 | — | — | intentionally-hidden |
| `savgol_window_frac` | 0.5 | — | — | intentionally-hidden |
| `sysrem_enabled` | False | — | — | intentionally-hidden |
| `sysrem_n_iter` | 3 | — | — | intentionally-hidden |
| `temporal_bin_window` | 0 | 0 … 51 | — | intentionally-hidden |
| `temporal_binning_enabled` | False | — | ui_settings.py:542, ui_settings.py:893 | yes |

## Phase 2A output

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `save_lightcurve_png` | False | — | ui_photometry.py:72, ui_photometry.py:77, ui_photometry.py:103 | yes |

## Photometry / COG correction

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `cog_ac_factor_max` | 5.0 | — | — | intentionally-hidden |
| `cog_aperture_correction_enabled` | False | — | — | intentionally-hidden |
| `cog_isolation_fwhm` | 6.0 | — | — | intentionally-hidden |
| `cog_ladder_step_px` | 0.5 | — | — | intentionally-hidden |
| `cog_min_stars` | 8 | 1 … 500 | — | intentionally-hidden |
| `cog_ref_fwhm` | 4.5 | 1.5 … 10.0 | — | intentionally-hidden |
| `cog_sat_frac` | 0.85 | — | — | intentionally-hidden |
| `cog_snr_min` | 50.0 | — | — | intentionally-hidden |

## Photometry / aperture

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `annulus_inner_fwhm` | 4.75 | 1.0 … 10.0 | ui_settings.py:77, ui_settings.py:495, ui_settings.py:498 | yes |
| `annulus_outer_fwhm` | 9.0 | 1.5 … 12.0 | ui_settings.py:77, ui_settings.py:502, ui_settings.py:505 | yes |
| `aperture_comp_factor` | 1.1 | 0.25 … 3.0 | ui_settings.py:532, ui_settings.py:892 | yes |
| `aperture_correction_enabled` | True | — | — | intentionally-hidden |
| `aperture_correction_max_contamination` | 0.15 | 0.0 … 2.0 | — | intentionally-hidden |
| `aperture_correction_max_scatter_mag` | 0.03 | 0.0 … 2.0 | — | intentionally-hidden |
| `aperture_correction_min_ref_stars` | 3 | 1 … 50 | — | intentionally-hidden |
| `aperture_fwhm_factor` | 1.9 | 0.5 … 6.0 | ui_dao_stars.py:78, ui_dao_stars.py:267, ui_dao_stars.py:290 | yes |
| `aperture_fwhm_factor_large` | 4.0 | — | — | intentionally-hidden |
| `aperture_fwhm_factor_medium` | 2.5 | — | — | intentionally-hidden |
| `aperture_fwhm_factor_small` | 1.5 | — | — | intentionally-hidden |
| `aperture_photometry_enabled` | True | — | ui_photometry.py:60, ui_photometry.py:65, ui_photometry.py:102 | yes |
| `aperture_variable_factor` | 1.0 | 0.25 … 3.0 | ui_settings.py:523, ui_settings.py:891 | yes |
| `nonlinearity_fwhm_ratio` | 1.25 | 1.01 … 3.0 | ui_settings.py:583, ui_settings.py:586, ui_settings.py:890 | yes |
| `nonlinearity_peak_percentile` | 20.0 | 0.0 … 50.0 | ui_settings.py:576, ui_settings.py:579, ui_settings.py:889 | yes |

## Photometry mode

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `photometry_mode` | 'both' | — | ui_photometry.py:32, ui_photometry.py:51, ui_photometry.py:101 | yes |

## Plate scale

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `plate_scale_arcsec_per_px` | 1.3 | 0.1 … 30.0 | — | intentionally-hidden |

## Plate scale / FOV

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `plate_solve_fov_deg` | 1.0 | — | — | intentionally-hidden ⚠ not in config.json |

## QC

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `dao_qc_in_calibrate` | True | — | — | no |
| `qc_after_calibrate_enabled` | True | — | ui_settings.py:78, ui_settings.py:331, ui_settings.py:332 | yes |
| `qc_dao_detection_sigma` | 5.0 | — | — | intentionally-hidden |
| `qc_elong_limit` | 1.8 | — | — | intentionally-hidden |
| `qc_fwhm_limit` | 8.0 | — | — | intentionally-hidden |
| `qc_max_background_rms` | None | — | — | intentionally-hidden |
| `qc_max_hfr` | 5.0 | — | ui_settings.py:78, ui_settings.py:345, ui_settings.py:348 | yes |
| `qc_min_stars` | 10 | — | ui_settings.py:78, ui_settings.py:359, ui_settings.py:362 | yes |
| `qc_preprocess_workers` | 1 → runtime 8 | — | — | intentionally-hidden ⚠ not in config.json |

## QC / FITS QA

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `auto_fwhm_enabled` | True | — | ui_quality_dashboard.py:510, ui_quality_dashboard.py:514, ui_quality_dashboard.py:554 | yes |
| `auto_fwhm_k_factor` | 1.5 | float(self.auto_fwhm_k_min) … float(self.auto_fwhm_k_max) | ui_quality_dashboard.py:511, ui_quality_dashboard.py:557, ui_quality_dashboard.py:570 | yes |
| `auto_fwhm_k_max` | 4.0 | — | ui_quality_dashboard.py:576 | yes |
| `auto_fwhm_k_min` | 1.0 | — | ui_quality_dashboard.py:575 | yes |

## QC / skip processed

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `skip_processed_directory` | False | — | — | intentionally-hidden |

## Sensor / noise model

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `gain` | 1.0 | — | — | intentionally-hidden |
| `read_noise` | 10.0 | — | — | intentionally-hidden |
| `err_background_mode` | `empirical` | `empirical` \| `howell` | — | intentionally-hidden |
| `err_empty_apertures_n` | 64 | 16..256 | — | intentionally-hidden |
| `err_empty_apertures_min` | 16 | 1..256 | — | intentionally-hidden |
| `sky_adu_fallback` | 1581.6 | — | — | intentionally-hidden |

## Sensor / saturation

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `saturate_limit_fraction` | 0.85 | — | — | intentionally-hidden ⚠ not in config.json |

## Sensor geometry

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `frame_height_px` | 1397 | — | — | intentionally-hidden |
| `frame_width_px` | 2082 | — | — | intentionally-hidden |

## TESS

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `tess_enabled` | False | — | ui_aperture_photometry.py:1453, ui_variability.py:926, ui_variability.py:1650 | yes |

## Trust / QA (comp_qa)

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `comp_qa_enabled` | True | — | — | intentionally-hidden |

## Trust / QA (trust gate)

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `trust_flag_enabled` | True | — | — | intentionally-hidden |

## Variability detection

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `variability_clip_ratio_min` | 0.8 | — | — | intentionally-hidden |
| `variability_comp_floor_factor` | 1.5 | — | — | intentionally-hidden |
| `variability_mag_limit` | 14.5 | — | — | intentionally-hidden |
| `variability_min_amplitude_mag` | 0.01 | — | — | intentionally-hidden |
| `variability_min_frames` | 30 | — | — | intentionally-hidden |
| `variability_min_frames_frac` | 0.5 | 0.05 … 0.99 | — | intentionally-hidden |
| `variability_min_points_rms` | 20 | — | — | intentionally-hidden |
| `variability_min_rms_pct` | 1.5 | — | — | intentionally-hidden |
| `variability_p85_filter` | 85 | — | — | intentionally-hidden |
| `variability_sigma_clip` | 5.0 | — | — | intentionally-hidden |
| `variability_sigma_threshold` | 2.3 | — | — | intentionally-hidden |
| `variability_slope_floor` | 0.02 | — | — | intentionally-hidden |
| `variability_smoothness_max` | 0.8 | — | — | intentionally-hidden |
| `variability_vdi_z_threshold` | 3.0 | — | — | intentionally-hidden |

---

## Simple differential approach (2026-06-15, **Workstream A landed**)

Production defaults (config.py + config.json + UI parity):

| key | default | note |
|-----|---------|------|
| `temporal_binning_enabled` | **False** | ALG-3 OFF — per-frame ensemble (V0612 validated) |
| `temporal_bin_window` | 0 (auto) | unchanged when binning off |
| `apply_color_term` | **off** | color-matched comps; revisit if tier3/0.79 cap widens |
| `phase01_comparison_n_comp_min` / `n_comp_max` | 3 / 8 | rank by comp_rms inside tier ladder |
| `comp_tier1/2/3_bprp_limit` | 0.15 / 0.30 / 0.55 | widen ladder until n_comp_min |
| `comp_max_delta_bprp` | 0.79 | hard cap |
| `comp_select_rms_floor` | 1e-6 | hidden; drops isolated_bin artefact comps |
| Phase-1 selector | `_select_comps_by_color_then_rms` | wired in `comp_selection_per_target._assign_comp_tiers_to_pool` |

**DoD-A (2026-06-15):** `tmp/phase10/dod_a_production_defaults.py` — V0612 `delta_mag` pre=0.0113,
corr=0.949, N=7, C3 absent. **PASS.**

**Workstream B (landed; DoD-B PASS):** ``apply_reporting_postprocess`` in ``photometry_core.py``.
Harness ``tmp/phase11/dod_b_workstream_b.json``.

**Gate:** recommend >=1 additional ground-truth field before treating n=1 as global risk closure.

---

## draft_409 trust/consistency cleanup (2026-06-16, **Fixes 1-3 landed**)

Summary / LC / PDF fields (not all `AppConfig` keys):

| field / symbol | source | note |
|----------------|--------|------|
| `aperture_px` | measured `aperture_r_px` from proc cache | card + LC; supersedes Phase-2A replan display |
| `aperture_px_planned` | Phase-2A SNR-opt replan | diagnostic only |
| `lc_rms_ooe` | brightest-tertile scatter of `mag_calib` | headline precision for variables on PDF card |
| `lc_rms` | undemeaned std of `mag_calib` | retained; not headline for variables |
| `n_stability_good` / `n_stability_suspect` | `check_comparison_stability` on ensemble residual | trust soft-warning when suspect > 0 |
| `_APERTURE_SIZING_MAG_COLS` | observed-band `mag` before `phot_g_mean_mag` | SNR-opt aperture sizing |

**Superseded:** `comp_color_window_bprp` single-step window — tier ladder (0.15/0.30/0.55, cap 0.79) used instead.

**DoD (draft_409 V0612 g):** `n_stability_good=8`, `n_stability_suspect=0`, trust GREEN, measured aperture 5.754 px,
`lc_rms_ooe` ~0.006, `delta_mag` pre-eclipse RMS ~0.010, SIPS eclipse + shared frame anomaly match.

---

