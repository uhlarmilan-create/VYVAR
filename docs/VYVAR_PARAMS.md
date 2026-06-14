# VYVAR — Config ↔ UI parameter registry

Generated **2026-06-09** from `config.py`, `config.json`, `psf_photometry.py`, and `ui*.py`.
Registry required by `docs/VYVAR_PROCESS.md` Definition of Done section 4.

**Legend -- exposed:** `yes` = Settings or tool UI widget; `intentionally-hidden` =
dev/gated flag documented here (edit via `config.json`); `no` = drift (config-only,
no UI yet); `production-constant` = hardcoded in production module (not a config key).

**Summary:** 63 exposed · 139 intentionally-hidden · 22 config-only (no UI) · 2 UI references without config key · 5 production-constants

---

## Recently-added flags (audit focus)

| key | default | clamp | UI | exposed | notes |
|-----|---------|-------|-----|---------|-------|
| `comp_qa_enabled` | True | — | — | **intentionally-hidden** | OK in config.json |
| `trust_flag_enabled` | True | — | — | **intentionally-hidden** | OK in config.json |
| `lc_quality_min_frames` | 20 | 3 … 500 | ui_settings.py (Data quality) | yes |
| `lc_quality_short_min_frames` | 3 | 2 … 100 (≤ min_frames) | ui_settings.py (Data quality) | yes |
| `lc_quality_min_normal_frac` | 0.5 | 0.1 … 1.0 | ui_settings.py (Data quality) | yes |
| `comp_trust_min_comps` | 5 | 3 … 20 (≤ n_comp_max) | ui_settings.py (Data quality) | yes |
| `check_star_min_epochs` | 5 | 3 … 50 | ui_settings.py (Data quality) | yes |
| `check_select_rms_floor` | 1e-4 | 0 … 0.01 | — | **intentionally-hidden** | CS-2 artefact floor |
| `aperture_correction_max_contamination` | 0.15 | 0.0 … 2.0 | — | **intentionally-hidden** | CS-4 check-star crowding gate |
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
| `masterdark_validity_days` | 80 | — | ui_settings.py:286, ui_settings.py:775 | yes |
| `masterflat_validity_days` | 524 | — | ui_settings.py:299, ui_settings.py:776 | yes |

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
| `masterstar_best_of_n` | 10 | 1 … 25 | — | intentionally-hidden |
| `masterstar_dao_pass2_sigma` | 1.9 | — | — | intentionally-hidden ⚠ not in config.json |
| `masterstar_dao_threshold_sigma` | 2.1 | 0.1 … 6.0 | ui_dao_stars.py:79, ui_dao_stars.py:98, ui_dao_stars.py:136 | yes |
| `masterstar_log_astroalign` | True | — | — | intentionally-hidden |
| `masterstar_optimizer_mirror_extra_log` | True | — | — | intentionally-hidden |
| `masterstar_platesolve_nn_refine_max_rms_px` | None | — | — | intentionally-hidden |
| `masterstar_platesolve_prewrite_relaxed_rms_max_px` | 35.0 | — | — | intentionally-hidden |
| `masterstar_platesolve_prewrite_rms_max_px` | 30.0 | — | — | intentionally-hidden |
| `masterstar_platesolve_sip_max_order` | 4 | — | ui_dao_stars.py:209, ui_dao_stars.py:214, ui_dao_stars.py:247 | yes |
| `masterstar_platesolve_sip_min_order` | 3 | — | ui_dao_stars.py:210, ui_dao_stars.py:221, ui_dao_stars.py:248 | yes |
| `masterstar_prematch_peak_sigma_floor` | 1.8 | 0.5 … 6.0 | ui_dao_stars.py:80, ui_dao_stars.py:97, ui_dao_stars.py:115 | yes |
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
| `observer_alt_m` | 275.0 | — | — | no |
| `observer_code` | 'UMIA' (dataclass '') | — | — | no |
| `observer_lat` | 50.1121658 | — | — | no |
| `observer_location_id` | 2 | — | — | no |
| `observer_location_name` | 'Jirny' (dataclass '') | — | — | no |
| `observer_lon` | 14.6982547 | — | — | no |
| `observer_name` | 'Milan Uhlar' (dataclass 'Unknown Observer') | — | — | no |
| `varastro_observer_name` | 'Milan Uhlar' | — | — | no |

## Other

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `apply_color_term` | 'auto' | — | — | no |
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
| `psf_photometry_enabled` | False | — | ui_epsf_dashboard.py:260, ui_photometry.py:84, ui_photometry.py:89 | yes |
| `psf_quality_fallback_enabled` | True | — | — | intentionally-hidden |
| `psf_spatial_enabled` | False | — | — | intentionally-hidden |
| `psf_spatial_grid` | '3x3' | — | — | intentionally-hidden |
| `psf_spatial_min_stars_per_cell` | 25 | — | — | intentionally-hidden |
| `psf_spatial_order` | 0 | 0 … 2 | — | intentionally-hidden |

### PSF production constants (hardcoded in `psf_photometry.py`)

| key | default | notes | exposed |
|-----|---------|-------|---------|
| `psf_weight_mode` | `sky_only` | Fit weights from sky + read noise only (not object Poisson) | production-constant |
| `psf_err_mode` | `sandwich_skyonly` | Reported err from true variance through sky-only weights | production-constant |
| `psf_sky_method` | per-star column | `annulus_local` / `residual_annulus` / `border_fallback` | production-constant |
| `epsf_fwhm_ratio_warn_lo` | 0.80 | EPSF-1 QC diagnostic warning (not flux gating) | production-constant |
| `epsf_fwhm_ratio_warn_hi` | 1.25 | EPSF-1 QC diagnostic warning (not flux gating) | production-constant |

## NEIGHBOR-SUB (gated OFF)

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `psf_neighbor_sub_enabled` | False | — | — | intentionally-hidden |
| `neighbor_sub_chi2_max` | 120.0 | — | — | intentionally-hidden |
| `neighbor_sub_residual_rms_max` | 150.0 | — | — | intentionally-hidden |
| `neighbor_sub_refuse_sep_fwhm` | 0.8 | — | — | intentionally-hidden |
| `neighbor_sub_centroid_max_fwhm` | 1.0 | — | — | intentionally-hidden |
| `neighbor_sub_nn_contam_dmag` | 2.5 | — | — | intentionally-hidden |
| `neighbor_sub_max_neighbor_overmag` | 0.3 | — | — | intentionally-hidden |
| `neighbor_sub_max_target_undermag` | 0.2 | — | — | intentionally-hidden |
| `neighbor_sub_min_recovered_snr` | 5.0 | — | — | intentionally-hidden |
| `neighbor_sub_regime_dmag_min` | 2.5 | — | — | intentionally-hidden |
| `neighbor_sub_regime_sep_max` | 1.1 | — | — | intentionally-hidden |

`bright_close_regime` guard: refuse when sep <= `neighbor_sub_regime_sep_max` FWHM and
delta_mag >= `neighbor_sub_regime_dmag_min` (in `psf_neighbor_sub.py`).

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
| `blind_index_fine_path` | 'C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\gaia_triangles_fine.pkl' (dataclass '') | — | ui_settings.py:158, ui_settings.py:779, ui_settings.py:781 | yes |
| `blind_index_path` | '' → runtime 'C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\gaia_triangles_fine.pkl' | — | ui_settings.py:781 | yes ⚠ not in config.json |
| `blind_index_select_mode` | 'auto' | — | — | no |
| `blind_index_wide_path` | 'C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\gaia_triangles_wide.pkl' (dataclass '') | — | ui_settings.py:193, ui_settings.py:780 | yes |
| `catalog_query_max_rows` | 15000 | 1000 … 500_000 | — | no |
| `gaia_db_path` | 'C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\vyvar_gaia_dr3.db' (dataclass '') | — | ui_hrd.py:49, ui_settings.py:123, ui_settings.py:778 | yes |
| `vsx_local_db_path` | 'C:\\ASTRO\\python\\VYVAR\\VSX\\vyvar_vsx_local_v2.db' (dataclass '') | — | ui_aperture_photometry.py:1416, ui_masterstar_qa.py:595, ui_settings.py:230 | yes |
| `vsx_variable_targets_mag_limit` | 14.5 | — | ui_aperture_photometry.py:1358, ui_settings.py:269, ui_settings.py:274 | yes |

## Phase 0+1

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `phase01_chip_interior_margin_px` | 50 | — | ui_select_stars.py:437, ui_select_stars.py:438, ui_select_stars.py:541 | yes |
| `phase01_ct_extrapolation_tol` | 0.0 | — | — | intentionally-hidden |
| `phase01_ct_min_comp` | 7 | 2 … 30 | — | intentionally-hidden |
| `phase01_flux_col` | 'dao_flux' | — | — | intentionally-hidden |
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
| `comp_max_delta_bprp` | 0.79 | 0.0 … 5.0 | ui_settings.py:654, ui_settings.py:813 | yes |
| `comp_max_slope_mmag_hr` | 5.0 | 0.0 … 500.0 | — | intentionally-hidden |
| `comp_slope_significance_k` | 3.0 | 0.0 … 10.0 | ui_settings.py (Advanced algorithms) | yes |
| `comp_tier1_bprp_limit` | 0.15 | 0.02 … 5.0 | ui_settings.py:625, ui_settings.py:809 | yes |
| `comp_tier1_weight` | 1.0 | 0.01 … 1.00 | ui_settings.py:659, ui_settings.py:662, ui_settings.py:814 | yes |
| `comp_tier2_bprp_limit` | 0.3 | 0.05 … 5.0 | ui_settings.py:633, ui_settings.py:810 | yes |
| `comp_tier2_weight` | 0.85 | 0.01 … 1.00 | ui_settings.py:666, ui_settings.py:669, ui_settings.py:815 | yes |
| `comp_tier3_bprp_limit` | 0.55 | 0.05 … 5.0 | ui_settings.py:640, ui_settings.py:811 | yes |
| `comp_tier3_weight` | 0.5 | 0.01 … 1.00 | ui_settings.py:673, ui_settings.py:676, ui_settings.py:816 | yes |
| `comp_tier4_bprp_limit` | 1.1 | 0.05 … 5.0 | ui_settings.py:647, ui_settings.py:812 | yes |
| `comp_tier4_weight` | 0.25 | 0.01 … 1.00 | ui_settings.py:680, ui_settings.py:683, ui_settings.py:817 | yes |
| `global_comp_pool_enabled` | True | — | — | intentionally-hidden |
| `phase01_comparison_exclude_gaia_extobj` | True | — | ui_select_stars.py:451, ui_select_stars.py:540, ui_settings.py:730 | yes |
| `phase01_comparison_exclude_gaia_nss` | True | — | ui_select_stars.py:450, ui_select_stars.py:539, ui_settings.py:726 | yes |
| `phase01_comparison_fov_fraction` | 0.75 | — | — | intentionally-hidden ⚠ not in config.json |
| `phase01_comparison_isolation_radius_px` | 25.0 | 1.0 … 200.0 | — | intentionally-hidden |
| `phase01_comparison_mag_bright_threshold` | 12.75 | 6.0 … 18.0 | ui_select_stars.py:443, ui_select_stars.py:531, ui_settings.py:594 | yes |
| `phase01_comparison_max_comp_rms` | 0.1 | 0.01 … 0.5 | ui_select_stars.py:447, ui_select_stars.py:536, ui_settings.py:699 | yes |
| `comp_sparse_fallback_enabled` | true | — | ui_settings.py:760, ui_settings.py:900 | yes |
| `comp_sparse_fallback_min` | 0 (= n_comp_min) | 2 … n_comp_max | ui_settings.py:768, ui_settings.py:902 | yes |
| `comp_iterative_clip_enabled` | false | — | (alias → sparse_fallback) | yes |
| `comp_clip_sigma` | 5.0 | 3.0 … 10.0 | ui_settings.py:776, ui_settings.py:904 | yes |
| `phase01_comparison_max_dist_deg` | 1.5 | 0.05 … 10.0 | ui_dao_stars.py:81, ui_dao_stars.py:191, ui_dao_stars.py:199 | yes |
| `phase01_comparison_max_fwhm_factor` | 1.5 | 0.5 … 5.0 | — | intentionally-hidden |
| `phase01_comparison_max_mag_diff` | 1.5 | 0.05 … 5.0 | ui_select_stars.py:442, ui_select_stars.py:530, ui_settings.py:80 | yes |
| `phase01_comparison_max_mag_diff_absolute` | 3.0 | 1.0 … 10.0 | ui_settings.py:611, ui_settings.py:808 | yes |
| `phase01_comparison_max_mag_diff_bright_floor` | 1.5 | 0.0 … 4.0 | ui_select_stars.py:444, ui_select_stars.py:532, ui_settings.py:601 | yes |
| `phase01_comparison_max_psf_chi2` | 50.0 | 1.0 … 500.0 | — | intentionally-hidden |
| `phase01_comparison_min_dist_arcsec` | 60.0 | 0.0 … 600.0 | ui_select_stars.py:448, ui_select_stars.py:537, ui_settings.py:706 | yes |
| `phase01_comparison_min_frames_frac` | 0.2 | 0.05 … 0.95 | ui_quality_dashboard.py:851, ui_select_stars.py:449, ui_select_stars.py:538 | yes |
| `phase01_comparison_n_comp_max` | 8 | — | ui_select_stars.py:446, ui_select_stars.py:535, ui_settings.py:693 | yes |
| `phase01_comparison_n_comp_min` | 3 | — | ui_select_stars.py:446, ui_select_stars.py:534, ui_settings.py:687 | yes |
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
| `democratic_detrend_enabled` | False | — | ui_settings.py:544, ui_settings.py:802 | yes |
| `democratic_sg_window_frac` | 0.5 | 0.05 … 0.95 | — | intentionally-hidden |
| `pytics_enabled` | True | — | ui_settings.py:533, ui_settings.py:800 | yes |
| `pytics_n_iter` | 5 | 1 … 20 | — | intentionally-hidden |
| `savgol_detrend_enabled` | False | — | ui_settings.py:539, ui_settings.py:801 | yes |
| `savgol_polyorder` | 2 | — | — | intentionally-hidden |
| `savgol_window_frac` | 0.5 | — | — | intentionally-hidden |
| `sysrem_enabled` | False | — | — | intentionally-hidden |
| `sysrem_n_iter` | 3 | — | — | intentionally-hidden |
| `temporal_bin_window` | 0 | 0 … 51 | — | intentionally-hidden |
| `temporal_binning_enabled` | True | — | ui_settings.py:528, ui_settings.py:799 | yes |

## Phase 2A output

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `save_lightcurve_png` | False | — | ui_photometry.py:72, ui_photometry.py:77, ui_photometry.py:103 | yes |

## Photometry

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `catalog_only_n_comps` | 5 | — | — | intentionally-hidden ⚠ not in config.json |

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
| `annulus_inner_fwhm` | 4.75 | 1.0 … 10.0 | ui_settings.py:77, ui_settings.py:481, ui_settings.py:484 | yes |
| `annulus_outer_fwhm` | 9.0 | 1.5 … 12.0 | ui_settings.py:77, ui_settings.py:488, ui_settings.py:491 | yes |
| `aperture_comp_factor` | 1.1 | 0.25 … 3.0 | ui_settings.py:518, ui_settings.py:798 | yes |
| `aperture_correction_enabled` | True | — | — | intentionally-hidden |
| `aperture_correction_max_contamination` | 0.15 | 0.0 … 2.0 | — | intentionally-hidden |
| `aperture_correction_max_scatter_mag` | 0.03 | 0.0 … 2.0 | — | intentionally-hidden |
| `aperture_correction_min_ref_stars` | 3 | 1 … 50 | — | intentionally-hidden |
| `aperture_fwhm_factor` | 1.9 | 0.5 … 6.0 | ui_dao_stars.py:78, ui_dao_stars.py:165, ui_dao_stars.py:188 | yes |
| `aperture_fwhm_factor_large` | 4.0 | — | — | intentionally-hidden |
| `aperture_fwhm_factor_medium` | 2.5 | — | — | intentionally-hidden |
| `aperture_fwhm_factor_small` | 1.5 | — | — | intentionally-hidden |
| `aperture_photometry_enabled` | True | — | ui_photometry.py:60, ui_photometry.py:65, ui_photometry.py:102 | yes |
| `aperture_variable_factor` | 1.0 | 0.25 … 3.0 | ui_settings.py:509, ui_settings.py:797 | yes |
| `nonlinearity_fwhm_ratio` | 1.25 | 1.01 … 3.0 | ui_settings.py:555, ui_settings.py:558, ui_settings.py:796 | yes |
| `nonlinearity_peak_percentile` | 20.0 | 0.0 … 50.0 | ui_settings.py:548, ui_settings.py:551, ui_settings.py:795 | yes |

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
| `auto_fwhm_enabled` | True | — | ui_quality_dashboard.py:510, ui_quality_dashboard.py:514, ui_quality_dashboard.py:553 | yes |
| `auto_fwhm_k_factor` | 1.5 | float(self.auto_fwhm_k_min) … float(self.auto_fwhm_k_max) | ui_quality_dashboard.py:511, ui_quality_dashboard.py:556, ui_quality_dashboard.py:569 | yes |
| `auto_fwhm_k_max` | 4.0 | — | ui_quality_dashboard.py:575 | yes |
| `auto_fwhm_k_min` | 1.0 | — | ui_quality_dashboard.py:574 | yes |

## QC / skip processed

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `skip_processed_directory` | False | — | — | intentionally-hidden |

## Sensor / noise model

| key | default | clamp/range | UI location | exposed |
|-----|---------|-------------|-------------|---------|
| `gain` | 1.0 | — | — | intentionally-hidden |
| `read_noise` | 10.0 | — | — | intentionally-hidden |
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
| `tess_enabled` | False | — | ui_aperture_photometry.py:1453, ui_variability.py:926, ui_variability.py:1651 | yes |

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

## UI references without `AppConfig` key (reverse drift)

These strings appear in UI code but are not fields on `AppConfig`:

- `phase01_comparison_max_bv_diff` → ui_select_stars.py:445, ui_select_stars.py:533
- `phase01_use_bprp_primary` → ui_aperture_photometry.py:1657, ui_select_stars.py:620
