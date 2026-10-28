"""Keys removed from config.json over time, with the reason shown when an old file still has one.

Moved out of ``config.py`` (code review 2 G13-40) so the loader module holds configuration logic only.
``config.load_config_json`` logs INFO once per removed key it finds (no difflib WARN).
"""

from __future__ import annotations

KNOWN_REMOVED_KEYS: dict[str, str] = {
    "per_frame_mp_reserve_ram_gb": (
        "per_frame_mp_reserve_ram_gb removed 2026-10-26 (RESOURCES-2026-10): a run uses half of the RAM and "
        "half of the CPUs and sizes every stage by its frame size by itself"
    ),
    "gain": (
        "gain removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): gain comes from the FITS header or EQUIPMENTS.GAIN_ADU; unknown -> error (no default)"
    ),
    "phase01_chip_interior_margin_px": (
        "phase01_chip_interior_margin_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): the edge margin is the sky-annulus radius (annulus_outer_fwhm x FWHM) or the annulus-aware safe box"
    ),
    "phase01_comparison_isolation_radius_px": (
        "phase01_comparison_isolation_radius_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): the crude neighbour-flux contamination index it sized was removed; neighbour light is the comparison pool's aperture blend model"
    ),
    "phase01_suspected_interior_margin_px": (
        "phase01_suspected_interior_margin_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): the edge margin is the sky-annulus radius (annulus_outer_fwhm x FWHM) or the annulus-aware safe box"
    ),
    "read_noise": (
        "read_noise removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): read noise comes from the FITS header or EQUIPMENTS.READNOISE_E; unknown -> error (no default)"
    ),
    "sips_dao_fwhm_px": (
        "sips_dao_fwhm_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7B): the DAO FWHM is the measured VY_FWHM of each frame; without one the fixed fallback pipeline_constants.DAO_FWHM_FALLBACK_PX = 3 px is used"
    ),
    "alignment_max_control_points": (
        "alignment_max_control_points removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.ALIGNMENT_MAX_CONTROL_POINTS = 80); alignment_max_stars stays"
    ),
    "check_select_rms_floor": (
        "check_select_rms_floor removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (check_star_kmag.CHECK_SELECT_RMS_FLOOR = 1e-4)"
    ),
    "check_star_min_epochs": (
        "check_star_min_epochs removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (trust_flag_core.CHECK_STAR_MIN_EPOCHS = 5)"
    ),
    "comp_contamination_penalty_k": (
        "comp_contamination_penalty_k removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (comp_selection_per_target.COMP_CONTAMINATION_PENALTY_K = 3)"
    ),
    "comp_rms_loo_photon_k": (
        "comp_rms_loo_photon_k removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (comp_rms_loo.COMP_RMS_LOO_PHOTON_K_DEFAULT = 5)"
    ),
    "comp_select_rms_floor": (
        "comp_select_rms_floor removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.COMP_SELECT_RMS_FLOOR = 1e-6)"
    ),
    "comp_slope_significance_k": (
        "comp_slope_significance_k removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed default of check_comparison_stability now (3 sigma)"
    ),
    "comparison_stars_pool_n": (
        "comparison_stars_pool_n removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the comparison pool is uncapped (was always 0)"
    ),
    "dao_centroid_max_shift_fwhm": (
        "dao_centroid_max_shift_fwhm removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.DAO_CENTROID_MAX_SHIFT_FWHM = 1)"
    ),
    "epsf_min_stars": (
        "epsf_min_stars removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.EPSF_MIN_BUILD_STARS = 30)"
    ),
    "err_empty_apertures_min": (
        "err_empty_apertures_min removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): fixed defaults of the empty-aperture sky error (64 apertures, at least 16)"
    ),
    "err_empty_apertures_n": (
        "err_empty_apertures_n removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): fixed defaults of the empty-aperture sky error (64 apertures, at least 16)"
    ),
    "exoplanet_match_max_sep_arcsec": (
        "exoplanet_match_max_sep_arcsec removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.EXOPLANET_MATCH_MAX_SEP_ARCSEC = 3 arcsec)"
    ),
    "forced_photometry_enabled": (
        "forced_photometry_enabled removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): forced photometry of MASTERSTAR stars is always on, without extra margin"
    ),
    "forced_photometry_margin_px": (
        "forced_photometry_margin_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): forced photometry of MASTERSTAR stars is always on, without extra margin"
    ),
    "frame_height_px": (
        "frame_height_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the frame size always comes from FITS NAXIS1/2"
    ),
    "frame_width_px": (
        "frame_width_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the frame size always comes from FITS NAXIS1/2"
    ),
    "gs11_dilution_aperture_arcsec": (
        "gs11_dilution_aperture_arcsec removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the dilution aperture is always the target's photometric aperture (0 = auto was the only consistent value)"
    ),
    "gs11_dilution_mag_limit_delta": (
        "gs11_dilution_mag_limit_delta removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed default of compute_dilution_factor now (5 mag)"
    ),
    "gs11_target_min_dilution": (
        "gs11_target_min_dilution removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (dilution.GS11_TARGET_MIN_DILUTION = 0.5)"
    ),
    "hrd_dsc_confirm_prob": (
        "hrd_dsc_confirm_prob removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_enrich_max_candidates": (
        "hrd_enrich_max_candidates removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_enrich_tap_timeout_s": (
        "hrd_enrich_tap_timeout_s removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_max_per_category": (
        "hrd_max_per_category removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_min_per_net": (
        "hrd_min_per_net removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_parallax_min_mas": (
        "hrd_parallax_min_mas removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_parallax_snr_min": (
        "hrd_parallax_snr_min removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed HRD constant now (hrd_analysis.HRD_*), same value"
    ),
    "hrd_simbad_enrich_enabled": (
        "hrd_simbad_enrich_enabled removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): merged into hrd_online_enrich_enabled (one switch for Gaia TAP and SIMBAD)"
    ),
    "lc_high_err_nsigma": (
        "lc_high_err_nsigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the four 5-sigma LC flag thresholds are one constant now (lc_outlier.OUTLIER_N_SIGMA = 5)"
    ),
    "lc_outlier_adjacent_sigma": (
        "lc_outlier_adjacent_sigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (lc_outlier.ADJACENT_SIGMA = 3)"
    ),
    "lc_outlier_enabled": (
        "lc_outlier_enabled removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): LC outlier flagging is always on (flags never change photometry)"
    ),
    "lc_outlier_evidence_n_sigma": (
        "lc_outlier_evidence_n_sigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the four 5-sigma LC flag thresholds are one constant now (lc_outlier.OUTLIER_N_SIGMA = 5)"
    ),
    "lc_outlier_frame_qc_n_sigma": (
        "lc_outlier_frame_qc_n_sigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the four 5-sigma LC flag thresholds are one constant now (lc_outlier.OUTLIER_N_SIGMA = 5)"
    ),
    "lc_outlier_n_sigma": (
        "lc_outlier_n_sigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the four 5-sigma LC flag thresholds are one constant now (lc_outlier.OUTLIER_N_SIGMA = 5)"
    ),
    "lc_quality_min_normal_frac": (
        "lc_quality_min_normal_frac removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed default of classify_lc_quality now (3 frames, 0.5)"
    ),
    "lc_quality_short_min_frames": (
        "lc_quality_short_min_frames removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed default of classify_lc_quality now (3 frames, 0.5)"
    ),
    "masterstar_dao_centroid_qa_cap_px": (
        "masterstar_dao_centroid_qa_cap_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_dao_centroid_qa_floor_px": (
        "masterstar_dao_centroid_qa_floor_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_dao_empty_sky_false_accept_max": (
        "masterstar_dao_empty_sky_false_accept_max removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): read by nobody since the empty-sky audit was removed"
    ),
    "masterstar_dao_empty_sky_target_n": (
        "masterstar_dao_empty_sky_target_n removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): read by nobody since the empty-sky audit was removed"
    ),
    "masterstar_dao_match_radius_k": (
        "masterstar_dao_match_radius_k removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_dao_pass1_dedup_px": (
        "masterstar_dao_pass1_dedup_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_dao_pass2_center_tol_px": (
        "masterstar_dao_pass2_center_tol_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_dao_pass2_sigma": (
        "masterstar_dao_pass2_sigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_forced_seed_centroid_max_px": (
        "masterstar_forced_seed_centroid_max_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_forced_seed_comp_pool_enabled": (
        "masterstar_forced_seed_comp_pool_enabled removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): forced seeds never enter the comparison pool (was always off)"
    ),
    "masterstar_forced_seed_snr_min": (
        "masterstar_forced_seed_snr_min removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_gaia_census_edge_margin_px": (
        "masterstar_gaia_census_edge_margin_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_lock_leftover_radius_px": (
        "masterstar_lock_leftover_radius_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_lock_pair_tol_px": (
        "masterstar_lock_pair_tol_px removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed MASTERSTAR constant now (pipeline_constants.MS_*), same value"
    ),
    "masterstar_use_best_frame_fwhm": (
        "masterstar_use_best_frame_fwhm removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): always on: the MASTERSTAR DAO FWHM is the best frame's FWHM"
    ),
    "phase01_comparison_min_dist_arcsec": (
        "phase01_comparison_min_dist_arcsec removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the target-comparison separation comes from geometry (annulus + aperture x FWHM)"
    ),
    "phase01_ct_extrapolation_tol": (
        "phase01_ct_extrapolation_tol removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the colour term is applied only inside the comparison-star colour range (was always 0)"
    ),
    "phase01_ct_min_comp": (
        "phase01_ct_min_comp removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (pipeline_constants.CT_MIN_COMP = 7)"
    ),
    "plate_solve_fov_deg": (
        "plate_solve_fov_deg removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the value was never used (the field size comes from FOCALLEN + PIXSIZE + NAXIS)"
    ),
    "preprocess_sky_surface_force_reapply": (
        "preprocess_sky_surface_force_reapply removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): sky surface is subtracted once; a new order needs recalibration from raw"
    ),
    "psf_chi2_locus_nsigma": (
        "psf_chi2_locus_nsigma removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (psf_chi2_locus.DEFAULT_NSIGMA = 5)"
    ),
    "sparse_trust_T_green": (
        "sparse_trust_T_green removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the sparse-trust band limits are the SparseTrustConfig defaults (1.5 / 4.0 / 0.0004)"
    ),
    "sparse_trust_T_red": (
        "sparse_trust_T_red removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the sparse-trust band limits are the SparseTrustConfig defaults (1.5 / 4.0 / 0.0004)"
    ),
    "sparse_trust_X2_RED": (
        "sparse_trust_X2_RED removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): the sparse-trust band limits are the SparseTrustConfig defaults (1.5 / 4.0 / 0.0004)"
    ),
    "tess_contam_max": (
        "tess_contam_max removed 2026-10-22 (CODE-REVIEW-2026-10 step 7): a fixed constant now (tess_analysis.TESS_CONTAM_MAX = 0.10)"
    ),
    "color_level_k_mag_per_bprp": (
        "color_level_k_mag_per_bprp removed 2026-10-19 (CFG-REVIEW-2026-10): the Clear-filter level colour coefficient was never applied since BANDAUTH-2026-10 (Clear is reported untransformed, TRANS=NO); it only kept the trust gate from seeing the colour term as off"
    ),
    "color_level_k_stderr_mag_per_bprp": (
        "color_level_k_stderr_mag_per_bprp removed 2026-10-19 (CFG-REVIEW-2026-10): the Clear-filter level colour coefficient was never applied since BANDAUTH-2026-10 (Clear is reported untransformed, TRANS=NO); it only kept the trust gate from seeing the colour term as off"
    ),
    "sigma_sys_mag": (
        "sigma_sys_mag removed 2026-10-19 (CFG-REVIEW-2026-10): a per-camera property; it is EQUIPMENTS.SIGMA_SYS_MAG now"
    ),
    "gain_container_scale": (
        "gain_container_scale removed 2026-10-19 (CFG-REVIEW-2026-10): a per-camera property; it is EQUIPMENTS.ADC_BITS now (container scale = 2^(16 - ADC_BITS))"
    ),
    "dao_qc_in_calibrate": (
        "dao_qc_in_calibrate removed 2026-10-19 (CFG-REVIEW-2026-10): the DAO QC during calibration (PERF-10) always runs"
    ),
    "qc_max_hfr": (
        "qc_max_hfr removed 2026-10-19 (CFG-REVIEW-2026-10): the post-calibration QC limits only set a pass/fail flag nobody read (frames were always kept); the QC measurements stay"
    ),
    "qc_min_stars": (
        "qc_min_stars removed 2026-10-19 (CFG-REVIEW-2026-10): the post-calibration QC limits only set a pass/fail flag nobody read (frames were always kept); the QC measurements stay"
    ),
    "qc_max_background_rms": (
        "qc_max_background_rms removed 2026-10-19 (CFG-REVIEW-2026-10): the post-calibration QC limits only set a pass/fail flag nobody read (frames were always kept); the QC measurements stay"
    ),
    "k2_fit_enabled": (
        "k2_fit_enabled removed 2026-10-19 (CFG-REVIEW-2026-10): the night fit of k'' (NIGHT_FIT v2, never enabled) is removed; k'' comes from the literature defaults (k2_mode literature/off)"
    ),
    "k2_fit_min_detectability": (
        "k2_fit_min_detectability removed 2026-10-19 (CFG-REVIEW-2026-10): the night fit of k'' (NIGHT_FIT v2, never enabled) is removed; k'' comes from the literature defaults (k2_mode literature/off)"
    ),
    "k2_fit_consistency_sigma": (
        "k2_fit_consistency_sigma removed 2026-10-19 (CFG-REVIEW-2026-10): the night fit of k'' (NIGHT_FIT v2, never enabled) is removed; k'' comes from the literature defaults (k2_mode literature/off)"
    ),
    "k2_fit_lit_factor": (
        "k2_fit_lit_factor removed 2026-10-19 (CFG-REVIEW-2026-10): the night fit of k'' (NIGHT_FIT v2, never enabled) is removed; k'' comes from the literature defaults (k2_mode literature/off)"
    ),
    "k2_ceiling": (
        "k2_ceiling removed 2026-10-19 (CFG-REVIEW-2026-10): the night fit of k'' (NIGHT_FIT v2, never enabled) is removed; k'' comes from the literature defaults (k2_mode literature/off)"
    ),
    "aperture_policy_mode": (
        "aperture_policy_mode removed 2026-10-19 (CFG-REVIEW-2026-10): per_target is the only aperture policy (APERTURE-K-2026-10); f_fixed_night / f_per_frame are removed"
    ),
    "cog_aperture_correction_enabled": (
        "cog_aperture_correction_enabled removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_ref_fwhm": (
        "cog_ref_fwhm removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_min_stars": (
        "cog_min_stars removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_isolation_fwhm": (
        "cog_isolation_fwhm removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_snr_min": (
        "cog_snr_min removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_ladder_step_px": (
        "cog_ladder_step_px removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "cog_ac_factor_max": (
        "cog_ac_factor_max removed 2026-10-19 (CFG-REVIEW-2026-10): the per-frame curve-of-growth aperture correction (default off, never validated) is removed; per-target k* apertures (APERTURE-K-2026-10) replace it"
    ),
    "aperture_photometry_enabled": (
        "aperture_photometry_enabled removed 2026-10-19 (CFG-REVIEW-2026-10): aperture photometry is the only photometry of the per-frame catalogs"
    ),
    "psf_photometry_enabled": (
        "psf_photometry_enabled removed 2026-10-19 (CFG-REVIEW-2026-10): the ePSF light curves are a separate product (ePSF tab, epsf_export_auto); reports and exports are aperture photometry"
    ),
    "blind_img_select_mode": (
        "blind_img_select_mode removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_img_star_budget": (
        "blind_img_star_budget removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_index_fine_path": (
        "blind_index_fine_path removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_index_select_mode": (
        "blind_index_select_mode removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_index_wide_path": (
        "blind_index_wide_path removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_use_rig_prior": (
        "blind_use_rig_prior removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_early_accept": (
        "blind_verify_early_accept removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_early_floor": (
        "blind_verify_early_floor removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_early_fraction": (
        "blind_verify_early_fraction removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_enabled": (
        "blind_verify_enabled removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_inmemory_catalog": (
        "blind_verify_inmemory_catalog removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_match_tol_px": (
        "blind_verify_match_tol_px removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_min_fraction": (
        "blind_verify_min_fraction removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_min_matches": (
        "blind_verify_min_matches removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_verify_top_n": (
        "blind_verify_top_n removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "catalog_query_max_rows": (
        "catalog_query_max_rows removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "debug_platesolver": (
        "debug_platesolver removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_catalog_recovery_min": (
        "masterstar_catalog_recovery_min removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_centre_rms_max_px": (
        "masterstar_centre_rms_max_px removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_detection_cap_adaptive": (
        "masterstar_detection_cap_adaptive removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_detection_cap_k": (
        "masterstar_detection_cap_k removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_detection_cap_max": (
        "masterstar_detection_cap_max removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_detection_cap_min": (
        "masterstar_detection_cap_min removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_distortion_benign_ratio_max": (
        "masterstar_distortion_benign_ratio_max removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_min_matched_floor": (
        "masterstar_min_matched_floor removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_platesolve_sip_max_order": (
        "masterstar_platesolve_sip_max_order removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_platesolve_sip_min_order": (
        "masterstar_platesolve_sip_min_order removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "masterstar_quality_crowded_n_cat_min": (
        "masterstar_quality_crowded_n_cat_min removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "verify_mag_limit": (
        "verify_mag_limit removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "blind_index_path": (
        "blind_index_path removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "BLIND_INDEX_PATH": (
        "BLIND_INDEX_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "BLIND_INDEX_FINE_PATH": (
        "BLIND_INDEX_FINE_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "BLIND_INDEX_WIDE_PATH": (
        "BLIND_INDEX_WIDE_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): it configured the old VYVAR plate solver / blind solver, unused since astrosolve solves every MASTERSTAR (PLATESOLVE-2026-10); the solver code is removed too"
    ),
    "psf_zp_membership": (
        "psf_zp_membership removed 2026-10-19 (CFG-REVIEW-2026-10): both modes admitted the same epochs (psf_fit_ok and a positive flux) since FIT-OK-ADMISSION-01"
    ),
    "psf_zp_for_zp_validated_rigs": (
        "psf_zp_for_zp_validated_rigs removed 2026-10-19 (CFG-REVIEW-2026-10): only selected between two identical psf_zp_membership modes"
    ),
    "calibration_library_native_binning": (
        "calibration_library_native_binning removed 2026-10-19 (CFG-REVIEW-2026-10): it was ignored on load since CAL-2026-10 (master binning comes from each master header)"
    ),
    "vsx_local_db_path": (
        "vsx_local_db_path removed 2026-10-20 (CATALOG-DB-2026-10): VSX is the table vsx_data of objects_db_path (catalogs/vyvar_objects.db, built by catalogs/objects_make.py)"
    ),
    "exoplanet_local_db_path": (
        "exoplanet_local_db_path removed 2026-10-20 (CATALOG-DB-2026-10): exoplanets are the table exoplanet_data of objects_db_path (catalogs/vyvar_objects.db, built by catalogs/objects_make.py)"
    ),
    "GAIA_DB_PATH": (
        "GAIA_DB_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): legacy alias - use gaia_db_path"
    ),
    "VSX_LOCAL_DB_PATH": (
        "VSX_LOCAL_DB_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): legacy alias - use objects_db_path"
    ),
    "EXOPLANET_LOCAL_DB_PATH": (
        "EXOPLANET_LOCAL_DB_PATH removed 2026-10-19 (CFG-REVIEW-2026-10): legacy alias - use objects_db_path"
    ),
    "trust_flag_enabled": (
        "trust_flag_enabled removed 2026-10-22 (TRUST-ALWAYS-2026-10): trust always runs; with it off RED targets were exported for submission"
    ),
    "photon_transfer_ci_max_width_factor": (
        "photon_transfer_ci_max_width_factor removed 2026-10-22 (GAIN-ONE-2026-10): the photon-transfer gain is gone; gain = header / EQUIPMENTS / ADC_BITS"
    ),
    "phase01_plate_scale_arcsec_per_px": (
        "phase01_plate_scale_arcsec_per_px removed 2026-10-22 (PLATE-SCALE-2026-10): the plate scale comes from the solved MASTERSTAR WCS"
    ),
    "plate_scale_arcsec_per_px": (
        "plate_scale_arcsec_per_px removed 2026-10-22 (PLATE-SCALE-2026-10): the plate scale comes from the solved MASTERSTAR WCS"
    ),
    "comp_pool_derived_admission": (
        "comp_pool_derived_admission removed 2026-10-22 (COMP-ADMIT-2026-10): known variables (VSX, Gaia DR3 variability catalogue) are always excluded; the derived noise-curve admission only recomputed that test"
    ),
    "masterstar_prematch_peak_sigma_floor": (
        "masterstar_prematch_peak_sigma_floor removed 2026-10-20 (DAO-FALSE-2026-10): the peak filter before the Gaia match had no effect at the detection threshold (530: 1 of 1957 dropped); false detections are measured on every MASTERSTAR instead (pipeline_meta dao_false_detection)"
    ),
    "observer_location_id": (
        "observer_location_id removed 2026-10-20 (SETTINGS-2026-10): the default site is LOCATION.IS_DEFAULT in the database (Database Explorer); the site of a run is chosen on the import page or with --location-id"
    ),
    "aavso_observer_code": (
        "aavso_observer_code removed 2026-10-19 (CFG-REVIEW-2026-10): legacy alias - use observer_code"
    ),
    "aperture_fwhm_factor_medium": (
        "aperture_fwhm_factor_medium removed (legacy, never read)"
    ),
    "phase01_flux_col": (
        "phase01_flux_col removed 2026-10-19 (CFG-REVIEW-2026-10): comparison-star selection runs before RUN ePSF, so dao_flux was its only usable value"
    ),
    "comp_iterative_clip_enabled": (
        "comp_iterative_clip_enabled removed (COMP-RULES-2026-10: no iterative comp clip)"
    ),
    "comp_tier1_bprp_limit": (
        "comp_tier1_bprp_limit removed 2026-10-19 (CFG-REVIEW-2026-10): legacy form of comp_color_tiers"
    ),
    "comp_tier1_weight": (
        "comp_tier1_weight removed (comparison ensemble weights are Broeg weights)"
    ),
    "comp_tier2_bprp_limit": (
        "comp_tier2_bprp_limit removed 2026-10-19 (CFG-REVIEW-2026-10): legacy form of comp_color_tiers"
    ),
    "comp_tier2_weight": (
        "comp_tier2_weight removed (comparison ensemble weights are Broeg weights)"
    ),
    "comp_tier3_bprp_limit": (
        "comp_tier3_bprp_limit removed 2026-10-19 (CFG-REVIEW-2026-10): legacy form of comp_color_tiers"
    ),
    "comp_tier3_weight": (
        "comp_tier3_weight removed (comparison ensemble weights are Broeg weights)"
    ),
    "comp_tier4_bprp_limit": (
        "comp_tier4_bprp_limit removed 2026-10-19 (CFG-REVIEW-2026-10): legacy form of comp_color_tiers"
    ),
    "comp_tier4_weight": (
        "comp_tier4_weight removed (comparison ensemble weights are Broeg weights)"
    ),
    "psf_spatial_order": (
        "psf_spatial_order removed 2026-10-18 (EPSF-SPATIAL-2026-10: the spatial ePSF is the night-averaged 3x3/2x2 perturbation chosen by the build-star count)"
    ),
    "psf_chi2_threshold": (
        "psf_chi2_threshold removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_fit_ok is set by the night chi2 locus (psf_chi2_locus_nsigma); the legacy absolute cut had no consumer)"
    ),
    "psf_grouper_enabled": (
        "psf_grouper_enabled removed 2026-10-18 (EPSF-SPATIAL-2026-10: the per-star cutout fitter with SourceGrouper is replaced by the frame-level fit)"
    ),
    "psf_group_sep_fwhm": (
        "psf_group_sep_fwhm removed 2026-10-18 (EPSF-SPATIAL-2026-10: the per-star cutout fitter with SourceGrouper is replaced by the frame-level fit)"
    ),
    "psf_neighbor_include_fwhm": (
        "psf_neighbor_include_fwhm removed 2026-10-18 (EPSF-SPATIAL-2026-10: the per-star cutout fitter with SourceGrouper is replaced by the frame-level fit)"
    ),
    "psf_neighbor_sub_enabled": (
        "psf_neighbor_sub_enabled removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_chi2_max": (
        "neighbor_sub_chi2_max removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_residual_rms_max": (
        "neighbor_sub_residual_rms_max removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_refuse_sep_fwhm": (
        "neighbor_sub_refuse_sep_fwhm removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_centroid_max_fwhm": (
        "neighbor_sub_centroid_max_fwhm removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_nn_contam_dmag": (
        "neighbor_sub_nn_contam_dmag removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_max_neighbor_overmag": (
        "neighbor_sub_max_neighbor_overmag removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_max_target_undermag": (
        "neighbor_sub_max_target_undermag removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_min_recovered_snr": (
        "neighbor_sub_min_recovered_snr removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_regime_dmag_min": (
        "neighbor_sub_regime_dmag_min removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "neighbor_sub_regime_sep_max": (
        "neighbor_sub_regime_sep_max removed 2026-10-18 (EPSF-SPATIAL-2026-10: psf_neighbor_sub had no production caller and is removed)"
    ),
    "psf_spatial_enabled": (
        "psf_spatial_enabled removed 2026-10-18 (EPSF-SPATIAL-2026-10: the spatial ePSF is always on (night-averaged perturbation, grid by build-star count))"
    ),
    "psf_quality_fallback_enabled": (
        "psf_quality_fallback_enabled removed 2026-10-18 (EPSF-SPATIAL-2026-10: the per-star quality grade went with the cutout fitter; the night chi2 locus sets psf_fit_ok)"
    ),
    "psf_adaptive_enabled": (
        "psf_adaptive_enabled removed 2026-10-18 (EPSF-SPATIAL-2026-10: the adaptive aperture/PSF picker had no inputs (psf_ac_applied, psf_quality) after the per-frame ePSF; published light curves are aperture, PSF light curves are internal)"
    ),
    "psf_adaptive_resolve_fwhm": (
        "psf_adaptive_resolve_fwhm removed 2026-10-18 (EPSF-SPATIAL-2026-10: the adaptive aperture/PSF picker is removed)"
    ),
    "psf_adaptive_snr_lo": (
        "psf_adaptive_snr_lo removed 2026-10-18 (EPSF-SPATIAL-2026-10: the adaptive aperture/PSF picker is removed)"
    ),
    "photometry_mode": (
        "photometry_mode removed 2026-10-18 (EPSF-SPATIAL-2026-10); aperture photometry always runs on the MASTERSTAR catalogue, ePSF photometry is the separate RUN ePSF stage with a per-frame ePSF (epsf_auto_run)"
    ),
    "hrd_color_field_enabled": (
        "hrd_color_field_enabled removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_saturation": (
        "hrd_color_saturation removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_highlight_mode": (
        "hrd_color_highlight_mode removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_chroma_snr": (
        "hrd_color_chroma_snr removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_white_point": (
        "hrd_color_white_point removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_chroma_boost": (
        "hrd_color_chroma_boost removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "hrd_color_bg_box_px": (
        "hrd_color_bg_box_px removed 2026-10-18 (HRD-TAB-2026-10); the field image has fixed modes (inverted, catalogue colour, physical blackbody) chosen in the HRD view"
    ),
    "bpm_dark_mad_sigma": (
        "bpm_dark_mad_sigma removed 2026-10-17 (BPM-REMOVE-2026-10); dark bad-column map retired (Milan: not used; the *_dark_bpm.json sidecar was never written)"
    ),
    "forced_photometry_centroid_bound_fwhm": (
        "forced_photometry_centroid_bound_fwhm removed 2026-10-17 (CFG-CLEAN-2026-10-17); forced photometry uses the MASTERSTAR position (no refine since POS-SUBPIX-2026-10), the bound was only echoed into debug meta"
    ),
    "blind_verify_match_tol_arcsec": (
        "blind_verify_match_tol_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "cog_ladder_step_fwhm": (
        "cog_ladder_step_fwhm removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "hrd_color_bg_box_arcsec": (
        "hrd_color_bg_box_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "masterstar_centre_rms_max_arcsec": (
        "masterstar_centre_rms_max_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "masterstar_sibling_recovery_enabled": (
        "masterstar_sibling_recovery_enabled removed 2026-10-20 (BLIND-SOLVE-2026-10); a filter that did not solve is solved by astrosolve with the hint of a solved sibling filter - one plate-solve code, no separate sibling odds"
    ),
    "masterstar_sibling_min_matched": (
        "masterstar_sibling_min_matched removed 2026-10-20 (BLIND-SOLVE-2026-10); a filter that did not solve is solved by astrosolve with the hint of a solved sibling filter - one plate-solve code, no separate sibling odds"
    ),
    "masterstar_sibling_rms_max_px": (
        "masterstar_sibling_rms_max_px removed 2026-10-20 (BLIND-SOLVE-2026-10); a filter that did not solve is solved by astrosolve with the hint of a solved sibling filter - one plate-solve code, no separate sibling odds"
    ),
    "masterstar_sibling_min_quadrants": (
        "masterstar_sibling_min_quadrants removed 2026-10-20 (BLIND-SOLVE-2026-10); a filter that did not solve is solved by astrosolve with the hint of a solved sibling filter - one plate-solve code, no separate sibling odds"
    ),
    "masterstar_sibling_stack_n": (
        "masterstar_sibling_stack_n removed 2026-10-20 (BLIND-SOLVE-2026-10); a filter that did not solve is solved by astrosolve with the hint of a solved sibling filter - one plate-solve code, no separate sibling odds"
    ),
    "masterstar_sibling_rms_max_arcsec": (
        "masterstar_sibling_rms_max_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "phase01_chip_interior_margin_arcsec": (
        "phase01_chip_interior_margin_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "phase01_comparison_isolation_radius_arcsec": (
        "phase01_comparison_isolation_radius_arcsec removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "qc_max_hfr_fwhm_ratio": (
        "qc_max_hfr_fwhm_ratio removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "sips_dao_fwhm_fwhm_factor": (
        "sips_dao_fwhm_fwhm_factor removed 2026-10-17 (CFG-UNITS-2026-10); one key per meaning - the pixel key is the only one (it was always the one in use)"
    ),
    "auto_fwhm_k_min": (
        "auto_fwhm_k_min removed 2026-10-17 (CFG-CLEAN-2026-10-17); auto_fwhm_k_factor is clamped to 1..4 directly"
    ),
    "auto_fwhm_k_max": (
        "auto_fwhm_k_max removed 2026-10-17 (CFG-CLEAN-2026-10-17); auto_fwhm_k_factor is clamped to 1..4 directly"
    ),
    "phase01_comparison_exclude_gaia_nss": (
        "phase01_comparison_exclude_gaia_nss removed 2026-10-14; a Gaia binary that does not vary is a "
        "valid comparison - comparisons must not be variable (VSX, Gaia variability flag, own stability test)"
    ),
    "masterstar_gaia_census_target_depth_g": (
        "masterstar_gaia_census_target_depth_g removed 2026-10-14; the DAO <-> Gaia matching depth is "
        "measured on each MASTERSTAR (FRAME-DEPTH-2026-10)"
    ),
    "phase01_comparison_exclude_gaia_extobj": "COMP-NSS-2026-10: the local Gaia DB has no QSO/galaxy columns - the filter never ran; key removed",
    "phase01_comparison_max_fwhm_factor": "PHASE1-ONE-PASS-2026-10: the FWHM blend filter only fed a diagnostic column; blends are gated by the neighbour-light rule",
    "gs11_comp_max_dilution": "PHASE1-ONE-PASS-2026-10: dilution never rejected a comparison star (the counter could not increment); key removed",
    "frame_align_residual_gate_enabled": "ALIGN-VERIFY-2026-10: the circular aperture-photometry alignment-residual gate is removed; every registration is verified on star pairs at alignment time and an unverified frame is not measured",
    "frame_align_residual_max_frac": "ALIGN-VERIFY-2026-10: the circular aperture-photometry alignment-residual gate is removed; every registration is verified on star pairs at alignment time and an unverified frame is not measured",
    "frame_align_residual_min_keep_frames": "ALIGN-VERIFY-2026-10: the circular aperture-photometry alignment-residual gate is removed; every registration is verified on star pairs at alignment time and an unverified frame is not measured",
    "comp_qa_enabled": "COMP-QA-2026-10: Sokolovsky LOO comp QA removed; comps are gated at selection (not variable in Gaia/VSX, neighbour light, frame edge)",
    "variability_min_frames": (
        "variability_min_frames removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_min_frames_frac": (
        "variability_min_frames_frac removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_p85_filter": (
        "variability_p85_filter removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_slope_floor": (
        "variability_slope_floor removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_sigma_threshold": (
        "variability_sigma_threshold removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_comp_floor_factor": (
        "variability_comp_floor_factor removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_smoothness_max": (
        "variability_smoothness_max removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_mag_limit": (
        "variability_mag_limit removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_min_rms_pct": (
        "variability_min_rms_pct removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_min_amplitude_mag": (
        "variability_min_amplitude_mag removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_clip_ratio_min": (
        "variability_clip_ratio_min removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_vdi_z_threshold": (
        "variability_vdi_z_threshold removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "variability_min_points_rms": (
        "variability_min_points_rms removed 2026-10-10 VAR-2026-10: the field search derives its noise model and threshold from the data"
    ),
    "aperture_f_grid": (
        "aperture_f_grid removed 2026-10; fixed k grid 0.6-2.0 x FWHM, k* by comparison-star "
        "scatter (APERTURE-K-2026-10)"
    ),
    "skip_processed_directory": (
        "skip_processed_directory removed 2026-07; skip behavior is now always on"
    ),
    "vsx_variable_targets_mag_limit": (
        "vsx_variable_targets_mag_limit removed 2026-07; VSX scope is detection-limited (DAO+Gaia match)"
    ),
    "phase01_match_radius_arcsec": (
        "phase01_match_radius_arcsec removed 2026-07; target selection uses catalog_id identity join (PHASE0-IDENTITY-GATE)"
    ),
    "aperture_selection_criterion": (
        "aperture_selection_criterion removed 2026-08-31 CONSOLIDATE-01B; scatter/SNR diagnostic table deleted"
    ),
    "aperture_scatter_r_min_px": (
        "aperture_scatter_r_min_px removed 2026-08-31 CONSOLIDATE-01B; scatter ladder module deleted"
    ),
    "aperture_scatter_r_max_px": (
        "aperture_scatter_r_max_px removed 2026-08-31 CONSOLIDATE-01B; scatter ladder module deleted"
    ),
    "aperture_scatter_r_step_px": (
        "aperture_scatter_r_step_px removed 2026-08-31 CONSOLIDATE-01B; scatter ladder module deleted"
    ),
    "global_comp_pool_enabled": (
        "global_comp_pool_enabled removed 2026-09-01 CONSOLIDATE-01D; global pool is always on (COMP-POOL-01)"
    ),
    "export_err_mode": (
        "export_err_mode removed 2026-09-01 CONSOLIDATE-01D (the ERR-CALIB layer itself was removed in CODE-REVIEW-2026-10 step 6: its sidecar was never written)"
    ),
    "err_background_mode": (
        "err_background_mode removed 2026-09-01 CONSOLIDATE-01D; empirical empty-aperture term is always on (F-BINGAIN-1); Howell math remains as missing-sigma fallback"
    ),
    "masterstar_accept_mode": (
        "masterstar_accept_mode removed 2026-09-01 CONSOLIDATE-01D; MASTERSTAR odds gate is the only accept policy"
    ),
    "psf_ac_policy": (
        "psf_ac_policy removed 2026-09-01 CONSOLIDATE-01D; p4_none is the only ePSF AC policy (ZP-OK v2 / P4)"
    ),
    "qc_fwhm_limit": (
        "qc_fwhm_limit removed 2026-09-01 CONSOLIDATE-01D; persist-only, never consumed"
    ),
    "qc_elong_limit": (
        "qc_elong_limit removed 2026-09-01 CONSOLIDATE-01D; persist-only, never consumed"
    ),
    "psf_spatial_grid": (
        "psf_spatial_grid removed 2026-09-01 CONSOLIDATE-01D; persist-only, never consumed"
    ),
    "psf_spatial_min_stars_per_cell": (
        "psf_spatial_min_stars_per_cell removed 2026-09-01 CONSOLIDATE-01D; persist-only, never consumed"
    ),
    "gs11_comp_suspect_dilution": (
        "gs11_comp_suspect_dilution removed 2026-09-01 CONSOLIDATE-01D; persist-only, never consumed"
    ),
    "phase01_use_bprp_primary": (
        "phase01_use_bprp_primary removed 2026-09-01 CONSOLIDATE-01D; BP-RP is always the color display basis"
    ),
    "aperture_correction_enabled": (
        "aperture_correction_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; aperture correction Method B deleted (mag_calib_final = mag_calib + CT)"
    ),
    "aperture_correction_min_ref_stars": (
        "aperture_correction_min_ref_stars removed 2026-10-10 SPECIAL-FNS-REMOVE; aperture correction Method B deleted (mag_calib_final = mag_calib + CT)"
    ),
    "aperture_correction_max_contamination": (
        "aperture_correction_max_contamination removed 2026-10-10 SPECIAL-FNS-REMOVE; aperture correction Method B deleted (mag_calib_final = mag_calib + CT)"
    ),
    "aperture_correction_max_scatter_mag": (
        "aperture_correction_max_scatter_mag removed 2026-10-10 SPECIAL-FNS-REMOVE; aperture correction Method B deleted (mag_calib_final = mag_calib + CT)"
    ),
    "temporal_binning_enabled": (
        "temporal_binning_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-3 temporal binning deleted"
    ),
    "temporal_bin_window": (
        "temporal_bin_window removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-3 temporal binning deleted"
    ),
    "savgol_detrend_enabled": (
        "savgol_detrend_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-2 Savitzky-Golay detrend deleted"
    ),
    "savgol_window_frac": (
        "savgol_window_frac removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-2 Savitzky-Golay detrend deleted"
    ),
    "savgol_polyorder": (
        "savgol_polyorder removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-2 Savitzky-Golay detrend deleted"
    ),
    "democratic_detrend_enabled": (
        "democratic_detrend_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-4 Democratic Detrender deleted"
    ),
    "democratic_sg_window_frac": (
        "democratic_sg_window_frac removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-4 Democratic Detrender deleted"
    ),
    "pytics_enabled": (
        "pytics_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-5 PyTICS weights deleted (ensemble uses comp_rms_map weights directly)"
    ),
    "pytics_n_iter": (
        "pytics_n_iter removed 2026-10-10 SPECIAL-FNS-REMOVE; ALG-5 PyTICS weights deleted (ensemble uses comp_rms_map weights directly)"
    ),
    "sysrem_enabled": (
        "sysrem_enabled removed 2026-10-10 SPECIAL-FNS-REMOVE; SysRem field detrend deleted"
    ),
    "sysrem_n_iter": (
        "sysrem_n_iter removed 2026-10-10 SPECIAL-FNS-REMOVE; SysRem field detrend deleted"
    ),
    "phase2a_airmass_before_outlier": (
        "phase2a_airmass_before_outlier removed 2026-10-10 SPECIAL-FNS-REMOVE; dead switch (no target airmass detrend exists)"
    ),
    "aperture_variable_factor": (
        "aperture_variable_factor removed 2026-10-10 SPECIAL-FNS-REMOVE; role-aware aperture scale was never applied"
    ),
    "aperture_comp_factor": (
        "aperture_comp_factor removed 2026-10-10 SPECIAL-FNS-REMOVE; role-aware aperture scale was never applied"
    ),
    "master_combine_method": (
        "master_combine_method removed 2026-10 CFG-CLEAN; never consumed (the Calibration Library "
        "build dialog chooses median/mean per build)"
    ),
    "export_arcsec_per_px": (
        "export_arcsec_per_px removed 2026-10 CFG-CLEAN; export plate scale is derived from "
        "pipeline_meta.json / MASTERSTAR WCS"
    ),
    "aperture_snr_sizing": (
        "aperture_snr_sizing removed 2026-10 CFG-CLEAN; SNR sizing sweep no longer exists, never consumed"
    ),
    "aperture_fwhm_factor_small": (
        "aperture_fwhm_factor_small removed 2026-10 CFG-CLEAN; SNR sizing sweep no longer exists, never consumed"
    ),
    "aperture_fwhm_factor_large": (
        "aperture_fwhm_factor_large removed 2026-10 CFG-CLEAN; SNR sizing sweep no longer exists, never consumed"
    ),
    "phase01_tiers": (
        "phase01_tiers removed 2026-10 CFG-CLEAN; per-tier |dmag| bounds were never applied in comp selection"
    ),
    "phase01_tier1_mag": (
        "phase01_tier1_mag removed 2026-10 CFG-CLEAN; per-tier |dmag| bounds were never applied in comp selection"
    ),
    "phase01_tier2_mag": (
        "phase01_tier2_mag removed 2026-10 CFG-CLEAN; per-tier |dmag| bounds were never applied in comp selection"
    ),
    "phase01_tier3_mag": (
        "phase01_tier3_mag removed 2026-10 CFG-CLEAN; per-tier |dmag| bounds were never applied in comp selection"
    ),
    "phase01_tier4_mag": (
        "phase01_tier4_mag removed 2026-10 CFG-CLEAN; per-tier |dmag| bounds were never applied in comp selection"
    ),
    "sips_dao_threshold_sigma": (
        "sips_dao_threshold_sigma removed 2026-10 CFG-CLEAN; replaced by detection_threshold_sigma"
    ),
    "masterstar_dao_threshold_sigma": (
        "masterstar_dao_threshold_sigma removed 2026-10 CFG-CLEAN; replaced by detection_threshold_sigma"
    ),
    "dao_detection_n_equiv": (
        "dao_detection_n_equiv removed 2026-10 CFG-CLEAN; replaced by detection_threshold_sigma"
    ),
    "alignment_detection_sigma": (
        "alignment_detection_sigma removed 2026-10 CFG-CLEAN; replaced by detection_threshold_sigma"
    ),
    "qc_dao_detection_sigma": (
        "qc_dao_detection_sigma removed 2026-10 CFG-CLEAN; replaced by detection_threshold_sigma"
    ),
    "saturate_limit_fraction": (
        "saturate_limit_fraction removed 2026-10-10; replaced by LINEARITY_FRAC = 0.90 x saturation level (SAT-LIN-2026-10)"
    ),
    "admission_sat_peak_frac": (
        "admission_sat_peak_frac removed 2026-10-10; replaced by LINEARITY_FRAC = 0.90 x saturation level (SAT-LIN-2026-10)"
    ),
    "cog_sat_frac": (
        "cog_sat_frac removed 2026-10-10; replaced by LINEARITY_FRAC = 0.90 x saturation level (SAT-LIN-2026-10)"
    ),
    "nonlinearity_peak_percentile": (
        "nonlinearity_peak_percentile removed 2026-10-10; replaced by LINEARITY_FRAC = 0.90 x saturation level (SAT-LIN-2026-10)"
    ),
    "nonlinearity_fwhm_ratio": (
        "nonlinearity_fwhm_ratio removed 2026-10-10; replaced by LINEARITY_FRAC = 0.90 x saturation level (SAT-LIN-2026-10)"
    ),
    "per_frame_saturation_enabled": (
        "per_frame_saturation_enabled removed 2026-10-10; per-frame saturation is now always on (SAT-LIN-2026-10)"
    ),
    "comp_sparse_fallback_enabled": (
        "comp_sparse_fallback_enabled removed 2026-10-10; sparse comp fallback removed, comparison stars always pass the same rules (COMP-RULES-2026-10)"
    ),
    "comp_sparse_fallback_min": (
        "comp_sparse_fallback_min removed 2026-10-10; sparse comp fallback removed, comparison stars always pass the same rules (COMP-RULES-2026-10)"
    ),
    "comp_weight_c_col_mag_per_bprp": (
        "comp_weight_c_col_mag_per_bprp removed 2026-10-10; comparison weights are Broeg (2005) weights computed in photometry_lightcurve.ensemble_normalize"
    ),
    "comp_weight_c_dist_mag_per_deg": (
        "comp_weight_c_dist_mag_per_deg removed 2026-10-10; comparison weights are Broeg (2005) weights computed in photometry_lightcurve.ensemble_normalize"
    ),
    "comp_weight_airmass_span": (
        "comp_weight_airmass_span removed 2026-10-10; comparison weights are Broeg (2005) weights computed in photometry_lightcurve.ensemble_normalize"
    ),
    "comp_weight_optics_kind": (
        "comp_weight_optics_kind removed 2026-10-10; comparison weights are Broeg (2005) weights computed in photometry_lightcurve.ensemble_normalize"
    ),
    "phase01_comparison_max_mag_diff": (
        "phase01_comparison_max_mag_diff removed 2026-10-10 (CLEAN-STEP5-2026-10); no magnitude window in comparison selection (|dmag| never gated; Broeg weights)"
    ),
    "phase01_comparison_max_mag_diff_bright_floor": (
        "phase01_comparison_max_mag_diff_bright_floor removed 2026-10-10 (CLEAN-STEP5-2026-10); no magnitude window in comparison selection (|dmag| never gated; Broeg weights)"
    ),
    "phase01_comparison_mag_bright_threshold": (
        "phase01_comparison_mag_bright_threshold removed 2026-10-10 (CLEAN-STEP5-2026-10); no magnitude window in comparison selection (|dmag| never gated; Broeg weights)"
    ),
    "phase01_comparison_max_mag_diff_absolute": (
        "phase01_comparison_max_mag_diff_absolute removed 2026-10-10 (CLEAN-STEP5-2026-10); no magnitude window in comparison selection (|dmag| never gated; Broeg weights)"
    ),
    "phase01_comparison_max_dist_deg": (
        "phase01_comparison_max_dist_deg removed 2026-10-10 (CLEAN-STEP5-2026-10); no distance limit in comparison selection (distance is only a tie-break after RMS and colour)"
    ),
    "phase01_comparison_max_dist_fov_frac": (
        "phase01_comparison_max_dist_fov_frac removed 2026-10-10 (CLEAN-STEP5-2026-10); no distance limit in comparison selection (distance is only a tie-break after RMS and colour)"
    ),
    "phase01_comparison_fov_fraction": (
        "phase01_comparison_fov_fraction removed 2026-10-10 (CLEAN-STEP5-2026-10); no distance limit in comparison selection (distance is only a tie-break after RMS and colour)"
    ),
    "phase01_comparison_max_psf_chi2": (
        "phase01_comparison_max_psf_chi2 removed 2026-10-10 (CLEAN-STEP5-2026-10); comparison-star selection never applied a PSF chi2 cut (proc CSVs are DAO-era)"
    ),
    "field_density_adaptive_enabled": (
        "field_density_adaptive_enabled removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "field_density_sparse_threshold": (
        "field_density_sparse_threshold removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "field_density_dense_threshold": (
        "field_density_dense_threshold removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "crowding_classifier_enabled": (
        "crowding_classifier_enabled removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "crowding_blend_tighten_threshold": (
        "crowding_blend_tighten_threshold removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "crowding_comp_availability_loosen_count": (
        "crowding_comp_availability_loosen_count removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
    "crowding_tighten_min_fwhm_px": (
        "crowding_tighten_min_fwhm_px removed 2026-10-10 (CLEAN-STEP5-2026-10); field density / crowding classifiers no longer change comparison rules (COMP-RULES-2026-10); field_density.json stays a diagnostic"
    ),
}
