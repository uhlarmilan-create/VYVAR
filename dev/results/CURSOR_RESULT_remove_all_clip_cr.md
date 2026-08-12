CURSOR RESULT - 2026-08-12

What I did
Removed ALL L.A.Cosmic / cosmic-ray cleaning (already gone at 0ab686f) and ALL
sigma-clip / kappa-sigma / outlier-rejection used while producing science data.
Replaced astropy sigma_clipped_stats call sites with plain_mean_med_std. Deleted
dead clip/CR config keys + Settings UI. Master dark/flat stacking already used
plain nanmean/nanmedian (unchanged).

## params_registry
- Total: 272 (was 279 after lacosmic removal)
- config_runtime: 244 (was 251)
- Removed keys: comp_clip_sigma, variability_sigma_clip,
  phase01_comparison_rms_outlier_sigma, frame_quality_gate_enabled,
  frame_quality_ratio_k, frame_quality_fwhm_factor, frame_quality_min_keep_frames

## FULL list of removed / replaced clip-CR sites

### Cosmic-ray (prior commit 0ab686f; confirmed absent)
| Site | What | Replacement |
|---|---|---|
| pipeline.py `_remove_cosmics_lacosmic` | L.A.Cosmic / astroscrappy | deleted |
| pipeline.py QC enrich call site | VY_COSM write | deleted |
| AppConfig enable_lacosmic / lacosmic_* | CR knobs | deleted |
| requirements.txt astroscrappy | dep | deleted |

### Science-path clips removed / neutered this arc
| File:area | What clipped | Replaced with |
|---|---|---|
| src_py/plain_stats.py (new) | n/a | plain_mean_med_std (mean/median/std, no clip) |
| pipeline.py (~1414,4402,7091,7120,7360,7697,8451,15412,15485,16683,16744,16851,17147,17171) | sigma_clipped_stats for bg/DAO/sky/QC | plain_mean_med_std |
| pipeline.py `_fit_subtract_preprocess_sky_surface` | sigma_clip on fit samples | plain stats + star mask + calm_adu only |
| photometry_core.py (~1791) | sigma_clipped_stats | plain_mean_med_std |
| photometry_core.py `fit_color_term_c1` / CT scatter | residual sigma-clip | no residual clip (kwargs ignored) |
| photometry_core.py `check_comparison_stability` MAD p2p | MAD+sigma exclude comps | hard p2p ceiling 0.10 only (no MAD) |
| photometry_core.py `_frame_quality_gate_select` | MAD/z-score frame drop | always passthrough |
| photometry_core.py Phase2A call of frame-quality gate | frame drop | call site removed |
| vyvar_platesolver.py (~4798,6187) | sigma_clipped_stats | plain_mean_med_std |
| vyvar_alignment_frame.py (~187) | sigma_clipped_stats | plain_mean_med_std |
| psf_runner.py (~134) | sigma_clipped_stats | plain_mean_med_std |
| astrometry_optimizer.py (~1047) | sigma_clipped_stats | plain_mean_med_std |
| wide_slope_noise_core.py | sigma_clipped_stats | plain_mean_med_std |
| xval_run.py ApertureStats | SigmaClip on annulus | sigma_clip=None |
| tess_verify.py `_iterative_sigma_clip_lc` | LC point drop | passthrough |
| hrd_colorfield.py `_sigma_clipped_median_sigma` | iterative keep-clip | plain median + MAD (no iter drop) |
| variability_detector.py per-star RMS prep | per-star sigma-clip | no clip; all finite points |
| comp_frame_normalize.py `robust_comp_rms` | flux-frac + MAD clip drop | MAD on all finite positive fluxes |
| comp_selection_per_target.py `_iterative_ensemble_clip_cm_residual` | ensemble sigma drop | passthrough |
| comp_selection_per_target.py `_ensemble_mad_filter_rms` | iterative MAD RMS drop | keep all candidates |
| importer.py master dark/flat | (already plain) | nanmean / nanmedian unchanged |
| pipeline.py / app.py / night_run.py | dead temporal_sigma_clip API | removed |

### Kept (reporting / trust labels only - do not alter shipped mag arrays)
| Site | Why kept |
|---|---|
| photometry_core.py `detect_outliers` | flags outlier_hi/lo only; mag arrays unchanged |
| photometry_core.py `_write_suspected_variables` MAD threshold | candidate list for report |
| variability_detector.py candidate filters (clip_ratio etc.) | classification of variable candidates |
| TRUST / quality labels in LC UI | display only |

### Borderline for Milan (not removed; please decide)
| Site | Notes |
|---|---|
| check_comparison_stability slope filter | fixed mmag/hr + significance; not MAD sigma-clip, but can exclude comps from ensemble |
| phase01_comparison_max_comp_rms hard gate | fixed threshold (not sigma) |
| ABS_MAX_P2P = 0.10 hard ceiling | kept after removing MAD; still excludes comps |
| frame_align_residual_gate_* | residual vs aperture fraction; default OFF; not MAD/sigma |
| bpm_dark_mad_sigma | hot-pixel BPM from dark MAD; calibration map, not light clip |
| auto_fwhm_k_* / QC limits | QC pass/fail thresholds using MAD scale |
| variability_clip_ratio_min | still gates variable-candidate flag; ratio~1 now that RMS clip is gone |
| comp_rms definition | now unclipped MAD; numbers will change vs recent robust clip |

## Guards
- --fast: see session run after this result (pytest targeted: 10/10 PASS for clip-related failures)
- ASCII-only
- Dead clip/CR config + UI removed
- Fresh BO CVn re-run still required (505/506 on-disk calibrated remain CR-damaged)

## Files changed
- src_py/plain_stats.py (new)
- src_py/pipeline.py, photometry_core.py, config.py, ui_settings.py, citations.py
- src_py/comp_frame_normalize.py, comp_selection_per_target.py, variability_detector.py
- src_py/vyvar_platesolver.py, vyvar_alignment_frame.py, psf_runner.py, astrometry_optimizer.py
- src_py/wide_slope_noise_core.py, xval_run.py, tess_verify.py, hrd_colorfield.py
- src_py/app.py, night_run.py
- config.json, dev/validation/params_registry.json, docs/VYVAR_PARAMS.md
- docs/VYVAR_CONFIG_GUIDE_EN.md, docs/VYVAR_CONFIG_GUIDE_CZ.md
- dev/tests/* (frame quality, comp rms/clip, stability, dashboard counts)
- dev/tools/classify_params_scope.py, gen_params_md outputs
