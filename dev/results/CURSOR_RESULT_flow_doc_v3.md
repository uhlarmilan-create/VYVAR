CURSOR RESULT -- 2026-07-19

What I did
Replaced `dev/tools/docs_pdf/build_flow_doc.py` with the provided v3.0 builder
(verbatim copy), regenerated `docs/VYVAR_FLOW_CZ.pdf`, verified function names
and config defaults against the live tree, ran docs guard / ruff / session
baseline --fast, committed (not pushed). Also archived the companion
`FLOW_DOC_V3_GAPS.md` under `dev/results/` for follow-up tracking.

RESULT path note: task asked for `docs/dev_results/RESULT_flow_doc_v3.md`;
REPO-REORG puts Cursor results in `dev/results/CURSOR_RESULT_*.md` -- used that.

## Output / findings

PDF: `docs/VYVAR_FLOW_CZ.pdf` -- 36 pages A4, 130842 bytes. Builder printed `ok`.
ASCII guard: PASS (`encoding='ascii'`).

Function name greps (all PASS on live tree):
- photometry_core.py: run_full_photometry_pipeline (14805),
  measure_empty_aperture_sigma_bkg (813), compute_snr_optimal_aperture_table (1240),
  compute_aperture_correction (3000), ensemble_normalize (3180),
  compute_mag_calib_final (4644), build_global_comp_pool (12747)
- k2_extinction.py: apply_k2_to_comp_mag_inst (242)
- band_classify.py: classify_photometric_band (281)
- sigma_floor_core.py: combine_production_err_rel (64)
- psf_photometry.py: build_epsf_model (1349), build_epsf_grid_model (1456),
  _psf_sandwich_flux_err (2217)
- crowding_index.py: compute_crowding_index (303)
- tess_verify.py: _period_consensus (521)
- check_star_kmag.py: select_check_star (241)
- trust_flag_core.py: CompTrustThresholds (48)
- calibration.py: resample_master_to_light_binning (276)

Config defaults (all PASS; key names where shorthand differs noted):
- comp_max_delta_bprp: 0.79
- n_comp min/max: 3 / 8 via phase01_comparison_n_comp_min/max
- comp_trust_min_comps: 3
- annulus: 4.75 / 9.0 (annulus_inner_fwhm / annulus_outer_fwhm)
- aperture_fwhm_factor: 1.9
- aperture_snr_sizing: {small:1.5, large:4.0}
- err_empty_apertures_n/min: 64 / 16
- sigma_sys_mag: {"4": 0.018}
- k2_mode: "literature"; k2_ceiling: 0.1; k2_fit_enabled: false
- temporal_binning_enabled: false
- pytics_enabled: true; pytics_n_iter: 5
- psf_photometry_enabled: false; epsf_min_stars: 30
- cal_diag_rel_tol: 0.02; cal_diag_hard_sigma: 5.0
- field_density sparse/dense: 300 / 1000
- comp_color_tiers: [0.15/1.0, 0.30/0.85, 0.55/0.50, 1.10/0.25]
- sparse_trust_T_green/T_red/X2: 1.5 / 4.0 / 0.0004
- vsx_variable_targets_mag_limit: 14.5; verify_mag_limit: 14.0

Mismatch list: none (live tree matches doc claims; n_comp keys use the
phase01_comparison_n_comp_* names, values 3/8 as stated).

Tests:
- pytest dev/tests/test_docs_layout.py: 5 passed
- ruff check dev/tools/docs_pdf/build_flow_doc.py: All checks passed
- session_baseline_check.py --fast (path remapped from task's
  `dev/tools/` to live `dev/scripts/`): OVERALL PASS
  (973 passed, 19 skipped; WARN only on known untracked / ledger-todo / deps)

Companion: `dev/results/FLOW_DOC_V3_GAPS.md` (program / real-data / synthetic
follow-ups surfaced while writing the doc; not implementation work in this
commit).

## Errors (if any)
None.

## Files changed
- dev/tools/docs_pdf/build_flow_doc.py (replaced, v3.0)
- docs/VYVAR_FLOW_CZ.pdf (regenerated, 36 pp)
- dev/results/FLOW_DOC_V3_GAPS.md (archived from task bundle)
- dev/results/CURSOR_RESULT_flow_doc_v3.md (this file)
- Commit: local only, not pushed (message: docs(flow): FLOW doc v3.0...)
