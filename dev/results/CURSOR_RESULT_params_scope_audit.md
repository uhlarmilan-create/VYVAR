CURSOR RESULT - 2026-08-05 (Parameter scope audit)

What I did
Added the `scope` editorial axis (universal/rig/site/session) plus `scope_confidence` to all
277 registry entries, wired parity guards and doc generator, classified every parameter on
dimensional/physical dependence, and regenerated `docs/VYVAR_PARAMS.md`.

## 1. Premise check

Compared against `docs/VYVAR_PARAMS.md` baseline (`_Generated 2026-08-04T12:12:38Z at git HEAD ab0f669_`,
277 entries). Task A landed first on this branch (`554f7a8` runtime, `cbfe639` docs); audit ran at
HEAD after those commits. Registry count confirmed **277** entries; generator HEAD at write time:
**pending commit** (post Task A + scope axis).

Existing axes (`tier`, `phase`, `kind`, `owner`, `widget`) describe storage and UX grouping only;
they do not answer whether the correct value depends on equipment. Built on `param_resolver.py`
taxonomy (equipment-intrinsic / observation-specific / site) without duplicating runtime lookup.

## 2. Scope distribution

**Overall:** universal 229, rig 31, site 11, session 6 (277 total). Low-confidence: 0.

| phase | universal | rig | site | session |
|-------|-----------|-----|------|---------|
| observer | 0 | 0 | 9 | 0 |
| paths | 10 | 0 | 0 | 1 |
| calibration | 5 | 6 | 0 | 0 |
| qc | 21 | 3 | 0 | 0 |
| alignment | 2 | 1 | 0 | 0 |
| detection | 57 | 12 | 0 | 3 |
| photometry | 67 | 6 | 0 | 0 |
| comp_selection | 31 | 2 | 0 | 1 |
| trust | 11 | 0 | 0 | 0 |
| extinction | 6 | 0 | 2 | 0 |
| reports | 16 | 1 | 0 | 0 |
| export | 1 | 0 | 0 | 1 |
| system | 2 | 0 | 0 | 0 |

Shortcuts applied: `owner=internal` (13) and `phase=paths` plumbing -> universal; `owner=db_static`
(9 observer) -> site; `owner=fits_dynamic` (6) verified against resolver (see disagreements).

## 3. Full `rig` list (31 keys)

| key | conf | justification |
|-----|------|---------------|
| admission_sat_peak_frac | high | Saturation fraction gate relative to full-well ADU per detector. |
| alignment_max_control_points | high | Control-point cap depends on plate scale and field star density. |
| blind_use_rig_prior | high | Blind index selection uses rig-specific FOV prior. |
| blind_verify_match_tol_px | high | Pixel tolerance for blind verify star matching. |
| bpm_dark_mad_sigma | high | Hot-pixel MAD sigma depends on dark noise structure per detector. |
| cal_diag_sat_warn_frac | high | Saturation warning fraction relative to detector full-well. |
| calibration_library_native_binning | high | Native binning of calibration library matches detector setup. |
| calibration_master_ccd_temp_tolerance_c | high | CCD temperature tolerance for master matching depends on camera. |
| cog_ladder_step_px | high | COG aperture ladder step in pixels depends on sampling. |
| crowding_tighten_min_fwhm_px | high | Minimum FWHM (px) below which crowding tightening is skipped. |
| err_background_mode | high | Empirical vs Howell term differed on Newton/bin4 vs wide rig (F-BINGAIN-1). |
| frame_height_px | high | Detector frame height in pixels (camera ROI/binning). |
| frame_width_px | high | Detector frame width in pixels (camera ROI/binning). |
| gain | high | Equipment-intrinsic e-/ADU. |
| gs11_dilution_aperture_arcsec | high | Physical dilution aperture diameter scales with plate-scale intent. |
| hrd_color_bg_box_px | high | HR color-field background box size in pixels. |
| masterdark_validity_days | high | Master dark shelf life depends on detector stability and storage. |
| masterflat_validity_days | high | Master flat shelf life depends on detector/optics handling. |
| masterstar_centre_rms_max_px | high | Masterstar centroid RMS gate in pixels. |
| masterstar_dao_threshold_sigma | high | Optimal DAO sigma varies with noise/calibration/rig (draft_501 anchor). |
| masterstar_sibling_rms_max_px | high | Sibling-recovery RMS gate in pixels. |
| osc_channel_binning | high | OSC channel binning matches camera read mode. |
| phase01_chip_interior_margin_px | high | Chip interior margin in pixels. |
| phase01_comparison_isolation_radius_px | high | Comp isolation radius in pixels. |
| plate_solve_fov_deg | high | Blind-solve FOV seed depends on telescope+camera field size. |
| qc_dao_detection_sigma | high | QC DAO sigma threshold coupled to rig noise and depth. |
| qc_max_hfr | high | Maximum HFR per frame in pixels. |
| read_noise | high | Equipment-intrinsic read noise in e-. |
| sigma_sys_mag | high | Systematic mag floor is per-rig (PROD-SIGMA-FLOOR). |
| sips_dao_fwhm_px | high | Initial DAO FWHM guess in pixels depends on seeing sampling. |
| sips_dao_threshold_sigma | high | Plate-solve DAO threshold varies with field depth and rig sampling. |

## 4. Disagreements with `param_resolver.py`

| key | registry scope | resolver category | note |
|-----|----------------|-------------------|------|
| gain | rig | equipment-intrinsic | Aligned on physics; `owner=fits_dynamic` shortcut would say session -- registry uses rig. |
| read_noise | rig | equipment-intrinsic | Same as gain. |
| plate_scale_arcsec_per_px | session | observation-specific | Aligned. |
| phase01_plate_scale_arcsec_per_px | session | observation-specific | Aligned. |
| export_arcsec_per_px | session | observation-specific | Aligned. |
| plate_solve_fov_deg | rig | observation-specific (FOV) | Classified rig because FOV seed is telescope+camera property, not per-exposure pointing. |

No other `kind=resolved` keys beyond the 18 already covered.

## 5. Low-confidence list (Milan review)

None at commit time (0/277). All entries classified `high` confidence after explicit review of
borderline keys (booleans mis-tagged via `binning`/`saturation` substrings were corrected).

## 6. Priority ordering for future per-rig config

Ranked by observed blast radius (evidence first):

1. **masterstar_dao_threshold_sigma** / **sips_dao_threshold_sigma** / **qc_dao_detection_sigma** --
   draft_501 DAO_ONLY 0.417 vs anchor 0.037; blocked entire LC output until INV-MS-01 removed.
2. **err_background_mode** -- Newton/bin4 chi-squared deficit required empirical empty-aperture term.
3. **sigma_sys_mag** -- wide-rig fit ~15 mmag anomalous; explicitly deferred per-rig (PROD-SIGMA-FLOOR).
4. **alignment_max_control_points** -- chi/h Persei tuning (80 after 8.6x speedup).
5. **calibration_library_native_binning**, **masterdark/flat_validity_days**, **bpm_dark_mad_sigma** --
   detector/storage coupling; theoretical rig sensitivity, less observed cross-rig pain so far.
6. **Pixel-native gates** (centre_rms_max_px, isolation_radius_px, qc_max_hfr, etc.) -- rig units but
   often scaled with FWHM elsewhere; lower observed blast radius.
7. **gain** / **read_noise** -- already resolved at runtime via `param_resolver`; config is fallback.

## 7. What per-rig config must do that global config cannot

Global `config.json` holds one value per algorithm knob, but at least 31 parameters have physically
correct values that differ between rigs (pixels, ADU, FOV, noise structure) while the science goal
is unchanged. Per-rig config would need an equipment/profile selector at session start that overlays
rig-scoped defaults onto the global base without breaking observation-specific/session-resolved keys
(gain/read_noise/plate scale still win from FITS/DB/resolver). Global config alone cannot express
"DAO threshold 3.8 on QHY wide pre-VYVAR-cal, 0.25-1.0 on Newton pre-calibrated" without silent
wrong science on the other rig.

## Errors

None.

## Files changed

- dev/validation/params_registry.json (scope + scope_confidence on 277 entries)
- src_py/params_registry.py (SCOPES, ENTRY_KEYS)
- dev/tests/test_params_registry.py (scope parity guard)
- dev/tools/gen_params_md.py (scope column + summary)
- dev/tools/classify_params_scope.py (classification tooling)
- docs/VYVAR_PARAMS.md (regenerated)
- dev/results/CURSOR_RESULT_params_scope_audit.md

Note: Czech parameter handbook PDF not regenerated (separate builder cycle); will drift until next
handbook build.

## Task B (calibration paths audit)

Read-only report at `dev/results/CURSOR_RESULT_calpath_audit.md` (completed in parallel).
