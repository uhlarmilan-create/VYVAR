CURSOR RESULT - 2026-08-05 (Scope audit remediation, Task C')

What I did
Fixed Task C scope classification defects: dead EXPLICIT key guard, reclassifications per
C'-1/C'-3, meaningful scope_confidence, new scope_key and scope_group axes, rig triage into
groups (a/b/c), sigma_sys_mag help correction, ROADMAP D1/D2 entries, extended parity tests.

## 1. Premise check

Task C baseline at `2b78386`: 277 entries; scope universal 229 / rig 31 / site 11 / session 6;
scope_confidence low 0/277. Verified against ground-truth anchors, help text, and project record.

Measured after remediation (HEAD pending commit):

**Errata (Task C''):** section 1 below reported `rig_sampling 19`; the registry held **18**
(15+2+18=35 rig entries). Corrected in `CURSOR_RESULT_params_scope_corrections.md`.

| metric | Task C | Task C' | delta |
|--------|--------|---------|-------|
| universal | 229 | 226 | -3 |
| rig | 31 | 35 | +4 |
| site | 11 | 10 | -1 |
| session | 6 | 6 | 0 |
| low confidence | 0 | 25 | +25 (expected) |

scope_key: none 226, rig 15, rig_band 2, rig_sampling 19, site 10, frame 6.
Rig triage: group a 20, group b 12, group c 3.

## 2. Corrected scope distribution and diffs

| key | Task C | Task C' | reason |
|-----|--------|---------|--------|
| k2_defaults_bprp | universal/high | rig/rig_band/low | Real k'' key; filter x detector QE + site; flat dict leak (C'-2) |
| apply_color_term | site/high | rig/rig_band/high | Color term is filter x detector, not location (C'-3) |
| phase01_comparison_max_dist_deg | universal/high | rig/rig_sampling/b/low | FOV-derived base; JOURNAL plate-scale leak incident (C'-3) |
| phase01_comparison_min_dist_arcsec | universal/high | rig/rig_sampling/b/low | Twin of isolation_radius_px; resolve to arcsec via plate scale (C'-3) |
| gs11_dilution_aperture_arcsec | rig/high | universal/low | Already arcsec on sky; 0 derives from photometric aperture |
| saturate_limit_fraction | (universal via _frac bug) | rig/a/high | Restored: detector saturation fraction |
| (dead keys) k2_coeff_v/r/i | in EXPLICIT, silently dropped | removed | C'-1: classifier now raises |

## 3. Rig triage table (35 keys)

### Group (a) -- genuine per-rig physics (20)

| key | scope_key | conf | reason |
|-----|-----------|------|--------|
| gain | rig | high | Equipment-intrinsic e-/ADU |
| read_noise | rig | high | Equipment-intrinsic e- |
| admission_sat_peak_frac | rig | high | Saturation fraction vs full-well ADU |
| saturate_limit_fraction | rig | high | Saturation fraction vs detector full-well |
| err_background_mode | rig | high | F-BINGAIN-1 Newton/bin4 vs wide |
| sigma_sys_mag | rig | high | PROD-SIGMA-FLOOR; equipment_id keyed |
| bpm_dark_mad_sigma | rig | high | Dark noise structure per detector |
| calibration_master_ccd_temp_tolerance_c | rig | high | CCD temp tolerance for master match |
| calibration_library_native_binning | rig_sampling | high | Native binning matches read mode |
| osc_channel_binning | rig_sampling | high | OSC: 2 x osc_channel_binning Bayer superpixel |
| blind_use_rig_prior | rig | high | Rig FOV prior for blind index |
| plate_solve_fov_deg | rig | high | Telescope+camera FOV seed |
| frame_height_px | rig | high | Camera ROI/binning |
| frame_width_px | rig | high | Camera ROI/binning |
| cal_diag_sat_warn_frac | rig | high | Saturation warning vs full-well |
| apply_color_term | rig_band | high | Filter x detector color transform |
| k2_defaults_bprp | rig_band | low | k'' per band; rig + site components |
| masterstar_dao_threshold_sigma | rig_sampling | low | Optimal sigma depends on sampling/depth/noise |
| sips_dao_threshold_sigma | rig_sampling | low | Same as masterstar_dao_threshold_sigma |
| qc_dao_detection_sigma | rig_sampling | low | Same as masterstar_dao_threshold_sigma |

### Group (b) -- unit artefacts; target normalisation (12)

| key | scope_key | target unit | reason |
|-----|-----------|-------------|--------|
| blind_verify_match_tol_px | rig_sampling | arcsec | Convert via resolved plate scale |
| cog_ladder_step_px | rig_sampling | FWHM multiples | Aperture ladder step |
| crowding_tighten_min_fwhm_px | rig_sampling | FWHM px floor | Undersampling gate |
| hrd_color_bg_box_px | rig_sampling | arcsec or field frac | HR diagram crop box |
| masterstar_centre_rms_max_px | rig_sampling | arcsec | Centroid RMS gate |
| masterstar_sibling_rms_max_px | rig_sampling | arcsec | Sibling recovery RMS |
| phase01_chip_interior_margin_px | rig_sampling | FWHM/aperture-driven | PHASE0-BORDER-MARGIN-GEOMETRY |
| phase01_comparison_isolation_radius_px | rig_sampling | arcsec | Twin of min_dist_arcsec |
| phase01_comparison_max_dist_deg | rig_sampling | FOV-derived deg | Derive base from resolved FOV |
| phase01_comparison_min_dist_arcsec | rig_sampling | arcsec (already) | FOV/density-adapted twin |
| qc_max_hfr | rig_sampling | FWHM-normalised ratio | Sharpness gate |
| sips_dao_fwhm_px | rig_sampling | FWHM multiples | Initial DAO FWHM guess |

### Group (c) -- operational tuning (3)

| key | scope_key | reason |
|-----|-----------|--------|
| alignment_max_control_points | rig_sampling | chi/h Persei performance tuning |
| masterdark_validity_days | rig | Staleness warning; storage habit |
| masterflat_validity_days | rig | Staleness warning; optics handling |

## 4. DAO threshold verdict

All three (`masterstar_dao_threshold_sigma`, `sips_dao_threshold_sigma`, `qc_dao_detection_sigma`)
land in **group (a)**, not (b) or (c).

Argument: the unit is already dimensionless (sigma above background), so group (b) normalisation
does not apply. Group (c) is ruled out because a wrong threshold changes which sources enter the
catalog -- science correctness, not just performance. The optimal numeric value plausibly depends on
sampling, PSF pixel count, and depth relative to the Gaia cut, so per-rig storage may be needed.
draft_501's DAO_ONLY fraction difference (0.417 vs 0.037) shows the *outcome* varies by rig/cal
mode but is not proof the *threshold* must differ; classified group (a) with **low** confidence.

## 5. Low-confidence list (25) -- questions for Milan

1. **k2_defaults_bprp** -- accept rig_band with site component deferred to D2?
2. **k2_defaults_bprp flat dict** -- how should per-rig k'' be entered until D2 schema exists?
3. **phase01_comparison_max_dist_deg / min_dist_arcsec / isolation_radius_px** -- converge all three
   to arcsec via plate scale (D1), or keep px twin?
4. **gs11_dilution_aperture_arcsec** -- universal angular aperture, or rig-scoped after all?
5. **masterstar/sips/qc DAO thresholds** -- group (a) storage vs recalibrate once per rig empirically?
6. **crowding_tighten_min_fwhm_px** -- stay px-native undersampling gate or FWHM-normalise?
7. **masterdark/flat_validity_days** -- group (c) global default sufficient?
8. **blind_index/img_select_mode, phase01_flux_col** -- session/frame scope correct?
9. **blind_verify_match_tol_px, cog_ladder_step_px, hrd_color_bg_box_px, qc_max_hfr** -- D1 priority order?
10. **lacosmic_sigclip / lacosmic_objlim / qc_elong_limit** -- mechanical low; confirm universal?

(Full key list: blind_img_select_mode, blind_index_select_mode, blind_verify_match_tol_px,
cog_ladder_step_px, crowding_tighten_min_fwhm_px, gs11_dilution_aperture_arcsec, hrd_color_bg_box_px,
k2_defaults_bprp, lacosmic_objlim, lacosmic_sigclip, masterdark_validity_days, masterflat_validity_days,
masterstar_centre_rms_max_px, masterstar_dao_threshold_sigma, masterstar_sibling_rms_max_px,
phase01_chip_interior_margin_px, phase01_comparison_isolation_radius_px,
phase01_comparison_max_dist_deg, phase01_comparison_min_dist_arcsec, phase01_flux_col,
qc_dao_detection_sigma, qc_elong_limit, qc_max_hfr, sips_dao_fwhm_px, sips_dao_threshold_sigma)

## 6. Findings recorded, not fixed

- **k2_defaults_bprp flat-dict leak:** `{band: value}` keyed by filter token only; one rig's k''
  applies silently to all rigs and imported third-party data. D2 scope.
- **sigma_sys_mag equipment-only key:** `resolve_sigma_sys_mag` keys on `equipment_id` only; same
  camera on two OTAs has two plate scales and two floors. Does not bite today (different cameras).
- **sigma_sys_mag help was wrong:** registry said band keys; code uses equipment_id strings. Fixed.
- **JOHNSON_K2_ZERO_TOKENS / K2Source:** V/R/I return 0.0 from `computed_k2_bprp_for_token`; measured
  per-rig k'' for those bands needs a K2Source token so literature default is not mislabeled. D2.
- **Dead EXPLICIT keys k2_coeff_v/r/i:** never existed in registry; validation now fails loudly.

## 7. D1 and D2 sizing

**D1 (group b normalisation):** 12 rig keys re-expressed as arcsec or FWHM multiples via resolved
plate scale at runtime. Science-path change; needs byte-identity gate. ROADMAP item added.

**D2 (group a per-rig storage):** 20 keys need equipment/profile lookup (`ID_EQUIPMENTS`,
`ID_TELESCOPE`, rig_band, rig_sampling). Pattern: `resolve_sigma_sys_mag`. Must widen
`sigma_sys_mag` key, add `K2Source` for measured k'', replace flat `k2_defaults_bprp` shape.
DAO thresholds may join group (a) if empirical per-rig calibration confirms.

## Runtime behaviour

Confirmed by inspection: no `src_py` module reads `scope`, `scope_key`, or `scope_group` at run
time (only `params_registry.py` defines the enums). Metadata-only change.

## Tests

- `dev/tests/test_params_registry.py`: 16 passed (added scope_key/group guards, invariant test,
  EXPLICIT key existence test, dead-key injection test).
- Full suite: **1240 passed, 26 skipped** (Task C baseline 1236 passed; +4 new registry guards).

## Files changed

- dev/tools/classify_params_scope.py
- dev/tools/emit_scope_report_data.py
- dev/tools/gen_params_md.py
- dev/validation/params_registry.json
- src_py/params_registry.py
- dev/tests/test_params_registry.py
- docs/VYVAR_PARAMS.md
- docs/VYVAR_ROADMAP.md
- dev/results/CURSOR_RESULT_params_scope_remediation.md

Czech parameter handbook PDF not regenerated (separate builder cycle).

## Errors

None.
