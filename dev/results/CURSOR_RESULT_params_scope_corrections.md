CURSOR RESULT - 2026-08-05 (Scope triage corrections, Task C'')

What I did
Corrected four triage-layer defects from Task C' before D1/D2: frame dimensions to session,
FOV/sampling key alignment, sampling-defining acyclicity guard, min_dist/isolation direction
for D1, and max_dist target unit. Regenerated registry and docs.

## 1. Premise check (measured)

Base `0613b33`. Post-conditions vs C':

| field | C' | C'' | match |
|-------|-----|-----|-------|
| universal | 226 | 227 | yes |
| rig | 35 | 32 | yes |
| site | 10 | 10 | yes |
| session | 6 | 8 | yes |
| scope_key rig / rig_band / rig_sampling | 15 / 2 / 18 | 16 / 2 / 14 | yes |
| group a / b / c | 20 / 12 / 3 | 18 / 11 / 3 | yes |

Totals: scopes 277; rig keys 16+2+14=32; groups 18+11+3=32. Low confidence 24 (was 25).

## 2. Per-key changes (code evidence verified)

| key | C' | C'' | evidence |
|-----|-----|-----|----------|
| frame_height_px | rig/a/rig/high | session/frame/n/a/high | Registry help: FITS NAXIS2 at run time; WAVE-B removed from config.json; owner=internal, kind=resolved |
| frame_width_px | rig/a/rig/high | session/frame/n/a/high | Same for NAXIS1 |
| phase01_comparison_max_dist_deg | rig/b/rig_sampling/low | rig/b/rig/low | config.py:2764 runtime additive to FOV result; :2796 density delta additive; :2828 apply_density_overrides skips cfg write, adds to effective max_dist_deg. FOV binning-invariant |
| phase01_comparison_min_dist_arcsec | rig/b/rig_sampling/low | universal/none/n/a/high | photometry_core.py:14798,15917 reads float directly in arcsec; default 60.0 arcsec |
| osc_channel_binning | rig/a/rig_sampling/high | rig/a/rig/high | Help: total linear scale = 2 x N; defines rig_sampling input |
| calibration_library_native_binning | rig/a/rig_sampling/high | rig/a/rig/high | Native binning defines sampling context, not resolved by it |

## 3. Group (a) -- genuine per-rig physics (18 keys)

| key | scope_key | conf |
|-----|-----------|------|
| gain | rig | high |
| read_noise | rig | high |
| admission_sat_peak_frac | rig | high |
| saturate_limit_fraction | rig | high |
| err_background_mode | rig | high |
| sigma_sys_mag | rig | high |
| bpm_dark_mad_sigma | rig | high |
| calibration_master_ccd_temp_tolerance_c | rig | high |
| calibration_library_native_binning | rig | high |
| osc_channel_binning | rig | high |
| blind_use_rig_prior | rig | high |
| plate_solve_fov_deg | rig | high |
| cal_diag_sat_warn_frac | rig | high |
| apply_color_term | rig_band | high |
| k2_defaults_bprp | rig_band | low |
| masterstar_dao_threshold_sigma | rig_sampling | low |
| sips_dao_threshold_sigma | rig_sampling | low |
| qc_dao_detection_sigma | rig_sampling | low |

## 4. Group (b) -- unit artefacts (11 keys)

| key | scope_key | target unit (D1) |
|-----|-----------|------------------|
| blind_verify_match_tol_px | rig_sampling | arcsec (via resolved plate scale) |
| cog_ladder_step_px | rig_sampling | FWHM multiples |
| crowding_tighten_min_fwhm_px | rig_sampling | FWHM px floor or FWHM-normalised undersampling gate |
| hrd_color_bg_box_px | rig_sampling | arcsec or field fraction |
| masterstar_centre_rms_max_px | rig_sampling | arcsec |
| masterstar_sibling_rms_max_px | rig_sampling | arcsec |
| phase01_chip_interior_margin_px | rig_sampling | FWHM/aperture-driven margin (PHASE0-BORDER-MARGIN-GEOMETRY) |
| phase01_comparison_isolation_radius_px | rig_sampling | arcsec |
| phase01_comparison_max_dist_deg | rig | fraction of resolved FOV (additive margin on FOV-derived base) |
| qc_max_hfr | rig_sampling | FWHM-normalised ratio |
| sips_dao_fwhm_px | rig_sampling | FWHM multiples |

All target units are rig-independent once normalised. No group (b) key resists such a unit.

## 5. Group (c) -- operational tuning (3 keys)

| key | scope_key | reason |
|-----|-----------|--------|
| alignment_max_control_points | rig_sampling | chi/h Persei performance; not science correctness |
| masterdark_validity_days | rig | staleness warning only |
| masterflat_validity_days | rig | staleness warning only |

## 6. Isolation-radius evidence (D1 argument)

- `phase01_comparison_isolation_radius_px` default **25.0 px** (config.py:993), consumed at
  photometry_core.py:15928 in pixels.
- `phase01_comparison_min_dist_arcsec` default **60.0 arcsec** (config.py:847), consumed at
  photometry_core.py:14798 in arcsec -- now **universal** (correct direction for D1).
- JOURNAL.md:3362-3363 -- Newton 300/1200 + C3-26000 ~**0.65 arcsec/px bin1**, ~**1.30 bin2**.
- Wide QHY294MM field resolves ~**9.77 arcsec/px** (JOURNAL.md:3233-3234, 3358).
- On Newton bin2: 25 px x 1.30 arcsec/px ~ **32.5 arcsec** (~half the 60 arcsec min_dist), silent.
- On wide 9.77"/px: 25 px ~ **244 arcsec** (isolation >> min_dist; px-native isolation is wrong direction on coarse scale too).
- Chi/h Persei dense fields use ~**1.302 arcsec/px** (JOURNAL.md:1908) -- same bin2-class sampling as Newton.

D1 must normalise `isolation_radius_px` to arcsec; leave `min_dist_arcsec` alone.

## 7. max_dist_deg additive-delta confirmation

config.py:2764 comment: max_dist_deg added to FOV result at runtime. DENSITY_OVERRIDES :2796
`+0.3` additive to FOV result. apply_density_overrides :2828-2829: for this param, delta is
**not** written to cfg; caller adds to effective FOV-derived max_dist_deg. D1 target unit:
**fraction of resolved FOV**, not a fixed degree margin.

## 8. Remaining triage concerns (last pass before D1/D2)

- DAO thresholds remain group (a) with low confidence; dimensionless but optimal value may differ by rig.
- `k2_defaults_bprp` flat dict leak unchanged (D2).
- `sigma_sys_mag` equipment_id-only key unchanged (D2).
- `gs11_dilution_aperture_arcsec` stays universal/low -- confirm with Milan.

## Runtime behaviour

No src_py module reads scope, scope_key, scope_group, or SAMPLING_DEFINING_KEYS at run time
(inspection grep; metadata only).

## Tests

- test_params_registry.py: 19 passed (+2 acyclicity guards, +1 internal/resolved/group-a guard).
- Full suite: **1243 passed, 26 skipped** (C' baseline 1240 passed; +3 new guards).

## Files changed

- dev/tools/classify_params_scope.py
- dev/validation/params_registry.json
- src_py/params_registry.py (SAMPLING_DEFINING_KEYS)
- dev/tests/test_params_registry.py
- docs/VYVAR_PARAMS.md
- dev/results/CURSOR_RESULT_params_scope_remediation.md (errata line)
- dev/results/CURSOR_RESULT_params_scope_corrections.md

## Errors

None.
