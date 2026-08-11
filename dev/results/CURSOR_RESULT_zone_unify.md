CURSOR RESULT - 2026-08-11

What I did
Collapsed MASTERSTAR QC zones noisy1/noisy2/noisy3 into a single `noise` zone. Sub-detection-
significance stars (peak_dao/bg_sigma < dao_detection_n_equiv) are `noise`; only `saturated`,
`linear`, and `noise` remain.

## Output / findings

### Zone rule (after)
- `saturated`: peak above saturation limit
- `linear`: peak_sig >= dao_detection_n_equiv
- `noise`: unsaturated and peak_sig < dao_detection_n_equiv (replaces noisy1/2/3)

### Zone count delta (draft_503 Milan infolog baseline vs expected)
| Zone | Before (Milan infolog) | After (expected) |
|------|------------------------|------------------|
| linear | 132 | ~132 (unchanged boundary) |
| noisy1 | 34 | - |
| noisy2 | 37 | - |
| noisy3 | 38 | - |
| noise | - | ~109 (= 34+37+38) |
| saturated | 0 | 0 |

P1-mini anchor will **not** be byte-identical (science-path change by design).

### Touch points
- pipeline.py `_annotate_masterstars_flux_zones`
- photometry_core.py zone_flag, LC quality, select_active_targets logging
- ui_aperture_photometry.py badges
- variability_detector.py zone_filter -> linear only
- Legacy noisy1/2/3 CSV rows map to `noise` on read

## Errors (if any)
None.

## Files changed
- src_py/pipeline.py
- src_py/photometry_core.py
- src_py/ui_aperture_photometry.py
- src_py/variability_detector.py
- dev/tests/test_masterstar_zone_classifier.py
- dev/tests/test_lc_quality.py
