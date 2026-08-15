TASK_ID: COMP-ADMIT-03

CURSOR RESULT - 2026-08-15

SCIENCE CHANGE. Removes comparison-star admission rejection layer.
Push: NO. Waiting for Milan.

Detail: `dev/results/CURSOR_RESULT_COMP_ADMIT_03.md`

## Forced photometry

Comps are DAO-detected (+ pass-2), then aperture at locked MASTERSTAR XY.
Not forced every catalogue star every frame. `detect_frac` deleted as a cut;
follow-up needed for true fixed membership.

## Design

Three gates only: saturation/non-linearity, known variable, geometry.
Weight: sigma_eff^2 = rms^2 + (c_col*|dBP-RP|)^2 + (c_dist*r)^2; w=1/sigma_eff^2.
Universality tests pass (permutation/subset exact; fire proof on old rank rule).

## Draft pools after

512: 141; 513: 1238; 435: 783 (not emptied). c_col/c_dist named zeros in harness.

## --fast

OVERALL PASS (1369 passed, 27 skipped). Commit `b6e0e29`.
