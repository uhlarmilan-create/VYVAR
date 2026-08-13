CURSOR RESULT - 2026-08-13 (INV-CAL-02 design)

What I did
Investigated writers/readers of `calibrated/`, on-disk header reliability across drafts 435/509/510 and full Archive sweep, double-subtract guard, parallel directory hazards, and field conventions. Wrote design spec (no implementation).

## Output / findings

Spec: `dev/results/specs/VYVAR_CAL_STAGE_SPEC.md`

Recommendation: **Option A** (stamp + verify, keep in-place mutation).

## Errors (if any)

None.

## Files changed

- `dev/results/specs/VYVAR_CAL_STAGE_SPEC.md` (new)
- `dev/results/CURSOR_RESULT_inv_cal_02_design.md` (this file)
