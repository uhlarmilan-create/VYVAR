# CURSOR RESULT - COMP-POOL-01 Stage 3

Date: 2026-08-14
Parent: Stage 2 commit `696c849`

## What I did

Assignment algorithms (colour tiers, adaptive delta-mag, Broeg scores, sparse_fallback) are **unchanged**. Stage 3 only makes the sparse-field **relaxation order explicit and recorded** in per-target `selection_note` provenance.

## Explicit relaxation order (pair criteria only)

1. `colour_tier_widen_T1_to_T4` - existing greedy tier ladder
2. `adaptive_delta_mag` - widen |dmag| toward absolute ceiling when too few candidates
3. `sparse_fallback_path` - existing `comp_path=sparse_fallback` route

Never relaxed in assignment (pool Stage 2 owns these): detect fraction, faint/bright limits, dilution, stability indices, catalogue variability.

Colour caution (task): Astrokit dmag/colour tolerances are for filtered work; unfiltered red CMOS colour term is larger (open D10-1). No silent tightening or loosening of colour limits was introduced here.

## Ensemble size design target

`phase01_comparison_n_comp_max` remains the per-target ensemble cap (design target ~8-10). That is assignment, not pool size. Pool remains uncapped.

## Validation note (rebuild required for LC deltas)

Stage 2 already showed BO CVn (`1498613634033133184`) loses one pool member on draft 512 (`1497368849430107904`: faint+dilution). After a photometry rebuild, assignment will select from the remaining admitted pool; expect TIER1 count 4 unless a different admitted star replaces it under the same colour/mag/spatial rules. LC metrics (`check_scatter`, `ac_scatter`, `lc_rms_ooe`) must be remeasured on that rebuild - not claimed here.

## Files

- `src_py/comp_selection_per_target.py`: `COMP_ASSIGNMENT_RELAX_ORDER`, `format_assignment_relax_provenance`
- `dev/tests/test_comp_pool_assignment_s3.py`
- This memo

## Pre-registered

- P-R3: any BO ensemble change after rebuild must cite Stage-2 drop reason or a recorded relax firing, not LC appearance.
