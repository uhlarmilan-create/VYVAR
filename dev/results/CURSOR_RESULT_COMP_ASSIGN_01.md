# CURSOR RESULT - COMP-ASSIGN-01

Date: 2026-08-15
Baseline: 5cfb285 (IMPL-04 tip stamp bb02bb8)
Tip: PENDING
Push: NO

## What I did

Restored Milan's decided comparison-star authority chain: uncapped pool
(step 1) -> per-target colour / RMS / distance clamp 3-8 (step 2) ->
photometry from that set -> stability as a post-LC verdict only.

## Deltas (D1-D7)

| # | Change |
|---|---|
| D1 | Pool RMS kept (`build_global_comp_pool` + `attach_comp_rms_to_pool_rows`) |
| D2 | Stopped dropping pool `comp_rms`; step 2 reads it; no per-target re-derivation on the default path; `comp_rms_fieldwide` NaN |
| D3 | `_select_comps_by_color_then_rms`: colour ladder -> RMS -> `_dist_deg` -> `catalog_id`; `head(n_comp_max)` |
| D4 | `check_comparison_stability` moved after LC write; sidecar + trust only |
| D5 | `ensemble_normalize` consumes delivered keys; no re-clamp / quality drop |
| D6 | Layer-2 selection restored via D3 (COMP-ADMIT uncapped truncation removed) |
| D7 | `phase01_comparison_n_comp_min/max` (3/8) honoured end-to-end |

## Draft 514 rebuild

Phase 1: `dev/tools/comp_assign_01_phase1.py` (variable_targets.csv, not active stub).
Phase 2A acceptance subset at r=9.5 (IMPL-04 radius held for separable check comparison).

Membership (`COMP_ASSIGN_01_measure.json`):

| metric | value |
|---|---:|
| targets with comps | 97 |
| pairs | 748 |
| min / max / median | 3 / 8 / 8 |
| outside 3-8 | **0** |

BO / FW comps listed with delta_bprp, comp_rms, dist_deg in the measure JSON.

Check-star scatter (aperture fixed 9.5 px):

| Target | check before (IMPL-04) | check after |
|---|---:|---:|
| BO CVn | 9.06 | 17.52 |
| FW CVn | 8.58 | 20.32 |

Quality sidecar membership equals CSV set for BO/FW (set equality). Stability
is post-LC; ensemble used all-good membership during photometry.

Backfill: when n_comp=8, independent check must come from the **field** pool
(`backfill_check_kmag_sidecars.py` aligned with Phase 2A field-pool pick).

## Tests

- `test_ensemble_normalize_consumes_exact_step2_set`
- `test_select_comps_color_rms_distance_clamp`
- `--fast` PENDING

## Files

- `src_py/photometry_core.py` - selection, pool RMS, stability order
- `dev/scripts/backfill_check_kmag_sidecars.py` - field-pool check pick
- `dev/tests/test_forced_phot_and_weights.py` - set equality + clamp tests
- `dev/tools/comp_assign_01_phase1.py`
- `dev/results/COMP_ASSIGN_01_measure.json`, this result
