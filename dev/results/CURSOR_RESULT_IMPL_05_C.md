# CURSOR RESULT - IMPL-05 Item C (RMS-first + single-source)

Date: 2026-08-16
Baseline: f200adb (Item B)
Tip: **9dfeaa3**
Push: NO

`--fast`: 1415 passed, 22 skipped (pytest `-m "not slow"`).

## What I did

Reordered comparison selection to RMS -> colour -> distance inside
`_select_comps_by_rms_then_color` (honest rename; thin alias retained).
Excluded blends using CoG `snr_cog_isolation_fwhm` x night seeing FWHM
(`VY_FWHM`, full masterstars catalogue for NN). Fire proofs + DECISIONS.

## Physics (DECISIONS)

Colour cheap on this rig; high RMS expensive in unweighted flux sum;
single-source supersedes PRE-IMPL Q5 for comps only.

## Implementation notes

1. Sort within colour ladder step: `comp_rms`, `|dBP-RP|`, distance.
2. NN isolation against **full** `masterstars_df` (not trimmed global pool) -
   pool-only NN missed faint neighbours inside 3 FWHM.
3. `run_phase0_and_phase1` resolves `fwhm_px` from MASTERSTAR `VY_FWHM` first
   (was defaulting to 3.7 / Gaussian core ? iso radius 11.1 px instead of 15.6).

## Fire proofs

- Fail on COMP-ASSIGN-02 snapshot: contains blends (masterstars NN).
- Pass on rebuilt CSV: 0 blends, 0 above-ceiling.
- Unit: RMS-first, blend exclude, ceiling still before head.

## Fixed meter (acceptance Phase 2A, per-mag apertures from B)

| Target | n_comp | check MAD [mmag] | check std [mmag] | comp_rms [mmag] |
|---|---:|---:|---:|---|
| BO CVn | 5 | **8.6** | 8.9 | 20-47 |
| FW CVn | 8 | **9.8** | 12.0 | 14-30 |

Prediction ~10-12 mmag: met. COMP-ASSIGN-02 was ~16-19 mmag at r=9.5.

FW CVn: quietest comps now fill the set (RMS-first); colour ladder still
admits |dBP-RP| up to the step that reaches n_comp_min.

## Artifacts

- `dev/results/COMP_ASSIGN_02_comparison_stars_per_target.csv` (fail side)
- `dev/results/IMPL_05_C_fixed_meter.json`
- `dev/tools/impl_05_phase2a_acceptance.py`

## Files changed

- `src_py/photometry_core.py`
- `src_py/comp_selection_per_target.py`
- `dev/tests/test_forced_phot_and_weights.py`
- `docs/VYVAR_DECISIONS.md` (COMP-ASSIGN-03)
