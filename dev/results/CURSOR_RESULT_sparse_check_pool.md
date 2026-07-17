CURSOR RESULT -- SPARSE-CHECK-POOL -- 2026-07-14

What I did
Merged SPARSE-TRUST Amendment 1 (external K sourcing), implemented sparse-branch K pool +
n=1/n=2 semantics, re-ran S2/S3/S4 validation. Seven-commit SPARSE-TRUST chain remains
local; push pending Milan explicit yes (Part 0).

## Part 0 -- Push gate
**Milan: push the 7-commit SPARSE-TRUST chain to origin/main?** (review-approved by Claude;
S1/S2/S4 PASS on completion run.) This task adds further local commits; those stay unpushed
until your next review.

## Part A -- Amendment merge
Merged into docs/VYVAR_SPARSE_TRUST_SPEC.md (changelog 2026-07-14 Amendment 1): section 2.1
external K sourcing, revised n semantics, scope guard, S3/S2 re-verify notes.

## Part B -- Implementation
- select_external_check_star(): constant + p2p-good + SNR floor; colour/tier NOT required;
  brightness proximity, tie-break p2p; k_source / k_tier_excluded / k_colour_offset.
- n = ensemble comps only; K external. n=2 + external K -> full triangulation; n=1 -> kmag
  + 2-star R test (band capped YELLOW, numbers recorded).
- Raw R for band; trust_R_detrend diagnostic column.
- build_comp_photon_mag_from_frames(): derive relative err from flux+sigma_bkg_ap when proc
  CSV lacks err column (r_60_4 fix).
- Sidecar columns: k_source, k_colour_offset, k_tier_excluded, k_colour_caveat,
  trust_R_detrend (+ lo/hi).
- tests/test_sparse_external_k.py (8 tests).

## Part C -- S2/S3/S4 re-verify

Anchor tree untouched: yes.
Backfill work dirs (sidecar-only):
- tmp/sparse_trust_validation/draft424_work/photometry/
- tmp/sparse_trust_validation/draft426_r60_work/photometry/

### S2 (wide n>=5, scope guard)
160 targets n>=5. GREEN->RED flips: 0. Band changes vs completion baseline: 0.
tmp/sparse_trust_validation/s2_flip_report.csv

### S3 r_60_4 + SS Cam (external K)

| target | K_id | k_source | n | N | R [lo, hi] | R_detrend | p_stab | band |
|--------|------|----------|---|----|------------|-----------|--------|------|
| SS Cam | 1112110935816253440 | comp_pool_external | 2 | 25 | 2.01 [1.22, 3.89] | 1.99 | 0.0 | YELLOW |
| (all 6 sparse) | external K | comp_pool_external | 2 | 25 | 2.0-2.9 | ~2.0-2.9 | 0-1 | YELLOW |

production_lc_err chi2 (SS Cam): 21.38 (spec-3.4 model alongside sem_cause baseline).
r baseline row: chi2=21.38 production; sparse trust R=2.01 [1.22, 3.89] YELLOW.
SS Cam band OPEN -- Milan confirms on evidence (not incapacity artifact).

Full tuples: tmp/sparse_trust_validation/s3_verdict_tuples.csv

### S4
Recomputed core SHA: bf3743a150d788283eab2ab51db7b31f59e6d1c481159208bbe3f573092ec975
(match). Work dir LC files: 0.

### S1 slow
PASS (unchanged): N=15/25/139 coverage 100.0% / 99.8% / 98.4%.

## Part D -- Docs corrections
1. k'' ROADMAP: UNCHANGED (wide rho=-0.125, p=0.70, n=12 underpowered null; Newton i
   rho=0.48 p=0.11 suggestive; re-test with larger cohort).
2. PZQ cross-checks recorded in STATE + SIGMA_FLOOR_SPEC: wide sigma_r ~5.5 mmag consistent
   with ~4.5 mmag rig constant (SIGMA-A4); Newton g sigma_r ~18.8 mmag ~ fitted floor 18.0
   mmag (correlated noise; binned Newton quantities still underestimated).
3. SS Cam ROADMAP: evidence-based YELLOW (R~2, not single_comp NaN); OPEN for Milan.

## Errors (if any)
None blocking. Initial r_60_4 R=NaN fixed (proc CSV err derivation).

## Files changed
- docs/VYVAR_SPARSE_TRUST_SPEC.md, VYVAR_STATE.md, VYVAR_JOURNAL.md, VYVAR_ROADMAP.md
- docs/VYVAR_SIGMA_FLOOR_SPEC.md
- sparse_trust_core.py, check_star_kmag.py
- scripts/backfill_check_kmag_sidecars.py, scripts/sparse_trust_complete_run.py
- tests/test_sparse_external_k.py
- CURSOR_RESULT_sparse_check_pool.md

## pytest
811 passed, 11 skipped (`-m "not slow`); +8 new external-K tests.
ruff clean on touched files.

**NOT PUSHED** -- awaiting Milan review.
