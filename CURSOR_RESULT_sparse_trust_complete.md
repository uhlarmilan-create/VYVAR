CURSOR RESULT -- SPARSE-TRUST-COMPLETE -- 2026-07-14

What I did
Completed SPARSE-TRUST validation (S1-S4), full PZQ sigma_r report (Part 1), test-suite
accounting, and anchor-safe backfill to tmp work dirs. Anchor tree read-only throughout.

## Part 1 -- PZQ sigma_r report

Output: tmp/pzq_sigma_r/ (pzq_sigma_r_summary.json, pzq_sigma_r_summary.md, 3x sigma_N vs N figures)

| rig | n_stars | median sigma_w [CI] mag | median sigma_r [CI] mag |
|-----|---------|-------------------------|-------------------------|
| wide_Carl-Zeiss | 12 | 0.01118 [0.0105,0.0119] | 0.00552 [0.0047,0.0065] |
| Newton_g | 12 | 0.02225 [0.019,0.025] | 0.01882 [0.015,0.022] |
| Newton_i | 12 | (see JSON per_rig_medians_bootstrap) | |

Power statement: wide rig (draft_424, N~139) is primary sigma_r result. Newton g/i
24/24 stars pass >=5-bin rule (indicative only; r_60_4 cohort pending in fit JSON).

k'' probe: wide rho=-0.125 (p=0.70, n=12) vs |BP-RP offset| x airmass range; lag1 n=0
(sem_cause has Newton comps only). Newton g rho=0.13 (n=12); Newton i rho=0.48 (p=0.11).
ROADMAP verdict: k'' priority DOWN on wide (weak colour-airmass correlation with sigma_r).

## Part 2 -- S1-S4 validation

Anchor tree untouched: yes.
Backfill output (sidecars only, no LC writes):
- tmp/sparse_trust_validation/draft424_work/photometry/lightcurves/ (178 sidecars)
- tmp/sparse_trust_validation/draft426_r60_work/photometry/lightcurves/ (6 sidecars)
Proc CSV read from live draft_000424 / draft_000426 detrended_aligned (read-only).

### S1 slow synthetic
| N | coverage | clip_rate | pass |
|---|----------|-----------|------|
| 15 | 100.0% | 20.2% | yes |
| 25 | 99.8% | 9.4% | yes |
| 139 | 98.4% | 0.2% | yes |

### S2 draft_424 (n>=5 control)
160 targets with n_pool>=5. GREEN->RED flips: **0**. Flip report:
tmp/sparse_trust_validation/s2_flip_report.csv

### S3 draft_426 r_60_4 + SS Cam
All 6 targets n_comps=2, check_sparse=1, single_comp flag (1 ensemble comp after check pick).
SS Cam (1112113066119992064):
- production_lc_err chi2 = 21.38 (sem_cause setup_r_60_4.json)
- sparse trust band = YELLOW (single_comp; R/p/x2 NaN -- kmag not produced at n=1 ensemble)
- n=2, N_epochs=0 finite kmag in sidecar (spec n==1 rule)
Tuples: tmp/sparse_trust_validation/s3_verdict_tuples.csv

### S4 anchor integrity
Recomputed core SHA: bf3743a150d788283eab2ab51db7b31f59e6d1c481159208bbe3f573092ec975
(expected match). Work dir contains 0 lightcurve_*.csv files (sidecar-only backfill).
Production err unchanged by construction.

Full summary: tmp/sparse_trust_validation/validation_summary.json

## Part 3 -- Test suite accounting (15 vs 11 skipped)

The "15 -> 11 skipped" in the prior result compared **full suite** skipped count (15) against
**`-m "not slow"`** skipped count (11). Skipped count is unchanged at **15** on full suite.

| item | count at 7ed7459 | count at HEAD | explanation |
|------|------------------|---------------|-------------|
| passed | 796 | 806 (803 + `-m not slow`) | +7 fast sparse_trust tests; +3 slow S1 when slow included |
| skipped (full) | 15 | 15 | unchanged -- all fixture-missing skips |
| deselected (`-m not slow`) | 4 | 7 | +3 new `@pytest.mark.slow` S1 tests |

The four tests deselected at 7ed7459 with `-m not slow`: 1x test_pre_cal_proc_csv_naming_e2e slow.
The seven deselected at HEAD: above + 3x test_s1_synthetic_triangulation_coverage.
No tests deleted. No skip semantics changed. No tests unskipped silently.

## Part 4 -- Scope accounting

**backfill_check_kmag_sidecars.py:** additive sidecar writer for export/trust validation.
Writes ONLY to `--out-phot` work dir; reads `--src-phot` + `--proc-dir` read-only. Used for
S2-S4 without touching anchor snapshot. Sparse fallback picks lowest comp_rms check when
select_check_star fails at n=2.

**select_constant_calibrators.py:** I/O-only change -- `compute_check_ensemble_mag_calib` now
returns CheckEnsembleResult; uses `.kmag` and `n_comp_min=2` in LOO helper. **No selection
behavior change** (COMP-POOL-R thresholds untouched; stability still n_comp_min=3 in chi2 path).

## Errors (if any)
None blocking. k'' wide probe n=0 (documented). SS Cam single_comp at n=2 pool (expected).

## Files changed
- sparse_trust_core.py (triangulation_hat_ci)
- tests/test_sparse_trust_core.py (S1 CI fix)
- scripts/pzq_sigma_r_report.py (complete Part 1)
- scripts/backfill_check_kmag_sidecars.py (work-dir + sparse fallback)
- scripts/sparse_trust_complete_run.py (S1-S4 runner)
- docs/VYVAR_STATE.md, VYVAR_JOURNAL.md, VYVAR_ROADMAP.md
- CURSOR_RESULT_sparse_trust_complete.md

## pytest
803 passed, 11 skipped (`-m not slow`); 806 passed, 15 skipped (full incl. slow S1).
ruff clean on touched files.

**NOT PUSHED** -- awaiting Milan review.
