CURSOR RESULT — 2026-07-09 (SIGMA-A3)

What I did
Added variant (e) with Honeycutt ensemble SEM (dual path a/b), joint refit (d)+(e), draft_424 + SS Cam/V0611 reruns, tests, docs, push.

## Output / findings

### Origin hash

| Item | Value |
|------|-------|
| Base | `b7db856` |
| Push | `a676848` |

### Variant median chi2/dof and IQR — draft_424 calibrator ensemble

| Variant | median chi2/dof | IQR |
|---------|-----------------|-----|
| (a) howell_only | 1.157 | 0.982 |
| (b) howell_scint_full | 1.150 | 0.916 |
| (c) howell_scint_fresid | 1.153 | 0.945 |
| (d) howell_scint_fresid_floor | 1.000 | 0.158 |
| (e) howell_scint_fresid_floor_ensemble | 1.000 | 0.137 |

### Joint fit (f_resid, sigma_floor) — variant (d) and (e)

| Variant | f_resid | f_resid CI (16–84%) | sigma_floor (mmag) | sigma_floor CI (16–84%) | median chi2/dof | IQR |
|---------|---------|---------------------|--------------------|-------------------------|-----------------|-----|
| (d) | 0.74 | [0.0, 1.0] | 10.5 | [9.5, 11.0] | 1.000 | 0.158 |
| (e) | 0.0 | [0.0, 0.62] | 6.5 | [5.5, 7.5] | 1.000 | 0.137 |

(e) f_resid pinned_edge: lower.

### Prediction verdict

| Item | Value |
|------|-------|
| Verdict | floor_did_not_collapse |
| sigma_floor (e) | 6.5 mmag |

### Ensemble SEM path (a)/(b) — draft_424

| Stat | Value |
|------|-------|
| clamp fraction median (1.1a) | 0.0 |
| agreement pooled median \|diff\| (mag) | 0.0406 |
| agreement pooled p95 \|diff\| (mag) | 0.0563 |
| n calibrators compared | 8 |

Per-calibrator production SEM (path b) median (mag): ~0.0078 (7.8 mmag) typical; p95 ~0.011–0.012.

### Chi2/dof per calibrator — all five variants (draft_424)

| G | (a) | (b) | (c) | (d) | (e) |
|---|-----|-----|-----|-----|-----|
| 9.29 | 12.26 | 9.07 | 10.28 | 1.23 | 1.32 |
| 10.47 | 2.13 | 2.02 | 2.07 | 0.90 | 0.93 |
| 11.15 | 1.92 | 1.86 | 1.89 | 1.08 | 1.09 |
| 11.85 | 1.16 | 1.16 | 1.16 | 0.99 | 0.99 |
| 12.35 | 1.15 | 1.14 | 1.15 | 1.01 | 1.01 |
| 12.68 | 0.83 | 0.83 | 0.83 | 0.78 | 0.78 |
| 12.90 | 1.04 | 1.03 | 1.03 | 0.97 | 1.01 |
| 13.23 | 0.55 | 0.55 | 0.55 | 0.52 | 0.54 |

Plot: `tmp/sigma_budget/chi2_vs_g_draft000424_NoFilter_60_2.png`
JSON: `tmp/sigma_budget/calibrator_chi2_summary.json`

### SS Cam / V0611 check-star — variant (e) + ensemble SEM

| Case | check_id | ensemble SEM median (mag) | ensemble SEM p95 (mag) | chi2/dof (e) | chi2/dof (d floor) | chi2/dof (howell_only) |
|------|----------|---------------------------|------------------------|--------------|--------------------|-----------------------|
| SS Cam g_60_4 | 1112108942951430528 | 0.0 | 0.0 | 0.158 | 0.156 | 0.158 |
| SS Cam r_60_4 | 1112117498526276864 | 0.0072 | 0.0186 | 0.716 | 0.672 | 0.811 |
| SS Cam i_70_4 | 1111749157833870208 | 0.0 | 0.0 | 0.326 | 0.246 | 0.329 |
| V0611 g_60_4 | 1112108942951430528 | 0.0054 | 0.0110 | 0.040 | 0.040 | 0.040 |

JSON: `tmp/sigma_budget/sparse_comp_diag.json`

### Gates

| Gate | Result |
|------|--------|
| pytest | 674 passed, 15 skipped |
| ruff | clean |
| session_baseline_check --fast | OVERALL PASS |
| production wiring | none |
| anchor | untouched |

### Deviations

1. Path (a) LC err on LOO calibrators inherits anchor-target `err` column (not star-specific); path (b) primary.
2. min_frames relaxed 200?120 (archive max 139 comp_n_frames).
3. (e) f_resid bootstrap CI lower edge pinned at 0.0.

## Errors (if any)

None.

## Files changed

- `sigma_budget.py` — variant (e) constant, `ensemble_sem_mag` in quadrature
- `scripts/chi2_sigma_gate.py` — dual SEM extraction, variant (e), vectorized joint fit
- `scripts/select_constant_calibrators.py` — production ensemble scatter, joint (e) refit
- `scripts/sparse_comp_diag.py` — check-star variant (e), ensemble SEM report
- `tests/test_sigma_a3.py` (new)
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`
