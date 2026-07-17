CURSOR RESULT ù 2026-07-09 (SIGMA-A4)

What I did
Floor attribution on draft_424 calibrators (k2 / phase / X / time); Newton bin4 gain/RN forensics + ensemble SEM=0 trace; hypothesis corrected-sigma chi2; tests; docs; push.

## Output / findings

### Origin hash

| Item | Value |
|------|-------|
| Base | `a676848` |
| Push | `98fc719` |

### Part 1 ù Floor attribution (draft_424, 8 calibrators)

**floor_before (variant e refit baseline):** 6.5 mmag

| Candidate | var_explained pooled | floor_before (mmag) | floor_after (mmag) | floor_delta (mmag) |
|-----------|---------------------|---------------------|--------------------|--------------------|
| k2_signature | 1.6e-07 | 6.5 | 6.5 | 0.0 |
| phase_signature | 1.5e-04 | 6.5 | 4.5 | 2.0 |
| X_linear | 9.7e-04 | 6.5 | 6.0 | 0.5 |
| time_linear | 1.1e-03 | 6.5 | 5.0 | 1.5 |

**k2_effective (shared slope, XùDelta_color):** -3.57e-05 mag/(airmassùmag color)  
**k2 CI (16ù84%):** [-0.00247, 0.00357]

**k'' recovery (Part 1.1c):** floor_after unchanged at 6.5 mmag ? **0.0 mmag recoverable via k2 component removal**

Plot: `tmp/sigma_budget/floor_attribution_draft000424_NoFilter_60_2.png`  
JSON: `tmp/sigma_budget/floor_attribution_draft000424_NoFilter_60_2.json`

Per-calibrator variance explained (k2 / phase), max: G9.3 k2=6.3e-04, phase=1.8e-02; G12.9 phase=3.4e-02.

### Part 2 ù Bin4 gain/RN forensics (draft_426)

**Archive equipment:** ID=4 C5A-150M (IMX411); draft OBS_DRAFT row. Header GAIN=12.48 e-/ADU matches DB equipment_id=2 (C3-26000/IMX571) bin4-scaled gain 0.78ù4ù=12.48, not eq4 DB (1.0ù16=16.0).

| Setup | bin | gain_used | gain_src | RN_used | RN_src | gain_exp | RN_exp | ?_ratio | ?ù_pred(1/?_ratioù) |
|-------|-----|-----------|----------|---------|--------|----------|--------|---------|---------------------|
| g_60_4 | 4 | 12.48 | header | 14.08 | db | 16.0 | 14.08 | 1.132 | 0.780 |
| i_70_4 | 4 | 12.48 | header | 14.08 | db | 16.0 | 14.08 | 1.133 | 0.779 |
| r_60_4 | 4 | 12.48 | header | 14.08 | db | 16.0 | 14.08 | 1.133 | 0.780 |
| z_90_4 | 4 | 12.48 | header | 14.08 | db | 16.0 | 14.08 | 1.134 | 0.778 |

Scaling: software_sum gainùbù, RNùb (param_resolver).

**Observed check-star ?ù/dof (howell_only):** g_60_4 SS Cam 0.158; i_70_4 0.329; r_60_4 0.811; V0611 g 0.040. ?_ratio accounting alone does not match full deficit (pred ~0.78 vs obs 0.04ù0.33).

### Ensemble SEM = 0 trace (producer photometry_core ~2608ù2624)

| Case | n_other_comps | n_resid p50 | frames resid<2 | scatter_zero_frac | scatter_nan_frac |
|------|---------------|-------------|----------------|-------------------|------------------|
| SS Cam g_60_4 | 2 | 2.0 | 0/24 | 0.0 | 0.042 |
| SS Cam i_70_4 | 2 | 2.0 | 0/25 | 0.04 | 0.0 |
| V0611 g_60_4 | 8 | 8.0 | 0/24 | 0.0 | 0.042 |

Producer: scatter=0 when len(comp_resid)<2. Sparse g/i: n_resid=2 always; i has 4% zero scatter (2-comp std=0 frames). Harness SEM=0 on g/i from LC err-decomposition clamp (no star-specific err on check sidecar).

### Hypothesis corrected-sigma (gain_exp=16, RN_exp=14.08, variant bin4_gain_rn_hypothesis)

| Case | baseline howell ?ù/dof | hypothesis ?ù/dof |
|------|-------------------------|-------------------|
| SS Cam g_60_4 | 0.158 | 2.563 |
| SS Cam i_70_4 | 0.329 | 1.688 |
| SS Cam r_60_4 | 0.811 | 3.745 |
| V0611 g_60_4 | 0.040 | 0.601 |

?ù moves **away** from 1 under DB-scaled gain hypothesis.

JSON: `tmp/sigma_budget/bin4_sigma_forensics.json`

### Gates

| Gate | Result |
|------|--------|
| pytest | 679 passed, 15 skipped |
| ruff | clean |
| session_baseline_check --fast | OVERALL PASS |
| production wiring | none |

### Deviations

1. Calibrator bp_rp from field-wide comp CSV (not anchor-only pool).
2. draft_426 DB equipment_id=4 vs header gain matching IMX571 (eq2) scale.
3. z setup is z_90_4 (not z_60_4) in archive.
4. Hypothesis uses DB bin1ùbù scaling; header gain already session-truth at 12.48.

## Errors (if any)

None.

## Files changed

- `scripts/sigma_floor_attribution.py` (new)
- `scripts/bin4_sigma_forensics.py` (new)
- `tests/test_sigma_a4.py` (new)
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_STATE.md`, `docs/VYVAR_JOURNAL.md`
