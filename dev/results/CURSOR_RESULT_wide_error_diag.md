CURSOR RESULT - 2026-08-04 09:10 UTC+2

**Verdict: H1-global (quoted error budget ~2x low). No floor applied.**

What I did
Ran T1-T4 wide-rig error-budget diagnostic on batch D production check-star LCs
(equipment_id 1, draft_435, n=162 fields). No re-cut. Pre-registered reading applied.

Harness: `dev/tools/wide_error_budget_diag.py`
Output: `tmp/wide_error_budget_diag.json`

## Summary numbers

| metric | value |
|--------|-------|
| median chi2_red_clipped | **3.55** |
| median quoted err (post-scint) | **9.4 mmag** |
| median measured scatter | **20.1 mmag** |
| scatter / err ratio (median) | **1.96** |
| scintillation at X=1 | **1.73 mmag** |

## T1 -- scatter vs magnitude

| quartile | n | G range | scatter mmag (median) | err mmag (median) | chi2_clip |
|----------|---|---------|----------------------|-------------------|-----------|
| bright | 160 | 8.74 | **20.3** | 9.5 | 3.57 |
| middle | 0 | -- | -- | -- | -- |
| faint | 2 | 9.22 | **6.5** | 4.5 | 2.08 |

Note: check_kmag metadata collapses to G~8.74 for 160/162 fields (same check star
1499906247391001088), so T1 magnitude stratification is weak. Where G differs (2 fields),
scatter is lower -- not inconsistent with H1 on the bulk population.

**T1 reading:** ~20 mmag scatter flat across the dominant magnitude bin -> systematic
(H1), not faint-star-dominated inflation.

## T2 -- scatter vs quoted err

| metric | value |
|--------|-------|
| linear fit slope | **1.54** |
| intercept | **+5.1 mmag** |
| slope through origin | **1.83** |
| median scatter/err | **1.96** |
| p95 scatter/err | **3.87** |

**T2 reading:** not on 1:1; ~**2x constant multiplier** on the bulk population (H1-global).
A few stars sit far above 1:1 (p95 3.9x) -- outlier fields, not the median story.

## T3 -- error decomposition (representative stars)

**Bright check (G=8.74, cid 1499906247391001088):**

| term | mmag |
|------|------|
| photon | 2.5 |
| ensemble | **58.1** |
| scintillation | 1.8 |
| sigma_sys | 0.0 |
| **total quoted** | **58.2** |
| measured scatter | 32.7 |
| chi2_red_clipped | **0.66** |

**Faint check (G=9.22, cid 1497528072458898432):**

| term | mmag |
|------|------|
| photon | 3.4 |
| ensemble | 2.0 |
| scintillation | 1.8 |
| **total quoted** | 4.3 |
| measured scatter | 6.5 |
| chi2_red_clipped | **2.19** |

**T3 reading:** no single term increased by ~2x closes the gap on all fields. The
**median** field has quoted err ~9.4 mmag vs scatter ~20 mmag (global ~2x). On the
bright representative, the **ensemble term dominates** (58 mmag) and actually
**overquotes** (chi2 0.66). The mis-scaling is population-heterogeneous: ensemble
SEM propagation is not uniformly wrong in one direction.

Defensible fix: **audit ensemble SEM + photon term propagation** (Honeycutt 1992),
not a quadrature floor.

## T4 -- ensemble

| metric | value |
|------|-------|
| n_comp median | **8** |
| n_comp range | 3-8 |
| fields with n_comp < 5 | **12** |
| LOO ensemble scatter (representative field) | **9.9 mmag** |
| high catalog comp_rms comps (>50 mmag) | **20 listed** |

**T4 reading:** comp count is adequate (median 8). Several comps have catalog
comp_rms 50-78 mmag. LOO production ensemble scatter ~10 mmag on representative
field. Supports **H1-ensemble contribution** on some fields but **not** a few-comp
Honeycutt failure as the primary median mechanism.

## Pre-registered verdict

**H1-global** (primary): median scatter/quoted-err ratio **~2.0**; chi2_red ~3.5
is a budget-scaling symptom, not an irreducible 15 mmag floor.

**H1-ensemble** (secondary): ensemble term dominates on some fields; variable/high-RMS
comps present; fix = comp QA + ensemble SEM propagation, not `sigma_sys` floor.

**H2 rejected** for the median population: not honest 20 mmag intrinsic scatter with
correct errors -- the budget is ~2x low on median.

## Action (R8)

- **Do NOT apply** sigma_sys floor outside 2-5 mmag (prior GATE 1 conclusion stands).
- Route error-budget fix to **post-batch-E numeric item**: ensemble SEM + photon term
  audit on equipment_id 1.
- **Batch E proceeds** (flux/detections; not blocked).

## Files changed

- `dev/tools/wide_error_budget_diag.py` (new)

## Errors (if any)

None. Initial harness runs failed on imports/magnitude column; fixed in-session.
