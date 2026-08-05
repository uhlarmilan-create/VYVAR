CURSOR RESULT - 2026-08-04 (WIDE-ERR E2)

What I did
Tested whether comp residuals are correlated (invalidating sqrt(n) SEM divisor) and whether
residual common mode is spatially structured. Retracted E1.4 via falsification. Read-only;
harness dev/tools/wide_err_e2.py; output tmp/wide_err_e2/wide_err_e2.json.

## Guard rails

| check | result |
|-------|--------|
| anchor_manifest_check.py (pre-run) | PASS (exit 0) |
| anchor_manifest_check.py (post-run) | PASS (exit 0) |

## E2.0 -- Retract E1.4

E1.4 compared quoted SEM to std(m_i - ens_med). That quantity is the ensemble **brightness
spread** (photometry_core.py:3423-3429 documents ens_med flux-sum vs comp_ref_map fix), not
measurement scatter.

**Falsification (166 fields):**

| correlate | Spearman rho | p | n |
|-----------|--------------|---|---|
| E1.4 ratio vs comp spread (max-min mag) | **0.714** | 3.6e-27 | 166 |
| E1.4 ratio vs n_comp | -0.056 | 0.47 | 166 |

Strong positive correlation with comp spread **confirms artifact**. **E1.4 retracted**;
**WIDE-ERR-SEM-ARITH withdrawn.**

Arithmetic sanity (W2): median comp spread 0.92 mag; ~8 comps -> brightness-spread std ~ 0.27 mag,
naive SEM ~ 93 mmag vs quoted ~ 9.4 mmag, ratio ~ 10 (E1.4 measured 13.2).

## E2.1 -- Correlation of comp residuals

Production ``comp_resid`` per frame (m - comp_ref_map[cid]); same comps as flux sum
(photometry_core.py:3430-3438). Pairwise Pearson correlation across comp pairs, mean rho_bar.

| metric | value |
|--------|-------|
| n_fields | 166 |
| rho_bar median | **0.393** |
| rho_bar IQR | 0.235 / 0.393 / 0.551 |
| n_comp median | 8 |
| predicted factor sqrt(1+(n-1)*rho_bar) median | **1.903** |
| predicted factor IQR | 1.517 / 1.903 / 2.166 |
| measured sigma_robust/err median (W1) | **1.828** |
| Spearman(predicted, measured) | **0.674**, p = 6.6e-23, n = 163 |

**Reading:** rho_bar measures across-frame comp correlation c(t), **not** within-frame SEM
independence. E2.2 check_pc1_corr ~ 0 confirms c(t) cancels between target and ensemble.
**Revised (E3.0):** The 1.90 vs 1.83 numerical agreement is **unexplained, not explanatory**.
~~**WIDE-ERR-CORRELATED-COMPS**~~ **withdrawn** as causal claim. rho_bar = 0.393 retained as fact.

## E2.2 -- Spatial structure of common mode

PC1 of comp x frames residual matrix; regress comp loadings on detector (x,y) and radial offset
from flux-weighted centroid. Check star offset from centroid reported.

| metric | value |
|--------|-------|
| n_fields | 166 |
| fraction fields significant (x gradient) | **9.0%** |
| fraction fields significant (y gradient) | **4.2%** |
| fraction fields significant (radial gradient) | **1.8%** |
| median check-star offset from centroid (px) | 958 |

Check-star PC1 correlation with common mode: near zero in sample fields (e.g. T3 field
1485540612577549568: rho_bar = 0.094, check_pc1_corr = 0.003).

**Reading:** rho_bar is high (E2.1) but **spatial gradients are NOT significant in most
fields** (9% x, 2% radial at p<0.05). A spatially uniform common mode **should cancel** between
target and ensemble. **Contradiction reported plainly:** temporal/non-spatial correlation can
inflate the sqrt(n) SEM divisor, but a **structured spatial missing correction (EPD-style) is
NOT established** on this data. Check-star excess is not predicted by fitted spatial gradients.

## E2.3 -- Newton cross-check

**NOT AVAILABLE** -- no Newton/Dablice draft with check_kmag LCs on disk (Archive/Drafts has
435, 435_p1mini, 435_snapshot, 499, 500 only; none identified as Newton). No new photometry run.

## Combined line

**E2 superseded by E3.0** -- ~~WIDE-ERR-CORRELATED-COMPS~~ withdrawn; see
CURSOR_RESULT_wide_err_e3.md.

## Files created

- dev/results/CURSOR_RESULT_wide_err_e2.md (this file)
- dev/tools/wide_err_e2.py
- tmp/wide_err_e2/wide_err_e2.json

## Errors

None blocking.
