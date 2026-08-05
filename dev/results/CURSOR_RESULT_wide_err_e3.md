CURSOR RESULT - 2026-08-04 (WIDE-ERR E3)

What I did
Tested whether a missing per-star noise term on the target explains the check-star underquote.
Corrected E2.1 interpretation. Read-only; harness dev/tools/wide_err_e3.py;
output tmp/wide_err_e3/wide_err_e3.json.

## Guard rails

| check | result |
|-------|--------|
| anchor_manifest_check.py (pre-run) | PASS (exit 0) |
| anchor_manifest_check.py (post-run) | PASS (exit 0) |

## E3.0 -- Correct the E2.1 interpretation

E2.1 measured pairwise Pearson correlation between comp residual **time series** (common
mode c(t)). The SEM is a **within-frame** dispersion across comps; c(t) is a constant offset
per frame and contributes nothing to that average. sqrt(1+(n-1)*rho) requires correlation
among quantities in the **same** average; across-frame rho is not that quantity.

E2.2 check_pc1_corr ~ 0.003 confirms c(t) cancelled between target and ensemble as expected.
E2.2 is **not** a contradiction; it is evidence the E2.1 mechanism **cannot operate**.

**Revised:** 1.90 vs 1.83 numerical agreement is **UNEXPLAINED, not explanatory**.
**WIDE-ERR-CORRELATED-COMPS withdrawn** as causal claim. rho_bar median 0.393 retained.

## E3.1 -- Decompose comp residual variance

166 fields; comps x frames residual matrix (production comp_ref_map path).

| quantity | median | IQR (p25/p50/p75) |
|----------|--------|-------------------|
| sigma_c (common mode, mmag) | **22.0** | 16.9 / 22.0 / 28.5 |
| sigma_eps (within-frame std, mmag) | **23.7** | 14.1 / 23.7 / 37.3 |
| SEM quoted (mmag) | **9.38** | -- |
| expected sigma_eps = SEM*sqrt(n) (mmag) | **25.2** | -- |

**Reading:** sigma_eps ~ 24 mmag matches the budget identity SEM ~ sigma_eps/sqrt(n) with
SEM ~ 9.4 mmag and n ~ 8. The within-frame comp scatter is consistent with the quoted SEM;
the decomposition assumption holds for comps.

## E3.2 -- Decisive prediction

Required missing term (W1 quadrature): **15.1 mmag**.
Check star G = 8.74. Brightness-matched comps: **+/-0.3 mag gave n=0**; widened to **+/-0.5 mag**.

| cohort | n | sigma_eps median (mmag) | IQR |
|--------|---|-------------------------|-----|
| G in 8.74 +/- 0.5 | 12 | **10.5** | 9.2 / 10.5 / 13.2 |
| all comps | 1068 | **20.2** | 15.8 / 20.2 / 25.5 |

**sigma_eps vs G (comp instances):**

| G bin | n | sigma_eps median (mmag) |
|-------|---|------------------------|
| (8, 10] | 57 | 11.9 |
| (10, 11] | 117 | 15.5 |
| (11, 12] | 370 | 18.5 |
| (12, 13] | 514 | 24.0 |
| (13, 14] | 109 | 30.5 |

**Reading:** Brightness-matched sigma_eps **10.5 mmag** is **below** the 12-19 mmag band
needed to match the 15.1 mmag required term and **below** the check-star excess. The check
star is **noisier than comps at its own brightness** (10.5 vs 15.1 mmag required). At G~10-11
comps reach ~15 mmag, but those are fainter than the check star. **Not generic per-star noise
at matched brightness**; selection effect (check star chosen for stability among bright
field stars) is plausible. n=12 at +/-0.5 mag is sparse but constrained.

## E3.3 -- Does the budget have a category-(d) term?

Target err quadrature (photometry_core.py:3550, sigma_floor_core.py:64-86):

| term | category | this run |
|------|----------|----------|
| err_photon | (a) target photon | non-zero |
| err_sem_rel / ensemble_scatter | (b) ensemble ZP | non-zero |
| err_scint_rel | (c) atmosphere | non-zero (~1.8 mmag) |
| err_sigma_sys_rel / sigma_sys_mag | (d) per-star systematic | **candidate** |

**sigma_sys_mag equipment_id=1: 0.0** (config.json has floor only for equipment_id 4: 0.018 mag).
Category (d) slot exists in code but is **zero on this wide rig run**. That is the missing
term location if a floor were configured.

## E3.4 -- Consistency across magnitude range

Using brightness-matched sigma_eps = **10.5 mmag** from E3.2:

| subset | n | predicted ratio median | measured ratio median |
|--------|---|------------------------|------------------------|
| photon-dominated | 150 | **1.01** | **1.13** |
| ensemble-dominated | 13 | **2.35** | **1.63** |
| all | 163 | -- | -- |

Spearman(predicted, measured): rho = **0.516**, p = 1.8e-12, n = 163.

**Reading:** Faint photon-dominated end ~ OK (pred 1.01 vs meas 1.13). Ensemble-dominated
regime **overshoots** (pred 2.35 vs meas 1.63); does **not** reproduce both ends with a single
sigma_eps = 10.5 mmag. Hypothesis does not close the full G-dependent pattern from A2.

## Combined line

**WIDE-ERR-CHECKSTAR-SPECIAL** -- brightness-matched comp sigma_eps ~ 10 mmag, below the
15.1 mmag required for check-star excess; per-star noise exists on comps (~24 mmag at ensemble
level) but check star is not generic; sigma_sys_mag=0 on equipment_id 1 is the vacant category-(d)
slot

## Files created

- dev/results/CURSOR_RESULT_wide_err_e3.md (this file)
- dev/tools/wide_err_e3.py
- tmp/wide_err_e3/wide_err_e3.json

## Errors

None blocking.
