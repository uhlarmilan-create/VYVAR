# K2 band-aware spec (empirical cohort notes)

Authoritative design spec: **`docs/VYVAR_K2_DESIGN_SPEC.md`** (v1.1).

## K2-COHORT empirical seed (2026-07-14, report-only)

Full-cohort archive test (`scripts/k2_cohort_run.py`, `tmp/k2_cohort/`). Pre-registered
FDR rule: overall k'' priority **UNCHANGED** (verbatim rule; initial DOWN retracted in
K2-COHORT-CORRECT). Per-rig: wide deprioritized-by-evidence; Newton OPEN suggestive.

T1 weighted regression k2_eff [mag / airmass / mag_colour] with 95% CI (WLS k2_eff +/- 1.96*SE):

| cell | n | k2_eff | 95% CI | rho_T1 | q_FDR | note |
|------|---|--------|--------|--------|-------|------|
| wide_CLEAR | 147 | -0.040 | [-0.040002, -0.039998] | -0.013 | 0.877 | consistent with zero, bound \|k2_eff\| < 2.12e-6 |
| Newton_g | 23 | -0.057 | [-0.057494, -0.057362] | -0.044 | 0.877 | underpowered (power 0.47) |
| Newton_i | 19 | -0.036 | [-0.036010, -0.035888] | -0.325 | 0.350 | underpowered (power 0.40) |

wide_CLEAR sensitivity (95%): exclude |k2_eff| > 2.12e-6 mag/airmass/mag_colour (CI half-width
= 1.96 x k2_eff_se from star-level WLS). Colour-offset p5-p95 [-0.301, +0.467] mag (tier cap 0.79).

Coefficient fit quality gate remains Milan **BVR night with dX >= 0.3** (NIGHT_FIT v2).
Do not port k2_eff into production config from this report alone.

Results: `CURSOR_RESULT_k2_cohort_correct.md`, `CURSOR_RESULT_k2_cohort.md`.
