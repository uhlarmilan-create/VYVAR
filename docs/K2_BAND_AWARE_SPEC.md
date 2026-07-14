# K2 band-aware spec (empirical cohort notes)

Authoritative design spec: **`docs/VYVAR_K2_DESIGN_SPEC.md`** (v1.1).

## K2-COHORT empirical seed (2026-07-14, report-only)

Full-cohort archive test (`scripts/k2_cohort_run.py`, `tmp/k2_cohort/`). Pre-registered
FDR rule yielded **k'' priority DOWN** (wide_CLEAR null with adequate power; Newton g/i
underpowered).

T1 weighted regression k2_eff [mag / airmass / mag_colour] (not significant; seeds only):

| cell | n | k2_eff | rho_T1 | q_FDR |
|------|---|--------|--------|-------|
| wide_CLEAR | 147 | -0.040 | -0.013 | 0.877 |
| Newton_g | 23 | -0.057 | -0.044 | 0.877 |
| Newton_i | 19 | -0.036 | -0.325 | 0.350 |

Coefficient fit quality gate remains Milan **BVR night with dX >= 0.3** (NIGHT_FIT v2).
Do not port k2_eff into production config from this report alone.

Result: `CURSOR_RESULT_k2_cohort.md`.
