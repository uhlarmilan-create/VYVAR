# K2 band-aware spec (empirical cohort notes)

Authoritative design spec: **`dev/results/specs/VYVAR_K2_DESIGN_SPEC.md`** (v1.1).

## K2-COHORT empirical seed (2026-07-14, report-only)

Full-cohort archive test (`scripts/k2_cohort_run.py`, `tmp/k2_cohort/`). Pre-registered
FDR rule: overall k'' priority **UNCHANGED** (verbatim rule; initial DOWN retracted in
K2-COHORT-CORRECT). Per-rig: wide **LOW PRIORITY (subdominance)**; Newton **OPEN suggestive**.

**Retraction (K2-STATS-FIX):** naive photon-weight WLS CIs and bound |k2_eff| < 2.12e-6
superseded -- internally inconsistent with rho=-0.013. Bootstrap CIs below are authoritative.

T1 k2_eff [mag / airmass / mag_colour] with honest uncertainties (2000-draw star bootstrap):

| cell | n | k2_eff | bootstrap 95% CI | rho_T1 | q_FDR | note |
|------|---|--------|------------------|--------|-------|------|
| wide_CLEAR | 147 | -0.040 | [-0.076, +0.046] | -0.013 | 0.877 | LOW PRIORITY subdominance; B=0.076 |
| Newton_g | 23 | -0.057 | [-0.186, +0.071] | -0.044 | 0.877 | underpowered; OPEN suggestive |
| Newton_i | 19 | -0.036 | [-0.338, +0.029] | -0.325 | 0.350 | underpowered; OPEN suggestive |

**Coherent-sign note:** all three cells k2_eff ~ -0.04 to -0.06 (same sign); bootstrap CIs
all contain zero and are mutually consistent -- recorded, not over-read.

wide (eq1) **LOW PRIORITY -- subdominance:** colour-driven slope variance bounded to <= ~0.031
mag/airmass spread across cohort colour range vs total b_X scatter 0.094; literature-maximal k''
would be subdominant. Plausible k'' (0.02-0.04) **NOT excluded** (B=0.076 > 0.04). Revisit if
dominant slope-noise source removed or filtered wide dataset appears.

Attainable rho (rho ~ |k2| * sigma_colour / sigma_bX): wide at |k2|=0.04 -> rho~0.12 (below
UP gate 0.3; power basis rho=0.4 was not physically grounded for this cohort).

Coefficient fit quality gate remains Milan **BVR night with dX >= 0.3** (NIGHT_FIT v2).
Do not port k2_eff into production config from this report alone.

Results: `CURSOR_RESULT_k2_stats_fix.md`, `CURSOR_RESULT_k2_cohort_correct.md`.
