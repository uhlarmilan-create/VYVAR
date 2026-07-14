CURSOR RESULT -- 2026-07-14 K2-STATS-FIX

What I did
Recomputed k2_eff uncertainties from tmp/k2_cohort/ star-level data (no new photometry).
Added overdispersion-scaled WLS, bootstrap CI (authoritative), internal-consistency warnings,
and attainable-rho table. Re-verdict wide as LOW PRIORITY subdominance. Updated docs; retracted
invalid naive-WLS CIs in prior result MDs.

## Corrected CI table (naive vs scaled vs bootstrap, chi2_red)

Bootstrap 95% CI is AUTHORITATIVE (2000 star resamples). Scaled WLS: se_scaled = se_naive * sqrt(chi2_red).

| cell | n | k2_eff | se_naive | chi2_red | se_scaled | scaled 95% CI | bootstrap 95% CI |
|------|---|--------|----------|----------|-----------|---------------|------------------|
| wide_CLEAR | 147 | -0.040 | 1.08e-6 | 131405 | 3.92e-4 | [-0.0407, -0.0392] | [-0.076, +0.046] |
| Newton_g | 23 | -0.057 | 2.34e-5 | 13495 | 2.72e-3 | [-0.0628, -0.0521] | [-0.186, +0.071] |
| Newton_i | 19 | -0.036 | 3.13e-5 | 4883 | 2.19e-3 | [-0.0402, -0.0317] | [-0.338, +0.029] |

Consistency: scaled-WLS and bootstrap intervals overlap on all cells (bootstrap wider; expected).
Internal-consistency warnings fired on wide_CLEAR and Newton_g (|k2_eff/se_naive| >> 5, |rho| < 0.1).

## Sensitivity bounds + physical-context lines

Bound B = max(|bootstrap CI_lo|, |bootstrap CI_hi|) when CI contains zero.

| cell | B | contains zero? | physical context |
|------|---|----------------|------------------|
| wide_CLEAR | 0.076 | yes | Literature k'' ~0.02-0.04 NOT excluded (B > 0.04). Null uninformative in exclusion terms. |
| Newton_g | 0.186 | yes | Plausible k'' NOT excluded. |
| Newton_i | 0.338 | yes | Plausible k'' NOT excluded. |

wide physical-context: at 95% we exclude |k2_eff| > 0.076 mag/airmass/mag_colour, but
literature-scale |k2_eff| = 0.02-0.04 lies well inside this bound -- the wide null does not
rule out physically plausible k''. Subdominance argument (slope spread <=0.031 vs b_X scatter
0.094) is the defensible basis for LOW PRIORITY, not exclusion.

## Attainable-rho table

rho_expected ~ |k2| * sigma_colour / sigma_bX (measured per cell).

| cell | sigma_colour | sigma_bX | rho@0.02 | rho@0.04 | rho@0.08 |
|------|--------------|----------|----------|----------|----------|
| wide_CLEAR | 0.267 | 0.092 | 0.058 | 0.116 | 0.232 |
| Newton_g | 0.194 | 0.059 | 0.066 | 0.131 | 0.262 |
| Newton_i | 0.182 | 0.073 | 0.050 | 0.100 | 0.199 |

At |k2|=0.04, expected rho ~ 0.10-0.13 -- below UP gate (0.3) and below power basis rho=0.4.
Pre-registered "power 0.999" for wide certified detectability of a correlation a plausible k''
could not produce.

## Re-verdict text (as written to ROADMAP/spec)

**wide (eq1): k'' correction LOW PRIORITY -- subdominance argument.** Colour-driven slope
variance bounded to <= ~0.031 mag/airmass spread across cohort colour range, versus total
per-star b_X scatter 0.094 -- even literature-maximal k'' would be subdominant; leading
slope-noise source is something else (unidentified; consistent with PZQ correlated-noise).
Plausible k'' magnitudes NOT excluded (B=0.076). Revisit if dominant slope-noise removed or
filtered wide dataset appears.

Newton: OPEN, suggestive (unchanged). Overall k'' priority: UNCHANGED.

Coherent-sign note: all three cells k2_eff ~ -0.04 to -0.06, same sign; honest bootstrap CIs
all contain zero -- recorded, not over-read.

## Process record + consistency-check test

JOURNAL lesson #2: power/effect-size bases must use attainable rho from physical k'' and measured
noise model. Guard added: check_k2_internal_consistency() warns when |k2_eff/se_naive| > 5 and
|rho| < 0.1 (tested). Emitted in report_warnings payload.

Retraction lines added to CURSOR_RESULT_k2_cohort.md and CURSOR_RESULT_k2_cohort_correct.md for
invalid CI table and 2.12e-6 sensitivity bound.

## Push hash + baseline PASS

(pending)

## Errors (if any)

None.

## Files changed

k2_cohort_core.py
scripts/k2_cohort_run.py
tests/test_k2_cohort_core.py
docs/VYVAR_ROADMAP.md, docs/VYVAR_STATE.md, docs/VYVAR_JOURNAL.md, docs/K2_BAND_AWARE_SPEC.md
CURSOR_RESULT_k2_cohort.md, CURSOR_RESULT_k2_cohort_correct.md
CURSOR_RESULT_k2_stats_fix.md

## pytest count

(pending)
