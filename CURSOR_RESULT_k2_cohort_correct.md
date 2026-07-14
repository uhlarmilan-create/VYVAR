CURSOR RESULT -- 2026-07-14 K2-COHORT-CORRECT

What I did
Applied frozen pre-registered rule verbatim (no re-run). Retracted initial DOWN verdict.
Added k2_eff CI / sensitivity / colour-range extraction to k2_cohort_core + report script.
Updated ROADMAP (per-rig record), STATE, JOURNAL, K2_BAND_AWARE_SPEC, CURSOR_RESULT_k2_cohort.md.
Fixed k2_priority_verdict to require each tested cell >=80% power for DOWN.

## Verdict correction + retraction

Frozen rule: "DOWN only if ALL tested (rig, band) cells are null AND each cell had >= 80% power."

Tested cells:
- wide_CLEAR: power 0.999, null (T1 rho=-0.013 q=0.877; T2 q=0.114)
- Newton_g: power 0.47, null
- Newton_i: power 0.40, null (T1 rho=-0.325 suggestive; T2 rho=+0.470 p=0.043 raw)

"Each cell >=80% power" FAILS (Newton g/i underpowered) -> verbatim outcome **UNCHANGED**
(per-cell power stated). Reading underpowered nulls as excluded from the power requirement
was post-hoc; retracted.

Retraction line (CURSOR_RESULT_k2_cohort.md): initial DOWN misapplied the frozen rule;
corrected without re-running anything.

## Per-rig record (as written to ROADMAP)

**wide (eq1, unfiltered): k'' deprioritized-by-evidence.**
n=147, power 0.999 for rho=0.4; T1 rho=-0.013 q=0.877; T2 q=0.114; sensitivity bound below.
Action: none; revisit only if a future filtered wide dataset appears.

**Newton (eq4): k'' OPEN, suggestive.**
T1 rho=-0.325 with literature-expected sign (n=19, power 0.40) AND T2 rho=+0.470 (p=0.043 raw)
-- two independent probes pointing the same way, both underpowered; consistent with SEM-CAUSE
D1.3 subset finding. Action: re-test when >=46 Newton constant stars are available (accumulates
naturally with future Newton nights; no dedicated task now).

Overall k'' priority: **UNCHANGED** (verbatim pre-registered rule).

## k2_eff CI table [mag / airmass / mag_colour]

From star-level WLS already in k2_cohort_summary.json (k2_eff +/- 1.96*k2_eff_se):

| cell | n | k2_eff | k2_eff_se | 95% CI lo | 95% CI hi | half-width |
|------|---|--------|-----------|-----------|-----------|------------|
| wide_CLEAR | 147 | -0.03996 | 1.08e-6 | -0.039963 | -0.039959 | 2.12e-6 |
| Newton_g | 23 | -0.05743 | 2.34e-5 | -0.057474 | -0.057383 | 4.59e-5 |
| Newton_i | 19 | -0.03595 | 3.13e-5 | -0.036011 | -0.035888 | 6.14e-5 |

## wide_CLEAR sensitivity bound

At 95% confidence we exclude |k2_eff| > **2.12e-6** mag/airmass/mag_colour.

Construction: two-sided WLS CI half-width = 1.96 x k2_eff_se (star-level inverse-variance
weights 1/b_X_se^2 on b_X vs signed colour offset).

## Colour-offset dynamic range (restriction-of-range context)

Tier-selected cohort (colour cap 0.79 mag). p5-p95 of signed BP-RP offsets per cell:

| cell | p5 | p95 | span | rho_T1 | power |
|------|----|-----|------|--------|-------|
| wide_CLEAR | -0.301 | +0.467 | 0.769 | -0.013 | 0.999 |
| Newton_g | -0.099 | +0.460 | 0.559 | -0.044 | 0.474 |
| Newton_i | -0.086 | +0.468 | 0.554 | -0.325 | 0.395 |

wide_CLEAR attenuation: offsets span [p5,p95]=[-0.301,+0.467] mag (span 0.769 mag). A true
|k2_eff|=0.040 would produce max b_X slope spread ~0.031 mag/airmass across that range, versus
measured per-star b_X scatter SD 0.094 mag/airmass.

## Spec annotation

docs/K2_BAND_AWARE_SPEC.md updated: CIs per cell; wide seed marked "consistent with zero,
bound |k2_eff| < 2.12e-6"; overall verdict UNCHANGED; per-rig notes.

## Push hash + baseline PASS

Pushed `be930ea..c5b6d3d` to origin/main (Milan-authorized).
HEAD `c5b6d3d`; session_baseline_check --fast PASS (826 passed, 15 skipped).

## Errors (if any)

None.

## Files changed

k2_cohort_core.py (verdict fix, extract_cell_report_stats, k2_eff_ci95)
scripts/k2_cohort_run.py (report_stats in payload)
tests/test_k2_cohort_core.py
docs/VYVAR_ROADMAP.md
docs/VYVAR_STATE.md
docs/VYVAR_JOURNAL.md
docs/K2_BAND_AWARE_SPEC.md
CURSOR_RESULT_k2_cohort.md
CURSOR_RESULT_k2_cohort_correct.md

## pytest count

826 passed, 15 skipped (session_baseline_check --fast on c5b6d3d).
