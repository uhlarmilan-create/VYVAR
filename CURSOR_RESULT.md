CURSOR RESULT -- 2026-07-14 K2-STATS-FIX

What I did
Fixed overdispersion-honest k2_eff CIs (bootstrap authoritative), wide re-verdict as LOW
PRIORITY subdominance, process record, internal-consistency guard. Pushed 036a570..c8aed4b.

## Output / findings
Bootstrap wide CI [-0.076,+0.046], B=0.076; plausible k'' NOT excluded. Naive WLS CIs retracted.
HEAD c8aed4b; baseline PASS 830 passed.

## Errors (if any)
None.

## Files changed
k2_cohort_core.py, scripts/k2_cohort_run.py, tests/test_k2_cohort_core.py,
docs (ROADMAP, STATE, JOURNAL, K2_BAND_AWARE_SPEC), CURSOR_RESULT_k2_*.md
Commits: 4e91c7a, 3c9b5f1, c8aed4b
