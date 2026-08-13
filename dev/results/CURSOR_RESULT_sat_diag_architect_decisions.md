CURSOR RESULT - 2026-08-13

What I did
Integrated Milan's architect recommendations (2026-08-13) for the four SAT-DIAG
open decisions and the separate CAL-DIAG / INV-GATE-REMOVAL finding into project
documentation. No code changes. No commit.

## Output / findings

### Governing principle filed
"The strength of the action must follow the provenance of the number" -- measured
limits may exclude; derived/default limits warn only.

### Four decisions (recommendations, not yet authorized)

| # | Topic | Recommendation |
|---|-------|----------------|
| 1 | Limit source + CONFLICT | Header -> equipment -> derived -> BITPIX; compatibility test refutes stated ceiling below max raw pixel; CONFLICT_DERIVED + loud log, continue |
| 2 | Target structure | Two levels keyed by (equipment, readmode, XBIN, YBIN); migrate SATURATE_ADU=16384 to null |
| 3 | Exposure ramp | DEFAULT_FRAC=0.85 + WARN now; AAVSO ramp later per rig config |
| 4 | Consumer policies | Three tiers; exclusion once per draft (INV-COMP-MEMBERSHIP); Tier 3 warn only |

### Separate finding
INV-GATE-REMOVAL recommended: gates cannot be removed on byte-identity alone.
CAL-DIAG reinstatement is a separate OPEN decision for Milan (not bundled with SAT-DIAG).

## Errors (if any)
None.

## Files changed

- `docs/VYVAR_SAT_DIAG_SPEC.md` -- full rewrite integrating section 3 architect
  decisions; sections 5-14 aligned (compatibility test, three-tier policy, INV-GATE-REMOVAL)
- `docs/VYVAR_DECISIONS.md` -- SAT-DIAG updated to four decisions + CAL-DIAG reinstatement OPEN
- `docs/VYVAR_INVARIANTS.md` -- INV-GATE-REMOVAL added; INV-SAT-01 Tier 3 clause
- `dev/results/CURSOR_RESULT_sat_diag_architect_decisions.md` -- this file
