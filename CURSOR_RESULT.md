CURSOR RESULT — 2026-06-11 (math/physics audit hygiene)

What I did
Landed byte-identity-neutral math/physics audit fixes: filed `VYVAR_MATH_PHYS_AUDIT.md`,
scoped Broeg vs AIJ/Honeycutt citations, EXPTIME warning guard + test, MAD constant unify,
citation-integrity notes (Mighell export-only), ROADMAP parked deep physics.

## Output / findings

**Photometry SHA (draft_387):** core `203254fd…` (2806) / full `95a5515a…` (4285) — **unchanged**.

**Tests:** 261 passed, 14 skipped (+2 `test_time_utils_mid_exposure`).

**D2 callers:** production uses `mid_exposure_jd` via `compute_time_columns`; only
`scripts/cross_validate_draft342.py` duplicates logic (non-production).

**Citation integrity:** `mighell1999` marked aspirational/export-only; `collins2017` +
`honeycutt1992` added and wired in `citations.py` core + VarAstro summary.

## Errors (if any)
None.

## Files changed
See commit (docs + citations + time_utils + comp_qa_core comment + tests).
