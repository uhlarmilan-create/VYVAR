CURSOR RESULT -- 2026-06-09

What I did
Full doc sweep: reconciled STATE + 6 core docs to 2026-06-09 reflecting the PSF publication-grade
arc, EPSF-1, NEIGHBOR-SUB state, and #4 fail-safety work.

## Output / findings

- **VYVAR_STATE.md** -- restored clean snapshot + index (session blocks moved to JOURNAL); date
  2026-06-09; tests 224/6; PSF / NEIGHBOR-SUB / #4 status blocks added.
- **VYVAR_JOURNAL.md** -- master 2026-06-09 arc entry; archived 2026-06-03/04 blocks from STATE.
- **VYVAR_ROADMAP.md** -- NEXT SESSION refreshed; closed rule-2 bug, realistic PSF uncertainties,
  TODO-GEO; reconciled open questions.
- **VYVAR_DECISIONS.md** -- sandwich uncertainty, EPSF-1, updated PSF/NEIGHBOR-SUB status.
- **VYVAR_PROCESS.md** -- harness 2-3 attempt rule; Brno gate confirmed.
- **VYVAR_PARAMS.md** -- PSF production constants + NEIGHBOR-SUB keys; count updated.
- **VYVAR_VALIDATION.md** -- A9/V3d/V3e matrix + proof CLIs; pytest/SHA refs.

Docs reconciled to 2026-06-09.

## Regression

- Numeric SHA `770966c3` unchanged (docs-only pass).
- pytest `tests/`: 224 passed / 6 skipped (unchanged from prior run).

## Errors (if any)

None.

## Files changed

- `docs/VYVAR_STATE.md`
- `docs/VYVAR_JOURNAL.md`
- `docs/VYVAR_ROADMAP.md`
- `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_PROCESS.md`
- `docs/VYVAR_PARAMS.md`
- `docs/VYVAR_VALIDATION.md`
- `CURSOR_RESULT.md`
