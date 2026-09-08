CURSOR RESULT - 2026-09-08 LEDGER-T4-RELAND-PLAN-01

Date: 2026-09-08. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 1f45370.
Class: DOCS-ONLY.

## Premise (Rule 0.1)

Compared: the RED-TARGET-T4-RELAND row / D-RED-TARGET-T4-01
close as left by RED-TARGET-T4-01 C (`1f45370`: "needs an
anchor plan / Milan GO") versus Milan's 2026-09-08 choice of
reland option (b). They differ: (b) binds reland of `817f1f9`
to the planned 520 era snapshot re-cut (one re-cut, two
reasons), rejects a scoped G2 exception, and requires a
CV CVn + HAT-188-0002048 RMS-first vs color-first LC-quality
micro-measurement. This task records that; it does not reland
code.

## What I did

Updated the ROADMAP row and appended one dated line inside
D-RED-TARGET-T4-01. Ran `--fast --clean`. Committed and
pushed `origin consolidate-01:consolidate-01` only.

## Output / findings

- ROADMAP **RED-TARGET-T4-RELAND** blocked-on: 520 era snapshot.
  Scoped G2 exception rejected. Micro-measurement required at
  reland: does color-first help CV CVn + HAT-188-0002048, not
  merely change them.
- D-RED-TARGET-T4-01: reland plan line added (no new heading).

## Errors (if any)

None.

## Files changed

- `docs/VYVAR_ROADMAP.md`
- `docs/VYVAR_DECISIONS.md`
- `dev/results/CURSOR_RESULT_LEDGER_T4_RELAND_PLAN.md`
