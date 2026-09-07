CURSOR RESULT - 2026-09-07 C-EXPORT-GAP-CLOSE-01

Date: 2026-09-07. Architect: Claude. Implementer: Cursor.
Branch: consolidate-01. Base: 59c69d2.
Class: DOCS-ONLY. No production code change. No science change.
Milan GO 2026-09-07.

What I did
Closed C-EXPORT-GAP as CLOSED-STALE from the VERIFY-01 measurement
under the documented eclipsing->VarAstro / all->AAVSO routing
policy. Pointed the 3 empty-point export failures at the existing
open row R-CVN-EMPTY-COMP. Register status updated; historical
claim text left as the audit wrote it.

Premise (Rule 0.1): VERIFY-01 counted 50 AAVSO + 10 VarAstro txt
against 53 aperture LC CSVs and treated VarAstro-for-every-LC as
H1. The governing routing artifact is DECISIONS "Reporting &
export / AAVSO / VarAstro correctness" and JOURNAL "ROUTING
(confirmed intended)": eclipsing -> VarAstro only. Under that
policy 50 = 53-3 and 10/10 eclipsing is complete. Compared those
two frames before closing.

## Refute check (no stop)

- Routing policy: docs/VYVAR_DECISIONS.md:3157-3163
  ("Routing: eclipsing -> VarAstro (LC); pulsating/all -> AAVSO.")
  and docs/VYVAR_JOURNAL.md:3637 ("ROUTING (confirmed intended)").
- Empty-point loud failures: export_reports.py:997-1004
  (`lc_normal.empty` -> record_export_failure
  "no exportable LC points (flags/mag empty)" -> return {}).
- VERIFY-01: n_lc=53, n_aavso=50, n_varastro txt=10, G4 PASS;
  ids 1498842882207281152, 1499842372636900992,
  1500410236033012352.
- R-CVN-EMPTY-COMP was OPEN at ROADMAP (pre-edit) line 93.

No refutation.

## Output / findings

OPEN table: C-EXPORT-GAP row removed.
CLOSED this arc: C-EXPORT-GAP CLOSED-STALE 2026-09-07 line added.
R-CVN-EMPTY-COMP one-line state updated; owner Cursor, blocked-on
POST-453 unchanged.
Register C-EXPORT-GAP status CLOSED-STALE with pointer
"measured 2026-09-07; CURSOR_RESULT_C_EXPORT_GAP_VERIFY_01.md";
evidence column still the original audit claim.

## Architect error ledger

14. C-EXPORT-GAP-VERIFY-01 hypothesis H1 pre-registered "VarAstro for
    every LC target" without reading the documented routing decision
    (eclipsing -> VarAstro only; DECISIONS "Reporting & export",
    JOURNAL "ROUTING (confirmed intended)"). Same class as errors
    9-13: claim from derivation instead of the governing artifact.
    Caught by the pre-registered R2 rule; nothing was wrongly closed.

## Errors (if any)

None.

## Files changed

docs/VYVAR_ROADMAP.md
docs/VYVAR_AUDIT_2026_REGISTER.md
dev/results/CURSOR_RESULT_C_EXPORT_GAP_CLOSE_01.md
dev/results/CURSOR_RESULT_C_EXPORT_GAP_VERIFY_01.md
dev/results/context/session_20260907_cexportgap/ (measurement
evidence; Rule 0.2)

## Gates

`--fast --clean` OVERALL PASS (1621 passed, 32 skipped; clean-tree
PASS). HEAD at gate: 59c69d2. No `--full`, no `--full-epsf`.
Push only `git push origin consolidate-01:consolidate-01`.
