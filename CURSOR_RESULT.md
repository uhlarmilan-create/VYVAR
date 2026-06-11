CURSOR RESULT — 2026-06-11 (session wrap-up)

What I did
Updated STATE/JOURNAL/ROADMAP/DECISIONS/PARAMS/RUNBOOK/PROCESS/AUDIT docs for the full
trust/anchor/reliability session. Re-trust draft_387 → **1382 YELLOW / 106 RED** (floor-5
baseline). Verified pytest + ruff + photometry SHA. Single commit prepared (push on Milan's go).

## Output / findings

**Disciplines:**
- `pytest tests` — 259 passed, 14 skipped
- `ruff check . --select BLE001,E722` — clean
- draft_387 SHA — core `203254fd...` (2806) / full `95a5515a...` (4285) unchanged
- Trust on draft_387 — 1382/106 (post `comp_trust_min_comps=5`)

**Docs updated:** STATE, JOURNAL, ROADMAP, DECISIONS, PARAMS, PROCESS, AUDIT_LEDGER,
AUDIT_FINDINGS, CHIANDH runbook, new `VYVAR_RUNBOOK.md`; trust/check-star/comp-floor specs
confirmed under `docs/`.

## Errors (if any)
None.

## Files changed
Session code + docs (see git commit).
