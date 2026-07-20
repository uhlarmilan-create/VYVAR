CURSOR RESULT - 2026-07-16 (VYVAR-INVARIANTS intake)

What I did
Recorded the program premise and queued phases. Did **not** start P1-P4 implementation.

## Gate status

**Anchor #3 is not closed.** STATE still: T3 FIX A+B on `89842ff`; Milan HEALTHY UI
validation pending; protocol-v2 snapshot not cut.

Task timing clause: *starts AFTER anchor #3 closes.* Honored - no golden dataset build,
no `VYVAR_INVARIANTS.md` yet, no PROCESS amendments beyond ledger queue.

## Output / findings

### Premise (DECISIONS - on record)

Static audits cannot catch integration-class defects. F-428/F-431 defects were caught by
measurement. Goal: no silent science defect; no recurrence of ledger defect classes. Not
"zero bugs."

### Queued phases (ROADMAP)

| Phase | First action when unblocked |
|-------|----------------------------|
| P1 | Golden ~5-8 frame BO CVn crop + slow UI<->night_run equivalence pytest |
| P2 | `docs/VYVAR_INVARIANTS.md` + runtime gates |
| P3 | PROCESS recurrence / forensic promotion / weekly report |
| P4 | STATE honest scope statement |

### Files changed (ledger only)

- `docs/VYVAR_DECISIONS.md` - VYVAR-INVARIANTS premise
- `docs/VYVAR_STATE.md` - queued note
- `docs/VYVAR_ROADMAP.md` - QUEUED section
- `docs/VYVAR_JOURNAL.md` - intake line
- `CURSOR_RESULT_invariants_queued.md`

## Errors (if any)

None. Blocked on Milan: post-`89842ff` clean UI RUN VYVAR -> HEALTHY? -> Anchor #3 -> then P1.

## Unblock checklist (Milan)

1. Pull `89842ff`; clean UI RUN VYVAR (A-durable watch during alignment).
2. Report draft id + HEALTHY / not (expected: cal!=proc, ~2875-class, sky ~1478, p95 < 1 px,
   `git_dirty_code=false`).
3. On HEALTHY: Cursor runs Anchor #3 protocol-v2, then starts INVARIANTS P1.
