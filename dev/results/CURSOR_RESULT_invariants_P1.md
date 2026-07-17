CURSOR RESULT — 2026-07-16 — INVARIANTS P1 (seed started)

What I did
Anchor #3 unblocked P1. Seeded snapshot-bound golden checks; full dual-entry golden crop
still open.

## Status: PARTIAL

| Deliverable | Status |
|-------------|--------|
| Snapshot SHA registered (core/ext) | DONE — `tests/test_invariants_p1_seed.py` |
| Census / identity / sky-surface asserts | DONE (seed) |
| Golden mini-dataset crop (5–8 frames) | OPEN |
| UI ↔ night_run byte-equivalence pytest | OPEN |
| Double-photometry determinism pytest (full) | Covered by LABBE-DET L3/L4 evidence; wire as always-on slow test later |

Enable seed: `VYVAR_INVARIANTS_P1=1 pytest tests/test_invariants_p1_seed.py -q`

## Files

- `tests/test_invariants_p1_seed.py`
- ROADMAP: INVARIANTS status → UNBLOCKED

## Next session

Build BO CVn crop under `tests/fixtures/` or `validation/golden_435/` + dual-entry harness.
