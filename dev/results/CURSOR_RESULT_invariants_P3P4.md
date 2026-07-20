# CURSOR RESULT - VYVAR-INVARIANTS P3 + P4

CURSOR RESULT - 2026-07-20 (local; not pushed)

Docs impact: PROCESS (recurrence + forensic + weekly cadence); ROADMAP P3+P4
DONE; STATE honest-scope section; DECISIONS INVARIANTS-P3P4-CLOSEOUT;
FLOW_DOC_V3_GAPS D2 void addendum; FLOW/facts: none.

Recurrence: n/a (first occurrence / not a bug-class) -- process discipline
install + forensic pilot promotion (new asserts inside existing
test_ui_chain_byte_identity, not a second bug-class fix).

## Verdict

Two commits (not pushed):
1. invariants(p3): recurrence + forensic-promotion discipline, weekly report
   tool, pilot triage of forensic_disc_ui_match2
2. invariants(p4): STATE honest-scope; GAPS D2 void; P1-P4 closeout

## P3 pilot triage (forensic_disc_ui_match2.py)

- Purpose: F3 discriminator, NightRun UI-parity match radius
  cat_match_arc=2.0 + config SysRem.
- VERIFY: P1 UI-chain helper did NOT pin cat_match_arc=2.0.
- Outcome: **PROMOTE+ARCHIVE**
  - PROMOTE: assertion into `test_ui_chain_byte_identity` (app.py source
    contains `cat_match_arc=2.0`; NightRunParams default == 2.0).
  - ARCHIVE: mandatory header on `dev/scripts/forensic_disc_ui_match2.py`
    (state ARCHIVED -- no live role).
- P1 golden after promote: 7 passed.

## invariants_report.py output (pasted)

```
# VYVAR weekly invariants report

Generated: 2026-07-20 12:26 UTC

## Registry (`docs/VYVAR_INVARIANTS.md`)

- by policy: FAIL=9, OTHER=1, WARN=1
- by enforcement: registry-only=3, wired=8
- wired IDs (8): INV-FLUX-01, INV-FLUX-02, INV-FLAT-01, INV-WCS-01, INV-DAG-01, INV-RNG-01, INV-PROV-01, INV-CFG-01

## Guards (cheap pytest)

- `dev/tests/test_docs_sync_guard.py`: **PASS** (4 passed in 0.06s)
- `dev/tests/test_ascii_policy.py`: **PASS** (3 passed in 0.45s)
- `dev/tests/test_invariants_p2.py`: **PASS** (13 passed in 1.94s)

## Ledger

- ledger version=1 updated=2026-07-20
- items total=15 active_passes_true~=13
- latest stamp: VL-COUNTERS-ZERO last_verified=2026-07-20 commit=69432ee
- ACTIVE (passes=true, not superseded): F-435-EXPORT-GHOSTS, VL-AAVSO-EXPORT, VL-ANCHOR-WCSINV, VL-CALDIAG-424, VL-COUNTERS-ZERO, VL-DETERMINISM-425, VL-K2-MATRIX, VL-P1-GOLD, VL-PROVENANCE, VL-PYTEST-FULL, VL-SHA-CUT1, VL-TRUST-BASELINE, VL-XVAL-V0612

## Runtime WARN/FAIL sweep (Archive drafts)

- scanned_meta_files=0 (of 4 found)
- WARN events=0 FAIL events=0
- by_id: (none)

## P1 golden pointer (opt-in; do not auto-run)

```
set VYVAR_INVARIANTS_P1=1 && pytest dev/tests/test_invariants_p1_seed.py dev/tests/test_invariants_p1_golden.py -q
```

P1 is ~10 min; run when locking a golden or after P1-touching changes.
```

Exit code: 0 (guard subprocesses all PASS).

## P3 gates

| Gate | Result |
|------|--------|
| ruff invariants_report.py | All checks passed |
| invariants_report.py | exit 0 |
| P1 golden | 7 passed |
| --fast | OVERALL PASS (1034 passed, 17 skipped) |

## P4 honest-scope verification notes

(filled in commit 2)

## P4 gates

(filled in commit 2)
