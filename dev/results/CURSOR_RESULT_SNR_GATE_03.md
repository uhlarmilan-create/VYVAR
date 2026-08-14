CURSOR RESULT - SNR-GATE-03 - 2026-08-14

What I did
Ran `session_baseline_check.py --fast` on the actual tip `cb9e695`.
Updated the SNR-GATE-01 memo on the working tree only (no amend, no push).

## Output / findings

```
SESSION BASELINE CHECK (fast)
------------------------------------------------------------------------
Check                        Status Detail
------------------------------------------------------------------------
git-branch                   PASS   main
git-head                     PASS   cb9e695
git-staged                   PASS   none
git-untracked-known          WARN   1 known untracked
git-untracked                WARN   DAO_DEPTH_01_* and other known scratch
git-origin-main              WARN   differs from origin/main (4a3e855)
config-paths                 PASS   all present
pytest                       PASS   1341 passed, 27 skipped
manifest-db-parity           PASS   draft_id=435
ledger                       PASS   v1 15 items
ledger-todo                  WARN   VL-ANCHOR-424, VL-ANCHOR-DQ-430
deps-outdated                WARN   numpy 2.4.4->2.5.2 (+96 other)
------------------------------------------------------------------------
OVERALL: PASS
```

- `git log --oneline -1`: `cb9e695 SNR-GATE-01: sky-MAD prematch noise; pass-2 exempt from global peak gate.`
- Verified tip: **cb9e695**
- Memo records that tip; memo edit left unstaged (no amend).

## Errors (if any)
None.

## Files changed
- Working tree only: `dev/results/CURSOR_RESULT_SNR_GATE_01.md` (unstaged)
- This file: `dev/results/CURSOR_RESULT_SNR_GATE_03.md` (untracked)
- Tip commit unchanged: `cb9e695`
