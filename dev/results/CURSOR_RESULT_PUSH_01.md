CURSOR RESULT - PUSH-01 - 2026-08-14

What I did
Committed the SNR-GATE-03 memo record on top of cb9e695, ran `--fast` on the
new tip, pushed `origin/main` (Milan authorized), re-ran `--fast` after push.

## Documentation commit

- SHA: **8fe1759**
- Parent science tip: **cb9e695** (SNR-GATE-01; verified earlier)
- Files: `dev/results/CURSOR_RESULT_SNR_GATE_01.md`,
  `dev/results/CURSOR_RESULT_SNR_GATE_03.md` only
- `--fast` @ 8fe1759 (pre-push): **OVERALL PASS** (1341 passed, 27 skipped;
  git-head PASS 8fe1759)

## Push

```
git push origin main
# 4a3e855..8fe1759  main -> main
```

Pushed tip: **8fe1759**
`origin/main`: **8fe1759** (matches HEAD)

`git log --oneline -6` on origin/main:

```
8fe1759 docs: record SNR-GATE-03 --fast verification of tip cb9e695.
cb9e695 SNR-GATE-01: sky-MAD prematch noise; pass-2 exempt from global peak gate.
ce404f5 docs: CLOSE-AND-PUSH memo with --fast SHA table for commits 1-3.
3fd4566 Gates, PP-KWARG, COG measurement, register; no photometry numeric change.
77d082a SKY-CLIP-01: plain annulus median, one estimator, three call sites.
3791b6c docs: ASCII-repair Wave 7 files and stop mapping U+FFFD in CHAR_MAP.
```

## Post-push `--fast` @ 8fe1759

```
SESSION BASELINE CHECK (fast)
------------------------------------------------------------------------
Check                        Status Detail
------------------------------------------------------------------------
git-branch                   PASS   main
git-head                     PASS   8fe1759
git-staged                   PASS   none
git-untracked-known          WARN   1 known untracked
git-untracked                WARN   DAO_DEPTH_01_* and other known scratch
config-paths                 PASS   all present
pytest                       PASS   1341 passed, 27 skipped
manifest-db-parity           PASS   draft_id=435
ledger                       PASS   v1 15 items
ledger-todo                  WARN   VL-ANCHOR-424, VL-ANCHOR-DQ-430
deps-outdated                WARN   numpy 2.4.4->2.5.2 (+96 other)
------------------------------------------------------------------------
OVERALL: PASS
```

`git-origin-main` no longer reports a difference (line absent; HEAD == origin/main).

## Working tree

Clean of staged/modified tracked files. Known untracked scratch left as-is:
DAO_DEPTH_01_*, wide_err_*, `_tmp_batch_e_lc/`, `src_py/tmp/`, sqlite shm/wal.
This PUSH-01 memo is new untracked until a later docs commit if desired.

## Section 4 -- what the push invalidates (follow-up; not fixed here)

| Item | Status |
|------|--------|
| `--full` anchor and P1 golden ledger SHA stale (invalidated by SKY-CLIP-01 and again by SNR-GATE-01) | Deferred / open; ledger-todo still warns VL-ANCHOR-424, VL-ANCHOR-DQ-430 |
| Draft 510 and draft 512 checksum manifests no longer describe current-code products | Deferred; manifests are superseded by subsequent science commits |
| Draft 512 stored products (dirty tree + broken prematch gate) superseded, not a reference | Deferred; rebuild required for "after" LC numbers |
| Every draft built since `c9e1f8f` carries shallow MASTERSTAR depth | Deferred; addressed in code by SNR-GATE-01, not yet by reprocessing those drafts |
| SNR-DEPTH-01 (G15 repeatability depth limit; no cut implemented) | OPEN / DEFERRED on register |

None of these is fixed by this push.

## Errors (if any)
None.

## Files changed
- Commit 8fe1759: SNR-GATE memo + SNR-GATE-03 result
- This memo: `dev/results/CURSOR_RESULT_PUSH_01.md` (untracked)
