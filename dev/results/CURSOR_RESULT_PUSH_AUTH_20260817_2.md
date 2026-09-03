# CURSOR RESULT - PUSH-AUTH 2026-08-17 #2 (GAIN-PT-RADIUS-01 stack)

Date: 2026-08-17
Authorization: Milan authorized push of the GAIN-PT-RADIUS-01 stack.
Scope: three local commits 057ecdc, a60b8c1, dde6ce0 on top of 6b23633.
Sync only. No science change in this task.

Premise (0.1): local main was three commits ahead of origin/main at
6b23633. Compared with origin/main after fetch. Hashes were not
rewritten.

## 1. Inventory (before rebase)

`git log --oneline origin/main..HEAD` (exactly three):

```
dde6ce0 docs: stamp GAIN-PT tip a60b8c1 and --fast PASS.
a60b8c1 docs+results: GAIN-PT-RADIUS-01, SUBMIT-01 ready, U-09 closed.
057ecdc fix(gain): pin photon-transfer aperture at 4.0 px (GAIN-PT-RADIUS-01).
```

Base: 6b23633 NET-TEST-01. Branch ahead of origin/main by 3.

Working tree (not product, not code):

- modified (intentional leftovers):
  `dev/results/CURSOR_RESULT_WIDE_ERR_04.md`
  `dev/results/WIDE_ERR_04_summary.json`
- modified (prior PUSH-AUTH origin-verify append, same class as
  leftovers; not product/code; left uncommitted):
  `dev/results/CURSOR_RESULT_PUSH_AUTH_20260817.md`
- untracked scratch:
  `dev/tests/_tmp_batch_e_lc/`
  `src_py/tmp/`
  `vyvar.sqlite3-shm`
  `vyvar.sqlite3-wal`

No unexpected product or code. Did not STOP.

## 2. Rebase

Unstaged leftovers blocked `git pull --rebase`. Stashed the three
modified result files, then:

`git pull --rebase origin main`

Result: Current branch main is up to date. No conflicts. Hashes
unchanged (dde6ce0 / a60b8c1 / 057ecdc). Stash popped; leftovers
restored.

## 3. --fast (post-rebase)

`python dev/scripts/session_baseline_check.py --fast` at git-head
**dde6ce0**: OVERALL PASS.

pytest 1447 passed, 28 skipped (P1 env unset).
git-origin-main WARN was the expected 3-commit ahead-of-origin state
before push (origin still 6b23633 at that moment).

## 4. Push and origin verify

`git push origin HEAD`: 6b23633..dde6ce0 HEAD -> main.

`git fetch` then `git log --oneline -3 origin/main`:

```
dde6ce0 docs: stamp GAIN-PT tip a60b8c1 and --fast PASS.
a60b8c1 docs+results: GAIN-PT-RADIUS-01, SUBMIT-01 ready, U-09 closed.
057ecdc fix(gain): pin photon-transfer aperture at 4.0 px (GAIN-PT-RADIUS-01).
```

HEAD == origin/main == **dde6ce0**
(`dde6ce0d3e63240bb749f9c3a1600655e02f1edb`)
verified by fetch, not assumed. Rebase did not rewrite hashes.

## Commits as pushed

| hash | subject |
|------|---------|
| 057ecdc | fix(gain): pin photon-transfer aperture at 4.0 px (GAIN-PT-RADIUS-01). |
| a60b8c1 | docs+results: GAIN-PT-RADIUS-01, SUBMIT-01 ready, U-09 closed. |
| dde6ce0 | docs: stamp GAIN-PT tip a60b8c1 and --fast PASS. |

## Leftovers remain uncommitted

WIDE-ERR-04 leftovers still uncommitted:
`dev/results/CURSOR_RESULT_WIDE_ERR_04.md`,
`dev/results/WIDE_ERR_04_summary.json`.
Prior PUSH-AUTH append on `CURSOR_RESULT_PUSH_AUTH_20260817.md` also
uncommitted. Scratch dirs / sqlite shm-wal untracked.

## PUSH-STAMP-01

CONTENT tip: **dde6ce0**. Do not treat a later docs SHA as the content
tip. The origin received SHA equals the content tip (no rewrite).

This RESULT is not committed (origin SHA not chased into a committed
file).

## Docs impact

Docs impact: none (sync-only; this RESULT is uncommitted).

## Recurrence

Recurrence: n/a (not a bug-class; authorized push).
