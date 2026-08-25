CURSOR RESULT - SESSION-CLOSE-20260824

Date: 2026-08-25 (close of 2026-08-24 session). Architect: Claude.
Decision authority: Milan. Push: YES (this close task is commit / sync /
push / verify; Milan sent it).

## Premise (Rule 0.1)

**What is compared:** local `main` before this close versus `origin/main`
`b1af049` (SESSION-CLOSE-20260823), plus unstaged AC-02 / REG-520-01 /
handoff work versus leftover DAO/VALID dirt.

**How they differ:** 11 local commits were already on disk and unpushed
(`dbb6967`..`92361a3`). AC-02 wiring and REG-520-01 results were still
unstaged. After this close, `origin/main` is `8842995` and matches
local HEAD.

## Inventory (before commits)

`git log origin/main..HEAD` matched the expected series exactly:

```
92361a3 docs(cal): CAL-520-01 measure STOP for draft 520 pre_cal misclass
505fa13 docs(dao): XFER-01 W6 MULTIFILTER-WCS-01 + g-WCS-on-z measure
e5a6149 fix(dao): pin STAGE-01 sandbox gate to hand params (XFER-01)
d66613c docs(dao): REGRESS-01 draft 520 preflight diagnosis (H-GATE-XFER)
2926a95 docs(epsf): PIN-CENSUS-01 chi2>=50 pin-drop census (WIDE rig 516)
876053a docs(epsf): EPSF-AC-01 A1-A3 measurements and policy ranking
d206b43 docs(epsf): AC chi2<5 brightness-cut finding and EPSF-CORE-01 future row
cf95c53 docs(epsf): SHAPE-01-F measurements; supersede M3 H5 fitter-scale
6fd1452 feat(psf): persist x_fit y_fit and psf_group_n through F6 merge
1f9f921 feat(epsf): SHAPE-01-M bright-star mismatch measurements
dbb6967 feat(psf): internal diagnostic PSF light curves and submit guard (EPSF-LC-LOG-01)
```

CAL-520-01 was already in `92361a3`. PIN-CENSUS already in `2926a95`.

## Commit SHAs per group

| Group | SHA | Subject |
|-------|-----|---------|
| 1 AC-02 wiring | `6fbfbea5ab5d7638d33130d75bee371981c6a85b` | feat(psf): wire P4 AC policy and INV-PSF-LC-PIN-01 |
| 2 REG-520-01 + 518 + four docs | `db628bec05a35e13ef3c27e08b7028d3c444972d` | docs(reg): REG-520-01 measure STOP and Newton 518 ePSF STOP |
| 3 handoff | `ccf0b5f7462e7e6d3ec78a8f519917d1187f0b27` | docs: SESSION-CLOSE-20260824 next-session handoff |
| 4 ASCII gate (required for --fast) | `88429950ebf4fdd7a9fdf887b301c410b3f22a3c` | docs(epsf): ASCII-migrate VALID-02/BRIGHT-01 results |

Handoff file: `docs/VYVAR_HANDOFF_2026-08-24.md`. Architect paste was
not in the Cursor user_query; the file is ASCII assembled from
STATE/ROADMAP/DECISIONS plus the close-task known-issues list quoted
verbatim. Provenance is stated at the top of the file.

Four docs (JOURNAL/STATE/ROADMAP/DECISIONS) unstaged diff was REG-520
one-liners only; AC-02/census/518 one-liners were already in earlier
commits. Group 2 also added uncommitted Newton 518 result + session
and the SHAPE-01-F 518 addendum.

## Leftover dirt (not committed)

Unrelated DAO/VALID/MASTERSTAR working files, not named done for this
close:

- `CURSOR_RESULT_DAO_*`, `CURSOR_TASK_DAO_*`
- `CURSOR_RESULT_MASTERSTAR_*`, `CURSOR_TASK_MASTERSTAR_*`
- `CURSOR_RESULT_EPSF_VALID_01.md`, `CURSOR_RESULT_ERA03_FINAL_CLOSE_20260820.md`
- `CURSOR_RESULT_PUSH_AUTH_20260817_2.md`, `CURSOR_RESULT_EPSF_ARC_PUSH.md`
- `dev/results/context/session_20260818_*`, `session_20260819_*`
- leftover VALID-02 accept/s1s4/s5 context dirs (the 2026-08-23 close
  already committed the curated subset)
- `session_20260824_epsf_ac_01/`, `session_20260824_epsf_shape_01_f/`,
  `session_20260824_epsf_shape_01_m/` (measure dirs whose results were
  already committed earlier)
- `session_20260821_d10_1` tracked CSV/PNG/JSON (stashed; ASCII-irrelevant)
- scratch: `logs/`, `src_py/tmp/`, `vyvar.sqlite3-shm/wal`,
  `dev/tests/_tmp_batch_e_lc/`

The five VALID-02/BRIGHT-01 markdown files were **not** left behind:
working-tree ASCII replacements were the only reason `--fast` was green
on a dirty tree. Clean HEAD failed `test_tracked_text_files_are_ascii`.
Those five files were committed as group 4 so origin checkout stays
green.

## Sync

`git pull --rebase origin main` after groups 1-3: **Current branch main
is up to date.** No conflicts. origin still `b1af049` at that moment.
Hashes not rewritten.

## `--fast` before push

First run on `ccf0b5f` (groups 1-3, ASCII stash hiding the 1252-byte
dashes): **OVERALL FAIL** -- `test_tracked_text_files_are_ascii` (5
offenders: BRIGHT-01 + VALID-02 R1R4/R5F5/S1S4/S5B).

After group 4 ASCII commit, second run on **`8842995`**:

```
OVERALL: PASS
pytest                       PASS   1530 passed, 32 skipped
git-head                     PASS   8842995
git-origin-main              WARN   differs from origin/main (b1af049)
db-quick-check               WARN   WAIVED
```

Expected WARNs: untracked, ledger-todo, deps-outdated. Wall ~8.7 min.

## Push / verify

Authorization: SESSION-CLOSE-20260824 task title is commit, sync, push,
verify; Milan sent it in chat.

```
git push origin HEAD
  b1af049..8842995  HEAD -> main
git fetch origin
git rev-parse HEAD        = 88429950ebf4fdd7a9fdf887b301c410b3f22a3c
git rev-parse origin/main = 88429950ebf4fdd7a9fdf887b301c410b3f22a3c
```

**HEAD == origin/main == `88429950ebf4fdd7a9fdf887b301c410b3f22a3c`.**
(This result file is a follow-up stamp commit after that verify.)

Pushed range `b1af049..8842995` (15 commits): the 11 expected local
series plus `6fbfbea`, `db628be`, `ccf0b5f`, `8842995`.

## Known issues (do not fix tonight)

- T1 test still rewrites the live 516 BO CVn PSF LC on every `--fast`
  (fix parked in EPSF-ZP-OK-01-WIRE v2 W3, which has not run).
- ZP-OK v2 GO undecided.
- DAO-GAIA production tolerance floor unwired (REG-520-01 menu a / M2).
- 520 selection-input defect unwired (REG-520-S2).

## Files changed (this close task)

- AC-02 code/config/tests/docs + `CURSOR_RESULT_EPSF_AC_02_WIRE.md`
- REG-520-01 + Newton 518 results/session/context + four docs one-liners
- `docs/VYVAR_HANDOFF_2026-08-24.md`
- ASCII migrate of five VALID-02/BRIGHT-01 result markdowns
- this file (stamped after origin verify)

## Errors

First `--fast` FAIL (ASCII) caught and fixed before push. No rebase
conflicts. Push verified by fetch, not assumed.
