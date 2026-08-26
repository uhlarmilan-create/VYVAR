CURSOR RESULT - 2026-08-26T07:25:00Z

What I did
C6 GO executed. C6-0 R1' three-file isolated import STOP.
Missing module: dao_gaia_stage_01. Did not copy a fourth file.
C6-1..C6-4 not started. era03 not overwritten or deleted.
era04 not locked. origin/main not pushed.

## Output / findings

### Premise (Rule 0.1)
Compared: isolated c592ecf worktree plus the three STAGE-01 iter
files from 0684ba9 (the set C8 named as first-tracked together)
versus T3 R2 at HEAD. Differ: R1' is the uncontaminated pre-B1
control column for the 60-row ensemble table; T3 R1 was contaminated
by HEAD iter4 on sys.path (C7/C8).

### C6-0 STOP
Worktree: `.worktrees/c6_r1p_c592ecf` at `c592ecf`.
Copied from `0684ba9` only:

- `src_py/dao_gaia_stage_01_iter2.py` (29378 bytes)
- `src_py/dao_gaia_stage_01_iter3.py` (26543 bytes)
- `src_py/dao_gaia_stage_01_iter4.py` (24571 bytes)

sys.path: HEAD `src_py` dropped; worktree `src_py` inserted at [0]
(same as `run_t3.py` C7-1 fix).

Import:

```
ModuleNotFoundError: No module named 'dao_gaia_stage_01'
```

Site: `iter4.py:42` imports iter2; `iter2.py:49`
`from dao_gaia_stage_01 import ...` after inserting `REPO/tmp`.

c592ecf does not contain `src_py/dao_gaia_stage_01.py`.
0684ba9 does (same commit that first-tracked the three iter files).
HEAD disk has `src_py/dao_gaia_stage_01.py`. Worktree `tmp/` has no
copy. Did not copy that file (task: do not copy further files
silently). Full chain not run. 60-row R2-vs-R1' table: blocked.
C6-0-P1 untested.

JSON: `dev/results/context/session_20260826_c6/c6_0_import.json`

### C6-1..C6-4 not started
No full-chain recut. Live 516 and era03
(`draft_000516_snapshot_era03_20260820`, core 9902d918 n=121,
ext 472bc9e4 n=179) untouched. No era04 snapshot directory created.
`--full` not run against a new freeze. Delta ledger not written.
INV-ANCHOR-00 still points at era03.

### C6-5 inventory (STOP, no lock)
See `dev/results/CURSOR_RESULT_SESSION_CLOSE_20260826.md`.
No SHA for `dev/PUSH_AUTH_main_<date>.txt` until era04 locks.

### Rule 0.3
worktree add 1.9 s; copy three files 1.0 s; import probe 4.4 s;
C6-0 total 7.3 s. Chain not started.

## Errors (if any)
C6-0 STOP: missing module `dao_gaia_stage_01`.

Milan GO needed to resume (pick one):
1. Authorize copy of `src_py/dao_gaia_stage_01.py` from 0684ba9
   (fourth file, same commit as the three), re-run C6-0, then C6-1.
2. Skip the R1' table; run C6-1 with C6-3 vs era03 only
   (C6-0-P1 stays untested).
3. Use worktree at 0684ba9 instead of c592ecf (not a c592ecf control).

## Files changed
- `dev/results/CURSOR_RESULT_ANCHOR_ERA04.md`
- `dev/results/CURSOR_RESULT_SESSION_CLOSE_20260826.md`
- `dev/results/context/session_20260826_c6/c6_0_import.json`
- `docs/VYVAR_STATE.md`, `docs/VYVAR_ROADMAP.md`,
  `docs/VYVAR_JOURNAL.md`
