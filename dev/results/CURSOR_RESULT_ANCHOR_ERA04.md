CURSOR RESULT - 2026-08-26T08:20:00Z

What I did
C6-0 resume (Milan GO, A-only from `git diff --name-status
c592ecf 0684ba9`). Copied the six A files from 0684ba9 into
`.worktrees/c6_r1p_c592ecf`. Did not copy any M file. Isolated
sys.path (HEAD src_py dropped). Import OK. Full chain FAIL on
4-tuple vs 3-tuple lock_existing. Did not widen the copy set.
C6-1..C6-4 not started. era03 untouched. era04 not locked.

## Premise (Rule 0.1)
Compared: isolated c592ecf plus exactly the A files of
c592ecf..0684ba9 versus T3 R2 at HEAD. Differ: A files include
STAGE-01 iter2/3/4 and dao_gaia_stage_01.py; M files (including
masterstar_gaia_accounting.py 4-tuple) stay at c592ecf.

## A files copied from 0684ba9 (path + bytes)

- `dev/results/context/session_20260819_daostage01_iter4/final_scores.csv`  1188
- `dev/tests/test_inv_match_identity_01.py`  5150
- `src_py/dao_gaia_stage_01.py`  28353
- `src_py/dao_gaia_stage_01_iter2.py`  29378
- `src_py/dao_gaia_stage_01_iter3.py`  26543
- `src_py/dao_gaia_stage_01_iter4.py`  24571

M not copied: test_masterstar_gaia_01.py, VYVAR_INVARIANTS.md,
astrometry_optimizer.py, dao_gaia_calibration.py, gaia_catalog_id.py,
invariants_runtime.py, masterstar_gaia_accounting.py, pipeline.py,
ui_aperture_photometry.py, wcs_invertibility.py.

JSON: `dev/results/context/session_20260826_c6/c6_0_resume_a_files.json`

## Import
OK. Loaded from worktree: dao_gaia_stage_01, iter2, iter3, iter4,
pipeline (c592ecf). No missing module.

## Chain
First attempt: FileNotFoundError on worktree-relative
`Archive/Drafts/draft_000516/.../MASTERSTAR.fits` (STAGE-01 FRAMES
use REPO=worktree). Junctioned live Archive into the worktree
(data path only; no extra git file). Not an M-file copy.

Second attempt: MS enrich calls the certificate validation gate,
which runs iter4.score_validation_params. iter4.py:238 unpacks

`lock_existing_and_leftover_assign` into 4 values.

c592ecf `masterstar_gaia_accounting.py` returns 3
`(det_to_g, gaia_owner, match_mode)`.
0684ba9 (M, not copied) returns 4
`(..., geometry_reject_dets)`. Commit 0684ba9 message is
"lock 4-tuple callers".

`ValueError: not enough values to unpack (expected 4, got 3)`

Did not copy `src_py/masterstar_gaia_accounting.py`. Photometry not
run. 60-row R2-vs-R1' table not produced. C6-0-P1 untested.

Rule 0.3: import probe ~3 s; attempt 1 MS ~60 s; attempt 2 MS ~50 s.

## C6-1..C6-4
Not started. Live 516 SHA unchanged. era03 freeze 9902d918 / 472bc9e4
untouched.

## Errors
C6-0 STOP after import OK: A-only iter4 is incompatible with c592ecf
lock_existing 3-tuple. The file that would make the 4-tuple is M.

Resume needs a named GO to copy M `src_py/masterstar_gaia_accounting.py`
(or skip the R1' table and run C6-1 vs era03).

## Files changed
- `dev/results/CURSOR_RESULT_ANCHOR_ERA04.md`
- `dev/results/CURSOR_RESULT_SESSION_CLOSE_20260826.md`
- `dev/results/context/session_20260826_c6/c6_0_resume_a_files.json`
- docs STATE / ROADMAP / JOURNAL / DECISIONS
