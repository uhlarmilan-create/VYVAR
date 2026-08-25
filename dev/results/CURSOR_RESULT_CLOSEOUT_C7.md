CURSOR RESULT - 2026-08-25 (CLOSE-OUT C7 pre-C6 verification)

What I did
Measure-only plus one harness isolation fix in the T3 runner
(`run_t3.py` drops HEAD `src_py` when inserting the R1 worktree).
Did not recut the era03 freeze. Did not expand-depth-wire C7-4.
Live 516/520 read-only. JSON:
`dev/results/session_20260825_closeout/c7_measure.json`.

## Premise (Rule 0.1)

**What is compared:** T3 R1 "at c592ecf" expand (3606) versus an
isolated c592ecf worktree; `--full` copy list versus INV-ANCHOR-00
blind zone; R0 vs R1 MS/LC for two headline IDs; freeze vs R1/R2
for lost target `1500387696044768384` and census 4+1.

**How they differ:** R1 harness left HEAD `src_py` on `sys.path`, so
`dao_gaia_stage_01_iter4` resolved to the main tree.

## C7-1 R1 purity -- harness-fixed

Harness: `dev/results/session_20260825_sel_ghost_01_b3/run_t3.py`
inserted `WT_R1/src_py` at sys.path[0] and did **not** remove
`ROOT/src_py`. Isolated `sys.path` with only the worktree:
`dao_gaia_stage_01_iter4` ModuleNotFoundError (file absent at
`c592ecf`). Contaminated insert: import `__file__` =
`C:\ASTRO\python\VYVAR\src_py\dao_gaia_stage_01_iter4.py` (HEAD).

T3 R1 expand to 3606 used HEAD STAGE-01. **R1 is not a pre-B1
control.** Isolated R1 cannot census-expand (C1c n=2643). Did not
redo the 60-row T3 R2-vs-R1 table by recontaminating.

Fix applied on the session harness: drop HEAD `src_py` before
inserting the worktree. Do not re-run T3 R1 (would be the
no-expand table).

## C7-2 What --full copies -- authorized-with-commit (file:line)

`session_baseline_check._copy_frozen_anchor_inputs`
`dev/scripts/session_baseline_check.py:578-608`:

- `platesolve/NoFilter_60_2/` except `photometry/`, `_hrd_cache/`,
  `*.pdf` (MASTERSTAR FITS+CSV, WCS in headers, field catalog, ePSF)
- `detrended_aligned/lights/NoFilter_60_2/` (aligned FITS; snapshot
  proc CSVs if present in that tree)
- `cal_diag.json`, `draft_manifest.json`, `sat_diag.json`
- then destination `photometry/` is deleted and recreated empty

Not copied: raw lights, CalibrationLibrary masters, live LCs.
T3 copied from **live** 516, not the era03 snapshot. Blind zone
recorded in PROCESS and INV-ANCHOR-00.

## C7-3 headline epochs -- authorized-with-commit-ad19e14 (named input)

MS row `1498000793739050368` identical R0 vs R1 on x, y,
source_state, vy_match_mode, name (DETECTED_P1, locked).

Named input: era03 snapshot `proc_BO_CVn_Light_029.csv` **has** the
target (x~703, y~536, dao_flux~37387). Live / T3 R1 / T3 R2 proc_029
**lack** it (n 3506 -> 3472). Freeze LC mag 10.734 is from snapshot
proc; R1 `flag=no_data` is `photometry_core.py:3494` default when
the star is not in the frame `id_map`. Same pattern for 17
R0-finite / R1-NaN epochs of `1485987151737107200`.

Archive proc CSVs are not git objects. Handling of missing rows is
the longstanding no_data path. Freeze snapshot SHA `ad19e14`.
Not unexplained: the input that changed is named (snapshot proc vs
later per-frame DAO/match rebuild).

## C7-4 lost target 1500387696044768384 -- authorized-with-commit (depth policy)

VSX VAR P=0.224918, G=15.559. Freeze: DETECTED_P1, 8-comp ensemble,
but `active_targets` already `below_target_depth` and LC `no_data`.
Absent R1 and R2 MS.

Why: DAO miss on the fresh stack **and**
`masterstar_gaia_census_target_depth_g` default 15.0
(`pipeline.py:13612-13618`) so expand will not add G>15. Chip
position ~1807,587 (on-chip). Not a ghost. Freeze science LC was
already no_data.

Not wired as a defect: raising census depth to keep every freeze
VSX row is a Milan GO, not a silent expand-rule change. C6 recut
would drop this MS row; aperture LC product was already empty.

## C7-5 census 4+1 -- authorized-with-commit (same depth / B1 identity)

R0-only (all G>15.28, DETECTED on freeze, absent R1/R2):

- `1496997386298488832` G=15.28
- `1497063283984301696` G=15.37
- `1498903316693061248` G=15.33
- `1500387696044768384` G=15.56

R1-only `1504304603139151872`: G=17.45, name=DET_0482, edge ~78,83,
**absent R2** (B1 identity).

R2 vs R1 unique cid sets are **not** equal (n_r0=3585, n_r1=3582,
n_r2=3584). `1500387696044768384` is in none of R1/R2. T3-P4
"R2 == S3 3583" is a different count grain.

## Classification table

| item | class |
|------|--------|
| C7-1 | harness-fixed |
| C7-2 | authorized-with-commit (session_baseline_check.py:578-608) |
| C7-3 | authorized-with-commit-ad19e14 (named snapshot proc) |
| C7-4 | authorized-with-commit (membership_depth_g=15.0); target lost from MS, freeze LC already no_data |
| C7-5 | authorized-with-commit (G>15 expand + B1 identity) |

## C6

Not asked. A legitimate VSX row is absent from R1/R2 MS (C7-4),
even though the freeze LC was already no_data. C7-1 means the T3
R1 60-row table is not a pre-B1 control. Nothing in this STOP is
classed unexplained.

## Errors

None.

## Files changed

- `dev/results/CURSOR_RESULT_CLOSEOUT_C7.md`
- `dev/results/session_20260825_closeout/c7_measure.json`
- `dev/results/session_20260825_sel_ghost_01_b3/run_t3.py` (sys.path isolation)
- `docs/VYVAR_PROCESS.md`, `docs/VYVAR_INVARIANTS.md` (copy list)

Docs impact: PROCESS, INVARIANTS INV-ANCHOR-00, STATE/JOURNAL/ROADMAP.
Recurrence: worktree harnesses must drop HEAD `src_py` before inserting
a historic tree.
