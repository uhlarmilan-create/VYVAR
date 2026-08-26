CURSOR RESULT - 2026-08-26 (CLOSE-OUT C8 + push incident)

What I did
Recorded the 2026-08-25 `git push origin HEAD` incident. Did not
force-push origin/main. Installed a pre-push guard that refuses
updates to `main` without a gitignored PUSH_AUTH file. Measured
C8-1 R1' import, C8-2 frame 29 QC, C8-3 census depth pointer.
Did not run C6. Live 516/520 read-only except later C4 W2.

JSON: `dev/results/context/session_20260826_closeout/c8_r1p_import.json`
and `c8_frame29.json`.

## 0. Push incident

Milan recommendation accepted: do not rewrite origin/main back to
`b1f5b8c`. Tree at `7c086e8` passed `--fast --clean`. Every commit
on that SHA is a reviewed STOP.

Rule (PROCESS + DECISIONS): pushes name the ref
(`git push origin <local>:<remote>`). Banned: `git push origin HEAD`
and bare `git push`.

P0 guard: `dev/scripts/push_guard.py`. `session_baseline_check.py`
installs `.git/hooks/pre-push`. Updating `refs/heads/main` requires
gitignored `dev/PUSH_AUTH_main_<YYYYMMDD>.txt` containing the target
SHA; the hook deletes the file after a successful allow. Branch
pushes are unaffected. Test: `dev/tests/test_push_guard.py` (dry-run
to main without the file is refused).

If Milan later orders the reset: `git push --force-with-lease origin
b1f5b8c:main` exactly once, on explicit GO in chat;
`sel-ghost-01` stays at the then-current tip.

## C8-1 R1' pure control

First commit that tracked `src_py/dao_gaia_stage_01_iter4.py`:
**`0684ba9`** (`B1 follow-up: lock 4-tuple callers`, 2026-08-25).
The same commit also first-tracked `dao_gaia_stage_01_iter2.py` and
`dao_gaia_stage_01_iter3.py`.

R1' worktree: `.worktrees/c8_r1p_c592ecf` at `c592ecf` + that ONE
file copied from `0684ba9`. Isolated `sys.path` (HEAD `src_py`
dropped). Import result:

`ModuleNotFoundError: No module named 'dao_gaia_stage_01_iter2'`

iter4 imports iter2 and iter3 at module load. Isolated one-file copy
cannot import. Did not copy extra files. Full chain not run.
C8-P1 60-row R2-vs-R1' table: **blocked**. Prediction C8-P1
(D3 SNR 7.57; CSS_J134925 recovered; AC-overlay SHA moves) is
untested against a true pre-B1 control. Contaminated T3 R1 was not
re-run.

## C8-2 Frame 29 of 516

Target `1498000793739050368`. Freeze
`proc_BO_CVn_Light_029.csv` has the row (dao_flux 37387, x=703,
y=536). Live and T3 R2 lack it (n 3506 -> 3472).

R2 QC verdict: **admitted**. Live `qc_metrics.csv` status=`ok` for
frames 028/029/030. Metrics:

| frame | status | fwhm_px | elong | n_stars | residual_flatness_p99 |
|-------|--------|---------|-------|---------|------------------------|
| 028 | ok | 5.153 | 1.072 | 103 | 30.8 |
| 029 | ok | 5.146 | 1.051 | 263 | 47.8 |
| 030 | ok | 5.149 | 1.059 | 101 | 31.2 |

FWHM and elongation on 029 match neighbours. n_stars_detected=263
is 2.6x neighbours. residual_flatness 47.8 vs ~31.

Proc n (freeze / live=R2): 028 3503/3475, 029 3506/3472,
030 3517/3478. The ~30-row shrink is night-wide, not 029-only.

Same cid dao_flux: freeze 028=3680, 029=37387, 030=3388. Live/R2
keep 028 and 030 at those same fluxes and drop 029. Freeze 029 is
10x (2.6 mag) off the neighbouring epochs. R2 LC: 133 normal +
1 `no_data` (source_file `proc_BO_CVn_Light_029.csv`). Freeze LC:
134 normal, mag_calib 10.734 on that epoch.

Cause of the missing 029 row: later per-frame catalog no longer
associates this Gaia ID on a 10x-flux detection (match/identity of
a spurious association), not a cloud dropout. FWHM/elongation are
fine; neighbours keep the star. "Later rebuild" is the mechanism;
the named input is freeze-vs-live `proc_BO_CVn_Light_029.csv`
(Archive, not a git object). QC still admitted the frame.

FRAME-QC-PARITY phase 2 (record, do not wire): no n_stars outlier
gate; QC admitted frame 29 with 263 detections vs ~100 neighbours.

## C8-3 Census depth

`masterstar_gaia_census_target_depth_g` default 15.0
(`pipeline.py:13612-13618`) is a config constant. ROADMAP
**DEPTH-AUTH-01** (LOW): derive target/census depth from the
MASTERSTAR's own detection completeness vs Gaia (DAO-GAIA
certificate already measures recovery per magnitude bin). Not
wired. G=15.56 VSX `1500387696044768384` stays absent at re-cut
with that pointer.

## C6

Not run. Waits Milan GO in chat after C8 + C4 STOPs. Spec remains:
full chain to a NEW era04 (never overwrite era03); `--full` twice
byte-identical; DECISIONS delta ledger with named causes; unnamed
LC change = STOP, no era04 lock.

## Errors

C8-1 R1' cannot import with the one-file copy. C8-P1 table not
produced. No science-path edit in this STOP except the push guard.

## Files changed

- `dev/scripts/push_guard.py`, `dev/tests/test_push_guard.py`
- `dev/scripts/session_baseline_check.py` (install_hook)
- `.gitignore` (`dev/PUSH_AUTH_main_*.txt`)
- `docs/VYVAR_PROCESS.md`, `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_ROADMAP.md` (DEPTH-AUTH-01)
- `dev/results/CURSOR_RESULT_CLOSEOUT_C8.md`
- `dev/results/context/session_20260826_closeout/c8_r1p_import.json`
- `dev/results/context/session_20260826_closeout/c8_frame29.json`
