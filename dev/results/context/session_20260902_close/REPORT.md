# CURSOR RESULT - SYNC-CLOSE 2026-09-02

Date: 2026-09-02 (close run 2026-09-03). Branch: consolidate-01.
Architect: Claude. Implementer: Cursor.
Compared with: working tree on `039db43` (E4 report tip, already
matching `origin/consolidate-01`) vs the leftover untracked pile.
Push only `git push origin consolidate-01:consolidate-01`. main not
pushed. PUSH_AUTH file not created.

Premise: this close is housekeeping, not a product extract. Approved
groups are leftover session evidence under `dev/results/`. Gitignored
`tmp/` was left alone. Large blobs, UTF-16 stdout, runtime logs, and
the E4 `src_py/tmp` shadow were listed, not committed.

## What I did

Inventoried `git ls-files --others --exclude-standard` (6203 files,
~4.24 GB at start). Committed three approved groups (one commit each),
then this close report. `git pull --rebase origin consolidate-01` was
up to date (no conflicts). Then push of `consolidate-01:consolidate-01`
and G1 `--fast --clean` on the final tip.

## Commit list

| group | commit | files | note |
| --- | --- | --- | --- |
| CURSOR_TASK / CURSOR_RESULT markdown | `36d6dcbe6c4e6c8d89fbefb1dcddd569a9b1912b` | 20 | leftover DAO / ePSF / masterstar-Gaia working docs under `dev/results/` |
| context session evidence | `1674fe933c9d7f3dec152faaa140db94f2658e42` | 154 | Rule 0.2 CSV/JSON/txt/md/py leftovers, ASCII, typically under 256 KiB |
| leftover gate logs | `a7f6429f32bbb765eaeaacf605efe9603021433a` | 10 | in-session G1/G2/G7 logs already sitting under `dev/results/context/` |
| ENCODING-POLICY fold | `8400e36d67eb13eeeb5aeacc5a75f8b80c66ba24` | 7 | cp1252 dashes/quotes in group-1 markdown; `ascii_migrate.py` |
| this close report | (this commit) | 1+ | `session_20260902_close/` |

Honest messages:

1. `SYNC-CLOSE: track leftover CURSOR_TASK/CURSOR_RESULT markdown under dev/results.`
2. `SYNC-CLOSE: track small leftover context-session CSV/JSON/txt evidence (Rule 0.2).`
3. `SYNC-CLOSE: track leftover in-session G1/G2/G7 gate logs under context sessions.`
4. `SYNC-CLOSE: ASCII-fold leftover CURSOR markdown so ENCODING-POLICY passes.`
5. `SYNC-CLOSE: session REPORT for 2026-09-02 close.`

No validation-ledger group: ledger was already stamped at E4 (`039db43`).
No docs group: no uncommitted tracked docs.

Filter used for context evidence: extension in `{.md,.json,.csv,.txt,.py}`,
size cap 256 KiB except two ASCII G7 parity logs (~261 KiB) that belong
in the gate-log group; deny `proc_backup_pre_accept/`, `s1_baseline_non_psf/`,
`overlays/`, `pre_merge_sidecars/`; skip UTF-16 (NUL) stdout.

## Left uncommitted (and why)

Gitignored scratch (`tmp/`) was not added.

| leftover | scale | why not committed |
| --- | --- | --- |
| `dev/results/session_*` (not under `context/`) | 5514 files, ~3478 MB | full draft sandboxes (`field_catalog_cone.csv` and other catalog dumps). Not Rule 0.2 session evidence. |
| `dev/results/context/` leftovers after the keep filter | 458 files, ~757 MB | almost all `proc_backup_pre_accept/` and `s1_baseline_non_psf/` (~349 MB each), DAO overlay PNG/NPZ, `pre_merge_sidecars/` (~7.7 MB), catalog CSVs over the size cap (`masterstars_full_match.csv`, `forced_seeds.csv`, `e3b_pass2_survival.csv`, ...) |
| UTF-16 stdout (PowerShell `Tee-Object`) | 5 files | encoding/NUL; would fail tracked-text gates. Paths: `session_20260822_epsf_valid_02_accept/full_baseline_stdout.txt`, `.../f4/build_stdout.txt`, `.../s1s4/fast_baseline_stdout.txt`, `.../s1s4/fast_baseline_stdout_v2.txt`, `session_20260827_parity/g2_after_3a.txt` |
| `logs/` | 43 files, 1.53 MB | runtime `run_preflight_error_*.log` plus a stale `logs/docs/` shadow of `docs/` (same class of mistake as the E4 `src_py/tmp` shadow) |
| `src_py/tmp/xval_out/xval_results.csv`, `xval_sources.csv` | 2 files, ~21 KB | E4 tmp-shadow lesson: do not track `src_py/tmp/` |
| `dev/tests/_tmp_batch_e_lc/` | 2 tiny CSVs | test scratch |
| secrets / PUSH_AUTH / large FITS | none in the would-add set | `*.fits`, `*.sqlite3`, `dev/PUSH_AUTH_main_*.txt` already gitignored |

No stale `photometry_core.py` / `photometry_phase2a.py` copies under `dev/`.
`delete_funcs.py` in `session_20260831_c01b` is a session AST helper, not a
module shadow; it was kept with the evidence group.

## Gate

First G1 `--fast --clean` at `a7f6429` (before the ASCII fold): **FAIL**.
`test_ascii_policy.py::test_tracked_text_files_are_ascii` found 7 group-1
markdown files with cp1252 bytes (0x96/0x97 dashes, 0x85 ellipsis, curly
quotes, 0xd7 multiply). `clean-tree` still PASS. Log: `g1_content_fail.txt`.

Retry at `8400e36` (ASCII fold, this report still untracked): **PASS**.
pytest 1638 passed, 32 skipped. clean-tree PASS
(`worktree=b1b_clean_d0b3df0f`; pytest 32 passed; ruff PASS; pyflakes PASS).
db-quick-check WARN waived (same as E4). Log: `g1_ascii_fold.txt`.

Final-tip G1 after this close-report commit: filled after the push. Log: `g1.txt`.

## FINAL_SHA

Pending until after `git push origin consolidate-01:consolidate-01`.

## Ancestry

`git merge-base --is-ancestor 5b1068d HEAD` at the content-sync tip
`a7f6429` exited 0. Re-checked on the final tip after push.

`5b1068d` = `5b1068d2b04c63a1b9dc44c2580d0a5b31ace729`
(`MERGE-MAIN-01` report).

## Main hand-off (print only; not executed)

Do **not** create the file. Do **not** push main.
Banned forever: `git push origin HEAD` and bare `git push`.
The guard refuses main without the auth file.

For Milan, after FINAL_SHA is known:

1. create `dev/PUSH_AUTH_main_20260902.txt` containing FINAL_SHA
   (full 40 chars, one line)
2. `git push origin FINAL_SHA:main`

## Errors

None. Rebase reported "Current branch consolidate-01 is up to date."
No conflict to resolve.
