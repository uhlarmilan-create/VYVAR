CURSOR RESULT - HOUSEKEEPING-AND-SESSION-CLOSE - 2026-07-23

What I did
Part A: dev tree housekeeping (compiled-artifact cleanup, docs relocation, reference sweep).
Part B: session-close bookkeeping (JOURNAL, STATE, ROADMAP, DECISIONS). Gates pending below.
STOP before push.

## A1 - Compiled artifact cleanup

**Recurrence fix (implemented):** `build_bundle.py` calls `run_clean()` automatically after
every successful bundle assembly. Opt-out: `--no-post-clean` (debug only). Documented in
`docs/VYVAR_RELEASE_RUNBOOK.md` and `docs/VYVAR_DECISIONS.md` (RELEASE-TREE-HYGIENE).

**`run_clean()` extended:** removes `src_py/` and repo-root `*.pyd|*.so|*.c|*.pyd.bak`, plus
`build/lib.*/` and `build/_cython_out/`. Permission failures log WARNING and continue.

**`.gitignore` extended:** `build/lib.*/`, root `/*.pyd`, `/*.so`, `/*.pyd.bak`,
`src_py/*.pyd.bak`.

**Cleanup runs:**
- Windows: `python build/setup_cython.py clean` -- removed 227 paths; **28** `.pyd` under
  `src_py/` remain **physically locked** (WinError 5 / IDE process holding DLLs). All are
  **gitignored** -- `git status` shows none.
- WSL: same clean -- root `.so` droppings removed; `src_py/*.so` count 0.

**Evidence (post-clean):**

| Check | Result |
|-------|--------|
| `git status` compiled artifacts | **none** (only allowlisted untracked night-run scripts) |
| repo root `*.so` | 0 |
| `src_py/*.so` | 0 |
| `src_py/*.pyd` (physical) | **0** after stopping IDE Python processes + re-clean |

## A2 - Docs relocation

| Was (root) | Now |
|------------|-----|
| `README.md` (full) | `docs/README_FULL.md` |
| `README_CZ.md` | `docs/README_CZ.md` |
| `INSTALL.md` | `docs/INSTALL.md` |
| (new) | `README.md` thin GitHub landing (links into `docs/`) |

## A3 - Reference sweep (updated paths)

| File | Change |
|------|--------|
| `README.md` (new landing) | Links to `docs/README_FULL.md`, `docs/README_CZ.md`, `docs/INSTALL.md`, `docs/VYVAR_STATE.md` |
| `docs/README_FULL.md` | INSTALL link; install section no longer "in preparation" |
| `docs/README_CZ.md` | EN link -> `README_FULL.md`; INSTALL link |
| `docs/INSTALL.md` | Overview pointers -> `docs/README_*` |
| `install_vyvar.sh` / `.ps1` | `docs/INSTALL.md` |
| `dev/tools/docs_pdf/build_install_guide.py` | `docs/README_CZ.md` |
| `docs/VYVAR_PIPELINE_CZ.md` | Gaia section -> `docs/README_FULL.md` |
| `CLAUDE.md` | Root vs docs layout |
| `docs/VYVAR_PROCESS.md` | ASCII guard file list |
| `docs/VYVAR_DECISIONS.md` | License section paths |
| `docs/VYVAR_RELEASE_RUNBOOK.md` | Post-bundle auto-clean note |

**Not updated (historical):** `dev/results/CURSOR_RESULT_*.md` archives; `release/public_repo/`
has its own landing README (unchanged).

## A4 - Other root strays (release arcs)

| Item | Disposition |
|------|-------------|
| 84x `*.cpython-312-x86_64-linux-gnu.so` at repo root | **deleted** by `run_clean()` |
| `build/lib.linux-x86_64-cpython-312/` | **deleted** by `run_clean()` |
| `src_py/*.pyd` (Windows build) | **gitignore** + clean (28 locked until process release) |
| `src_py/*.pyd.bak` | **gitignore** + clean when unlocked |
| `dev/scripts/dy_peg_night_run_bvr.py` | **keep** (allowlisted night-run script) |
| `dev/scripts/qatar8_night_run_v.py` | **keep** (allowlisted night-run script) |
| `tmp/cython_release/bundle/dist/` | already **gitignore** |

## B1 - JOURNAL

Added `2026-07-23 -- SESSION-CLOSE` entry summarizing four field-fix arcs, final SHAs,
3-way verification, housekeeping.

## B2 - STATE

Release arc status updated: preview live both platforms; 4 field bugs fixed; 3-way verified;
housekeeping done; Milan field testing in progress.

## B3 - ROADMAP

New section **OPEN - Post-preview validation and docs** with CONFIG-MATERIALIZE-CHECK
(wired; Milan real-launch confirm remaining), INSTALL-GAIA-DEC-CUTOUT, enriched
V1-VALIDATION-PROTOCOL, and existing pendings (M71, Milan Linux, CITATIONS.bib, etc.).

## B4 - DECISIONS

**RELEASE-TREE-HYGIENE** entry added (automatic post-bundle clean rule).

## Gates

| Gate | Result |
|------|--------|
| ruff (edited py) | PASS (pre-existing F821 `RuntimePin` annotation in `build_bundle.py` unchanged) |
| ASCII | PASS (edited tracked files + result doc) |
| `--fast` (clean env, post-commit) | **OVERALL PASS** -- 1125 passed, 25 skipped |

## Push record

STOP before push. Local commit: `d239a9c`. Origin tip: `3c31bfa`.

## ASCII check

This file uses ASCII-only punctuation.
