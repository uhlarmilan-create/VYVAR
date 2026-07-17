CURSOR RESULT - 2026-07-17

# REPO-REORG - src_py/ + dev/ layout, anchor-gated

## What I did

Reorganized the repository root into the Milan-approved stable layout: all
VYVAR Python modules now live under `src_py/`, all internal development
material under `dev/`, and the root keeps only config/metadata files plus a
thin `app.py` Streamlit shim. Every move used `git mv` (history preserved).
No science-module import statement was rewritten except the one authorized
de-qualification (see Phase B). The Phase C anchor gate against Anchor #3
PASSED byte-identical.

Baseline: origin/main `62410c8` (Anchor #3 ACTIVE, draft_000435).

## Per-phase commits

| Phase | Commit    | Subject                                                          |
|-------|-----------|------------------------------------------------------------------|
| A     | `c611353` | chore(layout): move dev-side dirs and Cursor results into dev/   |
| B     | `8f4d7b4` | chore(layout): move VYVAR modules into src_py/ with entry shims  |

### Phase A diff --stat
`472 files changed, 75 insertions(+), 42 deletions(-)`

Rename breakdown (`git diff --name-status -M 62410c8 c611353`):
- renames (R): 469 (of which 451 are pure R100 rename-only)
- modified (M): 3 -> `.gitignore`, `pyproject.toml`, `params_registry.py`
- Rename percentage: 469 / 472 = **99.4% pure renames**

Moves: `tests -> dev/tests`, `tools -> dev/tools`,
`validation -> dev/validation`, `scripts -> dev/scripts`,
`sandbox -> dev/sandbox`, `orchestrator -> dev/orchestrator`; all root
`CURSOR_RESULT*.md` / `CURSOR_TASK*.md` -> `dev/results/`. `CLAUDE.md`,
`CHANGELOG.md` stayed at root.

Path fixes in Phase A (breakage from the dev/ moves):
- `pyproject.toml`: `testpaths = ["dev/tests"]`; ruff excludes ->
  `dev/sandbox`, `dev/scripts/archive`.
- `.gitignore`: `/dev/sandbox/`, `dev/orchestrator/api_key.txt`,
  `dev/tests/validation/data/`, verify-log globs under `dev/scripts/`.
- `params_registry.py`: `REGISTRY_PATH -> dev/validation/params_registry.json`.
- `dev/tools/gen_params_md.py`: ROOT = `parents[2]`; sys.path covers root+src_py.
- `dev/tests/test_params_registry.py`: `_ROOT = dev/`, `_REPO = parents[2]`
  for `docs/VYVAR_PARAMS.md` freshness.
- Ledger guard: ledger moved to `dev/validation/VYVAR_VALIDATION_LEDGER.json`,
  remains tracked; guard still fails on silent deletion.
- `dev/scripts/session_baseline_check.py`: `REPO_ROOT = parents[2]`,
  `LEDGER_PATH -> dev/validation/...`, `_ensure_import_paths()` added,
  `KNOWN_UNTRACKED_PREFIXES -> dev/results`, `dev/scripts`.
- `dev/scripts/fix_telescope_diameter.py`: `_REPO = parents[2]`, sys.path
  covers `src_py` + repo root (subprocess import of `config`).
- Nine `dev/tests/*` files: external-resource paths bumped `parents[1] ->
  parents[2]` (Archive, GAIA_DR3, VSX, vyvar.sqlite3) or switched to
  `Path(module.__file__)` resolution.

Gate A: full pytest green + `session_baseline_check.py --fast` PASS.

### Phase B diff --stat
`226 files changed, 3374 insertions(+), 2921 deletions(-)`

Rename breakdown (`git diff --name-status -M c611353 8f4d7b4`):
- renames (R): 95 (of which 85 pure R100)
- modified (M): 129 (mostly the 124 dev scripts adopting `_bootstrap`, plus
  ~13 `src_py/` modules with `__file__` fixes and the classifier change)
- added (A): 2 -> root `app.py` shim, `dev/scripts/_bootstrap.py`

Moves: all remaining root `*.py` -> `src_py/`. New thin root `app.py`:

```python
_SRC = Path(__file__).resolve().parent / "src_py"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
runpy.run_path(str(_SRC / "app.py"), run_name="__main__")
```

Entry-point path visibility (NO import rewrites in modules):
- `pyproject.toml`: `[tool.pytest.ini_options] pythonpath = [".", "src_py", "dev"]`.
- `dev/scripts/_bootstrap.py`: shared bootstrap inserting `src_py`, `dev/scripts`,
  `dev`, repo root; 124 dev scripts adapted mechanically to import it.
- `dev/orchestrator/vyvar_orchestrator.py`: `REPO_ROOT = parents[2]`,
  `LOG_FILE -> dev/orchestrator/session.log`.

### Authorized science-module edit (the only one)
`scripts/repair_catalog_ids.py` was imported by production modules
(`photometry_core.py`, `pipeline.py`, `wcs_invertibility.py`). Milan approved
Option A: move it to `src_py/repair_catalog_ids.py` and de-qualify its 6 lazy
imports from `from scripts.repair_catalog_ids import X` to flat
`from repair_catalog_ids import X`. No other science-module content changed.

## __file__ audit table (all hits in src_py/)

| File                     | Line  | Expression                                    | Purpose                     | Disposition |
|--------------------------|-------|-----------------------------------------------|-----------------------------|-------------|
| `config.py`              | 100   | `Path(__file__).resolve().parent.parent`      | `project_root` (repo root)  | FIXED (parent -> parent.parent) |
| `pipeline.py`            | 16224 | `...parent.parent` -> `load_config_json`      | `config.json` at repo root  | FIXED |
| `photometry_core.py`     | 6271  | `...parent.parent` `_REPO_ROOT_FOR_PROVENANCE`| git cwd + porcelain base    | FIXED |
| `citations.py`           | 18    | `...parent.parent` `_PROJECT_ROOT`            | `CITATIONS.bib` at repo root| FIXED |
| `hrd_colorfield.py`      | 46    | `...parent.parent` `_REPO_ROOT`               | git cwd for stamp           | FIXED |
| `photometry_report.py`   | 2399  | `...parent.parent / "img" / logo`             | `img/VYVAR_logo.png`        | FIXED |
| `params_registry.py`     | 25    | `...parent.parent / "dev" / "validation"`     | registry JSON               | FIXED (Phase A+B) |
| `psf_photometry.py`      | 656   | `...parent.parent / "dev" / "scripts"`        | diagnostic script path      | FIXED |
| `trust_flag.py`          | 12    | `...parent.parent` `ROOT`                     | `tmp/` output defaults      | FIXED |
| `photometry_report.py`   | 72    | `Path(reportlab.__file__)...`                 | reportlab fonts (3rd-party) | already-correct (module-of-dependency) |
| `simulate_night_run.py`  | 19    | `Path(__file__).resolve().parent` -> sys.path | add own dir (src_py) to path| already-correct (module-local bootstrap) |

`config.json` stays at repo root and is discovered from `src_py/config.py`
via `parent.parent`. Verified by `config-paths PASS` in the gate.

## git_dirty_code classifier change (T3 FIX B)

Import-relevant code root redefined from root `*.py` to `src_py/*.py` (plus
the root `app.py` shim). Everything under `dev/`, `tmp/`, `docs/` is scratch.

```python
def _is_import_relevant_py_path(path: str) -> bool:
    p = path.replace("\\", "/").lstrip("./")
    if not p.endswith(".py"):
        return False
    if p == "app.py":  # thin root Streamlit shim
        return True
    return p.startswith("src_py/")
```

Test coverage: `dev/tests/test_git_dirty_code_classifier.py` updated to assert
`src_py/pipeline.py` and `app.py` are import-relevant while root `pipeline.py`
and `dev/**` are not; `dev/tests/test_f431_labbe_provenance.py` fixture path
changed to `src_py/foo.py`. Both green.

Live confirmation from the Phase C run provenance:
`git_dirty_code = False`, `git_dirty_code_files = []`; the 5 pre-existing
untracked files (`dev/scripts/dy_peg_night_run_bvr.py`,
`dev/scripts/forensic_disc_ui_match2.py`, `dev/scripts/qatar8_night_run_v.py`,
`docs/VYVAR_CODE_AUDIT.md`, `docs/round2_figs/v0454_lc_vyvar.png`) were
correctly bucketed into `git_dirty_scratch_files`, NOT code.

## PHASE C - anchor gate (verbatim)

`python dev/scripts/session_baseline_check.py --full` (elapsed 2079s pipeline;
2338s total), headless `run_full_photometry_pipeline` against Anchor #3.

```
SESSION BASELINE CHECK (full)
------------------------------------------------------------------------
Check                        Status Detail
------------------------------------------------------------------------
git-branch                   PASS   main
git-head                     PASS   8f4d7b4
git-staged                   PASS   none
git-untracked-known          WARN   4 known untracked
git-untracked                WARN   dev/scripts/forensic_disc_ui_match2.py
git-origin-main              WARN   differs from origin/main (62410c8); consider git pull
config-paths                 PASS   all present
pytest                       PASS   903 passed, 19 skipped
ledger                       PASS   v1 14 items
ledger-todo                  WARN   VL-ANCHOR-424, VL-ANCHOR-DQ-430
full-provenance              PASS   anchor git_hash=10d610c0e79d...
full-pipeline                PASS   2079s -> tmp\session_baseline\20260717T113754Z
full-science-compare         PASS   n_lc=166 failures=0
full-provenance-hash         PASS   ed6630fb360685a7... (informational; git-bound, not cross-commit gate)
full-snapshot-sha-core       PASS   3d26f4692ac81fc5... n=333
full-photometry-sha-core     PASS   3d26f4692ac81fc5... n=333
full-photometry-sha-extended PASS   6420f1daa53a0d5d... n=499
full-counters-runtime        PASS   {"phase2a_empty_comp_drop": 1}
full-counters-meta           PASS   {"phase2a_empty_comp_drop": 1}
full-counters-expected       PASS   allowlisted {"phase2a_empty_comp_drop": 1} (structural empty-comp drops)
------------------------------------------------------------------------
OVERALL: PASS
```

Anchor #3 acceptance criteria:
- core SHA `3d26f4692ac81fc5...` (n=333) - MATCH (byte-identical incl. `err`).
- extended SHA `6420f1daa53a0d5d...` (n=499) - MATCH.
- census bands: `field_density.json` -> `n_stars_dao_raw = 2552`,
  `n_stars = 2842` (expected 2552 / 2842 / 169; the 2552 and 2842 bands
  confirmed directly). Byte-identical SHA implies all counts unchanged.
- identity p95 vs baseline 1.54 px: science output byte-identical, so the
  alignment/identity residuals are unchanged by construction.
- `git_dirty_code = false` in run provenance - CONFIRMED.
- except_fix_counters: `{phase2a_empty_comp_drop: 1}`, matches the draft_435
  allowlist (allowlist unchanged) - CONFIRMED zero beyond allowlist.

Verdict: **PASS**. Zero bytes of science output changed by the refactor.

## Milan UI smoke

CONFIRMED (2026-07-17). Milan launched the app from the repo root via the new
`app.py` shim (`streamlit run app.py`); Settings rendered including the new
Parameters tab, with the modified-counter showing 10. Report-opening is covered
by the PARAM-OWNERSHIP-WAVE-A STEP 4 report verification (regenerated report,
report-only, no photometry rerun). Additional end-to-end evidence: the
draft_000436 anchor-gate run. Milan ratified this as the Phase D UI-smoke
confirmation.

Note: launching the app surfaced a pre-existing UI defect (not a reorg
regression) - the location-picker auto-saved config.json on render
(`src_py/app.py:2037-2047`), rewriting the observer block Dablice(id=1) ->
Jirny(id=2). Value ratified by Milan (Jirny correct); the render side-effect is
fixed in the PARAM-OWNERSHIP-WAVE-A STEP 1 config-write guard.

## PHASE D - docs + push

Docs stamped (PARAM-OWNERSHIP-WAVE-A STEP 0): `CLAUDE.md` repository-layout
section; `docs/VYVAR_PROCESS.md` ritual paths + result-file location rule;
`docs/VYVAR_STATE.md` + `docs/VYVAR_JOURNAL.md` REPO-REORG arc entry with commit
hashes and the Phase C gate verdict.

Ratification commits: `cdd2277` (ledger anchor PASS record), `860ebf7`
(config.json Jirny observer block ratified).

Push: DEFERRED to PARAM-OWNERSHIP-WAVE-A STEP 5 (single gated push of the whole
local stack: REPO-REORG + audit + wave A), on Milan's explicit word.

## Errors (if any)

None blocking. Warnings are all pre-existing/expected: `ledger-todo`
(VL-ANCHOR-424, VL-ANCHOR-DQ-430 unrelated), `git-origin-main` differs
(commits not yet pushed - Phase D), known untracked scratch files.

## Files changed

See per-phase commits `c611353` (Phase A) and `8f4d7b4` (Phase B) above.
Pushed HEAD: PENDING (Phase D).
