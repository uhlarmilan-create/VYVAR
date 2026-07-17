# VYVAR — Claude (architect) notes

## Project docs

State and process docs live under `docs/`:

- `docs/VYVAR_STATE.md` — entry point (current snapshot)
- `docs/VYVAR_ROADMAP.md`, `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_DECISIONS.md`
- `docs/VYVAR_PROCESS.md`, `docs/VYVAR_PARAMS.md`, `docs/config_schema.md`
- `docs/VYVAR_CONFIG_GUIDE_EN.md` / `docs/VYVAR_CONFIG_GUIDE_CZ.md` — plain-language guide to all 304 config.json parameters (EN + CZ)
- `docs/VYVAR_CLAUDE_OPERATING_PRINCIPLES.md` — Claude operating charter (session-init required read)

**Session init:** read STATE, ROADMAP, latest JOURNAL, PROCESS, and CLAUDE_OPERATING_PRINCIPLES
(the last governs how Claude reasons and answers; it is not optional context).

## Repository layout (post REPO-REORG, 2026-07-17)

- `src_py/` -- ALL VYVAR Python modules (production code). Imports stay FLAT
  (`from pipeline import ...`); path visibility is solved at entry points only.
  The root `app.py` is a thin shim that puts `src_py/` on `sys.path` and runs
  `src_py/app.py`, so `streamlit run app.py` still works from the repo root.
- `dev/` -- internal development material, git-tracked: `dev/tests/`,
  `dev/tools/`, `dev/validation/`, `dev/scripts/`, `dev/sandbox/`,
  `dev/orchestrator/`, and `dev/results/` for ALL `CURSOR_RESULT_*.md` /
  `CURSOR_TASK_*.md` working documents (current and future).
- Root keeps only: `config.json`, `pyproject.toml`, `requirements.txt`,
  `.gitignore`, `CLAUDE.md`, `CHANGELOG.md`, `CITATIONS.bib`, and the `app.py` shim.
- `tmp/` -- gitignored disposable scratch (helpers, one-off harnesses, outputs).
  Nothing tracked lives here.
- UNTOUCHED data roots: `GAIA_DR3/`, `Archive/`, `docs/`, `exoplanets/`,
  `CalibrationLibrary/`, `VSX/`, `img/`.

Layout stamped by the REPO-REORG arc (commits `c611353` dev/ move, `8f4d7b4`
src_py/ move); Anchor #3 byte-identical gate PASS on `8f4d7b4`.

## Orchestrator workflow

When CURSOR_TASK.md appears or is updated:

1. Read it completely
2. Execute the task
3. Write results to CURSOR_RESULT.md in this format:

```
CURSOR RESULT — <datetime>

What I did
<concise summary>

## Output / findings
<key data, file paths, metrics>

## Errors (if any)
<error messages>

## Files changed
<list of modified files + commit hash if committed>
```

4. Press Enter in the orchestrator terminal
