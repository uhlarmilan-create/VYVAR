CURSOR RESULT — 2026-06-08 19:18 UTC

What I did
Completed pre-Cursor root cleanup phase: created gitignored `sandbox/` + `tmp/`, moved state docs to `docs/`, relocated scratch artifacts off tracked root, updated CLAUDE/orchestrator/generators/script default paths, consolidated GAIA build scripts (B1/B2/A), verified import + pytest.

## Output / findings
- **Structure:** `docs/VYVAR_*.md` + `docs/config_schema.md`; scratch → `tmp/`; helpers → `sandbox/` (`_merge_vyvar_state.py`, `variables.py`, `tools/`)
- **Kept in root:** `orchestrator/`, `CURSOR_RESULT.md`, `ui_photometry_results.py`, `ui_suspected_lightcurves.py`, all core `.py` modules
- **`.gitignore`:** `/sandbox/`, `/tmp/`
- **Verification:** `python -c "import app"` OK; **174 passed, 6 skipped** (`pytest tests/`)

## Errors (if any)
None.

## Files changed
See commit (pending push).
