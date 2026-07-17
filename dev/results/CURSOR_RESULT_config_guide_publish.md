CURSOR RESULT - 2026-07-17

# CONFIG-GUIDE-PUBLISH - publish the config.json guides to docs/ and push

Baseline origin/main `4882f5c`. Milan authorized the push ("save it to GitHub in the
docs directory").

## What I did
Published Claude's two hand-authored config.json guides (EN + CZ, every one of the 304
parameters: type, source, layman explanation, where used) into `docs/` verbatim, added
lightweight cross-links, kept the full suite and the params freshness guard green, and
pushed.

## STEP 1 - add the files
- Copied verbatim (byte-identical, verified by file hash) into `docs/`:
  - `docs/VYVAR_CONFIG_GUIDE_EN.md`
  - `docs/VYVAR_CONFIG_GUIDE_CZ.md`
- Sanity checks (both files): **304 unique parameter rows**, **pure ASCII** (0 non-ASCII
  bytes; CZ follows the "cestina bez diakritiky" convention). The 304 parameter names in
  each guide match the registry (`dev/validation/params_registry.json`) set **exactly**
  (no extras, none missing).
- `dev/tests/test_docs_layout.py`: 4 passed (plain `.md`, no subdirs, no `CURSOR_*`).
- Commit `34ab16f docs(config): add VYVAR configuration guide EN+CZ (304 params)`.

## STEP 2 - cross-links
- `dev/tools/gen_params_md.py`: added one pointer line to the generated header
  ("Human-readable guide: `VYVAR_CONFIG_GUIDE_EN.md` / `VYVAR_CONFIG_GUIDE_CZ.md` ...")
  and regenerated `docs/VYVAR_PARAMS.md` (generated file, not hand-edited). Params
  freshness guard `dev/tests/test_params_registry.py`: 9 passed.
- `CLAUDE.md` Project-docs section: one line mentioning the two guides.
- Commit `55fef5e docs(config): cross-link the configuration guides`.

## STEP 3 - push
- Final full pytest (tree with both commits present): **928 passed, 19 skipped, 31 warnings**.
- Pushed HEAD + `git log --oneline 4882f5c..HEAD` recorded in the PUSH section below.

## Maintenance note
The guides are **hand-authored** from the registry + parameter source audit as of
2026-07-17. The freshness guard (which protects `VYVAR_PARAMS.md`) does **NOT** cover
them: when parameters change, `VYVAR_CONFIG_GUIDE_EN.md` / `_CZ.md` must be updated
manually. Proposed future arc (Milan to decide): port the per-key plain-language
explanations into the registry as `help_en` / `help_cz` fields so the guides become
generated and freshness-guarded like `VYVAR_PARAMS.md`.

## Errors (if any)
None.

## Files changed
Added `docs/VYVAR_CONFIG_GUIDE_EN.md`, `docs/VYVAR_CONFIG_GUIDE_CZ.md`; edited
`dev/tools/gen_params_md.py`, `docs/VYVAR_PARAMS.md` (regenerated), `CLAUDE.md`.
No science/numeric change (documentation + doc plumbing only).
