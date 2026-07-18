CURSOR RESULT — DOCS-FIX-ARC1 — 2026-07-18

What I did
==========
Executed DOCS-FIX-ARC1 (unblockers, stale-doc fixes, README rewrite, LICENSE) as six
separate commits on `main`, one per work package, on top of the DOCS-REVISION-RECON audit
commit (`0e67786`). Full pytest and the `--fast` anchor gate are green. The stack (recon
audit + arc 1) is HELD — not pushed — pending Milan's word.

Decision basis: DOCS_REVISION_RECON.md matrix + Milan's decisions A–E.

## Per-WP commits

| WP | Commit | Summary |
|----|--------|---------|
| WP1 | `bfac696` | `build(deps)`: pin numpy/astropy/photutils to anchor majors; add matplotlib + scikit-image |
| WP2 | `3e6ecfd` | `fix(gaia)`: make `build_gaia_catalog.py` root detection src_py-aware |
| WP3 | `e4074d6` | `docs(readme)`: rewrite README as GitHub front door + Czech twin |
| WP4 | `e03cd06` | `docs(stale)`: path/number fixes across dev docs (REPO-REORG + current anchor) |
| WP5 | `385cadf` | `docs(archive)`: move `config_schema.md` to `dev/results/` |
| WP6 | `df79775` | `docs(license)`: add proprietary LICENSE + record license/visibility decision |

### WP1 — requirements.txt fix + pin
- Added explicit `matplotlib` and `scikit-image` (previously only transitive).
- Pinned the three anchor-critical libs with **compatible-range** bounds, not exact pins:
  `numpy>=2.4,<3`, `astropy>=7.2,<8`, `photutils>=2.3,<3`.
  **Why ranges, not exact:** byte-identity risk lives across a *major* bump; holding the
  major while allowing patch/minor lets security fixes in without blocking installs. Current
  env matches the anchor versions exactly (numpy 2.4.3 / astropy 7.2.0 / photutils 2.3.0).
- Optional extras (`pyarrow`, `sep-pjw`, `cupy`) documented as a commented section; `pyraf`
  noted as intentionally omitted (Linux-only, guarded, LC cross-val only).
- Smoke: `pip check` -> **No broken requirements found.** (Fresh-venv resolution deferred to
  the Lenovo test per task.)

### WP2 — build_gaia_catalog.py root detection
- `_find_vyvar_root` now looks for `src_py/gaia_catalog_id.py` (with a legacy flat-layout
  fallback) and puts both `src_py/` and the repo root on `sys.path`.
- Functional check: `python GAIA_DR3/build_gaia_catalog.py --help` runs; the top-level
  `from gaia_catalog_id import normalize_gaia_source_id` import resolves; detected
  root=`C:\ASTRO\python\VYVAR`, src=`...\src_py`.

### WP3 — README rewrite + Czech twin
- `README.md` (EN) rewritten as the GitHub front door: 2-paragraph "what VYVAR is" in
  observer language, capability list (calibration -> detection/Gaia -> ensemble photometry
  -> trust -> AAVSO/reports), honest numbers (963 tests, 269 parameters, anchor-discipline
  one-liner), screenshot placeholder (`img/vyvar_ui.png`), INSTALL.md pointer (lands arc 2),
  docs index, correct dev basics (pytest via `pyproject.toml`, `dev/scripts` paths), and a
  proprietary license note matching WP6.
- `README_CZ.md` added as the Czech twin, cross-linked from the top of both files (taxonomy A).
- README/README_CZ UTF-8 exemption to the ASCII rule recorded as a one-liner in
  `docs/VYVAR_PROCESS.md`.

### WP4 — stale dev-doc UPDATE pass (pure path/number)
- `CLAUDE.md`: `304` -> `269` registered params (249 persisted); added `README.md`,
  `README_CZ.md`, `LICENSE` to the root-file list.
- `CHANGELOG.md` [Unreleased]: appended CONFIG-HUMAN-EDIT, README front door, WAVE-B
  reduction, and REPO-REORG entries; noted older `scripts/` citations predate the move.
- `VYVAR_STATE.md`: current test line `852 / 15 (tests/)` -> `963 / 19 (dev/tests/)`.
- `VYVAR_RUNBOOK.md`, `VYVAR_CLAUDE_OPERATING_PRINCIPLES.md`, `VYVAR_ROADMAP.md`:
  `scripts/session_baseline_check.py` -> `dev/scripts/...`; current anchor `draft_424` ->
  `draft_435` (snapshot `draft_000435_snapshot_skysurface_20260716`, SHAs `3d26f469` /
  `6420f1da`, ledger `VL-ANCHOR-WCSINV`) verified against the live script constants.
- `VYVAR_VALIDATION.md`: `tests/validation/` -> `dev/tests/validation/`.
- `VYVAR_CALIBRATION.md`: dead `VYVAR_FULL_AUDIT_LEDGER.md` links ->
  `dev/results/VYVAR_FULL_AUDIT_LEDGER.md`.
- `VYVAR_JOURNAL.md` left untouched (matrix: OK — history).

### WP5 — archives
- `git mv docs/config_schema.md dev/results/config_schema.md` (rename, 100%). Tombstone in
  the commit message points to `docs/VYVAR_PARAMS.md` + the CONFIG_GUIDEs as the authoritative
  parameter references.
- Updated the surviving living-doc references to the archived path: `CLAUDE.md` docs list
  (removed), `VYVAR_DECISIONS.md:1362` and `VYVAR_SIMPLE_DIFFERENTIAL_SPEC.md:25` (retargeted
  to `dev/results/config_schema.md (archived)`).
- **docs-layout guard: green** (docs/ still markdown-only, no subdirs, no CURSOR_* files).

**Superseded-spec reclassification (decision E).** The recon matrix had *proposed* archiving
eight 2026-06 design/decision specs. On closer inspection under decision E ("archive ONLY
specs marked superseded; live-behavior specs stay"), none qualifies as superseded, so **none
were archived**:

| Spec | Why it stays |
|------|--------------|
| `VYVAR_TRUST_CHECKSTAR_HARDENING_SPEC.md` | test-backed (`dev/tests/test_trust_checkstar_hardening.py`) |
| `VYVAR_NEIGHBOR_SUB_DESIGN.md` | referenced by `dev/tests/validation/recover.py` |
| `VYVAR_COMP_DEGRADATION_SPEC.md` | grounding cited by DECISIONS/STATE/ROADMAP |
| `VYVAR_LC_QUALITY_SHORT_BASELINE_SPEC.md` | grounding cited by DECISIONS/STATE/ROADMAP; pending workstream |
| `VYVAR_CANONICAL_COMBINATION_LOGIC.md` | grounding cited by CALIBRATION/ROADMAP/STATE (CONDITIONAL HOLD) |
| `VYVAR_CHECKSTAR_SELECTION_SPEC.md` | grounding cited by DECISIONS/STATE/ROADMAP |
| `VYVAR_COMP_FLOOR_POLICY_SPEC.md` | grounding cited by DECISIONS/STATE/ROADMAP |
| `VYVAR_SIMPLE_DIFFERENTIAL_SPEC.md` | Workstream A landed / B pending (live pending design) |

**Only `config_schema.md` was archived.** If Milan wants a broader spec sweep, he can name the
specific specs to move — flagged here rather than archiving live grounding docs and breaking
the DECISION_GROUNDING_RULE citation chain.

### WP6 — LICENSE
- Added root `LICENSE`: proprietary, all rights reserved (Copyright (c) 2026 Milan Uhlár;
  no use/copying/modification/distribution without written permission; no warranty).
- README/README_CZ license sections match the LICENSE text.
- `VYVAR_DECISIONS.md`: one new entry records the license choice + decision C (private repo
  applied on remote; compiled-library distribution deferred/recorded-not-adopted).

## README rendered-preview note
`README.md` is standard GitHub-Flavored Markdown: H1 title, an italic CZ-twin link line at the
very top, an image line (`img/vyvar_ui.png` — placeholder, file not yet committed, so GitHub
will show broken-image alt text until a screenshot lands), two prose paragraphs, a bullet
capability list, a "Project status" bullet block, fenced `bash` install blocks, and three
markdown tables (docs index in README; nothing exotic). `README_CZ.md` mirrors the structure
with UTF-8 diacritics. Both render cleanly in a standard Markdown previewer; no raw HTML.

## Gates
- Full `pytest -q`: **963 passed, 19 skipped** (308.79 s).
- `dev/scripts/session_baseline_check.py --fast`: **OVERALL PASS** (HEAD `df79775`; pytest
  963/19; WARN-only items: 2 known untracked helper scripts, branch ahead of origin because
  unpushed, ledger-todo VL-ANCHOR-424/VL-ANCHOR-DQ-430 — all pre-existing).
- docs-layout / params parity+freshness guards: **green** after every WP.

## HOLD
Not pushed. The stack to push on Milan's word (oldest first):
`0e67786` (recon audit) -> `bfac696` -> `3e6ecfd` -> `e4074d6` -> `e03cd06` -> `385cadf`
-> `df79775`.

## Files changed
See per-WP commit list above. New files: `README_CZ.md`, `LICENSE`,
`dev/results/CURSOR_RESULT_docs_fix_arc1.md` (this file). Renamed:
`docs/config_schema.md` -> `dev/results/config_schema.md`.

## Next (separate task, after Milan reviews README)
INSTALL arc: installer script (PowerShell + bash twin), INSTALL.md EN+CZ, machine-local config
paths block, catalog copy-vs-build flow, first-run checklist, Lenovo stranger-test protocol.
