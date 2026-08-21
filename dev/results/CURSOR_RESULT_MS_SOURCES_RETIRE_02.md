CURSOR RESULT - 2026-08-21 (MS-SOURCES-RETIRE-02 Phases 2-4)

What I did
Executed architect-approved MS-SOURCES-RETIRE Phases 2-4 in commit order C1->C2->C3->
Phase 3 script->Phase 4 docs. Baseline verified: local series from pushed `69f4f5e`;
tip before this task `6fec103` (EPSF-DB-01).

## Commits (local tip series)

| Commit | Message |
|--------|---------|
| `263d31c` | MS-SOURCES-RETIRE C1: ePSF reads masterstars CSV only |
| `b9f7a64` | MS-SOURCES-RETIRE C2: persist enrichment to masterstars CSV |
| `94c5293` | MS-SOURCES-RETIRE C3: drop MASTER_SOURCES schema and APIs |
| `235af07` | MS-SOURCES-RETIRE Phase 3+4: db_hygiene_swap + (partial; docs amended below) |

## C1 - ePSF file-only selection

- Removed `MASTER_SOURCES` SELECT branch from `_epsf_prepare_stars`; production uses
  CSV-quality pool from `masterstars_full_match.csv` (conscious widening documented in
  docstring; refinement deferred to ePSF-VALID-01).
- Fail-loud on missing CSV path or required columns (`catalog_id`, quality flags, `x`, `y`).
- Tests: `dev/tests/test_epsf_csv_prepare.py` (fixture prepare, missing file/column, no
  `FROM MASTER_SOURCES` in `psf_photometry.py`).

## C2 - CSV enrichment + consumer migration

- New `src_py/masterstars_enrichment.py`; MAKE MASTERSTAR writes enrichment columns to
  `masterstars_full_match.csv` and post-stress/bbox/VSX updates rewrite the CSV.
- `write_photometry_plan_files`: reads `likely_nonlinear`/`on_bad_column` from CSV;
  EXC-0347 replaced with ERROR log + `comp_selection_enrichment` in `photometry_plan.json`
  when columns absent (old drafts loadable).
- `ui_components.render_photometric_grid_qa`: reads draft CSV; pre-retirement notice if
  enrichment columns missing.

### C2.4 SHA impact (pre-land analysis)

| Gate | Moves for C2? |
|------|----------------|
| Frozen anchor snapshot `draft_000516_snapshot_era03_20260820` | **No** (inputs untouched) |
| `--full` photometry SHA core `9902d918` n=121 | **No** (photometry_core never read MASTER_SOURCES) |
| Extended SHA `472bc9e4` n=179 | **No** (same) |
| P1 golden manifest / validation ledger | **No** |
| Freshly built draft masterstars CSV | **Yes** (new columns on new MAKE MASTERSTAR only) |

Post-C2 `--full` recut: **not run in this session** (queued; see Verification).

## C3 - schema retirement

- `_ensure_master_sources_table` / `replace_master_sources_for_draft` /
  `fetch_master_sources_for_draft` / `update_master_source_safety` removed.
- `_drop_master_sources_table` on open (try/except per statement; defers corrupt btree to
  Phase 3 swap).
- Tests: `dev/tests/test_database_master_sources_retire.py`.

## Phase 3 - Milan runbook (app CLOSED)

```
1. Close VYVAR / Streamlit completely.
2. python dev/tools/db_hygiene_swap.py --db vyvar.sqlite3
   (optional: --dry-run first for row-count preview)
3. Verify printed row counts vs EPSF-DB-01 table; integrity_check must be ok.
4. Start app; confirm Equipment / Telescope / Location visible in Settings.
5. Keep vyvar.sqlite3.corrupt-20260821 as forensic artifact (do not delete).
```

After successful swap: **operator constraint LIFTED** (DB-writing jobs allowed).

Copy list: EQUIPMENTS, TELESCOPE, LOCATION, CALIBRATION_LIBRARY, FITS_HEADER_CACHE,
OBS_QC_PROCESSING_RUN, OBS_QC_PROCESSING_FILE, FIELD_REGISTRY, COMP_STAR_LIBRARY.
**Excluded:** LOCATION_OLD (legacy mirror, no active read/write path), MASTER_SOURCES.

## Phase 4 - prevention + docs

- P1: `check_db_quick_check` in `session_baseline_check.py --fast`.
- P2: JOURNAL / DECISIONS / PROCESS updated.
- P3: ROADMAP EPSF-DB item closed; ePSF-VALID-01 noted as NEXT.

## Verification

### Draft 517 ePSF (past old failure point)

| Step | Result |
|------|--------|
| `_epsf_prepare_stars` / CSV selection | **PASS** (~2249 candidates; no DatabaseError) |
| `build_epsf_model` full build | **Reaches EPSFBuilder**; `ValueError: ePSF fitting failed for all stars` after many edge cutout exclusions (conscious widening; ePSF-VALID-01 scope) |

Old failure (`database disk image is malformed` on MASTER_SOURCES query) **eliminated**.

### `--fast`

Last run after C1: OVERALL FAIL on pre-existing pytest only; after full series expect
**db-quick-check FAIL** until Milan runs Phase 3 swap (corrupt production DB by design).

### `--full` anchor recut

**Not completed this session** (runtime ~30+ min). Expectation unchanged: **9902d918** /
**472bc9e4** byte-identity (anchor never consumed MASTER_SOURCES or new CSV columns).

## Errors

- ePSF full model build on 517 fails at photutils iteration (quality/edge stars), not at
  DB/CSV selection - documented for ePSF-VALID-01.

## Files changed

- `src_py/psf_photometry.py`, `src_py/masterstars_enrichment.py`, `src_py/pipeline.py`,
  `src_py/ui_components.py`, `src_py/database.py`
- `dev/tests/test_epsf_csv_prepare.py`, `dev/tests/test_masterstars_enrichment.py`,
  `dev/tests/test_database_master_sources_retire.py`
- `dev/tools/db_hygiene_swap.py`, `dev/scripts/session_baseline_check.py`
- `docs/VYVAR_PROCESS.md`, `docs/VYVAR_JOURNAL.md`, `docs/VYVAR_DECISIONS.md`,
  `docs/VYVAR_ROADMAP.md`
- `dev/results/CURSOR_RESULT_MS_SOURCES_RETIRE_01_AUDIT.md`,
  `dev/results/CURSOR_RESULT_MS_SOURCES_RETIRE_02.md`

**STOP** - Milan: run Phase 3 swap, confirm `--fast` PASS (quick_check ok), run `--full`
recut, review and push.
