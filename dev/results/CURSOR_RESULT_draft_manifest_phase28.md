CURSOR RESULT - 2026-08-11 11:45 UTC+2

What I did
Phase 2.8 manifest-direct write inversion: draft creation, file ingest, QC/calib updates,
status/path/masterstar/finalize now write draft_manifest.json directly. Retired DB mirror
(record_draft_manifest_core no-op, parity check simplified). Repointed pipeline MASTERSTAR
WCS/stars update to manifest. Updated tests for manifest-direct model. Committed and pushed.

## Output / findings

**Commit:** `911adfb` Phase 2.8: manifest-direct draft writes (sole store for OBS_DRAFT/OBS_FILES)
**Pushed HEAD:** `911adfb96442fd6a2e8ecdf49f2a18af2132e46d` on `main`

**Gates (post-2.8):**
- `--fast`: OVERALL PASS (1292 passed, 27 skipped)
- P1 headless: core_sha=`24820ee282e5c03020e16757201bad624050d0a4bc78e3b137584f23debe517b` (n=325) -- byte-identical to pre-2.8 baseline
- manifest-db-parity: PASS draft_id=435 (backfill ok=55 earlier in arc)

**Write-path changes:**
- `create_draft` -> `create_draft_manifest` (allocate id from filesystem, no OBS_DRAFT INSERT)
- `insert_draft_files`, QC/calib/reject updaters -> manifest `files[]`
- `update_draft_import_log`, `update_obs_draft_*`, `finalize_draft_to_observation` -> manifest
- `apply_main_table_editor_save` OBS_DRAFT branch -> manifest patch
- `_update_masterstar_obs_file_status` -> manifest inspection wcs/stars

**Phase 3 NOT started** -- grep still shows live SQL for all four tables:
- SCANNING: database find_or_create/insert, ui_database_explorer SELECT, FINAL_DATA JOIN
- OBSERVATION: finalize_draft legacy path, FINAL_DATA, count_references
- OBS_FILES: CREATE/migrations, maintenance DELETE, legacy finalize paths, backfill reads
- OBS_DRAFT: CREATE/migrations, maintenance, legacy finalize, backfill reads

## Errors (if any)
None blocking 2.8 commit.

## Files changed
- src_py/draft_provenance.py, database.py, pipeline.py
- dev/tests (manifest test suite), dev/tools/backfill_draft_manifests.py, session_baseline_check.py
- Commit `911adfb`
