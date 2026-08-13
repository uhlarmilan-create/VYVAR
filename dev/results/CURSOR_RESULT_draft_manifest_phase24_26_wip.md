CURSOR RESULT - 2026-08-11 Phase 2.4-2.6 WIP (2.7/2.8 NOT STARTED)

What I did
Implemented manifest-first overlays and reader consolidation for steps 2.4-2.6.
Steps 2.7 (UI off tables) and 2.8 (DROP) are NOT done -- tables still exist and
UI still reads SQL directly.

## Step 2.4 -- OBS_DRAFT core fields (DONE in code, not committed)
- `apply_manifest_core_to_draft_row()` overlays paths, status, center, JD,
  is_calibrated, final_observation_id (+ existing rig overlay)
- `fetch_obs_draft_by_id` uses full core overlay
- `get_obs_draft_masterstar_*` route through fetch_obs_draft_by_id
- `time_utils.resolve_target_coordinates` uses fetch_obs_draft_by_id for center
- `sigma_budget` uses get_draft_telescope_id / get_draft_location_id (no OBS_DRAFT JOIN)
- Test: `dev/tests/test_manifest_core_overlay.py`

## Step 2.5 -- OBS_FILES files[] (DONE in code, not committed)
- `light_rows_from_manifest()`, `_manifest_entry_to_obs_file_row()`, `obs_file_id` in manifest
- `fetch_draft_light_rows_for_quality` manifest-first + `_fetch_draft_light_rows_raw` fallback
- `fetch_light_file_paths_not_rejected_for_draft`, `fetch_draft_scanning_ids` via manifest rows
- `psf_photometry._median_fwhm_obs_files` via fetch_draft_light_rows_for_quality
- `pipeline.py` preprocess filter reads via fetch_draft_light_rows_for_quality (writes stay SQL)

## Step 2.6 -- SCANNING header reads (DONE in code, not committed)
- `photometry_core` binning from first light FITS XBINNING (not SCANNING table)
- `photometry_core` frame NAXIS from first light FITS (not OBS_FILES JOIN SCANNING)
- `get_draft_scanning_id()` accessor added

## Remaining non-UI reads (grep -- NOT zero yet)
- `database.py` writers/finalize/maintenance (expected until 2.8)
- `draft_provenance.py` writer/parity raw (expected until 2.8)
- `pipeline.py` UPDATE OBS_FILES (writers)
- `database.py` fetch_light_file_paths_not_rejected_for_observation (OBSERVATION path)
- `ui_database_explorer.py`, `ui_finalization.py` (step 2.7)

## Step 2.7 -- NOT STARTED
Database Explorer + ui_finalization still SELECT from all four tables.

## Step 2.8 -- NOT STARTED
No DROP TABLE statements added. CREATE/migration code unchanged.

## Tests
- Manifest suite + null-island + sigma_a2: PASS locally
- Full pytest: 1289 passed after sigma_a2 mock fix (re-run --fast recommended)

## Gates NOT yet run for this increment
- Local P1 A/B for science-path steps (2.4 center, 2.5 reject filter)
- Separate commits per step (awaiting Milan)

## Transition note
Raw external SQL edits bypass manifest refresh; recovery = backfill_draft_manifests.py.
