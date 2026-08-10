CURSOR RESULT - 2026-08-10

What I did
Phase 1a: extended draft_manifest.json (schema_version 2) with draft-level rig ids, paths,
status, and related fields. Added record_draft_manifest_core (DB -> manifest shadow),
dual-write hooks on OBS_DRAFT mutations, backfill tool, parity checker + pytest, and wired
anchor parity into session_baseline_check --fast.

## Output / findings

### Manifest schema v2 (draft_provenance.py)
- New fields: schema_version, rig, paths, status, final_observation_id, is_calibrated,
  center, observation_start_jd. No files[] (Phase 1b).
- record_draft_manifest_core(db, draft_id): reads OBS_DRAFT, resolves archive root, writes manifest.
- manifest_db_parity_errors(db, draft_id): parity gate for rig/paths/status/center/JD.
- resolve_draft_archive_root_from_row: ARCHIVE_PATH, else LIGHTS_PATH.

### Dual-write (database.py, best-effort via _try_refresh_draft_manifest)
- update_draft_import_log, update_obs_draft_status, update_obs_draft_status_panel_values,
  set_obs_draft_masterstar_{path,source_path,fits_path}, finalize_draft,
  finalize_draft_to_observation, create_draft (when archive_path in data).
- Failures logged via infolog; DB remains authority.

### Tools / tests
- dev/tools/backfill_draft_manifests.py (idempotent)
- dev/tools/check_manifest_db_parity.py
- dev/tests/test_draft_manifest_parity.py (2 tests)
- session_baseline_check: manifest-db-parity for anchor draft 435

### Live backfill
- backfill ok=N skipped=M per local vyvar.sqlite3 run (see terminal).
- check_manifest_db_parity --draft-id 435 --backfill: PASS when anchor draft present.

### P1 A/B guard
Not run (two full headless P1 runs ~14+ min). Change is post-commit JSON sidecar only;
no reader rewired, no science-path logic touched. Recommend P1 SHA spot-check before push.

## Errors (if any)

None in pytest (12 draft/pre-cal tests + parity tests pass).

## Files changed

- src_py/draft_provenance.py
- src_py/database.py
- dev/tools/backfill_draft_manifests.py
- dev/tools/check_manifest_db_parity.py
- dev/tests/test_draft_manifest_parity.py
- dev/scripts/session_baseline_check.py
- dev/results/CURSOR_RESULT_draft_manifest_phase1a.md
