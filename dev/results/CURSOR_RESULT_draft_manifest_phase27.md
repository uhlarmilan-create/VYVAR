CURSOR RESULT - 2026-08-11 Phase 2.7 UI off tables

What I did
Repointed Database Explorer and finalization UI to read draft/observation/file data
from draft_manifest.json instead of SELECTs on OBS_DRAFT/OBS_FILES/OBSERVATION/SCANNING.

## Output / findings
- Added manifest UI helpers in draft_provenance.py:
  iter_draft_archive_dirs, obs_draft_row_from_manifest, collect_manifest_*,
  draft_scan_summary_from_manifest
- ui_database_explorer.py: OBS_DRAFT/OBS_FILES/OBSERVATION display from manifest;
  OBS_DRAFT edits save via apply_main_table_editor_save + _try_refresh_draft_manifest
- ui_finalization.py: scan summary, light count, location via manifest/accessors (no OBS_* SQL)
- Tests: dev/tests/test_manifest_ui_helpers.py (5 tests)
- --fast: OVERALL PASS (1294 passed, 27 skipped)

## Errors (if any)
None.

## Files changed
- src_py/draft_provenance.py
- src_py/ui_database_explorer.py
- src_py/ui_finalization.py
- dev/tests/test_manifest_ui_helpers.py
- dev/results/CURSOR_RESULT_draft_manifest_phase27.md
