CURSOR RESULT - 2026-08-11

What I did
Routed OBS_QC QC-table rebuild through _rebuild_table_safely; added orphan *_OLD self-heal on
DB open; guard only rebuilds when OBS_DRAFT FK still present.

## Output / findings
- `_heal_qc_orphan_old_tables`: drops orphan RUN_OLD/FILE_OLD when main table exists; renames
  *_OLD back to main when only *_OLD survived a crash
- `_rebuild_qc_tables_without_obs_draft_fk` uses safe helper for RUN and FILE (FILE rebuilt when
  RUN migrates or FILE has legacy OBS_DRAFT FK)
- Added OBS_QC_PROCESSING_RUN/FILE to `_REBUILD_MIGRATION_TABLES`
- Grep: no other hand-rolled RENAME TO *_OLD outside `_rebuild_table_safely`
- Tests: orphan heal + legacy FK migration (`test_database_sqlite_threading.py`)
- `--fast`: OVERALL PASS (1286 passed, 27 skipped)

## Milan immediate unstick (before pull)
With app closed, backup vyvar.sqlite3, then:
  DROP TABLE IF EXISTS OBS_QC_PROCESSING_RUN_OLD;
  DROP TABLE IF EXISTS OBS_QC_PROCESSING_FILE_OLD;
(only when OBS_QC_PROCESSING_RUN / _FILE exist with their rows)

After pull: next app open self-heals automatically.

## Errors (if any)
None.

## Files changed
- src_py/database.py
- dev/tests/test_database_sqlite_threading.py
