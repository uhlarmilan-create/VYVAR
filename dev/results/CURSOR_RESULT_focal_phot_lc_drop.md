CURSOR RESULT - 2026-08-10

What I did
Part A: Removed EQUIPMENTS.FOCAL (schema + migration). Repointed all focal resolution to
TELESCOPE.FOCAL via get_telescope_focal_mm / resolve_focal_mm. Deleted get_equipment_focal_mm.
Removed FOCAL from Database Explorer EQUIPMENTS editor.

Part B: Dropped dead PHOTOMETRY_LIGHT_CURVE (CREATE removed; DROP on open via _drop_vestigial_tables).

## Output / findings

### Part A
- Call sites updated: param_resolver.resolve_focal_mm, photometry_core plate-scale path,
  pipeline._resolve_focal_mm_for_plate_scale, pipeline._enrich_calibration_metadata_from_header,
  database.get_combined_metadata.
- grep src_py: zero hits for get_equipment_focal_mm / EQUIPMENTS.FOCAL / equipment_focal.
- Migration: _drop_equipments_focal_column (RENAME/CREATE/COPY/DROP, PRAGMA foreign_keys OFF).
- Tests: test_equipments_focal_column_dropped_on_open, test_photometry_light_curve_table_dropped_on_open.

### Focal before/after (Milan guard -- report, do not ignore)
On live vyvar.sqlite3 before migration, 1 draft had mismatched equipment vs telescope focal:
  draft 438: ID_EQUIPMENTS=5 (FOCAL=1480), ID_TELESCOPE=1 (FOCAL=200).
After this change, resolved focal for that draft uses TELESCOPE.FOCAL (200 mm), not the stray
EQUIPMENTS duplicate (1480 mm). This is the intended fix for ASI533/M71 RC duplication cited in
the task brief -- but Milan should confirm draft 438's ID_TELESCOPE assignment is correct.

### Part B
- PHOTOMETRY_LIGHT_CURVE and indexes removed from schema init; dropped idempotently on DB open.

### Verification
- dev/tests/test_database_sqlite_threading.py: 6 passed.
- session_baseline_check.py --fast: see below.

## Errors (if any)

First migration attempt failed with FOREIGN KEY constraint failed; fixed with PRAGMA foreign_keys OFF
during EQUIPMENTS rebuild.

## Files changed

- src_py/database.py
- src_py/param_resolver.py
- src_py/pipeline.py
- src_py/photometry_core.py
- src_py/ui_database_explorer.py
- dev/tests/test_database_sqlite_threading.py
- dev/results/CURSOR_RESULT_focal_phot_lc_drop.md (this file)
