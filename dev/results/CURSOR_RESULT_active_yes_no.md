CURSOR RESULT - 2026-08-10

What I did
Step 2: ACTIVE on EQUIPMENTS/TELESCOPE/LOCATION is physically stored as text YES/NO. Added
normalize_active_text (separate from normalize_active_db_value used by IS_DEFAULT). Schema migration
rebuilds TELESCOPE/LOCATION INTEGER ACTIVE to TEXT; EQUIPMENTS normalized in place. Updated SQL
predicates, soft-delete, Database Explorer editor, and picker filters. ROADMAP note for
FIELD_REGISTRY/COMP_STAR_LIBRARY retained as FUTURE multi-night infrastructure.

## Output / findings

### Schema
- normalize_active_text(raw) -> YES|NO via normalize_active_db_value (IS_DEFAULT path unchanged).
- _normalize_active_columns_to_text runs on DB open (idempotent).
- TELESCOPE/LOCATION rebuild uses PRAGMA foreign_keys=OFF inside executescript (FK-safe).

### SQL / code
- sql_expr_active_is_true: text YES/NO predicate (no numeric ACTIVE literals remain in src_py).
- Soft-delete: SET ACTIVE = 'NO'.
- _coerce_sql_param ACTIVE: always normalize_active_text.
- Removed _normalize_telescope_active_to_binary.

### UI
- Database Explorer: ACTIVE shown/edited as YES/NO Selectbox (all three tables); IS_DEFAULT unchanged.

### Picker verify (live vyvar.sqlite3)
- equipments: 5 total, 3 active
- telescopes: 8 total, 7 active
- DISTINCT ACTIVE values: YES/NO only (TEXT columns)

### ROADMAP
- TODO-GS8 bullet: FIELD_REGISTRY + COMP_STAR_LIBRARY retained as idle FUTURE schema.

## Errors (if any)

Initial live DB migration failed FK until PRAGMA foreign_keys=OFF moved inside executescript.

## Files changed

- src_py/database.py
- src_py/ui_database_explorer.py
- docs/VYVAR_ROADMAP.md
- dev/tests/test_database_sqlite_threading.py
- dev/tests/test_database_fk_draft.py
- dev/tests/test_run_vyvar_fk_milan_state.py
- dev/results/CURSOR_RESULT_active_yes_no.md (this file)

## Push gate (Milan/data, outside this task)

Do not push until config.json vsx_out_of_scope_types drift cleared and draft 438 ID_TELESCOPE corrected.
