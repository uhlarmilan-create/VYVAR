CURSOR RESULT - 2026-08-10

What I did
Fixed Database Explorer crash caused by pandas `read_sql_query` calling `con.cursor()` on
`ThreadSafeSQLiteConnection`, which intentionally has no bare cursor passthrough. Added
`description` to `_LockedCursor`, introduced `_read_df()` helper using the locked execute/fetch
path, and replaced all nine `read_sql_query` call sites in `ui_database_explorer.py`. Added
unit test `test_read_df_via_thread_safe_connection`.

## Output / findings

- Root cause: wrapper exposes only execute/executemany/executescript/commit/rollback/close;
  pandas DBAPI path requires `cursor()`.
- Fix: `_read_df(conn, sql, params=())` reads column names from `cur.description` before
  `fetchall()` (which releases the RLock).
- Grep: only `ui_database_explorer.py` used `pd.read_sql*` on `db.conn` in `src_py/`.
- Live DB smoke (all 15 tables via `_read_df`): PASS, including empty
  `PHOTOMETRY_LIGHT_CURVE` and `COMP_STAR_LIBRARY`.
- OBS_FILES param filters: OBS and DRAFT paths verified (DRAFT:406 -> 322 rows).
- `session_baseline_check.py --fast`: OVERALL FAIL due to pre-existing
  `test_flow_doc_config_facts` (vsx_out_of_scope_types); pytest 1268 passed (+1 new test),
  science/config checks PASS.

## Errors (if any)

None from this fix. Pre-existing pytest failure unrelated to UI change.

## Files changed

- `src_py/database.py` - `_LockedCursor.description` property
- `src_py/ui_database_explorer.py` - `_read_df()` helper; 9 call-site replacements
- `dev/tests/test_database_sqlite_threading.py` - `test_read_df_via_thread_safe_connection`
- `dev/results/CURSOR_RESULT_database_explorer_read_df.md` (this file)
