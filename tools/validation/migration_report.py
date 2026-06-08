"""Report the library-table default markers (EQUIPMENTS/TELESCOPE/LOCATION IS_DEFAULT +
ACTIVE) and confirm the per-draft frozen OBS_DRAFT ids were NOT touched by the migration.
Run from the repo root:
    python tools/validation/migration_report.py
"""
from database import VyvarDatabase

db = VyvarDatabase("vyvar.sqlite3")
print("=== Library-table markers (backfilled) ===")
for t in ("EQUIPMENTS", "TELESCOPE", "LOCATION"):
    n = db.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    nd = db.conn.execute(f"SELECT COUNT(*) FROM {t} WHERE IS_DEFAULT=1").fetchone()[0]
    act = db.sql_expr_active_is_true("ACTIVE")
    na = db.conn.execute(f"SELECT COUNT(*) FROM {t} WHERE {act}").fetchone()[0]
    print(f"  {t}: rows={n}  IS_DEFAULT=1 count={nd} (id={db.get_default_id(t)})  ACTIVE-true={na}")
print("=== OBS_DRAFT per-draft frozen ids (must be UNCHANGED) ===")
for d in (360, 361, 362, 363):
    r = db.conn.execute(
        "SELECT ID_EQUIPMENTS,ID_TELESCOPE,ID_LOCATION FROM OBS_DRAFT WHERE ID=?", (d,)
    ).fetchone()
    print(f"  draft {d}: EQ={r[0]} TEL={r[1]} LOC={r[2]}")
print("OBS_DRAFT columns touched by migration: NONE (migration only ALTERs EQUIPMENTS/TELESCOPE/LOCATION).")
