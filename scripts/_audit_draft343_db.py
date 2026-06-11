"""Read-only DB audit for draft_343 pre-run."""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
db = sqlite3.connect(ROOT / "vyvar.sqlite3")
db.row_factory = sqlite3.Row

print("=== EQUIPMENTS ===")
for r in db.execute("SELECT ID, CAMERANAME, GAIN_ADU, READNOISE_E, FOCAL FROM EQUIPMENTS ORDER BY ID"):
    print(dict(r))

print("\n=== TELESCOPES table exists? ===")
try:
    for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%TELE%'"):
        print(r[0])
        for c in db.execute(f"PRAGMA table_info({r[0]})"):
            pass
        cols = [c[1] for c in db.execute(f"PRAGMA table_info({r[0]})")]
        print("  cols:", cols)
        for row in db.execute(f"SELECT * FROM {r[0]} LIMIT 5"):
            print(" ", dict(row))
except Exception as e:  # noqa: BLE001
    print(e)

print("\n=== CALIBRATION_LIBRARY 60s (eq=1, tel=1) ===")
rows = db.execute(
    """
    SELECT ID, KIND, EXPTIME, FILTER_NAME, FILE_PATH, ID_EQUIPMENTS, ID_TELESCOPE
    FROM CALIBRATION_LIBRARY
    WHERE EXPTIME = 60 AND ID_EQUIPMENTS = 1 AND ID_TELESCOPE = 1
    ORDER BY KIND, ID
    """
).fetchall()
print("count", len(rows))
kinds = {}
for r in rows:
    kinds[r["KIND"]] = kinds.get(r["KIND"], 0) + 1
    fp = str(r["FILE_PATH"])
    print(dict(r) if len(fp) < 100 else {**dict(r), "FILE_PATH": "..." + fp[-70:]})
print("summary kinds:", kinds)

print("\n=== CALIBRATION_LIBRARY 60s ALL ===")
for r in db.execute(
    """
    SELECT KIND, ID_EQUIPMENTS, ID_TELESCOPE, COUNT(*) n
    FROM CALIBRATION_LIBRARY WHERE EXPTIME = 60
    GROUP BY KIND, ID_EQUIPMENTS, ID_TELESCOPE
    ORDER BY n DESC
    """
):
    print(dict(r))

print("\n=== OBS_DRAFT 342 ===")
r = db.execute("SELECT * FROM OBS_DRAFT WHERE ID = 342").fetchone()
if r:
    d = dict(r)
    for k in ("ARCHIVE_PATH", "LIGHTS_PATH", "MASTERSTAR_FITS_PATH"):
        if k in d and d[k]:
            d[k] = "..." + str(d[k])[-60:]
    print(d)

print("\n=== Setup folders draft 342 ===")
import os
p = ROOT / "Archive/Drafts/draft_000342/platesolve"
if p.is_dir():
    print([x.name for x in p.iterdir()])

print("\n=== CALIBRATION_LIBRARY all eq=1 tel=1 ===")
for r in db.execute(
    "SELECT ID, KIND, EXPTIME, FILTER_NAME, FILE_PATH FROM CALIBRATION_LIBRARY "
    "WHERE ID_EQUIPMENTS=1 AND ID_TELESCOPE=1 ORDER BY KIND, EXPTIME"
):
    print(dict(r))

print("\n=== LOCATION ===")
for r in db.execute("SELECT ID, NAME, LATITUDE, LONGITUDE, ALTITUDE FROM LOCATION ORDER BY ID"):
    print(dict(r))

print("\n=== SCANNING (setup templates) ===")
try:
    cols = [c[1] for c in db.execute("PRAGMA table_info(SCANNING)").fetchall()]
    print("cols", cols)
    for r in db.execute("SELECT * FROM SCANNING ORDER BY ID DESC LIMIT 10"):
        print(dict(r))
except Exception as e:  # noqa: BLE001
    print(e)

db.close()
