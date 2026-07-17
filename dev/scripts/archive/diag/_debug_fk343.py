import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import AppConfig
from night_run import _load_app_config
from pipeline import AstroPipeline
from importer import extract_fits_metadata, smart_scan_source

cfg = _load_app_config(Path("scripts/_draft343_run_config.json"))
print("db", cfg.database_path)
p = AstroPipeline(cfg)
m = extract_fits_metadata(Path(r"D:\V842_Her\Light\V842_Her_Light_001.fits"), db=p.db, app_config=cfg)
sid = p.db.find_or_create_scanning_id(m)
print("scanning_id", sid)
for q, args in [
    ("LOCATION 2", (2,)),
    ("SCANNING", (sid,)),
    ("EQUIPMENTS 1", (1,)),
    ("TELESCOPE 1", (1,)),
]:
    r = p.db.conn.execute(f"SELECT ID FROM {q.split()[0]} WHERE ID=?", args).fetchone()
    print(q, r)
try:
    did = p.db.create_draft(
        {
            "id_equipments": 1,
            "id_telescope": 1,
            "id_location": 2,
            "id_scanning": sid,
            "observation_start_jd": float(m["jd_start"]),
            "is_calibrated": 1,
        }
    )
    print("created draft", did)
    p.db.conn.execute("DELETE FROM OBS_DRAFT WHERE ID=?", (did,))
    p.db.conn.commit()
except Exception as e:
    print("create_draft failed:", e)
