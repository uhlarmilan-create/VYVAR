#!/usr/bin/env python3
"""M67 equipment registration (Telescope Live SPA-1 FSQ-106 + QHY600M, Spain)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402

CAMERA = {
    "name": "QHY 600M",
    "alias": "GAIN/RN APPROX (Telescope Live SPA-1)",
    "sensor_type": "CMOS",
    "sensor_size": "9576x6388",
    "pixel_size": 3.76,
    "gain": 1.0,
    "read_noise": 1.6,
    "saturate_adu": 60000.0,
}
TELESCOPE = {
    "name": "Takahashi FSQ-106ED",
    "alias": "SPA-1-CMOS F3.6 FL382",
    "focal": 382.0,
    "diameter": 106.0,
}
LOCATION = {
    "name": "Telescope Live SPA-1, Spain",
    "lat": 37.165,
    "lon": -2.314,
    "elev": 850.0,
}


def _find_equipment_id(db: VyvarDatabase, name: str) -> int | None:
    row = db.conn.execute(
        "SELECT ID FROM EQUIPMENTS WHERE CAMERANAME = ? ORDER BY ID LIMIT 1;",
        (name,),
    ).fetchone()
    return int(row[0]) if row else None


def _find_telescope_id(db: VyvarDatabase, name: str) -> int | None:
    row = db.conn.execute(
        "SELECT ID FROM TELESCOPE WHERE TELESCOPENAME = ? ORDER BY ID LIMIT 1;",
        (name,),
    ).fetchone()
    return int(row[0]) if row else None


def _find_location_id(db: VyvarDatabase, name: str) -> int | None:
    row = db.conn.execute(
        "SELECT ID FROM LOCATION WHERE PLACENAME = ? ORDER BY ID LIMIT 1;",
        (name,),
    ).fetchone()
    return int(row[0]) if row else None


def phase_a_register() -> dict[str, int]:
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    out: dict[str, int] = {}

    eq_id = _find_equipment_id(db, CAMERA["name"])
    if eq_id is None:
        eq_id = db.insert_equipment(
            CAMERA["name"],
            CAMERA["alias"],
            CAMERA["sensor_type"],
            CAMERA["sensor_size"],
            float(CAMERA["pixel_size"]),
        )
        status = "created"
    else:
        status = "reused"
    db.set_equipment_cosmic_params(eq_id, CAMERA["gain"], CAMERA["read_noise"])
    db.conn.execute(
        "UPDATE EQUIPMENTS SET SATURATE_ADU = ? WHERE ID = ?;",
        (float(CAMERA["saturate_adu"]), int(eq_id)),
    )
    db.conn.commit()
    out["camera_id"] = int(eq_id)
    out["camera_status"] = status

    tel_id = _find_telescope_id(db, TELESCOPE["name"])
    if tel_id is None:
        tel_id = db.insert_telescope(
            TELESCOPE["name"],
            TELESCOPE["alias"],
            float(TELESCOPE["diameter"]),
            float(TELESCOPE["focal"]),
        )
        status = "created"
    else:
        status = "reused"
    out["telescope_id"] = int(tel_id)
    out["telescope_status"] = status

    loc_id = _find_location_id(db, LOCATION["name"])
    if loc_id is None:
        loc_id = db.insert_location(
            LOCATION["name"],
            float(LOCATION["lat"]),
            float(LOCATION["lon"]),
            float(LOCATION["elev"]),
        )
        status = "created"
    else:
        status = "reused"
    out["location_id"] = int(loc_id)
    out["location_status"] = status
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(phase_a_register(), indent=2))
