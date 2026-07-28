# -*- coding: ascii -*-
"""FK preflight for OBS_DRAFT / OBSERVATION on fresh reference tables."""
from __future__ import annotations

import pytest

from database import VyvarDatabase


def _fresh_db(tmp_path, monkeypatch) -> VyvarDatabase:
    db_path = tmp_path / "vyvar.sqlite3"
    monkeypatch.chdir(tmp_path)
    return VyvarDatabase(db_path)


def _seed_equipment_telescope_location(db: VyvarDatabase) -> tuple[int, int, int]:
    eq = db.insert_equipment(
        camera_name="Cam",
        alias="C",
        sensor_type="mono",
        sensor_size="1000",
        pixel_size=3.76,
    )
    tel = int(
        db.conn.execute(
            "INSERT INTO TELESCOPE (TELESCOPENAME, ALIAS, DIAMETER, FOCAL, ACTIVE) "
            "VALUES (?, ?, ?, ?, 1);",
            ("Tel", "T", 200.0, 1000.0),
        ).lastrowid
    )
    loc = db.insert_location(
        place_name="Site",
        latitude=50.0,
        longitude=14.0,
        altitude=300.0,
    )
    return eq, tel, loc


def test_create_draft_missing_scanning_raises_clear_error(tmp_path, monkeypatch) -> None:
    db = _fresh_db(tmp_path, monkeypatch)
    eq, tel, loc = _seed_equipment_telescope_location(db)
    with pytest.raises(ValueError, match="scanning profile"):
        db.create_draft(
            {
                "id_equipments": eq,
                "id_telescope": tel,
                "id_location": loc,
                "observation_start_jd": 2450000.0,
            }
        )


def test_create_draft_stale_config_location_id_raises_clear_error(tmp_path, monkeypatch) -> None:
    db = _fresh_db(tmp_path, monkeypatch)
    eq, tel, loc = _seed_equipment_telescope_location(db)
    scan = db.insert_scanning(
        exp_time=60.0,
        filters="NoFilter",
        binning=1,
        sensor_temp=-10.0,
        gain=100,
    )
    with pytest.raises(ValueError, match="observatory location"):
        db.create_draft(
            {
                "id_equipments": eq,
                "id_telescope": tel,
                "id_location": 2,
                "id_scanning": scan,
                "observation_start_jd": 2450000.0,
            }
        )


def test_resolve_import_location_fails_on_stale_config_id(tmp_path, monkeypatch) -> None:
    db = _fresh_db(tmp_path, monkeypatch)
    _seed_equipment_telescope_location(db)
    with pytest.raises(ValueError, match="observer_location_id"):
        db.resolve_import_location_id(id_location=None, cfg_location_id=2)


def test_resolve_import_location_requires_row(tmp_path, monkeypatch) -> None:
    db = _fresh_db(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="observer_location_id"):
        db.resolve_import_location_id(id_location=None, cfg_location_id=2)


def test_create_draft_succeeds_with_valid_foreign_keys(tmp_path, monkeypatch) -> None:
    db = _fresh_db(tmp_path, monkeypatch)
    eq, tel, loc = _seed_equipment_telescope_location(db)
    scan = db.insert_scanning(
        exp_time=60.0,
        filters="NoFilter",
        binning=1,
        sensor_temp=-10.0,
        gain=100,
    )
    draft_id = db.create_draft(
        {
            "id_equipments": eq,
            "id_telescope": tel,
            "id_location": loc,
            "id_scanning": scan,
            "observation_start_jd": 2450000.0,
        }
    )
    assert draft_id == 1
