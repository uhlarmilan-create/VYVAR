"""Observer location hydration (WAVE-B STEP 5, DELETE-DB-DUP).

observer_lat/lon/alt_m/name are DB-authoritative: they are hydrated from the LOCATION row
selected by ``observer_location_id`` and are no longer read from config.json. On DB hydrate
failure the fields fall back to the dataclass site defaults and a WARNING is logged.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from config import AppConfig

# Dataclass site defaults (the fallback source now that config.json coords are gone).
DEFAULT_LAT = AppConfig.__dataclass_fields__["observer_lat"].default
DEFAULT_LON = AppConfig.__dataclass_fields__["observer_lon"].default
DEFAULT_ALT = AppConfig.__dataclass_fields__["observer_alt_m"].default


def test_observer_location_hydrate_db_failure_falls_back_to_defaults_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    # Legacy observer coords in config.json are IGNORED (DB-authoritative); on DB hydrate
    # failure the mirror falls back to the dataclass site defaults.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "observer_location_id": 1,
                "observer_lat": 50.073658,   # ignored (no longer loaded)
                "observer_lon": 14.41854,    # ignored
                "observer_alt_m": 250.0,     # ignored
                "observer_location_name": "Test site",  # ignored
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "database.get_observer_location_by_id",
        side_effect=sqlite3.Error("simulated DB hydrate failure"),
    ):
        with caplog.at_level(logging.WARNING, logger="config"):
            cfg = AppConfig(project_root=tmp_path)

    assert cfg.observer_lat == DEFAULT_LAT
    assert cfg.observer_lon == DEFAULT_LON
    assert cfg.observer_alt_m == DEFAULT_ALT
    assert any(
        "Observer location DB hydrate failed" in rec.message for rec in caplog.records
    )


def test_observer_coords_come_from_db_not_config_json(tmp_path: Path) -> None:
    # DB LOCATION row is authoritative; config.json coords are ignored entirely.
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "observer_location_id": 5,
                "observer_lat": 11.11,
                "observer_lon": 22.22,
                "observer_alt_m": 999.0,
                "observer_location_name": "IgnoredName",
            }
        ),
        encoding="utf-8",
    )
    fake_loc = {"name": "DB Site", "lat": 40.0, "lon": 10.0, "alt_m": 100.0}
    with patch("database.get_observer_location_by_id", return_value=fake_loc):
        cfg = AppConfig(project_root=tmp_path)

    assert cfg.observer_location_name == "DB Site"
    assert cfg.observer_lat == 40.0
    assert cfg.observer_lon == 10.0
    assert cfg.observer_alt_m == 100.0
    # save payload never carries the DB-authoritative mirrors
    payload = cfg.to_json()
    for key in ("observer_lat", "observer_lon", "observer_alt_m", "observer_location_name"):
        assert key not in payload


def test_observer_coords_absent_from_config_use_defaults(tmp_path: Path) -> None:
    # No location id -> no hydration -> dataclass site defaults.
    (tmp_path / "config.json").write_text(
        json.dumps({"observer_location_id": 0}), encoding="utf-8"
    )
    cfg = AppConfig(project_root=tmp_path)
    assert cfg.observer_lat == DEFAULT_LAT
    assert cfg.observer_lon == DEFAULT_LON
