"""Observer location DB hydrate: same fallback coords on DB failure + WARNING."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from config import AppConfig


def test_observer_location_hydrate_db_failure_keeps_json_coords_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    json_lat = 50.073658
    json_lon = 14.41854
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "observer_location_id": 1,
                "observer_lat": json_lat,
                "observer_lon": json_lon,
                "observer_alt_m": 250.0,
                "observer_location_name": "Test site",
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

    assert cfg.observer_lat == json_lat
    assert cfg.observer_lon == json_lon
    assert cfg.observer_alt_m == 250.0
    assert cfg.observer_location_name == "Test site"
    assert any(
        "Observer location DB hydrate failed" in rec.message for rec in caplog.records
    )


def test_observer_location_hydrate_db_failure_keeps_zero_coords(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "observer_location_id": 1,
                "observer_lat": 0.0,
                "observer_lon": 0.0,
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

    assert cfg.observer_lat == 0.0
    assert cfg.observer_lon == 0.0
    assert any(
        "Observer location DB hydrate failed" in rec.message for rec in caplog.records
    )
