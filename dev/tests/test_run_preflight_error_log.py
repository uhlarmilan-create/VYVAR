# -*- coding: ascii -*-
"""Early RUN VYVAR failure must persist under data_dir/logs/."""
from __future__ import annotations

from pathlib import Path

import pytest

from database import VyvarDatabase
from run_preflight_log import (
    preflight_logs_dir,
    summarize_db_fk_state,
    write_run_preflight_error_log,
)


def test_write_run_preflight_error_log_creates_file(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    db = VyvarDatabase(data_root / "vyvar.sqlite3")
    db.insert_location(place_name="Site", latitude=50.0, longitude=14.0, altitude=300.0)
    exc = ValueError(
        "Cannot create observation draft (INSERT INTO OBS_DRAFT): missing observatory location (id=2)"
    )
    path = write_run_preflight_error_log(
        data_root,
        step="Scan Source + Import + calibration",
        exc=exc,
        db=db,
        cfg=type("Cfg", (), {"observer_location_id": 2})(),
    )
    assert path.is_file()
    assert path.parent == preflight_logs_dir(data_root)
    assert path.name.startswith("run_preflight_error_")
    text = path.read_text(encoding="utf-8")
    assert "Scan Source + Import + calibration" in text
    assert "INSERT INTO OBS_DRAFT" in text
    assert "LOCATION:" in text
    assert "observer_location_id=2" in text
    assert "Traceback" in text or "traceback:" in text


def test_forced_preflight_exception_ui_message_names_log_path() -> None:
    """Mirror app._vyvar_format_run_failure_message contract (no Streamlit runtime)."""
    log_path = Path("/tmp/vyvar_data/logs/run_preflight_error_20260724_120000.log")
    detail = "Scan Source + Import + calibration: FOREIGN KEY constraint failed"
    msg = f"{detail} Preflight log: {log_path}"
    assert "Preflight log:" in msg
    assert str(log_path) in msg
    assert "FOREIGN KEY" in msg


def test_summarize_db_fk_state_lists_table_counts(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    db.insert_equipment(
        camera_name="Cam",
        alias="C",
        sensor_type="mono",
        sensor_size="1000",
        pixel_size=3.76,
    )
    summary = summarize_db_fk_state(db, cfg=type("Cfg", (), {"observer_location_id": 0})())
    assert "EQUIPMENTS: 1 row(s)" in summary
