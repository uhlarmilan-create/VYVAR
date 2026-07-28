# -*- coding: ascii -*-
"""Unified observer location resolution (POST-453 fixes Part 1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from database import VyvarDatabase
from observer_location import (
    ResolvedObserverLocation,
    apply_resolved_observer_location_to_config,
    resolve_observer_location_for_run,
)


def _seed_location(db: VyvarDatabase, *, lat: float = 50.112, lon: float = 14.698) -> int:
    return db.insert_location(
        place_name="Jirny",
        latitude=lat,
        longitude=lon,
        altitude=275.0,
    )


class _Cfg:
    def __init__(self, loc_id: int = 0) -> None:
        self.observer_location_id = loc_id
        self.observer_lat = 0.0
        self.observer_lon = 0.0
        self.observer_alt_m = 0.0
        self.observer_location_name = ""


def test_ui_cli_config_paths_yield_identical_site(tmp_path: Path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(str(db_path))
    loc_id = _seed_location(db)

    cfg = _Cfg(loc_id)
    ui = resolve_observer_location_for_run(
        db_path,
        explicit_location_id=loc_id,
        cfg=cfg,
        source_hint="ui_selection",
    )
    cli = resolve_observer_location_for_run(
        db_path,
        explicit_location_id=loc_id,
        cfg=_Cfg(0),
        source_hint="cli_arg",
    )
    conf = resolve_observer_location_for_run(db_path, explicit_location_id=None, cfg=cfg)

    for r in (ui, cli, conf):
        assert r.location_id == loc_id
        assert abs(r.lat - 50.112) < 0.001
        assert abs(r.lon - 14.698) < 0.001
    assert ui.source == "ui_selection"
    assert cli.source == "cli_arg"
    assert conf.source == "config"


def test_unresolvable_site_raises_naming_config_key(tmp_path: Path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    VyvarDatabase(str(db_path))
    with pytest.raises(ValueError, match="observer_location_id"):
        resolve_observer_location_for_run(db_path, cfg=_Cfg(0))
    with pytest.raises(ValueError, match="observer_location_id"):
        resolve_observer_location_for_run(db_path, explicit_location_id=99, cfg=_Cfg(0))


def test_provenance_id_matches_coordinates(tmp_path: Path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(str(db_path))
    loc_id = _seed_location(db, lat=50.5, lon=14.5)
    resolved = resolve_observer_location_for_run(db_path, explicit_location_id=loc_id, cfg=_Cfg(loc_id))
    prov = resolved.as_provenance_dict()
    assert prov["location_id"] == loc_id
    assert prov["lat"] == resolved.lat
    assert prov["lon"] == resolved.lon
    assert prov["source"] in ("ui_selection", "cli_arg", "config")


def test_apply_hydrates_config_consistently(tmp_path: Path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(str(db_path))
    loc_id = _seed_location(db)
    cfg = _Cfg(0)
    resolved = resolve_observer_location_for_run(db_path, explicit_location_id=loc_id, cfg=_Cfg(loc_id))
    apply_resolved_observer_location_to_config(cfg, resolved)
    assert cfg.observer_location_id == loc_id
    assert cfg.observer_lat == resolved.lat
    assert cfg.observer_lon == resolved.lon


def test_no_default_prague_site_without_operator_choice(tmp_path: Path) -> None:
    """Regression: must not silently pick LOCATION id=1 or MIN(ID)."""
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(str(db_path))
    dablice = db.insert_location(place_name="Dablice", latitude=50.074, longitude=14.419, altitude=355.0)
    jirny = db.insert_location(place_name="Jirny", latitude=50.112, longitude=14.698, altitude=275.0)
    assert dablice == 1 and jirny == 2
    with pytest.raises(ValueError, match="observer_location_id"):
        resolve_observer_location_for_run(db_path, cfg=_Cfg(0))


def test_importer_db_delegate_matches_resolver(tmp_path: Path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(str(db_path))
    loc_id = _seed_location(db)
    lid, warn = db.resolve_import_location_id(id_location=loc_id, cfg_location_id=0)
    assert warn is None
    assert lid == loc_id
