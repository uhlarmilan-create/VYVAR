# -*- coding: ascii -*-
"""SITE-UI-01: RUN must use the observing site the UI shows."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from database import VyvarDatabase
from draft_provenance import record_observer_location_provenance
from night_run import NightRunParams, missing_night_run_inputs, resolve_night_run_cli_ids
from observer_location import (
    location_preselect_id,
    resolve_observer_location_for_run,
)


def _seed_two_sites(db: VyvarDatabase) -> tuple[int, int]:
    zdanice = db.insert_location(
        place_name="Zdanice", latitude=49.0, longitude=17.0, altitude=300.0
    )
    jirny = db.insert_location(
        place_name="Jirny", latitude=50.112, longitude=14.698, altitude=275.0
    )
    return int(zdanice), int(jirny)


def test_t1_ui_selection_not_config_builds_night_run_params(tmp_path: Path) -> None:
    """Selection id != config id -> NightRunParams.location_id == selection."""
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(str(db_path))
    zdanice, jirny = _seed_two_sites(db)
    cfg = SimpleNamespace(
        observer_location_id=zdanice,
        database_path=str(db_path),
        data_root=str(tmp_path),
        observer_lat=0.0,
        observer_lon=0.0,
        observer_alt_m=0.0,
        observer_location_name="",
    )
    # Emulate the UI RUN path: pass selectbox id, not cfg.
    selection_id = jirny
    assert selection_id != int(cfg.observer_location_id)
    params = NightRunParams(
        source_dir=tmp_path,
        equipment_id=1,
        telescope_id=1,
        location_id=int(selection_id),
        location_source_hint="ui_selection",
    )
    assert params.location_id == jirny
    assert params.location_source_hint == "ui_selection"
    resolved = resolve_observer_location_for_run(
        db_path,
        explicit_location_id=params.location_id,
        cfg=cfg,
        source_hint=params.location_source_hint,
    )
    assert resolved.location_id == jirny
    assert resolved.name == "Jirny"
    assert resolved.source == "ui_selection"


def test_t2_explicit_overrides_manifest_and_updates_rig(tmp_path: Path) -> None:
    """Explicit site != manifest -> resolved=explicit, manifest updated, log line."""
    from infolog import log_milestone  # noqa: PLC0415

    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(str(db_path))
    zdanice, jirny = _seed_two_sites(db)
    draft = tmp_path / "draft_000521"
    draft.mkdir()
    (draft / "draft_manifest.json").write_text(
        json.dumps(
            {
                "draft_id": 521,
                "calibration_mode": "vyvar",
                "rig": {"equipment_id": 1, "telescope_id": 2, "location_id": zdanice},
            }
        ),
        encoding="ascii",
    )
    resolved = resolve_observer_location_for_run(
        db_path,
        explicit_location_id=jirny,
        manifest_location_id=zdanice,
        cfg=SimpleNamespace(observer_location_id=zdanice),
        source_hint="cli_arg",
    )
    assert resolved.location_id == jirny
    assert resolved.source == "cli_arg"
    messages: list[str] = []

    def _capture(msg: str, **_kw: object) -> None:
        messages.append(str(msg))

    with patch("infolog.log_milestone", _capture):
        record_observer_location_provenance(
            archive_path=draft, draft_id=521, resolved=resolved
        )
    man = json.loads((draft / "draft_manifest.json").read_text(encoding="ascii"))
    assert int(man["rig"]["location_id"]) == jirny
    assert int(man["observer_location"]["location_id"]) == jirny
    assert any(
        "manifest location changed" in m and str(zdanice) in m and str(jirny) in m
        for m in messages
    )


def test_t3_night_run_without_location_id_and_config_zero_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    VyvarDatabase(str(db_path))
    cfg = SimpleNamespace(observer_location_id=0)
    missing = missing_night_run_inputs(
        equipment_id=1, telescope_id=1, location_id=None, cfg=cfg
    )
    assert missing == ["observing site"]
    with pytest.raises(ValueError, match="observer_location_id"):
        resolve_observer_location_for_run(
            db_path,
            explicit_location_id=None,
            manifest_location_id=None,
            cfg=cfg,
        )


def test_t4_source_matches_precedence_branch(tmp_path: Path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(str(db_path))
    zdanice, jirny = _seed_two_sites(db)
    cfg = SimpleNamespace(observer_location_id=zdanice)

    ui = resolve_observer_location_for_run(
        db_path, explicit_location_id=jirny, cfg=cfg, source_hint="ui_selection"
    )
    assert ui.source == "ui_selection" and ui.location_id == jirny

    cli = resolve_observer_location_for_run(
        db_path, explicit_location_id=jirny, cfg=cfg, source_hint="cli_arg"
    )
    assert cli.source == "cli_arg" and cli.location_id == jirny

    man = resolve_observer_location_for_run(
        db_path,
        explicit_location_id=None,
        manifest_location_id=jirny,
        cfg=cfg,
    )
    assert man.source == "manifest" and man.location_id == jirny

    conf = resolve_observer_location_for_run(
        db_path, explicit_location_id=None, manifest_location_id=None, cfg=cfg
    )
    assert conf.source == "config" and conf.location_id == zdanice

    eq, tel, loc, missing, src = resolve_night_run_cli_ids(
        equipment_id=1,
        telescope_id=1,
        location_id=jirny,
        cfg=cfg,
    )
    assert missing == [] and loc == jirny and src == "cli_arg"


def test_preselect_prefers_config_over_is_default() -> None:
    locations = [
        {"id": 1, "name": "Zdanice", "is_default": 0},
        {"id": 2, "name": "Jirny", "is_default": 1},
    ]
    assert location_preselect_id(locations, cfg_observer_location_id=1) == 1
    assert location_preselect_id(locations, cfg_observer_location_id=0) == 2
    assert location_preselect_id(locations, cfg_observer_location_id=99) == 2
    assert location_preselect_id([], cfg_observer_location_id=1) is None
