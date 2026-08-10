# -*- coding: ascii -*-
"""Phase 1a draft_manifest.json core parity (DB shadow, not authority)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from draft_provenance import (
    MANIFEST_SCHEMA_VERSION,
    load_draft_manifest,
    manifest_db_parity_errors,
    record_draft_manifest_core,
    resolve_draft_archive_root_from_row,
)


def test_record_draft_manifest_core_mirrors_obs_draft(tmp_path, monkeypatch) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)
    seed_reference_observatory(db)
    archive = tmp_path / "draft_000001"
    lights = archive / "calibrated" / "lights"
    lights.mkdir(parents=True)

    draft_id = db.create_draft(
        {
            "id_equipments": 1,
            "id_telescope": 1,
            "id_location": 1,
            "id_scanning": 1,
            "observation_start_jd": 2450000.5,
            "is_calibrated": 0,
        }
    )
    db.update_draft_import_log(
        int(draft_id),
        lights_path=str(lights),
        calib_path=str(archive / "calibrated"),
        imported_at="2026-08-10T12:00:00Z",
        archive_path=str(archive),
        is_calibrated=False,
    )
    db.update_obs_draft_status(int(draft_id), "INGESTED")

    path = record_draft_manifest_core(db, int(draft_id))
    assert path is not None
    assert path.is_file()

    manifest = load_draft_manifest(archive)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["draft_id"] == int(draft_id)
    assert manifest["rig"]["equipment_id"] == 1
    assert manifest["paths"]["archive"] == str(archive.resolve())
    assert manifest["status"] == "INGESTED"
    assert manifest["is_calibrated"] == 0
    assert "files" not in manifest

    assert manifest_db_parity_errors(db, int(draft_id)) == []
    db.close()


def test_manifest_db_parity_detects_rig_mismatch(tmp_path, monkeypatch) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)
    seed_reference_observatory(db)
    archive = tmp_path / "draft_000002"
    archive.mkdir()
    draft_id = db.create_draft(
        {
            "id_equipments": 1,
            "id_telescope": 1,
            "id_location": 1,
            "id_scanning": 1,
            "observation_start_jd": 0.0,
            "is_calibrated": 1,
        }
    )
    db.update_draft_import_log(
        int(draft_id),
        lights_path=str(archive / "calibrated" / "lights"),
        calib_path=str(archive / "calibrated"),
        imported_at="2026-08-10T12:00:00Z",
        archive_path=str(archive),
    )
    record_draft_manifest_core(db, int(draft_id))
    row = db.fetch_obs_draft_by_id(int(draft_id)) or {}
    root = resolve_draft_archive_root_from_row(row)
    assert root is not None
    manifest_path = root / "draft_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["rig"]["equipment_id"] = 999
    manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    errors = manifest_db_parity_errors(db, int(draft_id))
    assert any("rig.equipment_id" in e for e in errors)
    db.close()
