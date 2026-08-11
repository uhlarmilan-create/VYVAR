# -*- coding: ascii -*-
"""Manifest store tests (Phase 2.8: manifest is sole store)."""
from __future__ import annotations

from draft_provenance import (
    MANIFEST_SCHEMA_VERSION,
    backfill_draft_manifest_from_db,
    load_draft_manifest,
    manifest_db_parity_errors,
    resolve_draft_dir_for_id,
)


def test_backfill_returns_existing_manifest_path(tmp_path) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)
    db._archive_root_override = tmp_path
    seed_reference_observatory(db)

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
    archive = resolve_draft_dir_for_id(db, int(draft_id))
    assert archive is not None
    lights = archive / "calibrated" / "lights"
    lights.mkdir(parents=True)
    db.update_draft_import_log(
        int(draft_id),
        lights_path=str(lights),
        calib_path=str(archive / "calibrated"),
        imported_at="2026-08-10T12:00:00Z",
        archive_path=str(archive),
        is_calibrated=False,
    )

    path = backfill_draft_manifest_from_db(db, int(draft_id))
    assert path is not None
    assert path.is_file()

    manifest = load_draft_manifest(archive)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest_db_parity_errors(db, int(draft_id)) == []


def test_manifest_exists_after_direct_files_write(tmp_path) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    db._archive_root_override = tmp_path
    seed_reference_observatory(db)

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
    archive = resolve_draft_dir_for_id(db, int(draft_id))
    assert archive is not None
    lights = archive / "calibrated" / "lights"
    lights.mkdir(parents=True)

    fp = str((lights / "frame001.fits").resolve())
    db.insert_draft_files(
        int(draft_id),
        [
            {
                "file_path": fp,
                "imagetyp": "light",
                "filter": "Clear",
                "observation_group_key": "g1",
                "id_scanning": 1,
                "is_calibrated": 1,
                "calib_type": "DF",
                "calib_flags": "DF",
            }
        ],
    )
    db.update_draft_import_log(
        int(draft_id),
        lights_path=str(lights),
        calib_path=str(archive / "calibrated"),
        imported_at="2026-08-10T12:00:00Z",
        archive_path=str(archive),
    )

    manifest = load_draft_manifest(archive)
    assert len(manifest.get("files") or []) == 1
    assert manifest["files"][0]["file_path"] == fp
    assert manifest_db_parity_errors(db, int(draft_id)) == []
