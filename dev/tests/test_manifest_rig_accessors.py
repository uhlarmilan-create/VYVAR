# -*- coding: ascii -*-
"""Phase 2.8 manifest-direct rig-id accessors (get_draft_*)."""
from __future__ import annotations

from draft_provenance import resolve_draft_dir_for_id, write_draft_manifest


def test_get_draft_equipment_id_returns_manifest_value(tmp_path) -> None:
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
            "observation_start_jd": 0.0,
            "is_calibrated": 0,
        }
    )
    archive = resolve_draft_dir_for_id(db, int(draft_id))
    assert archive is not None
    write_draft_manifest(
        archive,
        draft_id=int(draft_id),
        calibration_mode="vyvar_calibrated",
        rig={
            "equipment_id": 2,
            "telescope_id": 1,
            "location_id": 1,
            "scanning_id": 1,
        },
        paths={"archive": str(archive.resolve())},
        files=[],
    )

    assert db.get_draft_equipment_id(int(draft_id)) == 2
    db.close()


def test_get_draft_location_id_reads_manifest_after_create(tmp_path) -> None:
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
            "observation_start_jd": 0.0,
            "is_calibrated": 0,
        }
    )

    assert db.get_draft_location_id(int(draft_id)) == 1
    db.close()


def test_resolve_phase2a_equipment_id_uses_manifest(tmp_path) -> None:
    from pathlib import Path

    from database import VyvarDatabase
    from photometry_core import _resolve_phase2a_equipment_id
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
            "observation_start_jd": 0.0,
            "is_calibrated": 0,
        }
    )
    archive = resolve_draft_dir_for_id(db, int(draft_id))
    assert archive is not None
    write_draft_manifest(
        archive,
        draft_id=int(draft_id),
        calibration_mode="vyvar_calibrated",
        rig={
            "equipment_id": 2,
            "telescope_id": 1,
            "location_id": 1,
            "scanning_id": 1,
        },
        paths={"archive": str(archive.resolve())},
        files=[],
    )

    eq_id = _resolve_phase2a_equipment_id(
        db,
        draft_id=int(draft_id),
        output_dir=Path("."),
        masterstar_fits_path=Path("."),
    )
    assert eq_id == 2
    db.close()
