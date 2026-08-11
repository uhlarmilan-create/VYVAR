# -*- coding: ascii -*-
"""Phase 2.3 manifest-first rig-id accessors (get_draft_*)."""
from __future__ import annotations

from draft_provenance import (
    clear_manifest_shadow_load_cache,
    manifest_shadow_counter_snapshot,
    reset_manifest_shadow_counters,
    write_draft_manifest,
)


def test_get_draft_equipment_id_returns_manifest_value(tmp_path) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    reset_manifest_shadow_counters()
    clear_manifest_shadow_load_cache()
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    seed_reference_observatory(db)
    archive = tmp_path / "draft_eq_manifest"
    archive.mkdir()
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
    db.update_draft_import_log(
        int(draft_id),
        lights_path=str(archive / "calibrated" / "lights"),
        calib_path=str(archive / "calibrated"),
        imported_at="2026-08-11T09:00:00Z",
        archive_path=str(archive),
    )
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
    db.conn.execute("UPDATE OBS_DRAFT SET ID_EQUIPMENTS = 1 WHERE ID = ?;", (int(draft_id),))
    db.conn.commit()

    assert db.get_draft_equipment_id(int(draft_id)) == 2
    assert manifest_shadow_counter_snapshot()["fallback"] == 0
    db.close()


def test_get_draft_location_id_falls_back_without_manifest(tmp_path) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    reset_manifest_shadow_counters()
    clear_manifest_shadow_load_cache()
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
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
    assert manifest_shadow_counter_snapshot()["fallback"] >= 1
    db.close()


def test_resolve_phase2a_equipment_id_uses_manifest(tmp_path) -> None:
    from pathlib import Path

    from database import VyvarDatabase
    from photometry_core import _resolve_phase2a_equipment_id
    from tools.reference_seed import seed_reference_observatory

    reset_manifest_shadow_counters()
    clear_manifest_shadow_load_cache()
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    seed_reference_observatory(db)
    archive = tmp_path / "draft_p2a_eq"
    archive.mkdir()
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
    db.update_draft_import_log(
        int(draft_id),
        lights_path=str(archive / "calibrated" / "lights"),
        calib_path=str(archive / "calibrated"),
        imported_at="2026-08-11T09:00:00Z",
        archive_path=str(archive),
    )
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
    db.conn.execute("UPDATE OBS_DRAFT SET ID_EQUIPMENTS = 1 WHERE ID = ?;", (int(draft_id),))
    db.conn.commit()

    eq_id = _resolve_phase2a_equipment_id(
        db,
        draft_id=int(draft_id),
        output_dir=Path("."),
        masterstar_fits_path=Path("."),
    )
    assert eq_id == 2
    db.close()
