# -*- coding: ascii -*-
"""Phase 2.1 manifest rig-id shadow-observe (no reader flip)."""
from __future__ import annotations

from draft_provenance import (
    clear_manifest_shadow_load_cache,
    manifest_shadow_counter_snapshot,
    record_draft_manifest_core,
    reset_manifest_shadow_counters,
    write_draft_manifest,
)


def test_shadow_observe_equal_on_matching_manifest(tmp_path) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    reset_manifest_shadow_counters()
    clear_manifest_shadow_load_cache()
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)
    seed_reference_observatory(db)
    archive = tmp_path / "draft_shadow_ok"
    archive.mkdir()
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
        lights_path=str(archive / "calibrated" / "lights"),
        calib_path=str(archive / "calibrated"),
        imported_at="2026-08-10T12:00:00Z",
        archive_path=str(archive),
    )
    record_draft_manifest_core(db, int(draft_id))

    row = db.fetch_obs_draft_by_id(int(draft_id))
    assert row is not None
    assert row["ID_EQUIPMENTS"] == 1
    snap = manifest_shadow_counter_snapshot()
    assert snap["mismatch"] == 0
    assert snap["equal"] >= 1
    db.close()


def test_shadow_observe_increments_mismatch_counter(tmp_path) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    reset_manifest_shadow_counters()
    clear_manifest_shadow_load_cache()
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)
    seed_reference_observatory(db)
    archive = tmp_path / "draft_shadow_bad"
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
        imported_at="2026-08-10T12:00:00Z",
        archive_path=str(archive),
    )
    write_draft_manifest(
        archive,
        draft_id=int(draft_id),
        calibration_mode="vyvar_calibrated",
        rig={
            "equipment_id": 999,
            "telescope_id": 1,
            "location_id": 1,
            "scanning_id": 1,
        },
        paths={"archive": str(archive.resolve())},
        files=[],
    )

    row = db.fetch_obs_draft_by_id(int(draft_id))
    assert row is not None
    assert row["ID_EQUIPMENTS"] == 1
    assert manifest_shadow_counter_snapshot()["mismatch"] == 1
    db.close()


def test_fetch_telescope_equipment_return_shape_unchanged(tmp_path) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    reset_manifest_shadow_counters()
    clear_manifest_shadow_load_cache()
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)
    seed_reference_observatory(db)
    archive = tmp_path / "draft_te_ui"
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
        imported_at="2026-08-10T12:00:00Z",
        archive_path=str(archive),
    )
    record_draft_manifest_core(db, int(draft_id))

    info = db.fetch_obs_draft_telescope_equipment(int(draft_id))
    assert info is not None
    assert set(info.keys()) == {
        "draft_id",
        "telescope_name",
        "telescope_focal_mm",
        "equipment_name",
        "pixel_um",
    }
    db.close()
