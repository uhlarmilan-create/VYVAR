# -*- coding: ascii -*-
"""Phase 2.8 manifest-first rig-id reads + OBS_DRAFT editor manifest write."""
from __future__ import annotations

import pandas as pd

from draft_provenance import (
    clear_manifest_shadow_load_cache,
    load_draft_manifest,
    reset_manifest_shadow_counters,
    resolve_draft_dir_for_id,
    write_draft_manifest,
)


def _ensure_telescope_8(db) -> None:
    db.conn.execute(
        "INSERT OR IGNORE INTO TELESCOPE (ID, TELESCOPENAME, ALIAS, DIAMETER, FOCAL) "
        "VALUES (8, 'M71 RC', 'M71', 200.0, 1480.0);"
    )
    db.conn.commit()


def test_fetch_obs_draft_returns_manifest_rig_when_present(tmp_path) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    reset_manifest_shadow_counters()
    clear_manifest_shadow_load_cache()
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    db._archive_root_override = tmp_path
    seed_reference_observatory(db)
    _ensure_telescope_8(db)
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
            "equipment_id": 1,
            "telescope_id": 8,
            "location_id": 1,
            "scanning_id": 1,
        },
        paths={"archive": str(archive.resolve())},
        files=[],
    )

    row = db.fetch_obs_draft_by_id(int(draft_id))
    assert row is not None
    assert row["ID_TELESCOPE"] == 8
    db.close()


def test_create_draft_writes_manifest_rig(tmp_path) -> None:
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

    row = db.fetch_obs_draft_by_id(int(draft_id))
    assert row is not None
    assert row["ID_TELESCOPE"] == 1
    db.close()


def test_obs_draft_editor_save_writes_manifest_rig(tmp_path) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    reset_manifest_shadow_counters()
    clear_manifest_shadow_load_cache()
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    db._archive_root_override = tmp_path
    seed_reference_observatory(db)
    _ensure_telescope_8(db)
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
    db.update_draft_import_log(
        int(draft_id),
        lights_path=str(archive / "calibrated" / "lights"),
        calib_path=str(archive / "calibrated"),
        imported_at="2026-08-10T12:00:00Z",
        archive_path=str(archive),
    )

    orig = pd.DataFrame(
        [{"ID": int(draft_id), "ID_TELESCOPE": 1, "ID_EQUIPMENTS": 1, "ID_LOCATION": 1, "ID_SCANNING": 1}]
    )
    edited = orig.copy()
    edited.loc[0, "ID_TELESCOPE"] = 8

    stats = db.apply_main_table_editor_save(
        "OBS_DRAFT",
        "ID",
        orig,
        edited,
        editable_cols=["ID_TELESCOPE", "ID_EQUIPMENTS", "ID_LOCATION", "ID_SCANNING"],
    )
    assert stats["updated"] == 1

    manifest = load_draft_manifest(archive)
    assert manifest["rig"]["telescope_id"] == 8

    row = db.fetch_obs_draft_by_id(int(draft_id))
    assert row is not None
    assert row["ID_TELESCOPE"] == 8
    db.close()


def test_fetch_telescope_equipment_uses_manifest_telescope_for_join(tmp_path) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    db._archive_root_override = tmp_path
    seed_reference_observatory(db)
    _ensure_telescope_8(db)
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
            "equipment_id": 1,
            "telescope_id": 8,
            "location_id": 1,
            "scanning_id": 1,
        },
        paths={"archive": str(archive.resolve())},
        files=[],
    )

    info = db.fetch_obs_draft_telescope_equipment(int(draft_id))
    assert info is not None
    tel8 = db.conn.execute("SELECT FOCAL FROM TELESCOPE WHERE ID = 8;").fetchone()
    assert tel8 is not None
    assert info["telescope_focal_mm"] == tel8["FOCAL"]
    db.close()
