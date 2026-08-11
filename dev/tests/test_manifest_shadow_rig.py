# -*- coding: ascii -*-
"""Phase 2.8 manifest-direct rig reads (shadow observe retired)."""
from __future__ import annotations

from draft_provenance import resolve_draft_dir_for_id, write_draft_manifest


def test_fetch_obs_draft_reads_manifest_rig_directly(tmp_path) -> None:
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
    assert row["ID_EQUIPMENTS"] == 999
    db.close()


def test_create_draft_manifest_matches_fetch(tmp_path) -> None:
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

    row = db.fetch_obs_draft_by_id(int(draft_id))
    assert row is not None
    assert row["ID_EQUIPMENTS"] == 1
    assert float(row["OBSERVATIONSTARTJD"]) == 2450000.5
    db.close()
