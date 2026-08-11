# -*- coding: ascii -*-
"""Phase 2.4 manifest-first core draft field overlay."""
from __future__ import annotations

from draft_provenance import (
    apply_manifest_core_to_draft_row,
    clear_manifest_shadow_load_cache,
    reset_manifest_shadow_counters,
    write_draft_manifest,
)


def test_apply_manifest_core_overlays_center_and_paths(tmp_path) -> None:
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    reset_manifest_shadow_counters()
    clear_manifest_shadow_load_cache()
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    seed_reference_observatory(db)
    archive = tmp_path / "draft_core_overlay"
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
        imported_at="2026-08-11T10:00:00Z",
        archive_path=str(archive),
    )
    write_draft_manifest(
        archive,
        draft_id=int(draft_id),
        calibration_mode="vyvar_calibrated",
        rig={"equipment_id": 1, "telescope_id": 1, "location_id": 1, "scanning_id": 1},
        paths={"archive": str(archive.resolve()), "lights": str(archive / "lights")},
        center={"ra_deg": 202.5, "de_deg": 47.25},
        observation_start_jd=2460000.5,
        files=[],
    )
    db.conn.execute(
        "UPDATE OBS_DRAFT SET CENTEROFFIELDRA = 0, CENTEROFFIELDDE = 0, OBSERVATIONSTARTJD = 0 WHERE ID = ?;",
        (int(draft_id),),
    )
    db.conn.commit()

    row = db.fetch_obs_draft_by_id(int(draft_id))
    assert row is not None
    assert float(row["CENTEROFFIELDRA"]) == 202.5
    assert float(row["CENTEROFFIELDDE"]) == 47.25
    assert float(row["OBSERVATIONSTARTJD"]) == 2460000.5
    db.close()
