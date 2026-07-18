"""Tests for pre-calibrated run provenance and import plan wiring."""

from __future__ import annotations

import json
from pathlib import Path

from draft_provenance import (
    CALIBRATION_MODE_PRE,
    CALIBRATION_MODE_VYVAR,
    apply_pre_calibrated_import_plan,
    calibration_mode_report_line,
    load_draft_manifest,
    record_draft_calibration_provenance,
    resolve_calibration_mode,
    write_draft_manifest,
)
from importer import SmartImportPlan, SmartScanRow


def _minimal_plan(*, quick_look: bool = False) -> SmartImportPlan:
    return SmartImportPlan(
        source_root="/tmp/src",
        lights_files=["/tmp/src/a.fits"],
        dark_files=[],
        flat_files=[],
        lights_first_fits="/tmp/src/a.fits",
        metadata=None,
        scan_rows=[SmartScanRow(type="Lights", status="ok", count=1, parameters="")],
        dark_master="/lib/master_dark.fits",
        flat_master=None,
        masterflat_by_filter={"V": "/lib/flat_v.fits"},
        masterflat_status={"V": "ok"},
        missing_flat_filters=[],
        masterdark_status="found",
        quick_look=quick_look,
        detected_filters=["V"],
        warnings=[],
    )


def test_apply_pre_calibrated_import_plan_forces_quick_look_and_clears_masters():
    plan = _minimal_plan(quick_look=False)
    apply_pre_calibrated_import_plan(plan)
    assert plan.quick_look is True
    assert plan.dark_master is None
    assert plan.masterflat_by_filter == {}
    assert plan.masterflat_by_obs_key == {}
    assert plan.dark_master_by_obs_key == {}
    assert any("Pre-calibrated mode" in w for w in plan.warnings)


def test_calibration_mode_report_lines():
    assert "skipped" in calibration_mode_report_line(CALIBRATION_MODE_PRE)
    assert "VYVAR bias" in calibration_mode_report_line(CALIBRATION_MODE_VYVAR)


def test_draft_manifest_roundtrip(tmp_path: Path):
    manifest = write_draft_manifest(
        tmp_path,
        draft_id=42,
        calibration_mode=CALIBRATION_MODE_PRE,
    )
    assert manifest.is_file()
    loaded = load_draft_manifest(tmp_path)
    assert loaded["draft_id"] == 42
    assert loaded["calibration_mode"] == CALIBRATION_MODE_PRE


def test_record_draft_calibration_provenance_db_and_manifest(tmp_path: Path, monkeypatch):
    from database import VyvarDatabase
    from tools.reference_seed import seed_reference_observatory

    db_path = tmp_path / "test.db"
    db = VyvarDatabase(db_path)
    seed_reference_observatory(db)
    draft_id = db.create_draft(
        {
            "id_equipments": 1,
            "id_telescope": 1,
            "id_location": 1,
            "id_scanning": 1,
            "observation_start_jd": 2450000.0,
            "is_calibrated": 0,
        }
    )
    archive = tmp_path / "draft_000001"
    archive.mkdir()
    record_draft_calibration_provenance(
        db=db,
        archive_path=archive,
        draft_id=draft_id,
        calibration_mode=CALIBRATION_MODE_PRE,
    )
    row = db.fetch_obs_draft_by_id(draft_id) or {}
    assert row.get("CALIBRATION_MODE") == CALIBRATION_MODE_PRE
    assert load_draft_manifest(archive)["calibration_mode"] == CALIBRATION_MODE_PRE
    assert resolve_calibration_mode(draft_id=draft_id, db=db) == CALIBRATION_MODE_PRE
    db.conn.close()


def test_night_run_params_pre_calibrated_flag():
    from night_run import NightRunParams

    p = NightRunParams(
        source_dir=Path("/tmp"),
        equipment_id=1,
        telescope_id=1,
        pre_calibrated_mode=True,
    )
    assert p.pre_calibrated_mode is True


def test_resolve_draft_lights_root_pre_calibrated(tmp_path: Path):
    from draft_provenance import (
        CALIBRATION_MODE_PRE,
        resolve_draft_lights_root,
        write_draft_manifest,
    )

    ap = tmp_path / "draft_000099"
    noncal = ap / "non_calibrated" / "lights" / "V_20_2"
    noncal.mkdir(parents=True)
    (noncal / "frame001.fits").write_bytes(b"SIMPLE  =                    T / dummy\nEND\n")
    write_draft_manifest(ap, draft_id=99, calibration_mode=CALIBRATION_MODE_PRE)

    root = resolve_draft_lights_root(ap)
    assert root == ap / "non_calibrated" / "lights"
    assert not (ap / "calibrated").exists()


def test_resolve_draft_lights_root_vyvar_calibrated(tmp_path: Path):
    from draft_provenance import CALIBRATION_MODE_VYVAR, resolve_draft_lights_root, write_draft_manifest

    ap = tmp_path / "draft_000100"
    cal = ap / "calibrated" / "lights" / "B_20_2"
    cal.mkdir(parents=True)
    (cal / "frame001.fits").write_bytes(b"SIMPLE  =                    T / dummy\nEND\n")
    write_draft_manifest(ap, draft_id=100, calibration_mode=CALIBRATION_MODE_VYVAR)

    root = resolve_draft_lights_root(ap)
    assert root == ap / "calibrated" / "lights"


def test_resolve_obs_file_to_processed_fits_pre_calibrated(tmp_path: Path):
    from draft_provenance import CALIBRATION_MODE_PRE, write_draft_manifest
    from pipeline import resolve_obs_file_to_processed_fits

    ap = tmp_path / "draft_pre"
    setup = ap / "non_calibrated" / "lights" / "B_20_2"
    setup.mkdir(parents=True)
    fits = setup / "Chi_H_0046.fits"
    fits.write_bytes(b"SIMPLE  =                    T / dummy\nEND\n")
    write_draft_manifest(ap, draft_id=374, calibration_mode=CALIBRATION_MODE_PRE)

    db_path = str(fits)
    hit = resolve_obs_file_to_processed_fits(
        ap,
        db_path,
        setup_name="B_20_2",
        draft_id=374,
    )
    assert hit is not None
    assert hit.resolve() == fits.resolve()
    assert "non_calibrated" in str(hit).replace("\\", "/")


def test_map_masterstar_db_candidates_pre_calibrated(tmp_path: Path):
    """DB FILE_PATH under non_calibrated maps to on-disk FITS for MASTERSTAR selection."""
    from database import VyvarDatabase
    from draft_provenance import CALIBRATION_MODE_PRE, write_draft_manifest
    from pipeline import get_masterstar_candidates, resolve_obs_file_to_processed_fits
    from tools.reference_seed import seed_reference_observatory

    ap = tmp_path / "draft_ms"
    setup = ap / "non_calibrated" / "lights" / "V_20_2"
    setup.mkdir(parents=True)
    f1 = setup / "frame_a.fits"
    f2 = setup / "frame_b.fits"
    f1.write_bytes(b"SIMPLE  =                    T / dummy\nEND\n")
    f2.write_bytes(b"SIMPLE  =                    T / dummy\nEND\n")

    db = VyvarDatabase(tmp_path / "t.db")
    seed_reference_observatory(db)
    draft_id = db.create_draft(
        {
            "id_equipments": 1,
            "id_telescope": 1,
            "id_location": 1,
            "id_scanning": 1,
            "observation_start_jd": 2450000.0,
            "is_calibrated": 0,
        }
    )
    db.set_obs_draft_calibration_mode(draft_id, CALIBRATION_MODE_PRE)
    write_draft_manifest(ap, draft_id=draft_id, calibration_mode=CALIBRATION_MODE_PRE)
    db.insert_draft_files(
        draft_id,
        [
            {
                "file_path": str(f1),
                "imagetyp": "light",
                "filter": "V",
                "is_calibrated": 0,
            },
            {
                "file_path": str(f2),
                "imagetyp": "light",
                "filter": "V",
                "is_calibrated": 0,
            },
        ],
    )
    db.update_obs_file_quality_by_id(
        db.conn.execute("SELECT ID FROM OBS_FILES WHERE DRAFT_ID=?", (draft_id,)).fetchone()[0],
        fwhm=2.1,
        sky_level=100.0,
        star_count=200,
    )
    db.update_obs_file_quality_by_id(
        db.conn.execute("SELECT ID FROM OBS_FILES WHERE DRAFT_ID=? ORDER BY ID DESC LIMIT 1", (draft_id,)).fetchone()[0],
        fwhm=3.5,
        sky_level=120.0,
        star_count=150,
    )

    paths = get_masterstar_candidates(draft_id, 50.0, db=db)
    assert paths
    mapped = [
        resolve_obs_file_to_processed_fits(
            ap,
            p,
            setup_name="V_20_2",
            draft_id=draft_id,
            db=db,
        )
        for p in paths
    ]
    assert all(m is not None and m.is_file() for m in mapped)
    assert all("non_calibrated" in str(m).replace("\\", "/") for m in mapped if m)
    db.conn.close()


def test_resolve_masterstar_input_root_pre_calibrated_skip_processed(tmp_path: Path):
    from config import AppConfig
    from draft_provenance import CALIBRATION_MODE_PRE, write_draft_manifest
    from pipeline import resolve_masterstar_input_root

    ap = tmp_path / "draft_pre"
    setup = ap / "non_calibrated" / "lights" / "Green_60_2"
    setup.mkdir(parents=True)
    (setup / "light001.fits").write_bytes(b"SIMPLE  =                    T / dummy\nEND\n")
    write_draft_manifest(ap, draft_id=1, calibration_mode=CALIBRATION_MODE_PRE)

    cfg = AppConfig()
    cfg.skip_processed_directory = True
    hit = resolve_masterstar_input_root(ap, setup_name="Green_60_2", app_config=cfg)
    assert hit is not None
    assert "non_calibrated" in str(hit).replace("\\", "/")
    assert not (ap / "calibrated").exists()
