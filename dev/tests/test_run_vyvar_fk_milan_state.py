# -*- coding: ascii -*-
"""Milan-like fresh DB: equipment + telescope + location + cal library (expired dark)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from database import VyvarDatabase
from importer import smart_import_session, smart_scan_source


def _write_master_dark(path: Path, *, age_days: float) -> None:
    capture = datetime.now(timezone.utc) - timedelta(days=float(age_days))
    hdu = fits.PrimaryHDU(data=np.zeros((8, 8), dtype=np.float32))
    hdu.header["VY_CDATE"] = capture.strftime("%Y-%m-%dT%H:%M:%SZ")
    hdu.header["IMAGETYP"] = "MASTER DARK"
    hdu.header["XBINNING"] = 2
    hdu.header["YBINNING"] = 2
    hdu.header["EXPTIME"] = 60.0
    hdu.header["GAIN"] = 100
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)


def _write_master_flat(path: Path) -> None:
    fresh = datetime.now(timezone.utc) - timedelta(days=5)
    hdu = fits.PrimaryHDU(data=np.ones((8, 8), dtype=np.float32))
    hdu.header["VY_CDATE"] = fresh.strftime("%Y-%m-%dT%H:%M:%SZ")
    hdu.header["IMAGETYP"] = "MASTER FLAT"
    hdu.header["XBINNING"] = 2
    hdu.header["YBINNING"] = 2
    hdu.header["EXPTIME"] = 0.0
    hdu.header["FILTER"] = "NoFilter"
    hdu.header["GAIN"] = 100
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)


def _write_light(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=np.zeros((16, 16), dtype=np.float32))
    hdu.header["IMAGETYP"] = "LIGHT"
    hdu.header["EXPTIME"] = 60.0
    hdu.header["XBINNING"] = 2
    hdu.header["YBINNING"] = 2
    hdu.header["GAIN"] = 100
    hdu.header["CCD-TEMP"] = -10.0
    hdu.header["FILTER"] = "NoFilter"
    hdu.header["DATE-OBS"] = "2026-07-24T22:00:00"
    hdu.header["CRVAL1"] = 202.0
    hdu.header["CRVAL2"] = 47.0
    hdu.writeto(path, overwrite=True)


class _Cfg:
    calibration_master_ccd_temp_tolerance_c = 0.5

    def __init__(self, data_root: Path, *, observer_location_id: int = 1) -> None:
        self.observer_location_id = int(observer_location_id)
        self.data_root = data_root
        self.archive_root = data_root / "Archive"
        self.calibration_library_root = data_root / "CalibrationLibrary"
        self.archive_root.mkdir(parents=True, exist_ok=True)


class _Pipeline:
    def __init__(self, db: VyvarDatabase, cfg: _Cfg) -> None:
        self.db = db
        self.config = cfg
        self.db._archive_root_override = Path(cfg.archive_root).expanduser().resolve()


def test_milan_state_scan_shows_expired_dark_and_import_succeeds(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "logs").mkdir()
    db_path = data_root / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)

    eq = db.insert_equipment(
        camera_name="QHY600",
        alias="Cam",
        sensor_type="IMX571",
        sensor_size="9576*6388",
        pixel_size=3.76,
    )
    tel = int(
        db.conn.execute(
            "INSERT INTO TELESCOPE (TELESCOPENAME, ALIAS, DIAMETER, FOCAL, ACTIVE) "
            "VALUES (?, ?, ?, ?, 'YES');",
            ("RC12", "RC", 300.0, 2400.0),
        ).lastrowid
    )
    loc = db.insert_location(
        place_name="Observatory",
        latitude=50.1,
        longitude=14.4,
        altitude=400.0,
    )

    cal_root = data_root / "CalibrationLibrary"
    dark = cal_root / "Dark_60s_Bin2_expired.fits"
    flat = cal_root / "Flat_NoFilter_Bin2.fits"
    _write_master_dark(dark, age_days=150.0)
    _write_master_flat(flat)
    assert db.register_calibration_library_entry(
        kind="dark",
        file_path=dark,
        xbinning=2,
        exptime=60.0,
        ccd_temp=-10.0,
        gain=100,
        id_equipments=eq,
        id_telescope=tel,
    )
    assert db.register_calibration_library_entry(
        kind="flat",
        file_path=flat,
        xbinning=2,
        exptime=0.0,
        ccd_temp=-10.0,
        filter_name="NoFilter",
        gain=100,
        id_equipments=eq,
        id_telescope=tel,
    )

    source = tmp_path / "session" / "Lights"
    _write_light(source / "light_0001.fits")

    cfg = _Cfg(data_root, observer_location_id=int(loc))
    plan = smart_scan_source(
        source_root=source.parent,
        calibration_library_root=cal_root,
        masterdark_validity_days=90,
        masterflat_validity_days=200,
        db=db,
        id_equipments=eq,
        id_telescope=tel,
        calibration_master_ccd_temp_tolerance_c=0.5,
    )
    assert plan.lights_files
    assert any("expired" in (row.parameters or "").lower() or row.status == "expired" for row in plan.scan_rows if row.type != "Lights") or any(
        "expired" in str(plan.masterdark_status).lower()
        for _ in [0]
    )

    pipeline = _Pipeline(db, cfg)
    result = smart_import_session(
        plan=plan,
        pipeline=pipeline,
        id_equipment=eq,
        id_telescope=tel,
        cfg=cfg,
    )
    assert result.draft_id is not None
    assert int(result.draft_id) >= 1

    row = db.fetch_obs_draft_by_id(int(result.draft_id))
    assert row is not None
    assert int(row["ID_LOCATION"]) == int(loc)
    assert int(row["ID_SCANNING"]) >= 1


def test_milan_state_stale_location_without_fallback_raises_clear_preflight(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    eq = db.insert_equipment(
        camera_name="Cam",
        alias="C",
        sensor_type="mono",
        sensor_size="1000",
        pixel_size=3.76,
    )
    tel = int(
        db.conn.execute(
            "INSERT INTO TELESCOPE (TELESCOPENAME, ALIAS, DIAMETER, FOCAL, ACTIVE) "
            "VALUES (?, ?, ?, ?, 'YES');",
            ("Tel", "T", 200.0, 1000.0),
        ).lastrowid
    )
    scan = 42
    with pytest.raises(ValueError, match="INSERT INTO OBS_DRAFT.*observatory location"):
        db.create_draft(
            {
                "id_equipments": eq,
                "id_telescope": tel,
                "id_location": 2,
                "id_scanning": scan,
                "observation_start_jd": 2460000.0,
            }
        )
