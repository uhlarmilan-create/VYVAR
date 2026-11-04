"""G3-F001: scoped calibration library master selection.

v10 item 21 (2026-11-04): ported from the removed ``VyvarDatabase.register_calibration_library_entry`` /
``find_best_calibration_library_path`` (a second, unused copy of the library that truncated the gain with
``int()``) to the one implementation, ``calib_library`` (register_master / find_dark / find_flat). A DARK is
linked to the camera only (CAL-2026-10 spec), so the old "wrong telescope dark" case is now "wrong camera".
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import calib_library as cl
from database import VyvarDatabase

NOW = datetime(2026, 10, 1, 22, 0, tzinfo=timezone.utc)


def _db(tmp_path: Path, name: str = "vyvar.db") -> tuple[VyvarDatabase, int, int, int]:
    db = VyvarDatabase(tmp_path / name)
    cam = db.insert_equipment("CamA", "a", "CMOS", "x", 3.76)
    cam2 = db.insert_equipment("CamB", "b", "CMOS", "x", 3.76)
    tel = db.insert_telescope("Tel", "t", 200.0, 1000.0)
    return db, cam, cam2, tel


def _mk_fits(path: Path, *, ccd_temp: float | None, exptime: float = 60.0, gain: float = 0,
             filter_name: str = "", xbinning: int = 2) -> Path:
    h = fits.Header()
    h["XBINNING"] = xbinning
    h["YBINNING"] = xbinning
    h["EXPTIME"] = exptime
    h["GAIN"] = gain
    h["DATE-OBS"] = NOW.strftime("%Y-%m-%dT%H:%M:%S")
    if ccd_temp is not None:
        h["CCD-TEMP"] = ccd_temp
    if filter_name:
        h["FILTER"] = filter_name
    fits.writeto(path, np.zeros((4, 4), dtype=np.float32), header=h, overwrite=True)
    return path


def _find(db: VyvarDatabase, cam: int, *, ccd_temp: float | None = -10.0, exptime: float = 60.0,
          temp_tolerance: float = 0.5, gain: float = 0) -> str | None:
    r = cl.find_dark(db, camera_id=cam, gain=gain, offset=None, exposure=exptime, binning=2, temp=ccd_temp,
                     temp_tol=temp_tolerance, light_dates=[NOW], validity_days=90)
    return r.master.path if r.master is not None else None


def _find_flat(db: VyvarDatabase, cam: int, tel: int, filter_name: str) -> str | None:
    r = cl.find_flat(db, camera_id=cam, telescope_id=tel, filter_name=filter_name, gain=0, binning=2,
                     light_dates=[NOW], validity_days=200)
    return r.master.path if r.master is not None else None


def _insert_library_row(db: VyvarDatabase, path: Path, *, ccd_temp: float | None, id_eq: int | None) -> None:
    """Direct SQL insert for rows registration refuses (legacy / invalid)."""
    _mk_fits(path, ccd_temp=ccd_temp)
    db.conn.execute(
        "INSERT INTO CALIBRATION_LIBRARY (KIND, FILE_PATH, XBINNING, EXPTIME, CCD_TEMP, FILTER_NAME, GAIN, "
        "REGISTERED_AT, ID_EQUIPMENTS, CAPTURE_DATE, GAIN_EXACT) VALUES ('dark', ?, 2, 60, ?, '', 0, 'x', ?, ?, 1)",
        (str(path.resolve()), ccd_temp, id_eq, NOW.isoformat()),
    )
    db.conn.commit()


def test_proper_scoped_dark_still_selected(tmp_path: Path) -> None:
    db, cam, _, _ = _db(tmp_path)
    good = _mk_fits(tmp_path / "good_dark.fits", ccd_temp=-10.0)
    cl.register_master(db, good, kind="dark", camera_id=cam, telescope_id=None)
    assert Path(_find(db, cam)).resolve() == good.resolve()


def test_flat_different_exptime_still_matches(tmp_path: Path) -> None:
    db, cam, _, tel = _db(tmp_path)
    flat = _mk_fits(tmp_path / "flat_v.fits", ccd_temp=-10.0, exptime=5.0, filter_name="V")
    cl.register_master(db, flat, kind="flat", camera_id=cam, telescope_id=tel)
    assert Path(_find_flat(db, cam, tel, "V")).resolve() == flat.resolve()


def test_null_temp_dark_not_selected(tmp_path: Path) -> None:
    db, cam, _, _ = _db(tmp_path)
    bad = _mk_fits(tmp_path / "null_temp_dark.fits", ccd_temp=None)
    with pytest.raises(ValueError, match="no sensor temperature"):
        cl.register_master(db, bad, kind="dark", camera_id=cam, telescope_id=None)
    _insert_library_row(db, bad, ccd_temp=None, id_eq=cam)
    assert _find(db, cam) is None


def test_unscoped_dark_not_selected(tmp_path: Path) -> None:
    db, cam, _, _ = _db(tmp_path)
    global_dark = tmp_path / "global_dark.fits"
    _mk_fits(global_dark, ccd_temp=-10.0)
    with pytest.raises(ValueError, match="camera"):
        cl.register_master(db, global_dark, kind="dark", camera_id=None, telescope_id=None)
    _insert_library_row(db, global_dark, ccd_temp=-10.0, id_eq=None)
    assert _find(db, cam) is None


def test_wrong_camera_dark_rejected(tmp_path: Path) -> None:
    db, cam, cam2, _ = _db(tmp_path)
    other = _mk_fits(tmp_path / "other_cam_dark.fits", ccd_temp=-10.0)
    cl.register_master(db, other, kind="dark", camera_id=cam2, telescope_id=None)
    assert _find(db, cam) is None


def test_wrong_filter_flat_not_selected(tmp_path: Path) -> None:
    db, cam, _, tel = _db(tmp_path)
    v_flat = _mk_fits(tmp_path / "flat_v.fits", ccd_temp=-10.0, filter_name="V")
    cl.register_master(db, v_flat, kind="flat", camera_id=cam, telescope_id=tel)
    assert _find_flat(db, cam, tel, "R") is None


def test_flat_without_telescope_refused(tmp_path: Path) -> None:
    db, cam, _, _ = _db(tmp_path)
    flat = _mk_fits(tmp_path / "flat_v.fits", ccd_temp=-10.0, filter_name="V")
    with pytest.raises(ValueError, match="telescope"):
        cl.register_master(db, flat, kind="flat", camera_id=cam, telescope_id=None)
    assert db.conn.execute("SELECT COUNT(*) FROM CALIBRATION_LIBRARY").fetchone()[0] == 0


def test_unknown_light_temp_dark_match_fails(tmp_path: Path) -> None:
    db, cam, _, _ = _db(tmp_path)
    dark = _mk_fits(tmp_path / "dark.fits", ccd_temp=-10.0)
    cl.register_master(db, dark, kind="dark", camera_id=cam, telescope_id=None)
    assert _find(db, cam, ccd_temp=None) is None


def test_ccd_temp_within_and_outside_tolerance(tmp_path: Path) -> None:
    db, cam, _, _ = _db(tmp_path)
    dark = _mk_fits(tmp_path / "dark_tol.fits", ccd_temp=-10.2)
    cl.register_master(db, dark, kind="dark", camera_id=cam, telescope_id=None)
    assert Path(_find(db, cam, ccd_temp=-10.0, temp_tolerance=0.5)).resolve() == dark.resolve()
    assert _find(db, cam, ccd_temp=-5.0, temp_tolerance=0.5) is None


def test_config_temp_tolerance_default() -> None:
    from config import AppConfig

    cfg = AppConfig()
    assert cfg.calibration_master_ccd_temp_tolerance_c == 0.5


def test_dark_selection_honors_nondefault_tolerance(tmp_path: Path) -> None:
    """WAVE-B STEP 1: the CCD-temp tolerance governs library dark selection. Synthetic darks at dT
    0.4 / 0.6 / 5.0 C from the light CCD_TEMP; a nondefault tolerance changes which rows qualify."""
    light_temp = -10.0
    for dt, tol, hit in ((0.4, 0.5, True), (0.6, 0.5, False), (0.6, 0.7, True), (5.0, 1.0, False)):
        db, cam, _, _ = _db(tmp_path, f"d_{dt}_{tol}.db")
        d = _mk_fits(tmp_path / f"dark_{dt}_{tol}.fits", ccd_temp=light_temp - dt)
        cl.register_master(db, d, kind="dark", camera_id=cam, telescope_id=None)
        got = _find(db, cam, ccd_temp=light_temp, temp_tolerance=tol)
        assert (got is not None) == hit, (dt, tol)
