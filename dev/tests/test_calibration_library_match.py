"""G3-F001: unified scoped calibration library master selection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from database import VyvarDatabase
from importer import (
    _calibration_light_temp_c,
    _find_matching_master_in_library,
    _format_temp_deg_for_name,
    _params_string,
)
from pipeline import extract_fits_metadata


def _mk_fits(path: Path) -> None:
    fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32)).writeto(path, overwrite=True)


def _register(
    db: VyvarDatabase,
    path: Path,
    *,
    kind: str,
    xbinning: int = 2,
    exptime: float = 60.0,
    ccd_temp: float | None = -10.0,
    gain: int = 0,
    filter_name: str = "",
    id_eq: int | None = 1,
    id_tel: int | None = 1,
) -> bool:
    _mk_fits(path)
    return db.register_calibration_library_entry(
        kind=kind,
        file_path=path,
        xbinning=xbinning,
        exptime=exptime,
        ccd_temp=ccd_temp,
        filter_name=filter_name,
        gain=gain,
        id_equipments=id_eq,
        id_telescope=id_tel,
    )


def _find(
    db: VyvarDatabase,
    *,
    kind: str = "dark",
    ccd_temp: float | None = -10.0,
    filter_name: str = "",
    exptime: float = 60.0,
    id_eq: int = 1,
    id_tel: int = 1,
    temp_tolerance: float = 0.5,
) -> str | None:
    return db.find_best_calibration_library_path(
        kind=kind,
        xbinning=2,
        exptime=exptime,
        ccd_temp=ccd_temp,
        filter_name=filter_name,
        gain=0,
        temp_tolerance=temp_tolerance,
        prefer_unbinned_master=False,
        id_equipments=id_eq,
        id_telescope=id_tel,
    )


def _insert_library_row(
    db: VyvarDatabase,
    path: Path,
    *,
    kind: str,
    xbinning: int = 2,
    exptime: float = 60.0,
    ccd_temp: float | None = -10.0,
    gain: int = 0,
    filter_name: str = "",
    id_eq: int | None = 1,
    id_tel: int | None = 1,
) -> None:
    """Direct SQL insert for rows registration refuses (legacy / invalid)."""
    _mk_fits(path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.conn.execute(
        """
        INSERT INTO CALIBRATION_LIBRARY (
            KIND, FILE_PATH, XBINNING, EXPTIME, CCD_TEMP, FILTER_NAME, GAIN,
            REGISTERED_AT, ID_EQUIPMENTS, ID_TELESCOPE
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            kind,
            str(path.resolve()),
            xbinning,
            exptime,
            ccd_temp,
            filter_name,
            gain,
            now,
            id_eq,
            id_tel,
        ),
    )
    db.conn.commit()


def test_proper_scoped_dark_still_selected(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    good = tmp_path / "good_dark.fits"
    assert _register(db, good, kind="dark", ccd_temp=-10.0)
    hit = _find(db, ccd_temp=-10.0)
    assert hit is not None
    assert Path(hit).resolve() == good.resolve()


def test_flat_different_exptime_still_matches(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    flat = tmp_path / "flat_v.fits"
    assert _register(
        db,
        flat,
        kind="flat",
        filter_name="V",
        exptime=5.0,
        ccd_temp=-10.0,
    )
    hit = _find(
        db,
        kind="flat",
        filter_name="V",
        exptime=120.0,
        ccd_temp=None,
    )
    assert hit is not None
    assert Path(hit).resolve() == flat.resolve()


def test_null_temp_dark_not_selected(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    bad = tmp_path / "null_temp_dark.fits"
    assert not _register(db, bad, kind="dark", ccd_temp=None)
    _insert_library_row(db, bad, kind="dark", ccd_temp=None)
    assert _find(db, ccd_temp=-10.0) is None


def test_global_scope_dark_not_selected(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    global_dark = tmp_path / "global_dark.fits"
    assert not _register(db, global_dark, kind="dark", id_eq=None, id_tel=None)
    _insert_library_row(db, global_dark, kind="dark", id_eq=None, id_tel=None)
    assert _find(db, ccd_temp=-10.0, id_eq=1, id_tel=1) is None


def test_wrong_telescope_dark_rejected(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    other_tel = tmp_path / "other_tel_dark.fits"
    assert _register(db, other_tel, kind="dark", id_eq=1, id_tel=2)
    assert _find(db, ccd_temp=-10.0, id_eq=1, id_tel=1) is None


def test_wrong_filter_flat_not_selected(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    v_flat = tmp_path / "flat_v.fits"
    assert _register(db, v_flat, kind="flat", filter_name="V")
    assert _find(db, kind="flat", filter_name="R", ccd_temp=None) is None


def test_unknown_light_temp_dark_match_fails(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    dark = tmp_path / "dark.fits"
    assert _register(db, dark, kind="dark", ccd_temp=-10.0)
    assert _find(db, ccd_temp=None) is None


def test_ccd_temp_within_tolerance_matches(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    dark = tmp_path / "dark_tol.fits"
    assert _register(db, dark, kind="dark", ccd_temp=-10.2)
    hit = _find(db, ccd_temp=-10.0, temp_tolerance=0.5)
    assert hit is not None
    assert Path(hit).resolve() == dark.resolve()


def test_ccd_temp_outside_tolerance_rejects(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    dark = tmp_path / "dark_hot.fits"
    assert _register(db, dark, kind="dark", ccd_temp=-5.0)
    assert _find(db, ccd_temp=-10.0, temp_tolerance=0.5) is None


def test_scoped_preferred_when_both_scoped_and_global_on_disk(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    scoped = tmp_path / "scoped_dark.fits"
    global_dark = tmp_path / "global_dark.fits"
    assert _register(db, scoped, kind="dark", ccd_temp=-10.0, id_eq=1, id_tel=1)
    _insert_library_row(db, global_dark, kind="dark", id_eq=None, id_tel=None)
    hit = _find(db, ccd_temp=-10.0)
    assert hit is not None
    assert Path(hit).resolve() == scoped.resolve()


def test_register_without_scope_refused(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    path = tmp_path / "no_scope.fits"
    assert not db.register_calibration_library_entry(
        kind="dark",
        file_path=path,
        xbinning=2,
        exptime=60.0,
        ccd_temp=-10.0,
        gain=0,
        id_equipments=None,
        id_telescope=None,
    )
    rows = db.conn.execute("SELECT COUNT(*) FROM CALIBRATION_LIBRARY").fetchone()[0]
    assert rows == 0


def test_fallback_rejects_global_master_when_db_has_no_match(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    lib = tmp_path / "library"
    lib.mkdir()
    global_dark = lib / "global_dark.fits"
    _insert_library_row(db, global_dark, kind="dark", id_eq=None, id_tel=None)
    hit = _find_matching_master_in_library(
        lib,
        kind="dark",
        exp=60.0,
        gain=0,
        binning=2,
        temp=-10.0,
        flt=None,
        db=db,
        id_equipments=1,
        id_telescope=1,
    )
    assert hit is None


def test_fallback_rejects_unregistered_file(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.db")
    lib = tmp_path / "library"
    lib.mkdir()
    orphan = lib / "orphan_dark.fits"
    _mk_fits(orphan)
    hit = _find_matching_master_in_library(
        lib,
        kind="dark",
        exp=60.0,
        gain=0,
        binning=2,
        temp=-10.0,
        flt=None,
        db=db,
        id_equipments=1,
        id_telescope=1,
    )
    assert hit is None


def test_null_ccd_temp_non_calibration_safe_calibration_dark_fails(tmp_path: Path) -> None:
    """NULL/missing CCD_TEMP: meta stays float 0.0 for legacy paths; dark match fails loud."""
    db = VyvarDatabase(tmp_path / "vyvar.db")
    light = tmp_path / "light_no_temp.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32))
    hdu.header["EXPTIME"] = 60.0
    hdu.header["XBINNING"] = 2
    hdu.header["GAIN"] = 0
    hdu.writeto(light, overwrite=True)

    meta = extract_fits_metadata(light, db=db)
    assert meta["temp"] == 0.0
    assert float(meta["temp"]) == 0.0

    scanning_id = db.derive_scanning_id(meta)
    assert scanning_id > 0

    params = _params_string(meta, include_filter=True)
    assert "Temp=0C" in params.replace(" ", "")

    temp_token = _format_temp_deg_for_name(float(meta["temp"]))
    assert temp_token

    cache_key = db.fits_header_cache_file_key(light)
    db.conn.execute("DELETE FROM FITS_HEADER_CACHE WHERE FILE_PATH = ?", (cache_key,))
    db.conn.commit()

    assert _calibration_light_temp_c(light, db=db) is None

    dark = tmp_path / "scoped_dark.fits"
    assert _register(db, dark, kind="dark", ccd_temp=-10.0)
    lib = tmp_path / "library"
    lib.mkdir()
    hit = _find_matching_master_in_library(
        lib,
        kind="dark",
        exp=60.0,
        gain=0,
        binning=2,
        temp=None,
        flt=None,
        db=db,
        id_equipments=1,
        id_telescope=1,
    )
    assert hit is None
    assert _find(db, ccd_temp=None) is None

    meta_cached = extract_fits_metadata(light, db=db)
    assert meta_cached["temp"] == 0.0
    assert float(meta_cached["temp"]) == 0.0


def test_config_temp_tolerance_default() -> None:
    from config import AppConfig

    cfg = AppConfig()
    assert cfg.calibration_master_ccd_temp_tolerance_c == 0.5


def test_dark_selection_honors_nondefault_tolerance(tmp_path: Path) -> None:
    """WAVE-B STEP 1: the CCD-temp tolerance (now fed from cfg into smart_scan_source)
    governs library dark selection. Synthetic darks at dT 0.4 / 0.6 / 5.0 C from the light
    CCD_TEMP; a nondefault tolerance changes which rows qualify (rejected at 0.5, some
    accepted at 0.7, and a 5.0 gap rejected even at a bounded 1.0)."""
    light_temp = -10.0

    # dT = 0.4: accepted at the default 0.5 tolerance.
    db04 = VyvarDatabase(tmp_path / "d04.db")
    d04 = tmp_path / "dark_dt04.fits"
    assert _register(db04, d04, kind="dark", ccd_temp=light_temp - 0.4)
    hit04 = _find(db04, ccd_temp=light_temp, temp_tolerance=0.5)
    assert hit04 is not None and Path(hit04).resolve() == d04.resolve()

    # dT = 0.6: rejected at 0.5 but accepted at a nondefault 0.7 -> proves the knob is live.
    db06 = VyvarDatabase(tmp_path / "d06.db")
    d06 = tmp_path / "dark_dt06.fits"
    assert _register(db06, d06, kind="dark", ccd_temp=light_temp - 0.6)
    assert _find(db06, ccd_temp=light_temp, temp_tolerance=0.5) is None
    hit06 = _find(db06, ccd_temp=light_temp, temp_tolerance=0.7)
    assert hit06 is not None and Path(hit06).resolve() == d06.resolve()

    # dT = 5.0: rejected even at a generous-but-bounded 1.0 tolerance.
    db50 = VyvarDatabase(tmp_path / "d50.db")
    d50 = tmp_path / "dark_dt50.fits"
    assert _register(db50, d50, kind="dark", ccd_temp=light_temp - 5.0)
    assert _find(db50, ccd_temp=light_temp, temp_tolerance=1.0) is None
