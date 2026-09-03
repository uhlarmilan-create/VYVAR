# -*- coding: ascii -*-
"""Field-run findings #11-#13 (FI Boo / Milan Linux preview)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

from database import VSXCatalogError, require_vsx_local_db_path
from infolog import clear_log, get_lines
from pipeline import _query_vsx_local, _query_vsx_local_frame_bbox, write_photometry_plan_files
from photometry_core import run_full_photometry_pipeline


def _make_vsx_db(path: Path, rows: list[tuple[int, float, float]]) -> None:
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE vsx_data (oid INTEGER PRIMARY KEY, ra_deg REAL, dec_deg REAL)")
    con.executemany("INSERT INTO vsx_data (oid, ra_deg, dec_deg) VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()


def test_require_vsx_local_db_path_missing_raises() -> None:
    with pytest.raises(VSXCatalogError, match="vsx_local_db_path"):
        require_vsx_local_db_path("")
    with pytest.raises(VSXCatalogError, match="vsx_local_db_path"):
        require_vsx_local_db_path(None)


def test_require_vsx_local_db_path_zero_rows_raises(tmp_path: Path) -> None:
    db = tmp_path / "empty_vsx.db"
    _make_vsx_db(db, [])
    with pytest.raises(VSXCatalogError, match="zero rows"):
        require_vsx_local_db_path(db)


def test_vsx_cone_zero_in_field_logs_not_raises(tmp_path: Path) -> None:
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    clear_log()
    db = tmp_path / "vsx.db"
    _make_vsx_db(db, [(1, 0.0, 0.0)])
    center = SkyCoord(ra=180.0 * u.deg, dec=45.0 * u.deg, frame="icrs")
    out = _query_vsx_local(
        center=center,
        radius_deg=0.5,
        vsx_db_path=db,
        require_db=True,
    )
    assert out.empty
    joined = " ".join(get_lines())
    assert "VSX cone=0" in joined
    assert "field genuinely empty" in joined
    assert "1 total rows" in joined


def test_vsx_frame_bbox_zero_in_field_logs_not_raises(tmp_path: Path) -> None:
    clear_log()
    db = tmp_path / "vsx.db"
    _make_vsx_db(db, [(1, 0.0, 0.0)])
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [256.0, 256.0]
    wcs.wcs.crval = [180.0, 45.0]
    wcs.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
    out = _query_vsx_local_frame_bbox(
        wcs=wcs,
        width_px=512,
        height_px=512,
        vsx_db_path=db,
        require_db=True,
    )
    assert out.empty
    joined = " ".join(get_lines())
    assert "VSX cone=0" in joined
    assert "field genuinely empty" in joined


def test_border_deferred_when_no_aligned_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ps = tmp_path / "Drafts" / "draft_000001" / "platesolve" / "NoFilter_60_2"
    ps.mkdir(parents=True)
    ms_csv = ps / "masterstars_full_match.csv"
    ms_csv.write_text("catalog_id,ra_deg,dec_deg,mag\n", encoding="ascii")
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 64
    hdr["NAXIS2"] = 64
    hdr["VY_FWHM"] = 3.5
    w = WCS(naxis=2)
    w.wcs.crpix = [32.0, 32.0]
    w.wcs.crval = [180.0, 45.0]
    w.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
    hdr.update(w.to_header())
    ms_fits = ps / "MASTERSTAR.fits"
    fits.writeto(ms_fits, np.zeros((64, 64), dtype=np.float32), header=hdr, overwrite=True)

    vsx = tmp_path / "vsx.db"
    _make_vsx_db(vsx, [(1, 180.0, 45.0)])

    class _Cfg:
        vsx_local_db_path = str(vsx)
        aperture_fwhm_factor = 1.35
        annulus_inner_fwhm = 2.7
        annulus_outer_fwhm = 5.2
        gaia_db_path = ""
        exoplanet_local_db_path = ""

    monkeypatch.setattr("config.AppConfig", lambda *a, **k: _Cfg())

    clear_log()
    write_photometry_plan_files(
        platesolve_dir=ps,
        masterstar_fits=ms_fits,
        masterstars_csv=ms_csv,
    )

    joined = " ".join(get_lines())
    assert "Deferred: no aligned proc_*.fits on disk yet" in joined


def test_run_full_photometry_zero_targets_skips_phase2a(tmp_path: Path) -> None:
    out = tmp_path / "photometry"
    out.mkdir()
    vt = tmp_path / "variable_targets.csv"
    vt.write_text("name,catalog_id,catalog,ra_deg,dec_deg\n", encoding="ascii")
    ms = tmp_path / "masterstars.csv"
    ms.write_text(
        "catalog_id,ra_deg,dec_deg,mag,name,x,y,zone\n",
        encoding="ascii",
    )
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 32
    hdr["NAXIS2"] = 32
    w = WCS(naxis=2)
    w.wcs.crpix = [16.0, 16.0]
    w.wcs.crval = [180.0, 45.0]
    w.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
    hdr.update(w.to_header())
    ms_fits = tmp_path / "MASTERSTAR.fits"
    fits.writeto(ms_fits, np.zeros((32, 32), dtype=np.float32), header=hdr, overwrite=True)
    pf = tmp_path / "per_frame"
    pf.mkdir()
    dt = tmp_path / "aligned"
    dt.mkdir()

    result = run_full_photometry_pipeline(
        masterstar_fits_path=ms_fits,
        variable_targets_csv=vt,
        masterstars_csv=ms,
        per_frame_csv_dir=pf,
        detrended_aligned_dir=dt,
        output_dir=out,
    )
    assert result.get("zero_targets") is True
    assert int(result.get("n_active_targets") or 0) == 0
    assert result.get("phase2a") is None
    assert not (out / "photometry_summary.csv").exists()


def test_resolve_import_location_id_fresh_bootstrap_no_warning(tmp_path: Path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    from database import VyvarDatabase

    db = VyvarDatabase(str(db_path))
    try:
        db.conn.execute(
            "INSERT INTO LOCATION (ID, PLACENAME, LATITUDE, LONGITUDE, IS_DEFAULT) VALUES (1, 'Home', 0, 0, 1);"
        )
        db.conn.commit()
        lid, warn = db.resolve_import_location_id(id_location=None, cfg_location_id=1)
        assert lid == 1
        assert warn is None
    finally:
        db.conn.close()
