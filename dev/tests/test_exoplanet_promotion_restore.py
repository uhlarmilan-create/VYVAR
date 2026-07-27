"""Exoplanet promotion path resolution, fail-loud, and VT schema stability."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from config import AppConfig
from database import ExoplanetCatalogError, require_exoplanet_local_db_path
from pipeline import _build_exoplanet_promotion_rows_from_masterstars, _merge_vsx_exoplanet_variable_targets


def _make_exo_db(tmp_path: Path, *, ra: float, dec: float, obj_id: str = "TOI-TEST.01") -> Path:
    import sqlite3

    db = tmp_path / "exo.db"
    con = sqlite3.connect(db)
    con.execute(
        """
        CREATE TABLE exoplanet_data (
            obj_id TEXT, host_name TEXT, ra_deg REAL, dec_deg REAL,
            cat_source TEXT, disposition TEXT, pl_name TEXT, hostname TEXT,
            sy_snum REAL, sy_pnum REAL
        );
        """
    )
    con.execute(
        "INSERT INTO exoplanet_data VALUES (?,?,?,?,?,?,?,?,?,?);",
        (obj_id, "TOI-TEST", ra, dec, "TOI", "PC", "TOI-TEST b", "TOI-TEST", 1.0, 1.0),
    )
    con.commit()
    con.close()
    return db


def _masterstar_fits(tmp_path: Path, *, ra: float, dec: float) -> Path:
    import numpy as np
    from astropy.io.fits import PrimaryHDU

    w = WCS(naxis=2)
    w.wcs.crpix = [512.0, 512.0]
    w.wcs.crval = [ra, dec]
    w.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    data = np.zeros((1024, 1024), dtype=np.float32)
    p = tmp_path / "MASTERSTAR.fits"
    hdu = PrimaryHDU(data=data, header=w.to_header())
    hdu.writeto(p, overwrite=True)
    return p


def test_merge_preserves_exo_columns_when_exo_frame_empty() -> None:
    vsx = pd.DataFrame([{"name": "V", "catalog_id": "1111111111111111111", "catalog": "VSX"}])
    merged = _merge_vsx_exoplanet_variable_targets(vsx, pd.DataFrame())
    for col in (
        "exo_host_obj_id",
        "exo_host_name",
        "exo_cat_source",
        "exo_disposition",
        "exo_match_sep_arcsec",
        "target_origin",
    ):
        assert col in merged.columns
        assert str(merged.iloc[0][col]) == ""


def test_require_exoplanet_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing_exo.db"
    with pytest.raises(ExoplanetCatalogError, match="not found"):
        require_exoplanet_local_db_path(missing)


def test_promotion_from_synthetic_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ra, dec = 210.0, 41.0
    db = _make_exo_db(tmp_path, ra=ra, dec=dec)
    ms = pd.DataFrame(
        [
            {
                "catalog_id": "1497132660589966976",
                "ra_deg": ra,
                "dec_deg": dec,
                "x": 512.0,
                "y": 512.0,
                "mag": 12.5,
                "zone": "linear",
            }
        ]
    )
    fits_path = _masterstar_fits(tmp_path, ra=ra, dec=dec)
    cfg = AppConfig()
    monkeypatch.setattr(cfg, "exoplanet_local_db_path", str(db))
    monkeypatch.setattr(cfg, "exoplanet_match_max_sep_arcsec", 3.0)
    rows = _build_exoplanet_promotion_rows_from_masterstars(
        ms, fits.getheader(fits_path), cfg, frame_w_px=1024, frame_h_px=1024
    )
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["catalog"] == "EXOPLANET"
    assert row["exo_host_obj_id"] == "TOI-TEST.01"
    assert row["target_origin"] == "EXOPLANET"


def test_config_resolves_relative_exoplanet_path_against_data_root() -> None:
    cfg = AppConfig()
    p = Path(cfg.exoplanet_local_db_path)
    assert p.is_absolute()
    assert p.is_file()
