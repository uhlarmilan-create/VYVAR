"""ePSF candidate selection from masterstars_full_match.csv only (MS-SOURCES-RETIRE C1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from database import VyvarDatabase
from psf_photometry import _epsf_prepare_stars


def _write_min_masterstar(path: Path, *, n: int = 512) -> None:
    data = np.zeros((n, n), dtype=np.float32)
    hdr = fits.Header()
    hdr["NAXIS1"] = n
    hdr["NAXIS2"] = n
    hdr["VY_FWHM"] = 3.5
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def _write_min_csv(path: Path, *, with_xy: bool = True, drop_col: str | None = None) -> None:
    rows = []
    for i in range(40):
        row = {
            "catalog_id": f"GaiaDR3_{1000000000000000000 + i}",
            "catalog_known_variable": False,
            "likely_saturated": False,
            "photometry_ok": True,
            "is_saturated": False,
            "is_noisy": False,
            "is_usable": True,
            "ra_deg": 180.0 + i * 0.001,
            "dec_deg": 45.0,
            "mag": 12.0,
            "zone": "linear",
            "source_state": "DETECTED_P1",
        }
        if with_xy:
            row["x"] = 50.0 + i * 10.0
            row["y"] = 50.0 + i * 8.0
        rows.append(row)
    df = pd.DataFrame(rows)
    if drop_col and drop_col in df.columns:
        df = df.drop(columns=[drop_col])
    df.to_csv(path, index=False)


def test_epsf_prepare_stars_csv_path(tmp_path: Path) -> None:
    ms = tmp_path / "MASTERSTAR.fits"
    csv = tmp_path / "masterstars_full_match.csv"
    _write_min_masterstar(ms)
    _write_min_csv(csv)
    phot = tmp_path / "photometry"
    phot.mkdir()
    ids = [f"GaiaDR3_{1000000000000000000 + i}" for i in range(40)]
    pd.DataFrame({"catalog_id": ids[:10], "zone_flag": "linear"}).to_csv(
        phot / "active_targets.csv", index=False
    )
    pd.DataFrame({"catalog_id": ids[10:]}).to_csv(
        phot / "comparison_stars_per_target.csv", index=False
    )
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    try:
        prep = _epsf_prepare_stars(ms, csv, db, draft_id=1, min_stars=10)
        assert prep["n_join"] >= 10
        assert len(prep["stars"]) >= 10
    finally:
        db.close()


def test_epsf_prepare_missing_csv(tmp_path: Path) -> None:
    ms = tmp_path / "MASTERSTAR.fits"
    _write_min_masterstar(ms)
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    try:
        with pytest.raises(FileNotFoundError, match="masterstars_full_match.csv not found"):
            _epsf_prepare_stars(ms, tmp_path / "masterstars_full_match.csv", db, draft_id=1)
    finally:
        db.close()


def test_epsf_prepare_missing_required_column(tmp_path: Path) -> None:
    ms = tmp_path / "MASTERSTAR.fits"
    csv = tmp_path / "masterstars_full_match.csv"
    _write_min_masterstar(ms)
    _write_min_csv(csv, drop_col="x")
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            _epsf_prepare_stars(ms, csv, db, draft_id=1)
    finally:
        db.close()


def test_psf_photometry_has_no_master_sources_reads() -> None:
    text = Path(__file__).resolve().parents[2] / "src_py" / "psf_photometry.py"
    src = text.read_text(encoding="utf-8")
    assert "FROM MASTER_SOURCES" not in src
    assert "SELECT COUNT(*) AS n FROM MASTER_SOURCES" not in src
