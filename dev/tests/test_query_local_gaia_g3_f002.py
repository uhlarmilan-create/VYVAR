"""G3-F002: query_local_gaia mag_limit=None means no g_mag cap."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from database import query_local_gaia


def _make_mini_gaia_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE gaia_dr3 (
            source_id TEXT,
            ra REAL,
            dec REAL,
            g_mag REAL,
            bp_mag REAL,
            rp_mag REAL,
            bp_rp REAL,
            var_flag TEXT
        );
        """
    )
    rows = [
        ("bright", 180.0, 45.0, 10.0, 10.0, 9.5, 0.5, ""),
        ("mid", 180.01, 45.01, 12.0, 12.0, 11.5, 0.5, ""),
        ("faint", 180.02, 45.02, 14.5, 14.5, 13.5, 1.0, ""),
    ]
    conn.executemany(
        "INSERT INTO gaia_dr3 VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


@pytest.fixture
def mini_gaia_db(tmp_path: Path) -> Path:
    p = tmp_path / "gaia_mini.sqlite"
    _make_mini_gaia_db(p)
    return p


def test_explicit_mag_limit_11_5_excludes_faint(mini_gaia_db: Path) -> None:
    rows = query_local_gaia(
        mini_gaia_db,
        ra_min=179.9,
        ra_max=180.2,
        dec_min=44.9,
        dec_max=45.2,
        mag_limit=11.5,
    )
    g_vals = sorted(float(r["g_mag"]) for r in rows)
    assert g_vals == [10.0]


def test_mag_limit_none_includes_fainter_than_11_5(mini_gaia_db: Path) -> None:
    rows = query_local_gaia(
        mini_gaia_db,
        ra_min=179.9,
        ra_max=180.2,
        dec_min=44.9,
        dec_max=45.2,
        mag_limit=None,
    )
    g_vals = sorted(float(r["g_mag"]) for r in rows)
    assert g_vals == [10.0, 12.0, 14.5]


def test_explicit_mag_limit_20_matches_prior_cap_semantics(mini_gaia_db: Path) -> None:
    rows = query_local_gaia(
        mini_gaia_db,
        ra_min=179.9,
        ra_max=180.2,
        dec_min=44.9,
        dec_max=45.2,
        mag_limit=20.0,
        max_rows=2000,
    )
    assert len(rows) == 3


def test_explicit_mag_limit_20_byte_identical_row_set(mini_gaia_db: Path) -> None:
    """Explicit cap unchanged: all stars in box when mag_limit=20."""
    rows = query_local_gaia(
        mini_gaia_db,
        ra_min=179.9,
        ra_max=180.2,
        dec_min=44.9,
        dec_max=45.2,
        mag_limit=20.0,
    )
    assert {r["source_id"] for r in rows} == {"bright", "mid", "faint"}


def test_master_sources_bbox_faint_detection_match(mini_gaia_db: Path) -> None:
    """Simulate MASTER_SOURCES bbox query: faint Gaia now in catalog for 2 arcsec match."""
    det_ra, det_dec = 180.02001, 45.02001
    ga = query_local_gaia(
        mini_gaia_db,
        ra_min=det_ra - 0.01,
        ra_max=det_ra + 0.01,
        dec_min=det_dec - 0.01,
        dec_max=det_dec + 0.01,
        mag_limit=None,
    )
    assert any(float(r["g_mag"]) > 11.5 for r in ga)
    gcoo = SkyCoord(
        ra=[float(r["ra"]) for r in ga] * u.deg,
        dec=[float(r["dec"]) for r in ga] * u.deg,
    )
    dcoo = SkyCoord(ra=det_ra * u.deg, dec=det_dec * u.deg)
    idx, sep2d, _ = dcoo.match_to_catalog_sky(gcoo)
    assert sep2d.to(u.arcsec).value <= 2.0

    ga_capped = query_local_gaia(
        mini_gaia_db,
        ra_min=det_ra - 0.01,
        ra_max=det_ra + 0.01,
        dec_min=det_dec - 0.01,
        dec_max=det_dec + 0.01,
        mag_limit=11.5,
    )
    assert not any(float(r["g_mag"]) > 11.5 for r in ga_capped)
