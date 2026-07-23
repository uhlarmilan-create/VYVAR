"""Local exoplanet host DB reader + informational annotation (no comp filtering)."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

from database import query_local_exoplanet, validate_exoplanet_local_db_schema
from pipeline import (
    _exo_host_annotation_arrays,
    _query_exoplanet_local,
    select_comparison_stars_spatial_grid,
)


def _make_exo_db(path: Path, rows: list[tuple]) -> None:
    con = sqlite3.connect(str(path))
    con.execute(
        """
        CREATE TABLE exoplanet_data (
            obj_id TEXT PRIMARY KEY,
            name TEXT,
            host_name TEXT,
            ra_deg REAL,
            dec_deg REAL,
            cat_source TEXT,
            disposition TEXT,
            period REAL,
            mag REAL,
            mag_band TEXT
        );
        """
    )
    con.executemany(
        "INSERT INTO exoplanet_data VALUES (?,?,?,?,?,?,?,?,?,?);",
        rows,
    )
    con.commit()
    con.close()


def test_validate_exoplanet_schema_ok(tmp_path: Path):
    db = tmp_path / "exo.db"
    _make_exo_db(
        db,
        [
            (
                "K2-18 b",
                "K2-18 b",
                "K2-18",
                172.56,
                7.59,
                "CONFIRMED",
                "CONFIRMED",
                9.0,
                11.0,
                "K",
            ),
        ],
    )
    ok, code = validate_exoplanet_local_db_schema(db)
    assert ok
    assert code == "ok"


def test_query_local_exoplanet_missing_file_returns_empty(tmp_path: Path):
    assert query_local_exoplanet(tmp_path / "nope.db", ra_min=0, ra_max=1, dec_min=0, dec_max=1) == []


def test_query_local_exoplanet_box_and_cone(tmp_path: Path):
    db = tmp_path / "exo.db"
    _make_exo_db(
        db,
        [
            ("host1", "pl1", "Host-1", 10.0, 10.0, "CONFIRMED", "CONFIRMED", 1.0, 10.0, "V"),
            ("host2", "pl2", "Host-2", 10.01, 10.01, "TOI", "PC", 2.0, 11.0, "T"),
            ("far", "pl3", "Far", 50.0, 50.0, "TOI", "PC", 3.0, 12.0, "T"),
        ],
    )
    rows = query_local_exoplanet(
        db,
        ra_min=9.5,
        ra_max=10.5,
        dec_min=9.5,
        dec_max=10.5,
    )
    assert len(rows) == 2
    center = SkyCoord(10.0, 10.0, unit=(u.deg, u.deg), frame="icrs")
    df = _query_exoplanet_local(center=center, radius_deg=0.05, exoplanet_db_path=db)
    assert len(df) == 2
    assert set(df["obj_id"]) == {"host1", "host2"}


def test_query_local_exoplanet_ra_wrap(tmp_path: Path):
    db = tmp_path / "exo.db"
    _make_exo_db(
        db,
        [
            ("w1", "p", "Wrap", 359.9, 0.5, "CONFIRMED", "CONFIRMED", 1.0, 10.0, "V"),
        ],
    )
    rows = query_local_exoplanet(db, ra_min=359.5, ra_max=360.5, dec_min=0.0, dec_max=1.0)
    assert len(rows) == 1


def test_exo_match_nearest_within_3_arcsec(tmp_path: Path):
    db = tmp_path / "exo.db"
    _make_exo_db(
        db,
        [
            ("near", "p", "Near-Host", 100.0, 20.0, "CONFIRMED", "CONFIRMED", 1.0, 10.0, "V"),
            ("far", "p2", "Far-Host", 100.1, 20.1, "TOI", "PC", 2.0, 11.0, "T"),
        ],
    )
    exo_df = _query_exoplanet_local(
        center=SkyCoord(100.0, 20.0, unit=(u.deg, u.deg), frame="icrs"),
        radius_deg=0.05,
        exoplanet_db_path=db,
    )
    det = SkyCoord([100.0], [20.0], unit=(u.deg, u.deg), frame="icrs")
    ann, warns = _exo_host_annotation_arrays(det, exo_df, 3.0)
    assert ann["exo_host_obj_id"][0] == "near"
    assert ann["exo_host_name"][0] == "Near-Host"
    assert math.isclose(float(ann["exo_match_sep_arcsec"][0]), 0.0, abs_tol=0.01)
    assert not warns


def test_exo_match_double_host_warning(tmp_path: Path, caplog):
    db = tmp_path / "exo.db"
    _make_exo_db(
        db,
        [
            ("h1", "p1", "Host-A", 100.0, 20.0, "CONFIRMED", "CONFIRMED", 1.0, 10.0, "V"),
            ("h2", "p2", "Host-B", 100.0, 20.0001, "TOI", "PC", 2.0, 11.0, "T"),
        ],
    )
    exo_df = _query_exoplanet_local(
        center=SkyCoord(100.0, 20.0, unit=(u.deg, u.deg), frame="icrs"),
        radius_deg=0.05,
        exoplanet_db_path=db,
    )
    det = SkyCoord([100.0], [20.0], unit=(u.deg, u.deg), frame="icrs")
    caplog.set_level("WARNING")
    ann, warns = _exo_host_annotation_arrays(det, exo_df, 3.0)
    assert ann["exo_host_obj_id"][0] in ("h1", "h2")
    assert warns or any("EXO MATCH" in r.message for r in caplog.records)


def test_comp_exclusion_does_not_use_exoplanet_columns():
    import inspect

    from tests.cython_compat import skip_if_compiled

    skip_if_compiled(
        "comp_selection_per_target",
        "inspect.getsource requires interpreted comp_selection_per_target.py",
    )
    src = inspect.getsource(select_comparison_stars_spatial_grid)
    assert "catalog_known_variable" in src
    assert "exo_host" not in src
    assert "exoplanet" not in src.lower()


def test_catalog_known_variable_not_includes_exoplanet():
    from tests.cython_compat import skip_if_compiled

    skip_if_compiled(
        "pipeline",
        "source text scan requires interpreted pipeline.py",
    )
    import pipeline

    text = Path(pipeline.__file__).resolve()
    for line in text.read_text(encoding="utf-8").splitlines():
        if "catalog_known_variable" in line and "exo" in line.lower():
            raise AssertionError(f"exo referenced on catalog_known_variable line: {line.strip()}")
