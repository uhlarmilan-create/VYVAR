"""FRAME-DEPTH-2026-10 step 2: coverage record of the main Gaia DB and the online top-up of a field.

No network: the archive is replaced by a fetcher that serves a synthetic sky; "offline" is a
fetcher that raises (the same path a DNS / connection failure takes).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u

import gaia_coverage as gc


def _old_db(path: Path, *, dec=(30.0, 70.0), g_max=15.0, n=4000, seed=1) -> Path:
    """A main DB as built before the coverage record existed (no gaia_coverage table)."""
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE gaia_dr3 (source_id INTEGER PRIMARY KEY, ra REAL, dec REAL, g_mag REAL, bp_mag REAL,"
        " rp_mag REAL, bp_rp REAL, g_flux_error_rel REAL, parallax REAL, parallax_error REAL,"
        " parallax_over_error REAL, teff_gspphot REAL, logg_gspphot REAL, mh_gspphot REAL,"
        " distance_gspphot REAL, var_flag TEXT, non_single_star INTEGER)"
    )
    ra = rng.uniform(0, 360, n)
    de = rng.uniform(dec[0], dec[1], n)
    g = rng.uniform(8, g_max, n)
    g[0], de[0], de[1] = g_max, dec[0], dec[1]
    con.executemany(
        "INSERT INTO gaia_dr3 (source_id, ra, dec, g_mag) VALUES (?,?,?,?)",
        [(i + 1, float(a), float(d), float(m)) for i, (a, d, m) in enumerate(zip(ra, de, g))],
    )
    con.commit()
    con.close()
    return path


class _Sky:
    """Archive stand-in: answers the top-up ADQL from a synthetic deep catalogue."""

    def __init__(self, n=20000, seed=2):
        rng = np.random.default_rng(seed)
        self.df = pd.DataFrame({
            "SOURCE_ID": np.arange(10_000_000, 10_000_000 + n),
            "ra": rng.uniform(0, 360, n), "dec": rng.uniform(30, 70, n),
            "g_mag": rng.uniform(8, 20, n), "bp_rp": 0.8, "var_flag": "NOT_AVAILABLE",
            "non_single_star": 0,
        })
        self.calls: list[str] = []

    def __call__(self, adql: str) -> pd.DataFrame:
        import re

        self.calls.append(adql)
        ra_iv = [(float(a), float(b)) for a, b in re.findall(r"ra >= ([\d.]+) AND ra <= ([\d.]+)", adql)]
        d0, d1 = (float(v) for v in re.search(r"dec >= (-?[\d.]+) AND dec <= (-?[\d.]+)", adql).groups())
        g0 = float(re.search(r"phot_g_mean_mag > (-?[\d.]+)", adql).group(1))
        g1 = float(re.search(r"phot_g_mean_mag <= (-?[\d.]+)", adql).group(1))
        d = self.df
        m_ra = np.zeros(len(d), dtype=bool)
        for a, b in ra_iv:
            m_ra |= (d["ra"] >= a) & (d["ra"] <= b)
        m = m_ra & (d["dec"] >= d0) & (d["dec"] <= d1) & (d["g_mag"] > g0) & (d["g_mag"] <= g1)
        return d.loc[m].reset_index(drop=True)


def _count(db: Path, where: str) -> int:
    con = sqlite3.connect(db)
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM gaia_dr3 WHERE {where}").fetchone()[0])
    finally:
        con.close()


def test_existing_db_registered_as_all_sky_to_its_depth(tmp_path: Path) -> None:
    db = _old_db(tmp_path / "gaia.db")
    rows = gc.load_coverage(db)
    assert len(rows) == 1
    r = rows[0]
    assert (r["ra_min"], r["ra_max"]) == (0.0, 360.0)
    assert r["dec_min"] == pytest.approx(30.0) and r["dec_max"] == pytest.approx(70.0)
    assert r["g_max"] == pytest.approx(15.0) and r["source"] == "registered_existing_db"
    assert len(gc.load_coverage(db)) == 1, "registered once, then read back"
    # inside the Dec range, also across RA 0
    assert gc.covered_depth_g(db, (350.0, 370.0, 40.0, 41.0)) == pytest.approx(15.0)
    # leaving the Dec range of the DB: no record covers it (TODO-GAIA-COVERAGE-DEC)
    assert gc.covered_depth_g(db, (10.0, 12.0, 69.0, 71.0)) is None


def test_topup_fetches_field_box_and_records_depth(tmp_path: Path) -> None:
    db = _old_db(tmp_path / "gaia.db")
    sky = _Sky()
    center = SkyCoord(359.6 * u.deg, 50.0 * u.deg)  # field across RA 0
    res = gc.ensure_field_depth(db, center=center, radius_deg=0.5, g_needed=18.6, fetch=sky)
    assert res["status"] == "ok"
    assert res["depth_before"] == pytest.approx(15.0) and res["depth_after"] == pytest.approx(18.6)
    adql = sky.calls[0]
    assert "phot_g_mean_mag > 15.0000" in adql and "phot_g_mean_mag <= 18.6000" in adql
    assert adql.count("ra >= ") == 2, "RA 0 crossing split into two intervals"
    box = gc.field_box(center, 0.5, margin_deg=gc.TOPUP_MARGIN_DEG)
    assert box[2] == pytest.approx(48.5) and box[3] == pytest.approx(51.5), "+-1 deg around the field edges"
    n_expected = len(sky(adql))
    assert res["n_fetched"] == n_expected and res["n_new"] == n_expected > 0
    assert _count(db, "g_mag > 15.0") == n_expected
    # the next field inside the box is covered: no second archive call
    res2 = gc.ensure_field_depth(
        db, center=SkyCoord(0.2 * u.deg, 50.3 * u.deg), radius_deg=0.4, g_needed=18.0, fetch=sky
    )
    assert res2["status"] == "covered" and len(sky.calls) == 2  # (+1 call above for n_expected)
    # a field outside the box still has the base depth only
    assert gc.covered_depth_g(db, gc.field_box(SkyCoord(120 * u.deg, 50 * u.deg), 0.5)) == pytest.approx(15.0)
    # repeated top-up of the same stars inserts nothing twice (INSERT OR IGNORE)
    again = gc.top_up_field(db, box=box, g_from=15.0, g_to=18.6, fetch=sky)
    assert again["status"] == "ok" and again["n_new"] == 0


def test_offline_continues_with_local_db(tmp_path: Path) -> None:
    db = _old_db(tmp_path / "gaia.db")
    n0 = _count(db, "1=1")

    def _offline(_adql: str) -> pd.DataFrame:
        raise OSError("[Errno -2] Name or service not known")

    res = gc.ensure_field_depth(
        db, center=SkyCoord(100 * u.deg, 40 * u.deg), radius_deg=0.5, g_needed=17.0, fetch=_offline
    )
    assert res["status"] == "offline" and "Name or service" in res["error"]
    assert res["depth_after"] == pytest.approx(15.0)
    assert _count(db, "1=1") == n0
    assert len(gc.load_coverage(db)) == 1, "a failed download never claims depth"


def test_frame_within_db_needs_no_topup(tmp_path: Path) -> None:
    db = _old_db(tmp_path / "gaia.db")
    res = gc.ensure_field_depth(
        db, center=SkyCoord(100 * u.deg, 40 * u.deg), radius_deg=0.5, g_needed=14.2,
        fetch=lambda _q: pytest.fail("no archive call when the DB is deep enough"),
    )
    assert res["status"] == "covered"


def test_builder_shares_schema_and_insert(tmp_path: Path) -> None:
    db = tmp_path / "new.db"
    con = sqlite3.connect(db)
    gc.init_gaia_tables(con)
    df = pd.DataFrame({"SOURCE_ID": [5, 6], "RA": [1.0, 2.0], "DEC": [3.0, 4.0], "G_MAG": [12.0, 13.0],
                       "var_flag": ["VARIABLE", None]})
    assert gc.insert_gaia_rows(con, df) == 2
    assert gc.insert_gaia_rows(con, df) == 0
    con.commit()
    con.close()
    gc.record_coverage(db, box=(0.0, 360.0, -30.0, 90.0), g_max=16.5, source="build_gaia_catalog")
    assert gc.covered_depth_g(db, (10.0, 11.0, 0.0, 1.0)) == pytest.approx(16.5)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "catalogs"))
    import build_gaia_catalog as b

    q = b.build_adql(0, 7.5, 0, 5, last_ra=False, last_dec=False, mag_limit=16.5, use_full_source=True)
    assert "g_flux_error_rel" in q and "FROM gaiadr3.gaia_source\n" in q


def test_frame_depth_completeness_only_within_catalogue_depth() -> None:
    """Stars a neighbouring top-up left beyond the field depth are a partial sample: kept out of the curve."""
    from frame_depth import measure_frame_depth

    rng = np.random.default_rng(5)
    n = 6000
    g = 10.0 + 8.0 * rng.random(n) ** 0.35
    x = rng.uniform(0, 2000, n)
    y = rng.uniform(0, 2000, n)
    from scipy.special import erf

    det = rng.random(n) < 0.5 * (1.0 + erf((15.0 - g) / (np.sqrt(2.0) * 0.35)))
    peak = 10 ** (-0.4 * (g[det] - 25.0)) * 10 ** (0.4 * rng.normal(0, 0.35, det.sum()))
    keep = (g <= 14.0) | (x < 300)  # deeper stars only in one corner (partial top-up)
    fd = measure_frame_depth(
        det_x=x[det], det_y=y[det], det_peak=peak, gaia_x=x[keep], gaia_y=y[keep], gaia_g=g[keep],
        match_r_px=1.5, threshold_adu=10 ** (-0.4 * (15.0 - 25.0)), wpx=2000, h=2000, db_depth_g=14.0,
    )
    assert max(b["bin_hi"] for b in fd["completeness_bins"]) <= 14.0 + 1e-9
    assert fd["method"] == "detector_estimate" and fd["limited_by_catalogue"]
    assert fd["g90"] == pytest.approx(fd["g50"] - 1.2815516 * fd["sigma_mag"], abs=2e-3)


def test_one_depth_no_global_db_max() -> None:
    """The global MAX(g_mag) of the DB and the static 18 mag MASTERSTAR floor are gone."""
    src = Path(__file__).resolve().parents[2] / "src_py"
    import database

    assert not hasattr(database, "get_gaia_db_max_g_mag")
    for name in ("dao_reconcile.py", "catalog_match.py", "masterstar_build.py",
                 "vsx_gaia_crossmatch.py", "pipeline_catalog.py", "astrometry_align.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "get_gaia_db_max_g_mag" not in text, name
        assert "faintest_mag_limit" not in text, name
    assert "MASTERSTAR_FAINTEST_MAG_FLOOR" not in (src / "dao_reconcile.py").read_text(encoding="utf-8")


def test_read_only_db_is_not_extended_and_build_continues(tmp_path: Path, monkeypatch) -> None:
    db = _old_db(tmp_path / "gaia.db")
    real_connect = sqlite3.connect

    def _ro(path, *a, **k):
        return real_connect(f"file:{path}?mode=ro", uri=True)

    monkeypatch.setattr(gc.sqlite3, "connect", _ro)
    res = gc.ensure_field_depth(
        db, center=SkyCoord(100 * u.deg, 40 * u.deg), radius_deg=0.5, g_needed=17.0,
        fetch=lambda _q: pytest.fail("no download when the DB cannot take it"),
    )
    assert res["status"] == "db_not_writable"
    assert res["depth_before"] == pytest.approx(15.0), "in-memory registration still gives the depth"
    monkeypatch.setattr(gc.sqlite3, "connect", real_connect)
    assert _count(db, "1=1") == 4000


def test_unfinished_build_claims_no_depth(tmp_path: Path) -> None:
    """build_gaia_catalog.py creates gaia_coverage at start and fills it after the last strip."""
    db = _old_db(tmp_path / "gaia.db")
    con = sqlite3.connect(db)
    gc.init_gaia_tables(con)
    con.close()
    assert gc.load_coverage(db) == []
    assert gc.covered_depth_g(db, (10.0, 11.0, 40.0, 41.0)) is None


def test_polar_db_registered_to_the_pole(tmp_path: Path) -> None:
    db = _old_db(tmp_path / "gaia.db", dec=(30.0, 89.95))
    rec = gc.load_coverage(db)[0]
    assert rec["dec_max"] == 90.0
    box = gc.field_box(SkyCoord(10 * u.deg, 89.5 * u.deg), 0.8)
    assert box[3] == 90.0 and gc.covered_depth_g(db, box) == pytest.approx(15.0)


def test_builder_depth_is_the_smallest_strip_limit(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "catalogs"))
    import build_gaia_catalog as b

    db = _old_db(tmp_path / "gaia.db", g_max=15.0)
    con = sqlite3.connect(db)
    gc.init_gaia_tables(con)
    b._ensure_strip_mag_column(con)
    b.mark_strip_done(con, (0.0, 180.0, 30.0, 70.0), 10, 17.0)
    con.execute(  # strip of an older build: no limit stored -> its faintest star (15.0)
        "INSERT INTO strip_progress (strip_key, ra_min, ra_max, dec_min, dec_max, n_stars) VALUES (?,?,?,?,?,?)",
        (b.strip_key((180.0, 360.0, 30.0, 70.0)), 180.0, 360.0, 30.0, 70.0, 10),
    )
    keys = {b.strip_key((0.0, 180.0, 30.0, 70.0)), b.strip_key((180.0, 360.0, 30.0, 70.0))}
    assert b.built_depth_g(con, keys) <= 15.0 + 1e-9
    assert b.built_depth_g(con, {b.strip_key((0.0, 180.0, 30.0, 70.0))}) == pytest.approx(17.0)
    con.close()


def test_provenance_ignores_coverage_timestamps() -> None:
    from catalog_provenance import catalog_fingerprints_equal

    base = {"fingerprint_sha256": "x", "size_bytes": 1, "row_count": 2}
    a = dict(base, coverage=[{"ra_min": 0.0, "ra_max": 360.0, "dec_min": 0.0, "dec_max": 90.0, "g_max": 15.0,
                              "source": "registered_existing_db", "created_utc": "t1", "n_stars": 5}])
    b = dict(base, coverage=[dict(a["coverage"][0], created_utc="t2")])
    assert catalog_fingerprints_equal(a, b)
    c = dict(base, coverage=[dict(a["coverage"][0], g_max=17.0)])
    assert not catalog_fingerprints_equal(a, c)
