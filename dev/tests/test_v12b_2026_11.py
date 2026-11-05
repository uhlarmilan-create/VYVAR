# -*- coding: ascii -*-
"""VYVAR v12b (Milan's Brno 4h Cyg N1 run, 2026-11-05).

TOPUP-UNION-2026-11: the field depth is the union of all Gaia coverage records; a field shifted
against an earlier top-up fetches only the uncovered slivers (the r field, 4" off the g field, had
fetched the whole field again: 20 935 rows, 8 new).

TRAIL-PATH-2026-11: the frame-QC trail test reads the calibrated light of every setup (r frames fell
back to the RAW frame), and never another image in place of a missing one."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

import gaia_coverage as gc

# ------------------------------------------------------------------ TOPUP-UNION-2026-11
_BOX_RE = re.compile(r"ra >= ([\d.]+) AND ra <= ([\d.]+) AND dec >= (-?[\d.]+) AND dec <= (-?[\d.]+)")


class _Sky:
    """Archive stand-in answering every box of the ADQL WHERE (OR of boxes) from a synthetic sky."""

    def __init__(self, n=300_000, seed=4):
        rng = np.random.default_rng(seed)
        self.df = pd.DataFrame({
            "SOURCE_ID": np.arange(50_000_000, 50_000_000 + n),
            "ra": rng.uniform(318.0, 323.0, n), "dec": rng.uniform(50.0, 54.0, n),
            "g_mag": rng.uniform(8, 21, n), "bp_rp": 1.0, "var_flag": "NOT_AVAILABLE", "non_single_star": 0,
        })
        self.queries: list[str] = []

    def rows(self, adql: str) -> pd.DataFrame:
        d = self.df
        m = np.zeros(len(d), dtype=bool)
        for a, b, c, e in _BOX_RE.findall(adql):
            m |= (d["ra"] >= float(a)) & (d["ra"] <= float(b)) & (d["dec"] >= float(c)) & (d["dec"] <= float(e))
        g0 = float(re.search(r"phot_g_mean_mag > (-?[\d.]+)", adql).group(1))
        g1 = float(re.search(r"phot_g_mean_mag <= (-?[\d.]+)", adql).group(1))
        return d.loc[m & (d["g_mag"] > g0) & (d["g_mag"] <= g1)].reset_index(drop=True)

    def __call__(self, adql: str) -> pd.DataFrame:
        if "COUNT(*)" in adql:
            return pd.DataFrame({"n": [len(self.rows(adql))]})
        self.queries.append(adql)
        return self.rows(adql)


def _db(path: Path, g_max: float = 17.5) -> Path:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE gaia_dr3 (source_id INTEGER PRIMARY KEY, ra REAL, dec REAL, g_mag REAL, bp_mag REAL,"
        " rp_mag REAL, bp_rp REAL, g_flux_error_rel REAL, parallax REAL, parallax_error REAL,"
        " parallax_over_error REAL, teff_gspphot REAL, logg_gspphot REAL, mh_gspphot REAL,"
        " distance_gspphot REAL, var_flag TEXT, non_single_star INTEGER)")
    con.executemany("INSERT INTO gaia_dr3 (source_id, ra, dec, g_mag) VALUES (?,?,?,?)",
                    [(1, 0.0, -90.0, g_max), (2, 359.0, 90.0, 9.0)])
    con.commit()
    con.close()
    return path


G_FIELD = SkyCoord(320.36 * u.deg, 51.817 * u.deg)       # Brno 4h Cyg N1, g channel
R_FIELD = SkyCoord(320.3594 * u.deg, 51.8163 * u.deg)    # r channel, ~3" off
RADIUS = 0.476


def test_a_shifted_field_already_deep_enough_is_covered_without_a_query(tmp_path: Path) -> None:
    db, sky = _db(tmp_path / "g.db"), _Sky()
    assert gc.ensure_field_depth(db, center=G_FIELD, radius_deg=RADIUS, g_needed=19.5, fetch=sky)["status"] == "ok"
    n = len(sky.queries)
    res = gc.ensure_field_depth(db, center=R_FIELD, radius_deg=RADIUS, g_needed=19.4, fetch=sky)
    # the slivers outside the g box are at 17.5: those are fetched, the rest is not
    assert res["status"] == "ok" and len(sky.queries) == n + 1
    q = sky.queries[-1]
    assert "phot_g_mean_mag > 17.5000" in q
    area = sum(p["area_deg2"] for p in res["parts"])
    assert area < 0.01, f"only the slivers ({area:.4f} deg2), not the ~0.9 deg2 field"
    assert res["depth_after"] == pytest.approx(19.4)
    # the same r field again: covered, no query
    again = gc.ensure_field_depth(db, center=R_FIELD, radius_deg=RADIUS, g_needed=19.4, fetch=sky)
    assert again["status"] == "covered" and len(sky.queries) == n + 1


def test_union_depth_and_partial_fetch_are_complete(tmp_path: Path) -> None:
    """After the sliver top-up the DB holds exactly the stars of the r field to G 19.4."""
    db, sky = _db(tmp_path / "g.db"), _Sky()
    gc.ensure_field_depth(db, center=G_FIELD, radius_deg=RADIUS, g_needed=19.5, fetch=sky)
    gc.ensure_field_depth(db, center=R_FIELD, radius_deg=RADIUS, g_needed=19.4, fetch=sky)
    b = gc.field_box(R_FIELD, RADIUS)
    con = sqlite3.connect(db)
    got = {r[0] for r in con.execute(
        "SELECT source_id FROM gaia_dr3 WHERE ra BETWEEN ? AND ? AND dec BETWEEN ? AND ? AND g_mag > 17.5 AND g_mag <= 19.4",
        (b[0], b[1], b[2], b[3]))}
    con.close()
    d = sky.df
    want = set(d.loc[(d.ra >= b[0]) & (d.ra <= b[1]) & (d.dec >= b[2]) & (d.dec <= b[3])
                     & (d.g_mag > 17.5) & (d.g_mag <= 19.4), "SOURCE_ID"])
    assert got == want and len(want) > 1000


def test_deeper_need_fetches_each_part_from_its_own_depth(tmp_path: Path) -> None:
    db, sky = _db(tmp_path / "g.db"), _Sky()
    gc.ensure_field_depth(db, center=G_FIELD, radius_deg=RADIUS, g_needed=19.5, fetch=sky)
    n = len(sky.queries)
    res = gc.ensure_field_depth(db, center=R_FIELD, radius_deg=RADIUS, g_needed=20.0, fetch=sky)
    froms = sorted(re.search(r"phot_g_mean_mag > ([\d.]+)", q).group(1) for q in sky.queries[n:])
    assert froms == ["17.5000", "19.5000"], "slivers from 17.5, the rest from 19.5 - nothing twice"
    assert res["status"] == "ok" and res["depth_after"] == pytest.approx(20.0)


def test_covered_depth_is_the_union_of_records(tmp_path: Path) -> None:
    db = _db(tmp_path / "g.db")
    gc.register_existing_db(db)  # the whole-DB record (G 17.5)
    gc.record_coverage(db, box=(10.0, 10.6, 0.0, 1.0), g_max=19.0, source="gaia_tap_topup")
    gc.record_coverage(db, box=(10.4, 11.0, 0.0, 1.0), g_max=20.0, source="gaia_tap_topup")
    assert gc.covered_depth_g(db, (10.0, 11.0, 0.0, 1.0)) == pytest.approx(19.0)
    assert gc.covered_depth_g(db, (10.45, 10.55, 0.2, 0.8)) == pytest.approx(20.0)
    assert gc.covered_depth_g(db, (10.0, 11.2, 0.0, 1.0)) == pytest.approx(17.5)


def test_failed_part_reports_and_keeps_the_depth(tmp_path: Path) -> None:
    db, sky = _db(tmp_path / "g.db"), _Sky()
    gc.ensure_field_depth(db, center=G_FIELD, radius_deg=RADIUS, g_needed=19.5, fetch=sky)

    def _down(_adql: str) -> pd.DataFrame:
        raise OSError("Name or service not known")

    res = gc.ensure_field_depth(db, center=R_FIELD, radius_deg=RADIUS, g_needed=19.4, fetch=_down)
    assert res["status"] == "offline" and res["depth_after"] == pytest.approx(17.5)


# ------------------------------------------------------------------ TRAIL-PATH-2026-11
def _draft(tmp_path: Path) -> Path:
    ap = tmp_path / "draft"
    for setup in ("g_120_4", "r_100_4"):
        for sub in ("Raw", "calibrated"):
            d = ap / sub / "lights" / setup
            d.mkdir(parents=True)
            (d / f"s4_{setup[0]}_0001.fits").write_bytes(b"x")
    return ap


def test_trail_image_of_every_setup_is_its_calibrated_light(tmp_path: Path) -> None:
    import night_run

    ap = _draft(tmp_path)
    for setup in ("g_120_4", "r_100_4"):
        raw = ap / "Raw" / "lights" / setup / f"s4_{setup[0]}_0001.fits"
        got = night_run.frame_qc_image_path(ap, str(raw), draft_id=None, db=None)
        assert Path(got) == (ap / "calibrated" / "lights" / setup / raw.name).resolve() or \
            Path(got) == ap / "calibrated" / "lights" / setup / raw.name


def test_unmapped_frame_is_not_trail_tested_and_never_reads_raw(tmp_path: Path) -> None:
    import frame_qc as fq
    import night_run

    ap = _draft(tmp_path)
    raw = ap / "Raw" / "lights" / "r_100_4" / "s4_r_0002.fits"
    raw.write_bytes(b"x")  # no calibrated twin
    assert night_run.frame_qc_image_path(ap, str(raw), draft_id=None, db=None) == ""
    check = fq.fits_trail_check(lambda p: "")
    assert check(str(raw), 3.0, None, None) is None
