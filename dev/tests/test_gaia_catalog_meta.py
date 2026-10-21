"""CATALOG-DB-2026-10 step 5: Gaia DB in catalogs/, catalog_meta as its identity (no COUNT(*) per run)."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

import catalog_db as cdb
import catalog_provenance as cp
import gaia_coverage as gc
import settings_model as sm

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "catalogs" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _db(path: Path, n: int = 50) -> Path:
    con = sqlite3.connect(path)
    gc.init_gaia_tables(con)
    con.executemany("INSERT INTO gaia_dr3 (source_id, ra, dec, g_mag, bp_rp) VALUES (?,?,?,?,?)",
                    [(i + 1, float(i % 360), 10.0, 12.0, 0.8) for i in range(n)])
    con.commit()
    con.close()
    return path


def test_maintain_script_keeps_the_same_index_list() -> None:
    gm = _load("gaia_maintain")
    assert tuple(sorted(gm.NEEDED_INDEXES)) == tuple(sorted(cdb.GAIA_INDEXES))
    assert not set(gm.UNUSED_INDEXES) & set(cdb.GAIA_INDEXES)
    assert gm.CATALOG_META_SCHEMA.split() == cdb.CATALOG_META_SCHEMA.split()


def test_builder_creates_only_the_needed_indexes(tmp_path: Path) -> None:
    b = _load("build_gaia_catalog")
    db = _db(tmp_path / "g.db")
    con = sqlite3.connect(db)
    b.create_indexes(con)
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='gaia_dr3' AND sql IS NOT NULL")}
    con.close()
    assert names == set(cdb.GAIA_INDEXES)
    assert b.BUILDER == "catalogs/build_gaia_catalog.py"


def test_fingerprint_reads_row_count_from_catalog_meta(tmp_path: Path, monkeypatch) -> None:
    db = _db(tmp_path / "g.db", n=50)
    fp = cp.fingerprint_gaia_db(db)
    assert fp["row_count"] == 50 and fp["row_count_source"] == "count"  # never registered
    con = sqlite3.connect(db)
    rec = cdb.write_catalog_meta(con, "gaia_dr3", source="test", params={"g_max": 12.0}, builder="t",
                                 built_utc="2026-06-14T21:22:54Z")
    con.close()
    assert rec["content_sha256"] is None and rec["row_count"] == 50
    monkeypatch.setattr(cp, "_sqlite_row_count", lambda *_a, **_k: pytest.fail("COUNT(*) on a registered DB"))
    fp = cp.fingerprint_gaia_db(db)
    assert fp["row_count"] == 50 and fp["row_count_source"] == "catalog_meta"
    assert fp["built_utc"] == "2026-06-14T21:22:54Z" and fp["build_params"] == {"g_max": 12.0}


def test_topup_keeps_catalog_meta_row_count(tmp_path: Path) -> None:
    db = _db(tmp_path / "g.db", n=50)
    con = sqlite3.connect(db)
    cdb.write_catalog_meta(con, "gaia_dr3", source="test", params={}, builder="t")
    con.close()
    gc.record_coverage(db, box=(0.0, 360.0, -90.0, 90.0), g_max=12.0, source="build_gaia_catalog")
    new = pd.DataFrame({"SOURCE_ID": [1001, 1002, 1], "RA": [10.0, 11.0, 0.0], "DEC": [10.0, 10.0, 10.0],
                        "G_MAG": [13.0, 13.5, 12.0]})
    res = gc.top_up_field(db, box=(9.0, 12.0, 9.0, 11.0), g_from=12.0, g_to=14.0, fetch=lambda _q: new)
    assert res["status"] == "ok" and res["n_new"] == 2
    assert cdb.read_catalog_meta(db)["gaia_dr3"]["row_count"] == 52
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM gaia_dr3").fetchone()[0] == 52
    con.close()


def test_settings_card_reads_catalog_meta_and_flags_indexes(tmp_path: Path) -> None:
    db = _db(tmp_path / "g.db", n=50)
    st = sm.gaia_status(db)
    assert st.state == "ready" and "Register it" in st.note and "Missing index" in st.note
    con = sqlite3.connect(db)
    for sql in cdb.GAIA_INDEXES.values():
        con.execute(sql)
    con.execute("CREATE INDEX idx_teff ON gaia_dr3 (teff_gspphot)")
    cdb.write_catalog_meta(con, "gaia_dr3", source="test", params={}, builder="t", built_utc="2026-06-14T21:22:54Z")
    con.close()
    st = sm.gaia_status(db)
    facts = dict(st.facts)
    assert facts["Built"] == "2026-06-14" and facts["Stars"] == "0.0 million"
    assert "idx_teff" in st.note and "Drop unused" in st.note and "Register" not in st.note


def test_builders_default_to_the_catalogs_folder() -> None:
    import config

    # empty gaia_db_path stays "no catalog" (LIMITED MODE); the builder writes to catalogs/
    assert config.AppConfig.__dataclass_fields__["gaia_db_path"].default == ""
    assert cdb.GAIA_DB_DEFAULT.startswith("catalogs/") and cdb.OBJECTS_DB_DEFAULT.startswith("catalogs/")
