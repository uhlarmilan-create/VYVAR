"""CATALOG-DB-2026-10: one objects DB (VSX + exoplanets), read-only catalog readers, table fingerprints."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

import catalog_db as cdb
from catalog_provenance import fingerprint_objects_table

_ROOT = Path(__file__).resolve().parents[2]


def _load_builder():
    spec = importlib.util.spec_from_file_location("objects_make", _ROOT / "catalogs" / "objects_make.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["objects_make"] = mod
    spec.loader.exec_module(mod)
    return mod


def _old_vsx(path: Path) -> None:
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE vsx_data (oid INTEGER PRIMARY KEY, name TEXT, ra_deg REAL, dec_deg REAL, "
              "var_type TEXT, period REAL, mag_max REAL, mag_min REAL)")
    c.executemany("INSERT INTO vsx_data VALUES (?,?,?,?,?,?,?,?)", [
        (1, "V1", 359.9, 0.5, "EA", 1.2, 10.0, 10.5),
        (2, "V2", 0.1, 0.6, "RRAB", 0.5, 12.0, 13.0),
        (3, "V3", 180.0, 45.0, "M", None, 8.0, 12.0),
    ])
    c.commit()
    c.close()


def _old_exo(path: Path) -> None:
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE exoplanet_data (obj_id TEXT PRIMARY KEY, name TEXT, host_name TEXT, ra_deg REAL, "
              "dec_deg REAL, cat_source TEXT, disposition TEXT, period REAL, mag REAL, mag_band TEXT)")
    c.execute("INSERT INTO exoplanet_data VALUES ('TOI-1.01','TOI-1.01','TIC 1',0.05,0.55,'TOI','PC',3.1,9.0,'TESS')")
    c.commit()
    c.close()


@pytest.fixture()
def objects_db(tmp_path: Path) -> Path:
    vsx, exo, out = tmp_path / "vsx.db", tmp_path / "exo.db", tmp_path / "catalogs" / "vyvar_objects.db"
    _old_vsx(vsx)
    _old_exo(exo)
    mod = _load_builder()
    assert mod.main(["--db", str(out), "--import-vsx", str(vsx), "--import-exoplanets", str(exo)]) == 0
    return out


def test_import_builds_both_tables_and_meta(objects_db: Path) -> None:
    assert cdb.validate_vsx_schema(objects_db) == (True, "ok")
    assert cdb.validate_exoplanet_schema(objects_db) == (True, "ok")
    assert cdb.count_vsx_rows(objects_db) == 3
    meta = cdb.read_catalog_meta(objects_db)
    assert set(meta) == {"vsx_data", "exoplanet_data"}
    assert meta["vsx_data"]["row_count"] == 3 and meta["exoplanet_data"]["row_count"] == 1
    assert "imported from vsx.db" in meta["vsx_data"]["source"]


def test_import_keeps_content(objects_db: Path, tmp_path: Path) -> None:
    src = sqlite3.connect(tmp_path / "vsx.db")
    try:
        assert cdb.table_content_sha256(src, "vsx_data") == cdb.read_catalog_meta(objects_db)["vsx_data"]["content_sha256"]
    finally:
        src.close()


def test_box_queries_wrap_ra(objects_db: Path) -> None:
    rows = cdb.query_local_vsx(objects_db, ra_min=-0.5, ra_max=0.5, dec_min=0.0, dec_max=1.0)
    assert sorted(r["oid"] for r in rows) == [1, 2]
    exo = cdb.query_local_exoplanet(objects_db, ra_min=-0.5, ra_max=0.5, dec_min=0.0, dec_max=1.0)
    assert [r["obj_id"] for r in exo] == ["TOI-1.01"]
    assert cdb.query_local_vsx(objects_db, ra_min=-0.5, ra_max=0.5, dec_min=0.0, dec_max=1.0, max_rows=1)[0]["oid"] in (1, 2)


def test_readers_do_not_write(objects_db: Path) -> None:
    con = cdb.connect_ro(objects_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("CREATE INDEX idx_x ON vsx_data (name)")
    finally:
        con.close()


def test_require_messages_name_objects_db(tmp_path: Path) -> None:
    with pytest.raises(cdb.VSXCatalogError, match="objects_db_path"):
        cdb.require_vsx_db("")
    with pytest.raises(cdb.ExoplanetCatalogError, match="not found"):
        cdb.require_exoplanet_db(tmp_path / "nope.db")


def test_table_fingerprint_is_per_table(objects_db: Path) -> None:
    vsx0 = fingerprint_objects_table(objects_db, "vsx_data")
    exo0 = fingerprint_objects_table(objects_db, "exoplanet_data")
    c = sqlite3.connect(objects_db)
    c.execute("INSERT INTO exoplanet_data (obj_id, ra_deg, dec_deg, cat_source) VALUES ('X b', 10, 10, 'CONFIRMED')")
    c.commit()
    cdb.write_catalog_meta(c, "exoplanet_data", source="test", params={}, builder="test")
    c.close()
    assert fingerprint_objects_table(objects_db, "vsx_data")["fingerprint_sha256"] == vsx0["fingerprint_sha256"]
    assert fingerprint_objects_table(objects_db, "exoplanet_data")["fingerprint_sha256"] != exo0["fingerprint_sha256"]


def test_fingerprint_without_meta_matches_builder_definition(tmp_path: Path) -> None:
    vsx = tmp_path / "vsx.db"
    _old_vsx(vsx)
    fp = fingerprint_objects_table(vsx, "vsx_data")
    c = sqlite3.connect(vsx)
    try:
        assert fp["fingerprint_sha256"] == cdb.table_content_sha256(c, "vsx_data")
    finally:
        c.close()
