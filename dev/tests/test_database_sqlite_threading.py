# -*- coding: ascii -*-
"""Thread-safe vyvar.sqlite3 access for cached Streamlit pipeline (bug #10)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from config import AppConfig
from database import VyvarDatabase
from pipeline import AstroPipeline


def test_open_sqlite_connection_allows_cross_thread_use(tmp_path) -> None:
    from database import open_sqlite_connection

    db = tmp_path / "vyvar.sqlite3"
    conn = open_sqlite_connection(db)

    def _read() -> int:
        row = conn.execute("SELECT 1 AS n;").fetchone()
        return int(row["n"])

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            vals = list(pool.map(lambda _: _read(), range(4)))
    finally:
        conn.close()
    assert vals == [1, 1, 1, 1]


def test_cached_pipeline_db_cross_thread_reads_and_writes(tmp_path, monkeypatch) -> None:
    """Mirror Streamlit: pipeline (DB) created in thread A, UI ops from B and C."""
    db_path = tmp_path / "vyvar.sqlite3"
    monkeypatch.setenv("VYVAR_DATA_DIR", str(tmp_path))
    install = tmp_path / "install"
    install.mkdir()
    cfg = AppConfig(project_root=install)
    cfg.database_path = db_path
    cfg.ensure_base_dirs()

    pipeline = AstroPipeline(config=cfg)
    pipeline.db.conn.execute(
        "INSERT INTO EQUIPMENTS (CAMERANAME, ALIAS, ACTIVE) VALUES (?, ?, ?);",
        ("CamA", "seed", "YES"),
    )
    pipeline.db.conn.commit()

    errors: list[str] = []

    def _read_equip() -> int:
        try:
            rows = pipeline.db.get_equipments()
            return len(rows)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"read: {type(exc).__name__}: {exc}")
            raise

    def _write_alias(tag: str) -> None:
        try:
            pipeline.db.conn.execute(
                "UPDATE EQUIPMENTS SET ALIAS = ? WHERE ID = 1;",
                (tag,),
            )
            pipeline.db.conn.commit()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"write: {type(exc).__name__}: {exc}")
            raise

    read_counts: list[int] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [
            pool.submit(_read_equip),
            pool.submit(_write_alias, "from-B"),
            pool.submit(_read_equip),
            pool.submit(_write_alias, "from-C"),
            pool.submit(_read_equip),
        ]
        for fut in as_completed(futs):
            result = fut.result()
            if isinstance(result, int):
                read_counts.append(result)

    assert not errors, errors
    assert len(read_counts) == 3
    assert all(c >= 1 for c in read_counts)
    alias = pipeline.db.conn.execute(
        "SELECT ALIAS FROM EQUIPMENTS WHERE ID = 1;"
    ).fetchone()["ALIAS"]
    assert alias in ("from-B", "from-C")
    pipeline.db.close()


def test_vyvar_database_conn_is_thread_safe_wrapper(tmp_path) -> None:
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    from database import ThreadSafeSQLiteConnection

    assert isinstance(db.conn, ThreadSafeSQLiteConnection)
    db.close()


def test_read_df_via_thread_safe_connection(tmp_path) -> None:
    """Database Explorer _read_df must not call con.cursor() (wrapper has no cursor())."""
    from ui_database_explorer import _read_df

    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    db.conn.execute(
        "INSERT INTO EQUIPMENTS (CAMERANAME, ALIAS, ACTIVE) VALUES (?, ?, ?);",
        ("CamA", "alias-a", "YES"),
    )
    db.conn.commit()
    df = _read_df(db.conn, "SELECT ID, CAMERANAME, ALIAS FROM EQUIPMENTS ORDER BY ID;")
    assert list(df.columns) == ["ID", "CAMERANAME", "ALIAS"]
    assert len(df) == 1
    assert df.iloc[0]["CAMERANAME"] == "CamA"
    df2 = _read_df(
        db.conn,
        "SELECT ALIAS FROM EQUIPMENTS WHERE ID = ?;",
        (1,),
    )
    assert df2.iloc[0]["ALIAS"] == "alias-a"
    db.close()


def test_equipments_focal_column_dropped_on_open(tmp_path) -> None:
    """Legacy EQUIPMENTS.FOCAL is removed; focal is TELESCOPE-only."""
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)
    db.conn.execute(
        "INSERT INTO EQUIPMENTS (CAMERANAME, ALIAS, ACTIVE) VALUES (?, ?, ?);",
        ("CamA", "alias-a", "YES"),
    )
    db.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN FOCAL REAL;")
    db.conn.execute("UPDATE EQUIPMENTS SET FOCAL = 999.0 WHERE ID = 1;")
    db.conn.commit()
    db.close()

    db2 = VyvarDatabase(db_path)
    cols = {r["name"] for r in db2.conn.execute("PRAGMA table_info('EQUIPMENTS');").fetchall()}
    assert "FOCAL" not in cols
    row = db2.conn.execute("SELECT CAMERANAME FROM EQUIPMENTS WHERE ID = 1;").fetchone()
    assert row is not None
    assert row["CAMERANAME"] == "CamA"
    db2.close()


def test_equipments_focal_migration_heals_orphan_old_table(tmp_path) -> None:
    """Leftover EQUIPMENTS_OLD from a failed rebuild must not block reopen."""
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)
    db.conn.execute(
        "INSERT INTO EQUIPMENTS (CAMERANAME, ALIAS, ACTIVE) VALUES (?, ?, ?);",
        ("CamOrphan", "alias-o", "YES"),
    )
    db.conn.execute("ALTER TABLE EQUIPMENTS ADD COLUMN FOCAL REAL;")
    db.conn.execute("UPDATE EQUIPMENTS SET FOCAL = 1480.0 WHERE ID = 1;")
    db.conn.execute(
        """
        CREATE TABLE EQUIPMENTS_OLD (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            CAMERANAME TEXT,
            FOCAL REAL
        );
        """
    )
    db.conn.commit()
    db.close()

    db2 = VyvarDatabase(db_path)
    cols = {r["name"] for r in db2.conn.execute("PRAGMA table_info('EQUIPMENTS');").fetchall()}
    tables = {
        r[0] for r in db2.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    }
    assert "FOCAL" not in cols
    assert "EQUIPMENTS_OLD" not in tables
    row = db2.conn.execute("SELECT CAMERANAME FROM EQUIPMENTS WHERE ID = 1;").fetchone()
    assert row is not None
    assert row["CAMERANAME"] == "CamOrphan"
    db2.close()


def test_telescope_active_migration_heals_orphan_old_table(tmp_path) -> None:
    """Leftover TELESCOPE_OLD from a failed ACTIVE rebuild must not block reopen."""
    db_path = tmp_path / "vyvar.sqlite3"
    conn = __import__("sqlite3").connect(db_path)
    conn.executescript(
        """
        CREATE TABLE TELESCOPE (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            TELESCOPENAME TEXT,
            ALIAS TEXT,
            DIAMETER REAL,
            FOCAL REAL,
            ACTIVE INTEGER DEFAULT 1,
            IS_DEFAULT INTEGER DEFAULT 0
        );
        INSERT INTO TELESCOPE (TELESCOPENAME, ALIAS, DIAMETER, FOCAL, ACTIVE)
        VALUES ('TelOrphan', 'to', 200.0, 1480.0, 1);
        CREATE TABLE TELESCOPE_OLD (
            ID INTEGER PRIMARY KEY,
            TELESCOPENAME TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    db = VyvarDatabase(db_path)
    active_type = next(
        str(r["type"]).upper()
        for r in db.conn.execute("PRAGMA table_info('TELESCOPE');")
        if r["name"] == "ACTIVE"
    )
    tables = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table';")}
    assert "TEXT" in active_type
    assert "TELESCOPE_OLD" not in tables
    row = db.conn.execute(
        "SELECT TELESCOPENAME, ACTIVE FROM TELESCOPE WHERE ID = 1;"
    ).fetchone()
    assert row is not None
    assert row["TELESCOPENAME"] == "TelOrphan"
    assert row["ACTIVE"] == "YES"
    db.close()


def test_photometry_light_curve_table_dropped_on_open(tmp_path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)
    db.conn.execute(
        """
        CREATE TABLE PHOTOMETRY_LIGHT_CURVE (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            DRAFT_ID INTEGER,
            JD REAL
        );
        """
    )
    db.conn.commit()
    db.close()

    VyvarDatabase(db_path).close()
    conn = __import__("sqlite3").connect(db_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")}
    conn.close()
    assert "PHOTOMETRY_LIGHT_CURVE" not in names


def test_active_stored_as_yes_no_text(tmp_path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)
    db.conn.execute(
        "INSERT INTO EQUIPMENTS (CAMERANAME, ALIAS, ACTIVE) VALUES (?, ?, ?);",
        ("CamA", "a", "1"),
    )
    db.conn.execute(
        "INSERT INTO TELESCOPE (TELESCOPENAME, ALIAS, DIAMETER, FOCAL, ACTIVE) VALUES (?, ?, ?, ?, ?);",
        ("Tel", "t", 200.0, 1000.0, 0),
    )
    db.conn.execute(
        "INSERT INTO LOCATION (PLACENAME, LATITUDE, LONGITUDE, ALTITUDE, ACTIVE) VALUES (?, ?, ?, ?, ?);",
        ("Site", 50.0, 14.0, 300.0, 1),
    )
    db.conn.commit()
    db.close()

    db2 = VyvarDatabase(db_path)
    eq = db2.conn.execute("SELECT ACTIVE FROM EQUIPMENTS WHERE ID = 1;").fetchone()["ACTIVE"]
    tel = db2.conn.execute("SELECT ACTIVE FROM TELESCOPE WHERE ID = 1;").fetchone()["ACTIVE"]
    loc = db2.conn.execute("SELECT ACTIVE FROM LOCATION WHERE ID = 1;").fetchone()["ACTIVE"]
    assert eq == "YES"
    assert tel == "NO"
    assert loc == "YES"
    assert db2.get_equipments(active_only=True)
    assert db2.get_telescopes(active_only=True) == []
    assert db2.get_telescopes(active_only=False)
    assert VyvarDatabase.normalize_active_text(1) == "YES"
    assert VyvarDatabase.normalize_active_text(0) == "NO"
    assert VyvarDatabase.normalize_active_text("NO") == "NO"
    db2.close()


def test_get_observer_locations_text_active(tmp_path) -> None:
    from database import get_observer_locations

    db_path = tmp_path / "vyvar.sqlite3"
    db = VyvarDatabase(db_path)
    db.conn.execute(
        "INSERT INTO LOCATION (PLACENAME, LATITUDE, LONGITUDE, ALTITUDE, ACTIVE) VALUES (?, ?, ?, ?, ?);",
        ("ActiveSite", 50.0, 14.0, 300.0, "YES"),
    )
    db.conn.execute(
        "INSERT INTO LOCATION (PLACENAME, LATITUDE, LONGITUDE, ALTITUDE, ACTIVE) VALUES (?, ?, ?, ?, ?);",
        ("InactiveSite", 51.0, 15.0, 400.0, "NO"),
    )
    db.conn.commit()
    db.close()

    all_rows = get_observer_locations(db_path)
    assert len(all_rows) == 2
    assert all_rows[0]["active"] == 1
    assert all_rows[1]["active"] == 0

    active_rows = get_observer_locations(db_path, active_only=True)
    assert len(active_rows) == 1
    assert active_rows[0]["name"] == "ActiveSite"
    assert active_rows[0]["active"] == 1
