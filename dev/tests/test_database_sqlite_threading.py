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
