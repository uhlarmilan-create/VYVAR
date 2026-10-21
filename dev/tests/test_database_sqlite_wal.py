# -*- coding: ascii -*-
"""SQLite connection policy for vyvar.sqlite3 (Streamlit concurrent startup)."""
from __future__ import annotations

from database import VyvarDatabase, open_sqlite_connection


def test_open_sqlite_connection_uses_wal(tmp_path) -> None:
    db = tmp_path / "vyvar.sqlite3"
    conn = open_sqlite_connection(db)
    try:
        mode = str(conn.execute("PRAGMA journal_mode;").fetchone()[0]).lower()
        busy = int(conn.execute("PRAGMA busy_timeout;").fetchone()[0])
    finally:
        conn.close()
    assert mode == "wal"
    assert busy >= 30_000


def test_sequential_vyvar_database_opens(tmp_path) -> None:
    db = tmp_path / "vyvar.sqlite3"
    first = VyvarDatabase(db)
    first.close()
    second = VyvarDatabase(db)
    second.close()
