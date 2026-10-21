"""Tests for MASTER_SOURCES schema retirement (MS-SOURCES-RETIRE C3)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from database import VyvarDatabase


def test_fresh_db_has_no_master_sources(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "fresh.sqlite3")
    try:
        tables = {
            str(r[0])
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        assert "MASTER_SOURCES" not in tables
    finally:
        db.close()


def test_healthy_db_drops_master_sources(tmp_path: Path) -> None:
    db_path = tmp_path / "with_ms.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE MASTER_SOURCES (
            ID INTEGER PRIMARY KEY,
            DRAFT_ID INTEGER,
            SOURCE_ID_GAIA TEXT
        );
        """
    )
    conn.execute("CREATE INDEX IDX_MASTER_SOURCES_DRAFT ON MASTER_SOURCES (DRAFT_ID);")
    conn.commit()
    conn.close()

    db = VyvarDatabase(db_path)
    try:
        tables = {
            str(r[0])
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        assert "MASTER_SOURCES" not in tables
    finally:
        db.close()


def test_production_db_open_survives_if_present() -> None:
    db_path = Path(__file__).resolve().parents[2] / "vyvar.sqlite3"
    if not db_path.is_file():
        return
    db = VyvarDatabase(db_path)
    try:
        tables = {
            str(r[0])
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        assert "EQUIPMENTS" in tables
    finally:
        db.close()


def test_no_master_sources_outside_drop_migration() -> None:
    root = Path(__file__).resolve().parents[2] / "src_py"
    allowed = {
        "database.py",
        "pipeline.py",
        "psf_photometry.py",
        "masterstars_enrichment.py",
        "masterstar_build.py",  # E6b: generate_masterstar_and_catalog moved here
    }
    hits: list[str] = []
    for path in root.rglob("*.py"):
        if path.name not in allowed and "MASTER_SOURCES" in path.read_text(encoding="utf-8", errors="replace"):
            hits.append(str(path.relative_to(root)))
    assert hits == []
