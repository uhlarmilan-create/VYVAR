"""Guards for DB-SEED-SPLIT: fresh DB empty; reference seed is harness-only."""
from __future__ import annotations

from pathlib import Path

from database import VyvarDatabase
from tools.reference_seed import (
    REFERENCE_EQUIPMENTS,
    REFERENCE_LOCATION,
    REFERENCE_SCANNING,
    REFERENCE_TELESCOPE,
    seed_reference_observatory,
)


def test_initialize_database_leaves_reference_tables_empty(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "fresh.sqlite3")
    try:
        for table in ("EQUIPMENTS", "TELESCOPE", "LOCATION", "SCANNING"):
            n = int(db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            assert n == 0, f"{table} must be empty on a fresh DB; got {n}"
    finally:
        db.conn.close()


def test_seed_reference_observatory_pins_exact_author_rows(tmp_path: Path) -> None:
    db = VyvarDatabase(tmp_path / "seeded.sqlite3")
    try:
        seed_reference_observatory(db)

        eq = db.conn.execute(
            "SELECT ID, CAMERANAME, ALIAS, SENSORTYPE, SENSORSIZE, PIXELSIZE "
            "FROM EQUIPMENTS ORDER BY ID"
        ).fetchall()
        assert [tuple(r) for r in eq] == list(REFERENCE_EQUIPMENTS)

        tel = db.conn.execute(
            "SELECT ID, TELESCOPENAME, ALIAS, DIAMETER, FOCAL "
            "FROM TELESCOPE ORDER BY ID"
        ).fetchall()
        assert [tuple(r) for r in tel] == list(REFERENCE_TELESCOPE)

        loc = db.conn.execute(
            "SELECT ID, PLACENAME, LATITUDE, LONGITUDE, ALTITUDE "
            "FROM LOCATION ORDER BY ID"
        ).fetchall()
        assert [tuple(r) for r in loc] == list(REFERENCE_LOCATION)

        scan = db.conn.execute(
            "SELECT ID, EXPTIME, FILTERS, BINNING, SENSORTEMP "
            "FROM SCANNING ORDER BY ID"
        ).fetchall()
        assert [tuple(r) for r in scan] == list(REFERENCE_SCANNING)

        # Idempotent: second call must not duplicate or alter rows.
        seed_reference_observatory(db)
        n_eq = int(db.conn.execute("SELECT COUNT(*) FROM EQUIPMENTS").fetchone()[0])
        assert n_eq == len(REFERENCE_EQUIPMENTS)
    finally:
        db.conn.close()
