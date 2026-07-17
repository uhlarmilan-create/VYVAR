"""G1-F005: _first_db_optics_ids logs WARNING on DB failure but still returns None."""

import sqlite3
from unittest.mock import MagicMock

from optics_selection import _first_db_optics_ids


def test_first_db_optics_ids_logs_warning_and_returns_none_on_db_failure(caplog):
    db = MagicMock()
    db.conn.execute.side_effect = sqlite3.OperationalError("no such table: EQUIPMENTS")

    with caplog.at_level("WARNING", logger="optics_selection"):
        eq_id, tel_id = _first_db_optics_ids(db)

    assert eq_id is None
    assert tel_id is None
    assert any(
        "EQUIPMENTS optics lookup failed" in rec.message
        and "fell through to None" in rec.message
        for rec in caplog.records
    )
