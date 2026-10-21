"""Regression: sqlite3.Row iteration yields values, not column names."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hrd_analysis import _fetch_gaia_columns_by_source_id


def test_fetch_gaia_columns_maps_row_keys_not_values(tmp_path: Path) -> None:
    db = tmp_path / "vyvar_gaia_dr3.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE gaia_dr3 (source_id INTEGER PRIMARY KEY, teff_gspphot REAL, logg_gspphot REAL)"
    )
    sid = 4035720806645181440
    conn.execute(
        "INSERT INTO gaia_dr3 (source_id, teff_gspphot, logg_gspphot) VALUES (?, ?, ?)",
        (sid, 5500.0, 4.2),
    )
    conn.commit()
    conn.close()

    df = _fetch_gaia_columns_by_source_id(db, [str(sid)], ["teff_gspphot", "logg_gspphot"])
    assert not df.empty
    row = df.iloc[0]
    assert str(row["catalog_id"]) == str(sid)
    assert float(row["teff_gspphot"]) == 5500.0
    assert float(row["logg_gspphot"]) == 4.2
