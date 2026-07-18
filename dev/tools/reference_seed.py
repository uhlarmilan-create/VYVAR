"""Author's REFERENCE observatory fixture (harness/tests only) -- NOT product content.

Milan's product decision (DB-SEED-SPLIT, 2026-07-18): a new user's fresh database
MUST be empty; they create their own Location / Telescope / Equipment in the app.
The author's observatory rows are a reference FIXTURE for the anchor/test machinery,
kept out of the runtime ``src_py`` package. ``VyvarDatabase.initialize_database()``
no longer seeds; the ``--full`` anchor harness and the pytest fixtures that need the
anchor context call ``seed_reference_observatory(db)`` explicitly.

The rows below are EXACTLY the set ``initialize_database()`` seeded before the split
(byte-for-byte identical values). Seeding uses ``INSERT OR IGNORE``, so calling this
on an already-populated database (e.g. the author's production DB, which is never
re-initialised) is a no-op -- the anchor gate therefore stays byte-identical and, by
running the seed, also proves the split preserved the anchor context.
"""
from __future__ import annotations

# (ID, CAMERANAME, ALIAS, SENSORTYPE, SENSORSIZE, PIXELSIZE)
REFERENCE_EQUIPMENTS: tuple[tuple, ...] = (
    (1, "QHY294MM", "Camera1", "IMX492", "4164*2796", 4.63),
    (4, "C5A-150M", "C5A-150M", "CMOS", "4096*4096", 3.76),
)
# (ID, TELESCOPENAME, ALIAS, DIAMETER, FOCAL)
REFERENCE_TELESCOPE: tuple[tuple, ...] = (
    (1, "Carl-Zeiss", "Teleobjektiv1", 72.0, 200.0),
    (6, "AZ800", "AZ800", 800.0, 5480.0),
)
# (ID, PLACENAME, LATITUDE, LONGITUDE, ALTITUDE)
REFERENCE_LOCATION: tuple[tuple, ...] = (
    (1, "Dablice", 50.073658, 14.418540, 355.5),
)
# (ID, EXPTIME, FILTERS, BINNING, SENSORTEMP)
REFERENCE_SCANNING: tuple[tuple, ...] = (
    (1, 120.0, "Clear", 11, -10.0),
)


def seed_reference_observatory(db) -> None:
    """Insert the author's reference observatory rows into ``db`` (idempotent).

    ``db`` is a ``VyvarDatabase`` (only ``db.conn`` is used). ``INSERT OR IGNORE``
    makes this safe on an already-populated DB: existing rows are untouched, so an
    anchor run over the author's production DB stays byte-identical.
    """
    conn = db.conn
    conn.executemany(
        "INSERT OR IGNORE INTO EQUIPMENTS "
        "(ID, CAMERANAME, ALIAS, SENSORTYPE, SENSORSIZE, PIXELSIZE) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        REFERENCE_EQUIPMENTS,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO TELESCOPE (ID, TELESCOPENAME, ALIAS, DIAMETER, FOCAL) "
        "VALUES (?, ?, ?, ?, ?)",
        REFERENCE_TELESCOPE,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO LOCATION (ID, PLACENAME, LATITUDE, LONGITUDE, ALTITUDE) "
        "VALUES (?, ?, ?, ?, ?)",
        REFERENCE_LOCATION,
    )
    conn.executemany(
        "INSERT OR IGNORE INTO SCANNING (ID, EXPTIME, FILTERS, BINNING, SENSORTEMP) "
        "VALUES (?, ?, ?, ?, ?)",
        REFERENCE_SCANNING,
    )
    conn.commit()
