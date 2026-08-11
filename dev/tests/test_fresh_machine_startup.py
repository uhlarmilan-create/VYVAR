"""Guard: a fresh-machine install starts cleanly (INSTALL-ARC STEP 3).

Simulates the first run on a stranger's machine after the installer wrote a local
paths block: a project root with NO catalogs, a config.json whose path keys are
blanked (so project-root defaults resolve), and no pre-existing database. Asserts
that the startup-path functions initialise without crashing and without leaking the
author's absolute ``C:\\ASTRO`` paths:

  * AppConfig resolves archive/calibration/database roots under the fresh project root;
  * blanked catalog keys stay empty (so the app can detect LIMITED MODE, not crash);
  * VyvarDatabase self-initialises the schema on a brand-new sqlite file;
  * location hydration for a missing id (the shipped ``observer_location_id``) returns
    None gracefully rather than raising.
"""
from __future__ import annotations

from pathlib import Path

from config import AppConfig
from database import (
    VyvarDatabase,
    get_observer_location_by_id,
    get_observer_locations,
)

_FRESH_CONFIG = """// fresh-machine config.json (paths blanked by the installer)
{
  "observer_location_id": 2,
  "archive_root": "",
  "calibration_library_root": "",
  "database_path": "",
  "gaia_db_path": "",
  "vsx_local_db_path": "",
  "blind_index_fine_path": "",
  "blind_index_wide_path": ""
}
"""

_REQUIRED_TABLES = {"EQUIPMENTS", "TELESCOPE", "LOCATION"}


def _fresh_root(tmp_path: Path) -> Path:
    (tmp_path / "config.json").write_text(_FRESH_CONFIG, encoding="utf-8")
    return tmp_path


def test_appconfig_resolves_local_roots_when_paths_blanked(tmp_path: Path) -> None:
    root = _fresh_root(tmp_path)
    cfg = AppConfig(project_root=root)

    assert cfg.archive_root == root / "Archive"
    assert cfg.calibration_library_root == root / "CalibrationLibrary"
    assert cfg.database_path == root / "vyvar.sqlite3"

    # No author absolute paths must survive anywhere in the resolved config.
    for value in (
        str(cfg.archive_root),
        str(cfg.calibration_library_root),
        str(cfg.database_path),
        str(cfg.blind_index_fine_path),
        str(cfg.blind_index_wide_path),
    ):
        assert "ASTRO" not in value.upper() or str(root).upper().find("ASTRO") >= 0


def test_blanked_catalog_keys_stay_empty_for_limited_mode(tmp_path: Path) -> None:
    cfg = AppConfig(project_root=_fresh_root(tmp_path))
    # Empty gaia/vsx paths signal "no catalogs" (LIMITED MODE) rather than crashing.
    assert cfg.gaia_db_path == ""
    assert cfg.vsx_local_db_path == ""
    # Blind indices fall back to project-root-relative defaults (not author paths).
    assert cfg.blind_index_fine_path.endswith("gaia_triangles_fine.pkl")
    assert "ASTRO" not in cfg.blind_index_fine_path.upper() or "ASTRO" in str(tmp_path).upper()


def test_database_self_initialises_on_fresh_file(tmp_path: Path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    assert not db_path.exists()

    db = VyvarDatabase(db_path)
    try:
        assert db_path.exists(), "VyvarDatabase must create the sqlite file on construction"
        rows = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in rows}
        missing = _REQUIRED_TABLES - names
        assert not missing, f"fresh DB missing reference tables: {sorted(missing)}"
        # Product contract (DB-SEED-SPLIT): schema only - no author observatory rows.
        for table in ("EQUIPMENTS", "TELESCOPE", "LOCATION"):
            n = int(db.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            assert n == 0, f"fresh DB {table} must be empty; got {n}"
    finally:
        db.conn.close()


def test_location_hydration_is_graceful_for_missing_and_present_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "vyvar.sqlite3"
    VyvarDatabase(db_path).conn.close()  # self-init then release the handle

    # Never raises; fresh DB has zero locations.
    locations = get_observer_locations(db_path)
    assert locations == []

    # Shipped config may still point at observer_location_id=2; hydrate must return None.
    assert get_observer_location_by_id(db_path, 2) is None
    assert get_observer_location_by_id(db_path, 10_000_000) is None
