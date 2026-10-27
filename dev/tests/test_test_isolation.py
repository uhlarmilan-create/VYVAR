"""TEST-ISOLATION-2026-10: tests run on their own data root and never open the user's databases."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from config import AppConfig

CODE_TREE = Path(__file__).resolve().parents[2]


def test_appconfig_in_a_test_uses_the_isolated_data_root(isolated_data_root: Path) -> None:
    cfg = AppConfig()
    assert Path(os.environ["VYVAR_DATA_DIR"]) == isolated_data_root
    assert cfg.data_root == isolated_data_root.resolve()
    for p in (cfg.database_path, cfg.archive_root, cfg.calibration_library_root):
        assert Path(p).resolve().is_relative_to(isolated_data_root.resolve()), p


def test_opening_a_database_in_the_code_tree_fails() -> None:
    with pytest.raises(AssertionError, match="TEST-ISOLATION-2026-10"):
        sqlite3.connect(CODE_TREE / "vyvar.sqlite3")
    with pytest.raises(AssertionError, match="TEST-ISOLATION-2026-10"):
        sqlite3.connect(str(CODE_TREE / "catalogs" / "vyvar_gaia_dr3.db"))


def test_tmp_and_memory_databases_are_allowed(tmp_path: Path) -> None:
    sqlite3.connect(":memory:").close()
    sqlite3.connect(tmp_path / "x.sqlite3").close()
