"""Shared pytest configuration."""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: draft_000366 comp_qa integration (minutes per case)",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:
    """Vendor tree is not a pytest package; smoke test lives in dev/tests."""
    _ = config
    parts = set(Path(collection_path).parts)
    return "xval_pythonphot" in parts and "vendor" in parts


# TEST-ISOLATION-2026-10: a test never sees the user's data. In a git checkout the data root is
# the code tree, so AppConfig() read the user's config.json, opened the user's vyvar.sqlite3 and
# Archive (and on Linux created files named after the Windows paths in the code tree). Every test
# runs with VYVAR_DATA_DIR on a fresh data root of the session: code defaults (empty config.json),
# an empty database created on first use, no Archive.
@pytest.fixture(scope="session")
def isolated_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("vyvar_data_root")
    (root / "config.json").write_text("{}\n", encoding="ascii")
    return root


@pytest.fixture(autouse=True)
def _isolated_data_root(isolated_data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VYVAR_DATA_DIR", str(isolated_data_root))


_CODE_TREE = Path(__file__).resolve().parents[2]
_TEST_TREE = _CODE_TREE / "dev"


def _guard_sqlite_path(database: object) -> None:
    """A test may open a database in a temporary directory or a test fixture under dev/, never
    one of the user's databases in the code tree (vyvar.sqlite3, catalogs)."""
    raw = os.fspath(database) if isinstance(database, (str, os.PathLike)) else ""
    if not raw or raw == ":memory:" or raw.startswith("file:"):
        return
    path = Path(raw).resolve()
    if path.is_relative_to(_CODE_TREE) and not path.is_relative_to(_TEST_TREE):
        raise AssertionError(f"TEST-ISOLATION-2026-10: a test opened the database {path} in the code tree")


@pytest.fixture(scope="session", autouse=True)
def _no_user_database() -> Iterator[None]:
    import sqlite3  # noqa: PLC0415

    real_connect = sqlite3.connect

    def connect(database, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        _guard_sqlite_path(database)
        return real_connect(database, *args, **kwargs)

    sqlite3.connect = connect
    yield
    sqlite3.connect = real_connect


@pytest.fixture
def tmp_path_is_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """For tests that write their own config.json into ``tmp_path`` and load it."""
    monkeypatch.setenv("VYVAR_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_hrd_service_state():
    """hrd_enrich remembers a failed web service for 10 min; tests must not share that."""
    try:
        import hrd_enrich  # noqa: PLC0415

        hrd_enrich._UNAVAILABLE.clear()
    except Exception:  # noqa: BLE001
        pass
    yield


@pytest.fixture
def infolog_lines(tmp_path: Path):
    """INFOLOG-ONE-2026-10: open a run infolog in tmp_path; call the fixture to read its lines."""
    from infolog import end_infolog_session, ensure_infolog_logging, start_infolog_session

    ensure_infolog_logging()
    path = Path(start_infolog_session(tmp_path / "infolog_draft"))
    yield lambda: path.read_text(encoding="utf-8").splitlines()
    end_infolog_session()
