"""Shared pytest configuration."""
from __future__ import annotations

import sys
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
