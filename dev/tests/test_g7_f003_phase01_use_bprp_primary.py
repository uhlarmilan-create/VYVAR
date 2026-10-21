"""G7-F003: phase01_use_bprp_primary is a removed display-only key (KNOWN_REMOVED)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from config import AppConfig


def test_phase01_use_bprp_primary_removed_loads_silently(tmp_path: Path, caplog) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"phase01_use_bprp_primary": False}),
        encoding="utf-8",
    )
    caplog.set_level(logging.INFO)
    cfg = AppConfig(project_root=tmp_path)
    assert not hasattr(cfg, "phase01_use_bprp_primary")
    assert any("phase01_use_bprp_primary removed 2026-09-01" in r.message for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
