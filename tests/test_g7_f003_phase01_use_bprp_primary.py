"""G7-F003: phase01_use_bprp_primary is a real AppConfig field (persistable, default True)."""

from __future__ import annotations

import json
from pathlib import Path

from config import AppConfig


def test_appconfig_default_phase01_use_bprp_primary_true() -> None:
    cfg = AppConfig()
    assert cfg.phase01_use_bprp_primary is True


def test_phase01_use_bprp_primary_round_trips_config_json(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"phase01_use_bprp_primary": False}),
        encoding="utf-8",
    )
    cfg = AppConfig(project_root=tmp_path)
    assert cfg.phase01_use_bprp_primary is False
    assert cfg.to_dict()["phase01_use_bprp_primary"] is False
