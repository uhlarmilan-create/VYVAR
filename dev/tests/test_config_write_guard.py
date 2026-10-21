"""CONFIG-WRITE-GUARD (PARAM-OWNERSHIP-WAVE-A STEP 1).

config.json must be persisted ONLY from an explicit UI save action, wrapped in
``config.ui_config_persist()``. The headless / pipeline path must never write it.

Two levels of guard:
  1. Unit: ``save_config_json`` raises ``ConfigPersistError`` outside the context and
     succeeds inside it (tmp_path only; never touches the repo config.json).
  2. Contract: pipeline / headless modules do not call ``save_config_json`` nor open
     the ``ui_config_persist`` context (source scan), so persistence stays UI-only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from config import ConfigPersistError, save_config_json, ui_config_persist

_SRC = Path(config.__file__).resolve().parent  # src_py/

# Headless / pipeline entry surface that must never persist config.json.
_HEADLESS_MODULES = [
    "pipeline.py",
    "photometry_core.py",
    "night_run.py",
    "simulate_night_run.py",
]


def test_save_config_json_raises_without_context(tmp_path: Path) -> None:
    with pytest.raises(ConfigPersistError):
        save_config_json(tmp_path, {"observer_location_name": "should_not_write"})
    assert not (tmp_path / "config.json").exists()


def test_save_config_json_allowed_within_ui_context(tmp_path: Path) -> None:
    with ui_config_persist():
        save_config_json(tmp_path, {"observer_location_name": "ok"})
    assert (tmp_path / "config.json").exists()


def test_persist_flag_resets_after_context(tmp_path: Path) -> None:
    with ui_config_persist():
        pass
    # Flag must be back to disallowed once the context exits.
    with pytest.raises(ConfigPersistError):
        save_config_json(tmp_path, {"x": 1})


@pytest.mark.parametrize("module_name", _HEADLESS_MODULES)
def test_headless_modules_do_not_persist_config(module_name: str) -> None:
    path = _SRC / module_name
    assert path.is_file(), f"expected {path} to exist"
    text = path.read_text(encoding="utf-8")
    assert "save_config_json(" not in text, (
        f"{module_name} calls save_config_json(): headless/pipeline code must not persist config.json"
    )
    assert "ui_config_persist(" not in text, (
        f"{module_name} opens ui_config_persist(): only UI-layer modules may enable config persistence"
    )
