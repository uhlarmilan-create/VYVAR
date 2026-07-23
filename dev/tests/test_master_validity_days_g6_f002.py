"""G6-F002: masterdark/masterflat validity_days single source of truth (90 / 200)."""

from __future__ import annotations

import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

import config
from config import AppConfig

_EXPECTED_DARK = 90
_EXPECTED_FLAT = 200


def _post_init_fallback_literals() -> tuple[int, int]:
    src = Path(config.__file__).resolve()
    text = src.read_text(encoding="utf-8")
    dark_m = re.search(
        r'data\.get\("masterdark_validity_days",\s*(\d+)\)',
        text,
    )
    flat_m = re.search(
        r'data\.get\("masterflat_validity_days",\s*(\d+)\)',
        text,
    )
    assert dark_m and flat_m
    return int(dark_m.group(1)), int(flat_m.group(1))


def test_appconfig_empty_json_uses_90_200(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    cfg = AppConfig(project_root=tmp_path)
    assert cfg.masterdark_validity_days == _EXPECTED_DARK
    assert cfg.masterflat_validity_days == _EXPECTED_FLAT


def test_appconfig_json_missing_keys_uses_fallback(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"observer_code": "TEST"}),
        encoding="utf-8",
    )
    cfg = AppConfig(project_root=tmp_path)
    assert cfg.masterdark_validity_days == _EXPECTED_DARK
    assert cfg.masterflat_validity_days == _EXPECTED_FLAT


def test_dataclass_defaults_match_post_init() -> None:
    from tests.cython_compat import skip_if_compiled

    skip_if_compiled(
        "config",
        "source text scan requires interpreted config.py",
    )
    field_map = {f.name: f.default for f in fields(AppConfig)}
    post_dark, post_flat = _post_init_fallback_literals()

    assert field_map["masterdark_validity_days"] == _EXPECTED_DARK
    assert field_map["masterflat_validity_days"] == _EXPECTED_FLAT
    assert post_dark == _EXPECTED_DARK
    assert post_flat == _EXPECTED_FLAT


def test_settings_table_dropped_config_is_authoritative(tmp_path: Path) -> None:
    # WAVE-B STEP 5: the vestigial SETTINGS table is dropped on DB open; validity days are
    # config.json-authoritative (cfg.masterdark_validity_days / cfg.masterflat_validity_days).
    from database import VyvarDatabase

    db = VyvarDatabase(str(tmp_path / "vyvar.sqlite3"))
    row = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='SETTINGS';"
    ).fetchone()
    assert row is None, "SETTINGS table must be dropped (WAVE-B STEP 5)"
