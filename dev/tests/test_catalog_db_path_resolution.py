"""Catalog DB path resolution: data_root-relative, fail-loud, CWD-independent."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import AppConfig, resolve_config_path


def test_resolve_config_path_uses_data_root_not_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    db = data_root / "cats" / "test.db"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    (tmp_path / "other_cwd").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path / "other_cwd")
    resolved = resolve_config_path("cats/test.db", data_root)
    assert resolved == str(db.resolve())


def test_app_config_resolves_relative_catalog_paths_against_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    gaia = data_root / "GAIA_DR3" / "vyvar_gaia_dr3.db"
    gaia.parent.mkdir(parents=True)
    gaia.write_bytes(b"")
    objects = data_root / "catalogs" / "vyvar_objects.db"
    objects.parent.mkdir(parents=True)
    objects.write_bytes(b"")

    cfg_json = data_root / "config.json"
    cfg_json.write_text(
        '{"gaia_db_path":"GAIA_DR3/vyvar_gaia_dr3.db",'
        '"objects_db_path":"catalogs/vyvar_objects.db"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data_root))
    (tmp_path / "wrong").mkdir()
    monkeypatch.chdir(tmp_path / "wrong")

    cfg = AppConfig()
    cfg.project_root = data_root
    cfg.__post_init__()
    assert cfg.gaia_db_path == str(gaia.resolve())
    assert cfg.objects_db_path == str(objects.resolve())
