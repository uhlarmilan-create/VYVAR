"""Catalog DB path resolution: data_root-relative, fail-loud, CWD-independent."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from config import AppConfig, resolve_config_path
from database import (
    ExoplanetCatalogError,
    GaiaCatalogError,
    VSXCatalogError,
    require_exoplanet_local_db_path,
    require_gaia_db_path,
    require_vsx_local_db_path,
)


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


def test_require_catalog_paths_fail_loud_from_wrong_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    exo = data_root / "exoplanets" / "vyvar_exoplanet_local.db"
    exo.parent.mkdir(parents=True)
    exo.write_bytes(b"")
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data_root))
    (tmp_path / "Archive" / "Drafts").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "Archive" / "Drafts")

    cfg = AppConfig()
    cfg.project_root = data_root
    cfg.__post_init__()

    with pytest.raises(GaiaCatalogError):
        require_gaia_db_path("missing/gaia.db")

    with pytest.raises(VSXCatalogError):
        require_vsx_local_db_path("missing/vsx.db")

    with pytest.raises(ExoplanetCatalogError):
        require_exoplanet_local_db_path("missing/exo.db")


def test_app_config_resolves_relative_catalog_paths_against_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    gaia = data_root / "GAIA_DR3" / "vyvar_gaia_dr3.db"
    gaia.parent.mkdir(parents=True)
    gaia.write_bytes(b"")
    vsx = data_root / "VSX" / "vyvar_vsx_local_v2.db"
    vsx.parent.mkdir(parents=True)
    vsx.write_bytes(b"")
    exo = data_root / "exoplanets" / "vyvar_exoplanet_local.db"
    exo.parent.mkdir(parents=True)
    exo.write_bytes(b"")

    cfg_json = data_root / "config.json"
    cfg_json.write_text(
        '{"gaia_db_path":"GAIA_DR3/vyvar_gaia_dr3.db",'
        '"vsx_local_db_path":"VSX/vyvar_vsx_local_v2.db",'
        '"exoplanet_local_db_path":"exoplanets/vyvar_exoplanet_local.db"}',
        encoding="utf-8",
    )
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data_root))
    (tmp_path / "wrong").mkdir()
    monkeypatch.chdir(tmp_path / "wrong")

    cfg = AppConfig()
    cfg.project_root = data_root
    cfg.__post_init__()
    assert cfg.gaia_db_path == str(gaia.resolve())
    assert cfg.vsx_local_db_path == str(vsx.resolve())
    assert cfg.exoplanet_local_db_path == str(exo.resolve())
