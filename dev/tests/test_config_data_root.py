# -*- coding: ascii -*-
"""Tests for RELEASE-2 data-dir separation (dev-neutral B2)."""
from __future__ import annotations

import os
from pathlib import Path

import config


def test_git_dev_checkout_uses_install_root_as_data_root(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.delenv("VYVAR_DATA_DIR", raising=False)
    assert config.resolve_data_root(tmp_path) == tmp_path.resolve()


def test_non_git_tmp_path_uses_project_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VYVAR_DATA_DIR", raising=False)
    monkeypatch.delenv("VYVAR_RELEASE_BUNDLE", raising=False)
    assert config.resolve_data_root(tmp_path) == tmp_path.resolve()


def test_vyvar_data_dir_override(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "custom_data"
    data.mkdir()
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data))
    assert config.resolve_data_root(tmp_path / "install") == data.resolve()


def test_appconfig_data_root_matches_project_root_in_git_repo() -> None:
    repo = Path(__file__).resolve().parents[2]
    if not (repo / ".git").is_dir():
        return
    os.environ.pop("VYVAR_DATA_DIR", None)
    cfg = config.AppConfig(project_root=repo)
    assert cfg.data_root.resolve() == repo.resolve()
    assert cfg.archive_root.resolve() == (repo / "Archive").resolve()
