# -*- coding: ascii -*-
"""Bundled data-dir bootstrap wiring (BUNDLE-BOOTSTRAP-WIRING / bug #9)."""
from __future__ import annotations

import json
import os
import stat
import sys

import pytest

import config
import params_registry as pr
from database import VyvarDatabase


def _fake_install(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    (install / "RUNTIME_PIN.json").write_text("{}", encoding="ascii")
    (install / "python" / "python.exe").parent.mkdir(parents=True)
    (install / "python" / "python.exe").write_text("", encoding="ascii")
    return install


def test_bootstrap_fresh_data_dir_creates_skeleton_and_config(tmp_path, monkeypatch) -> None:
    install = _fake_install(tmp_path)
    data = tmp_path / "data"
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data))
    monkeypatch.delenv("VYVAR_RELEASE_BUNDLE", raising=False)

    from vyvar_runtime import bootstrap_release_data_dir

    data_root, report = bootstrap_release_data_dir(install)
    assert data_root == data.resolve()
    assert report["data_root"] == "created"
    assert report["Archive"] == "created"
    assert report["config.json"] == "created"
    assert report["vyvar.sqlite3"] == "created"

    for rel in (
        "Archive",
        "Archive/Drafts",
        "CalibrationLibrary",
        "GAIA_DR3",
        "VSX",
        "exoplanets",
        "logs",
    ):
        assert (data / rel).is_dir(), rel

    cfg_path = data / "config.json"
    payload = json.loads(config.strip_jsonc_comments(cfg_path.read_text(encoding="utf-8")))
    expected = set(config.AppConfig(project_root=install).to_json().keys())
    missing = expected - set(payload.keys())
    assert not missing, f"missing persisted keys: {sorted(missing)[:8]}"
    assert len(payload) == len(expected)
    assert int(payload.get("observer_location_id") or 0) == 1

    db = VyvarDatabase(str(data / "vyvar.sqlite3"))
    db.close()


def test_bootstrap_second_run_reports_preexisting(tmp_path, monkeypatch) -> None:
    install = _fake_install(tmp_path)
    data = tmp_path / "data"
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data))

    from vyvar_runtime import bootstrap_release_data_dir

    bootstrap_release_data_dir(install)
    _data_root, report2 = bootstrap_release_data_dir(install)
    assert report2["data_root"] == "preexisting"
    assert report2["Archive"] == "preexisting"
    assert report2["config.json"] == "preexisting"
    assert report2["vyvar.sqlite3"] == "preexisting"


def test_app_startup_bootstrap_same_as_selftest(tmp_path, monkeypatch) -> None:
    """Simulate bundled app startup calling ensure_release_data_dir before DB/pipeline."""
    install = _fake_install(tmp_path)
    data = tmp_path / "data"
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data))

    from vyvar_runtime import ensure_release_data_dir

    data_root = ensure_release_data_dir(install)
    assert data_root == data.resolve()
    assert (data / "config.json").is_file()
    assert (data / "vyvar.sqlite3").is_file()
    VyvarDatabase(str(data / "vyvar.sqlite3")).close()


def test_open_sqlite_connection_creates_parent_dir(tmp_path) -> None:
    from database import open_sqlite_connection

    db_path = tmp_path / "nested" / "vyvar.sqlite3"
    conn = open_sqlite_connection(db_path)
    conn.close()
    assert db_path.is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="readonly dir bootstrap not emulated on Windows")
def test_bootstrap_unwritable_data_dir_reports_failed(tmp_path, monkeypatch) -> None:
    install = _fake_install(tmp_path)
    ro = tmp_path / "readonly_parent"
    ro.mkdir()
    data = ro / "data"
    data.mkdir()
    os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data))

    from vyvar_runtime import bootstrap_failures, bootstrap_release_data_dir

    try:
        _data_root, report = bootstrap_release_data_dir(install)
        failures = bootstrap_failures(report)
        assert failures, f"expected FAILED entries, got {report}"
    finally:
        os.chmod(ro, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def test_selftest_bootstrap_root_cause_no_ephemeral_tmp(tmp_path, monkeypatch) -> None:
    """Regression: selftest must bootstrap resolve_data_root(), not an internal tempfile."""
    install = _fake_install(tmp_path)
    data = tmp_path / "user_data"
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data))

    from vyvar_runtime import bootstrap_release_data_dir

    _data_root, report = bootstrap_release_data_dir(install)
    assert data.is_dir()
    assert report["Archive"] == "created"
    assert not any(
        "vyvar_selftest_data_" in str(p)
        for p in data.parent.iterdir()
        if p.is_dir() and p != data
    )


def test_git_dev_checkout_skips_bootstrap(tmp_path, monkeypatch) -> None:
    install = tmp_path / "repo"
    install.mkdir()
    (install / ".git").mkdir()
    data = tmp_path / "data"
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data))

    from vyvar_runtime import bootstrap_release_data_dir

    _data_root, report = bootstrap_release_data_dir(install)
    assert report["_bootstrap"] == "skipped:git_dev_checkout"
    assert not data.exists()
