"""Tests for pipeline_meta.json run provenance stamping."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from config import AppConfig
from photometry_core import merge_photometry_pipeline_meta


def _git_head() -> str:
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        )
        .strip()
    )


def test_merge_stamps_provenance_with_full_config_snapshot(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir()
    cfg = AppConfig()
    cfg.k2_mode = "literature"
    cfg.cal_diag_gate_enabled = True

    merge_photometry_pipeline_meta(
        phot,
        {"dynamic_params": {"fwhm_px": 2.1}},
        cfg,
        entry_point="run_phase2a",
    )

    meta = json.loads((phot / "pipeline_meta.json").read_text(encoding="utf-8"))
    prov = meta["provenance"]
    assert prov["git_hash"] == _git_head()
    assert isinstance(prov["git_dirty"], bool)
    assert prov["entry_point"] == "run_phase2a"
    snap = prov["config_snapshot"]
    assert snap["k2_mode"] == "literature"
    assert snap["cal_diag_gate_enabled"] is True
    assert "archive_root" in snap


def test_merge_last_writer_wins_overwrites_provenance(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir()
    cfg1 = AppConfig()
    cfg1.k2_mode = "off"
    merge_photometry_pipeline_meta(phot, {}, cfg1, entry_point="run_phase2a")

    cfg2 = AppConfig()
    cfg2.k2_mode = "literature"
    cfg2.cal_diag_gate_enabled = False
    merge_photometry_pipeline_meta(phot, {}, cfg2, entry_point="generate_masterstar_and_catalog")

    meta = json.loads((phot / "pipeline_meta.json").read_text(encoding="utf-8"))
    prov = meta["provenance"]
    assert prov["config_snapshot"]["k2_mode"] == "literature"
    assert prov["config_snapshot"]["cal_diag_gate_enabled"] is False
    assert prov["entry_point"] == "generate_masterstar_and_catalog"


def test_merge_git_unavailable_sets_null_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import photometry_core as pc

    monkeypatch.setattr(pc, "_GIT_PROVENANCE_WARNED", False)
    phot = tmp_path / "photometry"
    phot.mkdir()

    def _boom(*_a: object, **_k: object) -> str:
        raise OSError("no git")

    monkeypatch.setattr(pc.subprocess, "check_output", _boom)

    merge_photometry_pipeline_meta(phot, {}, AppConfig(), entry_point="run_phase2a")
    prov = json.loads((phot / "pipeline_meta.json").read_text(encoding="utf-8"))["provenance"]
    assert prov["git_hash"] is None
    assert prov["git_dirty"] is None


def test_merge_without_cfg_skips_provenance(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir()
    merge_photometry_pipeline_meta(phot, {"catalog_rows": 42})
    meta = json.loads((phot / "pipeline_meta.json").read_text(encoding="utf-8"))
    assert "provenance" not in meta
    assert meta["catalog_rows"] == 42
