# -*- coding: ascii -*-
"""EPSF-CHAIN-01: run_epsf_stage wraps build / merge / LC; CLI and skip flags."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from epsf_stage import EpsfStagePaths, run_epsf_stage


def test_run_epsf_stage_skips_when_params_epsf_false(tmp_path: Path) -> None:
    params = SimpleNamespace(epsf=False)
    out = run_epsf_stage(
        params,
        EpsfStagePaths(platesolve_dir=tmp_path, frames_root=tmp_path),
        cfg=SimpleNamespace(epsf_auto_run=True),
    )
    assert out["skipped"] is True
    assert out["reason"] == "epsf=False"


def test_run_epsf_stage_skips_when_params_epsf_none_and_config_off(tmp_path: Path) -> None:
    params = SimpleNamespace(epsf=None)
    out = run_epsf_stage(
        params,
        EpsfStagePaths(platesolve_dir=tmp_path, frames_root=tmp_path),
        cfg=SimpleNamespace(epsf_auto_run=False),
    )
    assert out["skipped"] is True
    assert out["reason"] == "epsf=False"


def test_run_epsf_stage_params_none_runs_when_config_off(tmp_path: Path) -> None:
    """A2: UI ePSF button (params=None) forces the stage regardless of the key."""
    out = run_epsf_stage(
        None,
        {"platesolve_dir": tmp_path, "frames_root": tmp_path},
        cfg=SimpleNamespace(epsf_auto_run=False),
        dry_run=True,
    )
    assert out.get("skipped") is False
    assert out["dry_run"] is True


def test_run_epsf_stage_dry_run(tmp_path: Path) -> None:
    out = run_epsf_stage(
        None,
        {"platesolve_dir": tmp_path, "frames_root": tmp_path},
        cfg=SimpleNamespace(),
        dry_run=True,
    )
    assert out["dry_run"] is True
    assert out["skipped"] is False


def test_run_epsf_stage_lc_only_calls_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, object] = {}

    def _fake_lc(**kwargs):
        called["kwargs"] = kwargs
        return {"n_written": 2, "n_skipped": 0}

    monkeypatch.setattr("psf_internal_lc.write_internal_psf_lightcurves", _fake_lc)
    monkeypatch.setattr(
        "photometry_core.merge_photometry_pipeline_meta", lambda *a, **k: None
    )
    out = run_epsf_stage(
        None,
        EpsfStagePaths(platesolve_dir=tmp_path, frames_root=tmp_path / "frames"),
        cfg=SimpleNamespace(psf_photometry_enabled=False, photometry_mode="aperture"),
        do_build=False,
        do_fit_merge=False,
        do_lc=True,
    )
    assert called["kwargs"]["platesolve_dir"] == tmp_path
    assert out["lc"]["n_written"] == 2


def test_run_epsf_stage_fit_passes_write_internal_lc_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def _fake_merge(**kwargs):
        seen.update(kwargs)
        return {
            "written": 1,
            "frames_total": 1,
            "science_set": {},
            "epsf_job_summary": {},
        }

    monkeypatch.setattr("epsf_psf_merge.run_epsf_psf_merge_job", _fake_merge)
    monkeypatch.setattr(
        "photometry_core.merge_photometry_pipeline_meta", lambda *a, **k: None
    )
    run_epsf_stage(
        None,
        EpsfStagePaths(platesolve_dir=tmp_path, frames_root=tmp_path),
        cfg=SimpleNamespace(),
        do_build=False,
        do_fit_merge=True,
        do_lc=False,
        draft_id=516,
    )
    assert seen.get("write_internal_lc") is False
    assert seen.get("draft_id") == 516
