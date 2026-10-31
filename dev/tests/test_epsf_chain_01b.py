# -*- coding: ascii -*-
"""EPSF-CHAIN-01B: config default OFF, CLI three inputs, residual G3, button force."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from night_run import parse_night_run_cli, resolve_epsf_run, resolve_night_run_cli_ids, run_night_photometry


def test_resolve_epsf_run_none_reads_config_default_off() -> None:
    assert resolve_epsf_run(None, SimpleNamespace(epsf_auto_run=False)) is False
    assert resolve_epsf_run(None, None) is False


def test_resolve_epsf_run_none_key_on() -> None:
    assert resolve_epsf_run(None, SimpleNamespace(epsf_auto_run=True)) is True


def test_resolve_epsf_run_explicit_overrides() -> None:
    cfg_off = SimpleNamespace(epsf_auto_run=False)
    cfg_on = SimpleNamespace(epsf_auto_run=True)
    assert resolve_epsf_run(True, cfg_off) is True
    assert resolve_epsf_run(False, cfg_on) is False


def test_a3_cli_refuses_each_missing_input() -> None:
    with pytest.raises(SystemExit):
        parse_night_run_cli([])
    args = parse_night_run_cli(["--source", "D:\\x"])
    _eq, _tel, _loc, missing, _src = resolve_night_run_cli_ids(
        equipment_id=args.equipment_id,
        telescope_id=args.telescope_id,
        location_id=args.location_id,
        cfg=SimpleNamespace(observer_location_id=0),
    )
    assert "camera" in missing
    assert "telescope" in missing
    assert "observing site" in missing


def test_a3_cli_accepts_three_explicit() -> None:
    args = parse_night_run_cli(
        ["--source", "D:\\x", "--camera", "1", "--telescope", "2", "--site", "3"]
    )
    eq, tel, loc, missing, src = resolve_night_run_cli_ids(
        equipment_id=args.equipment_id,
        telescope_id=args.telescope_id,
        location_id=args.location_id,
        cfg=SimpleNamespace(observer_location_id=0),
    )
    assert missing == []
    assert (eq, tel, loc) == (1, 2, 3)
    assert src == "cli_arg"


def test_a3_manifest_fills_camera_telescope(tmp_path: Path) -> None:
    man = tmp_path / "draft_manifest.json"
    man.write_text(
        '{"rig": {"equipment_id": 4, "telescope_id": 5, "location_id": 6}}',
        encoding="ascii",
    )
    eq, tel, loc, missing, src = resolve_night_run_cli_ids(
        draft_dir=tmp_path,
        cfg=SimpleNamespace(observer_location_id=0),
    )
    assert missing == []
    assert (eq, tel, loc) == (4, 5, 6)
    assert src == "manifest"


def _stub_night_photometry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    setup = "S1"
    draft = tmp_path / "draft"
    lights = draft / "detrended_aligned" / "lights" / setup
    lights.mkdir(parents=True)
    (lights / "proc_001.csv").write_text("x\n1\n", encoding="ascii")
    og = tmp_path / "ps" / setup
    og.mkdir(parents=True)
    (og / "MASTERSTAR.fits").write_bytes(b"x")
    (og / "masterstars_full_match.csv").write_text("id\n1\n", encoding="ascii")
    (og / "variable_targets.csv").write_text("id\n1\n", encoding="ascii")
    out_d = og / "photometry"
    out_d.mkdir()
    pd.DataFrame({"catalog_id": ["T1"], "mag": [11.0]}).to_csv(
        out_d / "active_targets.csv", index=False
    )
    pd.DataFrame({"catalog_id": ["T1"], "lc_rms": [0.05]}).to_csv(
        out_d / "photometry_summary.csv", index=False
    )
    lc = out_d / "lightcurves"
    lc.mkdir()
    (lc / "lightcurve_T1.csv").write_text(
        "bjd,delta_mag,err\n2461154.3,0.1,0.01\n", encoding="ascii"
    )
    all_setups = {
        setup: {
            "masterstar_fits": str(og / "MASTERSTAR.fits"),
            "obs_group_dir": str(og),
            "per_frame_csv_dir": str(lights),
            "photometry_dir": str(out_d),
        }
    }
    spy: dict[str, object] = {"epsf_calls": 0, "require_psf": None}

    monkeypatch.setattr("platesolve_ui_paths.find_photometry_setups", lambda *a, **k: all_setups)
    monkeypatch.setattr("night_run.resolve_photometry_context_triple", lambda *a, **k: {})
    monkeypatch.setattr(
        "night_run.resolve_cfg_for_photometry", lambda cfg, *a, **k: (cfg, "live", [])
    )

    def _fake_phot(**_kwargs):
        return {"aperture_photometry": {"n_lightcurves": 1, "n_frames": 1}}

    monkeypatch.setattr("photometry_core.run_full_photometry_pipeline", _fake_phot)
    monkeypatch.setattr("photometry_core.merge_photometry_pipeline_meta", lambda *a, **k: None)

    def _fake_epsf(**_kwargs):
        spy["epsf_calls"] = int(spy["epsf_calls"]) + 1
        return {"lc": {"n_written": 1}, "n_stars": 1, "merge": {}}

    monkeypatch.setattr("epsf_stage.run_epsf_stage", _fake_epsf)

    from night_run import audit_photometry_completeness as _real_audit

    def _spy_audit(path, require_psf=False):
        spy["require_psf"] = bool(require_psf)
        return _real_audit(path, require_psf=require_psf)

    monkeypatch.setattr("night_run.audit_photometry_completeness", _spy_audit)
    spy["draft"] = draft
    spy["cfg"] = SimpleNamespace(
        archive_root=tmp_path,
        epsf_auto_run=False,
        k2_mode="literature",
    )
    spy["pipeline"] = SimpleNamespace(db=None)
    return spy


def test_a1_default_off_no_stage_no_psf_completeness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _stub_night_photometry(tmp_path, monkeypatch)
    cfg = spy["cfg"]
    cfg.epsf_auto_run = False
    out = run_night_photometry(
        cfg=cfg,
        pipeline=spy["pipeline"],
        draft_id=1,
        draft_dir_override=spy["draft"],
        write_pdfs=False,
        existing_draft=True,
        epsf=None,
    )
    assert spy["epsf_calls"] == 0
    assert spy["require_psf"] is False
    assert not out.get("epsf_stage")


def test_a1_key_on_runs_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _stub_night_photometry(tmp_path, monkeypatch)
    cfg = spy["cfg"]
    cfg.epsf_auto_run = True
    out = run_night_photometry(
        cfg=cfg,
        pipeline=spy["pipeline"],
        draft_id=1,
        draft_dir_override=spy["draft"],
        write_pdfs=False,
        existing_draft=True,
        epsf=None,
    )
    assert spy["epsf_calls"] == 1
    assert spy["require_psf"] is True
    assert "S1" in (out.get("epsf_stage") or {})


def test_a2_explicit_true_forces_stage_when_key_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _stub_night_photometry(tmp_path, monkeypatch)
    cfg = spy["cfg"]
    cfg.epsf_auto_run = False
    run_night_photometry(
        cfg=cfg,
        pipeline=spy["pipeline"],
        draft_id=1,
        draft_dir_override=spy["draft"],
        write_pdfs=False,
        existing_draft=True,
        epsf=True,
    )
    assert spy["epsf_calls"] == 1
    assert spy["require_psf"] is True
