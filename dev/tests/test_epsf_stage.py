# -*- coding: ascii -*-
"""EPSF-CHAIN-01: run_epsf_stage wraps build / merge / LC; CLI and skip flags."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from epsf_stage import EpsfStagePaths, run_epsf_stage


def _write_fit_set(ps: Path, vsx_type: str = "EW") -> None:
    """Minimal photometry/ tables: one VSX target of ``vsx_type`` with an aperture light curve and one
    comparison star."""
    phot = ps / "photometry"
    (phot / "lightcurves").mkdir(parents=True, exist_ok=True)
    (phot / "active_targets.csv").write_text(
        "name,vsx_type,catalog_id,zone_flag,catalog,skip_photometry\nT,%s,111,linear,VSX,False\n" % vsx_type,
        encoding="ascii",
    )
    (phot / "comparison_stars_per_target.csv").write_text(
        "catalog_id,target_catalog_id\n222,111\n", encoding="ascii"
    )
    (phot / "lightcurves" / "lightcurve_111.csv").write_text("bjd,delta_mag\n", encoding="ascii")


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
    _write_fit_set(tmp_path)
    monkeypatch.setattr(
        "photometry_core.merge_photometry_pipeline_meta", lambda *a, **k: None
    )
    out = run_epsf_stage(
        None,
        EpsfStagePaths(platesolve_dir=tmp_path, frames_root=tmp_path / "frames"),
        cfg=SimpleNamespace(),
        do_build=False,
        do_fit_merge=False,
        do_lc=True,
    )
    off = run_epsf_stage(None, EpsfStagePaths(platesolve_dir=tmp_path, frames_root=tmp_path / "frames"),
                         cfg=SimpleNamespace(epsf_export_auto=False), do_build=False, do_fit_merge=False, do_lc=True)
    assert off["export"] is None  # epsf_export_auto off: the ePSF tab's button writes the files
    assert called["kwargs"]["platesolve_dir"] == tmp_path
    assert called["kwargs"]["target_ids"] == ["111"]
    assert out["lc"]["n_written"] == 2
    assert out["export"] == {"n_written": 0, "skipped": {"111": "no ePSF light curve"}}  # EPSF-EXPORT-2026-10


def test_run_epsf_stage_fit_gets_the_selected_fit_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """EPSF-TARGETS-2026-10: the merge job fits the stage's fit set; ``targets`` overrides cfg.epsf_targets."""
    seen: dict[str, object] = {}

    def _fake_merge(**kwargs):
        seen.update(kwargs)
        return {"written": 1, "frames_total": 1, "epsf_job_summary": {}}

    monkeypatch.setattr("epsf_psf_merge.run_epsf_psf_merge_job", _fake_merge)
    monkeypatch.setattr("photometry_core.merge_photometry_pipeline_meta", lambda *a, **k: None)
    _write_fit_set(tmp_path)
    kw = dict(do_build=False, do_fit_merge=True, do_lc=False, draft_id=516)
    paths = EpsfStagePaths(platesolve_dir=tmp_path, frames_root=tmp_path)
    run_epsf_stage(None, paths, cfg=SimpleNamespace(epsf_targets="E*"), **kw)
    assert seen["fit_set"].selection == "E*" and seen["fit_set"].target_ids == {"111"}
    seen.clear()
    out = run_epsf_stage(None, paths, cfg=SimpleNamespace(epsf_targets="E*"), targets="EA", **kw)
    assert not seen and out["fit_set"]["selection"] == "EA" and out["fit_set"]["n_total"] == 0


def test_run_epsf_stage_skips_fit_and_lc_when_nothing_is_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EPSF-TARGETS-2026-10: no target matches the selection -> fit and LC skipped."""

    def _boom(**kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr("epsf_psf_merge.run_epsf_psf_merge_job", _boom)
    monkeypatch.setattr("psf_internal_lc.write_internal_psf_lightcurves", _boom)
    monkeypatch.setattr("photometry_core.merge_photometry_pipeline_meta", lambda *a, **k: None)
    _write_fit_set(tmp_path, vsx_type="ROT")
    out = run_epsf_stage(
        None,
        EpsfStagePaths(platesolve_dir=tmp_path, frames_root=tmp_path),
        cfg=SimpleNamespace(epsf_targets="E*"),
        do_build=False,
        draft_id=516,
    )
    assert out["merge"] is None and out["lc"] is None
    assert out["fit_set"]["n_total"] == 0


def test_stale_epsf_products_of_unselected_targets_are_removed(tmp_path: Path) -> None:
    """EPSF-STALE-2026-10: a new fit replaces every psf_* column, so older ePSF curves of other targets go."""
    from epsf_stage import remove_stale_epsf_products

    phot = tmp_path / "photometry"
    lc = phot / "lightcurves"
    exp = phot / "lightcurves_reports" / "epsf"
    lc.mkdir(parents=True)
    exp.mkdir(parents=True)
    for cid in ("1", "2"):
        (lc / f"lightcurve_{cid}_psf.csv").write_text("bjd\n")
        (lc / f"lightcurve_{cid}.csv").write_text("bjd\n")
        (exp / f"V{cid}_20261002_g_120_4_epsf.txt").write_text(f"# catalog_id={cid}\nbjd_tdb,delta_mag,err\n")
    (exp / "V2_20261002_r_100_4_epsf.txt").write_text("# catalog_id=2\nbjd_tdb,delta_mag,err\n")
    assert remove_stale_epsf_products(phot, "g_120_4", {"1"}) == ["2"]
    assert sorted(p.name for p in lc.iterdir()) == ["lightcurve_1.csv", "lightcurve_1_psf.csv", "lightcurve_2.csv"]
    assert sorted(p.name for p in exp.iterdir()) == ["V1_20261002_g_120_4_epsf.txt", "V2_20261002_r_100_4_epsf.txt"]
