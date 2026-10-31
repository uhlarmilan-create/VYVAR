# -*- coding: ascii -*-
"""Step 6 s6_fix (Milan's decisions 2026-10-31, second round)."""
from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src_py"


# ------------------------------------------------------------------ 1. ePSF target selection (G12-04)
def _phot_tables(ps: Path, rows: list[dict], comps: list[tuple[str, str]], lcs: list[str]) -> Path:
    phot = ps / "photometry"
    (phot / "lightcurves").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(phot / "active_targets.csv", index=False)
    pd.DataFrame(comps, columns=["catalog_id", "target_catalog_id"]).to_csv(
        phot / "comparison_stars_per_target.csv", index=False)
    for cid in lcs:
        (phot / "lightcurves" / f"lightcurve_{cid}.csv").write_text("bjd,delta_mag\n")
    return phot


def test_one_out_of_scope_predicate_for_both_photometries() -> None:
    """vsx_out_of_scope_types: VSX auto targets only, token match; the aperture target selection, its
    invariant and the ePSF candidates all use ``target_is_out_of_scope``."""
    from vsx_type_scope import target_is_out_of_scope

    assert target_is_out_of_scope({"catalog": "VSX", "vsx_type": "EA:|RS"}, ["ea"])
    assert not target_is_out_of_scope({"catalog": "MANUAL", "vsx_type": "EA"}, ["EA"])
    assert not target_is_out_of_scope({"catalog": "EXOPLANET", "vsx_type": "EA"}, ["EA"])
    assert target_is_out_of_scope({"vsx_name": "V1 Cyg", "vsx_type": "ROT"}, ["ROT"])  # legacy row = VSX auto
    assert not target_is_out_of_scope({"catalog": "VSX", "vsx_type": "EAX"}, ["EA"])  # no substring match
    assert not target_is_out_of_scope({"catalog": "VSX", "vsx_type": "EA"}, [])
    for mod in ("photometry_comp.py", "invariants_runtime.py", "epsf_science_set.py"):
        text = (SRC / mod).read_text(encoding="utf-8")
        assert "target_is_out_of_scope(" in text, mod
        assert "vsx_type_is_out_of_scope(" not in text, mod


def test_epsf_never_fits_an_out_of_scope_target_even_with_an_aperture_lc(tmp_path: Path) -> None:
    from epsf_science_set import build_epsf_fit_set, epsf_target_candidates

    rows = [
        {"catalog_id": "1", "vsx_type": "EW", "catalog": "VSX", "zone_flag": "linear", "skip_photometry": False},
        {"catalog_id": "2", "vsx_type": "ROT", "catalog": "VSX", "zone_flag": "linear", "skip_photometry": False},
        {"catalog_id": "3", "vsx_type": "ROT", "catalog": "MANUAL", "zone_flag": "linear", "skip_photometry": False},
    ]
    _phot_tables(tmp_path, rows, [("10", "1"), ("11", "2"), ("12", "3")], ["1", "2", "3"])
    fs = build_epsf_fit_set(tmp_path, ["ROT"])
    assert fs.target_ids == {"1", "3"} and fs.comp_ids == {"10", "12"}
    assert set(epsf_target_candidates(tmp_path, ["ROT"])["catalog_id"]) == {"1", "3"}
    assert build_epsf_fit_set(tmp_path, []).target_ids == {"1", "2", "3"}
    assert "selection" not in fs.to_meta_dict()


def test_epsf_targets_is_removed_and_an_old_config_still_loads(tmp_path: Path, caplog) -> None:
    import config
    from config_removed_keys import KNOWN_REMOVED_KEYS

    assert "epsf_targets" in KNOWN_REMOVED_KEYS
    (tmp_path / "config.json").write_text(json.dumps({"epsf_targets": "E*", "epsf_auto_run": True}), encoding="utf-8")
    with caplog.at_level(logging.INFO):
        cfg = config.AppConfig.from_file(tmp_path / "config.json")
    assert not hasattr(cfg, "epsf_targets") and cfg.epsf_auto_run is True
    assert "epsf_targets" not in cfg.to_json()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and "epsf_targets" in r.getMessage()]
    assert any("epsf_targets removed" in r.getMessage() for r in caplog.records)
    reg = json.loads((SRC.parent / "dev" / "validation" / "params_registry.json").read_text(encoding="utf-8"))
    assert "epsf_targets" not in reg
    for doc in ("VYVAR_PARAMS.md", "VYVAR_CONFIG_GUIDE_EN.md", "VYVAR_CONFIG_GUIDE_CZ.md"):
        assert "| `epsf_targets` |" not in (SRC.parent / "docs" / doc).read_text(encoding="utf-8"), doc
    assert '"epsf_targets"' not in (SRC.parent / "config.json").read_text(encoding="utf-8")


def test_epsf_stage_and_callers_have_no_target_selection() -> None:
    import epsf_stage
    import epsf_science_set

    assert "targets" not in inspect.signature(epsf_stage.run_epsf_stage).parameters
    assert not hasattr(epsf_stage, "configured_epsf_targets")
    assert not hasattr(epsf_science_set, "parse_epsf_selection")
    for mod in ("night_run.py", "task_jobs.py", "ui_epsf_dashboard.py", "epsf_export.py", "star_page_m3.py",
                "settings_model.py", "config.py"):
        assert "epsf_targets" not in (SRC / mod).read_text(encoding="utf-8"), mod


def test_completeness_gate_has_no_selection_exception(tmp_path: Path) -> None:
    """Every aperture LC needs a PSF LC (n_full > 0 or a drop reason) - with ePSF required, a missing one fails."""
    from night_run import audit_psf_lc_completeness

    lc = tmp_path / "lightcurves"
    lc.mkdir()
    for cid in ("1", "2"):
        (lc / f"lightcurve_{cid}.csv").write_text("bjd\n")
    (lc / "lightcurve_1_psf.csv").write_text("# psf_lc_n_epochs_full=5\nbjd\n")
    out = audit_psf_lc_completeness(tmp_path, require=True)
    assert out["ok"] is False and out["n_missing_psf"] == 1
    (lc / "lightcurve_2_psf.csv").write_text("# psf_lc_n_epochs_full=3\nbjd\n")
    assert audit_psf_lc_completeness(tmp_path, require=True)["ok"] is True


def test_epsf_runs_in_the_night_run_only_by_epsf_auto_run() -> None:
    """RUN VYVAR from the GUI is the night run (no epsf override passed); the night run decides with
    ``epsf_auto_run``; the ePSF tab starts the stage only from its button (a job)."""
    from types import SimpleNamespace

    from night_run import resolve_epsf_run

    assert resolve_epsf_run(None, SimpleNamespace(epsf_auto_run=True)) is True
    assert resolve_epsf_run(None, SimpleNamespace(epsf_auto_run=False)) is False
    app = (SRC / "app.py").read_text(encoding="utf-8")
    assert "epsf=" not in app and "run_epsf_stage" not in app
    dash = (SRC / "ui_epsf_dashboard.py").read_text(encoding="utf-8")
    i = dash.index('start_task_job(cfg.data_root, "run_epsf"')
    assert "if run:" in dash[i - 200:i]


# ------------------------------------------------------------------ 2. run lock of the CLI night run
def _fake_running_job(data_root: Path, pid: int, name: str = "run_20261031_120000") -> Path:
    d = data_root / "jobs" / name
    d.mkdir(parents=True)
    (d / "run_status.json").write_text(json.dumps({"state": "running", "pid": pid, "label": "RUN VYVAR"}))
    return d


def _cli(monkeypatch, data_root: Path, run):
    from types import SimpleNamespace

    import night_run
    import vyvar_runtime

    monkeypatch.setattr(vyvar_runtime, "ensure_data_dir_for_entry", lambda *a, **k: None)
    monkeypatch.setattr(night_run, "_load_app_config", lambda p: SimpleNamespace(data_root=data_root))
    monkeypatch.setattr(night_run, "resolve_night_run_cli_ids", lambda **k: (1, 2, 3, [], "cli"))
    monkeypatch.setattr(night_run, "run_night_pipeline", run)
    return night_run.main(["--source", str(data_root / "night"), "--camera", "1", "--telescope", "2",
                           "--site", "3"])


def test_cli_night_run_refuses_while_a_job_of_the_data_dir_runs(tmp_path: Path, monkeypatch, capsys) -> None:
    import os

    _fake_running_job(tmp_path, os.getpid())
    called = []
    rc = _cli(monkeypatch, tmp_path, lambda params, cfg=None: called.append(1))
    assert rc == 2 and not called
    assert "RUN VYVAR is running" in capsys.readouterr().err


def test_cli_night_run_registers_itself_and_a_dead_pid_does_not_block(tmp_path: Path, monkeypatch) -> None:
    import night_run
    import run_job

    _fake_running_job(tmp_path, 2 ** 22 + 12345)  # no such process: lost, not running
    seen = {}

    def run(params, cfg=None):
        busy = run_job.running_job(tmp_path)
        seen["busy"] = busy
        seen["label"] = (run_job.read_status(busy) or {}).get("label")
        with __import__("pytest").raises(run_job.JobBusyError):  # the GUI cannot start a job meanwhile
            run_job.spawn_job(tmp_path, spec={}, module="x", label="RUN ePSF", prefix="task")
        return night_run.NightRunResult(success=True, draft_id=7)

    assert _cli(monkeypatch, tmp_path, run) == 0
    assert seen["busy"] is not None and seen["busy"].name.startswith("cli_") and seen["label"] == "Night run (CLI)"
    st = run_job.read_status(seen["busy"])
    assert st["state"] == "done" and st["draft_id"] == 7 and run_job.running_job(tmp_path) is None


def test_cli_night_run_crash_releases_the_lock(tmp_path: Path, monkeypatch) -> None:
    import pytest

    import run_job

    def boom(params, cfg=None):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _cli(monkeypatch, tmp_path, boom)
    (d,) = [p for p in (tmp_path / "jobs").iterdir() if p.is_dir()]
    assert run_job.read_status(d)["state"] == "failed" and run_job.running_job(tmp_path) is None


# ------------------------------------------------------------------ 3. task jobs use the field's config snapshot
def test_variability_tess_and_catalogue_jobs_use_the_field_config_snapshot(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    import catalog_crossmatch
    import task_jobs
    import tess_runner
    import variability_field

    snap = SimpleNamespace(name="snapshot")
    asked = []
    monkeypatch.setattr(task_jobs, "draft_task_cfg", lambda d: asked.append(Path(d)) or (snap, "draft_snapshot", []))
    monkeypatch.setattr(task_jobs, "_live_cfg", lambda: (_ for _ in ()).throw(AssertionError("live cfg used")))
    seen = []
    monkeypatch.setattr(variability_field, "run_field_variability",
                        lambda frames, phot, cfg, platesolve_dir=None: seen.append(cfg) or {"status": "ok"})
    monkeypatch.setattr(tess_runner, "run_tess_for_star", lambda phot, cid, cfg, progress=None: seen.append(cfg))
    monkeypatch.setattr(variability_field, "load_variability_candidates",
                        lambda phot: pd.DataFrame([{"catalog_id": "1", "ra_deg": 1.0, "dec_deg": 2.0}]))
    monkeypatch.setattr(variability_field, "write_candidate_catalogs", lambda phot, d: None)
    monkeypatch.setattr(catalog_crossmatch, "_catalog_check", lambda row, cfg: seen.append(cfg) or "x")
    args = {"draft_dir": str(tmp_path), "setup": "V_60_1", "catalog_id": "1"}
    for task in ("variability_search", "variability_catalog_check", "tess"):
        task_jobs.TASKS[task](args, lambda m: None)
    assert seen == [snap, snap, snap] and asked == [tmp_path] * 3


# ------------------------------------------------------------------ 4. manifest field centre per setup
def _ms(ps_setup: Path, ra: float, dec: float) -> None:
    import numpy as np
    from astropy.io import fits

    ps_setup.mkdir(parents=True, exist_ok=True)
    h = fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": ra, "CRVAL2": dec, "CRPIX1": 50.5,
                     "CRPIX2": 40.5, "CDELT1": -4e-4, "CDELT2": 4e-4})
    fits.PrimaryHDU(np.zeros((80, 100), dtype=np.float32), header=h).writeto(ps_setup / "MASTERSTAR.fits")


def test_every_setup_keeps_its_own_centre_and_consumers_use_their_setups(tmp_path: Path) -> None:
    from astropy.io import fits

    import draft_provenance as dp
    from fits_meta import field_centre_radec
    from manifest_centre import record_manifest_centres_from_masterstars, setup_centre

    archive = tmp_path / "Archive"
    did = dp.create_draft_manifest(archive, {"object_name": "AA Boo", "observation_start_jd": 2460970.4})
    d = dp.draft_dir_for_id(archive, did)
    _ms(d / "platesolve" / "V_60_1", 201.0, 41.0)   # camera 1
    _ms(d / "platesolve" / "R_30_2", 201.5, 41.2)   # another instrument on the same object
    record_manifest_centres_from_masterstars(d, did)
    man = dp.load_draft_manifest(d)
    assert "center" not in man and set(man["center_by_setup"]) == {"V_60_1", "R_30_2"}
    row = dp.obs_draft_row_from_manifest(man, did)

    class _Db:
        def fetch_obs_draft_by_id(self, _i):
            return row

    hv = fits.Header({"FILTER": "V", "EXPTIME": 60.0, "XBINNING": 1})
    hr = fits.Header({"FILTER": "R", "EXPTIME": 30.0, "XBINNING": 2})
    hb = fits.Header({"FILTER": "B", "EXPTIME": 60.0, "XBINNING": 1})  # a setup without a solved centre
    rv, dv, sv = field_centre_radec(hv, db=_Db(), draft_id=did)
    rr, dr, sr = field_centre_radec(hr, db=_Db(), draft_id=did)
    assert sv == sr == "DRAFT_CENTRE" and abs(rv - 201.0) < 1e-3 and abs(rr - 201.5) < 1e-3 and abs(dr - 41.2) < 1e-3
    assert field_centre_radec(hb, db=_Db(), draft_id=did) == (None, None, "none")  # never another setup's centre
    # OSC: the channel setups of one light folder are one instrument
    assert setup_centre({"NoFilter_60_1_G": {"ra_deg": 10.0, "de_deg": 5.0}}, "NoFilter_60_1") == (10.0, 5.0)
    assert setup_centre({"NoFilter_60_1": {"ra_deg": 10.0, "de_deg": 5.0}}, "NoFilter_60_1_R") == (10.0, 5.0)
    assert setup_centre({"V_60_1": {"ra_deg": 10.0, "de_deg": 5.0}}, "R_60_1") is None


def test_preprocess_hint_uses_the_centre_of_each_light_folder(tmp_path: Path, monkeypatch) -> None:
    import pipeline_calibrate

    root = tmp_path / "calibrated" / "lights"
    for setup in ("V_60_1", "R_30_2", "B_60_1"):
        (root / setup).mkdir(parents=True)
        (root / setup / "f1.fits").write_bytes(b"")
    seen = {}

    def _map(fn, tasks, **k):
        for t in tasks:
            seen[Path(t[0]).parent.name] = (t[3], t[4])
        raise StopIteration

    monkeypatch.setattr("parallel_frames.map_frames", _map)
    monkeypatch.setattr("parallel_frames.frame_bytes_of_fits", lambda p: 1)
    monkeypatch.setattr("parallel_frames.stage_workers", lambda *a, **k: 1)
    monkeypatch.setattr(pipeline_calibrate, "_manifest_frame_qc_by_calibrated_path", lambda *a, **k: {})
    from types import SimpleNamespace

    cfg = SimpleNamespace(preprocess_sky_surface_order=0, detection_threshold_sigma=5.0)
    monkeypatch.setattr("config.resolve_detection_threshold_sigma", lambda c: 5.0)
    centres = {"V_60_1": {"ra_deg": 201.0, "de_deg": 41.0}, "R_30_2": {"ra_deg": 201.5, "de_deg": 41.2}}
    try:
        pipeline_calibrate._qc_enrich_calibrated_in_place(root, app_config=cfg, target_by_setup=centres)
    except StopIteration:
        pass
    assert seen == {"V_60_1": (201.0, 41.0), "R_30_2": (201.5, 41.2), "B_60_1": (None, None)}
    seen.clear()
    try:  # the run's own pointing wins for every frame
        pipeline_calibrate._qc_enrich_calibrated_in_place(root, app_config=cfg, target_ra=1.0, target_dec=2.0,
                                                          target_by_setup=centres)
    except StopIteration:
        pass
    assert set(seen.values()) == {(1.0, 2.0)}


def test_check_manifest_centres_compares_each_setup_with_its_own_masterstar(tmp_path: Path, capsys) -> None:
    import importlib.util

    import draft_provenance as dp

    spec = importlib.util.spec_from_file_location("cmc", SRC.parent / "dev" / "tools" / "check_manifest_centres.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    archive = tmp_path / "Archive"
    did = dp.create_draft_manifest(archive, {"object_name": "AA Boo", "observation_start_jd": 2460970.4})
    d = dp.draft_dir_for_id(archive, did)
    _ms(d / "platesolve" / "V_60_1", 201.0, 41.0)
    _ms(d / "platesolve" / "R_30_2", 150.0, 10.0)
    dp.patch_draft_manifest(d, did, center_by_setup={"V_60_1": {"ra_deg": 201.0, "de_deg": 41.0},
                                                     "R_30_2": {"ra_deg": 150.0, "de_deg": 10.0}})
    assert tool.main([str(archive / "Drafts")]) == 0
    # the old field-level rule (first setup's centre for every setup) is a mismatch for R_30_2
    dp.patch_draft_manifest(d, did, center_by_setup={"V_60_1": {"ra_deg": 201.0, "de_deg": 41.0},
                                                     "R_30_2": {"ra_deg": 201.0, "de_deg": 41.0}})
    assert tool.main([str(archive / "Drafts")]) == 1
    out = capsys.readouterr().out
    assert "R_30_2\tMISMATCH" in out and "V_60_1\tMISMATCH" not in out


# ------------------------------------------------------------------ 5. the observing site is never auto-selected
def test_autodetect_fills_camera_and_telescope_never_the_site(tmp_path: Path) -> None:
    import numpy as np
    from astropy.io import fits

    import optics_autodetect as oa

    src = tmp_path / "night"
    src.mkdir()
    h = fits.Header({"IMAGETYP": "LIGHT", "FOCALLEN": 800.0, "APTDIA": 200.0, "TELESCOP": "Newton 200",
                     "SITELAT": 50.112, "SITELONG": 14.698})
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32), header=h).writeto(src / "a.fits")
    tel = [{"ID": 1, "TELESCOPENAME": "Newton 200", "ALIAS": "a", "DIAMETER": 200.0, "FOCAL": 800.0}]
    rep = oa.autodetect_from_source(src, equipments=[], telescopes=tel)
    assert rep.telescope.matched_id == 1
    assert not hasattr(rep, "location") and not hasattr(oa, "detect_location")
    assert "locations" not in inspect.signature(oa.autodetect_from_source).parameters
    app = (SRC / "app.py").read_text(encoding="utf-8")
    assert "_rep.location" not in app and '_rig_set(_RIG_LOCATION' not in app
    # the run keeps the header site handling
    import param_resolver

    assert hasattr(param_resolver, "resolve_site")


# ------------------------------------------------------------------ 6. Calibration Library view is read-only
_CLIB_SCRIPT = """
import sys
sys.path.insert(0, {src!r})
from pathlib import Path
import streamlit as st
from database import VyvarDatabase
from ui_calibration_library import render_calibration_library_dashboard

db = VyvarDatabase({db!r})
stmts = []
db.conn._conn.set_trace_callback(stmts.append)
render_calibration_library_dashboard(calibration_library_root=Path({root!r}), dark_validity_days=30,
                                     flat_validity_days=30, db=db, data_root=Path({data!r}))
db.conn._conn.set_trace_callback(None)
st.session_state["stmts"] = stmts
"""


def test_calibration_library_render_creates_nothing(tmp_path: Path) -> None:
    from streamlit.testing.v1 import AppTest

    from database import VyvarDatabase

    db_path = tmp_path / "vyvar.sqlite3"
    VyvarDatabase(str(db_path)).close()  # the table exists from the database open
    root = tmp_path / "CalibrationLibrary"
    script = tmp_path / "clib_app.py"
    script.write_text(_CLIB_SCRIPT.format(src=str(SRC), db=str(db_path), root=str(root), data=str(tmp_path)),
                      encoding="utf-8")
    at = AppTest.from_file(str(script), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert not root.exists()
    writes = [s for s in at.session_state["stmts"]
              if s.strip().split()[0].upper() in ("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP")]
    assert writes == []
    assert any("does not exist" in str(w.value) for w in at.warning)


def test_calibration_library_schema_is_made_when_the_database_opens(tmp_path: Path) -> None:
    import calib_library as cl
    from database import VyvarDatabase

    db = VyvarDatabase(str(tmp_path / "v.sqlite3"))
    try:
        assert getattr(db, "_calib_library_schema_ok", False)
        cols = {r[1] for r in db.conn.execute("PRAGMA table_info('CALIBRATION_LIBRARY');").fetchall()}
        assert {"ID_EQUIPMENTS", "CAPTURE_DATE"} <= cols
        assert cl.list_masters(db) == []
    finally:
        db.close()
    for mod in ("ui_calibration_library.py", "calib_plan.py"):
        assert "ensure_schema(" not in (SRC / mod).read_text(encoding="utf-8"), mod


# ------------------------------------------------------------------ 7. RUN button gated by the run preconditions
def test_run_button_is_off_with_the_precondition_reasons_shown(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    import importlib.util

    import night_run

    spec = importlib.util.spec_from_file_location("s6ui_helpers", Path(__file__).with_name("test_step6_ui_2026_10.py"))
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    _app, _data_root = ui._app, ui._data_root

    _data_root(tmp_path, monkeypatch)
    src = tmp_path / "night"
    src.mkdir()
    calls = []

    def _pre(cfg, params, db):
        calls.append(params)
        return SimpleNamespace(problems=["Gaia DR3 database not usable (x): missing"], site=None)

    monkeypatch.setattr(night_run, "check_run_preconditions", _pre)
    at = _app(tmp_path)
    at.run()
    at.text_input(key="vyvar_varstrem_source").input(str(src)).run()
    assert not at.exception, at.exception
    assert all(b.disabled for b in at.button if "RUN VYVAR" in b.label)
    warn = "\n".join(str(w.value) for w in at.warning)
    assert "RUN VYVAR is off" in warn and "Gaia DR3 database not usable" in warn
    n = len(calls)
    at.run()
    assert len(calls) == n + 1  # one precondition check per render
    assert calls[-1].source_dir == src and calls[-1].location_source_hint == "ui_selection"
    monkeypatch.setattr(night_run, "check_run_preconditions", lambda cfg, params, db: SimpleNamespace(problems=[]))
    at.run()
    assert not any(b.disabled for b in at.button if "RUN VYVAR" in b.label)
    assert "RUN VYVAR is off" not in "\n".join(str(w.value) for w in at.warning)


# ------------------------------------------------------------------ 9. user docs of the Gaia DB check / repair
def test_gaia_repair_is_documented_for_users() -> None:
    docs = SRC.parent / "docs"
    for name, head in (("INSTALL.md", "Gaia DB incomplete"), ("README_FULL.md", "Gaia DB incomplete"),
                       ("README_CZ.md", "Gaia DB neuplna")):
        text = (docs / name).read_text(encoding="ascii")
        assert head in text, name
        for cmd in ("build_gaia_catalog.py --check", "build_gaia_catalog.py --repair --strip",
                    "gaia_maintain.py --report", ".venv\\Scripts\\python.exe", "GAIA_USER"):
            assert cmd in text, (name, cmd)
    from gaia_coverage import ESA_ANON_ASYNC_ROW_CAP

    assert f"({ESA_ANON_ASYNC_ROW_CAP} rows, truncated" in (docs / "INSTALL.md").read_text(encoding="ascii")
