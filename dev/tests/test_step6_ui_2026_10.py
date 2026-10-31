"""VYVAR step 6, task s6_ui (T15 UI cache / state, manifest centre, ePSF as a job; G09-43, G14-19)."""
from __future__ import annotations

import ast
import itertools
import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
APP_SCRIPT = f"import sys\nsys.path.insert(0, {str(SRC)!r})\nimport app\napp.main()\n"


# ------------------------------------------------------------------------------------------ fixtures
def _data_root(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    """A release data dir in tmp (config, DB with two cameras / telescopes and one site, Archive)."""
    from vyvar_runtime import ensure_release_data_dir

    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("VYVAR_DATA_DIR", str(root))
    ensure_release_data_dir(REPO)
    from config import AppConfig
    from database import VyvarDatabase

    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    cams = [db.insert_equipment("CamA", "A", "MONO", "6248*4176", 3.76),
            db.insert_equipment("CamB", "B", "MONO", "6248*4176", 3.76)]
    tels = [db.insert_telescope("TelA", "a", 200.0, 800.0), db.insert_telescope("TelB", "b", 100.0, 500.0)]
    loc = db.insert_location("Jirny", 50.112, 14.698, 275.0)
    db.conn.close()
    return SimpleNamespace(root=root, cfg=cfg, cams=cams, tels=tels, loc=loc)


def _draft(archive: Path, name: str = "BO CVn", files: list[dict] | None = None) -> tuple[int, Path]:
    import draft_provenance as dp

    did = dp.create_draft_manifest(archive, {"object_name": name, "observation_start_jd": 2460970.4})
    d = dp.draft_dir_for_id(archive, did)
    (d / "platesolve").mkdir(parents=True, exist_ok=True)
    if files is not None:
        dp.patch_draft_manifest(d, did, files=files)
    return did, d


def _light(path: str, fwhm: float = 3.0, group: str = "V|60|1") -> dict:
    return {"imagetyp": "light", "file_path": path, "group_key": group, "filter": "V",
            "inspection": {"fwhm": fwhm, "sky_level": 100.0, "is_rejected": 0, "exptime": 60.0}}


def _masterstar(ps_setup: Path, ra: float, dec: float, nx: int = 1000, ny: int = 800, scale_deg: float = 4e-4) -> Path:
    from astropy.io import fits

    ps_setup.mkdir(parents=True, exist_ok=True)
    h = fits.Header({"CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CRVAL1": ra, "CRVAL2": dec,
                     "CRPIX1": (nx + 1) / 2.0, "CRPIX2": (ny + 1) / 2.0, "CDELT1": -scale_deg, "CDELT2": scale_deg})
    p = ps_setup / "MASTERSTAR.fits"
    fits.PrimaryHDU(np.zeros((ny, nx), dtype=np.float32), header=h).writeto(p)
    return p


def _src_app():
    """The real application module ``src_py/app.py`` (the repo-root ``app.py`` is a runpy shim that another test
    can import first under the same name)."""
    mod = sys.modules.get("app")
    if mod is not None and Path(getattr(mod, "__file__", "") or "").resolve() != (SRC / "app.py").resolve():
        del sys.modules["app"]
    if str(SRC) in sys.path:
        sys.path.remove(str(SRC))
    sys.path.insert(0, str(SRC))
    import app

    return app


def _app(tmp_path: Path):
    _src_app()
    from streamlit.testing.v1 import AppTest

    f = tmp_path / "app_main.py"
    f.write_text(APP_SCRIPT, encoding="utf-8")
    return AppTest.from_file(str(f), default_timeout=120)


def _texts(at) -> str:
    out = []
    for kind in ("info", "warning", "error", "success", "caption", "markdown"):
        out.extend(str(e.value) for e in getattr(at, kind))
    return "\n".join(out)


# ------------------------------------------------------------------- G14-03 / NEW-5 manifest centre
def test_g14_03_no_view_writes_the_manifest_on_render(tmp_path, monkeypatch):
    """Every view of the Pipeline page renders a loaded field (and a newly loaded one) without one
    manifest write; before, VAR-STREM persisted a centre on every new session / draft id."""
    import draft_provenance as dp

    env = _data_root(tmp_path, monkeypatch)
    d7, _ = _draft(env.cfg.archive_root, "AA Boo")
    d8, _ = _draft(env.cfg.archive_root, "BB Cnv")
    writes: list = []
    monkeypatch.setattr(dp, "_persist_manifest", lambda *a, **k: writes.append(a) or (_ for _ in ()).throw(
        AssertionError("manifest written on render")))
    at = _app(tmp_path)
    at.session_state["vyvar_last_draft_id"] = d7
    at.run()
    assert not at.exception, at.exception
    at.session_state["vyvar_last_draft_id"] = d8
    for view in ("VAR-STREM", "FITS QA", "MASTERSTAR QA", "Aperture Photometry", "[microscope] ePSF",
                 "[search] Variability", "VAR-STREM"):
        at.session_state["vyvar_main_view"] = view
        at.run()
        assert not at.exception, (view, at.exception)
    assert writes == []


def test_g14_03_no_ui_module_writes_the_centre():
    for f in [SRC / "app.py", *sorted(SRC.glob("ui_*.py"))]:
        s = f.read_text(encoding="utf-8")
        for bad in ("update_obs_draft_status_panel_values", "patch_draft_manifest", "vyvar_pending_center",
                    "cur_draft_ra", "DRAFT_CENTER_"):
            assert bad not in s, (f.name, bad)
    assert not (SRC / "ui_components.py").exists()


def test_g14_03_night_run_writes_the_masterstar_wcs_centre(tmp_path):
    """S6-FIX CENTRE-PER-SETUP: one centre per setup (each setup is its own instrument)."""
    from draft_provenance import load_draft_manifest
    from manifest_centre import CENTRE_SOURCE_MASTERSTAR_WCS, masterstar_wcs_centre, record_manifest_centres_from_masterstars

    did, d = _draft(tmp_path / "Archive")
    assert record_manifest_centres_from_masterstars(d, did) == {}  # no MASTERSTAR yet: nothing written
    assert "center_by_setup" not in load_draft_manifest(d)
    _masterstar(d / "platesolve" / "V_60_1", 201.0, 41.0)
    _masterstar(d / "platesolve" / "R_60_1", 150.0, 10.0)
    c = record_manifest_centres_from_masterstars(d, did)
    assert set(c) == {"R_60_1", "V_60_1"} and c["V_60_1"]["source"] == CENTRE_SOURCE_MASTERSTAR_WCS
    man = load_draft_manifest(d)["center_by_setup"]
    assert abs(man["R_60_1"]["ra_deg"] - 150.0) < 1e-3 and abs(man["R_60_1"]["de_deg"] - 10.0) < 1e-3
    assert abs(man["V_60_1"]["ra_deg"] - 201.0) < 1e-3 and abs(man["V_60_1"]["de_deg"] - 41.0) < 1e-3
    r = masterstar_wcs_centre(d / "platesolve" / "V_60_1" / "MASTERSTAR.fits")[2]
    assert abs(r - 0.5 * np.hypot(1000 * 4e-4, 800 * 4e-4)) < 1e-4
    nr = (SRC / "night_run.py").read_text(encoding="utf-8")
    i = nr.index("ps_out = _night_run_platesolve(")
    assert "record_manifest_centres_from_masterstars(ap, int(draft_id))" in nr[i:i + 600]


def test_g14_03_preprocess_hint_prefers_the_run_pointing_over_the_manifest_centre():
    from pipeline_preprocess import resolve_preprocess_target_coordinates

    class _Db:
        def __init__(self, rows):
            self.rows = rows

        def fetch_obs_draft_by_id(self, i):
            return {"CENTER_BY_SETUP": {"V_60_1": {"ra_deg": 30.0, "de_deg": 60.0}}}

        def fetch_draft_light_rows_for_quality(self, i):
            return self.rows

    db = _Db([{"RA": 201.0, "DE": 41.0}])
    assert resolve_preprocess_target_coordinates(db=db, draft_id=7, ui_ra_deg=200.5, ui_dec_deg=40.5) == (200.5, 40.5)
    assert resolve_preprocess_target_coordinates(db=db, draft_id=7, ui_ra_deg=None, ui_dec_deg=None) == (201.0, 41.0)
    # S6-FIX CENTRE-PER-SETUP: no field-level manifest centre; the per-setup centres are the QC pass fallback
    assert resolve_preprocess_target_coordinates(db=_Db([]), draft_id=7, ui_ra_deg=None, ui_dec_deg=None) == (None, None)


def test_g14_03_check_manifest_centres_tool(tmp_path, capsys):
    import importlib.util

    import draft_provenance as dp

    spec = importlib.util.spec_from_file_location("cmc", REPO / "dev" / "tools" / "check_manifest_centres.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    archive = tmp_path / "Archive"
    did, d = _draft(archive, "AA Boo")
    _masterstar(d / "platesolve" / "V_60_1", 201.0, 41.0)
    dp.patch_draft_manifest(d, did, center_by_setup={"V_60_1": {"ra_deg": 201.0, "de_deg": 41.0}})
    assert tool.main([str(archive / "Drafts")]) == 0
    # 6 arcmin; radius 15.4 arcmin
    dp.patch_draft_manifest(d, did, center_by_setup={"V_60_1": {"ra_deg": 201.0, "de_deg": 41.1}})
    assert tool.main([str(archive / "Drafts")]) == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out and "distance 6.00 arcmin" in out and "V_60_1" in out
    before = (d / "draft_manifest.json").read_bytes()
    tool.main([str(archive / "Drafts"), "--all"])
    assert (d / "draft_manifest.json").read_bytes() == before  # read-only


# ----------------------------------------------------------------------- G14-11 / G14-12 / G14-19
def test_g14_12_no_view_resolves_the_draft_folder_itself():
    for f in [SRC / "app.py", *sorted(SRC.glob("ui_*.py"))]:
        s = f.read_text(encoding="utf-8")
        assert "resolve_draft_dir(" not in s and "resolve_draft_dir_path" not in s, f.name
        assert "vyvar_last_job_output" not in s and "vyvar_post_cal_archive_path" not in s, f.name
        assert "vyvar_last_import_result" not in s, f.name


def test_g14_11_fits_qa_reads_the_loaded_folder_not_the_archive_folder_with_its_id(tmp_path, monkeypatch):
    """A field loaded by path from elsewhere whose id is also an (empty) archive field: FITS QA shows the
    loaded folder's frames."""
    import shutil

    env = _data_root(tmp_path, monkeypatch)
    did, d_arch = _draft(env.cfg.archive_root, "AA Boo")
    ext = tmp_path / "usb" / d_arch.name
    shutil.copytree(d_arch, ext)
    import draft_provenance as dp

    dp.patch_draft_manifest(ext, did, files=[_light(str(ext / "l1.fits")), _light(str(ext / "l2.fits"), 3.2)])
    at = _app(tmp_path)
    at.session_state["vyvar_draft_dir_override"] = str(ext)  # as the Load button sets them
    at.session_state["vyvar_last_draft_id"] = did
    at.session_state["vyvar_last_job_output"] = {"archive_path": str(d_arch)}  # an old session key: ignored
    at.session_state["vyvar_main_view"] = "FITS QA"
    at.run()
    assert not at.exception, at.exception
    assert "No light frames in manifest" not in _texts(at)
    assert int(at.dataframe[0].value["Frames"].iloc[0]) == 2  # the loaded folder's two frames


def test_g14_12_load_with_empty_field_clears_folder_and_id(tmp_path, monkeypatch):
    env = _data_root(tmp_path, monkeypatch)
    did, d = _draft(env.cfg.archive_root, "AA Boo")
    at = _app(tmp_path)
    at.run()
    at.text_input(key="vyvar_draft_path_field").input(str(did)).run()
    at.button(key="vyvar_draft_path_apply").click().run()
    assert at.session_state["vyvar_last_draft_id"] == did
    assert f"ID: **{did}**" in _texts(at)
    at.text_input(key="vyvar_draft_path_field").input("").run()
    at.button(key="vyvar_draft_path_apply").click().run()
    assert not at.exception, at.exception
    assert "vyvar_last_draft_id" not in at.session_state and "vyvar_draft_dir_override" not in at.session_state
    assert "ID: -" in _texts(at)


def test_g14_12_masterstar_qa_has_no_manual_path_input(tmp_path, monkeypatch):
    _data_root(tmp_path, monkeypatch)
    at = _app(tmp_path)
    at.session_state["vyvar_main_view"] = "MASTERSTAR QA"
    at.run()
    assert not at.exception, at.exception
    assert not [w for w in at.text_input if w.key == "vyvar_masterstar_ap"]
    assert "No processed field loaded" in _texts(at)


def test_g14_19_fits_qa_old_field_says_frame_qc_is_missing(tmp_path, monkeypatch):
    import frame_qc

    env = _data_root(tmp_path, monkeypatch)
    did, d = _draft(env.cfg.archive_root, "AA Boo", files=[_light("/x/l1.fits"), _light("/x/l2.fits", 3.3)])
    monkeypatch.setattr(frame_qc, "evaluate_frame_qc", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recompute")))
    at = _app(tmp_path)
    at.session_state["vyvar_last_draft_id"] = did
    at.session_state["vyvar_main_view"] = "FITS QA"
    at.run()
    assert not at.exception, at.exception
    t = _texts(at)
    assert "run's star lists are missing" in t and "computed here for display only" not in t


# -------------------------------------------------------------------- G14-04 / G09-43 / G14-14 jobs
def _fake_launch(monkeypatch, tmp_path):
    import task_jobs

    calls: list = []

    def _launch(task, args, data_root, *, label):
        calls.append((task, args, label))
        jd = tmp_path / f"job_{len(calls)}"
        jd.mkdir()
        return jd

    monkeypatch.setattr(task_jobs, "launch_task", _launch)
    return calls


def test_g14_04_run_epsf_starts_a_job_from_the_epsf_view(tmp_path, monkeypatch):
    import platesolve_ui_paths
    import ui_epsf_dashboard

    env = _data_root(tmp_path, monkeypatch)
    did, d = _draft(env.cfg.archive_root, "AA Boo")
    calls = _fake_launch(monkeypatch, tmp_path)
    fit = SimpleNamespace(target_ids=frozenset({"1"}))
    monkeypatch.setattr(platesolve_ui_paths, "find_photometry_setups",
                        lambda *a, **k: {"V_60_1": {"obs_group_dir": d / "platesolve" / "V_60_1"}})
    monkeypatch.setattr(ui_epsf_dashboard, "_render_run_plan",
                        lambda *a, **k: [SimpleNamespace(runnable=True, fit=fit, setup="V_60_1")])
    at = _app(tmp_path)
    at.session_state["vyvar_last_draft_id"] = did
    at.session_state["vyvar_main_view"] = "[microscope] ePSF"
    at.run()
    btn = next(b for b in at.button if b.label == "RUN ePSF photometry")
    btn.click().run()
    assert not at.exception, at.exception
    assert calls == [("run_epsf", {"draft_dir": str(d), "draft_id": did, "setups": ["V_60_1"]},
                      "RUN ePSF photometry")]
    assert at.session_state["vyvar_main_view"] == "[microscope] ePSF"  # no view switch needed
    assert at.session_state["vyvar_active_job"]["kind"] == "run_epsf"
    assert "vyvar_pending_job" not in at.session_state
    assert "_render_pending_job_dispatcher" not in (SRC / "app.py").read_text(encoding="utf-8")


def test_g09_43_epsf_job_uses_the_field_config_snapshot(tmp_path, monkeypatch):
    import epsf_stage
    import task_jobs

    env = _data_root(tmp_path, monkeypatch)
    did, d = _draft(env.cfg.archive_root, "AA Boo")
    ps = d / "platesolve" / "V_60_1"
    _masterstar(ps, 201.0, 41.0)
    (ps / "masterstars_full_match.csv").write_text("catalog_id\n1\n")
    (ps / "per_frame_catalog_index.csv").write_text("file\n")
    (ps / "photometry").mkdir()
    (ps / "photometry" / "pipeline_meta.json").write_text(json.dumps(
        {"provenance": {"config_snapshot": {"detection_threshold_sigma": 7.5}}}))
    (d / "detrended_aligned" / "lights" / "V_60_1").mkdir(parents=True)
    seen = {}

    def _stage(**kw):
        seen.update(kw)
        return {"lc": {"n_written": 2}, "merge": {"written": 5}}

    monkeypatch.setattr(epsf_stage, "run_epsf_stage", _stage)
    out = task_jobs.task_run_epsf({"draft_dir": str(d), "draft_id": did, "setups": ["V_60_1"]},
                                  lambda m: None)
    assert seen["cfg"].detection_threshold_sigma == 7.5  # the snapshot, not the live Settings
    assert "targets" not in seen and seen["draft_id"] == did
    assert seen["paths"].photometry_dir == ps / "photometry" and seen["paths"].frames_root == d / "detrended_aligned" / "lights" / "V_60_1"
    assert "detection_threshold_sigma" in out["cfg_changed_keys"] and out["n_lightcurves"] == 2


def test_g14_14_task_child_protocol(tmp_path, monkeypatch):
    import task_jobs
    from run_job import read_status

    monkeypatch.setitem(task_jobs.TASKS, "t_ok", lambda args, progress: (progress("half"), {"message": f"ok {args['x']}"})[1])
    monkeypatch.setitem(task_jobs.TASKS, "t_bad", lambda args, progress: 1 / 0)
    for task, code, state in (("t_ok", 0, "done"), ("t_bad", 1, "failed")):
        jd = tmp_path / task
        jd.mkdir()
        (jd / "job.json").write_text(json.dumps({"task": task, "args": {"x": 3}}))
        assert task_jobs.run_task_file(jd / "job.json") == code
        assert read_status(jd)["state"] == state
    assert task_jobs.load_task_result(tmp_path / "t_ok") == {"message": "ok 3"}
    assert "ZeroDivisionError" in read_status(tmp_path / "t_bad")["message"]
    assert {"run_epsf", "variability_search", "variability_catalog_check", "tess", "hrd_enrich",
            "build_masters"} <= set(task_jobs.TASKS)
    assert "import streamlit" not in (SRC / "task_jobs.py").read_text(encoding="utf-8")


def test_g14_14_build_masters_and_variability_search_start_jobs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    env = _data_root(tmp_path, monkeypatch)
    calls = _fake_launch(monkeypatch, tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    f = tmp_path / "clib.py"
    f.write_text(
        f"import sys\nfrom pathlib import Path\nsys.path.insert(0, {str(SRC)!r})\n"
        "from config import AppConfig\nfrom database import VyvarDatabase\nimport ui_calibration_library as u\n"
        "cfg = AppConfig()\n"
        "u.render_calibration_library_dashboard(calibration_library_root=Path(cfg.calibration_library_root), "
        "dark_validity_days=30, flat_validity_days=30, db=VyvarDatabase(cfg.database_path), data_root=cfg.data_root)\n")
    at = AppTest.from_file(str(f), default_timeout=60)
    at.run()
    at.text_input(key="clib_b_src").input(str(raw)).run()
    at.button(key="clib_b_go").click().run()
    assert not at.exception, at.exception
    assert calls[-1][0] == "build_masters" and calls[-1][1]["src"] == str(raw) and calls[-1][1]["what"] == "auto"

    did, d = _draft(env.cfg.archive_root, "AA Boo")
    phot = d / "platesolve" / "V_60_1" / "photometry"
    phot.mkdir(parents=True)
    (phot / "photometry_summary.csv").write_text("catalog_id\n1\n")
    at = _app(tmp_path)
    at.session_state["vyvar_last_draft_id"] = did
    at.session_state["vyvar_main_view"] = "[search] Variability"
    at.run()
    at.button(key="var_run_now").click().run()
    assert not at.exception, at.exception
    assert calls[-1][:2] == ("variability_search", {"draft_dir": str(d), "setup": "V_60_1"})


def test_g14_14_build_masters_from_folder_fails_loud(tmp_path):
    from task_jobs import build_masters_from_folder

    kw = dict(db=None, camera_id=1, telescope_id=1, library_root=tmp_path, method="median")
    with pytest.raises(FileNotFoundError):
        build_masters_from_folder(tmp_path / "nope", what="auto", **kw)
    with pytest.raises(ValueError):
        build_masters_from_folder(tmp_path, what="bias", **kw)


# ---------------------------------------------------------------------------------- G14-37 run lock
def test_g14_37_one_job_per_data_directory_across_sessions(tmp_path, monkeypatch):
    import run_job

    env = _data_root(tmp_path, monkeypatch)
    jd = env.root / "jobs" / "run_20261031_010101"
    jd.mkdir(parents=True)
    run_job.write_status(jd, state="running", pid=os.getpid(), label="RUN VYVAR")  # another session's job
    assert run_job.running_job(env.root) == jd
    monkeypatch.setattr(run_job.subprocess, "Popen", lambda *a, **k: pytest.fail("a second job was started"))
    with pytest.raises(run_job.JobBusyError):
        run_job.spawn_job(env.root, spec={}, module="run_job", label="RUN VYVAR", prefix="run")
    at = _app(tmp_path)
    at.run()
    assert not at.exception, at.exception
    assert "started in another browser session" in _texts(at)
    assert all(b.disabled for b in at.button if "RUN VYVAR" in b.label)
    run_job.write_status(jd, state="done")
    assert run_job.running_job(env.root) is None
    assert not (env.root / "jobs" / run_job.LAUNCH_MUTEX).exists()


# ------------------------------------------------------------------------ G14-06 rig / site / source
def test_g14_06_selection_survives_a_view_switch_and_reaches_the_run(tmp_path, monkeypatch):
    _src_app()
    import run_job

    env = _data_root(tmp_path, monkeypatch)
    src = tmp_path / "usb_night"
    src.mkdir()
    import night_run

    # the run's own precondition check (G12-15) is tested in test_step6_orch; here only the UI selection
    monkeypatch.setattr(night_run, "check_run_preconditions", lambda cfg, params, db: SimpleNamespace(problems=[]))
    seen = {}

    def _launch(params, data_root, label="RUN VYVAR"):
        seen["params"] = params
        jd = tmp_path / "runjob"
        jd.mkdir(exist_ok=True)
        return jd

    monkeypatch.setattr(run_job, "launch_job", _launch)
    at = _app(tmp_path)
    at.run()
    lbl_b = next(o for o in at.selectbox(key="vyvar_varstrem_equipment").options if "CamB" in o)
    tel_b = next(o for o in at.selectbox(key="vyvar_varstrem_telescope").options if "TelB" in o)
    at.selectbox(key="vyvar_varstrem_equipment").select(lbl_b).run()
    at.selectbox(key="vyvar_varstrem_telescope").select(tel_b).run()
    at.text_input(key="vyvar_varstrem_source").input(str(src)).run()
    at.session_state["vyvar_main_view"] = "FITS QA"
    at.run()
    at.session_state["vyvar_main_view"] = "VAR-STREM"
    at.run()
    assert not at.exception, at.exception
    assert at.selectbox(key="vyvar_varstrem_equipment").value == lbl_b
    assert at.selectbox(key="vyvar_varstrem_telescope").value == tel_b
    assert at.text_input(key="vyvar_varstrem_source").value == str(src)
    next(b for b in at.button if b.label == "! RUN VYVAR").click().run()
    assert not at.exception, at.exception
    p = seen["params"]
    assert (p.equipment_id, p.telescope_id, p.location_id) == (env.cams[1], env.tels[1], env.loc)
    assert Path(p.source_dir) == src


# ------------------------------------------------------------------------- G14-05 / G14-31 variability
def _var_draft(tmp_path: Path) -> Path:
    """The Variability tab fixture of test_var_tab_2026_10 (a draft with one candidate and TESS files)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_var_tab_fixture", Path(__file__).with_name("test_var_tab_2026_10.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._draft(tmp_path)


def _var_app(tmp_path: Path, draft: Path):
    from streamlit.testing.v1 import AppTest

    f = tmp_path / "var_app.py"
    f.write_text(f"import sys\nfrom pathlib import Path\nsys.path.insert(0, {str(SRC)!r})\n"
                 "from config import AppConfig\nimport ui_variability as uv\n"
                 f"uv.render_variability_dashboard(AppConfig(), draft_dir=Path({str(draft)!r}))\n")
    return AppTest.from_file(str(f), default_timeout=60)


def test_g14_05_periodogram_click_sets_the_period_without_crash(tmp_path):
    from streamlit.proto.WidgetStates_pb2 import WidgetState

    at = _var_app(tmp_path, _var_draft(tmp_path))
    at.run()
    at.session_state["var_sel_NoFilter_60_2"] = "1001"
    at.run()
    pid = next(e.proto.id for e in at.get("plotly_chart") if e.proto.id.endswith("-var_pg_1001"))
    ws = at._tree.get_widget_states()
    sel = {"selection": {"points": [{"x": 0.93, "y": 0.5}], "point_indices": [0], "box": [], "lasso": []}}
    ws.widgets.append(WidgetState(id=pid, json_value=json.dumps(json.dumps(sel))))  # a click on the chart
    at._run(widget_state=ws)
    assert not at.exception, [e.value for e in at.exception]
    pin = next(n for n in at.number_input if n.key == "var_pin_1001").value
    assert abs(pin - 0.4647) > 0.1 and abs(pin - 0.93) < 0.1  # the refined peak near the click


def test_g14_31_transient_table_without_in_galaxy_column(tmp_path):
    draft = _var_draft(tmp_path)
    phot = draft / "platesolve" / "NoFilter_60_2" / "photometry"
    (phot / "transient_meta.json").write_text(json.dumps({"status": "ok", "time_column": "bjd_tdb"}))
    pd.DataFrame({"candidate": [True], "label": ["T1"], "kind": ["appeared"], "first_time": [2461000.5],
                  "mag_g_est": [15.0], "snr_max": [12.0], "n_frames_snr5": [8], "ra_deg": [200.0],
                  "dec_deg": [40.0], "x": [100.0], "y": [100.0]}).to_csv(phot / "transient_candidates.csv", index=False)
    at = _var_app(tmp_path, draft)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert "New sources (transients) and minor bodies" in [h.value for h in at.subheader]


# --------------------------------------------------------------------------------- G03-03 autodetect
def _hdr(**kw):
    from astropy.io import fits

    return fits.Header(kw)


def test_g03_03_twin_cameras_mono_wins_in_every_order():
    from optics_autodetect import detect_equipment, detect_telescope

    h = _hdr(INSTRUME="QHY294PROM", SENSOR="MONO", NAXIS1=4164, NAXIS2=2796)
    mono = {"ID": 1, "CAMERANAME": "QHY294PRO M", "ALIAS": "m", "SENSORTYPE": "MONO", "SENSORSIZE": "4164*2796"}
    colour = {"ID": 2, "CAMERANAME": "QHY294PRO C", "ALIAS": "c", "SENSORTYPE": "COLOR", "SENSORSIZE": "4164*2796"}
    other = {"ID": 3, "CAMERANAME": "ASI1600", "ALIAS": "x", "SENSORTYPE": "MONO", "SENSORSIZE": "4656*3520"}
    for rows in itertools.permutations([mono, colour, other]):
        d = detect_equipment(h, list(rows))
        assert d.matched_id == 1 and d.ok and any("SENSORTYPE" in r for r in d.reasons), rows
    # equal raw scores of different rows: no auto-fill, the tie is named
    twin = dict(mono, ID=4, CAMERANAME="QHY294PRO M", ALIAS="m2")
    for rows in itertools.permutations([mono, twin]):
        d = detect_equipment(h, list(rows))
        assert d.matched_id is None and not d.ok and "tie" in d.reasons[0]
    ht = _hdr(FOCALLEN=800.0, APTDIA=200.0, TELESCOP="Newton 200")
    t1 = {"ID": 1, "TELESCOPENAME": "Newton 200", "ALIAS": "a", "DIAMETER": 200.0, "FOCAL": 800.0}
    t2 = {"ID": 2, "TELESCOPENAME": "Other", "ALIAS": "b", "DIAMETER": 200.0, "FOCAL": 800.0}
    assert {detect_telescope(ht, list(r)).matched_id for r in itertools.permutations([t1, t2])} == {1}
    # S6-FIX SITE-MANUAL-2026-10: the site is never auto-detected (detect_location removed)


# ------------------------------------------------------------------------ G04-34 Settings MASTERSTAR
def test_g04_34_suggestions_use_the_masterstar_of_the_selected_setup(tmp_path):
    from masterstar_context import resolve_masterstar_fits_path

    import draft_provenance as dp

    did, d = _draft(tmp_path / "Archive")
    light = d / "calibrated" / "lights" / "V_60_1" / "l1.fits"
    light.parent.mkdir(parents=True)
    light.write_bytes(b"x")
    dp.patch_draft_manifest(d, did, paths={"masterstar": str(light)})
    m_v = _masterstar(d / "platesolve" / "V_60_1", 1.0, 2.0)
    m_r = _masterstar(d / "platesolve" / "R_60_1", 1.0, 2.0)
    assert resolve_masterstar_fits_path(d, "V_60_1") == m_v.resolve()
    assert resolve_masterstar_fits_path(d, "R_60_1") == m_r.resolve()
    assert resolve_masterstar_fits_path(d, "B_60_1") is None


# ----------------------------------------------------------------------------- G14-09 cached pipeline
def test_g14_09_settings_save_closes_the_previous_connection(tmp_path, monkeypatch):
    import app

    env = _data_root(tmp_path, monkeypatch)
    app._cached_astro_pipeline.clear()
    p1 = app._cached_astro_pipeline(str(env.cfg.database_path), "a", env.cfg)
    assert app._cached_astro_pipeline(str(env.cfg.database_path), "a", env.cfg) is p1
    p2 = app._cached_astro_pipeline(str(env.cfg.database_path), "b", env.cfg)
    assert p2.config is env.cfg  # the run's AppConfig, not a second one
    with pytest.raises(sqlite3.ProgrammingError):
        p1.db.conn.execute("SELECT 1")
    p2.db.conn.execute("SELECT 1")
    app._cached_astro_pipeline.clear()


# ------------------------------------------------------------------- NEW-1 / NEW-2 / NEW-3 caches
def test_new1_hrd_render_reads_the_enrichment_cache_only(tmp_path, monkeypatch):
    import hrd_enrich

    monkeypatch.setattr(hrd_enrich, "_fetch_gaia_tap_with_retry", lambda *a, **k: pytest.fail("network on render"))
    monkeypatch.setattr(hrd_enrich, "_fetch_simbad", lambda *a, **k: pytest.fail("network on render"))
    cand = pd.DataFrame({"catalog_id": ["123456789012"], "teff_gspphot": [5000.0], "logg_gspphot": [4.0]})
    cache = tmp_path / "_hrd_cache" / "hrd_enrich.json"
    out = hrd_enrich.enrich_candidates(cand, cache, fetch=False)
    assert not cache.exists() and out["enrich_source"].tolist() == ["local"]
    cache.parent.mkdir()
    hrd_enrich._save_cache(cache, {"123456789012": {"simbad_main_id": "HD 1", "enrich_source": "gaia_tap"}}, last_enrich={})
    before = cache.read_bytes()
    out = hrd_enrich.enrich_candidates(cand, cache, fetch=False)
    assert out["simbad_main_id"].tolist() == ["HD 1"] and cache.read_bytes() == before
    import ui_hrd

    ui_src = (SRC / "ui_hrd.py").read_text(encoding="utf-8")
    assert "fetch_online=False" in ui_src
    ps = tmp_path / "platesolve" / "V_60_1"
    assert ps not in ui_hrd._render_cache_dir(ps).parents  # render cache outside the processed field


def test_new2_new3_view_caches_key_on_every_file_and_the_config(tmp_path):
    from config import AppConfig
    from ui_aperture_photometry import _pages_stamp
    from ui_cache import config_fingerprint

    phot = tmp_path / "photometry"
    (phot / "lightcurves").mkdir(parents=True)
    (phot / "photometry_summary.csv").write_text("catalog_id\n1\n")
    s0 = _pages_stamp(phot)
    (phot / "lightcurves_reports").mkdir()
    (phot / "lightcurves_reports" / "export_manifest.json").write_text("{}")  # written after the summary
    s1 = _pages_stamp(phot)
    (phot / "lightcurves" / "lightcurve_1.csv").write_text("x")
    assert len({s0, s1, _pages_stamp(phot)}) == 3
    cfg = AppConfig()
    f0 = config_fingerprint(cfg)
    cfg.hrd_online_enrich_enabled = not bool(getattr(cfg, "hrd_online_enrich_enabled", True))
    assert config_fingerprint(cfg) != f0
    for name, fn, arg in (("ui_aperture_photometry.py", "_group_pages", "cfg_key"), ("ui_hrd.py", "_hrd_data", "cfg_key")):
        tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn)
        assert arg in [a.arg for a in node.args.args], (name, fn)


# ------------------------------------------------------------------------------------- NEW-4 PTC
def test_new4_ptc_has_no_assumed_saturation(tmp_path, monkeypatch):
    import inspect

    from streamlit.testing.v1 import AppTest

    import gain_ptc

    assert inspect.signature(gain_ptc.measure_from_files).parameters["saturation_adu"].default is inspect.Parameter.empty
    assert "65535" not in (SRC / "ui_calibration_library.py").read_text(encoding="utf-8")
    _data_root(tmp_path, monkeypatch)
    f = tmp_path / "clib.py"
    f.write_text(
        f"import sys\nfrom pathlib import Path\nsys.path.insert(0, {str(SRC)!r})\n"
        "from config import AppConfig\nfrom database import VyvarDatabase\nimport ui_calibration_library as u\n"
        "cfg = AppConfig()\n"
        "u.render_calibration_library_dashboard(calibration_library_root=Path(cfg.calibration_library_root), "
        "dark_validity_days=30, flat_validity_days=30, db=VyvarDatabase(cfg.database_path), data_root=cfg.data_root)\n")
    at = AppTest.from_file(str(f), default_timeout=60)
    at.run()
    at.text_input(key="clib_g_src").input(str(tmp_path)).run()
    assert not at.exception, at.exception
    sat = [n for n in at.number_input if n.label.startswith("Saturation level of this camera")]
    assert sat and sat[0].value is None
    assert at.button(key="clib_g_go").disabled  # asked, never assumed
    sat[0].set_value(60000.0).run()
    assert not at.button(key="clib_g_go").disabled


# ------------------------------------------------------------------------ G14-36 pure view helpers
def test_g14_36_equipment_detector_warnings():
    from ui_database_explorer import (
        GAIN_PLAUSIBLE_MAX_E_PER_ADU,
        READ_NOISE_PLAUSIBLE_MAX_E,
        equipment_detector_warnings,
    )

    rows = [{"ID": 1, "CAMERANAME": "ok", "GAIN": 0.8, "RN": 3.0},
            {"ID": 2, "CAMERANAME": "nogain", "GAIN": None, "RN": 3.0},
            {"ID": 3, "CAMERANAME": "nan", "GAIN": float("nan"), "RN": 3.0},
            {"ID": 4, "CAMERANAME": "hig", "GAIN": GAIN_PLAUSIBLE_MAX_E_PER_ADU + 1, "RN": 3.0},
            {"ID": 5, "CAMERANAME": "hirn", "GAIN": 1.0, "RN": READ_NOISE_PLAUSIBLE_MAX_E + 1},
            {"ID": 6, "ALIAS": "edge", "GAIN": GAIN_PLAUSIBLE_MAX_E_PER_ADU, "RN": READ_NOISE_PLAUSIBLE_MAX_E}]
    w = equipment_detector_warnings(rows)
    assert len(w) == 4
    assert w[0].startswith("nogain: gain or read noise unknown") and w[1].startswith("nan: gain or read noise unknown")
    assert "gain 51 e-/ADU is unusually high" in w[2] and "read noise 51 e- is unusually high" in w[3]


def test_g14_36_load_measurement_quality(tmp_path):
    from ui_photometry_quality import load_measurement_quality

    q = load_measurement_quality(tmp_path)
    assert q.stars.empty and "No masterstars_full_match.csv" in q.problems[0]
    pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3], "is_saturated": [True, False, False], "is_noisy": [False, True, False],
                  "is_usable": [False, False, True], "saturate_limit_adu": [60000, 60000, 60000],
                  "catalog_id": ["1", "2", "3"]}).to_csv(tmp_path / "masterstars_full_match.csv", index=False)
    pd.DataFrame({"catalog_id": ["3"]}).to_csv(tmp_path / "comparison_stars.csv", index=False)
    pd.DataFrame({"file": ["b", "a", "c"], "n_detected": [100, 100, 100], "n_matched": [80, 100, 10]}).to_csv(
        tmp_path / "per_frame_catalog_index.csv", index=False)
    q = load_measurement_quality(tmp_path)
    assert q.zone_counts["saturated"] == 1 and q.zone_counts["noise"] == 1 and q.zone_counts["linear"] == 1
    assert q.n_comp_pool == 1 and q.sat_limit_adu == 60000 and q.median_matched == 80
    assert q.frames["file"].tolist() == ["a", "b", "c"] and q.frames["low"].tolist() == [False, False, True]
    assert q.problems == []


def test_g14_36_fits_qa_tables():
    from ui_quality_dashboard import build_qc_summary_table, build_quality_frame

    rows = [dict(FILE_PATH="/x/a.fits", FWHM=3.0, SKY_LEVEL=100.0, IS_REJECTED=0, OBSERVATION_GROUP_KEY="V|60|1",
                 FILTER="V", EXPTIME=60.0),
            dict(FILE_PATH="/x/b.fits", FWHM=6.0, SKY_LEVEL=110.0, IS_REJECTED=1, QC_REASON="fwhm (6.0 > 4.0)",
                 OBSERVATION_GROUP_KEY="V|60|1", FILTER="V", EXPTIME=60.0)]
    df = build_quality_frame(rows, None)
    assert df["frame_index"].tolist() == [1, 2] and df["_name"].tolist() == ["a.fits", "b.fits"]
    assert df["_preview"].tolist() == ["/x/a.fits", "/x/b.fits"]
    t = build_qc_summary_table(df, {df["GROUP"].iloc[0]: {"best_path": "/x/a.fits", "frames": {}}})
    r = t.iloc[0]
    assert (r["Frames"], r["Kept"], r["Rejected"], r["MASTERSTAR source"]) == (2, 1, "1x fwhm", "a.fits")
    assert r["FWHM px (median, range)"].startswith("3.00")
