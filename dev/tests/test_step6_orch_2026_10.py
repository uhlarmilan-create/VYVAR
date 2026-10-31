# -*- coding: ascii -*-
"""VYVAR step 6, s6_orch (T14): orchestration, logging, invariants, GUI-vs-night parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src_py"


# ---------------------------------------------------------------------------------------------
# G12-02 / N1 / N2: INV-CAL-01 finds cal_diag.json at the field root; PASSTHROUGH = no frame darked
# ---------------------------------------------------------------------------------------------
def _field(tmp_path: Path, draft_id: int = 7) -> Path:
    f = tmp_path / f"draft_{draft_id:06d}"
    f.mkdir(parents=True)
    (f / "draft_manifest.json").write_text(json.dumps({"draft_id": draft_id}), encoding="utf-8")
    return f


def test_g12_02_cal_diag_json_at_field_root_is_found(tmp_path: Path) -> None:
    from invariants_runtime import check_cal_diag

    f = _field(tmp_path)
    (f / "cal_diag.json").write_text("{}", encoding="utf-8")
    phot = f / "platesolve" / "V_60_1" / "photometry"
    phot.mkdir(parents=True)
    meta = {"calibration_mode": "vyvar_calibrated"}
    check_cal_diag(meta, photometry_dir=phot)  # old code: InvariantViolation INV-CAL-01
    rec = [r for r in meta["invariants"] if r["id"] == "INV-CAL-01"]
    assert rec and rec[-1]["ok"] is True


@pytest.mark.parametrize(
    ("flags", "mode"),
    [
        ({"P": 4}, "PASSTHROUGH"),
        ({"FS": 4}, "PASSTHROUGH"),
        ({"P": 2, "FS": 2}, "PASSTHROUGH"),
        ({"D": 4}, "vyvar_calibrated"),
        ({"DF": 4}, "vyvar_calibrated"),
    ],
)
def test_n1_passthrough_when_no_frame_got_a_dark(tmp_path: Path, flags: dict, mode: str) -> None:
    from draft_provenance import load_draft_manifest, record_calibration_outcome
    from invariants_runtime import check_cal_diag

    f = _field(tmp_path)
    record_calibration_outcome(f, draft_id=7, flags_by_group={"V_60_1": flags})
    got = load_draft_manifest(f)["calibration_mode"]
    assert got == mode
    if mode == "PASSTHROUGH":
        meta = {"calibration_mode": got}
        check_cal_diag(meta, photometry_dir=f / "platesolve" / "V_60_1" / "photometry")
        assert meta["invariants"][-1]["ok"] is True


def test_n2_recalibration_with_darks_resets_passthrough(tmp_path: Path) -> None:
    from draft_provenance import load_draft_manifest, record_calibration_outcome

    f = _field(tmp_path)
    record_calibration_outcome(f, draft_id=7, flags_by_group={"V_60_1": {"P": 4}})
    record_calibration_outcome(f, draft_id=7, flags_by_group={"V_60_1": {"DF": 4}})
    assert load_draft_manifest(f)["calibration_mode"] == "vyvar_calibrated"


# ---------------------------------------------------------------------------------------------
# G12-27: INV-DAG-01 kept; re-running a stage clears the later stamps; never hidden at DEBUG
# ---------------------------------------------------------------------------------------------
def test_g12_27_photometry_rerun_on_existing_field_passes(tmp_path: Path) -> None:
    from invariants_runtime import load_pipeline_meta, stamp_pipeline_stage, stamp_stage_on_disk

    phot = tmp_path / "photometry"
    for name in ("masterstar", "target_comp_selection", "aperture_photometry", "postprocess"):
        stamp_stage_on_disk(phot, name)
    # photometry re-run (old code: INV-DAG-01 'goes backwards' on target_comp_selection)
    stamp_stage_on_disk(phot, "target_comp_selection")
    stamp_stage_on_disk(phot, "aperture_photometry")
    stamp_stage_on_disk(phot, "postprocess")
    stages = load_pipeline_meta(phot)["stages"]
    assert [s["name"] for s in stages] == ["masterstar", "target_comp_selection", "aperture_photometry", "postprocess"]
    assert not any(s["gap"] or s["cold_start"] for s in stages)
    assert stages[1]["rerun_cleared"] == ["target_comp_selection", "aperture_photometry", "postprocess"]
    meta = load_pipeline_meta(phot)
    stamp_pipeline_stage(meta, "postprocess")  # the same stage again: replaced, not duplicated
    assert [s["name"] for s in meta["stages"]].count("postprocess") == 1


def test_g12_27_stamp_failures_are_not_swallowed_at_debug() -> None:
    for mod in ("photometry_core.py", "masterstar_build.py"):
        text = (SRC / mod).read_text(encoding="utf-8")
        i = text.index("stamp_stage_on_disk(")
        assert "logging.debug" not in text[i - 400 : i + 400], mod
    text = (SRC / "invariants_runtime.py").read_text(encoding="utf-8")
    assert "postprocess stamp skipped" not in text


# ---------------------------------------------------------------------------------------------
# G12-30 / N3 / G12-48: EXCEPT-FIX counters once per run, merged from workers, no dead counters
# ---------------------------------------------------------------------------------------------
def _bump_counter(n: int) -> int:
    """Pool worker (module level for spawn): count one importer failure per item."""
    from except_fix_counters import get_except_fix_counters

    get_except_fix_counters().importer_filter_read_fail += 1
    return n


def test_g12_30_worker_counts_reach_the_run_process() -> None:
    from except_fix_counters import get_except_fix_counters, reset_except_fix_counters
    from parallel_frames import map_frames

    reset_except_fix_counters()
    assert list(map_frames(_bump_counter, [1, 2, 3, 4], workers=2, stage="test")) == [1, 2, 3, 4]
    assert get_except_fix_counters().importer_filter_read_fail == 4


def test_g12_30_counts_before_photometry_are_kept() -> None:
    from except_fix_counters import get_except_fix_counters, reset_except_fix_counters

    text = (SRC / "aperture_photometry.py").read_text(encoding="utf-8")
    assert "reset_except_fix_counters()" not in text  # old code wiped import/calibration counts here
    nr = (SRC / "night_run.py").read_text(encoding="utf-8")
    assert nr.count("reset_except_fix_counters()") == 1
    reset_except_fix_counters()
    get_except_fix_counters().importer_filter_read_fail += 1
    assert get_except_fix_counters().snapshot()["importer_filter_read_fail"] == 1


def test_n3_g12_48_every_counter_has_an_increment_site_and_is_in_snapshot() -> None:
    import re
    from dataclasses import fields

    from except_fix_counters import ExceptFixCounters

    names = [f.name for f in fields(ExceptFixCounters)]
    assert set(ExceptFixCounters().snapshot()) == set(names)
    code = "\n".join(p.read_text(encoding="utf-8") for p in SRC.glob("*.py") if p.name != "except_fix_counters.py")
    dead = [n for n in names if not re.search(rf"\.{n}\s*\+=", code)]
    assert dead == []


# ---------------------------------------------------------------------------------------------
# G12-07 / G12-08 / G12-09 / G12-34: one logging setup per run, lines kept until the infolog opens
# ---------------------------------------------------------------------------------------------
def _infolog_text(draft: Path) -> str:
    return next(draft.glob("infolog_*.txt")).read_text(encoding="utf-8")


def test_g12_07_info_of_every_vyvar_module_reaches_the_infolog(tmp_path: Path, capsys) -> None:
    import logging

    from infolog import end_infolog_session, setup_run_logging, start_infolog_session

    teardown = setup_run_logging()
    try:
        logging.getLogger("night_run").info("[Progress] nr-info-before-open")  # before the file opens
        start_infolog_session(tmp_path / "d")
        logging.getLogger("night_run").info("nr-after-open")
        logging.getLogger("aperture_photometry").warning("ap-warning")
        logging.getLogger("night_run").debug("nr-debug")
        logging.getLogger("matplotlib.font_manager").info("lib-info")
        logging.getLogger("matplotlib.font_manager").warning("lib-warning")
        end_infolog_session()
    finally:
        teardown()
    text = _infolog_text(tmp_path / "d")
    for needle in ("nr-info-before-open", "nr-after-open", "ap-warning", "lib-warning"):
        assert text.count(needle) == 1, needle
    assert "nr-debug" not in text and "lib-info" not in text
    out = capsys.readouterr().out
    assert "[Progress] nr-info-before-open" in out and "ap-warning" in out


def test_g12_07_setup_is_shared_and_removed_after_the_run() -> None:
    import logging

    from infolog import InfologHandler, setup_run_logging

    n0 = len(logging.getLogger().handlers)
    t1 = setup_run_logging()
    t2 = setup_run_logging()  # nested (CLI main + run_night_pipeline): no second handler
    assert sum(isinstance(h, InfologHandler) for h in logging.getLogger().handlers) == 1
    t2()
    t1()
    assert len(logging.getLogger().handlers) == n0
    for name in ("night_run.py", "run_job.py"):
        assert "ensure_infolog_logging" not in (SRC / name).read_text(encoding="utf-8")


def test_g12_08_log_event_before_the_infolog_opens_is_written(tmp_path: Path) -> None:
    from infolog import end_infolog_session, log_event, start_infolog_session

    log_event("EVENT-before-session")
    start_infolog_session(tmp_path / "d")
    end_infolog_session()
    assert "EVENT-before-session" in _infolog_text(tmp_path / "d")


def test_g12_09_lines_of_a_run_without_infolog_do_not_reach_the_next_run(tmp_path: Path) -> None:
    from infolog import end_infolog_session, log_event, log_milestone, start_infolog_session

    log_milestone("old-milestone")
    log_event("old-event")
    end_infolog_session()  # the run ends before its infolog opened
    start_infolog_session(tmp_path / "d")
    end_infolog_session()
    text = _infolog_text(tmp_path / "d")
    assert "old-milestone" not in text and "old-event" not in text


def test_g12_34_one_stage_helper_discover_started_once(tmp_path: Path, monkeypatch) -> None:
    import inspect

    import night_run
    import platesolve_ui_paths
    from infolog import end_infolog_session, start_infolog_session

    src = inspect.getsource(night_run.run_night_pipeline)
    assert 'Discover observation groups' not in src and "def _stage" not in src and "def _t(" not in src
    assert "def _stage" not in inspect.getsource(night_run.run_night_photometry)
    monkeypatch.setattr(platesolve_ui_paths, "find_photometry_setups", lambda *a, **k: {})
    timings: dict = {}
    stages = night_run._RunStages(timings, None)
    start_infolog_session(tmp_path / "d")
    out = night_run.run_night_photometry(
        cfg=object(), pipeline=None, draft_id=None, draft_dir_override=tmp_path / "d",
        timings=timings, stages=stages,
    )
    end_infolog_session()
    assert out["errors"]
    text = _infolog_text(tmp_path / "d")
    assert text.count("[PHASE] Discover observation groups start") == 1
    assert "[PHASE] discover_obs_groups done" in text and "discover_obs_groups" in timings


# ---------------------------------------------------------------------------------------------
# G12-10 / G12-11 / G14-07 / G12-15 / G12-16 / N4 / N5 / G12-29: GUI and CLI build the same run
# ---------------------------------------------------------------------------------------------
def _gaia_db(path: Path) -> Path:
    import sqlite3

    con = sqlite3.connect(path)
    con.execute("CREATE TABLE gaia_dr3 (source_id INTEGER, ra REAL, dec REAL, g_mag REAL, bp_rp REAL)")
    con.commit()
    con.close()
    return path


@pytest.fixture
def rig(tmp_path_is_data_root: Path) -> dict:
    """Data root (VYVAR_DATA_DIR) with config.json, a database with camera / telescope / site, a
    Gaia DB with the required schema and a source folder."""
    from database import VyvarDatabase

    root = tmp_path_is_data_root
    gaia = _gaia_db(root / "gaia.sqlite")
    (root / "config.json").write_text(json.dumps({"gaia_db_path": str(gaia)}), encoding="ascii")
    db = VyvarDatabase(root / "vyvar.sqlite3")
    eq = db.insert_equipment(camera_name="QHY600", alias="Cam", sensor_type="IMX571",
                             sensor_size="9576*6388", pixel_size=3.76)
    tel = int(db.conn.execute(
        "INSERT INTO TELESCOPE (TELESCOPENAME, ALIAS, DIAMETER, FOCAL, ACTIVE) VALUES (?, ?, ?, ?, 'YES');",
        ("RC12", "RC", 300.0, 2400.0)).lastrowid)
    loc = db.insert_location(place_name="Observatory", latitude=50.1, longitude=14.4, altitude=400.0)
    db.conn.commit()
    db.close()
    src = root / "night"
    (src / "lights").mkdir(parents=True)
    return {"root": root, "eq": int(eq), "tel": tel, "loc": int(loc), "src": src, "gaia": gaia}


def test_g12_10_params_have_no_live_objects_and_round_trip_entirely() -> None:
    from dataclasses import fields

    import run_job
    from night_run import NightRunParams

    names = {f.name for f in fields(NightRunParams)}
    for dead in ("existing_pipeline", "optics", "platesolve_equipment_id", "post_platesolve_hook",
                 "manual_flat_map", "flat_fallback_choices", "apply_smart_plan_flat_fallbacks",
                 "masterdark_validity_days", "masterflat_validity_days", "dao_threshold_sigma"):
        assert dead not in names, dead
    p = NightRunParams(source_dir=Path("/x"), equipment_id=1, telescope_id=2)
    assert set(run_job.params_to_job(p)) == names - {"progress_cb"}
    assert "_SKIP_FIELDS" not in (SRC / "run_job.py").read_text(encoding="utf-8")


def _gui_params(rig: dict, monkeypatch) -> tuple:
    import importlib.util

    import run_job
    from config import AppConfig
    from pipeline import AstroPipeline

    seen: dict = {}

    def fake_launch(params, data_root, label="RUN VYVAR"):
        seen["params"] = params
        return Path(data_root) / "jobs" / "run_test"

    monkeypatch.setattr(run_job, "launch_job", fake_launch)
    # src_py/app.py itself (``import app`` would find the root shim, which runs the app)
    spec = importlib.util.spec_from_file_location("vyvar_src_app", SRC / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    cfg = AppConfig()
    pipeline = AstroPipeline(cfg)
    cfg.observer_lat = 1.25  # sentinel: the old GUI wrote the resolved site (lat 50.1) into its cfg
    loc_before = (cfg.observer_location_id, cfg.observer_lat)
    started = app._run_vyvar_full_pipeline(
        pipeline=pipeline, cfg=cfg, source_root=str(rig["src"]), import_equipment_id=rig["eq"],
        import_telescope_id=rig["tel"], import_location_id=rig["loc"],
    )
    pipeline.db.close()
    return started, seen.get("params"), cfg, loc_before


def test_g12_11_gui_and_cli_build_identical_params(rig: dict, monkeypatch) -> None:
    import night_run
    import run_job

    started, gui, cfg, loc_before = _gui_params(rig, monkeypatch)
    assert started and gui is not None
    seen: dict = {}

    def fake_run(params, cfg=None):
        seen["params"] = params
        return night_run.NightRunResult(success=True)

    monkeypatch.setattr(night_run, "run_night_pipeline", fake_run)
    rc = night_run.main(["--source", str(rig["src"]), "--camera", str(rig["eq"]),
                         "--telescope", str(rig["tel"]), "--site", str(rig["loc"])])
    assert rc == 0
    a, b = run_job.params_to_job(gui), run_job.params_to_job(seen["params"])
    assert a.pop("location_source_hint") == "ui_selection" and b.pop("location_source_hint") == "cli_arg"
    assert a == b
    # G12-16: the GUI validated the site without writing it into its (cached) cfg
    assert (cfg.observer_location_id, cfg.observer_lat) == loc_before


def test_g12_15_gui_refuses_a_run_the_night_run_would_refuse(rig: dict, monkeypatch) -> None:
    (rig["root"] / "config.json").write_text(json.dumps({"gaia_db_path": str(rig["root"] / "none.sqlite")}),
                                             encoding="ascii")
    started, params, _cfg, _ = _gui_params(rig, monkeypatch)
    assert started is False and params is None


def test_g12_15_n5_cli_without_gaia_db_exits_2_before_import(rig: dict, capsys) -> None:
    import night_run
    from config import AppConfig

    (rig["root"] / "config.json").write_text(json.dumps({"gaia_db_path": str(rig["root"] / "none.sqlite")}),
                                             encoding="ascii")
    rc = night_run.main(["--source", str(rig["src"]), "--camera", str(rig["eq"]),
                         "--telescope", str(rig["tel"]), "--site", str(rig["loc"])])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Gaia DR3 database not usable" in err and "Preflight log:" in err
    drafts = AppConfig().archive_root / "Drafts"
    assert not drafts.is_dir() or not any(drafts.iterdir())
    logs = list((rig["root"] / "logs").glob("run_preflight_error_*.log"))
    assert len(logs) == 1 and "Gaia DR3 database not usable" in logs[0].read_text(encoding="utf-8")


def test_g12_15_unknown_camera_id_refused(rig: dict) -> None:
    from config import AppConfig
    from database import VyvarDatabase
    from night_run import NightRunParams, check_run_preconditions

    db = VyvarDatabase(rig["root"] / "vyvar.sqlite3")
    try:
        p = NightRunParams(source_dir=rig["src"], equipment_id=999, telescope_id=rig["tel"], location_id=rig["loc"])
        pre = check_run_preconditions(AppConfig(), p, db)
    finally:
        db.close()
    assert any("camera id=999" in m for m in pre.problems)


def test_n4_cli_builds_the_config_once(rig: dict, monkeypatch) -> None:
    import night_run

    calls = []
    real = night_run._load_app_config
    monkeypatch.setattr(night_run, "_load_app_config", lambda p: calls.append(p) or real(p))
    (rig["root"] / "config.json").write_text(json.dumps({"gaia_db_path": ""}), encoding="ascii")
    night_run.main(["--source", str(rig["src"]), "--camera", str(rig["eq"]),
                    "--telescope", str(rig["tel"]), "--site", str(rig["loc"])])
    assert len(calls) == 1


def test_n5_job_child_crash_writes_the_preflight_log(rig: dict, monkeypatch) -> None:
    import night_run
    import run_job

    def boom(params, cfg=None):
        raise RuntimeError("child boom")

    monkeypatch.setattr(night_run, "run_night_pipeline", boom)
    job_dir = rig["root"] / "jobs" / "j1"
    job_dir.mkdir(parents=True)
    p = night_run.NightRunParams(source_dir=rig["src"], equipment_id=rig["eq"], telescope_id=rig["tel"])
    (job_dir / run_job.JOB_FILE).write_text(json.dumps(run_job.params_to_job(p)), encoding="utf-8")
    assert run_job.run_job_file(job_dir / run_job.JOB_FILE) == 1
    st = run_job.read_status(job_dir)
    assert st["preflight_log"] and "child boom" in Path(st["preflight_log"]).read_text(encoding="utf-8")


def test_g12_07_fatal_traceback_goes_into_the_infolog(rig: dict, monkeypatch) -> None:
    import night_run
    from infolog import start_infolog_session

    draft = rig["root"] / "Archive" / "Drafts" / "draft_000001"

    def fake_scan(**kw):
        return False, object()

    def fake_import(**kw):
        return kw["site"], type("I", (), {"draft_id": 1})()

    def fake_calibrate(**kw):
        start_infolog_session(draft)
        raise RuntimeError("calibration boom")

    monkeypatch.setattr(night_run, "_nr_scan_source", fake_scan)
    monkeypatch.setattr(night_run, "_nr_import", fake_import)
    monkeypatch.setattr(night_run, "_nr_calibrate", fake_calibrate)
    res = night_run.run_night_pipeline(night_run.NightRunParams(
        source_dir=rig["src"], equipment_id=rig["eq"], telescope_id=rig["tel"], location_id=rig["loc"]))
    assert not res.success and res.failure_log is None  # the field has its infolog
    text = _infolog_text(draft)
    assert "FATAL: calibration boom" in text and "Traceback (most recent call last)" in text
    assert "RUN VYVAR (camera / telescope): camera id=" in text  # G12-08: optics line from the run


def test_g12_29_bootstrap_used_by_cli_and_db_at_database_path(tmp_path: Path, monkeypatch) -> None:
    import night_run
    import vyvar_runtime

    install = tmp_path / "install"
    install.mkdir()
    (install / "RUNTIME_PIN.json").write_text("{}", encoding="ascii")  # bundled layout
    data = tmp_path / "data"
    data.mkdir()
    (data / "config.json").write_text(json.dumps({"database_path": str(tmp_path / "db" / "other.sqlite3")}),
                                      encoding="ascii")
    monkeypatch.setenv("VYVAR_DATA_DIR", str(data))
    report_root, report = vyvar_runtime.bootstrap_release_data_dir(install)
    assert report["vyvar.sqlite3"] == "created"
    assert (tmp_path / "db" / "other.sqlite3").is_file() and not (data / "vyvar.sqlite3").exists()
    assert (data / "Archive" / "Drafts").is_dir()
    calls = []
    monkeypatch.setattr(vyvar_runtime, "ensure_data_dir_for_entry", lambda: calls.append(1))
    night_run.main(["--source", str(tmp_path), "--camera", "1", "--telescope", "1"])  # exits on the missing ids
    assert calls == [1]
    assert "ensure_data_dir_for_entry()" in (SRC / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# G12-53 / G12-54: version in one place; purge test gaps
# ---------------------------------------------------------------------------------------------
def test_g12_53_pyproject_version_is_the_code_version() -> None:
    import tomllib

    from vyvar_version import __version__

    data = tomllib.loads((SRC.parent / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == __version__


def _purge_field(tmp_path: Path, source: str) -> Path:
    import numpy as np
    from astropy.io import fits

    d = _field(tmp_path, draft_id=1)
    for rel in (f"{source}/NoFilter_60_2/Light_001.fits", "calibrated/lights/NoFilter_60_2/Light_001.fits"):
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        hdu = fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32))
        hdu.header["VYCALSTG"] = "DF"
        hdu.writeto(p)
    return d


def test_g12_54_purge_accepts_lowercase_raw_lights(tmp_path: Path) -> None:
    from draft_provenance import purge_calibrated_light_frames

    d = _purge_field(tmp_path, "raw/lights")
    out = purge_calibrated_light_frames(d, draft_id=1)
    assert out["n_files"] == 1 and out["skipped"] == ""
    assert (d / "raw" / "lights" / "NoFilter_60_2" / "Light_001.fits").is_file()


def test_g12_54_photometry_rerun_after_purge_passes_inv_cal_02(tmp_path: Path) -> None:
    from draft_provenance import purge_calibrated_light_frames
    from invariants_runtime import check_cal_stage

    d = _purge_field(tmp_path, "Raw/lights")
    purge_calibrated_light_frames(d, draft_id=1)
    meta: dict = {}
    check_cal_stage(meta, photometry_dir=d / "platesolve" / "NoFilter_60_2" / "photometry")
    rec = meta["invariants"][-1]
    assert rec["id"] == "INV-CAL-02" and rec["ok"] is True and rec["policy"] == "WARN"
