"""V12-STATUS (2026-11-05): RUN VYVAR status bar - phase k/M, frame i/N <file>, percent within the
phase, one bar, and an end state (DONE / FAILED / STOPPED / LOST) that stays until the next job."""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))

import night_run  # noqa: E402
import run_job  # noqa: E402
from night_run import RunProgress, _RunStages, night_run_stage_plan  # noqa: E402


def _load_test_module(name: str):
    spec = importlib.util.spec_from_file_location(f"_v12s_{name}", Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ night_run: stage plan and events
def _stages():
    events: list[RunProgress] = []
    return _RunStages({}, events.append), events


def test_phase_k_of_m_follows_the_runs_stage_plan_calibrated():
    st, ev = _stages()
    st.set_plan(night_run_stage_plan(False), final=False)
    st.stage(night_run.PH_SCAN)
    # Scan, Import, Calibration, Frame QC, MASTERSTAR ref, Preprocess, Plate solve, Discover = 8 certain;
    # RAM QC is decided after calibration, the setups after discovery -> not final
    assert (ev[-1].phase, ev[-1].phase_index, ev[-1].phase_count, ev[-1].phase_count_final) == (
        "Scan source", 1, 8, False)
    st.stage(night_run.PH_IMPORT)
    st.stage(night_run.PH_CALIBRATION)
    st.frame(2, 4, "Calibrating light_0002.fits")
    e = ev[-1]
    assert (e.phase_index, e.frame_i, e.frame_n, e.file, e.pct) == (3, 2, 4, "light_0002.fits", 50)
    assert e.message == "[2/4] Calibrating light_0002.fits"
    st.stage(night_run.PH_FRAME_QC)  # PERF-10: RAM QC not needed -> left out, M unchanged
    assert (ev[-1].phase_index, ev[-1].phase_count, ev[-1].frame_i, ev[-1].pct) == (4, 8, None, None)
    for ph in (night_run.PH_MS_REF, night_run.PH_PREPROCESS, night_run.PH_PLATESOLVE, night_run.PH_DISCOVER):
        st.stage(ph)
    st.set_plan([(night_run.ph_photometry("V_60_1"), True), (night_run.ph_photometry("B_60_1"), True)], final=True)
    st.stage(night_run.ph_photometry("V_60_1"))
    assert (ev[-1].phase_index, ev[-1].phase_count, ev[-1].phase_count_final) == (9, 10, True)


def test_ram_qc_decided_to_run_and_pre_calibrated_plan():
    st, ev = _stages()
    st.set_plan(night_run_stage_plan(False), final=False)
    for ph in (night_run.PH_SCAN, night_run.PH_IMPORT, night_run.PH_CALIBRATION, night_run.PH_RAM_QC):
        st.stage(ph)
    assert (ev[-1].phase_index, ev[-1].phase_count) == (4, 9)
    st2, ev2 = _stages()
    st2.set_plan(night_run_stage_plan(True), final=False)
    st2.stage(night_run.PH_SCAN)
    assert ev2[-1].phase_count == 8  # no Calibration, RAM QC certain
    assert night_run.PH_CALIBRATION not in [n for n, _c in night_run_stage_plan(True)]


def test_unplanned_phase_is_counted_with_a_warning(caplog):
    st, ev = _stages()
    st.set_plan([("A", True), ("B", True)], final=True)
    st.stage("A")
    with caplog.at_level("WARNING"):
        st.stage("X")
    assert "not in the run's stage plan" in caplog.text
    assert (ev[-1].phase, ev[-1].phase_index, ev[-1].phase_count) == ("X", 2, 3)


def test_photometry_phases_planned_from_the_setups_with_inputs(tmp_path, monkeypatch):
    ep = _load_test_module("test_epsf_chain_01b")
    spy = ep._stub_night_photometry(tmp_path, monkeypatch)
    import platesolve_ui_paths

    one = platesolve_ui_paths.find_photometry_setups()
    two = dict(one, S2={"obs_group_dir": str(tmp_path / "nope")})  # S2: inputs missing -> not run
    monkeypatch.setattr(platesolve_ui_paths, "find_photometry_setups", lambda *a, **k: two)
    (spy["draft"] / "detrended_aligned" / "lights" / "S2").mkdir()
    (spy["draft"] / "detrended_aligned" / "lights" / "S2" / "proc_001.csv").write_text("x\n1\n")
    st, ev = _stages()
    out = night_run.run_night_photometry(
        cfg=spy["cfg"], pipeline=spy["pipeline"], draft_id=1, draft_dir_override=spy["draft"],
        write_pdfs=False, existing_draft=True, epsf=True, stages=st,
    )
    assert any("S2: missing" in e for e in out["errors"])
    phases = [(e.phase, e.phase_index, e.phase_count, e.phase_count_final) for e in ev if e.message == e.phase]
    assert phases == [
        ("Discover observation groups", 1, 1, False),
        ("Photometry - S1", 2, 3, True),
        ("ePSF photometry - S1", 3, 3, True),
    ]


# ---------------------------------------------------------------- run_job: run_status.json fields
def _params(tmp_path: Path) -> night_run.NightRunParams:
    return night_run.NightRunParams(source_dir=tmp_path / "src", equipment_id=3, telescope_id=4, location_id=5)


def test_job_status_carries_phase_frame_file_pct_and_throttles_frames(tmp_path, monkeypatch):
    seen = {}
    real_write = run_job.write_status
    writes = []

    def _count(job_dir, **upd):
        writes.append(dict(upd))
        return real_write(job_dir, **upd)

    def fake_run(params):
        st = _RunStages({}, params.progress_cb)
        st.set_plan(night_run_stage_plan(False), final=False)
        st.stage(night_run.PH_SCAN)
        st.stage(night_run.PH_IMPORT)
        st.stage(night_run.PH_CALIBRATION)
        seen["mid"] = run_job.read_status(tmp_path)
        n0 = len(writes)
        for i in range(1, 201):
            st.frame(i, 200, f"Calibrating f{i:04d}.fits")
        seen["frame_writes"] = len(writes) - n0
        seen["end"] = run_job.read_status(tmp_path)
        return night_run.NightRunResult(success=True)

    monkeypatch.setattr(night_run, "run_night_pipeline", fake_run)
    monkeypatch.setattr(run_job, "write_status", _count)
    (tmp_path / run_job.JOB_FILE).write_text(json.dumps(run_job.params_to_job(_params(tmp_path))))
    real_write(tmp_path, state="starting", label="RUN VYVAR")
    assert run_job.run_job_file(tmp_path / run_job.JOB_FILE) == 0
    mid = seen["mid"]
    assert (mid["phase"], mid["phase_index"], mid["phase_count"], mid["phase_count_final"]) == (
        "Calibration", 3, 8, False)
    assert (mid["frame_i"], mid["pct"], mid["message"]) == (None, None, "Calibration")
    # 200 frames in well under FRAME_STATUS_MIN_INTERVAL_S after the phase start: only the last one
    assert seen["frame_writes"] == 1
    end = seen["end"]
    assert (end["frame_i"], end["pct"], end["file"], end["message"]) == (200, 100, "f0200.fits",
                                                                         "[200/200] Calibrating f0200.fits")
    assert run_job.FRAME_STATUS_MIN_INTERVAL_S == run_job.MONITOR_POLL_S / 2
    assert run_job.read_status(tmp_path)["state"] == "done"


def test_status_write_failure_is_a_warning_not_silent(tmp_path, monkeypatch, caplog):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(run_job, "write_status", boom)
    w = run_job.status_progress_writer(tmp_path, "RUN VYVAR")
    with caplog.at_level("WARNING"):
        w(RunProgress(message="x"))
    assert "status update" in caplog.text and "disk full" in caplog.text


# --------------------------------------------------------------------------------- GUI status bar
def _footers(at) -> list[str]:
    return [str(m.value) for m in at.markdown if "vyvar-footer-bar" in str(m.value)]


def _running_status(jd: Path) -> None:
    run_job.write_status(
        jd, state="running", pid=os.getpid(), label="RUN VYVAR", message="[2/4] Calibrating light_0002.fits",
        phase="Calibration", phase_index=3, phase_count=8, phase_count_final=False,
        frame_i=2, frame_n=4, file="light_0002.fits", pct=50,
    )


def test_run_click_follows_the_job_at_once_one_bar_then_done(tmp_path, monkeypatch):
    ui = _load_test_module("test_step6_ui_2026_10")
    ui._src_app()
    env = ui._data_root(tmp_path, monkeypatch)
    src = tmp_path / "usb_night"
    src.mkdir()
    monkeypatch.setattr(night_run, "check_run_preconditions", lambda cfg, params, db: SimpleNamespace(problems=[]))
    jd = env.root / "jobs" / "run_20261105_010101"

    def _launch(params, data_root, label="RUN VYVAR"):
        jd.mkdir(parents=True, exist_ok=True)
        _running_status(jd)
        return jd

    monkeypatch.setattr(run_job, "launch_job", _launch)
    at = ui._app(tmp_path)
    at.run()
    at.text_input(key="vyvar_varstrem_source").input(str(src)).run()
    next(b for b in at.button if b.label == "! RUN VYVAR").click().run()
    assert not at.exception, at.exception
    bars = _footers(at)
    assert len(bars) == 1, bars  # one status bar (the running fragment drew a second one before)
    assert "Phase 3/8+ Calibration" in bars[0]
    assert "frame 2/4 light_0002.fits" in bars[0] and "50 %" in bars[0]
    # the job ends: the bar switches to DONE and stays there
    with (jd / run_job.RESULT_FILE).open("wb") as fh:
        pickle.dump(night_run.NightRunResult(success=True), fh)
    run_job.write_status(jd, state="done", message="Done")
    at.run()
    assert not at.exception, at.exception
    bars = _footers(at)
    assert len(bars) == 1 and "DONE" in bars[0] and "Done. [OK]" in bars[0], bars
    at.run()
    assert "DONE" in _footers(at)[0]


def test_failed_and_stopped_jobs_end_with_their_state(tmp_path, monkeypatch):
    ui = _load_test_module("test_step6_ui_2026_10")
    env = ui._data_root(tmp_path, monkeypatch)
    jd = env.root / "jobs" / "run_20261105_020202"
    jd.mkdir(parents=True)
    _running_status(jd)
    at = ui._app(tmp_path)
    at.session_state["vyvar_active_job"] = {"job_dir": str(jd), "label": "RUN VYVAR", "kind": "night",
                                            "equipment_id": 1, "applied": False}
    at.run()
    assert len(_footers(at)) == 1, _footers(at)  # the running fragment drew a second bar over the app's
    assert "Phase 3/8+" in _footers(at)[0]
    run_job.write_status(jd, state="failed", message="Platesolve failed: no stars", errors=["Platesolve failed: no stars"])
    at.run()
    assert not at.exception, at.exception
    bar = _footers(at)[0]
    assert "FAILED" in bar and "Platesolve failed: no stars" in bar, bar

    jd2 = env.root / "jobs" / "run_20261105_030303"
    jd2.mkdir(parents=True)
    _running_status(jd2)
    at.session_state["vyvar_active_job"] = {"job_dir": str(jd2), "label": "RUN VYVAR", "kind": "night",
                                            "equipment_id": 1, "applied": False}
    at.run()
    run_job.write_status(jd2, state="failed", message="Stopped by user")
    (jd2 / run_job.STOP_MARKER).write_text("x")
    at.run()
    assert "STOPPED" in _footers(at)[0]


def test_job_of_another_session_shows_its_progress_and_frees_the_buttons(tmp_path, monkeypatch):
    ui = _load_test_module("test_step6_ui_2026_10")
    env = ui._data_root(tmp_path, monkeypatch)
    jd = env.root / "jobs" / "cli_20261105_040404"
    jd.mkdir(parents=True)
    _running_status(jd)
    at = ui._app(tmp_path)
    at.run()
    assert not at.exception, at.exception
    bars = _footers(at)
    assert len(bars) == 1 and "Phase 3/8+ Calibration" in bars[0], bars
    assert "started in another browser session" in ui._texts(at)
    run_job.write_status(jd, state="done", message="Done")
    at.run()
    assert not at.exception, at.exception
    assert "DONE" in _footers(at)[0]
    assert "started in another browser session" not in ui._texts(at)
