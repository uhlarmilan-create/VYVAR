"""JOB-2026-10: RUN VYVAR as a separate process - job spec round trip and child runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

import night_run  # noqa: E402
import run_job  # noqa: E402


def _params(tmp_path: Path) -> night_run.NightRunParams:
    return night_run.NightRunParams(
        source_dir=tmp_path / "src",
        equipment_id=3,
        telescope_id=4,
        location_id=5,
        masterdark_validity_days=30,
        manual_flat_map={"a": "b"},
        flat_fallback_choices={"g": "o"},
        plate_fov_deg=1.2,
        dao_fwhm_px=2.5,
        masters_from="library",
        existing_pipeline=object(),
        optics=object(),
        progress_cb=lambda m: None,
    )


def test_job_round_trip_keeps_every_plain_field(tmp_path):
    p = _params(tmp_path)
    job = run_job.params_to_job(p)
    json.dumps(job)  # must be JSON-safe
    assert "existing_pipeline" not in job and "progress_cb" not in job and "optics" not in job
    back = run_job.job_to_params(json.loads(json.dumps(job)))
    for k, v in job.items():
        got = getattr(back, k)
        assert (str(got) if isinstance(got, Path) else got) == v, k
    assert back.existing_pipeline is None and back.progress_cb is None


def test_unknown_job_field_rejected(tmp_path):
    job = run_job.params_to_job(_params(tmp_path))
    job["no_such_field"] = 1
    try:
        run_job.job_to_params(job)
    except ValueError as exc:
        assert "no_such_field" in str(exc)
    else:
        raise AssertionError("unknown field accepted")


def test_child_runner_writes_status_and_result(tmp_path, monkeypatch):
    seen = {}

    def fake_run(params):
        params.progress_cb("Step 3: Calibration")
        seen["status_mid"] = run_job.read_status(tmp_path)
        seen["params"] = params
        return night_run.NightRunResult(success=True, draft_id=523, completed_setups=["g_120_4"])

    monkeypatch.setattr(night_run, "run_night_pipeline", fake_run)
    (tmp_path / run_job.JOB_FILE).write_text(json.dumps(run_job.params_to_job(_params(tmp_path))))
    run_job.write_status(tmp_path, state="starting", label="RUN VYVAR")
    rc = run_job.run_job_file(tmp_path / run_job.JOB_FILE)
    assert rc == 0
    assert seen["status_mid"]["state"] == "running"
    assert seen["status_mid"]["message"] == "Step 3: Calibration"
    assert seen["params"].masters_from == "library"
    st = run_job.read_status(tmp_path)
    assert st["state"] == "done" and st["draft_id"] == 523
    assert run_job.job_state(tmp_path) == "done"
    res = run_job.load_result(tmp_path)
    assert res.success and res.completed_setups == ["g_120_4"]


def test_child_runner_failure_is_reported(tmp_path, monkeypatch):
    def boom(params):
        raise RuntimeError("Astrometria: ziadny set")

    monkeypatch.setattr(night_run, "run_night_pipeline", boom)
    (tmp_path / run_job.JOB_FILE).write_text(json.dumps(run_job.params_to_job(_params(tmp_path))))
    assert run_job.run_job_file(tmp_path / run_job.JOB_FILE) == 1
    st = run_job.read_status(tmp_path)
    assert st["state"] == "failed" and "Astrometria" in st["message"]


def test_cli_job_flag_dispatches(tmp_path, monkeypatch):
    called = {}
    monkeypatch.setattr(run_job, "run_job_file", lambda p: called.setdefault("p", p) and 0)
    night_run.main(["--job", str(tmp_path / "job.json")])
    assert Path(called["p"]) == tmp_path / "job.json"
