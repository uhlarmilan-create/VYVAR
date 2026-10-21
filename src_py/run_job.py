"""RUN VYVAR as a separate process (JOB-2026-10).

The GUI writes a job spec (JSON) and starts ``python -m night_run --job <spec>``.
The child process runs exactly the same ``run_night_pipeline`` as the night
runner and reports progress into ``run_status.json`` next to the spec. The GUI
only reads that file, so a Streamlit rerun (any widget click) no longer stops
the run.

Files in the job directory:
  job.json          - serialisable NightRunParams fields
  run_status.json   - state running|done|failed, pid, process, message, timestamps
  result.pkl        - pickled NightRunResult (best effort, for the GUI session state)
  run_stdout.log    - stdout/stderr of the child process
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

JOB_FILE = "job.json"
STATUS_FILE = "run_status.json"
RESULT_FILE = "result.pkl"
STDOUT_FILE = "run_stdout.log"

#: NightRunParams fields that are plain data and can travel through JSON.
_PATH_FIELDS = {"source_dir", "config_path"}
_SKIP_FIELDS = {"existing_pipeline", "optics", "progress_cb", "post_platesolve_hook"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def params_to_job(params: Any) -> dict[str, Any]:
    """NightRunParams -> JSON-safe dict (callables / live objects dropped)."""
    out: dict[str, Any] = {}
    for f in fields(params):
        if f.name in _SKIP_FIELDS:
            continue
        v = getattr(params, f.name)
        if f.name in _PATH_FIELDS and v is not None:
            v = str(v)
        out[f.name] = v
    return out


def job_to_params(job: dict[str, Any], **extra: Any) -> Any:
    """Inverse of :func:`params_to_job`; unknown keys are rejected."""
    from night_run import NightRunParams  # noqa: PLC0415

    names = {f.name for f in fields(NightRunParams)}
    unknown = sorted(set(job) - names)
    if unknown:
        raise ValueError(f"job.json has unknown NightRunParams fields: {unknown}")
    kw = dict(job)
    for k in _PATH_FIELDS:
        if kw.get(k) is not None:
            kw[k] = Path(kw[k])
    kw.update(extra)
    return NightRunParams(**kw)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def write_status(job_dir: Path, **upd: Any) -> dict[str, Any]:
    p = Path(job_dir) / STATUS_FILE
    cur = read_status(job_dir) or {}
    cur.update(upd)
    cur["updated_utc"] = _now()
    _write_json_atomic(p, cur)
    return cur


def read_status(job_dir: Path | str | None) -> dict[str, Any] | None:
    if not job_dir:
        return None
    p = Path(job_dir) / STATUS_FILE
    for _ in range(3):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError):
            time.sleep(0.05)
    return None


def new_job_dir(data_root: Path | str) -> Path:
    d = Path(data_root) / "jobs" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def launch_job(params: Any, data_root: Path | str, label: str = "RUN VYVAR") -> Path:
    """Write the job spec and start the child process. Returns the job directory."""
    job_dir = new_job_dir(data_root)
    _write_json_atomic(job_dir / JOB_FILE, params_to_job(params))
    write_status(job_dir, state="starting", label=label, process=label, message="Starting...",
                 started_utc=_now())
    src_py = Path(__file__).resolve().parent
    cmd = [sys.executable, "-m", "night_run", "--job", str(job_dir / JOB_FILE)]
    kw: dict[str, Any] = {"cwd": str(src_py)}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kw["start_new_session"] = True
    out = open(job_dir / STDOUT_FILE, "w", encoding="utf-8")  # noqa: SIM115 - child owns it
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, **kw)
    write_status(job_dir, pid=int(proc.pid))
    return job_dir


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        import psutil  # noqa: PLC0415

        p = psutil.Process(int(pid))
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except Exception:  # noqa: BLE001
        return False


def job_state(job_dir: Path | str | None) -> str:
    """running | done | failed | lost | none (lost = child died without a final status)."""
    st = read_status(job_dir)
    if not st:
        return "none"
    s = str(st.get("state") or "")
    if s in ("done", "failed"):
        return s
    return "running" if pid_alive(st.get("pid")) else "lost"


def stop_job(job_dir: Path | str) -> bool:
    st = read_status(job_dir) or {}
    pid = st.get("pid")
    if not pid_alive(pid):
        return False
    try:
        import psutil  # noqa: PLC0415

        p = psutil.Process(int(pid))
        for c in p.children(recursive=True):
            c.terminate()
        p.terminate()
        write_status(Path(job_dir), state="failed", message="Stopped by user", finished_utc=_now())
        return True
    except Exception:  # noqa: BLE001
        return False


def load_result(job_dir: Path | str) -> Any | None:
    p = Path(job_dir) / RESULT_FILE
    if not p.is_file():
        return None
    try:
        with p.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 - written by our own child process
    except Exception:  # noqa: BLE001
        return None


def run_job_file(job_path: Path | str) -> int:
    """Child side: run the pipeline described by ``job.json``."""
    from night_run import run_night_pipeline  # noqa: PLC0415

    job_path = Path(job_path)
    job_dir = job_path.parent
    job = json.loads(job_path.read_text(encoding="utf-8"))
    label = (read_status(job_dir) or {}).get("label") or "RUN VYVAR"
    write_status(job_dir, state="running", pid=os.getpid(), message="Running...")

    def _progress(msg: str) -> None:
        try:
            write_status(job_dir, state="running", process=label, message=str(msg)[:800])
        except Exception:  # noqa: BLE001 - progress must never stop the run
            pass

    try:
        params = job_to_params(job, progress_cb=_progress)
        result = run_night_pipeline(params)
    except BaseException as exc:  # noqa: BLE001
        write_status(job_dir, state="failed", message=f"{type(exc).__name__}: {exc}",
                     errors=[f"{type(exc).__name__}: {exc}"], finished_utc=_now())
        return 1
    try:
        with (job_dir / RESULT_FILE).open("wb") as fh:
            pickle.dump(result, fh)
    except Exception as exc:  # noqa: BLE001
        write_status(job_dir, result_pickle_error=str(exc))
    write_status(
        job_dir,
        state="done" if result.success else "failed",
        message="Done" if result.success else "; ".join(result.errors or ["night run failed"])[:800],
        success=bool(result.success),
        draft_id=result.draft_id,
        draft_dir=str(result.draft_dir or ""),
        errors=list(result.errors or []),
        completed_setups=list(result.completed_setups or []),
        zero_target_setups=list(result.zero_target_setups or []),
        finished_utc=_now(),
    )
    return 0 if result.success else 1
