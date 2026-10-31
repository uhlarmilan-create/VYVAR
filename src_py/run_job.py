"""RUN VYVAR as a separate process (JOB-2026-10).

The GUI writes a job spec (JSON) and starts ``python -m run_job --job <spec>``.
The child process runs exactly the same ``run_night_pipeline`` as the night
runner and reports progress into ``run_status.json`` next to the spec. The GUI
only reads that file, so a Streamlit rerun (any widget click) no longer stops
the run.

Files in the job directory:
  job.json          - serialisable NightRunParams fields
  run_status.json   - state running|done|failed, pid, process, message, timestamps
  result.pkl        - pickled NightRunResult (best effort, for the GUI session state)
  run_stdout.log    - stdout/stderr of the child process

G12-25: the child is started as ``-m run_job`` (was ``-m night_run``). Each spawned worker process
imports the main module again as ``__mp_main__``; this module imports only the standard library
and ``astro_env`` (the astropy IERS settings every process of the run had through night_run), not
the whole night_run graph (~1700 modules, ~150 MB per worker). ``night_run`` is imported by the
child when the job runs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import logging
import subprocess
import sys
import threading
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import astro_env  # noqa: F401  (IERS-OFF-2026-10, in the job process and its spawned workers)

JOB_FILE = "job.json"
STATUS_FILE = "run_status.json"
RESULT_FILE = "result.pkl"
STDOUT_FILE = "run_stdout.log"
#: G12-39: written by ``stop_job``; the child never overwrites the stopped state afterwards.
STOP_MARKER = "stopped"
#: G12-39: lock file serialising the read-modify-write of ``run_status.json`` (parent and child).
_STATUS_LOCK = "run_status.lock"
#: Longest a status update waits for the lock; a lock older than this is a crashed writer's and is
#: broken with a WARNING (a status write itself takes milliseconds).
_STATUS_LOCK_TIMEOUT_S = 10.0
#: G05-17: a job in state "starting" counts as running for this long without a live pid (time from
#: the "starting" status to the child's pid written after Popen; process start takes < 1 s, slow
#: machines and virus scanners a few seconds).
SPAWN_TIMEOUT_S = 60.0

LOGGER = logging.getLogger(__name__)

#: NightRunParams fields that are plain data and can travel through JSON.
_PATH_FIELDS = {"source_dir", "config_path"}
#: the one NightRunParams field that is not data (G12-10: no live objects in the params any more)
_CALLBACK_FIELD = "progress_cb"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def params_to_job(params: Any) -> dict[str, Any]:
    """NightRunParams -> JSON-safe dict (the progress callback dropped)."""
    out: dict[str, Any] = {}
    for f in fields(params):
        if f.name == _CALLBACK_FIELD:
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
    """G12-39: temp name per process and thread (parent and child never share a temp file)."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class _StatusLock:
    """Exclusive lock file (O_CREAT | O_EXCL) around one status read-modify-write."""

    def __init__(self, job_dir: Path) -> None:
        self._path = Path(job_dir) / _STATUS_LOCK

    def __enter__(self) -> _StatusLock:
        t0 = time.monotonic()
        while True:
            try:
                os.close(os.open(self._path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                if time.monotonic() - t0 > _STATUS_LOCK_TIMEOUT_S:
                    LOGGER.warning("run_job: status lock %s older than %.0f s broken", self._path,
                                   _STATUS_LOCK_TIMEOUT_S)
                    self._path.unlink(missing_ok=True)
                    t0 = time.monotonic()
                    continue
                time.sleep(0.005)

    def __exit__(self, *_exc: Any) -> None:
        self._path.unlink(missing_ok=True)


def is_stopped(job_dir: Path | str) -> bool:
    return (Path(job_dir) / STOP_MARKER).exists()


def write_status(job_dir: Path, **upd: Any) -> dict[str, Any]:
    """Merge ``upd`` into ``run_status.json`` (locked read-modify-write, atomic replace).

    G12-39: after ``stop_job`` the state / message are frozen; later updates of a still running
    child keep only their other fields."""
    job_dir = Path(job_dir)
    with _StatusLock(job_dir):
        return _write_status_locked(job_dir, upd, honour_stop=True)


def _write_status_locked(job_dir: Path, upd: dict[str, Any], *, honour_stop: bool) -> dict[str, Any]:
    cur = read_status(job_dir) or {}
    upd = dict(upd)
    if honour_stop and is_stopped(job_dir):
        for k in ("state", "message", "finished_utc"):
            upd.pop(k, None)
    cur.update(upd)
    cur["updated_utc"] = _now()
    _write_json_atomic(job_dir / STATUS_FILE, cur)
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


def unique_job_dir(parent: Path | str, stem: str) -> Path:
    """G12-41: create ``parent/stem`` or, when it exists, ``stem_2``, ``stem_3`` ... (claimed by
    ``mkdir(exist_ok=False)``, so two jobs started in the same second never share a directory)."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        d = parent / (stem if n == 1 else f"{stem}_{n}")
        try:
            d.mkdir(exist_ok=False)
            return d
        except FileExistsError:
            n += 1


def new_job_dir(data_root: Path | str, prefix: str = "run") -> Path:
    return unique_job_dir(Path(data_root) / "jobs", datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S"))


class JobBusyError(RuntimeError):
    """Another job of this data directory is running (G14-37: one job at a time per data directory,
    across every browser session)."""


LAUNCH_MUTEX = ".launch.lock"
#: A launch mutex older than this is left over from a crashed launcher and is removed.
_LAUNCH_MUTEX_STALE_S = 30.0


def running_job(data_root: Path | str) -> Path | None:
    """The job directory of a running job of this data directory, None when none runs."""
    jobs = Path(data_root) / "jobs"
    if not jobs.is_dir():
        return None
    for d in sorted(jobs.iterdir(), reverse=True):
        if d.is_dir() and (d / STATUS_FILE).is_file() and job_state(d) == "running":
            return d
    return None


@contextlib.contextmanager
def _launch_mutex(data_root: Path | str, timeout_s: float = 10.0):
    """Serialise "is a job running? -> start one" across browser sessions (O_EXCL lock file)."""
    jobs = Path(data_root) / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    lock = jobs / LAUNCH_MUTEX
    t_end = time.time() + timeout_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > _LAUNCH_MUTEX_STALE_S:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() > t_end:
                raise JobBusyError(f"another job is being started ({lock})") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def spawn_job(data_root: Path | str, *, spec: dict[str, Any], module: str, label: str, prefix: str) -> Path:
    """Write ``job.json`` and start ``python -m <module> --job job.json``; returns the job directory.

    Raises :class:`JobBusyError` when a job of this data directory is running (G14-37)."""
    with _launch_mutex(data_root):
        busy = running_job(data_root)
        if busy is not None:
            lbl = (read_status(busy) or {}).get("label") or busy.name
            raise JobBusyError(f"{lbl} is running (job {busy.name}); one job at a time per data directory")
        job_dir = new_job_dir(data_root, prefix)
        _write_json_atomic(job_dir / JOB_FILE, spec)
        write_status(job_dir, state="starting", label=label, process=label, message="Starting...",
                     started_utc=_now())
        src_py = Path(__file__).resolve().parent
        cmd = [sys.executable, "-m", module, "--job", str(job_dir / JOB_FILE)]
        kw: dict[str, Any] = {"cwd": str(src_py)}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kw["start_new_session"] = True
        with open(job_dir / STDOUT_FILE, "w", encoding="utf-8") as out:  # G12-40: the child has its own handle
            proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, **kw)
        write_status(job_dir, pid=int(proc.pid))
    return job_dir


def register_external_job(data_root: Path | str, *, spec: dict[str, Any], label: str, prefix: str) -> Path:
    """S6-FIX RUN-LOCK-2026-10: a run started outside :func:`spawn_job` (the CLI night run) takes the same
    one-job-per-data-directory lock: under the launch mutex it raises :class:`JobBusyError` while a job of
    the data directory runs, else it registers this process as a running job (``pid`` = this process), so the
    GUI shows it and keeps its start buttons off. A registration whose process died is ``lost``
    (:func:`job_state`), never running, so it does not block. :func:`finish_job` writes the end state."""
    with _launch_mutex(data_root):
        busy = running_job(data_root)
        if busy is not None:
            lbl = (read_status(busy) or {}).get("label") or busy.name
            raise JobBusyError(f"{lbl} is running (job {busy.name}); one job at a time per data directory")
        job_dir = new_job_dir(data_root, prefix)
        _write_json_atomic(job_dir / JOB_FILE, spec)
        write_status(job_dir, state="running", label=label, process=label, message="Running...",
                     pid=os.getpid(), started_utc=_now())
    return job_dir


def finish_job(job_dir: Path | str, result: Any) -> None:
    """Final ``run_status.json`` of a night-run job from its ``NightRunResult`` (GUI child and CLI)."""
    write_status(
        Path(job_dir),
        state="done" if result.success else "failed",
        message="Done" if result.success else "; ".join(result.errors or ["night run failed"])[:800],
        success=bool(result.success),
        draft_id=result.draft_id,
        draft_dir=str(result.draft_dir or ""),
        errors=list(result.errors or []),
        completed_setups=list(result.completed_setups or []),
        zero_target_setups=list(result.zero_target_setups or []),
        preflight_log=result.failure_log,
        finished_utc=_now(),
    )


def fail_job(job_dir: Path | str, exc: BaseException, preflight_log: str | None) -> None:
    """Final ``run_status.json`` of a job whose run raised."""
    write_status(Path(job_dir), state="failed", message=f"{type(exc).__name__}: {exc}",
                 errors=[f"{type(exc).__name__}: {exc}"], preflight_log=preflight_log, finished_utc=_now())


def launch_job(params: Any, data_root: Path | str, label: str = "RUN VYVAR") -> Path:
    """Write the job spec and start the child process. Returns the job directory."""
    return spawn_job(data_root, spec=params_to_job(params), module="run_job", label=label, prefix="run")


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
    if pid_alive(st.get("pid")):
        return "running"
    if s == "starting" and st.get("pid") is None and _age_s(st.get("started_utc")) < SPAWN_TIMEOUT_S:
        return "running"  # G05-17: spawned, pid not written yet
    return "lost"


def _age_s(utc_iso: Any) -> float:
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(str(utc_iso))).total_seconds()
    except (TypeError, ValueError):
        return float("inf")


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
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("run_job: stop of pid %s failed: %s", pid, exc)
        return False
    # G12-39: the process is stopped; recording it is a separate step that cannot undo that
    job_dir = Path(job_dir)
    try:
        with _StatusLock(job_dir):
            _write_status_locked(job_dir, {"state": "failed", "message": "Stopped by user",
                                           "finished_utc": _now()}, honour_stop=False)
            (job_dir / STOP_MARKER).write_text(_now() + "\n", encoding="utf-8")
    except OSError as exc:
        LOGGER.warning("run_job: job %s stopped, but its status was not written: %s", job_dir, exc)
    return True


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
        from night_run import _write_failure_log  # noqa: PLC0415

        # N5: the same preflight / failure log as the GUI and the CLI
        fail_job(job_dir, exc, _write_failure_log(None, step="RUN VYVAR job", exc=exc))
        return 1
    try:
        with (job_dir / RESULT_FILE).open("wb") as fh:
            pickle.dump(result, fh)
    except Exception as exc:  # noqa: BLE001
        write_status(job_dir, result_pickle_error=str(exc))
    finish_job(job_dir, result)
    return 0 if result.success else 1


def main(argv: list[str] | None = None) -> int:
    """Child entry of a RUN VYVAR job (``python -m run_job --job <job.json>``)."""
    ap = argparse.ArgumentParser(prog="run_job", description="Run one RUN VYVAR job spec.")
    ap.add_argument("--job", type=Path, required=True, help="job.json written by launch_job")
    return run_job_file(ap.parse_args(argv).job)


if __name__ == "__main__":
    raise SystemExit(main())
