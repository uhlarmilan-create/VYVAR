"""Catalog builder jobs started from Settings -> Paths and catalogs (CATALOG-WIZARD-2026-10).

The scripts in ``catalogs/`` run as separate processes, like RUN VYVAR (``run_job``): a Streamlit
rerun or closing the browser tab does not stop a build that takes hours. One job directory per
start under ``<data>/jobs/catalog_<tool>_<time>/``:

  job.json         - tool, script, arguments
  run_status.json  - state starting|running|done|failed, pid, return code, timestamps
  run_stdout.log   - everything the script prints

The child is ``python -c 'import catalog_jobs ...' --run <job dir>``: it runs the script with the same
Python, waits for it and writes the final state with the return code. The GUI reads the status
file and the end of the log; progress comes from the lines the scripts already print.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_constants import GAIA_BUILD_MAG_LIMIT_DEFAULT, VSX_BUILD_MAG_LIMIT_DEFAULT

from run_job import STATUS_FILE, STDOUT_FILE, read_status, stop_job, write_status
from run_job import job_state as _run_job_state

JOB_FILE = "job.json"

#: tool -> (script in catalogs/, catalog it writes, short label)
TOOLS: dict[str, tuple[str, str, str]] = {
    "gaia_build": ("build_gaia_catalog.py", "gaia", "Build Gaia DR3 catalog"),
    "gaia_report": ("gaia_maintain.py", "gaia", "Gaia DB report"),
    "gaia_register": ("gaia_maintain.py", "gaia", "Register Gaia DB (catalog_meta)"),
    "gaia_drop_unused": ("gaia_maintain.py", "gaia", "Drop unused Gaia indexes"),
    "objects_update": ("objects_make.py", "objects", "Update VSX + exoplanets"),
    "astro_index": ("build_astro_index.py", "astro_index", "Build blind plate-solve index"),
}

#: progress patterns of the scripts' own output -> fraction done
_P_GAIA = re.compile(r"^\[\s*(\d+)/(\d+)\]")                      # build_gaia_catalog: [  12/800] RA ...
_P_VSX = re.compile(r"^\[VSX\] Dec (-?[\d.]+) \.\. (-?[\d.]+) deg")  # objects_make: [VSX] Dec -90.0 .. -89.0 deg
_P_BASE = re.compile(r"base stars: strip (\d+)/(\d+)")             # astro_index base pass
_P_BAND = re.compile(r"^band (\d+) ")                               # astro_index: band k done


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def catalogs_dir() -> Path:
    """``catalogs/`` of the install (checkout or release bundle)."""
    return Path(__file__).resolve().parent.parent / "catalogs"


def jobs_root(data_root: Path | str) -> Path:
    return Path(data_root) / "jobs"


def tool_args(tool: str, **opt: Any) -> list[str]:
    """Command-line arguments of ``tool`` from the wizard options (paths always explicit)."""
    if tool == "gaia_build":
        a = ["--out", str(opt["out"]), "--mag-limit", f"{float(opt.get('mag_limit', GAIA_BUILD_MAG_LIMIT_DEFAULT)):g}",
             "--dec-min", f"{float(opt.get('dec_min', -90.0)):g}", "--dec-max", f"{float(opt.get('dec_max', 90.0)):g}"]
        if opt.get("full_source"):
            a.append("--full-source")
        return a
    if tool in ("gaia_report", "gaia_register", "gaia_drop_unused"):
        flag = {"gaia_report": "--report", "gaia_register": "--register", "gaia_drop_unused": "--drop-unused"}[tool]
        return ["--db", str(opt["db"]), flag]
    if tool == "objects_update":
        a = ["--db", str(opt["db"])]
        if opt.get("vsx", True):
            a += ["--vsx", "--vsx-mag-limit", f"{float(opt.get('vsx_mag_limit', VSX_BUILD_MAG_LIMIT_DEFAULT)):g}",
                  "--vsx-dec-min", f"{float(opt.get('vsx_dec_min', -90.0)):g}",
                  "--vsx-dec-max", f"{float(opt.get('vsx_dec_max', 90.0)):g}"]
        if opt.get("exoplanets", True):
            a.append("--exoplanets")
        return a
    if tool == "astro_index":
        return ["--db", str(opt["db"]), "--out", str(opt["out"])]
    raise ValueError(f"unknown catalog tool {tool!r}")


def launch(tool: str, args: list[str], data_root: Path | str) -> Path:
    """Start ``tool`` as a separate process; returns its job directory."""
    if tool not in TOOLS:
        raise ValueError(f"unknown catalog tool {tool!r}")
    busy = active_job(data_root, TOOLS[tool][1])
    if busy is not None:
        raise RuntimeError(f"{read_status(busy).get('label')} is still running ({busy.name})")
    script, catalog, label = TOOLS[tool]
    job_dir = jobs_root(data_root) / f"catalog_{tool}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir.mkdir(parents=True, exist_ok=False)
    job = {"tool": tool, "catalog": catalog, "label": label, "script": str(catalogs_dir() / script),
           "args": [str(a) for a in args]}
    (job_dir / JOB_FILE).write_text(json.dumps(job, indent=2), encoding="utf-8")
    write_status(job_dir, state="starting", tool=tool, catalog=catalog, label=label, started_utc=_now(),
                 message="Starting...")
    # "-c import" instead of "-m": runpy cannot run a compiled (Cython) module as __main__ (release bundle)
    cmd = [sys.executable, "-c", "import sys, catalog_jobs; sys.exit(catalog_jobs.main(sys.argv[1:]))",
           "--run", str(job_dir)]
    kw: dict[str, Any] = {"cwd": str(Path(__file__).resolve().parent)}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    write_status(job_dir, pid=int(proc.pid))
    return job_dir


def run_child(job_dir: Path | str) -> int:
    """Child side: run the script, stream its output to the log, record the return code."""
    job_dir = Path(job_dir)
    job = json.loads((job_dir / JOB_FILE).read_text(encoding="utf-8"))
    cmd = [sys.executable, "-u", str(job["script"]), *job["args"]]
    write_status(job_dir, state="running", pid=os.getpid(), message="Running...", command=cmd)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    with (job_dir / STDOUT_FILE).open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        try:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(Path(job["script"]).parent.parent), env=env)
        except BaseException as exc:  # noqa: BLE001
            write_status(job_dir, state="failed", message=f"{type(exc).__name__}: {exc}", finished_utc=_now())
            return 1
    last = [ln for ln in tail(job_dir, 40) if ln.strip()]
    msg = last[-1] if last else ""
    write_status(job_dir, state="done" if rc == 0 else "failed", returncode=int(rc),
                 message=("Done" if rc == 0 else f"exit code {rc}") + (f" - {msg}" if msg else ""),
                 finished_utc=_now())
    return int(rc)


def jobs(data_root: Path | str, catalog: str | None = None) -> list[Path]:
    """Catalog job directories, newest first (optionally of one catalog)."""
    root = jobs_root(data_root)
    if not root.is_dir():
        return []
    out = []
    for d in root.glob("catalog_*"):
        st = read_status(d) or {}
        if d.is_dir() and (catalog is None or st.get("catalog") == catalog):
            out.append(d)
    return sorted(out, key=lambda d: d.name.split("_")[-2:], reverse=True)


def job_state(job_dir: Path | str) -> str:
    """running | done | failed | lost (process gone without a final state; also no status file).
    G05-19: ``run_job.job_state`` with its "none" reported as "lost" (a listed catalog job always ran)."""
    s = _run_job_state(job_dir)
    return "lost" if s == "none" else s


def active_job(data_root: Path | str, catalog: str) -> Path | None:
    for d in jobs(data_root, catalog):
        if job_state(d) == "running":
            return d
    return None


def latest_job(data_root: Path | str, catalog: str, tool: str | None = None) -> Path | None:
    for d in jobs(data_root, catalog):
        if tool is None or (read_status(d) or {}).get("tool") == tool:
            return d
    return None


def job_args(job_dir: Path | str) -> list[str]:
    try:
        return list(json.loads((Path(job_dir) / JOB_FILE).read_text(encoding="utf-8")).get("args") or [])
    except (OSError, ValueError):
        return []


def tail(job_dir: Path | str, n: int = 30) -> list[str]:
    p = Path(job_dir) / STDOUT_FILE
    try:
        with p.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 64 * 1024))
            data = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return data.splitlines()[-n:]


def progress(job_dir: Path | str) -> float | None:
    """Fraction done from the script's own progress lines (None when it prints none)."""
    st = read_status(job_dir) or {}
    tool = st.get("tool")
    lines = tail(job_dir, 400)
    if tool == "gaia_build":
        for ln in reversed(lines):
            m = _P_GAIA.match(ln.strip())
            if m:
                return min(1.0, int(m.group(1)) / max(1, int(m.group(2))))
    if tool == "objects_update":
        args = job_args(job_dir)
        d_lo = float(args[args.index("--vsx-dec-min") + 1]) if "--vsx-dec-min" in args else -90.0
        d_hi = float(args[args.index("--vsx-dec-max") + 1]) if "--vsx-dec-max" in args else 90.0
        for ln in reversed(lines):
            if ln.startswith("[EXO]") or ln.startswith("[vsx_data] replaced"):
                return 0.97
            m = _P_VSX.match(ln.strip())
            if m:
                return min(0.95, 0.95 * (float(m.group(2)) - d_lo) / max(1e-6, d_hi - d_lo))
    if tool == "astro_index":
        from astro_index import N_BANDS  # noqa: PLC0415

        for ln in reversed(lines):
            m = _P_BAND.match(ln.strip())
            if m:
                return min(1.0, 0.3 + 0.7 * (int(m.group(1)) + 1) / N_BANDS)
            m = _P_BASE.search(ln)
            if m:
                return 0.3 * int(m.group(1)) / max(1, int(m.group(2)))
    return None


def stop(job_dir: Path | str) -> bool:
    """Stop a running job (the script and the child wrapper). Resumable builds continue later."""
    return stop_job(job_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="run one catalog job (started by the Settings page)")
    ap.add_argument("--run", required=True, help="job directory")
    a = ap.parse_args(argv)
    return run_child(a.run)


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = ["STATUS_FILE", "TOOLS", "active_job", "job_state", "jobs", "latest_job", "launch", "progress",
           "run_child", "stop", "tail", "tool_args"]
