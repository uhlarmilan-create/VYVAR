"""RESOURCES-2026-10: the run budget (half the RAM, half the CPUs), worker counts per stage from
the frame size, and the one ordered process pool (bounded, survives a broken pool, workers exit
with the main process)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

import pipeline  # noqa: F401  app import order
import parallel_frames as pf
from infolog import capture_events, log_event
from parallel_frames import map_frames

SRC = Path(__file__).resolve().parents[2] / "src_py"


# --- ordered map -------------------------------------------------------------------------------

def test_map_frames_keeps_order_serial_and_parallel():
    items = [-3, 1, -2, 7, -5]
    serial = list(map_frames(abs, items, workers=1, initializer=len, ctx={}))
    parallel = list(map_frames(abs, items, workers=2, initializer=len, ctx={}))
    assert serial == parallel == [3, 1, 2, 7, 5]


def test_map_frames_replays_worker_log_lines_in_frame_order():
    with capture_events() as events:
        list(map_frames(log_event, ["a", "b", "c", "d"], workers=2, initializer=len, ctx={}))
    assert events == ["a", "b", "c", "d"]


def test_map_frames_without_initializer():
    assert list(map_frames(abs, [-1, -2, -3], workers=2)) == [1, 2, 3]


def test_map_frames_keeps_at_most_two_items_per_worker_in_flight():
    """Results do not pile up: an item is submitted only when the consumer has taken the one
    2 x workers places earlier (the alignment kept every frame of a setup in memory)."""
    submitted: list[int] = []

    def pack(i: int) -> int:
        submitted.append(i)
        return i

    for k, out in enumerate(map_frames(abs, list(range(20)), workers=2, pack=pack)):
        assert out == k
        assert max(submitted) <= k + pf.IN_FLIGHT_PER_WORKER * 2


# --- failures ------------------------------------------------------------------------------------

def _die_in_worker(item: tuple[int, int]) -> int:
    x, main_pid = item
    if x == 3 and os.getpid() != main_pid:
        os._exit(9)  # a worker killed for memory
    return x * 10


def _memory_error_in_worker(item: tuple[int, int]) -> int:
    x, main_pid = item
    if x == 2 and os.getpid() != main_pid:
        raise MemoryError("simulated")
    return x * 10


@pytest.mark.parametrize("fn", [_die_in_worker, _memory_error_in_worker])
def test_broken_pool_continues_with_fewer_workers(fn, caplog):
    items = [(i, os.getpid()) for i in range(8)]
    with caplog.at_level("WARNING"):
        out = list(map_frames(fn, items, workers=2, stage="test"))
    assert out == [i * 10 for i in range(8)]
    assert any("continuing with 1 worker" in r.getMessage() for r in caplog.records)


def test_memory_error_in_the_main_process_is_raised():
    def boom(_x: int) -> int:
        raise MemoryError("one frame does not fit")

    with pytest.raises(MemoryError):
        list(map_frames(boom, [1, 2], workers=1))


def test_workers_exit_when_the_main_process_dies(tmp_path):
    """A worker must not outlive the run (orphans held 2.2 GB after a killed run)."""
    pids_file = tmp_path / "pids.txt"
    script = tmp_path / "run_pool.py"
    script.write_text(textwrap.dedent(f"""
        import os, sys, time
        sys.path.insert(0, {str(SRC)!r})
        from parallel_frames import map_frames

        def task(_x):
            with open({str(pids_file)!r}, "a") as fh:
                fh.write(f"{{os.getpid()}}\\n")
            time.sleep(120)
            return 0

        if __name__ == "__main__":
            list(map_frames(task, list(range(4)), workers=2))
    """))
    proc = subprocess.Popen([sys.executable, str(script)])
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if pids_file.is_file() and len(pids_file.read_text().split()) >= 2:
                break
            time.sleep(0.2)
        workers = [int(p) for p in pids_file.read_text().split()]
        assert len(workers) >= 2
        proc.kill()  # SIGKILL: no clean-up in the main process
        proc.wait(timeout=10)

        def alive() -> list[int]:
            out = []
            for p in workers:
                try:
                    if psutil.Process(p).status() != psutil.STATUS_ZOMBIE:
                        out.append(p)
                except psutil.NoSuchProcess:
                    pass
            return out

        deadline = time.time() + 3 * pf.PARENT_WATCH_S + 5
        while alive() and time.time() < deadline:
            time.sleep(0.2)
        assert not alive(), f"orphaned workers still alive: {alive()}"
    finally:
        if proc.poll() is None:
            proc.kill()


# --- budget and worker counts -----------------------------------------------------------------

def _budget(cpu: int, ram_gb: float) -> pf.ResourceBudget:
    b = int(ram_gb * 1024**3)
    return pf.ResourceBudget(logical_cpus=2 * cpu, cpu_workers=cpu, total_ram_bytes=2 * b, ram_budget_bytes=b,
                             ram_source="test", cpu_source="test")


def _plenty_available(monkeypatch) -> None:
    vm = type("vm", (), {"available": 512 * 1024**3, "total": 512 * 1024**3})()
    monkeypatch.setattr(psutil, "virtual_memory", lambda: vm)


def test_resource_budget_is_half_of_ram_and_cpus(monkeypatch):
    monkeypatch.delenv("VYVAR_PARALLEL_WORKERS", raising=False)
    monkeypatch.setattr(pf, "_cgroup_memory_limit", lambda: None)
    monkeypatch.setattr(pf.os, "cpu_count", lambda: 16)
    b = pf.resource_budget()
    assert b.cpu_workers == 8
    assert b.ram_budget_bytes == int(psutil.virtual_memory().total * 0.5)


def test_resource_budget_respects_the_container_limit(monkeypatch):
    monkeypatch.setattr(pf, "_cgroup_memory_limit", lambda: 4 * 1024**3)
    if psutil.virtual_memory().total <= 4 * 1024**3:
        pytest.skip("machine smaller than the simulated limit")
    b = pf.resource_budget()
    assert b.total_ram_bytes == 4 * 1024**3 and b.ram_budget_bytes == 2 * 1024**3
    assert b.ram_source == "container limit"


def test_env_override_replaces_the_cpu_budget_only(monkeypatch):
    monkeypatch.setenv("VYVAR_PARALLEL_WORKERS", "3")
    b = pf.resource_budget()
    assert b.cpu_workers == 3 and b.ram_budget_bytes == int(b.total_ram_bytes * pf.RAM_BUDGET_FRACTION)


def test_stage_workers_one_item_is_one_worker():
    assert pf.stage_workers(1, pf.frame_bytes(4000, 6000), stage="t") == 1
    assert pf.stage_workers(0, pf.frame_bytes(4000, 6000), stage="t") == 1


def test_stage_workers_limited_by_ram_for_large_frames(monkeypatch):
    """Brno 534 frames (3552 x 2664) on an 8 GB / 16-thread laptop: the old 48 MB-per-worker guess
    gave 7 workers and a crash; with the frame-size cost 3 GB left in the budget allows 2."""
    monkeypatch.setattr(pf, "resource_budget", lambda: _budget(cpu=8, ram_gb=4.0))
    monkeypatch.setattr(pf, "_main_process_rss", lambda: int(1.0 * 1024**3))
    _plenty_available(monkeypatch)
    fb = pf.frame_bytes(2664, 3552)
    cost = pf.WORKER_FRAME_FACTOR * fb + pf.WORKER_BASE_BYTES
    assert pf.stage_workers(97, fb, stage="t") == int((3.0 * 1024**3) // cost) == 2


def test_stage_workers_limited_by_cpu_for_small_frames(monkeypatch):
    monkeypatch.setattr(pf, "resource_budget", lambda: _budget(cpu=8, ram_gb=64.0))
    monkeypatch.setattr(pf, "_main_process_rss", lambda: 0)
    _plenty_available(monkeypatch)
    assert pf.stage_workers(150, pf.frame_bytes(1398, 2082), stage="t") == 8
    assert pf.stage_workers(3, pf.frame_bytes(1398, 2082), stage="t") == 3


def test_stage_workers_limited_by_what_the_system_has_free(monkeypatch):
    monkeypatch.setattr(pf, "resource_budget", lambda: _budget(cpu=8, ram_gb=64.0))
    monkeypatch.setattr(pf, "_main_process_rss", lambda: 0)
    vm = type("vm", (), {"available": 2 * 1024**3, "total": 128 * 1024**3})()
    monkeypatch.setattr(psutil, "virtual_memory", lambda: vm)
    fb = pf.frame_bytes(1398, 2082)
    cost = pf.WORKER_FRAME_FACTOR * fb + pf.WORKER_BASE_BYTES
    assert pf.stage_workers(150, fb, stage="t") == int((0.9 * 2 * 1024**3) // cost)


def test_stage_workers_never_below_one(monkeypatch):
    monkeypatch.setattr(pf, "resource_budget", lambda: _budget(cpu=8, ram_gb=0.5))
    monkeypatch.setattr(pf, "_main_process_rss", lambda: int(2 * 1024**3))
    assert pf.stage_workers(100, pf.frame_bytes(10000, 10000), stage="t") == 1


# --- one pool -------------------------------------------------------------------------------------

def test_process_pools_are_made_only_by_parallel_frames():
    """Every per-frame pool goes through map_frames (budget, order, watchdog, recovery)."""
    offenders = []
    for p in sorted(SRC.glob("*.py")):
        if p.name == "parallel_frames.py":
            continue
        if re.search(r"ProcessPoolExecutor\(|multiprocessing\.Pool\(|\.Pool\(", p.read_text(encoding="utf-8")):
            offenders.append(p.name)
    assert offenders == []


def test_no_legacy_worker_settings_left():
    pat = re.compile(r"VYVAR_QC_PREPROCESS_WORKERS|VYVAR_PER_FRAME_CSV_WORKERS|VYVAR_PARALLEL_BACKEND|"
                     r"qc_preprocess_workers|recommended_vyvar_parallel_workers|ram_capped_workers|"
                     r"_vyvar_parallel_worker_count|per_frame_mp_reserve_ram_gb")
    hits = []
    for p in sorted(SRC.glob("*.py")):
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            if pat.search(line) and "removed 2026-10-26" not in line and not line.strip().startswith('"per_frame_mp'):
                hits.append(f"{p.name}:{n}")
    assert hits == []
