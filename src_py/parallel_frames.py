"""Resources and the one process pool of the pipeline (RESOURCES-2026-10).

VYVAR sizes its parallel work by itself (Milan 2026-10-26): a run uses at most half of the
computer's RAM and half of its logical CPUs, and every per-frame stage takes as many worker
processes as fit into that budget for the size of its frames. Nothing to set.

* :func:`resource_budget` - the run budget: ``RAM_BUDGET_FRACTION`` of the total RAM (on Linux
  also the container limit) and ``CPU_FRACTION`` of the logical CPUs.
* :func:`stage_workers` - worker processes for one stage: CPU budget, number of items, and the
  RAM budget left beside the main process divided by the cost of one worker,
  ``WORKER_FRAME_FACTOR`` x frame bytes (float32) + ``WORKER_BASE_BYTES``.
* :func:`map_frames` - the one pool: runs a module-level function over items in a spawn process
  pool and yields the results in item order, so the caller merges them exactly as a serial
  loop would. ``log_event`` lines a worker writes are replayed in the parent in item order.
  At most two items per worker are in flight, so finished results never pile up in the main
  process. A worker exits by itself when the main process is gone. When the pool breaks
  (a worker killed for memory, ``MemoryError``), the remaining items run again with half the
  workers, down to one (in the main process). With one worker the function runs in the parent.

``VYVAR_PARALLEL_WORKERS`` (environment) replaces the CPU budget; it is for tests and
benchmarks, the RAM budget still applies.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy.io import fits

LOGGER = logging.getLogger(__name__)

#: Share of the computer a run may use (Milan 2026-10-26: half of the RAM, half of the CPUs).
RAM_BUDGET_FRACTION = 0.5
CPU_FRACTION = 0.5
#: Memory of one worker process = factor x one frame as float32 + interpreter and libraries.
#: Measured on Brno 534 (9.5 Mpx frames, 2026-10-26): QC/preprocess 29x, per-frame catalogs 27x,
#: alignment 24x, ePSF 19x, calibration 11x; the largest is used for every stage.
WORKER_FRAME_FACTOR = 30
WORKER_BASE_BYTES = 300 * 1024**2
#: Items in flight per worker (bounded queue: results and pickled inputs do not pile up).
IN_FLIGHT_PER_WORKER = 2
#: Seconds between the worker's checks that the main process is still alive.
PARENT_WATCH_S = 2.0


@dataclass(frozen=True)
class ResourceBudget:
    logical_cpus: int
    cpu_workers: int
    total_ram_bytes: int
    ram_budget_bytes: int
    ram_source: str
    cpu_source: str

    def describe(self) -> str:
        return (
            f"{self.logical_cpus} logical CPUs -> up to {self.cpu_workers} worker(s) ({self.cpu_source}); "
            f"RAM {self.total_ram_bytes / 1024**3:.1f} GB ({self.ram_source}) -> "
            f"budget {self.ram_budget_bytes / 1024**3:.1f} GB"
        )


def _cgroup_memory_limit() -> int | None:
    """Memory limit of the container (Linux cgroup v2 / v1), None when there is none."""
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(p).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 1 << 60:
            return int(raw)
    return None


def resource_budget() -> ResourceBudget:
    """RAM and CPU budget of a run (see module docstring)."""
    import psutil  # noqa: PLC0415

    logical = int(os.cpu_count() or 1)
    env = (os.environ.get("VYVAR_PARALLEL_WORKERS") or "").strip()
    if env.isdigit() and int(env) >= 1:
        cpu_workers, cpu_src = int(env), "VYVAR_PARALLEL_WORKERS"
    else:
        cpu_workers, cpu_src = max(1, int(logical * CPU_FRACTION)), f"{CPU_FRACTION:.0%} of logical CPUs"
    total = int(psutil.virtual_memory().total)
    src = "physical"
    lim = _cgroup_memory_limit()
    if lim is not None and lim < total:
        total, src = lim, "container limit"
    return ResourceBudget(
        logical_cpus=logical,
        cpu_workers=cpu_workers,
        total_ram_bytes=total,
        ram_budget_bytes=int(total * RAM_BUDGET_FRACTION),
        ram_source=src,
        cpu_source=cpu_src,
    )


def frame_bytes(ny: int, nx: int) -> int:
    """Bytes of one frame as float32."""
    return max(int(ny), 1) * max(int(nx), 1) * 4


def frame_bytes_of_fits(path: Path | str) -> int:
    """Bytes of the primary image of a FITS file as float32 (header only)."""
    with fits.open(path, memmap=False) as h:
        hdr = h[0].header
        return frame_bytes(int(hdr.get("NAXIS2", 0) or 0), int(hdr.get("NAXIS1", 0) or 0))


#: Disk a run writes per light frame beside the imported copy: calibrated + aligned frame (float32).
DISK_FRAMES_PER_LIGHT = 2
DISK_MARGIN = 1.10


def run_disk_need(light_paths: Sequence[Path | str]) -> int:
    """Bytes a run writes into the archive (DISK-PREFLIGHT-2026-10): the imported copy of every
    light plus ``DISK_FRAMES_PER_LIGHT`` float32 frames per light, times ``DISK_MARGIN`` for the
    per-frame tables, MASTERSTAR and reports. Measured on Brno 534 (196 lights, 3.6 GB raw): 15 GB."""
    raw = sum(Path(p).stat().st_size for p in light_paths)
    frames = sum(frame_bytes_of_fits(p) for p in light_paths)
    return int(DISK_MARGIN * (raw + DISK_FRAMES_PER_LIGHT * frames))


def check_disk_for_run(archive_root: Path | str, light_paths: Sequence[Path | str]) -> str:
    """Raise ``OSError`` when the archive's disk cannot hold the run; else return the logged line."""
    import shutil  # noqa: PLC0415

    root = Path(archive_root)
    probe = root if root.exists() else next((q for q in root.parents if q.exists()), root)
    free = int(shutil.disk_usage(probe).free)
    need = run_disk_need(light_paths)
    line = (f"[RESOURCES] disk: run needs ~{need / 1024**3:.1f} GB for {len(light_paths)} light(s), "
            f"{free / 1024**3:.1f} GB free on {probe}")
    if need > free:
        raise OSError(f"{line} - not enough disk space; free {(need - free) / 1024**3:.1f} GB and run again")
    return line


def _main_process_rss() -> int:
    import psutil  # noqa: PLC0415

    return int(psutil.Process().memory_info().rss)


def stage_workers(n_items: int, frame_nbytes: int, *, stage: str, cpu_workers: int | None = None) -> int:
    """Worker processes for one per-frame stage (see module docstring); logged with the reason.

    ``cpu_workers`` replaces the CPU budget (tests that need an exact count); RAM still applies."""
    import dataclasses  # noqa: PLC0415

    import psutil  # noqa: PLC0415

    n_items = int(n_items)
    if n_items <= 1:
        return 1
    b = resource_budget()
    if cpu_workers is not None:
        b = dataclasses.replace(b, cpu_workers=max(1, int(cpu_workers)))
    cost = WORKER_FRAME_FACTOR * int(frame_nbytes) + WORKER_BASE_BYTES
    main = _main_process_rss()
    free_for_workers = min(b.ram_budget_bytes - main, int(0.9 * psutil.virtual_memory().available))
    by_ram = max(1, free_for_workers // cost) if free_for_workers > 0 else 1
    nw = int(max(1, min(b.cpu_workers, n_items, by_ram)))
    LOGGER.info(
        "[RESOURCES] %s: %d worker(s) for %d item(s) - CPU %d, RAM %d (worker %.2f GB, main %.2f GB, "
        "budget %.1f GB)",
        stage, nw, n_items, b.cpu_workers, by_ram, cost / 1024**3, main / 1024**3, b.ram_budget_bytes / 1024**3,
    )
    return nw


def _watch_parent(parent_pid: int) -> None:
    import psutil  # noqa: PLC0415

    while True:
        time.sleep(PARENT_WATCH_S)
        if os.getppid() != parent_pid or not psutil.pid_exists(parent_pid):
            os._exit(1)


def _worker_bootstrap(parent_pid: int, initializer: Callable[[Any], None] | None, ctx: Any) -> None:
    """Initializer of every pool worker: watch the main process, then the stage's own init."""
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True, name="vyvar-parent-watch").start()
    from infolog import quiet_repeated_library_warnings  # noqa: PLC0415

    quiet_repeated_library_warnings()
    if initializer is not None:
        initializer(ctx)


def _run_captured(fn: Callable[[Any], Any], item: Any) -> tuple[Any, list[str], dict[str, int]]:
    """Worker side: the result, the collected infolog events and the EXCEPT-FIX counts of this item
    (G12-30: the run process adds them to its own counters)."""
    from except_fix_counters import counter_deltas, get_except_fix_counters  # noqa: PLC0415
    from infolog import capture_events  # noqa: PLC0415

    before = get_except_fix_counters().snapshot()
    with capture_events() as events:
        out = fn(item)
    return out, list(events), counter_deltas(before)


def _run_serial(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    start: int,
    initializer: Callable[[Any], None] | None,
    ctx: Any,
    pack: Callable[[Any], Any] | None,
    serial_ctx: Any,
) -> Iterator[Any]:
    if initializer is not None:
        initializer(serial_ctx if serial_ctx is not None else ctx)
    try:
        for it in items[start:]:
            yield fn(pack(it) if pack is not None else it)
    finally:
        if initializer is not None:
            initializer({})


def map_frames(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    workers: int,
    initializer: Callable[[Any], None] | None = None,
    ctx: Any = None,
    pack: Callable[[Any], Any] | None = None,
    serial_ctx: Any = None,
    stage: str = "",
) -> Iterator[Any]:
    """Yield ``fn(pack(item))`` for every item, in item order (see module docstring).

    ``initializer(ctx)`` sets each worker's read-only state; in the serial path it gets
    ``serial_ctx`` when given (else ``ctx``) and ``initializer({})`` clears it afterwards.
    ``pack(item)`` builds the argument when the item is submitted (large inputs are made only
    then)."""
    import pickle  # noqa: PLC0415

    n = len(items)
    nw = max(1, min(int(workers or 1), n))
    start = 0
    while start < n:
        if nw == 1:
            yield from _run_serial(fn, items, start, initializer, ctx, pack, serial_ctx)
            return
        import multiprocessing  # noqa: PLC0415
        from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415
        from concurrent.futures.process import BrokenProcessPool  # noqa: PLC0415

        from except_fix_counters import add_except_fix_counts  # noqa: PLC0415
        from infolog import replay_events  # noqa: PLC0415

        task = functools.partial(_run_captured, fn)
        ex = ProcessPoolExecutor(
            max_workers=nw,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_bootstrap,
            initargs=(os.getpid(), initializer, ctx),
        )
        pending: deque[Any] = deque()
        nxt = start
        try:
            while start < n:
                while nxt < n and len(pending) < IN_FLIGHT_PER_WORKER * nw:
                    it = items[nxt]
                    pending.append(ex.submit(task, pack(it) if pack is not None else it))
                    nxt += 1
                out, events, counts = pending.popleft().result()
                replay_events(events)
                add_except_fix_counts(counts)
                start += 1
                yield out
        except (BrokenProcessPool, MemoryError) as exc:
            for f in pending:
                f.cancel()
            ex.shutdown(wait=True, cancel_futures=True)
            new_nw = max(1, nw // 2)
            LOGGER.warning(
                "[RESOURCES] %s: worker pool failed at item %d of %d (%s: %s) - "
                "continuing with %d worker(s)",
                stage or getattr(fn, "__name__", "stage"), start + 1, n, type(exc).__name__, exc, new_nw,
            )
            nw = new_nw
            continue
        except pickle.PicklingError as exc:
            # The worker function could not be sent to the pool (e.g. a module reloaded by the
            # Streamlit file watcher). Nothing of this pool has been yielded past ``start``.
            for f in pending:
                f.cancel()
            ex.shutdown(wait=True, cancel_futures=True)
            LOGGER.warning("[RESOURCES] %s: cannot send the task to worker processes (%s) - running "
                           "in the main process", stage or "stage", exc)
            nw = 1
            continue
        except BaseException:
            for f in pending:
                f.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        ex.shutdown(wait=True)
        return
