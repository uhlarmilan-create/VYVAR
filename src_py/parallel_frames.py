"""Frame-parallel map for per-frame pipeline stages (CODE-REVIEW-2026-10 step 5).

``map_frames`` runs a module-level function over frames in a spawn process pool and yields the
results in frame order, so the caller merges them exactly as a serial loop would; ``log_event``
lines a worker writes are collected there and replayed in the parent in frame order. With one
worker the function runs in the parent, unchanged. ``ram_capped_workers`` is the pipeline's one
worker count (``VYVAR_PARALLEL_WORKERS`` / ``qc_preprocess_workers``) capped by free RAM.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

from astropy.io import fits


def ram_capped_workers(cfg: Any, files: Sequence[Path], *, bytes_per_pixel: int, max_workers: int | None = None) -> int:
    """Worker processes for a per-frame stage: the pipeline's one count, capped by free RAM.

    A worker holds ``bytes_per_pixel`` bytes per frame pixel plus the interpreter (~300 MB)."""
    from pipeline_calibrate import _vyvar_parallel_worker_count  # noqa: PLC0415

    nw = int(max_workers) if max_workers is not None else int(_vyvar_parallel_worker_count(cfg))
    nw = max(1, min(32, nw, len(files)))
    if nw <= 1 or not files:
        return 1
    try:
        import psutil  # noqa: PLC0415

        with fits.open(files[0], memmap=False) as h:
            ny = int(h[0].header.get("NAXIS2", 0) or 0)
            nx = int(h[0].header.get("NAXIS1", 0) or 0)
        per_worker = max(ny * nx, 1) * int(bytes_per_pixel) + 300 * 1024**2
        reserve = int(max(0.0, float(getattr(cfg, "per_frame_mp_reserve_ram_gb", 0.0) or 0.0)) * 1024**3)
        avail = int(psutil.virtual_memory().available) - reserve
        nw = max(1, min(nw, avail // per_worker if avail > 0 else 1))
    except Exception:  # noqa: BLE001 - RAM probe is advisory
        pass
    return int(nw)


def _run_captured(fn: Callable[[Any], Any], item: Any) -> tuple[Any, list[str]]:
    from infolog import capture_events  # noqa: PLC0415

    with capture_events() as events:
        out = fn(item)
    return out, list(events)


def map_frames(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    *,
    workers: int,
    initializer: Callable[[dict[str, Any]], None],
    ctx: dict[str, Any],
) -> Iterator[Any]:
    """Yield ``fn(item)`` for every item, in order. ``initializer(ctx)`` sets each worker's
    read-only state (``initializer({})`` clears it after a serial run)."""
    nw = max(1, min(int(workers or 1), len(items)))
    if nw == 1:
        initializer(ctx)
        try:
            for it in items:
                yield fn(it)
        finally:
            initializer({})
        return
    import multiprocessing  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor  # noqa: PLC0415

    from infolog import replay_events  # noqa: PLC0415

    with ProcessPoolExecutor(
        max_workers=nw, mp_context=multiprocessing.get_context("spawn"), initializer=initializer, initargs=(ctx,)
    ) as ex:
        for out, events in ex.map(functools.partial(_run_captured, fn), items):
            replay_events(events)
            yield out
