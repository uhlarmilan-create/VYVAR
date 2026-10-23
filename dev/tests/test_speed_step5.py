"""CODE-REVIEW-2026-10 step 5 (speed): the frame-parallel map keeps order, results and log lines."""

from __future__ import annotations

import pipeline  # noqa: F401  app import order
from infolog import capture_events, log_event
from parallel_frames import map_frames, ram_capped_workers


def test_map_frames_keeps_order_serial_and_parallel():
    items = [-3, 1, -2, 7, -5]
    serial = list(map_frames(abs, items, workers=1, initializer=len, ctx={}))
    parallel = list(map_frames(abs, items, workers=2, initializer=len, ctx={}))
    assert serial == parallel == [3, 1, 2, 7, 5]


def test_map_frames_replays_worker_log_lines_in_frame_order():
    with capture_events() as events:
        list(map_frames(log_event, ["a", "b", "c", "d"], workers=2, initializer=len, ctx={}))
    assert events == ["a", "b", "c", "d"]


def test_ram_capped_workers_one_frame_is_one_worker(tmp_path):
    assert ram_capped_workers(None, [tmp_path / "x.fits"], bytes_per_pixel=48, max_workers=8) == 1
    assert ram_capped_workers(None, [], bytes_per_pixel=48, max_workers=8) == 1
