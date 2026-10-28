"""CONSOLIDATE-01E6a: MP spawn smoke for the export-per-frame worker.

Imports from the pipeline facade so the test is unchanged across C3
(the move). After C3, pickle follows __module__ into pipeline_catalog
in the spawn child. Note for E-final retarget: this file still imports
from pipeline on purpose.

Exercises the production initargs SHAPE (1-tuple wrapping a state dict)
and one state-reader round-trip. A full export job is not required:
G2 --full already exercises export_per_frame_catalogs end-to-end.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import dataclasses

from config import AppConfig
from pipeline import (
    _EXPORT_PER_FRAME_WORKER_STATE,
    _init_export_per_frame_worker,
)


def _probe_worker_state_key(key: str):
    """Return one value from the spawn-worker global (same module as initializer)."""
    st = _EXPORT_PER_FRAME_WORKER_STATE
    if key == "_cfg_observer_lat":
        return float(st["cfg"].observer_lat)  # CONFIG-ONE-2026-10: the run config travels in the state
    return st.get(key)


def test_export_per_frame_mp_spawn_state_roundtrip() -> None:
    import multiprocessing

    state = {
        "cfg": dataclasses.replace(AppConfig(), observer_lat=48.5, observer_lon=17.1, observer_alt_m=200.0),
        "probe": "e6a",
    }
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=ctx,
        initializer=_init_export_per_frame_worker,
        initargs=(state,),
    ) as ex:
        probe = ex.submit(_probe_worker_state_key, "probe").result(timeout=25)
        lat = ex.submit(_probe_worker_state_key, "_cfg_observer_lat").result(timeout=25)

    assert probe == "e6a"
    assert lat == 48.5
