"""Unit tests for iterative ensemble-relative comp clip."""

from __future__ import annotations

import math

import numpy as np

from comp_selection_per_target import _iterative_ensemble_clip_cm_residual


def _synthetic_pool(n_stable: int = 8, *, outlier_amp: float = 0.05) -> tuple[dict, dict, dict]:
    n_frames = 40
    bjd0 = 2460000.0
    flux_map: dict[str, list[float]] = {}
    bjd_map: dict[str, list[float]] = {}
    rms_map: dict[str, float] = {}
    for i in range(n_stable):
        cid = f"100000000000000{i:04d}"
        bjds = [bjd0 + j * 0.001 for j in range(n_frames)]
        fluxes = [1.0 + 0.002 * math.sin(j * 0.3) for j in range(n_frames)]
        flux_map[cid] = fluxes
        bjd_map[cid] = bjds
        rms_map[cid] = 0.02
    outlier = "9999999999999999999"
    bjds_o = [bjd0 + j * 0.001 for j in range(n_frames)]
    fluxes_o = [1.0 + outlier_amp * math.sin(j * 0.5 + 1.0) for j in range(n_frames)]
    flux_map[outlier] = fluxes_o
    bjd_map[outlier] = bjds_o
    rms_map[outlier] = 0.02
    return flux_map, bjd_map, rms_map


def test_iterative_clip_keeps_stable_pool() -> None:
    flux_map, bjd_map, rms_map = _synthetic_pool(8, outlier_amp=0.002)
    out = _iterative_ensemble_clip_cm_residual(
        flux_map, bjd_map, rms_map, clip_sigma=5.0, n_comp_min=3
    )
    assert out is not None
    active, meta = out
    assert meta["comp_pool_n_candidates"] == 9
    assert meta["comp_pool_n_final"] >= 8
    assert len(active) >= 8


def test_iterative_clip_excludes_divergent_comp() -> None:
    flux_map, bjd_map, rms_map = _synthetic_pool(8, outlier_amp=0.15)
    out = _iterative_ensemble_clip_cm_residual(
        flux_map, bjd_map, rms_map, clip_sigma=5.0, n_comp_min=3
    )
    assert out is not None
    active, meta = out
    assert "9999999999999999999" not in active
    assert meta["comp_pool_n_clipped"] >= 1
    assert meta["comp_pool_n_final"] >= 3


def test_global_pool_rms_prefilter_bypass_flag() -> None:
    from comp_pool_rms import compute_global_pool_rms_map

    # Empty paths -> empty map; flag wiring only (no crash).
    m = compute_global_pool_rms_map(set(), None, [], {}, apply_rms_prefilter=False, max_comp_rms=0.1)
    assert m == {}
