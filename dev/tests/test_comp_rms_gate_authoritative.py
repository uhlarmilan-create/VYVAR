"""Tests for known-issue (b): per-target comp_rms gate authoritative for N_good.

Two loci:
  1. ``_detrend_and_compute_comp_rms_map`` RMS fallback must never admit a comp
     with ``comp_rms > max_comp_rms`` (no above-gate relaxation).
  2. ``_count_gate_passing_comps`` routing helper counts only gate-passers, so
     auto routing flips to sparse_fallback at zero gate-passers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from comp_selection_per_target import _detrend_and_compute_comp_rms_map
from photometry_core import _count_gate_passing_comps

_TARGET = pd.Series({"x": float("nan"), "y": float("nan"), "vsx_name": "TEST"})


def _alternating_flux(amp: float, n_frames: int = 20) -> list[float]:
    """Flux series whose detrended RMS ~= ``amp`` (quadratic detrend can't flatten)."""
    return [1.0 + (amp if (j % 2 == 0) else -amp) for j in range(n_frames)]


def _build_flux_map(amps: dict[str, float]) -> dict[str, list[float]]:
    return {cid: _alternating_flux(a) for cid, a in amps.items()}


def test_fallback_never_admits_above_gate_comp() -> None:
    # 2 gate-passers (0.02) + 3 above-gate (0.13). Old code relaxed to 0.15 and
    # returned the above-gate comps; the gate is now authoritative.
    flux_map = _build_flux_map(
        {
            "1000000000000000001": 0.02,
            "1000000000000000002": 0.02,
            "1000000000000000003": 0.13,
            "1000000000000000004": 0.13,
            "1000000000000000005": 0.13,
        }
    )
    rms_map, _sorted = _detrend_and_compute_comp_rms_map(
        flux_map,
        min_frames=10,
        max_comp_rms=0.1,
        n_comp_min=3,
        target_cid="9000000000000000000",
        target=_TARGET,
        chip_fw=None,
        chip_fh=None,
        chip_interior_margin_px=0,
    )
    # Only 2 gate-passers (< n_comp_min) and no above-gate padding -> None.
    assert rms_map is None


def test_gate_passers_retained_when_enough() -> None:
    # 4 gate-passers (>= n_comp_min): fallback never fires; no above-gate comp.
    flux_map = _build_flux_map(
        {
            "1000000000000000001": 0.02,
            "1000000000000000002": 0.03,
            "1000000000000000003": 0.04,
            "1000000000000000004": 0.05,
            "1000000000000000009": 0.13,
        }
    )
    rms_map, _sorted = _detrend_and_compute_comp_rms_map(
        flux_map,
        min_frames=10,
        max_comp_rms=0.1,
        n_comp_min=3,
        target_cid="9000000000000000000",
        target=_TARGET,
        chip_fw=None,
        chip_fh=None,
        chip_interior_margin_px=0,
    )
    assert rms_map is not None
    assert "1000000000000000009" not in rms_map  # above gate, dropped
    assert all(v <= 0.1 + 1e-9 for v in rms_map.values())
    assert len(rms_map) == 4


def test_count_gate_passing_comps_excludes_above_gate() -> None:
    result = pd.DataFrame(
        {
            "name": ["1000000000000000001", "1000000000000000002"],
            "comp_rms": [0.034, 0.134],
        }
    )
    rms_map = {"1000000000000000001": 0.034, "1000000000000000002": 0.134}
    assert _count_gate_passing_comps(result, rms_map, 0.1, "name") == 1


def test_count_gate_passing_comps_zero_routes_sparse() -> None:
    # SS Cam analogue: single comp above gate -> zero gate-passers.
    result = pd.DataFrame({"name": ["1112113066119992064"], "comp_rms": [0.134]})
    rms_map = {"1112113066119992064": 0.1337371886361533}
    assert _count_gate_passing_comps(result, rms_map, 0.1, "name") == 0


def test_count_gate_passing_comps_thin_set_kept() -> None:
    # V0612 analogue: single sub-gate comp -> stays on default.
    result = pd.DataFrame({"name": ["1111749368289526912"], "comp_rms": [0.034]})
    rms_map = {"1111749368289526912": 0.0340308304009247}
    assert _count_gate_passing_comps(result, rms_map, 0.1, "name") == 1


def test_count_gate_passing_comps_gate_disabled_uses_len() -> None:
    result = pd.DataFrame({"name": ["a", "b"], "comp_rms": [0.5, 9.9]})
    assert _count_gate_passing_comps(result, {}, 0.0, "name") == 2
    assert _count_gate_passing_comps(result, {}, float("nan"), "name") == 2


def test_count_gate_passing_comps_empty_result() -> None:
    assert _count_gate_passing_comps(pd.DataFrame(), {}, 0.1, "name") == 0
    assert _count_gate_passing_comps(None, {}, 0.1, "name") == 0


def test_count_gate_passing_comps_uses_per_target_map_for_id() -> None:
    # On the default path the row's comp_rms is field-wide; the per-target map is
    # the authoritative source. Verify lookup keys on id, not on the row value.
    result = pd.DataFrame({"name": ["x1", "x2"]})
    rms_map = {"x1": np.float64(0.09), "x2": np.float64(0.2)}
    assert _count_gate_passing_comps(result, rms_map, 0.1, "name") == 1
