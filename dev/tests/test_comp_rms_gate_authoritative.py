"""COMP-ADMIT-03: max_comp_rms is diagnostic; scatter enters weights, not a hard cut.

Legacy known-issue (b) tests for authoritative RMS gate are retained as
``_count_gate_passing_comps`` diagnostics (routing helper), but the detrend path
must no longer drop above-gate comps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from comp_selection_per_target import _detrend_and_compute_comp_rms_map
from photometry_core import _count_gate_passing_comps

_TARGET = pd.Series({"x": float("nan"), "y": float("nan"), "vsx_name": "TEST"})


def _alternating_flux(amp: float, n_frames: int = 20) -> list[float]:
    return [1.0 + (amp if (j % 2 == 0) else -amp) for j in range(n_frames)]


def _build_flux_map(amps: dict[str, float]) -> dict[str, list[float]]:
    return {cid: _alternating_flux(a) for cid, a in amps.items()}


def test_comp_admit_03_keeps_above_gate_comps() -> None:
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
    assert rms_map is not None
    assert len(rms_map) == 5
    assert "1000000000000000003" in rms_map


def test_comp_admit_03_retains_all_finite_rms() -> None:
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
    assert "1000000000000000009" in rms_map
    assert len(rms_map) == 5


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
    result = pd.DataFrame({"name": ["1112113066119992064"], "comp_rms": [0.134]})
    rms_map = {"1112113066119992064": 0.1337371886361533}
    assert _count_gate_passing_comps(result, rms_map, 0.1, "name") == 0


def test_count_gate_passing_comps_thin_set_kept() -> None:
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
    result = pd.DataFrame({"name": ["x1", "x2"]})
    rms_map = {"x1": np.float64(0.09), "x2": np.float64(0.2)}
    assert _count_gate_passing_comps(result, rms_map, 0.1, "name") == 1
