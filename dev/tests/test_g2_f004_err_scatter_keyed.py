# -*- coding: utf-8 -*-
"""G2-F004 - err paired to ensemble_scatter by source_file (not positional index)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pytest

from photometry_core import (
    _PSF_ERR_MAG_SCALE,
    _combine_err_with_ensemble_scatter_keyed,
    _ensemble_scatter_by_source_file,
)


def _scatter_mag_to_rel(sc_mag: float) -> float:
    return sc_mag / _PSF_ERR_MAG_SCALE if sc_mag > 0.0 else 0.0


def _expected_combine(err_rel: float, scatter_mag: float) -> float:
    sc_rel = _scatter_mag_to_rel(scatter_mag)
    return math.sqrt(err_rel * err_rel + sc_rel * sc_rel)


def _make_all_frames(target_cid: str, source_files: list[str]) -> pd.DataFrame:
    rows = []
    for sf in source_files:
        rows.append(
            {
                "catalog_id": target_cid,
                "source_file": sf,
                "mag_inst": 10.0,
                "err": 0.01,
            }
        )
    return pd.DataFrame(rows)


def test_keyed_domain_consistent_quadrature() -> None:
    """Photon err (rel flux) and ensemble SEM (mag) combine in one domain."""
    err_photon = np.array([0.01, 0.02])
    scatter_mag_by_file = {"proc_a.csv": 0.01, "proc_b.csv": 0.02}
    keyed, unmatched = _combine_err_with_ensemble_scatter_keyed(
        err_photon, ["proc_a.csv", "proc_b.csv"], scatter_mag_by_file,
    )
    assert not np.any(unmatched)
    assert keyed[0] == pytest.approx(_expected_combine(0.01, 0.01))
    assert keyed[1] == pytest.approx(_expected_combine(0.02, 0.02))


def test_keyed_matches_hand_computation_vector() -> None:
    """Vector case: keyed join matches per-epoch hand combine in rel-flux domain."""
    target_cid = "458415401545371264"
    files = [f"proc_{i:04d}.csv" for i in range(12)]
    scatter_mag = np.array(
        [0.005, 0.006, np.nan, 0.004, 0.0, 0.007, 0.003, 0.008, 0.002, 0.001, 0.009, 0.004]
    )
    err_photon = np.linspace(0.01, 0.05, 12)
    all_frames = _make_all_frames(target_cid, files)
    scatter_by_file = _ensemble_scatter_by_source_file(all_frames, target_cid, scatter_mag)
    keyed, unmatched = _combine_err_with_ensemble_scatter_keyed(
        err_photon, files, scatter_by_file, target_name="chi_h_sim"
    )
    expected = np.array(
        [
            _expected_combine(float(ep), float(sm)) if math.isfinite(sm) else float(ep)
            for ep, sm in zip(err_photon, scatter_mag, strict=True)
        ]
    )
    assert not np.any(unmatched)
    assert np.allclose(keyed, expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_adversarial_reorder_pairs_correct_epoch() -> None:
    """Reordered err rows pick scatter for their own source_file."""
    files_ordered = ["proc_a.csv", "proc_b.csv", "proc_c.csv"]
    scatter_vals = np.array([0.01, 0.02, 0.03])
    target_cid = "1000000000000000001"
    all_frames = _make_all_frames(target_cid, files_ordered)
    scatter_by_file = _ensemble_scatter_by_source_file(all_frames, target_cid, scatter_vals)

    files_reordered = ["proc_c.csv", "proc_a.csv", "proc_b.csv"]
    err_reordered = np.array([0.3, 0.1, 0.2])
    keyed, unmatched = _combine_err_with_ensemble_scatter_keyed(
        err_reordered, files_reordered, scatter_by_file
    )
    assert not np.any(unmatched)
    expected = np.array(
        [
            _expected_combine(0.3, 0.03),
            _expected_combine(0.1, 0.01),
            _expected_combine(0.2, 0.02),
        ]
    )
    assert np.allclose(keyed, expected)


def test_dropped_epoch_flagged_photon_only_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Missing scatter map entry: flagged, photon-only err, WARNING logged."""
    files = ["proc_a.csv", "proc_b.csv", "proc_c.csv"]
    err_photon = np.array([0.1, 0.2, 0.3])
    scatter_by_file = {"proc_a.csv": 0.01, "proc_b.csv": 0.02}

    with caplog.at_level(logging.WARNING):
        keyed, unmatched = _combine_err_with_ensemble_scatter_keyed(
            err_photon, files, scatter_by_file, target_name="V-test"
        )

    assert unmatched.tolist() == [False, False, True]
    assert keyed[0] == _expected_combine(0.1, 0.01)
    assert keyed[1] == _expected_combine(0.2, 0.02)
    assert keyed[2] == 0.3
    assert any("[G2-F004]" in r.message and "V-test" in r.message for r in caplog.records)


def test_nan_scatter_treated_as_zero_contribution() -> None:
    scatter_by_file = {"proc_x.csv": float("nan")}
    keyed, unmatched = _combine_err_with_ensemble_scatter_keyed(
        np.array([0.05]), ["proc_x.csv"], scatter_by_file
    )
    assert not unmatched.any()
    assert keyed[0] == 0.05


def test_empty_source_file_unmatched() -> None:
    scatter_by_file = {"proc_a.csv": 0.01}
    keyed, unmatched = _combine_err_with_ensemble_scatter_keyed(
        np.array([0.1, 0.2]), ["proc_a.csv", ""], scatter_by_file
    )
    assert unmatched.tolist() == [False, True]
    assert keyed[0] == _expected_combine(0.1, 0.01)
    assert keyed[1] == 0.2
