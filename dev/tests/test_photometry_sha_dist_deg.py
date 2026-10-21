"""Tolerance path for comparison_stars_per_target _dist_deg (derived metadata)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from tests.photometry_sha import TOL_COMP_DIST_DEG, compare_photometry_science_meaningful


def _write_comp_csv(root: Path, setup: str, dist_values: list[float]) -> None:
    phot = root / "platesolve" / setup / "photometry"
    phot.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "target_catalog_id": "100",
            "catalog_id": str(200 + i),
            "name": f"comp{i}",
            "_dist_deg": v,
        }
        for i, v in enumerate(dist_values)
    ]
    pd.DataFrame(rows).to_csv(phot / "comparison_stars_per_target.csv", index=False)


def test_dist_deg_sub_ulp_diff_passes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root_a = Path(tmpdir) / "a"
        root_b = Path(tmpdir) / "b"
        base = 0.7995422726901543
        delta = 1e-14
        _write_comp_csv(root_a, "NoFilter_60_2", [base])
        _write_comp_csv(root_b, "NoFilter_60_2", [base - delta])
        rep = compare_photometry_science_meaningful(root_a, root_b, setups=("NoFilter_60_2",))
        setup = rep["setups"]["NoFilter_60_2"]
        assert setup["shared_row_value_diffs"] == 0
        assert setup["comp_csv_ok"] is True
        assert delta < TOL_COMP_DIST_DEG


def test_dist_deg_above_tolerance_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root_a = Path(tmpdir) / "a"
        root_b = Path(tmpdir) / "b"
        base = 1.0
        delta = 1e-9
        _write_comp_csv(root_a, "NoFilter_60_2", [base])
        _write_comp_csv(root_b, "NoFilter_60_2", [base - delta])
        rep = compare_photometry_science_meaningful(root_a, root_b, setups=("NoFilter_60_2",))
        setup = rep["setups"]["NoFilter_60_2"]
        assert setup["shared_row_value_diffs"] == 1
        assert setup["comp_csv_ok"] is False
        assert delta > TOL_COMP_DIST_DEG
