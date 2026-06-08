"""Tests for canonical pre-cal proc-CSV resolution (PROC_CSV_GLOB / list_proc_csvs)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from comp_qa_core import load_proc_pivot
from proc_frame_store import PROC_CSV_GLOB, list_proc_csvs


def _write_frame_csv(path: Path, *, catalog_id: str, flux: float, frame_name: str) -> None:
    pd.DataFrame(
        {
            "catalog_id": [catalog_id, "999999999999999999"],
            "dao_flux": [flux, 100.0],
            "source_file": [frame_name, frame_name],
            "bjd_tdb_mid": [2459000.0 + flux * 1e-6, 2459000.1],
        }
    ).to_csv(path, index=False)


def test_list_proc_csvs_pre_cal_and_light_naming() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_frame_csv(root / "proc_Chi_H_001.csv", catalog_id="111", flux=1000.0, frame_name="f1.fits")
        _write_frame_csv(
            root / "proc_V842_Her_Light_064.csv",
            catalog_id="222",
            flux=2000.0,
            frame_name="f2.fits",
        )
        found = [p.name for p in list_proc_csvs(root)]
        assert found == ["proc_Chi_H_001.csv", "proc_V842_Her_Light_064.csv"]
        assert PROC_CSV_GLOB == "proc_*.csv"


def test_load_proc_pivot_pre_cal_basenames() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        target_id = "458468246824340992"
        comp_ids = ["458308955066917120", "458309573551916800"]
        for i, cid in enumerate([target_id, *comp_ids]):
            _write_frame_csv(
                root / f"proc_Chi_H_{i:03d}.csv",
                catalog_id=cid,
                flux=1000.0 + i,
                frame_name=f"frame_{i}.fits",
            )
        flux_w, _time_df = load_proc_pivot(root, {target_id, *comp_ids})
        assert not flux_w.empty
        assert target_id in flux_w.columns
        for cid in comp_ids:
            assert cid in flux_w.columns
        assert len(flux_w.index) == 3
