# -*- coding: ascii -*-
"""OUTPUT-REVIEW-2026-10 step 4b: proc precision, frames.csv and the one proc reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proc_frame_store import (
    FRAME_TABLE_NAME,
    PROC_FRAME_COLS,
    ProcFrameStore,
    frame_values_of,
    proc_csv_columns,
    read_proc_csv,
    round_proc_columns,
    write_frame_tables,
    write_proc_csv,
)


def _frame(n: int = 3, jd: float = 2461161.464123) -> pd.DataFrame:
    return pd.DataFrame({
        "catalog_id": [str(1424733255874112896 + i) for i in range(n)],
        "x": [185.107878519984 + i for i in range(n)],
        "y": [9.456983295537876] * n,
        "ra_deg": [245.10169966195164] * n,
        "dao_flux": [1083884.2310264856, -12.345678912, np.nan][:n],
        "mag": [6.624114513397217, np.nan, 12.0][:n],
        "jd_mid": jd, "hjd_mid": jd + 0.002, "bjd_tdb_mid": jd + 0.003, "airmass": 1.2345678912,
        "aperture_r_px": 3.444912663453288, "fwhm_px_for_aperture": 2.5517871581135463,
        "sky_annulus_r_out_px": 13.26929322219044, "catalog_match_mode": "master_reference_sky",
    })


def test_precision_is_bounded_and_idempotent() -> None:
    r = round_proc_columns(_frame())
    assert r.loc[0, "x"] == 185.1079 and r.loc[0, "ra_deg"] == 245.1016997
    assert r.loc[0, "dao_flux"] == 1083884.2 and r.loc[1, "dao_flux"] == -12.345679
    assert abs(r.loc[0, "dao_flux"] / 1083884.2310264856 - 1) <= 5e-8  # 8 significant digits: <= 5.4e-8 mag
    assert np.isnan(r.loc[2, "dao_flux"]) and np.isnan(r.loc[1, "mag"])
    assert round_proc_columns(r).equals(r)


def test_frame_table_roundtrip_and_reader(tmp_path: Path) -> None:
    d = tmp_path / "lights" / "g_120_4"
    d.mkdir(parents=True)
    rows = {}
    for i, jd in enumerate((2461161.464123, 2461161.470001)):
        df = round_proc_columns(_frame(jd=jd))
        p = d / f"proc_f{i}.csv"
        rows[str(p)] = frame_values_of(df)
        write_proc_csv(df.drop(columns=list(PROC_FRAME_COLS)), p)
        assert not set(PROC_FRAME_COLS) & set(pd.read_csv(p, nrows=0).columns)
    write_frame_tables(rows)
    assert (d / FRAME_TABLE_NAME).is_file()
    full = read_proc_csv(d / "proc_f1.csv", dtype={"catalog_id": str})
    assert full["jd_mid"].tolist() == [2461161.470001] * 3
    assert full["catalog_match_mode"].iloc[0] == "master_reference_sky"
    assert full["airmass"].iloc[0] == pytest.approx(1.2345679, abs=1e-12)
    assert set(PROC_FRAME_COLS) <= set(proc_csv_columns(d / "proc_f0.csv"))
    sub = read_proc_csv(d / "proc_f0.csv", usecols=["catalog_id", "aperture_r_px"], dtype={"catalog_id": str})
    assert list(sub.columns) == ["catalog_id", "aperture_r_px"]
    with pytest.raises(ValueError):
        read_proc_csv(d / "proc_f0.csv", usecols=["catalog_id", "no_such_column"])
    cal = read_proc_csv(d / "proc_f0.csv", usecols=lambda c: c in {"x", "bjd_tdb_mid"})
    assert set(cal.columns) == {"x", "bjd_tdb_mid"}
    only = read_proc_csv(d / "proc_f0.csv", usecols=["fwhm_px_for_aperture"])  # frame values only
    assert list(only.columns) == ["fwhm_px_for_aperture"] and len(only) == 3
    st = ProcFrameStore.build(d)
    assert st.n_frames == 2 and "fwhm_px_for_aperture" in st.get_frame(str(d / "proc_f0.csv")).columns
    # a rewrite (e.g. the hybrid background pass) keeps the frame values out of the proc file
    write_proc_csv(full, d / "proc_f1.csv")
    assert "jd_mid" not in pd.read_csv(d / "proc_f1.csv", nrows=0).columns
    assert read_proc_csv(d / "proc_f1.csv")["jd_mid"].iloc[0] == 2461161.470001


def test_legacy_inline_proc_is_read_as_is(tmp_path: Path) -> None:
    p = tmp_path / "proc_old.csv"
    _frame().to_csv(p, index=False)
    df = read_proc_csv(p)
    assert df["jd_mid"].iloc[0] == 2461161.464123 and "aperture_r_px" in df.columns
    write_proc_csv(df, p)  # no frames.csv: the values stay inline
    assert "jd_mid" in pd.read_csv(p, nrows=0).columns
