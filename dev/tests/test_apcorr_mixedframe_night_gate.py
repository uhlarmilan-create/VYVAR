"""APCORR-MIXEDFRAME-ALLORNOTHING: night-level COG gate unit tests."""

from __future__ import annotations

import math

import pandas as pd

from photometry_core import (
    evaluate_cog_night_apcorr_gate,
    read_flux_from_csv,
)


def _frame(cog_ok: bool | None, *, flux: float = 100.0, apcorr: float = 110.0) -> pd.DataFrame:
    """Minimal proc-frame row set for night-gate / read_flux tests."""
    if cog_ok is None:
        return pd.DataFrame(
            {
                "catalog_id": ["100", "200"],
                "dao_flux": [flux, flux * 2],
                "aperture_r_px": [3.0, 3.0],
                "x": [10.0, 20.0],
                "y": [10.0, 20.0],
            }
        )
    return pd.DataFrame(
        {
            "catalog_id": ["100", "200"],
            "dao_flux": [flux, flux * 2],
            "dao_flux_apcorr": [apcorr, apcorr * 2],
            "ac_factor": [apcorr / flux, apcorr / flux],
            "cog_ok": [cog_ok, cog_ok],
            "aperture_r_px": [3.0, 3.0],
            "x": [10.0, 20.0],
            "y": [10.0, 20.0],
        }
    )


def test_night_gate_disabled_untouched() -> None:
    frames = [_frame(True), _frame(False)]
    gate = evaluate_cog_night_apcorr_gate(frames, enabled=False)
    assert gate["use_apcorr_flux"] is False
    assert gate["cog_night_fallback"] is False
    assert gate["n_frames"] == 0
    assert gate["n_without_cog_ok"] == 0


def test_night_gate_mixed_cog_ok_falls_back_whole_night() -> None:
    frames = [_frame(True), _frame(True), _frame(False)]
    gate = evaluate_cog_night_apcorr_gate(frames, enabled=True)
    assert gate["use_apcorr_flux"] is False
    assert gate["cog_night_fallback"] is True
    assert gate["n_frames"] == 3
    assert gate["n_without_cog_ok"] == 1


def test_night_gate_missing_cog_column_falls_back() -> None:
    frames = [_frame(True), _frame(None)]
    gate = evaluate_cog_night_apcorr_gate(frames, enabled=True)
    assert gate["use_apcorr_flux"] is False
    assert gate["cog_night_fallback"] is True
    assert gate["n_without_cog_ok"] == 1


def test_night_gate_all_ok_applies_cog() -> None:
    frames = [_frame(True), _frame(True)]
    gate = evaluate_cog_night_apcorr_gate(frames, enabled=True)
    assert gate["use_apcorr_flux"] is True
    assert gate["cog_night_fallback"] is False
    assert gate["n_without_cog_ok"] == 0
    assert gate["n_frames"] == 2


def test_read_flux_respects_night_gate_flag(tmp_path) -> None:
    """With use_apcorr_flux=False, rows keep raw dao_flux even when cog_ok=True."""
    df = _frame(True, flux=100.0, apcorr=150.0)
    csv_path = tmp_path / "proc_0001.csv"
    df.to_csv(csv_path, index=False)
    apertures = {"100": 3.0, "200": 3.0}
    out_off = read_flux_from_csv(
        csv_path,
        ["100", "200"],
        apertures,
        csv_df=df,
        use_apcorr_flux=False,
    )
    out_on = read_flux_from_csv(
        csv_path,
        ["100", "200"],
        apertures,
        csv_df=df,
        use_apcorr_flux=True,
    )
    assert not out_off.empty and not out_on.empty
    # mag_inst from flux: brighter (larger) flux -> smaller mag
    row_off = out_off.loc[out_off["catalog_id"].astype(str) == "100"].iloc[0]
    row_on = out_on.loc[out_on["catalog_id"].astype(str) == "100"].iloc[0]
    assert math.isclose(float(row_off["flux_raw"]), 100.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(float(row_on["flux_raw"]), 100.0, rel_tol=0, abs_tol=1e-9)
    # Standard path uses raw flux; COG path uses apcorr flux for mag_inst routing
    mag_off = float(row_off["mag_inst"])
    mag_on = float(row_on["mag_inst"])
    assert math.isfinite(mag_off) and math.isfinite(mag_on)
    assert mag_on < mag_off  # 150 ADU brighter than 100 ADU
