"""G5-F011: canonical mag_calib_final (CT+AC) for export and publication figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from export_reports import _select_export_lc_rows
from photometry_core import compute_lc_rms_ooe, compute_mag_calib_final, save_lightcurve_csv
from photometry_report import _publication_lc_mag_column, _resolve_candidate_lc_mag_for_plot


def test_compute_mag_calib_final_neither_gate() -> None:
    mag = np.array([12.0, 12.1, 12.2])
    out = compute_mag_calib_final(mag, ct_ok=False, ac_ok=False)
    assert out.tolist() == mag.tolist()


def test_compute_mag_calib_final_ac_only() -> None:
    mag = np.array([12.0, 12.1])
    ac = mag + 0.02
    out = compute_mag_calib_final(
        mag, ct_ok=False, ac_ok=True, delta_m_corr=0.02, mag_calib_ac=ac
    )
    assert out.tolist() == ac.tolist()


def test_compute_mag_calib_final_ct_only() -> None:
    mag = np.array([12.0, 12.1])
    out = compute_mag_calib_final(mag, ct_ok=True, ct_correction=0.03, ac_ok=False)
    assert np.allclose(out, [12.03, 12.13])


def test_compute_mag_calib_final_both() -> None:
    mag = np.array([12.0, 12.1])
    out = compute_mag_calib_final(
        mag, ct_ok=True, ct_correction=0.03, ac_ok=True, delta_m_corr=-0.02
    )
    assert out.tolist() == [12.01, 12.11]


def test_save_lc_csv_mag_calib_final_matches_ac_when_ct_off(tmp_path: Path) -> None:
    n = 4
    bjd = np.linspace(2460000.5, 2460000.8, n)
    mag = np.array([12.0, 12.1, 12.2, 12.3])
    mag_ac = mag + 0.025
    flags = ["normal"] * n
    out = tmp_path / "lc.csv"
    save_lightcurve_csv(
        out,
        bjd,
        bjd,
        bjd,
        np.full(n, 1.1),
        None,
        mag,
        mag,
        mag,
        mag,
        mag_ac,
        np.zeros(n),
        np.full(n, 0.01),
        np.full(n, 5.0),
        flags,
        ["f1"] * n,
        ct_ok=False,
        ac_result={"ok": True, "delta_m_corr": 0.025, "scatter_mag": 0.01, "n_ref_stars": 5},
    )
    df = pd.read_csv(out)
    assert "mag_calib_final" in df.columns
    assert df["mag_calib_final"].tolist() == df["mag_calib_ac"].tolist()


def test_save_lc_csv_rounding_byte_identity_ct_off(tmp_path: Path) -> None:
    """Rounded CSV values: mag_calib_final == mag_calib_ac when CT off."""
    n = 3
    bjd = np.array([2460000.5, 2460000.6, 2460000.7])
    mag = np.array([12.123456, 12.234567, 12.345678])
    delta = -0.017891
    mag_ac = mag + delta
    save_lightcurve_csv(
        tmp_path / "lc2.csv",
        bjd,
        bjd,
        bjd,
        np.full(n, 1.0),
        None,
        mag,
        mag,
        mag,
        mag.copy(),
        mag_ac,
        np.zeros(n),
        np.full(n, 0.01),
        np.full(n, 5.0),
        ["normal"] * n,
        ["a"] * n,
        ct_ok=False,
        ac_result={"ok": True, "delta_m_corr": delta, "scatter_mag": 0.01, "n_ref_stars": 4},
    )
    df = pd.read_csv(tmp_path / "lc2.csv")
    assert df["mag_calib_final"].equals(df["mag_calib_ac"])


def test_export_select_uses_mag_calib_final() -> None:
    df = pd.DataFrame(
        {
            "bjd": [2460000.5, 2460000.6],
            "mag_calib": [12.5, 12.6],
            "mag_calib_ac": [12.3, 12.4],
            "mag_calib_final": [12.15, 12.25],
            "ac_ok": [True, True],
            "flag": ["normal", "normal"],
        }
    )
    out = _select_export_lc_rows(df)
    assert out["mag_calib"].tolist() == [12.15, 12.25]


def test_publication_mag_column_prefers_final() -> None:
    cols = ["bjd", "mag_calib", "mag_calib_ct", "mag_calib_final"]
    assert _publication_lc_mag_column(cols) == "mag_calib_final"


def test_candidate_plot_uses_mag_calib_final() -> None:
    df = pd.DataFrame(
        {
            "mag_calib": [12.5, 12.6],
            "mag_calib_ac": [12.3, 12.4],
            "mag_calib_final": [12.15, 12.25],
        }
    )
    ylab, mag = _resolve_candidate_lc_mag_for_plot(df)
    assert ylab == "mag_calib_final"
    assert mag.tolist() == [12.15, 12.25]


def test_lc_rms_invariant_under_ct_ac_constants() -> None:
    rng = np.random.default_rng(42)
    mag = 12.0 + 0.03 * rng.standard_normal(60)
    flags = ["normal"] * 60
    mag_final = mag + 0.04  # CT+AC constant
    rms_base = float(np.std(mag))
    rms_final = float(np.std(mag_final))
    assert rms_base == rms_final
    ooe_base = compute_lc_rms_ooe(mag, flags)
    ooe_final = compute_lc_rms_ooe(mag_final, flags)
    assert ooe_base == ooe_final


def test_export_byte_identical_legacy_ac_path_when_no_final_column() -> None:
    """Old CSVs without mag_calib_final still use AC precedence."""
    df = pd.DataFrame(
        {
            "bjd": [2460000.5],
            "mag_calib": [12.5],
            "mag_calib_ac": [12.3],
            "ac_ok": [True],
            "flag": ["normal"],
        }
    )
    out = _select_export_lc_rows(df)
    assert out["mag_calib"].tolist() == [12.3]


def test_export_ct_on_flows_to_rows() -> None:
    df = pd.DataFrame(
        {
            "bjd": [2460000.5],
            "mag_calib": [12.5],
            "mag_calib_ac": [12.45],
            "mag_calib_final": [12.40],
            "ct_ok": [True],
            "ac_ok": [True],
            "flag": ["normal"],
        }
    )
    out = _select_export_lc_rows(df)
    assert out["mag_calib"].tolist() == [12.40]
