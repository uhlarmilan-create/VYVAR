"""G5-F003: candidate LC PDF figures use calibrated mag, not mag_inst."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from photometry_report import _resolve_candidate_lc_mag_for_plot


def test_candidate_lc_prefers_mag_calib_over_mag_inst() -> None:
    df = pd.DataFrame(
        {
            "bjd": [2460000.5, 2460000.6],
            "mag_inst": [10.0, 10.1],
            "mag_calib": [12.5, 12.6],
            "flag": ["normal", "normal"],
        }
    )
    ylab, mag = _resolve_candidate_lc_mag_for_plot(df)
    assert ylab == "mag_calib"
    assert mag.tolist() == [12.5, 12.6]


def test_candidate_lc_uses_mag_calib_ac_when_ac_on() -> None:
    df = pd.DataFrame(
        {
            "bjd": [2460000.5, 2460000.6],
            "mag_inst": [10.0, 10.1],
            "mag_calib": [12.5, 12.6],
            "mag_calib_ac": [12.3, 12.4],
            "ac_ok": [True, True],
            "flag": ["normal", "normal"],
        }
    )
    ylab, mag = _resolve_candidate_lc_mag_for_plot(df)
    assert ylab == "mag_calib_ac"
    assert mag.tolist() == [12.3, 12.4]


def test_candidate_lc_mixed_ac_uses_calib_label_and_ac_values() -> None:
    df = pd.DataFrame(
        {
            "mag_calib": [12.5, 12.6],
            "mag_calib_ac": [12.3, np.nan],
            "ac_ok": [True, False],
        }
    )
    ylab, mag = _resolve_candidate_lc_mag_for_plot(df)
    assert ylab == "mag_calib"
    assert mag.tolist() == [12.3, 12.6]


def test_candidate_lc_instrumental_only_when_no_calib() -> None:
    df = pd.DataFrame({"mag_inst": [10.0, 10.1]})
    ylab, mag = _resolve_candidate_lc_mag_for_plot(df)
    assert ylab == "mag_inst"
    assert mag.tolist() == [10.0, 10.1]


def test_candidate_lc_png_ylabel_calibrated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration: generated PNG path comes from plot with calibrated y-label."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.figure import Figure

    from photometry_report import _PhotometryReportBuilder

    phot = tmp_path / "photometry"
    lc_dir = phot / "lightcurves"
    cache = phot / "_report_cache"
    lc_dir.mkdir(parents=True)
    cache.mkdir(parents=True)

    cid = "Gaia DR3 1234567890123456789"
    lc_path = lc_dir / f"lightcurve_{cid[:18]}.csv"
    pd.DataFrame(
        {
            "bjd": [2460000.5, 2460000.6, 2460000.7],
            "mag_inst": [10.0, 10.1, 10.2],
            "mag_calib": [12.5, 12.6, 12.7],
            "flag": ["normal", "normal", "normal"],
        }
    ).to_csv(lc_path, index=False)

    captured_ylabel: list[str] = []
    _orig_savefig = Figure.savefig

    def _capture_savefig(self, _path, *args, **kwargs):  # noqa: ANN001
        ax = self.axes[0]
        captured_ylabel.append(str(ax.get_ylabel()))
        return _orig_savefig(self, _path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _capture_savefig)

    builder = _PhotometryReportBuilder.__new__(_PhotometryReportBuilder)
    png = builder._generate_candidate_lc_png(cid, phot, cache)

    assert png is not None
    assert captured_ylabel == ["mag_calib"]
    assert png.exists()


def test_candidate_lc_png_ac_case_ylabel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from photometry_report import _PhotometryReportBuilder

    phot = tmp_path / "photometry"
    lc_dir = phot / "lightcurves"
    cache = phot / "_report_cache"
    lc_dir.mkdir(parents=True)
    cache.mkdir(parents=True)

    cid = "Gaia DR3 9999999999999999999"
    lc_path = lc_dir / f"lightcurve_{cid[:18]}.csv"
    pd.DataFrame(
        {
            "bjd": [2460000.5, 2460000.6],
            "mag_inst": [10.0, 10.1],
            "mag_calib": [12.5, 12.6],
            "mag_calib_ac": [12.2, 12.3],
            "ac_ok": [True, True],
            "flag": ["normal", "normal"],
        }
    ).to_csv(lc_path, index=False)

    captured_ylabel: list[str] = []
    captured_ydata: list[np.ndarray] = []
    _orig_savefig = Figure.savefig

    def _capture_savefig(self, _path, *args, **kwargs):  # noqa: ANN001
        ax = self.axes[0]
        captured_ylabel.append(str(ax.get_ylabel()))
        captured_ydata.append(np.asarray(ax.collections[0].get_offsets()[:, 1]))
        return _orig_savefig(self, _path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", _capture_savefig)

    builder = _PhotometryReportBuilder.__new__(_PhotometryReportBuilder)
    png = builder._generate_candidate_lc_png(cid, phot, cache)

    assert png is not None
    assert captured_ylabel == ["mag_calib_ac"]
    assert captured_ydata[0].tolist() == [12.2, 12.3]
