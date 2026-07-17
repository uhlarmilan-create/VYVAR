"""G5-F003 / G5-F011: candidate LC PDF figures use canonical calibrated mag."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from photometry_report import _resolve_candidate_lc_mag_for_plot


def test_candidate_lc_prefers_mag_calib_final_over_legacy() -> None:
    df = pd.DataFrame(
        {
            "bjd": [2460000.5, 2460000.6],
            "mag_inst": [10.0, 10.1],
            "mag_calib": [12.5, 12.6],
            "mag_calib_ac": [12.3, 12.4],
            "mag_calib_final": [12.15, 12.25],
            "flag": ["normal", "normal"],
        }
    )
    ylab, mag = _resolve_candidate_lc_mag_for_plot(df)
    assert ylab == "mag_calib_final"
    assert mag.tolist() == [12.15, 12.25]


def test_candidate_lc_mag_calib_when_no_final() -> None:
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


def test_candidate_lc_instrumental_only_when_no_calib() -> None:
    df = pd.DataFrame({"mag_inst": [10.0, 10.1]})
    ylab, mag = _resolve_candidate_lc_mag_for_plot(df)
    assert ylab == "mag_inst"
    assert mag.tolist() == [10.0, 10.1]


def test_candidate_lc_png_ylabel_mag_calib_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib

    matplotlib.use("Agg")
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
            "mag_calib_final": [12.45, 12.55, 12.65],
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
    assert captured_ylabel == ["mag_calib_final"]
    assert png.exists()
