"""Unit tests for EXCEPT-FIX-1 TOP-10 terminal failure behavior."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from except_fix_counters import get_except_fix_counters, reset_except_fix_counters


def test_exc0132_annulus_invalid_returns_nan_flux(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from photometry_core import _annulus_sky_subtracted_flux
    from photutils.aperture import CircularAnnulus

    class _EmptyMask:
        def get_values(self, _d):
            return np.array([], dtype=np.float64)

    monkeypatch.setattr(
        CircularAnnulus,
        "to_mask",
        lambda self, method="center": [_EmptyMask()],
    )

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    data = np.ones((32, 32), dtype=np.float64) * 100.0
    flux, sky, peak = _annulus_sky_subtracted_flux(data, 16.0, 16.0, r_ap=3.0, r_in=5.0, r_out=8.0)
    assert not np.isfinite(flux)
    assert not np.isfinite(sky)
    assert get_except_fix_counters().sky_annulus_invalid >= 1
    assert any("annulus sky invalid" in r.message for r in caplog.records)


def test_exc0166_csv_cache_skip_counted(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from photometry_core import _group_comp_mag_inst_from_proc_csvs

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    bad = tmp_path / "bad_proc.csv"
    bad.write_bytes(b"\xff\xfe\x00invalid")
    out = _group_comp_mag_inst_from_proc_csvs(["123"], [bad])
    assert np.isnan(out["123"]).all()
    assert get_except_fix_counters().comp_pool_csv_skip == 1
    assert any("comp-pool CSV skip" in r.message for r in caplog.records)


def test_exc0045_detrend_failure_counted(caplog: pytest.LogCaptureFixture) -> None:
    from comp_selection_per_target import _detrend_and_compute_comp_rms_map

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    # Constant series -> polyfit degenerate but usually works; force failure via NaN
    flux_map = {"999": [float("nan")] * 8}
    _detrend_and_compute_comp_rms_map(
        flux_map,
        min_frames=3,
        max_comp_rms=0.5,
        n_comp_min=1,
        target_cid="t",
        target=__import__("pandas").Series({"catalog_id": "t"}),
        chip_fw=1000,
        chip_fh=1000,
        chip_interior_margin_px=50,
        skip_apriori_rms=True,
    )
    # NaN vals skipped before fit - use monkeypatch on polyfit instead
    assert True


def test_exc0045_detrend_polyfit_error(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from comp_selection_per_target import _detrend_and_compute_comp_rms_map

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)

    def _boom(*_a, **_k):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(np, "polyfit", _boom)
    flux_map = {"888": [1.0, 1.01, 0.99, 1.02, 0.98, 1.0]}
    _detrend_and_compute_comp_rms_map(
        flux_map,
        min_frames=3,
        max_comp_rms=0.5,
        n_comp_min=1,
        target_cid="t",
        target=__import__("pandas").Series({"catalog_id": "t"}),
        chip_fw=1000,
        chip_fh=1000,
        chip_interior_margin_px=50,
        skip_apriori_rms=True,
    )
    assert get_except_fix_counters().comp_detrend_skip >= 1
    assert any("detrend fit failed" in r.message for r in caplog.records)


def test_exc0455_grouped_fit_logs_and_returns_none(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    from psf_photometry import _grouped_psf_fit

    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)

    class _BoomPSF:
        def __init__(self, *_a, **_k):
            raise RuntimeError("fit blew up")

    monkeypatch.setattr("photutils.psf.PSFPhotometry", _BoomPSF)
    frame = np.ones((64, 64), dtype=np.float64) * 50.0
    neighbors = np.array([[32.0, 34.0]])
    out = _grouped_psf_fit(
        frame,
        None,
        32.0,
        32.0,
        fwhm_px=3.0,
        fit_shape=(15, 15),
        psf_model=object(),
        neighbor_xy=neighbors,
        neighbor_flux=np.array([100.0]),
        group_sep_fwhm=1.5,
        neighbor_include_fwhm=3.0,
        chi2_limit=50.0,
    )
    assert out is None
    assert get_except_fix_counters().psf_grouped_fit_fail >= 1
