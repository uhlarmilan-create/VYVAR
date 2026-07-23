"""G2-F003: GS11 dilution aperture layered fallback (no fixed 3.0 px)."""

from __future__ import annotations

import logging

import pytest

from photometry_core import (
    _aperture_radius_from_snr_table,
    _resolve_photometric_aperture_px_for_gs11,
    compute_snr_optimal_aperture_table,
)


def _sample_snr_table() -> dict:
    return compute_snr_optimal_aperture_table(
        fwhm_px=3.0,
        sky_adu_per_px=100.0,
        gain=1.0,
        read_noise=10.0,
    )


def test_gs11_aperture_from_map_when_present() -> None:
    apertures = {"1234567890123456789": 4.25}
    ap, src = _resolve_photometric_aperture_px_for_gs11(
        "1234567890123456789",
        apertures,
        float("nan"),
        None,
        aperture_fwhm_factor=2.5,
        fwhm_px=3.0,
    )
    assert src == "map"
    assert ap == 4.25
    assert ap != 3.0


def test_gs11_aperture_derived_from_snr_table_not_3() -> None:
    snr = _sample_snr_table()
    mag = 12.0
    expected = _aperture_radius_from_snr_table(
        mag,
        snr,
        aperture_fwhm_factor=2.5,
        fwhm_px=3.0,
    )
    ap, src = _resolve_photometric_aperture_px_for_gs11(
        "missing_cid",
        {},
        mag,
        snr,
        aperture_fwhm_factor=2.5,
        fwhm_px=3.0,
    )
    assert src == "snr_derived"
    assert ap == expected
    assert ap != 3.0


def test_gs11_aperture_skip_when_undeterminable(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        ap, src = _resolve_photometric_aperture_px_for_gs11(
            "no_map_cid",
            {},
            float("nan"),
            None,
            aperture_fwhm_factor=2.5,
            fwhm_px=3.0,
        )
    assert ap is None
    assert src == "unavailable"


def test_no_fixed_3_in_dilution_resolver_source() -> None:
    import inspect

    from tests.cython_compat import skip_if_compiled

    skip_if_compiled(
        "photometry_core",
        "inspect.getsource requires interpreted photometry_core.py",
    )
    src = inspect.getsource(_resolve_photometric_aperture_px_for_gs11)
    assert "3.0" not in src
