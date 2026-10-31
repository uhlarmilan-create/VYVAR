"""Tests for dilution.py (TODO-GS11 Step A)."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from dilution import compute_dilution_factor, flux_from_gmag, query_gaia_neighbors

# DILUTION-ONE-2026-10: the light-curve aperture, its annulus and the measured Moffat PSF (arcsec)
_GEOM = dict(fwhm_arcsec=3.0, beta=2.7, annulus_arcsec=(8.1, 15.6))


def _nb(g: float, dx_arcsec: float, dec0: float = 40.0, ra0: float = 210.0) -> dict:
    """A neighbour dx arcsec east of (ra0, dec0)."""
    return {"g_mag": g, "ra": ra0 + dx_arcsec / 3600.0 / math.cos(math.radians(dec0)), "dec": dec0,
            "sep_arcsec": abs(dx_arcsec)}


def test_flux_from_gmag_ratio() -> None:
    f10 = flux_from_gmag(10.0)
    f125 = flux_from_gmag(12.5)
    assert f10 > 0 and f125 > 0
    assert abs(f10 / f125 - 10.0) < 0.01


def test_dilution_no_neighbors() -> None:
    with patch("dilution.query_gaia_neighbors", return_value=[]):
        r = compute_dilution_factor(210.0, 40.0, 12.0, 4.0, "dummy.db", catalog_id=123, **_GEOM)
    assert r["dilution_factor"] == 1.0
    assert r["dilution_delta_mag"] == 0.0
    assert r["n_neighbors"] == 0


def test_dilution_is_the_comp_blend_model() -> None:
    """G07-33: the target dilution is comp_blend.contamination_fractions (Moffat, annulus median) on the
    light-curve aperture; an equal neighbour on top gives D = 0.5."""
    import numpy as np

    from comp_blend import contamination_fractions

    with patch("dilution.query_gaia_neighbors", return_value=[_nb(12.0, 0.0)]):
        r0 = compute_dilution_factor(210.0, 40.0, 12.0, 4.0, "dummy.db", **_GEOM)
    assert abs(r0["dilution_factor"] - 0.5) < 1e-3
    with patch("dilution.query_gaia_neighbors", return_value=[_nb(12.0, 4.5)]):
        r = compute_dilution_factor(210.0, 40.0, 12.0, 4.0, "dummy.db", **_GEOM)
    c = float(contamination_fractions(np.zeros((1, 2)), np.array([1.0]), np.array([[4.5, 0.0]]), np.array([1.0]),
                                      r_ap_px=4.0, fwhm_px=3.0, beta=2.7, annulus_px=(8.1, 15.6))[0])
    assert r["dilution_factor"] == pytest.approx(1.0 / (1.0 + c), rel=1e-6)
    assert r["n_neighbors"] == 1


def test_dilution_requires_psf_and_annulus() -> None:
    with pytest.raises(ValueError):
        compute_dilution_factor(210.0, 40.0, 12.0, 4.0, "dummy.db", fwhm_arcsec=3.0, beta=float("nan"),
                                annulus_arcsec=(8.1, 15.6))


def test_dilution_faint_neighbor_excluded() -> None:
    """Neighbor 6 mag fainter: mag_limit_delta=5 excludes it via SQL mag_limit."""
    with patch("dilution.query_gaia_neighbors", return_value=[]) as mock_q:
        r = compute_dilution_factor(
            210.0,
            40.0,
            10.0,
            4.0,
            "dummy.db",
            mag_limit_delta=5.0,
            **_GEOM,
        )
        mock_q.assert_called_once()
        _args, kwargs = mock_q.call_args
        assert kwargs["mag_limit"] == pytest.approx(15.0)
    assert r["dilution_factor"] == 1.0


def test_dilution_nan_gmag() -> None:
    r = compute_dilution_factor(210.0, 40.0, float("nan"), 4.0, "dummy.db", **_GEOM)
    assert r["dilution_factor"] == 1.0
    assert r["dilution_delta_mag"] == 0.0
    assert r["dilution_skipped"] is True


def test_query_neighbors_excludes_self() -> None:
    self_row = {
        "source_id": 1497525907795379456,
        "ra": 210.594396,
        "dec": 39.413370,
        "g_mag": 13.34,
        "bp_mag": None,
        "rp_mag": None,
        "bp_rp": 0.98,
    }
    other_row = {
        "source_id": 9999999999999999999,
        "ra": 210.595,
        "dec": 39.414,
        "g_mag": 15.0,
        "bp_mag": None,
        "rp_mag": None,
        "bp_rp": None,
    }

    with patch("dilution.query_local_gaia", return_value=[self_row, other_row]), patch(
        "dilution.Path.is_file", return_value=True
    ):
        neighbors = query_gaia_neighbors(
            210.594396,
            39.413370,
            30.0,
            "dummy.db",
            mag_limit=20.0,
            exclude_source_id=1497525907795379456,
        )
    assert len(neighbors) == 1
    assert neighbors[0]["source_id"] != 1497525907795379456


def test_dilution_delta_mag_formula() -> None:
    d = 0.5
    expected = -2.5 * math.log10(d)
    assert abs(expected - 0.752574989) < 0.001
    neighbors = [_nb(10.0, 0.0, dec0=0.0, ra0=0.0)]
    with patch("dilution.query_gaia_neighbors", return_value=neighbors):
        r = compute_dilution_factor(0.0, 0.0, 10.0, 4.0, "dummy.db", **_GEOM)
    assert abs(r["dilution_delta_mag"] - expected) < 0.02


@pytest.mark.skipif(
    not Path(r"C:\ASTRO\python\VYVAR\catalogs\vyvar_gaia_dr3.db").is_file(),
    reason="local Gaia DB not present",
)
def test_dilution_real_db() -> None:
    db = r"C:\ASTRO\python\VYVAR\catalogs\vyvar_gaia_dr3.db"
    r = compute_dilution_factor(
        210.59459,
        39.41332,
        13.34097957611084,
        60.0,
        db,
        catalog_id=1497525907795379456,
        **_GEOM,
    )
    required = {
        "dilution_factor",
        "dilution_delta_mag",
        "n_neighbors",
        "neighbor_flux_sum",
        "aperture_arcsec",
        "search_radius_arcsec",
    }
    assert required <= set(r.keys())
    assert 0.0 < r["dilution_factor"] <= 1.0
