"""Tests for dilution.py (TODO-GS11 Step A)."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from dilution import (
    compute_dilution_batch,
    compute_dilution_factor,
    flux_from_gmag,
    query_gaia_neighbors,
)


def test_flux_from_gmag_ratio() -> None:
    f10 = flux_from_gmag(10.0)
    f125 = flux_from_gmag(12.5)
    assert f10 > 0 and f125 > 0
    assert abs(f10 / f125 - 10.0) < 0.01


def test_dilution_no_neighbors() -> None:
    with patch("dilution.query_gaia_neighbors", return_value=[]):
        r = compute_dilution_factor(
            210.0,
            40.0,
            12.0,
            4.0,
            "dummy.db",
            catalog_id=123,
        )
    assert r["dilution_factor"] == 1.0
    assert r["dilution_delta_mag"] == 0.0
    assert r["n_neighbors"] == 0


def test_dilution_equal_neighbor() -> None:
    neighbors = [{"g_mag": 12.0, "sep_arcsec": 2.0}]
    with patch("dilution.query_gaia_neighbors", return_value=neighbors):
        r = compute_dilution_factor(210.0, 40.0, 12.0, 4.0, "dummy.db")
    assert abs(r["dilution_factor"] - 0.5) < 0.001
    assert abs(r["dilution_delta_mag"] - 0.75) < 0.01
    assert r["n_neighbors"] == 1


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
        )
        mock_q.assert_called_once()
        _args, kwargs = mock_q.call_args
        assert kwargs["mag_limit"] == pytest.approx(15.0)
    assert r["dilution_factor"] == 1.0


def test_dilution_nan_gmag() -> None:
    r = compute_dilution_factor(210.0, 40.0, float("nan"), 4.0, "dummy.db")
    assert r["dilution_factor"] == 1.0
    assert r["dilution_delta_mag"] == 0.0


def test_dilution_batch_basic() -> None:
    def _side_effect(ra, dec, g, ap, db, **kwargs):
        if g == 11.0:
            return {
                "dilution_factor": 0.8,
                "dilution_delta_mag": -0.2,
                "n_neighbors": 1,
                "neighbor_flux_sum": 0.25,
                "aperture_arcsec": ap,
                "search_radius_arcsec": ap,
            }
        return {
            "dilution_factor": 1.0,
            "dilution_delta_mag": 0.0,
            "n_neighbors": 0,
            "neighbor_flux_sum": 0.0,
            "aperture_arcsec": ap,
            "search_radius_arcsec": ap,
        }

    stars = [
        {"catalog_id": "1", "ra_deg": 1.0, "dec_deg": 2.0, "g_mag": 12.0},
        {"catalog_id": "2", "ra_deg": 1.1, "dec_deg": 2.1, "g_mag": 11.0},
        {"catalog_id": "3", "ra_deg": 1.2, "dec_deg": 2.2, "g_mag": 13.0},
    ]
    with patch("dilution.compute_dilution_factor", side_effect=_side_effect):
        out = compute_dilution_batch(stars, 4.0, "dummy.db")
    assert len(out) == 3
    blended = [r for r in out if r["dilution_factor"] < 1.0]
    assert len(blended) == 1
    assert blended[0]["catalog_id"] == "2"


def test_dilution_batch_missing_gmag_key() -> None:
    with patch("dilution.query_gaia_neighbors", return_value=[]):
        out = compute_dilution_batch(
            [{"catalog_id": "99", "ra_deg": 10.0, "dec_deg": 20.0, "phot_g_mean_mag": 12.5}],
            4.0,
            "dummy.db",
        )
    assert out[0]["dilution_factor"] == 1.0


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
    neighbors = [{"g_mag": 10.0}]
    with patch("dilution.query_gaia_neighbors", return_value=neighbors):
        r = compute_dilution_factor(0.0, 0.0, 10.0, 4.0, "dummy.db")
    assert abs(r["dilution_delta_mag"] - expected) < 0.02


@pytest.mark.skipif(
    not Path(r"C:\ASTRO\python\VYVAR\GAIA_DR3\vyvar_gaia_dr3.db").is_file(),
    reason="local Gaia DB not present",
)
def test_dilution_real_db() -> None:
    db = r"C:\ASTRO\python\VYVAR\GAIA_DR3\vyvar_gaia_dr3.db"
    r = compute_dilution_factor(
        210.59459,
        39.41332,
        13.34097957611084,
        60.0,
        db,
        catalog_id=1497525907795379456,
        search_radius_arcsec=60.0,
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
