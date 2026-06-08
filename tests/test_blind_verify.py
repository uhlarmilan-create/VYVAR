"""Blind solver geometric verification tests."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import fit_wcs_from_points

from config import AppConfig
from vyvar_blind_solver import BlindCandidate, _MatchHit
from vyvar_platesolver import _verify_blind_candidates


def _verify_cfg(**overrides) -> AppConfig:
    cfg = AppConfig()
    cfg.blind_verify_inmemory_catalog = False
    cfg.blind_verify_early_fraction = 0.0
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


def _build_scenario(*, truth_ra: float, truth_dec: float, false_ra: float, false_dec: float):
    rng = np.random.default_rng(7)
    n = 50
    xs = 200.0 + rng.uniform(0, 600, n)
    ys = 200.0 + rng.uniform(0, 600, n)
    dao_df = pd.DataFrame({"x": xs, "y": ys, "flux": np.linspace(1000, 100, n)})

    img3 = dao_df.iloc[:3][["x", "y"]].to_numpy(dtype=np.float64)
    cat3 = np.array(
        [
            [truth_ra + 0.02, truth_dec + 0.01],
            [truth_ra - 0.01, truth_dec + 0.02],
            [truth_ra + 0.01, truth_dec - 0.02],
        ],
        dtype=np.float64,
    )
    world = SkyCoord(ra=cat3[:, 0] * 1.0, dec=cat3[:, 1] * 1.0, unit="deg")
    wcs_truth = fit_wcs_from_points((img3[:, 0], img3[:, 1]), world, projection="TAN")
    wcs_truth.array_shape = (900, 1200)
    cat3_ra, cat3_dec = wcs_truth.all_pix2world(img3[:, 0], img3[:, 1], 0)
    cat3 = np.column_stack([cat3_ra, cat3_dec])
    fc_ra, fc_dec = wcs_truth.all_pix2world(600.0, 450.0, 0)
    ra_proj, dec_proj = wcs_truth.all_pix2world(
        dao_df["x"].to_numpy(dtype=np.float64),
        dao_df["y"].to_numpy(dtype=np.float64),
        0,
    )
    catalog = [{"ra": float(r), "dec": float(d), "g_mag": 12.0} for r, d in zip(ra_proj, dec_proj)]

    good = BlindCandidate(
        center_ra=float(fc_ra),
        center_dec=float(fc_dec),
        img_px=img3.copy(),
        cat_sky=cat3.copy(),
        hash_dist=0.0005,
        vote_count=2,
    )
    false_cat3 = cat3.copy()
    false_cat3[:, 0] = false_ra + (cat3[:, 0] - truth_ra)
    false_cat3[:, 1] = false_dec + (cat3[:, 1] - truth_dec)
    bad = BlindCandidate(
        center_ra=false_ra,
        center_dec=false_dec,
        img_px=img3.copy(),
        cat_sky=false_cat3,
        hash_dist=0.002,
        vote_count=50,
    )
    return dao_df, catalog, good, bad


def test_verify_accepts_truth_rejects_false_cluster() -> None:
    truth_ra, truth_dec = 35.03, 57.14
    false_ra, false_dec = 196.4, 38.4
    dao_df, catalog, good, bad = _build_scenario(
        truth_ra=truth_ra, truth_dec=truth_dec, false_ra=false_ra, false_dec=false_dec
    )

    cfg = _verify_cfg(
        blind_verify_min_matches=3,
        blind_verify_min_fraction=0.05,
        blind_verify_match_tol_px=25.0,
    )

    def _fake_query(_root, **kwargs):
        _ = kwargs
        return catalog

    with patch("database.query_local_gaia", side_effect=_fake_query):
        hint = _verify_blind_candidates(
            [bad, good],
            dao_df=dao_df,
            gaia_db_path="dummy.db",
            fov_deg=1.2,
            naxis1=1200,
            naxis2=900,
            pixel_pitch_um=5.4,
            focal_length_mm=600.0,
            max_cat_mag=16.0,
            app_config=cfg,
        )

    assert hint is not None
    dra = (hint[0] - truth_ra) * np.cos(np.radians(truth_dec))
    ddec = hint[1] - truth_dec
    assert float(np.hypot(dra, ddec)) < 2.0


def test_verify_debug_sink_does_not_change_winner() -> None:
    truth_ra, truth_dec = 35.03, 57.14
    dao_df, catalog, good, _bad = _build_scenario(
        truth_ra=truth_ra, truth_dec=truth_dec, false_ra=196.0, false_dec=38.0
    )
    cfg = _verify_cfg(
        blind_verify_min_matches=6,
        blind_verify_min_fraction=0.10,
        blind_verify_match_tol_px=6.0,
    )

    def _fake_query(_root, **kwargs):
        _ = kwargs
        return catalog

    with patch("database.query_local_gaia", side_effect=_fake_query):
        base = _verify_blind_candidates(
            [good],
            dao_df=dao_df,
            gaia_db_path="dummy.db",
            fov_deg=1.2,
            naxis1=1200,
            naxis2=900,
            pixel_pitch_um=5.4,
            focal_length_mm=600.0,
            max_cat_mag=16.0,
            app_config=cfg,
        )
        sink: dict = {}
        with_sink = _verify_blind_candidates(
            [good],
            dao_df=dao_df,
            gaia_db_path="dummy.db",
            fov_deg=1.2,
            naxis1=1200,
            naxis2=900,
            pixel_pitch_um=5.4,
            focal_length_mm=600.0,
            max_cat_mag=16.0,
            app_config=cfg,
            debug_sink=sink,
        )
    assert base == with_sink
    assert sink.get("verified_candidates")


def test_cluster_ransac_verify_pooled_triangles() -> None:
    """Pooled cluster correspondences + RANSAC beat a single noisy triangle WCS."""
    truth_ra, truth_dec = 241.54, 50.30
    rng = np.random.default_rng(11)
    n = 80
    xs = 200.0 + rng.uniform(0, 1600, n)
    ys = 200.0 + rng.uniform(0, 1200, n)
    dao_df = pd.DataFrame({"x": xs, "y": ys, "flux": np.linspace(1000, 100, n)})

    img3 = dao_df.iloc[:3][["x", "y"]].to_numpy(dtype=np.float64)
    cat3 = np.array(
        [
            [truth_ra + 0.02, truth_dec + 0.01],
            [truth_ra - 0.01, truth_dec + 0.02],
            [truth_ra + 0.01, truth_dec - 0.02],
        ],
        dtype=np.float64,
    )
    world = SkyCoord(ra=cat3[:, 0] * u.deg, dec=cat3[:, 1] * u.deg, frame="icrs")
    wcs_truth = fit_wcs_from_points((img3[:, 0], img3[:, 1]), world, projection="TAN")
    wcs_truth.array_shape = (1400, 1800)
    ra_proj, dec_proj = wcs_truth.all_pix2world(xs, ys, 0)
    catalog = [{"ra": float(r), "dec": float(d), "g_mag": 12.0} for r, d in zip(ra_proj, dec_proj)]

    members: list[_MatchHit] = []
    for t in range(10):
        idx = rng.choice(n, size=3, replace=False)
        tri_px = dao_df.iloc[idx][["x", "y"]].to_numpy(dtype=np.float64)
        tri_ra, tri_dec = wcs_truth.all_pix2world(tri_px[:, 0], tri_px[:, 1], 0)
        tri_sky = np.column_stack([tri_ra, tri_dec])
        fc_ra, fc_dec = wcs_truth.all_pix2world(900.0, 700.0, 0)
        members.append(
            _MatchHit(
                center_ra=float(fc_ra),
                center_dec=float(fc_dec),
                img_px=tri_px,
                cat_sky=tri_sky,
                hash_dist=0.001 + 0.0001 * t,
            )
        )

    cluster = BlindCandidate(
        center_ra=float(members[0].center_ra),
        center_dec=float(members[0].center_dec),
        img_px=members[0].img_px.copy(),
        cat_sky=members[0].cat_sky.copy(),
        hash_dist=float(members[0].hash_dist),
        vote_count=len(members),
        cluster_members=members,
    )

    cfg = _verify_cfg(
        blind_verify_min_matches=8,
        blind_verify_min_fraction=0.05,
        blind_verify_match_tol_px=25.0,
    )

    def _fake_query(_root, **kwargs):
        _ = kwargs
        return catalog

    with patch("database.query_local_gaia", side_effect=_fake_query):
        hint = _verify_blind_candidates(
            [cluster],
            dao_df=dao_df,
            gaia_db_path="dummy.db",
            fov_deg=1.2,
            naxis1=1800,
            naxis2=1400,
            pixel_pitch_um=5.4,
            focal_length_mm=600.0,
            max_cat_mag=16.0,
            app_config=cfg,
        )

    assert hint is not None
    dra = (hint[0] - truth_ra) * np.cos(np.radians(truth_dec))
    ddec = hint[1] - truth_dec
    assert float(np.hypot(dra, ddec)) < 2.0


def test_early_exit_same_winner_as_full_sweep() -> None:
    """Early-exit must not change the accepted hint on a solvable field."""
    truth_ra, truth_dec = 35.03, 57.14
    false_ra, false_dec = 196.4, 38.4
    dao_df, catalog, good, bad = _build_scenario(
        truth_ra=truth_ra, truth_dec=truth_dec, false_ra=false_ra, false_dec=false_dec
    )
    good.vote_count = 50
    good.hash_dist = 0.0001
    bad.vote_count = 2
    bad.hash_dist = 0.05

    def _fake_query(_root, **kwargs):
        _ = kwargs
        return catalog

    cfg_fast = _verify_cfg(
        blind_verify_min_matches=3,
        blind_verify_min_fraction=0.05,
        blind_verify_match_tol_px=25.0,
        blind_verify_early_fraction=0.15,
    )
    cfg_full = _verify_cfg(
        blind_verify_min_matches=3,
        blind_verify_min_fraction=0.05,
        blind_verify_match_tol_px=25.0,
        blind_verify_early_fraction=0.0,
        blind_verify_early_accept=9999,
        blind_verify_early_floor=9999,
    )

    with patch("database.query_local_gaia", side_effect=_fake_query):
        sink: dict = {}
        hint_fast = _verify_blind_candidates(
            [bad, good],
            dao_df=dao_df,
            gaia_db_path="dummy.db",
            fov_deg=1.2,
            naxis1=1200,
            naxis2=900,
            pixel_pitch_um=5.4,
            focal_length_mm=600.0,
            max_cat_mag=16.0,
            app_config=cfg_fast,
            debug_sink=sink,
        )
        hint_full = _verify_blind_candidates(
            [bad, good],
            dao_df=dao_df,
            gaia_db_path="dummy.db",
            fov_deg=1.2,
            naxis1=1200,
            naxis2=900,
            pixel_pitch_um=5.4,
            focal_length_mm=600.0,
            max_cat_mag=16.0,
            app_config=cfg_full,
        )

    assert hint_fast is not None and hint_full is not None
    assert hint_fast == hint_full
    assert any(r.get("accepted") for r in sink.get("verified_candidates", []))
    assert len(sink.get("verified_candidates", [])) <= 2
