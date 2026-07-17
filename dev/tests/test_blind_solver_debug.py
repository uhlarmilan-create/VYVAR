"""Blind solver: debug instrumentation must not change decisions."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import KDTree

from config import AppConfig
from infolog import clear_log, get_lines
from vyvar_blind_solver import find_blind_hint


def _write_synthetic_index(path: Path, *, n_votes: int = 20) -> tuple[float, float]:
    """Minimal 3D index with clustered metadata at a known sky position."""
    truth_ra, truth_dec = 35.03, 57.14
    # Hash aligned to _synthetic_dao_stars triangle: r1≈0.530, r2≈0.729, log_L3_norm≈0.433
    hashes = np.tile([0.530, 0.729, 0.433], (n_votes, 1)).astype(np.float32)
    meta = np.tile([truth_ra, truth_dec], (n_votes, 1)).astype(np.float32)
    tree = KDTree(hashes)
    data = {
        "tree": tree,
        "metadata": meta,
        "hash_dim": 3,
        "log_L3_min": 1.0,
        "log_L3_max": 3.0,
        "tolerance": 0.02,
        "mag_limit": 14.0,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return truth_ra, truth_dec


def _synthetic_dao_stars() -> pd.DataFrame:
    """Three stars forming triangle with L3 in index range (plate scale 1.3 arcsec/px)."""
    # sides 30, 40, 50 px → L3=50 px → 65 arcsec → log10(65)≈1.81 → norm 0.405 in [1,3]
    return pd.DataFrame(
        {
            "x": [500.0, 530.0, 540.0],
            "y": [500.0, 500.0, 540.0],
            "flux": [1000.0, 900.0, 800.0],
        }
    )


@pytest.fixture
def synthetic_index(tmp_path: Path) -> Path:
    p = tmp_path / "triangles.pkl"
    _write_synthetic_index(p)
    return p


def test_find_blind_hint_debug_does_not_change_result(synthetic_index: Path) -> None:
    stars = _synthetic_dao_stars()
    kwargs = dict(
        dao_stars=stars,
        index_path=synthetic_index,
        n_top=30,
        min_votes=3,
        plate_scale_arcsec_per_px=1.3,
        fov_deg=1.5,
    )
    cfg_off = AppConfig()
    cfg_off.debug_platesolver = False
    cfg_on = AppConfig()
    cfg_on.debug_platesolver = True

    off = find_blind_hint(**kwargs, app_config=cfg_off)
    on = find_blind_hint(**kwargs, app_config=cfg_on)

    assert off == on


def test_find_blind_hint_debug_on_emits_log_l3_stats(synthetic_index: Path) -> None:
    stars = _synthetic_dao_stars()
    cfg = AppConfig()
    cfg.debug_platesolver = True
    clear_log()
    find_blind_hint(
        stars,
        synthetic_index,
        plate_scale_arcsec_per_px=1.3,
        fov_deg=1.5,
        app_config=cfg,
    )
    lines = "\n".join(get_lines())
    assert "DEBUG: Blind log_L3(img)" in lines


def test_find_blind_hint_debug_off_no_log_l3_stats(synthetic_index: Path) -> None:
    stars = _synthetic_dao_stars()
    cfg = AppConfig()
    cfg.debug_platesolver = False
    clear_log()
    find_blind_hint(
        stars,
        synthetic_index,
        plate_scale_arcsec_per_px=1.3,
        fov_deg=1.5,
        app_config=cfg,
    )
    lines = "\n".join(get_lines())
    assert "DEBUG: Blind log_L3(img)" not in lines
    assert "DEBUG: Blind votes near truth" not in lines


def test_find_blind_hint_debug_sink_and_truth_do_not_change_result(synthetic_index: Path) -> None:
    stars = _synthetic_dao_stars()
    kwargs = dict(
        dao_stars=stars,
        index_path=synthetic_index,
        n_top=30,
        min_votes=3,
        plate_scale_arcsec_per_px=1.3,
        fov_deg=1.5,
    )
    cfg = AppConfig()
    cfg.debug_platesolver = False
    baseline = find_blind_hint(**kwargs, app_config=cfg)

    cfg.debug_platesolver = True
    sink: dict = {}
    with_diag = find_blind_hint(
        **kwargs,
        app_config=cfg,
        debug_truth_radec=(35.03, 57.14),
        debug_sink=sink,
    )
    assert with_diag == baseline
    assert "passes" in sink
    assert len(sink["passes"]) >= 1


def test_find_blind_hint_debug_sink_collects_vote_stats(synthetic_index: Path) -> None:
    stars = _synthetic_dao_stars()
    cfg = AppConfig()
    cfg.debug_platesolver = True
    sink: dict = {}
    find_blind_hint(
        stars,
        synthetic_index,
        plate_scale_arcsec_per_px=1.3,
        fov_deg=1.5,
        app_config=cfg,
        debug_truth_radec=(35.03, 57.14),
        debug_sink=sink,
    )
    assert sink["passes"]
    with_votes = [p for p in sink["passes"] if p.get("n_votes", 0) >= 2]
    assert with_votes, "expected at least one pass with votes"
    p = with_votes[0]
    assert "match_mult_mean" in p
    assert "votes_near_truth_5deg" in p
    assert "votes" in p
