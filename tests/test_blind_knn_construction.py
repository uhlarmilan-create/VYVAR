"""Blind solver: image kNN triangles + index per-cell capping."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import KDTree

import sys
from pathlib import Path

_GAIA = Path(__file__).resolve().parents[1] / "GAIA_DR3"
if str(_GAIA) not in sys.path:
    sys.path.insert(0, str(_GAIA))

from build_blind_index import cap_brightest_per_cell  # noqa: E402
from vyvar_blind_solver import (  # noqa: E402
    cap_brightest_per_pixel_cell,
    iter_local_knn_triangle_indices,
)


def _builder_style_triangles(coords: np.ndarray, k_neighbors: int) -> set[tuple[int, int, int]]:
    n = len(coords)
    k = min(k_neighbors, n)
    tree = KDTree(coords)
    _, all_idx = tree.query(coords, k=k)
    all_idx = np.atleast_2d(all_idx)
    combos = list(itertools.combinations(range(k), 3))
    out: set[tuple[int, int, int]] = set()
    for i in range(n):
        neighbor_idx = all_idx[i]
        for c in combos:
            i_tri = neighbor_idx[list(c)]
            if i != int(np.min(i_tri)):
                continue
            out.add(tuple(sorted(int(x) for x in i_tri)))
    return out


def test_image_knn_matches_builder_dedup() -> None:
    rng = np.random.default_rng(3)
    coords = rng.uniform(0, 500, (12, 2))
    k = 8
    img = {tuple(sorted(t)) for t in iter_local_knn_triangle_indices(coords, k_neighbors=k)}
    bld = _builder_style_triangles(coords, k)
    assert img == bld


def test_knn_fewer_than_all_pairs() -> None:
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 400, (10, 2))
    n_knn = sum(1 for _ in iter_local_knn_triangle_indices(coords, k_neighbors=8))
    n_pairs = len(list(itertools.combinations(range(10), 3)))
    assert n_knn < n_pairs
    assert n_knn > 0


def test_cap_brightest_per_cell() -> None:
    df = pd.DataFrame(
        {
            "ra": [0.0, 0.01, 0.02, 5.0],
            "dec": [0.0, 0.0, 0.0, 0.0],
            "g_mag": [10.0, 11.0, 12.0, 9.0],
        }
    )
    capped = cap_brightest_per_cell(df, cell_deg=0.5, stars_per_cell=2)
    assert len(capped) == 3
    assert set(capped["g_mag"].tolist()) == {9.0, 10.0, 11.0}


def test_cap_brightest_per_pixel_cell() -> None:
    df = pd.DataFrame(
        {
            "x": [10.0, 20.0, 25.0, 800.0],
            "y": [10.0, 15.0, 18.0, 800.0],
            "flux": [100.0, 50.0, 80.0, 200.0],
        }
    )
    capped = cap_brightest_per_pixel_cell(df, cell_px=100.0, stars_per_cell=2)
    assert len(capped) == 3
    assert capped["flux"].max() == 200.0


def test_per_cell_degenerates_single_bucket() -> None:
    from vyvar_blind_solver import _select_blind_image_stars

    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "x": rng.uniform(0, 500, n),
            "y": rng.uniform(0, 400, n),
            "flux": rng.uniform(1, 100, n),
        }
    )
    idx = {"cell_deg": 1.0, "stars_per_cell": 12, "k_neighbors": 8}
    stars, _ = _select_blind_image_stars(
        df,
        idx,
        plate_scale_arcsec_per_px=1.3,
        app_config=None,
        n_top=80,
        fov_deg=1.0,
        use_rig_prior=True,
        log_L3_max=3.0,
        tri_k=8,
    )
    assert len(stars) == 12


def test_dbscan_truth_cluster_representative() -> None:
    from vyvar_blind_solver import BlindCandidate, _MatchHit, _hits_to_candidates_dbscan

    truth_ra, truth_dec = 241.54, 50.30
    hits: list[_MatchHit] = []
    for i in range(11):
        hits.append(
            _MatchHit(
                center_ra=truth_ra + 0.01 * (i - 5),
                center_dec=truth_dec + 0.008 * (i % 3),
                img_px=np.array([[100.0, 100.0], [110.0, 105.0], [105.0, 115.0]]),
                cat_sky=np.array([[truth_ra, truth_dec]] * 3),
                hash_dist=0.001 + 0.0001 * i,
            )
        )
    for i in range(30):
        hits.append(
            _MatchHit(
                center_ra=10.0 + i * 0.1,
                center_dec=20.0,
                img_px=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
                cat_sky=np.array([[10.0, 20.0]] * 3),
                hash_dist=0.00001,
            )
        )
    cands = _hits_to_candidates_dbscan(hits, app_config=None, top_n=15)
    assert cands
    near = [
        c
        for c in cands
        if abs(c.center_ra - truth_ra) < 2.0 and abs(c.center_dec - truth_dec) < 2.0
    ]
    assert near, "truth cluster should yield a verify candidate"
    assert near[0].vote_count >= 4


def test_index_k_neighbors_default_legacy() -> None:
    from vyvar_blind_solver import _index_k_neighbors

    assert _index_k_neighbors({}) == 8
    assert _index_k_neighbors({"k_neighbors": 6}) == 6
