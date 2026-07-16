"""LABBE-DET: empty-aperture sigma + star-list canonicalization are process-stable."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from photometry_core import (
    _ensemble_scatter_by_source_file,
    measure_empty_aperture_sigma_bkg,
)

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    data = rng.normal(1000.0, 25.0, size=(180, 180))
    xs = np.array([40.0, 90.0, 130.0, 55.0, 150.0])
    ys = np.array([45.0, 95.0, 120.0, 140.0, 60.0])
    return data, xs, ys


def test_labbe_order_invariant_shuffled_stars() -> None:
    data, xs, ys = _synthetic_frame()
    seed = 424242
    a, na, ra = measure_empty_aperture_sigma_bkg(
        data, xs, ys, 3.5, 6.0, 10.0, n_apertures=40, min_valid=10, seed=seed
    )
    perm = np.random.default_rng(7).permutation(xs.size)
    b, nb, rb = measure_empty_aperture_sigma_bkg(
        data,
        xs[perm],
        ys[perm],
        3.5,
        6.0,
        10.0,
        n_apertures=40,
        min_valid=10,
        seed=seed,
    )
    assert ra == "" and rb == ""
    assert na == nb
    assert a == b


def test_labbe_identical_across_pythonhashseed_subprocesses() -> None:
    """Same synthetic inputs → identical sigma under different PYTHONHASHSEED."""
    code = textwrap.dedent(
        """
        import numpy as np
        from photometry_core import measure_empty_aperture_sigma_bkg
        rng = np.random.default_rng(42)
        data = rng.normal(1000.0, 25.0, size=(180, 180))
        xs = np.array([40.0, 90.0, 130.0, 55.0, 150.0])
        ys = np.array([45.0, 95.0, 120.0, 140.0, 60.0])
        sig, n, reason = measure_empty_aperture_sigma_bkg(
            data, xs, ys, 3.5, 6.0, 10.0, n_apertures=40, min_valid=10, seed=424242
        )
        assert reason == ""
        print(f"{sig:.17g}|{n}")
        """
    )
    outs: list[str] = []
    for hashseed in ("0", "1", "42"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hashseed
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=True,
            env=env,
        )
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] == outs[2], outs


def test_ensemble_scatter_map_invariant_to_row_order() -> None:
    """SEM keyed join must follow sorted source_file, not DataFrame insertion order."""
    tid = "T1"
    rows_a = [
        {"catalog_id": tid, "source_file": "proc_b.csv"},
        {"catalog_id": tid, "source_file": "proc_a.csv"},
        {"catalog_id": tid, "source_file": "proc_c.csv"},
    ]
    rows_b = list(reversed(rows_a))
    # Scatter indices assume _get_lc order = sorted source_file: a, b, c
    scatter = np.array([0.01, 0.02, 0.03], dtype=np.float64)
    map_a = _ensemble_scatter_by_source_file(pd.DataFrame(rows_a), tid, scatter)
    map_b = _ensemble_scatter_by_source_file(pd.DataFrame(rows_b), tid, scatter)
    assert map_a == map_b
    assert map_a["proc_a.csv"] == 0.01
    assert map_a["proc_b.csv"] == 0.02
    assert map_a["proc_c.csv"] == 0.03
