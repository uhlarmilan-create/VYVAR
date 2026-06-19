"""Alignment / astroalign reproducibility (Step 2)."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from utils import seeded_numpy_default_rng, VYVAR_RANDOM_SEED
from vyvar_alignment_frame import _alignment_run_astroalign_points

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_point_sets() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    src = rng.uniform(60, 440, (96, 2)).astype(np.float32)
    tgt = src + rng.normal(0, 0.35, (96, 2)).astype(np.float32)
    return src, tgt


def test_seeded_numpy_default_rng_makes_astroalign_repeatable():
    import astroalign

    src, tgt = _synthetic_point_sets()

    def run_once() -> np.ndarray:
        with seeded_numpy_default_rng(VYVAR_RANDOM_SEED):
            t, _ = astroalign.find_transform(src, tgt, max_control_points=55)
        return np.asarray(t.params, dtype=np.float64)

    p1 = run_once()
    p2 = run_once()
    assert np.array_equal(p1, p2)


def test_seeded_astroalign_identical_across_subprocesses():
    """Cross-process: patched find_transform must return the same matrix."""
    code = textwrap.dedent(
        f"""
        import numpy as np
        from utils import seeded_numpy_default_rng, VYVAR_RANDOM_SEED
        import astroalign
        rng = np.random.default_rng(17)
        src = rng.uniform(60, 440, (96, 2)).astype(np.float32)
        tgt = src + rng.normal(0, 0.35, (96, 2)).astype(np.float32)
        with seeded_numpy_default_rng(VYVAR_RANDOM_SEED):
            t, _ = astroalign.find_transform(src, tgt, max_control_points=55)
        print(np.array2string(t.params, precision=12, separator=','))
        """
    )
    outs: list[str] = []
    for _ in range(3):
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=True,
        )
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] == outs[2]


def test_alignment_run_astroalign_points_byte_identical():
    src, tgt = _synthetic_point_sets()
    img_s = np.random.RandomState(0).rand(512, 512).astype(np.float32)
    img_t = np.random.RandomState(1).rand(512, 512).astype(np.float32)
    kw = {
        "source_pts": src,
        "target_pts": tgt,
        "image_source": img_s,
        "image_target": img_t,
        "max_control_points": 50,
    }
    out1, err1 = _alignment_run_astroalign_points(**kw)
    out2, err2 = _alignment_run_astroalign_points(**kw)
    assert err1 is None and err2 is None
    assert out1 is not None and out2 is not None
    assert np.array_equal(out1, out2)


def test_alignment_run_astroalign_points_insufficient_stars():
    """Negative: too few points still fails (seeding does not force a bogus success)."""
    src = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    tgt = np.array([[11.0, 10.0], [21.0, 20.0]], dtype=np.float32)
    img = np.zeros((64, 64), dtype=np.float32)
    out, err = _alignment_run_astroalign_points(
        source_pts=src,
        target_pts=tgt,
        image_source=img,
        image_target=img,
        max_control_points=50,
    )
    assert out is None
    assert err is not None
