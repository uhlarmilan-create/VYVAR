"""CT-2026-10: polynomial colour term, order by BIC, robust fit, ensemble-weighted correction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

from color_transform import fit_colour_term  # noqa: E402


def _data(coeffs, n=400, noise=0.02, seed=1):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.4, 2.0, n)
    y = np.polyval(coeffs[::-1], c) + rng.normal(0, noise, n)
    return c, y


def test_linear_truth_gives_order_1():
    c, y = _data([0.3, 0.12])
    f = fit_colour_term(c, y)
    assert f.order == 1 and abs(f.coeffs[0] - 0.12) < 0.01


def test_quadratic_truth_gives_order_2_and_variables_do_not_pull():
    c, y = _data([0.3, 0.12, -0.08])
    y[:20] += 0.5  # variable / blended stars: down-weighted, not clipped
    f = fit_colour_term(c, y)
    assert f.order == 2
    pred = f.poly(np.array([0.5, 1.8])) - f.poly(np.array([1.0, 1.0]))
    true = (0.12 * np.array([0.5, 1.8]) - 0.08 * np.array([0.5, 1.8]) ** 2) - (0.12 - 0.08)
    assert np.allclose(pred, true, atol=0.01)


def test_no_colour_dependence_gives_order_0():
    c, y = _data([0.3])
    assert fit_colour_term(c, y).order == 0


def test_correction_uses_weighted_mean_of_p_not_p_of_mean():
    c, y = _data([0.0, 0.0, -0.2], noise=0.001)
    f = fit_colour_term(c, y)
    comps, w = [0.5, 1.9], [1.0, 1.0]
    dm = f.correction(1.2, comps, w)
    expect = float(f.poly(1.2)) - float(np.mean(f.poly(np.array(comps))))
    assert abs(dm - expect) < 1e-12
    assert abs(dm - (float(f.poly(1.2)) - float(f.poly(np.mean(comps))))) > 0.05


def test_too_few_stars():
    c, y = _data([0.3, 0.1], n=15)
    assert fit_colour_term(c, y) is None
