"""POINT-SOURCE-SHAPE-01 (F1, RUN522-CRASH-01): single-pixel events must not pass
pass-2 / forced-seed acceptance; real stars must, for well- and under-sampled PSFs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

from masterstar_gaia_accounting import (  # noqa: E402
    ForcedSeedAcceptParams,
    Pass2AcceptParams,
    dao_pass2_try_at_position,
    derive_point_source_q_floor,
    forced_seed_accept,
    forced_seed_measure_at_position,
    point_source_shape_ok,
    point_source_shape_q,
)

SIG = 3.0  # sky noise [ADU], image is background-subtracted


def _gauss(img, x0, y0, flux, fwhm):
    s = fwhm / 2.3548
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    # pixel-integrated approximation via 5x5 subsampling
    acc = np.zeros_like(img)
    for dx in np.linspace(-0.4, 0.4, 5):
        for dy in np.linspace(-0.4, 0.4, 5):
            acc += np.exp(-((xx + dx - x0) ** 2 + (yy + dy - y0) ** 2) / (2 * s * s))
    acc /= 25.0
    img += flux * acc / (2 * np.pi * s * s)


def _field(fwhm, seed=1):
    rng = np.random.default_rng(seed)
    h, w = 400, 400
    img = rng.normal(0.0, SIG, (h, w))
    ref = []
    for _ in range(60):  # bright reference stars on a grid-ish layout
        x, y = rng.uniform(30, w - 30), rng.uniform(30, h - 30)
        _gauss(img, x, y, rng.uniform(4000, 20000), fwhm)
        ref.append((x, y))
    ref = np.array(ref)
    return img, ref


def _free_spot(ref, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    while True:
        x, y = rng.uniform(40, 360), rng.uniform(40, 360)
        if np.min(np.hypot(ref[:, 0] - x, ref[:, 1] - y)) > 35:
            return float(round(x)), float(round(y))


def _floor(img, ref):
    return derive_point_source_q_floor(img, ref[:, 0], ref[:, 1], ref[:, 0], ref[:, 1], match_r_px=2.0)


@pytest.mark.parametrize("fwhm", [3.5, 1.3])
def test_floor_measured_from_stars(fwhm):
    img, ref = _field(fwhm)
    fl = _floor(img, ref)
    assert fl["n_ref"] >= 30 and fl["q_floor"] is not None
    # q of a star is far above a single pixel's q~0 for a sampled PSF; small but >0 when under-sampled
    assert fl["q_floor"] > (2.0 if fwhm > 3 else 0.2)


def test_hot_pixel_rejected_star_accepted_pass2():
    img, ref = _field(3.5, seed=2)
    fl = _floor(img, ref)
    hx, hy = _free_spot(ref, 3)
    img[int(hy), int(hx)] += 60.0  # 20 sigma single pixel
    sx, sy = _free_spot(np.vstack([ref, [[hx, hy]]]), 4)
    sx, sy = sx + 0.3, sy + 0.6
    _gauss(img, sx, sy, 400.0, 3.5)  # faint real star
    p_on = Pass2AcceptParams(sigma=4.0, center_tol_px=2.5, fwhm_px=3.5, shape_q_floor=fl["q_floor"])
    p_off = Pass2AcceptParams(sigma=4.0, center_tol_px=2.5, fwhm_px=3.5)
    h, w = img.shape
    assert dao_pass2_try_at_position(img, hx, hy, wpx=w, h=h, params=p_off)["accepted"]
    r = dao_pass2_try_at_position(img, hx, hy, wpx=w, h=h, params=p_on)
    assert not r["accepted"] and r["reason"] == "shape_not_point_source"
    assert dao_pass2_try_at_position(img, sx, sy, wpx=w, h=h, params=p_on)["accepted"]


def test_hot_pixel_rejected_star_accepted_forced_seed():
    img, ref = _field(3.5, seed=3)
    fl = _floor(img, ref)
    hx, hy = _free_spot(ref, 1)
    img[int(hy), int(hx)] += 150.0  # cosmic-ray-like single pixel
    sx, sy = _free_spot(np.vstack([ref, [[hx, hy]]]), 2)
    sx, sy = sx + 0.3, sy - 0.2
    _gauss(img, sx, sy, 600.0, 3.5)
    sp = ForcedSeedAcceptParams(centroid_max_px=2.5, snr_min=4.0, shape_q_floor=fl["q_floor"])
    sp_off = ForcedSeedAcceptParams(centroid_max_px=2.5, snr_min=4.0)
    m_hot = forced_seed_measure_at_position(img, hx, hy, fwhm_px=3.5, params=sp)
    assert forced_seed_accept(m_hot, params=sp_off)[0]  # passes without the gate
    ok_hot, why = forced_seed_accept(m_hot, params=sp)
    assert not ok_hot and why == "shape_not_point_source"
    ok_star, _ = forced_seed_accept(forced_seed_measure_at_position(img, sx, sy, fwhm_px=3.5, params=sp), params=sp)
    assert ok_star


def test_gate_off_without_floor_and_low_snr_not_rejected():
    assert point_source_shape_ok(0.0, 50.0, None)
    # at peak S/N 2 the shape is unmeasurable -> not rejected by this gate
    assert point_source_shape_ok(0.0, 2.0, 4.7)
    assert not point_source_shape_ok(0.0, 15.0, 4.7)


def test_q_statistic_values():
    img = np.zeros((21, 21))
    img[10, 10] = 100.0
    q, snr = point_source_shape_q(img, 10, 10, local_bg=0.0, local_sigma=5.0)
    assert q == pytest.approx(0.0) and snr == pytest.approx(20.0)


def test_too_few_references_disables_gate():
    img, ref = _field(3.5)
    fl = derive_point_source_q_floor(img, ref[:5, 0], ref[:5, 1], ref[:5, 0], ref[:5, 1], match_r_px=2.0)
    assert fl["q_floor"] is None and fl["n_ref"] <= 5
