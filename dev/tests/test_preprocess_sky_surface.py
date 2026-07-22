"""T3: preprocess order-2 sky surface subtract (shared helper)."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from pipeline import _fit_subtract_preprocess_sky_surface


def _synthetic_gradient_with_stars(shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    grad = 1200.0 + 0.15 * xx.astype(np.float32) + 0.08 * yy.astype(np.float32)
    stars = np.zeros_like(grad)
    for cy, cx, amp in ((80, 90, 800.0), (170, 200, 1200.0), (120, 180, 600.0)):
        stars += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 2.0**2)))
    return (grad + stars).astype(np.float32)


def test_order0_bypass_leaves_frame_identical() -> None:
    data = _synthetic_gradient_with_stars()
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=0)
    assert stats["sky_surface_applied"] is False
    np.testing.assert_array_equal(out, data)


def test_order2_flattens_gradient_only_frame() -> None:
    h, w = 256, 256
    yy, xx = np.mgrid[0:h, 0:w]
    data = (1000.0 + 0.05 * xx + 0.03 * yy).astype(np.float32)
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=2)
    assert stats["sky_surface_applied"] is True
    assert stats["sky_surface_p2p_adu"] > 1.0
    assert not np.allclose(out, data)


def test_order2_preserves_star_fluxes() -> None:
    data = _synthetic_gradient_with_stars()
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=2)
    assert stats["sky_surface_applied"] is True
    assert stats["sky_surface_p2p_adu"] > 10.0

    for cy, cx in ((80, 90), (170, 200), (120, 180)):
        peak = float(out[cy, cx])
        local = out[cy - 6 : cy + 7, cx - 6 : cx + 7]
        assert peak - float(np.nanmedian(local)) > 200.0


def test_sky_surface_helper_writes_stats_dict() -> None:
    data = _synthetic_gradient_with_stars((128, 128))
    _, stats = _fit_subtract_preprocess_sky_surface(data, order=2)
    assert stats["sky_surface_order"] == 2
    assert stats["sky_surface_applied"] is True
    assert stats["sky_surface_p2p_adu"] > 0.0
