"""T3: preprocess order-2 sky surface subtract (shared helper)."""

from __future__ import annotations

import numpy as np

from invariants_runtime import residual_large_scale_p99_adu
from pipeline import _fit_subtract_preprocess_sky_surface


def _synthetic_gradient_with_stars(shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    grad = 1200.0 + 0.15 * xx.astype(np.float32) + 0.08 * yy.astype(np.float32)
    stars = np.zeros_like(grad)
    for cy, cx, amp in ((80, 90, 800.0), (170, 200, 1200.0), (120, 180, 600.0)):
        stars += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 2.0**2)))
    return (grad + stars).astype(np.float32)


def _gradient_only_frame(h: int = 256, w: int = 256, *, order: int = 1) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    pedestal = 1200.0
    if order == 1:
        return (pedestal + 0.12 * xx.astype(np.float32) + 0.08 * yy.astype(np.float32)).astype(np.float32)
    return (
        pedestal
        + 0.0008 * (xx.astype(np.float32) - w / 2) ** 2
        + 0.0006 * (yy.astype(np.float32) - h / 2) ** 2
        + 0.05 * xx.astype(np.float32)
    ).astype(np.float32)


def test_order0_bypass_leaves_frame_identical() -> None:
    data = _synthetic_gradient_with_stars()
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=0)
    assert stats["sky_surface_applied"] is False
    np.testing.assert_array_equal(out, data)


def test_order2_flattens_gradient_only_frame() -> None:
    data = _gradient_only_frame(order=2)
    in_flat = float(residual_large_scale_p99_adu(data, order=2))
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=2)
    assert stats["sky_surface_applied"] is True
    out_flat = float(residual_large_scale_p99_adu(out, order=2))
    assert out_flat < 0.15 * in_flat
    np.testing.assert_allclose(float(np.nanmedian(out)), float(np.nanmedian(data)), rtol=0, atol=6.0)


def test_order1_gradient_removal_ratio_and_pedestal() -> None:
    """P-10 regression: must remove gradient, not double it (ratio ~2 before fix)."""
    data = _gradient_only_frame(order=1)
    in_flat = float(residual_large_scale_p99_adu(data, order=1))
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=1)
    assert stats["sky_surface_applied"] is True
    out_flat = float(residual_large_scale_p99_adu(out, order=1))
    assert in_flat > 20.0
    assert out_flat < 1.0
    assert out_flat < 0.15 * in_flat
    np.testing.assert_allclose(float(np.nanmedian(out)), float(np.nanmedian(data)), rtol=0, atol=6.0)


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


def test_strong_gradient_with_faint_stars_is_removed() -> None:
    """SKY-MASK-2026-10 (D6): a 600 ADU gradient with noise and many faint stars. The former +-100 ADU
    'calm' cut left a large part of such a gradient in the frame; the source mask removes it."""
    rng = np.random.default_rng(5)
    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w]
    sky = 1000.0 + 1.2 * xx + 0.0 * yy
    img = sky + rng.normal(0.0, 8.0, (h, w))
    for _ in range(400):
        cy, cx = rng.uniform(10, h - 10), rng.uniform(10, w - 10)
        img += rng.uniform(30, 3000) * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 1.5**2)))
    out, stats = _fit_subtract_preprocess_sky_surface(img.astype(np.float32), order=1, fwhm_px=3.5)
    assert stats["sky_surface_applied"] is True
    flat = out - (img - sky)  # what is left of the sky after subtraction
    left = float(np.percentile(flat[40:-40, 40:-40], 99) - np.percentile(flat[40:-40, 40:-40], 1))
    assert left < 0.03 * 600.0
    assert stats["sky_surface_source_mask_px"] > 0
