"""T3: preprocess order-2 sky surface subtract (shared helper)."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

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
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=2, calm_adu=500.0)
    assert stats["sky_surface_applied"] is True
    out_flat = float(residual_large_scale_p99_adu(out, order=2))
    assert out_flat < 0.15 * in_flat
    np.testing.assert_allclose(float(np.nanmedian(out)), float(np.nanmedian(data)), rtol=0, atol=6.0)


def test_order1_gradient_removal_ratio_and_pedestal() -> None:
    """P-10 regression: must remove gradient, not double it (ratio ~2 before fix)."""
    data = _gradient_only_frame(order=1)
    in_flat = float(residual_large_scale_p99_adu(data, order=1))
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=1, calm_adu=500.0)
    assert stats["sky_surface_applied"] is True
    out_flat = float(residual_large_scale_p99_adu(out, order=1))
    assert in_flat > 20.0
    assert out_flat < 1.0
    assert out_flat < 0.15 * in_flat
    np.testing.assert_allclose(float(np.nanmedian(out)), float(np.nanmedian(data)), rtol=0, atol=6.0)


def _fit_subtract_preprocess_sky_surface_prefix_bug(
    data: np.ndarray,
    *,
    order: int,
    fwhm_px: float = 2.5,
    calm_adu: float = 500.0,
) -> tuple[np.ndarray, dict]:
    """Pre-P-10 bug reproduction: fit z = bg_median - work (inverted sign)."""
    import numpy as np
    from astropy.stats import sigma_clip, sigma_clipped_stats
    from photutils.detection import DAOStarFinder

    from pipeline import DAO_STAR_FINDER_NO_ROUNDNESS_FILTER

    arr = np.asarray(data, dtype=np.float32)
    order_i = min(2, max(1, int(order)))
    h, w = arr.shape
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
    work = np.where(finite, arr, fill)
    mask = np.ones((h, w), dtype=bool)
    margin = 40
    if h > 2 * margin and w > 2 * margin:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False
    fwhm_eff = max(1.2, float(fwhm_px))
    _, med, std = sigma_clipped_stats(work, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((work - med).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    thr = max(3.0 * float(std), 1e-6)
    finder = DAOStarFinder(
        fwhm=fwhm_eff,
        threshold=float(thr),
        n_brightest=5000,
        **DAO_STAR_FINDER_NO_ROUNDNESS_FILTER,
    )
    tbl = finder(data0)
    stamp_r = int(max(4, round(3.5 * fwhm_eff)))
    if tbl is not None and len(tbl) > 0:
        r2 = stamp_r * stamp_r
        for row in tbl:
            cy = int(round(float(row["y_centroid"])))
            cx = int(round(float(row["x_centroid"])))
            if not (0 <= cy < h and 0 <= cx < w):
                continue
            y0, y1 = max(0, cy - stamp_r), min(h, cy + stamp_r + 1)
            x0, x1 = max(0, cx - stamp_r), min(w, cx + stamp_r + 1)
            yy_l, xx_l = np.ogrid[y0:y1, x0:x1]
            mask[y0:y1, x0:x1] &= (yy_l - cy) ** 2 + (xx_l - cx) ** 2 > r2
    bg_median, _, _ = sigma_clipped_stats(work, mask=mask, sigma=3.0, maxiters=5)
    calm_thr = max(5.0, float(calm_adu))
    fit_mask = mask & (np.abs(work - float(bg_median)) < calm_thr)
    step = 4
    yy_s, xx_s = np.mgrid[0:h:step, 0:w:step]
    z_s = (float(bg_median) - work[::step, ::step]).astype(np.float64)  # BUG: inverted sign
    m_s = fit_mask[::step, ::step]
    use_mask = m_s & np.isfinite(z_s)
    min_coef = (order_i + 1) * (order_i + 2) // 2
    if int(np.count_nonzero(use_mask)) < min_coef + 10:
        return arr.copy(), {"sky_surface_applied": False}
    z_samples = z_s[use_mask]
    clipped = sigma_clip(z_samples, sigma=3.0, maxiters=5, masked=True)
    good = ~clipped.mask
    x_fit = xx_s[use_mask][good].astype(np.float64)
    y_fit = yy_s[use_mask][good].astype(np.float64)
    z_fit = z_samples[good]
    cols: list[np.ndarray] = []
    for i in range(order_i + 1):
        for j in range(order_i + 1 - i):
            cols.append((x_fit**i) * (y_fit**j))
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), z_fit, rcond=None)
    yy_f, xx_f = np.mgrid[0:h, 0:w]
    cols_f: list[np.ndarray] = []
    x_flat = xx_f.ravel().astype(np.float64)
    y_flat = yy_f.ravel().astype(np.float64)
    for i in range(order_i + 1):
        for j in range(order_i + 1 - i):
            cols_f.append((x_flat**i) * (y_flat**j))
    surf = (np.column_stack(cols_f) @ coef).reshape(h, w).astype(np.float32)
    out = (work - surf).astype(np.float32)
    out = np.where(finite, out, np.nan).astype(np.float32)
    return out, {"sky_surface_applied": True}


def test_order1_prefix_bug_doubles_gradient_p2p_ratio_near_two() -> None:
    """Independent expectation (R2): pre-fix code doubles large-scale gradient (~2.0 ratio)."""
    data = _gradient_only_frame(order=1)
    in_flat = float(residual_large_scale_p99_adu(data, order=1))
    buggy, _ = _fit_subtract_preprocess_sky_surface_prefix_bug(data, order=1)
    fixed, _ = _fit_subtract_preprocess_sky_surface(data, order=1, calm_adu=500.0)
    buggy_flat = float(residual_large_scale_p99_adu(buggy, order=1))
    fixed_flat = float(residual_large_scale_p99_adu(fixed, order=1))
    ratio_buggy = buggy_flat / in_flat
    ratio_fixed = fixed_flat / in_flat
    assert 1.85 <= ratio_buggy <= 2.15, f"pre-fix ratio={ratio_buggy:.3f}, expected ~2.0"
    assert ratio_fixed < 0.15, f"post-fix ratio={ratio_fixed:.3f}, expected <<1"


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
