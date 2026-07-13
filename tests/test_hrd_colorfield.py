"""Unit tests for catalog-color HRD field rendering (hrd_colorfield.py)."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from hrd_colorfield import (
    BP_RP_DOMAIN,
    TEFF_MAX_K,
    TEFF_MIN_K,
    apply_chroma_boost,
    apply_chroma_snr_gate,
    background_neutrality_grid,
    build_colorfield_caption,
    build_local_background_maps,
    build_star_exclusion_mask,
    compose_catalog_color_rgb,
    hrd_color_bg_box_px_from_cfg,
    hrd_color_chroma_boost_from_cfg,
    hrd_color_chroma_snr_from_cfg,
    hrd_color_highlight_mode_from_cfg,
    hrd_color_saturation_from_cfg,
    hrd_color_white_point_from_cfg,
    make_tapered_gaussian_stamp,
    render_catalog_color_field,
    splat_chroma_layer,
    teff_from_bp_rp,
    teff_to_srgb_chroma,
)


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_teff_from_bp_rp_monotonic_and_clamps():
    grid = np.linspace(BP_RP_DOMAIN[0], BP_RP_DOMAIN[1], 200)
    teff = teff_from_bp_rp(grid)
    assert np.all(np.diff(teff) <= 0)
    assert float(teff[0]) <= TEFF_MAX_K + 1
    assert float(teff[-1]) >= TEFF_MIN_K - 1
    assert math.isfinite(float(teff_from_bp_rp(-0.4)))
    assert math.isfinite(float(teff_from_bp_rp(4.5)))


def test_teff_to_srgb_sanity():
    sat = 0.85
    r3000, g3000, b3000 = teff_to_srgb_chroma(3000.0, saturation=sat)
    assert r3000 > g3000 > b3000

    r6500, g6500, b6500 = teff_to_srgb_chroma(6500.0, saturation=sat)
    mx = max(r6500, g6500, b6500)
    mn = min(r6500, g6500, b6500)
    assert mx / mn < 1.15

    r15000, g15000, b15000 = teff_to_srgb_chroma(15000.0, saturation=sat)
    assert b15000 > r15000

    temps = np.linspace(3000.0, 15000.0, 50)
    rb = []
    for t in temps:
        r, _g, b = teff_to_srgb_chroma(t, saturation=sat)
        rb.append(r / b if b > 0 else math.inf)
    assert all(rb[i] > rb[i + 1] for i in range(len(rb) - 1))


def test_splat_blending_two_stars():
    shape = (32, 32)
    red = np.array([1.4, 0.9, 0.85])
    blue = np.array([0.85, 0.95, 1.4])
    xs = np.array([14.0, 18.0])
    ys = np.array([16.0, 16.0])
    rgbs = np.stack([red, blue])
    amps = np.array([1.0, 1.0])
    chroma = splat_chroma_layer(shape, xs, ys, rgbs, amps, sigma_px=2.0)
    mid = chroma[16, 16]
    assert np.allclose(mid, (red + blue) / 2.0, rtol=0.05, atol=0.05)
    corner = chroma[2, 2]
    assert np.allclose(corner, [1.0, 1.0, 1.0], atol=1e-6)


def test_hue_preserving_scale_preserves_rb_ratio():
    l = np.array([[1.0]])
    chroma = np.array([[[1.8, 1.0, 0.7]]])
    rgb = compose_catalog_color_rgb(l, chroma, highlight_mode="scale")
    rb_before = 1.8 / 0.7
    rb_after = float(rgb[0, 0, 0] / rgb[0, 0, 2])
    assert abs(rb_before - rb_after) < 0.02
    assert float(rgb.max()) <= 1.0 + 1e-9


def test_chroma_snr_gate_neutralizes_zero_signal():
    l = np.array([[0.0, 0.5]])
    chroma = np.array([[[1.5, 0.9, 0.8], [1.2, 1.0, 0.9]]])
    out = apply_chroma_snr_gate(l, chroma, 3.0, sigma_bg=0.05)
    assert np.allclose(out[0, 0], [1.0, 1.0, 1.0], atol=1e-6)
    assert not np.allclose(out[0, 1], [1.0, 1.0, 1.0])


def test_field_median_white_point_maps_median_star_neutral():
    temps = np.array([4000.0, 5500.0, 7000.0])
    med = float(np.median(temps))
    from hrd_colorfield import _planck_srgb_absolute

    wp = _planck_srgb_absolute(np.array([med]))[0]
    rgb_med = teff_to_srgb_chroma(np.array([med]), saturation=1.0, white_point_rgb=wp)[0]
    assert abs(float(rgb_med[0]) - float(rgb_med[2])) < 0.08


def test_config_clamps_and_enums():
    assert hrd_color_saturation_from_cfg(_Cfg(hrd_color_saturation=1.5)) == 1.0
    assert hrd_color_saturation_from_cfg(None) == 0.85
    assert hrd_color_chroma_snr_from_cfg(_Cfg(hrd_color_chroma_snr=99)) == 20.0
    assert hrd_color_chroma_snr_from_cfg(_Cfg(hrd_color_chroma_snr=-1)) == 0.0
    assert hrd_color_highlight_mode_from_cfg(_Cfg(hrd_color_highlight_mode="scale")) == "scale"
    assert hrd_color_highlight_mode_from_cfg(_Cfg(hrd_color_highlight_mode="bogus")) == "soft"
    assert hrd_color_white_point_from_cfg(_Cfg(hrd_color_white_point="d65")) == "d65"
    assert hrd_color_white_point_from_cfg(_Cfg(hrd_color_white_point="other")) == "field_median"


def test_render_fail_open_missing_fits(tmp_path: Path):
    ps = tmp_path / "ps"
    pt = tmp_path / "pt"
    ps.mkdir()
    pt.mkdir()
    csv = ps / "masterstars_full_match.csv"
    pd.DataFrame(
        {
            "catalog_id": ["123"],
            "x": [10.0],
            "y": [10.0],
            "dao_flux": [100.0],
            "bp_rp": [0.8],
        }
    ).to_csv(csv, index=False)
    out = tmp_path / "out.png"
    with patch("hrd_colorfield.log_event") as log_mock:
        result = render_catalog_color_field(ps, pt, _Cfg(), out)
    assert result is None
    assert log_mock.called
    assert not out.exists()


def test_render_disabled_returns_none(tmp_path: Path):
    ps = tmp_path / "ps"
    pt = tmp_path / "pt"
    ps.mkdir()
    pt.mkdir()
    out = tmp_path / "out.png"
    result = render_catalog_color_field(ps, pt, _Cfg(hrd_color_field_enabled=False), out)
    assert result is None


def test_chroma_boost_identity_at_one():
    rgb = np.array([[0.92, 0.88, 0.85], [1.0, 0.95, 0.9]])
    out = apply_chroma_boost(rgb, 1.0)
    assert np.array_equal(out, rgb)


def test_chroma_boost_monotonic_rb_separation():
    from hrd_colorfield import _planck_srgb_absolute

    wp_teff = 4817.0
    wp = _planck_srgb_absolute(np.array([wp_teff]))[0]
    r4000, _, b4000 = teff_to_srgb_chroma(4000.0, saturation=0.85, white_point_rgb=wp)
    r9000, _, b9000 = teff_to_srgb_chroma(9000.0, saturation=0.85, white_point_rgb=wp)
    sep_base = abs((r4000 / b4000) - (r9000 / b9000))
    seps = []
    for boost in (1.0, 1.6, 2.2, 3.0):
        c4000 = apply_chroma_boost(np.array([r4000, 0.88, b4000]), boost)
        c9000 = apply_chroma_boost(np.array([r9000, 0.95, b9000]), boost)
        seps.append(abs((c4000[0] / c4000[2]) - (c9000[0] / c9000[2])))
    assert seps[0] == sep_base
    assert all(seps[i] < seps[i + 1] for i in range(len(seps) - 1))


def test_chroma_boost_hue_order_preserved():
    from hrd_colorfield import _planck_srgb_absolute

    for wp_mode, wp_rgb in (
        ("d65", None),
        ("field_median", _planck_srgb_absolute(np.array([4817.0]))[0]),
    ):
        for teff in (3000.0, 4817.0, 6500.0, 12000.0):
            rgb = teff_to_srgb_chroma(teff, saturation=0.85, white_point_rgb=wp_rgb)
            boosted = apply_chroma_boost(rgb, 2.2)
            assert list(np.argsort(rgb)) == list(np.argsort(boosted)), f"{wp_mode} {teff}"


def test_chroma_boost_clamp_at_max():
    rgb = teff_to_srgb_chroma(np.linspace(3000, 12000, 20), saturation=0.85)
    out = apply_chroma_boost(rgb, 3.0)
    assert np.all(np.isfinite(out))
    mx = np.max(out, axis=-1)
    assert np.allclose(mx, 1.0)


def test_caption_chroma_boost_suffix():
    cap_off = build_colorfield_caption(white_point="d65", chroma_boost=1.0)
    cap_on = build_colorfield_caption(white_point="field_median", field_median_teff_k=4817.0, chroma_boost=1.6)
    assert "chroma enhanced" not in cap_off
    assert "chroma enhanced x1.6" in cap_on


def test_chroma_boost_config_clamp():
    assert hrd_color_chroma_boost_from_cfg(_Cfg(hrd_color_chroma_boost=0.5)) == 1.0
    assert hrd_color_chroma_boost_from_cfg(_Cfg(hrd_color_chroma_boost=5.0)) == 3.0
    assert hrd_color_chroma_boost_from_cfg(None) == 1.6


def test_bg_box_px_config_clamp():
    assert hrd_color_bg_box_px_from_cfg(_Cfg(hrd_color_bg_box_px=10)) == 32
    assert hrd_color_bg_box_px_from_cfg(_Cfg(hrd_color_bg_box_px=999)) == 512
    assert hrd_color_bg_box_px_from_cfg(None) == 96


def test_local_bg_map_reproduces_gradient():
    h, w = 384, 384
    yy, xx = np.mgrid[0:h, 0:w]
    lum = 0.1 + 0.8 * (xx / max(w - 1, 1))
    bg, sig = build_local_background_maps(lum, 96)
    corr = float(np.corrcoef(bg.ravel(), lum.ravel())[0, 1])
    assert corr > 0.995
    assert np.nanmean(np.abs(bg - lum)) < 0.05
    assert np.all(sig > 0)


def test_tapered_stamp_zero_at_boundary():
    stamp, radius = make_tapered_gaussian_stamp(2.0)
    assert stamp.shape == (2 * radius + 1, 2 * radius + 1)
    assert stamp[radius, radius] == 1.0
    edge_vals = np.concatenate(
        [
            stamp[0, :],
            stamp[-1, :],
            stamp[:, 0],
            stamp[:, -1],
        ]
    )
    assert np.all(edge_vals <= 1e-12)


def test_local_snr_gate_suppresses_faint_splat_sky_tint():
    h, w = 200, 200
    luminance = np.full((h, w), 0.55)
    chroma = np.ones((h, w, 3), dtype=np.float64)
    chroma[..., 0] = 1.22
    chroma[..., 2] = 0.84
    bg_local, sigma_local = build_local_background_maps(luminance, 64)
    old = apply_chroma_snr_gate(luminance, chroma, 3.0, sigma_bg=0.05)
    new = apply_chroma_snr_gate(
        luminance, chroma, 3.0, bg_local=bg_local, sigma_local=sigma_local
    )
    old_rgb = compose_catalog_color_rgb(luminance, old, highlight_mode="soft")
    new_rgb = compose_catalog_color_rgb(luminance, new, highlight_mode="soft")
    old_worst = background_neutrality_grid(old_rgb)["worst_patch_metric"]
    new_worst = background_neutrality_grid(new_rgb)["worst_patch_metric"]
    assert new_worst < 0.01
    assert old_worst > 0.03


def test_local_snr_gate_passes_bright_star_chroma():
    h, w = 64, 64
    luminance = np.full((h, w), 0.35)
    luminance[32, 32] = 0.95
    chroma = np.ones((h, w, 3))
    chroma[32, 32] = [1.45, 0.92, 0.78]
    bg_local, sigma_local = build_local_background_maps(luminance, 32)
    out = apply_chroma_snr_gate(
        luminance, chroma, 3.0, bg_local=bg_local, sigma_local=sigma_local
    )
    assert not np.allclose(out[32, 32], [1.0, 1.0, 1.0], atol=0.05)


def test_snr_zero_preserves_splat_chroma():
    shape = (48, 48)
    xs = np.array([24.0])
    ys = np.array([24.0])
    rgb = np.array([[1.35, 0.9, 0.82]])
    amps = np.array([1.0])
    chroma = splat_chroma_layer(shape, xs, ys, rgb, amps, sigma_px=2.0)
    lum = np.full(shape, 0.4)
    lum[24, 24] = 0.9
    bg, sig = build_local_background_maps(lum, 48)
    out = apply_chroma_snr_gate(lum, chroma, 0.0, bg_local=bg, sigma_local=sig)
    assert np.array_equal(out, chroma)
    boosted = apply_chroma_boost(rgb, 1.0)
    assert np.array_equal(boosted, rgb)
