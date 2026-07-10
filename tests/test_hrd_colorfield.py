"""Unit tests for catalog-color HRD field rendering (hrd_colorfield.py)."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hrd_colorfield import (
    BP_RP_DOMAIN,
    TEFF_MAX_K,
    TEFF_MIN_K,
    hrd_color_saturation_from_cfg,
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
    sat = 0.7
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


def test_config_saturation_clamp():
    assert hrd_color_saturation_from_cfg(_Cfg(hrd_color_saturation=1.5)) == 1.0
    assert hrd_color_saturation_from_cfg(_Cfg(hrd_color_saturation=-0.2)) == 0.0
    assert hrd_color_saturation_from_cfg(_Cfg(hrd_color_saturation=0.55)) == 0.55
    assert hrd_color_saturation_from_cfg(None) == 0.7


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
