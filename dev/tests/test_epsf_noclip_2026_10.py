"""EPSF-NOCLIP-2026-10 (D1): the ePSF builder runs without sigma clipping (plain median per grid point)."""

from __future__ import annotations

import pipeline  # noqa: F401  app import order
import psf_photometry as pp


def test_epsf_builder_gets_no_sigma_clip(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def fake_builder(**kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(pp, "EPSFBuilder", fake_builder)
    try:
        pp._epsf_build_imagepsf_from_stars(object(), osamp=2, fwhm_px=2.0, cutout_size=25)
    except Stop:
        pass
    assert "sigma_clip" in seen and seen["sigma_clip"] is None
