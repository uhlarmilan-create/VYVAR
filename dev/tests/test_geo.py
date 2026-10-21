"""Tests for observer location resolution and AltAz airmass fallback."""

from __future__ import annotations

import math

import pytest
from astropy.io import fits

from config import AppConfig
from pipeline import _compute_airmass_from_altaz, _extract_airmass_from_header
from time_utils import resolve_observer_location


def _dablice_cfg() -> AppConfig:
    c = AppConfig()
    c.observer_lat = 50.075
    c.observer_lon = 14.437
    c.observer_alt_m = 525.0
    c.observer_location_id = 1
    return c


def test_resolve_location_fits_header():
    hdr = fits.Header()
    hdr["SITELAT"] = 49.5
    hdr["SITELONG"] = 18.0
    hdr["SITEELEV"] = 400.0
    lat, lon, elev = resolve_observer_location(hdr, cfg=_dablice_cfg())
    assert lat is not None and lon is not None
    assert abs(float(lat) - 49.5) < 0.01
    assert abs(float(lon) - 18.0) < 0.01
    assert float(elev) == pytest.approx(400.0, abs=1.0)


def test_resolve_location_config_fallback():
    hdr = fits.Header()
    cfg = _dablice_cfg()
    lat, lon, elev = resolve_observer_location(hdr, db=None, draft_id=None, cfg=cfg)
    assert lat is not None and lon is not None
    assert abs(float(lat) - 50.075) < 0.02
    assert abs(float(lon) - 14.437) < 0.02


def test_resolve_location_unknown():
    hdr = fits.Header()
    lat, lon, elev = resolve_observer_location(hdr, db=None, draft_id=None, cfg=None)
    assert lat is None and lon is None and elev is None


def test_compute_airmass_altaz():
    cfg = _dablice_cfg()
    hdr = fits.Header()
    hdr["CRVAL1"] = 202.0
    hdr["CRVAL2"] = 47.0
    hdr["DATE-OBS"] = "2026-04-23T22:00:00"
    hdr["EXPTIME"] = 60.0
    am = _compute_airmass_from_altaz(hdr, cfg)
    assert math.isfinite(am)
    assert 1.0 <= am <= 5.0


def test_compute_airmass_altaz_no_cfg():
    hdr = fits.Header()
    hdr["CRVAL1"] = 202.0
    hdr["CRVAL2"] = 47.0
    hdr["DATE-OBS"] = "2026-04-23T22:00:00"
    assert not math.isfinite(_compute_airmass_from_altaz(hdr, cfg=None))


def test_compute_airmass_altaz_below_horizon():
    cfg = _dablice_cfg()
    hdr = fits.Header()
    hdr["CRVAL1"] = 0.0
    hdr["CRVAL2"] = -60.0
    hdr["DATE-OBS"] = "2026-04-23T12:00:00"
    hdr["EXPTIME"] = 60.0
    am = _compute_airmass_from_altaz(hdr, cfg)
    assert not math.isfinite(am)


def test_extract_airmass_uses_altaz_when_header_empty():
    cfg = _dablice_cfg()
    hdr = fits.Header()
    hdr["CRVAL1"] = 202.0
    hdr["CRVAL2"] = 47.0
    hdr["DATE-OBS"] = "2026-04-23T22:00:00"
    hdr["EXPTIME"] = 60.0
    am = _extract_airmass_from_header(hdr, cfg=cfg)
    assert math.isfinite(am)
    assert 1.0 <= am <= 5.0


def _jirnt_cfg() -> AppConfig:
    c = AppConfig()
    c.observer_lat = 50.1121658
    c.observer_lon = 14.6982547
    c.observer_alt_m = 275.0
    return c


