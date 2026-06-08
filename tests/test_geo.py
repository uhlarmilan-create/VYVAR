"""Tests for observer location resolution and AltAz airmass fallback."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_bjd_pertarget_vs_fieldcenter():
    """Per-target BJD must differ from field-center BJD for an off-axis star."""
    import numpy as np

    from photometry_core import _recompute_bjd_hjd_per_target
    from time_utils import compute_hjd_bjd

    cfg = _jirnt_cfg()
    jd = 2461154.316555035
    lat, lon, alt = cfg.observer_lat, cfg.observer_lon, cfg.observer_alt_m
    _, bjd_fc = compute_hjd_bjd(jd, 209.5043, 41.19122, lat, lon, alt)
    assert bjd_fc is not None

    bjd_pt, _ = _recompute_bjd_hjd_per_target(
        np.array([jd], dtype=float),
        207.4983,
        39.4037,
        cfg,
    )
    diff_sec = abs(float(bjd_pt[0]) - float(bjd_fc)) * 86400.0
    assert diff_sec > 5.0


def test_bjd_pertarget_precision():
    """BJD - JD should be order ~8 minutes (barycentric light-travel time)."""
    import numpy as np

    from photometry_core import _recompute_bjd_hjd_per_target

    cfg = _jirnt_cfg()
    jd = 2461154.316555035
    bjd, _ = _recompute_bjd_hjd_per_target(
        np.array([jd], dtype=float),
        207.4983,
        39.4037,
        cfg,
    )
    ltt_min = abs(float(bjd[0]) - jd) * 86400.0 / 60.0
    assert 3.0 < ltt_min < 15.0


def test_bjd_pertarget_fallback_no_location():
    import numpy as np

    from photometry_core import _recompute_bjd_hjd_per_target

    cfg = AppConfig()
    cfg.observer_lat = 0.0
    cfg.observer_lon = 0.0
    jd = np.array([2461154.31, 2461154.32], dtype=float)
    bjd, hjd = _recompute_bjd_hjd_per_target(jd, 200.0, 40.0, cfg)
    np.testing.assert_array_equal(bjd, jd)
    np.testing.assert_array_equal(hjd, jd)


def test_bjd_pertarget_fallback_nan_coords():
    import numpy as np

    from photometry_core import _recompute_bjd_hjd_per_target

    cfg = _jirnt_cfg()
    jd = np.array([2461154.31], dtype=float)
    bjd, hjd = _recompute_bjd_hjd_per_target(jd, float("nan"), 40.0, cfg)
    np.testing.assert_array_equal(bjd, jd)
    np.testing.assert_array_equal(hjd, jd)


def test_bjd_pertarget_batch_matches_scalar():
    import numpy as np

    from photometry_core import _recompute_bjd_hjd_per_target
    from time_utils import compute_hjd_bjd

    cfg = _jirnt_cfg()
    lat, lon, alt = cfg.observer_lat, cfg.observer_lon, cfg.observer_alt_m
    ra, dec = 207.4983, 39.4037
    jds = np.linspace(2461154.316, 2461154.320, 12, dtype=float)

    bjd_b, hjd_b = _recompute_bjd_hjd_per_target(jds, ra, dec, cfg)

    bjd_s = []
    hjd_s = []
    for jd in jds:
        hjd, bjd = compute_hjd_bjd(float(jd), ra, dec, lat, lon, alt)
        bjd_s.append(bjd)
        hjd_s.append(hjd)

    max_bjd_sec = float(np.nanmax(np.abs(bjd_b - np.asarray(bjd_s, dtype=float))) * 86400.0)
    max_hjd_sec = float(np.nanmax(np.abs(hjd_b - np.asarray(hjd_s, dtype=float))) * 86400.0)
    assert max_bjd_sec < 0.001
    assert max_hjd_sec < 0.001
