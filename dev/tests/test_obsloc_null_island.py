"""TIER1-OBSLOC-ZERO: null-island observer site guard."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from astropy.io import fits

from config import AppConfig
from param_resolver import NULL_ISLAND_LAT_LON_THRESHOLD_DEG, is_null_island_coords, resolve_site
from photometry_core import TIME_BASE_JD_FALLBACK, _recompute_bjd_hjd_with_status
from pipeline import _compute_airmass_from_altaz
from time_utils import resolve_observer_location
import numpy as np


def test_null_island_coords_threshold() -> None:
    assert is_null_island_coords(0.0, 0.0)
    assert is_null_island_coords(0.005, 0.005)
    assert not is_null_island_coords(0.0, 0.5)
    assert not is_null_island_coords(50.0, 14.0)


def test_resolve_site_null_island_unresolved(caplog: pytest.LogCaptureFixture) -> None:
    db = MagicMock()
    db.get_draft_location_id.return_value = 42

    def _exec(sql: str, params: tuple) -> MagicMock:
        cur = MagicMock()
        if "LATITUDE" in sql:
            cur.fetchone.return_value = (0.0, 0.0, 0.0)
        else:
            cur.fetchone.return_value = None
        return cur

    db.conn.execute.side_effect = _exec
    cfg = AppConfig()
    cfg.observer_lat = 50.0
    cfg.observer_lon = 14.0
    with caplog.at_level(logging.ERROR):
        site = resolve_site(fits.Header(), db=db, draft_id=424, cfg=cfg)
    assert not site.ok
    assert site.source == "unresolved"
    assert any("null-island" in r.message for r in caplog.records)
    assert any("ID_LOCATION=42" in r.message for r in caplog.records)


def test_resolve_site_real_location_unchanged() -> None:
    db = MagicMock()
    db.get_draft_location_id.return_value = 1
    db.conn.execute.return_value.fetchone.return_value = (50.075, 14.437, 525.0)
    site = resolve_site(fits.Header(), db=db, draft_id=1, cfg=AppConfig())
    assert site.ok
    assert abs(site.lat - 50.075) < 0.01


def test_resolve_observer_location_null_island_returns_none() -> None:
    db = MagicMock()
    db.get_draft_location_id.return_value = 1
    db.conn.execute.return_value.fetchone.return_value = (0.0, 0.0, 0.0)
    lat, lon, elev = resolve_observer_location(fits.Header(), db=db, draft_id=1, cfg=AppConfig())
    assert lat is None and lon is None and elev is None


def test_airmass_refuses_null_island() -> None:
    db = MagicMock()
    db.get_draft_location_id.return_value = 1
    db.conn.execute.return_value.fetchone.return_value = (0.0, 0.0, 0.0)
    hdr = fits.Header()
    hdr["CRVAL1"] = 202.0
    hdr["CRVAL2"] = 47.0
    hdr["DATE-OBS"] = "2026-04-23T22:00:00"
    hdr["EXPTIME"] = 60.0
    am = _compute_airmass_from_altaz(hdr, AppConfig(), db=db, draft_id=1)
    assert not np.isfinite(am)


def test_bjd_null_island_jd_fallback_time_base() -> None:
    cfg = AppConfig()
    jd = np.array([2461154.31, 2461154.32], dtype=float)
    bjd, hjd, time_base = _recompute_bjd_hjd_with_status(
        jd, 200.0, 40.0, cfg, site=(0.0, 0.0, 0.0)
    )
    np.testing.assert_array_equal(bjd, jd)
    np.testing.assert_array_equal(hjd, jd)
    assert time_base == TIME_BASE_JD_FALLBACK


def test_null_island_threshold_is_module_constant() -> None:
    assert NULL_ISLAND_LAT_LON_THRESHOLD_DEG == 0.01
