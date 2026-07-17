"""Mid-exposure JD / BJD guard tests (math/physics audit D2)."""

from __future__ import annotations

import astropy.units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta

from time_utils import mid_exposure_jd


def _header_with_exptime(exptime: float = 30.0) -> fits.Header:
    hdr = fits.Header()
    hdr["DATE-OBS"] = "2024-01-15T22:30:00"
    hdr["EXPTIME"] = float(exptime)
    return hdr


def test_mid_exposure_jd_offset_equals_half_exptime() -> None:
    exptime = 30.0
    hdr = _header_with_exptime(exptime)
    jd_mid = mid_exposure_jd(hdr)
    assert jd_mid is not None
    t_start = Time(hdr["DATE-OBS"], format="isot", scale="utc")
    t_mid_expected = t_start + TimeDelta(exptime / 2.0 * u.s)
    delta_sec = (Time(jd_mid, format="jd", scale="utc") - t_start).to_value(u.s)
    assert abs(delta_sec - exptime / 2.0) < 1e-6


def test_mid_exposure_jd_warns_when_exptime_missing(monkeypatch) -> None:
    import time_utils as tu

    messages: list[str] = []
    monkeypatch.setattr(tu, "_WARNED_ONCE", set())
    monkeypatch.setattr(tu, "log_event", lambda m: messages.append(m))

    hdr = fits.Header()
    hdr["DATE-OBS"] = "2024-01-15T22:30:00"
    jd_mid = tu.mid_exposure_jd(hdr)
    assert jd_mid is not None
    t_start = Time(hdr["DATE-OBS"], format="isot", scale="utc")
    assert abs(jd_mid - float(t_start.jd)) < 1e-9
    assert any("EXPTIME" in m for m in messages)
