"""Step 4 G02-17 / G02-18 (decision C6, 2026-10-29): one field-centre authority, per-star BJD_TDB.

Eastman, Siverd & Gaudi 2010 (PASP 122, 935): BJD_TDB needs the target's own direction; across a
wide field the barycentric light time differs by ~8.8 s per degree.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import EarthLocation, SkyCoord, get_body_barycentric
from astropy.io import fits
from astropy.time import Time

SITE = (50.075, 14.437, 525.0)  # Dablice
CENTRE = (200.0, 45.0)
DATE_OBS = "2026-03-21T22:00:00"  # Earth-Sun line roughly along RA 0h/12h: large offset in RA
EXPTIME = 60.0


def _wcs_header(ra0: float = CENTRE[0], dec0: float = CENTRE[1]) -> fits.Header:
    """An aligned frame: TAN WCS whose centre pixel is (ra0, dec0), 1000 x 800 px of 10"/px."""
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = 1000
    h["NAXIS2"] = 800
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRPIX1"] = 500.5  # 1-based: centre of a 1000 px axis
    h["CRPIX2"] = 400.5
    h["CRVAL1"] = ra0
    h["CRVAL2"] = dec0
    h["CDELT1"] = -10.0 / 3600.0
    h["CDELT2"] = 10.0 / 3600.0
    h["CUNIT1"] = "deg"
    h["CUNIT2"] = "deg"
    h["DATE-OBS"] = DATE_OBS
    h["EXPTIME"] = EXPTIME
    h["SITELAT"] = SITE[0]
    h["SITELONG"] = SITE[1]
    h["SITEELEV"] = SITE[2]
    return h


def _offset_star(ra0: float, dec0: float, sep_deg: float, pa_deg: float) -> tuple[float, float]:
    c = SkyCoord(ra0 * u.deg, dec0 * u.deg).directional_offset_by(pa_deg * u.deg, sep_deg * u.deg)
    return float(c.ra.deg), float(c.dec.deg)


def _ltt_bary_s(jd_utc: float, ra: float, dec: float) -> float:
    """Independent scalar reference: astropy light_travel_time for one instant and one direction."""
    loc = EarthLocation(lat=SITE[0] * u.deg, lon=SITE[1] * u.deg, height=SITE[2] * u.m)
    t = Time(jd_utc, format="jd", scale="utc", location=loc)
    return float(t.light_travel_time(SkyCoord(ra * u.deg, dec * u.deg), "barycentric").to_value(u.s))


# ------------------------------------------------------------------ G02-17 field centre


def test_wcs_centre_is_the_field_centre() -> None:
    from fits_meta import field_centre_radec

    h = _wcs_header()
    h["VYTARGRA"] = 201.0  # a target hint must not override the solved centre
    h["VYTARGDE"] = 44.0
    ra, dec, src = field_centre_radec(h)
    assert src == "WCS"
    assert abs(ra - CENTRE[0]) < 1e-9 and abs(dec - CENTRE[1]) < 1e-9


def test_sexagesimal_ra_resolves_like_decimal() -> None:
    """Brno 534 headers carry RA/DEC as 'HH MM SS' strings - they were skipped by time_utils."""
    from fits_meta import field_centre_radec
    from time_utils import compute_time_columns

    dec_h = fits.Header()
    dec_h["RA"] = 200.0
    dec_h["DEC"] = 45.5
    sex_h = fits.Header()
    sex_h["RA"] = "13 20 00"
    sex_h["DEC"] = "+45 30 00"
    for h in (dec_h, sex_h):
        h["DATE-OBS"] = DATE_OBS
        h["EXPTIME"] = EXPTIME
        h["SITELAT"], h["SITELONG"], h["SITEELEV"] = SITE
    a = field_centre_radec(dec_h)
    b = field_centre_radec(sex_h)
    assert a[2] == b[2] == "RA_DEC"
    assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9
    ta, tb = compute_time_columns(dec_h), compute_time_columns(sex_h)
    assert ta["bjd_tdb_mid"] is not None
    assert abs(ta["bjd_tdb_mid"] - tb["bjd_tdb_mid"]) < 1e-9
    assert abs(ta["hjd_mid"] - tb["hjd_mid"]) < 1e-9


def test_one_order_for_all_consumers() -> None:
    """One header with VYTARG, an RA string, OBJCTRA and a bare CRVAL: BJD, airmass, lunar and the
    raw-frame metadata use the same coordinates (G02-17 test of the review)."""
    from aperture_photometry import _aperture_resolve_field_center_ra_dec
    from fits_meta import field_centre_radec, fits_metadata_from_primary_header
    from pipeline_catalog import _compute_airmass_from_altaz
    from time_utils import compute_time_columns, light_time_corrected_jd, mid_exposure_jd

    h = fits.Header()
    h["VYTARGRA"] = 210.0
    h["VYTARGDE"] = 40.0
    h["RA"] = "13 20 00"
    h["DEC"] = "+45 00 00"
    h["OBJCTRA"] = "13 00 00"
    h["OBJCTDEC"] = "+30 00 00"
    h["CRVAL1"] = 150.0
    h["CRVAL2"] = 20.0
    h["DATE-OBS"] = DATE_OBS
    h["EXPTIME"] = EXPTIME
    h["SITELAT"], h["SITELONG"], h["SITEELEV"] = SITE
    ra, dec, src = field_centre_radec(h)
    assert (ra, dec, src) == (210.0, 40.0, "VY_TARG")

    md = fits_metadata_from_primary_header(h)
    assert (md["ra"], md["dec"]) == (ra, dec)
    lra, ldec, lsrc = _aperture_resolve_field_center_ra_dec(h, pd.DataFrame())
    assert (lra, ldec) == (ra, dec) and lsrc == "MASTERSTAR_VY_TARG"

    jd = mid_exposure_jd(h)
    bjd, _hjd = light_time_corrected_jd(jd, ra, dec, SITE)
    assert compute_time_columns(h)["bjd_tdb_mid"] == pytest.approx(float(bjd), abs=1e-10)

    from astropy.coordinates import AltAz

    from time_utils import _airmass_from_altitude_deg

    loc = EarthLocation(lat=SITE[0] * u.deg, lon=SITE[1] * u.deg, height=SITE[2] * u.m)
    alt = SkyCoord(ra * u.deg, dec * u.deg).transform_to(
        AltAz(obstime=Time(jd, format="jd", scale="utc", location=loc), location=loc)).alt.deg
    am = _compute_airmass_from_altaz(h, None)
    assert am == pytest.approx(round(_airmass_from_altitude_deg(float(alt)), 5), abs=1e-9)


def test_draft_centre_is_last_and_zero_is_unset() -> None:
    from fits_meta import field_centre_radec

    class _Db:
        def __init__(self, row):
            self.row = row

        def fetch_obs_draft_by_id(self, _did):
            return self.row

    h = fits.Header()
    own = {"NoFilter_0_1": {"ra_deg": 12.5, "de_deg": -3.0}}  # S6-FIX: the centre of the frame's setup
    assert field_centre_radec(h, db=_Db({"CENTER_BY_SETUP": own}), draft_id=7) == (12.5, -3.0, "DRAFT_CENTRE")
    zero = {"NoFilter_0_1": {"ra_deg": 0.0, "de_deg": 0.0}}
    assert field_centre_radec(h, db=_Db({"CENTER_BY_SETUP": zero}), draft_id=7) == (None, None, "none")


# ------------------------------------------------------------------ G02-18 per-star BJD


def test_centre_star_bjd_equals_field_centre_bjd() -> None:
    """A star at the field centre: the target-LC BJD == the proc-CSV frame BJD (review invariant)."""
    from config import AppConfig
    from photometry_lightcurve import _recompute_bjd_hjd_with_status
    from time_utils import compute_time_columns

    h = _wcs_header()
    cols = compute_time_columns(h)
    bjd, hjd, base = _recompute_bjd_hjd_with_status(
        np.array([cols["jd_mid"]]), CENTRE[0], CENTRE[1], AppConfig(), site=SITE)
    assert base == "BJD_TDB"
    assert float(bjd[0]) == pytest.approx(cols["bjd_tdb_mid"], abs=1e-10)
    assert float(hjd[0]) == pytest.approx(cols["hjd_mid"], abs=1e-10)


def test_star_far_from_centre_differs_by_the_light_time_difference() -> None:
    """3.5 deg off-centre (BO CVn 200 mm field): per-star BJD - field BJD == ltt(star) - ltt(centre),
    checked against independent scalar astropy calls (< 1 ms) and the Roemer geometry."""
    from time_utils import compute_time_columns, light_time_corrected_jd

    h = _wcs_header()
    cols = compute_time_columns(h)
    jd = float(cols["jd_mid"])
    star = _offset_star(*CENTRE, 3.5, 90.0)  # east of the centre
    bjd_star, _ = light_time_corrected_jd(jd, star[0], star[1], SITE)
    d_s = (float(bjd_star) - float(cols["bjd_tdb_mid"])) * 86400.0
    d_ref = _ltt_bary_s(jd, *star) - _ltt_bary_s(jd, *CENTRE)
    assert abs(d_s - d_ref) < 1e-3
    # Roemer geometry: (r_earth_bary . (n_star - n_centre)) / c; the topocentric part is < 2 ms here
    t = Time(jd, format="jd", scale="utc")
    r = get_body_barycentric("earth", t.tdb).xyz.to_value(u.m)
    n1 = SkyCoord(star[0] * u.deg, star[1] * u.deg).cartesian.xyz.value
    n0 = SkyCoord(CENTRE[0] * u.deg, CENTRE[1] * u.deg).cartesian.xyz.value
    d_geo = float(np.dot(r, n1 - n0)) / 299792458.0
    assert abs(d_s - d_geo) < 2e-3
    # it is a real effect at this offset (up to ~31 s at 3.5 deg) and never more than 499 s x angle
    assert 5.0 < abs(d_s) <= 499.1 * math.radians(3.5)


def test_grid_equals_pairwise() -> None:
    """The stars x frames broadcast equals one scalar call per (star, frame)."""
    from time_utils import light_time_corrected_jd

    jd = 2461121.4 + np.array([0.0, 0.05, 0.11])
    ra = np.array([200.0, 204.9, 195.0, np.nan])
    dec = np.array([45.0, 45.0, 47.5, 45.0])
    bjd, hjd = light_time_corrected_jd(jd[None, :], ra[:, None], dec[:, None], SITE)
    assert bjd.shape == (4, 3) and np.isnan(bjd[3]).all() and np.isnan(hjd[3]).all()
    for i in range(3):
        for j in range(3):
            b1, h1 = light_time_corrected_jd(jd[j], ra[i], dec[i], SITE)
            assert bjd[i, j] == float(b1) and hjd[i, j] == float(h1)
    with pytest.raises(ValueError):
        light_time_corrected_jd(jd, 200.0, 45.0, (None, 14.0, 0.0))


def test_variability_candidate_time_is_per_star(tmp_path: Path) -> None:
    """varcand_*.csv / transient times: the star's own BJD_TDB, not the field-centre frame time."""
    from time_utils import compute_time_columns, light_time_corrected_jd, mid_exposure_jd
    from time_utils import run_site
    from variability_field import star_times_bjd

    hdrs = []
    for k in range(4):
        h = _wcs_header()
        h["DATE-OBS"] = Time(Time(DATE_OBS).jd + 0.02 * k, format="jd").isot
        hdrs.append(h)
    frame = pd.DataFrame({
        "time": [compute_time_columns(h)["bjd_tdb_mid"] for h in hdrs],  # proc bjd_tdb_mid (field centre)
        "jd_utc": [mid_exposure_jd(h) for h in hdrs],
    })
    (tmp_path / "pipeline_meta.json").write_text(json.dumps(
        {"observer_location": {"lat": SITE[0], "lon": SITE[1], "alt_m": SITE[2], "source": "draft"}}), encoding="utf-8")
    site = run_site(tmp_path)
    assert site == SITE
    edge = _offset_star(*CENTRE, 3.5, 90.0)
    t, col = star_times_bjd(frame, [CENTRE[0], edge[0]], [CENTRE[1], edge[1]], site)
    assert col == "bjd_tdb" and t.shape == (2, 4)
    np.testing.assert_allclose(t[0], frame["time"].to_numpy(), rtol=0, atol=1e-10)  # centre star
    ref, _ = light_time_corrected_jd(frame["jd_utc"].to_numpy(), edge[0], edge[1], SITE)
    np.testing.assert_array_equal(t[1], ref)
    assert np.all(np.abs(t[1] - frame["time"].to_numpy()) * 86400.0 > 5.0)

    # no site: the UTC JD of mid-exposure, named so
    (tmp_path / "pipeline_meta.json").write_text(json.dumps({"observer_location": {"lat": 0.0, "lon": 0.0}}),
                                                 encoding="utf-8")
    assert run_site(tmp_path) is None
    t0, col0 = star_times_bjd(frame, [edge[0]], [edge[1]], None)
    assert col0 == "jd_mid"
    np.testing.assert_array_equal(t0[0], frame["jd_utc"].to_numpy())
