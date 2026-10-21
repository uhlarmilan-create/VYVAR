"""PLATESOLVE-2026-10: astrosolve core on synthetic fields (rotation, mirror, flip, SIP,
wide / narrow scale, no scale hint)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS, Sip

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

import astrosolve as A  # noqa: E402


def _case(seed, *, scale_as, rot, mirror=False, nx=1600, ny=1200, density=3000.0, sip=False, cone=0.2,
          hint_off=0.02, drop=0.2, spur=20, ra0=320.36, dec0=51.82):
    rng = np.random.default_rng(seed)
    R = 2.0 * cone
    n = int(density * 4 * R * R)
    xi, eta = rng.uniform(-R, R, n), rng.uniform(-R, R, n)
    ra, dec = A.tan_deproject(xi, eta, ra0, dec0)
    mag = 8 + 9.5 * rng.random(n) ** 0.35
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.crval = [ra0, dec0]
    s, c, sn = scale_as / 3600, math.cos(math.radians(rot)), math.sin(math.radians(rot))
    cd = s * np.array([[-c, sn], [sn, c]])
    if mirror:
        cd[:, 0] *= -1
    w.wcs.cd = cd
    if sip:
        w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
        a, b = np.zeros((4, 4)), np.zeros((4, 4))
        a[3, 0] = a[1, 2] = b[0, 3] = b[2, 1] = 3e-9
        a[2, 0], b[0, 2] = 3e-6, -3e-6
        w.sip = Sip(a, b, None, None, w.wcs.crpix)
    x, y = w.all_world2pix(ra, dec, 0)
    ins = (x > 0) & (x < nx - 1) & (y > 0) & (y < ny - 1)
    xs, ys, ms = x[ins], y[ins], mag[ins]
    o = np.argsort(ms)[:300]
    keep = o[rng.random(len(o)) > drop]
    det = np.column_stack([xs[keep], ys[keep]]) + rng.normal(0, 0.3, (len(keep), 2))
    flux = 10 ** (-0.4 * (ms[keep] - 20))
    sp = np.column_stack([rng.uniform(0, nx, spur), rng.uniform(0, ny, spur)])
    det = np.vstack([det, sp])
    flux = np.concatenate([flux, rng.uniform(flux.min(), np.median(flux), spur)])
    hra, hdec = A.tan_deproject(np.array([hint_off]), np.array([0.0]), ra0, dec0)
    cxi, ceta = A.tan_project(ra, dec, float(hra[0]), float(hdec[0]))
    sel = np.hypot(cxi, ceta) < cone
    return det, flux, (ny, nx), ra[sel], dec[sel], mag[sel], w, float(hra[0]), float(hdec[0])


def _max_err(r, w, shape):
    ny, nx = shape
    gx, gy = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 7))
    rt, dt = w.all_pix2world(gx.ravel(), gy.ravel(), 0)
    px, py = r.wcs.all_world2pix(rt, dt, 0)
    return float(np.max(np.hypot(px - gx.ravel(), py - gy.ravel())))


@pytest.mark.parametrize("kw", [
    dict(seed=1, scale_as=0.57, rot=37.0),
    dict(seed=2, scale_as=0.57, rot=-20.0, mirror=True),
    dict(seed=3, scale_as=0.57, rot=181.0),            # meridian flip
    dict(seed=4, scale_as=0.57, rot=10.0, sip=True),   # distortion
    dict(seed=5, scale_as=10.0, rot=90.0, density=300.0, cone=4.0, hint_off=0.3),  # wide field
])
def test_solve_against_catalog(kw):
    det, flux, shape, ra, dec, mag, w, hra, hdec = _case(**kw)
    r = A.solve_against_catalog(det, flux, shape, ra, dec, mag, ra0=hra, dec0=hdec, sigma_px=1.5)
    assert r.ok, r.message
    assert r.log_odds >= A.LOG_ODDS_ACCEPT
    assert abs(r.scale_arcsec / kw["scale_as"] - 1) < 2e-3
    assert r.mirrored == bool(kw.get("mirror", False))
    assert _max_err(r, w, shape) < 0.6
    if kw.get("sip"):
        assert r.sip_order and r.sip_order >= 2


def test_wrong_catalog_is_rejected():
    det, flux, shape, ra, dec, mag, w, hra, hdec = _case(7, scale_as=0.57, rot=0.0)
    det2, flux2, *_ = _case(8, scale_as=0.57, rot=0.0)  # detections of a different random sky
    r = A.solve_against_catalog(det2, flux2, shape, ra, dec, mag, ra0=hra, dec0=hdec, sigma_px=1.5)
    assert not r.ok


def test_tan_roundtrip_and_pattern_invariance():
    ra, dec = A.tan_deproject(np.array([0.3, -1.2]), np.array([0.7, 0.1]), 359.9, 45.0)
    xi, eta = A.tan_project(ra, dec, 359.9, 45.0)
    assert np.allclose(xi, [0.3, -1.2]) and np.allclose(eta, [0.7, 0.1])
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 100, (4, 2))
    q = np.array([[0, 1, 2, 3]])
    c1, _, s1 = A.pattern_codes(p, q)
    ang = 1.234
    rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    p2 = (p @ rot.T) * 3.7 + 5.0
    p2[:, 0] *= -1  # mirror
    c2, _, s2 = A.pattern_codes(p2, q)
    assert np.allclose(c1, c2) and math.isclose(s2[0] / s1[0], 3.7)
