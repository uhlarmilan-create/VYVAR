"""BLIND-SOLVE-2026-10: blind solving with a small synthetic Gaia DB and its index.

Real MASTERSTARs (drafts 529 / 535 / 536) were "solved" blind into a wrong sky with 4-5 chance
matches: the tweak re-fit counted its own pairs and a field with few catalog stars made one
match worth +11 in log-odds. A blind solution must now survive the refinement against the full
catalog (credible_solution); random or foreign detections must never be accepted.
"""
from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

import astro_index as AI  # noqa: E402
import astrosolve as A  # noqa: E402

RA0, DEC0, HALF = 102.0, 32.0, 2.0  # synthetic sky patch RA 100-104, Dec 30-34


def _sky(seed: int, density: float = 2500.0):
    rng = np.random.default_rng(seed)
    area = (2 * HALF) * (2 * HALF) * math.cos(math.radians(DEC0))
    n = int(density * area)
    ra = rng.uniform(RA0 - HALF, RA0 + HALF, n)
    dec = np.degrees(np.arcsin(rng.uniform(math.sin(math.radians(DEC0 - HALF)), math.sin(math.radians(DEC0 + HALF)), n)))
    mag = 8 + 9.5 * rng.random(n) ** 0.35
    return ra, dec, mag


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    d = tmp_path_factory.mktemp("blind")
    db = d / "gaia.db"
    ra, dec, mag = _sky(11)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE gaia_dr3 (source_id INTEGER PRIMARY KEY, ra REAL, dec REAL, g_mag REAL)")
    con.executemany("INSERT INTO gaia_dr3 VALUES (?,?,?,?)", [(i, float(a), float(b), float(m)) for i, (a, b, m) in enumerate(zip(ra, dec, mag))])
    con.execute("CREATE INDEX idx_ra_dec ON gaia_dr3 (ra, dec)")
    con.commit()
    con.close()
    out = d / "index"
    AI.build_index(str(db), out, ra_range=(RA0 - HALF, RA0 + HALF), dec_range=(DEC0 - HALF, DEC0 + HALF), progress=None)
    return str(db), str(out), (ra, dec, mag)


def _frame(ra, dec, mag, *, scale_as, rot, nx=1600, ny=1200, seed=0, mirror=False):
    rng = np.random.default_rng(seed)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.crval = [RA0 + 0.1, DEC0 - 0.1]
    s, c, sn = scale_as / 3600, math.cos(math.radians(rot)), math.sin(math.radians(rot))
    cd = s * np.array([[-c, sn], [sn, c]])
    if mirror:
        cd[:, 0] *= -1
    w.wcs.cd = cd
    x, y = w.all_world2pix(ra, dec, 0)
    ins = (x > 0) & (x < nx - 1) & (y > 0) & (y < ny - 1)
    xs, ys, ms = x[ins], y[ins], mag[ins]
    o = np.argsort(ms)[:400]
    keep = o[rng.random(len(o)) > 0.15]
    det = np.column_stack([xs[keep], ys[keep]]) + rng.normal(0, 0.3, (len(keep), 2))
    flux = 10 ** (-0.4 * (ms[keep] - 20))
    sp = np.column_stack([rng.uniform(0, nx, 30), rng.uniform(0, ny, 30)])
    det = np.vstack([det, sp])
    flux = np.concatenate([flux, rng.uniform(flux.min(), np.median(flux), 30)])
    return det, flux, (ny, nx), w


def _max_err(res, w, shape):
    ny, nx = shape
    gx, gy = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 7))
    rt, dt = w.all_pix2world(gx.ravel(), gy.ravel(), 0)
    px, py = res.wcs.all_world2pix(rt, dt, 0)
    return float(np.max(np.hypot(px - gx.ravel(), py - gy.ravel())))


@pytest.mark.parametrize("kw", [
    dict(scale_as=1.2, rot=33.0),                 # 0.53 x 0.40 deg (Zdanice-like)
    dict(scale_as=1.2, rot=-75.0, mirror=True),
    dict(scale_as=4.5, rot=120.0, seed=3),        # 2.0 x 1.5 deg
])
def test_blind_solves_true_field(index, kw):
    db, out, (ra, dec, mag) = index
    det, flux, shape, w = _frame(ra, dec, mag, **kw)
    res = AI.solve_blind(det, flux, shape, index_dir=out, gaia_db=db, fwhm_px=3.0)
    assert res.ok, res.message
    assert abs(res.scale_arcsec / kw["scale_as"] - 1) < 2e-3
    assert res.mirrored == bool(kw.get("mirror", False))
    assert _max_err(res, w, shape) < 0.6
    ok, why = A.credible_solution(res, len(det), max(1.0, 0.5 * 3.0))
    assert ok, why


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_blind_rejects_random_detections(index, seed):
    db, out, _ = index
    rng = np.random.default_rng(100 + seed)
    nx, ny = 1600, 1200
    det = np.column_stack([rng.uniform(0, nx, 400), rng.uniform(0, ny, 400)])
    flux = rng.lognormal(8.0, 1.5, 400)
    res = AI.solve_blind(det, flux, (ny, nx), index_dir=out, gaia_db=db, fwhm_px=3.0, max_hypotheses=5000)
    assert not res.ok, f"random detections accepted: {res.message}"


def test_blind_rejects_foreign_sky(index):
    db, out, _ = index
    ra2, dec2, mag2 = _sky(999)  # another random sky at the same place: not in the DB
    det, flux, shape, _w = _frame(ra2, dec2, mag2, scale_as=1.2, rot=10.0)
    res = AI.solve_blind(det, flux, shape, index_dir=out, gaia_db=db, fwhm_px=3.0, max_hypotheses=5000)
    assert not res.ok, f"foreign sky accepted: {res.message}"


def test_verify_needs_enough_catalog_stars():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [800, 600]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.cd = np.array([[-1.2 / 3600, 0], [0, 1.2 / 3600]])
    rng = np.random.default_rng(5)
    n = A.MIN_VERIFY_CATALOG_STARS - 1
    xy = np.column_stack([rng.uniform(0, 1600, n), rng.uniform(0, 1200, n)])
    ra, dec = w.all_pix2world(xy[:, 0], xy[:, 1], 0)
    lo, nm, _ = A.verify_hypothesis(w, xy, ra, dec, np.arange(n, dtype=float), (1200, 1600), sigma_px=1.5)
    assert lo == -math.inf and nm == 0  # a perfect match of too few stars proves nothing


def test_credible_solution_gates():
    good = A.SolveResult(True, n_match=500, rms_px=0.3)
    assert A.credible_solution(good, 1000, 1.5)[0]
    assert not A.credible_solution(A.SolveResult(True, n_match=19, rms_px=0.3), 50, 1.5)[0]
    assert not A.credible_solution(A.SolveResult(True, n_match=60, rms_px=0.3), 1000, 1.5)[0]  # < 10 %
    assert not A.credible_solution(A.SolveResult(True, n_match=500, rms_px=2.0), 1000, 1.5)[0]
    assert not A.credible_solution(A.SolveResult(False, message="x"), 1000, 1.5)[0]


def test_tweak_cannot_vouch_for_itself():
    """Reproduces the real false positives (529 / 536): 6 catalog stars in the field, 4
    detections loosely near 4 of them (a 4-star pattern hit), the rest unrelated. The old
    tweak re-fitted TAN on the 4 pairs and counted them: log-odds 21.9 > ln(1e9)."""
    nx, ny, sig = 3552, 2664, 2.0
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.cd = np.array([[-0.16 / 3600, 0], [0, 0.16 / 3600]])
    rng = np.random.default_rng(7)
    cxy = np.column_stack([rng.uniform(200, nx - 200, 6), rng.uniform(200, ny - 200, 6)])
    ra, dec = w.all_pix2world(cxy[:, 0], cxy[:, 1], 0)
    # 4 detections near catalog stars 0-3, displaced by a common affine error (fits exactly)
    aff = np.array([[1.004, 0.003], [-0.002, 0.997]])
    near = (cxy[:4] - cxy[:4].mean(0)) @ aff.T + cxy[:4].mean(0) + np.array([9.0, -7.0])
    other = np.column_stack([rng.uniform(0, nx, 146), rng.uniform(0, ny, 146)])
    det = np.vstack([near, other])
    lo, nm, _ = A.verify_hypothesis(w, det, ra, dec, np.arange(6, dtype=float), (ny, nx), sigma_px=sig)
    assert not lo >= A.LOG_ODDS_ACCEPT, f"chance pattern verified: log-odds {lo:.1f}, {nm} matches"


def test_vectorised_lookup_matches_brute_force(index):
    """Band.lookup (all patterns at once) returns exactly the index patterns whose code is
    within the tolerance in every dimension."""
    _db, out, _ = index
    band = AI.load_index(out)[1]
    codes_all = np.asarray(band.codes, dtype=np.float32)
    rng = np.random.default_rng(3)
    pick = rng.choice(len(codes_all), 200, replace=False)
    q = codes_all[pick] + rng.normal(0, 0.004, (200, 5)).astype(np.float32)
    tol = rng.uniform(0.002, A.CODE_TOL, 200).astype(np.float32)
    rows, hits = band.lookup(q, tol)
    got = set(zip(rows.tolist(), hits.tolist()))
    want = set()
    for r in range(200):
        ok = np.where(np.max(np.abs(codes_all - q[r]), axis=1) <= tol[r])[0]
        want |= {(r, int(i)) for i in ok}
    assert got == want and len(want) > 0
