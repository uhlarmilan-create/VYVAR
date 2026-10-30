"""Step 4 (tag ast): plate-solver hypothesis / verification loop, the one RA-box rule and the
VSX-Gaia density (G03-01, G03-02, G03-04, G03-05, G03-45, G04-12, G05-04, G05-05, G05-50,
G09-38 and the same box pattern in catalog_crossmatch / target_comp_selection)."""
from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

import astrosolve as A  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _true_wcs(ra0, dec0, scale_as, rot_deg, nx, ny, mirror=False, crpix=None):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = list(crpix) if crpix is not None else [nx / 2, ny / 2]
    w.wcs.crval = [ra0, dec0]
    s, c, sn = scale_as / 3600, math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    cd = s * np.array([[-c, sn], [sn, c]])
    if mirror:
        cd[:, 0] *= -1
    w.wcs.cd = cd
    return w


def _unit(ra, dec):
    r, d = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], axis=-1)


def _grid_err(w_hyp, w_true, shape):
    ny, nx = shape
    gx, gy = np.meshgrid(np.linspace(0, nx - 1, 9), np.linspace(0, ny - 1, 7))
    rt, dt = w_true.all_pix2world(gx.ravel(), gy.ravel(), 0)
    px, py = w_hyp.all_world2pix(rt, dt, 0)
    return float(np.max(np.hypot(px - gx.ravel(), py - gy.ravel())))


# ---------------------------------------------------------------------------
# G03-01: hypothesis WCS about the tangent point of the similarity
# ---------------------------------------------------------------------------
def _hypothesis(w_true, pix, ra0, de0, shape):
    ra, dec = w_true.all_pix2world(pix[:, 0], pix[:, 1], 0)
    xi, eta = A.tan_project(ra, dec, ra0, de0)
    mirror = bool(np.linalg.det(w_true.wcs.cd) < 0)  # sky parity = conjugate pixel plane
    a, b, rms = A.fit_similarity(pix, np.column_stack([xi, eta]), mirror)
    return A.wcs_from_similarity(a, b, mirror, ra0, de0, shape)


@pytest.mark.parametrize("seed", range(16))
def test_g03_01_hypothesis_wcs_exact_about_tangent_point(seed):
    """Property: true TAN anywhere (0.5-10 "/px, Dec -80..85, rotation, mirror, CRVAL anywhere on
    or off the chip); 4 arbitrary pixels; tangent point = the true CRVAL. The pixel -> plane map
    is then exactly a similarity, so the hypothesis WCS reproduces the true grid."""
    rng = np.random.default_rng(seed)
    nx, ny = 4144, 2822
    w_true = _true_wcs(float(rng.uniform(0, 360)), float(rng.uniform(-80, 85)), float(rng.uniform(0.5, 10.0)),
                       float(rng.uniform(0, 360)), nx, ny, mirror=bool(rng.integers(2)),
                       crpix=(float(rng.uniform(-nx, 2 * nx)), float(rng.uniform(-ny, 2 * ny))))
    pix = np.column_stack([rng.uniform(0, nx, 4), rng.uniform(0, ny, 4)])
    w = _hypothesis(w_true, pix, *w_true.wcs.crval, (ny, nx))
    assert _grid_err(w, w_true, (ny, nx)) < 1e-3


@pytest.mark.parametrize("seed", range(8))
def test_g03_01_corner_pattern_high_dec(seed):
    """Blind-index geometry: a 4-star pattern in a chip corner, tangent point at the centroid of
    its sky positions, Dec 60-85, 0.5-1 "/px: the hypothesis is good to < 1 px over the whole chip
    (old construction: 2-60 px)."""
    rng = np.random.default_rng(100 + seed)
    nx, ny = 4144, 2822
    w_true = _true_wcs(float(rng.uniform(0, 360)), float(rng.uniform(60, 85)), float(rng.uniform(0.5, 1.0)),
                       float(rng.uniform(0, 360)), nx, ny, mirror=bool(rng.integers(2)))
    pix = np.column_stack([rng.uniform(0, 0.3 * nx, 4), rng.uniform(0, 0.3 * ny, 4)])
    ra, dec = w_true.all_pix2world(pix[:, 0], pix[:, 1], 0)
    v = _unit(ra, dec).mean(axis=0)
    ra0 = math.degrees(math.atan2(v[1], v[0])) % 360.0
    de0 = math.degrees(math.asin(v[2] / np.linalg.norm(v)))
    w = _hypothesis(w_true, pix, ra0, de0, (ny, nx))
    assert _grid_err(w, w_true, (ny, nx)) < 1.0


def test_g03_01_hinted_far_tangent_point_solves():
    """Hinted solve with the hint 0.6 deg off at Dec 75 on a wide field (the hypothesis error
    grew with hint offset: 42 px per deg at Dec 75)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_astrosolve as T  # noqa: PLC0415

    det, flux, shape, ra, dec, mag, w, hra, hdec = T._case(
        21, scale_as=10.0, rot=30.0, density=300.0, cone=5.0, hint_off=0.6, ra0=150.0, dec0=75.0)
    r = A.solve_against_catalog(det, flux, shape, ra, dec, mag, ra0=hra, dec0=hdec, sigma_px=1.5)
    assert r.ok, r.message
    assert T._max_err(r, w, shape) < 0.6


# ---------------------------------------------------------------------------
# G03-02: the 4 pattern stars do not vouch for their own hypothesis
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k_cat", [20, 30])
@pytest.mark.parametrize("seed", range(5))
def test_g03_02_quad_stars_excluded_from_verification(k_cat, seed):
    """K catalogue stars in the field, the 4 pattern stars matched exactly, the other detections
    random: without the quad stars the log-odds stays below the blind tune-up threshold. Counting
    them (old code, mutation) passes it (blind accept -> Gaia refinement of a chance pattern)."""
    rng = np.random.default_rng(seed)
    nx, ny, sig = 3552, 2664, 1.5
    w = _true_wcs(102.0, 32.0, 1.2, 0.0, nx, ny)
    cxy = np.column_stack([rng.uniform(0, nx - 1, k_cat), rng.uniform(0, ny - 1, k_cat)])
    ra, dec = w.all_pix2world(cxy[:, 0], cxy[:, 1], 0)
    det = np.vstack([cxy[:4], np.column_stack([rng.uniform(0, nx, 146), rng.uniform(0, ny, 146)])])
    mag = np.arange(k_cat, dtype=float)
    lo, _nm, _ = A.verify_hypothesis(w, det, ra, dec, mag, (ny, nx), sigma_px=sig, exclude=np.arange(4))
    assert lo < A.LOG_ODDS_TUNE_UP
    lo_self, _, _ = A.verify_hypothesis(w, det, ra, dec, mag, (ny, nx), sigma_px=sig)
    if k_cat == 20:
        assert lo_self >= A.LOG_ODDS_TUNE_UP  # the self-vouching the exclusion removes


# ---------------------------------------------------------------------------
# G03-04: a verified but non-credible hypothesis does not end the cone
# ---------------------------------------------------------------------------
def test_g03_04_rejected_hypothesis_keeps_searching(monkeypatch):
    """The first hypotheses that verify fail the refinement / credibility check; the true field
    in the same cone must still be solved (old: solve_against_catalog returned the first
    verified hypothesis and solve_hinted skipped the whole cone)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_astrosolve as T  # noqa: PLC0415

    det, flux, shape, ra, dec, mag, w, hra, hdec = T._case(1, scale_as=0.57, rot=37.0)
    real = A.credible_solution
    calls = {"n": 0}

    def flaky(res, n_det, sigma_px):
        calls["n"] += 1
        if calls["n"] <= 2:
            return False, "planted rejection"
        return real(res, n_det, sigma_px)

    monkeypatch.setattr(A, "credible_solution", flaky)
    r = A.solve_against_catalog(det, flux, shape, ra, dec, mag, ra0=hra, dec0=hdec, sigma_px=1.5)
    assert r.ok, r.message
    assert calls["n"] >= 3
    assert T._max_err(r, w, shape) < 0.6


# ---------------------------------------------------------------------------
# G03-05 / D36: one RA-box rule
# ---------------------------------------------------------------------------
def _cone_edge(ra0, dec0, r, n=1441):
    c = SkyCoord(ra0 * u.deg, dec0 * u.deg)
    e = c.directional_offset_by(np.linspace(0, 360, n) * u.deg, r * u.deg)
    return e.ra.deg, e.dec.deg


def _inside_boxes(ra, dec, boxes, dmin, dmax, tol=1e-7):
    ra = np.mod(ra, 360.0)
    in_ra = np.zeros(len(ra), dtype=bool)
    for a, b in boxes:
        in_ra |= (ra >= a - tol) & (ra <= b + tol)
    return in_ra & (dec >= dmin - tol) & (dec <= dmax + tol)


@pytest.mark.parametrize("seed", range(40))
def test_g03_05_ra_boxes_contain_cone(seed):
    """Property: random cones Dec -89..89 (half of them at |Dec| 60-89), r 0.1-15 deg, any RA:
    every point of the cone lies inside the returned boxes, and the RA intervals stay inside
    [0, 360]."""
    from catalog_db import ra_boxes_for_cone

    rng = np.random.default_rng(seed)
    dec0 = float(rng.uniform(-89, 89)) if seed % 2 else float(rng.choice([-1, 1]) * rng.uniform(60, 89))
    ra0, r = float(rng.uniform(0, 360)), float(rng.uniform(0.1, 15.0))
    boxes, dmin, dmax = ra_boxes_for_cone(ra0, dec0, r)
    assert all(0.0 <= a <= b <= 360.0 for a, b in boxes)
    era, edec = _cone_edge(ra0, dec0, r)
    assert np.all(_inside_boxes(era, edec, boxes, dmin, dmax))
    # interior points too
    c = SkyCoord(ra0 * u.deg, dec0 * u.deg).directional_offset_by(
        rng.uniform(0, 360, 500) * u.deg, rng.uniform(0, r, 500) * u.deg)
    assert np.all(_inside_boxes(c.ra.deg, c.dec.deg, boxes, dmin, dmax))


def test_g03_05_old_rule_was_too_narrow():
    """Regression anchor: at Dec 80, r 2 the RA half-width is asin(sin r / cos dec) = 11.6 deg
    (the solver used r / cos(equator-ward edge) = 9.6 deg)."""
    from catalog_db import cone_ra_half_width_deg

    assert math.isclose(cone_ra_half_width_deg(80.0, 2.0),
                        math.degrees(math.asin(math.sin(math.radians(2.0)) / math.cos(math.radians(80.0)))), rel_tol=1e-9)
    assert cone_ra_half_width_deg(80.0, 2.0) > 2.0 / math.cos(math.radians(78.0)) + 1.0
    assert cone_ra_half_width_deg(88.0, 2.0) is None and cone_ra_half_width_deg(-89.0, 1.5) is None


def _gaia_db(path, ra, dec, mag):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE gaia_dr3 (source_id INTEGER PRIMARY KEY, ra REAL, dec REAL, g_mag REAL, "
                "bp_mag REAL, rp_mag REAL, bp_rp REAL, var_flag TEXT)")
    con.executemany("INSERT INTO gaia_dr3 (source_id, ra, dec, g_mag) VALUES (?,?,?,?)",
                    [(i, float(a), float(b), float(m)) for i, (a, b, m) in enumerate(zip(ra, dec, mag))])
    con.execute("CREATE INDEX idx_ra_dec ON gaia_dr3 (ra, dec)")
    con.execute("CREATE INDEX idx_g_mag ON gaia_dr3 (g_mag)")
    con.commit()
    con.close()
    return str(path)


def _sky_cap(rng, ra0, dec0, rmax, n):
    """n uniform positions on the sphere within rmax deg of (ra0, dec0)."""
    c = SkyCoord(ra0 * u.deg, dec0 * u.deg)
    rr = np.degrees(np.arccos(1 - rng.uniform(0, 1, n) * (1 - math.cos(math.radians(rmax)))))
    p = c.directional_offset_by(rng.uniform(0, 360, n) * u.deg, rr * u.deg)
    return p.ra.deg, p.dec.deg


@pytest.mark.parametrize("ra0,dec0,r", [(10.0, 82.0, 4.0), (0.3, 20.0, 2.0), (200.0, -70.0, 7.5), (100.0, 87.0, 4.0)])
def test_g03_05_g03_45_query_gaia_cone_exact_brightest(tmp_path, ra0, dec0, r):
    """query_gaia_cone returns exactly the ``want`` brightest stars of the great-circle cone, at
    high Dec, across RA 0 and over the pole, including stars fainter than the old G 17.5 cap."""
    rng = np.random.default_rng(int(ra0 + dec0))
    ra, dec = _sky_cap(rng, ra0, dec0, r * 1.6, 6000)
    mag = rng.uniform(12.0, 20.0, len(ra))
    db = _gaia_db(tmp_path / "g.db", ra, dec, mag)
    sep = SkyCoord(ra0 * u.deg, dec0 * u.deg).separation(SkyCoord(ra * u.deg, dec * u.deg)).deg
    in_cone = np.where(sep <= r)[0]
    for want in (50, 1500, 10 ** 6):
        qra, qdec, qmag = A.query_gaia_cone(db, ra0, dec0, r, want=want)
        expect = np.sort(mag[in_cone])[:want]
        assert np.allclose(qmag, expect)
    assert qmag.max() > 17.5


# ---------------------------------------------------------------------------
# G04-12: reconcile reference population at RA 0
# ---------------------------------------------------------------------------
def test_g04_12_reference_population_across_ra0(tmp_path):
    """The same field at RA 0.1 and at RA 180.1 gives the same reference count; the old corner
    min/max box at RA 0.1 was the complement of the field."""
    from dao_reconcile import apply_footprint_filter, query_reference_population

    rng = np.random.default_rng(3)
    nx, ny = 2000, 1500
    out = []
    for ra_c in (0.1, 180.1):
        ra, dec = _sky_cap(rng, ra_c, 50.0, 1.2, 8000)
        mag = rng.uniform(10, 17, len(ra))
        db = _gaia_db(tmp_path / f"g{int(ra_c)}.db", ra, dec, mag)
        w = _true_wcs(ra_c, 50.0, 2.0, 30.0, nx, ny)
        ref = query_reference_population(db, w, nx, ny, mag_limit=None)
        ref, _ = apply_footprint_filter(ref, w, nx, ny, fwhm_px=3.0, edge_margin_fwhm=0.0)
        x, y = w.all_world2pix(ra, dec, 0)
        truth = int(np.sum((x >= 0) & (x < nx) & (y >= 0) & (y < ny)))
        n_in = int(ref["_in_frame"].astype(bool).sum()) if len(ref) else 0
        out.append((n_in, truth))
    for n_in, truth in out:
        assert truth > 0 and abs(n_in - truth) <= 2


def test_g04_12_frame_cone_holds_pole_ward_bulge():
    """Wide chip at Dec 80: every border point is inside the chip cone (the corner box missed
    the pole-ward bulge of the edges)."""
    from dao_reconcile import frame_sky_cone_deg

    nx, ny = 4144, 2822
    w = _true_wcs(33.0, 80.0, 9.8, 0.0, nx, ny)
    ra_c, de_c, rad = frame_sky_cone_deg(w, nx, ny)
    xs = np.linspace(-0.5, nx - 0.5, 5000)
    for bx, by in ((xs, np.full_like(xs, -0.5)), (xs, np.full_like(xs, ny - 0.5))):
        ra, dec = w.all_pix2world(bx, by, 0)
        sep = SkyCoord(ra_c * u.deg, de_c * u.deg).separation(SkyCoord(ra * u.deg, dec * u.deg)).deg
        assert np.all(sep <= rad)
    corners_ra, corners_dec = w.all_pix2world([0, nx - 1, nx - 1, 0], [0, 0, ny - 1, ny - 1], 0)
    _top_ra, top_dec = w.all_pix2world(xs, np.full_like(xs, ny - 1.0), 0)
    assert top_dec.max() > np.max(corners_dec) + 0.1  # the bulge the corner box missed


# ---------------------------------------------------------------------------
# G05-50 and the VSX point query (catalog_crossmatch)
# ---------------------------------------------------------------------------
def _vsx_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE vsx_data (oid INTEGER PRIMARY KEY, name TEXT, ra_deg REAL, dec_deg REAL, "
                "var_type TEXT, period REAL, mag_max REAL, mag_min REAL)")
    con.executemany("INSERT INTO vsx_data VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def test_g05_50_capped_box_keeps_nearest(tmp_path):
    """Box with 400 rows across RA 0, max_rows 300 -> the 300 nearest to the box centre."""
    from catalog_db import query_local_vsx

    rng = np.random.default_rng(1)
    ra = np.mod(rng.uniform(-1, 1, 400), 360.0)
    dec = rng.uniform(-1, 1, 400)
    db = _vsx_db(tmp_path / "o.db", [(i, f"V{i}", float(a), float(b), "EW", 0.3, 12.0, 12.5)
                                     for i, (a, b) in enumerate(zip(ra, dec))])
    rows = query_local_vsx(db, ra_min=-1.0, ra_max=1.0, dec_min=-1.0, dec_max=1.0, max_rows=300)
    got = {r["oid"] for r in rows}
    sep = SkyCoord(0 * u.deg, 0 * u.deg).separation(SkyCoord(ra * u.deg, dec * u.deg)).deg
    assert got == set(np.argsort(sep)[:300].tolist())


def test_vsx_point_query_high_dec(tmp_path):
    """catalog_crossmatch VSX point query at Dec 85: a star 50" away along RA is found (the
    cos(dec) clamp at 0.2 cut the box below the radius above |Dec| 78.5)."""
    from catalog_crossmatch import _query_vsx_local_point

    c = SkyCoord(120.0 * u.deg, 85.0 * u.deg)
    p = c.directional_offset_by(90 * u.deg, 50 * u.arcsec)
    db = _vsx_db(tmp_path / "o.db", [(1, "V1", float(p.ra.deg), float(p.dec.deg), "EW", 0.3, 12.0, 12.5)])
    out = _query_vsx_local_point(db, c, 60.0)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# G09-38: ePSF build-star isolation near the pole and across RA 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ra_s,dec_s,pa", [(40.0, 85.0, 90.0), (359.9995, 10.0, 90.0), (10.0, 89.5, 90.0)])
def test_g09_38_isolation_finds_neighbour(tmp_path, monkeypatch, ra_s, dec_s, pa):
    import pandas as pd

    import psf_photometry as P

    monkeypatch.setattr(P, "plate_scale_from_fits", lambda _p: 1.0)
    star = SkyCoord(ra_s * u.deg, dec_s * u.deg)
    nb = star.directional_offset_by(pa * u.deg, 6 * u.arcsec)  # 2 FWHM at 3 px, 1 "/px
    pd.DataFrame({"ra_deg": [ra_s, nb.ra.deg], "dec_deg": [dec_s, nb.dec.deg], "mag": [12.0, 12.5]}).to_csv(
        tmp_path / P._CONE_CATALOG_NAME, index=False)
    cand = pd.DataFrame({"ra_deg": [ra_s], "dec_deg": [dec_s], "mag": [12.0], "x": [100.0], "y": [100.0]})
    funnel: dict = {}
    *_, out = P._epsf_isolate_build_stars(csvpath=tmp_path / "m.csv", funnel=funnel, fwhm_px=3.0, min_stars=0,
                                      mpath=tmp_path / "ms.fits", psf_stars_df=cand)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# G05-04 / G05-05: VSX -> Gaia density and likelihood ratio
# ---------------------------------------------------------------------------
def test_g05_04_reliability_units():
    """A 0.2" counterpart in a 1e4 deg^-2 field is a "high" match (was "poor": the chance term
    was per degree, the true term per arcsec); reliability decreases with separation and with
    density."""
    import vsx_gaia_crossmatch as V

    q, w, sn, sb = 0.9, 0.7, 0.3, 0.8
    rel = V._reliability_from_sep(0.2, q, w, sn, sb, 1e4)
    assert V._quality_from_reliability(rel) == "high"
    seps = np.linspace(0.05, 5.0, 60)
    r_sep = [V._reliability_from_sep(float(s), q, w, sn, sb, 1e4) for s in seps]
    assert all(b <= a + 1e-12 for a, b in zip(r_sep, r_sep[1:]))
    r_rho = [V._reliability_from_sep(1.0, q, w, sn, sb, float(rho)) for rho in np.geomspace(1e2, 1e6, 20)]
    assert all(b <= a + 1e-12 for a, b in zip(r_rho, r_rho[1:]))


def test_g05_04_mixture_pdf_normalised():
    import vsx_gaia_crossmatch as V

    r = np.linspace(0.0, 600.0, 600001)
    pdf = V._mixture_pdf(r, 0.8, 0.6, 0.4, 1.2, 2e4)
    assert abs(np.trapezoid(pdf, r) - 1.0) < 1e-3


def _vsx_field(rot_deg, rng_seed=5):
    import vsx_gaia_crossmatch as V

    rng = np.random.default_rng(rng_seed)
    nx, ny = 4000, 3000
    ra_c, de_c = 200.0, 45.0
    ra, dec = _sky_cap(rng, ra_c, de_c, 3.2, 120000)  # ~1.1e4 deg^-2
    w = _true_wcs(ra_c, de_c, 2.6, rot_deg, nx, ny)
    ra_min, ra_max, de_min, de_max = V.frame_bbox_radec_limits(w, nx, ny, margin_px=50.0)
    box = (ra >= ra_min) & (ra <= ra_max) & (dec >= de_min) & (dec <= de_max)
    g_ra, g_dec = ra[box], dec[box]
    # the same 120 VSX stars for every rotation: Gaia stars within 1 deg of the centre (on chip)
    sep_c = SkyCoord(ra_c * u.deg, de_c * u.deg).separation(SkyCoord(ra * u.deg, dec * u.deg)).deg
    pick = np.random.default_rng(77).choice(np.where(sep_c < 1.0)[0], 120, replace=False)
    off = np.random.default_rng(78).normal(0, 0.35, (len(pick), 2)) / 3600.0
    v_ra = ra[pick] + off[:, 0] / np.cos(np.radians(dec[pick]))
    v_dec = dec[pick] + off[:, 1]
    sid = np.where(box)[0].astype(str)  # global identity of the queried rows
    area = V.field_area_deg2_from_wcs(w, nx, ny)
    n_in = V.count_in_field(w, nx, ny, g_ra, g_dec)
    rows, diag = V.match_vsx_to_gaia_density_aware(v_ra, v_dec, sid, g_ra, g_dec, field_area_deg2=area,
                                                   n_gaia_in_field=n_in)
    true_rho = 120000 / (2 * math.pi * (1 - math.cos(math.radians(3.2))) * (180 / math.pi) ** 2)
    accepted = {(i, r.catalog_id) for i, r in enumerate(rows) if r.accepted}
    return diag, accepted, true_rho, len(g_ra) / area


def test_g05_05_rho_independent_of_rotation():
    d0, acc0, true_rho, box_rho0 = _vsx_field(0.0)
    d45, acc45, _, box_rho45 = _vsx_field(45.0)
    assert abs(d0.rho_per_deg2 / d45.rho_per_deg2 - 1) < 0.05
    assert abs(d45.rho_per_deg2 / true_rho - 1) < 0.05
    assert box_rho45 > 1.5 * true_rho  # what the bbox count gave
    assert acc0 == acc45


def test_g05_05_count_must_fit_the_query():
    import vsx_gaia_crossmatch as V

    with pytest.raises(V.VsxGaiaCrossmatchError):
        V.match_vsx_to_gaia_density_aware(np.array([1.0] * 6), np.array([1.0] * 6), np.array(["1"]),
                                          np.array([1.0]), np.array([1.0]), field_area_deg2=1.0, n_gaia_in_field=5)
