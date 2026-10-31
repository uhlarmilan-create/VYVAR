"""Step 5 batch D (tag sci, Milan 2026-10-30): merges that change published numbers.

CCD-ONE-2026-10 (G06-20 / G08-28): one CCD equation (ccd_noise) with the sky-estimation term and the pi/2
variance of a median sky, the real number of sky pixels, no default gain / read noise; LC rows never keep an
error of another aperture.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ccd_noise import ccd_rel_error, ccd_sky_variance_adu2, ccd_variance_adu2


# ----------------------------------------------------------------------------------------------- G06-20
def _mc_pulls(F_e: float, sky_e: float, A: int, n_sky: int, gain: float, rn: float, n_trials: int, seed: int):
    """Monte Carlo of an aperture of A pixels on a star of F_e electrons and a sky of sky_e e-/px, the sky
    subtracted with the MEDIAN of n_sky annulus pixels; Poisson + Gaussian read noise. Returns
    (flux estimates [ADU], true flux [ADU])."""
    rng = np.random.default_rng(seed)
    star_px = np.full(A, F_e / A)
    ap = rng.poisson(sky_e + star_px, size=(n_trials, A)) + rng.normal(0.0, rn, size=(n_trials, A))
    ann = rng.poisson(sky_e, size=(n_trials, n_sky)) + rng.normal(0.0, rn, size=(n_trials, n_sky))
    flux_adu = ap.sum(axis=1) / gain - A * np.median(ann / gain, axis=1)
    return flux_adu, F_e / gain


@pytest.mark.parametrize(
    "F_e,sky_e,A,n_sky",
    [
        (2.0e4, 400.0, 50, 400),  # typical geometry
        (2.0e3, 800.0, 60, 80),  # small annulus: the sky-estimation term dominates
        (2.0e5, 50.0, 30, 300),  # photon limited
    ],
)
def test_ccd_variance_matches_monte_carlo_with_median_sky(F_e, sky_e, A, n_sky):
    gain, rn = 0.8, 3.5
    est, true = _mc_pulls(F_e, sky_e, A, n_sky, gain, rn, n_trials=6000, seed=1)
    var = ccd_variance_adu2(true, sky_e / gain, float(A), float(n_sky), gain=gain, read_noise=rn)
    pull = (est - true) / math.sqrt(var)
    assert abs(float(np.mean(pull))) < 0.05
    assert float(np.std(pull)) == pytest.approx(1.0, abs=0.035)


def test_median_factor_is_needed_and_old_howell_form_underestimates():
    """Small annulus: without pi/2 the pull width is > 1; the former form without the sky-estimation term
    (copies 1-3: F/g + A S/g + A (RN/g)^2) underestimates it much more."""
    F_e, sky_e, A, n_sky, gain, rn = 2.0e3, 800.0, 60, 80, 0.8, 3.5
    est, true = _mc_pulls(F_e, sky_e, A, n_sky, gain, rn, n_trials=6000, seed=2)
    v_med = ccd_variance_adu2(true, sky_e / gain, A, n_sky, gain=gain, read_noise=rn)
    v_mean = ccd_variance_adu2(true, sky_e / gain, A, n_sky, gain=gain, read_noise=rn, sky_estimator="mean")
    v_old = true / gain + A * (sky_e / gain) / gain + A * (rn / gain) ** 2
    s = float(np.std(est - true))
    assert s / math.sqrt(v_med) == pytest.approx(1.0, abs=0.035)
    assert s / math.sqrt(v_mean) > 1.07
    assert s / math.sqrt(v_old) > 1.3


def test_ccd_equation_closed_form_and_vectorised():
    F, S, A, N, g, rn = 1.0e4, 100.0, 28.0, 400.0, 3.17, 7.6
    exp = F / g + A * (1 + (math.pi / 2) * A / N) * (S / g + (rn / g) ** 2)
    assert ccd_variance_adu2(F, S, A, N, gain=g, read_noise=rn) == pytest.approx(exp, rel=1e-15)
    v = ccd_variance_adu2(np.array([F, F, -5.0, np.nan]), np.array([S, -3.0, S, S]), A, np.array([N, N, N, N]),
                          gain=g, read_noise=rn)
    assert v[0] == pytest.approx(exp, rel=1e-15)
    assert v[1] == pytest.approx(F / g + A * (1 + (math.pi / 2) * A / N) * (rn / g) ** 2)  # negative sky -> 0
    assert v[2] == pytest.approx(ccd_sky_variance_adu2(S, A, N, gain=g, read_noise=rn))  # F <= 0: no source term
    assert np.isnan(v[3])
    assert np.isnan(ccd_variance_adu2(F, S, A, 0.0, gain=g, read_noise=rn))  # no sky pixels: no estimate
    assert np.isnan(ccd_rel_error(0.0, S, A, N, gain=g, read_noise=rn))


@pytest.mark.parametrize("gain,rn", [(None, 3.0), (float("nan"), 3.0), (0.0, 3.0), (1.0, None), (1.0, -1.0)])
def test_no_default_gain_or_read_noise(gain, rn):
    with pytest.raises(ValueError):
        ccd_variance_adu2(1000.0, 10.0, 20.0, 300.0, gain=gain, read_noise=rn)


def test_annulus_sky_count_is_the_real_number_of_sky_pixels():
    """n_sky = finite sky pixels of the median (source mask and footprint cut), not pi (r_out^2 - r_in^2)."""
    from photutils.aperture import CircularAnnulus

    from aperture_pertarget import measure_flux_on_f_grid

    rng = np.random.default_rng(5)
    img = rng.normal(500.0, 5.0, size=(120, 120))
    xy = np.array([[60.0, 60.0]])
    kw = dict(f_grid=[1.0], fwhm_px=3.0, annulus_inner_fwhm=2.7, annulus_outer_fwhm=5.2, return_n_sky=True)
    _f, _s, _e, n0 = measure_flux_on_f_grid(img, xy, **kw)
    n_geo = int(np.sum(CircularAnnulus(xy, 8.1, 15.6).to_mask(method="center")[0].data > 0))
    assert int(n0[0]) == n_geo
    cov = np.ones_like(img, dtype=bool)
    cov[:, :55] = False  # the left part of the annulus has no data
    _f, _s, _e, n1 = measure_flux_on_f_grid(img, xy, covered=cov, **kw)
    assert 0 < int(n1[0]) < n_geo


def _night_one_frame():
    from aperture_pertarget import ApertureGridNight

    night = ApertureGridNight(f_grid=[1.0], fwhm_night_px=4.0)
    night.fwhm_by_frame = {"a": 4.0}
    night.frames = {"a": {"T": {1.0: 1.0e4}, "C": {1.0: float("nan")}, "D": {1.0: 5.0e3}}}
    night.sky_pp = {"a": {"T": {1.0: 100.0}, "C": {1.0: float("nan")}, "D": {1.0: float("nan")}}}
    night.sky_n = {"a": {"T": 700, "C": 0, "D": 0}}
    return night


def test_lc_rows_never_keep_the_proc_error_of_another_aperture():
    """G08-28: the old code kept the per-frame proc err (other aperture, other error model) when the grid sky
    or flux was missing; now such rows have err = NaN with mag_inst = NaN."""
    from aperture_pertarget import apply_grid_fluxes_to_frames

    df = pd.DataFrame([{"catalog_id": c, "source_file": "proc_a.csv", "mag_inst": 0.0, "err": 9.9, "flag": "normal"}
                       for c in ("T", "C", "D")]).set_index("catalog_id", drop=False)
    out = apply_grid_fluxes_to_frames(df, _night_one_frame(), catalog_ids=["T", "C", "D"], f_star=1.0, gain=2.0,
                                      read_noise=4.0)
    A = math.pi * 16.0
    assert out.loc["T", "err"] == pytest.approx(
        math.sqrt(1.0e4 / 2 + A * (1 + (math.pi / 2) * A / 700) * (100.0 / 2 + 4.0)) / 1.0e4, rel=1e-14)
    for c in ("C", "D"):
        assert np.isnan(out.loc[c, "err"]) and np.isnan(out.loc[c, "mag_inst"])
    assert out.loc["D", "flag"] == "no_data"  # a flux without a sky has no error: not published


def test_lc_err_requires_gain():
    from aperture_pertarget import apply_grid_fluxes_to_frames

    df = pd.DataFrame([{"catalog_id": "T", "source_file": "proc_a.csv", "mag_inst": 0.0, "err": 9.9}])
    with pytest.raises(ValueError):
        apply_grid_fluxes_to_frames(df, _night_one_frame(), catalog_ids=["T"], f_star=1.0, gain=None, read_noise=4.0)


def test_check_star_photon_sigma_is_the_row_err_not_a_default_gain_recomputation():
    """The old fallback recomputed a missing err with gain 1, RN 10 and area pi x 9 (G06-20 copy 8)."""
    from check_star_kmag import build_comp_photon_mag_from_frames

    rows = pd.DataFrame([
        {"catalog_id": "1", "source_file": "a", "err": 0.01, "dao_flux": 1e4, "sky_adu_per_px_annulus": 100.0},
        {"catalog_id": "1", "source_file": "b", "err": float("nan"), "dao_flux": 1e4, "sky_adu_per_px_annulus": 100.0},
    ])
    m = build_comp_photon_mag_from_frames(rows, ["1"], ["a", "b"])["1"]
    assert m[0] == pytest.approx(2.5 / math.log(10) * 0.01, rel=1e-3)
    assert np.isnan(m[1])


def test_proc_photometry_writes_sky_pixel_count():
    from photometry_defs import SKY_N_PX_COL
    from proc_frame_store import PROC_STORE_COLS

    assert SKY_N_PX_COL == "sky_n_px" and SKY_N_PX_COL in PROC_STORE_COLS


# ----------------------------------------------------------------------------------------------- G10-20
def _star(shape, x0, y0, flux, sig):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return flux / (2 * np.pi * sig**2) * np.exp(-0.5 * (((xx - x0) / sig) ** 2 + ((yy - y0) / sig) ** 2))


def _write_frame(path, img):
    from astropy.io import fits

    h = fits.Header()
    h["NAXIS1"], h["NAXIS2"] = img.shape[1], img.shape[0]
    for k, v in enumerate((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):
        h[f"VY_AM{k}"] = v
    fits.writeto(path, img.astype(np.float32), h, overwrite=True)


def test_transient_flux_is_the_production_engine_and_ignores_a_neighbour_in_the_annulus(tmp_path):
    """G10-20: the transient forced photometry is ``measure_flux_on_f_grid`` (source-masked annulus). With
    three bright neighbours 11-13 px away (inside the 2.7-5.2 FWHM annulus) the old private copy (median of
    the unmasked annulus) lost > 1 % of the flux; the production engine stays at the isolated-star flux."""
    from aperture_pertarget import measure_flux_on_f_grid
    from transient_field import _measure_one

    shape, fw, sig = (200, 200), 3.0, 3.0 / 2.3548
    rng = np.random.default_rng(11)
    base = rng.normal(500.0, 3.0, shape)
    iso = base + _star(shape, 100.0, 100.0, 5.0e3, sig)
    crowd = iso + sum(_star(shape, x, y, 3.0e5, sig) for x, y in ((111.0, 100.0), (91.0, 107.0), (97.0, 88.0)))
    _write_frame(tmp_path / "iso.fits", iso)
    _write_frame(tmp_path / "crowd.fits", crowd)
    xy = np.array([[100.0, 100.0]])
    k, a_in, a_out, g, rn = 1.5, 2.7, 5.2, 0.8, 3.5
    r_iso = _measure_one((str(tmp_path / "iso.fits"), xy, fw, k, a_in, a_out, g, rn))
    r_crowd = _measure_one((str(tmp_path / "crowd.fits"), xy, fw, k, a_in, a_out, g, rn))
    # same engine as the light curves
    fl, sk, _e, ns = measure_flux_on_f_grid(crowd, xy, f_grid=[k], fwhm_px=fw, annulus_inner_fwhm=a_in,
                                            annulus_outer_fwhm=a_out, covered=np.ones(shape, bool), return_n_sky=True)
    assert r_crowd[0, 0] == pytest.approx(fl[k][0], rel=1e-6)
    area = math.pi * (k * fw) ** 2
    assert r_crowd[0, 1] == pytest.approx(math.sqrt(ccd_variance_adu2(fl[k][0], sk[k][0], area, ns[0], gain=g,
                                                                       read_noise=rn)), rel=1e-6)
    # the neighbour no longer biases the sky
    new_bias = r_crowd[0, 0] / r_iso[0, 0] - 1.0
    assert abs(new_bias) < 0.005
    # the old private copy, for the record of the bias it had
    from photutils.aperture import ApertureStats, CircularAnnulus, CircularAperture, aperture_photometry

    sky_old = float(ApertureStats(crowd, CircularAnnulus(xy, a_in * fw, a_out * fw)).median[0])
    f_old = float(aperture_photometry(crowd, CircularAperture(xy, k * fw))["aperture_sum"][0]) - sky_old * area
    old_bias = f_old / r_iso[0, 0] - 1.0
    assert old_bias < -0.01 and abs(old_bias) > 3 * abs(new_bias)


# ----------------------------------------------------------------------------------------------- G08-55
def _comps_with_one_outlier_frame(amp: float, n_c: int = 6, n_f: int = 200, seed: int = 4):
    rng = np.random.default_rng(seed)
    bjd = 2460000.0 + np.linspace(0.0, 0.3, n_f)
    lc = {f"c{i}": 12.0 + 0.005 * rng.standard_normal(n_f) for i in range(n_c)}
    lc["c0"][97] += amp
    return lc, {c: bjd.copy() for c in lc}


def test_one_outlier_frame_no_longer_makes_a_comp_suspect():
    """G08-55: one 2-mag outlier frame gave std(diff)/sqrt2 = 0.14 > 0.1 (suspect -> comp_smart replaced the
    comp); the robust p2p stays at the white-noise level."""
    from photometry_lightcurve import check_comparison_stability

    lc, bjd = _comps_with_one_outlier_frame(2.0)
    q = check_comparison_stability(lc, comp_bjd=bjd, max_comp_p2p=0.1)
    assert q["c0"]["quality"] == "good"
    assert q["c0"]["rms_p2p"] < 0.01


def test_comp_smart_keeps_a_comp_with_one_outlier_frame():
    from comp_smart import smart_select

    lc, bjd = _comps_with_one_outlier_frame(2.0)
    lc["r0"] = 12.0 + 0.005 * np.random.default_rng(9).standard_normal(200)
    bjd["r0"] = bjd["c0"].copy()
    final, log = smart_select([f"c{i}" for i in range(6)], ["r0"], lc, bjd, n_comp_min=3, n_comp_max=6,
                              stability_kwargs={"max_comp_p2p": 0.1})
    assert "c0" in final and "r0" not in log["added"]


def test_robust_p2p_recovers_white_noise_sigma_and_ignores_slow_trend():
    from stats_core import robust_p2p_scatter

    rng = np.random.default_rng(1)
    est = [robust_p2p_scatter(0.01 * rng.standard_normal(400) + np.linspace(0, 0.2, 400), min_points=5)
           for _ in range(200)]
    assert float(np.mean(est)) == pytest.approx(0.01, rel=0.03)
    assert np.isnan(robust_p2p_scatter([1.0, np.nan, 2.0], min_points=3))


def test_epsf_p2p_is_bit_identical_to_the_old_copy():
    from epsf_lc_compare import p2p_scatter

    def old(values):
        m = np.asarray(values, dtype=np.float64)
        m = m[np.isfinite(m)]
        if m.size < 5:
            return float("nan")
        d = np.diff(m)
        return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0))

    rng = np.random.default_rng(3)
    for n in (3, 5, 6, 57, 300):
        v = 14.0 + 0.02 * rng.standard_normal(n)
        v[rng.random(n) < 0.1] = np.nan
        a, b = p2p_scatter(v), old(v)
        assert (np.isnan(a) and np.isnan(b)) or a == b


# ----------------------------------------------------------------------------------------------- G09-26
def test_sector_aov_fap_is_bit_identical_to_the_old_definition():
    from scipy.stats import f as fdist

    from tess_analysis import _N_HARM, aov_fap, frequency_grid, n_independent_frequencies

    rng = np.random.default_rng(2)
    t = np.sort(2459000.0 + rng.uniform(0, 25.0, 3000))
    freq = frequency_grid(t, 120.0 / 86400.0)
    for theta, n in ((3.0, 3000), (12.0, 3000), (40.0, 500)):
        p1 = float(fdist.sf(theta, 2 * _N_HARM, n - 2 * _N_HARM - 1))
        m_ind = max(1.0, (freq[-1] - freq[0]) * float(np.ptp(t)))
        old = float(-np.expm1(m_ind * np.log1p(-min(p1, 1 - 1e-16))))
        assert aov_fap(theta, n, n_independent_frequencies(freq[0], freq[-1], float(np.ptp(t)))) == old


def test_combined_fap_uses_the_searched_grid_not_the_cadence_nyquist():
    """G09-26: the combined FAP used M = f_Nyquist(median cadence) x span (no f_min, a different f_max);
    now M = (union f_max - union f_min) x span, the grid the sectors searched."""
    import tess_analysis as ta

    rng = np.random.default_rng(6)
    P, secs = 0.731, []
    t_all, m_all, e_all = [], [], []
    for k, (t0, cad) in enumerate(((2459000.0, 120.0), (2459700.0, 600.0))):
        t = t0 + np.arange(0, 20.0, cad / 86400.0)
        m = 12.0 + 0.05 * np.sin(2 * np.pi * t / P) + 0.003 * rng.standard_normal(t.size)
        e = np.full(t.size, 0.003)
        s = ta.TessSector(sector=k + 1, cadence_s=cad, n_points=t.size, t_start=t[0], t_end=t[-1], contamination=0.0,
                          n_aperture_px=9, prf_sigma_px=1.0)
        s.period = ta.PeriodSolution("AoV", P * (1 + 1e-5 * k), 1e-5, 1e-30, 100.0)
        secs.append((s, {"t": t, "m": m, "e": e, "mult": 1}))
        t_all.append(t); m_all.append(m); e_all.append(e)
    t, m, e = (np.concatenate(x) for x in (t_all, m_all, e_all))
    best, _mult = ta.combine_sectors(t, m, e, secs, None)
    lims = [ta.frequency_limits(a["t"], s.cadence_s / 86400.0) for s, a in secs]
    m_new = ta.n_independent_frequencies(min(lo for lo, _ in lims), max(hi for _, hi in lims), float(np.ptp(t)))
    m_old = max(1.0, (1.0 / (2 * np.median([s.cadence_s for s, _ in secs]) / 86400.0)) * float(np.ptp(t)))
    assert m_new != m_old
    assert best.fap == ta.aov_fap(best.power, t.size, m_new)


# ----------------------------------------------------------------------------------------------- G07-33
@pytest.mark.parametrize("sep_fwhm,c_moffat", [(1.5, 0.458), (2.0, 0.153), (3.0, 0.0129)])
def test_gs11_dilution_uses_the_moffat_annulus_model(sep_fwhm, c_moffat):
    """G07-33: equal neighbour, aperture 1.5 FWHM, annulus 2.7-5.2 FWHM, beta 2.7 (triage t733.py). The old
    Gaussian model gave 0.444 / 0.097 / 0.0001; the comparison-star model gives 0.458 / 0.153 / 0.0129."""
    from unittest.mock import patch

    from dilution import compute_dilution_factor

    fw = 3.0
    nb = [{"g_mag": 12.0, "ra": 10.0 + sep_fwhm * fw / 3600.0, "dec": 0.0, "sep_arcsec": sep_fwhm * fw}]
    with patch("dilution.query_gaia_neighbors", return_value=nb):
        r = compute_dilution_factor(10.0, 0.0, 12.0, 1.5 * fw, "dummy.db", fwhm_arcsec=fw, beta=2.7,
                                    annulus_arcsec=(2.7 * fw, 5.2 * fw))
    assert r["contamination_frac"] == pytest.approx(c_moffat, abs=0.002)
    assert r["dilution_factor"] == pytest.approx(1.0 / (1.0 + c_moffat), abs=0.002)


def test_gaussian_dilution_model_is_gone():
    import dilution

    assert not hasattr(dilution, "enclosed_fraction")


# ----------------------------------------------------------------------------------------------- G04-26
def _tan_wcs(shape, scale_arcsec=1.5):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [shape[1] / 2 + 1, shape[0] / 2 + 1]
    w.wcs.crval = [150.0, 30.0]
    w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    return w


def test_per_frame_pass2_keeps_one_detection_per_centroid():
    """G04-26: two MASTERSTAR seeds 1.2 px apart on one star both accepted the same DAO centroid and the
    per-frame pass 2 kept two rows at one pixel (full flux twice); now the G04-07 rule keeps one."""
    from pipeline_catalog import _dao_targeted_pass2_unmatched_gaia

    shape = (160, 160)
    rng = np.random.default_rng(8)
    img = rng.normal(0.0, 5.0, shape) + _star(shape, 80.0, 80.0, 2.0e4, 3.0 / 2.3548)
    w = _tan_wcs(shape)
    ra, dec = w.pixel_to_world_values(np.array([80.0, 81.2]), np.array([80.0, 80.0]))
    cat = pd.DataFrame({"ra_deg": ra, "dec_deg": dec, "mag": [13.0, 14.5], "catalog_id": ["1", "2"]})
    tbl, n_unm, n_raw = _dao_targeted_pass2_unmatched_gaia(
        img, None, cat_df=cat, wcs_obj=w, bfac=1, fwhm_px=3.0, pass2_sigma=5.0, pass2_center_tol_px=2.0,
        match_sep_arcsec=2.0, wpx=shape[1], h=shape[0])
    assert n_unm == 2 and n_raw == 2  # both seeds accepted the centroid
    assert tbl is not None and len(tbl) == 1
    assert int(tbl["vy_dao_pass"][0]) == 2


def test_masterstar_dedup_is_bit_identical_after_sharing_the_rule():
    from scipy.spatial import cKDTree

    from masterstar_gaia_accounting import _dedup_pass2_by_position, _norm_cid_strip_dotzero

    def old(rows, gaia_chip, sep_px=0.75):
        if len(rows) <= 1:
            return rows
        gmag = {}
        gm = pd.to_numeric(gaia_chip.get("g_mag", gaia_chip.get("mag")), errors="coerce").to_numpy(dtype=np.float64)
        for cid, g in zip(gaia_chip["catalog_id"].map(_norm_cid_strip_dotzero), gm):
            if cid and math.isfinite(float(g)):
                gmag[cid] = float(g)

        def _key(r):
            cid = _norm_cid_strip_dotzero(r.get("vy_seed_catalog_id", ""))
            return (gmag.get(cid, math.inf), cid)

        order = sorted(range(len(rows)), key=lambda i: _key(rows[i]))
        xy = np.array([[float(r["x_full"]), float(r["y_full"])] for r in rows], dtype=np.float64)
        tree = cKDTree(xy)
        kept = np.zeros(len(rows), dtype=bool)
        for i in order:
            if any(kept[j] for j in tree.query_ball_point(xy[i], r=float(sep_px))):
                continue
            kept[i] = True
        return [rows[i] for i in range(len(rows)) if kept[i]]

    rng = np.random.default_rng(12)
    n = 400
    xy = rng.uniform(0, 60, size=(n, 2))
    xy[1::3] = xy[0::3][: len(xy[1::3])] + rng.normal(0, 0.3, size=(len(xy[1::3]), 2))  # near-duplicates
    chip = pd.DataFrame({"catalog_id": [str(1000 + i) for i in range(n)], "g_mag": rng.uniform(10, 18, n)})
    chip.loc[::7, "g_mag"] = np.nan
    rows = [{"x_full": float(x), "y_full": float(y), "vy_seed_catalog_id": str(1000 + i)} for i, (x, y) in enumerate(xy)]
    a, b = _dedup_pass2_by_position(rows, chip), old(rows, chip)
    assert [r["vy_seed_catalog_id"] for r in a] == [r["vy_seed_catalog_id"] for r in b]
    assert len(a) < n


# ----------------------------------------------------------------------------------------------- G01-16
def _light(path, header_cards=None, shape=(200, 200)):
    from astropy.io import fits

    rng = np.random.default_rng(1)
    img = rng.normal(1000.0, 8.0, shape)
    for i, (x, y) in enumerate((x, y) for x in range(25, 190, 30) for y in range(25, 190, 30)):
        img += _star(shape, x + 0.3, y - 0.2, 1.0e4 * (1 + i % 7), 1.4)
    h = fits.Header()
    for k, v in (header_cards or {}).items():
        h[k] = v
    fits.writeto(path, img.astype(np.float32), h, overwrite=True)


def _count_measure_frame(monkeypatch):
    import frame_metrics

    calls = {"n": 0}
    real = frame_metrics.measure_frame

    def counting(img):
        calls["n"] += 1
        return real(img)

    monkeypatch.setattr(frame_metrics, "measure_frame", counting)
    return calls


def _capture_badpix_fwhm(monkeypatch):
    import badpix_repair

    seen = {}
    real = badpix_repair.repair_missing_pixels

    def capture(data, *, fwhm_px, **kw):
        seen["fwhm"] = fwhm_px
        return real(data, fwhm_px=fwhm_px, **kw)

    monkeypatch.setattr(badpix_repair, "repair_missing_pixels", capture)
    return seen


def _enrich(fp, frame_qc=None):
    from pipeline_calibrate import _qc_enrich_one_frame

    return _qc_enrich_one_frame(str(fp), sky_order=0, prefilter_status=None, target_ra=None, target_dec=None,
                                inject_pointing_only_if_missing=True, detection_sigma=5.0, frame_qc=frame_qc)


def test_preprocess_reads_the_calibration_fwhm_instead_of_measuring_again(tmp_path, monkeypatch):
    """G01-16: a frame measured at calibration (VY_FWHM stamped) is not measured again; the badpix repair and
    VY_FWHM use that one value (the old preprocess ran measure_frame a second time)."""
    from astropy.io import fits

    fp = tmp_path / "l.fits"
    _light(fp, {"VY_FWHM": 3.2101, "VY_ELONG": 1.05, "VY_NSTAR": 4, "VY_QCBG": 1000.5})
    calls, seen = _count_measure_frame(monkeypatch), _capture_badpix_fwhm(monkeypatch)
    row = _enrich(fp)
    assert calls["n"] == 0
    assert row["fwhm_px"] == 3.2101 and row["fwhm_source"] == "calibration header"
    assert seen["fwhm"] == 3.2101
    assert fits.getheader(fp)["VY_FWHM"] == 3.2101


def test_preprocess_takes_the_manifest_frame_qc_of_pre_calibrated_lights(tmp_path, monkeypatch):
    fp = tmp_path / "l.fits"
    _light(fp)
    calls = _count_measure_frame(monkeypatch)
    row = _enrich(fp, {"fwhm": 2.75, "elongation": 1.1, "star_count": 9, "sky_level": 999.0})
    assert calls["n"] == 0
    assert row["fwhm_px"] == 2.75 and row["n_stars_detected"] == 9 and row["fwhm_source"] == "frame QC (manifest)"


def test_a_never_measured_frame_is_measured_once_in_preprocess(tmp_path, monkeypatch):
    from astropy.io import fits

    fp = tmp_path / "l.fits"
    _light(fp)
    calls = _count_measure_frame(monkeypatch)
    row = _enrich(fp)
    assert calls["n"] == 1 and row["fwhm_source"] == "preprocess (first measurement)"
    h = fits.getheader(fp)
    assert h["VY_FWHM"] == round(row["fwhm_px"], 4) and "VY_QCBG" in h


def test_calibration_stamps_the_one_measurement_and_layer_a_is_gone():
    from astropy.io import fits

    import pipeline_calibrate as pc

    for gone in ("_post_calibration_qc_eval", "_mean_hfr_bright_stars_dao", "_qc_fwhm_elongation"):
        assert not hasattr(pc, gone)
    h = fits.Header()
    pc.stamp_frame_qc_headers(h, {"fwhm_mean": 3.123456, "elongation_mean": 1.0712, "star_count": 37,
                                  "sky_background": 812.25})
    assert (h["VY_FWHM"], h["VY_ELONG"], h["VY_NSTAR"], h["VY_QCBG"]) == (3.1235, 1.0712, 37, 812.25)
    assert "VY_QCHFR" not in h and "VY_QCRMS" not in h
