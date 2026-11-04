"""VYVAR step 7 (T17), tag err: G08-14 filter G, k'' per band (G09-05 / G08-47 / G08-48), LC headline
precision (G08-17), ensemble error (G08-08, Collier Cameron 2006 eq. 4) and the night floor (G08-03)."""
from __future__ import annotations

import numpy as np
import pytest


# ------------------------------------------------------------------------------- G08-14 bare "G" / "Z"
@pytest.mark.parametrize("tok", ["G", "Z"])
def test_g08_14_bare_g_z_have_no_standard_band(tok):
    from band_classify import PhotometricBand, classify_photometric_band, standard_band_of_filter
    from gaia_bands import filter_band, filter_band_meta
    from k2_extinction import resolve_k2_bprp_value

    assert standard_band_of_filter(tok) is None
    assert classify_photometric_band(f"{tok}_60_2") is PhotometricBand.UNKNOWN
    fb = filter_band(tok)
    assert fb.zp_band == "" and fb.aavso_code == "" and not fb.transformable
    meta = filter_band_meta(f"{tok}_60_2")
    assert meta["warning"].startswith(f"ambiguous filter {tok}")
    v, src = resolve_k2_bprp_value(None, f"{tok}_60_2")
    assert src.value == "none" and not np.isfinite(v)


@pytest.mark.parametrize("tok", ["g", "gp", "g'", "sg", "SG", "sloan_g", "sdss_g", "Sloan g"])
def test_g08_14_explicit_sloan_g_names_stay_sloan_g(tok):
    from band_classify import PhotometricBand, classify_photometric_band, standard_band_of_filter
    from gaia_bands import filter_band

    assert standard_band_of_filter(tok) == "g"
    assert classify_photometric_band("", fits_filter=tok) is PhotometricBand.STANDARD_FILTER
    assert filter_band(tok).aavso_code == "SG"


def test_g08_14_r_b_unchanged_and_tables_consistent():
    from band_classify import standard_band_of_filter
    from gaia_bands import band_for_filter, filter_band
    from k2_extinction import computed_k2_bprp_for_token

    assert filter_band("R").aavso_code == "R" and filter_band("B").aavso_code == "B"
    for tok in ("u", "g", "r", "i", "z", "U", "B", "V", "R", "I", "G", "Z", "Rc", "Ic", "SG", "SR", "TG", "L", "C"):
        band = standard_band_of_filter(tok)
        # one table: gaia_bands and k2 see the same band (or none) for every token
        if band is None:
            assert band_for_filter(tok) == "G" and computed_k2_bprp_for_token(tok) is None
        elif band in ("g", "r", "i", "V", "B", "Rc"):
            assert band_for_filter(tok) != "G"


def test_g08_14_differential_lc_without_catalogue_magnitudes():
    from gaia_bands import filter_band, zp_mag_from_gaia

    assert not np.isfinite(zp_mag_from_gaia(12.0, 0.8, filter_band("G")))


# ------------------------------------------------------------- G09-05 / G08-47 / G08-48 k'' per band
def test_k2_r_band_uses_r_minus_i_slope_jordi_table5():
    from k2_extinction import SLOPE_RI_PER_BPRP, computed_k2_bprp_for_token

    # Jordi 2010 Table 5: BP-RP = 0.4649 + 2.3969 x - 0.2976 x^2 + 0.0207 x^3, x = r-i, at r-i = 0.15
    assert SLOPE_RI_PER_BPRP == pytest.approx(0.433, rel=0.01)
    assert computed_k2_bprp_for_token("r") == pytest.approx(-0.004 * 0.433, rel=0.01)
    # K2-RELIABLE-2026-11: the g slope is the Jordi 2010 Table 5 g-r row (0.757; the uncited 1.054 is gone)
    assert computed_k2_bprp_for_token("g") == pytest.approx(-0.016 * 0.757, rel=1e-3)


@pytest.mark.parametrize("og,src,why", [
    ("V_60_1", "none", "negligible per AAVSO PEP Guide 3.5.4"),
    ("R_60_1", "none", "negligible per AAVSO PEP Guide 3.5.4"),
    ("I_60_1", "none", "negligible per AAVSO PEP Guide 3.5.4"),
    ("i_60_1", "none", "no citable colour slope"),
    ("z_60_1", "none", "no citable colour slope"),
    ("NoFilter_60_2", "none", "uncorrected"),
    ("L_60_1", "none", "uncorrected"),
    ("TG_60_1", "none", "uncorrected"),
    ("G_60_1", "none", "ambiguous filter G"),
    ("g_60_1", "literature_default", "Smith et al. 2002"),
    ("B_60_1", "literature_default", "Henden"),
])
def test_k2_source_and_reason_per_band(og, src, why):
    from k2_extinction import resolve_k2

    v, s, reason = resolve_k2(None, og)
    assert s.value == src and why in reason
    assert np.isfinite(v) == (src != "none")


def test_k2_none_band_writes_no_k2_delta_zeros():
    from k2_extinction import K2Source, apply_k2_per_frame, resolve_k2

    v, s, _ = resolve_k2(None, "V_60_1")
    out, delta, rows = apply_k2_per_frame(np.full(4, 12.0), np.linspace(1, 2, 4), object_bp_rp=1.5,
                                          bp_rp_comp_med=0.7, k2_value=v, k2_source=s)
    assert s is K2Source.NONE and np.all(np.isnan(delta)) and set(rows) == {"none"}


def test_k2_calibrated_and_differential_get_the_same_term():
    from k2_extinction import K2Source, apply_k2_per_frame

    am = np.linspace(1.0, 2.3, 50)
    dm = np.zeros(50)
    mc = dm + 14.2
    kw = dict(object_bp_rp=2.5, bp_rp_comp_med=0.8, k2_value=-0.0169, k2_source=K2Source.LITERATURE_DEFAULT)
    d2, _, _ = apply_k2_per_frame(dm, am, **kw)
    m2, _, _ = apply_k2_per_frame(mc, am, **kw)
    assert np.ptp(m2 - d2) < 1e-12


def test_k2_colour_reference_is_the_broeg_weighted_colour():
    from k2_extinction import colourless_weight_fraction, ensemble_bp_rp

    col = {"a": 0.5, "b": 1.0, "c": 2.0}
    w = {"a": 0.9, "b": 0.05, "c": 0.05}
    assert ensemble_bp_rp(col, w) == pytest.approx(0.9 * 0.5 + 0.05 * 1.0 + 0.05 * 2.0)
    assert colourless_weight_fraction(col, w) == 0.0
    col_missing = {"a": float("nan"), "b": 1.0, "c": 2.0}
    assert colourless_weight_fraction(col_missing, w) == pytest.approx(0.9)


def test_k2_psf_minus_aperture_has_no_airmass_slope():
    """PSF LC subtracts the aperture LC's k2_delta_mag; with an injected k'' X dC both are flat."""
    from k2_extinction import K2Source, apply_k2_per_frame

    am = np.linspace(1.0, 2.3, 80)
    k2, dc = -0.0169, 2.0
    inj = k2 * dc * am
    ap, k2_delta, _ = apply_k2_per_frame(inj.copy(), am, object_bp_rp=2.8, bp_rp_comp_med=0.8, k2_value=k2,
                                         k2_source=K2Source.LITERATURE_DEFAULT)
    psf = inj - k2_delta
    assert abs(np.polyfit(am, psf - ap, 1)[0]) < 1e-12 and abs(np.polyfit(am, ap, 1)[0]) < 1e-12


# ---------------------------------------------------------------------- G08-17 headline precision
def _p2p_ratio(kind: str, seed: int, n: int = 600, sig: float = 0.01) -> float:
    from photometry_lightcurve import compute_lc_precision_p2p

    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n)
    m = sig * rng.standard_normal(n)
    if kind == "eclipse":
        m = m + 0.5 * np.exp(-0.5 * ((t - 0.5) / 0.05) ** 2)  # one 0.5 mag eclipse, ~60 points wide
    elif kind == "pulsator":
        m = m + 0.2 * np.sin(2 * np.pi * t * 4.0)
    elif kind == "outliers":
        bad = rng.random(n) < 0.01
        m = np.where(bad, m + 0.3 * rng.standard_normal(n), m)
    return compute_lc_precision_p2p(m, ["normal"] * n) / sig


@pytest.mark.parametrize("kind,lo,hi", [("gauss", 0.95, 1.05), ("eclipse", 0.9, 1.1), ("pulsator", 0.9, 1.1),
                                        ("outliers", 0.9, 1.1)])
def test_g08_17_p2p_precision_recovers_sigma(kind, lo, hi):
    r = float(np.median([_p2p_ratio(kind, s) for s in range(20)]))
    assert lo <= r <= hi, r


def test_g08_17_p2p_uses_normal_epochs_only_and_needs_the_flags():
    from photometry_lightcurve import compute_lc_precision_p2p

    m = np.array([0.0, 0.01, 5.0, -0.01, 0.0, 0.01, 0.0])
    fl = ["normal", "normal", "outlier", "normal", "normal", "normal", "normal"]
    assert compute_lc_precision_p2p(m, fl) < 0.02
    with pytest.raises(ValueError):
        compute_lc_precision_p2p(m, fl[:3])


def test_g08_17_target_page_headline_is_p2p_not_ooe():
    from target_page import star_fields

    sd = star_fields({"catalog_id": "1", "lc_rms": 0.2, "lc_rms_ooe": 0.004, "lc_precision_p2p": 0.0105}, {})
    assert sd["lc_precision_p2p"] == 0.0105 and sd["lc_rms_full"] == 0.2
    assert "lc_rms" not in sd


# ------------------------------------------------- G08-08 ensemble error (Collier Cameron 2006 eq. 4)
def _ens_z(sig, *, floor=0.0, var_amp=0.0, miss=0.0, seed=0, nf=400):
    """std of z = (Z_f - truth) / err_f on good (noise x 0.8-1.2) and bad (x 1.5-3) frames."""
    from photometry_lightcurve import ensemble_normalize

    rng = np.random.default_rng(seed)
    sig = np.asarray(sig, float)
    n = sig.size
    g = np.where(rng.random(nf) < 0.2, rng.uniform(1.5, 3.0, nf), rng.uniform(0.8, 1.2, nf))
    common = 0.03 * rng.standard_normal(nf)
    noise = np.sqrt((sig[:, None] * g[None, :]) ** 2 + floor ** 2) * rng.standard_normal((n, nf))
    if var_amp:
        noise[0] += var_amp * np.sin(np.linspace(0, 6 * np.pi, nf))
    m = 12 + np.arange(n)[:, None] * 0.2 + common[None, :] + noise
    if miss:
        m[rng.random((n, nf)) < miss] = np.nan
    comps = {f"c{j}": m[j] for j in range(n)}
    model = {f"c{j}": sig[j] * g for j in range(n)}  # the per-frame noise model the pipeline knows
    _, dm, es = ensemble_normalize(13 + common, comps, {}, {k: {} for k in comps}, n_comp_min=1,
                                   comp_noise_mag=model)
    z = (dm - np.nanmean(dm)) / es
    ok = np.isfinite(z)
    return np.std(z[ok & (g < 1.3)]), np.std(z[ok & (g >= 1.5)])


@pytest.mark.parametrize("name,kw", [
    ("3 x 10 mmag", dict(sig=[0.01] * 3)),
    ("3, 10, 30 mmag", dict(sig=[0.003, 0.01, 0.03])),
    ("5 x 10 mmag", dict(sig=[0.01] * 5)),
    ("5 mixed, one comp variable 30 mmag", dict(sig=[0.003, 0.005, 0.008, 0.015, 0.03], var_amp=0.03)),
    ("8 x 10, 20 % missing", dict(sig=[0.01] * 8, miss=0.2)),
    ("3 x 3 mmag + 7 mmag floor", dict(sig=[0.003] * 3, floor=0.007)),
])
def test_g08_08_ensemble_error_calibrated_on_good_and_bad_frames(name, kw):
    good, bad = np.median([_ens_z(seed=s, **kw) for s in range(8)], axis=0)
    assert 0.85 <= good <= 1.15, (name, good)
    assert 0.85 <= bad <= 1.15, (name, bad)


def test_g08_08_two_comps_raise():
    from photometry_lightcurve import ensemble_error_per_frame, ensemble_normalize

    n = 30
    comps = {"a": np.full(n, 11.0), "b": np.full(n, 11.5)}
    with pytest.raises(ValueError):
        ensemble_normalize(np.full(n, 12.0), comps, {}, {}, n_comp_min=1,
                           comp_noise_mag={c: np.full(n, 0.01) for c in comps})
    with pytest.raises(ValueError):
        ensemble_error_per_frame(comps, {c: np.full(n, 0.01) for c in comps}, {"a": 0.5, "b": 0.5},
                                 {"a": 11.0, "b": 11.5})


def test_g08_08_excess_recovered_and_missing_model_is_nan_not_zero():
    from photometry_lightcurve import ensemble_error_per_frame

    rng = np.random.default_rng(3)
    nf = 600
    comps = {c: 12.0 + s * rng.standard_normal(nf) for c, s in (("a", 0.01), ("b", 0.01), ("c", 0.01), ("d", 0.01))}
    model = {c: np.full(nf, 0.006) for c in comps}  # true noise 10 mmag, model 6 -> excess 8 mmag
    model["a"][5] = np.nan
    info: dict = {}
    w = {c: 0.25 for c in comps}
    err = ensemble_error_per_frame(comps, model, w, {c: 12.0 for c in comps}, info_out=info)
    assert np.isnan(err[5]) and info["n_frames_unknown"] == 1
    assert np.nanmedian(err) == pytest.approx(0.005, rel=0.1)  # 10 mmag / sqrt(4)
    for c in comps:
        assert info["x_mag"][c] == pytest.approx(0.008, abs=0.0015)


def test_g08_08_target_error_uses_the_comp_noise_model():
    import inspect

    import aperture_photometry_target as apt
    import photometry_lightcurve as pl

    src = inspect.getsource(apt._tgt_time_and_errors)
    assert "ensemble_error_per_frame(" in src and "_scint_mag_arr" in src
    assert "inv_var" not in inspect.getsource(pl.ensemble_normalize)


# ------------------------------------------------------------------ G08-03 night floor (pooled pull)
def _floor_night(seed, n_comp, scint, floor, nf=300):
    """Bright comps (photon noise only from the flux), scintillation ~ X^1.5 over X 1.0-2.08 (x3 in sigma)."""
    from sigma_floor_core import night_sigma_sys_from_comps

    rng = np.random.default_rng(seed)
    am = np.linspace(1.0, 2.08, nf)
    flux, sky, nsky = {}, {}, {}
    for i, mm in enumerate(rng.uniform(10, 12, n_comp)):
        f0 = 10 ** (-0.4 * (mm - 25))
        dm = rng.normal(0, scint, nf) * am ** 1.5 + rng.normal(0, floor, nf)
        flux[str(i)] = (f0 + rng.normal(0, np.sqrt(f0), nf)) * 10 ** (-0.4 * dm)
        sky[str(i)] = np.zeros(nf)
        nsky[str(i)] = np.full(nf, 1e9)
    return night_sigma_sys_from_comps(flux, sky, np.full(nf, 3.0), k=1e-6, gain=1.0, read_noise=0.0,
                                      n_sky_by_cid=nsky, scint_mag_by_frame=scint * am ** 1.5)


def test_g08_03_floor_recovered_under_airmass_varying_scintillation():
    rs = [_floor_night(s, 10, 0.005, 0.003) for s in range(10)]
    v = np.median([r["sigma_sys_mag"] for r in rs])
    assert abs(v - 0.003) < 0.0005, v  # 6f08c7b: 1.87 mmag (moment shortcut)
    assert np.median([r["pull_comps"] for r in rs]) == pytest.approx(1.0, abs=0.02)


def test_g08_03_zero_floor_small_and_scint_warning():
    rs = [_floor_night(s, 10, 0.003, 0.0) for s in range(10)]
    assert np.median([r["sigma_sys_mag"] for r in rs]) < 0.001
    assert all(r["scint_share"] > 0.5 and "not separable" in r["warning"] for r in rs)


def test_g08_03_few_comps_no_reference_bias():
    """The reference carries the others' floor: (1 + sum u^2) removes the +sqrt(1 + 1/n) bias (5 comps)."""
    rs = [_floor_night(s, 5, 0.0, 0.004) for s in range(10)]
    v = np.median([r["sigma_sys_mag"] for r in rs])
    assert abs(v - 0.004) < 0.0003, v
    assert all(r["warning"] == "" for r in rs)
