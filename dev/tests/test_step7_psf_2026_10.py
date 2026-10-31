"""Step 7 (T17) psf items, 2026-10-31: C4, C1, C3, G09-07, G09-09, G09-02, G10-08, G10-30.

Synthetic frames with known truth: stars are rendered with the same photutils ``ImagePSF`` that is fitted
("exact model") or with a Moffat profile, so a bias is the method's and not the model's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from photutils.psf import ImagePSF

import epsf_frame as ef

_OS = 2


def _moffat_epsf(fwhm: float, cutout: int, beta: float = 2.7) -> ImagePSF:
    """Oversampled Moffat ePSF on a ``cutout`` px grid, normalised like EPSFBuilder (sum = os^2)."""
    n = cutout * _OS + 1
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    r2 = ((xx - c) ** 2 + (yy - c) ** 2) / _OS**2
    a = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    arr = (1.0 + r2 / a**2) ** (-beta)
    arr *= _OS**2 / arr.sum()
    return ImagePSF(arr, oversampling=_OS)


def _render(model: ImagePSF, shape: tuple[int, int], xs, ys, fluxes, sky: float = 1000.0) -> np.ndarray:
    img = np.full(shape, sky, dtype=np.float64)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    m = model.copy()
    for x, y, f in zip(xs, ys, fluxes, strict=True):
        m.x_0, m.y_0, m.flux = x, y, f
        img += m(xx, yy)
    return img


def _noisy(img: np.ndarray, sigma: float = 1.0, seed: int = 3) -> np.ndarray:
    """Low Gaussian noise: the production source mask needs a finite frame MAD; flux noise stays << 1 mmag."""
    return img + np.random.default_rng(seed).normal(0.0, sigma, img.shape)


# --- C4 ---------------------------------------------------------------------------------------------------

def test_c4_nan_local_sky_is_not_a_converged_fit() -> None:
    """C4: annulus and its fallback box without sky pixels -> photutils flags 2048 and fits with sky 0; the fit
    must not count as converged (old _BAD_FIT_FLAGS 2|4|8|16 let it through), and the frame stats count it."""
    fw = 3.0
    model = _moffat_epsf(fw, 37)
    img = _noisy(_render(model, (200, 200), [100.0, 40.0], [100.0, 40.0], [3e4, 3e4]))
    d = img.copy()
    half_box = int(2.0 * 6.0 * fw) + 2
    fsz = ef.fit_box_px(fw)
    blk = d[100 - half_box : 100 + half_box + 1, 100 - half_box : 100 + half_box + 1]
    keep = blk[half_box - fsz // 2 : half_box + fsz // 2 + 1, half_box - fsz // 2 : half_box + fsz // 2 + 1].copy()
    blk[:] = np.nan
    blk[half_box - fsz // 2 : half_box + fsz // 2 + 1, half_box - fsz // 2 : half_box + fsz // 2 + 1] = keep
    st = pd.DataFrame({"catalog_id": ["nan_sky", "ok"], "x": [100.0, 40.0], "y": [100.0, 40.0]})
    df = ef.fit_frame_psf(d, model, st, None, fwhm_px=fw, gain=1.0, read_noise_e=5.0)
    r = df.set_index("catalog_id")
    assert int(r.at["nan_sky", "psf_flags"]) & 2048
    assert not bool(r.at["nan_sky", "psf_converged"])
    assert bool(r.at["ok", "psf_converged"])
    stats = ef.frame_fit_stats(df, "f")
    assert stats["n_bad_fit_flags"] == 1 and stats["bad_fit_flag_counts"].get("non_finite_local_bkg") == 1


# --- C1 ---------------------------------------------------------------------------------------------------

@pytest.fixture
def poisson_source_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Noise-free frame, production source mask computed on a Poisson-noisy copy (gain 1): a noise-free frame
    has a frame MAD of ~0 and every wing pixel would count as a source (lit_psf common_sim.py)."""
    import sky_estimation as se

    def _sky_image(data, fwhm_px, *, covered=None):
        d = np.asarray(data, dtype=np.float64)
        noisy = d + np.random.default_rng(11).normal(0.0, np.sqrt(np.clip(np.nan_to_num(d, nan=1000.0), 1.0, None)))
        out = d.copy()
        out[se.sky_source_mask(noisy, fwhm_px, valid=covered)] = np.nan
        return out

    monkeypatch.setattr(ef, "sky_image", _sky_image)


def test_c1_allstar_sky_removes_the_bright_neighbour_wing(poisson_source_mask: None) -> None:
    """C1 (DAOPHOT ALLSTAR sky): a faint star with a 5.75 mag brighter neighbour 3.6 FWHM away, exact model.
    The pass-1 annulus takes the neighbour's wing outside the source mask; the pass-2 sky measured with all
    fitted stars subtracted brings the flux within 1 mmag of the truth."""
    fw, cut = 3.34, 41
    model = _moffat_epsf(fw, cut)
    ft, fn, sep = 2e4, 4e6, 12.0
    img = _render(model, (200, 230), [100.0, 100.0 + sep], [100.0, 100.0], [ft, fn])
    st = pd.DataFrame({"catalog_id": ["t"], "x": [100.0], "y": [100.0]})
    nb = pd.DataFrame({"x": [100.0 + sep], "y": [100.0]})
    df = ef.fit_frame_psf(img, model, st, nb, fwhm_px=fw, gain=1.0, read_noise_e=5.0)
    bias_mmag = -2500.0 * np.log10(float(df["psf_flux"].iloc[0]) / ft)
    assert abs(bias_mmag) < 1.0, bias_mmag
    # the pass-1 sky was lifted by the wing; the sidecar keeps both
    assert float(df["psf_sky_pass1"].iloc[0]) > float(df["psf_sky"].iloc[0])
    assert abs(float(df["psf_sky"].iloc[0]) - 1000.0) < 0.05
    assert ef.frame_fit_stats(df, "f")["sky_allstar_median_abs_diff_adu"] > 0


# --- C3 ---------------------------------------------------------------------------------------------------

def _moffat_frame(fw: float, xs, ys, fl, beta: float = 2.5, sky: float = 1000.0, rmax: int = 200) -> np.ndarray:
    """Pixel-centre sampled Moffat stars normalised to their TOTAL flux (infinite integral)."""
    side = int(max(np.max(xs), np.max(ys)) + 130)
    img = np.full((side, side), sky)
    a = fw / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    norm = (beta - 1.0) / (np.pi * a * a)
    for x, y, f in zip(xs, ys, fl, strict=True):
        xi, yi = int(round(x)), int(round(y))
        y0, y1, x0, x1 = max(0, yi - rmax), min(side, yi + rmax + 1), max(0, xi - rmax), min(side, xi + rmax + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        img[y0:y1, x0:x1] += f * norm * (1.0 + ((xx - x) ** 2 + (yy - y) ** 2) / (a * a)) ** (-beta)
    return img


def test_c3_frame_cutout_follows_the_frame_seeing(poisson_source_mask: None) -> None:
    """C3 (EPSF-CUTOUT-FRAME-2026-10): with the model size odd(12 x frame FWHM) a Moffat beta 2.5 frame at FWHM 3.4
    and at 10 gives a faint-bright differential < 0.5 mmag and the same light in the model (within 0.2 %); the fixed
    41 px MASTERSTAR cutout lost 11 % of the light and 5.7 mmag faint-bright at FWHM 10 (lit_psf c3_cutout.py)."""
    from psf_photometry import epsf_cutout_px

    rng = np.random.default_rng(7)
    n, sp = 10, 130
    gx, gy = np.meshgrid(np.arange(1, n + 1) * sp, np.arange(1, n + 1) * sp)
    gx = gx.ravel() + rng.uniform(-0.5, 0.5, gx.size)
    gy = gy.ravel() + rng.uniform(-0.5, 0.5, gy.size)
    n_build = 70
    roles = np.array(["build"] * n_build + ["faint"] * 15 + ["bright"] * 15)
    flux = np.where(roles == "build", 2e5, np.where(roles == "faint", 3e3, 1e6))
    stars = pd.DataFrame({"catalog_id": [str(i) for i in range(gx.size)], "x": gx, "y": gy})
    light = {}
    for fw in (3.4, 10.0):
        img = _moffat_frame(fw, gx, gy, flux)
        cut = epsf_cutout_px(fw)
        assert cut == (41 if fw == 3.4 else 121)
        built = ef._build_epsf(img, stars.iloc[:n_build][["x", "y"]], cut, fw)
        model = ImagePSF(built["arr"].astype(float), oversampling=ef.OVERSAMPLING)
        r = ef.fit_frame_psf(img, model, stars, None, fwhm_px=fw, gain=1.0, read_noise_e=5.0)
        rat = r["psf_flux"].to_numpy(float) / flux
        light[fw] = float(np.median(rat[roles == "build"]))
        diff = -2500.0 * np.log10(np.median(rat[roles == "faint"]) / np.median(rat[roles == "bright"]))
        assert abs(diff) < 0.5, (fw, diff)
    assert abs(light[3.4] / light[10.0] - 1.0) < 0.002, light


def test_c3_spatial_delta_is_applied_inside_the_masterstar_cutout() -> None:
    """C3: the night perturbation (MASTERSTAR cutout) is centred on a larger frame grid with zeros outside and cut
    centrally on a smaller one - no resampling."""
    rng = np.random.default_rng(1)
    delta = rng.normal(size=(4, 83, 83))
    big = ef._delta_on_grid(delta, (123, 123))
    assert big.shape == (4, 123, 123)
    assert np.array_equal(big[:, 20:103, 20:103], delta) and big[:, :20].sum() == 0 and big[:, 103:].sum() == 0
    small = ef._delta_on_grid(delta, (61, 61))
    assert np.array_equal(small, delta[:, 11:72, 11:72])


def test_c3_fit_task_builds_the_model_at_the_frame_fwhm(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C3 wiring: the per-frame task sizes the model from the frame's own fwhm_px_for_aperture (proc sidecar),
    not from the MASTERSTAR cutout, and fails loud when the frame FWHM is unknown."""
    from astropy.io import fits

    import epsf_psf_merge as em
    import psf_photometry as pp

    fw = 4.0
    rng = np.random.default_rng(2)
    n, sp = 8, 70
    gx, gy = np.meshgrid(np.arange(1, n + 1) * sp, np.arange(1, n + 1) * sp)
    gx = gx.ravel() + rng.uniform(-0.5, 0.5, gx.size)
    gy = gy.ravel() + rng.uniform(-0.5, 0.5, gy.size)
    img = np.random.default_rng(3).poisson(_moffat_frame(fw, gx, gy, np.full(gx.size, 1e5))).astype(np.float32)
    hdr = fits.Header()
    for k, v in enumerate((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):
        hdr[f"VY_AM{k}"] = v
    fp = tmp_path / "f_0001.fits"
    fits.PrimaryHDU(img, header=hdr).writeto(fp)
    ids = [str(i) for i in range(gx.size)]
    pd.DataFrame({"catalog_id": ids, "fwhm_px_for_aperture": fw, "saturate_limit_adu_85pct": 6e4}).to_csv(
        tmp_path / "proc_f_0001.csv", index=False)
    stars = pd.DataFrame({"catalog_id": ids, "x": gx, "y": gy})
    monkeypatch.setattr(pp, "_psf_resolve_gain_read_noise", lambda *a, **k: (1.0, 5.0))
    em._fit_worker_init({"build_stars": stars, "delta": None, "grid": (1, 1), "platesolve_dir": str(tmp_path),
                         "app_config": None, "fit_stars": stars.iloc[:4], "neighbours": None})
    out = em._fit_frame_task(fp)
    assert out["exception_class"] is None, out["exception_message"]
    assert out["mrec"]["cutout_size"] == pp.epsf_cutout_px(fw) == 49
    assert out["mrec"]["frame_fwhm_px"] == pytest.approx(fw)
    pd.DataFrame({"catalog_id": ids, "fwhm_px_for_aperture": np.nan, "saturate_limit_adu_85pct": 6e4}).to_csv(
        tmp_path / "proc_f_0001.csv", index=False)
    out = em._fit_frame_task(fp)
    assert out["exception_class"] == "ValueError" and "fwhm_px_for_aperture" in out["exception_message"]


# --- G09-07 -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sep", [16.0, 20.0, 24.0])
def test_g0907_flux_does_not_depend_on_a_group_radius_beyond_the_reach(
    poisson_source_mask: None, monkeypatch: pytest.MonkeyPatch, sep: float
) -> None:
    """G09-07: a faint star with a 5.75 mag brighter neighbour 16-24 px away (FWHM 3.34, 41 px model, fit box 7:
    reach 25 px). The production group radius gives the same flux as one group of everything (< 1 mmag); the old
    2 fit boxes (14 px) fitted the faint star alone next to the unmodelled wing (-70 mmag at 16 px)."""
    fw, cut = 3.34, 41
    model = _moffat_epsf(fw, cut)
    assert ef.model_cutout_px(model) == cut and ef.model_reach_px(cut, fw) == 25.0
    img = _render(model, (200, 240), [100.0, 100.0 + sep], [100.0, 100.0], [2e4, 4e6])
    st = pd.DataFrame({"catalog_id": ["t"], "x": [100.0], "y": [100.0]})
    nb = pd.DataFrame({"x": [100.0 + sep], "y": [100.0]})
    prod = ef.fit_frame_psf(img, model, st, nb, fwhm_px=fw, gain=1.0, read_noise_e=5.0)
    orig = ef.SourceGrouper
    monkeypatch.setattr(ef, "SourceGrouper", lambda min_separation: orig(min_separation=1000.0))
    one = ef.fit_frame_psf(img, model, st, nb, fwhm_px=fw, gain=1.0, read_noise_e=5.0)
    d_mmag = 2500.0 * np.log10(float(prod["psf_flux"].iloc[0]) / float(one["psf_flux"].iloc[0]))
    assert abs(d_mmag) < 1.0, d_mmag
    assert int(prod["psf_group_size"].iloc[0]) == 2
    stats = ef.frame_fit_stats(prod, "f")
    assert stats["group_size_max"] == 2 and stats["group_size_p99"] == 2.0


def test_g0907_frame_fit_sources_reach_is_the_shared_rule(tmp_path) -> None:
    """G09-07: the neighbour reach of frame_fit_sources is model_reach_px (one helper with the group radius)."""
    from tests.test_epsf_csv_prepare import _synthetic_field

    _synthetic_field(tmp_path)
    _, _, meta = ef.frame_fit_sources(tmp_path, {"2000000000000000000"}, cutout_size=31, fwhm_px=2.5, merge_fwhm_px=2.5)
    assert meta["neighbour_reach_px"] == ef.model_reach_px(31, 2.5) == 15.5 + 2.5 + 1.0


# --- G09-09 -----------------------------------------------------------------------------------------------

def _locus_night(seed: int):
    """60 faint stars (log F 3.0-4.0) + 20 bright stars spread over log F 4.0-6.0, 100 frames; every star has a
    persistent chi2 offset (sd 0.05 dex) on top of per-frame scatter 0.08 dex."""
    rng = np.random.default_rng(seed)
    lf_star = np.r_[rng.uniform(3.0, 4.0, 60), np.linspace(4.0, 6.0, 20)]
    off = rng.normal(0.0, 0.05, lf_star.size)
    n_fr = 100
    lf = np.repeat(lf_star, n_fr) + rng.normal(0.0, 0.02, lf_star.size * n_fr)
    base = 0.3 * np.clip(lf - 4.0, 0.0, None)  # curved: chi2 rises at the bright end
    lc = base + np.repeat(off, n_fr) + rng.normal(0.0, 0.08, lf.size)
    ids = np.repeat(np.arange(lf_star.size), n_fr)
    return lf, lc, ids


def test_g0909_flux_groups_hold_at_least_15_distinct_stars() -> None:
    """G09-09 (CHI2-LOCUS-STARS-2026-10): a group needs MIN_N_BIN samples AND MIN_STARS_BIN distinct stars; the
    sparse bright stars (100 samples each) no longer form groups of 1-3 stars. n_stars per group is in the record."""
    from psf_chi2_locus import MIN_STARS_BIN, fit_chi2_locus

    lf, lc, ids = _locus_night(1)
    night = fit_chi2_locus(10**lf, 10**lc, star_ids=ids)
    assert night is not None and MIN_STARS_BIN == 15
    assert min(night.n_stars_bin) >= 15 and len(night.n_stars_bin) == len(night.centres)
    assert night.to_dict()["n_stars_bin"] == list(night.n_stars_bin)
    # a night with fewer stars than MIN_STARS_BIN in all: one group, said so in the record
    few = ids < 5
    one = fit_chi2_locus(10 ** lf[few], 10 ** lc[few], star_ids=ids[few])
    assert one is not None and len(one.centres) == 1 and "MIN_STARS_BIN" in one.note


def test_g0909_gate_is_stable_and_flags_per_frame_outliers() -> None:
    """G09-09: with >= 15 stars per group the 5-sigma gate at the sparse bright end varies little from night to night
    (relative 5-95 % spread 0.19 < 0.25; with the old sample-count groups of 1-3 bright stars: 0.30), a per-frame misfit
    10 sigma (per-frame scatter) above the curve is flagged in >= 95 % of cases and clean samples are not (< 0.5 %).
    The scale includes the stars' persistent offsets (lit_psf G09-09: the gate is per-frame misfit over stars of
    similar flux)."""
    from psf_chi2_locus import DEFAULT_NSIGMA, fit_chi2_locus, locus_residual_sigma

    hits, n, false, gate = 0, 0, [], []
    for seed in range(20):
        lf, lc, ids = _locus_night(100 + seed)
        night = fit_chi2_locus(10**lf, 10**lc, star_ids=ids)
        r = locus_residual_sigma(10**lf, 10**lc, night)
        false.append(float(np.mean(r > DEFAULT_NSIGMA)))
        gate.append(DEFAULT_NSIGMA * float(np.interp(5.5, night.centres, night.scatter)))
        rng = np.random.default_rng(seed)
        pick = rng.choice(np.flatnonzero(lf > 4.5), 10, replace=False)
        ro = locus_residual_sigma(10 ** lf[pick], 10 ** (lc[pick] + 10 * 0.08), night)
        hits += int(np.sum(ro > DEFAULT_NSIGMA))
        n += pick.size
    g = np.asarray(gate)
    assert (np.percentile(g, 95) - np.percentile(g, 5)) / np.median(g) < 0.25
    assert hits / n >= 0.95, hits / n
    assert float(np.mean(false)) < 0.005


# --- G09-02 -----------------------------------------------------------------------------------------------

def test_g0902_build_star_meta_records_count_minimum_and_neighbour_light(tmp_path) -> None:
    """G09-02: epsf_build_stars_meta.json records the build-star count, the minimum it was held to, the warning
    floor and the share of build stars with > 1 % Gaia neighbour light in their cutout."""
    import json

    from database import VyvarDatabase
    from psf_photometry import EPSF_BUILD_STARS_META_NAME, build_star_neighbour_light, select_epsf_build_stars
    from tests.test_epsf_csv_prepare import _synthetic_field

    ms, csv, _ = _synthetic_field(tmp_path)
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    try:
        meta = select_epsf_build_stars(ms, csv, db, 1, min_stars=10)
    finally:
        db.close()
    on_disk = json.loads((tmp_path / EPSF_BUILD_STARS_META_NAME).read_text())
    assert on_disk["min_build_stars"] == 10 and on_disk["build_stars_warn_below"] == ef.EPSF_BUILD_STARS_WARN == 100
    assert meta["n_build_stars_nb_light_known"] == meta["n_stars"]
    from psf_photometry import EPSF_BUILD_STARS_NAME

    bs = pd.read_csv(tmp_path / EPSF_BUILD_STARS_NAME, dtype={"catalog_id": str})
    share = float(np.mean(build_star_neighbour_light(bs, ms, 31) > 0.01))
    assert meta["frac_build_stars_nb_light_gt_1pct"] == pytest.approx(share, abs=1e-4)
    # the bright stars with a 4 mag fainter companion 6.5 px away pass the 2.5 mag isolation and carry 2.5 %
    assert 0.0 < share < 0.2
    # a star next to a 4 mag brighter one carries 40x its own light in the box
    probe = pd.DataFrame({"x": [90.0], "y": [90.0], "mag": [12.5]})
    assert build_star_neighbour_light(probe, ms, 31)[0] == pytest.approx(10 ** 1.6, rel=1e-6)


def test_g0902_no_clip_epsf_ignores_minority_neighbour_contamination() -> None:
    """G09-02 (EPSF-NOCLIP note): 300 build stars (noise-free), 30 % with a 2.5 mag fainter neighbour 5 FWHM away. The plain median
    ePSF (sigma_clip=None) has the wing light r > 2 FWHM within 0.05 % of total and the peak within 0.5 % of the ePSF
    built from the clean stars only."""
    fw, cut = 2.5, 31
    rng = np.random.default_rng(5)
    n_side, sp = 18, 40
    gx, gy = np.meshgrid(np.arange(1, n_side + 1) * sp, np.arange(1, n_side + 1) * sp)
    x = gx.ravel()[:300] + rng.uniform(-0.5, 0.5, 300)
    y = gy.ravel()[:300] + rng.uniform(-0.5, 0.5, 300)
    cont = rng.random(300) < 0.3
    ang = rng.uniform(0, 2 * np.pi, 300)
    nx, ny = x[cont] + 5 * fw * np.cos(ang[cont]), y[cont] + 5 * fw * np.sin(ang[cont])
    img = _moffat_frame(fw, np.r_[x, nx], np.r_[y, ny], np.r_[np.full(300, 2e5), np.full(cont.sum(), 2e5 * 10**-1.0)],
                        rmax=60)
    built = {lab: ef._build_epsf(img, pd.DataFrame({"x": x[m], "y": y[m]}), cut, fw)["arr"].astype(float)
             for lab, m in (("all", np.ones(300, bool)), ("clean", ~cont))}
    a = {k: v / v.sum() for k, v in built.items()}
    n = a["all"].shape[0]
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - c, yy - c) / ef.OVERSAMPLING
    wing = {k: float(v[r > 2 * fw].sum()) for k, v in a.items()}
    assert abs(wing["all"] - wing["clean"]) < 5e-4, wing
    assert abs(a["all"].max() / a["clean"].max() - 1.0) < 0.005


# --- G10-08 -----------------------------------------------------------------------------------------------

def _var_field(seed: int = 4):
    """Synthetic night: 300 stars x 80 frames, airmass 1.0 -> 2.2 and FWHM 3.4 -> 9 both monotonic in time,
    colour extinction 0.02 mag / airmass per mag BP-RP, white noise 8 mmag. Star 'slow': sinusoid P = 4 x night,
    semi-amplitude 0.05 mag. Star 'seeing': 10 x the field's FWHM response (5 mmag/px). Star 'red': BP-RP 4.0
    (colour extinction alone)."""
    import variability_field as vf

    rng = np.random.default_rng(seed)
    n_st, n_t = 300, 80
    tn = np.linspace(0.0, 1.0, n_t)
    am = 1.0 + 1.2 * tn**1.5
    fw = 3.4 + 5.6 * tn + rng.normal(0.0, 0.2, n_t)
    ids = [f"s{i:03d}" for i in range(n_st - 3)] + ["slow", "seeing", "red"]
    mag = np.r_[rng.uniform(10.0, 15.0, n_st - 3), 12.0, 12.0, 12.0]
    col = np.r_[rng.uniform(0.3, 2.5, n_st - 3), 1.2, 1.2, 4.0]
    k_col, k_fw = 0.02, 0.0005
    base = (np.outer(col - np.median(col), am - am.mean()) * k_col + k_fw * (fw - fw.mean())[None, :])
    R = base + rng.normal(0.0, 0.008, (n_st, n_t))
    R[ids.index("slow")] += 0.05 * np.sin(2 * np.pi * tn / 4.0 - np.pi / 4)
    R[ids.index("seeing")] += 10 * 0.005 * (fw - fw.mean())
    R = R - np.median(R, axis=1, keepdims=True)
    R = pd.DataFrame(R, index=ids, columns=range(n_t))
    info = pd.DataFrame({"mag": mag, "bp_rp": col}, index=ids)
    info["s"] = 1.4826 * R.abs().median(axis=1)
    info["s_exp"] = 1.4826 * 0.008 * 0.6745
    info["eta"] = [vf._eta(R.loc[c].to_numpy(float)) for c in ids]
    info["gates_ok"] = True
    info["known_variable"] = False
    frame = pd.DataFrame({"fwhm": fw, "airmass": am}, index=range(n_t))
    return vf, R, info, frame


def test_g1008_slow_variable_survives_and_flags_report_own_responses() -> None:
    """G10-08 (SYS-FIELD-RESPONSE-2026-10): a slow sinusoid on a night where FWHM and airmass run monotonically is a
    candidate (the per-star OLS removed it: trend_like), a star with 10x the field's FWHM response is systematic
    (its own frame-to-frame seeing response removes it, SYS-SEEING-DIFF-2026-10; it also is a seeing_response_z
    outlier), and a red star explained by the field's colour extinction is systematic. The field coefficients and
    their bootstrap errors are in meta. Updated for SYS-SEEING-DIFF-2026-10: the seeing star was listed before."""
    from types import SimpleNamespace

    vf, R, info, frame = _var_field()
    use = np.ones(len(info), bool)
    meta: dict = {}
    _, _, out, _, _, zthr = vf._var_indices(E=None, M=None, R=R, cfg=SimpleNamespace(transient_search_enabled=False),
                                            frame=frame, info=info, meta=meta, ref=None, use=use)
    o = out.set_index("catalog_id")
    assert bool(o.at["slow", "signal"]) and bool(o.at["slow", "candidate"]) and not bool(o.at["slow", "systematic"])
    assert bool(o.at["slow", "trend_like"])  # the old rule (own OLS) would have dropped it
    assert bool(o.at["seeing", "signal"]) and o.at["seeing", "seeing_response_z"] > zthr
    assert bool(o.at["seeing", "systematic"]) and not bool(o.at["seeing", "candidate"])
    assert o.at["seeing", "fwhm_coef_diff"] == pytest.approx(0.05, rel=0.25)
    assert bool(o.at["red", "signal"]) and bool(o.at["red", "systematic"]) and not bool(o.at["red", "candidate"])
    sysm = meta["systematics"]
    ts = sysm["airmass_coef_vs_bp_rp_theil_sen"]
    assert ts["slope_mag_per_airmass_per_mag"] == pytest.approx(0.02, abs=3 * ts["slope_err_bootstrap"] + 1e-3)
    assert 0 < ts["slope_err_bootstrap"] < 0.005 and sysm["n_bootstrap"] == vf.FIELD_RESPONSE_BOOTSTRAP
    assert sysm["n_field"] < len(info)  # the signal stars are not in the field set
    # constant stars are not flagged by the seeing z beyond the field's tail
    const = o.index.str.startswith("s")
    assert float(np.mean(np.abs(o.loc[const, "seeing_response_z"]) > zthr)) < 0.02


def test_seeing_diff_slow_variable_with_a_seeing_response_stays_a_candidate() -> None:
    """SYS-SEEING-DIFF-2026-10: a star with BOTH a slow sinusoid (P = 4 x night, 0.05 mag) and 10x the field's
    seeing response is still a candidate: its own seeing coefficient comes from frame-to-frame differences, which the
    slow sinusoid does not enter (a level OLS on [1, FWHM, airmass] would absorb most of it)."""
    from types import SimpleNamespace

    vf, R, info, frame = _var_field()
    fw = frame["fwhm"].to_numpy(float)
    both = R.loc["slow"].to_numpy(float) + 10 * 0.005 * (fw - fw.mean())
    R.loc["both"] = both - np.median(both)
    info.loc["both"] = info.loc["slow"]
    info.at["both", "s"] = 1.4826 * float(R.loc["both"].abs().median())
    info.at["both", "eta"] = vf._eta(R.loc["both"].to_numpy(float))
    _, _, out, _, _, _ = vf._var_indices(E=None, M=None, R=R, cfg=SimpleNamespace(transient_search_enabled=False),
                                         frame=frame, info=info, meta={}, ref=None, use=np.ones(len(info), bool))
    o = out.set_index("catalog_id")
    assert bool(o.at["both", "candidate"]) and not bool(o.at["both", "systematic"])
    assert o.at["both", "fwhm_coef_diff"] == pytest.approx(0.05, rel=0.25)


def test_seeing_coef_from_differences_ignores_a_slow_trend() -> None:
    """SYS-SEEING-DIFF-2026-10: the difference estimator recovers the seeing coefficient under a slow trend that
    is perfectly correlated with the FWHM level, and returns NaN without frame-to-frame FWHM change."""
    import variability_field as vf

    rng = np.random.default_rng(1)
    n = 120
    fw = 3.0 + 4.0 * np.linspace(0, 1, n) + rng.normal(0, 0.3, n)
    y = 0.2 * np.linspace(0, 1, n) - 0.03 * fw + rng.normal(0, 0.003, n)
    assert vf.seeing_coef_from_differences(y, fw) == pytest.approx(-0.03, abs=0.004)
    X = np.c_[np.ones(n), fw]
    assert abs(np.linalg.lstsq(X, y, rcond=None)[0][1] + 0.03) > 0.02  # level OLS is biased by the trend
    assert np.isnan(vf.seeing_coef_from_differences(y, np.full(n, 4.0)))
    y[::2] = np.nan
    assert np.isnan(vf.seeing_coef_from_differences(y, fw))


# --- G10-30 -----------------------------------------------------------------------------------------------

def test_g1030_varcand_lightcurve_marks_points_below_the_limit(tmp_path) -> None:
    """G10-30: a point whose flux is below its 1-sigma error takes the 1-sigma flux (COVERAGE-FAINT) and is marked
    below_limit in varcand_<id>.csv."""
    import variability_field as vf

    flux = pd.Series([1000.0, 50.0, -20.0, 900.0])
    var = pd.Series([100.0**2, 100.0**2, 100.0**2, 100.0**2])
    m, e, bl = vf.limit_magnitudes(flux, var)
    assert list(bl) == [False, True, True, False]
    assert m.iloc[1] == pytest.approx(-2.5 * np.log10(100.0)) and m.iloc[2] == pytest.approx(-2.5 * np.log10(100.0))
    t = range(4)
    frame = pd.DataFrame({"fwhm": [3.0] * 4, "airmass": [1.1] * 4}, index=t)
    R = pd.DataFrame([m.to_numpy() - np.median(m)], index=["x"], columns=t)
    S = pd.DataFrame([e.to_numpy()], index=["x"], columns=t)
    BL = pd.DataFrame([bl.to_numpy()], index=["x"], columns=t)
    vf.write_candidate_lightcurves(tmp_path, ["x"], R=R, S=S, BL=BL, frame=frame, t_star=np.array([[1.0, 2.0, 3.0, 4.0]]),
                                   a=1.0, fl=0.0)
    out = pd.read_csv(tmp_path / "varcand_x.csv")
    assert list(out["below_limit"]) == [False, True, True, False]


def test_g1030_aavso_writes_a_below_limit_point_as_fainter_than(tmp_path) -> None:
    """G10-30: an exported light-curve row with below_limit = True goes to the AAVSO file as "<mag" (MAGERR na,
    NOTES LIMIT=1sigma) and is not written to VarAstro as a measured magnitude."""
    from config import AppConfig
    from export_reports import export_lightcurve_reports
    from photometry_core import TIME_BASE_BJD_TDB

    lc = pd.DataFrame({"jd": [2460000.5, 2460000.6], "bjd": [2460000.5, 2460000.6], "mag_calib": [12.5, 15.2],
                       "mag_calib_final": [12.5, 15.2], "err": [0.01, 1.0], "flag": ["normal", "normal"],
                       "time_base": [TIME_BASE_BJD_TDB] * 2, "below_limit": [False, True]})
    reports = tmp_path / "lightcurves_reports"
    reports.mkdir(parents=True)
    paths = export_lightcurve_reports(
        reports, pd.Series({"vsx_name": "LIM_STAR", "vsx_type": "EA", "catalog_id": "222"}), lc,
        pd.DataFrame({"catalog_id": ["999"], "mag": [12.0]}),
        pd.Series({"aperture_px": 5.0, "n_frames": 10, "n_good_comp": 5, "lc_rms": 0.02, "band_system": "B"}),
        observer_code="TEST", cfg=AppConfig(), obs_group="B_20_2",
    )
    rows = [ln for ln in paths["aavso"].read_text(encoding="ascii").splitlines() if ln.startswith("LIM_STAR,")]
    assert len(rows) == 2
    f0, f1 = rows[0].split(","), rows[1].split(",")
    assert f0[2] == "12.500" and f1[2] == "<15.200" and f1[3] == "na" and "LIMIT=1sigma" in f1[-1]
    var_rows = [ln for ln in paths["varastro"].read_text(encoding="ascii").splitlines() if ln and not ln.startswith("#")]
    assert len(var_rows) == 1 and "12.5000" in var_rows[0]
