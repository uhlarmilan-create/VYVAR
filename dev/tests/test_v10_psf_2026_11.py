"""VYVAR 1.0 closing items, tag psf (Milan approved 2026-11-04): items 12, 13, 11.

Item 12: badpix repair takes its ePSF cutout from the one shared rule (psf_photometry.epsf_cutout_px).
Item 13: a source without a local sky of its own no longer biases its neighbours (PSF-SKY-GROUP-2026-11).
Item 11: target aperture light curves mark points below the 1-sigma limit (LC-BELOW-LIMIT-2026-11).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SRC = Path(__file__).resolve().parents[2] / "src_py"
# the synthetic-frame helpers of the step-7 psf tests (test_step7_psf_2026_10), whatever the pytest rootdir
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


# --- item 12 ----------------------------------------------------------------------------------------------

class _CutoutSeen(Exception):
    pass


@pytest.mark.parametrize("fw", [1.6, 2.5, 3.34, 5.0, 8.3])
def test_item12_badpix_repair_cutout_is_the_epsf_helper(monkeypatch: pytest.MonkeyPatch, fw: float) -> None:
    """The repair's model size is whatever psf_photometry.epsf_cutout_px returns for the frame FWHM (a spy that
    shifts the helper's value by 2 px proves the repair calls the helper and has no copy of the rule)."""
    import badpix_repair as bp
    import psf_photometry as pp

    true_helper = pp.epsf_cutout_px
    monkeypatch.setattr(pp, "epsf_cutout_px", lambda f: true_helper(f) + 2)
    seen: dict[str, int] = {}

    def spy(*args, **kw):
        seen["cutout"] = int(kw["cutout"])
        raise _CutoutSeen

    monkeypatch.setattr(bp, "_star_models", spy)
    n = 160
    yy, xx = np.mgrid[0:n, 0:n]
    sig = fw / 2.3548
    img = 1000.0 + 5e4 * np.exp(-((xx - 80.0) ** 2 + (yy - 80.0) ** 2) / (2 * sig**2))
    img += np.random.default_rng(1).normal(0.0, 5.0, img.shape)
    img[80, 81] = np.nan
    with pytest.raises(_CutoutSeen):
        bp.repair_missing_pixels(img, fwhm_px=fw, detection_sigma=5.0, sat_cal_adu=1e9)
    assert seen["cutout"] == true_helper(fw) + 2


def test_item12_one_cutout_rule_in_the_code() -> None:
    """The 12 x FWHM cutout rule lives only in psf_photometry (epsf_cutout_px); no module rebuilds it."""
    offenders = []
    for p in sorted(_SRC.glob("*.py")):
        if p.name == "psf_photometry.py":
            continue
        txt = p.read_text(encoding="utf-8")
        if re.search(r"_EPSF_CUTOUT_FWHM|_to_odd_cutout", txt):
            offenders.append(p.name)
    assert offenders == []


# --- item 13 ----------------------------------------------------------------------------------------------

def _fit_pair(monkeypatch: pytest.MonkeyPatch | None, sep: float, *, s_skyless: bool):
    """Faint star N (3e3) at ``sep`` px from a bright star S (3e5), Moffat ePSF fitted exactly; S optionally has no
    local sky of its own (the annulus helper returns NaN for it, as in a saturated star's halo)."""
    import epsf_frame as ef
    from test_step7_psf_2026_10 import _moffat_epsf, _noisy, _render

    fw = 3.34
    model = _moffat_epsf(fw, 41)
    s_xy, n_xy = (100.0, 100.0), (100.0 + sep, 100.0)
    img = _noisy(_render(model, (220, 220), [n_xy[0], s_xy[0]], [n_xy[1], s_xy[1]], [3e3, 3e5]))
    if s_skyless:
        orig = ef.annulus_sky_medians

        def no_sky_for_s(sky, pos, r_in, r_out):
            out = orig(sky, pos, r_in, r_out)
            out[1] = np.nan
            return out

        monkeypatch.setattr(ef, "annulus_sky_medians", no_sky_for_s)
    stars = pd.DataFrame({"catalog_id": ["N", "S"], "x": [n_xy[0], s_xy[0]], "y": [n_xy[1], s_xy[1]]})
    return ef.fit_frame_psf(img, model, stars, None, fwhm_px=fw, gain=1.0, read_noise_e=5.0)


def test_item13_skyless_star_does_not_bias_its_neighbour(monkeypatch: pytest.MonkeyPatch) -> None:
    """Old code: S without a sky was fitted with sky 0 (flux holds its fit box's sky pedestal) and that model was
    subtracted for N's pass-2 sky and fitted jointly with N: N 55 mmag too faint, its sky 0.7 ADU low. Now S takes
    its group's sky: N as with S's own sky (< 1 mmag, < 0.05 ADU); S itself stays unusable (flag 2048)."""
    import epsf_frame as ef

    ref = _fit_pair(None, 10.0, s_skyless=False)
    got = _fit_pair(monkeypatch, 10.0, s_skyless=True)
    dm = -2.5 * math.log10(float(got["psf_flux"].iloc[0]) / float(ref["psf_flux"].iloc[0]))
    assert abs(dm) < 1e-3
    assert abs(float(got["psf_sky"].iloc[0]) - float(ref["psf_sky"].iloc[0])) < 0.05
    assert bool(got["psf_converged"].iloc[0])
    assert not bool(got["psf_converged"].iloc[1])
    assert int(got["psf_flags"].iloc[1]) & 2048
    st = ef.frame_fit_stats(got, "f.fits")
    assert st["sky_from_group"] == 1 and st["sky_models_not_subtracted"] == 0
    st0 = ef.frame_fit_stats(ref, "f.fits")
    assert st0["sky_from_group"] == 0 and st0["sky_models_not_subtracted"] == 0


def test_item13_source_without_any_sky_is_not_subtracted_for_the_others() -> None:
    """A source without any sky (none in its group) is left out of the image subtracted for the pass-2 sky; with
    every sky finite the photutils model image is returned unchanged."""
    import epsf_frame as ef
    from photutils.psf import PSFPhotometry, SourceGrouper
    from astropy.table import Table
    from test_step7_psf_2026_10 import _moffat_epsf, _noisy, _render

    fw = 3.0
    model = _moffat_epsf(fw, 37)
    img = _noisy(_render(model, (160, 160), [50.0, 110.0], [80.0, 80.0], [2e4, 2e4]))
    psf = model.copy()
    psf.x_0.fixed = True
    psf.y_0.fixed = True
    phot = PSFPhotometry(psf, (7, 7), aperture_radius=4.5, grouper=SourceGrouper(min_separation=5.0))
    lb = np.array([1000.0, 1000.0])
    phot(img, init_params=Table({"id": [1, 2], "x": [50.0, 110.0], "y": [80.0, 80.0], "local_bkg": lb}))
    full = phot.make_model_image(img.shape)
    same, n0 = ef._pass1_model_without_skyless(phot, full, lb)
    assert n0 == 0 and same is full
    part, n1 = ef._pass1_model_without_skyless(phot, full, np.array([1000.0, np.nan]))
    assert n1 == 1
    assert part[80, 110] == pytest.approx(0.0, abs=1e-9)  # source 2 left out
    assert part[80, 50] == pytest.approx(full[80, 50], rel=1e-9)  # source 1 kept


def test_item13_group_sky_rule() -> None:
    import epsf_frame as ef

    xs = np.array([0.0, 10.0, 15.0, 200.0])
    ys = np.zeros(4)
    sky = np.array([np.nan, 990.0, 1010.0, np.nan])
    out = ef._sky_of_group_for_skyless(sky, xs, ys, 20.0)
    assert out[0] == pytest.approx(1000.0) and out[1] == 990.0 and out[2] == 1010.0 and np.isnan(out[3])


# --- item 11 ----------------------------------------------------------------------------------------------

def _grid_night(fluxes: dict[str, list[float]]):
    """One star per catalog id, one frame per flux; sky 100 ADU/px, 700 sky px, FWHM 4 px, k = 1."""
    from aperture_pertarget import ApertureGridNight

    n = len(next(iter(fluxes.values())))
    night = ApertureGridNight(f_grid=[1.0], fwhm_night_px=4.0)
    night.fwhm_by_frame = {f"f{j}": 4.0 for j in range(n)}
    night.frames = {f"f{j}": {c: {1.0: v[j]} for c, v in fluxes.items()} for j in range(n)}
    night.sky_pp = {f"f{j}": {c: {1.0: 100.0} for c in fluxes} for j in range(n)}
    night.sky_n = {f"f{j}": {c: 700 for c in fluxes} for j in range(n)}
    rows = [{"catalog_id": c, "source_file": f"proc_f{j}.csv", "mag_inst": 0.0, "err": 9.9, "flag": "normal"}
            for j in range(n) for c in fluxes]
    return night, pd.DataFrame(rows)


def _sigma_f(flux: float) -> float:
    from ccd_noise import ccd_variance_adu2

    return math.sqrt(ccd_variance_adu2(flux, 100.0, math.pi * 16.0, 700.0, gain=2.0, read_noise=4.0))


def test_item11_target_points_below_one_sigma_take_the_limit() -> None:
    """Target fluxes -50 / 0 / 20 ADU (below 1 sigma ~ 46 ADU) -> below_limit, mag_inst = -2.5 log10(sigma_F),
    err 1 (relative); 1e4 ADU measured as before; a comparison star with F <= 0 stays NaN (not a limit)."""
    from aperture_pertarget import apply_grid_fluxes_to_frames

    night, df = _grid_night({"T": [-50.0, 0.0, 20.0, 1.0e4], "C": [-50.0, 0.0, 20.0, 1.0e4]})
    out = apply_grid_fluxes_to_frames(df, night, catalog_ids=["T", "C"], f_star=1.0, gain=2.0, read_noise=4.0,
                                      limit_ids=["T"])
    t = out[out["catalog_id"] == "T"].reset_index(drop=True)
    assert t["below_limit"].tolist() == [True, True, True, False]
    for j, f in enumerate([-50.0, 0.0, 20.0]):
        assert t.loc[j, "mag_inst"] == pytest.approx(-2.5 * math.log10(_sigma_f(f)), rel=1e-12)
        assert t.loc[j, "err"] == pytest.approx(1.0)
    assert t.loc[3, "mag_inst"] == pytest.approx(-2.5 * math.log10(1.0e4), rel=1e-12)
    c = out[out["catalog_id"] == "C"].reset_index(drop=True)
    assert np.isnan(c.loc[0, "mag_inst"]) and np.isnan(c.loc[1, "mag_inst"])
    assert not c["below_limit"].any()


def test_item11_one_limit_rule() -> None:
    """The variability search and the target LCs share ccd_noise.limit_magnitudes (no copy)."""
    import ccd_noise
    import variability_field as vf

    assert vf.limit_magnitudes is ccd_noise.limit_magnitudes
    assert "def limit_magnitudes" not in (_SRC / "variability_field.py").read_text(encoding="utf-8")


def test_item11_below_limit_flag_is_kept_and_not_seen_by_the_outlier_tests() -> None:
    """assign_lc_flags keeps 'below_limit'; the other points get the flags they get with that point missing."""
    from lc_outlier import assign_lc_flags

    rng = np.random.default_rng(4)
    n = 40
    m = 12.0 + rng.normal(0, 0.01, n)
    e = np.full(n, 0.01)
    t = np.arange(n) / 100.0
    m_lim = m.copy()
    m_lim[20] = 16.0  # the 1-sigma limit, far fainter than the star
    e_lim = e.copy()
    e_lim[20] = 1.0
    base = ["normal"] * n
    base[20] = "below_limit"
    got = assign_lc_flags(m_lim, e_lim, t, base_flags=base)
    m_nan = m.copy()
    m_nan[20] = np.nan
    ref = assign_lc_flags(m_nan, e, t)
    assert got.flags[20] == "below_limit"
    assert [f for i, f in enumerate(got.flags) if i != 20] == [f for i, f in enumerate(ref.flags) if i != 20]


def test_item11_lc_csv_has_the_column_and_metrics_skip_limits(tmp_path: Path) -> None:
    from photometry_lightcurve import compute_lc_precision_p2p, compute_lc_rms_ooe, save_lightcurve_csv

    n = 6
    flags = ["normal", "normal", "below_limit", "normal", "normal", "normal"]
    mag = np.array([12.0, 12.01, 16.0, 11.99, 12.0, 12.02])
    p = tmp_path / "lightcurve_1.csv"
    z = np.zeros(n)
    save_lightcurve_csv(p, bjd=np.arange(n) + 2460000.0, hjd=z, jd=np.arange(n) + 2460000.0, airmass=z + 1.2,
                        mag_inst=mag, mag_calib=mag, delta_mag=mag, err=np.full(n, 0.01), aperture_r_px=z + 4,
                        flags=flags, source_files=[f"proc_{i}.csv" for i in range(n)])
    lc = pd.read_csv(p)
    assert lc["below_limit"].tolist() == [False, False, True, False, False, False]
    keep = np.array(flags) != "below_limit"
    assert compute_lc_precision_p2p(mag, flags) == compute_lc_precision_p2p(mag[keep], ["normal"] * 5)
    assert compute_lc_rms_ooe(mag, flags) == compute_lc_rms_ooe(mag[keep], ["normal"] * 5)


def test_item11_export_rows_limits_only_for_the_writers() -> None:
    """export_row_mask: a below_limit row is no measured epoch (PDF, trust, check star) unless include_limits
    (the AAVSO / VarAstro writers); an LC without the column (before LC-BELOW-LIMIT-2026-11) reads as before."""
    from photometry_lightcurve import export_row_mask, lc_below_limit_mask

    lc = pd.DataFrame({"jd": [1.0, 2.0, 3.0], "bjd": [1.0, 2.0, 3.0], "mag_calib": [12.0, 15.2, 12.1],
                       "err": [0.01, 1.0, 0.01], "flag": ["normal", "below_limit", "normal"],
                       "below_limit": ["False", "True", "False"]})
    assert export_row_mask(lc, mag_column="mag_calib").tolist() == [True, False, True]
    assert export_row_mask(lc, mag_column="mag_calib", include_limits=True).tolist() == [True, True, True]
    old = lc.drop(columns=["below_limit"]).assign(flag="normal")
    assert export_row_mask(old, mag_column="mag_calib").tolist() == [True, True, True]
    assert not lc_below_limit_mask(old).any()


def test_item11_target_lc_limit_point_is_written_as_fainter_than(tmp_path: Path) -> None:
    """A target LC row flagged below_limit (S/N <= 1: it used to be low_snr and dropped) reaches the AAVSO file
    as "<mag", MAGERR na; VarAstro leaves it out."""
    from config import AppConfig
    from export_reports import export_lightcurve_reports
    from photometry_core import TIME_BASE_BJD_TDB

    n = 12
    jd = 2460000.5 + np.arange(n) / 100.0
    mag = np.full(n, 12.5)
    mag[5] = 15.2
    lc = pd.DataFrame({"jd": jd, "bjd": jd, "mag_calib": mag, "mag_calib_final": mag,
                       "err": np.where(np.arange(n) == 5, 1.0, 0.01), "time_base": [TIME_BASE_BJD_TDB] * n,
                       "flag": ["below_limit" if i == 5 else "normal" for i in range(n)],
                       "below_limit": [i == 5 for i in range(n)]})
    reports = tmp_path / "lightcurves_reports"
    reports.mkdir(parents=True)
    paths = export_lightcurve_reports(
        reports, pd.Series({"vsx_name": "LIM_T", "vsx_type": "EA", "catalog_id": "222"}), lc,
        pd.DataFrame({"catalog_id": ["999"], "mag": [12.0]}),
        pd.Series({"aperture_px": 5.0, "n_frames": n, "n_good_comp": 5, "lc_rms": 0.02, "band_system": "B"}),
        observer_code="TEST", cfg=AppConfig(), obs_group="B_20_2",
    )
    rows = [ln.split(",") for ln in paths["aavso"].read_text(encoding="ascii").splitlines() if ln.startswith("LIM_T,")]
    assert len(rows) == n
    assert rows[5][2] == "<15.200" and rows[5][3] == "na"
    assert sum(r[2].startswith("<") for r in rows) == 1
    var_rows = [ln for ln in paths["varastro"].read_text(encoding="ascii").splitlines() if ln and not ln.startswith("#")]
    assert len(var_rows) == n - 1


def test_item11_psf_compare_does_not_use_limits() -> None:
    from epsf_lc_compare import AP_HARD_FAIL_FLAGS

    assert "below_limit" in AP_HARD_FAIL_FLAGS
