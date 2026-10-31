"""Phase 2 step 4 (code review 2 result bugs): small contract tests."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
sys.path.insert(0, str(SRC))


def test_comp_rms_printed_as_magnitude() -> None:
    """NEW-1 (step-4 triage): comp_rms is a LOO magnitude MAD -> mmag = 1000 x, not 1085.7 x."""
    for name in ("export_reports.py", "photometry_report.py", "star_page_m3.py"):
        src = (SRC / name).read_text(encoding="utf-8")
        assert not re.search(r"1085\.7\d*\s*\*\s*(float\(pd\.to_numeric\(_cr\.get\(\"comp_rms\"|med_rms_pool)", src), name
        assert "rms * 1085.7" not in src, name


def test_methods_matrix_comp_rules_from_constants() -> None:
    """G11-08: the comparison-star line of the methods matrix is built from the rule constants."""
    from comp_blend import BLEND_FLUX_FRAC_MAX
    from config import AppConfig
    from pipeline_constants import COMP_SNR_MIN

    src = (SRC / "citations.py").read_text(encoding="utf-8")
    assert ">= 0.90 SAT" not in src
    for c in ("ctx.comp_peak_sat_frac", "COMP_SNR_MIN", "BLEND_FLUX_FRAC_MAX"):
        assert c in src
    assert (AppConfig().comp_peak_sat_frac, COMP_SNR_MIN, BLEND_FLUX_FRAC_MAX) == (0.80, 20.0, 0.01)


def test_pdf_name_date_is_the_night() -> None:
    """G11-09: "YYYY.MM.DD" of the first epoch -> the file-name token (was today's date)."""
    src = (SRC / "photometry_report.py").read_text(encoding="utf-8")
    i = src.index("self.obs_date_human = self._obs_date_str()")
    block = src[i:i + 600]
    m = re.search(r"re\.match\(r\"(.+?)\"", block)
    assert m
    mm = re.match(m.group(1).replace("\\\\", "\\"), "2026.10.03")
    assert mm and "".join(mm.groups()) == "20261003"


def test_kepler_converges_for_high_eccentricity() -> None:
    """G10-01: Danby start, M in [-pi, pi]; every (M, e) converges (E0 = pi failed for M < 0 at e >= 0.8)."""
    import numpy as np

    from minor_bodies import _kepler_E

    M = np.linspace(-12.0, 12.0, 4001)
    for e in (0.0, 0.5, 0.8, 0.9, 0.99, 0.999):
        E = _kepler_E(M, np.full_like(M, e))
        assert np.all(np.isfinite(E)), e
        assert np.max(np.abs(E - e * np.sin(E) - M)) < 1e-9, e


def test_mk_luminosity_class_parser() -> None:
    """G10-14: the first class after the temperature type, longest token first."""
    from hrd_analysis import _parse_luminosity_class as p

    cases = {"K3IIIb": "III", "G8IIIa": "III", "K0IVa": "IV", "G8III:": "III", "K2III+F": "III",
             "M2Iab": "IAB", "B1Ia+": "IA", "A0V": "V", "K2/3III": "III", "G8-K0III": "III", "M1.5Ib": "IB",
             "sdB": None, "DA": None, "CV": None, "": None}
    for sp, want in cases.items():
        assert p(sp) == want, (sp, p(sp))


def test_simbad_otype_sets() -> None:
    """G10-15: only explicit SIMBAD otypes confirm 'very cool' / 'binary'."""
    from hrd_analysis import _SIMBAD_OTYPES_BINARY, _SIMBAD_OTYPES_VERY_COOL

    for ot in ("Cl*", "CV*", "Ce*"):
        assert ot.upper() not in _SIMBAD_OTYPES_VERY_COOL
    assert "C*" in _SIMBAD_OTYPES_VERY_COOL and "MI*" in _SIMBAD_OTYPES_VERY_COOL
    assert "*" not in _SIMBAD_OTYPES_BINARY and "EB*" in _SIMBAD_OTYPES_BINARY


def test_bls_finds_the_eclipse_period() -> None:
    """G09-08: BLS runs on -mag (eclipses are dips in brightness)."""
    src = (SRC / "tess_analysis.py").read_text(encoding="utf-8")
    assert "BoxLeastSquares(t, -m, e)" in src


def test_epoch_of_minimum_ignores_empty_bins() -> None:
    """G10-18: a bin with fewer than 3 finite points is not the minimum."""
    import numpy as np

    from variability_store import epoch_of_minimum

    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 5, 2000))
    P = 0.5
    ph = (t % P) / P
    m = 10.0 + 0.5 * np.exp(-((ph - 0.3) ** 2) / (2 * 0.02**2))
    m[(ph > 0.6) & (ph < 0.62)] = np.nan  # an all-NaN bin
    t0 = epoch_of_minimum(t, m, P, n_bins=50)
    assert abs(((t0 - t.min()) % P) / P - 0.3) < 0.02


def test_field_search_removes_previous_products(tmp_path: Path) -> None:
    """G10-06: a rerun that stops early leaves no stale candidates / transient files."""
    from config import AppConfig
    from variability_field import run_field_variability

    phot = tmp_path / "photometry"
    (phot / "lightcurves").mkdir(parents=True)
    frames = tmp_path / "frames"
    frames.mkdir()
    stale = [phot / "variability_field.csv", phot / "transient_candidates.csv", phot / "moving_objects.csv",
             phot / "minor_bodies_in_field.csv", phot / "lightcurves" / "varcand_1.csv",
             phot / "lightcurves" / "transient_T1.csv"]
    for p in stale:
        p.write_text("x\n", encoding="utf-8")
    keep = phot / "lightcurves" / "lightcurve_1.csv"
    keep.write_text("x\n", encoding="utf-8")
    meta = run_field_variability(frames, phot, AppConfig(), platesolve_dir=tmp_path)
    assert meta["status"] == "too few frames"
    assert not any(p.exists() for p in stale)
    assert keep.exists()


def test_failed_psf_frame_keeps_no_old_values(tmp_path: Path) -> None:
    """G09-06: the PSF columns of a frame that failed in this run are cleared (not the previous run's)."""
    import pandas as pd

    from epsf_psf_merge import clear_psf_frame_result

    p = tmp_path / "proc_Light_001.csv"
    pd.DataFrame({"catalog_id": ["1", "2"], "x": [1.5, 2.5], "flux_k100": [100.0, 200.0],
                  "psf_flux": [99.0, 198.0], "psf_flux_err": [1.0, 2.0], "psf_chi2": [1.1, 0.9],
                  "psf_converged": [True, True], "psf_sat_masked": [False, True], "psf_edge_masked": [False, False],
                  "psf_fit_ok": [True, True]}).to_csv(p, index=False)
    clear_psf_frame_result(p, frame_name="Light_001.fits")
    d = pd.read_csv(p, dtype={"catalog_id": str})
    assert d["psf_flux"].isna().all() and d["psf_chi2"].isna().all()
    assert not d["psf_fit_ok"].astype(bool).any() and not d["psf_converged"].astype(bool).any()
    assert not d["psf_sat_masked"].astype(bool).any()
    assert list(d["flux_k100"]) == [100.0, 200.0] and list(d["x"]) == [1.5, 2.5]


def test_sat_limit_provenance_is_not_relabelled_as_measured() -> None:
    """G02-02: no frames -> the stated header level stands (unknown max is no refutation); a refuted stated
    value never turns another winner into the measured CONFLICT_DERIVED."""
    import math

    from astropy.io import fits

    from sat_diag import PileupResult, resolve_sat_limit

    h = fits.Header()
    h["BITPIX"] = 16
    h["SATURATE"] = 60000
    ctx = resolve_sat_limit(hdr=h, pileup=PileupResult(False, None, 0, 0, float("nan"), 0, None), equipment_adu=None)
    assert ctx.sat_source == "HEADER" and ctx.sat_adu == 60000.0
    assert ctx.measured_sat_adu() is None
    h2 = fits.Header()
    h2["BITPIX"] = 16
    h2["BZERO"] = 32768
    h2["SATURATE"] = 30000  # refuted by a 40000 ADU pixel, no pile-up
    ctx2 = resolve_sat_limit(hdr=h2, pileup=PileupResult(False, None, 1, 1, 40000.0, 3, 65535.0), equipment_adu=60000.0)
    assert ctx2.sat_source == "EQUIPMENT" and ctx2.measured_sat_adu() is None
    assert not math.isnan(ctx2.sat_adu)


def test_image_adu_array_is_physical(tmp_path: Path) -> None:
    """G02-06: astropy applies BZERO/BSCALE on read; uint16, uint32, int16 and float32 come back unchanged."""
    import numpy as np
    from astropy.io import fits

    from sat_diag import image_adu_array

    for dt, vals in ((np.uint16, [1, 40000]), (np.uint32, [1, 40000]), (np.int16, [-5, 3000]), (np.float32, [1.5, 4e4])):
        p = tmp_path / f"{np.dtype(dt).name}.fits"
        fits.PrimaryHDU(np.array([vals], dtype=dt)).writeto(p)
        with fits.open(p) as h:
            assert list(image_adu_array(h[0]).ravel()) == [float(v) for v in vals], dt


def test_plateau_is_relative_to_the_sky() -> None:
    """G05-01 (bug part): a faint star on a high sky is not a saturated plateau; a clipped core is."""
    import numpy as np

    from pipeline_catalog import _saturated_core_plateau_vectorized

    yy, xx = np.mgrid[0:41, 0:41].astype(float)
    star = 100.0 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 1.5**2))
    for sky in (1500.0, 33000.0):
        img = sky + star + np.random.default_rng(1).normal(0, 0.5, star.shape)
        assert not _saturated_core_plateau_vectorized(img, np.array([20.0]), np.array([20.0]))[0], sky
    clipped = np.minimum(1500.0 + 300000.0 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 1.5**2)), 50000.0)
    assert _saturated_core_plateau_vectorized(clipped, np.array([20.0]), np.array([20.0]))[0]


def test_one_flat_top_star_is_not_a_plateau(tmp_path: Path) -> None:
    """G02-35: the pixels of one clipped core count as one peak."""
    import numpy as np
    from astropy.io import fits

    from sat_diag import N_PLATEAU_MIN_PEAKS, _star_peak_plateau

    yy, xx = np.mgrid[0:200, 0:200].astype(float)
    img = 100.0 + np.minimum(80000.0 * np.exp(-((xx - 100) ** 2 + (yy - 100) ** 2) / (2 * 3.0**2)), 30000.0)
    p = tmp_path / "one.fits"
    fits.PrimaryHDU(img.astype(np.float32)).writeto(p)
    assert not _star_peak_plateau([p]).pileup_detected
    img2 = np.full((200, 200), 100.0)
    rng = np.random.default_rng(3)
    for x0, y0 in rng.uniform(15, 185, (N_PLATEAU_MIN_PEAKS + 4, 2)):
        img2 = np.maximum(img2, 100.0 + np.minimum(80000.0 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 3.0**2)), 30000.0))
    p2 = tmp_path / "many.fits"
    fits.PrimaryHDU(img2.astype(np.float32)).writeto(p2)
    assert _star_peak_plateau([p2]).pileup_detected


def test_noisy_moon_is_informational_like_noisy() -> None:
    """G08-04: every informational LC quality gives the same trust treatment; noisy_moon is not a hard RED."""
    from photometry_defs import LC_QUALITY_FLAGS, LC_QUALITY_INFORMATIONAL
    from trust_flag_core import _LC_QUALITY_OK

    assert _LC_QUALITY_OK is LC_QUALITY_INFORMATIONAL
    assert {"good", "noisy", "noisy_moon"} == set(LC_QUALITY_INFORMATIONAL)
    assert set(LC_QUALITY_INFORMATIONAL) <= set(LC_QUALITY_FLAGS)


def test_comp_stability_residual_is_leave_one_out() -> None:
    """G08-01: a variable comp of middle brightness is the only suspect; a constant offset changes nothing."""
    import numpy as np

    from photometry_lightcurve import _comp_lc_frame_ensemble_residual, check_comparison_stability

    rng = np.random.default_rng(5)
    n = 120
    t = np.linspace(0.0, 0.25, n)  # 6 h in days
    common = 0.05 * np.sin(2 * np.pi * t / 0.25)
    lcs = {f"c{i}": 10.0 + i + common + rng.normal(0, 0.003, n) for i in range(5)}
    lcs["c2"] = lcs["c2"] + 0.06 * t / 0.25  # 60 mmag over 6 h (10 mmag/h): the middle comp varies
    q = check_comparison_stability(lcs, comp_bjd={k: 2460000.0 + t for k in lcs})
    sus = sorted(k for k, v in q.items() if v["quality"] != "good")
    assert sus == ["c2"], q
    r1 = _comp_lc_frame_ensemble_residual(lcs)
    lcs2 = dict(lcs)
    lcs2["c4"] = lcs2["c4"] + 0.7
    r2 = _comp_lc_frame_ensemble_residual(lcs2)
    for k in lcs:
        assert np.allclose(r1[k], r2[k], equal_nan=True)


def test_export_site_from_the_draft_meta(tmp_path: Path) -> None:
    """G11-04: the site of the export header comes from photometry/pipeline_meta.json (was looked up in
    lightcurves_reports/, never found -> the live config site)."""
    import json

    from export_reports import _resolved_site_from_meta

    phot = tmp_path / "photometry"
    (phot / "lightcurves_reports").mkdir(parents=True)
    (phot / "pipeline_meta.json").write_text(json.dumps(
        {"observer_location": {"name": "Jirny", "lat": 50.11, "lon": 14.70, "alt_m": 240.0}}), encoding="utf-8")
    assert _resolved_site_from_meta(phot)["name"] == "Jirny"
    src = (SRC / "export_reports.py").read_text(encoding="utf-8")
    assert "_resolved_site_from_meta(_phot_dir)" in src and "_resolved_site_from_meta(out_base)" not in src


def test_chip_edge_tested_on_full_coverage_and_untransformed_frames() -> None:
    """NEW-2 / G06-07: a star whose aperture leaves the chip is an edge star even when the frame covers the
    whole chip or has no transform (the early return skipped the test)."""
    import numpy as np

    from aperture_pertarget import measure_flux_on_f_grid
    from frame_footprint import off_footprint

    off = np.zeros((100, 100), dtype=bool)
    xs, ys = np.array([1.0, 50.0]), np.array([50.0, 50.0])
    assert list(off_footprint(off, xs, ys, 3.0, 8.0, 15.0)) == [True, False]
    img = np.random.default_rng(1).normal(100.0, 3.0, (100, 100))
    _, _, edge = measure_flux_on_f_grid(img, np.c_[xs, ys], f_grid=[1.0], fwhm_px=3.0,
                                        annulus_inner_fwhm=2.7, annulus_outer_fwhm=5.2, covered=None)
    assert list(edge) == [True, False]


def test_edge_fail_survives_the_lc_flags() -> None:
    """G08-05: upstream verdicts enter assign_lc_flags as base_flags (edge_fail was turned into no_data)."""
    import numpy as np

    from lc_outlier import assign_lc_flags

    m = np.array([10.0, np.nan, 10.01, np.nan, 10.0, 10.02])
    e = np.full(6, 0.01)
    r = assign_lc_flags(m, e, np.arange(6.0), base_flags=["normal", "edge_fail", "normal", "nondetection", "low_snr", "normal"])
    assert r.flags[1] == "edge_fail" and r.flags[3] == "nondetection" and r.flags[4] == "low_snr"
    src = (SRC / "aperture_photometry.py").read_text(encoding="utf-8")
    assert "base_flags=base_flags" in src and "_preserve_nondetection_flags_helper" not in src


def test_css_is_the_catalina_periodic_catalogue() -> None:
    """G10-11: the CSS check queries Drake et al. 2014 (J/ApJS/213/9 table3), not Coughlin 2014 (Kepler)."""
    src = (SRC / "catalog_crossmatch.py").read_text(encoding="utf-8")
    assert "J/ApJS/213/9/table3" in src and "\"J/AJ/147/119/table1\"" not in src
    from catalog_crossmatch import _CSS_CLASS

    assert len(_CSS_CLASS) == 17 and _CSS_CLASS[1] == "EW" and _CSS_CLASS[4] == "RRAB"


def test_no_aigrain_citation() -> None:
    """G11-42: only methods VYVAR uses are cited."""
    assert "aigrain2004" not in (SRC / "citations.py").read_text(encoding="utf-8")


def test_no_comparison_pool_copy_between_setups() -> None:
    """G03-09 / B6 (Milan 2026-10-29): every filter setup selects its own comparison stars."""
    for name in ("pipeline_astrometry.py", "masterstar_build.py"):
        assert "_sync_comparison_stars_across_setups" not in (SRC / name).read_text(encoding="utf-8")


def test_comparison_saturation_is_one_configurable_rule(tmp_path: Path) -> None:
    """COMP-SAT-ONE-2026-10 (B5, Milan 2026-10-29): comp_peak_sat_frac (config, default 0.80) on any frame."""
    import pandas as pd

    from comp_pool_rms import comp_frame_stats
    from config import AppConfig
    from pipeline_constants import LINEARITY_FRAC

    assert AppConfig().comp_peak_sat_frac == 0.80
    lim85 = 0.9 * 60000.0
    cache, paths = {}, []
    for i in range(20):
        p = tmp_path / f"proc_{i:03d}.csv"
        paths.append(p)
        pk_a = 0.85 * 60000.0 if i == 7 else 1000.0  # one frame at 0.85 SAT
        cache[str(p)] = pd.DataFrame({"catalog_id": ["A", "B"], "dao_flux": [1.0, 1.0],
                                      "peak_max_adu": [pk_a, 0.79 * 60000.0], "saturate_limit_adu_85pct": [lim85, lim85]})
    st = comp_frame_stats({"A", "B"}, paths, cache, comp_peak_sat_frac=0.80)
    assert st["A"]["n_over"] == 1 and st["B"]["n_over"] == 0  # any frame decides (was: > 10 % of frames)
    st9 = comp_frame_stats({"A", "B"}, paths, cache, comp_peak_sat_frac=0.86)
    assert st9["A"]["n_over"] == 0
    assert LINEARITY_FRAC == 0.90
    src = (SRC / "photometry_comp.py").read_text(encoding="utf-8")
    assert "0.10 * _n_pk" not in src and "saturated_any_frame_ids" not in src


def test_one_colour_term_path() -> None:
    """CT-ONE-2026-10 (B3): no G-based "fit" colour term anywhere."""
    for name in ("aperture_photometry.py", "aperture_photometry_target.py", "photometry_lightcurve.py",
                 "band_classify.py", "photometry_core.py"):
        src = (SRC / name).read_text(encoding="utf-8")
        assert "fit_color_term_c1" not in src and "should_apply_color_term" not in src, name
        assert 'mode="fit"' not in src, name


def test_check_star_processed_like_the_target() -> None:
    """B2 / G11-06 / G08-45 (Milan 2026-10-29): one band colour-term correction for target and check star,
    and the check K series receives the target's k'' + colour-term corrections."""
    import numpy as np

    from aperture_photometry_target import _band_ct_correction
    from color_transform import fit_colour_term

    rng = np.random.default_rng(2)
    col = rng.uniform(0.5, 2.0, 40)
    pf = fit_colour_term(col, 0.1 * (col - 1.0) + rng.normal(0, 0.002, 40), max_order=1)
    comps = {f"c{i}": float(c) for i, c in enumerate(col[:8])}
    cat = {k: 12.0 for k in comps}
    w = {k: 1.0 for k in comps}
    c_t, ens = _band_ct_correction(pf, 1.5, comps, cat, w)
    c_k, _ = _band_ct_correction(pf, 0.8, comps, cat, w)
    assert abs((c_t - c_k) - 0.1 * (1.5 - 0.8)) < 0.01
    assert np.isnan(_band_ct_correction(pf, 9.0, comps, cat, w)[0])  # no extrapolation
    src = (SRC / "aperture_photometry_target.py").read_text(encoding="utf-8")
    assert "kmag=np.asarray(_chk_result.kmag, dtype=np.float64) + _k_corr" in src


def test_no_telescope_diameter_no_scintillation_term() -> None:
    """C3 / G08-16 (Milan 2026-10-29): no D -> the scintillation term is left out (run continues) with a note;
    no generic 0.07 m / 300 m."""
    import math

    from config import AppConfig
    from sigma_budget import resolve_rig_scintillation_params
    from sigma_floor_core import scintillation_mag_per_epoch

    rig = resolve_rig_scintillation_params(draft_id=None, setup="g|120|4", exposure_s=120.0, cfg=AppConfig())
    assert math.isnan(rig.telescope_diameter_m)
    assert any("scintillation NOT included" in n for n in rig.source_notes)
    assert scintillation_mag_per_epoch(telescope_diameter_m=rig.telescope_diameter_m, airmass=1.5,
                                       exposure_s=rig.exposure_s, altitude_m=rig.altitude_m) == 0.0
    src = (SRC / "export_reports.py").read_text(encoding="utf-8")
    assert "scintillation NOT included" in src


# --- B1 (G08-06): high_err tests the random error against its airmass model --------------------
import math  # noqa: E402

import numpy as np  # noqa: E402


def _b1_series(n=120, seed=3):
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.full(100, 1.15) + rng.normal(0, 0.01, 100), np.linspace(1.2, 2.8, n - 100)])
    t = np.arange(n) / 1440.0
    m = np.full(n, 12.0) + rng.normal(0, 0.002, n)
    return rng, x, t, m


def test_b1_floor_dominated_jitter_is_not_high_err():
    """Review example: 6 mmag floor + 1 mmag photon with 5 % jitter -> nothing flagged."""
    from lc_outlier import assign_lc_flags
    rng, x, t, m = _b1_series()
    ep = 0.001 * (1 + 0.05 * rng.uniform(-1, 1, len(m)))
    err = np.sqrt(ep ** 2 + 0.006 ** 2)
    r = assign_lc_flags(m, err, t, err_photon=ep, airmass=np.full(len(m), 1.15))
    assert r.n_high_err == 0


def test_b1_airmass_growth_is_not_high_err():
    """Photon error growing with airmass (extinction, sky) and scintillation growth are expected."""
    from lc_outlier import assign_lc_flags
    rng, x, t, m = _b1_series()
    ep = 0.002 * np.exp(0.5 * (x - 1.0)) * (1 + 0.02 * rng.normal(size=len(m)))
    scint = 0.003 * x ** 1.75
    err = np.sqrt(ep ** 2 + scint ** 2)
    r = assign_lc_flags(m, err, t, err_photon=ep, airmass=x)
    assert r.n_high_err == 0


def test_b1_inflated_photon_error_is_high_err():
    """A frame with 2x photon error at normal brightness and airmass is flagged, with its ratio."""
    from lc_outlier import FLAG_HIGH_ERR, assign_lc_flags
    rng, x, t, m = _b1_series()
    ep = 0.002 * np.exp(0.5 * (x - 1.0)) * (1 + 0.02 * rng.normal(size=len(m)))
    ep[40] *= 2.0
    r = assign_lc_flags(m, ep, t, err_photon=ep, airmass=x)
    assert r.flags[40] == FLAG_HIGH_ERR and r.n_high_err == 1
    q = float(r.reasons[40].split(" x")[1].split()[0])
    assert r.reasons[40].startswith("high_err:err_photon x") and 1.9 < q < 2.1


# --- B4 (G03-14): no fixed 10" veto; a variable neighbour's light at its VSX maximum ---------------
def test_b4_vsx_amplitude_conventions():
    from catalog_db import vsx_amplitude_mag

    assert math.isclose(vsx_amplitude_mag(12.0, 13.5), 1.5)
    assert math.isclose(vsx_amplitude_mag(11.497, 0.859), 0.859)  # VSX "(0.859)": the amplitude itself
    assert math.isnan(vsx_amplitude_mag(12.0, None))


def _b4_field(tmp_path, with_vt):
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [200.0, 40.0]
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]  # 1"/px
    fits.PrimaryHDU(np.zeros((200, 200), np.float32), header=w.to_header()).writeto(tmp_path / "MASTERSTAR.fits")
    (ra_c, de_c), (ra_n, de_n) = w.all_pix2world([[99.0, 99.0], [105.0, 99.0]], 0)
    pd.DataFrame({"catalog_id": ["111", "222"], "ra_deg": [ra_c, ra_n], "dec_deg": [de_c, de_n],
                  "mag": [12.0, 16.0], "bp_rp": [np.nan, np.nan]}).to_csv(tmp_path / "field_catalog_cone.csv", index=False)
    if with_vt:
        pd.DataFrame({"name": ["V1"], "catalog_id": ["222"], "vsx_mag_max": [12.5],
                      "vsx_amplitude_mag": [4.0]}).to_csv(tmp_path / "variable_targets.csv", index=False)
    pool = pd.DataFrame({"catalog_id": ["111"], "x": [99.0], "y": [99.0], "ra_deg": [ra_c], "dec_deg": [de_c],
                         "mag": [12.0], "bp_rp": [np.nan]})
    return pool


def test_b4_variable_neighbour_judged_at_maximum(tmp_path):
    from comp_blend import BLEND_FLUX_FRAC_MAX, blend_contamination_for_pool

    out = {}
    for with_vt in (False, True):
        d = tmp_path / str(with_vt)
        d.mkdir()
        pool = _b4_field(d, with_vt)
        c, meta = blend_contamination_for_pool(
            pool, field_catalog_csv=d / "field_catalog_cone.csv", masterstar_fits=d / "MASTERSTAR.fits",
            zp_band="G", fwhm_px=3.0, profile_fwhm_px=3.0, r_ap_px=3.0, sat_adu=None,
            annulus_px=(8.1, 15.6), psf={"beta": 2.7})
        out[with_vt] = (float(c.iloc[0]), meta["variable_neighbours"])
    assert abs(out[False][0]) <= BLEND_FLUX_FRAC_MAX < abs(out[True][0])  # clean at the mean, a blend at maximum
    assert out[True][1]["n_at_maximum"] == 1 and out[False][1]["n_at_maximum"] == 0


def test_b4_no_fixed_arcsec_veto():
    import inspect

    import pipeline_astrometry

    src = inspect.getsource(pipeline_astrometry.select_comparison_stars_spatial_grid)
    assert "veto_arcsec" not in src and "variable_targets_df" not in src


def test_g08_09_nan_epoch_keeps_photon_errors_aligned():
    from sparse_trust_core import two_star_model_ratio_ci

    rng = np.random.default_rng(1)
    n = 60
    pk = np.full(n, 0.01)
    pc = np.full(n, 0.01)
    mk = 12 + rng.normal(0, 0.01, n)
    mc = 11 + rng.normal(0, 0.01, n)
    base = two_star_model_ratio_ci(mk, mc, pk, pc, sigma_sys_mag=0.0)
    mk2, pk2, pc2, mc2 = (np.insert(v, 5, x) for v, x in ((mk, np.nan), (pk, 3.0), (pc, 3.0), (mc, 11.0)))
    r = two_star_model_ratio_ci(mk2, mc2, pk2, pc2, sigma_sys_mag=0.0)
    assert math.isclose(r.v_model, base.v_model) and math.isclose(r.v_obs, base.v_obs)


def test_g08_07_series_in_time_order_not_file_name():
    import pandas as pd
    from photometry_lightcurve import _ensemble_scatter_by_source_file, _get_comp_bjd_series, _get_lc

    names = [f"proc_Light_{i}.csv" for i in (1, 2, 9, 10, 11)]  # string order 1, 10, 11, 2, 9
    jd = [1.0, 2.0, 9.0, 10.0, 11.0]
    rows = pd.DataFrame({"catalog_id": "A", "source_file": names, "jd": jd, "bjd": jd, "mag_inst": jd})
    rows = rows.sample(frac=1.0, random_state=3)
    assert list(_get_lc("A", rows)) == jd
    assert list(_get_comp_bjd_series("A", rows)) == jd
    m = _ensemble_scatter_by_source_file(rows, "A", np.array(jd) / 100)
    assert m["proc_Light_10.csv"] == 0.10 and m["proc_Light_2.csv"] == 0.02


def test_g08_34_frame_without_comp_sigma_has_unknown_ensemble_error():
    from photometry_lightcurve import ensemble_normalize

    n = 30
    t = np.full(n, 12.0)
    comps = {"a": np.full(n, 11.0), "b": np.full(n, 11.5)}
    comps["a"][3:] = np.nan  # two finite points: no sigma for a
    comps["b"][2:] = np.nan
    _m, _d, es = ensemble_normalize(t, comps, {"a": 11.0, "b": 11.5}, {}, n_comp_min=1)
    assert np.isnan(es[0]) and not np.any(es == 0.0)


def test_g08_49_gspc_zero_point_keeps_the_common_systematic():
    from gspc_zp import SIGMA_SYS_MAG, gspc_zero_point

    ids = [str(i) for i in range(25)]
    rows = {c: {"v_jkc_mag": 12.0, "v_jkc_relerr": 0.002, "v_jkc_flag": 1, "g": 12.0} for c in ids}
    gz = gspc_zero_point({c: 1.0 for c in ids}, {c: 0.0 for c in ids}, rows, "V")
    assert gz["n_comp"] == 25 and gz["sigma_model"] >= SIGMA_SYS_MAG and gz["sigma_data"] >= SIGMA_SYS_MAG


def test_g07_35_colour_ladder_widens_to_the_cap():
    from types import SimpleNamespace

    from photometry_comp import _bprp_tier_ladder_for_selection

    cfg = SimpleNamespace(comp_tier_bprp_limits=lambda: (0.15, 0.30, 0.55), comp_max_delta_bprp=0.5)
    assert _bprp_tier_ladder_for_selection(cfg, 0.5) == [0.15, 0.30, 0.5]


def test_g06_28_star_missing_by_id_never_takes_a_neighbour(tmp_path):
    import pandas as pd
    from photometry_core import read_flux_from_csv

    row = {"catalog_id": "222", "name": "222", "x": 110.0, "y": 100.0, "dao_flux": 5000.0, "bjd_tdb_mid": 2460898.41,
           "hjd_mid": 2460898.409, "jd_mid": 2460898.41, "aperture_r_px": 3.0, "noise_floor_adu": 1.0,
           "peak_max_adu": 1000.0, "sigma_bkg_ap": 10.0}
    p = tmp_path / "proc_frame.csv"
    pd.DataFrame([row]).to_csv(p, index=False)
    out = read_flux_from_csv(p, ["111"], {"111": 3.0}, gain=1.0, read_noise=10.0)
    assert len(out) == 1 and not np.isfinite(float(out.iloc[0]["mag_inst"]))


def test_g07_03_no_post_selection_colour_from_a_neighbour():
    src = (Path(__file__).resolve().parents[2] / "src_py" / "target_comp_selection.py").read_text(encoding="utf-8")
    assert "gaia_db_fallback" not in src and "query_local_gaia(" not in src


def test_g08_11_empty_apertures_avoid_the_alignment_fill():
    from photometry_gate_helpers import measure_empty_aperture_sigma_bkg

    rng = np.random.default_rng(11)
    clean = rng.normal(100.0, 5.0, (240, 240))
    filled = clean.copy()
    covered = np.ones(clean.shape, dtype=bool)
    covered[:, :48] = False  # 20 % constant alignment fill on the left
    filled[~covered] = 100.0
    sky_f = np.where(covered, filled, np.nan)
    e = np.array([])
    a, _, ra = measure_empty_aperture_sigma_bkg(clean, e, e, 3.0, 6.0, 10.0, n_apertures=64, min_valid=16, seed=4,
                                                sky=clean, covered=None)
    b, _, rb = measure_empty_aperture_sigma_bkg(filled, e, e, 3.0, 6.0, 10.0, n_apertures=64, min_valid=16, seed=4,
                                                sky=sky_f, covered=covered)
    assert ra == rb == "" and abs(b / a - 1.0) < 0.15


def test_g10_17_sun_colour_gives_the_sun_teff():
    from hrd_field_render import teff_from_bp_rp
    from mag_constants import SOLAR_BP_RP

    assert abs(float(teff_from_bp_rp(SOLAR_BP_RP)[0]) - 5772.0) < 50.0
    t = teff_from_bp_rp(np.linspace(-1.0, 6.0, 200))
    assert np.all(np.diff(t) <= 0) and np.all(np.isfinite(t))


def test_g09_41_tess_overlay_uses_delta_mag_without_zero_point(tmp_path, monkeypatch):
    import pandas as pd
    import tess_analysis
    from tess_runner import run_tess_for_star

    (tmp_path / "lightcurves").mkdir()
    pd.DataFrame({"catalog_id": ["7"], "ra_deg": [10.0], "dec_deg": [20.0], "mag": [12.0]}).to_csv(
        tmp_path / "active_targets.csv", index=False)
    pd.DataFrame({"bjd": [1.0, 2.0, 3.0], "mag_calib": [np.nan] * 3, "delta_mag": [0.0, 0.1, 0.0]}).to_csv(
        tmp_path / "lightcurves" / "lightcurve_7.csv", index=False)
    seen = {}

    class _R:
        verdict, reasons = "x", []

    def _fake(cid, ra, dec, out, *, cfg, vyvar_lc, progress, target_mag):
        seen["lc"] = vyvar_lc
        return _R()

    monkeypatch.setattr(tess_analysis, "run_tess", _fake)
    run_tess_for_star(tmp_path, "7", cfg=None)
    assert np.isfinite(seen["lc"][1]).all()


def test_g10_42_unprojected_catalogue_rows_are_dropped_before_the_tree():
    import inspect

    import transient_field

    src = inspect.getsource(transient_field.run_field_transients)
    i = src.index("gtree = cKDTree(")
    assert "np.isfinite(gx) & np.isfinite(gy)" in src[:i]


def test_c1_field_k_plan_from_brightness_groups(tmp_path):
    import json

    from photometry_defs import DEFAULT_APERTURE_F_GRID
    from aperture_pertarget import field_k_plan

    assert field_k_plan(tmp_path, DEFAULT_APERTURE_F_GRID) is None  # no aperture_per_target.json: no guessed k
    groups = [{"mag_lo": float("-inf"), "mag_hi": -11.0, "k": 1.5}, {"mag_lo": -11.0, "mag_hi": -9.0, "k": 1.0},
              {"mag_lo": -9.0, "mag_hi": float("inf"), "k": 0.6}]
    (tmp_path / "aperture_per_target.json").write_text(json.dumps({"brightness_k": {"brightness_k": {"groups": groups}}}))
    p = field_k_plan(tmp_path, DEFAULT_APERTURE_F_GRID)
    assert p["rule"] == "brightness_k" and p["k_ref"] == 1.5 and p["k_transient"] == 0.6
    assert p["groups"].k_for(-10.0) == 1.0
    (tmp_path / "aperture_per_target.json").write_text(json.dumps({"brightness_k": None}))
    q = field_k_plan(tmp_path, DEFAULT_APERTURE_F_GRID)
    assert q["rule"] == "no_group" and q["k_ref"] == q["k_transient"] == 1.0


def test_c1_no_silent_15_fallback():
    import inspect

    import variability_field

    src = inspect.getsource(variability_field.run_field_variability)
    assert "1.5" not in src and "COMP_RMS_APERTURE_K" not in src


def test_g10_41_hrd_field_png_north_up_and_markers_follow(tmp_path):
    from astropy.io import fits
    from PIL import Image

    from hrd_analysis import field_annotation_y, fits_first_image_to_png

    a = np.zeros((40, 30), np.float32)
    a[35, 10] = 1000.0  # star near the top of the FITS image (high y)
    fits.PrimaryHDU(a).writeto(tmp_path / "m.fits")
    assert fits_first_image_to_png(tmp_path / "m.fits", tmp_path / "m.png")
    g = np.asarray(Image.open(tmp_path / "m.png").convert("L"))
    row = int(np.argmax(g[:, 10]))
    assert row < 10 and abs(field_annotation_y(35, 1.0, 40) - (row + 0.5)) < 1.0


def test_g09_40_tess_sources_in_one_band():
    from tess_analysis import source_magnitudes_one_band

    s = [{"source_id": 1, "rp_mag": 10.0, "g_mag": 10.6}, {"source_id": 2, "rp_mag": None, "g_mag": 12.0}]
    assert source_magnitudes_one_band(s) == {"1": 10.6, "2": 12.0}
    s[1]["rp_mag"] = 11.5
    assert source_magnitudes_one_band(s) == {"1": 10.0, "2": 11.5}


def test_g09_22_neighbour_reach_covers_the_worst_seeing_fit_box():
    import inspect

    import epsf_psf_merge
    from epsf_frame import fit_box_px

    assert fit_box_px(3.9) == 9 and fit_box_px(8.7) == 19 and fit_box_px(1.0) == 5
    src = inspect.getsource(epsf_psf_merge)
    assert "fwhm_px=fwhm_reach" in src and "max(max(_fw_frames), fwhm_ms)" in src


def test_coverage_faint_point_counts_as_measured_upper_limit():
    import pandas as pd

    from variability_field import limit_magnitudes

    f = pd.Series([10000.0, -50.0, 30.0])
    var = pd.Series([100.0 ** 2, 100.0 ** 2, 100.0 ** 2])
    m, e, below = limit_magnitudes(f, var)
    assert list(below) == [False, True, True]
    assert np.isfinite(m).all() and m[1] == m[2] == -2.5 * np.log10(100.0) and abs(e[1] - 1.0857) < 1e-9


def test_g10_03_empty_sky_noise_only_on_frame_data():
    import inspect

    import transient_field

    src = inspect.getsource(transient_field.run_field_transients)
    assert "np.where(Fm[:, len(xy):, 3] > 0.5" in src and transient_field.EMPTY_SKY_MIN_VALID == 100


def test_g08_22_frame_qc_one_sided_for_seeing_and_sky():
    import pandas as pd

    from lc_outlier import frame_qc_mask_from_night_table

    rng = np.random.default_rng(2)
    fw = 4.0 + rng.normal(0, 0.1, 40)
    fw[5], fw[9] = 2.0, 6.0  # exceptionally good, bad
    out = frame_qc_mask_from_night_table(pd.DataFrame({"frame_key": [f"f{i}" for i in range(40)], "fwhm": fw}))
    assert out["f5"] == "" and out["f9"].startswith("fwhm_z=")


def test_g10_02_minor_body_detection_uses_unmatched_detections_only():
    import inspect

    import transient_field

    src = inspect.getsource(transient_field.run_field_transients)
    assert 'un.groupby("t")' in src and 'cKDTree(det[["x", "y"]]' not in src


def test_g10_04_transient_master_dark_from_the_frame_stamp(tmp_path):
    from types import SimpleNamespace

    from astropy.io import fits

    from transient_field import _master_dark

    lib = tmp_path / "lib"
    lib.mkdir()
    fits.PrimaryHDU(np.ones((20, 30), np.float32)).writeto(lib / "Dark_60s_Bin1.fits")
    fits.PrimaryHDU(np.ones((10, 15), np.float32)).writeto(lib / "Dark_30s_other.fits")
    h = fits.Header()
    h["VY_DARK"], h["VY_MDP"] = True, "Dark_60s_Bin1.fits"
    fits.PrimaryHDU(np.zeros((10, 15), np.float32), header=h).writeto(tmp_path / "f.fits")
    cfg = SimpleNamespace(calibration_library_root=lib)
    d, why = _master_dark(cfg, tmp_path / "f.fits")
    assert why == "" and d["file"].endswith("Dark_60s_Bin1.fits") and d["raw_shape"] == (10, 15)
    h["VY_DARK"] = False
    fits.PrimaryHDU(np.zeros((10, 15), np.float32), header=h).writeto(tmp_path / "g.fits")
    assert _master_dark(cfg, tmp_path / "g.fits")[0] is None  # never another group's dark


def test_g02_28_transient_threshold_noise_on_resampled_frame(tmp_path):
    from astropy.io import fits
    from scipy.ndimage import shift

    import transient_field

    rng = np.random.default_rng(1)
    b = shift(rng.normal(100, 10, (300, 300)), (0.4, 0.5), order=3, mode="nearest")
    h = fits.Header()
    for k, v in enumerate((1, 0, 0, 0, 1, 0, 0, 0, 1)):
        h[f"VY_AM{k}"] = float(v)
    fits.PrimaryHDU(b.astype(np.float32), header=h).writeto(tmp_path / "a.fits")
    seen = {}

    class _Dao:
        def __init__(self, **kw):
            seen["thr"] = kw["threshold"]

        def __call__(self, *a, **k):
            return None

    import photutils.detection as pd_

    orig = pd_.DAOStarFinder
    pd_.DAOStarFinder = _Dao
    try:
        transient_field._detect_one((str(tmp_path / "a.fits"), 0, 3.0))
    finally:
        pd_.DAOStarFinder = orig
    assert abs(seen["thr"] / transient_field.DETECT_NSIGMA / b.std() - 1.0) < 0.05


def test_g10_13_simbad_star_without_variable_type_is_not_known(monkeypatch):
    import pandas as pd

    import catalog_crossmatch as cc

    res = cc.CrossmatchResult(ra=1.0, dec=2.0, mag=12.0, radius_arcsec=15.0,
                              matches={"SIMBAD": [cc.CatalogMatch(catalog="SIMBAD", name="TYC 1-2-3", var_type="*")],
                                       "VSX": []}, errors={"ZTF": "timeout"})
    monkeypatch.setattr(cc, "check_candidate_in_catalogs", lambda *a, **k: res)
    row = pd.Series({"ra_deg": 1.0, "dec_deg": 2.0, "mag": 12.0})
    cfg = type("C", (), {"objects_db_path": None})()
    assert cc._catalog_check(row, cfg) == "none of the catalogues lists it (1 not reachable)"
    res.matches["SIMBAD"][0].var_type = "EB?"
    assert cc._catalog_check(row, cfg).startswith("SIMBAD: TYC 1-2-3")


def test_g11_22_vsx_name_only_for_vsx_targets():
    import pandas as pd

    from photometry_comp import _ensure_active_target_names
    from star_page_m3 import target_display_name

    df = pd.DataFrame({"catalog_id": ["1", "2"], "name": ["V1 Cyg", ""], "vsx_name": ["V1 Cyg", ""],
                       "exo_host_name": ["", "TOI-123"]})
    out = _ensure_active_target_names(df)
    assert list(out["vsx_name"]) == ["V1 Cyg", ""]
    assert target_display_name(out.iloc[1]) == "TOI-123"


def test_g06_18_colour_term_magnitudes_on_group_k(tmp_path):
    import pandas as pd

    from aperture_pertarget import BrightnessK
    from aperture_photometry import _group_comp_mag_inst_on_group_k

    # bright star (group k 1.5 = k_ref) and faint star (group k 0.6); ap. correction 0.6 -> 1.5 = -0.5 mag
    bk = BrightnessK([float("-inf"), -9.0], [-9.0, float("inf")], [1.5, 0.6], [30, 30], [{}, {}], [0.0, 0.0])
    kplan = {"groups": bk, "k_default": 1.0, "k0": 0.6, "k_ref": 1.5, "k_transient": 0.6, "k_all": [0.6, 1.0, 1.5]}
    files = []
    for i in range(3):
        df = pd.DataFrame({"catalog_id": ["1", "2"], "flux_k060": [10000.0, 1000.0], "flux_k100": [14000.0, 1400.0],
                           "flux_k150": [15848.93, 1584.893], "likely_saturated": [False, False],
                           "photometry_ok": [True, True]})
        p = tmp_path / f"proc_f{i}.csv"
        df.to_csv(p, index=False)
        files.append(p)
    out = _group_comp_mag_inst_on_group_k(["1", "2"], files, kplan)
    assert np.allclose(out["1"], -2.5 * np.log10(15848.93))  # bright: its own k_ref flux
    assert np.allclose(out["2"], -2.5 * np.log10(1000.0) - 0.5, atol=1e-4)  # faint: k 0.6 + correction


def test_g06_18_colour_term_fitted_after_the_grid():
    import inspect

    import aperture_photometry
    import aperture_photometry_state

    assert "_compute_group_color_term_fit(" not in inspect.getsource(aperture_photometry_state)
    src = inspect.getsource(aperture_photometry.run_aperture_photometry)
    assert src.index("_rap_measure_grid_and_choose_k(") < src.index("_rap_group_colour_term(")


def test_g10_43_dropped_groups_are_counted():
    import pandas as pd

    import transient_field as TF

    t = np.arange(8, dtype=float) / 24.0
    # one group: two crossing tracks merged (not one line), one 2-frame group
    xs = [100 + 10 * i if i % 2 == 0 else 140 - 10 * i for i in range(8)]
    un = pd.DataFrame({"t": list(range(8)) + [0, 1], "x": xs + [500, 500], "y": [200.0] * 8 + [50, 50]})
    lab = np.array([0] * 8 + [1, 1])
    dropped = {}
    TF.classify_groups(un, lab, t, fwhm_med=3.0, dropped=dropped)
    assert dropped == {"short": 1, "not_linear": 1}


def test_fwhm_bound_from_frame_size_and_no_fixed_fallback():
    import pytest
    from astropy.io import fits

    from aperture_pertarget import _fwhm_from_fits_header
    from aperture_policy import clamp_fwhm_px, fwhm_from_header_vy_fwhm

    assert clamp_fwhm_px(40.0, shape=(1000, 1200)) == 40.0  # long-focus seeing: valid (bound 50 px)
    assert clamp_fwhm_px(60.0, shape=(1000, 1200)) is None
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"], h["VY_FWHM"] = 2, 4000, 3000, 35.0
    assert fwhm_from_header_vy_fwhm(h) == 35.0
    del h["VY_FWHM"]
    with pytest.raises(ValueError):
        _fwhm_from_fits_header(h, night_fallback=float("nan"))


def test_no_data_aperture_has_no_proc_flux():
    import inspect

    import photometry_shared

    src = inspect.getsource(photometry_shared.enhance_catalog_dataframe_aperture)
    assert "_no_data = _no_data | ~_covered" in src and "_fk[_touches_no_data(_r_k)] = np.nan" in src


def test_warp_nan_stays_local_for_every_map():
    """WARP-NAN-2026-10: one missing pixel must not spread over a polynomial-aligned frame."""
    import numpy as np
    from skimage.transform import PolynomialTransform, SimilarityTransform, warp

    from vyvar_alignment_frame import warp_to_reference

    d = np.random.default_rng(0).normal(1000, 10, (120, 160)).astype(np.float32)
    d[50, 60] = np.nan
    sim = SimilarityTransform(translation=(3.5, -6.2), rotation=0.01)
    poly = PolynomialTransform(np.array([[3.5, 1, -0.01, 0, 0, 0], [-6.2, 0.01, 1, 0, 0, 0]]))
    old = warp(d, inverse_map=sim, output_shape=d.shape, order=3, mode="constant",
               cval=float(np.nanmedian(d)), clip=True, preserve_range=True)
    new = warp_to_reference(d, d.shape, sim)
    assert np.array_equal(np.isnan(old), np.isnan(new))
    assert np.nanmax(np.abs(old - new)) == 0.0
    assert 0 < int(np.isnan(warp_to_reference(d, d.shape, poly)).sum()) <= 16


def test_uncalibrated_lc_points_are_normal_not_no_data():
    """UNCAL-LC: without a band zero point mag_calib is NaN, but a finite delta_mag is a measurement."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from aperture_photometry_target import _tgt_lc_flags

    n = 12
    nan = np.full(n, np.nan)
    out = _tgt_lc_flags(
        _calibrated=False, _cfg=SimpleNamespace(snr_min=10.0), _err_rel_for_snr=np.full(n, 0.01),
        _raw_tf=pd.Series([""] * n), airmass_arr=np.linspace(1.1, 1.5, n),
        bjd=np.linspace(2460000.0, 2460000.2, n), ct_ok=False,
        delta_mag=np.random.default_rng(1).normal(0.0, 0.01, n), err=np.full(n, 0.01),
        err_photon_export=np.full(n, 0.01), mag_calib=nan.copy(), mag_calib_ct=nan.copy(),
        sat_flags=np.zeros(n, dtype=bool), state=SimpleNamespace(), target_cid="1",
        target_frames=pd.DataFrame({"source_file": [f"proc_f{i:03d}.csv" for i in range(n)]}), target_name="T",
    )
    assert out[3] == ["normal"] * n


def test_repaired_pixel_in_target_aperture_is_a_note_not_a_flag():
    """BADPIX-REPAIR: bp_repaired_px > 0 adds 'bad_pixel_repaired: N px' to flag_reason; the flag stays."""
    from types import SimpleNamespace

    import numpy as np
    import pandas as pd

    from aperture_photometry_target import _tgt_lc_flags

    n = 12
    nan = np.full(n, np.nan)
    tf = pd.DataFrame({"source_file": [f"proc_f{i:03d}.csv" for i in range(n)], "bp_repaired_px": [0] * n})
    tf.loc[4, "bp_repaired_px"] = 2
    out = _tgt_lc_flags(
        _calibrated=False, _cfg=SimpleNamespace(snr_min=10.0), _err_rel_for_snr=np.full(n, 0.01),
        _raw_tf=pd.Series([""] * n), airmass_arr=np.linspace(1.1, 1.5, n),
        bjd=np.linspace(2460000.0, 2460000.2, n), ct_ok=False,
        delta_mag=np.random.default_rng(1).normal(0.0, 0.01, n), err=np.full(n, 0.01),
        err_photon_export=np.full(n, 0.01), mag_calib=nan.copy(), mag_calib_ct=nan.copy(),
        sat_flags=np.zeros(n, dtype=bool), state=SimpleNamespace(), target_cid="1",
        target_frames=tf, target_name="T",
    )
    assert out[3] == ["normal"] * n
    assert out[2][4] == "bad_pixel_repaired: 2 px"
    assert all("bad_pixel" not in r for i, r in enumerate(out[2]) if i != 4)
