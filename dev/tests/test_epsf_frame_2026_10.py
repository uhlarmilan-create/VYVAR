"""EPSF-FRAME-2026-10: per-frame ePSF and the E-type fit set of RUN ePSF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from photutils.psf import CircularGaussianPRF

from epsf_science_set import build_epsf_fit_set
from epsf_frame import build_frame_model, fit_frame_psf, grid_shape_for, night_spatial_delta


def _frame(tmp_path: Path, fwhm: float, *, n: int = 64, seed: int = 5) -> tuple[Path, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    g = int(np.ceil(np.sqrt(n)))
    step = 36
    side = step * g + step
    yy, xx = np.mgrid[0:side, 0:side]
    img = np.zeros((side, side))
    rows = []
    for i in range(n):
        cx = step + step * (i % g) + rng.uniform(-0.5, 0.5)
        cy = step + step * (i // g) + rng.uniform(-0.5, 0.5)
        f = 10 ** rng.uniform(4.5, 5.2)
        img += CircularGaussianPRF(flux=f, x_0=cx, y_0=cy, fwhm=fwhm)(xx, yy)
        rows.append({"catalog_id": str(1000 + i), "x": cx, "y": cy, "flux": f})
    img = rng.poisson(img + 200.0).astype(np.float32)
    tmp_path.mkdir(parents=True, exist_ok=True)
    fp = tmp_path / f"frame_fwhm{fwhm}.fits"
    hdr = fits.Header()
    for k, v in enumerate((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):  # identity frame->reference transform
        hdr[f"VY_AM{k}"] = v
    fits.PrimaryHDU(img, header=hdr).writeto(fp, overwrite=True)
    return fp, pd.DataFrame(rows), img


def test_grid_shape_needs_fifty_stars_per_cell() -> None:
    # G09-12: every cell of the grid needs >= 50 build stars (not the total)
    rng = np.random.default_rng(0)

    def stars(n, xmax=900.0):
        return pd.DataFrame({"x": rng.uniform(0, xmax, n), "y": rng.uniform(0, 900, n)})

    assert grid_shape_for(stars(900), (900, 900)) == (3, 3)
    assert grid_shape_for(stars(400), (900, 900)) == (2, 2)
    assert grid_shape_for(stars(150), (900, 900)) == (1, 1)
    assert grid_shape_for(stars(900, xmax=440.0), (900, 900)) == (1, 1)  # all in the left half


@pytest.mark.parametrize("fwhm", [2.0, 3.0])
def test_frame_model_follows_the_frame_seeing(tmp_path: Path, fwhm: float) -> None:
    fp, stars, _ = _frame(tmp_path, fwhm)
    model, rec = build_frame_model(fp, stars, cutout_size=17, fwhm_px=2.5, delta=None, grid=(1, 1))
    assert rec["converged"] is True and rec["n_stars_used"] == len(stars)
    # pixel-integrated Gaussian: FWHM_eff ~ sqrt(fwhm^2 + 0.48); the model measures the frame, not the 2.5 px input
    assert rec["fwhm_px"] == pytest.approx(np.sqrt(fwhm**2 + 0.48), rel=0.12)
    arr = np.asarray(model.data, dtype=float)
    assert arr.sum() / 4 == pytest.approx(1.0, abs=1e-5)
    assert len(rec["sha256"]) == 64 and "model_file" not in rec
    assert not (tmp_path / "epsf_frames").exists()  # OUTPUT-REVIEW-2026-10: models in memory only


def test_frame_fit_recovers_fluxes_and_beats_a_model_of_other_seeing(tmp_path: Path) -> None:
    models = {}
    for fwhm in (2.0, 3.0):
        fp, stars, _ = _frame(tmp_path, fwhm, seed=9)
        models[fwhm], _ = build_frame_model(fp, stars, cutout_size=17, fwhm_px=2.5, delta=None, grid=(1, 1))
    _, stars, img = _frame(tmp_path, 3.0, seed=9)
    res = {}
    for label, model in (("own", models[3.0]), ("other", models[2.0])):
        df = fit_frame_psf(img, model, stars, fwhm_px=3.0, gain=1.0, read_noise_e=5.0)
        m = df.merge(stars, on="catalog_id")
        assert m["psf_converged"].all()
        res[label] = (float(np.median(m["psf_flux"] / m["flux"])), float(np.std(m["psf_flux"] / m["flux"])))
    assert res["own"][0] == pytest.approx(1.0, abs=0.02)
    assert abs(res["other"][0] - 1.0) > abs(res["own"][0] - 1.0)
    # flux errors: the fit (Fisher) error ~ the real scatter (within 2x)
    df = fit_frame_psf(img, models[3.0], stars, fwhm_px=3.0, gain=1.0, read_noise_e=5.0).merge(stars, on="catalog_id")
    pull = (df["psf_flux"] - df["flux"]) / df["psf_flux_err"]
    assert 0.5 < float(np.std(pull)) < 2.0


def test_night_delta_and_gridded_model(tmp_path: Path) -> None:
    fps = []
    for k in range(3):
        fp, stars, _ = _frame(tmp_path / f"f{k}", 2.5, n=225, seed=20 + k)
        fps.append(fp)
    sp = night_spatial_delta(fps, stars, cutout_size=17, fwhm_px=2.5, out_path=tmp_path / "d.fits")
    assert sp["grid"] == [2, 2] and sp["n_frames"] == 3  # 225 build stars: >= 50 per cell only on 2x2
    delta = fits.getdata(tmp_path / "d.fits")
    assert delta.shape[0] == 4
    # a field without spatial PSF variation: the perturbation is small
    assert max(sp["max_abs_delta_over_peak"]) < 0.1
    model, rec = build_frame_model(fps[0], stars, cutout_size=17, fwhm_px=2.5, delta=delta, grid=(2, 2))
    assert type(model).__name__ == "GriddedPSFModel" and rec["grid"] == [2, 2]
    assert np.asarray(model.data).shape[0] == 4


def _write(ps: Path, targets: list[tuple], comps: list[tuple[str, str]]) -> None:
    """targets: (catalog_id, vsx_type, zone_flag, catalog, skip_photometry, has_aperture_lc, name)."""
    phot = ps / "photometry"
    (phot / "lightcurves").mkdir(parents=True)
    pd.DataFrame([t[:5] + (t[6],) for t in targets],
                 columns=["catalog_id", "vsx_type", "zone_flag", "catalog", "skip_photometry", "name"]).to_csv(
        phot / "active_targets.csv", index=False)
    for t in targets:
        if t[5]:
            (phot / "lightcurves" / f"lightcurve_{t[0]}.csv").write_text("bjd,delta_mag\n")
    pd.DataFrame(comps, columns=["catalog_id", "target_catalog_id"]).to_csv(
        phot / "comparison_stars_per_target.csv", index=False
    )


_TARGETS = [
    ("1", "EW", "linear", "VSX", False, True, "V1 Cyg"),
    ("2", "EA:", "linear", "VSX", False, True, "V2 Cyg"),
    ("3", "RS|BY", "noise", "VSX", False, True, "ZTF J1"),
    ("4", "ROT", "linear", "VSX", True, False, "V4 Cyg"),          # out of scope: skipped, no aperture LC
    ("5", "EB", "catalog_only", "VSX", False, True, "V5 Cyg"),     # catalog_only zone
    ("6", "", "linear", "EXOPLANET", False, True, "HAT-P-1"),     # exoplanet host: fitted too
    ("7", "EW", "linear", "VSX", False, False, "V7 Cyg"),          # no aperture light curve
]
_COMPS = [("10", "1"), ("11", "1"), ("12", "2"), ("13", "3"), ("14", "5"), ("15", "6")]


def test_fit_set_all_takes_every_target_with_aperture_photometry(tmp_path: Path) -> None:
    """EPSF-ALL-TARGETS-2026-10: ALL = every target with an aperture light curve (VSX, exoplanet host,
    manual), not skipped, not catalog_only."""
    _write(tmp_path, _TARGETS, _COMPS)
    fs = build_epsf_fit_set(tmp_path, "ALL")
    assert fs.target_ids == {"1", "2", "3", "6"} and fs.n_candidates == 4
    assert fs.comp_ids == {"10", "11", "12", "13", "15"}
    assert fs.target_types == {"1": "EW", "2": "EA:", "3": "RS|BY", "6": "EXOPLANET"}
    assert build_epsf_fit_set(tmp_path, "EXOPLANET").target_ids == {"6"}
    assert build_epsf_fit_set(tmp_path, "").target_ids == fs.target_ids  # empty = ALL


def test_fit_set_selection_by_type_name_wildcard(tmp_path: Path) -> None:
    _write(tmp_path, _TARGETS, _COMPS)
    assert build_epsf_fit_set(tmp_path, "EA").target_ids == {"2"}  # 'EA:' matches EA
    assert build_epsf_fit_set(tmp_path, "e*").target_ids == {"1", "2", "6"}  # EW, EA:, EXOPLANET
    assert build_epsf_fit_set(tmp_path, "BY").target_ids == {"3"}  # token of 'RS|BY'
    # G09-42 (Milan 2026-10-29): the wildcard is for variability types only; a name pattern selects nothing
    assert build_epsf_fit_set(tmp_path, "V1 Cyg; ztf*").target_ids == {"1"}
    fs = build_epsf_fit_set(tmp_path, "EW, V9 Cyg")
    assert fs.target_ids == {"1"} and fs.unmatched_tokens == ("V9 Cyg",)
    none = build_epsf_fit_set(tmp_path, "DSCT")
    assert not none.catalog_ids and "DSCT" in none.empty_reason


def test_faint_star_next_to_a_bright_one_needs_the_joint_fit(tmp_path: Path) -> None:
    """EPSF-BLEND-2026-10: a G+4 mag star 5 px from a bright one. Fitted with the bright neighbour
    (fixed positions) it keeps its own flux; without the neighbour its fit takes the wing light."""
    fp, stars, img = _frame(tmp_path, 2.5, seed=11)
    model, _ = build_frame_model(fp, stars, cutout_size=31, fwhm_px=2.5, delta=None, grid=(1, 1))
    side = img.shape[0]
    yy, xx = np.mgrid[0:side, 0:side]
    rng = np.random.default_rng(2)
    clean = np.zeros_like(img, dtype=float)
    bright, faint = [], []
    for k in range(6):
        bx, by = 54.0 + 36 * k, 18.0
        fb, ff = 4e6, 4e6 * 10 ** (-0.4 * 4.0)
        clean += CircularGaussianPRF(flux=fb, x_0=bx, y_0=by, fwhm=2.5)(xx, yy)
        clean += CircularGaussianPRF(flux=ff, x_0=bx + 5.0, y_0=by, fwhm=2.5)(xx, yy)
        bright.append((bx, by)); faint.append((f"f{k}", bx + 5.0, by, ff))
    data = img + rng.poisson(clean).astype(float)
    st = pd.DataFrame({"catalog_id": [f[0] for f in faint], "x": [f[1] for f in faint], "y": [f[2] for f in faint]})
    truth = np.array([f[3] for f in faint])
    nb = pd.DataFrame({"x": [b[0] for b in bright], "y": [b[1] for b in bright]})
    joint = fit_frame_psf(data, model, st, nb, fwhm_px=2.5, gain=1.0, read_noise_e=5.0)
    alone = fit_frame_psf(data, model, st, None, fwhm_px=2.5, gain=1.0, read_noise_e=5.0)
    r_joint = np.median(joint["psf_flux"] / truth)
    r_alone = np.median(alone["psf_flux"] / truth)
    assert r_joint == pytest.approx(1.0, abs=0.05)
    assert r_alone > 1.3  # the neighbour's wing light goes into the faint star
    assert (joint["psf_group_size"] == 2).all()
    # blend error term: the bright neighbour's light in the fit box widens the quoted error
    assert (joint["psf_nb_light_frac"] > 0.2).all()
    assert (joint["psf_flux_err"] > joint["psf_flux_err_fit"]).all()
    # EPSF-BLEND-ERR2-2026-10: the term is linear in the neighbour light, 0.040 x f_nb x F
    term = np.sqrt(joint["psf_flux_err"] ** 2 - joint["psf_flux_err_fit"] ** 2)
    assert np.allclose(term, 0.040 * joint["psf_nb_light_frac"] * joint["psf_flux"].abs(), rtol=1e-6)


def test_fit_sources_use_gaia_positions_and_catalogue_neighbours(tmp_path: Path) -> None:
    from epsf_frame import frame_fit_sources
    from test_epsf_csv_prepare import _synthetic_field

    _synthetic_field(tmp_path)
    ids = {"2000000000000000000", "1000000000000000005"}
    stars, nb, meta = frame_fit_sources(tmp_path, ids, cutout_size=31, fwhm_px=2.5, merge_fwhm_px=2.5)
    assert set(stars["catalog_id"]) == ids
    assert "3000000000000000000" in set(nb["catalog_id"])  # the bright neighbour 6.5 px away
    assert not set(nb["catalog_id"]) & ids
    s = stars.set_index("catalog_id").loc["2000000000000000000"]
    assert s["x"] == pytest.approx(90.0, abs=0.01) and s["y"] == pytest.approx(90.0, abs=0.01)
    assert meta["neighbour_mag_limit"] == 15.0 and meta["position_source"] == "gaia_dr3_masterstar_wcs"


def test_saturated_core_is_masked_and_fitted_from_the_wings(tmp_path: Path) -> None:
    fp, stars, img = _frame(tmp_path, 2.5, seed=13)
    model, _ = build_frame_model(fp, stars, cutout_size=31, fwhm_px=2.5, delta=None, grid=(1, 1))
    side = img.shape[0]
    yy, xx = np.mgrid[0:side, 0:side]
    f_true = 3.0e6  # peak ~ 3e6 / (1.13 * 6.25) ~ 4.2e5 ADU: far above the 6e4 limit
    x0, y0 = 18.3, 18.6
    data = img + CircularGaussianPRF(flux=f_true, x_0=x0, y_0=y0, fwhm=2.5)(xx, yy)
    data = np.minimum(data, 65535.0)
    st = pd.DataFrame({"catalog_id": ["s"], "x": [x0], "y": [y0]})
    masked = fit_frame_psf(data, model, st, None, fwhm_px=2.5, gain=1.0, read_noise_e=5.0, sat_limit_adu=60000.0)
    plain = fit_frame_psf(data, model, st, None, fwhm_px=2.5, gain=1.0, read_noise_e=5.0)
    assert bool(masked["psf_sat_masked"].iloc[0]) and not bool(plain["psf_sat_masked"].iloc[0])
    assert abs(masked["psf_flux"].iloc[0] / f_true - 1.0) < 0.15
    assert plain["psf_flux"].iloc[0] / f_true < 0.5  # the clipped core drags an unmasked fit far down


def test_unresolved_fainter_companion_is_merged_brighter_one_is_fitted(tmp_path: Path) -> None:
    """EPSF-MERGE-2026-10: a fainter Gaia neighbour closer than 0.5 FWHM is not fitted (its light stays
    with the fit star, Gaia flux ratio recorded); a brighter one that close stays in the joint fit."""
    from astropy.wcs import WCS

    from epsf_frame import frame_fit_sources
    from test_epsf_csv_prepare import _synthetic_field

    _synthetic_field(tmp_path)
    cone_p = tmp_path / "field_catalog_cone.csv"
    cone = pd.read_csv(cone_p, dtype={"catalog_id": str})
    w = WCS(fits.getheader(tmp_path / "MASTERSTAR.fits"))
    ids = ["2000000000000000000", "1000000000000000005"]
    sx, sy = w.all_world2pix(*[cone.set_index("catalog_id").loc[ids, c].to_numpy(float) for c in ("ra_deg", "dec_deg")], 0)
    mag = cone.set_index("catalog_id").loc[ids, "mag"].to_numpy(float)
    ra, dec = w.all_pix2world([sx[0] + 1.0, sx[1] + 1.0], [sy[0], sy[1]], 0)
    extra = cone.iloc[[0, 0]].copy()
    extra["catalog_id"] = ["4000000000000000001", "4000000000000000002"]
    extra["ra_deg"], extra["dec_deg"] = ra, dec
    extra["mag"] = [mag[0] + 2.0, mag[1] - 1.0]  # fainter companion (within the G 15 limit) / brighter one
    pd.concat([cone, extra], ignore_index=True).to_csv(cone_p, index=False)
    stars, nb, meta = frame_fit_sources(tmp_path, set(ids), cutout_size=31, fwhm_px=1.0, merge_fwhm_px=2.5)  # merge uses the night FWHM
    assert "4000000000000000001" not in set(nb["catalog_id"])  # merged into its fit star
    assert "4000000000000000002" in set(nb["catalog_id"])  # brighter: fitted jointly
    assert "merged_light_frac" not in stars.columns  # no reader (Milan 2026-10-29); the pair is in the manifest
    assert meta["merge_radius_px"] == pytest.approx(1.25)
    assert [m["neighbour_id"] for m in meta["merged_neighbours"]] == ["4000000000000000001"]
    assert meta["merged_neighbours"][0]["dmag"] == pytest.approx(2.0, abs=1e-3)


def test_aligned_footprint_excludes_the_constant_fill(tmp_path: Path) -> None:
    """EPSF-FOOTPRINT-2026-10: the aligner fills uncovered reference pixels with a constant (Brno 533
    r_0199: x < 48). A build star there has no light (its ePSF build diverged); it is left out of the
    build, and a fit star whose box or annulus reaches the fill is marked psf_edge_masked."""
    from epsf_frame import _covered_stars, build_frame_model
    from frame_footprint import aligned_footprint

    fp, stars, img = _frame(tmp_path, 2.5, seed=21)
    shift = 40.0  # frame pixel (0, 0) lands on reference x = 40: reference x < 40 is fill
    data = img.copy()
    data[:, : int(shift)] = float(np.median(img))
    hdr = fits.getheader(fp)
    hdr["VY_AM2"] = shift
    fits.PrimaryHDU(data, header=hdr).writeto(fp, overwrite=True)
    cov = aligned_footprint(hdr, data.shape)
    assert not cov[:, : int(shift) + 1].any() and cov[5:-5, int(shift) + 2 : -5].all()
    kept = _covered_stars(stars, cov, 31)
    assert len(kept) < len(stars) and (kept["x"] - 15 > shift).all()
    model, rec = build_frame_model(fp, stars, cutout_size=31, fwhm_px=2.5, delta=None, grid=(1, 1))
    assert rec["n_stars_used"] <= len(kept)
    st = stars.iloc[[0, len(stars) - 1]].reset_index(drop=True)  # first column: in the fill; last: inside
    r = fit_frame_psf(data, model, st, None, fwhm_px=2.5, gain=1.0, read_noise_e=5.0, covered=cov)
    assert list(r["psf_edge_masked"]) == [True, False]
    # box on data, annulus (10-15 px) only grazing the fill: sky from the covered part, not flagged
    graze = pd.DataFrame({"catalog_id": ["g"], "x": [shift + 11.0], "y": [float(data.shape[0]) / 2]})
    assert not bool(fit_frame_psf(data, model, graze, None, fwhm_px=2.5, gain=1.0, read_noise_e=5.0, covered=cov)["psf_edge_masked"].iloc[0])
    del hdr["VY_AM0"]
    with pytest.raises(ValueError, match="frame->reference transform"):
        aligned_footprint(hdr, data.shape)
