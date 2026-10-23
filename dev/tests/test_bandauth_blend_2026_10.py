"""BANDAUTH-2026-10 / BLEND-2026-10: band authority and neighbour light in comp apertures."""

from __future__ import annotations

import math

import numpy as np
import pytest

from comp_blend import contamination_fractions
from gaia_bands import filter_band, filter_band_for_obs_group, zp_mag_from_gaia


@pytest.mark.parametrize(
    ("tok", "zp", "code", "trans"),
    [
        ("g", "g", "SG", True),
        ("r", "r", "SR", True),
        ("V", "V", "V", True),
        ("R", "R", "R", True),
        ("NoFilter", "V", "CV", False),
        ("L", "V", "CV", False),
        ("TG", "V", "TG", False),
        ("I", "G", "", False),
    ],
)
def test_filter_band_authority(tok: str, zp: str, code: str, trans: bool) -> None:
    fb = filter_band(tok)
    assert (fb.zp_band, fb.aavso_code, fb.transformable) == (zp, code, trans)


def test_obs_group_osc_channel() -> None:
    assert filter_band_for_obs_group("NoFilter_60_2_G").aavso_code == "TG"
    assert filter_band_for_obs_group("g_120_4").aavso_code == "SG"


def test_zp_mag_outside_relation_range_is_nan() -> None:
    fb = filter_band("g")
    assert math.isnan(zp_mag_from_gaia(15.0, 2.5, fb))  # g valid only to BP-RP 2.0
    assert math.isfinite(zp_mag_from_gaia(15.0, 1.0, fb))


def test_contamination_near_neighbour_counts_far_one_cancels() -> None:
    fwhm, beta, r_ap = 5.0, 2.7, 10.0
    ann = (2.7 * fwhm, 5.2 * fwhm)
    star = np.array([[100.0, 100.0]])
    f_star = np.array([1.0])
    # equal-brightness neighbour 2 px away: ~ +100 % light
    near = contamination_fractions(
        star, f_star, np.array([[102.0, 100.0]]), np.array([1.0]),
        r_ap_px=r_ap, fwhm_px=fwhm, beta=beta, annulus_px=ann,
    )[0]
    assert near > 0.9
    # same neighbour far outside the annulus: its flat wing is removed with the sky
    far = contamination_fractions(
        star, f_star, np.array([[100.0 + ann[1] + 3 * fwhm, 100.0]]), np.array([1.0]),
        r_ap_px=r_ap, fwhm_px=fwhm, beta=beta, annulus_px=ann,
    )[0]
    assert abs(far) < 0.003


def test_contamination_skips_self_by_id_and_uses_g_when_band_missing() -> None:
    star = np.array([[50.0, 50.0]])
    c = contamination_fractions(
        star, np.array([np.nan]), np.array([[50.3, 50.0], [53.0, 50.0]]), np.array([np.nan, np.nan]),
        r_ap_px=8.0, fwhm_px=4.0, beta=2.7, star_ids=["A"], neighbour_ids=["A", "B"],
        star_flux_g=np.array([1.0]), neighbour_flux_g=np.array([1.0, 0.1]),
    )[0]
    assert 0.05 < c < 0.11  # only B (10 % flux, mostly inside the aperture)


# --- COMP-BAND-2026-10: no band gate on comps; zero point from the valid subset -----------
def test_ensemble_shape_from_all_comps_level_from_valid_band_subset() -> None:
    """Comps without a valid band magnitude shape the LC; the level comes from the others."""
    import numpy as np

    import photometry_core  # noqa: F401  (import order of the photometry package)
    from photometry_lightcurve import ensemble_normalize

    rng = np.random.default_rng(5)
    nf = 80
    common = 0.3 * np.sin(np.linspace(0, 6, nf))  # clouds / airmass, common to all stars
    true_t = 12.0 + 0.2 * (np.arange(nf) > 40)  # target steps by 0.2 mag
    inst_zp = -24.0
    cat = {"a": 11.5, "b": 12.3, "c": 13.0, "d": 11.9}
    comps = {k: v + inst_zp + common + rng.normal(0, 0.003, nf) for k, v in cat.items()}
    t = true_t + inst_zp + common + rng.normal(0, 0.003, nf)
    q = {k: {"quality": "good"} for k in cat}
    full, _, _ = ensemble_normalize(t, comps, cat, q)
    part_cat = {"a": 11.5, "b": float("nan"), "c": 13.0, "d": float("nan")}
    part, _, _ = ensemble_normalize(t, comps, part_cat, q)
    # same shape (frame zero point uses all four comps)
    assert np.nanstd((part - np.nanmedian(part)) - (full - np.nanmedian(full))) < 1e-9
    # level from the valid comps a and c only, still the true magnitude
    assert abs(np.nanmedian(part - true_t)) < 0.005
    # no valid comp at all -> not calibrated
    none, _, _ = ensemble_normalize(t, comps, {k: float("nan") for k in cat}, q)
    assert not np.isfinite(none).any()


def test_global_pool_keeps_comps_outside_band_relation(monkeypatch) -> None:
    """A red comp (BP-RP > 2.0, no Gaia -> g relation) must stay in the pool."""
    import inspect

    import photometry_core  # noqa: F401  (import order of the photometry package)
    import photometry_comp as pc

    src = inspect.getsource(pc.build_global_comp_pool)
    assert "COMP_ZP_BAND_VALID" not in src
    assert "pool = pool.loc[_keep_b]" not in src


# --- BLEND-K-2026-10: aperture only where all comps keep <= 1 % neighbour light ---------
def test_allowed_f_from_comp_blend() -> None:
    import pandas as pd

    import photometry_core  # noqa: F401  (import order of the photometry package)
    from aperture_pertarget import allowed_f_from_comp_blend, blend_k_column

    grid = [1.0, 1.25, 1.5, 1.75, 2.0]
    comps = pd.DataFrame({
        blend_k_column(1.0): [0.001, 0.004, float("nan")],
        blend_k_column(1.25): [0.002, 0.008, 0.0],
        blend_k_column(1.5): [0.003, 0.012, 0.0],     # comp 2 too much light from k = 1.5
        blend_k_column(1.75): [0.004, 0.015, 0.0],
        blend_k_column(2.0): [-0.02, 0.02, 0.0],      # negative (sky pushed up) counts as well
    })
    assert blend_k_column(1.25) == "blend_k125"
    assert allowed_f_from_comp_blend(comps, grid, 0.01) == [1.0, 1.25]
    assert allowed_f_from_comp_blend(pd.DataFrame({"x": [1]}), grid, 0.01) is None  # old drafts


def test_choose_f_respects_allowed_apertures() -> None:
    import numpy as np

    import photometry_core  # noqa: F401
    from aperture_pertarget import ApertureGridNight, choose_f_for_target

    rng = np.random.default_rng(1)
    grid = [1.0, 1.25, 1.5, 1.75, 2.0]
    night = ApertureGridNight(f_grid=grid, fwhm_night_px=4.0)
    frames = [f"f{i:03d}" for i in range(60)]
    # comp scatter is smallest at k = 2.0 -> unrestricted choice 2.0
    noise = {1.0: 0.02, 1.25: 0.015, 1.5: 0.01, 1.75: 0.006, 2.0: 0.003}
    for st in frames:
        night.frames[st] = {}
        night.sky_pp[st] = {}
        for cid in ("T", "c1", "c2", "c3"):
            night.frames[st][cid] = {k: 1e5 * (1 + rng.normal(0, noise[k])) for k in grid}
            night.sky_pp[st][cid] = {k: 100.0 for k in grid}
    free = choose_f_for_target(night, target_cid="T", comp_ids=["c1", "c2", "c3"], frame_order=frames)
    assert free.f_star == 2.0 and free.f_allowed is None
    lim = choose_f_for_target(night, target_cid="T", comp_ids=["c1", "c2", "c3"], frame_order=frames,
                              allowed_f=[1.0, 1.25])
    assert lim.f_star == 1.25 and lim.f_allowed == [1.0, 1.25]


# --- ZP-ERR-2026-10: uncertainty of the absolute level ---------------------------------
def test_zero_point_uncertainty() -> None:
    import photometry_core  # noqa: F401
    from aperture_photometry_target import zero_point_uncertainty

    w = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 5.0}
    cat = {"a": 12.0, "b": 12.5, "c": 13.0, "d": 13.3, "e": float("nan")}
    lev = {"a": -12.0, "b": -11.5, "c": -10.95, "d": -10.7, "e": -12.0}  # c is 0.05 mag off
    zp = zero_point_uncertainty(w, cat, lev, "g")
    assert zp["n_comp"] == 4 and abs(zp["n_eff"] - 4.0) < 1e-9          # e (no band mag) not used
    assert abs(zp["sigma_rel_mag"] - 0.0745) < 1e-9                     # Gaia DR3 Table 5.8, g
    assert abs(zp["sigma_zp_model_mag"] - 0.0745 / 2.0) < 1e-6
    # d = cat - M = {24, 24, 23.95, 24}: unbiased std 0.025 -> / sqrt(4)
    assert abs(zp["sigma_zp_data_mag"] - 0.025 / 2.0) < 1e-4
    one = zero_point_uncertainty({"a": 1.0}, {"a": 12.0}, {"a": -12.0}, "g")
    assert one["n_comp"] == 1 and one["sigma_zp_data_mag"] is None and abs(one["sigma_zp_model_mag"] - 0.0745) < 1e-9
    assert zero_point_uncertainty(w, cat, lev, "NONE")["n_comp"] == 0


def test_zero_point_data_sigma_unbiased_with_unequal_weights() -> None:
    """Monte Carlo: sigma_zp_data matches the true error of the weighted mean."""
    import numpy as np

    import photometry_core  # noqa: F401
    from aperture_photometry_target import zero_point_uncertainty

    rng = np.random.default_rng(7)
    w = {f"c{i}": v for i, v in enumerate([10.0, 5.0, 2.0, 1.0, 1.0, 0.5, 0.5])}
    sw = np.array(list(w.values()))
    true_err = 0.0745 * np.sqrt(np.sum(sw**2)) / sw.sum()
    est = []
    for _ in range(3000):
        cat = {k: 12.0 + rng.normal(0, 0.0745) for k in w}
        est.append(zero_point_uncertainty(w, cat, {k: -12.0 for k in w}, "g")["sigma_zp_data_mag"])
    ratio = float(np.sqrt(np.mean(np.square(est)))) / true_err
    assert 0.95 < ratio < 1.05, ratio


def test_zero_point_levels_from_ensemble_not_biased_by_missing_frames() -> None:
    """Perfect catalogue, night trend, one comp missing late frames -> sigma_data ~ 0."""
    import numpy as np

    import photometry_core  # noqa: F401
    from aperture_photometry_target import zero_point_uncertainty
    from photometry_lightcurve import ensemble_normalize

    rng = np.random.default_rng(3)
    nf = 100
    trend = np.linspace(0, 0.3, nf)
    cat = {f"c{i}": 11.5 + 0.4 * i for i in range(5)}
    comps = {k: v - 24.0 + trend + rng.normal(0, 0.001, nf) for k, v in cat.items()}
    comps["c4"][60:] = np.nan
    w: dict = {}
    lev: dict = {}
    ensemble_normalize(16.0 - 24.0 + trend, comps, cat, {k: {} for k in cat}, weights_out=w, levels_out=lev)
    zp = zero_point_uncertainty(w, cat, lev, "g")
    assert zp["sigma_zp_data_mag"] < 0.002, zp


def test_report_zero_point_note(tmp_path) -> None:
    import json

    import pandas as pd

    from target_page import TargetPages

    (tmp_path / "photometry").mkdir()
    lc = tmp_path / "photometry" / "lightcurves"
    lc.mkdir()
    b = TargetPages(tmp_path, summary_df=pd.DataFrame(), at_df=pd.DataFrame(), comp_df=pd.DataFrame(), cfg=None)
    tmp_path = lc
    (tmp_path / "comp_quality_1.json").write_text(json.dumps({"zero_point": {
        "band": "g", "n_comp": 3, "n_eff": 2.6, "sigma_rel_mag": 0.0745,
        "sigma_zp_model_mag": 0.0462, "sigma_zp_data_mag": 0.012}}), encoding="utf-8")
    (tmp_path / "comp_quality_2.json").write_text(json.dumps({"zero_point": {"band": "NONE", "n_comp": 0}}), encoding="utf-8")
    n1 = b.zero_point_note("1")
    assert "3 comparison stars" in n1 and "46 mmag" in n1 and "12 mmag" in n1
    assert "not calibrated" in b.zero_point_note("2")
    assert b.zero_point_note("missing") == ""


# --- BLEND-FWHM-2026-10 / check star -------------------------------------------------
def test_night_median_fwhm_from_group_frames(tmp_path) -> None:
    import pandas as pd

    import photometry_core  # noqa: F401
    from photometry_comp import night_median_fwhm_px

    paths = []
    for i, fw in enumerate([4.2, 5.0, 5.3, 7.9, 8.1]):
        p = tmp_path / f"proc_{i}.csv"
        pd.DataFrame({"fwhm_px_for_aperture": [fw, fw, fw]}).to_csv(p, index=False)
        paths.append(p)
    fw, src = night_median_fwhm_px(paths, 4.6)
    assert fw == 5.3 and "5 frames" in src
    assert night_median_fwhm_px([], 4.6) == (4.6, "fallback (no per-frame FWHM)")


def test_check_candidates_follow_neighbour_light_rule() -> None:
    import pandas as pd

    import photometry_core  # noqa: F401
    from check_star_kmag import drop_blended_check_candidates

    pool = pd.DataFrame({"catalog_id": ["1", "2", "3", "4"],
                         "blend_k100": [0.002, 0.004, float("nan"), 0.001],
                         "blend_k150": [0.003, 0.02, float("nan"), -0.015]})
    assert list(drop_blended_check_candidates(pool, 1.0)["catalog_id"]) == ["1", "2", "3", "4"]
    assert list(drop_blended_check_candidates(pool, 1.5)["catalog_id"]) == ["1", "3"]
    assert len(drop_blended_check_candidates(pool, 2.0)) == 4  # no column -> nothing dropped
