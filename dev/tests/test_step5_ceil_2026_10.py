"""COMP-CEIL-NOISEMODEL-2026-10 (Milan 2026-10-30): reference of the comparison RMS ceiling.

sigma_ref^2 = (a e_photon)^2 + e_scint^2 + floor^2 per pool star at the field k; a / floor fitted on the
constant pool stars with the one robust log fit of the field variability search (stats_core
``fit_scatter_noise_model``, NOISE-FIT-ONE-2026-10). Ceiling = min(abs cap, 5 x sigma_ref).
AP-BRIGHT-K diagnostic: every k within one SE of the minimum (``k_within_se``) is recorded.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline  # noqa: F401  app import order (import cycles, T13)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ccd_noise import ccd_variance_adu2  # noqa: E402
from comp_rms_loo import (  # noqa: E402
    COMP_CEIL_FIT_MIN_STARS,
    COMP_RMS_LOO_PHOTON_K_DEFAULT,
    COMP_SIGMA_REF_COL,
    comp_ceiling_reference,
    choose_field_aperture_k,
    k_within_one_se,
    ks_within_one_se,
    loo_mag_rms_maps,
    photon_mag_err_map,
    scint_mag_err_map,
)
from photometry_comp import _select_comps_by_rms_then_color  # noqa: E402
from photometry_defs import flux_k_column, flux_k_radius_px  # noqa: E402
from sigma_budget import RigScintillationParams  # noqa: E402
from sigma_floor_core import frame_scintillation_mag, scintillation_mag_per_epoch  # noqa: E402
from stats_core import fit_scatter_noise_model  # noqa: E402

GAIN, RN = 1.3, 9.0
SKY, N_SKY = 300.0, 600
K = 0.75
FLOOR = 0.006  # true night floor [mag] (aperture loss etc.), common to all stars
RIG = RigScintillationParams(None, "test", 0.2, 300.0, 30.0, 1.5, [])  # D 0.2 m: scintillation ~ 2-4 mmag


def _enclosed(k: float) -> float:
    return 1.0 - math.exp(-4.0 * math.log(2.0) * k * k)


def _field_with_floor(n_stars=300, n_frames=80, seed=11, var_idx=5, var_factor=10.0):
    """Proc frames at one grid k: per star and frame CCD noise + scintillation (frame airmass) + a floor.
    Star ``var_idx`` (bright) gets ``var_factor`` x its total noise: an uncatalogued variable."""
    rng = np.random.default_rng(seed)
    ids = [f"{7100000000000000000 + i}" for i in range(n_stars)]
    total = np.sort(10 ** rng.uniform(4.0, 7.0, n_stars))[::-1]  # bright first
    airmass = np.linspace(1.05, 2.2, n_frames)
    col = flux_k_column(K)
    cache, paths = {}, []
    for f in range(n_frames):
        fw = rng.uniform(3.0, 5.0)
        F = total * _enclosed(K)
        area = math.pi * flux_k_radius_px(K, fw) ** 2
        ph = 2.5 / math.log(10) * np.sqrt(ccd_variance_adu2(F, SKY, area, N_SKY, gain=GAIN, read_noise=RN)) / F
        sc = frame_scintillation_mag(RIG, float(airmass[f]))
        sig = np.sqrt(ph**2 + sc**2 + FLOOR**2)
        sig[var_idx] *= var_factor
        dm = rng.normal(0.0, 1.0, n_stars) * sig
        d = {"catalog_id": ids, "fwhm_px_for_aperture": fw, "sky_adu_per_px_annulus": SKY, "sky_n_px": N_SKY,
             "airmass": float(airmass[f]), col: F * 10 ** (-0.4 * dm)}
        p = Path(f"proc_ceil_{f:03d}.csv")
        paths.append(p)
        cache[str(p)] = pd.DataFrame(d)
    return ids, paths, cache, airmass


def _maps(ids, paths, cache):
    col = flux_k_column(K)
    loo = loo_mag_rms_maps(set(ids), paths, cache, [col], strict_flux_col=True)[col]
    ph = photon_mag_err_map(set(ids), paths, cache, col, k=K, gain=GAIN, read_noise=RN)
    sc, sc_info = scint_mag_err_map(set(ids), paths, cache, col, rig=RIG)
    rms = np.array([loo[c] for c in ids])
    e_ph = np.array([ph[c] for c in ids])
    e_sc = np.array([sc[c] for c in ids])
    return rms, e_ph, e_sc, sc_info


# ------------------------------------------------------------------------------------------------
# the ceiling reference
# ------------------------------------------------------------------------------------------------
def test_scintillation_map_is_the_median_of_the_light_curve_per_frame_value():
    ids, paths, cache, airmass = _field_with_floor(n_stars=40, n_frames=21)
    _r, _p, e_sc, info = _maps(ids, paths, cache)
    want = float(np.median([scintillation_mag_per_epoch(telescope_diameter_m=0.2, airmass=float(a),
                                                        exposure_s=30.0, altitude_m=300.0, c_y=1.5)
                            for a in airmass]))
    assert e_sc[0] == want and np.all(e_sc == want)
    assert info["n_frames"] == 21 and info["n_frames_no_airmass"] == 0


def test_floor_field_all_constant_admitted_variable_rejected():
    """Bright stars carry a floor the CCD equation does not know. The photon-only ceiling (01950ab) rejects
    many bright constant stars; the fitted reference admits every constant star and still rejects the
    injected 10x-scatter variable."""
    var_idx = 5
    ids, paths, cache, _am = _field_with_floor(var_idx=var_idx)
    rms, e_ph, e_sc, _ = _maps(ids, paths, cache)
    constant = np.ones(len(ids), dtype=bool)
    constant[var_idx] = False  # the fit set excludes catalogued variables; this one is "uncatalogued" below
    sref, info = comp_ceiling_reference(rms, e_ph, e_sc, constant)
    assert info["status"] == "ok" and info["n_stars"] == len(ids) - 1
    assert info["a"] == pytest.approx(1.0, abs=0.1)
    assert info["floor_mag"] == pytest.approx(FLOOR, rel=0.15)
    k5 = COMP_RMS_LOO_PHOTON_K_DEFAULT
    old_pass = rms <= np.minimum(0.1, k5 * e_ph)
    new_pass = rms <= np.minimum(0.1, k5 * sref)
    cst = np.arange(len(ids)) != var_idx
    assert (~old_pass[cst]).sum() >= 10  # photon-only: bright constant stars rejected (the defect)
    assert new_pass[cst].all()
    assert not new_pass[var_idx]
    # through the real selector (all candidates allowed, one colour)
    cand = pd.DataFrame({"catalog_id": ids, "bp_rp": 1.0, "comp_rms": rms, "_dist_deg": np.linspace(0.05, 0.4, len(ids)),
                         "_nn_dist_fwhm": 10.0, COMP_SIGMA_REF_COL: sref})
    out = _select_comps_by_rms_then_color(cand, 1.0, 3, len(ids), 0.5)
    assert set(out["catalog_id"]) == {c for i, c in enumerate(ids) if i != var_idx}
    # the variable also when it is in the fit set (uncatalogued): the robust fit is not pulled by it
    sref2, info2 = comp_ceiling_reference(rms, e_ph, e_sc, np.ones(len(ids), dtype=bool))
    assert info2["status"] == "ok" and rms[var_idx] > k5 * sref2[var_idx]
    assert (rms[cst] <= k5 * sref2[cst]).all()


def test_ceiling_reference_too_few_stars_warns_and_records(caplog):
    ids, paths, cache, _am = _field_with_floor(n_stars=40, n_frames=30)
    rms, e_ph, e_sc, _ = _maps(ids, paths, cache)
    constant = np.zeros(len(ids), dtype=bool)
    constant[: COMP_CEIL_FIT_MIN_STARS - 1] = True
    with caplog.at_level(logging.WARNING):
        sref, info = comp_ceiling_reference(rms, e_ph, e_sc, constant)
    assert info["status"].startswith("too_few_stars") and str(COMP_CEIL_FIT_MIN_STARS - 1) in info["status"]
    assert info["a"] == 1.0 and info["floor_mag"] == 0.0 and info["n_stars"] == COMP_CEIL_FIT_MIN_STARS - 1
    assert np.array_equal(sref, np.sqrt(e_ph**2 + e_sc**2))
    assert any("too_few_stars" in r.getMessage() for r in caplog.records)
    # stars without a photon error / comp_rms never enter the fit
    e_ph2 = e_ph.copy()
    e_ph2[:] = np.nan
    _s, info3 = comp_ceiling_reference(rms, e_ph2, e_sc, np.ones(len(ids), dtype=bool))
    assert info3["n_stars"] == 0 and info3["status"].startswith("too_few_stars")


def test_pool_carries_sigma_ref_and_meta(tmp_path):
    from config import AppConfig
    from photometry_comp import build_global_comp_pool
    from test_comp_determinism_synthetic import _TARGET_ID, _build_synthetic_dataset

    _target, ms, paths, cache = _build_synthetic_dataset()
    cfg = AppConfig()
    cfg.gs11_dilution_enabled = False
    args = (ms, paths, cache, frozenset({_TARGET_ID}), None, 800, 600, 0, cfg)
    with pytest.raises(TypeError):
        build_global_comp_pool(*args, min_frames_frac=0.3, fwhm_px=3.7, gain=1.0, read_noise=10.0)  # no rig
    pool = build_global_comp_pool(*args, min_frames_frac=0.3, fwhm_px=3.7, gain=1.0, read_noise=10.0,
                                  scint_params=RIG, photometry_dir_for_meta=tmp_path)
    assert pool[COMP_SIGMA_REF_COL].notna().all()
    meta = json.loads((tmp_path / "pipeline_meta.json").read_text(encoding="utf-8"))
    rule = {r["rule_id"]: r for r in meta["comp_pool_admission"]["rules"]}["COMP_CEIL_NOISE_MODEL"]
    # 29 stars: fewer than the fit needs -> unfitted model, said so in the meta
    assert rule["status"].startswith("too_few_stars") and rule["a"] == 1.0 and rule["floor_mag"] == 0.0
    assert rule["threshold_value"] == COMP_RMS_LOO_PHOTON_K_DEFAULT
    assert rule["scintillation"]["telescope_diameter_m"] == 0.2
    # the synthetic frames carry no airmass: scintillation 0 there, counted
    assert rule["scintillation"]["n_frames_no_airmass"] == rule["scintillation"]["n_frames"] > 0


# ------------------------------------------------------------------------------------------------
# NOISE-FIT-ONE: the variability_field fit is bit-identical after the refactor
# ------------------------------------------------------------------------------------------------
def _old_variability_fit(fitset: pd.DataFrame) -> tuple[float, float]:
    """variability_field._var_frame_zero_point before NOISE-FIT-ONE-2026-10 (verbatim)."""
    from scipy import optimize

    def _res(p: np.ndarray) -> np.ndarray:
        a, fl = p
        return np.log(fitset["s"]) - 0.5 * np.log((a * fitset["e"]) ** 2 + fl**2)

    fit = optimize.least_squares(_res, [1.2, 0.005], loss="soft_l1", f_scale=0.1, bounds=([0.2, 0.0], [10.0, 0.2]))
    return float(fit.x[0]), float(fit.x[1])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_variability_field_fit_bit_identical(seed):
    rng = np.random.default_rng(seed)
    n = 1500
    e = 10 ** rng.uniform(-3.3, -1.0, n)
    s = np.sqrt((1.15 * e) ** 2 + 0.004**2) * np.exp(rng.normal(0.0, 0.12, n))
    s[rng.random(n) < 0.03] *= 6.0  # variables in the fit set
    idx = [f"{6000000000000000000 + i}" for i in range(n)]
    fitset = pd.DataFrame({"s": s, "e": e}, index=idx)
    a0, f0 = _old_variability_fit(fitset)
    a1, f1, ok = fit_scatter_noise_model(fitset["s"].to_numpy(dtype=float), fitset["e"].to_numpy(dtype=float))
    assert ok
    assert np.float64(a0).tobytes() == np.float64(a1).tobytes()
    assert np.float64(f0).tobytes() == np.float64(f1).tobytes()


def test_variability_field_calls_the_shared_fit():
    import inspect

    import variability_field

    src = inspect.getsource(variability_field._var_frame_zero_point)
    assert "fit_scatter_noise_model(" in src and "least_squares" not in src


# ------------------------------------------------------------------------------------------------
# AP-BRIGHT-K diagnostic: k_within_se
# ------------------------------------------------------------------------------------------------
def test_ks_within_one_se_matches_the_rule():
    rng = np.random.default_rng(4)
    ks = [0.6, 0.75, 0.875, 1.0, 1.25, 1.5]
    r = 1.0 + np.abs(rng.normal(0.0, 0.05, (80, len(ks)))) + np.array([0.1, 0.02, 0.0, 0.01, 0.03, 0.2])
    k, curve, se = k_within_one_se(ks, r, np.random.default_rng(0))
    kw = ks_within_one_se(ks, curve, se)
    assert max(kw) == k and len(kw) >= 1
    assert kw == [kk for kk, v in zip(ks, curve) if v <= float(np.min(curve)) + se]


def test_brightness_groups_record_k_within_se(tmp_path):
    from aperture_pertarget import field_k_by_brightness, load_field_k
    from photometry_defs import DEFAULT_APERTURE_F_GRID
    from test_step5_perf2_2026_10 import _synthetic_proc_dir

    paths = sorted(_synthetic_proc_dir(tmp_path))
    bk = field_k_by_brightness(paths, DEFAULT_APERTURE_F_GRID)
    assert bk is not None and len(bk.k_within_se) == len(bk.k)
    d = bk.to_dict()
    for g, kk in zip(d["groups"], bk.k):
        assert g["k_within_se"] and max(g["k_within_se"]) == kk
    (tmp_path / "aperture_per_target.json").write_text(json.dumps({"brightness_k": {"brightness_k": d}}))
    back, status = load_field_k(tmp_path)
    assert status == "brightness_k" and back.to_dict() == d


def test_field_k_info_records_k_within_se():
    from test_step5_comp_2026_10 import _photon_limited_frames

    ids, _t, paths, cache = _photon_limited_frames(ks=(0.6, 0.75, 0.875, 1.0), n_stars=60, n_frames=40)
    k, _loo, _scores, info = choose_field_aperture_k(set(ids), paths, cache)
    assert k is not None and info["k_within_se"] and max(info["k_within_se"]) == k
