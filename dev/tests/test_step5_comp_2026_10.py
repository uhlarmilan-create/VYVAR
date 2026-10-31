"""COMP-CEIL-FIELDK-2026-10 (Milan 2026-10-30): comparison-star RMS ceiling and field aperture k.

A. The photon reference of the comp_rms ceiling comes from the comp_rms aperture (field k) and frames
   (CCD equation per frame, median over the night), not from the MASTERSTAR S/N.
B. The field k takes the AP-BRIGHT-K rule (largest k within one bootstrap SE of the minimum median RMS
   ratio, ``comp_rms_loo.k_within_one_se``) instead of a fixed 1 % tolerance.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pipeline  # noqa: F401  app import order (import cycles, T13)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ccd_noise import ccd_rel_error, ccd_variance_adu2  # noqa: E402
from comp_rms_loo import (  # noqa: E402
    COMP_PHOTON_MAG_COL,
    COMP_SIGMA_REF_COL,
    LN10_OVER_2P5,
    choose_field_aperture_k,
    loo_mag_rms_maps,
    photon_mag_err_map,
)
from photometry_comp import _select_comps_by_rms_then_color  # noqa: E402
from photometry_defs import flux_k_column, flux_k_radius_px  # noqa: E402

GAIN, RN = 1.3, 9.0
SKY, N_SKY = 300.0, 600


def _enclosed(k: float) -> float:
    """Gaussian PSF: light inside r = k x FWHM."""
    return 1.0 - math.exp(-4.0 * math.log(2.0) * k * k)


def _photon_limited_frames(ks=(0.75, 0.875), n_stars=40, n_frames=60, seed=3):
    """Proc frames whose scatter is exactly the CCD-equation noise of each grid flux (no systematics)."""
    rng = np.random.default_rng(seed)
    ids = [f"{7000000000000000000 + i}" for i in range(n_stars)]
    total = 10 ** rng.uniform(4.0, 6.5, n_stars)  # bright (photon-limited) to faint (sky-limited)
    cache, paths = {}, []
    for f in range(n_frames):
        fw = rng.uniform(3.0, 5.0)
        d = {"catalog_id": ids, "fwhm_px_for_aperture": fw, "sky_adu_per_px_annulus": SKY, "sky_n_px": N_SKY}
        for k in ks:
            F = total * _enclosed(k)
            area = math.pi * flux_k_radius_px(k, fw) ** 2
            sig = np.sqrt(ccd_variance_adu2(F, SKY, area, N_SKY, gain=GAIN, read_noise=RN))
            d[flux_k_column(k)] = F + rng.normal(0.0, 1.0, n_stars) * sig
        p = Path(f"proc_photon_{f:03d}.csv")
        paths.append(p)
        cache[str(p)] = pd.DataFrame(d)
    return ids, total, paths, cache


# ------------------------------------------------------------------------------------------------
# A: the ceiling follows k
# ------------------------------------------------------------------------------------------------
def test_photon_reference_is_the_ccd_error_at_the_field_k():
    ids, _total, paths, cache = _photon_limited_frames()
    for k in (0.75, 0.875):
        col = flux_k_column(k)
        ph = photon_mag_err_map(set(ids), paths, cache, col, k=k, gain=GAIN, read_noise=RN)
        cid = ids[7]
        per_frame = []
        for p in paths:
            df = cache[str(p)]
            row = df[df["catalog_id"] == cid].iloc[0]
            r = flux_k_radius_px(k, float(row["fwhm_px_for_aperture"]))
            per_frame.append(ccd_rel_error(float(row[col]), SKY, math.pi * r * r, N_SKY, gain=GAIN, read_noise=RN))
        assert ph[cid] == pytest.approx(LN10_OVER_2P5 * float(np.median(per_frame)), rel=1e-12)
    # a smaller aperture (less light, sky-limited stars) gives a larger reference for faint stars
    p75 = photon_mag_err_map(set(ids), paths, cache, flux_k_column(0.75), k=0.75, gain=GAIN, read_noise=RN)
    p88 = photon_mag_err_map(set(ids), paths, cache, flux_k_column(0.875), k=0.875, gain=GAIN, read_noise=RN)
    assert p75 != p88


def test_photon_limited_comp_admitted_at_both_k_independent_of_masterstar_snr():
    """A comp whose night scatter is its own CCD noise passes the ceiling at every field k. The old gate
    used the MASTERSTAR S/N (snr_ap_pixscaled): with a MASTERSTAR S/N far above the frame S/N it rejected
    such comps - the new gate does not read it."""
    ids, total, paths, cache = _photon_limited_frames()
    for k in (0.75, 0.875):
        col = flux_k_column(k)
        loo = loo_mag_rms_maps(set(ids), paths, cache, [col], strict_flux_col=True)[col]
        ph = photon_mag_err_map(set(ids), paths, cache, col, k=k, gain=GAIN, read_noise=RN)
        assert all(loo[c] <= 5.0 * ph[c] for c in ids)  # photon-limited: LOO ~ photon error
        cand = pd.DataFrame({
            "catalog_id": ids,
            "bp_rp": 1.0,
            "comp_rms": [loo[c] for c in ids],
            "_dist_deg": np.linspace(0.05, 0.4, len(ids)),
            "_nn_dist_fwhm": 10.0,
            COMP_SIGMA_REF_COL: [ph[c] for c in ids],
            # a MASTERSTAR S/N 10x the frame S/N: the old ceiling 5 x 1.0857 / snr is below the scatter
            "snr_ap_pixscaled": [10.0 * LN10_OVER_2P5 / ph[c] for c in ids],
        })
        old_ceiling = 5.0 * LN10_OVER_2P5 / cand["snr_ap_pixscaled"]
        assert (cand["comp_rms"] > old_ceiling).sum() > len(ids) // 2  # the old gate rejected most of them
        out = _select_comps_by_rms_then_color(cand, 1.0, 3, len(ids), 0.5)
        assert set(out["catalog_id"]) == set(ids)


# ------------------------------------------------------------------------------------------------
# A: fail loud
# ------------------------------------------------------------------------------------------------
def test_ceiling_requires_the_photon_column():
    cand = pd.DataFrame({"catalog_id": ["1", "2", "3"], "bp_rp": 1.0, "comp_rms": 0.01, "_dist_deg": 0.1,
                         "_nn_dist_fwhm": 5.0, "snr_ap_pixscaled": 100.0})
    with pytest.raises(ValueError, match=COMP_SIGMA_REF_COL):
        _select_comps_by_rms_then_color(cand, 1.0, 2, 3, 0.5)


@pytest.mark.parametrize("drop", ["sky_n_px", "sky_adu_per_px_annulus", "fwhm_px_for_aperture"])
def test_photon_reference_fails_loud_on_missing_proc_columns(drop):
    ids, _t, paths, cache = _photon_limited_frames(n_frames=5)
    cache = {k: v.drop(columns=[drop]) for k, v in cache.items()}
    with pytest.raises(RuntimeError, match=drop):
        photon_mag_err_map(set(ids), paths, cache, flux_k_column(0.75), k=0.75, gain=GAIN, read_noise=RN)


def test_photon_reference_fails_loud_without_gain():
    ids, _t, paths, cache = _photon_limited_frames(n_frames=5)
    with pytest.raises(ValueError):
        photon_mag_err_map(set(ids), paths, cache, flux_k_column(0.75), k=0.75, gain=float("nan"), read_noise=RN)


def test_pool_requires_gain_and_carries_the_photon_column():
    from config import AppConfig
    from photometry_comp import build_global_comp_pool
    from test_comp_determinism_synthetic import _TARGET_ID, _build_synthetic_dataset

    _target, ms, paths, cache = _build_synthetic_dataset()
    cfg = AppConfig()
    cfg.gs11_dilution_enabled = False
    args = (ms, paths, cache, frozenset({_TARGET_ID}), None, 800, 600, 0, cfg)
    with pytest.raises(TypeError):
        build_global_comp_pool(*args, min_frames_frac=0.3, fwhm_px=3.7)  # no gain / read noise: no default
    from sigma_budget import RigScintillationParams

    no_scint = RigScintillationParams(None, "test", float("nan"), float("nan"), 60.0, 1.5, [])
    pool = build_global_comp_pool(*args, min_frames_frac=0.3, fwhm_px=3.7, gain=1.0, read_noise=10.0,
                                  scint_params=no_scint)
    # frames without the grid: dao_flux at aperture_r_px (7 px), sky 200 ADU/px over 500 px (fixture)
    cid = str(pool["catalog_id"].iloc[0])
    fl = [float(cache[str(p)].set_index("catalog_id").loc[cid, "dao_flux"]) for p in paths]
    want = LN10_OVER_2P5 * float(np.median([ccd_rel_error(f, 200.0, math.pi * 49.0, 500, gain=1.0, read_noise=10.0)
                                            for f in fl]))
    assert float(pool[COMP_PHOTON_MAG_COL].iloc[0]) == pytest.approx(want, rel=1e-12)
    assert pool[COMP_PHOTON_MAG_COL].notna().all()


# ------------------------------------------------------------------------------------------------
# B: field k within one bootstrap SE (no fixed 1 %)
# ------------------------------------------------------------------------------------------------
def _old_choose_k_1pct(scores: dict[float, float]) -> float:
    best = min(scores.values())
    return max(k for k, v in scores.items() if v <= best * 1.01)


CORR_K = 0.97


def _scaled_noise_frames(c: dict[float, float], n_stars=60, n_frames=50, seed=11):
    """Nested apertures: the noise of neighbouring k is strongly correlated (the same pixels; correlation
    ``CORR_K``); the LOO RMS of column k scales exactly with c[k] (no common term over the stars)."""
    rng = np.random.default_rng(seed)
    ids = [f"{8000000000000000000 + i}" for i in range(n_stars)]
    m0 = rng.uniform(11.0, 15.0, n_stars)
    sig = 0.002 + 0.01 * (m0 - 11.0) / 4.0
    common = rng.normal(0.0, 1.0, (n_stars, n_frames))
    noise = {k: math.sqrt(CORR_K) * common + math.sqrt(1.0 - CORR_K) * rng.normal(0.0, 1.0, (n_stars, n_frames))
             for k in sorted(c)}
    cache, paths = {}, []
    for f in range(n_frames):
        d = {"catalog_id": ids}
        for k in sorted(c):
            d[flux_k_column(k)] = 10 ** (-0.4 * (m0 + c[k] * sig * noise[k][:, f]))
        p = Path(f"proc_se_{f:03d}.csv")
        paths.append(p)
        cache[str(p)] = pd.DataFrame(d)
    return ids, paths, cache


def test_field_k_does_not_flip_on_a_score_difference_below_the_se():
    """Brno 534 g: the score ratio 0.875 / 0.75 moved from 1.0087 to 1.0103 with a 0.1 % flux shift and the
    1 % rule flipped the field k. Here the ratio is set to 1.0095 and then shifted by 0.1 %: the old rule
    flips, the SE rule keeps the larger k both times (the difference is far below one SE)."""
    base = {0.6: 1.10, 0.75: 1.0, 0.875: 1.0, 1.0: 1.3}
    ids, paths, cache = _scaled_noise_frames(base)
    _k, _m, s0, _i = choose_field_aperture_k(set(ids), paths, cache)
    c88 = 1.0095 * s0[0.75] / s0[0.875]
    res = {}
    for shift in (1.0, 1.001):
        c = {**base, 0.875: c88 * shift}
        ids, paths, cache = _scaled_noise_frames(c)
        res[shift] = choose_field_aperture_k(set(ids), paths, cache)
    r_a, r_b = res[1.0][2][0.875] / res[1.0][2][0.75], res[1.001][2][0.875] / res[1.001][2][0.75]
    assert r_a < 1.01 < r_b  # straddles the old 1 % edge
    assert _old_choose_k_1pct(res[1.0][2]) == 0.875 and _old_choose_k_1pct(res[1.001][2]) == 0.75
    for k, _m, _s, info in res.values():
        assert k == 0.875
        curve = info["ratio_curve"]
        assert curve["0.88"] - curve["0.75"] < info["se"]
        assert info["se"] > 0 and info["n_stars"] == len(ids)


def test_field_k_takes_the_minimum_when_the_difference_exceeds_the_se():
    ids, paths, cache = _scaled_noise_frames({0.6: 1.10, 0.75: 1.0, 0.875: 1.15, 1.0: 1.3})
    k, m, scores, info = choose_field_aperture_k(set(ids), paths, cache)
    assert k == 0.75 and info["ratio_curve"]["0.88"] - info["ratio_curve"]["0.75"] > info["se"]
    assert m == loo_mag_rms_maps(set(ids), paths, cache, [flux_k_column(0.75)], strict_flux_col=True)[
        flux_k_column(0.75)]


def test_k_within_one_se_equals_the_former_brightness_group_code():
    """The shared helper is the former inline AP-BRIGHT-K group code, bit for bit (one generator for all
    groups, draws in the same order)."""
    from comp_rms_loo import BRIGHT_K_BOOT, k_within_one_se

    ks = [0.6, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0]
    gen = np.random.default_rng(4)
    groups = []
    for n in (30, 57, 211, 33):
        raw = gen.uniform(0.002, 0.05, (n, 1)) * gen.uniform(0.9, 1.6, (n, len(ks)))
        groups.append(raw / raw.min(axis=1, keepdims=True))
    rng_old, rng_new = np.random.default_rng(0), np.random.default_rng(0)
    for r in groups:
        curve = np.median(r, axis=0)
        j = int(np.argmin(curve))
        boot = [np.median(r[rng_old.integers(0, len(r), len(r)), j]) for _ in range(BRIGHT_K_BOOT)]
        se = float(np.std(boot))
        k_old = max(k for k, v in zip(ks, curve, strict=False) if v <= curve[j] + se)
        k_new, curve_new, se_new = k_within_one_se(ks, r, rng_new)
        assert k_new == k_old and curve_new.tobytes() == curve.tobytes() and np.float64(se_new).tobytes() == \
            np.float64(se).tobytes()
