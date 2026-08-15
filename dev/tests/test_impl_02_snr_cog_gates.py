"""IMPL-02: fire proofs for SNR CoG hard gates (each gate must be able to fail)."""
from __future__ import annotations

import math

import numpy as np

from snr_cog_gates import evaluate_snr_cog_gates


def _base_good_curve(fwhm: float = 5.0) -> dict:
    """A Moffat-like EE that is monotone, converges, and has r90 ~ 5.5 px."""
    radii = np.arange(0.5, 1.3 * 4.5 * fwhm + 0.01, 0.5)
    # Shape tuned so EE(5.5)~0.9, flat beyond ~4 FWHM.
    ee = 1.0 - np.exp(-((radii / 3.2) ** 1.55))
    i_norm = int(np.argmin(np.abs(radii - 4.5 * fwhm)))
    ee = ee / ee[i_norm]
    ee[i_norm] = 1.0
    # Soft outer slightly above 1 then clip for realism of residual
    ee = np.minimum(ee, 1.02)
    table = {
        8.0: 7.5,
        10.0: 6.8,
        12.0: 6.0,
        14.0: 5.5,
        16.0: 5.2,
    }
    snr = {
        "table": table,
        "r_min_px": 0.8 * fwhm,
        "r_max_px": 2.5 * fwhm,
        "fwhm_px": fwhm,
    }
    return {
        "snr": snr,
        "fwhm": fwhm,
        "ee_radii": radii,
        "ee_curve": ee,
        "ref_r": 4.5 * fwhm,
        "outer_r": float(radii[-1]),
        "flat": float(ee[-1]),
        "r90": float(radii[int(np.argmin(np.abs(ee - 0.9)))]),
    }


def test_good_curve_passes_all_gates():
    g = _base_good_curve()
    rep = evaluate_snr_cog_gates(
        snr_table=g["snr"],
        fwhm_px=g["fwhm"],
        annulus_inner_fwhm=4.75,
        ee_radii=g["ee_radii"],
        ee_curve=g["ee_curve"],
        ref_r_px=g["ref_r"],
        r90_px=g["r90"],
        flatness_outer_over_norm=g["flat"],
        ladder_outer_r_px=g["outer_r"],
    )
    assert rep["ok"] is True, rep.get("failures")


def test_fire_inv_cog_monotone():
    g = _base_good_curve()
    ee = np.array(g["ee_curve"], dtype=np.float64)
    # Inject a rising dEE/dr beyond 1.5 FWHM
    ee[-3] = ee[-4] + 0.01
    ee[-2] = ee[-3] + 0.05
    ee[-1] = ee[-2] + 0.08
    rep = evaluate_snr_cog_gates(
        snr_table=g["snr"],
        fwhm_px=g["fwhm"],
        ee_radii=g["ee_radii"],
        ee_curve=ee,
        ref_r_px=g["ref_r"],
        r90_px=5.5,
        flatness_outer_over_norm=float(ee[-1]),
        ladder_outer_r_px=g["outer_r"],
    )
    assert "INV-COG-MONOTONE" in rep["failures"]
    assert rep["gates"]["INV-COG-MONOTONE"]["pass"] is False


def test_fire_inv_cog_flatness_tautology():
    g = _base_good_curve()
    # Check radius equals norm -> must fail (the IMPL-01 tautology)
    rep = evaluate_snr_cog_gates(
        snr_table=g["snr"],
        fwhm_px=g["fwhm"],
        ee_radii=g["ee_radii"],
        ee_curve=g["ee_curve"],
        ref_r_px=g["ref_r"],
        r90_px=5.5,
        flatness_outer_over_norm=1.0,
        ladder_outer_r_px=g["ref_r"],  # same as norm
    )
    assert "INV-COG-FLATNESS-REAL" in rep["failures"]
    assert rep["gates"]["INV-COG-FLATNESS-REAL"]["check_equals_norm"] is True


def test_fire_inv_cog_convergence():
    g = _base_good_curve()
    ee = np.array(g["ee_curve"], dtype=np.float64)
    ee[-1] = 1.10
    ee[-2] = 0.95
    ee[-3] = 0.85
    rep = evaluate_snr_cog_gates(
        snr_table=g["snr"],
        fwhm_px=g["fwhm"],
        ee_radii=g["ee_radii"],
        ee_curve=ee,
        ref_r_px=g["ref_r"],
        r90_px=5.5,
        flatness_outer_over_norm=float(ee[-1]),
        ladder_outer_r_px=g["outer_r"],
    )
    assert "INV-COG-CONVERGENCE" in rep["failures"]


def test_fire_inv_cog_r90():
    g = _base_good_curve()
    rep = evaluate_snr_cog_gates(
        snr_table=g["snr"],
        fwhm_px=g["fwhm"],
        ee_radii=g["ee_radii"],
        ee_curve=g["ee_curve"],
        ref_r_px=g["ref_r"],
        r90_px=12.0,  # far from Q4 5-6
        flatness_outer_over_norm=g["flat"],
        ladder_outer_r_px=g["outer_r"],
    )
    assert "INV-COG-R90" in rep["failures"]


def test_fire_inv_aperture_mag_monotone_flat():
    g = _base_good_curve()
    # Identical across four magnitudes (IMPL-01 contamination signature)
    g["snr"]["table"] = {8.0: 10.481, 10.0: 10.481, 12.0: 10.481, 14.0: 10.481, 16.0: 10.481}
    rep = evaluate_snr_cog_gates(
        snr_table=g["snr"],
        fwhm_px=g["fwhm"],
        ee_radii=g["ee_radii"],
        ee_curve=g["ee_curve"],
        ref_r_px=g["ref_r"],
        r90_px=5.5,
        flatness_outer_over_norm=g["flat"],
        ladder_outer_r_px=g["outer_r"],
    )
    assert "INV-APERTURE-MAG-MONOTONE" in rep["failures"]


def test_fire_inv_aperture_bound():
    g = _base_good_curve()
    r_max = float(g["snr"]["r_max_px"])
    g["snr"]["table"] = {8.0: r_max, 10.0: r_max - 0.01, 12.0: r_max - 0.02, 14.0: 6.0, 16.0: 5.5}
    rep = evaluate_snr_cog_gates(
        snr_table=g["snr"],
        fwhm_px=g["fwhm"],
        ee_radii=g["ee_radii"],
        ee_curve=g["ee_curve"],
        ref_r_px=g["ref_r"],
        r90_px=5.5,
        flatness_outer_over_norm=g["flat"],
        ladder_outer_r_px=g["outer_r"],
    )
    assert "INV-APERTURE-BOUND" in rep["failures"]


def test_fire_inv_aperture_annulus():
    g = _base_good_curve()
    # Aperture larger than annulus inner (4.75 * 5 = 23.75)
    g["snr"]["table"] = {8.0: 24.0, 10.0: 23.0, 12.0: 6.0, 14.0: 5.5, 16.0: 5.2}
    g["snr"]["r_max_px"] = 30.0
    rep = evaluate_snr_cog_gates(
        snr_table=g["snr"],
        fwhm_px=g["fwhm"],
        annulus_inner_fwhm=4.75,
        ee_radii=g["ee_radii"],
        ee_curve=g["ee_curve"],
        ref_r_px=g["ref_r"],
        r90_px=5.5,
        flatness_outer_over_norm=g["flat"],
        ladder_outer_r_px=g["outer_r"],
    )
    assert "INV-APERTURE-ANNULUS" in rep["failures"]


def test_impl01_flat_table_fails_gates():
    """Regression: the IMPL-01 contaminated table must not pass."""
    fwhm = 3.48
    # Contaminated-like curve: still rising at large r, tautological flatness
    radii = np.arange(0.5, 15.64 + 0.01, 0.5)
    ee = 0.5 + 0.03 * radii  # area-like
    ee = ee / ee[-1]
    snr = {
        "table": {8.0: 10.481, 10.0: 10.481, 12.0: 10.481, 14.0: 10.481, 16.0: 10.481},
        "r_min_px": 0.8 * fwhm,
        "r_max_px": 15.64,
        "fwhm_px": fwhm,
    }
    rep = evaluate_snr_cog_gates(
        snr_table=snr,
        fwhm_px=fwhm,
        ee_radii=radii,
        ee_curve=ee,
        ref_r_px=float(radii[-1]),
        r90_px=8.5,
        flatness_outer_over_norm=1.0,
        ladder_outer_r_px=float(radii[-1]),
    )
    assert rep["ok"] is False
    assert "INV-COG-FLATNESS-REAL" in rep["failures"]
    assert "INV-APERTURE-MAG-MONOTONE" in rep["failures"]
