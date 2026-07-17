"""Unit tests for diagnostic sigma budget (Howell + Osborn scintillation)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from photometry_core import (
    ERR_BKG_SOURCE_EMPIRICAL,
    SIGMA_BKG_AP_COL,
    _photometric_error,
)
from scripts.chi2_sigma_gate import (
    evaluate_lc_chi2_variants,
    production_lc_err_sigma_mag,
    reduced_chi2_constant,
    sigma_arrays_from_lc_and_proc,
)
from sigma_budget import (
    SIGMA_VARIANT_HOWELL_ONLY,
    SIGMA_VARIANT_HOWELL_SCINT_FULL,
    SIGMA_VARIANT_PRODUCTION_LC_ERR,
    RigScintillationParams,
    howell_sigma,
    scintillation_sigma,
    total_sigma,
)


def test_howell_sigma_matches_production():
    flux, sky, area = 10000.0, 100.0, 28.27
    gain, rn = 3.17, 7.6
    expected = _photometric_error(flux, sky, area, gain=gain, read_noise=rn)
    assert howell_sigma(flux, sky, area, gain=gain, read_noise=rn) == pytest.approx(expected)


def test_scintillation_osborn_paper_scale_sanity():
    """Larger aperture -> lower scintillation variance (D^-4/3)."""
    v200 = scintillation_sigma(telescope_diameter_m=0.2, airmass=1.5, exposure_s=60.0, altitude_m=275.0) ** 2
    v300 = scintillation_sigma(telescope_diameter_m=0.3, airmass=1.5, exposure_s=60.0, altitude_m=275.0) ** 2
    assert v200 > v300
    ratio = v200 / v300
    assert abs(ratio - (0.3 / 0.2) ** (4.0 / 3.0)) < 0.05


def test_total_sigma_quadrature_variants():
    kwargs = dict(
        flux=5000.0,
        sky_pp=50.0,
        area=100.0,
        telescope_diameter_m=0.2,
        airmass=1.5,
        exposure_s=60.0,
        altitude_m=250.0,
    )
    h_only, sh, _ = total_sigma(**kwargs, variant=SIGMA_VARIANT_HOWELL_ONLY)
    full, sh2, ss_full = total_sigma(**kwargs, variant=SIGMA_VARIANT_HOWELL_SCINT_FULL)
    assert h_only == pytest.approx(sh)
    assert sh2 == pytest.approx(sh)
    assert full == pytest.approx(math.sqrt(sh**2 + ss_full**2))


def test_chi2_near_one_with_correct_sigmas():
    rng = np.random.default_rng(0)
    sig = 0.01
    mags = rng.normal(10.0, sig, 80)
    sigmas = np.full(80, sig)
    _, dof, chi2_dof, _ = reduced_chi2_constant(mags, sigmas)
    assert dof == 79
    assert 0.6 < chi2_dof < 1.4


def test_chi2_inflated_when_sigma_too_small():
    rng = np.random.default_rng(1)
    sig = 0.012
    mags = rng.normal(10.0, sig, 50)
    _, _, ok, _ = reduced_chi2_constant(mags, np.full(50, sig))
    _, _, bad, _ = reduced_chi2_constant(mags, np.full(50, sig * 0.4))
    assert bad > ok


def _rig_params() -> RigScintillationParams:
    return RigScintillationParams(
        draft_id=426,
        setup="g_60_4",
        telescope_diameter_m=0.2,
        altitude_m=275.0,
        exposure_s=60.0,
        c_y=1.5,
        source_notes=["test"],
    )


def _write_proc_row(
    proc_dir: Path,
    *,
    source_file: str,
    catalog_id: str,
    sigma_bkg_ap: float | None,
    err_bkg_source: str | None = None,
) -> None:
    row = {
        "catalog_id": catalog_id,
        "dao_flux": 12000.0,
        "sky_adu_per_px_annulus": 80.0,
        "aperture_area_px": 50.0,
        "aperture_r_px": 4.0,
        SIGMA_BKG_AP_COL: sigma_bkg_ap if sigma_bkg_ap is not None else float("nan"),
    }
    if err_bkg_source is not None:
        row["err_bkg_source"] = err_bkg_source
    pd.DataFrame([row]).to_csv(proc_dir / source_file, index=False)


def test_sigma_arrays_uses_empirical_sigma_bkg_ap(tmp_path: Path) -> None:
    cid = "1112127291051695744"
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    sf = "proc_frame_001.csv"
    sig_ap = 42.0
    _write_proc_row(
        proc_dir,
        source_file=sf,
        catalog_id=cid,
        sigma_bkg_ap=sig_ap,
        err_bkg_source=ERR_BKG_SOURCE_EMPIRICAL,
    )
    lc_df = pd.DataFrame(
        {
            "source_file": [sf],
            "airmass": [1.2],
            "delta_mag": [10.0],
            "err": [0.02],
            "bjd": [2459000.0],
        }
    )
    _, variants, _, _, meta = sigma_arrays_from_lc_and_proc(
        lc_df, proc_dir, cid, rig_params=_rig_params(), gain=12.48, read_noise=7.6,
    )
    analytic = howell_sigma(12000.0, 80.0, 50.0, gain=12.48, read_noise=7.6)
    empirical = math.sqrt(12000.0 / 12.48 + sig_ap * sig_ap) / 12000.0
    assert empirical != pytest.approx(analytic, rel=0.01)
    got = variants[SIGMA_VARIANT_HOWELL_ONLY][0] / (2.5 / math.log(10))
    assert got == pytest.approx(empirical, rel=1e-6)
    assert meta["bkg_term_source"]["primary"] == ERR_BKG_SOURCE_EMPIRICAL


def test_sigma_arrays_analytic_fallback_when_sigma_bkg_ap_missing(tmp_path: Path) -> None:
    cid = "1497674651102612992"
    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    sf = "proc_frame_002.csv"
    _write_proc_row(proc_dir, source_file=sf, catalog_id=cid, sigma_bkg_ap=None)
    lc_df = pd.DataFrame(
        {
            "source_file": [sf],
            "airmass": [1.1],
            "delta_mag": [11.0],
            "err": [0.015],
            "bjd": [2459000.1],
        }
    )
    _, variants, _, _, meta = sigma_arrays_from_lc_and_proc(
        lc_df, proc_dir, cid, rig_params=_rig_params(), gain=12.48, read_noise=7.6,
    )
    analytic = howell_sigma(12000.0, 80.0, 50.0, gain=12.48, read_noise=7.6)
    got = variants[SIGMA_VARIANT_HOWELL_ONLY][0] / (2.5 / math.log(10))
    assert got == pytest.approx(analytic, rel=1e-6)
    assert meta["bkg_term_source"]["primary"] == "analytic_fallback"
    assert meta["bkg_term_source"]["counts"]["analytic_fallback"] == 1


def test_production_lc_err_variant_chi2_hand_value() -> None:
    mags = np.array([10.0, 10.01, 9.99], dtype=float)
    err_rel = np.array([0.01, 0.01, 0.01], dtype=float)
    lc_df = pd.DataFrame({"err": err_rel, "bjd": [1.0, 1.1, 1.2]})
    sig = production_lc_err_sigma_mag(lc_df)
    _, _, chi2_dof, _ = reduced_chi2_constant(mags, sig)
    resid = mags - float(np.mean(mags))
    hand = float(np.sum((resid / sig) ** 2) / 2.0)
    assert chi2_dof == pytest.approx(hand, rel=1e-9)
    results = evaluate_lc_chi2_variants(
        mags,
        {SIGMA_VARIANT_PRODUCTION_LC_ERR: sig},
        catalog_id="test",
        mag_g=None,
        bjd=lc_df["bjd"].to_numpy(dtype=float),
    )
    assert len(results) == 1
    assert results[0].variant == SIGMA_VARIANT_PRODUCTION_LC_ERR
    assert results[0].chi2_dof == pytest.approx(hand, rel=1e-9)
