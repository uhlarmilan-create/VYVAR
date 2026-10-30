"""APERTURE-01 unit tests: one r/frame, FWHM-AUTH-01, continuous EE r90."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from aperture_policy import fwhm_for_radius, fwhm_from_header_vy_fwhm, resolve_aperture_geometry, resolve_frame_fwhm_px
from photometry_core import enhance_catalog_dataframe_aperture


def test_radius_fwhm_is_frame_then_night() -> None:
    assert fwhm_for_radius(fwhm_frame_px=6.0, fwhm_night_median_px=5.0) == 6.0
    assert fwhm_for_radius(fwhm_frame_px=None, fwhm_night_median_px=5.0) == 5.0
    assert fwhm_for_radius(fwhm_frame_px=0.3, fwhm_night_median_px=5.0) == 5.0  # FWHM-BOUND: below 0.5 px invalid


def test_geometry_same_fwhm_scales_annulus() -> None:
    r_ap, r_in, r_out = resolve_aperture_geometry(
        f=1.2, fwhm_px=5.0, annulus_inner_fwhm=4.75, annulus_outer_fwhm=9.0
    )
    assert abs(r_ap - 6.0) < 1e-12
    assert abs(r_in - 4.75 * 5.0) < 1e-12
    assert abs(r_out - 9.0 * 5.0) < 1e-12
    assert r_out > r_in > r_ap


def test_fwhm_auth_no_gaussian_conversion() -> None:
    hdr = {"VY_FWHM": 5.19465}
    v = fwhm_from_header_vy_fwhm(hdr)
    assert v is not None
    assert abs(v - 5.19465) < 1e-9
    # Must NOT be the 0.667 Gaussian conversion of the same card.
    assert abs(v - 5.19465 / 1.5) > 1.0


def test_qc_map_beats_header() -> None:
    hdr = {"VY_FWHM": 4.0}
    v = resolve_frame_fwhm_px(
        hdr=hdr,
        frame_name="BO_CVn_Light_001.fits",
        qc_fwhm_by_name={"BO_CVn_Light_001.fits": 5.2},
        fwhm_night_median_px=5.0,
    )
    assert v == 5.2


def test_draft_dir_resolves_snapshot_and_qc() -> None:
    from photometry_core import _draft_dir_from_aperture_paths

    snap = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000516_snapshot_era04_20260826")
    phot = snap / "platesolve" / "NoFilter_60_2" / "photometry"
    ms = snap / "platesolve" / "NoFilter_60_2" / "MASTERSTAR.fits"
    if not snap.is_dir():
        return
    got = _draft_dir_from_aperture_paths(phot, ms)
    assert got.resolve() == snap.resolve()
    assert (got / "calibrated" / "lights" / "qc_metrics.csv").is_file()


def test_enhance_one_radius_all_stars_ignores_snr_table() -> None:
    rng = np.random.default_rng(516)
    ny, nx = 200, 200
    data = rng.normal(900.0, 2.0, size=(ny, nx)).astype(np.float32)
    n = 8
    df = pd.DataFrame(
        {
            "x": rng.uniform(40, nx - 40, size=n),
            "y": rng.uniform(40, ny - 40, size=n),
            "flux": rng.uniform(4000, 8000, size=n),
            "mag": np.linspace(9.0, 14.0, n),
            "phot_g_mean_mag": np.linspace(9.0, 14.0, n),
            "catalog_id": [f"G{i:012d}" for i in range(n)],
            "peak_max_adu": np.full(n, 120.0),
        }
    )
    out = enhance_catalog_dataframe_aperture(
        df,
        data,
        {"VY_FWHM": 5.2},
        aperture_fwhm_factor=1.2,
        annulus_inner_fwhm=4.75,
        annulus_outer_fwhm=9.0,
        nonlinearity_peak_percentile=20.0,
        nonlinearity_fwhm_ratio=1.25,
        fwhm_frame_px=5.2,
        fwhm_night_median_px=5.0,
    )
    r = pd.to_numeric(out["aperture_r_px"], errors="coerce")
    assert r.nunique() == 1
    assert abs(float(r.iloc[0]) - 1.2 * 5.2) < 1e-6
    assert abs(float(out["fwhm_px_for_aperture"].iloc[0]) - 5.2) < 1e-12
    # OUTPUT-REVIEW-2026-10: constant labels are not written into proc rows
    assert not {"aperture_policy_mode", "aperture_f", "sky_annulus_r_in_px", "snr_aperture_mode",
                "aperture_factor_applied", "fwhm_px_scope", "noise_floor_adu", "flux_small"} & set(out.columns)


def test_radius_tracks_frame_fwhm() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(800.0, 2.0, size=(120, 120)).astype(np.float32)
    df = pd.DataFrame(
        {
            "x": [40.0, 80.0],
            "y": [40.0, 80.0],
            "flux": [5000.0, 6000.0],
            "mag": [10.0, 11.0],
            "catalog_id": ["A", "B"],
            "peak_max_adu": [100.0, 100.0],
        }
    )
    out = enhance_catalog_dataframe_aperture(
        df,
        data,
        {"VY_FWHM": 6.0},
        aperture_fwhm_factor=1.0,
        annulus_inner_fwhm=4.0,
        annulus_outer_fwhm=6.0,
        nonlinearity_peak_percentile=20.0,
        nonlinearity_fwhm_ratio=1.25,
        fwhm_frame_px=6.0,
        fwhm_night_median_px=5.0,
    )
    r = float(out["aperture_r_px"].iloc[0])
    assert abs(r - 6.0) < 1e-6
    assert math.isfinite(float(out["flux"].iloc[0]))


def test_resolve_aperture_geometry_raises_on_invalid_fwhm() -> None:
    import pytest

    with pytest.raises(ValueError, match="invalid fwhm_px"):
        resolve_aperture_geometry(
            f=1.35, fwhm_px=0.0, annulus_inner_fwhm=2.7, annulus_outer_fwhm=5.2
        )
    with pytest.raises(ValueError, match="cannot be resolved"):
        resolve_aperture_geometry(
            f=1.35, fwhm_px=None, annulus_inner_fwhm=2.7, annulus_outer_fwhm=5.2
        )


def test_production_annulus_matches_aij_14_27() -> None:
    """APERTURE-01d: 2.7/5.2 FWHM at 516 night FWHM is AIJ 14/27 px."""
    r_ap, r_in, r_out = resolve_aperture_geometry(
        f=1.35, fwhm_px=5.191733, annulus_inner_fwhm=2.7, annulus_outer_fwhm=5.2
    )
    assert abs(r_ap - 1.35 * 5.191733) < 1e-6
    assert abs(r_in - 14.02) < 0.02
    assert abs(r_out - 27.00) < 0.02
    assert r_out > r_in > r_ap
