"""APERTURE-01 unit tests: one r/frame, FWHM-AUTH-01, continuous EE r90."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from aperture_policy import (
    FWHM_AUTHORITY,
    MODE_FIXED_NIGHT,
    MODE_PER_FRAME,
    ee_r90_continuous,
    fwhm_for_radius,
    fwhm_from_header_vy_fwhm,
    normalize_aperture_policy_mode,
    policy_header_line,
    resolve_aperture_geometry,
    resolve_frame_fwhm_px,
)
from photometry_core import enhance_catalog_dataframe_aperture_bpm


def test_mode_aliases() -> None:
    assert normalize_aperture_policy_mode("a") == MODE_FIXED_NIGHT
    assert normalize_aperture_policy_mode("f_per_frame") == MODE_PER_FRAME
    assert normalize_aperture_policy_mode("bogus") == MODE_FIXED_NIGHT


def test_fixed_night_ignores_frame_fwhm() -> None:
    used = fwhm_for_radius(
        MODE_FIXED_NIGHT, fwhm_frame_px=6.0, fwhm_night_median_px=5.0
    )
    assert used == 5.0
    used_b = fwhm_for_radius(
        MODE_PER_FRAME, fwhm_frame_px=6.0, fwhm_night_median_px=5.0
    )
    assert used_b == 6.0


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


def test_ee_r90_interpolates_not_nearest_bin() -> None:
    radii = np.array([4.0, 4.5, 5.0, 5.5, 6.0], dtype=np.float64)
    ee = np.array([0.70, 0.82, 0.88, 0.94, 0.97], dtype=np.float64)
    r90 = ee_r90_continuous(radii, ee)
    # Crosses 0.9 between 5.0 (0.88) and 5.5 (0.94)
    expected = 5.0 + (0.9 - 0.88) * (5.5 - 5.0) / (0.94 - 0.88)
    assert abs(r90 - expected) < 1e-9
    nearest = float(radii[int(np.argmin(np.abs(ee - 0.9)))])
    assert nearest in (5.0, 5.5)
    assert abs(r90 - nearest) > 0.01


def test_policy_header_ascii_json() -> None:
    rec = {
        "policy_id": "APERTURE-01",
        "mode": MODE_FIXED_NIGHT,
        "f": 1.2,
        "fwhm_authority": FWHM_AUTHORITY,
        "fwhm_night_median_px": 5.19,
        "r_ap_px": 6.228,
        "r_in_px": 24.65,
        "r_out_px": 46.71,
    }
    line = policy_header_line(rec)
    assert line.startswith("# aperture_policy: ")
    assert "f_fixed_night" in line
    assert line.encode("ascii")


def test_draft_dir_resolves_snapshot_and_qc() -> None:
    from photometry_core import _draft_dir_from_phase2a_paths

    snap = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000516_snapshot_era04_20260826")
    phot = snap / "platesolve" / "NoFilter_60_2" / "photometry"
    ms = snap / "platesolve" / "NoFilter_60_2" / "MASTERSTAR.fits"
    if not snap.is_dir():
        return
    got = _draft_dir_from_phase2a_paths(phot, ms)
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
    snr_table = {
        "table": {f"{m:.1f}": 3.0 + 0.4 * i for i, m in enumerate(df["mag"])},
        "fwhm_px": 5.0,
        "fwhm_px_scope": "test",
    }
    out = enhance_catalog_dataframe_aperture_bpm(
        df,
        data,
        {"VY_FWHM": 5.2},
        aperture_enabled=True,
        aperture_fwhm_factor=1.2,
        annulus_inner_fwhm=4.75,
        annulus_outer_fwhm=9.0,
        nonlinearity_peak_percentile=20.0,
        nonlinearity_fwhm_ratio=1.25,
        master_dark_path=None,
        snr_aperture_table=snr_table,
        aperture_policy_mode=MODE_FIXED_NIGHT,
        fwhm_frame_px=5.2,
        fwhm_night_median_px=5.0,
    )
    r = pd.to_numeric(out["aperture_r_px"], errors="coerce")
    assert r.nunique() == 1
    assert abs(float(r.iloc[0]) - 1.2 * 5.0) < 1e-6
    assert str(out["aperture_policy_mode"].iloc[0]) == MODE_FIXED_NIGHT
    assert abs(float(out["aperture_f"].iloc[0]) - 1.2) < 1e-12
    assert abs(float(out["fwhm_px_for_aperture"].iloc[0]) - 5.2) < 1e-12
    assert abs(float(out["sky_annulus_r_in_px"].iloc[0]) - 4.75 * 5.0) < 1e-6
    assert str(out["snr_aperture_mode"].iloc[0]) == "aperture_01"


def test_per_frame_mode_tracks_frame_fwhm() -> None:
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
    out = enhance_catalog_dataframe_aperture_bpm(
        df,
        data,
        {"VY_FWHM": 6.0},
        aperture_enabled=True,
        aperture_fwhm_factor=1.0,
        annulus_inner_fwhm=4.0,
        annulus_outer_fwhm=6.0,
        nonlinearity_peak_percentile=20.0,
        nonlinearity_fwhm_ratio=1.25,
        master_dark_path=None,
        aperture_policy_mode=MODE_PER_FRAME,
        fwhm_frame_px=6.0,
        fwhm_night_median_px=5.0,
    )
    r = float(out["aperture_r_px"].iloc[0])
    assert abs(r - 6.0) < 1e-6
    assert math.isfinite(float(out["flux"].iloc[0]))
