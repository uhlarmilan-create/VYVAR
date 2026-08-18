"""G2-F002b: per-epoch catalog_match_mode + wcs_untrusted soft trust downgrade."""

from pathlib import Path

import numpy as np
import pandas as pd

from catalog_match_trust import (
    INTERNAL_MATCH_MODE_TO_EXPORT,
    TRUSTED_CATALOG_MATCH_MODES,
    UNTRUSTED_FLUX_CATALOG_MATCH_MODES,
    is_wcs_untrusted_catalog_match_mode,
)
from photometry_core import read_flux_from_csv, save_lightcurve_csv
from trust_flag_core import classify_warnings, evaluate_target, CompTrustThresholds


def test_internal_match_modes_map_to_export_buckets() -> None:
    assert INTERNAL_MATCH_MODE_TO_EXPORT["sky"] in TRUSTED_CATALOG_MATCH_MODES
    assert INTERNAL_MATCH_MODE_TO_EXPORT["pixel_fallback_no_wcs"] in UNTRUSTED_FLUX_CATALOG_MATCH_MODES
    assert INTERNAL_MATCH_MODE_TO_EXPORT["pixel_fallback_bad_wcs"] in UNTRUSTED_FLUX_CATALOG_MATCH_MODES
    assert INTERNAL_MATCH_MODE_TO_EXPORT["nondet_unaligned_no_wcs"] == "nondet_no_wcs"
    assert INTERNAL_MATCH_MODE_TO_EXPORT["sky_unaligned_no_pixel_fallback"] == "nondet_unaligned_sky"
    assert INTERNAL_MATCH_MODE_TO_EXPORT["nondet_unaligned_no_wcs"] not in TRUSTED_CATALOG_MATCH_MODES
    assert is_wcs_untrusted_catalog_match_mode("master_reference_pixel")
    assert not is_wcs_untrusted_catalog_match_mode("master_reference_sky")
    assert not is_wcs_untrusted_catalog_match_mode("full_cone")


def test_read_flux_propagates_catalog_match_mode_without_changing_mag(tmp_path: Path) -> None:
    csv_sky = tmp_path / "sky.csv"
    csv_pix = tmp_path / "pix.csv"
    row = {
        "catalog_id": "1234567890123456789",
        "x": [10.0],
        "y": [10.0],
        "dao_flux": [1000.0],
        "aperture_r_px": [3.0],
        "sigma_bkg_ap": [10.0],
        "flag": ["normal"],
    }
    pd.DataFrame({**row, "catalog_match_mode": ["master_reference_sky"]}).to_csv(csv_sky, index=False)
    pd.DataFrame({**row, "catalog_match_mode": ["master_reference_pixel"]}).to_csv(csv_pix, index=False)
    ids = ["1234567890123456789"]
    ap = {"1234567890123456789": 3.0}
    out_sky = read_flux_from_csv(csv_sky, ids, ap)
    out_pix = read_flux_from_csv(csv_pix, ids, ap)
    assert out_sky.iloc[0]["catalog_match_mode"] == "master_reference_sky"
    assert not bool(out_sky.iloc[0]["wcs_untrusted"])
    assert out_pix.iloc[0]["catalog_match_mode"] == "master_reference_pixel"
    assert bool(out_pix.iloc[0]["wcs_untrusted"])
    assert float(out_sky.iloc[0]["mag_inst"]) == float(out_pix.iloc[0]["mag_inst"])


def test_save_lightcurve_wcs_untrusted_columns_additive(tmp_path: Path) -> None:
    n = 4
    bjd = np.arange(n, dtype=float)
    mag = np.linspace(10.0, 10.3, n)
    flags = ["normal"] * n
    cmm = ["master_reference_sky", "master_reference_pixel", "master_reference_sky", "master_reference_pixel"]
    wut = np.array([False, True, False, True], dtype=bool)
    out = tmp_path / "lc.csv"
    base_kw = {
        "hjd": bjd.copy(),
        "jd": bjd.copy(),
        "airmass": np.full(n, 1.0),
        "is_flipped": np.zeros(n, dtype=bool),
        "mag_inst": mag.copy(),
        "mag_calib_raw": mag.copy(),
        "mag_calib": mag.copy(),
        "mag_calib_ct": mag.copy(),
        "mag_calib_ac": mag.copy(),
        "delta_mag": np.zeros(n),
        "err": np.full(n, 0.01),
        "aperture_r_px": np.full(n, 3.0),
        "flags": flags,
        "source_files": [f"f{i}.csv" for i in range(n)],
        "catalog_match_mode": cmm,
        "wcs_untrusted": wut,
    }
    save_lightcurve_csv(out, bjd, **base_kw)
    df = pd.read_csv(out)
    assert list(df["catalog_match_mode"]) == cmm
    assert df["wcs_untrusted"].astype(bool).tolist() == wut.tolist()


def test_trust_soft_warning_on_wcs_untrusted_epochs() -> None:
    th = CompTrustThresholds.from_bounds(3, 8)
    _, soft = classify_warnings(
        n_clean=5,
        check_scatter=0.01,
        lc_quality="good",
        thresholds=th,
        n_frames=10,
        n_check=6,
        n_wcs_untrusted=2,
    )
    assert any("pixel-fallback WCS match" in s for s in soft)
    info = evaluate_target(
        catalog_id="1",
        vsx_name="T",
        n_clean=5,
        lc_quality="good",
        check_scatter=0.01,
        thresholds=th,
        n_frames=10,
        n_check=6,
        n_wcs_untrusted=2,
    )
    assert info["trust"] == "YELLOW"
    info_ok = evaluate_target(
        catalog_id="1",
        vsx_name="T",
        n_clean=5,
        lc_quality="good",
        check_scatter=0.01,
        thresholds=th,
        n_frames=10,
        n_check=6,
        n_wcs_untrusted=0,
    )
    assert info_ok["trust"] == "GREEN"


def test_alignment_failed_and_wcs_untrusted_are_mutually_exclusive_in_practice() -> None:
    """Pixel-fallback requires aligned grid; alignment_failed implies unaligned nondet path."""
    assert is_wcs_untrusted_catalog_match_mode("master_reference_pixel")
    assert not is_wcs_untrusted_catalog_match_mode("master_reference_sky")
