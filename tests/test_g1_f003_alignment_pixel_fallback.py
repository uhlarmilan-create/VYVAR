"""G1-F003: pixel-fallback gated on VY_ALGN; alignment-failed LC flags + trust."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from photometry_core import read_flux_from_csv, save_lightcurve_csv
from trust_flag_core import CompTrustThresholds, classify_warnings, evaluate_target

_TH = CompTrustThresholds.from_bounds(3, 8)


def _master_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "catalog_id": ["STAR1"],
            "ra_deg": [180.0],
            "dec_deg": [45.0],
            "x": [50.0],
            "y": [50.0],
            "mag": [10.0],
            "catalog": ["Gaia"],
            "name": ["STAR1"],
        }
    )


def _bad_sky_wcs_hdr(aligned: bool) -> fits.Header:
    hdr = fits.Header()
    hdr["VY_ALGN"] = (aligned, "test")
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRVAL1"] = 0.0
    hdr["CRVAL2"] = 0.0
    hdr["CRPIX1"] = 1
    hdr["CRPIX2"] = 1
    hdr["CD1_1"] = -0.0001
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = 0.0001
    return hdr


def _no_wcs_hdr(aligned: bool) -> fits.Header:
    hdr = fits.Header()
    hdr["VY_ALGN"] = (aligned, "test")
    return hdr


def _fake_dao_table() -> Table:
    return Table(
        {
            "xcentroid": [50.0],
            "ycentroid": [50.0],
            "flux": [1000.0],
            "peak": [500.0],
        }
    )


def _run_detect(aligned: bool, with_wcs: bool) -> tuple[pd.DataFrame, dict]:
    import pipeline as pl

    rng = np.random.default_rng(0)
    data = rng.normal(1000.0, 50.0, (100, 100)).astype(np.float32)
    hdr = _bad_sky_wcs_hdr(aligned) if with_wcs else _no_wcs_hdr(aligned)
    master = _master_df()
    fake_tbl = _fake_dao_table()

    class _FakeFinder:
        def __init__(self, **kwargs):
            pass

        def __call__(self, _data):
            return fake_tbl

    def _pass2(data0, tbl, **kwargs):
        return tbl, 0, 0

    with patch("photutils.detection.DAOStarFinder", _FakeFinder):
        with patch.object(pl, "_dao_targeted_pass2_unmatched_gaia", _pass2):
            return pl.detect_stars_match_master_reference(
                data, hdr, master, match_sep_arcsec=8.0
            )


def test_aligned_bad_wcs_uses_pixel_fallback() -> None:
    df, meta = _run_detect(aligned=True, with_wcs=True)
    assert int(meta.get("n_matched", 0)) >= 1
    assert meta.get("catalog_match_mode") == "master_reference_pixel"
    assert "STAR1" in df["catalog_id"].astype(str).tolist()


def test_unaligned_bad_wcs_skips_pixel_fallback() -> None:
    df, meta = _run_detect(aligned=False, with_wcs=True)
    assert int(meta.get("n_matched", 0)) == 0
    assert meta.get("catalog_match_mode") == "master_reference_sky"
    assert not any(str(c).strip() for c in df.get("catalog_id", pd.Series(dtype=str)))


def test_aligned_no_wcs_uses_pixel_fallback() -> None:
    df, meta = _run_detect(aligned=True, with_wcs=False)
    assert int(meta.get("n_matched", 0)) >= 1
    assert meta.get("catalog_match_mode") == "master_reference_pixel"
    assert "STAR1" in df["catalog_id"].astype(str).tolist()


def test_unaligned_no_wcs_nondetection() -> None:
    df, meta = _run_detect(aligned=False, with_wcs=False)
    assert int(meta.get("n_matched", 0)) == 0
    assert meta.get("catalog_match_mode") == "master_reference_sky"
    assert not any(str(c).strip() for c in df.get("catalog_id", pd.Series(dtype=str)))


def test_read_flux_propagates_alignment_failed(tmp_path: Path) -> None:
    csv_path = tmp_path / "frame.csv"
    pd.DataFrame(
        {
            "catalog_id": ["STAR1"],
            "x": [10.0],
            "y": [10.0],
            "dao_flux": [1000.0],
            "aperture_r_px": [3.0],
            "flag": ["normal"],
        }
    ).to_csv(csv_path, index=False)
    ft = {"alignment_failed": True, "aligned": False, "airmass": 1.1}
    out = read_flux_from_csv(
        csv_path,
        ["STAR1"],
        {"STAR1": 3.0},
        frame_times=ft,
    )
    assert bool(out.iloc[0]["alignment_failed"])
    mag_with = float(out.iloc[0]["mag_inst"])
    ft_ok = {"alignment_failed": False, "aligned": True, "airmass": 1.1}
    out_ok = read_flux_from_csv(csv_path, ["STAR1"], {"STAR1": 3.0}, frame_times=ft_ok)
    assert not bool(out_ok.iloc[0]["alignment_failed"])
    assert mag_with == float(out_ok.iloc[0]["mag_inst"])


def test_save_lightcurve_alignment_failed_column_do_no_harm(tmp_path: Path) -> None:
    n = 5
    bjd = np.arange(n, dtype=float)
    mag = np.linspace(10.0, 10.4, n)
    flags = ["normal"] * n
    src = [f"f{i}.csv" for i in range(n)]
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
        "source_files": src,
    }
    p_default = tmp_path / "lc_default.csv"
    save_lightcurve_csv(p_default, bjd, **base_kw)
    p_flag = tmp_path / "lc_flag.csv"
    save_lightcurve_csv(
        p_flag,
        bjd,
        **base_kw,
        alignment_failed=np.zeros(n, dtype=bool),
    )
    df_def = pd.read_csv(p_default)
    df_flag = pd.read_csv(p_flag)
    assert list(df_def["mag_inst"]) == list(df_flag["mag_inst"])
    assert df_flag["alignment_failed"].astype(bool).sum() == 0


def test_trust_soft_warning_on_alignment_failed_epochs() -> None:
    _, soft = classify_warnings(
        n_clean=_TH.strong,
        check_scatter=0.01,
        lc_quality="good",
        thresholds=_TH,
        n_check=10,
        n_frames=20,
        n_alignment_failed=2,
    )
    assert any("failed alignment" in s for s in soft)
    info = evaluate_target(
        catalog_id="1",
        vsx_name="T",
        n_clean=_TH.strong,
        lc_quality="good",
        check_scatter=0.01,
        thresholds=_TH,
        n_check=10,
        n_frames=20,
        n_alignment_failed=0,
    )
    assert info["trust"] == "GREEN"
    assert not any("failed alignment" in s for s in info["soft_warnings"])
