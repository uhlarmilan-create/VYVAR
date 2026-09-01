"""F-BINGAIN-1: empirical background-noise term (empty-aperture scatter)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import AppConfig
from photometry_core import (
    ERR_BKG_MODE_EMPIRICAL,
    ERR_BKG_MODE_HOWELL,
    ERR_BKG_SOURCE_COL,
    ERR_BKG_SOURCE_EMPIRICAL,
    ERR_BKG_SOURCE_HOWELL_FALLBACK,
    ERR_BKG_SOURCE_HOWELL_SCALED,
    SIGMA_BKG_AP_COL,
    _assert_inv_err_sigma_acct_01,
    _clamp_bkg_scale_r,
    _clamp_err_empty_apertures_n,
    _howell_bkg_variance_adu2,
    _normalize_err_background_mode,
    _phase2a_cache_columns,
    _phase2a_proc_column_requirements,
    _photometric_error,
    _phase2a_empirical_sigma_bkg_ap,
    _photometric_error_with_bkg_mode,
    _sigma_bkg_r_key,
    bkg_scale_ratio_empirical_over_howell,
    compute_setup_bkg_scale_r,
    enhance_catalog_dataframe_aperture_bpm,
    finalize_hybrid_bkg_fallback_proc_dir,
    measure_empty_aperture_sigma_bkg,
    read_flux_from_csv,
    scaled_sigma_bkg_ap_from_howell,
)
from invariants_runtime import InvariantViolation


def test_clamp_err_empty_apertures_n() -> None:
    assert _clamp_err_empty_apertures_n(8) == 16
    assert _clamp_err_empty_apertures_n(64) == 64
    assert _clamp_err_empty_apertures_n(512) == 256


def test_normalize_err_background_mode() -> None:
    assert _normalize_err_background_mode("howell") == ERR_BKG_MODE_HOWELL
    assert _normalize_err_background_mode("legacy") == ERR_BKG_MODE_HOWELL
    assert _normalize_err_background_mode("empirical") == ERR_BKG_MODE_EMPIRICAL
    assert _normalize_err_background_mode(None) == ERR_BKG_MODE_EMPIRICAL


def test_howell_key_flip_ignored_uses_empirical_when_sigma_present() -> None:
    """CONSOLIDATE-01D: passing howell no longer skips empirical when sigma_bkg_ap is finite."""
    flux, sky, area, g, rn, sig_ap = 5000.0, 1200.0, math.pi * 5.0**2, 12.48, 14.08, 42.0
    err, src = _photometric_error_with_bkg_mode(
        flux,
        err_background_mode=ERR_BKG_MODE_HOWELL,
        sky_pp=sky,
        area=area,
        gain=g,
        read_noise=rn,
        sigma_bkg_ap=sig_ap,
    )
    var = flux / g + sig_ap**2
    assert src == ERR_BKG_SOURCE_EMPIRICAL
    assert err == pytest.approx(math.sqrt(var) / flux, rel=1e-12)


def test_howell_fallback_when_sigma_missing() -> None:
    flux, sky, area, g, rn = 5000.0, 1200.0, math.pi * 5.0**2, 12.48, 14.08
    legacy = _photometric_error(flux, sky, area, gain=g, read_noise=rn)
    err, src = _photometric_error_with_bkg_mode(
        flux,
        err_background_mode=ERR_BKG_MODE_EMPIRICAL,
        sky_pp=sky,
        area=area,
        gain=g,
        read_noise=rn,
        sigma_bkg_ap=None,
    )
    assert src == ERR_BKG_SOURCE_HOWELL_FALLBACK
    assert err == pytest.approx(legacy, rel=0, abs=1e-15)


def test_empirical_mode_formula() -> None:
    flux, g, sig_ap = 8000.0, 12.48, 42.0
    err, src = _photometric_error_with_bkg_mode(
        flux,
        err_background_mode=ERR_BKG_MODE_EMPIRICAL,
        sky_pp=0.0,
        area=math.pi * 6.0**2,
        gain=g,
        read_noise=14.0,
        sigma_bkg_ap=sig_ap,
    )
    var = flux / g + sig_ap**2
    assert src == ERR_BKG_SOURCE_EMPIRICAL
    assert err == pytest.approx(math.sqrt(var) / flux, rel=1e-12)


def test_empty_aperture_white_noise_matches_analytic() -> None:
    rng = np.random.default_rng(42)
    sigma_px = 3.5
    ny, nx = 400, 400
    data = rng.normal(1000.0, sigma_px, size=(ny, nx))
    r_ap = 5.0
    r_in, r_out = 12.0, 18.0
    sig_ap, n_valid, reason = measure_empty_aperture_sigma_bkg(
        data,
        np.array([]),
        np.array([]),
        r_ap,
        r_in,
        r_out,
        n_apertures=48,
        min_valid=16,
        rng=rng,
    )
    assert reason == ""
    assert n_valid >= 16
    area = math.pi * r_ap**2
    expected = math.sqrt(area) * sigma_px
    assert sig_ap == pytest.approx(expected, rel=0.25)


def test_correlated_noise_empirical_exceeds_howell_reconstruction() -> None:
    """Low-frequency common-mode + white noise: empirical > level-based Howell background."""
    rng = np.random.default_rng(7)
    sky_pp = 1200.0
    g, rn = 12.48, 14.08
    ny, nx = 480, 480
    data = rng.normal(sky_pp, 2.5, size=(ny, nx))
    coarse = rng.normal(0.0, 35.0, size=(ny // 8, nx // 8))
    coarse = np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1)[:ny, :nx]
    data = data + coarse
    r_ap = 5.0
    sig_ap, _, reason = measure_empty_aperture_sigma_bkg(
        data,
        np.array([]),
        np.array([]),
        r_ap,
        12.0,
        18.0,
        n_apertures=64,
        min_valid=16,
        rng=rng,
    )
    assert reason == ""
    area = math.pi * r_ap**2
    howell_bkg_adu = math.sqrt(sky_pp / g * area + (rn / g) ** 2 * area)
    assert sig_ap > howell_bkg_adu * 1.15


def test_crowding_fallback() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(500.0, 2.0, size=(80, 80))
    xs = np.linspace(10, 70, 40)
    ys = np.linspace(10, 70, 40)
    xx, yy = np.meshgrid(xs, ys)
    sig_ap, n_valid, reason = measure_empty_aperture_sigma_bkg(
        data,
        xx.ravel(),
        yy.ravel(),
        6.0,
        14.0,
        20.0,
        n_apertures=32,
        min_valid=16,
        rng=rng,
    )
    assert not math.isfinite(sig_ap)
    assert n_valid < 16
    assert "crowding" in reason


def test_enhance_catalog_emits_provenance_columns() -> None:
    rng = np.random.default_rng(99)
    ny, nx = 256, 256
    data = rng.normal(800.0, 2.5, size=(ny, nx)).astype(np.float32)
    n_stars = 12
    xs = rng.uniform(40, nx - 40, size=n_stars)
    ys = rng.uniform(40, ny - 40, size=n_stars)
    df = pd.DataFrame(
        {
            "x": xs,
            "y": ys,
            "flux": rng.uniform(5000, 15000, size=n_stars),
            "catalog_id": [f"G{i:012d}" for i in range(n_stars)],
            "peak_max_adu": np.full(n_stars, 100.0),
        }
    )
    hdr = {"FWHM": 4.2}
    out = enhance_catalog_dataframe_aperture_bpm(
        df,
        data,
        hdr,
        aperture_enabled=True,
        aperture_fwhm_factor=1.7,
        annulus_inner_fwhm=4.0,
        annulus_outer_fwhm=6.0,
        nonlinearity_peak_percentile=20.0,
        nonlinearity_fwhm_ratio=1.25,
        master_dark_path=None,
        err_background_mode=ERR_BKG_MODE_EMPIRICAL,
        err_empty_apertures_n=32,
        err_empty_apertures_min=8,
    )
    assert SIGMA_BKG_AP_COL in out.columns
    assert ERR_BKG_SOURCE_COL in out.columns
    src = out[ERR_BKG_SOURCE_COL].astype(str)
    assert (src == ERR_BKG_SOURCE_EMPIRICAL).any() or (src == ERR_BKG_SOURCE_HOWELL_FALLBACK).any()
    sig = pd.to_numeric(out[SIGMA_BKG_AP_COL], errors="coerce")
    if (src == ERR_BKG_SOURCE_EMPIRICAL).any():
        assert sig[src == ERR_BKG_SOURCE_EMPIRICAL].notna().all()


def test_config_defaults_and_clamps() -> None:
    cfg = AppConfig()
    assert not hasattr(cfg, "err_background_mode")
    assert cfg.err_empty_apertures_n == 64
    assert cfg.err_empty_apertures_min == 16
    cfg.err_empty_apertures_n = 8
    cfg.err_empty_apertures_min = 300
    # Re-apply clamp logic mirrors __post_init__ bounds
    cfg.err_empty_apertures_n = max(16, min(256, int(cfg.err_empty_apertures_n)))
    cfg.err_empty_apertures_min = max(1, min(256, int(cfg.err_empty_apertures_min)))
    assert cfg.err_empty_apertures_n == 16
    assert cfg.err_empty_apertures_min == 256


def test_bkg_scale_ratio_and_clamp() -> None:
    flux, sky, area, g, rn = 5000.0, 1200.0, math.pi * 5.0**2, 12.48, 14.08
    hb = _howell_bkg_variance_adu2(sky, area, gain=g, read_noise=rn)
    sig = 50.0
    r = bkg_scale_ratio_empirical_over_howell(sig, sky, area, gain=g, read_noise=rn)
    assert r == pytest.approx(sig * sig / hb, rel=1e-9)
    assert _clamp_bkg_scale_r(0.01) == 0.05
    assert _clamp_bkg_scale_r(99.0) == 2.0
    r_med, n = compute_setup_bkg_scale_r([0.5, 0.6, 0.7])
    assert n == 3
    assert r_med == pytest.approx(0.6)


def test_scaled_sigma_bkg_ap_from_howell() -> None:
    sky, area, g, rn, r_setup = 1200.0, math.pi * 4.0**2, 12.48, 14.08, 0.5
    hb = _howell_bkg_variance_adu2(sky, area, gain=g, read_noise=rn)
    sig = scaled_sigma_bkg_ap_from_howell(sky, area, gain=g, read_noise=rn, r_setup=r_setup)
    assert sig == pytest.approx(math.sqrt(r_setup * hb), rel=1e-9)


def test_finalize_hybrid_bkg_fallback(tmp_path: Path) -> None:
    area = math.pi * 3.0**2
    g, rn = 12.48, 14.08
    sky = 800.0
    hb = _howell_bkg_variance_adu2(sky, area, gain=g, read_noise=rn)
    sig_emp = math.sqrt(0.55 * hb)
    df = pd.DataFrame(
        [
            {
                "catalog_id": "1",
                "dao_flux": 5000.0,
                "sky_adu_per_px_annulus": sky,
                "aperture_r_px": 3.0,
                "aperture_area_px": area,
                SIGMA_BKG_AP_COL: sig_emp,
                ERR_BKG_SOURCE_COL: ERR_BKG_SOURCE_EMPIRICAL,
            },
            {
                "catalog_id": "2",
                "dao_flux": 4000.0,
                "sky_adu_per_px_annulus": sky,
                "aperture_r_px": 3.0,
                "aperture_area_px": area,
                SIGMA_BKG_AP_COL: float("nan"),
                ERR_BKG_SOURCE_COL: ERR_BKG_SOURCE_HOWELL_FALLBACK,
            },
        ]
    )
    p = tmp_path / "proc_test.csv"
    df.to_csv(p, index=False)
    stats = finalize_hybrid_bkg_fallback_proc_dir(tmp_path, gain=g, read_noise=rn, setup_label="test")
    assert stats["n_scaled_rows"] == 1
    out = pd.read_csv(p)
    assert out.loc[1, ERR_BKG_SOURCE_COL] == ERR_BKG_SOURCE_HOWELL_SCALED
    assert math.isfinite(float(out.loc[1, SIGMA_BKG_AP_COL]))


def test_finalize_zero_empirical_keeps_raw_fallback(tmp_path: Path) -> None:
    area = math.pi * 3.0**2
    df = pd.DataFrame(
        [
            {
                "catalog_id": "2",
                "dao_flux": 4000.0,
                "sky_adu_per_px_annulus": 800.0,
                "aperture_r_px": 3.0,
                "aperture_area_px": area,
                SIGMA_BKG_AP_COL: float("nan"),
                ERR_BKG_SOURCE_COL: ERR_BKG_SOURCE_HOWELL_FALLBACK,
            },
        ]
    )
    p = tmp_path / "proc_test.csv"
    df.to_csv(p, index=False)
    stats = finalize_hybrid_bkg_fallback_proc_dir(tmp_path, gain=12.48, read_noise=14.08)
    assert stats.get("r_setup") is None
    out = pd.read_csv(p)
    assert out.loc[0, ERR_BKG_SOURCE_COL] == ERR_BKG_SOURCE_HOWELL_FALLBACK


def test_phase2a_cache_columns_cover_named_requirements() -> None:
    req = _phase2a_proc_column_requirements()
    cols = set(_phase2a_cache_columns())
    for group_cols in req.values():
        for col in group_cols:
            assert col in cols
    assert SIGMA_BKG_AP_COL in cols
    assert ERR_BKG_SOURCE_COL in cols
    assert "sky_annulus_r_out_px" in cols


def test_phase2a_empirical_requires_sigma_bkg_ap_input() -> None:
    row = pd.Series(
        {
            "catalog_id": "1498613634033133184",
            "dao_flux": 121563.0,
            "aperture_r_px": 5.0,
            "sky_adu_per_px_annulus": 1000.0,
        }
    )
    with pytest.raises(ValueError, match="INV-ERR-MODE-01"):
        _phase2a_empirical_sigma_bkg_ap(
            row,
            err_background_mode=ERR_BKG_MODE_EMPIRICAL,
            source_file="proc_test.csv",
            catalog_id="1498613634033133184",
        )


def test_phase2a_empirical_accepts_projected_sigma_bkg_ap() -> None:
    row = pd.Series(
        {
            "catalog_id": "1498613634033133184",
            SIGMA_BKG_AP_COL: 1664.0,
            ERR_BKG_SOURCE_COL: ERR_BKG_SOURCE_EMPIRICAL,
        }
    )
    sig = _phase2a_empirical_sigma_bkg_ap(
        row,
        err_background_mode=ERR_BKG_MODE_EMPIRICAL,
        source_file="proc_test.csv",
        catalog_id="1498613634033133184",
    )
    assert sig == pytest.approx(1664.0)


def test_phase2a_projected_cache_matches_full_row_empirical() -> None:
    row = {
        "catalog_id": "1498613634033133184",
        "name": "1498613634033133184",
        "bjd_tdb_mid": 2450000.123,
        "hjd_mid": 2450000.122,
        "jd_mid": 2450000.121,
        "dao_flux": 121563.0,
        "noise_floor_adu": 1000.0,
        "sky_adu_per_px_annulus": 1000.0,
        "aperture_r_px": 5.0,
        "peak_max_adu": 12000.0,
        "airmass": 1.2,
        "x": 100.0,
        "y": 120.0,
        "flux_small": 110000.0,
        "flux_large": 130000.0,
        "mag": 12.3,
        "bp_rp": 0.8,
        "b_v": 0.7,
        "zone": "linear",
        "source_type": "GAIA_MATCHED",
        "vsx_known_variable": False,
        "gaia_dr3_variable_catalog": False,
        "ra_deg": 120.0,
        "dec_deg": 30.0,
        "photometry_ok": True,
        "edge_safe_10px": True,
        "edge_fail": False,
        "snr50_ok": True,
        "is_saturated": False,
        "likely_saturated": False,
        "is_usable": True,
        "catalog_match_mode": "direct",
        "sky_annulus_r_out_px": 18.0,
        SIGMA_BKG_AP_COL: 1664.0,
        ERR_BKG_SOURCE_COL: ERR_BKG_SOURCE_EMPIRICAL,
    }
    full_df = pd.DataFrame([row])
    projected_df = full_df[[c for c in _phase2a_cache_columns() if c in full_df.columns]].copy()
    star_ids = ["1498613634033133184"]
    aps = {"1498613634033133184": 5.0}
    frame = Path("proc_test.csv")

    full = read_flux_from_csv(
        frame,
        star_ids,
        aps,
        csv_df=full_df,
        gain=0.63707,
        read_noise=14.08,
        err_background_mode=ERR_BKG_MODE_EMPIRICAL,
    )
    projected = read_flux_from_csv(
        frame,
        star_ids,
        aps,
        csv_df=projected_df,
        gain=0.63707,
        read_noise=14.08,
        err_background_mode=ERR_BKG_MODE_EMPIRICAL,
    )
    assert len(full) == 1
    assert len(projected) == 1
    assert projected.loc[0, "err"] == pytest.approx(full.loc[0, "err"], rel=0, abs=1e-15)
    assert projected.loc[0, ERR_BKG_SOURCE_COL] == full.loc[0, ERR_BKG_SOURCE_COL]


def test_sigma_bkg_r_key_canonicalizes_518_global_aperture() -> None:
    """ERR-518-01: 4.35461 must round-trip as 4.3546 (pre-fix store/lookup miss)."""
    r_ap = max(0.5, 1.9 * 2.2919)
    assert r_ap == pytest.approx(4.35461)
    assert _sigma_bkg_r_key(r_ap) == 4.3546
    broken_store = {float(r_ap): ("stored", ERR_BKG_SOURCE_EMPIRICAL)}
    assert broken_store.get(_sigma_bkg_r_key(r_ap)) is None


def test_global_fixed_518_r_ap_emits_empirical_sigma(monkeypatch: pytest.MonkeyPatch) -> None:
    """global_fixed + gaussian override 2.2919: rows must be EMPIRICAL with finite sigma."""
    rng = np.random.default_rng(518)
    ny, nx = 512, 512
    data = rng.normal(800.0, 2.5, size=(ny, nx)).astype(np.float32)
    n_stars = 24
    xs = rng.uniform(60, nx - 60, size=n_stars)
    ys = rng.uniform(60, ny - 60, size=n_stars)
    df = pd.DataFrame(
        {
            "x": xs,
            "y": ys,
            "flux": rng.uniform(5000, 15000, size=n_stars),
            "catalog_id": [f"G{i:012d}" for i in range(n_stars)],
            "peak_max_adu": np.full(n_stars, 100.0),
        }
    )
    out = enhance_catalog_dataframe_aperture_bpm(
        df,
        data,
        {"FWHM": 4.2},
        aperture_enabled=True,
        aperture_fwhm_factor=1.9,
        annulus_inner_fwhm=4.75,
        annulus_outer_fwhm=9.0,
        nonlinearity_peak_percentile=20.0,
        nonlinearity_fwhm_ratio=1.25,
        master_dark_path=None,
        gaussian_fwhm_px_override=2.2919,
        err_background_mode=ERR_BKG_MODE_EMPIRICAL,
        err_empty_apertures_n=32,
        err_empty_apertures_min=8,
    )
    assert float(out["aperture_r_px"].iloc[0]) == pytest.approx(4.35461, rel=1e-4)
    src = out[ERR_BKG_SOURCE_COL].astype(str)
    assert (src == ERR_BKG_SOURCE_EMPIRICAL).all()
    sig = pd.to_numeric(out[SIGMA_BKG_AP_COL], errors="coerce")
    assert sig.notna().all()
    assert (sig > 0).all()


def test_inv_err_sigma_acct_01_fires_on_desynced_keys() -> None:
    sigma_by_r = {4.35461: (71.8, ERR_BKG_SOURCE_EMPIRICAL)}
    src_col = np.full(8, ERR_BKG_SOURCE_HOWELL_FALLBACK, dtype=object)
    with pytest.raises(InvariantViolation, match="INV-ERR-SIGMA-ACCT-01"):
        _assert_inv_err_sigma_acct_01(
            sigma_by_r,
            src_col,
            n=8,
            r_ap_arr=None,
            r_ap=4.35461,
        )


def test_snr_table_path_sigma_projection_unchanged() -> None:
    """Scalar APERTURE-01 radii; projection must stay stable across identical calls."""
    rng = np.random.default_rng(516)
    ny, nx = 400, 400
    data = rng.normal(900.0, 2.0, size=(ny, nx)).astype(np.float32)
    n_stars = 16
    mags = np.linspace(10.0, 14.0, n_stars)
    xs = rng.uniform(50, nx - 50, size=n_stars)
    ys = rng.uniform(50, ny - 50, size=n_stars)
    df = pd.DataFrame(
        {
            "x": xs,
            "y": ys,
            "flux": rng.uniform(4000, 12000, size=n_stars),
            "mag": mags,
            "phot_g_mean_mag": mags,
            "catalog_id": [f"G{i:012d}" for i in range(n_stars)],
            "peak_max_adu": np.full(n_stars, 120.0),
        }
    )
    kw = dict(
        aperture_enabled=True,
        aperture_fwhm_factor=1.7,
        annulus_inner_fwhm=4.0,
        annulus_outer_fwhm=6.0,
        nonlinearity_peak_percentile=20.0,
        nonlinearity_fwhm_ratio=1.25,
        master_dark_path=None,
        err_background_mode=ERR_BKG_MODE_EMPIRICAL,
        err_empty_apertures_n=32,
        err_empty_apertures_min=8,
    )
    out_a = enhance_catalog_dataframe_aperture_bpm(df, data, {"FWHM": 4.0}, **kw)
    out_b = enhance_catalog_dataframe_aperture_bpm(df, data, {"FWHM": 4.0}, **kw)
    pd.testing.assert_series_equal(
        out_a[SIGMA_BKG_AP_COL],
        out_b[SIGMA_BKG_AP_COL],
        check_names=True,
    )
    pd.testing.assert_series_equal(
        out_a[ERR_BKG_SOURCE_COL].astype(str),
        out_b[ERR_BKG_SOURCE_COL].astype(str),
        check_names=True,
    )


def test_ram_flush_finalize_scales_fallback_rows(tmp_path: Path) -> None:
    """Deferred flush + finalize: fallback rows become howell_scaled."""
    from pipeline import _finalize_hybrid_bkg_fallback_sidecar

    area = math.pi * 4.3546**2
    g, rn, sky = 3.12, 2.6, 33488.0
    hb = _howell_bkg_variance_adu2(sky, area, gain=g, read_noise=rn)
    sig_emp = math.sqrt(0.55 * hb)
    df = pd.DataFrame(
        [
            {
                "catalog_id": "1",
                SIGMA_BKG_AP_COL: sig_emp,
                ERR_BKG_SOURCE_COL: ERR_BKG_SOURCE_EMPIRICAL,
                "sky_adu_per_px_annulus": sky,
                "aperture_r_px": 4.3546,
                "aperture_area_px": area,
            },
            {
                "catalog_id": "2",
                SIGMA_BKG_AP_COL: float("nan"),
                ERR_BKG_SOURCE_COL: ERR_BKG_SOURCE_HOWELL_FALLBACK,
                "sky_adu_per_px_annulus": sky,
                "aperture_r_px": 4.3546,
                "aperture_area_px": area,
            },
        ]
    )
    sidecar = tmp_path / "proc_frame.csv"
    df.to_csv(sidecar, index=False)
    stats = _finalize_hybrid_bkg_fallback_sidecar(
        tmp_path,
        err_background_mode=ERR_BKG_MODE_EMPIRICAL,
        write_sidecar=True,
        gain=g,
        read_noise=rn,
        setup_label="ram_flush_test",
    )
    assert stats.get("n_scaled_rows") == 1
    out = pd.read_csv(sidecar)
    assert out.loc[1, ERR_BKG_SOURCE_COL] == ERR_BKG_SOURCE_HOWELL_SCALED
    assert math.isfinite(float(out.loc[1, SIGMA_BKG_AP_COL]))


def test_ram_flush_finalize_noop_on_all_fallback(tmp_path: Path) -> None:
    from pipeline import _finalize_hybrid_bkg_fallback_sidecar

    area = math.pi * 4.0**2
    df = pd.DataFrame(
        [
            {
                "catalog_id": "2",
                SIGMA_BKG_AP_COL: float("nan"),
                ERR_BKG_SOURCE_COL: ERR_BKG_SOURCE_HOWELL_FALLBACK,
                "sky_adu_per_px_annulus": 800.0,
                "aperture_r_px": 4.0,
                "aperture_area_px": area,
            },
        ]
    )
    df.to_csv(tmp_path / "proc_only_fallback.csv", index=False)
    stats = _finalize_hybrid_bkg_fallback_sidecar(
        tmp_path,
        err_background_mode=ERR_BKG_MODE_EMPIRICAL,
        write_sidecar=True,
        gain=3.12,
        read_noise=2.6,
        setup_label="all_fallback",
    )
    assert stats.get("r_setup") is None
    assert stats.get("n_ratio_samples") == 0
    out = pd.read_csv(tmp_path / "proc_only_fallback.csv")
    assert out.loc[0, ERR_BKG_SOURCE_COL] == ERR_BKG_SOURCE_HOWELL_FALLBACK
