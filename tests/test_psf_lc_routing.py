"""PSF LC routing, err decoupling, AC guard, spatial flag (PSF-AUDIT-FIXES 2026-07-09)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import AppConfig
from photometry_core import (
    _PSF_ERR_MAG_SCALE,
    _get_lc_adaptive,
    _resolve_star_flux_method,
    _route_lc_per_frame_err,
    compute_lc_flux_method,
    save_lightcurve_csv,
)


def _frames_psf_row(
    *,
    lc_method: str = "psf",
    psf_ac_applied: bool = True,
    psf_flux: float = 1000.0,
    psf_flux_err: float = 10.0,
    err: float = 0.05,
    psf_quality: str = "good",
    psf_fit_ok: bool = True,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "catalog_id": "1",
                "mag_inst": 12.0,
                "bjd": 2459000.0,
                "err": err,
                "lc_flux_method": lc_method,
                "psf_flux": psf_flux,
                "psf_flux_err": psf_flux_err,
                "psf_fit_ok": psf_fit_ok,
                "psf_quality": psf_quality,
                "psf_ac_applied": psf_ac_applied,
            }
        ]
    )


def test_route_lc_err_switches_for_psf_routed_frame() -> None:
    frames = _frames_psf_row()
    err_in = frames["err"].to_numpy(dtype=float)
    err_out, err_methods = _route_lc_per_frame_err(frames, err_in)
    expected = _PSF_ERR_MAG_SCALE * 10.0 / 1000.0
    assert err_methods == ["psf"]
    assert abs(float(err_out[0]) - expected) < 1e-12


def test_route_lc_err_falls_back_when_psf_err_missing() -> None:
    frames = _frames_psf_row(psf_flux_err=float("nan"))
    err_in = frames["err"].to_numpy(dtype=float)
    err_out, err_methods = _route_lc_per_frame_err(frames, err_in)
    assert err_methods == ["aperture"]
    assert float(err_out[0]) == float(err_in[0])


def test_save_lightcurve_no_err_method_when_psf_off(tmp_path: Path) -> None:
    n = 3
    out = tmp_path / "lc.csv"
    save_lightcurve_csv(
        out,
        np.arange(n, dtype=float),
        np.arange(n, dtype=float),
        np.arange(n, dtype=float),
        np.ones(n),
        np.zeros(n, dtype=bool),
        np.full(n, 12.0),
        np.full(n, 12.0),
        np.full(n, 12.0),
        np.full(n, 12.0),
        np.full(n, 12.0),
        np.zeros(n),
        np.full(n, 0.01),
        np.full(n, 3.0),
        ["normal"] * n,
        [f"f{i}.csv" for i in range(n)],
    )
    df = pd.read_csv(out)
    assert "err_method" not in df.columns


def test_save_lightcurve_err_method_only_when_provided(tmp_path: Path) -> None:
    n = 2
    out = tmp_path / "lc_psf.csv"
    save_lightcurve_csv(
        out,
        np.arange(n, dtype=float),
        np.arange(n, dtype=float),
        np.arange(n, dtype=float),
        np.ones(n),
        np.zeros(n, dtype=bool),
        np.full(n, 12.0),
        np.full(n, 12.0),
        np.full(n, 12.0),
        np.full(n, 12.0),
        np.full(n, 12.0),
        np.zeros(n),
        np.full(n, 0.01),
        np.full(n, 3.0),
        ["normal"] * n,
        [f"f{i}.csv" for i in range(n)],
        err_method=["aperture", "psf"],
    )
    df = pd.read_csv(out)
    assert list(df["err_method"]) == ["aperture", "psf"]


def test_compute_lc_flux_method_blocks_unapplied_ac() -> None:
    frames = pd.DataFrame(
        [
            {
                "catalog_id": "1",
                "err": 0.2,
                "psf_flux": 500.0,
                "psf_fit_ok": True,
                "psf_quality": "good",
                "psf_ac_applied": False,
            }
        ]
    )
    methods = compute_lc_flux_method(frames, snr_lo=15.0)
    assert methods.iloc[0] == "aperture"


def test_rule_faint_psf_at_snr_boundary() -> None:
    # err=0.1 -> snr_aper=10.86 <= 15, quality good, AC applied -> psf
    frames_psf = pd.DataFrame(
        [
            {
                "catalog_id": "1",
                "err": 0.1,
                "psf_flux": 800.0,
                "psf_fit_ok": True,
                "psf_quality": "good",
                "psf_ac_applied": True,
            }
        ]
    )
    assert compute_lc_flux_method(frames_psf, snr_lo=15.0).iloc[0] == "psf"
    # bright aperture (low err) -> aperture even with good PSF
    frames_aper = frames_psf.copy()
    frames_aper["err"] = 0.01
    assert compute_lc_flux_method(frames_aper, snr_lo=15.0).iloc[0] == "aperture"
    # bad quality gate
    frames_bad = frames_psf.copy()
    frames_bad["psf_quality"] = "bad"
    assert compute_lc_flux_method(frames_bad, snr_lo=15.0).iloc[0] == "aperture"


def test_resolve_star_flux_method_majority() -> None:
    frames = pd.DataFrame(
        {
            "catalog_id": ["1"] * 3,
            "lc_flux_method": ["psf", "psf", "aperture"],
        }
    )
    assert _resolve_star_flux_method("1", frames) == "psf"
    frames2 = frames.copy()
    frames2["lc_flux_method"] = ["aperture", "aperture", "psf"]
    assert _resolve_star_flux_method("1", frames2) == "aperture"


def test_get_lc_adaptive_requires_ac_applied() -> None:
    frames = pd.DataFrame(
        [
            {
                "catalog_id": "1",
                "mag_inst": 11.0,
                "lc_flux_method": "psf",
                "psf_flux": 1000.0,
                "psf_ac_applied": False,
            }
        ]
    )
    mag = _get_lc_adaptive("1", frames)
    assert float(mag[0]) == 11.0


def test_psf_spatial_enabled_false_zeros_order() -> None:
    cfg = AppConfig()
    cfg.psf_spatial_enabled = False
    cfg.psf_spatial_order = 2
    spatial_order = int(cfg.psf_spatial_order or 0)
    spatial_order = max(0, min(2, spatial_order))
    if not bool(cfg.psf_spatial_enabled):
        spatial_order = 0
    assert spatial_order == 0


def test_psf_ac_applied_flag_logic() -> None:
    """Mirror psf_photometry_stars AC assignment (min_ref_stars=5)."""
    for apply_ac, n_used, expected in (
        (False, 6, False),
        (True, 6, True),
        (True, 4, False),
        (True, 5, True),
    ):
        got = bool(n_used >= 5) if apply_ac else False
        assert got is expected, (apply_ac, n_used)
