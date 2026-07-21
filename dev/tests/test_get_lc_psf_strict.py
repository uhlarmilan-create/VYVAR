"""Tests for _get_lc_psf_strict (LATENT-NAMES-COMPILE-GATE restore + AC guard)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from method_lc_output import _build_flux, MethodLcWriteContext
from photometry_core import _get_lc_psf_strict


def _psf_frames(
    *,
    psf_flux: float = 1000.0,
    psf_fit_ok: bool = True,
    psf_ac_applied: bool | None = True,
    n: int = 2,
) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        row: dict = {
            "catalog_id": "1496795041799526400",
            "psf_flux": psf_flux,
            "psf_fit_ok": psf_fit_ok,
        }
        if psf_ac_applied is not None:
            row["psf_ac_applied"] = psf_ac_applied
        rows.append(row)
    return pd.DataFrame(rows)


def test_get_lc_psf_strict_ok_with_ac() -> None:
    mag = _get_lc_psf_strict("1496795041799526400", _psf_frames())
    expected = -2.5 * math.log10(1000.0)
    assert len(mag) == 2
    assert np.allclose(mag, expected, equal_nan=False)


def test_get_lc_psf_strict_nan_when_ac_false() -> None:
    mag = _get_lc_psf_strict("1496795041799526400", _psf_frames(psf_ac_applied=False))
    assert len(mag) == 2
    assert np.all(np.isnan(mag))


def test_get_lc_psf_strict_nan_when_ac_column_missing() -> None:
    mag = _get_lc_psf_strict("1496795041799526400", _psf_frames(psf_ac_applied=None))
    assert len(mag) == 2
    assert np.all(np.isnan(mag))


def test_get_lc_psf_strict_nan_when_psf_columns_missing() -> None:
    df = pd.DataFrame([{"catalog_id": "1496795041799526400", "mag_inst": 12.0}])
    mag = _get_lc_psf_strict("1496795041799526400", df)
    assert len(mag) == 1
    assert np.isnan(mag[0])


def test_get_lc_psf_strict_empty_subframe() -> None:
    mag = _get_lc_psf_strict("missing", _psf_frames())
    assert mag.size == 0


def test_method_lc_output_build_flux_resolves_psf_strict(tmp_path) -> None:
    """Smoke: lazy import of _get_lc_psf_strict from photometry_core succeeds."""
    frames = _psf_frames(n=1)
    ctx = MethodLcWriteContext(
        method="psf",
        target_cid="1496795041799526400",
        comp_ids=[],
        all_frames=frames,
        lc_dir=tmp_path,
        cfg=None,
        stability_sigma=0.1,
        outlier_sigma=3.0,
        comp_catalog_mag={},
        comp_rms_map={},
        comp_tier_map={},
        tier_weights={},
        target_row=frames.iloc[0],
        state=None,
        apertures_px={},
        ac_result=None,
        comp_bp_rp={},
        target_bp_rp=float("nan"),
        bjd=np.array([2459000.0]),
        hjd=np.array([2459000.0]),
        jd=np.array([2459000.0]),
        airmass_arr=np.array([1.2]),
        flip_arr=np.array([0.0]),
        err=np.array([0.01]),
        ap_arr=np.array([3.0]),
        src_files=["f.csv"],
        sat_flags=np.array([False]),
        target_frames=frames,
        lunar_phase_pct=0.0,
        lunar_separation_deg=90.0,
        lunar_risk="low",
    )
    tlc, _ = _build_flux(ctx)
    assert np.isfinite(float(tlc[0]))
