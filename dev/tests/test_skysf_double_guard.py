"""SKYSF-DOUBLE guard: idempotent in-place sky-surface subtract (T1-T6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from config import AppConfig
from pipeline import (
    SkySurfaceOrderConflictError,
    _decide_preprocess_sky_action,
    _header_has_vy_skysf,
    _qc_enrich_calibrated_in_place,
    _qc_enrich_one_frame,
    qc_enrich_calibrated_lights_in_place,
    preprocess_sky_summary_from_df,
)


def _synthetic_gradient_with_stars(shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    grad = 1200.0 + 0.15 * xx.astype(np.float32) + 0.08 * yy.astype(np.float32)
    stars = np.zeros_like(grad)
    for cy, cx, amp in ((40, 45, 800.0), (85, 100, 1200.0), (60, 90, 600.0)):
        stars += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * 2.0**2)))
    return (grad + stars).astype(np.float32)


def _write_mono_light(path: Path, *, hdr_extra: fits.Header | None = None) -> None:
    data = _synthetic_gradient_with_stars()
    hdr = fits.Header()
    if hdr_extra is not None:
        hdr.update(hdr_extra)
    fits.writeto(path, data, hdr, overwrite=True)


def _read_data(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float32)


def test_t1_no_marker_subtract_and_write_headers(tmp_path: Path) -> None:
    fp = tmp_path / "light.fits"
    _write_mono_light(fp)
    cfg = AppConfig()
    cfg.preprocess_sky_surface_order = 2
    _qc_enrich_calibrated_in_place(tmp_path, app_config=cfg)
    with fits.open(fp) as hdul:
        hdr = hdul[0].header
        assert _header_has_vy_skysf(hdr)
        assert int(hdr["VYSKYORD"]) == 2
        assert float(hdr["VYSKYP2P"]) > 0.0


def test_t2_second_pass_same_order_skips_byte_identical(tmp_path: Path) -> None:
    fp = tmp_path / "light.fits"
    _write_mono_light(fp)
    cfg = AppConfig()
    cfg.preprocess_sky_surface_order = 2
    _qc_enrich_calibrated_in_place(tmp_path, app_config=cfg)
    after_first = _read_data(fp)
    out2 = _qc_enrich_calibrated_in_place(tmp_path, app_config=cfg)
    after_second = _read_data(fp)
    np.testing.assert_array_equal(after_first, after_second)
    assert int(out2.get("sky_surface_skip_count") or 0) == 1
    row = out2["results"][0]
    assert row.get("sky_surface_skipped") is True


def test_t3_order_mismatch_aborts_with_recal_message(tmp_path: Path) -> None:
    fp = tmp_path / "light.fits"
    _write_mono_light(fp)
    with fits.open(fp, mode="update") as hdul:
        hdul[0].header["VY_SKYSF"] = (True, "")
        hdul[0].header["VYSKYORD"] = (1, "")
        hdul.flush()
    cfg = AppConfig()
    cfg.preprocess_sky_surface_order = 2
    with pytest.raises(SkySurfaceOrderConflictError, match="recalibration from raw"):
        _qc_enrich_calibrated_in_place(tmp_path, app_config=cfg)


def test_t3_unit_decide_raises_order_conflict() -> None:
    hdr = fits.Header()
    hdr["VY_SKYSF"] = True
    hdr["VYSKYORD"] = 1
    with pytest.raises(SkySurfaceOrderConflictError, match="recalibration from raw"):
        _decide_preprocess_sky_action(hdr, sky_order=2, force_reapply=False)


def test_t4_force_reapply_second_subtract(tmp_path: Path) -> None:
    fp = tmp_path / "light.fits"
    _write_mono_light(fp)
    cfg = AppConfig()
    cfg.preprocess_sky_surface_order = 2
    _qc_enrich_calibrated_in_place(tmp_path, app_config=cfg)
    after_first = _read_data(fp)
    cfg.preprocess_sky_surface_force_reapply = True
    out = _qc_enrich_calibrated_in_place(tmp_path, app_config=cfg)
    after_force = _read_data(fp)
    assert not np.array_equal(after_first, after_force)
    assert out.get("sky_surface_force_reapply") is True
    assert out["results"][0].get("sky_surface_force_reapply") is True


def test_t5_legacy_copy_tree_calibrated_no_markers_subtracts(tmp_path: Path) -> None:
    """Copy-tree-era calibrated FITS: no VY_SKYSF; guard must subtract on first pass."""
    fp = tmp_path / "legacy_cal.fits"
    _write_mono_light(fp)
    with fits.open(fp) as hdul:
        assert "VY_SKYSF" not in hdul[0].header
    cfg = AppConfig()
    cfg.preprocess_sky_surface_order = 2
    out = _qc_enrich_calibrated_in_place(tmp_path, app_config=cfg)
    with fits.open(fp) as hdul:
        assert _header_has_vy_skysf(hdul[0].header)
    assert out["results"][0].get("sky_surface_applied") is True
    assert int(out.get("sky_surface_skip_count") or 0) == 0


def test_t6_skip_counter_in_preprocess_summary_not_log_only(tmp_path: Path) -> None:
    fp = tmp_path / "light.fits"
    _write_mono_light(fp)
    cfg = AppConfig()
    cfg.preprocess_sky_surface_order = 2
    _qc_enrich_calibrated_in_place(tmp_path, app_config=cfg)
    df = qc_enrich_calibrated_lights_in_place(
        calibrated_root=tmp_path,
        app_config=cfg,
    )
    summary = preprocess_sky_summary_from_df(df)
    assert summary["sky_surface_skip_count"] == 1
    job_summary = {
        "kind": "preprocess",
        "rows": int(len(df)),
        "sky_surface_skip_count": summary["sky_surface_skip_count"],
    }
    assert job_summary["sky_surface_skip_count"] == 1


def test_guard_reads_marker_from_frame_being_modified_not_sibling(tmp_path: Path) -> None:
    """Sibling with VY_SKYSF must not suppress subtract on an unmarked frame."""
    marked = tmp_path / "marked.fits"
    bare = tmp_path / "bare.fits"
    _write_mono_light(marked)
    _write_mono_light(bare)
    with fits.open(marked, mode="update") as hdul:
        hdul[0].header["VY_SKYSF"] = (True, "")
        hdul[0].header["VYSKYORD"] = (2, "")
        hdul.flush()
    row = _qc_enrich_one_frame(
        str(bare),
        sky_order=2,
        force_reapply=False,
        prefilter_status=None,
        target_ra=None,
        target_dec=None,
        inject_pointing_only_if_missing=True,
    )
    assert row.get("sky_surface_applied") is True
    with fits.open(bare) as hdul:
        assert _header_has_vy_skysf(hdul[0].header)
