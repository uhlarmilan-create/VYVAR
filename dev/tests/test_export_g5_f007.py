"""G5-F007: export plate scale derive-or-None + software version in AAVSO header."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import AppConfig
from export_reports import (
    VYVAR_SOFTWARE_VERSION,
    _aavso_software_header_line,
    _resolve_export_arcsec_per_px,
    export_lightcurve_reports,
)
from photometry_core import TIME_BASE_BJD_TDB


def test_resolve_arcsec_from_pipeline_meta_home_rig(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir(parents=True)
    (phot / "pipeline_meta.json").write_text(
        json.dumps({"plate_scale_arcsec_px": 9.77}),
        encoding="utf-8",
    )
    v = _resolve_export_arcsec_per_px(phot, AppConfig())
    assert v == 9.77


def test_resolve_arcsec_from_pipeline_meta_fine_rig(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir(parents=True)
    (phot / "pipeline_meta.json").write_text(
        json.dumps({"plate_scale_arcsec_px": 0.65}),
        encoding="utf-8",
    )
    v = _resolve_export_arcsec_per_px(phot, AppConfig())
    assert v == 0.65


def test_resolve_arcsec_none_when_not_derivable(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    phot.mkdir(parents=True)
    v = _resolve_export_arcsec_per_px(phot, AppConfig())
    assert v is None


def test_aavso_software_header_uses_version_constant() -> None:
    line = _aavso_software_header_line(VYVAR_SOFTWARE_VERSION, "aperture")
    assert line.startswith("#SOFTWARE=VYVAR/1.0")
    assert "aperture photometry" in line


def test_aavso_software_header_custom_version() -> None:
    line = _aavso_software_header_line("VYVAR 2.1-beta", "aperture")
    assert "#SOFTWARE=VYVAR/2.1-beta" in line


def test_varastro_omits_aperture_arcsec_when_scale_unknown(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    reports = phot / "lightcurves_reports"
    reports.mkdir(parents=True)

    lc = pd.DataFrame(
        {
            "bjd": [2460000.5],
            "mag_calib": [12.5],
            "err": [0.01],
            "airmass": [1.1],
            "flag": ["normal"],
            "time_base": [TIME_BASE_BJD_TDB],
        }
    )
    target = pd.Series({"vsx_name": "TEST_STAR", "vsx_type": "EA", "catalog_id": "123"})
    summary = pd.Series(
        {
            "aperture_px": 5.0,
            "fwhm_px": 3.0,
            "n_frames": 1,
            "n_good_comp": 5,
            "lc_rms": 0.02,
            "obs_group": "B_20_2",
        }
    )
    comp = pd.DataFrame({"catalog_id": ["999"], "mag": [12.0]})

    paths = export_lightcurve_reports(
        reports,
        target,
        lc,
        comp,
        summary,
        observer_code="TEST",
        cfg=AppConfig(),
    )
    assert "aavso" in paths
    aavso = paths["aavso"].read_text(encoding="utf-8")
    assert "#SOFTWARE=VYVAR/1.0" in aavso

    if "varastro" in paths:
        var = paths["varastro"].read_text(encoding="utf-8")
        assert "arcsec" not in var.lower() or "#   Aperture:" not in var


def test_varastro_shows_derived_aperture_arcsec(tmp_path: Path) -> None:
    phot = tmp_path / "photometry"
    reports = phot / "lightcurves_reports"
    reports.mkdir(parents=True)
    (phot / "pipeline_meta.json").write_text(
        json.dumps({"plate_scale_arcsec_px": 9.77}),
        encoding="utf-8",
    )

    lc = pd.DataFrame(
        {
            "bjd": [2460000.5],
            "mag_calib": [12.5],
            "err": [0.01],
            "airmass": [1.1],
            "flag": ["normal"],
            "delta_mag": [0.1],
            "time_base": [TIME_BASE_BJD_TDB],
        }
    )
    target = pd.Series({"vsx_name": "TEST_EA", "vsx_type": "EA", "catalog_id": "123"})
    summary = pd.Series(
        {
            "aperture_px": 5.0,
            "fwhm_px": 3.0,
            "n_frames": 1,
            "n_good_comp": 5,
            "lc_rms": 0.02,
            "obs_group": "B_20_2",
        }
    )
    comp = pd.DataFrame({"catalog_id": ["999"], "mag": [12.0]})

    paths = export_lightcurve_reports(
        reports,
        target,
        lc,
        comp,
        summary,
        observer_code="TEST",
        cfg=AppConfig(),
    )
    if "varastro" in paths:
        var = paths["varastro"].read_text(encoding="utf-8")
        assert "48.85arcsec" in var  # 5.0 * 9.77


def test_aavso_body_byte_identical_when_scale_matches_meta(tmp_path: Path) -> None:
    """Regression: data rows unchanged when meta carries the true plate scale."""
    phot = tmp_path / "photometry"
    reports = phot / "lightcurves_reports"
    reports.mkdir(parents=True)
    (phot / "pipeline_meta.json").write_text(
        json.dumps({"plate_scale_arcsec_px": 9.77}),
        encoding="utf-8",
    )

    lc = pd.DataFrame(
        {
            "bjd": [2460000.500000],
            "mag_calib": [12.500],
            "err": [0.010],
            "airmass": [1.100],
            "flag": ["normal"],
            "time_base": [TIME_BASE_BJD_TDB],
        }
    )
    target = pd.Series({"vsx_name": "REG_STAR", "vsx_type": "EA", "catalog_id": "456"})
    summary = pd.Series(
        {
            "aperture_px": 5.0,
            "fwhm_px": 3.0,
            "n_frames": 1,
            "n_good_comp": 5,
            "lc_rms": 0.02,
            "obs_group": "B_20_2",
        }
    )
    comp = pd.DataFrame({"catalog_id": ["999"], "mag": [12.0]})

    paths = export_lightcurve_reports(
        reports,
        target,
        lc,
        comp,
        summary,
        observer_code="TEST",
        cfg=AppConfig(),
    )
    aavso = paths["aavso"].read_text(encoding="utf-8")
    data_lines = [ln for ln in aavso.splitlines() if ln and not ln.startswith("#")]
    assert len(data_lines) == 1
    assert data_lines[0].startswith("REG_STAR,2460000.500000,12.500,0.010,")
