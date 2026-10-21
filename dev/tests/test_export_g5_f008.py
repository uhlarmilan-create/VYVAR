"""G5-F008 / EXPORT-2026-10: VarAstro header states the ensemble actually used."""

from export_reports import export_lightcurve_reports
import pandas as pd
from config import AppConfig
from photometry_core import TIME_BASE_BJD_TDB


def test_varastro_header_uses_n_ensemble_comp_label(tmp_path):

    phot = tmp_path / "photometry"
    reports = phot / "lightcurves_reports"
    reports.mkdir(parents=True)

    lc = pd.DataFrame(
        {
            "jd": [2460000.5],
            "bjd": [2460000.5],
            "mag_calib": [12.5],
            "mag_calib_final": [12.5],
            "err": [0.01],
            "flag": ["normal"],
            "time_base": [TIME_BASE_BJD_TDB],
        }
    )
    target = pd.Series({"vsx_name": "EA_STAR", "vsx_type": "EA", "catalog_id": "123"})
    summary = pd.Series(
        {
            "aperture_px": 5.0,
            "n_frames": 10,
            "n_good_comp": 7,
            "n_clean": 4,
            "lc_rms": 0.02,
            "band_system": "B",
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
        obs_group="B_20_2",  # production passes the setup (OUTPUT-REVIEW step 6)
    )
    assert "varastro" in paths
    text = paths["varastro"].read_text(encoding="utf-8")
    assert "# Ensemble: 1 comparison stars" in text
    assert "#   999" in text
    assert "n_clean" not in text
