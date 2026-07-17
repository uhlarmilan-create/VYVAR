"""G5-F008: VarAstro comp count label distinct from trust n_clean."""

from export_reports import export_lightcurve_reports
import pandas as pd
from config import AppConfig


def test_varastro_header_uses_n_ensemble_comp_label(tmp_path):
    from pathlib import Path

    phot = tmp_path / "photometry"
    reports = phot / "lightcurves_reports"
    reports.mkdir(parents=True)

    lc = pd.DataFrame(
        {
            "bjd": [2460000.5],
            "mag_calib": [12.5],
            "mag_calib_final": [12.5],
            "err": [0.01],
            "flag": ["normal"],
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
    assert "varastro" in paths
    text = paths["varastro"].read_text(encoding="utf-8")
    assert "n_ensemble_comp: 7" in text
    assert "not comp_qa n_clean" in text
    assert "n_good_comp:" not in text.split("# COMP TABLE")[0]
