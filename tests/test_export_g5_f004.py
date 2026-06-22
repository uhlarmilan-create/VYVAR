"""G5-F004: export failures logged at ERROR and collected; batch still completes."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from config import AppConfig
from export_reports import (
    export_all_method_lightcurve_reports,
    export_lightcurve_reports,
    log_export_batch_summary,
    record_export_failure,
)


def _minimal_lc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bjd": [2460000.5],
            "mag_calib": [12.5],
            "mag_calib_final": [12.5],
            "err": [0.01],
            "flag": ["normal"],
        }
    )


def _minimal_target(catalog_id: str = "222") -> pd.Series:
    return pd.Series(
        {
            "vsx_name": "GOOD_STAR",
            "vsx_type": "EA",
            "catalog_id": catalog_id,
        }
    )


def _minimal_summary() -> pd.Series:
    return pd.Series(
        {
            "aperture_px": 5.0,
            "n_frames": 10,
            "n_good_comp": 5,
            "lc_rms": 0.02,
            "obs_group": "B_20_2",
        }
    )


def test_empty_lc_records_failure_and_returns_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    reports = tmp_path / "lightcurves_reports"
    reports.mkdir(parents=True)
    failures: list[dict[str, str]] = []

    with caplog.at_level(logging.ERROR):
        paths = export_lightcurve_reports(
            reports,
            _minimal_target("111"),
            pd.DataFrame(),
            pd.DataFrame({"catalog_id": ["999"], "mag": [12.0]}),
            _minimal_summary(),
            observer_code="TEST",
            cfg=AppConfig(),
            export_failures=failures,
        )

    assert paths == {}
    assert len(failures) == 1
    assert failures[0]["target_id"] == "111"
    assert "no exportable LC" in failures[0]["reason"]
    assert any("[EXPORT] failed" in r.message for r in caplog.records)


def test_read_error_and_success_in_same_batch(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    phot = tmp_path / "photometry"
    lc_dir = phot / "lightcurves"
    reports = phot / "lightcurves_reports"
    lc_dir.mkdir(parents=True)
    reports.mkdir(parents=True)

    bad_cid = "1111111111111111111"
    good_cid = "2222222222222222222"
    (lc_dir / f"lightcurve_{bad_cid}.csv").write_text("not,a,valid\ncsv{{{", encoding="utf-8")
    _minimal_lc().to_csv(lc_dir / f"lightcurve_{good_cid}.csv", index=False)

    failures: list[dict[str, str]] = []
    comp = pd.DataFrame({"catalog_id": ["999"], "mag": [12.0]})

    with caplog.at_level(logging.ERROR):
        bad_paths = export_all_method_lightcurve_reports(
            reports,
            _minimal_target(bad_cid),
            lc_dir=lc_dir,
            target_cid=bad_cid,
            comp_df=comp,
            summary_row=_minimal_summary(),
            observer_code="TEST",
            cfg=AppConfig(),
            export_failures=failures,
        )
        good_paths = export_all_method_lightcurve_reports(
            reports,
            _minimal_target(good_cid),
            lc_dir=lc_dir,
            target_cid=good_cid,
            comp_df=comp,
            summary_row=_minimal_summary(),
            observer_code="TEST",
            cfg=AppConfig(),
            export_failures=failures,
        )
        log_export_batch_summary(failures)

    assert bad_paths == {}
    assert good_paths
    assert "aavso" in good_paths["aperture"]
    assert any(f["target_id"] == bad_cid for f in failures)
    assert any("batch finished with" in r.message for r in caplog.records)


def test_clean_export_byte_identical_without_failure_collector(tmp_path: Path) -> None:
    """Successful export output unchanged when export_failures is not passed."""
    reports = tmp_path / "lightcurves_reports"
    reports.mkdir(parents=True)
    lc = _minimal_lc()
    target = _minimal_target()
    summary = _minimal_summary()
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
    aavso_text = paths["aavso"].read_text(encoding="utf-8")
    var_text = paths["varastro"].read_text(encoding="utf-8")

    paths2 = export_lightcurve_reports(
        reports,
        target,
        lc,
        comp,
        summary,
        observer_code="TEST",
        cfg=AppConfig(),
        export_failures=[],
    )
    assert paths2["aavso"].read_text(encoding="utf-8") == aavso_text
    assert paths2["varastro"].read_text(encoding="utf-8") == var_text


def test_log_export_batch_summary_empty_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        log_export_batch_summary([])
    assert not caplog.records


def test_record_export_failure_without_collector(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        record_export_failure(None, "123", "aperture", "test reason")
    assert any("test reason" in r.message for r in caplog.records)
