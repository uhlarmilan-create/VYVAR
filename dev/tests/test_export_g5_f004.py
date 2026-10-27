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
from photometry_core import TIME_BASE_BJD_TDB


def _minimal_lc() -> pd.DataFrame:
    return pd.DataFrame(
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
            "band_system": "B",
        }
    )


def test_empty_lc_is_not_exported_by_rule_at_info(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """EXPORT-LOG-2026-10: no usable LC points is a rule, not a failure (was logged as ERROR)."""
    reports = tmp_path / "lightcurves_reports"
    reports.mkdir(parents=True)
    failures: list[dict[str, str]] = []
    stats: dict[str, int] = {}

    with caplog.at_level(logging.INFO):
        paths = export_lightcurve_reports(
            reports,
            _minimal_target("111"),
            pd.DataFrame(),
            pd.DataFrame({"catalog_id": ["999"], "mag": [12.0]}),
            _minimal_summary(),
            observer_code="TEST",
            cfg=AppConfig(),
            obs_group="B_20_2",  # production passes the setup (OUTPUT-REVIEW step 6)
            export_failures=failures,
            export_stats=stats,
        )
        log_export_batch_summary(failures, stats)

    assert paths == {}
    assert failures == []
    assert stats["not_exported_by_rule"] == 1
    rule = [r for r in caplog.records if "not exported by rule 111" in r.getMessage()]
    assert rule and all(r.levelno == logging.INFO for r in rule)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and "[EXPORT]" in r.getMessage()]


def test_export_problem_levels(caplog: pytest.LogCaptureFixture) -> None:
    """Attention items are WARNING, real failures ERROR; the summary takes the highest level."""
    failures: list[dict[str, str]] = []
    with caplog.at_level(logging.INFO):
        record_export_failure(failures, "1", "aperture", "VarAstro: time_base=None (BJD_TDB required)",
                              level=logging.WARNING)
        log_export_batch_summary(failures)
    assert [r.levelno for r in caplog.records] == [logging.WARNING] * 3
    caplog.clear()
    with caplog.at_level(logging.INFO):
        record_export_failure(failures, "2", "aperture", "AAVSO write error: disk full")
        log_export_batch_summary(failures)
    assert caplog.records[-3].levelno == logging.ERROR and "batch finished" in caplog.records[-3].getMessage()
    assert [f["level"] for f in failures] == ["WARNING", "ERROR"]


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
            obs_group="B_20_2",  # production passes the setup (OUTPUT-REVIEW step 6)
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
            obs_group="B_20_2",  # production passes the setup (OUTPUT-REVIEW step 6)
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
        obs_group="B_20_2",  # production passes the setup (OUTPUT-REVIEW step 6)
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
        obs_group="B_20_2",  # production passes the setup (OUTPUT-REVIEW step 6)
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
