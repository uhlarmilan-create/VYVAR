"""B1: AAVSO export must not declare BJD when time_base is JD_FALLBACK or absent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import AppConfig
from export_reports import export_lightcurve_reports
from photometry_core import TIME_BASE_BJD_TDB, TIME_BASE_JD_FALLBACK


def _minimal_lc(*, time_base: str | None = TIME_BASE_BJD_TDB) -> pd.DataFrame:
    data = {
        "bjd": [2459000.5, 2459000.6],
        "mag_calib": [12.0, 12.01],
        "err": [0.02, 0.02],
        "flag": ["normal", "normal"],
        "airmass": [1.2, 1.3],
    }
    if time_base is not None:
        data["time_base"] = [time_base, time_base]
    return pd.DataFrame(data)


def test_aavso_writes_date_bjd_when_time_base_bjd_tdb(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    target = pd.Series({"vsx_name": "TEST STAR", "catalog_id": "123", "vsx_type": "EA"})
    summary = pd.Series({"obs_group": "NoFilter_60_2"})
    paths = export_lightcurve_reports(
        out,
        target,
        _minimal_lc(time_base=TIME_BASE_BJD_TDB),
        pd.DataFrame(),
        summary,
        cfg=AppConfig(),
    )
    assert "aavso" in paths
    text = paths["aavso"].read_text(encoding="utf-8")
    assert "#DATE=BJD" in text


@pytest.mark.parametrize(
    "time_base",
    [TIME_BASE_JD_FALLBACK, None],
)
def test_aavso_refuses_non_bjd_time_base(tmp_path: Path, time_base: str | None) -> None:
    out = tmp_path / "reports"
    target = pd.Series({"vsx_name": "TEST STAR", "catalog_id": "123", "vsx_type": "EA"})
    summary = pd.Series({"obs_group": "NoFilter_60_2"})
    stats: dict[str, int] = {}
    paths = export_lightcurve_reports(
        out,
        target,
        _minimal_lc(time_base=time_base),
        pd.DataFrame(),
        summary,
        cfg=AppConfig(),
        export_stats=stats,
    )
    assert paths == {}
    assert stats.get("time_base_refused") == 1
    aavso_dir = out / "aavso"
    if aavso_dir.is_dir():
        assert list(aavso_dir.glob("*.txt")) == []


def test_aavso_refuses_mixed_time_base(tmp_path: Path) -> None:
    lc = _minimal_lc(time_base=TIME_BASE_BJD_TDB)
    lc.loc[1, "time_base"] = TIME_BASE_JD_FALLBACK
    paths = export_lightcurve_reports(
        tmp_path / "reports",
        pd.Series({"vsx_name": "MIX", "catalog_id": "1", "vsx_type": "EA"}),
        lc,
        pd.DataFrame(),
        pd.Series({}),
        cfg=AppConfig(),
        export_stats={},
    )
    assert paths == {}
