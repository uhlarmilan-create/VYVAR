"""EXPORT-2026-10: AAVSO uses #DATE=JD (UTC mid-exposure); VarAstro needs BJD_TDB."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import AppConfig
from export_reports import export_lightcurve_reports
from photometry_core import TIME_BASE_BJD_TDB, TIME_BASE_JD_FALLBACK


def _minimal_lc(*, time_base: str | None = TIME_BASE_BJD_TDB) -> pd.DataFrame:
    data = {
        "jd": [2459000.4990, 2459000.5990],
        "bjd": [2459000.5, 2459000.6],
        "mag_calib": [12.0, 12.01],
        "err": [0.02, 0.02],
        "flag": ["normal", "normal"],
        "airmass": [1.2, 1.3],
        "aperture_r_px": [5.0, 5.2],
    }
    if time_base is not None:
        data["time_base"] = [time_base, time_base]
    return pd.DataFrame(data)


def _run(tmp_path: Path, lc: pd.DataFrame, stats: dict | None = None) -> dict:
    return export_lightcurve_reports(
        tmp_path / "reports",
        pd.Series({"vsx_name": "TEST STAR", "catalog_id": "123", "vsx_type": "EA"}),
        lc,
        pd.DataFrame(),
        pd.Series({"obs_group": "V_60_2", "band_system": "V"}),
        cfg=AppConfig(),
        obs_group="V_60_2",
        export_stats=stats,
    )


def test_aavso_date_is_jd_utc(tmp_path: Path) -> None:
    paths = _run(tmp_path, _minimal_lc())
    text = paths["aavso"].read_text(encoding="ascii")
    assert "#DATE=JD\n" in text and "#DATE=BJD" not in text
    rows = [ln for ln in text.splitlines() if not ln.startswith("#")]
    assert rows[0].split(",")[1] == "2459000.499000"
    assert rows[0].split(",")[4] == "V"
    assert "APER=5.00px" in rows[0]


def test_varastro_bjd_columns(tmp_path: Path) -> None:
    paths = _run(tmp_path, _minimal_lc())
    rows = [ln for ln in paths["varastro"].read_text(encoding="ascii").splitlines() if not ln.startswith("#")]
    # REPORT-AUDIT-2026-10 (Milan 2026-10-21): VarAstro data = BJD, mag, error (3 columns)
    assert rows[0].split() == ["2459000.500000", "12.0000", "0.0217"]


@pytest.mark.parametrize("time_base", [TIME_BASE_JD_FALLBACK, None])
def test_varastro_refused_without_bjd(tmp_path: Path, time_base: str | None) -> None:
    stats: dict[str, int] = {}
    paths = _run(tmp_path, _minimal_lc(time_base=time_base), stats)
    assert "aavso" in paths  # JD is always valid for AAVSO
    assert "varastro" not in paths
    assert stats.get("time_base_refused") == 1


def test_not_on_band_scale_is_exported_as_differential(tmp_path: Path) -> None:
    # UNCAL-EXPORT-2026-10 (Milan 2026-10-11): band_system NONE -> differential export
    # (VarAstro dif mag only since REPORT-AUDIT-2026-10); was: not exported.
    def _exp(lc):
        return export_lightcurve_reports(
            tmp_path / "reports",
            pd.Series({"vsx_name": "RED STAR", "catalog_id": "1", "vsx_type": "EA"}),
            lc,
            pd.DataFrame(),
            pd.Series({"band_system": "NONE"}),
            cfg=AppConfig(),
            obs_group="g_120_4",
        )

    assert _exp(_minimal_lc()) == {}  # no delta_mag column -> nothing to export
    lc = _minimal_lc().assign(mag_calib=[float("nan")] * 2, delta_mag=[0.40, 0.41])
    stats: dict[str, int] = {}
    paths = export_lightcurve_reports(
        tmp_path / "reports2",
        pd.Series({"vsx_name": "RED STAR", "catalog_id": "1", "vsx_type": "EA"}),
        lc, pd.DataFrame(), pd.Series({"band_system": "NONE"}),
        cfg=AppConfig(), obs_group="g_120_4", export_stats=stats,
    )
    # REPORT-AUDIT-2026-10 (Milan 2026-10-21): differential -> VarAstro only, no AAVSO file
    assert "aavso" not in paths and stats.get("aavso_skipped_differential") == 1
    v = paths["varastro"].read_text()
    assert "dif_mag" in v and "DIFFERENTIAL" in v


def test_gspc_standard_export_for_note_a(tmp_path: Path) -> None:
    # GSPC-STD-2026-10 (Milan 2026-10-11, option a): band NONE + GSPC zero point -> STD on the GSPC scale
    import json

    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir()
    (lc_dir / "comp_quality_1.json").write_text(json.dumps({"band_system": "NONE", "gspc_zero_point": {
        "note": "A", "band": "g", "C": 24.8, "n_comp": 3, "sigma_data": 0.017, "comp_bp_rp_min": 2.1, "comp_bp_rp_max": 2.4}}))
    lc = _minimal_lc().assign(mag_calib=[float("nan")] * 2, delta_mag=[0.40, 0.41], mag_gspc=[15.50, 15.51])
    paths = export_lightcurve_reports(
        tmp_path / "reports", pd.Series({"vsx_name": "RED STAR", "catalog_id": "1", "vsx_type": "EA"}), lc,
        pd.DataFrame(), pd.Series({"band_system": "NONE"}), cfg=AppConfig(), obs_group="g_120_4", lc_dir=lc_dir)
    rows = [r.split(",") for r in paths["aavso"].read_text().splitlines() if r and not r.startswith("#")]
    assert [r[2] for r in rows] == ["15.500", "15.510"] and {r[6] for r in rows} == {"STD"} and {r[5] for r in rows} == {"NO"}
    assert all("ZP=GaiaDR3-GSPC" in r[14] and "ZPERR=0.017" in r[14] for r in rows)
    v = paths["varastro"].read_text()
    assert "dif_mag" not in v and "15.5000" in v


def test_varastro_field_map_copied(tmp_path: Path) -> None:
    # FIX-VARASTRO-MAP-2026-10: field chart next to the VarAstro file
    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir()
    (lc_dir / "field_map_123.png").write_bytes(b"\\x89PNG fake")
    paths = export_lightcurve_reports(
        tmp_path / "reports", pd.Series({"vsx_name": "TEST STAR", "catalog_id": "123", "vsx_type": "EA"}),
        _minimal_lc(), pd.DataFrame(), pd.Series({"band_system": "g"}), cfg=AppConfig(), obs_group="g_120_4", lc_dir=lc_dir)
    v = paths["varastro"]
    maps = list(v.parent.glob("*_field.png"))
    assert len(maps) == 1 and maps[0].name == v.stem + "_field.png"
    assert f"# FIELD IMAGE: {maps[0].name}" in v.read_text()
