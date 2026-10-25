"""CAMERA-PROFILE-2026-10: gain / read noise from the header, the camera's curve and the catalog."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from astropy.io import fits

from camera_profile import (
    CameraProfile,
    CurvePoint,
    catalog_model,
    catalog_models,
    load_profile,
    lookup,
    match_catalog_model,
    match_catalog_models,
    scale_for_binning,
)
from param_resolver import resolve_gain, resolve_read_noise


def _hdr(binning: int = 1, **cards: tuple[float, str]) -> fits.Header:
    h = fits.Header()
    for key, (val, comment) in cards.items():
        h[key] = (val, comment)
    h["XBINNING"] = binning
    h["YBINNING"] = binning
    return h


def _qhy294(**kw) -> CameraProfile:
    base = dict(
        equipment_id=1, name="QHY294MM", bin_mode="sum", header_gain="setting", default_setting=0.0,
        adc_bits=14, points=(CurvePoint(0.0, 1, 3.17, 7.6, "equipment"),),
    )
    base.update(kw)
    return CameraProfile(**base)


# --------------------------------------------------------------------------- binning conventions
@pytest.mark.parametrize(
    "mode,gain,rn",
    [("average", 0.78 * 16, 3.52 * 4), ("sum", 0.78, 3.52 * 4), ("charge", 0.78, 3.52)],
)
def test_binning_conventions(mode: str, gain: float, rn: float) -> None:
    assert scale_for_binning(0.78, 4, mode, "gain") == pytest.approx(gain)
    assert scale_for_binning(3.52, 4, mode, "read_noise") == pytest.approx(rn)
    assert scale_for_binning(0.78, 1, mode, "gain") == pytest.approx(0.78)


# --------------------------------------------------------------------------- header e-/ADU
def test_header_e_per_adu_wins_and_cross_checks_against_profile() -> None:
    prof = CameraProfile(equipment_id=4, bin_mode="average", points=(CurvePoint(None, 1, 0.78, 3.52, "equipment"),))
    res = resolve_gain(_hdr(4, GAIN=(12.48, "e-/ADU")), profile=prof)
    assert res.ok and res.source == "header" and res.value == pytest.approx(12.48)
    assert not res.warnings  # 0.78 x 16 = 12.48: the profile agrees
    bad = CameraProfile(equipment_id=4, bin_mode="average", points=(CurvePoint(None, 1, 1.0, 3.52, "equipment"),))
    res = resolve_gain(_hdr(4, GAIN=(12.48, "e-/ADU")), profile=bad)
    assert res.value == pytest.approx(12.48) and "disagrees" in res.warnings[0]


def test_egain_preferred_over_gain_setting() -> None:
    res = resolve_gain(_hdr(EGAIN=(2.5, "e-/ADU"), GAIN=(99.0, "Gain")), profile=_qhy294(header_gain=None))
    assert res.ok and res.source == "header" and res.key == "EGAIN" and res.value == pytest.approx(2.5)


def test_camera_header_gain_kind_overrides_the_comment() -> None:
    prof = CameraProfile(equipment_id=9, header_gain="e_per_adu", points=())
    res = resolve_gain(_hdr(GAIN=(1.5, "Gain")), profile=prof)
    assert res.source == "header" and res.value == pytest.approx(1.5)


# --------------------------------------------------------------------------- header gain setting
def test_qhy294_setting_zero_bin2_keeps_the_earlier_values() -> None:
    hdr = _hdr(2, GAIN=(0.0, "Gain"))
    g = resolve_gain(hdr, profile=_qhy294())
    rn = resolve_read_noise(hdr, profile=_qhy294())
    assert g.value == pytest.approx(3.17) and g.source == "equipment" and g.key == "GAIN=0"
    assert rn.value == pytest.approx(15.2) and rn.source == "equipment"


def test_setting_without_a_point_is_unresolved_not_guessed() -> None:
    res = resolve_gain(_hdr(2, GAIN=(56.0, "Gain")), profile=_qhy294())
    assert not res.ok and res.value is None and "56" in res.warnings[0]


def test_default_point_without_default_setting_serves_any_setting_with_a_warning() -> None:
    prof = _qhy294(default_setting=None, points=(CurvePoint(None, 1, 3.17, 7.6, "equipment"),))
    res = resolve_gain(_hdr(2, GAIN=(56.0, "Gain")), profile=prof)
    assert res.ok and res.value == pytest.approx(3.17) and "no default setting" in res.warnings[0]


def test_no_setting_in_header_uses_the_default_setting() -> None:
    res = resolve_gain(_hdr(2), profile=_qhy294())
    assert res.ok and res.value == pytest.approx(3.17)


def test_measured_beats_manual_beats_equipment_and_exact_binning_is_not_scaled() -> None:
    prof = _qhy294(points=(
        CurvePoint(0.0, 1, 3.17, 7.6, "equipment"),
        CurvePoint(0.0, 1, 3.0, 7.0, "manual"),
        CurvePoint(0.0, 2, 3.21, 7.4, "measured"),
    ))
    hdr = _hdr(2, GAIN=(0.0, "Gain"))
    assert resolve_gain(hdr, profile=prof).value == pytest.approx(3.21)
    assert resolve_read_noise(hdr, profile=prof).value == pytest.approx(7.4)  # measured at bin 2: as is
    hdr1 = _hdr(1, GAIN=(0.0, "Gain"))
    assert resolve_gain(hdr1, profile=prof).value == pytest.approx(3.0)  # no bin-1 measurement: manual


def test_catalog_curve_interpolates_gain_log_linear_and_never_extrapolates() -> None:
    prof = CameraProfile(equipment_id=5, header_gain="setting", bin_mode="sum", points=(
        CurvePoint(0.0, 1, 4.0, 3.8, "catalog"), CurvePoint(200.0, 1, 1.0, 1.5, "catalog"),
    ))
    mid = lookup(prof, "gain", 100.0, 1)
    assert mid.source == "catalog" and mid.value == pytest.approx(2.0)  # geometric mean
    assert lookup(prof, "read_noise", 100.0, 1).value == pytest.approx(2.65)
    assert lookup(prof, "gain", 300.0, 1).value is None


def test_read_noise_header_card_only_as_fallback() -> None:
    prof = CameraProfile(equipment_id=3, points=())
    res = resolve_read_noise(_hdr(RDNOISE=(5.0, "e-")), profile=prof)
    assert res.ok and res.source == "header" and res.value == pytest.approx(5.0)
    res = resolve_read_noise(_hdr(VY_RDNOIS=(2.2, "effective")), profile=_qhy294())
    assert res.source == "header" and res.value == pytest.approx(2.2)


def test_nothing_known_is_unresolved() -> None:
    assert not resolve_gain(fits.Header(), profile=CameraProfile()).ok
    assert not resolve_read_noise(None, profile=CameraProfile()).ok


# --------------------------------------------------------------------------- database + catalog
def _db(tmp_path: Path) -> Path:
    from database import VyvarDatabase

    p = tmp_path / "v.sqlite3"
    db = VyvarDatabase(p)
    db.conn.execute("DELETE FROM EQUIPMENTS")
    db.conn.execute(
        "INSERT INTO EQUIPMENTS (ID, CAMERANAME, GAIN_ADU, READNOISE_E, ADC_BITS, MODEL_KEY, READOUT_MODE, "
        "BIN_MODE, DEFAULT_GAIN_SETTING) VALUES (1, 'QHY294MM', 3.17, 7.6, 14, 'qhy294m_pro', '11mp', 'sum', 0)"
    )
    db.conn.execute(
        "INSERT INTO EQUIPMENTS (ID, CAMERANAME, GAIN_ADU, READNOISE_E, ADC_BITS, MODEL_KEY) "
        "VALUES (4, 'C5A-150M', 0.78, 3.52, 16, 'moravian_c5a_150m')"
    )
    db.conn.execute(
        "INSERT INTO EQUIPMENT_CURVE (EQUIPMENT_ID, READOUT_MODE, GAIN_SETTING, BINNING, E_PER_ADU, SOURCE) "
        "VALUES (1, '47mp', 0, 1, 9.9, 'measured')"
    )
    db.conn.commit()
    db.conn.close()
    return p


def test_profile_from_database_and_catalog(tmp_path: Path) -> None:
    p = _db(tmp_path)
    qhy = load_profile(p, 1)
    assert qhy.model_key == "qhy_qhy294m_pro" and qhy.mode_key == "11mp" and qhy.bin_mode == "sum"  # legacy key
    assert not any(pt.e_per_adu == pytest.approx(9.9) for pt in qhy.points)  # other readout mode
    assert any(pt.source == "catalog" and pt.setting == 1600 for pt in qhy.points)
    c5a = load_profile(p, 4)
    assert c5a.header_gain == "e_per_adu" and c5a.bin_mode == "average" and c5a.bin_mode_source == "catalog"


def test_photometric_gain_regression_qhy294_and_c5a(tmp_path: Path) -> None:
    from config import AppConfig
    from rig_photometry import photometric_gain_read_noise

    cfg = AppConfig()
    cfg.database_path = str(_db(tmp_path))
    pg = photometric_gain_read_noise(_hdr(2, GAIN=(0.0, "Gain")), cfg, 1)
    assert (pg.gain_native, pg.gain, pg.read_noise, pg.container_scale) == pytest.approx((3.17, 0.7925, 15.2, 4.0))
    pg = photometric_gain_read_noise(_hdr(4, GAIN=(12.48, "e-/ADU")), cfg, 4)
    assert (pg.gain, pg.read_noise, pg.container_scale) == pytest.approx((12.48, 14.08, 1.0))
    with pytest.raises(ValueError, match="gain / read noise unknown"):
        photometric_gain_read_noise(_hdr(2, GAIN=(56.0, "Gain")), cfg, 1)


def test_catalog_is_consistent() -> None:
    models = catalog_models()
    keys = [m["key"] for m in models]
    assert len(keys) == len(set(keys)) and len(keys) >= 300
    for m in models:
        assert m.get("header_gain") in ("e_per_adu", "setting")
        assert m.get("sources"), m["key"]
        for mode in m["modes"]:
            assert mode.get("binning") in (None, "average", "sum", "charge")
            assert mode.get("adc_bits") is None or int(mode["adc_bits"]) in range(8, 21)
            for pt in mode.get("curve", []):
                assert pt.get("source"), (m["key"], pt)
                assert pt.get("e_per_adu") is not None or pt.get("read_noise_e") is not None
    assert match_catalog_model("QHY CCD QHY294PROM")["key"] == "qhy_qhy294m_pro"
    for legacy in ("qhy294m_pro", "qhy600m", "moravian_c3_26000", "moravian_c5a_150m", "zwo_asi533mc_pro"):
        assert catalog_model(legacy) is not None, legacy  # keys stored in existing databases
    assert match_catalog_model("some unknown camera") is None


def test_no_hard_coded_camera_ids_left() -> None:
    import param_resolver

    assert not hasattr(param_resolver, "GAIN_SETTING_INDEX_MAP")


def test_schema_on_an_old_database(tmp_path: Path) -> None:
    con = sqlite3.connect(tmp_path / "old.sqlite3")
    con.execute(
        "CREATE TABLE EQUIPMENTS (ID INTEGER PRIMARY KEY, CAMERANAME TEXT, ALIAS TEXT, SENSORTYPE TEXT, "
        "SENSORSIZE TEXT, PIXELSIZE REAL, ACTIVE TEXT, GAIN_ADU REAL)"
    )
    con.commit()
    con.close()
    from database import VyvarDatabase

    db = VyvarDatabase(tmp_path / "old.sqlite3")
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info('EQUIPMENTS')")}
    assert {"MODEL_KEY", "READOUT_MODE", "BIN_MODE", "HEADER_GAIN", "DEFAULT_GAIN_SETTING"} <= cols
    assert db.conn.execute("SELECT name FROM sqlite_master WHERE name='EQUIPMENT_CURVE'").fetchone()


# --------------------------------------------------------------------------- error-model check
def _check_sidecars(tmp_path: Path, scale: float, n_stars: int = 6) -> Path:
    import numpy as np
    import pandas as pd

    lc = tmp_path / "lightcurves"
    lc.mkdir()
    rng = np.random.default_rng(2)
    for i in range(n_stars):
        err = np.full(200, 0.01)
        pd.DataFrame({"kmag": 10 + rng.normal(0, 0.01 * scale, 200), "kmag_err": err}).to_csv(
            lc / f"check_kmag_{1000 + i}.csv", index=False
        )
    return tmp_path


@pytest.mark.parametrize("scale,status", [(1.0, "ok"), (0.5, "low"), (2.0, "high")])
def test_error_model_check(tmp_path: Path, scale: float, status: str) -> None:
    from trust_flag_core import error_model_check

    chk = error_model_check(_check_sidecars(tmp_path, scale))
    assert chk["n"] == 6 and chk["status"] == status and chk["ratio"] == pytest.approx(scale, rel=0.15)


def test_error_model_check_in_report_rows(tmp_path: Path) -> None:
    from photometry_report import resolved_facts_model

    m = resolved_facts_model({"resolved_facts": {"gain": {"value": 0.79, "source": "catalog"}}}, _check_sidecars(tmp_path, 0.5))
    row = next(r for r in m["rows"] if r["label"] == "Error-model check")
    assert row["source"] == "errors too large"
    assert any("Measure the gain" in w for w in m["warnings"])
    gain_row = next(r for r in m["rows"] if r["label"] == "Gain (e-/ADU)")
    assert gain_row["source"].startswith("camera catalog")


def test_mixed_gain_settings_in_one_group_are_refused() -> None:
    from importer import mixed_camera_settings_message

    assert mixed_camera_settings_message("NoFilter|60|2", {(0, None, None): 140}) is None
    msg = mixed_camera_settings_message("NoFilter|60|2", {(0, None, None): 62, (100, None, None): 78})
    assert "gain 100: 78 frames" in msg and "gain 0: 62 frames" in msg and "separately" in msg


def test_fixed_gain_readout_mode_value() -> None:
    from camera_profile import _catalog_points

    pts = _catalog_points({"key": "hdr", "e_per_adu": 0.8, "read_noise_e": 1.6})
    prof = CameraProfile(equipment_id=7, mode_key="hdr", bin_mode="sum", points=tuple(pts))
    assert lookup(prof, "gain", 5.0, 1).value == pytest.approx(0.8)
    assert lookup(prof, "read_noise", None, 2).value == pytest.approx(3.2)


# INSTRUME as real capture programs write it. INDI: device name = driver prefix + SDK model
# (indi-3rdparty asi_ccd_hotplug_handler.cpp "ZWO CCD ", qhy_ccd.cpp "QHY CCD %.15s" of the SDK id,
# mi_ccd.cpp "MI %s", playerone_single_ccd.h "PlayerOne ", indi_toupbase.cpp DNAME + " "), a counter
# for a second identical camera; N.I.N.A.: QHY = SDK short model (QHYCamera.cs), ZWO = SDK name.
@pytest.mark.parametrize(
    ("instrume", "key"),
    [
        ("QHY CCD QHY294PROM-cfd3", "qhy_qhy294m_pro"),  # Milan's real header (draft_000532)
        ("QHY CCD QHY294PROM", "qhy_qhy294m_pro"),  # Milan's real header (draft_000537)
        ("C5A-150M", "moravian_c5a_150m"),  # Milan's real header (draft_000534)
        ("QHY294PROM", "qhy_qhy294m_pro"),
        ("QHY CCD QHY600M-1d84b26", "qhy_qhy600m_ph"),
        ("QHY CCD QHY268M 2", "qhy_qhy268m_ph"),
        ("ZWO ASI2600MM Pro", "zwo_asi2600mm_pro"),
        ("ZWO CCD ASI2600MM Pro", "zwo_asi2600mm_pro"),
        ("ZWO CCD ASI2600MM Pro 1", "zwo_asi2600mm_pro"),
        ("MI C3-61000", "moravian_c3_61000"),
        ("MI C3-26000C", "moravian_c3_26000c"),
        ("MI C1-12000A", "moravian_c1_12000a"),
        ("C1+3000", "moravian_c1plus_3000"),
        ("C1-3000", "moravian_c1_3000"),
        ("PlayerOne Poseidon-M Pro", "player_one_poseidon_m_pro"),
        ("ToupTek ATR3CMOS26000KMA", "touptek_atr2600m"),
    ],
)
def test_header_names_from_capture_programs(instrume: str, key: str) -> None:
    hit = match_catalog_model(instrume)
    assert hit is not None and hit["key"] == key, (instrume, hit and hit["key"])


def test_ambiguous_or_generic_header_names_are_not_guessed() -> None:
    # The mono and the colour Atik Horizon report the same name: no single model is chosen ...
    assert match_catalog_model("Atik Horizon") is None
    assert {m["key"] for m in match_catalog_models("Atik Horizon")} == {"atik_horizon_mono", "atik_horizon_color"}
    # ... and driver names that carry no model stay unmatched.
    for generic in ("ASI Camera (1)", "SBIG CCD", "FLI CCD", "MI Camera", ""):
        assert match_catalog_models(generic) == []


def test_every_catalog_alias_matches_its_model_when_decorated() -> None:
    prefixes = {"ZWO": ("ZWO CCD ", "ZWO "), "QHYCCD": ("QHY CCD ", ""), "Moravian Instruments": ("MI ", "Moravian "),
                "Player One": ("PlayerOne ", "Player One "), "ToupTek": ("ToupTek ", "ToupTek ")}
    for m in catalog_models():
        new, old = prefixes.get(m["vendor"], ("", ""))
        for alias in (m["model"], *m["aliases"]):
            core = alias[len(old):] if old and alias.startswith(old) else alias
            forms = [alias, new + core] + ([new + core + "-1d84b26"] if m["vendor"] == "QHYCCD" else [])
            for f in forms:
                assert m["key"] in {x["key"] for x in match_catalog_models(f)}, (m["key"], f)


def test_missing_catalog_is_reported_not_silent(tmp_path, caplog) -> None:
    import logging

    from camera_profile import catalog_missing_message, load_catalog

    gone = tmp_path / "camera_catalog.json"
    with caplog.at_level(logging.WARNING):
        assert load_catalog(gone) == {"models": []}
    assert "camera catalog camera_catalog.json is missing" in caplog.text
    assert catalog_missing_message(gone) and catalog_missing_message() is None  # the shipped one exists


def test_release_bundle_ships_the_camera_catalog() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "cython_release" / "bundle"))
    import bundle_layout as bl

    assert "src_py/camera_catalog.json" in bl.REQUIRED_RUNTIME_FILES
    assert bl.RUNTIME_FILE_SOURCES["src_py/camera_catalog.json"].is_file()
