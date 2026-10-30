# -*- coding: ascii -*-
"""Step 4 T01 (code review 2): calibration and import findings.

G01-15, G01-08/G13-18, G01-04/G01-19/G01-41, G01-39/G13-19, G01-09, G01-18, G01-36, G01-45, G01-46,
G01-10/G02-40/G06-30, G01-25, G01-32, cal_diag.json per lights tree.

Synthetic FITS / databases in tmp_path. Each test fails on the code before the change.
"""
from __future__ import annotations

import json
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import calib_library as cl
from database import VyvarDatabase

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _hdr(**kw) -> fits.Header:
    h = fits.Header()
    for k, v in kw.items():
        h[k.replace("_", "-") if k in ("CCD_TEMP", "DATE_OBS", "SET_TEMP") else k] = v
    return h


def _light_hdr(filt: str, *, when: datetime = NOW, binning: int = 1, **extra) -> fits.Header:
    kw = dict(IMAGETYP="LIGHT", EXPTIME=60.0, FILTER=filt, XBINNING=binning, YBINNING=binning, GAIN=0,
              OFFSET=10, CCD_TEMP=-10.0, DATE_OBS=when.strftime("%Y-%m-%dT%H:%M:%S"),
              CRVAL1=202.0, CRVAL2=47.0)
    kw.update(extra)
    return _hdr(**kw)


def _write(path: Path, data: np.ndarray, hdr: fits.Header) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.asarray(data, dtype=np.float32), hdr, overwrite=True)
    return path


class _Cfg:
    calibration_master_ccd_temp_tolerance_c = 0.5

    def __init__(self, data_root: Path, loc: int) -> None:
        self.observer_location_id = int(loc)
        self.data_root = data_root
        self.archive_root = data_root / "Archive"
        self.calibration_library_root = data_root / "CalibrationLibrary"
        self.archive_root.mkdir(parents=True, exist_ok=True)


class _Pipe:
    def __init__(self, db: VyvarDatabase, cfg: _Cfg) -> None:
        self.db = db
        self.config = cfg
        self.db._archive_root_override = Path(cfg.archive_root).expanduser().resolve()


def _rig(tmp_path: Path):
    data_root = tmp_path / "data"
    (data_root / "logs").mkdir(parents=True)
    db = VyvarDatabase(data_root / "vyvar.sqlite3")
    cam = db.insert_equipment(camera_name="QHY294MM", alias="c", sensor_type="CMOS", sensor_size="x", pixel_size=4.63)
    tel = db.insert_telescope("T", "t", 200.0, 1000.0)
    loc = db.insert_location(place_name="Obs", latitude=50.1, longitude=14.4, altitude=300.0)
    cl.ensure_schema(db)
    return db, cam, tel, loc, data_root


def _register_dark(db, cam, lib: Path, *, binning=1, shape=(32, 48), level=1000.0, gain=0) -> Path:
    p = _write(lib / f"dark_b{binning}_g{gain}.fits", np.full(shape, level),
               _hdr(IMAGETYP="Dark Frame", EXPTIME=60.0, XBINNING=binning, YBINNING=binning, GAIN=gain, OFFSET=10,
                    CCD_TEMP=-10.0, DATE_OBS=NOW.strftime("%Y-%m-%dT%H:%M:%S")))
    cl.register_master(db, p, kind=cl.KIND_DARK, camera_id=cam, telescope_id=None)
    return p


def _register_flat(db, cam, tel, lib: Path, filt: str, data: np.ndarray, *, binning=1, gain=0, **extra) -> Path:
    kw = dict(IMAGETYP="Flat Field", EXPTIME=1.0, FILTER=filt, XBINNING=binning, YBINNING=binning, GAIN=gain,
              OFFSET=10, CCD_TEMP=-10.0, DATE_OBS=NOW.strftime("%Y-%m-%dT%H:%M:%S"))
    kw.update(extra)
    p = _write(lib / f"flat_{filt}_g{gain}.fits", data, _hdr(**kw))
    cl.register_master(db, p, kind=cl.KIND_FLAT, camera_id=cam, telescope_id=tel)
    return p


def _calibrate_draft(db, data_root: Path, archive: Path, plan, draft_id, *, qc: bool = False) -> dict:
    import pipeline
    from config import AppConfig

    cfg = AppConfig()
    cfg.database_path = data_root / "vyvar.sqlite3"
    cfg.archive_root = data_root / "Archive"
    cfg.qc_after_calibrate_enabled = bool(qc)
    self_ = types.SimpleNamespace(config=cfg, db=db)
    return pipeline.AstroPipeline.quick_calibrate_last_import(
        self_, archive_path=archive, master_dark_path=None, masterflat_by_filter={},
        draft_id=draft_id,
        masterflat_by_obs_key=dict(plan.masterflat_by_obs_key),
        master_dark_by_obs_key=dict(plan.dark_master_by_obs_key),
        flat_substitute_by_obs_key=dict(plan.flat_substitute_by_obs_key),
    )


def _scan_import(db, cam, tel, loc, data_root: Path, src: Path):
    from importer import smart_import_session, smart_scan_source

    lib = data_root / "CalibrationLibrary"
    plan = smart_scan_source(source_root=src.parent, calibration_library_root=lib, masterdark_validity_days=90,
                             masterflat_validity_days=200, db=db, id_equipments=cam, id_telescope=tel)
    res = smart_import_session(plan=plan, pipeline=_Pipe(db, _Cfg(data_root, loc)), id_equipment=cam,
                               id_telescope=tel, cfg=_Cfg(data_root, loc))
    return plan, res


# ---------------------------------------------------------------------------------------------
# G01-15: one run DB handle independent of the QC switch
# ---------------------------------------------------------------------------------------------
def test_g0115_qc_off_still_resolves_flat_offset_and_records_calib_flags(tmp_path: Path) -> None:
    db, cam, tel, loc, data_root = _rig(tmp_path)
    lib = data_root / "CalibrationLibrary"
    _register_dark(db, cam, lib)
    rng = np.random.default_rng(5)
    # flat without VYOFFSUB: its offset comes from the library row (needs the run database)
    _register_flat(db, cam, tel, lib, "V", (20000.0 * (1 + rng.normal(0, 0.004, (32, 48)))).astype(np.float32))
    src = tmp_path / "night" / "Lights"
    for i in range(2):
        _write(src / f"V_{i}.fits", np.full((32, 48), 3000.0), _light_hdr("V"))
    plan, res = _scan_import(db, cam, tel, loc, data_root, src)
    arch = Path(res.archive_path)
    _calibrate_draft(db, data_root, arch, plan, int(res.draft_id), qc=False)
    cal = sorted((arch / "calibrated" / "lights").rglob("*.fits"))
    assert [fits.getheader(p)["VY_CFLAG"] for p in cal] == ["DF", "DF"]
    files = json.loads((arch / "draft_manifest.json").read_text())["files"]
    lights = [f for f in files if str(f.get("file_path", f.get("FILE_PATH", ""))).endswith(".fits")]
    assert lights and all(str(f.get("calib_flags")) == "DF" for f in lights)


def test_g0115_ram_qc_uses_the_run_database_for_flat_offsets(tmp_path: Path) -> None:
    from pipeline_calibrate import run_draft_ram_calibration_qc_to_obs_files

    db, cam, tel, loc, data_root = _rig(tmp_path)
    lib = data_root / "CalibrationLibrary"
    _register_dark(db, cam, lib)
    rng = np.random.default_rng(6)
    _register_flat(db, cam, tel, lib, "V", (20000.0 * (1 + rng.normal(0, 0.004, (32, 48)))).astype(np.float32))
    src = tmp_path / "night" / "Lights"
    for i in range(2):
        _write(src / f"V_{i}.fits", np.full((32, 48), 3000.0), _light_hdr("V"))
    plan, res = _scan_import(db, cam, tel, loc, data_root, src)
    out = run_draft_ram_calibration_qc_to_obs_files(
        db=db, draft_id=int(res.draft_id), archive_path=res.archive_path, master_dark_path=None,
        masterflat_by_filter={}, masterflat_by_obs_key=dict(plan.masterflat_by_obs_key),
        master_dark_by_obs_key=dict(plan.dark_master_by_obs_key), equipment_id=cam, pedestal_temp_tol_c=0.5)
    assert out["errors"] == []


# ---------------------------------------------------------------------------------------------
# G01-08 / G13-18: unknown CCD temperature is NULL / None, never 0 C
# ---------------------------------------------------------------------------------------------
def _light_without_temp(path: Path) -> Path:
    h = _light_hdr("V")
    del h["CCD-TEMP"]
    return _write(path, np.full((32, 48), 3000.0), h)


def test_g0108_light_without_temperature_is_unknown_with_db_and_no_zero_c_dark(tmp_path: Path) -> None:
    from fits_meta import extract_fits_metadata
    from importer import _calibration_light_temp_c, smart_scan_source

    db, cam, tel, loc, data_root = _rig(tmp_path)
    lib = data_root / "CalibrationLibrary"
    p = _write(lib / "dark_0C.fits", np.full((32, 48), 1000.0),
               _hdr(IMAGETYP="Dark Frame", EXPTIME=60.0, XBINNING=1, YBINNING=1, GAIN=0, OFFSET=10, CCD_TEMP=0.0,
                    DATE_OBS=NOW.strftime("%Y-%m-%dT%H:%M:%S")))
    cl.register_master(db, p, kind=cl.KIND_DARK, camera_id=cam, telescope_id=None)
    src = tmp_path / "night" / "Lights"
    lt = _light_without_temp(src / "V_0.fits")
    # cache row written, then read back from the cache
    assert extract_fits_metadata(lt, db=db)["temp"] is None
    assert extract_fits_metadata(lt, db=db)["temp"] is None
    row = db.conn.execute("SELECT CCD_TEMP FROM FITS_HEADER_CACHE").fetchone()
    assert row["CCD_TEMP"] is None
    assert _calibration_light_temp_c(lt, db=db) is None
    plan = smart_scan_source(source_root=src.parent, calibration_library_root=lib, masterdark_validity_days=90,
                             masterflat_validity_days=200, db=db, id_equipments=cam, id_telescope=tel)
    g = plan.calibration_plan["groups"][0]
    assert g["temp_c"] is None
    assert g["dark"] is None and "no sensor temperature" in str(g["dark_reason"])


def test_g1318_scanning_id_unknown_temperature_differs_from_zero_c() -> None:
    from draft_provenance import derive_scanning_id

    base = {"exposure": 60.0, "filter": "V", "binning": 1, "gain": 0}
    assert derive_scanning_id(dict(base, temp=None)) != derive_scanning_id(dict(base, temp=0.0))


def _real_db_copy(tmp_path: Path) -> Path:
    import shutil

    for cand in (Path("/root/p0env_pristine/vyvar.sqlite3"), Path("/root/p0data/vyvar.sqlite3")):
        if cand.is_file():
            dst = tmp_path / "copy.sqlite3"
            shutil.copy2(cand, dst)
            return dst
    pytest.skip("no real VYVAR database copy available")


def test_g0108_cache_migration_on_a_real_database_copy(tmp_path: Path) -> None:
    import sqlite3

    dbp = _real_db_copy(tmp_path)
    c = sqlite3.connect(dbp)
    n_lib = c.execute("SELECT COUNT(*) FROM CALIBRATION_LIBRARY").fetchone()[0]
    n_eq = c.execute("SELECT COUNT(*) FROM EQUIPMENTS").fetchone()[0]
    c.close()
    db = VyvarDatabase(dbp)
    info = {r["name"]: r for r in db.conn.execute("PRAGMA table_info('FITS_HEADER_CACHE')")}
    assert int(info["CCD_TEMP"]["notnull"]) == 0
    assert db.conn.execute("SELECT COUNT(*) FROM CALIBRATION_LIBRARY").fetchone()[0] == n_lib
    assert db.conn.execute("SELECT COUNT(*) FROM EQUIPMENTS").fetchone()[0] == n_eq
    db.conn.close()
    # idempotent: a second open keeps the new table (and its rows)
    db2 = VyvarDatabase(dbp)
    lt = _light_without_temp(tmp_path / "L.fits")
    from fits_meta import extract_fits_metadata

    assert extract_fits_metadata(lt, db=db2)["temp"] is None
    db2.conn.close()
    db3 = VyvarDatabase(dbp)
    assert db3.conn.execute("SELECT COUNT(*) FROM FITS_HEADER_CACHE").fetchone()[0] == 1


# ---------------------------------------------------------------------------------------------
# G01-10 / G02-40 / G06-30: one exact gain key; library migration; one gain authority
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("card, want", [(1.5, 1.5), ("1.5", 1.5), (120, 120.0), (12.48, 12.48), (0.33, 0.33)])
def test_g0240_gain_card_is_an_exact_float(tmp_path: Path, card, want) -> None:
    from fits_meta import extract_fits_metadata

    lt = _write(tmp_path / "L.fits", np.zeros((8, 8)), _light_hdr("V", GAIN=card))
    assert cl.header_gain(fits.getheader(lt)) == want
    assert extract_fits_metadata(lt)["gain"] == want


def test_g0110_fractional_gain_light_matches_its_masters(tmp_path: Path) -> None:
    from importer import smart_scan_source

    db, cam, tel, loc, data_root = _rig(tmp_path)
    lib = data_root / "CalibrationLibrary"
    _register_dark(db, cam, lib, gain=1.6)
    _register_flat(db, cam, tel, lib, "V", np.full((32, 48), 20000.0), gain=1.6, VYOFFSUB="constant")
    _register_dark(db, cam, lib, gain=2)  # the rounded value must not be taken
    src = tmp_path / "night" / "Lights"
    _write(src / "V_0.fits", np.full((32, 48), 3000.0), _light_hdr("V", GAIN=1.6))
    plan = smart_scan_source(source_root=src.parent, calibration_library_root=lib, masterdark_validity_days=90,
                             masterflat_validity_days=200, db=db, id_equipments=cam, id_telescope=tel)
    g = plan.calibration_plan["groups"][0]
    assert g["gain"] == 1.6
    assert g["dark"] == "dark_b1_g1.6.fits" and g["flat"] == "flat_V_g1.6.fits"


def test_g0110_library_gain_migration_reads_the_exact_card(tmp_path: Path) -> None:
    db, cam, tel, loc, data_root = _rig(tmp_path)
    lib = data_root / "CalibrationLibrary"
    p = _register_dark(db, cam, lib, gain=12.48)
    # a row as the old code stored it: rounded gain, not migrated
    db.conn.execute("UPDATE CALIBRATION_LIBRARY SET GAIN = 12, GAIN_EXACT = NULL")
    gone = lib / "gone.fits"
    db.conn.execute(
        "INSERT INTO CALIBRATION_LIBRARY (KIND, FILE_PATH, XBINNING, EXPTIME, CCD_TEMP, GAIN, REGISTERED_AT, "
        "ID_EQUIPMENTS) VALUES ('dark', ?, 1, 60, -10, 3, 'x', ?)", (str(gone), cam))
    db.conn.commit()
    cl.ensure_schema(db)
    rows = {r["FILE_PATH"]: dict(r) for r in db.conn.execute("SELECT FILE_PATH, GAIN, GAIN_EXACT FROM CALIBRATION_LIBRARY")}
    assert rows[str(p.resolve())]["GAIN"] == 12.48 and rows[str(p.resolve())]["GAIN_EXACT"] == 1
    assert rows[str(gone)]["GAIN"] == 3 and rows[str(gone)]["GAIN_EXACT"] == 0
    m = cl.find_dark(db, camera_id=cam, gain=12.48, offset=10, exposure=60.0, binning=1, temp=-10.0, temp_tol=0.5,
                     light_dates=[NOW], validity_days=90)
    assert m.master is not None and Path(m.master.path) == p.resolve()


def test_g0110_library_gain_migration_on_a_real_database_copy(tmp_path: Path) -> None:
    dbp = _real_db_copy(tmp_path)
    db = VyvarDatabase(dbp)
    cl.ensure_schema(db)
    rows = [dict(r) for r in db.conn.execute("SELECT FILE_PATH, GAIN, GAIN_EXACT FROM CALIBRATION_LIBRARY")]
    assert rows and all(r["GAIN_EXACT"] in (0, 1) for r in rows)
    for r in rows:
        if r["GAIN_EXACT"] == 1:
            assert r["GAIN"] == cl.header_gain(fits.getheader(r["FILE_PATH"]))
        if "cam4" in str(r["FILE_PATH"]) and r["GAIN_EXACT"] == 1:
            assert r["GAIN"] == 12.48  # C5A-150M e-/ADU card (was stored as 12)
    n = len(rows)
    cl.ensure_schema(db)  # idempotent
    assert db.conn.execute("SELECT COUNT(*) FROM CALIBRATION_LIBRARY").fetchone()[0] == n


def test_g0630_gain_provenance_is_the_resolution_used() -> None:
    from rig_photometry import photometric_gain_read_noise

    h = _hdr(RDNOISE=3.5, XBINNING=1)
    h["EGAIN"] = (1.25, "e-/ADU")
    pg = photometric_gain_read_noise(h, None, None)
    assert pg.gain_resolved is not None and pg.read_noise_resolved is not None
    assert float(pg.gain_resolved.value) == pg.gain_native
    assert float(pg.read_noise_resolved.value) == pg.read_noise


# ---------------------------------------------------------------------------------------------
# G01-04 / G01-19 / G01-41: OSC channel header - VYVAR stamps are the authority
# ---------------------------------------------------------------------------------------------
def _osc_src_header(**extra) -> fits.Header:
    h = _light_hdr("NoFilter", GAIN=120, XPIXSZ=4.63, YPIXSZ=4.63, **extra)
    return h


def test_g0104_channel_gain_stamp_is_e_per_adu_for_a_setting_profile() -> None:
    from camera_profile import CameraProfile, CurvePoint
    from osc_extract import build_channel_header
    from param_resolver import resolve_gain

    prof = CameraProfile(equipment_id=7, header_gain="setting",
                         points=(CurvePoint(0.0, 1, 4.0, 7.0, "catalog"), CurvePoint(120.0, 1, 1.0, 2.0, "catalog")))
    ch = build_channel_header(_osc_src_header(), channel="G", bayermask="RGGB", osc_bin=1, gain_eff=25.6,
                              rn_eff=3.0, src_name="L.fits")
    res = resolve_gain(ch, profile=prof)
    assert res.ok and res.source == "header" and float(res.value) == 25.6
    # a plain mono frame of the same camera: GAIN is still the driver setting
    res_m = resolve_gain(_osc_src_header(), profile=prof)
    assert res_m.source == "catalog" and float(res_m.value) == 1.0


def test_g0141_channel_header_effective_pixel_scaled_once() -> None:
    from fits_meta import fits_metadata_from_primary_header
    from osc_extract import build_channel_header

    for osc_bin in (1, 2):
        ch = build_channel_header(_osc_src_header(), channel="R", bayermask="RGGB", osc_bin=osc_bin, gain_eff=1.0,
                                  rn_eff=3.0, src_name="L.fits")
        # XPIXSZ is the binned pitch (HDR-ONE optics convention, merged with s4_hdr); the property is the
        # effective pixel, scaled once
        assert int(ch["XBINNING"]) == 2 * osc_bin
        meta = fits_metadata_from_primary_header(ch)
        assert meta["effective_pixel_um_plate_scale"] == pytest.approx(4.63 * 2 * osc_bin)


def test_g0119_planes_follow_the_equipment_mask_not_bayerpat(tmp_path: Path) -> None:
    from osc_extract import extract_one_light_to_channels

    mosaic = np.zeros((8, 8), dtype=np.float32)
    mosaic[0::2, 0::2] = 100.0  # R under RGGB
    mosaic[1::2, 1::2] = 7.0  # B under RGGB
    mosaic[0::2, 1::2] = 50.0
    mosaic[1::2, 0::2] = 50.0
    src = _write(tmp_path / "L.fits", mosaic, _osc_src_header(BAYERPAT="BGGR"))
    from osc_extract import OSC_CHANNELS

    out = {c: tmp_path / c for c in OSC_CHANNELS}
    w = extract_one_light_to_channels(src, out_dirs=out, bayermask="RGGB", osc_bin=1, gain_e_per_adu=1.0,
                                      read_noise_e=3.0)
    assert float(np.nanmedian(fits.getdata(w["R"]))) == 100.0
    assert float(np.nanmedian(fits.getdata(w["B"]))) == 7.0
    assert fits.getheader(w["R"])["VYBAYMSK"] == "RGGB"


# ---------------------------------------------------------------------------------------------
# G01-39 / G13-19: "rows exist but none accepted" is not "no rows"
# ---------------------------------------------------------------------------------------------
def test_g0139_all_rows_rejected_selects_no_frame(tmp_path: Path, monkeypatch) -> None:
    from calibration import filter_light_paths_for_calibration_db

    dbp = tmp_path / "v.sqlite3"
    VyvarDatabase(dbp).conn.close()
    monkeypatch.setattr(VyvarDatabase, "count_obs_files_for_draft", lambda self, d: 2)
    monkeypatch.setattr(VyvarDatabase, "fetch_light_file_paths_not_rejected_for_draft", lambda self, d: set())
    paths = [tmp_path / "L_1.fits", tmp_path / "L_2.fits"]
    assert filter_light_paths_for_calibration_db(paths, database_path=dbp, draft_id=1) == []
    # no rows at all: unchanged (backward compatible)
    monkeypatch.setattr(VyvarDatabase, "count_obs_files_for_draft", lambda self, d: 0)
    assert filter_light_paths_for_calibration_db(paths, database_path=dbp, draft_id=1) == paths


def test_g1319_observation_with_all_lights_rejected_is_empty(tmp_path: Path, monkeypatch) -> None:
    import draft_provenance

    db = VyvarDatabase(tmp_path / "v.sqlite3")
    db._archive_root_override = tmp_path / "Archive"
    rej = [{"IMAGETYP": "light", "IS_REJECTED": 1, "FILE_PATH": str(tmp_path / "a" / "L_1.fits")}]
    monkeypatch.setattr(draft_provenance, "collect_manifest_obs_file_rows", lambda root, observation_id: rej)
    monkeypatch.setattr(draft_provenance, "iter_draft_archive_dirs", lambda root: [(9, tmp_path / "other")])
    monkeypatch.setattr(VyvarDatabase, "fetch_draft_light_rows_for_quality",
                        lambda self, d: [{"IS_REJECTED": 0, "FILE_PATH": str(tmp_path / "other" / "X.fits")}])
    assert db.fetch_light_file_paths_not_rejected_for_observation("OBS1") == set()


# ---------------------------------------------------------------------------------------------
# G01-09: offset / readout mode reach the mixed-camera-settings check
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("extra", [{"OFFSET": 30}, {"READOUTM": "High Gain Mode"}])
def test_g0109_mixed_offset_or_readout_mode_is_reported(tmp_path: Path, extra) -> None:
    from importer import smart_scan_source

    db, cam, tel, loc, data_root = _rig(tmp_path)
    lib = data_root / "CalibrationLibrary"
    lib.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "night" / "Lights"
    _write(src / "V_0.fits", np.full((16, 16), 3000.0), _light_hdr("V", READOUTM="Photographic"))
    _write(src / "V_1.fits", np.full((16, 16), 3000.0), _light_hdr("V", **{"READOUTM": "Photographic", **extra}))
    for _ in range(2):  # second scan reads the header cache
        plan = smart_scan_source(source_root=src.parent, calibration_library_root=lib, masterdark_validity_days=90,
                                 masterflat_validity_days=200, db=db, id_equipments=cam, id_telescope=tel)
        assert any("different camera settings" in w for w in plan.warnings), plan.warnings


# ---------------------------------------------------------------------------------------------
# G01-18 / G01-32 / G01-36: import robustness
# ---------------------------------------------------------------------------------------------
def _manifest_lights(arch: Path) -> list[dict]:
    files = json.loads((arch / "draft_manifest.json").read_text())["files"]
    return [f for f in files if str(f.get("imagetyp", f.get("IMAGETYP", ""))).lower() == "light"]


def test_g0118_same_file_name_in_two_subfolders_is_not_overwritten(tmp_path: Path) -> None:
    db, cam, tel, loc, data_root = _rig(tmp_path)
    (data_root / "CalibrationLibrary").mkdir(parents=True, exist_ok=True)
    src = tmp_path / "night" / "Lights"
    _write(src / "a" / "Light_001.fits", np.full((16, 16), 100.0), _light_hdr("V"))
    _write(src / "b" / "Light_001.fits", np.full((16, 16), 200.0), _light_hdr("V"))
    plan, res = _scan_import(db, cam, tel, loc, data_root, src)
    got = sorted(Path(res.archive_path).rglob("*.fits"))
    assert len(got) == 2
    assert sorted(float(fits.getdata(p)[0, 0]) for p in got) == [100.0, 200.0]
    assert len({str(f.get("file_path", f.get("FILE_PATH"))) for f in _manifest_lights(Path(res.archive_path))}) == 2
    assert any("same name" in w for w in res.warnings)


def test_g0132_manifest_group_key_uses_the_raw_filter(tmp_path: Path) -> None:
    from fits_meta import fits_metadata_from_primary_header, observation_group_key_from_metadata

    db, cam, tel, loc, data_root = _rig(tmp_path)
    (data_root / "CalibrationLibrary").mkdir(parents=True, exist_ok=True)
    src = tmp_path / "night" / "Lights"
    lt = _write(src / "L_1.fits", np.full((16, 16), 100.0), _light_hdr("Sloan r'"))
    plan, res = _scan_import(db, cam, tel, loc, data_root, src)
    want = observation_group_key_from_metadata(fits_metadata_from_primary_header(fits.getheader(lt)))
    assert want == "Sloan r'|60|1"
    assert list(plan.observation_groups) == [want]
    rows = _manifest_lights(Path(res.archive_path))
    assert [r.get("group_key") for r in rows] == [want]
    # folder name unchanged (sanitised filter)
    assert [p.parent.name for p in Path(res.archive_path).rglob("*.fits")] == ["Sloanr__60_1"]


def test_g0136_no_silent_cap_on_the_lights_scanned(tmp_path: Path) -> None:
    from importer import smart_scan_source

    db, cam, tel, loc, data_root = _rig(tmp_path)
    lib = data_root / "CalibrationLibrary"
    lib.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "night" / "Lights"
    hv = _light_hdr("V")
    for i in range(500):
        _write(src / f"A_{i:04d}.fits", np.zeros((4, 4)), hv)
    _write(src / "Z_9999.fits", np.zeros((4, 4)), _light_hdr("B"))
    plan = smart_scan_source(source_root=src.parent, calibration_library_root=lib, masterdark_validity_days=90,
                             masterflat_validity_days=200, db=db, id_equipments=cam, id_telescope=tel)
    assert "B" in plan.detected_filters
    assert sorted(plan.observation_groups) == ["B|60|1", "V|60|1"]


# ---------------------------------------------------------------------------------------------
# G01-45: preprocess disk fallback matches exact file stems
# ---------------------------------------------------------------------------------------------
def test_g0145_disk_fallback_exact_stem_match(tmp_path: Path) -> None:
    from pipeline_preprocess import calibrated_paths_for_draft_apply_filters

    ap = tmp_path / "draft"
    src_root = ap / "calibrated" / "lights"
    for n in ("Light_1", "Light_10", "Light_11"):
        _write(src_root / "V_60_1" / f"{n}.fits", np.zeros((4, 4)), _light_hdr("V"))
    rows = [{"FILE_PATH": str(tmp_path / "elsewhere" / "Light_1.fits"), "IS_REJECTED": 0},
            {"FILE_PATH": str(tmp_path / "elsewhere" / "Light_10.fits"), "IS_REJECTED": 1}]
    db = types.SimpleNamespace(fetch_draft_light_rows_for_quality=lambda d: rows)
    got = calibrated_paths_for_draft_apply_filters(ap, db, 1, source_dir=src_root)
    assert [p.name for p in got] == ["Light_1.fits"]


# ---------------------------------------------------------------------------------------------
# G01-25: binning from the plan / header only, a shape mismatch stops
# ---------------------------------------------------------------------------------------------
def test_g0125_half_chip_bin1_roi_is_not_treated_as_bin2(tmp_path: Path) -> None:
    from calibration import MasterResamplingError, get_processed_master

    dk = _write(tmp_path / "dark.fits", np.full((32, 48), 100.0),
                _hdr(IMAGETYP="Dark Frame", EXPTIME=60.0, XBINNING=1, YBINNING=1, CCD_TEMP=-10.0))
    with pytest.raises(MasterResamplingError, match="16x24"):
        get_processed_master(dk, 1, kind="dark", light_shape=(16, 24), dark_resample_mode="sum")
    # the real bin-2 light (header binning 2) is still rebinned
    res = get_processed_master(dk, 2, kind="dark", light_shape=(16, 24), dark_resample_mode="sum")
    assert res.data.shape == (16, 24) and float(res.data[0, 0]) == 400.0


# ---------------------------------------------------------------------------------------------
# G01-46 (Milan 2026-10-30, option a): pedestal = OLS intercept, sigma_P from the fit; pedestal darks
# of the same sensor setup only
# ---------------------------------------------------------------------------------------------
def _dark(path: Path, level: float, exptime: float, *, instr="CAM-A", gain=1.6, temp=-10.0, binning=1,
          noise=0.0, seed=0) -> Path:
    rng = np.random.default_rng(seed)
    data = np.full((40, 50), level, dtype=np.float64) + (rng.normal(0, noise, (40, 50)) if noise else 0.0)
    h = fits.Header()
    h["IMAGETYP"] = "Dark Frame"
    h["INSTRUME"] = instr
    h["GAIN"] = gain
    h["EXPTIME"] = exptime
    h["CCD-TEMP"] = temp
    h["XBINNING"] = binning
    h["YBINNING"] = binning
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data.astype(np.float32), header=h).writeto(path, overwrite=True)
    return path


def test_g0146_three_exposures_sigma_p_is_the_ols_intercept_error(tmp_path):
    from cal_diag import _pedestal_intercept_fit

    # exact line + scatter: sigma_P == the textbook standard error of the intercept
    t = np.array([30.0, 60.0, 120.0, 300.0])
    m = 100.0 + 0.02 * t + np.array([0.3, -0.5, 0.4, -0.1])
    p, k, sp, note = _pedestal_intercept_fit([(ti, mi, 0.0) for ti, mi in zip(t, m)])
    k_ref, p_ref = np.polyfit(t, m, 1)
    resid = m - (p_ref + k_ref * t)
    s2 = np.sum(resid**2) / (len(t) - 2)
    se = np.sqrt(s2 * (1 / len(t) + t.mean() ** 2 / np.sum((t - t.mean()) ** 2)))
    assert p == pytest.approx(p_ref) and k == pytest.approx(k_ref) and sp == pytest.approx(se) and note == ""


def test_g0146_two_exposures_sigma_p_is_statistical_not_zero(tmp_path):
    from cal_diag import derive_pedestal_from_masters

    a = _dark(tmp_path / "d60.fits", 24.5, 60.0, noise=1.7, seed=1)
    b = _dark(tmp_path / "d120.fits", 24.5, 120.0, noise=1.7, seed=2)
    ped = derive_pedestal_from_masters([a, b])
    assert ped.method == "INTERCEPT" and ped.n_exptimes == 2
    assert ped.p_adu == pytest.approx(24.5, abs=0.5)
    # propagated SE of two medians of 2000 px with sigma 1.7: ~1.25*1.7/sqrt(2000) * |w|
    assert 0.0 < ped.sigma_p < 0.5


def test_g0146_pedestal_darks_match_camera_gain_binning_and_temperature(tmp_path):
    from cal_diag import discover_pedestal_dark_paths

    lib = tmp_path / "lib"
    use = _dark(lib / "imported_Dark_120s.fits", 512.0, 120.0, instr="C5A", gain=12.48, temp=-14.96, binning=4)
    ok = _dark(lib / "imported_Dark_100s.fits", 512.0, 100.0, instr="C5A", gain=12.48, temp=-14.95, binning=4)
    _dark(lib / "Dark_60s.fits", 24.5, 60.0, instr="QHY294", gain=0.0, temp=-10.0)  # other camera
    _dark(lib / "imported_Dark_100s_warm.fits", 512.0, 100.0, instr="C5A", gain=12.48, temp=-12.0, binning=4)
    _dark(lib / "imported_Dark_100s_b1.fits", 512.0, 100.0, instr="C5A", gain=12.48, temp=-14.95, binning=1)
    got = discover_pedestal_dark_paths(use, temp_tol_c=0.5)
    assert got == [use.resolve(), ok.resolve()]
