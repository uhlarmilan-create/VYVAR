"""CAL-2026-10 (dev/results/specs/VYVAR_CALIBRATION_SPEC.md): library, masters, offset rule, plan.

Synthetic CMOS frames: pedestal P, dark current, vignetting v(x,y), Poisson-free noise.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

import calib_library as cl  # noqa: E402
import calib_masters as cm  # noqa: E402
import calib_plan as cp  # noqa: E402
from calibration import get_processed_master  # noqa: E402
from database import VyvarDatabase  # noqa: E402

NY, NX = 64, 96
P = 500.0  # pedestal / offset [ADU]
RATE = 0.5  # dark current [ADU/s]
T0 = datetime(2026, 3, 15, 22, 0, tzinfo=timezone.utc)


def _vign() -> np.ndarray:
    yy, xx = np.mgrid[0:NY, 0:NX]
    r2 = ((xx - NX / 2) / NX) ** 2 + ((yy - NY / 2) / NY) ** 2
    return (1.0 - 1.2 * r2).astype(np.float64)  # corner ~0.7


def _write(path: Path, data: np.ndarray, *, imagetyp: str, exp: float, when: datetime,
           filt: str | None = None, binning: int = 1, temp: float = -10.0, gain: int = 0,
           offset: int = 10, extra: dict | None = None) -> Path:
    h = fits.Header()
    h["IMAGETYP"] = imagetyp
    h["EXPTIME"] = exp
    h["DATE-OBS"] = when.strftime("%Y-%m-%dT%H:%M:%S")
    h["XBINNING"] = binning
    h["YBINNING"] = binning
    h["CCD-TEMP"] = temp
    h["SET-TEMP"] = temp
    h["GAIN"] = gain
    h["OFFSET"] = offset
    if filt:
        h["FILTER"] = filt
    for k, v in (extra or {}).items():
        h[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, data.astype(np.float32), header=h, overwrite=True)
    return path


@pytest.fixture()
def env(tmp_path: Path):
    db = VyvarDatabase(tmp_path / "t.sqlite3")
    cam = db.insert_equipment("C5A-150M", "cam", "CMOS", "x", 3.76)
    cam2 = db.insert_equipment("QHY294MM", "cam2", "CMOS", "x", 4.63)
    tel = db.insert_telescope("AZ800", "az", 800.0, 5480.0)
    cl.ensure_schema(db)
    return db, cam, cam2, tel, tmp_path / "lib", tmp_path / "src"


def _raw_darks(src: Path, exp: float, n: int, when: datetime, rng, *, imagetyp="Dark Frame") -> list[Path]:
    out = []
    for i in range(n):
        a = P + RATE * exp + rng.normal(0, 2.0, (NY, NX))
        if i == 0:
            a[5, 7] += 3000.0  # cosmic ray in one frame
        a[20, 30] += 50.0 * exp  # hot pixel in all
        out.append(_write(src / f"dark_{exp:g}_{i}.fits", a, imagetyp=imagetyp, exp=exp, when=when + timedelta(minutes=i)))
    return out


def _raw_flats(src: Path, n: int, when: datetime, rng, *, filt="V", exp=0.5) -> list[Path]:
    v = _vign()
    out = []
    for i in range(n):
        s = 20000.0 * (1.0 + 0.1 * i)  # twilight level changes frame to frame
        a = P + RATE * exp + s * v + rng.normal(0, 3.0, (NY, NX))
        out.append(_write(src / f"flat_{filt}_{i}.fits", a, imagetyp="Flat Field", exp=exp, when=when + timedelta(minutes=i), filt=filt))
    return out


def test_median_dark_rejects_cosmic_mean_keeps_trace(env):
    db, cam, _, tel, lib, src = env
    rng = np.random.default_rng(1)
    files = _raw_darks(src, 60.0, 9, T0, rng)
    med = cm.build_dark(files, method="median", library_root=lib, db=db, camera_id=cam)
    mean = cm.build_dark(files, method="mean", library_root=lib / "m", db=db, camera_id=cam)
    dm = fits.getdata(med.path)
    dn = fits.getdata(mean.path)
    assert abs(dm[5, 7] - (P + 60 * RATE)) < 10  # median: cosmic gone
    assert dn[5, 7] - (P + 60 * RATE) > 250  # mean: 3000/9 residual
    assert dm[20, 30] > P + 2000  # hot pixel kept (it is real dark signal)
    assert med.master.combine_method == "median" and med.master.origin == cl.ORIGIN_VYVAR
    assert fits.getheader(med.path)["VYSTKMOD"] == "MEDIAN"


def test_flat_offset_removed_by_constant_from_darks(env):
    db, cam, _, tel, lib, src = env
    rng = np.random.default_rng(2)
    _ = cm.build_dark(_raw_darks(src, 60.0, 5, T0, rng), library_root=lib, db=db, camera_id=cam)
    _ = cm.build_dark(_raw_darks(src, 120.0, 5, T0, rng), library_root=lib, db=db, camera_id=cam)
    est = cl.measure_offset_adu(db, camera_id=cam, gain=0, offset=10, binning=1)
    assert est.adu == pytest.approx(P, abs=0.5) and "intercept" in est.method
    r = cm.build_flat(_raw_flats(src, 7, T0, rng), library_root=lib, db=db, camera_id=cam, telescope_id=tel)
    f = fits.getdata(r.path)
    v = _vign()
    true = v / np.median(v)
    # the residual offset is only RATE*0.5 s = 0.25 ADU -> flat matches true vignetting to ~1e-4
    assert np.max(np.abs(f[2:-2, 2:-2] - true[2:-2, 2:-2])) < 2e-3
    assert fits.getheader(r.path)["VYOFFSUB"] == cl.OFFSET_CONSTANT
    assert r.master.offset_handling == cl.OFFSET_CONSTANT


def test_darkflat_preferred_and_long_dark_never_used_on_flat(env):
    db, cam, _, tel, lib, src = env
    rng = np.random.default_rng(3)
    lights_exp = [60.0]
    darks = _raw_darks(src, 60.0, 5, T0, rng) + _raw_darks(src, 0.5, 5, T0, rng)
    flats = _raw_flats(src, 5, T0, rng, exp=0.5)
    built, msgs = cp.build_masters_from_source(dark_files=darks, flat_files=flats, light_exposures=lights_exp,
                                               db=db, camera_id=cam, telescope_id=tel, library_root=lib)
    kinds = sorted(m.kind for m in cl.list_masters(db))
    assert kinds == ["dark", "darkflat", "flat"]
    flat = cl.list_masters(db, kind="flat")[0]
    assert flat.offset_handling == cl.OFFSET_DARKFLAT
    assert any("DARKFLAT" in m for m in msgs)


def test_validity_window(env):
    D = T0
    ok, _ = cl.validity_status(D, 60, [D - timedelta(hours=10), D + timedelta(days=59)])
    assert ok
    bad_before, why = cl.validity_status(D, 60, [D - timedelta(days=2)])
    assert not bad_before and "before" in why
    expired, why = cl.validity_status(D, 60, [D + timedelta(days=61)])
    assert not expired and "expired" in why
    assert not cl.validity_status(None, 60, [D])[0]


def test_plan_matches_camera_and_never_uses_expired(env):
    db, cam, cam2, tel, lib, src = env
    rng = np.random.default_rng(4)
    cm.build_dark(_raw_darks(src, 60.0, 5, T0, rng), library_root=lib, db=db, camera_id=cam)
    cm.build_flat(_raw_flats(src, 5, T0, rng), library_root=lib, db=db, camera_id=cam, telescope_id=tel)
    # a second camera's dark with the same keys must never be chosen for cam
    d2 = _raw_darks(src / "c2", 60.0, 5, T0 + timedelta(days=1), rng)
    cm.build_dark(d2, library_root=lib / "c2", db=db, camera_id=cam2)

    def _lights(when: datetime) -> dict:
        p = [_write(src / f"L_{when:%Y%m%d}_{i}.fits", np.full((NY, NX), 2000.0), imagetyp="Light Frame",
                    exp=60.0, when=when + timedelta(minutes=i), filt="V") for i in range(3)]
        return {"V|60|1": {"filter": "V", "exposure_s": 60.0, "binning": 1, "gain": 0, "temp": -10.0,
                           "light_paths": [str(x) for x in p]}}

    plan = cp.build_calibration_plan(_lights(T0 + timedelta(days=10)), db=db, camera_id=cam, telescope_id=tel,
                                     dark_validity_days=90, flat_validity_days=200, temp_tol=0.5)
    g = plan.groups["V|60|1"]
    assert g.status == cp.STATUS_OK and g.dark.camera_id == cam and g.flat.telescope_id == tel
    late = cp.build_calibration_plan(_lights(T0 + timedelta(days=120)), db=db, camera_id=cam, telescope_id=tel,
                                     dark_validity_days=90, flat_validity_days=200, temp_tol=0.5)
    g2 = late.groups["V|60|1"]
    assert g2.dark is None and g2.status == cp.STATUS_NO_DARK and "expired" in g2.dark_reason
    assert g2.flat is not None  # flat still within 200 d


def test_import_master_records_origin_and_links(env, tmp_path):
    db, cam, _, tel, lib, src = env
    ext = _write(tmp_path / "sips_flat.fits", _vign() * 30000 + P, imagetyp="Flat Field", exp=1.0, when=T0,
                 filt="V", extra={"SWCREATE": "SIPS 4.2"})
    r = cm.import_master(ext, kind="flat", library_root=lib, db=db, camera_id=cam, telescope_id=tel,
                         flat_offset_removed=False)
    assert r.master.origin == cl.ORIGIN_IMPORTED and r.master.origin_software == "SIPS 4.2"
    assert r.master.offset_handling == cl.OFFSET_UNKNOWN
    with pytest.raises(ValueError):
        cl.register_master(db, r.path, kind="flat", camera_id=cam, telescope_id=None)


def test_legacy_flat_offset_removed_at_use(env, tmp_path):
    """Old VYVAR flat = median of raw flats (offset inside). With darks in the library the
    plan measures P and get_processed_master removes it before normalisation."""
    db, cam, _, tel, lib, src = env
    rng = np.random.default_rng(5)
    cm.build_dark(_raw_darks(src, 60.0, 5, T0, rng), library_root=lib, db=db, camera_id=cam)
    cm.build_dark(_raw_darks(src, 120.0, 5, T0, rng), library_root=lib, db=db, camera_id=cam)
    v = _vign()
    legacy = _write(lib / "Flat_legacy.fits", P + 20000.0 * v, imagetyp="Flat Field", exp=0.5, when=T0, filt="V",
                    extra={"VYFLNRD": 1})
    cl.register_master(db, legacy, kind="flat", camera_id=cam, telescope_id=tel)
    lp = [_write(src / f"L{i}.fits", np.full((NY, NX), 2000.0), imagetyp="Light Frame", exp=60.0,
                 when=T0 + timedelta(days=1, minutes=i), filt="V") for i in range(2)]
    plan = cp.build_calibration_plan({"V|60|1": {"filter": "V", "exposure_s": 60.0, "binning": 1, "gain": 0,
                                                 "temp": -10.0, "light_paths": [str(x) for x in lp]}},
                                     db=db, camera_id=cam, telescope_id=tel, dark_validity_days=90,
                                     flat_validity_days=200, temp_tol=0.5,
                                     persist_offsets=True)  # the RUN plan (G01-22: a scan stores nothing)
    g = plan.groups["V|60|1"]
    assert g.flat_offset_handling == "removed-at-use" and g.flat_offset_adu == pytest.approx(P, abs=0.5)
    pm = get_processed_master(legacy, 1, kind="flat", master_binning=None, db=db)
    true = v / np.median(v)
    assert np.max(np.abs(pm.data - true)) < 1e-3


def _lights(lights_dir: Path, rng, *, n: int = 3, star_adu: float = 5000.0) -> np.ndarray:
    v = _vign()
    star = np.zeros((NY, NX))
    star[30:33, 40:43] = star_adu
    for i in range(n):
        a = P + RATE * 60 + (800.0 + star) * v + rng.normal(0, 1.0, (NY, NX))
        _write(lights_dir / f"L_{i}.fits", a, imagetyp="Light Frame", exp=60.0,
               when=T0 + timedelta(hours=1, minutes=i), filt="V", extra={"INSTRUME": "C5A-150M"})
    return star


def _calibrate_and_check(plan, lights_dir: Path, cam: int, tmp_path: Path) -> fits.Header:
    from pipeline_calibrate import calibrate_lights_to_calibrated
    from config import AppConfig

    cfg = AppConfig(project_root=tmp_path)
    cfg.database_path = str(tmp_path / "t.sqlite3")
    out_dir = tmp_path / "cal"
    calibrate_lights_to_calibrated(
        lights_root=lights_dir, calibrated_root=out_dir, master_dark_path=None,
        masterflat_by_filter=plan.masterflat_by_filter, pipeline_config=cfg, equipment_id=cam,
        masterflat_by_obs_key=plan.masterflat_by_obs_key, master_dark_by_obs_key=plan.dark_master_by_obs_key,
    )
    outs = sorted(out_dir.rglob("*.fits"))
    assert outs, "no calibrated output"
    c = fits.getdata(outs[0]).astype(np.float64)
    # flat-fielded sky is flat to < 0.5 % between corner and centre
    corner = np.median(c[2:12, 2:12])
    centre = np.median(c[NY // 2 - 5:NY // 2 + 5, 10:20])
    assert abs(corner / centre - 1.0) < 5e-3, (corner, centre)
    h = fits.getheader(outs[0])
    assert h["VY_DARK"] and h["VY_FLAT"]
    return h


def test_library_masters_calibrate_end_to_end_raw_frames_only_reported(env, tmp_path):
    """Masters built in the Calibration Library -> scan -> plan -> (L-D)/F.
    Raw calibration frames in the source folder are reported, never turned into masters."""
    from importer import smart_scan_source

    db, cam, _, tel, lib, src = env
    rng = np.random.default_rng(7)
    raw = tmp_path / "raw_cal"
    darks = _raw_darks(raw / "darks", 60.0, 7, T0 + timedelta(hours=8), rng)
    dflats = _raw_darks(raw / "darkflats", 0.5, 7, T0 + timedelta(hours=9), rng)
    flats = _raw_flats(raw / "flats", 7, T0 + timedelta(hours=9), rng, exp=0.5)
    built, _ = cp.build_masters_from_source(dark_files=darks + dflats, flat_files=flats, light_exposures=[60.0],
                                            db=db, camera_id=cam, telescope_id=tel, library_root=lib)
    assert len(built) == 3
    lights_dir = src / "lights"
    _lights(lights_dir, rng)
    _raw_darks(src / "darks", 60.0, 3, T0 + timedelta(hours=8), rng)  # raw frames next to the lights
    n_before = len(cl.list_masters(db))
    plan = smart_scan_source(
        source_root=src, calibration_library_root=lib, masterdark_validity_days=90,
        masterflat_validity_days=200, db=db, id_equipments=cam, id_telescope=tel,
        calibration_master_ccd_temp_tolerance_c=0.5,
    )
    cp_ = plan.calibration_plan
    g = cp_["groups"][0]
    assert g["status"] == "OK" and g["dark_from"] == "library" and g["flat_from"] == "library", cp_
    assert len(cl.list_masters(db)) == n_before  # scan wrote nothing
    assert any("raw dark" in w and "Calibration Library" in w for w in plan.warnings)
    _calibrate_and_check(plan, lights_dir, cam, tmp_path)


def _cmunipack_masters(folder: Path, rng, *, when: datetime) -> None:
    """Masters as C-Munipack writes them: ORIGIN names the program; the flat says
    'Dark frame subtracted.' (its offset is removed) and is normalised to ~mean level."""
    v = _vign()
    dark = P + RATE * 60 + rng.normal(0, 0.5, (NY, NX))
    _write(folder / "masterdark_60.fts", dark, imagetyp="DARK", exp=60.0, when=when,
           extra={"ORIGIN": "C-Munipack 2.1.34", "INSTRUME": "C5A-150M"})
    flat = 20000.0 * v
    h = fits.Header()
    path = _write(folder / "masterflat_V.fts", flat, imagetyp="FLAT", exp=3.0, when=when - timedelta(days=40),
                  filt="V", extra={"ORIGIN": "C-Munipack 2.1.34", "INSTRUME": "C5A-150M"})
    with fits.open(path, mode="update") as hd:
        hd[0].header.add_history("Dark frame subtracted.")
    del h


def test_masters_next_to_lights_recognised_preferred_and_imported(env, tmp_path):
    """C-Munipack masters lying with the LIGHTs: recognised from the header, preferred
    (masters_from=lights), scan writes nothing, RUN imports them into the library."""
    from importer import smart_scan_source

    db, cam, _, tel, lib, src = env
    rng = np.random.default_rng(11)
    _lights(src, rng)
    _cmunipack_masters(src, rng, when=T0 + timedelta(hours=7))
    kw = dict(source_root=src, calibration_library_root=lib, masterdark_validity_days=90,
              masterflat_validity_days=200, db=db, id_equipments=cam, id_telescope=tel,
              calibration_master_ccd_temp_tolerance_c=0.5)
    scan = smart_scan_source(**kw)
    g = scan.calibration_plan["groups"][0]
    assert g["status"] == "OK", scan.calibration_plan
    assert (g["dark"], g["dark_from"]) == ("masterdark_60.fts", "lights")
    assert (g["flat"], g["flat_from"]) == ("masterflat_V.fts", "lights")
    assert g["flat_offset_handling"] == cl.OFFSET_ALREADY
    assert g["dark_origin"] == "IMPORTED/C-Munipack 2.1.34"
    assert cl.list_masters(db) == []  # Scan Source is information only
    assert not any("are not used" in w for w in scan.warnings)

    run = smart_scan_source(**kw, import_source_masters=True)
    ms = cl.list_masters(db)
    assert sorted(m.kind for m in ms) == ["dark", "flat"]
    assert all(m.origin == cl.ORIGIN_IMPORTED and m.origin_software == "C-Munipack 2.1.34" for m in ms)
    flat = next(m for m in ms if m.kind == "flat")
    assert flat.telescope_id == tel and flat.offset_handling == cl.OFFSET_ALREADY
    assert all(Path(p).parent == lib for p in run.dark_master_by_obs_key.values())
    _calibrate_and_check(run, src, cam, tmp_path)
    # a second run re-uses the library copies (no duplicates)
    smart_scan_source(**kw, import_source_masters=True)
    assert len(cl.list_masters(db)) == 2


def test_masters_from_library_preferred_when_chosen(env, tmp_path):
    from importer import smart_scan_source

    db, cam, _, tel, lib, src = env
    rng = np.random.default_rng(13)
    _lights(src, rng)
    _cmunipack_masters(src, rng, when=T0 + timedelta(hours=7))
    cm.build_dark(_raw_darks(tmp_path / "rd", 60.0, 5, T0 + timedelta(hours=8), rng),
                  library_root=lib, db=db, camera_id=cam)
    kw = dict(source_root=src, calibration_library_root=lib, masterdark_validity_days=90,
              masterflat_validity_days=200, db=db, id_equipments=cam, id_telescope=tel,
              calibration_master_ccd_temp_tolerance_c=0.5)
    g = smart_scan_source(**kw, masters_from="library").calibration_plan["groups"][0]
    assert g["dark_from"] == "library"
    assert g["flat_from"] == "lights"  # no flat in the library -> fallback to the LIGHT folder
    g2 = smart_scan_source(**kw).calibration_plan["groups"][0]
    assert g2["dark_from"] == "lights"


def test_master_of_another_camera_next_to_lights_not_used(env, tmp_path):
    from importer import smart_scan_source

    db, cam, _, tel, lib, src = env
    rng = np.random.default_rng(17)
    _lights(src, rng)
    _write(src / "masterdark_60.fts", np.full((NY, NX), P + 30.0), imagetyp="DARK", exp=60.0,
           when=T0 + timedelta(hours=7), extra={"INSTRUME": "QHY294MM"})
    scan = smart_scan_source(source_root=src, calibration_library_root=lib, masterdark_validity_days=90,
                             masterflat_validity_days=200, db=db, id_equipments=cam, id_telescope=tel)
    assert scan.calibration_plan["groups"][0]["dark"] is None
    assert any("not the LIGHTs' camera" in w for w in scan.warnings)


def test_calibration_workers_give_identical_files(env, tmp_path):
    """SPEED-2026-10: one per-frame function for any number of workers; 1 and 2 workers write the
    same calibrated FITS (pixels and headers) and the same per-frame results."""
    from config import AppConfig
    from importer import smart_scan_source
    from pipeline_calibrate import calibrate_lights_to_calibrated

    db, cam, _, tel, lib, src = env
    rng = np.random.default_rng(7)
    raw = tmp_path / "raw_cal"
    darks = _raw_darks(raw / "darks", 60.0, 7, T0 + timedelta(hours=8), rng)
    dflats = _raw_darks(raw / "darkflats", 0.5, 7, T0 + timedelta(hours=9), rng)
    flats = _raw_flats(raw / "flats", 7, T0 + timedelta(hours=9), rng, exp=0.5)
    cp.build_masters_from_source(dark_files=darks + dflats, flat_files=flats, light_exposures=[60.0],
                                 db=db, camera_id=cam, telescope_id=tel, library_root=lib)
    lights_dir = src / "lights"
    _lights(lights_dir, rng, n=4)
    plan = smart_scan_source(
        source_root=src, calibration_library_root=lib, masterdark_validity_days=90,
        masterflat_validity_days=200, db=db, id_equipments=cam, id_telescope=tel,
        calibration_master_ccd_temp_tolerance_c=0.5,
    )
    cfg = AppConfig(project_root=tmp_path)
    cfg.database_path = str(tmp_path / "t.sqlite3")
    outs = {}
    for w in (1, 2):
        out_dir = tmp_path / f"cal_w{w}"
        st = calibrate_lights_to_calibrated(
            lights_root=lights_dir, calibrated_root=out_dir, master_dark_path=None,
            masterflat_by_filter=plan.masterflat_by_filter, pipeline_config=cfg, equipment_id=cam,
            masterflat_by_obs_key=plan.masterflat_by_obs_key, master_dark_by_obs_key=plan.dark_master_by_obs_key,
            max_workers=w,
        )
        assert st["calibrate_workers"] == w and st["processed"] == 4 and st["errors"] == 0
        files = sorted(out_dir.rglob("*.fits"))
        outs[w] = {f.name: (fits.getdata(f).tobytes(),
                            [(c.keyword, c.value) for c in fits.getheader(f).cards if c.keyword not in ("HISTORY", "DATE")])
                   for f in files}
        outs[w]["perf10"] = sorted((Path(k).name, v.get("fwhm_mean")) for k, v in st["perf10_qc_results"].items())
    assert outs[1] == outs[2]


def test_worker_infolog_lines_are_collected():
    from infolog import capture_events, log_event

    with capture_events() as ev:
        log_event("a")
        log_event("b")
    assert ev == ["a", "b"]
