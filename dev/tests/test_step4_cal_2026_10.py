# -*- coding: ascii -*-
"""Step 4 calibration decisions (Milan 2026-10-29): A1 (G01-01), A2 (G01-29), A3 (G01-24), G01-02, G01-03.

Synthetic FITS in tmp_path. Each test fails on the code before the change.
"""
from __future__ import annotations

import json
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import calib_library as cl
import calib_plan as cp
import pipeline_calibrate as pc
from calibration import (
    MasterResamplingError,
    ProcessedMasterResult,
    get_processed_master,
    normalize_flat_master,
    resample_master_to_light_binning,
)
from database import VyvarDatabase

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _hdr(**kw) -> fits.Header:
    h = fits.Header()
    for k, v in kw.items():
        h[k.replace("_", "-") if k in ("CCD_TEMP", "DATE_OBS", "SET_TEMP") else k] = v
    return h


def _light_hdr(filt: str, *, when: datetime = NOW, binning: int = 1) -> fits.Header:
    return _hdr(IMAGETYP="LIGHT", EXPTIME=60.0, FILTER=filt, XBINNING=binning, YBINNING=binning, GAIN=0,
                OFFSET=10, CCD_TEMP=-10.0, DATE_OBS=when.strftime("%Y-%m-%dT%H:%M:%S"),
                CRVAL1=202.0, CRVAL2=47.0)


def _write(path: Path, data: np.ndarray, hdr: fits.Header) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.asarray(data, dtype=np.float32), hdr, overwrite=True)
    return path


def _noisy_flat(shape, rng, *, level=1.0, vignetting=0.0, noise=0.004) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = ((xx - nx / 2) / (nx / 2)) ** 2 + ((yy - ny / 2) / (ny / 2)) ** 2
    v = 1.0 - vignetting * r2 / 2.0
    return (level * v * (1.0 + rng.normal(0.0, noise, shape))).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# G01-01 (decision A1): one lights tree, every light calibrated, no-flat group D-only + flagged
# ---------------------------------------------------------------------------------------------
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


def _register_dark(db, cam, lib: Path, *, binning=1, shape=(32, 48), when=NOW, level=1000.0) -> Path:
    p = _write(lib / f"dark_b{binning}_{when:%Y%m%d}.fits", np.full(shape, level),
               _hdr(IMAGETYP="Dark Frame", EXPTIME=60.0, XBINNING=binning, YBINNING=binning, GAIN=0, OFFSET=10,
                    CCD_TEMP=-10.0, DATE_OBS=when.strftime("%Y-%m-%dT%H:%M:%S")))
    cl.register_master(db, p, kind=cl.KIND_DARK, camera_id=cam, telescope_id=None)
    return p


def _register_flat(db, cam, tel, lib: Path, filt: str, data: np.ndarray, *, when=NOW, binning=1) -> Path:
    p = _write(lib / f"flat_{filt}_{when:%Y%m%d}.fits", data,
               _hdr(IMAGETYP="Flat Field", EXPTIME=1.0, FILTER=filt, XBINNING=binning, YBINNING=binning, GAIN=0,
                    OFFSET=10, CCD_TEMP=-10.0, VYOFFSUB="constant",
                    DATE_OBS=when.strftime("%Y-%m-%dT%H:%M:%S")))
    cl.register_master(db, p, kind=cl.KIND_FLAT, camera_id=cam, telescope_id=tel)
    return p


def _calibrate_draft(db, data_root: Path, archive: Path, plan, draft_id) -> dict:
    import pipeline
    from config import AppConfig

    cfg = AppConfig()
    cfg.database_path = data_root / "vyvar.sqlite3"
    cfg.qc_after_calibrate_enabled = False
    self_ = types.SimpleNamespace(config=cfg, db=db)
    return pipeline.AstroPipeline.quick_calibrate_last_import(
        self_, archive_path=archive, master_dark_path=None, masterflat_by_filter={},
        draft_id=draft_id,
        masterflat_by_obs_key=dict(plan.masterflat_by_obs_key),
        master_dark_by_obs_key=dict(plan.dark_master_by_obs_key),
        flat_substitute_by_obs_key=dict(plan.flat_substitute_by_obs_key),
    )


def test_a1_group_without_flat_does_not_lose_the_other_groups(tmp_path: Path) -> None:
    from importer import smart_import_session, smart_scan_source

    db, cam, tel, loc, data_root = _rig(tmp_path)
    lib = data_root / "CalibrationLibrary"
    _register_dark(db, cam, lib)
    rng = np.random.default_rng(3)
    _register_flat(db, cam, tel, lib, "V", _noisy_flat((32, 48), rng, level=20000.0))
    src = tmp_path / "night" / "Lights"
    for i in range(2):
        _write(src / f"V_{i}.fits", np.full((32, 48), 3000.0), _light_hdr("V"))
        _write(src / f"B_{i}.fits", np.full((32, 48), 3000.0), _light_hdr("B"))
    plan = smart_scan_source(source_root=src.parent, calibration_library_root=lib, masterdark_validity_days=90,
                             masterflat_validity_days=200, db=db, id_equipments=cam, id_telescope=tel)
    assert plan.missing_obs_keys == ["B|60|1"]
    g_b = next(g for g in plan.calibration_plan["groups"] if g["filter"] == "B")
    assert g_b["status"] == cp.STATUS_NO_FLAT
    assert any("DARK only" in m for m in g_b["messages"])
    res = smart_import_session(plan=plan, pipeline=_Pipe(db, _Cfg(data_root, loc)), id_equipment=cam,
                               id_telescope=tel, cfg=_Cfg(data_root, loc))
    arch = Path(res.archive_path)
    # one lights tree: every group in Raw/lights, nothing split off into non_calibrated/
    assert not (arch / "non_calibrated").exists()
    assert sorted(p.name for p in (arch / "Raw" / "lights").iterdir()) == ["B_60_1", "V_60_1"]

    out = _calibrate_draft(db, data_root, arch, plan, int(res.draft_id))
    cal = {p.name: fits.getheader(p) for p in (arch / "calibrated" / "lights").rglob("*.fits")}
    assert sorted(cal) == ["B_0.fits", "B_1.fits", "V_0.fits", "V_1.fits"]
    assert all(cal[f"V_{i}.fits"]["VY_CFLAG"] == "DF" for i in range(2))
    assert all(cal[f"B_{i}.fits"]["VY_CFLAG"] == "D" and cal[f"B_{i}.fits"]["VY_WARN"] for i in range(2))
    groups = json.loads((arch / "draft_manifest.json").read_text())["calibration_groups"]
    assert groups["V_60_1"]["relative_only"] is False and groups["V_60_1"]["vy_cflag"] == {"DF": 2}
    assert groups["B_60_1"]["relative_only"] is True and "dark only" in groups["B_60_1"]["reason"]
    assert out["calibration_groups"]["B_60_1"]["relative_only"] is True


def test_a1_legacy_split_draft_calibrates_both_trees(tmp_path: Path) -> None:
    """A draft imported before the fix (groups split into Raw/ and non_calibrated/): both are calibrated."""
    import pipeline
    from config import AppConfig

    ap = tmp_path / "draft_000001"
    dk = _write(tmp_path / "dark.fits", np.full((32, 32), 1000.0), _hdr(EXPTIME=60.0, XBINNING=1))
    for sub, flt in (("Raw/lights/V_60_1", "V"), ("non_calibrated/lights/B_60_1", "B")):
        _write(ap / sub / f"{flt}_001.fits", np.full((32, 32), 3000.0), _light_hdr(flt))
    cfg = AppConfig()
    cfg.database_path = tmp_path / "none.sqlite3"
    cfg.qc_after_calibrate_enabled = False
    out = pipeline.AstroPipeline.quick_calibrate_last_import(
        types.SimpleNamespace(config=cfg, db=None), archive_path=ap, master_dark_path=None,
        masterflat_by_filter={}, master_dark_by_obs_key={"V|60|1": str(dk), "B|60|1": str(dk)},
        masterflat_by_obs_key={"V|60|1": None, "B|60|1": None})
    assert sorted(out["results"]) == ["Raw", "non_calibrated"]
    assert sorted(p.name for p in (ap / "calibrated" / "lights").rglob("*.fits")) == ["B_001.fits", "V_001.fits"]


def _plan_groups(tmp: Path, filt: str, when: datetime) -> dict:
    p = _write(tmp / "src" / f"{filt}_1.fits", np.zeros((8, 8)), _light_hdr(filt, when=when))
    return {f"{filt}|60|1": {"filter": filt, "exposure_s": 60.0, "binning": 1, "gain": 0, "temp": -10.0,
                             "light_paths": [str(p)]}}


def test_a1_flat_from_another_night_is_a_flagged_substitute(tmp_path: Path) -> None:
    db, cam, tel, _loc, data_root = _rig(tmp_path)
    lib = data_root / "CalibrationLibrary"
    rng = np.random.default_rng(4)
    _register_dark(db, cam, lib)
    old = NOW - timedelta(days=400)
    older = NOW - timedelta(days=700)
    _register_flat(db, cam, tel, lib, "V", _noisy_flat((8, 8), rng, level=20000.0), when=older)
    near = _register_flat(db, cam, tel, lib, "V", _noisy_flat((8, 8), rng, level=20000.0), when=old)
    plan = cp.build_calibration_plan(_plan_groups(tmp_path, "V", NOW), db=db, camera_id=cam, telescope_id=tel,
                                     dark_validity_days=90, flat_validity_days=200, temp_tol=0.5)
    g = plan.groups["V|60|1"]
    assert g.flat is not None and Path(g.flat.path) == near.resolve()  # nearest capture date
    assert g.status == cp.STATUS_FLAT_SUBSTITUTE
    assert g.flat_substitute == old.date().isoformat()
    assert g.row()["flat_substitute"] == old.date().isoformat()
    assert any("substitute" in m for m in g.messages)
    # no flat of the filter at all -> MISSING_FLAT (dark only, flagged)
    plan_b = cp.build_calibration_plan(_plan_groups(tmp_path, "B", NOW), db=db, camera_id=cam, telescope_id=tel,
                                       dark_validity_days=90, flat_validity_days=200, temp_tol=0.5)
    assert plan_b.groups["B|60|1"].flat is None and plan_b.groups["B|60|1"].status == cp.STATUS_NO_FLAT


def test_a1_substitute_flat_stamped_in_frame_header(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    lp = _write(tmp_path / "l.fits", np.full((16, 24), 3000.0), _light_hdr("V"))
    dp = _write(tmp_path / "d.fits", np.full((16, 24), 1000.0), _hdr(EXPTIME=60.0, XBINNING=1))
    fp = _write(tmp_path / "f.fits", _noisy_flat((16, 24), rng, level=20000.0), _hdr(IMAGETYP="FLAT", XBINNING=1))
    _d, hdr, _ud, uf = pc._calibrate_one_light_apply_masters_in_ram(
        src=lp, master_dark_path=dp, masterflat_by_filter={"V|60|1": fp}, flat_offsets={str(fp.resolve()): 0.0},
        flat_substitutes={"V|60|1": "2025-09-25"})
    assert uf and hdr["VY_CFLAG"] == "DF"
    assert hdr["VY_FLSUB"] == "2025-09-25" and hdr["VY_WARN"]


def test_a1_relative_only_group_is_red_and_substitute_is_soft(tmp_path: Path) -> None:
    from draft_provenance import calibration_trust_reasons, record_calibration_outcome
    from trust_flag_core import CompTrustThresholds, evaluate_target

    draft = tmp_path / "draft"
    draft.mkdir()
    from trust_flag_core import _calibration_reasons_for_photometry_dir

    record_calibration_outcome(draft, draft_id=7, flags_by_group={"B_60_1": {"D": 3}, "V_60_1": {"DF": 3}},
                               flat_substitute_by_group={"V_60_1": "2025-01-02"})
    hard_b, soft_b = _calibration_reasons_for_photometry_dir(draft / "platesolve" / "B_60_1" / "photometry")
    hard_v, soft_v = calibration_trust_reasons(draft, "V_60_1")
    assert _calibration_reasons_for_photometry_dir(draft / "platesolve" / "V_60_1_R" / "photometry")[1] == soft_v
    assert "calibration incomplete" in hard_b and not soft_b
    assert not hard_v and "2025-01-02" in soft_v
    th = CompTrustThresholds.from_bounds(3, 8)
    base = dict(catalog_id="1", vsx_name="X", n_clean=5, lc_quality="good", check_scatter=0.005, thresholds=th,
                n_frames=50, n_check=50, n_export=50)
    assert evaluate_target(**base)["trust"] == "GREEN"
    assert evaluate_target(**base, calibration_hard=hard_b)["trust"] == "RED"  # -> _dontsubmit/
    assert evaluate_target(**base, calibration_soft=soft_v)["trust"] == "YELLOW"


def test_g12_passthrough_recorded_as_calibration_mode(tmp_path: Path) -> None:
    from draft_provenance import CALIBRATION_MODE_PASSTHROUGH, load_draft_manifest, record_calibration_outcome

    record_calibration_outcome(tmp_path, draft_id=3, flags_by_group={"V_60_1": {"P": 4}})
    assert load_draft_manifest(tmp_path)["calibration_mode"] == CALIBRATION_MODE_PASSTHROUGH


# ---------------------------------------------------------------------------------------------
# G01-29 (decision A2): bad flat pixels -> NaN (no 0.15 floor); flagged in aperture photometry
# ---------------------------------------------------------------------------------------------
def test_a2_mask_marks_defects_not_deep_smooth_vignetting() -> None:
    from calibration import flat_bad_pixel_mask

    rng = np.random.default_rng(11)
    f = _noisy_flat((1500, 2000), rng, vignetting=0.95)  # sensor-size frame, corners at 0.05 of the centre
    f[300, 400] *= 0.5  # dead-ish pixel
    f[:, 200] *= 0.9  # low column
    f /= np.median(f)
    assert np.min(f) < 0.15
    bad = flat_bad_pixel_mask(f)
    assert bad[300, 400]
    assert bad[5:-5, 200].all()
    other = bad.copy()
    other[300, 400] = False
    other[:, 200] = False
    assert int(other.sum()) <= 5  # vignetting is not bad (Gaussian false alarms: ~1 expected)
    # a noiseless flat has no measurable noise: nothing but invalid pixels is bad
    assert not flat_bad_pixel_mask(np.linspace(0.2, 1.0, 400, dtype=np.float32).reshape(20, 20)).any()


def test_a2_mask_cfa_planes_tested_separately() -> None:
    from calibration import flat_bad_pixel_mask

    rng = np.random.default_rng(12)
    f = _noisy_flat((64, 64), rng)
    f[0::2, 0::2] *= 0.6  # R plane at a different level (legacy global normalisation)
    f[20, 20] *= 0.5  # a defect in the R plane
    assert flat_bad_pixel_mask(f, cfa=True).sum() <= 3 and flat_bad_pixel_mask(f, cfa=True)[20, 20]
    assert flat_bad_pixel_mask(f, cfa=False).sum() > 100  # mixed planes would mark a whole plane


def test_a2_calibrated_pixel_is_nan_at_bad_flat_and_vignetting_is_corrected(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    shape = (1500, 2000)
    flat = _noisy_flat(shape, rng, level=20000.0, vignetting=0.95)
    flat[300, 400] = 20000.0 * 0.05  # dead pixel
    lp = _write(tmp_path / "l.fits", 1000.0 + 2000.0 * flat / np.median(flat), _light_hdr("V"))
    dp = _write(tmp_path / "d.fits", np.full(shape, 1000.0), _hdr(EXPTIME=60.0, XBINNING=1))
    fp = _write(tmp_path / "f.fits", flat, _hdr(IMAGETYP="FLAT", XBINNING=1))
    data, hdr, _ud, uf = pc._calibrate_one_light_apply_masters_in_ram(
        src=lp, master_dark_path=dp, masterflat_by_filter={"V|60|1": fp}, flat_offsets={str(fp.resolve()): 0.0})
    assert uf and np.isnan(data[300, 400])
    assert int(hdr["VY_FLBAD"]) >= 1 and "VY_FLFL" not in hdr
    fin = np.isfinite(data)
    assert fin.sum() >= data.size - 4
    # deep corners (flat < 0.15 of the median) are flat-fielded, not floored
    assert np.nanmax(np.abs(data[fin] - 2000.0)) < 1.0


def test_a2_aperture_touching_nan_pixel_is_flagged_not_filled() -> None:
    from aperture_pertarget import measure_flux_on_f_grid, prepare_frame_images

    rng = np.random.default_rng(14)
    img = rng.normal(100.0, 3.0, (80, 80))
    yy, xx = np.mgrid[0:80, 0:80]
    for x0, y0 in ((20.0, 40.0), (60.0, 40.0)):
        img += 5000.0 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.5**2))
    img[40, 61] = np.nan  # bad flat pixel inside the second star's aperture
    xy = np.array([[20.0, 40.0], [60.0, 40.0]])
    fl, _sk, edge = measure_flux_on_f_grid(img, xy, f_grid=[1.5], fwhm_px=3.5, annulus_inner_fwhm=2.7,
                                           annulus_outer_fwhm=5.2)
    assert list(edge) == [False, True]
    assert np.isfinite(fl[1.5][0])
    assert prepare_frame_images(np.ones((4, 4)), 2.0).off is None  # a clean frame: unchanged path


def test_a2_warp_keeps_nan_local_and_fill_finite() -> None:
    from skimage.transform import SimilarityTransform

    from vyvar_alignment_frame import warp_to_reference

    a = np.random.default_rng(15).normal(1000.0, 10.0, (60, 80)).astype(np.float32)
    a[30, 40] = np.nan
    out = warp_to_reference(a, a.shape, SimilarityTransform(translation=(3.2, -2.6)).inverse)
    assert 0 < int(np.isnan(out).sum()) <= 16
    assert np.isfinite(out[0, 0]) and abs(out[0, 0] - np.nanmedian(a)) < 1e-2  # the fill


# ---------------------------------------------------------------------------------------------
# G01-24 (decision A3): dark rebin convention measured, never defaulted
# ---------------------------------------------------------------------------------------------
def test_a3_dark_rebin_without_convention_raises() -> None:
    d = np.full((8, 8), 10.0, dtype=np.float32)
    with pytest.raises(MasterResamplingError, match="convention"):
        resample_master_to_light_binning(d, master_binning=1, light_binning=2, kind="dark")
    s, _ = resample_master_to_light_binning(d, master_binning=1, light_binning=2, kind="dark", dark_resample_mode="sum")
    m, _ = resample_master_to_light_binning(d, master_binning=1, light_binning=2, kind="dark", dark_resample_mode="mean")
    assert float(s[0, 0]) == 40.0 and float(m[0, 0]) == 10.0
    same, bf = resample_master_to_light_binning(d, master_binning=2, light_binning=2, kind="dark")
    assert bf == 1 and same is not None


def test_a3_undecided_convention_stops_calibration() -> None:
    from cal_diag import (
        CONVENTION_UNDECIDED,
        CalDiagGateResult,
        CalDiagSession,
        DarkConventionUndecidedError,
        raise_if_dark_convention_undecided,
    )

    s = CalDiagSession()
    raise_if_dark_convention_undecided(s)
    s.gate_results["k"] = CalDiagGateResult(obs_group_key="V|60|2", dark_path="/x/d.fits", light_binning=2,
                                            status="ABORT", convention="SUM", aborted=True,
                                            abort_reason=CONVENTION_UNDECIDED, block_factor=2, messages=["why"])
    with pytest.raises(DarkConventionUndecidedError, match="V\\|60\\|2"):
        raise_if_dark_convention_undecided(s)


def test_a3_same_binning_dark_preferred_over_rebin(tmp_path: Path) -> None:
    db, cam, _tel, _loc, data_root = _rig(tmp_path)
    lib = data_root / "CalibrationLibrary"
    _register_dark(db, cam, lib, binning=1, shape=(16, 16))
    d2 = _register_dark(db, cam, lib / "b2", binning=2, shape=(8, 8), when=NOW - timedelta(days=3))
    r = cl.find_dark(db, camera_id=cam, gain=0, offset=10, exposure=60.0, binning=2, temp=-10.0, temp_tol=0.5,
                     light_dates=[NOW], validity_days=90)
    assert Path(r.master.path) == d2.resolve() and r.rebin_from == 1


# ---------------------------------------------------------------------------------------------
# G01-02 / G01-03: normalise first; no pedestal removal from a normalised flat
# ---------------------------------------------------------------------------------------------
def test_g0102_normalised_flat_with_offset_is_refused(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    fp = _write(tmp_path / "nf.fits", _noisy_flat((16, 24), rng, vignetting=0.5), _hdr(IMAGETYP="FLAT", XBINNING=1))
    with pytest.raises(MasterResamplingError, match="normalised"):
        get_processed_master(fp, 1, kind="flat", flat_offset_adu=24.5)
    # a raw-ADU flat keeps the rule (offset removed at use)
    fa = _write(tmp_path / "af.fits", _noisy_flat((16, 24), rng, level=30000.0), _hdr(IMAGETYP="FLAT", XBINNING=1))
    pm = get_processed_master(fa, 1, kind="flat", flat_offset_adu=24.5)
    assert abs(float(np.median(pm.data)) - (30000.0 - 24.5)) < 300.0


def test_g0103_legacy_flat_invalid_pixels_are_bad_not_amplified(tmp_path: Path) -> None:
    rng = np.random.default_rng(22)
    flat = _noisy_flat((40, 60), rng, level=20000.0, noise=0.002)
    flat[:, :3] = 0.0
    flat[10, 10] = np.nan
    lp = _write(tmp_path / "l.fits", np.full((40, 60), 3000.0), _light_hdr("V"))
    dp = _write(tmp_path / "d.fits", np.full((40, 60), 1000.0), _hdr(EXPTIME=60.0, XBINNING=1))
    fp = _write(tmp_path / "f.fits", flat, _hdr(IMAGETYP="FLAT", XBINNING=1))  # imported: no VYFLNRD
    data, hdr, _ud, _uf = pc._calibrate_one_light_apply_masters_in_ram(
        src=lp, master_dark_path=dp, masterflat_by_filter={"V|60|1": fp}, flat_offsets={str(fp.resolve()): 0.0})
    assert np.isnan(data[:, :3]).all() and np.isnan(data[10, 10])  # was 13333 (x 6.7)
    good = data[20, 30]
    fm = float(np.nanmedian(flat[:, 3:]))
    assert abs(good - 2000.0 * fm / flat[20, 30]) < 1e-2
    assert abs(float(hdr["VY_FLATM"]) - fm) < 1e-3


def test_g0103_normalize_flat_master_marks_invalid_nan() -> None:
    m = np.full((8, 8), 2.0, dtype=np.float32)
    m[0, 0] = 0.0
    out = normalize_flat_master(m, fits.Header())
    assert np.isnan(out[0, 0]) and float(out[4, 4]) == pytest.approx(1.0)


def test_byte_identical_without_bad_flat_pixels(tmp_path: Path) -> None:
    """(L - D) / (flat / median) exactly as before on every pixel that is not a bad flat pixel."""
    from calibration import flat_bad_pixel_mask

    rng = np.random.default_rng(30)
    flat = _noisy_flat((300, 400), rng, level=30000.0, vignetting=0.3, noise=0.003)
    light = (1000.0 + 2500.0 * rng.random((300, 400))).astype(np.float32)
    lp = _write(tmp_path / "l.fits", light, _light_hdr("V"))
    dark = np.full((300, 400), 1000.0, dtype=np.float32)
    dp = _write(tmp_path / "d.fits", dark, _hdr(EXPTIME=60.0, XBINNING=1))
    fp = _write(tmp_path / "f.fits", flat, _hdr(IMAGETYP="FLAT", XBINNING=1))
    bad = flat_bad_pixel_mask((flat / np.median(flat)).astype(np.float32))
    assert int(bad.sum()) <= 3  # the false-alarm level of the test (~1 expected)
    data, hdr, _ud, _uf = pc._calibrate_one_light_apply_masters_in_ram(
        src=lp, master_dark_path=dp, masterflat_by_filter={"V|60|1": fp}, flat_offsets={str(fp.resolve()): 0.0})
    fm = float(np.nanmedian(flat))
    ref = ((light - dark) / (flat / fm).astype(np.float32)).astype(np.float32)
    assert np.array_equal(data[~bad], ref[~bad]) and np.isnan(data[bad]).all()
    assert int(hdr["VY_FLBAD"]) == int(bad.sum())


def test_prepare_flat_refuses_mostly_invalid_flat() -> None:
    from calibration import prepare_flat_for_division

    f = np.zeros((10, 10), dtype=np.float32)
    f[:4] = 1.0
    with pytest.raises(MasterResamplingError, match="not a usable flat"):
        prepare_flat_for_division(ProcessedMasterResult(f, 1, False, 1), cfa=False, name="x")
