"""Step 7 (T17) - calibration, binning / gain, resampler, CCD equation (tag cal, 2026-10-31).

Findings G01-06/G01-24, G01-07, G01-42, G02-25, G06-34, G06-25 and the C2 photon-transfer tooling
(/root/scr/s7/lit_cal.md, FINAL conclusions approved by Milan 2026-10-31). Synthetic data with known truth.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))

# G01-42: the synthetic library / lights of the calibration spec tests
from test_calibration_spec_2026 import (  # noqa: E402
    T0,
    _lights,
    _raw_darks,
    _raw_flats,
    cl,
    cp,
)


@pytest.fixture()
def env(tmp_path: Path):
    """The calibration spec test environment (database, cameras, telescope, library, source folder)."""
    from database import VyvarDatabase

    db = VyvarDatabase(tmp_path / "t.sqlite3")
    cam = db.insert_equipment("C5A-150M", "cam", "CMOS", "x", 3.76)
    cam2 = db.insert_equipment("QHY294MM", "cam2", "CMOS", "x", 4.63)
    tel = db.insert_telescope("AZ800", "az", 800.0, 5480.0)
    cl.ensure_schema(db)
    return db, cam, cam2, tel, tmp_path / "lib", tmp_path / "src"


# --------------------------------------------------------------------------- #
# G01-42 - parallel calibration parity (MP-CALIB-ON-2026-10 supersedes D-MP-CALIB-OFF-01)
# --------------------------------------------------------------------------- #
def _md5(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_g01_42_calibration_1_vs_2_workers_identical_md5(env, tmp_path):
    """4 synthetic lights + master dark + master flat, max_workers 1 vs 2: the md5 of every calibrated
    FITS data unit, the header cards and the per-group calibration flags are identical."""
    from config import AppConfig
    from importer import smart_scan_source
    from pipeline_calibrate import calibrate_lights_to_calibrated

    db, cam, _, tel, lib, src = env
    rng = np.random.default_rng(42)
    raw = tmp_path / "raw_cal"
    darks = _raw_darks(raw / "darks", 60.0, 5, T0 + timedelta(hours=8), rng)
    dflats = _raw_darks(raw / "darkflats", 0.5, 5, T0 + timedelta(hours=9), rng)
    flats = _raw_flats(raw / "flats", 5, T0 + timedelta(hours=9), rng, exp=0.5)
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
    runs = {}
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
        assert len(files) == 4
        runs[w] = {
            "md5": {f.name: _md5(fits.getdata(f)) for f in files},
            "hdr": {f.name: [(c.keyword, c.value) for c in fits.getheader(f).cards
                             if c.keyword not in ("HISTORY", "DATE")] for f in files},
            "flags": st["flags_by_group"],
        }
    assert runs[1]["md5"] == runs[2]["md5"]
    assert runs[1]["hdr"] == runs[2]["hdr"]
    assert runs[1]["flags"] == runs[2]["flags"]


# --------------------------------------------------------------------------- #
# G01-06 / G01-24 - gain / read noise at bin N: no default convention
# --------------------------------------------------------------------------- #
def _bin_hdr(binning: int, conv: str | None = None, **cards) -> fits.Header:
    h = fits.Header()
    h["XBINNING"] = binning
    h["YBINNING"] = binning
    h["GAIN"] = (0.0, "Gain")
    if conv:
        h["VYBINCNV"] = conv
    for k, v in cards.items():
        h[k] = v
    return h


def _cmos(**kw):
    from camera_profile import CameraProfile, CurvePoint

    base = dict(equipment_id=9, name="cam", header_gain="setting", default_setting=0.0, adc_bits=14,
                sensor_class="cmos", points=(CurvePoint(0.0, 1, 3.17, 7.6, "equipment"),))
    base.update(kw)
    return CameraProfile(**base)


def test_g01_06_default_bin_mode_removed():
    import camera_profile

    assert not hasattr(camera_profile, "DEFAULT_BIN_MODE")
    assert camera_profile.CameraProfile().bin_mode is None


def test_g01_06_no_bin_mode_cal_diag_sum_cmos_gain_unchanged_rn_x_n():
    from param_resolver import resolve_gain, resolve_read_noise

    prof = _cmos()
    g = resolve_gain(_bin_hdr(2, "SUM"), profile=prof)
    r = resolve_read_noise(_bin_hdr(2, "SUM"), profile=prof)
    assert g.ok and g.value == pytest.approx(3.17)  # sum: e-/ADU unchanged (old default: x4)
    assert r.ok and r.value == pytest.approx(15.2)  # digital sum of a CMOS: RN x N
    assert g.bin_mode == "sum" and g.bin_mode_source == "cal_diag"


def test_g01_06_cal_diag_mean_means_average():
    from param_resolver import resolve_gain, resolve_read_noise

    prof = _cmos()
    assert resolve_gain(_bin_hdr(2, "MEAN"), profile=prof).value == pytest.approx(3.17 * 4)
    assert resolve_read_noise(_bin_hdr(2, "MEAN"), profile=prof).value == pytest.approx(15.2)


def test_g01_06_nothing_known_stops():
    from config import AppConfig
    from param_resolver import resolve_gain

    prof = _cmos()
    g = resolve_gain(_bin_hdr(2), profile=prof)
    assert not g.ok and "binning convention unknown" in " ".join(g.warnings)
    # bin 1 needs no convention
    assert resolve_gain(_bin_hdr(1), profile=prof).value == pytest.approx(3.17)
    # the run's one authority raises (like an unknown gain)
    import rig_photometry

    cfg = AppConfig()
    cfg.database_path = ""
    with pytest.raises(ValueError, match="gain / read noise unknown"):
        rig_photometry.photometric_gain_read_noise(_bin_hdr(2), cfg, None)


def test_g01_06_bin_mode_contradicting_cal_diag_stops():
    from param_resolver import resolve_gain, resolve_read_noise

    prof = _cmos(bin_mode="average", bin_mode_source="equipment")
    g = resolve_gain(_bin_hdr(2, "SUM"), profile=prof)
    assert not g.ok and "contradicts CAL-DIAG SUM" in " ".join(g.warnings)
    # a header read-noise card does not paper over the contradiction
    r = resolve_read_noise(_bin_hdr(2, "SUM", RDNOISE=15.0), profile=prof)
    assert not r.ok
    # consistent pairs pass: sum / charge with SUM, average with MEAN
    assert resolve_gain(_bin_hdr(2, "SUM"), profile=_cmos(bin_mode="sum", bin_mode_source="equipment")).ok
    assert resolve_gain(_bin_hdr(2, "SUM"), profile=_cmos(bin_mode="charge", bin_mode_source="equipment")).ok
    assert resolve_gain(_bin_hdr(2, "MEAN"), profile=_cmos(bin_mode="average", bin_mode_source="catalog")).ok


def test_g01_06_measured_point_at_bin_n_used_as_is():
    from camera_profile import CurvePoint
    from param_resolver import resolve_gain, resolve_read_noise

    prof = _cmos(points=(CurvePoint(0.0, 2, 3.85, 11.0, "measured"), CurvePoint(0.0, 1, 3.17, 7.6, "equipment")))
    for conv in (None, "SUM", "MEAN"):  # no convention needed, none applied
        g = resolve_gain(_bin_hdr(2, conv), profile=prof)
        r = resolve_read_noise(_bin_hdr(2, conv), profile=prof)
        assert (g.value, r.value) == pytest.approx((3.85, 11.0)) and g.exact_binning and g.bin_mode is None


def test_g01_06_cal_diag_sum_read_noise_of_ccd_or_unknown_sensor_stops():
    from param_resolver import resolve_gain, resolve_read_noise

    for sensor in ("ccd", None):
        prof = _cmos(sensor_class=sensor)
        assert resolve_gain(_bin_hdr(2, "SUM"), profile=prof).value == pytest.approx(3.17)
        r = resolve_read_noise(_bin_hdr(2, "SUM"), profile=prof)
        assert not r.ok and "charge binning" in " ".join(r.warnings)
    # BIN_MODE charge says it: RN x 1
    r = resolve_read_noise(_bin_hdr(2, "SUM"), profile=_cmos(sensor_class="ccd", bin_mode="charge",
                                                             bin_mode_source="equipment"))
    assert r.ok and r.value == pytest.approx(7.6)


def test_g01_06_sensor_class_and_catalog_qhy294_11mp_sum(tmp_path):
    from camera_profile import _sensor_class, catalog_mode, catalog_model, load_profile
    from database import VyvarDatabase

    mode = catalog_mode(catalog_model("qhy_qhy294m_pro"), "11mp")
    assert mode["binning"] == "sum" and "CAL-DIAG 527" in mode["binning_source"]
    assert _sensor_class("CMOS", None) == "cmos" and _sensor_class("Kodak CCD", None) == "ccd"
    assert _sensor_class("x", None) is None and _sensor_class("", {"key": "m"}) == "cmos"
    p = tmp_path / "v.sqlite3"
    db = VyvarDatabase(p)
    db.conn.execute("DELETE FROM EQUIPMENTS")
    # the 527 camera with BIN_MODE NULL: the catalog now says sum (was: default average -> gain x 4)
    db.conn.execute("INSERT INTO EQUIPMENTS (ID, CAMERANAME, GAIN_ADU, READNOISE_E, ADC_BITS, MODEL_KEY, "
                    "READOUT_MODE, DEFAULT_GAIN_SETTING) VALUES (1, 'QHY294MM', 3.17, 7.6, 14, 'qhy294m_pro', "
                    "'11mp', 0)")
    db.conn.commit()
    db.conn.close()
    prof = load_profile(p, 1)
    assert (prof.bin_mode, prof.bin_mode_source, prof.sensor_class) == ("sum", "catalog", "cmos")


def test_g01_06_cal_diag_stamps_the_measured_convention_only_when_rebinned():
    from cal_diag import CalDiagGateResult, apply_cal_diag_headers

    def _hdr(conv: str, bf: int) -> fits.Header:
        h = fits.Header()
        apply_cal_diag_headers(h, CalDiagGateResult("g", "d.fits", 2, "PASS", conv, block_factor=bf))
        return h

    assert _hdr("SUM", 2)["VYBINCNV"] == "SUM" and _hdr("MEAN", 4)["VYBINCNV"] == "MEAN"
    assert "VYBINCNV" not in _hdr("NONE", 1)  # dark at the light binning: nothing measured


# --------------------------------------------------------------------------- #
# G06-34 - LSB granularity (container scale) and photon transfer vs header EGAIN
# --------------------------------------------------------------------------- #
def _x4_frame(rng, shape=(64, 96), level=1500.0, binning_avg: int = 1) -> np.ndarray:
    """14-bit ADU (Poisson sky) written x4 into the 16-bit container; ``binning_avg`` N: N x N average of
    those values rounded to an integer (in-camera average binning)."""
    n = int(binning_avg)
    native = rng.poisson(level, (shape[0] * n, shape[1] * n)).astype(np.float64)
    cont = native * 4.0
    if n > 1:
        cont = np.rint(cont.reshape(shape[0], n, shape[1], n).mean(axis=(1, 3)))
    return cont


def test_g06_34_granularity_exact_and_check_rules():
    from gain_ptc import container_scale_check, lsb_granularity

    rng = np.random.default_rng(3)
    a = _x4_frame(rng)
    assert lsb_granularity(a) == 4 and lsb_granularity(a.astype(np.uint16)) == 4
    a_sat = a.copy()
    a_sat[0, 0] = 65535.0  # container ceiling (unsaturated pixels only)
    assert lsb_granularity(a_sat) == 4
    assert lsb_granularity(a + 0.25) is None  # not raw integers
    avg = _x4_frame(rng, binning_avg=2)
    assert lsb_granularity(avg) == 1
    assert container_scale_check(4, 14, 1)[0] == "ok" and container_scale_check(4, 14, 2)[0] == "ok"
    assert container_scale_check(4, 16, 2)[0] == "contradiction"  # 527 copy with ADC_BITS 16
    assert container_scale_check(1, 14, 1)[0] == "contradiction"
    assert container_scale_check(1, 14, 2)[0] == "ambiguous"  # average-binned x4: WARNING only
    assert container_scale_check(None, 14, 1)[0] == "unmeasured"


def _int_lights(lights_dir, rng, *, binning: int, avg: bool, n: int = 2) -> None:
    for i in range(n):
        d = _x4_frame(rng, binning_avg=2 if avg else 1)
        h = fits.Header()
        h["IMAGETYP"] = "Light Frame"
        h["EXPTIME"] = 60.0
        h["DATE-OBS"] = (T0 + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S")
        h["XBINNING"] = binning
        h["YBINNING"] = binning
        h["FILTER"] = "V"
        lights_dir.mkdir(parents=True, exist_ok=True)
        fits.writeto(lights_dir / f"L_{i}.fits", d.astype(np.uint16), header=h, overwrite=True)


def _cal_cfg_with_adc(tmp_path, adc_bits: int):
    from config import AppConfig
    from database import VyvarDatabase

    db = VyvarDatabase(tmp_path / "t.sqlite3")
    cam = db.insert_equipment("QHY294MM", "cam", "CMOS", "x", 4.63)
    db.conn.execute("UPDATE EQUIPMENTS SET ADC_BITS = ? WHERE ID = ?", (int(adc_bits), int(cam)))
    db.conn.commit()
    db.conn.close()
    cfg = AppConfig(project_root=tmp_path)
    cfg.database_path = str(tmp_path / "t.sqlite3")
    return cfg, cam


def test_g06_34_calibration_stops_on_contradiction_stamps_vy_lsb(tmp_path):
    from gain_ptc import ContainerScaleError
    from pipeline_calibrate import calibrate_lights_to_calibrated

    rng = np.random.default_rng(5)
    lights = tmp_path / "lights"
    _int_lights(lights, rng, binning=1, avg=False)
    cfg, cam = _cal_cfg_with_adc(tmp_path, 16)
    with pytest.raises(ContainerScaleError, match="granularity 4"):
        calibrate_lights_to_calibrated(lights_root=lights, calibrated_root=tmp_path / "cal16", master_dark_path=None,
                                       masterflat_by_filter={}, pipeline_config=cfg, equipment_id=cam)
    cfg.database_path = ""
    tmp2 = tmp_path / "ok"
    tmp2.mkdir()
    cfg, cam = _cal_cfg_with_adc(tmp2, 14)
    st = calibrate_lights_to_calibrated(lights_root=lights, calibrated_root=tmp2 / "cal", master_dark_path=None,
                                        masterflat_by_filter={}, pipeline_config=cfg, equipment_id=cam)
    assert next(iter(st["container_scale"].values()))["status"] == "ok"
    outs = sorted((tmp2 / "cal").rglob("*.fits"))
    assert outs and all(fits.getheader(f)["VY_LSB"] == 4 for f in outs)


def test_g06_34_average_binned_x4_frame_is_a_warning_only(tmp_path, caplog):
    from pipeline_calibrate import calibrate_lights_to_calibrated

    rng = np.random.default_rng(6)
    lights = tmp_path / "lights"
    _int_lights(lights, rng, binning=2, avg=True)
    cfg, cam = _cal_cfg_with_adc(tmp_path, 14)
    with caplog.at_level("WARNING"):
        st = calibrate_lights_to_calibrated(lights_root=lights, calibrated_root=tmp_path / "cal",
                                            master_dark_path=None, masterflat_by_filter={}, pipeline_config=cfg,
                                            equipment_id=cam)
    rec = next(iter(st["container_scale"].values()))
    assert rec["status"] == "ambiguous" and rec["granularity"] == 1
    assert "G06-34" in caplog.text


def test_g06_34_x4_container_and_16_bit_frames_give_the_same_relative_error():
    """Same photons: a 14-bit camera (x4 container, file gain = native / 4) and a 16-bit camera (file gain =
    native) give identical relative CCD-equation errors."""
    from camera_profile import CameraProfile, CurvePoint
    from ccd_noise import ccd_rel_error
    from gain_ptc import container_scale

    g_nat, rn, F_e, S_e = 3.8, 10.0, 50000.0, 1500.0
    rel = []
    for adc in (14, 16):
        prof = CameraProfile(equipment_id=1, header_gain="setting", default_setting=0.0, adc_bits=adc,
                             bin_mode="sum", bin_mode_source="equipment", points=(CurvePoint(0.0, 1, g_nat, rn, "equipment"),))
        from param_resolver import resolve_gain

        g_file = resolve_gain(_bin_hdr(1), profile=prof).value / container_scale(adc)
        rel.append(ccd_rel_error(F_e / g_file, S_e / g_file, 30.0, 400.0, gain=g_file, read_noise=rn))
    assert rel[0] == pytest.approx(rel[1], rel=1e-12)


def test_g06_34_measured_ptc_point_beats_disagreeing_header_egain():
    from camera_profile import CurvePoint
    from param_resolver import resolve_gain

    prof = _cmos(bin_mode="sum", bin_mode_source="equipment", header_gain=None,
                 points=(CurvePoint(0.0, 2, 3.85, 11.0, "measured", e_per_adu_err=0.08),
                         CurvePoint(0.0, 1, 3.17, 7.6, "equipment")))
    # EGAIN in container units (x4 too small): the photon transfer wins, WARNING + facts
    r = resolve_gain(_bin_hdr(2, EGAIN=(0.96, "e-/ADU")), profile=prof)
    assert r.ok and r.source == "measured" and r.value == pytest.approx(3.85)
    assert r.facts["winner"] == "measured_over_header" and r.warnings
    # within 3 x gain_err: the header stays the session truth
    r = resolve_gain(_bin_hdr(2, EGAIN=(3.9, "e-/ADU")), profile=prof)
    assert r.source == "header" and r.value == pytest.approx(3.9)
    # no measured point at that binning: header as before
    assert resolve_gain(_bin_hdr(1, EGAIN=(0.96, "e-/ADU")), profile=prof).source == "header"


# --------------------------------------------------------------------------- #
# G06-25 - CCD equation: dark-current shot noise and digitisation
# --------------------------------------------------------------------------- #
def _pull_std(*, g: float, rn: float, sky_e: float, dark_e: float, flux_e: float, quantise: bool,
              pixel_noise_e: float, n_trials: int = 20000, area: int = 20, n_sky: int = 2000) -> float:
    """Monte Carlo aperture photometry (mean sky) on dark-subtracted pixels: std of (F_meas - F_true) / sigma
    with sigma from ccd_noise and the per-pixel noise ``pixel_noise_e``."""
    from ccd_noise import ccd_variance_adu2

    rng = np.random.default_rng(11)

    def _pix(shape, extra_e):
        e = rng.poisson(sky_e + dark_e + extra_e, shape) + rng.normal(0.0, rn, shape)
        adu = e / g
        if quantise:
            adu = np.rint(adu)
        return adu - dark_e / g  # master dark removes the dark level, not its shot noise

    ap = _pix((n_trials, area), flux_e / area)
    sky = _pix((n_trials, n_sky), 0.0)
    f = ap.sum(axis=1) - area * sky.mean(axis=1)
    var = ccd_variance_adu2(flux_e / g, sky_e / g, float(area), float(n_sky), gain=g, read_noise=pixel_noise_e,
                            sky_estimator="mean")
    return float(np.std((f - flux_e / g) / np.sqrt(var)))


def test_g06_25_injected_dark_equal_to_sky_pull_one_only_with_dark_term():
    from rig_photometry import ccd_pixel_noise_e

    g, rn, sky_e, t = 1.5, 5.0, 400.0, 60.0
    dark_e = sky_e  # uncooled camera / long exposure
    h = fits.Header()
    h["EXPTIME"] = t
    h["VY_CDKR"] = dark_e / g / t  # CAL-DIAG fitted slope [file ADU/s per light pixel]
    noise, facts = ccd_pixel_noise_e(h, gain_file=g, read_noise_e=rn, rn_measured_at_binning=True)
    assert facts["dark_current"] == "measured" and facts["dark_e_per_px"] == pytest.approx(dark_e)
    assert noise == pytest.approx(np.sqrt(rn ** 2 + dark_e))
    p_with = _pull_std(g=g, rn=rn, sky_e=sky_e, dark_e=dark_e, flux_e=2000.0, quantise=False, pixel_noise_e=noise)
    p_without = _pull_std(g=g, rn=rn, sky_e=sky_e, dark_e=dark_e, flux_e=2000.0, quantise=False, pixel_noise_e=rn)
    assert abs(p_with - 1.0) < 0.02
    assert p_without > 1.15


def test_g06_25_quantised_average_binned_data_pull_one_only_with_digitisation():
    from rig_photometry import ccd_pixel_noise_e

    g, rn, sky_e = 12.5, 3.0, 60.0  # C5A bin-4 average type: 1 ADU = 12.5 e-, sky-limited faint star
    h = fits.Header()
    h["VY_LSB"] = 1
    noise, facts = ccd_pixel_noise_e(h, gain_file=g, read_noise_e=rn, rn_measured_at_binning=False)
    assert facts["digitization"] == "added" and facts["digitization_e2"] == pytest.approx(g * g / 12.0)
    assert facts["dark_current"] == "unmeasured"
    p_with = _pull_std(g=g, rn=rn, sky_e=sky_e, dark_e=0.0, flux_e=50.0, quantise=True, pixel_noise_e=noise)
    p_without = _pull_std(g=g, rn=rn, sky_e=sky_e, dark_e=0.0, flux_e=50.0, quantise=True, pixel_noise_e=rn)
    assert abs(p_with - 1.0) < 0.02
    assert p_without > 1.05


def test_g06_25_rn_measured_at_the_binning_already_holds_the_digitisation():
    from rig_photometry import ccd_pixel_noise_e

    h = fits.Header()
    h["VY_LSB"] = 4
    noise, facts = ccd_pixel_noise_e(h, gain_file=0.95, read_noise_e=10.0, rn_measured_at_binning=True)
    assert noise == pytest.approx(10.0) and facts["rn_includes_digitization"] and facts["digitization"] == "in_read_noise"


def test_g06_25_photometric_gain_folds_the_terms_and_records_them(tmp_path):
    from config import AppConfig
    from database import VyvarDatabase
    from rig_photometry import photometric_gain_read_noise

    db = VyvarDatabase(tmp_path / "v.sqlite3")
    db.conn.execute("DELETE FROM EQUIPMENTS")
    db.conn.execute("INSERT INTO EQUIPMENTS (ID, CAMERANAME, GAIN_ADU, READNOISE_E, ADC_BITS, MODEL_KEY, READOUT_MODE, "
                    "BIN_MODE, DEFAULT_GAIN_SETTING) VALUES (1, 'QHY294MM', 3.17, 7.6, 14, 'qhy294m_pro', '11mp', "
                    "'sum', 0)")
    db.conn.commit()
    db.conn.close()
    cfg = AppConfig()
    cfg.database_path = str(tmp_path / "v.sqlite3")
    h = _bin_hdr(2, "SUM", EXPTIME=60.0, VY_CDKR=0.0018 * 4, VY_LSB=4)  # 527-like numbers
    pg = photometric_gain_read_noise(h, cfg, 1)
    g = 3.17 / 4
    dark_e = 0.0018 * 4 * 60.0 * g
    assert pg.read_noise_camera == pytest.approx(15.2)
    assert pg.read_noise == pytest.approx(np.sqrt(15.2 ** 2 + dark_e + (g * 4) ** 2 / 12.0))
    assert pg.read_noise / 15.2 - 1.0 < 5e-3  # 527: RN term +0.25 %, total sigma (sky-dominated) < 0.1 %
    f = pg.read_noise_resolved.facts
    assert f["dark_current"] == "measured" and f["digitization"] == "added" and not f["rn_includes_digitization"]
    assert pg.gain_resolved.facts["container_scale_measured"] == 4


def test_g06_25_cal_diag_dark_rate_per_light_pixel_and_header():
    from cal_diag import CalDiagGateResult, PedestalResult, apply_cal_diag_headers, dark_rate_per_light_pixel

    ped = PedestalResult(24.5, 0.01, 0.002, "FITTED", "INTERCEPT", 2)
    assert dark_rate_per_light_pixel(ped, "SUM", 2) == pytest.approx(0.008)
    assert dark_rate_per_light_pixel(ped, "MEAN", 2) == pytest.approx(0.002)
    assert dark_rate_per_light_pixel(ped, "NONE", 1) == pytest.approx(0.002)
    assert dark_rate_per_light_pixel(PedestalResult(24.5, 0.01, -0.001, "FITTED", "INTERCEPT", 2), "SUM", 2) == 0.0
    assert dark_rate_per_light_pixel(PedestalResult(24.5, 0.01, 0.0, "UNKNOWN", "SINGLE_MASTER_MEDIAN", 1),
                                     "NONE", 1) is None
    h = fits.Header()
    apply_cal_diag_headers(h, CalDiagGateResult("g", "d", 2, "PASS", "SUM", block_factor=2, dark_rate_adu_s=0.008))
    assert h["VY_CDKR"] == pytest.approx(0.008)


# --------------------------------------------------------------------------- #
# G01-07 - sky surface: star-mask scales in FWHM units, coverage mask instead of a 40 px band
# --------------------------------------------------------------------------- #
def _star_field(rng, *, h: int = 384, w: int = 384, sigma_px: float = 1.5, n: int = 250, noise: float = 8.0):
    yy, xx = np.mgrid[0:h, 0:w]
    sky = 1000.0 + 0.6 * xx + 0.3 * yy
    img = sky + rng.normal(0.0, noise, (h, w))
    for _ in range(n):
        cy, cx = rng.uniform(5, h - 5), rng.uniform(5, w - 5)
        img += rng.uniform(100, 4000) * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma_px**2)))
    return img.astype(np.float32), sky


def test_g01_07_mask_scales_with_fwhm():
    from pipeline_calibrate import _fit_subtract_preprocess_sky_surface

    img, _ = _star_field(np.random.default_rng(1))
    cover = (img.shape[0] // 4) * (img.shape[1] // 4)  # subsampled pixels (step 4)
    excl = {}
    for fw in (2.0, 4.0):
        _, st = _fit_subtract_preprocess_sky_surface(img, order=1, fwhm_px=fw)
        assert st["sky_surface_applied"] and st["sky_surface_fwhm_source"] == "header"
        assert st["sky_surface_fwhm_px"] == pytest.approx(fw)
        excl[fw] = cover - st["sky_surface_n_fit_pixels"]
    # stamps (3.5 FWHM) and dilation (1 FWHM) are in FWHM units: the masked area grows ~ FWHM^2
    assert 2.5 < excl[4.0] / excl[2.0] < 4.5


def test_g01_07_fwhm_measured_from_pass1_detections_with_warning(caplog):
    from pipeline_calibrate import _fit_subtract_preprocess_sky_surface

    img, sky = _star_field(np.random.default_rng(2), sigma_px=1.5)
    with caplog.at_level("WARNING"):
        out, st = _fit_subtract_preprocess_sky_surface(img, order=1)
    truth = 1.5 * 2.3548
    assert st["sky_surface_applied"] and st["sky_surface_fwhm_source"] == "measured"
    assert abs(st["sky_surface_fwhm_px"] / truth - 1.0) < 0.15
    assert "no QC FWHM on record" in caplog.text
    left = out - (img - sky)
    assert float(np.percentile(left, 99) - np.percentile(left, 1)) < 0.03 * float(sky.max() - sky.min())


def test_g01_07_fwhm_unknown_skips_with_warning(caplog):
    from pipeline_calibrate import _fit_subtract_preprocess_sky_surface

    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:128, 0:128]
    img = (1000.0 + 0.5 * xx + rng.normal(0, 5.0, (128, 128))).astype(np.float32)  # no stars
    with caplog.at_level("WARNING"):
        out, st = _fit_subtract_preprocess_sky_surface(img, order=1)
    assert not st["sky_surface_applied"] and st["sky_surface_skip_reason"] == "fwhm_unknown"
    assert st["sky_surface_fwhm_source"] == "unknown" and "FWHM unknown" in caplog.text
    np.testing.assert_array_equal(out, img)


def test_g01_07_coverage_mask_not_a_fixed_40px_band():
    """The fit uses every covered pixel except one DAO-kernel half-width at the coverage edge: NaN regions
    (no data) are excluded through the coverage mask, and the frame edge is no longer a 40 px dead band."""
    from pipeline_calibrate import _dao_kernel_half_width, _fit_subtract_preprocess_sky_surface

    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w]
    sky = (1000.0 + 0.8 * xx + 0.4 * yy).astype(np.float32)
    img = sky.copy()
    img[:, :60] = np.nan  # no coverage (e.g. outside a footprint)
    out, st = _fit_subtract_preprocess_sky_surface(img, order=1, fwhm_px=3.0)
    hw = _dao_kernel_half_width(3.0)
    n_cover = len(range(0, h, 4)) * len(range(0, w, 4))
    yy_s, xx_s = np.mgrid[0:h:4, 0:w:4]
    inside = (yy_s >= hw) & (yy_s < h - hw) & (xx_s >= 60 + hw) & (xx_s < w - hw)
    assert st["sky_surface_n_fit_pixels"] == int(inside.sum()) < n_cover
    assert st["sky_surface_n_fit_pixels"] > ((h - 80) // 4) * ((w - 60 - 80) // 4)  # more than the old band
    assert np.all(np.isnan(out[:, :60]))
    assert float(np.nanmax(out) - np.nanmin(out)) < 0.5  # the gradient is gone up to the coverage edge


def test_g01_07_qc_enrich_passes_the_frame_fwhm(tmp_path):
    from pipeline_calibrate import _qc_enrich_one_frame

    img, _ = _star_field(np.random.default_rng(4), h=256, w=256, n=120)
    fp = tmp_path / "light.fits"
    h = fits.Header()
    h["VY_FWHM"] = 3.4
    h["VY_QCBG"] = 1000.0
    fits.writeto(fp, img, header=h)
    row = _qc_enrich_one_frame(str(fp), sky_order=1, prefilter_status=None, target_ra=None, target_dec=None,
                               inject_pointing_only_if_missing=True, detection_sigma=4.5)
    assert row["sky_surface_fwhm_source"] == "header" and row["sky_surface_fwhm_px"] == pytest.approx(3.4)


# --------------------------------------------------------------------------- #
# G02-25 - one resampler: WCS-registered frames go through warp_to_reference
# --------------------------------------------------------------------------- #
def _align_field(tmp_path):
    import test_align_verify_2026_10 as A

    xy, flux = A._stars()
    ref = A._image(xy, flux, seed=1)
    fxy, m_true = A._rot(xy, 0.4, -4.0, -5.5)
    frame = A._image(fxy, flux, seed=2)
    a, t = m_true[:2, :2], m_true[:2, 2]
    crpix_f = np.linalg.solve(a, np.array([A.W / 2, A.H / 2]) - t)
    hdr = A._wcs_hdr(A._CD_REF @ a, (crpix_f[0], crpix_f[1]))
    p_wcs = A._write_wcs(tmp_path, "wcs.fits", frame, hdr)
    p_star = A._write(tmp_path, "nowcs.fits", frame)
    return A, ref, frame, p_wcs, p_star


def test_g02_25_wcs_path_uses_the_one_warp_cubic_median_fill(tmp_path, monkeypatch):
    import vyvar_alignment_frame as V
    from skimage.transform import AffineTransform

    A, ref, frame, p_wcs, p_star = _align_field(tmp_path)
    r_w = V._alignment_compute_one_frame(p_wcs, 1, A._ctx_wcs(ref, tmp_path), [])
    assert r_w["kind"] == "aligned" and r_w["aligned_method"] == "wcs_affine"
    m = np.asarray(r_w["registration_handoff"]["matrix"], dtype=np.float64).reshape(3, 3)
    expect = V.warp_to_reference(frame, ref.shape, AffineTransform(matrix=m).inverse).astype(np.float32)
    np.testing.assert_array_equal(r_w["aligned_data"], expect)  # exactly the one warp
    assert not np.isnan(r_w["aligned_data"]).any()  # median fill outside the footprint (reproject: NaN)

    monkeypatch.setattr(V, "_alignment_run_astroalign_points", lambda **k: (None, "forced", None))
    r_s = V._alignment_compute_one_frame(p_star, 1, A._ctx(ref, tmp_path), [])
    assert r_s["aligned_method"] == "star_similarity"
    w_, s_ = np.asarray(r_w["aligned_data"], dtype=np.float64), np.asarray(r_s["aligned_data"], dtype=np.float64)
    # same kernel -> same noise correlation: sky pixel std equal (bilinear reproject gave 0.66 / 0.81 of it)
    sky = (np.abs(ref - np.median(ref)) < 3 * 8.0)
    sky[:40, :] = sky[-40:, :] = False
    sky[:, :40] = sky[:, -40:] = False
    assert np.std(w_[sky]) / np.std(s_[sky]) == pytest.approx(1.0, abs=0.02)
    # and the same image: aperture sums of the brightest stars agree within 0.5 %
    import test_align_verify_2026_10 as Av

    xy, flux = Av._stars()
    for i in np.argsort(-flux)[:20]:
        x, y = int(round(xy[i, 0])), int(round(xy[i, 1]))
        aw = w_[y - 4:y + 5, x - 4:x + 5].sum() - 81 * np.median(w_[sky])
        as_ = s_[y - 4:y + 5, x - 4:x + 5].sum() - 81 * np.median(s_[sky])
        assert aw / as_ == pytest.approx(1.0, abs=5e-3)


def test_g02_25_reproject_is_not_used_by_alignment():
    import inspect

    import vyvar_alignment_frame as V

    src = inspect.getsource(V)
    assert "reproject_interp" not in src and not hasattr(V, "_roundtrip_needed")


# --------------------------------------------------------------------------- #
# C2 - photon-transfer tooling (gain_ptc) and the CAL-DIAG sky-pair gain diagnostic
# --------------------------------------------------------------------------- #
PTC_SHAPE = (200, 300)


def _ptc_set(tmp_path, *, g_native, rn_e, adc_bits, levels_e, dark_rate_e_s=0.0, ipc=0.0, seed=1):
    """Raw flat and dark pairs: vignetting + PRNU, Poisson light and dark current, Gaussian RN, optional IPC
    (charge-conserving 4-neighbour coupling), quantised to the ADC and written x 2^(16-adc_bits)."""
    from scipy.ndimage import convolve

    rng = np.random.default_rng(seed)
    scale = 2 ** (16 - adc_bits)
    yy, xx = np.mgrid[0:PTC_SHAPE[0], 0:PTC_SHAPE[1]]
    pattern = (1.0 - 0.2 * (((xx - 150) / 150) ** 2 + ((yy - 100) / 100) ** 2)) * (1 + 0.01 * rng.standard_normal(PTC_SHAPE))
    k = np.array([[0, ipc, 0], [ipc, 1 - 4 * ipc, ipc], [0, ipc, 0]])

    def frame(mean_e, t):
        e = rng.poisson(np.clip(mean_e * pattern, 0, None) + dark_rate_e_s * t).astype(np.float64)
        if ipc:
            e = convolve(e, k, mode="nearest")
        e = e + rng.normal(0, rn_e, PTC_SHAPE)
        return np.clip(np.round(e / g_native + 100.0), 0, 2 ** adc_bits - 1) * scale

    def w(path, data, t, kind):
        h = fits.Header()
        h["IMAGETYP"], h["EXPTIME"], h["GAIN"], h["XBINNING"] = kind, t, (0, "Gain"), 1
        fits.writeto(path, data.astype(np.uint16), h, overwrite=True)
        return path

    flats, darks = [], []
    for i, lev in enumerate(levels_e):
        t = 10.0 * (i + 1)
        flats += [w(tmp_path / f"f{i}_{j}.fits", frame(lev, t), t, "Flat Field") for j in range(2)]
        darks += [w(tmp_path / f"d{i}_{j}.fits", frame(0.0, t), t, "Dark Frame") for j in range(2)]
    return flats, darks


def test_c2_dark_pair_variance_subtracted_per_level(tmp_path):
    from gain_ptc import measure_from_files

    # dark current shot noise grows with the exposure like the signal: without the per-level dark pair the
    # slope is (1 + dark/signal) / g
    levels = [3e3, 6e3, 9e3, 12e3, 15e3, 18e3]
    flats, darks = _ptc_set(tmp_path, g_native=3.8, rn_e=8.0, adc_bits=14, levels_e=levels, dark_rate_e_s=40.0)
    r = measure_from_files(flats, darks, saturation_adu=65000.0)
    g_true = 3.8 / 4
    assert r.ok and r.through_origin and all(p.dark_var_half for p in r.points if p.used)
    assert abs(r.gain - g_true) < 3 * r.gain_err and r.gain == pytest.approx(g_true, rel=0.03)
    # the report: local slopes, linearity, covariance, JSON
    # small synthetic frames: the local slopes are noisy (the 3 % EMVA test needs full frames); report present
    assert np.isfinite(r.linearity["max_local_slope_dev"]) and r.linearity["linearity_error_max"] < 0.01
    assert r.covariance["summed"] is False and r.covariance["max_abs_rho"] < 0.01
    import json

    d = json.loads(r.to_json())
    assert d["gain"] == pytest.approx(r.gain) and len(d["points"]) == len(levels)


def test_c2_neighbour_covariance_is_summed_into_the_variance(tmp_path):
    from gain_ptc import measure_from_files

    levels = [4e3, 8e3, 12e3, 16e3, 20e3]
    flats, darks = _ptc_set(tmp_path, g_native=2.0, rn_e=3.0, adc_bits=16, levels_e=levels, ipc=0.03, seed=4)
    r = measure_from_files(flats, darks, saturation_adu=65000.0)
    assert r.covariance["summed"] and r.covariance["max_abs_rho"] > 0.01
    assert r.gain == pytest.approx(2.0, rel=0.04)  # without the covariances: var x (1 - 8 alpha) -> g + ~30 %
    assert any("covariances summed" in m for m in r.messages)


def test_c2_measured_point_is_stored_with_error_and_report(tmp_path):
    from camera_profile import load_profile
    from database import VyvarDatabase
    from gain_ptc import measure_from_files, native_gain

    flats, darks = _ptc_set(tmp_path, g_native=3.8, rn_e=8.0, adc_bits=14, levels_e=[3e3, 8e3, 14e3, 20e3])
    r = measure_from_files(flats, darks, saturation_adu=65000.0)
    db = VyvarDatabase(tmp_path / "v.sqlite3")
    cam = db.insert_equipment("QHY294MM", "cam", "CMOS", "x", 4.63)
    db.save_measured_curve_point(cam, readout_mode=None, gain_setting=0.0, binning=2, e_per_adu=native_gain(r.gain, 14),
                                 readnoise_e=r.read_noise_e, e_per_adu_err=native_gain(r.gain_err, 14),
                                 ptc_json=r.to_json())
    row = db.conn.execute("SELECT E_PER_ADU_ERR, PTC_JSON FROM EQUIPMENT_CURVE WHERE EQUIPMENT_ID = ?", (cam,)).fetchone()
    assert row[0] == pytest.approx(native_gain(r.gain_err, 14)) and '"points"' in row[1]
    db.conn.close()
    prof = load_profile(tmp_path / "v.sqlite3", cam)
    pt = next(p for p in prof.points if p.source == "measured")
    assert pt.binning == 2 and pt.e_per_adu_err == pytest.approx(native_gain(r.gain_err, 14))


def _sky_pairs(rng, *, g, rn_e, sky_lo, sky_hi, n_pairs=8, shape=(256, 384), adc_scale=4, dark=100.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    sky_adu = sky_lo + (sky_hi - sky_lo) * xx / shape[1]  # gradient: the tiles span the sky range
    out = []
    for _ in range(n_pairs):
        pair = []
        for _ in range(2):
            e = rng.poisson(sky_adu * g) + rng.normal(0, rn_e, shape)
            pair.append(np.round(e / g / adc_scale) * adc_scale + dark)  # 14-bit x4 quantisation
        out.append(tuple(pair))
    return out


def test_c2_sky_pair_gain_recovers_the_gain_and_flags_a_wrong_profile():
    from gain_ptc import sky_pair_gain

    g = 0.95
    pairs = _sky_pairs(np.random.default_rng(8), g=g, rn_e=10.0, sky_lo=1300.0, sky_hi=2400.0)
    ok = sky_pair_gain(pairs, dark_level=100.0, profile_gain_file=g, profile_read_noise_e=10.0, seed=42)
    assert ok["status"] == "ok" and ok["ratio_to_profile"] == pytest.approx(1.0, abs=0.05) and ok["gain_file_sigma"] > 0
    bad = sky_pair_gain(pairs, dark_level=100.0, profile_gain_file=0.7925, profile_read_noise_e=15.2, seed=42)
    assert bad["status"] == "warning" and bad["ratio_to_profile"] > 1.1  # 527: stored 3.17 / 4
    again = sky_pair_gain(pairs, dark_level=100.0, profile_gain_file=0.7925, profile_read_noise_e=15.2, seed=42)
    assert again == bad  # seeded bootstrap


def test_c2_sky_pair_gain_skipped_when_read_noise_dominates():
    from gain_ptc import sky_pair_gain

    # Brno 534 type: 28 ADU sky at 12.5 e-/ADU, average-binned (step 1)
    pairs = _sky_pairs(np.random.default_rng(9), g=12.5, rn_e=14.0, sky_lo=26.0, sky_hi=30.0, adc_scale=1, dark=512.0)
    r = sky_pair_gain(pairs, dark_level=512.0, profile_gain_file=12.48, profile_read_noise_e=14.08, seed=42)
    assert r["status"] == "skipped" and "does not dominate" in r["reason"]
    assert sky_pair_gain(pairs[:1], dark_level=512.0, profile_gain_file=12.48, profile_read_noise_e=14.08)["status"] == "skipped"


def test_c2_cal_diag_session_gets_the_sky_pair_diagnostic(tmp_path, caplog):
    from cal_diag import CalDiagGateResult, CalDiagSession
    from config import AppConfig
    from database import VyvarDatabase
    from pipeline_calibrate import _obs_group_key_from_light_path, add_sky_pair_gain_diagnostic

    rng = np.random.default_rng(10)
    pairs = _sky_pairs(rng, g=0.95, rn_e=10.0, sky_lo=1300.0, sky_hi=2400.0, n_pairs=8)
    lights = tmp_path / "lights" / "V_60_2"
    lights.mkdir(parents=True)
    files = []
    for i, (a, b) in enumerate(pairs):
        for j, d in enumerate((a, b)):
            h = fits.Header()
            h["IMAGETYP"], h["EXPTIME"], h["FILTER"] = "Light Frame", 60.0, "V"
            h["XBINNING"] = h["YBINNING"] = 2
            h["GAIN"] = (0.0, "Gain")
            h["DATE-OBS"] = (T0 + timedelta(minutes=2 * i + j)).strftime("%Y-%m-%dT%H:%M:%S")
            p = lights / f"L_{i}_{j}.fits"
            fits.writeto(p, d.astype(np.uint16), h)
            files.append(p)
    db = VyvarDatabase(tmp_path / "v.sqlite3")
    db.conn.execute("DELETE FROM EQUIPMENTS")
    db.conn.execute("INSERT INTO EQUIPMENTS (ID, CAMERANAME, GAIN_ADU, READNOISE_E, ADC_BITS, MODEL_KEY, READOUT_MODE, "
                    "BIN_MODE, DEFAULT_GAIN_SETTING) VALUES (1, 'QHY294MM', 3.17, 7.6, 14, 'qhy294m_pro', '11mp', "
                    "'sum', 0)")
    db.conn.commit()
    db.conn.close()
    cfg = AppConfig()
    cfg.database_path = str(tmp_path / "v.sqlite3")
    og = _obs_group_key_from_light_path(files[0])
    s = CalDiagSession()
    s.gate_results["k"] = CalDiagGateResult(og, "d.fits", 2, "PASS", "SUM", block_factor=2, m_D_sum=100.0, m_D_mean=25.0)
    with caplog.at_level("WARNING"):
        add_sky_pair_gain_diagnostic(s, sorted(files), cfg=cfg, equipment_id=1)
    spg = s.gate_results["k"].sky_pair_gain
    assert spg["status"] == "warning" and spg["profile_gain_file"] == pytest.approx(0.7925)
    assert spg["ratio_to_profile"] == pytest.approx(0.95 / 0.7925, rel=0.08) and spg["n_pairs"] == 8
    assert "C2 sky-pair gain" in caplog.text
    assert "sky_pair_gain" in s.json_export()["keys"]["k"]
