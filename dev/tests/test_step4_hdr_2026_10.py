"""Step 4 T08 "one authority per header fact" (code review 2): exposure start time, filter token,
optics (focal length / pixel / sensor size / plate scale), observation-group key, VY_FWHM reader.
"""
from __future__ import annotations

import logging

import astropy.units as u
import pytest
from astropy.io import fits
from astropy.time import Time

# --------------------------------------------------------------------------------------------
# Exposure start time: G02-01, G02-16, G01-05, G02-49
# --------------------------------------------------------------------------------------------
_TRUE_START = Time("2026-10-01T21:15:03.250", format="isot", scale="utc")


@pytest.mark.parametrize(
    "date_obs, extra",
    [
        ("2026-10-01T21:15:03.25", {}),
        ("2026-10-01 21:15:03.25", {}),  # space separator (G02-01)
        ("2026-10-01T21:15:03.25Z", {}),
        ("2026-10-01T21:15:03.25+00:00", {}),  # offset suffix (G02-01)
        ("2026-10-01T23:15:03.25+02:00", {}),  # numeric offset is removed
        ("2026-10-01", {"TIME-OBS": "21:15:03.25"}),  # date + TIME-OBS
    ],
)
def test_exposure_start_and_mid_from_date_obs_forms(date_obs: str, extra: dict) -> None:
    from time_utils import exposure_start_utc, mid_exposure_jd

    h = fits.Header()
    h["DATE-OBS"] = date_obs
    h["EXPTIME"] = 120.0
    for k, v in extra.items():
        h[k] = v
    t = exposure_start_utc(h)
    assert t is not None
    assert abs((t - _TRUE_START).to_value(u.s)) < 1e-3
    jd_mid = mid_exposure_jd(h)
    assert abs((Time(jd_mid, format="jd", scale="utc") - _TRUE_START).to_value(u.s) - 60.0) < 1e-3


def test_unparsable_date_time_is_logged_not_midnight(caplog: pytest.LogCaptureFixture) -> None:
    from except_fix_counters import get_except_fix_counters, reset_except_fix_counters
    from time_utils import exposure_start_utc, mid_exposure_jd

    h = fits.Header()
    h["DATE-OBS"] = "2026-10-01T25:99:03"
    h["EXPTIME"] = 60.0
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    assert exposure_start_utc(h) is None
    assert mid_exposure_jd(h) is None
    assert get_except_fix_counters().date_obs_parse_fail >= 1
    assert any("DATE-OBS" in r.message for r in caplog.records)


def test_date_only_falls_back_to_mjd_then_logged_midnight(caplog: pytest.LogCaptureFixture) -> None:
    from except_fix_counters import get_except_fix_counters, reset_except_fix_counters
    from time_utils import exposure_start_utc

    h = fits.Header()
    h["DATE-OBS"] = "2026-10-01"
    h["MJD-OBS"] = float(_TRUE_START.mjd)
    assert abs((exposure_start_utc(h) - _TRUE_START).to_value(u.s)) < 1e-3

    h2 = fits.Header()
    h2["DATE-OBS"] = "2026-10-01"
    reset_except_fix_counters()
    caplog.set_level(logging.ERROR)
    t = exposure_start_utc(h2)
    assert t is not None and t.isot.startswith("2026-10-01T00:00:00")
    assert get_except_fix_counters().timeobs_parse_fallback == 1
    assert any("midnight" in r.message for r in caplog.records)


def test_timesys_tt_is_converted_to_utc() -> None:
    """G02-49: TIMESYS='TT' -> the UTC instant is 69.184 s earlier (2026: TAI-UTC = 37 s)."""
    from time_utils import exposure_start_utc, mid_exposure_jd

    h = fits.Header()
    h["DATE-OBS"] = "2026-10-01T21:15:03.25"
    h["EXPTIME"] = 60.0
    jd_utc = mid_exposure_jd(h)
    h["TIMESYS"] = "TT"
    jd_tt = mid_exposure_jd(h)
    assert abs((jd_utc - jd_tt) * 86400.0 - 69.184) < 1e-3
    h["TIMESYS"] = "TAI"
    assert abs((_TRUE_START - exposure_start_utc(h)).to_value(u.s) - 37.0) < 1e-3
    h["TIMESYS"] = "LOCAL"
    assert exposure_start_utc(h) is None


@pytest.mark.parametrize(
    "hdr",
    [
        {"DATE-OBS": "2026-10-01 21:15:03.25", "EXPTIME": 30.0},
        {"DATE-OBS": "2026-10-01T21:15:03.25+00:00", "EXPTIME": 120.0},
        {"DATE-OBS": "2026-10-01T21:15:03.25", "JD": 2461315.0, "EXPTIME": 60.0},
        {"MJD-OBS": 61314.88545, "EXPTIME": 10.0},
    ],
)
def test_jd_start_mid_and_inspection_jd_agree(hdr: dict) -> None:
    """G02-16 / G01-05: fits_meta jd_start, time_utils jd_mid and the QC INSPECTION_JD share one start."""
    from fits_meta import fits_metadata_from_primary_header
    from pipeline_calibrate import _inspection_jd_from_header
    from time_utils import mid_exposure_jd

    h = fits.Header()
    for k, v in hdr.items():
        h[k] = v
    jd_start = fits_metadata_from_primary_header(h)["jd_start"]
    assert jd_start > 2400000.0
    assert abs(_inspection_jd_from_header(h) - jd_start) < 1e-9
    assert abs((mid_exposure_jd(h) - jd_start) * 86400.0 - hdr["EXPTIME"] / 2.0) < 1e-3


def test_inspection_jd_keeps_time_of_day() -> None:
    """G01-05: DATE-OBS with T and no TIME-OBS / JD used to give midnight (2461100.5)."""
    from pipeline_calibrate import _inspection_jd_from_header

    h = fits.Header()
    h["DATE-OBS"] = "2026-03-01T21:33:12.5"
    ref = Time("2026-03-01T21:33:12.5", format="isot", scale="utc").jd
    assert abs(_inspection_jd_from_header(h) - ref) < 1e-8


# --------------------------------------------------------------------------------------------
# Filter token: G05-13, G05-14, G08-13 (one parser, one filter -> band table)
# --------------------------------------------------------------------------------------------
_FILTER_NAMES = ["g", "r", "i", "V", "B", "R", "Rc", "SG", "sr", "gp", "r'", "Sloan_r", "Sloan_g",
                 "Johnson_V", "Johnson_B", "Cousins_R", "Bessell_V", "NoFilter", "Clear", "L", "IR_CUT",
                 "UV_IR_Cut", "Baader_L", "TG", "foo"]
_EXPOSURES = ["60", "120", "2.5", "0.5"]


@pytest.mark.parametrize("flt", _FILTER_NAMES)
@pytest.mark.parametrize("exp", _EXPOSURES)
def test_obs_group_forms_classify_like_fits_filter(flt: str, exp: str) -> None:
    """G05-14: class(f|e|b) == class(folder(f|e|b)) == class(FILTER=f), int and fractional e."""
    from band_classify import classify_photometric_band, obs_group_first_token
    from fits_meta import observation_group_folder_name

    pipe = f"{flt}|{exp}|2"
    folder = observation_group_folder_name(pipe)
    assert obs_group_first_token(pipe) == flt
    assert obs_group_first_token(folder) == observation_group_folder_name(flt).strip("_")
    ref = classify_photometric_band("", fits_filter=flt)
    assert classify_photometric_band(pipe) is ref
    assert classify_photometric_band(folder) is ref


@pytest.mark.parametrize(
    "name, expected",
    [
        ("IR_CUT", "LUMINANCE"), ("IRCUT", "LUMINANCE"), ("UV/IR Cut", "LUMINANCE"),
        ("UV_IR_Cut", "LUMINANCE"), ("Baader_L", "LUMINANCE"), ("Astronomik L2", "LUMINANCE"),
        ("ZWO IR-cut", "LUMINANCE"), ("Optolong L-Pro", "LUMINANCE"), ("Lum", "LUMINANCE"),
        ("Clear", "CLEAR_UNFILTERED"), ("NoFilter", "CLEAR_UNFILTERED"),
        ("RGB_Red", "UNKNOWN"), ("Gamma", "UNKNOWN"), ("Bin", "UNKNOWN"), ("ZWO_R", "UNKNOWN"),
        ("Ha", "UNKNOWN"), ("Sloan_r", "STANDARD_FILTER"), ("Bessell_V", "STANDARD_FILTER"),
        ("V", "STANDARD_FILTER"), ("g", "STANDARD_FILTER"),
    ],
)
def test_vendor_filter_names_classify_without_prefix_heuristic(name: str, expected: str) -> None:
    """G05-13: an IR-cut / L filter is not a standard filter (no colour term); unknown -> UNKNOWN."""
    from band_classify import classify_photometric_band, resolve_apply_color_term

    assert classify_photometric_band("", fits_filter=name).name == expected
    folder = f"{name.replace(' ', '_').replace('/', '_')}_60_2"
    assert resolve_apply_color_term(None, folder) is (expected == "STANDARD_FILTER")


def test_ir_cut_obs_group_no_colour_term() -> None:
    from band_classify import resolve_apply_color_term

    assert resolve_apply_color_term(None, "IR_CUT_60_2") is False
    assert resolve_apply_color_term(None, "IR_CUT|60|2") is False


_ALIASES = ["g", "gp", "g'", "sg", "sloan_g", "sloang", "r", "rp", "r'", "sr", "sloan_r", "sloanr",
            "i", "ip", "i'", "si", "sloan_i", "sloani", "v", "johnson_v", "b", "johnson_b", "rc",
            "cousins_r", "V", "B", "R", "Rc", "SG", "SR", "Johnson_V", "Sloan_g", "NoFilter", "TG", "L"]


@pytest.mark.parametrize("alias", _ALIASES)
def test_filter_band_and_k2_token_agree(alias: str) -> None:
    """G08-13: FilterBand, k2 token and band class agree for every alias, folder and pipe form."""
    import math

    from band_classify import PhotometricBand, classify_photometric_band
    from gaia_bands import filter_band, filter_band_for_obs_group
    from k2_extinction import filter_token_from_obs_group, resolve_k2_bprp_value

    ref = filter_band(alias)
    for og in (f"{alias}_60_1", f"{alias}|60|1"):
        fb = filter_band_for_obs_group(og)
        assert (fb.zp_band, fb.aavso_code, fb.transformable) == (ref.zp_band, ref.aavso_code, ref.transformable)
        assert filter_token_from_obs_group(og) == alias
        k2, _src = resolve_k2_bprp_value(None, og)
        if ref.transformable:
            # a reportable standard band always has a k'' (Sloan Smith 2002, B Henden, V/R zero)
            assert classify_photometric_band(og) is PhotometricBand.STANDARD_FILTER
            assert math.isfinite(k2)
        else:
            assert not math.isfinite(k2)


def test_underscored_filter_names_keep_their_band() -> None:
    """G08-13: Sloan_g_60_1 / Johnson_V_60_1 used to fall to Gaia G (not reportable)."""
    from gaia_bands import filter_band_for_obs_group
    from k2_extinction import computed_k2_bprp_for_token

    assert filter_band_for_obs_group("Sloan_g_60_1").aavso_code == "SG"
    assert filter_band_for_obs_group("Johnson_V_60_1").aavso_code == "V"
    assert filter_band_for_obs_group("Sloan_r|2.5|1").aavso_code == "SR"
    assert computed_k2_bprp_for_token("Sloan_g") == pytest.approx(computed_k2_bprp_for_token("g"))


# --------------------------------------------------------------------------------------------
# Optics module: G13-06, G13-07, G03-19, G13-17, G02-41, G08-29, G09-24, G04-30, G01-41
# --------------------------------------------------------------------------------------------
class _FakeRigDb:
    def __init__(self, focal: float | None, pixel: float | None) -> None:
        self.focal, self.pixel = focal, pixel

    def get_telescope_focal_mm(self, telescope_id):  # noqa: ANN001
        return self.focal

    def get_equipment_pixel_size_um(self, equipment_id):  # noqa: ANN001
        return self.pixel


@pytest.mark.parametrize("focal", [14.0, 20.0, 24.0, 200.0, 1750.0, 2000.0, 2032.0, 2250.0, 5480.0])
def test_plate_scale_never_rewrites_focal(focal: float) -> None:
    """G13-06: 1750-2250 mm used to be divided by 10; G02-41: 14-24 mm lenses stay mm."""
    from optics import optics_hint, plate_scale_arcsec_per_pixel

    assert plate_scale_arcsec_per_pixel(pixel_pitch_um=3.76, focal_length_mm=focal) == pytest.approx(
        3.76 / focal * 206.265
    )
    h = fits.Header()
    h["FOCALLEN"] = focal
    h["XBINNING"] = 1
    assert optics_hint(h).focal_mm == pytest.approx(focal)
    hint = optics_hint(None, _FakeRigDb(focal, 3.76), equipment_id=1, telescope_id=1, binning=(2, 2))
    assert hint.focal_mm == pytest.approx(focal)
    assert hint.arcsec_per_px == pytest.approx(7.52 / focal * 206.265)


@pytest.mark.parametrize(
    "value, comment, expected",
    [(20.0, "", 20.0), (0.2, "", 200.0), (0.4, "", 400.0), (200.0, "", 200.0), (2.0, "[m] focal length", 2000.0),
     (2000.0, "Focal length in mm", 2000.0), (0.0, "", None), ("abc", "", None)],
)
def test_focal_unit_rule(value, comment: str, expected) -> None:
    """G02-41: metres only below 0.5 or when the card comment says metres."""
    from optics import focal_length_mm_from_value, header_focal_length_mm

    got = focal_length_mm_from_value(value, comment)
    assert got == (pytest.approx(expected) if expected is not None else None)
    h = fits.Header()
    h["FOCALLEN"] = (value, comment)
    assert header_focal_length_mm(h) == got


@pytest.mark.parametrize("text", ["6280*4176", "6280x4176", "6280 X 4176", "6280 * 4176"])
def test_sensor_size_parser_accepts_db_format(text: str) -> None:
    """G13-07: the DB writes W*H; one parser for database binning inference and autodetect."""
    from optics import parse_sensor_size
    from optics_autodetect import detect_equipment
    from utils import parse_fits_binning_int

    assert parse_sensor_size(text) == (6280, 4176)
    assert parse_fits_binning_int("2*2") == 2
    h = fits.Header()
    h["NAXIS1"] = 3140
    h["NAXIS2"] = 2088
    h["XBINNING"] = 2
    h["YBINNING"] = 2
    det = detect_equipment(h, [{"ID": 7, "CAMERANAME": "X", "ALIAS": "x", "SENSORSIZE": text}])
    assert det.matched_id == 7 and any("sensor" in r for r in det.reasons)


def test_header_pixel_convention_xpixsz_is_binned() -> None:
    """XPIXSZ = binned pitch (MaxIm / QHY comment), PIXSIZE1 = native: both give the same native pitch."""
    from fits_meta import fits_metadata_from_primary_header
    from optics import header_native_pixel_um

    a = fits.Header()
    a["XBINNING"] = 2
    a["XPIXSZ"] = (9.26, "X binned pixel size in microns")
    a["YPIXSZ"] = 9.26
    b = fits.Header()
    b["XBINNING"] = 2
    b["PIXSIZE1"] = 4.63
    b["PIXSIZE2"] = 4.63
    c = a.copy()
    c["PIXSIZE1"] = 4.63
    c["PIXSIZE2"] = 4.63
    for h in (a, b, c):
        assert header_native_pixel_um(h) == (pytest.approx(4.63), pytest.approx(4.63))
        meta = fits_metadata_from_primary_header(h)
        assert meta["pixel_size_um_physical"] == pytest.approx(4.63)
        assert meta["effective_pixel_um_plate_scale"] == pytest.approx(9.26)


@pytest.mark.parametrize("osc_bin", [1, 2])
@pytest.mark.parametrize("src_bin", [1, 2])
def test_osc_channel_header_effective_pixel(osc_bin: int, src_bin: int) -> None:
    """G01-41: channel frame effective pixel = native x source binning x 2 x osc_bin (not twice)."""
    from fits_meta import fits_metadata_from_primary_header
    from osc_extract import build_channel_header

    src = fits.Header()
    src["XBINNING"] = src_bin
    src["XPIXSZ"] = 3.76 * src_bin
    src["DATE-OBS"] = "2026-10-01T21:15:03"
    ch = build_channel_header(src, channel="G", bayermask="RGGB", osc_bin=osc_bin, gain_eff=1.0, rn_eff=1.0,
                              src_name="x.fits")
    meta = fits_metadata_from_primary_header(ch)
    assert meta["pixel_size_um_physical"] == pytest.approx(3.76)
    assert meta["effective_pixel_um_plate_scale"] == pytest.approx(3.76 * src_bin * 2 * osc_bin)


def _rotated_wcs_header(rot_deg: float, scale_arcsec: float = 2.6) -> fits.Header:
    import math

    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = 1000
    h["NAXIS2"] = 800
    h["CTYPE1"] = "RA---TAN"
    h["CTYPE2"] = "DEC--TAN"
    h["CRPIX1"] = 500.5
    h["CRPIX2"] = 400.5
    h["CRVAL1"] = 200.0
    h["CRVAL2"] = 45.0
    s = scale_arcsec / 3600.0
    t = math.radians(rot_deg)
    h["CD1_1"] = -s * math.cos(t)
    h["CD1_2"] = s * math.sin(t)
    h["CD2_1"] = s * math.sin(t)
    h["CD2_2"] = s * math.cos(t)
    return h


@pytest.mark.parametrize("rot", [0.0, 37.0, 90.0, 180.0])
def test_one_plate_scale_from_wcs_any_rotation(rot: float, tmp_path) -> None:
    """G04-30 / G08-29 / G09-24: every reader gives 2.6 "/px, also for a 90 deg rotated WCS."""
    from astropy.wcs import WCS

    from dao_gaia_calibration import plate_scale_arcsec_per_px_from_wcs_nan
    from optics import frame_plate_scale, plate_scale_arcsec_per_px_from_wcs, plate_scale_from_fits

    h = _rotated_wcs_header(rot)
    h["VY_PLTS"] = 9.9  # disagrees: the WCS wins
    p = tmp_path / "MASTERSTAR.fits"
    fits.PrimaryHDU(header=h).writeto(p)
    assert plate_scale_arcsec_per_px_from_wcs(WCS(h)) == pytest.approx(2.6, rel=1e-9)
    assert plate_scale_arcsec_per_px_from_wcs_nan(WCS(h)) == pytest.approx(2.6, rel=1e-9)
    assert plate_scale_from_fits(p) == pytest.approx(2.6, rel=1e-9)
    assert plate_scale_from_fits(header=h) == pytest.approx(2.6, rel=1e-9)
    assert frame_plate_scale(p) == pytest.approx(2.6, rel=1e-9)


def test_wide_lens_scale_not_clamped() -> None:
    """G09-24: a solved 55 "/px (14 mm lens) is real - no 0.1-30 window."""
    from optics import plate_scale_from_header

    assert plate_scale_from_header(_rotated_wcs_header(0.0, 55.0)) == pytest.approx(55.0, rel=1e-9)


def test_optics_hint_db_first_header_fallback_and_parity() -> None:
    """G03-19 / G13-17: one rule (DB valid -> header) shared by every consumer."""
    from astrosolve import header_scale_hint
    from optics import optics_hint
    from pipeline_astrometry import compute_plate_scale_from_db

    h = fits.Header()
    h["XBINNING"] = 2
    h["FOCALLEN"] = 200.0
    h["XPIXSZ"] = 9.26
    db = _FakeRigDb(200.0, 4.63)
    hdr_only = optics_hint(h)
    with_db = optics_hint(h, db, equipment_id=1, telescope_id=1)
    assert hdr_only.arcsec_per_px == pytest.approx(with_db.arcsec_per_px)
    assert (with_db.focal_source, with_db.pixel_source) == ("database_telescope", "database_equipment")
    assert (hdr_only.focal_source, hdr_only.pixel_source) == ("fits_header", "fits_header")
    assert header_scale_hint(h) == pytest.approx(hdr_only.arcsec_per_px)
    assert compute_plate_scale_from_db(1, 1, db, binning=2) == pytest.approx(with_db.arcsec_per_px)
    # the DB wins over a disagreeing header; no telescope id -> no DB guess
    other = optics_hint(h, _FakeRigDb(1200.0, 3.76), equipment_id=1, telescope_id=1)
    assert other.focal_mm == 1200.0 and other.pixel_native_um == 3.76
    assert optics_hint(h, _FakeRigDb(1200.0, 3.76)).focal_mm == 200.0


def test_scale_mismatch_is_reported(monkeypatch) -> None:
    """G13-06: a ~10x DB focal error is reported against the solved scale, never rewritten."""
    import optics

    msgs: list[str] = []
    monkeypatch.setattr(optics, "log_event", msgs.append)
    assert optics.report_scale_mismatch(0.388, 3.88, context="T") == pytest.approx(10.0)
    assert msgs and "TELESCOPE.FOCAL" in msgs[0]
    msgs.clear()
    assert optics.report_scale_mismatch(3.88, 3.90, context="T") == pytest.approx(3.90 / 3.88)
    assert msgs == []


# --------------------------------------------------------------------------------------------
# Group key: G02-42 (one key builder, one filter normaliser)
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("flt", ["g", "Sloan r'", "", "unknown", "  V  ", None])
@pytest.mark.parametrize("exp", [60.0, 2.5, 120])
def test_one_group_key_from_header_metadata_and_rows(flt, exp) -> None:
    """Same key from header (fits_meta), metadata dict, importer and the frame-QC row fallback."""
    import importer
    from fits_meta import fits_metadata_from_primary_header, observation_group_key, observation_group_key_from_metadata
    from frame_qc import group_key_of

    h = fits.Header()
    if flt is not None:
        h["FILTER"] = flt
    h["EXPTIME"] = exp
    h["XBINNING"] = 2
    meta = fits_metadata_from_primary_header(h)
    key = observation_group_key_from_metadata(meta)
    assert key == observation_group_key(flt, exp, 2)
    assert importer.observation_group_key is observation_group_key
    assert group_key_of({"FILTER": flt, "EXPTIME": exp}) == key.rsplit("|", 1)[0] + "|?"
    assert group_key_of({"OBSERVATION_GROUP_KEY": key}) == key


# --------------------------------------------------------------------------------------------
# FWHM reader: G03-29, G06-35
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("vy", [2.03, 3.44, 29.9, 40.0, 0.3, "abc", None])
def test_vy_fwhm_treated_identically_by_all_readers(vy, tmp_path) -> None:
    """G03-29: one VY_FWHM reader and one valid range (aperture_policy.fwhm_from_header_vy_fwhm)."""
    from aperture_policy import fwhm_from_header_vy_fwhm
    from masterstar_context import masterstar_header_fwhm_px
    from platesolve_ui_paths import masterstar_fwhm_px
    from utils import dao_detection_fwhm_pixels

    h = fits.Header()
    if vy is not None:
        h["VY_FWHM"] = vy
    p = tmp_path / "MASTERSTAR.fits"
    fits.PrimaryHDU(header=h).writeto(p)
    ref = fwhm_from_header_vy_fwhm(h)
    assert masterstar_header_fwhm_px(p) == ref
    if ref is None:
        with pytest.raises(ValueError):
            masterstar_fwhm_px(p)  # was a silent 3.7 px
        assert dao_detection_fwhm_pixels(h, configured_fallback=None) is None
    else:
        assert masterstar_fwhm_px(p) == pytest.approx(round(ref, 3))
        assert dao_detection_fwhm_pixels(h, configured_fallback=None) == pytest.approx(max(1.2, min(20.0, ref)))


def test_no_constant_fwhm_defaults() -> None:
    """G03-29: no 4.5 px when the configured DAO fallback is unusable; G06-35: no MASTERSTAR Gaussian fit."""
    import aperture_photometry
    from utils import dao_detection_fwhm_pixels

    assert dao_detection_fwhm_pixels(fits.Header(), configured_fallback=float("nan")) is None
    assert dao_detection_fwhm_pixels(fits.Header(), configured_fallback=2.0) == 3.0
    assert not hasattr(aperture_photometry, "measure_fwhm_from_masterstar")


def test_masterstar_fwhm_without_vy_fwhm_uses_measurement_or_night_median(tmp_path) -> None:
    """G03-29: MASTERSTAR without VY_FWHM: measured as frame QC would, else night median, else error."""
    import numpy as np

    from pipeline_astrometry import build_masterstar_from_detrended

    src = tmp_path / "det" / "f1.fits"
    src.parent.mkdir()
    fits.PrimaryHDU(data=np.zeros((64, 64), dtype=np.float32)).writeto(src)  # flat: nothing measurable
    out = tmp_path / "MASTERSTAR.fits"
    with pytest.raises(ValueError, match="no FWHM"):
        build_masterstar_from_detrended(detrended_root=src.parent, output_fits=out, only_paths=[src],
                                        fwhm_fallback_px=None, pre_calibrated=True)
    build_masterstar_from_detrended(detrended_root=src.parent, output_fits=out, only_paths=[src],
                                    fwhm_fallback_px=3.2, pre_calibrated=True)
    assert fits.getheader(out)["VY_FWHM"] == pytest.approx(3.2)
