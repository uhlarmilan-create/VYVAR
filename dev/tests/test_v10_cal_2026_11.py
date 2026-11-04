"""v10 closing items, tag cal (Milan approved 2026-11-04).

Item 21: the gain of a calibration library master is never truncated (one implementation, calib_library).
Item 14: the WCS alignment method label is ``wcs_affine`` (old ``wcs_reproject`` read as an alias).
Item 18: the PTC covariance check's standard error of the level-averaged correlation.
PTC folder tool: dev/tools/ptc_from_folder.py recovers a known gain from synthetic raw pairs.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import calib_library as cl
from database import VyvarDatabase

NOW = datetime(2026, 10, 1, 22, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# item 21 - exact gain in the calibration library
# --------------------------------------------------------------------------- #
def _cam_db(tmp_path: Path) -> tuple[VyvarDatabase, int]:
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    cam = db.insert_equipment("Cam", "c", "CMOS", "x", 3.76)
    return db, cam


def _dark(path: Path, gain: float | int) -> Path:
    h = fits.Header()
    h["IMAGETYP"] = "Dark Frame"
    h["XBINNING"] = 1
    h["EXPTIME"] = 60.0
    h["CCD-TEMP"] = -10.0
    h["GAIN"] = gain
    h["DATE-OBS"] = NOW.strftime("%Y-%m-%dT%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.zeros((4, 4), dtype=np.float32), header=h, overwrite=True)
    return path


def _find(db: VyvarDatabase, cam: int, gain: float) -> str | None:
    r = cl.find_dark(db, camera_id=cam, gain=gain, offset=None, exposure=60.0, binning=1, temp=-10.0,
                     temp_tol=0.5, light_dates=[NOW], validity_days=90)
    return r.master.path if r.master is not None else None


def test_item21_truncating_library_copy_is_gone() -> None:
    """The second library implementation in database.py stored int(gain) (0.79 e-/ADU -> 0)."""
    for name in ("register_calibration_library_entry", "find_best_calibration_library_path"):
        assert not hasattr(VyvarDatabase, name), name


def test_item21_e_per_adu_gain_round_trips(tmp_path: Path) -> None:
    db, cam = _cam_db(tmp_path)
    p = _dark(tmp_path / "lib" / "dark_g079.fits", 0.79)
    m = cl.register_master(db, p, kind="dark", camera_id=cam, telescope_id=None)
    assert m.gain == 0.79 and m.gain_exact
    row = db.conn.execute("SELECT GAIN, GAIN_EXACT FROM CALIBRATION_LIBRARY").fetchone()
    assert row["GAIN"] == 0.79 and row["GAIN_EXACT"] == 1
    assert [x.gain for x in cl.list_masters(db, camera_id=cam)] == [0.79]
    assert Path(_find(db, cam, 0.79)).resolve() == p.resolve()
    assert _find(db, cam, 0.0) is None and _find(db, cam, 1.0) is None


def test_item21_integer_setting_matches_and_fractional_setting_is_kept(tmp_path: Path) -> None:
    db, cam = _cam_db(tmp_path)
    p12 = _dark(tmp_path / "lib" / "dark_g12.fits", 12)
    cl.register_master(db, p12, kind="dark", camera_id=cam, telescope_id=None)
    assert Path(_find(db, cam, 12)).resolve() == p12.resolve()
    assert Path(_find(db, cam, 12.0)).resolve() == p12.resolve()
    p125 = _dark(tmp_path / "lib" / "dark_g125.fits", 12.5)
    cl.register_master(db, p125, kind="dark", camera_id=cam, telescope_id=None)
    assert Path(_find(db, cam, 12.5)).resolve() == p125.resolve()
    assert Path(_find(db, cam, 12)).resolve() == p12.resolve()  # 12.5 is not truncated onto 12
    with pytest.raises(ValueError):
        cl.gain_key("twelve")  # a non-numeric card fails loud


def test_item21_inexact_row_is_recovered_when_its_master_returns(tmp_path: Path) -> None:
    """A GAIN_EXACT = 0 row (G01-10 migration found no file) is retried on the next open."""
    db, cam = _cam_db(tmp_path)
    p = _dark(tmp_path / "lib" / "dark_g079.fits", 0.79)
    cl.register_master(db, p, kind="dark", camera_id=cam, telescope_id=None)
    db.conn.execute("UPDATE CALIBRATION_LIBRARY SET GAIN = 1, GAIN_EXACT = 0")  # as the old code left it
    db.conn.commit()
    db.conn.close()
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    row = db.conn.execute("SELECT GAIN, GAIN_EXACT FROM CALIBRATION_LIBRARY").fetchone()
    assert row["GAIN"] == 0.79 and row["GAIN_EXACT"] == 1


def test_item21_inexact_row_read_side(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    db, cam = _cam_db(tmp_path)
    p = _dark(tmp_path / "lib" / "dark_g079.fits", 0.79)
    cl.register_master(db, p, kind="dark", camera_id=cam, telescope_id=None)
    db.conn.execute("UPDATE CALIBRATION_LIBRARY SET GAIN = 1, GAIN_EXACT = 0")
    db.conn.commit()
    # file readable in this session (drive mounted after the open): the exact card is used, not 1
    m = cl.list_masters(db, camera_id=cam)[0]
    assert m.gain == 0.79 and m.gain_exact
    assert Path(_find(db, cam, 0.79)).resolve() == p.resolve() and _find(db, cam, 1.0) is None
    # file gone: rounded value, flagged, WARNING (never matched: the matchers need the file)
    p.unlink()
    cl._WARNED_INEXACT_GAIN.discard(str(p.resolve()))
    with caplog.at_level(logging.WARNING, logger="calib_library"):
        m = cl.list_masters(db, camera_id=cam)[0]
    assert m.gain == 1.0 and not m.gain_exact
    assert any("rounded gain" in r.getMessage() for r in caplog.records)
    assert _find(db, cam, 1.0) is None


# --------------------------------------------------------------------------- #
# item 14 - WCS alignment label wcs_affine; old wcs_reproject outputs still read
# --------------------------------------------------------------------------- #
def test_item14_label_and_legacy_alias() -> None:
    import vyvar_alignment_frame as vaf

    assert vaf.ALIGN_METHOD_WCS_AFFINE == "wcs_affine"
    assert vaf.canonical_align_method("wcs_reproject") == "wcs_affine"
    for m in ("astroalign", "star_similarity", "star_polynomial", "wcs_affine"):
        assert vaf.canonical_align_method(m) == m
    src = Path(vaf.__file__).read_text(encoding="utf-8")
    assert src.count('"wcs_reproject"') == 1  # only the read-side alias


def test_item14_old_handoff_label_is_read_as_wcs_affine() -> None:
    from osc_align import apply_registration_handoff_to_frame

    rng = np.random.default_rng(1)
    frame = rng.normal(100.0, 1.0, (32, 32)).astype(np.float32)
    out, hdr, method = apply_registration_handoff_to_frame(
        frame_data=frame, frame_hdr=fits.Header(), ref_data=frame, ref_hdr=fits.Header(),
        handoff_entry={"method": "wcs_reproject", "matrix": list(np.eye(3).ravel())},
    )
    assert out is not None
    assert method == "osc_handoff_wcs_affine" and hdr["VYALGM"] == "osc_handoff_wcs_affine"


# --------------------------------------------------------------------------- #
# item 18 - PTC covariance check: SE of the level-averaged correlation
# --------------------------------------------------------------------------- #
def _cov_result(n_px: list[int], rho: float):
    from gain_ptc import PairPoint, PtcResult, _covariance_check

    res = PtcResult(ok=False)
    res.points = [PairPoint(("a", "b"), 1.0, 1000.0 * (i + 1), 500.0 * (i + 1), var_half_raw=500.0 * (i + 1),
                            rho_x=rho, rho_y=0.0, n_px=n) for i, n in enumerate(n_px)]
    _covariance_check(res)
    return res


def test_item18_se_equal_levels_is_one_over_sqrt_total() -> None:
    res = _cov_result([10_000] * 4, 0.02)
    assert res.covariance["rho_stderr"] == pytest.approx(1.0 / np.sqrt(40_000), rel=1e-12)
    assert res.covariance["summed"]  # 2 % > 1 % and > 3 x 0.5 %


def test_item18_se_of_unweighted_mean_with_unequal_levels() -> None:
    """The level mean weights every level 1 / L: a small noisy level dominates the SE (old 1 / sqrt(sum n)
    gave 0.001 here and summed the covariances on a 2 % correlation that is not significant)."""
    res = _cov_result([100, 1_000_000], 0.02)
    assert res.covariance["rho_stderr"] == pytest.approx(np.sqrt(1 / 100 + 1 / 1_000_000) / 2, rel=1e-12)
    assert not res.covariance["summed"]


def test_item18_null_se_matches_the_estimator() -> None:
    """Uncorrelated, x4-quantised noise: the spread of the lag-1 correlation is 1 / sqrt(n)."""
    from gain_ptc import _neighbour_cov

    rng = np.random.default_rng(7)
    rs = []
    for _ in range(300):
        d = np.round(rng.normal(0.0, 30.0, (60, 60)) / 4.0) * 4.0
        v, cx, _ = _neighbour_cov(d)
        rs.append(cx / v)
    assert np.std(rs) == pytest.approx(1.0 / 60.0, rel=0.15)


# --------------------------------------------------------------------------- #
# PTC from a folder (dev/tools/ptc_from_folder.py)
# --------------------------------------------------------------------------- #
def _ptc_tool():
    import importlib.util

    p = Path(__file__).resolve().parents[1] / "tools" / "ptc_from_folder.py"
    spec = importlib.util.spec_from_file_location("ptc_from_folder", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ptc_folder(root: Path, *, g_native: float, rn_e: float, levels_e: list[float], seed: int = 3) -> Path:
    """Raw pairs of known gain: Poisson electrons, Gaussian read noise, 14-bit ADC written x4 (16-bit files);
    two darks per flat exposure; plus a master flat and a bias that must be ignored."""
    rng = np.random.default_rng(seed)
    shape = (120, 160)
    prnu = 1 + 0.01 * rng.standard_normal(shape)

    def frame(mean_e: float) -> np.ndarray:
        e = rng.poisson(mean_e * prnu).astype(np.float64) + rng.normal(0.0, rn_e, shape)
        return np.clip(np.round(e / g_native + 200.0), 0, 2 ** 14 - 1) * 4

    def w(path: Path, data: np.ndarray, t: float, kind: str, **extra) -> None:
        h = fits.Header()
        h["IMAGETYP"], h["EXPTIME"], h["GAIN"], h["XBINNING"] = kind, t, 0, 2
        for k, v in extra.items():
            h[k] = v
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.writeto(path, data.astype(np.uint16), h, overwrite=True)

    for i, lev in enumerate(levels_e):
        t = 1.0 + i
        for j in range(2):
            w(root / "flats" / f"FLAT_{t:g}s_{j}.fits", frame(lev), t, "Flat Field")
            w(root / "darks" / f"DARK_{t:g}s_{j}.fits", frame(0.0), t, "Dark Frame")
    w(root / "MasterFlat.fits", frame(levels_e[0]), 1.0, "Flat Field", NCOMBINE=20)
    w(root / "bias_0.fits", frame(0.0), 0.0, "Bias Frame")
    return root


def test_ptc_tool_recovers_a_known_gain(tmp_path: Path) -> None:
    tool = _ptc_tool()
    folder = _ptc_folder(tmp_path / "FLAT56", g_native=2.0, rn_e=6.0,
                         levels_e=[3e3, 6e3, 9e3, 12e3, 15e3, 18e3])  # < 70 % of saturation
    out_json = tmp_path / "ptc.json"
    rc = tool.main([str(folder), "--saturation-adu", "65532", "--adc-bits", "14", "--json", str(out_json)])
    assert rc == 0
    import json

    rep = json.loads(out_json.read_text(encoding="ascii"))
    s = rep["summary"]
    assert rep["inventory"]["n_flats"] == 12 and rep["inventory"]["n_darks"] == 12  # master + bias ignored
    assert s["ok"] and s["through_origin"] and s["n_levels"] == 6 and s["binning"] == 2
    assert abs(s["gain_native"] - 2.0) < 3 * s["gain_native_err"]
    assert s["gain_native"] == pytest.approx(2.0, rel=0.05)
    assert s["gain_file"] == pytest.approx(0.5, rel=0.05)  # x4 container
    assert s["read_noise_e"] == pytest.approx(6.0, rel=0.1)


def test_ptc_tool_reads_saturation_and_adc_from_the_database(tmp_path: Path) -> None:
    tool = _ptc_tool()
    folder = _ptc_folder(tmp_path / "ptc", g_native=2.0, rn_e=6.0, levels_e=[5e3, 10e3, 15e3, 20e3])
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    cam = db.insert_equipment("Cam", "c", "CMOS", "x", 3.76)
    db.conn.execute("UPDATE EQUIPMENTS SET SATURATE_ADU = 65532, ADC_BITS = 14 WHERE ID = ?", (cam,))
    db.conn.commit()
    db.conn.close()
    assert tool._camera_values(tmp_path / "vyvar.sqlite3", cam) == (65532.0, 14)
    rc = tool.main([str(folder), "--db", str(tmp_path / "vyvar.sqlite3"), "--camera-id", str(cam)])
    assert rc == 0


def test_ptc_tool_inventory_flags_a_master_flat_set(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """D:\\FLAT56 may hold only flats of one exposure (for a master flat): not a PTC ladder."""
    tool = _ptc_tool()
    folder = _ptc_folder(tmp_path / "one", g_native=2.0, rn_e=6.0, levels_e=[10e3])
    assert tool.main([str(folder), "--inventory"]) == 0
    flats, darks = __import__("gain_ptc").ptc_frames_from_folder(folder)
    inv = tool.inventory(flats, darks)
    assert inv["flat_levels_with_pair"] == 1
    assert any("not a PTC ladder" in p for p in inv["problems"])
