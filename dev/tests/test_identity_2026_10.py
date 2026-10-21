"""IDENTITY / CATALOG-FOOTPRINT / WCS-CONFIRM / PHASE1-ONE-PASS / FRAME-DEPTH (2026-10).

Synthetic sky (local Gaia SQLite + rendered image) - the same rules on any rig:
* identity radius is max(1 px, FWHM) in pixels;
* a detection takes the BRIGHTEST Gaia star in the radius, never a fainter neighbour that
  happens to be nearer, and never a star more than 2 mag fainter than its flux;
* a WCS handed over from another MASTERSTAR is verified (log-odds) and refined, a wrong one
  is rejected;
* the catalogue box covers the cone at any declination and across RA 0.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u

SCALE = 2.0  # arcsec / px
NX = NY = 400
FWHM = 3.0
ZP = 25.0  # flux = 10**(-0.4 (G - ZP))


def _wcs(ra0: float = 150.0, de0: float = 45.0, dx: float = 0.0, dy: float = 0.0) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [NX / 2 + 0.5 + dx, NY / 2 + 0.5 + dy]
    w.wcs.crval = [ra0, de0]
    w.wcs.cd = np.array([[-SCALE / 3600.0, 0.0], [0.0, SCALE / 3600.0]])
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _render(xy_flux: list[tuple[float, float, float]], seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:NY, 0:NX].astype(np.float64)
    img = np.full((NY, NX), 1000.0)
    s = FWHM / 2.3548
    for x, y, f in xy_flux:
        img += f / (2 * math.pi * s * s) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s))
    img += rng.normal(0.0, 5.0, img.shape)
    return img.astype(np.float32)


def _gaia_db(path: Path, rows: list[tuple[int, float, float, float]]) -> Path:
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE gaia_dr3 (source_id INTEGER PRIMARY KEY, ra REAL, dec REAL, g_mag REAL, bp_mag REAL,"
        " rp_mag REAL, bp_rp REAL, g_flux_error_rel REAL, parallax REAL, parallax_error REAL,"
        " parallax_over_error REAL, teff_gspphot REAL, logg_gspphot REAL, mh_gspphot REAL,"
        " distance_gspphot REAL, var_flag TEXT, non_single_star INTEGER)"
    )
    con.executemany(
        "INSERT INTO gaia_dr3 (source_id, ra, dec, g_mag, bp_rp, non_single_star) VALUES (?,?,?,?,0.8,0)", rows
    )
    con.commit()
    con.close()
    return path


def _field(tmp_path: Path):
    """60 field stars + the two identity traps; returns (image, true WCS, db, ids)."""
    rng = np.random.default_rng(7)
    w = _wcs()
    stars: list[tuple[int, float, float, float]] = []  # id, x, y, G (image truth)
    sid = 1000
    while len(stars) < 60:
        x, y = rng.uniform(20, NX - 20, 2)
        if math.hypot(x - 200, y - 200) < 25 or math.hypot(x - 100, y - 300) < 25:
            continue
        stars.append((sid, x, y, float(rng.uniform(10.5, 13.5))))
        sid += 1
    rows = []
    for s_id, x, y, g in stars:
        c = w.pixel_to_world(x, y)
        rows.append((s_id, float(c.ra.deg), float(c.dec.deg), g))
    # trap 1: bright A in the image at (200, 200); its catalogue position is 0.6 px off (WCS
    # error), a faint Gaia neighbour B (G + 4.5) sits right on the detection centroid.
    ca = w.pixel_to_world(200.6, 200.0)
    cb = w.pixel_to_world(200.05, 200.0)
    rows.append((1, float(ca.ra.deg), float(ca.dec.deg), 11.0))
    rows.append((2, float(cb.ra.deg), float(cb.dec.deg), 15.5))
    # trap 2: star C (G 12) is in the image but not in the catalogue; only a G 16 star D is there
    cd = w.pixel_to_world(100.0, 300.0)
    rows.append((4, float(cd.ra.deg), float(cd.dec.deg), 16.0))
    img = _render(
        [(x, y, 10 ** (-0.4 * (g - ZP))) for _, x, y, g in stars]
        + [(200.0, 200.0, 10 ** (-0.4 * (11.0 - ZP))), (200.05, 200.0, 10 ** (-0.4 * (15.5 - ZP))),
           (100.0, 300.0, 10 ** (-0.4 * (12.0 - ZP)))]
    )
    db = _gaia_db(tmp_path / "gaia.db", rows)
    return img, w, db


def test_identity_radius_is_pixel_based() -> None:
    from dao_gaia_calibration import catalog_identity_radius_arcsec, solve_rms_px_from_fits_header

    for scale in (0.56, 2.0, 9.8):
        used, inp = catalog_identity_radius_arcsec(fwhm_dao_px=2.4, plate_scale_arcsec_per_px=scale)
        assert inp["radius_px"] == pytest.approx(2.4)
        assert used == pytest.approx(2.4 * scale)
    used, inp = catalog_identity_radius_arcsec(fwhm_dao_px=0.6, plate_scale_arcsec_per_px=9.8)
    assert inp["radius_px"] == 1.0 and used == pytest.approx(9.8)
    h = fits.Header()
    assert solve_rms_px_from_fits_header(h) is None
    h["VY_SOLRM"] = 0.41
    assert solve_rms_px_from_fits_header(h) == pytest.approx(0.41)


def test_brightest_star_wins_and_flux_consistency(tmp_path: Path) -> None:
    from catalog_match import detect_stars_and_match_catalog

    img, w, db = _field(tmp_path)
    hdr = w.to_header()
    hdr["VY_SOLVR"] = "astrosolve"
    df, meta = detect_stars_and_match_catalog(
        img, hdr, gaia_db_path=db, vsx_df=pd.DataFrame(), dao_threshold_sigma=5.0, dao_fwhm_px=FWHM,
        dao_fwhm_bypass_header=True, max_catalog_rows=10000,
    )
    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")
    cid = df["catalog_id"].fillna("").astype(str).str.strip()
    a = df[(x - 200.0).abs().lt(1.0) & (y - 200.0).abs().lt(1.0)]
    assert len(a) == 1
    assert cid[a.index[0]] == "1", "bright A must keep its identity, not the nearer faint B"
    c = df[(x - 100.0).abs().lt(1.0) & (y - 300.0).abs().lt(1.0)]
    assert len(c) == 1
    assert cid[c.index[0]] == "", "a G 16 catalogue star cannot be a G 12 detection"
    # field stars: identified, 1:1
    ids = cid[cid.ne("")]
    assert not ids.duplicated().any()
    assert int(ids.astype(int).ge(1000).sum()) >= 55


def test_confirm_handed_over_wcs(tmp_path: Path) -> None:
    import astrosolve as A

    img, w, db = _field(tmp_path)
    ok = tmp_path / "ok.fits"
    fits.PrimaryHDU(img, header=_wcs(dx=3.0, dy=-2.0).to_header()).writeto(ok)
    r = A.confirm_fits(ok, gaia_db=str(db))
    assert r["solved"], r
    w2 = WCS(fits.getheader(ok))
    xs, ys = np.meshgrid(np.linspace(10, NX - 10, 5), np.linspace(10, NY - 10, 5))
    x2, y2 = w2.world_to_pixel(w.pixel_to_world(xs.ravel(), ys.ravel()))
    assert float(np.max(np.hypot(x2 - xs.ravel(), y2 - ys.ravel()))) < 0.2
    assert fits.getheader(ok)["VY_SOLVR"] == "astrosolve"
    assert fits.getheader(ok)["VY_SOLMD"] == "confirmed"

    bad = tmp_path / "bad.fits"
    fits.PrimaryHDU(img, header=_wcs(dx=150.0).to_header()).writeto(bad)
    r = A.confirm_fits(bad, gaia_db=str(db))
    assert not r["solved"]
    assert "VY_SOLVR" not in fits.getheader(bad)


@pytest.mark.parametrize("dec", [0.0, 45.0, 61.0, 80.0])
def test_cone_sky_box_contains_cone(dec: float) -> None:
    from pipeline_catalog import cone_sky_box

    c = SkyCoord(100.0 * u.deg, dec * u.deg)
    r = 4.7
    ra_min, ra_max, de_min, de_max = cone_sky_box(c, r)
    pa = np.linspace(0, 360, 721) * u.deg
    edge = c.directional_offset_by(pa, r * u.deg)
    ra = edge.ra.deg
    ra = np.where(ra < ra_min, ra + 360.0, ra)
    assert np.all(ra >= ra_min - 1e-6) and np.all(ra <= ra_max + 1e-6)
    assert np.all(edge.dec.deg >= de_min - 1e-6) and np.all(edge.dec.deg <= de_max + 1e-6)


def test_cone_sky_box_pole_and_ra_wrap() -> None:
    from catalog_db import ra_intervals_deg
    from pipeline_catalog import cone_sky_box

    assert cone_sky_box(SkyCoord(10 * u.deg, 88 * u.deg), 3.0)[:2] == (0.0, 360.0)
    ra_min, ra_max, _, _ = cone_sky_box(SkyCoord(1.0 * u.deg, 10 * u.deg), 2.0)
    assert ra_min < 0.0
    iv = ra_intervals_deg(ra_min, ra_max)
    assert len(iv) == 2 and iv[0][1] == 360.0 and iv[1][0] == 0.0


def test_comp_frame_stats_counts() -> None:
    from comp_pool_rms import COMP_SAT_LIMIT_SCALE, comp_frame_stats

    lim85 = 50000.0
    over = lim85 * COMP_SAT_LIMIT_SCALE + 1.0
    cache = {}
    paths = []
    for i in range(10):
        p = Path(f"/x/proc_{i:03d}.csv")
        paths.append(p)
        cache[str(p)] = pd.DataFrame(
            {
                "name": ["A", "B"],
                "dao_flux": [100.0, 100.0 if i < 8 else np.nan],
                "peak_max_adu": [over if i < 3 else 1000.0, 1000.0],
                "saturate_limit_adu_85pct": [lim85, lim85],
                "bjd_tdb_mid": [2460000.0 + i] * 2,
            }
        )
    st = comp_frame_stats({"A", "B"}, paths, cache)
    assert st["A"] == {"n_frames": 10, "n_peak": 10, "n_over": 3}
    assert st["B"] == {"n_frames": 8, "n_peak": 10, "n_over": 0}


def test_removed_paths_stay_removed() -> None:
    src = Path(__file__).resolve().parents[2] / "src_py"
    assert not (src / "astrometry_optimizer.py").exists()
    cm = (src / "catalog_match.py").read_text(encoding="utf-8")
    assert "_refine_wcs_tan_nn_gaia" not in cm and "catalog_kd_pack" not in cm and ">= 120" not in cm
    pc = (src / "pipeline_catalog.py").read_text(encoding="utf-8")
    assert "_catalog_df_cap_brightest_by_mag" not in pc and "optics_floor" not in pc
    pce = (src / "pipeline_catalog.py").read_text(encoding="utf-8")
    assert "detect_stars_and_match_catalog(" not in pce, "per-frame export is MASTERSTAR-locked only"


def _pool_df(n: int = 8) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "catalog_id": str(2000 + i), "name": str(2000 + i), "x": 100.0 + 40 * i, "y": 300.0,
                "ra_deg": 150.0 + i / 100, "dec_deg": 45.0, "mag": 11.0 + 0.1 * i, "bp_rp": 0.8,
                "is_usable": True, "is_saturated": False, "is_noisy": False, "vsx_known_variable": False,
                "likely_saturated": False, "zone": "linear", "source_state": "DETECTED_P1",
                "vy_identity_gate": "ok", "gaia_dao_resid_px": 0.2, "snr_ap_pixscaled": 200.0,
                "non_single_star": 1 if i == 1 else 0,
            }
        )
    return pd.DataFrame(rows)


def test_pool_keeps_binaries_drops_per_frame_saturated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A Gaia binary (non_single_star > 0) that does not vary is a valid comparison (Milan
    2026-10-14: what matters is that a comparison is not variable). PHASE1-ONE-PASS-2026-10: a comp
    over COMP-SAT-80 in more than 10 % of its frames leaves the pool; one over it in <= 10 % stays."""
    from config import AppConfig
    from photometry_comp import build_global_comp_pool

    stats = {str(2000 + i): {"n_frames": 40, "n_peak": 40, "n_over": 0} for i in range(8)}
    stats["2002"]["n_over"] = 5   # 12.5 % -> out
    stats["2003"]["n_over"] = 4   # 10 % -> stays
    monkeypatch.setattr("photometry_comp.comp_frame_stats", lambda *_a, **_k: stats)
    cfg = AppConfig()
    cfg.comp_pool_derived_admission = False
    pool = build_global_comp_pool(
        masterstars_df=_pool_df(), per_frame_csv_paths=[tmp_path / "proc_001.csv"], csv_cache={},
        variable_target_catalog_ids=frozenset(), safe_bbox=None, chip_fw=1000, chip_fh=1000,
        chip_interior_margin_px=0, max_comp_rms=0.1, cfg=cfg,
    )
    ids = set(pool["catalog_id"].astype(str))
    assert "2001" in ids, "a non-variable binary stays a comparison candidate"
    assert "2002" not in ids, "comp saturated in > 10 % of frames must leave the pool"
    assert {"2000", "2003", "2004"} <= ids
    assert set(pool["comp_n_frames"].astype(int)) == {40}


def test_aperture_photometry_survives_a_nan_position() -> None:
    """One row without a finite position is not measured; it must not wipe the aperture
    photometry of the whole frame (532 NEW run: every frame lost flux_k* / sky columns)."""
    from photometry_shared import enhance_catalog_dataframe_aperture

    rng = np.random.default_rng(5)
    data = rng.normal(800.0, 2.5, size=(256, 256)).astype(np.float32)
    xs = rng.uniform(40, 216, size=10)
    ys = rng.uniform(40, 216, size=10)
    xs[3] = np.nan
    df = pd.DataFrame({"x": xs, "y": ys, "flux": np.full(10, 1e4), "catalog_id": [f"G{i}" for i in range(10)],
                       "peak_max_adu": np.full(10, 100.0)})
    out = enhance_catalog_dataframe_aperture(
        df, data, {"FWHM": 4.2}, aperture_fwhm_factor=1.7, annulus_inner_fwhm=4.0,
        annulus_outer_fwhm=6.0, err_empty_apertures_n=32, err_empty_apertures_min=8,
    )
    assert "sky_adu_per_px_annulus" in out.columns
    k_cols = [c for c in out.columns if c.startswith("flux_k")]
    assert k_cols
    fin = np.isfinite(pd.to_numeric(out["sky_adu_per_px_annulus"], errors="coerce").to_numpy())
    assert fin.sum() == 9 and not fin[3]
    assert np.isfinite(pd.to_numeric(out[k_cols[0]], errors="coerce").to_numpy()[[0, 1, 2, 4]]).all()


def test_off_chip_detection_is_dropped() -> None:
    """A DAO 'detection' with a centroid off the chip is no star and gets no row; the DAO flux
    sign is not a criterion (faint stars on a bright local background have DAO flux <= 0)."""
    from catalog_match import detection_on_chip

    x = np.array([10.0, -0.4, 399.0, 399.5, 50.0, np.nan])
    y = np.array([10.0, 50.0, 399.0, 10.0, -0.01, 10.0])
    ok = detection_on_chip(x, y, wpx=NX, h=NY)
    assert ok.tolist() == [True, False, True, False, False, False]
    src = (Path(__file__).resolve().parents[2] / "src_py" / "catalog_match.py").read_text(encoding="utf-8")
    assert "on_chip = detection_on_chip(" in src


def _depth_field(g50: float, sigma: float, g_max: float, seed: int = 3):
    from scipy.special import erf

    rng = np.random.default_rng(seed)
    n = 6000
    g = 10.0 + 8.0 * rng.random(n) ** 0.35  # more faint stars, like the sky
    x = rng.uniform(0, 2000, n)
    y = rng.uniform(0, 2000, n)
    p_det = 0.5 * (1.0 + erf((g50 - g) / (np.sqrt(2.0) * sigma)))
    det = rng.random(n) < p_det
    zp = 25.0
    peak = 10 ** (-0.4 * (g[det] - zp)) * 10 ** (0.4 * rng.normal(0, sigma, det.sum()))
    in_cat = g <= g_max
    return dict(det_x=x[det] + rng.normal(0, 0.2, det.sum()), det_y=y[det] + rng.normal(0, 0.2, det.sum()),
                det_peak=peak, gaia_x=x[in_cat], gaia_y=y[in_cat], gaia_g=g[in_cat],
                threshold_adu=10 ** (-0.4 * (g50 - zp)), wpx=2000, h=2000, match_r_px=1.5)


def test_frame_depth_fit_and_detector_estimate() -> None:
    """FRAME-DEPTH-2026-10: limit measured from the frame; with a catalogue deeper than the frame the
    completeness fit decides; with a shallower one the detector estimate gives the frame limit and
    the matching depth is capped by the catalogue."""
    from frame_depth import measure_frame_depth

    deep = measure_frame_depth(**_depth_field(15.0, 0.35, 18.0), db_depth_g=18.0)
    assert deep["method"] == "completeness_fit"
    assert abs(deep["g50"] - 15.0) < 0.15
    assert abs(deep["g_lim"] - (15.0 + 3 * 0.35)) < 0.4 and not deep["limited_by_catalogue"]
    shallow = measure_frame_depth(**_depth_field(15.0, 0.35, 14.0), db_depth_g=14.0)
    assert shallow["method"] == "detector_estimate" and shallow["censored"]
    assert abs(shallow["g_lim_frame"] - deep["g_lim_frame"]) < 0.5
    assert shallow["g_lim"] == 14.0 and shallow["limited_by_catalogue"]


def test_no_static_depth_parameter() -> None:
    from config import KNOWN_REMOVED_KEYS, AppConfig

    cfg = AppConfig()
    for k in ("masterstar_gaia_census_target_depth_g", "phase01_comparison_exclude_gaia_nss"):
        assert not hasattr(cfg, k) and k in KNOWN_REMOVED_KEYS
    src = (Path(__file__).resolve().parents[2] / "src_py" / "masterstar_build.py").read_text(encoding="utf-8")
    assert "_census_depth_g = 17.5" not in src
