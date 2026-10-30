"""Step 4 (T02 frame QC / registration / MASTERSTAR leftovers): contract tests (tag frm)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
sys.path.insert(0, str(SRC))


def _gauss(shape: tuple[int, int], x0: float, y0: float, amp: float, sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))


def test_pass2_gaia_pair_over_one_psf_gives_one_row() -> None:
    """G04-07: two Gaia seeds 1.6 px apart over one PSF -> exactly one pass-2 row (brighter seed owns)."""
    from masterstar_gaia_accounting import (
        Pass2AcceptParams,
        dao_pass2_born_owned_rows,
        merge_dao_pass1_pass2_born_owned,
    )
    from pipeline_constants import MS_DAO_PASS1_DEDUP_PX

    rng = np.random.default_rng(7)
    img = 100.0 + rng.normal(0.0, 3.0, (200, 200)) + _gauss((200, 200), 100.0, 100.0, 2000.0, 1.5)
    gaia = pd.DataFrame(
        {"catalog_id": ["905", "906"], "x_gaia": [99.2, 100.8], "y_gaia": [100.0, 100.0], "g_mag": [15.0, 14.0]}
    )
    params = Pass2AcceptParams(sigma=4.0, center_tol_px=2.0, fwhm_px=3.5)
    rows, n_unm, n_acc, _amb = dao_pass2_born_owned_rows(
        img, None, gaia_chip=gaia, bfac=1, pass2_params=params, target_depth_g=18.0,
        edge_margin_px=5.0, match_r_px=3.0, wpx=200, h=200,
    )
    assert n_unm == 2 and n_acc == 2  # the defect's precondition: both seeds accept the centroid
    tbl = merge_dao_pass1_pass2_born_owned(None, rows, gaia_chip=gaia)
    assert len(tbl) == 1
    assert str(tbl["vy_seed_catalog_id"][0]) == "906"  # G 14.0 < 15.0
    # invariant: no two detection rows closer than the pass-1 duplicate radius
    xy = np.column_stack([np.asarray(tbl["x_centroid"]), np.asarray(tbl["y_centroid"])])
    d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    assert np.all(d[np.triu_indices(len(xy), 1)] >= MS_DAO_PASS1_DEDUP_PX)


def test_pass2_separate_stars_are_kept() -> None:
    """G04-07: two separate stars 30 px apart stay two pass-2 rows."""
    from masterstar_gaia_accounting import (
        Pass2AcceptParams,
        dao_pass2_born_owned_rows,
        merge_dao_pass1_pass2_born_owned,
    )

    rng = np.random.default_rng(8)
    img = 100.0 + rng.normal(0.0, 3.0, (200, 200))
    img += _gauss((200, 200), 85.0, 100.0, 2000.0, 1.5) + _gauss((200, 200), 115.0, 100.0, 1500.0, 1.5)
    gaia = pd.DataFrame(
        {"catalog_id": ["1", "2"], "x_gaia": [85.0, 115.0], "y_gaia": [100.0, 100.0], "g_mag": [14.0, 14.3]}
    )
    rows, _n, _a, _amb = dao_pass2_born_owned_rows(
        img, None, gaia_chip=gaia, bfac=1, pass2_params=Pass2AcceptParams(sigma=4.0, center_tol_px=2.0, fwhm_px=3.5),
        target_depth_g=18.0, edge_margin_px=5.0, match_r_px=3.0, wpx=200, h=200,
    )
    tbl = merge_dao_pass1_pass2_born_owned(None, rows, gaia_chip=gaia)
    assert len(tbl) == 2


def test_dao_only_negative_dao_flux_with_positive_aperture_flux_is_not_artifact() -> None:
    """G04-24: a faint star on a bright local background (DAO flux < 0, aperture flux > 0) is not
    artifact_negative; a non-positive aperture flux is."""
    from dao_reconcile import (
        DAO_ONLY_CLASS_ARTIFACT_NEGATIVE,
        ConfirmableDepthResult,
        FluxToGFitResult,
        classify_dao_only_dataframe,
    )

    depth = ConfirmableDepthResult(17.5, "catalogue_depth_g", {"catalogue_depth_g": 17.5}, True)
    fit = FluxToGFitResult(True, 22.0, 10, 0.05, 0.431, "median_zp")
    df = pd.DataFrame(
        {
            "source_type": ["DAO_ONLY", "DAO_ONLY"],
            "flux": [-20.0, 300.0],
            "flux_ap": [450.0, -3.0],
            "peak_max_adu": [40.0, 40.0],
            "likely_saturated": [False, False],
        }
    )
    out, _meta = classify_dao_only_dataframe(df, depth=depth, flux_fit=fit, frame_noise_adu=10.0)
    assert out["dao_only_class"].iloc[0] != DAO_ONLY_CLASS_ARTIFACT_NEGATIVE
    assert out["dao_only_class"].iloc[1] == DAO_ONLY_CLASS_ARTIFACT_NEGATIVE
    with pytest.raises(ValueError, match="flux_ap"):
        classify_dao_only_dataframe(df.drop(columns=["flux_ap"]), depth=depth, flux_fit=fit, frame_noise_adu=10.0)


def test_generate_masterstar_and_catalog_end_to_end_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """G04-46: generate_masterstar_and_catalog end to end on a synthetic field (local Gaia / objects DBs,
    MASTERSTAR with a slightly wrong WCS -> astrosolve confirm). Invariants of the written catalogue:
    one detection per centroid (G04-07: a Gaia pair over one faint pass-2 star), unique ids, membership
    rows never DETECTED, DETECTED == own DAO pass, Gaia variable flag by identity (G04-01),
    artifact_negative from the aperture flux (G04-24)."""
    import math
    import sqlite3

    from astropy.io import fits

    sys.path.insert(0, str(ROOT / "dev" / "tests"))
    import rig_photometry
    import test_identity_2026_10 as T
    from config import AppConfig
    from masterstar_build import generate_masterstar_and_catalog
    from pipeline_constants import MS_DAO_PASS1_DEDUP_PX

    monkeypatch.setattr(
        rig_photometry, "photometric_gain_read_noise",
        lambda *a, **k: rig_photometry.PhotometricGain(1.0, 5.0, 1.0, 1.0, "test", "test"),
    )
    w = T._wcs()
    rng = np.random.default_rng(7)
    stars: list[tuple[int, float, float, float]] = []
    sid = 1000
    while len(stars) < 60:
        x, y = rng.uniform(20, T.NX - 20, 2)
        if math.hypot(x - 300, y - 100) < 30:
            continue
        stars.append((sid, float(x), float(y), float(rng.uniform(10.5, 13.5))))
        sid += 1
    rows = []
    for s, x, y, g in stars:
        c = w.pixel_to_world(x, y)
        rows.append((s, float(c.ra.deg), float(c.dec.deg), g))
    # Gaia pair 1.6 px apart over ONE faint star (below the pass-1 threshold, found by pass 2 only)
    for s, dx in ((1, -0.8), (2, 0.8)):
        c = w.pixel_to_world(300.0 + dx, 100.0)
        rows.append((s, float(c.ra.deg), float(c.dec.deg), 13.0 + 0.1 * s))
    img = T._render([(x, y, 10 ** (-0.4 * (g - T.ZP))) for _, x, y, g in stars] + [(300.0, 100.0, 10 ** (-0.4 * (19.2 - T.ZP)))])
    db = T._gaia_db(tmp_path / "gaia.db", rows)
    var_id = stars[5][0]
    con = sqlite3.connect(db)
    con.execute("UPDATE gaia_dr3 SET var_flag = 'VARIABLE' WHERE source_id = ?", (var_id,))
    con.commit()
    con.close()
    odb = tmp_path / "objects.db"
    con = sqlite3.connect(odb)
    con.execute("CREATE TABLE vsx_data (oid INTEGER PRIMARY KEY, name TEXT, ra_deg REAL, dec_deg REAL, var_type TEXT,"
                " period REAL, mag_max REAL, mag_min REAL)")
    con.execute("INSERT INTO vsx_data VALUES (1, 'FAR V1', 10.0, -40.0, 'EA', 1.2, 10.0, 10.5)")
    con.execute("CREATE TABLE exoplanet_data (obj_id TEXT PRIMARY KEY, name TEXT, host_name TEXT, ra_deg REAL,"
                " dec_deg REAL, cat_source TEXT, disposition TEXT, period REAL, mag REAL, mag_band TEXT)")
    con.execute("INSERT INTO exoplanet_data VALUES ('TOI-1.01','TOI-1.01','TIC 1',10.0,-40.0,'TOI','PC',3.1,9.0,'TESS')")
    con.commit()
    con.close()
    ap = tmp_path / "draft_000001"
    (ap / "platesolve").mkdir(parents=True)
    h = T._wcs(dx=1.0, dy=-1.0).to_header()
    h["RA"], h["DEC"] = 150.0, 45.0
    fits.PrimaryHDU(img, header=h).writeto(ap / "platesolve" / "MASTERSTAR.fits")
    cfg = AppConfig()
    cfg.gaia_db_path = str(db)
    cfg.objects_db_path = str(odb)

    out = generate_masterstar_and_catalog(
        archive_path=ap, app_config=cfg, masterstar_skip_build=True, masterstar_platesolve_skip_solve=True
    )
    df = pd.read_csv(out["masterstars_csv"], dtype={"catalog_id": str})
    cid = df["catalog_id"].fillna("").astype(str).str.strip()
    vp = pd.to_numeric(df["vy_dao_pass"], errors="coerce")
    det = vp.isin([1, 2])
    st = df["source_state"].fillna("").astype(str)
    # the pass-2 pair: exactly one detection row at the star
    at_pair = det & (df["x"] - 300.0).abs().lt(2.0) & (df["y"] - 100.0).abs().lt(2.0)
    assert int(at_pair.sum()) == 1 and int(vp[at_pair].iloc[0]) == 2
    xy = df.loc[det, ["x", "y"]].to_numpy(dtype=float)
    dd = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    assert float(dd[np.triu_indices(len(xy), 1)].min()) >= MS_DAO_PASS1_DEDUP_PX
    assert not cid[cid.ne("")].duplicated().any()
    assert not st[~det].str.startswith("DETECTED").any()
    assert st[det & cid.ne("")].str.startswith("DETECTED").all()
    assert int(out["catalog_matched"]) == int((det & cid.ne("")).sum())
    gv = df["gaia_dr3_variable_catalog"].astype(str).str.lower().isin(["true", "1"])
    assert set(cid[gv]) == {str(var_id)}
    dao_only = st.eq("DAO_ONLY")
    neg = df["dao_only_class"].fillna("").eq("artifact_negative")
    assert (neg == (dao_only & pd.to_numeric(df["flux_ap"], errors="coerce").le(0))).all()


def test_polynomial_warp_failure_reports_the_similarity_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """G02-23: a verified similarity is replaced by the polynomial (dBIC > 10); when the polynomial warp
    fails, the frame keeps the similarity and alignment_report gets the similarity's residual / pairs."""
    sys.path.insert(0, str(ROOT / "dev" / "tests"))
    import test_align_verify_2026_10 as AV
    import vyvar_alignment_frame as V

    xy, flux = AV._stars(seed=21)
    ref = AV._image(xy, flux, seed=22)
    fxy0, _m = AV._rot(xy, 0.2, 12.0, -9.0)
    p = AV._write(tmp_path, "mild.fits", AV._image(AV._distort(fxy0, 1.5e-6), flux, seed=23))

    _orig = V.warp_to_reference

    def _boom(data, shape, inverse_map):
        # only the polynomial warp fails (the similarity warp shares warp_to_reference since CAL-BADFLAT)
        if "Polynomial" in type(inverse_map).__name__ or not hasattr(inverse_map, "params"):
            raise RuntimeError("forced warp failure")
        return _orig(data, shape, inverse_map)

    monkeypatch.setattr(V, "warp_to_reference", _boom)
    ctx = AV._ctx(ref, tmp_path)
    r = V._alignment_compute_one_frame(p, 1, ctx, [])
    assert r["kind"] == "aligned" and r["aligned_method"] in ("astroalign", "star_similarity")
    assert r["star_count"]["align_model_dbic"] > 10.0  # the polynomial was preferred
    chk = V.registration_check(r["xy_used"], ctx["fixed_target_pts"], r["registration_handoff"]["matrix"], fwhm_px=AV.FWHM)
    assert r["star_count"]["align_residual_px"] == pytest.approx(chk.resid_px)
    assert r["star_count"]["align_pairs"] == chk.pairs


def test_reference_and_frame_star_lists_use_one_detection_rule(tmp_path: Path) -> None:
    """G02-48: the reference stars (astrometry_align) and a frame's stars (first attempt) of the same
    image are the same list - one rule (the run's DAO sigma, then the same relaxed ladder)."""
    from astropy.io import fits

    sys.path.insert(0, str(ROOT / "dev" / "tests"))
    import astrometry_align as AA
    import test_align_verify_2026_10 as AV
    import vyvar_alignment_frame as V
    from config import AppConfig

    xy, flux = AV._stars(n=400, seed=5)
    img = AV._image(xy, flux, seed=6)
    p = AV._write(tmp_path, "ref.fits", img)
    hdr = fits.getheader(p)
    for sig in (4.5, 3.0):
        out = AA._al_reference_stars(
            _align_det_sigma=sig, _align_star_cap=1000, _cfg_align=AppConfig(), _fb_align=AV.FWHM,
            _t_platesolve=0.0, has_wcs=True, ram_align_and_catalog=False, ref_data=img, ref_fp=p, ref_hdr=hdr,
        )
        ref_xy = np.asarray(out[6])
        ctx = AV._ctx(img, tmp_path)
        ctx.update(det_sigma=sig, fixed_target_pts=ref_xy.astype(np.float32))
        r = V._alignment_compute_one_frame(p, 1, ctx, [])
        assert r["kind"] == "aligned"
        assert np.array_equal(np.asarray(r["xy_used"]), ref_xy), sig
    assert V.alignment_detection_attempts(4.5)[0]["dao_sigma"] == 4.5
    assert [a["dao_sigma"] for a in V.alignment_detection_attempts(2.2)] == [2.2, 2.0, 1.5]


def _am_header(mat: np.ndarray, *, raw_shape: tuple[int, int] | None = None, amerr: float | None = None):
    from astropy.io import fits

    h = fits.Header()
    for k, v in enumerate(np.asarray(mat, dtype=float).reshape(-1)):
        h[f"VY_AM{k}"] = float(v)
    if raw_shape is not None:
        h["VY_RAWNX"], h["VY_RAWNY"] = int(raw_shape[1]), int(raw_shape[0])
    if amerr is not None:
        h["VY_AMERR"] = float(amerr)
    return h


def test_osc_channel_frames_get_the_donor_transform(tmp_path: Path) -> None:
    """G02-03: an OSC channel frame aligned by the registration handoff carries VY_AM0..8 equal to the
    donor matrix (and its raw size); the aligned reference carries the identity."""
    from astropy.io import fits

    import astrometry_align as AA
    from frame_footprint import has_alignment_transform

    det = tmp_path / "det"
    det.mkdir()
    rng = np.random.default_rng(1)
    ref_data = (1000.0 + rng.normal(0, 5, (120, 160))).astype(np.float32)
    ref_data += _gauss((120, 160), 60.0, 50.0, 3000.0, 1.5).astype(np.float32)
    ref_fp, fr_fp = det / "ref.fits", det / "f2.fits"
    fits.writeto(ref_fp, ref_data, overwrite=True)
    fits.writeto(fr_fp, np.roll(ref_data, (2, 3), axis=(0, 1)), overwrite=True)
    mat = [1.0, 0.0, -3.0, 0.0, 1.0, -2.0, 0.0, 0.0, 1.0]
    handoff = {"reference_file": "ref.fits", "frames": {"f2.fits": {"matrix": mat, "method": "astroalign", "aligned": True}}}
    aligned_root = tmp_path / "aligned"
    star_counts: list = []
    AA._al_align_frames(
        _align_det_sigma=4.5, _fb_align=2.4, _flip_logged=False, _frame_nbytes=0, _prog=lambda m: None,
        align_cp=80, align_tasks=[], aligned_files=[], aligned_ram_buffer=[], aligned_root=aligned_root,
        detrended_root=det, files=[ref_fp, fr_fp], fixed_target_pts=np.zeros((0, 2), np.float32), n_files=2,
        n_written_align=0, osc_registration_handoff=handoff, ref_data=ref_data, ref_fp=ref_fp,
        ref_hdr=fits.getheader(ref_fp), ref_wcs_obj=None, rotation_flip_first_index_1based=None,
        rotation_flip_frame_indices_1based=[], rotation_ref_angle_deg=None, star_counts=star_counts,
        use_ram_handoff=False,
    )
    h = fits.getheader(aligned_root / "f2.fits")
    assert has_alignment_transform(h)
    assert [h[f"VY_AM{k}"] for k in range(9)] == mat
    assert (h["VY_RAWNX"], h["VY_RAWNY"]) == (160, 120)


def test_aligned_reference_carries_identity_transform() -> None:
    """G02-03: stamp_alignment_transform refuses an aligned frame without a finite matrix."""
    from astropy.io import fits

    from frame_footprint import stamp_alignment_transform

    h = fits.Header()
    with pytest.raises(ValueError):
        stamp_alignment_transform(h, {"method": "wcs_reproject"}, (10, 10))
    stamp_alignment_transform(h, {"method": "reference", "matrix": list(np.eye(3).ravel())}, (10, 12))
    assert h["VY_AM0"] == 1.0 and h["VY_AM2"] == 0.0 and h["VY_RAWNX"] == 12 and "VY_AMERR" not in h
    src = (SRC / "astrometry_align.py").read_text(encoding="utf-8")
    assert '"method": "reference"' in src  # the aligned reference copy is stamped too


def test_footprint_uses_the_raw_frame_size() -> None:
    """G02-34: a 1000x800 frame shifted onto a 1200x900 reference - the footprint ends at the mapped raw
    corners, not at the reference size taken from the aligned header's NAXIS."""
    from frame_footprint import aligned_footprint, footprint_bbox

    mat = np.eye(3)
    mat[:2, 2] = [50.0, 40.0]
    h = _am_header(mat, raw_shape=(800, 1000))
    h["NAXIS1"], h["NAXIS2"] = 1200, 900  # what astropy writes for the aligned (reference) array
    cov = aligned_footprint(h, (900, 1200))
    cols = np.nonzero(cov[450])[0]
    rows = np.nonzero(cov[:, 600])[0]
    assert (cols.min(), cols.max()) == (51, 1048)  # x 50 .. 1049 mapped, 1 px inside
    assert (rows.min(), rows.max()) == (41, 838)
    assert footprint_bbox(h, (900, 1200)) == (51.0, 41.0, 1048.0, 838.0)


def test_footprint_margin_follows_vy_amerr() -> None:
    """G02-24: the footprint margin is 1 px plus VY_AMERR (the projective VY_AM of a polynomial
    registration is off by up to VY_AMERR px)."""
    from frame_footprint import aligned_footprint, footprint_bbox

    mat = np.eye(3)
    mat[:2, 2] = [30.0, 0.0]
    base = aligned_footprint(_am_header(mat, raw_shape=(200, 300)), (200, 300))
    wide = aligned_footprint(_am_header(mat, raw_shape=(200, 300), amerr=2.0), (200, 300))
    assert np.nonzero(base[100])[0].min() == 31
    assert np.nonzero(wide[100])[0].min() == 33
    assert footprint_bbox(_am_header(mat, raw_shape=(200, 300), amerr=2.0), (200, 300))[0] == 33.0
    with pytest.raises(ValueError):
        aligned_footprint(_am_header(mat, raw_shape=(200, 300), amerr=float("nan")), (200, 300))


def test_forced_rows_outside_the_aligned_footprint_are_not_measured() -> None:
    """G05-11: a frame shifted by 50 px (VY_AM) - a MASTERSTAR star at x=20 lies on the constant fill:
    forced stub with geometry_ok False and no position; a star near x=50 whose annulus is mostly on the
    fill too; a star well inside is measured."""
    from forced_photometry import inject_forced_masterstar_rows

    mat = np.eye(3)
    mat[:2, 2] = [50.0, 0.0]
    hdr = _am_header(mat, raw_shape=(200, 300))
    img = np.full((200, 300), 1000.0)
    master = pd.DataFrame({
        "catalog_id": ["1", "2", "3"], "x": [20.0, 53.0, 150.0], "y": [100.0, 100.0, 100.0],
        "zone": ["linear"] * 3, "is_saturated": [False] * 3,
    })
    out, meta = inject_forced_masterstar_rows(pd.DataFrame(), master, image=img, header=hdr, aperture_radii=(4.0, 8.0, 12.0))
    by = out.set_index("catalog_id")
    assert not bool(by.loc["1", "geometry_ok"]) and by.loc["1", "geometry_miss_reason"] == "outside_aligned_footprint"
    assert np.isnan(by.loc["1", "x"])
    assert not bool(by.loc["2", "geometry_ok"])
    assert bool(by.loc["3", "geometry_ok"]) and by.loc["3", "x"] == 150.0
    assert meta["n_geometry_miss"] == 2 and meta["n_injected"] == 1
    # without the aperture radii the centre pixel decides
    out2, _ = inject_forced_masterstar_rows(pd.DataFrame(), master, image=img, header=hdr)
    assert list(out2.set_index("catalog_id")["geometry_ok"].astype(bool)) == [False, True, True]


def _qc_list(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.c_[rng.uniform(0, 1000, n), rng.uniform(0, 1000, n), 10 ** rng.uniform(3, 5, n)]


@pytest.mark.parametrize("deg", [0.0, 0.5, 1.0, 1.46, 2.0, 3.0])
def test_frame_qc_match_survives_field_rotation(deg: float) -> None:
    """G02-04: transparency_vs_reference on a star list rotated 0-3 deg (and shifted) keeps
    n_common >= 0.8 n and dZP ~ 0 (the translation-only match lost the crop corners)."""
    from frame_qc import transparency_vs_reference

    ref = _qc_list(150, 4)
    a = np.radians(deg)
    c = np.array([500.0, 500.0])
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    cur = ref.copy()
    cur[:, :2] = (ref[:, :2] - c) @ r.T + c + [7.0, -4.0]
    inside = (cur[:, 0] >= 0) & (cur[:, 0] < 1000) & (cur[:, 1] >= 0) & (cur[:, 1] < 1000)
    cur = cur[inside]
    dzp, _patchy, n = transparency_vs_reference(ref, cur, tol=2.0, shape=(1000, 1000))
    assert n >= 0.8 * len(cur), (deg, n, len(cur))
    assert abs(dzp) < 1e-9


def test_frame_qc_meridian_flip_still_matches() -> None:
    """G02-04: the 180 deg flip guess is refined like the translation."""
    from frame_qc import transparency_vs_reference

    ref = _qc_list(150, 5)
    cur = ref.copy()
    cur[:, 0] = 999.0 - ref[:, 0] + 3.0
    cur[:, 1] = 999.0 - ref[:, 1] - 2.0
    _dzp, _p, n = transparency_vs_reference(ref, cur, tol=2.0, shape=(1000, 1000))
    assert n >= 0.8 * len(ref)


def test_frame_without_star_list_is_not_ranked_best() -> None:
    """G02-32: reject=False, the frame with the best FWHM / noise but no star list (no transparency)
    is not the best frame."""
    from frame_qc import evaluate_frame_qc

    rows, stars = [], {}
    ref = _qc_list(120, 6)
    for i in range(6):
        p = f"/f{i}.fits"
        rows.append({"FILE_PATH": p, "OBSERVATION_GROUP_KEY": "V|60|1", "FWHM": 3.0 + 0.1 * i,
                     "ELONGATION_MEAN": 1.05, "SKY_LEVEL": 100.0, "NOISE_SIGMA": 10.0})
        if i != 0:
            s = ref.copy()
            s[:, 2] *= 0.98
            stars[p] = (s, (1000, 1000))
    stars["/f1.fits"] = (ref.copy(), (1000, 1000))  # the provisional reference has stars
    g = evaluate_frame_qc(rows, k=3.0, reject=False, stars=stars)["V|60|1"]
    assert g.best_path != "/f0.fits"
    assert g.best_path == "/f1.fits"


def _ms_ref_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ms_result: dict):
    import hashlib

    from astropy.io import fits

    import astrometry_align as AA
    import masterstar_build

    det = tmp_path / "processed" / "lights" / "V_60_1"
    det.mkdir(parents=True)
    light = det / "light_001.fits"
    h = fits.Header()
    h["DATE-OBS"] = "2026-03-01T22:00:00"
    fits.writeto(light, np.full((40, 50), 100.0, np.float32), h)
    ps = tmp_path / "platesolve" / "V_60_1"
    ps.mkdir(parents=True)
    monkeypatch.setattr(masterstar_build, "generate_masterstar_and_catalog", lambda **k: dict(ms_result))

    def _hashes() -> dict[str, str]:
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tmp_path.rglob("*.fits"))}

    before = _hashes()
    call = dict(
        _cat_info_root=None, _cfg_align=None, _masterstar_built=False, _prog=lambda m: None, _ps_root=ps,
        ap=tmp_path, build_masterstar_and_catalogs=True, detrended_root=det, draft_id=None, equip_sat_adu=None,
        has_wcs=False, id_equipment=None, job={}, n_comparison_stars=0, n_files=1, obs_group_key="V_60_1",
        platesolve_dir=ps, ref_data=np.zeros((40, 50), np.float32), ref_fp=light, ref_hdr=fits.getheader(light),
        require_non_variable_comparisons=True, sibling_recovery_use_masterstar=False,
    )
    return AA._al_masterstar_for_reference, call, before, _hashes, ps, light


def test_reference_switch_to_masterstar_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """G02-09 / G03-20: after the MASTERSTAR build the reference IS MASTERSTAR; no FITS is rewritten
    (the second block re-wrote MASTERSTAR onto itself and, after a failed switch, copied the whole
    MASTERSTAR header - DATE-OBS, EXPTIME - into a processed light)."""
    from astropy.io import fits

    sys.path.insert(0, str(ROOT / "dev" / "tests"))
    import test_identity_2026_10 as T

    ps = tmp_path / "platesolve" / "V_60_1"
    ms = ps / "MASTERSTAR.fits"
    fn, call, _b, hashes, _ps, _light = _ms_ref_call(tmp_path, monkeypatch, {"masterstar_fits": str(ms)})
    hh = T._wcs().to_header()
    hh["DATE-OBS"] = "2026-01-01T00:00:00"
    fits.writeto(ms, np.full((40, 50), 7.0, np.float32), hh)
    before = hashes()
    _cat, built, has_wcs, ref_data, ref_fp, ref_hdr = fn(**call)
    assert Path(ref_fp) == ms.resolve() and has_wcs and built and float(ref_data[0, 0]) == 7.0
    assert hashes() == before
    import pipeline_astrometry

    assert not hasattr(pipeline_astrometry, "_apply_wcs_header_to_fits")


def test_failed_reference_switch_stops_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """G02-09: a MASTERSTAR build without a MASTERSTAR file (or without a WCS) stops the alignment; no
    file under the processed lights changes."""
    sys.path.insert(0, str(ROOT / "dev" / "tests"))
    fn, call, before, hashes, _ps, _light = _ms_ref_call(tmp_path, monkeypatch, {"masterstar_fits": ""})
    with pytest.raises(RuntimeError, match="no alignment reference"):
        fn(**call)
    assert hashes() == before


def test_sibling_donor_keyed_by_setup() -> None:
    """G03-35: a failed g_120_2 takes the solved g_60_2 (same filter); otherwise the nearest filter;
    never itself."""
    from pipeline_astrometry import pick_sibling_donor_setup

    assert pick_sibling_donor_setup("g_120_2", ["r_60_2", "g_60_2", "i_60_2"]) == "g_60_2"
    assert pick_sibling_donor_setup("g_120_2", ["r_60_2", "i_60_2"]) == "r_60_2"
    assert pick_sibling_donor_setup("g_120_2", ["g_120_2"]) is None
    assert pick_sibling_donor_setup("V_60_1", ["B_60_1", "R_60_1"]) == "R_60_1"  # V 551 nm: R 623 nearer than B 440
    src = (SRC / "astrometry_orchestration.py").read_text(encoding="utf-8")
    assert "pick_sibling_donor_setup(setup" in src and "pick_sibling_donor_filter" not in src


def test_best_effort_path_is_an_exact_basename_hit(tmp_path: Path) -> None:
    """G03-41: light_1 + flight_1 -> the request for light_1 hits light_1 (the suffix match could pick
    flight_1); proc_ prefix still matches; one index for many lookups."""
    from pipeline_preprocess import _resolve_best_effort_path_under, fits_basename_index

    d = tmp_path / "lights" / "V_60_1"
    d.mkdir(parents=True)
    for n in ("flight_1.fits", "light_1.fits", "proc_Light_2.fits"):
        (d / n).write_bytes(b"")
    idx = fits_basename_index(tmp_path)
    hit = _resolve_best_effort_path_under(tmp_path, "/raw/elsewhere/light_1.fits", index=idx)
    assert hit is not None and hit.name == "light_1.fits"
    hit2 = _resolve_best_effort_path_under(tmp_path, "C:/raw/Light_2.fits", index=idx)
    assert hit2 is not None and hit2.name == "proc_Light_2.fits"
    (d / "light_1.fits").unlink()
    assert _resolve_best_effort_path_under(tmp_path, "/raw/elsewhere/light_1.fits") is None


@pytest.mark.parametrize("lim", [None, 60000.0])
def test_unsaturated_gaussian_is_never_likely_saturated(lim: float | None) -> None:
    """G05-01 / G02-50 property: Gaussian stars of peak 100-5000 ADU on sky 0-40000 ADU with Poisson
    noise are never likely_saturated (sky + peak < limit); a clipped core is."""
    from pipeline_catalog import _vectorized_star_saturation_columns

    rng = np.random.default_rng(11)
    for sky in (0.0, 1500.0, 12500.0, 33000.0, 40000.0):
        for peak in (100.0, 500.0, 2000.0, 5000.0):
            for sig in (1.0, 1.5, 3.0):
                img = sky + _gauss((41, 41), 20.3, 19.8, peak, sig)
                img = rng.poisson(np.clip(img, 0, None)).astype(float)
                col = _vectorized_star_saturation_columns(
                    img, np.array([20.3]), np.array([19.8]), sat_limit=lim, sat_frac=1.0,
                    peak_dao=np.array([peak]), peak_max_adu=np.array([float(img[18:23, 18:23].max())]),
                )
                assert not bool(col["likely_saturated"][0]), (sky, peak, sig)
    clipped = np.minimum(1500.0 + _gauss((41, 41), 20.0, 20.0, 300000.0, 1.5), 50000.0)
    col = _vectorized_star_saturation_columns(
        clipped, np.array([20.0]), np.array([20.0]), sat_limit=lim, sat_frac=1.0,
        peak_dao=np.array([48500.0]), peak_max_adu=np.array([50000.0]),
    )
    assert bool(col["likely_saturated"][0])
