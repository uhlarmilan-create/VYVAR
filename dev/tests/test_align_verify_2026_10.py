"""ALIGN-VERIFY-2026-10 / CHECK-NOT-SUSPECT-2026-10 / COMP-SNR-20-2026-10 (Milan 2026-10-12).

Draft 526 (BO CVn): astroalign failed on 5 frames, the translation-only phase-correlation
fallback was accepted although the frames were rotated ~0.4 deg against the post-flip
reference, stars sat 1-7 px off the apertures and every frame was flagged aligned.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

import vyvar_alignment_frame as V  # noqa: E402
from check_star_kmag import drop_smart_rejected_check_candidates  # noqa: E402
from pipeline_constants import ALIGN_P90_FLOOR_PX, ALIGN_P90_MAX_FWHM, COMP_SNR_MIN  # noqa: E402

H, W, FWHM = 600, 800, 2.4


def _stars(n: int = 220, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xy = np.c_[rng.uniform(30, W - 30, n), rng.uniform(30, H - 30, n)]
    flux = 10 ** rng.uniform(3.0, 5.0, n)
    return xy, flux


def _image(xy: np.ndarray, flux: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 1000.0)
    s = FWHM / 2.3548
    for (x, y), f in zip(xy, flux):
        x0, y0 = int(x), int(y)
        sl = (slice(max(0, y0 - 8), y0 + 9), slice(max(0, x0 - 8), x0 + 9))
        img[sl] += f / (2 * np.pi * s * s) * np.exp(-((xx[sl] - x) ** 2 + (yy[sl] - y) ** 2) / (2 * s * s))
    return (img + rng.normal(0, 8.0, img.shape)).astype(np.float32)


def _rot(xy: np.ndarray, deg: float, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """frame = R(ref) + shift; returns frame xy and the frame->reference 3x3 matrix."""
    a = np.radians(deg)
    c = np.array([W / 2, H / 2])
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    fxy = (xy - c) @ r.T + c + [dx, dy]
    m = np.eye(3)
    m[:2, :2] = r.T
    m[:2, 2] = c - r.T @ (c + [dx, dy])
    return fxy, m


def _ctx(ref_img: np.ndarray, tmp: Path) -> dict:
    rxy = V._alignment_detect_xy(ref_img, 300, det_sigma=4.5, fwhm_px=FWHM)[:160]
    return dict(
        ref_data=ref_img, ref_hdr=fits.Header(), ref_fp_name="REF.fits", fixed_target_pts=rxy,
        reference_list=rxy.tolist(), has_ref_wcs=False, platesolve_dir=str(tmp), align_star_cap=160,
        max_control_points=80, min_detected_stars=30, max_detected_stars=500, fb_align=FWHM, det_sigma=4.5,
        rotation_ref_angle_deg=None,
    )


def _write(tmp: Path, name: str, img: np.ndarray) -> Path:
    h = fits.Header()
    h["VY_FWHM"] = FWHM
    p = tmp / name
    fits.writeto(p, img, h, overwrite=True)
    return p


@pytest.fixture(scope="module")
def field(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("align")
    xy, flux = _stars()
    ref = _image(xy, flux, seed=1)
    fxy, m_true = _rot(xy, 0.4, -4.0, -5.5)
    frame = _image(fxy, flux, seed=2)
    return tmp, xy, flux, ref, fxy, m_true, _write(tmp, "frame.fits", frame)


def test_translation_only_is_rejected(field):
    tmp, xy, flux, ref, fxy, m_true, fp = field
    m_shift = np.eye(3)
    m_shift[:2, 2] = (m_true @ np.array([W / 2, H / 2, 1.0]))[:2] - [W / 2, H / 2]
    assert not V.registration_ok(V.registration_check(fxy, xy, m_shift, fwhm_px=FWHM), fwhm_px=FWHM)
    chk = V.registration_check(fxy, xy, m_true, fwhm_px=FWHM)
    assert V.registration_ok(chk, fwhm_px=FWHM) and chk.resid_px < 1e-6 and chk.pairs == len(xy)


def test_star_similarity_recovers_rotation_when_astroalign_fails(field, monkeypatch):
    tmp, xy, flux, ref, fxy, m_true, fp = field
    monkeypatch.setattr(V, "_alignment_run_astroalign_points", lambda **k: (None, "forced failure", None))
    log: list[str] = []
    r = V._alignment_compute_one_frame(fp, 1, _ctx(ref, tmp), log)
    assert r["kind"] == "aligned" and r["aligned_method"] == "star_similarity"
    sc = r["star_count"]
    assert sc["align_residual_px"] <= max(ALIGN_P90_MAX_FWHM * FWHM, ALIGN_P90_FLOOR_PX) and sc["align_pairs"] >= 50
    m = np.asarray(r["registration_handoff"]["matrix"]).reshape(3, 3)
    assert np.degrees(np.arctan2(m[1, 0], m[0, 0])) == pytest.approx(-0.4, abs=0.02)
    assert any("forced failure" in s for s in log)  # astroalign error text is logged


def test_astroalign_path_is_verified_and_reported(field):
    tmp, xy, flux, ref, fxy, m_true, fp = field
    r = V._alignment_compute_one_frame(fp, 1, _ctx(ref, tmp), [])
    assert r["kind"] == "aligned" and r["aligned_method"] == "astroalign"
    assert r["star_count"]["align_residual_px"] <= max(ALIGN_P90_MAX_FWHM * FWHM, ALIGN_P90_FLOOR_PX)


def test_unverifiable_frame_is_skipped_not_identity(field, monkeypatch):
    tmp, xy, flux, ref, fxy, m_true, fp = field
    other_xy, other_flux = _stars(seed=99)  # a different field: nothing can be verified
    p = _write(tmp, "other.fits", _image(other_xy, other_flux, seed=5))
    monkeypatch.setattr(V, "_alignment_run_astroalign_points", lambda **k: (None, "forced failure", None))
    r = V._alignment_compute_one_frame(p, 1, _ctx(ref, tmp), [])
    assert r["kind"] == "failed_skip"
    assert r["star_count"]["reason"] == "alignment_not_verified" and r["star_count"]["aligned"] is False
    assert "aligned_data" not in r


def test_no_translation_or_identity_fallback_left():
    src = Path(V.__file__).read_text(encoding="utf-8")
    assert '"phase_correlation"' not in src and '"wcs_shift"' not in src and '"identity"' not in src


def test_check_star_excludes_comps_dropped_by_comp_smart():
    pool = pd.DataFrame({"catalog_id": ["1500576880763957248", "1497771992240531712", "42"]})
    out = drop_smart_rejected_check_candidates(pool, {"replaced": ["1500576880763957248"]}, "T")
    assert out["catalog_id"].tolist() == ["1497771992240531712", "42"]
    assert drop_smart_rejected_check_candidates(pool, None).equals(pool)


def test_comp_snr_min_is_20():
    assert COMP_SNR_MIN == 20.0


def test_nan_inputs_do_not_crash():
    xy = np.array([[10.0, 10.0], [np.nan, 5.0], [50.0, 60.0]])
    assert V.registration_check(xy, xy, np.full(9, np.nan), fwhm_px=2.0).pairs == 0
    assert V.registration_check(xy, xy, np.eye(3), fwhm_px=2.0).pairs == 2
    t, chk = V.fit_similarity_from_stars(xy, xy, [np.full((3, 3), np.nan)], fwhm_px=2.0)
    assert t is None and chk.pairs == 0


def test_small_rotation_on_large_chip_is_caught_by_p90():
    rng = np.random.default_rng(7)
    w, h = 6000, 4000
    ref = np.c_[rng.uniform(0, w, 400), rng.uniform(0, h, 400)]
    a = np.radians(0.012)
    c = np.array([w / 2, h / 2])
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    frame = (ref - c) @ r.T + c
    chk = V.registration_check(frame, ref, np.eye(3), fwhm_px=2.0)
    assert chk.resid_px <= 0.5  # the median alone would pass
    assert not V.registration_ok(chk, fwhm_px=2.0)


def test_meridian_flip_recovered_without_astroalign(field, monkeypatch):
    tmp, xy, flux, ref, fxy, m_true, fp = field
    flipped = np.c_[W - 1.0 - xy[:, 0], H - 1.0 - xy[:, 1]] + [1.5, -2.0]
    p = _write(tmp, "flip.fits", _image(flipped, flux, seed=11))
    monkeypatch.setattr(V, "_alignment_run_astroalign_points", lambda **k: (None, "forced failure", None))
    r = V._alignment_compute_one_frame(p, 1, _ctx(ref, tmp), [])
    assert r["kind"] == "aligned" and r["aligned_method"] == "star_similarity"
    assert r["star_count"]["align_residual_px"] <= max(ALIGN_P90_MAX_FWHM * FWHM, ALIGN_P90_FLOOR_PX)


def test_osc_handoff_without_matrix_is_not_aligned():
    from osc_align import apply_registration_handoff_to_frame

    img = np.zeros((20, 20), dtype=np.float32)
    for entry in ({}, {"method": "phase_correlation", "dx": 1.0, "dy": 2.0}, {"method": "astroalign", "matrix": [np.nan] * 9}):
        out, _h, m = apply_registration_handoff_to_frame(
            frame_data=img, frame_hdr=fits.Header(), ref_data=img,
            ref_hdr=fits.Header(), handoff_entry=entry,
        )
        assert out is None and m == "osc_handoff_none"


# ALIGN-WCS-2026-10: a frame WCS is checked on the frame's stars before any reprojection.
def _wcs_hdr(cd: np.ndarray, crpix: tuple[float, float]) -> fits.Header:
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [241.3, 50.26]
    w.wcs.crpix = [crpix[0] + 1.0, crpix[1] + 1.0]  # FITS 1-based
    w.wcs.cd = cd
    return w.to_header()


_CD_REF = np.array([[-2.6e-4, 0.0], [0.0, 2.6e-4]])


def _ctx_wcs(ref: np.ndarray, tmp: Path) -> dict:
    c = _ctx(ref, tmp)
    c.update(ref_hdr=_wcs_hdr(_CD_REF, (W / 2, H / 2)), has_ref_wcs=True)
    return c


def _write_wcs(tmp: Path, name: str, img: np.ndarray, hdr: fits.Header) -> Path:
    hdr["VY_FWHM"] = FWHM
    p = tmp / name
    fits.writeto(p, img, hdr, overwrite=True)
    return p


def test_capture_wcs_failing_the_star_check_is_not_reprojected(field, monkeypatch):
    tmp, xy, flux, ref, fxy, m_true, fp = field
    import reproject

    # A capture-program WCS that ignores the 0.4 deg rotation and the shift of the frame.
    p = _write_wcs(tmp, "capwcs.fits", fits.getdata(fp), _wcs_hdr(_CD_REF, (W / 2, H / 2)))
    monkeypatch.setattr(reproject, "reproject_interp", lambda *a, **k: (_ for _ in ()).throw(AssertionError("reprojected")))
    log: list[str] = []
    r = V._alignment_compute_one_frame(p, 1, _ctx_wcs(ref, tmp), log)
    assert r["kind"] == "aligned" and r["aligned_method"] == "star_similarity"
    assert any("wcs_reproject not verified on stars" in s for s in log)
    assert r["star_count"]["align_residual_px"] <= max(ALIGN_P90_MAX_FWHM * FWHM, ALIGN_P90_FLOOR_PX)


def test_accurate_frame_wcs_is_reprojected(field):
    tmp, xy, flux, ref, fxy, m_true, fp = field
    # The frame WCS that maps the frame stars exactly onto the reference: CD_f = CD_ref M,
    # CRPIX_f = M^-1 (CRPIX_ref - t) for frame->reference M x + t.
    a, t = m_true[:2, :2], m_true[:2, 2]
    crpix_f = np.linalg.solve(a, np.array([W / 2, H / 2]) - t)
    p = _write_wcs(tmp, "goodwcs.fits", fits.getdata(fp), _wcs_hdr(_CD_REF @ a, (crpix_f[0], crpix_f[1])))
    r = V._alignment_compute_one_frame(p, 1, _ctx_wcs(ref, tmp), [])
    assert r["kind"] == "aligned" and r["aligned_method"] == "wcs_reproject"


def _distort(xy: np.ndarray, k: float) -> np.ndarray:
    """Quadratic field distortion about the chip centre (a wide-field lens after a shift)."""
    c = np.array([W / 2, H / 2])
    u = xy - c
    return xy + k * np.c_[u[:, 0] ** 2 - u[:, 1] ** 2, 2.0 * u[:, 0] * u[:, 1]]


def test_distorted_wide_field_is_registered_with_the_polynomial(tmp_path, monkeypatch):
    """ALIGN-POLY-2026-10 (MN Boo 538): the similarity leaves the distortion (p90 over the gate);
    the 2nd-order polynomial verifies, the warp puts the stars on the reference positions and
    VY_AM carries a close projective approximation for the footprint."""
    xy, flux = _stars(seed=7)
    ref = _image(xy, flux, seed=11)
    fxy0, _m = _rot(xy, 0.3, 30.0, -20.0)
    fxy = _distort(fxy0, 6e-6)
    p = _write(tmp_path, "wide.fits", _image(fxy, flux, seed=12))
    rxy = V._alignment_detect_xy(ref, 300, det_sigma=4.5, fwhm_px=FWHM)[:160]
    sxy = V._alignment_detect_xy(_image(fxy, flux, seed=12), 300, det_sigma=4.5, fwhm_px=FWHM)
    t, chk_sim = V.fit_similarity_from_stars(sxy, rxy, [np.asarray(_m)], fwhm_px=FWHM)
    assert not V.registration_ok(chk_sim, fwhm_px=FWHM)  # the case of MN Boo frames 1-101

    monkeypatch.setattr(V, "_alignment_run_astroalign_points", lambda **k: (None, "forced failure", None))
    r = V._alignment_compute_one_frame(p, 1, _ctx(ref, tmp_path), [])
    assert r["kind"] == "aligned" and r["aligned_method"] == "star_polynomial"
    assert r["star_count"]["align_p90_px"] < 0.25
    rh = r["registration_handoff"]
    assert rh["poly_order"] == 2 and len(rh["poly_inv"]) == 12 and rh["matrix_max_err_px"] < 1.0
    # bright stars of the warped frame sit on the reference positions
    al = np.asarray(r["aligned_data"], dtype=np.float64)
    bright = np.argsort(-flux)[:20]
    errs = []
    for i in bright:
        x, y = xy[i]
        if not (12 < x < W - 12 and 12 < y < H - 12):
            continue
        yy, xx = np.mgrid[int(y) - 4 : int(y) + 5, int(x) - 4 : int(x) + 5]
        w = np.clip(al[yy, xx] - np.median(al), 0, None)
        errs.append(np.hypot((w * xx).sum() / w.sum() - x, (w * yy).sum() / w.sum() - y))
    assert np.median(errs) < 0.15


def test_model_selection_keeps_similarity_without_distortion(field):
    """ALIGN-MODEL-2026-10: a field without distortion keeps the verified similarity (BIC charges
    the polynomial's extra parameters)."""
    tmp, xy, flux, ref, fxy, m_true, fp = field
    r = V._alignment_compute_one_frame(fp, 1, _ctx(ref, tmp), [])
    assert r["aligned_method"] == "astroalign"
    assert r["star_count"]["align_model_dbic"] <= 10.0


def test_model_selection_takes_polynomial_when_the_stars_show_distortion(tmp_path):
    """A mild distortion that the similarity still verifies, but the stars clearly prefer the
    polynomial (MN Boo frames 105-135): the frame is registered with the polynomial."""
    xy, flux = _stars(seed=21)
    ref = _image(xy, flux, seed=22)
    fxy0, _m = _rot(xy, 0.2, 12.0, -9.0)
    fxy = _distort(fxy0, 1.5e-6)
    img = _image(fxy, flux, seed=23)
    sxy = V._alignment_detect_xy(img, 300, det_sigma=4.5, fwhm_px=FWHM)
    rxy = V._alignment_detect_xy(ref, 300, det_sigma=4.5, fwhm_px=FWHM)[:160]
    t, chk = V.fit_similarity_from_stars(sxy, rxy, [np.asarray(_m)], fwhm_px=FWHM)
    assert V.registration_ok(chk, fwhm_px=FWHM)  # the similarity alone would pass
    p = _write(tmp_path, "mild.fits", img)
    r = V._alignment_compute_one_frame(p, 1, _ctx(ref, tmp_path), [])
    assert r["aligned_method"] == "star_polynomial"
    assert r["star_count"]["align_model_dbic"] > 10.0
    assert r["star_count"]["align_p90_px"] < chk.p90_px
