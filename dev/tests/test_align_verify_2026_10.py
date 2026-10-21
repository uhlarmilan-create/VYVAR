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
        max_control_points=80, min_detected_stars=30, max_detected_stars=500, fb_align=FWHM,
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
