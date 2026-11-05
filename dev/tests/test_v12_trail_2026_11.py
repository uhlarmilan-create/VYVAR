# -*- coding: ascii -*-
"""VYVAR v12 (Milan 2026-11-05): the MASTERSTAR source is never a frame with a satellite trail
(MASTERSTAR-TRAIL-2026-11). Milan's PC: CX CVn Light_059 and V842 Her Light_116 had trails.

Synthetic frames with known truth: Gaussian noise, stars, gradients, a CMOS row offset, and
injected trails; frame QC ranking with a fake trail test."""
from __future__ import annotations

import math

import numpy as np
import pytest

import frame_qc as fq
import trail_detect as td

FWHM = 2.4


def _frame(rng, *, shape=(700, 1040), n_stars=900, sky=1500.0, noise=25.0, gradient=0.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    img = sky + gradient * (xx / shape[1]) + rng.normal(0.0, noise, shape)
    s = FWHM / 2.3548
    for _ in range(n_stars):
        x0, y0 = rng.uniform(5, shape[1] - 5), rng.uniform(5, shape[0] - 5)
        amp = noise * 10 ** rng.uniform(0.3, 2.5)
        x1, x2 = int(max(0, x0 - 6)), int(min(shape[1], x0 + 7))
        y1, y2 = int(max(0, y0 - 6)), int(min(shape[0], y0 + 7))
        img[y1:y2, x1:x2] += amp * np.exp(-0.5 * (((xx[y1:y2, x1:x2] - x0) ** 2 + (yy[y1:y2, x1:x2] - y0) ** 2) / s ** 2))
    return img


def _add_trail(img, *, amp, angle_deg, through=None):
    """Gaussian-profile line of peak ``amp`` (ADU), FWHM wide, crossing the whole frame."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx, cy = through if through is not None else (w / 2.0, h / 2.0)
    th = math.radians(angle_deg)
    d = -(xx - cx) * math.sin(th) + (yy - cy) * math.cos(th)
    s = FWHM / 2.3548
    return img + amp * np.exp(-0.5 * (d / s) ** 2)


# ------------------------------------------------------------------ the detector
@pytest.mark.parametrize("angle", [17.0, 63.0, 101.0, 152.0])
def test_a_one_sigma_trail_is_found_with_its_amplitude(angle: float) -> None:
    rng = np.random.default_rng(3)
    img = _add_trail(_frame(rng), amp=25.0, angle_deg=angle)  # 1.0 x pixel noise
    r = td.detect_trail(img, FWHM)
    assert r.has_trail and r.z_max >= td.TRAIL_Z_MIN
    assert 0.4 <= r.amplitude_sigma <= 1.2  # lower when the trail straddles two bins (informational)


@pytest.mark.parametrize("case", ["plain", "gradient", "row_offset"])
def test_clean_frames_have_no_trail(case: str) -> None:
    rng = np.random.default_rng(11)
    img = _frame(rng, gradient=400.0 if case == "gradient" else 0.0)
    if case == "row_offset":
        img[350:352, :] += 0.15 * 25.0  # CMOS row pattern of 0.15 sigma (Milan's QHY294 frames)
    r = td.detect_trail(img, FWHM)
    assert not r.has_trail, r.to_dict()


def test_strong_trail_like_v842_her_light_116() -> None:
    rng = np.random.default_rng(5)
    img = _add_trail(_frame(rng), amp=6.0 * 25.0, angle_deg=100.0)  # measured on Milan's frame: A = 6.0
    r = td.detect_trail(img, FWHM)
    assert r.has_trail and r.z_max > 50
    assert 2.4 <= r.amplitude_sigma <= 7.2


def test_bad_input_fails_loud() -> None:
    with pytest.raises(ValueError):
        td.detect_trail(np.zeros((50, 50)), 0.0)
    with pytest.raises(ValueError):
        td.detect_trail(np.zeros(50), FWHM)


# ------------------------------------------------------------------ frame QC ranking
def _rows(n: int = 6):
    # depth T/(FWHM*sigma): frame 0 best, then 1, 2, ...
    return [{"FILE_PATH": f"/a/f{i:03d}.fits", "OBSERVATION_GROUP_KEY": "L|60|2", "FWHM": 2.4 + 0.05 * i,
             "NOISE_SIGMA": 20.0, "ELONGATION_MEAN": 1.05, "SKY_LEVEL": 1500.0} for i in range(n)]


class _FakeCheck:
    def __init__(self, trails: set[str], z: dict[str, float] | None = None) -> None:
        self.trails, self.z, self.calls = trails, z or {}, []

    def __call__(self, path: str, fwhm: float, ref: str | None = None, ref_fwhm: float | None = None):
        self.refs = getattr(self, "refs", []) + [ref]
        self.calls.append(path)
        has = path in self.trails
        z = self.z.get(path, 30.0 if has else 7.0)
        return td.TrailCheck(has, z, td.TRAIL_Z_MIN, 45.0 if has else 0.0, 10, 1.0 if has else 0.1, 8, 1000)


def test_best_frame_with_a_trail_is_skipped_for_the_next_clean_one() -> None:
    chk = _FakeCheck({"/a/f000.fits", "/a/f001.fits"})
    g = fq.evaluate_frame_qc(_rows(), k=3.0, trail_check=chk)["L|60|2"]
    assert g.ranked_paths[:3] == ["/a/f000.fits", "/a/f001.fits", "/a/f002.fits"]
    assert g.best_path == "/a/f002.fits" and g.best_trail_free is True
    assert chk.calls == ["/a/f000.fits", "/a/f001.fits", "/a/f002.fits"], "walks in rank order, stops at clean"
    assert [c["has_trail"] for c in g.trail_checks] == [True, True, False]
    assert g.to_dict()["trail_checks"][0]["path"] == "/a/f000.fits"


def test_every_frame_with_a_trail_takes_the_weakest_and_says_so() -> None:
    paths = [f"/a/f{i:03d}.fits" for i in range(6)]
    chk = _FakeCheck(set(paths), z={p: 40.0 - i for i, p in enumerate(paths)} | {"/a/f003.fits": 11.0})
    g = fq.evaluate_frame_qc(_rows(), k=3.0, trail_check=chk)["L|60|2"]
    assert g.best_path == "/a/f003.fits" and g.best_trail_free is False


def test_without_a_trail_test_the_ranking_is_unchanged() -> None:
    g = fq.evaluate_frame_qc(_rows(), k=3.0)["L|60|2"]
    assert g.best_path == "/a/f000.fits" and g.best_trail_free is None and g.trail_checks == []
    assert g.ranked_paths[0] == g.best_path


def test_unreadable_images_keep_the_rank_order() -> None:
    g = fq.evaluate_frame_qc(_rows(3), k=3.0, trail_check=lambda p, f, r, rf: None)["L|60|2"]
    assert g.best_path == "/a/f000.fits" and all(not c["tested"] for c in g.trail_checks)


def test_run_frame_qc_reads_the_mapped_image(tmp_path) -> None:
    from astropy.io import fits

    rng = np.random.default_rng(7)
    good = tmp_path / "clean.fits"
    bad = tmp_path / "trail.fits"
    fits.writeto(good, _frame(rng).astype(np.float32))
    fits.writeto(bad, _add_trail(_frame(rng), amp=50.0, angle_deg=30.0).astype(np.float32))
    check = fq.fits_trail_check(lambda p: str(bad) if p.endswith("000.fits") else str(good))
    g = fq.evaluate_frame_qc(_rows(3), k=3.0, trail_check=check)["L|60|2"]
    assert g.best_path == "/a/f001.fits"
    assert g.trail_checks[0]["has_trail"] is True and g.trail_checks[1]["has_trail"] is False


# ------------------------------------------------------------------ fixed features vs transient trails
def test_a_line_fixed_on_the_sky_is_not_a_trail_when_the_reference_has_it_too() -> None:
    """Milan's Brno r frames: one line at ~50 deg with Z 12-13 in many frames (a sky feature)."""
    rng = np.random.default_rng(21)
    sky_line = dict(amp=1.0 * 25.0, angle_deg=50.0, through=(400.0, 300.0))
    cand = _add_trail(_frame(rng), **sky_line)
    ref = td.line_plane(_add_trail(_frame(np.random.default_rng(22)), **sky_line), FWHM)
    alone = td.detect_trail(cand, FWHM)
    with_ref = td.detect_trail(cand, FWHM, reference=ref)
    assert alone.has_trail, "the feature alone looks like a trail"
    assert not with_ref.has_trail, with_ref.to_dict()


def test_a_trail_is_still_found_against_a_reference() -> None:
    rng = np.random.default_rng(23)
    ref = td.line_plane(_frame(np.random.default_rng(24)), FWHM)
    r = td.detect_trail(_add_trail(_frame(rng), amp=40.0, angle_deg=120.0), FWHM, reference=ref)
    assert r.has_trail


def test_reference_is_frame_qc_reference_or_next_ranked() -> None:
    chk = _FakeCheck({"/a/f000.fits"})
    g = fq.evaluate_frame_qc(_rows(), k=3.0, trail_check=chk)["L|60|2"]
    assert g.best_path == "/a/f001.fits"
    assert all(r is not None for r in chk.refs) and chk.refs[0] != "/a/f000.fits"


def test_run_frame_qc_reference_plane_with_another_fwhm_is_rebinned(tmp_path) -> None:
    from astropy.io import fits

    rng = np.random.default_rng(31)
    paths = {}
    for i in range(3):
        p = tmp_path / f"f{i}.fits"
        img = _frame(rng)
        if i == 0:
            img = _add_trail(img, amp=50.0, angle_deg=30.0)
        fits.writeto(p, img.astype(np.float32))
        paths[f"/a/f{i:03d}.fits"] = str(p)
    rows = _rows(3)
    rows[2]["FWHM"] = 3.9  # another bin size than the candidates: the reference plane is rebuilt
    g = fq.evaluate_frame_qc(rows, k=3.0, trail_check=fq.fits_trail_check(paths.get))["L|60|2"]
    assert g.best_path == "/a/f001.fits"
