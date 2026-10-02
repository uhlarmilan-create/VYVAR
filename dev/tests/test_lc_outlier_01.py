"""LC-OUTLIER-01 tests: flag with evidence; never delete; protect flares."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lc_outlier import (  # noqa: E402
    FLAG_ARTIFACT,
    FLAG_FRAME_QC,
    FLAG_HIGH_ERR,
    FLAG_NORMAL,
    FLAG_SPIKE_UNCONFIRMED,
    ImageEvidence,
    assign_lc_flags,
    evaluate_image_evidence,
    export_keep_mask,
    frame_qc_mask_from_night_table,
    high_err_mask,
    isolated_spike_mask,
)


def _gauss_stamp(size: int = 31, *, peak: float = 1000.0, fwhm: float = 4.0, elong: float = 1.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = (size - 1) / 2.0
    sig_x = fwhm / 2.355
    sig_y = (fwhm * elong) / 2.355
    g = peak * np.exp(-0.5 * (((xx - cx) / sig_x) ** 2 + ((yy - cy) / sig_y) ** 2))
    return g.astype(np.float64)


def test_t1_spike_with_trail_is_artifact() -> None:
    """T1: single spike + elongated stamp evidence -> artifact."""
    n = 40
    rng = np.random.default_rng(1)
    mag = 12.0 + 0.02 * rng.normal(size=n)
    err = np.full(n, 0.02)
    bjd = 2460000.0 + np.arange(n) * 0.001
    mag[20] = 11.5  # ~0.5 mag bright spike
    # Build a frame with a trailed target vs round peers.
    canvas = np.full((200, 200), 100.0, dtype=np.float64)
    # Peers: round Gaussians
    peers = []
    for i, (px, py) in enumerate([(40, 40), (80, 40), (120, 40), (160, 40), (40, 80), (80, 80), (120, 80), (160, 80), (40, 120), (80, 120)]):
        stamp = _gauss_stamp(25, peak=800.0, fwhm=4.0, elong=1.05)
        canvas[py - 12 : py + 13, px - 12 : px + 13] += stamp
        peers.append((float(px), float(py)))
    # Target with trail (high elongation)
    tx, ty = 100.0, 150.0
    trail = _gauss_stamp(25, peak=800.0, fwhm=4.0, elong=3.5)
    canvas[int(ty) - 12 : int(ty) + 13, int(tx) - 12 : int(tx) + 13] += trail

    def _ev(i: int) -> ImageEvidence | None:
        if i != 20:
            return None
        return evaluate_image_evidence(canvas, tx, ty, peer_xy=peers, n_sigma=5.0)

    res = assign_lc_flags(mag, err, bjd, evidence_for_index=_ev, n_sigma=5.0, adjacent_sigma=3.0)
    assert res.flags[20] == FLAG_ARTIFACT
    assert "artifact:" in res.reasons[20]
    # Photometry unchanged by design (caller responsibility); flags only.
    assert res.n_artifact == 1


def test_t2_spike_clean_stamp_is_unconfirmed() -> None:
    """T2: single spike, clean stamp -> spike_unconfirmed; export keeps it."""
    n = 40
    rng = np.random.default_rng(2)
    mag = 12.0 + 0.02 * rng.normal(size=n)
    err = np.full(n, 0.02)
    bjd = 2460000.0 + np.arange(n) * 0.001
    mag[15] = 11.4
    canvas = np.full((180, 180), 50.0, dtype=np.float64)
    peers = []
    for px, py in [(40, 40), (80, 40), (120, 40), (40, 80), (80, 80), (120, 80), (40, 120), (80, 120), (120, 120), (60, 60)]:
        stamp = _gauss_stamp(21, peak=700.0, fwhm=3.5, elong=1.05)
        canvas[py - 10 : py + 11, px - 10 : px + 11] += stamp
        peers.append((float(px), float(py)))
    tx, ty = 100.0, 100.0
    canvas[int(ty) - 10 : int(ty) + 11, int(tx) - 10 : int(tx) + 11] += _gauss_stamp(
        21, peak=700.0, fwhm=3.5, elong=1.05
    )

    def _ev(i: int) -> ImageEvidence | None:
        if i != 15:
            return None
        return evaluate_image_evidence(canvas, tx, ty, peer_xy=peers, n_sigma=5.0)

    res = assign_lc_flags(mag, err, bjd, evidence_for_index=_ev)
    assert res.flags[15] == FLAG_SPIKE_UNCONFIRMED
    keep = export_keep_mask(res.flags)
    assert bool(keep[15]) is True


def test_t3_flare_profile_stays_normal() -> None:
    """T3: fast rise + 4+ elevated epochs -> all normal (not isolated)."""
    n = 50
    mag = np.full(n, 13.0)
    err = np.full(n, 0.02)
    bjd = 2460000.0 + np.arange(n) * 0.001
    # Flare: epochs 20..24 elevated (bright = smaller mag)
    mag[20:25] = 12.4
    spike, _, _ = isolated_spike_mask(mag, err, bjd, n_sigma=5.0, adjacent_sigma=3.0)
    assert not bool(spike.any())
    res = assign_lc_flags(mag, err, bjd)
    assert all(f == FLAG_NORMAL for f in res.flags)


def test_t4_eclipse_dip_stays_normal() -> None:
    """T4: eclipse dip (fainter = larger mag) run stays normal."""
    n = 60
    mag = np.full(n, 12.5)
    err = np.full(n, 0.015)
    bjd = 2460000.0 + np.arange(n) * 0.001
    mag[30:38] = 13.2  # eclipse
    res = assign_lc_flags(mag, err, bjd)
    assert all(f == FLAG_NORMAL for f in res.flags[30:38])


def test_t5_frame_qc_propagates_to_all_stars() -> None:
    """T5: night-outlier FWHM frame -> frame_qc for every star on that frame."""
    df = pd.DataFrame(
        {
            "frame_key": [f"f{i:03d}" for i in range(30)],
            "fwhm": [5.5] * 29 + [12.0],
            "elongation": [1.1] * 30,
            "sky_level": [1000.0] * 30,
        }
    )
    reasons = frame_qc_mask_from_night_table(df, n_sigma=5.0)
    assert reasons["f029"]  # non-empty reason
    n = 10
    mag = np.full(n, 12.0)
    err = np.full(n, 0.02)
    bjd = 2460000.0 + np.arange(n) * 0.001
    src = ["proc_f005.csv"] * 9 + ["proc_f029.csv"]
    res = assign_lc_flags(
        mag,
        err,
        bjd,
        source_files=src,
        frame_qc_reasons=reasons,
    )
    assert res.flags[-1] == FLAG_FRAME_QC
    assert res.flags[0] == FLAG_NORMAL


def test_t6_export_drops_artifact_keeps_spike_unconfirmed() -> None:
    """T6: export filter drops artifact/frame_qc/high_err; keeps spike_unconfirmed."""
    flags = [
        FLAG_NORMAL,
        FLAG_ARTIFACT,
        FLAG_FRAME_QC,
        FLAG_SPIKE_UNCONFIRMED,
        "saturated",
        FLAG_HIGH_ERR,
    ]
    keep = export_keep_mask(flags)
    assert keep.tolist() == [True, False, False, True, False, False]

    from export_reports import _select_export_lc_rows

    lc = pd.DataFrame(
        {
            "bjd": [2460000.1, 2460000.2, 2460000.3, 2460000.4, 2460000.5],
            "mag_calib": [12.1, 11.5, 12.0, 11.6, 13.7],
            "flag": [
                FLAG_NORMAL,
                FLAG_ARTIFACT,
                FLAG_FRAME_QC,
                FLAG_SPIKE_UNCONFIRMED,
                FLAG_HIGH_ERR,
            ],
        }
    )
    out = _select_export_lc_rows(lc)
    assert set(out["flag"].astype(str)) == {FLAG_NORMAL, FLAG_SPIKE_UNCONFIRMED}


def test_high_err_inflated_error_flagged() -> None:
    """LC-FLAG-ERR-01: huge err + large deviation -> high_err (spike test silent)."""
    n = 40
    mag = np.full(n, 12.0)
    err = np.full(n, 0.05)
    bjd = 2460000.0 + np.arange(n) * 0.001
    # Frame with exploded err hides spike residual; high_err must catch it.
    mag[20] = 13.5
    err[20] = 0.80
    err_photon = err.copy()
    err_sem = np.full(n, 0.01)
    res = assign_lc_flags(
        mag,
        err,
        bjd,
        err_photon=err_photon,
        err_sem_rel=err_sem,
        high_err_n_sigma=5.0,
    )
    assert res.flags[20] == FLAG_HIGH_ERR
    assert "high_err:" in res.reasons[20]
    assert "err_photon=" in res.reasons[20]
    assert res.n_high_err >= 1
    # Spike test alone would not fire (r = (m-S)/err small).
    spike, _, _ = isolated_spike_mask(mag, err, bjd, n_sigma=5.0)
    assert not bool(spike[20])


def test_high_err_flare_still_normal() -> None:
    """Flare run stays normal; high_err does not fire on flat err."""
    n = 50
    mag = np.full(n, 13.0)
    err = np.full(n, 0.02)
    bjd = 2460000.0 + np.arange(n) * 0.001
    mag[20:25] = 12.4
    res = assign_lc_flags(mag, err, bjd, high_err_n_sigma=5.0)
    assert all(f == FLAG_NORMAL for f in res.flags)
    assert res.n_high_err == 0


def test_high_err_precedence() -> None:
    """Precedence: artifact > frame_qc > high_err > spike_unconfirmed."""
    n = 20
    mag = np.full(n, 12.0)
    err = np.full(n, 0.05)
    bjd = 2460000.0 + np.arange(n) * 0.001
    # i=5: high err (spike silent when err huge)
    mag[5] = 13.5
    err[5] = 0.9
    # i=10: spike_unconfirmed candidate (small err, isolated spike)
    mag[10] = 11.4
    # i=15: frame_qc
    src = [f"proc_f{i:03d}.csv" for i in range(n)]
    fq = {"f015": "fwhm"}
    res = assign_lc_flags(
        mag,
        err,
        bjd,
        source_files=src,
        frame_qc_reasons=fq,
        err_photon=err,
        high_err_n_sigma=5.0,
    )
    assert res.flags[15] == FLAG_FRAME_QC
    assert res.flags[5] == FLAG_HIGH_ERR
    assert res.flags[10] == FLAG_SPIKE_UNCONFIRMED
    # artifact beats high_err when spike fires AND evidence fires (err still
    # above MAD threshold but small enough that |r| > n_sigma).
    mag2 = mag.copy()
    err2 = err.copy()
    mag2[5] = 11.0  # 1 mag bright
    err2[5] = 0.12  # high_err vs median 0.05; r = 1/0.12 ~ 8.3 > 5

    def _ev(i: int) -> ImageEvidence | None:
        if i != 5:
            return None
        return ImageEvidence(fired=True, reasons=["annulus_sky_sigma_z=9.0"])

    res2 = assign_lc_flags(
        mag2,
        err2,
        bjd,
        evidence_for_index=_ev,
        err_photon=err2,
        high_err_n_sigma=5.0,
    )
    assert res2.flags[5] == FLAG_ARTIFACT
    assert high_err_mask(err2, n_sigma=5.0)[0][5]


def test_pre_vsx_skip_no_longer_blanks_known_variables() -> None:
    """Regression: VSX-known vars must still get spike flags (2a8355b skip retired)."""
    n = 30
    mag = np.full(n, 12.0)
    err = np.full(n, 0.02)
    bjd = 2460000.0 + np.arange(n) * 0.001
    mag[10] = 11.4
    res = assign_lc_flags(mag, err, bjd)
    assert res.flags[10] in (FLAG_SPIKE_UNCONFIRMED, FLAG_ARTIFACT)
