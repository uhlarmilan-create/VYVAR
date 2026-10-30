"""TRANSIENT-2026-10: new sources in the night's frames, minor bodies from MPC orbits."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src_py", ROOT / "dev" / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import minor_bodies as MB  # noqa: E402
import transient_field as TF  # noqa: E402

PALLAS = ("00002    4.12  0.15 K2669 254.24963  310.96993  172.88661   34.93279  0.2307001  0.21383960   2.7695590  0 "
          "E2026-U44  9075 124 1804-2026 0.77 M-c 28k MPCORBFIT  4000      (2) Pallas             20261024")
HALE_BOPP = ("    C1995O1   1997 04  1.1332  0.926190  0.994895  130.7330  281.8108   89.7329  20261023  -2.0  4.0  "
             "C/1995 O1 (Hale-Bopp)                                    MPEC 2026-K60")


def test_parse_mpc_lines() -> None:
    mp = MB.parse_mpcorb(["-" * 160, PALLAS])
    r = mp.iloc[0]
    assert r["name"] == "(2) Pallas" and abs(r["H"] - 4.12) < 1e-9 and abs(r["a"] - 2.769559) < 1e-6
    assert abs(r["epoch_jd"] - MB._jd(2026, 6, 9.0)) < 1e-9  # packed K2669 = 2026 Jun 9
    cm = MB.parse_cometels([HALE_BOPP])
    c = cm.iloc[0]
    assert c["name"].startswith("C/1995 O1") and abs(c["q"] - 0.92619) < 1e-9 and abs(c["G"] - 4.0) < 1e-9


def _ellipse(a: float, e: float, M: float) -> pd.DataFrame:
    return pd.DataFrame([{"packed": "X", "name": "x", "H": 10.0, "G": 0.15, "epoch_jd": 2461000.5, "M": M, "peri": 30.0,
                          "node": 40.0, "incl": 10.0, "e": e, "n": math.degrees(MB.K_GAUSS / a**1.5), "a": a}])


def test_two_body_radius_and_period() -> None:
    df = _ellipse(2.5, 0.2, 0.0)
    p0 = MB.heliocentric_minor_planets(df, 2461000.5)
    assert abs(np.linalg.norm(p0) - 2.5 * 0.8) < 1e-9  # perihelion
    period = 2 * math.pi / MB.K_GAUSS * 2.5**1.5
    p1 = MB.heliocentric_minor_planets(df, 2461000.5 + period)
    assert np.allclose(p0, p1, atol=1e-9)


def test_comet_conics_meet_at_e_one() -> None:
    """Elliptic, parabolic and hyperbolic orbits with e -> 1 give the same position."""
    rows = [{"name": n, "tp_jd": 2461000.5, "q": 1.0, "e": e, "peri": 20.0, "node": 50.0, "incl": 30.0, "H": 10.0, "G": 4.0}
            for n, e in (("ell", 1 - 2e-6), ("par", 1.0), ("hyp", 1 + 2e-6))]
    p = MB.heliocentric_comets(pd.DataFrame(rows), 2461000.5 + 40.0)
    assert np.max(np.abs(p - p[1])) < 1e-4


def test_hg_magnitude_at_opposition() -> None:
    v = MB._hg_mag(np.array([10.0]), np.array([0.15]), np.array([2.0]), np.array([1.0]), 1.0)
    assert abs(float(v[0]) - (10.0 + 5 * math.log10(2.0))) < 1e-6  # phase 0


def test_onset_kinds() -> None:
    n = 60
    snr = np.r_[np.full(30, 0.5), np.full(30, 12.0)]
    assert TF.onset_kind(snr) == ("appeared", 30)
    assert TF.onset_kind(snr[::-1])[0] == "faded"
    assert TF.onset_kind(np.full(n, 9.0))[0] == "present"
    one = np.full(n, 0.5)
    one[20:22] = 30.0  # two frames only: a cosmic ray or a satellite
    assert TF.onset_kind(one) == ("", -1)


def test_link_and_classify_stationary_and_moving() -> None:
    t = np.arange(20, dtype=float) / 100.0  # days
    stat = pd.DataFrame({"t": np.arange(20), "x": 100 + 0.1 * np.sin(np.arange(20)), "y": 200.0})
    mov = pd.DataFrame({"t": np.arange(20), "x": 500 + 1.5 * np.arange(20), "y": 300 + 0.5 * np.arange(20)})
    un = pd.concat([stat, mov], ignore_index=True).assign(flux=1.0, sharp=0.5)
    lab = TF.link_detections(un, 3.0)
    s, m = TF.classify_groups(un, lab, t, fwhm_med=3.0)
    assert len(s) == 1 and abs(s[0]["x"] - 100) < 0.2
    assert len(m) == 1 and m[0]["n_frames"] == 20 and m[0]["rate_px_day"] > 100


def test_detector_fixed_track_is_rejected() -> None:
    """A hot pixel: fixed on the detector, so it 'moves' on the aligned grid with the pointing drift."""
    mats = [np.array([[1, 0, 2.0 * k], [0, 1, 1.0 * k], [0, 0, 1]]) for k in range(10)]
    un = pd.DataFrame({"t": np.arange(10), "x": [50 + 2.0 * k for k in range(10)], "y": [60 + 1.0 * k for k in range(10)]})
    lab = np.zeros(10, int)
    assert TF._detector_fixed(un, lab, {"_g": 0}, mats, fwhm=3.0)
    sky = pd.DataFrame({"t": np.arange(10), "x": 50.0, "y": 60.0})  # fixed on the sky
    assert not TF._detector_fixed(sky, lab, {"_g": 0}, mats, fwhm=3.0)


def test_galaxy_core_vs_inside(tmp_path: Path) -> None:
    (tmp_path / TF.OPENNGC_NAME).write_text(
        "Name;Type;RA;Dec;MajAx\nNGC5689;G;14:35:29.69;+48:44:29.9;3.66\nNGC0000;**;00:00:00;+00:00:00;\n", encoding="ascii")
    gal = TF.load_galaxies(tmp_path)
    assert list(gal["Name"]) == ["NGC5689"]
    ra, dec = float(gal.at[0, "ra"]), float(gal.at[0, "dec"])
    name, d, semi = TF.galaxy_near(gal, ra, dec + 1.0 / 3600)
    assert name == "NGC5689" and d < 1.5 and abs(semi - 109.8) < 0.1
    name2, d2, _ = TF.galaxy_near(gal, ra, dec + 60.0 / 3600)
    assert name2 == "NGC5689" and 59 < d2 < 61  # inside the disk: a possible supernova
    assert TF.galaxy_near(gal, ra + 1.0, dec)[0] == ""


def test_injection_transient_signals() -> None:
    from injection_recovery import OFF_MAG, Signal

    t = np.array([0.0, 1.0, 1.5, 2.0, 3.0])
    assert np.all(Signal("step", t0=1.0).dm(t) == [OFF_MAG, 0, 0, 0, 0])
    r = Signal("rise", amp=3.0, t0=1.0, width_d=1.0).dm(t)
    assert r[0] == OFF_MAG and abs(r[1] - 3.0) < 1e-12 and abs(r[2] - 1.5) < 1e-12 and r[3] == 0.0
    f = Signal("flare", t0=1.0, width_d=1.0).dm(t)
    assert f[1] == 0.0 and abs(f[3] - 1.0857362) < 1e-6


def test_dark_hot_pixel_binned_and_shifted() -> None:
    """A hot pixel of a 2x2-larger master dark, found through the inverse alignment of the frame."""
    rng = np.random.default_rng(1)
    dk = rng.normal(100.0, 2.0, (200, 300))
    dk[81, 121] = 5000.0  # binned pixel (40, 60)
    mats = [np.array([[1, 0, 5.0], [0, 1, -3.0], [0, 0, 1]])]  # aligned = detector + (5, -3)
    d = {"data": dk, "raw_shape": (100, 150)}
    assert TF.dark_hot_pixel(d, mats, np.array([65.0, 37.0]), 0)
    assert not TF.dark_hot_pixel(d, mats, np.array([60.0, 40.0]), 0)
    assert not TF.dark_hot_pixel(None, mats, np.array([65.0, 37.0]), 0)


def test_shape_text_shown_to_the_user() -> None:
    """G10-44 (Milan 2026-10-29): extended / fwhm_ratio are shown in the UI table and the PDF report."""
    import math
    from pathlib import Path

    from transient_field import EXTENDED_RATIO, shape_text

    assert shape_text(True, 1.83) == "extended (FWHM x1.83)"
    assert shape_text("False", 1.02) == "point (FWHM x1.02)"
    assert shape_text(False, math.nan) == "-"
    assert EXTENDED_RATIO == 1.5
    src = Path(__file__).resolve().parents[2] / "src_py"
    for name in ("ui_variability.py", "photometry_report.py"):
        assert "shape_text(" in (src / name).read_text(encoding="utf-8"), name
