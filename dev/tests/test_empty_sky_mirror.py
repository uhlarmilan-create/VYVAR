"""Empty-sky audit on the background mirror: uncatalogued faint stars are not false accepts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

from dao_gaia_calibration import mirror_about_background, run_empty_sky_audits  # noqa: E402


def _field(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = 600, 800
    yy, xx = np.mgrid[0:h, 0:w]
    img = 1000.0 + 0.05 * xx + rng.normal(0.0, 10.0, (h, w))
    sig = 3.0 / 2.3548
    # 600 "uncatalogued" faint stars (peak 6-15 sigma)
    pos = np.column_stack([rng.uniform(20, w - 20, 600), rng.uniform(20, h - 20, 600)])
    for (x0, y0), a in zip(pos, rng.uniform(60, 150, 600)):
        x0i, y0i = int(x0), int(y0)
        sl = (slice(y0i - 8, y0i + 9), slice(x0i - 8, x0i + 9))
        img[sl] += a * np.exp(-((xx[sl] - x0) ** 2 + (yy[sl] - y0) ** 2) / (2 * sig * sig))
    return img, pos


def _audit(img: np.ndarray, empty: pd.DataFrame):
    h, w = img.shape
    return run_empty_sky_audits(
        empty, img, wpx=w, h=h, fwhm_px=3.0, pass2_sigma=4.0, pass2_center_tol_px=1.5,
        seed_centroid_max_px=1.5, seed_snr_min=4.0,
    )


def test_mirror_removes_sources_and_keeps_background():
    img, pos = _field()
    mir, box = mirror_about_background(img, 3.0)
    assert box == 30
    x, y = int(pos[0, 0]), int(pos[0, 1])
    assert mir[y, x] < img[y, x] - 50.0  # star became a hole
    assert abs(float(np.median(mir)) - float(np.median(img))) < 2.0


def test_uncatalogued_stars_do_not_count_as_false_accepts():
    img, pos = _field()
    rng = np.random.default_rng(1)
    # positions placed ON the uncatalogued stars: the old audit counted these as false
    on_stars = pd.DataFrame({"x": pos[:300, 0], "y": pos[:300, 1]})
    a = _audit(img, on_stars)
    assert a.method == "mirror_about_background"
    assert a.seed_rate <= 0.01 and a.inv_seed == "PASS"
    rand = pd.DataFrame({"x": rng.uniform(20, 780, 1500), "y": rng.uniform(20, 580, 1500)})
    b = _audit(img, rand)
    assert b.seed_rate <= 0.01 and b.pass2_rate <= 0.01
