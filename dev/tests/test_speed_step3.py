"""CODE-REVIEW-2026-10 step 3 (speed): every faster path gives exactly the old result."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

import pipeline  # noqa: F401  app import order


def _shifted_header(shape: tuple[int, int], dx: float, rot: float = 0.0) -> fits.Header:
    h = fits.Header()
    h["NAXIS1"], h["NAXIS2"] = shape[1], shape[0]
    c, s = math.cos(rot), math.sin(rot)
    for k, v in enumerate((c, -s, dx, s, c, 3.3, 0.0, 0.0, 1.0)):
        h[f"VY_AM{k}"] = v
    h["VY_FWHM"] = 3.1
    return h


def _old_footprint(header, shape):
    mat = np.array([float(header[f"VY_AM{k}"]) for k in range(9)], dtype=np.float64).reshape(3, 3)
    h, w = int(shape[0]), int(shape[1])
    wr, hr = int(header.get("NAXIS1", w)), int(header.get("NAXIS2", h))
    corners = np.array([[0, 0, 1], [wr - 1, 0, 1], [wr - 1, hr - 1, 1], [0, hr - 1, 1]], dtype=np.float64).T
    q = mat @ corners
    quad = (q[:2] / q[2]).T
    yy, xx = np.mgrid[0:h, 0:w]
    area = 0.5 * np.sum(quad[:, 0] * np.roll(quad[:, 1], -1) - np.roll(quad[:, 0], -1) * quad[:, 1])
    sgn = 1.0 if area > 0 else -1.0
    inside = np.ones((h, w), dtype=bool)
    for a, b in zip(quad, np.roll(quad, -1, axis=0), strict=False):
        ex, ey = b - a
        inside &= sgn * (ex * (yy - a[1]) - ey * (xx - a[0])) / math.hypot(ex, ey) >= 1.0
    return inside


@pytest.mark.parametrize("rot", [0.0, 0.013, -0.2])
def test_aligned_footprint_same_as_full_grid(rot: float) -> None:
    from frame_footprint import aligned_footprint

    hdr = _shifted_header((97, 143), 17.37, rot)
    assert np.array_equal(aligned_footprint(hdr, (97, 143)), _old_footprint(hdr, (97, 143)))


def _frames(tmp_path: Path, n: int = 5) -> list[str]:
    rng = np.random.default_rng(7)
    shape = (90, 130)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    ids = [str(1000000000000 + i) for i in range(12)]
    pos = rng.uniform(15, [115, 75], size=(len(ids), 2))
    for k in range(n):
        img = rng.normal(200.0, 3.0, shape)
        for x0, y0 in pos:
            img += 4000.0 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 1.3**2))
        if k == 2:
            img[5:9, 40:44] = np.nan  # a NaN patch: the filled-image path
        hdr = _shifted_header(shape, 4.0 + k)
        fits.writeto(tmp_path / f"fr_{k}.fits", img.astype(np.float32), hdr)
        pd.DataFrame({"catalog_id": ids, "x": pos[:, 0].round(), "y": pos[:, 1].round()}).to_csv(
            tmp_path / f"proc_fr_{k}.csv", index=False)
    return ids


def test_night_grid_pool_equals_serial_and_old_measure(tmp_path: Path) -> None:
    from aperture_pertarget import measure_flux_on_f_grid, measure_night_grid

    ids = _frames(tmp_path)
    kw = dict(frames_dir=tmp_path, catalog_ids=ids, f_grid=[1.0, 1.5, 2.0], fwhm_night_px=3.0,
              scale_by_cid={ids[0]: 1.3, ids[1]: 1.3})
    g1 = measure_night_grid(**kw, workers=1)
    g2 = measure_night_grid(**kw, workers=2)
    assert g1.frames == g2.frames and g1.sky_pp == g2.sky_pp and g1.edge_by_frame == g2.edge_by_frame
    assert g1.fwhm_by_frame == g2.fwhm_by_frame and g1.n_frames_measured == 5
    # one frame against measure_flux_on_f_grid without a prepared image (the old per-call path)
    from frame_footprint import aligned_footprint

    with fits.open(tmp_path / "fr_2.fits") as h:
        img = np.asarray(h[0].data, dtype=float)
        cov = aligned_footprint(h[0].header, img.shape)
    df = pd.read_csv(tmp_path / "proc_fr_2.csv", dtype={"catalog_id": str})
    xy = df[["x", "y"]].to_numpy(float)[2:]
    f, s, e = measure_flux_on_f_grid(img, xy, f_grid=[1.0, 1.5, 2.0], fwhm_px=3.1, annulus_inner_fwhm=2.7,
                                     annulus_outer_fwhm=5.2, covered=cov)
    for j, cid in enumerate(df["catalog_id"].tolist()[2:]):
        for k in (1.0, 1.5, 2.0):
            a, b = g1.frames["fr_2"][cid][k], f[k][j]
            assert (a == b) or (math.isnan(a) and math.isnan(b))


def test_csv_lookup_keys_are_catalog_id_only() -> None:
    """GAIA-ID-STRICT-2026-10 (G04-17): the row index keys on catalog_id; ``name`` never gives an identity."""
    from aperture_photometry import _build_csv_lookup

    df = pd.DataFrame({
        "catalog_id": ["4000000000000000001", "4000000000000000002", "", "4000000000000000001", "DET_5"],
        "name": ["4000000000000000001", "DET_2", "4000000000000000003", "4000000000000000009", ""],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    id_map, _xy = _build_csv_lookup(df, "catalog_id")
    assert set(id_map) == {"4000000000000000001", "4000000000000000002"}
    assert id_map["4000000000000000001"].equals(df.iloc[0])  # first row wins
    assert "4000000000000000003" not in id_map and "4000000000000000009" not in id_map


def test_stamp_metrics_window_equals_full_image() -> None:
    from lc_outlier import _stamp_metrics

    rng = np.random.default_rng(3)
    img = rng.normal(100.0, 5.0, (80, 90))
    for x0, y0 in ((40.3, 39.7), (3.2, 4.9), (88.6, 77.1), (14.5, 15.5)):
        m = _stamp_metrics(img, x0, y0)
        yy2, xx2 = np.ogrid[0:80, 0:90]
        rr = np.hypot(xx2 - x0, yy2 - y0)
        ann = img[(rr >= 8.0) & (rr <= 15.0)]
        assert m["annulus_sky_sigma"] == float(np.std(ann))


def test_deferred_field_maps_pool_equals_serial(tmp_path: Path) -> None:
    from field_map import DeferredTargetFieldMaps

    rng = np.random.default_rng(5)
    data = rng.normal(100.0, 10.0, (120, 160))
    jobs = []
    for i in range(3):
        t = pd.Series({"catalog_id": f"77{i}", "vsx_name": f"V{i}", "x": 30.0 + 30 * i, "y": 60.0})
        c = pd.DataFrame({"catalog_id": ["1", "2"], "x": [20.0, 120.0], "y": [20.0, 90.0]})
        jobs.append((t, c))
    outs = {}
    for w in (1, 2):
        d = tmp_path / f"w{w}"
        fm = DeferredTargetFieldMaps(data, None, workers=w)
        for i, (t, c) in enumerate(jobs):
            fm.save(d / f"m{i}.jpg", t, c)
        fm.close()
        outs[w] = [(d / f"m{i}.jpg").read_bytes() for i in range(3)]
    assert outs[1] == outs[2]
