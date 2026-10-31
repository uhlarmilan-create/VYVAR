"""AP-FOOTPRINT-2026-10: aperture photometry on the aligned frame's data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from aperture_pertarget import ApertureGridNight, apply_grid_fluxes_to_frames, measure_flux_on_f_grid
from frame_footprint import aligned_footprint, has_alignment_transform


def _shifted_header(shape: tuple[int, int], dx: float) -> fits.Header:
    h = fits.Header()
    h["NAXIS1"], h["NAXIS2"] = shape[1], shape[0]
    for k, v in enumerate((1.0, 0.0, dx, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)):
        h[f"VY_AM{k}"] = v
    return h


def _frame(shape=(120, 200), sky=100.0, fill=60.0, dx=40.0):
    rng = np.random.default_rng(2)
    img = rng.normal(sky, 1.0, shape)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for x0 in (52.0, 150.0):
        img += 5000.0 / (2 * np.pi * 1.7**2) * np.exp(-0.5 * (((xx - x0) / 1.7) ** 2 + ((yy - 60.0) / 1.7) ** 2))
    img[:, : int(dx)] = fill  # the aligner's constant fill left of x = dx
    return img, _shifted_header(shape, dx)


def test_sky_ignores_the_fill_and_edge_marks_annulus_mostly_off() -> None:
    img, hdr = _frame()
    cov = aligned_footprint(hdr, img.shape)
    assert has_alignment_transform(hdr) and not cov[:, :41].any()
    xy = np.array([[52.0, 60.0], [150.0, 60.0], [44.0, 60.0]])
    kw = dict(f_grid=[1.0], fwhm_px=4.0, annulus_inner_fwhm=2.7, annulus_outer_fwhm=5.2)
    f0, s0, e0 = measure_flux_on_f_grid(img, xy, **kw)
    f1, s1, e1 = measure_flux_on_f_grid(img, xy, covered=cov, **kw)
    # star at x = 52: its annulus (10.8-20.8 px) reaches the fill at x < 41
    assert s0[1.0][0] < s1[1.0][0] - 0.1  # plain median pulled down by the 60 ADU fill
    assert s1[1.0][0] == pytest.approx(100.0, abs=0.3)
    assert s1[1.0][1] == pytest.approx(s0[1.0][1])  # far from the fill: unchanged
    assert list(e0) == [False, False, False]
    assert list(e1) == [False, False, True]  # x = 44: aperture (4 px) touches the fill


def test_edge_star_gets_edge_fail_in_the_light_curve() -> None:
    night = ApertureGridNight(f_grid=[1.0], fwhm_night_px=4.0)
    night.frames = {"F1": {"t": {1.0: 1000.0}, "c": {1.0: 2000.0}}}
    night.fwhm_by_frame = {"F1": 4.0}
    night.edge_by_frame = {"F1": {"t"}}
    night.sky_pp = {"F1": {"t": {1.0: 100.0}, "c": {1.0: 100.0}}}
    night.sky_n = {"F1": {"t": 500, "c": 500}}
    df = pd.DataFrame({"catalog_id": ["t", "c"], "source_file": ["proc_F1.csv"] * 2, "flag": ["normal"] * 2, "mag_inst": [0.0, 0.0]})
    out = apply_grid_fluxes_to_frames(df, night, catalog_ids=["t", "c"], f_star=1.0, gain=1.0, read_noise=5.0)
    t = out.set_index("catalog_id")
    assert np.isnan(t.loc["t", "mag_inst"]) and t.loc["t", "flag"] == "edge_fail" and bool(t.loc["t", "edge_fail"])
    assert t.loc["c", "mag_inst"] == pytest.approx(-2.5 * np.log10(2000.0))
