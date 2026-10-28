"""TESS-2026-10 period/localisation core; neighbour-light model vs pixel photometry."""

from __future__ import annotations

from config import AppConfig

import types

import numpy as np
from astropy.wcs import WCS

import photometry_core  # noqa: F401  (import order of the photometry package)
import tess_analysis as ta


def _tpf(sector: int, t0: float, var: str, rng: np.random.Generator):
    t = np.arange(t0, t0 + 27, 1800 / 86400.0)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [320.0, 51.8]
    w.wcs.crpix = [8, 8]
    w.wcs.cdelt = [-21 / 3600, 21 / 3600]
    srcs = [("T", 320.0, 51.8, 11.0), ("N1", 320.0 + 3 * 21 / 3600 / np.cos(np.radians(51.8)), 51.8, 12.0)]
    cube = np.zeros((t.size, 15, 15)) + 100.0
    for sid, ra, de, rp in srcs:
        x, y = w.all_world2pix(ra, de, 0)
        ph = 2 * np.pi * t / 0.4
        dm = 0.2 * 0.5 * (1 - np.cos(2 * ph)) + 0.03 * (1 - np.cos(ph)) if sid == var else np.zeros_like(t)
        cube += (10 ** (-0.4 * (rp - 20)) * 10 ** (-0.4 * dm))[:, None, None] * ta.prf_image((15, 15), float(x), float(y), 0.9)[None]
    noise = np.sqrt(cube * 1800) / 1800
    cube = cube + rng.normal(0, 1, cube.shape) * noise
    o = types.SimpleNamespace(sector=sector, flux=types.SimpleNamespace(value=cube), flux_err=types.SimpleNamespace(value=noise),
                              time=types.SimpleNamespace(value=t - 2457000.0), wcs=w)
    return o, [{"source_id": s, "ra": r, "dec": d, "rp_mag": m} for s, r, d, m in srcs]


def test_tess_period_two_p_and_localisation(tmp_path) -> None:
    rng = np.random.default_rng(7)
    a, src = _tpf(15, 2459800.0, "T", rng)
    b, _ = _tpf(55, 2460500.0, "T", rng)
    r = ta.run_tess("T", 320.0, 51.8, tmp_path / "t", sources=src, tpfs=[a, b], cfg=AppConfig())
    assert r.best is not None and abs(r.best.period_d - 0.4) < 5e-4  # 2P chosen (unequal minima/maxima)
    assert r.two_p_preferred
    assert max(r.localization, key=r.localization.get) == "T"
    # VAR-TAB-2026-10: the interactive UI plots read these files
    pg = np.loadtxt(tmp_path / "t" / "periodogram.csv", delimiter=",", skiprows=1)
    assert pg.shape[1] == 3 and pg.shape[0] > 100
    import json as _json

    pix = _json.loads((tmp_path / "t" / "pixel_map.json").read_text())
    assert "T" in pix["stars"] and len(pix["amplitude"]) == len(pix["mask"])
    assert "files" not in _json.loads((tmp_path / "t" / "tess_result.json").read_text())
    a2, src2 = _tpf(15, 2459800.0, "N1", rng)
    b2, _ = _tpf(55, 2460500.0, "N1", rng)
    r2 = ta.run_tess("T", 320.0, 51.8, tmp_path / "n", sources=src2, tpfs=[a2, b2], cfg=AppConfig())
    assert r2.verdict == "NEIGHBOUR"


# --- independent-review fixes (2026-10-10) ---------------------------------------
def test_blend_model_matches_pixel_photometry() -> None:
    """The neighbour-light model must reproduce what the photometry measures on pixels:
    exact aperture sum minus area x plain median of the annulus."""
    from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

    from comp_blend import _moffat_psf, contamination_fractions, moffat_alpha

    fw, beta, r_ap, ann = 4.0, 2.65, 6.0, (10.8, 20.8)
    alpha = moffat_alpha(fw, beta)
    yy, xx = np.mgrid[0:200, 0:200].astype(float)
    sub = 8  # sub-pixel sampling of the PSF
    ys, xs = np.mgrid[0:200 * sub, 0:200 * sub].astype(float)
    ys, xs = (ys + 0.5) / sub - 0.5, (xs + 0.5) / sub - 0.5

    def img(x0: float, y0: float, f: float) -> np.ndarray:
        a = _moffat_psf((xs - x0) ** 2 + (ys - y0) ** 2, alpha, beta) * f / sub**2
        return a.reshape(200, sub, 200, sub).sum(axis=(1, 3))

    def measure(d: np.ndarray, nbx: list[tuple[float, float]]) -> float:
        _ = nbx
        pos = [(100.0, 100.0)]
        am = CircularAnnulus(pos, r_in=ann[0], r_out=ann[1]).to_mask(method="center")
        am = am[0] if isinstance(am, list) else am
        sky = float(np.median(np.asarray(am.get_values(d))))  # VYVAR sky: plain annulus median
        ap = CircularAperture(pos, r=r_ap)
        return float(aperture_photometry(d, ap, method="exact")["aperture_sum"][0]) - sky * float(ap.area)

    star = img(100.0, 100.0, 1.0)
    alone = measure(star, [])
    for nx_, ny_, fn in ((109.0, 100.0, 0.3), (115.0, 103.0, 50.0), (122.0, 100.0, 300.0)):
        meas = measure(star + img(nx_, ny_, fn), [(nx_, ny_)])
        c_pix = meas / alone - 1.0
        c_mod = contamination_fractions(np.array([[100.0, 100.0]]), np.array([1.0]), np.array([[nx_, ny_]]), np.array([fn]),
                                        r_ap_px=r_ap, fwhm_px=fw, beta=beta, annulus_px=ann)[0]
        assert abs(c_mod - c_pix) < 0.05 * max(abs(c_pix), 0.02), (nx_, fn, c_mod, c_pix)


def test_skewnorm_quantile_fit_robust_to_field_variables() -> None:
    from scipy import stats

    from variability_field import _skewnorm_quantile_fit

    rng = np.random.default_rng(1)
    u = stats.skewnorm.rvs(3.0, loc=-1.0, scale=1.7, size=1500, random_state=rng)
    p0 = _skewnorm_quantile_fit(u)
    p1 = _skewnorm_quantile_fit(np.r_[u, rng.uniform(5, 12, 45)])  # 3 % variables
    t_true = stats.skewnorm.isf(1e-5, 3.0, -1.0, 1.7)
    assert abs(stats.skewnorm.isf(1e-5, *p0) / t_true - 1) < 0.15
    assert abs(stats.skewnorm.isf(1e-5, *p1) / t_true - 1) < 0.25


def test_period_error_is_calibrated() -> None:
    """chi^2 + 1 width (Schwarzenberg-Czerny 1991) vs the scatter of 25 realisations."""
    rng = np.random.default_rng(2)
    t = np.arange(0, 27, 1800 / 86400)
    f0 = 1 / 0.5672
    fs, sg = [], []
    for _ in range(25):
        m = 0.05 * np.sin(2 * np.pi * f0 * t) + rng.normal(0, 0.002, t.size)
        e = np.full(t.size, 0.002)
        f, _th, s = ta._zoom_peak(t, m, e, f0 * (1 + 1e-4), 0.01, 0.05, 20.0, nterms=1)
        fs.append(f)
        sg.append(s)
    ratio = np.std(fs) / np.median(sg)
    assert 0.6 < ratio < 1.6


def test_zoom_peak_on_range_edge_does_not_crash() -> None:
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 27, 1300))
    m = rng.normal(0, 0.01, t.size)
    f, _th, s = ta._zoom_peak(t, m, np.full(t.size, 0.01), 24.0, 0.01, 0.1, 24.0)
    assert 0.1 <= f <= 24.0
