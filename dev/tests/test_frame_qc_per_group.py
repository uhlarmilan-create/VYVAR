"""QC-PER-GROUP-2026-10 / QC-METRICS-2026-10: frame QC per channel (frame_qc.py)
and per-frame metrics (frame_metrics.py)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src_py"))

import frame_metrics as fmx  # noqa: E402
import frame_qc as fq  # noqa: E402


def _rows(group, fwhm, noise, elong=None, prefix="f", sky=None):
    n = len(fwhm)
    elong = elong if elong is not None else [1.05 + 0.002 * (i % 5) for i in range(n)]
    sky = sky if sky is not None else [50.0] * n
    return [
        {"FILE_PATH": f"/a/{prefix}{i:03d}.fits", "OBSERVATION_GROUP_KEY": group, "FWHM": f,
         "NOISE_SIGMA": s, "ELONGATION_MEAN": e, "SKY_LEVEL": b}
        for i, (f, s, e, b) in enumerate(zip(fwhm, noise, elong, sky))
    ]


def _star_field(rng, n=120, size=1000):
    xy = rng.uniform(30, size - 30, (n, 2))
    flux = 10 ** rng.uniform(3, 5, n)
    return xy, flux


def _stars(xy, flux, *, dzp=0.0, shift=(0.0, 0.0), flip=False, size=1000, patchy=0.0, rng=None):
    f = flux * 10 ** (-0.4 * dzp)
    if patchy and rng is not None:
        f = f * 10 ** (-0.4 * rng.normal(0, patchy, f.size))
    p = xy + np.asarray(shift)
    if flip:
        p = (size - 1) - p
    return np.column_stack([p, f]), (size, size)


def test_groups_independent_and_fwhm_alone_never_rejects():
    rng = np.random.default_rng(1)
    g = list(4.0 + rng.normal(0, 0.1, 30)) + [9.0]  # one very wide frame in g
    r = list(3.0 + rng.normal(0, 0.1, 90))
    rows = _rows("g|120|4", g, [2.0] * 31, prefix="g") + _rows("r|100|4", r, [2.0] * 90, prefix="r")
    res = fq.evaluate_frame_qc(rows, k=3.0)
    assert res["g|120|4"].n_rejected == 0 and res["r|100|4"].n_rejected == 0
    assert res["g|120|4"].fwhm_median > 3.9 > 3.1 > res["r|100|4"].fwhm_median


def test_rejects_missing_fwhm_and_elongation():
    fwhm = [3.0, 3.1, 2.9, 3.05, 2.95, 3.0, 3.0, None]
    elong = [1.05, 1.06, 1.04, 1.055, 1.045, 1.5, 1.052, 1.048]
    res = fq.evaluate_frame_qc(_rows("V|60|1", fwhm, [2.0] * 8, elong), k=3.0)["V|60|1"]
    reasons = {Path(p).name: r for p, r in res.rejected.items()}
    assert set(reasons) == {"f005.fits", "f007.fits"}
    assert reasons["f005.fits"].startswith(fq.REASON_ELONG)
    assert reasons["f007.fits"] == fq.REASON_NO_FWHM


def test_transparency_rejects_cloudy_and_patchy_frames_and_handles_flip():
    rng = np.random.default_rng(3)
    xy, flux = _star_field(rng)
    n = 12
    rows = _rows("V|60|1", [3.0 + 0.01 * i for i in range(n)], [2.0] * n)
    stars = {}
    for i, r in enumerate(rows):
        dz = 0.02 * rng.normal()
        kw = dict(dzp=dz, shift=tuple(rng.uniform(-30, 30, 2)), flip=(i >= 6), rng=rng, patchy=0.005)
        if i == 3:
            kw["dzp"] = 1.0  # thick cloud
        if i == 4:
            kw["patchy"] = 0.3  # patchy cloud
        stars[r["FILE_PATH"]] = _stars(xy, flux, **kw)
    res = fq.evaluate_frame_qc(rows, k=3.0, stars=stars)["V|60|1"]
    reasons = {Path(p).name: r for p, r in res.rejected.items()}
    assert reasons["f003.fits"].startswith(fq.REASON_TRANSP)
    assert reasons["f004.fits"].startswith(fq.REASON_PATCHY)
    assert len(reasons) == 2
    # flipped frames were matched (meridian flip)
    assert all(res.frames[f"/a/f{i:03d}.fits"]["n_common"] >= 50 for i in range(6, n))


def test_best_frame_is_deepest_with_noise_and_transparency():
    rng = np.random.default_rng(5)
    xy, flux = _star_field(rng)
    # f000 sharpest but noisy (moon); f001 slightly wider on dark sky; f002 like f001 but 0.3 mag cloud
    rows = _rows("r|60|1", [2.8, 3.0, 3.0, 3.4, 3.4, 3.4], [6.0, 2.0, 2.0, 2.2, 2.2, 2.2])
    dz = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0]
    stars = {r["FILE_PATH"]: _stars(xy, flux, dzp=d) for r, d in zip(rows, dz)}
    res = fq.evaluate_frame_qc(rows, k=3.0, stars=stars)["r|60|1"]
    assert Path(res.best_path).name == "f001.fits"
    assert math.isclose(res.best_transparency, 1.0, abs_tol=1e-6)


def test_no_limit_for_small_or_degenerate_groups():
    res = fq.evaluate_frame_qc(_rows("V|60|1", [3.0] * 6, [2.0] * 6, elong=[1.05] * 5 + [1.3]), k=3.0)["V|60|1"]
    assert res.elong_limit is None and res.n_rejected == 0  # MAD = 0
    res = fq.evaluate_frame_qc(_rows("V|60|1", [3.0, 3.1, 5.0], [2.0] * 3, elong=[1.05, 1.06, 1.5]), k=3.0)["V|60|1"]
    assert res.elong_limit is None and res.n_rejected == 0  # n < 5


def test_best_path_for_setup_roundtrip_and_star_sidecar(tmp_path):
    res = fq.evaluate_frame_qc(_rows("g|120|4", [3, 2.5], [2, 2], prefix="g")
                               + _rows("r|100|4", [3, 3.5], [2, 2], prefix="r"), k=3.0)
    fq.write_frame_qc(tmp_path, res, k=3.0, reject=True)
    assert Path(fq.best_path_for_setup(tmp_path, "g_120_4")).name == "g001.fits"
    assert Path(fq.best_path_for_setup(tmp_path, "r_100_4")).name == "r000.fits"
    assert fq.best_path_for_setup(tmp_path, None) is None  # ambiguous with two groups
    fq.save_qc_stars(tmp_path, {r"C:\x\a.fits": ([[1, 2, 3.5]], (10, 20))})
    fq.save_qc_stars(tmp_path, {r"C:\x\b.fits": ([[4, 5, 6.0]], None)})  # merges
    got = fq.load_qc_stars(tmp_path)
    a, shp = got[r"C:\x\a.fits"]
    assert a.tolist() == [[1.0, 2.0, 3.5]] and shp == (10, 20) and got[r"C:\x\b.fits"][1] is None
    assert [p.name for p in (tmp_path / "qc").iterdir()] == ["qc_stars.json"]  # one file per draft


def test_frame_metrics_on_synthetic_gaussian_field():
    rng = np.random.default_rng(0)
    size, s_true, noise = 600, 6.0 / 2.3548, 3.0
    a = np.full((size, size), 100.0) + rng.normal(0, noise, (size, size))
    yy, xx = np.mgrid[-15:16, -15:16]
    for _ in range(120):
        x, y = rng.uniform(40, size - 40, 2)
        amp = rng.uniform(300, 5000)
        dx, dy = xx - (x - int(x)), yy - (y - int(y))
        a[int(y) - 15:int(y) + 16, int(x) - 15:int(x) + 16] += amp * np.exp(
            -0.5 * ((dx / (1.3 * s_true)) ** 2 + (dy / s_true) ** 2))
    m = fmx.measure_frame(a)
    assert abs(m["fwhm_mean"] - 6.0 * math.sqrt(1.3)) < 0.15
    assert abs(m["elongation_mean"] - 1.3) < 0.03
    assert abs(m["sky_background"] - 100.0) < 0.5
    assert abs(m["noise_sigma"] - noise) / noise < 0.15
    assert m["star_count"] >= 100 and len(m["qc_stars"]) >= 80


def test_pdf_report_uses_run_frame_qc(tmp_path):
    import photometry_report as pr

    res = fq.evaluate_frame_qc(_rows("g|120|4", [3.0, 3.1, 2.9, 3.05, 2.95, 6.0], [2.0] * 6, prefix="g"), k=3.0)
    fq.write_frame_qc(tmp_path, res, k=3.0, reject=True)
    cls = next(v for v in vars(pr).values() if isinstance(v, type) and hasattr(v, "_frame_qc_group"))
    rep = cls.__new__(cls)
    rep.draft_dir = tmp_path
    rep.obs_group = "g_120_4"
    assert rep._frame_qc_group()["n_frames"] == 6
    assert math.isnan(rep._qa_fwhm_limit_px())  # FWHM is not a rejection criterion
    rep.obs_group = "r_100_4"
    assert rep._frame_qc_group() == {}


def test_physical_floors_keep_usable_frames():
    """QC-FLOORS-2026-10: above the tight group limit but below the physical floor -> kept."""
    rng = np.random.default_rng(7)
    xy, flux = _star_field(rng)
    n = 20
    elong = [1.07 + 0.01 * rng.normal() for _ in range(n)]
    elong[5] = 1.205   # S4 Cyg g case: > group limit, < 1.30 floor -> kept
    elong[6] = 1.40    # trailed -> rejected
    rows = _rows("g|120|4", [4.5 + 0.05 * rng.normal() for _ in range(n)], [2.0] * n, elong)
    dz = [0.02 * rng.normal() for _ in range(n)]
    dz[8] = 0.6        # thin cloud: > group limit, < 1.0 mag floor -> kept
    dz[9] = 1.5        # thick cloud -> rejected
    stars = {r["FILE_PATH"]: _stars(xy, flux, dzp=d, rng=rng, patchy=0.005) for r, d in zip(rows, dz)}
    res = fq.evaluate_frame_qc(rows, k=3.0, stars=stars)["g|120|4"]
    names = {Path(p).name for p in res.rejected}
    assert names == {"f006.fits", "f009.fits"}
    assert res.elong_limit >= fq.ELONG_FLOOR and res.dzp_limit >= fq.DZP_FLOOR_MAG
