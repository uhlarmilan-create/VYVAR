"""VYVAR v12 orient (Milan approved 2026-11-05): SKY-ORIENT-2026-11.

Every displayed sky image is shown North up, East left through one lossless display transform
(``sky_display_orient``: transpose / flips only, chosen from the WCS at the frame centre; mirrored optics get a
flip so East ends up LEFT). Markers go through the same transform; an N / E compass shows the residual rotation
(|residual| <= 45 deg). No WCS -> FITS-natural view + "orientation unknown (no WCS)". Display only: FITS / CSV
files are never written.

The checks of North / East below are made with the WCS itself (position angle of display "up" / "left"
seen from the display centre), not with the module's own direction vectors.
"""
from __future__ import annotations

import hashlib
import io
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

_SRC = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

NY, NX = 80, 120


def _wcs(rot_deg: float, *, mirrored: bool = False, ny: int = NY, nx: int = NX, dec: float = 41.0,
         scale_arcsec: float = 2.0) -> WCS:
    """TAN WCS; ``rot_deg`` as VYVAR reports it (arctan2(CD2_1, CD2_2), Milan's frames -177.7)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [209.5, dec]
    w.wcs.crpix = [nx / 2.0 + 0.5, ny / 2.0 + 0.5]
    s = scale_arcsec / 3600.0
    t = math.radians(rot_deg)
    # standard sky (East left with North up in the FITS-natural view) rotated by t
    cd = np.array([[-s * math.cos(t), s * math.sin(t)], [s * math.sin(t), s * math.cos(t)]])
    if mirrored:
        cd[:, 0] *= -1.0
    w.wcs.cd = cd
    if not mirrored:
        assert abs(math.degrees(math.atan2(cd[1, 0], cd[1, 1])) - rot_deg) < 1e-9  # the VYVAR rotation convention
    return w


def _pa_deg(w: WCS, x0: float, y0: float, x1: float, y1: float) -> float:
    """Position angle (N through E) of pixel 1 seen from pixel 0."""
    c0, c1 = w.pixel_to_world(x0, y0), w.pixel_to_world(x1, y1)
    return float(c0.position_angle(c1).deg)


def _ang_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


CASES = [(r, m) for r in (0.0, 90.0, 180.0, -177.7, 30.0, -30.0, 44.0, 46.0, 135.0, -90.0) for m in (False, True)]


# ---------------------------------------------------------------------------------------------- transform
@pytest.mark.parametrize("rot,mirrored", CASES)
def test_north_up_east_left_after_transform(rot: float, mirrored: bool) -> None:
    from sky_display_orient import display_transform

    w = _wcs(rot, mirrored=mirrored)
    o = display_transform(w, (NY, NX))
    assert o.has_wcs and o.mirrored == mirrored and abs(o.residual_deg) <= 45.0 + 1e-9
    h, wd = o.out_shape
    c0, r0 = (wd - 1) / 2.0, (h - 1) / 2.0
    x0, y0 = o.inverse_xy(c0, r0)
    xu, yu = o.inverse_xy(c0, r0 - 10.0)  # display up
    xl, yl = o.inverse_xy(c0 - 10.0, r0)  # display left
    pa_up = _pa_deg(w, float(x0), float(y0), float(xu), float(yu))
    pa_left = _pa_deg(w, float(x0), float(y0), float(xl), float(yl))
    assert _ang_diff(pa_up, 0.0) <= 45.0 + 1e-6, (rot, mirrored, pa_up)  # North within 45 deg of up
    assert _ang_diff(pa_left, 90.0) <= 45.0 + 1e-6, (rot, mirrored, pa_left)  # East on the left
    # the residual is the honest rotation: up is North rotated by -residual (residual counter-clockwise)
    assert _ang_diff(pa_up, -o.residual_deg) < 0.2


def test_milan_rotation_is_turned_by_180_without_resampling() -> None:
    from sky_display_orient import display_transform

    o = display_transform(_wcs(-177.7).to_header(), (NY, NX))
    assert (o.transpose, o.flip_c, o.flip_r) == (False, True, False)  # = rot180 of the FITS-natural view
    assert abs(o.residual_deg - (-2.3)) < 0.05
    a = np.arange(NY * NX, dtype=np.float32).reshape(NY, NX)
    d = o.apply_image(a)
    assert np.array_equal(np.sort(d.ravel()), np.sort(a.ravel()))  # the same pixel values, none resampled
    assert np.shares_memory(d, a)  # a view


def test_mirrored_optics_get_a_flip() -> None:
    from sky_display_orient import display_transform

    normal = display_transform(_wcs(0.0), (NY, NX))
    mirr = display_transform(_wcs(0.0, mirrored=True), (NY, NX))
    assert np.linalg.det(normal.matrix) * np.linalg.det(mirr.matrix) < 0  # opposite parity of the transform
    assert mirr.east is not None and mirr.east[0] < -0.7  # East left also for the mirrored chip


def test_no_wcs_is_natural_view_and_says_so() -> None:
    from sky_display_orient import NO_WCS_NOTE, display_transform

    hdr = fits.Header()
    hdr["NAXIS"] = 2
    o = display_transform(hdr, (NY, NX))
    assert not o.has_wcs and o.note == NO_WCS_NOTE == "orientation unknown (no WCS)"
    assert (o.transpose, o.flip_c, o.flip_r) == (False, False, True)  # FITS row 0 at the bottom, never guessed
    n = display_transform(None, (NY, NX))
    assert (n.has_wcs, n.transpose, n.flip_c, n.flip_r) == (False, False, False, True)


@pytest.mark.parametrize("tr,fc,fr", list(itertools.product((False, True), repeat=3)))
def test_xy_round_trip_and_image_agree(tr: bool, fc: bool, fr: bool) -> None:
    from sky_display_orient import DisplayOrient

    o = DisplayOrient(nx=NX, ny=NY, transpose=tr, flip_c=fc, flip_r=fr, has_wcs=False)
    xs, ys = np.meshgrid(np.arange(NX, dtype=float), np.arange(NY, dtype=float))
    xs, ys = np.r_[xs.ravel(), 3.5, 0.5], np.r_[ys.ravel(), 7.5, 79.5]
    c, r = o.apply_xy(xs, ys)
    bx, by = o.inverse_xy(c, r)
    assert np.array_equal(bx, xs) and np.array_equal(by, ys)  # exact on pixel centres / half pixels
    rng = np.random.default_rng(5)
    fx, fy = rng.uniform(-0.5, NX - 0.5, 500), rng.uniform(-0.5, NY - 0.5, 500)
    gx, gy = o.inverse_xy(*o.apply_xy(fx, fy))
    assert np.max(np.abs(gx - fx)) < 1e-9 and np.max(np.abs(gy - fy)) < 1e-9
    cm, ym = o.apply_xy_mpl(fx, fy)
    hx, hy = o.inverse_xy_mpl(cm, ym)
    assert np.max(np.abs(hx - fx)) < 1e-9 and np.max(np.abs(hy - fy)) < 1e-9
    a = np.zeros((NY, NX))
    a[13, 101] = 1.0  # pixel x=101, y=13
    d = o.apply_image(a)
    assert d.shape == o.out_shape
    rr, cc = np.unravel_index(int(np.argmax(d)), d.shape)
    c1, r1 = o.apply_xy(101, 13)
    assert (cc, rr) == (int(c1), int(r1))
    m = o.mpl_image(a)
    ym1 = int(np.argmax(m[:, int(c1)]))
    assert ym1 == int(o.apply_xy_mpl(101, 13)[1])
    sub = o.for_shape((5, 7))
    assert sub.out_shape == ((7, 5) if tr else (5, 7))


def test_mpl_wcs_maps_display_to_the_same_sky() -> None:
    from sky_display_orient import display_transform

    w = _wcs(-177.7)
    o = display_transform(w, (NY, NX))
    dw = o.mpl_wcs(w)
    x, y = 17.0, 61.0
    cm, ym = o.apply_xy_mpl(x, y)
    assert np.allclose(dw.pixel_to_world_values(cm, ym), w.pixel_to_world_values(x, y), atol=1e-12)
    back = dw.world_to_pixel_values(*w.pixel_to_world_values(x, y))
    assert np.allclose(back, (cm, ym), atol=1e-6)


# ------------------------------------------------------------------------------------------------ renders
def _star_image(x: float, y: float, ny: int = NY, nx: int = NX, amp: float = 5000.0) -> np.ndarray:
    yy, xx = np.mgrid[:ny, :nx]
    rng = np.random.default_rng(2)
    return (rng.normal(1000.0, 5.0, (ny, nx)) + amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5**2))).astype(
        np.float32)


def _msqa_bundle(tmp_path: Path, rot: float, star_xy: tuple[float, float]) -> Path:
    w = _wcs(rot)
    hdr = w.to_header()
    fits.PrimaryHDU(_star_image(*star_xy), header=hdr).writeto(tmp_path / "MASTERSTAR.fits")
    pd.DataFrame([dict(x=star_xy[0], y=star_xy[1], source_state="DETECTED_P1", gaia_dao_resid_px=0.1,
                       phot_g_mean_mag=12.0)]).to_csv(tmp_path / "masterstars_full_match.csv", index=False)
    ra, dec = w.wcs_pix2world([list(star_xy)], 0).T
    pd.DataFrame({"ra_deg": ra, "dec_deg": dec, "mag": [12.0]}).to_csv(tmp_path / "field_catalog_cone.csv", index=False)
    return tmp_path


def _digest(paths: list[Path]) -> dict[str, tuple[str, int]]:
    return {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in paths}


def test_masterstar_map_marker_on_star_and_north_up(tmp_path: Path) -> None:
    """Plate solving -> Map: star north-east of the centre (sky) is drawn upper-left; the matched-DAO ring
    (green) sits on the star after the transform."""
    from PIL import Image

    from masterstar_qa_plot import load_masterstar_qa, render_masterstar_overlay

    w = _wcs(-177.7)
    cx, cy = (NX - 1) / 2.0, (NY - 1) / 2.0
    c0 = w.pixel_to_world(cx, cy)
    import astropy.units as u

    sx, sy = (float(v) for v in w.world_to_pixel(c0.directional_offset_by(45 * u.deg, 60 * u.arcsec)))
    b = _msqa_bundle(tmp_path, -177.7, (sx, sy))
    files = [b / "MASTERSTAR.fits", b / "masterstars_full_match.csv", b / "field_catalog_cone.csv"]
    before = _digest(files)
    qa = load_masterstar_qa(b)
    jpg, cap = render_masterstar_overlay(qa["image"], orient=qa["orient"], matched_xy=qa["matched_xy"],
                                         unmatched_xy=qa["unmatched_xy"], gaia_xy=qa["gaia_xy"], show_gaia=False,
                                         show_matches=True)
    assert "North up, East left" in cap
    rgb = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")).astype(int)
    h, wd = rgb.shape[:2]
    gray = rgb.mean(axis=2)
    core = np.unravel_index(int(np.argmin(gray[:, : wd - 30])), gray[:, : wd - 30].shape)  # negative: star darkest
    assert core[0] < h / 2 and core[1] < wd / 2  # North-East of the centre -> upper left
    green = (rgb[..., 1] > 110) & (rgb[..., 0] < 90) & (rgb[..., 2] < 140)
    gy, gx = np.nonzero(green)
    assert gx.size > 5
    assert abs(gx.mean() - core[1]) < 2.0 and abs(gy.mean() - core[0]) < 2.0
    assert _digest(files) == before  # display only: nothing written


def test_masterstar_map_compass_is_drawn(tmp_path: Path) -> None:
    from PIL import Image

    from masterstar_qa_plot import render_masterstar_overlay
    from sky_display_orient import display_transform

    img = np.full((NY, NX), 1000.0, np.float32)
    e = np.empty((0, 2))
    kw = dict(matched_xy=e, unmatched_xy=e, gaia_xy=e, show_gaia=False, show_matches=False)
    jpg, _ = render_masterstar_overlay(img, orient=display_transform(_wcs(30.0), img.shape), **kw)
    rgb = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")).astype(int)
    orange = (rgb[..., 0] > 150) & (rgb[..., 1] < 120) & (rgb[..., 2] < 80)
    ys, xs = np.nonzero(orange)
    assert xs.size > 10 and xs.mean() > NX / 2 and ys.mean() < NY / 2  # compass in the upper right
    jpg_n, cap_n = render_masterstar_overlay(img, orient=display_transform(None, img.shape), **kw)
    assert "orientation unknown (no WCS)" in cap_n


@pytest.mark.parametrize("rot,mirrored", [(-177.7, False), (90.0, True), (30.0, False)])
def test_field_map_image_and_markers_share_the_transform(tmp_path: Path, rot: float, mirrored: bool) -> None:
    from field_map import FieldMapCanvas

    data = _star_image(23.0, 61.0)
    canvas = FieldMapCanvas(data, _wcs(rot, mirrored=mirrored).to_header())
    try:
        shown = np.asarray(canvas.ax.images[0].get_array())
        c, yu = canvas.xy(23.0, 61.0)
        assert shown[int(round(yu)), int(round(c))] == pytest.approx(float(data[61, 23]))
        assert "North up, East left" in canvas.subtitle
        o = canvas.orient
        assert o.has_wcs and abs(o.residual_deg) <= 45.0
    finally:
        canvas.close()


def test_hrd_field_jpeg_and_figure_markers(tmp_path: Path) -> None:
    from PIL import Image

    from hrd_field_render import field_image_jpeg
    from sky_display_orient import orient_from_fits
    from ui_hrd import image_figure

    fp = tmp_path / "MASTERSTAR.fits"
    fits.PrimaryHDU(_star_image(30.0, 20.0, amp=50000.0), header=_wcs(-177.7).to_header()).writeto(fp)
    stars = pd.DataFrame({"x": [30.0], "y": [20.0], "bp_rp": [0.8], "phot_g_mean_mag": [10.0]})
    before = _digest([fp])
    jpg = field_image_jpeg(fp, stars, "inverted", tmp_path / "f.jpg", fwhm_px=2.0)
    assert _digest([fp]) == before
    o = orient_from_fits(fp)
    g = np.asarray(Image.open(jpg).convert("L")).astype(int)
    rr, cc = np.unravel_index(int(np.argmin(g[:, : NX - 30])), g[:, : NX - 30].shape)
    c, r = o.apply_xy(30.0, 20.0)
    assert abs(cc - float(c)) <= 1 and abs(rr - float(r)) <= 1
    df = pd.DataFrame({"catalog_id": ["1"], "x": [30.0], "y": [20.0], "vsx_name": [np.nan], "bp_rp": [0.8],
                       "phot_g_mean_mag": [10.0], "abs_mag_g": [4.0], "parallax": [5.0], "parallax_over_error": [50.0],
                       "teff_gspphot": [np.nan], "measured": [True], "reliable": [True]})
    fig = image_figure(jpg, {"df": df, "top": df.iloc[:0]}, set(), None, light=True, size=(NX, NY), orient=o)
    assert float(fig.data[0].x[0]) == float(c) and float(fig.data[0].y[0]) == float(r)
    assert any(getattr(a, "text", "") == "<b>N</b>" for a in fig.layout.annotations)


def test_quicklook_preview_is_oriented(tmp_path: Path) -> None:
    from PIL import Image

    from importer import quicklook_preview_png_bytes
    from sky_display_orient import orient_from_fits

    fp = tmp_path / "frame.fits"
    fits.PrimaryHDU(_star_image(100.0, 10.0, amp=50000.0), header=_wcs(-177.7).to_header()).writeto(fp)
    before = _digest([fp])
    g = np.asarray(Image.open(io.BytesIO(quicklook_preview_png_bytes(fp))).convert("L")).astype(int)
    assert _digest([fp]) == before
    c, r = orient_from_fits(fp).apply_xy(100.0, 10.0)  # rot180: lands at the lower left, away from the compass
    sub = g[:, : NX - 30]
    rr, cc = np.nonzero(sub == sub.min())  # ZScale clips the core: centroid of the clipped plateau
    assert abs(cc.mean() - float(c)) <= 1 and abs(rr.mean() - float(r)) <= 1
    # without a WCS the note is drawn (orange text), lower left
    fp2 = tmp_path / "raw.fits"
    fits.PrimaryHDU(_star_image(15.0, 70.0)).writeto(fp2)
    rgb = np.asarray(Image.open(io.BytesIO(quicklook_preview_png_bytes(fp2))).convert("RGB")).astype(int)
    orange = (rgb[..., 0] > 150) & (rgb[..., 1] < 120) & (rgb[..., 2] < 80)
    assert np.nonzero(orange)[0].size > 5 and np.nonzero(orange)[0].mean() > NY / 2


def test_cutout_needs_orientation_for_in_memory_image(tmp_path: Path) -> None:
    import star_page_m3 as m3
    from sky_display_orient import display_transform

    data = _star_image(60.0, 40.0)
    with pytest.raises(ValueError):
        m3.render_cutout(data, 60.0, 40.0, 3.0, 5.0, 8.0, tmp_path / "c.png")
    out = m3.render_cutout(data, 60.0, 40.0, 3.0, 5.0, 8.0, tmp_path / "c.png",
                           orient=display_transform(_wcs(-177.7), data.shape))
    assert out is not None and out.is_file()


def test_tess_pixel_figure_uses_the_cutout_wcs() -> None:
    from sky_display_orient import display_transform
    from ui_variability import pixel_figure

    amp = np.arange(12 * 10, dtype=float).reshape(12, 10)
    w = _wcs(90.0, ny=12, nx=10)
    pix = {"amplitude": amp.tolist(), "mask": np.zeros((12, 10), int).tolist(), "stars": {"7": [2.0, 3.0]},
           "wcs_header": w.to_header_string()}
    fig = pixel_figure(pix, "7")
    o = display_transform(w, amp.shape)
    assert np.array_equal(np.asarray(fig.data[0].z), o.apply_image(amp))
    c, r = o.apply_xy(2.0, 3.0)
    assert float(fig.data[-1].x[0]) == float(c) and float(fig.data[-1].y[0]) == float(r)
    old = dict(pix, wcs_header=None)  # pixel_map.json written before: no WCS -> the note, never a guess
    assert any("orientation unknown" in str(a.text) for a in pixel_figure(old, "7").layout.annotations)


def test_variability_field_png_writes_only_the_png(tmp_path: Path) -> None:
    from variability_plots import field_png

    fp = tmp_path / "MASTERSTAR.fits"
    fits.PrimaryHDU(_star_image(30.0, 30.0), header=_wcs(135.0).to_header()).writeto(fp)
    before = _digest([fp])
    out = field_png(fp, pd.DataFrame({"x": [30.0], "y": [30.0], "label": ["V1"]}), tmp_path / "v.png")
    assert out is not None and out.is_file() and _digest([fp]) == before


def test_every_sky_render_goes_through_the_shared_helper() -> None:
    """Guard: the sky renders listed in the v12 orient report import sky_display_orient."""
    files = ["masterstar_qa_plot.py", "field_map.py", "hrd_field_render.py", "ui_hrd.py", "hrd_analysis.py",
             "importer.py", "star_page_m3.py", "variability_plots.py", "ui_variability.py", "tess_analysis.py",
             "ui_photometry_quality.py"]
    for f in files:
        assert "sky_display_orient" in (_SRC / f).read_text(encoding="utf-8"), f
    hrd = (_SRC / "hrd_analysis.py").read_text(encoding="utf-8")
    assert "def field_annotation_y" not in hrd and "np.flipud(gray)" not in hrd  # the old copy is gone
