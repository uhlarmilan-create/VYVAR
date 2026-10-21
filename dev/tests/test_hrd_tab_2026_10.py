"""HRD-TAB-2026-10: field HRD view helpers and the three field-image modes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hrd_field_render import FIELD_MODES, blackbody_rgb, bp_rp_css, bp_rp_rgb, render_field, teff_from_bp_rp

SRC = Path(__file__).resolve().parents[2] / "src_py"


def test_palette_and_temperatures() -> None:
    assert bp_rp_css(float("nan")) == "rgb(140,145,155)"
    blue, white, red = bp_rp_rgb([-0.3, 0.6, 3.2])
    assert blue[2] > blue[0] and np.allclose(white, 1.0) and red[0] > red[2]
    t = teff_from_bp_rp([0.0, 0.82, 2.5])
    assert t[0] > 9000 > t[1] > 5000 > t[2]
    hot, sun, cool = blackbody_rgb([20000.0, 6500.0, 3000.0])
    assert hot[2] >= hot[0] and np.ptp(sun) < 0.12 and cool[0] > cool[2]  # 6500 K ~ D65 white


def _frame() -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(2)
    h, w = 80, 120
    yy, xx = np.mgrid[0:h, 0:w]
    stars = pd.DataFrame({"x": [30.0, 90.0], "y": [40.0, 40.0], "bp_rp": [-0.2, 2.8], "phot_g_mean_mag": [9.0, 9.0]})
    img = rng.normal(100.0, 5.0, (h, w))
    for x, y in zip(stars["x"], stars["y"]):
        img += 5e4 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.5**2))
    return img, stars


def test_render_modes_keep_star_colour_and_neutral_sky() -> None:
    img, stars = _frame()
    out = {m: render_field(img, stars, m, fwhm_px=1.5) for m in FIELD_MODES}
    inv = out["inverted"]
    assert inv.dtype == np.uint8 and inv[40, 30, 0] < 60 and inv[5, 5, 0] > 180  # dark star on white sky
    for m in ("vivid", "physical"):
        o = out[m].astype(int)
        hot, cool = o[40, 30], o[40, 90]
        assert hot[2] > hot[0] and cool[0] > cool[2], m  # saturated cores keep their hue
        sky = o[5, 5]
        assert np.ptp(sky) <= 2, m  # no colour on the sky
    v, p = out["vivid"].astype(int)[40, 90], out["physical"].astype(int)[40, 90]
    assert (v[0] - v[2]) > (p[0] - p[2])  # vivid is more saturated than the blackbody


def test_figures_and_table() -> None:
    from ui_hrd import hrd_figure, image_figure, notable_table, side_figure

    df = pd.DataFrame({"catalog_id": ["1", "2", "3"], "bp_rp": [0.5, 1.5, 1.0], "abs_mag_g": [4.0, 0.0, np.nan],
                       "phot_g_mean_mag": [10.0, 9.0, 14.0], "teff_gspphot": [np.nan] * 3, "parallax": [5.0, 1.0, np.nan],
                       "parallax_over_error": [50.0, 10.0, np.nan], "x": [10.0, 20.0, 30.0], "y": [10.0, 20.0, 30.0],
                       "vsx_name": [np.nan, "XX Cvn", np.nan], "vsx_type": [np.nan, "EW", np.nan],
                       "measured": [True, False, False], "reliable": [True, True, False]})
    top = pd.DataFrame({"catalog_id": ["2"], "category": ["Red giant (K0III, SIMBAD)"], "ident": ["confirmed"], "ident_detail": ["K0III, SIMBAD"],
                        "simbad_main_id": ["HD 1"], "sp_type_raw": ["K0III"], "otype_raw": ["RG*"], "mag_g": [9.0], "abs_mag_g": [0.0],
                        "bp_rp": [1.5], "teff": [np.nan], "logg": [2.5], "dist_pc": [1000.0], "parallax_snr": [10.0], "ag_gspphot": [0.1],
                        "ra_dec_sex": ["13:00:00 +40:00:00"], "x_px": [20.0], "y_px": [20.0], "vsx_name": ["XX Cvn"], "vsx_type": ["EW"],
                        "measured": [False]})
    data = {"df": df, "top": top}
    for variant in ("cmd", "teff", "dist"):
        fig = hrd_figure(data, variant, {"vsx", "meas", "labels"}, "2")
        assert len(fig.data) == 4  # field, VSX, measured, the notable star
    assert any("HD 1" in str(a.text) for a in hrd_figure(data, "cmd", {"labels"}, None).layout.annotations)
    assert len(side_figure(df).data[0].x) == 1
    tab = notable_table(top)
    assert tab.loc[0, "Star"] == "HD 1" and tab.loc[0, "Variable (VSX)"] == "XX Cvn (EW)" and tab.loc[0, "Teff [K]"] > 3000
    from PIL import Image

    jpg = Path(__file__).parent / "_tmp_hrd_field.jpg"
    Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(jpg)
    try:
        f = image_figure(jpg, data, {"notable", "vsx"}, "2", light=True, size=(60, 40))
        assert f.layout.images[0].source.startswith("data:image/jpeg;base64,") and len(f.data) == 3
    finally:
        jpg.unlink()


def test_old_colour_field_is_gone() -> None:
    assert not (SRC / "hrd_colorfield.py").exists()
    from config import KNOWN_REMOVED_KEYS

    assert {"hrd_color_field_enabled", "hrd_color_white_point", "hrd_color_chroma_boost"} <= set(KNOWN_REMOVED_KEYS)
