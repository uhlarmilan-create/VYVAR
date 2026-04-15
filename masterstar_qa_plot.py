"""MASTERSTAR QA: downsampling + vrstvy DAO / MATCH / Gaia kužeľ na PNG (Streamlit)."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd


def downsample_array_2d(arr: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    h, w = arr.shape
    m = max(h, w)
    if m <= max_side:
        return arr, 1.0, 1.0
    scale = max_side / float(m)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    from PIL import Image

    im = Image.fromarray(np.nan_to_num(arr, nan=0.0).astype(np.float32), mode="F")
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32), float(nw) / float(w), float(nh) / float(h)


def percentile_stretch_rgb(data: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    d = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(d)
    if not np.any(finite):
        z = np.zeros_like(d, dtype=np.float64)
    else:
        p_lo = float(np.clip(p_lo, 0.001, 49.999))
        p_hi = float(np.clip(p_hi, p_lo + 0.01, 99.999))
        lo, hi = np.nanpercentile(d[finite], (p_lo, p_hi))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi) or hi <= lo:
            hi = lo + 1e-6
        z = np.clip((d - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    z = np.where(np.isfinite(z), z, 0.0)
    u8 = (z * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def build_starfield_qa_png_mapping(
    raw: np.ndarray,
    hdr: Any,
    mapped_df: pd.DataFrame | None,
    *,
    max_side: int,
    mark_r: float,
    show_labels: bool,
    invert: bool,
    stretch_lo: float,
    stretch_hi: float,
    crosshair: bool,
    overlay_field_cat: bool,
    field_cat_path: Path | None,
    field_cat_mtime: float | None = None,
    show_dao: bool = True,
    show_gaia: bool = True,
    show_match: bool = True,
    dao_match_xy_source: str = "reproj",
    vsx_chip_df: pd.DataFrame | None = None,
) -> tuple[bytes, float, float, str]:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.wcs import FITSFixedWarning, WCS
    from PIL import Image, ImageDraw, ImageFont

    note = ""
    try:
        with catch_warnings():
            simplefilter("ignore", FITSFixedWarning)
            wcs_plot = WCS(hdr)
        if not getattr(wcs_plot, "has_celestial", False):
            wcs_plot = None
    except Exception:  # noqa: BLE001
        wcs_plot = None

    disp, scx, scy = downsample_array_2d(raw, int(max_side))
    rgb = percentile_stretch_rgb(disp, stretch_lo, stretch_hi)
    if invert:
        rgb = 255 - rgb
    im = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(im)
    h, w = disp.shape[:2]

    if crosshair:
        cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
        fill_c = (40, 40, 40) if not invert else (220, 220, 220)
        draw.line([(cx, 0), (cx, h - 1)], fill=fill_c, width=1)
        draw.line([(0, cy), (w - 1, cy)], fill=fill_c, width=1)

    _ = field_cat_mtime
    if show_gaia and overlay_field_cat and field_cat_path is not None and field_cat_path.is_file() and wcs_plot is not None:
        try:
            fdf_all = pd.read_csv(field_cat_path)
            if "ra_deg" in fdf_all.columns and "dec_deg" in fdf_all.columns and len(fdf_all) > 0:
                fdf = fdf_all
                coo = SkyCoord(
                    ra=fdf["ra_deg"].astype(float).values * u.deg,
                    dec=fdf["dec_deg"].astype(float).values * u.deg,
                    frame="icrs",
                )
                xp, yp = wcs_plot.world_to_pixel(coo)
                rb = max(0.8, float(mark_r) * 0.35)
                outline = (60, 100, 200) if invert else (90, 140, 255)
                fill = (70, 130, 230) if invert else (120, 170, 255)
                n_drawn = 0
                for xi, yi in zip(xp, yp):
                    if not (np.isfinite(xi) and np.isfinite(yi)):
                        continue
                    sx, sy = float(xi) * scx, float(yi) * scy
                    rp = max(0.9, rb * 0.55)
                    draw.ellipse((sx - rp, sy - rp, sx + rp, sy + rp), fill=fill, outline=fill, width=1)
                    draw.ellipse((sx - rb, sy - rb, sx + rb, sy + rb), outline=outline, width=1)
                    n_drawn += 1
                note = f"Katalógový kužeľ: {n_drawn} bodov. "
        except Exception as exc:  # noqa: BLE001
            note = f"Katalógový overlay: {exc}. "

    if wcs_plot is not None:
        try:
            r_deg = float(hdr.get("VY_GAIR", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            r_deg = 0.0
        if math.isfinite(r_deg) and r_deg > 0:
            try:
                ra0 = float(wcs_plot.wcs.crval[0])
                de0 = float(wcs_plot.wcs.crval[1])
                c0 = SkyCoord(ra=ra0 * u.deg, dec=de0 * u.deg, frame="icrs")
                c1 = SkyCoord(ra=(ra0 + r_deg) * u.deg, dec=de0 * u.deg, frame="icrs")
                x0, y0 = wcs_plot.world_to_pixel(c0)
                x1, y1 = wcs_plot.world_to_pixel(c1)
                if np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1):
                    rp = float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)))
                    sx0, sy0 = float(x0) * scx, float(y0) * scy
                    rr = float(rp) * float(scx)
                    outline = (60, 100, 200) if invert else (90, 140, 255)
                    draw.ellipse((sx0 - rr, sy0 - rr, sx0 + rr, sy0 + rr), outline=outline, width=2)
                    note = note + f"Gaia query r≈{r_deg:.2f}° (VY_GAIR). "
            except Exception:  # noqa: BLE001
                pass

    if mapped_df is not None and not mapped_df.empty and "x" in mapped_df.columns and "y" in mapped_df.columns:
        dfx = mapped_df.copy()
        _src = str(dao_match_xy_source or "reproj").strip().lower()
        if _src == "measured" and "x_meas" in dfx.columns and "y_meas" in dfx.columns:
            xs = pd.to_numeric(dfx["x_meas"], errors="coerce").astype(float).values * scx
            ys = pd.to_numeric(dfx["y_meas"], errors="coerce").astype(float).values * scy
        else:
            xs = dfx["x"].astype(float).values * scx
            ys = dfx["y"].astype(float).values * scy
        matched = (
            dfx["matched"].fillna(False).astype(bool).values if "matched" in dfx.columns else np.zeros(len(dfx), bool)
        )
        r = float(mark_r)
        for i in range(len(dfx)):
            x, y = float(xs[i]), float(ys[i])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if show_dao:
                red = (230, 60, 60) if not invert else (255, 110, 110)
                draw.ellipse((x - r, y - r, x + r, y + r), outline=red, width=2)
            if show_match and matched[i]:
                green = (20, 160, 40) if not invert else (0, 200, 80)
                draw.ellipse((x - r, y - r, x + r, y + r), outline=green, width=2)

        if show_labels and "name" in dfx.columns:
            try:
                font = ImageFont.load_default()
            except Exception:  # noqa: BLE001
                font = None
            for i in range(len(dfx)):
                x, y = float(xs[i]), float(ys[i])
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                sid = str(dfx.get("catalog_id", pd.Series([""] * len(dfx))).iloc[i]).strip()
                mg = dfx.get("mag", pd.Series([np.nan] * len(dfx))).iloc[i]
                mg_s = f"{float(mg):.2f}" if pd.notna(mg) else ""
                label = (sid or str(dfx["name"].iloc[i])[:14]).strip()
                if mg_s:
                    label = f"{label} m{mg_s}"
                fill = (240, 240, 0) if matched[i] else (255, 170, 170)
                if font:
                    draw.text((x + r + 1, y - r), label, fill=fill, font=font)
                else:
                    draw.text((x + r + 1, y - r), label, fill=fill)

        if str(dao_match_xy_source or "").strip().lower() == "measured":
            note = note + "DAO/MATCH: merané x,y z CSV. "

    if vsx_chip_df is not None and not vsx_chip_df.empty and "x" in vsx_chip_df.columns and "y" in vsx_chip_df.columns:
        vdf = vsx_chip_df
        xs_v = pd.to_numeric(vdf["x"], errors="coerce").astype(float).values * scx
        ys_v = pd.to_numeric(vdf["y"], errors="coerce").astype(float).values * scy
        s = max(2.0, float(mark_r) * 0.85)
        outline = (255, 210, 0) if not invert else (255, 235, 120)
        n_v = 0
        for i in range(len(vdf)):
            x, y = float(xs_v[i]), float(ys_v[i])
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            draw.rectangle((x - s, y - s, x + s, y + s), outline=outline, width=2)
            n_v += 1
        if n_v:
            note = note + f"VSX: {n_v} značiek (žltý štvorec). "

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), float(scx), float(scy), note


def build_msqa_vsx_plotly_figure(
    rgb: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    names: list[str],
    mag_labels: list[str],
    var_types: list[str],
) -> Any:
    """Plotly: podkladový snímok + žlté štvorce VSX s hoverom (meno, magnitúda, typ)."""
    import plotly.graph_objects as go

    rgb_u8 = np.asarray(rgb, dtype=np.uint8)
    if rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8")
    h, w = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])

    fig = go.Figure()
    fig.add_trace(
        go.Image(
            z=rgb_u8,
            hoverinfo="skip",
            colormodel="rgb",
        )
    )
    n = int(len(xs))
    if n > 0:
        cd = np.column_stack(
            [
                np.asarray(names, dtype=object),
                np.asarray(mag_labels, dtype=object),
                np.asarray(var_types, dtype=object),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=np.asarray(xs, dtype=float),
                y=np.asarray(ys, dtype=float),
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=11,
                    color="rgba(255,220,0,0.75)",
                    line=dict(width=1.5, color="darkorange"),
                ),
                customdata=cd,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Magnitúda: %{customdata[1]}<br>"
                    "Typ premennosti: %{customdata[2]}<extra></extra>"
                ),
                name="VSX",
            )
        )

    fig.update_layout(
        title=dict(text="VSX — hover myšou nad žltým štvorcom", font=dict(size=14)),
        margin=dict(l=4, r=4, t=40, b=4),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.5, w - 0.5],
            constrain="domain",
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[h - 0.5, -0.5],
            scaleanchor="x",
            scaleratio=1,
        ),
        showlegend=False,
        hovermode="closest",
    )
    return fig


def msqa_prepare_vsx_plotly_series(
    vsx_filt: pd.DataFrame,
    scx: float,
    scy: float,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Z VSX tabuľky (stĺpce x,y v plnom rozlíšení snímku) urob série pre Plotly."""
    if vsx_filt is None or vsx_filt.empty:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            [],
            [],
            [],
        )
    xv = pd.to_numeric(vsx_filt["x"], errors="coerce").astype(float).to_numpy() * float(scx)
    yv = pd.to_numeric(vsx_filt["y"], errors="coerce").astype(float).to_numpy() * float(scy)
    ok = np.isfinite(xv) & np.isfinite(yv)
    mm = pd.to_numeric(vsx_filt.get("mag_max", pd.Series(np.nan, index=vsx_filt.index)), errors="coerce")
    mn = pd.to_numeric(vsx_filt.get("mag_min", pd.Series(np.nan, index=vsx_filt.index)), errors="coerce")
    oid_s = (
        vsx_filt["oid"].astype(str).str.strip()
        if "oid" in vsx_filt.columns
        else pd.Series([""] * len(vsx_filt), index=vsx_filt.index)
    )
    nam_s = (
        vsx_filt["name"].fillna("").astype(str).str.strip()
        if "name" in vsx_filt.columns
        else pd.Series([""] * len(vsx_filt), index=vsx_filt.index)
    )
    vt_s = (
        vsx_filt["var_type"].fillna("").astype(str).str.strip()
        if "var_type" in vsx_filt.columns
        else pd.Series([""] * len(vsx_filt), index=vsx_filt.index)
    )
    xs_l: list[float] = []
    ys_l: list[float] = []
    names: list[str] = []
    mag_labels: list[str] = []
    var_types: list[str] = []
    for i in range(len(vsx_filt)):
        if not bool(ok[i]):
            continue
        xs_l.append(float(xv[i]))
        ys_l.append(float(yv[i]))
        nam = str(nam_s.iloc[i])
        oid = str(oid_s.iloc[i])
        names.append(nam if nam else (oid if oid else "—"))
        mmf, mnf = mm.iloc[i], mn.iloc[i]
        if pd.notna(mmf) and pd.notna(mnf):
            mag_labels.append(f"max {float(mmf):.2f}, min {float(mnf):.2f}")
        elif pd.notna(mmf):
            mag_labels.append(f"max {float(mmf):.2f}")
        elif pd.notna(mnf):
            mag_labels.append(f"min {float(mnf):.2f}")
        else:
            mag_labels.append("—")
        vt = str(vt_s.iloc[i])
        var_types.append(vt if vt else "—")
    return (
        np.asarray(xs_l, dtype=float),
        np.asarray(ys_l, dtype=float),
        names,
        mag_labels,
        var_types,
    )
