"""Field HRD view of the Aperture Photometry tab (HRD-TAB-2026-10, Milan 2026-10-18).

* HR diagram in three variants: colour-magnitude, temperature-luminosity, distance; stars without a
  reliable parallax in their own panel (apparent G is not comparable with M_G);
* one table of the notable stars (classification, SIMBAD identity, physics, distance, variability,
  photometry status);
* the field image: inverted, catalogue colour (vivid) or physical blackbody, with hover on every star.

All plotly. A notable star picked in the diagram, the table or the image is highlighted everywhere.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from hrd_field_render import FIELD_MODES, bp_rp_css, field_image_jpeg, teff_from_bp_rp

if TYPE_CHECKING:
    from config import AppConfig

logger = logging.getLogger(__name__)

VARIANTS = {
    "cmd": ("A \u00b7 Colour\u2013magnitude",
            "Classic CMD: BP\u2212RP against M_G, field coloured by BP\u2212RP, spectral classes on the top band."),
    "teff": ("B \u00b7 Temperature\u2013luminosity",
             "The HR diagram proper: Teff from BP\u2212RP (hot on the left, Pecaut & Mamajek 2013) against M_G."),
    "dist": ("C \u00b7 Distance",
             "The CMD coloured by distance (log pc) and sized by apparent brightness: which stars are near, which far."),
}
MODES = {
    "inverted": ("Inverted (print)", "Black stars on white: the MASTERSTAR with an asinh stretch, inverted."),
    "vivid": ("Catalogue colour \u00b7 vivid",
              "Luminance from the MASTERSTAR, colour from Gaia BP\u2212RP with a hue-preserving stretch (Lupton et al. 2004): "
              "bright stars keep their colour, the sky stays neutral."),
    "physical": ("Physical blackbody", "Same method with the true blackbody colour of each star's Teff (D65 white): honest, pale."),
}
SPECTRAL_BANDS = (("O/B", -0.45, 0.0), ("A", 0.0, 0.38), ("F", 0.38, 0.73), ("G", 0.73, 0.98), ("K", 0.98, 1.84), ("M", 1.84, 6.2))
_PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def category_color(cat: str) -> str:
    c = str(cat)
    for key, col in (("Very cool", "#ff5a36"), ("Red supergiant", "#ff9a1f"), ("Red giant", "#ffc861"),
                     ("White dwarf", "#a8c8ff"), ("Very hot", "#5b8cff"), ("Hot", "#5b8cff")):
        if c.startswith(key):
            return col
    return "#7ee2a8"


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _fmt(v: Any, nd: int) -> str:
    x = _num(v)
    return f"{x:.{nd}f}" if math.isfinite(x) else "-"


def _stamp(*paths: Path) -> tuple:
    return tuple((str(p), p.stat().st_mtime_ns if p.is_file() else 0) for p in paths)


@st.cache_resource(show_spinner=False, max_entries=4)
def _hrd_data(photometry_s: str, gaia_db_s: str, pmin: float, psnr: float, stamp_key: tuple, cfg_key: str,
              _cfg: Any) -> dict[str, Any]:
    """HRD table, notable stars, VSX names and measured targets of one setup (loaded once per file state:
    ``stamp_key`` is hashed, G14-27; ``cfg_key`` = ``ui_cache.config_fingerprint``, NEW-3) and the hover
    texts of every star (built once, not per rerun). Read-only (NEW-1): the online enrichment is read
    from ``_hrd_cache/hrd_enrich.json`` (written by the night run's report or the lookup job)."""
    _ = cfg_key
    from hrd_analysis import build_hrd_dataframe, get_top_interesting_stars  # noqa: PLC0415

    phot = Path(photometry_s)
    ps = phot.parent
    ms_csv = ps / "masterstars_full_match.csv"
    df = build_hrd_dataframe(ms_csv, Path(gaia_db_s), parallax_min_mas=pmin, parallax_snr_min=psnr)
    df["catalog_id"] = df["catalog_id"].astype(str)
    for c in ("bp_rp", "abs_mag_g", "phot_g_mean_mag", "teff_gspphot", "logg_gspphot", "parallax", "parallax_over_error", "x", "y"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    vt = ps / "variable_targets.csv"
    if vt.is_file():
        v = pd.read_csv(vt, dtype={"catalog_id": str}, low_memory=False)
        v = v[[c for c in ("catalog_id", "vsx_name", "vsx_type") if c in v]].drop_duplicates("catalog_id")
        df = df.merge(v, on="catalog_id", how="left")
    for c in ("vsx_name", "vsx_type"):
        if c not in df:
            df[c] = np.nan
    sm = phot / "photometry_summary.csv"
    measured = set(pd.read_csv(sm, dtype={"catalog_id": str})["catalog_id"]) if sm.is_file() else set()
    df["measured"] = df["catalog_id"].isin(measured)
    df["reliable"] = df["hrd_reliable"].astype(bool)
    top = get_top_interesting_stars(df, cfg=_cfg, cache_path=ps / "_hrd_cache" / "hrd_enrich.json", fetch_online=False)
    if not top.empty and "_empty_field" in top:
        top = top[~top["_empty_field"].astype(bool)]
    if not top.empty:
        top = top.copy()
        top["catalog_id"] = top["catalog_id"].astype(str)
        top = top.merge(df[["catalog_id", "vsx_name", "vsx_type", "measured"]], on="catalog_id", how="left")
    return {"df": df, "top": top.reset_index(drop=True) if not top.empty else top,
            "hover": _texts(df, _hover), "side_hover": _texts(df, _side_hover)}


def _hover(r: pd.Series) -> str:
    head = f"<b>{r['vsx_name']}</b>" + (f" \u00b7 {r['vsx_type']}" if pd.notna(r.get("vsx_type")) else "") + f"<br>Gaia DR3 {r['catalog_id']}" \
        if pd.notna(r.get("vsx_name")) else f"<b>Gaia DR3 {r['catalog_id']}</b>"
    teff = float(teff_from_bp_rp(r["bp_rp"])[0]) if math.isfinite(r["bp_rp"]) else float("nan")
    out = (f"{head}<br>G {_fmt(r['phot_g_mean_mag'], 2)} \u00b7 M_G {_fmt(r['abs_mag_g'], 2)} \u00b7 BP\u2212RP {_fmt(r['bp_rp'], 2)}"
           f"<br>Teff \u2248 {_fmt(teff, 0)} K")
    if math.isfinite(_num(r.get("teff_gspphot"))):
        out += f" (Gaia GSP-Phot {_fmt(r['teff_gspphot'], 0)} K)"
    if math.isfinite(_num(r.get("parallax"))) and r["parallax"] > 0:
        out += f"<br>distance {1000 / r['parallax']:.0f} pc (\u03d6/\u03c3 {_fmt(r.get('parallax_over_error'), 1)})"
    if bool(r.get("measured")):
        out += "<br>measured target"
    return out


def _texts(df: pd.DataFrame, fn: Any) -> pd.Series:
    """Hover text of every star (G14-27: built once in the cached loader, not on every rerun)."""
    return pd.Series([fn(r) for _, r in df.iterrows()], index=df.index, dtype=object)


def _hover_of(data: dict[str, Any], rows: pd.DataFrame) -> list[str]:
    hov = data.get("hover")
    if hov is None:  # a data dict not built by _hrd_data
        hov = _texts(data["df"], _hover)
    return hov.loc[rows.index].tolist()


def _side_hover(r: pd.Series) -> str:
    return (f"Gaia DR3 {r['catalog_id']}<br>G {r['phot_g_mean_mag']:.2f} \u00b7 BP\u2212RP {r['bp_rp']:.2f}<br>"
            + (f"\u03d6 {r['parallax']:.2f} mas (S/N {_fmt(r.get('parallax_over_error'), 1)})"
               if math.isfinite(_num(r.get("parallax"))) else "no parallax"))


def hrd_figure(data: dict[str, Any], variant: str, opts: set[str], sel: str | None) -> go.Figure:
    df, top = data["df"], data["top"]
    rel = df[df["reliable"] & np.isfinite(df["bp_rp"]) & np.isfinite(df["abs_mag_g"])]
    tx = (lambda b: teff_from_bp_rp(b)) if variant == "teff" else (lambda b: np.asarray(b, float))
    fig = go.Figure()
    hov = _hover_of(data, rel)
    if variant == "dist":
        dist = np.where(rel["parallax"] > 0, np.log10(1000.0 / rel["parallax"].clip(lower=1e-6)), np.nan)
        size = np.clip(15.0 - rel["phot_g_mean_mag"].fillna(15.0), 3, 11)
        marker = dict(size=size, color=dist, colorscale="Viridis", showscale=True, opacity=0.85,
                      colorbar=dict(title=dict(text="log d [pc]"), thickness=12, len=0.6))
    else:
        marker = dict(size=5, color=[bp_rp_css(b) for b in rel["bp_rp"]], opacity=0.85)
    fig.add_trace(go.Scattergl(x=tx(rel["bp_rp"]), y=rel["abs_mag_g"], mode="markers", marker=marker, text=hov,
                               hovertemplate="%{text}<extra></extra>"))
    if "vsx" in opts:
        v = rel[rel["vsx_name"].notna()]
        fig.add_trace(go.Scatter(x=tx(v["bp_rp"]), y=v["abs_mag_g"], mode="markers", text=_hover_of(data, v),
                                 marker=dict(size=10, symbol="diamond-open", color="#4c8dff", line=dict(width=1.6)),
                                 hovertemplate="%{text}<extra>VSX variable</extra>"))
    if "meas" in opts:
        m = rel[rel["measured"]]
        fig.add_trace(go.Scatter(x=tx(m["bp_rp"]), y=m["abs_mag_g"], mode="markers", text=_hover_of(data, m),
                                 marker=dict(size=13, symbol="circle-open", color="#9aa0ab", line=dict(width=1)),
                                 hovertemplate="%{text}<extra>measured target</extra>"))
    ann = []
    for _, r in top.iterrows():
        b, mg = _num(r.get("bp_rp")), _num(r.get("abs_mag_g"))
        if not (math.isfinite(b) and math.isfinite(mg)):
            continue
        x = float(tx([b])[0])
        col, on = category_color(r["category"]), r["catalog_id"] == sel
        name = str(r.get("simbad_main_id") or f"Gaia DR3 {r['catalog_id']}")
        fig.add_trace(go.Scatter(
            x=[x], y=[mg], customdata=[r["catalog_id"]], mode="markers",
            marker=dict(size=22 if on else 16, color="rgba(0,0,0,0)", line=dict(color=col, width=3.5 if on else 2.5)),
            hovertemplate=f"<b>{name}</b><br>{r['category']} \u00b7 {r['ident']}<br>G {_fmt(r.get('mag_g'), 2)} \u00b7 M_G {mg:.2f} \u00b7 "
                          f"BP\u2212RP {b:.2f}<extra>click to select</extra>"))
        if "labels" in opts:
            ann.append(dict(x=math.log10(x) if variant == "teff" else x, y=mg, text=name.replace("V* ", ""), showarrow=True,
                            arrowhead=0, arrowcolor=col, ax=34, ay=-26, font=dict(color=col, size=12), bgcolor="rgba(14,17,23,0.75)"))
    shapes = []
    if variant != "teff":
        for lab, a, b in SPECTRAL_BANDS:
            shapes.append(dict(type="rect", xref="x", yref="paper", x0=a, x1=b, y0=1.0, y1=1.045,
                               fillcolor=bp_rp_css((a + min(b, 3.0)) / 2), line=dict(width=0)))
            ann.append(dict(x=(a + min(b, 2.6)) / 2, y=1.022, yref="paper", text=f"<b>{lab}</b>", showarrow=False,
                            font=dict(size=12, color="#111111")))
        regions = ((0.55, 4.6, "main sequence"), (1.45, -0.2, "red giant branch"), (0.1, 12.8, "white dwarfs"))
        xa = dict(title="BP \u2212 RP [mag]", range=[-0.6, max(3.5, float(np.nanmax(rel["bp_rp"])) + 0.3) if len(rel) else 3.5])
    else:
        regions = ((math.log10(6500), 4.6, "main sequence"), (math.log10(4300), 0.3, "red giants"), (math.log10(9000), 12.6, "white dwarfs"))
        ticks = [30000, 20000, 10000, 7000, 5000, 4000, 3000]
        xa = dict(title="effective temperature Teff [K]", type="log", autorange="reversed", tickvals=ticks,
                  ticktext=[f"{t:,}".replace(",", " ") for t in ticks])
    for x, y, t in regions:
        ann.append(dict(x=x, y=y, text=t, showarrow=False, font=dict(size=13, color="rgba(160,166,178,0.75)")))
    fig.update_layout(height=640, margin=dict(l=64, r=20, t=56, b=52), showlegend=False, hovermode="closest",
                      xaxis=xa, yaxis=dict(title="absolute magnitude M_G [mag]", autorange="reversed"),
                      shapes=shapes, annotations=ann, clickmode="event+select")
    return fig


def side_figure(df: pd.DataFrame, side_hover: pd.Series | None = None) -> go.Figure:
    un = df[~df["reliable"] & np.isfinite(df["bp_rp"]) & np.isfinite(df["phot_g_mean_mag"])]
    txt = (side_hover if side_hover is not None else _texts(df, _side_hover)).loc[un.index].tolist()
    fig = go.Figure(go.Scatter(x=un["bp_rp"], y=un["phot_g_mean_mag"], mode="markers", text=txt, hovertemplate="%{text}<extra></extra>",
                               marker=dict(size=6, color=[bp_rp_css(b) for b in un["bp_rp"]], opacity=0.85)))
    fig.update_layout(height=612, margin=dict(l=48, r=10, t=40, b=52), showlegend=False,
                      xaxis_title="BP \u2212 RP", yaxis=dict(title="apparent G [mag]", autorange="reversed"))
    return fig


def image_figure(jpg: Path, data: dict[str, Any], marks: set[str], sel: str | None, *, light: bool, size: tuple[int, int]) -> go.Figure:
    w, h = size
    df, top = data["df"], data["top"]
    src = "data:image/jpeg;base64," + base64.b64encode(Path(jpg).read_bytes()).decode("ascii")
    pts = df[np.isfinite(df["x"]) & np.isfinite(df["y"])]
    fig = go.Figure(go.Scattergl(x=pts["x"], y=pts["y"], mode="markers", marker=dict(size=10, color="rgba(0,0,0,0)"),
                                 text=_hover_of(data, pts), hovertemplate="%{text}<extra></extra>"))
    if "vsx" in marks:
        v = pts[pts["vsx_name"].notna()]
        fig.add_trace(go.Scatter(x=v["x"], y=v["y"], mode="markers", hoverinfo="skip",
                                 marker=dict(size=18, symbol="diamond-open", color="#2563eb" if light else "#4c8dff", line=dict(width=2))))
    ann = []
    if "notable" in marks:
        for _, r in top.iterrows():
            x, y = _num(r.get("x_px")), _num(r.get("y_px"))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            col = "#c2410c" if light else category_color(r["category"])
            on = r["catalog_id"] == sel
            name = str(r.get("simbad_main_id") or f"Gaia DR3 {r['catalog_id']}")
            fig.add_trace(go.Scatter(x=[x], y=[y], customdata=[r["catalog_id"]], mode="markers",
                                     marker=dict(size=36 if on else 26, symbol="circle-open", color=col, line=dict(width=4 if on else 2.5)),
                                     hovertemplate=f"<b>{name}</b><br>{r['category']}<extra>click to select</extra>"))
            ann.append(dict(x=x, y=y, text=name.replace("V* ", ""), showarrow=True, arrowhead=0, arrowcolor=col, ax=40, ay=-34,
                            font=dict(color=col, size=13), bgcolor="rgba(255,255,255,0.85)" if light else "rgba(14,17,23,0.8)"))
    fig.update_layout(height=820, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, dragmode="pan",
                      plot_bgcolor="#ffffff" if light else "#000000", annotations=ann, clickmode="event+select",
                      xaxis=dict(range=[0, w], visible=False, constrain="domain"),
                      yaxis=dict(range=[h, 0], visible=False, scaleanchor="x", scaleratio=1),
                      images=[dict(source=src, xref="x", yref="y", x=0, y=0, sizex=w, sizey=h, sizing="stretch", layer="below")])
    return fig


def notable_table(top: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in top.iterrows():
        teff = _num(r.get("teff"))
        if not math.isfinite(teff) and math.isfinite(_num(r.get("bp_rp"))):
            teff = round(float(teff_from_bp_rp(r["bp_rp"])[0]), -1)
        rows.append({
            "Category": str(r["category"]).split(" (")[0],
            "Status": str(r["ident"]) + (f" \u00b7 {r['ident_detail']}" if str(r.get("ident_detail") or "") not in ("", "None", "nan") else ""),
            "Star": str(r.get("simbad_main_id") or f"Gaia DR3 {r['catalog_id']}"),
            "Spectral / object type": " \u00b7 ".join(str(x) for x in (r.get("sp_type_raw"), r.get("otype_raw")) if pd.notna(x) and str(x) not in ("", "None")),
            "Gaia DR3": f"https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=I/355/gaiadr3&Source={r['catalog_id']}",
            "G": _num(r.get("mag_g")), "M_G": _num(r.get("abs_mag_g")), "BP-RP": _num(r.get("bp_rp")),
            "Teff [K]": teff, "log g": _num(r.get("logg")),
            "Distance [pc]": _num(r.get("dist_pc")), "\u03d6/\u03c3": _num(r.get("parallax_snr")), "A_G": _num(r.get("ag_gspphot")),
            "Variable (VSX)": (f"{r['vsx_name']} ({r['vsx_type']})" if pd.notna(r.get("vsx_type")) else str(r["vsx_name"])) if pd.notna(r.get("vsx_name")) else "",
            "Photometry": "light curve" if bool(r.get("measured")) else "not measured",
            "RA / Dec": str(r.get("ra_dec_sex") or ""),
        })
    return pd.DataFrame(rows)


def _pick_from(event: Any) -> str | None:
    try:
        pts = list(event.selection.points)
    except AttributeError:
        return None
    for p in pts:
        cd = p.get("customdata")
        if cd:
            return str(cd[0] if isinstance(cd, (list, tuple)) else cd)
    return None


def _select(key: str, cid: str | None, sel_key: str) -> None:
    """Apply a widget selection once (selections persist across reruns)."""
    if cid is None or st.session_state.get(key) == cid:
        return
    st.session_state[key] = cid
    st.session_state[sel_key] = None if st.session_state.get(sel_key) == cid else cid
    st.rerun()


def render_hrd_tab(photometry_dir: Path, cfg: "AppConfig | None") -> None:
    from hrd_analysis import HRD_EMPTY_FIELD_MSG, HRD_PARALLAX_MIN_MAS, HRD_PARALLAX_SNR_MIN, _resolve_masterstar_fits_path  # noqa: PLC0415
    from platesolve_ui_paths import masterstar_fwhm_px  # noqa: PLC0415

    phot = Path(photometry_dir)
    ps = phot.parent
    ms_csv = ps / "masterstars_full_match.csv"
    if not ms_csv.is_file():
        st.warning("**masterstars_full_match.csv** not found (expected in `platesolve/<setup>/`).")
        return
    gdb = Path(str(getattr(cfg, "gaia_db_path", "") or "").strip()) if cfg is not None else Path("")
    if not gdb.is_file():
        st.warning("Set **Gaia DB path** in Settings (`gaia_db_path`): without it there is no parallax, Teff or log g.")
        return
    pmin, psnr = HRD_PARALLAX_MIN_MAS, HRD_PARALLAX_SNR_MIN
    from ui_cache import config_fingerprint  # noqa: PLC0415

    with st.spinner("Loading the HR diagram..."):
        data = _hrd_data(str(phot), str(gdb), float(pmin), float(psnr),
                         _stamp(ms_csv, gdb, ps / "_hrd_cache" / "hrd_enrich.json", ps / "variable_targets.csv", phot / "photometry_summary.csv"),
                         config_fingerprint(cfg), cfg)
    df, top = data["df"], data["top"]
    if df.empty:
        st.info("Empty star catalogue.")
        return
    og = ps.name
    sel_key = f"hrd_sel_{og}"
    sel = st.session_state.get(sel_key)

    n_rel = int(df["reliable"].sum())
    n_conf = int((top["ident"] == "confirmed").sum()) if not top.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stars on the frame", f"{len(df)}", "DAO detected, matched to Gaia DR3", delta_color="off")
    c2.metric("Reliable parallax", f"{n_rel}", "absolute magnitude M_G known", delta_color="off")
    c3.metric("Without distance", f"{len(df) - n_rel}", "apparent G only (side panel)", delta_color="off")
    c4.metric("Notable stars", f"{len(top)}", f"{n_conf} confirmed by SIMBAD, {len(top) - n_conf} candidates", delta_color="off")

    st.subheader("Hertzsprung\u2013Russell diagram of the field")
    variant = st.segmented_control("Diagram", list(VARIANTS), format_func=lambda k: VARIANTS[k][0], default="cmd",
                                   key=f"hrd_variant_{og}", label_visibility="collapsed") or "cmd"
    st.caption(VARIANTS[variant][1])
    opt_labels = {"vsx": "VSX variables", "meas": "Measured targets", "labels": "Labels of notable stars", "side": "Stars without distance"}
    picked = st.pills("Show", list(opt_labels), format_func=lambda k: opt_labels[k], selection_mode="multi",
                      default=list(opt_labels), key=f"hrd_opts_{og}", label_visibility="collapsed") or []
    opts = set(picked)
    cols = st.columns([4, 1.3]) if "side" in opts else [st.container()]
    with cols[0]:
        ev = st.plotly_chart(hrd_figure(data, variant, opts, sel), width="stretch", config=_PLOT_CFG,
                             on_select="rerun", selection_mode="points", key=f"hrd_plot_{og}_{variant}")
        _select(f"hrd_pick_plot_{og}_{variant}", _pick_from(ev), sel_key)
    if "side" in opts:
        with cols[1]:
            st.caption("No reliable parallax: colour vs apparent G (not comparable with M_G)")
            st.plotly_chart(side_figure(df, data["side_hover"]), width="stretch", config=_PLOT_CFG, key=f"hrd_side_{og}")
    st.caption("Positions use Gaia DR3 catalogue values (BP\u2212RP; M_G from the parallax). Field stars use the BP\u2212RP palette of the "
               "colour field image. Hover any star; click a notable star (or a table row) to highlight it everywhere.")

    st.subheader("Notable stars")
    if cfg is not None and bool(getattr(cfg, "hrd_online_enrich_enabled", True)):
        from ui_session import any_job_running, start_task_job  # noqa: PLC0415

        _have = (ps / "_hrd_cache" / "hrd_enrich.json").is_file()
        st.caption("SIMBAD / Gaia lookups of the notable stars: " + ("from the last lookup." if _have else
                   "none yet - the tab never queries online by itself."))
        if st.button("Look up the notable stars online (Gaia TAP, SIMBAD)", key=f"hrd_enrich_{og}",
                     disabled=any_job_running(cfg.data_root)):
            # NEW-1: an explicit job (task_jobs.task_hrd_enrich), never on render
            start_task_job(cfg.data_root, "hrd_enrich", {"draft_dir": str(ps.parent.parent), "setup": og},
                           label=f"HRD lookup {og}")
    if top.empty:
        st.info(HRD_EMPTY_FIELD_MSG)
    else:
        tab = notable_table(top)
        tev = st.dataframe(
            tab, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key=f"hrd_table_{og}",
            column_config={
                "Gaia DR3": st.column_config.LinkColumn("Gaia DR3", display_text=r"Source=(\d+)"),
                "G": st.column_config.NumberColumn(format="%.2f"), "M_G": st.column_config.NumberColumn(format="%.2f"),
                "BP-RP": st.column_config.NumberColumn(format="%.2f"), "Teff [K]": st.column_config.NumberColumn(format="%.0f"),
                "log g": st.column_config.NumberColumn(format="%.2f"), "Distance [pc]": st.column_config.NumberColumn(format="%.0f"),
                "\u03d6/\u03c3": st.column_config.NumberColumn(format="%.0f"), "A_G": st.column_config.NumberColumn(format="%.2f"),
            })
        try:
            rows = list(tev.selection.rows)
        except AttributeError:
            rows = []
        if rows:
            _select(f"hrd_pick_table_{og}", str(top.iloc[rows[0]]["catalog_id"]), sel_key)
        st.caption("Confirmed = the SIMBAD spectral type agrees with the category; candidate = only Gaia photometry or "
                   "astrophysical parameters point to it. Distance = 1/parallax (not corrected for the Gaia zero point).")

    st.subheader("Field image")
    mode = st.segmented_control("Image", list(FIELD_MODES), format_func=lambda k: MODES[k][0], default="vivid",
                                key=f"hrd_mode_{og}", label_visibility="collapsed") or "vivid"
    st.caption(MODES[mode][1])
    mark_labels = {"notable": "Mark notable stars", "vsx": "Mark VSX variables"}
    marks = set(st.pills("Marks", list(mark_labels), format_func=lambda k: mark_labels[k], selection_mode="multi",
                         default=["notable"], key=f"hrd_marks_{og}", label_visibility="collapsed") or [])
    fits_path = _resolve_masterstar_fits_path(ps)
    if fits_path is None:
        st.info("MASTERSTAR.fits not found: no field image.")
        return
    jpg = _field_jpeg(str(fits_path), str(_render_cache_dir(ps)), mode, _stamp(fits_path, ms_csv), data,
                      masterstar_fwhm_px(fits_path))
    from PIL import Image  # noqa: PLC0415

    with Image.open(jpg) as im:
        size = im.size
    iev = st.plotly_chart(image_figure(jpg, data, marks, sel, light=mode == "inverted", size=size), width="stretch",
                          config={**_PLOT_CFG, "scrollZoom": True}, on_select="rerun", selection_mode="points", key=f"hrd_img_{og}_{mode}")
    _select(f"hrd_pick_img_{og}_{mode}", _pick_from(iev), sel_key)
    st.caption("Hover any star of the frame for its Gaia data; zoom with the mouse wheel. Catalogue colours are Gaia BP\u2212RP, "
               "not measured by this camera, and include interstellar reddening.")


def _render_cache_dir(platesolve_dir: Path) -> Path:
    """NEW-1: the field images are a render cache of this computer (temp folder), never written into the
    processed field."""
    import tempfile  # noqa: PLC0415

    tag = hashlib.md5(str(Path(platesolve_dir).resolve()).encode()).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / "vyvar_hrd_render" / tag


def _field_jpeg(fits_s: str, cache_s: str, mode: str, stamp: tuple, data: dict[str, Any], fwhm_px: float) -> Path:
    """Field image of one mode, rendered once per MASTERSTAR / star list (file cache)."""
    cache = Path(cache_s)
    cache.mkdir(parents=True, exist_ok=True)
    tag = hashlib.md5(repr((stamp, round(fwhm_px, 3))).encode()).hexdigest()[:10]
    out = cache / f"field_{mode}_{tag}.jpg"
    if not out.is_file():
        for old in cache.glob(f"field_{mode}_*.jpg"):
            old.unlink(missing_ok=True)
        with st.spinner(f"Rendering the {MODES[mode][0].lower()} field image..."):
            field_image_jpeg(Path(fits_s), data["df"], mode, out, fwhm_px=fwhm_px)
    return out
