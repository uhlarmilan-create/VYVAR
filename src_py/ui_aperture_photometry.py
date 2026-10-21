"""Aperture Photometry tab (AP-TAB-2026-10, mock-up approved by Milan 2026-10-17).

UI-INFO-2026-10: informational only - no phase is started from here (runs: VAR-STREM / the
night runner, one code path). One target at a time, laid out like its page in the PDF
Summary Measure Report and built from the same data (target_page.TargetPages). The whole
observation group is loaded into memory once per draft, so switching targets is instant.
"""

from __future__ import annotations

import datetime
import html
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import streamlit as st

from mag_constants import MAG_ERR_SCALE
from platesolve_ui_paths import find_photometry_setups, setup_label
from utils import resolve_draft_dir_path

if TYPE_CHECKING:
    from config import AppConfig
    from target_page import TargetPages

_TRUST_COLOUR = {"GREEN": "#3fb950", "YELLOW": "#e3b341", "RED": "#f85149"}
_BLUE = "#3b82f6"
_RED = "#ef4444"
_DOT = "\u00b7"


def _latest_report_pdf(platesolve_setup_dir: Path) -> Path | None:
    """Newest PDF report of this setup (``VYVAR_report_<setup>_*.pdf`` or ``photometry/report_<setup>.pdf``)."""
    d = Path(platesolve_setup_dir)
    cands = sorted(d.glob(f"VYVAR_report_{d.name}_*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    p = d / "photometry" / f"report_{d.name}.pdf"
    return p if p.is_file() else None


def _measured_cids(pages: "TargetPages") -> list[str]:
    """Targets with a light curve, in report order (VSX type, then lc_rms)."""
    sd = pages.summary_df
    if "_cid" not in sd.columns:
        return []
    out = []
    for cid, nfr, rms in zip(sd["_cid"], pd.to_numeric(sd.get("n_frames"), errors="coerce"),
                             pd.to_numeric(sd.get("lc_rms"), errors="coerce")):
        if cid and nfr > 0 and math.isfinite(rms) and (pages.lc_dir / f"lightcurve_{cid}.csv").is_file():
            out.append(str(cid))
    return out


@st.cache_resource(show_spinner="Loading the draft into memory...", max_entries=4)
def _group_pages(platesolve_setup_dir: str, _stamp: float) -> "TargetPages":
    """All page data of one observation group, read once (``_stamp`` = summary mtime, part of the key)."""
    from target_page import load_target_pages  # noqa: PLC0415

    pages = load_target_pages(Path(platesolve_setup_dir))
    pages.preload(_measured_cids(pages))
    return pages


def _ut_ticks(x_rel: np.ndarray, jd: np.ndarray) -> tuple[list[float], list[str]]:
    """Full-hour UT ticks positioned on the relative BJD axis."""
    ok = np.isfinite(x_rel) & np.isfinite(jd)
    if int(ok.sum()) < 2:
        return [], []
    dj = float(np.median(jd[ok] - x_rel[ok]))
    lo, hi = float(np.nanmin(jd[ok])), float(np.nanmax(jd[ok]))
    vals, labs = [], []
    for h in range(math.ceil((lo + 0.5) * 24.0), math.floor((hi + 0.5) * 24.0) + 1):
        vals.append(h / 24.0 - 0.5 - dj)
        labs.append(f"{h % 24:02d}:00")
    return vals, labs


def _lc_figure(pages: "TargetPages", pg: dict[str, Any], *, show_airmass: bool, show_flagged: bool) -> Any:
    """Interactive light curve with the check-star strip (same content as the PDF figure)."""
    import plotly.graph_objects as go  # noqa: PLC0415
    from plotly.subplots import make_subplots  # noqa: PLC0415

    from lc_outlier import EXPORT_EXCLUDE_FLAGS  # noqa: PLC0415

    cid = pg["cid"]
    df = pages.lc_df(cid)
    uncal = pg["uncalibrated"]
    from photometry_lightcurve import published_mag_column  # noqa: PLC0415

    ycol = "delta_mag" if uncal else (published_mag_column(df) or "mag_calib")
    ylab = "delta_mag (not calibrated)" if uncal else (f"{pg['band']} [mag]" if pg["band"] else ycol)
    xcol = "bjd" if "bjd" in df.columns else "jd"
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(float)
    e = MAG_ERR_SCALE * pd.to_numeric(df["err"], errors="coerce").to_numpy(float) if "err" in df.columns else None
    n = len(df)
    fl = df["flag"].astype(str).str.strip().str.lower().to_numpy() if "flag" in df.columns else np.array(["normal"] * n)
    reason = df["flag_reason"].fillna("").astype(str).to_numpy() if "flag_reason" in df.columns else np.array([""] * n)
    ok = np.isfinite(x) & np.isfinite(y)
    x0 = float(np.floor(np.nanmin(x[ok]))) if ok.any() else 0.0
    xr = x - x0
    good = ok & ~np.isin(fl, list(EXPORT_EXCLUDE_FLAGS))
    bad = ok & ~good

    chk = pages.check_df(cid)
    has_chk = chk is not None and {"bjd", "kmag"} <= set(chk.columns) and len(chk) >= 2
    fig = make_subplots(rows=2 if has_chk else 1, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.77, 0.23] if has_chk else [1.0],
                        specs=[[{"secondary_y": True}]] + ([[{}]] if has_chk else []))
    fig.add_trace(go.Scatter(
        x=xr[good], y=y[good], mode="markers", name=f"exported ({int(good.sum())})",
        marker=dict(color=_BLUE, size=7), customdata=x[good],
        error_y=dict(array=e[good], color="rgba(59,130,246,0.45)", thickness=1.2, width=0) if e is not None else None,
        hovertemplate="BJD %{customdata:.5f}<br>" + ylab + " %{y:.4f}<extra></extra>",
    ), row=1, col=1, secondary_y=False)
    if show_flagged and bad.any():
        fig.add_trace(go.Scatter(
            x=xr[bad], y=y[bad], mode="markers", name=f"flagged, not exported ({int(bad.sum())})",
            marker=dict(color=_RED, size=8, symbol="x"),
            customdata=np.column_stack([x[bad], fl[bad], reason[bad]]),
            hovertemplate="BJD %{customdata[0]}<br>%{customdata[1]}: %{customdata[2]}<br>" + ylab
                          + " %{y:.4f}<extra></extra>",
        ), row=1, col=1, secondary_y=False)
    if show_airmass and "airmass" in df.columns:
        am = pd.to_numeric(df["airmass"], errors="coerce").to_numpy(float)
        oka = np.isfinite(am) & np.isfinite(x)
        order = np.argsort(xr[oka])
        fig.add_trace(go.Scatter(
            x=xr[oka][order], y=am[oka][order], mode="lines", name="airmass",
            line=dict(color="rgba(20,184,166,0.85)", width=2), hovertemplate="airmass %{y:.3f}<extra></extra>",
        ), row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="airmass", secondary_y=True, row=1, col=1, showgrid=False)
    else:
        fig.update_yaxes(visible=False, secondary_y=True, row=1, col=1)
    if has_chk:
        kx = pd.to_numeric(chk["bjd"], errors="coerce").to_numpy(float)
        ky = pd.to_numeric(chk["kmag"], errors="coerce").to_numpy(float)
        kok = np.isfinite(kx) & np.isfinite(ky)
        kr = ky[kok] - float(np.nanmedian(ky[kok])) if kok.any() else np.array([])
        fig.add_trace(go.Scatter(
            x=kx[kok] - x0, y=kr, mode="markers", name="check star", marker=dict(color="#9ca3af", size=5),
            hovertemplate="check - median %{y:+.4f}<extra></extra>",
        ), row=2, col=1)
        lim = max(0.03, float(np.nanpercentile(np.abs(kr), 98)) * 1.3) if kr.size else 0.03
        fig.update_yaxes(title_text="check - med", range=[lim, -lim], row=2, col=1)
        if kr.size:
            fig.add_annotation(text=f"check star scatter {np.nanstd(kr):.3f} mag", row=2, col=1,
                               xref="x domain", yref="y domain", x=0.01, y=0.95, showarrow=False,
                               font=dict(size=11, color="#6b7280"), xanchor="left")
    fig.update_yaxes(title_text=ylab, autorange="reversed", row=1, col=1, secondary_y=False)
    fig.update_xaxes(title_text=(f"BJD_TDB - {x0:.0f}" if xcol == "bjd" else f"JD - {x0:.0f}"),
                     row=2 if has_chk else 1, col=1)
    if "jd" in df.columns:
        tv, tt = _ut_ticks(xr, pd.to_numeric(df["jd"], errors="coerce").to_numpy(float))
        if tv:
            # UT (HH:MM) on the top axis of the light-curve panel, same scale as the BJD axis
            fig.update_xaxes(side="top", tickvals=tv, ticktext=tt, title_text="UT (HH:MM)", showticklabels=True,
                             row=1, col=1)
    fig.update_layout(height=580, margin=dict(l=60, r=60, t=60, b=40), template="plotly_white",
                      legend=dict(orientation="h", y=-0.12, x=0), hovermode="closest")
    return fig


def _all_setups_figure(setups: dict[str, dict], cid: str) -> Any | None:
    """Light curves of one target from every observation setup on one time axis."""
    import plotly.graph_objects as go  # noqa: PLC0415

    from gaia_bands import lc_uncalibrated  # noqa: PLC0415
    from photometry_lightcurve import published_mag_column  # noqa: PLC0415

    traces, xs = [], []
    for name, p in setups.items():
        lc_dir = Path(p["photometry_dir"]) / "lightcurves"
        f = lc_dir / f"lightcurve_{cid}.csv"
        if not f.is_file():
            continue
        df = pd.read_csv(f, low_memory=False)
        ycol = ("delta_mag" if lc_uncalibrated(lc_dir, cid)
                else (published_mag_column(df) or "mag_calib"))
        x = pd.to_numeric(df.get("bjd"), errors="coerce").to_numpy(float)
        traces.append((name, x, pd.to_numeric(df[ycol], errors="coerce").to_numpy(float), ycol))
        xs.append(x)
    if not traces:
        return None
    x0 = float(np.floor(np.nanmin(np.concatenate(xs))))
    fig = go.Figure()
    for name, x, y, ycol in traces:
        fig.add_trace(go.Scatter(x=x - x0, y=y, mode="markers", name=f"{setup_label(name)} ({ycol})",
                                 marker=dict(size=6), hovertemplate="%{y:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="mag")
    fig.update_xaxes(title_text=f"BJD_TDB - {x0:.0f}")
    fig.update_layout(height=520, template="plotly_white", legend=dict(orientation="h", y=-0.15))
    return fig


def _facts_html(pg: dict[str, Any]) -> str:
    rows = []
    for a, b in pg["facts"]:
        colour = "#f0883e" if a in pg["warn"] else "inherit"
        rows.append(f"<tr><td style='opacity:0.65;padding:2px 12px 2px 0;vertical-align:top;white-space:nowrap'>"
                    f"{html.escape(a)}</td><td style='padding:2px 0;color:{colour}'>{html.escape(b)}</td></tr>")
    return "<table style='font-size:0.9rem;border-collapse:collapse'>" + "".join(rows) + "</table>"


def _gaia_link(cid: str) -> str:
    url = f"https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=I/355/gaiadr3&Source={quote_plus(cid)}"
    return f"<a href='{html.escape(url)}' target='_blank' rel='noopener noreferrer'>{html.escape(cid)}</a>"


def _tier_text(raw: str) -> str:
    """Colour class of a comparison star as stored by Phase 1 ('1.0', 'TIER2_...') -> '1', '2'."""
    s = str(raw or "").strip()
    if not s:
        return "-"
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else s


def _comp_table_html(pg: dict[str, Any]) -> str:
    """The PDF comparison table plus a tier column; the Gaia id links to Gaia DR3 in VizieR."""
    import star_page_m3 as m3  # noqa: PLC0415

    hdr = [h.replace("{band}", pg["band"] or "g") for h in m3.COMP_HEADERS]
    hdr = hdr[:-1] + ["tier", hdr[-1]]
    left = {0, 1, 9, 10}
    th = "".join(f"<th style='padding:6px 8px;text-align:{'left' if i in left else 'right'}'>{html.escape(h)}</th>"
                 for i, h in enumerate(hdr))
    body = []
    for r in pg["comp_rows"]:
        tier = _tier_text(pg["comp_tier"].get(r[1], ""))
        status = r[9]
        st_col = "#e3b341" if status.startswith("suspect") else "#3fb950"
        cells = ([html.escape(r[0]), _gaia_link(r[1])] + [html.escape(v) for v in r[2:9]]
                 + [html.escape(tier), f"<span style='color:{st_col}'>{html.escape(status)}</span>"])
        body.append("<tr style='border-top:1px solid rgba(128,128,128,0.25)'>" + "".join(
            f"<td style='padding:5px 8px;text-align:{'left' if i in left else 'right'}'>{c}</td>"
            for i, c in enumerate(cells)) + "</tr>")
    return ("<div style='overflow-x:auto'><table style='width:100%;border-collapse:collapse;font-size:0.9rem'>"
            f"<thead><tr style='background:rgba(128,128,128,0.15)'>{th}</tr></thead><tbody>{''.join(body)}</tbody>"
            "</table></div>")


def _cutout_png(pages: "TargetPages", pg: dict[str, Any]) -> Path | None:
    """Aperture / sky-annulus cutout (same file the PDF report caches)."""
    import star_page_m3 as m3  # noqa: PLC0415

    out = pages.photometry_dir / "_report_cache" / f"m3_cut_{pg['cid']}.png"
    # OUTPUT-REVIEW-2026-10 step 7: reuse only when newer than what it draws (the light curve and the
    # aperture choice of the last Phase 2A); otherwise draw it again
    _src = [pages.lc_dir / f"lightcurve_{pg['cid']}.csv", pages.photometry_dir / "aperture_per_target.json"]
    _src_mt = max((p.stat().st_mtime for p in _src if p.is_file()), default=0.0)
    if out.is_file() and out.stat().st_mtime >= _src_mt:
        return out
    ms = pages.platesolve_dir / "MASTERSTAR.fits"
    if not (ms.is_file() and math.isfinite(pg["x"]) and math.isfinite(pg["r_ap"]) and math.isfinite(pg["fwhm_night"])):
        return None
    return m3.render_cutout(ms, pg["x"], pg["y"], pg["r_ap"], pg["r_in"], pg["r_out"], out,
                            neighbours=pages.field_neighbours(pg["x"], pg["y"], pg["r_out"], exclude_id=pg["cid"]))


def _target_label(pages: "TargetPages", cid: str) -> str:
    import star_page_m3 as m3  # noqa: PLC0415

    row = pages.summary_row(cid)
    vt = str(row.get("vsx_type", "") or "").strip()
    name = m3.target_display_name(pages.at_row(cid, row), cid)
    return name + (f" \u2014 {vt}" if vt and vt.lower() != "nan" else "")


def _step_target(key: str, n: int, delta: int) -> None:
    st.session_state[key] = (int(st.session_state.get(key, 0)) + delta) % n


def _target_header(pg: dict[str, Any], setup: str) -> None:
    sub = [s for s in pg["subline"] if s != setup and not re.fullmatch(r"\d{4}-\d\d-\d\d", s)]
    if pg["name"].startswith("Gaia DR3"):
        vsx = ("https://www.aavso.org/vsx/index.php?view=results.get&coords="
               + quote_plus(f"{pg['ra_deg']:.5f} {pg['dec_deg']:+.5f}") + "&format=d&size=0.01")
    else:
        vsx = "https://www.aavso.org/vsx/index.php?view=results.get&ident=" + quote_plus(pg["name"])
    viz = f"https://vizier.cds.unistra.fr/viz-bin/VizieR?-c={pg['ra_deg']:.6f}{pg['dec_deg']:+.6f}&-c.rs=5"
    sub_txt = html.escape(f" {_DOT} ".join(sub))
    st.markdown(
        "<div style='display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 16px'>"
        f"<span style='font-size:1.9rem;font-weight:700'>{html.escape(pg['name'])}</span>"
        f"<span style='opacity:0.8'>{sub_txt}</span>"
        f"<span><a href='{html.escape(vsx)}' target='_blank'>VSX</a> {_DOT} "
        f"<a href='{html.escape(viz)}' target='_blank'>VizieR</a></span></div>",
        unsafe_allow_html=True,
    )
    if pg["trust"]:
        col = _TRUST_COLOUR.get(pg["trust"], "#9ca3af")
        st.markdown(f"<div style='color:{col}'><b>TRUST: {html.escape(pg['trust'])}</b> &nbsp; "
                    f"{html.escape(pg['trust_reason'])}</div>", unsafe_allow_html=True)


def render_aperture_photometry(
    cfg: "AppConfig",
    draft_id: int | None,
    *,
    draft_dir_override: Path | None = None,
) -> None:
    """Aperture Photometry tab: one target page at a time + Field HRD."""
    import star_page_m3 as m3  # noqa: PLC0415
    from target_page import star_fields  # noqa: PLC0415

    st.header("Aperture Photometry")
    if draft_id is None and draft_dir_override is None:
        st.info("No active draft. Load a draft above or run VAR-STREM.")
        return
    draft_dir = resolve_draft_dir_path(draft_dir_override, draft_id, cfg.archive_root)
    setups = find_photometry_setups(cfg, draft_id, draft_dir_override=draft_dir) if draft_dir else {}
    setups = {k: v for k, v in setups.items() if (Path(v["photometry_dir"]) / "photometry_summary.csv").is_file()}
    if not setups:
        st.warning("Photometry has not been run for this draft (runs are started by VAR-STREM / the night runner).")
        return

    c_info, c_setup, c_pdf = st.columns([3, 1.4, 0.8], vertical_alignment="bottom")
    with c_setup:
        setup = st.selectbox("Observation setup", list(setups), format_func=setup_label, key="phase2a_setup_select")
    ps_dir = Path(setups[setup]["obs_group_dir"])
    summary_csv = Path(setups[setup]["photometry_dir"]) / "photometry_summary.csv"
    pages = _group_pages(str(ps_dir), summary_csv.stat().st_mtime)
    cids = _measured_cids(pages)
    with c_info:
        done = datetime.datetime.fromtimestamp(summary_csv.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
        st.caption(f"{Path(draft_dir).name} {_DOT} photometry completed {done} {_DOT} {len(cids)} light curves")
    with c_pdf:
        pdf = _latest_report_pdf(ps_dir)
        if pdf is not None:
            st.download_button("PDF report", data=pdf.read_bytes(), file_name=pdf.name, mime="application/pdf",
                               key=f"vyvar_ap_pdf_{setup}")

    # A view switch, not st.tabs: Streamlit runs every tab body on each rerun, and the HRD
    # panel (~5 s) would slow down every target switch.
    view = st.segmented_control("View", ["Light curves", "Field HRD", "Measurement quality"], default="Light curves",
                                key="vyvar_ap_view", label_visibility="collapsed")
    if view == "Measurement quality":
        from ui_photometry_quality import render_measurement_quality  # noqa: PLC0415

        render_measurement_quality(ps_dir)
        return
    if view == "Field HRD":
        from ui_hrd import render_hrd_tab  # noqa: PLC0415

        render_hrd_tab(Path(setups[setup]["photometry_dir"]), cfg)
        return
    if view in (None, "Light curves"):
        if not cids:
            st.info("No measured light curves in this setup.")
            return
        idx_key = f"vyvar_ap_target_idx_{setup}"
        idx = int(st.session_state.get(idx_key, 0)) % len(cids)
        c_sel, c_prev, c_next, c_pos = st.columns([6, 0.5, 0.5, 1], vertical_alignment="bottom")
        with c_sel:
            pick = st.selectbox("Target", range(len(cids)), index=idx, format_func=lambda i: _target_label(pages, cids[i]),
                                key=f"vyvar_ap_target_sel_{setup}_{idx}")
        if pick != idx:
            st.session_state[idx_key] = idx = int(pick)
        with c_prev:
            st.button("\u2039", key="vyvar_ap_prev", help="Previous target", on_click=_step_target,
                      args=(idx_key, len(cids), -1))
        with c_next:
            st.button("\u203a", key="vyvar_ap_next", help="Next target", on_click=_step_target,
                      args=(idx_key, len(cids), 1))
        with c_pos:
            st.caption(f"{idx + 1} / {len(cids)}")

        cid = cids[idx]
        sd = star_fields(pages.summary_row(cid), pages.lc_stats_by_cid.get(cid, {}))
        pg = pages.page(sd, obs_group=setup)
        _target_header(pg, setup)

        left, right = st.columns([1.75, 1])
        with left:
            t1, t2, t3 = st.columns(3)
            show_am = t1.toggle("Show airmass", value=True, key="toggle_show_airmass")
            show_fl = t2.toggle("Show flagged points", value=True, key="phase2a_show_outliers")
            all_setups = t3.toggle("All setups in one chart", value=False, key="phase2a_show_all_filters")
            fig = _all_setups_figure(setups, cid) if all_setups else None
            if fig is None:
                fig = _lc_figure(pages, pg, show_airmass=show_am, show_flagged=show_fl)
            st.plotly_chart(fig, width="stretch", key=f"vyvar_ap_lc_{setup}_{cid}")
        with right:
            st.markdown(_facts_html(pg), unsafe_allow_html=True)
            ci, cm = st.columns([1, 1.6])
            cut = _cutout_png(pages, pg)
            if cut is not None:
                ci.image(str(cut), caption="aperture / sky annulus", width="stretch")
            from field_map import field_map_path  # noqa: PLC0415

            fm = field_map_path(pages.lc_dir, cid)
            if fm is not None:
                cm.image(str(fm), caption="field: target red, comparison stars green", width="stretch")

        st.markdown("#### Comparison stars")
        if pg["comp_rows"]:
            st.markdown(_comp_table_html(pg), unsafe_allow_html=True)
            st.caption((m3.COMP_FOOT + "  tier = colour class of the comparison star (Phase 1).").replace("*", "\\*"))
        else:
            st.info("Comparison star table not available.")
        if pg["note_text"]:
            st.warning(pg["note_text"])
