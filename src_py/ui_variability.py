"""Variability tab (VAR-2026-10, rebuilt VAR-TAB-2026-10 - Milan 2026-10-18).

Interactive view of :func:`variability_field.run_field_variability` (aperture photometry):

* hockey-stick of every star (hover = VSX name / type / period, click opens the star);
* one table of variable stars: new candidates first, then the known VSX variables the
  search recovered;
* the star: night light curve (with FWHM and airmass), TESS on demand into
  ``photometry/TESS/<gaia id>/`` and a Peranso-like period / phase view.

Everything is plotly; nothing is a PNG.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import AppConfig
from platesolve_ui_paths import setup_label
from tess_runner import tess_dir_for
from utils import resolve_draft_dir_path
from variability_plots import noise_curves
from variability_store import (
    epoch_of_minimum,
    load_adopted_periods,
    load_report_include,
    pdm_theta,
    peranso_text,
    phase_of,
    phased_text,
    save_adopted_period,
    save_report_include,
    tess_cell,
    tess_summary,
)

C_CAND = "#f5a524"
C_VSX = "#4c8dff"
C_KNOWN = "#6aa8ff"
C_SYS = "#d9b43a"
C_FIELD = "#7c8494"
C_REJ = "rgba(150,156,168,0.35)"
C_SEL = "#ff4b4b"
C_TESS = "rgba(91,141,239,0.55)"

LAYERS = (
    ("cand", "New candidates"),
    ("kdet", "VSX variables found"),
    ("known", "Other known / targets"),
    ("sys", "Seeing / airmass signal"),
    ("field", "Field stars"),
    ("rej", "Rejected by gates"),
)
PERIOD_STEPS = (1e-3, 1e-4, 1e-5, 1e-6)
_PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}


def _detect_obs_groups(draft_dir: Path) -> list[str]:
    root = Path(draft_dir) / "platesolve"
    if not root.is_dir():
        return []
    return [d.name for d in sorted(root.iterdir()) if d.is_dir() and (d / "photometry").is_dir()]


def _read_json(p: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _fmt(v: Any, nd: int) -> str:
    x = _num(v)
    return f"{x:.{nd}f}" if math.isfinite(x) else "-"


def gaia_url(cid: str) -> str:
    return f"https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=I/355/gaiadr3&Source={cid}"


def _stamp(*paths: Path) -> tuple:
    return tuple((str(p), p.stat().st_mtime_ns if p.is_file() else 0) for p in paths)


def star_class(row: pd.Series) -> str:
    """Hockey-stick layer of one star (same precedence as the PDF figure)."""
    if bool(row["candidate"]):
        return "cand"
    if bool(row["known_detected"]):
        return "kdet"
    if bool(row["known_variable"]) or bool(row["target"]):
        return "known"
    if bool(row["gates_ok"]) and bool(row["systematic"]):
        return "sys"
    return "field" if bool(row["gates_ok"]) else "rej"


@st.cache_resource(show_spinner=False, max_entries=4)
def _group_data(phot_s: str, _stamp_key: tuple) -> dict[str, Any]:
    """Everything the tab shows for one setup, loaded once (cache follows file mtimes)."""
    phot = Path(phot_s)
    ps = phot.parent
    meta = _read_json(phot / "variability_meta.json")
    field = pd.read_csv(phot / "variability_field.csv", dtype={"catalog_id": str}, low_memory=False)
    for c in ("mag", "s", "e", "bp_rp", "z_s", "z_eta", "s_exp", "amplitude_mag", "ra_deg", "dec_deg"):
        if c in field:
            field[c] = pd.to_numeric(field[c], errors="coerce")
    vt = ps / "variable_targets.csv"
    vsx = pd.DataFrame(columns=["catalog_id", "vsx_name", "vsx_type", "vsx_period"])
    if vt.is_file():
        v = pd.read_csv(vt, dtype={"catalog_id": str}, low_memory=False)
        vsx = v[[c for c in vsx.columns if c in v.columns]].drop_duplicates("catalog_id")
    field = field.merge(vsx, on="catalog_id", how="left")
    field["cls"] = field.apply(star_class, axis=1)
    from variability_field import load_variability_candidates  # noqa: PLC0415

    cand = load_variability_candidates(phot)
    labels = dict(zip(cand["catalog_id"].astype(str), cand["label"].astype(str)))
    catalogs = dict(zip(cand["catalog_id"].astype(str), cand.get("catalogs", pd.Series([""] * len(cand))).astype(str)))
    field["label"] = field["catalog_id"].map(labels)
    rows = []
    for _, r in field[field["cls"] == "cand"].sort_values("z_s", ascending=False).iterrows():
        rows.append(_table_row(r, "NEW", labels.get(r["catalog_id"], ""), catalogs.get(r["catalog_id"], "not checked")))
    for _, r in field[field["cls"] == "kdet"].sort_values("z_s", ascending=False).iterrows():
        rows.append(_table_row(r, "VSX", "", ""))
    curves = noise_curves(field, meta)
    return {"meta": meta, "field": field, "rows": pd.DataFrame(rows), "curves": curves}


def _table_row(r: pd.Series, kind: str, label: str, catalogs: str) -> dict[str, Any]:
    name = f"Candidate {label}" if kind == "NEW" else (str(r["vsx_name"]) if pd.notna(r.get("vsx_name")) else f"Gaia DR3 {r['catalog_id']}")
    vtype = "" if kind == "NEW" or pd.isna(r.get("vsx_type")) else str(r["vsx_type"])
    vper = float("nan") if kind == "NEW" else _num(r.get("vsx_period"))
    return {
        "kind": kind, "label": label, "name": name, "vsx_type": vtype, "vsx_period": vper,
        "catalog_id": str(r["catalog_id"]), "G": _num(r["mag"]), "bp_rp": _num(r.get("bp_rp")),
        "s": _num(r["s"]), "z_s": _num(r["z_s"]), "z_eta": _num(r["z_eta"]), "amplitude": _num(r.get("amplitude_mag")),
        "index": str(r.get("index") or ""), "catalogs": catalogs,
    }


def _hover_name(r: pd.Series) -> str:
    if pd.notna(r.get("label")):
        return f"<b>{r['label']}</b> new candidate"
    if pd.notna(r.get("vsx_name")):
        bits = [f"<b>{r['vsx_name']}</b>"]
        if pd.notna(r.get("vsx_type")):
            bits.append(str(r["vsx_type"]))
        if math.isfinite(_num(r.get("vsx_period"))):
            bits.append(f"P {_num(r['vsx_period']):.6g} d")
        return " \u00b7 ".join(bits)
    return f"<b>Gaia DR3 {r['catalog_id']}</b>"


def hockey_figure(data: dict[str, Any], layers: set[str], sel: str | None) -> go.Figure:
    f = data["field"]
    fig = go.Figure()
    styles = {
        "rej": dict(size=3, color=C_REJ),
        "field": dict(size=4, color=C_FIELD),
        "sys": dict(size=9, symbol="triangle-up-open", color=C_SYS, line=dict(width=1.5)),
        "known": dict(size=6, symbol="circle-open", color=C_KNOWN, line=dict(width=1), opacity=0.55),
        "kdet": dict(size=11, color=C_VSX, line=dict(color="white", width=1)),
        "cand": dict(size=14, color=C_CAND, line=dict(color="black", width=1)),
    }
    detail = ("%{customdata[1]}<br>Gaia DR3 %{customdata[0]}<br>G %{x:.2f} \u00b7 BP-RP %{customdata[2]}<br>"
              "scatter %{y:.4f} (expected %{customdata[3]})<br>z scatter %{customdata[4]} \u00b7 z 1/\u03b7 %{customdata[5]}"
              "<extra>%{customdata[6]}</extra>")
    names = dict(LAYERS)
    for key in ("rej", "field", "sys", "known", "kdet", "cand"):
        sub = f[(f["cls"] == key) & np.isfinite(f["mag"]) & (f["s"] > 0)]
        if sub.empty or key not in layers:
            continue
        if key in ("rej", "field"):
            fig.add_trace(go.Scattergl(x=sub["mag"], y=sub["s"], mode="markers", marker=styles[key], name=names[key],
                                       hovertemplate=f"G %{{x:.2f}} \u00b7 scatter %{{y:.4f}}<extra>{names[key]}</extra>"))
            continue
        cd = np.column_stack([
            sub["catalog_id"], [_hover_name(r) for _, r in sub.iterrows()],
            [_fmt(v, 2) for v in sub["bp_rp"]], [_fmt(v, 4) for v in sub["s_exp"]],
            [_fmt(v, 1) for v in sub["z_s"]], [_fmt(v, 1) for v in sub["z_eta"]], [names[key]] * len(sub),
        ])
        fig.add_trace(go.Scatter(
            x=sub["mag"], y=sub["s"], mode="markers+text" if key == "cand" else "markers", marker=styles[key],
            text=sub["label"] if key == "cand" else None, textposition="top right", textfont=dict(color=C_CAND, size=13),
            customdata=cd, hovertemplate=detail, name=names[key]))
    cur = data["curves"]
    if cur is not None:
        mids, sexp, slim = cur
        fig.add_trace(go.Scatter(x=mids, y=sexp, mode="lines", line=dict(color="#9aa0ab", width=2), name="noise model",
                                 hovertemplate="expected %{y:.4f}<extra>noise model</extra>"))
        if slim is not None:
            fig.add_trace(go.Scatter(x=mids, y=slim, mode="lines", line=dict(color=C_CAND, width=1.6, dash="dash"),
                                     name="candidate limit", hovertemplate="limit %{y:.4f}<extra>candidate limit</extra>"))
    if sel:
        hit = f[f["catalog_id"] == sel]
        if not hit.empty:
            fig.add_trace(go.Scatter(x=hit["mag"], y=hit["s"], mode="markers", hoverinfo="skip", name="selected",
                                     marker=dict(size=26, symbol="circle-open", color=C_SEL, line=dict(width=2.5))))
    fig.update_layout(height=500, margin=dict(l=60, r=16, t=10, b=46), showlegend=False, hovermode="closest",
                      xaxis_title="Gaia G [mag]", yaxis_title="robust scatter 1.4826 MAD [mag]", clickmode="event+select")
    ticks = [k * 10.0 ** e for e in range(-4, 1) for k in (1, 2, 5)]
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"{v:g}" for v in ticks])
    return fig


def night_figure(lc: pd.DataFrame, time_label: str) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.64, 0.18, 0.18], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=lc["time"], y=lc["dmag"], mode="markers", marker=dict(size=6, color=C_VSX),
                             error_y=dict(type="data", array=lc["err"], color="rgba(76,141,255,0.45)", thickness=1, width=0),
                             hovertemplate="dmag %{y:.4f} \u00b1 %{error_y.array:.4f}<extra></extra>"), row=1, col=1)
    if "fwhm" in lc:
        fig.add_trace(go.Scatter(x=lc["time"], y=lc["fwhm"], mode="lines", line=dict(color=C_FIELD, width=1.4),
                                 hovertemplate="FWHM %{y:.2f} px<extra></extra>"), row=2, col=1)
    if "airmass" in lc:
        fig.add_trace(go.Scatter(x=lc["time"], y=lc["airmass"], mode="lines", line=dict(color="#d4a72c", width=1.6),
                                 hovertemplate="airmass %{y:.3f}<extra></extra>"), row=3, col=1)
    fig.update_yaxes(autorange="reversed", title_text="dmag", row=1, col=1)
    fig.update_yaxes(title_text="FWHM px", row=2, col=1)
    fig.update_yaxes(title_text="airmass", row=3, col=1)
    fig.update_xaxes(title_text=time_label, tickformat=".3f", row=3, col=1)
    fig.update_layout(height=440, margin=dict(l=60, r=16, t=10, b=46), showlegend=False, hovermode="x unified")
    return fig


@st.cache_data(show_spinner=False, max_entries=16)
def _tess_files(tdir_s: str, _stamp_key: tuple) -> dict[str, Any]:
    tdir = Path(tdir_s)
    out: dict[str, Any] = {"lc": None, "pgram": None, "pix": None}
    if (tdir / "tess_lc.csv").is_file():
        d = pd.read_csv(tdir / "tess_lc.csv")
        out["lc"] = d[np.isfinite(d["dmag"])].reset_index(drop=True)
    if (tdir / "periodogram.csv").is_file():
        p = pd.read_csv(tdir / "periodogram.csv")
        p = p[p["frequency_per_d"] > 0]
        out["pgram"] = p.assign(period_d=1.0 / p["frequency_per_d"]).sort_values("period_d")
    if (tdir / "pixel_map.json").is_file():
        out["pix"] = _read_json(tdir / "pixel_map.json")
    return out


def _period_state(cid: str) -> dict[str, Any]:
    return st.session_state.setdefault(f"var_pstate_{cid}", {})


def _set_period(cid: str, p: float, *, rebase: bool, t_lc: np.ndarray | None = None, m_lc: np.ndarray | None = None) -> None:
    if not (math.isfinite(p) and p > 0):
        return
    stt = _period_state(cid)
    stt["P"] = float(p)
    if rebase or not stt.get("base"):
        stt["base"] = float(p)
    if t_lc is not None and m_lc is not None:
        stt["t0"] = epoch_of_minimum(t_lc, m_lc, float(p))
        st.session_state[f"var_t0_{cid}"] = float(stt["t0"])
    st.session_state[f"var_pin_{cid}"] = float(p)
    b = stt["base"]
    st.session_state[f"var_scan_{cid}"] = float(min(max(p, b * 0.995), b * 1.005))


def periodogram_figure(pg: pd.DataFrame, period: float, vsx_period: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=pg["period_d"], y=pg["ls_power"], mode="lines", line=dict(color="#9aa0ab", width=1.2),
                               hovertemplate="P %{x:.6f} d<br>power %{y:.3f}<extra>click to use</extra>"))
    fig.add_vline(x=period, line=dict(color=C_SEL, width=2))
    if math.isfinite(vsx_period):
        fig.add_vline(x=vsx_period, line=dict(color=C_VSX, width=1.5, dash="dot"))
    fig.update_layout(height=360, margin=dict(l=60, r=16, t=10, b=46), showlegend=False, hovermode="x",
                      xaxis_title="period [d]", yaxis_title="LS power", clickmode="event+select")
    fig.update_xaxes(type="log")
    fig.update_yaxes(rangemode="tozero")
    return fig


def phase_figure(t: np.ndarray, m: np.ndarray, period: float, t0: float, *, bins: bool, night: pd.DataFrame | None) -> go.Figure:
    ph = phase_of(t, period, t0)
    fig = go.Figure(go.Scattergl(x=np.r_[ph, ph + 1], y=np.r_[m, m], mode="markers", marker=dict(size=3, color=C_TESS),
                                 hovertemplate="phase %{x:.3f}<br>dmag %{y:.4f}<extra>TESS</extra>"))
    if bins:
        nb = 50
        k = np.minimum((ph * nb).astype(int), nb - 1)
        cx = [(i + 0.5) / nb for i in range(nb) if np.any(k == i)]
        cy = [float(np.mean(m[k == i])) for i in range(nb) if np.any(k == i)]
        # WebGL traces draw above SVG ones: the overlays are WebGL too so they stay on top of the TESS points
        fig.add_trace(go.Scattergl(x=np.r_[cx, np.add(cx, 1)], y=np.r_[cy, cy], mode="lines", line=dict(color=C_CAND, width=2.5),
                                 hovertemplate="mean %{y:.4f}<extra>binned</extra>"))
    if night is not None and not night.empty:
        pn = phase_of(night["time"].to_numpy(float), period, t0)
        mn = night["dmag"].to_numpy(float) + (np.nanmedian(m) - np.nanmedian(night["dmag"]))
        fig.add_trace(go.Scattergl(x=np.r_[pn, pn + 1], y=np.r_[mn, mn], mode="markers",
                                   marker=dict(size=6, color=C_SEL, line=dict(color="white", width=0.8)),
                                 hovertemplate="phase %{x:.3f}<br>dmag %{y:.4f}<extra>this night (shifted to TESS median)</extra>"))
    fig.add_vline(x=1.0, line=dict(color="rgba(150,156,168,0.5)", width=1, dash="dot"))
    fig.update_layout(height=360, margin=dict(l=60, r=16, t=10, b=46), showlegend=False, hovermode="closest",
                      xaxis_title=f"phase (P = {period:.7f} d)", yaxis_title="dmag")
    fig.update_xaxes(range=[0, 2], dtick=0.25)
    fig.update_yaxes(autorange="reversed")
    return fig


def pixel_figure(pix: dict[str, Any], cid: str) -> go.Figure:
    amp = np.asarray(pix["amplitude"], float)
    fig = go.Figure(go.Heatmap(z=amp, colorscale="Greys", showscale=False, hovertemplate="x %{x} y %{y}<br>amplitude %{z:.4f}<extra></extra>"))
    my, mx = np.where(np.asarray(pix["mask"], int) > 0)
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(symbol="square-open", size=18, color=C_VSX, line=dict(width=2)),
                             hovertemplate="aperture pixel<extra></extra>"))
    st_ids = list(pix.get("stars", {}))
    xy = np.array([pix["stars"][k] for k in st_ids]) if st_ids else np.zeros((0, 2))
    if len(xy):
        is_t = np.array([k == cid for k in st_ids])
        fig.add_trace(go.Scatter(x=xy[~is_t, 0], y=xy[~is_t, 1], mode="markers", marker=dict(size=6, color=C_FIELD),
                                 text=[k for k, t in zip(st_ids, is_t) if not t], hovertemplate="Gaia DR3 %{text}<extra>neighbour</extra>"))
        fig.add_trace(go.Scatter(x=xy[is_t, 0], y=xy[is_t, 1], mode="markers", marker=dict(size=10, color=C_SEL),
                                 hovertemplate="target<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=40, r=10, t=10, b=40), showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _selected_points(event: Any) -> list[dict[str, Any]]:
    try:
        return list(event.selection.points)
    except AttributeError:
        try:
            return list(event["selection"]["points"])
        except (KeyError, TypeError):
            return []


def _apply_once(key: str, signature: Any) -> bool:
    """True when a widget selection is new (selections persist across reruns)."""
    if st.session_state.get(key) == signature:
        return False
    st.session_state[key] = signature
    return True


def render_variability_dashboard(
    cfg: AppConfig,
    *,
    draft_id: int | None = None,
    draft_dir_override: Path | None = None,
) -> None:
    st.header("Variability")
    if draft_id is None and draft_dir_override is None:
        st.info("Load a draft first.")
        return
    draft_dir = resolve_draft_dir_path(draft_dir_override, draft_id, cfg.archive_root)
    if draft_dir is None:
        st.info("Load a draft first.")
        return
    groups = _detect_obs_groups(Path(draft_dir))
    if not groups:
        st.info("No photometry yet - run Aperture Photometry first.")
        return
    og = st.selectbox("Observation setup", groups, format_func=setup_label, key="var_obs_group_v2")
    ps = Path(draft_dir) / "platesolve" / og
    phot = ps / "photometry"
    fcsv = phot / "variability_field.csv"
    if _read_json(phot / "variability_meta.json").get("status") != "ok" or not fcsv.is_file():
        st.warning("The variability search has not run for this setup (it runs inside Aperture Photometry).")
        if st.button("Run the variability search now", key="var_run_now"):
            from variability_field import run_field_variability  # noqa: PLC0415

            with st.spinner("Searching the field..."):
                run_field_variability(Path(draft_dir) / "detrended_aligned" / "lights" / og, phot, cfg, platesolve_dir=ps)
            st.rerun()
        return
    data = _group_data(str(phot), _stamp(fcsv, phot / "variability_candidates.csv", phot / "variability_meta.json", ps / "variable_targets.csv"))
    meta, rows = data["meta"], data["rows"]
    sel_key = f"var_sel_{og}"
    if sel_key not in st.session_state and not rows.empty:
        st.session_state[sel_key] = rows.iloc[0]["catalog_id"]

    nm = meta.get("noise_model") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stars measured", f"{meta.get('n_stars', 0)}", f"{meta.get('n_frames', 0)} frames", delta_color="off")
    c2.metric("Pass quality gates", f"{meta.get('n_gates_ok', 0)}", f"{meta.get('n_rejected_gates', 0)} rejected", delta_color="off")
    c3.metric("Noise model", f"a {_fmt(nm.get('a'), 2)}", f"floor {_fmt(1000 * _num(nm.get('floor_mag')), 1)} mmag", delta_color="off")
    c4.metric("Detection limit", f"z > {_fmt(meta.get('z_threshold'), 2)}", "field false-alarm 1 %", delta_color="off")
    c5.metric("Variable stars found", f"{meta.get('n_candidates', 0)} new \u00b7 {meta.get('n_known_detected', 0)} VSX",
              "new candidates \u00b7 known recovered", delta_color="off")

    _hockey_section(data, og, sel_key)
    _table_section(rows, og, sel_key)
    sel = st.session_state.get(sel_key)
    if rows.empty or sel not in set(rows["catalog_id"]):
        st.info("No variable star to analyse in this setup.")
        return
    _star_section(rows, sel, og, sel_key, phot, ps, meta, cfg)


def _hockey_section(data: dict[str, Any], og: str, sel_key: str) -> None:
    st.subheader("Hockey-stick: scatter of every star vs brightness")
    counts = data["field"]["cls"].value_counts().to_dict()
    opts = [f"{name} ({counts.get(k, 0)})" for k, name in LAYERS]
    picked = st.pills("Show", opts, selection_mode="multi", default=opts, key=f"var_layers_{og}", label_visibility="collapsed")
    layers = {k for (k, _), o in zip(LAYERS, opts) if o in (picked or [])}
    ev = st.plotly_chart(hockey_figure(data, layers, st.session_state.get(sel_key)), width="stretch", config=_PLOT_CFG,
                         on_select="rerun", selection_mode="points", key=f"var_hockey_{og}")
    pts = [p for p in _selected_points(ev) if p.get("customdata")]
    if pts:
        cid = str(pts[0]["customdata"][0])
        if _apply_once(f"var_hockey_pick_{og}", cid) and cid in set(data["rows"]["catalog_id"]) and cid != st.session_state.get(sel_key):
            st.session_state[sel_key] = cid
            st.rerun()
    st.caption("Hover a star for its details, click a variable to open it below, drag to zoom, double-click to reset. "
               "Grey line: scatter expected from each star's CCD error, scaled and floored by the field; dashed orange: "
               "candidate limit of the scatter index (a star can also pass on the 1/\u03b7 smoothness index).")


def _table_section(rows: pd.DataFrame, og: str, sel_key: str) -> None:
    st.subheader("Variable stars in this field")
    if rows.empty:
        st.success("Neither a new candidate nor a known variable stands out in this setup.")
        return
    n_new = int((rows["kind"] == "NEW").sum())
    c1, c2 = st.columns([2, 3])
    flt = c1.segmented_control("Filter", [f"All {len(rows)}", f"New {n_new}", f"VSX {len(rows) - n_new}"],
                               default=f"All {len(rows)}", key=f"var_filter_{og}", label_visibility="collapsed")
    q = c2.text_input("Search", placeholder="Search name or Gaia id", key=f"var_q_{og}", label_visibility="collapsed").strip().lower()
    view = rows
    if flt and flt.startswith("New"):
        view = view[view["kind"] == "NEW"]
    elif flt and flt.startswith("VSX"):
        view = view[view["kind"] == "VSX"]
    if q:
        view = view[view["name"].str.lower().str.contains(q, regex=False) | view["catalog_id"].str.contains(q, regex=False)]
    view = view.reset_index(drop=True)
    st.session_state[f"var_view_{og}"] = list(view["catalog_id"])
    show = pd.DataFrame({
        "Status": np.where(view["kind"] == "NEW", "NEW " + view["label"], "VSX"),
        "Name": view["name"],
        "VSX type": view["vsx_type"],
        "VSX period [d]": view["vsx_period"],
        "Gaia DR3": [gaia_url(c) for c in view["catalog_id"]],
        "G": view["G"], "BP-RP": view["bp_rp"], "scatter [mag]": view["s"],
        "z scatter": view["z_s"], "z 1/\u03b7": view["z_eta"], "amplitude [mag]": view["amplitude"],
    })
    ev = st.dataframe(
        show, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key=f"var_table_{og}",
        column_config={
            "Gaia DR3": st.column_config.LinkColumn("Gaia DR3", display_text=r"Source=(\d+)"),
            "VSX period [d]": st.column_config.NumberColumn(format="%.6f"),
            "G": st.column_config.NumberColumn(format="%.2f"), "BP-RP": st.column_config.NumberColumn(format="%.2f"),
            "scatter [mag]": st.column_config.NumberColumn(format="%.4f"),
            "z scatter": st.column_config.NumberColumn(format="%.1f"), "z 1/\u03b7": st.column_config.NumberColumn(format="%.1f"),
            "amplitude [mag]": st.column_config.NumberColumn(format="%.3f"),
        })
    try:
        picked = list(ev.selection.rows)
    except AttributeError:
        picked = []
    if picked:
        cid = str(view.iloc[picked[0]]["catalog_id"])
        if _apply_once(f"var_table_pick_{og}", cid) and cid != st.session_state.get(sel_key):
            st.session_state[sel_key] = cid
            st.rerun()
    st.caption("New candidates first (highest z first), then the known VSX variables the search recovered. Select a row to open the star.")


def _step_star(og: str, sel_key: str, k: int) -> None:
    order = st.session_state.get(f"var_view_{og}") or []
    if not order:
        return
    cur = st.session_state.get(sel_key)
    i = order.index(cur) if cur in order else -1
    st.session_state[sel_key] = order[(i + k) % len(order)]


def _toggle_include(phot: Path, cid: str) -> None:
    inc = [x for x in load_report_include(phot) if x != cid]
    save_report_include(phot, inc + ([cid] if st.session_state.get(f"var_inc_{cid}") else []))


def _adopt_period(phot: Path, cid: str, period: float, t0: float) -> None:
    save_adopted_period(phot, cid, period, t0)
    # the period is printed in the TESS block of the candidate page: that block must be included
    save_report_include(phot, load_report_include(phot) + [cid])


def _star_section(rows: pd.DataFrame, cid: str, og: str, sel_key: str, phot: Path, ps: Path, meta: dict[str, Any], cfg: AppConfig) -> None:
    r = rows[rows["catalog_id"] == cid].iloc[0]
    st.divider()
    h1, h2 = st.columns([10, 2])
    badge = (f":orange-background[**NEW {r['label']}**]" if r["kind"] == "NEW" else ":blue-background[**VSX**]")
    vsx = "not in VSX" if r["kind"] == "NEW" else " \u00b7 ".join(x for x in (r["vsx_type"], f"P = {r['vsx_period']:.6f} d (VSX)" if math.isfinite(r["vsx_period"]) else "") if x)
    h1.markdown(f"### {badge} {r['name']}  \n{vsx}")
    h1.markdown(f"Gaia DR3 [{cid}]({gaia_url(cid)}) \u00b7 G {_fmt(r['G'], 2)} \u00b7 BP-RP {_fmt(r['bp_rp'], 2)} \u00b7 z scatter {_fmt(r['z_s'], 1)} \u00b7 "
                f"z 1/\u03b7 {_fmt(r['z_eta'], 1)}" + (f" \u00b7 catalogues: {r['catalogs']}" if r["kind"] == "NEW" else f" \u00b7 index: {r['index']}"))
    b1, b2 = h2.columns(2)
    b1.button("\u2039", key=f"var_prev_{og}", help="Previous star", on_click=_step_star, args=(og, sel_key, -1), width="stretch")
    b2.button("\u203a", key=f"var_next_{og}", help="Next star", on_click=_step_star, args=(og, sel_key, 1), width="stretch")
    if r["kind"] == "NEW":
        c1, c2 = st.columns([3, 2])
        if c1.button("Check catalogues again (VSX, SIMBAD, ASAS-SN, ZTF, ATLAS, Gaia vari)", key=f"var_xm_{cid}"):
            from variability_field import _catalog_check  # noqa: PLC0415

            from variability_field import load_variability_candidates, write_candidate_catalogs  # noqa: PLC0415

            cand = load_variability_candidates(phot)
            with st.spinner("Querying the catalogues..."):
                write_candidate_catalogs(phot, {str(x["catalog_id"]): _catalog_check(x, cfg) for _, x in cand.iterrows()})
            st.rerun()
        # the file is the state: the box shows it and a click writes it
        st.session_state[f"var_inc_{cid}"] = cid in load_report_include(phot)
        c2.checkbox("Include the TESS result in the SUMMARY MEASURE REPORT", key=f"var_inc_{cid}",
                    on_change=_toggle_include, args=(phot, cid))

    lc_csv = phot / "lightcurves" / f"varcand_{cid}.csv"
    night = pd.read_csv(lc_csv) if lc_csv.is_file() else None
    t1, t2 = st.columns([4, 1])
    t1.markdown("#### Night light curve (this draft, field zero point)")
    if night is not None and not night.empty:
        t2.download_button("Export for Peranso (.txt)", peranso_text(lc_csv), file_name=f"vyvar_{cid}.txt", key=f"var_per_{cid}", width="stretch")
        st.plotly_chart(night_figure(night, "BJD_TDB" if "bjd" in str(meta.get("time_column", "")).lower() else "JD"), width="stretch", config=_PLOT_CFG, key=f"var_night_{cid}")
    else:
        st.caption("No night light curve for this star.")
    _tess_section(cid, r, phot, og, night, cfg)


@st.fragment
def _tess_section(cid: str, r: pd.Series, phot: Path, og: str, night: pd.DataFrame | None, cfg: AppConfig) -> None:
    """TESS download and the period / phase view (reruns alone while the period changes)."""
    tdir = tess_dir_for(phot, cid)
    h1, h2 = st.columns([3, 2])
    h1.markdown("#### TESS")
    h2.caption(f"`{tdir.relative_to(phot.parent.parent.parent)}`")
    tr = tess_summary(phot, cid)
    files = _tess_files(str(tdir), _stamp(tdir / "tess_lc.csv", tdir / "periodogram.csv", tdir / "pixel_map.json"))
    label = "Download TESS again and analyse" if tr else "Download TESS and analyse"
    if not tr:
        st.info("TESS data are not downloaded for this star yet. The button downloads every TESS sector into the folder "
                "above and runs the period search (works whether tess_enabled is on or off).")
    if st.button(label, key=f"var_tess_{cid}", type="secondary" if tr else "primary"):
        from tess_runner import run_tess_for_star  # noqa: PLC0415

        bar = st.progress(0.0, text="TESS")
        try:
            run_tess_for_star(phot, cid, cfg, progress=lambda m, v: bar.progress(min(max(v, 0.0), 1.0), text=m))
        except Exception as exc:  # noqa: BLE001
            st.error(f"TESS failed: {exc}")
        for k in ("pstate", "pin", "scan", "t0"):
            st.session_state.pop(f"var_{k}_{cid}", None)
        st.rerun()
    if not tr:
        return
    verdict = str(tr.get("verdict"))
    b = tr.get("best") or {}
    msg = f"**{verdict}** \u00b7 {tess_cell(tr)}" + (" \u00b7 " + "; ".join(tr.get("reasons") or []) if tr.get("reasons") else "")
    (st.success if verdict == "RELIABLE" else st.warning)(msg)
    secs = tr.get("sectors") or []
    st.caption(" \u00b7 ".join(f"S{s.get('sector')} ({round(_num(s.get('cadence_s')))} s, {s.get('n_points')} pts)" for s in secs)
               + f" \u00b7 neighbour light {_fmt(100 * _num(tr.get('contamination')), 1)} % \u00b7 amplitude S/N {_fmt(tr.get('snr_amplitude'), 0)}"
               + (f" \u00b7 {b.get('note')}" if b.get("note") else ""))
    lc = files["lc"]
    if lc is None or lc.empty:
        st.caption("No TESS light curve in the folder.")
        return
    t = lc["bjd_tdb"].to_numpy(float)
    m = lc["dmag"].to_numpy(float)
    e = lc["err"].to_numpy(float) if "err" in lc else np.full_like(m, np.nan)
    fig = go.Figure(go.Scattergl(x=t, y=m, mode="markers", marker=dict(size=2.5, color=C_TESS),
                                 hovertemplate="BJD %{x:.4f}<br>dmag %{y:.4f}<extra>TESS</extra>"))
    fig.update_layout(height=230, margin=dict(l=60, r=16, t=6, b=40), showlegend=False,
                      xaxis_title=f"BJD_TDB (all sectors, {len(t)} points)", yaxis_title="dmag")
    fig.update_xaxes(tickformat=".0f")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", config=_PLOT_CFG, key=f"var_tlc_{cid}")

    stt = _period_state(cid)
    tess_p = _num(b.get("period_d"))
    vsx_p = _num(r["vsx_period"])
    if not stt.get("P"):
        p0 = tess_p if math.isfinite(tess_p) else vsx_p
        if not math.isfinite(p0):
            st.caption("No period to fold on yet: pick a peak in the periodogram.")
            p0 = 1.0
        _set_period(cid, p0, rebase=True)
        stt["t0"] = _num(b.get("epoch_min_bjd")) if math.isfinite(_num(b.get("epoch_min_bjd"))) else epoch_of_minimum(t, m, p0)
    period, base, t0 = stt["P"], stt["base"], stt["t0"]
    st.session_state.setdefault(f"var_pin_{cid}", float(period))
    st.session_state.setdefault(f"var_scan_{cid}", float(min(max(period, base * 0.995), base * 1.005)))
    st.session_state.setdefault(f"var_t0_{cid}", float(t0))
    st.session_state.setdefault(f"var_step_{cid}", 1e-4)
    step = float(st.session_state[f"var_step_{cid}"])

    with st.container(border=True):
        c = st.columns([2.2, 2.6, 1.6, 2.0, 2.4, 2.2], vertical_alignment="bottom")
        c[0].number_input("Period [d]", min_value=1e-4, step=step, format="%.7f", key=f"var_pin_{cid}",
                          on_change=lambda: _set_period(cid, float(st.session_state[f"var_pin_{cid}"]), rebase=True))
        n1, n2, n3, n4 = c[1].columns(4)
        n1.button("\u00ab", key=f"var_m10_{cid}", help="Period minus 10 steps", on_click=_set_period, args=(cid, period - 10 * step), kwargs={"rebase": False}, width="stretch")
        n2.button("\u2039", key=f"var_m1_{cid}", help="Period minus one step", on_click=_set_period, args=(cid, period - step), kwargs={"rebase": False}, width="stretch")
        n3.button("\u203a", key=f"var_p1_{cid}", help="Period plus one step", on_click=_set_period, args=(cid, period + step), kwargs={"rebase": False}, width="stretch")
        n4.button("\u00bb", key=f"var_p10_{cid}", help="Period plus 10 steps", on_click=_set_period, args=(cid, period + 10 * step), kwargs={"rebase": False}, width="stretch")
        c[2].selectbox("Step [d]", PERIOD_STEPS, format_func=lambda v: f"{v:g}", key=f"var_step_{cid}")
        h_1, h_2 = c[3].columns(2)
        h_1.button("P \u00f7 2", key=f"var_half_{cid}", on_click=_set_period, args=(cid, period / 2), kwargs={"rebase": True, "t_lc": t, "m_lc": m}, width="stretch")
        h_2.button("P \u00d7 2", key=f"var_dbl_{cid}", on_click=_set_period, args=(cid, period * 2), kwargs={"rebase": True, "t_lc": t, "m_lc": m}, width="stretch")
        r_1, r_2 = c[4].columns(2)
        r_1.button("TESS fit", key=f"var_ptess_{cid}", disabled=not math.isfinite(tess_p), on_click=_set_period,
                   args=(cid, tess_p), kwargs={"rebase": True, "t_lc": t, "m_lc": m}, width="stretch")
        r_2.button("VSX", key=f"var_pvsx_{cid}", disabled=not math.isfinite(vsx_p), on_click=_set_period,
                   args=(cid, vsx_p), kwargs={"rebase": True, "t_lc": t, "m_lc": m}, width="stretch",
                   help=f"VSX period {vsx_p:.6f} d" if math.isfinite(vsx_p) else "no VSX period")
        theta = pdm_theta(phase_of(t, period, t0), m)
        c[5].metric("\u0398 (PDM, lower = better)", _fmt(theta, 3))
        st.slider(f"Fine scan around {base:.6f} d (\u00b10.5 %) - drag and watch the phase curve", min_value=float(base * 0.995),
                  max_value=float(base * 1.005), step=float(base * 2e-6),
                  format="%.6f", key=f"var_scan_{cid}",
                  on_change=lambda: _set_period(cid, float(st.session_state[f"var_scan_{cid}"]), rebase=False))
        d = st.columns([2.2, 1.6, 1.4, 2.0, 2.2, 2.6], vertical_alignment="bottom")
        new_t0 = d[0].number_input("Epoch T0 (BJD_TDB)", format="%.5f", step=0.001, key=f"var_t0_{cid}")
        if new_t0 != t0:
            stt["t0"] = t0 = float(new_t0)
        d[1].button("Minimum \u2192 phase 0", key=f"var_min0_{cid}", width="stretch", on_click=_set_period, args=(cid, period),
                    kwargs={"rebase": False, "t_lc": t, "m_lc": m})
        show_bins = d[2].checkbox("Binned mean", value=True, key=f"var_bins_{cid}")
        show_night = d[3].checkbox("Overlay this night", value=night is not None, disabled=night is None, key=f"var_ovl_{cid}")
        d[4].download_button("Export phased (Peranso .txt)", phased_text(t, m, e, period, t0, f"TESS {r['name']}"),
                             file_name=f"vyvar_{cid}_phased.txt", key=f"var_phx_{cid}", width="stretch")
        if r["kind"] == "NEW":
            d[5].button("Use this period in the report", key=f"var_adopt_{cid}", type="primary", width="stretch",
                        on_click=_adopt_period, args=(phot, cid, period, t0))
            ad = load_adopted_periods(phot).get(cid)
            if ad:
                d[5].caption(f"in the report: P = {ad['period_d']:.7f} d")

    g1, g2 = st.columns([2, 3])
    pg = files["pgram"]
    with g1:
        st.caption("Periodogram (Lomb-Scargle) - click a peak to take its period")
        if pg is not None and not pg.empty:
            ev = st.plotly_chart(periodogram_figure(pg, period, vsx_p), width="stretch", config=_PLOT_CFG,
                                 on_select="rerun", selection_mode="points", key=f"var_pg_{cid}")
            pts = _selected_points(ev)
            if pts and _apply_once(f"var_pg_pick_{cid}", round(float(pts[0]["x"]), 9)):
                _set_period(cid, _refine_peak(pg, float(pts[0]["x"])), rebase=True, t_lc=t, m_lc=m)
                st.rerun()
        else:
            st.caption("No periodogram data in the folder - download TESS again to get it.")
    with g2:
        st.caption("Phase curve, two cycles - follows the period live")
        st.plotly_chart(phase_figure(t, m, period, t0, bins=show_bins, night=night if show_night else None),
                        width="stretch", config=_PLOT_CFG, key=f"var_phase_{cid}")
    pix = files["pix"]
    if pix:
        with st.expander("TESS pixel map (aperture pixels, target and neighbours)"):
            st.plotly_chart(pixel_figure(pix, cid), width="stretch", config=_PLOT_CFG, key=f"var_pix_{cid}")
            if pix.get("target_likelihood") is not None:
                st.caption(f"Target likelihood {float(pix['target_likelihood']):.2f}")


def _refine_peak(pg: pd.DataFrame, period: float) -> float:
    """Highest periodogram point near the clicked period."""
    p = pg["period_d"].to_numpy(float)
    w = pg["ls_power"].to_numpy(float)
    i = int(np.argmin(np.abs(np.log(p / period))))
    lo, hi = max(0, i - 15), min(len(p), i + 16)
    return float(p[lo + int(np.argmax(w[lo:hi]))])
