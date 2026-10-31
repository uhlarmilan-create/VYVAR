"""Measurement quality view of the Aperture Photometry tab (SETTINGS-2026-10, moved from Settings).

Three questions, one chart each, for the active draft and observation setup:
  1. Which MASTERSTAR stars are measurable?  (field map by zone)
  2. Are bright stars saturated?              (peak histogram with the saturation limit)
  3. Did any frame lose its stars?            (matched stars per frame)
Inputs: ``masterstars_full_match.csv``, ``per_frame_catalog_index.csv`` and ``comparison_stars.csv``
of the setup folder; no per-frame CSV is read.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from platesolve_ui_paths import masterstars_csv_in_dir

#: zone of masterstars_full_match.csv -> (label, colour); order = drawing order
ZONES: dict[str, tuple[str, str]] = {
    "unknown": ("not detected (Gaia only)", "#C9CED6"),
    "noise": ("too faint (noise)", "#8A94A3"),
    "linear": ("measurable (linear)", "#2E8B57"),
    "saturated": ("saturated", "#D1495B"),
}
#: a frame with fewer matched stars than this fraction of the night median is marked
LOW_FRAME_FRACTION = 0.5


@dataclass
class MeasurementQuality:
    stars: pd.DataFrame = field(default_factory=pd.DataFrame)  # x, y, zone, peak, name, mag
    frames: pd.DataFrame = field(default_factory=pd.DataFrame)  # frame, file, n_detected, n_matched, low
    zone_counts: dict[str, int] = field(default_factory=dict)
    n_comp_pool: int = 0
    sat_limit_adu: float | None = None
    median_matched: float | None = None
    problems: list[str] = field(default_factory=list)


def _bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def load_measurement_quality(setup_dir: str | Path) -> MeasurementQuality:
    """Data of the view (pure; tested in ``dev/tests/test_step6_ui_2026_10.py``)."""
    d = Path(setup_dir)
    q = MeasurementQuality()
    ms_csv = masterstars_csv_in_dir(d)
    if ms_csv is None or not ms_csv.is_file():
        q.problems.append("No masterstars_full_match.csv in this setup (MASTERSTAR not built).")
        return q
    ms = pd.read_csv(ms_csv, low_memory=False, dtype={"catalog_id": str, "name": str})
    if "zone" not in ms.columns:
        if "is_saturated" in ms.columns or "is_noisy" in ms.columns:
            sat = _bool(ms.get("is_saturated", pd.Series(False, index=ms.index)))
            noi = _bool(ms.get("is_noisy", pd.Series(False, index=ms.index)))
            use = _bool(ms.get("is_usable", pd.Series(False, index=ms.index)))
            ms["zone"] = np.where(sat, "saturated", np.where(noi, "noise", np.where(use, "linear", "unknown")))
        else:
            ms["zone"] = "unknown"
    ms["zone"] = ms["zone"].fillna("unknown").astype(str).where(ms["zone"].isin(list(ZONES)), "unknown")
    keep = [c for c in ("x", "y", "zone", "peak_max_adu", "name", "mag", "catalog_id") if c in ms.columns]
    st_df = ms[keep].copy()
    for c in ("x", "y", "peak_max_adu", "mag"):
        if c in st_df.columns:
            st_df[c] = pd.to_numeric(st_df[c], errors="coerce")
    q.stars = st_df
    q.zone_counts = {z: int((st_df["zone"] == z).sum()) for z in ZONES}
    for col in ("saturate_limit_adu_85pct", "zone_sat_limit_used", "saturate_limit_adu"):
        if col in ms.columns:
            v = pd.to_numeric(ms[col], errors="coerce").dropna()
            if len(v):
                q.sat_limit_adu = float(v.median())
                break
    cs = d / "comparison_stars.csv"
    if cs.is_file():
        try:
            q.n_comp_pool = int(len(pd.read_csv(cs, usecols=[0])))
        except (OSError, ValueError):
            q.n_comp_pool = 0
    idx = d / "per_frame_catalog_index.csv"
    if idx.is_file():
        fr = pd.read_csv(idx, usecols=lambda c: c in {"file", "status", "n_detected", "n_matched"})
        if "n_matched" in fr.columns and len(fr):
            fr["n_matched"] = pd.to_numeric(fr["n_matched"], errors="coerce")
            fr = fr.sort_values("file").reset_index(drop=True)
            fr.insert(0, "frame", np.arange(1, len(fr) + 1))
            med = float(fr["n_matched"].median())
            q.median_matched = med if math.isfinite(med) else None
            fr["low"] = fr["n_matched"] < LOW_FRAME_FRACTION * med if q.median_matched else False
            q.frames = fr
    else:
        q.problems.append("No per_frame_catalog_index.csv (per-frame catalogs not built).")
    return q


# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------
def render_measurement_quality(setup_dir: str | Path) -> None:
    import plotly.graph_objects as go  # noqa: PLC0415
    import streamlit as st  # noqa: PLC0415

    q = load_measurement_quality(setup_dir)
    for p in q.problems:
        st.info(p)
    if q.stars.empty:
        return
    n_all = int(len(q.stars))
    n_lin = q.zone_counts.get("linear", 0)
    n_frames = int(len(q.frames))
    n_low = int(q.frames["low"].sum()) if n_frames else 0
    k = st.columns(5)
    k[0].metric("MASTERSTAR stars", f"{n_all}")
    k[1].metric("Measurable", f"{n_lin} ({100 * n_lin / n_all:.0f} %)" if n_all else "0",
                help="Detected, unsaturated and above the noise (zone linear)")
    k[2].metric("Saturated", f"{q.zone_counts.get('saturated', 0)}")
    k[3].metric("Comparison pool", f"{q.n_comp_pool}")
    k[4].metric("Frames with few stars", f"{n_low} / {n_frames}" if n_frames else "-",
                help=f"Fewer matched stars than {LOW_FRAME_FRACTION:.0%} of the night median")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("**Field: which stars are measurable**")
        fig = go.Figure()
        for z, (label, colour) in ZONES.items():
            sub = q.stars[q.stars["zone"] == z]
            if sub.empty or "x" not in sub:
                continue
            hover = sub.get("name", pd.Series("", index=sub.index)).fillna("").astype(str)
            if "mag" in sub:
                hover = hover + sub["mag"].map(lambda m: f"  G {m:.2f}" if pd.notna(m) else "")
            fig.add_trace(go.Scattergl(
                x=sub["x"], y=sub["y"], mode="markers", name=f"{label} ({len(sub)})",
                marker=dict(size=4 if z in ("unknown", "noise") else 6, color=colour,
                            line=dict(width=0)), text=hover, hovertemplate="%{text}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, title="y [px]", showgrid=False)
        fig.update_xaxes(title="x [px]", showgrid=False)
        fig.update_layout(height=470, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
        st.plotly_chart(fig, width="stretch", key="vyvar_mq_field")
    with right:
        st.markdown("**Bright stars: peak against the saturation limit**")
        pk = q.stars.get("peak_max_adu")
        if pk is None or pk.dropna().empty:
            st.caption("No peak values in the MASTERSTAR catalog.")
        else:
            fig_h = go.Figure()
            bins = np.logspace(np.log10(max(1.0, float(pk.min()))), np.log10(float(pk.max()) * 1.05), 50)
            for z in ("noise", "linear", "saturated"):
                v = pk[(q.stars["zone"] == z) & pk.notna()]
                if len(v):
                    h, e = np.histogram(v, bins=bins)
                    fig_h.add_trace(go.Bar(x=np.sqrt(e[:-1] * e[1:]), y=h, width=np.diff(e), name=ZONES[z][0],
                                           marker_color=ZONES[z][1], opacity=0.85))
            if q.sat_limit_adu:
                fig_h.add_vline(x=q.sat_limit_adu, line_dash="dash", line_color="#D1495B",
                                annotation_text=f"limit {q.sat_limit_adu:,.0f} ADU".replace(",", " "),
                                annotation_position="top right")
            fig_h.update_xaxes(type="log", title="peak [ADU]")
            fig_h.update_yaxes(title="stars")
            fig_h.update_layout(barmode="stack", height=240, margin=dict(l=10, r=10, t=24, b=10), showlegend=False)
            st.plotly_chart(fig_h, width="stretch", key="vyvar_mq_peak")
        st.markdown("**Frames: matched stars per frame**")
        if n_frames:
            fr = q.frames
            fig_f = go.Figure(go.Scatter(x=fr["frame"], y=fr["n_matched"], mode="lines",
                                         line=dict(color="#4C6A92", width=1.5), name="matched",
                                         hovertext=fr["file"], hovertemplate="%{hovertext}: %{y}<extra></extra>"))
            if q.median_matched:
                fig_f.add_hline(y=q.median_matched, line_dash="dot", line_color="#5A6370")
            low = fr[fr["low"]]
            if len(low):
                fig_f.add_trace(go.Scatter(x=low["frame"], y=low["n_matched"], mode="markers",
                                           marker=dict(color="#D1495B", size=8), name="few stars",
                                           hovertext=low["file"], hovertemplate="%{hovertext}: %{y}<extra></extra>"))
            fig_f.update_xaxes(title="frame")
            fig_f.update_yaxes(title="matched stars", rangemode="tozero")
            fig_f.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_f, width="stretch", key="vyvar_mq_frames")
            if len(low):
                with st.expander(f"{len(low)} frame(s) with few stars"):
                    st.dataframe(low[["frame", "file", "n_detected", "n_matched"]], hide_index=True, width="stretch")
        else:
            st.caption("No per-frame catalog index.")
