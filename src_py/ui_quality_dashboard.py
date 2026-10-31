"""FITS QA tab - Quality Dashboard (information only, QC-PER-GROUP-2026-10).

Shows what the frame QC of RUN VYVAR decided, per channel (observation group
FILTER|EXPTIME|BINNING): FWHM and sky level of every frame (green kept / red
rejected, dashed line = the channel's FWHM limit) and a preview of the frame
selected as MASTERSTAR source for each channel. Nothing here changes the run:
rejections and the MASTERSTAR source are decided by ``frame_qc`` (same in the
GUI and the night runner).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from astropy.time import Time

from config import AppConfig
from database import VyvarDatabase
from frame_qc import evaluate_frame_qc, group_key_of, load_frame_qc
from importer import quicklook_preview_png_bytes
from pipeline_ui_helpers import _resolve_light_fits_for_quality_inspection
from ui_components import DRAFT_CENTER_DE_STATE_KEY, DRAFT_CENTER_RA_STATE_KEY
from ui_session import resolve_draft_dir

X_AXIS_FRAME_TITLE = "Frame number (Frame Index)"

_GROUP_COLORS = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
)
_KEPT = "#27ae60"
_REJECTED = "#c0392b"


def group_label(group_key: str) -> str:
    """'g|120|4' -> 'g 120 s bin4'."""
    parts = str(group_key).split("|")
    if len(parts) == 3:
        return f"{parts[0]} {parts[1]} s bin{parts[2]}"
    return str(group_key)


def _preview_path_from_plotly_state(state: Any) -> str | None:
    if state is None:
        return None
    sel = getattr(state, "selection", None)
    if sel is None and isinstance(state, dict):
        sel = state.get("selection")
    if not sel:
        return None
    pts = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    if not pts:
        return None
    p0 = pts[0]
    if not isinstance(p0, dict):
        return None
    cd = p0.get("customdata")
    if cd is None:
        return None
    if isinstance(cd, (list, tuple, np.ndarray)):
        return str(cd[0]) if len(cd) else None
    return str(cd)


def _utc_str(jd_val: Any) -> str:
    try:
        jf = float(jd_val)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(jf):
        return "-"
    return Time(jf, format="jd", scale="utc").to_datetime().strftime("%H:%M:%S")


def build_quality_frame(rows: list[dict[str, Any]], ap: Path | None) -> pd.DataFrame:
    """Manifest light rows -> plotting frame (group, frame index, preview path, status)."""
    df = pd.DataFrame(rows)
    for col in ("FWHM", "SKY_LEVEL", "IS_REJECTED", "INSPECTION_JD", "QC_REASON", "FILE_PATH",
                "ELONGATION_MEAN", "NOISE_SIGMA", "STAR_COUNT"):
        if col not in df.columns:
            df[col] = np.nan
    df["FWHM"] = pd.to_numeric(df["FWHM"], errors="coerce")
    df["SKY_LEVEL"] = pd.to_numeric(df["SKY_LEVEL"], errors="coerce")
    df["IS_REJECTED"] = pd.to_numeric(df["IS_REJECTED"], errors="coerce").fillna(0).astype(int)
    df["GROUP"] = [group_key_of(r) for r in rows]
    df = df.reset_index(drop=True)
    df["frame_index"] = np.arange(1, len(df) + 1, dtype=int)
    df["_utc"] = [_utc_str(v) for v in df["INSPECTION_JD"]]
    df["_name"] = [Path(str(p)).name for p in df["FILE_PATH"]]
    df["_reason"] = [str(r) if isinstance(r, str) and r else "" for r in df["QC_REASON"]]
    raw_paths = [Path(str(p)) for p in df["FILE_PATH"].tolist()]
    if ap is not None and ap.is_dir():
        df["_preview"] = [
            str(_resolve_light_fits_for_quality_inspection(ap, rp) or rp) for rp in raw_paths
        ]
    else:
        df["_preview"] = [str(rp) for rp in raw_paths]
    return df


def _metric_figure(
    df: pd.DataFrame,
    *,
    y_col: str,
    y_title: str,
    title: str,
    limits: dict[str, float | None] | None = None,
) -> go.Figure:
    fig = go.Figure()
    for i, gk in enumerate(sorted(df["GROUP"].unique())):
        sub = df[df["GROUP"] == gk].sort_values("frame_index")
        color = _GROUP_COLORS[i % len(_GROUP_COLORS)]
        mcol = np.where(sub["IS_REJECTED"].to_numpy() == 1, _REJECTED, _KEPT)
        cd = np.stack(
            [
                sub["_preview"].astype(str).to_numpy(),
                sub["_name"].astype(str).to_numpy(),
                sub["_utc"].astype(str).to_numpy(),
                np.where(sub["IS_REJECTED"].to_numpy() == 1, "rejected: " + sub["_reason"].astype(str), "kept"),
            ],
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["frame_index"],
                y=sub[y_col],
                mode="lines+markers",
                name=group_label(gk),
                legendgroup=str(gk),
                line=dict(color=color, width=1.5),
                connectgaps=False,
                marker=dict(size=9, color=list(mcol), line=dict(width=1, color=color)),
                customdata=cd,
                hovertemplate=(
                    "Frame #%{x}<br>%{customdata[1]}<br>UTC %{customdata[2]}<br>"
                    + y_title
                    + " %{y:.4g}<br>%{customdata[3]}<extra></extra>"
                ),
            )
        )
        lim = (limits or {}).get(gk)
        if lim is not None and math.isfinite(float(lim)) and not sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=[int(sub["frame_index"].min()) - 0.5, int(sub["frame_index"].max()) + 0.5],
                    y=[float(lim), float(lim)],
                    mode="lines",
                    line=dict(color=color, width=2, dash="dash"),
                    name=f"{group_label(gk)} limit {float(lim):.2f}",
                    legendgroup=str(gk),
                    hoverinfo="skip",
                )
            )
    if not fig.data:
        fig.add_annotation(text="No data in manifest files[].", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        title=title,
        xaxis_title=X_AXIS_FRAME_TITLE,
        yaxis_title=y_title,
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def build_qc_summary_table(df: pd.DataFrame, qc_groups: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per channel: frame counts, rejection reasons and medians of the QC metrics."""
    def _med(s: pd.Series, fmt: str) -> str:
        v = pd.to_numeric(s, errors="coerce").dropna()
        return format(float(v.median()), fmt) if len(v) else "-"

    def _rng(s: pd.Series, fmt: str) -> str:
        v = pd.to_numeric(s, errors="coerce").dropna()
        return f"{format(float(v.min()), fmt)} - {format(float(v.max()), fmt)}" if len(v) else "-"

    recs: list[dict[str, Any]] = []
    for gk in sorted(df["GROUP"].unique()):
        sub = df[df["GROUP"] == gk]
        kept = sub[sub["IS_REJECTED"] == 0]
        reasons = (
            sub.loc[sub["IS_REJECTED"] == 1, "_reason"].astype(str).str.split(" \\(").str[0].value_counts()
        )
        g = qc_groups.get(gk) or {}
        tr = [v.get("transparency") for p, v in (g.get("frames") or {}).items()
              if v.get("transparency") is not None and p not in (g.get("rejected") or {})]
        best = g.get("best_path")
        recs.append({
            "Channel": group_label(gk),
            "Frames": int(len(sub)),
            "Kept": int(len(kept)),
            "Rejected": ", ".join(f"{n}x {r}" for r, n in reasons.items()) or "0",
            "FWHM px (median, range)": f"{_med(kept['FWHM'], '.2f')} ({_rng(kept['FWHM'], '.2f')})",
            "Elongation": _med(kept["ELONGATION_MEAN"], ".3f"),
            "Sky ADU (median, range)": f"{_med(kept['SKY_LEVEL'], '.1f')} ({_rng(kept['SKY_LEVEL'], '.1f')})",
            "Noise ADU": _med(kept["NOISE_SIGMA"], ".2f"),
            "Stars (5 sigma)": _med(kept["STAR_COUNT"], ".0f"),
            "Transparency (min kept)": format(min(tr), ".2f") if tr else "-",
            "MASTERSTAR source": Path(str(best).replace(chr(92), "/")).name if best else "-",
        })
    return pd.DataFrame(recs)


def _seed_draft_center(df_rows: pd.DataFrame) -> None:
    """Seed VARSTREM field centre from the median frame pointing when it is unset."""
    try:
        _ra_cur = float(st.session_state.get(DRAFT_CENTER_RA_STATE_KEY, 0.0))
        _de_cur = float(st.session_state.get(DRAFT_CENTER_DE_STATE_KEY, 0.0))
    except (TypeError, ValueError):
        _ra_cur, _de_cur = 0.0, 0.0
    if abs(_ra_cur) > 1e-9 or abs(_de_cur) > 1e-9:
        return
    if "RA" not in df_rows.columns or "DE" not in df_rows.columns:
        return
    ra = pd.to_numeric(df_rows.get("RA"), errors="coerce").dropna()
    de = pd.to_numeric(df_rows.get("DE"), errors="coerce").dropna()
    if ra.empty or de.empty:
        return
    st.session_state["vyvar_pending_center_ra"] = float(ra.median())
    st.session_state["vyvar_pending_center_de"] = float(de.median())
    st.session_state["center_ra"] = float(ra.median())
    st.session_state["center_de"] = float(de.median())


@st.cache_data(max_entries=64, show_spinner=False)
def _preview_png_cached(path: str, stamp: tuple) -> bytes:
    """G14-26: the ZScale preview of a frame once per file state (every channel tab and plot click
    reruns from the cache)."""
    return quicklook_preview_png_bytes(path)


def _show_preview(path: str, caption: str) -> None:
    if not path or not Path(path).is_file():
        st.caption(f"File not found: {Path(path).name if path else '-'}")
        return
    if "calibrated" not in {part.casefold() for part in Path(path).parts}:
        # The calibrated frames are deleted after a finished run (OUTPUT-REVIEW-2026-10).
        caption = f"{caption} - source frame (calibrated frames are removed after the run)"
    try:
        from ui_cache import file_stamp  # noqa: PLC0415

        st.image(_preview_png_cached(str(path), file_stamp(path)), caption=caption, width="stretch")
    except (OSError, ValueError) as exc:
        st.caption(f"Preview failed: {exc}")


def render_quality_dashboard(
    *,
    db: VyvarDatabase,
    draft_id: int | None,
    archive_text: str,
    cfg: AppConfig,
) -> None:
    if draft_id is None or int(draft_id) <= 0:
        st.info("No processed field loaded - run RUN VYVAR or load one above (Load processed field).")
        return
    did = int(draft_id)
    default_ap = resolve_draft_dir(draft_id=did, archive_root=cfg.archive_root) or ""
    ap_str = (archive_text or "").strip() or default_ap
    ap = Path(ap_str) if ap_str else None

    rows = db.fetch_draft_light_rows_for_quality(did)
    if not rows:
        st.warning("No light frames in manifest files[] for this processed field.")
        return
    raw_df = pd.DataFrame(rows)
    _seed_draft_center(raw_df)
    df = build_quality_frame(rows, ap)

    qc_doc = load_frame_qc(ap) if ap is not None else {}
    qc_groups: dict[str, dict[str, Any]] = dict(qc_doc.get("groups") or {})
    if not qc_groups:
        # Draft processed before per-channel QC: compute for display only (nothing is written).
        qc_groups = {
            gk: g.to_dict()
            for gk, g in evaluate_frame_qc(
                rows, k=float(cfg.auto_fwhm_k_factor), reject=bool(cfg.auto_fwhm_enabled)
            ).items()
        }
        st.caption(
            "This field was processed before per-channel frame QC - limits and best frames are "
            "computed here for display only."
        )

    st.dataframe(build_qc_summary_table(df, qc_groups), hide_index=True, width="stretch")
    fwhm_limits = {gk: g.get("fwhm_limit") for gk, g in qc_groups.items()}
    key_fwhm = f"vyvar_qdash_plot_fwhm_{did}"
    key_sky = f"vyvar_qdash_plot_sky_{did}"
    st.plotly_chart(
        _metric_figure(
            df,
            y_col="FWHM",
            y_title="FWHM (px)",
            title="FWHM per channel - green kept, red rejected (hover shows the reason)",
            limits=fwhm_limits,
        ),
        key=key_fwhm,
        on_select="rerun",
        selection_mode="points",
        width="stretch",
    )
    st.plotly_chart(
        _metric_figure(
            df,
            y_col="SKY_LEVEL",
            y_title="Sky level (ADU)",
            title="Background sky level per channel - green kept, red rejected",
        ),
        key=key_sky,
        on_select="rerun",
        selection_mode="points",
        width="stretch",
    )

    p_sel = _preview_path_from_plotly_state(st.session_state.get(key_fwhm)) or _preview_path_from_plotly_state(
        st.session_state.get(key_sky)
    )
    if p_sel:
        _show_preview(p_sel, f"Clicked frame: {Path(p_sel).name}")

    st.markdown("**MASTERSTAR source per channel**")
    gks = sorted(qc_groups)
    preview_by_raw = dict(zip(df["FILE_PATH"].astype(str), df["_preview"].astype(str)))
    for tab, gk in zip(st.tabs([group_label(g) for g in gks]), gks):
        g = qc_groups[gk]
        with tab:
            best = g.get("best_path")
            lim = g.get("fwhm_limit")
            _bt = g.get("best_transparency")
            st.caption(
                f"{g.get('n_frames', 0)} frames, {g.get('n_rejected', 0)} rejected"
                + (f", FWHM limit {float(lim):.2f} px" if lim is not None else "")
                + (
                    f" | selected: {Path(str(best).replace(chr(92), '/')).name} (FWHM {float(g['best_fwhm']):.2f} px"
                    + (f", noise {float(g['best_noise']):.2f} ADU" if g.get("best_noise") is not None else "")
                    + (f", transparency {float(_bt):.2f}" if _bt is not None else "")
                    + ")"
                    if best and g.get("best_fwhm") is not None
                    else ""
                )
            )
            if best:
                _show_preview(preview_by_raw.get(str(best), str(best)), f"{group_label(gk)}: {Path(best).name}")
            else:
                st.warning("No usable frame in this channel.")
