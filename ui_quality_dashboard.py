"""Streamlit Quality Dashboard: OBS_FILES metrics, Plotly, FITS preview, data editor."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy.time import Time
import streamlit as st

from database import VyvarDatabase
from infolog import log_event
from importer import quicklook_preview_png_bytes
from pipeline import _resolve_light_fits_for_quality_inspection
from ui_components import DRAFT_CENTER_DE_STATE_KEY, DRAFT_CENTER_RA_STATE_KEY

X_AXIS_FRAME_TITLE = "Číslo snímky (Frame Index)"


def _compute_masterstar_score(df: pd.DataFrame) -> pd.Series:
    """Skóre vhodnosti snímky pre MASTERSTAR (vyššie = lepšie).
    Kombinácia: FWHM (nízke), elongation (nízka = bez stopy), star_count (vysoký), sky_level (nízke).
    Každá metrika sa normalizuje 0-1, váhy sú empirické.
    """
    score = pd.Series(0.0, index=df.index)
    n = len(df)
    if n == 0:
        return score

    def _norm_inverse(s: pd.Series) -> pd.Series:
        """Nízka hodnota = lepšie → invertovaná normalizácia 0-1."""
        mn, mx = s.min(), s.max()
        if not math.isfinite(mn) or not math.isfinite(mx) or mx <= mn:
            return pd.Series(1.0, index=s.index)
        return 1.0 - (s - mn) / (mx - mn)

    def _norm_direct(s: pd.Series) -> pd.Series:
        """Vysoká hodnota = lepšie → priama normalizácia 0-1."""
        mn, mx = s.min(), s.max()
        if not math.isfinite(mn) or not math.isfinite(mx) or mx <= mn:
            return pd.Series(1.0, index=s.index)
        return (s - mn) / (mx - mn)

    fwhm = pd.to_numeric(df["FWHM"], errors="coerce")
    elong = pd.to_numeric(df.get("ELONGATION_MEAN", pd.Series(np.nan, index=df.index)), errors="coerce")
    stars = pd.to_numeric(df["STAR_COUNT"], errors="coerce")
    sky = pd.to_numeric(df["SKY_LEVEL"], errors="coerce")

    if fwhm.notna().sum() >= 2:
        score += 0.45 * _norm_inverse(fwhm.fillna(fwhm.max()))
    if elong.notna().sum() >= 2:
        score += 0.30 * _norm_inverse(elong.fillna(elong.max()))
    if stars.notna().sum() >= 2:
        score += 0.15 * _norm_direct(stars.fillna(stars.min()))
    if sky.notna().sum() >= 2:
        score += 0.10 * _norm_inverse(sky.fillna(sky.max()))

    return score

_FILTER_LINE_COLORS = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
)


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


def _make_is_rejected_persist_handler(
    draft_id: int,
    editor_key: str,
    db: VyvarDatabase,
    *,
    editor_row_ids: list[int],
):
    def _on_change() -> None:
        raw = st.session_state.get(editor_key)
        if raw is None:
            return
        updates: list[tuple[int, int]] = []
        # Streamlit data_editor stores edits as a dict: {"edited_rows": {row_idx: {col: value}} , ...}
        if isinstance(raw, dict):
            edited = raw.get("edited_rows") or {}
            if isinstance(edited, dict):
                for k, changes in edited.items():
                    try:
                        idx = int(k)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(changes, dict):
                        continue
                    if "IS_REJECTED" not in changes:
                        continue
                    if idx < 0 or idx >= len(editor_row_ids):
                        continue
                    rid = int(editor_row_ids[idx])
                    v = changes.get("IS_REJECTED")
                    rej = 1 if (v is True or v == 1 or str(v).lower() in ("true", "1")) else 0
                    updates.append((rid, rej))
        elif isinstance(raw, pd.DataFrame):
            if raw.empty or "ID" not in raw.columns:
                return
            for _, row in raw.iterrows():
                try:
                    rid = int(row["ID"])
                except (TypeError, ValueError):
                    continue
                v = row.get("IS_REJECTED")
                rej = 1 if (v is True or v == 1 or str(v).lower() in ("true", "1")) else 0
                updates.append((rid, rej))
        if updates:
            db.bulk_update_obs_file_is_rejected(int(draft_id), updates)

    return _on_change


def _jd_and_utc_strings(jd_val: Any) -> tuple[str, str]:
    if jd_val is None:
        return "—", "—"
    try:
        jf = float(jd_val)
    except (TypeError, ValueError):
        return "—", "—"
    if not math.isfinite(jf):
        return "—", "—"
    t = Time(jf, format="jd", scale="utc")
    dt = t.to_datetime()
    return f"{jf:.8f}", dt.strftime("%H:%M:%S")


def _prepare_hover_time_fields(df: pd.DataFrame) -> pd.DataFrame:
    """JD a civilný čas (UTC) len pre hover — os X je ``frame_index``."""
    df = df.copy()
    jd = pd.to_numeric(df["INSPECTION_JD"], errors="coerce")
    js, us = [], []
    for v in jd:
        a, b = _jd_and_utc_strings(v)
        js.append(a)
        us.append(b)
    df["_jd_str"] = js
    df["_utc_str"] = us
    return df


def _add_traces_by_filter(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    y_col: str,
    y_hover_label: str,
) -> None:
    flt_series = df["FILTER"].fillna("NoFilter").astype(str)
    uniq = sorted(flt_series.unique())
    for i, flt in enumerate(uniq):
        sub = df.loc[flt_series == flt].copy()
        sub = sub.sort_values("frame_index")
        sub = sub.dropna(subset=[y_col], how="any")
        if sub.empty:
            continue
        line_c = _FILTER_LINE_COLORS[i % len(_FILTER_LINE_COLORS)]
        mcol = np.where(sub["REJECTED_AUTO"].to_numpy() == 1, "red", "green")
        _fn = sub["FILE_PATH"].astype(str).apply(lambda s: Path(s).name).to_numpy()
        cd = np.stack(
            [
                sub["_preview"].astype(str).to_numpy(),
                _fn,
                sub["_jd_str"].astype(str).to_numpy(),
                sub["_utc_str"].astype(str).to_numpy(),
            ],
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["frame_index"],
                y=sub[y_col],
                mode="lines+markers",
                name=str(flt),
                legendgroup=str(flt),
                line=dict(color=line_c, width=1.5),
                marker=dict(size=9, color=list(mcol), line=dict(width=1, color=line_c)),
                customdata=cd,
                hovertemplate=(
                    "Snímka č. %{x}<br>"
                    "Súbor %{customdata[1]}<br>"
                    "JD %{customdata[2]}<br>"
                    "Čas (UTC) %{customdata[3]}<br>"
                    + y_hover_label
                    + "=%{y:.4g}<extra></extra>"
                ),
            )
        )


def _add_fwhm_traces_with_limit(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    fwhm_limit_px: float,
) -> None:
    """FWHM vs ``frame_index``; farby podľa prahu. Červenú hranicu kreslí ``render_quality_dashboard`` cez ``add_hline``."""
    lim = float(fwhm_limit_px)
    lim_active = math.isfinite(lim) and lim > 0.0
    flt_series = df["FILTER"].fillna("NoFilter").astype(str)
    uniq = sorted(flt_series.unique())
    for i, flt in enumerate(uniq):
        sub = df.loc[flt_series == flt].copy()
        sub = sub.sort_values("frame_index")
        if sub.empty:
            continue
        line_c = _FILTER_LINE_COLORS[i % len(_FILTER_LINE_COLORS)]
        fwhm = pd.to_numeric(sub["FWHM"], errors="coerce")
        arr = fwhm.to_numpy(dtype=float)
        if lim_active:
            ok = np.isfinite(arr) & (arr <= lim)
            mcol = np.where(ok, "#27ae60", "#c0392b")
        else:
            mcol = np.where(np.isfinite(arr), "#27ae60", "#95a5a6")
        _fn = sub["FILE_PATH"].astype(str).apply(lambda s: Path(s).name).to_numpy()
        exp_raw = pd.to_numeric(sub["EXPTIME"], errors="coerce")
        exp_str = np.array(
            [
                f"{float(e):.3f} s" if pd.notna(e) and math.isfinite(float(e)) else "—"
                for e in exp_raw.to_numpy()
            ],
            dtype=object,
        )
        fwhm_str = np.array(
            [f"{float(v):.6f}" if np.isfinite(v) else "—" for v in arr],
            dtype=object,
        )
        jd_hover = sub["_jd_str"].astype(str).to_numpy()
        utc_hover = sub["_utc_str"].astype(str).to_numpy()
        cd = np.stack(
            [
                sub["_preview"].astype(str).to_numpy(),
                _fn,
                exp_str.astype(str),
                fwhm_str.astype(str),
                jd_hover,
                utc_hover,
            ],
            axis=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["frame_index"],
                y=fwhm,
                mode="lines+markers",
                name=str(flt),
                legendgroup=str(flt),
                line=dict(color=line_c, width=1.5),
                connectgaps=False,
                marker=dict(size=10, color=list(mcol), line=dict(width=1, color=line_c)),
                customdata=cd,
                hovertemplate=(
                    "Snímka č. %{x}<br>"
                    "Súbor %{customdata[1]}<br>"
                    "JD %{customdata[4]}<br>"
                    "Čas (UTC) %{customdata[5]}<br>"
                    "Expozícia %{customdata[2]}<br>"
                    "FWHM %{customdata[3]} px<extra></extra>"
                ),
            )
        )


def render_quality_dashboard(
    *,
    db: VyvarDatabase,
    draft_id: int | None,
    archive_text: str,
) -> None:
    st.subheader("Quality Dashboard")
    st.caption(
        "Metriky z tabuľky `OBS_FILES` (DAO analýza). Klik na bod v grafe → náhľad FITS. "
        "Stĺpec **IS_REJECTED** sa ukladá okamžite do databázy."
    )

    last_res = st.session_state.get("vyvar_last_import_result")
    default_ap = ""
    if last_res and getattr(last_res, "archive_path", None):
        default_ap = str(last_res.archive_path)
    ap_path_str = (archive_text or "").strip() or default_ap
    ap = Path(ap_path_str) if ap_path_str else None
    if ap_path_str:
        st.caption(f"Archív (session / import): `{ap_path_str}`")

    if draft_id is None or int(draft_id) <= 0:
        st.info("Najprv importujte Draft (session `vyvar_last_draft_id`).")
        return

    did = int(draft_id)
    def _stage_center_update(ra_val: Any, de_val: Any) -> None:
        """Stage center updates for app-level apply before widgets instantiate."""
        try:
            if ra_val is not None and math.isfinite(float(ra_val)):
                st.session_state["vyvar_pending_center_ra"] = float(ra_val)
                st.session_state["center_ra"] = float(ra_val)
            if de_val is not None and math.isfinite(float(de_val)):
                st.session_state["vyvar_pending_center_de"] = float(de_val)
                st.session_state["center_de"] = float(de_val)
        except (TypeError, ValueError):
            return

    _last_job = st.session_state.get("vyvar_last_job_output") or {}
    if isinstance(_last_job, dict) and _last_job.get("job_kind") == "analyze":
        _rq = _last_job.get("ram_qc_summary") or {}
        _stage_center_update(_rq.get("median_ra_deg"), _rq.get("median_de_deg"))

    if "fwhm_threshold" not in st.session_state:
        _fb = st.session_state.get("fwhm_limit")
        if _fb is None:
            _fb = st.session_state.get("vyvar_ui_reject_fwhm", 0.0)
        try:
            st.session_state["fwhm_threshold"] = float(_fb)
        except (TypeError, ValueError):
            st.session_state["fwhm_threshold"] = 0.0
    rows = db.fetch_draft_light_rows_for_quality(did)
    if not rows:
        st.warning("Pre tento draft nie sú v `OBS_FILES` žiadne light snímky.")
        return

    st.caption("**Center RA/DE** uprav v paneli VARSTREM (Krok 2).")

    df = pd.DataFrame(rows)
    try:
        _ra_cur = float(st.session_state.get(DRAFT_CENTER_RA_STATE_KEY, 0.0))
        _de_cur = float(st.session_state.get(DRAFT_CENTER_DE_STATE_KEY, 0.0))
        _need_seed = (abs(_ra_cur) < 1e-9 and abs(_de_cur) < 1e-9) or (
            DRAFT_CENTER_RA_STATE_KEY not in st.session_state or DRAFT_CENTER_DE_STATE_KEY not in st.session_state
        )
        if _need_seed:
            _ra_med = pd.to_numeric(df.get("RA"), errors="coerce").dropna()
            _de_med = pd.to_numeric(df.get("DE"), errors="coerce").dropna()
            if not _ra_med.empty and not _de_med.empty:
                _stage_center_update(float(_ra_med.median()), float(_de_med.median()))
    except Exception:  # noqa: BLE001
        pass
    try:
        _calib_t = (
            df.get("CALIB_TYPE", pd.Series([""] * len(df)))
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        _is_cal = pd.to_numeric(df.get("IS_CALIBRATED"), errors="coerce").fillna(1).astype(int)
        if bool((_calib_t.eq("PASSTHROUGH") | _is_cal.eq(0)).any()):
            st.warning("POZOR: Zobrazujete nekalibrované dáta (Passthrough Mode).")
    except Exception:  # noqa: BLE001
        pass
    for col in (
        "CALIB_FLAGS",
        "FWHM",
        "SKY_LEVEL",
        "STAR_COUNT",
        "REJECTED_AUTO",
        "IS_REJECTED",
        "INSPECTION_JD",
        "FILTER",
        "RA",
        "DE",
        "EXPTIME",
        "DRIFT",
        "DRIFT_DRA",
        "DRIFT_DDE",
        "ROUNDNESS_MEAN",
        "ELONGATION_MEAN",
    ):
        if col not in df.columns:
            df[col] = np.nan

    df["FWHM"] = pd.to_numeric(df["FWHM"], errors="coerce")
    df["SKY_LEVEL"] = pd.to_numeric(df["SKY_LEVEL"], errors="coerce")
    df["STAR_COUNT"] = pd.to_numeric(df["STAR_COUNT"], errors="coerce").fillna(0).astype(int)
    df["REJECTED_AUTO"] = pd.to_numeric(df["REJECTED_AUTO"], errors="coerce").fillna(0).astype(int)
    df["IS_REJECTED"] = pd.to_numeric(df["IS_REJECTED"], errors="coerce").fillna(0).astype(int)
    df["DRIFT"] = pd.to_numeric(df["DRIFT"], errors="coerce")
    df["DRIFT_DRA"] = pd.to_numeric(df["DRIFT_DRA"], errors="coerce")
    df["DRIFT_DDE"] = pd.to_numeric(df["DRIFT_DDE"], errors="coerce")
    df["ROUNDNESS_MEAN"] = pd.to_numeric(df["ROUNDNESS_MEAN"], errors="coerce")
    df["ELONGATION_MEAN"] = pd.to_numeric(df["ELONGATION_MEAN"], errors="coerce")
    df["CALIB_FLAGS"] = df["CALIB_FLAGS"].fillna("").astype(str)

    df = df.reset_index(drop=True)
    df["frame_index"] = np.arange(1, len(df) + 1, dtype=int)
    df = _prepare_hover_time_fields(df)

    raw_paths = [Path(str(p)) for p in df["FILE_PATH"].tolist()]
    if ap is not None and ap.is_dir():
        df["_preview"] = [
            str(p) if p is not None else str(rp)
            for rp, p in zip(
                raw_paths,
                [_resolve_light_fits_for_quality_inspection(ap, rp) for rp in raw_paths],
            )
        ]
    else:
        df["_preview"] = [str(rp) for rp in raw_paths]

    n_total = len(df)
    n_auto = int((df["REJECTED_AUTO"] == 1).sum())
    n_man = int((df["IS_REJECTED"] == 1).sum())
    st.markdown(
        f"**Total Frames:** {n_total} &nbsp;|&nbsp; **Auto-Rejected:** {n_auto} &nbsp;|&nbsp; "
        f"**Manually Excluded:** {n_man}"
    )

    key_fwhm = f"vyvar_qdash_plot_fwhm_{did}"
    key_sky = f"vyvar_qdash_plot_sky_{did}"

    st.markdown("#### FWHM limit (px)")
    st.caption("Rovnaký prah ako **MAKE MASTERSTAR** / detrend. Vypnuté = 0 (bez filtra).")
    fwhm_enabled = st.toggle(
        "Zapnúť FWHM filter",
        value=bool(float(st.session_state.get("fwhm_threshold", 0.0)) > 0.0),
        key="vyvar_qdash_fwhm_enable",
        help="Ak je vypnuté, FWHM limit sa nastaví na 0 (žiadny filter).",
    )
    _cur_raw = float(st.session_state.get("fwhm_threshold", 0.0))
    if _cur_raw > 0:
        st.session_state["vyvar_qdash_last_fwhm_nonzero"] = float(_cur_raw)
    _base = float(st.session_state.get("vyvar_qdash_last_fwhm_nonzero") or 0.0)
    if not math.isfinite(_base) or _base <= 0:
        _med = pd.to_numeric(df["FWHM"], errors="coerce").dropna()
        _base = float(_med.median()) if len(_med) else 4.0
    _base = max(0.05, float(_base))
    _pct = 0.15
    _lo = max(0.0, float(_base * (1.0 - _pct)))
    _hi = float(_base * (1.0 + _pct))
    slider_lo: float | None = None
    slider_hi: float | None = None
    if fwhm_enabled:
        st.slider(
            "FWHM limit",
            min_value=float(_lo),
            max_value=float(_hi),
            value=float(min(max(_cur_raw if _cur_raw > 0 else _base, _lo), _hi)),
            step=0.01,
            key="fwhm_threshold",
            help="Posuvník je ±15% okolo poslednej použitej hodnoty (kvôli pohodliu).",
        )
        slider_lo = float(_lo)
        slider_hi = float(_hi)
    else:
        st.session_state["fwhm_threshold"] = 0.0

    fwhm_limit_raw = float(st.session_state.get("fwhm_threshold", 0.0))
    # Pre graf a filtre: 0 = bez limitu (žiadna červená čiara, žiadne rozťahovanie osi).
    fwhm_limit_plot = float(fwhm_limit_raw) if fwhm_limit_raw > 0.0 else None
    # Pre zvyšok UI (reject indikácia) potrebujeme číslo; None -> veľké číslo.
    fwhm_limit = float(fwhm_limit_raw if fwhm_limit_raw > 0.0 else 99.0)
    st.session_state["fwhm_limit"] = float(fwhm_limit)
    st.session_state["vyvar_ui_reject_fwhm"] = float(fwhm_limit)
    _last_fwhm_log = st.session_state.get("vyvar_last_logged_fwhm_limit")
    if _last_fwhm_log is None or abs(float(_last_fwhm_log) - float(fwhm_limit_raw)) > 1e-9:
        log_event(f"FWHM threshold (px) changed to {float(fwhm_limit_raw):.2f}")
        st.session_state["vyvar_last_logged_fwhm_limit"] = float(fwhm_limit_raw)

    fig_f = go.Figure()
    _add_fwhm_traces_with_limit(fig_f, df, fwhm_limit_px=float(fwhm_limit_plot or 0.0))
    if not fig_f.data:
        fig_f.add_annotation(
            text="Žiadne body vo FWHM grafe (skontrolujte FILTER / dáta v OBS_FILES).",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    if fwhm_limit_plot is not None and math.isfinite(float(fwhm_limit_plot)):
        fig_f.add_hline(y=float(fwhm_limit_plot), line_dash="dash", line_color="red")
    try:
        _job = st.session_state.get("vyvar_last_job_output") or {}
        _flip_i = _job.get("rotation_flip_first_index_1based")
        if _flip_i is not None:
            fi = int(_flip_i)
            if 1 <= fi <= int(len(df)):
                fig_f.add_vline(
                    x=fi,
                    line_width=2,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text="Meridian flip / 180° rotácia",
                    annotation_position="top left",
                )
    except Exception:  # noqa: BLE001
        pass
    fig_f.update_layout(
        title="FWHM — červená čiara = limit z posuvníka vyššie; zelená ≤ limit, červená > limit",
        xaxis_title=X_AXIS_FRAME_TITLE,
        yaxis_title="FWHM (px)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    # Pri zapnutom FWHM filtri drž Y-os v rozsahu slideru (±15%).
    if slider_lo is not None and slider_hi is not None and slider_hi > slider_lo:
        fig_f.update_yaxes(range=[float(slider_lo), float(slider_hi)])
    st.plotly_chart(
        fig_f, key=key_fwhm, on_select="rerun", selection_mode="points", use_container_width=True
    )

    fig_s = go.Figure()
    _add_traces_by_filter(fig_s, df, y_col="SKY_LEVEL", y_hover_label="Sky level")
    fig_s.update_layout(
        title="Background sky level (podľa filtra; červená = auto-outlier)",
        xaxis_title=X_AXIS_FRAME_TITLE,
        yaxis_title="SKY_LEVEL",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(
        fig_s, key=key_sky, on_select="rerun", selection_mode="points", use_container_width=True
    )

    # --- MASTERSTAR kandidáti (presunuté pod Sky graf) ---
    _ms_eligible = df[df["IS_REJECTED"] == 0].copy()
    if fwhm_limit_raw > 0.0:
        _ms_eligible = _ms_eligible[
            _ms_eligible["FWHM"].notna() & (_ms_eligible["FWHM"] <= fwhm_limit_raw)
        ]
    if not _ms_eligible.empty:
        _ms_eligible = _ms_eligible.copy()
        _ms_eligible["_ms_score"] = _compute_masterstar_score(_ms_eligible)
        _ms_eligible = _ms_eligible.sort_values("_ms_score", ascending=False).reset_index(drop=True)
        _top5 = _ms_eligible.head(5)

        st.markdown("#### Navrhované snímky pre MASTERSTAR")
        st.caption(
            "VYVAR zoradil snímky podľa skóre (FWHM 45 %, elongácia 30 %, počet hviezd 15 %, sky 10 %). "
            "Systém automaticky predvyberie TOP1, ale nič sa neuloží bez potvrdenia."
        )
        _cand_df = pd.DataFrame(
            {
                "Poradie": range(1, len(_top5) + 1),
                "Frame": _top5["frame_index"].values,
                "Filename": [Path(str(p)).name for p in _top5["FILE_PATH"].values],
                "FWHM": _top5["FWHM"].values,
                "Elongácia": (
                    _top5["ELONGATION_MEAN"].values
                    if "ELONGATION_MEAN" in _top5.columns
                    else [None] * len(_top5)
                ),
                "Hviezdy": _top5["STAR_COUNT"].values,
                "Sky": _top5["SKY_LEVEL"].values,
                "Skóre": _top5["_ms_score"].round(3).values,
                "_path": _top5["_preview"].values,
            }
        )
        st.dataframe(
            _cand_df.drop(columns=["_path"]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "FWHM": st.column_config.NumberColumn("FWHM", format="%.4f"),
                "Elongácia": st.column_config.NumberColumn("Elongácia", format="%.3f"),
                "Sky": st.column_config.NumberColumn("Sky", format="%.2g"),
                "Skóre": st.column_config.NumberColumn("Skóre", format="%.3f"),
            },
        )

        _opts = [str(p) for p in _cand_df["Filename"].tolist()]
        _paths = [str(p) for p in _top5["_preview"].tolist()]
        _paths_by_name = dict(zip(_opts, _paths))
        _db_paths_by_name = dict(zip(_opts, [str(p) for p in _top5["FILE_PATH"].tolist()]))
        _default_name = str(_opts[0]) if _opts else ""
        pick_name = st.selectbox(
            "Predvybraný MASTERSTAR kandidát (z tabuľky):",
            options=_opts,
            index=0,
            key="vyvar_qdash_ms_pick_name",
        )
        st.session_state["vyvar_ms_candidates"] = [str(p) for p in _top5["FILE_PATH"].values]
        st.session_state["vyvar_ms_candidate_top1_path"] = str(_top5["FILE_PATH"].iloc[0])

        _pick_preview = _paths_by_name.get(str(pick_name), "")
        if _pick_preview and Path(_pick_preview).is_file():
            try:
                png_b = quicklook_preview_png_bytes(_pick_preview)
                st.image(png_b, caption=f"Náhľad (auto): {Path(_pick_preview).name}", use_container_width=True)
            except Exception:  # noqa: BLE001
                pass

        if st.button(
            "Potvrdiť výber MASTERSTAR",
            key="vyvar_qdash_confirm_masterstar",
            type="primary",
            help="Zapíše vybranú snímku ako MASTERSTAR pre tento draft (DB).",
        ):
            p_db = _db_paths_by_name.get(str(pick_name), "")
            if p_db:
                try:
                    _db2 = VyvarDatabase(db.db_path)
                    try:
                        _db2.set_obs_draft_masterstar_path(int(did), str(p_db))
                    finally:
                        _db2.conn.close()
                except Exception:  # noqa: BLE001
                    pass
                log_event(f"MASTERSTAR potvrdený z tabuľky: {Path(p_db).name}")
                st.success(f"Nastavené: `{Path(p_db).name}` bude použitý ako MASTERSTAR.")
    else:
        st.info("Žiadne vhodné snímky pre MASTERSTAR (skontroluj FWHM limit alebo IS_REJECTED).")
        st.session_state["vyvar_ms_candidates"] = []
        st.session_state["vyvar_ms_candidate_top1_path"] = ""
    st.markdown("---")
    # --- koniec MASTERSTAR kandidáti ---

    df["RA"] = pd.to_numeric(df["RA"], errors="coerce")
    df["DE"] = pd.to_numeric(df["DE"], errors="coerce")
    df["EXPTIME"] = pd.to_numeric(df["EXPTIME"], errors="coerce")
    lim_act = math.isfinite(fwhm_limit) and fwhm_limit > 0.0
    reject_by_fwhm = (
        (df["FWHM"].isna() | (df["FWHM"] > float(fwhm_limit)))
        if lim_act
        else pd.Series(False, index=df.index)
    )
    reject_qc = reject_by_fwhm
    editor_df = pd.DataFrame(
        {
            "ID": df["ID"].astype(int),
            "Frame": df["frame_index"].astype(int),
            "Filename": [Path(str(p)).name for p in df["FILE_PATH"]],
            "EXPTIME (s)": df["EXPTIME"],
            "FWHM": df["FWHM"],
            "Roundness": df["ROUNDNESS_MEAN"],
            "REJECT": reject_qc.astype(bool),
            "RA (°)": df["RA"],
            "DE (°)": df["DE"],
            "Sky Level": df["SKY_LEVEL"],
            "Star Count": df["STAR_COUNT"],
            "Calib Flags": df["CALIB_FLAGS"],
            "IS_REJECTED": df["IS_REJECTED"].astype(bool),
        }
    )

    editor_key = f"vyvar_qdash_editor_{did}"
    _editor_row_ids = editor_df["ID"].astype(int).tolist()
    st.data_editor(
        editor_df,
        column_config={
            "ID": st.column_config.NumberColumn("ID", disabled=True, format="%d"),
            "Frame": st.column_config.NumberColumn(
                "Frame",
                disabled=True,
                format="%d",
                help="Rovnaké číslo ako os X v grafoch (1 … N).",
            ),
            "Filename": st.column_config.TextColumn("Filename", disabled=True),
            "EXPTIME (s)": st.column_config.NumberColumn("EXPTIME (s)", disabled=True, format="%.3f"),
            "FWHM": st.column_config.NumberColumn("FWHM", disabled=True, format="%.6f"),
            "Roundness": st.column_config.NumberColumn("Roundness", disabled=True, format="%.4f"),
            "REJECT": st.column_config.CheckboxColumn(
                "REJECT",
                disabled=True,
                help="Indikácia: mimo FWHM limitu (prahy > 0). Ukladá sa len IS_REJECTED.",
            ),
            "RA (°)": st.column_config.NumberColumn("RA (°)", disabled=True, format="%.6f"),
            "DE (°)": st.column_config.NumberColumn("DE (°)", disabled=True, format="%.6f"),
            "Sky Level": st.column_config.NumberColumn("Sky Level", disabled=True, format="%.4g"),
            "Star Count": st.column_config.NumberColumn("Star Count", disabled=True, format="%d"),
            "Calib Flags": st.column_config.TextColumn(
                "Calib Flags",
                disabled=True,
                help="D=Dark applied, F=Flat applied, P=Passthrough (nič).",
            ),
            "IS_REJECTED": st.column_config.CheckboxColumn(
                "IS_REJECTED",
                help="Zamietnuté snímky sa vylúčia z kalibrácie / preprocessu (podľa OBS_FILES).",
            ),
        },
        hide_index=True,
        num_rows="fixed",
        key=editor_key,
        on_change=_make_is_rejected_persist_handler(
            did, editor_key, db, editor_row_ids=_editor_row_ids
        ),
        use_container_width=True,
    )

    p_sel = _preview_path_from_plotly_state(st.session_state.get(key_fwhm)) or _preview_path_from_plotly_state(
        st.session_state.get(key_sky)
    )
    if p_sel and Path(p_sel).is_file():
        try:
            png_b = quicklook_preview_png_bytes(p_sel)
            st.image(png_b, caption=f"Náhľad: {Path(p_sel).name}", use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Náhľad zlyhal: {exc}")
        st.markdown(f"**Vybraná snímka:** `{Path(p_sel).name}`")
        if st.button(
            "Použiť ako MASTERSTAR",
            key="vyvar_set_masterstar_from_preview",
            type="primary",
            help="Uloží cestu tejto snímky ako MASTERSTAR kandidáta pre plate-solving.",
        ):
            st.session_state["vyvar_masterstar_candidate_paths"] = [p_sel]
            st.session_state["vyvar_ms_candidate_top1_path"] = p_sel
            try:
                _db2 = VyvarDatabase(db.db_path)
                try:
                    _db2.set_obs_draft_masterstar_path(int(did), str(p_sel))
                finally:
                    _db2.conn.close()
            except Exception:  # noqa: BLE001
                pass
            log_event(f"MASTERSTAR manuálne nastavený: {Path(p_sel).name}")
            st.success(f"Nastavené: `{Path(p_sel).name}` bude použitý ako MASTERSTAR.")

