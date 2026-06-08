"""Light curves for suspected variables — inštrumentálna mag z per-frame CSV."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import streamlit as st

from gaia_catalog_id import normalize_gaia_source_id
from jd_axis_format import jd_axis_title, jd_series_relative
from photometry_core import _flux_to_mag, _normalize_gaia_id
from ui_aperture_photometry import _find_phase2a_paths
from ui_select_stars import _sanitize_suspected_variables_df

if TYPE_CHECKING:
    from config import AppConfig
    from pipeline import AstroPipeline

# NOTE: This module is not currently wired into app.py and is inactive.
# Functions are preserved for potential future use.
# Last audit: 2026-05-18


def _collect_instrumental_lc(
    catalog_id: str,
    per_frame_dir: Path,
    *,
    csv_cache: Any | None = None,
) -> pd.DataFrame:
    """Z per-frame ``proc_*.csv`` zloží časovú radu ``bjd_tdb_mid`` + ``mag_inst`` pre jednu hviezdu."""
    target = _normalize_gaia_id(catalog_id)
    if not target:
        return pd.DataFrame()

    proc_paths = sorted(per_frame_dir.glob("proc_*.csv"))
    logging.warning(
        "[PERF-7] _collect_instrumental_lc: scanning all proc_*.csv "
        "per star — consider passing ProcFrameStore via csv_cache. "
        "N_frames=%d",
        len(proc_paths),
    )

    rows: list[dict[str, float | str]] = []
    for csv_path in proc_paths:
        try:
            _key = str(csv_path)
            if csv_cache is not None and _key in csv_cache:
                df = csv_cache.get(_key)
            else:
                hdr = pd.read_csv(csv_path, nrows=0)
                if "dao_flux" not in hdr.columns:
                    continue
                use = (
                    ["dao_flux", "bjd_tdb_mid", "catalog_id"]
                    if "catalog_id" in hdr.columns
                    else ["dao_flux", "bjd_tdb_mid", "name"]
                )
                if "name" in hdr.columns and "name" not in use:
                    use.append("name")
                use = [c for c in use if c in hdr.columns]
                _dtypes: dict[str, type] = {}
                if "catalog_id" in use:
                    _dtypes["catalog_id"] = str
                if "name" in use:
                    _dtypes["name"] = str
                df = pd.read_csv(csv_path, usecols=use, low_memory=False, dtype=_dtypes or None)

            if df is None or df.empty or "dao_flux" not in df.columns:
                continue

            idc = "catalog_id" if "catalog_id" in df.columns else "name"
            fluxes: list[float] = []
            bjd0: float | None = None
            for _, row in df.iterrows():
                cid = _normalize_gaia_id(row.get(idc)) if idc in row.index else ""
                name_hit = ""
                if "name" in df.columns:
                    nk = normalize_gaia_source_id(row.get("name"))
                    if nk and re.fullmatch(r"\d{12,22}", nk):
                        name_hit = nk
                if cid != target and name_hit != target:
                    continue
                fx = float(pd.to_numeric(row.get("dao_flux"), errors="coerce"))
                if math.isfinite(fx) and fx > 0:
                    fluxes.append(fx)
                if bjd0 is None:
                    bj = float(pd.to_numeric(row.get("bjd_tdb_mid"), errors="coerce"))
                    if math.isfinite(bj):
                        bjd0 = bj
            if not fluxes or bjd0 is None:
                continue
            med_flux = float(np.median(np.asarray(fluxes, dtype=np.float64)))
            if not math.isfinite(med_flux) or med_flux <= 0:
                continue
            rows.append(
                {
                    "bjd_tdb_mid": float(bjd0),
                    "mag_inst": float(_flux_to_mag(med_flux)),
                    "source": csv_path.name,
                }
            )
        except Exception:  # noqa: BLE001
            logging.debug("Suspected LC: preskočený súbor %s", csv_path, exc_info=False)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("bjd_tdb_mid").reset_index(drop=True)
    return out


def render_suspected_lightcurves(
    cfg: "AppConfig",
    draft_id: int | None,
    pipeline: "AstroPipeline",
    *,
    draft_dir_override: Path | None = None,
) -> None:
    _ = pipeline
    st.header("LightCurves — Suspected Stars")
    st.caption(
        "Quick preview of instrumental magnitude (``dao_flux`` → ``mag_inst``) for candidates "
        "from ``suspected_variables.csv``. Not a calibrated curve like Phase 2A."
    )

    if draft_id is None and draft_dir_override is None:
        st.info("No active draft.")
        return

    all_setups = _find_phase2a_paths(cfg, draft_id, draft_dir_override=draft_dir_override)
    if not all_setups:
        st.warning("No platesolve setups found.")
        return

    setup_names = sorted(all_setups.keys())
    chosen = st.selectbox(
        "Filter / skupina:",
        options=setup_names,
        key="suspected_lc_setup",
    )
    paths = all_setups.get(str(chosen)) or {}
    out_dir = paths.get("output_dir")
    pf_dir = paths.get("per_frame_csv_dir")
    if out_dir is None or not Path(out_dir).is_dir():
        st.error("Missing photometry output directory.")
        return

    sus_path = Path(out_dir) / "suspected_variables.csv"
    if not sus_path.exists():
        st.info("``suspected_variables.csv`` is not ready for this setup yet. Run Phase 0+1.")
        return

    try:
        suspected_df = _sanitize_suspected_variables_df(
            pd.read_csv(
                sus_path,
                low_memory=False,
                dtype={"catalog_id": str, "name": str},  # Gaia ID musí byť str — float64 stráca cifry
            )
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load suspected CSV: {exc}")
        return

    if suspected_df.empty:
        st.success("No suspected candidates (after table cleanup).")
        return

    if pf_dir is None or not Path(pf_dir).is_dir():
        st.error("Missing per-frame CSV directory (``detrended_aligned``).")
        return

    id_col = "catalog_id" if "catalog_id" in suspected_df.columns else suspected_df.columns[0]
    labels = suspected_df[id_col].astype(str).tolist()
    pick = st.selectbox("Suspected star:", labels, key="suspected_lc_star")
    row0 = suspected_df[suspected_df[id_col].astype(str) == str(pick)].iloc[0]
    cid = str(row0.get("catalog_id", pick))

    lc_df = _collect_instrumental_lc(cid, Path(pf_dir))
    if lc_df.empty:
        st.warning("No points found in per-frame CSV for this star (check ``catalog_id`` / Gaia ID).")
        return

    try:
        import plotly.graph_objects as go  # type: ignore

        t_raw = pd.to_numeric(lc_df["bjd_tdb_mid"], errors="coerce")
        t_rel, t_off = jd_series_relative(t_raw)
        t_cd = t_raw.to_numpy(dtype=float)
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=t_rel,
                    y=lc_df["mag_inst"],
                    mode="markers",
                    name="mag_inst",
                    marker=dict(size=5),
                    customdata=t_cd,
                    hovertemplate="BJD=%{customdata:.6f}<br>mag_inst=%{y:.4f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=f"Suspected — {cid} ({chosen})",
            xaxis_title=jd_axis_title("BJD (TDB mid)", t_off),
            yaxis_title="mag_inst (instrumental)",
            yaxis_autorange="reversed",
            height=480,
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Plotly: {exc}")
        t_raw = pd.to_numeric(lc_df["bjd_tdb_mid"], errors="coerce")
        t_rel, t_off = jd_series_relative(t_raw)
        if t_off is not None:
            st.line_chart(lc_df.assign(_xrel=t_rel).set_index("_xrel")["mag_inst"])
        else:
            st.line_chart(lc_df.set_index("bjd_tdb_mid")["mag_inst"])

    with st.expander("Raw points (CSV)", expanded=False):
        st.dataframe(lc_df, width="stretch", hide_index=True)
