"""Light curves for suspected variables — inštrumentálna mag z per-frame CSV."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

from gaia_catalog_id import normalize_gaia_source_id
from photometry_core import _flux_to_mag, _normalize_gaia_id
from ui_aperture_photometry import _find_phase2a_paths
from ui_select_stars import _sanitize_suspected_variables_df

if TYPE_CHECKING:
    from config import AppConfig
    from pipeline import AstroPipeline


def _collect_instrumental_lc(per_frame_dir: Path, catalog_id: str) -> pd.DataFrame:
    """Z per-frame ``proc_*.csv`` zloží časovú radu ``bjd_tdb_mid`` + ``mag_inst`` pre jednu hviezdu."""
    target = _normalize_gaia_id(catalog_id)
    if not target:
        return pd.DataFrame()

    rows: list[dict[str, float | str]] = []
    for csv_path in sorted(per_frame_dir.glob("proc_*.csv")):
        try:
            hdr = pd.read_csv(csv_path, nrows=0)
            if "dao_flux" not in hdr.columns:
                continue
            use = ["dao_flux", "bjd_tdb_mid", "catalog_id"] if "catalog_id" in hdr.columns else ["dao_flux", "bjd_tdb_mid", "name"]
            if "name" in hdr.columns and "name" not in use:
                use.append("name")
            use = [c for c in use if c in hdr.columns]
            df = pd.read_csv(csv_path, usecols=use, low_memory=False)
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
        "Rýchly náhľad inštrumentálnej magnitúdy (``dao_flux`` → ``mag_inst``) pre kandidátov "
        "zo ``suspected_variables.csv``. Nie je to kalibrovaná krivka ako v Fáze 2A."
    )

    if draft_id is None and draft_dir_override is None:
        st.info("Žiadny aktívny draft.")
        return

    all_setups = _find_phase2a_paths(cfg, draft_id, draft_dir_override=draft_dir_override)
    if not all_setups:
        st.warning("Nenájdené platesolve setupy.")
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
        st.error("Chýba photometry výstupný adresár.")
        return

    sus_path = Path(out_dir) / "suspected_variables.csv"
    if not sus_path.exists():
        st.info("Pre tento setup ešte nie je ``suspected_variables.csv``. Spusti Fázu 0+1.")
        return

    try:
        suspected_df = _sanitize_suspected_variables_df(pd.read_csv(sus_path, low_memory=False))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Nepodarilo sa načítať suspected CSV: {exc}")
        return

    if suspected_df.empty:
        st.success("Žiadni suspected kandidáti (po vyčistení tabuľky).")
        return

    if pf_dir is None or not Path(pf_dir).is_dir():
        st.error("Chýba per-frame CSV adresár (``detrended_aligned``).")
        return

    id_col = "catalog_id" if "catalog_id" in suspected_df.columns else suspected_df.columns[0]
    labels = suspected_df[id_col].astype(str).tolist()
    pick = st.selectbox("Suspected hviezda:", labels, key="suspected_lc_star")
    row0 = suspected_df[suspected_df[id_col].astype(str) == str(pick)].iloc[0]
    cid = str(row0.get("catalog_id", pick))

    lc_df = _collect_instrumental_lc(Path(pf_dir), cid)
    if lc_df.empty:
        st.warning("Pre túto hviezdu sa nenašli žiadne body v per-frame CSV (skontroluj ``catalog_id`` / Gaia ID).")
        return

    try:
        import plotly.graph_objects as go  # type: ignore

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=lc_df["bjd_tdb_mid"],
                    y=lc_df["mag_inst"],
                    mode="markers",
                    name="mag_inst",
                    marker=dict(size=5),
                )
            ]
        )
        fig.update_layout(
            title=f"Suspected — {cid} ({chosen})",
            xaxis_title="BJD (TDB mid)",
            yaxis_title="mag_inst (inštrumentálna)",
            yaxis_autorange="reversed",
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Plotly: {exc}")
        st.line_chart(lc_df.set_index("bjd_tdb_mid")["mag_inst"])

    with st.expander("Surové body (CSV)", expanded=False):
        st.dataframe(lc_df, use_container_width=True, hide_index=True)
