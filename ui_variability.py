from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from variability_detector import compute_rms_variability, compute_vdi, load_field_flux_matrix

if TYPE_CHECKING:
    from config import AppConfig
    from pipeline import AstroPipeline


def _detect_obs_groups(draft_dir: Path) -> list[str]:
    root = Path(draft_dir) / "detrended_aligned" / "lights"
    if not root.is_dir():
        return []
    out: list[str] = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and any(d.glob("proc_*.csv")):
            out.append(d.name)
    return out


@st.cache_data(show_spinner=False)
def _cached_load_matrix(
    per_frame_dir_s: str,
    flux_col: str,
    min_frames_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    return load_field_flux_matrix(
        Path(per_frame_dir_s),
        flux_col=flux_col,
        min_frames_frac=min_frames_frac,
    )


def _read_comp_catalog_ids(platesolve_dir: Path) -> list[str]:
    p = Path(platesolve_dir) / "comparison_stars.csv"
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        return []
    for col in ("catalog_id", "name"):
        if col in df.columns:
            vals = df[col].dropna().astype(str).tolist()
            return vals
    return []


def _vizier_link(ra: float, dec: float) -> str:
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return ""
    return f"https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-c={ra:.6f}%20{dec:.6f}&-c.rs=2"


def _raw_lightcurve_from_frames(per_frame_dir: Path, catalog_id: str, flux_col: str) -> pd.DataFrame:
    frames = sorted(Path(per_frame_dir).glob("proc_*.csv"))
    rows: list[dict[str, Any]] = []
    for p in frames:
        try:
            df = pd.read_csv(p, usecols=["catalog_id", flux_col, "bjd_tdb_mid"], low_memory=False)
        except Exception:
            continue
        df["catalog_id"] = df["catalog_id"].astype(str)
        sub = df[df["catalog_id"].astype(str).str.contains(str(catalog_id).split(".")[0], regex=False)]
        if sub.empty:
            continue
        # Use first match
        r = sub.iloc[0]
        flux = float(pd.to_numeric(r.get(flux_col), errors="coerce"))
        bjd = float(pd.to_numeric(r.get("bjd_tdb_mid"), errors="coerce"))
        if not (np.isfinite(flux) and flux > 0 and np.isfinite(bjd)):
            continue
        mag_inst = -2.5 * float(np.log10(flux))
        rows.append({"bjd_tdb_mid": bjd, "mag_inst": mag_inst})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("bjd_tdb_mid").reset_index(drop=True)
    return out


def render_variability_dashboard(
    pipeline: "AstroPipeline",
    cfg: "AppConfig",
    *,
    draft_id: int | None = None,
    draft_dir_override: Path | None = None,
) -> None:
    st.header("🔍 Variability Detection")

    # Draft resolution consistent with other dashboards (Aperture Photometry).
    if draft_id is None and draft_dir_override is None:
        st.info("Najprv načítaj draft.")
        return
    if draft_dir_override is not None and Path(draft_dir_override).is_dir():
        draft_dir = Path(draft_dir_override).resolve()
    else:
        draft_dir = (Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft_id):06d}").resolve()

    with st.expander("Debug", expanded=False):
        st.write(f"draft_dir: {draft_dir}")
        try:
            st.write(f"pipeline.draft_dir: {getattr(pipeline, 'draft_dir', None)}")
        except Exception:
            pass

    obs_groups = _detect_obs_groups(draft_dir)
    if not obs_groups:
        st.warning("Nenájdené `proc_*.csv` v detrended_aligned/lights/.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        obs_group = st.selectbox("Setup:", options=obs_groups, key="var_obs_group")
    with col2:
        run = st.button("▶ Analyzovať pole", type="primary", key="var_run")

    flux_source = st.radio("Flux zdroj:", options=["dao_flux", "psf_flux"], horizontal=True, key="var_flux_source")
    method = st.radio("Metóda:", options=["RMS", "VDI", "Obe"], horizontal=True, key="var_method")
    sigma_thr = st.slider(
        "Sigma prah (RMS score):",
        min_value=2.0,
        max_value=4.0,
        value=3.0,
        step=0.1,
        key="var_sigma_thr",
        help="3.0 = konzervatívne (~12 kandidátov), 2.5 = balans, 2.0 = exploratívne",
    )
    min_frames_pct = st.slider("Min framov (%):", min_value=20, max_value=80, value=50, step=5, key="var_min_frames_pct")

    per_frame_dir = draft_dir / "detrended_aligned" / "lights" / str(obs_group)
    platesolve_dir = draft_dir / "platesolve" / str(obs_group)
    with st.expander("Debug paths", expanded=False):
        st.write(f"per_frame_dir: {per_frame_dir} (exists={per_frame_dir.exists()})")
        st.write(f"platesolve_dir: {platesolve_dir} (exists={platesolve_dir.exists()})")

    if run:
        try:
            with st.spinner("Načítavam flux maticu…"):
                fm, meta, _bjd = _cached_load_matrix(
                    str(per_frame_dir),
                    flux_source,
                    float(min_frames_pct) / 100.0,
                )
            comp_ids = _read_comp_catalog_ids(platesolve_dir)
            results: dict[str, Any] = {
                "flux_matrix": fm,
                "metadata": meta,
                "comp_ids": comp_ids,
                "obs_group": str(obs_group),
                "flux_col": flux_source,
            }

            if method in ("RMS", "Obe"):
                with st.spinner("Počítam RMS variabilitu…"):
                    rms_df = compute_rms_variability(
                        fm,
                        meta,
                        comp_ids,
                        sigma_threshold=float(sigma_thr),
                        vsx_targets_csv=(platesolve_dir / "variable_targets.csv"),
                    )
                results["rms_df"] = rms_df
            if method in ("VDI", "Obe"):
                with st.spinner("Počítam VDI…"):
                    vdi_df = compute_vdi(fm, meta, min_frames=30)
                results["vdi_df"] = vdi_df

            st.session_state["var_results"] = results
            st.success("Hotovo.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Chyba analýzy: {exc}")
            logging.exception("Variability analysis failed")

    res = st.session_state.get("var_results") or {}
    rms_df: pd.DataFrame = res.get("rms_df") if isinstance(res.get("rms_df"), pd.DataFrame) else pd.DataFrame()
    vdi_df: pd.DataFrame = res.get("vdi_df") if isinstance(res.get("vdi_df"), pd.DataFrame) else pd.DataFrame()

    if not rms_df.empty:
        # Merge RMS + VDI (if available)
        results_df = rms_df.copy()
        if not vdi_df.empty:
            results_df = results_df.merge(
                vdi_df[["catalog_id", "vdi_score", "vdi_z_score", "is_variable_candidate"]],
                on="catalog_id",
                how="left",
                suffixes=("_rms", "_vdi"),
            )
            # Rename the VDI candidate flag to avoid confusion
            results_df = results_df.rename(columns={"is_variable_candidate": "is_variable_candidate_vdi"})
        else:
            results_df["vdi_score"] = np.nan
            results_df["vdi_z_score"] = np.nan
            results_df["is_variable_candidate_vdi"] = False

        # RMS candidate flag rename
        if "is_variable_candidate" in results_df.columns and "is_variable_candidate_rms" not in results_df.columns:
            results_df = results_df.rename(columns={"is_variable_candidate": "is_variable_candidate_rms"})

        results_df["is_variable_candidate_rms"] = results_df["is_variable_candidate_rms"].fillna(False).astype(bool)
        results_df["is_variable_candidate_vdi"] = results_df["is_variable_candidate_vdi"].fillna(False).astype(bool)
        results_df["is_candidate_combined"] = (
            results_df["is_variable_candidate_rms"] | results_df["is_variable_candidate_vdi"]
        )
        results_df["detection_method"] = "—"
        results_df.loc[results_df["is_variable_candidate_rms"], "detection_method"] = "RMS"
        results_df.loc[results_df["is_variable_candidate_vdi"], "detection_method"] = "VDI"
        results_df.loc[
            results_df["is_variable_candidate_rms"] & results_df["is_variable_candidate_vdi"],
            "detection_method",
        ] = "RMS+VDI"

        st.subheader("Hockey stick (RMS)")
        work = results_df.copy()
        work["mag"] = pd.to_numeric(work["mag"], errors="coerce")
        work["rms_pct"] = pd.to_numeric(work["rms_pct"], errors="coerce")
        work["expected_rms_pct"] = pd.to_numeric(work["expected_rms_pct"], errors="coerce")
        work["variability_score"] = pd.to_numeric(work["variability_score"], errors="coerce")
        work["vsx_known_variable"] = work["vsx_known_variable"].fillna(False).astype(bool)
        work["gaia_dr3_variable_catalog"] = work["gaia_dr3_variable_catalog"].fillna(False).astype(bool)

        comp_mask = (~work["vsx_known_variable"]) & (work["is_variable_candidate_rms"] == False)
        cand_mask = (work["is_candidate_combined"] == True) & (~work["vsx_known_variable"])
        vsx_mask = work["vsx_known_variable"]
        gaia_mask = work["gaia_dr3_variable_catalog"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=work.loc[comp_mask, "mag"],
                y=work.loc[comp_mask, "rms_pct"],
                mode="markers",
                name="Stabilné (COMP)",
                marker=dict(color="#2ecc71", size=6, opacity=0.5),
                hovertemplate="cid=%{customdata}<br>mag=%{x:.3f}<br>rms=%{y:.3f}%<extra></extra>",
                customdata=work.loc[comp_mask, "catalog_id"],
            )
        )
        # expected curve (sorted by mag)
        curve = work[["mag", "expected_rms_pct"]].dropna().sort_values("mag")
        if not curve.empty:
            fig.add_trace(
                go.Scatter(
                    x=curve["mag"],
                    y=curve["expected_rms_pct"],
                    mode="lines",
                    name="Expected noise",
                    line=dict(color="#888888", width=2),
                    hoverinfo="skip",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=work.loc[cand_mask, "mag"],
                y=work.loc[cand_mask, "rms_pct"],
                mode="markers",
                name="Kandidáti",
                marker=dict(color="#e74c3c", size=9, opacity=0.9),
                hovertemplate="cid=%{customdata}<br>mag=%{x:.3f}<br>rms=%{y:.3f}%<br>score=%{text:.2f}<extra></extra>",
                customdata=work.loc[cand_mask, "catalog_id"],
                text=work.loc[cand_mask, "variability_score"],
            )
        )
        vsx_data = work.loc[vsx_mask].copy()
        if not vsx_data.empty:
            if "vsx_name" not in vsx_data.columns:
                vsx_data["vsx_name"] = np.nan
            if "vsx_type" not in vsx_data.columns:
                vsx_data["vsx_type"] = ""
            vsx_data["vsx_name_display"] = vsx_data["vsx_name"].fillna(
                vsx_data["catalog_id"].astype(str)
            )
            fig.add_trace(
                go.Scatter(
                    x=vsx_data["mag"],
                    y=vsx_data["rms_pct"],
                    mode="markers",
                    name="Known VSX",
                    marker=dict(color="#f39c12", symbol="x", size=10, opacity=0.9),
                    customdata=vsx_data[["vsx_name_display", "vsx_type", "catalog_id"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Typ: %{customdata[1]}<br>"
                        "mag: %{x:.2f}<br>"
                        "rms: %{y:.2f}%<br>"
                        "ID: %{customdata[2]}<br>"
                        "<extra>Known VSX</extra>"
                    ),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=work.loc[gaia_mask, "mag"],
                y=work.loc[gaia_mask, "rms_pct"],
                mode="markers",
                name="Gaia variable",
                marker=dict(color="#3498db", symbol="square", size=9, opacity=0.9),
                customdata=work.loc[gaia_mask, "catalog_id"],
            )
        )

        fig.update_yaxes(type="log", title="rms_pct (%)")
        fig.update_xaxes(title="mag (Gaia G)")
        fig.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Výsledky")
        n_all = int(len(work))
        n_rms_candidates = int((work["is_variable_candidate_rms"] & (~work["vsx_known_variable"])).sum())
        n_vdi_candidates = int((work["is_variable_candidate_vdi"] & (~work["vsx_known_variable"])).sum())
        n_combined = int(cand_mask.sum())
        n_vsx = int(vsx_mask.sum())
        n_gaia = int(gaia_mask.sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Analyzovaných hviezd", f"{n_all}")
        m2.metric("RMS kandidáti", f"{n_rms_candidates}")
        m3.metric("VDI kandidáti", f"{n_vdi_candidates}")
        m4.metric("Kombinovaných", f"{n_combined}")
        st.caption(f"Known VSX: {n_vsx} | Gaia variable: {n_gaia}")

        cand = work.loc[cand_mask].copy()
        cand["Vizier"] = [
            _vizier_link(float(pd.to_numeric(r.get("ra_deg"), errors="coerce")), float(pd.to_numeric(r.get("dec_deg"), errors="coerce")))
            for _, r in cand.iterrows()
        ]
        show_cols = [
            "catalog_id",
            "mag",
            "bp_rp",
            "rms_pct",
            "smoothness_ratio",
            "vdi_score",
            "vdi_z_score",
            "detection_method",
            "variability_score",
            "vsx_match",
            "vsx_known_variable",
            "gaia_dr3_variable_catalog",
            "zone",
            "Vizier",
        ]
        candidates_df = cand[show_cols].copy()
        # Formatting
        if "smoothness_ratio" in candidates_df.columns:
            candidates_df["smoothness_ratio"] = pd.to_numeric(candidates_df["smoothness_ratio"], errors="coerce").round(2)
        if "vdi_score" in candidates_df.columns:
            candidates_df["vdi_score"] = pd.to_numeric(candidates_df["vdi_score"], errors="coerce").round(3)
        if "vdi_z_score" in candidates_df.columns:
            candidates_df["vdi_z_score"] = pd.to_numeric(candidates_df["vdi_z_score"], errors="coerce").round(2)

        st.markdown("**Vyber kandidátov na pridanie do `variable_targets.csv`:**")
        selected_ids: list[str] = []
        for _, row in candidates_df.iterrows():
            cid = str(row.get("catalog_id", ""))
            label = (
                f"{row.get('vsx_name', cid) or cid} "
                f"mag={float(pd.to_numeric(row.get('mag'), errors='coerce')):.2f} "
                f"rms={float(pd.to_numeric(row.get('rms_pct'), errors='coerce')):.1f}%"
            )
            if st.checkbox(label, key=f"chk_{cid}"):
                selected_ids.append(cid)

        st.dataframe(candidates_df, use_container_width=True, hide_index=True)

        colA, colB = st.columns(2)
        with colA:
            if st.button("📥 Export kandidátov CSV", key="var_export"):
                out_csv = draft_dir / "platesolve" / str(obs_group) / "variability_candidates.csv"
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                candidates_df.to_csv(out_csv, index=False)
                st.success(f"Uložené: {out_csv}")
        with colB:
            if st.button("➕ Pridať vybrané do variable_targets.csv", key="var_add_to_var2"):
                if not selected_ids:
                    st.warning("Nie sú vybrané žiadne hviezdy.")
                else:
                    vt_path = platesolve_dir / "variable_targets.csv"
                    if vt_path.exists():
                        vt_df = pd.read_csv(vt_path, low_memory=False)
                    else:
                        vt_df = pd.DataFrame()

                    # normalize existing ids
                    if "catalog_id" in vt_df.columns:
                        existing_ids = set(
                            pd.to_numeric(vt_df["catalog_id"], errors="coerce").dropna().round().astype("int64").tolist()
                        )
                    else:
                        existing_ids = set()

                    n_added = 0
                    new_rows: list[dict[str, Any]] = []
                    for cid in selected_ids:
                        try:
                            icid = int(float(cid))
                        except Exception:
                            continue
                        if icid in existing_ids:
                            continue
                        row = results_df[results_df["catalog_id"].astype(str) == str(cid)].iloc[0]
                        new_rows.append(
                            {
                                "catalog_id": cid,
                                "name": str(cid),
                                "vsx_name": row.get("vsx_name", ""),
                                "vsx_type": row.get("vsx_type", "CAND"),
                                "ra_deg": row.get("ra_deg", ""),
                                "dec_deg": row.get("dec_deg", ""),
                                "x": row.get("x", ""),
                                "y": row.get("y", ""),
                                "mag": row.get("mag", ""),
                                "zone": row.get("zone", "linear"),
                                "priority": 2,
                                "notes": (
                                    f"VarDetect: RMS={float(pd.to_numeric(row.get('rms_pct'), errors='coerce')):.1f}% "
                                    f"smooth={float(pd.to_numeric(row.get('smoothness_ratio'), errors='coerce')):.2f} "
                                    f"method={row.get('detection_method','—')}"
                                ),
                                "gaia_match_source": "variability_detection",
                            }
                        )
                        n_added += 1

                    if new_rows:
                        new_df = pd.DataFrame(new_rows)
                        vt_df = pd.concat([vt_df, new_df], ignore_index=True)
                        vt_df.to_csv(vt_path, index=False)
                        st.success(
                            f"✅ Pridaných {n_added} hviezd do variable_targets.csv\n"
                            f"Spusti RUN Aperture Photometry pre plnú kalibráciu."
                        )
                    else:
                        st.info("Všetky vybrané hviezdy už sú v variable_targets.csv")

        # ---- Light curve section (cached) ----
        st.subheader("📈 Light curve kandidáta")

        candidate_options = {
            f"{row.get('vsx_name', str(row.catalog_id)) or str(row.catalog_id)} "
            f"(mag={float(pd.to_numeric(row.mag, errors='coerce')):.2f}, "
            f"rms={float(pd.to_numeric(row.rms_pct, errors='coerce')):.1f}%, "
            f"{row.get('detection_method','—')})": str(row.catalog_id)
            for _, row in candidates_df.iterrows()
        }
        if candidate_options:
            selected_label = st.selectbox(
                "Vyber kandidáta:",
                options=list(candidate_options.keys()),
                key="var_lc_select2",
            )
            selected_cid = candidate_options[selected_label]

            @st.cache_data(ttl=300, show_spinner=False)
            def load_candidate_lc(per_frame_dir_s: str, catalog_id: str, flux_col_in: str = "dao_flux") -> pd.DataFrame:
                records: list[dict[str, Any]] = []
                for csv in sorted(Path(per_frame_dir_s).glob("proc_*.csv")):
                    try:
                        df = pd.read_csv(csv, usecols=["catalog_id", flux_col_in, "bjd_tdb_mid", "airmass"], low_memory=False)
                    except Exception:
                        continue
                    # robust id match (string contains to survive scientific notation)
                    s = str(catalog_id)
                    df["_cid"] = pd.to_numeric(df["catalog_id"], errors="coerce").round().astype("Int64")
                    try:
                        cid_int = int(float(s))
                    except Exception:
                        continue
                    row = df[df["_cid"] == cid_int]
                    if row.empty:
                        continue
                    r0 = row.iloc[0]
                    flux = float(pd.to_numeric(r0.get(flux_col_in), errors="coerce"))
                    if not (np.isfinite(flux) and flux > 0):
                        continue
                    records.append(
                        {
                            "bjd": float(pd.to_numeric(r0.get("bjd_tdb_mid"), errors="coerce")),
                            "mag_inst": -2.5 * float(np.log10(flux)),
                            "airmass": float(pd.to_numeric(r0.get("airmass"), errors="coerce")),
                        }
                    )
                if not records:
                    return pd.DataFrame()
                return pd.DataFrame(records).sort_values("bjd").reset_index(drop=True)

            lc_df = load_candidate_lc(str(per_frame_dir), str(selected_cid), flux_source)
            if len(lc_df) > 0:
                fig2 = go.Figure()
                bjd_rel = lc_df["bjd"] - lc_df["bjd"].iloc[0]
                fig2.add_trace(
                    go.Scatter(
                        x=bjd_rel,
                        y=lc_df["mag_inst"],
                        mode="markers",
                        marker=dict(size=5, color="steelblue", opacity=0.8),
                        name="mag_inst",
                        hovertemplate="BJD+%{x:.4f}<br>mag=%{y:.3f}<extra></extra>",
                    )
                )
                fig2.update_layout(
                    title=f"Raw light curve — {selected_label}",
                    xaxis_title="BJD - BJD0",
                    yaxis_title="mag_inst (nekalibrovaná)",
                    yaxis_autorange="reversed",
                    height=350,
                    margin=dict(l=50, r=20, t=50, b=40),
                    hovermode="closest",
                )
                st.plotly_chart(fig2, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("N framov", f"{len(lc_df)}")
                c2.metric("RMS", f"{lc_df.mag_inst.std():.3f} mag")
                c3.metric("Amplitúda", f"{(lc_df.mag_inst.max() - lc_df.mag_inst.min()):.3f} mag")
                c4.metric("Median mag", f"{lc_df.mag_inst.median():.3f}")
            else:
                st.warning("Light curve nie je k dispozícii pre túto hviezdu.")

