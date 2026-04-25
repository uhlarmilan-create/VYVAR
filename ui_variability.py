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
    cfg_dict: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    return load_field_flux_matrix(
        Path(per_frame_dir_s),
        flux_col=flux_col,
        min_frames_frac=min_frames_frac,
        config=cfg_dict,
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


def _edge_ok_from_masterstar(
    masterstar_fits: Path,
    stars_df: pd.DataFrame,
    cfg_dict: dict[str, Any],
) -> pd.Series:
    """
    Per-star edge safety (annulus-aware, best-effort).

    Uses MASTERSTAR dimensions and margin ~= outer annulus radius in px.
    If FITS/header missing, returns True for all rows.
    """
    if stars_df is None or stars_df.empty:
        return pd.Series(dtype=bool)
    if not masterstar_fits.exists():
        return pd.Series(True, index=stars_df.index)

    try:
        from astropy.io import fits as astrofits
    except Exception:
        return pd.Series(True, index=stars_df.index)

    nx = ny = None
    fwhm_px = float("nan")
    try:
        with astrofits.open(masterstar_fits, memmap=False) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data
        try:
            fwhm_px = float(hdr.get("VY_FWHM", float("nan")))
        except Exception:
            fwhm_px = float("nan")
        try:
            if data is not None and hasattr(data, "shape") and len(data.shape) >= 2:
                ny, nx = int(data.shape[-2]), int(data.shape[-1])
        except Exception:
            nx = ny = None
    except Exception:
        nx = ny = None

    # Base margin (same spirit as phase01 interior margin)
    try:
        base_margin = float(cfg_dict.get("phase01_chip_interior_margin_px", 100))
    except Exception:
        base_margin = 100.0

    # Annulus-aware margin: outer annulus radius in px + small guard
    try:
        ann_outer_fwhm = float(cfg_dict.get("annulus_outer_fwhm", 9.0))
    except Exception:
        ann_outer_fwhm = 9.0
    ann_margin = float(ann_outer_fwhm) * float(fwhm_px) + 5.0 if np.isfinite(fwhm_px) else float("nan")

    margin = float(base_margin)
    if np.isfinite(ann_margin):
        margin = max(float(margin), float(ann_margin))

    x = pd.to_numeric(stars_df.get("x"), errors="coerce")
    y = pd.to_numeric(stars_df.get("y"), errors="coerce")
    ok = np.isfinite(x) & np.isfinite(y)
    if nx is not None and ny is not None and nx > 0 and ny > 0 and np.isfinite(margin) and margin >= 0:
        ok = ok & (x >= margin) & (x <= float(nx) - margin) & (y >= margin) & (y <= float(ny) - margin)

    return ok.fillna(False).astype(bool)


@st.cache_data(ttl=600, show_spinner=False)
def _render_field_image_with_candidate(
    masterstar_fits_path_s: str,
    *,
    x: float,
    y: float,
    label: str,
    percentile_lo: float = 5.0,
    percentile_hi: float = 99.5,
) -> bytes | None:
    """Render MASTERSTAR FITS as PNG and mark candidate at (x,y)."""
    try:
        from astropy.io import fits as astrofits
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
    except Exception:
        return None

    p = Path(masterstar_fits_path_s)
    if not p.exists():
        return None

    try:
        with astrofits.open(p, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:
        return None

    if data.size == 0:
        return None

    ok = np.isfinite(data)
    if not ok.any():
        return None
    try:
        vmin = float(np.percentile(data[ok], float(percentile_lo)))
        vmax = float(np.percentile(data[ok], float(percentile_hi)))
    except Exception:
        vmin, vmax = float("nan"), float("nan")

    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=140)
    ax.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=vmin if np.isfinite(vmin) else None,
        vmax=vmax if np.isfinite(vmax) else None,
        aspect="equal",
    )
    ax.scatter([float(x)], [float(y)], s=140, facecolors="none", edgecolors="#ff3333", linewidths=2.0)
    ax.scatter([float(x)], [float(y)], s=18, c="#ff3333", alpha=0.95)
    ax.text(float(x) + 18, float(y), str(label)[:24], color="#ff3333", fontsize=9, va="center")
    ax.set_title("Hviezdne pole (MASTERSTAR) — vyznačený kandidát", fontsize=11)
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@st.cache_data(ttl=600, show_spinner=False)
def _render_field_image_with_candidates(
    masterstar_fits_path_s: str,
    cand_xy_label: list[tuple[float, float, str]],
    *,
    selected_xy: tuple[float, float] | None = None,
    selected_label: str = "",
    percentile_lo: float = 5.0,
    percentile_hi: float = 99.5,
) -> bytes | None:
    """Render MASTERSTAR FITS as PNG and mark ONLY candidate stars."""
    try:
        from astropy.io import fits as astrofits
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
    except Exception:
        return None

    p = Path(masterstar_fits_path_s)
    if not p.exists():
        return None

    try:
        with astrofits.open(p, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
    except Exception:
        return None

    if data.size == 0:
        return None

    ok = np.isfinite(data)
    if not ok.any():
        return None
    try:
        vmin = float(np.percentile(data[ok], float(percentile_lo)))
        vmax = float(np.percentile(data[ok], float(percentile_hi)))
    except Exception:
        vmin, vmax = float("nan"), float("nan")

    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=140)
    ax.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=vmin if np.isfinite(vmin) else None,
        vmax=vmax if np.isfinite(vmax) else None,
        aspect="equal",
    )

    # Candidates: red circles (no comps/targets)
    if cand_xy_label:
        xs = [float(t[0]) for t in cand_xy_label]
        ys = [float(t[1]) for t in cand_xy_label]
        ax.scatter(xs, ys, s=110, facecolors="none", edgecolors="#ff3333", linewidths=1.8, alpha=0.95)

        # Add a few labels (avoid clutter on dense fields)
        for (cx, cy, lab) in cand_xy_label[:25]:
            try:
                ax.text(float(cx) + 16, float(cy), str(lab)[:18], color="#ff3333", fontsize=7, va="center")
            except Exception:
                continue

    # Selected candidate highlight (yellow)
    if selected_xy is not None:
        try:
            sx, sy = float(selected_xy[0]), float(selected_xy[1])
            if np.isfinite(sx) and np.isfinite(sy):
                ax.scatter([sx], [sy], s=190, facecolors="none", edgecolors="#ffd54a", linewidths=2.6)
                ax.scatter([sx], [sy], s=26, c="#ffd54a", alpha=0.95)
                if selected_label:
                    ax.text(sx + 16, sy, str(selected_label)[:22], color="#ffd54a", fontsize=9, va="center")
        except Exception:
            pass

    ax.set_title("Hviezdne pole (MASTERSTAR) — vyznačení iba kandidáti", fontsize=11)
    ax.axis("off")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


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
    cfg_dict = cfg.to_dict()
    sigma_thr = st.slider(
        "Sigma prah (RMS score):",
        min_value=2.0,
        max_value=4.0,
        value=float(cfg_dict.get("variability_sigma_threshold", 3.0)),
        step=0.1,
        key="var_sigma_thr",
        help="3.0 = konzervatívne (~12 kandidátov), 2.5 = balans, 2.0 = exploratívne",
    )
    mag_limit = st.slider(
        "Mag limit (RMS kandidáti):",
        min_value=10.0,
        max_value=20.0,
        value=float(cfg_dict.get("variability_mag_limit", 14.5)),
        step=0.1,
        key="var_mag_limit",
        help="Kandidáti fainter než tento limit sa z RMS kandidátov vyradia.",
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
                cfg_run = dict(cfg_dict)
                cfg_run["variability_min_frames_frac"] = float(min_frames_pct) / 100.0
                fm, meta, _bjd = _cached_load_matrix(
                    str(per_frame_dir),
                    flux_source,
                    float(min_frames_pct) / 100.0,
                    cfg_run,
                )
            comp_ids = _read_comp_catalog_ids(platesolve_dir)
            comp_rms_map: dict[str, float] = {}
            comp_csv = platesolve_dir / "photometry" / "comparison_stars_per_target.csv"
            if comp_csv.exists():
                try:
                    comp_df = pd.read_csv(comp_csv, low_memory=False)
                    if "catalog_id" in comp_df.columns and "comp_rms" in comp_df.columns:
                        for _, row in comp_df.iterrows():
                            cid = str(row.get("catalog_id", "")).strip()
                            rms = float(pd.to_numeric(row.get("comp_rms"), errors="coerce"))
                            if cid and np.isfinite(rms) and rms > 1e-4:
                                if cid not in comp_rms_map or rms < comp_rms_map[cid]:
                                    comp_rms_map[cid] = float(rms)
                except Exception:
                    comp_rms_map = {}
            results: dict[str, Any] = {
                "flux_matrix": fm,
                "metadata": meta,
                "comp_ids": comp_ids,
                "obs_group": str(obs_group),
                "flux_col": flux_source,
                "comp_rms_map": comp_rms_map,
            }

            if method in ("RMS", "Obe"):
                with st.spinner("Počítam RMS variabilitu…"):
                    cfg_run = dict(cfg_dict)
                    cfg_run["variability_sigma_threshold"] = float(sigma_thr)
                    cfg_run["variability_mag_limit"] = float(mag_limit)
                    cfg_run["variability_min_frames_frac"] = float(min_frames_pct) / 100.0
                    rms_df = compute_rms_variability(
                        fm,
                        meta,
                        comp_ids,
                        sigma_threshold=float(sigma_thr),
                        vsx_targets_csv=(platesolve_dir / "variable_targets.csv"),
                        config=cfg_run,
                        comp_rms_map=comp_rms_map,
                    )
                results["rms_df"] = rms_df
            if method in ("VDI", "Obe"):
                with st.spinner("Počítam VDI…"):
                    cfg_run = dict(cfg_dict)
                    cfg_run["variability_min_frames"] = int(cfg_run.get("variability_min_frames", 30))
                    vdi_df = compute_vdi(fm, meta, min_frames=30, config=cfg_run)
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

        # Per-star edge filter (annulus-aware) — avoid false candidates near chip border
        masterstar_fits = platesolve_dir / "MASTERSTAR.fits"
        edge_ok = _edge_ok_from_masterstar(masterstar_fits, work, cfg_dict)
        work["edge_ok"] = edge_ok.reindex(work.index).fillna(False).astype(bool)

        comp_mask = (~work["vsx_known_variable"]) & (work["is_variable_candidate_rms"] == False)
        cand_mask = (work["is_candidate_combined"] == True) & (~work["vsx_known_variable"]) & (work["edge_ok"] == True)
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

        # Keep only edge-safe candidates in the candidate table
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

        # ---- Candidate pick (shared for map + light curve) ----
        candidate_options = {
            f"{row.get('vsx_name', str(row.catalog_id)) or str(row.catalog_id)} "
            f"(mag={float(pd.to_numeric(row.mag, errors='coerce')):.2f}, "
            f"rms={float(pd.to_numeric(row.rms_pct, errors='coerce')):.1f}%, "
            f"{row.get('detection_method','—')})": str(row.catalog_id)
            for _, row in candidates_df.iterrows()
        }
        selected_cid = ""
        selected_label_lc = ""
        if candidate_options:
            selected_label_lc = st.selectbox(
                "Vyber kandidáta:",
                options=list(candidate_options.keys()),
                key="var_lc_select2",
            )
            selected_cid = str(candidate_options.get(selected_label_lc, ""))
        elif not candidates_df.empty and "catalog_id" in candidates_df.columns:
            selected_cid = str(candidates_df["catalog_id"].iloc[0])

        # ---- Field map ----
        st.subheader("🗺️ Hviezdne pole (vyznačení kandidáti)")
        try:
            selected_row = results_df[results_df["catalog_id"].astype(str) == str(selected_cid)].iloc[0]
            cx = float(pd.to_numeric(selected_row.get("x"), errors="coerce"))
            cy = float(pd.to_numeric(selected_row.get("y"), errors="coerce"))
            _sel_name = str(selected_row.get("vsx_name", "") or "").strip()
            selected_label_map = _sel_name if _sel_name else str(selected_cid)
        except Exception:
            cx, cy = float("nan"), float("nan")
            selected_label_map = str(selected_cid)
        masterstar_fits = platesolve_dir / "MASTERSTAR.fits"
        if masterstar_fits.exists():
            cand_rows = work.loc[cand_mask].copy()
            cand_rows["x"] = pd.to_numeric(cand_rows.get("x"), errors="coerce")
            cand_rows["y"] = pd.to_numeric(cand_rows.get("y"), errors="coerce")
            cand_rows = cand_rows[np.isfinite(cand_rows["x"]) & np.isfinite(cand_rows["y"])].copy()
            cand_xy_label: list[tuple[float, float, str]] = []
            for _, rr in cand_rows.iterrows():
                lab = str(rr.get("vsx_name", "") or rr.get("catalog_id", ""))
                if not lab:
                    lab = str(rr.get("catalog_id", ""))
                cand_xy_label.append((float(rr["x"]), float(rr["y"]), lab))

            png_bytes = _render_field_image_with_candidates(
                str(masterstar_fits),
                cand_xy_label,
                selected_xy=(float(cx), float(cy)) if (np.isfinite(cx) and np.isfinite(cy)) else None,
                selected_label=str(selected_label_map),
            )
            if png_bytes:
                st.image(png_bytes, use_container_width=True)
            else:
                st.info("Nepodarilo sa vykresliť pole z MASTERSTAR.fits (chýba astropy/matplotlib?).")
        else:
            st.info("Pole nie je k dispozícii (chýba `MASTERSTAR.fits`).")

        st.subheader("📈 Light curve kandidáta")
        if selected_cid:

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
                    title=f"Raw light curve — {selected_label_lc or str(selected_cid)}",
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
        else:
            st.info("Nie sú dostupní kandidáti pre zobrazenie light curve.")

