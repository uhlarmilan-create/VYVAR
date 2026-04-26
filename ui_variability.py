from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from catalog_crossmatch import check_candidate_in_catalogs
from tess_verify import TessResult, run_tess_analysis
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


def count_edge_safe_combined_candidates(
    rms_df: pd.DataFrame,
    vdi_df: pd.DataFrame,
    platesolve_dir: Path,
    cfg_dict: dict[str, Any],
) -> int:
    """Počet kandidátov (RMS|VDI) bez VSX a s edge_ok — rovnaká logika ako v dashboarde."""
    if rms_df is None or rms_df.empty:
        return 0
    results_df = rms_df.copy()
    if vdi_df is not None and not vdi_df.empty:
        results_df = results_df.merge(
            vdi_df[["catalog_id", "vdi_score", "vdi_z_score", "is_variable_candidate"]],
            on="catalog_id",
            how="left",
            suffixes=("_rms", "_vdi"),
        )
        results_df = results_df.rename(columns={"is_variable_candidate": "is_variable_candidate_vdi"})
    else:
        results_df["vdi_score"] = np.nan
        results_df["vdi_z_score"] = np.nan
        results_df["is_variable_candidate_vdi"] = False
    if "is_variable_candidate" in results_df.columns and "is_variable_candidate_rms" not in results_df.columns:
        results_df = results_df.rename(columns={"is_variable_candidate": "is_variable_candidate_rms"})
    results_df["is_variable_candidate_rms"] = results_df["is_variable_candidate_rms"].fillna(False).astype(bool)
    results_df["is_variable_candidate_vdi"] = results_df["is_variable_candidate_vdi"].fillna(False).astype(bool)
    results_df["is_candidate_combined"] = (
        results_df["is_variable_candidate_rms"] | results_df["is_variable_candidate_vdi"]
    )
    work = results_df.copy()
    work["is_candidate_combined"] = work["is_candidate_combined"].fillna(False).astype(bool)
    work["vsx_known_variable"] = work["vsx_known_variable"].fillna(False).astype(bool)
    work["gaia_dr3_variable_catalog"] = work["gaia_dr3_variable_catalog"].fillna(False).astype(bool)
    masterstar_fits = platesolve_dir / "MASTERSTAR.fits"
    edge_ok = _edge_ok_from_masterstar(masterstar_fits, work, cfg_dict)
    work["edge_ok"] = edge_ok.reindex(work.index).fillna(False).astype(bool)
    cand_mask = work["is_candidate_combined"] & ~work["vsx_known_variable"] & work["edge_ok"]
    return int(cand_mask.sum())


def run_variability_detection_session(
    *,
    cfg: "AppConfig",
    draft_dir: Path,
    obs_group: str,
    flux_col: str,
    min_frames_pct: int,
    sigma_thr: float,
    mag_limit: float,
) -> tuple[dict[str, Any], int, tuple[str, str, int, float, float]]:
    """
    Načíta maticu fluxov, RMS + VDI (vždy obe). Nevolá Streamlit API.
    Vracia (results dict ako var_results, počet edge-safe kandidátov, _var_run_sig).
    """
    cfg_dict = cfg.to_dict()
    per_frame_dir = draft_dir / "detrended_aligned" / "lights" / str(obs_group)
    platesolve_dir = draft_dir / "platesolve" / str(obs_group)
    cfg_run = dict(cfg_dict)
    cfg_run["variability_min_frames_frac"] = float(min_frames_pct) / 100.0
    fm, meta, _bjd = _cached_load_matrix(
        str(per_frame_dir),
        flux_col,
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
        except Exception:  # noqa: BLE001
            comp_rms_map = {}
    results: dict[str, Any] = {
        "flux_matrix": fm,
        "metadata": meta,
        "comp_ids": comp_ids,
        "obs_group": str(obs_group),
        "flux_col": flux_col,
        "comp_rms_map": comp_rms_map,
    }
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
    cfg_run = dict(cfg_dict)
    cfg_run["variability_min_frames"] = int(cfg_run.get("variability_min_frames", 30))
    vdi_df = compute_vdi(fm, meta, min_frames=30, config=cfg_run)
    results["vdi_df"] = vdi_df
    var_sig = (str(obs_group), str(flux_col), int(min_frames_pct), float(sigma_thr), float(mag_limit))
    n_cand = count_edge_safe_combined_candidates(rms_df, vdi_df, platesolve_dir, cfg_dict)
    return results, n_cand, var_sig


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


def _variability_crossmatch_dialog_body() -> None:
    cid = str(st.session_state.get("var_cm_cid", "") or "")
    ra = st.session_state.get("var_cm_ra")
    dec = st.session_state.get("var_cm_dec")
    mag = st.session_state.get("var_cm_mag")
    st.markdown(f"**catalog_id:** `{cid}`")
    if not (isinstance(ra, (int, float)) and isinstance(dec, (int, float)) and np.isfinite(ra) and np.isfinite(dec)):
        st.error("Neplatné RA/Dec pre crossmatch.")
        return
    mag_f = float(mag) if isinstance(mag, (int, float)) and np.isfinite(mag) else None
    with st.spinner("Vyhľadávam v katalógoch (do 30 s)…"):
        res = check_candidate_in_catalogs(float(ra), float(dec), mag=mag_f, radius_arcsec=10.0)
    st.session_state["crossmatch_result"] = res
    bullets_map = st.session_state.setdefault("var_catalog_bullets", {})
    bullets_map[str(cid)] = "\n".join(res.catalog_summary_bullets())

    tab1, tab2, tab3 = st.tabs(["Katalógy", "TESS", "Export"])
    with tab1:
        catalog_order = [
            "SIMBAD",
            "VSX",
            "ASAS-SN",
            "ZTF",
            "Gaia varisum",
            "ATLAS",
            "CSS",
            "KELT",
            "VSBS",
            "TESS-EB",
        ]
        for cat in catalog_order:
            lst = res.matches.get(cat, [])
            err = res.errors.get(cat)
            if err:
                st.markdown(f":gray[**{cat}** — chyba: {err}]")
            elif not lst:
                st.markdown(f":gray[**{cat}** — žiadny match]")
            else:
                with st.expander(f"{cat} — {len(lst)} záznam(ov)", expanded=True):
                    for m in lst:
                        st.markdown(
                            f"**{m.name}**  \n"
                            f"typ: {m.var_type or '—'} · P: {m.period} · amp: {m.amplitude} · "
                            f"dr: {m.delta_r} · epoch: {m.epoch} · mag: {m.mag}"
                        )
        bp = res.best_period()
        if bp is not None and np.isfinite(bp):
            st.metric("Best period (priorita VSX→ASAS-SN→ZTF…)", f"{bp:.6g} d")
        else:
            st.metric("Best period (priorita VSX→ASAS-SN→ZTF…)", "—")
    with tab2:
        candidate_catalog_id = str(cid)
        candidate_ra = float(ra)
        candidate_dec = float(dec)
        candidate_mag = mag_f
        tess_store: dict[str, TessResult] = st.session_state["tess_results"]
        tess_result = tess_store.get(candidate_catalog_id)

        if tess_result is None:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.info(
                    "TESS poskytuje typicky 1–20+ sektorov závisí od ekliptiky a dĺžky pozorovania. "
                    "Analýza stiahne FFI cutouty (TessCut), odočíta pozadie, očistí LC, nájde periódu "
                    "(Lomb–Scargle v denných oknách alebo použije hint z katalógov) a uloží CSV + PNG pre každý sektor."
                )
            with c2:
                if st.button("Spustiť TESS analýzu", type="primary", key=f"var_tess_run_{candidate_catalog_id}"):
                    cm = st.session_state.get("crossmatch_result")
                    period_hint = None
                    if cm is not None and hasattr(cm, "best_period"):
                        try:
                            period_hint = cm.best_period()
                        except Exception:
                            period_hint = None
                    photometry_dir = str(st.session_state.get("var_photometry_dir") or "").strip()
                    if not photometry_dir:
                        st.error("Chýba cesta k photometry (var_photometry_dir).")
                    else:
                        with st.spinner("Sťahujem TESS dáta…"):
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            def progress_callback(message: str, value: float) -> None:
                                status_text.text(str(message))
                                progress_bar.progress(float(min(1.0, max(0.0, value))))

                            result = run_tess_analysis(
                                catalog_id=candidate_catalog_id,
                                ra=candidate_ra,
                                dec=candidate_dec,
                                mag=candidate_mag,
                                photometry_dir=photometry_dir,
                                period_hint=period_hint,
                                progress_callback=progress_callback,
                            )
                        st.session_state["tess_results"][candidate_catalog_id] = result
                        st.rerun()
        else:
            if tess_result.error_global:
                st.error(tess_result.error_global)
                if st.button("Skúsiť znova", key=f"var_tess_retry_{candidate_catalog_id}"):
                    st.session_state["tess_results"].pop(candidate_catalog_id, None)
                    st.rerun()
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sektory nájdené", str(tess_result.total_sectors_found))
                m2.metric("Sektory OK", str(tess_result.total_sectors_ok))
                bp = tess_result.best_period()
                m3.metric("Perioda P", f"{bp:.6f} d" if bp is not None and np.isfinite(bp) else "—")
                p2c = tess_result.period_2p_consensus
                m4.metric("Perioda 2P", f"{p2c:.6f} d" if p2c is not None and np.isfinite(p2c) else "—")

                ok_sectors = [s for s in tess_result.sectors if s.error is None]
                if not ok_sectors:
                    st.warning("Žiadny sektor bez chyby.")
                else:
                    sector_ids = [s.sector for s in ok_sectors]
                    pick = st.radio(
                        "Sektor",
                        options=sector_ids,
                        format_func=lambda s: f"Sektor {s}",
                        horizontal=True,
                        key=f"tess_sector_pick_{candidate_catalog_id}",
                    )
                    sector = next((s for s in ok_sectors if int(s.sector) == int(pick)), ok_sectors[0])
                    st.session_state["tess_selected_sector"] = int(pick)

                    if sector.lc_raw_path and Path(sector.lc_raw_path).exists():
                        try:
                            df_lc = pd.read_csv(sector.lc_raw_path)
                        except Exception:
                            df_lc = pd.DataFrame()
                        tcol = "time" if "time" in df_lc.columns else df_lc.columns[0]
                        fcol = "flux" if "flux" in df_lc.columns else df_lc.columns[1]
                        tt = pd.to_numeric(df_lc[tcol], errors="coerce")
                        ff = pd.to_numeric(df_lc[fcol], errors="coerce")
                        ok = np.isfinite(tt) & np.isfinite(ff)
                        figp = go.Figure(
                            data=[
                                go.Scatter(
                                    x=tt[ok],
                                    y=ff[ok],
                                    mode="markers",
                                    marker=dict(size=3, color="#1D9E75"),
                                )
                            ]
                        )
                        figp.update_layout(
                            title=f"{candidate_catalog_id} | Sektor {sector.sector} | raw LC",
                            xaxis_title="BJD",
                            yaxis_title="Flux [e-/s]",
                            height=300,
                            margin=dict(l=40, r=20, t=40, b=40),
                        )
                        st.plotly_chart(figp, use_container_width=True)
                    else:
                        st.warning("Chýba CSV lightcurve pre tento sektor.")

                    col_p, col_2p = st.columns(2)
                    pf = sector.period_found
                    with col_p:
                        if pf is not None and np.isfinite(pf) and sector.plot_phased_p_path and Path(sector.plot_phased_p_path).exists():
                            st.caption(f"Fázovaný P = {float(pf):.6f} d")
                            st.image(str(sector.plot_phased_p_path))
                        else:
                            st.caption("Fázovaný P — nedostupné")
                    with col_2p:
                        if pf is not None and np.isfinite(pf) and sector.plot_phased_2p_path and Path(sector.plot_phased_2p_path).exists():
                            st.caption(f"Fázovaný 2P = {float(pf) * 2.0:.6f} d")
                            st.image(str(sector.plot_phased_2p_path))
                        else:
                            st.caption("Fázovaný 2P — nedostupné")
                    st.caption(
                        "Porovnaj P vs 2P — pre EA/EB binárky je správna 2P ak vidíš 2 nestejné minimá"
                    )

                    _acc_msg = st.session_state.get("accepted_period_msg", {})
                    if _acc_msg.get(candidate_catalog_id):
                        st.success(_acc_msg[candidate_catalog_id])

                    acc = st.session_state.setdefault("accepted_period", {})
                    col_accept, col_custom = st.columns(2)
                    with col_accept:
                        b1, b2 = st.columns(2)
                        with b1:
                            if bp is not None and np.isfinite(bp):
                                if st.button(f"Použiť P = {float(bp):.6f} d", key=f"tess_use_p_{candidate_catalog_id}"):
                                    acc[candidate_catalog_id] = float(bp)
                                    st.session_state.setdefault("accepted_period_msg", {})[
                                        candidate_catalog_id
                                    ] = f"Uložená perioda P = {float(bp):.6f} d pre {candidate_catalog_id}."
                        with b2:
                            if p2c is not None and np.isfinite(p2c):
                                if st.button(f"Použiť 2P = {float(p2c):.6f} d", key=f"tess_use_2p_{candidate_catalog_id}"):
                                    acc[candidate_catalog_id] = float(p2c)
                                    st.session_state.setdefault("accepted_period_msg", {})[
                                        candidate_catalog_id
                                    ] = f"Uložená perioda 2P = {float(p2c):.6f} d pre {candidate_catalog_id}."
                    with col_custom:
                        cust = st.number_input(
                            "Vlastná perioda (d)",
                            min_value=0.0,
                            value=float(bp) if bp is not None and np.isfinite(bp) else 0.0,
                            step=0.0001,
                            format="%.6f",
                            key=f"tess_custom_num_{candidate_catalog_id}",
                        )
                        if st.button("Použiť vlastnú", key=f"tess_use_custom_{candidate_catalog_id}"):
                            if cust > 0:
                                acc[candidate_catalog_id] = float(cust)
                                st.session_state.setdefault("accepted_period_msg", {})[
                                    candidate_catalog_id
                                ] = f"Uložená vlastná perioda = {float(cust):.6f} d pre {candidate_catalog_id}."
                            else:
                                st.warning("Zadaj kladnú periodu.")
    with tab3:
        st.info("Export LC dát — bude implementovaný po TESS analýze")


_variability_crossmatch_dialog = (
    st.dialog("Kandidát — katalógový crossmatch")(_variability_crossmatch_dialog_body)
    if hasattr(st, "dialog")
    else _variability_crossmatch_dialog_body
)


def render_variability_dashboard(
    pipeline: "AstroPipeline",
    cfg: "AppConfig",
    *,
    draft_id: int | None = None,
    draft_dir_override: Path | None = None,
) -> None:
    st.header("🔍 Variability Detection")
    st.session_state.setdefault("tess_results", {})
    st.session_state.setdefault("accepted_period", {})
    st.session_state.setdefault("accepted_period_msg", {})
    st.session_state.setdefault("var_analysis_done", False)
    st.session_state.setdefault("var_analysis_timestamp", None)
    st.session_state.setdefault("pdf_ready", False)
    st.session_state.setdefault("var_candidate_count_autorun", 0)

    # Draft resolution consistent with other dashboards (Aperture Photometry).
    if draft_id is None and draft_dir_override is None:
        st.info("Najprv načítaj draft.")
        return
    if draft_dir_override is not None and Path(draft_dir_override).is_dir():
        draft_dir = Path(draft_dir_override).resolve()
    else:
        draft_dir = (Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft_id):06d}").resolve()

    if st.session_state.get("var_analysis_done"):
        _vts = str(st.session_state.get("var_analysis_timestamp") or "—")
        _nc = int(st.session_state.get("var_candidate_count_autorun", 0))
        st.success(
            f"Analýza dokončená: {_vts} | Kandidáti: {_nc} | Spustená automaticky po Aperture Photometry"
        )
    elif not st.session_state.get("var_analysis_done"):
        _vr = st.session_state.get("var_results") or {}
        _rms_m = _vr.get("rms_df")
        if isinstance(_rms_m, pd.DataFrame) and not _rms_m.empty:
            st.info("Výsledky z manuálnej analýzy — spusti Aperture Photometry pre auto-update")

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

    obs_group = st.selectbox("Setup:", options=obs_groups, key="var_obs_group")

    flux_source = st.radio("Flux zdroj:", options=["dao_flux", "psf_flux"], horizontal=True, key="var_flux_source")
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
    _phot_dir = platesolve_dir / "photometry"
    _phot_dir.mkdir(parents=True, exist_ok=True)
    st.session_state["var_photometry_dir"] = str(_phot_dir.resolve())
    with st.expander("Debug paths", expanded=False):
        st.write(f"per_frame_dir: {per_frame_dir} (exists={per_frame_dir.exists()})")
        st.write(f"platesolve_dir: {platesolve_dir} (exists={platesolve_dir.exists()})")

    _var_sig = (str(obs_group), str(flux_source), int(min_frames_pct), float(sigma_thr), float(mag_limit))
    if st.session_state.get("_var_run_sig") != _var_sig:
        try:
            with st.spinner("Načítavam flux maticu a počítam RMS + VDI…"):
                results, _n_cand_unused, sig = run_variability_detection_session(
                    cfg=cfg,
                    draft_dir=draft_dir,
                    obs_group=str(obs_group),
                    flux_col=str(flux_source),
                    min_frames_pct=int(min_frames_pct),
                    sigma_thr=float(sigma_thr),
                    mag_limit=float(mag_limit),
                )
            st.session_state["var_results"] = results
            st.session_state["_var_run_sig"] = sig
            st.session_state["var_catalog_bullets"] = {}
        except Exception as exc:  # noqa: BLE001
            st.error(f"Chyba analýzy: {exc}")
            logging.exception("Variability analysis failed")

    res = st.session_state.get("var_results") or {}
    rms_df: pd.DataFrame = res.get("rms_df") if isinstance(res.get("rms_df"), pd.DataFrame) else pd.DataFrame()
    vdi_df: pd.DataFrame = res.get("vdi_df") if isinstance(res.get("vdi_df"), pd.DataFrame) else pd.DataFrame()

    if rms_df.empty:
        st.session_state["var_candidates"] = []

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
        st.session_state["var_candidates"] = [str(x).strip() for x in cand["catalog_id"].astype(str).tolist() if str(x).strip()]
        cand["Vizier"] = [
            _vizier_link(float(pd.to_numeric(r.get("ra_deg"), errors="coerce")), float(pd.to_numeric(r.get("dec_deg"), errors="coerce")))
            for _, r in cand.iterrows()
        ]
        bullets_map: dict[str, str] = st.session_state.setdefault("var_catalog_bullets", {})
        cand["katalógy"] = cand["catalog_id"].astype(str).map(lambda cid: bullets_map.get(str(cid), "—"))

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
            "katalógy",
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

        sel_export = set(str(x) for x in (st.session_state.get("selected_for_export") or []))
        candidates_df["export"] = candidates_df["catalog_id"].astype(str).isin(sel_export)

        st.markdown("**Kandidáti** — zaškrtni `export` pre pridanie do `variable_targets.csv`; vyber riadok a otvor crossmatch.")
        disabled_cols = [c for c in candidates_df.columns if c != "export"]
        edited_cand = st.data_editor(
            candidates_df,
            column_config={
                "export": st.column_config.CheckboxColumn("export", default=False, help="Pridať do variable_targets.csv"),
                "katalógy": st.column_config.TextColumn("katalógy", width="large"),
            },
            disabled=disabled_cols,
            use_container_width=True,
            hide_index=True,
            key="var_candidates_editor",
        )
        if "export" in edited_cand.columns:
            _em = edited_cand["export"].fillna(False).astype(bool)
            st.session_state["selected_for_export"] = edited_cand.loc[_em, "catalog_id"].astype(str).tolist()
        else:
            st.session_state["selected_for_export"] = []

        coord_by_cid: dict[str, dict[str, float | None]] = {}
        for _, r in cand.iterrows():
            cid = str(r.get("catalog_id", ""))
            if not cid:
                continue
            coord_by_cid[cid] = {
                "ra": float(pd.to_numeric(r.get("ra_deg"), errors="coerce")),
                "dec": float(pd.to_numeric(r.get("dec_deg"), errors="coerce")),
                "mag": float(pd.to_numeric(r.get("mag"), errors="coerce")),
            }
        st.session_state["var_cm_coord_by_cid"] = coord_by_cid

        def _cm_format(cid: str) -> str:
            if not cid:
                return "(žiadni kandidáti)"
            sub = candidates_df[candidates_df["catalog_id"].astype(str) == str(cid)]
            if sub.empty:
                return str(cid)
            r = sub.iloc[0]
            return (
                f"{cid} · mag={float(pd.to_numeric(r.get('mag'), errors='coerce')):.2f} · "
                f"rms={float(pd.to_numeric(r.get('rms_pct'), errors='coerce')):.1f}%"
            )

        def _on_crossmatch_pick() -> None:
            cid = str(st.session_state.get("var_cm_pick_id") or "")
            if not cid:
                return
            cr = st.session_state.get("var_cm_coord_by_cid", {}).get(cid, {})
            st.session_state["var_cm_cid"] = cid
            st.session_state["var_cm_ra"] = cr.get("ra")
            st.session_state["var_cm_dec"] = cr.get("dec")
            st.session_state["var_cm_mag"] = cr.get("mag")
            st.session_state["var_cm_do_open"] = True

        cm_cids = [str(x) for x in candidates_df["catalog_id"].astype(str).tolist() if str(x)]
        if cm_cids:
            cx1, cx2 = st.columns([3, 1])
            with cx1:
                st.selectbox(
                    "Vyber kandidáta pre katalógový crossmatch (modal):",
                    options=cm_cids,
                    format_func=_cm_format,
                    key="var_cm_pick_id",
                    on_change=_on_crossmatch_pick,
                )
            with cx2:
                st.caption("")  # align button vertically
                if st.button("Otvoriť crossmatch", key="var_cm_open_btn", type="secondary"):
                    cid0 = str(st.session_state.get("var_cm_pick_id", cm_cids[0]) or cm_cids[0])
                    cr0 = coord_by_cid.get(cid0, {})
                    st.session_state["var_cm_cid"] = cid0
                    st.session_state["var_cm_ra"] = cr0.get("ra")
                    st.session_state["var_cm_dec"] = cr0.get("dec")
                    st.session_state["var_cm_mag"] = cr0.get("mag")
                    _variability_crossmatch_dialog()
        else:
            st.caption("Žiadni kandidáti pre crossmatch.")

        if st.session_state.pop("var_cm_do_open", False):
            _variability_crossmatch_dialog()

        colA, colB = st.columns(2)
        with colA:
            if st.button("📥 Export kandidátov CSV", key="var_export"):
                out_csv = draft_dir / "platesolve" / str(obs_group) / "variability_candidates.csv"
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                candidates_df.drop(columns=["export"], errors="ignore").to_csv(out_csv, index=False)
                st.success(f"Uložené: {out_csv}")
        with colB:
            st.caption("")

        if st.button("➕ Pridať vybrané do variable_targets.csv", key="var_add_to_var2"):
            selected_ids = list(st.session_state.get("selected_for_export") or [])
            if not selected_ids:
                st.warning("Nie sú vybrané žiadne hviezdy (stĺpec export).")
            else:
                vt_path = platesolve_dir / "variable_targets.csv"
                if vt_path.exists():
                    vt_df = pd.read_csv(vt_path, low_memory=False)
                else:
                    vt_df = pd.DataFrame()

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

