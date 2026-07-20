"""Standalone ePSF photometry dashboard tab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import streamlit as st

from utils import resolve_draft_dir_path

if TYPE_CHECKING:
    from config import AppConfig

_GAIA_ID_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}


def _find_epsf_model_path(draft_dir: Path, setup_name: str | None = None) -> Path | None:
    if setup_name:
        p = draft_dir / "platesolve" / str(setup_name) / "masterstar_epsf.fits"
        if p.is_file():
            return p
    matches = sorted(draft_dir.glob("platesolve/*/masterstar_epsf.fits"))
    return matches[0] if matches else None


def _find_proc_csv_dir(draft_dir: Path, setup_name: str | None = None) -> Path | None:
    if setup_name:
        p = draft_dir / "detrended_aligned" / "lights" / str(setup_name)
        if p.is_dir() and any(p.glob("proc_*.csv")):
            return p
    for cand in sorted(draft_dir.glob("detrended_aligned/lights/*/")):
        if cand.is_dir() and any(cand.glob("proc_*.csv")):
            return cand
    return None


def _load_psf_lc_from_proc(proc_dir: Path, cid: str) -> pd.DataFrame | None:
    """Load per-frame PSF/DAO flux and time for one star from ``proc_*.csv``."""
    cid_s = str(cid).strip()
    if not cid_s:
        return None
    usecols = [
        "catalog_id",
        "bjd_tdb_mid",
        "jd_mid",
        "psf_flux",
        "psf_fit_ok",
        "psf_chi2",
        "dao_flux",
    ]
    chunks: list[pd.DataFrame] = []
    for csv_path in sorted(proc_dir.glob("proc_*.csv")):
        try:
            df = pd.read_csv(
                csv_path,
                usecols=usecols,
                low_memory=False,
                dtype={"catalog_id": str},
            )
        except Exception:  # noqa: BLE001
            # EXC-0510: T3 -- UI diagnostic/plot only (csv_path, low_memory=False, dtype={'catalog_id': str} / ) / ex... (EXCEPT-BULK 2026-07-08)
            try:
                df = pd.read_csv(
                    csv_path, low_memory=False, dtype={"catalog_id": str}
                )
            except Exception:  # noqa: BLE001
                continue
            if "catalog_id" not in df.columns:
                continue
            keep = [c for c in usecols if c in df.columns]
            df = df[keep]
        try:
            from gaia_catalog_id import normalize_gaia_source_id as _norm_cid  # noqa: PLC0415

            df = df.copy()
            df["catalog_id"] = df["catalog_id"].map(_norm_cid)
        except Exception:  # noqa: BLE001
            df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        sub = df[df["catalog_id"].astype(str).str.strip() == cid_s]
        if not sub.empty:
            chunks.append(sub)
    if not chunks:
        return None
    out = pd.concat(chunks, ignore_index=True)
    if "bjd_tdb_mid" in out.columns:
        out["bjd"] = pd.to_numeric(out["bjd_tdb_mid"], errors="coerce")
    elif "jd_mid" in out.columns:
        out["bjd"] = pd.to_numeric(out["jd_mid"], errors="coerce")
    else:
        out["bjd"] = float("nan")
    return out


def _render_epsf_dashboard_body(
    *,
    draft_dir: Path,
    setup_name: str,
    output_dir: Path,
    active_targets_df: pd.DataFrame,
    cfg: "AppConfig",
    epsf_fits: Path,
) -> None:
    """Metrics table + aperture vs PSF LC overlay."""
    st.caption(f"Model: `{epsf_fits}` . setup: `{setup_name}`")

    proc_dir = _find_proc_csv_dir(draft_dir, setup_name)
    if proc_dir is None:
        st.info("No per-frame proc_*.csv found for this setup.")
        return

    from photometry_core import load_epsf_metrics_for_draft

    with st.spinner("Loading ePSF metrics..."):
        epsf_df = load_epsf_metrics_for_draft(proc_dir, active_targets_df)

    if epsf_df.empty:
        st.info("No PSF columns in per-frame catalogs (run pipeline with PSF enabled).")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stars with PSF stats", f"{len(epsf_df)}")
    c2.metric("Stars > 50% PSF frames", f"{int((epsf_df['pct_psf_ok'] > 50).sum())}")
    med_chi2 = pd.to_numeric(epsf_df["median_chi2"], errors="coerce").median()
    c3.metric("Median chi^2 (per star)", f"{med_chi2:.1f}" if pd.notna(med_chi2) else "-")
    c4.metric("chi^2 threshold", f"{float(getattr(cfg, 'psf_chi2_threshold', 50.0)):.0f}")

    st.markdown("**Per-star ePSF metrics**")
    display_cols = [
        "catalog_id",
        "n_frames",
        "n_psf_ok",
        "pct_psf_ok",
        "mean_chi2",
        "min_chi2",
        "psf_dao_ratio",
    ]
    if "vsx_name" in epsf_df.columns:
        display_cols = ["vsx_name"] + display_cols
    st.dataframe(
        epsf_df[display_cols].head(50),
        width="stretch",
        column_config={
            "pct_psf_ok": st.column_config.ProgressColumn(
                "PSF %", min_value=0, max_value=100
            ),
            "mean_chi2": st.column_config.NumberColumn("mean chi^2", format="%.1f"),
            "min_chi2": st.column_config.NumberColumn("min chi^2", format="%.1f"),
            "psf_dao_ratio": st.column_config.NumberColumn("PSF/DAO ratio", format="%.3f"),
        },
    )

    def _label(cid: str) -> str:
        if "vsx_name" not in epsf_df.columns:
            return str(cid)
        hit = epsf_df.loc[epsf_df["catalog_id"].astype(str) == str(cid), "vsx_name"]
        if hit.empty:
            return str(cid)
        nm = str(hit.iloc[0]).strip()
        return f"{nm} ({cid})" if nm else str(cid)

    selected_cid = st.selectbox(
        "Select star for aperture vs PSF LC overlay",
        options=epsf_df["catalog_id"].astype(str).tolist(),
        format_func=_label,
        key=f"epsf_lc_overlay_{setup_name}",
    )
    if not selected_cid:
        return

    lc_csv = output_dir / "lightcurves" / f"lightcurve_{selected_cid}.csv"
    proc_data = _load_psf_lc_from_proc(proc_dir, selected_cid)
    if not lc_csv.is_file() or proc_data is None or proc_data.empty:
        st.caption("LC CSV or per-frame PSF series not available for overlay.")
        return

    try:
        import plotly.graph_objects as go
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Plotly unavailable: {exc}")
        return

    lc_df = pd.read_csv(lc_csv, low_memory=False)
    if "flag" in lc_df.columns:
        normal = lc_df[lc_df["flag"].astype(str) == "normal"].copy()
    else:
        normal = lc_df.copy()

    fig = go.Figure()
    if "bjd" in normal.columns and "mag_calib" in normal.columns:
        fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(normal["bjd"], errors="coerce"),
                y=pd.to_numeric(normal["mag_calib"], errors="coerce"),
                mode="markers",
                name="Aperture",
                marker=dict(color="royalblue", size=4),
            )
        )

    psf_ok = proc_data[proc_data["psf_fit_ok"].fillna(False).astype(bool)].copy()
    if not psf_ok.empty and "psf_flux" in psf_ok.columns:
        psf_flux = pd.to_numeric(psf_ok["psf_flux"], errors="coerce").clip(lower=1e-10)
        psf_mag = -2.5 * np.log10(psf_flux)
        if "mag_calib" in normal.columns and psf_mag.notna().any():
            apt_med = pd.to_numeric(normal["mag_calib"], errors="coerce").median()
            psf_med = psf_mag.median()
            offset = float(apt_med - psf_med) if pd.notna(apt_med) and pd.notna(psf_med) else 0.0
        else:
            offset = 0.0
        fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(psf_ok["bjd"], errors="coerce"),
                y=psf_mag + offset,
                mode="markers",
                name="PSF (ZP norm. to aperture)",
                marker=dict(color="orange", size=5, symbol="diamond"),
            )
        )

    fig.update_layout(
        title=f"Aperture vs PSF - {_label(selected_cid)}",
        xaxis_title="BJD",
        yaxis_title="mag",
        yaxis_autorange="reversed",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")


def _epsf_meta_caption(epsf_fits: Path | None) -> str:
    if epsf_fits is None or not epsf_fits.is_file():
        return "No ePSF model found - click **RUN ePSF** to build one."
    meta_json = epsf_fits.parent / "masterstar_epsf_meta.json"
    if not meta_json.is_file():
        return f"Model: `{epsf_fits.name}` (no meta JSON)"
    try:
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        fwhm = meta.get("fwhm_px")
        fwhm_s = f"{float(fwhm):.2f}px" if fwhm is not None else "?"
        return (
            f"Model: {meta.get('n_stars_used', '?')} PSF stars, "
            f"FWHM {fwhm_s}, oversampling {meta.get('oversampling', '?')}x"
        )
    except Exception:  # noqa: BLE001
        # EXC-0511: T3 -- UI diagnostic/plot only (f'FWHM {fwhm_s}, oversampling {meta.get('oversampling', '?')}x... (EXCEPT-BULK 2026-07-08)
        return f"Model: `{epsf_fits.name}`"


def render_epsf_dashboard(
    draft_dir: Path | None,
    cfg: "AppConfig",
    *,
    draft_id: int | None = None,
) -> None:
    """Standalone ePSF dashboard tab."""
    st.header("[microscope] ePSF Photometry")

    if not bool(getattr(cfg, "psf_photometry_enabled", False)):
        st.warning(
            "! `psf_photometry_enabled` is **False** in Settings - PSF columns will be "
            "empty after export. Enable PSF in Settings -> Tools before **RUN ePSF**."
        )

    resolved = resolve_draft_dir_path(
        draft_dir,
        draft_id,
        cfg.archive_root,
        drafts_before_session=True,
    )
    if resolved is None:
        st.info("No active draft. Load a draft above or run VAR-STREM.")
        return

    draft_path = Path(resolved)
    st.caption(f"Draft directory: `{draft_path}`")

    from ui_aperture_photometry import _find_phase2a_paths

    all_setups = _find_phase2a_paths(cfg, draft_id, draft_dir_override=draft_path)
    if not all_setups:
        st.warning("No platesolve setups with per-frame catalogs found for this draft.")
        return

    setup_options = list(all_setups.keys())
    selected_setup = st.selectbox(
        "Platesolve setup:",
        options=setup_options,
        key="epsf_setup_select",
    )
    paths = all_setups.get(str(selected_setup)) or {}
    output_dir = paths.get("output_dir")
    ms_fits = paths.get("masterstar_fits")
    og_dir = paths.get("obs_group_dir")
    pf_dir = paths.get("per_frame_csv_dir")
    ms_csv = (Path(og_dir) / "masterstars_full_match.csv") if og_dir is not None else None
    ps_dir = Path(og_dir) if og_dir is not None else None

    epsf_fits = _find_epsf_model_path(draft_path, str(selected_setup))

    st.divider()
    _col_btn, _col_info = st.columns([1, 3])
    with _col_btn:
        _run_epsf = st.button(
            "! RUN ePSF Photometry",
            type="primary",
            key=f"epsf_run_btn_{selected_setup}",
            help="Build ePSF model from MASTERSTAR + re-export per-frame catalogs with PSF columns",
        )
    with _col_info:
        st.caption(_epsf_meta_caption(epsf_fits))

    if _run_epsf:
        if ms_fits is None or not Path(ms_fits).is_file():
            st.error("MASTERSTAR.fits not found. Run MAKE MASTERSTAR first.")
        elif pf_dir is None or not Path(pf_dir).is_dir():
            st.error("Aligned frames / proc_*.csv directory not found.")
        elif ps_dir is None or not ps_dir.is_dir():
            st.error("Platesolve setup directory not found.")
        elif ms_csv is None or not ms_csv.is_file():
            st.error(f"masterstars_full_match.csv not found under `{ps_dir}`.")
        else:
            st.session_state["vyvar_pending_job"] = {
                "kind": "run_epsf",
                "label": "RUN ePSF Photometry...",
                "archive_path": str(draft_path),
                "masterstar_fits_path": str(ms_fits),
                "masterstars_csv_path": str(ms_csv),
                "per_frame_csv_dir": str(pf_dir),
                "platesolve_dir": str(ps_dir),
                "draft_id": int(draft_id) if draft_id is not None else None,
                "setup_name": str(selected_setup),
                "dao_fwhm_px": float(getattr(cfg, "sips_dao_fwhm_px", 3.7) or 3.7),
                "dao_threshold_sigma": float(
                    getattr(cfg, "sips_dao_threshold_sigma", 3.5) or 3.5
                ),
            }
            st.rerun()

    _last = st.session_state.get("vyvar_last_job_output")
    if isinstance(_last, dict) and _last.get("job_kind") == "run_epsf":
        if _last.get("status") == "ok":
            st.success(str(_last.get("message") or "RUN ePSF completed."))
        elif _last.get("error"):
            st.error(str(_last.get("error")))

    if not isinstance(output_dir, Path):
        st.warning("Photometry output directory missing for this setup.")
        return

    if epsf_fits is None or not epsf_fits.is_file():
        st.info(
            "No `masterstar_epsf.fits` for this setup yet. Click **RUN ePSF** above "
            "(requires MASTERSTAR + aligned frames)."
        )
        return

    at_df = pd.DataFrame()
    at_csv = paths.get("active_targets_csv")
    if at_csv is not None and Path(at_csv).is_file():
        try:
            at_df = pd.read_csv(at_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
        except Exception:  # noqa: BLE001
            at_df = pd.DataFrame()

    _render_epsf_dashboard_body(
        draft_dir=draft_path,
        setup_name=str(selected_setup),
        output_dir=Path(output_dir),
        active_targets_df=at_df,
        cfg=cfg,
        epsf_fits=epsf_fits,
    )
