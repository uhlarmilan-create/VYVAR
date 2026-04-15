"""Aperture Photometry Lightcurves — Fáza 2A UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from config import AppConfig
    from pipeline import AstroPipeline


# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------


def _find_phase2a_paths(cfg: "AppConfig", draft_id: int | None) -> dict[str, Path | None]:
    """Nájdi všetky cesty potrebné pre Fázu 2A."""
    if draft_id is None:
        return {}
    try:
        archive = Path(cfg.archive_root)
        draft_dir = archive / "Drafts" / f"draft_{int(draft_id):06d}"
        ps_dir = draft_dir / "platesolve"

        obs_group_dir: Path | None = None
        for subdir in sorted(ps_dir.iterdir()):
            if subdir.is_dir() and (subdir / "per_frame_catalog_index.csv").exists():
                obs_group_dir = subdir
                break
        if obs_group_dir is None:
            return {}

        aligned_root = draft_dir / "detrended_aligned" / "lights"
        per_frame_dir: Path | None = None
        detrended_dir: Path | None = None
        if aligned_root.exists():
            for subdir in sorted(aligned_root.iterdir()):
                if subdir.is_dir():
                    per_frame_dir = subdir
                    detrended_dir = subdir
                    break

        photometry_dir = obs_group_dir / "photometry"

        return {
            "masterstar_fits": ps_dir / "MASTERSTAR.fits",
            "active_targets_csv": photometry_dir / "active_targets.csv",
            "comparison_stars_csv": photometry_dir / "comparison_stars_per_target.csv",
            "per_frame_csv_dir": per_frame_dir,
            "detrended_aligned_dir": detrended_dir,
            # Fáza 2A zapisuje:
            # - photometry_summary.csv do output_dir priamo
            # - lightcurve CSV/PNG + per-target field_map do output_dir / "lightcurves"
            # - globálny field_map.png do output_dir priamo
            "output_dir": photometry_dir,
            "photometry_dir": photometry_dir,
        }
    except Exception:  # noqa: BLE001
        return {}


def _load_fwhm(cfg: "AppConfig", draft_id: int | None) -> float:
    if draft_id is None:
        return 3.7
    try:
        from astropy.io import fits as astrofits

        ms = (
            Path(cfg.archive_root)
            / "Drafts"
            / f"draft_{int(draft_id):06d}"
            / "platesolve"
            / "MASTERSTAR.fits"
        )
        if ms.exists():
            with astrofits.open(ms, memmap=False) as hdul:
                v = float(hdul[0].header.get("VY_FWHM", 3.7))
                if 1.0 < v < 15.0:
                    return round(v, 3)
    except Exception:  # noqa: BLE001
        pass
    return 3.7


def _phase2a_results_exist(output_dir: Path | None) -> bool:
    if output_dir is None:
        return False
    return (output_dir / "photometry_summary.csv").exists()


def _phase2a_timestamp(output_dir: Path | None) -> str:
    if output_dir is None:
        return ""
    p = output_dir / "photometry_summary.csv"
    if p.exists():
        import datetime

        return datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%d.%m.%Y %H:%M")
    return ""


def _load_summary(output_dir: Path) -> pd.DataFrame:
    p = output_dir / "photometry_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Per-target lightcurve + field map view
# ---------------------------------------------------------------------------


def _render_target_detail(
    target_row: pd.Series,
    output_dir: Path,
    show_outliers: bool,
) -> None:
    """Zobraz lightcurve PNG + field map PNG pre jeden target."""
    catalog_id = str(target_row.get("catalog_id", ""))
    vsx_name = str(target_row.get("vsx_name", catalog_id))
    lc_dir = output_dir / "lightcurves"

    col_lc, col_map = st.columns([3, 2])

    with col_lc:
        st.markdown(f"**Svetelná krivka — {vsx_name}**")
        lc_png = lc_dir / f"lightcurve_{catalog_id}.png"
        if lc_png.exists():
            st.image(str(lc_png), use_container_width=True)
        else:
            st.info("PNG svetelnej krivky neexistuje. Spusti Fázu 2A.")

        lc_csv = lc_dir / f"lightcurve_{catalog_id}.csv"
        if lc_csv.exists():
            try:
                lc_df = pd.read_csv(lc_csv)
                if not show_outliers and "flag" in lc_df.columns:
                    lc_df = lc_df[lc_df["flag"] == "normal"]

                if (
                    not lc_df.empty
                    and "bjd" in lc_df.columns
                    and "mag_calib" in lc_df.columns
                ):
                    try:
                        import plotly.graph_objects as go
                    except Exception:  # noqa: BLE001
                        go = None  # type: ignore[assignment]

                    if go is not None:
                        fig = go.Figure()
                        flag_colors_plotly = {
                            "normal": "#1a1a2e",
                            "outlier_hi": "#ff6b35",
                            "outlier_lo": "#7b2d8b",
                            "saturated": "#aaaaaa",
                            "no_data": "#cccccc",
                        }

                        if "flag" not in lc_df.columns:
                            lc_df = lc_df.assign(flag="normal")

                        for flag, color in flag_colors_plotly.items():
                            sub = lc_df[lc_df["flag"] == flag].dropna(
                                subset=["bjd", "mag_calib"]
                            )
                            if sub.empty:
                                continue
                            err = (
                                sub["err"].fillna(0).tolist()
                                if "err" in sub.columns
                                else None
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=sub["bjd"],
                                    y=sub["mag_calib"],
                                    error_y=(
                                        dict(array=err, visible=True)
                                        if err is not None
                                        else None
                                    ),
                                    mode="markers",
                                    marker=dict(color=color, size=5),
                                    name=flag,
                                )
                            )

                        fig.update_layout(
                            yaxis=dict(autorange="reversed", title="mag_calib"),
                            xaxis=dict(title="BJD (TDB)"),
                            height=350,
                            margin=dict(l=40, r=20, t=20, b=40),
                            legend=dict(orientation="h", y=1.1),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption("Interaktívny graf nedostupný (plotly nie je nainštalovaný).")
            except Exception as e:  # noqa: BLE001
                st.caption(f"Interaktívny graf nedostupný: {e}")

    with col_map:
        st.markdown("**Field map**")
        fm_png = lc_dir / f"field_map_{catalog_id}.png"
        if fm_png.exists():
            st.image(str(fm_png), use_container_width=True)
        else:
            global_fm = output_dir / "field_map.png"
            if global_fm.exists():
                st.image(str(global_fm), use_container_width=True)
                st.caption("(globálny field map — per-target nie je dostupný)")
            else:
                st.info("Field map neexistuje.")

        rms = target_row.get("lc_rms")
        n_comp = target_row.get("n_good_comp")
        ap = target_row.get("aperture_px")
        cols = st.columns(3)
        if rms is not None:
            cols[0].metric("lc_rms", f"{float(rms):.4f}")
        if n_comp is not None:
            cols[1].metric("good comp", int(n_comp))
        if ap is not None:
            cols[2].metric("apertura", f"{float(ap):.1f}px")


# ---------------------------------------------------------------------------
# Hlavný render
# ---------------------------------------------------------------------------


def render_aperture_photometry(
    cfg: "AppConfig",
    draft_id: int | None,
    pipeline: "AstroPipeline",
) -> None:
    """Hlavná funkcia pre Aperture Photometry tab."""
    _ = pipeline
    st.header("Aperture Photometry")
    st.caption("Fáza 2A: Aperturná fotometria premenných hviezd.")

    if draft_id is None:
        st.info("Žiadny aktívny draft.")
        return

    paths = _find_phase2a_paths(cfg, draft_id)
    if not paths:
        st.warning("Nenájdené vstupné súbory.")
        return

    output_dir = paths.get("output_dir")
    fwhm_px = _load_fwhm(cfg, draft_id)

    at_csv = paths.get("active_targets_csv")
    comp_csv = paths.get("comparison_stars_csv")
    if at_csv is None or not at_csv.exists() or comp_csv is None or not comp_csv.exists():
        st.error("❌ Najprv spusti Fázu 0+1 (Select Stars).")
        return

    exists = _phase2a_results_exist(output_dir)

    if exists:
        ts = _phase2a_timestamp(output_dir)
        st.success(f"✅ Fáza 2A prebehla: {ts}")
        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button(
                "🔄 Prepočítať Fázu 2A", key="phase2a_run", type="secondary"
            )
        with col2:
            st.caption(f"FWHM: {fwhm_px} px")
    else:
        st.info("Fáza 2A ešte nebehala.")
        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button(
                "▶ Spustiť Fázu 2A", key="phase2a_run_first", type="primary"
            )
        with col2:
            st.caption(f"FWHM: {fwhm_px} px")

    if run_btn:
        ms_fits = paths.get("masterstar_fits")
        pf_dir = paths.get("per_frame_csv_dir")
        dt_dir = paths.get("detrended_aligned_dir")

        missing: list[str] = []
        if ms_fits is None or not ms_fits.exists():
            missing.append("MASTERSTAR.fits")
        if pf_dir is None or not pf_dir.exists():
            missing.append("per-frame CSV adresár")
        if dt_dir is None or not dt_dir.exists():
            missing.append("detrended_aligned adresár")

        if missing:
            st.error(f"❌ Chýbajú súbory: {', '.join(missing)}")
        else:
            from photometry_phase2a import run_phase2a

            with st.spinner("Prebieha Fáza 2A — aperturná fotometria..."):
                try:
                    result = run_phase2a(
                        masterstar_fits_path=ms_fits,
                        active_targets_csv=at_csv,
                        comparison_stars_csv=comp_csv,
                        per_frame_csv_dir=pf_dir,
                        detrended_aligned_dir=dt_dir,
                        output_dir=output_dir,
                        fwhm_px=fwhm_px,
                    )
                    st.success(
                        f"✅ Hotovo: {result['n_lightcurves']} svetelných kriviek "
                        f"z {result['n_frames']} snímok."
                    )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ Chyba: {exc}")
                    logging.exception("Fáza 2A zlyhala")
            return

    if not exists:
        return

    if output_dir is None:
        st.warning("Output adresár nie je dostupný.")
        return

    summary_df = _load_summary(output_dir)
    if summary_df.empty:
        st.warning("photometry_summary.csv je prázdny.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Svetelné krivky", len(summary_df))
    if "lc_rms" in summary_df.columns:
        c2.metric("Median lc_rms", f"{summary_df['lc_rms'].median():.4f}")
        good = int((summary_df["lc_rms"] < 0.05).sum())
        c3.metric("RMS < 0.05 mag", good)
    if "n_good_comp" in summary_df.columns:
        c4.metric("Avg good comp", f"{summary_df['n_good_comp'].mean():.1f}")

    st.divider()

    show_outliers = st.toggle(
        "Zobraziť outlier a saturated body",
        value=True,
        key="phase2a_show_outliers",
    )

    name_col = "vsx_name" if "vsx_name" in summary_df.columns else "catalog_id"
    target_options = summary_df[name_col].fillna(summary_df["catalog_id"]).tolist()

    selected_idx = st.selectbox(
        "Vyber premennú hviezdu:",
        options=range(len(target_options)),
        format_func=lambda i: target_options[i],
        key="phase2a_target_select",
    )

    st.divider()
    _render_target_detail(summary_df.iloc[int(selected_idx)], output_dir, show_outliers)

