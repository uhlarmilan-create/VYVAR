"""Aperture Photometry Lightcurves — Fáza 2A UI."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        return pd.read_csv(p, low_memory=False)
    return pd.DataFrame()


def _float_coord_row(row: pd.Series, *keys: str) -> float:
    for k in keys:
        if k not in row.index:
            continue
        v = row.get(k)
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        try:
            f = float(v)
            if math.isfinite(f):
                return f
        except (TypeError, ValueError):
            continue
    return 0.0


def _fmt_opt_num(v: Any, fmt: str, empty: str = "—") -> str:
    if v is None:
        return empty
    if isinstance(v, float) and not math.isfinite(v):
        return empty
    s = str(v).strip()
    if s.lower() in ("", "nan", "none"):
        return empty
    if s == "—":
        return empty
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return empty


# ---------------------------------------------------------------------------
# Per-target lightcurve + field map view
# ---------------------------------------------------------------------------


def _render_target_detail(
    target_row: pd.Series,
    output_dir: Path,
    show_outliers: bool,
    comp_df: pd.DataFrame | None = None,
) -> None:
    """Interaktívna krivka (Plotly z CSV), field map PNG, metriky, odkazy Vizier/VSX."""
    from photometry_phase2a import _normalize_gaia_id

    catalog_id = str(target_row.get("catalog_id", ""))
    vsx_name = str(target_row.get("vsx_name", catalog_id))
    ra_target = _float_coord_row(target_row, "ra_deg", "ra")
    dec_target = _float_coord_row(target_row, "dec_deg", "dec")

    at_path = output_dir / "active_targets.csv"
    if (ra_target == 0.0 and dec_target == 0.0) or (
        not math.isfinite(ra_target) or not math.isfinite(dec_target)
    ):
        if at_path.exists():
            try:
                at_df = pd.read_csv(at_path, low_memory=False)
                if "catalog_id" in at_df.columns:
                    cid_norm = _normalize_gaia_id(catalog_id)
                    at_df = at_df.copy()
                    at_df["_nid"] = at_df["catalog_id"].apply(_normalize_gaia_id)
                    hit = at_df[at_df["_nid"] == cid_norm]
                    if not hit.empty:
                        r0 = hit.iloc[0]
                        ra_target = _float_coord_row(r0, "ra_deg", "ra")
                        dec_target = _float_coord_row(r0, "dec_deg", "dec")
            except Exception:  # noqa: BLE001
                pass

    lc_dir = output_dir / "lightcurves"
    lc_csv = lc_dir / f"lightcurve_{catalog_id}.csv"

    col_lc, col_map = st.columns([3, 2])

    with col_lc:
        st.markdown(f"**Svetelná krivka — {vsx_name}**")
        if lc_csv.exists():
            lc_df = pd.read_csv(lc_csv, low_memory=False)
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
            else:
                st.info("V CSV chýbajú stĺpce bjd / mag_calib alebo súbor je prázdny.")
        else:
            st.info("Lightcurve CSV neexistuje. Spusti Fázu 2A.")

    with col_map:
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
    if rms is not None and pd.notna(rms):
        cols[0].metric("lc_rms", f"{float(rms):.4f}")
    if n_comp is not None and pd.notna(n_comp):
        cols[1].metric("good comp", int(n_comp))
    if ap is not None and pd.notna(ap):
        cols[2].metric("apertura", f"{float(ap):.1f}px")

    st.markdown("**Premenná hviezda**")
    vizier_url = (
        f"https://vizier.cds.unistra.fr/viz-bin/VizieR?"
        f"&-c={ra_target:.6f}{dec_target:+.6f}&-c.rs=5"
    )
    vsx_url = (
        f"https://www.aavso.org/vsx/index.php?view=search.top"
        f"&RA={ra_target:.5f}&Dec={dec_target:.5f}&radiusUnit=deg&radius=0.01"
    )
    st.markdown(
        f"**{vsx_name}** &nbsp; "
        f"[Vizier]({vizier_url}) &nbsp; "
        f"[VSX]({vsx_url})"
    )

    if comp_df is not None and not comp_df.empty and "target_catalog_id" in comp_df.columns:
        comp_work = comp_df.copy()
        comp_work["_tcid"] = comp_work["target_catalog_id"].apply(_normalize_gaia_id)
        target_comps = comp_work[comp_work["_tcid"] == _normalize_gaia_id(catalog_id)].copy()

        if not target_comps.empty:
            st.markdown("**Porovnávacie hviezdy**")
            header = "| # | mag | B-V | p2p RMS | Vizier |\n|:---|:---|:---|:---|:---|\n"
            body_lines: list[str] = []
            for i, (_, row) in enumerate(target_comps.iterrows(), 1):
                ra_c = _float_coord_row(row, "ra_deg", "ra")
                dec_c = _float_coord_row(row, "dec_deg", "dec")
                mag_c = row.get("mag")
                bv_c = row.get("b_v")
                if bv_c is None or (isinstance(bv_c, float) and not math.isfinite(bv_c)):
                    bv_c = row.get("bp_rp")
                rms_c = row.get("comp_rms")

                viz_c = (
                    f"https://vizier.cds.unistra.fr/viz-bin/VizieR?"
                    f"&-c={ra_c:.6f}{dec_c:+.6f}&-c.rs=2"
                )
                mag_str = _fmt_opt_num(mag_c, ".3f")
                bv_str = _fmt_opt_num(bv_c, ".3f")
                rms_str = _fmt_opt_num(rms_c, ".4f")
                body_lines.append(
                    f"| C{i:02d} | {mag_str} | {bv_str} | {rms_str} | [↗]({viz_c}) |\n"
                )
            st.markdown(header + "".join(body_lines), unsafe_allow_html=False)


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
                        cfg=cfg,
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
    comp_df = pd.DataFrame()
    if comp_csv is not None and Path(comp_csv).exists():
        comp_df = pd.read_csv(comp_csv, low_memory=False)

    if summary_df.empty:
        st.info("Zatiaľ žiadne výsledky.")
        return

    name_col = "vsx_name" if "vsx_name" in summary_df.columns else "catalog_id"
    fill_series = (
        summary_df["catalog_id"]
        if "catalog_id" in summary_df.columns
        else pd.Series([""] * len(summary_df), index=summary_df.index)
    )
    names = summary_df[name_col].fillna(fill_series).astype(str).tolist()
    selected = st.selectbox(
        "Vyber premennú hviezdu:",
        names,
        key="phase2a_target_select",
    )
    mask = summary_df[name_col].fillna(fill_series).astype(str) == str(selected)
    if not bool(mask.any()):
        st.warning("Vybraný target sa nenašiel v summary.")
        return
    target_row = summary_df.loc[mask].iloc[0]

    show_outliers = st.toggle(
        "Zobraziť outlier a saturated body",
        value=True,
        key="phase2a_show_outliers",
    )

    _render_target_detail(target_row, output_dir, show_outliers, comp_df=comp_df)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Svetelné krivky", len(summary_df))
    if "lc_rms" in summary_df.columns:
        c2.metric("Median lc_rms", f"{summary_df['lc_rms'].median():.4f}")
        good = int((summary_df["lc_rms"] < 0.05).sum())
        c3.metric("RMS < 0.05 mag", good)
    if "n_good_comp" in summary_df.columns:
        c4.metric("Avg good comp", f"{summary_df['n_good_comp'].mean():.1f}")
