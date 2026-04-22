"""Aperture Photometry Lightcurves — Fáza 2A UI."""

from __future__ import annotations

import html
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from config import AppConfig
    from pipeline import AstroPipeline

from platesolve_ui_paths import default_bundle_dir


# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------


def _find_phase2a_paths(
    cfg: "AppConfig",
    draft_id: int | None,
    *,
    draft_dir_override: Path | None = None,
) -> dict[str, Path | None]:
    """Return all available filter/setup groups for Phase 2A.

    Returns:
        Dict keyed by setup_name (e.g. ``R_60_1``) with a nested dict of paths.
    """
    if draft_id is None and draft_dir_override is None:
        return {}
    try:
        archive = Path(cfg.archive_root)
        if draft_dir_override is not None and draft_dir_override.is_dir():
            draft_dir = draft_dir_override.resolve()
        elif draft_id is not None:
            draft_dir = archive / "Drafts" / f"draft_{int(draft_id):06d}"
        else:
            return {}
        ps_dir = draft_dir / "platesolve"
        aligned_root = draft_dir / "detrended_aligned" / "lights"

        result: dict[str, dict[str, Path | None]] = {}
        if not ps_dir.exists():
            return {}

        for subdir in sorted(ps_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if not (subdir / "per_frame_catalog_index.csv").exists():
                continue
            setup_name = subdir.name  # napr. "R_60_1"

            per_frame_dir = (aligned_root / setup_name) if (aligned_root / setup_name).exists() else None
            photometry_dir = subdir / "photometry"

            result[str(setup_name)] = {
                "setup_name": subdir,
                "masterstar_fits": subdir / "MASTERSTAR.fits",
                "active_targets_csv": photometry_dir / "active_targets.csv",
                "comparison_stars_csv": photometry_dir / "comparison_stars_per_target.csv",
                "per_frame_csv_dir": per_frame_dir,
                "detrended_aligned_dir": per_frame_dir,
                "output_dir": photometry_dir,
                "photometry_dir": photometry_dir,
                "obs_group_dir": subdir,
            }
        return result  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return {}


def _load_fwhm(masterstar_fits: Path | None) -> float:
    if masterstar_fits is None or not masterstar_fits.is_file():
        return 3.7
    try:
        from astropy.io import fits as astrofits

        with astrofits.open(masterstar_fits, memmap=False) as hdul:
            for key in ("VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN", "VY_FWHM"):
                v = hdul[0].header.get(key)
                if v is not None:
                    fv = float(v)
                    if 0.5 < fv < 30.0:
                        return round(fv, 3)
    except Exception:  # noqa: BLE001
        pass
    return 3.7


def _fallback_masterstar_fits(
    cfg: "AppConfig",
    draft_id: int | None,
    *,
    draft_dir_override: Path | None = None,
) -> Path | None:
    if draft_id is None and draft_dir_override is None:
        return None
    if draft_dir_override is not None:
        ps = draft_dir_override / "platesolve"
    else:
        ps = Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft_id):06d}" / "platesolve"
    d = default_bundle_dir(ps)
    if d is None:
        return None
    p = d / "MASTERSTAR.fits"
    return p if p.is_file() else None


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
    *,
    show_detrended: bool = True,
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

            y_col = "mag_calib" if show_detrended else "mag_calib_raw"
            y_label = (
                "mag_calib (detrend)" if show_detrended else "mag_calib_raw (bez detrend)"
            )

            if (
                not lc_df.empty
                and "bjd" in lc_df.columns
                and y_col in lc_df.columns
            ):
                try:
                    import plotly.graph_objects as go
                except Exception:  # noqa: BLE001
                    go = None  # type: ignore[assignment]

                if go is not None:
                    fig = go.Figure()
                    # Svetlé pozadie + výrazné farby bodov (čitateľné aj v tmavom Streamlit)
                    flag_colors_plotly = {
                        "normal": "#2563eb",
                        "outlier_hi": "#ea580c",
                        "outlier_lo": "#9333ea",
                        "saturated": "#64748b",
                        "no_data": "#94a3b8",
                    }

                    if "flag" not in lc_df.columns:
                        lc_df = lc_df.assign(flag="normal")

                    for flag, color in flag_colors_plotly.items():
                        sub = lc_df[lc_df["flag"] == flag].dropna(
                            subset=["bjd", y_col]
                        )
                        if sub.empty:
                            continue
                        err = (
                            sub["err"].fillna(0).tolist()
                            if "err" in sub.columns
                            else None
                        )
                        err_kwargs: dict = {}
                        if err is not None:
                            err_kwargs = dict(
                                array=err,
                                visible=True,
                                color=color,
                                thickness=1,
                                width=2,
                            )
                        fig.add_trace(
                            go.Scatter(
                                x=sub["bjd"],
                                y=sub[y_col],
                                error_y=err_kwargs if err_kwargs else None,
                                mode="markers",
                                marker=dict(color=color, size=7, line=dict(width=0.5, color="#ffffff")),
                                name=flag,
                            )
                        )

                    _axis_title = dict(font=dict(color="#000000", size=13))
                    fig.update_layout(
                        paper_bgcolor="#f1f5f9",
                        plot_bgcolor="#ffffff",
                        font=dict(color="#000000", size=12),
                        yaxis=dict(
                            autorange="reversed",
                            title=dict(text=y_label, **_axis_title),
                            tickfont=dict(color="#000000", size=12),
                            gridcolor="#cbd5e1",
                            zerolinecolor="#94a3b8",
                        ),
                        xaxis=dict(
                            title=dict(text="BJD (TDB)", **_axis_title),
                            tickfont=dict(color="#000000", size=12),
                            gridcolor="#e2e8f0",
                        ),
                        height=350,
                        margin=dict(l=40, r=20, t=36, b=40),
                        legend=dict(
                            orientation="h",
                            y=1.12,
                            font=dict(size=11, color="#000000"),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Interaktívny graf nedostupný (plotly nie je nainštalovaný).")
            else:
                st.info(
                    f"V CSV chýbajú stĺpce bjd / {y_col} alebo súbor je prázdny."
                )
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
        cols[1].caption("good + suspect (ensemble)")
    if ap is not None and pd.notna(ap):
        cols[2].metric("apertura", f"{float(ap):.1f}px")

    _am_d = target_row.get("am_detrended")
    if _am_d is not None and pd.notna(_am_d):
        _am_on = str(_am_d).strip().lower() in ("true", "1", "yes")
        if not _am_on:
            st.caption("bez detrend (signál zachovaný)")

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
            cq_path = lc_dir / f"comp_quality_{catalog_id}.json"
            quality_by_cid: dict[str, str] = {}
            if cq_path.exists():
                try:
                    quality_by_cid = json.loads(cq_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    quality_by_cid = {}

            def _row_bg(q: str) -> str:
                if q == "good":
                    return "background-color:rgba(34,197,94,0.35);"
                if q == "suspect":
                    return "background-color:rgba(234,179,8,0.28);"
                return ""

            def _tier_badge(v: object) -> str:
                s = str(v or "").strip()
                if not s:
                    return "—"
                key = s.split("_", 1)[0].upper()  # TIER1 / TIER2 / ...
                bg = {
                    "TIER1": "rgba(34,197,94,0.25)",   # green
                    "TIER2": "rgba(59,130,246,0.25)",  # blue
                    "TIER3": "rgba(234,179,8,0.25)",   # yellow
                    "TIER4": "rgba(239,68,68,0.25)",   # red
                }.get(key, "rgba(148,163,184,0.25)")
                fg = {
                    "TIER4": "rgba(127,29,29,1.0)",
                }.get(key, "rgba(15,23,42,1.0)")
                return (
                    "<span style=\"display:inline-block;padding:2px 6px;border-radius:999px;"
                    f"background-color:{bg};color:{fg};font-weight:600;font-size:0.85rem;\">"
                    f"{html.escape(key)}</span>"
                )

            rows_html: list[str] = []
            for i, (_, row) in enumerate(target_comps.iterrows(), 1):
                ra_c = _float_coord_row(row, "ra_deg", "ra")
                dec_c = _float_coord_row(row, "dec_deg", "dec")
                mag_c = row.get("mag")
                bv_c = row.get("b_v")
                if bv_c is None or (isinstance(bv_c, float) and not math.isfinite(bv_c)):
                    bv_c = row.get("bp_rp")
                rms_c = row.get("comp_rms")
                tier_c = row.get("comp_tier")
                cid_c = _normalize_gaia_id(row.get("catalog_id", ""))
                q = str(quality_by_cid.get(cid_c, "")).lower()
                stav = {"good": "good", "suspect": "suspect", "excluded": "excluded"}.get(
                    q, "—"
                )

                viz_c = (
                    f"https://vizier.cds.unistra.fr/viz-bin/VizieR?"
                    f"&-c={ra_c:.6f}{dec_c:+.6f}&-c.rs=2"
                )
                mag_str = _fmt_opt_num(mag_c, ".3f")
                bv_str = _fmt_opt_num(bv_c, ".3f")
                rms_str = _fmt_opt_num(rms_c, ".4f")
                bg = _row_bg(q)
                rows_html.append(
                    "<tr style=\""
                    + bg
                    + "\">"
                    f"<td>C{i:02d}</td>"
                    f"<td>{html.escape(mag_str)}</td>"
                    f"<td>{html.escape(bv_str)}</td>"
                    f"<td>{html.escape(rms_str)}</td>"
                    f"<td>{_tier_badge(tier_c)}</td>"
                    f"<td>{html.escape(stav)}</td>"
                    f"<td><a href=\"{html.escape(viz_c)}\" target=\"_blank\" rel=\"noopener noreferrer\">↗</a></td>"
                    "</tr>"
                )

            thead = (
                "<thead><tr>"
                "<th>#</th><th>mag</th><th>B-V</th><th>p2p RMS</th>"
                "<th>tier</th><th>stav</th><th>Vizier</th>"
                "</tr></thead>"
            )
            table_html = (
                "<table style=\"width:100%;border-collapse:collapse;font-size:0.95rem;\">"
                + thead
                + "<tbody>"
                + "".join(rows_html)
                + "</tbody></table>"
            )
            st.markdown(table_html, unsafe_allow_html=True)
            if not quality_by_cid:
                st.caption(
                    "Stav (good / suspect / excluded) sa zobrazí po ďalšom behu Fázy 2A "
                    "(súbor comp_quality_*.json)."
                )


# ---------------------------------------------------------------------------
# Hlavný render
# ---------------------------------------------------------------------------


def render_aperture_photometry(
    cfg: "AppConfig",
    draft_id: int | None,
    pipeline: "AstroPipeline",
    *,
    draft_dir_override: Path | None = None,
) -> None:
    """Hlavná funkcia pre Aperture Photometry tab."""
    _ = pipeline
    st.header("Aperture Photometry")
    st.caption("Fáza 2A: Aperturná fotometria premenných hviezd.")

    if draft_id is None and draft_dir_override is None:
        st.info("Žiadny aktívny draft. Načítaj draft vyššie alebo spusti VAR-STREM.")
        return

    all_setups = _find_phase2a_paths(cfg, draft_id, draft_dir_override=draft_dir_override)
    if not all_setups:
        st.warning("Nenájdené vstupné súbory.")
        return

    setup_options = list(all_setups.keys())
    selected_setup = st.selectbox(
        "Filter / skupina:",
        options=setup_options,
        key="phase2a_setup_select",
    )
    paths = all_setups.get(str(selected_setup)) or {}
    if not paths:
        st.warning("Vybraný setup nemá platné cesty.")
        return

    output_dir = paths.get("output_dir")
    ms_for_fwhm = paths.get("masterstar_fits")
    if not (isinstance(ms_for_fwhm, Path) and ms_for_fwhm.is_file()):
        ms_for_fwhm = _fallback_masterstar_fits(cfg, draft_id, draft_dir_override=draft_dir_override)
    fwhm_px = _load_fwhm(ms_for_fwhm)

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
        if len(setup_options) > 1:
            st.caption("Tlačidlo spustí Fázu **2A pre všetky** filtre / setupy naraz.")
    else:
        st.info("Fáza 2A ešte nebehala.")
        col1, col2 = st.columns(2)
        with col1:
            run_btn = st.button(
                "▶ Spustiť Fázu 2A", key="phase2a_run_first", type="primary"
            )
        with col2:
            st.caption(f"FWHM: {fwhm_px} px")
        if len(setup_options) > 1:
            st.caption("Tlačidlo spustí Fázu **2A pre všetky** filtre / setupy naraz.")

    if run_btn:
        from photometry_core import run_phase2a
        from vyvar_ui_status import vyvar_footer_idle, vyvar_footer_running

        try:
            vyvar_footer_running("Fáza 2A", "Štartujem aperturnú fotometriu (všetky filtre)…")

            def _p2a_ui(msg: str) -> None:
                vyvar_footer_running("Fáza 2A", msg)

            errors: list[str] = []
            last_result: dict | None = None
            n_ok = 0
            for nm in setup_options:
                p = all_setups.get(str(nm)) or {}
                ms_fits = p.get("masterstar_fits")
                pf_dir = p.get("per_frame_csv_dir")
                dt_dir = p.get("detrended_aligned_dir")
                out_d = p.get("output_dir")
                at_p = p.get("active_targets_csv")
                co_p = p.get("comparison_stars_csv")
                ms_ff = ms_fits
                if not (isinstance(ms_ff, Path) and ms_ff.is_file()):
                    ms_ff = _fallback_masterstar_fits(cfg, draft_id, draft_dir_override=draft_dir_override)
                _fw = _load_fwhm(ms_ff if isinstance(ms_ff, Path) else None)

                missing: list[str] = []
                if ms_ff is None or not Path(ms_ff).exists():
                    missing.append("MASTERSTAR.fits")
                if pf_dir is None or not pf_dir.exists():
                    missing.append("per-frame CSV adresár")
                if dt_dir is None or not dt_dir.exists():
                    missing.append("detrended_aligned adresár")
                if at_p is None or not at_p.exists() or co_p is None or not co_p.exists():
                    missing.append("Fáza 0+1 výstupy (active_targets / comparison)")

                if missing:
                    errors.append(f"{nm}: {', '.join(missing)}")
                    continue

                try:
                    _p2a_ui(f"Fáza 2A: {nm} …")
                    last_result = run_phase2a(
                        masterstar_fits_path=Path(ms_ff),
                        active_targets_csv=Path(at_p),
                        comparison_stars_csv=Path(co_p),
                        per_frame_csv_dir=Path(pf_dir),
                        detrended_aligned_dir=Path(dt_dir),
                        output_dir=Path(out_d),
                        fwhm_px=_fw,
                        cfg=cfg,
                        progress_cb=_p2a_ui,
                    )
                    n_ok += 1
                except Exception as exc_nm:  # noqa: BLE001
                    errors.append(f"{nm}: {exc_nm}")
                    logging.exception("Fáza 2A zlyhala pre %s", nm)

            if n_ok == len(setup_options) and not errors and last_result is not None:
                st.success(
                    f"✅ Hotovo pre všetky filtre ({len(setup_options)}): "
                    f"{last_result['n_lightcurves']} kriviek z {last_result['n_frames']} snímok (posledný setup)."
                )
            elif n_ok > 0 and last_result is not None:
                st.success(
                    f"✅ Fáza 2A: {n_ok}/{len(setup_options)} setupov OK "
                    f"(posledný: {last_result['n_lightcurves']} kriviek)."
                )
            if errors:
                (st.error if n_ok == 0 else st.warning)(
                    "Problémy pri niektorých filtroch:\n" + "\n".join(errors)
                )
            if n_ok:
                st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Chyba: {exc}")
            logging.exception("Fáza 2A zlyhala")
        finally:
            vyvar_footer_idle()
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
    try:
        catalog_id = str(target_row.get("catalog_id", ""))
    except Exception:
        catalog_id = ""

    col1, col2 = st.columns([1, 3])
    with col1:
        show_detrended = st.toggle(
            "Airmass detrend",
            value=True,
            key="toggle_am_detrend",
        )
    with col2:
        show_outliers = st.toggle(
            "Zobraziť outlier a saturated body",
            value=True,
            key="phase2a_show_outliers",
        )

    show_all_filters = st.checkbox(
        "Zobraziť všetky filtre v jednom grafe",
        value=False,
        key="phase2a_show_all_filters",
    )

    if show_all_filters and catalog_id:
        try:
            import plotly.graph_objects as go  # type: ignore

            fig = go.Figure()
            FILTER_COLORS = {"R": "red", "V": "green", "B": "blue", "I": "darkred"}

            for setup_name, p in all_setups.items():
                obs_dir = p.get("obs_group_dir")
                if obs_dir is None:
                    continue
                lc_dir = Path(obs_dir) / "photometry" / "lightcurves"
                lc_csv = lc_dir / f"lightcurve_{catalog_id}.csv"
                if not lc_csv.exists():
                    continue
                lc_df = pd.read_csv(lc_csv, low_memory=False)

                filter_letter = setup_name[0] if setup_name else "?"
                color = FILTER_COLORS.get(filter_letter, "gray")

                x_col = "bjd_tdb_mid" if "bjd_tdb_mid" in lc_df.columns else ("bjd_tdb" if "bjd_tdb" in lc_df.columns else lc_df.columns[0])
                y_col = "mag_calib" if "mag_calib" in lc_df.columns else ("mag_calib_raw" if "mag_calib_raw" in lc_df.columns else lc_df.columns[1])
                err_col = "mag_err" if "mag_err" in lc_df.columns else None

                fig.add_trace(
                    go.Scatter(
                        x=lc_df[x_col],
                        y=lc_df[y_col],
                        mode="markers+lines",
                        name=str(setup_name),
                        marker=dict(color=color, size=4),
                        line=dict(color=color, width=0.5),
                        error_y=dict(
                            type="data",
                            array=(lc_df[err_col].tolist() if err_col is not None else None),
                            visible=bool(err_col is not None),
                        ),
                    )
                )

            fig.update_layout(
                title=f"Svetelné krivky — {catalog_id}",
                xaxis_title="BJD (TDB)",
                yaxis_title="mag (calib)",
                yaxis_autorange="reversed",
                legend_title="Filter",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Overlay graf zlyhal: {exc}")
            _render_target_detail(
                target_row,
                output_dir,
                show_outliers,
                comp_df=comp_df,
                show_detrended=show_detrended,
            )
    else:
        _render_target_detail(
            target_row,
            output_dir,
            show_outliers,
            comp_df=comp_df,
            show_detrended=show_detrended,
        )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Svetelné krivky", len(summary_df))
    if "lc_rms" in summary_df.columns:
        c2.metric("Median lc_rms", f"{summary_df['lc_rms'].median():.4f}")
        good = int((summary_df["lc_rms"] < 0.05).sum())
        c3.metric("RMS < 0.05 mag", good)
    if "n_good_comp" in summary_df.columns:
        c4.metric("Avg good comp", f"{summary_df['n_good_comp'].mean():.1f}")
