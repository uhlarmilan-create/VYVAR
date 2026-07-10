"""Streamlit tab: field Hertzsprung–Russell diagram (HRD) from masterstars + Gaia."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import AppConfig

logger = logging.getLogger(__name__)


def render_hrd_tab(photometry_dir: Path, cfg: "AppConfig | None") -> None:
    """Interactive HRD tab (Plotly)."""
    import numpy as np
    import pandas as pd
    import streamlit as st

    try:
        import plotly.graph_objects as go
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Plotly is not available: {exc}")
        return

    from hrd_analysis import (
        HRD_CAPTION,
        HRD_EMPTY_FIELD_MSG,
        _f,
        annotate_field_image,
        build_hrd_dataframe,
        ensure_clean_field_background_png,
        get_top_interesting_stars,
        hrd_parallax_params_from_cfg,
    )

    st.subheader("Field Hertzsprung–Russell diagram")

    photometry_dir = Path(photometry_dir)
    ms_csv = photometry_dir / "masterstars_full_match.csv"
    if not ms_csv.exists():
        candidates = list(photometry_dir.parent.rglob("masterstars_full_match.csv"))
        if candidates:
            ms_csv = candidates[0]
        else:
            st.warning("**masterstars_full_match.csv** not found (expected in `platesolve/<setup>/`).")
            return

    gdb = Path("")
    if cfg is not None:
        gdb = Path(str(getattr(cfg, "gaia_db_path", "") or "").strip())
    if not gdb.is_file():
        st.warning("Set **Gaia DB path** in Settings (`gaia_db_path`) — without it, teff/log g/parallax are not filled in.")
        gdb = Path("")

    gaia_for_build = gdb if gdb.is_file() else Path("/nonexistent/vyvar_gaia_placeholder.db")
    pmin, psnr = hrd_parallax_params_from_cfg(cfg)
    with st.spinner("Loading data…"):
        try:
            hrd_df = build_hrd_dataframe(
                ms_csv, gaia_for_build, parallax_min_mas=pmin, parallax_snr_min=psnr
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error loading HRD: {exc}")
            logger.exception("render_hrd_tab build_hrd_dataframe")
            return

    if hrd_df.empty:
        st.info("Empty star catalog.")
        return

    platesolve_dir = photometry_dir.parent
    obs_group = platesolve_dir.name
    enrich_cache = platesolve_dir / "_hrd_cache" / "hrd_enrich.json"
    top_stars = get_top_interesting_stars(hrd_df, cfg=cfg, cache_path=enrich_cache)
    reliable = hrd_df[hrd_df["hrd_reliable"] == True]  # noqa: E712
    unreliable = hrd_df[hrd_df["hrd_reliable"] != True]

    st.caption(
        f"Stars on frame (DAO detected): **{len(hrd_df)}** | Reliable parallax: **{len(reliable)}** | "
        f"Apparent magnitude only (no M_G): **{len(unreliable)}**"
    )

    fig = go.Figure()

    if not unreliable.empty:
        gapp = pd.to_numeric(unreliable.get("phot_g_mean_mag"), errors="coerce")
        bpr = pd.to_numeric(unreliable.get("bp_rp"), errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=bpr,
                y=gapp,
                mode="markers",
                marker=dict(size=4, color="gray", opacity=0.35),
                name="No/low-quality parallax (apparent G)",
                customdata=unreliable[["catalog_id"]].astype(str).values,
                hovertemplate=(
                    "ID: %{customdata[0]}<br>"
                    "BP-RP: %{x:.3f}<br>"
                    "G: %{y:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    if not reliable.empty:
        teff_h = pd.to_numeric(reliable.get("teff_gspphot"), errors="coerce").fillna(0.0)
        logg_h = pd.to_numeric(reliable.get("logg_gspphot"), errors="coerce").fillna(0.0)
        cd = np.stack(
            [reliable["catalog_id"].astype(str).values, teff_h.values, logg_h.values],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(reliable["bp_rp"], errors="coerce"),
                y=pd.to_numeric(reliable["abs_mag_g"], errors="coerce"),
                mode="markers",
                marker=dict(
                    size=6,
                    color=pd.to_numeric(reliable["bp_rp"], errors="coerce"),
                    colorscale="RdYlBu_r",
                    cmin=-0.5,
                    cmax=4.0,
                    colorbar=dict(title="BP−RP"),
                    opacity=0.75,
                ),
                name="Reliable (M_G)",
                customdata=cd,
                hovertemplate=(
                    "ID: %{customdata[0]}<br>"
                    "BP-RP: %{x:.3f}<br>"
                    "M_G: %{y:.2f}<br>"
                    "Teff: %{customdata[1]:.0f} K<br>"
                    "log g: %{customdata[2]:.2f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    cat_colors = {
        "Very hot": "blue",
        "Hot luminous": "cyan",
        "Very cool": "red",
        "White dwarf": "lightblue",
        "Red giant": "orange",
        "Red supergiant": "darkorange",
        "Binary candidate": "lime",
    }

    is_empty = bool(top_stars.get("_empty_field", pd.Series([False])).any()) if not top_stars.empty else True
    plot_stars = top_stars[~top_stars.get("_empty_field", pd.Series(False)).astype(bool)] if not top_stars.empty else top_stars

    if not plot_stars.empty:
        for cat, grp in plot_stars.groupby("category"):
            xs: list[float] = []
            ys: list[float] = []
            labels: list[str] = []
            cids: list[str] = []
            for _, star in grp.iterrows():
                cid = str(star.get("catalog_id", "")).strip()
                match = hrd_df[hrd_df["catalog_id"].astype(str) == cid]
                if match.empty:
                    continue
                r = match.iloc[0]
                bprp = _f(r.get("bp_rp"))
                if bool(r.get("hrd_reliable")):
                    y_val = _f(r.get("abs_mag_g"))
                else:
                    y_val = _f(r.get("phot_g_mean_mag"))
                if bprp is None or y_val is None:
                    continue
                xs.append(float(bprp))
                ys.append(float(y_val))
                labels.append(str(cat).split("/")[0].split()[0])
                cids.append(cid)
            if not xs:
                continue
            col = "yellow"
            for key, ccol in cat_colors.items():
                if str(cat).startswith(key):
                    col = ccol
                    break
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    marker=dict(size=14, color=col, symbol="circle-open", line=dict(width=2, color="white")),
                    text=labels,
                    textposition="top right",
                    textfont=dict(size=9, color=col),
                    name=str(cat),
                    customdata=np.stack([cids], axis=-1),
                    hovertemplate=(
                        f"<b>{cat}</b><br>"
                        "ID: %{customdata[0]}<br>"
                        "x: %{x:.3f}<br>"
                        "y: %{y:.2f}<br>"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_yaxes(autorange="reversed", title="M<sub>G</sub> [mag] / G [mag]")
    fig.update_xaxes(title="BP − RP [mag]")
    title = f"Field HRD -- {obs_group}" if obs_group else "Field Hertzsprung–Russell diagram"
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=600,
        legend=dict(font=dict(size=9)),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(HRD_CAPTION)

    if is_empty:
        st.info(HRD_EMPTY_FIELD_MSG)
    elif not top_stars.empty:
        st.subheader("Extreme objects")
        display_cols = [
            c
            for c in (
                "category",
                "ident",
                "catalog_id",
                "simbad_id",
                "mag_g",
                "abs_mag_g",
                "bp_rp",
                "teff",
                "logg",
                "src",
            )
            if c in top_stars.columns
        ]
        show = top_stars[display_cols]
        st.dataframe(show, width="stretch", hide_index=True)

        detail_cols = [
            c
            for c in (
                "category",
                "ident",
                "simbad_id",
                "sp_type_raw",
                "otype_raw",
                "ra_dec_sex",
                "dist_pc",
                "parallax_mas",
                "mag_g",
                "abs_mag_g",
                "bp_rp",
                "teff",
                "teff_source",
                "logg",
                "logg_source",
                "dsc_wd_p",
                "catalog_id",
                "x_px",
                "y_px",
                "src",
            )
            if c in top_stars.columns
        ]
        with st.expander("Extreme objects -- details"):
            st.dataframe(top_stars[detail_cols], width="stretch", hide_index=True)

    _cache = photometry_dir / "_hrd_cache"
    _cache.mkdir(parents=True, exist_ok=True)
    bg_png, from_fits = ensure_clean_field_background_png(platesolve_dir, photometry_dir, cache_dir=_cache)
    if bg_png is not None and not plot_stars.empty:
        try:
            nss_on = bool(getattr(cfg, "hrd_nss_category_enabled", False)) if cfg else False
            annotated = annotate_field_image(
                bg_png,
                plot_stars,
                hrd_df,
                platesolve_dir=platesolve_dir,
                nss_category_enabled=nss_on,
                png_from_fits=from_fits,
            )
            if annotated is not None:
                st.subheader("Field image — interesting stars")
                st.image(str(annotated), width="stretch")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Field annotation failed: {exc}")

    from hrd_colorfield import HRD_COLORFIELD_CAPTION, hrd_color_field_enabled, render_catalog_color_field

    if hrd_color_field_enabled(cfg):
        color_cache = _cache / "hrd_field_color.png"
        with st.expander("Catalog-color field (Gaia tinted)"):
            st.caption(HRD_COLORFIELD_CAPTION)
            try:
                color_png = render_catalog_color_field(
                    platesolve_dir, photometry_dir, cfg, color_cache
                )
                if color_png is not None and color_png.is_file():
                    st.image(str(color_png), width="stretch")
                else:
                    st.info("Catalog-color field not available for this setup.")
            except Exception as exc:  # noqa: BLE001
                st.caption(f"Catalog-color field failed: {exc}")
                logger.exception("render_hrd_tab catalog color field")
