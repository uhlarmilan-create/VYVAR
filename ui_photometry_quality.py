"""Streamlit dashboard: Photometry Quality Diagnostic (MASTERSTAR + per-frame catalogs)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _platesolve_dir_with_comparison(archive: Path) -> Path:
    ps_root = archive / "platesolve"
    if not ps_root.is_dir():
        return ps_root
    for sub in sorted(ps_root.iterdir()):
        if sub.is_dir() and (sub / "comparison_stars.csv").is_file():
            return sub
    return ps_root


@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Načítavam MASTERSTAR.fits…")
def _read_fits_image_cached(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    from astropy.io import fits

    p = Path(path)
    with fits.open(p, memmap=False) as hdul:
        raw = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = dict(hdul[0].header)
    return raw, hdr


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False)
    if str(s.dtype).lower().startswith("bool"):
        return s.fillna(False).astype(bool)
    if str(s.dtype).lower().startswith("int"):
        return s.fillna(0).astype(int) != 0
    if str(s.dtype).lower().startswith("float"):
        return s.fillna(0.0).astype(float) != 0.0
    # strings: "true"/"false"/"1"
    return s.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _catalog_id_key(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series([], dtype="string")
    out = s.copy()
    out = out.astype("string").str.strip()
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def render_photometry_quality_diagnostic(*, pipeline: Any, draft_id: int | None) -> None:
    st.subheader("Photometry Quality Diagnostic")
    st.caption("MASTERSTAR zóny (linear / saturated / noisy), histogramy a per-frame prehľad. Grafy sú interaktívne (Plotly).")

    _ = (pipeline, draft_id)
    archive_path = st.text_input(
        "Cesta k archívu (draft)",
        placeholder=r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000180",
        help="Zadaj cestu k draftu ručne",
    )
    if (not archive_path) or (not os.path.isdir(str(archive_path))):
        st.info("Zadaj platnú cestu k draftu.")
        return
    ap = Path(str(archive_path)).expanduser().resolve()

    ps_root = ap / "platesolve"
    ps_setup = _platesolve_dir_with_comparison(ap)
    ms_path = ps_root / "masterstars_full_match.csv"
    ms_fits_path = ps_root / "MASTERSTAR.fits"
    cs_path = ps_setup / "comparison_stars.csv"
    idx_path = ps_setup / "per_frame_catalog_index.csv"
    per_frame_root = ap / "detrended_aligned" / "lights"

    if not ms_path.is_file():
        st.warning(f"Nenašiel som `{ms_path}`. Spusti **MAKE MASTERSTAR** v záložke VAR-STREM.")
        return

    ms = _read_csv_cached(str(ms_path))
    if ms.empty:
        st.warning("`masterstars_full_match.csv` je prázdny.")
        return

    # --- Controls (shared) ---
    colc1, colc2, colc3 = st.columns([1.2, 1.2, 2.2])
    with colc1:
        show_linear = st.checkbox("Show linear", value=True)
        show_saturated = st.checkbox("Show saturated", value=True)
    with colc2:
        show_noisy = st.checkbox("Show noisy", value=True)
        show_comp = st.checkbox("Show comparison stars", value=True)
    with colc3:
        sat0 = float(pd.to_numeric(ms.get("saturate_limit_adu_85pct", pd.Series([np.nan])), errors="coerce").dropna().median()) if "saturate_limit_adu_85pct" in ms.columns else float("nan")
        sat_default = sat0 if math.isfinite(sat0) else float(pd.to_numeric(ms.get("saturate_limit_adu", pd.Series([65535.0])), errors="coerce").dropna().median() * 0.85) if "saturate_limit_adu" in ms.columns else 55000.0
        sat_thresh = st.slider(
            "Saturate threshold (ADU) — live test",
            min_value=0.0,
            max_value=float(max(1.0, float(pd.to_numeric(ms.get("peak_max_adu", pd.Series([sat_default])), errors="coerce").max() or sat_default) * 1.1)),
            value=float(max(0.0, sat_default)),
            step=100.0,
        )

    # --- Section 1: MASTERSTAR saturation visualization ---
    st.markdown("## MASTERSTAR saturácia — vizualizácia")

    fits_path = st.text_input(
        "FITS súbor pre pozadie (voliteľné)",
        placeholder=r"C:\...\proc_TYC_4062-1642-1_Light_082.fits",
        help="Nechaj prázdne pre MASTERSTAR.fits, alebo zadaj cestu k inému FITS",
        key="vyvar_photqa_bg_fits_path",
    ).strip()
    bg_fits_path = Path(fits_path).expanduser().resolve() if fits_path else ms_fits_path
    if not bg_fits_path.is_file():
        st.warning(
            f"Chýba FITS pre pozadie: `{bg_fits_path}`. "
            f"Nechaj pole prázdne pre `{ms_fits_path.name}` alebo zadaj platnú cestu."
        )
        return

    for c in ("x", "y"):
        if c not in ms.columns:
            st.error(f"`masterstars_full_match.csv` chýba stĺpec `{c}`.")
            return
    ms_xy = ms.copy()
    ms_xy["x"] = pd.to_numeric(ms_xy["x"], errors="coerce")
    ms_xy["y"] = pd.to_numeric(ms_xy["y"], errors="coerce")
    ms_xy = ms_xy[ms_xy["x"].notna() & ms_xy["y"].notna()].copy()

    is_noisy = _bool_series(ms_xy, "is_noisy")
    is_sat = _bool_series(ms_xy, "is_saturated") | _bool_series(ms_xy, "likely_saturated")
    # live override: treat any star with peak_max_adu > sat_thresh as saturated
    if "peak_max_adu" in ms_xy.columns:
        pk = pd.to_numeric(ms_xy["peak_max_adu"], errors="coerce")
        is_sat = is_sat | (pk.notna() & (pk > float(sat_thresh)))
    is_usable = _bool_series(ms_xy, "is_usable")
    is_linear = is_usable & (~is_sat) & (~is_noisy)

    comp_ids: set[str] = set()
    if cs_path.is_file():
        cs = _read_csv_cached(str(cs_path))
        if not cs.empty and "catalog_id" in cs.columns:
            comp_ids = set(_catalog_id_key(cs["catalog_id"]).dropna().astype(str).tolist())
    ms_xy["_cid_key"] = _catalog_id_key(ms_xy.get("catalog_id", pd.Series([pd.NA] * len(ms_xy)))).astype("string")
    is_comp = ms_xy["_cid_key"].isin(list(comp_ids)) if comp_ids else pd.Series([False] * len(ms_xy), index=ms_xy.index)

    n_linear = int(is_linear.sum())
    n_sat = int(is_sat.sum())
    n_noisy = int(is_noisy.sum())
    n_comp = int(is_comp.sum())
    st.write(f"Counts: **linear={n_linear}**, **saturated={n_sat}**, **noisy={n_noisy}**, **comparison={n_comp}**")

    from masterstar_qa_plot import downsample_array_2d, percentile_stretch_rgb
    from PIL import Image, ImageDraw

    raw_ms, hdr_ms = _read_fits_image_cached(str(bg_fits_path))
    disp, scx, scy = downsample_array_2d(raw_ms, 1600)
    rgb = percentile_stretch_rgb(disp, 1.0, 99.0)
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(im)

    # Coordinates scaled to the display image
    xs = ms_xy["x"].to_numpy(dtype=float) * float(scx)
    ys = ms_xy["y"].to_numpy(dtype=float) * float(scy)

    r_lin = 5.0
    r_sat = 6.0
    r_noi = 2.0
    s_comp = 5.0

    def _draw_circles(mask: pd.Series, *, outline: tuple[int, int, int], r: float, width: int = 2) -> None:
        idxs = np.nonzero(mask.to_numpy(dtype=bool))[0]
        for i in idxs:
            x0, y0 = float(xs[i]), float(ys[i])
            if not (math.isfinite(x0) and math.isfinite(y0)):
                continue
            draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), outline=outline, width=width)

    def _draw_dots(mask: pd.Series, *, fill: tuple[int, int, int], r: float) -> None:
        idxs = np.nonzero(mask.to_numpy(dtype=bool))[0]
        for i in idxs:
            x0, y0 = float(xs[i]), float(ys[i])
            if not (math.isfinite(x0) and math.isfinite(y0)):
                continue
            draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), fill=fill, outline=fill, width=1)

    def _draw_squares(mask: pd.Series, *, outline: tuple[int, int, int], s: float, width: int = 2) -> None:
        idxs = np.nonzero(mask.to_numpy(dtype=bool))[0]
        for i in idxs:
            x0, y0 = float(xs[i]), float(ys[i])
            if not (math.isfinite(x0) and math.isfinite(y0)):
                continue
            draw.rectangle((x0 - s, y0 - s, x0 + s, y0 + s), outline=outline, width=width)

    # Draw order: noisy (background), saturated, linear, comp (top)
    if show_noisy:
        _draw_dots(is_noisy & (~is_comp), fill=(140, 140, 140), r=r_noi)
    if show_saturated:
        _draw_circles(is_sat & (~is_comp), outline=(220, 60, 60), r=r_sat, width=2)
    if show_linear:
        _draw_circles(is_linear & (~is_comp), outline=(30, 170, 60), r=r_lin, width=2)
    if show_comp:
        _draw_squares(is_comp, outline=(60, 120, 255), s=s_comp, width=2)

    st.image(
        im,
        caption=f"{bg_fits_path.name} + zones overlay (scx×{scx:.4f}, scy×{scy:.4f})",
        use_container_width=True,
    )

    # --- Section 2: Histogram peak_max_adu ---
    st.markdown("## Histogram peak_max_adu")
    if "peak_max_adu" not in ms_xy.columns:
        st.info("`peak_max_adu` nie je v MASTERSTAR CSV — histogram preskočený.")
    else:
        pk = pd.to_numeric(ms_xy["peak_max_adu"], errors="coerce")
        fig_h = go.Figure()
        def _hist(mask: pd.Series, name: str, color: str) -> None:
            vals = pk[mask & pk.notna()]
            if len(vals) == 0:
                return
            fig_h.add_trace(go.Histogram(x=vals, name=name, marker_color=color, opacity=0.65, nbinsx=120))
        _hist(is_linear, "linear", "#2ca02c")
        _hist(is_sat, "saturated", "#d62728")
        _hist(is_noisy, "noisy", "#7f7f7f")
        fig_h.add_vline(x=float(sat_thresh), line_width=2, line_dash="dash", line_color="black")
        fig_h.update_layout(barmode="overlay", height=380, xaxis_title="peak_max_adu", yaxis_title="count")
        st.plotly_chart(fig_h, use_container_width=True)

    # --- Section 3: Per-frame CSV overview ---
    st.markdown("## Per-frame CSV prehľad")
    if not per_frame_root.is_dir():
        st.info(f"Nenašiel som `{per_frame_root}`.")
        return

    low_match_thr = st.number_input("Highlight frames with n_matched < ...", min_value=0, max_value=50000, value=500, step=50)

    idx_df = None
    if idx_path.is_file():
        try:
            idx_df = _read_csv_cached(str(idx_path))
        except Exception:  # noqa: BLE001
            idx_df = None

    # Load per-frame CSVs by reading per_frame_catalog_index.csv when available (fast).
    rows: list[dict[str, Any]] = []
    if idx_df is not None and not idx_df.empty and "file" in idx_df.columns:
        for _, r in idx_df.iterrows():
            rows.append(dict(r))
    else:
        # fallback: enumerate CSV next to FITS (sidecar mode)
        for p in sorted(per_frame_root.rglob("*.csv")):
            if p.name in {"per_frame_catalog_index.csv", "comparison_stars.csv", "variable_targets.csv"}:
                continue
            rows.append({"file": p.stem + ".fits", "csv": str(p)})

    # Build frame metrics table (try: read sidecar CSV to compute usable/saturated counts)
    frame_metrics: list[dict[str, Any]] = []
    for r in rows:
        fp = str(r.get("file") or "")
        csvp = str(r.get("csv") or "")
        if not csvp:
            # infer sidecar
            stem = Path(fp).stem
            cand = list(per_frame_root.rglob(stem + ".csv"))
            csvp = str(cand[0]) if cand else ""
        if not csvp or not Path(csvp).is_file():
            continue
        try:
            dfi = _read_csv_cached(csvp)
        except Exception:
            continue
        if dfi.empty:
            continue
        cid = _catalog_id_key(dfi.get("catalog_id", pd.Series([pd.NA] * len(dfi)))).notna()
        n_matched = int(cid.sum())
        n_sat_i = int(_bool_series(dfi, "is_saturated").sum()) if "is_saturated" in dfi.columns else int(_bool_series(dfi, "likely_saturated").sum()) if "likely_saturated" in dfi.columns else 0
        n_use_i = int(_bool_series(dfi, "is_usable").sum()) if "is_usable" in dfi.columns else 0
        # time: prefer jd_mid
        t = pd.to_numeric(dfi.get("jd_mid", pd.Series([np.nan] * len(dfi))), errors="coerce")
        t0 = float(t.dropna().iloc[0]) if t.notna().any() else np.nan
        frame_metrics.append(
            {
                "file": fp,
                "csv": csvp,
                "jd_mid": t0,
                "n_rows": int(len(dfi)),
                "n_matched": n_matched,
                "n_saturated": n_sat_i,
                "n_usable": n_use_i,
                "match_mode": str(r.get("catalog_match_mode") or r.get("catalog_match_mode", "")) or str(r.get("catalog_match_mode") or ""),
                "status": str(r.get("status") or "ok"),
            }
        )

    if not frame_metrics:
        st.info("Nenašli sa per-frame CSV alebo sú prázdne.")
        return
    fm = pd.DataFrame(frame_metrics)
    fm = fm.sort_values("jd_mid", ascending=True, na_position="last")
    st.dataframe(fm[["file", "jd_mid", "n_matched", "n_saturated", "n_usable", "status"]], use_container_width=True, height=320)

    fig_t = go.Figure()
    bad = fm["n_matched"] < int(low_match_thr)
    fig_t.add_trace(go.Scatter(x=fm["jd_mid"], y=fm["n_matched"], mode="lines+markers", name="n_matched"))
    if bool(bad.any()):
        fig_t.add_trace(
            go.Scatter(
                x=fm.loc[bad, "jd_mid"],
                y=fm.loc[bad, "n_matched"],
                mode="markers",
                name=f"n_matched < {int(low_match_thr)}",
                marker=dict(color="#d62728", size=9),
            )
        )
    fig_t.update_layout(height=380, xaxis_title="JD mid", yaxis_title="n_matched")
    st.plotly_chart(fig_t, use_container_width=True)

    # --- Section 4: peak_dao vs peak_max_adu ---
    st.markdown("## peak_dao vs peak_max_adu")
    if "peak_dao" not in ms_xy.columns or "peak_max_adu" not in ms_xy.columns:
        st.info("Chýba `peak_dao` alebo `peak_max_adu` v MASTERSTAR CSV.")
        return
    xpk = pd.to_numeric(ms_xy["peak_dao"], errors="coerce")
    ypk = pd.to_numeric(ms_xy["peak_max_adu"], errors="coerce")
    mask = xpk.notna() & ypk.notna()
    xpk = xpk[mask]
    ypk = ypk[mask]
    sat_m = (is_sat[mask] if isinstance(is_sat, pd.Series) else pd.Series([False] * len(xpk)))
    fig_p = go.Figure()
    fig_p.add_trace(
        go.Scattergl(
            x=xpk[~sat_m],
            y=ypk[~sat_m],
            mode="markers",
            name="non-saturated",
            marker=dict(color="#2ca02c", size=4, opacity=0.6),
        )
    )
    fig_p.add_trace(
        go.Scattergl(
            x=xpk[sat_m],
            y=ypk[sat_m],
            mode="markers",
            name="saturated",
            marker=dict(color="#d62728", size=5, opacity=0.7),
        )
    )
    lo = float(min(xpk.min(), ypk.min()))
    hi = float(max(xpk.max(), ypk.max()))
    if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
        fig_p.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1", line=dict(color="black", dash="dash")))
    fig_p.update_layout(height=520, xaxis_title="peak_dao", yaxis_title="peak_max_adu")
    st.plotly_chart(fig_p, use_container_width=True)

