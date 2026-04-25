"""Aperture Photometry Lightcurves — Fáza 2A UI."""

from __future__ import annotations

import html
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import streamlit as st
from urllib.parse import quote_plus

if TYPE_CHECKING:
    from config import AppConfig
    from pipeline import AstroPipeline

from jd_axis_format import jd_axis_title, jd_series_relative
from platesolve_ui_paths import default_bundle_dir


# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_read_csv(path_s: str) -> pd.DataFrame:
    return pd.read_csv(path_s, low_memory=False)


def _airmass_column(df: pd.DataFrame) -> str | None:
    for c in ("airmass", "air_mass", "AIRMASS", "am"):
        if c in df.columns:
            return c
    return None


def _ut_tick_labels_from_jd(jd_vals: "list[float]") -> list[str]:
    """Format JD-like values (BJD/HJD/JD) to UT HH:MM labels."""
    try:
        from astropy.time import Time
    except Exception:  # noqa: BLE001
        # Fallback: HH:MM from fractional day (approx; ignores leap seconds).
        out_f: list[str] = []
        for jd in jd_vals:
            try:
                x = float(jd)
                if not math.isfinite(x):
                    out_f.append("")
                    continue
                # JD starts at noon; shift by +0.5 to get civil day fraction.
                frac = (x + 0.5) % 1.0
                mins = int(round(frac * 24.0 * 60.0)) % (24 * 60)
                hh = mins // 60
                mm = mins % 60
                out_f.append(f"{hh:02d}:{mm:02d}")
            except Exception:  # noqa: BLE001
                out_f.append("")
        return out_f


def _latest_report_pdf(draft_dir: Path, obs_group: str) -> Path | None:
    """Return latest matching VYVAR report PDF for given obs_group."""
    try:
        d = Path(draft_dir) / "platesolve" / str(obs_group)
        pat = f"VYVAR_report_{str(obs_group)}_*.pdf"
        candidates = list(d.glob(pat))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        return candidates[0]
    except Exception:  # noqa: BLE001
        return None
    out: list[str] = []
    for jd in jd_vals:
        try:
            t = Time(float(jd), format="jd", scale="tdb").utc
            out.append(t.to_datetime().strftime("%H:%M"))
        except Exception:  # noqa: BLE001
            out.append("")
    return out


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


def _enrich_summary_with_zone_flags(
    summary_df: pd.DataFrame,
    active_targets_csv: Path | None,
) -> pd.DataFrame:
    """Doplní target meta z ``active_targets.csv`` (badge v UI, aj starší summary).

    Enriches:
      - zone_flag
      - skip_photometry
      - vsx_type
      - bp_rp
    """
    out = summary_df.copy()
    if "zone_flag" not in out.columns:
        out["zone_flag"] = ""
    else:
        out["zone_flag"] = out["zone_flag"].fillna("").astype(str)
    if "skip_photometry" not in out.columns:
        out["skip_photometry"] = False
    if "vsx_type" not in out.columns:
        out["vsx_type"] = ""
    if "bp_rp" not in out.columns:
        out["bp_rp"] = float("nan")
    if active_targets_csv is None or not Path(active_targets_csv).is_file():
        return out
    try:
        from photometry_phase2a import _normalize_gaia_id

        at = pd.read_csv(active_targets_csv, low_memory=False)
        if at.empty or "catalog_id" not in at.columns or "catalog_id" not in out.columns:
            return out
        zf_by: dict[str, str] = {}
        sk_by: dict[str, bool] = {}
        vt_by: dict[str, str] = {}
        bp_by: dict[str, float] = {}
        for _, r in at.iterrows():
            k = str(_normalize_gaia_id(r.get("catalog_id"))).strip()
            if not k:
                continue
            if "zone_flag" in at.columns:
                zf_by[k] = str(r.get("zone_flag", "") or "").strip()
            if "skip_photometry" in at.columns:
                _v = r.get("skip_photometry", False)
                sk_by[k] = (
                    bool(_v)
                    if isinstance(_v, (bool, np.bool_))
                    else str(_v).strip().lower() in ("1", "true", "yes", "t")
                )
            if "vsx_type" in at.columns:
                vt_by[k] = str(r.get("vsx_type", "") or "").strip()
            if "bp_rp" in at.columns:
                v = pd.to_numeric(r.get("bp_rp"), errors="coerce")
                try:
                    fv = float(v)
                except Exception:  # noqa: BLE001
                    fv = float("nan")
                if math.isfinite(fv):
                    bp_by[k] = float(fv)
        cids = out["catalog_id"].map(_normalize_gaia_id)
        n = len(out)
        if zf_by:
            out["zone_flag"] = [
                zf_by.get(str(cids.iloc[i] or "").strip(), str(out["zone_flag"].iloc[i] or ""))
                for i in range(n)
            ]
        if sk_by:
            sk_list: list[bool] = []
            prev_sk = out["skip_photometry"].tolist() if n else []
            for i in range(n):
                ck = str(cids.iloc[i] or "").strip()
                if ck in sk_by:
                    sk_list.append(bool(sk_by[ck]))
                else:
                    v0 = prev_sk[i] if i < len(prev_sk) else False
                    sk_list.append(
                        bool(v0)
                        if isinstance(v0, (bool, np.bool_))
                        else str(v0).strip().lower() in ("1", "true", "yes", "t")
                    )
            out["skip_photometry"] = sk_list
        if vt_by:
            prev_vt = out["vsx_type"].tolist() if n else []
            out["vsx_type"] = [
                vt_by.get(str(cids.iloc[i] or "").strip(), str(prev_vt[i] if i < len(prev_vt) else ""))
                for i in range(n)
            ]
        if bp_by:
            prev_bp = pd.to_numeric(out["bp_rp"], errors="coerce").tolist() if n else []
            bp_list: list[float] = []
            for i in range(n):
                ck = str(cids.iloc[i] or "").strip()
                if ck in bp_by:
                    bp_list.append(float(bp_by[ck]))
                else:
                    v0 = prev_bp[i] if i < len(prev_bp) else float("nan")
                    try:
                        bp_list.append(float(v0))
                    except Exception:  # noqa: BLE001
                        bp_list.append(float("nan"))
            out["bp_rp"] = pd.to_numeric(bp_list, errors="coerce")
    except Exception:  # noqa: BLE001
        return out
    return out


def _phase2a_target_choice_label(row: pd.Series) -> str:
    """Text pre selectbox Fázy 2A — názov + badge podľa ``zone_flag`` / ``skip_photometry``."""
    vsx = str(row.get("vsx_name", "") or "").strip()
    cid = str(row.get("catalog_id", "") or "").strip()
    base = vsx if vsx else cid
    if not base:
        base = "(bez mena)"
    zf = str(row.get("zone_flag", "") or "").strip().lower()
    sk = row.get("skip_photometry", False)
    sk_b = (
        bool(sk)
        if isinstance(sk, (bool, np.bool_))
        else str(sk).strip().lower() in ("1", "true", "yes", "t")
    )
    if sk_b or zf == "saturated":
        badge = "🔴 saturated — fotometria nedostupná"
    elif zf == "linear":
        badge = "🟢 linear"
    elif zf in ("noisy1", "noisy2"):
        badge = f"🟡 {zf}"
    elif zf == "noisy3":
        badge = "🟠 noisy3"
    elif zf == "neznáma_zóna":
        badge = "⚪ neznáma zóna"
    elif zf:
        badge = f"⚪ {zf}"
    else:
        return base
    return f"{base}  {badge}"


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
    show_airmass: bool = False,
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
            lc_df = _cached_read_csv(str(lc_csv))
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
                    bjd_num = pd.to_numeric(lc_df["bjd"], errors="coerce")
                    _, bjd_x_off = jd_series_relative(bjd_num)
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
                        x_raw = pd.to_numeric(sub["bjd"], errors="coerce").to_numpy(dtype=float)
                        x_plot = x_raw - float(bjd_x_off) if bjd_x_off is not None else x_raw
                        fig.add_trace(
                            go.Scatter(
                                x=x_plot,
                                y=sub[y_col],
                                error_y=err_kwargs if err_kwargs else None,
                                mode="markers",
                                marker=dict(color=color, size=7, line=dict(width=0.5, color="#ffffff")),
                                name=flag,
                                customdata=x_raw,
                                hovertemplate=(
                                    "<b>%{fullData.name}</b><br>BJD=%{customdata:.6f}<br>"
                                    + y_label
                                    + "=%{y:.4f}<extra></extra>"
                                ),
                            )
                        )

                    # Optional AIRMASS overlay (right axis).
                    am_col = _airmass_column(lc_df)
                    if bool(show_airmass) and am_col is not None:
                        am = pd.to_numeric(lc_df[am_col], errors="coerce")
                        ok_am = am.notna() & bjd_num.notna()
                        if bool(ok_am.any()):
                            x_raw_am = bjd_num[ok_am].to_numpy(dtype=float)
                            x_plot_am = x_raw_am - float(bjd_x_off) if bjd_x_off is not None else x_raw_am
                            fig.add_trace(
                                go.Scatter(
                                    x=x_plot_am,
                                    y=am[ok_am].to_numpy(dtype=float),
                                    mode="lines",
                                    name="AIR MASS",
                                    yaxis="y2",
                                    line=dict(color="rgba(56,189,248,0.85)", width=2),
                                    hovertemplate="AIRMASS=%{y:.3f}<extra></extra>",
                                )
                            )

                    # Secondary X axis on top: UT (HH:MM) labels.
                    # Tick *positions* must be in the same units as the plotted x (BJD offset),
                    # but labels should come from JD when available (closest to "UT time").
                    x_ticks = []
                    ut_text = []
                    try:
                        bjd_arr = pd.to_numeric(lc_df.get("bjd"), errors="coerce").to_numpy(dtype=float)
                        jd_arr = (
                            pd.to_numeric(lc_df.get("jd"), errors="coerce").to_numpy(dtype=float)
                            if "jd" in lc_df.columns
                            else None
                        )
                        hjd_arr = (
                            pd.to_numeric(lc_df.get("hjd"), errors="coerce").to_numpy(dtype=float)
                            if "hjd" in lc_df.columns
                            else None
                        )
                        ok = np.isfinite(bjd_arr)
                        if int(np.count_nonzero(ok)) >= 2:
                            bjd_ok = bjd_arr[ok]
                            # Evenly spaced indices over the *sorted by time* array.
                            order = np.argsort(bjd_ok)
                            bjd_ok = bjd_ok[order]
                            n_ticks = 6
                            idx = np.linspace(0, bjd_ok.size - 1, num=n_ticks, dtype=int).tolist()
                            bjd_ticks = [float(bjd_ok[i]) for i in idx]
                            x_ticks = [
                                (bt - float(bjd_x_off)) if bjd_x_off is not None else bt for bt in bjd_ticks
                            ]

                            # Build UT labels from JD at nearest BJD positions.
                            if jd_arr is not None and int(np.count_nonzero(np.isfinite(jd_arr))) >= 2:
                                # Map back to original row indices for label pick.
                                row_idx_ok = np.flatnonzero(ok)[order]
                                pick_rows = [int(row_idx_ok[i]) for i in idx]
                                jd_ticks_for_label = [float(jd_arr[r]) for r in pick_rows]
                                ut_text = _ut_tick_labels_from_jd(jd_ticks_for_label)
                            elif hjd_arr is not None and int(np.count_nonzero(np.isfinite(hjd_arr))) >= 2:
                                row_idx_ok = np.flatnonzero(ok)[order]
                                pick_rows = [int(row_idx_ok[i]) for i in idx]
                                hjd_ticks_for_label = [float(hjd_arr[r]) for r in pick_rows]
                                ut_text = _ut_tick_labels_from_jd(hjd_ticks_for_label)
                            else:
                                ut_text = _ut_tick_labels_from_jd(bjd_ticks)
                    except Exception:  # noqa: BLE001
                        x_ticks, ut_text = [], []

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
                        yaxis2=dict(
                            title=dict(text="airmass", **_axis_title),
                            tickfont=dict(color="#000000", size=12),
                            overlaying="y",
                            side="right",
                            showgrid=False,
                        ),
                        xaxis=dict(
                            title=dict(text=jd_axis_title("BJD (TDB)", bjd_x_off), **_axis_title),
                            tickfont=dict(color="#000000", size=12),
                            gridcolor="#e2e8f0",
                        ),
                        xaxis2=dict(
                            overlaying="x",
                            side="top",
                            anchor="y",
                            position=1.0,
                            title=dict(text="UT (HH:MM)", **_axis_title),
                            tickfont=dict(color="#000000", size=12),
                            showticklabels=True,
                            ticks="outside",
                            tickmode="array",
                            tickvals=x_ticks,
                            ticktext=ut_text,
                            showgrid=False,
                            automargin=True,
                        ),
                        height=350,
                        margin=dict(l=40, r=50, t=70, b=40),
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
            _sk = target_row.get("skip_photometry", False)
            _sk_b = (
                bool(_sk)
                if isinstance(_sk, (bool, np.bool_))
                else str(_sk).strip().lower() in ("1", "true", "yes", "t")
            )
            _zf_lc = str(target_row.get("zone_flag", "") or "").strip().lower()
            if _sk_b or _zf_lc == "saturated":
                st.info(
                    "Fotometria pre tento cieľ bola preskočená (saturovaná hviezda v masterstars — "
                    "meranie by nebolo spoľahlivé; pozícia a mapa poľa sú v zozname cieľov)."
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
    vsx_url = ""
    try:
        nm = str(vsx_name or "").strip()
    except Exception:  # noqa: BLE001
        nm = ""
    if nm and nm != catalog_id:
        # Prefer name search (more user-friendly than coordinate form).
        # VSX supports HTTP GET queries via view=results.get with ident=<name>.
        # Spec: https://www.aavso.org/direct-web-query-vsxvsp
        vsx_url = f"https://www.aavso.org/vsx/index.php?view=results.get&ident={quote_plus(nm)}"
    else:
        vsx_url = (
            f"https://www.aavso.org/vsx/index.php?view=results.get&coords="
            f"{quote_plus(f'{ra_target:.5f} {dec_target:+.5f}')}&format=d&size=0.01"
        )
    st.markdown(
        f"**{vsx_name}** &nbsp; "
        f"[Vizier]({vizier_url}) &nbsp; "
        f"[VSX]({vsx_url})"
    )
    # Extra meta (if available in enriched summary).
    vt = str(target_row.get("vsx_type", "") or "").strip()
    zf = str(target_row.get("zone_flag", "") or "").strip().lower()
    bp = target_row.get("bp_rp")
    bp_s = _fmt_opt_num(bp, ".3f")
    if vt or zf or bp_s:
        badge = ""
        if zf == "linear":
            badge = "🟢 linear"
        elif zf in ("noisy1", "noisy2"):
            badge = f"🟡 {zf}"
        elif zf == "noisy3":
            badge = "🟠 noisy3"
        elif zf == "saturated":
            badge = "🔴 saturated"
        elif zf:
            badge = f"⚪ {zf}"
        parts = []
        if vt:
            parts.append(f"vsx_type: **{vt}**")
        if badge:
            parts.append(f"zone_flag: {badge}")
        if bp_s and bp_s != "—":
            parts.append(f"bp_rp: **{bp_s}**")
        if parts:
            st.caption(" | ".join(parts))

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
            # Relative weights for display (normalize to max=1.0 within this target).
            w_num = (
                pd.to_numeric(target_comps.get("comp_weight"), errors="coerce")
                if "comp_weight" in target_comps.columns
                else pd.Series([], dtype=float)
            )
            try:
                w_valid = w_num[np.isfinite(w_num.to_numpy(dtype=float)) & (w_num.to_numpy(dtype=float) > 0)]
                max_w = float(w_valid.max()) if int(w_valid.size) > 0 else float("nan")
            except Exception:  # noqa: BLE001
                max_w = float("nan")
            for i, (_, row) in enumerate(target_comps.iterrows(), 1):
                ra_c = _float_coord_row(row, "ra_deg", "ra")
                dec_c = _float_coord_row(row, "dec_deg", "dec")
                mag_c = row.get("mag")
                # Important: keep B-V and BP-RP separate (do not fallback between them).
                bv_c = pd.to_numeric(row.get("b_v"), errors="coerce")
                bp_c = pd.to_numeric(row.get("bp_rp"), errors="coerce")
                dist_deg_c = row.get("_dist_deg")
                nfr_c = row.get("comp_n_frames")
                rms_c = row.get("comp_rms")
                w_c = pd.to_numeric(row.get("comp_weight"), errors="coerce")
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
                bp_str = _fmt_opt_num(bp_c, ".3f")
                dist_str = _fmt_opt_num(dist_deg_c, ".6f")
                nfr_str = _fmt_opt_num(nfr_c, ".0f")
                rms_str = _fmt_opt_num(rms_c, ".4f")
                wrel_str = (
                    f"{float(w_c) / max_w:.3f}"
                    if np.isfinite(pd.to_numeric(w_c, errors="coerce")) and float(pd.to_numeric(w_c, errors="coerce")) > 0 and np.isfinite(max_w) and max_w > 0
                    else "—"
                )
                bg = _row_bg(q)
                rows_html.append(
                    "<tr style=\""
                    + bg
                    + "\">"
                    f"<td>C{i:02d}</td>"
                    f"<td>{html.escape(mag_str)}</td>"
                    f"<td>{html.escape(bv_str)}</td>"
                    f"<td>{html.escape(bp_str)}</td>"
                    f"<td>{html.escape(dist_str)}</td>"
                    f"<td>{html.escape(nfr_str)}</td>"
                    f"<td>{html.escape(rms_str)}</td>"
                    f"<td title=\"Relatívna váha 1/σ² (Broeg 2005)\">{html.escape(wrel_str)}</td>"
                    f"<td>{_tier_badge(tier_c)}</td>"
                    f"<td>{html.escape(stav)}</td>"
                    f"<td><a href=\"{html.escape(viz_c)}\" target=\"_blank\" rel=\"noopener noreferrer\">↗</a></td>"
                    "</tr>"
                )

            thead = (
                "<thead><tr>"
                "<th>#</th><th>mag</th><th>B-V</th><th>bp_rp</th><th>dist_deg</th><th>comp_n_frames</th><th>p2p RMS</th>"
                "<th title=\"Relatívna váha 1/σ² (Broeg 2005)\">w (rel)</th>"
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
            st.caption("B-V vypočítané z Gaia BP-RP (Riello et al. 2021, ±0.05 mag)")


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
    st.caption("Fáza 0+1 + Fáza 2A ako jeden neoddeliteľný krok.")

    if draft_id is None and draft_dir_override is None:
        st.info("Žiadny aktívny draft. Načítaj draft vyššie alebo spusti VAR-STREM.")
        return

    # Draft dir for PDF reports and fallbacks.
    if draft_dir_override is not None and Path(draft_dir_override).is_dir():
        draft_dir = Path(draft_dir_override).resolve()
    else:
        draft_dir = (Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft_id):06d}").resolve()

    all_setups = _find_phase2a_paths(cfg, draft_id, draft_dir_override=draft_dir_override)
    if not all_setups:
        st.warning("Nenájdené vstupné súbory.")
        return

    setup_options = list(all_setups.keys())

    def _detect_obs_groups() -> list[str]:
        """Obs_groups from detrended_aligned/lights/{obs_group}/ (proc_*.csv)."""
        try:
            if draft_dir_override is not None and draft_dir_override.is_dir():
                draft_dir = draft_dir_override.resolve()
            else:
                draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft_id):06d}"
            root = draft_dir / "detrended_aligned" / "lights"
            if not root.is_dir():
                return []
            out: list[str] = []
            for d in sorted(root.iterdir()):
                if not d.is_dir():
                    continue
                if any(d.glob("proc_*.csv")):
                    out.append(d.name)
            return out
        except Exception:  # noqa: BLE001
            return []

    detected = _detect_obs_groups()
    run_groups = [g for g in detected if g in setup_options] if detected else setup_options

    selected_setup = st.selectbox(
        "Platesolve setup:",
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

    exists = _phase2a_results_exist(output_dir)

    # Header/status line + global run button.
    col_info, col_run = st.columns([3, 2])
    with col_info:
        st.markdown(f"**Platesolve setup:** `{selected_setup}` &nbsp; | &nbsp; **FWHM:** `{float(fwhm_px):.3f}px`")
        if exists:
            ts = _phase2a_timestamp(output_dir)
            st.success(f"✅ Prebehlo: {ts}")
        else:
            st.warning("⚠️ Nespustené")

        # Always show PDF download if it already exists (even after rerun).
        pdf_latest = _latest_report_pdf(draft_dir, str(selected_setup))
        if pdf_latest is not None and pdf_latest.exists():
            try:
                with open(pdf_latest, "rb") as f:
                    st.download_button(
                        label=f"📥 Stiahnuť PDF správu ({selected_setup})",
                        data=f.read(),
                        file_name=pdf_latest.name,
                        mime="application/pdf",
                        key=f"pdf_dl_hdr_{selected_setup}",
                    )
            except Exception:  # noqa: BLE001
                pass
    with col_run:
        run_btn = st.button("🔄 RUN Aperture Photometry", key="phase2a_run_full", type="primary")

    with st.expander("Stav setupov", expanded=False):
        for nm in setup_options:
            p = all_setups.get(str(nm)) or {}
            out_d = p.get("output_dir")
            if _phase2a_results_exist(out_d):
                st.success(f"✅ {nm}: {_phase2a_timestamp(out_d)}")
            else:
                st.warning(f"⚠️ {nm}: nespustené")

    if run_btn:
        from photometry_core import run_full_photometry_pipeline
        from vyvar_ui_status import vyvar_footer_idle, vyvar_footer_running

        try:
            total = len(run_groups)
            if total <= 0:
                st.warning("Nenájdené žiadne obs_group v detrended_aligned/lights.")
                return
            vyvar_footer_running("Aperture Photometry", f"Štartujem ({total} setups)…")
            prog = st.progress(0, text="Starting…")
            lines_ph = st.empty()
            statuses: dict[str, str] = {}
            errors: list[str] = []
            n_ok = 0

            def _render_lines() -> None:
                lines = [statuses.get(g, f"{g} …") for g in run_groups]
                lines_ph.markdown("\n".join(lines))

            for i, nm in enumerate(run_groups, start=1):
                statuses[nm] = f"{nm} ██████ …"
                _render_lines()
                prog.progress(int(round(100 * (i - 1) / max(total, 1))), text=f"{nm}: spúšťam…")
                vyvar_footer_running("Aperture Photometry", f"{nm}: Fáza 0+1 + 2A…")

                p = all_setups.get(str(nm)) or {}
                try:
                    ms_fits = Path(p.get("masterstar_fits")) if p.get("masterstar_fits") else None
                    og_dir = Path(p.get("obs_group_dir")) if p.get("obs_group_dir") else None
                    ms_csv = (og_dir / "masterstars_full_match.csv") if og_dir is not None else None
                    vt_csv = (og_dir / "variable_targets.csv") if og_dir is not None else None
                    pf_dir = Path(p.get("per_frame_csv_dir")) if p.get("per_frame_csv_dir") else None
                    dt_dir = Path(p.get("detrended_aligned_dir")) if p.get("detrended_aligned_dir") else None
                    out_d = Path(p.get("output_dir")) if p.get("output_dir") else None

                    missing: list[str] = []
                    if ms_fits is None or not ms_fits.exists():
                        missing.append("MASTERSTAR.fits")
                    if ms_csv is None or not ms_csv.exists():
                        missing.append("masterstars_full_match.csv")
                    if vt_csv is None or not vt_csv.exists():
                        missing.append("variable_targets.csv")
                    if pf_dir is None or not pf_dir.exists():
                        missing.append("per-frame CSV adresár")
                    if dt_dir is None or not dt_dir.exists():
                        missing.append("detrended_aligned adresár")
                    if out_d is None:
                        missing.append("output_dir")
                    if missing:
                        raise FileNotFoundError(", ".join(missing))

                    def _cb(msg: str) -> None:
                        vyvar_footer_running("Aperture Photometry", f"{nm}: {msg}")

                    _ = run_full_photometry_pipeline(
                        masterstar_fits_path=ms_fits,
                        variable_targets_csv=vt_csv,
                        masterstars_csv=ms_csv,
                        per_frame_csv_dir=pf_dir,
                        detrended_aligned_dir=dt_dir,
                        output_dir=out_d,
                        cfg=cfg,
                        progress_cb=_cb,
                    )
                    n_ok += 1
                    statuses[nm] = f"{nm} ████████████ ✓"

                    # PDF report (optional)
                    try:
                        from photometry_report import generate_photometry_report

                        pdf_path = generate_photometry_report(
                            draft_dir=draft_dir,
                            obs_group=str(nm),
                            output_pdf=None,
                        )
                        if pdf_path is not None:
                            st.success(f"📄 PDF uložené: {Path(pdf_path).name}")
                            try:
                                st.session_state.setdefault("vyvar_pdf_paths", {})[str(nm)] = str(pdf_path)
                            except Exception:  # noqa: BLE001
                                pass
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label=f"📥 Stiahnuť {Path(pdf_path).name}",
                                    data=f.read(),
                                    file_name=Path(pdf_path).name,
                                    mime="application/pdf",
                                    key=f"pdf_download_{nm}",
                                )
                    except Exception as _pdf_exc:  # noqa: BLE001
                        st.warning(f"PDF sa nepodarilo vygenerovať: {_pdf_exc}")
                except Exception as exc_nm:  # noqa: BLE001
                    statuses[nm] = f"{nm} ███████ ✗"
                    errors.append(f"{nm}: {exc_nm}")
                _render_lines()
                prog.progress(int(round(100 * i / max(total, 1))), text=f"{nm}: hotovo")

            if n_ok:
                st.success(f"Hotovo — {n_ok} setups spracovaných")
            if errors:
                (st.error if n_ok == 0 else st.warning)(
                    "Problémy pri niektorých setupoch:\n" + "\n".join(errors)
                )
            vyvar_footer_idle()
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Chyba: {exc}")
            logging.exception("RUN Aperture Photometry zlyhal")
        finally:
            try:
                vyvar_footer_idle()
            except Exception:  # noqa: BLE001
                pass
        return

    if not exists:
        st.warning(
            f"⚠️ Fotometria nebola spustená pre **{selected_setup}**.\n\n"
            "Klikni **RUN Aperture Photometry**."
        )
        return

    if output_dir is None:
        st.warning("Output adresár nie je dostupný.")
        return

    # Phase01 artifacts (generated by full pipeline and used for UI enrich/table).
    at_csv = paths.get("active_targets_csv")
    comp_csv = paths.get("comparison_stars_csv")

    summary_df = _load_summary(output_dir)
    comp_df = pd.DataFrame()
    if comp_csv is not None and Path(comp_csv).exists():
        comp_df = pd.read_csv(comp_csv, low_memory=False)

    if summary_df.empty:
        st.info("Zatiaľ žiadne výsledky.")
        return

    at_path_for_zf = Path(at_csv) if at_csv is not None else None
    summary_df = _enrich_summary_with_zone_flags(summary_df, at_path_for_zf)

    _n_sum = int(len(summary_df))
    _idx_opts = list(range(_n_sum))
    _sel_i = st.selectbox(
        "Vyber premennú hviezdu:",
        options=_idx_opts,
        format_func=lambda i: _phase2a_target_choice_label(summary_df.iloc[int(i)]),
        key="phase2a_target_select",
    )
    target_row = summary_df.iloc[int(_sel_i)]
    try:
        catalog_id = str(target_row.get("catalog_id", ""))
    except Exception:
        catalog_id = ""

    col1, col2, col3 = st.columns([1, 2, 2])
    with col1:
        show_detrended = st.toggle(
            "Airmass detrend",
            value=True,
            key="toggle_am_detrend",
        )
        show_airmass = st.toggle(
            "Zobraziť airmass v grafe",
            value=True,
            key="toggle_show_airmass",
        )
    with col2:
        show_outliers = st.toggle(
            "Zobraziť outlier a saturated body",
            value=True,
            key="phase2a_show_outliers",
        )
    with col3:
        preload_all = st.toggle(
            "Načítať všetky krivky do pamäte",
            value=False,
            key="phase2a_preload_all_curves",
        )

    if preload_all:
        lc_dir = output_dir / "lightcurves"
        if lc_dir.is_dir():
            with st.spinner("Načítavam lightcurves do pamäte…"):
                try:
                    for p in sorted(lc_dir.glob("lightcurve_*.csv")):
                        _ = _cached_read_csv(str(p))
                except Exception:  # noqa: BLE001
                    pass

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

            x_series_for_offset: list[pd.Series] = []
            trace_specs: list[tuple[str, pd.DataFrame, str, str, str | None, str]] = []
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
                x_series_for_offset.append(pd.to_numeric(lc_df[x_col], errors="coerce"))
                trace_specs.append((str(setup_name), lc_df, x_col, y_col, err_col, color))

            combined_x = pd.concat(x_series_for_offset, ignore_index=True) if x_series_for_offset else pd.Series(dtype=float)
            _, overlay_x_off = jd_series_relative(combined_x)

            for setup_name, lc_df, x_col, y_col, err_col, color in trace_specs:
                x_raw = pd.to_numeric(lc_df[x_col], errors="coerce").to_numpy(dtype=float)
                x_plot = x_raw - float(overlay_x_off) if overlay_x_off is not None else x_raw
                fig.add_trace(
                    go.Scatter(
                        x=x_plot,
                        y=lc_df[y_col],
                        mode="markers+lines",
                        name=setup_name,
                        marker=dict(color=color, size=4),
                        line=dict(color=color, width=0.5),
                        error_y=dict(
                            type="data",
                            array=(lc_df[err_col].tolist() if err_col is not None else None),
                            visible=bool(err_col is not None),
                        ),
                        customdata=x_raw,
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>BJD=%{customdata:.6f}<br>mag=%{y:.4f}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(
                title=f"Svetelné krivky — {catalog_id}",
                xaxis_title=jd_axis_title("BJD (TDB)", overlay_x_off),
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
                show_airmass=show_airmass,
            )
    else:
        _render_target_detail(
            target_row,
            output_dir,
            show_outliers,
            comp_df=comp_df,
            show_detrended=show_detrended,
            show_airmass=show_airmass,
        )

    st.divider()

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Svetelné krivky", int(len(summary_df)))

    rms_cur = pd.to_numeric(summary_df.get("lc_rms"), errors="coerce")
    if rms_cur.notna().any():
        c2.metric("Median lc_rms", f"{float(rms_cur.median()):.4f}")
        good = int((rms_cur < 0.05).sum())
        c3.metric("RMS < 0.05 mag", good)
    else:
        c2.metric("Median lc_rms", "—")
        c3.metric("RMS < 0.05 mag", "—")

    ngc = pd.to_numeric(summary_df.get("n_good_comp"), errors="coerce")
    c4.metric("Avg good comp", f"{float(ngc.mean()):.1f}" if ngc.notna().any() else "—")

    # Cross-setup metrics (based on existing photometry_summary.csv files).
    done_setups = 0
    frames: list[pd.DataFrame] = []
    for nm in setup_options:
        p = all_setups.get(str(nm)) or {}
        out_d = p.get("output_dir")
        if not _phase2a_results_exist(out_d):
            continue
        done_setups += 1
        try:
            df0 = _load_summary(Path(out_d)) if out_d is not None else pd.DataFrame()
        except Exception:  # noqa: BLE001
            df0 = pd.DataFrame()
        if df0 is None or df0.empty:
            continue
        at_p = p.get("active_targets_csv")
        df0 = _enrich_summary_with_zone_flags(df0, Path(at_p) if at_p is not None else None)
        frames.append(df0)

    c5.metric("Setups", int(done_setups))
    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_df.empty:
        rms_all = pd.to_numeric(all_df.get("lc_rms"), errors="coerce")
        c6.metric("Best lc_rms", f"{float(rms_all.min()):.4f}" if rms_all.notna().any() else "—")
        c7.metric("Worst lc_rms", f"{float(rms_all.max()):.4f}" if rms_all.notna().any() else "—")
        bp_all = pd.to_numeric(all_df.get("bp_rp"), errors="coerce")
        c8.metric("Avg bp_rp", f"{float(bp_all.mean()):.3f}" if bp_all.notna().any() else "—")
    else:
        c6.metric("Best lc_rms", "—")
        c7.metric("Worst lc_rms", "—")
        c8.metric("Avg bp_rp", "—")
