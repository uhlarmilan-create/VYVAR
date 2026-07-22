"""Aperture Photometry Lightcurves - Faza 2A UI."""

from __future__ import annotations

import html
import json
import logging
import math
from datetime import datetime
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
from utils import resolve_draft_dir_path
from vyvar_ui_status import is_bv_related_phase01_ui_column, log_if_ui_hiding_bv_for_bprp_primary

# Gaia ID musi byt str - float64 straca cifry
_GAIA_ID_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}

# Columns loaded from lightcurve_*.csv for charts / preload (see _render_target_detail, multi-filter overlay).
_LC_OVERVIEW_COLS = [
    "bjd",
    "bjd_tdb_mid",
    "bjd_tdb",
    "hjd",
    "jd",
    "airmass",
    "air_mass",
    "AIRMASS",
    "am",
    "mag_calib",
    "mag_calib_raw",
    "mag_err",
    "err",
    "flag",
]

_MAX_LC_PRELOAD = 200

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pomocne funkcie
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_read_csv(path_s: str) -> pd.DataFrame:
    if not Path(path_s).is_file():
        return pd.DataFrame()
    return pd.read_csv(
        path_s,
        usecols=lambda c: c in _LC_OVERVIEW_COLS,
        low_memory=False,
        dtype=_GAIA_ID_DTYPE,
    )


def _airmass_column(df: pd.DataFrame) -> str | None:
    for c in ("airmass", "air_mass", "AIRMASS", "am"):
        if c in df.columns:
            return c
    return None


def _ut_tick_labels_from_jd(jd_vals: "list[float]") -> list[str]:
    """Format JD-like values (BJD/HJD/JD) to UT HH:MM labels."""
    try:
        pass
    except Exception as exc:  # noqa: BLE001 - primary astropy path removed; body is ``pass``; broad retained if restored
        _log.debug("UT tick label primary path skipped: %s", exc)
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
            except (TypeError, ValueError) as inner_exc:
                _log.debug("UT tick label fallback skipped for jd=%r: %s", jd, inner_exc)
                out_f.append("")
        return out_f


def _latest_report_pdf(draft_dir: Path, obs_group: str) -> Path | None:
    """Return newest PDF report for this setup (``report_{setup}.pdf`` or legacy glob)."""
    try:
        d = Path(draft_dir) / "platesolve" / str(obs_group) / "photometry"
        primary = d / f"report_{obs_group!s}.pdf"
        if primary.exists():
            return primary
        legacy_dir = Path(draft_dir) / "platesolve" / str(obs_group)
        pat = f"VYVAR_report_{obs_group!s}_*.pdf"
        candidates = list(legacy_dir.glob(pat))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        return candidates[0]
    except Exception:  # noqa: BLE001
        # EXC-0492: T3 -- UI diagnostic/plot only (candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() ... (EXCEPT-BULK 2026-07-08)
        return None


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
        draft_dir = resolve_draft_dir_path(
            draft_dir_override, draft_id, cfg.archive_root
        )
        if draft_dir is None:
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
        # EXC-0493: T3 -- UI diagnostic/plot only (} / return result  # type: ignore[return-value] / except Excep... (EXCEPT-BULK 2026-07-08)
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
        # EXC-0494: T3 -- UI diagnostic/plot only (if 0.5 < fv < 30.0: / return round(fv, 3) / except Exception: ... (EXCEPT-BULK 2026-07-08)
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
    dd = resolve_draft_dir_path(draft_dir_override, draft_id, cfg.archive_root)
    if dd is None:
        return None
    ps = dd / "platesolve"
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


@st.cache_data(ttl=300)
def _load_summary(summary_csv_str: str) -> pd.DataFrame:
    """Load photometry_summary.csv - cached 5 min to avoid re-read on every render."""
    summary_csv = Path(summary_csv_str)
    if not summary_csv.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(summary_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    except Exception as exc:  # noqa: BLE001
        # EXC-0495: T3 -- UI diagnostic/plot only (try: / return pd.read_csv(summary_csv, low_memory=False, dtype... (EXCEPT-BULK 2026-07-08)
        logging.warning("[PERF-7] Cannot read summary: %s", exc)
        return pd.DataFrame()


def _enrich_summary_with_zone_flags(
    summary_df: pd.DataFrame,
    active_targets_csv: Path | None,
) -> pd.DataFrame:
    """Doplni target meta z ``active_targets.csv`` (badge v UI, aj starsi summary).

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
    if "b_v" not in out.columns:
        out["b_v"] = float("nan")
    if active_targets_csv is None or not Path(active_targets_csv).is_file():
        return out
    try:
        from photometry_phase2a import _normalize_gaia_id

        at = pd.read_csv(active_targets_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
        if at.empty or "catalog_id" not in at.columns or "catalog_id" not in out.columns:
            return out
        zf_by: dict[str, str] = {}
        sk_by: dict[str, bool] = {}
        vt_by: dict[str, str] = {}
        bp_by: dict[str, float] = {}
        bv_by: dict[str, float] = {}
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
            if "b_v" in at.columns:
                v2 = pd.to_numeric(r.get("b_v"), errors="coerce")
                try:
                    fv2 = float(v2)
                except Exception:  # noqa: BLE001
                    fv2 = float("nan")
                if math.isfinite(fv2):
                    bv_by[k] = float(fv2)
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
        if bv_by:
            prev_bv = pd.to_numeric(out["b_v"], errors="coerce").tolist() if n else []
            bv_list: list[float] = []
            for i in range(n):
                ck = str(cids.iloc[i] or "").strip()
                if ck in bv_by:
                    bv_list.append(float(bv_by[ck]))
                else:
                    v0 = prev_bv[i] if i < len(prev_bv) else float("nan")
                    try:
                        bv_list.append(float(v0))
                    except Exception:  # noqa: BLE001
                        bv_list.append(float("nan"))
            out["b_v"] = pd.to_numeric(bv_list, errors="coerce")
    except Exception:  # noqa: BLE001
        # EXC-0496: T3 -- UI diagnostic/plot only (bv_list.append(float('nan')) / out['b_v'] = pd.to_numeric(bv_l... (EXCEPT-BULK 2026-07-08)
        return out
    return out


def _phase2a_target_choice_label(row: pd.Series) -> str:
    """Text pre selectbox Fazy 2A - nazov + badge podla ``zone_flag`` / ``skip_photometry``."""
    vsx = str(row.get("vsx_name", "") or "").strip()
    cid = str(row.get("catalog_id", "") or "").strip()
    base = vsx if vsx else cid
    if not base:
        base = "(no name)"
    zf = str(row.get("zone_flag", "") or "").strip().lower()
    sk = row.get("skip_photometry", False)
    sk_b = (
        bool(sk)
        if isinstance(sk, (bool, np.bool_))
        else str(sk).strip().lower() in ("1", "true", "yes", "t")
    )
    if sk_b or zf == "saturated":
        sr = str(row.get("skip_reason", "") or "").strip().lower()
        if sr == "vsx_type_out_of_scope":
            badge = "[grey] VSX type out of scope - not measured"
        else:
            badge = "[red] saturated - photometry unavailable"
    elif zf == "linear":
        badge = "[green] linear"
    elif zf in ("noisy1", "noisy2"):
        badge = f"[yellow] {zf}"
    elif zf == "noisy3":
        badge = "[orange] noisy3"
    elif zf == "neznama_zona":
        badge = "o unknown zone"
    elif zf:
        badge = f"o {zf}"
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


def _fmt_opt_num(v: Any, fmt: str, empty: str = "-") -> str:
    if v is None:
        return empty
    if isinstance(v, float) and not math.isfinite(v):
        return empty
    s = str(v).strip()
    if s.lower() in ("", "nan", "none"):
        return empty
    if s == "-":
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
    phase01_use_bprp_primary: bool = True,
) -> None:
    """Interaktivna krivka (Plotly z CSV), field map PNG, metriky, odkazy Vizier/VSX."""
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
                at_df = pd.read_csv(at_path, low_memory=False, dtype=_GAIA_ID_DTYPE)
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
                # EXC-0497: T3 -- UI diagnostic/plot only (ra_target = _float_coord_row(r0, 'ra_deg', 'ra') / dec_target ... (EXCEPT-BULK 2026-07-08)
                pass

    lc_dir = output_dir / "lightcurves"
    lc_csv = lc_dir / f"lightcurve_{catalog_id}.csv"

    col_lc, col_map = st.columns([3, 2])

    with col_lc:
        st.markdown(f"**Light curve - {vsx_name}**")
        if lc_csv.exists():
            lc_df = _cached_read_csv(str(lc_csv))
            if not show_outliers and "flag" in lc_df.columns:
                lc_df = lc_df[lc_df["flag"] == "normal"]

            y_col = "mag_calib" if show_detrended else "mag_calib_raw"
            y_label = (
                "mag_calib (detrend)" if show_detrended else "mag_calib_raw (without detrend)"
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
                    # Svetle pozadie + vyrazne farby bodov (citatelne aj v tmavom Streamlit)
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
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.caption("Interactive chart unavailable (plotly is not installed).")
            else:
                st.info(
                    f"CSV is missing columns bjd / {y_col} or the file is empty."
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
                    "Photometry for this target was skipped (saturated star in masterstars - "
                    "measurement would not be reliable; position and field map remain in the target list)."
                )
            else:
                st.info("Lightcurve CSV does not exist. Run Phase 2A.")

    with col_map:
        fm_png = lc_dir / f"field_map_{catalog_id}.png"
        if fm_png.exists():
            st.image(str(fm_png), width="stretch")
        else:
            global_fm = output_dir / "field_map.png"
            if global_fm.exists():
                st.image(str(global_fm), width="stretch")
                st.caption("(global field map - per-target not available)")
            else:
                st.info("Field map does not exist.")

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
        cols[2].metric("aperture", f"{float(ap):.1f}px")

    _am_d = target_row.get("am_detrended")
    if _am_d is not None and pd.notna(_am_d):
        _am_on = str(_am_d).strip().lower() in ("true", "1", "yes")
        if not _am_on:
            st.caption("without detrend (signal preserved)")

    st.markdown("**Variable star**")
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
    bv = target_row.get("b_v")
    bv_s = _fmt_opt_num(bv, ".2f")
    hide_bv_meta = bool(phase01_use_bprp_primary)
    if vt or zf or bp_s or (bv_s and not hide_bv_meta):
        badge = ""
        if zf == "linear":
            badge = "[green] linear"
        elif zf in ("noisy1", "noisy2"):
            badge = f"[yellow] {zf}"
        elif zf == "noisy3":
            badge = "[orange] noisy3"
        elif zf == "saturated":
            badge = "[red] saturated"
        elif zf:
            badge = f"o {zf}"
        parts = []
        if vt:
            parts.append(f"vsx_type: **{vt}**")
        if badge:
            parts.append(f"zone_flag: {badge}")
        if bp_s and bp_s != "-":
            parts.append(f"bp_rp: **{bp_s}**")
        if (not hide_bv_meta) and bv_s and bv_s != "-":
            parts.append(f"B-V: **{bv_s}**")
        if parts:
            st.caption(" | ".join(parts))

    _exo_host = str(target_row.get("exo_host_name", "") or "").strip()
    if not _exo_host:
        from pipeline import resolve_masterstars_metadata_csv  # noqa: PLC0415

        ms_csv = resolve_masterstars_metadata_csv(output_dir.parent)
        if ms_csv is not None:
            try:
                ms_df = pd.read_csv(ms_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
                if "catalog_id" in ms_df.columns:
                    cid_norm = _normalize_gaia_id(catalog_id)
                    ms_df = ms_df.copy()
                    ms_df["_nid"] = ms_df["catalog_id"].apply(_normalize_gaia_id)
                    hit_ms = ms_df[ms_df["_nid"] == cid_norm]
                    if not hit_ms.empty and "exo_host_name" in hit_ms.columns:
                        _exo_host = str(hit_ms.iloc[0].get("exo_host_name", "") or "").strip()
            except Exception:  # noqa: BLE001
                # EXC-0498: T3 -- UI diagnostic/plot only (if not hit_ms.empty and 'exo_host_name' in hit_ms.columns: / _... (EXCEPT-BULK 2026-07-08)
                pass
    if _exo_host:
        _exo_disp = str(target_row.get("exo_disposition", "") or "").strip()
        _exo_src = str(target_row.get("exo_cat_source", "") or "").strip()
        exo_parts = [f"exo host: **{_exo_host}**"]
        if _exo_src:
            exo_parts.append(f"source={_exo_src}")
        if _exo_disp:
            exo_parts.append(f"disposition={_exo_disp}")
        st.caption(" | ".join(exo_parts))

    if comp_df is not None and not comp_df.empty and "target_catalog_id" in comp_df.columns:
        comp_work = comp_df.copy()
        comp_work["_tcid"] = comp_work["target_catalog_id"].apply(_normalize_gaia_id)
        target_comps = comp_work[comp_work["_tcid"] == _normalize_gaia_id(catalog_id)].copy()

        if not target_comps.empty:
            st.markdown("**Comparison stars**")
            hide_bv = bool(phase01_use_bprp_primary)
            has_bv_tab = any(is_bv_related_phase01_ui_column(c) for c in target_comps.columns)
            if hide_bv and has_bv_tab:
                log_if_ui_hiding_bv_for_bprp_primary(bprp_primary_ui_active=True)
            else:
                log_if_ui_hiding_bv_for_bprp_primary(bprp_primary_ui_active=False)
            cq_path = lc_dir / f"comp_quality_{catalog_id}.json"
            quality_by_cid: dict[str, dict[str, str]] = {}
            excluded_ui_notes: list[str] = []
            comp_quality_tier4_warning = False
            if cq_path.exists():
                try:
                    _cq_raw = json.loads(cq_path.read_text(encoding="utf-8"))
                    if isinstance(_cq_raw, dict):
                        from photometry_core import parse_comp_quality_json_map

                        quality_by_cid = parse_comp_quality_json_map(_cq_raw)
                        comp_quality_tier4_warning = bool(_cq_raw.get("tier4_warning"))
                except Exception:  # noqa: BLE001
                    quality_by_cid = {}
                    comp_quality_tier4_warning = False

            def _cid_short_ui(catalog_id: str) -> str:
                digits = "".join(ch for ch in str(catalog_id) if ch.isdigit())
                return digits[-6:] if len(digits) >= 6 else digits or str(catalog_id)[-6:]

            def _row_bg(q: str) -> str:
                if q == "good":
                    return "background-color:rgba(34,197,94,0.35);"
                if q == "suspect":
                    return "background-color:rgba(234,179,8,0.28);"
                return ""

            def _tier_badge(v: object) -> str:
                s = str(v or "").strip()
                if not s:
                    return "-"
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
            def _format_comp_catalog_id(row_like: pd.Series) -> str:
                bv_src = str(row_like.get("bv_source", "") or "").strip().lower()
                cid0 = _normalize_gaia_id(row_like.get("catalog_id", ""))
                if not cid0:
                    return "-"
                if bv_src in ("gaia_bprp", "gaia_teff", "unknown", ""):
                    return cid0
                if bv_src == "tycho2":
                    tycho_id = str(row_like.get("tycho_id", row_like.get("tycho2_id", "")) or "").strip()
                    if tycho_id and tycho_id.lower() not in ("nan", "none"):
                        return f"TYC {tycho_id}"
                    return cid0
                if bv_src == "apass":
                    apass_id = str(row_like.get("apass_id", "") or "").strip()
                    if apass_id and apass_id.lower() not in ("nan", "none"):
                        return f"AP {apass_id}"
                    return cid0
                return cid0

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
            shown_comp_i = 0
            for _, row in target_comps.iterrows():
                ra_c = _float_coord_row(row, "ra_deg", "ra")
                dec_c = _float_coord_row(row, "dec_deg", "dec")
                mag_c = row.get("mag")
                # Important: keep B-V and BP-RP separate (do not fallback between them).
                bv_c = pd.to_numeric(row.get("b_v"), errors="coerce")
                bv_src_c = str(row.get("bv_source", "") or "").strip().lower()
                bp_c = pd.to_numeric(row.get("bp_rp"), errors="coerce")
                dbprp_c = pd.to_numeric(row.get("delta_bprp_abs"), errors="coerce")
                dist_deg_c = row.get("_dist_deg")
                nfr_c = row.get("comp_n_frames")
                rms_c = row.get("comp_rms")
                w_c = pd.to_numeric(row.get("comp_weight"), errors="coerce")
                tier_c = row.get("comp_tier")
                cid_c = _normalize_gaia_id(row.get("catalog_id", ""))
                q_entry = quality_by_cid.get(cid_c, {}) if cid_c else {}
                q = str(q_entry.get("quality", "") or "").strip().lower()
                q_note = str(q_entry.get("note", "") or "").strip()
                if q == "excluded":
                    excluded_ui_notes.append(
                        f"{_cid_short_ui(cid_c)} ({q_note or 'excluded'})"
                    )
                    continue
                if q == "suspect":
                    stav = f"suspect - {q_note}" if q_note else "suspect"
                elif q == "good":
                    stav = "good"
                else:
                    stav = "-"

                viz_c = (
                    f"https://vizier.cds.unistra.fr/viz-bin/VizieR?"
                    f"&-c={ra_c:.6f}{dec_c:+.6f}&-c.rs=2"
                )
                mag_str = _fmt_opt_num(mag_c, ".3f")
                catalog_id_str = _format_comp_catalog_id(row)
                bv_str = _fmt_opt_num(bv_c, ".3f")
                bv_src_str = (bv_src_c or "unknown")
                bp_str = _fmt_opt_num(bp_c, ".3f")
                dbprp_str = _fmt_opt_num(dbprp_c, ".3f")
                tier_color_raw = row.get("color_tier_src", "")
                dist_str = _fmt_opt_num(dist_deg_c, ".6f")
                nfr_str = _fmt_opt_num(nfr_c, ".0f")
                rms_str = _fmt_opt_num(rms_c, ".4f")
                wrel_str = (
                    f"{float(w_c) / max_w:.3f}"
                    if np.isfinite(pd.to_numeric(w_c, errors="coerce")) and float(pd.to_numeric(w_c, errors="coerce")) > 0 and np.isfinite(max_w) and max_w > 0
                    else "-"
                )
                def _bv_src_badge(src: str) -> str:
                    key = str(src or "").strip().lower() or "unknown"
                    label = key
                    bg = {
                        "gaia_bprp": "rgba(34,197,94,0.25)",   # green
                        "gaia_teff": "rgba(59,130,246,0.25)",  # blue
                        "apass": "rgba(249,115,22,0.25)",      # orange
                        "tycho2": "rgba(234,179,8,0.25)",      # yellow
                        "unknown": "rgba(148,163,184,0.25)",   # gray
                    }.get(key, "rgba(148,163,184,0.25)")
                    fg = "rgba(15,23,42,1.0)"
                    return (
                        "<span style=\"display:inline-block;padding:2px 6px;border-radius:999px;"
                        f"background-color:{bg};color:{fg};font-weight:600;font-size:0.85rem;\">"
                        f"{html.escape(label)}</span>"
                    )

                def _tier_color_src_badge(raw: object) -> str:
                    """Phase 1 tier color: native BP-RP vs B-V->BP-RP fallback (same space as DeltaBPRP)."""
                    key = str(raw or "").strip().lower()
                    if not key:
                        return "-"
                    label = {
                        "bprp": "bp_rp",
                        "bv_converted": "BV->BP-RP",
                        "unknown": "unknown",
                    }.get(key, key)
                    bg = {
                        "bprp": "rgba(34,197,94,0.25)",
                        "bv_converted": "rgba(234,179,8,0.28)",
                        "unknown": "rgba(148,163,184,0.25)",
                    }.get(key, "rgba(148,163,184,0.25)")
                    fg = "rgba(15,23,42,1.0)"
                    return (
                        "<span style=\"display:inline-block;padding:2px 6px;border-radius:999px;"
                        f"background-color:{bg};color:{fg};font-weight:600;font-size:0.85rem;\">"
                        f"{html.escape(label)}</span>"
                    )

                shown_comp_i += 1
                bg = _row_bg(q)
                cells: list[str] = [
                    f"<td>C{shown_comp_i:02d}</td>",
                    f"<td>{html.escape(catalog_id_str)}</td>",
                    f"<td>{html.escape(mag_str)}</td>",
                ]
                if not hide_bv:
                    cells.append(f"<td>{html.escape(bv_str)}</td>")
                    cells.append(f"<td>{_bv_src_badge(bv_src_str)}</td>")
                cells.extend(
                    [
                        f"<td>{html.escape(bp_str)}</td>",
                        f"<td><strong>{html.escape(dbprp_str)}</strong></td>",
                        f"<td>{_tier_color_src_badge(tier_color_raw)}</td>",
                        f"<td>{html.escape(dist_str)}</td>",
                        f"<td>{html.escape(nfr_str)}</td>",
                        f"<td>{html.escape(rms_str)}</td>",
                        f"<td title=\"Relative weight 1/sigma^2 (Broeg 2005)\">{html.escape(wrel_str)}</td>",
                        f"<td>{_tier_badge(tier_c)}</td>",
                        f"<td>{html.escape(stav)}</td>",
                        f"<td><a href=\"{html.escape(viz_c)}\" target=\"_blank\" rel=\"noopener noreferrer\">/</a></td>",
                    ]
                )
                rows_html.append("<tr style=\"" + bg + "\">" + "".join(cells) + "</tr>")

            if hide_bv:
                thead = (
                    "<thead><tr>"
                    "<th>#</th><th>catalog_id</th><th>mag</th><th>bp_rp</th>"
                    "<th title=\"Primary Phase 1 color filter (|DeltaBP-RP| in Gaia space)\">DeltaBPRP</th>"
                    "<th title=\"Tier color source: native Gaia bp_rp vs conversion from B-V\">tier color</th>"
                    "<th>dist_deg</th><th>comp_n_frames</th><th>comp_rms</th>"
                    "<th title=\"Relative weight 1/sigma^2 (Broeg 2005)\">w (rel)</th>"
                    "<th>tier</th><th>status</th><th>Vizier</th>"
                    "</tr></thead>"
                )
            else:
                thead = (
                    "<thead><tr>"
                    "<th>#</th><th>catalog_id</th><th>mag</th><th>B-V</th><th>B-V src</th><th>bp_rp</th>"
                    "<th title=\"Primary Phase 1 color filter (|DeltaBP-RP| in Gaia space)\">DeltaBPRP</th>"
                    "<th title=\"Tier color source: native Gaia bp_rp vs conversion from B-V\">tier color</th>"
                    "<th>dist_deg</th><th>comp_n_frames</th><th>comp_rms</th>"
                    "<th title=\"Relative weight 1/sigma^2 (Broeg 2005)\">w (rel)</th>"
                    "<th>tier</th><th>status</th><th>Vizier</th>"
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
            if excluded_ui_notes:
                st.caption(
                    f"{len(excluded_ui_notes)} comp star(s) excluded from ensemble: "
                    + ", ".join(excluded_ui_notes)
                )
            if comp_quality_tier4_warning:
                st.info(
                    "Phase 1: the ensemble also includes comp stars **TIER3/4** (larger color difference vs target - "
                    "primarily **|DeltaBP-RP|**). Column **tier** = color class from Phase 1; **status** = LC stability from Phase 2A (good/suspect)."
                )
            if not quality_by_cid:
                st.caption(
                    "Status (good / suspect / excluded) will appear after the next Phase 2A run "
                    "(comp_quality_*.json file)."
                )
            if hide_bv:
                st.caption(
                    "**Phase 1 color selection** uses Gaia **BP-RP** space (column **DeltaBPRP** = |DeltaBP-RP| vs target; bold). "
                    "**tier color** = native **bp_rp** or fallback **B-V->BP-RP**. "
                    "**bp_rp** from Gaia when ID is known. **Vizier** column = sky link (/)."
                )
            else:
                st.caption(
                    "**Phase 1 color selection** uses Gaia **BP-RP** space (column **DeltaBPRP** = |DeltaBP-RP| vs target; bold). "
                    "**tier color** = native **bp_rp** or fallback **B-V->BP-RP**. "
                    "Johnson **B-V** and **B-V src** are informational (report / compatibility). "
                    "**bp_rp** from Gaia when ID is known. **Vizier** column = sky link (/)."
                )


# ---------------------------------------------------------------------------
# Hlavny render
# ---------------------------------------------------------------------------


def render_aperture_photometry(
    cfg: "AppConfig",
    draft_id: int | None,
    pipeline: "AstroPipeline",
    *,
    draft_dir_override: Path | None = None,
) -> None:
    """Hlavna funkcia pre Aperture Photometry tab."""
    _ = pipeline
    st.header("Aperture Photometry")
    st.caption("Phase 0+1 + Phase 2A as one integrated step.")
    st.session_state.setdefault("var_analysis_done", False)
    st.session_state.setdefault("var_analysis_timestamp", None)
    st.session_state.setdefault("pdf_ready", False)

    if draft_id is None and draft_dir_override is None:
        st.info("No active draft. Load a draft above or run VAR-STREM.")
        return

    # Draft dir for PDF reports and fallbacks.
    draft_dir = resolve_draft_dir_path(
        draft_dir_override, draft_id, cfg.archive_root
    )
    if draft_dir is None:
        st.warning("No active draft. Load a draft above or run VAR-STREM.")
        return

    all_setups = _find_phase2a_paths(cfg, draft_id, draft_dir_override=draft_dir)
    if not all_setups:
        st.warning("Input files not found.")
        return

    setup_options = list(all_setups.keys())

    def _detect_obs_groups() -> list[str]:
        """Obs_groups from detrended_aligned/lights/{obs_group}/ (proc_*.csv)."""
        try:
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
            # EXC-0499: T3 -- UI diagnostic/plot only (out.append(d.name) / return out / except Exception:  # noqa: B... (EXCEPT-BULK 2026-07-08)
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
        st.warning("Selected setup has no valid paths.")
        return

    output_dir = paths.get("output_dir")
    ms_for_fwhm = paths.get("masterstar_fits")
    if not (isinstance(ms_for_fwhm, Path) and ms_for_fwhm.is_file()):
        ms_for_fwhm = _fallback_masterstar_fits(cfg, draft_id, draft_dir_override=draft_dir_override)
    fwhm_px = _load_fwhm(ms_for_fwhm)

    if isinstance(output_dir, Path):
        _fd_p = output_dir / "field_density.json"
        if _fd_p.is_file():
            try:
                _fd = json.loads(_fd_p.read_text(encoding="utf-8"))
                _d_mpx = float(_fd.get("density_h_star_per_mpx", 0) or 0)
                _d_cls = str(_fd.get("density_class", "?"))
                st.info(f"Field density: {_d_mpx:.0f} stars/Mpx ({_d_cls})")
            except Exception:  # noqa: BLE001
                # EXC-0500: T3 -- UI diagnostic/plot only (_d_cls = str(_fd.get('density_class', '?')) / st.info(f'Field ... (EXCEPT-BULK 2026-07-08)
                pass

    exists = _phase2a_results_exist(output_dir)

    if st.session_state.get("pdf_ready"):
        _ps = Path(draft_dir) / "platesolve" / str(selected_setup) / "photometry"
        _ps.mkdir(parents=True, exist_ok=True)
        with st.spinner("Generating PDF report..."):
            from pdf_report import generate_report

            _draft_lbl = Path(str(draft_dir).rstrip("/\\")).name
            try:
                _did = int(str(_draft_lbl).split("draft_", 1)[1])
                _draft_id_str = f"draft_{_did:06d}"
            except Exception:  # noqa: BLE001
                _draft_id_str = _draft_lbl
            pdf_path = generate_report(
                photometry_dir=str(_ps.resolve()),
                setup_name=str(selected_setup),
                draft_id=_draft_id_str,
                var_results=st.session_state.get("var_results"),
                candidates=st.session_state.get("var_candidates"),
                crossmatch_bullets=st.session_state.get("var_catalog_bullets", {}),
                accepted_periods=st.session_state.get("accepted_period", {}),
                variability_timestamp=st.session_state.get("var_analysis_timestamp"),
                tess_results=st.session_state.get("tess_results", {}),
                report_title="VYVAR - Summary Measure Report",
            )
        if pdf_path and Path(pdf_path).exists():
            st.success("PDF report generated.")
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF report",
                    data=f.read(),
                    file_name=Path(pdf_path).name,
                    mime="application/pdf",
                    key=f"pdf_gen_dl_{selected_setup}",
                )
        else:
            st.warning("PDF report could not be generated (missing data or reportlab).")
        st.session_state["pdf_ready"] = False

    # Header/status line + global run button.
    col_info, col_run = st.columns([3, 2])
    with col_info:
        st.markdown(f"**Platesolve setup:** `{selected_setup}` &nbsp; | &nbsp; **FWHM:** `{float(fwhm_px):.3f}px`")
        if exists:
            p2a_ts = _phase2a_timestamp(output_dir)
            if st.session_state.get("var_analysis_done"):
                vts = str(st.session_state.get("var_analysis_timestamp") or "")
                st.success(f"Completed: {vts} - Variability Detection finished automatically")
                st.caption(f"Phase 2A (photometry): {p2a_ts}")
            else:
                st.success(f"[OK] Completed: {p2a_ts}")
            if st.session_state.get("var_analysis_done"):
                if (
                    st.session_state.get("crossmatch_auto_done") is True
                    and st.session_state.get("tess_auto_done") is True
                ):
                    if st.button(
                        "Generate PDF report",
                        type="primary",
                        width="stretch",
                        key=f"vyvar_gen_pdf_{selected_setup}",
                    ):
                        st.session_state["pdf_ready"] = True
                        st.rerun()
                else:
                    st.info(
                        "... Crossmatch and TESS analysis in progress - PDF will be available when finished."
                    )
            # Always show PDF download if it already exists (even after rerun).
            pdf_latest = _latest_report_pdf(draft_dir, str(selected_setup))
            if pdf_latest is not None and pdf_latest.exists():
                try:
                    with open(pdf_latest, "rb") as f:
                        st.download_button(
                            label=f"Download PDF report ({selected_setup})",
                            data=f.read(),
                            file_name=pdf_latest.name,
                            mime="application/pdf",
                            key=f"pdf_dl_hdr_{selected_setup}",
                        )
                except Exception:  # noqa: BLE001
                    # EXC-0501: T3 -- UI diagnostic/plot only (key=f'pdf_dl_hdr_{selected_setup}', / ) / except Exception:  #... (EXCEPT-BULK 2026-07-08)
                    pass
        else:
            st.warning("! Not started")
    with col_run:
        run_btn = st.button("[refresh] RUN Aperture Photometry", key="phase2a_run_full", type="primary")

    with st.expander("Setup status", expanded=False):
        for nm in setup_options:
            p = all_setups.get(str(nm)) or {}
            out_d = p.get("output_dir")
            if _phase2a_results_exist(out_d):
                st.success(f"[OK] {nm}: {_phase2a_timestamp(out_d)}")
            else:
                st.warning(f"! {nm}: not started")

    # Stale lock: beh bol preruseny (timeout / kill) pred `finally` -> inak ostane True a UI sa zasekne.
    if st.session_state.get("phase2a_running") and not run_btn:
        st.session_state["phase2a_running"] = False

    if run_btn:
        from photometry_core import run_full_photometry_pipeline
        from vyvar_ui_status import vyvar_footer_idle, vyvar_footer_running

        try:
            st.session_state["phase2a_running"] = True
            st.session_state["_ap_session_id"] = st.session_state.get("_current_session_id")
            total = len(run_groups)
            if total <= 0:
                st.warning("No obs_group found in detrended_aligned/lights.")
                return
            vyvar_footer_running("Aperture Photometry", f"Starting ({total} setups)...")
            prog = st.progress(0, text="Starting...")
            lines_ph = st.empty()
            statuses: dict[str, str] = {}
            errors: list[str] = []
            n_ok = 0

            def _render_lines() -> None:
                lines = [statuses.get(g, f"{g} ...") for g in run_groups]
                lines_ph.markdown("\n".join(lines))

            for i, nm in enumerate(run_groups, start=1):
                statuses[nm] = f"{nm} ###### ..."
                _render_lines()
                prog.progress(int(round(100 * (i - 1) / max(total, 1))), text=f"{nm}: starting...")
                vyvar_footer_running("Aperture Photometry", f"{nm}: Phase 0+1 + 2A...")

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
                        missing.append("per-frame CSV directory")
                    if dt_dir is None or not dt_dir.exists():
                        missing.append("detrended_aligned directory")
                    if out_d is None:
                        missing.append("output_dir")
                    if missing:
                        raise FileNotFoundError(", ".join(missing))

                    def _cb(msg: str, nm=nm) -> None:
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
                    _load_summary.clear()
                    logging.debug("[PERF-7] _load_summary cache cleared after pipeline run")
                    n_ok += 1
                    statuses[nm] = f"{nm} ############ [OK]"

                    # PDF report (optional)
                    try:
                        from photometry_report import generate_photometry_report

                        pdf_path = generate_photometry_report(
                            draft_dir=draft_dir,
                            obs_group=str(nm),
                            output_pdf=None,
                            tess_results=st.session_state.get("tess_results", {}),
                            report_title="VYVAR - Summary Measure Report",
                        )
                        if pdf_path is not None:
                            st.success(f"[page] PDF saved: {Path(pdf_path).name}")
                            try:
                                st.session_state.setdefault("vyvar_pdf_paths", {})[str(nm)] = str(pdf_path)
                            except Exception:  # noqa: BLE001
                                # EXC-0502: T3 -- UI diagnostic/plot only (try: / st.session_state.setdefault('vyvar_pdf_paths', {})[str(... (EXCEPT-BULK 2026-07-08)
                                pass
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label=f"[inbox] Download {Path(pdf_path).name}",
                                    data=f.read(),
                                    file_name=Path(pdf_path).name,
                                    mime="application/pdf",
                                    key=f"pdf_download_{nm}",
                                )
                    except Exception as _pdf_exc:  # noqa: BLE001
                        st.warning(f"PDF could not be generated: {_pdf_exc}")
                except Exception as exc_nm:  # noqa: BLE001
                    statuses[nm] = f"{nm} ####### x"
                    errors.append(f"{nm}: {exc_nm}")
                _render_lines()
                prog.progress(int(round(100 * i / max(total, 1))), text=f"{nm}: done")

            if n_ok:
                st.success(f"Done - {n_ok} setups processed")
            if errors:
                (st.error if n_ok == 0 else st.warning)(
                    "Issues with some setups:\n" + "\n".join(errors)
                )
            if len(errors) == 0 and n_ok == total and n_ok > 0 and not st.session_state.get("var_analysis_done"):
                st.info("Phase 2A complete - starting Variability Detection...")
                try:
                    from ui_variability import run_variability_detection_session

                    setup_for_var = str(
                        st.session_state.get("var_obs_group")
                        or st.session_state.get("phase2a_setup_select")
                        or selected_setup
                    )
                    flux_col_v = str(st.session_state.get("var_flux_source", "dao_flux"))
                    min_pct_v = 100
                    cfg_dct = cfg.to_dict()
                    sigma_v = float(
                        st.session_state.get("var_sigma_thr", cfg_dct.get("variability_sigma_threshold", 2.3))
                    )
                    mag_v = float(cfg_dct.get("variability_mag_limit", 14.5) or 14.5)
                    results_v, n_cand_v, var_sig_v = run_variability_detection_session(
                        cfg=cfg,
                        draft_dir=draft_dir,
                        obs_group=setup_for_var,
                        flux_col=flux_col_v,
                        min_frames_pct=min_pct_v,
                        sigma_thr=sigma_v,
                        mag_limit=mag_v,
                    )
                    st.session_state["var_results"] = results_v
                    st.session_state["_var_run_sig"] = var_sig_v
                    st.session_state.pop("crossmatch_auto_done", None)
                    st.session_state["var_obs_group"] = setup_for_var
                    st.session_state["var_analysis_done"] = True
                    st.session_state["var_analysis_timestamp"] = datetime.now().strftime("%d.%m.%Y %H:%M")
                    st.session_state["pdf_ready"] = False
                    st.session_state["var_candidate_count_autorun"] = int(n_cand_v)
                except Exception as _v_exc:  # noqa: BLE001
                    logging.exception("Auto Variability Detection po 2A zlyhala")
                    st.warning(f"Variability Detection could not be started automatically: {_v_exc}")
            # Trigger crossmatch/TESS status messaging if automation not completed yet.
            if st.session_state.get("var_analysis_done") and not st.session_state.get("crossmatch_auto_done"):
                st.info("... Catalog crossmatch in progress...")
            elif (
                st.session_state.get("var_analysis_done")
                and st.session_state.get("crossmatch_auto_done")
                and not st.session_state.get("tess_auto_done")
            ):
                st.info("... TESS analysis in progress - PDF will be available when finished.")
            vyvar_footer_idle()
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"[X] Error: {exc}")
            logging.exception("RUN Aperture Photometry zlyhal")
        finally:
            st.session_state["phase2a_running"] = False
            try:
                vyvar_footer_idle()
            except Exception:  # noqa: BLE001
                # EXC-0503: T3 -- UI diagnostic/plot only (try: / vyvar_footer_idle() / except Exception:  # noqa: BLE001... (EXCEPT-BULK 2026-07-08)
                pass
        return

    # ------------------------------------------------------------------
    # Block B: Auto pipeline (crossmatch + TESS), separate from Phase 2A
    # ------------------------------------------------------------------
    if (
        st.session_state.get("var_analysis_done")
        and st.session_state.get("_ap_session_id") == st.session_state.get("_current_session_id")
        and not st.session_state.get("tess_auto_done")
    ):
        # Crossmatch
        if not st.session_state.get("crossmatch_auto_done"):
            candidates = st.session_state.get("var_candidates", [])
            if candidates:
                from catalog_crossmatch import check_candidate_in_catalogs  # noqa: PLC0415
                from ui_variability import _get_candidate_row  # noqa: PLC0415

                _vsx_db_ap = str(getattr(cfg, "vsx_local_db_path", "") or "").strip() or None
                bullets_map = st.session_state.get("var_catalog_bullets", {})
                missing = [c for c in candidates if str(c) not in {str(k) for k in bullets_map}]
                if missing:
                    pb = st.progress(0, text="Running catalog crossmatch...")
                    xr = st.session_state.setdefault("var_crossmatch_results", {})
                    for i, cid in enumerate(missing):
                        row = _get_candidate_row(
                            st.session_state.get("var_results"),
                            cid,
                            draft_dir=draft_dir,
                            platesolve_dir=draft_dir / "platesolve" / str(selected_setup),
                        )
                        if row:
                            try:
                                cr = check_candidate_in_catalogs(
                                    ra=float(row["ra"]),
                                    dec=float(row["dec"]),
                                    mag=row.get("mag"),
                                    radius_arcsec=10.0,
                                    vsx_local_db_path=_vsx_db_ap,
                                )
                                b = cr.catalog_summary_bullets()
                                bullets_map[str(cid)] = "\n".join(b) if b else "-"
                                xr[str(cid)] = cr
                            except Exception as exc:  # noqa: BLE001
                                bullets_map[str(cid)] = f"Error: {exc}"
                        else:
                            bullets_map[str(cid)] = "-"
                        pb.progress((i + 1) / len(missing))
                    pb.empty()
                    st.session_state["var_catalog_bullets"] = bullets_map
                # TODO-17: do not mark done when there are no candidates (was unconditional)
                st.session_state["crossmatch_auto_done"] = True
                st.rerun()

        # TESS (one candidate per rerun)
        if not bool(getattr(cfg, "tess_enabled", False)):
            if not st.session_state.get("tess_auto_done"):
                logging.info("[TESS] preskocene - tess_enabled=False (Aperture Photometry auto vetva)")
                st.session_state["tess_auto_done"] = True
                st.rerun()
        else:
            from ui_variability import (  # noqa: PLC0415
                _get_candidate_row,
                _should_trigger_tess,
                tess_catalog_ids_for_auto_run,
            )
            from tess_verify import run_tess_analysis  # noqa: PLC0415

            candidates = st.session_state.get("var_candidates", [])
            bullets_map = st.session_state.get("var_catalog_bullets", {})
            tess_results = st.session_state.get("tess_results", {})
            photometry_dir = st.session_state.get("var_photometry_dir")
            _obs_for_tess = str(
                st.session_state.get("var_obs_group")
                or st.session_state.get("phase2a_setup_select")
                or selected_setup
            )
            _memory_cids = [str(x).strip() for x in candidates if str(x).strip()]
            _cid_rows = tess_catalog_ids_for_auto_run(draft_dir, _obs_for_tess, _memory_cids)
            _done_tess = {str(k) for k in (tess_results or {})}
            to_tess = [
                c
                for c in _cid_rows
                if c not in _done_tess
                and _should_trigger_tess(bullets_map.get(c, "-"))
                and photometry_dir
            ]
            if to_tess:
                cid = to_tess[0]
                _need_tess = [c for c in _cid_rows if _should_trigger_tess(bullets_map.get(c, "-"))]
                total = len(_need_tess)
                done = len([c for c in _need_tess if str(c) in _done_tess])
                st.info(f"[telescope] TESS: {done}/{total} - processing {str(cid)[:16]}...")

                row = _get_candidate_row(
                    st.session_state.get("var_results"),
                    cid,
                    draft_dir=draft_dir,
                    platesolve_dir=draft_dir / "platesolve" / str(selected_setup),
                )
                if row:
                    cr = st.session_state.get("var_crossmatch_results", {}).get(str(cid))
                    period_hint = cr.best_period() if cr is not None and hasattr(cr, "best_period") else None
                    try:
                        tres = run_tess_analysis(
                            catalog_id=str(cid),
                            ra=float(row["ra"]),
                            dec=float(row["dec"]),
                            mag=row.get("mag"),
                            photometry_dir=photometry_dir,
                            period_hint=period_hint,
                            cfg=cfg,
                        )
                        tess_results[str(cid)] = tres
                    except Exception as exc:  # noqa: BLE001
                        tess_results[str(cid)] = None
                        st.warning(f"TESS failed: {str(cid)[:16]}: {exc}")
                else:
                    tess_results[str(cid)] = None

                st.session_state["tess_results"] = tess_results
                st.rerun()
            else:
                st.session_state["tess_auto_done"] = True
                st.rerun()

    if not exists:
        st.warning(
            f"! Photometry was not run for **{selected_setup}**.\n\n"
            "Click **RUN Aperture Photometry**."
        )
        return

    if output_dir is None:
        st.warning("Output directory is not available.")
        return

    # Phase01 artifacts (generated by full pipeline and used for UI enrich/table).
    at_csv = paths.get("active_targets_csv")
    comp_csv = paths.get("comparison_stars_csv")

    summary_df = _load_summary(str(output_dir / "photometry_summary.csv"))
    comp_df = pd.DataFrame()
    if comp_csv is not None and Path(comp_csv).exists():
        comp_df = pd.read_csv(comp_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)

    if summary_df.empty:
        st.info("No results yet.")
        return

    at_path_for_zf = Path(at_csv) if at_csv is not None else None
    summary_df = _enrich_summary_with_zone_flags(summary_df, at_path_for_zf)

    tab_lc, tab_hrd = st.tabs(["Photometry / LC", "[star] Field HRD"])
    with tab_lc:
        _n_total = int(len(summary_df))
        _lc_qopts = ["good", "noisy", "noisy_moon", "short_baseline", "no_data", "saturated"]
        _lc_qdefault = ["good", "noisy", "noisy_moon", "short_baseline"]
        if "vyvar_lc_quality_filter" not in st.session_state:
            st.session_state["vyvar_lc_quality_filter"] = list(_lc_qdefault)
        st.multiselect(
            "LC quality filter",
            options=_lc_qopts,
            default=_lc_qdefault,
            key="vyvar_lc_quality_filter",
        )
        _has_lc_q = "lc_quality_flag" in summary_df.columns
        if _has_lc_q:
            _sel_q = st.session_state.get("vyvar_lc_quality_filter", _lc_qdefault)
            if not _sel_q:
                _sel_q = _lc_qdefault
            _before_q = summary_df.copy()
            summary_df = summary_df[
                summary_df["lc_quality_flag"].astype(str).str.strip().str.lower().isin(_sel_q)
            ].copy()
            _hidden = _before_q.loc[~_before_q.index.isin(summary_df.index)]
            if not _hidden.empty:
                for _, _hr in _hidden.iterrows():
                    _hn = str(_hr.get("vsx_name", _hr.get("catalog_id", "")) or "").strip()
                    _hq = str(_hr.get("lc_quality_flag", "") or "").strip()
                    logging.info(
                        "[UI LC] Hidden target %s - lc_quality_flag=%s (filter=%s)",
                        _hn,
                        _hq or "?",
                        ",".join(_sel_q),
                    )
            st.caption(f"Showing {len(summary_df)} / {_n_total} targets")
        else:
            st.caption(
                "Quality filter unavailable - re-run Phase 2A to generate lc_quality_flag"
            )

        if summary_df.empty:
            st.info("No targets match the LC quality filter.")
            return

        _n_sum = int(len(summary_df))
        _idx_opts = list(range(_n_sum))
        _sel_i = st.selectbox(
            "Select variable star:",
            options=_idx_opts,
            format_func=lambda i: _phase2a_target_choice_label(summary_df.iloc[int(i)]),
            key="phase2a_target_select",
        )
        target_row = summary_df.iloc[int(_sel_i)]
        try:
            catalog_id = str(target_row.get("catalog_id", ""))
        except Exception:  # noqa: BLE001
            catalog_id = ""

        col1, col2, col3 = st.columns([1, 2, 2])
        with col1:
            show_detrended = st.toggle(
                "Airmass detrend",
                value=True,
                key="toggle_am_detrend",
            )
            show_airmass = st.toggle(
                "Show airmass on chart",
                value=True,
                key="toggle_show_airmass",
            )
        with col2:
            show_outliers = st.toggle(
                "Show outlier and saturated points",
                value=True,
                key="phase2a_show_outliers",
            )
        with col3:
            preload_all = st.toggle(
                "Load all light curves into memory",
                value=False,
                key="phase2a_preload_all_curves",
            )

        if preload_all:
            lc_dir = output_dir / "lightcurves"
            if lc_dir.is_dir():
                lc_files = sorted(lc_dir.glob("lightcurve_*.csv"))
                total_lc_count = len(lc_files)
                lc_files = lc_files[:_MAX_LC_PRELOAD]
                if total_lc_count > _MAX_LC_PRELOAD:
                    st.info(
                        f"Showing first {_MAX_LC_PRELOAD} of {total_lc_count} light curves. "
                        "Use target search to load others."
                    )
                with st.spinner("Loading light curves into memory..."):
                    try:
                        for p in lc_files:
                            _ = _cached_read_csv(str(p))
                    except Exception:  # noqa: BLE001
                        # EXC-0504: T3 -- UI diagnostic/plot only (for p in lc_files: / _ = _cached_read_csv(str(p)) / except Exc... (EXCEPT-BULK 2026-07-08)
                        pass

        show_all_filters = st.checkbox(
            "Show all filters in one chart",
            value=False,
            key="phase2a_show_all_filters",
        )

        _phase01_bprp_pri = bool(cfg.phase01_use_bprp_primary)

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
                    lc_df = _cached_read_csv(str(lc_csv))

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
                    title=f"Light curves - {catalog_id}",
                    xaxis_title=jd_axis_title("BJD (TDB)", overlay_x_off),
                    yaxis_title="mag (calib)",
                    yaxis_autorange="reversed",
                    legend_title="Filter",
                    height=500,
                )
                st.plotly_chart(fig, width="stretch")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Overlay chart failed: {exc}")
                _render_target_detail(
                    target_row,
                    output_dir,
                    show_outliers,
                    comp_df=comp_df,
                    show_detrended=show_detrended,
                    show_airmass=show_airmass,
                    phase01_use_bprp_primary=_phase01_bprp_pri,
                )
        else:
            _render_target_detail(
                target_row,
                output_dir,
                show_outliers,
                comp_df=comp_df,
                show_detrended=show_detrended,
                show_airmass=show_airmass,
                phase01_use_bprp_primary=_phase01_bprp_pri,
            )

    with tab_hrd:
        from ui_hrd import render_hrd_tab

        render_hrd_tab(Path(output_dir), cfg)

    st.divider()

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Light curves", int(len(summary_df)))

    rms_cur = pd.to_numeric(summary_df.get("lc_rms"), errors="coerce")
    if rms_cur.notna().any():
        c2.metric("Median lc_rms", f"{float(rms_cur.median()):.4f}")
        good = int((rms_cur < 0.05).sum())
        c3.metric("RMS < 0.05 mag", good)
    else:
        c2.metric("Median lc_rms", "-")
        c3.metric("RMS < 0.05 mag", "-")

    ngc = pd.to_numeric(summary_df.get("n_good_comp"), errors="coerce")
    c4.metric("Avg good comp", f"{float(ngc.mean()):.1f}" if ngc.notna().any() else "-")

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
            df0 = (
                _load_summary(str(Path(out_d) / "photometry_summary.csv"))
                if out_d is not None
                else pd.DataFrame()
            )
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
        c6.metric("Best lc_rms", f"{float(rms_all.min()):.4f}" if rms_all.notna().any() else "-")
        c7.metric("Worst lc_rms", f"{float(rms_all.max()):.4f}" if rms_all.notna().any() else "-")
        bp_all = pd.to_numeric(all_df.get("bp_rp"), errors="coerce")
        c8.metric("Avg bp_rp", f"{float(bp_all.mean()):.3f}" if bp_all.notna().any() else "-")
    else:
        c6.metric("Best lc_rms", "-")
        c7.metric("Worst lc_rms", "-")
        c8.metric("Avg bp_rp", "-")
