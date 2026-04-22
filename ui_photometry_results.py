"""Streamlit tabs: aperture / PSF time series, comp validation, variable detection (per-frame sidecars)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from infolog import log_event
from jd_axis_format import jd_axis_title, jd_series_relative

_SKIP_CSV_NAMES = frozenset(
    {
        "per_frame_catalog_index.csv",
        "field_catalog_cone.csv",
        "comparison_stars.csv",
        "variable_targets.csv",
    }
)

# Per-frame photometry CSVs under detrended_aligned/lights (exclude pipeline aggregates / QA).
_LOAD_PER_FRAME_SIDECAR_SKIP = frozenset(
    {
        "masterstars_full_match.csv",
        "comparison_stars.csv",
        "variable_targets.csv",
        "field_catalog_cone.csv",
        "masterstars.csv",
        "qc_metrics.csv",
        "alignment_report.csv",
        "per_frame_catalog_index.csv",
    }
)


def _draft_archive_path(pipeline: Any, draft_id: int | None) -> Path | None:
    if draft_id is None:
        return None
    try:
        row = pipeline.db.conn.execute(
            "SELECT ARCHIVE_PATH FROM OBS_DRAFT WHERE ID = ?;",
            (int(draft_id),),
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    if row is None or row[0] is None:
        return None
    s = str(row[0]).strip()
    if not s:
        return None
    return Path(s).expanduser().resolve()


def _cid_key(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, str) and not v.strip():
        return ""
    try:
        f = float(v)
        if math.isfinite(f) and abs(f) > 1e10:
            return str(int(f))
        if math.isfinite(f) and float(int(f)) == f:
            return str(int(f))
    except (TypeError, ValueError, OverflowError):
        pass
    return str(v).strip()


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "y")


def _jd_from_fits_header(path: Path) -> float | None:
    try:
        from astropy.io import fits
        from astropy.time import Time

        with fits.open(path, memmap=False) as hdul:
            hdr = hdul[0].header
        for k in ("MJD-OBS", "MJD_OBS"):
            if k in hdr:
                try:
                    mjd = float(hdr[k])
                    if math.isfinite(mjd):
                        return mjd + 2400000.5
                except (TypeError, ValueError):
                    pass
        for k in ("JD-OBS", "JD_OBS", "JD"):
            if k in hdr:
                try:
                    jd = float(hdr[k])
                    if math.isfinite(jd):
                        return jd
                except (TypeError, ValueError):
                    pass
        for k in ("DATE-OBS", "DATEOBS"):
            if k not in hdr:
                continue
            raw = hdr[k]
            if hasattr(raw, "strip"):
                ds = str(raw).strip()
            else:
                ds = str(raw)
            if not ds:
                continue
            ds = ds.replace("T", " ", 1)
            try:
                t = Time(ds, format="isot", scale="utc")
                return float(t.jd)
            except Exception:  # noqa: BLE001
                try:
                    t = Time(ds.split("T")[0], scale="utc")
                    return float(t.jd)
                except Exception:  # noqa: BLE001
                    pass
    except Exception:  # noqa: BLE001
        return None
    return None


def _collect_per_frame_csv_paths(archive: Path) -> list[Path]:
    proc = archive / "processed"
    if not proc.is_dir():
        return []
    found: set[Path] = set()
    for p in proc.rglob("*.csv"):
        if not p.is_file():
            continue
        if p.name in _SKIP_CSV_NAMES:
            continue
        low = p.name.lower()
        if "masterstars" in low and "platesolve" in str(p).lower():
            continue
        if p.name.endswith("_catalog.csv"):
            found.add(p.resolve())
            continue
        stem = p.stem
        parent = p.parent
        for sfx in (".fits", ".fit", ".FIT", ".FITS"):
            if (parent / f"{stem}{sfx}").is_file():
                found.add(p.resolve())
                break
    return sorted(found)


def _normalize_sidecar_dataframe(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    out = df.copy()
    out["filename"] = csv_path.stem

    if "aperture_flux" not in out.columns and "flux" in out.columns:
        out["aperture_flux"] = pd.to_numeric(out["flux"], errors="coerce")
    if "aperture_flux_err" not in out.columns:
        out["aperture_flux_err"] = np.nan

    aflux = pd.to_numeric(out.get("aperture_flux"), errors="coerce")
    if "aperture_mag" not in out.columns:
        if "mag" in out.columns:
            out["aperture_mag"] = pd.to_numeric(out["mag"], errors="coerce")
        else:
            out["aperture_mag"] = np.where(
                aflux > 0,
                -2.5 * np.log10(np.maximum(aflux, 1e-30)),
                np.nan,
            )
    if "aperture_mag_err" not in out.columns:
        out["aperture_mag_err"] = np.nan

    pf = pd.to_numeric(out.get("psf_flux"), errors="coerce")
    if "psf_mag" not in out.columns:
        out["psf_mag"] = np.where(
            pf.notna() & (pf > 0),
            -2.5 * np.log10(np.maximum(pf, 1e-30)),
            np.nan,
        )
    if "psf_mag_err" not in out.columns:
        pfe = pd.to_numeric(out.get("psf_flux_err"), errors="coerce")
        out["psf_mag_err"] = np.where(
            pfe.notna() & (pf > 0) & (pfe > 0),
            (2.5 / math.log(10.0)) * (pfe / pf),
            np.nan,
        )

    for col in ("vsx_known_variable", "catalog_known_variable", "gaia_dr3_variable_catalog"):
        if col in out.columns:
            out[col] = out[col].map(_to_bool)
        else:
            out[col] = False

    if "psf_fit_ok" in out.columns:
        out["psf_fit_ok"] = out["psf_fit_ok"].map(_to_bool)
    else:
        out["psf_fit_ok"] = False

    jd_col = None
    for _jdc in ("jd_mid", "jd", "inspection_jd", "hjd_mid"):
        if _jdc in out.columns:
            jd_col = _jdc
            break
    if jd_col:
        out["_jd_sort"] = pd.to_numeric(out[jd_col], errors="coerce")
    else:
        out["_jd_sort"] = np.nan

    src = out["source_file"].iloc[0] if "source_file" in out.columns and len(out) else None
    if isinstance(src, str) and src.strip():
        fits_path = csv_path.parent / src.strip()
        jd0 = _jd_from_fits_header(fits_path)
        if jd0 is not None:
            out["jd"] = float(jd0)
            out["inspection_jd"] = float(jd0)
            out["_jd_sort"] = float(jd0)
    elif jd_col is None:
        fits_guess = csv_path.with_suffix(".fits")
        if not fits_guess.is_file():
            for sfx in (".fit", ".FIT", ".FITS"):
                alt = csv_path.with_suffix(sfx)
                if alt.is_file():
                    fits_guess = alt
                    break
        jd0 = _jd_from_fits_header(fits_guess)
        if jd0 is not None:
            out["jd"] = float(jd0)
            out["inspection_jd"] = float(jd0)
            out["_jd_sort"] = float(jd0)

    out["catalog_id_key"] = out["catalog_id"].map(_cid_key) if "catalog_id" in out.columns else ""
    return out


def _load_comp_and_targets(
    archive_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp_cols = ["catalog_id", "name", "catalog_mag", "mag", "bp_rp"]
    var_cols = ["catalog_id", "name", "ra_deg", "dec_deg"]
    ps = _platesolve_dir(archive_path)
    comp_df = pd.DataFrame(columns=comp_cols)
    var_df = pd.DataFrame(columns=var_cols)
    try:
        cp = ps / "comparison_stars.csv"
        if cp.is_file():
            comp_df = pd.read_csv(cp)
    except Exception:  # noqa: BLE001
        comp_df = pd.DataFrame(columns=comp_cols)
    try:
        vp = ps / "variable_targets.csv"
        if vp.is_file():
            var_df = pd.read_csv(vp)
    except Exception:  # noqa: BLE001
        var_df = pd.DataFrame(columns=var_cols)
    try:
        if not comp_df.empty:
            comp_df.columns = comp_df.columns.str.lower()
    except Exception:  # noqa: BLE001
        pass
    try:
        if not var_df.empty:
            var_df.columns = var_df.columns.str.lower()
    except Exception:  # noqa: BLE001
        pass
    return comp_df, var_df


def _compute_differential_mag(
    frames_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    target_catalog_id: str,
    mag_col: str,
    err_col: str,
    *,
    time_col: str = "jd_mid",
    weighted: bool = True,
) -> pd.DataFrame:
    result_rows: list[dict[str, Any]] = []
    if frames_df.empty or comp_df.empty or mag_col not in frames_df.columns:
        return pd.DataFrame(result_rows)

    comp_ids = [_cid_key(x) for x in comp_df["catalog_id"].tolist()]
    comp_ids = [c for c in comp_ids if c]
    if not comp_ids:
        return pd.DataFrame(result_rows)

    cdf = comp_df.copy()
    cdf["_mk"] = cdf["catalog_id"].map(_cid_key)
    if "catalog_mag" not in cdf.columns:
        cdf["catalog_mag"] = np.nan
    if "mag" not in cdf.columns:
        cdf["mag"] = np.nan
    cdf["ref_mag"] = pd.to_numeric(cdf["catalog_mag"], errors="coerce")
    cdf["ref_mag"] = cdf["ref_mag"].fillna(pd.to_numeric(cdf["mag"], errors="coerce"))

    gb_col = "source_file" if "source_file" in frames_df.columns else (
        "filename" if "filename" in frames_df.columns else None
    )
    if gb_col is None:
        return pd.DataFrame(result_rows)

    tgt_key = _cid_key(target_catalog_id)
    if not tgt_key:
        return pd.DataFrame(result_rows)

    for frame_id, frame in frames_df.groupby(gb_col, sort=False):
        frame = frame.copy()
        frame["_mk"] = frame["catalog_id"].map(_cid_key) if "catalog_id" in frame.columns else ""
        tgt = frame[frame["_mk"] == tgt_key]
        if tgt.empty:
            continue
        r0 = tgt.iloc[0]
        if pd.isna(r0.get(mag_col)):
            continue
        try:
            tgt_mag = float(r0[mag_col])
        except (TypeError, ValueError):
            continue
        try:
            ev = r0.get(err_col)
            tgt_err = float(ev) if ev is not None and pd.notna(ev) else 0.0
        except (TypeError, ValueError):
            tgt_err = 0.0

        def _time_val(row: pd.Series) -> float:
            v = row.get(time_col)
            if v is not None and pd.notna(v):
                try:
                    fv = float(v)
                    if math.isfinite(fv) and fv != 0.0:
                        return fv
                except (TypeError, ValueError):
                    pass
            for fb in ("jd_mid", "jd", "inspection_jd", "_jd_sort"):
                if fb in row.index:
                    vv = row.get(fb)
                    if vv is not None and pd.notna(vv):
                        try:
                            f2 = float(vv)
                            if math.isfinite(f2):
                                return f2
                        except (TypeError, ValueError):
                            pass
            return 0.0

        t = _time_val(r0)

        comps = frame[frame["_mk"].isin(comp_ids)].copy()
        if mag_col not in comps.columns:
            continue
        comps = comps[pd.notna(comps[mag_col])]
        if comps.empty:
            continue

        comps["_mk2"] = comps["catalog_id"].map(_cid_key)
        cm = cdf[["_mk", "ref_mag"]].copy()
        comps = comps.merge(cm, left_on="_mk2", right_on="_mk", how="left")
        comps = comps.drop(columns=["_mk"], errors="ignore")
        comps_with_ref = comps[pd.notna(comps["ref_mag"]) & pd.notna(comps[mag_col])].copy()
        if comps_with_ref.empty:
            continue

        comp_inst = comps_with_ref[mag_col].astype(float).to_numpy()
        comp_cat = comps_with_ref["ref_mag"].astype(float).to_numpy()
        if err_col in comps_with_ref.columns:
            comp_err = pd.to_numeric(comps_with_ref[err_col], errors="coerce").fillna(0.0).astype(float).to_numpy()
        else:
            comp_err = np.zeros(len(comps_with_ref), dtype=float)

        zp_each = comp_cat - comp_inst

        if weighted and len(zp_each) > 1:
            w = np.where(comp_err > 0, 1.0 / (comp_err**2), 1.0)
            s = float(np.sum(w))
            if s > 0:
                w = w / s
            else:
                w = np.ones(len(zp_each), dtype=float) / len(zp_each)
            zp = float(np.sum(w * zp_each))
            ensemble_err = float(np.sqrt(np.sum((w * comp_err) ** 2)))
        else:
            zp = float(np.mean(zp_each))
            ensemble_err = float(np.std(zp_each)) if len(zp_each) > 1 else 0.0

        cal_mag = tgt_mag + zp
        cal_err = float(np.sqrt(tgt_err**2 + ensemble_err**2))

        def _opt_float(row: pd.Series, k: str) -> float | None:
            v = row.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            try:
                f = float(v)
                if math.isfinite(f) and f != 0.0:
                    return f
            except (TypeError, ValueError):
                pass
            return None

        result_rows.append(
            {
                time_col: t,
                "jd_mid": _opt_float(r0, "jd_mid"),
                "hjd_mid": _opt_float(r0, "hjd_mid"),
                "bjd_tdb_mid": _opt_float(r0, "bjd_tdb_mid"),
                "source_file": str(frame_id),
                "instrumental_mag": tgt_mag,
                "zero_point": zp,
                "cal_mag": cal_mag,
                "cal_mag_err": cal_err,
                "n_comp_used": len(comps_with_ref),
                "ensemble_zp_std": float(np.std(zp_each)) if len(zp_each) > 1 else 0.0,
            }
        )

    out = pd.DataFrame(result_rows)
    if not out.empty and time_col in out.columns:
        out = out.sort_values(time_col, ascending=True, na_position="last")
    return out


def _plot_light_curve(
    lc_df: pd.DataFrame,
    *,
    title: str,
    time_col: str = "jd_mid",
    mag_col: str = "cal_mag",
    err_col: str = "cal_mag_err",
    show_errorbars: bool = True,
) -> Any:
    fig = go.Figure()
    if time_col not in lc_df.columns or mag_col not in lc_df.columns:
        fig.update_layout(title=title)
        return fig

    t_num = pd.to_numeric(lc_df[time_col], errors="coerce")
    t_rel, t_off = jd_series_relative(t_num)
    t_cd = t_num.to_numpy(dtype=float)
    x_short = time_col.upper().replace("_", " ")
    _hover_t = f"{x_short}=%{{customdata:.6f}}<br>Magnitúda=%{{y:.4f}}<extra></extra>"

    if show_errorbars and err_col in lc_df.columns:
        err_vals = pd.to_numeric(lc_df[err_col], errors="coerce").fillna(0.0).tolist()
        fig.add_trace(
            go.Scatter(
                x=t_rel,
                y=lc_df[mag_col],
                error_y=dict(
                    type="data",
                    array=err_vals,
                    visible=True,
                    color="#ff6b6b",
                    thickness=1.5,
                ),
                mode="markers+lines",
                marker=dict(color="#ff6b6b", size=6),
                line=dict(color="#ff6b6b", width=1),
                name="VAR",
                customdata=t_cd,
                hovertemplate=_hover_t,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=t_rel,
                y=lc_df[mag_col],
                mode="markers+lines",
                marker=dict(color="#ff6b6b", size=6),
                line=dict(color="#ff6b6b", width=1),
                name="VAR",
                customdata=t_cd,
                hovertemplate=_hover_t,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=jd_axis_title(x_short, t_off),
        yaxis_title="Magnitúda",
        yaxis=dict(autorange="reversed"),
        hovermode="x unified",
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


def _load_per_frame_sidecars(pipeline: Any, draft_id: int | None) -> pd.DataFrame | None:
    if draft_id is None:
        st.info("Nie je vybraný draft alebo chýba ARCHIVE_PATH v databáze.")
        return None
    try:
        draft_row = pipeline.db.fetch_obs_draft_by_id(int(draft_id))
    except Exception:  # noqa: BLE001
        draft_row = None
    if not draft_row:
        st.info("Nie je vybraný draft alebo chýba ARCHIVE_PATH v databáze.")
        return None
    archive_path = draft_row.get("ARCHIVE_PATH") or draft_row.get("archive_path")
    if not archive_path or not str(archive_path).strip():
        st.info("Nie je vybraný draft alebo chýba ARCHIVE_PATH v databáze.")
        return None
    archive = Path(str(archive_path).strip()).expanduser().resolve()
    if archive.name.casefold() == "non_calibrated":
        archive = archive.parent

    search_roots = [
        archive / "detrended_aligned" / "lights",
        archive / "processed" / "lights",
        archive / "detrended" / "lights",
        archive / "calibrated" / "lights",
    ]

    all_csvs: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            found = [
                p
                for p in root.rglob("*.csv")
                if p.is_file() and p.name not in _LOAD_PER_FRAME_SIDECAR_SKIP
            ]
            if found:
                all_csvs.extend(found)
                log_event(f"_load_per_frame_sidecars: found {len(found)} CSV files under {root}")
                break

    if not all_csvs:
        st.info(
            "Nenašli sa per-frame katalógové CSV. "
            "Spusti **MAKE MASTERSTAR** v záložke VAR-STREM."
        )
        return None

    paths = sorted({p.resolve() for p in all_csvs})
    parts: list[pd.DataFrame] = []
    for p in paths:
        try:
            raw = pd.read_csv(p)
            if raw.empty:
                continue
            parts.append(_normalize_sidecar_dataframe(raw, p))
        except Exception:  # noqa: BLE001
            continue
    if not parts:
        st.info("Najprv spusti export per-frame katalógov.")
        return None
    all_df = pd.concat(parts, ignore_index=True)
    if "_jd_sort" in all_df.columns and all_df["_jd_sort"].notna().any():
        all_df = all_df.sort_values("_jd_sort", ascending=True, na_position="last")
    return all_df


def _platesolve_dir(archive: Path, combined: pd.DataFrame | None = None) -> Path:
    """Return the platesolve subdirectory that contains comparison_stars.csv.

    For multi-group runs the pipeline writes into platesolve/<group>/
    e.g. platesolve/NoFilter_120_2/.
    Search strategy:
      1. Find first subdir of platesolve/ containing comparison_stars.csv
      2. Fall back to platesolve/ root if it has comparison_stars.csv
      3. Otherwise return platesolve/ root
    """
    _ = combined
    ps_root = archive / "platesolve"
    if not ps_root.is_dir():
        return ps_root
    for sub in sorted(ps_root.iterdir()):
        if sub.is_dir() and (sub / "comparison_stars.csv").is_file():
            return sub
    if (ps_root / "comparison_stars.csv").is_file():
        return ps_root
    return ps_root


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def _target_options(archive: Path, combined: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """label -> catalog_id_key"""
    vt = _read_optional_csv(_platesolve_dir(archive) / "variable_targets.csv")
    mapping: dict[str, str] = {}
    labels: list[str] = []
    if vt is not None and not vt.empty and "catalog_id" in vt.columns:
        for _, r in vt.iterrows():
            cid = _cid_key(r.get("catalog_id"))
            if not cid:
                continue
            nm = str(r.get("name") or cid).strip() or cid
            lab = f"{nm} ({cid})"
            if lab not in mapping:
                labels.append(lab)
                mapping[lab] = cid
    if not labels and not combined.empty and "vsx_known_variable" in combined.columns:
        m = combined[combined["vsx_known_variable"].map(_to_bool)]["catalog_id_key"].dropna().unique()
        for cid in m:
            if not cid:
                continue
            sub = combined[combined["catalog_id_key"] == cid]
            nm = str(sub["name"].dropna().iloc[0]) if "name" in sub.columns and sub["name"].notna().any() else cid
            lab = f"{nm} ({cid})"
            if lab not in mapping:
                labels.append(lab)
                mapping[lab] = str(cid)
    return labels, mapping


def _comparison_options(archive: Path, combined: pd.DataFrame, *, max_opts: int = 10) -> tuple[list[str], dict[str, str]]:
    cs = _read_optional_csv(_platesolve_dir(archive) / "comparison_stars.csv")
    mapping: dict[str, str] = {}
    labels: list[str] = []
    if cs is not None and not cs.empty and "catalog_id" in cs.columns:
        for _, r in cs.iterrows():
            cid = _cid_key(r.get("catalog_id"))
            if not cid:
                continue
            nm = str(r.get("name") or cid).strip() or cid
            lab = f"{nm} ({cid})"
            if lab not in mapping:
                labels.append(lab)
                mapping[lab] = cid
            if len(labels) >= max_opts:
                break
    if not labels and not combined.empty and "catalog_known_variable" in combined.columns:
        cand = combined[~combined["catalog_known_variable"].map(_to_bool)]
        u = cand["catalog_id_key"].drop_duplicates().dropna().unique()[:max_opts]
        for cid in u:
            if not str(cid).strip():
                continue
            sub = combined[combined["catalog_id_key"] == cid]
            nm = str(sub["name"].dropna().iloc[0]) if "name" in sub.columns and sub["name"].notna().any() else cid
            lab = f"{nm} ({cid})"
            labels.append(lab)
            mapping[lab] = str(cid)
    return labels, mapping


def _series_for_star(combined: pd.DataFrame, cid: str, *, mag_col: str, flux_col: str) -> pd.DataFrame:
    sub = combined[combined["catalog_id_key"] == _cid_key(cid)].copy()
    if sub.empty:
        return sub
    xcol = next(
        (
            c
            for c in ("jd_mid", "jd", "inspection_jd", "_jd_sort")
            if c in sub.columns and sub[c].notna().any()
        ),
        "_jd_sort",
    )
    sub = sub.sort_values(xcol, ascending=True)
    return sub


def _jd_plot_x_and_raw(s: pd.DataFrame, xcol: str, jd_off: int | None) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.to_numeric(s[xcol], errors="coerce").to_numpy(dtype=float)
    rel = raw - float(jd_off) if jd_off is not None else raw
    return rel, raw


def render_aperture_results(pipeline: Any, draft_id: int | None) -> None:
    st.subheader("Aperture Photometry — Results")
    combined = _load_per_frame_sidecars(pipeline, draft_id)
    if combined is None:
        return
    ap = _draft_archive_path(pipeline, draft_id)
    if ap is None:
        return

    tgt_labels, tgt_map = _target_options(ap, combined)
    if not tgt_labels:
        st.warning("Žiadne cieľové hviezdy (variable_targets.csv alebo VSX v sidecaroch).")
        return
    tlab = st.selectbox("Target star", options=tgt_labels, key="vyvar_ap_targ")
    tgt_cid = tgt_map[tlab]

    clabels, cmap = _comparison_options(ap, combined)
    if not clabels:
        st.warning("Žiadne porovnávacie hviezdy.")
        return
    sel = st.multiselect(
        "Comparison stars",
        options=clabels,
        default=clabels[: min(5, len(clabels))],
        key="vyvar_ap_comp",
    )
    ymode = st.radio("Y-axis", ("Magnitude", "Flux"), horizontal=True, key="vyvar_ap_y")
    ycol = "aperture_mag" if ymode == "Magnitude" else "aperture_flux"

    fig = go.Figure()
    xcol = next(
        (
            c
            for c in ("jd_mid", "jd", "inspection_jd", "_jd_sort")
            if c in combined.columns and combined[c].notna().any()
        ),
        "_jd_sort",
    )
    _, ap_jd_off = jd_series_relative(pd.to_numeric(combined[xcol], errors="coerce"))
    ap_x_title = jd_axis_title("JD", ap_jd_off)
    ap_hover = "JD=%{customdata:.6f}<extra></extra>"
    for lab in sel:
        cc = cmap.get(lab)
        if not cc:
            continue
        s = _series_for_star(combined, cc, mag_col="aperture_mag", flux_col="aperture_flux")
        if s.empty:
            continue
        x_rel, x_raw = _jd_plot_x_and_raw(s, xcol, ap_jd_off)
        fig.add_trace(
            go.Scatter(
                x=x_rel,
                y=s[ycol],
                mode="lines+markers",
                name=lab,
                line=dict(width=1, color="rgba(128,128,128,0.8)"),
                marker=dict(size=4),
                customdata=x_raw,
                hovertemplate=ap_hover,
            )
        )
    ts = _series_for_star(combined, tgt_cid, mag_col="aperture_mag", flux_col="aperture_flux")
    if not ts.empty:
        x_rel, x_raw = _jd_plot_x_and_raw(ts, xcol, ap_jd_off)
        fig.add_trace(
            go.Scatter(
                x=x_rel,
                y=ts[ycol],
                mode="lines+markers",
                name=tlab,
                line=dict(width=3, color="red"),
                marker=dict(size=7),
                customdata=x_raw,
                hovertemplate=ap_hover,
            )
        )
    fig.update_layout(
        xaxis_title=ap_x_title,
        yaxis_title=ycol,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    if ymode == "Magnitude":
        fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Diferenciálna magnitúda — svetelná krivka")

    comp_df, var_df = _load_comp_and_targets(ap)

    if comp_df.empty:
        st.warning(
            "comparison_stars.csv nie je k dispozícii — "
            "spusti **MAKE MASTERSTAR** (plate solve + katalóg)."
        )
    elif var_df.empty:
        st.info(
            "variable_targets.csv je prázdny — "
            "doplň cieľové premenné hviezdy do súboru."
        )
    else:
        var_work = var_df[var_df["catalog_id"].map(lambda x: bool(_cid_key(x)))].reset_index(drop=True)
        if var_work.empty:
            st.info("variable_targets.csv neobsahuje platné catalog_id pre výber cieľa.")
        else:
            _time_options = {
                "JD (mid exposure)": "jd_mid",
                "HJD": "hjd_mid",
                "BJD(TDB)": "bjd_tdb_mid",
            }
            _time_label = st.radio(
                "Časová os",
                options=list(_time_options.keys()),
                horizontal=True,
                key="vyvar_ap_lc_time_axis",
            )
            _time_col = _time_options[_time_label]

            _weighted = st.checkbox(
                "Vážený priemer comp ensemblu (1/σ²)",
                value=True,
                key="vyvar_ap_lc_weighted",
            )

            _var_names = var_work["name"].fillna(var_work["catalog_id"]).astype(str).tolist()
            _var_ids = var_work["catalog_id"].astype(str).tolist()

            _sel_idx = st.selectbox(
                "Cieľová hviezda (VAR)",
                options=range(len(_var_names)),
                format_func=lambda i: _var_names[i],
                key="vyvar_ap_lc_target",
            )
            _target_id = _var_ids[_sel_idx]

            lc_df = _compute_differential_mag(
                combined,
                comp_df,
                _target_id,
                mag_col="aperture_mag",
                err_col="aperture_mag_err",
                time_col=_time_col,
                weighted=_weighted,
            )

            if lc_df.empty:
                st.warning(
                    f"Hviezda {_var_names[_sel_idx]} nebola nájdená "
                    "v per-frame CSV alebo chýbajú comp hviezdy."
                )
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Počet bodov", len(lc_df))
                col2.metric("Priemer mag", f"{lc_df['cal_mag'].mean():.4f}")
                col3.metric("RMS", f"{lc_df['cal_mag'].std():.4f}")
                col4.metric("Comp hviezd", int(lc_df["n_comp_used"].median()))

                fig_lc = _plot_light_curve(
                    lc_df,
                    title=f"Svetelná krivka — {_var_names[_sel_idx]} (Aperture)",
                    time_col=_time_col,
                )
                st.plotly_chart(fig_lc, use_container_width=True)

                with st.expander("Zero point stabilita comp ensemblu", expanded=False):
                    _, zp_off = jd_series_relative(pd.to_numeric(lc_df[_time_col], errors="coerce"))
                    zp_x, zp_raw = _jd_plot_x_and_raw(lc_df, _time_col, zp_off)
                    fig_zp = go.Figure()
                    fig_zp.add_trace(
                        go.Scatter(
                            x=zp_x,
                            y=lc_df["zero_point"],
                            mode="markers+lines",
                            marker=dict(color="#4ecdc4", size=5),
                            name="ZP",
                            customdata=zp_raw,
                            hovertemplate="Čas=%{customdata:.6f}<extra></extra>",
                        )
                    )
                    fig_zp.add_trace(
                        go.Scatter(
                            x=zp_x,
                            y=lc_df["ensemble_zp_std"],
                            mode="markers",
                            marker=dict(color="#ffe66d", size=4),
                            name="ZP scatter",
                            customdata=zp_raw,
                            hovertemplate="Čas=%{customdata:.6f}<extra></extra>",
                        )
                    )
                    fig_zp.update_layout(
                        title="Zero point (ZP = catalog_mag - instrumental_mag ensemblu)",
                        yaxis_title="ZP [mag]",
                        xaxis_title=jd_axis_title(_time_col.upper().replace("_", " "), zp_off),
                        margin=dict(l=50, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_zp, use_container_width=True)

                _fn_ap = "".join(c if c.isalnum() or c in "-._" else "_" for c in str(_var_names[_sel_idx]))[:120]
                st.download_button(
                    "⬇️ Stiahnuť svetelnú krivku (CSV)",
                    data=lc_df.to_csv(index=False),
                    file_name=f"light_curve_{_fn_ap}_aperture.csv",
                    mime="text/csv",
                    key="vyvar_ap_lc_download",
                )

    with st.expander("Raw data (filterable)"):
        show = combined.copy()
        if "name" in show.columns:
            q = st.text_input("Filter by name contains", key="vyvar_ap_raw_q")
            if q.strip():
                show = show[show["name"].astype(str).str.contains(q.strip(), case=False, na=False)]
        st.dataframe(show, use_container_width=True, height=400)


def render_psf_results(pipeline: Any, draft_id: int | None) -> None:
    st.subheader("PSF Photometry — Results")
    cfg = pipeline.config
    if not bool(getattr(cfg, "psf_photometry_enabled", False)):
        st.warning(
            "PSF fotometria je vypnutá. Zapni ju v nastaveniach (`config.json` → `psf_photometry_enabled`)."
        )
        return

    combined = _load_per_frame_sidecars(pipeline, draft_id)
    if combined is None:
        return
    ap = _draft_archive_path(pipeline, draft_id)
    if ap is None:
        return

    tgt_labels, tgt_map = _target_options(ap, combined)
    if not tgt_labels:
        st.warning("Žiadne cieľové hviezdy.")
        return
    tlab = st.selectbox("Target star", options=tgt_labels, key="vyvar_psf_targ")
    tgt_cid = tgt_map[tlab]

    clabels, cmap = _comparison_options(ap, combined)
    if not clabels:
        st.warning("Žiadne porovnávacie hviezdy.")
        return
    sel = st.multiselect(
        "Comparison stars",
        options=clabels,
        default=clabels[: min(5, len(clabels))],
        key="vyvar_psf_comp",
    )

    xcol = next(
        (
            c
            for c in ("jd_mid", "jd", "inspection_jd", "_jd_sort")
            if c in combined.columns and combined[c].notna().any()
        ),
        "_jd_sort",
    )
    _, psf_jd_off = jd_series_relative(pd.to_numeric(combined[xcol], errors="coerce"))
    psf_x_title = jd_axis_title("JD", psf_jd_off)
    psf_hover = "JD=%{customdata:.6f}<extra></extra>"

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.62, 0.38])
    for lab in sel:
        cc = cmap.get(lab)
        if not cc:
            continue
        s = _series_for_star(combined, cc, mag_col="psf_mag", flux_col="psf_flux")
        if s.empty:
            continue
        x_rel, x_raw = _jd_plot_x_and_raw(s, xcol, psf_jd_off)
        fig.add_trace(
            go.Scatter(
                x=x_rel,
                y=s["psf_mag"],
                mode="lines+markers",
                name=lab,
                line=dict(width=1, color="rgba(128,128,128,0.8)"),
                marker=dict(size=4),
                customdata=x_raw,
                hovertemplate=psf_hover,
            ),
            row=1,
            col=1,
        )
    ts = _series_for_star(combined, tgt_cid, mag_col="psf_mag", flux_col="psf_flux")
    if not ts.empty:
        x_rel, x_raw = _jd_plot_x_and_raw(ts, xcol, psf_jd_off)
        fig.add_trace(
            go.Scatter(
                x=x_rel,
                y=ts["psf_mag"],
                mode="lines+markers",
                name=tlab,
                line=dict(width=3, color="red"),
                marker=dict(size=7),
                customdata=x_raw,
                hovertemplate=psf_hover,
            ),
            row=1,
            col=1,
        )
    fig.update_yaxes(autorange="reversed", row=1, col=1)

    if not ts.empty:
        chi = pd.to_numeric(ts["psf_chi2"], errors="coerce")
        bad = chi > 5.0
        colors = ["red" if bool(b) else "royalblue" for b in bad.fillna(False).tolist()]
        x_rel_chi, x_raw_chi = _jd_plot_x_and_raw(ts, xcol, psf_jd_off)
        fig.add_trace(
            go.Scatter(
                x=x_rel_chi,
                y=chi,
                mode="markers",
                name="PSF χ² (target)",
                marker=dict(size=8, color=colors),
                customdata=x_raw_chi,
                hovertemplate=psf_hover,
            ),
            row=2,
            col=1,
        )
    fig.add_hline(y=5.0, line_dash="dash", line_color="orange", row=2, col=1)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0))
    fig.update_yaxes(title_text="psf_mag", row=1, col=1)
    fig.update_yaxes(title_text="psf_chi2", row=2, col=1)
    fig.update_xaxes(title_text=psf_x_title, row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Diferenciálna magnitúda — svetelná krivka")

    comp_df, var_df = _load_comp_and_targets(ap)

    if comp_df.empty:
        st.warning(
            "comparison_stars.csv nie je k dispozícii — "
            "spusti **MAKE MASTERSTAR** (plate solve + katalóg)."
        )
    elif var_df.empty:
        st.info(
            "variable_targets.csv je prázdny — "
            "doplň cieľové premenné hviezdy do súboru."
        )
    else:
        var_work = var_df[var_df["catalog_id"].map(lambda x: bool(_cid_key(x)))].reset_index(drop=True)
        if var_work.empty:
            st.info("variable_targets.csv neobsahuje platné catalog_id pre výber cieľa.")
        else:
            _time_options = {
                "JD (mid exposure)": "jd_mid",
                "HJD": "hjd_mid",
                "BJD(TDB)": "bjd_tdb_mid",
            }
            _time_label = st.radio(
                "Časová os",
                options=list(_time_options.keys()),
                horizontal=True,
                key="vyvar_psf_lc_time_axis",
            )
            _time_col = _time_options[_time_label]

            _weighted = st.checkbox(
                "Vážený priemer comp ensemblu (1/σ²)",
                value=True,
                key="vyvar_psf_lc_weighted",
            )

            _var_names = var_work["name"].fillna(var_work["catalog_id"]).astype(str).tolist()
            _var_ids = var_work["catalog_id"].astype(str).tolist()

            _sel_idx = st.selectbox(
                "Cieľová hviezda (VAR)",
                options=range(len(_var_names)),
                format_func=lambda i: _var_names[i],
                key="vyvar_psf_lc_target",
            )
            _target_id = _var_ids[_sel_idx]

            lc_df = _compute_differential_mag(
                combined,
                comp_df,
                _target_id,
                mag_col="psf_mag",
                err_col="psf_flux_err",
                time_col=_time_col,
                weighted=_weighted,
            )

            if lc_df.empty:
                st.warning(
                    f"Hviezda {_var_names[_sel_idx]} nebola nájdená "
                    "v per-frame CSV alebo chýbajú comp hviezdy."
                )
            else:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Počet bodov", len(lc_df))
                col2.metric("Priemer mag", f"{lc_df['cal_mag'].mean():.4f}")
                col3.metric("RMS", f"{lc_df['cal_mag'].std():.4f}")
                col4.metric("Comp hviezd", int(lc_df["n_comp_used"].median()))

                fig_lc = _plot_light_curve(
                    lc_df,
                    title=f"Svetelná krivka — {_var_names[_sel_idx]} (PSF)",
                    time_col=_time_col,
                )
                st.plotly_chart(fig_lc, use_container_width=True)

                if st.checkbox(
                    "Porovnať Aperture vs PSF",
                    key="vyvar_psf_lc_compare",
                ):
                    lc_ap = _compute_differential_mag(
                        combined,
                        comp_df,
                        _target_id,
                        mag_col="aperture_mag",
                        err_col="aperture_mag_err",
                        time_col=_time_col,
                        weighted=_weighted,
                    )
                    _cmp_t_parts = []
                    if not lc_ap.empty:
                        _cmp_t_parts.append(pd.to_numeric(lc_ap[_time_col], errors="coerce"))
                    if not lc_df.empty:
                        _cmp_t_parts.append(pd.to_numeric(lc_df[_time_col], errors="coerce"))
                    _cmp_t = pd.concat(_cmp_t_parts, ignore_index=True) if _cmp_t_parts else pd.Series(dtype=float)
                    _, cmp_off = jd_series_relative(_cmp_t)
                    cmp_hover = "Čas=%{customdata:.6f}<extra></extra>"
                    fig_cmp = go.Figure()
                    if not lc_ap.empty:
                        xa, xar = _jd_plot_x_and_raw(lc_ap, _time_col, cmp_off)
                        fig_cmp.add_trace(
                            go.Scatter(
                                x=xa,
                                y=lc_ap["cal_mag"],
                                mode="markers",
                                name="Aperture",
                                marker=dict(color="#4ecdc4", size=5, symbol="circle"),
                                customdata=xar,
                                hovertemplate=cmp_hover,
                            )
                        )
                    if not lc_df.empty:
                        xp, xpr = _jd_plot_x_and_raw(lc_df, _time_col, cmp_off)
                        fig_cmp.add_trace(
                            go.Scatter(
                                x=xp,
                                y=lc_df["cal_mag"],
                                mode="markers",
                                name="PSF",
                                marker=dict(color="#ff6b6b", size=5, symbol="diamond"),
                                customdata=xpr,
                                hovertemplate=cmp_hover,
                            )
                        )
                    fig_cmp.update_layout(
                        title="Aperture vs PSF — diferenciálna magnitúda",
                        yaxis=dict(autorange="reversed"),
                        yaxis_title="Magnitúda",
                        xaxis_title=jd_axis_title(_time_col.upper().replace("_", " "), cmp_off),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

                with st.expander("Zero point stabilita comp ensemblu", expanded=False):
                    _, zp_off_psf = jd_series_relative(pd.to_numeric(lc_df[_time_col], errors="coerce"))
                    zp_xp, zp_rawp = _jd_plot_x_and_raw(lc_df, _time_col, zp_off_psf)
                    fig_zp = go.Figure()
                    fig_zp.add_trace(
                        go.Scatter(
                            x=zp_xp,
                            y=lc_df["zero_point"],
                            mode="markers+lines",
                            marker=dict(color="#4ecdc4", size=5),
                            name="ZP",
                            customdata=zp_rawp,
                            hovertemplate="Čas=%{customdata:.6f}<extra></extra>",
                        )
                    )
                    fig_zp.add_trace(
                        go.Scatter(
                            x=zp_xp,
                            y=lc_df["ensemble_zp_std"],
                            mode="markers",
                            marker=dict(color="#ffe66d", size=4),
                            name="ZP scatter",
                            customdata=zp_rawp,
                            hovertemplate="Čas=%{customdata:.6f}<extra></extra>",
                        )
                    )
                    fig_zp.update_layout(
                        title="Zero point (ZP = catalog_mag - instrumental_mag ensemblu)",
                        yaxis_title="ZP [mag]",
                        xaxis_title=jd_axis_title(_time_col.upper().replace("_", " "), zp_off_psf),
                        margin=dict(l=50, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_zp, use_container_width=True)

                _fn_psf = "".join(c if c.isalnum() or c in "-._" else "_" for c in str(_var_names[_sel_idx]))[:120]
                st.download_button(
                    "⬇️ Stiahnuť svetelnú krivku (CSV)",
                    data=lc_df.to_csv(index=False),
                    file_name=f"light_curve_{_fn_psf}_psf.csv",
                    mime="text/csv",
                    key="vyvar_psf_lc_download",
                )

    with st.expander("Raw data (PSF)"):
        show = combined.copy()

        def _hl(row: pd.Series) -> list[str]:
            if _to_bool(row.get("psf_fit_ok")):
                return [""] * len(row)
            return ["background-color: #ffcccc"] * len(row)

        try:
            styler = show.style.apply(_hl, axis=1)
            st.dataframe(styler, use_container_width=True, height=400)
        except Exception:  # noqa: BLE001
            st.dataframe(show, use_container_width=True, height=400)


def render_comp_validation(pipeline: Any, draft_id: int | None) -> None:
    st.subheader("Comp Star Validation")
    st.caption("Overenie stability porovnávacích hviezd. Červená = potenciálne premenná.")

    combined = _load_per_frame_sidecars(pipeline, draft_id)
    if combined is None:
        return
    ap = _draft_archive_path(pipeline, draft_id)
    if ap is None:
        return

    cs = _read_optional_csv(_platesolve_dir(ap) / "comparison_stars.csv")
    if cs is None or cs.empty or "catalog_id" not in cs.columns:
        st.warning("Chýba `platesolve/comparison_stars.csv`.")
        return
    comp_ids = [_cid_key(x) for x in cs["catalog_id"].tolist() if _cid_key(x)]
    comp_ids = list(dict.fromkeys(comp_ids))

    thr_mult = st.slider(
        "RMS outlier threshold (× median RMS)",
        min_value=1.2,
        max_value=5.0,
        value=2.0,
        step=0.1,
        key="vyvar_comp_rms_mult",
    )

    rows: list[dict[str, Any]] = []
    ap_rms_list: list[float] = []
    for cid in comp_ids:
        sub = combined[combined["catalog_id_key"] == cid]
        if sub.empty:
            continue
        mags = pd.to_numeric(sub["aperture_mag"], errors="coerce")
        pmags = pd.to_numeric(sub["psf_mag"], errors="coerce")
        r_ap = float(mags.std(ddof=0)) if mags.notna().sum() > 1 else float("nan")
        r_pf = float(pmags.std(ddof=0)) if pmags.notna().sum() > 1 else float("nan")
        if math.isfinite(r_ap):
            ap_rms_list.append(r_ap)
        vsx = bool(sub["vsx_known_variable"].iloc[0]) if "vsx_known_variable" in sub.columns else False
        ckv = bool(sub["catalog_known_variable"].iloc[0]) if "catalog_known_variable" in sub.columns else False
        name = str(sub["name"].iloc[0]) if "name" in sub.columns else cid
        rows.append(
            {
                "name": name,
                "catalog_id": cid,
                "aperture_mag_rms": r_ap,
                "psf_mag_rms": r_pf,
                "vsx_known_variable": vsx,
                "catalog_known_variable": ckv,
            }
        )

    if not rows:
        st.warning("Žiadne dáta pre comp hviezdy v sidecaroch.")
        return

    tab_df = pd.DataFrame(rows)
    med_rms = float(np.nanmedian(tab_df["aperture_mag_rms"])) if tab_df["aperture_mag_rms"].notna().any() else float("nan")
    if not math.isfinite(med_rms) or med_rms <= 0:
        med_rms = 1e-6

    def _verdict(r: pd.Series) -> str:
        if _to_bool(r.get("vsx_known_variable")) or _to_bool(r.get("catalog_known_variable")):
            return "Known Variable"
        rms = float(r.get("aperture_mag_rms") or float("nan"))
        if math.isfinite(rms) and rms > thr_mult * med_rms:
            return "Suspicious"
        return "Stable"

    tab_df["verdict"] = tab_df.apply(_verdict, axis=1)

    suspicious_ids = tab_df.loc[tab_df["verdict"] == "Suspicious", "catalog_id"].astype(str).tolist()
    st.session_state["vyvar_suspicious_comp_stars"] = suspicious_ids

    n_known = int((tab_df["verdict"] == "Known Variable").sum())
    n_susp = int((tab_df["verdict"] == "Suspicious").sum())
    n_stable = int((tab_df["verdict"] == "Stable").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total comp stars", len(tab_df))
    c2.metric("Stable", n_stable)
    c3.metric("Known variable (VSX/cat)", n_known)
    c4.metric("Suspicious (RMS outlier)", n_susp)

    colors = []
    labels = []
    for _, r in tab_df.iterrows():
        lab = f"{r['name']}"
        labels.append(lab)
        if _verdict(r) == "Known Variable":
            colors.append("red")
        elif _verdict(r) == "Suspicious":
            colors.append("orange")
        elif _to_bool(r.get("catalog_known_variable")) is False and float(r.get("aperture_mag_rms") or 1) < thr_mult * med_rms:
            colors.append("green")
        else:
            colors.append("gray")

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=tab_df["aperture_mag_rms"].tolist(),
                marker_color=colors,
                text=tab_df["catalog_id"].astype(str),
                hovertemplate="%{text}<br>RMS=%{y:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(xaxis_title="Star", yaxis_title="RMS (aperture_mag)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(tab_df, use_container_width=True)

    csv_bytes = tab_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export validation table (CSV)",
        data=csv_bytes,
        file_name="comp_validation_export.csv",
        mime="text/csv",
        key="vyvar_comp_dl",
    )


def render_variable_detection(pipeline: Any, draft_id: int | None) -> None:
    st.subheader("Variable Detection — Neznáme premenné hviezdy")
    st.caption("Hviezdy ktoré nie sú v katalógoch ale vykazujú variabilitu v časovej rade.")

    if "vyvar_var_detection_exclusions" not in st.session_state:
        st.session_state["vyvar_var_detection_exclusions"] = []
    if "vyvar_newly_added_var_targets" not in st.session_state:
        st.session_state["vyvar_newly_added_var_targets"] = []

    combined = _load_per_frame_sidecars(pipeline, draft_id)
    if combined is None:
        return
    ap = _draft_archive_path(pipeline, draft_id)
    if ap is None:
        return

    ms_path = _platesolve_dir(ap) / "masterstars_full_match.csv"
    ms = _read_optional_csv(ms_path)
    if ms is None or ms.empty:
        st.warning("Chýba `platesolve/masterstars_full_match.csv`.")
        return
    if "catalog_id" not in ms.columns or "x" not in ms.columns or "y" not in ms.columns:
        st.warning("Masterstars CSV nemá očakávané stĺpce.")
        return

    ms = ms.copy()
    ms["catalog_id_key"] = ms["catalog_id"].map(_cid_key)
    for col in ("vsx_known_variable", "catalog_known_variable", "gaia_dr3_variable_catalog"):
        if col in ms.columns:
            ms[col] = ms[col].map(_to_bool)
        else:
            ms[col] = False

    excl = {str(x) for x in (st.session_state.get("vyvar_var_detection_exclusions") or [])}

    per_star = combined.groupby("catalog_id_key", dropna=False)
    rms_rows: list[dict[str, Any]] = []
    for cid, grp in per_star:
        cks = str(cid).strip()
        if not cks:
            continue
        mags = pd.to_numeric(grp["aperture_mag"], errors="coerce")
        nfr = int(mags.notna().sum())
        if nfr < 2:
            continue
        rms = float(mags.std(ddof=0))
        rms_rows.append({"catalog_id_key": cks, "n_frames": nfr, "aperture_mag_rms": rms})
    rms_df = pd.DataFrame(rms_rows)
    if rms_df.empty:
        st.warning("Nedostatok snímok na výpočet RMS.")
        return

    med_all = float(np.nanmedian(rms_df["aperture_mag_rms"]))
    if not math.isfinite(med_all) or med_all <= 0:
        med_all = 1e-6

    rms_hi = float(max(med_all * 8.0, med_all * 1.06, med_all * 2.0 + 1e-9))
    rms_lo = float(med_all * 1.05)
    min_rms = st.slider(
        "Min RMS for suspicion",
        min_value=rms_lo,
        max_value=rms_hi,
        value=float(min(max(med_all * 2.0, rms_lo), rms_hi)),
        key="vyvar_vdet_min_rms",
    )
    min_frames = st.slider("Min frames with data", 2, 50, 5, key="vyvar_vdet_min_fr")
    only_unknown = st.checkbox(
        "Show only stars NOT flagged in variable catalogs",
        value=True,
        key="vyvar_vdet_only_unk",
    )

    suspicious_detail: list[dict[str, Any]] = []
    for _, rr in rms_df.iterrows():
        cid = str(rr["catalog_id_key"])
        if cid in excl:
            continue
        if int(rr["n_frames"]) < min_frames:
            continue
        if float(rr["aperture_mag_rms"]) < min_rms:
            continue
        mrow = ms[ms["catalog_id_key"] == cid]
        if mrow.empty:
            continue
        m0 = mrow.iloc[0]
        known = (
            _to_bool(m0.get("vsx_known_variable"))
            or _to_bool(m0.get("catalog_known_variable"))
            or _to_bool(m0.get("gaia_dr3_variable_catalog"))
        )
        if only_unknown and known:
            continue
        suspicious_detail.append(
            {
                "catalog_id_key": cid,
                "name": str(m0.get("name") or cid),
                "x": float(m0.get("x", float("nan"))),
                "y": float(m0.get("y", float("nan"))),
                "rms": float(rr["aperture_mag_rms"]),
                "known_catalog_variable": known,
            }
        )

    from_comp = set(str(x) for x in (st.session_state.get("vyvar_suspicious_comp_stars") or []))
    for cid in from_comp:
        if cid in excl:
            continue
        if any(d["catalog_id_key"] == cid for d in suspicious_detail):
            continue
        mrow = ms[ms["catalog_id_key"] == cid]
        if mrow.empty:
            continue
        m0 = mrow.iloc[0]
        known = (
            _to_bool(m0.get("vsx_known_variable"))
            or _to_bool(m0.get("catalog_known_variable"))
            or _to_bool(m0.get("gaia_dr3_variable_catalog"))
        )
        if only_unknown and known:
            continue
        rms_g = rms_df[rms_df["catalog_id_key"] == cid]
        rms_v = float(rms_g["aperture_mag_rms"].iloc[0]) if not rms_g.empty else float("nan")
        suspicious_detail.append(
            {
                "catalog_id_key": cid,
                "name": str(m0.get("name") or cid),
                "x": float(m0.get("x", float("nan"))),
                "y": float(m0.get("y", float("nan"))),
                "rms": rms_v,
                "known_catalog_variable": known,
                "from_comp_validation": True,
            }
        )

    n_field = len(ms["catalog_id_key"].unique())
    kv_mask = ms["vsx_known_variable"] | ms["catalog_known_variable"] | ms["gaia_dr3_variable_catalog"]
    n_known = int(kv_mask.sum())
    n_susp = len(suspicious_detail)
    n_stable = max(0, n_field - n_known - n_susp)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total stars in field", n_field)
    m2.metric("Known variables (VSX+GAIA flags)", n_known)
    m3.metric("Unknown suspicious", n_susp)
    m4.metric("Confirmed stable (approx.)", n_stable)

    gx = ms["x"].astype(float)
    gy = ms["y"].astype(float)
    figm = go.Figure()
    figm.add_trace(
        go.Scatter(
            x=gx,
            y=gy,
            mode="markers",
            name="Field",
            marker=dict(size=4, color="lightgray"),
        )
    )
    kv = ms[ms["vsx_known_variable"] | ms["catalog_known_variable"] | ms["gaia_dr3_variable_catalog"]]
    if not kv.empty:
        figm.add_trace(
            go.Scatter(
                x=kv["x"].astype(float),
                y=kv["y"].astype(float),
                mode="markers",
                name="Known variable",
                marker=dict(size=10, symbol="square", color="gold", line=dict(width=1, color="black")),
            )
        )
    if suspicious_detail:
        sx = [d["x"] for d in suspicious_detail]
        sy = [d["y"] for d in suspicious_detail]
        figm.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="markers",
                name="Suspicious",
                marker=dict(size=12, symbol="circle", color="red", line=dict(width=1, color="darkred")),
                text=[d["catalog_id_key"] for d in suspicious_detail],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    figm.update_layout(
        yaxis_autorange="reversed",
        xaxis_title="x [px]",
        yaxis_title="y [px]",
        height=520,
    )
    st.plotly_chart(figm, use_container_width=True)

    if not suspicious_detail:
        st.info("Žiadne podozrivé hviezdy pri aktuálnych prahoch.")
        st.markdown(
            "#### Info\nHviezdy pridané do sledovania VAR budú zahrnuté do PSF modelu "
            "pri ďalšom spustení MASTERSTAR generovania (po úprave výberu v pipeline)."
        )
        return

    opts = [f"{d['name']} ({d['catalog_id_key']})" for d in suspicious_detail]
    pick = st.selectbox("Vybrať hviezdu pre detail", options=opts, key="vyvar_vdet_pick")
    sel_cid = suspicious_detail[opts.index(pick)]["catalog_id_key"]

    ts = _series_for_star(combined, sel_cid, mag_col="aperture_mag", flux_col="aperture_flux")
    xcol = next(
        (
            c
            for c in ("jd_mid", "jd", "inspection_jd", "_jd_sort")
            if c in ts.columns and ts[c].notna().any()
        ),
        "_jd_sort",
    )
    if not ts.empty:
        _, vdet_off = jd_series_relative(pd.to_numeric(ts[xcol], errors="coerce"))
        xv, xr = _jd_plot_x_and_raw(ts, xcol, vdet_off)
        vdet_xtitle = jd_axis_title("JD", vdet_off)
        fa = go.Figure(
            go.Scatter(
                x=xv,
                y=ts["aperture_mag"],
                mode="lines+markers",
                name="aperture_mag",
                customdata=xr,
                hovertemplate="JD=%{customdata:.6f}<extra></extra>",
            )
        )
        fa.update_layout(yaxis_autorange="reversed", title="Aperture magnitude", xaxis_title=vdet_xtitle)
        st.plotly_chart(fa, use_container_width=True)
        if ts["psf_mag"].notna().any():
            fp = go.Figure(
                go.Scatter(
                    x=xv,
                    y=ts["psf_mag"],
                    mode="lines+markers",
                    name="psf_mag",
                    customdata=xr,
                    hovertemplate="JD=%{customdata:.6f}<extra></extra>",
                )
            )
            fp.update_layout(yaxis_autorange="reversed", title="PSF magnitude", xaxis_title=vdet_xtitle)
            st.plotly_chart(fp, use_container_width=True)

    msel = ms[ms["catalog_id_key"] == sel_cid]
    if msel.empty:
        st.caption("Masterstars nemajú tento `catalog_id` — metadáta Gaia nie sú dostupné.")
        mrow = None
    else:
        mrow = msel.iloc[0]
        gmag = mrow.get("phot_g_mean_mag", mrow.get("mag"))
        bprp = mrow.get("bp_rp", mrow.get("b_v"))
        st.info(
            f"Gaia G≈{gmag!s}, BP-RP≈{bprp!s}, RA={mrow.get('ra_deg', '—')}, Dec={mrow.get('dec_deg', '—')}"
        )

    st.markdown("#### Akcia")
    for d in suspicious_detail:
        cid = d["catalog_id_key"]
        mloc = ms[ms["catalog_id_key"] == cid]
        m0 = mloc.iloc[0] if not mloc.empty else None
        cols = st.columns([3, 3, 2])
        with cols[0]:
            st.write(f"**{d['name']}** `{cid}`")
        with cols[1]:
            if st.button("Pridať do sledovania VAR", key=f"vyvar_add_var_{cid}"):
                vpath = _platesolve_dir(ap) / "variable_targets.csv"
                vpath.parent.mkdir(parents=True, exist_ok=True)
                new_row = {
                    "name": str(d["name"]),
                    "catalog_id": cid,
                    "catalog": "GAIA_DR3",
                    "ra_deg": (m0.get("ra_deg", "") if m0 is not None else ""),
                    "dec_deg": (m0.get("dec_deg", "") if m0 is not None else ""),
                    "priority": 5,
                    "notes": "VYVAR Variable Detection tab",
                }
                if vpath.is_file():
                    old = pd.read_csv(vpath)
                    existing = set(old.get("catalog_id", pd.Series(dtype=str)).map(_cid_key))
                    if cid not in existing:
                        old = pd.concat([old, pd.DataFrame([new_row])], ignore_index=True)
                        old.to_csv(vpath, index=False)
                else:
                    pd.DataFrame([new_row]).to_csv(vpath, index=False)
                added = list(st.session_state.get("vyvar_newly_added_var_targets") or [])
                added.append(cid)
                st.session_state["vyvar_newly_added_var_targets"] = added
                log_event(f"Variable Detection: pridané do variable_targets.csv — {cid} ({d['name']})")
                st.success(f"Pridané: {d['name']}")
        with cols[2]:
            if st.button("Označiť ako stabilnú", key=f"vyvar_excl_{cid}"):
                ex = list(st.session_state.get("vyvar_var_detection_exclusions") or [])
                if cid not in ex:
                    ex.append(cid)
                st.session_state["vyvar_var_detection_exclusions"] = ex
                st.success("Označené — po rerun zmizne zo zoznamu.")

    st.markdown(
        "#### Info\nHviezdy pridané do sledovania VAR budú zahrnuté do PSF modelu "
        "pri ďalšom spustení MASTERSTAR generovania."
    )
