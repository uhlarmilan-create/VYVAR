from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_MAD_CONSISTENCY = 0.6745  # MAD -> sigma for normal dist


def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if not math.isfinite(mad) or mad <= 0:
        return float("nan")
    return mad / _MAD_CONSISTENCY


def _norm_cid(x: Any) -> str:
    s = str(x or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    # Try exact integer parse first.
    try:
        return str(int(s))
    except Exception:
        pass
    # Scientific notation fallback (still avoid precision loss when possible).
    try:
        from decimal import Decimal, InvalidOperation

        return str(int(Decimal(s)))
    except Exception:
        pass
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_field_flux_matrix(
    per_frame_dir: Path,
    *,
    flux_col: str = "dao_flux",
    zone_filter: list[str] | None = None,
    min_frames_frac: float = 0.5,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Načíta všetky proc_*.csv a zostaví pivot tabuľku.

    Returns:
      flux_matrix_df: index=catalog_id, columns=frame_stem, values=flux_col
      metadata_df:    index=catalog_id, columns: mag, bp_rp, b_v, x, y, zone,
                      vsx_known_variable, gaia_dr3_variable_catalog, ra_deg, dec_deg
      bjd_array:      array of bjd_tdb_mid per frame column order (NaN if missing)
    """
    per_frame_dir = Path(per_frame_dir)
    cfg = config or {}
    try:
        min_frames_frac = float(cfg.get("variability_min_frames_frac", min_frames_frac))
    except Exception:  # noqa: BLE001
        pass

    if zone_filter is None:
        zone_filter = ["linear", "noisy1"]

    frames = sorted(per_frame_dir.glob("proc_*.csv"))
    if not frames:
        raise FileNotFoundError(f"Žiadne proc_*.csv v: {per_frame_dir}")

    rows: list[pd.DataFrame] = []
    frame_bjd: dict[str, float] = {}
    kept_cols = [
        "catalog_id",
        flux_col,
        "mag",
        "bp_rp",
        "b_v",
        "x",
        "y",
        "zone",
        "vsx_known_variable",
        "gaia_dr3_variable_catalog",
        "ra_deg",
        "dec_deg",
        "bjd_tdb_mid",
        "photometry_ok",
        "edge_safe_10px",
        "edge_fail",
        "snr50_ok",
        "is_saturated",
        "likely_saturated",
        "source_type",
        "is_usable",
    ]

    for p in frames:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            logging.warning("proc CSV read failed (%s): %s", p.name, exc)
            continue

        if "catalog_id" not in df.columns or flux_col not in df.columns:
            continue

        sub = df[[c for c in kept_cols if c in df.columns]].copy()
        sub["catalog_id"] = sub["catalog_id"].map(_norm_cid)
        sub = sub[sub["catalog_id"].astype(str).str.len() > 0]

        # Filters (if columns exist)
        def _apply_bool_filter(col: str, want: bool) -> None:
            nonlocal sub
            if col in sub.columns:
                sub = sub[sub[col].fillna(False).astype(bool) == bool(want)]

        _apply_bool_filter("photometry_ok", True)
        _apply_bool_filter("edge_safe_10px", True)
        _apply_bool_filter("snr50_ok", True)
        if "is_saturated" in sub.columns:
            sub = sub[~sub["is_saturated"].fillna(False).astype(bool)]
        if "likely_saturated" in sub.columns:
            sub = sub[~sub["likely_saturated"].fillna(False).astype(bool)]
        if "edge_fail" in sub.columns:
            n0 = int(len(sub))
            sub = sub[~sub["edge_fail"].fillna(False).astype(bool)]
            n1 = int(len(sub))
            logging.info("[VARIABILITY] edge_fail filter: vyradených %s meraní", int(n0 - n1))

        if "zone" in sub.columns:
            sub["zone"] = sub["zone"].fillna("").astype(str).str.strip()
            sub = sub[sub["zone"].isin(list(zone_filter))]

        # Dedupe catalog_id within frame:
        # keep GAIA_MATCHED row if present, else first row.
        if "source_type" in sub.columns:
            sub["_src_prio"] = (sub["source_type"].fillna("").astype(str) == "GAIA_MATCHED").astype(int)
            sub = (
                sub.sort_values(["catalog_id", "_src_prio"], ascending=[True, False])
                .drop_duplicates("catalog_id", keep="first")
                .drop(columns=["_src_prio"], errors="ignore")
            )
        else:
            sub = sub.drop_duplicates("catalog_id", keep="first")

        sub["_frame"] = p.stem
        rows.append(sub)

        # frame BJD (median of available values)
        if "bjd_tdb_mid" in sub.columns:
            bv = pd.to_numeric(sub["bjd_tdb_mid"], errors="coerce")
            frame_bjd[p.stem] = float(bv.median()) if bool(np.isfinite(bv).any()) else float("nan")
        else:
            frame_bjd[p.stem] = float("nan")

    if not rows:
        raise FileNotFoundError(f"proc_*.csv sa nepodarilo načítať (flux_col={flux_col})")

    all_df = pd.concat(rows, ignore_index=True)
    all_df[flux_col] = pd.to_numeric(all_df[flux_col], errors="coerce")

    # Metadata: take first non-null per catalog_id (stable across frames)
    meta_cols = [
        "mag",
        "bp_rp",
        "b_v",
        "x",
        "y",
        "zone",
        "vsx_known_variable",
        "gaia_dr3_variable_catalog",
        "ra_deg",
        "dec_deg",
        "snr50_ok",
        "is_usable",
    ]
    meta_present = [c for c in meta_cols if c in all_df.columns]
    meta = (
        all_df[["catalog_id"] + meta_present]
        .sort_values("catalog_id")
        .groupby("catalog_id", as_index=True)
        .first()
    )

    # Pivot flux matrix
    pivot = all_df.pivot_table(
        index="catalog_id",
        columns="_frame",
        values=flux_col,
        aggfunc="median",
    ).sort_index()

    # Frame-by-frame normalization using median flux of is_usable stars in that frame.
    usable_pivot = None
    if "is_usable" in all_df.columns:
        usable_pivot = (
            all_df.pivot_table(
                index="catalog_id",
                columns="_frame",
                values="is_usable",
                aggfunc="max",
            )
            .fillna(False)
            .astype(bool)
        )

    pivot_norm = pivot.copy()
    for col in pivot.columns:
        try:
            series = pd.to_numeric(pivot[col], errors="coerce")
            if usable_pivot is not None and col in usable_pivot.columns:
                usable_mask = usable_pivot[col].reindex(series.index).fillna(False).astype(bool)
                med = float(np.nanmedian(series[usable_mask].to_numpy(dtype=float)))
            else:
                med = float(np.nanmedian(series.to_numpy(dtype=float)))
            if not (math.isfinite(med) and med > 0):
                continue
            pivot_norm[col] = series / med
        except Exception:  # noqa: BLE001
            continue

    # Filter by min frames fraction
    min_frames = max(1, int(math.ceil(float(min_frames_frac) * float(pivot_norm.shape[1]))))
    n_ok = pivot_norm.notna().sum(axis=1)
    keep = n_ok >= min_frames
    pivot_norm = pivot_norm.loc[keep].copy()
    meta = meta.loc[pivot_norm.index].copy()

    bjd_array = np.asarray([float(frame_bjd.get(str(c), float("nan"))) for c in pivot_norm.columns], dtype=float)
    return pivot_norm, meta, bjd_array


def compute_rms_variability(
    flux_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    comp_catalog_ids: list,
    *,
    sigma_threshold: float = 3.0,
    vsx_targets_csv: Path | None = None,
    config: dict[str, Any] | None = None,
    comp_rms_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    RMS metóda — pre každú hviezdu vypočíta relatívnu RMS.
    """
    if flux_matrix.empty:
        return pd.DataFrame()

    # Work on numeric normalized flux (per-frame normalized so mean ~ 1).
    flux0 = flux_matrix.apply(pd.to_numeric, errors="coerce")
    n_frames_used = flux0.notna().sum(axis=1)

    cfg = config or {}
    try:
        sigma_threshold = float(cfg.get("variability_sigma_threshold", sigma_threshold))
    except Exception:  # noqa: BLE001
        pass
    try:
        sigma_clip = float(cfg.get("variability_sigma_clip", 5.0))
    except Exception:  # noqa: BLE001
        sigma_clip = 5.0
    try:
        min_points_rms = int(cfg.get("variability_min_points_rms", 20))
    except Exception:  # noqa: BLE001
        min_points_rms = 20
    try:
        p85_filter = int(cfg.get("variability_p85_filter", 85))
    except Exception:  # noqa: BLE001
        p85_filter = 85
    try:
        slope_floor = float(cfg.get("variability_slope_floor", 0.02))
    except Exception:  # noqa: BLE001
        slope_floor = 0.02
    try:
        smoothness_max = float(cfg.get("variability_smoothness_max", 0.80))
    except Exception:  # noqa: BLE001
        smoothness_max = 0.80
    try:
        mag_limit = float(cfg.get("variability_mag_limit", 14.5))
    except Exception:  # noqa: BLE001
        mag_limit = 14.5
    try:
        min_frames = int(cfg.get("variability_min_frames", 30))
    except Exception:  # noqa: BLE001
        min_frames = 30
    try:
        min_rms_pct = float(cfg.get("variability_min_rms_pct", 1.5))
    except Exception:  # noqa: BLE001
        min_rms_pct = 1.5
    try:
        clip_ratio_min = float(cfg.get("variability_clip_ratio_min", 0.80))
    except Exception:  # noqa: BLE001
        clip_ratio_min = 0.80
    try:
        min_amplitude_mag = float(cfg.get("variability_min_amplitude_mag", 0.01))
    except Exception:  # noqa: BLE001
        min_amplitude_mag = 0.01

    # Sigma clipping per star before RMS (MAD-based).
    rms_map: dict[str, float] = {}
    n_used_map: dict[str, int] = {}
    mean_clean_map: dict[str, float] = {}
    smooth_map: dict[str, float] = {}
    amplitude_map: dict[str, float] = {}
    for cid, row in flux0.iterrows():
        vals = pd.to_numeric(row, errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size < int(min_points_rms):
            rms_map[str(cid)] = float("nan")
            n_used_map[str(cid)] = int(vals.size)
            mean_clean_map[str(cid)] = float("nan")
            smooth_map[str(cid)] = float("nan")
            amplitude_map[str(cid)] = float("nan")
            continue
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        sigma_mad = mad / _MAD_CONSISTENCY if mad > 0 and math.isfinite(mad) else float("nan")
        if not (math.isfinite(sigma_mad) and sigma_mad > 0):
            vals_clean = vals
        else:
            mask = np.abs(vals - med) < float(sigma_clip) * sigma_mad
            vals_clean = vals[mask]
        if vals_clean.size < int(min_points_rms):
            rms_map[str(cid)] = float("nan")
            n_used_map[str(cid)] = int(vals_clean.size)
            mean_clean_map[str(cid)] = float("nan")
            smooth_map[str(cid)] = float("nan")
            amplitude_map[str(cid)] = float("nan")
            continue
        mu = float(np.mean(vals_clean))
        sig = float(np.std(vals_clean))
        rms_map[str(cid)] = (sig / mu) * 100.0 if (math.isfinite(mu) and mu != 0) else float("nan")
        n_used_map[str(cid)] = int(vals_clean.size)
        mean_clean_map[str(cid)] = float(mu) if math.isfinite(mu) else float("nan")
        # Smoothness ratio (Abbe-like p2p RMS / total RMS): <0.8 trend, >0.8 noise.
        if vals_clean.size < 2:
            smooth_map[str(cid)] = float("nan")
        else:
            diffs = np.diff(vals_clean)
            p2p_rms = float(np.std(diffs)) / float(np.sqrt(2.0)) if diffs.size else float("nan")
            total_rms = float(np.std(vals_clean))
            smooth_map[str(cid)] = (p2p_rms / total_rms) if (math.isfinite(total_rms) and total_rms > 0) else float("nan")

        # Amplitúda = p95 - p05 z mag hodnôt (robustný odhad peak-to-peak)
        flux_arr = vals_clean
        flux_arr = flux_arr[np.isfinite(flux_arr) & (flux_arr > 0)]
        if flux_arr.size >= 10:
            mag_arr = -2.5 * np.log10(flux_arr)
            amplitude = float(np.percentile(mag_arr, 95) - np.percentile(mag_arr, 5))
        else:
            amplitude = float("nan")
        amplitude_map[str(cid)] = amplitude

    rms_pct = pd.Series(rms_map)
    n_frames_used_clean = pd.Series(n_used_map).astype(int)
    mean_flux_norm_clean = pd.Series(mean_clean_map)
    smoothness_ratio = pd.Series(smooth_map)
    amplitude_mag = pd.Series(amplitude_map)

    out = pd.DataFrame(
        {
            "catalog_id": flux_matrix.index.astype(str),
            "rms_pct": pd.to_numeric(rms_pct, errors="coerce"),
            "n_frames_used": pd.to_numeric(n_frames_used, errors="coerce").astype(int),
            "n_frames_used_clean": pd.to_numeric(n_frames_used_clean, errors="coerce").astype(int),
            "mean_flux_norm_clean": pd.to_numeric(mean_flux_norm_clean, errors="coerce"),
            "smoothness_ratio": pd.to_numeric(smoothness_ratio, errors="coerce"),
            "amplitude_mag": pd.to_numeric(amplitude_mag, errors="coerce"),
        }
    ).set_index("catalog_id", drop=False)

    # Attach metadata
    if metadata is not None and not metadata.empty:
        meta = metadata.copy()
        meta.index = meta.index.astype(str)
        out = out.join(meta, how="left")

    # VSX crossmatch from variable_targets.csv (optional).
    vsx_ids: set[str] = set()
    vsx_meta: dict[str, tuple[str, str]] = {}
    if vsx_targets_csv is not None:
        try:
            vtp = Path(vsx_targets_csv)
            if vtp.exists():
                vtdf = pd.read_csv(vtp, low_memory=False)
                if "catalog_id" in vtdf.columns:
                    vtdf["_cid"] = vtdf["catalog_id"].map(_norm_cid)
                    vsx_ids = set(vtdf["_cid"].dropna().astype(str).tolist())
                    for _, r in vtdf.iterrows():
                        cid = str(r.get("_cid", "") or "")
                        if not cid:
                            continue
                        vsx_meta[cid] = (
                            str(r.get("vsx_name", "") or ""),
                            str(r.get("vsx_type", "") or ""),
                        )
        except Exception:  # noqa: BLE001
            pass
    out["vsx_match"] = out.index.isin(vsx_ids)
    out["vsx_name"] = out.index.map(lambda c: (vsx_meta.get(str(c), ("", ""))[0]))
    out["vsx_type"] = out.index.map(lambda c: (vsx_meta.get(str(c), ("", ""))[1]))

    # ------------------------------------------------------------------
    # Hockey stick fit calibrated to the whole field (stable stars),
    # not only COMP (COMP can be "too good" and underestimates envelope).
    # ------------------------------------------------------------------
    field_mask = pd.Series(True, index=out.index)
    if "is_usable" in out.columns:
        field_mask &= out["is_usable"].fillna(False).astype(bool)
    if "vsx_known_variable" in out.columns:
        field_mask &= ~out["vsx_known_variable"].fillna(False).astype(bool)
    if "gaia_dr3_variable_catalog" in out.columns:
        field_mask &= ~out["gaia_dr3_variable_catalog"].fillna(False).astype(bool)

    field = out.loc[field_mask].copy()
    field_mag = pd.to_numeric(field.get("mag"), errors="coerce")
    field_rms = pd.to_numeric(field.get("rms_pct"), errors="coerce")
    ok = field_mag.notna() & field_rms.notna() & np.isfinite(field_mag) & np.isfinite(field_rms) & (field_rms > 0)
    field_mag = field_mag[ok].astype(float)
    field_rms = field_rms[ok].astype(float)

    # Remove top tail RMS as potential variables (field calibration).
    if len(field_rms) >= 20:
        p85 = float(np.nanpercentile(field_rms.to_numpy(dtype=float), float(p85_filter)))
        keep = field_rms < p85
        field_mag = field_mag[keep]
        field_rms = field_rms[keep]

    # Hockey stick: log10(rms_pct) = a + b * mag
    expected = pd.Series(float("nan"), index=out.index, dtype=float)
    sigma_log = float("nan")
    if len(field_rms) >= 20:
        try:
            x = field_mag.to_numpy(dtype=float)
            y = np.log10(np.clip(field_rms.to_numpy(dtype=float), 1e-6, np.inf))
            # Broeg 2005: optional weights for defining the noise floor.
            w_fit = None
            try:
                if comp_rms_map:
                    ids_fit = field_mag.index.astype(str).tolist()
                    w_arr = np.ones(len(ids_fit), dtype=float)
                    for i, cid in enumerate(ids_fit):
                        rv = comp_rms_map.get(str(cid))
                        if rv is not None and math.isfinite(float(rv)) and float(rv) > 1e-4:
                            w_arr[i] = 1.0 / (float(rv) ** 2)
                    if np.isfinite(w_arr).any() and float(np.nanmax(w_arr)) > 0:
                        w_fit = w_arr / float(np.nanmax(w_arr))
            except Exception:  # noqa: BLE001
                w_fit = None

            b, a = np.polyfit(x, y, 1, w=w_fit)  # y = b*x + a
            # Constrain to minimal physical slope (noise grows with mag).
            if not math.isfinite(b):
                b = float(slope_floor)
            b = max(float(b), float(slope_floor))
            mag_all = pd.to_numeric(out.get("mag"), errors="coerce").to_numpy(dtype=float)
            fit_all = a + b * mag_all
            expected_vals = np.power(10.0, fit_all)
            expected[:] = expected_vals
            resid = y - (a + b * x)
            sigma_log = _mad_sigma(resid)
        except Exception:  # noqa: BLE001
            pass

    if expected.isna().all():
        med = float(np.nanmedian(field_rms.to_numpy(dtype=float))) if len(field_rms) else float("nan")
        expected[:] = med
        sigma_log = _mad_sigma(np.log10(np.clip(field_rms.to_numpy(dtype=float), 1e-6, np.inf)))

    out["expected_rms_pct"] = pd.to_numeric(expected, errors="coerce")

    # Score in log space
    log_rms = np.log10(np.clip(pd.to_numeric(out["rms_pct"], errors="coerce").to_numpy(dtype=float), 1e-6, np.inf))
    log_exp = np.log10(np.clip(pd.to_numeric(out["expected_rms_pct"], errors="coerce").to_numpy(dtype=float), 1e-6, np.inf))
    out["variability_score"] = (log_rms - log_exp) / float(sigma_log if math.isfinite(sigma_log) and sigma_log > 0 else float("nan"))

    # Envelope for candidates
    upper = np.power(10.0, log_exp + float(sigma_threshold) * float(sigma_log if math.isfinite(sigma_log) and sigma_log > 0 else 0.0))
    out["upper_envelope_rms_pct"] = upper
    out["is_variable_candidate"] = pd.to_numeric(out["rms_pct"], errors="coerce") > pd.to_numeric(out["upper_envelope_rms_pct"], errors="coerce")

    # Candidate-only filters (must all hold)
    # 1) clipping ratio
    ratio = pd.to_numeric(out["n_frames_used_clean"], errors="coerce") / pd.to_numeric(out["n_frames_used"], errors="coerce")
    out["clip_ratio"] = ratio
    ok_clip = ratio >= float(clip_ratio_min)

    # 2) minimum RMS above envelope floor
    env = pd.to_numeric(out["upper_envelope_rms_pct"], errors="coerce")
    thr = np.maximum(env.to_numpy(dtype=float), float(min_rms_pct))
    ok_min_rms = pd.to_numeric(out["rms_pct"], errors="coerce").to_numpy(dtype=float) > thr

    # 3) SNR + mean flux_norm clean
    ok_snr = True
    if "snr50_ok" in out.columns:
        ok_snr = out["snr50_ok"].fillna(False).astype(bool)
    ok_mean = pd.to_numeric(out["mean_flux_norm_clean"], errors="coerce") > 0.001
    ok_smooth = pd.to_numeric(out["smoothness_ratio"], errors="coerce") < float(smoothness_max)

    out["is_variable_candidate"] = out["is_variable_candidate"] & ok_clip & ok_min_rms & ok_snr & ok_mean & ok_smooth

    # 4) minimum robust amplitude in magnitudes
    out["is_variable_candidate"] = out["is_variable_candidate"] & (
        pd.to_numeric(out.get("amplitude_mag"), errors="coerce") >= float(min_amplitude_mag)
    )

    # Final candidate quality filters
    out["mag"] = pd.to_numeric(out.get("mag"), errors="coerce")
    out["is_variable_candidate"] = out["is_variable_candidate"] & (out["mag"].notna()) & (out["mag"] <= float(mag_limit))
    out["is_variable_candidate"] = out["is_variable_candidate"] & (pd.to_numeric(out["n_frames_used"], errors="coerce") >= int(min_frames))

    # Standard columns
    for c in (
        "mag",
        "bp_rp",
        "x",
        "y",
        "ra_deg",
        "dec_deg",
        "zone",
        "vsx_known_variable",
        "gaia_dr3_variable_catalog",
        "snr50_ok",
        "is_usable",
    ):
        if c not in out.columns:
            out[c] = (False if c in ("snr50_ok", "is_usable") else np.nan)

    out = out.reset_index(drop=True)
    out = out.sort_values("variability_score", ascending=False, na_position="last").reset_index(drop=True)
    return out[
        [
            "catalog_id",
            "mag",
            "bp_rp",
            "x",
            "y",
            "ra_deg",
            "dec_deg",
            "rms_pct",
            "expected_rms_pct",
            "variability_score",
            "amplitude_mag",
            "is_variable_candidate",
            "vsx_known_variable",
            "gaia_dr3_variable_catalog",
            "vsx_match",
            "vsx_name",
            "vsx_type",
            "zone",
            "snr50_ok",
            "is_usable",
            "n_frames_used",
            "n_frames_used_clean",
            "clip_ratio",
            "mean_flux_norm_clean",
            "smoothness_ratio",
            "upper_envelope_rms_pct",
        ]
    ]


def compute_vdi(
    flux_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    min_frames: int = 20,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    VDI (Variability Detection Index) — počet prechodov cez median.
    """
    cfg = config or {}
    try:
        min_frames = int(cfg.get("variability_min_frames", min_frames))
    except Exception:  # noqa: BLE001
        pass
    try:
        vdi_z_thr = float(cfg.get("variability_vdi_z_threshold", 3.0))
    except Exception:  # noqa: BLE001
        vdi_z_thr = 3.0

    if flux_matrix.empty:
        return pd.DataFrame()

    ids = flux_matrix.index.astype(str).tolist()
    vdi_vals: list[float] = []
    n_used: list[int] = []

    for cid in ids:
        fluxes = pd.to_numeric(flux_matrix.loc[cid], errors="coerce").dropna().to_numpy(dtype=float)
        if fluxes.size < int(min_frames):
            vdi_vals.append(float("nan"))
            n_used.append(int(fluxes.size))
            continue
        med = float(np.median(fluxes))
        centered = fluxes - med
        signs = np.sign(centered)
        signs = signs[signs != 0]
        if signs.size < 2:
            vdi_vals.append(0.0)
            n_used.append(int(fluxes.size))
            continue
        crossings = int(np.sum(np.diff(signs) != 0))
        vdi = float(crossings) / float(np.sqrt(float(fluxes.size)))
        vdi_vals.append(vdi)
        n_used.append(int(fluxes.size))

    vdi_arr = np.asarray(vdi_vals, dtype=float)
    med_vdi = float(np.nanmedian(vdi_arr))
    sig_vdi = _mad_sigma(vdi_arr)

    z = (vdi_arr - med_vdi) / (sig_vdi if math.isfinite(sig_vdi) and sig_vdi > 0 else float("nan"))

    out = pd.DataFrame(
        {
            "catalog_id": ids,
            "vdi_score": vdi_arr,
            "vdi_z_score": z,
            "n_frames_used": n_used,
        }
    )
    if metadata is not None and not metadata.empty:
        meta = metadata.copy()
        meta.index = meta.index.astype(str)
        out = out.set_index("catalog_id").join(meta, how="left").reset_index()

    # Premenné hviezdy majú NÍZKE VDI (málo prechodov cez medián)
    # → negatívne z-score → kandidát ak z < -threshold
    out["is_variable_candidate"] = pd.to_numeric(out["vdi_z_score"], errors="coerce") < -float(vdi_z_thr)
    if "mag" not in out.columns:
        out["mag"] = np.nan
    # Najlepší kandidáti majú najnižšie (najnegatívnejšie) z-score
    out = out.sort_values("vdi_z_score", ascending=True, na_position="last").reset_index(drop=True)
    return out[["catalog_id", "mag", "vdi_score", "vdi_z_score", "is_variable_candidate", "n_frames_used"]]

