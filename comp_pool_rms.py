"""Globálny RMS výpočet pre comp pool (jedna pass naprieč framami).

Oddelené od ``photometry_core`` kvôli objemu a cyklickým importom.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

_POOL_USECOLS: list[str] = [
    "name",
    "catalog_id",
    "bjd_tdb_mid",
    "flux",
    "dao_flux",
    "noise_floor_adu",
    "aperture_r_px",
    "snr50_ok",
    "vsx_known_variable",
    "likely_saturated",
]


def _norm_id_val(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        return str(int(float(s)))
    except Exception:  # noqa: BLE001
        return s


def _norm_id_series(s: pd.Series) -> pd.Series:
    return s.apply(_norm_id_val)


def sort_per_frame_csv_paths(
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
) -> list[Path]:
    """Sort frames by median ``bjd_tdb_mid`` per CSV, then path (deterministic time order)."""

    def _key(p: Path) -> tuple[float, str]:
        bjd = float("inf")
        df = csv_cache.get(str(p))
        if df is not None and not getattr(df, "empty", True) and "bjd_tdb_mid" in df.columns:
            med = float(pd.to_numeric(df["bjd_tdb_mid"], errors="coerce").median())
            if math.isfinite(med):
                bjd = med
        return (bjd, str(p).casefold())

    return sorted(per_frame_csv_paths, key=_key)


def compute_global_pool_rms_map(
    cand_ids: set[str],
    _masterstars_df: pd.DataFrame,
    per_frame_csv_paths: list[Path],
    csv_cache: dict[str, pd.DataFrame],
    *,
    flux_col: str = "dao_flux",
    min_frames_frac: float = 0.3,
    edge_bad_frame_frac_max: float = 0.10,
    max_psf_chi2: float = 3.0,
    max_fwhm_factor: float = 1.5,
    fwhm_px: float = 3.7,
    chip_fw: int | None = None,
    chip_fh: int | None = None,
    max_comp_rms: float = 0.05,
) -> dict[str, float]:
    """Vráti ``{star_id: comp_rms}`` po rovnakom flux→RMS reťazci ako Fáza 1 (bez per-target ensemble).

    ``star_id`` zodpovedá ``name`` stĺpcu v per-frame CSV (fallback ``catalog_id``).
    """
    if not cand_ids:
        return {}

    _sorted_cids = sorted(cand_ids)
    flux_map: dict[str, list[float]] = {cid: [] for cid in _sorted_cids}
    peak_over_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    peak_total_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    edge_bad_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    edge_total_map: dict[str, int] = {cid: 0 for cid in _sorted_cids}
    snr_map: dict[str, list[float]] = {cid: [] for cid in _sorted_cids}
    psf_chi2_map: dict[str, list[float]] = {cid: [] for cid in _sorted_cids}
    fwhm_map: dict[str, list[float]] = {cid: [] for cid in _sorted_cids}
    frame_fwhm_medians: list[float] = []

    _chip_w_eff: int | None = int(chip_fw) if chip_fw is not None else None
    _chip_h_eff: int | None = int(chip_fh) if chip_fh is not None else None
    _edge_log_done = False
    n_frames_loaded = 0
    avail_cols = _POOL_USECOLS.copy()

    for csv_path in sort_per_frame_csv_paths(per_frame_csv_paths, csv_cache):
        df = csv_cache.get(str(csv_path))
        if df is None or df.empty:
            continue
        try:
            name_col = "name" if "name" in df.columns else ("catalog_id" if "catalog_id" in df.columns else "name")
            actual_flux_col = flux_col if flux_col in df.columns else "flux"

            if (_chip_w_eff is None or _chip_h_eff is None) and ("x" in df.columns and "y" in df.columns):
                try:
                    _xmax = float(pd.to_numeric(df["x"], errors="coerce").max())
                    _ymax = float(pd.to_numeric(df["y"], errors="coerce").max())
                except Exception:  # noqa: BLE001
                    _xmax, _ymax = float("nan"), float("nan")
                if math.isfinite(_xmax) and _xmax > 0:
                    _chip_w_eff = max(int(_chip_w_eff or 0), int(math.ceil(_xmax)) + 2)
                if math.isfinite(_ymax) and _ymax > 0:
                    _chip_h_eff = max(int(_chip_h_eff or 0), int(math.ceil(_ymax)) + 2)

            have_edge_cols = (
                "x" in df.columns
                and "y" in df.columns
                and "sky_annulus_r_out_px" in df.columns
                and _chip_w_eff is not None
                and _chip_h_eff is not None
                and int(_chip_w_eff) > 0
                and int(_chip_h_eff) > 0
            )
            if have_edge_cols and not _edge_log_done:
                logging.info(
                    "[GLOBAL COMP POOL RMS] chip=%sx%s px — edge kontrola zo sky_annulus_r_out_px",
                    int(_chip_w_eff),
                    int(_chip_h_eff),
                )
                _edge_log_done = True

            _cand = df[df[name_col].isin(cand_ids)]

            if "peak_max_adu" in df.columns and "saturate_limit_adu_85pct" in df.columns and not _cand.empty:
                sp = _cand[[name_col, "peak_max_adu", "saturate_limit_adu_85pct"]].copy()
                sp["_peak"] = pd.to_numeric(sp["peak_max_adu"], errors="coerce")
                sp["_limit"] = pd.to_numeric(sp["saturate_limit_adu_85pct"], errors="coerce")
                sp = sp[sp["_limit"].gt(0) & sp["_peak"].notna() & sp["_limit"].notna()]
                if not sp.empty:
                    sp["_over"] = sp["_peak"] > sp["_limit"]
                    for cid, n_tot in sp.groupby(name_col, sort=True).size().items():
                        cid_s = str(cid)
                        peak_total_map[cid_s] = int(peak_total_map.get(cid_s, 0)) + int(n_tot)
                    for cid, n_over in sp.loc[sp["_over"]].groupby(name_col, sort=True).size().items():
                        cid_s = str(cid)
                        peak_over_map[cid_s] = int(peak_over_map.get(cid_s, 0)) + int(n_over)

            if "psf_chi2" in df.columns and not _cand.empty:
                sp = _cand[[name_col, "psf_chi2"]].copy()
                sp["_chi2"] = pd.to_numeric(sp["psf_chi2"], errors="coerce")
                sp = sp[sp["_chi2"].gt(0)]
                for cid, vals in sp.groupby(name_col, sort=True)["_chi2"]:
                    psf_chi2_map[str(cid)].extend(vals.astype(float).tolist())

            if "fwhm_estimate_px" in df.columns:
                _fwhm_col = pd.to_numeric(df["fwhm_estimate_px"], errors="coerce")
                _frame_fwhm_med = float(_fwhm_col.median())
                if math.isfinite(_frame_fwhm_med) and _frame_fwhm_med > 0:
                    frame_fwhm_medians.append(_frame_fwhm_med)
                if not _cand.empty:
                    sp = _cand[[name_col, "fwhm_estimate_px"]].copy()
                    sp["_fwhm"] = pd.to_numeric(sp["fwhm_estimate_px"], errors="coerce")
                    sp = sp[sp["_fwhm"].gt(0)]
                    for cid, vals in sp.groupby(name_col, sort=True)["_fwhm"]:
                        fwhm_map[str(cid)].extend(vals.astype(float).tolist())

            sub = df[df[name_col].isin(cand_ids) & df[actual_flux_col].gt(0)].copy()
            if sub.empty:
                continue

            mag_col_frame = "mag" if "mag" in sub.columns else None
            if mag_col_frame and mag_col_frame in sub.columns:
                sub = sub.copy()
                sub["_mag_num"] = pd.to_numeric(sub[mag_col_frame], errors="coerce")
                sub["_mag_bin"] = (sub["_mag_num"] / 0.5).apply(
                    lambda x: int(x) if math.isfinite(x) else -1
                )
                bin_meds: dict[int, float] = {}
                for b, grp in sub.groupby("_mag_bin"):
                    bmed = float(grp[actual_flux_col].median())
                    if math.isfinite(bmed) and bmed > 0:
                        bin_meds[int(b)] = bmed
                if not bin_meds:
                    continue
            else:
                frame_med = float(sub[actual_flux_col].median())
                if not math.isfinite(frame_med) or frame_med <= 0:
                    continue
                bin_meds = {}

            n_frames_loaded += 1
            sub_work = sub.copy()
            raw_flux = pd.to_numeric(sub_work[actual_flux_col], errors="coerce")
            sub_work["_raw_flux"] = raw_flux

            if bin_meds:
                _bin_keys = np.fromiter(bin_meds.keys(), dtype=np.int64)

                def _norm_med_for_bin(b: int) -> float:
                    bi = int(b)
                    if bi in bin_meds:
                        return float(bin_meds[bi])
                    if len(_bin_keys) == 0:
                        return float("nan")
                    ck = int(_bin_keys[int(np.argmin(np.abs(_bin_keys - bi)))])
                    return float(bin_meds[ck])

                sub_work["_norm_med"] = sub_work["_mag_bin"].map(_norm_med_for_bin)
            else:
                sub_work["_norm_med"] = float(frame_med)

            sub_work["_rel"] = sub_work["_raw_flux"] / pd.to_numeric(sub_work["_norm_med"], errors="coerce")
            # Safe: _norm_med == 0 rows filtered by _rel_ok mask before use.
            _rel_ok = sub_work["_rel"].notna() & np.isfinite(sub_work["_rel"].to_numpy(dtype=np.float64))
            _rel_ok = _rel_ok & sub_work["_rel"].gt(0)

            if have_edge_cols:
                x0 = pd.to_numeric(sub_work["x"], errors="coerce")
                y0 = pd.to_numeric(sub_work["y"], errors="coerce")
                r_out = pd.to_numeric(sub_work["sky_annulus_r_out_px"], errors="coerce")
                w = float(int(_chip_w_eff))
                h = float(int(_chip_h_eff))
                _edge_valid = (
                    x0.notna()
                    & y0.notna()
                    & r_out.notna()
                    & r_out.gt(0)
                    & np.isfinite(x0.to_numpy(dtype=np.float64))
                    & np.isfinite(y0.to_numpy(dtype=np.float64))
                    & np.isfinite(r_out.to_numpy(dtype=np.float64))
                )
                _edge_ok = _edge_valid & (x0 - r_out >= 0.0) & (x0 + r_out <= w) & (y0 - r_out >= 0.0) & (y0 + r_out <= h)
                sub_work["_edge_count"] = _edge_valid.astype(np.int64)
                sub_work["_edge_bad"] = (_edge_valid & ~_edge_ok).astype(np.int64)
            else:
                sub_work["_edge_count"] = 0
                sub_work["_edge_bad"] = 0

            if "dao_flux" in sub_work.columns:
                flux_snr = pd.to_numeric(sub_work["dao_flux"], errors="coerce")
                flux_snr = flux_snr.where(flux_snr.notna(), sub_work["_raw_flux"])
            else:
                flux_snr = sub_work["_raw_flux"].copy()
            sky = pd.to_numeric(sub_work.get("noise_floor_adu", pd.Series(0.0, index=sub_work.index)), errors="coerce")
            r_ap = pd.to_numeric(sub_work.get("aperture_r_px", pd.Series(7.0, index=sub_work.index)), errors="coerce")
            area = np.pi * r_ap * r_ap
            denom = flux_snr + np.maximum(0.0, sky) * area
            _snr_ok = (
                flux_snr.gt(0)
                & sky.notna()
                & area.notna()
                & np.isfinite(flux_snr.to_numpy(dtype=np.float64))
                & np.isfinite(sky.to_numpy(dtype=np.float64))
                & np.isfinite(area.to_numpy(dtype=np.float64))
                & denom.gt(0)
            )
            sub_work["_snr"] = np.where(_snr_ok, flux_snr / np.sqrt(denom), np.nan)

            if have_edge_cols:
                for cid, grp in sub_work.groupby(name_col, sort=True):
                    cid_s = str(cid)
                    edge_total_map[cid_s] = int(edge_total_map.get(cid_s, 0)) + int(grp["_edge_count"].sum())
                    edge_bad_map[cid_s] = int(edge_bad_map.get(cid_s, 0)) + int(grp["_edge_bad"].sum())

            for cid, grp in sub_work.loc[_rel_ok].groupby(name_col, sort=True):
                cid_s = str(cid)
                flux_map[cid_s].extend(grp["_rel"].astype(float).tolist())

            for cid, grp in sub_work.groupby(name_col, sort=True):
                cid_s = str(cid)
                snr_vals = grp["_snr"].to_numpy(dtype=np.float64)
                snr_vals = snr_vals[np.isfinite(snr_vals)]
                if snr_vals.size > 0:
                    snr_map[cid_s].extend(snr_vals.astype(float).tolist())

        except Exception as _pool_rms_exc:  # noqa: BLE001
            LOGGER.debug(
                "[POOL_RMS] frame %s skipped: %s",
                getattr(csv_path, "name", csv_path),
                _pool_rms_exc,
            )
            continue

    logging.info(
        "[PERF-4] comp_pool_rms: %d frames loaded, %d paths × %d candidates",
        int(n_frames_loaded),
        len(per_frame_csv_paths),
        len(cand_ids),
    )

    min_frames = max(3, int(n_frames_loaded * float(min_frames_frac)))

    for cid in sorted(flux_map.keys()):
        total = int(peak_total_map.get(cid, 0) or 0)
        over = int(peak_over_map.get(cid, 0) or 0)
        if total >= 10 and total > 0 and (float(over) / float(total)) > 0.10:
            flux_map.pop(cid, None)

    try:
        bad_thr = float(edge_bad_frame_frac_max)
    except (TypeError, ValueError):
        bad_thr = 0.10
    if not math.isfinite(bad_thr) or bad_thr < 0:
        bad_thr = 0.10
    for cid in sorted(flux_map.keys()):
        total_e = int(edge_total_map.get(cid, 0) or 0)
        bad_e = int(edge_bad_map.get(cid, 0) or 0)
        if total_e > 0:
            bad_frac = float(bad_e) / float(total_e) if total_e > 0 else 0.0
            if bad_frac > bad_thr:
                flux_map.pop(cid, None)

    for cid in sorted(flux_map.keys()):
        snrs = snr_map.get(cid, [])
        if len(snrs) >= 5:
            snr_median = float(np.median(np.asarray(snrs, dtype=np.float64)))
            if math.isfinite(snr_median) and snr_median < 5.0:
                flux_map.pop(cid, None)

    _global_fwhm_med = float(np.median(frame_fwhm_medians)) if frame_fwhm_medians else float("nan")
    _b_rejected: set[str] = set()

    if math.isfinite(max_psf_chi2):
        for _cid, _chi2_vals in psf_chi2_map.items():
            valid = [v for v in _chi2_vals if math.isfinite(v) and v > 0]
            if len(valid) < 3:
                continue  # not enough valid PSF data — skip filter
            _med_chi2 = float(np.median(valid))
            if _med_chi2 > max_psf_chi2:
                _b_rejected.add(_cid)

    if math.isfinite(max_fwhm_factor) and math.isfinite(_global_fwhm_med) and _global_fwhm_med > 0:
        for _cid, _fwhm_vals in fwhm_map.items():
            if len(_fwhm_vals) < 3:
                continue
            _med_fwhm = float(np.median(_fwhm_vals))
            _fwhm_ratio = _med_fwhm / _global_fwhm_med
            if _fwhm_ratio > max_fwhm_factor:
                _b_rejected.add(_cid)

    for _cid in _b_rejected:
        flux_map.pop(_cid, None)

    for cid in sorted(flux_map.keys()):
        vals = flux_map[cid]
        if len(vals) < 6:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        t = np.linspace(0.0, 1.0, len(arr))
        try:
            coeffs = np.polyfit(t, arr, 2)
            trend_fit = np.polyval(coeffs, t)
            safe_trend = np.where(np.abs(trend_fit) > 1e-9, trend_fit, 1.0)
            detrended = arr / safe_trend
            med_dt = float(np.median(detrended))
            if math.isfinite(med_dt) and med_dt > 0:
                flux_map[cid] = (detrended / med_dt).tolist()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[POOL_RMS] Detrend fallback per star (non-critical): %s", exc)

    rms_map: dict[str, float] = {}
    for cid, vals in sorted(flux_map.items(), key=lambda kv: str(kv[0])):
        if len(vals) < min_frames:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        rms = float(np.sqrt(np.mean((arr - 1.0) ** 2)))
        if math.isfinite(rms):
            rms_map[cid] = rms

    ISOLATED_BIN_RMS_FLOOR = 1e-4
    ISOLATED_BIN_MIN_FRAMES = 50
    for cid in list(rms_map.keys()):
        try:
            rms_v = float(rms_map.get(cid, float("nan")))
        except Exception:  # noqa: BLE001
            rms_v = float("nan")
        nfr = int(len(flux_map.get(cid, [])))
        if math.isfinite(rms_v) and rms_v < float(ISOLATED_BIN_RMS_FLOOR) and nfr >= int(ISOLATED_BIN_MIN_FRAMES):
            # Isolated brightness-bin normalization artefact → flat LC; do not use as comp.
            rms_map.pop(cid, None)
            LOGGER.debug(
                "[POOL_RMS] %s: comp_rms < %.1e at %d frames → isolated bin → excluded",
                str(cid),
                float(ISOLATED_BIN_RMS_FLOOR),
                int(nfr),
            )

    if math.isfinite(max_comp_rms) and max_comp_rms > 0:
        rms_map = {
            cid: rms
            for cid, rms in sorted(rms_map.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
            if rms <= float(max_comp_rms)
        }

    return rms_map


def attach_comp_rms_to_pool_rows(
    pool: pd.DataFrame,
    rms_map: dict[str, float],
    *,
    id_col: str,
) -> pd.DataFrame:
    """Pridá stĺpec ``comp_rms`` podľa ID zhody s ``rms_map``."""
    if pool.empty or not rms_map:
        return pool
    out = pool.copy()
    keys = out[id_col].astype(str).str.strip()

    # Build canonical normalized-id → rms map (min catalog_id wins on collision)
    _norm_rms: dict[str, float] = {}
    _norm_key: dict[str, str] = {}
    for rk, rv in sorted(rms_map.items(), key=lambda kv: (float(kv[1]), str(kv[0]))):
        nk = _norm_id_val(rk)
        if nk not in _norm_key or str(rk) < str(_norm_key[nk]):
            _norm_key[nk] = rk
            _norm_rms[nk] = float(rv)

    def _lookup(k: str) -> float:
        if k in rms_map:
            return float(rms_map[k])
        nk = _norm_id_val(k)
        return _norm_rms.get(nk, float("nan"))

    out["comp_rms"] = keys.map(_lookup)
    _cr = pd.to_numeric(out["comp_rms"], errors="coerce")
    return out[_cr.notna() & np.isfinite(_cr.to_numpy(dtype=np.float64))].reset_index(drop=True)
