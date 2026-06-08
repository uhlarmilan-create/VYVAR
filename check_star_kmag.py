"""Check-star measured KMAG for AAVSO ensemble exports (additive reporting)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import AppConfig
from gaia_catalog_id import normalize_gaia_source_id
from photometry_core import (
    check_comparison_stability,
    ensemble_normalize,
)

_GAIA_ID_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}


def normalize_comp_df_export_columns(
    comp_df: pd.DataFrame,
    comp_quality_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = comp_df.copy()
    if df.empty:
        return df
    if "p2p_rms" not in df.columns and "comp_rms" in df.columns:
        df["p2p_rms"] = df["comp_rms"]
    if "w_rel" not in df.columns and "comp_weight" in df.columns:
        from photometry_core import apply_comp_w_rel_for_display

        df = apply_comp_w_rel_for_display(df, comp_quality_map)
    return df


def select_check_star(comp_df: pd.DataFrame, n_comp_min: int = 3) -> pd.Series | None:
    """Pick check star — best stability; only if >= n_comp_min comps remain."""
    if comp_df is None or comp_df.empty:
        return None

    df = normalize_comp_df_export_columns(comp_df)
    try:
        if "is_check_star" in df.columns:
            mchk = df["is_check_star"].fillna(False).astype(bool)
            if bool(mchk.any()):
                return df.loc[mchk].iloc[0]
    except Exception:  # noqa: BLE001
        pass

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip().str.lower() == "good"]
    if len(df) <= int(n_comp_min):
        return None

    for col in ("is_ensemble", "in_ensemble", "used_in_ensemble", "ensemble", "is_used"):
        if col in df.columns:
            try:
                m = df[col].fillna(False).astype(bool)
                df = df.loc[~m].copy()
            except Exception:  # noqa: BLE001
                pass
            break

    tier_col = "tier" if "tier" in df.columns else ("comp_tier" if "comp_tier" in df.columns else None)
    if tier_col is not None:
        df[tier_col] = pd.to_numeric(df[tier_col], errors="coerce")

    if "p2p_rms" in df.columns:
        df["p2p_rms"] = pd.to_numeric(df["p2p_rms"], errors="coerce")
        tiers = (1, 2, 3) if tier_col is not None else (None,)
        for tier in tiers:
            cand = df
            if tier is not None:
                cand = cand[(cand[tier_col].notna()) & (cand[tier_col].astype(int) == int(tier))]
            cand = cand[cand["p2p_rms"].notna()].sort_values("p2p_rms", ascending=True)
            if not cand.empty:
                return cand.iloc[0]

    if "comp_rms" in df.columns:
        df["comp_rms"] = pd.to_numeric(df["comp_rms"], errors="coerce")
        tiers = (1, 2, 3) if tier_col is not None else (None,)
        for tier in tiers:
            cand = df
            if tier is not None:
                cand = cand[(cand[tier_col].notna()) & (cand[tier_col].astype(int) == int(tier))]
            cand = cand[cand["comp_rms"].notna()].sort_values("comp_rms", ascending=True)
            if not cand.empty:
                return cand.iloc[0]

    if "comp_score" in df.columns:
        df["comp_score"] = pd.to_numeric(df["comp_score"], errors="coerce")
        cand = df[df["comp_score"].notna()].sort_values("comp_score", ascending=False)
        if not cand.empty:
            return cand.iloc[0]

    try:
        return df.iloc[0]
    except Exception:  # noqa: BLE001
        return None


def comp_ensemble_maps(
    comp_df: pd.DataFrame,
    cfg: AppConfig,
) -> tuple[dict[str, float], dict[str, int], dict[str, float], dict[int, float]]:
    tier_weights = {
        1: float(cfg.comp_tier1_weight),
        2: float(cfg.comp_tier2_weight),
        3: float(cfg.comp_tier3_weight),
        4: float(cfg.comp_tier4_weight),
    }
    for k in list(tier_weights.keys()):
        try:
            v = float(tier_weights[k])
        except Exception:  # noqa: BLE001
            v = float("nan")
        tier_weights[k] = max(0.01, v) if math.isfinite(v) and v > 0 else 0.01

    comp_catalog_mag: dict[str, float] = {}
    comp_tier_map: dict[str, int] = {}
    comp_rms_map: dict[str, float] = {}
    if comp_df is None or comp_df.empty:
        return comp_catalog_mag, comp_tier_map, comp_rms_map, tier_weights

    for _, r in comp_df.iterrows():
        cid = str(normalize_gaia_source_id(r.get("catalog_id", "")) or "").strip()
        if not cid:
            continue
        mag_v = pd.to_numeric(r.get("mag", r.get("catalog_mag", float("nan"))), errors="coerce")
        comp_catalog_mag[cid] = float(mag_v) if math.isfinite(float(mag_v)) else float("nan")
        tier = int(pd.to_numeric(r.get("comp_tier", 4), errors="coerce") or 4)
        comp_tier_map[cid] = max(1, min(4, tier))
        rms_raw = float(pd.to_numeric(r.get("comp_rms", float("nan")), errors="coerce"))
        tw = float(tier_weights.get(comp_tier_map[cid], 0.25))
        if math.isfinite(rms_raw) and rms_raw > 1e-6 and math.isfinite(tw) and tw > 0:
            comp_rms_map[cid] = rms_raw / math.sqrt(tw)
        else:
            comp_rms_map[cid] = rms_raw
    return comp_catalog_mag, comp_tier_map, comp_rms_map, tier_weights


def compute_check_ensemble_mag_calib(
    check_cid: str,
    comp_ids: list[str],
    comp_lc: dict[str, np.ndarray],
    comp_catalog_mag: dict[str, float],
    comp_quality: dict[str, dict],
    *,
    comp_rms_map: dict[str, float],
    comp_tier_map: dict[str, int],
    tier_weights: dict[int, float],
    cfg: AppConfig,
) -> np.ndarray | None:
    """Ensemble-standardize the check star, excluding it from its own ensemble."""
    cid = str(normalize_gaia_source_id(check_cid) or "").strip()
    if not cid or cid not in comp_lc:
        return None
    other_ids = [c for c in comp_ids if c != cid and c in comp_lc]
    if len(other_ids) < 3:
        return None
    other_lc = {c: comp_lc[c] for c in other_ids}
    other_cat = {c: comp_catalog_mag[c] for c in other_ids if c in comp_catalog_mag}
    other_quality = {
        c: comp_quality[c]
        for c in other_ids
        if c in comp_quality and str(comp_quality[c].get("quality", "")).strip().lower() != "excluded"
    }
    if len(other_quality) < 3:
        return None
    mag_calib, _, _ = ensemble_normalize(
        comp_lc[cid],
        other_lc,
        other_cat,
        other_quality,
        comp_rms_map=comp_rms_map,
        comp_tier_map=comp_tier_map,
        tier_weights=tier_weights,
        n_comp_min=3,
        n_comp_max=int(cfg.phase01_comparison_n_comp_max),
    )
    if not np.isfinite(mag_calib).any():
        return None
    return mag_calib


def check_kmag_sidecar_path(lc_dir: Path | str, target_cid: str) -> Path:
    return Path(lc_dir) / f"check_kmag_{str(target_cid).strip()}.csv"


def save_check_kmag_sidecar(
    path: Path,
    *,
    check_cid: str,
    bjd: np.ndarray,
    source_files: list[str],
    kmag: np.ndarray,
) -> None:
    n = min(len(bjd), len(source_files), len(kmag))
    if n <= 0:
        return
    df = pd.DataFrame(
        {
            "check_catalog_id": [str(check_cid)] * n,
            "bjd": np.asarray(bjd[:n], dtype=float),
            "source_file": [str(s) for s in source_files[:n]],
            "kmag": np.round(np.asarray(kmag[:n], dtype=float), 6),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.debug("[CHECK-KMAG] wrote %s (%d rows)", path.name, n)


def _fmt_kmag(v: float) -> str:
    return f"{float(v):.3f}" if math.isfinite(float(v)) else "na"


def kmag_from_sidecar(
    sidecar_path: Path,
    source_files: list[str],
) -> tuple[list[str], str] | None:
    if not sidecar_path.is_file():
        return None
    try:
        df = pd.read_csv(sidecar_path, low_memory=False)
    except Exception:  # noqa: BLE001
        return None
    if df.empty or "kmag" not in df.columns:
        return None
    by_sf: dict[str, float] = {}
    if "source_file" in df.columns:
        for _, r in df.iterrows():
            sf = str(r.get("source_file", "") or "").strip()
            mv = float(pd.to_numeric(r.get("kmag", float("nan")), errors="coerce"))
            if sf and math.isfinite(mv):
                by_sf[sf] = mv
    per_row = [_fmt_kmag(by_sf.get(str(sf).strip(), float("nan"))) for sf in source_files]
    if not any(v != "na" for v in per_row):
        med = float(pd.to_numeric(df["kmag"], errors="coerce").median())
        if math.isfinite(med):
            return ([_fmt_kmag(med)] * len(source_files), "median-sidecar")
        return None
    finite = [float(v) for v in per_row if v != "na"]
    if len(finite) >= 2 and (max(finite) - min(finite)) <= 0.0005:
        return ([_fmt_kmag(float(np.median(finite)))] * len(source_files), "median-sidecar-constant")
    return (per_row, "per-row-sidecar")


def resolve_proc_csv_dir(photometry_dir: Path, obs_group: str) -> Path | None:
    phot = Path(photometry_dir)
    setup = str(obs_group or "").strip() or phot.parent.name
    draft_dir = phot.parent.parent.parent
    proc = draft_dir / "detrended_aligned" / "lights" / setup
    return proc if proc.is_dir() else None


def _inst_mag_from_proc_row(row: pd.Series, export_method: str) -> float:
    m = str(export_method or "aperture").strip().lower()
    if m == "psf":
        from photometry_core import _coerce_bool_cell  # noqa: PLC0415

        psf_flux = float(pd.to_numeric(row.get("psf_flux", float("nan")), errors="coerce"))
        psf_ok = _coerce_bool_cell(row.get("psf_fit_ok"))
        if psf_ok and math.isfinite(psf_flux) and psf_flux > 0:
            return float(-2.5 * math.log10(psf_flux))
        return float("nan")
    mv = float(pd.to_numeric(row.get("mag_inst", float("nan")), errors="coerce"))
    if math.isfinite(mv):
        return mv
    flux = float(pd.to_numeric(row.get("dao_flux", float("nan")), errors="coerce"))
    if math.isfinite(flux) and flux > 0:
        return float(-2.5 * math.log10(flux))
    return float("nan")


def build_aligned_comp_inst(
    proc_dir: Path,
    comp_ids: list[str],
    source_files: list[str],
    cfg: AppConfig,
    export_method: str,
    *,
    csv_cache: dict[str, pd.DataFrame] | None = None,
) -> dict[str, np.ndarray]:
    n = len(source_files)
    out: dict[str, np.ndarray] = {cid: np.full(n, float("nan"), dtype=float) for cid in comp_ids}
    cache = csv_cache if csv_cache is not None else {}
    m = str(export_method or "aperture").strip().lower()

    for i, sf in enumerate(source_files):
        sf_name = str(sf or "").strip()
        if not sf_name:
            continue
        path = proc_dir / sf_name
        if not path.is_file():
            continue
        key = str(path)
        csv_df = cache.get(key)
        if csv_df is None:
            try:
                csv_df = pd.read_csv(path, low_memory=False, dtype=_GAIA_ID_DTYPE)
            except Exception:  # noqa: BLE001
                continue
            cache[key] = csv_df
        id_col = "catalog_id" if "catalog_id" in csv_df.columns else "name"
        work = csv_df
        if m == "psf" or "_nid" not in work.columns:
            work = csv_df.copy()
            work["_nid"] = work[id_col].map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
        for cid in comp_ids:
            sub = work[work["_nid"] == cid]
            if sub.empty:
                continue
            if m == "psf":
                out[cid][i] = _inst_mag_from_proc_row(sub.iloc[0], "psf")
            else:
                flux = float(pd.to_numeric(sub.iloc[0].get("dao_flux", float("nan")), errors="coerce"))
                if math.isfinite(flux) and flux > 0:
                    out[cid][i] = float(-2.5 * math.log10(flux))
    return out


def kmag_values_for_export(
    check_row: pd.Series | None,
    comp_df: pd.DataFrame,
    lc_normal: pd.DataFrame,
    *,
    target_cid: str,
    lc_dir: Path | None,
    proc_dir: Path | None,
    comp_quality_map: dict[str, str] | None,
    cfg: AppConfig,
    export_method: str = "aperture",
    proc_csv_cache: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[str], str]:
    """Return KMAG strings aligned to ``lc_normal`` rows."""
    n_rows = len(lc_normal)
    if n_rows == 0:
        return ([], "na")
    if check_row is None:
        return (["na"] * n_rows, "na")

    check_cid = str(normalize_gaia_source_id(check_row.get("catalog_id", "")) or "").strip()
    if not check_cid:
        return (["na"] * n_rows, "na")

    source_files = lc_normal.get("source_file", pd.Series([""] * n_rows)).astype(str).tolist()

    if lc_dir is not None:
        side = kmag_from_sidecar(check_kmag_sidecar_path(lc_dir, target_cid), source_files)
        if side is not None:
            return side

    comp_ids: list[str] = []
    if comp_df is not None and not comp_df.empty and "catalog_id" in comp_df.columns:
        for _, crow in comp_df.iterrows():
            cid = str(normalize_gaia_source_id(crow.get("catalog_id", "")) or "").strip()
            if not cid:
                continue
            if comp_quality_map and str(comp_quality_map.get(cid, "") or "").strip().lower() == "excluded":
                continue
            if cid not in comp_ids:
                comp_ids.append(cid)
    if check_cid not in comp_ids:
        comp_ids.append(check_cid)
    other_ids = [c for c in comp_ids if c != check_cid]
    if len(other_ids) < 3 or proc_dir is None or not proc_dir.is_dir():
        return (["na"] * n_rows, "na")

    comp_lc = build_aligned_comp_inst(
        proc_dir,
        comp_ids,
        source_files,
        cfg,
        export_method,
        csv_cache=proc_csv_cache,
    )
    comp_catalog_mag, comp_tier_map, comp_rms_map, tier_weights = comp_ensemble_maps(comp_df, cfg)
    comp_quality = check_comparison_stability(
        {c: comp_lc[c] for c in other_ids if c in comp_lc},
        comp_rms_map=comp_rms_map,
        n_comp_min=3,
        outlier_sigma=3.0,
        common_mode_detrend=True,
    )
    if comp_quality_map:
        for cid, q in comp_quality_map.items():
            q2 = str(q or "").strip().lower()
            if cid in comp_quality:
                if q2 == "excluded":
                    comp_quality[cid]["quality"] = "excluded"
                elif q2 in ("good", "suspect"):
                    comp_quality[cid]["quality"] = q2

    kmag_arr = compute_check_ensemble_mag_calib(
        check_cid,
        comp_ids,
        comp_lc,
        comp_catalog_mag,
        comp_quality,
        comp_rms_map=comp_rms_map,
        comp_tier_map=comp_tier_map,
        tier_weights=tier_weights,
        cfg=cfg,
    )
    if kmag_arr is None:
        return (["na"] * n_rows, "na")

    per_row = [_fmt_kmag(float(kmag_arr[i])) if i < len(kmag_arr) else "na" for i in range(n_rows)]
    if not any(v != "na" for v in per_row):
        return (["na"] * n_rows, "na")
    finite = [float(v) for v in per_row if v != "na"]
    if len(finite) >= 2 and (max(finite) - min(finite)) <= 0.0005:
        return ([_fmt_kmag(float(np.median(finite)))] * n_rows, "median-ensemble-constant")
    return (per_row, "per-row-ensemble")
