"""Check-star measured KMAG for AAVSO ensemble exports (additive reporting)."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
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


def _norm_ensemble_id_set(ensemble_ids: set[str] | None) -> set[str]:
    out: set[str] = set()
    if not ensemble_ids:
        return out
    for raw in ensemble_ids:
        cid = str(normalize_gaia_source_id(raw) or "").strip()
        if cid:
            out.add(cid)
    return out


def _comp_rms_map_from_df(comp_df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if comp_df is None or comp_df.empty or "catalog_id" not in comp_df.columns:
        return out
    for _, r in comp_df.iterrows():
        cid = str(normalize_gaia_source_id(r.get("catalog_id", "")) or "").strip()
        if not cid:
            continue
        try:
            out[cid] = float(pd.to_numeric(r.get("comp_rms", float("nan")), errors="coerce"))
        except (TypeError, ValueError):
            out[cid] = float("nan")
    return out


def resolve_ensemble_ids_for_check(
    target_cid: str,
    comp_df: pd.DataFrame,
    *,
    lc_dir: Path | None,
    comp_quality_map: dict[str, str] | None,
    cfg: AppConfig | None,
) -> set[str]:
    """Best-effort ensemble id set for check-star exclusion (empty -> legacy pool)."""
    from photometry_core import ensemble_member_ids, parse_comp_quality_json_map  # noqa: PLC0415

    tid = str(normalize_gaia_source_id(target_cid) or "").strip()
    if not tid:
        return set()
    _cfg = cfg or AppConfig()
    cq: dict[str, dict] = {}
    if lc_dir is not None:
        cq_path = Path(lc_dir) / f"comp_quality_{tid}.json"
        if cq_path.is_file():
            try:
                raw = json.loads(cq_path.read_text(encoding="utf-8"))
                cq = parse_comp_quality_json_map(raw)
            except Exception:  # noqa: BLE001
                cq = {}
    if not cq and comp_quality_map:
        for cid, q in comp_quality_map.items():
            nk = str(normalize_gaia_source_id(cid) or "").strip()
            if not nk:
                continue
            q2 = str(q or "").strip().lower()
            if q2 in ("good", "suspect", "excluded"):
                cq[nk] = {"quality": q2}
    if not cq:
        return set()
    return ensemble_member_ids(
        cq,
        _comp_rms_map_from_df(comp_df),
        n_comp_min=3,
        n_comp_max=int(getattr(_cfg, "phase01_comparison_n_comp_max", 12) or 12),
    )


def _resolve_check_select_rms_floor(
    df: pd.DataFrame,
    cfg: AppConfig | None,
    floor_override: float | None,
) -> float:
    if floor_override is not None and math.isfinite(float(floor_override)):
        base = float(floor_override)
    elif cfg is not None:
        try:
            base = float(getattr(cfg, "check_select_rms_floor", 1e-4) or 1e-4)
        except (TypeError, ValueError):
            base = 1e-4
    else:
        base = 1e-4
    base = max(0.0, base)
    metrics: list[float] = []
    for col in ("p2p_rms", "comp_rms"):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        metrics.extend(float(v) for v in vals if math.isfinite(float(v)) and float(v) > 0.0)
    if metrics:
        med = float(np.median(metrics))
        if math.isfinite(med) and med > 0.0:
            return max(base, 0.1 * med)
    return base


def _drop_rms_artefacts(
    df: pd.DataFrame,
    *,
    cfg: AppConfig | None,
    floor_override: float | None,
) -> pd.DataFrame:
    if df.empty:
        return df
    floor = _resolve_check_select_rms_floor(df, cfg, floor_override)
    work = df.copy()
    if "p2p_rms" in work.columns:
        work["p2p_rms"] = pd.to_numeric(work["p2p_rms"], errors="coerce")
    if "comp_rms" in work.columns:
        work["comp_rms"] = pd.to_numeric(work["comp_rms"], errors="coerce")

    def _metric_ok(row: pd.Series) -> bool:
        vals: list[float] = []
        if "p2p_rms" in row.index:
            v = float(pd.to_numeric(row.get("p2p_rms"), errors="coerce"))
            if math.isfinite(v):
                vals.append(v)
        if "comp_rms" in row.index:
            v = float(pd.to_numeric(row.get("comp_rms"), errors="coerce"))
            if math.isfinite(v):
                vals.append(v)
        if not vals:
            return True
        m = min(vals)
        return m > floor

    mask = work.apply(_metric_ok, axis=1)
    return work.loc[mask].copy()


def _apply_crowding_exclusion(df: pd.DataFrame, cfg: AppConfig | None) -> pd.DataFrame:
    """CS-4: drop high-contamination candidates when ``contamination_idx`` is present."""
    if df.empty or "contamination_idx" not in df.columns:
        return df
    _cfg = cfg or AppConfig()
    try:
        thr = float(getattr(_cfg, "aperture_correction_max_contamination", 0.15) or 0.15)
    except (TypeError, ValueError):
        thr = 0.15
    work = df.copy()
    work["contamination_idx"] = pd.to_numeric(work["contamination_idx"], errors="coerce")
    return work.loc[
        work["contamination_idx"].isna() | (work["contamination_idx"] <= thr)
    ].copy()


def _exclude_ensemble_members(df: pd.DataFrame, ensemble_ids: set[str]) -> pd.DataFrame:
    ens = _norm_ensemble_id_set(ensemble_ids)
    if ens and "catalog_id" in df.columns:
        cids = df["catalog_id"].map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
        df = df.loc[~cids.isin(ens)].copy()
    for col in ("is_ensemble", "in_ensemble", "used_in_ensemble", "ensemble", "is_used"):
        if col in df.columns:
            try:
                m = df[col].fillna(False).astype(bool)
                df = df.loc[~m].copy()
            except Exception as exc:  # noqa: BLE001
                from except_fix_counters import get_except_fix_counters

                get_except_fix_counters().check_star_ensemble_filter_skip += 1
                logging.error(
                    "[CHECK-KMAG] ensemble-flag column filter skipped for column %r: %s",
                    col,
                    exc,
                )
            break
    return df


def field_check_star_candidate_pool(
    comp_field_df: pd.DataFrame,
    *,
    target_comps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Union field comp metrics (one row per catalog_id) for check-star selection."""
    frames: list[pd.DataFrame] = []
    if comp_field_df is not None and not getattr(comp_field_df, "empty", True):
        frames.append(comp_field_df)
    if target_comps is not None and not getattr(target_comps, "empty", True):
        frames.append(target_comps)
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    pool = normalize_comp_df_export_columns(pool)
    if pool.empty or "catalog_id" not in pool.columns:
        return pool
    pool = pool.copy()
    pool["catalog_id"] = pool["catalog_id"].map(
        lambda x: str(normalize_gaia_source_id(x) or "").strip()
    )
    pool = pool.loc[pool["catalog_id"].astype(str).str.len().gt(0)].copy()
    if "comp_rms" in pool.columns:
        pool["comp_rms"] = pd.to_numeric(pool["comp_rms"], errors="coerce")
        pool = pool.sort_values(
            ["comp_rms", "catalog_id"], ascending=[True, True], kind="mergesort"
        )
    pool = pool.drop_duplicates(subset=["catalog_id"], keep="first")
    return pool.reset_index(drop=True)


def select_check_star(
    comp_df: pd.DataFrame,
    *,
    ensemble_ids: set[str] | None = None,
    n_comp_min: int = 3,
    cfg: AppConfig | None = None,
    check_select_rms_floor: float | None = None,
) -> pd.Series | None:
    """Pick independent check star - best stability among non-ensemble good comps."""
    if comp_df is None or comp_df.empty:
        return None

    df = normalize_comp_df_export_columns(comp_df)
    try:
        if "is_check_star" in df.columns:
            mchk = df["is_check_star"].fillna(False).astype(bool)
            if bool(mchk.any()):
                row = df.loc[mchk].iloc[0]
                cid = str(normalize_gaia_source_id(row.get("catalog_id", "")) or "").strip()
                ens = _norm_ensemble_id_set(ensemble_ids)
                if ens and cid in ens:
                    return None
                return row
    except Exception as exc:  # noqa: BLE001
        logging.debug('[EXC-0023] intent unclear (return None / return row / except Exception:  # noqa: BLE001 / pass): %s', exc)
        pass

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip().str.lower() == "good"]

    df = _exclude_ensemble_members(df, ensemble_ids or set())
    df = _drop_rms_artefacts(df, cfg=cfg, floor_override=check_select_rms_floor)
    df = _apply_crowding_exclusion(df, cfg)

    if len(df) < int(n_comp_min):
        return None

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

    if "comp_rms_fieldwide" in df.columns:
        # Field-wide RMS is for sparse-path diagnostics only - never rank check stars on it.
        pass

    if "comp_score" in df.columns:
        df["comp_score"] = pd.to_numeric(df["comp_score"], errors="coerce")
        cand = df[df["comp_score"].notna()].sort_values("comp_score", ascending=False)
        if not cand.empty:
            return cand.iloc[0]

    try:
        return df.iloc[0]
    except IndexError as exc:  # noqa: BLE001
        return None


def _target_mag_from_row(comp_df: pd.DataFrame, target_cid: str) -> float:
    if comp_df is None or comp_df.empty:
        return float("nan")
    if "target_catalog_id" in comp_df.columns:
        sub = comp_df[comp_df["target_catalog_id"].astype(str).str.strip() == str(target_cid).strip()]
        if not sub.empty and "target_mag" in sub.columns:
            v = float(pd.to_numeric(sub.iloc[0].get("target_mag"), errors="coerce"))
            if math.isfinite(v):
                return v
    for col in ("mag", "catalog_mag", "target_mag"):
        if col in comp_df.columns:
            v = float(pd.to_numeric(comp_df[col].iloc[0], errors="coerce"))
            if math.isfinite(v):
                return v
    return float("nan")


def _ensemble_median_bprp(comp_df: pd.DataFrame, ensemble_ids: set[str]) -> float:
    if comp_df is None or comp_df.empty or "bp_rp" not in comp_df.columns:
        return float("nan")
    ens = _norm_ensemble_id_set(ensemble_ids)
    cids = comp_df["catalog_id"].map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
    sub = comp_df[cids.isin(ens)]
    br = pd.to_numeric(sub.get("bp_rp"), errors="coerce")
    if br.notna().any():
        return float(br.median())
    return float("nan")


def _p2p_good_mask(df: pd.DataFrame, cfg: AppConfig | None) -> pd.Series:
    _cfg = cfg or AppConfig()
    thr = float(getattr(_cfg, "phase01_comparison_max_comp_rms", 0.1) or 0.1)
    work = df.copy()
    p2p = pd.to_numeric(work.get("p2p_rms", work.get("comp_rms")), errors="coerce")
    return p2p.notna() & (p2p <= thr)


def select_external_check_star(
    pool_df: pd.DataFrame,
    *,
    ensemble_ids: set[str],
    target_mag: float,
    target_bprp: float | None = None,
    ensemble_bprp_median: float | None = None,
    cfg: AppConfig | None = None,
) -> ExternalCheckSelection | None:
    """Pick external K for sparse branch (Amendment 1 section 2.1)."""
    if pool_df is None or pool_df.empty:
        return None
    df = normalize_comp_df_export_columns(pool_df)
    df = _exclude_ensemble_members(df, ensemble_ids)
    df = _drop_rms_artefacts(df, cfg=cfg, floor_override=None)
    df = _apply_crowding_exclusion(df, cfg)
    if df.empty:
        return None
    good_p2p = _p2p_good_mask(df, cfg)
    df = df.loc[good_p2p].copy()
    if df.empty:
        return None
    _cfg = cfg or AppConfig()
    colour_win = float(getattr(_cfg, "comp_max_delta_bprp", 0.79) or 0.79)
    if "mag" not in df.columns and "catalog_mag" in df.columns:
        df["mag"] = pd.to_numeric(df["catalog_mag"], errors="coerce")
    if "mag" in df.columns:
        df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    else:
        df["mag"] = float("nan")
    if "p2p_rms" in df.columns:
        df["p2p_rms"] = pd.to_numeric(df["p2p_rms"], errors="coerce")
    elif "comp_rms" in df.columns:
        df["p2p_rms"] = pd.to_numeric(df["comp_rms"], errors="coerce")
    tm = float(target_mag) if math.isfinite(float(target_mag)) else float("nan")
    if math.isfinite(tm) and "mag" in df.columns:
        df["_dmag"] = (df["mag"] - tm).abs()
        df = df.sort_values(["_dmag", "p2p_rms", "catalog_id"], ascending=[True, True, True], kind="mergesort")
    else:
        df = df.sort_values(["p2p_rms", "catalog_id"], ascending=[True, True], kind="mergesort")
    row = df.iloc[0]
    cid = str(normalize_gaia_source_id(row.get("catalog_id", "")) or "").strip()
    k_bprp = float(pd.to_numeric(row.get("bp_rp"), errors="coerce"))
    med = float(ensemble_bprp_median) if ensemble_bprp_median is not None else float("nan")
    k_colour_offset = abs(k_bprp - med) if math.isfinite(k_bprp) and math.isfinite(med) else float("nan")
    tier_excluded = False
    k_source = "comp_pool_external"
    t_bp = float(target_bprp) if target_bprp is not None and math.isfinite(float(target_bprp)) else float("nan")
    if math.isfinite(t_bp) and math.isfinite(k_bprp) and abs(k_bprp - t_bp) > colour_win:
        tier_excluded = True
        k_source = "tier_excluded"
    return ExternalCheckSelection(
        row=row,
        k_source=k_source,
        k_tier_excluded=tier_excluded,
        k_colour_offset=k_colour_offset,
    )


def evaluate_k_colour_caveat(
    k_colour_offset: float,
    *,
    colour_window: float,
    airmass_range: float,
) -> bool:
    return (
        math.isfinite(float(k_colour_offset))
        and math.isfinite(float(colour_window))
        and math.isfinite(float(airmass_range))
        and float(k_colour_offset) > float(colour_window)
        and float(airmass_range) > 0.2
    )


def comp_ensemble_maps(
    comp_df: pd.DataFrame,
    cfg: AppConfig,
) -> tuple[dict[str, float], dict[str, int], dict[str, float], dict[int, float]]:
    _cfg_tw = cfg.comp_tier_weights()
    tier_weights = {
        1: float(_cfg_tw[0]),
        2: float(_cfg_tw[1]),
        3: float(_cfg_tw[2]),
        4: float(_cfg_tw[3]),
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


@dataclass(frozen=True, slots=True)
class ExternalCheckSelection:
  row: pd.Series
  k_source: str
  k_tier_excluded: bool
  k_colour_offset: float


@dataclass(frozen=True, slots=True)
class CheckEnsembleResult:
    """Check-star ensemble output + sparse-trust sidecar metadata."""

    kmag: np.ndarray
    check_sparse: bool = False
    n_comps: int = 0
    trust_R: float = float("nan")
    trust_R_lo: float = float("nan")
    trust_R_hi: float = float("nan")
    comp_stability_p: float = float("nan")
    x2_pair_mag2: float = float("nan")
    triangulation_clipped: bool = False
    zp_sem_ratio: float | None = None
    single_comp: bool = False
    sparse_flags: tuple[str, ...] = ()
    k_source: str = ""
    k_colour_offset: float = float("nan")
    k_tier_excluded: bool = False
    k_colour_caveat: bool = False
    trust_R_detrend: float = float("nan")
    trust_R_detrend_lo: float = float("nan")
    trust_R_detrend_hi: float = float("nan")


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
    n_comp_min: int | None = None,
    n_comp_max: int | None = None,
    comp_photon_mag: dict[str, np.ndarray] | None = None,
    sigma_sys_mag: float = 0.0,
    airmass: np.ndarray | None = None,
    k_source: str = "",
    k_colour_offset: float = float("nan"),
    k_tier_excluded: bool = False,
    sparse_external_k: bool = False,
) -> CheckEnsembleResult | None:
    """Ensemble-standardize the check star; K external on sparse branch (Amendment 1)."""
    cid = str(normalize_gaia_source_id(check_cid) or "").strip()
    if not cid or cid not in comp_lc:
        return None
    other_ids = [c for c in comp_ids if c != cid and c in comp_lc]
    other_quality: dict[str, dict] = {}
    for c in other_ids:
        if c not in comp_quality:
            continue
        info = comp_quality[c]
        if isinstance(info, str):
            q = str(info).strip().lower()
            if q == "excluded":
                continue
            other_quality[c] = {"quality": q or "good"}
        elif isinstance(info, dict):
            q = str(info.get("quality", "") or "").strip().lower()
            if q == "excluded":
                continue
            other_quality[c] = info
        else:
            continue
    n_ensemble = len(other_quality)
    _min_other = max(2, int(n_comp_min if n_comp_min is not None else 2))
    _max_other = int(n_comp_max if n_comp_max is not None else cfg.phase01_comparison_n_comp_max)

    k_meta = {
        "k_source": str(k_source or ""),
        "k_colour_offset": float(k_colour_offset),
        "k_tier_excluded": bool(k_tier_excluded),
        "k_colour_caveat": False,
    }
    colour_win = float(getattr(cfg, "comp_max_delta_bprp", 0.79) or 0.79)
    if airmass is not None:
        am = np.asarray(airmass, dtype=np.float64)
        am_ok = am[np.isfinite(am)]
        am_range = float(am_ok.max() - am_ok.min()) if am_ok.size >= 2 else float("nan")
        k_meta["k_colour_caveat"] = evaluate_k_colour_caveat(
            float(k_colour_offset), colour_window=colour_win, airmass_range=am_range,
        )

    if n_ensemble < 1:
        return None

    if n_ensemble == 1 and (sparse_external_k or n_ensemble < _min_other):
        c1 = next(iter(other_quality))
        m_k = np.asarray(comp_lc[cid], dtype=np.float64)
        m_c1 = np.asarray(comp_lc[c1], dtype=np.float64)
        mag_calib = m_k - m_c1
        if not np.isfinite(mag_calib).any():
            return None
        sparse_stats = None
        sparse_flags: tuple[str, ...] = ("single_comp",)
        if comp_photon_mag is not None:
            from sparse_trust_core import (  # noqa: PLC0415
                compute_sparse_trust_stats,
                sparse_trust_config_from_app,
                trust_band,
            )

            phot = dict(comp_photon_mag)
            phot["__check__"] = phot.get(cid, np.full(m_k.size, float("nan")))
            sparse_stats = compute_sparse_trust_stats(
                kmag=mag_calib,
                m_K=m_k,
                comp_mags={c1: m_c1},
                comp_photon_mag=phot,
                sigma_sys_mag=float(sigma_sys_mag),
                n_comps=1,
                cfg=sparse_trust_config_from_app(cfg),
                airmass=airmass,
            )
            _, sparse_flags = trust_band(
                R_hi=sparse_stats.trust_R_hi,
                R_lo=sparse_stats.trust_R_lo,
                stability_p=sparse_stats.comp_stability_p,
                x2_pair_mag2=sparse_stats.x2_pair_mag2,
                n_comps=1,
                triangulation_clipped=False,
                cfg=sparse_trust_config_from_app(cfg),
            )
        if sparse_stats is not None:
            return CheckEnsembleResult(
                kmag=mag_calib,
                check_sparse=True,
                n_comps=1,
                trust_R=sparse_stats.trust_R,
                trust_R_lo=sparse_stats.trust_R_lo,
                trust_R_hi=sparse_stats.trust_R_hi,
                comp_stability_p=sparse_stats.comp_stability_p,
                x2_pair_mag2=sparse_stats.x2_pair_mag2,
                triangulation_clipped=False,
                single_comp=True,
                sparse_flags=sparse_flags,
                trust_R_detrend=sparse_stats.trust_R_detrend,
                trust_R_detrend_lo=sparse_stats.trust_R_detrend_lo,
                trust_R_detrend_hi=sparse_stats.trust_R_detrend_hi,
                **k_meta,
            )
        return CheckEnsembleResult(
            kmag=mag_calib,
            check_sparse=True,
            n_comps=1,
            single_comp=True,
            sparse_flags=sparse_flags,
            **k_meta,
        )

    if n_ensemble < _min_other and not sparse_external_k:
        return None

    other_lc = {c: comp_lc[c] for c in other_quality}
    other_cat = {c: comp_catalog_mag[c] for c in other_quality if c in comp_catalog_mag}
    if len(other_quality) < 2:
        return None
    mag_calib, _, ensemble_scatter = ensemble_normalize(
        comp_lc[cid],
        other_lc,
        other_cat,
        other_quality,
        comp_rms_map=comp_rms_map,
        comp_tier_map=comp_tier_map,
        tier_weights=tier_weights,
        n_comp_min=max(2, min(_min_other, len(other_quality))),
        n_comp_max=_max_other,
    )
    if not np.isfinite(mag_calib).any():
        return None

    n_other = len(other_quality)
    check_sparse = n_other <= 2
    sparse_stats = None
    sparse_flags: tuple[str, ...] = ()
    zp_ratio: float | None = None
    if comp_photon_mag is not None and n_other >= 2:
        from sparse_trust_core import (  # noqa: PLC0415
            compute_sparse_trust_stats,
            sparse_trust_config_from_app,
            trust_band,
        )

        phot = dict(comp_photon_mag)
        phot["__check__"] = phot.get(cid, np.full(len(comp_lc[cid]), float("nan")))
        use_ids = list(other_quality.keys())[:2]
        comp_mags = {c: comp_lc[c] for c in use_ids}
        sparse_stats = compute_sparse_trust_stats(
            kmag=mag_calib,
            m_K=comp_lc[cid],
            comp_mags=comp_mags,
            comp_photon_mag=phot,
            sigma_sys_mag=float(sigma_sys_mag),
            n_comps=n_other,
            cfg=sparse_trust_config_from_app(cfg),
            airmass=airmass,
        )
        _, sparse_flags = trust_band(
            R_hi=sparse_stats.trust_R_hi,
            R_lo=sparse_stats.trust_R_lo,
            stability_p=sparse_stats.comp_stability_p,
            x2_pair_mag2=sparse_stats.x2_pair_mag2,
            n_comps=n_other,
            triangulation_clipped=sparse_stats.triangulation_clipped,
            cfg=sparse_trust_config_from_app(cfg),
        )
        if n_other in (3, 4):
            sem_med = float(np.nanmedian(ensemble_scatter[np.isfinite(ensemble_scatter)]))
            if math.isfinite(sem_med) and sem_med > 0 and comp_photon_mag:
                from sparse_trust_core import sigma_zp_per_epoch, triangulate_variances  # noqa: PLC0415

                c1, c2 = use_ids[0], use_ids[1] if len(use_ids) > 1 else use_ids[0]
                tri = triangulate_variances(
                    float(np.nanvar(comp_lc[cid] - comp_lc[c1], ddof=1)),
                    float(np.nanvar(comp_lc[cid] - comp_lc[c2], ddof=1)),
                    float(np.nanvar(comp_lc[c1] - comp_lc[c2], ddof=1)),
                )
                x2_c1 = max(tri.sig2_C1, 0.0)
                x2_c2 = max(tri.sig2_C2, 0.0)
                flux = np.vstack(
                    [
                        10.0 ** (-0.4 * np.asarray(comp_lc[c1], dtype=float)),
                        10.0 ** (-0.4 * np.asarray(comp_lc[c2], dtype=float)),
                    ]
                )
                phot_stack = np.vstack(
                    [
                        comp_photon_mag.get(c1, np.full(len(comp_lc[c1]), float("nan"))),
                        comp_photon_mag.get(c2, np.full(len(comp_lc[c2]), float("nan"))),
                    ]
                )
                tri_zp = sigma_zp_per_epoch(flux, phot_stack, np.array([x2_c1, x2_c2]))
                tri_med = float(np.nanmedian(tri_zp[np.isfinite(tri_zp)]))
                if math.isfinite(tri_med) and tri_med > 0:
                    zp_ratio = sem_med / tri_med

    if sparse_stats is not None:
        return CheckEnsembleResult(
            kmag=mag_calib,
            check_sparse=check_sparse,
            n_comps=n_other,
            trust_R=sparse_stats.trust_R,
            trust_R_lo=sparse_stats.trust_R_lo,
            trust_R_hi=sparse_stats.trust_R_hi,
            comp_stability_p=sparse_stats.comp_stability_p,
            x2_pair_mag2=sparse_stats.x2_pair_mag2,
            triangulation_clipped=sparse_stats.triangulation_clipped,
            zp_sem_ratio=zp_ratio,
            sparse_flags=sparse_flags,
            single_comp=n_other == 1,
            trust_R_detrend=sparse_stats.trust_R_detrend,
            trust_R_detrend_lo=sparse_stats.trust_R_detrend_lo,
            trust_R_detrend_hi=sparse_stats.trust_R_detrend_hi,
            **k_meta,
        )
    return CheckEnsembleResult(
        kmag=mag_calib,
        check_sparse=check_sparse,
        n_comps=n_other,
        zp_sem_ratio=zp_ratio,
        single_comp=n_other == 1,
        **k_meta,
    )


def build_comp_photon_mag_from_frames(
    all_frames: pd.DataFrame,
    star_ids: list[str],
    source_files: list[str],
    *,
    cfg: AppConfig | None = None,
) -> dict[str, np.ndarray]:
    """Per-epoch photon sigma (mag) aligned to ``source_files`` order."""
    from photometry_core import (  # noqa: PLC0415
        ERR_BKG_MODE_EMPIRICAL,
        _photometric_error_with_bkg_mode,
        _sky_pp_for_photometric_error,
    )
    from sparse_trust_core import rel_flux_err_to_photon_mag  # noqa: PLC0415

    n = len(source_files)
    out: dict[str, np.ndarray] = {}
    if all_frames.empty or n <= 0:
        return out
    work = all_frames.copy()
    if "catalog_id" in work.columns:
        work["_nid"] = work["catalog_id"].map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
    else:
        work["_nid"] = work.get("name", pd.Series(dtype=str)).astype(str)
    sf_col = "source_file" if "source_file" in work.columns else None
    _cfg = cfg or AppConfig()
    bkg_mode = str(getattr(_cfg, "err_background_mode", ERR_BKG_MODE_EMPIRICAL) or ERR_BKG_MODE_EMPIRICAL)

    def _row_rel_err(row: pd.Series) -> float:
        e = float(pd.to_numeric(row.get("err", float("nan")), errors="coerce"))
        if math.isfinite(e) and e > 0:
            return e
        flux = float(pd.to_numeric(row.get("dao_flux", row.get("flux", float("nan"))), errors="coerce"))
        sig_ap = float(pd.to_numeric(row.get("sigma_bkg_ap", float("nan")), errors="coerce"))
        sky = _sky_pp_for_photometric_error(row)
        r_px = float(pd.to_numeric(row.get("aperture_r_px", float("nan")), errors="coerce"))
        area = math.pi * r_px * r_px if math.isfinite(r_px) and r_px > 0 else math.pi * 9.0
        err_rel, _ = _photometric_error_with_bkg_mode(
            flux,
            err_background_mode=bkg_mode,
            sky_pp=sky,
            area=area,
            sigma_bkg_ap=sig_ap,
        )
        return float(err_rel)

    for cid in star_ids:
        arr = np.full(n, float("nan"), dtype=float)
        sub = work[work["_nid"] == str(cid).strip()]
        if sub.empty or sf_col is None:
            out[cid] = arr
            continue
        by_sf = {str(r[sf_col]).strip(): _row_rel_err(r) for _, r in sub.iterrows()}
        err_rel = np.asarray([by_sf.get(str(sf).strip(), float("nan")) for sf in source_files], dtype=float)
        arr = rel_flux_err_to_photon_mag(err_rel)
        out[cid] = arr
    return out


def check_kmag_sidecar_path(lc_dir: Path | str, target_cid: str) -> Path:
    return Path(lc_dir) / f"check_kmag_{str(target_cid).strip()}.csv"


def save_check_kmag_sidecar(
    path: Path,
    *,
    check_cid: str,
    bjd: np.ndarray,
    source_files: list[str],
    kmag: np.ndarray,
    ensemble: CheckEnsembleResult | None = None,
) -> None:
    n = min(len(bjd), len(source_files), len(kmag))
    if n <= 0:
        return
    row: dict[str, object] = {
        "check_catalog_id": [str(check_cid)] * n,
        "bjd": np.asarray(bjd[:n], dtype=float),
        "source_file": [str(s) for s in source_files[:n]],
        "kmag": np.round(np.asarray(kmag[:n], dtype=float), 6),
    }
    if ensemble is not None:
        row["check_sparse"] = [int(bool(ensemble.check_sparse))] * n
        row["trust_R"] = [round(float(ensemble.trust_R), 6) if math.isfinite(float(ensemble.trust_R)) else ""] * n
        row["trust_R_lo"] = [
            round(float(ensemble.trust_R_lo), 6) if math.isfinite(float(ensemble.trust_R_lo)) else ""
        ] * n
        row["trust_R_hi"] = [
            round(float(ensemble.trust_R_hi), 6) if math.isfinite(float(ensemble.trust_R_hi)) else ""
        ] * n
        row["comp_stability_p"] = [
            round(float(ensemble.comp_stability_p), 6) if math.isfinite(float(ensemble.comp_stability_p)) else ""
        ] * n
        row["x2_pair_mag2"] = [
            round(float(ensemble.x2_pair_mag2), 8) if math.isfinite(float(ensemble.x2_pair_mag2)) else ""
        ] * n
        row["triangulation_clipped"] = [int(bool(ensemble.triangulation_clipped))] * n
        if ensemble.zp_sem_ratio is not None and math.isfinite(float(ensemble.zp_sem_ratio)):
            row["zp_sem_ratio"] = [round(float(ensemble.zp_sem_ratio), 6)] * n
        if ensemble.sparse_flags:
            row["sparse_flags"] = [";".join(ensemble.sparse_flags)] * n
        if ensemble.k_source:
            row["k_source"] = [str(ensemble.k_source)] * n
        if math.isfinite(float(ensemble.k_colour_offset)):
            row["k_colour_offset"] = [round(float(ensemble.k_colour_offset), 6)] * n
        row["k_tier_excluded"] = [int(bool(ensemble.k_tier_excluded))] * n
        row["k_colour_caveat"] = [int(bool(ensemble.k_colour_caveat))] * n
        if math.isfinite(float(ensemble.trust_R_detrend)):
            row["trust_R_detrend"] = [
                round(float(ensemble.trust_R_detrend), 6) if math.isfinite(float(ensemble.trust_R_detrend)) else ""
            ] * n
            row["trust_R_detrend_lo"] = [
                round(float(ensemble.trust_R_detrend_lo), 6)
                if math.isfinite(float(ensemble.trust_R_detrend_lo))
                else ""
            ] * n
            row["trust_R_detrend_hi"] = [
                round(float(ensemble.trust_R_detrend_hi), 6)
                if math.isfinite(float(ensemble.trust_R_detrend_hi))
                else ""
            ] * n
    df = pd.DataFrame(row)
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
    except Exception as exc:  # noqa: BLE001
        logging.debug('[EXC-0025] intent unclear (try: / df = pd.read_csv(sidecar_path, low_memory=False) / except Except...: %s', exc)
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
            except Exception as exc:  # noqa: BLE001
                logging.warning('[EXC-0026] intent unclear (try: / csv_df = pd.read_csv(path, low_memory=False, dtype=_GAIA_ID_DTYP...: %s', exc)
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
    if len(other_ids) < 2 or proc_dir is None or not proc_dir.is_dir():
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
        n_comp_min=2,
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

    kmag_result = compute_check_ensemble_mag_calib(
        check_cid,
        comp_ids,
        comp_lc,
        comp_catalog_mag,
        comp_quality,
        comp_rms_map=comp_rms_map,
        comp_tier_map=comp_tier_map,
        tier_weights=tier_weights,
        cfg=cfg,
        n_comp_min=2,
    )
    if kmag_result is None:
        return (["na"] * n_rows, "na")
    kmag_arr = kmag_result.kmag

    per_row = [_fmt_kmag(float(kmag_arr[i])) if i < len(kmag_arr) else "na" for i in range(n_rows)]
    if not any(v != "na" for v in per_row):
        return (["na"] * n_rows, "na")
    finite = [float(v) for v in per_row if v != "na"]
    if len(finite) >= 2 and (max(finite) - min(finite)) <= 0.0005:
        return ([_fmt_kmag(float(np.median(finite)))] * n_rows, "median-ensemble-constant")
    return (per_row, "per-row-ensemble")
