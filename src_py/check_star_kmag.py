"""Check-star measured KMAG for AAVSO ensemble exports (additive reporting)."""

from __future__ import annotations

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


#: Base RMS floor (mag) below which a check-star candidate's RMS is treated as an artefact.
CHECK_SELECT_RMS_FLOOR = 1e-4


def _resolve_check_select_rms_floor(df: pd.DataFrame) -> float:
    base = CHECK_SELECT_RMS_FLOOR
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


def _drop_rms_artefacts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    floor = _resolve_check_select_rms_floor(df)
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


def drop_blended_check_candidates(pool: pd.DataFrame, f_star: float) -> pd.DataFrame:
    """BLEND-K-2026-10: drop check candidates with |neighbour light| > 1 % at the target's
    aperture k (column blend_kNNN from the comparison-star selection pool). NaN (unknown) does not block,
    as for comparison stars; without the column nothing is dropped (older drafts)."""
    from aperture_pertarget import blend_k_column  # noqa: PLC0415
    from comp_blend import BLEND_FLUX_FRAC_MAX  # noqa: PLC0415

    col = blend_k_column(float(f_star))
    if pool is None or pool.empty or col not in pool.columns:
        return pool
    bad = pd.to_numeric(pool[col], errors="coerce").abs() > BLEND_FLUX_FRAC_MAX
    return pool.loc[~bad].reset_index(drop=True)


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
    target_bp_rp: float | None = None,
    target_mag: float | None = None,
) -> pd.Series | None:
    """Pick independent check star - best stability among non-ensemble good comps.

    CHECK-RULES-2026-10: with ``target_bp_rp`` the check star follows the comparison-star
    rules relative to THIS target: colour ladder |BP-RP - target| (comp_color_tiers, then
    comp_max_delta_bprp), inside the first non-empty step the lowest LOO ``comp_rms``,
    ties by |mag - target_mag|.
    """
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
    df = _drop_rms_artefacts(df)

    # IMPL-04: do not promote a tier-1 star whose RMS exceeds the existing
    # phase01_comparison_max_comp_rms ceiling (draft 514: lone T1 at 0.125 mag
    # beat quieter T2/T3 and produced 222 mmag check LCs).
    _cfg = cfg or AppConfig()
    try:
        _rms_ceil = float(getattr(_cfg, "phase01_comparison_max_comp_rms", 0.1) or 0.1)
    except (TypeError, ValueError):
        _rms_ceil = 0.1
    if math.isfinite(_rms_ceil) and _rms_ceil > 0:
        work = df.copy()
        for _col in ("p2p_rms", "comp_rms"):
            if _col in work.columns:
                work[_col] = pd.to_numeric(work[_col], errors="coerce")
        if "p2p_rms" in work.columns or "comp_rms" in work.columns:
            def _under_ceil(row: pd.Series) -> bool:
                vals = []
                for _col in ("p2p_rms", "comp_rms"):
                    if _col in row.index:
                        v = float(pd.to_numeric(row.get(_col), errors="coerce"))
                        if math.isfinite(v):
                            vals.append(v)
                if not vals:
                    return True
                return min(vals) <= _rms_ceil

            work = work.loc[work.apply(_under_ceil, axis=1)].copy()
            if not work.empty:
                df = work

    if len(df) < int(n_comp_min):
        return None

    try:
        _tb = float(target_bp_rp) if target_bp_rp is not None else float("nan")
    except (TypeError, ValueError):
        _tb = float("nan")
    if math.isfinite(_tb) and "bp_rp" in df.columns:
        from photometry_comp import _bprp_tier_ladder_for_selection  # noqa: PLC0415

        w = df.copy()
        w["_dc"] = (pd.to_numeric(w["bp_rp"], errors="coerce") - _tb).abs()
        _rc = "comp_rms" if "comp_rms" in w.columns else ("p2p_rms" if "p2p_rms" in w.columns else None)
        w["_rms"] = pd.to_numeric(w[_rc], errors="coerce") if _rc else np.nan
        try:
            _tm = float(target_mag) if target_mag is not None else float("nan")
        except (TypeError, ValueError):
            _tm = float("nan")
        _mc = "catalog_mag" if "catalog_mag" in w.columns else ("mag" if "mag" in w.columns else None)
        if math.isfinite(_tm) and _mc:
            w["_dm"] = (pd.to_numeric(w[_mc], errors="coerce") - _tm).abs()
        else:
            w["_dm"] = 0.0
        for _lim in _bprp_tier_ladder_for_selection(_cfg, float(getattr(_cfg, "comp_max_delta_bprp", 0.79))):
            cand = w[w["_dc"].le(float(_lim)) & w["_rms"].notna()]
            if not cand.empty:
                cand = cand.sort_values(["_rms", "_dm"], ascending=[True, True], kind="mergesort")
                return df.loc[cand.index[0]]

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
    except IndexError:  # noqa: BLE001
        return None


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
    cfg: AppConfig,
    n_comp_min: int | None = None,
    comp_photon_mag: dict[str, np.ndarray] | None = None,
    sigma_sys_mag: float = 0.0,
    airmass: np.ndarray | None = None,
    k_source: str = "",
    k_colour_offset: float = float("nan"),
    k_tier_excluded: bool = False,
    sparse_external_k: bool = False,
    allow_relative: bool = False,
) -> CheckEnsembleResult | None:
    """Ensemble-standardize the check star; K external on sparse branch (Amendment 1).

    ``allow_relative`` (UNCAL-LC-2026-10): only for a target without a band zero point -
    the check is then verified on its differential magnitude (its level is not exported)."""
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
        # FIX-KMAG1-2026-10: K = m_check - m_comp + catalogue(comp), like the ensemble branch;
        # m_check - m_comp alone is a relative level and ended up as KMAG in the AAVSO file.
        # Relative only for a target without a band zero point (allow_relative).
        _cat_c1 = float(comp_catalog_mag.get(c1, float("nan")))
        if math.isfinite(_cat_c1):
            mag_calib = m_k - m_c1 + _cat_c1
        elif allow_relative:
            mag_calib = m_k - m_c1
        else:
            return None
        if not np.isfinite(mag_calib).any():
            return None
        sparse_stats = None
        sparse_flags: tuple[str, ...] = ("single_comp",)
        if comp_photon_mag is not None:
            from sparse_trust_core import (  # noqa: PLC0415
                compute_sparse_trust_stats,
                SparseTrustConfig,
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
                airmass=airmass,
            )
            _, sparse_flags = trust_band(
                R_hi=sparse_stats.trust_R_hi,
                R_lo=sparse_stats.trust_R_lo,
                stability_p=sparse_stats.comp_stability_p,
                x2_pair_mag2=sparse_stats.x2_pair_mag2,
                n_comps=1,
                triangulation_clipped=False,
                cfg=SparseTrustConfig(),
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
    mag_calib, _delta_k, ensemble_scatter = ensemble_normalize(
        comp_lc[cid],
        other_lc,
        other_cat,
        other_quality,
        n_comp_min=max(2, min(_min_other, len(other_quality))),
    )
    # UNCAL-LC-2026-10: without a band zero point (no comp with a valid band magnitude) the
    # check is verified on its differential magnitude - the stability tests use scatter only,
    # a constant level does not enter them. (The target is not exportable in that case.)
    if allow_relative and not np.isfinite(mag_calib).any() and np.isfinite(_delta_k).any():
        mag_calib = np.asarray(_delta_k, dtype=np.float64)
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
            SparseTrustConfig,
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
            airmass=airmass,
        )
        _, sparse_flags = trust_band(
            R_hi=sparse_stats.trust_R_hi,
            R_lo=sparse_stats.trust_R_lo,
            stability_p=sparse_stats.comp_stability_p,
            x2_pair_mag2=sparse_stats.x2_pair_mag2,
            n_comps=n_other,
            triangulation_clipped=sparse_stats.triangulation_clipped,
            cfg=SparseTrustConfig(),
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
) -> dict[str, np.ndarray]:
    """Per-epoch photon sigma (mag) aligned to ``source_files`` order."""
    from photometry_core import (  # noqa: PLC0415
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


#: OUTPUT-REVIEW-2026-10 step 6: the per-check values (one per target) live in
#: ``comp_quality_<target>.json`` under this key; ``check_kmag_<target>.csv`` keeps the epochs.
CHECK_STAR_JSON_KEY = "check_star"


def _json_num(v: Any, nd: int) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def check_star_block(check_cid: str, ensemble: CheckEnsembleResult | None) -> dict[str, Any]:
    """The check star of a target and its trust statistics (``comp_quality`` JSON block)."""
    out: dict[str, Any] = {"catalog_id": str(check_cid)}
    if ensemble is None:
        return out
    out.update(
        {
            "check_sparse": bool(ensemble.check_sparse),
            "trust_R": _json_num(ensemble.trust_R, 6),
            "trust_R_lo": _json_num(ensemble.trust_R_lo, 6),
            "trust_R_hi": _json_num(ensemble.trust_R_hi, 6),
            "comp_stability_p": _json_num(ensemble.comp_stability_p, 6),
            "x2_pair_mag2": _json_num(ensemble.x2_pair_mag2, 8),
            "triangulation_clipped": bool(ensemble.triangulation_clipped),
            "sparse_flags": list(ensemble.sparse_flags or []),
        }
    )
    if ensemble.k_source:
        out["k_source"] = str(ensemble.k_source)
    if math.isfinite(float(ensemble.k_colour_offset)):
        out["k_colour_offset"] = _json_num(ensemble.k_colour_offset, 6)
    if math.isfinite(float(ensemble.trust_R_detrend)):
        out["trust_R_detrend"] = _json_num(ensemble.trust_R_detrend, 6)
        out["trust_R_detrend_lo"] = _json_num(ensemble.trust_R_detrend_lo, 6)
        out["trust_R_detrend_hi"] = _json_num(ensemble.trust_R_detrend_hi, 6)
    return out


def load_check_star(lc_dir: Path | str, target_cid: str) -> dict[str, Any]:
    """The check-star block of a target (``{}`` when there is none).

    From ``comp_quality_<target>.json``; drafts before OUTPUT-REVIEW step 6 repeat the values on
    every row of ``check_kmag_<target>.csv`` (read from its first row)."""
    import json  # noqa: PLC0415

    tid = str(target_cid).strip()
    cq = Path(lc_dir) / f"comp_quality_{tid}.json"
    if cq.is_file():
        try:
            raw = json.loads(cq.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raw = {}
        blk = raw.get(CHECK_STAR_JSON_KEY) if isinstance(raw, dict) else None
        if isinstance(blk, dict):
            out = dict(blk)
            out["catalog_id"] = str(normalize_gaia_source_id(out.get("catalog_id", "")) or "").strip()
            return out
    p = check_kmag_sidecar_path(lc_dir, tid)
    if not p.is_file():
        return {}
    try:
        sdf = pd.read_csv(p, low_memory=False, nrows=1, dtype={"check_catalog_id": str})
    except (OSError, ValueError):
        return {}
    if sdf.empty:
        return {}
    row = sdf.iloc[0]
    out = {"catalog_id": str(normalize_gaia_source_id(row.get("check_catalog_id", "")) or "").strip()}
    if "check_sparse" in sdf.columns:
        def _f(col: str) -> float | None:
            return _json_num(pd.to_numeric(row.get(col, float("nan")), errors="coerce"), 12)

        out.update(
            {
                "check_sparse": str(row.get("check_sparse", 0)).strip().lower() in ("1", "true", "yes"),
                "trust_R": _f("trust_R"),
                "trust_R_lo": _f("trust_R_lo"),
                "trust_R_hi": _f("trust_R_hi"),
                "comp_stability_p": _f("comp_stability_p"),
                "x2_pair_mag2": _f("x2_pair_mag2"),
                "triangulation_clipped": str(row.get("triangulation_clipped", 0)).strip() in ("1", "true"),
                "sparse_flags": [x for x in str(row.get("sparse_flags", "") or "").strip().split(";")
                                 if x and x.lower() != "nan"],
            }
        )
    return out


def save_check_kmag_sidecar(
    path: Path,
    *,
    bjd: np.ndarray,
    source_files: list[str],
    kmag: np.ndarray,
    kmag_err: np.ndarray | None = None,
) -> None:
    """``check_kmag_<target>.csv``: the check star per epoch (bjd, source_file, kmag, kmag_err)."""
    n = min(len(bjd), len(source_files), len(kmag))
    if n <= 0:
        return
    row: dict[str, object] = {
        "bjd": np.asarray(bjd[:n], dtype=float),
        "source_file": [str(s) for s in source_files[:n]],
        "kmag": np.round(np.asarray(kmag[:n], dtype=float), 6),
    }
    if kmag_err is not None and len(kmag_err) >= n:
        row["kmag_err"] = np.round(np.asarray(kmag_err[:n], dtype=float), 6)
    df = pd.DataFrame(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.debug("[CHECK-KMAG] wrote %s (%d rows)", path.name, n)


def drop_smart_rejected_check_candidates(
    pool: pd.DataFrame,
    smart_log: dict | None,
    target_cid: str = "",
    *,
    catalog_offset_excluded: Any = None,
) -> pd.DataFrame:
    """CHECK-NOT-SUSPECT-2026-10 (Milan 2026-10-12): a comparison star that COMP-SMART dropped as
    unstable for this target is not a check star either (draft 526 BO CVn: the dropped comp
    became the check star). ``smart_log`` is the per-target COMP-SMART log (``replaced`` ids).
    COMP-CATOFF-2026-10: a star excluded for its catalogue offset in any target's candidate set
    (``catalog_offset_excluded``) is no check star either (its catalogue magnitude is wrong for it)."""
    from photometry_shared import _normalize_gaia_id  # noqa: PLC0415

    drop = {str(_normalize_gaia_id(c)) for c in ((smart_log or {}).get("replaced") or [])}
    drop |= {str(_normalize_gaia_id(c)) for c in (catalog_offset_excluded or ())}
    if not drop or pool is None or pool.empty or "catalog_id" not in pool.columns:
        return pool

    keep = ~pool["catalog_id"].map(_normalize_gaia_id).astype(str).isin(drop)
    if int((~keep).sum()):
        logging.info("[CHECK-KMAG] %s: %d check candidates dropped by COMP-SMART as unstable excluded",
                     target_cid, int((~keep).sum()))
    return pool.loc[keep]
