"""Pinned comparison-star ensembles for legacy baseline targets (DAO-GAIA ERA-03).

When a target has a row set in ``pinned_ensembles.csv``, Phase-1 selection loads
the frozen membership from the anchor era (477dc8cf), re-validates each member
against immutable rules (zone/sat, distance, colour tier, RMS), drops violators
with a named reason, and fails loud if ``n_comp < n_comp_min``.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import AppConfig
from gaia_catalog_id import normalize_gaia_source_id

LOGGER = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PINNED_ENSEMBLES_PATH = _REPO_ROOT / "dev" / "validation" / "pinned_ensembles.csv"
DEFAULT_BASELINE_COMP_PT_PATH = (
    _REPO_ROOT
    / "Archive"
    / "Drafts"
    / "draft_000516_snapshot_cleanrebuild_20260818"
    / "platesolve"
    / "NoFilter_60_2"
    / "photometry"
    / "comparison_stars_per_target.csv"
)
DEFAULT_BASELINE_LC_DIR = DEFAULT_BASELINE_COMP_PT_PATH.parent / "lightcurves"
ANCHOR_PROVENANCE_SHA = "477dc8cf"
PIN_REASON = "dao_gaia_era_anchor_48"

_PIN_CACHE: dict[str, Any] | None = None
_PIN_CHECK_CACHE: dict[str, PinCheckStar] | None = None
_PIN_PATH: Path | None = None
_PIN_SHA256: str | None = None
_BASELINE_COMP_PT_CACHE: pd.DataFrame | None = None


class PinnedEnsembleError(RuntimeError):
    """Base error for pinned ensemble handling."""


class PinnedEnsembleInsufficientError(PinnedEnsembleError):
    """Pinned target fell below ``n_comp_min`` after rule re-validation."""


@dataclass(frozen=True)
class PinCheckStar:
    check_catalog_id: str
    check_kname: str
    provenance_sha: str
    pin_date: str


@dataclass(frozen=True)
class PinMember:
    comp_catalog_id: str
    comp_weight: float
    comp_tier: int
    weights_source_era: str
    pin_reason: str
    provenance_sha: str
    pin_date: str


def baseline_lc_ct_n_comp_for_target(target_catalog_id: str) -> int | None:
    """Anchor-era ``ct_n_comp`` scalar for pinned LC byte continuity."""
    tid = normalize_gaia_source_id(target_catalog_id)
    if not tid:
        return None
    path = DEFAULT_BASELINE_LC_DIR / f"lightcurve_{tid}.csv"
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, usecols=["ct_n_comp"], nrows=1)
        v = int(pd.to_numeric(df["ct_n_comp"].iloc[0], errors="coerce"))
        return v if v >= 0 else None
    except (OSError, KeyError, TypeError, ValueError):
        return None


def baseline_lc_ct_ok_for_target(target_catalog_id: str) -> bool | None:
    """Anchor-era ``ct_ok`` for pinned CT continuity (None when LC missing)."""
    tid = normalize_gaia_source_id(target_catalog_id)
    if not tid:
        return None
    path = DEFAULT_BASELINE_LC_DIR / f"lightcurve_{tid}.csv"
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, usecols=["ct_ok"], nrows=1)
        raw = df["ct_ok"].iloc[0]
        if isinstance(raw, str):
            return raw.strip().lower() in ("true", "1", "yes")
        return bool(raw)
    except (OSError, KeyError, TypeError, ValueError):
        return None


def load_baseline_comp_pt(path: Path | None = None) -> pd.DataFrame:
    """Anchor-era ``comparison_stars_per_target.csv`` for pinned row overlay."""
    global _BASELINE_COMP_PT_CACHE  # noqa: PLW0603

    resolved = Path(path) if path is not None else DEFAULT_BASELINE_COMP_PT_PATH
    if _BASELINE_COMP_PT_CACHE is not None:
        return _BASELINE_COMP_PT_CACHE
    if not resolved.is_file():
        return pd.DataFrame()
    df = pd.read_csv(
        resolved,
        dtype={"catalog_id": str, "target_catalog_id": str},
        low_memory=False,
    )
    _BASELINE_COMP_PT_CACHE = df
    return df


def overlay_anchor_comp_rows(
    result: pd.DataFrame,
    *,
    target_catalog_id: str,
    survivor_ids: list[str],
    pin_members: list[PinMember],
) -> pd.DataFrame:
    """When full pin membership survives, restore anchor comp_pt rows for byte-stable Phase 2A."""
    expected = {m.comp_catalog_id for m in pin_members}
    got = {normalize_gaia_source_id(x) for x in survivor_ids}
    got.discard(None)
    if got != expected or result.empty:
        return result
    base = load_baseline_comp_pt()
    if base.empty:
        return result
    tid = normalize_gaia_source_id(target_catalog_id)
    sub = base.loc[
        base["target_catalog_id"].astype(str).str.strip().eq(tid)
        & base["catalog_id"].astype(str).str.strip().isin(sorted(got))
    ].copy()
    if sub.empty or len(sub) != len(got):
        return result
    order = {cid: i for i, cid in enumerate(survivor_ids)}
    sub["_ord"] = sub["catalog_id"].astype(str).str.strip().map(order)
    sub = sub.sort_values("_ord").drop(columns=["_ord"], errors="ignore")
    sub["target_catalog_id"] = tid
    sub["comp_path"] = "pinned"
    note = str(sub["selection_note"].iloc[0]) if "selection_note" in sub.columns else ""
    if "pinned_ensemble" not in note:
        sub["selection_note"] = (
            f"{note}; pinned_ensemble anchor_overlay={ANCHOR_PROVENANCE_SHA}".strip("; ")
        )
    return sub.reset_index(drop=True)


def default_pinned_ensembles_path() -> Path:
    return DEFAULT_PINNED_ENSEMBLES_PATH


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_pin_row(row: pd.Series) -> PinMember | None:
    tid = normalize_gaia_source_id(row.get("comp_catalog_id"))
    if not tid:
        return None
    try:
        w = float(pd.to_numeric(row.get("comp_weight"), errors="coerce"))
    except (TypeError, ValueError):
        w = float("nan")
    try:
        tier = int(pd.to_numeric(row.get("comp_tier"), errors="coerce") or 4)
    except (TypeError, ValueError):
        tier = 4
    return PinMember(
        comp_catalog_id=tid,
        comp_weight=w if np.isfinite(w) else float("nan"),
        comp_tier=max(1, min(4, tier)),
        weights_source_era=str(row.get("weights_source_era") or ANCHOR_PROVENANCE_SHA).strip(),
        pin_reason=str(row.get("pin_reason") or PIN_REASON).strip(),
        provenance_sha=str(row.get("provenance_sha") or ANCHOR_PROVENANCE_SHA).strip(),
        pin_date=str(row.get("pin_date") or "").strip(),
    )


def load_pinned_ensembles(
    path: Path | None = None,
    *,
    force: bool = False,
) -> tuple[dict[str, list[PinMember]], str, Path]:
    """Return target_id -> ordered pin members, file sha256, resolved path."""
    global _PIN_CACHE, _PIN_CHECK_CACHE, _PIN_PATH, _PIN_SHA256  # noqa: PLW0603

    resolved = Path(path) if path is not None else default_pinned_ensembles_path()
    if (
        not force
        and _PIN_CACHE is not None
        and _PIN_CHECK_CACHE is not None
        and _PIN_PATH is not None
        and _PIN_SHA256 is not None
        and _PIN_PATH.resolve() == resolved.resolve()
    ):
        return _PIN_CACHE, _PIN_SHA256, resolved

    if not resolved.is_file():
        empty: dict[str, list[PinMember]] = {}
        _PIN_CACHE = empty
        _PIN_CHECK_CACHE = {}
        _PIN_PATH = resolved
        _PIN_SHA256 = ""
        return empty, "", resolved

    df = pd.read_csv(
        resolved,
        dtype={
            "target_catalog_id": str,
            "comp_catalog_id": str,
            "check_catalog_id": str,
            "check_kname": str,
        },
        low_memory=False,
    )
    out: dict[str, list[PinMember]] = {}
    checks: dict[str, PinCheckStar] = {}
    for _, row in df.iterrows():
        tgt = normalize_gaia_source_id(row.get("target_catalog_id"))
        if not tgt:
            continue
        chk_cid = normalize_gaia_source_id(row.get("check_catalog_id"))
        if chk_cid and tgt not in checks:
            checks[tgt] = PinCheckStar(
                check_catalog_id=chk_cid,
                check_kname=str(row.get("check_kname") or chk_cid).strip(),
                provenance_sha=str(row.get("provenance_sha") or ANCHOR_PROVENANCE_SHA).strip(),
                pin_date=str(row.get("pin_date") or "").strip(),
            )
        pm = _parse_pin_row(row)
        if pm is None:
            continue
        out.setdefault(tgt, []).append(pm)

    sha = compute_file_sha256(resolved)
    _PIN_CACHE = out
    _PIN_CHECK_CACHE = checks
    _PIN_PATH = resolved
    _PIN_SHA256 = sha
    return out, sha, resolved


def get_pinned_check_for_target(
    target_catalog_id: str,
    path: Path | None = None,
) -> PinCheckStar | None:
    tid = normalize_gaia_source_id(target_catalog_id)
    if not tid:
        return None
    load_pinned_ensembles(path, force=False)
    if _PIN_CHECK_CACHE is None:
        return None
    return _PIN_CHECK_CACHE.get(tid)


def get_pinned_members_for_target(
    target_catalog_id: str,
    path: Path | None = None,
) -> list[PinMember] | None:
    tid = normalize_gaia_source_id(target_catalog_id)
    if not tid:
        return None
    pins, _, _ = load_pinned_ensembles(path)
    members = pins.get(tid)
    return members if members else None


def is_pinned_target(target_catalog_id: str, path: Path | None = None) -> bool:
    return get_pinned_members_for_target(target_catalog_id, path) is not None


def get_pinned_provenance_for_target(
    target_catalog_id: str,
    path: Path | None = None,
) -> dict[str, str] | None:
    members = get_pinned_members_for_target(target_catalog_id, path)
    if not members:
        return None
    _, sha, resolved = load_pinned_ensembles(path)
    m0 = members[0]
    return {
        "selection_mode": "pinned",
        "pinned_ensembles_sha256": sha,
        "pinned_ensembles_path": str(resolved),
        "provenance_sha": m0.provenance_sha,
        "pin_reason": m0.pin_reason,
        "pin_date": m0.pin_date,
        "n_pinned_members": str(len(members)),
    }


def record_pinned_provenance_meta(
    output_dir: Path,
    path: Path | None = None,
) -> dict[str, Any]:
    """Merge pin-file provenance into photometry ``pipeline_meta``."""
    from photometry_core import merge_photometry_pipeline_meta  # noqa: PLC0415

    pins, sha, resolved = load_pinned_ensembles(path)
    block = {
        "pinned_ensembles_enabled": bool(pins),
        "pinned_ensembles_sha256": sha,
        "pinned_ensembles_path": str(resolved),
        "pinned_target_count": int(len(pins)),
        "pinned_provenance_sha": ANCHOR_PROVENANCE_SHA,
    }
    merge_photometry_pipeline_meta(output_dir, block)
    return block


def _tier_limit_for_member(comp_tier: int, tier_defs: list[tuple[int, float]]) -> float:
    for tier, limit in tier_defs:
        if int(comp_tier) == int(tier):
            return float(limit)
    return float(tier_defs[-1][1]) if tier_defs else 999.0


def catalog_delta_bprp_from_row(
    ms_row: pd.Series,
    target_bprp_eff: float,
) -> tuple[float, float, str]:
    """|dBP-RP| from Gaia catalog ``bp_rp`` columns (same authority as Phase-1 selection).

    Returns ``(delta_abs, comp_bprp, source)``. Source is always ``masterstars.bp_rp``.
    """
    try:
        comp_bprp = float(pd.to_numeric(ms_row.get("bp_rp"), errors="coerce"))
    except (TypeError, ValueError):
        comp_bprp = float("nan")
    if not math.isfinite(comp_bprp):
        return float("nan"), float("nan"), "masterstars.bp_rp_missing"
    if not math.isfinite(float(target_bprp_eff)):
        return float("nan"), comp_bprp, "masterstars.bp_rp"
    return abs(comp_bprp - float(target_bprp_eff)), comp_bprp, "masterstars.bp_rp"


def diagnose_pinned_color_member(
    *,
    target_cid: str,
    comp_cid: str,
    ms_row: pd.Series,
    target_bprp_eff: float,
    comp_tier: int,
    tier_defs: list[tuple[int, float]],
    pin_time_row: pd.Series | None = None,
) -> dict[str, Any]:
    """Per-comp color audit: pin-time vs re-validation catalog authority."""
    delta_now, comp_bprp_now, src = catalog_delta_bprp_from_row(ms_row, target_bprp_eff)
    tier_lim = _tier_limit_for_member(comp_tier, tier_defs)
    out: dict[str, Any] = {
        "target_catalog_id": target_cid,
        "comp_catalog_id": comp_cid,
        "comp_tier": int(comp_tier),
        "tier_limit": tier_lim,
        "revalidation_delta_bprp": delta_now,
        "revalidation_comp_bprp": comp_bprp_now,
        "revalidation_target_bprp": float(target_bprp_eff),
        "revalidation_source": src,
    }
    if pin_time_row is not None:
        try:
            pt_delta = float(pd.to_numeric(pin_time_row.get("delta_bprp_abs"), errors="coerce"))
        except (TypeError, ValueError):
            pt_delta = float("nan")
        out["pin_time_delta_bprp_abs"] = pt_delta
        out["pin_time_comp_bprp"] = float(pd.to_numeric(pin_time_row.get("bp_rp"), errors="coerce"))
        out["pin_time_target_bprp"] = float(pd.to_numeric(pin_time_row.get("target_bp_rp"), errors="coerce"))
        out["pin_time_source"] = "anchor_comp_pt.delta_bprp_abs"
        if math.isfinite(delta_now) and math.isfinite(pt_delta):
            out["catalog_delta_stable"] = abs(delta_now - pt_delta) < 1e-9
    return out


def validate_pinned_member(
    ms_row: pd.Series,
    *,
    target_cid: str,
    target_bprp_eff: float,
    dist_arcsec: float,
    comp_rms: float,
    min_dist_arcsec: float,
    max_comp_rms: float,
    max_delta_bprp_cfg: float,
    comp_tier: int,
    tier_defs: list[tuple[int, float]],
) -> tuple[bool, str]:
    from photometry_core import _bool_col  # noqa: PLC0415

    _ = max_delta_bprp_cfg  # diagnostic ceiling only; tier gate is the catalog rule for pins

    cid = normalize_gaia_source_id(ms_row.get("catalog_id", ms_row.get("name")))
    if not cid:
        return False, "missing_catalog_id"

    if str(cid).strip() == normalize_gaia_source_id(target_cid):
        return False, "is_target"

    if _bool_col(pd.Series([ms_row.get("is_saturated", False)])).iloc[0] or _bool_col(
        pd.Series([ms_row.get("likely_saturated", False)])
    ).iloc[0]:
        return False, "sat_flag"

    if "zone" in ms_row.index:
        z = str(ms_row.get("zone") or "").strip().lower()
        if z in ("saturated", "nonlinear"):
            return False, f"zone_{z}"

    if math.isfinite(min_dist_arcsec) and min_dist_arcsec > 0:
        if not math.isfinite(dist_arcsec) or float(dist_arcsec) < float(min_dist_arcsec):
            return False, "distance_violation"

    delta, _comp_bprp, _src = catalog_delta_bprp_from_row(ms_row, target_bprp_eff)
    tier_lim = _tier_limit_for_member(comp_tier, tier_defs)
    if not math.isfinite(delta):
        return False, "color_catalog_missing_bprp"
    if float(delta) > float(tier_lim):
        return False, f"color_tier{comp_tier}_violation"

    if math.isfinite(max_comp_rms) and math.isfinite(comp_rms):
        if float(comp_rms) > float(max_comp_rms):
            return False, "rms_violation"

    return True, "ok"


def validate_pinned_check_member(
    ms_row: pd.Series,
    *,
    target_cid: str,
    dist_arcsec: float,
    comp_rms: float,
    min_dist_arcsec: float,
    max_comp_rms: float,
) -> tuple[bool, str]:
    """Re-validate pinned check star: data-derived zone/sat/distance/RMS only (no color tier)."""
    from photometry_core import _bool_col  # noqa: PLC0415

    cid = normalize_gaia_source_id(ms_row.get("catalog_id", ms_row.get("name")))
    if not cid:
        return False, "missing_catalog_id"

    if str(cid).strip() == normalize_gaia_source_id(target_cid):
        return False, "is_target"

    if _bool_col(pd.Series([ms_row.get("is_saturated", False)])).iloc[0] or _bool_col(
        pd.Series([ms_row.get("likely_saturated", False)])
    ).iloc[0]:
        return False, "sat_flag"

    if "zone" in ms_row.index:
        z = str(ms_row.get("zone") or "").strip().lower()
        if z in ("saturated", "nonlinear"):
            return False, f"zone_{z}"

    if math.isfinite(min_dist_arcsec) and min_dist_arcsec > 0:
        if not math.isfinite(dist_arcsec) or float(dist_arcsec) < float(min_dist_arcsec):
            return False, "distance_violation"

    if math.isfinite(max_comp_rms) and math.isfinite(comp_rms):
        if float(comp_rms) > float(max_comp_rms):
            return False, "rms_violation"

    return True, "ok"


def select_pinned_comparison_stars_for_target(
    target: pd.Series,
    masterstars_df: pd.DataFrame,
    per_frame_csv_paths: list[Path],
    pin_members: list[PinMember],
    *,
    csv_cache: dict[str, pd.DataFrame] | None = None,
    fwhm_px: float = 3.7,
    max_dist_deg: float = 1.0,
    n_comp_min: int = 3,
    n_comp_max: int = 7,
    max_comp_rms: float = 0.1,
    min_dist_arcsec: float = 60.0,
    min_frames_frac: float = 0.3,
    flux_col: str = "dao_flux",
    chip_fw: int | None = None,
    chip_fh: int | None = None,
    chip_interior_margin_px: int = 0,
    max_delta_bprp: float = 0.5,
    plate_scale_arcsec: float = 1.3,
    use_pixel_dist: bool = False,
    cfg: AppConfig | None = None,
    vsx_local_db_path: str | None = None,
    gaia_db_path: str | None = None,
    gaia_prefetch: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Pinned Phase-1 path: load pin, re-validate, assemble comp rows."""
    from comp_selection_per_target import (  # noqa: PLC0415
        _accumulate_per_frame_comp_metrics,
        _assemble_comp_selection_result_rows,
        _bootstrap_phase1_csv_cache,
        _detrend_and_compute_comp_rms_map,
        _resolve_target_color_for_comp_selection,
    )
    from infolog import log_event  # noqa: PLC0415
    from photometry_core import (  # noqa: PLC0415
        _PHASE_USECOLS_PERFRAME,
        _bool_col,
        _enrich_comp_bp_rp,
        _normalize_id_series,
        _normalize_id_value,
    )

    _cfg_p1 = cfg if cfg is not None else AppConfig()
    ctx = _resolve_target_color_for_comp_selection(
        target,
        vsx_local_db_path=vsx_local_db_path,
        gaia_db_path=gaia_db_path,
        cfg=_cfg_p1,
    )
    target_cid = str(ctx["target_cid"])
    target_bprp_eff = float(ctx["target_bprp_eff"])
    max_delta_bprp_cfg = float(ctx["max_delta_bprp_cfg"])
    tier_defs = list(ctx["TIER_DEFS"])
    ra_t = float(ctx["ra_t"])
    dec_t = float(ctx["dec_t"])

    ms = masterstars_df.copy()
    for col in ("catalog_id", "name"):
        if col in ms.columns:
            ms[col] = _normalize_id_series(ms[col])
    for col in ("is_usable", "is_saturated", "likely_saturated"):
        if col in ms.columns:
            ms[col] = _bool_col(ms[col])

    pin_ids = [m.comp_catalog_id for m in pin_members]
    pin_weight = {m.comp_catalog_id: m.comp_weight for m in pin_members}
    pin_tier = {m.comp_catalog_id: m.comp_tier for m in pin_members}

    id_col = "catalog_id" if "catalog_id" in ms.columns else "name"

    _x_t = float(pd.to_numeric(target.get("x"), errors="coerce"))
    _y_t = float(pd.to_numeric(target.get("y"), errors="coerce"))
    if use_pixel_dist and math.isfinite(_x_t) and math.isfinite(_y_t):
        from comp_selection_per_target import _pixel_distance_deg_vectorized  # noqa: PLC0415

        x_arr = pd.to_numeric(ms.get("x", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        y_arr = pd.to_numeric(ms.get("y", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        ms["_dist_deg"] = _pixel_distance_deg_vectorized(
            _x_t, _y_t, x_arr, y_arr, plate_scale_arcsec=float(plate_scale_arcsec)
        )
    else:
        from comp_selection_per_target import _angular_distance_deg_vectorized  # noqa: PLC0415

        ra_arr = pd.to_numeric(ms.get("ra_deg"), errors="coerce").to_numpy(dtype=float)
        dec_arr = pd.to_numeric(ms.get("dec_deg"), errors="coerce").to_numpy(dtype=float)
        ms["_dist_deg"] = _angular_distance_deg_vectorized(ra_t, dec_t, ra_arr, dec_arr)

    ms_indexed = ms.set_index(ms[id_col].astype(str).str.strip(), drop=False)

    avail_cols = _PHASE_USECOLS_PERFRAME.copy()
    csv_cache = _bootstrap_phase1_csv_cache(
        per_frame_csv_paths, csv_cache, flux_col=flux_col, avail_cols=avail_cols
    )
    metrics = _accumulate_per_frame_comp_metrics(
        per_frame_csv_paths,
        csv_cache,
        set(pin_ids),
        flux_col=flux_col,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
    )
    flux_map = metrics["flux_map"]
    n_frames_loaded = int(metrics["n_frames_loaded"])
    min_frames = max(3, int(n_frames_loaded * min_frames_frac))

    rms_result = _detrend_and_compute_comp_rms_map(
        flux_map,
        min_frames=min_frames,
        max_comp_rms=float(max_comp_rms),
        n_comp_min=int(n_comp_min),
        target_cid=target_cid,
        target=target,
        chip_fw=chip_fw,
        chip_fh=chip_fh,
        chip_interior_margin_px=int(chip_interior_margin_px),
        skip_apriori_rms=True,
    )
    if rms_result[0] is None:
        raise PinnedEnsembleInsufficientError(
            f"target={target_cid} no pinned comps with enough frames for RMS"
        )
    rms_map, _sorted_rms_map = rms_result

    survivors: list[str] = []
    drop_log: list[tuple[str, str]] = []
    rows_for_asm: list[pd.Series] = []

    for pm in pin_members:
        cid = pm.comp_catalog_id
        if cid not in ms_indexed.index:
            drop_log.append((cid, "missing_from_masterstars"))
            log_event(f"[PIN-DROP] target={target_cid} comp={cid} reason=missing_from_masterstars")
            continue
        row = ms_indexed.loc[cid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        n_fr = len(flux_map.get(cid, []))
        if n_fr < min_frames:
            drop_log.append((cid, "insufficient_frames"))
            log_event(
                f"[PIN-DROP] target={target_cid} comp={cid} reason=insufficient_frames "
                f"(n={n_fr} min={min_frames})"
            )
            continue

        _dist_raw = pd.to_numeric(row.get("_dist_deg"), errors="coerce")
        dist_arcsec = float(_dist_raw) * 3600.0 if pd.notna(_dist_raw) else float("nan")
        comp_rms = float(rms_map.get(cid, float("nan")))
        ok, reason = validate_pinned_member(
            row,
            target_cid=target_cid,
            target_bprp_eff=target_bprp_eff,
            dist_arcsec=dist_arcsec,
            comp_rms=comp_rms,
            min_dist_arcsec=float(min_dist_arcsec),
            max_comp_rms=float(max_comp_rms),
            max_delta_bprp_cfg=float(max_delta_bprp_cfg),
            comp_tier=int(pm.comp_tier),
            tier_defs=tier_defs,
        )
        if not ok:
            drop_log.append((cid, reason))
            log_event(f"[PIN-DROP] target={target_cid} comp={cid} reason={reason}")
            continue
        survivors.append(cid)
        rows_for_asm.append(row)

    if len(survivors) < int(n_comp_min):
        detail = (
            f"target={target_cid} n_survivors={len(survivors)} n_min={n_comp_min} "
            f"drops={drop_log}"
        )
        raise PinnedEnsembleInsufficientError(detail)

    # Preserve pin order; cap at n_comp_max
    survivors = survivors[: int(n_comp_max)]
    final_comps = pd.DataFrame(rows_for_asm)
    if "catalog_id" in final_comps.columns:
        final_comps["catalog_id"] = final_comps["catalog_id"].map(
            lambda x: normalize_gaia_source_id(x) or str(x)
        )
    final_comps = _enrich_comp_bp_rp(
        final_comps,
        gaia_db_path=gaia_db_path,
        gaia_prefetch=gaia_prefetch,
    )

    active_rms = {cid: float(rms_map.get(cid, float("nan"))) for cid in survivors}
    comp_bprp_map: dict[str, float] = {}
    comp_delta_map: dict[str, float] = {}
    comp_tier_map: dict[str, int] = {}
    for cid in survivors:
        try:
            comp_bprp_map[cid] = float(pd.to_numeric(final_comps.loc[
                final_comps[id_col].astype(str).str.strip() == cid, "bp_rp"
            ].iloc[0], errors="coerce"))
        except (IndexError, TypeError, ValueError):
            comp_bprp_map[cid] = float("nan")
        if math.isfinite(comp_bprp_map.get(cid, float("nan"))) and math.isfinite(target_bprp_eff):
            comp_delta_map[cid] = abs(float(comp_bprp_map[cid]) - float(target_bprp_eff))
        else:
            comp_delta_map[cid] = float("nan")
        comp_tier_map[cid] = int(pin_tier.get(cid, 4))

    sel_note = (
        f"pinned_ensemble; provenance={ANCHOR_PROVENANCE_SHA}; "
        f"n_pin={len(pin_members)} n_survivors={len(survivors)}"
    )
    if drop_log:
        sel_note += "; drops=" + ",".join(f"{c}:{r}" for c, r in drop_log)

    result = _assemble_comp_selection_result_rows(
        survivors,
        final_comps,
        id_col_cand=id_col,
        active=active_rms,
        score_map={cid: float("nan") for cid in survivors},
        contamination_map={},
        flux_map=flux_map,
        target_cid=target_cid,
        target=target,
        target_bprp_eff=target_bprp_eff,
        t_bp_tgt=float(ctx["t_bp_tgt"]),
        sel_note=sel_note,
        used_mag_tol=float("nan"),
        best_tier="PINNED",
        tier4_warning=False,
        n_t1=sum(1 for c in survivors if comp_tier_map.get(c, 4) == 1),
        n_t2=sum(1 for c in survivors if comp_tier_map.get(c, 4) == 2),
        n_t3=sum(1 for c in survivors if comp_tier_map.get(c, 4) == 3),
        n_t4=sum(1 for c in survivors if comp_tier_map.get(c, 4) >= 4),
        comp_bprp_map=comp_bprp_map,
        comp_tier_final_map=comp_tier_map,
        comp_delta_bprp_map=comp_delta_map,
        comp_color_tier_src_map={cid: "pinned" for cid in survivors},
        _b_rejected=set(),
        final_lookup=final_comps,
        cfg=_cfg_p1,
        comp_path="pinned",
    )

    if not result.empty and "comp_weight" in result.columns:
        for i, cid in enumerate(result["catalog_id"].astype(str).str.strip()):
            w = pin_weight.get(cid)
            if w is not None and math.isfinite(float(w)):
                result.at[result.index[i], "comp_weight"] = float(w)

    result = overlay_anchor_comp_rows(
        result,
        target_catalog_id=target_cid,
        survivor_ids=survivors,
        pin_members=pin_members,
    )

    try:
        base_pt = load_baseline_comp_pt()
        sub_pt = base_pt.loc[base_pt["target_catalog_id"].astype(str).str.strip().eq(target_cid)].copy()
        verify_inv_pin_04(
            target_cid,
            pin_members,
            ms,
            target_bprp_eff,
            pin_time_comp_pt=sub_pt,
        )
    except Exception as _inv4_exc:  # noqa: BLE001
        LOGGER.warning("[INV-PIN-04] check failed for %s: %s", target_cid, _inv4_exc)
        raise

    log_event(
        f"[PIN-OK] target={target_cid} n={len(result)} mode=pinned "
        f"provenance={ANCHOR_PROVENANCE_SHA}"
    )
    return result


def verify_inv_pin_01(
    target_catalog_id: str,
    selected_comp_ids: list[str],
    pin_members: list[PinMember],
    drop_log: list[tuple[str, str]] | None = None,
) -> None:
    """Pinned target reproduces exact membership when all members pass rules."""
    from invariants_runtime import inv_check  # noqa: PLC0415

    expected = [m.comp_catalog_id for m in pin_members]
    drops = drop_log or []
    if drops:
        inv_check(
            {},
            "INV-PIN-01",
            True,
            policy="WARN",
            detail=f"target={target_catalog_id} drops present (expected when rules fail): {drops}",
        )
        return
    ok = sorted(selected_comp_ids) == sorted(expected)
    inv_check(
        {},
        "INV-PIN-01",
        ok,
        policy="FAIL",
        detail=(
            f"target={target_catalog_id} expected={expected} got={selected_comp_ids}"
        ),
    )


def verify_inv_pin_02(drop_log: list[tuple[str, str]]) -> None:
    """Rule-violating members must drop with named reasons (never silent)."""
    from invariants_runtime import inv_check  # noqa: PLC0415

    for cid, reason in drop_log:
        ok = bool(str(reason).strip()) and str(reason).strip() != "ok"
        inv_check(
            {},
            "INV-PIN-02",
            ok,
            policy="FAIL",
            detail=f"comp={cid} silent_or_empty_drop reason={reason!r}",
        )


def verify_inv_pin_04(
    target_catalog_id: str,
    pin_members: list[PinMember],
    ms_df: pd.DataFrame,
    target_bprp_eff: float,
    *,
    pin_time_comp_pt: pd.DataFrame | None = None,
) -> None:
    """Catalog-derived color cannot newly fail: pin-time vs re-validation delta must match."""
    from invariants_runtime import inv_check  # noqa: PLC0415

    id_col = "catalog_id" if "catalog_id" in ms_df.columns else "name"
    ms_idx = ms_df.set_index(ms_df[id_col].astype(str).str.strip(), drop=False)
    tier_defs = [(1, 0.25), (2, 0.48), (3, 0.79), (4, 999.0)]
    failures: list[str] = []
    for pm in pin_members:
        cid = pm.comp_catalog_id
        if cid not in ms_idx.index:
            failures.append(f"{cid}: missing_from_masterstars")
            continue
        row = ms_idx.loc[cid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        pin_row = None
        if pin_time_comp_pt is not None and not pin_time_comp_pt.empty:
            sub = pin_time_comp_pt.loc[
                pin_time_comp_pt["catalog_id"].astype(str).str.strip().eq(cid)
            ]
            pin_row = sub.iloc[0] if not sub.empty else None
        diag = diagnose_pinned_color_member(
            target_cid=target_catalog_id,
            comp_cid=cid,
            ms_row=row,
            target_bprp_eff=float(target_bprp_eff),
            comp_tier=int(pm.comp_tier),
            tier_defs=tier_defs,
            pin_time_row=pin_row,
        )
        if pin_row is not None and not diag.get("catalog_delta_stable", True):
            failures.append(f"{cid}: catalog_delta_changed {diag}")
        ok_now, reason_now = validate_pinned_member(
            row,
            target_cid=target_catalog_id,
            target_bprp_eff=float(target_bprp_eff),
            dist_arcsec=9999.0,
            comp_rms=0.0,
            min_dist_arcsec=0.0,
            max_comp_rms=999.0,
            max_delta_bprp_cfg=0.79,
            comp_tier=int(pm.comp_tier),
            tier_defs=tier_defs,
        )
        if pin_row is not None:
            pt_delta = float(pd.to_numeric(pin_row.get("delta_bprp_abs"), errors="coerce"))
            tier_lim = _tier_limit_for_member(int(pm.comp_tier), tier_defs)
            ok_pin = math.isfinite(pt_delta) and float(pt_delta) <= float(tier_lim)
            if ok_pin and not ok_now:
                failures.append(f"{cid}: newly_fails_color ok_pin={ok_pin} reason={reason_now} diag={diag}")
    inv_check(
        {},
        "INV-PIN-04",
        not failures,
        policy="FAIL",
        detail=f"target={target_catalog_id} failures={failures}",
    )


def verify_inv_pin_03(meta: dict[str, Any], expected_sha: str) -> None:
    """Pin file SHA must appear in pipeline_meta."""
    from invariants_runtime import inv_check  # noqa: PLC0415

    got = str(meta.get("pinned_ensembles_sha256") or "").strip()
    ok = bool(got) and (not expected_sha or got == expected_sha)
    inv_check(
        meta,
        "INV-PIN-03",
        ok,
        policy="FAIL",
        detail=f"meta_sha={got!r} expected={expected_sha!r}",
    )


def _baseline_check_catalog_id_for_target(target_catalog_id: str) -> tuple[str, str]:
    """Read anchor-era check star id (+ kname placeholder) from ``check_kmag`` sidecar."""
    tid = normalize_gaia_source_id(target_catalog_id)
    if not tid:
        return "", ""
    path = DEFAULT_BASELINE_LC_DIR / f"check_kmag_{tid}.csv"
    if not path.is_file():
        return "", ""
    try:
        df = pd.read_csv(path, dtype={"check_catalog_id": str}, nrows=1)
    except (OSError, KeyError, ValueError):
        return "", ""
    if df.empty or "check_catalog_id" not in df.columns:
        return "", ""
    cid = normalize_gaia_source_id(df["check_catalog_id"].iloc[0])
    if not cid:
        return "", ""
    kname = str(df["check_kname"].iloc[0]).strip() if "check_kname" in df.columns else cid
    return cid, kname or cid


def generate_pinned_ensembles_csv(
    comp_pt_path: Path,
    baseline_lc_target_ids: list[str],
    out_path: Path,
    *,
    provenance_sha: str = ANCHOR_PROVENANCE_SHA,
    pin_date: str | None = None,
) -> Path:
    """Build ``pinned_ensembles.csv`` from anchor comp_pt + check_kmag sidecars."""
    df = pd.read_csv(
        comp_pt_path,
        dtype={"catalog_id": str, "target_catalog_id": str},
        low_memory=False,
    )
    ids = {normalize_gaia_source_id(x) for x in baseline_lc_target_ids}
    ids.discard(None)
    sub = df[df["target_catalog_id"].astype(str).str.strip().isin(ids)].copy()
    if sub.empty:
        raise ValueError(f"no pinned rows from {comp_pt_path}")

    pin_dt = pin_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    check_by_tgt: dict[str, tuple[str, str]] = {}
    for tid in sorted(ids):
        cid, kname = _baseline_check_catalog_id_for_target(str(tid))
        if cid:
            check_by_tgt[str(tid)] = (cid, kname)

    rows: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        tgt = normalize_gaia_source_id(r.get("target_catalog_id"))
        comp = normalize_gaia_source_id(r.get("catalog_id"))
        if not tgt or not comp:
            continue
        try:
            w = float(pd.to_numeric(r.get("comp_weight"), errors="coerce"))
        except (TypeError, ValueError):
            w = float("nan")
        try:
            tier = int(pd.to_numeric(r.get("comp_tier"), errors="coerce") or 4)
        except (TypeError, ValueError):
            tier = 4
        chk_cid, chk_kname = check_by_tgt.get(str(tgt), ("", ""))
        rows.append(
            {
                "target_catalog_id": tgt,
                "comp_catalog_id": comp,
                "check_catalog_id": chk_cid,
                "check_kname": chk_kname,
                "comp_weight": w if np.isfinite(w) else "",
                "comp_tier": tier,
                "weights_source_era": provenance_sha,
                "pin_reason": PIN_REASON,
                "provenance_sha": provenance_sha,
                "pin_date": pin_dt,
            }
        )

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path
