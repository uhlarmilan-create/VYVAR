"""Per-target trust flag (GREEN / YELLOW / RED) — shared by pipeline and CLI.

Production trust inputs (no SEP / xval axis):
- **Comp-set quality:** `n_clean` from comp_qa (Sokolovsky leave-one-out locus)
- **Check-star scatter** (population RMS vs configured soft/hard thresholds)
- **`lc_quality_flag`** from Phase 2A summary (`good` / `noisy` informational; saturated etc. hard)

SEP vs DAO cross-validation is **offline-only** (`xval_run.py` / `xval_harness_core.py`) — a
one-off validation study tool, not imported by the production pipeline and not a trust axis.

Unevaluated targets default **RED** (fail-closed).

Comp-count thresholds follow ``comp_trust_min_comps`` / ``phase01_comparison_n_comp_max``
(Phase-1 selection floor ``phase01_comparison_n_comp_min`` unchanged). Read-only w.r.t.
numeric photometry.

Gate semantics (Phase-1 degradation, green_min=3):
- **RED:** ``n_clean == 0`` OR no check star OR hard warnings (bad lc_quality, check ≥ 0.05).
- **YELLOW:** thin comp set (1–2 clean), sparse_fallback, T3+ with colour-term off, soft check scatter.
- **GREEN:** ``n_clean ≥ green_min`` (T1/T2 count ≥ green_min) + check + comp_qa OK, no soft warnings.
- ``short_baseline`` is a non-escalating soft (excluded from ``len(soft) >= 3`` -> RED).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gaia_catalog_id import norm_id_or_empty as norm_id

LOGGER = logging.getLogger(__name__)

_LC_QUALITY_OK = frozenset({"good", "noisy"})
_LC_QUALITY_SOFT = frozenset({"short_baseline"})
_CHECK_SOFT_LO = 0.02
_CHECK_HARD_LO = 0.05
_UNEVALUATED_TRUST = "RED"
_UNEVALUATED_REASON = "not evaluated (no comp QA / missing from trust map)"


@dataclass(frozen=True, slots=True)
class CompTrustThresholds:
    min_comps: int
    max_comps: int
    strong: int

    @classmethod
    def from_bounds(cls, green_min: int, max_comps: int) -> CompTrustThresholds:
        """``green_min`` = GREEN threshold (``comp_trust_min_comps``); not a RED gate."""
        gm = max(1, int(green_min))
        mx = max(gm, int(max_comps))
        return cls(min_comps=1, max_comps=mx, strong=gm)


def comp_thresholds_from_config(cfg: Any | None) -> CompTrustThresholds:
    """Trust GREEN threshold (``comp_trust_min_comps`` = green_min); selection floor unchanged."""
    if cfg is None:
        return CompTrustThresholds.from_bounds(3, 12)
    if getattr(cfg, "comp_trust_min_comps", None) is not None:
        gm = int(getattr(cfg, "comp_trust_min_comps"))
    else:
        gm = int(getattr(cfg, "phase01_comparison_n_comp_min", 3))
    mx = int(getattr(cfg, "phase01_comparison_n_comp_max", 12))
    return CompTrustThresholds.from_bounds(gm, mx)


def check_star_min_epochs_from_config(cfg: Any | None) -> int:
    """Minimum check-star epochs before scatter thresholds apply."""
    if cfg is None:
        return 5
    try:
        v = int(getattr(cfg, "check_star_min_epochs", 5) or 5)
    except (TypeError, ValueError):
        v = 5
    return max(3, v)


def check_star_scatter(photometry_dir: Path, target_id: str) -> tuple[float, int]:
    """Return (scatter mag, finite epoch count). Missing file -> (nan, 0)."""
    p = Path(photometry_dir) / "lightcurves" / f"check_kmag_{target_id}.csv"
    if not p.is_file():
        return float("nan"), 0
    try:
        sdf = pd.read_csv(p, low_memory=False)
        if sdf.empty or "kmag" not in sdf.columns:
            return float("nan"), 0
        km = pd.to_numeric(sdf["kmag"], errors="coerce")
        n_finite = int(km.notna().sum())
        if n_finite < 2:
            return float("nan"), n_finite
        return float(np.nanstd(km, ddof=1)), n_finite
    except Exception:  # noqa: BLE001
        return float("nan"), 0


def _escalating_soft_count(soft: list[str]) -> int:
    """Soft warnings that count toward the ``len(soft) >= 3`` RED escalation guard."""
    return sum(1 for s in soft if not str(s).startswith("short baseline"))


def classify_warnings(
    *,
    n_clean: int,
    check_scatter: float,
    lc_quality: str,
    thresholds: CompTrustThresholds,
    n_frames: int | None = None,
    n_check: int = 0,
    check_min_epochs: int = 5,
    iterative_clip_used: bool = False,
    sparse_fallback_used: bool = False,
    comp_pool_n_final: int = 0,
    n_stability_suspect: int = 0,
    n_tier12: int | None = None,
    color_term_off: bool = False,
    n_alignment_failed: int = 0,
    n_wcs_untrusted: int = 0,
) -> tuple[list[str], list[str]]:
    """Return (hard_labels, soft_labels) for one target."""
    hard: list[str] = []
    soft: list[str] = []
    nc = int(n_clean)
    th = thresholds
    n_chk = int(n_check)
    min_chk = max(3, int(check_min_epochs))
    _sparse = bool(sparse_fallback_used or iterative_clip_used)

    if nc <= 0:
        hard.append("no clean comps")

    if _sparse:
        _nf = int(comp_pool_n_final) if int(comp_pool_n_final) > 0 else nc
        soft.append(
            f"sparse_fallback comp path ({nc} clean / {_nf} selected)"
        )
    elif 0 < nc < th.strong:
        soft.append(
            f"thin comp set ({nc} clean, GREEN requires >={th.strong})"
        )

    _nt12 = int(n_tier12) if n_tier12 is not None else nc
    if nc > 0 and _nt12 < nc:
        soft.append("ensemble includes T3+ comps (colour mismatch)")
    if nc >= th.strong and _nt12 < th.strong:
        soft.append(f"only {_nt12} T1/T2 comps (GREEN requires >={th.strong})")
    if bool(color_term_off) and nc > 0 and _nt12 < th.strong:
        soft.append("T3+ comps with color term off (trust capped YELLOW)")

    lq = str(lc_quality or "").strip().lower()
    if lq in _LC_QUALITY_SOFT:
        nf_note = ""
        if n_frames is not None:
            try:
                nf_i = int(n_frames)
                if nf_i > 0:
                    nf_note = f"{nf_i} frames, "
            except (TypeError, ValueError):
                pass
        soft.append(f"short baseline ({nf_note}thin series)")
    elif lq and lq not in _LC_QUALITY_OK and lq != "—":
        hard.append(f"LC quality: {lq}")

    if n_chk == 0:
        hard.append("no check-star verification")
    elif 0 < n_chk < min_chk:
        soft.append(f"insufficient check-star verification (n={n_chk})")
    elif math.isfinite(check_scatter):
        if check_scatter >= _CHECK_HARD_LO:
            hard.append(f"check-star scatter {check_scatter:.3f} mag (high)")
        elif check_scatter >= _CHECK_SOFT_LO:
            soft.append(f"check-star scatter {check_scatter:.3f} mag")

    ns = int(n_stability_suspect)
    if ns > 0:
        soft.append(f"{ns} comp(s) flagged suspect by stability (kept in ensemble)")

    n_af = int(n_alignment_failed)
    if n_af > 0:
        nf_af = int(n_frames) if n_frames is not None else 0
        frac_af = float(n_af) / max(nf_af, 1)
        if frac_af >= 0.05 or n_af >= 2:
            soft.append(f"{n_af} epoch(s) failed alignment ({frac_af:.0%} of LC)")

    n_wut = int(n_wcs_untrusted)
    if n_wut > 0:
        nf_wut = int(n_frames) if n_frames is not None else 0
        frac_wut = float(n_wut) / max(nf_wut, 1)
        if frac_wut >= 0.05 or n_wut >= 2:
            soft.append(f"{n_wut} epoch(s) pixel-fallback WCS match ({frac_wut:.0%} of LC)")

    return hard, soft


def trust_level(
    n_clean: int,
    hard: list[str],
    soft: list[str],
    thresholds: CompTrustThresholds,
) -> str:
    # len(soft)>=3 is a forward guard (short_baseline excluded from escalation count).
    if hard or _escalating_soft_count(soft) >= 3:
        return "RED"
    if soft:
        return "YELLOW"
    if int(n_clean) >= thresholds.strong:
        return "GREEN"
    return "YELLOW"


def build_reason(
    trust: str,
    n_clean: int,
    lc_quality: str,
    check_scatter: float,
    hard: list[str],
    soft: list[str],
    thresholds: CompTrustThresholds,
) -> str:
    positives: list[str] = [
        f"{int(n_clean)} clean comp{'s' if int(n_clean) != 1 else ''}"
    ]

    lq = str(lc_quality or "").strip().lower()
    if lq == "noisy":
        positives.append("noisy LC (informational)")

    segments: list[str] = []
    if positives:
        segments.append(", ".join(positives))
    if hard:
        segments.append("hard: " + "; ".join(hard))
    if soft:
        segments.append("soft: " + "; ".join(soft))

    text = " | ".join(segments) if segments else "no warnings"
    if trust == "GREEN":
        return text
    if trust == "YELLOW":
        return text + " — review before submitting"
    return text + " — inspect before submitting"


def evaluate_target(
    *,
    catalog_id: str,
    vsx_name: str,
    n_clean: int,
    lc_quality: str,
    check_scatter: float,
    thresholds: CompTrustThresholds | None = None,
    n_frames: int | None = None,
    n_check: int = 0,
    check_min_epochs: int = 5,
    iterative_clip_used: bool = False,
    sparse_fallback_used: bool = False,
    comp_pool_n_final: int = 0,
    n_stability_suspect: int = 0,
    n_tier12: int | None = None,
    color_term_off: bool = False,
    n_alignment_failed: int = 0,
    n_wcs_untrusted: int = 0,
) -> dict[str, Any]:
    th = thresholds or CompTrustThresholds.from_bounds(3, 8)
    lq = str(lc_quality or "").strip().lower() or "—"
    hard, soft = classify_warnings(
        n_clean=int(n_clean),
        check_scatter=float(check_scatter),
        lc_quality=lq,
        thresholds=th,
        n_frames=n_frames,
        n_check=int(n_check),
        check_min_epochs=int(check_min_epochs),
        iterative_clip_used=bool(iterative_clip_used),
        sparse_fallback_used=bool(sparse_fallback_used),
        comp_pool_n_final=int(comp_pool_n_final),
        n_stability_suspect=int(n_stability_suspect),
        n_tier12=n_tier12,
        color_term_off=bool(color_term_off),
        n_alignment_failed=int(n_alignment_failed),
        n_wcs_untrusted=int(n_wcs_untrusted),
    )
    trust = trust_level(int(n_clean), hard, soft, th)
    reason = build_reason(
        trust, int(n_clean), lq, float(check_scatter), hard, soft, th
    )
    return {
        "catalog_id": catalog_id,
        "vsx_name": vsx_name,
        "trust": trust,
        "trust_reason": reason,
        "n_clean": int(n_clean),
        "lc_quality": lq,
        "check_scatter": float(check_scatter) if math.isfinite(check_scatter) else None,
        "n_check": int(n_check),
        "check_min_epochs": max(3, int(check_min_epochs)),
        "hard_warnings": hard,
        "soft_warnings": soft,
        "n_hard": len(hard),
        "n_soft": len(soft),
        "min_comps": th.min_comps,
        "strong_comps": th.strong,
    }


def compute_trust_for_photometry_dir(
    photometry_dir: Path,
    *,
    thresholds: CompTrustThresholds | None = None,
    check_min_epochs: int | None = None,
    cfg: Any | None = None,
) -> dict[str, Any]:
    phot = Path(photometry_dir)
    summ_path = phot / "photometry_summary.csv"
    if not summ_path.is_file():
        raise FileNotFoundError(f"missing {summ_path}")

    th = thresholds or CompTrustThresholds.from_bounds(3, 8)
    min_chk = max(3, int(check_min_epochs if check_min_epochs is not None else 5))
    summ = pd.read_csv(summ_path, dtype={"catalog_id": str}, low_memory=False)
    _color_term_off = True
    if cfg is not None:
        try:
            _color_term_off = not bool(getattr(cfg, "apply_color_term", True))
        except Exception:  # noqa: BLE001
            _color_term_off = True
    comp_meta: dict[str, dict[str, int]] = {}
    comp_path = phot / "comparison_stars_per_target.csv"
    if comp_path.is_file():
        try:
            comp_df = pd.read_csv(
                comp_path,
                dtype={"target_catalog_id": str, "catalog_id": str},
                low_memory=False,
            )
            if "target_catalog_id" in comp_df.columns:
                for tid, grp in comp_df.groupby(comp_df["target_catalog_id"].astype(str).str.strip()):
                    if not tid:
                        continue
                    ci = pd.to_numeric(grp.get("comp_clip_iterations"), errors="coerce")
                    nf = pd.to_numeric(grp.get("comp_pool_n_final"), errors="coerce")
                    _paths = (
                        grp.get("comp_path", pd.Series("", index=grp.index))
                        .astype(str)
                        .str.strip()
                        .str.lower()
                    )
                    comp_meta[tid] = {
                        "comp_clip_iterations": int(ci.max()) if ci.notna().any() else 0,
                        "comp_pool_n_final": int(nf.max()) if nf.notna().any() else 0,
                        "sparse_fallback_used": bool((_paths == "sparse_fallback").any()),
                    }
        except Exception:  # noqa: BLE001
            comp_meta = {}
    per_target: dict[str, dict[str, Any]] = {}
    conf_counts: dict[str, int] = {}

    for _, row in summ.iterrows():
        cid = norm_id(row.get("catalog_id"))
        if not cid:
            continue
        n_clean_raw = pd.to_numeric(row.get("n_clean"), errors="coerce")
        if math.isfinite(float(n_clean_raw)):
            n_clean = int(n_clean_raw)
        else:
            ng_raw = pd.to_numeric(row.get("n_good_comp"), errors="coerce")
            n_clean = int(ng_raw) if math.isfinite(float(ng_raw)) else 0
        lc_quality = str(row.get("lc_quality_flag", "") or "").strip()
        vsx = str(row.get("vsx_name", "") or "")
        nf_raw = pd.to_numeric(row.get("n_frames"), errors="coerce")
        n_frames = int(nf_raw) if math.isfinite(float(nf_raw)) else None
        chk, n_chk = check_star_scatter(phot, cid)
        _cm = comp_meta.get(cid, {})
        _clip_iters = int(_cm.get("comp_clip_iterations", 0))
        _sparse_fb = bool(_cm.get("sparse_fallback_used", False))
        _stab_sus_raw = pd.to_numeric(row.get("n_stability_suspect"), errors="coerce")
        _stab_sus = int(_stab_sus_raw) if math.isfinite(float(_stab_sus_raw)) else 0
        _nt12_raw = pd.to_numeric(row.get("n_tier12"), errors="coerce")
        _nt12 = int(_nt12_raw) if math.isfinite(float(_nt12_raw)) else None
        _naf_raw = pd.to_numeric(row.get("n_alignment_failed"), errors="coerce")
        _n_af = int(_naf_raw) if math.isfinite(float(_naf_raw)) else 0
        _nwut_raw = pd.to_numeric(row.get("n_wcs_untrusted"), errors="coerce")
        _n_wut = int(_nwut_raw) if math.isfinite(float(_nwut_raw)) else 0
        info = evaluate_target(
            catalog_id=cid,
            vsx_name=vsx,
            n_clean=n_clean,
            lc_quality=lc_quality,
            check_scatter=chk,
            thresholds=th,
            n_frames=n_frames,
            n_check=n_chk,
            check_min_epochs=min_chk,
            iterative_clip_used=_clip_iters > 0,
            sparse_fallback_used=_sparse_fb,
            comp_pool_n_final=int(_cm.get("comp_pool_n_final", 0)),
            n_stability_suspect=_stab_sus,
            n_tier12=_nt12,
            color_term_off=_color_term_off,
            n_alignment_failed=_n_af,
            n_wcs_untrusted=_n_wut,
        )
        per_target[cid] = info
        conf_counts[info["trust"]] = conf_counts.get(info["trust"], 0) + 1

    return {
        "per_target": per_target,
        "stats": {
            "n_targets": len(per_target),
            "trust": conf_counts,
            "min_comps": th.min_comps,
            "max_comps": th.max_comps,
            "strong_comps": th.strong,
        },
    }


def write_trust_artifacts(
    result: dict[str, Any],
    *,
    photometry_dir: Path,
    lc_dir: Path | None = None,
    update_summary: bool = True,
    write_per_target_json: bool = True,
) -> list[Path]:
    phot = Path(photometry_dir)
    lc = Path(lc_dir) if lc_dir is not None else phot / "lightcurves"
    lc.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    trust_map = {
        tid: str(info.get("trust") or _UNEVALUATED_TRUST)
        for tid, info in result.get("per_target", {}).items()
    }
    reason_map = {
        tid: str(info.get("trust_reason") or _UNEVALUATED_REASON)
        for tid, info in result.get("per_target", {}).items()
    }

    if write_per_target_json:
        for tid, info in result.get("per_target", {}).items():
            out_path = lc / f"trust_{tid}.json"
            payload = {
                "target_catalog_id": tid,
                "trust": info.get("trust"),
                "trust_reason": info.get("trust_reason"),
                "n_hard": info.get("n_hard"),
                "n_soft": info.get("n_soft"),
                "hard_warnings": info.get("hard_warnings"),
                "soft_warnings": info.get("soft_warnings"),
                "n_clean": info.get("n_clean"),
                "lc_quality": info.get("lc_quality"),
                "check_scatter": info.get("check_scatter"),
                "min_comps": info.get("min_comps"),
                "strong_comps": info.get("strong_comps"),
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(out_path)

    if update_summary:
        summ_path = phot / "photometry_summary.csv"
        if summ_path.is_file():
            df = pd.read_csv(summ_path, dtype={"catalog_id": str}, low_memory=False)
            if "catalog_id" in df.columns:
                n_missing = sum(
                    1 for x in df["catalog_id"] if norm_id(x) not in trust_map
                )
                if n_missing > 0:
                    LOGGER.warning(
                        "[TRUST] %d summary target(s) absent from trust map -> defaulted to %s",
                        n_missing,
                        _UNEVALUATED_TRUST,
                    )
                df["trust"] = df["catalog_id"].map(
                    lambda x: trust_map.get(norm_id(x), _UNEVALUATED_TRUST)
                )
                df["trust_reason"] = df["catalog_id"].map(
                    lambda x: reason_map.get(norm_id(x), _UNEVALUATED_REASON)
                )
                df.to_csv(summ_path, index=False)
                written.append(summ_path)

    return written


def run_trust_flag_for_photometry_dir(
    *,
    photometry_dir: Path,
    lc_dir: Path | None = None,
    update_summary: bool = True,
    cfg: Any | None = None,
    thresholds: CompTrustThresholds | None = None,
) -> dict[str, Any]:
    """Pipeline / CLI entry: trust columns + optional per-target JSON."""
    th = thresholds or comp_thresholds_from_config(cfg)
    min_chk = check_star_min_epochs_from_config(cfg)
    result = compute_trust_for_photometry_dir(
        photometry_dir, thresholds=th, check_min_epochs=min_chk, cfg=cfg
    )
    paths = write_trust_artifacts(
        result,
        photometry_dir=photometry_dir,
        lc_dir=lc_dir,
        update_summary=update_summary,
    )
    st = result.get("stats", {})
    LOGGER.info(
        "[TRUST] min/strong=%s/%s targets=%s trust=%s → %d artifacts",
        st.get("min_comps"),
        st.get("strong_comps"),
        st.get("n_targets"),
        st.get("trust"),
        len(paths),
    )
    result["written_paths"] = [str(p) for p in paths]
    return result


def format_export_trust_note(trust: str, trust_reason: str, *, max_len: int = 36) -> str:
    """Compact trust tag for AAVSO NOTES (fits after ``meth=…``)."""
    t = str(trust or _UNEVALUATED_TRUST).strip().upper()[:6] or _UNEVALUATED_TRUST
    note = f"trust={t}"
    if trust_reason:
        short = str(trust_reason).split(" — ")[0].strip()
        if len(short) > max_len - len(note) - 1:
            short = short[: max(0, max_len - len(note) - 4)] + "..."
        if short:
            note = f"{note}|{short}"
    return note[:max_len]


def format_varastro_trust_comment(trust: str, trust_reason: str) -> str:
    t = str(trust or _UNEVALUATED_TRUST).strip().upper() or _UNEVALUATED_TRUST
    reason = str(trust_reason or "").strip()
    if reason:
        return f"#   Trust: {t} — {reason}\n"
    return f"#   Trust: {t}\n"
