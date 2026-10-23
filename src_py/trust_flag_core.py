"""Per-target trust flag (GREEN / YELLOW / RED) - shared by pipeline and CLI.

Production trust inputs (no SEP / xval axis):
- **Comp-set size:** `n_clean` = comparison stars in the target's ensemble (comparison_stars_per_target.csv;
  every one passed the selection gates: not variable in Gaia/VSX, neighbour light, frame edge)
- **Check-star scatter** (population RMS vs configured soft/hard thresholds)
- **`lc_quality_flag`** from the aperture photometry summary (`good` / `noisy` informational; saturated etc. hard)

SEP vs DAO cross-validation is not a trust axis (the offline study tools were removed in
CODE-REVIEW-2026-10 step 6).

Unevaluated targets default **RED** (fail-closed).

Comp-count thresholds follow ``comp_trust_min_comps`` / ``phase01_comparison_n_comp_max``
(comparison-star selection floor ``phase01_comparison_n_comp_min`` unchanged). Read-only w.r.t.
numeric photometry.

Gate semantics (comparison-star selection degradation, green_min=3):
- **RED:** ``n_clean == 0`` OR no check star OR hard warnings (bad lc_quality, check >= 0.05).
- **YELLOW:** thin comp set (1-2 clean), T3+ with colour-term off, soft check scatter.
- **GREEN:** ``n_clean >= green_min`` (T1/T2 count >= green_min) + check OK, no soft warnings.
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
# Check-star consistency: robust scatter of (K - median)/sigma_K (1.4826 MAD, so single
# cosmic-ray epochs do not dominate). 1 = scatter explained by the errors.
CHECK_RATIO_SOFT = 1.5
CHECK_RATIO_HARD = 3.0
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


#: TRUST-COVERAGE-2026-10: exported epochs below this fraction of the frames -> YELLOW
COVERAGE_YELLOW_FRAC = 0.5
#: ... below this fraction or below COVERAGE_MIN_POINTS -> RED (insufficient data)
COVERAGE_RED_FRAC = 0.2
COVERAGE_MIN_POINTS = 10
#: TARGET-BLEND-2026-10: neighbour light in the target aperture (fraction of own flux)
#: TRUST-LEVEL-2026-10 (Milan 2026-10-21, AAVSO practice: offsets of several tenths of a magnitude
#: are gross errors; a comparison-star error above 0.05 mag is "extremely high"; Gaia -> band
#: relation scatter ~0.05 mag): measured check star against its catalogue magnitude, and the
#: uncertainty of the zero point from the scatter of the comparison stars.
CHECK_DELTA_SOFT_MAG = 0.10
CHECK_DELTA_HARD_MAG = 0.25
ZP_SIGMA_SOFT_MAG = 0.050
ZP_SIGMA_HARD_MAG = 0.150
TARGET_BLEND_SOFT = 0.01
TARGET_BLEND_HARD = 0.05


def coverage_insufficient(n_export: int, n_frames: int) -> bool:
    """TRUST-COVERAGE-2026-10: fewer usable epochs than max(min(10, n/2), 0.2 n)."""
    n = int(n_frames)
    if n <= 0:
        return False
    return int(n_export) < max(min(COVERAGE_MIN_POINTS, 0.5 * n), COVERAGE_RED_FRAC * n)


def lc_epoch_counts(photometry_dir: Path, target_id: str) -> tuple[int | None, int | None]:
    """(n_exportable, n_frames) from the target's light curve (export rules of lc_outlier)."""
    p = Path(photometry_dir) / "lightcurves" / f"lightcurve_{target_id}.csv"
    if not p.is_file():
        return None, None
    try:
        df = pd.read_csv(p, low_memory=False)
    except (OSError, ValueError):
        return None, None
    # same row selection as the exporters (single source: export_reports._select_export_lc_rows)
    from export_reports import _select_export_lc_rows  # noqa: PLC0415

    n = int(len(df))
    from photometry_lightcurve import published_mag_column  # noqa: PLC0415

    mag_col = next((c for c in (published_mag_column(df), "delta_mag")
                    if c and c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any()), None)
    if mag_col is None:
        return 0, n
    return int(len(_select_export_lc_rows(df, mag_column=mag_col))), n


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


#: Minimum check-star epochs before the scatter thresholds apply.
CHECK_STAR_MIN_EPOCHS = 5


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
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().trust_kmag_sidecar_read_fail += 1
        LOGGER.error(
            "[TRUST] check-star kmag sidecar read failed for %s: %s",
            p,
            exc,
        )
        return float("nan"), 0


def check_star_chi2(photometry_dir: Path, target_id: str) -> tuple[float, float]:
    """(ratio, chi2_p) for the check-star series.

    ratio = 1.4826 MAD of (K_i - median K) / sigma_i - the robust "scatter in units of the
    reported errors" (1 = consistent); chi2_p = chi2 survival of the plain chi2 (info only).
    Needs ``kmag_err`` in the sidecar; otherwise (nan, nan).
    """
    p = Path(photometry_dir) / "lightcurves" / f"check_kmag_{target_id}.csv"
    if not p.is_file():
        return float("nan"), float("nan")
    try:
        sdf = pd.read_csv(p, low_memory=False)
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan")
    if sdf.empty or "kmag" not in sdf.columns or "kmag_err" not in sdf.columns:
        return float("nan"), float("nan")
    k = pd.to_numeric(sdf["kmag"], errors="coerce").to_numpy(dtype=np.float64)
    e = pd.to_numeric(sdf["kmag_err"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(k) & np.isfinite(e) & (e > 0)
    n = int(ok.sum())
    if n < 3:
        return float("nan"), float("nan")
    kk, ee = k[ok], e[ok]
    z = (kk - float(np.median(kk))) / ee
    ratio = float(1.4826 * np.median(np.abs(z - np.median(z))))
    w = 1.0 / (ee**2)
    mu = float(np.sum(w * kk) / np.sum(w))
    chi2 = float(np.sum(w * (kk - mu) ** 2))
    from scipy.stats import chi2 as _chi2  # noqa: PLC0415

    return ratio, float(_chi2.sf(chi2, n - 1))


def check_delta_and_zp_sigma(photometry_dir: Path, target_id: str) -> tuple[float, float]:
    """TRUST-LEVEL-2026-10: (median measured check-star magnitude minus its catalogue magnitude in the
    zero-point band, zero-point uncertainty from the comparison-star scatter - model value when the
    data value is missing). NaN when the target has no zero point or no check star."""
    lc_dir = Path(photometry_dir) / "lightcurves"
    try:
        cq = json.loads((lc_dir / f"comp_quality_{target_id}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return float("nan"), float("nan")
    if not isinstance(cq, dict) or str(cq.get("band_system", "") or "").upper() in ("", "NONE"):
        return float("nan"), float("nan")
    zp = cq.get("zero_point") if isinstance(cq.get("zero_point"), dict) else {}
    zs = zp.get("sigma_zp_data_mag")
    if zs is None:
        zs = zp.get("sigma_zp_model_mag")
    zp_sigma = float(zs) if zs is not None else float("nan")
    delta = float("nan")
    from check_star_kmag import load_check_star  # noqa: PLC0415

    kc = str(load_check_star(lc_dir, str(target_id)).get("catalog_id", "") or "")
    bm = cq.get("comp_band_mag") if isinstance(cq.get("comp_band_mag"), dict) else {}
    kref = pd.to_numeric(pd.Series([bm.get(kc)]), errors="coerce").iloc[0] if kc else float("nan")
    p = lc_dir / f"check_kmag_{target_id}.csv"
    if kc and pd.notna(kref) and p.is_file():
        try:
            km = pd.to_numeric(pd.read_csv(p, usecols=["kmag"])["kmag"], errors="coerce").dropna()
        except (OSError, ValueError):
            km = pd.Series(dtype=float)
        if len(km):
            delta = float(km.median()) - float(kref)
    return delta, zp_sigma


def read_sparse_trust_sidecar(photometry_dir: Path, target_id: str) -> dict[str, Any] | None:
    """The check-star trust statistics of a target (``check_star_kmag.load_check_star``)."""
    from check_star_kmag import load_check_star  # noqa: PLC0415

    blk = load_check_star(Path(photometry_dir) / "lightcurves", str(target_id))
    if "check_sparse" not in blk:
        return None

    def _f(key: str) -> float:
        v = blk.get(key)
        return float(v) if v is not None else float("nan")

    return {
        "check_sparse": bool(blk.get("check_sparse")),
        "trust_R": _f("trust_R"),
        "trust_R_lo": _f("trust_R_lo"),
        "trust_R_hi": _f("trust_R_hi"),
        "comp_stability_p": _f("comp_stability_p"),
        "x2_pair_mag2": _f("x2_pair_mag2"),
        "triangulation_clipped": bool(blk.get("triangulation_clipped")),
        "sparse_flags": tuple(blk.get("sparse_flags") or ()),
    }


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
    n_stability_suspect: int = 0,
    n_tier12: int | None = None,
    color_term_off: bool = False,
    n_alignment_failed: int = 0,
    n_wcs_untrusted: int = 0,
    sparse_trust_band: str | None = None,
    sparse_trust_flags: tuple[str, ...] | None = None,
    use_sparse_check_trust: bool = False,
    check_chi2_nu: float = float("nan"),
    n_export: int | None = None,
    blend_frac: float = float("nan"),
    skip_reason: str = "",
    check_delta_mag: float = float("nan"),
    zp_sigma_mag: float = float("nan"),
) -> tuple[list[str], list[str]]:
    """Return (hard_labels, soft_labels) for one target.

    CHECK-RULES-2026-10: when the check-star sidecar carries per-epoch errors, the check
    star is judged by whether its scatter agrees with its errors (chi2 test, p-value)
    instead of the fixed 0.02 / 0.05 mag levels (kept only for sidecars without errors).
    """
    hard: list[str] = []
    soft: list[str] = []
    nc = int(n_clean)
    th = thresholds
    # a target aperture photometry skipped (e.g. per-frame saturation) has no light curve of this run
    if str(skip_reason or "").strip():
        hard.append(f"not measured: {str(skip_reason).strip()}")
    # TRUST-COVERAGE-2026-10 (Milan 2026-10-11): trust follows the epochs that are actually
    # exported (draft 526: CSS J153101.6 GREEN with 5 of 122 points).
    if n_export is not None and n_frames is not None and int(n_frames) > 0:
        _ne, _nf = int(n_export), int(n_frames)
        if coverage_insufficient(_ne, _nf):
            hard.append(f"insufficient data: {_ne} of {_nf} epochs usable")
        elif _ne < COVERAGE_YELLOW_FRAC * _nf:
            soft.append(f"only {_ne} of {_nf} epochs usable")
    # TARGET-BLEND-2026-10: neighbour light inside the target aperture (Moffat model, comp_blend)
    if math.isfinite(float(blend_frac)):
        if float(blend_frac) > TARGET_BLEND_HARD:
            hard.append(f"blended: neighbours add {100 * float(blend_frac):.0f} % light in the aperture")
        elif float(blend_frac) > TARGET_BLEND_SOFT:
            soft.append(f"neighbours add {100 * float(blend_frac):.1f} % light in the aperture")
    n_chk = int(n_check)
    min_chk = max(3, int(check_min_epochs))

    if nc <= 0:
        hard.append("no clean comps")

    if 0 < nc < th.strong:
        soft.append(
            f"thin comp set ({nc} clean, GREEN requires >={th.strong})"
        )

    # TRUST-LEVEL-2026-10: one cause, one warning (T3+ comps were counted up to three times and
    # pushed targets to RED through the three-soft escalation)
    _nt12 = int(n_tier12) if n_tier12 is not None else nc
    if nc > 0 and _nt12 < nc:
        _t3 = f"ensemble includes {nc - _nt12} T3+ comps (colour mismatch)"
        if _nt12 < th.strong:
            _t3 += f"; only {_nt12} T1/T2 comps (GREEN requires >={th.strong})"
            if bool(color_term_off):
                _t3 += ", colour term off"
        soft.append(_t3)
    # TRUST-LEVEL-2026-10: absolute level (the scatter tests above do not see an offset)
    if math.isfinite(float(check_delta_mag)):
        _cd = abs(float(check_delta_mag))
        if _cd > CHECK_DELTA_HARD_MAG:
            hard.append(f"check star {float(check_delta_mag):+.2f} mag from its catalogue magnitude")
        elif _cd > CHECK_DELTA_SOFT_MAG:
            soft.append(f"check star {float(check_delta_mag):+.2f} mag from its catalogue magnitude")
    if math.isfinite(float(zp_sigma_mag)):
        _zs = float(zp_sigma_mag)
        if _zs > ZP_SIGMA_HARD_MAG:
            hard.append(f"zero point uncertain by {1000 * _zs:.0f} mmag (comparison stars disagree)")
        elif _zs > ZP_SIGMA_SOFT_MAG:
            soft.append(f"zero point uncertain by {1000 * _zs:.0f} mmag (comparison stars disagree)")

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
    elif lq and lq not in _LC_QUALITY_OK and lq != "-":
        hard.append(f"LC quality: {lq}")

    if n_chk == 0:
        hard.append("no check-star verification")
    elif 0 < n_chk < min_chk:
        soft.append(f"insufficient check-star verification (n={n_chk})")
    elif use_sparse_check_trust and sparse_trust_band:
        band = str(sparse_trust_band).strip().upper()
        flag_txt = "; ".join(sparse_trust_flags) if sparse_trust_flags else ""
        if "single_comp" in (sparse_trust_flags or ()):
            soft.append("sparse check: single_comp (no ensemble)")
        elif band == "RED":
            hard.append(f"sparse check trust RED ({flag_txt or 'CI/stability'})")
        elif band == "YELLOW":
            soft.append(f"sparse check trust YELLOW ({flag_txt or 'marginal CI/stability'})")
    elif math.isfinite(check_chi2_nu):
        # check_chi2_nu carries the robust scatter/error ratio (see check_star_chi2)
        if check_chi2_nu > CHECK_RATIO_HARD:
            hard.append(
                f"check-star scatter {check_scatter:.3f} mag = {check_chi2_nu:.1f} x its errors"
            )
        elif check_chi2_nu > CHECK_RATIO_SOFT:
            soft.append(f"check-star scatter {check_scatter:.3f} mag = {check_chi2_nu:.1f} x its errors")
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
    hard: list[str],
    soft: list[str],
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
        return text + " - review before submitting"
    return text + " - inspect before submitting"


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
    n_stability_suspect: int = 0,
    n_tier12: int | None = None,
    color_term_off: bool = False,
    n_alignment_failed: int = 0,
    n_wcs_untrusted: int = 0,
    sparse_trust: dict[str, Any] | None = None,
    cfg: Any | None = None,
    check_chi2_nu: float = float("nan"),
    n_export: int | None = None,
    blend_frac: float = float("nan"),
    skip_reason: str = "",
    check_delta_mag: float = float("nan"),
    zp_sigma_mag: float = float("nan"),
) -> dict[str, Any]:
    th = thresholds or CompTrustThresholds.from_bounds(3, 8)
    lq = str(lc_quality or "").strip().lower() or "-"
    sparse_band: str | None = None
    sparse_flags: tuple[str, ...] = ()
    use_sparse = False
    if sparse_trust and bool(sparse_trust.get("check_sparse")):
        from sparse_trust_core import SparseTrustConfig, trust_band  # noqa: PLC0415

        use_sparse = True
        sparse_band, sparse_flags = trust_band(
            R_hi=float(sparse_trust.get("trust_R_hi", float("nan"))),
            R_lo=float(sparse_trust.get("trust_R_lo", float("nan"))),
            stability_p=float(sparse_trust.get("comp_stability_p", float("nan"))),
            x2_pair_mag2=float(sparse_trust.get("x2_pair_mag2", float("nan"))),
            n_comps=int(sparse_trust.get("n_comps", 2) or 2),
            triangulation_clipped=bool(sparse_trust.get("triangulation_clipped")),
            cfg=SparseTrustConfig(),
        )
    hard, soft = classify_warnings(
        check_chi2_nu=float(check_chi2_nu),
        n_clean=int(n_clean),
        check_scatter=float(check_scatter),
        lc_quality=lq,
        thresholds=th,
        n_frames=n_frames,
        n_check=int(n_check),
        check_min_epochs=int(check_min_epochs),
        n_stability_suspect=int(n_stability_suspect),
        n_tier12=n_tier12,
        color_term_off=bool(color_term_off),
        n_alignment_failed=int(n_alignment_failed),
        n_wcs_untrusted=int(n_wcs_untrusted),
        sparse_trust_band=sparse_band,
        sparse_trust_flags=sparse_flags,
        use_sparse_check_trust=use_sparse,
        n_export=n_export,
        blend_frac=float(blend_frac),
        skip_reason=str(skip_reason or ""),
        check_delta_mag=float(check_delta_mag),
        zp_sigma_mag=float(zp_sigma_mag),
    )
    trust = trust_level(int(n_clean), hard, soft, th)
    reason = build_reason(
        trust, int(n_clean), lq, hard, soft
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
        "sparse_trust_band": sparse_band,
        "sparse_trust_flags": list(sparse_flags),
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
    # "T3+ comps with color term off" applies only to a standard filter, where a colour term could correct the
    # comps' colour mismatch. Clear / luminance / tri-colour are untransformed by design (BANDAUTH-2026-10); their
    # colour mismatch is already the "ensemble includes T3+ comps" reason (CFG-REVIEW2-2026-10).
    from gaia_bands import filter_band_for_obs_group  # noqa: PLC0415

    _obs_group = phot.parent.name if phot.parent else ""
    _color_term_off = False
    if filter_band_for_obs_group(_obs_group).transformable:
        _color_term_off = True
        if cfg is not None:
            try:
                from photometry_core import resolve_apply_color_term  # noqa: PLC0415

                _color_term_off = not resolve_apply_color_term(cfg, _obs_group)
            except Exception:  # noqa: BLE001
                _color_term_off = True
    comp_meta: dict[str, dict[str, int]] = {}
    comp_path = phot / "comparison_stars_per_target.csv"
    if comp_path.is_file():
        try:
            from comp_table import load_comp_table, used  # noqa: PLC0415

            comp_df = used(load_comp_table(comp_path))
            if "target_catalog_id" in comp_df.columns:
                for tid, grp in comp_df.groupby(comp_df["target_catalog_id"].astype(str).str.strip()):
                    if not tid:
                        continue
                    comp_meta[norm_id(tid) or tid] = {
                        "n_comps": int(grp["catalog_id"].astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
                    }
        except Exception:  # noqa: BLE001
            comp_meta = {}
    per_target: dict[str, dict[str, Any]] = {}
    conf_counts: dict[str, int] = {}

    for _, row in summ.iterrows():
        cid = norm_id(row.get("catalog_id"))
        if not cid:
            continue
        _cm0 = comp_meta.get(cid, {})
        if "n_comps" in _cm0:
            n_clean = int(_cm0["n_comps"])
        else:
            ng_raw = pd.to_numeric(row.get("n_good_comp"), errors="coerce")
            n_clean = int(ng_raw) if math.isfinite(float(ng_raw)) else 0
        lc_quality = str(row.get("lc_quality_flag", "") or "").strip()
        vsx = str(row.get("vsx_name", "") or "")
        nf_raw = pd.to_numeric(row.get("n_frames"), errors="coerce")
        n_frames = int(nf_raw) if math.isfinite(float(nf_raw)) else None
        chk, n_chk = check_star_scatter(phot, cid)
        _chi2_nu, _chi2_p = check_star_chi2(phot, cid)
        _sparse_side = read_sparse_trust_sidecar(phot, cid)
        if _sparse_side is not None:
            _sparse_side["n_comps"] = int(n_clean)
        _stab_sus_raw = pd.to_numeric(row.get("n_stability_suspect"), errors="coerce")
        _stab_sus = int(_stab_sus_raw) if math.isfinite(float(_stab_sus_raw)) else 0
        _nt12_raw = pd.to_numeric(row.get("n_tier12"), errors="coerce")
        _nt12 = int(_nt12_raw) if math.isfinite(float(_nt12_raw)) else None
        _naf_raw = pd.to_numeric(row.get("n_alignment_failed"), errors="coerce")
        _n_af = int(_naf_raw) if math.isfinite(float(_naf_raw)) else 0
        _nwut_raw = pd.to_numeric(row.get("n_wcs_untrusted"), errors="coerce")
        _n_wut = int(_nwut_raw) if math.isfinite(float(_nwut_raw)) else 0
        _skip = str(row.get("skip_reason", "") or "").strip()
        if _skip.lower() in ("nan", "none"):
            _skip = ""
        if _skip:
            # skipped in this run: a light curve on disk is from an earlier run (stale)
            _n_exp, _n_lc = None, None
        else:
            _n_exp, _n_lc = lc_epoch_counts(phot, cid)
        if _n_lc is not None and _n_lc > 0:
            n_frames = _n_lc
        _bl_raw = pd.to_numeric(row.get("target_blend_frac"), errors="coerce")
        _chk_delta, _zp_sigma = check_delta_and_zp_sigma(phot, cid)
        info = evaluate_target(
            skip_reason=_skip,
            n_export=_n_exp,
            blend_frac=float(_bl_raw) if math.isfinite(float(_bl_raw)) else float("nan"),
            catalog_id=cid,
            vsx_name=vsx,
            n_clean=n_clean,
            lc_quality=lc_quality,
            check_scatter=chk,
            thresholds=th,
            n_frames=n_frames,
            n_check=n_chk,
            check_min_epochs=min_chk,
            n_stability_suspect=_stab_sus,
            n_tier12=_nt12,
            color_term_off=_color_term_off,
            n_alignment_failed=_n_af,
            n_wcs_untrusted=_n_wut,
            sparse_trust=_sparse_side,
            cfg=cfg,
            check_chi2_nu=_chi2_nu,
            check_delta_mag=_chk_delta,
            zp_sigma_mag=_zp_sigma,
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
    update_summary: bool = True,
) -> list[Path]:
    """Write trust + trust_reason into ``photometry_summary.csv`` (the only trust output)."""
    phot = Path(photometry_dir)
    written: list[Path] = []

    trust_map = {
        tid: str(info.get("trust") or _UNEVALUATED_TRUST)
        for tid, info in result.get("per_target", {}).items()
    }
    reason_map = {
        tid: str(info.get("trust_reason") or _UNEVALUATED_REASON)
        for tid, info in result.get("per_target", {}).items()
    }

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
    update_summary: bool = True,
    cfg: Any | None = None,
    thresholds: CompTrustThresholds | None = None,
) -> dict[str, Any]:
    """Pipeline / CLI entry: trust columns in photometry_summary.csv."""
    th = thresholds or comp_thresholds_from_config(cfg)
    min_chk = CHECK_STAR_MIN_EPOCHS
    result = compute_trust_for_photometry_dir(
        photometry_dir, thresholds=th, check_min_epochs=min_chk, cfg=cfg
    )
    paths = write_trust_artifacts(
        result,
        photometry_dir=photometry_dir,
        update_summary=update_summary,
    )
    st = result.get("stats", {})
    LOGGER.info(
        "[TRUST] min/strong=%s/%s targets=%s trust=%s -> %d artifacts",
        st.get("min_comps"),
        st.get("strong_comps"),
        st.get("n_targets"),
        st.get("trust"),
        len(paths),
    )
    result["written_paths"] = [str(p) for p in paths]
    return result


