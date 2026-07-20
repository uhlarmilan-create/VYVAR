"""VYVAR-INVARIANTS P2 - runtime check helpers (check-only; no science mutation).

Records results under ``pipeline_meta["invariants"]``. FAIL policy raises
:class:`InvariantViolation`; WARN logs and records. See ``docs/VYVAR_INVARIANTS.md``.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

PROV_SCHEMA_VERSION = 1

# Minimal stage DAG (INV-DAG-01).
STAGE_ORDER: tuple[str, ...] = (
    "calibrate",
    "preprocess",
    "align",
    "masterstar",
    "perframe",
    "phase01",
    "phase2a",
    "postprocess",
)

# Wired invariant IDs (parity-tested against docs/VYVAR_INVARIANTS.md).
WIRED_INV_IDS: frozenset[str] = frozenset(
    {
        "INV-FLUX-01",
        "INV-FLUX-02",
        "INV-FLAT-01",
        "INV-WCS-01",
        "INV-DAG-01",
        "INV-RNG-01",
        "INV-PROV-01",
        "INV-CFG-01",
    }
)

FLATNESS_P99_WARN_ADU = 400.0
WCS_IDENTITY_P95_WARN_PX = 2.0
FLAT_MEAN_REL_TOL = 1e-3
FLUX_SUM_REL_TOL = 1e-6

COG_META_KEYS = (
    "cog_night_fallback",
    "cog_night_fallback_n_without_ok",
    "cog_night_fallback_n_frames",
)

PER_FRAME_SAT_META_KEYS = (
    "per_frame_sat_enabled",
    "per_frame_sat_min_clean_frac",
    "per_frame_sat_n_targets",
    "per_frame_sat_n_fallback",
    "per_frame_sat_n_rescued",
    "per_frame_sat_n_skipped",
)


class InvariantViolation(RuntimeError):
    """FAIL-CLOSED invariant breach."""

    def __init__(self, inv_id: str, detail: str) -> None:
        self.inv_id = str(inv_id)
        self.detail = str(detail)
        super().__init__(f"{self.inv_id}: {self.detail}")


def inv_check(
    meta: dict[str, Any],
    inv_id: str,
    ok: bool,
    *,
    policy: str,
    detail: str = "",
) -> None:
    """Append an invariant record; raise on FAIL+not-ok."""
    pol = str(policy or "").strip().upper()
    if pol in ("FAIL-CLOSED", "FAIL_CLOSED", "FAILCLOSED"):
        pol = "FAIL"
    if pol not in ("FAIL", "WARN"):
        pol = "WARN"
    rec = {
        "id": str(inv_id),
        "ok": bool(ok),
        "policy": pol,
        "detail": str(detail or ""),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    block = meta.setdefault("invariants", [])
    if not isinstance(block, list):
        meta["invariants"] = [rec]
    else:
        block.append(rec)
    if ok:
        return
    if pol == "WARN":
        LOGGER.warning("[INV] %s WARN: %s", inv_id, detail)
        return
    LOGGER.error("[INV] %s FAIL: %s", inv_id, detail)
    raise InvariantViolation(inv_id, detail)


def check_dark_resample_flux_conservation(
    src: np.ndarray,
    out: np.ndarray,
    *,
    block_factor: int,
    mode: str = "sum",
    rel_tol: float = FLUX_SUM_REL_TOL,
) -> tuple[bool, str]:
    """INV-FLUX-01: block-sum downscale (and uniform upscale) preserve SUM."""
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(out, dtype=np.float64)
    bf = int(block_factor)
    if bf <= 1:
        return True, "block_factor=1 (no resample)"
    if a.ndim != 2 or b.ndim != 2:
        return False, f"expected 2D arrays, got {a.shape} / {b.shape}"
    mode_l = str(mode).strip().lower()
    if mode_l == "sum":
        # Downscale: out is smaller; trim src to exact blocks.
        h = (a.shape[0] // bf) * bf
        w = (a.shape[1] // bf) * bf
        if h < bf or w < bf:
            return False, "source too small for block factor"
        a_trim = a[:h, :w]
        sum_a = float(np.nansum(a_trim))
        sum_b = float(np.nansum(b))
        if not (math.isfinite(sum_a) and math.isfinite(sum_b)):
            return False, "non-finite sums"
        denom = max(abs(sum_a), 1e-12)
        rel = abs(sum_b - sum_a) / denom
        ok = rel <= float(rel_tol)
        return ok, f"downscale sum rel_err={rel:.3e} (tol={rel_tol:g})"
    if mode_l == "upscale":
        # Uniform upscale: each src pixel -> bf x bf block of value/bf^2 (SUM-preserving).
        h, w = a.shape
        if b.shape != (h * bf, w * bf):
            return False, f"upscale shape {b.shape} != {(h * bf, w * bf)}"
        sum_a = float(np.nansum(a))
        sum_b = float(np.nansum(b))
        denom = max(abs(sum_a), 1e-12)
        rel = abs(sum_b - sum_a) / denom
        ok = rel <= float(rel_tol)
        return ok, f"upscale sum rel_err={rel:.3e} (tol={rel_tol:g})"
    return False, f"unknown mode {mode!r}"


def uniform_sum_preserving_upscale(src: np.ndarray, block_factor: int) -> np.ndarray:
    """Reference uniform upscale that preserves SUM (for INV-FLUX-01 tests)."""
    bf = int(block_factor)
    a = np.asarray(src, dtype=np.float64)
    cell = a / float(bf * bf)
    return np.repeat(np.repeat(cell, bf, axis=0), bf, axis=1)


def check_flat_mean_near_one(
    flat: np.ndarray,
    *,
    rel_tol: float = FLAT_MEAN_REL_TOL,
) -> tuple[bool, str]:
    """INV-FLUX-02: normalized master flat mean ~ 1.0."""
    m = np.asarray(flat, dtype=np.float64)
    mean_a = float(np.nanmean(m))
    if not math.isfinite(mean_a):
        return False, "non-finite mean"
    rel = abs(mean_a - 1.0)
    ok = rel <= float(rel_tol)
    return ok, f"mean={mean_a:.6f} |mean-1|={rel:.3e} (tol={rel_tol:g})"


def residual_large_scale_p99_adu(
    frame: np.ndarray,
    *,
    order: int = 2,
    subsample_step: int = 8,
) -> float:
    """Order-N polynomial refit on a processed frame; return p99 of |surface| [ADU]."""
    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2 or arr.size < 64:
        return float("nan")
    order_i = max(1, min(2, int(order)))
    h, w = arr.shape
    step = max(1, int(subsample_step))
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    z = arr[::step, ::step]
    finite = np.isfinite(z)
    if int(np.count_nonzero(finite)) < 20:
        return float("nan")
    med = float(np.nanmedian(z[finite]))
    z0 = z - med
    use = finite & np.isfinite(z0)
    x_fit = xx[use].astype(np.float64)
    y_fit = yy[use].astype(np.float64)
    z_fit = z0[use]
    cols = []
    for i in range(order_i + 1):
        for j in range(order_i + 1 - i):
            cols.append((x_fit**i) * (y_fit**j))
    coef, *_ = np.linalg.lstsq(np.column_stack(cols), z_fit, rcond=None)
    yy_f, xx_f = np.mgrid[0:h:step, 0:w:step]
    cols_f = []
    xf = xx_f.ravel().astype(np.float64)
    yf = yy_f.ravel().astype(np.float64)
    for i in range(order_i + 1):
        for j in range(order_i + 1 - i):
            cols_f.append((xf**i) * (yf**j))
    surf = (np.column_stack(cols_f) @ coef).reshape(yy_f.shape)
    return float(np.nanpercentile(np.abs(surf), 99))


def check_residual_flatness(
    frame: np.ndarray,
    *,
    p99_max_adu: float = FLATNESS_P99_WARN_ADU,
) -> tuple[bool, str, float]:
    """INV-FLAT-01: residual large-scale flatness (WARN band)."""
    p99 = residual_large_scale_p99_adu(frame)
    if not math.isfinite(p99):
        return True, "flatness n/a (insufficient samples)", float("nan")
    ok = p99 <= float(p99_max_adu)
    return ok, f"residual_flatness_p99={p99:.1f} ADU (band={p99_max_adu:g})", p99


def check_wcs_identity_p95(
    p95_px: float | None,
    *,
    warn_px: float = WCS_IDENTITY_P95_WARN_PX,
) -> tuple[bool, str]:
    """INV-WCS-01: identity p95 within WARN band (invertibility is INV-WCS-00 elsewhere)."""
    if p95_px is None:
        return True, "identity p95 n/a"
    try:
        p95 = float(p95_px)
    except (TypeError, ValueError):
        return True, "identity p95 n/a"
    if not math.isfinite(p95):
        return True, "identity p95 n/a"
    ok = p95 <= float(warn_px)
    return ok, f"matched_world2pix_identity_p95_px={p95:.3f} (warn<{warn_px:g})"


def stamp_pipeline_stage(
    meta: dict[str, Any],
    name: str,
    *,
    enforce_upstream: bool = True,
    head_inputs_present: bool = True,
) -> dict[str, Any]:
    """INV-DAG-01: append a stage stamp; optionally require the previous stage.

    Cold start (empty ``stages`` and ``name`` not first): allowed - stamps with
    ``cold_start=true`` so mid-pipeline entry points do not fail-closed.
    """
    name_s = str(name).strip()
    if name_s not in STAGE_ORDER:
        raise ValueError(f"unknown stage {name_s!r}; expected one of {STAGE_ORDER}")
    stages = meta.setdefault("stages", [])
    if not isinstance(stages, list):
        stages = []
        meta["stages"] = stages
    stamped = {str(s.get("name")) for s in stages if isinstance(s, dict)}
    idx = STAGE_ORDER.index(name_s)
    cold_start = False
    gap = False
    if enforce_upstream:
        seqs = [STAGE_ORDER.index(n) for n in stamped if n in STAGE_ORDER]
        max_seq = max(seqs) if seqs else -1
        if not stamped:
            if idx > 0:
                cold_start = True
        elif idx < max_seq:
            inv_check(
                meta,
                "INV-DAG-01",
                False,
                policy="FAIL",
                detail=f"stage {name_s!r} seq={idx} goes backwards (max stamped seq={max_seq})",
            )
        elif idx > max_seq + 1:
            # Sparse wiring (early stages not yet stamped) - allow with gap flag.
            gap = True
        # idx == max_seq or idx == max_seq+1: re-stamp or contiguous - ok
    rec = {
        "name": name_s,
        "seq": int(idx),
        "head_inputs_present": bool(head_inputs_present),
        "ts": datetime.now(timezone.utc).isoformat(),
        "cold_start": bool(cold_start),
        "gap": bool(gap),
    }
    stages.append(rec)
    detail = f"stamped {name_s} seq={idx}"
    if cold_start:
        detail += " (cold_start)"
    if gap:
        detail += " (gap)"
    inv_check(meta, "INV-DAG-01", True, policy="FAIL", detail=detail)
    return rec


def load_pipeline_meta(photometry_dir: Path | str) -> dict[str, Any]:
    path = Path(photometry_dir) / "pipeline_meta.json"
    if not path.is_file():
        return {}
    try:
        import json

        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def save_pipeline_meta(photometry_dir: Path | str, meta: dict[str, Any]) -> None:
    import json

    path = Path(photometry_dir) / "pipeline_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def stamp_stage_on_disk(
    photometry_dir: Path | str,
    name: str,
    *,
    enforce_upstream: bool = True,
) -> None:
    """Load/merge/save a stage stamp on ``pipeline_meta.json``."""
    meta = load_pipeline_meta(photometry_dir)
    stamp_pipeline_stage(meta, name, enforce_upstream=enforce_upstream)
    save_pipeline_meta(photometry_dir, meta)


def _cfg_flag(meta: dict[str, Any], key: str, default: bool = False) -> bool:
    prov = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
    snap = prov.get("config_snapshot") if isinstance(prov.get("config_snapshot"), dict) else {}
    if key in snap:
        return bool(snap[key])
    return bool(default)


def validate_config_behavior(meta: dict[str, Any], photometry_dir: Path | str | None) -> None:
    """INV-CFG-01: gating flag OFF => related markers absent."""
    issues: list[str] = []

    if not _cfg_flag(meta, "cog_aperture_correction_enabled", False):
        for k in COG_META_KEYS:
            if k in meta:
                issues.append(f"cog meta key present while COG disabled: {k}")

    if not _cfg_flag(meta, "per_frame_saturation_enabled", False):
        for k in PER_FRAME_SAT_META_KEYS:
            if k in meta:
                issues.append(
                    f"per-frame-sat meta key present while per_frame_saturation_enabled=False: {k}"
                )
        if photometry_dir is not None:
            import pandas as pd

            p = Path(photometry_dir) / "photometry_summary.csv"
            if p.is_file():
                try:
                    df = pd.read_csv(p, nrows=20, low_memory=False)
                except Exception:  # noqa: BLE001
                    df = None
                if df is not None:
                    for col in ("sat_clean_frac", "per_frame_sat_fallback"):
                        if col in df.columns:
                            issues.append(
                                f"{col} in photometry_summary.csv while "
                                "per_frame_saturation_enabled=False"
                            )
                    if "skip_reason" in df.columns:
                        vals = {
                            str(v).strip().lower() for v in df["skip_reason"].tolist()
                        }
                        if "per_frame_saturation" in vals:
                            issues.append(
                                "skip_reason=per_frame_saturation in "
                                "photometry_summary.csv while flag OFF"
                            )

    if not _cfg_flag(meta, "temporal_binning_enabled", False):
        if meta.get("temporal_binning_applied") is True:
            issues.append("temporal_binning_applied=true while temporal_binning_enabled=False")

    if not _cfg_flag(meta, "psf_photometry_enabled", False) and photometry_dir is not None:
        lc_dir = Path(photometry_dir) / "lightcurves"
        if lc_dir.is_dir():
            import pandas as pd

            for p in sorted(lc_dir.glob("lightcurve_*.csv"))[:80]:
                try:
                    df = pd.read_csv(p, nrows=5, low_memory=False)
                except Exception:  # noqa: BLE001
                    continue
                for col in ("err_method", "method", "lc_flux_method"):
                    if col not in df.columns:
                        continue
                    vals = {str(v).strip().lower() for v in df[col].tolist()}
                    if "psf" in vals:
                        issues.append(f"psf method rows in {p.name} while psf_photometry_enabled=False")
                        break
                if issues and issues[-1].startswith("psf method"):
                    break

    # Empty vsx_out_of_scope_types => no out-of-scope skip markers.
    _cfg_snap = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    _voos = meta.get("vsx_out_of_scope_types")
    if _voos is None:
        _voos = _cfg_snap.get("vsx_out_of_scope_types", [])
    if isinstance(_voos, str):
        _voos_list = [p.strip() for p in _voos.split(",") if p.strip()]
    elif isinstance(_voos, (list, tuple)):
        _voos_list = [str(p).strip() for p in _voos if str(p).strip()]
    else:
        _voos_list = []
    if not _voos_list and photometry_dir is not None:
        import pandas as pd

        for rel in (
            Path(photometry_dir) / "photometry_summary.csv",
            Path(photometry_dir).parent / "active_targets.csv",
        ):
            if not rel.is_file():
                continue
            try:
                df = pd.read_csv(rel, low_memory=False)
            except Exception:  # noqa: BLE001
                continue
            if "skip_reason" not in df.columns:
                continue
            vals = {str(v).strip().lower() for v in df["skip_reason"].tolist()}
            if "vsx_type_out_of_scope" in vals:
                issues.append(
                    f"skip_reason=vsx_type_out_of_scope in {rel.name} while "
                    "vsx_out_of_scope_types=[]"
                )

    ok = not issues
    inv_check(
        meta,
        "INV-CFG-01",
        ok,
        policy="FAIL",
        detail="; ".join(issues) if issues else "config<->behavior markers clean",
    )


def validate_provenance_schema(
    meta: dict[str, Any],
    *,
    photometry_dir: Path | str | None = None,
) -> None:
    """INV-PROV-01 (+ CFG-01) end-of-run provenance schema gate."""
    meta["prov_schema_version"] = int(PROV_SCHEMA_VERSION)
    issues: list[str] = []

    prov = meta.get("provenance")
    if not isinstance(prov, dict):
        issues.append("missing provenance block")
    else:
        for k in ("git_hash", "config_snapshot", "entry_point"):
            if k not in prov:
                issues.append(f"provenance missing {k}")
        if "labbe_rng_seed_policy" not in prov:
            issues.append("provenance missing labbe_rng_seed_policy")

    if "invariants" not in meta or not isinstance(meta.get("invariants"), list):
        issues.append("missing invariants list")

    # sky_stats / sky_surface summary when preprocess applied
    n_applied = meta.get("sky_surface_n_applied")
    try:
        n_app_i = int(n_applied) if n_applied is not None else 0
    except (TypeError, ValueError):
        n_app_i = 0
    if n_app_i > 0:
        if meta.get("sky_surface_order") is None:
            issues.append("sky_surface applied but sky_surface_order missing")

    # COG keys present iff enabled
    cog_on = _cfg_flag(meta, "cog_aperture_correction_enabled", False)
    if cog_on:
        if "cog_night_fallback" not in meta:
            issues.append("COG enabled but cog_night_fallback missing")
    else:
        for k in COG_META_KEYS:
            if k in meta:
                issues.append(f"COG disabled but {k} present")

    # Census fingerprint keys (best-effort when masterstar stamped)
    stages = meta.get("stages") if isinstance(meta.get("stages"), list) else []
    stage_names = {str(s.get("name")) for s in stages if isinstance(s, dict)}
    if "masterstar" in stage_names:
        if meta.get("n_gaia_matched") is None and meta.get("n_raw_dao") is None:
            if meta.get("catalog_rows") is None and meta.get("n_gaia_detected") is None:
                issues.append("masterstar stage without census fingerprint keys")

    ok = not issues
    inv_check(
        meta,
        "INV-PROV-01",
        ok,
        policy="FAIL",
        detail="; ".join(issues) if issues else f"prov_schema_version={PROV_SCHEMA_VERSION} ok",
    )
    validate_config_behavior(meta, photometry_dir)


def run_end_of_run_invariants(
    photometry_dir: Path | str,
    *,
    stamp_postprocess: bool = True,
) -> dict[str, Any]:
    """Load meta, stamp postprocess, run PROV/CFG validation, save."""
    meta = load_pipeline_meta(photometry_dir)
    if stamp_postprocess:
        try:
            stamp_pipeline_stage(meta, "postprocess", enforce_upstream=True)
        except InvariantViolation:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[INV] postprocess stamp skipped: %s", exc)
    validate_provenance_schema(meta, photometry_dir=photometry_dir)
    save_pipeline_meta(photometry_dir, meta)
    return meta


# Touch time module so unused import not stripped when only using datetime.
_ = time
