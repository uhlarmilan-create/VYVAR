"""VYVAR-INVARIANTS P2 - runtime check helpers (check-only; no science mutation).

Records results under ``pipeline_meta["invariants"]``. FAIL policy raises
:class:`InvariantViolation`; WARN logs and records. See ``docs/VYVAR_INVARIANTS.md``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
        "INV-ERR-SIGMA-ACCT-01",
        "INV-PSF-FRAME-01",
        "INV-PSF-ADDITIVE-01",
        "INV-EPSF-BUILD-GUARD-01",
        "INV-RNG-01",
        "INV-PROV-01",
        "INV-CFG-01",
        "INV-CFG-01R",
        "INV-PHASE0-ID",
        "INV-PREP-01",
        "INV-SAT-01",
        "INV-CAL-01",
        "INV-CAL-02",
        "INV-COMP-MEMBERSHIP",
        "INV-PIN-01",
        "INV-PIN-02",
        "INV-PIN-03",
        "INV-PIN-04",
        "INV-MASTER-01",
        "INV-MS-CENSUS-01",
        "INV-NOCLIP-01",
        "INV-NOCOSMIC-01",
        "INV-PIXELS-01",
        "QC-01",
        "OSC-01",
        "OSC-02",
        "OSC-03",
    }
)

FLATNESS_P99_WARN_ADU = 400.0
WCS_IDENTITY_P95_WARN_PX = 2.0
FLAT_MEAN_REL_TOL = 1e-3
FLUX_SUM_REL_TOL = 1e-6
PREPROCESS_LARGE_SMALL_RATIO_WARN = 10.0

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


class PopulationEmptiedError(InvariantViolation):
    """INV-NO-SILENT-EMPTY: a filter removed every row of a non-empty population."""

    def __init__(
        self,
        *,
        rule_id: str,
        threshold: object,
        unit: str,
        population: str,
        n_in: int,
    ) -> None:
        self.rule_id = str(rule_id)
        self.threshold = threshold
        self.unit = str(unit)
        self.population = str(population)
        self.n_in = int(n_in)
        detail = (
            f"rule_id={self.rule_id} emptied population={self.population!r} "
            f"n_in={self.n_in} n_out=0 threshold={self.threshold!r} unit={self.unit}"
        )
        super().__init__("INV-NO-SILENT-EMPTY", detail)


def assert_population_nonempty(
    *,
    n_in: int,
    n_out: int,
    rule_id: str,
    threshold: object,
    unit: str,
    population: str,
) -> None:
    """Raise when a gate empties a previously non-empty population (INV-NO-SILENT-EMPTY)."""
    if int(n_in) > 0 and int(n_out) == 0:
        raise PopulationEmptiedError(
            rule_id=str(rule_id),
            threshold=threshold,
            unit=str(unit),
            population=str(population),
            n_in=int(n_in),
        )


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
    """INV-FLUX-02: normalized master flat median ~ 1.0 (matches ``normalize_flat_master``)."""
    m = np.asarray(flat, dtype=np.float64)
    med_a = float(np.nanmedian(m))
    if not math.isfinite(med_a):
        return False, "non-finite median"
    rel = abs(med_a - 1.0)
    ok = rel <= float(rel_tol)
    mean_a = float(np.nanmean(m))
    return ok, f"median={med_a:.6f} |median-1|={rel:.3e} mean={mean_a:.6f} (tol={rel_tol:g})"


def read_qc_metrics_status_by_path(qc_csv: Path | str) -> dict[str, str]:
    """Map normalized absolute FITS path -> status (for QC-01 / allowlist checks)."""
    import pandas as pd

    from pipeline import norm_fits_path_key  # noqa: PLC0415

    p = Path(qc_csv)
    if not p.is_file():
        raise FileNotFoundError(f"qc_metrics.csv not found: {p}")
    df = pd.read_csv(p)
    if "status" not in df.columns:
        raise ValueError(f"qc_metrics.csv missing status column: {p}")
    src_col = "src" if "src" in df.columns else "dst"
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        raw = row.get(src_col)
        if raw is None or (isinstance(raw, float) and not math.isfinite(float(raw))):
            continue
        out[norm_fits_path_key(str(raw))] = str(row["status"]).strip()
    return out


def check_qc01_skipproc_alignment(
    selected_files: Sequence[Path | str],
    qc_csv: Path | str,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """QC-01: every aligned frame must appear in qc_metrics.csv with status=ok."""
    from pipeline import norm_fits_path_key  # noqa: PLC0415

    meta_block = meta if meta is not None else {"invariants": []}
    status_map = read_qc_metrics_status_by_path(qc_csv)
    ok_keys = {k for k, v in status_map.items() if v == "ok"}
    bad: list[str] = []
    for raw in selected_files:
        fp = Path(raw)
        key = norm_fits_path_key(fp)
        st = status_map.get(key)
        if st != "ok":
            bad.append(f"{fp.name}:{st or 'missing'}")
    n_sel = len(list(selected_files))
    n_ok_match = n_sel - len(bad)
    ok = len(bad) == 0 and all(norm_fits_path_key(f) in ok_keys for f in selected_files)
    detail = f"n_selected={n_sel} n_ok_matched={n_ok_match} violations={bad[:8]}"
    inv_check(meta_block, "QC-01", ok, policy="FAIL", detail=detail)


def check_osc01_channel_extraction_required(
    selected_files: Sequence[Path | str],
    *,
    equipment_bayermask: str | None,
    meta: dict[str, Any] | None = None,
) -> None:
    """OSC-01: OSC equipment frames must carry VY_CHANNEL (no raw mosaic in alignment)."""
    from astropy.io import fits

    from osc_extract import is_osc_bayermask, valid_bayer_pattern_4

    meta_block = meta if meta is not None else {"invariants": []}
    if not is_osc_bayermask(equipment_bayermask):
        inv_check(
            meta_block,
            "OSC-01",
            True,
            policy="FAIL",
            detail="mono equipment (OSC-01 N/A)",
        )
        return
    bad: list[str] = []
    for raw in selected_files:
        fp = Path(raw)
        try:
            with fits.open(fp, memmap=False) as hdul:
                hdr = hdul[0].header
                if hdr.get("VY_CHANNEL"):
                    continue
                if valid_bayer_pattern_4(str(hdr.get("BAYERPAT") or "")):
                    bad.append(fp.name)
        except OSError:
            bad.append(fp.name)
    ok = len(bad) == 0
    detail = f"equipment_bayermask={equipment_bayermask} mosaic_without_VY_CHANNEL={bad[:8]}"
    inv_check(meta_block, "OSC-01", ok, policy="FAIL", detail=detail)


def check_osc02_unified_frame_sets(
    osc_bundles: Mapping[str, Mapping[str, Any]],
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """OSC-02: the four channel groups of one OSC draft share identical frame ID sets."""
    from osc_align import unified_allowlist_frame_ids

    meta_block = meta if meta is not None else {"invariants": []}
    if not osc_bundles:
        inv_check(meta_block, "OSC-02", True, policy="FAIL", detail="mono path (OSC-02 N/A)")
        return
    violations: list[str] = []
    for base, bundle in osc_bundles.items():
        files_by_ch: dict[str, list[Path]] = {}
        for ch, job in bundle.items():
            files_by_ch[str(ch)] = [Path(f) for f in (job.get("files") or [])]
        ids = unified_allowlist_frame_ids(files_by_ch)
        for ch, paths in files_by_ch.items():
            stems = {p.name.casefold() for p in paths}
            if stems != ids:
                violations.append(f"{base}:{ch} n={len(stems)} expected={len(ids)}")
    ok = len(violations) == 0
    detail = f"n_bundles={len(osc_bundles)} violations={violations[:6]}"
    inv_check(meta_block, "OSC-02", ok, policy="FAIL", detail=detail)


def check_osc03_export_eligibility(
    obs_group: str,
    aavso_filter: str,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    """OSC-03: oneRGGB must never reach AAVSO/VarAstro exports; R/G/B must use TR/TG/TB."""
    from osc_align import is_onerggb_internal_obs_group, parse_osc_channel

    meta_block = meta if meta is not None else {"invariants": []}
    og = str(obs_group or "").strip()
    if not og:
        inv_check(meta_block, "OSC-03", True, policy="FAIL", detail="mono path (no obs_group)")
        return
    if is_onerggb_internal_obs_group(og):
        inv_check(
            meta_block,
            "OSC-03",
            False,
            policy="FAIL",
            detail=f"oneRGGB export blocked: {og}",
        )
        return
    ch = parse_osc_channel(og)
    if ch in ("R", "G", "B"):
        expected = {"R": "TR", "G": "TG", "B": "TB"}.get(ch, "")
        ok = str(aavso_filter or "").strip().upper() == expected
        inv_check(
            meta_block,
            "OSC-03",
            ok,
            policy="FAIL",
            detail=f"channel={ch} filt={aavso_filter!r} expected={expected!r}",
        )
        return
    inv_check(meta_block, "OSC-03", True, policy="FAIL", detail=f"mono/non-OSC obs_group={og}")


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


def preprocess_large_small_ratio(
    frame: np.ndarray,
    *,
    sigma: float = 30.0,
) -> float:
    """Ratio of large-scale to small-scale variance (post-preprocess gradient guard)."""
    from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

    arr = np.asarray(frame, dtype=np.float64)
    if arr.ndim != 2 or arr.size < 64:
        return float("nan")
    finite = np.isfinite(arr)
    if not int(np.count_nonzero(finite)):
        return float("nan")
    fill = float(np.nanmedian(arr[finite]))
    work = np.where(finite, arr, fill)
    blur = gaussian_filter(work, sigma=float(sigma))
    resid = work - blur
    var_large = float(np.var(blur))
    var_small = float(np.var(resid))
    if var_small <= 0.0 or not math.isfinite(var_small):
        return 0.0
    return var_large / var_small


def check_preprocess_large_small_ratio(
    frame: np.ndarray,
    *,
    warn_ratio: float = PREPROCESS_LARGE_SMALL_RATIO_WARN,
) -> tuple[bool, str, float]:
    """INV-PREP-01: large-scale gradient residual vs anchor band (~1-5x good; >>10x regression)."""
    ratio = preprocess_large_small_ratio(frame)
    if not math.isfinite(ratio):
        return True, "large_small_ratio n/a (insufficient samples)", float("nan")
    ok = ratio <= float(warn_ratio)
    return ok, f"large_small_ratio={ratio:.2f}x (warn>{warn_ratio:g})", ratio


def dao_only_fraction_from_masterstars(df: Any) -> float:
    """Informational census: fraction of masterstar rows with empty catalog_id / DAO_ONLY.

    No runtime policy is attached. Used by the pipeline log line, threshold-sweep
    audit scripts, and anchor fixture regression tests.
    """
    import pandas as pd

    pdf = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    if pdf.empty:
        return 0.0
    if "source_type" in pdf.columns:
        st = pdf["source_type"].fillna("").astype(str).str.strip()
        n_dao = int((st == "DAO_ONLY").sum())
    else:
        cid = pdf.get("catalog_id", pd.Series([""] * len(pdf))).fillna("").astype(str).str.strip()
        n_dao = int((cid == "").sum())
    return float(n_dao) / float(len(pdf))


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
    # Authority is provenance.config_snapshot (same as _cfg_flag), not a
    # non-existent top-level meta["config"] key.
    prov = meta.get("provenance") if isinstance(meta.get("provenance"), dict) else {}
    snap = prov.get("config_snapshot") if isinstance(prov.get("config_snapshot"), dict) else {}
    _cfg_legacy = meta.get("config") if isinstance(meta.get("config"), dict) else {}
    _voos = meta.get("vsx_out_of_scope_types")
    if _voos is None:
        _voos = snap.get("vsx_out_of_scope_types", _cfg_legacy.get("vsx_out_of_scope_types", []))
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
            Path(photometry_dir) / "active_targets.csv",
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

    # INV-CFG-01 reverse: non-empty vsx_out_of_scope_types with matching VSX types must mask rows.
    if _voos_list and photometry_dir is not None:
        import pandas as pd

        at_path = Path(photometry_dir).parent / "active_targets.csv"
        vt_path = Path(photometry_dir).parent.parent / "variable_targets.csv"
        if at_path.is_file() and vt_path.is_file():
            try:
                at_df = pd.read_csv(at_path, low_memory=False)
                vt_df = pd.read_csv(vt_path, low_memory=False)
                from vsx_type_scope import is_vsx_auto_selected_target, vsx_type_is_out_of_scope  # noqa: PLC0415

                expect_mask = False
                for _, vrow in vt_df.iterrows():
                    if not is_vsx_auto_selected_target(vrow):
                        continue
                    if vsx_type_is_out_of_scope(str(vrow.get("vsx_type", "") or ""), _voos_list):
                        expect_mask = True
                        break
                if expect_mask and "skip_reason" in at_df.columns:
                    masked = (at_df["skip_reason"].astype(str).str.strip() == "vsx_type_out_of_scope").sum()
                    if int(masked) <= 0:
                        inv_check(
                            meta,
                            "INV-CFG-01R",
                            False,
                            policy="WARN",
                            detail="vsx_out_of_scope_types non-empty but zero masked active targets",
                        )
            except Exception:  # noqa: BLE001
                pass

    # INV-PHASE0-ID: active catalog_id must match planner catalog_id for same vsx_name.
    if photometry_dir is not None:
        import pandas as pd

        at_path = Path(photometry_dir).parent / "active_targets.csv"
        vt_path = Path(photometry_dir).parent.parent / "variable_targets.csv"
        if at_path.is_file() and vt_path.is_file():
            try:
                from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

                at_df = pd.read_csv(at_path, low_memory=False)
                vt_df = pd.read_csv(vt_path, low_memory=False)
                if "vsx_name" in at_df.columns and "vsx_name" in vt_df.columns and "catalog_id" in at_df.columns:
                    vt_map = {
                        str(r["vsx_name"]).strip(): normalize_gaia_source_id(r.get("catalog_id"))
                        for _, r in vt_df.iterrows()
                        if str(r.get("vsx_name", "") or "").strip()
                    }
                    mism: list[str] = []
                    for _, ar in at_df.iterrows():
                        vn = str(ar.get("vsx_name", "") or "").strip()
                        if not vn or vn not in vt_map:
                            continue
                        plan = vt_map[vn]
                        act = normalize_gaia_source_id(ar.get("catalog_id"))
                        if plan and act and plan != act:
                            mism.append(vn)
                    inv_check(
                        meta,
                        "INV-PHASE0-ID",
                        len(mism) == 0,
                        policy="FAIL",
                        detail=f"catalog_id mismatch vs planner: {mism[:5]}" if mism else "identity join clean",
                    )
            except Exception:  # noqa: BLE001
                pass


def _draft_root_from_photometry_dir(photometry_dir: Path | str | None) -> Path | None:
    """Resolve draft archive root from a photometry or platesolve path."""
    if photometry_dir is None:
        return None
    p = Path(photometry_dir).resolve()
    for cand in (p, *p.parents):
        if cand.name.startswith("draft_") and (cand / "calibrated").is_dir():
            return cand
        if (cand / "sat_diag.json").is_file():
            return cand
    return None


def _load_sat_diag_block_from_disk(photometry_dir: Path | str | None) -> dict[str, Any] | None:
    root = _draft_root_from_photometry_dir(photometry_dir)
    if root is None:
        return None
    sd_path = root / "sat_diag.json"
    if not sd_path.is_file():
        return None
    try:
        loaded = json.loads(sd_path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _sample_cal_stage_from_disk(photometry_dir: Path | str | None) -> dict[str, Any] | None:
    """Return calibrated-stage evidence from ``cal_stage.json`` or FITS headers."""
    root = _draft_root_from_photometry_dir(photometry_dir)
    if root is None:
        return None
    cal_json = root / "cal_stage.json"
    if cal_json.is_file():
        try:
            loaded = json.loads(cal_json.read_text(encoding="ascii"))
            if isinstance(loaded, dict):
                return loaded
        except (OSError, json.JSONDecodeError, UnicodeError):
            pass

    cal_lights = root / "calibrated" / "lights"
    if not cal_lights.is_dir():
        return None
    try:
        from astropy.io import fits  # noqa: PLC0415

        from cal_stage import verify_fits_datasum  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None

    for fp in sorted(cal_lights.rglob("*.fits"))[:5]:
        try:
            with fits.open(fp, memmap=True) as hdul:
                hdr = hdul[0].header
                data = hdul[0].data
            stage = hdr.get("VY_CALSTAGE")
            if stage is None:
                continue
            ds = hdr.get("VY_CALDATASUM")
            ok_ds = verify_fits_datasum(np.asarray(data), str(ds) if ds is not None else None)
            return {
                "source": "calibrated_header",
                "vy_calstage": str(stage),
                "vy_caldatasum_ok": bool(ok_ds),
                "sample": fp.name,
            }
        except Exception:  # noqa: BLE001
            continue
    return None


def check_sat_diag(meta: dict[str, Any], *, photometry_dir: Path | str | None = None) -> None:
    """INV-SAT-01: SAT-DIAG limits and raw-peak provenance when gate ran."""
    block = meta.get("sat_diag")
    disk_block: dict[str, Any] | None = None
    if not isinstance(block, dict):
        disk_block = _load_sat_diag_block_from_disk(photometry_dir)
        if disk_block is not None:
            block = disk_block
    if not isinstance(block, dict):
        inv_check(
            meta,
            "INV-SAT-01",
            True,
            policy="WARN",
            detail="sat_diag not stamped (no raw lights or pre-SAT-DIAG draft)",
        )
        return
    src = str(block.get("sat_source") or "")
    sat_adu = block.get("sat_adu")
    ok = bool(src) and sat_adu is not None
    detail = f"sat_source={src!r} sat_adu={sat_adu}"
    if disk_block is not None and not isinstance(meta.get("sat_diag"), dict):
        detail = f"{detail} (loaded sat_diag.json from draft root; meta block missing)"
    inv_check(
        meta,
        "INV-SAT-01",
        ok,
        policy="FAIL",
        detail=detail,
    )
    lin_src = str((block or {}).get("lin_source") or "")
    if lin_src == "DEFAULT_FRAC" and bool((block or {}).get("tier3_exclusion_fired")):
        inv_check(
            meta,
            "INV-SAT-01",
            False,
            policy="FAIL",
            detail="Tier 3 DEFAULT_FRAC must not trigger exclusion",
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
    check_sat_diag(meta, photometry_dir=photometry_dir)
    check_cal_diag(meta, photometry_dir=photometry_dir)
    check_cal_stage(meta, photometry_dir=photometry_dir)


def check_cal_diag(meta: dict[str, Any], *, photometry_dir: Path | str | None = None) -> None:
    """INV-CAL-01: CAL-DIAG v2 provenance when dark calibration ran."""
    block = meta.get("cal_diag")
    cd_path_ok = False
    if photometry_dir is not None:
        for candidate in (
            Path(photometry_dir).resolve().parent.parent / "cal_diag.json",
            Path(photometry_dir).resolve().parent / "cal_diag.json",
        ):
            if candidate.is_file():
                cd_path_ok = True
                break
    cal_mode = str(meta.get("calibration_mode") or "")
    dark_applied = cal_mode not in ("", "PASSTHROUGH", "RAW")
    if not dark_applied and not isinstance(block, dict) and not cd_path_ok:
        inv_check(
            meta,
            "INV-CAL-01",
            True,
            policy="WARN",
            detail="cal_diag not stamped (no dark calibration or pre-CAL-DIAG draft)",
        )
        return
    if not isinstance(block, dict) and not cd_path_ok:
        inv_check(
            meta,
            "INV-CAL-01",
            False,
            policy="FAIL",
            detail="cal_diag block missing after dark calibration",
        )
        return
    keys = (block or {}).get("keys") if isinstance(block, dict) else None
    ok = bool(keys) or cd_path_ok
    spec_v = str((block or {}).get("spec_version") or "")
    inv_check(
        meta,
        "INV-CAL-01",
        ok,
        policy="FAIL",
        detail=f"cal_diag keys={len(keys or {})} spec_version={spec_v!r}",
    )


def check_cal_stage(meta: dict[str, Any], *, photometry_dir: Path | str | None = None) -> None:
    """INV-CAL-02: calibrated stage provenance when ``cal_stage.json`` is present."""
    block = meta.get("cal_stage")
    if not isinstance(block, dict):
        disk = _sample_cal_stage_from_disk(photometry_dir)
        if disk is None:
            inv_check(
                meta,
                "INV-CAL-02",
                True,
                policy="WARN",
                detail="cal_stage not stamped (legacy draft or pre-INV-CAL-02 run)",
            )
            return
        if disk.get("source") == "calibrated_header":
            ok = bool(disk.get("vy_calstage")) and bool(disk.get("vy_caldatasum_ok"))
            inv_check(
                meta,
                "INV-CAL-02",
                ok,
                policy="FAIL",
                detail=(
                    f"cal_stage meta missing; disk VY_CALSTAGE={disk.get('vy_calstage')!r} "
                    f"datasum_ok={disk.get('vy_caldatasum_ok')} sample={disk.get('sample')!r}"
                ),
            )
            return
        verify = disk.get("verify_last") if isinstance(disk.get("verify_last"), dict) else {}
        fail_n = int(verify.get("fail_stamp") or 0) + int(verify.get("fail_corrupt") or 0)
        inv_check(
            meta,
            "INV-CAL-02",
            fail_n == 0,
            policy="FAIL",
            detail=(
                f"cal_stage meta missing; loaded cal_stage.json verify pass={verify.get('pass')} "
                f"fail_stamp={verify.get('fail_stamp')} fail_corrupt={verify.get('fail_corrupt')}"
            ),
        )
        return
    verify = block.get("verify_last") if isinstance(block.get("verify_last"), dict) else {}
    fail_n = int(verify.get("fail_stamp") or 0) + int(verify.get("fail_corrupt") or 0)
    ok = fail_n == 0
    inv_check(
        meta,
        "INV-CAL-02",
        ok,
        policy="FAIL",
        detail=(
            f"cal_stage verify pass={verify.get('pass')} fail_stamp={verify.get('fail_stamp')} "
            f"fail_corrupt={verify.get('fail_corrupt')} indet_legacy={verify.get('indeterminate_legacy')} "
            f"indet_unknown={verify.get('indeterminate_unknown')}"
        ),
    )


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
