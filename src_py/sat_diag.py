"""SAT-DIAG: saturation and linearity limit gate (INV-SAT-01).

Camera-agnostic Check A (pile-up derivation), Check B (limit resolution),
placed-aperture raw peak measurement, and tier policy helpers.
See ``dev/results/specs/VYVAR_SAT_DIAG_SPEC.md``.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

LOGGER = logging.getLogger(__name__)

# Implementation constants (spec section 5.2 / 8 -- not user config).
N_PILEUP_MIN = 100
PILEUP_RATIO = 10.0
SAT_DIAG_MAX_FRAMES = 30
LINEARITY_DEFAULT_FRAC = 0.85
SATURATE_LIMIT_FRACTION = 0.85
ADMISSION_SAT_FRAC = 0.70
RESCALED_MAX_FRAC = 0.85
DRAFT_SAT_EXCLUDE_FRAME_FRAC = 0.50

# Placed-aperture peak (7x7 footprint, half=3 -- matches pipeline / DAOPHOT).
PEAK_BOX_HALF = 3
PEAK_DRIFT_SEARCH_HALF = 22  # mag-guided on drift references only (not targets/comps)

# Drift reference selection (AstroImageJ rule: centroid only bright, reliable stars).
DRIFT_REF_MIN_PEAK_ADU = 8000.0
DRIFT_REF_FALLBACK_PEAK_ADU = 4000.0
DRIFT_REF_MIN_COUNT = 2
DRIFT_CENTROID_BOX_HALF = 5  # 11 px cutout (photutils default box_size)

SAT_PEAK_SOURCE_PLACED = "PLACED_APERTURE"

# Tier-1 sat sources that may exclude (spec section 9.1).
TIER1_SAT_SOURCES = frozenset(
    {"MEASURED", "HEADER", "EQUIPMENT", "DERIVED", "CONFLICT_DERIVED", "BITPIX"}
)
# Accepted-but-unverified (too-high ceiling not testable without ramp).
UNVERIFIED_SAT_SOURCES = frozenset({"DERIVED_NO_PILEUP", "LEGACY_ALIGNED"})


@dataclass
class FrameDriftResult:
    """Per-frame drift diagnostic (spec 8.2)."""

    dx: float
    dy: float
    n_refs: int
    residual_rms_px: float
    method: str


@dataclass
class StarPeakDraftStats:
    """Once-per-draft peak aggregate for one catalog star."""

    catalog_id: str
    n_frames: int = 0
    peak_max: float = float("nan")
    peak_median: float = float("nan")
    n_over_admission: int = 0
    n_saturated: int = 0
    admission_reject: bool = False


@dataclass
class PileupResult:
    pileup_detected: bool
    v_ceiling: float | None
    n_at_ceiling: int
    n_at_shoulder: int
    max_pixel: float
    frames_sampled: int
    bitpix_ceiling: float | None
    refused: bool = False
    refuse_reason: str | None = None


@dataclass
class SatDiagContext:
    """Resolved SAT-DIAG limits and provenance for one obs_group."""

    sat_adu: float | None
    lin_adu: float | None
    sat_source: str
    lin_source: str
    bitpix_ceiling: float | None = None
    derived_ceiling: float | None = None
    header_value: float | None = None
    equipment_value: float | None = None
    refuted_source: str | None = None
    refuted_value: float | None = None
    warnings: list[str] = field(default_factory=list)
    pileup: PileupResult | None = None
    xbinning: int | None = None
    ybinning: int | None = None
    raw_peaks_used: bool = False
    sat_peak_source: str = SAT_PEAK_SOURCE_PLACED
    lin_adu_native: float | None = None
    sat_adu_native: float | None = None
    last_frame_drift: FrameDriftResult | None = None
    star_peak_draft: dict[str, StarPeakDraftStats] = field(default_factory=dict)
    frame_drift_residuals_px: list[float] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.pileup is not None:
            d["pileup"] = asdict(self.pileup)
        if self.last_frame_drift is not None:
            d["last_frame_drift"] = asdict(self.last_frame_drift)
        d["star_peak_draft"] = {
            k: asdict(v) for k, v in self.star_peak_draft.items()
        }
        if self.lin_adu is not None and self.lin_adu_native is None:
            bf = max(int(self.xbinning or 1), 1)
            d["lin_adu_native"] = float(self.lin_adu) / float(bf)
        if self.sat_adu is not None and self.sat_adu_native is None:
            bf = max(int(self.xbinning or 1), 1)
            d["sat_adu_native"] = float(self.sat_adu) / float(bf)
        if self.frame_drift_residuals_px:
            rs = np.asarray(self.frame_drift_residuals_px, dtype=np.float64)
            rs = rs[np.isfinite(rs)]
            if rs.size:
                d["placement_residual_median_px"] = float(np.median(rs))
                d["placement_residual_p95_px"] = float(np.percentile(rs, 95))
        return d

    def may_exclude_saturation(self) -> bool:
        return str(self.sat_source) in TIER1_SAT_SOURCES

    def may_exclude_linearity(self) -> bool:
        return str(self.lin_source) == "MEASURED"

    def admission_threshold_adu(self) -> float | None:
        if self.sat_adu is None:
            return None
        return float(self.sat_adu) * SATURATE_LIMIT_FRACTION * (
            ADMISSION_SAT_FRAC / SATURATE_LIMIT_FRACTION
        )

    def likely_saturated_threshold_adu(self) -> float | None:
        if self.sat_adu is None:
            return None
        return float(self.sat_adu) * 0.85

    def saturate_limit_adu_85pct(self) -> float | None:
        if self.sat_adu is None:
            return None
        return float(self.sat_adu) * SATURATE_LIMIT_FRACTION


def image_adu_array(hdu: fits.PrimaryHDU | fits.ImageHDU) -> np.ndarray:
    """Image ADU per spec section 4.1 (stored 0..65535 for unsigned 16-bit)."""
    hdr = hdu.header
    d = np.asarray(hdu.data, dtype=np.float64)
    try:
        bitpix = int(hdr.get("BITPIX", 0))
        bzero = float(hdr.get("BZERO", 0.0))
        bscale = float(hdr.get("BSCALE", 1.0))
    except (TypeError, ValueError):
        return d
    if bitpix < 0:
        return d
    if bitpix == 16 and abs(bzero - 32768.0) < 1.0 and abs(bscale - 1.0) < 1e-9:
        return d
    return d * bscale + bzero


def bitpix_container_ceiling(hdr: fits.Header) -> float | None:
    from pipeline import _infer_sat_limit_from_bitpix  # noqa: PLC0415

    return _infer_sat_limit_from_bitpix(hdr)


def header_sat_value(hdr: fits.Header) -> float | None:
    from pipeline import _saturate_limit_adu_from_header  # noqa: PLC0415

    v = _saturate_limit_adu_from_header(hdr)
    if v is not None:
        return v
    for dk in ("DATAMAX", "MAXPIX"):
        if dk not in hdr:
            continue
        try:
            fv = float(hdr[dk])
            if math.isfinite(fv) and fv > 0:
                return fv
        except (TypeError, ValueError):
            continue
    return None


def sample_raw_light_paths(
    archive_root: Path,
    *,
    max_frames: int = SAT_DIAG_MAX_FRAMES,
) -> list[Path]:
    """Deterministic subsample of raw light FITS under a draft archive."""
    ap = Path(archive_root)
    for sub in ("Raw/lights", "raw/lights"):
        root = ap / sub
        if not root.is_dir():
            continue
        files = sorted(root.rglob("*.fits"))
        if not files:
            continue
        if len(files) <= max_frames:
            return files
        idx = np.linspace(0, len(files) - 1, max_frames, dtype=int)
        return [files[int(i)] for i in idx]
    return []


def derive_ceiling_from_paths(paths: Sequence[Path]) -> PileupResult:
    """Check A: pile-up detection from raw frame histograms."""
    if not paths:
        return PileupResult(False, None, 0, 0, float("nan"), 0, None)

    counts: Counter[int] = Counter()
    max_px = -math.inf
    bitpix_ceil: float | None = None
    refused = False
    refuse_reason: str | None = None

    for fp in paths:
        with fits.open(fp, memmap=False) as hdul:
            hdr = hdul[0].header
            if int(hdr.get("BITPIX", 0)) < 0:
                refused = True
                refuse_reason = "REFUSE_NON_RAW"
                break
            if bitpix_ceil is None:
                bitpix_ceil = bitpix_container_ceiling(hdr)
            arr = image_adu_array(hdul[0])
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            max_px = max(max_px, float(np.max(finite)))
            vals, cnts = np.unique(finite.astype(np.int64), return_counts=True)
            for v, c in zip(vals, cnts, strict=False):
                counts[int(v)] += int(c)

    if refused:
        return PileupResult(
            False, None, 0, 0, max_px, len(paths), bitpix_ceil,
            refused=True, refuse_reason=refuse_reason,
        )
    if not counts:
        return PileupResult(False, None, 0, 0, max_px, len(paths), bitpix_ceil)

    v_max = max(counts)
    n_max = counts[v_max]
    lower_vals = [v for v in counts if v < v_max]
    n_shoulder = counts[max(lower_vals)] if lower_vals else 0
    pileup = False
    if n_max >= N_PILEUP_MIN and bitpix_ceil is not None and v_max >= int(bitpix_ceil) - 1:
        if n_shoulder <= 0:
            pileup = True
        elif n_max >= PILEUP_RATIO * n_shoulder:
            pileup = True
    if pileup:
        return PileupResult(True, float(v_max), n_max, n_shoulder, max_px, len(paths), bitpix_ceil)
    return PileupResult(False, None, n_max, n_shoulder, max_px, len(paths), bitpix_ceil)


def _compatible_stated(stated: float | None, max_pixel: float) -> bool:
    if stated is None or not math.isfinite(stated) or stated <= 0:
        return False
    return float(stated) >= float(max_pixel)


def resolve_sat_limit(
    *,
    hdr: fits.Header,
    pileup: PileupResult,
    equipment_adu: float | None,
) -> SatDiagContext:
    """Check B: header -> equipment -> derived -> BITPIX."""
    warnings: list[str] = []
    if pileup.refused:
        warnings.append(f"SAT-DIAG refused: {pileup.refuse_reason}")
        bitpix = pileup.bitpix_ceiling or bitpix_container_ceiling(hdr)
        return SatDiagContext(
            sat_adu=bitpix,
            lin_adu=(float(bitpix) * LINEARITY_DEFAULT_FRAC) if bitpix else None,
            sat_source=str(pileup.refuse_reason or "REFUSE_NON_RAW"),
            lin_source="DEFAULT_FRAC",
            bitpix_ceiling=bitpix,
            warnings=warnings,
            pileup=pileup,
        )

    header_val = header_sat_value(hdr)
    derived = pileup.v_ceiling if pileup.pileup_detected else None
    bitpix = pileup.bitpix_ceiling or bitpix_container_ceiling(hdr)
    max_px = pileup.max_pixel

    refuted_source: str | None = None
    refuted_value: float | None = None
    stated_before_refute: float | None = None

    def try_stated(value: float | None, source: str) -> tuple[float | None, str | None]:
        nonlocal refuted_source, refuted_value, stated_before_refute
        if value is None:
            return None, None
        if _compatible_stated(value, max_px):
            return float(value), source
        refuted_source = source
        refuted_value = float(value)
        if stated_before_refute is None:
            stated_before_refute = float(value)
        return None, None

    win_val: float | None = None
    prov = "none"

    v, src = try_stated(header_val, "HEADER")
    if v is not None:
        win_val, prov = v, src

    if win_val is None:
        v, src = try_stated(equipment_adu, "EQUIPMENT")
        if v is not None:
            win_val, prov = v, src

    if (
        bitpix is not None
        and math.isfinite(max_px)
        and max_px < float(bitpix) * RESCALED_MAX_FRAC
        and stated_before_refute is not None
        and stated_before_refute >= max_px
        and stated_before_refute < float(bitpix) * 0.95
        and not pileup.pileup_detected
    ):
        warnings.append("POSSIBLE_RESCALED_STACK")
        win_val = None
        prov = "none"

    if win_val is None and derived is not None:
        if refuted_source is not None:
            win_val, prov = float(derived), "CONFLICT_DERIVED"
            warnings.append(
                f"CONFLICT: refuted {refuted_source}={refuted_value}; using derived {derived}"
            )
        else:
            win_val, prov = float(derived), "DERIVED"

    if win_val is None and bitpix is not None:
        if pileup.pileup_detected and derived is not None:
            win_val, prov = float(derived), "CONFLICT_DERIVED" if refuted_source else "DERIVED"
        else:
            win_val, prov = float(bitpix), "DERIVED_NO_PILEUP"
            warnings.append("No saturation pile-up detected; ceiling set to BITPIX maximum.")

    lin_adu: float | None = None
    lin_source = "DEFAULT_FRAC"
    if win_val is not None:
        lin_adu = float(win_val) * LINEARITY_DEFAULT_FRAC
        warnings.append("Linearity level is a default fraction, not a measured knee.")

    try:
        xbin = int(hdr.get("XBINNING", 1))
        ybin = int(hdr.get("YBINNING", 1))
    except (TypeError, ValueError):
        xbin, ybin = 1, 1

    if refuted_source and prov not in ("CONFLICT_DERIVED",):
        prov = "CONFLICT_DERIVED"

    return SatDiagContext(
        sat_adu=win_val,
        lin_adu=lin_adu,
        sat_source=prov,
        lin_source=lin_source,
        bitpix_ceiling=bitpix,
        derived_ceiling=derived,
        header_value=header_val,
        equipment_value=equipment_adu,
        refuted_source=refuted_source,
        refuted_value=refuted_value,
        warnings=warnings,
        pileup=pileup,
        xbinning=xbin,
        ybinning=ybin,
    )


def expected_raw_from_aligned_centroid(
    aligned_x: float,
    aligned_y: float,
    ra_deg: float,
    dec_deg: float,
    aligned_hdr: fits.Header,
    raw_hdr: fits.Header,
) -> tuple[float, float] | None:
    """Map aligned DAO centroid to raw pixel via WCS residual (master-grid lock)."""
    try:
        wcs_a = WCS(aligned_hdr)
        wcs_r = WCS(raw_hdr)
        if not (wcs_a.has_celestial and wcs_r.has_celestial):
            return None
        with np.errstate(all="ignore"):
            axw, ayw = wcs_a.all_world2pix(float(ra_deg), float(dec_deg), 0)
            rxw, ryw = wcs_r.all_world2pix(float(ra_deg), float(dec_deg), 0)
        if not all(math.isfinite(v) for v in (axw, ayw, rxw, ryw)):
            return None
        return float(rxw + (float(aligned_x) - axw)), float(ryw + (float(aligned_y) - ayw))
    except Exception:  # noqa: BLE001
        return None


def _centroid_com_cutout(arr: np.ndarray, x0: float, y0: float, half: int) -> tuple[float, float] | None:
    """Centre-of-mass centroid in a small cutout (photutils-style, no max search)."""
    h, w = arr.shape
    xi = int(round(x0))
    yi = int(round(y0))
    x0b, x1b = max(0, xi - half), min(w, xi + half + 1)
    y0b, y1b = max(0, yi - half), min(h, yi + half + 1)
    if x0b >= x1b or y0b >= y1b:
        return None
    sub = np.asarray(arr[y0b:y1b, x0b:x1b], dtype=np.float64)
    if sub.size == 0:
        return None
    bg = float(np.percentile(sub, 20))
    wt = sub - bg
    wt = np.where(wt > 0, wt, 0.0)
    s = float(wt.sum())
    if s <= 0:
        return None
    yy, xx = np.indices(sub.shape)
    cx = float(x0b) + float((wt * xx).sum() / s)
    cy = float(y0b) + float((wt * yy).sum() / s)
    return cx, cy


def _select_drift_ref_indices(aligned_peak: np.ndarray) -> np.ndarray:
    """Bright stars eligible for drift measurement (AIJ: centroid only when reliable)."""
    pk = np.asarray(aligned_peak, dtype=np.float64)
    ok = np.isfinite(pk)
    primary = ok & (pk >= DRIFT_REF_MIN_PEAK_ADU)
    if int(primary.sum()) >= DRIFT_REF_MIN_COUNT:
        return np.nonzero(primary)[0]
    fallback = ok & (pk >= DRIFT_REF_FALLBACK_PEAK_ADU)
    return np.nonzero(fallback)[0]


def compute_frame_drift(
    arr: np.ndarray,
    raw_hdr: fits.Header,
    *,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    aligned_peak: np.ndarray,
    placed_x: np.ndarray,
    placed_y: np.ndarray,
    wcs_x: np.ndarray,
    wcs_y: np.ndarray,
) -> FrameDriftResult:
    """Frame drift diagnostic: offset of placed positions from WCS on drift refs."""
    ref_idx = _select_drift_ref_indices(aligned_peak)
    n_refs = int(ref_idx.size)
    if n_refs == 0:
        return FrameDriftResult(0.0, 0.0, 0, float("nan"), "none")

    dx_arr = placed_x[ref_idx] - wcs_x[ref_idx]
    dy_arr = placed_y[ref_idx] - wcs_y[ref_idx]
    ok = np.isfinite(dx_arr) & np.isfinite(dy_arr)
    if not ok.any():
        return FrameDriftResult(0.0, 0.0, 0, float("nan"), "none")

    dx = float(np.median(dx_arr[ok]))
    dy = float(np.median(dy_arr[ok]))
    resid = np.sqrt((dx_arr[ok] - dx) ** 2 + (dy_arr[ok] - dy) ** 2)
    rms = float(np.sqrt(np.mean(resid ** 2))) if resid.size else float("nan")
    method = "aligned_wcs_residual" if n_refs >= DRIFT_REF_MIN_COUNT else "aligned_wcs_residual_sparse"
    return FrameDriftResult(dx, dy, int(ok.sum()), rms, method)


def compute_frame_drift_from_centroids(
    arr: np.ndarray,
    raw_hdr: fits.Header,
    *,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    aligned_peak: np.ndarray,
) -> FrameDriftResult:
    """Fallback drift: COM centroid on bright refs at WCS position (no max search)."""
    try:
        wcs = WCS(raw_hdr)
        if not wcs.has_celestial:
            return FrameDriftResult(0.0, 0.0, 0, float("nan"), "wcs_missing")
        wx, wy = wcs.all_world2pix(ra_deg, dec_deg, 0)
    except Exception:  # noqa: BLE001
        return FrameDriftResult(0.0, 0.0, 0, float("nan"), "wcs_error")

    ref_idx = _select_drift_ref_indices(aligned_peak)
    dxs: list[float] = []
    dys: list[float] = []
    for i in ref_idx:
        if not (math.isfinite(float(wx[i])) and math.isfinite(float(wy[i]))):
            continue
        hit = _centroid_com_cutout(arr, float(wx[i]), float(wy[i]), DRIFT_CENTROID_BOX_HALF)
        if hit is None:
            continue
        cx, cy = hit
        dxs.append(float(cx - wx[i]))
        dys.append(float(cy - wy[i]))

    if len(dxs) < DRIFT_REF_MIN_COUNT:
        return FrameDriftResult(0.0, 0.0, len(dxs), float("nan"), "insufficient_refs")

    dx = float(np.median(np.asarray(dxs)))
    dy = float(np.median(np.asarray(dys)))
    resid = np.sqrt((np.asarray(dxs) - dx) ** 2 + (np.asarray(dys) - dy) ** 2)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return FrameDriftResult(dx, dy, len(dxs), rms, "com_centroid_refs")


def mag_guided_centroid(
    arr: np.ndarray, x0: float, y0: float, half: int = PEAK_DRIFT_SEARCH_HALF
) -> tuple[int, int]:
    """Brightest pixel in search window -- drift reference stars only."""
    h, w = arr.shape
    xi = int(round(x0))
    yi = int(round(y0))
    x0b, x1b = max(0, xi - half), min(w, xi + half + 1)
    y0b, y1b = max(0, yi - half), min(h, yi + half + 1)
    if x0b >= x1b or y0b >= y1b:
        return xi, yi
    sub = arr[y0b:y1b, x0b:x1b]
    flat_idx = int(np.argmax(sub))
    sy, sx = np.unravel_index(flat_idx, sub.shape)
    return int(x0b + sx), int(y0b + sy)


def _drift_offset_at_wcs(arr: np.ndarray, wx: float, wy: float, *, use_mag_guided: bool) -> tuple[float, float] | None:
    """Offset of true centroid from WCS prediction for one drift reference."""
    if not (math.isfinite(wx) and math.isfinite(wy)):
        return None
    if use_mag_guided:
        gx, gy = mag_guided_centroid(arr, wx, wy)
        return float(gx - wx), float(gy - wy)
    hit = _centroid_com_cutout(arr, wx, wy, DRIFT_CENTROID_BOX_HALF)
    if hit is None:
        return None
    return float(hit[0] - wx), float(hit[1] - wy)


def box_peak_max(arr: np.ndarray, x: float, y: float, half: int = PEAK_BOX_HALF) -> float:
    from pipeline import _box_peak_max_adu  # noqa: PLC0415

    return _box_peak_max_adu(arr, x, y, half=half)


def measure_raw_peaks_frame(
    raw_arr: np.ndarray,
    raw_hdr: fits.Header,
    *,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    aligned_x: np.ndarray | None = None,
    aligned_y: np.ndarray | None = None,
    aligned_hdr: fits.Header | None = None,
    aligned_peak: np.ndarray | None = None,
    drift_ref_ra: float | None = None,
    drift_ref_dec: float | None = None,
    drift_ref_catalog_id: str | None = None,
    catalog_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FrameDriftResult]:
    """Place aperture at WCS + collective frame drift; peak = max over footprint.

    AstroImageJ rule: one frame drift from stars centroided reliably; every other
    star moves with the frame (no per-star search or faint-star centroid).

    Drift references: variable target (``drift_ref_catalog_id`` / sky) measured with mag-guided
    at WCS for frame-shift diagnostic only.

    Placement: per-frame aligned DAO ``(x, y)`` on the raw grid (master-grid lock); fallback
    WCS + collective drift when aligned coordinates are missing. Optional 11 px COM centroid
    refinement (photutils-style); never a brightest-pixel search on targets or comps.

    Returns ``(peak_max_adu_raw, placed_x, placed_y, frame_drift)``.
    """
    n = int(ra_deg.size)
    peaks = np.full(n, np.nan, dtype=np.float64)
    px = np.full(n, np.nan, dtype=np.float64)
    py = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return peaks, px, py, FrameDriftResult(0.0, 0.0, 0, float("nan"), "empty")

    arr = np.asarray(raw_arr, dtype=np.float64)
    ap_arr = aligned_peak if aligned_peak is not None else np.full(n, np.nan)
    ax_arr = np.asarray(aligned_x, dtype=np.float64) if aligned_x is not None else None
    ay_arr = np.asarray(aligned_y, dtype=np.float64) if aligned_y is not None else None

    try:
        wcs = WCS(raw_hdr)
        if not wcs.has_celestial:
            return peaks, px, py, FrameDriftResult(0.0, 0.0, 0, float("nan"), "wcs_missing")
        wx, wy = wcs.all_world2pix(ra_deg, dec_deg, 0)
    except Exception:  # noqa: BLE001
        return peaks, px, py, FrameDriftResult(0.0, 0.0, 0, float("nan"), "wcs_error")

    dxs: list[float] = []
    dys: list[float] = []
    method = "wcs_only"
    n_refs = 0

    target_i: int | None = None
    if drift_ref_catalog_id and catalog_ids is not None:
        for j, cid in enumerate(catalog_ids):
            if str(cid) == str(drift_ref_catalog_id):
                target_i = j
                break
    if target_i is None and drift_ref_ra is not None and drift_ref_dec is not None:
        dra = (np.asarray(ra_deg, dtype=np.float64) - float(drift_ref_ra)) * math.cos(
            math.radians(float(drift_ref_dec))
        )
        dde = np.asarray(dec_deg, dtype=np.float64) - float(drift_ref_dec)
        d2 = dra * dra + dde * dde
        if d2.size and np.isfinite(d2).any():
            j = int(np.nanargmin(d2))
            if float(d2[j]) < (5.0 / 3600.0) ** 2:
                target_i = j

    if target_i is not None and math.isfinite(float(wx[target_i])):
        off = _drift_offset_at_wcs(
            arr, float(wx[target_i]), float(wy[target_i]), use_mag_guided=True
        )
        if off is not None:
            dxs, dys = [off[0]], [off[1]]
            method = "mag_guided_target"
            n_refs = 1

    if not dxs:
        ref_idx = set(_select_drift_ref_indices(ap_arr).tolist())
        pk = np.asarray(ap_arr, dtype=np.float64)
        order = sorted(ref_idx, key=lambda j: float(pk[j]) if math.isfinite(float(pk[j])) else 0.0, reverse=True)
        for i in order[:10]:
            if not (math.isfinite(float(wx[i])) and math.isfinite(float(wy[i]))):
                continue
            off = _drift_offset_at_wcs(arr, float(wx[i]), float(wy[i]), use_mag_guided=True)
            if off is None:
                continue
            dxs.append(off[0])
            dys.append(off[1])
        if len(dxs) >= DRIFT_REF_MIN_COUNT:
            method = "mag_guided_refs"
        elif len(dxs) == 1:
            method = "mag_guided_single_ref"

    if dxs:
        drift_x = float(np.median(np.asarray(dxs)))
        drift_y = float(np.median(np.asarray(dys)))
        n_refs = len(dxs)
        resid = np.hypot(np.asarray(dxs) - drift_x, np.asarray(dys) - drift_y)
        rms = float(np.sqrt(np.mean(np.square(resid)))) if resid.size else float("nan")
    else:
        drift_x = drift_y = 0.0
        method = "wcs_only"
        n_refs = 0
        rms = float("nan")

    for i in range(n):
        if ax_arr is not None and ay_arr is not None:
            if math.isfinite(float(ax_arr[i])) and math.isfinite(float(ay_arr[i])):
                placed_x = float(ax_arr[i])
                placed_y = float(ay_arr[i])
            elif math.isfinite(float(wx[i])) and math.isfinite(float(wy[i])):
                placed_x = float(wx[i]) + drift_x
                placed_y = float(wy[i]) + drift_y
            else:
                continue
        elif math.isfinite(float(wx[i])) and math.isfinite(float(wy[i])):
            placed_x = float(wx[i]) + drift_x
            placed_y = float(wy[i]) + drift_y
        else:
            continue
        seed_peak = box_peak_max(arr, placed_x, placed_y)
        com = _centroid_com_cutout(arr, placed_x, placed_y, DRIFT_CENTROID_BOX_HALF)
        if com is not None:
            trial_peak = box_peak_max(arr, com[0], com[1])
            if trial_peak >= max(seed_peak * 0.75, 2500.0):
                placed_x, placed_y = float(com[0]), float(com[1])
        px[i] = placed_x
        py[i] = placed_y
        peaks[i] = box_peak_max(arr, placed_x, placed_y)

    frame_drift = FrameDriftResult(drift_x, drift_y, n_refs, rms, method)
    return peaks, px, py, frame_drift


def _update_draft_star_stats(
    ctx: SatDiagContext,
    catalog_ids: Sequence[str],
    peaks: np.ndarray,
    *,
    admission_thr: float | None,
    sat_thr: float | None,
) -> None:
    """Accumulate once-per-draft peak stats (INV-COMP-MEMBERSHIP)."""
    for cid, pk in zip(catalog_ids, peaks, strict=False):
        if not math.isfinite(float(pk)):
            continue
        cid_s = str(cid)
        st = ctx.star_peak_draft.get(cid_s)
        if st is None:
            st = StarPeakDraftStats(catalog_id=cid_s)
            ctx.star_peak_draft[cid_s] = st
        st.n_frames += 1
        fv = float(pk)
        if not math.isfinite(st.peak_max) or fv > st.peak_max:
            st.peak_max = fv
        if admission_thr is not None and fv > float(admission_thr):
            st.n_over_admission += 1
        if sat_thr is not None and fv >= float(sat_thr):
            st.n_saturated += 1


def finalize_draft_star_stats(ctx: SatDiagContext) -> None:
    """Compute medians after all frames processed."""
    for st in ctx.star_peak_draft.values():
        if st.n_frames >= 10 and st.n_over_admission / st.n_frames > 0.10:
            st.admission_reject = True


def run_sat_diag(
    archive_root: Path,
    *,
    equipment_adu: float | None = None,
    hdr: fits.Header | None = None,
    raw_paths: Sequence[Path] | None = None,
) -> SatDiagContext:
    """Run Check A + B for a draft archive root."""
    paths = list(raw_paths) if raw_paths is not None else sample_raw_light_paths(archive_root)
    pileup = derive_ceiling_from_paths(paths)
    ref_hdr = hdr if hdr is not None else fits.Header()
    if not ref_hdr and paths:
        with fits.open(paths[0], memmap=False) as hdul:
            ref_hdr = hdul[0].header
    ctx = resolve_sat_limit(hdr=ref_hdr, pileup=pileup, equipment_adu=equipment_adu)
    if not paths:
        ctx.warnings.append("UNVERIFIED_INPUT: no raw light frames found")
    return ctx


def write_sat_diag_json(ctx: SatDiagContext, out_path: Path) -> None:
    finalize_draft_star_stats(ctx)
    payload = {
        "schema_version": 2,
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **ctx.to_json_dict(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def commit_sat_diag_provenance(
    ctx: SatDiagContext,
    archive: Path | str,
    *,
    placed_aperture_used: bool,
) -> None:
    """Persist ``sat_diag.json`` after per-frame catalog (same operation as placed aperture)."""
    if placed_aperture_used:
        ctx.raw_peaks_used = True
        ctx.sat_peak_source = SAT_PEAK_SOURCE_PLACED
    write_sat_diag_json(ctx, Path(archive) / "sat_diag.json")


def load_sat_diag_json(path: Path) -> SatDiagContext | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pileup_raw = data.pop("pileup", None)
        pileup = PileupResult(**pileup_raw) if isinstance(pileup_raw, dict) else None
        drift_raw = data.pop("last_frame_drift", None)
        last_drift = FrameDriftResult(**drift_raw) if isinstance(drift_raw, dict) else None
        star_raw = data.pop("star_peak_draft", {})
        star_peak_draft = {
            str(k): StarPeakDraftStats(**v) if isinstance(v, dict) else v
            for k, v in (star_raw or {}).items()
        }
        data.pop("placement_residual_median_px", None)
        data.pop("placement_residual_p95_px", None)
        data.pop("schema_version", None)
        data.pop("updated_utc", None)
        ctx = SatDiagContext(
            sat_adu=data.get("sat_adu"),
            lin_adu=data.get("lin_adu"),
            sat_source=str(data.get("sat_source", "none")),
            lin_source=str(data.get("lin_source", "DEFAULT_FRAC")),
            bitpix_ceiling=data.get("bitpix_ceiling"),
            derived_ceiling=data.get("derived_ceiling"),
            header_value=data.get("header_value"),
            equipment_value=data.get("equipment_value"),
            refuted_source=data.get("refuted_source"),
            refuted_value=data.get("refuted_value"),
            warnings=list(data.get("warnings") or []),
            pileup=pileup,
            xbinning=data.get("xbinning"),
            ybinning=data.get("ybinning"),
            raw_peaks_used=bool(data.get("raw_peaks_used")),
            sat_peak_source=str(data.get("sat_peak_source") or SAT_PEAK_SOURCE_PLACED),
            lin_adu_native=data.get("lin_adu_native"),
            sat_adu_native=data.get("sat_adu_native"),
            last_frame_drift=last_drift,
            star_peak_draft=star_peak_draft,
            frame_drift_residuals_px=list(data.get("frame_drift_residuals_px") or []),
        )
        return ctx
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("load_sat_diag_json failed: %s", exc)
        return None


def apply_raw_peaks_to_proc_df(
    df: Any,
    raw_arr: np.ndarray,
    raw_hdr: fits.Header,
    ctx: SatDiagContext,
    *,
    ref_ra: float | None = None,
    ref_dec: float | None = None,
    drift_ref_catalog_id: str | None = None,
    aligned_hdr: fits.Header | None = None,
) -> SatDiagContext:
    """Merge placed-aperture raw peak columns into a per-frame proc catalog."""
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return ctx
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        return ctx

    ra = pd.to_numeric(df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
    de = pd.to_numeric(df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(ra) & np.isfinite(de)

    aligned_pk = np.full(len(df), np.nan, dtype=np.float64)
    if "peak_max_adu_aligned" in df.columns:
        aligned_pk = pd.to_numeric(df["peak_max_adu_aligned"], errors="coerce").to_numpy(
            dtype=np.float64
        )
    elif "peak_max_adu" in df.columns:
        aligned_pk = pd.to_numeric(df["peak_max_adu"], errors="coerce").to_numpy(dtype=np.float64)

    ax = ay = None
    if "x" in df.columns and "y" in df.columns:
        ax = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=np.float64)
        ay = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=np.float64)

    id_col = "catalog_id" if "catalog_id" in df.columns else "name"
    cids_ok: list[str] | None = None
    if id_col in df.columns and ok.any():
        cids_ok = df[id_col].astype(str).to_numpy()[ok].tolist()

    peaks, placed_x, placed_y, frame_drift = measure_raw_peaks_frame(
        raw_arr,
        raw_hdr,
        ra_deg=ra[ok],
        dec_deg=de[ok],
        aligned_x=ax[ok] if ax is not None else None,
        aligned_y=ay[ok] if ay is not None else None,
        aligned_hdr=aligned_hdr,
        aligned_peak=aligned_pk[ok] if aligned_pk is not None else None,
        drift_ref_ra=ref_ra,
        drift_ref_dec=ref_dec,
        drift_ref_catalog_id=drift_ref_catalog_id,
        catalog_ids=cids_ok,
    )

    pk_col = np.full(len(df), np.nan, dtype=np.float64)
    px_col = np.full(len(df), np.nan, dtype=np.float64)
    py_col = np.full(len(df), np.nan, dtype=np.float64)
    if ok.any():
        idx = np.nonzero(ok)[0]
        pk_col[idx] = peaks
        px_col[idx] = placed_x
        py_col[idx] = placed_y

    df["peak_max_adu_raw"] = pk_col
    df["peak_placed_x_raw"] = px_col
    df["peak_placed_y_raw"] = py_col
    df["peak_max_adu_aligned"] = aligned_pk
    df["peak_max_adu"] = pk_col
    df["sat_peak_source"] = SAT_PEAK_SOURCE_PLACED

    sat = ctx.sat_adu
    lin = ctx.lin_adu
    adm = ctx.admission_threshold_adu()
    use_pk = pk_col
    pk_finite = np.isfinite(use_pk)
    likely_sat = np.zeros(len(df), dtype=bool)
    likely_nl = np.zeros(len(df), dtype=bool)
    is_sat = np.zeros(len(df), dtype=bool)
    if sat is not None:
        thr85 = ctx.likely_saturated_threshold_adu()
        if thr85 is not None:
            likely_sat = pk_finite & (use_pk >= thr85)
        is_sat = pk_finite & (use_pk >= float(sat))
    if lin is not None:
        likely_nl = pk_finite & (use_pk >= float(lin))

    df["likely_saturated_raw"] = likely_sat
    df["likely_nonlinear_raw"] = likely_nl
    df["is_saturated_raw"] = is_sat

    if ctx.saturate_limit_adu_85pct() is not None:
        df["saturate_limit_adu"] = float(ctx.sat_adu)
        df["saturate_limit_adu_85pct"] = float(ctx.saturate_limit_adu_85pct())
        df["linearity_limit_adu"] = float(lin) if lin is not None else np.nan
        df["sat_limit_source"] = str(ctx.sat_source)

    id_col = "catalog_id" if "catalog_id" in df.columns else "name"
    if id_col in df.columns:
        cids = df[id_col].astype(str).to_numpy()
        _update_draft_star_stats(
            ctx,
            [cids[i] for i in range(len(df)) if ok[i]],
            peaks,
            admission_thr=adm,
            sat_thr=float(sat) if sat is not None else None,
        )

    ctx.last_frame_drift = frame_drift
    if math.isfinite(frame_drift.residual_rms_px):
        ctx.frame_drift_residuals_px.append(float(frame_drift.residual_rms_px))

    bf = max(int(ctx.xbinning or 1), 1)
    if ctx.lin_adu is not None:
        ctx.lin_adu_native = float(ctx.lin_adu) / float(bf)
    if ctx.sat_adu is not None:
        ctx.sat_adu_native = float(ctx.sat_adu) / float(bf)

    ctx.sat_peak_source = SAT_PEAK_SOURCE_PLACED
    ctx.raw_peaks_used = True
    return ctx


def stamp_sat_fits_headers(hdr: fits.Header, ctx: SatDiagContext) -> None:
    """VY_SAT* provenance headers (spec 10.1)."""
    src_map = {
        "HEADER": "HEADER",
        "EQUIPMENT": "EQUIPMENT",
        "DERIVED": "DERIVED",
        "CONFLICT_DERIVED": "CONFLICT_DERIVED",
        "DERIVED_NO_PILEUP": "DERIVED_NO_PILEUP",
        "BITPIX": "DERIVED_NO_PILEUP",
        "LEGACY_ALIGNED": "LEGACY_ALIGNED",
    }
    vy_src = src_map.get(str(ctx.sat_source), str(ctx.sat_source))
    hdr["VY_SATSRC"] = (vy_src, "SAT-DIAG limit source")
    if ctx.sat_adu is not None:
        hdr["VY_SATADU"] = (float(ctx.sat_adu), "Saturation level image ADU")
    if ctx.lin_adu is not None:
        hdr["VY_LINADU"] = (float(ctx.lin_adu), "Linearity level image ADU")
    hdr["VY_LINSRC"] = (str(ctx.lin_source), "Linearity provenance")
    bf = ctx.xbinning or 1
    hdr["VY_SATBF"] = (int(bf), "Binning key for SAT-DIAG")
    hdr["VY_SATPS"] = (
        str(ctx.sat_peak_source),
        "Peak source for saturation (PLACED_APERTURE)",
    )


def resolve_drift_ref_sky_deg(
    platesolve_dir: Path,
    *,
    frame_name_hint: str | None = None,
) -> tuple[float | None, float | None, str | None]:
    """Sky position and catalog id of the variable target for frame drift.

    Prefers ``photometry/active_targets.csv`` matched to the frame name prefix
    (e.g. ``BO_CVn_Light_001`` -> ``BO CVn``). Returns ``(ra_deg, dec_deg, catalog_id)``.
    """
    import pandas as pd

    ps = Path(platesolve_dir)
    at_path = ps / "photometry" / "active_targets.csv"
    if at_path.is_file():
        try:
            at = pd.read_csv(at_path, dtype={"catalog_id": str})
            if (
                not at.empty
                and "ra_deg" in at.columns
                and "dec_deg" in at.columns
            ):
                work = at.copy()
                if "skip_photometry" in work.columns:
                    sk = work["skip_photometry"].astype(str).str.lower().isin(
                        ("true", "1", "yes")
                    )
                    work = work.loc[~sk]
                if not work.empty and frame_name_hint and "name" in work.columns:
                    stem = Path(frame_name_hint).stem.replace("proc_", "")
                    parts = stem.replace("-", " ").replace("_", " ").split()
                    if len(parts) >= 2 and parts[-1].lower() == "light":
                        target_hint = " ".join(parts[:-1])
                    else:
                        target_hint = stem.replace("_", " ")
                    names = work["name"].astype(str).str.replace("_", " ", regex=False)
                    hit = work[
                        names.str.contains(target_hint, case=False, na=False, regex=False)
                    ]
                    if not hit.empty:
                        row = hit.iloc[0]
                        ra = float(row["ra_deg"])
                        de = float(row["dec_deg"])
                        cid = str(row.get("catalog_id", "")) or None
                        if math.isfinite(ra) and math.isfinite(de):
                            return ra, de, cid
                if len(work) == 1:
                    row = work.iloc[0]
                    ra = float(row["ra_deg"])
                    de = float(row["dec_deg"])
                    cid = str(row.get("catalog_id", "")) or None
                    if math.isfinite(ra) and math.isfinite(de):
                        return ra, de, cid
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("resolve_drift_ref_sky_deg active_targets: %s", exc)

    vt_path = ps / "variable_targets.csv"
    if vt_path.is_file():
        try:
            vt = pd.read_csv(vt_path, dtype={"catalog_id": str}, nrows=5000)
            if (
                not vt.empty
                and "ra_deg" in vt.columns
                and "dec_deg" in vt.columns
            ):
                work = vt.copy()
                if "exo_host_name" in work.columns:
                    exo = work["exo_host_name"].astype(str).str.strip()
                    work = work.loc[exo.ne("") & exo.ne("nan")]
                if frame_name_hint and "exo_host_name" in vt.columns:
                    stem = Path(frame_name_hint).stem.replace("proc_", "")
                    hint = stem.replace("_", " ").replace("-", " ")
                    exo = vt["exo_host_name"].astype(str)
                    hit = vt[exo.str.contains(hint.split("_Light")[0].replace("_", " "), case=False, na=False)]
                    if not hit.empty:
                        row = hit.iloc[0]
                        ra = float(row["ra_deg"])
                        de = float(row["dec_deg"])
                        cid = str(row.get("catalog_id", "")) or None
                        if math.isfinite(ra) and math.isfinite(de):
                            return ra, de, cid
                if not work.empty and "priority" in work.columns:
                    pr = pd.to_numeric(work["priority"], errors="coerce")
                    work = work.loc[pr == pr.min()]
                if len(work) == 1:
                    row = work.iloc[0]
                    ra = float(row["ra_deg"])
                    de = float(row["dec_deg"])
                    cid = str(row.get("catalog_id", "")) or None
                    if math.isfinite(ra) and math.isfinite(de):
                        return ra, de, cid
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("resolve_drift_ref_sky_deg variable_targets: %s", exc)

    return None, None, None


def draft_archive_from_platesolve(platesolve_dir: Path) -> Path | None:
    """Draft archive root from obs_group platesolve folder."""
    ps = Path(platesolve_dir).resolve()
    for parent in ps.parents:
        if parent.name.startswith("draft_") and (parent / "draft_manifest.json").is_file():
            return parent
        if parent.name == "Drafts":
            break
    return None
