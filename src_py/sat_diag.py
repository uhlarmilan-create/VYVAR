"""SAT-DIAG: saturation and linearity limit gate (INV-SAT-01).

Camera-agnostic Check A (pile-up derivation), Check B (limit resolution),
raw peak measurement with self-check, and tier policy helpers.
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

# Implementation constants (spec section 5.2 / 8.3 -- not user config).
N_PILEUP_MIN = 100
PILEUP_RATIO = 10.0
SAT_DIAG_MAX_FRAMES = 30
LINEARITY_DEFAULT_FRAC = 0.85
SATURATE_LIMIT_FRACTION = 0.85
ADMISSION_SAT_FRAC = 0.70
RESCALED_MAX_FRAC = 0.85
DRAFT_SAT_EXCLUDE_FRAME_FRAC = 0.50

PEAK_SEARCH_HALF = 22  # reference-star drift only (mag-guided)
PEAK_TARGET_SEARCH_HALF = 5  # anchored search at aligned/WCS position
PEAK_BOX_HALF = 3
PEAK_MIN_ADU = 4000.0
PEAK_RING_CONTRAST_MIN = 1.8
PEAK_RING_R_IN = 11
PEAK_RING_R_OUT = 15
PEAK_ALIGNED_MAX_DIST_PX = 12.0
PEAK_RAW_ALIGNED_MAX_RATIO = 3.0

SAT_PEAK_SOURCE_RAW_VERIFIED = "RAW_VERIFIED"
SAT_PEAK_SOURCE_ALIGNED_INTERIM = "ALIGNED_INTERIM"

# Tier-1 sat sources that may exclude (spec section 9.1).
TIER1_SAT_SOURCES = frozenset(
    {"MEASURED", "HEADER", "EQUIPMENT", "DERIVED", "CONFLICT_DERIVED", "BITPIX"}
)
# Accepted-but-unverified (too-high ceiling not testable without ramp).
UNVERIFIED_SAT_SOURCES = frozenset({"DERIVED_NO_PILEUP", "LEGACY_ALIGNED"})


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
    peak_loc_fail_count: dict[str, int] = field(default_factory=dict)
    raw_peaks_used: bool = False
    sat_peak_source: str = SAT_PEAK_SOURCE_ALIGNED_INTERIM
    lin_adu_native: float | None = None
    sat_adu_native: float | None = None
    sat_peak_verified_measurements: int = 0
    sat_peak_interim_measurements: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.pileup is not None:
            d["pileup"] = asdict(self.pileup)
        if self.lin_adu is not None and self.lin_adu_native is None:
            bf = max(int(self.xbinning or 1), 1)
            d["lin_adu_native"] = float(self.lin_adu) / float(bf)
        if self.sat_adu is not None and self.sat_adu_native is None:
            bf = max(int(self.xbinning or 1), 1)
            d["sat_adu_native"] = float(self.sat_adu) / float(bf)
        return d

    def may_exclude_saturation(self) -> bool:
        return str(self.sat_source) in TIER1_SAT_SOURCES

    def may_exclude_linearity(self) -> bool:
        return str(self.lin_source) == "MEASURED"

    def admission_threshold_adu(self) -> float | None:
        if self.sat_adu is None:
            return None
        return float(self.sat_adu) * SATURATE_LIMIT_FRACTION * (ADMISSION_SAT_FRAC / SATURATE_LIMIT_FRACTION)

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

    # POSSIBLE_RESCALED_STACK (spec 6.3)
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


def _ring_median(arr: np.ndarray, cx: int, cy: int, r_in: int, r_out: int) -> float:
    h, w = arr.shape
    yy, xx = np.ogrid[:h, :w]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (dist2 >= r_in * r_in) & (dist2 <= r_out * r_out)
    vals = arr[mask]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def peak_self_check(arr: np.ndarray, cx: int, cy: int, peak: float) -> bool:
    """Local maximum + ring contrast + minimum signal (spec 8.3)."""
    if not math.isfinite(peak) or peak < PEAK_MIN_ADU:
        return False
    h, w = arr.shape
    if cx < 1 or cy < 1 or cx >= w - 1 or cy >= h - 1:
        return False
    patch = arr[cy - 1 : cy + 2, cx - 1 : cx + 2]
    if patch.size == 0:
        return False
    centre = float(arr[cy, cx])
    if centre < float(np.max(patch)) - 1e-6:
        return False
    ring_med = _ring_median(arr, cx, cy, PEAK_RING_R_IN, PEAK_RING_R_OUT)
    if not math.isfinite(ring_med) or ring_med <= 0:
        return False
    return (centre / ring_med) >= PEAK_RING_CONTRAST_MIN


def mag_guided_centroid(arr: np.ndarray, x0: float, y0: float, half: int = PEAK_SEARCH_HALF) -> tuple[int, int]:
    """Brightest pixel in search window (reference-star drift only)."""
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


def expected_raw_from_aligned_centroid(
    aligned_x: float,
    aligned_y: float,
    ra_deg: float,
    dec_deg: float,
    aligned_hdr: fits.Header,
    raw_hdr: fits.Header,
) -> tuple[float, float] | None:
    """Apply aligned DAO residual (centroid minus aligned WCS) to raw WCS."""
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


def peak_verify_near_expected(
    arr: np.ndarray,
    expected_x: float,
    expected_y: float,
    aligned_peak: float | None,
    *,
    sat_adu: float | None,
    max_dist_px: float = PEAK_ALIGNED_MAX_DIST_PX,
) -> tuple[int, int, float] | None:
    """Search within ``max_dist_px`` of expected for a verified peak (best ADU)."""
    h, w = arr.shape
    xi = int(round(expected_x))
    yi = int(round(expected_y))
    best: tuple[int, int, float] | None = None
    r2_max = float(max_dist_px) ** 2
    for dy in range(-int(math.ceil(max_dist_px)), int(math.ceil(max_dist_px)) + 1):
        for dx in range(-int(math.ceil(max_dist_px)), int(math.ceil(max_dist_px)) + 1):
            if float(dx * dx + dy * dy) > r2_max:
                continue
            gx, gy = xi + dx, yi + dy
            if gx < 1 or gy < 1 or gx >= w - 1 or gy >= h - 1:
                continue
            pk = box_peak_max(arr, gx, gy)
            if not peak_self_check(arr, gx, gy, pk):
                continue
            if not peak_raw_plausible(pk, aligned_peak, sat_adu=sat_adu):
                continue
            if best is None or pk > best[2]:
                best = (gx, gy, pk)
    return best


def raw_ref_pixel_from_aligned_ref(
    arr: np.ndarray,
    raw_hdr: fits.Header,
    ref_ra: float,
    ref_dec: float,
) -> tuple[float, float] | None:
    """Raw pixel of reference star (mag-guided on WCS position)."""
    try:
        wcs = WCS(raw_hdr)
        if not wcs.has_celestial:
            return None
        rx, ry = wcs.all_world2pix(float(ref_ra), float(ref_dec), 0)
        if not (math.isfinite(rx) and math.isfinite(ry)):
            return None
        gx, gy = mag_guided_centroid(arr, float(rx), float(ry))
        return float(gx), float(gy)
    except Exception:  # noqa: BLE001
        return None


def peak_raw_plausible(
    raw_peak: float,
    aligned_peak: float | None,
    *,
    sat_adu: float | None,
) -> bool:
    """Brightness plausibility: raw must track aligned unless aligned is near ceiling."""
    if aligned_peak is None or not math.isfinite(aligned_peak) or aligned_peak <= 0:
        return True
    if not math.isfinite(raw_peak) or raw_peak <= 0:
        return False
    if sat_adu is not None and math.isfinite(sat_adu) and float(aligned_peak) >= float(sat_adu) * 0.85:
        # Aligned resampling can exceed raw container; ratio test not meaningful.
        return True
    ratio = float(raw_peak) / float(aligned_peak)
    lo = 1.0 / PEAK_RAW_ALIGNED_MAX_RATIO
    hi = PEAK_RAW_ALIGNED_MAX_RATIO
    return lo <= ratio <= hi


def box_peak_max(arr: np.ndarray, x: float, y: float, half: int = PEAK_BOX_HALF) -> float:
    from pipeline import _box_peak_max_adu  # noqa: PLC0415

    return _box_peak_max_adu(arr, x, y, half=half)


def measure_raw_peaks_frame(
    raw_arr: np.ndarray,
    raw_hdr: fits.Header,
    *,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    ref_ra: float | None = None,
    ref_dec: float | None = None,
    ref_aligned_x: float | None = None,
    ref_aligned_y: float | None = None,
    aligned_x: np.ndarray | None = None,
    aligned_y: np.ndarray | None = None,
    aligned_hdr: fits.Header | None = None,
    aligned_peak: np.ndarray | None = None,
    sat_adu: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw peaks with plate-offset anchor + verification for matched stars.

    Uses aligned-frame centroids transferred to raw via a per-frame reference
    star (same plate scale). Falls back to WCS+drift when aligned coords absent.

    Returns ``(peak_max_adu_raw, peak_loc_ok, peak_loc_fail)`` arrays length N.
    ``peak_loc_ok`` means RAW_VERIFIED (right star, plausible peak).
    """
    n = int(ra_deg.size)
    peaks = np.full(n, np.nan, dtype=np.float64)
    loc_ok = np.zeros(n, dtype=bool)
    loc_fail = np.zeros(n, dtype=bool)

    if n == 0:
        return peaks, loc_ok, loc_fail

    arr = np.asarray(raw_arr, dtype=np.float64)

    raw_ref: tuple[float, float] | None = None
    if ref_ra is not None and ref_dec is not None and math.isfinite(ref_ra) and math.isfinite(ref_dec):
        raw_ref = raw_ref_pixel_from_aligned_ref(arr, raw_hdr, float(ref_ra), float(ref_dec))

    drift_x, drift_y = 0.0, 0.0
    try:
        wcs = WCS(raw_hdr)
        if wcs.has_celestial and raw_ref is not None and ref_ra is not None and ref_dec is not None:
            rx, ry = wcs.all_world2pix(float(ref_ra), float(ref_dec), 0)
            drift_x = float(raw_ref[0] - rx)
            drift_y = float(raw_ref[1] - ry)
        xs, ys = wcs.all_world2pix(ra_deg, dec_deg, 0)
    except Exception:  # noqa: BLE001
        loc_fail[:] = True
        return peaks, loc_ok, loc_fail

    ax_arr = aligned_x if aligned_x is not None else np.full(n, np.nan)
    ay_arr = aligned_y if aligned_y is not None else np.full(n, np.nan)
    ap_arr = aligned_peak if aligned_peak is not None else np.full(n, np.nan)

    plate_ok = (
        raw_ref is not None
        and ref_aligned_x is not None
        and ref_aligned_y is not None
        and aligned_x is not None
        and aligned_y is not None
    )
    wcs_res_ok = aligned_hdr is not None and aligned_x is not None and aligned_y is not None

    for i in range(n):
        try:
            expected_x = expected_y = float("nan")
            if (
                wcs_res_ok
                and math.isfinite(float(ax_arr[i]))
                and math.isfinite(float(ay_arr[i]))
            ):
                hit_xy = expected_raw_from_aligned_centroid(
                    float(ax_arr[i]),
                    float(ay_arr[i]),
                    float(ra_deg[i]),
                    float(dec_deg[i]),
                    aligned_hdr,
                    raw_hdr,
                )
                if hit_xy is not None:
                    expected_x, expected_y = hit_xy
            if not (math.isfinite(expected_x) and math.isfinite(expected_y)):
                if (
                    plate_ok
                    and math.isfinite(float(ax_arr[i]))
                    and math.isfinite(float(ay_arr[i]))
                ):
                    expected_x = float(raw_ref[0]) + (float(ax_arr[i]) - float(ref_aligned_x))
                    expected_y = float(raw_ref[1]) + (float(ay_arr[i]) - float(ref_aligned_y))
                else:
                    expected_x = float(xs[i]) + drift_x
                    expected_y = float(ys[i]) + drift_y

            if not (math.isfinite(expected_x) and math.isfinite(expected_y)):
                loc_fail[i] = True
                continue

            ap_i = float(ap_arr[i]) if math.isfinite(float(ap_arr[i])) else None
            hit = peak_verify_near_expected(
                arr, expected_x, expected_y, ap_i, sat_adu=sat_adu
            )
            if hit is not None:
                _, _, pk = hit
                peaks[i] = pk
                loc_ok[i] = True
            else:
                loc_fail[i] = True
        except Exception:  # noqa: BLE001
            loc_fail[i] = True

    return peaks, loc_ok, loc_fail


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
    payload = {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **ctx.to_json_dict(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def load_sat_diag_json(path: Path) -> SatDiagContext | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pileup_raw = data.pop("pileup", None)
        pileup = PileupResult(**pileup_raw) if isinstance(pileup_raw, dict) else None
        peak_fails = data.pop("peak_loc_fail_count", {})
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
            peak_loc_fail_count=dict(peak_fails or {}),
            raw_peaks_used=bool(data.get("raw_peaks_used")),
            sat_peak_source=str(
                data.get("sat_peak_source") or SAT_PEAK_SOURCE_ALIGNED_INTERIM
            ),
            lin_adu_native=data.get("lin_adu_native"),
            sat_adu_native=data.get("sat_adu_native"),
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
    aligned_hdr: fits.Header | None = None,
) -> SatDiagContext:
    """Merge raw peak columns into a per-frame proc catalog DataFrame."""
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

    ref_aligned_x = ref_aligned_y = None
    if ax is not None and ay is not None and "flux" in df.columns:
        flux_s = pd.to_numeric(df["flux"], errors="coerce").fillna(0)
        if ref_ra is not None and ref_dec is not None and "ra_deg" in df.columns:
            _ra = pd.to_numeric(df["ra_deg"], errors="coerce")
            _de = pd.to_numeric(df["dec_deg"], errors="coerce")
            _d2 = (_ra - float(ref_ra)) ** 2 + (_de - float(ref_dec)) ** 2
            if _d2.notna().any() and float(_d2.min()) < 1e-10:
                _j = int(_d2.idxmin())
            else:
                _j = int(flux_s.idxmax())
        else:
            _j = int(flux_s.idxmax())
        ref_aligned_x = float(ax[_j])
        ref_aligned_y = float(ay[_j])

    peaks, loc_ok, loc_fail = measure_raw_peaks_frame(
        raw_arr,
        raw_hdr,
        ra_deg=ra[ok],
        dec_deg=de[ok],
        ref_ra=ref_ra,
        ref_dec=ref_dec,
        ref_aligned_x=ref_aligned_x,
        ref_aligned_y=ref_aligned_y,
        aligned_x=ax[ok] if ax is not None else None,
        aligned_y=ay[ok] if ay is not None else None,
        aligned_hdr=aligned_hdr,
        aligned_peak=aligned_pk[ok] if aligned_pk is not None else None,
        sat_adu=ctx.sat_adu,
    )

    pk_col = np.full(len(df), np.nan, dtype=np.float64)
    ok_col = np.zeros(len(df), dtype=bool)
    fail_col = np.zeros(len(df), dtype=bool)
    if ok.any():
        pk_col[np.nonzero(ok)[0]] = peaks
        ok_col[np.nonzero(ok)[0]] = loc_ok
        fail_col[np.nonzero(ok)[0]] = loc_fail

    df["peak_max_adu_raw"] = pk_col
    df["peak_loc_ok"] = ok_col
    df["peak_loc_fail"] = fail_col
    df["peak_max_adu_aligned"] = aligned_pk

    # Authoritative peak for saturation: RAW_VERIFIED when search passes, else aligned.
    auth_pk = np.where(ok_col & np.isfinite(pk_col), pk_col, aligned_pk)
    sat_src_col = np.where(
        ok_col & np.isfinite(pk_col),
        SAT_PEAK_SOURCE_RAW_VERIFIED,
        SAT_PEAK_SOURCE_ALIGNED_INTERIM,
    )
    df["peak_max_adu"] = auth_pk
    df["sat_peak_source"] = sat_src_col

    sat = ctx.sat_adu
    lin = ctx.lin_adu
    use_pk = pd.to_numeric(df["peak_max_adu"], errors="coerce").to_numpy(dtype=np.float64)
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
    if id_col in df.columns and "peak_loc_fail" in df.columns:
        fail_mask = df["peak_loc_fail"].astype(bool)
        for cid, cnt in df.loc[fail_mask, id_col].astype(str).value_counts().items():
            ctx.peak_loc_fail_count[str(cid)] = int(ctx.peak_loc_fail_count.get(str(cid), 0)) + int(cnt)

    n_verified = int((ok_col & np.isfinite(pk_col)).sum())
    n_interim = int(len(df) - n_verified)
    ctx.sat_peak_verified_measurements += n_verified
    ctx.sat_peak_interim_measurements += n_interim
    if ctx.sat_peak_verified_measurements > 0 and ctx.sat_peak_interim_measurements > 0:
        ctx.sat_peak_source = "MIXED"
    elif ctx.sat_peak_verified_measurements > 0:
        ctx.sat_peak_source = SAT_PEAK_SOURCE_RAW_VERIFIED
    else:
        ctx.sat_peak_source = SAT_PEAK_SOURCE_ALIGNED_INTERIM

    bf = max(int(ctx.xbinning or 1), 1)
    if ctx.lin_adu is not None:
        ctx.lin_adu_native = float(ctx.lin_adu) / float(bf)
    if ctx.sat_adu is not None:
        ctx.sat_adu_native = float(ctx.sat_adu) / float(bf)

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
        "Peak source for saturation (RAW_VERIFIED/ALIGNED_INTERIM/MIXED)",
    )


def draft_archive_from_platesolve(platesolve_dir: Path) -> Path | None:
    """Draft archive root from obs_group platesolve folder."""
    ps = Path(platesolve_dir).resolve()
    # .../Archive/Drafts/draft_NNNNNN/platesolve/<obs_group>
    for parent in ps.parents:
        if parent.name.startswith("draft_") and (parent / "draft_manifest.json").is_file():
            return parent
        if parent.name == "Drafts":
            break
    return None
