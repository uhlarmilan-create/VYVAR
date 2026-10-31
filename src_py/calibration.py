from __future__ import annotations

"""Master calibration resampling (software binning) for VYVAR.

**CalibrationLibrary** convention: master dark / flat FITS are stored as a **native** stack of
calibration frames (no software bin-down to match light binning). Prefer sensor **1x1** readout for
darks/flats; header ``XBINNING``/``YBINNING`` reflect the calibration frames.

At **calibrate time**, masters are resampled in RAM (temporary arrays) to match each light's
``XBINNING`` and image shape:

- **Dark** - **sum** over each ``block_factor``x``block_factor`` block (charge-like quantity).
- **Flat** - **arithmetic mean** over each block only (e.g. four pixels for 1x1 master -> 2x2 light).
  Spatial re-binning must **not** use sum or median on the flat; mean preserves relative illumination.
  Median normalization to ~1 is applied **after** resample for new library flats marked ``VYFLNRD=1``;
  legacy flats were normalized at stack time and are only resampled here.

Then ``(light - dark) / flat`` runs at matching resolution.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Literal, NamedTuple

DarkResampleMode = Literal["sum", "mean"]

import numpy as np
from astropy.io import fits

from database import VyvarDatabase
from infolog import log_event

logger = logging.getLogger(__name__)

class MasterResamplingError(ValueError):
    """Raised when master binning cannot be matched to the light frame."""


class ProcessedMasterResult(NamedTuple):
    """Output of :func:`get_processed_master`."""

    data: np.ndarray
    master_binning: int
    resampled: bool
    block_factor: int
    #: Median ADU of flat **before** :func:`normalize_flat_master` at calibrate (new VYFLNRD masters only).
    flat_median_adu_before_norm: float | None = None
    #: True if flat was normalized in RAM here (after resample); legacy flats stay False (pipeline divides).
    flat_normalized_at_calibrate: bool = False


def read_master_binning_from_header(header: fits.Header) -> int:
    """Return X axis binning recorded in a master (or light) primary header."""
    from utils import fits_binning_xy_from_header

    return fits_binning_xy_from_header(header)[0]


def resample_master_to_light_binning(
    data: np.ndarray,
    *,
    master_binning: int,
    light_binning: int,
    kind: Literal["dark", "flat"],
    dark_resample_mode: DarkResampleMode | None = None,
) -> tuple[np.ndarray, int]:
    """Bin a 2D master down so one output pixel matches ``light_binning`` / ``master_binning``.

    - **Dark:** sum or mean over each *block_factor*x*block_factor* block, as the camera bins
      (``dark_resample_mode``, measured by CAL-DIAG). G01-24 (decision A3): there is no default - a dark
      that must be rebinned without a known convention raises :class:`MasterResamplingError`.
    - **Flat:** ``np.mean`` over each block - **only** the plain average of those pixels (e.g. four
      pixels for ``block_factor==2``). Never ``sum`` or ``median`` for flat spatial re-binning.

    If the array is not an exact multiple of ``block_factor`` (e.g. one extra row), the **trailing**
    rows/columns are clipped so binning is well-defined (typically 1 px at full chip).

    Returns
    -------
    (array, block_factor)
        ``block_factor`` is 1 when no resampling was done.
    """
    mb = max(1, int(master_binning))
    lb = max(1, int(light_binning))
    if lb < mb:
        raise MasterResamplingError(
            f"Light binning {lb}x is smaller than master binning {mb}x - upsampling is not supported."
        )
    if lb % mb != 0:
        raise MasterResamplingError(
            f"Light binning {lb} must be an integer multiple of master binning {mb}."
        )
    bf = lb // mb
    if bf == 1:
        return np.asarray(data, dtype=np.float32), 1
    if kind == "dark" and dark_resample_mode not in ("sum", "mean"):
        raise MasterResamplingError(
            f"DARK {mb}x{mb} must be rebinned to the LIGHT binning {lb}x{lb}, but the camera's binning "
            "convention (SUM or MEAN) is not known - it is measured by CAL-DIAG from the data. "
            f"Make a master DARK at {lb}x{lb} binning."
        )

    a = np.asarray(data, dtype=np.float32)
    if a.ndim != 2:
        raise MasterResamplingError(f"Expected a 2D master, got shape {a.shape}.")

    h, w = int(a.shape[0]), int(a.shape[1])
    h_trim = (h // bf) * bf
    w_trim = (w // bf) * bf
    if h_trim < bf or w_trim < bf:
        raise MasterResamplingError(
            f"Master size {h}x{w} is too small after fitting to block {bf}x{bf}."
        )
    if h_trim != h or w_trim != w:
        a = a[:h_trim, :w_trim]
        h, w = h_trim, w_trim

    v = a.reshape(h // bf, bf, w // bf, bf)
    if kind == "dark":
        if dark_resample_mode == "mean":
            out = np.mean(v, axis=(1, 3)).astype(np.float32, copy=False)
        else:  # "sum" (checked above)
            out = np.sum(v, axis=(1, 3)).astype(np.float32, copy=False)
    else:
        # Flat: block **mean** only - not sum (would skew flat field) and not median (not a block average).
        out = np.mean(v, axis=(1, 3)).astype(np.float32, copy=False)
    return out, bf


def _flat_saved_unnormalized(hdr: fits.Header) -> bool:
    """True if CalibrationLibrary flat was saved without median norm (VYFLNRD=1). Legacy masters omit this."""
    v = hdr.get("VYFLNRD")
    if v is None:
        return False
    try:
        return int(v) == 1
    except (TypeError, ValueError):
        return str(v).strip().upper() in ("T", "TRUE", "YES", "1")


def _valid_bayer_pattern_4(s: str | None) -> str | None:
    """Return 4-letter RGB Bayer code (e.g. RGGB) or None."""
    if not s:
        return None
    p = "".join(str(s).upper().split())
    if len(p) < 4:
        return None
    p = p[:4]
    if not all(c in "RGB" for c in p):
        return None
    return p


def _db_equipment_suggests_osc(db: VyvarDatabase | None, equipment_id: int | None) -> bool:
    """True if EQUIPMENTS.SENSORTYPE hints one-shot / Bayer / colour camera."""
    if db is None or equipment_id is None:
        return False
    try:
        row = db.conn.execute(
            "SELECT SENSORTYPE FROM EQUIPMENTS WHERE ID = ?",
            (int(equipment_id),),
        ).fetchone()
    except sqlite3.Error:  # noqa: BLE001
        # EXC-0020: ? -- intent unclear ((int(equipment_id),), / ).fetchone() / except Exception:  # noqa: BLE00... (EXCEPT-BULK 2026-07-08)
        return False
    if not row or row[0] in (None, ""):
        return False
    u = str(row[0]).upper()
    needles = (
        "OSC",
        "BAYER",
        "ONE-SHOT",
        "ONESHOT",
        "ONE SHOT",
        "COLOUR",
        "COLOR",
        "RGB",
    )
    return any(n in u for n in needles)


def normalize_flat_master(
    master: np.ndarray,
    header: fits.Header,
    *,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    bin_factor: int = 1,
) -> np.ndarray:
    """Scale master flat to median ~1 (global or per-Bayer-tile). Called after resample to light binning."""
    m = np.asarray(master, dtype=np.float32).copy()
    if m.ndim != 2:  # G01-31: masters are 2-D images; the old global branch had no producer
        raise ValueError(f"master flat must be a 2-D image, got shape {m.shape}")

    h, w = int(m.shape[0]), int(m.shape[1])
    hdr_pat = _valid_bayer_pattern_4(str(header.get("BAYERPAT") or ""))
    assumed_pat = False
    pat = hdr_pat
    if pat is None and bin_factor == 1 and db is not None and id_equipments is not None:
        try:
            db_pat = db.get_equipment_bayermask(int(id_equipments))
        except Exception:  # noqa: BLE001
            db_pat = None
        if db_pat and h % 2 == 0 and w % 2 == 0 and h >= 2 and w >= 2:
            pat = db_pat
            assumed_pat = True
    if pat is None and bin_factor == 1 and _db_equipment_suggests_osc(db, id_equipments):
        if h % 2 == 0 and w % 2 == 0 and h >= 2 and w >= 2:
            pat = "RGGB"
            assumed_pat = True

    use_bayer = (
        pat is not None
        and bin_factor == 1
        and h >= 2
        and w >= 2
        and h % 2 == 0
        and w % 2 == 0
    )
    if hdr_pat and bin_factor > 1:
        log_event(
            "Master flat: global normalization is used after binning (binning mixes the Bayer planes)."
        )

    import calib_masters as _cm  # noqa: PLC0415  (G01-23: the one median normaliser, also used at build)

    if use_bayer and pat is not None:
        labels = (f"[0,0]={pat[0]}", f"[0,1]={pat[1]}", f"[1,0]={pat[2]}", f"[1,1]={pat[3]}")
        m, medians = _cm.normalise_flat(m, pat)
        for mv, lab in zip(medians, labels, strict=True):
            if not (np.isfinite(mv) and mv > 0):
                log_event(f"Master flat: warning - invalid median for tile {lab}, division skipped.")
        note = f"Bayer {pat}" + (" (from EQUIPMENTS.BAYERMASK)" if assumed_pat else "")
        log_event(
            "Master flat: normalization per Bayer tile "
            f"({note}); ADU medians before norm: "
            f"{labels[0]}={medians[0]:.6g}, {labels[1]}={medians[1]:.6g}, "
            f"{labels[2]}={medians[2]:.6g}, {labels[3]}={medians[3]:.6g}."
        )
        header["VYFLTNRM"] = ("BAYER4", "Per-tile median normalization (OSC/Bayer)")
        header["VYFLTPAT"] = (pat, "Bayer pattern used for flat normalization")
        gpost = float(np.nanmedian(m))
        if np.isfinite(gpost):
            header["VYFLTMD"] = (gpost, "Global median ADU after per-tile norm")
    else:
        m, (gmed,) = _cm.normalise_flat(m, None)
        if not (np.isfinite(gmed) and gmed > 0):
            log_event("Master flat: warning - global median invalid, normalization skipped.")
        log_event(f"Master flat: global median (normalization factor): {gmed:.6g} ADU.")
        header["VYFLTNRM"] = ("GLOBAL", "Global median flat normalization")
        if np.isfinite(gmed):
            header["VYFLTMD"] = (float(gmed), "Global median ADU before normalization")

    # G01-02/G01-03/G01-29: a pixel without a valid flat value is bad (NaN), never replaced by 1.0
    bad = (m <= 0) | ~np.isfinite(m)
    if np.any(bad):
        log_event(f"Master flat: {int(np.sum(bad))} pixels <= 0 or invalid -> bad (NaN in the calibrated frame).")
    m = np.where(bad, np.float32(np.nan), m).astype(np.float32)

    gfin = float(np.nanmedian(m))
    if np.isfinite(gfin) and abs(gfin - 1.0) > 0.02 and abs(gfin) > 0:
        m = (m / gfin).astype(np.float32)
        log_event(f"Master flat: extra scale 1/{gfin:.4f} so that the global median ~ 1.0.")
    mean_a = float(np.nanmean(m))
    log_event(f"Master flat: after normalization median={float(np.nanmedian(m)):.4f}, mean ~ {mean_a:.4f}.")
    return m


def _flat_offset_at_use(path: Path, hdr: fits.Header, db: VyvarDatabase | None) -> float:
    """Offset [ADU] to remove from a FLAT master at use time (see calib_masters).

    CONFIG-ONE-2026-10 (G01-34): the run resolves the offsets once with its own database
    (``flat_offset_for_master``); a call without a database is an error, it never opens the database of a
    freshly read config.json (workers used to do that, so the result depended on the worker count)."""
    if hdr.get("VYOFFSUB") is not None:
        return 0.0
    if db is None:
        raise ValueError(f"flat offset of {path.name}: no database (resolve it in the run with flat_offset_for_master)")
    import calib_library as _cl  # noqa: PLC0415
    import calib_masters as _cm  # noqa: PLC0415

    row = _cl.get_master_by_path(db, path)
    return float(_cm.flat_offset_to_subtract(hdr, row))


def flat_offset_for_master(path: str | Path, db: VyvarDatabase | None) -> float:
    """The offset ``get_processed_master`` removes from this FLAT at use (SPEED-2026-10: resolved
    once in the run process, so calibration workers never open the database)."""
    with fits.open(Path(path), memmap=False) as hdul:
        hdr = hdul[0].header.copy()
    return float(_flat_offset_at_use(Path(path), hdr, db))


def check_flat_offset_removable(flat_median: float, offset_adu: float, name: str) -> None:
    """G01-02: refuse to subtract a pedestal from a flat that is not in raw ADU.

    A raw-ADU flat contains the pedestal P in every pixel plus the illumination signal S = median - P,
    and a flat is exposed far above its pedestal (S >> P). When the flat's own signal does not exceed
    the pedestal (median - P <= P, e.g. a normalised flat with median ~1 and P = 24.5 ADU), the flat is
    normalised or already offset-corrected and subtracting P would destroy it - stop with a message
    instead (no silent replacement of the flat).
    """
    m = float(flat_median)
    p = float(offset_adu)
    if not np.isfinite(m) or (m - p) <= p:
        raise MasterResamplingError(
            f"FLAT {name}: median {m:.6g} is not above twice the offset {p:.6g} ADU that should be "
            "removed (its signal does not exceed the pedestal) - the flat looks normalised or already "
            "offset-corrected. Mark its offset as already removed in the Calibration Library."
        )


#: G01-29 (decision A2, IRAF ccdmask method): a flat pixel is bad when its normalised value deviates
#: from the level predicted by its neighbours by more than the flat's own robust pixel noise times the
#: false-alarm threshold. The box is the smallest odd square in which a dead column through or next to
#: the pixel touches a minority of the 12 neighbour pairs (2 or 5), so the local median is the surrounding
#: response; smooth structures (vignetting, dust donuts tens of pixels wide) cancel and only pixel-scale
#: defects remain.
FLAT_BAD_LOCAL_BOX_PX = 5
#: Expected number of normal pixels marked bad per flat (Gaussian tail of the flat noise); the
#: threshold in sigma follows from the number of pixels: isf(FALSE_ALARM / (2 N)).
FLAT_BAD_FALSE_ALARM_PIXELS = 1.0


def flat_bad_pixel_threshold_sigma(n_pixels: int) -> float:
    """Two-sided Gaussian threshold with FLAT_BAD_FALSE_ALARM_PIXELS expected false marks in ``n_pixels``."""
    from scipy.stats import norm  # noqa: PLC0415

    n = max(1, int(n_pixels))
    return float(norm.isf(FLAT_BAD_FALSE_ALARM_PIXELS / (2.0 * n)))


def _local_pair_median(a: np.ndarray, half: int) -> np.ndarray:
    """Median over the FLAT_BAD_LOCAL_BOX_PX box of the means of opposite neighbour pairs (d, -d).

    A pair mean cancels a linear gradient exactly; the median of the 12 pair means rejects a defect that
    touches a minority of the pairs (a dead pixel: 1, a dead column through or next to the pixel: 2 or 5).
    The pixel itself is not used. Rows/columns closer than ``half`` to the border use a reflected border
    and are not tested.
    """
    p = np.pad(np.asarray(a, dtype=np.float32), half, mode="reflect")
    h, w = a.shape
    offs = [(dy, dx) for dy in range(0, half + 1) for dx in range(-half, half + 1) if (dy, dx) > (0, 0)]
    out = np.empty((h, w), dtype=np.float32)
    rows = max(1, (1 << 24) // max(1, len(offs) * w))  # ~64 MB of float32 per block
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        stack = np.empty((len(offs), y1 - y0, w), dtype=np.float32)
        for i, (dy, dx) in enumerate(offs):
            stack[i] = 0.5 * (p[half + y0 + dy: half + y1 + dy, half + dx: half + dx + w]
                              + p[half + y0 - dy: half + y1 - dy, half - dx: half - dx + w])
        out[y0:y1] = np.median(stack, axis=0)
    return out


def _robust_sigma(x: np.ndarray) -> float:
    """1.4826 x MAD; NaN when there is no measurable noise (empty, or 0). A noise below the float32
    resolution of a ratio is rounding, not pixel noise, so the result is at least float32 eps."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    s_ = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    if not np.isfinite(s_) or s_ <= 0:
        return float("nan")
    return max(s_, float(np.finfo(np.float32).eps))


def flat_bad_pixel_mask(flat_norm: np.ndarray, *, cfa: bool = False) -> np.ndarray:
    """Bad pixels of a normalised flat (True = bad): non-finite or <= 0, or a local outlier.

    Local outlier: |flat / ref - 1| > t * sigma, where ``ref`` is :func:`_local_pair_median` over the
    FLAT_BAD_LOCAL_BOX_PX box (the pixel itself excluded) plus the local vignetting curvature it leaves
    (smoothed over the non-defect pixels); tested only where the box lies inside the flat - border pixels
    are bad only when invalid; sigma is 1.4826 x MAD of that ratio over the
    valid pixels (the flat's own pixel noise: PRNU + shot noise); t = :func:`flat_bad_pixel_threshold_sigma`
    for the number of pixels of the flat. ``cfa``: one-shot-colour mosaic - each of the four CFA planes is
    tested against neighbours of its own colour. A flat without measurable pixel noise (sigma 0 or below the
    float32 resolution, e.g. a synthetic flat) has no statistical outliers; only invalid pixels are bad.
    """
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    f = np.asarray(flat_norm, dtype=np.float32)
    bad = ~np.isfinite(f) | (f <= 0)
    planes = [(slice(dy, None, 2), slice(dx, None, 2)) for dy in (0, 1) for dx in (0, 1)] if cfa else [
        (slice(None), slice(None))
    ]
    half = FLAT_BAD_LOCAL_BOX_PX // 2
    t = flat_bad_pixel_threshold_sigma(int(f.size))  # false alarms counted over the whole flat
    for sl in planes:
        p = f[sl]
        good = ~bad[sl]
        if int(np.count_nonzero(good)) < FLAT_BAD_LOCAL_BOX_PX * FLAT_BAD_LOCAL_BOX_PX or min(p.shape) <= 2 * half:
            continue
        fill = np.where(good, p, np.float32(np.median(p[good]))).astype(np.float32)
        # only pixels with the full symmetric neighbourhood can be tested
        inner = np.zeros(p.shape, dtype=bool)
        inner[half:-half, half:-half] = True
        ref0 = _local_pair_median(fill, half)
        # the pair means leave the curvature of the vignetting (slowly varying): its local mean (Gaussian of
        # the box scale) is added back to the reference - first over all testable pixels, then again
        # without the pixels that this first test marks (defects must not bend the curvature model)
        wgt = (good & inner).astype(np.float32)
        r = np.full(p.shape, np.nan, dtype=np.float32)
        sigma = float("nan")
        for _pass in range(2):
            with np.errstate(divide="ignore", invalid="ignore"):
                curv = (gaussian_filter((fill - ref0) * wgt, sigma=FLAT_BAD_LOCAL_BOX_PX, mode="reflect")
                        / gaussian_filter(wgt, sigma=FLAT_BAD_LOCAL_BOX_PX, mode="reflect"))
                r = fill / (ref0 + np.nan_to_num(curv, nan=0.0)) - np.float32(1.0)
            sigma = _robust_sigma(r[good & inner])
            if not np.isfinite(sigma):
                break
            wgt = (good & inner & (np.abs(r) <= t * sigma)).astype(np.float32)
        if not np.isfinite(sigma):
            continue  # no measurable pixel noise: no statistical outliers
        sub = bad[sl]
        sub |= good & inner & ~(np.abs(r) <= t * sigma)
        bad[sl] = sub
    return bad


def prepare_flat_for_division(
    pmf: ProcessedMasterResult, *, cfa: bool, name: str
) -> tuple[np.ndarray, float, int]:
    """The one flat-apply preparation (G01-03, G01-29): normalise first, then mark bad pixels NaN.

    Returns ``(flat_norm, median_adu_before_norm, n_bad)``. A flat normalised by
    :func:`get_processed_master` (VYFLNRD) is used as it is; any other flat is divided by the median of
    its valid pixels. Bad pixels (:func:`flat_bad_pixel_mask`) become NaN, so the division leaves NaN in
    the calibrated frame (missing data) instead of dividing by a floored value. There is no floor.
    """
    flat = np.asarray(pmf.data, dtype=np.float32)
    valid = np.isfinite(flat) & (flat > 0)
    # the median is meaningless when most pixels are invalid (breakdown point 50 %)
    if int(np.count_nonzero(valid)) * 2 <= flat.size:
        raise MasterResamplingError(
            f"FLAT {name}: {flat.size - int(np.count_nonzero(valid))} of {flat.size} pixels are <= 0 or "
            "invalid - not a usable flat."
        )
    import calib_masters as _cm  # noqa: PLC0415  (G01-23: the one flat normaliser)

    if pmf.flat_normalized_at_calibrate:
        fm = pmf.flat_median_adu_before_norm
        if fm is None or not np.isfinite(fm) or fm <= 0:
            fm = _cm.flat_valid_median(flat)
    else:
        flat, (fm,) = _cm.normalise_flat(flat, None)
    bad = flat_bad_pixel_mask(flat, cfa=cfa)
    n_bad = int(np.count_nonzero(bad))
    if n_bad:
        flat = np.where(bad, np.float32(np.nan), flat).astype(np.float32)
        log_event(
            f"Master flat {name}: {n_bad} bad pixels ({100.0 * n_bad / flat.size:.4f} %) -> NaN in the "
            "calibrated frames (stars whose aperture touches them are flagged, not filled)."
        )
    return flat, float(fm), n_bad


def get_processed_master(
    master_path: str | Path,
    target_binning: int,
    *,
    kind: Literal["dark", "flat"],
    master_binning: int | None = None,
    light_shape: tuple[int, int] | None = None,
    light_filename: str = "",
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
    dark_resample_mode: DarkResampleMode | None = None,
    flat_offset_adu: float | None = None,
) -> ProcessedMasterResult:
    """Load a master FITS and optionally bin it to match the light's ``XBINNING``.

    - **dark** - sum or mean over each bin block (``dark_resample_mode``, measured by CAL-DIAG; required
      when the dark must be rebinned).
    - **flat** - mean over each bin block (spatial resample), then **median normalization** to ~1
      **after** resample when ``VYFLNRD=1`` on disk (new saves). Legacy flats normalize in pipeline.

    Pass a positive ``master_binning`` to override the master's own binning; ``None`` (production,
    CAL-2026-10) reads ``XBINNING`` from the master FITS header.

    If ``light_shape`` is given, the resampled array must match exactly or
    :class:`MasterResamplingError` is raised.
    """
    p = Path(master_path)
    if not p.is_file():
        raise MasterResamplingError(f"Master file does not exist: {p}")

    with fits.open(p, memmap=False) as hdul:
        hdr = hdul[0].header
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        if master_binning is not None and int(master_binning) > 0:
            mb = max(1, int(master_binning))
        else:
            mb = read_master_binning_from_header(hdr)

    flat_offset_removed_adu = 0.0
    if kind == "flat":
        # CAL-2026-10 spec 3: legacy / unknown-offset flats get the measured constant
        # offset removed here (median commutes with a constant), before normalisation.
        flat_offset_removed_adu = (
            float(flat_offset_adu) if flat_offset_adu is not None else _flat_offset_at_use(p, hdr, db)
        )
        if flat_offset_removed_adu:
            check_flat_offset_removable(float(np.nanmedian(data)), flat_offset_removed_adu, p.name)
            data = (data - np.float32(flat_offset_removed_adu)).astype(np.float32)
            log_event(
                f"Master flat {p.name}: offset {flat_offset_removed_adu:.2f} ADU removed at use "
                "(flat built without offset subtraction)."
            )

    # G01-25: the binning is the plan's / header's (one authority); it is never re-inferred from the array
    # shape (a half-chip bin-1 ROI looked like bin 2 and got a 2x2-summed dark). A shape mismatch stops.
    eff_lb = int(target_binning)

    _drm: DarkResampleMode | None = dark_resample_mode if kind == "dark" else None
    out, bf = resample_master_to_light_binning(
        data,
        master_binning=mb,
        light_binning=int(eff_lb),
        kind=kind,
        dark_resample_mode=_drm,
    )

    # INV-FLUX-01: dark block-sum downscale must conserve total flux (FAIL-CLOSED).
    if kind == "dark" and bf > 1 and _drm == "sum":
        from invariants_runtime import check_dark_resample_flux_conservation  # noqa: PLC0415
        from invariants_runtime import inv_check  # noqa: PLC0415

        _ok_fx, _det_fx = check_dark_resample_flux_conservation(
            data, out, block_factor=int(bf), mode="sum"
        )
        inv_check({"invariants": []}, "INV-FLUX-01", _ok_fx, policy="FAIL", detail=_det_fx)

    if light_shape is not None:
        eh, ew = int(light_shape[0]), int(light_shape[1])
        if out.shape[0] != eh or out.shape[1] != ew:
            raise MasterResamplingError(
                f"After resampling {mb}x{mb} -> {eff_lb}x{eff_lb}, master ({p.name}) has shape "
                f"{out.shape[0]}x{out.shape[1]}, but light '{light_filename or p.name}' is {eh}x{ew} "
                "(binning from the headers; a sub-frame / ROI or a wrong XBINNING does not match) - "
                "stopping calibration."
            )

    flat_median_adu_before_norm: float | None = None
    flat_normalized_at_calibrate = False
    import calib_masters as _cm  # noqa: PLC0415  (G01-23: one valid-pixel median rule)

    if kind == "flat" and _flat_saved_unnormalized(hdr):
        fm_pre = _cm.flat_valid_median(out)
        hdr_norm = hdr.copy()
        out = normalize_flat_master(
            out,
            hdr_norm,
            db=db,
            id_equipments=id_equipments,
            bin_factor=max(1, int(eff_lb)),
        )
        flat_median_adu_before_norm = fm_pre
        flat_normalized_at_calibrate = True
        # INV-FLUX-02: normalized flat median ~ 1.0 (FAIL-CLOSED; matches normalize_flat_master).
        from invariants_runtime import check_flat_mean_near_one  # noqa: PLC0415
        from invariants_runtime import inv_check  # noqa: PLC0415

        _ok_fl, _det_fl = check_flat_mean_near_one(out)
        inv_check({"invariants": []}, "INV-FLUX-02", _ok_fl, policy="FAIL", detail=_det_fl)
    elif kind == "flat":
        flat_median_adu_before_norm = _cm.flat_valid_median(out)

    return ProcessedMasterResult(
        data=out,
        master_binning=mb,
        resampled=bf > 1,
        block_factor=bf,
        flat_median_adu_before_norm=flat_median_adu_before_norm if kind == "flat" else None,
        flat_normalized_at_calibrate=flat_normalized_at_calibrate if kind == "flat" else False,
    )


def _norm_light_path_key(p: Path) -> str:
    try:
        return str(p.resolve()).casefold()
    except OSError:
        return str(p).casefold()


def filter_light_paths_for_calibration_db(
    paths: list[Path],
    *,
    database_path: Path | str,
    draft_id: int | None = None,
    observation_id: str | None = None,
) -> list[Path]:
    """Restrict light paths to ``manifest files[]`` rows with ``IS_REJECTED`` 0 or NULL.

    Runs the equivalent of filtering on ``SELECT * FROM manifest files[] WHERE IS_REJECTED = 0``
    (plus NULL) for the active draft or observation. If the database has **no** ``manifest files[]``
    rows for that scope, ``paths`` are returned unchanged (backward compatible). G01-39: rows that exist
    but are all rejected select no frame (empty list), never every frame.
    """
    from database import VyvarDatabase

    dbp = Path(database_path)
    if not dbp.is_file() or not paths:
        return paths
    db = VyvarDatabase(dbp)
    try:
        accepted: set[str] | None = None
        if draft_id is not None and db.count_obs_files_for_draft(int(draft_id)) > 0:
            accepted = db.fetch_light_file_paths_not_rejected_for_draft(int(draft_id))
        elif observation_id and db.count_obs_files_for_observation(str(observation_id)) > 0:
            accepted = db.fetch_light_file_paths_not_rejected_for_observation(str(observation_id))
        if accepted is None:
            return paths
        out = [p for p in paths if _norm_light_path_key(Path(p)) in accepted]
        if not out:
            log_event(
                f"Calibration: none of the {len(paths)} LIGHT(s) is accepted in the draft manifest "
                "(all rejected) - nothing to calibrate"
            )
        return out
    finally:
        db.conn.close()
