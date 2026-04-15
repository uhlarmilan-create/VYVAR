from __future__ import annotations

"""Master calibration resampling (software binning) for VYVAR.

**CalibrationLibrary** convention: master dark / flat FITS are stored as a **native** stack of
calibration frames (no software bin-down to match light binning). Prefer sensor **1×1** readout for
darks/flats; header ``XBINNING``/``YBINNING`` reflect the calibration frames.

At **calibrate time**, masters are resampled in RAM (temporary arrays) to match each light’s
``XBINNING`` and image shape:

- **Dark** — **sum** over each ``block_factor``×``block_factor`` block (charge-like quantity).
- **Flat** — **arithmetic mean** over each block only (e.g. four pixels for 1×1 master → 2×2 light).
  Spatial re-binning must **not** use sum or median on the flat; mean preserves relative illumination.
  Median normalization to ~1 is applied **after** resample for new library flats marked ``VYFLNRD=1``;
  legacy flats were normalized at stack time and are only resampled here.

Then ``(light − dark) / flat`` runs at matching resolution.
"""

# Native binning assumed for masters **read from CalibrationLibrary** when matching to lights
# (overrides misleading FITS headers; set to 1 for full-chip masters).
CALIBRATION_LIBRARY_NATIVE_BINNING: int = 1

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from astropy.io import fits

from database import VyvarDatabase
from infolog import log_event


class MasterResamplingError(ValueError):
    """Raised when master binning cannot be matched to the light frame."""


class ProcessedMasterResult(NamedTuple):
    """Output of :func:`get_processed_master`."""

    data: np.ndarray
    master_binning: int
    resampled: bool
    block_factor: int
    is_passthrough: bool
    #: Median ADU of flat **before** :func:`normalize_flat_master` at calibrate (new VYFLNRD masters only).
    flat_median_adu_before_norm: float | None = None
    #: True if flat was normalized in RAM here (after resample); legacy flats stay False (pipeline divides).
    flat_normalized_at_calibrate: bool = False


def _parse_master_header_datetime(raw: object) -> datetime | None:
    if raw in (None, ""):
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    txt = txt.replace("Z", "+00:00")
    for cand in (txt, txt.replace(" ", "T")):
        try:
            dt = datetime.fromisoformat(cand)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(txt, fmt).replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    return None


def get_master_age_days(file_path: str | Path) -> float:
    """Age of master in days from header date or filesystem mtime."""
    p = Path(file_path)
    now = datetime.now(timezone.utc)
    dt: datetime | None = None
    if p.is_file():
        try:
            with fits.open(p, memmap=True) as hdul:
                hdr = hdul[0].header
            dt = _parse_master_header_datetime(hdr.get("VY_CDATE") or hdr.get("DATE-OBS") or hdr.get("DATEOBS"))
        except Exception:  # noqa: BLE001
            dt = None
    if dt is None:
        try:
            ts = os.path.getmtime(p)
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        except OSError:
            dt = now
    age = (now - dt).total_seconds() / 86400.0
    return max(0.0, float(age))


def read_master_binning_from_header(header: fits.Header) -> int:
    """Return X axis binning recorded in a master (or light) primary header."""
    raw = header.get("XBINNING", header.get("BINNING", 1))
    try:
        x = int(raw) if raw is not None else 1
    except (TypeError, ValueError):
        x = 1
    return max(1, x)


def read_master_binning_from_fits(path: str | Path) -> int:
    """Read ``XBINNING`` from the primary HDU without loading the full image array."""
    p = Path(path)
    with fits.open(p, memmap=True) as hdul:
        return read_master_binning_from_header(hdul[0].header)


def infer_spatial_block_factor(master_shape: tuple[int, int], light_shape: tuple[int, int]) -> int | None:
    """Infer integer downscale factor ``k`` when master is ~k×``light`` in each axis (after edge crop).

    Handles common mismatches (e.g. 1×1 master flat 2795×4164 vs 2×2 light 1397×2082) when FITS
    ``XBINNING`` on the light is missing or 1. Returns ``None`` when shapes already match or cannot infer.
    """
    mh, mw = int(master_shape[0]), int(master_shape[1])
    lh, lw = int(light_shape[0]), int(light_shape[1])
    if lh <= 0 or lw <= 0:
        return None
    if mh == lh and mw == lw:
        return None
    rh = mh // lh
    rw = mw // lw
    if rh != rw or rh < 2:
        return None
    if mh < rh * lh or mw < rw * lw:
        return None
    if mh - rh * lh >= rh or mw - rw * lw >= rw:
        return None
    return int(rh)


def infer_spatial_upscale_factor(master_shape: tuple[int, int], light_shape: tuple[int, int]) -> int | None:
    """Return integer ``k`` when light is ~``k``×``k`` master pixels (hardware / binned master vs binned light).

    Used to expand a binned master (e.g. 698×1041) to the light grid (e.g. 1397×2082) with ``k``=2.
    Allows up to ``k−1`` rows/cols padding on the light vs ``k``×master (trim mismatch).
    """
    mh, mw = int(master_shape[0]), int(master_shape[1])
    lh, lw = int(light_shape[0]), int(light_shape[1])
    if mh <= 0 or mw <= 0 or lh <= 0 or lw <= 0:
        return None
    if mh >= lh and mw >= lw:
        return None
    rh = lh / float(mh)
    rw = lw / float(mw)
    if abs(rh - rw) > 0.08:
        return None
    k = int(round((rh + rw) / 2.0))
    if k < 2:
        return None
    if lh < mh * k or lw < mw * k:
        return None
    if lh > mh * k + k - 1 or lw > mw * k + k - 1:
        return None
    return int(k)


def align_resampled_master_to_light_shape(
    out: np.ndarray,
    light_shape: tuple[int, int],
    *,
    kind: Literal["dark", "flat"],
) -> np.ndarray | None:
    """If resampled master is an integer factor smaller than the light (each axis), expand by ``kron`` then crop/pad.

    Each master pixel is replicated to a k×k block (same as constant bias/gain per hardware superpixel).
    This is **not** the same as :func:`resample_master_to_light_binning` (no block mean/sum/median here).
    Returns ``None`` if alignment is not possible.
    """
    eh, ew = int(light_shape[0]), int(light_shape[1])
    a = np.asarray(out, dtype=np.float32)
    if a.ndim != 2:
        return None
    oh, ow = int(a.shape[0]), int(a.shape[1])
    if (oh, ow) == (eh, ew):
        return a
    k = infer_spatial_upscale_factor((oh, ow), (eh, ew))
    if k is None:
        return None
    exp = np.kron(a, np.ones((k, k), dtype=np.float32))
    if exp.shape[0] > eh or exp.shape[1] > ew:
        exp = exp[:eh, :ew]
    if exp.shape[0] < eh or exp.shape[1] < ew:
        pad_h = eh - exp.shape[0]
        pad_w = ew - exp.shape[1]
        exp = np.pad(exp, ((0, pad_h), (0, pad_w)), mode="edge")
    if exp.shape != (eh, ew):
        return None
    _ = kind
    return exp


def resample_master_to_light_binning(
    data: np.ndarray,
    *,
    master_binning: int,
    light_binning: int,
    kind: Literal["dark", "flat"],
) -> tuple[np.ndarray, int]:
    """Bin a 2D master down so one output pixel matches ``light_binning`` / ``master_binning``.

    - **Dark:** ``np.sum`` over each *block_factor*×*block_factor* block (axes 1 and 3 after reshape).
    - **Flat:** ``np.mean`` over each block — **only** the plain average of those pixels (e.g. four
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
            f"Light binning {lb}× je menšia ako master binning {mb}× — upsampling nie je podporovaný."
        )
    if lb % mb != 0:
        raise MasterResamplingError(
            f"Light binning {lb} musí byť celočíselný násobok master binningu {mb}."
        )
    bf = lb // mb
    if bf == 1:
        return np.asarray(data, dtype=np.float32), 1

    a = np.asarray(data, dtype=np.float32)
    if a.ndim != 2:
        raise MasterResamplingError(f"Očakávaný 2D master, dostal som tvar {a.shape}.")

    h, w = int(a.shape[0]), int(a.shape[1])
    h_trim = (h // bf) * bf
    w_trim = (w // bf) * bf
    if h_trim < bf or w_trim < bf:
        raise MasterResamplingError(
            f"Rozmer masteru {h}×{w} je po prispôsobení bloku {bf}×{bf} príliš malý."
        )
    if h_trim != h or w_trim != w:
        a = a[:h_trim, :w_trim]
        h, w = h_trim, w_trim

    v = a.reshape(h // bf, bf, w // bf, bf)
    if kind == "dark":
        out = np.sum(v, axis=(1, 3)).astype(np.float32, copy=False)
    else:
        # Flat: block **mean** only — not sum (would skew flat field) and not median (not a block average).
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
    except Exception:  # noqa: BLE001
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
    if m.ndim != 2:
        gmed = float(np.nanmedian(m))
        if np.isfinite(gmed) and abs(gmed) > 0:
            m = (m / gmed).astype(np.float32)
        log_event(
            f"Master flat: globálny medián (normalizačný faktor): {gmed:.6g} ADU "
            f"(ne-2D dáta, len globálna normalizácia)."
        )
        header["VYFLTNRM"] = ("GLOBAL", "Master flat normalization mode")
        if np.isfinite(gmed):
            header["VYFLTMD"] = (float(gmed), "Global median ADU before normalization")
        bad = (m <= 0) | ~np.isfinite(m)
        if np.any(bad):
            log_event(f"Master flat: {int(np.sum(bad))} pixelov nastavených na 1.0 (≤0 alebo neplatné).")
        m = np.where(bad, np.float32(1.0), m).astype(np.float32)
        mean_a = float(np.nanmean(m))
        log_event(f"Master flat: po normalizácii priemer ≈ {mean_a:.4f} (cieľ ~1.0).")
        return m

    h, w = int(m.shape[0]), int(m.shape[1])
    hdr_pat = _valid_bayer_pattern_4(str(header.get("BAYERPAT") or ""))
    assumed_pat = False
    pat = hdr_pat
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
            "Master flat: po binovaní sa používa globálna normalizácia (binovanie zmieša Bayer vrstvy)."
        )

    if use_bayer and pat is not None:
        m = m[: h - (h % 2), : w - (w % 2)]
        slices = (
            m[0::2, 0::2],
            m[0::2, 1::2],
            m[1::2, 0::2],
            m[1::2, 1::2],
        )
        labels = (f"[0,0]={pat[0]}", f"[0,1]={pat[1]}", f"[1,0]={pat[2]}", f"[1,1]={pat[3]}")
        medians: list[float] = []
        for sl, lab in zip(slices, labels):
            mv = float(np.nanmedian(sl))
            medians.append(mv)
            if np.isfinite(mv) and mv > 0:
                sl /= mv
            else:
                log_event(f"Master flat: varovanie — neplatný medián pre dlaždicu {lab}, preskočené delenie.")
        note = f"Bayer {pat}" + (" (predvolené RGGB z EQUIPMENTS)" if assumed_pat else "")
        log_event(
            "Master flat: normalizácia po Bayer dlaždiciach "
            f"({note}); mediány ADU pred norm: "
            f"{labels[0]}={medians[0]:.6g}, {labels[1]}={medians[1]:.6g}, "
            f"{labels[2]}={medians[2]:.6g}, {labels[3]}={medians[3]:.6g}."
        )
        header["VYFLTNRM"] = ("BAYER4", "Per-tile median normalization (OSC/Bayer)")
        header["VYFLTPAT"] = (pat, "Bayer pattern used for flat normalization")
        gpost = float(np.nanmedian(m))
        if np.isfinite(gpost):
            header["VYFLTMD"] = (gpost, "Global median ADU after per-tile norm")
    else:
        gmed = float(np.nanmedian(m))
        if np.isfinite(gmed) and abs(gmed) > 0:
            m = (m / gmed).astype(np.float32)
        else:
            log_event("Master flat: varovanie — globálny medián neplatný, normalizácia preskočená.")
        log_event(f"Master flat: globálny medián (normalizačný faktor): {gmed:.6g} ADU.")
        header["VYFLTNRM"] = ("GLOBAL", "Global median flat normalization")
        if np.isfinite(gmed):
            header["VYFLTMD"] = (float(gmed), "Global median ADU before normalization")

    bad = (m <= 0) | ~np.isfinite(m)
    if np.any(bad):
        log_event(f"Master flat: {int(np.sum(bad))} pixelov nastavených na 1.0 (≤0 alebo neplatné).")
    m = np.where(bad, np.float32(1.0), m).astype(np.float32)

    gfin = float(np.nanmedian(m))
    if np.isfinite(gfin) and abs(gfin - 1.0) > 0.02 and abs(gfin) > 0:
        m = (m / gfin).astype(np.float32)
        log_event(f"Master flat: dodatočná škála 1/{gfin:.4f} aby globálny medián ≈ 1.0.")
    mean_a = float(np.nanmean(m))
    log_event(f"Master flat: po normalizácii medián={float(np.nanmedian(m)):.4f}, priemer ≈ {mean_a:.4f}.")
    return m


def get_processed_master(
    master_path: str | Path,
    target_binning: int,
    *,
    kind: Literal["dark", "flat"],
    master_binning: int | None = None,
    light_shape: tuple[int, int] | None = None,
    light_filename: str = "",
    allow_passthrough: bool = False,
    db: VyvarDatabase | None = None,
    id_equipments: int | None = None,
) -> ProcessedMasterResult:
    """Load a master FITS and optionally bin it to match the light's ``XBINNING``.

    - **dark** — sum over each bin block (total charge / dark current preserved).
    - **flat** — mean over each bin block (spatial resample), then **median normalization** to ~1
      **after** resample when ``VYFLNRD=1`` on disk (new saves). Legacy flats normalize in pipeline.

    Pass ``master_binning=CALIBRATION_LIBRARY_NATIVE_BINNING`` (or a positive int) when applying
    CalibrationLibrary masters so resampling follows library convention. Pass ``master_binning=None``
    to read ``XBINNING`` from the master FITS header.

    If ``light_shape`` is given, the resampled array must match exactly or
    :class:`MasterResamplingError` is raised.
    """
    p = Path(master_path)
    if not p.is_file():
        if allow_passthrough and light_shape is not None:
            eh, ew = int(light_shape[0]), int(light_shape[1])
            base = np.zeros((eh, ew), dtype=np.float32) if kind == "dark" else np.ones((eh, ew), dtype=np.float32)
            return ProcessedMasterResult(
                data=base,
                master_binning=max(1, int(target_binning)),
                resampled=False,
                block_factor=1,
                is_passthrough=True,
                flat_median_adu_before_norm=1.0 if kind == "flat" else None,
                flat_normalized_at_calibrate=False,
            )
        raise MasterResamplingError(f"Master súbor neexistuje: {p}")

    with fits.open(p, memmap=False) as hdul:
        hdr = hdul[0].header
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        if master_binning is not None and int(master_binning) > 0:
            mb = max(1, int(master_binning))
        else:
            mb = read_master_binning_from_header(hdr)

    eff_lb = int(target_binning)
    if light_shape is not None:
        sp = infer_spatial_block_factor(data.shape, light_shape)
        if sp is not None:
            eff_lb = max(eff_lb, mb * sp)

    out, bf = resample_master_to_light_binning(
        data, master_binning=mb, light_binning=int(eff_lb), kind=kind
    )

    if light_shape is not None:
        eh, ew = int(light_shape[0]), int(light_shape[1])
        if out.shape[0] != eh or out.shape[1] != ew:
            aligned = align_resampled_master_to_light_shape(out, (eh, ew), kind=kind)
            if aligned is not None:
                out = aligned
            else:
                raise MasterResamplingError(
                    f"Po resamplingu má master ({p.name}) tvar {out.shape[0]}×{out.shape[1]}, "
                    f"ale light „{light_filename or p.name}“ má {eh}×{ew} — kalibráciu zastavujem."
                )

    flat_median_adu_before_norm: float | None = None
    flat_normalized_at_calibrate = False
    if kind == "flat" and _flat_saved_unnormalized(hdr):
        fm_pre = float(np.nanmedian(out))
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
    elif kind == "flat":
        flat_median_adu_before_norm = float(np.nanmedian(out))

    return ProcessedMasterResult(
        data=out,
        master_binning=mb,
        resampled=bf > 1,
        block_factor=bf,
        is_passthrough=False,
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
    """Restrict light paths to ``OBS_FILES`` rows with ``IS_REJECTED`` 0 or NULL.

    Runs the equivalent of filtering on ``SELECT * FROM OBS_FILES WHERE IS_REJECTED = 0``
    (plus NULL) for the active draft or observation. If the database has **no** ``OBS_FILES``
    rows for that scope, ``paths`` are returned unchanged (backward compatible).
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
        if not accepted:
            return paths
        return [p for p in paths if _norm_light_path_key(Path(p)) in accepted]
    finally:
        db.conn.close()
