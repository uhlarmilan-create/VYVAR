"""CAL-2026-10 master building and import (dev/results/specs/VYVAR_CALIBRATION_SPEC.md section 3).

- DARK / DARKFLAT: per-pixel median (default) or mean of raw frames. No clipping.
- FLAT: each raw flat -> subtract offset (DARKFLAT master, else constant P from darks,
  else nothing) -> normalise by its own median (OSC: per CFA channel) -> combine ->
  normalise. A long-exposure dark is never subtracted from a flat.
- Import: an external master is copied into the library and linked to a camera
  (and telescope for FLAT) with origin IMPORTED + software name.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from astropy.io import fits

import calib_library as cl
from infolog import log_event

COMBINE_METHODS = ("median", "mean")
DEFAULT_COMBINE = "median"

_FULL_RAM_BYTES = 512 * 1024 * 1024
_SLAB_RAM_BYTES = 256 * 1024 * 1024
_MONO_PATTERNS = {"", "MONO", "NONE", "N/A", "NA"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _f32(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def cfa_pattern(hdr: fits.Header, equipment_mask: str | None = None) -> str | None:
    """Bayer pattern (RGGB, ...) for OSC data, None for mono."""
    for v in (hdr.get("BAYERPAT"), hdr.get("COLORTYP"), equipment_mask):
        s = str(v or "").strip().upper()
        if s and s not in _MONO_PATTERNS and re.fullmatch(r"[RGB]{4}", s):
            return s
    return None


def normalise_flat(a: np.ndarray, pattern: str | None) -> tuple[np.ndarray, list[float]]:
    """Divide by the median (mono) or by each CFA channel's median (OSC, equalised CFA)."""
    out = _f32(a).copy()
    if pattern:
        meds = []
        for dy in (0, 1):
            for dx in (0, 1):
                v = out[dy::2, dx::2]
                m = float(np.nanmedian(v))
                meds.append(m)
                if math.isfinite(m) and m > 0:
                    out[dy::2, dx::2] = v / m
        return out, meds
    m = float(np.nanmedian(out))
    if math.isfinite(m) and m > 0:
        out /= m
    return out, [m]


def combine_frames(
    files: Sequence[Path],
    *,
    method: str,
    preprocess: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, fits.Header]:
    """Per-pixel median or mean (no clipping). Large stacks via a temporary memmap cube."""
    if method not in COMBINE_METHODS:
        raise ValueError(f"combine method must be one of {COMBINE_METHODS}, got {method!r}")
    if not files:
        raise ValueError("no frames to combine")
    with fits.open(files[0], memmap=False) as h:
        hdr0 = h[0].header.copy()
        a0 = _f32(h[0].data)
    if preprocess is not None:
        a0 = _f32(preprocess(a0))
    n, (ny, nx) = len(files), a0.shape
    red = np.nanmedian if method == "median" else np.nanmean

    def _load(i: int) -> np.ndarray:
        with fits.open(files[i], memmap=False) as h:
            a = _f32(h[0].data)
        if a.shape != (ny, nx):
            raise ValueError(f"{Path(files[i]).name}: shape {a.shape} != {(ny, nx)}")
        return _f32(preprocess(a)) if preprocess is not None else a

    if n * ny * nx * 4 <= _FULL_RAM_BYTES:
        cube = np.empty((n, ny, nx), dtype=np.float32)
        cube[0] = a0
        for i in range(1, n):
            cube[i] = _load(i)
        master = red(cube, axis=0).astype(np.float32)
    else:
        fd, tmp = tempfile.mkstemp(prefix="vyvar_master_", suffix=".dat")
        os.close(fd)
        try:
            cube = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(n, ny, nx))
            cube[0] = a0
            for i in range(1, n):
                cube[i] = _load(i)
            cube.flush()
            del cube
            cr = np.memmap(tmp, dtype=np.float32, mode="r", shape=(n, ny, nx))
            master = np.empty((ny, nx), dtype=np.float32)
            th = max(1, int(_SLAB_RAM_BYTES // max(1, n * nx * 4)))
            for y0 in range(0, ny, th):
                master[y0 : y0 + th] = red(np.asarray(cr[:, y0 : y0 + th]), axis=0)
            del cr
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    bad = ~np.isfinite(master)
    if bad.any():
        good = master[~bad]
        master[bad] = float(np.median(good)) if good.size else 0.0
    hdr0["VYSTKMOD"] = (method.upper(), "Per-pixel combine of the stack (no clipping)")
    hdr0["NCOMBINE"] = (n, "Number of frames combined")
    return master, hdr0


def _earliest_capture(files: Sequence[Path]) -> Any:
    dates = cl.iter_light_dates(files)
    return min(dates) if dates else None


def master_filename(kind: str, hdr: fits.Header, *, camera_id: int, capture: Any) -> str:
    exp = cl._hfloat(hdr, "EXPTIME", "EXPOSURE") or 0.0
    temp = cl.header_temp(hdr)
    off = cl.header_offset(hdr)
    parts = [
        {"dark": "Dark", "darkflat": "DarkFlat", "flat": "Flat"}[kind],
        f"{exp:g}s",
    ]
    if kind == "flat":
        parts.append(re.sub(r"[^A-Za-z0-9.+-]", "", cl.header_filter(hdr)) or "NoFilter")
    parts.append(f"{cl.header_gain(hdr)}G")
    if off is not None:
        parts.append(f"O{off}")
    if temp is not None:
        parts.append(f"{temp:g}C")
    parts.append(f"Bin{cl.header_binning(hdr)}")
    parts.append(capture.strftime("%Y%m%d") if capture is not None else "nodate")
    parts.append(f"cam{int(camera_id)}")
    return "_".join(parts) + ".fits"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
@dataclass
class BuildResult:
    master: cl.LibraryMaster | None
    path: Path | None
    messages: list[str] = field(default_factory=list)


def build_dark(
    files: Sequence[Path],
    *,
    kind: str = cl.KIND_DARK,
    method: str = DEFAULT_COMBINE,
    library_root: Path,
    db: Any,
    camera_id: int,
    telescope_id: int | None = None,
) -> BuildResult:
    """DARK or DARKFLAT master from raw darks (all must share exposure/gain/offset/binning)."""
    if kind not in (cl.KIND_DARK, cl.KIND_DARKFLAT):
        raise ValueError("build_dark: kind must be dark or darkflat")
    files = [Path(f) for f in files]
    master, hdr = combine_frames(files, method=method)
    cap = _earliest_capture(files)
    hdr["IMAGETYP"] = ("Dark Frame" if kind == cl.KIND_DARK else "Dark Flat", "VYVAR master")
    hdr["HISTORY"] = f"VYVAR master {kind.upper()} ({method}, {len(files)} frames, no clipping)"
    if cap is not None:
        hdr["VY_CDATE"] = (cap.strftime("%Y-%m-%dT%H:%M:%S"), "Earliest source capture (UTC)")
    out = Path(library_root) / master_filename(kind, hdr, camera_id=camera_id, capture=cap)
    out.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(out, master.astype(np.float32), header=hdr, overwrite=True)
    m = cl.register_master(db, out, kind=kind, camera_id=camera_id, telescope_id=telescope_id,
                           origin=cl.ORIGIN_VYVAR, combine_method=method)
    return BuildResult(m, out, [f"[OK] {kind.upper()} {out.name} ({len(files)} frames, {method})"])


def build_flat(
    files: Sequence[Path],
    *,
    method: str = DEFAULT_COMBINE,
    library_root: Path,
    db: Any,
    camera_id: int,
    telescope_id: int,
    darkflat_path: Path | None = None,
    temp_tol: float = 1.0,
    equipment_bayer_mask: str | None = None,
) -> BuildResult:
    """FLAT master with offset removal: DARKFLAT > constant P from darks > none."""
    files = [Path(f) for f in files]
    msgs: list[str] = []
    with fits.open(files[0], memmap=True) as h:
        hdr0 = h[0].header.copy()
    pattern = cfa_pattern(hdr0, equipment_bayer_mask)
    off_arr: np.ndarray | float = 0.0
    handling = cl.OFFSET_NONE
    off_adu: float | None = None
    if darkflat_path is not None and Path(darkflat_path).is_file():
        with fits.open(darkflat_path, memmap=False) as h:
            off_arr = _f32(h[0].data)
        handling = cl.OFFSET_DARKFLAT
        off_adu = float(np.nanmedian(off_arr))
        msgs.append(f"offset: DARKFLAT {Path(darkflat_path).name} (median {off_adu:.1f} ADU)")
    else:
        est = cl.measure_offset_adu(
            db, camera_id=camera_id, gain=cl.header_gain(hdr0), offset=cl.header_offset(hdr0),
            binning=cl.header_binning(hdr0), temp=cl.header_temp(hdr0), temp_tol=temp_tol,
        )
        if est.adu is not None:
            off_arr, off_adu, handling = float(est.adu), float(est.adu), cl.OFFSET_CONSTANT
            msgs.append(f"offset: constant {est.adu:.1f} ADU ({est.method})")
        else:
            msgs.append("offset: none available (no DARKFLAT, no dark of this camera/gain/binning)")

    def _prep(a: np.ndarray) -> np.ndarray:
        if isinstance(off_arr, np.ndarray) and off_arr.shape != a.shape:
            raise ValueError(f"DARKFLAT shape {off_arr.shape} != flat {a.shape}")
        return normalise_flat(a - off_arr, pattern)[0]

    master, hdr = combine_frames(files, method=method, preprocess=_prep)
    master, _ = normalise_flat(master, pattern)
    bad = ~np.isfinite(master) | (master <= 0)
    master[bad] = 1.0
    cap = _earliest_capture(files)
    hdr["IMAGETYP"] = ("Flat Field", "VYVAR master")
    hdr["VYOFFSUB"] = (handling, "Flat offset removed: darkflat|constant|none")
    if off_adu is not None:
        hdr["VYOFFADU"] = (round(off_adu, 3), "Offset level removed [ADU]")
    hdr["VYFLTNRM"] = ("CFA4" if pattern else "GLOBAL", "Per-frame and master normalisation")
    hdr["VYFLNRD"] = (1, "Renormalised after resample (no-op, ~1)")
    hdr["HISTORY"] = f"VYVAR master FLAT ({method}, {len(files)} frames, offset {handling}, no clipping)"
    if cap is not None:
        hdr["VY_CDATE"] = (cap.strftime("%Y-%m-%dT%H:%M:%S"), "Earliest source capture (UTC)")
    out = Path(library_root) / master_filename(cl.KIND_FLAT, hdr, camera_id=camera_id, capture=cap)
    out.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(out, master.astype(np.float32), header=hdr, overwrite=True)
    m = cl.register_master(db, out, kind=cl.KIND_FLAT, camera_id=camera_id, telescope_id=telescope_id,
                           origin=cl.ORIGIN_VYVAR, combine_method=method,
                           offset_handling=handling, offset_adu=off_adu)
    msgs.insert(0, f"[OK] FLAT {out.name} ({len(files)} frames, {method})")
    log_event("CALIB MASTER: " + " | ".join(msgs))
    return BuildResult(m, out, msgs)


# ---------------------------------------------------------------------------
# import
# ---------------------------------------------------------------------------
def import_master(
    src: str | Path,
    *,
    kind: str,
    library_root: Path,
    db: Any,
    camera_id: int,
    telescope_id: int | None,
    origin_software: str = "",
    flat_offset_removed: bool | None = None,
) -> BuildResult:
    """Copy an externally made master into the library and register it (origin IMPORTED).

    ``flat_offset_removed``: user statement for FLAT (True -> already_removed,
    False -> unknown: the offset is measured from darks and removed at use time;
    None -> read from the header: VYOFFSUB, or HISTORY "dark ... subtracted").
    Re-importing an identical file keeps the existing library copy.
    """
    k = str(kind).lower()
    p = Path(src)
    if not p.is_file():
        return BuildResult(None, None, [f"[X] file not found: {p}"])
    with fits.open(p, memmap=False) as h:
        hdr = h[0].header.copy()
        data = h[0].data
    if data is None or np.asarray(data).ndim != 2:
        return BuildResult(None, None, [f"[X] {p.name}: not a 2-D image"])
    if not np.isfinite(np.asarray(data, dtype=np.float64)).any():
        return BuildResult(None, None, [f"[X] {p.name}: no finite pixels"])
    it = str(hdr.get("IMAGETYP") or hdr.get("FRAME") or "").lower()
    msgs: list[str] = []
    if it and k == cl.KIND_FLAT and "flat" not in it:
        msgs.append(f"! IMAGETYP '{hdr.get('IMAGETYP')}' does not say flat")
    if it and k in (cl.KIND_DARK, cl.KIND_DARKFLAT) and "dark" not in it:
        msgs.append(f"! IMAGETYP '{hdr.get('IMAGETYP')}' does not say dark")
    if k in (cl.KIND_DARK, cl.KIND_DARKFLAT) and cl.header_temp(hdr) is None:
        return BuildResult(None, None, msgs + [f"[X] {p.name}: dark without sensor temperature (CCD-TEMP)"])
    cap = cl.header_capture_utc(hdr)
    if cap is None:
        return BuildResult(None, None, msgs + [f"[X] {p.name}: no capture date (DATE-OBS) - validity cannot be checked"])
    sw = origin_software or cl.header_software(hdr) or "unknown"
    dst = Path(library_root) / ("imported_" + master_filename(k, hdr, camera_id=camera_id, capture=cap))
    dst.parent.mkdir(parents=True, exist_ok=True)
    same = dst.is_file() and dst.stat().st_size == p.stat().st_size and _same_bytes(p, dst)
    if not same:
        shutil.copy2(p, dst)
    oh = None
    if k == cl.KIND_FLAT:
        if flat_offset_removed is None:
            # no user statement: read it from the header (VYOFFSUB / HISTORY dark subtracted)
            oh = cl.header_flat_offset_handling(hdr)
        else:
            oh = cl.OFFSET_ALREADY if flat_offset_removed else cl.OFFSET_UNKNOWN
    m = cl.register_master(db, dst, kind=k, camera_id=camera_id, telescope_id=telescope_id,
                           origin=cl.ORIGIN_IMPORTED, origin_software=sw,
                           combine_method=str(hdr.get("VYSTKMOD") or ""), offset_handling=oh)
    if same:
        msgs.insert(0, f"[OK] {k.upper()} {p.name} already in the library as {dst.name}")
    else:
        msgs.insert(0, f"[OK] imported {k.upper()} {p.name} -> {dst.name} (from {sw})")
    return BuildResult(m, dst, msgs)


def _same_bytes(a: Path, b: Path, chunk: int = 1 << 20) -> bool:
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(chunk), fb.read(chunk)
            if x != y:
                return False
            if not x:
                return True


# ---------------------------------------------------------------------------
# use-time offset correction for legacy / unknown flats (spec 3, last bullet)
# ---------------------------------------------------------------------------
def flat_offset_to_subtract(hdr: fits.Header, row: cl.LibraryMaster | None) -> float:
    """ADU to subtract from a FLAT master before normalisation.

    New VYVAR flats carry VYOFFSUB (offset already handled) -> 0. Flats with unknown
    handling (legacy VYVAR median-of-raw, or imports without statement) -> the
    measured constant offset stored in the library row (median commutes with a
    constant). Already-removed imports -> 0.
    """
    if hdr.get("VYOFFSUB") is not None:
        return 0.0
    if row is None:
        return 0.0
    if row.offset_handling in (cl.OFFSET_UNKNOWN, "") and row.offset_adu is not None:
        return float(row.offset_adu)
    return 0.0
