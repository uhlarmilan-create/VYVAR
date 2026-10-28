"""CAL-2026-10 calibration library: master registry, validity window, matching, offset.

Contract: dev/results/specs/VYVAR_CALIBRATION_SPEC.md (sections 2 and 4). CMOS only: kinds are
DARK, DARKFLAT and FLAT.

Every master row is linked to a camera (EQUIPMENTS); a FLAT also to a telescope.
A master with capture date D and validity V days is valid for a LIGHT dated L when
``D - 1 day <= L <= D + V``; outside that window it is never used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from astropy.io import fits

from infolog import log_event

KIND_DARK = "dark"
KIND_DARKFLAT = "darkflat"
KIND_FLAT = "flat"
KINDS = (KIND_DARK, KIND_DARKFLAT, KIND_FLAT)

ORIGIN_VYVAR = "VYVAR"
ORIGIN_IMPORTED = "IMPORTED"

#: Offset handling recorded for a FLAT (spec section 3).
OFFSET_DARKFLAT = "darkflat"
OFFSET_CONSTANT = "constant"
OFFSET_NONE = "none"
OFFSET_ALREADY = "already_removed"  # imported flat, user states offset was removed
OFFSET_UNKNOWN = "unknown"  # legacy VYVAR flat or import without statement

#: Lower edge of the validity window: flats/darks are often taken the morning after.
VALIDITY_LEAD_DAYS = 1.0

_NEW_COLUMNS: dict[str, str] = {
    "OFFSET": "INTEGER",
    "ORIGIN": "TEXT",
    "ORIGIN_SOFTWARE": "TEXT",
    "COMBINE_METHOD": "TEXT",
    "CAPTURE_DATE": "TEXT",
    "OFFSET_HANDLING": "TEXT",
    "OFFSET_ADU": "REAL",
}


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------
def ensure_schema(db: Any) -> None:
    """Create/migrate CALIBRATION_LIBRARY (idempotent). Legacy rows keep NULLs."""
    db._ensure_calibration_library_table()
    cols = {str(r[1]).upper() for r in db.conn.execute("PRAGMA table_info('CALIBRATION_LIBRARY');")}
    for name, typ in _NEW_COLUMNS.items():
        if name not in cols:
            db.conn.execute(f"ALTER TABLE CALIBRATION_LIBRARY ADD COLUMN {name} {typ};")
    db.conn.commit()


# ---------------------------------------------------------------------------
# header helpers
# ---------------------------------------------------------------------------
def _hfloat(hdr: fits.Header, *keys: str) -> float | None:
    for k in keys:
        v = hdr.get(k)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            return f
    return None


def header_binning(hdr: fits.Header) -> int:
    from utils import fits_binning_xy_from_header  # noqa: PLC0415

    bx, _ = fits_binning_xy_from_header(hdr)
    return max(1, int(bx))


def header_offset(hdr: fits.Header) -> int | None:
    v = _hfloat(hdr, "OFFSET", "PEDESTAL")
    return int(round(v)) if v is not None else None


def header_gain(hdr: fits.Header) -> int:
    v = _hfloat(hdr, "GAIN")
    return int(round(v)) if v is not None else 0


def header_temp(hdr: fits.Header) -> float | None:
    return _hfloat(hdr, "CCD-TEMP", "SENSORTEMP", "SET-TEMP", "CCD_TEMP")


def header_filter(hdr: fits.Header) -> str:
    s = str(hdr.get("FILTER") or hdr.get("FILT") or "").strip()
    if not s or s.lower() in {"unknown", "none", "nan"}:
        return "NoFilter"
    return s


def header_software(hdr: fits.Header) -> str:
    # FITS ORIGIN names the organisation or program that wrote the file
    # (C-Munipack writes "C-Munipack 2.1.34" there).
    for k in ("SWCREATE", "CREATOR", "PROGRAM", "SOFTWARE", "ORIGIN", "SWOWNER"):
        v = str(hdr.get(k) or "").strip()
        if v:
            return v
    return ""


def header_kind(hdr: fits.Header) -> str | None:
    """Master kind from IMAGETYP / FRAME: darkflat, flat, dark or None."""
    it = str(hdr.get("IMAGETYP") or hdr.get("FRAME") or "").lower().replace(" ", "").replace("_", "").replace("-", "")
    if "darkflat" in it or "flatdark" in it:
        return KIND_DARKFLAT
    if "flat" in it:
        return KIND_FLAT
    if "dark" in it:
        return KIND_DARK
    return None


def header_flat_offset_handling(hdr: fits.Header) -> str:
    """How the offset was handled in a FLAT master, read from its header.

    VYVAR flats carry VYOFFSUB. Other programs record dark subtraction in HISTORY
    (C-Munipack: "Dark frame subtracted."); a dark-subtracted flat has its offset
    removed. Anything else is unknown (offset measured from darks, removed at use).
    """
    v = str(hdr.get("VYOFFSUB") or "").strip()
    if v:
        return v
    for k in ("HISTORY", "COMMENT"):
        for line in hdr.get(k, []) or []:
            t = str(line).lower()
            if "dark" in t and "subtract" in t:
                return OFFSET_ALREADY
    return OFFSET_UNKNOWN


def header_capture_utc(hdr: fits.Header) -> datetime | None:

    dt, _ = _header_capture_datetime(hdr)
    if dt is not None and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_iso(s: Any) -> datetime | None:
    if s is None:
        return None
    try:
        dt = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------
@dataclass
class LibraryMaster:
    id: int
    kind: str
    path: str
    binning: int
    exptime: float
    temp: float | None
    filter_name: str
    gain: int
    offset: int | None
    ncombine: int | None
    camera_id: int | None
    telescope_id: int | None
    origin: str
    origin_software: str
    combine_method: str
    capture_utc: datetime | None
    offset_handling: str
    offset_adu: float | None
    camera_name: str = ""
    telescope_name: str = ""

    def valid_until(self, validity_days: float) -> datetime | None:
        if self.capture_utc is None:
            return None
        return self.capture_utc + timedelta(days=float(validity_days))

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["capture_utc"] = self.capture_utc.isoformat() if self.capture_utc else None
        return d


def _row_to_master(row: Any) -> LibraryMaster:
    r = dict(row)
    kind = str(r.get("KIND") or "").lower()
    cap = _parse_iso(r.get("CAPTURE_DATE"))
    if cap is None:
        p = Path(str(r.get("FILE_PATH")))
        if p.is_file():
            try:
                with fits.open(p, memmap=True) as h:
                    cap = header_capture_utc(h[0].header)
            except (OSError, ValueError):
                cap = None
    origin = str(r.get("ORIGIN") or ORIGIN_VYVAR)
    oh = str(r.get("OFFSET_HANDLING") or "")
    if not oh and kind == KIND_FLAT:
        oh = OFFSET_UNKNOWN
    return LibraryMaster(
        id=int(r.get("ID") or 0),
        kind=kind,
        path=str(r.get("FILE_PATH")),
        binning=int(r.get("XBINNING") or 1),
        exptime=float(r.get("EXPTIME") or 0.0),
        temp=(float(r["CCD_TEMP"]) if r.get("CCD_TEMP") is not None else None),
        filter_name=str(r.get("FILTER_NAME") or ""),
        gain=int(r.get("GAIN") or 0),
        offset=(int(r["OFFSET"]) if r.get("OFFSET") is not None else None),
        ncombine=(int(r["NCOMBINE"]) if r.get("NCOMBINE") is not None else None),
        camera_id=(int(r["ID_EQUIPMENTS"]) if r.get("ID_EQUIPMENTS") is not None else None),
        telescope_id=(int(r["ID_TELESCOPE"]) if r.get("ID_TELESCOPE") is not None else None),
        origin=origin,
        origin_software=str(r.get("ORIGIN_SOFTWARE") or ("VYVAR" if origin == ORIGIN_VYVAR else "")),
        combine_method=str(r.get("COMBINE_METHOD") or ""),
        capture_utc=cap,
        offset_handling=oh,
        offset_adu=(float(r["OFFSET_ADU"]) if r.get("OFFSET_ADU") is not None else None),
        camera_name=str(r.get("CAMERA_NAME") or ""),
        telescope_name=str(r.get("TELESCOPE_NAME") or ""),
    )


def list_masters(db: Any, *, kind: str | None = None, camera_id: int | None = None) -> list[LibraryMaster]:
    """All registered masters with camera/telescope names (GUI table)."""
    ensure_schema(db)
    q = (
        "SELECT c.*, e.CAMERANAME AS CAMERA_NAME, t.TELESCOPENAME AS TELESCOPE_NAME "
        "FROM CALIBRATION_LIBRARY c "
        "LEFT JOIN EQUIPMENTS e ON c.ID_EQUIPMENTS = e.ID "
        "LEFT JOIN TELESCOPE t ON c.ID_TELESCOPE = t.ID WHERE 1=1"
    )
    args: list[Any] = []
    if kind:
        q += " AND LOWER(c.KIND) = ?"
        args.append(str(kind).lower())
    if camera_id is not None:
        q += " AND c.ID_EQUIPMENTS = ?"
        args.append(int(camera_id))
    q += " ORDER BY c.ID_EQUIPMENTS, c.KIND, c.XBINNING, c.EXPTIME"
    return [_row_to_master(r) for r in db.conn.execute(q, tuple(args)).fetchall()]


def register_master(
    db: Any,
    path: str | Path,
    *,
    kind: str,
    camera_id: int,
    telescope_id: int | None,
    origin: str = ORIGIN_VYVAR,
    origin_software: str = "",
    combine_method: str = "",
    offset_handling: str | None = None,
    offset_adu: float | None = None,
) -> LibraryMaster:
    """Register (insert or update) one master. Header supplies the physical keys.

    Raises ValueError when a required link is missing (camera always; telescope for FLAT).
    """
    k = str(kind).lower()
    if k not in KINDS:
        raise ValueError(f"unknown master kind {kind!r}; allowed {KINDS}")
    if camera_id is None:
        raise ValueError("a master must be linked to a camera")
    if k == KIND_FLAT and telescope_id is None:
        raise ValueError("a FLAT master must be linked to a telescope")
    ensure_schema(db)
    p = Path(path).resolve()
    with fits.open(p, memmap=True) as h:
        hdr = h[0].header
    temp = header_temp(hdr)
    if k in (KIND_DARK, KIND_DARKFLAT) and temp is None:
        raise ValueError(f"{p.name}: dark master has no sensor temperature in the header")
    exp = _hfloat(hdr, "EXPTIME", "EXPOSURE")
    cap = header_capture_utc(hdr)
    ncomb = _hfloat(hdr, "NCOMBINE")
    flt = header_filter(hdr) if k == KIND_FLAT else ""
    oh = offset_handling
    if k == KIND_FLAT and oh is None:
        oh = str(hdr.get("VYOFFSUB") or OFFSET_UNKNOWN)
    if k == KIND_FLAT and offset_adu is None:
        offset_adu = _hfloat(hdr, "VYOFFADU")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db.conn.execute(
        """
        INSERT INTO CALIBRATION_LIBRARY (
            KIND, FILE_PATH, XBINNING, EXPTIME, CCD_TEMP, FILTER_NAME, GAIN, NCOMBINE,
            REGISTERED_AT, ID_EQUIPMENTS, ID_TELESCOPE, OFFSET, ORIGIN, ORIGIN_SOFTWARE,
            COMBINE_METHOD, CAPTURE_DATE, OFFSET_HANDLING, OFFSET_ADU)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(FILE_PATH) DO UPDATE SET
            KIND=excluded.KIND, XBINNING=excluded.XBINNING, EXPTIME=excluded.EXPTIME,
            CCD_TEMP=excluded.CCD_TEMP, FILTER_NAME=excluded.FILTER_NAME, GAIN=excluded.GAIN,
            NCOMBINE=excluded.NCOMBINE, REGISTERED_AT=excluded.REGISTERED_AT,
            ID_EQUIPMENTS=excluded.ID_EQUIPMENTS, ID_TELESCOPE=excluded.ID_TELESCOPE,
            OFFSET=excluded.OFFSET, ORIGIN=excluded.ORIGIN, ORIGIN_SOFTWARE=excluded.ORIGIN_SOFTWARE,
            COMBINE_METHOD=excluded.COMBINE_METHOD, CAPTURE_DATE=excluded.CAPTURE_DATE,
            OFFSET_HANDLING=excluded.OFFSET_HANDLING, OFFSET_ADU=excluded.OFFSET_ADU;
        """,
        (
            k,
            str(p),
            header_binning(hdr),
            float(exp) if exp is not None else 0.0,
            temp,
            flt,
            header_gain(hdr),
            int(ncomb) if ncomb is not None else None,
            now,
            int(camera_id),
            int(telescope_id) if telescope_id is not None else None,
            header_offset(hdr),
            str(origin),
            str(origin_software or ("VYVAR" if origin == ORIGIN_VYVAR else header_software(hdr))),
            str(combine_method or hdr.get("VYSTKMOD") or ""),
            cap.isoformat() if cap is not None else None,
            oh if k == KIND_FLAT else None,
            float(offset_adu) if offset_adu is not None else None,
        ),
    )
    db.conn.commit()
    row = db.conn.execute("SELECT * FROM CALIBRATION_LIBRARY WHERE FILE_PATH = ?", (str(p),)).fetchone()
    m = _row_to_master(row)
    log_event(
        f"CALIB LIB: registered {k.upper()} {p.name} camera={camera_id} tel={telescope_id} "
        f"origin={m.origin}/{m.origin_software} method={m.combine_method or '?'}"
    )
    return m


def update_flat_offset(db: Any, path: str | Path, *, handling: str, offset_adu: float | None) -> None:
    ensure_schema(db)
    db.conn.execute(
        "UPDATE CALIBRATION_LIBRARY SET OFFSET_HANDLING = ?, OFFSET_ADU = ? WHERE FILE_PATH = ?",
        (str(handling), float(offset_adu) if offset_adu is not None else None, str(Path(path).resolve())),
    )
    db.conn.commit()


def get_master_by_path(db: Any, path: str | Path) -> LibraryMaster | None:
    ensure_schema(db)
    try:
        key = str(Path(path).resolve())
    except OSError:
        key = str(path)
    row = db.conn.execute("SELECT * FROM CALIBRATION_LIBRARY WHERE FILE_PATH = ?", (key,)).fetchone()
    return _row_to_master(row) if row is not None else None


def master_from_header(
    path: str | Path,
    *,
    camera_id: int | None,
    telescope_id: int | None,
    kind: str | None = None,
) -> tuple[LibraryMaster | None, str]:
    """In-memory master (not registered) for a master file found next to the LIGHTs.

    Same physical keys as a library row; origin IMPORTED with the program named in
    the header. Returns (master, reason); master is None when it cannot be used.
    """
    p = Path(path)
    try:
        with fits.open(p, memmap=True) as h:
            hdr = h[0].header.copy()
    except (OSError, ValueError) as exc:
        return None, f"{p.name}: unreadable ({exc})"
    k = kind or header_kind(hdr)
    if k not in KINDS:
        return None, f"{p.name}: IMAGETYP does not say dark / flat"
    temp = header_temp(hdr)
    if k in (KIND_DARK, KIND_DARKFLAT) and temp is None:
        return None, f"{p.name}: dark master without sensor temperature"
    cap = header_capture_utc(hdr)
    if cap is None:
        return None, f"{p.name}: no capture date (DATE-OBS)"
    ncomb = _hfloat(hdr, "NCOMBINE")
    return LibraryMaster(
        id=0,
        kind=k,
        path=str(p),
        binning=header_binning(hdr),
        exptime=float(_hfloat(hdr, "EXPTIME", "EXPOSURE") or 0.0),
        temp=temp,
        filter_name=header_filter(hdr) if k == KIND_FLAT else "",
        gain=header_gain(hdr),
        offset=header_offset(hdr),
        ncombine=int(ncomb) if ncomb is not None else None,
        camera_id=int(camera_id) if camera_id is not None else None,
        telescope_id=int(telescope_id) if (telescope_id is not None and k == KIND_FLAT) else None,
        origin=ORIGIN_IMPORTED,
        origin_software=header_software(hdr) or "unknown",
        combine_method=str(hdr.get("VYSTKMOD") or ""),
        capture_utc=cap,
        offset_handling=header_flat_offset_handling(hdr) if k == KIND_FLAT else "",
        offset_adu=_hfloat(hdr, "VYOFFADU") if k == KIND_FLAT else None,
    ), f"{k.upper()} {p.name}"


# ---------------------------------------------------------------------------
# validity window
# ---------------------------------------------------------------------------
def validity_status(
    capture_utc: datetime | None,
    validity_days: float,
    light_dates: Sequence[datetime],
) -> tuple[bool, str]:
    """Window rule: ``D - 1 d <= L <= D + V`` for every LIGHT date L.

    Returns (valid, reason). A master without a capture date is not valid.
    """
    if capture_utc is None:
        return False, "master has no capture date"
    if not light_dates:
        return False, "no LIGHT date"
    lo = capture_utc - timedelta(days=VALIDITY_LEAD_DAYS)
    hi = capture_utc + timedelta(days=float(validity_days))
    first, last = min(light_dates), max(light_dates)
    if first < lo:
        return False, f"LIGHT {first.date()} is before master {capture_utc.date()} - 1 d"
    if last > hi:
        return False, f"expired: LIGHT {last.date()} after {hi.date()} (master {capture_utc.date()} + {validity_days:g} d)"
    return True, f"valid {lo.date()} .. {hi.date()}"


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    master: LibraryMaster | None
    reason: str
    rebin_from: int = 1  # factor master->light (1 = same binning)
    rejected: list[str] = field(default_factory=list)


def _same_offset(a: int | None, b: int | None) -> bool:
    return a is None or b is None or int(a) == int(b)


def _pick_best(
    cands: list[LibraryMaster],
    *,
    validity_days: float,
    light_dates: Sequence[datetime],
    light_binning: int,
    temp: float | None,
    what: str,
) -> MatchResult:
    rejected: list[str] = []
    ok: list[tuple[int, float, float, LibraryMaster]] = []
    for m in cands:
        if not Path(m.path).is_file():
            rejected.append(f"{Path(m.path).name}: file missing")
            continue
        if m.binning == light_binning:
            rb = 1
        elif m.binning == 1 and light_binning > 1:
            rb = int(light_binning)
        else:
            rejected.append(f"{Path(m.path).name}: binning {m.binning} cannot serve {light_binning}")
            continue
        valid, why = validity_status(m.capture_utc, validity_days, light_dates)
        if not valid:
            rejected.append(f"{Path(m.path).name}: {why}")
            continue
        dt = abs(float(m.temp) - float(temp)) if (temp is not None and m.temp is not None) else 0.0
        cap = -(m.capture_utc.timestamp() if m.capture_utc else 0.0)
        ok.append((0 if rb == 1 else 1, dt, cap, m))
    if not ok:
        return MatchResult(None, f"no valid {what}" + (f" ({'; '.join(rejected[:4])})" if rejected else ""), 1, rejected)
    ok.sort(key=lambda t: (t[0], t[1], t[2]))
    best = ok[0][3]
    rb = 1 if best.binning == light_binning else int(light_binning)
    return MatchResult(best, f"{what} {Path(best.path).name}", rb, rejected)


def find_dark(
    db: Any,
    *,
    camera_id: int,
    gain: int,
    offset: int | None,
    exposure: float,
    binning: int,
    temp: float | None,
    temp_tol: float,
    light_dates: Sequence[datetime],
    validity_days: float,
    kind: str = KIND_DARK,
    candidates: Sequence[LibraryMaster] | None = None,
) -> MatchResult:
    """DARK (or DARKFLAT with ``kind``): camera, gain, offset, exposure, binning, temp +- tol.

    ``candidates`` replaces the library rows (masters found next to the LIGHTs).
    """
    if temp is None:
        return MatchResult(None, "LIGHT has no sensor temperature - cannot match a dark")
    pool = (
        [m for m in candidates if m.kind == kind]
        if candidates is not None
        else list_masters(db, kind=kind, camera_id=camera_id)
    )
    cands = [
        m
        for m in pool
        if m.gain == int(gain)
        and _same_offset(m.offset, offset)
        and abs(m.exptime - float(exposure)) <= 1e-6 * max(1.0, abs(float(exposure)))
        and m.temp is not None
        and abs(m.temp - float(temp)) <= float(temp_tol) + 1e-9
    ]
    return _pick_best(cands, validity_days=validity_days, light_dates=light_dates,
                 light_binning=int(binning), temp=temp, what=kind.upper())


def find_flat(
    db: Any,
    *,
    camera_id: int,
    telescope_id: int | None,
    filter_name: str,
    gain: int,
    binning: int,
    light_dates: Sequence[datetime],
    validity_days: float,
    candidates: Sequence[LibraryMaster] | None = None,
) -> MatchResult:
    """FLAT: camera, telescope, filter, gain, binning (exposure/temperature irrelevant).

    ``candidates`` replaces the library rows (masters found next to the LIGHTs).
    """
    if telescope_id is None:
        return MatchResult(None, "no telescope selected - FLAT is linked to camera + telescope")
    flt = str(filter_name or "NoFilter")
    pool = (
        [m for m in candidates if m.kind == KIND_FLAT]
        if candidates is not None
        else list_masters(db, kind=KIND_FLAT, camera_id=camera_id)
    )
    cands = [
        m
        for m in pool
        if m.telescope_id == int(telescope_id) and m.gain == int(gain) and m.filter_name == flt
    ]
    return _pick_best(cands, validity_days=validity_days, light_dates=light_dates,
                 light_binning=int(binning), temp=None, what="FLAT")


# ---------------------------------------------------------------------------
# offset (pedestal) measured from darks - spec 3.1b
# ---------------------------------------------------------------------------
@dataclass
class OffsetEstimate:
    adu: float | None
    method: str
    n_darks: int = 0
    darks: list[str] = field(default_factory=list)


def _robust_level(path: str) -> float:
    with fits.open(path, memmap=True) as h:
        a = np.asarray(h[0].data, dtype=np.float64)
    return float(np.nanmedian(a))


def measure_offset_adu(
    db: Any,
    *,
    camera_id: int,
    gain: int,
    offset: int | None,
    binning: int,
    temp: float | None = None,
    temp_tol: float | None = None,
) -> OffsetEstimate:
    """Constant offset P from DARK/DARKFLAT masters of the same camera/gain/offset/binning.

    >= 2 distinct exposures: intercept at t=0 of median level vs exposure (least squares).
    1 exposure: median of that dark (upper bound: includes dark current).
    No dark: None.
    """
    darks = [
        m
        for kind in (KIND_DARK, KIND_DARKFLAT)
        for m in list_masters(db, kind=kind, camera_id=camera_id)
        if m.gain == int(gain) and _same_offset(m.offset, offset) and m.binning == int(binning)
        and Path(m.path).is_file()
        and (temp is None or temp_tol is None or m.temp is None or abs(m.temp - float(temp)) <= float(temp_tol) + 1e-9)
    ]
    if not darks:
        return OffsetEstimate(None, "no dark of this camera/gain/offset/binning")
    pts = [(m.exptime, _robust_level(m.path), m.path) for m in darks]
    exps = sorted({p[0] for p in pts})
    if len(exps) >= 2:
        t = np.array([p[0] for p in pts], dtype=np.float64)
        y = np.array([p[1] for p in pts], dtype=np.float64)
        slope, icpt = np.polyfit(t, y, 1)
        return OffsetEstimate(float(icpt), f"intercept of {len(pts)} darks ({', '.join(f'{e:g}s' for e in exps)}); "
                              f"dark current {slope:.4g} ADU/s", len(pts), [p[2] for p in pts])
    p0 = min(pts, key=lambda p: p[0])
    return OffsetEstimate(float(p0[1]), f"median of shortest dark {p0[0]:g}s (upper bound)", 1, [p0[2]])


def iter_light_dates(paths: Iterable[str | Path]) -> list[datetime]:
    out: list[datetime] = []
    for p in paths:
        try:
            with fits.open(p, memmap=True) as h:
                dt = header_capture_utc(h[0].header)
        except (OSError, ValueError):
            dt = None
        if dt is not None:
            out.append(dt)
    return out


# --- moved from calibration (LAYER-2026-10) ---


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


def _header_capture_datetime(hdr: fits.Header) -> tuple[datetime | None, str | None]:
    """Priority: VY_CDATE (VYVAR stack stamp) -> DATE-OBS -> DATEOBS; naive -> UTC."""
    for key in ("VY_CDATE", "DATE-OBS", "DATEOBS"):
        dt = _parse_master_header_datetime(hdr.get(key))
        if dt is not None:
            return dt, key
    return None, None
