"""Camera profiles: gain and read noise of a camera as a function of gain setting, readout mode and binning.

CAMERA-PROFILE-2026-10. Two levels of data:

* the shipped catalog (``camera_catalog.json``, read-only): camera models, their readout modes (ADC bits,
  binning convention, what the header GAIN card means) and the manufacturer's gain / read-noise curve
  points, each with its source;
* the user's cameras (database): ``EQUIPMENTS`` (model + mode chosen from the catalog, default gain
  setting, the camera's default point ``GAIN_ADU`` / ``READNOISE_E`` at bin 1) and ``EQUIPMENT_CURVE``
  (measured or manually entered points, optionally at a specific binning).

The effective gain of the delivered pixel values depends on how the camera bins:

* ``average`` - N x N pixels digitally averaged: gain x N^2, read noise x N (Moravian SIPS);
* ``sum`` - digitally summed: gain unchanged, read noise x N (QHY software binning);
* ``charge`` - binned on the chip before readout: gain and read noise unchanged.

A point measured at the binning of the frames is used as is; a bin-1 point is scaled by the
convention. The container scale of N-bit ADCs (``ADC_BITS``) is applied later by ``rig_photometry``.
"""
from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

CATALOG_PATH = Path(__file__).resolve().with_name("camera_catalog.json")

BIN_MODES = ("average", "sum", "charge")
#: Binning convention when a camera states none - the behaviour VYVAR always had for DB values.
DEFAULT_BIN_MODE = "average"
HEADER_GAIN_KINDS = ("e_per_adu", "setting")

#: Priority of point sources (lower wins) at the same setting.
_SOURCE_RANK = {"measured": 0, "manual": 1, "equipment": 2, "catalog": 3}


@dataclass(frozen=True)
class CurvePoint:
    setting: float | None  # gain setting of the driver; None = "any / unknown setting"
    binning: int  # binning the values were measured at (1 = per native pixel)
    e_per_adu: float | None
    read_noise_e: float | None
    source: str  # measured | manual | equipment | catalog
    note: str = ""


@dataclass(frozen=True)
class CameraProfile:
    equipment_id: int | None = None
    name: str = ""
    model_key: str | None = None
    mode_key: str | None = None
    bin_mode: str = DEFAULT_BIN_MODE
    bin_mode_source: str = "default"
    header_gain: str | None = None  # e_per_adu | setting | None (decide from the card comment)
    default_setting: float | None = None
    adc_bits: int | None = None
    points: tuple[CurvePoint, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Lookup:
    value: float | None
    source: str  # measured | manual | equipment | catalog | unresolved
    detail: str = ""


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _load_catalog_cached(path_s: str, mtime: float) -> dict[str, Any]:
    with open(path_s, encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


_MISSING_WARNED: set[str] = set()


def catalog_missing_message(path: Path | None = None) -> str | None:
    """A user-facing message when the shipped camera catalog file is missing (None when present)."""
    p = Path(path) if path is not None else CATALOG_PATH
    if p.is_file():
        return None
    return (f"The camera catalog {p.name} is missing ({p}). Camera models, gain curves and header "
            "matching are unavailable - reinstall VYVAR or restore the file.")


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    """The camera catalog (``{"schema", "version", "models": [...]}``). A missing file is logged once
    (it ships with VYVAR) and gives an empty catalog."""
    p = Path(path) if path is not None else CATALOG_PATH
    if not p.is_file():
        if str(p) not in _MISSING_WARNED:
            _MISSING_WARNED.add(str(p))
            logging.getLogger(__name__).warning("[CAMERA-CATALOG] %s", catalog_missing_message(p))
        return {"models": []}
    return _load_catalog_cached(str(p), p.stat().st_mtime)


def catalog_models(path: Path | None = None) -> list[dict[str, Any]]:
    return [m for m in load_catalog(path).get("models", []) if isinstance(m, dict) and m.get("key")]


def catalog_model(model_key: str | None, path: Path | None = None) -> dict[str, Any] | None:
    if not model_key:
        return None
    k = str(model_key).strip().casefold()
    ms = catalog_models(path)
    hit = next((m for m in ms if str(m["key"]).casefold() == k), None)
    if hit is None:  # keys of an earlier catalog version stay valid
        hit = next((m for m in ms if k in {str(x).casefold() for x in m.get("legacy_keys", [])}), None)
    return hit


def catalog_mode(model: dict[str, Any] | None, mode_key: str | None) -> dict[str, Any] | None:
    """Readout mode of ``model``: ``mode_key``, else the model's only / first mode."""
    if not model:
        return None
    modes = [m for m in model.get("modes", []) if isinstance(m, dict)]
    if not modes:
        return None
    if mode_key:
        k = str(mode_key).strip().casefold()
        hit = next((m for m in modes if str(m.get("key", "")).casefold() == k), None)
        if hit is not None:
            return hit
    return modes[0]


def _norm_name(s: Any) -> str:
    """Letters, digits and '+' (C1+3000 is not C1-3000), case-insensitive."""
    return "".join(ch for ch in str(s or "").casefold() if ch.isalnum() or ch == "+")


# How capture programs and drivers decorate the model name in INSTRUME (checked in the INDI and
# N.I.N.A. driver sources): vendor / driver prefixes ("ZWO CCD ASI2600MM Pro", "QHY CCD QHY600M",
# "MI C3-61000", "PlayerOne Poseidon-M Pro", "ToupTek ATR2600M"), the QHY SDK id serial
# ("QHY294PROM-cfd3"; only on QHY names, so "C3-26000C" keeps its "C") and a counter for a second
# identical camera ("... 2", "(1)").
_SERIAL_SUFFIX = re.compile(r"-[0-9a-f]{3,}$", re.IGNORECASE)
_VENDOR_PREFIX = re.compile(
    r"^(?:zwo\s+ccd|zwo|qhy\s*ccd|qhyccd|playerone(?:\s+ccd)?|player\s+one|mi|moravian(?:\s+instruments)?"
    r"|touptek|altair|atik|svbony2?|omegon|sbig|fli)\s+",
    re.IGNORECASE,
)
_COUNTER_SUFFIX = re.compile(r"(?:\s+#?\d|\s*\(\d+\))$")


def _name_variants(name: Any) -> list[str]:
    """``name`` and its undecorated forms, most literal first."""
    out: list[str] = []
    cur = str(name or "").strip()
    for step in (None, _SERIAL_SUFFIX, _VENDOR_PREFIX, _COUNTER_SUFFIX, _SERIAL_SUFFIX):
        if step is _SERIAL_SUFFIX and "qhy" not in cur.casefold():
            continue
        if step is not None:
            cur = step.sub("", cur).strip()
        if cur and cur not in out:
            out.append(cur)
    return out


def match_catalog_models(*names: Any, path: Path | None = None) -> list[dict[str, Any]]:
    """All catalog models whose model name or header alias equals one of ``names`` (or its
    undecorated form, see ``_name_variants``). Exact normalised matches only; the most literal
    form that matches anything decides. More than one model means the name is ambiguous (e.g. the
    mono and colour version report the same name)."""
    models = catalog_models(path)
    index: dict[str, list[dict[str, Any]]] = {}
    for m in models:
        for k in {_norm_name(m.get("model")), *(_norm_name(a) for a in m.get("aliases", []))}:
            if k and m not in index.setdefault(k, []):
                index[k].append(m)
    for tier in range(max((len(_name_variants(n)) for n in names), default=0)):
        hits: list[dict[str, Any]] = []
        for n in names:
            vs = _name_variants(n)
            if tier < len(vs):
                for m in index.get(_norm_name(vs[tier]), []):
                    if m not in hits:
                        hits.append(m)
        if hits:
            return hits
    return []


def _catalog_points(mode: dict[str, Any] | None) -> list[CurvePoint]:
    out: list[CurvePoint] = []
    # fixed-gain readout modes (scientific sCMOS): one value for the mode, any setting
    if mode and (_pos(mode.get("e_per_adu")) is not None or _pos(mode.get("read_noise_e")) is not None):
        out.append(CurvePoint(None, 1, _pos(mode.get("e_per_adu")), _pos(mode.get("read_noise_e")), "catalog",
                              note=str(mode.get("source") or "readout mode value")))
    for p in (mode or {}).get("curve", []) or []:
        if not isinstance(p, dict):
            continue
        out.append(
            CurvePoint(
                setting=_num(p.get("setting")),
                binning=1,
                e_per_adu=_pos(p.get("e_per_adu")),
                read_noise_e=_pos(p.get("read_noise_e")),
                source="catalog",
                note=str(p.get("source") or ""),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Profile from the database
# --------------------------------------------------------------------------- #
def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pos(v: Any) -> float | None:
    f = _num(v)
    return f if f is not None and f > 0 else None


def _norm_bin_mode(v: Any) -> str | None:
    s = str(v or "").strip().lower()
    return s if s in BIN_MODES else None


def _norm_header_gain(v: Any) -> str | None:
    s = str(v or "").strip().lower()
    return s if s in HEADER_GAIN_KINDS else None


_EQ_PROFILE_COLS = (
    "CAMERANAME", "GAIN_ADU", "READNOISE_E", "ADC_BITS",
    "MODEL_KEY", "READOUT_MODE", "BIN_MODE", "HEADER_GAIN", "DEFAULT_GAIN_SETTING",
)


def profile_from_connection(conn: sqlite3.Connection, equipment_id: int | None) -> CameraProfile:
    """Camera profile of ``equipment_id`` (EQUIPMENTS + EQUIPMENT_CURVE + catalog)."""
    if equipment_id is None:
        return CameraProfile()
    eid = int(equipment_id)
    cols = {r[1] for r in conn.execute("PRAGMA table_info('EQUIPMENTS');").fetchall()}
    want = [c for c in _EQ_PROFILE_COLS if c in cols]
    row = conn.execute(f"SELECT {', '.join(want)} FROM EQUIPMENTS WHERE ID = ?;", (eid,)).fetchone() if want else None
    if row is None:
        return CameraProfile(equipment_id=eid)
    v = dict(zip(want, row))
    user_points: list[CurvePoint] = []
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
    mode_key = (str(v.get("READOUT_MODE")).strip() or None) if v.get("READOUT_MODE") is not None else None
    if "EQUIPMENT_CURVE" in tables:
        for r in conn.execute(
            "SELECT READOUT_MODE, GAIN_SETTING, BINNING, E_PER_ADU, READNOISE_E, SOURCE, NOTE "
            "FROM EQUIPMENT_CURVE WHERE EQUIPMENT_ID = ? ORDER BY MEASURED_AT DESC, ID DESC;",  # NEW-1: newest first
            (eid,),
        ).fetchall():
            r_mode = (str(r[0]).strip() or None) if r[0] is not None else None
            if r_mode is not None and mode_key is not None and r_mode.casefold() != mode_key.casefold():
                continue
            src = str(r[5] or "manual").strip().lower()
            user_points.append(
                CurvePoint(
                    setting=_num(r[1]),
                    binning=max(1, int(_num(r[2]) or 1)),
                    e_per_adu=_pos(r[3]),
                    read_noise_e=_pos(r[4]),
                    source=src if src in ("measured", "manual") else "manual",
                    note=str(r[6] or ""),
                )
            )
    default_setting = _num(v.get("DEFAULT_GAIN_SETTING"))
    g_eq, rn_eq = _pos(v.get("GAIN_ADU")), _pos(v.get("READNOISE_E"))
    if g_eq is not None or rn_eq is not None:
        user_points.append(CurvePoint(default_setting, 1, g_eq, rn_eq, "equipment"))
    model = catalog_model(v.get("MODEL_KEY"))
    mode = catalog_mode(model, mode_key)
    bin_mode = _norm_bin_mode(v.get("BIN_MODE"))
    bin_src = "equipment"
    if bin_mode is None and mode is not None and _norm_bin_mode(mode.get("binning")):
        bin_mode, bin_src = _norm_bin_mode(mode.get("binning")), "catalog"
    if bin_mode is None:
        bin_mode, bin_src = DEFAULT_BIN_MODE, "default"
    header_gain = _norm_header_gain(v.get("HEADER_GAIN"))
    if header_gain is None and model is not None:
        header_gain = _norm_header_gain(model.get("header_gain"))
    adc = _num(v.get("ADC_BITS"))
    if adc is None and mode is not None:
        adc = _num(mode.get("adc_bits"))
    return CameraProfile(
        equipment_id=eid,
        name=str(v.get("CAMERANAME") or ""),
        model_key=str(model["key"]) if model else None,
        mode_key=str(mode.get("key")) if mode else mode_key,
        bin_mode=bin_mode,
        bin_mode_source=bin_src,
        header_gain=header_gain,
        default_setting=default_setting,
        adc_bits=int(adc) if adc is not None and adc > 0 else None,
        points=tuple(user_points + _catalog_points(mode)),
    )


def load_profile(db_path: str | Path | None, equipment_id: int | None) -> CameraProfile:
    """Read-only profile from the database file; an empty profile when the database is missing."""
    if equipment_id is None:
        return CameraProfile()
    p = Path(str(db_path or "")).expanduser()
    if not str(db_path or "").strip() or not p.is_file():
        return CameraProfile(equipment_id=int(equipment_id))
    con = sqlite3.connect(f"file:{p.resolve().as_posix()}?mode=ro", uri=True)
    try:
        return profile_from_connection(con, equipment_id)
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #
def scale_for_binning(value: float, binning: int, bin_mode: str, param: str) -> float:
    """Bin-1 value -> value at ``binning`` under the camera's binning convention."""
    n = max(1, int(binning))
    if n == 1:
        return float(value)
    if param == "gain":
        return float(value) * (n * n if bin_mode == "average" else 1)
    return float(value) * (1 if bin_mode == "charge" else n)


def _attr(p: CurvePoint, param: str) -> float | None:
    return p.e_per_adu if param == "gain" else p.read_noise_e


def _same_setting(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6)


def lookup(profile: CameraProfile, param: str, setting: float | None, binning: int) -> Lookup:
    """Gain (``param="gain"``, e-/ADU native) or read noise (``"read_noise"``, e-) at the frame's gain
    setting and binning. ``setting`` None = the header states no setting (the camera default is used).

    Order: user points at that setting (measured > manual > the camera's default point), exact binning
    first, else bin 1 scaled; then the catalog curve (exact point, or interpolated between its
    points - log-linear for gain, linear for read noise; never extrapolated). The camera's default
    point also serves a header without a setting, and any setting when no default setting is stored
    (the behaviour of databases without curves).
    """
    eff_setting = setting if setting is not None else profile.default_setting
    users = [p for p in profile.points if p.source != "catalog" and _attr(p, param) is not None]
    users.sort(key=lambda p: _SOURCE_RANK.get(p.source, 9))

    def _ok(p: CurvePoint) -> bool:
        if p.setting is None:  # default point without a stored default setting: any setting
            return True
        return eff_setting is not None and _same_setting(p.setting, eff_setting)

    for want_exact in (True, False):
        for p in users:
            if not _ok(p):
                continue
            if want_exact and p.binning == binning:
                return Lookup(float(_attr(p, param)), p.source, _detail(p, binning, profile, scaled=False))
            if not want_exact and p.binning == 1:
                val = scale_for_binning(float(_attr(p, param)), binning, profile.bin_mode, param)
                return Lookup(val, p.source, _detail(p, binning, profile, scaled=binning > 1))
    cat = [p for p in profile.points if p.source == "catalog" and p.setting is not None and _attr(p, param) is not None]
    if eff_setting is not None and cat:
        v = _interp(cat, float(eff_setting), param)
        if v is not None:
            val = scale_for_binning(v, binning, profile.bin_mode, param)
            return Lookup(val, "catalog", f"catalog curve at setting {eff_setting:g}, bin {binning} ({profile.bin_mode})")
    fixed = [p for p in profile.points if p.source == "catalog" and p.setting is None and _attr(p, param) is not None]
    if fixed:
        val = scale_for_binning(float(_attr(fixed[0], param)), binning, profile.bin_mode, param)
        return Lookup(val, "catalog", f"catalog value of readout mode {profile.mode_key or '-'}, bin {binning}")
    return Lookup(None, "unresolved", f"no {param} for setting {eff_setting if eff_setting is not None else '?'}")


_SOURCE_TEXT = {"measured": "measured point", "manual": "manual point", "equipment": "camera default point"}


def _detail(p: CurvePoint, binning: int, profile: CameraProfile, *, scaled: bool) -> str:
    s = "any setting" if p.setting is None else f"setting {p.setting:g}"
    src = _SOURCE_TEXT.get(p.source, p.source)
    if scaled:
        return f"{src}, {s}, bin 1 scaled to bin {binning} ({profile.bin_mode})"
    return f"{src}, {s}, bin {p.binning}"


def _interp(points: list[CurvePoint], x: float, param: str) -> float | None:
    pts = sorted(((float(p.setting), float(_attr(p, param))) for p in points), key=lambda t: t[0])
    for xs, ys in pts:
        if math.isclose(xs, x, abs_tol=1e-6):
            return ys
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 < x < x1:
            t = (x - x0) / (x1 - x0)
            if param == "gain":
                return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
            return y0 + t * (y1 - y0)
    return None
