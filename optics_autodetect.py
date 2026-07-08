"""Scan Source auto-detect (VYVAR Phase 3).

Fingerprint the camera / telescope / observer site from a representative FITS
header so the Scan Source UI can pre-fill its selectors, *overriding* the
``IS_DEFAULT`` baseline. Detection is evidence-based and never blocks: the user
can always override, and an unconfident match simply leaves the default in place.

Why fingerprinting (not just name keywords):
    * draft 360 carries ``INSTRUME='QHY CCD QHY294PROM'`` but a useless
      ``TELESCOP='Sample Primary ...'`` string — so the camera is found by name +
      full-resolution sensor dims, the telescope by ``FOCALLEN`` + ``APTDIA``.
    * draft 363 has NO name keywords at all and a wrong pixel size, but its image
      dimensions (6252x4176) + ``GAIN`` (0.78) uniquely fingerprint the C3-26000.

Each detector returns a :class:`Detection` with the matched DB id, a 0..1
confidence, and the human-readable evidence that fired.
"""

from __future__ import annotations

import glob
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Auto-fill must be CONSERVATIVE: a wrong equipment auto-fill silently corrupts gain/
# pixel via the resolver's DB-set authority, whereas a *missed* match only prompts.
#   high   (>= 0.80) -> auto-fill (override default)
#   medium (0.50..0.80) -> PRE-FILL but flag "unconfirmed — verify"
#   low    (< 0.50) -> prompt (leave default)
AUTOFILL_THRESHOLD = 0.80   # high band: silently auto-fill
PREFILL_THRESHOLD = 0.50    # medium band floor: pre-fill but require verification


@dataclass
class Detection:
    matched_id: int | None = None
    label: str = ""
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Usable match (pre-fill or better) — not flagged as an unresolved gap."""
        return self.matched_id is not None and self.confidence >= PREFILL_THRESHOLD

    @property
    def autofill(self) -> bool:
        """High-confidence: safe to silently override the default selector."""
        return self.matched_id is not None and self.confidence >= AUTOFILL_THRESHOLD

    def band(self) -> str:
        if self.matched_id is None or self.confidence < PREFILL_THRESHOLD:
            return "none"
        if self.confidence >= AUTOFILL_THRESHOLD:
            return "high"
        return "medium"


@dataclass
class AutodetectReport:
    equipment: Detection = field(default_factory=Detection)
    telescope: Detection = field(default_factory=Detection)
    location: Detection = field(default_factory=Detection)
    header_path: str | None = None
    #: Phase 4 — fields neither present in the FITS header nor auto-detected.
    #: Each item: {"field", "detail", "fallback"} so the UI can surface ONLY the gaps
    #: and pre-fill from the default where sensible.
    unresolved: list[dict[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Header helpers
# --------------------------------------------------------------------------- #
def _hdr_float(header: Any, *keys: str) -> float | None:
    if header is None:
        return None
    for k in keys:
        try:
            if k in header:
                v = float(header[k])
                if math.isfinite(v):
                    return v
        except (TypeError, ValueError):
            continue
    return None


def _hdr_str(header: Any, *keys: str) -> str:
    if header is None:
        return ""
    for k in keys:
        try:
            if k in header and header[k] is not None:
                s = str(header[k]).strip()
                if s:
                    return s
        except (TypeError, ValueError):
            continue
    return ""


def _norm(s: Any) -> str:
    """Uppercase, strip everything but [A-Z0-9]."""
    return re.sub(r"[^A-Z0-9]", "", str(s or "").upper())


def _model_core(norm_name: str) -> str:
    """Longest letters+digits model token, e.g. QHY294MM -> QHY294, IMX457 -> IMX457."""
    m = re.search(r"[A-Z]*\d{2,}", norm_name)
    return m.group(0) if m else ""


def _parse_sensorsize(raw: Any) -> tuple[int, int] | None:
    """Parse ``"4164*2796"`` / ``"6252x4176"`` -> (w, h)."""
    if raw is None:
        return None
    m = re.search(r"(\d{2,})\s*[*xX×]\s*(\d{2,})", str(raw))
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except ValueError:
        return None


def _dims_close(a: float, b: float, tol_frac: float = 0.01, tol_abs: float = 8.0) -> bool:
    """Sensor dims close, allowing for binning truncation (e.g. 1397*2=2794 vs 2796)."""
    return abs(a - b) <= max(tol_abs, tol_frac * max(a, b))


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def detect_equipment(header: Any, equipments: list[dict[str, Any]]) -> Detection:
    instr_n = _norm(_hdr_str(header, "INSTRUME", "CAMERA", "INSTRUMENT", "DETNAM"))
    naxis1 = _hdr_float(header, "NAXIS1", "IMAGEW")
    naxis2 = _hdr_float(header, "NAXIS2", "IMAGEH")
    xbin = _hdr_float(header, "XBINNING", "BINNING") or 1.0
    ybin = _hdr_float(header, "YBINNING", "BINNING") or 1.0
    full_w = naxis1 * xbin if naxis1 else None
    full_h = naxis2 * ybin if naxis2 else None
    gain_h = _hdr_float(header, "EGAIN", "GAIN")
    xpix = _hdr_float(header, "XPIXSZ", "PIXSIZE1", "XPIXSIZE", "PIXSIZE")
    sensortype_n = _norm(_hdr_str(header, "SENSOR", "SENSORTYP"))

    best: Detection = Detection()
    for row in equipments:
        score = 0.0
        reasons: list[str] = []
        name = _norm(row.get("CAMERANAME"))
        core = _model_core(name)
        if instr_n and name and (name in instr_n or (core and core in instr_n)):
            score += 0.50
            reasons.append(f"name INSTRUME~{row.get('CAMERANAME')}")
        ss = _parse_sensorsize(row.get("SENSORSIZE"))
        if ss and full_w and full_h and _dims_close(full_w, ss[0]) and _dims_close(full_h, ss[1]):
            score += 0.40
            reasons.append(
                f"sensor {int(full_w)}x{int(full_h)} (NAXIS×bin) ~ {ss[0]}*{ss[1]}"
            )
        g_db = row.get("GAIN_ADU")
        if gain_h and gain_h > 0 and g_db and float(g_db) > 0 and math.isclose(
            gain_h, float(g_db), rel_tol=0.05
        ):
            score += 0.30
            reasons.append(f"GAIN {gain_h:g} ~ DB {float(g_db):g}")
        px_db = row.get("PIXELSIZE")
        if xpix and xpix > 0 and px_db and float(px_db) > 0:
            px_full = xpix / (xbin or 1.0)
            if math.isclose(px_full, float(px_db), rel_tol=0.05):
                score += 0.15
                reasons.append(f"pixel {px_full:g}µm ~ DB {float(px_db):g}µm")
        if sensortype_n and _norm(row.get("SENSORTYPE")) and sensortype_n == _norm(
            row.get("SENSORTYPE")
        ):
            score += 0.20
            reasons.append(f"SENSORTYPE {row.get('SENSORTYPE')}")
        if score > best.confidence:
            best = Detection(
                matched_id=int(row["ID"]),
                label=f"{int(row['ID'])}: {row.get('CAMERANAME')} ({row.get('ALIAS')})",
                confidence=min(0.99, score),
                reasons=reasons,
            )
    return best


def detect_telescope(header: Any, telescopes: list[dict[str, Any]]) -> Detection:
    focal_h = _hdr_float(header, "FOCALLEN", "FOCLEN", "TELFOCA")
    if focal_h is not None and 0.0 < focal_h < 25.0:
        focal_h *= 1000.0  # metres -> mm for fast scopes
    apt_h = _hdr_float(header, "APTDIA", "APERTURE", "TELDIAM", "DIAMETER")
    tel_n = _norm(_hdr_str(header, "TELESCOP", "TELESCOPE"))

    best: Detection = Detection()
    for row in telescopes:
        score = 0.0
        reasons: list[str] = []
        focal_db = row.get("FOCAL")
        diam_db = row.get("DIAMETER")
        if focal_h and focal_h > 0 and focal_db and float(focal_db) > 0 and math.isclose(
            focal_h, float(focal_db), rel_tol=0.05
        ):
            score += 0.60
            reasons.append(f"FOCALLEN {focal_h:g}mm ~ DB {float(focal_db):g}mm")
        if apt_h and apt_h > 0 and diam_db and float(diam_db) > 0 and math.isclose(
            apt_h, float(diam_db), rel_tol=0.10
        ):
            score += 0.30
            reasons.append(f"APTDIA {apt_h:g}mm ~ DB {float(diam_db):g}mm")
        name = _norm(row.get("TELESCOPENAME"))
        core = _model_core(name)
        if tel_n and core and len(core) >= 3 and core in tel_n:
            score += 0.20
            reasons.append(f"name TELESCOP~{row.get('TELESCOPENAME')}")
        if score > best.confidence:
            best = Detection(
                matched_id=int(row["ID"]),
                label=f"{int(row['ID'])}: {row.get('TELESCOPENAME')} ({row.get('ALIAS')})",
                confidence=min(0.99, score),
                reasons=reasons,
            )
    return best


def detect_location(header: Any, locations: list[dict[str, Any]]) -> Detection:
    lat_h = _hdr_float(header, "SITELAT", "OBSLAT", "OBSGEO-B", "LAT-OBS")
    lon_h = _hdr_float(header, "SITELONG", "OBSLONG", "OBSGEO-L", "LONG-OBS")
    if lat_h is None or lon_h is None:
        return Detection()
    best: Detection = Detection()
    for row in locations:
        try:
            lat_db = float(row.get("lat"))
            lon_db = float(row.get("lon"))
        except (TypeError, ValueError):
            continue
        if abs(lat_h - lat_db) <= 0.02 and abs(lon_h - lon_db) <= 0.02:
            conf = 0.9 if (abs(lat_h - lat_db) <= 0.005 and abs(lon_h - lon_db) <= 0.005) else 0.6
            if conf > best.confidence:
                best = Detection(
                    matched_id=int(row["id"]),
                    label=f"{int(row['id'])}: {row.get('name')}",
                    confidence=conf,
                    reasons=[
                        f"SITELAT/LONG {lat_h:.4f},{lon_h:.4f} ~ {row.get('name')} "
                        f"{lat_db:.4f},{lon_db:.4f}"
                    ],
                )
    return best


# --------------------------------------------------------------------------- #
# Sample-header discovery + orchestration
# --------------------------------------------------------------------------- #
def find_sample_light_header(source_root: str | Path, *, max_scan: int = 400) -> tuple[Any, str | None]:
    """Find a representative LIGHT FITS header under ``source_root``.

    Prefers ``IMAGETYP`` containing 'LIGHT'/'OBJECT'; otherwise the first readable
    header that is not a calibration frame (DARK/FLAT/BIAS).
    """
    try:
        from astropy.io import fits  # local import (heavy)
    except ImportError:
        return None, None
    root = Path(source_root)
    if not root.is_dir():
        return None, None
    exts = ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS")
    files: list[str] = []
    for ext in exts:
        files.extend(glob.glob(str(root / "**" / ext), recursive=True))
        if len(files) >= max_scan:
            break
    files = sorted(set(files))[:max_scan]
    fallback: tuple[Any, str] | None = None
    for fp in files:
        try:
            hdr = fits.getheader(fp)
        except Exception:  # noqa: BLE001
            # EXC-0114: ? -- intent unclear (try: / hdr = fits.getheader(fp) / except Exception:  # noqa: BLE001 / c... (EXCEPT-BULK 2026-07-08)
            continue
        imagetyp = str(hdr.get("IMAGETYP", "") or "").upper()
        if "LIGHT" in imagetyp or "OBJECT" in imagetyp or "SCIENCE" in imagetyp:
            return hdr, fp
        if not any(c in imagetyp for c in ("DARK", "FLAT", "BIAS", "ZERO")) and fallback is None:
            fallback = (hdr, fp)
    if fallback is not None:
        return fallback
    return None, None


def _has_any(header: Any, *keys: str) -> bool:
    if header is None:
        return False
    for k in keys:
        try:
            if k in header and str(header[k]).strip() not in ("", "0", "0.0"):
                return True
        except (TypeError, ValueError):
            continue
    return False


def assess_unresolved(
    header: Any,
    equipment: Detection,
    telescope: Detection,
    location: Detection,
) -> list[dict[str, str]]:
    """Fields neither in the FITS header nor auto-detected (Phase 4 poor-FITS prompt).

    Only genuine gaps are returned, so the UI surfaces the minimum the user must
    confirm. ``fallback`` names where a sensible default comes from.
    """
    gaps: list[dict[str, str]] = []
    if not equipment.ok:
        gaps.append(
            {
                "field": "Equipment (camera)",
                "detail": "No INSTRUME / sensor-dimension / GAIN fingerprint matched a DB camera.",
                "fallback": "default camera (IS_DEFAULT)",
            }
        )
    if not telescope.ok and not _has_any(header, "FOCALLEN", "FOCLEN", "APTDIA", "APERTURE"):
        gaps.append(
            {
                "field": "Telescope",
                "detail": "No FOCALLEN / APTDIA in header and no name match (TELESCOP absent or generic).",
                "fallback": "default telescope (IS_DEFAULT)",
            }
        )
    if not location.ok and not _has_any(header, "SITELAT", "OBSLAT", "OBSGEO-B"):
        gaps.append(
            {
                "field": "Observer site",
                "detail": "No SITELAT / SITELONG in header — BJD/airmass site cannot come from FITS.",
                "fallback": "default location (IS_DEFAULT)",
            }
        )
    if not _has_any(header, "OBJCTRA", "RA", "CRVAL1", "OBJRA"):
        gaps.append(
            {
                "field": "Pointing (RA/Dec)",
                "detail": "No pointing/WCS keywords in header — field center will rely on blind plate-solve.",
                "fallback": "blind solve / manual draft center",
            }
        )
    return gaps


def autodetect_from_source(
    source_root: str | Path,
    *,
    equipments: list[dict[str, Any]],
    telescopes: list[dict[str, Any]],
    locations: list[dict[str, Any]],
) -> AutodetectReport:
    """Read a sample light header and fingerprint optics + site against the DB."""
    hdr, path = find_sample_light_header(source_root)
    if hdr is None:
        return AutodetectReport(header_path=None)
    eq = detect_equipment(hdr, equipments)
    tel = detect_telescope(hdr, telescopes)
    loc = detect_location(hdr, locations)
    return AutodetectReport(
        equipment=eq,
        telescope=tel,
        location=loc,
        header_path=path,
        unresolved=assess_unresolved(hdr, eq, tel, loc),
    )
