"""Scan Source auto-detect (VYVAR Phase 3).

Fingerprint the camera / telescope from a representative FITS header so the Scan
Source UI can pre-fill its selectors, *overriding* the ``IS_DEFAULT`` baseline.
Detection is evidence-based: an unconfident match leaves the selection in place, and the
user can always override. RIG-AUTOSELECT-2026-11 (supersedes S6-FIX SITE-MANUAL-2026-10):
the observing site is selected too when the header states one (``param_resolver.
header_site_reading``) and a LOCATION row lies within the derived tolerance
(:func:`site_match_tolerance_m`). RIG-GUARD-2026-11: :func:`rig_header_mismatches` stops a
run whose selected camera / telescope / site contradicts the light headers (the GUI RUN
button and ``night_run.check_run_preconditions`` call it).

Why fingerprinting (not just name keywords):
    * draft 360 carries ``INSTRUME='QHY CCD QHY294PROM'`` but a useless
      ``TELESCOP='Sample Primary ...'`` string - so the camera is found by name +
      full-resolution sensor dims, the telescope by ``FOCALLEN`` + ``APTDIA``.
    * draft 363 has NO name keywords at all and a wrong pixel size, but its image
      dimensions (6252x4176) + ``GAIN`` (0.78) uniquely fingerprint the C3-26000.

Each detector returns a :class:`Detection` with the matched DB id, a 0..1
confidence, and the human-readable evidence that fired.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from optics import header_focal_length_mm, header_native_pixel_um_mean, parse_sensor_size
from param_resolver import (  # noqa: F401 - SITE_SAME_RADIUS_M re-exported for the site rule
    SITE_SAME_RADIUS_M,
    header_site_reading,
    site_distance_m,
    site_match_tolerance_m,
)

# Auto-fill must be CONSERVATIVE: a wrong equipment auto-fill silently corrupts gain/
# pixel via the resolver's DB-set authority, whereas a *missed* match only prompts.
#   high   (>= 0.80) -> auto-fill (override default)
#   medium (0.50..0.80) -> PRE-FILL but flag "unconfirmed - verify"
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
        """Usable match (pre-fill or better) - not flagged as an unresolved gap."""
        return self.matched_id is not None and self.confidence >= PREFILL_THRESHOLD

    def band(self) -> str:
        if self.matched_id is None or self.confidence < PREFILL_THRESHOLD:
            return "none"
        if self.confidence >= AUTOFILL_THRESHOLD:
            return "high"
        return "medium"


@dataclass
class SiteDetection:
    """RIG-AUTOSELECT-2026-11: the LOCATION row at the header site (``matched_id``), or why there is none."""

    #: (lat, lon, elev or None) stated by the header; None when the header has no site
    header_site: tuple[float, float, float | None] | None = None
    tolerance_m: float | None = None
    matched_id: int | None = None
    label: str = ""
    distance_m: float | None = None
    #: DB sites within the tolerance (> 1: the header cannot tell them apart)
    n_within: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.matched_id is not None

    @property
    def addable(self) -> bool:
        """The header states a site that no DB row is (RIG-ADD-2026-11 offers to add it)."""
        return self.header_site is not None and self.n_within == 0

    def band(self) -> str:
        """A site within the tolerance is the same site (no partial evidence): high, else none."""
        return "high" if self.matched_id is not None else "none"


@dataclass
class AutodetectReport:
    equipment: Detection = field(default_factory=Detection)
    telescope: Detection = field(default_factory=Detection)
    location: SiteDetection = field(default_factory=SiteDetection)
    header_path: str | None = None
    #: RIG-ADD-2026-11: header facts that pre-fill the "Add to DB" forms (:func:`header_rig_prefill`)
    prefill: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: SAMPLE-BOUNDED-2026-11: why no header was found (``header_path`` None)
    no_light: str = ""
    #: Phase 4 - fields neither present in the FITS header nor auto-detected.
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


def _dims_close(a: float, b: float, tol_frac: float = 0.01, tol_abs: float = 8.0) -> bool:
    """Sensor dims close, allowing for binning truncation (e.g. 1397*2=2794 vs 2796)."""
    return abs(a - b) <= max(tol_abs, tol_frac * max(a, b))


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def _pick_best(scored: list[tuple[float, Detection]]) -> Detection:
    """G03-03: the row with the highest *raw* score wins, independent of the row order; the confidence
    (capped at 0.99) is for display only. Equal top scores of different rows -> no match (the user picks;
    the reasons name the tied rows)."""
    if not scored:
        return Detection()
    top = max(sc for sc, _ in scored)
    if top <= 0.0:
        return Detection()
    best = [d for sc, d in scored if sc == top]
    if len(best) > 1:
        return Detection(reasons=["tie: " + " / ".join(sorted(d.label for d in best)) + " - pick one"])
    return best[0]


def equipment_label(row: dict[str, Any]) -> str:
    """The selector label of an EQUIPMENTS row (the GUI options use the same text)."""
    return f"{int(row['ID'])}: {row.get('CAMERANAME')} ({row.get('ALIAS')})"


def telescope_label(row: dict[str, Any]) -> str:
    """The selector label of a TELESCOPE row."""
    return f"{int(row['ID'])}: {row.get('TELESCOPENAME')} ({row.get('ALIAS')})"


def location_label(row: dict[str, Any]) -> str:
    """The selector label of a LOCATION row (``database.get_observer_locations`` dict)."""
    return f"{int(row['id'])}: {row.get('name')}"


def detect_equipment(header: Any, equipments: list[dict[str, Any]]) -> Detection:
    return _pick_best(score_equipment(header, equipments))


def score_equipment(header: Any, equipments: list[dict[str, Any]]) -> list[tuple[float, Detection]]:
    """Raw fingerprint score of every camera row (RIG-GUARD-2026-11 needs the selected row's own score)."""
    instr_n = _norm(_hdr_str(header, "INSTRUME", "CAMERA", "INSTRUMENT", "DETNAM"))
    naxis1 = _hdr_float(header, "NAXIS1", "IMAGEW")
    naxis2 = _hdr_float(header, "NAXIS2", "IMAGEH")
    xbin = _hdr_float(header, "XBINNING", "BINNING") or 1.0
    ybin = _hdr_float(header, "YBINNING", "BINNING") or 1.0
    full_w = naxis1 * xbin if naxis1 else None
    full_h = naxis2 * ybin if naxis2 else None
    gain_h = _hdr_float(header, "EGAIN", "GAIN")
    # native pitch by the optics convention (XPIXSZ binned, PIXSIZE1 native)
    px_full = header_native_pixel_um_mean(header) if header is not None else None
    sensortype_n = _norm(_hdr_str(header, "SENSOR", "SENSORTYP"))

    scored: list[tuple[float, Detection]] = []
    for row in equipments:
        score = 0.0
        reasons: list[str] = []
        name = _norm(row.get("CAMERANAME"))
        core = _model_core(name)
        if instr_n and name and (name in instr_n or (core and core in instr_n)):
            score += 0.50
            reasons.append(f"name INSTRUME~{row.get('CAMERANAME')}")
        elif instr_n and row.get("MODEL_KEY"):
            # CAMERA-PROFILE-2026-10: INSTRUME equals a header alias of the camera's catalog model
            # (an ambiguous name - mono and colour version alike - still fits each of them)
            from camera_profile import catalog_model, match_catalog_models  # noqa: PLC0415

            _own = catalog_model(row.get("MODEL_KEY"))
            _hits = match_catalog_models(_hdr_str(header, "INSTRUME", "CAMERA", "INSTRUMENT", "DETNAM"))
            if _own is not None and any(str(m.get("key")) == str(_own.get("key")) for m in _hits):
                score += 0.50
                reasons.append(f"INSTRUME = catalog model {_own.get('model')}")
        ss = parse_sensor_size(row.get("SENSORSIZE"))
        if ss and full_w and full_h and _dims_close(full_w, ss[0]) and _dims_close(full_h, ss[1]):
            score += 0.40
            reasons.append(
                f"sensor {int(full_w)}x{int(full_h)} (NAXISxbin) ~ {ss[0]}*{ss[1]}"
            )
        g_db = row.get("GAIN_ADU")
        if gain_h and gain_h > 0 and g_db and float(g_db) > 0 and math.isclose(
            gain_h, float(g_db), rel_tol=0.05
        ):
            score += 0.30
            reasons.append(f"GAIN {gain_h:g} ~ DB {float(g_db):g}")
        px_db = row.get("PIXELSIZE")
        if px_full and px_db and float(px_db) > 0:
            if math.isclose(px_full, float(px_db), rel_tol=0.05):
                score += 0.15
                reasons.append(f"pixel {px_full:g}um ~ DB {float(px_db):g}um")
        if sensortype_n and _norm(row.get("SENSORTYPE")) and sensortype_n == _norm(
            row.get("SENSORTYPE")
        ):
            score += 0.20
            reasons.append(f"SENSORTYPE {row.get('SENSORTYPE')}")
        scored.append((score, Detection(
            matched_id=int(row["ID"]),
            label=equipment_label(row),
            confidence=min(0.99, score),
            reasons=reasons,
        )))
    return scored


def detect_telescope(header: Any, telescopes: list[dict[str, Any]]) -> Detection:
    return _pick_best(score_telescope(header, telescopes))


def score_telescope(header: Any, telescopes: list[dict[str, Any]]) -> list[tuple[float, Detection]]:
    """Raw fingerprint score of every telescope row."""
    focal_h = header_focal_length_mm(header)  # optics: one unit rule (G02-41)
    apt_h = _hdr_float(header, "APTDIA", "APERTURE", "TELDIAM", "DIAMETER")
    tel_n = _norm(_hdr_str(header, "TELESCOP", "TELESCOPE"))

    scored: list[tuple[float, Detection]] = []
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
        scored.append((score, Detection(
            matched_id=int(row["ID"]),
            label=telescope_label(row),
            confidence=min(0.99, score),
            reasons=reasons,
        )))
    return scored


# --------------------------------------------------------------------------- #
# Sample-header discovery + orchestration
# --------------------------------------------------------------------------- #
#: SAMPLE-BOUNDED-2026-11: the sample search reads at most this many headers and visits at most this many
#: folders (breadth-first), so a render never walks a whole archive tree. 400 headers was the former bound; the
#: same budget for folders. A night's source folder holds its lights within a few levels.
SAMPLE_SCAN_MAX_HEADERS = 400
SAMPLE_SCAN_MAX_DIRS = 400
_LIGHT_DIR_WORDS = ("light", "object", "science")


def find_sample_light_header(
    source_root: str | Path, *, max_scan: int = SAMPLE_SCAN_MAX_HEADERS, max_dirs: int = SAMPLE_SCAN_MAX_DIRS,
    stats: dict[str, Any] | None = None,
) -> tuple[Any, str | None]:
    """A representative LIGHT header under ``source_root``, searched breadth-first (folders named like light /
    object / science first at every level) and stopped at the first frame with IMAGETYP LIGHT / OBJECT /
    SCIENCE. Bounded by ``max_scan`` headers and ``max_dirs`` folders; without a light frame in that budget the
    first non-calibration header found is returned, else (None, None). ``stats`` (optional) receives
    ``dirs`` / ``headers`` visited and ``exhausted`` (True when the bound stopped the search)."""
    import os  # noqa: PLC0415
    from collections import deque  # noqa: PLC0415

    from fits_suffixes import path_suffix_is_fits  # noqa: PLC0415

    st_ = stats if stats is not None else {}
    st_.update(dirs=0, headers=0, exhausted=False)
    try:
        from astropy.io import fits  # local import (heavy)
    except ImportError:
        return None, None
    root = Path(source_root)
    if not root.is_dir():
        return None, None
    queue: deque[Path] = deque([root])
    fallback: tuple[Any, str] | None = None
    while queue:
        if st_["dirs"] >= max_dirs or st_["headers"] >= max_scan:
            st_["exhausted"] = True
            break
        d = queue.popleft()
        st_["dirs"] += 1
        files: list[str] = []
        subdirs: list[Path] = []
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name.startswith("."):
                        continue
                    try:
                        if e.is_dir():
                            subdirs.append(Path(e.path))
                        elif e.is_file() and path_suffix_is_fits(e.name):
                            files.append(e.path)
                    except OSError:
                        continue
        except OSError:
            continue
        for fp in sorted(files):
            if st_["headers"] >= max_scan:
                break
            st_["headers"] += 1
            try:
                hdr = fits.getheader(fp)
            except Exception:  # noqa: BLE001 - an unreadable file is skipped; the next one is tried
                continue
            imagetyp = str(hdr.get("IMAGETYP", "") or "").upper()
            if "LIGHT" in imagetyp or "OBJECT" in imagetyp or "SCIENCE" in imagetyp:
                return hdr, fp
            if not any(c in imagetyp for c in ("DARK", "FLAT", "BIAS", "ZERO")) and fallback is None:
                fallback = (hdr, fp)
        queue.extend(sorted(subdirs, key=lambda p: (not any(w in p.name.lower() for w in _LIGHT_DIR_WORDS),
                                                     p.name.lower())))
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
    if header_site_reading(header) is None:
        gaps.append(
            {
                "field": "Observer site",
                "detail": "No site (SITELAT/SITELONG, OBSLAT/OBSLONG, OBSGEO-B/L) in the header - the "
                "Location selector was kept; the run uses the site selected there.",
                "fallback": "the site selected above",
            }
        )
    if not _has_any(header, "OBJCTRA", "RA", "CRVAL1", "OBJRA"):
        gaps.append(
            {
                "field": "Pointing (RA/Dec)",
                "detail": "No pointing/WCS keywords in header - field center will rely on blind plate-solve.",
                "fallback": "blind solve / manual draft center",
            }
        )
    return gaps


def no_light_message(source_root: str | Path, stats: dict[str, Any]) -> str:
    """SAMPLE-BOUNDED-2026-11: what the bounded search did when it found no light frame."""
    bound = (f" (search bound reached: {SAMPLE_SCAN_MAX_DIRS} folders / {SAMPLE_SCAN_MAX_HEADERS} headers - "
             "choose the night's own folder)") if stats.get("exhausted") else ""
    return (f"No light frame (IMAGETYP LIGHT / OBJECT / SCIENCE) found in {source_root}: "
            f"{stats.get('dirs', 0)} folder(s), {stats.get('headers', 0)} FITS header(s) searched{bound}")


def autodetect_from_source(
    source_root: str | Path,
    *,
    equipments: list[dict[str, Any]],
    telescopes: list[dict[str, Any]],
    locations: list[dict[str, Any]] | None = None,
) -> AutodetectReport:
    """Read a sample light header and fingerprint the camera, telescope and (RIG-AUTOSELECT-2026-11) the
    observing site against the DB rows."""
    stats: dict[str, Any] = {}
    hdr, path = sample_light_header(source_root, stats=stats)
    if hdr is None:
        return AutodetectReport(header_path=None, no_light=no_light_message(source_root, stats))
    eq = detect_equipment(hdr, equipments)
    tel = detect_telescope(hdr, telescopes)
    return AutodetectReport(
        equipment=eq,
        telescope=tel,
        location=detect_location(hdr, locations or []),
        header_path=path,
        unresolved=assess_unresolved(hdr, eq, tel),
        prefill=header_rig_prefill(hdr),
    )


# --------------------------------------------------------------------------- #
# Sample header of a source folder (RIG-GUARD-2026-11: read on every GUI render)
# --------------------------------------------------------------------------- #
_SAMPLE_CACHE: dict[str, tuple[str, int, int, Any]] = {}


def sample_light_header(source_root: str | Path, *, stats: dict[str, Any] | None = None) -> tuple[Any, str | None]:
    """:func:`find_sample_light_header`, remembered per source folder while the chosen file is unchanged
    (same path, mtime and size). The run-button guard calls it on every render; without the memo each
    render would walk the source tree again."""
    root = Path(source_root).expanduser()
    key = str(root.resolve()) if root.is_dir() else str(root)
    hit = _SAMPLE_CACHE.get(key)
    if hit is not None:
        path, mtime_ns, size, hdr = hit
        try:
            stt = Path(path).stat()
            if stt.st_mtime_ns == mtime_ns and stt.st_size == size:
                return hdr, path
        except OSError:
            pass
        _SAMPLE_CACHE.pop(key, None)
    hdr, path = find_sample_light_header(root, stats=stats)
    if hdr is not None and path is not None:
        stt = Path(path).stat()
        _SAMPLE_CACHE[key] = (path, stt.st_mtime_ns, stt.st_size, hdr)
    return hdr, path


# --------------------------------------------------------------------------- #
# Observing site (RIG-AUTOSELECT-2026-11 / SITE-MATCH-2026-11)
# --------------------------------------------------------------------------- #
def detect_location(header: Any, locations: list[dict[str, Any]]) -> SiteDetection:
    """The LOCATION row within :func:`site_match_tolerance_m` of the header site. No header site, no row
    within the tolerance, or more than one row within it (the header cannot tell them apart) -> no match."""
    hs = header_site_reading(header)
    if hs is None:
        return SiteDetection(reasons=[
            "no site in the header (SITELAT/SITELONG, OBSLAT/OBSLONG, OBSGEO-B/L) - Location selector kept"
        ])
    tol = site_match_tolerance_m(hs)
    det = SiteDetection(header_site=(hs.lat, hs.lon, hs.elev), tolerance_m=tol)
    near = sorted(
        ((site_distance_m(hs.lat, hs.lon, float(r["lat"]), float(r["lon"])), r) for r in locations),
        key=lambda t: t[0],
    )
    within = [(d, r) for d, r in near if d <= tol]
    det.n_within = len(within)
    where = f"header {hs.lat_key}/{hs.lon_key} {hs.lat:.5f}, {hs.lon:.5f}"
    if len(within) == 1:
        d, r = within[0]
        det.matched_id, det.label, det.distance_m = int(r["id"]), location_label(r), d
        det.reasons.append(f"{where} is {d:.0f} m from {r.get('name')} (tolerance {tol:.0f} m)")
    elif len(within) > 1:
        det.reasons.append(
            f"{where}: {len(within)} DB sites lie within the tolerance {tol:.0f} m ("
            + ", ".join(f"{r.get('name')} {d:.0f} m" for d, r in within) + ") - pick one"
        )
    elif near:
        d, r = near[0]
        det.distance_m = d
        det.reasons.append(
            f"{where}: no DB site within {tol:.0f} m (nearest {r.get('name')}, {d / 1000.0:.1f} km) - "
            "Location selector kept"
        )
    else:
        det.reasons.append(f"{where}: no site in the DB - Location selector kept")
    return det


# --------------------------------------------------------------------------- #
# RIG-GUARD-2026-11: the selection must not contradict the light headers
# --------------------------------------------------------------------------- #
_CLI_FLAG = {"camera": "camera", "telescope": "telescope"}
_HDR_KEYS = {"camera": "INSTRUME / NAXIS / XBINNING / GAIN", "telescope": "FOCALLEN / APTDIA / TELESCOP"}


def rig_header_mismatches(
    header: Any,
    *,
    equipment_id: int,
    telescope_id: int,
    equipments: list[dict[str, Any]],
    telescopes: list[dict[str, Any]],
    site: Any | None = None,
    header_path: str | None = None,
) -> list[str]:
    """Why the selected camera / telescope / site contradicts the sample light header (empty = no
    contradiction). A camera or telescope contradicts when ANOTHER row is a high-confidence match and the
    selected row is not; the site when the header site is farther than :func:`site_match_tolerance_m`
    from the selected site (``site``: anything with ``lat`` / ``lon`` / ``name``). Missing header facts
    are no contradiction."""
    src = f" ({Path(header_path).name})" if header_path else ""
    out: list[str] = []
    for kind, scored, sel in (
        ("camera", score_equipment(header, equipments), equipment_id),
        ("telescope", score_telescope(header, telescopes), telescope_id),
    ):
        best = _pick_best(scored)
        if best.band() != "high" or best.matched_id == int(sel):
            continue
        own = next((d for _, d in scored if d.matched_id == int(sel)), None)
        if own is not None and own.confidence >= AUTOFILL_THRESHOLD:
            continue
        own_txt = (f"{own.label} (confidence {own.confidence:.0%})" if own is not None else f"id={sel}")
        out.append(
            f"The light frames{src} match {kind} {best.label} (confidence {best.confidence:.0%}: "
            f"{'; '.join(best.reasons)}), but {kind} {own_txt} is selected. To resolve: select {best.label} "
            f"(button 'Auto-detect optics from FITS' or the {kind} selector; CLI: --{_CLI_FLAG[kind]} "
            f"{best.matched_id}), or, if these frames really were taken with the selected {kind}, correct its DB "
            f"row (Database Explorer) or the FITS headers ({_HDR_KEYS[kind]})"
        )
    hs = header_site_reading(header)
    if site is not None and hs is not None:
        tol = site_match_tolerance_m(hs)
        d = site_distance_m(hs.lat, hs.lon, float(site.lat), float(site.lon))
        if d > tol:
            out.append(
                f"The light frames{src} state the site {hs.lat_key}/{hs.lon_key} {hs.lat:.5f}, {hs.lon:.5f}, "
                f"{d / 1000.0:.1f} km from the selected site {getattr(site, 'name', '')} "
                f"({float(site.lat):.5f}, {float(site.lon):.5f}; same-site tolerance {tol:.0f} m). To resolve: "
                "select the header site (button 'Auto-detect optics from FITS'; CLI: --site <its LOCATION id>), "
                "or add it with 'Add this site from the header to the DB' (Database Explorer > Sites), or, if the "
                f"headers are wrong, correct {hs.lat_key}/{hs.lon_key} in the FITS headers"
            )
    return out


# --------------------------------------------------------------------------- #
# RIG-ADD-2026-11: header facts for the "Add to DB" forms (never gain / read noise / saturation / ADC)
# --------------------------------------------------------------------------- #
def _catalog_mode_for_frame(
    model: dict[str, Any], naxis1: float | None, naxis2: float | None, xbin: float, ybin: float
) -> str | None:
    """Key of the catalogue readout mode whose size matches the frame (NAXIS x binning); None when no mode
    or more than one matches (the catalogue path then takes the model's first mode, as the DB editor)."""
    if not naxis1 or not naxis2:
        return None
    hits = [str(md.get("key")) for md in model.get("modes", []) or []
            if isinstance(md, dict) and md.get("width") and md.get("height")
            and _dims_close(naxis1 * xbin, float(md["width"])) and _dims_close(naxis2 * ybin, float(md["height"]))]
    return hits[0] if len(hits) == 1 else None


def header_rig_prefill(header: Any) -> dict[str, dict[str, Any]]:
    """Pre-fill values of new EQUIPMENTS / TELESCOPE / LOCATION rows from a light header. Only geometry and
    names are taken: a header GAIN is usually the driver's setting index, not e-/ADU, so gain, read noise,
    saturation and ADC bits come from the camera catalogue or the DB editor, never from here."""
    cam = _hdr_str(header, "INSTRUME", "CAMERA", "INSTRUMENT", "DETNAM")
    naxis1 = _hdr_float(header, "NAXIS1", "IMAGEW")
    naxis2 = _hdr_float(header, "NAXIS2", "IMAGEH")
    xbin = _hdr_float(header, "XBINNING", "BINNING") or 1.0
    ybin = _hdr_float(header, "YBINNING", "BINNING") or 1.0
    catalog: list[dict[str, str]] = []
    if cam:
        from camera_profile import match_catalog_models  # noqa: PLC0415

        catalog = [{"key": str(m.get("key")), "model": str(m.get("model") or m.get("key")),
                    "mode": _catalog_mode_for_frame(m, naxis1, naxis2, xbin, ybin)}
                   for m in match_catalog_models(cam)]
    out: dict[str, dict[str, Any]] = {
        "equipment": {
            "camera_name": cam,
            "sensor_type": _hdr_str(header, "SENSOR", "SENSORTYP"),
            "sensor_size": (f"{int(round(naxis1 * xbin))}*{int(round(naxis2 * ybin))}" if naxis1 and naxis2 else ""),
            "pixel_um": header_native_pixel_um_mean(header) if header is not None else None,
            "catalog_models": catalog,
        },
        "telescope": {
            "telescope_name": _hdr_str(header, "TELESCOP", "TELESCOPE"),
            "focal_mm": header_focal_length_mm(header),
            "aperture_mm": _hdr_float(header, "APTDIA", "APERTURE", "TELDIAM", "DIAMETER"),
        },
    }
    hs = header_site_reading(header)
    if hs is not None:
        out["location"] = {
            "name": _hdr_str(header, "OBSERVAT", "SITENAME"),
            "lat": hs.lat, "lon": hs.lon, "elev": hs.elev,
        }
    return out


# --------------------------------------------------------------------------- #
# RIG-ADD-2026-11 (amended, Milan 2026-11-05): one press adds what the header states and the DB lacks
# --------------------------------------------------------------------------- #
def _not_in_db(det: Detection) -> bool:
    """No row matched (band none) and not because rows tie (a tie means the camera / telescope IS in the DB)."""
    return det.band() == "none" and not (det.reasons and det.reasons[0].startswith("tie:"))


def _fmt(v: Any, unit: str = "") -> str:
    return "EMPTY" if v is None or v == "" else f"{v:g}{unit}" if isinstance(v, float) else f"{v}{unit}"


def add_header_rig_to_db(db: Any, database_path: str, rep: AutodetectReport) -> list[str]:
    """Add the camera / telescope / site of the sample header that the DB does not have (``rep`` detections with
    no match; a site only when the header states one and no DB site lies within the SITE-MATCH tolerance), and
    point ``rep`` at the new rows (high band, so the GUI selects them). Names: INSTRUME / TELESCOP / OBSERVAT, else
    generated from the values. Gain, read noise, saturation and ADC bits never come from the header: a camera of
    the model catalogue (exactly one catalogue model for INSTRUME) is added through the catalogue, any other
    camera incomplete (RUN stays blocked until the DB editor completes it). A row with the same name and geometry
    is selected instead of added twice. Returns one line per row for the "Added to DB" summary."""
    from database import get_observer_locations  # noqa: PLC0415

    if rep.header_path is None:
        return []
    pre = rep.prefill or {}
    notes: list[str] = []
    added: set[str] = set()

    eq = pre.get("equipment") or {}
    if _not_in_db(rep.equipment) and eq.get("sensor_size"):
        size = str(eq["sensor_size"])
        px = eq.get("pixel_um")
        name = eq.get("camera_name") or (f"Camera {size.replace('*', 'x')}"
                                         + (f" {float(px):.2f}um" if px else ""))
        rows, _ = db.autodetect_candidates()
        same = next((r for r in rows if _norm(r.get("CAMERANAME")) == _norm(name)
                     and str(r.get("SENSORSIZE") or "") == size), None)
        try:
            if same is not None:
                rid, text = int(same["ID"]), f"camera {equipment_label(same)}: already in the DB (same name and sensor) - selected"
            else:
                cat = eq.get("catalog_models") or []
                if len(cat) == 1:
                    rid = db.add_equipment_from_catalog(cat[0]["key"], cat[0].get("mode"), name)
                    text = (f"camera id={rid} '{name}' from the camera catalogue (model {cat[0]['model']}, readout mode "
                            f"{cat[0].get('mode') or 'first'}): sensor, pixel, ADC bits and gain curve from the catalogue - "
                            "check the gain setting you use in Database Explorer > Cameras")
                else:
                    rid = db.add_equipment_from_header(camera_name=name, alias="", sensor_type=eq.get("sensor_type") or "",
                                                       sensor_size=size, pixel_size=px)
                    amb = (f" (INSTRUME fits several catalogue models: {', '.join(m['model'] for m in cat)})"
                           if cat else "")
                    text = (f"camera id={rid} '{name}': sensor {size}, pixel {_fmt(px, ' um')}, type "
                            f"{_fmt(eq.get('sensor_type'))}; INCOMPLETE{amb} - gain / read noise / saturation are never "
                            "taken from the header: enter them in Database Explorer > Cameras (RUN is blocked until then)")
            row = next(r for r in db.autodetect_candidates(equipment_id=rid)[0] if int(r["ID"]) == int(rid))
            rep.equipment = Detection(int(rid), equipment_label(row), 0.99, ["added to the DB from the header"])
            added.add("Equipment (camera)")
        except ValueError as exc:
            text = f"camera NOT added: {exc}"
        notes.append(text)

    tel = pre.get("telescope") or {}
    if _not_in_db(rep.telescope) and tel.get("focal_mm"):
        focal = float(tel["focal_mm"])
        name = tel.get("telescope_name") or f"Telescope f={focal:g} mm"
        _, rows = db.autodetect_candidates()
        same = next((r for r in rows if _norm(r.get("TELESCOPENAME")) == _norm(name)
                     and r.get("FOCAL") and math.isclose(float(r["FOCAL"]), focal, rel_tol=1e-6)), None)
        try:
            if same is not None:
                rid, text = int(same["ID"]), f"telescope {telescope_label(same)}: already in the DB (same name and focal) - selected"
            else:
                rid = db.add_telescope_from_header(telescope_name=name, alias="", diameter=tel.get("aperture_mm"),
                                                   focal=focal)
                text = (f"telescope id={rid} '{name}': focal {focal:g} mm, aperture {_fmt(tel.get('aperture_mm'), ' mm')}"
                        + ("" if tel.get("aperture_mm") else " (scintillation is left out until you enter it)"))
            row = next(r for r in db.autodetect_candidates(telescope_id=rid)[1] if int(r["ID"]) == int(rid))
            rep.telescope = Detection(int(rid), telescope_label(row), 0.99, ["added to the DB from the header"])
            added.add("Telescope")
        except ValueError as exc:
            text = f"telescope NOT added: {exc}"
        notes.append(text)

    loc = pre.get("location")
    if loc is not None and rep.location.addable:
        name = loc.get("name") or f"SITE {float(loc['lat']):.4f} {float(loc['lon']):.4f}"
        try:
            rid = db.add_location_from_header(place_name=name, latitude=loc["lat"], longitude=loc["lon"],
                                              altitude=loc.get("elev"))
            row = next(r for r in get_observer_locations(database_path, active_only=True) if int(r["id"]) == int(rid))
            rep.location.matched_id, rep.location.label, rep.location.n_within = int(rid), location_label(row), 1
            rep.location.reasons.append("added to the DB from the header")
            text = (f"site id={rid} '{name}': lat {float(loc['lat']):.6f}, lon {float(loc['lon']):.6f}, elevation "
                    + (f"{float(loc['elev']):g} m" if loc.get("elev") is not None
                       else "EMPTY - enter it in Database Explorer > Sites (RUN is blocked until then)"))
        except ValueError as exc:
            text = f"site NOT added: {exc}"
        notes.append(text)
    rep.unresolved = [g for g in rep.unresolved if g.get("field") not in added]
    return notes
