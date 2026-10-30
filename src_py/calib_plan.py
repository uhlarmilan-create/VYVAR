"""CAL-2026-10 calibration plan - one function for the GUI and the night runner.

dev/results/specs/VYVAR_CALIBRATION_SPEC.md sections 3-5. For every LIGHT group (filter, exposure,
binning) the plan chooses a DARK and a FLAT from the library under the validity-window
rule, records the flat offset handling, and gives a status.

Masters are made only in the Calibration Library. Masters lying next to the LIGHTs
(e.g. C-Munipack / SIPS masters copied with the night) are recognised from their
headers; with ``masters_from="lights"`` (default) they are preferred over the
library and, at run time, imported into the library (origin IMPORTED).
Raw DARK / FLAT frames in the source folder are only reported.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from astropy.io import fits

import calib_library as cl
import calib_masters as cm
from infolog import log_event

STATUS_OK = "OK"
STATUS_NO_DARK = "MISSING_DARK"
STATUS_NO_FLAT = "MISSING_FLAT"
STATUS_NO_MASTERS = "MISSING_DARK_AND_FLAT"
STATUS_NO_RIG = "NO_CAMERA_SELECTED"
STATUS_PRECAL = "PRECALIBRATED"
#: G01-01 (decision A1): no valid flat in the window; a FLAT of the same filter from another night is used
STATUS_FLAT_SUBSTITUTE = "FLAT_FROM_OTHER_NIGHT"

FROM_LIGHTS = "lights"
FROM_LIBRARY = "library"
MASTERS_FROM = (FROM_LIGHTS, FROM_LIBRARY)

MIN_FRAMES_WARN = 5


@dataclass
class GroupPlan:
    group_key: str
    filter_name: str
    exposure_s: float
    binning: int
    gain: float
    offset: int | None
    temp: float | None
    n_lights: int
    date_first: datetime | None
    date_last: datetime | None
    dark: cl.LibraryMaster | None = None
    dark_reason: str = ""
    dark_rebin: int = 1
    dark_from: str = ""
    flat: cl.LibraryMaster | None = None
    flat_reason: str = ""
    flat_rebin: int = 1
    flat_from: str = ""
    flat_offset_handling: str = ""
    flat_offset_adu: float | None = None
    #: capture date (ISO) of a FLAT from another night used as a flagged substitute, '' otherwise
    flat_substitute: str = ""
    status: str = ""
    messages: list[str] = field(default_factory=list)

    def row(self) -> dict[str, Any]:
        """One row of the plan table (GUI / log / JSON)."""
        def _m(m: cl.LibraryMaster | None) -> str | None:
            return Path(m.path).name if m is not None else None

        return {
            "group": self.group_key,
            "filter": self.filter_name,
            "exposure_s": self.exposure_s,
            "binning": self.binning,
            "gain": self.gain,
            "offset": self.offset,
            "temp_c": self.temp,
            "n_lights": self.n_lights,
            "date_first": self.date_first.isoformat() if self.date_first else None,
            "date_last": self.date_last.isoformat() if self.date_last else None,
            "dark": _m(self.dark),
            "dark_path": self.dark.path if self.dark else None,
            "dark_capture": self.dark.capture_utc.isoformat() if self.dark and self.dark.capture_utc else None,
            "dark_origin": f"{self.dark.origin}/{self.dark.origin_software}" if self.dark else None,
            "dark_rebin": self.dark_rebin,
            "dark_reason": self.dark_reason,
            "dark_from": self.dark_from,
            "flat": _m(self.flat),
            "flat_path": self.flat.path if self.flat else None,
            "flat_capture": self.flat.capture_utc.isoformat() if self.flat and self.flat.capture_utc else None,
            "flat_origin": f"{self.flat.origin}/{self.flat.origin_software}" if self.flat else None,
            "flat_rebin": self.flat_rebin,
            "flat_reason": self.flat_reason,
            "flat_from": self.flat_from,
            "flat_offset_handling": self.flat_offset_handling,
            "flat_offset_adu": self.flat_offset_adu,
            "flat_substitute": self.flat_substitute or None,
            "status": self.status,
            "messages": list(self.messages),
        }


@dataclass
class CalibrationPlan:
    camera_id: int | None
    telescope_id: int | None
    groups: dict[str, GroupPlan]
    built: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    precalibrated: bool = False
    masters_from: str = FROM_LIGHTS
    source_masters: list[str] = field(default_factory=list)
    imported: list[str] = field(default_factory=list)

    def dark_by_group(self) -> dict[str, str | None]:
        return {k: (g.dark.path if g.dark else None) for k, g in self.groups.items()}

    def flat_by_group(self) -> dict[str, str | None]:
        return {k: (g.flat.path if g.flat else None) for k, g in self.groups.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": "CAL-2026-10",
            "camera_id": self.camera_id,
            "telescope_id": self.telescope_id,
            "precalibrated": self.precalibrated,
            "masters_from": self.masters_from,
            "masters_next_to_lights": list(self.source_masters),
            "imported_into_library": list(self.imported),
            "warnings": list(self.warnings),
            "groups": [g.row() for g in self.groups.values()],
        }

    def write_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
        return p

    def log(self) -> None:
        log_event(f"CALIBRATION PLAN camera={self.camera_id} telescope={self.telescope_id} "
                  f"masters_from={self.masters_from} masters_next_to_lights={len(self.source_masters)}"
                  + (" PRECALIBRATED (calibration skipped)" if self.precalibrated else ""))
        for g in self.groups.values():
            log_event(
                f"  [{g.status}] {g.group_key} n={g.n_lights} | dark={Path(g.dark.path).name if g.dark else '-'}"
                f"{f' [{g.dark_from}]' if g.dark else ''}"
                f"{f' (rebin {g.dark_rebin}x)' if g.dark_rebin > 1 else ''} | flat="
                f"{Path(g.flat.path).name if g.flat else '-'}{f' [{g.flat_from}]' if g.flat else ''} "
                f"offset={g.flat_offset_handling or '-'}"
                + (f" | {'; '.join(g.messages)}" if g.messages else "")
            )
        for w in self.warnings:
            log_event(f"  ! {w}")


# ---------------------------------------------------------------------------
# raw calibration frames in the source folder -> masters (spec 5, K6)
# ---------------------------------------------------------------------------
def _hdr(p: Path) -> fits.Header:
    with fits.open(p, memmap=True) as h:
        return h[0].header.copy()


def _temp_bucket(hdr: fits.Header) -> float | None:
    v = cl._hfloat(hdr, "SET-TEMP")
    if v is None:
        v = cl.header_temp(hdr)
    return round(float(v)) if v is not None else None


def _is_darkflat_header(hdr: fits.Header) -> bool:
    it = str(hdr.get("IMAGETYP") or hdr.get("FRAME") or "").lower().replace(" ", "").replace("_", "")
    return "darkflat" in it or "flatdark" in it


def build_masters_from_source(
    *,
    dark_files: Sequence[str | Path],
    flat_files: Sequence[str | Path],
    light_exposures: Sequence[float],
    db: Any,
    camera_id: int,
    telescope_id: int | None,
    library_root: str | Path,
    method: str = cm.DEFAULT_COMBINE,
    temp_tol: float = 1.0,
    equipment_bayer_mask: str | None = None,
    dark_kind: str | None = None,
) -> tuple[list[str], list[str]]:
    """Group raw frames and build DARK, DARKFLAT and FLAT masters into the library
    (Calibration Library tab).

    A raw dark is a DARKFLAT when its IMAGETYP says so, or when its exposure equals
    a raw flat exposure and no LIGHT exposure; ``dark_kind`` (DARK / DARKFLAT) forces
    the kind for all dark frames. Returns (built names, messages).
    """
    lib = Path(library_root)
    built: list[str] = []
    msgs: list[str] = []
    flat_hdrs = {Path(f): _hdr(Path(f)) for f in flat_files}
    flat_exps = {round(cl._hfloat(h, "EXPTIME", "EXPOSURE") or 0.0, 4) for h in flat_hdrs.values()}
    light_exps = {round(float(e), 4) for e in light_exposures}

    groups: dict[tuple, list[Path]] = defaultdict(list)
    for f in dark_files:
        p = Path(f)
        h = _hdr(p)
        exp = round(cl._hfloat(h, "EXPTIME", "EXPOSURE") or 0.0, 4)
        if dark_kind in (cl.KIND_DARK, cl.KIND_DARKFLAT):
            kind = dark_kind
        else:
            kind = cl.KIND_DARKFLAT if (_is_darkflat_header(h) or (exp in flat_exps and exp not in light_exps)) else cl.KIND_DARK
        groups[(kind, exp, cl.header_gain(h), cl.header_offset(h), cl.header_binning(h), _temp_bucket(h))].append(p)
    for (kind, exp, gain, off, b, t), files in sorted(groups.items(), key=lambda x: (x[0][0] != cl.KIND_DARKFLAT, str(x[0]))):
        if len(files) < MIN_FRAMES_WARN:
            msgs.append(f"! only {len(files)} raw {kind} frames for {exp:g}s bin{b} - master will be noisy")
        r = cm.build_dark(files, kind=kind, method=method, library_root=lib, db=db,
                          camera_id=camera_id, telescope_id=telescope_id)
        msgs += r.messages
        if r.path:
            built.append(r.path.name)

    fgroups: dict[tuple, list[Path]] = defaultdict(list)
    for p, h in flat_hdrs.items():
        fgroups[(cl.header_filter(h), cl.header_gain(h), cl.header_offset(h), cl.header_binning(h))].append(p)
    for (flt, gain, off, b), files in sorted(fgroups.items(), key=lambda x: str(x[0])):
        if telescope_id is None:
            msgs.append(f"[X] FLAT {flt}: no telescope selected - cannot build")
            continue
        h0 = flat_hdrs[files[0]]
        fexp = cl._hfloat(h0, "EXPTIME", "EXPOSURE") or 0.0
        fdates = cl.iter_light_dates(files)
        dfm = cl.find_dark(db, camera_id=camera_id, gain=gain, offset=off, exposure=fexp, binning=b,
                           temp=cl.header_temp(h0), temp_tol=temp_tol, light_dates=fdates,
                           validity_days=3650.0, kind=cl.KIND_DARKFLAT)
        if len(files) < MIN_FRAMES_WARN:
            msgs.append(f"! only {len(files)} raw flats for {flt} bin{b} - master will be noisy")
        r = cm.build_flat(files, method=method, library_root=lib, db=db, camera_id=camera_id,
                          telescope_id=telescope_id,
                          darkflat_path=Path(dfm.master.path) if (dfm.master and dfm.rebin_from == 1) else None,
                          temp_tol=temp_tol, equipment_bayer_mask=equipment_bayer_mask)
        msgs += r.messages
        if r.path:
            built.append(r.path.name)
    return built, msgs


# ---------------------------------------------------------------------------
# masters lying next to the LIGHTs
# ---------------------------------------------------------------------------
def masters_next_to_lights(
    paths: Sequence[str | Path],
    *,
    camera_id: int | None,
    telescope_id: int | None,
    light_instrume: str = "",
) -> tuple[list[cl.LibraryMaster], list[str]]:
    """Read master files found in the source tree into in-memory masters.

    A master whose INSTRUME differs from the LIGHTs' INSTRUME is not used (it was
    taken with another camera). Returns (masters, messages).
    """
    out: list[cl.LibraryMaster] = []
    msgs: list[str] = []
    li = str(light_instrume or "").strip().casefold()
    for f in paths:
        p = Path(f)
        try:
            ins = str(_hdr(p).get("INSTRUME") or "").strip().casefold()
        except (OSError, ValueError):
            ins = ""
        if li and ins and ins != li:
            msgs.append(f"! {p.name}: INSTRUME '{ins}' is not the LIGHTs' camera '{li}' - not used")
            continue
        m, why = cl.master_from_header(p, camera_id=camera_id, telescope_id=telescope_id)
        if m is None:
            msgs.append(f"! {why} - not used")
            continue
        out.append(m)
    return out, msgs


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def _pick_source(order: Sequence[str], finders: dict[str, Any]) -> tuple[cl.MatchResult, str]:
    """First valid match in ``order``; if none, the reasons of all tried sources."""
    reasons: list[str] = []
    for src in order:
        r = finders[src]()
        if r.master is not None:
            return r, src
        reasons.append(f"{'LIGHT folder' if src == FROM_LIGHTS else 'library'}: {r.reason}")
    return cl.MatchResult(None, " | ".join(reasons)), ""


def build_calibration_plan(
    observation_groups: dict[str, dict[str, Any]],
    *,
    db: Any,
    camera_id: int | None,
    telescope_id: int | None,
    dark_validity_days: float,
    flat_validity_days: float,
    temp_tol: float,
    precalibrated: bool = False,
    built: Sequence[str] = (),
    warnings: Sequence[str] = (),
    source_masters: Sequence[cl.LibraryMaster] = (),
    masters_from: str = FROM_LIGHTS,
) -> CalibrationPlan:
    """Choose masters for every LIGHT group. ``observation_groups`` as built by
    ``importer.smart_scan_source`` (keys: filter, exposure_s, binning, gain, temp, light_paths).

    ``source_masters``: masters found next to the LIGHTs. ``masters_from`` decides which
    source is tried first (``lights`` or ``library``); the other is the fallback. The
    matching rules (camera, gain, offset, exposure, binning, temperature, filter,
    telescope, validity window) are the same for both.
    """
    mf = masters_from if masters_from in MASTERS_FROM else FROM_LIGHTS
    plan = CalibrationPlan(camera_id, telescope_id, {}, list(built), list(warnings), bool(precalibrated),
                           masters_from=mf, source_masters=[Path(m.path).name for m in source_masters])
    if db is not None:
        cl.ensure_schema(db)
    src = list(source_masters)
    order = [mf] + [x for x in MASTERS_FROM if x != mf]
    if not src:
        order = [FROM_LIBRARY]
    for gk, g in sorted(observation_groups.items()):
        paths = list(g.get("light_paths") or [])
        hdr0 = _hdr(Path(paths[0])) if paths else fits.Header()
        dates = cl.iter_light_dates(paths)
        gp = GroupPlan(
            group_key=gk,
            filter_name=str(g.get("filter") or "NoFilter"),
            exposure_s=float(g.get("exposure_s") or 0.0),
            binning=int(g.get("binning") or 1),
            gain=cl.gain_key(g.get("gain")),
            offset=cl.header_offset(hdr0),
            temp=(float(g["temp"]) if g.get("temp") is not None and math.isfinite(float(g["temp"])) else None),
            n_lights=len(paths),
            date_first=min(dates) if dates else None,
            date_last=max(dates) if dates else None,
        )
        plan.groups[gk] = gp
        if precalibrated:
            gp.status = STATUS_PRECAL
            gp.messages.append("LIGHTs used as they are (pre-calibrated elsewhere or deliberately uncalibrated)")
            continue
        if camera_id is None or db is None:
            gp.status = STATUS_NO_RIG
            gp.messages.append("select the camera (and telescope) to match masters")
            continue
        if not dates:
            gp.messages.append("LIGHT headers have no DATE-OBS - validity cannot be checked")

        def _dark(cands: Sequence[cl.LibraryMaster] | None, gp: GroupPlan = gp, dates: list = dates) -> cl.MatchResult:
            return cl.find_dark(db, camera_id=int(camera_id), gain=gp.gain, offset=gp.offset,
                                exposure=gp.exposure_s, binning=gp.binning, temp=gp.temp, temp_tol=temp_tol,
                                light_dates=dates, validity_days=dark_validity_days, candidates=cands)

        def _flat(cands: Sequence[cl.LibraryMaster] | None, gp: GroupPlan = gp, dates: list = dates) -> cl.MatchResult:
            return cl.find_flat(db, camera_id=int(camera_id), telescope_id=telescope_id, filter_name=gp.filter_name,
                                gain=gp.gain, binning=gp.binning, light_dates=dates,
                                validity_days=flat_validity_days, candidates=cands)

        d, gp.dark_from = _pick_source(order, {FROM_LIGHTS: lambda: _dark(src), FROM_LIBRARY: lambda: _dark(None)})
        gp.dark, gp.dark_reason, gp.dark_rebin = d.master, d.reason, d.rebin_from
        f, gp.flat_from = _pick_source(order, {FROM_LIGHTS: lambda: _flat(src), FROM_LIBRARY: lambda: _flat(None)})
        gp.flat, gp.flat_reason, gp.flat_rebin = f.master, f.reason, f.rebin_from
        if gp.flat is None and dates:
            # G01-01 (decision A1): no valid flat in the window - nearest FLAT of the same filter from
            # another night, recorded as a flagged substitute (plan, manifest, frame headers)
            def _sub(cands: Sequence[cl.LibraryMaster] | None, gp: GroupPlan = gp, dates: list = dates) -> cl.MatchResult:
                return cl.find_flat_substitute(db, camera_id=int(camera_id), telescope_id=telescope_id,
                                               filter_name=gp.filter_name, gain=gp.gain, binning=gp.binning,
                                               light_dates=dates, candidates=cands)

            fs, from_s = _pick_source(order, {FROM_LIGHTS: lambda: _sub(src), FROM_LIBRARY: lambda: _sub(None)})
            if fs.master is not None:
                gp.messages.append(f"no valid FLAT ({gp.flat_reason}); {fs.reason} - flagged")
                gp.flat, gp.flat_from, gp.flat_rebin = fs.master, from_s, fs.rebin_from
                gp.flat_reason = fs.reason
                gp.flat_substitute = fs.master.capture_utc.date().isoformat() if fs.master.capture_utc else "unknown"
        if gp.flat is not None:
            _resolve_flat_offset(db, gp)
        if gp.dark is None and gp.flat is None:
            gp.status = STATUS_NO_MASTERS
        elif gp.dark is None:
            gp.status = STATUS_NO_DARK
        elif gp.flat is None:
            gp.status = STATUS_NO_FLAT
        elif gp.flat_substitute:
            gp.status = STATUS_FLAT_SUBSTITUTE
        else:
            gp.status = STATUS_OK
        if gp.dark is None:
            gp.messages.append(gp.dark_reason)
        if gp.flat is None:
            gp.messages.append(gp.flat_reason)
            if gp.dark is not None:
                gp.messages.append("calibrated with the DARK only and flagged: relative photometry, "
                                   "not for AAVSO/VarAstro submission")
        if gp.dark is None and gp.flat is not None:
            gp.messages.append("without a DARK the FLAT is not applied (offset would remain in the LIGHT)")
    plan.log()
    return plan


def import_planned_source_masters(
    plan: CalibrationPlan,
    *,
    db: Any,
    library_root: str | Path,
) -> list[str]:
    """Import the masters the plan takes from the LIGHT folder into the library.

    Each file is copied once (identical re-imports keep the existing copy), registered
    as IMPORTED with the program named in its header and linked to the selected camera
    (and telescope for a FLAT). The plan then points at the library copies. Darks are
    imported first so that a flat with unknown offset can be corrected from them.
    """
    msgs: list[str] = []
    if plan.camera_id is None or db is None:
        return msgs
    done: dict[str, cl.LibraryMaster | None] = {}

    def _imp(m: cl.LibraryMaster) -> cl.LibraryMaster | None:
        key = str(Path(m.path))
        if key in done:
            return done[key]
        r = cm.import_master(m.path, kind=m.kind, library_root=Path(library_root), db=db,
                             camera_id=int(plan.camera_id),
                             telescope_id=plan.telescope_id if m.kind == cl.KIND_FLAT else None,
                             origin_software=m.origin_software,
                             flat_offset_removed=None)  # read from the header
        msgs.extend(r.messages)
        done[key] = r.master
        if r.master is not None and r.path is not None:
            plan.imported.append(r.path.name)
        return r.master

    for gp in plan.groups.values():
        if gp.dark is not None and gp.dark_from == FROM_LIGHTS:
            lm = _imp(gp.dark)
            if lm is not None:
                gp.dark = lm
    for gp in plan.groups.values():
        if gp.flat is not None and gp.flat_from == FROM_LIGHTS:
            lm = _imp(gp.flat)
            if lm is not None:
                gp.flat = lm
                _resolve_flat_offset(db, gp)
                gp.messages = list(dict.fromkeys(gp.messages))
    for x in msgs:
        log_event(f"CALIB MASTERS FROM LIGHT FOLDER: {x}")
    return msgs


def _resolve_flat_offset(db: Any, gp: GroupPlan, ) -> None:
    """Record how the chosen flat's offset is handled; measure P for legacy/unknown flats."""
    m = gp.flat
    assert m is not None
    try:
        with fits.open(m.path, memmap=True) as h:
            hs = h[0].header.get("VYOFFSUB")
    except OSError:
        hs = None
    if hs is not None:
        gp.flat_offset_handling, gp.flat_offset_adu = str(hs), m.offset_adu
        return
    if m.offset_handling == cl.OFFSET_ALREADY:
        gp.flat_offset_handling, gp.flat_offset_adu = cl.OFFSET_ALREADY, None
        return
    est = cl.measure_offset_adu(db, camera_id=int(m.camera_id or 0), gain=m.gain, offset=m.offset,
                                binning=m.binning, temp=m.temp, temp_tol=None)
    if est.adu is not None:
        cl.update_flat_offset(db, m.path, handling=cl.OFFSET_UNKNOWN, offset_adu=est.adu)
        m.offset_adu = est.adu
        gp.flat_offset_handling, gp.flat_offset_adu = "removed-at-use", est.adu
        gp.messages.append(f"flat built without offset removal: {est.adu:.1f} ADU removed at use ({est.method})")
        # G01-02: show early (scan) that the run will refuse this flat; the calibration stops on it
        import numpy as np  # noqa: PLC0415

        from calibration import MasterResamplingError, check_flat_offset_removable  # noqa: PLC0415

        with fits.open(m.path, memmap=False) as h:
            fmed = float(np.nanmedian(np.asarray(h[0].data, dtype=np.float32)))
        try:
            check_flat_offset_removable(fmed, float(est.adu), Path(m.path).name)
        except MasterResamplingError as exc:
            gp.messages.append(f"[X] {exc}")
    else:
        gp.flat_offset_handling, gp.flat_offset_adu = cl.OFFSET_NONE, None
        gp.messages.append("flat offset unknown and no dark to measure it - flat used as is")
