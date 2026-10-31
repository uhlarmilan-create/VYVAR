"""Per-camera photometric properties from the EQUIPMENTS table (CFG-REVIEW-2026-10).

Properties of one camera belong to its EQUIPMENTS row, not to a global config value:

* gain / read noise - ``photometric_gain_read_noise`` through the camera profile (``camera_profile``).
  A camera with an N-bit ADC (``ADC_BITS``) writes its samples into the 16-bit FITS container
  multiplied by 2^(16-N), so the native gain divided by that container scale is the gain on the
  pixel values.
* ``SIGMA_SYS_MAG`` - per-camera white noise floor (mag) used when the night has too few comparison
  stars to measure it (SIGMA-SYS-NIGHT-2026-10).

Read-only access by ``database_path`` of the config; the columns are created by
``VyvarDatabase._ensure_equipments_rig_photometry_columns``. A DB without them (not yet migrated)
reads as "unset".
"""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gain_ptc import container_scale

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RigPhotometry:
    equipment_id: int | None
    sigma_sys_mag: float | None = None


def _finite_or_none(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def rig_photometry(cfg: Any | None, equipment_id: int | None) -> RigPhotometry:
    """EQUIPMENTS row of ``equipment_id`` from ``cfg.database_path``; all fields unset when unknown."""
    if equipment_id is None:
        return RigPhotometry(None)
    eid = int(equipment_id)
    db_path = str(getattr(cfg, "database_path", "") or "").strip() if cfg is not None else ""
    if not db_path or not Path(db_path).is_file():
        return RigPhotometry(eid)
    try:
        con = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        LOGGER.warning("[RIG] database %s not readable: %s", db_path, exc)
        return RigPhotometry(eid)
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info('EQUIPMENTS');").fetchall()}
        if "SIGMA_SYS_MAG" not in cols:
            return RigPhotometry(eid)
        row = con.execute("SELECT SIGMA_SYS_MAG FROM EQUIPMENTS WHERE ID = ?;", (eid,)).fetchone()
    except sqlite3.Error as exc:
        LOGGER.warning("[RIG] EQUIPMENTS read failed (equipment_id=%s): %s", eid, exc)
        return RigPhotometry(eid)
    finally:
        con.close()
    ssm = _finite_or_none(row[0]) if row is not None else None
    return RigPhotometry(eid, sigma_sys_mag=ssm if ssm is not None and ssm >= 0 else None)


@dataclass(frozen=True)
class PhotometricGain:
    """Gain on the pixel values (e-/ADU container) and the per-pixel noise (e-) of the photometry.

    G06-25: ``read_noise`` is the per-pixel noise term of the CCD equation other than the sky,
    sqrt(RN^2 + F_D + g^2 sigma_f^2) (Collins et al. 2017, AIJ eq. B1; Merline & Howell 1995): camera read
    noise, dark-current electrons F_D and the digitisation term (see ``ccd_pixel_noise_e``);
    ``read_noise_camera`` is RN alone."""

    gain: float
    read_noise: float
    gain_native: float
    container_scale: float
    gain_source: str
    read_noise_source: str
    #: G06-30: the ``param_resolver.Resolved`` records behind ``gain_native`` / ``read_noise`` (provenance)
    gain_resolved: Any = field(default=None, compare=False)
    read_noise_resolved: Any = field(default=None, compare=False)
    read_noise_camera: float | None = None


def _header_int(header: Any, key: str) -> int | None:
    try:
        v = header.get(key) if header is not None else None
        return int(v) if v is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def ccd_pixel_noise_e(
    header: Any, *, gain_file: float, read_noise_e: float, rn_measured_at_binning: bool
) -> tuple[float, dict[str, Any]]:
    """G06-25: per-pixel noise [e-] of the CCD equation other than the sky (AIJ eq. B1 grouping):

        sqrt(RN^2 + F_D + g^2 sigma_f^2)

    F_D = dark current x exposure [e-]: ``VY_CDKR`` (CAL-DIAG fitted dark slope, file ADU/s per light pixel)
    x ``EXPTIME`` x g; 0 and ``dark_current: unmeasured`` when CAL-DIAG measured no slope.
    sigma_f = g_LSB / sqrt(12) [file ADU] with g_LSB the raw granularity ``VY_LSB`` (G06-34); left out when the
    read noise is a dark-pair measurement at the frame binning (the digitisation is inside it,
    ``rn_includes_digitization``); 0 and ``digitization: unmeasured`` when VY_LSB is absent.
    Returns (noise e-, facts for pipeline_meta.resolved_facts.read_noise)."""
    g = float(gain_file)
    facts: dict[str, Any] = {"read_noise_camera_e": float(read_noise_e)}
    rate = _header_float(header, "VY_CDKR")
    exptime = _header_float(header, "EXPTIME", "EXPOSURE")
    if rate is not None and exptime is not None:
        dark_e = max(rate, 0.0) * exptime * g
        facts.update({"dark_current": "measured", "dark_e_per_px": dark_e})
    else:
        dark_e = 0.0
        facts.update({"dark_current": "unmeasured", "dark_e_per_px": 0.0})
    facts["rn_includes_digitization"] = bool(rn_measured_at_binning)
    lsb = _header_int(header, "VY_LSB")
    if rn_measured_at_binning:
        digit_e2 = 0.0
        facts["digitization"] = "in_read_noise"
    elif lsb is not None and lsb > 0:
        digit_e2 = (g * float(lsb)) ** 2 / 12.0
        facts["digitization"] = "added"
    else:
        digit_e2 = 0.0
        facts["digitization"] = "unmeasured"
    facts["digitization_e2"] = digit_e2
    noise = math.sqrt(float(read_noise_e) ** 2 + dark_e + digit_e2)
    facts["pixel_noise_e"] = noise
    unmeasured = [k for k in ("dark_current", "digitization") if facts[k] == "unmeasured"]
    if unmeasured and tuple(unmeasured) not in _NOISE_WARNED:
        _NOISE_WARNED.add(tuple(unmeasured))
        LOGGER.warning("[CCD] %s not measured for this frame (no %s) - term left out of the CCD equation",
                       " / ".join(unmeasured), " / ".join("VY_CDKR" if k == "dark_current" else "VY_LSB" for k in unmeasured))
    return noise, facts


_NOISE_WARNED: set[tuple[str, ...]] = set()


def _header_float(header: Any, *keys: str) -> float | None:
    for k in keys:
        try:
            v = header.get(k) if header is not None else None
            f = float(v) if v is not None else None
        except (AttributeError, TypeError, ValueError):
            continue
        if f is not None and math.isfinite(f):
            return f
    return None


def photometric_gain_read_noise(
    header: Any, cfg: Any | None, equipment_id: int | None, *, noise_terms: bool = True
) -> PhotometricGain:
    """GAIN-ONE-2026-10: the one gain / read-noise authority of every stage (MASTERSTAR SNR, per-frame
    catalogs, aperture photometry errors, ePSF weights, variability). CAMERA-PROFILE-2026-10: header
    e-/ADU, else the camera profile at the header gain setting and binning (``param_resolver``,
    ``camera_profile``); neither known -> ValueError. Gain on the pixel values = native / container
    scale (ADC_BITS of the camera, else of its catalog mode)."""
    from camera_profile import load_profile  # noqa: PLC0415
    from param_resolver import resolve_gain, resolve_read_noise  # noqa: PLC0415

    db_path = str(getattr(cfg, "database_path", "") or "") if cfg is not None else ""
    prof = load_profile(db_path, equipment_id)
    g_res = resolve_gain(header, equipment_id=equipment_id, cfg=cfg, profile=prof)
    rn_res = resolve_read_noise(header, equipment_id=equipment_id, cfg=cfg, profile=prof)
    # CODE-REVIEW step 7B: no default gain / read noise - an error is better than wrong errors.
    if not (g_res.ok and g_res.value is not None and rn_res.ok and rn_res.value is not None):
        why = "; ".join(g_res.warnings + rn_res.warnings)
        raise ValueError(
            f"Camera gain / read noise unknown (camera {prof.name or equipment_id}): choose the camera model "
            "in Database Explorer > Cameras, or enter the gain and read noise at the gain setting you use. "
            + why
        )
    g_nat = float(g_res.value)
    rn = float(rn_res.value)
    scale = container_scale(prof.adc_bits)
    # G06-34: the container scale used and the raw LSB granularity measured at calibration (VY_LSB)
    g_res.facts.update({"container_scale": scale, "container_scale_measured": _header_int(header, "VY_LSB")})
    noise = rn
    if noise_terms:  # False: a raw frame (C2 sky-pair diagnostic) - gain and camera read noise only
        noise, nfacts = ccd_pixel_noise_e(
            header, gain_file=g_nat / scale, read_noise_e=rn,
            rn_measured_at_binning=str(rn_res.source) == "measured" and bool(getattr(rn_res, "exact_binning", False)),
        )
        rn_res.facts.update(nfacts)
    return PhotometricGain(
        gain=g_nat / scale,
        read_noise=noise,
        read_noise_camera=rn,
        gain_native=g_nat,
        container_scale=scale,
        gain_source=str(g_res.source),
        read_noise_source=str(rn_res.source),
        gain_resolved=g_res,
        read_noise_resolved=rn_res,
    )
