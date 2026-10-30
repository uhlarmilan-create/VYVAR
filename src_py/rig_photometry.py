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

LOGGER = logging.getLogger(__name__)

#: FITS integer container of the calibrated frames.
CONTAINER_BITS = 16


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
    """Gain on the pixel values (e-/ADU container) and read noise (e-) of the photometry."""

    gain: float
    read_noise: float
    gain_native: float
    container_scale: float
    gain_source: str
    read_noise_source: str
    #: G06-30: the ``param_resolver.Resolved`` records behind ``gain_native`` / ``read_noise`` (provenance)
    gain_resolved: Any = field(default=None, compare=False)
    read_noise_resolved: Any = field(default=None, compare=False)


def photometric_gain_read_noise(header: Any, cfg: Any | None, equipment_id: int | None) -> PhotometricGain:
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
    bits = prof.adc_bits
    scale = float(2 ** (CONTAINER_BITS - int(bits))) if bits is not None and 1 <= int(bits) <= CONTAINER_BITS else 1.0
    return PhotometricGain(
        gain=g_nat / scale if scale > 0 else g_nat,
        read_noise=rn,
        gain_native=g_nat,
        container_scale=scale,
        gain_source=str(g_res.source),
        read_noise_source=str(rn_res.source),
        gain_resolved=g_res,
        read_noise_resolved=rn_res,
    )
