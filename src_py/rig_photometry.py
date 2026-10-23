"""Per-camera photometric properties from the EQUIPMENTS table (CFG-REVIEW-2026-10).

Properties of one camera belong to its EQUIPMENTS row, not to a global config value:

* ``ADC_BITS`` - ADC resolution. A camera with an N-bit ADC writes its samples into the 16-bit FITS
  container multiplied by 2^(16-N), so the native gain of the DB (e-/ADU native) divided by that
  container scale is the gain on the pixel values (fallback of the photon-transfer gain, WIDE-ERR-03).
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

#: FITS integer container of the calibrated frames.
CONTAINER_BITS = 16


@dataclass(frozen=True)
class RigPhotometry:
    equipment_id: int | None
    adc_bits: int | None = None
    sigma_sys_mag: float | None = None
    gain_adu: float | None = None  # EQUIPMENTS.GAIN_ADU, e-/ADU native
    read_noise_e: float | None = None  # EQUIPMENTS.READNOISE_E

    def container_scale(self) -> float:
        """2^(16 - ADC_BITS); 1.0 when ADC_BITS is unset (container ADU taken as native ADU)."""
        if self.adc_bits is None or not (1 <= int(self.adc_bits) <= CONTAINER_BITS):
            return 1.0
        return float(2 ** (CONTAINER_BITS - int(self.adc_bits)))


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
        want = [c for c in ("ADC_BITS", "SIGMA_SYS_MAG", "GAIN_ADU", "READNOISE_E") if c in cols]
        if not want:
            return RigPhotometry(eid)
        row = con.execute(f"SELECT {', '.join(want)} FROM EQUIPMENTS WHERE ID = ?;", (eid,)).fetchone()
    except sqlite3.Error as exc:
        LOGGER.warning("[RIG] EQUIPMENTS read failed (equipment_id=%s): %s", eid, exc)
        return RigPhotometry(eid)
    finally:
        con.close()
    if row is None:
        return RigPhotometry(eid)
    vals = dict(zip(want, row))
    bits = _finite_or_none(vals.get("ADC_BITS"))
    ssm = _finite_or_none(vals.get("SIGMA_SYS_MAG"))
    g = _finite_or_none(vals.get("GAIN_ADU"))
    rn = _finite_or_none(vals.get("READNOISE_E"))
    return RigPhotometry(
        eid,
        adc_bits=int(bits) if bits is not None and bits > 0 else None,
        sigma_sys_mag=ssm if ssm is not None and ssm >= 0 else None,
        gain_adu=g if g is not None and g > 0 else None,
        read_noise_e=rn if rn is not None and rn > 0 else None,
    )


@dataclass(frozen=True)
class PhotometricGain:
    """Gain on the pixel values (e-/ADU container) and read noise (e-) of the photometry."""

    gain: float
    read_noise: float
    gain_native: float
    container_scale: float
    gain_source: str
    read_noise_source: str


def photometric_gain_read_noise(header: Any, cfg: Any | None, equipment_id: int | None) -> PhotometricGain:
    """GAIN-ONE-2026-10: the one gain / read-noise authority of every stage (MASTERSTAR SNR, per-frame
    catalogs, aperture photometry errors, ePSF weights, variability). Native gain: FITS header (e-/ADU or a
    setting index mapped by the camera) -> EQUIPMENTS.GAIN_ADU; read noise: header ->
    EQUIPMENTS.READNOISE_E (param_resolver); neither known -> ValueError. Gain on the pixel values = native / container
    scale (ADC_BITS). Before, the per-frame catalogs and the MASTERSTAR SNR used the raw config gain
    1.0 / read noise 10 and aperture photometry tried a photon-transfer estimate whose empty-aperture variances
    include faint stars and sky structure (537: 0.48 vs 0.79, 534 g: 6.1 vs 12.5 e-/ADU) - never
    accepted on real data and removed."""
    from param_resolver import resolve_gain, resolve_read_noise  # noqa: PLC0415

    rig = rig_photometry(cfg, equipment_id)
    g_res = resolve_gain(header, equipment_id=equipment_id, cfg=cfg, db_value=rig.gain_adu)
    rn_res = resolve_read_noise(header, equipment_id=equipment_id, cfg=cfg, db_value=rig.read_noise_e)
    # CODE-REVIEW step 7B: no default gain / read noise - an error is better than wrong errors.
    if not (g_res.ok and g_res.value is not None and rn_res.ok and rn_res.value is not None):
        raise ValueError(
            f"Camera gain / read noise unknown (equipment_id={equipment_id}): set EQUIPMENTS.GAIN_ADU "
            "and READNOISE_E in Database Explorer (or EGAIN in the FITS header)."
        )
    g_nat = float(g_res.value)
    rn = float(rn_res.value)
    scale = rig.container_scale()
    return PhotometricGain(
        gain=g_nat / scale if scale > 0 else g_nat,
        read_noise=rn,
        gain_native=g_nat,
        container_scale=scale,
        gain_source=str(g_res.source),
        read_noise_source=str(rn_res.source),
    )
