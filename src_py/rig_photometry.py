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
        want = [c for c in ("ADC_BITS", "SIGMA_SYS_MAG") if c in cols]
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
    return RigPhotometry(
        eid,
        adc_bits=int(bits) if bits is not None and bits > 0 else None,
        sigma_sys_mag=ssm if ssm is not None and ssm >= 0 else None,
    )
