"""Diagnostic sigma budget: Howell (production) + Young/Osborn scintillation.

Production LC export wires scintillation via ``sigma_floor_core.scintillation_mag_per_epoch``
(batch D, P-02). This module also provides diagnostic harness variants.
See dev/results/specs/VYVAR_SIGMA_BUDGET_SPEC.md.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import AppConfig


# Osborn et al. (2015) MNRAS 452, 1707 -- modified Young, eq. (7)
_TURBULENCE_SCALE_HEIGHT_M = 8000.0
_YOUNG_PREFACTOR = 10.0e-6
OSBORN_CY_DEFAULT = 1.5

from mag_constants import MAG_ERR_SCALE


LOGGER = logging.getLogger(__name__)
_WARNED_NO_SCINT: set[tuple[Any, str]] = set()


@dataclass
class RigScintillationParams:
    draft_id: int | None
    setup: str
    telescope_diameter_m: float
    altitude_m: float
    exposure_s: float
    c_y: float
    source_notes: list[str]


def scintillation_sigma(
    *,
    telescope_diameter_m: float,
    airmass: float,
    exposure_s: float,
    altitude_m: float,
    c_y: float = OSBORN_CY_DEFAULT,
) -> float:
    """Relative flux scintillation sigma (modified Young / Osborn 2015)."""
    if not (
        math.isfinite(telescope_diameter_m)
        and telescope_diameter_m > 0
        and math.isfinite(exposure_s)
        and exposure_s > 0
        and math.isfinite(airmass)
        and airmass >= 1.0
        and math.isfinite(altitude_m)
    ):
        return float("nan")
    zenith_dist_rad = math.acos(min(1.0, 1.0 / float(airmass)))
    cos_z = max(math.cos(zenith_dist_rad), 1e-6)
    alt_factor = math.exp(-2.0 * float(altitude_m) / _TURBULENCE_SCALE_HEIGHT_M)
    cy = float(c_y) if math.isfinite(c_y) and c_y > 0 else OSBORN_CY_DEFAULT
    d = float(telescope_diameter_m)
    t = float(exposure_s)
    var_i = (
        _YOUNG_PREFACTOR
        * (cy**2)
        * (d ** (-4.0 / 3.0))
        * (t ** -1.0)
        * (cos_z ** -3.0)
        * alt_factor
    )
    if not math.isfinite(var_i) or var_i < 0:
        return float("nan")
    return math.sqrt(var_i)


def relative_flux_err_to_mag_sigma(err_rel: float) -> float:
    """Convert relative flux error to differential magnitude sigma."""
    if not math.isfinite(err_rel) or err_rel <= 0:
        return float("nan")
    return float(MAG_ERR_SCALE * err_rel)


def resolve_rig_scintillation_params(
    *,
    draft_id: int | None,
    setup: str,
    exposure_s: float,
    cfg: AppConfig,
    pipeline_meta: dict[str, Any] | None = None,
) -> RigScintillationParams:
    """Resolve telescope diameter and altitude from DB + pipeline_meta (report sources).

    ``exposure_s``: the frame exposure from the FITS header (utils.exposure_seconds, EXPOSURE-ONE-2026-10);
    it was parsed from the setup name, which failed for the "g|120|4" form and fell back to 60 s."""
    from database import VyvarDatabase

    _cfg = cfg
    notes: list[str] = []
    alt_m: float | None = None
    diam_m: float | None = None
    try:
        exp_s = float(exposure_s)
    except (TypeError, ValueError):
        exp_s = float("nan")
    if not (math.isfinite(exp_s) and exp_s > 0):
        raise ValueError(f"scintillation: exposure {exposure_s!r} s of setup {setup!r} is not a positive number "
                         "(EXPTIME missing in the MASTERSTAR header?)")

    meta = pipeline_meta or {}
    ol = meta.get("observer_location") if isinstance(meta.get("observer_location"), dict) else {}
    if ol.get("alt_m") is not None:
        alt_candidate = float(ol["alt_m"])
        if math.isfinite(alt_candidate) and alt_candidate > 0:
            alt_m = alt_candidate
            notes.append(f"altitude_m={alt_m} from pipeline_meta.observer_location")
        elif math.isfinite(alt_candidate):
            notes.append(f"altitude_m={alt_candidate} from pipeline_meta ignored (<=0)")

    if draft_id is not None:
        try:
            db = VyvarDatabase.for_config(_cfg)
            tel_id = db.get_draft_telescope_id(int(draft_id))
            loc_id = db.get_draft_location_id(int(draft_id))
            if tel_id is not None:
                trow = db.conn.execute(
                    "SELECT DIAMETER AS diameter_mm, TELESCOPENAME AS telescope_name FROM TELESCOPE WHERE ID = ?;",
                    (int(tel_id),),
                ).fetchone()
                if trow:
                    rd = dict(trow)
                    if rd.get("diameter_mm") is not None and math.isfinite(float(rd["diameter_mm"])):
                        diam_m = float(rd["diameter_mm"]) / 1000.0
                        notes.append(
                            f"D={diam_m:.3f} m from TELESCOPE.DIAMETER ({rd.get('telescope_name')})"
                        )
            if loc_id is not None:
                lrow = db.conn.execute(
                    "SELECT ALTITUDE AS altitude_m, PLACENAME AS place_name FROM LOCATION WHERE ID = ?;",
                    (int(loc_id),),
                ).fetchone()
                if lrow:
                    rd = dict(lrow)
                    if alt_m is None and rd.get("altitude_m") is not None:
                        loc_alt = float(rd["altitude_m"])
                        if math.isfinite(loc_alt) and loc_alt > 0:
                            alt_m = loc_alt
                            notes.append(f"altitude_m={alt_m} from LOCATION ({rd.get('place_name')})")
                        elif math.isfinite(loc_alt):
                            notes.append(f"LOCATION altitude_m={loc_alt} ignored (<=0)")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"DB rig lookup failed: {exc!s}")

    # C3 / G08-16 (Milan 2026-10-29): no generic telescope or site - without the telescope diameter or the
    # site altitude the scintillation term is left out (the run continues), with a warning in the log and
    # a note in the meta; the export error line then says so. (Was: D = 0.07 m and 300 m silently.)
    if diam_m is None:
        diam_m = float("nan")
        notes.append("scintillation NOT included: no TELESCOPE.DIAMETER for this draft (enter it in Settings)")
    if alt_m is None:
        alt_m = float("nan")
        notes.append("scintillation NOT included: no site altitude for this draft (enter it in Settings)")
    if not (math.isfinite(diam_m) and math.isfinite(alt_m)):
        _key = (draft_id, str(setup))
        if _key not in _WARNED_NO_SCINT:
            _WARNED_NO_SCINT.add(_key)
            LOGGER.warning("[SCINT] %s: %s", setup, "; ".join(n for n in notes if n.startswith("scintillation NOT")))

    notes.append(f"exposure_s={exp_s:g} from the frame header (EXPTIME)")

    return RigScintillationParams(
        draft_id=draft_id,
        setup=str(setup),
        telescope_diameter_m=float(diam_m),
        altitude_m=float(alt_m),
        exposure_s=float(exp_s),
        c_y=OSBORN_CY_DEFAULT,
        source_notes=notes,
    )
