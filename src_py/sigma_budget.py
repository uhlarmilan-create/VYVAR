"""Diagnostic sigma budget: Howell (production) + Young/Osborn scintillation.

Production LC export wires scintillation via ``sigma_floor_core.scintillation_mag_per_epoch``
(batch D, P-02). This module also provides diagnostic harness variants.
See dev/results/specs/VYVAR_SIGMA_BUDGET_SPEC.md.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


# Osborn et al. (2015) MNRAS 452, 1707 -- modified Young, eq. (7)
_TURBULENCE_SCALE_HEIGHT_M = 8000.0
_YOUNG_PREFACTOR = 10.0e-6
OSBORN_CY_DEFAULT = 1.5

from mag_constants import MAG_ERR_SCALE


SIGMA_VARIANT_HOWELL_ONLY = "howell_only"
SIGMA_VARIANT_HOWELL_SCINT_FULL = "howell_scint_full"
SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE = "howell_scint_fresid_floor_ensemble"
# Acceptance-authoritative harness variant: LC ``err`` column (production uncertainty).
SIGMA_VARIANT_PRODUCTION_LC_ERR = "production_lc_err"


@dataclass
class RigScintillationParams:
    draft_id: int | None
    setup: str
    telescope_diameter_m: float
    altitude_m: float
    exposure_s: float
    c_y: float
    source_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "setup": self.setup,
            "telescope_diameter_m": self.telescope_diameter_m,
            "altitude_m": self.altitude_m,
            "exposure_s": self.exposure_s,
            "c_y": self.c_y,
            "source_notes": list(self.source_notes),
        }


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


def parse_setup_exposure_s(setup_name: str) -> float | None:
    """Parse exposure seconds from setup folder name (e.g. ``NoFilter_60_2`` -> 60)."""
    m = re.match(r"^.+_(\d+)_\d+$", str(setup_name or "").strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def resolve_rig_scintillation_params(
    *,
    draft_id: int | None,
    setup: str,
    cfg: Any | None = None,
    pipeline_meta: dict[str, Any] | None = None,
) -> RigScintillationParams:
    """Resolve telescope diameter and altitude from DB + pipeline_meta (report sources)."""
    from config import AppConfig
    from database import VyvarDatabase

    _cfg = cfg or AppConfig()
    notes: list[str] = []
    alt_m: float | None = None
    diam_m: float | None = None
    exp_s = parse_setup_exposure_s(setup)

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
            db = VyvarDatabase(_cfg.database_path)
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

    # Documented rig fallbacks when DB row missing (reported, not silent)
    # COMP-POOL-02 item 1: Carl-Zeiss "200mm" is FOCAL; aperture DIAMETER=70 mm (f/2.86).
    if diam_m is None:
        if draft_id in (424,):
            diam_m = 0.07
            notes.append("D=0.07 m fallback: wide Carl-Zeiss Jirny aperture (draft_424; 200mm=focal)")
        elif draft_id in (425, 426, 427):
            diam_m = 0.3
            notes.append("D=0.3 m fallback: Newton Brno/Dablice/Zdanice family")
        else:
            diam_m = 0.07
            notes.append("D=0.07 m generic fallback (unknown draft; was wrongly 0.2=focal)")

    if alt_m is None:
        if draft_id in (424,):
            alt_m = 250.0
            notes.append("altitude_m=250 fallback: Jirny ~250 m")
        elif draft_id in (425, 426, 427):
            alt_m = 275.0
            notes.append("altitude_m=275 fallback: Dablice/Zdanice ~250-300 m mean")
        else:
            alt_m = 300.0
            notes.append("altitude_m=300 generic fallback")

    if exp_s is None or not math.isfinite(exp_s):
        exp_s = 60.0
        notes.append(f"exposure_s={exp_s} fallback (setup name parse failed for {setup!r})")
    else:
        notes.append(f"exposure_s={exp_s} from setup name {setup!r}")

    return RigScintillationParams(
        draft_id=draft_id,
        setup=str(setup),
        telescope_diameter_m=float(diam_m),
        altitude_m=float(alt_m),
        exposure_s=float(exp_s),
        c_y=OSBORN_CY_DEFAULT,
        source_notes=notes,
    )
