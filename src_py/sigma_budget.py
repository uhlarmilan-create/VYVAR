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

from photometry_core import _photometric_error

# Osborn et al. (2015) MNRAS 452, 1707 -- modified Young, eq. (7)
_TURBULENCE_SCALE_HEIGHT_M = 8000.0
_YOUNG_PREFACTOR = 10.0e-6
OSBORN_CY_DEFAULT = 1.5

from mag_constants import MAG_ERR_SCALE

# Relative intensity fluctuation -> magnitude (small-signal limit)
_REL_FLUX_TO_MAG = MAG_ERR_SCALE

SIGMA_VARIANT_HOWELL_ONLY = "howell_only"
SIGMA_VARIANT_HOWELL_SCINT_FULL = "howell_scint_full"
SIGMA_VARIANT_HOWELL_SCINT_FRESID = "howell_scint_fresid"
SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR = "howell_scint_fresid_floor"
SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR_ENSEMBLE = "howell_scint_fresid_floor_ensemble"
# Acceptance-authoritative harness variant: LC ``err`` column (production uncertainty).
SIGMA_VARIANT_PRODUCTION_LC_ERR = "production_lc_err"


def combine_sigma_mag_quadrature(
    sig_howell_mag: float,
    sig_scint_mag: float,
    *,
    sigma_floor_mag: float = 0.0,
    ensemble_sem_mag: float = 0.0,
) -> float:
    """Combine magnitude-domain sigmas in quadrature (Howell + scint + floor + ensemble SEM)."""
    terms: list[float] = []
    if math.isfinite(sig_howell_mag) and sig_howell_mag > 0:
        terms.append(sig_howell_mag * sig_howell_mag)
    if math.isfinite(sig_scint_mag) and sig_scint_mag > 0:
        terms.append(sig_scint_mag * sig_scint_mag)
    if math.isfinite(ensemble_sem_mag) and ensemble_sem_mag > 0:
        terms.append(ensemble_sem_mag * ensemble_sem_mag)
    if math.isfinite(sigma_floor_mag) and sigma_floor_mag > 0:
        terms.append(sigma_floor_mag * sigma_floor_mag)
    if not terms:
        return float("nan")
    return math.sqrt(sum(terms))


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


def howell_sigma(
    flux: float,
    sky_pp: float,
    area: float,
    *,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> float:
    """Relative flux sigma from production ``_photometric_error`` (Howell 1989 eq. 2)."""
    return float(
        _photometric_error(
            float(flux),
            float(sky_pp),
            float(area),
            gain=float(gain),
            read_noise=float(read_noise),
        )
    )


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


def total_sigma(
    flux: float,
    sky_pp: float,
    area: float,
    *,
    gain: float = 1.0,
    read_noise: float = 10.0,
    telescope_diameter_m: float,
    airmass: float,
    exposure_s: float,
    altitude_m: float,
    c_y: float = OSBORN_CY_DEFAULT,
    f_resid: float = 0.0,
    variant: str = SIGMA_VARIANT_HOWELL_ONLY,
) -> tuple[float, float, float]:
    """Return (sigma_total_rel, sigma_howell_rel, sigma_scint_rel_used).

    Variants:
      - howell_only
      - howell_scint_full (f_resid=1)
      - howell_scint_fresid (f_resid free parameter)
    """
    sig_h = howell_sigma(flux, sky_pp, area, gain=gain, read_noise=read_noise)
    sig_s_full = scintillation_sigma(
        telescope_diameter_m=telescope_diameter_m,
        airmass=airmass,
        exposure_s=exposure_s,
        altitude_m=altitude_m,
        c_y=c_y,
    )
    if variant == SIGMA_VARIANT_HOWELL_ONLY:
        return sig_h, sig_h, 0.0
    f = 1.0 if variant == SIGMA_VARIANT_HOWELL_SCINT_FULL else float(f_resid)
    if not math.isfinite(f) or f < 0:
        f = 0.0
    f = min(f, 1.0)
    sig_s = sig_s_full * f if math.isfinite(sig_s_full) else float("nan")
    if not math.isfinite(sig_h):
        return float("nan"), sig_h, sig_s
    if not math.isfinite(sig_s):
        return sig_h, sig_h, sig_s
    return math.sqrt(sig_h * sig_h + sig_s * sig_s), sig_h, sig_s


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
            row = db.conn.execute(
                """
                SELECT t.DIAMETER AS diameter_mm, t.TELESCOPENAME AS telescope_name,
                       l.ALTITUDE AS altitude_m, l.PLACENAME AS place_name
                FROM OBS_DRAFT d
                LEFT JOIN TELESCOPE t ON d.ID_TELESCOPE = t.ID
                LEFT JOIN LOCATION l ON d.ID_LOCATION = l.ID
                WHERE d.ID = ?;
                """,
                (int(draft_id),),
            ).fetchone()
            if row:
                rd = dict(row)
                if rd.get("diameter_mm") is not None and math.isfinite(float(rd["diameter_mm"])):
                    diam_m = float(rd["diameter_mm"]) / 1000.0
                    notes.append(
                        f"D={diam_m:.3f} m from TELESCOPE.DIAMETER ({rd.get('telescope_name')})"
                    )
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
    if diam_m is None:
        if draft_id in (424,):
            diam_m = 0.2
            notes.append("D=0.2 m fallback: wide Carl-Zeiss Jirny (draft_424)")
        elif draft_id in (425, 426, 427):
            diam_m = 0.3
            notes.append("D=0.3 m fallback: Newton Brno/Dablice/Zdanice family")
        else:
            diam_m = 0.2
            notes.append("D=0.2 m generic fallback (unknown draft)")

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
