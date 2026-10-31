"""Field centres of a processed field in its manifest (G14-03; S6-FIX CENTRE-PER-SETUP, Milan 2026-10-31).

Every setup of a field is an independent instrument (other camera / telescope), so the centre is stored per
setup: ``center_by_setup = {setup: {ra_deg, de_deg, source: MASTERSTAR_WCS}}``, written by the night run only,
right after the plate solve, from the solved WCS of each setup's MASTERSTAR (centre pixel). The GUI never writes
it and there is no field-level centre. Consumers use the order solved WCS > FITS header pointing > the manifest
centre of the frame's OWN setup (``fits_meta.field_centre_radec`` per frame, the preprocess plate-solve hint per
light folder).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from fits_meta import frame_setup_name, setup_centre  # noqa: F401 - the per-setup centre rules live in fits_meta (leaf)

LOGGER = logging.getLogger(__name__)

CENTRE_SOURCE_MASTERSTAR_WCS = "MASTERSTAR_WCS"


def masterstar_files(draft_dir: Path | str) -> list[Path]:
    """``platesolve/<setup>/MASTERSTAR.fits`` of a processed field, in setup name order."""
    ps = Path(draft_dir) / "platesolve"
    if not ps.is_dir():
        return []
    return sorted(p for p in ps.glob("*/MASTERSTAR.fits") if p.is_file())


def masterstar_wcs_centre(fits_path: Path | str) -> tuple[float, float, float] | None:
    """``(ra_deg, dec_deg, field_radius_deg)`` of a solved MASTERSTAR; None without a celestial WCS.

    The centre is the sky position of the centre pixel (``fits_meta.field_centre_radec``, source
    ``WCS``); the field radius is half the frame diagonal at the WCS pixel scale."""
    import warnings  # noqa: PLC0415

    from astropy.io import fits  # noqa: PLC0415
    from astropy.wcs import WCS, FITSFixedWarning  # noqa: PLC0415
    from astropy.wcs.utils import proj_plane_pixel_scales  # noqa: PLC0415

    from fits_meta import field_centre_radec  # noqa: PLC0415

    header = fits.getheader(str(fits_path))
    ra, dec, src = field_centre_radec(header)
    if src != "WCS" or ra is None or dec is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        w = WCS(header).celestial
    scales = proj_plane_pixel_scales(w)
    nx, ny = float(header.get("NAXIS1") or 0), float(header.get("NAXIS2") or 0)
    radius = 0.5 * math.hypot(nx * float(scales[0]), ny * float(scales[1]))
    return float(ra), float(dec), float(radius)


def record_manifest_centres_from_masterstars(draft_dir: Path | str, draft_id: int) -> dict[str, dict[str, Any]]:
    """Write the solved MASTERSTAR WCS centre of every setup into the manifest ``center_by_setup`` (night run,
    after the solve). A setup without a solved MASTERSTAR gets no entry (WARNING). Returns the written map."""
    from draft_provenance import patch_draft_manifest  # noqa: PLC0415

    centres: dict[str, dict[str, Any]] = {}
    for ms in masterstar_files(draft_dir):
        c = masterstar_wcs_centre(ms)
        setup = ms.parent.name
        if c is None:
            LOGGER.warning("[NightRun] WARNING: MASTERSTAR of setup %s has no solved WCS - no field centre for it",
                           setup)
            continue
        centres[setup] = {"ra_deg": c[0], "de_deg": c[1], "source": CENTRE_SOURCE_MASTERSTAR_WCS}
        LOGGER.info("[NightRun] field centre of setup %s from its MASTERSTAR WCS: RA %.6f Dec %.6f", setup, c[0], c[1])
    if not centres:
        LOGGER.warning("[NightRun] WARNING: no solved MASTERSTAR in %s - no manifest field centre written", draft_dir)
        return {}
    patch_draft_manifest(Path(draft_dir), int(draft_id), center_by_setup=centres)
    return centres


