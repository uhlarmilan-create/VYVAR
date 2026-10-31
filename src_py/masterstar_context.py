"""MASTERSTAR.fits lookup and its FWHM header for Settings / UI."""

from __future__ import annotations

from pathlib import Path

from aperture_policy import fwhm_from_header_vy_fwhm


def resolve_masterstar_fits_path(draft_dir: Path, setup: str) -> Path | None:
    """``platesolve/<setup>/MASTERSTAR.fits`` of a processed field, None when it is missing.

    G04-34: the setup is chosen by the caller (the Settings view's setup selector). The manifest
    ``paths.masterstar`` is the MASTERSTAR *source light frame*, not the MASTERSTAR, and is not read."""
    m = Path(draft_dir) / "platesolve" / str(setup) / "MASTERSTAR.fits"
    return m.resolve() if m.is_file() else None


def masterstar_header_fwhm_px(fits_path: Path) -> float | None:
    """VY_FWHM of a MASTERSTAR file (None when missing / unreadable); G04-33 replaced the unused
    MasterstarContext summary (only the FWHM had a reader)."""
    from astropy.io import fits  # noqa: PLC0415

    try:
        return fwhm_from_header_vy_fwhm(fits.getheader(str(fits_path)))
    except (OSError, ValueError):
        return None
