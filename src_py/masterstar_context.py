"""MASTERSTAR.fits lookup and its FWHM header for Settings / UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aperture_policy import fwhm_from_header_vy_fwhm
from config import AppConfig
from platesolve_ui_paths import default_bundle_dir


def resolve_masterstar_fits_path(
    *,
    cfg: AppConfig,
    db: Any,
    draft_id: int | None,
    draft_dir_override: Path | None,
) -> Path | None:
    """Resolve ``MASTERSTAR.fits`` for the active draft (DB path, else platesolve default bundle)."""
    if draft_id is not None and db is not None:
        try:
            raw = db.get_obs_draft_masterstar_source_path(int(draft_id))
        except Exception:  # noqa: BLE001
            raw = None
        if raw:
            p = Path(str(raw).strip())
            if p.is_file() and p.suffix.lower() == ".fits":
                return p.resolve()
    draft_root: Path | None = draft_dir_override
    if draft_root is None and draft_id is not None:
        from draft_provenance import draft_dir_for_id  # noqa: PLC0415
        cand = draft_dir_for_id(cfg.archive_root, int(draft_id)).resolve()
        if cand.is_dir():
            draft_root = cand
    if draft_root is None:
        return None
    ps = draft_root / "platesolve"
    if not ps.is_dir():
        return None
    bundle = default_bundle_dir(ps)
    if bundle is None:
        return None
    m = bundle / "MASTERSTAR.fits"
    return m.resolve() if m.is_file() else None


def masterstar_header_fwhm_px(fits_path: Path) -> float | None:
    """VY_FWHM of a MASTERSTAR file (None when missing / unreadable); G04-33 replaced the unused
    MasterstarContext summary (only the FWHM had a reader)."""
    from astropy.io import fits  # noqa: PLC0415

    try:
        return fwhm_from_header_vy_fwhm(fits.getheader(str(fits_path)))
    except (OSError, ValueError):
        return None
