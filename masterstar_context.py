"""Read-only summary from ``MASTERSTAR.fits`` for Settings / UI (WCS, scale, VY_* headers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from config import AppConfig
from platesolve_ui_paths import default_bundle_dir


@dataclass
class MasterstarContext:
    """Fields derived from an on-disk MASTERSTAR FITS (best effort)."""

    fits_path: Path | None = None
    exists: bool = False
    chip_width: int | None = None
    chip_height: int | None = None
    vy_fwhm_px: float | None = None
    vy_fwhm_gauss_px: float | None = None
    pixel_scale_arcsec: float | None = None
    ra_deg: float | None = None
    dec_deg: float | None = None
    wcs_ok: bool = False
    vy_psolv: bool | None = None
    vy_siprf: str | None = None
    header_snippet: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


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
        cand = (Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft_id):06d}").resolve()
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


def load_masterstar_context(fits_path: Path | None) -> MasterstarContext:
    ctx = MasterstarContext(fits_path=fits_path)
    if fits_path is None or not fits_path.is_file():
        ctx.error = "MASTERSTAR.fits sa nenašiel (skontroluj draft v Pipeline alebo DB cestu)."
        return ctx
    ctx.exists = True
    try:
        from astropy.io import fits
        from astropy.wcs import FITSFixedWarning, WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
        from warnings import catch_warnings, simplefilter

        with fits.open(str(fits_path), memmap=False) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data
            if data is not None:
                sh = np.asarray(data).shape
                if len(sh) >= 2:
                    ctx.chip_height, ctx.chip_width = int(sh[-2]), int(sh[-1])
            if ctx.chip_width is None:
                try:
                    ctx.chip_width = int(hdr.get("NAXIS1") or 0) or None
                except (TypeError, ValueError):
                    ctx.chip_width = None
            if ctx.chip_height is None:
                try:
                    ctx.chip_height = int(hdr.get("NAXIS2") or 0) or None
                except (TypeError, ValueError):
                    ctx.chip_height = None

            def _fhdr(key: str) -> float | None:
                if key not in hdr:
                    return None
                try:
                    v = float(hdr[key])
                    return v if np.isfinite(v) else None
                except (TypeError, ValueError):
                    return None

            ctx.vy_fwhm_px = _fhdr("VY_FWHM")
            ctx.vy_fwhm_gauss_px = _fhdr("VY_FWHM_GAUSS") or _fhdr("VY_FWHM_GAUSSIAN")
            if "VY_PSOLV" in hdr:
                ctx.vy_psolv = bool(hdr["VY_PSOLV"])
            for k in ("VY_SIPRF",):
                if k in hdr:
                    ctx.vy_siprf = str(hdr[k]).strip()[:120] or None

            with catch_warnings():
                simplefilter("ignore", FITSFixedWarning)
                w = WCS(hdr)
            if getattr(w, "has_celestial", False) and ctx.chip_width and ctx.chip_height:
                ctx.wcs_ok = True
                try:
                    scales_deg = proj_plane_pixel_scales(w)
                    ctx.pixel_scale_arcsec = float(np.mean(scales_deg) * 3600.0)
                except Exception:  # noqa: BLE001
                    ctx.pixel_scale_arcsec = None
                try:
                    cx = (float(ctx.chip_width) - 1.0) / 2.0
                    cy = (float(ctx.chip_height) - 1.0) / 2.0
                    c = w.pixel_to_world(cx, cy)
                    ctx.ra_deg = float(c.icrs.ra.deg)
                    ctx.dec_deg = float(c.icrs.dec.deg)
                except Exception:  # noqa: BLE001
                    ctx.ra_deg = ctx.dec_deg = None
            for hk in ("VY_MIRR", "VY_ALGN", "VY_REF", "VY_CFLAG"):
                if hk in hdr:
                    ctx.header_snippet[hk] = hdr[hk]
    except Exception as exc:  # noqa: BLE001
        ctx.error = str(exc)
        ctx.exists = False
    return ctx


def masterstar_context_markdown(ctx: MasterstarContext) -> str:
    """Short markdown block for Streamlit."""
    if ctx.error and not ctx.exists:
        return f"**MASTERSTAR:** _{ctx.error}_"
    lines = ["**MASTERSTAR (odčítané z FITS)**"]
    if ctx.fits_path:
        lines.append(f"- Súbor: `{ctx.fits_path}`")
    if ctx.chip_width and ctx.chip_height:
        lines.append(f"- Čip: **{ctx.chip_width}×{ctx.chip_height}** px")
    if ctx.vy_fwhm_px is not None:
        lines.append(f"- `VY_FWHM` (DAO odhad): **{ctx.vy_fwhm_px:.3f}** px")
    if ctx.vy_fwhm_gauss_px is not None:
        lines.append(f"- `VY_FWHM_GAUSS`: **{ctx.vy_fwhm_gauss_px:.3f}** px")
    if ctx.pixel_scale_arcsec is not None:
        lines.append(f"- Mierka (WCS priemer osí): **{ctx.pixel_scale_arcsec:.4f}** arcsec/px")
    if ctx.ra_deg is not None and ctx.dec_deg is not None:
        lines.append(f"- Stred WCS: **RA={ctx.ra_deg:.5f}°, Dec={ctx.dec_deg:.5f}°**")
    if ctx.vy_psolv is not None:
        lines.append(f"- `VY_PSOLV` (plate-solve OK): **{ctx.vy_psolv}**")
    if ctx.header_snippet:
        lines.append("- Ďalšie kľúče: " + ", ".join(f"`{k}`" for k in ctx.header_snippet))
    if ctx.error:
        lines.append(f"- _Varovanie pri čítaní: {ctx.error}_")
    return "\n".join(lines)
