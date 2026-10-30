"""MASTERSTAR QA: plate-solve facts, DAO <-> Gaia match counts and the overlay map (MSQA-2026-10)."""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd

#: source_state values of a DAO detection matched to a Gaia star (pass 1 / pass 2).
MATCHED_STATES = frozenset({"DETECTED_P1", "DETECTED_P2"})
#: source_state of a DAO detection without a Gaia star.
UNMATCHED_STATE = "DAO_ONLY"


def _sexagesimal_ra(ra_deg: float) -> str:
    h = (ra_deg % 360.0) / 15.0
    hh = int(h)
    mm = int((h - hh) * 60.0)
    ss = ((h - hh) * 60.0 - mm) * 60.0
    if ss >= 59.95:
        ss, mm = 0.0, mm + 1
    if mm >= 60:
        mm, hh = 0, (hh + 1) % 24
    return f"{hh:02d}h {mm:02d}m {ss:04.1f}s"


def _sexagesimal_dec(dec_deg: float) -> str:
    sign = "-" if dec_deg < 0 else "+"
    a = abs(dec_deg)
    dd = int(a)
    mm = int((a - dd) * 60.0)
    ss = int(round(((a - dd) * 60.0 - mm) * 60.0))
    if ss >= 60:
        ss, mm = 0, mm + 1
    if mm >= 60:
        mm, dd = 0, dd + 1
    return f"{sign}{dd:02d}\u00b0 {mm:02d}\u2032 {ss:02d}\u2033"


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def load_masterstar_qa(setup_dir: Path) -> dict[str, Any]:
    """Everything the MASTERSTAR QA tab shows, read from one platesolve setup folder.

    Plate solve facts come from the MASTERSTAR.fits header and WCS; match counts and
    residuals from ``masterstars_full_match.csv`` (``source_state``, ``gaia_dao_resid_px``);
    the Gaia layer from ``field_catalog_cone.csv`` down to the matching depth (frame depth
    ``g_lim`` from pipeline_meta when present, else the faintest matched star).
    Raises ``ValueError`` with a user-facing message when the bundle is not usable.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.io import fits
    from astropy.wcs import FITSFixedWarning, WCS

    from gaia_catalog_id import read_vyvar_csv
    from platesolve_ui_paths import cone_csv_path, masterstars_csv_in_dir

    setup_dir = Path(setup_dir)
    fits_path = setup_dir / "MASTERSTAR.fits"
    if not fits_path.is_file():
        raise ValueError(f"Missing {fits_path}.")
    with fits.open(fits_path, memmap=False) as hdul:
        raw = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()
    h0, w0 = int(raw.shape[0]), int(raw.shape[1])
    with catch_warnings():
        simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(hdr)
    if not getattr(wcs, "has_celestial", False):
        raise ValueError("MASTERSTAR.fits has no WCS - plate solving did not succeed.")
    cel = wcs.celestial

    centre = wcs.pixel_to_world((w0 - 1) / 2.0, (h0 - 1) / 2.0)
    sx, sy = (abs(float(s.to(u.arcsec).value)) for s in cel.proj_plane_pixel_scales())
    scale = 0.5 * (sx + sy)
    cd = cel.pixel_scale_matrix
    rotation = float(np.degrees(np.arctan2(cd[1, 0], cd[1, 1])))
    sip = hdr.get("A_ORDER")
    solve = {
        "solver": str(hdr.get("VY_SOLVR") or "").strip() or None,
        "mode": str(hdr.get("VY_SOLMD") or "").strip() or None,
        "log_odds": _finite(hdr.get("VY_SOLLO")),
        "n_fit": int(hdr["VY_SOLNM"]) if _finite(hdr.get("VY_SOLNM")) is not None else None,
        "rms_px": _finite(hdr.get("VY_SOLRM")),
        "sip_order": int(sip) if _finite(sip) is not None else None,
    }
    xb = _finite(hdr.get("XBINNING"))
    yb = _finite(hdr.get("YBINNING"))
    frame = {
        "width_px": w0,
        "height_px": h0,
        "mpix": w0 * h0 / 1.0e6,
        "binning": f"{int(xb)}\u00d7{int(yb or xb)}" if xb is not None else None,
        "ra_deg": float(centre.ra.deg),
        "dec_deg": float(centre.dec.deg),
        "ra_str": _sexagesimal_ra(float(centre.ra.deg)),
        "dec_str": _sexagesimal_dec(float(centre.dec.deg)),
        "scale_arcsec_px": scale,
        "fov_w_deg": w0 * sx / 3600.0,
        "fov_h_deg": h0 * sy / 3600.0,
        "rotation_deg": rotation,
        "object": str(hdr.get("OBJECT") or "").strip() or None,
        "date_obs": str(hdr.get("DATE-OBS") or "").strip()[:10] or None,
    }

    ms_csv = masterstars_csv_in_dir(setup_dir)
    if ms_csv is None or not ms_csv.is_file():
        raise ValueError(f"Missing masterstars CSV in {setup_dir}.")
    ms = read_vyvar_csv(ms_csv, low_memory=False)
    if "source_state" not in ms.columns:
        raise ValueError(f"{ms_csv.name} has no source_state column - rebuild the MASTERSTAR.")
    state = ms["source_state"].fillna("").astype(str).str.strip().str.upper()
    is_m = state.isin(MATCHED_STATES).to_numpy()
    is_u = state.eq(UNMATCHED_STATE).to_numpy()
    xs = pd.to_numeric(ms.get("x"), errors="coerce").to_numpy(dtype=np.float64)
    ys = pd.to_numeric(ms.get("y"), errors="coerce").to_numpy(dtype=np.float64)
    resid = pd.to_numeric(ms.get("gaia_dao_resid_px"), errors="coerce").to_numpy(dtype=np.float64)
    r_m = resid[is_m & np.isfinite(resid)]
    n_m, n_u = int(is_m.sum()), int(is_u.sum())
    match = {
        "n_detected": n_m + n_u,
        "n_matched": n_m,
        "n_unmatched": n_u,
        "matched_pct": (100.0 * n_m / (n_m + n_u)) if (n_m + n_u) else None,
        "resid_p50_px": float(np.median(r_m)) if r_m.size else None,
        "resid_p95_px": float(np.percentile(r_m, 95)) if r_m.size else None,
    }

    depth = None
    meta_p = setup_dir / "photometry" / "pipeline_meta.json"
    if meta_p.is_file():
        try:
            fd = (json.loads(meta_p.read_text(encoding="utf-8")) or {}).get("frame_depth") or {}
            depth = _finite(fd.get("g_lim")) if isinstance(fd, dict) else None
        except (OSError, ValueError):
            depth = None
    if depth is None:
        g_m = pd.to_numeric(ms.get("phot_g_mean_mag"), errors="coerce").to_numpy(dtype=np.float64)[is_m]
        g_m = g_m[np.isfinite(g_m)]
        depth = float(g_m.max()) if g_m.size else None

    gaia_xy = np.empty((0, 2))
    cone_p = cone_csv_path(setup_dir)
    if cone_p.is_file():
        cone = read_vyvar_csv(cone_p, low_memory=False)
        ra = pd.to_numeric(cone.get("ra_deg"), errors="coerce").to_numpy(dtype=np.float64)
        dec = pd.to_numeric(cone.get("dec_deg"), errors="coerce").to_numpy(dtype=np.float64)
        gmag = pd.to_numeric(cone.get("mag"), errors="coerce").to_numpy(dtype=np.float64)
        keep = np.isfinite(ra) & np.isfinite(dec)
        if depth is not None:
            keep &= np.isfinite(gmag) & (gmag <= depth)
        if keep.any():
            gx, gy = wcs.world_to_pixel(SkyCoord(ra=ra[keep] * u.deg, dec=dec[keep] * u.deg, frame="icrs"))
            gx, gy = np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64)
            inb = np.isfinite(gx) & np.isfinite(gy) & (gx >= 0) & (gy >= 0) & (gx < w0) & (gy < h0)
            gaia_xy = np.column_stack([gx[inb], gy[inb]])

    fin = np.isfinite(xs) & np.isfinite(ys)
    return {
        "image": raw,
        "fits_name": fits_path.name,
        "solve": solve,
        "frame": frame,
        "match": match,
        "gaia_depth_g": depth,
        "matched_xy": np.column_stack([xs[is_m & fin], ys[is_m & fin]]),
        "unmatched_xy": np.column_stack([xs[is_u & fin], ys[is_u & fin]]),
        "gaia_xy": gaia_xy,
    }


def render_masterstar_overlay(
    image: np.ndarray,
    *,
    matched_xy: np.ndarray,
    unmatched_xy: np.ndarray,
    gaia_xy: np.ndarray,
    show_gaia: bool,
    show_matches: bool,
    detail: bool = False,
    max_side: int = 1600,
) -> tuple[bytes, str]:
    """JPEG of the MASTERSTAR (negative, sqrt stretch 1-99.6 %) with the selected layers.

    Matched DAO = green circle, DAO without Gaia = orange circle, Gaia = blue square.
    ``detail`` shows a 3x enlarged window of 1/8 of the frame width around the centre.
    Returns (JPEG bytes, caption).
    """
    from PIL import Image, ImageDraw

    h0, w0 = image.shape
    if detail:
        cw = max(64, w0 // 8)
        ch = max(48, int(round(cw * 0.7)))
        x0, y0 = max(0, w0 // 2 - cw // 2), max(0, h0 // 2 - ch // 2)
        x1, y1 = min(w0, x0 + cw), min(h0, y0 + ch)
        scale = 3.0
        caption = f"Detail {x1 - x0} \u00d7 {y1 - y0} px around the frame centre, 3\u00d7 enlarged."
    else:
        x0, y0, x1, y1 = 0, 0, w0, h0
        scale = min(1.0, float(max_side) / float(max(h0, w0)))
        caption = f"Whole frame, {w0} \u00d7 {h0} px."
    sub = image[y0:y1, x0:x1].astype(np.float64)
    fin = np.isfinite(sub)
    if fin.any():
        lo, hi = np.nanpercentile(sub[fin], (1.0, 99.6))
        z = np.sqrt(np.clip((sub - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
    else:
        z = np.zeros_like(sub)
    u8 = (255.0 - 255.0 * np.where(np.isfinite(z), z, 0.0)).astype(np.uint8)
    im = Image.fromarray(u8).convert("RGB")
    out_w, out_h = max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale)))
    im = im.resize((out_w, out_h), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(im)
    r = 7.0 if detail else max(2.5, 4.0 * scale)
    lw = 2 if detail else 1

    def _pts(xy: np.ndarray) -> np.ndarray:
        if xy.size == 0:
            return xy.reshape(0, 2)
        inb = (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)
        return np.column_stack([(xy[inb, 0] - x0) * scale, (xy[inb, 1] - y0) * scale])

    if show_gaia:
        q = 0.6 * r
        for px, py in _pts(gaia_xy):
            draw.rectangle([px - q, py - q, px + q, py + q], outline=(30, 100, 210), width=1)
    if show_matches:
        for px, py in _pts(matched_xy):
            draw.ellipse([px - r, py - r, px + r, py + r], outline=(16, 140, 70), width=lw)
        for px, py in _pts(unmatched_xy):
            draw.ellipse([px - r - 1, py - r - 1, px + r + 1, py + r + 1], outline=(220, 90, 0), width=2)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return buf.getvalue(), caption
