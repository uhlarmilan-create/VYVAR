"""APERTURE-01: one radius per frame for every star (Collins 2017 / Everett & Howell 2001).

Production radii are r_ap = f x FWHM, same f for target, comps, and pool.
The Howell SNR mag-bin table remains a diagnostic artifact only.

FWHM-AUTH-01: FWHM(frame) is qc_metrics.fwhm_px (preprocess median moment-FWHM,
stamped on the frame as VY_FWHM). That quantity is already a moment FWHM.
Do not convert it by DAO_TO_GAUSSIAN (1/1.5), and do not substitute
MASTERSTAR VY_FWHM_GAUSS or the SNR-table draft-constant FWHM.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)

#: The only aperture policy (CFG-REVIEW-2026-10): per-frame QC FWHM, f* chosen per target in aperture photometry.
MODE_PER_TARGET = "per_target"

FWHM_AUTHORITY = "qc_metrics.fwhm_px"
FWHM_AUTH_NOTE = (
    "FWHM-AUTH-01: per-frame QC moment FWHM (qc_metrics.fwhm_px / header VY_FWHM). "
    "Not VY_FWHM_GAUSS, not DAO_TO_GAUSSIAN 0.667, not the SNR-table draft-constant FWHM."
)

POLICY_ID = "APERTURE-01"


def clamp_fwhm_px(value: Any, *, fallback: float | None = None) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if math.isfinite(v) and 0.5 < v < 30.0:
        return float(v)
    if fallback is not None:
        try:
            fb = float(fallback)
        except (TypeError, ValueError):
            return None
        if math.isfinite(fb) and 0.5 < fb < 30.0:
            return float(fb)
    return None


def fwhm_for_radius(
    *,
    fwhm_frame_px: float | None,
    fwhm_night_median_px: float | None,
) -> float | None:
    """FWHM that enters r_ap = f x FWHM (annulus uses the same value): the frame QC FWHM, night median
    when a frame has none (APERTURE-DYNAMIC-01; f* is chosen per target in aperture photometry)."""
    frame = clamp_fwhm_px(fwhm_frame_px)
    return frame if frame is not None else clamp_fwhm_px(fwhm_night_median_px)


def resolve_aperture_geometry(
    *,
    f: float,
    fwhm_px: float,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
) -> tuple[float, float, float]:
    """Return (r_ap, r_in, r_out) in pixels. Same FWHM scales aperture and annulus.

    CONSOLIDATE-01B A1: never invent fallbacks. If FWHM or the annulus factors
    cannot be resolved, raise.
    """
    try:
        fw = float(fwhm_px)
        fac = float(f)
        ain = float(annulus_inner_fwhm)
        aout = float(annulus_outer_fwhm)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "resolve_aperture_geometry: FWHM or annulus factors cannot be resolved"
        ) from exc
    if not math.isfinite(fw) or fw <= 0:
        raise ValueError(f"resolve_aperture_geometry: invalid fwhm_px={fwhm_px!r}")
    if not math.isfinite(fac) or fac <= 0:
        raise ValueError(f"resolve_aperture_geometry: invalid f={f!r}")
    if not math.isfinite(ain) or ain <= 0:
        raise ValueError(
            f"resolve_aperture_geometry: invalid annulus_inner_fwhm={annulus_inner_fwhm!r}"
        )
    if not math.isfinite(aout) or aout <= 0:
        raise ValueError(
            f"resolve_aperture_geometry: invalid annulus_outer_fwhm={annulus_outer_fwhm!r}"
        )
    r_ap = max(0.5, fac * fw)
    r_in = max(r_ap + 0.5, ain * fw)
    r_out = max(r_in + 0.5, aout * fw)
    return float(r_ap), float(r_in), float(r_out)


def stars_fit_on_chip(
    x: Any,
    y: Any,
    geometry: tuple[float, float, float],
    naxis: tuple[float, ...],
) -> Any:
    """Vectorized ``star_fits_on_chip`` (pandas Series or ndarray in, boolean mask out)."""
    import numpy as np

    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(xa) & np.isfinite(ya)
    if len(naxis) >= 4:
        x0, y0, x1, y1 = (float(naxis[0]), float(naxis[1]), float(naxis[2]), float(naxis[3]))
        inside = (xa >= x0) & (xa <= x1) & (ya >= y0) & (ya <= y1)
    else:
        r_out = float(geometry[2])
        nx, ny = float(naxis[0]), float(naxis[1])
        inside = (xa >= r_out) & (xa <= nx - r_out) & (ya >= r_out) & (ya <= ny - r_out)
    return finite & inside


def fwhm_from_header_vy_fwhm(hdr: Any) -> float | None:
    """QC moment FWHM stamped as VY_FWHM. No Gaussian conversion (FWHM-AUTH-01)."""
    if hdr is None:
        return None
    try:
        raw = hdr.get("VY_FWHM")
    except Exception:  # noqa: BLE001
        return None
    if isinstance(raw, tuple):
        raw = raw[0]
    return clamp_fwhm_px(raw)


def lookup_qc_fwhm_px(
    qc_fwhm_by_name: Mapping[str, float] | None,
    frame_name: str | None,
) -> float | None:
    if not qc_fwhm_by_name or not frame_name:
        return None
    p = Path(str(frame_name))
    for key in (p.name, p.stem, str(frame_name)):
        if key in qc_fwhm_by_name:
            return clamp_fwhm_px(qc_fwhm_by_name[key])
    stem_cf = p.stem.casefold()
    name_cf = p.name.casefold()
    for k, v in qc_fwhm_by_name.items():
        kk = str(k)
        if kk.casefold() in (stem_cf, name_cf) or Path(kk).stem.casefold() == stem_cf:
            return clamp_fwhm_px(v)
    return None


def resolve_frame_fwhm_px(
    *,
    hdr: Any = None,
    frame_name: str | None = None,
    qc_fwhm_by_name: Mapping[str, float] | None = None,
    fwhm_night_median_px: float | None = None,
) -> float | None:
    """Authority order: qc_metrics map, then header VY_FWHM, then night median."""
    v = lookup_qc_fwhm_px(qc_fwhm_by_name, frame_name)
    if v is not None:
        return v
    v = fwhm_from_header_vy_fwhm(hdr)
    if v is not None:
        return v
    return clamp_fwhm_px(fwhm_night_median_px)


def policy_record(
    *,
    f: float,
    fwhm_frame_px: float | None,
    fwhm_night_median_px: float | None,
    r_ap: float,
    r_in: float,
    r_out: float,
    fwhm_used_px: float | None,
) -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "mode": MODE_PER_TARGET,
        "f": float(f),
        "fwhm_authority": FWHM_AUTHORITY,
        "fwhm_auth_note": FWHM_AUTH_NOTE,
        "fwhm_frame_px": (
            float(fwhm_frame_px) if fwhm_frame_px is not None and math.isfinite(float(fwhm_frame_px)) else None
        ),
        "fwhm_night_median_px": (
            float(fwhm_night_median_px)
            if fwhm_night_median_px is not None and math.isfinite(float(fwhm_night_median_px))
            else None
        ),
        "fwhm_used_px": (
            float(fwhm_used_px) if fwhm_used_px is not None and math.isfinite(float(fwhm_used_px)) else None
        ),
        "r_ap_px": float(r_ap),
        "r_in_px": float(r_in),
        "r_out_px": float(r_out),
    }


def load_qc_fwhm_map(qc_csv: Path | str | None) -> tuple[dict[str, float], float | None]:
    """Map frame basename/stem -> fwhm_px; night median over status==ok rows with finite FWHM."""
    out: dict[str, float] = {}
    if qc_csv is None or not str(qc_csv).strip():
        return out, None
    p = Path(qc_csv)
    if not p.is_file():
        return out, None
    try:
        import pandas as pd

        df = pd.read_csv(p, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[APERTURE-01] qc_metrics read failed: %s", exc)
        return out, None
    if df is None or df.empty or "fwhm_px" not in df.columns:
        return out, None
    fwhm = pd.to_numeric(df["fwhm_px"], errors="coerce")
    status = (
        df["status"].astype(str).str.strip().str.lower()
        if "status" in df.columns
        else pd.Series(["ok"] * len(df))
    )
    ok_vals: list[float] = []
    src_col = "src" if "src" in df.columns else ("dst" if "dst" in df.columns else None)
    for i in range(len(df)):
        fv = clamp_fwhm_px(fwhm.iloc[i])
        if fv is None:
            continue
        if str(status.iloc[i]).strip().lower() == "ok":
            ok_vals.append(fv)
        if src_col is None:
            continue
        raw = str(df[src_col].iloc[i] or "")
        if not raw.strip():
            continue
        pp = Path(raw)
        out[pp.name] = fv
        out[pp.stem] = fv
    night = None
    if ok_vals:
        import numpy as np

        night = float(np.median(np.asarray(ok_vals, dtype=np.float64)))
        night = clamp_fwhm_px(night)
    return out, night


