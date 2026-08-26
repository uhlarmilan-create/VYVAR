"""APERTURE-01: one radius per frame for every star (Collins 2017 / Everett & Howell 2001).

Production radii are r_ap = f x FWHM, same f for target, comps, and pool.
The Howell SNR mag-bin table remains a diagnostic artifact only.

FWHM-AUTH-01: FWHM(frame) is qc_metrics.fwhm_px (preprocess median moment-FWHM,
stamped on the frame as VY_FWHM). That quantity is already a moment FWHM.
Do not convert it by DAO_TO_GAUSSIAN (1/1.5), and do not substitute
MASTERSTAR VY_FWHM_GAUSS or the SNR-table draft-constant FWHM.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger(__name__)

MODE_FIXED_NIGHT = "f_fixed_night"
MODE_PER_FRAME = "f_per_frame"
APERTURE_POLICY_MODES = (MODE_FIXED_NIGHT, MODE_PER_FRAME)

FWHM_AUTHORITY = "qc_metrics.fwhm_px"
FWHM_AUTH_NOTE = (
    "FWHM-AUTH-01: per-frame QC moment FWHM (qc_metrics.fwhm_px / header VY_FWHM). "
    "Not VY_FWHM_GAUSS, not DAO_TO_GAUSSIAN 0.667, not the SNR-table draft-constant FWHM."
)

POLICY_ID = "APERTURE-01"


def normalize_aperture_policy_mode(raw: Any) -> str:
    s = str(raw or MODE_FIXED_NIGHT).strip().lower()
    if s in ("a", "fixed", "fixed_night", "night", MODE_FIXED_NIGHT):
        return MODE_FIXED_NIGHT
    if s in ("b", "per_frame", "per-frame", "frame", MODE_PER_FRAME):
        return MODE_PER_FRAME
    return MODE_FIXED_NIGHT


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
    mode: str,
    *,
    fwhm_frame_px: float | None,
    fwhm_night_median_px: float | None,
) -> float | None:
    """FWHM that enters r_ap = f x FWHM (annulus uses the same value)."""
    m = normalize_aperture_policy_mode(mode)
    frame = clamp_fwhm_px(fwhm_frame_px)
    night = clamp_fwhm_px(fwhm_night_median_px, fallback=frame)
    if m == MODE_FIXED_NIGHT:
        return night if night is not None else frame
    return frame if frame is not None else night


def resolve_aperture_geometry(
    *,
    f: float,
    fwhm_px: float,
    annulus_inner_fwhm: float,
    annulus_outer_fwhm: float,
) -> tuple[float, float, float]:
    """Return (r_ap, r_in, r_out) in pixels. Same FWHM scales aperture and annulus."""
    fw = float(fwhm_px)
    fac = float(f)
    if not math.isfinite(fw) or fw <= 0:
        fw = 3.5
    if not math.isfinite(fac) or fac <= 0:
        fac = 1.9
    r_ap = max(0.5, fac * fw)
    r_in = max(r_ap + 0.5, float(annulus_inner_fwhm) * fw)
    r_out = max(r_in + 0.5, float(annulus_outer_fwhm) * fw)
    return float(r_ap), float(r_in), float(r_out)


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
    mode: str,
    f: float,
    fwhm_frame_px: float | None,
    fwhm_night_median_px: float | None,
    r_ap: float,
    r_in: float,
    r_out: float,
    fwhm_used_px: float | None,
) -> dict[str, Any]:
    m = normalize_aperture_policy_mode(mode)
    return {
        "policy_id": POLICY_ID,
        "mode": m,
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


def policy_header_line(rec: Mapping[str, Any]) -> str:
    """One-line LC provenance header (comment)."""
    payload = {
        "policy_id": rec.get("policy_id", POLICY_ID),
        "mode": rec.get("mode"),
        "f": rec.get("f"),
        "fwhm_authority": rec.get("fwhm_authority", FWHM_AUTHORITY),
        "fwhm_night_median_px": rec.get("fwhm_night_median_px"),
        "r_ap_px": rec.get("r_ap_px"),
        "r_in_px": rec.get("r_in_px"),
        "r_out_px": rec.get("r_out_px"),
    }
    return "# aperture_policy: " + json.dumps(payload, separators=(",", ":"), sort_keys=True)


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


def ee_r90_continuous(ee_radii: Any, ee_curve: Any) -> float:
    """Radius where EE=0.9 by linear interpolation (no 0.5-px nearest-bin snap)."""
    import numpy as np

    rr = np.asarray(ee_radii, dtype=np.float64)
    ee = np.asarray(ee_curve, dtype=np.float64)
    if rr.size < 2 or ee.size != rr.size:
        return float("nan")
    ok = np.isfinite(rr) & np.isfinite(ee)
    if int(ok.sum()) < 2:
        return float("nan")
    rr = rr[ok]
    ee = ee[ok]
    order = np.argsort(rr)
    rr = rr[order]
    ee = ee[order]
    above = ee >= 0.9
    if not np.any(above):
        return float(rr[-1])
    i = int(np.argmax(above))
    if i == 0:
        return float(rr[0])
    e0 = float(ee[i - 1])
    e1 = float(ee[i])
    r0 = float(rr[i - 1])
    r1 = float(rr[i])
    if not math.isfinite(e0 + e1) or e1 == e0:
        return float(r1)
    return float(r0 + (0.9 - e0) * (r1 - r0) / (e1 - e0))
