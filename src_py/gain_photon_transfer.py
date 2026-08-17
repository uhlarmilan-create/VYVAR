#!/usr/bin/env python3
"""Per-draft photon-transfer gain (container ADU) and photometric gain authority.

WIDE-ERR-03: DB ``GAIN_ADU`` is native-domain (e-/ADU_native). Photometry runs on
container-domain ADU (14-bit samples left-shifted into 16-bit; stride 4). Prefer
data-derived ``g_pt`` from empty-aperture variance vs sky (Theil-Sen).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 14-bit samples in a 16-bit FITS container -> ADU stride 4 (S1a proof).
DEFAULT_CONTAINER_SCALE = 4.0
GAIN_PT_SIDECAR_NAME = "gain_photon_transfer.json"
# GAIN-PT-RADIUS-01: empty-aperture PT must be sky-dominated (~4 px on wide).
# Never read leftover pipeline_meta dynamic_params.aperture_r_px (WIDE-ERR-03B B3).
PHOTON_TRANSFER_APERTURE_R_PX = 4.0
PHOTON_TRANSFER_APERTURE_SOURCE = "pinned_sky_dominated_4px"


def resolve_photon_transfer_aperture_r_px(
    leftover_dynamic_params: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Pinned sky-dominated PT radius. Leftover meta is ignored (GAIN-PT-RADIUS-01).

    ``leftover_dynamic_params`` is accepted so callers/tests can pass previous-run
    ``dynamic_params`` without changing the result.
    """
    _ = leftover_dynamic_params
    return float(PHOTON_TRANSFER_APERTURE_R_PX), str(PHOTON_TRANSFER_APERTURE_SOURCE)


def legacy_pt_aperture_from_leftover_dynamic_params(
    leftover_dynamic_params: dict[str, Any] | None,
    default_r_px: float = PHOTON_TRANSFER_APERTURE_R_PX,
) -> float:
    """Pre-GAIN-PT-RADIUS-01 hole replica: leftover ``aperture_r_px`` overrode the pin.

    Production Phase 2A must not call this. Kept for fire-proof (a).
    """
    r = float(default_r_px)
    if not isinstance(leftover_dynamic_params, dict):
        return r
    raw = leftover_dynamic_params.get("aperture_r_px")
    if raw is None:
        return r
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return r
    if math.isfinite(v) and v > 0:
        return v
    return r


@dataclass
class PhotonTransferGain:
    """Photon-transfer estimate in e-/ADU_container."""

    g_pt: float
    g_pt_ci_lo: float
    g_pt_ci_hi: float
    n_frames: int
    aperture_r_px: float
    slope: float
    intercept: float
    method: str = "theil_sen_empty_ap_var_vs_npix_sky"
    domain: str = "e-/ADU_container"
    ok: bool = False
    notes: list[str] | None = None

    def ci_width_factor(self) -> float:
        if not (math.isfinite(self.g_pt_ci_lo) and math.isfinite(self.g_pt_ci_hi)):
            return float("inf")
        if self.g_pt_ci_lo <= 0:
            return float("inf")
        return float(self.g_pt_ci_hi / self.g_pt_ci_lo)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ci_width_factor"] = self.ci_width_factor()
        return d


@dataclass
class PhotometricGainAuthority:
    """Resolved gain for science photon/SNR terms (container ADU)."""

    value_e_per_adu_container: float
    source: str  # g_pt | db_div_container_scale | unresolved
    g_pt: float | None = None
    g_db_native: float | None = None
    container_scale: float = DEFAULT_CONTAINER_SCALE
    g_db_div_scale: float | None = None
    warn: str | None = None
    ok: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _theil_sen(y: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """Return slope, intercept, slope_lo, slope_hi (scipy theilslopes)."""
    from scipy.stats import theilslopes  # noqa: PLC0415

    res = theilslopes(y, x)
    return float(res[0]), float(res[1]), float(res[2]), float(res[3])


def estimate_photon_transfer_gain_from_proc_dir(
    proc_dir: Path,
    *,
    aperture_r_px: float,
    r_tol: float = 0.05,
    min_frames: int = 20,
    n_bootstrap: int = 800,
    seed: int = 1,
) -> PhotonTransferGain:
    """Regress sigma_bkg_ap^2 vs npix*sky (Theil-Sen); g = 1/slope."""
    notes: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    if not proc_dir.is_dir():
        return PhotonTransferGain(
            g_pt=float("nan"),
            g_pt_ci_lo=float("nan"),
            g_pt_ci_hi=float("nan"),
            n_frames=0,
            aperture_r_px=float(aperture_r_px),
            slope=float("nan"),
            intercept=float("nan"),
            ok=False,
            notes=[f"missing proc_dir {proc_dir}"],
        )

    for p in sorted(proc_dir.glob("proc_*.csv")):
        try:
            df = pd.read_csv(
                p,
                usecols=lambda c: c
                in ("sigma_bkg_ap", "sky_adu_per_px_annulus", "aperture_r_px"),
            )
        except (ValueError, OSError) as exc:
            notes.append(f"skip {p.name}: {exc}")
            continue
        need = {"sigma_bkg_ap", "sky_adu_per_px_annulus", "aperture_r_px"}
        if not need.issubset(df.columns):
            continue
        rcol = pd.to_numeric(df["aperture_r_px"], errors="coerce")
        sigcol = pd.to_numeric(df["sigma_bkg_ap"], errors="coerce")
        skycol = pd.to_numeric(df["sky_adu_per_px_annulus"], errors="coerce")
        sub = df[np.isfinite(rcol) & (np.abs(rcol - float(aperture_r_px)) <= float(r_tol))]
        if sub.empty:
            continue
        sig = float(np.nanmedian(pd.to_numeric(sub["sigma_bkg_ap"], errors="coerce")))
        sky = float(np.nanmedian(skycol))
        r_med = float(np.nanmedian(pd.to_numeric(sub["aperture_r_px"], errors="coerce")))
        if not (math.isfinite(sig) and sig > 0 and math.isfinite(sky) and sky > 0 and r_med > 0):
            continue
        area = math.pi * r_med * r_med
        xs.append(area * sky)
        ys.append(sig * sig)

    n = len(xs)
    if n < int(min_frames):
        return PhotonTransferGain(
            g_pt=float("nan"),
            g_pt_ci_lo=float("nan"),
            g_pt_ci_hi=float("nan"),
            n_frames=n,
            aperture_r_px=float(aperture_r_px),
            slope=float("nan"),
            intercept=float("nan"),
            ok=False,
            notes=notes + [f"n_frames={n} < min_frames={min_frames}"],
        )

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    slope, intercept, slope_lo, slope_hi = _theil_sen(y, x)
    g = (1.0 / slope) if math.isfinite(slope) and slope > 0 else float("nan")
    # CI from Theil slope bounds (invert); also report bootstrap for robustness.
    g_lo_ts = (1.0 / slope_hi) if math.isfinite(slope_hi) and slope_hi > 0 else float("nan")
    g_hi_ts = (1.0 / slope_lo) if math.isfinite(slope_lo) and slope_lo > 0 else float("nan")
    if math.isfinite(g_lo_ts) and math.isfinite(g_hi_ts) and g_lo_ts > g_hi_ts:
        g_lo_ts, g_hi_ts = g_hi_ts, g_lo_ts

    rng = np.random.default_rng(int(seed))
    boot: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, n)
        s = _theil_sen(y[idx], x[idx])[0]
        if math.isfinite(s) and s > 0:
            boot.append(1.0 / s)
    if len(boot) >= 20:
        p16, p84 = np.percentile(boot, [16, 84])
        # Prefer Theil analytical CI when finite; else bootstrap.
        if not (math.isfinite(g_lo_ts) and math.isfinite(g_hi_ts)):
            g_lo_ts, g_hi_ts = float(p16), float(p84)
        notes.append(f"bootstrap_p16={float(p16):.6g} bootstrap_p84={float(p84):.6g}")

    ok = bool(math.isfinite(g) and g > 0 and math.isfinite(g_lo_ts) and math.isfinite(g_hi_ts))
    return PhotonTransferGain(
        g_pt=float(g),
        g_pt_ci_lo=float(g_lo_ts),
        g_pt_ci_hi=float(g_hi_ts),
        n_frames=n,
        aperture_r_px=float(aperture_r_px),
        slope=float(slope),
        intercept=float(intercept),
        ok=ok,
        notes=notes or None,
    )


def container_scale_from_s1a(dominant_residue_frac: float = 1.0) -> float:
    """Return ADU container scale (native->container). Default 4 for 14-in-16."""
    if dominant_residue_frac >= 0.85:
        return DEFAULT_CONTAINER_SCALE
    return DEFAULT_CONTAINER_SCALE


def resolve_photometric_gain(
    *,
    g_pt_result: PhotonTransferGain | None,
    g_db_native: float | None,
    container_scale: float = DEFAULT_CONTAINER_SCALE,
    ci_max_width_factor: float = 3.0,
    native_source: str = "db",
) -> PhotometricGainAuthority:
    """Authority: g_pt when CI finite and width factor <~3, else scaled native.

    Never returns bare DB/index-mapped native gain for use on container ADU.
    Header-resolved true e-/ADU (source ``header``) is already science-domain and
    is NOT divided by container_scale (OSC / VY_EGAIN path).
    """
    scale = float(container_scale) if math.isfinite(container_scale) and container_scale > 0 else DEFAULT_CONTAINER_SCALE
    g_db = float(g_db_native) if g_db_native is not None and math.isfinite(float(g_db_native)) else float("nan")
    src = str(native_source or "db")
    # True header e-/ADU already matches pixel ADU domain (e.g. VY_EGAIN on OSC).
    if src == "header" and math.isfinite(g_db) and g_db > 0:
        g_div = g_db
        scale_used = 1.0
    else:
        g_div = (g_db / scale) if math.isfinite(g_db) and g_db > 0 else float("nan")
        scale_used = scale

    warn = None
    use_pt = False
    g_pt = float("nan")
    if g_pt_result is not None and g_pt_result.ok:
        g_pt = float(g_pt_result.g_pt)
        width = g_pt_result.ci_width_factor()
        if math.isfinite(width) and width <= float(ci_max_width_factor) and math.isfinite(g_pt) and g_pt > 0:
            use_pt = True

    if use_pt:
        if math.isfinite(g_div) and g_div > 0:
            ratio = max(g_pt / g_div, g_div / g_pt)
            if ratio > 2.0:
                warn = (
                    f"g_pt={g_pt:.4g} disagrees >2x with fallback={g_div:.4g} "
                    f"(native={g_db:.4g}, source={src}, scale={scale_used:g}); using g_pt"
                )
                logger.warning("[GAIN-AUTH] %s", warn)
        logger.info(
            "[GAIN-AUTH] authority=g_pt value=%.4g e-/ADU_container (CI [%.4g, %.4g], n=%d)",
            g_pt,
            g_pt_result.g_pt_ci_lo if g_pt_result else float("nan"),
            g_pt_result.g_pt_ci_hi if g_pt_result else float("nan"),
            g_pt_result.n_frames if g_pt_result else 0,
        )
        return PhotometricGainAuthority(
            value_e_per_adu_container=g_pt,
            source="g_pt",
            g_pt=g_pt,
            g_db_native=g_db if math.isfinite(g_db) else None,
            container_scale=scale_used,
            g_db_div_scale=g_div if math.isfinite(g_div) else None,
            warn=warn,
            ok=True,
        )

    if math.isfinite(g_div) and g_div > 0:
        auth_src = "header" if src == "header" else "db_div_container_scale"
        logger.info(
            "[GAIN-AUTH] authority=%s value=%.4g (native=%.4g src=%s scale=%g); "
            "g_pt unavailable or CI too wide",
            auth_src,
            g_div,
            g_db,
            src,
            scale_used,
        )
        return PhotometricGainAuthority(
            value_e_per_adu_container=g_div,
            source=auth_src,
            g_pt=g_pt if math.isfinite(g_pt) else None,
            g_db_native=g_db,
            container_scale=scale_used,
            g_db_div_scale=g_div,
            warn=warn,
            ok=True,
        )

    logger.warning("[GAIN-AUTH] unresolved photometric gain")
    return PhotometricGainAuthority(
        value_e_per_adu_container=float("nan"),
        source="unresolved",
        g_pt=g_pt if math.isfinite(g_pt) else None,
        g_db_native=g_db if math.isfinite(g_db) else None,
        container_scale=scale_used,
        g_db_div_scale=g_div if math.isfinite(g_div) else None,
        ok=False,
    )


def fire_proof_bare_db_vs_gpt(
    *,
    g_pt: float = 0.635,
    g_db_bare: float = 3.17,
    container_scale: float = DEFAULT_CONTAINER_SCALE,
) -> dict[str, Any]:
    """Show guard fires when bare DB is compared to g_pt (S2c)."""
    g_div = g_db_bare / container_scale
    # Compare bare DB (wrong domain) to g_pt
    ratio_bare = max(g_pt / g_db_bare, g_db_bare / g_pt) if g_pt > 0 and g_db_bare > 0 else float("nan")
    ratio_scaled = max(g_pt / g_div, g_div / g_pt) if g_pt > 0 and g_div > 0 else float("nan")
    fires = bool(math.isfinite(ratio_bare) and ratio_bare > 2.0)
    return {
        "g_pt": g_pt,
        "g_db_bare_native_misapplied": g_db_bare,
        "g_db_div_scale": g_div,
        "ratio_bare_vs_gpt": ratio_bare,
        "ratio_scaled_vs_gpt": ratio_scaled,
        "guard_fires_on_bare_db": fires,
        "threshold": 2.0,
    }


def write_gain_pt_sidecar(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_gain_pt_sidecar(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def apply_photometric_gain_authority(
    *,
    g_db_native: float,
    native_source: str,
    proc_dir: Path | None,
    aperture_r_px: float,
    container_scale: float = DEFAULT_CONTAINER_SCALE,
    ci_max_width_factor: float = 3.0,
    persist_sidecar: Path | None = None,
    draft_meta: dict[str, Any] | None = None,
    aperture_r_px_source: str | None = None,
) -> tuple[float, PhotometricGainAuthority, PhotonTransferGain | None]:
    """Estimate g_pt if possible and return (gain_container, authority, pt).

    Logs INV-NO-SILENT authority. Never returns bare DB native for science use.
    """
    pt: PhotonTransferGain | None = None
    if proc_dir is not None and Path(proc_dir).is_dir() and math.isfinite(aperture_r_px) and aperture_r_px > 0:
        pt = estimate_photon_transfer_gain_from_proc_dir(
            Path(proc_dir),
            aperture_r_px=float(aperture_r_px),
        )
    auth = resolve_photometric_gain(
        g_pt_result=pt,
        g_db_native=float(g_db_native),
        container_scale=float(container_scale),
        ci_max_width_factor=float(ci_max_width_factor),
        native_source=str(native_source or "db"),
    )
    if not auth.ok:
        # Last resort
        scale = float(container_scale) if container_scale > 0 else DEFAULT_CONTAINER_SCALE
        if str(native_source) == "header" and math.isfinite(g_db_native) and g_db_native > 0:
            fallback = float(g_db_native)
            fb_src = "header"
        else:
            fallback = float(g_db_native) / scale if math.isfinite(g_db_native) and g_db_native > 0 else float("nan")
            fb_src = "db_div_container_scale"
        logger.warning(
            "[GAIN-AUTH] authority unresolved; fallback %s=%.4g (native=%.4g src=%s)",
            fb_src,
            fallback,
            g_db_native,
            native_source,
        )
        auth = PhotometricGainAuthority(
            value_e_per_adu_container=fallback,
            source=fb_src,
            g_db_native=float(g_db_native) if math.isfinite(g_db_native) else None,
            container_scale=1.0 if fb_src == "header" else scale,
            g_db_div_scale=fallback if math.isfinite(fallback) else None,
            ok=math.isfinite(fallback) and fallback > 0,
        )
    if persist_sidecar is not None:
        _r_src = str(aperture_r_px_source or "").strip() or PHOTON_TRANSFER_APERTURE_SOURCE
        payload = {
            "photon_transfer": pt.to_dict() if pt is not None else None,
            "authority": auth.to_dict(),
            "native_gain": {"value": float(g_db_native), "source": native_source},
            "aperture_r_px": float(aperture_r_px),
            "aperture_r_px_source": _r_src,
        }
        if draft_meta:
            payload.update(draft_meta)
        write_gain_pt_sidecar(Path(persist_sidecar), payload)
    return float(auth.value_e_per_adu_container), auth, pt
