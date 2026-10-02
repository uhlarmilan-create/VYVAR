"""LC-OUTLIER-01 / LC-FLAG-ERR-01: flag outlier LC epochs; never delete; protect flares.

Product rules (Milan 2026-10-01 / 2026-10-02):
- NEVER alter photometry values; only set ``flag`` / ``flag_reason``.
- Isolated spike alone is not enough for ``artifact``; need image evidence.
- Runs of >=2 same-sign elevated residuals (flares/eclipses) stay ``normal``.
- ``spike_unconfirmed`` is never excluded from exports.
- ``high_err``: err_i > median(err) + n_sigma x 1.4826 x MAD(err) on the
  star's own LC (LC-FLAG-ERR-01). Hides points whose huge error bar would
  otherwise silence the spike test.

Flag vocabulary::
  normal | saturated | frame_qc | artifact | high_err | spike_unconfirmed | no_data
  (+ preserved nondetection / edge_fail when already set upstream)

Precedence (LC-FLAG-ERR-01)::
  artifact > frame_qc > high_err > spike_unconfirmed > normal
  (preserve saturated/no_data/nondetection/edge_fail untouched)

Thresholds are statistical conventions (MAD-sigma), registered as config keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import json
import logging
import math
import re

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

_MAD_CONSISTENCY = 0.6745
# Explicit 1.4826 used in LC-FLAG-ERR-01 product formula (= 1/0.6745).
_MAD_TO_SIGMA = 1.4826

FLAG_NORMAL = "normal"
FLAG_SATURATED = "saturated"
FLAG_FRAME_QC = "frame_qc"
FLAG_ARTIFACT = "artifact"
FLAG_HIGH_ERR = "high_err"
FLAG_SPIKE_UNCONFIRMED = "spike_unconfirmed"
FLAG_NO_DATA = "no_data"

# Upstream-preserved flags (not overwritten by this module).
_PRESERVE_FLAGS = frozenset(
    {"saturated", "nondetection", "edge_fail", "no_data"}
)

# Export / UI: hide when toggle off; drop from AAVSO/VarAstro/minima.
EXPORT_EXCLUDE_FLAGS = frozenset(
    {
        FLAG_FRAME_QC,
        FLAG_ARTIFACT,
        FLAG_HIGH_ERR,
        "saturated",
        "no_data",
        "edge_fail",
        "nondetection",
    }
)
# LC-FLAG-ERR-01: all non-normal classes drawn red; toggle hides them all.
UI_HIDE_WHEN_TOGGLE_OFF = frozenset(
    {
        FLAG_FRAME_QC,
        FLAG_ARTIFACT,
        FLAG_HIGH_ERR,
        FLAG_SPIKE_UNCONFIRMED,
        "saturated",
        "outlier_hi",
        "outlier_lo",
        "no_data",
    }
)
# spike_unconfirmed stays in exports; display is red like other non-normal.
ALWAYS_SHOW_FLAGS = frozenset()

DEFAULT_N_SIGMA = 5.0
DEFAULT_ADJACENT_SIGMA = 3.0
DEFAULT_FRAME_QC_N_SIGMA = 5.0
DEFAULT_EVIDENCE_N_SIGMA = 5.0
DEFAULT_HIGH_ERR_N_SIGMA = 5.0
# Cadence window: half-width covers ~30 min of samples, clamped to [3, 12].
_WINDOW_HALF_DAYS = 0.020833333333333332  # 30 minutes
_WINDOW_K_MIN = 3
_WINDOW_K_MAX = 12

_ERR_COMPONENT_NAMES = (
    "err_photon",
    "err_sem_rel",
    "err_scint_rel",
    "err_sigma_sys_rel",
)


def mad_sigma(values: np.ndarray) -> float:
    """Robust sigma = 1.4826 * MAD; floor for empty/degenerate samples."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 1e-12
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return max(mad / _MAD_CONSISTENCY, 1e-12)


def high_err_mask(
    err: np.ndarray,
    *,
    n_sigma: float = DEFAULT_HIGH_ERR_N_SIGMA,
) -> tuple[np.ndarray, float, float]:
    """LC-FLAG-ERR-01: err_i > median(err) + n_sigma x 1.4826 x MAD(err).

    Returns (is_high_err, median_err, mad_err) where mad_err is the raw MAD
    (not yet scaled by 1.4826).
    """
    e = np.asarray(err, dtype=float)
    n = len(e)
    out = np.zeros(n, dtype=bool)
    finite = e[np.isfinite(e) & (e > 0)]
    if finite.size < 3:
        return out, float("nan"), float("nan")
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    thr = med + float(n_sigma) * _MAD_TO_SIGMA * mad
    for i in range(n):
        ei = float(e[i])
        if math.isfinite(ei) and ei > thr:
            out[i] = True
    return out, med, mad


def dominant_err_component(
    i: int,
    *,
    err_photon: np.ndarray | None = None,
    err_sem_rel: np.ndarray | None = None,
    err_scint_rel: np.ndarray | None = None,
    err_sigma_sys_rel: np.ndarray | None = None,
) -> str:
    """Name the largest finite err component at index i for flag_reason."""
    comps: list[tuple[str, float]] = []
    for name, arr in (
        ("err_photon", err_photon),
        ("err_sem_rel", err_sem_rel),
        ("err_scint_rel", err_scint_rel),
        ("err_sigma_sys_rel", err_sigma_sys_rel),
    ):
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        if i < 0 or i >= len(a):
            continue
        v = float(a[i])
        if math.isfinite(v):
            comps.append((name, v))
    if not comps:
        return "err:unknown"
    name, val = max(comps, key=lambda t: t[1])
    return f"{name}={val:.6g}"


def cadence_half_window(bjd: np.ndarray) -> int:
    """Half-window k (neighbours each side) from night cadence.

    Rule: k = clamp(round(30_min / median_dt), 3, 12) where dt is the median
    positive finite BJD difference. Short nights fall back to k=3.
    """
    t = np.asarray(bjd, dtype=float)
    finite = np.isfinite(t)
    if int(finite.sum()) < 3:
        return _WINDOW_K_MIN
    tt = np.sort(t[finite])
    dt = np.diff(tt)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size < 1:
        return _WINDOW_K_MIN
    med_dt = float(np.median(dt))
    if not math.isfinite(med_dt) or med_dt <= 0:
        return _WINDOW_K_MIN
    k = int(round(_WINDOW_HALF_DAYS / med_dt))
    return max(_WINDOW_K_MIN, min(_WINDOW_K_MAX, k))


def running_median(values: np.ndarray, half_window: int) -> np.ndarray:
    """Robust running median; NaN inputs stay NaN in the output."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    out = np.full(n, float("nan"), dtype=float)
    k = max(1, int(half_window))
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        window = x[lo:hi]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[i] = float(np.median(finite))
    return out


def isolated_spike_mask(
    mag: np.ndarray,
    err: np.ndarray,
    bjd: np.ndarray,
    *,
    n_sigma: float = DEFAULT_N_SIGMA,
    adjacent_sigma: float = DEFAULT_ADJACENT_SIGMA,
    half_window: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (is_isolated_spike, residual_sigma, smooth).

    Residual r_i = (m_i - S_i) / err_i with S = running median.
    Candidate if |r| > n_sigma. Not isolated if an adjacent epoch has the
    same sign and |r_adj| > adjacent_sigma (flare/eclipse protection).
    """
    m = np.asarray(mag, dtype=float)
    e = np.asarray(err, dtype=float)
    n = len(m)
    spike = np.zeros(n, dtype=bool)
    resid = np.full(n, float("nan"), dtype=float)
    k = int(half_window) if half_window is not None else cadence_half_window(bjd)
    smooth = running_median(m, k)
    for i in range(n):
        if not (math.isfinite(m[i]) and math.isfinite(smooth[i])):
            continue
        ei = float(e[i]) if math.isfinite(float(e[i])) and float(e[i]) > 0 else float("nan")
        if not math.isfinite(ei) or ei <= 0:
            # Fall back to night MAD of (m - S) so sparse-err LCs still work.
            continue
        resid[i] = (m[i] - smooth[i]) / ei

    # Fill resid for points with missing err using MAD of finite residuals proxy.
    finite_diff = (m - smooth)
    finite_diff = finite_diff[np.isfinite(finite_diff)]
    fallback_sig = mad_sigma(finite_diff) if finite_diff.size >= 3 else 1e-3
    for i in range(n):
        if math.isfinite(resid[i]):
            continue
        if not (math.isfinite(m[i]) and math.isfinite(smooth[i])):
            continue
        resid[i] = (m[i] - smooth[i]) / fallback_sig

    thr = float(n_sigma)
    adj = float(adjacent_sigma)
    for i in range(n):
        r = resid[i]
        if not math.isfinite(r) or abs(r) <= thr:
            continue
        sign = 1.0 if r > 0 else -1.0
        # Same-sign neighbour protection (runs >= 2 never flagged here).
        protected = False
        for j in (i - 1, i + 1):
            if j < 0 or j >= n:
                continue
            rj = resid[j]
            if not math.isfinite(rj):
                continue
            if (rj * sign) > adj:
                protected = True
                break
        if not protected:
            spike[i] = True
    return spike, resid, smooth


@dataclass
class ImageEvidence:
    """Per-star per-frame stamp evidence vs same-frame similar-flux peers."""

    fired: bool = False
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    zscores: dict[str, float] = field(default_factory=dict)


def _stamp_metrics(img: np.ndarray, x0: float, y0: float, *, half: int = 11) -> dict[str, float]:
    """Centroid offset, FWHM, elongation, peak/flux, annulus sky sigma at (x0,y0)."""
    h, w = img.shape
    xi = int(round(float(x0)))
    yi = int(round(float(y0)))
    y1 = max(0, yi - half)
    y2 = min(h, yi + half + 1)
    x1 = max(0, xi - half)
    x2 = min(w, xi + half + 1)
    cut = np.asarray(img[y1:y2, x1:x2], dtype=np.float64)
    if cut.size < 25:
        return {
            "centroid_offset_px": float("nan"),
            "fwhm_px": float("nan"),
            "elongation": float("nan"),
            "peak_to_flux": float("nan"),
            "annulus_sky_sigma": float("nan"),
        }
    border = np.concatenate([cut[0, :], cut[-1, :], cut[:, 0], cut[:, -1]])
    sky = float(np.median(border))
    cut0 = np.clip(cut - sky, 0.0, None)
    flux = float(np.sum(cut0))
    peak = float(np.max(cut0)) if cut0.size else 0.0
    conc = float(peak / flux) if flux > 0 else float("nan")
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float64)
    if flux <= 0:
        return {
            "centroid_offset_px": float("nan"),
            "fwhm_px": float("nan"),
            "elongation": float("nan"),
            "peak_to_flux": conc,
            "annulus_sky_sigma": float("nan"),
        }
    cx = float(np.sum(xx * cut0) / flux)
    cy = float(np.sum(yy * cut0) / flux)
    off = float(math.hypot(cx - float(x0), cy - float(y0)))
    dx = xx - cx
    dy = yy - cy
    mxx = float(np.sum((dx * dx) * cut0) / flux)
    myy = float(np.sum((dy * dy) * cut0) / flux)
    mxy = float(np.sum((dx * dy) * cut0) / flux)
    tr = mxx + myy
    det = mxx * myy - mxy * mxy
    disc = max(tr * tr - 4.0 * det, 0.0)
    l1 = 0.5 * (tr + math.sqrt(disc))
    l2 = 0.5 * (tr - math.sqrt(disc))
    sig1 = math.sqrt(max(l1, 0.0))
    sig2 = math.sqrt(max(l2, 0.0))
    fwhm = 2.355 * 0.5 * (sig1 + sig2)
    elong = (sig1 / sig2) if sig2 > 0 else float("nan")
    # Annulus sky sigma on full image (trail crossing annulus).
    yy2, xx2 = np.ogrid[0:h, 0:w]
    rr = np.hypot(xx2 - float(x0), yy2 - float(y0))
    ann = img[(rr >= 8.0) & (rr <= 15.0)]
    asigma = float(np.std(ann)) if ann.size > 10 else float("nan")
    return {
        "centroid_offset_px": off,
        "fwhm_px": float(fwhm) if math.isfinite(fwhm) else float("nan"),
        "elongation": float(elong) if math.isfinite(elong) else float("nan"),
        "peak_to_flux": conc,
        "annulus_sky_sigma": asigma,
    }


def _robust_z(value: float, peers: Sequence[float]) -> float:
    v = float(value)
    if not math.isfinite(v):
        return float("nan")
    arr = np.asarray(list(peers), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return float("nan")
    med = float(np.median(arr))
    return (v - med) / mad_sigma(arr)


def evaluate_image_evidence(
    img: np.ndarray,
    x: float,
    y: float,
    *,
    peer_xy: Sequence[tuple[float, float]],
    n_sigma: float = DEFAULT_EVIDENCE_N_SIGMA,
    psf_chi2_locus_resid_sigma: float | None = None,
    star_annulus_sky_sigma_night: Sequence[float] | None = None,
    star_annulus_sky_sigma_now: float | None = None,
) -> ImageEvidence:
    """Compare target stamp metrics to same-frame similar-flux peers.

    Evidence = any metric |z| > n_sigma (or psf_chi2_locus_resid_sigma, or
    annulus sky sigma vs the star's own night median).
    """
    ev = ImageEvidence()
    target = _stamp_metrics(img, x, y)
    ev.metrics.update(target)
    peer_metrics: dict[str, list[float]] = {k: [] for k in target}
    for px, py in peer_xy:
        pm = _stamp_metrics(img, float(px), float(py))
        for k, val in pm.items():
            if math.isfinite(float(val)):
                peer_metrics[k].append(float(val))

    thr = float(n_sigma)
    for key, val in target.items():
        z = _robust_z(float(val), peer_metrics.get(key, []))
        ev.zscores[key] = z
        if math.isfinite(z) and abs(z) > thr:
            ev.fired = True
            ev.reasons.append(f"{key}_z={z:.2f}")

    if psf_chi2_locus_resid_sigma is not None and math.isfinite(float(psf_chi2_locus_resid_sigma)):
        zchi = float(psf_chi2_locus_resid_sigma)
        ev.metrics["psf_chi2_locus_resid_sigma"] = zchi
        ev.zscores["psf_chi2_locus_resid_sigma"] = zchi
        if abs(zchi) > thr:
            ev.fired = True
            ev.reasons.append(f"psf_chi2_locus_resid_sigma={zchi:.2f}")

    if (
        star_annulus_sky_sigma_now is not None
        and star_annulus_sky_sigma_night is not None
        and math.isfinite(float(star_annulus_sky_sigma_now))
    ):
        zsky = _robust_z(float(star_annulus_sky_sigma_now), star_annulus_sky_sigma_night)
        ev.metrics["annulus_sky_sigma_vs_night"] = float(star_annulus_sky_sigma_now)
        ev.zscores["annulus_sky_sigma_vs_night"] = zsky
        if math.isfinite(zsky) and abs(zsky) > thr:
            ev.fired = True
            ev.reasons.append(f"annulus_sky_sigma_vs_night_z={zsky:.2f}")

    return ev


def frame_qc_mask_from_night_table(
    frame_metrics: pd.DataFrame,
    *,
    n_sigma: float = DEFAULT_FRAME_QC_N_SIGMA,
    metric_cols: Sequence[str] = ("fwhm", "elongation", "sky_level", "ensemble_zp"),
) -> dict[str, str]:
    """Robust MAD-sigma frame QC vs the night distribution.

    ``frame_metrics`` must have a ``frame_key`` column (basename of source_file
    or FITS stem) plus optional metric columns. Returns frame_key -> reason
    (empty string = OK).
    """
    out: dict[str, str] = {}
    if frame_metrics is None or frame_metrics.empty or "frame_key" not in frame_metrics.columns:
        return out
    thr = float(n_sigma)
    work = frame_metrics.copy()
    work["frame_key"] = work["frame_key"].astype(str)
    zcols: dict[str, np.ndarray] = {}
    for col in metric_cols:
        if col not in work.columns:
            continue
        vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if int(finite.sum()) < 5:
            continue
        sig = mad_sigma(vals[finite])
        med = float(np.median(vals[finite]))
        z = np.full_like(vals, float("nan"))
        z[finite] = (vals[finite] - med) / sig
        zcols[col] = z
    for i, key in enumerate(work["frame_key"].tolist()):
        reasons: list[str] = []
        for col, zarr in zcols.items():
            z = float(zarr[i])
            if math.isfinite(z) and abs(z) > thr:
                reasons.append(f"{col}_z={z:.2f}")
        out[str(key)] = ";".join(reasons)
    return out


def load_frame_metrics_from_manifest(manifest_path: Path) -> pd.DataFrame:
    """Build night frame-metric table from draft_manifest.json inspection block."""
    path = Path(manifest_path)
    if not path.is_file():
        return pd.DataFrame()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("[LC-OUTLIER] manifest read failed (%s): %s", path, exc)
        return pd.DataFrame()
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            continue
        imagetyp = str(entry.get("imagetyp") or "").strip().lower()
        if imagetyp and imagetyp not in ("light", "science", ""):
            continue
        fp = str(entry.get("file_path") or "")
        stem = Path(fp).stem if fp else ""
        if not stem:
            continue
        insp = entry.get("inspection") if isinstance(entry.get("inspection"), dict) else {}
        qc = entry.get("qc") if isinstance(entry.get("qc"), dict) else {}
        rows.append(
            {
                "frame_key": stem,
                "fwhm": insp.get("fwhm"),
                "elongation": insp.get("elongation_mean"),
                "sky_level": insp.get("sky_level", qc.get("background")),
                "ensemble_zp": float("nan"),  # filled later if available
                "stars": qc.get("stars", insp.get("star_count")),
            }
        )
    return pd.DataFrame(rows)


_PROC_RE = re.compile(r"^proc_(.+)\.csv$", re.IGNORECASE)


def fits_path_for_proc(proc_path: Path) -> Path | None:
    """Map ``proc_STEM.csv`` -> ``STEM.fits`` beside it (aligned lights preferred)."""
    m = _PROC_RE.match(proc_path.name)
    if not m:
        return None
    cand = proc_path.with_name(f"{m.group(1)}.fits")
    return cand if cand.is_file() else None


def frame_key_from_source(source_file: str) -> str:
    """Normalize LC ``source_file`` / FITS name to a comparable frame key."""
    name = Path(str(source_file)).name
    m = _PROC_RE.match(name)
    if m:
        return m.group(1)
    if name.lower().endswith(".fits"):
        return Path(name).stem
    if name.lower().endswith(".csv"):
        return Path(name).stem
    return name


def similar_flux_peers(
    proc_df: pd.DataFrame,
    catalog_id: str,
    *,
    flux_lo: float = 0.5,
    flux_hi: float = 2.0,
    max_peers: int = 40,
) -> list[tuple[float, float]]:
    """Return (x,y) of same-frame stars with flux in [flux_lo, flux_hi] x target."""
    if proc_df is None or proc_df.empty:
        return []
    cid = str(catalog_id)
    if "catalog_id" not in proc_df.columns or "x" not in proc_df.columns or "y" not in proc_df.columns:
        return []
    flux_col = "flux" if "flux" in proc_df.columns else ("dao_flux" if "dao_flux" in proc_df.columns else None)
    if flux_col is None:
        return []
    hit = proc_df[proc_df["catalog_id"].astype(str) == cid]
    if hit.empty:
        return []
    tf = float(pd.to_numeric(hit.iloc[0][flux_col], errors="coerce"))
    if not math.isfinite(tf) or tf <= 0:
        return []
    flux = pd.to_numeric(proc_df[flux_col], errors="coerce")
    mask = (
        (proc_df["catalog_id"].astype(str) != cid)
        & flux.notna()
        & (flux > flux_lo * tf)
        & (flux < flux_hi * tf)
    )
    sub = proc_df.loc[mask, ["x", "y"]].copy()
    if sub.empty:
        return []
    if len(sub) > max_peers:
        sub = sub.sample(n=max_peers, random_state=0)
    out: list[tuple[float, float]] = []
    for _, row in sub.iterrows():
        try:
            out.append((float(row["x"]), float(row["y"])))
        except (TypeError, ValueError):
            continue
    return out


@dataclass
class FlagResult:
    flags: list[str]
    reasons: list[str]
    n_artifact: int = 0
    n_spike_unconfirmed: int = 0
    n_frame_qc: int = 0
    n_saturated: int = 0
    n_high_err: int = 0


def assign_lc_flags(
    mag: np.ndarray,
    err: np.ndarray,
    bjd: np.ndarray,
    *,
    base_flags: Sequence[str] | None = None,
    sat_flags: np.ndarray | None = None,
    source_files: Sequence[str] | None = None,
    frame_qc_reasons: Mapping[str, str] | None = None,
    evidence_for_index: Callable[[int], ImageEvidence | None] | None = None,
    n_sigma: float = DEFAULT_N_SIGMA,
    adjacent_sigma: float = DEFAULT_ADJACENT_SIGMA,
    high_err_n_sigma: float = DEFAULT_HIGH_ERR_N_SIGMA,
    err_photon: np.ndarray | None = None,
    err_sem_rel: np.ndarray | None = None,
    err_scint_rel: np.ndarray | None = None,
    err_sigma_sys_rel: np.ndarray | None = None,
    enabled: bool = True,
) -> FlagResult:
    """Assign flag / flag_reason for one light curve (photometry unchanged).

    Precedence: artifact > frame_qc > high_err > spike_unconfirmed > normal
    (preserve saturated / no_data / nondetection / edge_fail).
    """
    m = np.asarray(mag, dtype=float)
    e = np.asarray(err, dtype=float)
    t = np.asarray(bjd, dtype=float)
    n = len(m)
    flags = [FLAG_NORMAL] * n
    reasons = [""] * n

    if base_flags is not None and len(base_flags) == n:
        for i, bf in enumerate(base_flags):
            s = str(bf or "").strip().lower()
            if s in _PRESERVE_FLAGS:
                flags[i] = s
                reasons[i] = s
    if sat_flags is not None and len(sat_flags) == n:
        for i in range(n):
            if bool(sat_flags[i]):
                flags[i] = FLAG_SATURATED
                reasons[i] = "saturated"

    for i in range(n):
        if flags[i] in _PRESERVE_FLAGS:
            continue
        if not math.isfinite(m[i]):
            flags[i] = FLAG_NO_DATA
            reasons[i] = "no_data"

    if frame_qc_reasons and source_files is not None and len(source_files) == n:
        for i, sf in enumerate(source_files):
            if flags[i] in _PRESERVE_FLAGS:
                continue
            key = frame_key_from_source(str(sf))
            why = str(frame_qc_reasons.get(key) or "").strip()
            if why:
                flags[i] = FLAG_FRAME_QC
                reasons[i] = f"frame_qc:{why}"

    # LC-FLAG-ERR-01: inflated photometric error (before spike so high_err
    # beats spike_unconfirmed; artifact may still overwrite below).
    he_mask = np.zeros(n, dtype=bool)
    if enabled:
        he_mask, _med_e, _mad_e = high_err_mask(e, n_sigma=float(high_err_n_sigma))
        for i in range(n):
            if not bool(he_mask[i]):
                continue
            if flags[i] in _PRESERVE_FLAGS or flags[i] == FLAG_FRAME_QC:
                continue
            dom = dominant_err_component(
                i,
                err_photon=err_photon,
                err_sem_rel=err_sem_rel,
                err_scint_rel=err_scint_rel,
                err_sigma_sys_rel=err_sigma_sys_rel,
            )
            flags[i] = FLAG_HIGH_ERR
            reasons[i] = f"high_err:{dom}"

    if enabled:
        spike, _resid, _smooth = isolated_spike_mask(
            m, e, t, n_sigma=n_sigma, adjacent_sigma=adjacent_sigma
        )
        for i in range(n):
            if not bool(spike[i]):
                continue
            if flags[i] in _PRESERVE_FLAGS or flags[i] == FLAG_FRAME_QC:
                continue
            ev: ImageEvidence | None = None
            if evidence_for_index is not None:
                try:
                    ev = evidence_for_index(i)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.debug("[LC-OUTLIER] evidence failed at i=%s: %s", i, exc)
                    ev = None
            if ev is not None and ev.fired:
                # artifact beats high_err
                flags[i] = FLAG_ARTIFACT
                reasons[i] = "artifact:" + ",".join(ev.reasons[:6])
            elif flags[i] == FLAG_HIGH_ERR:
                # high_err beats spike_unconfirmed: keep high_err
                continue
            else:
                flags[i] = FLAG_SPIKE_UNCONFIRMED
                reasons[i] = "spike_unconfirmed:isolated_residual"

    n_art = sum(1 for f in flags if f == FLAG_ARTIFACT)
    n_su = sum(1 for f in flags if f == FLAG_SPIKE_UNCONFIRMED)
    n_fq = sum(1 for f in flags if f == FLAG_FRAME_QC)
    n_sat = sum(1 for f in flags if f == FLAG_SATURATED)
    n_he = sum(1 for f in flags if f == FLAG_HIGH_ERR)
    return FlagResult(
        flags=flags,
        reasons=reasons,
        n_artifact=n_art,
        n_spike_unconfirmed=n_su,
        n_frame_qc=n_fq,
        n_saturated=n_sat,
        n_high_err=n_he,
    )


class EvidenceCache:
    """Lazy FITS / peer cache for per-target spike evidence during Phase 2A."""

    def __init__(
        self,
        *,
        frames_dir: Path | None,
        catalog_id: str,
        n_sigma: float = DEFAULT_EVIDENCE_N_SIGMA,
        proc_cache: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.frames_dir = Path(frames_dir) if frames_dir is not None else None
        self.catalog_id = str(catalog_id)
        self.n_sigma = float(n_sigma)
        self.proc_cache = proc_cache if proc_cache is not None else {}
        self._img_cache: dict[str, np.ndarray] = {}
        self._night_sky_sig: list[float] | None = None

    def _load_proc(self, source_file: str) -> pd.DataFrame | None:
        name = Path(str(source_file)).name
        if name in self.proc_cache:
            return self.proc_cache[name]
        if self.frames_dir is None:
            return None
        path = self.frames_dir / name
        if not path.is_file():
            # Sometimes source_file is already a basename under platesolve cache.
            return None
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:  # noqa: BLE001
            return None
        self.proc_cache[name] = df
        return df

    def _load_img(self, source_file: str) -> np.ndarray | None:
        key = frame_key_from_source(source_file)
        if key in self._img_cache:
            return self._img_cache[key]
        if self.frames_dir is None:
            return None
        proc_name = Path(str(source_file)).name
        proc_path = self.frames_dir / proc_name
        fits_path = fits_path_for_proc(proc_path) if proc_path.is_file() else None
        if fits_path is None:
            cand = self.frames_dir / f"{key}.fits"
            fits_path = cand if cand.is_file() else None
        if fits_path is None:
            return None
        try:
            from astropy.io import fits  # noqa: PLC0415

            with fits.open(fits_path, memmap=True) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("[LC-OUTLIER] FITS open failed %s: %s", fits_path, exc)
            return None
        self._img_cache[key] = data
        return data

    def night_annulus_sky_sigma(self, source_files: Sequence[str]) -> list[float]:
        """Collect target annulus sky sigma across the night (from proc)."""
        if self._night_sky_sig is not None:
            return self._night_sky_sig
        vals: list[float] = []
        for sf in source_files:
            df = self._load_proc(str(sf))
            if df is None or "sigma_bkg_ap" not in df.columns:
                continue
            hit = df[df["catalog_id"].astype(str) == self.catalog_id]
            if hit.empty:
                continue
            v = float(pd.to_numeric(hit.iloc[0]["sigma_bkg_ap"], errors="coerce"))
            if math.isfinite(v):
                vals.append(v)
        self._night_sky_sig = vals
        return vals

    def evidence_at(self, index: int, source_files: Sequence[str]) -> ImageEvidence | None:
        if index < 0 or index >= len(source_files):
            return None
        sf = str(source_files[index])
        img = self._load_img(sf)
        proc = self._load_proc(sf)
        if img is None or proc is None:
            return None
        hit = proc[proc["catalog_id"].astype(str) == self.catalog_id]
        if hit.empty:
            return None
        row = hit.iloc[0]
        try:
            x = float(row["x"])
            y = float(row["y"])
        except (TypeError, ValueError, KeyError):
            return None
        peers = similar_flux_peers(proc, self.catalog_id)
        chi2 = None
        if "psf_chi2_locus_resid_sigma" in proc.columns:
            try:
                chi2 = float(pd.to_numeric(row["psf_chi2_locus_resid_sigma"], errors="coerce"))
            except (TypeError, ValueError):
                chi2 = None
        sky_now = None
        if "sigma_bkg_ap" in proc.columns:
            try:
                sky_now = float(pd.to_numeric(row["sigma_bkg_ap"], errors="coerce"))
            except (TypeError, ValueError):
                sky_now = None
        return evaluate_image_evidence(
            img,
            x,
            y,
            peer_xy=peers,
            n_sigma=self.n_sigma,
            psf_chi2_locus_resid_sigma=chi2,
            star_annulus_sky_sigma_night=self.night_annulus_sky_sigma(source_files),
            star_annulus_sky_sigma_now=sky_now,
        )


def export_keep_mask(flags: Sequence[str]) -> np.ndarray:
    """True where epoch should be kept in AAVSO / VarAstro / minima exports."""
    keep = []
    for f in flags:
        s = str(f or "").strip().lower()
        if s in ALWAYS_SHOW_FLAGS or s in ("", FLAG_NORMAL, "outlier_hi", "outlier_lo"):
            # Legacy outlier_* kept for old CSVs until re-run; new scheme excludes
            # frame_qc / artifact / high_err (+ hard bad); keeps spike_unconfirmed.
            keep.append(True)
            continue
        keep.append(s not in EXPORT_EXCLUDE_FLAGS)
    return np.asarray(keep, dtype=bool)


__all__ = [
    "ALWAYS_SHOW_FLAGS",
    "DEFAULT_ADJACENT_SIGMA",
    "DEFAULT_EVIDENCE_N_SIGMA",
    "DEFAULT_FRAME_QC_N_SIGMA",
    "DEFAULT_HIGH_ERR_N_SIGMA",
    "DEFAULT_N_SIGMA",
    "EvidenceCache",
    "EXPORT_EXCLUDE_FLAGS",
    "FLAG_ARTIFACT",
    "FLAG_FRAME_QC",
    "FLAG_HIGH_ERR",
    "FLAG_NO_DATA",
    "FLAG_NORMAL",
    "FLAG_SATURATED",
    "FLAG_SPIKE_UNCONFIRMED",
    "FlagResult",
    "ImageEvidence",
    "UI_HIDE_WHEN_TOGGLE_OFF",
    "assign_lc_flags",
    "cadence_half_window",
    "dominant_err_component",
    "evaluate_image_evidence",
    "export_keep_mask",
    "fits_path_for_proc",
    "frame_key_from_source",
    "frame_qc_mask_from_night_table",
    "high_err_mask",
    "isolated_spike_mask",
    "load_frame_metrics_from_manifest",
    "mad_sigma",
    "running_median",
    "similar_flux_peers",
]
