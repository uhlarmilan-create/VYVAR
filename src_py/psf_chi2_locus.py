# -*- coding: ascii -*-
"""EPSF-CHI2-LOCUS-01: data-derived chi2(flux) locus for psf_fit_ok.

Replaces the fixed ``psf_chi2_threshold`` SET. Shared by iterative and
grouped PSF paths (no duplicated criterion).

Night slope ``k`` and residual scatter come from a pooled MAD-clipped
OLS fit over the night: log10(chi2) = a_night + k * log10(flux).

Per frame, the intercept is local:
  a_f = robust median of (log10(chi2) - k * log10(flux))
over that frame's locus sample. Residuals use the night scatter.
If the frame has fewer than ``MIN_N_LOCUS_FIT`` locus members, the
night intercept is used (source=night). Never falls back to a fixed
chi2 cut. No night-specific numeric constants from a single rig.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Absolute floor to estimate an intercept (statistical DOF, not equipment).
MIN_N_LOCUS_FIT = 10
MAD_TO_SIGMA = 1.4826
DEFAULT_NSIGMA = 5.0
_CLIP_ROUNDS = 3
_CLIP_K = 3.0


@dataclass(frozen=True)
class Chi2Locus:
    a: float
    k: float
    scatter: float  # robust sigma of log10 residuals (1.4826*MAD)
    n: int
    source: str  # "frame" | "night"
    k_err: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_mask(
    flux: np.ndarray,
    chi2: np.ndarray,
    converged: np.ndarray | None,
    saturated: np.ndarray | None,
) -> np.ndarray:
    f = np.asarray(flux, dtype=np.float64)
    c = np.asarray(chi2, dtype=np.float64)
    m = np.isfinite(f) & (f > 0) & np.isfinite(c) & (c > 0)
    if converged is not None:
        m = m & np.asarray(converged, dtype=bool)
    if saturated is not None:
        sat = np.asarray(saturated, dtype=bool)
        m = m & ~sat
    return m


def fit_chi2_locus(
    flux: Sequence[float] | np.ndarray,
    chi2: Sequence[float] | np.ndarray,
    *,
    converged: Sequence[bool] | np.ndarray | None = None,
    saturated: Sequence[bool] | np.ndarray | None = None,
    source: str = "night",
) -> Chi2Locus | None:
    """Fit log10(chi2) = a + k*log10(flux) by iterative MAD-clipped OLS.

    Used for the night-pooled slope and scatter. Returns None if fewer
    than ``MIN_N_LOCUS_FIT`` locus members remain.
    """
    f = np.asarray(flux, dtype=np.float64)
    c = np.asarray(chi2, dtype=np.float64)
    m = _finite_mask(f, c, converged, saturated)
    if int(m.sum()) < MIN_N_LOCUS_FIT:
        return None
    x = np.log10(f[m])
    y = np.log10(c[m])
    keep = np.ones(x.shape[0], dtype=bool)
    a = k = float("nan")
    resid = np.zeros_like(x)
    for _ in range(_CLIP_ROUNDS):
        if int(keep.sum()) < MIN_N_LOCUS_FIT:
            return None
        xx = x[keep]
        yy = y[keep]
        A = np.column_stack([np.ones(len(xx)), xx])
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        a = float(coef[0])
        k = float(coef[1])
        resid = y - (a + k * x)
        med = float(np.median(resid[keep]))
        mad = float(np.median(np.abs(resid[keep] - med)))
        if mad <= 0:
            mad = float(np.std(resid[keep])) if keep.sum() > 1 else 1e-6
        if mad <= 0:
            mad = 1e-6
        keep = np.abs(resid - med) <= (_CLIP_K * mad)
    if int(keep.sum()) < MIN_N_LOCUS_FIT:
        return None
    xx = x[keep]
    yy = y[keep]
    A = np.column_stack([np.ones(len(xx)), xx])
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    a = float(coef[0])
    k = float(coef[1])
    resid_f = yy - (a + k * xx)
    med = float(np.median(resid_f))
    mad = float(np.median(np.abs(resid_f - med)))
    scatter = MAD_TO_SIGMA * mad if mad > 0 else float(np.std(resid_f))
    if not (math.isfinite(scatter) and scatter > 0):
        scatter = 1e-6
    s2 = float(np.sum((resid_f - med) ** 2) / max(1, len(xx) - 2))
    try:
        xtx_inv = np.linalg.inv(A.T @ A)
        k_err = float(math.sqrt(max(0.0, s2 * float(xtx_inv[1, 1]))))
    except np.linalg.LinAlgError:
        k_err = float("nan")
    return Chi2Locus(a=a, k=k, scatter=scatter, n=int(keep.sum()), source=source, k_err=k_err)


def frame_intercept_with_night_k(
    flux: Sequence[float] | np.ndarray,
    chi2: Sequence[float] | np.ndarray,
    *,
    night_k: float,
    night_a: float,
    night_scatter: float,
    converged: Sequence[bool] | np.ndarray | None = None,
    saturated: Sequence[bool] | np.ndarray | None = None,
) -> Chi2Locus | None:
    """Build a per-frame locus: night slope k + robust frame intercept a_f.

    Returns None only when the night slope itself is unusable. When the
    frame sample is below ``MIN_N_LOCUS_FIT``, returns the night intercept
    with ``source='night'``.
    """
    if not (math.isfinite(night_k) and math.isfinite(night_scatter) and night_scatter > 0):
        return None
    f = np.asarray(flux, dtype=np.float64)
    c = np.asarray(chi2, dtype=np.float64)
    m = _finite_mask(f, c, converged, saturated)
    n_f = int(m.sum())
    if n_f < MIN_N_LOCUS_FIT:
        return Chi2Locus(
            a=float(night_a),
            k=float(night_k),
            scatter=float(night_scatter),
            n=n_f,
            source="night",
        )
    # a_f = median(log10(chi2) - k*log10(flux)); MAD-clip once around that median.
    delta = np.log10(c[m]) - float(night_k) * np.log10(f[m])
    med = float(np.median(delta))
    mad = float(np.median(np.abs(delta - med)))
    if mad <= 0:
        mad = float(np.std(delta)) if n_f > 1 else 1e-6
    if mad <= 0:
        mad = 1e-6
    keep = np.abs(delta - med) <= (_CLIP_K * mad)
    if int(keep.sum()) < MIN_N_LOCUS_FIT:
        return Chi2Locus(
            a=float(night_a),
            k=float(night_k),
            scatter=float(night_scatter),
            n=n_f,
            source="night",
        )
    a_f = float(np.median(delta[keep]))
    return Chi2Locus(
        a=a_f,
        k=float(night_k),
        scatter=float(night_scatter),
        n=int(keep.sum()),
        source="frame",
    )


def locus_residual_sigma(
    flux: float | np.ndarray,
    chi2: float | np.ndarray,
    locus: Chi2Locus,
) -> float | np.ndarray:
    """Residual of log10(chi2) from the locus, in units of locus.scatter."""
    f = np.asarray(flux, dtype=np.float64)
    c = np.asarray(chi2, dtype=np.float64)
    if np.ndim(f) == 0:
        if not (math.isfinite(float(f)) and float(f) > 0 and math.isfinite(float(c)) and float(c) > 0):
            return float("nan")
        pred = locus.a + locus.k * math.log10(float(f))
        return float((math.log10(float(c)) - pred) / locus.scatter)
    m = np.isfinite(f) & (f > 0) & np.isfinite(c) & (c > 0)
    pred = locus.a + locus.k * np.log10(f[m])
    out = np.full(f.shape, float("nan"), dtype=np.float64)
    out[m] = (np.log10(c[m]) - pred) / locus.scatter
    return out


def chi2_ok_from_locus(
    *,
    converged: bool,
    chi2: float,
    resid_sigma: float,
    n_sigma: float = DEFAULT_NSIGMA,
) -> bool:
    """Unified SET: converged AND finite chi2 AND resid_sigma <= n_sigma.

    Nonfinite chi2 always fails (both iterative and grouped paths).
    """
    if not converged:
        return False
    if not math.isfinite(chi2):
        return False
    if not math.isfinite(resid_sigma):
        return False
    return float(resid_sigma) <= float(n_sigma)


def resolve_n_sigma(cfg: Any | None = None) -> float:
    if cfg is None:
        try:
            from config import AppConfig

            cfg = AppConfig()
        except Exception:  # noqa: BLE001
            return DEFAULT_NSIGMA
    try:
        v = float(getattr(cfg, "psf_chi2_locus_nsigma", DEFAULT_NSIGMA))
    except (TypeError, ValueError):
        return DEFAULT_NSIGMA
    if not math.isfinite(v) or v <= 0:
        return DEFAULT_NSIGMA
    return v


def apply_chi2_locus_to_rows(
    rows: list[dict[str, Any]],
    *,
    night_locus: Chi2Locus | None = None,
    n_sigma: float | None = None,
    saturated_by_cid: dict[str, bool] | None = None,
) -> tuple[list[dict[str, Any]], Chi2Locus | None, dict[str, Any]]:
    """Apply locus criterion to per-star result rows (shared SET).

    Requires a night locus for the slope. Per-frame intercept a_f is
    estimated when the frame sample is large enough; otherwise night a.
    """
    ns = float(n_sigma) if n_sigma is not None else resolve_n_sigma()
    n = len(rows)
    flux = np.full(n, float("nan"))
    chi2 = np.full(n, float("nan"))
    conv = np.zeros(n, dtype=bool)
    sat = np.zeros(n, dtype=bool)
    for i, r in enumerate(rows):
        try:
            flux[i] = float(r.get("psf_flux", float("nan")))
        except (TypeError, ValueError):
            flux[i] = float("nan")
        try:
            chi2[i] = float(r.get("psf_chi2", float("nan")))
        except (TypeError, ValueError):
            chi2[i] = float("nan")
        if "psf_converged" in r:
            conv[i] = bool(r.get("psf_converged"))
        else:
            conv[i] = bool(
                math.isfinite(flux[i])
                and flux[i] > 0
                and (math.isfinite(chi2[i]) or r.get("psf_fit_ok") is not None)
            )
            if r.get("_psf_converged") is not None:
                conv[i] = bool(r.get("_psf_converged"))
        cid = str(r.get("catalog_id", ""))
        if saturated_by_cid and cid in saturated_by_cid:
            sat[i] = bool(saturated_by_cid[cid])
        else:
            sat[i] = bool(r.get("likely_saturated") or r.get("is_saturated"))

    locus: Chi2Locus | None
    if night_locus is None:
        locus = None
        source = "none_need_night"
    else:
        locus = frame_intercept_with_night_k(
            flux,
            chi2,
            night_k=night_locus.k,
            night_a=night_locus.a,
            night_scatter=night_locus.scatter,
            converged=conv,
            saturated=sat,
        )
        source = locus.source if locus is not None else "none_need_night"
        if locus is None:
            source = "none_need_night"

    meta: dict[str, Any] = {
        "n_sigma": ns,
        "source": source,
        "min_n_locus_fit": MIN_N_LOCUS_FIT,
        "night_locus": night_locus.to_dict() if night_locus else None,
        "used_locus": locus.to_dict() if locus else None,
    }

    for i, r in enumerate(rows):
        converged_i = bool(conv[i])
        r["psf_converged"] = converged_i
        if locus is None:
            r["psf_chi2_locus_resid_sigma"] = float("nan")
            r["psf_fit_ok"] = False
            continue
        rs = locus_residual_sigma(flux[i], chi2[i], locus)
        r["psf_chi2_locus_resid_sigma"] = float(rs) if math.isfinite(float(rs)) else float("nan")
        r["psf_fit_ok"] = chi2_ok_from_locus(
            converged=converged_i,
            chi2=float(chi2[i]),
            resid_sigma=float(r["psf_chi2_locus_resid_sigma"]),
            n_sigma=ns,
        )
    if locus is not None:
        for r in rows:
            r["psf_chi2_locus_a"] = locus.a
            r["psf_chi2_locus_k"] = locus.k
            r["psf_chi2_locus_scatter"] = locus.scatter
            r["psf_chi2_locus_n"] = locus.n
            r["psf_chi2_locus_source"] = locus.source
    else:
        for r in rows:
            r["psf_chi2_locus_a"] = float("nan")
            r["psf_chi2_locus_k"] = float("nan")
            r["psf_chi2_locus_scatter"] = float("nan")
            r["psf_chi2_locus_n"] = 0
            r["psf_chi2_locus_source"] = source
    return rows, locus, meta


def fit_night_locus_from_proc_dir(proc_dir: Path) -> Chi2Locus | None:
    """Pool locus members from all proc_*.csv under ``proc_dir``."""
    files = sorted(Path(proc_dir).glob("proc_*.csv"))
    if not files:
        return None
    fluxes: list[np.ndarray] = []
    chi2s: list[np.ndarray] = []
    convs: list[np.ndarray] = []
    sats: list[np.ndarray] = []
    usecols = [
        "psf_flux",
        "psf_chi2",
        "psf_converged",
        "psf_fit_ok",
        "likely_saturated",
        "is_saturated",
    ]
    for fp in files:
        hdr = pd.read_csv(fp, nrows=0).columns.tolist()
        cols = [c for c in usecols if c in hdr]
        if "psf_flux" not in cols or "psf_chi2" not in cols:
            continue
        df = pd.read_csv(fp, usecols=cols, low_memory=False)
        f = pd.to_numeric(df["psf_flux"], errors="coerce").to_numpy(dtype=np.float64)
        c = pd.to_numeric(df["psf_chi2"], errors="coerce").to_numpy(dtype=np.float64)
        if "psf_converged" in df.columns:
            cv = df["psf_converged"].fillna(False).astype(bool).to_numpy()
        else:
            cv = np.isfinite(f) & (f > 0) & np.isfinite(c)
        sat = np.zeros(len(df), dtype=bool)
        if "likely_saturated" in df.columns:
            sat |= df["likely_saturated"].fillna(False).astype(bool).to_numpy()
        if "is_saturated" in df.columns:
            sat |= df["is_saturated"].fillna(False).astype(bool).to_numpy()
        fluxes.append(f)
        chi2s.append(c)
        convs.append(cv)
        sats.append(sat)
    if not fluxes:
        return None
    return fit_chi2_locus(
        np.concatenate(fluxes),
        np.concatenate(chi2s),
        converged=np.concatenate(convs),
        saturated=np.concatenate(sats),
        source="night",
    )


def reapply_night_locus_to_proc_dir(
    proc_dir: Path,
    *,
    n_sigma: float | None = None,
    night_locus: Chi2Locus | None = None,
) -> dict[str, Any]:
    """Rewrite fit_ok using night k + per-frame intercept a_f."""
    ns = float(n_sigma) if n_sigma is not None else resolve_n_sigma()
    night = night_locus or fit_night_locus_from_proc_dir(proc_dir)
    summary: dict[str, Any] = {
        "n_sigma": ns,
        "night_locus": night.to_dict() if night else None,
        "frames": [],
        "n_rewritten": 0,
        "n_kept_frame": 0,
        "a_f_values": [],
    }
    if night is None:
        summary["error"] = "night_locus_unavailable"
        return summary
    files = sorted(Path(proc_dir).glob("proc_*.csv"))
    for fp in files:
        df = pd.read_csv(fp, low_memory=False)
        if "psf_flux" not in df.columns or "psf_chi2" not in df.columns:
            continue
        flux = pd.to_numeric(df["psf_flux"], errors="coerce").to_numpy(dtype=np.float64)
        chi2 = pd.to_numeric(df["psf_chi2"], errors="coerce").to_numpy(dtype=np.float64)
        if "psf_converged" in df.columns:
            conv = df["psf_converged"].fillna(False).astype(bool).to_numpy()
        else:
            conv = np.isfinite(flux) & (flux > 0)
        sat = np.zeros(len(df), dtype=bool)
        if "likely_saturated" in df.columns:
            sat |= df["likely_saturated"].fillna(False).astype(bool).to_numpy()
        if "is_saturated" in df.columns:
            sat |= df["is_saturated"].fillna(False).astype(bool).to_numpy()
        locus = frame_intercept_with_night_k(
            flux,
            chi2,
            night_k=night.k,
            night_a=night.a,
            night_scatter=night.scatter,
            converged=conv,
            saturated=sat,
        )
        if locus is None:
            continue
        source = locus.source
        resid = locus_residual_sigma(flux, chi2, locus)
        if not isinstance(resid, np.ndarray):
            resid = np.asarray([resid], dtype=np.float64)
        ok = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            ok[i] = chi2_ok_from_locus(
                converged=bool(conv[i]),
                chi2=float(chi2[i]) if math.isfinite(chi2[i]) else float("nan"),
                resid_sigma=float(resid[i]) if math.isfinite(resid[i]) else float("nan"),
                n_sigma=ns,
            )
        df["psf_converged"] = conv
        df["psf_chi2_locus_resid_sigma"] = resid
        df["psf_fit_ok"] = ok
        df["psf_chi2_locus_a"] = locus.a
        df["psf_chi2_locus_k"] = locus.k
        df["psf_chi2_locus_scatter"] = locus.scatter
        df["psf_chi2_locus_n"] = locus.n
        df["psf_chi2_locus_source"] = source
        df.to_csv(fp, index=False)
        entry = {
            "frame": fp.name,
            "source": source,
            "n_locus": int(locus.n),
            "a": locus.a,
            "k": locus.k,
            "scatter": locus.scatter,
            "n_ok": int(ok.sum()),
        }
        summary["frames"].append(entry)
        if source == "frame":
            summary["n_kept_frame"] += 1
            summary["a_f_values"].append(locus.a)
        else:
            summary["n_rewritten"] += 1
    a_vals = summary["a_f_values"]
    if a_vals:
        arr = np.asarray(a_vals, dtype=np.float64)
        summary["a_f_spread"] = {
            "n": int(arr.size),
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
            "night_a": night.a,
            "night_scatter": night.scatter,
            "spread_over_night_scatter": float(
                (np.max(arr) - np.min(arr)) / night.scatter
            )
            if night.scatter > 0
            else float("nan"),
        }
    else:
        summary["a_f_spread"] = {"n": 0}
    meta_path = Path(proc_dir) / "psf_chi2_locus_night.json"
    meta_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="ascii")
    return summary


def _count_psf_fit_ok_in_sidecar(sidecar: Path) -> int:
    if not sidecar.is_file():
        return 0
    df = pd.read_csv(sidecar, usecols=["psf_fit_ok"], low_memory=False)
    col = df["psf_fit_ok"]
    if col.dtype == bool:
        return int(col.fillna(False).sum())
    s = col.astype(str).str.strip().str.lower()
    return int(s.isin(("true", "1", "yes")).sum())


def finalize_night_locus_for_inv_psf_frame_01(
    proc_dir: Path,
    frame_records: Sequence[dict[str, Any]] | None = None,
    *,
    n_sigma: float | None = None,
) -> dict[str, Any]:
    """Night-k + frame-a_f SET on all procs, then refresh ``n_ok`` for INV-PSF-FRAME-01.

    Per-frame PSF merge leaves ``psf_fit_ok`` closed until the night locus exists;
    this must run before ``finalize_epsf_frame_job``.
    """
    summary = reapply_night_locus_to_proc_dir(proc_dir, n_sigma=n_sigma)
    recs = [r for r in (frame_records or []) if isinstance(r, dict)]
    if recs:
        from proc_frame_store import proc_csv_path_for_aligned_fits

        root = Path(proc_dir)
        refreshed = 0
        for rec in recs:
            fn = str(rec.get("frame_name") or "")
            if not fn:
                continue
            sidecar = proc_csv_path_for_aligned_fits(root / fn)
            if sidecar.is_file():
                rec["n_ok"] = _count_psf_fit_ok_in_sidecar(sidecar)
                refreshed += 1
        summary["n_ok_refreshed_records"] = refreshed
    return summary


__all__ = [
    "Chi2Locus",
    "MIN_N_LOCUS_FIT",
    "DEFAULT_NSIGMA",
    "fit_chi2_locus",
    "frame_intercept_with_night_k",
    "locus_residual_sigma",
    "chi2_ok_from_locus",
    "resolve_n_sigma",
    "apply_chi2_locus_to_rows",
    "fit_night_locus_from_proc_dir",
    "reapply_night_locus_to_proc_dir",
    "finalize_night_locus_for_inv_psf_frame_01",
]
