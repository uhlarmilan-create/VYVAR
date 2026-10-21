# -*- coding: ascii -*-
"""ePSF residual (PSF-minus-aperture) ZP-OK meters.

Single implementation of the EPSF-PIN-CENSUS-01 residual statistic
(psf_delta - ap_delta, then median / RMS / demeaned RMS). Used by
the census script and by session_baseline_check G3. Do not demean
raw psf_delta_mag: that statistic tracks target variability (BO CVn).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def residual_stats(
    psf_delta: np.ndarray,
    ap_delta: np.ndarray,
    pin_ok: np.ndarray,
) -> dict[str, Any]:
    """Census residual meters: residual = psf_delta - ap_delta on pin-ok epochs.

    pin_ok is full-membership (no pin drop). n_full_membership is pin_ok.sum();
    pin drops must not hide as a smaller RMS (G3 n_full is a hard gate).
    """
    psf = np.asarray(psf_delta, dtype=float)
    ap = np.asarray(ap_delta, dtype=float)
    pin = np.asarray(pin_ok, dtype=bool)
    n = int(pin.size)
    n_full = int(pin.sum())
    both = pin & np.isfinite(psf) & np.isfinite(ap)
    res = psf[both] - ap[both]
    n_both = int(both.sum())
    med = float(np.median(res)) if n_both else float("nan")
    rms = float(np.sqrt(np.mean(res**2))) if n_both else float("nan")
    dem = float(np.sqrt(np.mean((res - med) ** 2))) if n_both else float("nan")
    rms_vs = bool(
        n_both > 0 and abs(rms - abs(med)) <= 0.25 * max(abs(med), abs(rms), 1e-12)
    )
    return {
        "n_epochs": n,
        "n_full_membership": n_full,
        "n_dropped_pin": n - n_full,
        "coverage": (n_full / n) if n else float("nan"),
        "n_finite_pairs": n_both,
        "level_offset_mag": med,
        "level_offset_mmag": med * 1000.0 if math.isfinite(med) else float("nan"),
        "rms_mag": rms,
        "rms_mmag": rms * 1000.0 if math.isfinite(rms) else float("nan"),
        "demeaned_rms_mmag": dem * 1000.0 if math.isfinite(dem) else float("nan"),
        "rms_vs_abs_median": rms_vs,
    }


def _psf_lc_n_epochs_full(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines()[:40]:
        s = line.strip()
        if not s.startswith("#"):
            break
        if "psf_lc_n_epochs_full=" in s:
            raw = s.split("psf_lc_n_epochs_full=", 1)[1].strip().split()[0]
            try:
                return int(raw)
            except ValueError:
                return None
    return None


def residual_meters_from_lightcurves(
    work_root: Path,
    tid: str,
    *,
    setup: str = "NoFilter_60_2",
) -> dict[str, Any]:
    """Residual meters from the existing LC product (PSF + aperture CSVs).

    Aligns on source_file. pin_ok is finite psf_delta_mag (LC writer already
    NaN-ed pin drops). n_full prefers the PSF LC header (B4).
    """
    lc_dir = Path(work_root) / "platesolve" / setup / "photometry" / "lightcurves"
    psf_path = lc_dir / f"lightcurve_{tid}_psf.csv"
    ap_path = lc_dir / f"lightcurve_{tid}.csv"
    if not psf_path.is_file():
        return {"catalog_id": tid, "missing": True, "path": str(psf_path)}
    if not ap_path.is_file():
        return {"catalog_id": tid, "missing": True, "path": str(ap_path)}
    psf = pd.read_csv(psf_path, comment="#", low_memory=False)
    ap = pd.read_csv(ap_path, comment="#", low_memory=False)
    if "psf_delta_mag" not in psf.columns:
        return {"catalog_id": tid, "missing_col": True, "path": str(psf_path)}
    if "delta_mag" not in ap.columns:
        return {"catalog_id": tid, "missing_col": True, "path": str(ap_path)}
    if "source_file" in psf.columns and "source_file" in ap.columns:
        merged = pd.merge(
            ap[["source_file", "delta_mag"]].assign(
                source_file=lambda d: d["source_file"].astype(str).str.strip()
            ),
            psf[["source_file", "psf_delta_mag"]].assign(
                source_file=lambda d: d["source_file"].astype(str).str.strip()
            ),
            on="source_file",
            how="inner",
        )
        ap_d = pd.to_numeric(merged["delta_mag"], errors="coerce").to_numpy(dtype=float)
        psf_d = pd.to_numeric(merged["psf_delta_mag"], errors="coerce").to_numpy(dtype=float)
    else:
        ap_d = pd.to_numeric(ap["delta_mag"], errors="coerce").to_numpy(dtype=float)
        psf_d = pd.to_numeric(psf["psf_delta_mag"], errors="coerce").to_numpy(dtype=float)
        n = min(int(ap_d.size), int(psf_d.size))
        ap_d = ap_d[:n]
        psf_d = psf_d[:n]
    pin_ok = np.isfinite(psf_d)
    out = residual_stats(psf_d, ap_d, pin_ok)
    n_hdr = _psf_lc_n_epochs_full(psf_path)
    out["catalog_id"] = str(tid)
    out["missing"] = False
    out["n_full"] = int(n_hdr) if n_hdr is not None else int(out["n_full_membership"])
    return out
