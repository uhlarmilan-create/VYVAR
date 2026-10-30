"""Aperture vs ePSF light curves of one draft, side by side (EPSF-LC-COMPARE-2026-10).

Both light curves are differential (the ePSF ensemble is the aperture ensemble without the comparison
stars that have no ePSF measurement on the night): the aperture curve
(``lightcurve_<id>.csv``: ``delta_mag``, ``err``) and the internal PSF curve
(``lightcurve_<id>_psf.csv``: ``psf_delta_mag``, ``psf_delta_mag_err``) written by RUN ePSF.
Comparing raw PSF instrumental magnitudes with the calibrated aperture curve would mix the
transparency of the night into the PSF series, so only the differential curves are compared.

Per star:
  * point-to-point scatter ``p2p`` = 1.4826 * MAD(diff(m)) / sqrt(2) - insensitive to slow
    variability (the median removes a steady trend, the MAD the eclipse branches as long as they
    cover a minority of the epochs); it measures the epoch-to-epoch noise. Both methods see the same
    star, so their p2p values compare directly;
  * median quoted error and the ratio p2p / median error (1 = errors describe the scatter);
  * PSF - aperture difference: its p2p and its slope against the frame FWHM (mmag per px) - a
    seeing-dependent method bias shows up here. The median of the difference is only a level: each
    curve's ensemble uses Broeg weights from that method's own comparison-star scatter, so the
    weighted comparison level differs (draft 535 BO CVn: -205 mmag although the PSF/aperture flux
    ratio of the target and of its comparison stars agree within 2 %).
Epochs: aperture epochs with a finite ``delta_mag`` whose flag is not a hard failure
(``saturated``, ``edge_fail``, ``no_data``, ``nondetection``) - ``low_snr`` epochs are measurements and stay
in, otherwise the faint targets, where PSF matters most, would drop out - and PSF epochs with
``psf_fit_ok`` and a finite ``psf_delta_mag``; the pair statistics use the epochs both methods kept.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mag_constants import MAG_ERR_SCALE
from psf_photometry import EPSF_FRAMES_DIR, EPSF_FRAMES_MANIFEST
from report_methods import lc_csv_path

_MAD_TO_SIGMA = 1.4826
AP_HARD_FAIL_FLAGS = frozenset({"saturated", "edge_fail", "no_data", "nondetection"})


def p2p_scatter(values: Any) -> float:
    """Robust epoch-to-epoch scatter of one series (mag); NaN for fewer than 5 points."""
    m = np.asarray(values, dtype=np.float64)
    m = m[np.isfinite(m)]
    if m.size < 5:
        return float("nan")
    d = np.diff(m)
    return float(_MAD_TO_SIGMA * np.median(np.abs(d - np.median(d))) / math.sqrt(2.0))


def _read_lc(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", low_memory=False)


def frame_fwhm_map(platesolve_dir: Path | str) -> dict[str, float]:
    """``{frame stem: ePSF FWHM px}`` from the RUN ePSF manifest (empty without a manifest)."""
    man = Path(platesolve_dir) / EPSF_FRAMES_DIR / EPSF_FRAMES_MANIFEST
    if not man.is_file():
        return {}
    data = json.loads(man.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for rec in data.get("frames") or []:
        name = str(rec.get("frame") or "")
        fw = rec.get("fwhm_px")
        if name and fw is not None and math.isfinite(float(fw)):
            out[Path(name).stem] = float(fw)
    return out


def load_pair(lc_dir: Path | str, catalog_id: str, fwhm_by_frame: dict[str, float] | None = None) -> pd.DataFrame:
    """Epoch table of one star: aperture and PSF differential magnitudes on the same epochs.

    Columns: ``bjd``, ``source_file``, ``frame``, ``fwhm_px``, ``ap_mag``, ``ap_err``, ``ap_ok``,
    ``psf_mag``, ``psf_err``, ``psf_ok`` (both errors in magnitudes). Raises ``FileNotFoundError`` when a curve is missing.
    """
    ap_path = lc_csv_path(lc_dir, catalog_id, "aperture")
    psf_path = lc_csv_path(lc_dir, catalog_id, "psf")
    if not ap_path.is_file():
        raise FileNotFoundError(ap_path)
    if not psf_path.is_file():
        raise FileNotFoundError(psf_path)
    ap = _read_lc(ap_path)
    ps = _read_lc(psf_path)
    a = pd.DataFrame(
        {
            "source_file": ap["source_file"].astype(str),
            "bjd": pd.to_numeric(ap["bjd"], errors="coerce"),
            "ap_mag": pd.to_numeric(ap["delta_mag"], errors="coerce"),
            # the aperture light curve's ``err`` is relative (flux); the PSF error is in magnitudes
            "ap_err": MAG_ERR_SCALE * pd.to_numeric(ap["err"], errors="coerce"),
            "ap_ok": ~ap["flag"].astype(str).isin(AP_HARD_FAIL_FLAGS) if "flag" in ap.columns else True,
        }
    )
    p = pd.DataFrame(
        {
            "source_file": ps["source_file"].astype(str),
            "psf_mag": pd.to_numeric(ps["psf_delta_mag"], errors="coerce"),
            "psf_err": pd.to_numeric(ps["psf_delta_mag_err"], errors="coerce"),
            "psf_ok": ps["psf_fit_ok"].astype(str).str.lower().isin(["true", "1"]),
        }
    )
    # The PSF curve is written on the aperture curve's epochs (same source_file order).
    df = a.merge(p, on="source_file", how="left")
    df["psf_ok"] = df["psf_ok"].fillna(False).astype(bool) & np.isfinite(df["psf_mag"])
    df["ap_ok"] = df["ap_ok"].astype(bool) & np.isfinite(df["ap_mag"])
    df["frame"] = [Path(str(s)).stem.replace("proc_", "", 1) for s in df["source_file"]]
    fw = fwhm_by_frame or {}
    df["fwhm_px"] = [fw.get(f, float("nan")) for f in df["frame"]]
    return df


def psf_level_shift(df: pd.DataFrame) -> float:
    """G14-17: the ePSF curve is drawn at the aperture level - median of (ePSF - aperture) over the epochs
    both methods kept (``load_pair``); 0 without a common epoch. One computation for the PDF and the UI."""
    both = df.loc[df["ap_ok"] & df["psf_ok"]]
    return float(np.median(both["psf_mag"] - both["ap_mag"])) if len(both) else 0.0


def pair_stats(df: pd.DataFrame) -> dict[str, float]:
    """Statistics of one star's epoch table (``load_pair``); magnitudes in mmag. Scatter and median error of
    both methods are taken on the common epochs only: the p2p scatter of a variable star grows with the time
    step, so curves with different epoch sets (ePSF drops frames) are not comparable."""
    both = df.loc[df["ap_ok"] & df["psf_ok"]].sort_values("bjd")
    diff = (both["psf_mag"] - both["ap_mag"]).to_numpy(dtype=np.float64)
    out: dict[str, float] = {
        "n_epochs": float(len(df)),
        "n_ap": float(df["ap_ok"].sum()),
        "n_psf": float(df["psf_ok"].sum()),
        "n_both": float(len(both)),
        "ap_p2p_mmag": 1e3 * p2p_scatter(both["ap_mag"]),
        "psf_p2p_mmag": 1e3 * p2p_scatter(both["psf_mag"]),
        "ap_err_med_mmag": 1e3 * float(np.nanmedian(both["ap_err"])) if len(both) else float("nan"),
        "psf_err_med_mmag": 1e3 * float(np.nanmedian(both["psf_err"])) if len(both) else float("nan"),
        "diff_median_mmag": 1e3 * float(np.median(diff)) if diff.size else float("nan"),
        "diff_p2p_mmag": 1e3 * p2p_scatter(diff),
        "diff_fwhm_slope_mmag_px": float("nan"),
    }
    out["ap_p2p_over_err"] = out["ap_p2p_mmag"] / out["ap_err_med_mmag"] if out["ap_err_med_mmag"] > 0 else float("nan")
    out["psf_p2p_over_err"] = (
        out["psf_p2p_mmag"] / out["psf_err_med_mmag"] if out["psf_err_med_mmag"] > 0 else float("nan")
    )
    fw = both["fwhm_px"].to_numpy(dtype=np.float64)
    ok = np.isfinite(fw) & np.isfinite(diff)
    if ok.sum() >= 10 and np.ptp(fw[ok]) > 0:
        out["diff_fwhm_slope_mmag_px"] = 1e3 * float(np.polyfit(fw[ok], diff[ok], 1)[0])
    return out


def compare_draft(platesolve_dir: Path | str, catalog_ids: list[str]) -> pd.DataFrame:
    """One row of ``pair_stats`` per star that has both curves (others are skipped)."""
    ps = Path(platesolve_dir)
    lc_dir = ps / "photometry" / "lightcurves"
    fw = frame_fwhm_map(ps)
    rows = []
    for cid in catalog_ids:
        try:
            df = load_pair(lc_dir, str(cid), fw)
        except FileNotFoundError:
            continue
        rows.append({"catalog_id": str(cid), **pair_stats(df)})
    return pd.DataFrame(rows)
