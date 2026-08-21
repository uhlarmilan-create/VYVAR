"""ERR-518-01: measure Labbe empty-aperture sigma on draft 518 Newton frame."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

from config import AppConfig
from photometry_core import measure_empty_aperture_sigma_bkg

DRAFT = REPO / "Archive" / "Drafts" / "draft_000518"
FRAME = "TOI-1131.01.b_2025-04-22_23-05-09_V.fits"
PROC = DRAFT / "detrended_aligned" / "lights" / "V_60_2" / f"proc_{FRAME.replace('.fits', '.csv')}"


def _fwhm_dao_moment(data: np.ndarray) -> float:
    from photutils.detection import DAOStarFinder

    img = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(img)
    if not finite.any():
        return float("nan")
    med = float(np.nanmedian(img[finite]))
    std = float(np.nanstd(img[finite]))
    if std <= 0:
        return float("nan")
    img2 = np.nan_to_num(img - med, nan=0.0)
    finder = DAOStarFinder(fwhm=3.0, threshold=max(3.0 * std, 1e-6))
    tbl = finder(img2)
    if tbl is None or len(tbl) == 0:
        return float("nan")
    col = "fwhm" if "fwhm" in tbl.colnames else ("fwhmfit" if "fwhmfit" in tbl.colnames else None)
    if col is None:
        return float("nan")
    return float(np.median(tbl[col]))


def analyze_fits(path: Path) -> dict:
    with fits.open(path, memmap=False) as hd:
        hdr = hd[0].header
        data = np.asarray(hd[0].data)
    finite = np.isfinite(data)
    nan_frac = 1.0 - float(finite.sum()) / float(data.size)
    inf_frac = float(np.isinf(data).sum()) / float(data.size)
    vals = data[finite]
    edge = np.zeros_like(finite, dtype=bool)
    h, w = data.shape
    m = 20
    edge[:m, :] = edge[-m:, :] = edge[:, :m] = edge[:, -m:] = True
    edge_nan_frac = 1.0 - float(np.isfinite(data[edge]).sum()) / float(edge.sum())
    return {
        "path": str(path),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "bitpix": int(hdr.get("BITPIX", 0)),
        "bzero": hdr.get("BZERO"),
        "bscale": hdr.get("BSCALE"),
        "min": float(np.nanmin(vals)) if vals.size else None,
        "max": float(np.nanmax(vals)) if vals.size else None,
        "median": float(np.nanmedian(vals)) if vals.size else None,
        "nan_frac": nan_frac,
        "inf_frac": inf_frac,
        "edge20_nan_frac": edge_nan_frac,
        "fwhm_dao_median_px": _fwhm_dao_moment(data),
        "vyvarpr": hdr.get("VYVARPR"),
        "cal_stage": hdr.get("VY_CALSTG") or hdr.get("CALSTAGE"),
        "sky_surface_applied": hdr.get("VY_SKYSF"),
    }


def main() -> None:
    cfg = AppConfig()
    proc = pd.read_csv(PROC, low_memory=False)
    xs = pd.to_numeric(proc["x"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    ys = pd.to_numeric(proc["y"], errors="coerce").dropna().to_numpy(dtype=np.float64)

    r_ap = float(proc["aperture_r_px"].iloc[0])
    fw = float(proc["fwhm_px_for_aperture"].iloc[0])
    ri = max(r_ap + 0.5, float(cfg.annulus_inner_fwhm) * fw)
    ro = max(ri + 0.5, float(cfg.annulus_outer_fwhm) * fw)

    aligned = DRAFT / "detrended_aligned" / "lights" / "V_60_2" / FRAME
    raw = DRAFT / "non_calibrated" / "lights" / "V_60_2" / FRAME

    results = {
        "config": {
            "annulus_inner_fwhm": float(cfg.annulus_inner_fwhm),
            "annulus_outer_fwhm": float(cfg.annulus_outer_fwhm),
            "err_background_mode": cfg.err_background_mode,
            "err_empty_apertures_n": cfg.err_empty_apertures_n,
            "err_empty_apertures_min": cfg.err_empty_apertures_min,
        },
        "aperture_geometry": {
            "r_ap_px": r_ap,
            "fwhm_px_for_aperture": fw,
            "r_in_px": ri,
            "r_out_px": ro,
            "n_stars_in_proc": int(len(xs)),
        },
        "fits": {
            "aligned": analyze_fits(aligned),
            "non_calibrated": analyze_fits(raw),
        },
        "labbe": {},
    }

    for label, fp in [("aligned", aligned), ("non_calibrated", raw)]:
        with fits.open(fp, memmap=False) as hd:
            data = np.asarray(hd[0].data, dtype=np.float64)
        sig, n_valid, reason = measure_empty_aperture_sigma_bkg(
            data,
            xs,
            ys,
            r_ap,
            ri,
            ro,
            n_apertures=int(cfg.err_empty_apertures_n),
            min_valid=int(cfg.err_empty_apertures_min),
            seed=42,
            frame_id=FRAME,
            star_list_source="proc_csv",
        )
        results["labbe"][label] = {
            "sigma_bkg_ap": sig if np.isfinite(sig) else None,
            "n_valid": int(n_valid),
            "reason": reason,
        }

    out_dir = REPO / "dev" / "results" / "context" / "session_20260821_err518"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "labbe_measurements.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
