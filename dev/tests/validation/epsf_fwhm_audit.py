"""ePSF-vs-star Moffat FWHM audit (VYVAR_EPSF_FWHM_TEST method). Validation/diagnostic only."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian2D, Moffat2D
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig
from crowding_index import _load_wcs_meta
from database import VyvarDatabase
from masterstar_context import header_core_fwhm_px
from psf_photometry import (
    _epsf_build_imagepsf_from_stars,
    _epsf_prepare_stars,
    _median_fwhm_obs_files,
)

OSAMP = 2
FWHM_FROM_STD = 2.3548200450277027
MOFFAT_ALPHA = 2.5


def _moffat_fwhm_oversampled(gamma: float, alpha: float) -> float:
    return 2.0 * float(gamma) * math.sqrt(2.0 ** (1.0 / float(alpha)) - 1.0)


def _fit_moffat2d_array(arr: np.ndarray, *, fwhm_guess: float) -> dict:
    z = np.asarray(arr, dtype=np.float64)
    h, w = z.shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h // 2, w // 2
    peak = float(np.nanmax(z))
    if not math.isfinite(peak) or peak <= 0:
        return {"ok": False}
    alpha = MOFFAT_ALPHA
    gamma0 = max(0.5, fwhm_guess * OSAMP / (2.0 * math.sqrt(2.0 ** (1.0 / alpha) - 1.0)))
    model = Moffat2D(amplitude=peak, x_0=float(cx), y_0=float(cy), gamma=gamma0, alpha=alpha)
    model.gamma.bounds = (0.05, None)
    fitter = LevMarLSQFitter()
    try:
        fit = fitter(model, xx, yy, z)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)}
    g = float(fit.gamma.value)
    a = float(fit.alpha.value)
    fwhm_os = _moffat_fwhm_oversampled(g, a)
    return {
        "ok": True,
        "fwhm_oversampled": fwhm_os,
        "fwhm_native": fwhm_os / OSAMP,
        "gamma": g,
        "alpha": a,
        "peak": float(fit.amplitude.value),
        "x_0": float(fit.x_0.value),
        "y_0": float(fit.y_0.value),
    }


def _fit_gauss2d_array(arr: np.ndarray) -> dict:
    z = np.asarray(arr, dtype=np.float64)
    h, w = z.shape
    yy, xx = np.mgrid[:h, :w]
    cy, cx = np.unravel_index(int(np.nanargmax(z)), z.shape)
    peak = float(np.nanmax(z))
    if not math.isfinite(peak) or peak <= 0:
        return {"ok": False}
    sig0 = max(h, w) / 8.0
    model = Gaussian2D(
        amplitude=peak,
        x_mean=float(cx),
        y_mean=float(cy),
        x_stddev=sig0,
        y_stddev=sig0,
        theta=0.0,
    )
    model.x_stddev.bounds = (0.05, None)
    model.y_stddev.bounds = (0.05, None)
    fitter = LevMarLSQFitter()
    try:
        fit = fitter(model, xx, yy, z)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": str(exc)}
    sx = float(fit.x_stddev.value)
    sy = float(fit.y_stddev.value)
    if not (math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0):
        return {"ok": False}
    sig_mean = 0.5 * (sx + sy)
    fwhm_os = FWHM_FROM_STD * sig_mean
    return {"ok": True, "fwhm_oversampled": fwhm_os, "fwhm_native": fwhm_os / OSAMP}


def _azimuthal_fwhm(arr: np.ndarray, *, peak: float, cy: float, cx: float) -> dict:
    z = np.asarray(arr, dtype=np.float64)
    h, w = z.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
    v = z.ravel()
    ok = np.isfinite(v) & np.isfinite(r)
    r = r[ok]
    v = v[ok]
    if r.size < 10 or peak <= 0:
        return {"ok": False}
    rmax = min(h, w) * 0.45
    bins = np.arange(0.0, rmax + 0.125, 0.25)
    medians = []
    centers = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        sel = (r >= lo) & (r < hi)
        if int(sel.sum()) < 3:
            continue
        centers.append(0.5 * (lo + hi))
        medians.append(float(np.median(v[sel])))
    if len(centers) < 4:
        return {"ok": False}
    centers = np.asarray(centers)
    medians = np.asarray(medians)
    medians = medians / peak
    cross = None
    for i in range(len(centers) - 1):
        a, b = medians[i], medians[i + 1]
        if a >= 0.5 >= b and a != b:
            frac = (a - 0.5) / (a - b)
            cross = float(centers[i] + frac * (centers[i + 1] - centers[i]))
            break
    if cross is None:
        return {"ok": False}
    fwhm_os = 2.0 * cross
    return {"ok": True, "fwhm_oversampled": fwhm_os, "fwhm_native": fwhm_os / OSAMP}


def _obs_fwhm_filter_median(db: VyvarDatabase, draft_id: int, filt: str) -> float:
    cur = db.conn.execute(
        """
        SELECT FWHM FROM OBS_FILES
        WHERE DRAFT_ID = ? AND FWHM IS NOT NULL AND FILTER = ?;
        """,
        (int(draft_id), str(filt)),
    )
    vals = [float(r[0]) for r in cur.fetchall() if r[0] is not None]
    return float(np.median(vals)) if vals else float("nan")


def _select_isolated_stars(ms_csv: Path, fwhm_px: float, n_target: int = 30) -> pd.DataFrame:
    df = pd.read_csv(ms_csv, low_memory=False, dtype={"catalog_id": str})
    req = df.copy()
    for col, rule in (
        ("photometry_ok", True),
        ("likely_saturated", False),
        ("is_saturated", False),
        ("is_usable", True),
        ("edge_safe_10px", True),
    ):
        if col in req.columns:
            if isinstance(rule, bool):
                req = req[req[col].astype(str).str.lower().isin(["true", "1", "yes", "1.0"]) == rule]
            else:
                req = req[req[col] == rule]
    req["x"] = pd.to_numeric(req["x"], errors="coerce")
    req["y"] = pd.to_numeric(req["y"], errors="coerce")
    req["flux"] = pd.to_numeric(req.get("flux"), errors="coerce")
    req = req.dropna(subset=["x", "y", "flux"])
    if req.empty:
        return req
    pts = req[["x", "y"]].to_numpy(dtype=float)
    tree = cKDTree(pts)
    iso_r = 6.0 * float(fwhm_px)
    keep = []
    for i, row in req.iterrows():
        dists, _ = tree.query([row["x"], row["y"]], k=min(6, len(pts)))
        dists = np.atleast_1d(dists)
        nn = float(dists[1]) if len(dists) > 1 else float("inf")
        if nn >= iso_r:
            keep.append(i)
    return req.loc[keep].sort_values("flux", ascending=False).head(n_target)


def _fit_star_moffat_native(
    data: np.ndarray,
    stars: pd.DataFrame,
    *,
    fwhm_guess: float,
    cutout: int,
) -> dict:
    h, w = data.shape
    half = cutout // 2
    fwhms = []
    ellips = []
    for _, srow in stars.iterrows():
        sx = float(srow["x"])
        sy = float(srow["y"])
        x0, x1 = max(0, int(sx) - half), min(w, int(sx) + half + 1)
        y0, y1 = max(0, int(sy) - half), min(h, int(sy) + half + 1)
        cut = data[y0:y1, x0:x1]
        if cut.shape[0] < 7 or cut.shape[1] < 7:
            continue
        border = np.ones(cut.shape, dtype=bool)
        if cut.shape[0] > 4 and cut.shape[1] > 4:
            border[2:-2, 2:-2] = False
        bvals = cut[border]
        bvals = bvals[np.isfinite(bvals)]
        sky = float(np.median(bvals)) if bvals.size >= 4 else 0.0
        cut_sub = cut - sky
        if float(np.nanmax(cut_sub)) <= 0:
            continue
        yy, xx = np.mgrid[: cut_sub.shape[0], : cut_sub.shape[1]]
        xc = sx - x0
        yc = sy - y0
        alpha = MOFFAT_ALPHA
        gamma0 = max(0.5, fwhm_guess / (2.0 * math.sqrt(2.0 ** (1.0 / alpha) - 1.0)))
        amp = float(np.nanmax(cut_sub))
        model = Moffat2D(amplitude=amp, x_0=xc, y_0=yc, gamma=gamma0, alpha=alpha)
        model.gamma.bounds = (0.05, None)
        try:
            fitter = LevMarLSQFitter()
            fit = fitter(model, xx, yy, cut_sub)
            g = float(fit.gamma.value)
            a = float(fit.alpha.value)
            fwhms.append(_moffat_fwhm_oversampled(g, a))
            # round Moffat fit: use centroid offset as mild asymmetry proxy
            dx = float(fit.x_0.value) - xc
            dy = float(fit.y_0.value) - yc
            ellips.append(math.hypot(dx, dy) / max(g, 1e-6))
        except Exception:  # noqa: BLE001
            continue
    if not fwhms:
        return {"ok": False, "n": 0}
    return {
        "ok": True,
        "n": len(fwhms),
        "fwhm_native": float(np.median(fwhms)),
        "ellipticity_proxy_median": float(np.median(ellips)) if ellips else 0.0,
    }


def analyze_setup(
    draft_id: int,
    draft_dir: Path,
    db: VyvarDatabase,
    *,
    setup: str,
    filt: str | None = None,
) -> dict:
    """Build ePSF in-memory and measure Moffat FWHM on ePSF vs isolated stars."""
    ps = draft_dir / "platesolve" / setup
    ms_fits = ps / "MASTERSTAR.fits"
    ms_csv_path = ps / "masterstars_full_match.csv"
    if not ms_csv_path.is_file():
        ms_csv_path = ps / "comparison_stars.csv"
    meta_wcs = _load_wcs_meta(ms_fits)
    with fits.open(ms_fits, memmap=False) as hdul:
        hdr = hdul[0].header
        ms_data = np.asarray(hdul[0].data, dtype=np.float64)
    vy_fwhm = float(hdr.get("VY_FWHM", meta_wcs["fwhm_px"]) or meta_wcs["fwhm_px"])
    vy_gauss = float(header_core_fwhm_px(hdr))
    plate = float(meta_wcs["plate_scale_arcsec"])

    prep = _epsf_prepare_stars(ms_fits, ms_csv_path, db, draft_id)
    built = _epsf_build_imagepsf_from_stars(
        prep["stars"], osamp=OSAMP, fwhm_px=prep["fwhm_px"], cutout_size=prep["cutout_size"]
    )
    epsf_fit = np.asarray(built["arr"], dtype=np.float64)
    epsf_fit = epsf_fit / max(float(np.nanmax(epsf_fit)), 1e-12)

    moff = _fit_moffat2d_array(epsf_fit, fwhm_guess=vy_gauss)
    gauss = _fit_gauss2d_array(epsf_fit)
    peak = moff.get("peak", float(np.nanmax(epsf_fit))) if moff.get("ok") else float(np.nanmax(epsf_fit))
    cx = moff.get("x_0", epsf_fit.shape[1] / 2) if moff.get("ok") else epsf_fit.shape[1] / 2
    cy = moff.get("y_0", epsf_fit.shape[0] / 2) if moff.get("ok") else epsf_fit.shape[0] / 2
    azim = _azimuthal_fwhm(epsf_fit, peak=peak, cy=cy, cx=cx)
    buggy = float(built["qc"].get("epsf_fwhm_native_px") or float("nan"))
    seeing_all = _median_fwhm_obs_files(db, draft_id)
    seeing_f = _obs_fwhm_filter_median(db, draft_id, filt or "Red")

    iso = _select_isolated_stars(ms_csv_path, vy_gauss, n_target=30)
    stars_fit = _fit_star_moffat_native(
        ms_data, iso, fwhm_guess=vy_gauss, cutout=int(prep["cutout_size"])
    )

    fm = float(moff.get("fwhm_native", float("nan"))) if moff.get("ok") else float("nan")
    fs = float(stars_fit.get("fwhm_native", float("nan"))) if stars_fit.get("ok") else float("nan")
    ratio = fm / fs if math.isfinite(fm) and math.isfinite(fs) and fs > 0 else float("nan")

    n_frames = 0
    if filt:
        row = db.conn.execute(
            "SELECT COUNT(*) FROM OBS_FILES WHERE DRAFT_ID=? AND FILTER=?",
            (int(draft_id), str(filt)),
        ).fetchone()
        n_frames = int(row[0]) if row else 0

    return {
        "draft_id": draft_id,
        "setup": setup,
        "filter": filt,
        "n_frames": n_frames,
        "osamp": OSAMP,
        "n_epsf_stars": int(prep.get("n_ext", 0)),
        "plate_scale_arcsec": round(plate, 4),
        "vy_fwhm_header": round(vy_fwhm, 4),
        "vy_fwhm_gauss": round(vy_gauss, 4),
        "fwhm_moffat_native": round(fm, 4) if math.isfinite(fm) else None,
        "fwhm_gauss_native": round(gauss.get("fwhm_native", float("nan")), 4)
        if gauss.get("ok")
        else None,
        "fwhm_azim_native": round(azim.get("fwhm_native", float("nan")), 4) if azim.get("ok") else None,
        "buggy_halfmax_native": round(buggy, 4) if math.isfinite(buggy) else None,
        "fwhm_stars_native": round(fs, 4) if math.isfinite(fs) else None,
        "n_stars_fitted": int(stars_fit.get("n", 0)),
        "star_ellipticity_proxy_median": round(float(stars_fit.get("ellipticity_proxy_median", 0.0)), 4),
        "seeing_filter_obs_files": round(seeing_f, 4) if math.isfinite(seeing_f) else None,
        "seeing_all_obs_files": round(float(seeing_all), 4) if seeing_all is not None else None,
        "ratio_moffat_vs_stars": round(ratio, 4) if math.isfinite(ratio) else None,
        "fwhm_moffat_arcsec": round(fm * plate, 4) if math.isfinite(fm) else None,
        "fwhm_stars_arcsec": round(fs * plate, 4) if math.isfinite(fs) else None,
        "note": "ePSF built in-memory via production path; no persisted masterstar_epsf.fits.",
    }


def pick_richest_filter(db: VyvarDatabase, draft_id: int) -> tuple[str, int]:
    rows = db.conn.execute(
        "SELECT FILTER, COUNT(*) n FROM OBS_FILES WHERE DRAFT_ID=? GROUP BY FILTER ORDER BY n DESC",
        (int(draft_id),),
    ).fetchall()
    if not rows:
        return "Red", 0
    filt, n = rows[0][0], int(rows[0][1])
    return str(filt), n


def analyze_draft_367(draft_dir: Path | None = None, db: VyvarDatabase | None = None) -> dict:
    """Part-1 audit for draft 367 (richest filter setup)."""
    draft_dir = draft_dir or (_ROOT / "Archive" / "Drafts" / "draft_000367")
    if db is None:
        db = VyvarDatabase(AppConfig().database_path)
    filt, n_frames = pick_richest_filter(db, 367)
    # 367: equal 16 frames/filter; Red_180_2 best SNR (180s, red)
    setup_map = {"Red": "Red_180_2", "Green": "Green_180_2", "Blue": "Blue_180_2"}
    setup = setup_map.get(filt, "Red_180_2")
    row = analyze_setup(367, draft_dir, db, setup=setup, filt=filt)
    row["richest_filter"] = filt
    row["richest_filter_frames"] = n_frames
    return row


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="ePSF-vs-star FWHM audit")
    ap.add_argument("--draft", type=int, default=367)
    ap.add_argument("--setup", type=str, default="")
    ap.add_argument("--out", type=Path, default=_ROOT / "tmp" / "epsf_fwhm_367.json")
    args = ap.parse_args()
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    draft_dir = _ROOT / "Archive" / "Drafts" / f"draft_{args.draft:06d}"
    if args.draft == 367 and not args.setup:
        row = analyze_draft_367(draft_dir, db)
    else:
        setup = args.setup or "Red_180_2"
        filt = setup.split("_")[0]
        row = analyze_setup(args.draft, draft_dir, db, setup=setup, filt=filt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2), encoding="ascii")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
