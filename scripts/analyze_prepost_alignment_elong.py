#!/usr/bin/env python3
"""Part 1: pre-alignment vs post-alignment elongation on same sky stars."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from psf_photometry import _read_plate_scale_arcsec_px_from_fits, get_epsf_fwhm_from_context  # noqa: E402

diag_path = _ROOT / "scripts" / "diagnose_psf_elongation_362.py"
spec = importlib.util.spec_from_file_location("diag", diag_path)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

CASES = [
    (362, "NoFilter_60_2", "proc_V842_Her_Light_064.fits"),
    (364, "Luminance_180_2", "proc_CHI-1-CMOS_2024-08-10T23-10-14_Palomar_7_Luminance_180s_456068_cal.fits"),
]


def _fit_stars_wcs(fits_path: Path, stars: pd.DataFrame, cutout_h: int) -> pd.DataFrame:
    rows = []
    with fits.open(fits_path, memmap=True) as hd:
        data = np.asarray(hd[0].data, dtype=np.float32)
        w = WCS(hd[0].header)
    h, wpx = data.shape
    half = cutout_h // 2
    for _, st in stars.iterrows():
        ra = float(st["ra_deg"])
        de = float(st["dec_deg"])
        try:
            xp, yp = w.world_to_pixel_values(ra, de)
            x0, y0 = int(round(float(xp))), int(round(float(yp)))
        except Exception:  # noqa: BLE001
            continue
        y1, y2 = max(0, y0 - half), min(h, y0 + half + 1)
        x1, x2 = max(0, x0 - half), min(wpx, x0 + half + 1)
        cut = data[y1:y2, x1:x2]
        if cut.size < 9:
            continue
        fit = diag._fit_elliptical_gaussian(cut)
        if not fit.get("ok"):
            continue
        rows.append({"elongation": fit["elongation"], "pa_deg": fit["pa_deg"]})
    return pd.DataFrame(rows)


def run_case(draft_id: int, setup: str, frame_name: str) -> dict:
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}"
    proc_fits = draft / "processed" / "lights" / setup / frame_name
    ali_fits = draft / "detrended_aligned" / "lights" / setup / frame_name
    ali_csv = draft / "detrended_aligned" / "lights" / setup / frame_name.replace(".fits", ".csv")
    ps = draft / "platesolve" / setup
    ms = ps / "MASTERSTAR.fits"

    fwhm = float(get_epsf_fwhm_from_context(ms, db, draft_id))
    cutout_h = 9
    plate_scale = float(_read_plate_scale_arcsec_px_from_fits(ms))

    frame_df = pd.read_csv(ali_csv, low_memory=False, dtype={"catalog_id": str})
    with fits.open(ali_fits, memmap=True) as hd:
        img_shape = hd[0].data.shape
    from param_resolver import resolve_gain, resolve_read_noise  # noqa: E402

    with fits.open(ms, memmap=True) as hd:
        mhdr = hd[0].header
    g = float(resolve_gain(mhdr, db=db, equipment_id=None, cfg=cfg).value or 1.0)
    rn = float(resolve_read_noise(mhdr, db=db, equipment_id=None, cfg=cfg).value or 10.0)

    picked = diag._select_frame_stars_from_proc(
        frame_df,
        ps,
        fwhm_px=fwhm,
        plate_scale_arcsec_px=plate_scale,
        fit_shape=(cutout_h, cutout_h),
        gain=g,
        rn=rn,
        img_shape=img_shape,
    )
    if len(picked) > 40:
        picked = picked.head(40)
    picked = picked[picked["ra_deg"].notna() & picked["dec_deg"].notna()].copy()

    pre = _fit_stars_wcs(proc_fits, picked, cutout_h)
    post = _fit_stars_wcs(ali_fits, picked, cutout_h)

    with fits.open(ali_fits) as hd:
        vyalgm = str(hd[0].header.get("VYALGM", "?"))

    return {
        "draft_id": draft_id,
        "setup": setup,
        "frame": frame_name,
        "pre_file": str(proc_fits),
        "post_file": str(ali_fits),
        "n_picked": len(picked),
        "pre_n": len(pre),
        "post_n": len(post),
        "pre_median_elong": float(pre["elongation"].median()) if len(pre) else float("nan"),
        "post_median_elong": float(post["elongation"].median()) if len(post) else float("nan"),
        "pre_pa": diag._circular_mean_deg(pre["pa_deg"].to_numpy()) if len(pre) else float("nan"),
        "post_pa": diag._circular_mean_deg(post["pa_deg"].to_numpy()) if len(post) else float("nan"),
        "vyalgm": vyalgm,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for draft_id, setup, frame in CASES:
        r = run_case(draft_id, setup, frame)
        print(f"=== draft {draft_id} ({setup}) ===")
        print(f"frame: {r['frame']}")
        print(f"(a) PRE  file: {r['pre_file']}")
        print(f"(b) POST file: {r['post_file']}")
        print(f"stars selected: {r['n_picked']}  fits ok pre/post: {r['pre_n']}/{r['post_n']}")
        print(f"(a) PRE  median elong={r['pre_median_elong']:.4f}  circ-PA={r['pre_pa']:.1f} deg")
        print(f"(b) POST median elong={r['post_median_elong']:.4f}  circ-PA={r['post_pa']:.1f} deg")
        print(f"VYALGM (aligned): {r['vyalgm']}")
        print()


if __name__ == "__main__":
    main()
