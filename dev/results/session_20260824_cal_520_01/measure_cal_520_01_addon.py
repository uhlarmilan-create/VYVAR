"""CAL-520-01 addon: compact star mask + bias-floor multiplicative metric."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation, gaussian_filter

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src_py"))

G520 = REPO / "Archive" / "Drafts" / "draft_000520" / "non_calibrated" / "lights" / "g_60_4"
REF516 = (
    REPO
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "calibrated"
    / "lights"
    / "NoFilter_60_2"
    / "BO_CVn_Light_001.fits"
)
COMP = HERE / "m4_comp_forensics.csv"
TARGET = "1111749368289526912"


def load(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)


def compact_star_mask(arr: np.ndarray, *, k: float = 8.0, dilate: int = 8) -> np.ndarray:
    finite = np.isfinite(arr)
    fill = float(np.nanmedian(arr[finite]))
    work = np.where(finite, arr, fill)
    hp = work - gaussian_filter(work, 2.0)
    med = float(np.median(hp[finite]))
    mad = float(np.median(np.abs(hp[finite] - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(hp[finite]))
    stars = (hp > (med + k * sigma)) & finite
    if dilate:
        stars = binary_dilation(stars, iterations=int(dilate))
    return stars


def metrics(arr: np.ndarray, sigma: float = 80.0) -> dict:
    finite = np.isfinite(arr)
    stars = compact_star_mask(arr)
    sky = finite & (~stars)
    vals = arr[sky]
    p01 = float(np.percentile(vals, 1.0))
    p50 = float(np.median(vals))
    sky_above_floor = p50 - p01
    fill = p50
    work = np.where(sky, arr, fill)
    blur = gaussian_filter(work, sigma=float(sigma))
    bv = blur[sky]
    rms = float(np.std(bv))
    lo, hi = (float(x) for x in np.percentile(bv, [1.0, 99.0]))
    ptp = hi - lo
    illum = np.clip(work - p01, 0.0, None)
    blur_i = gaussian_filter(illum, sigma=float(sigma))
    bi = blur_i[sky]
    med_i = float(np.median(bi))
    rms_i = float(np.std(bi))
    lo_i, hi_i = (float(x) for x in np.percentile(bi, [1.0, 99.0]))
    ptp_i = hi_i - lo_i
    return {
        "n_sky": int(np.count_nonzero(sky)),
        "n_star_mask": int(np.count_nonzero(stars)),
        "p01_adu": p01,
        "sky_median_adu": p50,
        "sky_above_floor_adu": sky_above_floor,
        "lp80_rms_adu": rms,
        "lp80_p99_p1_adu": ptp,
        "rel_p99_p1_to_raw_median": ptp / p50 if p50 else None,
        "rel_p99_p1_to_sky_above_floor": ptp / sky_above_floor if sky_above_floor else None,
        "illum_lp80_rms_adu": rms_i,
        "illum_lp80_p99_p1_adu": ptp_i,
        "illum_rel_p99_p1": ptp_i / med_i if med_i else None,
        "illum_rel_rms": rms_i / med_i if med_i else None,
    }


def main() -> None:
    g_files = sorted(G520.glob("SSCam_*_g_*.fits"))
    paths = {
        "g_0000": g_files[0],
        "g_mid": g_files[len(g_files) // 2],
        "g_last": g_files[-1],
        "ref516_BO_CVn_Light_001": REF516,
    }
    out = {k: metrics(load(p)) for k, p in paths.items()}
    (HERE / "m2_large_scale_compact_mask.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    import pandas as pd

    df = pd.read_csv(HERE / "m4_comp_forensics.csv", dtype={"catalog_id": str})
    june = df[(df["band"] == "june_G_11.6_13.9") & (df["catalog_id"] != TARGET)]
    june_rms = pd.to_numeric(june["comp_rms_fieldwide_today"], errors="coerce").dropna()
    sel = df[df["selected_for_V0612"] == True]  # noqa: E712
    sel_row = pd.to_numeric(sel["comp_rms_on_selected_row"], errors="coerce").dropna()
    summary = json.loads((HERE / "m4_comp_forensics_summary.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "june_band_n_in_field_excluding_target": int(len(june)),
            "june_band_n_with_fieldwide_rms": int(len(june_rms)),
            "june_band_comp_rms_today_median": float(june_rms.median()) if len(june_rms) else None,
            "june_band_comp_rms_today_min": float(june_rms.min()) if len(june_rms) else None,
            "june_band_comp_rms_today_max": float(june_rms.max()) if len(june_rms) else None,
            "target_excluded_from_june_band": TARGET,
            "selected_comp_rms_from_photometry_csv_median": float(sel_row.median()) if len(sel_row) else None,
            "selected_comp_rms_from_photometry_csv_min": float(sel_row.min()) if len(sel_row) else None,
            "selected_comp_rms_from_photometry_csv_max": float(sel_row.max()) if len(sel_row) else None,
            "selected_phot_g_list": [float(x) for x in sel["phot_g_mean_mag"].tolist()],
        }
    )
    (HERE / "m4_comp_forensics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if "n_" in kk or "rel_" in kk or "p01" in kk or "sky_" in kk or "p99" in kk} for k, v in out.items()}, indent=2))
    print("june median", summary["june_band_comp_rms_today_median"], "n", summary["june_band_n_with_fieldwide_rms"])
    print("selected median", summary["selected_comp_rms_from_photometry_csv_median"])


if __name__ == "__main__":
    main()
