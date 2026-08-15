#!/usr/bin/env python
"""IMPL-02 Part A: diagnose CoG / SNR background contamination (ASCII)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    _estimate_annulus_sky_pp,
    _frame_data_for_snr_ee,
    _load_star_xy_for_snr_ee,
    _median_bkg_var_from_aligned_frames,
    _measure_ee_curve_for_snr_table,
    estimate_median_sky_adu_per_px_for_snr_table,
    estimate_star_free_per_pixel_variance_adu2,
    measure_growth_curve_ee,
)

DRAFT = ROOT / "Archive" / "Drafts" / "draft_000514"
PROC = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"
OUT = ROOT / "dev" / "results"


def _sha() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def star_free_median(data: np.ndarray) -> dict:
    """Median / MAD of star-free edge patches (same idea as variance estimator)."""
    d = np.asarray(data, dtype=np.float64)
    h, w = d.shape
    # Corner patches away from centre
    patches = []
    bw, bh = max(32, w // 16), max(32, h // 16)
    for y0, x0 in ((0, 0), (0, w - bw), (h - bh, 0), (h - bh, w - bw)):
        patches.append(d[y0 : y0 + bh, x0 : x0 + bw].ravel())
    # Also use pixels below global median (sky-dominated)
    med0 = float(np.nanmedian(d))
    low = d[np.isfinite(d) & (d <= med0)]
    patch = np.concatenate(patches)
    patch = patch[np.isfinite(patch)]
    mad = float(1.4826 * np.median(np.abs(patch - np.median(patch)))) if patch.size else float("nan")
    return {
        "edge_patch_median": float(np.nanmedian(patch)) if patch.size else float("nan"),
        "edge_patch_mad": mad,
        "edge_patch_mean": float(np.nanmean(patch)) if patch.size else float("nan"),
        "full_frame_median": med0,
        "low_half_median": float(np.nanmedian(low)) if low.size else float("nan"),
        "n_edge": int(patch.size),
    }


def main() -> int:
    cfg = AppConfig(project_root=ROOT)
    ms = None
    for cand in (
        DRAFT / "platesolve" / "NoFilter_60_2" / "MASTERSTAR.fits",
        DRAFT / "platesolve" / "MASTERSTAR.fits",
    ):
        if cand.is_file():
            ms = cand
            break
    aligned = sorted(PROC.glob("proc_*.fits"))[:12] if PROC.is_dir() else []

    data = _frame_data_for_snr_ee(
        masterstar_fits_path=ms,
        aligned_fits_paths=aligned,
        aligned_ram_frames=None,
    )
    if data is None:
        print("FAIL: no frame data")
        return 1

    level = star_free_median(data)
    bkg_var_frames = _median_bkg_var_from_aligned_frames(aligned_fits_paths=aligned)
    bkg_var_this = estimate_star_free_per_pixel_variance_adu2(data)
    sky_for_snr = estimate_median_sky_adu_per_px_for_snr_table(
        aligned_fits_paths=aligned,
        fallback=1581.6,
    )

    # Per-star annulus sky used by CoG path
    xy = _load_star_xy_for_snr_ee(
        masterstars_csv=None, draft_dir=DRAFT, masterstar_fits_path=ms
    )
    fwhm = 3.476  # from IMPL-01 artifact; also resolve if possible
    from photometry_core import resolve_fwhm_px_for_snr_aperture_table

    fwhm_r, _ = resolve_fwhm_px_for_snr_aperture_table(
        masterstar_fits_path=ms,
        masterstar_selection={},
        aligned_fits_paths=aligned,
    )
    if fwhm_r and math.isfinite(float(fwhm_r)):
        fwhm = float(fwhm_r)

    cog_sky = {}
    if xy is not None:
        x, y, flux, sky_cat = xy
        r_in = max(4.0, 3.0 * fwhm)
        r_out = max(r_in + 2.0, 5.0 * fwhm)
        sky_est = _estimate_annulus_sky_pp(data, x, y, r_in=r_in, r_out=r_out)
        cog_sky = {
            "annulus_r_in": r_in,
            "annulus_r_out": r_out,
            "sky_est_median": float(np.nanmedian(sky_est)),
            "sky_est_p16": float(np.nanpercentile(sky_est[np.isfinite(sky_est)], 16)),
            "sky_est_p84": float(np.nanpercentile(sky_est[np.isfinite(sky_est)], 84)),
            "catalog_sky_median": float(np.nanmedian(sky_cat)),
            "n_stars": int(len(x)),
            "ref_r_px": 4.5 * fwhm,
            "overlap_note": (
                "COG ladder extends to cog_ref_fwhm*fwhm; sky annulus starts at 3*fwhm "
                "- ladder outer radii sit inside the sky annulus"
            ),
        }

    # sigma_bkg_ap from proc -> per-px variance
    proc_csv = sorted(PROC.glob("proc_*.csv"))[:20]
    sig_rows = []
    for p in proc_csv:
        df = pd.read_csv(p)
        if "sigma_bkg_ap" not in df.columns or "aperture_r_px" not in df.columns:
            continue
        sig = pd.to_numeric(df["sigma_bkg_ap"], errors="coerce")
        rap = pd.to_numeric(df["aperture_r_px"], errors="coerce")
        sky = pd.to_numeric(df.get("sky_adu_per_px_annulus"), errors="coerce")
        ok = sig.notna() & rap.notna() & (rap > 0)
        if not ok.any():
            continue
        area = math.pi * rap[ok].to_numpy(float) ** 2
        var_px = (sig[ok].to_numpy(float) ** 2) / np.maximum(area, 1e-12)
        sig_rows.append(
            {
                "file": p.name,
                "sigma_bkg_ap_med": float(np.nanmedian(sig[ok])),
                "var_px_med": float(np.nanmedian(var_px)),
                "sky_annulus_med": float(np.nanmedian(sky)) if sky is not None else float("nan"),
                "r_ap_med": float(np.nanmedian(rap[ok])),
            }
        )
    var_px_all = [r["var_px_med"] for r in sig_rows]
    sigma_bkg_summary = {
        "n_frames": len(sig_rows),
        "var_px_from_sigma_bkg_ap_median": float(np.nanmedian(var_px_all)) if var_px_all else None,
        "sigma_bkg_ap_median": float(np.nanmedian([r["sigma_bkg_ap_med"] for r in sig_rows]))
        if sig_rows
        else None,
        "sky_annulus_median": float(np.nanmedian([r["sky_annulus_med"] for r in sig_rows]))
        if sig_rows
        else None,
    }

    # Reconstructed Howell bkg_var from sky_for_snr (what SNR uses if bkg_var None)
    g = 1.0  # what IMPL-01 sometimes used; also try cfg/equipment
    rn = 10.0
    howell_var = float(sky_for_snr) / g + (rn / g) ** 2

    # Measure EE with current path and also with Q4-style norm at 12 and forced local sky
    ee = _measure_ee_curve_for_snr_table(
        fwhm_px=fwhm,
        gain=g,
        read_noise=rn,
        draft_dir=DRAFT,
        masterstar_fits_path=ms,
        masterstars_csv=None,
        aligned_fits_paths=aligned,
        aligned_ram_frames=None,
        cfg=cfg,
    )

    # Derivative profile
    dee = []
    if ee.get("ok") and ee.get("ee_radii") is not None:
        rr = np.asarray(ee["ee_radii"], float)
        cc = np.asarray(ee["ee_curve"], float)
        for i in range(1, len(rr)):
            dee.append(
                {
                    "r_mid": float(0.5 * (rr[i] + rr[i - 1])),
                    "dEE_dr": float((cc[i] - cc[i - 1]) / (rr[i] - rr[i - 1])),
                    "beyond_1p5_fwhm": bool(rr[i] > 1.5 * fwhm),
                }
            )
        # r90
        r90 = float(rr[np.argmin(np.abs(cc - 0.9))]) if cc.size else float("nan")
    else:
        r90 = float("nan")

    # Single-frame check: is one proc FITS sky-subtracted?
    single = {}
    if aligned:
        with fits.open(aligned[0], memmap=True) as hd:
            d0 = np.asarray(hd[0].data, float)
            hdr = hd[0].header
        single = {
            "file": aligned[0].name,
            "level": star_free_median(d0),
            "VY_CALSTAGE": str(hdr.get("VY_CALSTAGE", "")),
            "bkg_var": estimate_star_free_per_pixel_variance_adu2(d0),
        }

    out = {
        "commit_sha": _sha(),
        "fwhm_px": fwhm,
        "cog_frame": {
            "source": "median of up to 5 aligned proc FITS (or MASTERSTAR)",
            "n_aligned_available": len(aligned),
            "shape": list(data.shape),
            "star_free_level": level,
            "interpretation": (
                "near-zero median => sky-subtracted; large positive median => not subtracted"
            ),
        },
        "single_proc_frame": single,
        "background_used": {
            "cog_annulus_sky": cog_sky,
            "snr_sky_adu_per_px_estimate": sky_for_snr,
            "snr_sky_note": (
                "estimate_median_sky_adu_per_px_for_snr_table uses noise-floor helper, "
                "NOT a sky pedestal measurement"
            ),
            "bkg_var_from_aligned_frames": bkg_var_frames,
            "bkg_var_from_cog_median_frame": bkg_var_this,
            "howell_reconstructed_var_sky_g_rn": howell_var,
            "gain_assumed_for_howell": g,
            "rn_assumed_for_howell": rn,
        },
        "sigma_bkg_ap_proc": sigma_bkg_summary,
        "agreement": {
            "bkg_var_frames_vs_sigma_bkg_ap_var_px": {
                "frames": bkg_var_frames,
                "from_sigma_bkg_ap": sigma_bkg_summary.get("var_px_from_sigma_bkg_ap_median"),
                "ratio": (
                    float(bkg_var_frames)
                    / float(sigma_bkg_summary["var_px_from_sigma_bkg_ap_median"])
                    if bkg_var_frames
                    and sigma_bkg_summary.get("var_px_from_sigma_bkg_ap_median")
                    and float(sigma_bkg_summary["var_px_from_sigma_bkg_ap_median"]) > 0
                    else None
                ),
            }
        },
        "ee_current": {
            "ok": ee.get("ok"),
            "n_cog": ee.get("n_cog"),
            "ref_r_px": ee.get("ref_r_px"),
            "flatness_tail_over_norm": ee.get("flatness_tail_over_norm"),
            "r90_px": r90,
            "dEE_dr": dee,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "IMPL_02_part_a_bkg.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
