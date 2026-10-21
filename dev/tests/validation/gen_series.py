"""Tier-B Gaia-structured series generator (~60 frames).

Uses fallback (b): synthetic catalog with Gaia-like source_id fields fed through the same
matching/QA paths as real data. Stars placed at catalog (ra, dec) via frame WCS.
Deterministic RNG seed 43.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from astropy.io import fits

from tests.validation.gen_frame import (
    CRVAL,
    GAIN_E_PER_ADU,
    MOFFAT_BETA,
    NX,
    NY,
    PLATE_SCALE_ARCSEC,
    READ_NOISE_E,
    SAT_ADU,
    SKY_ADU,
    ZP,
    moffat_stamp,
    wcs_for_frame,
)
from tests.validation.score import RNG_SEEDS

RNG = np.random.default_rng(RNG_SEEDS["gen_series"])

N_FRAMES = 60
TARGET_ID = "486430957815961344"
BAD_COMP_ID = "486430957815961345"
STRONG_TARGET_ID = "486430957815961346"
WEAK_TARGET_ID = "486430957815961347"

# B1 variability
B1_AMP_MAG = 0.15
B1_PERIOD_D = 2.5

# B2 bad comp
B2_AMP_MAG = 0.05
B2_CR_FRAMES = {10, 25, 40}

# B4 color-term injection (mag offset per bp_rp)
B4_COLOR_SLOPE = 0.08  # mag per (bp_rp - median)

# B6 seeing jitter
FWHM_MIN = 2.8
FWHM_MAX = 3.6


def _mag_to_flux(mag: float) -> float:
    return 10.0 ** (-0.4 * (mag - ZP))


def _catalog() -> list[dict]:
    """Synthetic Gaia-structured field catalog (15 stars)."""
    w = wcs_for_frame()
    entries: list[dict] = []
    specs = [
        (TARGET_ID, 512, 512, 12.0, 0.65, "target_b1"),
        (BAD_COMP_ID, 680, 420, 12.3, 0.55, "bad_comp_b2"),
        (STRONG_TARGET_ID, 350, 650, 12.1, 0.70, "strong_trust_b3"),
        (WEAK_TARGET_ID, 750, 750, 13.5, 0.50, "weak_trust_b3"),
    ]
    for i in range(11):
        sid = f"48643095781596{i + 1348:04d}"
        y = 100 + (i % 4) * 220 + RNG.uniform(-10, 10)
        x = 120 + (i // 4) * 280 + RNG.uniform(-10, 10)
        mag = 11.8 + i * 0.12
        bprp = 0.8 + 0.15 * (i % 5)
        specs.append((sid, x, y, mag, bprp, f"comp_{i:02d}"))

    for sid, x, y, mag, bprp, name in specs:
        ra, dec = w.all_pix2world(x, y, 0)
        entries.append(
            {
                "source_id": sid,
                "catalog_id": sid,
                "name": name,
                "ra_deg": float(ra),
                "dec_deg": float(dec),
                "x": float(x),
                "y": float(y),
                "phot_g_mean_mag": float(mag),
                "mag": float(mag),
                "bp_rp": float(bprp),
            }
        )
    return entries


def _frame_truth(frame_i: int, catalog: list[dict]) -> dict:
    jd_mid = 2461154.0 + frame_i * (B1_PERIOD_D / N_FRAMES)
    phase = 2.0 * math.pi * (jd_mid - 2461154.0) / B1_PERIOD_D
    b1_delta = B1_AMP_MAG * math.sin(phase)
    b2_delta = B2_AMP_MAG * math.sin(phase + 0.3) if frame_i not in B2_CR_FRAMES else 0.0
    fwhm = FWHM_MIN + (FWHM_MAX - FWHM_MIN) * (0.5 + 0.5 * math.sin(frame_i * 0.17))
    sky_off = 5.0 * math.sin(frame_i * 0.11)
    return {
        "frame_index": frame_i,
        "fwhm_px": fwhm,
        "sky_offset_adu": sky_off,
        "jd_mid": jd_mid,
        "target_delta_mag": {TARGET_ID: b1_delta},
        "bad_comp_delta_mag": {BAD_COMP_ID: b2_delta},
        "cr_frames_bad_comp": sorted(B2_CR_FRAMES),
        "color_slope_injected": B4_COLOR_SLOPE,
    }


def build_series_frame(frame_i: int, catalog: list[dict]) -> tuple[np.ndarray, dict]:
    ft = _frame_truth(frame_i, catalog)
    fwhm = ft["fwhm_px"]
    img = np.zeros((NY, NX), dtype=np.float64)

    gy, gx = np.mgrid[0:NY, 0:NX]
    gradient = 40.0 * (gx / NX)
    sky = SKY_ADU + gradient + ft["sky_offset_adu"]
    img += sky

    star_flux: dict[str, float] = {}
    bp_med = float(np.median([c["bp_rp"] for c in catalog]))

    for star in catalog:
        sid = star["catalog_id"]
        mag = star["mag"]
        if sid == TARGET_ID:
            mag += ft["target_delta_mag"][TARGET_ID]
        elif sid == BAD_COMP_ID:
            mag += ft["bad_comp_delta_mag"].get(BAD_COMP_ID, 0.0)
        if sid not in (TARGET_ID, BAD_COMP_ID, WEAK_TARGET_ID, STRONG_TARGET_ID):
            mag += B4_COLOR_SLOPE * (star["bp_rp"] - bp_med)
        flux = _mag_to_flux(mag)
        img += moffat_stamp(star["y"], star["x"], flux, fwhm, MOFFAT_BETA)
        star_flux[sid] = flux

    electrons = np.clip(img * GAIN_E_PER_ADU, 0, None)
    img = RNG.poisson(electrons).astype(np.float64) / GAIN_E_PER_ADU
    img += RNG.normal(0.0, READ_NOISE_E / GAIN_E_PER_ADU, size=img.shape)

    if frame_i in B2_CR_FRAMES:
        bc = next(c for c in catalog if c["catalog_id"] == BAD_COMP_ID)
        cy, cx = int(bc["y"]), int(bc["x"])
        img[cy, cx] += RNG.uniform(12000, 25000)

    n_cr = RNG.integers(1, 4)
    cr_pixels = []
    for _ in range(int(n_cr)):
        cy, cx = RNG.integers(30, NY - 30), RNG.integers(30, NX - 30)
        img[cy, cx] += RNG.uniform(5000, 15000)
        cr_pixels.append([int(cy), int(cx)])

    img = np.clip(img, 0, SAT_ADU)

    truth = {
        "tier": "B",
        "frame_index": frame_i,
        "frame_params": ft,
        "catalog": catalog,
        "star_flux_adu": star_flux,
        "cr_pixels": cr_pixels,
        "b1_period_d": B1_PERIOD_D,
        "b1_amp_mag": B1_AMP_MAG,
    }
    return img.astype(np.float32), truth


def write_series(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    proc_dir = out_dir / "proc"
    phot_dir = out_dir / "photometry"
    frames_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    phot_dir.mkdir(parents=True, exist_ok=True)

    catalog = _catalog()
    series_meta = {
        "tier": "B",
        "n_frames": N_FRAMES,
        "catalog_source": "synthetic_gaia_structured_fallback_b",
        "catalog_note": "Matching uses Gaia-like source_id; positions are injected, not DR3 cone.",
        "rng_seed": RNG_SEEDS["gen_series"],
        "target_id": TARGET_ID,
        "bad_comp_id": BAD_COMP_ID,
        "b1_amp_mag": B1_AMP_MAG,
        "b1_period_d": B1_PERIOD_D,
        "b4_color_slope": B4_COLOR_SLOPE,
        "frames": [],
    }

    w = wcs_for_frame()
    comp_ids = [c["catalog_id"] for c in catalog if c["catalog_id"] != TARGET_ID][:8]

    for fi in range(N_FRAMES):
        img, truth = build_series_frame(fi, catalog)
        fname = f"frame_{fi:03d}.fits"
        fpath = frames_dir / fname
        hdr = w.to_header()
        hdr["VY_FWHM"] = (truth["frame_params"]["fwhm_px"], "injected FWHM [px]")
        hdr["GAIN"] = (GAIN_E_PER_ADU, "e-/ADU")
        hdr["RDNOISE"] = (READ_NOISE_E, "e-")
        hdr["SATURATE"] = (SAT_ADU, "ADU")
        hdr["PXSCALE"] = (PLATE_SCALE_ARCSEC, "arcsec/px")
        hdr["FILTER"] = ("V", "synthetic")
        hdr["OBJECT"] = ("SYNTH_SERIES", "")
        hdr["DATE-OBS"] = (f"2026-04-23T{20 + fi // 10:02d}:{(fi * 7) % 60:02d}:00", "UTC")
        hdr["EXPTIME"] = (60.0, "s")
        hdr["SITELAT"] = (50.075, "deg")
        hdr["SITELONG"] = (14.437, "deg")
        hdr["SITEELEV"] = (525.0, "m")
        hdr["CRVAL1"] = (CRVAL[0], "deg")
        hdr["CRVAL2"] = (CRVAL[1], "deg")
        fits.PrimaryHDU(data=img, header=hdr).writeto(fpath, overwrite=True)

        tpath = frames_dir / f"frame_{fi:03d}_truth.json"
        with open(tpath, "w", encoding="ascii") as f:
            json.dump(truth, f, indent=2)

        jd_mid = truth["frame_params"]["jd_mid"]
        proc_rows = []
        for star in catalog:
            sid = star["catalog_id"]
            flux = truth["star_flux_adu"][sid]
            proc_rows.append(
                {
                    "catalog_id": sid,
                    "source_file": fname,
                    "dao_flux": flux,
                    "flux": flux,
                    "x": star["x"],
                    "y": star["y"],
                    "ra_deg": star["ra_deg"],
                    "dec_deg": star["dec_deg"],
                    "phot_g_mean_mag": star["phot_g_mean_mag"],
                    "bp_rp": star["bp_rp"],
                    "bjd_tdb_mid": jd_mid,
                    "jd_mid": jd_mid,
                    "airmass": 1.2 + 0.05 * math.sin(fi * 0.2),
                }
            )
        proc_csv = proc_dir / f"proc_{fi:03d}.csv"
        import pandas as pd

        pd.DataFrame(proc_rows).to_csv(proc_csv, index=False)
        series_meta["frames"].append({"fits": str(fpath.name), "proc": str(proc_csv.name)})

    import pandas as pd

    comp_rows = []
    for tid in (TARGET_ID, STRONG_TARGET_ID, WEAK_TARGET_ID):
        pool = [c for c in comp_ids if c != BAD_COMP_ID]
        for cid in pool[:6]:
            comp_rows.append({"target_catalog_id": tid, "catalog_id": cid})
    pd.DataFrame(comp_rows).to_csv(phot_dir / "comparison_stars_per_target.csv", index=False)

    summ_rows = [
        {
            "catalog_id": TARGET_ID,
            "vsx_name": "SYNTH_B1",
            "n_clean": 55,
            "lc_quality_flag": "good",
            "lc_rms": 0.02,
        },
        {
            "catalog_id": STRONG_TARGET_ID,
            "vsx_name": "SYNTH_STRONG",
            "n_clean": 58,
            "lc_quality_flag": "good",
            "lc_rms": 0.015,
        },
        {
            "catalog_id": WEAK_TARGET_ID,
            "vsx_name": "SYNTH_WEAK",
            "n_clean": 4,
            "lc_quality_flag": "poor",
            "lc_rms": 0.08,
        },
    ]
    pd.DataFrame(summ_rows).to_csv(phot_dir / "photometry_summary.csv", index=False)

    meta_path = out_dir / "series_meta.json"
    with open(meta_path, "w", encoding="ascii") as f:
        json.dump(series_meta, f, indent=2)
    return series_meta


if __name__ == "__main__":
    meta = write_series(Path(__file__).resolve().parent / "data" / "tier_b")
    print(f"wrote tier B series: {meta['n_frames']} frames")
