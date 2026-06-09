"""Tier-A single-frame synthetic generator (promoted from synth_fits.py PoC).

Generates a FITS frame with KNOWN ground truth plus injected contaminations.
Truth sidecar schema: stars[], contamination[], frame params.
ASCII only. Deterministic RNG seed 42.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from tests.validation.score import RNG_SEEDS

RNG = np.random.default_rng(RNG_SEEDS["gen_frame"])

# ---- frame / optics ----
NY, NX = 1024, 1024
PLATE_SCALE_ARCSEC = 1.30
GAIN_E_PER_ADU = 1.5
READ_NOISE_E = 9.0
SKY_ADU = 220.0
SAT_ADU = 60000.0
FWHM_PX = 3.2
MOFFAT_BETA = 2.5
ZP = 25.0
CRVAL = (56.75, 57.10)


def wcs_for_frame() -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [NX / 2, NY / 2]
    w.wcs.cdelt = [-PLATE_SCALE_ARCSEC / 3600.0, PLATE_SCALE_ARCSEC / 3600.0]
    w.wcs.crval = list(CRVAL)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def moffat_stamp(
    yc: float,
    xc: float,
    flux: float,
    fwhm: float,
    beta: float,
    *,
    ny: int = NY,
    nx: int = NX,
    ellip: float = 0.0,
    theta: float = 0.0,
) -> np.ndarray:
    """Render an (optionally elongated) Moffat PSF onto a full-frame array."""
    alpha = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
    half = int(np.ceil(6 * alpha)) + 1
    y0, y1 = max(0, int(yc) - half), min(ny, int(yc) + half + 1)
    x0, x1 = max(0, int(xc) - half), min(nx, int(xc) + half + 1)
    if y0 >= y1 or x0 >= x1:
        return np.zeros((ny, nx), dtype=np.float64)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dy, dx = yy - yc, xx - xc
    ct, st = np.cos(theta), np.sin(theta)
    xr = dx * ct + dy * st
    yr = -dx * st + dy * ct
    ax = alpha
    ay = alpha * (1.0 - ellip)
    r2 = (xr / ax) ** 2 + (yr / ay) ** 2
    psf = (beta - 1.0) / (np.pi * ax * ay) * (1.0 + r2) ** (-beta)
    stamp = np.zeros((ny, nx), dtype=np.float64)
    stamp[y0:y1, x0:x1] = flux * psf
    return stamp


def build_frame() -> tuple[np.ndarray, dict]:
    img = np.zeros((NY, NX), dtype=np.float64)
    truth: dict = {
        "tier": "A",
        "plate_scale_arcsec_px": PLATE_SCALE_ARCSEC,
        "fwhm_px": FWHM_PX,
        "zp": ZP,
        "gain_e_per_adu": GAIN_E_PER_ADU,
        "read_noise_e": READ_NOISE_E,
        "sat_adu": SAT_ADU,
        "crval": list(CRVAL),
        "stars": [],
        "contamination": [],
        "frame_params": {"rng_seed": RNG_SEEDS["gen_frame"]},
    }

    def add_star(name, yc, xc, mag, **kw):
        flux = 10.0 ** (-0.4 * (mag - ZP))
        img[:] += moffat_stamp(yc, xc, flux, FWHM_PX, MOFFAT_BETA, **kw)
        truth["stars"].append(
            {
                "name": name,
                "y": float(yc),
                "x": float(xc),
                "mag": float(mag),
                "flux_adu": float(flux),
                **{k: float(v) for k, v in kw.items()},
            }
        )
        return flux

    for i in range(12):
        yy = 120 + (i % 4) * 230 + RNG.uniform(-15, 15)
        xx = 150 + (i // 4) * 320 + RNG.uniform(-15, 15)
        add_star(f"comp_{i:02d}", yy, xx, 11.5 + i * 0.25)

    add_star("blend_A_target", 500, 500, 12.8)
    add_star("blend_B_neighbor", 500 + 1.0 * FWHM_PX, 500, 14.5)
    truth["contamination"].append(
        {
            "type": "blend_unresolved",
            "sep_fwhm": 1.0,
            "target": "blend_A_target",
            "neighbor": "blend_B_neighbor",
            "expect": "is_blended=True (nn<=1.5 FWHM); flagged for deblend",
        }
    )

    add_star("pair_C", 300, 760, 13.0)
    add_star("pair_D", 300, 760 + 2.5 * FWHM_PX, 13.4)
    truth["contamination"].append(
        {
            "type": "blend_resolvable",
            "sep_fwhm": 2.5,
            "stars": ["pair_C", "pair_D"],
            "expect": "is_blended=False (nn>1.5 FWHM)",
        }
    )

    add_star("smeared_star", 780, 300, 12.2, ellip=0.72, theta=0.6)
    truth["contamination"].append(
        {
            "type": "tracking_smear",
            "star": "smeared_star",
            "ellip": 0.72,
            "expect": "epsf_asymmetry > 0.1 QC warning",
        }
    )

    gy, gx = np.mgrid[0:NY, 0:NX]
    gradient = 90.0 * (gx / NX) + 60.0 * (gy / NY)
    sky = SKY_ADU + gradient
    truth["contamination"].append(
        {
            "type": "illumination_gradient_moonlight",
            "peak_adu": float(gradient.max()),
            "expect": "flat-only leaves gradient; CoLiTecVS inverse-median would remove it",
        }
    )
    img += sky

    electrons = np.clip(img * GAIN_E_PER_ADU, 0, None)
    img = RNG.poisson(electrons).astype(np.float64) / GAIN_E_PER_ADU
    img += RNG.normal(0.0, READ_NOISE_E / GAIN_E_PER_ADU, size=img.shape)

    cr_truth = []
    for _ in range(6):
        cy, cx = RNG.integers(40, NY - 40), RNG.integers(40, NX - 40)
        img[cy, cx] += RNG.uniform(8000, 30000)
        cr_truth.append([int(cy), int(cx)])
    truth["contamination"].append(
        {
            "type": "cosmic_rays",
            "pixels": cr_truth,
            "expect": "rejected by sigma-clip / spike index; not in clean photometry",
        }
    )

    add_star("saturated_star", 620, 880, 8.0)
    truth["contamination"].append(
        {
            "type": "saturation",
            "star": "saturated_star",
            "sat_adu": SAT_ADU,
            "expect": "flagged saturated; excluded from comps",
        }
    )
    img = np.clip(img, 0, SAT_ADU)

    w = wcs_for_frame()
    for star in truth["stars"]:
        ra, dec = w.all_pix2world(star["x"], star["y"], 0)
        star["ra_deg"] = float(ra)
        star["dec_deg"] = float(dec)

    return img.astype(np.float32), truth


def write_frame(out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fits_path = out_dir / "tier_a_frame.fits"
    truth_path = out_dir / "tier_a_truth.json"

    img, truth = build_frame()
    w = wcs_for_frame()
    hdr = w.to_header()
    hdr["VY_FWHM"] = (FWHM_PX, "synthetic injected FWHM [px]")
    hdr["GAIN"] = (GAIN_E_PER_ADU, "e-/ADU")
    hdr["RDNOISE"] = (READ_NOISE_E, "e-")
    hdr["SATURATE"] = (SAT_ADU, "ADU")
    hdr["PXSCALE"] = (PLATE_SCALE_ARCSEC, "arcsec/px")
    hdr["FILTER"] = ("V", "synthetic")
    hdr["OBJECT"] = ("SYNTH_VALIDATION", "")
    hdr["DATE-OBS"] = ("2026-04-23T22:00:00", "synthetic mid-exposure UTC")
    hdr["EXPTIME"] = (60.0, "s")
    hdr["SITELAT"] = (50.075, "deg")
    hdr["SITELONG"] = (14.437, "deg")
    hdr["SITEELEV"] = (525.0, "m")
    hdr["VYTARGRA"] = (CRVAL[0], "target RA deg (synthetic)")
    hdr["VYTARGDE"] = (CRVAL[1], "target Dec deg (synthetic)")
    hdr["COMMENT"] = "Synthetic VYVAR validation frame -- injected truth in sidecar JSON"
    fits.PrimaryHDU(data=img, header=hdr).writeto(fits_path, overwrite=True)
    with open(truth_path, "w", encoding="ascii") as f:
        json.dump(truth, f, indent=2)
    return fits_path, truth_path


if __name__ == "__main__":
    data = Path(__file__).resolve().parent / "data" / "tier_a"
    fp, tp = write_frame(data)
    print(f"wrote {fp}")
    print(f"wrote {tp}")
