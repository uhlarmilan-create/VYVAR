#!/usr/bin/env python3
"""Crop M67 LRGB archive frames to a ~1 deg box centred on the cluster."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
M67_RA = 132.846
M67_DEC = 11.814
DEFAULT_HALF_PX = 900  # ~1800 px @ ~2 arcsec/px ≈ 1 deg


def _crop_box_from_red(red_fits: Path, half_px: int) -> tuple[int, int, int, int]:
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    import astropy.units as u

    with fits.open(red_fits, memmap=False) as hd:
        w = WCS(hd[0].header)
        nx = int(hd[0].header.get("NAXIS1", 0) or 0)
        ny = int(hd[0].header.get("NAXIS2", 0) or 0)
    c = SkyCoord(M67_RA * u.deg, M67_DEC * u.deg, frame="icrs")
    cx, cy = w.world_to_pixel(c)
    cx_i, cy_i = int(round(float(cx))), int(round(float(cy)))
    x0 = max(0, cx_i - int(half_px))
    y0 = max(0, cy_i - int(half_px))
    x1 = min(nx, cx_i + int(half_px))
    y1 = min(ny, cy_i + int(half_px))
    return x0, y0, x1, y1


def crop_fits(src: Path, dst: Path, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    with fits.open(src, memmap=False) as hd:
        data = np.array(hd[0].data[y0:y1, x0:x1], copy=True)
        hdr = hd[0].header.copy()
    hdr["NAXIS1"] = int(x1 - x0)
    hdr["NAXIS2"] = int(y1 - y0)
    if "CRPIX1" in hdr:
        hdr["CRPIX1"] = float(hdr["CRPIX1"]) - float(x0)
    if "CRPIX2" in hdr:
        hdr["CRPIX2"] = float(hdr["CRPIX2"]) - float(y0)
    hdr["VYCROPX0"] = (int(x0), "VYVAR crop origin X [0-based]")
    hdr["VYCROPY0"] = (int(y0), "VYVAR crop origin Y [0-based]")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=data, header=hdr).writeto(dst, overwrite=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=_ROOT / "Archive" / "m67")
    ap.add_argument("--out", type=Path, default=_ROOT / "Archive" / "m67" / "LRGB_cropped")
    ap.add_argument("--half-px", type=int, default=DEFAULT_HALF_PX)
    args = ap.parse_args()

    red_dir = args.archive / "Red"
    red_files = sorted(red_dir.glob("*.fits"))
    if not red_files:
        print(f"No Red FITS in {red_dir}")
        return 1
    box = _crop_box_from_red(red_files[0], int(args.half_px))
    report = {"box_px": box, "m67_ra_deg": M67_RA, "m67_dec_deg": M67_DEC, "written": []}

    for filt in ("Red", "Green", "Blue", "Luminance"):
        src_dir = args.archive / filt
        dst_dir = args.out / filt
        if not src_dir.is_dir():
            continue
        for fp in sorted(src_dir.glob("*.fits")):
            dst = dst_dir / fp.name
            crop_fits(fp, dst, box)
            report["written"].append(str(dst))

    meta = args.out / "crop_meta.json"
    meta.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"n_files": len(report["written"]), "box_px": box, "out": str(args.out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
