from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def _check(path: Path) -> None:
    print(f"MASTERSTAR: {path}")
    if not path.is_file():
        print("  missing")
        return
    hdr = fits.getheader(path)
    try:
        w = WCS(hdr)
        pm = getattr(w, "pixel_scale_matrix", None)
        det = float(np.linalg.det(pm)) if pm is not None else float("nan")
        print(f"  has_celestial={getattr(w, 'has_celestial', False)} det(pixel_scale_matrix)={det}")
    except Exception as exc:  # noqa: BLE001
        print(f"  WCS error: {exc}")
    print(
        "  CD:",
        hdr.get("CD1_1"),
        hdr.get("CD1_2"),
        hdr.get("CD2_1"),
        hdr.get("CD2_2"),
    )
    print("  CRVAL:", hdr.get("CRVAL1"), hdr.get("CRVAL2"))


def main() -> None:
    base = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts")
    for did in (218, 219):
        _check(base / f"draft_{did:06d}" / "platesolve" / "MASTERSTAR.fits")


if __name__ == "__main__":
    main()

