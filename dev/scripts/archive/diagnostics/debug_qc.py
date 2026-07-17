from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python debug_qc.py <fits_path>")
        return 2

    fp = Path(sys.argv[1])
    print("file:", fp)
    print("exists:", fp.exists())
    if not fp.exists():
        return 1

    with fits.open(fp, memmap=False) as hdul:
        hdr = hdul[0].header
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)

    print("shape:", data.shape, "dtype:", data.dtype)
    finite = np.isfinite(data)
    print("finite_frac:", float(finite.mean()))
    arr = data[finite]
    for p in [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]:
        print(f"p{p}:", float(np.percentile(arr, p)))
    print("min/max:", float(arr.min()), float(arr.max()))

    med = float(np.median(arr))
    std = float(np.std(arr))
    res = arr - med
    print("median/std:", med, std)
    print("tail counts (>+4std / <-4std):", int((res > 4 * std).sum()), int((res < -4 * std).sum()))

    mean2, med2, std2 = sigma_clipped_stats(arr, sigma=3.0, maxiters=5)
    std2 = float(std2)
    print("sigma_clipped med/std:", float(med2), std2)

    img2 = np.nan_to_num((data - float(med2)).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    try:
        from photutils.detection import DAOStarFinder

        for thresh in [5, 4, 3, 2.5, 2.0]:
            tbl = DAOStarFinder(fwhm=3.0, threshold=thresh * std2)(img2)
            tbl2 = DAOStarFinder(fwhm=3.0, threshold=thresh * std2)(-img2)
            print("DAO thresh", thresh, "n:", 0 if tbl is None else len(tbl), "| flipped n:", 0 if tbl2 is None else len(tbl2))
    except Exception as exc:  # noqa: BLE001
        print("DAOStarFinder failed:", exc)

    print("FILTER:", hdr.get("FILTER"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

