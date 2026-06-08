from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from vyvar_alignment_frame import _alignment_detect_xy


def _count_detected_stars(fp: Path) -> int:
    # Mirror alignment attempts (sigma ladder) and take best count.
    attempts = [
        (3.5, 200),
        (2.5, 300),
        (2.0, 400),
        (1.5, 500),
    ]
    best = 0
    with fits.open(fp, memmap=False) as hdul:
        hdr = hdul[0].header
        data = np.asarray(hdul[0].data, dtype=np.float32)

    # In logs we typically use ~3px when header value is missing.
    fwhm_px = 3.0
    for det_sigma, max_stars in attempts:
        try:
            xy = _alignment_detect_xy(
                data,
                want_max=max_stars,
                det_sigma=float(det_sigma),
                fwhm_px=float(fwhm_px),
                label=fp.name,
                log_sink=None,  # no spam to infolog
            )
            n = int(len(xy)) if xy is not None else 0
            if n > best:
                best = n
        except Exception:
            continue

    return int(best)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="Path to alignment_report.csv")
    ap.add_argument(
        "--fits-root",
        required=True,
        help="Directory where proc_*.fits live (e.g. processed/lights/<setup>)",
    )
    ap.add_argument(
        "--only-when-zero",
        action="store_true",
        help="Only recompute detected_stars when current value is 0",
    )
    args = ap.parse_args()

    report_path = Path(args.report).expanduser().resolve()
    fits_root = Path(args.fits_root).expanduser().resolve()

    df = pd.read_csv(report_path)
    if "file" not in df.columns or "detected_stars" not in df.columns:
        raise RuntimeError("alignment_report.csv missing required columns: file, detected_stars")

    updated = 0
    missing = 0
    for i, row in df.iterrows():
        try:
            cur = int(row.get("detected_stars") or 0)
        except Exception:
            cur = 0
        if args.only_when_zero and cur != 0:
            continue
        fname = str(row["file"])
        fp = fits_root / fname
        if not fp.is_file():
            missing += 1
            continue
        n = _count_detected_stars(fp)
        if int(cur) != int(n):
            df.at[i, "detected_stars"] = int(n)
            updated += 1

    out_path = report_path
    df.to_csv(out_path, index=False)
    print(f"OK: updated rows={updated}, missing_fits={missing}, report={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

