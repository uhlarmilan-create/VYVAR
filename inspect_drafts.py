"""VYVAR draft inspector -- summarize each draft for the PSF-suitability decision.

Run on the dev PC (Cursor or Milan):
    python inspect_drafts.py "C:\\ASTRO\\python\\VYVAR\\Archive\\Drafts"

Prints, per draft: #FITS frames, filters, binning, plate scale (-> bin1 ~0.65 vs bin2 ~1.3),
image size, and median FWHM if a VYVAR proc/lightcurve CSV or a FWHM header keyword is present.
ASCII only. Depends only on astropy.io.fits + stdlib (already in the VYVAR env).
"""
from __future__ import annotations

import glob
import os
import statistics as st
import sys

try:
    from astropy.io import fits
except Exception:  # pragma: no cover  # noqa: BLE001
    print("astropy not importable in this env -- run inside the VYVAR venv")
    sys.exit(1)

FWHM_KEYS = ("VY_FWHM", "FWHM", "L1FWHM", "FWHMPX")
FILTER_KEYS = ("FILTER", "FILTER1", "INSTFILT")
BIN_KEYS = ("XBINNING", "YBINNING", "BINX", "BINY", "CCDXBIN", "CCDYBIN")
SCALE_KEYS = ("PXSCALE", "PIXSCALE", "SCALE", "SECPIX")


def _first(hdr, keys):
    for k in keys:
        if k in hdr:
            return hdr[k]
    return None


def _plate_scale_from_cd(hdr):
    # arcsec/px from CD/CDELT if WCS present
    try:
        import math

        if "CD1_1" in hdr:
            cd11, cd12 = float(hdr["CD1_1"]), float(hdr.get("CD1_2", 0.0))
            return math.hypot(cd11, cd12) * 3600.0
        if "CDELT1" in hdr:
            return abs(float(hdr["CDELT1"])) * 3600.0
    except Exception:  # noqa: BLE001
        pass
    return None


def summarize_draft(path: str) -> dict:
    fits_files = []
    for ext in ("*.fit", "*.fits", "*.fts"):
        fits_files += glob.glob(os.path.join(path, "**", ext), recursive=True)
    info = {
        "name": os.path.basename(path.rstrip("/\\")),
        "n_fits": len(fits_files),
        "filters": {},
        "binning": set(),
        "scales": [],
        "sizes": set(),
        "fwhm": [],
    }
    # sample up to 40 headers (fast)
    for f in fits_files[:40]:
        try:
            hdr = fits.getheader(f)
        except Exception:  # noqa: BLE001
            continue
        filt = _first(hdr, FILTER_KEYS)
        if filt is not None:
            key = str(filt).strip()
            info["filters"][key] = info["filters"].get(key, 0) + 1
        bx = _first(hdr, ("XBINNING", "BINX", "CCDXBIN"))
        if bx is not None:
            info["binning"].add(int(bx))
        sc = _first(hdr, SCALE_KEYS) or _plate_scale_from_cd(hdr)
        if sc:
            try:
                info["scales"].append(round(float(sc), 3))
            except Exception:  # noqa: BLE001
                pass
        nx, ny = hdr.get("NAXIS1"), hdr.get("NAXIS2")
        if nx and ny:
            info["sizes"].add(f"{nx}x{ny}")
        fw = _first(hdr, FWHM_KEYS)
        if fw is not None:
            try:
                info["fwhm"].append(float(fw))
            except Exception:  # noqa: BLE001
                pass
    # try a VYVAR proc/lightcurve CSV for FWHM if header had none
    if not info["fwhm"]:
        csv_paths = glob.glob(os.path.join(path, "**", "*proc*.csv"), recursive=True)[:3]
        csv_paths += glob.glob(os.path.join(path, "**", "lightcurve_*.csv"), recursive=True)[:3]
        for csv in csv_paths:
            try:
                import csv as _csv

                with open(csv, newline="") as fh:
                    rd = _csv.DictReader(fh)
                    col = next((c for c in (rd.fieldnames or []) if "fwhm" in c.lower()), None)
                    if col:
                        for row in rd:
                            try:
                                info["fwhm"].append(float(row[col]))
                            except Exception:  # noqa: BLE001
                                pass
            except Exception:  # noqa: BLE001
                pass
    return info


def main(root: str) -> None:
    drafts = [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]
    if not drafts:
        drafts = [root]  # root itself may be a single draft
    print(f"\nDrafts root: {root}   ({len(drafts)} subdir(s))\n")
    hdr = (
        f"{'draft':28} {'#fits':>5} {'bin':>4} {'scale(\"/px)':>11} {'medFWHM':>8}  "
        "filters / size"
    )
    print(hdr)
    print("-" * len(hdr))
    for d in drafts:
        i = summarize_draft(d)
        scale = f"{st.median(i['scales']):.2f}" if i["scales"] else "?"
        binv = "/".join(map(str, sorted(i["binning"]))) if i["binning"] else "?"
        medfw = f"{st.median(i['fwhm']):.2f}" if i["fwhm"] else "?"
        filt = ",".join(f"{k}:{v}" for k, v in i["filters"].items()) or "?"
        size = ",".join(sorted(i["sizes"])) or "?"
        # bin1 vs bin2 hint
        hint = ""
        if i["scales"]:
            m = st.median(i["scales"])
            if m <= 0.9:
                hint = "  <- FINE (~0.65, PSF-validate)"
            elif m <= 1.8:
                hint = "  <- coarse (~1.3, design/blends)"
        print(
            f"{i['name'][:28]:28} {i['n_fits']:>5} {binv:>4} {scale:>11} {medfw:>8}  "
            f"{filt} / {size}{hint}"
        )
    print("\nRule of thumb: scale <= ~0.9 \"/px (bin1) qualifies for PSF-vs-aperture validation;")
    print("~1.3 \"/px (bin2) is coarse but a dense cluster is still good for blend/crowding/asymmetry work.")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else r"C:\ASTRO\python\VYVAR\Archive\Drafts"
    main(root)
