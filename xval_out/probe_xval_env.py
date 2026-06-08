#!/usr/bin/env python3
"""
VYVAR cross-validation — ENVIRONMENT + FITS PROBE  (read-only, zero side effects)

Purpose
-------
Before writing the real independent photutils cross-validation script, find out:
  1) which Python packages are installed on this PC,
  2) whether Gaia (astroquery) is reachable from here,
  3) what the FITS in detrended_aligned actually contain
     (dimensions, gain/read-noise/pixel headers, WCS presence),
  4) whether VYVAR's per-draft photometry outputs exist nearby
     (only to know if we *can* compare against VYVAR — not required as input).

This script ONLY reads. It writes nothing, deletes nothing, touches no network
except one optional TCP reachability check to the Gaia host.

Usage
-----
    python3 probe_xval_env.py /path/to/<draft>/detrended_aligned

If no path is given it tries a few common locations and recurses to find FITS.
Just paste the entire printed output back into the chat.
"""
from __future__ import annotations

import glob
import importlib
import os
import socket
import sys
from pathlib import Path

LINE = "=" * 72


def _ver(mod_name: str) -> str:
    try:
        m = importlib.import_module(mod_name)
        return str(getattr(m, "__version__", "installed (no __version__)"))
    except Exception as exc:  # noqa: BLE001
        return f"NOT AVAILABLE ({type(exc).__name__})"


def probe_packages() -> None:
    print(LINE)
    print("PYTHON :", sys.version.split()[0], "@", sys.executable)
    print(LINE)
    print("PACKAGES (cross-val relevant)")
    for mod in [
        "numpy", "pandas", "scipy", "matplotlib",
        "astropy", "photutils", "astroquery",
        "sep", "ccdproc", "skimage", "reproject",
    ]:
        print(f"  {mod:12s}: {_ver(mod)}")


def probe_network() -> None:
    print(LINE)
    print("NETWORK / GAIA REACHABILITY")
    host = "gea.esac.esa.int"  # Gaia archive (astroquery.gaia)
    try:
        socket.setdefaulttimeout(6)
        sock = socket.create_connection((host, 443), timeout=6)
        sock.close()
        print(f"  TCP 443 -> {host}: OK")
        print("  => Gaia crossmatch via astroquery should work (independent star list).")
    except Exception as exc:  # noqa: BLE001
        print(f"  TCP 443 -> {host}: FAIL ({exc})")
        print("  => No Gaia here; we'll detect stars locally (DAOStarFinder) instead.")


def find_fits(root: Path) -> list[Path]:
    pats = ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS", "*.FTS")
    found: list[Path] = []
    for p in pats:
        found += [Path(x) for x in glob.glob(str(root / "**" / p), recursive=True)]
    # de-dupe, stable order
    seen, out = set(), []
    for f in sorted(found):
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def resolve_root() -> Path | None:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser()
    # fallbacks — adjust if your layout differs
    for guess in [
        Path.cwd(),
        Path.home() / "Public" / "VYVAR",
        Path("/home/milan/Public/VYVAR"),
    ]:
        if guess.exists():
            # only accept if it actually contains FITS somewhere
            if find_fits(guess):
                print(f"[info] no path arg given; using detected root: {guess}")
                return guess
    return None


def probe_fits(root: Path) -> None:
    print(LINE)
    print(f"FITS INVENTORY under: {root}")
    fits = find_fits(root)
    print(f"  total FITS found (recursive): {len(fits)}")
    if not fits:
        print("  !! none found — pass the correct detrended_aligned path as argument.")
        return

    # show directory grouping (the layout matters for the real script)
    dirs: dict[str, int] = {}
    for f in fits:
        dirs[str(f.parent)] = dirs.get(str(f.parent), 0) + 1
    print("  by directory:")
    for d, n in sorted(dirs.items()):
        print(f"    {n:4d}  {d}")

    # inspect ONE sample header in detail
    sample = fits[0]
    print(LINE)
    print(f"SAMPLE HEADER: {sample}")
    try:
        from astropy.io import fits as pyfits  # noqa: WPS433
        from astropy.wcs import WCS  # noqa: WPS433

        with pyfits.open(sample, memmap=True) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data
            shape = None if data is None else getattr(data, "shape", None)
            dtype = None if data is None else getattr(data, "dtype", None)
            print(f"  data shape: {shape}   dtype: {dtype}")

            keys = [
                "NAXIS1", "NAXIS2", "DATE-OBS", "EXPTIME", "EXPOSURE",
                "GAIN", "EGAIN", "RDNOISE", "RDNOISE1", "RON",
                "XPIXSZ", "YPIXSZ", "XBINNING", "YBINNING",
                "FOCALLEN", "APTDIA", "INSTRUME", "TELESCOP",
                "FILTER", "OBJECT", "AIRMASS", "JD", "BJD",
                "SITELAT", "SITELONG", "SITEELEV",
            ]
            print("  selected header cards (missing = '-'):")
            for k in keys:
                print(f"    {k:10s}= {hdr.get(k, '-')}")

            # WCS check
            wcs_keys = [k for k in ("CTYPE1", "CRVAL1", "CD1_1", "CDELT1") if k in hdr]
            print(f"  WCS cards present: {wcs_keys if wcs_keys else 'NONE'}")
            if "CTYPE1" in hdr:
                try:
                    w = WCS(hdr)
                    ny, nx = (data.shape if data is not None else (hdr.get("NAXIS2", 0),
                                                                   hdr.get("NAXIS1", 0)))
                    cx, cy = nx / 2.0, ny / 2.0
                    sky = w.pixel_to_world(cx, cy)
                    print(f"  WCS solves: center pixel ({cx:.0f},{cy:.0f}) -> "
                          f"RA={sky.ra.deg:.5f} deg, Dec={sky.dec.deg:.5f} deg")
                    # rough plate scale from CD/CDELT
                    try:
                        import numpy as np  # noqa: WPS433
                        scale = np.sqrt(np.abs(np.linalg.det(w.pixel_scale_matrix))) * 3600.0
                        print(f"  plate scale (from WCS): {scale:.3f} arcsec/px")
                    except Exception:  # noqa: BLE001
                        pass
                except Exception as exc:  # noqa: BLE001
                    print(f"  WCS present but failed to instantiate: {exc}")
            else:
                print("  => No WCS in header. Independent Gaia crossmatch needs a plate "
                      "solution; tell me if these are solved elsewhere.")
    except Exception as exc:  # noqa: BLE001
        print(f"  !! could not read FITS via astropy: {exc}")


def probe_vyvar_outputs(root: Path) -> None:
    """Only reports presence — so we know if we CAN compare to VYVAR's own LC."""
    print(LINE)
    print("VYVAR PER-DRAFT OUTPUTS (presence only — for comparison column)")
    # search the draft tree (parent of detrended_aligned, plus root itself)
    search_roots = {root, root.parent, root.parent.parent}
    wanted = [
        "photometry_summary.csv",
        "active_targets.csv",
        "comparison_stars_per_target.csv",
    ]
    for name in wanted:
        hits: list[str] = []
        for sr in search_roots:
            try:
                hits += glob.glob(str(sr / "**" / name), recursive=True)
            except Exception:  # noqa: BLE001
                pass
        hits = sorted(set(hits))
        if hits:
            print(f"  FOUND  {name}: {hits[0]}" + (f"  (+{len(hits)-1} more)"
                                                   if len(hits) > 1 else ""))
        else:
            print(f"  absent {name}")


def main() -> int:
    probe_packages()
    probe_network()
    root = resolve_root()
    if root is None:
        print(LINE)
        print("NO FITS ROOT FOUND. Re-run with the path, e.g.:")
        print("  python3 probe_xval_env.py /home/milan/Public/VYVAR/draft_000XYZ/detrended_aligned")
        return 1
    probe_fits(root)
    probe_vyvar_outputs(root)
    print(LINE)
    print("DONE — paste this whole output back into the chat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
