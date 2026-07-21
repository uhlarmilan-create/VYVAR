# -*- coding: ascii -*-
"""Extract OSC Bayer frames into L / G / B / R mono channels (2x2 superpixel).

AAVSO-style channel separation (no interpolated demosaic):
  L = (R + G1 + G2 + B) / 4   luminance superpixel (CV-equivalent for differential)
  G = (G1 + G2) / 2           TG ~ V band
  B, R                        native Bayer cells, 2x2 binned to mono grid

Usage:
  python dev/scripts/osc_extract_channels.py --input Archive/M71/Lights --output tmp/m71_extract/L/Lights
  python dev/scripts/osc_extract_channels.py --input Archive/M71 --output tmp/m71_extract/L --recursive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src_py") not in sys.path:
    sys.path.insert(0, str(ROOT / "src_py"))

from importer import path_suffix_is_fits  # noqa: E402

SUPPORTED_PATTERNS = {"RGGB", "GRBG"}
COPY_KEYS = (
    "DATE-OBS",
    "EXPTIME",
    "EXPOSURE",
    "GAIN",
    "EGAIN",
    "CCD-TEMP",
    "INSTRUME",
    "TELESCOP",
    "FOCALLEN",
    "FOCLEN",
    "OBJCTRA",
    "OBJCTDEC",
    "RA",
    "DEC",
    "IMAGETYP",
    "SITELAT",
    "SITELONG",
    "SITELON",
    "LAT-OBS",
    "LONG-OBS",
    "ALT-OBS",
    "XORGSUBF",
    "YORGSUBF",
)


def _bayer_cells(data: np.ndarray, pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return R, G1, G2, B arrays each (H/2, W/2)."""
    d = np.asarray(data, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {d.shape}")
    h, w = d.shape
    if h % 2 or w % 2:
        d = d[: h - (h % 2), : w - (w % 2)]
        h, w = d.shape
    tiles = {
        "R": d[0::2, 0::2],
        "G1": d[0::2, 1::2],
        "G2": d[1::2, 0::2],
        "B": d[1::2, 1::2],
    }
    if pattern == "RGGB":
        return tiles["R"], tiles["G1"], tiles["G2"], tiles["B"]
    if pattern == "GRBG":
        # [[G1, R], [B, G2]]
        return tiles["G1"], tiles["R"], tiles["B"], tiles["G2"]
    raise ValueError(f"Unsupported BAYERPAT: {pattern}")


def _derive_channels(r: np.ndarray, g1: np.ndarray, g2: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    g = (g1 + g2) * 0.5
    lum = (r + g1 + g2 + b) * 0.25
    return {"L": lum.astype(np.float32), "G": g.astype(np.float32), "B": b.astype(np.float32), "R": r.astype(np.float32)}


def _copy_header(src: fits.Header, *, channel: str, src_path: Path, pattern: str) -> fits.Header:
    hdr = fits.Header()
    for key in COPY_KEYS:
        if key in src:
            hdr[key] = src[key]
    # Pixel scale after 2x2 bin
    pix = float(src.get("XPIXSZ") or src.get("PIXSIZE") or 3.76)
    hdr["XBINNING"] = 2
    hdr["YBINNING"] = 2
    hdr["BINNING"] = 2
    hdr["XPIXSZ"] = pix * 2.0
    hdr["YPIXSZ"] = pix * 2.0
    hdr["FILTER"] = channel
    hdr["OSC-MODE"] = (channel, "OSC superpixel channel (no demosaic)")
    hdr["OSC-SRC"] = (str(src_path.name), f"Source Bayer frame; pattern={pattern}")
    if "BAYERPAT" in hdr:
        del hdr["BAYERPAT"]
    return hdr


def _output_name(src_path: Path, channel: str, hdr: fits.Header) -> str:
    """Avoid Dark_/Flat_ prefixes that trigger _looks_like_master in importer."""
    stem = src_path.stem
    imagetyp = str(hdr.get("IMAGETYP") or "").lower()
    if "dark" in imagetyp or "flat" in imagetyp:
        return f"osc_{stem}_{channel}.fits"
    return f"{stem}_{channel}.fits"


def extract_frame(src_path: Path, out_dir: Path, *, channels: tuple[str, ...] = ("L", "G", "B", "R")) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with fits.open(src_path, memmap=False) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
        pattern = str(hdr.get("BAYERPAT") or "RGGB").upper().replace(" ", "")[:4]
        if pattern not in SUPPORTED_PATTERNS:
            raise ValueError(f"{src_path}: BAYERPAT {pattern!r} not in {SUPPORTED_PATTERNS}")
        r, g1, g2, b = _bayer_cells(data, pattern)
        derived = _derive_channels(r, g1, g2, b)

    written: dict[str, Path] = {}
    for ch in channels:
        out_path = out_dir / _output_name(src_path, ch, hdr)
        if out_path.exists():
            written[ch] = out_path
            continue
        chdr = _copy_header(hdr, channel=ch, src_path=src_path, pattern=pattern)
        fits.writeto(out_path, derived[ch], chdr, overwrite=True)
        written[ch] = out_path
    return written


def _iter_inputs(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root] if path_suffix_is_fits(root) else []
    globber = root.rglob("*") if recursive else root.iterdir()
    out: list[Path] = []
    for fp in globber:
        if fp.is_file() and path_suffix_is_fits(fp):
            out.append(fp)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="OSC Bayer -> L/G/B/R superpixel extraction")
    ap.add_argument("--input", type=Path, required=True, help="Input FITS file or directory")
    ap.add_argument("--output", type=Path, required=True, help="Output directory for one channel OR channel root")
    ap.add_argument(
        "--channel",
        choices=("L", "G", "B", "R", "all"),
        default="all",
        help="Which channel(s) to write (default all)",
    )
    ap.add_argument("--recursive", action="store_true", help="Recurse input directory")
    ap.add_argument(
        "--layout",
        choices=("flat", "per_channel"),
        default="per_channel",
        help="flat: all in --output; per_channel: --output/L, /G, /B, /R subdirs",
    )
    args = ap.parse_args()

    inputs = _iter_inputs(args.input.resolve(), args.recursive)
    if not inputs:
        raise SystemExit(f"No FITS under {args.input}")

    chans: tuple[str, ...]
    if args.channel == "all":
        chans = ("L", "G", "B", "R")
    else:
        chans = (args.channel,)

    n_ok = 0
    for fp in inputs:
        if args.layout == "flat":
            extract_frame(fp, args.output.resolve(), channels=chans)
        else:
            for ch in chans:
                out_sub = args.output.resolve() / ch / fp.parent.name if args.recursive else args.output.resolve() / ch
                extract_frame(fp, out_sub, channels=(ch,))
        n_ok += 1
        if n_ok % 25 == 0:
            print(f"  extracted {n_ok}/{len(inputs)}...", flush=True)
    print(f"Done: {n_ok} source frame(s) -> channels {chans} under {args.output}")


if __name__ == "__main__":
    main()
