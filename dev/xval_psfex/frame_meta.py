# -*- coding: ascii -*-
"""Per-frame SExtractor meta for the A2 kit (no VYVAR src_py import).

Refute notes (era04 snapshot + live 516 Light_001, 2026-09-08):
- Files are BO_CVn_Light_NNN.fits, not Light_*.fits.
- No CD matrix; WCS is PC + CDELT=1 (SCALE keyword 9.55 != WCS 9.77).
- Header GAIN is 0.0 (invalid). Fallback is A1 g_pt=0.637067.
- SATURATE/MAXLIN absent; fallback 60000 as specified.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

GAIN_GPT = 0.637067
SATUR_FALLBACK = 60000.0


def light_key(name: str) -> str:
    m = re.search(r"Light_\d+", str(name))
    return m.group(0) if m else ""


def plate_scale_arcsec(header) -> tuple[float, str]:
    """WCS pixel scale in arcsec/px. Cite VYVAR unit_resolver.py:34-58 (same math)."""
    try:
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales

        w = WCS(header)
        if getattr(w, "has_celestial", False):
            sc = proj_plane_pixel_scales(w)
            ps = float(abs(float(sc[0])) + abs(float(sc[1]))) * 0.5 * 3600.0
            if math.isfinite(ps) and ps > 0.05:
                return ps, "wcs_pc"
    except Exception as exc:  # noqa: BLE001
        last = f"{type(exc).__name__}:{exc}"
    else:
        last = "wcs_non_celestial_or_tiny"
    # Last-resort PC * CDELT * 3600 (this snapshot: CDELT=1, PC in degrees).
    try:
        pc11 = float(header.get("PC1_1") or 0.0)
        cdelt1 = float(header.get("CDELT1") or 1.0)
        ps = abs(pc11 * cdelt1) * 3600.0
        if math.isfinite(ps) and ps > 0.05:
            return ps, "pc1_1_cdelt"
    except (TypeError, ValueError):
        pass
    raise SystemExit(f"FAIL plate_scale: no usable WCS ({last})")


def resolve_gain(header) -> tuple[float, str]:
    raw = header.get("GAIN")
    try:
        g = float(raw)
    except (TypeError, ValueError):
        g = float("nan")
    if math.isfinite(g) and g > 0:
        if abs(g - GAIN_GPT) > 0.01:
            return g, f"header_GAIN_DIVERGES_from_g_pt_{GAIN_GPT:g}"
        return g, "header_GAIN"
    return GAIN_GPT, "g_pt_fallback (header GAIN missing or <=0)"


def resolve_satur(header) -> tuple[float, str]:
    for key in ("SATURATE", "MAXLIN"):
        raw = header.get(key)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            return v, key
    return SATUR_FALLBACK, "fallback_60000"


def load_qc_fwhm(qc_path: Path) -> dict[str, float]:
    import pandas as pd

    out: dict[str, float] = {}
    if not qc_path.is_file():
        return out
    df = pd.read_csv(qc_path)
    col = "src" if "src" in df.columns else df.columns[0]
    for _, row in df.iterrows():
        key = light_key(str(row.get(col, "")))
        fw = float(pd.to_numeric(row.get("fwhm_px"), errors="coerce"))
        if key and math.isfinite(fw) and fw > 0:
            out[key] = fw
    return out


def meta_for_frame(fits_path: Path, qc_map: dict[str, float]) -> dict:
    from astropy.io import fits

    hdr = fits.getheader(fits_path)
    scale, scale_src = plate_scale_arcsec(hdr)
    gain, gain_src = resolve_gain(hdr)
    satur, satur_src = resolve_satur(hdr)
    key = light_key(fits_path.name)
    fwhm_px = qc_map.get(key)
    if fwhm_px is None:
        raise SystemExit(f"FAIL qc fwhm_px missing for {fits_path.name} key={key}")
    seeing = float(fwhm_px) * float(scale)
    return {
        "fits": fits_path.name,
        "light": key,
        "plate_scale_arcsec_per_px": scale,
        "plate_scale_source": scale_src,
        "fwhm_px": float(fwhm_px),
        "seeing_fwhm_arcsec": seeing,
        "gain": gain,
        "gain_source": gain_src,
        "satur_level": satur,
        "satur_source": satur_src,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A2 per-frame SExtractor meta")
    ap.add_argument("--fits", required=True)
    ap.add_argument("--qc", required=True)
    args = ap.parse_args(argv)
    qc_map = load_qc_fwhm(Path(args.qc))
    print(json.dumps(meta_for_frame(Path(args.fits), qc_map), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
