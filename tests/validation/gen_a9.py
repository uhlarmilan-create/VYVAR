"""A9 generator: blend grid cells + isolated controls with truth sidecars."""
from __future__ import annotations

import json
from pathlib import Path

from tests.validation.a9_core import (
    A9_CONTEXTS,
    A9_RNG_SEED,
    DELTA_MAGS,
    N_FRAMES,
    SEPARATIONS_FWHM,
    TARGET_MAG,
    ZP,
    build_blend_frame,
    measure_isolated_bias_pct,
    _rng_for_cell,
)

DATA_DIR = Path(__file__).resolve().parent / "data" / "tier_a9"


def write_a9_truth(out_dir: Path | None = None) -> Path:
    """Write a9_truth.json describing the full blend grid (no FITS blobs on disk)."""
    out_dir = Path(out_dir or DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth: dict = {
        "tier": "A9",
        "rng_seed": A9_RNG_SEED,
        "target_mag": TARGET_MAG,
        "zp": ZP,
        "n_frames": N_FRAMES,
        "separations_fwhm": list(SEPARATIONS_FWHM),
        "delta_mags": list(DELTA_MAGS),
        "contexts": {},
        "cells": [],
    }
    for ctx_name, ctx in A9_CONTEXTS.items():
        tflux = ctx.target_flux_adu()
        iso_bias = measure_isolated_bias_pct(ctx)
        r_ap, r_in, r_out = ctx.radii_px()
        truth["contexts"][ctx_name] = {
            "fwhm_px": ctx.fwhm_px,
            "plate_scale_arcsec": ctx.plate_scale_arcsec,
            "target_flux_adu": tflux,
            "isolated_bias_pct": iso_bias,
            "aperture_r_px": r_ap,
            "annulus_r_in_px": r_in,
            "annulus_r_out_px": r_out,
        }
        for sep in SEPARATIONS_FWHM:
            for dM in DELTA_MAGS:
                nflux = tflux * (10.0 ** (-0.4 * dM))
                truth["cells"].append(
                    {
                        "context": ctx_name,
                        "sep_fwhm": sep,
                        "delta_mag": dM,
                        "target_flux_adu": tflux,
                        "neighbour_flux_adu": nflux if sep > 0 else 0.0,
                        "sep_px": sep * ctx.fwhm_px,
                    }
                )
    # Isolated control reference (one frame checksum per context)
    for ctx_name, ctx in A9_CONTEXTS.items():
        rng = _rng_for_cell(ctx_name, -1.0, 0)
        frame = build_blend_frame(
            ctx,
            target_flux=ctx.target_flux_adu(),
            neighbour_flux=0.0,
            sep_fwhm=0.0,
            rng=rng,
            jitter=False,
        )
        truth["contexts"][ctx_name]["isolated_frame_sum"] = float(frame.sum())

    path = out_dir / "a9_truth.json"
    with open(path, "w", encoding="ascii") as f:
        json.dump(truth, f, indent=2)
    return path


if __name__ == "__main__":
    p = write_a9_truth()
    print(f"wrote {p}")
