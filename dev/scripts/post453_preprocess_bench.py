#!/usr/bin/env python3
"""POST-453 Part 1: preprocess profile + byte-identity check vs draft 452."""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))

REF_DRAFT = REPO / "Archive" / "Drafts" / "draft_000452"
REF_LIGHT = REF_DRAFT / "calibrated" / "lights" / "NoFilter_60_2" / "BO_CVn_Light_001.fits"
OUT_DIR = REPO / "tmp" / "post453_preprocess_bench"
CTX = REPO / "dev" / "results" / "context" / "session_20260727_post453"


def profile_one_frame() -> dict[str, float]:
    from astropy.io import fits
    from pipeline import _fit_subtract_preprocess_sky_surface

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work = OUT_DIR / "profile_work.fits"
    shutil.copy2(REF_LIGHT, work)

    times: dict[str, float] = {}

    t0 = time.perf_counter()
    with fits.open(work, mode="readonly", memmap=True) as hdul:
        data = hdul[0].data.astype(np.float32, copy=True)
        hdr = hdul[0].header
    times["FITS read"] = time.perf_counter() - t0
    _ = hdr

    t0 = time.perf_counter()
    out, stats = _fit_subtract_preprocess_sky_surface(data, order=2)
    times["mask+fit+eval (combined)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with fits.open(work, mode="update") as hdul:
        hdul[0].data = out
        for k, v in stats.items():
            if isinstance(v, (int, float, str, bool)):
                hdul[0].header[f"VYP_{k}"] = v
    times["FITS write-back"] = time.perf_counter() - t0

    return times


def byte_compare(n: int = 10) -> dict[str, float]:
    from astropy.io import fits
    from config import AppConfig
    from pipeline import _qc_enrich_calibrated_in_place

    lights_root = REF_DRAFT / "calibrated" / "lights" / "NoFilter_60_2"
    if not lights_root.is_dir():
        raise FileNotFoundError(f"Missing lights directory: {lights_root}")
    lights = sorted(lights_root.glob("BO_CVn_Light_*.fits"))
    sample = lights[:n]
    if not sample:
        raise RuntimeError(
            f"No BO_CVn_Light_*.fits frames found under {lights_root}; "
            "cannot run byte-compare bench (zero frames)."
        )

    bench_root = OUT_DIR / "bench_lights"
    if bench_root.exists():
        shutil.rmtree(bench_root)
    bench_root.mkdir(parents=True)
    for fp in sample:
        shutil.copy2(fp, bench_root / fp.name)

    cfg = AppConfig()
    t0 = time.perf_counter()
    _qc_enrich_calibrated_in_place(str(bench_root), app_config=cfg)
    elapsed = time.perf_counter() - t0

    max_diff = 0.0
    for fp in sample:
        ref = fp
        new = bench_root / fp.name
        with fits.open(ref, memmap=True) as a, fits.open(new, memmap=True) as b:
            d = np.asarray(a[0].data, dtype=np.float64) - np.asarray(b[0].data, dtype=np.float64)
            finite = np.isfinite(d)
            if finite.any():
                max_diff = max(max_diff, float(np.max(np.abs(d[finite]))))

    return {
        "n_frames": float(len(sample)),
        "total_s": elapsed,
        "per_frame_s": elapsed / max(1, len(sample)),
        "max_abs_diff": max_diff,
    }


def main() -> None:
    if not REF_LIGHT.is_file():
        print("MISSING ref frame:", REF_LIGHT)
        sys.exit(1)

    prof = profile_one_frame()
    print("=== Part 1.1 profile (one frame) ===")
    for k, v in prof.items():
        print(f"{k:30s} {v:8.3f}s")

    cmp = byte_compare(10)
    print("\n=== Part 1.2 byte compare (10 frames) ===")
    for k, v in cmp.items():
        print(f"{k:30s} {v}")

    lines = [
        "step,seconds",
        *[f"{k},{v:.6f}" for k, v in prof.items()],
        "",
        "metric,value",
        *[f"{k},{v}" for k, v in cmp.items()],
    ]
    out_csv = CTX / "preprocess_profile.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("\n".join(lines) + "\n", encoding="ascii")
    print("\nWrote", out_csv)


if __name__ == "__main__":
    main()
