"""V3e: ePSF FWHM QC estimator vs known analytic FWHM (EPSF-1).

Inject synthetic isolated Moffat stars at known FWHM, build ePSF via production
``_epsf_build_imagepsf_from_stars``, compare OLD (legacy half-max) vs NEW
(azimuthally-binned radial profile) ``epsf_fwhm_native``. ASCII, deterministic.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import extract_stars

from psf_photometry import (
    _EPSF_FWHM_RATIO_WARN_HI,
    _EPSF_FWHM_RATIO_WARN_LO,
    _epsf_build_imagepsf_from_stars,
    _epsf_fwhm_native_from_profile,
    _epsf_fwhm_native_legacy_px,
)
from tests.validation.gen_frame import moffat_stamp
from tests.validation.score import RNG_SEEDS

V3E_RNG_SEED = RNG_SEEDS.get("v3e_epsf", 370)
FRAME_N = 400
N_STARS = 16
BUILD_MAG = 11.0
ZP = 25.0
SKY_ADU = 300.0
GAIN = 1.5
READ_NOISE = 9.0
MOFFAT_BETA = 2.5
PASS_RATIO_LO = 0.85
PASS_RATIO_HI = 1.15


@dataclass(frozen=True)
class V3eCase:
    name: str
    fwhm_px: float
    osamp: int = 2
    beta: float = MOFFAT_BETA


CASES: tuple[V3eCase, ...] = (
    V3eCase("coarse_moffat", 2.7),
    V3eCase("fine_moffat", 5.4),
    V3eCase("v3d_moffat", 6.0203),
)


def mag_to_flux(mag: float, zp: float = ZP) -> float:
    return float(10.0 ** (-0.4 * (float(mag) - float(zp))))


def _star_positions(cutout: int, n: int = N_STARS) -> list[tuple[float, float]]:
    margin = cutout // 2 + 4
    span = FRAME_N - 2 * margin
    n_side = int(math.ceil(math.sqrt(n)))
    step = span / max(n_side - 1, 1)
    pts: list[tuple[float, float]] = []
    for iy in range(n_side):
        for ix in range(n_side):
            if len(pts) >= n:
                break
            pts.append((float(margin + ix * step), float(margin + iy * step)))
    return pts


def build_training_frame(case: V3eCase, rng: np.random.Generator) -> tuple[np.ndarray, Table]:
    flux = mag_to_flux(BUILD_MAG)
    img = np.full((FRAME_N, FRAME_N), SKY_ADU, dtype=np.float64)
    cutout = int(case.fwhm_px * 5) | 1
    xs, ys, names = [], [], []
    for i, (x, y) in enumerate(_star_positions(cutout)):
        img += moffat_stamp(y, x, flux, case.fwhm_px, case.beta, ny=FRAME_N, nx=FRAME_N)
        xs.append(x)
        ys.append(y)
        names.append(f"epsf_{i:02d}")
    el = np.clip(img * GAIN, 0.0, None)
    img = rng.poisson(el).astype(np.float64) / GAIN
    img += rng.normal(0.0, READ_NOISE / GAIN, size=img.shape)
    cat = Table()
    cat["x"] = xs
    cat["y"] = ys
    cat["name"] = names
    return img, cat


def measure_case(case: V3eCase, rng: np.random.Generator) -> dict[str, Any]:
    frame, cat = build_training_frame(case, rng)
    cutout = int(case.fwhm_px * 5) | 1
    epsf_stars = extract_stars(NDData(frame.astype(np.float64)), cat, size=cutout)
    built = _epsf_build_imagepsf_from_stars(
        epsf_stars, osamp=case.osamp, fwhm_px=case.fwhm_px, cutout_size=cutout
    )
    epsf_data = built["arr"]
    qc = built["qc"]
    native_new = float(qc.get("epsf_fwhm_native_px") or float("nan"))
    ratio_new = float(qc.get("epsf_vs_input_fwhm_ratio") or float("nan"))
    native_old = _epsf_fwhm_native_legacy_px(epsf_data, osamp=case.osamp)
    ratio_old = (
        native_old / float(case.fwhm_px) if math.isfinite(native_old) and case.fwhm_px > 0 else float("nan")
    )
    # Sanity: harness profile matches production helper.
    native_profile = _epsf_fwhm_native_from_profile(epsf_data, osamp=case.osamp)
    pass_new = math.isfinite(ratio_new) and PASS_RATIO_LO <= ratio_new <= PASS_RATIO_HI
    return {
        "name": case.name,
        "fwhm_input_px": float(case.fwhm_px),
        "osamp": int(case.osamp),
        "n_stars": len(epsf_stars) if epsf_stars is not None else 0,
        "native_old_px": native_old,
        "ratio_old": ratio_old,
        "native_new_px": native_new,
        "ratio_new": ratio_new,
        "native_profile_px": native_profile,
        "pass_new": pass_new,
    }


def run_v3e_epsf_fwhm(*, rng_seed: int = V3E_RNG_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(int(rng_seed))
    rows = [measure_case(c, rng) for c in CASES]
    all_pass = all(r["pass_new"] for r in rows)
    ratios = [r["ratio_new"] for r in rows if math.isfinite(r["ratio_new"])]
    ratio_min = min(ratios) if ratios else float("nan")
    ratio_max = max(ratios) if ratios else float("nan")
    return {
        "status": "PASS" if all_pass else "FAIL",
        "rng_seed": int(rng_seed),
        "pass_criterion": [PASS_RATIO_LO, PASS_RATIO_HI],
        "warn_band": [_EPSF_FWHM_RATIO_WARN_LO, _EPSF_FWHM_RATIO_WARN_HI],
        "ratio_span_new": [ratio_min, ratio_max],
        "cases": rows,
    }


def write_v3e_report(out_dir: Path, result: dict[str, Any]) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "v3e_epsf_fwhm.json"
    mp = out_dir / "v3e_epsf_fwhm.md"
    jp.write_text(json.dumps(result, indent=2), encoding="ascii")
    span = result.get("ratio_span_new") or [float("nan"), float("nan")]
    lo, hi = result["pass_criterion"]
    wlo, whi = result["warn_band"]
    lines = [
        "# V3e ePSF FWHM QC estimator (EPSF-1)",
        "",
        f"Status: **{result['status']}**",
        f"RNG seed: {result['rng_seed']}",
        "",
        f"PASS criterion (new estimator): ratio in [{lo}, {hi}]",
        f"QC warning band (production): [{wlo}, {whi}]",
        f"Measured new ratio span: [{span[0]:.4f}, {span[1]:.4f}]",
        "",
        "| case | FWHM in | native OLD | ratio OLD | native NEW | ratio NEW | PASS |",
        "|------|---------|------------|-----------|------------|-----------|------|",
    ]
    for r in result["cases"]:
        lines.append(
            f"| {r['name']} | {r['fwhm_input_px']:.3f} | "
            f"{r['native_old_px']:.4f} | {r['ratio_old']:.4f} | "
            f"{r['native_new_px']:.4f} | {r['ratio_new']:.4f} | "
            f"{'PASS' if r['pass_new'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "OLD estimator: first radius-sorted pixel below 0.5*peak (biased low on coarse grids).",
            "NEW estimator: azimuthally-binned radial profile mean + linear 0.5 crossing.",
        ]
    )
    mp.write_text("\n".join(lines) + "\n", encoding="ascii")
    return jp, mp
