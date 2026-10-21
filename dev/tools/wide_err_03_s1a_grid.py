#!/usr/bin/env python3
"""WIDE-ERR-03 Stage 1a: grid-of-4 proof on raw vs calibrated/resampled pixels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
OUT = ROOT / "dev" / "results" / "WIDE_ERR_03_S1a_grid.json"
RUN_SHA = "da9cce4a5edd1392b8ba842d3c8488589b9d0ac9"


def mod4_hist(data: np.ndarray) -> dict:
    x = np.asarray(data, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    # Integer-like ADU for residue; float calibrated may not be exact ints
    xi = np.rint(x).astype(np.int64)
    mods = xi % 4
    counts = {str(i): int(np.sum(mods == i)) for i in range(4)}
    n = int(mods.size)
    fracs = {k: (counts[k] / n if n else float("nan")) for k in counts}
    # Dominant residue concentration
    dom = max(counts, key=counts.get)
    return {
        "n_pixels": n,
        "counts_mod4": counts,
        "frac_mod4": fracs,
        "dominant_residue": int(dom),
        "dominant_frac": fracs[dom],
        "min": float(np.min(x)) if n else float("nan"),
        "max": float(np.max(x)) if n else float("nan"),
        "median": float(np.median(x)) if n else float("nan"),
        "frac_exact_int": float(np.mean(np.abs(x - np.rint(x)) < 1e-6)) if n else float("nan"),
    }


def load_frame(path: Path) -> tuple[np.ndarray, dict]:
    with fits.open(path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data)
        hdr = dict(hdul[0].header)
    meta = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "BITPIX": hdr.get("BITPIX"),
        "XBINNING": hdr.get("XBINNING"),
        "YBINNING": hdr.get("YBINNING"),
        "GAIN": hdr.get("GAIN"),
        "BZERO": hdr.get("BZERO"),
        "BSCALE": hdr.get("BSCALE"),
        "VY_CALSTAGE": hdr.get("VY_CALSTAGE"),
    }
    return data, meta


def main() -> None:
    raw_cands = sorted((DRAFT / "Raw" / "lights" / SETUP).glob("*.fits"))
    if not raw_cands:
        raw_cands = sorted((DRAFT / "Raw").rglob("*.fits"))
    cal_cands = sorted((DRAFT / "calibrated" / "lights" / SETUP).glob("*.fits"))
    det_cands = sorted((DRAFT / "detrended_aligned" / "lights" / SETUP).glob("*.fits"))

    if not raw_cands:
        raise SystemExit("STOP: no raw light FITS found")
    if not cal_cands and not det_cands:
        raise SystemExit("STOP: no calibrated/detrended FITS found")

    raw_path = raw_cands[0]
    # Prefer same basename if possible
    stem = raw_path.stem
    cal_path = next((p for p in cal_cands if p.stem == stem), cal_cands[0] if cal_cands else None)
    det_path = next((p for p in det_cands if p.stem == stem), det_cands[0] if det_cands else None)

    raw, raw_meta = load_frame(raw_path)
    frames = {"raw": {"meta": raw_meta, "mod4": mod4_hist(raw)}}

    if cal_path is not None:
        cal, cal_meta = load_frame(cal_path)
        frames["calibrated"] = {"meta": cal_meta, "mod4": mod4_hist(cal)}
    if det_path is not None:
        det, det_meta = load_frame(det_path)
        frames["detrended_aligned"] = {"meta": det_meta, "mod4": mod4_hist(det)}

    raw_dom_frac = frames["raw"]["mod4"]["dominant_frac"]
    # Gate: raw must be concentrated on one residue (expect >~0.9 on grid of 4)
    s1a_stop = raw_dom_frac < 0.85
    note = (
        f"Raw dominant residue {frames['raw']['mod4']['dominant_residue']} "
        f"holds {raw_dom_frac:.4f} of pixels."
    )
    if s1a_stop:
        note += " STOP: raw NOT concentrated on a grid of 4."
    else:
        note += " PASS: raw on grid of 4 (14-bit samples in 16-bit container, stride 4)."

    # What calibration does
    cal_note = ""
    if "calibrated" in frames:
        cf = frames["calibrated"]["mod4"]["frac_mod4"]
        cal_note = (
            "Calibrated (bias/dark/flat): residues redistribute "
            f"(fracs {cf}); float/division breaks exact integer grid."
        )
    if "detrended_aligned" in frames:
        df = frames["detrended_aligned"]["mod4"]["frac_mod4"]
        cal_note += (
            f" Detrended+aligned/resampled further mixes residues (fracs {df})."
        )

    payload = {
        "task": "WIDE-ERR-03 Stage S1a",
        "run_sha": RUN_SHA,
        "draft_id": 515,
        "setup": SETUP,
        "domain": "pixel ADU values; residue = rint(ADU) mod 4",
        "frames": frames,
        "s1a_stop": s1a_stop,
        "s1a_note": note,
        "calibration_effect_note": cal_note,
        "container_scale_hypothesis": {
            "native_bit_depth": 14,
            "container_bit_depth": 16,
            "expected_adu_stride": 4,
            "g_native_over_g_container": 4.0,
            "db_gain_native_e_per_adu": 3.17,
            "implied_g_container_e_per_adu": 3.17 / 4.0,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT)
    print(note)
    print(cal_note)
    if s1a_stop:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
