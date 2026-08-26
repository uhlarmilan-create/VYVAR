# -*- coding: ascii -*-
"""APERTURE-01: measure f on the 516 scatter ladder (BO/FW/GH joint demeaned RMS).

Pre-registered: search the existing pixel ladder (1.5-12 px step 0.5) at
r = f x median QC FWHM (mode a geometry). No tuning after the minimum.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))

from aperture_policy import load_qc_fwhm_map, resolve_aperture_geometry  # noqa: E402
from aperture_scatter_select import (  # noqa: E402
    DEFAULT_R_MAX_PX,
    DEFAULT_R_MIN_PX,
    DEFAULT_R_STEP_PX,
    LadderSpec,
    differential_mag_series,
    measure_flux_ladder_frame,
)
from masterstar_gaia_accounting import _norm_cid  # noqa: E402

ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
OUT = ROOT / "dev" / "results" / "context" / "session_20260826_c6"
BO = "1498613634033133184"
FW = "1497343732462852864"
GH = "1498804639818507904"
GROUPS = {
    "BO": (
        BO,
        [
            "1497771992240531712",
            "1499200223486564608",
            "1497974027502858240",
            "1497368849430107904",
        ],
    ),
    "FW": (
        FW,
        [
            "1497442379271632384",
            "1499906247391001088",
            "1497674651102612992",
            "1498020894186918144",
            "1498812233320666368",
            "1497370563121917952",
            "1497313255374892800",
            "1500486102335278592",
        ],
    ),
    "GH": (
        GH,
        [
            "1497442379271632384",
            "1499906247391001088",
            "1497674651102612992",
            "1498020894186918144",
            "1498812233320666368",
            "1500486102335278592",
            "1496315070616056064",
            "1497196054307837696",
        ],
    ),
}
ANN_IN = 4.75
ANN_OUT = 9.0


def demeaned_rms_mmag(mag: np.ndarray) -> float:
    x = np.asarray(mag, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float("nan")
    d = x - float(np.median(x))
    return float(np.sqrt(np.mean(d * d)) * 1000.0)


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    qc_path = ERA04 / "calibrated" / "lights" / "qc_metrics.csv"
    qc_map, night = load_qc_fwhm_map(qc_path)
    ms = pd.read_csv(
        ERA04 / "platesolve" / SETUP / "masterstars_full_match.csv",
        dtype={"catalog_id": str},
        low_memory=False,
    )
    ms["catalog_id"] = ms["catalog_id"].map(lambda s: _norm_cid(s) or str(s).strip())
    xy = {}
    for _, row in ms.iterrows():
        cid = str(row["catalog_id"])
        try:
            xy[cid] = (float(row["x"]), float(row["y"]))
        except (TypeError, ValueError):
            continue
    want: list[str] = []
    for _name, (tid, comps) in GROUPS.items():
        want.append(tid)
        want.extend(comps)
    want = sorted(set(want))
    missing = [c for c in want if c not in xy]
    ids = [c for c in want if c in xy]
    xs = np.array([xy[c][0] for c in ids], dtype=np.float64)
    ys = np.array([xy[c][1] for c in ids], dtype=np.float64)

    spec = LadderSpec(
        r_min_px=DEFAULT_R_MIN_PX, r_max_px=DEFAULT_R_MAX_PX, r_step_px=DEFAULT_R_STEP_PX
    )
    radii = [float(r) for r in spec.radii_px()]
    lights = sorted((ERA04 / "detrended_aligned" / "lights" / SETUP).glob("*.fits"))
    lights = [p for p in lights if p.stem.upper() != "MASTERSTAR"]
    if night is None or not math_isfinite(night):
        night = 5.19465
    r_in_n, r_out_n = resolve_aperture_geometry(
        f=1.0, fwhm_px=float(night), annulus_inner_fwhm=ANN_IN, annulus_outer_fwhm=ANN_OUT
    )[1:]

    flux_by_r: dict[float, dict[str, list[float]]] = {
        r: {cid: [] for cid in ids} for r in radii
    }
    frames_used = 0
    for fp in lights:
        with fits.open(fp, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
        out, _sky = measure_flux_ladder_frame(
            data, xs, ys, radii, annulus_inner_px=r_in_n, annulus_outer_px=r_out_n
        )
        frames_used += 1
        for r, arr in out.items():
            for j, cid in enumerate(ids):
                flux_by_r[float(r)][cid].append(float(arr[j]))

    rows = []
    for r in radii:
        f = float(r) / float(night)
        rec = {"r_px": r, "f": round(f, 6), "night_fwhm_px": float(night)}
        rms_list = []
        for gname, (tid, comps) in GROUPS.items():
            tflux = np.asarray(flux_by_r[r][tid], dtype=np.float64)
            cflux = {
                c: np.asarray(flux_by_r[r][c], dtype=np.float64)
                for c in comps
                if c in flux_by_r[r]
            }
            dmag = differential_mag_series(tflux, cflux)
            rms = demeaned_rms_mmag(dmag)
            rec[f"rms_{gname}_mmag"] = None if not np.isfinite(rms) else round(rms, 4)
            if np.isfinite(rms):
                rms_list.append(rms)
        rec["joint_mean_rms_mmag"] = (
            round(float(np.mean(rms_list)), 4) if rms_list else None
        )
        rec["n_frames"] = frames_used
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a1_f_ladder.csv", index=False)
    finite = df.dropna(subset=["joint_mean_rms_mmag"])
    best = finite.loc[finite["joint_mean_rms_mmag"].idxmin()]
    summary = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "n_frames": frames_used,
        "n_stars": len(ids),
        "missing_xy": missing,
        "night_fwhm_px": float(night),
        "fwhm_authority": "qc_metrics.fwhm_px",
        "qc_csv": str(qc_path),
        "qc_n": len(qc_map),
        "ladder_r_px": radii,
        "annulus_inner_px": r_in_n,
        "annulus_outer_px": r_out_n,
        "best_r_px": float(best["r_px"]),
        "best_f": float(best["f"]),
        "best_joint_mean_rms_mmag": float(best["joint_mean_rms_mmag"]),
        "best_rms_BO_mmag": best["rms_BO_mmag"],
        "best_rms_FW_mmag": best["rms_FW_mmag"],
        "best_rms_GH_mmag": best["rms_GH_mmag"],
        "curve": df.to_dict(orient="records"),
    }
    (OUT / "a1_f_ladder.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="ascii"
    )
    print(
        "best f=%.4f r=%.3f joint=%.3f mmag BO=%s FW=%s GH=%s n_frames=%d elapsed=%.1fs"
        % (
            float(best["f"]),
            float(best["r_px"]),
            float(best["joint_mean_rms_mmag"]),
            best["rms_BO_mmag"],
            best["rms_FW_mmag"],
            best["rms_GH_mmag"],
            frames_used,
            time.perf_counter() - t0,
        )
    )
    return 0


def math_isfinite(v: float) -> bool:
    import math

    return math.isfinite(float(v))


if __name__ == "__main__":
    raise SystemExit(main())
