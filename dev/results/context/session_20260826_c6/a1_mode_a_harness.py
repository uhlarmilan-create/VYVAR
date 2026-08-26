# -*- coding: ascii -*-
"""Mode (a) harness: f_fixed_night, same estimator as a1_mode_b_harness.py."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.stats import spearmanr

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))
from aperture_policy import (  # noqa: E402
    load_qc_fwhm_map,
    resolve_aperture_geometry,
    resolve_frame_fwhm_px,
)
from aperture_scatter_select import differential_mag_series  # noqa: E402
from masterstar_gaia_accounting import _norm_cid  # noqa: E402
from photometry_core import _aperture_flux_sky_batch  # noqa: E402

CAND1 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_candidate1_20260826"
SETUP = "NoFilter_60_2"
OUT = ROOT / "dev" / "results" / "context" / "session_20260826_c6"
F = 0.385228
ANN_IN = 4.75
ANN_OUT = 9.0
GROUPS = {
    "BO": (
        "1498613634033133184",
        [
            "1497771992240531712",
            "1499200223486564608",
            "1497974027502858240",
            "1497368849430107904",
        ],
    ),
    "FW": (
        "1497343732462852864",
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
        "1498804639818507904",
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


def demeaned_rms_mmag(mag: np.ndarray) -> float:
    x = np.asarray(mag, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float("nan")
    d = x - float(np.median(x))
    return float(np.sqrt(np.mean(d * d)) * 1000.0)


def main() -> int:
    t0 = time.perf_counter()
    qc_map, night = load_qc_fwhm_map(CAND1 / "calibrated" / "lights" / "qc_metrics.csv")
    night_f = float(night) if night is not None else 5.1917332681208865
    r_ap, r_in, r_out = resolve_aperture_geometry(
        f=F, fwhm_px=night_f, annulus_inner_fwhm=ANN_IN, annulus_outer_fwhm=ANN_OUT
    )
    ms = pd.read_csv(
        CAND1 / "platesolve" / SETUP / "masterstars_full_match.csv",
        dtype={"catalog_id": str},
        low_memory=False,
    )
    ms["catalog_id"] = ms["catalog_id"].map(lambda s: _norm_cid(s) or str(s).strip())
    xy = {}
    for _, row in ms.iterrows():
        try:
            xy[str(row["catalog_id"])] = (float(row["x"]), float(row["y"]))
        except (TypeError, ValueError):
            continue
    want = []
    for tid, comps in GROUPS.values():
        want.append(tid)
        want.extend(comps)
    ids = [c for c in sorted(set(want)) if c in xy]
    xs = np.array([xy[c][0] for c in ids], dtype=np.float64)
    ys = np.array([xy[c][1] for c in ids], dtype=np.float64)
    lights = sorted(
        p
        for p in (CAND1 / "detrended_aligned" / "lights" / SETUP).glob("*.fits")
        if p.stem.upper() != "MASTERSTAR"
    )
    flux = {cid: [] for cid in ids}
    fwhm_frames = []
    for fp in lights:
        with fits.open(fp, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
            hdr = hdul[0].header
        fw = resolve_frame_fwhm_px(
            hdr=hdr, frame_name=fp.name, qc_fwhm_by_name=qc_map, fwhm_night_median_px=night_f
        )
        if fw is None:
            fw = night_f
        pos = np.column_stack([xs, ys])
        fl, _sky = _aperture_flux_sky_batch(data, pos, r_ap, r_in, r_out)
        for j, cid in enumerate(ids):
            flux[cid].append(float(fl[j]))
        fwhm_frames.append(float(fw))
    fwhm_arr = np.asarray(fwhm_frames, dtype=np.float64)
    stars = {}
    for name, (tid, comps) in GROUPS.items():
        tflux = np.asarray(flux[tid], dtype=np.float64)
        cflux = {c: np.asarray(flux[c], dtype=np.float64) for c in comps if c in flux}
        dmag = differential_mag_series(tflux, cflux)
        rms = demeaned_rms_mmag(dmag)
        d0 = dmag - float(np.nanmedian(dmag))
        ok = np.isfinite(d0) & np.isfinite(fwhm_arr)
        rho = float("nan")
        if int(ok.sum()) >= 8:
            rho = float(spearmanr(d0[ok], fwhm_arr[ok]).statistic)
        stars[name] = {
            "rms_mmag": None if not np.isfinite(rms) else round(float(rms), 4),
            "spearman_dmag_vs_fwhm": None if not np.isfinite(rho) else round(rho, 4),
        }
    out = {
        "mode": "f_fixed_night",
        "f": F,
        "n_frames": len(lights),
        "r_ap": round(float(r_ap), 4),
        "night_fwhm_px": round(night_f, 6),
        "stars": stars,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    (OUT / "a1_mode_a_harness.json").write_text(json.dumps(out, indent=2), encoding="ascii")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
