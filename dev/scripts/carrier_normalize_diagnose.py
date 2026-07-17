#!/usr/bin/env python3
"""Diagnose carrier_matches_normalize residual (draft_426 regen verify)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from scripts.bingain_err_decompose import _gain_from_lights  # noqa: E402
from scripts.sigma_sem_cause import _photon_err_mag_per_frame, extract_production_trace  # noqa: E402
from scripts.sparse_comp_diag import V0611_CID  # noqa: E402
from mag_constants import MAG_ERR_SCALE  # noqa: E402

DRAFT = 426
SETUPS = ("i_70_4", "r_60_4")
TOL = 1e-5


def diagnose(archive: Path, setup: str, cfg: AppConfig) -> dict:
    phot = archive / "Drafts" / f"draft_{DRAFT:06d}" / "platesolve" / setup / "photometry"
    lights = archive / "Drafts" / f"draft_{DRAFT:06d}" / "detrended_aligned" / "lights" / setup
    lc_path = phot / "lightcurves" / f"lightcurve_{V0611_CID}.csv"
    lc = pd.read_csv(lc_path, low_memory=False)
    gain = _gain_from_lights(lights, float(cfg.gain))
    rn = float(cfg.read_noise)
    trace = extract_production_trace(
        phot_dir=phot, setup=setup, target_cid=V0611_CID, cfg=cfg, gain=gain, read_noise=rn,
    )
    from check_star_kmag import resolve_proc_csv_dir  # noqa: PLC0415

    proc_dir = resolve_proc_csv_dir(phot, setup)
    phot_mag = _photon_err_mag_per_frame(lc, proc_dir, V0611_CID, gain=gain, read_noise=rn)
    sf = lc["source_file"].astype(str).tolist()
    ens = np.array([trace.get("scatter_by_file", {}).get(s.strip(), float("nan")) for s in sf], dtype=float)
    err_rel = lc["err"].to_numpy(dtype=float)
    err_round = np.round(err_rel, 6)
    phot_rel = phot_mag / MAG_ERR_SCALE
    implied_round = np.sqrt(np.maximum(0.0, err_round * err_round - phot_rel * phot_rel)) * MAG_ERR_SCALE
    implied_raw = np.sqrt(np.maximum(0.0, err_rel * err_rel - phot_rel * phot_rel)) * MAG_ERR_SCALE
    diff = np.abs(implied_raw - ens)
    ok = np.isfinite(implied_raw) & np.isfinite(ens)
    rows = []
    for i in np.where(ok)[0]:
        rows.append(
            {
                "source_file": sf[i],
                "trace_scatter_mag": float(ens[i]),
                "implied_ens_mag": float(implied_raw[i]),
                "abs_diff_mag": float(diff[i]),
                "err_rel": float(err_rel[i]),
                "err_rel_round6": float(err_round[i]),
                "phot_rel": float(phot_rel[i]),
                "err_scatter_unmatched": bool(lc["err_scatter_unmatched"].iloc[i])
                if "err_scatter_unmatched" in lc.columns
                else None,
                "mechanism": (
                    "fp_decomposition_photon_dominated"
                    if float(ens[i]) == 0.0 and float(diff[i]) > TOL
                    else (
                        "np_round_err_6"
                        if float(abs(implied_round[i] - ens[i])) < float(diff[i])
                        else "ensemble_join_ok"
                    )
                ),
            }
        )
    rows.sort(key=lambda r: r["abs_diff_mag"], reverse=True)
    max_diff = float(np.max(diff[ok])) if ok.any() else float("nan")
    return {
        "setup": setup,
        "carrier_matches_normalize": bool(max_diff < TOL),
        "max_abs_diff_mag": max_diff,
        "tolerance_mag": TOL,
        "top_frames": rows[:5],
    }


def main() -> int:
    cfg = AppConfig()
    archive = _ROOT / "Archive"
    out = [_diagnose := diagnose(archive, s, cfg) for s in SETUPS]
    payload = {"setups": out, "attribution": "np.round(err,6) + finite-precision photon-only decomposition"}
    path = _ROOT / "tmp" / "carrier_normalize_diagnose.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
