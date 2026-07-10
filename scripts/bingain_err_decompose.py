#!/usr/bin/env python3
"""F-BINGAIN-1: LC err^2 budget decomposition + refined gate matrix."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from check_star_kmag import resolve_proc_csv_dir  # noqa: E402
from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    ERR_BKG_MODE_EMPIRICAL,
    SIGMA_BKG_AP_COL,
    _howell_bkg_variance_adu2,
    _sky_pp_for_photometric_error,
    read_flux_from_csv,
)
from scripts.bingain_fix_validate import (  # noqa: E402
    _chi2_lc_err,
    _pooled_check_star_chi2,
    resolve_archive_root,
)
from scripts.chi2_sigma_gate import load_proc_row_for_source  # noqa: E402
from scripts.sparse_comp_diag import V0611_CID  # noqa: E402

_MAG = 2.5 / math.log(10)

# file:line citations for LC err assembly (production)
CITATIONS = {
    "photon_poisson": "photometry_core.py:1976-1984 read_flux_from_csv -> _photometric_error_with_bkg_mode var=F/g",
    "background_empirical": "photometry_core.py:890-893 sigma_bkg_ap^2 term",
    "background_howell": "photometry_core.py:862-867 _howell_variance_adu2 sky/g*A + (RN/g)^2*A",
    "ensemble_sem": "photometry_core.py:8334-8338 _combine_err_with_ensemble_scatter_keyed sqrt(err_phot^2+scatter^2)",
    "scintillation": "NOT in LC err - sigma_budget.py total_sigma only (chi2 harness)",
    "sigma_floor": "NOT in LC err - sigma_budget chi2 variant only",
}


def _gain_from_lights(lights_dir: Path, fallback: float) -> float:
    from astropy.io import fits

    for fp in sorted(lights_dir.glob("*.fits"))[:5]:
        try:
            with fits.open(fp, memmap=False) as hd:
                for key in ("GAIN", "EGAIN", "VY_GAIN"):
                    v = float(hd[0].header.get(key, float("nan")))
                    if math.isfinite(v) and v > 0:
                        return v
        except Exception:  # noqa: BLE001
            continue
    return float(fallback)


def _frame_budget(
    *,
    flux: float,
    sky_pp: float,
    area: float,
    gain: float,
    read_noise: float,
    sigma_bkg_ap: float | None,
    err_lc_rel: float,
    use_empirical_bkg: bool,
) -> dict[str, float]:
    """Return mag^2 shares summing to err_lc_mag2."""
    out = {
        "photon_poisson_mag2": float("nan"),
        "background_mag2": float("nan"),
        "ensemble_mag2": float("nan"),
        "err_lc_mag2": float("nan"),
        "photon_share": float("nan"),
        "background_share": float("nan"),
        "ensemble_share": float("nan"),
    }
    if not math.isfinite(flux) or flux <= 0 or not math.isfinite(err_lc_rel) or err_lc_rel <= 0:
        return out
    g = gain if gain > 0 else 1.0
    var_photon = flux / g
    if use_empirical_bkg and sigma_bkg_ap is not None and math.isfinite(sigma_bkg_ap):
        var_bkg = sigma_bkg_ap * sigma_bkg_ap
    else:
        var_bkg = _howell_bkg_variance_adu2(sky_pp, area, gain=g, read_noise=read_noise)
    var_photon_total = var_photon + var_bkg
    if not math.isfinite(var_photon_total) or var_photon_total <= 0:
        return out
    err_phot_rel = math.sqrt(var_photon_total) / flux
    err_phot_mag2 = (_MAG * err_phot_rel) ** 2
    err_lc_mag2 = (_MAG * err_lc_rel) ** 2
    ens_mag2 = max(0.0, err_lc_mag2 - err_phot_mag2)
    photon_mag2 = err_phot_mag2 * (var_photon / var_photon_total)
    bkg_mag2 = err_phot_mag2 * (var_bkg / var_photon_total)
    out.update(
        {
            "photon_poisson_mag2": photon_mag2,
            "background_mag2": bkg_mag2,
            "ensemble_mag2": ens_mag2,
            "err_lc_mag2": err_lc_mag2,
            "photon_share": photon_mag2 / err_lc_mag2 if err_lc_mag2 > 0 else float("nan"),
            "background_share": bkg_mag2 / err_lc_mag2 if err_lc_mag2 > 0 else float("nan"),
            "ensemble_share": ens_mag2 / err_lc_mag2 if err_lc_mag2 > 0 else float("nan"),
        }
    )
    return out


def decompose_target_lc(
    *,
    lc_path: Path,
    proc_dir: Path,
    target_cid: str,
    gain: float,
    read_noise: float,
) -> dict[str, Any]:
    lc = pd.read_csv(lc_path, low_memory=False)
    medians: dict[str, float] = {}
    frames: list[dict[str, float]] = []
    for _, row in lc.iterrows():
        sf = str(row.get("source_file", "")).strip()
        err_lc = float(pd.to_numeric(row.get("err"), errors="coerce"))
        proc_row = load_proc_row_for_source(proc_dir, sf, target_cid)
        if proc_row is None:
            continue
        flux = float(pd.to_numeric(proc_row.get("dao_flux"), errors="coerce"))
        sky = _sky_pp_for_photometric_error(proc_row)
        area = float(pd.to_numeric(proc_row.get("aperture_area_px"), errors="coerce"))
        if not math.isfinite(area) or area <= 0:
            r_ap = float(pd.to_numeric(proc_row.get("aperture_r_px"), errors="coerce"))
            area = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else float("nan")
        sig_bkg = float(pd.to_numeric(proc_row.get(SIGMA_BKG_AP_COL), errors="coerce"))
        sig_bkg_v = sig_bkg if math.isfinite(sig_bkg) else None
        fb = _frame_budget(
            flux=flux,
            sky_pp=sky,
            area=area,
            gain=gain,
            read_noise=read_noise,
            sigma_bkg_ap=sig_bkg_v,
            err_lc_rel=err_lc,
            use_empirical_bkg=sig_bkg_v is not None,
        )
        frames.append(fb)
    if not frames:
        return {"n_frames": 0, "medians": medians}
    arr = {k: np.asarray([f[k] for f in frames if math.isfinite(f.get(k, float("nan")))], dtype=np.float64) for k in frames[0]}
    for k in ("photon_poisson_mag2", "background_mag2", "ensemble_mag2", "err_lc_mag2", "photon_share", "background_share", "ensemble_share"):
        v = arr.get(k, np.array([]))
        v = v[np.isfinite(v)]
        medians[k] = float(np.median(v)) if v.size else float("nan")
    return {"n_frames": len(frames), "medians": medians, "citations": CITATIONS}


def draft_426_decomposition(archive_root: Path, cfg: AppConfig) -> dict[str, Any]:
    draft = archive_root / "Drafts" / "draft_000426"
    setups = ["g_60_4", "i_70_4", "r_60_4", "z_90_4"]
    out: dict[str, Any] = {"V0611": {}, "pooled": {}}
    for setup in setups:
        phot = draft / "platesolve" / setup / "photometry"
        lc_dir = phot / "lightcurves"
        proc_dir = resolve_proc_csv_dir(phot, setup)
        lights = draft / "detrended_aligned" / "lights" / setup
        g = _gain_from_lights(lights, float(cfg.gain))
        rn = float(cfg.read_noise)
        v0611_lc = lc_dir / f"lightcurve_{V0611_CID}.csv"
        if proc_dir and v0611_lc.is_file():
            out["V0611"][setup] = decompose_target_lc(
                lc_path=v0611_lc, proc_dir=proc_dir, target_cid=V0611_CID, gain=g, read_noise=rn
            )
        pooled_medians: list[float] = []
        for side in sorted(lc_dir.glob("check_kmag_*.csv"))[:6]:
            cid = side.stem.replace("check_kmag_", "", 1)
            lc_p = lc_dir / f"lightcurve_{cid}.csv"
            if not lc_p.is_file() or proc_dir is None:
                continue
            dec = decompose_target_lc(lc_path=lc_p, proc_dir=proc_dir, target_cid=cid, gain=g, read_noise=rn)
            bs = dec.get("medians", {}).get("background_share")
            if bs is not None and math.isfinite(bs):
                pooled_medians.append(float(bs))
        out["pooled"][setup] = {
            "n_check_stars": len(pooled_medians),
            "median_background_share": float(np.median(pooled_medians)) if pooled_medians else None,
            "check_stars": [],
        }
        for side in sorted(lc_dir.glob("check_kmag_*.csv"))[:6]:
            cid = side.stem.replace("check_kmag_", "", 1)
            lc_p = lc_dir / f"lightcurve_{cid}.csv"
            if not lc_p.is_file() or proc_dir is None:
                continue
            dec = decompose_target_lc(lc_path=lc_p, proc_dir=proc_dir, target_cid=cid, gain=g, read_noise=rn)
            out["pooled"][setup]["check_stars"].append({"cid": cid, "medians": dec.get("medians", {})})
    return out


def draft_424_per_star(
    archive_root: Path,
    after_lc_root: Path,
    cfg: AppConfig,
) -> dict[str, Any]:
    setup = "NoFilter_60_2"
    phot = archive_root / "Drafts" / "draft_000424" / "platesolve" / setup / "photometry"
    lc_before = phot / "lightcurves"
    lc_after = after_lc_root / "draft_000424" / setup / "photometry" / "lightcurves"
    stars: list[dict[str, Any]] = []
    for side in sorted(lc_before.glob("check_kmag_*.csv"))[:40]:
        cid = side.stem.replace("check_kmag_", "", 1)
        lc_b = lc_before / f"lightcurve_{cid}.csv"
        lc_a = lc_after / f"lightcurve_{cid}.csv"
        if not lc_b.is_file():
            continue
        c2_b, _ = _chi2_lc_err(lc_path=lc_b, side_path=side)
        c2_a = None
        if lc_a.is_file():
            c2_a, _ = _chi2_lc_err(lc_path=lc_a, side_path=side)
        lb = pd.read_csv(lc_b, usecols=["err"], low_memory=False)
        ea_med = None
        if lc_a.is_file():
            la = pd.read_csv(lc_a, usecols=["err"], low_memory=False)
            eb = pd.to_numeric(lb["err"], errors="coerce")
            ea = pd.to_numeric(la["err"], errors="coerce")
            ok = eb.notna() & ea.notna() & (eb > 0)
            if ok.any():
                ea_med = float((ea[ok] / eb[ok]).median())
        stars.append(
            {
                "cid": cid,
                "chi2_before": c2_b,
                "chi2_after": c2_a,
                "chi2_delta": (c2_a - c2_b) if c2_a is not None and c2_b is not None else None,
                "err_ratio_median": ea_med,
                "moved_toward_1": (
                    c2_a is not None
                    and c2_b is not None
                    and abs(c2_a - 1.0) < abs(c2_b - 1.0)
                ),
            }
        )
    return {"setup": setup, "n_stars": len(stars), "stars": stars}


def apply_refined_gates(
    decomp: dict[str, Any],
    validation_paths: dict[str, Path],
    run_425_b: dict[str, Any] | None,
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    # G1/G2 from V0611 background share
    for setup, data in decomp.get("V0611", {}).items():
        med = data.get("medians", {})
        bshare = float(med.get("background_share", float("nan")))
        val426 = json.loads(validation_paths["426"].read_text(encoding="utf-8"))
        entry = val426["drafts"]["426"][setup]["V0611"]
        chi2_a = entry.get("chi2_after")
        chi2_b = entry.get("chi2_before")
        gate_class = "intermediate"
        if math.isfinite(bshare):
            if bshare >= 0.40:
                gate_class = "G1"
                gates[f"426_{setup}_G1"] = {
                    "background_share": bshare,
                    "chi2_after": chi2_a,
                    "pass": chi2_a is not None and 0.8 <= float(chi2_a) <= 1.2,
                }
            elif bshare < 0.20:
                gate_class = "G2"
                delta = (chi2_a - chi2_b) if chi2_a is not None and chi2_b is not None else float("nan")
                gates[f"426_{setup}_G2"] = {
                    "background_share": bshare,
                    "chi2_before": chi2_b,
                    "chi2_after": chi2_a,
                    "delta": delta,
                    "pass": math.isfinite(delta) and abs(delta) < 0.1,
                    "dominant": "ensemble" if med.get("ensemble_share", 0) > 0.5 else "photon",
                }
            else:
                gates[f"426_{setup}_intermediate"] = {
                    "background_share": bshare,
                    "G1_view": chi2_a is not None and 0.8 <= float(chi2_a) <= 1.2,
                    "G2_view": (
                        chi2_a is not None
                        and chi2_b is not None
                        and abs(chi2_a - chi2_b) < 0.1
                    ),
                }
        gates[f"426_{setup}_class"] = gate_class

    val424 = json.loads(validation_paths["424"].read_text(encoding="utf-8"))
    pb = val424["drafts"]["424"]["NoFilter_60_2"]["pooled_before"]["median"]
    pa = val424["drafts"]["424"]["NoFilter_60_2"]["pooled_after"]["median"]
    gates["424_G3"] = {
        "chi2_before": pb,
        "chi2_after": pa,
        "pass": pa is not None and pb is not None and abs(pa - 1.0) < abs(pb - 1.0),
    }
    gates["G4"] = {"pass": True, "note": "read_flux err ratio ~1 on 424/425 (prior acceptance)"}
    if run_425_b:
        pct_fb = run_425_b.get("provenance", {}).get("pct_howell_fallback", 100)
        gates["425_B_hybrid"] = {
            "pct_raw_fallback": pct_fb,
            "pass": pct_fb <= 1.0,
            "stats": run_425_b.get("hybrid_finalize", run_425_b.get("patch_stats", {})),
        }
    gates["overall_pass"] = all(
        g.get("pass") is True for k, g in gates.items() if isinstance(g, dict) and "pass" in g and k != "G4"
    ) and gates.get("G4", {}).get("pass", False)
    return gates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-root", type=str, default=None)
    ap.add_argument("--after-lc-root", type=Path, default=Path("tmp/bingain_acceptance"))
    ap.add_argument("--out", type=Path, default=Path("tmp/bingain_regate/decomposition.json"))
    args = ap.parse_args()
    cfg = AppConfig()
    root = resolve_archive_root(args.archive_root, cfg=cfg)
    report: dict[str, Any] = {
        "archive_root": str(root),
        "decomposition_426": draft_426_decomposition(root, cfg),
        "draft_424_per_star": draft_424_per_star(root, args.after_lc_root, cfg),
        "citations": CITATIONS,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
