"""Validate sampling-gated tighten: reprocess 361 + 362 (OFF legacy vs ON gated),
plus 360 ON to confirm the gate suppresses its tighten (OFF == ON for 360 by construction).

OFF = crowding_classifier_enabled FALSE -> legacy stars/Mpx (361/362 DENSE -> tighten).
ON  = crowding_classifier_enabled TRUE  -> sampling-gated (undersampled -> NO tighten).
Both runs resolve plate scale from the WCS (~9.77), fixing the stale-1.3 A4 artifact.
Archived photometry/ is NOT touched (isolated output dirs, removed at end).

Run from the repo root (writes _repro_ab2_result.json + _repro_ab2.log to the CWD):
    python tools/validation/crowding_ab.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# kill cp1252 console crashes on Slovak diacritics in log messages
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

from config import AppConfig
from database import VyvarDatabase
from photometry_core import run_full_photometry_pipeline

SETUP = "NoFilter_60_2"
# (draft_id, [modes]); 360 ON-only (OFF==ON: both apply zero overrides under the gate)
PLAN = [(360, ["on"]), (361, ["off", "on"]), (362, ["off", "on"])]

BPRP_OFF, BPRP_ON = 0.79, 0.64
RMS_OFF, RMS_ON = 0.10, 0.08
MINDIST_ON = 90.0


def run(draft: int, flag: bool) -> Path:
    c = AppConfig()
    c.crowding_classifier_enabled = bool(flag)
    ddir = Path(c.archive_root) / "Drafts" / f"draft_{draft:06d}"
    ps = ddir / "platesolve" / SETUP
    aligned = ddir / "detrended_aligned" / "lights" / SETUP
    out = ps / ("photometry_abtest_on" if flag else "photometry_abtest_off")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logging.info("=== REPROCESS %d crowding=%s -> %s ===", draft, flag, out.name)
    run_full_photometry_pipeline(
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        variable_targets_csv=ps / "variable_targets.csv",
        masterstars_csv=ps / "masterstars_full_match.csv",
        per_frame_csv_dir=aligned,
        detrended_aligned_dir=aligned,
        output_dir=out,
        cfg=c,
        db=VyvarDatabase(c.database_path),
        draft_id=draft,
        progress_cb=None,
    )
    logging.info("=== draft %d flag=%s done in %.1fs ===", draft, flag, time.time() - t0)
    return out


def _robust(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _p2p(df: pd.DataFrame, col: str) -> float:
    d = df.dropna(subset=["bjd", col]).sort_values("bjd")
    v = pd.to_numeric(d[col], errors="coerce").to_numpy()
    v = v[np.isfinite(v)]
    if v.size < 3:
        return float("nan")
    return float(np.median(np.abs(np.diff(v))))


def summarize(out: Path) -> dict:
    meta = json.loads((out / "pipeline_meta.json").read_text(encoding="utf-8"))
    dp = meta.get("dynamic_params", {})
    comp = pd.read_csv(out / "comparison_stars_per_target.csv", low_memory=False)
    summ = pd.read_csv(out / "photometry_summary.csv", low_memory=False)
    comp_rms = pd.to_numeric(comp.get("comp_rms"), errors="coerce")
    dbprp = pd.to_numeric(comp.get("delta_bprp_abs"), errors="coerce")
    dist_arcsec = pd.to_numeric(comp.get("_dist_deg"), errors="coerce") * 3600.0
    info = {
        "plate_scale": round(float(dp.get("plate_scale_arcsec_px", float("nan"))), 4),
        "fwhm_px": dp.get("fwhm_px"),
        "aperture_r_px": dp.get("aperture_r_px"),
        "comp_pairs": int(len(comp)),
        "unique_comps": int(comp["catalog_id"].astype(str).nunique()) if "catalog_id" in comp else None,
        "median_comp_rms": round(float(comp_rms.median()), 5),
        "median_n_good_comp": round(float(pd.to_numeric(summ.get("n_good_comp"), errors="coerce").median()), 2),
        "median_lc_rms": round(float(pd.to_numeric(summ.get("lc_rms"), errors="coerce").median()), 5),
        "good": int((summ.get("lc_quality_flag") == "good").sum()) if "lc_quality_flag" in summ else None,
        "pairs_rms_0.08_0.10": int(((comp_rms > RMS_ON) & (comp_rms <= RMS_OFF)).sum()),
    }
    cj = out / "crowding_index.json"
    if cj.is_file():
        info["classifier"] = json.loads(cj.read_text(encoding="utf-8")).get("classifier")
    return info


def lc_compare(off: Path, on: Path) -> dict:
    def load(d: Path) -> dict:
        res = {}
        for f in (d / "lightcurves").glob("lightcurve_*.csv"):
            cid = f.stem.replace("lightcurve_", "")
            try:
                df = pd.read_csv(f, low_memory=False)
            except Exception:
                continue
            col = "mag_calib" if "mag_calib" in df.columns else None
            if col is None or "bjd" not in df.columns:
                continue
            res[cid] = (_robust(pd.to_numeric(df[col], errors="coerce").to_numpy()), _p2p(df, col))
        return res

    a, b = load(off), load(on)
    common = sorted(set(a) & set(b))
    ro = np.array([a[c][0] for c in common], float)
    rn = np.array([b[c][0] for c in common], float)
    po = np.array([a[c][1] for c in common], float)
    pn = np.array([b[c][1] for c in common], float)
    m = np.isfinite(ro) & np.isfinite(rn)
    return {
        "n_common": int(len(common)),
        "robust_scatter_OFF": round(float(np.nanmedian(ro)), 5),
        "robust_scatter_ON": round(float(np.nanmedian(rn)), 5),
        "p2p_OFF": round(float(np.nanmedian(po)), 5),
        "p2p_ON": round(float(np.nanmedian(pn)), 5),
        "improved_ON": int(np.sum(rn[m] < ro[m])),
        "worse_ON": int(np.sum(rn[m] > ro[m])),
        "equal": int(np.sum(rn[m] == ro[m])),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler("_repro_ab2.log", encoding="utf-8")],
    )
    results: dict = {}
    for draft, modes in PLAN:
        outs = {}
        for mode in modes:
            outs[mode] = str(run(draft, mode == "on"))
        entry = {mode: summarize(Path(p)) for mode, p in outs.items()}
        if "off" in outs and "on" in outs:
            entry["lc"] = lc_compare(Path(outs["off"]), Path(outs["on"]))
        results[draft] = entry
        json.dump(results, open("_repro_ab2_result.json", "w", encoding="utf-8"), indent=2, default=str)

    print("\n" + "=" * 84)
    print("SAMPLING-GATED TIGHTEN VALIDATION — 361/362 recovery (+ 360 neutrality)")
    print("=" * 84)
    for draft, entry in results.items():
        print(f"\nDRAFT {draft}")
        for mode in ("off", "on"):
            if mode in entry:
                s = entry[mode]
                cls = s.get("classifier")
                tg = f" tighten={cls['tighten']} sampled={cls['well_sampled']} fwhm={cls['fwhm_px']:.2f}" if cls else ""
                print(f"  {mode.upper():3s}: scale={s['plate_scale']} fwhm={s['fwhm_px']} "
                      f"comps={s['unique_comps']} medRMS={s['median_comp_rms']} "
                      f"n_good={s['median_n_good_comp']} lc_rms={s['median_lc_rms']} good={s['good']}"
                      f" pairs[0.08-0.10]={s['pairs_rms_0.08_0.10']}{tg}")
        if "lc" in entry:
            print(f"  LC(common): robust OFF={entry['lc']['robust_scatter_OFF']} ON={entry['lc']['robust_scatter_ON']} | "
                  f"p2p OFF={entry['lc']['p2p_OFF']} ON={entry['lc']['p2p_ON']} | "
                  f"improved={entry['lc']['improved_ON']} worse={entry['lc']['worse_ON']} eq={entry['lc']['equal']}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
