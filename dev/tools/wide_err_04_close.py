#!/usr/bin/env python3
"""WIDE-ERR-04: identity physical-model calib (s=1, sigma_r=0), re-export, accuracy table."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "tools"))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from err_calibration import ERR_CALIB_SIDECAR, SmoothCalibration, write_sidecar  # noqa: E402
from invariants_runtime import STAGE_ORDER, load_pipeline_meta, save_pipeline_meta  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402

RUN_SHA = "da9cce4a5edd1392b8ba842d3c8488589b9d0ac9"
DRAFT_ID = 515
SETUP = "NoFilter_60_2"
DRAFT = ROOT / "Archive" / "Drafts" / f"draft_{DRAFT_ID:06d}"
PS = DRAFT / "platesolve" / SETUP
LIGHTS = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOT = PS / "photometry"
LC = PHOT / "lightcurves"
BACKUP = ROOT / "tmp" / "wide_err_04_lc_before"
OUT_RE = ROOT / "dev" / "results" / "WIDE_ERR_04_reexport.json"
OUT_1C = ROOT / "dev" / "results" / "WIDE_ERR_04_accuracy.json"
OUT_SUM = ROOT / "dev" / "results" / "WIDE_ERR_04_summary.json"

BO = "1498613634033133184"
FW = "1497343732462852864"
MAG_COLS = (
    "mag_inst",
    "mag_calib_raw",
    "mag_calib",
    "mag_calib_ct",
    "mag_calib_ac",
    "mag_calib_final",
    "delta_mag",
)


def identity_cal() -> SmoothCalibration:
    return SmoothCalibration(
        s=1.0,
        sigma_r0_mmag=0.0,
        sigma_r_slope_mmag_per_G=0.0,
        form="constant_sigma_r",
        n_stars=0,
        s_clamped=False,
        median_ratio_pre=float("nan"),
        median_ratio_post=float("nan"),
    )


def write_identity_sidecar() -> Path:
    cal = identity_cal()
    path = PHOT / ERR_CALIB_SIDECAR
    write_sidecar(
        path,
        {
            "task": "WIDE-ERR-04",
            "run_sha": RUN_SHA,
            "draft_id": DRAFT_ID,
            "form": "err_exported^2 = (s * err_model)^2 + sigma_r(G)^2",
            "note": (
                "Identity physical model: s=1, sigma_r=0. Container-domain g_pt + "
                "weighted SEM; no empirical floor. Machinery retained for future rigs."
            ),
            "smooth": cal.to_dict(),
            "s_min_clamp": 1.0,
            "gain_authority": 0.6370667331227862,
            "default_for_new_drafts": (
                "identity (s=1, sigma_r=0) unless a per-draft sidecar overrides"
            ),
        },
    )
    return path


def snapshot_lcs(src: Path, dst: Path) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("lightcurve_*.csv")):
        shutil.copy2(p, dst / p.name)
        n += 1
    return n


def median_err(path: Path) -> float:
    df = pd.read_csv(path, comment="#", low_memory=False)
    return float(pd.to_numeric(df["err"], errors="coerce").median())


def mag_identity(before: Path, after: Path) -> dict:
    files_a = {p.name: p for p in before.glob("lightcurve_*.csv")}
    files_b = {p.name: p for p in after.glob("lightcurve_*.csv")}
    common = sorted(set(files_a) & set(files_b))
    n_ok = n_fail = 0
    fails: list[str] = []
    for name in common:
        da = pd.read_csv(files_a[name], comment="#", low_memory=False)
        db = pd.read_csv(files_b[name], comment="#", low_memory=False)
        cols = [c for c in MAG_COLS if c in da.columns and c in db.columns]
        ok = True
        for c in cols:
            a = pd.to_numeric(da[c], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(db[c], errors="coerce").to_numpy(dtype=float)
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                ok = False
                break
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            if len(fails) < 5:
                fails.append(name)
    return {
        "n_common": len(common),
        "n_mag_byte_identical": n_ok,
        "n_mag_mismatch": n_fail,
        "pass": n_fail == 0 and len(common) > 0,
        "fail_examples": fails,
    }


def reexport() -> dict:
    before_bo = median_err(LC / f"lightcurve_{BO}.csv")
    before_fw = median_err(LC / f"lightcurve_{FW}.csv")
    n_snap = snapshot_lcs(LC, BACKUP)
    print(
        f"snapshot before n={n_snap}: BO={before_bo * 1000:.3f} "
        f"FW={before_fw * 1000:.3f} mmag",
        flush=True,
    )

    cfg = AppConfig()
    cfg.export_err_mode = "calibrated"
    db = VyvarDatabase(Path(cfg.database_path))
    meta = load_pipeline_meta(PHOT)
    stages = meta.get("stages") if isinstance(meta.get("stages"), list) else []
    p2_seq = STAGE_ORDER.index("phase2a")
    meta["stages"] = [
        s
        for s in stages
        if isinstance(s, dict)
        and str(s.get("name") or "") in STAGE_ORDER
        and STAGE_ORDER.index(str(s.get("name"))) < p2_seq
    ]
    save_pipeline_meta(PHOT, meta)

    fw = float(_load_fwhm(PS / "MASTERSTAR.fits"))
    t0 = time.time()
    run_phase2a(
        masterstar_fits_path=PS / "MASTERSTAR.fits",
        active_targets_csv=PHOT / "active_targets.csv",
        comparison_stars_csv=PHOT / "comparison_stars_per_target.csv",
        per_frame_csv_dir=LIGHTS,
        detrended_aligned_dir=LIGHTS,
        output_dir=PHOT,
        fwhm_px=fw,
        cfg=cfg,
        db=db,
        draft_id=DRAFT_ID,
        progress_cb=lambda m: print(m, flush=True),
    )
    elapsed = time.time() - t0
    after_bo = median_err(LC / f"lightcurve_{BO}.csv")
    after_fw = median_err(LC / f"lightcurve_{FW}.csv")
    ident = mag_identity(BACKUP, LC)
    payload = {
        "task": "WIDE-ERR-04 1b re-export",
        "run_sha": RUN_SHA,
        "elapsed_s": elapsed,
        "mag_byte_identity": ident,
        "err_before_after": {
            "BO": {
                "catalog_id": BO,
                "median_err_before_mmag": before_bo * 1000.0,
                "median_err_after_mmag": after_bo * 1000.0,
                "frame": "exported LC err [mmag]",
            },
            "FW": {
                "catalog_id": FW,
                "median_err_before_mmag": before_fw * 1000.0,
                "median_err_after_mmag": after_fw * 1000.0,
                "frame": "exported LC err [mmag]",
            },
        },
        "backup_dir": str(BACKUP),
    }
    OUT_RE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT_RE)
    print("mag identity", ident)
    print(
        f"BO {before_bo * 1000:.3f}->{after_bo * 1000:.3f}; "
        f"FW {before_fw * 1000:.3f}->{after_fw * 1000:.3f} mmag"
    )
    return payload


def accuracy_table() -> dict:
    """Product-frame even-half ratios with identity calib (documentation, not a gate)."""
    import wide_err_03c as w03c  # noqa: PLC0415

    # Re-run 03c measure path by calling main's early section via a dedicated entry.
    # Use the C2 even_gate from 03C (identity winner) and refresh by calling eval
    # after a focused measure: invoke w03c with ACCURACY_ONLY env.
    print("Recomputing product-frame even-half accuracy (identity)...", flush=True)
    # Call into 03c by running measure through importing and using a thin wrapper
    # added below as process: execute w03c.main pieces - use saved C2 + verify
    # err_model unchanged (identity) by recomputing via w03c.eval_bins on fresh rows.
    #
    # Fresh measure: run python -c importing w03c helpers - keep in-process:
    rows_even = _measure_even_rows(w03c)
    cal = identity_cal()
    cal.n_stars = len(rows_even)
    ev = w03c.eval_bins(rows_even, cal)
    payload = {
        "task": "WIDE-ERR-04 1c accuracy statement",
        "run_sha": RUN_SHA,
        "frame": "product mag_calib (XVAL-BO-01); EVEN-indexed frames held-out",
        "calibration": {"s": 1.0, "sigma_r_mmag": 0.0, "form": "identity physical model"},
        "n_stars": len(rows_even),
        "eval": ev,
        "note": "Documentation of what ships; not a pass/fail closure gate.",
    }
    OUT_1C.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT_1C)
    for lab, v in ev["by_G_bin"].items():
        if v.get("gated"):
            print(
                lab,
                "n",
                v["n"],
                "ratio",
                round(v.get("median_ratio", float("nan")), 3),
                "err_mmag",
                round(v.get("median_err_mmag", float("nan")), 2),
            )
    print("G8_9", ev["G8_9"])
    return payload


def _measure_even_rows(w03c):
    """Build even-half product-frame rows using wide_err_03c helpers."""
    import math

    from gain_photon_transfer import estimate_photon_transfer_gain_from_proc_dir
    from mag_constants import MAG_ERR_SCALE
    from photometry_core import _photometric_error_with_bkg_mode
    from sigma_budget import scintillation_sigma

    meta = json.loads((PHOT / "pipeline_meta.json").read_text(encoding="utf-8"))
    dyn = meta.get("dynamic_params") or {}
    ap_r = float(dyn.get("aperture_r_px") or 3.999)
    g_wrong = float(dyn.get("gain") or 3.17)
    rn = float(dyn.get("read_noise") or 15.2)
    pt = estimate_photon_transfer_gain_from_proc_dir(w03c.PROC, aperture_r_px=ap_r)
    g_new = float(pt.g_pt) if pt.ok else (g_wrong / 4.0)
    area = math.pi * ap_r * ap_r
    alt_m = float((meta.get("observer_location") or {}).get("alt_m") or 275.0)
    scint_rel = scintillation_sigma(
        telescope_diameter_m=0.070, airmass=1.2, exposure_s=60.0, altitude_m=alt_m, c_y=1.5
    )
    emp = pd.read_csv(w03c.EMP, dtype={"catalog_id": str})
    emp["catalog_id"] = emp["catalog_id"].astype(str).str.strip()
    clean = set(emp["catalog_id"])
    g_emp = {str(r.catalog_id): float(r.G) for r in emp.itertuples()}

    frames = []
    for p in sorted(w03c.PROC.glob("proc_*.csv")):
        df = pd.read_csv(
            p,
            dtype={"catalog_id": str},
            usecols=lambda c: c
            in (
                "catalog_id",
                "flux",
                "dao_flux",
                "sigma_bkg_ap",
                "sky_adu_per_px_annulus",
                "aperture_r_px",
            ),
        )
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        df["frame"] = p.stem.replace("proc_", "")
        if "dao_flux" in df.columns:
            df["flux_use"] = pd.to_numeric(df["dao_flux"], errors="coerce")
        else:
            df["flux_use"] = pd.to_numeric(df["flux"], errors="coerce")
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    for c in ("flux_use", "sigma_bkg_ap", "sky_adu_per_px_annulus", "aperture_r_px"):
        allp[c] = pd.to_numeric(allp[c], errors="coerce")
    frame_list = sorted(allp["frame"].unique())
    even_frames = {fr for i, fr in enumerate(frame_list) if i % 2 == 0}
    frame_index = {fr: i for i, fr in enumerate(frame_list)}

    comp = pd.read_csv(
        PHOT / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp["catalog_id"] = comp["catalog_id"].astype(str).str.strip()
    comp["target_catalog_id"] = comp["target_catalog_id"].astype(str).str.strip()
    sus_ids: set[str] = set()
    sus = PHOT / "suspected_variables.csv"
    if sus.is_file():
        sdf = pd.read_csv(sus, dtype={"catalog_id": str})
        if "catalog_id" in sdf.columns:
            sus_ids = set(sdf["catalog_id"].astype(str).str.strip())

    mag_by: dict = {}
    flux_rows_by: dict = {}
    night_med: dict = {}
    median_ap: dict = {}
    for cid, gdf in allp.groupby("catalog_id", sort=False):
        gdf = gdf.set_index("frame")
        flux_rows_by[cid] = gdf
        mags = np.full(len(frame_list), float("nan"))
        aps = []
        for i, fr in enumerate(frame_list):
            if fr not in gdf.index:
                continue
            row = gdf.loc[fr]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            mags[i] = w03c.flux_to_mag(float(row["flux_use"]))
            apv = float(row["aperture_r_px"])
            if math.isfinite(apv):
                aps.append(apv)
        mag_by[cid] = mags
        finite = mags[np.isfinite(mags)]
        night_med[cid] = float(np.median(finite)) if finite.size else float("nan")
        median_ap[cid] = float(np.median(aps)) if aps else float("nan")

    primary: dict = {}
    for tid in sorted(comp["target_catalog_id"].unique()):
        sub = comp.loc[comp["target_catalog_id"] == tid]
        ens = [str(x) for x in sub["catalog_id"].tolist()]
        cat_g = {}
        rms = {}
        for _, r in sub.iterrows():
            cid = str(r["catalog_id"]).strip()
            cat_g[cid] = float(
                pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce")
            )
            rms[cid] = float(pd.to_numeric(r.get("comp_rms"), errors="coerce"))
        for cid in ens:
            if cid in clean and cid not in sus_ids and cid not in primary:
                primary[cid] = (tid, ens, cat_g, rms)

    rows = []
    mask_idx = np.asarray(
        [frame_index[fr] for fr in frame_list if fr in even_frames], dtype=int
    )
    for cid, (_tid, ens, cat_g, rms) in primary.items():
        if cid not in mag_by:
            continue
        kcal, case, _ = w03c.mag_calib_series(
            m_star=mag_by[cid],
            ens_ids=ens,
            mag_by=mag_by,
            cat_g=cat_g,
            rms_phase1=rms,
            self_exclude=(cid in ens),
            focus_id=cid,
        )
        scat = w03c.mad_mmag(kcal[mask_idx])
        if not math.isfinite(scat):
            continue
        others = [c for c in ens if c != cid]
        err_new = []
        gdf = flux_rows_by[cid]
        for fr in frame_list:
            if fr not in even_frames or fr not in gdf.index:
                continue
            row = gdf.loc[fr]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            f_c = float(row["flux_use"])
            resid_o = []
            for oid in others:
                if oid not in mag_by:
                    continue
                mi = mag_by[oid][frame_index[fr]]
                nm = night_med.get(oid, float("nan"))
                if math.isfinite(mi) and math.isfinite(nm):
                    resid_o.append(mi - nm)
            sem_mag = w03c.sem_unweighted(resid_o)
            sem_rel = (
                (sem_mag / MAG_ERR_SCALE) if math.isfinite(sem_mag) and sem_mag > 0 else 0.0
            )
            sig = float(row["sigma_bkg_ap"])
            sky = float(row["sky_adu_per_px_annulus"])
            r_ap = float(row["aperture_r_px"])
            ar = math.pi * r_ap * r_ap if math.isfinite(r_ap) and r_ap > 0 else area
            e_n, _ = _photometric_error_with_bkg_mode(
                f_c,
                err_background_mode="empirical",
                sky_pp=sky if math.isfinite(sky) else 0.0,
                area=ar,
                gain=g_new,
                read_noise=rn,
                sigma_bkg_ap=sig,
            )
            if math.isfinite(e_n):
                err_new.append(math.sqrt(e_n * e_n + sem_rel * sem_rel + scint_rel * scint_rel))
        en = w03c.rel_to_mmag(float(np.median(err_new))) if err_new else float("nan")
        rows.append(
            {
                "catalog_id": cid,
                "G": g_emp.get(cid, float("nan")),
                "scatter_mmag": scat,
                "err_model_mmag": en,
                "ensemble_case": case,
                "r_ap_px": median_ap.get(cid, float("nan")),
            }
        )
    return rows


def main() -> int:
    side = write_identity_sidecar()
    print("WROTE identity sidecar", side, flush=True)
    re = reexport()
    if not re["mag_byte_identity"]["pass"]:
        return 3
    acc = accuracy_table()
    summary = {
        "task": "WIDE-ERR-04",
        "run_sha": RUN_SHA,
        "identity_sidecar": str(side),
        "reexport": re,
        "accuracy_G8_9": acc["eval"]["G8_9"],
        "accuracy_bins_outside": acc["eval"]["bins_outside"],
    }
    OUT_SUM.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT_SUM)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
