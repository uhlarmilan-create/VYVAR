# -*- coding: ascii -*-
"""EPSF-XVAL-A1B: root-cause of A1 M2=34/32 mmag (dev-only, measurement).

Governing code (read before this rerun; cited in the result file):
- Snapshot proc x,y: pipeline_catalog._lock_matched_centroids_to_master_grid
  :2181-2247 (integer brightest pixel after snap-to-master).
- VYVAR PSF: psf_photometry.psf_photometry_stars refits x_0,y_0
  (psf_fix_position_enabled default False at :2868); flux at fitted
  centroid; proc column x,y stay the init (:3230).
- Vendor: getpsf.py:56-57 IDL first pixel (0,0); daoerf.py:58-59
  integrates [x-0.5, x+0.5] so integer N is that pixel's centre.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "dev"
SRC = REPO / "src_py"
for _p in (str(SRC), str(DEV)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.session_baseline_check import SETUP, SNAPSHOT_NAME, _copy_frozen_anchor_inputs  # noqa: E402
from xval_pythonphot.run_xval_a1 import (  # noqa: E402
    CHECK_CID,
    TARGET_CID,
    WORK,
    ZEROPOINT,
    _flux_to_inst_mag,
    _proc_rows,
    apply_reading as _unused_a1_reading,
    g4_live_516,
    list_psf_lc_ids,
    load_vyvar_psf_lc,
    measure_frame,
    rebuild_delta,
    reconstruct_psf_star_ids,
    resolve_gain_rn,
    rms_after_median,
)

_ = _unused_a1_reading
OUT = REPO / "tmp" / "session_20260907_epsfxval_a1b" / "out"
CTX = REPO / "dev" / "results" / "context" / "session_20260907_epsfxval_a1b"


def apply_reading_a1b(rms_mmag: float) -> str:
    if not math.isfinite(rms_mmag):
        return (
            "R-A1B-3: recentered M2 > 10 mmag -> position was not the (main) "
            "cause; H-C/H-D tables become the primary evidence; no validation "
            "claim; A2 leg becomes the decisive external reference. (RMS non-finite)"
        )
    if rms_mmag <= 3.0:
        return (
            "R-A1B-1: recentered M2 (target AND check) <= 3.0 mmag -> A1's "
            "R-X3 was harness-positional; the ePSF chain agrees with the "
            "DAOPHOT-lineage reference at AIJ-class level in native mode."
        )
    if rms_mmag <= 10.0:
        return (
            "R-A1B-2: recentered M2 in (3, 10] mmag -> positional cause "
            "partially confirmed; remaining budget attributed per H-C/H-D "
            "measurements; feeds the A2 design (PSFEx is spatially varying by "
            "construction)."
        )
    return (
        "R-A1B-3: recentered M2 > 10 mmag -> position was not the (main) "
        "cause; H-C/H-D tables become the primary evidence; no validation "
        "claim; A2 leg becomes the decisive external reference."
    )


def _light_key(name: str) -> str:
    m = re.search(r"Light_\d+", str(name))
    return m.group(0) if m else str(name)


def load_qc_fwhm(qc_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not qc_path.is_file():
        return out
    df = pd.read_csv(qc_path)
    col = "src" if "src" in df.columns else df.columns[0]
    for _, row in df.iterrows():
        key = _light_key(str(row.get(col, "")))
        fw = float(pd.to_numeric(row.get("fwhm_px"), errors="coerce"))
        if key and math.isfinite(fw):
            out[key] = fw
    return out


def assemble_m1_m2(
    meas_df: pd.DataFrame,
    vyvar_lcs: dict[str, pd.DataFrame],
    lc_ids: list[str],
    ens_ids: list[str],
    ens_w: dict[str, float],
    catmag: dict[str, float],
    live_lc: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    m1_rows: list[dict[str, Any]] = []
    for cid in lc_ids:
        lc = vyvar_lcs[cid]
        sub = meas_df[(meas_df["catalog_id"] == cid) & (meas_df["fail"] < 0.5)]
        merged = lc.merge(sub[["source_file", "pp_flux"]], on="source_file", how="left")
        m_v = _flux_to_inst_mag(pd.to_numeric(merged["psf_flux"], errors="coerce").to_numpy())
        m_p = _flux_to_inst_mag(pd.to_numeric(merged["pp_flux"], errors="coerce").to_numpy())
        d = m_v - m_p
        ok = np.isfinite(d)
        rms = rms_after_median(d)
        m1_rows.append({
            "catalog_id": cid,
            "catalog_mag": catmag.get(cid, float("nan")),
            "n_epochs_lc": int(len(lc)),
            "n_ok": int(ok.sum()),
            "n_nan_or_fail": int((~ok).sum()),
            "rms_mag": rms,
            "rms_mmag": rms * 1000.0 if math.isfinite(rms) else float("nan"),
            "median_d_mag": float(np.nanmedian(d)) if int(ok.sum()) else float("nan"),
        })
    m1 = pd.DataFrame(m1_rows).sort_values(["catalog_mag", "catalog_id"], na_position="last")

    tgt_lc = vyvar_lcs[TARGET_CID]
    tgt_sub = meas_df[meas_df["catalog_id"] == TARGET_CID]
    tgt_m = tgt_lc.merge(tgt_sub[["source_file", "pp_flux"]], on="source_file", how="left")
    n_ep = len(tgt_m)
    pp_comp: dict[str, np.ndarray] = {}
    vy_comp: dict[str, np.ndarray] = {}
    for cid in ens_ids:
        if cid not in vyvar_lcs:
            arr_pp = np.full(n_ep, np.nan)
            arr_vy = np.full(n_ep, np.nan)
            for i, sf in enumerate(tgt_m["source_file"].tolist()):
                hit = meas_df[(meas_df["catalog_id"] == cid) & (meas_df["source_file"] == sf)]
                if not hit.empty and float(hit.iloc[0]["fail"]) < 0.5:
                    arr_pp[i] = float(hit.iloc[0]["pp_flux"])
                arr_vy[i] = float(hit.iloc[0]["vyvar_psf_flux_live_proc"]) if not hit.empty else float("nan")
            pp_comp[cid] = arr_pp
            vy_comp[cid] = arr_vy
            continue
        c_lc = vyvar_lcs[cid]
        c_sub = meas_df[meas_df["catalog_id"] == cid]
        cm = tgt_m[["source_file"]].merge(c_lc[["source_file", "psf_flux"]], on="source_file", how="left")
        cm = cm.merge(c_sub[["source_file", "pp_flux"]], on="source_file", how="left")
        pp_comp[cid] = pd.to_numeric(cm["pp_flux"], errors="coerce").to_numpy()
        vy_comp[cid] = pd.to_numeric(cm["psf_flux"], errors="coerce").to_numpy()

    pp_tgt = pd.to_numeric(tgt_m["pp_flux"], errors="coerce").to_numpy()
    pp_delta = rebuild_delta(pp_tgt, pp_comp, ens_w, ens_ids)
    vy_delta = pd.to_numeric(tgt_m["psf_delta_mag"], errors="coerce").to_numpy()
    d_tgt = vy_delta - pp_delta
    rms_tgt = rms_after_median(d_tgt)
    resid = np.asarray(d_tgt, dtype=np.float64)
    if np.isfinite(resid).any():
        resid = resid - float(np.nanmedian(resid[np.isfinite(resid)]))

    chk_pp = np.full(n_ep, np.nan)
    chk_vy = np.full(n_ep, np.nan)
    for i, sf in enumerate(tgt_m["source_file"].tolist()):
        hit = meas_df[(meas_df["catalog_id"] == CHECK_CID) & (meas_df["source_file"] == sf)]
        if hit.empty:
            continue
        if float(hit.iloc[0]["fail"]) < 0.5:
            chk_pp[i] = float(hit.iloc[0]["pp_flux"])
        chk_vy[i] = float(hit.iloc[0]["vyvar_psf_flux_live_proc"])
    pp_chk = rebuild_delta(chk_pp, pp_comp, ens_w, ens_ids)
    vy_chk = rebuild_delta(chk_vy, vy_comp, ens_w, ens_ids)
    d_chk = vy_chk - pp_chk
    rms_chk = rms_after_median(d_chk)
    resid_c = np.asarray(d_chk, dtype=np.float64)
    if np.isfinite(resid_c).any():
        resid_c = resid_c - float(np.nanmedian(resid_c[np.isfinite(resid_c)]))

    def _top(resid_arr: np.ndarray, src: pd.Series) -> list[dict[str, Any]]:
        idx = np.argsort(-np.abs(np.where(np.isfinite(resid_arr), resid_arr, 0.0)))[:8]
        out = []
        for j in idx:
            if j >= len(resid_arr) or not math.isfinite(float(resid_arr[j])):
                continue
            out.append({
                "source_file": str(src.iloc[j]),
                "resid_mag": float(resid_arr[j]),
                "resid_mmag": float(resid_arr[j]) * 1000.0,
            })
        return out

    rms_tgt_mm = rms_tgt * 1000.0 if math.isfinite(rms_tgt) else float("nan")
    rms_chk_mm = rms_chk * 1000.0 if math.isfinite(rms_chk) else float("nan")
    finite = [x for x in (rms_tgt_mm, rms_chk_mm) if math.isfinite(x)]
    worst = max(finite) if finite else float("nan")
    epoch_rows = []
    for i, sf in enumerate(tgt_m["source_file"].tolist()):
        epoch_rows.append({
            "source_file": str(sf),
            "light": _light_key(str(sf)),
            "m2_resid_target_mag": float(resid[i]) if i < len(resid) else float("nan"),
            "m2_resid_check_mag": float(resid_c[i]) if i < len(resid_c) else float("nan"),
        })
    m2 = {
        "target_cid": TARGET_CID,
        "target_rms_mag": rms_tgt,
        "target_rms_mmag": rms_tgt_mm,
        "target_n_finite": int(np.isfinite(d_tgt).sum()),
        "target_top_epochs": _top(resid, tgt_m["source_file"]),
        "check_cid": CHECK_CID,
        "check_rms_mag": rms_chk,
        "check_rms_mmag": rms_chk_mm,
        "check_n_finite": int(np.isfinite(d_chk).sum()),
        "check_top_epochs": _top(resid_c, tgt_m["source_file"]),
        "reading_applied": apply_reading_a1b(worst if math.isfinite(rms_tgt_mm) and math.isfinite(rms_chk_mm) else float("inf")),
        "reading_target": apply_reading_a1b(rms_tgt_mm),
        "reading_check": apply_reading_a1b(rms_chk_mm),
        "epochs": epoch_rows,
        "check_has_psf_lc": (live_lc / f"lightcurve_{CHECK_CID}_psf.csv").is_file(),
    }
    return m1, m2


def ha_variants() -> list[tuple[str, float, float, bool]]:
    """(name, dx, dy, swap_xy). Offsets {0, +/-0.5, +/-1} per axis + swap."""
    out: list[tuple[str, float, float, bool]] = []
    offs = (-1.0, -0.5, 0.0, 0.5, 1.0)
    for dx in offs:
        for dy in offs:
            out.append((f"dx{dx:+.1f}_dy{dy:+.1f}", dx, dy, False))
    out.append(("swap_xy", 0.0, 0.0, True))
    out.append(("swap_xy_p0.5", 0.5, 0.5, True))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="EPSF-XVAL-A1B root-cause harness")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--skip-copy", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    CTX.mkdir(parents=True, exist_ok=True)
    snapshot = REPO / "Archive" / "Drafts" / SNAPSHOT_NAME
    live_ps = REPO / "Archive" / "Drafts" / "draft_000516" / "platesolve" / SETUP
    live_lc = live_ps / "photometry" / "lightcurves"
    live_phot = live_ps / "photometry"
    live_lights = REPO / "Archive" / "Drafts" / "draft_000516" / "detrended_aligned" / "lights" / SETUP
    g4_before = g4_live_516()

    sandbox = WORK / "sandbox"
    lights_dst = sandbox / "detrended_aligned" / "lights" / SETUP
    if args.skip_copy and lights_dst.is_dir() and any(lights_dst.glob("*.fits")):
        print("[a1b] reuse existing sandbox lights")
    else:
        print("[a1b] copy frozen era04 snapshot")
        _ps, lights_dst = _copy_frozen_anchor_inputs(snapshot, sandbox)

    frames = sorted(p for p in lights_dst.glob("*.fits") if p.name.upper() != "MASTERSTAR.FITS")
    if int(args.max_frames) > 0:
        frames = frames[: int(args.max_frames)]
    print(f"[a1b] frames={len(frames)}")

    gain_info = resolve_gain_rn(live_ps / "MASTERSTAR.fits", live_phot)
    psf_ids, psf_meta = reconstruct_psf_star_ids(live_ps)
    lc_ids = list_psf_lc_ids(live_lc)
    from psf_internal_lc import resolve_ensemble_ids

    ens_ids, ens_w, ens_src = resolve_ensemble_ids(TARGET_CID, live_phot)
    measure_ids = sorted(set(lc_ids) | {TARGET_CID, CHECK_CID} | set(ens_ids))
    wanted = set(measure_ids) | set(psf_ids)
    vyvar_lcs = {cid: load_vyvar_psf_lc(live_lc / f"lightcurve_{cid}_psf.csv") for cid in lc_ids}
    catmag: dict[str, float] = {}
    fwhm_px = float(psf_meta.get("fwhm_px") or 3.3)
    fitrad = float(fwhm_px)
    psfrad = float(max(fitrad + 1.0, 8.0))
    qc_map = load_qc_fwhm(sandbox / "calibrated" / "lights" / "qc_metrics.csv")

    picks = [0, len(frames) // 4, len(frames) // 2, (3 * len(frames)) // 4, len(frames) - 1]
    ha_frames = [frames[i] for i in picks]
    variants = ha_variants()
    print(f"[a1b] H-A grid frames={[p.name for p in ha_frames]} n_var={len(variants)}")

    ha_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for fp in ha_frames:
        stem = fp.stem
        proc_name = f"proc_{stem}.csv"
        snap_proc = lights_dst / proc_name
        if not snap_proc.is_file():
            failures.append({"frame": fp.name, "stage": "ha_proc", "reason": "missing", "catalog_id": ""})
            continue
        xy = _proc_rows(snap_proc, wanted)
        psf_xy = [(cid, xy[cid]["x"], xy[cid]["y"]) for cid in psf_ids if cid in xy
                  and math.isfinite(xy[cid]["x"]) and math.isfinite(xy[cid]["y"])]
        star_base = [(cid, xy[cid]["x"], xy[cid]["y"]) for cid in measure_ids if cid in xy
                     and math.isfinite(xy[cid]["x"]) and math.isfinite(xy[cid]["y"])]
        image = np.asarray(fits.getdata(fp), dtype=np.float64)
        psf_fits = OUT.parent / "psf" / f"{stem}_ha.fits"
        base_meas, st0 = measure_frame(
            image, psf_xy=psf_xy, star_xy=star_base, gain=float(gain_info["gain"]),
            rn=float(gain_info["read_noise"]), fitrad=fitrad, psfrad=psfrad,
            psf_fits=psf_fits, recenter=False,
        )
        if not st0["ok"]:
            failures.append({"frame": fp.name, "stage": "ha_getpsf", "reason": st0["reason"], "catalog_id": ""})
            continue
        reused = st0.pop("psf_model", None)
        for vname, dx, dy, swap in variants:
            star_xy = []
            for cid, x, y in star_base:
                if swap:
                    star_xy.append((cid, y + dy, x + dx))
                else:
                    star_xy.append((cid, x + dx, y + dy))
            nore, stn = measure_frame(
                image, psf_xy=psf_xy, star_xy=star_xy, gain=float(gain_info["gain"]),
                rn=float(gain_info["read_noise"]), fitrad=fitrad, psfrad=psfrad,
                psf_fits=OUT.parent / "psf" / f"{stem}_ha_{vname}.fits", recenter=False,
                psf_model=reused,
            )
            rec, strr = measure_frame(
                image, psf_xy=psf_xy, star_xy=star_xy, gain=float(gain_info["gain"]),
                rn=float(gain_info["read_noise"]), fitrad=fitrad, psfrad=psfrad,
                psf_fits=OUT.parent / "psf" / f"{stem}_ha_{vname}_r.fits", recenter=True,
                psf_model=reused,
            )
            stn.pop("psf_model", None)
            strr.pop("psf_model", None)
            if not stn["ok"]:
                failures.append({"frame": fp.name, "stage": "ha_nore", "reason": stn["reason"], "catalog_id": vname})
            if not strr["ok"]:
                failures.append({"frame": fp.name, "stage": "ha_rec", "reason": strr["reason"], "catalog_id": vname})
            for cid, x0, y0 in star_base:
                b = base_meas.get(cid, {})
                nrec = nore.get(cid, {})
                rrec = rec.get(cid, {})
                x_start = (y0 + dy) if swap else (x0 + dx)
                y_start = (x0 + dx) if swap else (y0 + dy)
                dx_r = float(rrec.get("x_fit", np.nan)) - float(x_start)
                dy_r = float(rrec.get("y_fit", np.nan)) - float(y_start)
                f0 = b.get("flux")
                fr = nrec.get("flux")
                dmag = float("nan")
                try:
                    if f0 and fr and float(f0) > 0 and float(fr) > 0:
                        dmag = -2.5 * math.log10(float(fr) / float(f0))
                except (TypeError, ValueError):
                    dmag = float("nan")
                ha_rows.append({
                    "fits": fp.name,
                    "variant": vname,
                    "dx": dx,
                    "dy": dy,
                    "swap_xy": int(swap),
                    "catalog_id": cid,
                    "shift_px": float(math.hypot(dx_r, dy_r)) if math.isfinite(dx_r) and math.isfinite(dy_r) else float("nan"),
                    "abs_dx": abs(dx_r) if math.isfinite(dx_r) else float("nan"),
                    "abs_dy": abs(dy_r) if math.isfinite(dy_r) else float("nan"),
                    "dmag_vs_a1": dmag,
                    "fail_nore": nrec.get("fail", 1.0),
                    "fail_rec": rrec.get("fail", 1.0),
                })
        print(f"[a1b] H-A done {fp.name}")

    ha_df = pd.DataFrame(ha_rows)
    ha_sum_rows = []
    if not ha_df.empty:
        for vname, g in ha_df.groupby("variant"):
            finite = g[np.isfinite(g["shift_px"])]
            ok = finite[(finite["fail_rec"] < 0.5)] if "fail_rec" in finite.columns else finite
            ha_sum_rows.append({
                "variant": vname,
                "n": int(len(finite)),
                "n_ok": int(len(ok)),
                "n_fail": int((g["fail_rec"] >= 0.5).sum()) if "fail_rec" in g.columns else 0,
                "median_shift_px": float(np.median(ok["shift_px"])) if not ok.empty else float("nan"),
                "median_abs_dx": float(np.median(ok["abs_dx"])) if not ok.empty else float("nan"),
                "median_abs_dy": float(np.median(ok["abs_dy"])) if not ok.empty else float("nan"),
                "median_abs_dmag_vs_a1": float(np.nanmedian(np.abs(g["dmag_vs_a1"]))) if len(g) else float("nan"),
                "median_shift_incl_fail": float(np.median(finite["shift_px"])) if not finite.empty else float("nan"),
            })
    ha_sum = pd.DataFrame(ha_sum_rows).sort_values("median_shift_px") if ha_sum_rows else pd.DataFrame()
    ha_df.to_csv(OUT / "ha_pairs.csv", index=False)
    ha_df.to_csv(CTX / "ha_pairs.csv", index=False)
    ha_sum.to_csv(OUT / "ha_summary.csv", index=False)
    ha_sum.to_csv(CTX / "ha_summary.csv", index=False)
    print("[a1b] H-A summary")
    print(ha_sum.to_string(index=False) if not ha_sum.empty else "empty")

    # Full native recentering rerun
    print("[a1b] full recenter rerun")
    rows: list[dict[str, Any]] = []
    hb_frame: list[dict[str, Any]] = []
    for i_fr, fp in enumerate(frames):
        t_fr = time.perf_counter()
        stem = fp.stem
        proc_name = f"proc_{stem}.csv"
        snap_proc = lights_dst / proc_name
        if not snap_proc.is_file():
            failures.append({"frame": fp.name, "stage": "proc", "reason": "snapshot_proc_missing", "catalog_id": ""})
            continue
        xy = _proc_rows(snap_proc, wanted)
        psf_xy = [(cid, xy[cid]["x"], xy[cid]["y"]) for cid in psf_ids if cid in xy
                  and math.isfinite(xy[cid]["x"]) and math.isfinite(xy[cid]["y"])]
        star_xy = [(cid, xy[cid]["x"], xy[cid]["y"]) for cid in measure_ids if cid in xy
                   and math.isfinite(xy[cid]["x"]) and math.isfinite(xy[cid]["y"])]
        for cid in measure_ids:
            if cid not in xy:
                failures.append({"frame": fp.name, "stage": "position", "reason": "missing_in_proc", "catalog_id": cid})
            elif cid in xy:
                cm = xy[cid].get("catalog_mag", float("nan"))
                if cid not in catmag and math.isfinite(cm):
                    catmag[cid] = cm
        try:
            image = np.asarray(fits.getdata(fp), dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            failures.append({"frame": fp.name, "stage": "fits", "reason": f"{type(exc).__name__}:{exc}", "catalog_id": ""})
            continue
        ny, nx = image.shape[0], image.shape[1]
        cx, cy = 0.5 * (nx - 1), 0.5 * (ny - 1)
        meas, status = measure_frame(
            image, psf_xy=psf_xy, star_xy=star_xy, gain=float(gain_info["gain"]),
            rn=float(gain_info["read_noise"]), fitrad=fitrad, psfrad=psfrad,
            psf_fits=OUT.parent / "psf" / f"{stem}_rec.fits", recenter=True,
        )
        status.pop("psf_model", None)
        if not status["ok"]:
            failures.append({"frame": fp.name, "stage": "getpsf", "reason": status["reason"], "catalog_id": ""})
            print(f"[a1b] FAIL {fp.name} {status['reason']}")
            continue
        live_proc = live_lights / proc_name
        live_xy = _proc_rows(live_proc, wanted) if live_proc.is_file() else {}
        dxs, dys = [], []
        for cid, rec in meas.items():
            reason = str(rec.get("_reason") or "")
            if rec.get("fail", 1.0) >= 0.5:
                failures.append({"frame": fp.name, "stage": "pkfit", "reason": reason or "pkfit_fail", "catalog_id": cid})
            dx = float(rec.get("x_fit", np.nan)) - float(rec.get("x", np.nan))
            dy = float(rec.get("y_fit", np.nan)) - float(rec.get("y", np.nan))
            if math.isfinite(dx) and math.isfinite(dy) and rec.get("fail", 1) < 0.5:
                dxs.append(dx)
                dys.append(dy)
            x0 = float(rec.get("x", np.nan))
            y0 = float(rec.get("y", np.nan))
            r_field = float(math.hypot(x0 - cx, y0 - cy)) if math.isfinite(x0) and math.isfinite(y0) else float("nan")
            rows.append({
                "source_file": proc_name,
                "fits": fp.name,
                "catalog_id": cid,
                "pp_flux": rec.get("flux"),
                "pp_scale": rec.get("scale"),
                "pp_chi": rec.get("chi"),
                "pp_niter": rec.get("niter"),
                "pp_x": rec.get("x"),
                "pp_y": rec.get("y"),
                "pp_x_fit": rec.get("x_fit"),
                "pp_y_fit": rec.get("y_fit"),
                "dx_fit": dx,
                "dy_fit": dy,
                "r_field_px": r_field,
                "vyvar_psf_flux_live_proc": live_xy.get(cid, {}).get("psf_flux", float("nan")),
                "fail": rec.get("fail"),
                "fail_reason": reason,
            })
        if dxs:
            hb_frame.append({
                "fits": fp.name,
                "n": len(dxs),
                "median_dx": float(np.median(dxs)),
                "median_dy": float(np.median(dys)),
                "median_abs_dx": float(np.median(np.abs(dxs))),
                "median_abs_dy": float(np.median(np.abs(dys))),
                "median_shift_px": float(np.median(np.hypot(dxs, dys))),
            })
        print(f"[a1b] {i_fr+1}/{len(frames)} {fp.name} n_psf={status.get('n_psf_used')} dt={time.perf_counter()-t_fr:.1f}s")

    meas_df = pd.DataFrame(rows)
    if meas_df.empty:
        meas_df = pd.DataFrame(columns=["source_file", "catalog_id", "pp_flux", "fail"])
    meas_df.to_csv(OUT / "pp_fluxes_recenter.csv", index=False)
    hb_df = pd.DataFrame(hb_frame)
    hb_df.to_csv(OUT / "hb_per_frame.csv", index=False)
    hb_df.to_csv(CTX / "hb_per_frame.csv", index=False)

    m1, m2 = assemble_m1_m2(meas_df, vyvar_lcs, lc_ids, ens_ids, ens_w, catmag, live_lc)
    m1.to_csv(OUT / "m1_recenter.csv", index=False)
    m1.to_csv(CTX / "m1_recenter.csv", index=False)

    # H-C: per-star median offset vs field position
    hc_rows = []
    okm = meas_df[(meas_df["fail"] < 0.5) & np.isfinite(meas_df.get("dx_fit", pd.Series(dtype=float)))]
    for cid, g in okm.groupby("catalog_id"):
        dxm = float(np.nanmedian(g["dx_fit"]))
        dym = float(np.nanmedian(g["dy_fit"]))
        xm = float(np.nanmedian(g["pp_x"]))
        ym = float(np.nanmedian(g["pp_y"]))
        rm = float(np.nanmedian(g["r_field_px"]))
        sub = m1[m1["catalog_id"] == cid]
        med_d = float(sub.iloc[0]["median_d_mag"]) if not sub.empty else float("nan")
        hc_rows.append({
            "catalog_id": cid,
            "x_med": xm,
            "y_med": ym,
            "r_field_px": rm,
            "median_dx": dxm,
            "median_dy": dym,
            "median_shift_px": float(math.hypot(dxm, dym)) if math.isfinite(dxm) and math.isfinite(dym) else float("nan"),
            "median_d_mag": med_d,
            "catalog_mag": catmag.get(str(cid), float("nan")),
        })
    hc = pd.DataFrame(hc_rows)
    hc.to_csv(OUT / "hc_per_star.csv", index=False)
    hc.to_csv(CTX / "hc_per_star.csv", index=False)
    hc_corr: dict[str, Any] = {}
    if not hc.empty:
        for xcol, ycol in (("r_field_px", "median_d_mag"), ("r_field_px", "median_shift_px"),
                           ("x_med", "median_d_mag"), ("y_med", "median_d_mag")):
            a = pd.to_numeric(hc[xcol], errors="coerce").to_numpy()
            b = pd.to_numeric(hc[ycol], errors="coerce").to_numpy()
            msk = np.isfinite(a) & np.isfinite(b)
            if int(msk.sum()) >= 5:
                pr = stats.pearsonr(a[msk], b[msk])
                sr = stats.spearmanr(a[msk], b[msk])
                hc_corr[f"{xcol}__{ycol}"] = {
                    "n": int(msk.sum()),
                    "pearson_r": float(pr.statistic),
                    "pearson_p": float(pr.pvalue),
                    "spearman_r": float(sr.statistic),
                    "spearman_p": float(sr.pvalue),
                }
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7.2, 5.4))
            sc = ax.scatter(
                hc["x_med"], hc["y_med"], c=hc["median_d_mag"],
                cmap="coolwarm", s=36, edgecolors="k", linewidths=0.3,
            )
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("per-star median d mag (VYVAR - PP)")
            ax.set_xlabel("x (proc, px)")
            ax.set_ylabel("y (proc, px)")
            ax.set_title("H-C: per-star median mag residual vs field position")
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()
            fig.savefig(OUT / "hc_residual_map.png", dpi=120)
            fig.savefig(CTX / "hc_residual_map.png", dpi=120)
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            hc_corr["plot_error"] = str(exc)

    # H-D: M2 residual vs FWHM
    ep = pd.DataFrame(m2.pop("epochs"))
    ep["fwhm_px"] = ep["light"].map(lambda k: qc_map.get(str(k), float("nan")))
    ep.to_csv(OUT / "hd_epochs.csv", index=False)
    ep.to_csv(CTX / "hd_epochs.csv", index=False)
    hd: dict[str, Any] = {}
    for col, label in (("m2_resid_target_mag", "target"), ("m2_resid_check_mag", "check")):
        a = pd.to_numeric(ep["fwhm_px"], errors="coerce").to_numpy()
        b = pd.to_numeric(ep[col], errors="coerce").to_numpy()
        msk = np.isfinite(a) & np.isfinite(b)
        if int(msk.sum()) >= 5:
            pr = stats.pearsonr(a[msk], b[msk])
            sr = stats.spearmanr(a[msk], b[msk])
            hd[label] = {
                "n": int(msk.sum()),
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "spearman_r": float(sr.statistic),
                "spearman_p": float(sr.pvalue),
            }
        else:
            hd[label] = {"n": int(msk.sum())}

    fail_df = pd.DataFrame(failures)
    if fail_df.empty:
        fail_df = pd.DataFrame(columns=["frame", "stage", "reason", "catalog_id"])
    fail_df.to_csv(OUT / "failures.csv", index=False)
    fail_df.to_csv(CTX / "failures.csv", index=False)

    g4_after = g4_live_516()
    hb_all = {
        "n_frames": int(len(hb_df)),
        "median_of_frame_median_abs_dx": float(hb_df["median_abs_dx"].median()) if not hb_df.empty else float("nan"),
        "median_of_frame_median_abs_dy": float(hb_df["median_abs_dy"].median()) if not hb_df.empty else float("nan"),
        "median_of_frame_median_shift": float(hb_df["median_shift_px"].median()) if not hb_df.empty else float("nan"),
    }
    _ha_nok = ha_sum["n_ok"] if (not ha_sum.empty and "n_ok" in ha_sum.columns) else (
        ha_sum["n"] if (not ha_sum.empty and "n" in ha_sum.columns) else pd.Series(dtype=float)
    )
    payload = {
        "task": "EPSF-XVAL-A1B-ROOTCAUSE-01",
        "runtime_s": time.perf_counter() - t0,
        "n_frames": len(frames),
        "gain": gain_info,
        "psf_pool_meta": psf_meta,
        "ensemble": {"ids": ens_ids, "weights": ens_w, "source": ens_src},
        "ha_best": (
            ha_sum[(_ha_nok >= 200) & (ha_sum["median_abs_dmag_vs_a1"] < 1.0)]
            .sort_values("median_shift_px").iloc[0].to_dict()
            if (not ha_sum.empty and ((_ha_nok >= 200) & (ha_sum["median_abs_dmag_vs_a1"] < 1.0)).any())
            else (ha_sum.iloc[0].to_dict() if not ha_sum.empty else {})
        ),
        "ha_identity": (
            ha_sum[ha_sum["variant"] == "dx+0.0_dy+0.0"].iloc[0].to_dict()
            if (not ha_sum.empty and (ha_sum["variant"] == "dx+0.0_dy+0.0").any())
            else {}
        ),
        "m1": m1.to_dict(orient="records"),
        "m2": m2,
        "hb": hb_all,
        "hc_corr": hc_corr,
        "hd": hd,
        "failures_n": int(len(fail_df)),
        "failures_by_stage": fail_df["stage"].value_counts().to_dict() if not fail_df.empty else {},
        "g4_before": g4_before,
        "g4_after": g4_after,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (CTX / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "runtime_s": payload["runtime_s"],
        "m2_target_mmag": m2["target_rms_mmag"],
        "m2_check_mmag": m2["check_rms_mmag"],
        "reading": m2["reading_applied"],
        "hb": hb_all,
        "ha_best": payload["ha_best"],
        "hd": hd,
        "failures_n": payload["failures_n"],
        "g4": g4_after["pass"],
    }, indent=2))
    return 0 if g4_after["pass"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception:
        traceback.print_exc()
        raise
