# -*- coding: ascii -*-
"""IDENT-JUMP-01 M3: remasure LCs at fixed master positions vs current jumped proc."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))

from aperture_pertarget import abbe_p2p_scatter, _aperture_flux_uniform  # noqa: E402
from aperture_policy import resolve_aperture_geometry  # noqa: E402

OUT = ROOT / "dev" / "results" / "context" / "session_20261001_ident_jump_01"
DRAFT = ROOT / "Archive" / "Drafts" / "draft_000521"
SETUP = "NoFilter_60_2"
FRAMES = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOTO = DRAFT / "platesolve" / SETUP / "photometry"
V1023 = "1403049512185012992"
F = 1.35
ANN_IN, ANN_OUT = 2.7, 5.2


def rms_mmag(mag: np.ndarray) -> float:
    m = np.asarray(mag, dtype=float)
    m = m[np.isfinite(m)]
    if m.size < 3:
        return float("nan")
    return float(np.std(m) * 1000.0)


def equal_weight_delta(t_mag: np.ndarray, comp_mags: dict[str, np.ndarray]) -> np.ndarray:
    t = np.asarray(t_mag, dtype=float)
    n = len(t)
    series = [np.asarray(v, dtype=float) for v in comp_mags.values() if len(v) == n]
    if not series:
        return np.full(n, np.nan)
    stack = np.vstack(series)
    out = np.full(n, np.nan)
    for i in range(n):
        if not math.isfinite(t[i]):
            continue
        cols = stack[:, i]
        ok = np.isfinite(cols)
        if ok.sum() < 1:
            continue
        flux_sum = float(np.sum(10.0 ** (-0.4 * cols[ok])))
        if flux_sum <= 0:
            continue
        out[i] = t[i] - (-2.5 * math.log10(flux_sum))
    return out


def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    f = np.asarray(flux, dtype=float)
    out = np.full(f.shape, np.nan)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    at = pd.read_csv(PHOTO / "active_targets.csv", dtype={"catalog_id": str})
    comps = pd.read_csv(
        PHOTO / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    # master xy
    ms = pd.read_csv(
        DRAFT / "platesolve" / SETUP / "masterstars_full_match.csv",
        dtype={"catalog_id": str},
        low_memory=False,
    )
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    xy_map = {
        str(r.catalog_id): (float(r.x), float(r.y))
        for r in ms.itertuples()
        if math.isfinite(float(r.x)) and math.isfinite(float(r.y))
    }
    # pick V1023 + 5 worst jumpers that have comps
    jumps = pd.read_csv(OUT / "per_target_jumps_521.csv", dtype={"catalog_id": str})
    with_comps = set(comps["target_catalog_id"].astype(str))
    picks = [V1023] if V1023 in xy_map else []
    for cid in jumps.sort_values("n_gt8", ascending=False)["catalog_id"]:
        if cid in picks:
            continue
        if cid in with_comps and cid in xy_map:
            picks.append(cid)
        if len(picks) >= 6:
            break
    # also ensure V1023 comps exist
    if V1023 not in with_comps:
        # use nearest name match from active_targets
        pass

    # FWHM from policy or header median
    pol = PHOTO / "aperture_policy.json"
    fwhm = 5.492
    if pol.is_file():
        fwhm = float(json.loads(pol.read_text(encoding="utf-8")).get("fwhm_night_median_px") or fwhm)
    r_ap, r_in, r_out = resolve_aperture_geometry(
        f=F, fwhm_px=fwhm, annulus_inner_fwhm=ANN_IN, annulus_outer_fwhm=ANN_OUT
    )

    lights = sorted(p for p in FRAMES.glob("*.fits") if p.stem.upper() != "MASTERSTAR")
    stems = [p.stem for p in lights]

    # load existing LC flags if present
    flag_counts_before = {}
    lc_dir = PHOTO / "lightcurves"
    results = []
    for tid in picks:
        cids = (
            comps.loc[comps["target_catalog_id"].astype(str) == tid, "catalog_id"]
            .astype(str)
            .tolist()
        )
        if not cids:
            # try without comps: skip
            results.append({"catalog_id": tid, "ok": False, "reason": "no_comps"})
            continue
        star_ids = [tid] + cids
        # current positions from proc
        cur_flux = {cid: np.full(len(lights), np.nan) for cid in star_ids}
        fix_flux = {cid: np.full(len(lights), np.nan) for cid in star_ids}
        n_jump = 0
        for i, (fp, stem) in enumerate(zip(lights, stems)):
            proc = FRAMES / f"proc_{stem}.csv"
            if not proc.is_file():
                continue
            pdf = pd.read_csv(proc, dtype={"catalog_id": str}, low_memory=False)
            pdf["catalog_id"] = pdf["catalog_id"].astype(str).str.strip()
            with fits.open(fp, memmap=True) as hdul:
                img = np.asarray(hdul[0].data, dtype=float)
            # current
            pos_cur = []
            ids_ok = []
            for cid in star_ids:
                row = pdf[pdf["catalog_id"] == cid]
                if row.empty:
                    continue
                xc = float(pd.to_numeric(row.iloc[0]["x"], errors="coerce"))
                yc = float(pd.to_numeric(row.iloc[0]["y"], errors="coerce"))
                if not (math.isfinite(xc) and math.isfinite(yc)):
                    continue
                xref, yref = xy_map[cid]
                if math.hypot(xc - xref, yc - yref) > 3:
                    n_jump += 1
                pos_cur.append((xc, yc))
                ids_ok.append(cid)
            if pos_cur:
                fl = _aperture_flux_uniform(img, np.asarray(pos_cur), r_ap, r_in, r_out)
                for j, cid in enumerate(ids_ok):
                    cur_flux[cid][i] = float(fl[j])
            # fixed master
            pos_fix = []
            ids_fix = []
            for cid in star_ids:
                if cid not in xy_map:
                    continue
                pos_fix.append(xy_map[cid])
                ids_fix.append(cid)
            if pos_fix:
                fl = _aperture_flux_uniform(img, np.asarray(pos_fix, dtype=float), r_ap, r_in, r_out)
                for j, cid in enumerate(ids_fix):
                    fix_flux[cid][i] = float(fl[j])

        t_cur = flux_to_mag(cur_flux[tid])
        t_fix = flux_to_mag(fix_flux[tid])
        c_cur = {c: flux_to_mag(cur_flux[c]) for c in cids}
        c_fix = {c: flux_to_mag(fix_flux[c]) for c in cids}
        d_cur = equal_weight_delta(t_cur, c_cur)
        d_fix = equal_weight_delta(t_fix, c_fix)
        # demean for RMS of scatter
        def _demean(a):
            a = np.asarray(a, dtype=float)
            ok = np.isfinite(a)
            if ok.sum() < 3:
                return a
            out = a.copy()
            out[ok] = a[ok] - np.median(a[ok])
            return out

        name = ""
        if tid in set(at["catalog_id"].astype(str)):
            name = str(at.loc[at["catalog_id"].astype(str) == tid, "vsx_name"].iloc[0] or "")

        # LC-OUTLIER flags on existing LC if any
        flag_n = {"artifact": 0, "spike_unconfirmed": 0}
        for p in lc_dir.glob("*.csv") if lc_dir.is_dir() else []:
            try:
                hdf = pd.read_csv(p, nrows=3, dtype=str)
                if "catalog_id" in hdf.columns and str(hdf["catalog_id"].iloc[0]).strip() == tid:
                    full = pd.read_csv(p, dtype=str, low_memory=False)
                    if "flag" in full.columns:
                        flg = full["flag"].fillna("").astype(str).str.lower()
                        flag_n["artifact"] = int(flg.str.contains("artifact").sum())
                        flag_n["spike_unconfirmed"] = int(flg.str.contains("spike_unconfirmed").sum())
                    break
            except Exception:
                continue

        results.append(
            {
                "catalog_id": tid,
                "vsx_name": name,
                "n_comps": len(cids),
                "n_frames": len(lights),
                "n_position_jump_events": n_jump,
                "rms_cur_mmag": rms_mmag(_demean(d_cur)),
                "rms_fix_mmag": rms_mmag(_demean(d_fix)),
                "p2p_cur_mmag": abbe_p2p_scatter(d_cur) * 1000.0
                if math.isfinite(abbe_p2p_scatter(d_cur))
                else None,
                "p2p_fix_mmag": abbe_p2p_scatter(d_fix) * 1000.0
                if math.isfinite(abbe_p2p_scatter(d_fix))
                else None,
                "lc_flags_artifact": flag_n["artifact"],
                "lc_flags_spike_unconfirmed": flag_n["spike_unconfirmed"],
                "r_ap_px": r_ap,
            }
        )
        print(results[-1], flush=True)

    pd.DataFrame(results).to_csv(OUT / "m3_lc_before_after.csv", index=False)
    (OUT / "m3_summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
