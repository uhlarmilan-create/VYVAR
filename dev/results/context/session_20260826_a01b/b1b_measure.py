# -*- coding: ascii -*-
"""APERTURE-01b: measure f by accuracy (colour-flat EE + COG + AIJ), not scatter.

Premise (Rule 0.1): compared is enclosed-energy and photometric accuracy as a
function of aperture factor f (mode a: r = f x night-median QC FWHM) versus
the APERTURE-01 scatter-ladder f=0.385. Differ: B1/B2 select f from colour-flat
EE, COG level vs f=2.5, and AIJ RMS; scatter RMS is a tie-break only.

B1: EE(r) on the 516 frame set for comp-pool BP-RP quartiles Q1/Q4.
    continuous r = 0.50..3.00 FWHM step 0.05 (not a 0.5-px grid).
B2: BO/FW/GH harness on the f report grid (no lock).
B3: f* rule applied here; not relaxed if empty.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.aperture import aperture_photometry as _aphot
from scipy.spatial import cKDTree
from scipy.stats import linregress, spearmanr

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))

from aperture_policy import (  # noqa: E402
    FWHM_AUTHORITY,
    load_qc_fwhm_map,
    resolve_aperture_geometry,
)
from aperture_scatter_select import differential_mag_series, flux_to_inst_mag  # noqa: E402
from masterstar_gaia_accounting import _norm_cid  # noqa: E402
from photometry_core import _sky_pp_from_annulus_image  # noqa: E402

ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
OUT = ROOT / "dev" / "results" / "context" / "session_20260826_a01b"
AIJ_BO = ROOT / "dev" / "results" / "XVAL_AIJ_01_bo_compare.csv"

ANN_IN = 4.75
ANN_OUT = 9.0
ISO_FWHM = 3.0
EE_F_MIN = 0.50
EE_F_MAX = 3.00
EE_F_STEP = 0.05
F_REPORT = (0.75, 1.0, 1.25, 1.35, 1.5, 1.75, 2.0, 2.5)
F_REF = 2.5

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


def _cid(v: object) -> str:
    s = _norm_cid(v)
    return s if s else str(v).strip()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def frame_key(s: str) -> str:
    name = Path(str(s)).name
    stem = Path(name).stem
    if stem.startswith("proc_"):
        stem = stem[5:]
    return stem.lower()


def demeaned_rms_mmag(mag: np.ndarray) -> float:
    x = np.asarray(mag, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float("nan")
    d = x - float(np.median(x))
    return float(np.sqrt(np.mean(d * d)) * 1000.0)


def nearest_finite(grid: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(grid - float(value))))


def main() -> int:
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    qc_path = ERA04 / "calibrated" / "lights" / "qc_metrics.csv"
    _qc_map, night = load_qc_fwhm_map(qc_path)
    if night is None or not math.isfinite(float(night)):
        night = 5.1917332681208865
    night_f = float(night)

    ee_f = np.arange(EE_F_MIN, EE_F_MAX + 1e-9, EE_F_STEP, dtype=np.float64)
    f_report = np.asarray(F_REPORT, dtype=np.float64)
    radii_f = ee_f
    radii_px = radii_f * night_f
    _r_ap0, r_in, r_out = resolve_aperture_geometry(
        f=1.0, fwhm_px=night_f, annulus_inner_fwhm=ANN_IN, annulus_outer_fwhm=ANN_OUT
    )

    ms = pd.read_csv(
        ERA04 / "platesolve" / SETUP / "masterstars_full_match.csv",
        dtype={"catalog_id": str},
        low_memory=False,
    )
    ms["catalog_id"] = ms["catalog_id"].map(_cid)
    pool = pd.read_csv(
        ERA04 / "platesolve" / SETUP / "comparison_stars.csv",
        dtype={"catalog_id": str},
        low_memory=False,
    )
    pool["catalog_id"] = pool["catalog_id"].map(_cid)

    xy_all = {}
    for _, row in ms.iterrows():
        try:
            xy_all[str(row["catalog_id"])] = (float(row["x"]), float(row["y"]))
        except (TypeError, ValueError):
            continue
    tree = cKDTree(np.array(list(xy_all.values()), dtype=np.float64)) if xy_all else None
    iso_px = ISO_FWHM * night_f

    want: list[str] = []
    for tid, comps in GROUPS.values():
        want.append(tid)
        want.extend(comps)
    want.extend(str(c) for c in pool["catalog_id"].tolist())
    want = [c for c in dict.fromkeys(want) if c in xy_all]

    meta = pool.drop_duplicates(subset=["catalog_id"]).set_index("catalog_id")
    pool_ok = []
    for cid in pool["catalog_id"].tolist():
        cid = str(cid)
        if cid not in xy_all or cid not in meta.index:
            continue
        try:
            bp = float(meta.at[cid, "bp_rp"])
        except (TypeError, ValueError, KeyError):
            continue
        if not math.isfinite(bp):
            continue
        zone = str(meta.at[cid, "zone"] if "zone" in meta.columns else "")
        sat = bool(meta.at[cid, "is_saturated"]) if "is_saturated" in meta.columns else False
        if zone and zone.strip().lower() != "linear":
            continue
        if sat:
            continue
        pool_ok.append(cid)
    pool_ok = list(dict.fromkeys(pool_ok))
    bprp = np.array([float(meta.at[c, "bp_rp"]) for c in pool_ok], dtype=np.float64)
    q_cuts = np.quantile(bprp, [0.25, 0.50, 0.75])
    q1_ids = [c for c, b in zip(pool_ok, bprp) if b <= q_cuts[0]]
    q4_ids = [c for c, b in zip(pool_ok, bprp) if b >= q_cuts[2]]

    def is_isolated(cid: str) -> bool:
        if tree is None:
            return False
        xy = np.asarray(xy_all[cid], dtype=np.float64)
        dist, _idx = tree.query(xy, k=2)
        nn = float(dist[1]) if np.ndim(dist) else float("nan")
        return math.isfinite(nn) and nn > iso_px

    q1_iso = [c for c in q1_ids if is_isolated(c)]
    q4_iso = [c for c in q4_ids if is_isolated(c)]

    lights = sorted(
        p
        for p in (ERA04 / "detrended_aligned" / "lights" / SETUP).glob("*.fits")
        if p.stem.upper() != "MASTERSTAR"
    )
    qc_df = pd.read_csv(qc_path, low_memory=False)
    qc_df["key"] = qc_df["src"].map(frame_key)
    qc_fwhm = {}
    for k, v in zip(qc_df["key"], pd.to_numeric(qc_df["fwhm_px"], errors="coerce")):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            qc_fwhm[str(k)] = fv

    ids = want
    n_star = len(ids)
    n_r = int(radii_f.size)
    pos = np.array([xy_all[c] for c in ids], dtype=np.float64)
    id_index = {c: i for i, c in enumerate(ids)}

    with fits.open(lights[0], memmap=False) as hdul:
        ny, nx = np.asarray(hdul[0].data).shape
    edge = float(r_out)
    onchip = (
        (pos[:, 0] >= edge)
        & (pos[:, 0] < (nx - edge))
        & (pos[:, 1] >= edge)
        & (pos[:, 1] < (ny - edge))
    )

    flux = np.full((len(lights), n_star, n_r), np.nan, dtype=np.float64)
    fwhm_frames = np.full(len(lights), np.nan, dtype=np.float64)
    frame_keys: list[str] = []

    for fi, fp in enumerate(lights):
        with fits.open(fp, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
        key = frame_key(fp.name)
        frame_keys.append(key)
        fwhm_frames[fi] = float(qc_fwhm.get(key, night_f))
        sky = np.full(n_star, np.nan, dtype=np.float64)
        try:
            an = CircularAnnulus(pos, r_in=float(r_in), r_out=float(r_out))
            masks = an.to_mask(method="center")
            if not isinstance(masks, (list, tuple)):
                masks = [masks]
            for i, m in enumerate(masks):
                if not bool(onchip[i]):
                    continue
                try:
                    ann_img = m.to_image(data.shape)
                    sky[i] = _sky_pp_from_annulus_image(data, ann_img)
                except Exception:  # noqa: BLE001
                    sky[i] = float("nan")
        except Exception:  # noqa: BLE001
            sky[:] = np.nan
        for ri, rpx in enumerate(radii_px):
            try:
                ap = CircularAperture(pos, r=float(rpx))
                phot = _aphot(data, ap, method="exact")
                sums = np.asarray(phot["aperture_sum"], dtype=np.float64)
                area = float(ap.area)
                fl = sums - sky * area
                fl[~onchip] = np.nan
                flux[fi, :, ri] = fl
            except Exception:  # noqa: BLE001
                continue
        if (fi + 1) % 20 == 0 or fi == 0:
            print("frame %d/%d elapsed=%.1fs" % (fi + 1, len(lights), time.perf_counter() - t0), flush=True)

    i_norm = nearest_finite(radii_f, EE_F_MAX)

    def quartile_ee(member_ids: list[str]) -> np.ndarray:
        idx = [id_index[c] for c in member_ids if c in id_index]
        if not idx:
            return np.full(n_r, np.nan, dtype=np.float64)
        sub = flux[:, idx, :]
        norm = sub[:, :, i_norm]
        ee = sub / norm[:, :, None]
        bad = ~(np.isfinite(norm) & (norm > 0))
        ee[bad, :] = np.nan
        per_frame = np.nanmedian(ee, axis=1)
        return np.nanmedian(per_frame, axis=0)

    ee_q1 = quartile_ee(q1_iso)
    ee_q4 = quartile_ee(q4_iso)
    dee = np.abs(ee_q1 - ee_q4)

    b1_rows = []
    for fval in f_report:
        i = nearest_finite(radii_f, float(fval))
        b1_rows.append(
            {
                "f": float(fval),
                "r_px": round(float(radii_px[i]), 4),
                "ee_q1": None if not math.isfinite(ee_q1[i]) else round(float(ee_q1[i]), 6),
                "ee_q4": None if not math.isfinite(ee_q4[i]) else round(float(ee_q4[i]), 6),
                "dee": None if not math.isfinite(dee[i]) else round(float(dee[i]), 6),
                "dee_lt_0p01": bool(math.isfinite(dee[i]) and float(dee[i]) < 0.01),
            }
        )
    b1_curve = []
    for i, fval in enumerate(radii_f):
        b1_curve.append(
            {
                "f": round(float(fval), 4),
                "r_px": round(float(radii_px[i]), 4),
                "ee_q1": None if not math.isfinite(ee_q1[i]) else round(float(ee_q1[i]), 6),
                "ee_q4": None if not math.isfinite(ee_q4[i]) else round(float(ee_q4[i]), 6),
                "dee": None if not math.isfinite(dee[i]) else round(float(dee[i]), 6),
            }
        )

    g_map = {}
    bp_map = {}
    for cid in ids:
        if cid in meta.index:
            try:
                g_map[cid] = float(meta.at[cid, "phot_g_mean_mag"])
            except (TypeError, ValueError, KeyError):
                g_map[cid] = float("nan")
            try:
                bp_map[cid] = float(meta.at[cid, "bp_rp"])
            except (TypeError, ValueError, KeyError):
                bp_map[cid] = float("nan")
        else:
            hit = ms.loc[ms["catalog_id"] == cid]
            g_map[cid] = float("nan")
            bp_map[cid] = float("nan")
            if not hit.empty:
                try:
                    g_map[cid] = float(hit.iloc[0]["phot_g_mean_mag"])
                except (TypeError, ValueError, KeyError):
                    pass
                try:
                    bp_map[cid] = float(hit.iloc[0]["bp_rp"])
                except (TypeError, ValueError, KeyError):
                    pass

    aij_tbl = None
    aij_sha = None
    if AIJ_BO.is_file():
        aij_sha = sha256_file(AIJ_BO)
        aij_tbl = pd.read_csv(AIJ_BO)
        aij_tbl["Label"] = aij_tbl["Label"].map(frame_key)

    i_ref = nearest_finite(radii_f, F_REF)
    b2_rows = []
    magcalib_by_f: dict[str, dict[str, np.ndarray]] = {}
    for fval in f_report:
        i = nearest_finite(radii_f, float(fval))
        mag_inst = flux_to_inst_mag(flux[:, :, i])
        zp = np.full(len(lights), np.nan, dtype=np.float64)
        pool_idx = [id_index[c] for c in pool_ok if c in id_index]
        for fi in range(len(lights)):
            diffs = []
            for pi in pool_idx:
                cid = ids[pi]
                g = g_map.get(cid, float("nan"))
                m = mag_inst[fi, pi]
                if math.isfinite(g) and math.isfinite(float(m)):
                    diffs.append(g - float(m))
            if diffs:
                zp[fi] = float(np.median(np.asarray(diffs, dtype=np.float64)))
        mag_calib = mag_inst + zp[:, None]

        stars_out = {}
        rms_list = []
        for name, (tid, comps) in GROUPS.items():
            ti = id_index[tid]
            cflux = {
                c: flux[:, id_index[c], i]
                for c in comps
                if c in id_index
            }
            dmag = differential_mag_series(flux[:, ti, i], cflux)
            rms = demeaned_rms_mmag(dmag)
            cal = mag_calib[:, ti]
            level = float(np.nanmedian(cal))
            d0 = cal - level
            ok = np.isfinite(d0) & np.isfinite(fwhm_frames)
            rho = float("nan")
            if int(ok.sum()) >= 8:
                rho = float(spearmanr(d0[ok], fwhm_frames[ok]).statistic)
            stars_out[name] = {
                "level_mag": None if not math.isfinite(level) else round(level, 6),
                "rms_mmag": None if not math.isfinite(rms) else round(float(rms), 4),
                "spearman_abs_rho": None if not math.isfinite(rho) else round(abs(rho), 4),
                "spearman_rho": None if not math.isfinite(rho) else round(rho, 4),
                "n_frames": int(np.isfinite(cal).sum()),
            }
            if math.isfinite(rms):
                rms_list.append(float(rms))
            magcalib_by_f.setdefault(name, {})[str(fval)] = cal

        xs = []
        ys = []
        for cid in pool_ok:
            if cid not in id_index:
                continue
            mi = mag_inst[:, id_index[cid]]
            g = g_map.get(cid, float("nan"))
            bp = bp_map.get(cid, float("nan"))
            mnight = float(np.nanmedian(mi))
            if math.isfinite(mnight) and math.isfinite(g) and math.isfinite(bp):
                xs.append(bp)
                ys.append(mnight - g)
        slope = float("nan")
        intercept = float("nan")
        slope_err = float("nan")
        n_col = 0
        if len(xs) >= 8:
            lr = linregress(np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
            slope = float(lr.slope)
            intercept = float(lr.intercept)
            slope_err = float(lr.stderr)
            n_col = int(len(xs))

        aij_rms = None
        aij_n = None
        if aij_tbl is not None:
            cal_bo = mag_calib[:, id_index[BO]]
            tmp = pd.DataFrame({"Label": frame_keys, "mag_calib": cal_bo})
            j = aij_tbl.merge(tmp, on="Label", how="inner")
            aij_rel = pd.to_numeric(j["rel_flux_T1"], errors="coerce")
            vy_mag = pd.to_numeric(j["mag_calib"], errors="coerce")
            vy_rel = np.power(10.0, -0.4 * vy_mag.to_numpy())
            ok = np.isfinite(aij_rel) & np.isfinite(vy_rel) & (aij_rel > 0) & (vy_rel > 0)
            aij_n = int(ok.sum())
            if aij_n >= 8:
                aij_n_v = aij_rel.to_numpy()[ok]
                vy_n = vy_rel[ok]
                aij_n_v = aij_n_v / float(np.median(aij_n_v))
                vy_n = vy_n / float(np.median(vy_n))
                diff = -2.5 * np.log10(aij_n_v / vy_n) * 1000.0
                aij_rms = float(np.sqrt(np.mean(diff * diff)))

        b2_rows.append(
            {
                "f": float(fval),
                "r_px": round(float(radii_px[i]), 4),
                "stars": stars_out,
                "joint_mean_rms_mmag": (
                    None if not rms_list else round(float(np.mean(rms_list)), 4)
                ),
                "colour_slope_mmag_per_bprp": (
                    None if not math.isfinite(slope) else round(slope * 1000.0, 4)
                ),
                "colour_intercept_mag": (
                    None if not math.isfinite(intercept) else round(intercept, 6)
                ),
                "colour_slope_stderr_mmag": (
                    None if not math.isfinite(slope_err) else round(slope_err * 1000.0, 4)
                ),
                "colour_n": n_col,
                "aij_rms_diff_mmag": None if aij_rms is None else round(aij_rms, 4),
                "aij_n": aij_n,
                "aij_pass": bool(aij_rms is not None and aij_rms <= 3.3),
            }
        )

    # COG level vs f=2.5 (needs the ref row).
    ref_row = next(r for r in b2_rows if abs(r["f"] - F_REF) < 1e-9)
    for row in b2_rows:
        dlev = {}
        ok_lev = True
        for name in GROUPS:
            lv = row["stars"][name]["level_mag"]
            rv = ref_row["stars"][name]["level_mag"]
            if lv is None or rv is None:
                dlev[name] = None
                ok_lev = False
            else:
                mm = (float(lv) - float(rv)) * 1000.0
                dlev[name] = round(mm, 4)
                if abs(mm) >= 3.0:
                    ok_lev = False
        row["level_vs_f25_mmag"] = dlev
        row["level_flat_lt_3mmag"] = bool(ok_lev)

    # B3 selection
    survivors = []
    for b1, b2 in zip(b1_rows, b2_rows):
        dee_ok = bool(b1["dee_lt_0p01"])
        lev_ok = bool(b2["level_flat_lt_3mmag"])
        aij_ok = bool(b2["aij_pass"])
        rec = {
            "f": b1["f"],
            "dee": b1["dee"],
            "dee_ok": dee_ok,
            "level_vs_f25_mmag": b2["level_vs_f25_mmag"],
            "level_ok": lev_ok,
            "aij_rms_diff_mmag": b2["aij_rms_diff_mmag"],
            "aij_ok": aij_ok,
            "joint_mean_rms_mmag": b2["joint_mean_rms_mmag"],
            "all_three": bool(dee_ok and lev_ok and aij_ok),
        }
        survivors.append(rec)
    passed = [s for s in survivors if s["all_three"]]
    f_star = None
    f_star_reason = "no f on the grid satisfies dEE<0.01 AND |level-f2.5|<3 mmag (BO/FW/GH) AND AIJ RMS<=3.3 mmag; rule not relaxed"
    if passed:
        passed_sorted = sorted(
            passed,
            key=lambda s: (
                float(s["f"]),
                float(s["joint_mean_rms_mmag"])
                if s["joint_mean_rms_mmag"] is not None
                else 1e9,
            ),
        )
        # smallest f among all-three; if several at that f (won't happen), lowest RMS
        fmin = float(passed_sorted[0]["f"])
        at_fmin = [s for s in passed if abs(float(s["f"]) - fmin) < 1e-12]
        at_fmin.sort(
            key=lambda s: (
                float(s["joint_mean_rms_mmag"])
                if s["joint_mean_rms_mmag"] is not None
                else 1e9
            )
        )
        f_star = float(at_fmin[0]["f"])
        f_star_reason = "smallest f with all three gates; RMS tie-break among that f"

    elapsed = round(time.perf_counter() - t0, 2)
    out = {
        "task": "APERTURE-01b",
        "premise": (
            "EE and accuracy vs f (mode a, night QC FWHM) vs scatter-ladder f=0.385; "
            "f* from dEE/COG/AIJ, not RMS."
        ),
        "mode": "f_fixed_night",
        "fwhm_authority": FWHM_AUTHORITY,
        "night_fwhm_px": round(night_f, 6),
        "annulus_inner_px": round(float(r_in), 4),
        "annulus_outer_px": round(float(r_out), 4),
        "ee_f_grid": [round(float(x), 4) for x in radii_f],
        "n_frames": len(lights),
        "naxis": [int(nx), int(ny)],
        "pool_n": len(pool_ok),
        "bprp_qcuts": [round(float(x), 4) for x in q_cuts],
        "q1_n": len(q1_ids),
        "q4_n": len(q4_ids),
        "q1_iso_n": len(q1_iso),
        "q4_iso_n": len(q4_iso),
        "iso_px": round(iso_px, 4),
        "aij_path": str(AIJ_BO),
        "aij_present": bool(AIJ_BO.is_file()),
        "aij_sha256": aij_sha,
        "B1": b1_rows,
        "B1_curve": b1_curve,
        "B2": b2_rows,
        "B3_survivors": survivors,
        "f_star": f_star,
        "f_star_reason": f_star_reason,
        "elapsed_s": elapsed,
    }
    (OUT / "b1b_measure.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="ascii"
    )
    pd.DataFrame(b1_rows).to_csv(OUT / "b1_dee_grid.csv", index=False)
    b2_flat = []
    for row in b2_rows:
        rec = {
            "f": row["f"],
            "r_px": row["r_px"],
            "joint_mean_rms_mmag": row["joint_mean_rms_mmag"],
            "colour_slope_mmag_per_bprp": row["colour_slope_mmag_per_bprp"],
            "aij_rms_diff_mmag": row["aij_rms_diff_mmag"],
            "aij_n": row["aij_n"],
            "aij_pass": row["aij_pass"],
            "level_flat_lt_3mmag": row["level_flat_lt_3mmag"],
        }
        for name in GROUPS:
            rec[f"rms_{name}_mmag"] = row["stars"][name]["rms_mmag"]
            rec[f"rho_{name}"] = row["stars"][name]["spearman_abs_rho"]
            rec[f"dlevel_{name}_mmag"] = row["level_vs_f25_mmag"][name]
        b2_flat.append(rec)
    pd.DataFrame(b2_flat).to_csv(OUT / "b2_f_grid.csv", index=False)
    pd.DataFrame(survivors).to_csv(OUT / "b3_selection.csv", index=False)
    print("f_star=%s elapsed=%.1fs" % (f_star, elapsed), flush=True)
    print(json.dumps({"B1": b1_rows, "B3": survivors, "f_star": f_star}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
