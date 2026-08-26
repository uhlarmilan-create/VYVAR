# -*- coding: ascii -*-
"""C6-3c X1 WCS-APERTURE + X3 era03-only targets. Measure only."""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402

ERA03 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era03_20260820"
ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
SESSION = Path(__file__).resolve().parent
BO = "1498613634033133184"
FW = "1497343732462852864"
GH = "1498804639818507904"
FWHM_PX = 5.15
R_AP = 5.499
BOX_HALF = 3.0 * FWHM_PX
ID_DTYPE = {"catalog_id": str, "name": str, "target_catalog_id": str, "comp_id": str}


def nid(x: object) -> str:
    return normalize_gaia_source_id(x) or str(x).strip()


def id_series(df: pd.DataFrame) -> pd.Series:
    if "name" in df.columns:
        n = df["name"].map(nid)
        if float(n.str.fullmatch(r"\d{12,22}").fillna(False).mean()) > 0.5:
            return n
    if "catalog_id" in df.columns:
        return df["catalog_id"].map(nid)
    return df.iloc[:, 0].astype(str).map(nid)


def read_ids(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, dtype=ID_DTYPE, low_memory=False)


def aligned_dir(root: Path) -> Path:
    return root / "detrended_aligned" / "lights" / SETUP


def phot(root: Path) -> Path:
    return root / "platesolve" / SETUP / "photometry"


def ensemble_ids(root: Path, tid: str) -> list[str]:
    p = phot(root) / "comparison_stars_per_target.csv"
    df = read_ids(p)
    sub = df.loc[df["target_catalog_id"].map(nid).eq(tid)]
    return [i for i in id_series(sub).tolist() if i]


def gaussian_ee(r: float, fwhm: float, d: float, n: int = 161) -> float:
    s = float(fwhm) / 2.35482
    half = float(r) + 4.0 * s
    xs = np.linspace(-half, half, int(n))
    xx, yy = np.meshgrid(xs, xs)
    psf = np.exp(-((xx - float(d)) ** 2 + yy**2) / (2.0 * s * s))
    m = (xx**2 + yy**2) <= (float(r) ** 2)
    tot = float(psf.sum())
    if tot <= 0:
        return float("nan")
    return float(psf[m].sum() / tot)


def dmag_from_offset(d: float) -> float:
    e0 = gaussian_ee(R_AP, FWHM_PX, 0.0)
    e1 = gaussian_ee(R_AP, FWHM_PX, float(d))
    if not (e0 > 0 and e1 > 0 and math.isfinite(e0) and math.isfinite(e1)):
        return float("nan")
    return float(-2.5 * math.log10(e1 / e0) * 1000.0)


def com_centroid(img: np.ndarray, x: float, y: float, half: float) -> tuple[float, float]:
    h, w = img.shape
    x0 = int(max(0, math.floor(x - half)))
    x1 = int(min(w, math.ceil(x + half) + 1))
    y0 = int(max(0, math.floor(y - half)))
    y1 = int(min(h, math.ceil(y + half) + 1))
    if x1 <= x0 or y1 <= y0:
        return float("nan"), float("nan")
    patch = np.asarray(img[y0:y1, x0:x1], dtype=np.float64)
    if not np.any(np.isfinite(patch)):
        return float("nan"), float("nan")
    sky = float(np.nanmedian(patch))
    wgt = np.clip(patch - sky, 0.0, None)
    s = float(np.nansum(wgt))
    if s <= 0:
        iy, ix = np.unravel_index(int(np.nanargmax(patch)), patch.shape)
        return float(x0 + ix), float(y0 + iy)
    yy, xx = np.indices(patch.shape)
    return float(x0 + np.nansum(wgt * xx) / s), float(y0 + np.nansum(wgt * yy) / s)


def proc_star_xy(root: Path, cid: str) -> pd.DataFrame:
    rows = []
    for p in sorted(aligned_dir(root).glob("proc_*.csv")):
        df = read_ids(p)
        hit = df.loc[id_series(df).eq(cid)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append(
            {
                "frame": p.name.replace("proc_", "").replace(".csv", ""),
                "proc": p.name,
                "x": float(pd.to_numeric(r.get("x"), errors="coerce")),
                "y": float(pd.to_numeric(r.get("y"), errors="coerce")),
            }
        )
    return pd.DataFrame(rows)


def lc_dmag(tid: str) -> pd.DataFrame:
    a = pd.read_csv(phot(ERA03) / "lightcurves" / f"lightcurve_{tid}.csv")
    b = pd.read_csv(phot(ERA04) / "lightcurves" / f"lightcurve_{tid}.csv")
    a["sf"] = a["source_file"].astype(str).map(lambda s: Path(s).name)
    b["sf"] = b["source_file"].astype(str).map(lambda s: Path(s).name)
    j = a.merge(b, on="sf", suffixes=("_e3", "_e4"))
    j["frame"] = j["sf"].str.replace("proc_", "", regex=False).str.replace(".csv", "", regex=False)
    j["dmag_mmag"] = (j["mag_calib_e4"] - j["mag_calib_e3"]) * 1000.0
    return j[["frame", "sf", "dmag_mmag"]]


def pct(a: np.ndarray, q: float) -> float:
    b = np.asarray(a, dtype=np.float64)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return float("nan")
    return float(np.percentile(b, q))


def measure_x1() -> dict:
    t0 = time.perf_counter()
    stars: list[tuple[str, str, str]] = [("BO", "target", BO)]
    for cid in ensemble_ids(ERA04, BO):
        stars.append(("BO", "comp", cid))
    stars.append(("FW", "target", FW))
    for cid in ensemble_ids(ERA04, FW):
        stars.append(("FW", "comp", cid))
    stars.append(("GH", "target", GH))
    for cid in ensemble_ids(ERA04, GH):
        stars.append(("GH", "comp", cid))

    ms3 = read_ids(ERA03 / "platesolve" / SETUP / "masterstars_full_match.csv")
    ms4 = read_ids(ERA04 / "platesolve" / SETUP / "masterstars_full_match.csv")
    ms3["_id"] = id_series(ms3)
    ms4["_id"] = id_series(ms4)

    hdr = fits.getheader(aligned_dir(ERA04) / "BO_CVn_Light_109.fits")
    crpix1 = float(hdr.get("CRPIX1", float("nan")))
    crpix2 = float(hdr.get("CRPIX2", float("nan")))

    per_rows = []
    star_sum = []
    xy_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for group, role, cid in stars:
        e3 = proc_star_xy(ERA03, cid)
        e4 = proc_star_xy(ERA04, cid)
        xy_cache[(group, cid)] = e3.merge(e4, on="frame", suffixes=("_e3", "_e4"))
        j = xy_cache[(group, cid)]
        j["dx"] = j["x_e4"] - j["x_e3"]
        j["dy"] = j["y_e4"] - j["y_e3"]
        j["d"] = np.hypot(j["dx"], j["dy"])
        j["r_crpix"] = np.hypot(j["x_e4"] - crpix1, j["y_e4"] - crpix2)
        m3 = ms3.loc[ms3["_id"].eq(cid)]
        m4 = ms4.loc[ms4["_id"].eq(cid)]
        ms_dx = float("nan")
        ms_dy = float("nan")
        if not m3.empty and not m4.empty:
            ms_dx = float(pd.to_numeric(m4["x"].iloc[0], errors="coerce")) - float(
                pd.to_numeric(m3["x"].iloc[0], errors="coerce")
            )
            ms_dy = float(pd.to_numeric(m4["y"].iloc[0], errors="coerce")) - float(
                pd.to_numeric(m3["y"].iloc[0], errors="coerce")
            )
        star_sum.append(
            {
                "group": group,
                "role": role,
                "catalog_id": cid,
                "n": int(len(j)),
                "dx_p50": pct(j["dx"].to_numpy(), 50),
                "dy_p50": pct(j["dy"].to_numpy(), 50),
                "d_p50": pct(j["d"].to_numpy(), 50),
                "d_p95": pct(j["d"].to_numpy(), 95),
                "r_crpix_p50": pct(j["r_crpix"].to_numpy(), 50),
                "ms_dx": ms_dx,
                "ms_dy": ms_dy,
                "ms_d": float(math.hypot(ms_dx, ms_dy)) if math.isfinite(ms_dx) else float("nan"),
            }
        )
        for _, r in j.iterrows():
            per_rows.append(
                {
                    "group": group,
                    "role": role,
                    "catalog_id": cid,
                    "frame": r["frame"],
                    "dx": float(r["dx"]),
                    "dy": float(r["dy"]),
                    "d": float(r["d"]),
                    "r_crpix": float(r["r_crpix"]),
                }
            )
    per_df = pd.DataFrame(per_rows)
    per_df.to_csv(SESSION / "c63c_x1_xy_per_frame.csv", index=False)
    pd.DataFrame(star_sum).to_csv(SESSION / "c63c_x1_xy_per_star.csv", index=False)

    # |d| vs r_crpix: linear trend over all stars/frames
    rr = per_df["r_crpix"].to_numpy(dtype=float)
    dd = per_df["d"].to_numpy(dtype=float)
    m = np.isfinite(rr) & np.isfinite(dd)
    slope = float("nan")
    corr_rd = float("nan")
    if int(m.sum()) >= 10 and float(np.std(rr[m])) > 0:
        slope = float(np.polyfit(rr[m], dd[m], 1)[0])
        corr_rd = float(np.corrcoef(rr[m], dd[m])[0, 1])

    epoch_rows = []
    dmag_vs = {}
    for group, tid in (("BO", BO), ("FW", FW), ("GH", GH)):
        dlc = lc_dmag(tid)
        tgt = xy_cache[(group, tid)].copy()
        tgt["d_tgt"] = tgt["d"]
        comps = [c for g, role, c in stars if g == group and role == "comp"]
        d_comp = None
        dx_comp = None
        dy_comp = None
        for c in comps:
            cj = xy_cache[(group, c)][["frame", "dx", "dy", "d"]]
            if d_comp is None:
                d_comp = cj.rename(columns={"d": f"d_{c}", "dx": f"dx_{c}", "dy": f"dy_{c}"})
            else:
                d_comp = d_comp.merge(cj.rename(columns={"d": f"d_{c}", "dx": f"dx_{c}", "dy": f"dy_{c}"}), on="frame")
        dcols = [c for c in d_comp.columns if c.startswith("d_") and c != "d_tgt"]
        dxcols = [c for c in d_comp.columns if c.startswith("dx_")]
        dycols = [c for c in d_comp.columns if c.startswith("dy_")]
        d_comp["d_comp_med"] = d_comp[dcols].median(axis=1)
        d_comp["dx_comp_med"] = d_comp[dxcols].median(axis=1)
        d_comp["dy_comp_med"] = d_comp[dycols].median(axis=1)
        mrg = tgt.merge(d_comp[["frame", "d_comp_med", "dx_comp_med", "dy_comp_med"]], on="frame")
        mrg["d_diff"] = np.hypot(mrg["dx"] - mrg["dx_comp_med"], mrg["dy"] - mrg["dy_comp_med"])
        mrg["dmag_pred_tgt_mmag"] = mrg["d_tgt"].map(dmag_from_offset)
        mrg["dmag_pred_comp_mmag"] = mrg["d_comp_med"].map(dmag_from_offset)
        mrg["dmag_pred_diff_mmag"] = mrg["dmag_pred_tgt_mmag"] - mrg["dmag_pred_comp_mmag"]
        mrg = mrg.merge(dlc, on="frame", how="inner")
        xs = mrg["d_diff"].to_numpy(dtype=float)
        ys = mrg["dmag_mmag"].to_numpy(dtype=float)
        ps = mrg["dmag_pred_diff_mmag"].to_numpy(dtype=float)
        mm = np.isfinite(xs) & np.isfinite(ys)
        corr_d = float(np.corrcoef(xs[mm], ys[mm])[0, 1]) if int(mm.sum()) >= 5 and float(np.std(xs[mm])) > 0 else float("nan")
        mp = np.isfinite(ps) & np.isfinite(ys)
        corr_p = float(np.corrcoef(ps[mp], ys[mp])[0, 1]) if int(mp.sum()) >= 5 and float(np.std(ps[mp])) > 0 else float("nan")
        dmag_vs[group] = {
            "n": int(len(mrg)),
            "dmag_p50_mmag": pct(ys, 50),
            "d_tgt_p50": pct(mrg["d_tgt"].to_numpy(), 50),
            "d_comp_p50": pct(mrg["d_comp_med"].to_numpy(), 50),
            "d_diff_p50": pct(xs, 50),
            "d_diff_p95": pct(xs, 95),
            "pred_diff_p50_mmag": pct(ps, 50),
            "corr_dmag_vs_d_diff": corr_d,
            "corr_dmag_vs_pred": corr_p,
        }
        for _, r in mrg.iterrows():
            epoch_rows.append(
                {
                    "group": group,
                    "frame": r["frame"],
                    "dmag_mmag": float(r["dmag_mmag"]),
                    "d_tgt": float(r["d_tgt"]),
                    "d_comp_med": float(r["d_comp_med"]),
                    "d_diff": float(r["d_diff"]),
                    "pred_diff_mmag": float(r["dmag_pred_diff_mmag"]),
                }
            )
    pd.DataFrame(epoch_rows).to_csv(SESSION / "c63c_x1_dmag_vs_dxy.csv", index=False)

    # Truth: COM vs aperture centres on era04 aligned pixels (identical to era03)
    truth_rows = []
    probe_cids = [c for _, _, c in stars]
    frames = sorted({r["frame"] for r in per_rows})
    # subsample: all 134 is OK with memmap; do all
    for fr in frames:
        fp = aligned_dir(ERA04) / f"{fr}.fits"
        if not fp.is_file():
            continue
        with fits.open(fp, memmap=True) as hdul:
            img = np.asarray(hdul[0].data, dtype=np.float64)
        for group, role, cid in stars:
            j = xy_cache[(group, cid)]
            row = j.loc[j["frame"].eq(fr)]
            if row.empty:
                continue
            x3 = float(row["x_e3"].iloc[0])
            y3 = float(row["y_e3"].iloc[0])
            x4 = float(row["x_e4"].iloc[0])
            y4 = float(row["y_e4"].iloc[0])
            # box centre: era04 aperture (same pixels; seed near both)
            cx, cy = com_centroid(img, x4, y4, BOX_HALF)
            truth_rows.append(
                {
                    "group": group,
                    "role": role,
                    "catalog_id": cid,
                    "frame": fr,
                    "res_e3": float(math.hypot(cx - x3, cy - y3)) if math.isfinite(cx) else float("nan"),
                    "res_e4": float(math.hypot(cx - x4, cy - y4)) if math.isfinite(cx) else float("nan"),
                }
            )
    tdf = pd.DataFrame(truth_rows)
    tdf.to_csv(SESSION / "c63c_x1_centroid_truth.csv", index=False)
    truth = {
        "n": int(len(tdf)),
        "res_e3_p50": pct(tdf["res_e3"].to_numpy(), 50) if not tdf.empty else float("nan"),
        "res_e3_p95": pct(tdf["res_e3"].to_numpy(), 95) if not tdf.empty else float("nan"),
        "res_e4_p50": pct(tdf["res_e4"].to_numpy(), 50) if not tdf.empty else float("nan"),
        "res_e4_p95": pct(tdf["res_e4"].to_numpy(), 95) if not tdf.empty else float("nan"),
        "closer": "era04"
        if (not tdf.empty and pct(tdf["res_e4"].to_numpy(), 50) < pct(tdf["res_e3"].to_numpy(), 50))
        else "era03",
    }
    by_star = []
    for cid, g in tdf.groupby("catalog_id"):
        by_star.append(
            {
                "catalog_id": cid,
                "res_e3_p50": pct(g["res_e3"].to_numpy(), 50),
                "res_e4_p50": pct(g["res_e4"].to_numpy(), 50),
                "closer": "era04" if pct(g["res_e4"].to_numpy(), 50) < pct(g["res_e3"].to_numpy(), 50) else "era03",
            }
        )
    pd.DataFrame(by_star).to_csv(SESSION / "c63c_x1_centroid_truth_per_star.csv", index=False)

    hdr3 = fits.getheader(ERA03 / "platesolve" / SETUP / "MASTERSTAR.fits")
    hdr4 = fits.getheader(ERA04 / "platesolve" / SETUP / "MASTERSTAR.fits")
    return {
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "governing": {
            "aligned_lock": "src_py/pipeline.py:_lock_matched_centroids_to_master_grid:7491 call 8401-8410",
            "on_ref_grid": "VY_ALGN via _fits_header_vy_algn_aligned:252",
            "mechanism": "DAO detect then match; aligned frames snap matched stars to MASTERSTAR x,y then brightest pixel in ~2.5 FWHM window. Not per-frame WCS world2pix of Gaia. Unaligned uses _apply_dao_centroid_wcs_guard:7455.",
            "aperture": "photometry_core.py:enhance_catalog_dataframe_aperture_bpm:14136 uses catalog x,y already written",
        },
        "wcs_cards": {
            "era03_MS_CRPIX1": hdr3.get("CRPIX1"),
            "era04_MS_CRPIX1": hdr4.get("CRPIX1"),
            "era03_MS_CRPIX2": hdr3.get("CRPIX2"),
            "era04_MS_CRPIX2": hdr4.get("CRPIX2"),
            "era04_aligned_CRPIX1": crpix1,
            "era04_aligned_CRPIX2": crpix2,
            "origin": "D2 accepted optimizer refit on era04 MASTERSTAR (rejected=false, p95 1.307->1.228). Aligned-light WCS cards match era04 MASTERSTAR, not era03/live.",
        },
        "n_stars": int(len(stars)),
        "star_summary": star_sum,
        "d_vs_r_crpix_slope_px_per_px": slope,
        "d_vs_r_crpix_corr": corr_rd,
        "dmag_vs": dmag_vs,
        "truth": truth,
        "fwhm_px": FWHM_PX,
        "r_ap_px": R_AP,
    }


def measure_x3() -> dict:
    t0 = time.perf_counter()
    def lc_ids(root: Path) -> set[str]:
        d = phot(root) / "lightcurves"
        out = set()
        for p in d.glob("lightcurve_*.csv"):
            if "_psf" in p.name or "_adaptive" in p.name:
                continue
            out.add(p.stem.replace("lightcurve_", ""))
        return out

    e3 = lc_ids(ERA03)
    e4 = lc_ids(ERA04)
    only3 = sorted(e3 - e4)
    only4 = sorted(e4 - e3)
    vt3 = read_ids(ERA03 / "platesolve" / SETUP / "variable_targets.csv")
    vt4p = ERA04 / "platesolve" / SETUP / "variable_targets.csv"
    vt4 = read_ids(vt4p) if vt4p.is_file() else pd.DataFrame()
    vt3["_id"] = id_series(vt3)
    if not vt4.empty:
        vt4["_id"] = id_series(vt4)
    rows = []
    for tid in only3:
        rec = {"catalog_id": tid, "in_era03_lc": True, "in_era04_lc": False}
        r3 = vt3.loc[vt3["_id"].eq(tid)]
        if not r3.empty:
            rec["name"] = str(r3.iloc[0].get("name", "") or r3.iloc[0].get("vsx_name", "") or "")
            rec["skip_photometry_e3"] = str(r3.iloc[0].get("skip_photometry", ""))
            rec["skip_reason_e3"] = str(r3.iloc[0].get("skip_reason", "") or "")
            rec["zone_flag_e3"] = str(r3.iloc[0].get("zone_flag", r3.iloc[0].get("zone", "")) or "")
        r4 = vt4.loc[vt4["_id"].eq(tid)] if not vt4.empty else pd.DataFrame()
        if not r4.empty:
            rec["name"] = rec.get("name") or str(r4.iloc[0].get("name", "") or "")
            rec["skip_photometry_e4"] = str(r4.iloc[0].get("skip_photometry", ""))
            rec["skip_reason_e4"] = str(r4.iloc[0].get("skip_reason", "") or "")
            rec["zone_flag_e4"] = str(r4.iloc[0].get("zone_flag", r4.iloc[0].get("zone", "")) or "")
            rec["in_era04_vt"] = True
        else:
            rec["in_era04_vt"] = False
            rec["skip_reason_e4"] = "absent_from_variable_targets"
        rows.append(rec)
    pd.DataFrame(rows).to_csv(SESSION / "c63c_x3_era03_only.csv", index=False)
    return {
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "n_era03": int(len(e3)),
        "n_era04": int(len(e4)),
        "n_only_era03": int(len(only3)),
        "n_only_era04": int(len(only4)),
        "only_era03": rows,
        "only_era04": only4,
    }


def main() -> None:
    x1 = measure_x1()
    x3 = measure_x3()
    out = {"utc": datetime.now(timezone.utc).isoformat(), "x1": x1, "x3": x3}
    (SESSION / "c63c_x1_x3.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="ascii")
    print("X1 elapsed", x1["elapsed_s"], "truth closer", x1["truth"]["closer"], "d_vs_r corr", x1["d_vs_r_crpix_corr"])
    print("dmag", {k: v.get("dmag_p50_mmag") for k, v in x1["dmag_vs"].items()})
    print("pred", {k: v.get("pred_diff_p50_mmag") for k, v in x1["dmag_vs"].items()})
    print("corr d_diff", {k: v.get("corr_dmag_vs_d_diff") for k, v in x1["dmag_vs"].items()})
    print("X3 only_era03", x3["n_only_era03"], [r["catalog_id"] for r in x3["only_era03"]])
    print("wrote", SESSION / "c63c_x1_x3.json")


if __name__ == "__main__":
    main()
