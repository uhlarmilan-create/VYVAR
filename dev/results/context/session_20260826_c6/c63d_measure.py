# -*- coding: ascii -*-
"""C6-3d X6a/X6b/X6e: aperture-radius governing inputs, P1-P3, EDGE bbox."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.stats import spearmanr

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))
from photometry_core import (  # noqa: E402
    _gaussian_ee_fraction,
    _star_mag_for_aperture_sizing,
    compute_snr_optimal_aperture_table,
)
from masterstar_gaia_accounting import _norm_cid  # noqa: E402

ERA03 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era03_20260820"
ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
OUT = Path(__file__).resolve().parent
BO = "1498613634033133184"
FW = "1497343732462852864"
GH = "1498804639818507904"
BO_COMPS = [
    "1497771992240531712",
    "1499200223486564608",
    "1497974027502858240",
    "1497368849430107904",
]
FW_COMPS = [
    "1497442379271632384",
    "1499906247391001088",
    "1497674651102612992",
    "1498020894186918144",
    "1498812233320666368",
    "1497370563121917952",
    "1497313255374892800",
    "1500486102335278592",
]
GH_COMPS = [
    "1497442379271632384",
    "1499906247391001088",
    "1497674651102612992",
    "1498020894186918144",
    "1498812233320666368",
    "1500486102335278592",
    "1496315070616056064",
    "1497196054307837696",
]
EDGE_IDS = [
    "1485560025830226432",
    "1496037650087948160",
    "1496733984545821696",
    "1497491273179203456",
]


def phot(root: Path) -> Path:
    return root / "platesolve" / SETUP / "photometry"


def lights(root: Path) -> Path:
    return root / "detrended_aligned" / "lights" / SETUP


def load_snr(root: Path) -> dict:
    p = root / "aperture_snr_table.json"
    return json.loads(p.read_text(encoding="ascii"))


def nearest_bin(table: dict, mag: float) -> float:
    bins = [float(k) for k in table]
    return min(bins, key=lambda m: abs(m - float(mag)))


def table_r(snr: dict, mag: float) -> float:
    b = nearest_bin(snr["table"], mag)
    return float(snr["table"][str(b) if str(b) in snr["table"] else f"{b:.1f}"]), b


def star_row(ms: pd.DataFrame, cid: str) -> pd.Series:
    hit = ms[ms["_cid"] == _norm_cid(cid)]
    if hit.empty:
        raise KeyError(cid)
    return hit.iloc[0]


def recompute_table(snr: dict, *, ee=None, sky=None, bkg=None, fwhm=None, zp=None) -> dict:
    return compute_snr_optimal_aperture_table(
        fwhm_px=float(snr["fwhm_px"] if fwhm is None else fwhm),
        sky_adu_per_px=float(snr["sky_adu_per_px"] if sky is None else sky),
        gain=float(snr["gain"]),
        read_noise=float(snr["read_noise"]),
        zero_point=float(snr["zero_point"] if zp is None else zp),
        bkg_var_adu2_per_px=float(snr["bkg_var_adu2_per_px"] if bkg is None else bkg),
        ee_radii=None if ee is False else np.asarray(snr["ee_radii"], dtype=float),
        ee_curve=None if ee is False else np.asarray(snr["ee_curve"], dtype=float),
        ee_source=None if ee is False else "measured_growth_curve",
    )


def x6a() -> dict:
    t0 = time.perf_counter()
    s3, s4 = load_snr(ERA03), load_snr(ERA04)
    ms3 = pd.read_csv(ERA03 / "platesolve" / SETUP / "masterstars_full_match.csv", dtype={"catalog_id": str}, low_memory=False)
    ms4 = pd.read_csv(ERA04 / "platesolve" / SETUP / "masterstars_full_match.csv", dtype={"catalog_id": str}, low_memory=False)
    ms3["catalog_id"] = ms3["catalog_id"].astype(str)
    ms4["catalog_id"] = ms4["catalog_id"].astype(str)
    ms3["_cid"] = ms3["catalog_id"].map(_norm_cid)
    ms4["_cid"] = ms4["catalog_id"].map(_norm_cid)
    p3 = pd.read_csv(lights(ERA03) / "proc_BO_CVn_Light_001.csv", dtype={"catalog_id": str})
    p4 = pd.read_csv(lights(ERA04) / "proc_BO_CVn_Light_001.csv", dtype={"catalog_id": str})
    p3["catalog_id"] = p3["catalog_id"].map(lambda x: _norm_cid(x) or str(x))
    p4["catalog_id"] = p4["catalog_id"].map(lambda x: _norm_cid(x) or str(x))

    groups = [("BO", BO, BO_COMPS), ("FW", FW, FW_COMPS), ("GH", GH, GH_COMPS)]
    stars = []
    for gname, tid, comps in groups:
        for role, cid in [("target", tid)] + [("comp", c) for c in comps]:
            r3s = star_row(ms3, cid)
            r4s = star_row(ms4, cid)
            mag3 = _star_mag_for_aperture_sizing(r3s)
            mag4 = _star_mag_for_aperture_sizing(r4s)
            bin3 = nearest_bin(s3["table"], float(mag3))
            bin4 = nearest_bin(s4["table"], float(mag4))
            pr3 = p3[p3["catalog_id"] == cid]
            pr4 = p4[p4["catalog_id"] == cid]
            rec = {
                "group": gname,
                "role": role,
                "catalog_id": cid,
                "mag_e3": float(mag3) if mag3 is not None else float("nan"),
                "mag_e4": float(mag4) if mag4 is not None else float("nan"),
                "bin_e3": bin3,
                "bin_e4": bin4,
                "table_r_e3": float(s3["table"][f"{bin3:.1f}"]),
                "table_r_e4": float(s4["table"][f"{bin4:.1f}"]),
                "proc_r_e3": float(pr3.iloc[0]["aperture_r_px"]) if len(pr3) else float("nan"),
                "proc_r_e4": float(pr4.iloc[0]["aperture_r_px"]) if len(pr4) else float("nan"),
                "phot_g_e3": float(pd.to_numeric(pd.Series([r3s.get("phot_g_mean_mag")]), errors="coerce").iloc[0]),
                "phot_g_e4": float(pd.to_numeric(pd.Series([r4s.get("phot_g_mean_mag")]), errors="coerce").iloc[0]),
            }
            stars.append(rec)

    # Ablation: which input moves mag 9.5 from 5.999 to 5.499
    mag_key = 9.5
    variants = {
        "e3": recompute_table(s3),
        "e4": recompute_table(s4),
        "e3_ee__e4_sky_bkg": compute_snr_optimal_aperture_table(
            fwhm_px=float(s3["fwhm_px"]),
            sky_adu_per_px=float(s4["sky_adu_per_px"]),
            gain=float(s3["gain"]),
            read_noise=float(s3["read_noise"]),
            zero_point=25.0,
            bkg_var_adu2_per_px=float(s4["bkg_var_adu2_per_px"]),
            ee_radii=np.asarray(s3["ee_radii"]),
            ee_curve=np.asarray(s3["ee_curve"]),
            ee_source="measured_growth_curve",
        ),
        "e4_ee__e3_sky_bkg": compute_snr_optimal_aperture_table(
            fwhm_px=float(s3["fwhm_px"]),
            sky_adu_per_px=float(s3["sky_adu_per_px"]),
            gain=float(s3["gain"]),
            read_noise=float(s3["read_noise"]),
            zero_point=25.0,
            bkg_var_adu2_per_px=float(s3["bkg_var_adu2_per_px"]),
            ee_radii=np.asarray(s4["ee_radii"]),
            ee_curve=np.asarray(s4["ee_curve"]),
            ee_source="measured_growth_curve",
        ),
        "e3_ee_sky__e4_bkg": compute_snr_optimal_aperture_table(
            fwhm_px=float(s3["fwhm_px"]),
            sky_adu_per_px=float(s3["sky_adu_per_px"]),
            gain=float(s3["gain"]),
            read_noise=float(s3["read_noise"]),
            zero_point=25.0,
            bkg_var_adu2_per_px=float(s4["bkg_var_adu2_per_px"]),
            ee_radii=np.asarray(s3["ee_radii"]),
            ee_curve=np.asarray(s3["ee_curve"]),
            ee_source="measured_growth_curve",
        ),
        "e3_ee_bkg__e4_sky": compute_snr_optimal_aperture_table(
            fwhm_px=float(s3["fwhm_px"]),
            sky_adu_per_px=float(s4["sky_adu_per_px"]),
            gain=float(s3["gain"]),
            read_noise=float(s3["read_noise"]),
            zero_point=25.0,
            bkg_var_adu2_per_px=float(s3["bkg_var_adu2_per_px"]),
            ee_radii=np.asarray(s3["ee_radii"]),
            ee_curve=np.asarray(s3["ee_curve"]),
            ee_source="measured_growth_curve",
        ),
        "e3_gauss_ee__e4_sky_bkg": compute_snr_optimal_aperture_table(
            fwhm_px=float(s3["fwhm_px"]),
            sky_adu_per_px=float(s4["sky_adu_per_px"]),
            gain=float(s3["gain"]),
            read_noise=float(s3["read_noise"]),
            zero_point=25.0,
            bkg_var_adu2_per_px=float(s4["bkg_var_adu2_per_px"]),
        ),
        "e4_gauss_ee__e3_sky_bkg": compute_snr_optimal_aperture_table(
            fwhm_px=float(s3["fwhm_px"]),
            sky_adu_per_px=float(s3["sky_adu_per_px"]),
            gain=float(s3["gain"]),
            read_noise=float(s3["read_noise"]),
            zero_point=25.0,
            bkg_var_adu2_per_px=float(s3["bkg_var_adu2_per_px"]),
        ),
    }
    ablate = {}
    for name, tab in variants.items():
        ablate[name] = float(tab["table"][mag_key])

    inputs = {
        "fwhm_px_e3": s3["fwhm_px"],
        "fwhm_px_e4": s4["fwhm_px"],
        "fwhm_px_scope": s3.get("fwhm_px_scope"),
        "fwhm_estimator": s3.get("fwhm_estimator"),
        "vy_fwhm_dao_e3": s3.get("vy_fwhm_dao_px"),
        "vy_fwhm_dao_e4": s4.get("vy_fwhm_dao_px"),
        "vy_fwhm_gauss_e3": s3.get("vy_fwhm_gauss_px"),
        "vy_fwhm_gauss_e4": s4.get("vy_fwhm_gauss_px"),
        "sky_adu_e3": s3["sky_adu_per_px"],
        "sky_adu_e4": s4["sky_adu_per_px"],
        "bkg_var_e3": s3["bkg_var_adu2_per_px"],
        "bkg_var_e4": s4["bkg_var_adu2_per_px"],
        "gain_e3": s3["gain"],
        "gain_e4": s4["gain"],
        "rn_e3": s3["read_noise"],
        "rn_e4": s4["read_noise"],
        "zp_e3": s3["zero_point"],
        "zp_e4": s4["zero_point"],
        "zp_cal_ok_e3": (s3.get("zero_point_calibration") or {}).get("ok"),
        "zp_cal_ok_e4": (s4.get("zero_point_calibration") or {}).get("ok"),
        "zp_cal_reason_e3": (s3.get("zero_point_calibration") or {}).get("reason"),
        "zp_cal_reason_e4": (s4.get("zero_point_calibration") or {}).get("reason"),
        "ee_path_e3": s3["ee_path"],
        "ee_path_e4": s4["ee_path"],
        "ee_n_cog_e3": s3.get("ee_n_cog"),
        "ee_n_cog_e4": s4.get("ee_n_cog"),
        "ee_r90_e3": s3.get("ee_r90_px"),
        "ee_r90_e4": s4.get("ee_r90_px"),
        "bound_9.5_e3": s3["bound_hit_by_mag"]["9.5"],
        "bound_9.5_e4": s4["bound_hit_by_mag"]["9.5"],
        "table_9.5_e3": s3["table"]["9.5"],
        "table_9.5_e4": s4["table"]["9.5"],
    }
    # Name the mover: which swap reproduces e4's 5.499
    target = float(s4["table"]["9.5"])
    movers = [k for k, v in ablate.items() if abs(v - target) < 0.01 and k != "e4"]
    out = {
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "inputs": inputs,
        "ablation_r_at_9.5": ablate,
        "named_mover": movers,
        "stars": stars,
    }
    pd.DataFrame(stars).to_csv(OUT / "c63d_x6a_stars.csv", index=False)
    return out


def p1_p2() -> dict:
    t0 = time.perf_counter()
    qc4 = pd.read_csv(ERA04 / "calibrated" / "lights" / "qc_metrics.csv")
    qc4["frame"] = qc4["src"].map(lambda p: Path(str(p)).stem)
    qc = qc4.rename(columns={"fwhm_px": "fwhm_px_e4"})

    d3 = pd.read_csv(phot(ERA03) / "lightcurves" / f"lightcurve_{BO}.csv")
    d4 = pd.read_csv(phot(ERA04) / "lightcurves" / f"lightcurve_{BO}.csv")
    d3["frame"] = d3["source_file"].str.replace("proc_", "", regex=False).str.replace(".csv", "", regex=False)
    d4["frame"] = d4["source_file"].str.replace("proc_", "", regex=False).str.replace(".csv", "", regex=False)
    m = d3.merge(d4, on="source_file", suffixes=("_e3", "_e4"))
    m["frame"] = m["source_file"].str.replace("proc_", "", regex=False).str.replace(".csv", "", regex=False)
    m = m.merge(qc[["frame", "fwhm_px_e4"]], on="frame", how="left")
    r3 = 6.0
    r4 = 5.5
    fwhm = pd.to_numeric(m["fwhm_px_e4"], errors="coerce")
    ee3 = np.array([_gaussian_ee_fraction(r3, float(f)) for f in fwhm])
    ee4 = np.array([_gaussian_ee_fraction(r4, float(f)) for f in fwhm])
    pred = 2.5 * np.log10(ee3 / ee4) * 1000.0
    d_inst = (
        pd.to_numeric(m["mag_inst_e4"], errors="coerce")
        - pd.to_numeric(m["mag_inst_e3"], errors="coerce")
    ) * 1000.0
    p1_med = float(np.nanmedian(pred))
    obs = float(np.nanmedian(d_inst))
    p1 = {
        "obs_mag_inst_p50_mmag": obs,
        "pred_gaussian_ee_p50_mmag": p1_med,
        "pred_minus_obs_mmag": p1_med - obs,
        "fwhm_qc_p50": float(np.nanmedian(fwhm)),
        "n": int(np.isfinite(pred).sum()),
        "pass": abs(p1_med - 18.62) <= 3.0,
        "pass_vs_obs": abs(p1_med - obs) <= 3.0,
        "criterion": "pred within +-3 mmag of +18.62",
    }

    p2 = {}
    for name, cid in [("BO", BO), ("FW", FW), ("GH", GH)]:
        a = pd.read_csv(phot(ERA03) / "lightcurves" / f"lightcurve_{cid}.csv")
        b = pd.read_csv(phot(ERA04) / "lightcurves" / f"lightcurve_{cid}.csv")
        j = a.merge(b, on="source_file", suffixes=("_e3", "_e4"))
        j["frame"] = j["source_file"].str.replace("proc_", "", regex=False).str.replace(".csv", "", regex=False)
        j = j.merge(qc[["frame", "fwhm_px_e4"]], on="frame", how="inner")
        dmag = (
            pd.to_numeric(j["mag_calib_e4"], errors="coerce")
            - pd.to_numeric(j["mag_calib_e3"], errors="coerce")
        ) * 1000.0
        fw = pd.to_numeric(j["fwhm_px_e4"], errors="coerce")
        mask = np.isfinite(dmag) & np.isfinite(fw)
        rho, pval = spearmanr(fw[mask], dmag[mask])
        p2[name] = {
            "n": int(mask.sum()),
            "spearman_rho": float(rho),
            "pvalue": float(pval),
            "pass": abs(float(rho)) > 0.5,
        }
    pd.DataFrame(
        {
            "source_file": m["source_file"],
            "fwhm_qc": fwhm,
            "pred_mmag": pred,
            "dmag_inst_mmag": d_inst,
        }
    ).to_csv(OUT / "c63d_p1_per_epoch.csv", index=False)
    return {"elapsed_s": round(time.perf_counter() - t0, 3), "P1": p1, "P2": p2}


def p3() -> dict:
    """Recompute era04 aperture fluxes at era03 per-star radii; mag_calib vs era03."""
    t0 = time.perf_counter()
    ens = pd.read_csv(phot(ERA04) / "comparison_stars_per_target.csv", dtype=str)
    ens_map: dict[str, list[str]] = {}
    for _, row in ens.iterrows():
        t = str(row.get("target_catalog_id") or "")
        c = str(row.get("catalog_id") or "")
        if t and c:
            ens_map.setdefault(t, []).append(c)

    unnamed = pd.read_csv(OUT / "c63c_era03_era04_ledger_v2.csv", dtype=str)
    targets = [
        str(x)
        for x in unnamed.loc[unnamed["cause"].str.contains("UNNAMED", na=False), "target"]
    ]
    for extra in (BO, FW, GH):
        if extra not in targets:
            targets.append(extra)

    r_e3: dict[str, float] = {}
    p0 = pd.read_csv(lights(ERA03) / "proc_BO_CVn_Light_001.csv", dtype={"catalog_id": str})
    p0["catalog_id"] = p0["catalog_id"].map(lambda x: _norm_cid(x) or str(x))
    for _, row in p0.iterrows():
        r_e3[str(row["catalog_id"])] = float(row["aperture_r_px"])

    need: set[str] = set()
    for tid in targets:
        need.add(tid)
        for c in ens_map.get(tid, []):
            need.add(c)

    frames = sorted(lights(ERA04).glob("proc_*.csv"))
    # mag_inst at era03 r, keyed (frame_stem, cid)
    mag_p3: dict[tuple[str, str], float] = {}
    n_meas = 0
    for pc in frames:
        df = pd.read_csv(pc, usecols=lambda c: c in ("catalog_id", "x", "y", "sky_adu_per_px_annulus"), dtype={"catalog_id": str})
        df["catalog_id"] = df["catalog_id"].map(lambda x: _norm_cid(x) or str(x))
        fits_name = pc.name.replace("proc_", "").replace(".csv", ".fits")
        fp = lights(ERA04) / fits_name
        if not fp.is_file():
            continue
        with fits.open(fp, memmap=False) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float64)
        sub = df[df["catalog_id"].isin(need)]
        for _, row in sub.iterrows():
            cid = str(row["catalog_id"])
            r = r_e3.get(cid)
            if r is None or not math.isfinite(r) or r <= 0:
                continue
            x, y = float(row["x"]), float(row["y"])
            sky = float(pd.to_numeric(pd.Series([row.get("sky_adu_per_px_annulus")]), errors="coerce").iloc[0])
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(sky)):
                continue
            ap = CircularAperture((x, y), r=r)
            tbl = aperture_photometry(data, ap)
            flux = float(tbl["aperture_sum"][0]) - sky * (math.pi * r * r)
            if flux <= 0:
                continue
            mag_p3[(pc.name, cid)] = -2.5 * math.log10(flux)
            n_meas += 1

    mag_e3: dict[tuple[str, str], float] = {}
    for pc in sorted(lights(ERA03).glob("proc_*.csv")):
        df = pd.read_csv(pc, usecols=lambda c: c in ("catalog_id", "flux"), dtype={"catalog_id": str})
        df["catalog_id"] = df["catalog_id"].map(lambda x: _norm_cid(x) or str(x))
        for _, row in df.iterrows():
            cid = str(row["catalog_id"])
            if cid not in need:
                continue
            fl = float(pd.to_numeric(pd.Series([row["flux"]]), errors="coerce").iloc[0])
            if math.isfinite(fl) and fl > 0:
                mag_e3[(pc.name, cid)] = -2.5 * math.log10(fl)

    rows = []
    for tid in targets:
        p3lc = phot(ERA03) / "lightcurves" / f"lightcurve_{tid}.csv"
        p4lc = phot(ERA04) / "lightcurves" / f"lightcurve_{tid}.csv"
        if not p3lc.is_file() or not p4lc.is_file():
            continue
        a = pd.read_csv(p3lc)
        b = pd.read_csv(p4lc)
        j = a.merge(b, on="source_file", suffixes=("_e3", "_e4"))
        comps = ens_map.get(tid, [])
        d_cal_obs = (
            pd.to_numeric(j["mag_calib_e4"], errors="coerce")
            - pd.to_numeric(j["mag_calib_e3"], errors="coerce")
        ) * 1000.0
        d_cal_p3 = []
        for _, row in j.iterrows():
            src = str(row["source_file"])
            mt = mag_p3.get((src, tid))
            mt3 = mag_e3.get((src, tid))
            if mt is None or mt3 is None:
                d_cal_p3.append(float("nan"))
                continue
            mc = [x for x in (mag_p3.get((src, c)) for c in comps) if x is not None]
            mc3 = [x for x in (mag_e3.get((src, c)) for c in comps) if x is not None]
            if len(mc) < 1 or len(mc3) < 1:
                d_cal_p3.append(float("nan"))
                continue
            rel_p3 = mt - float(np.median(mc))
            rel_e3 = mt3 - float(np.median(mc3))
            d_cal_p3.append((rel_p3 - rel_e3) * 1000.0)
        arr = np.asarray(d_cal_p3, dtype=float)
        med = float(np.nanmedian(arr))
        rows.append(
            {
                "target": tid,
                "n": int(np.isfinite(arr).sum()),
                "dmag_calib_obs_mmag": float(np.nanmedian(d_cal_obs)),
                "dmag_calib_p3_mmag": med,
                "collapse_lt_1": bool(math.isfinite(med) and abs(med) < 1.0),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "c63d_p3_recompute.csv", index=False)
    named = [BO, FW, GH]
    sub = df[df["target"].isin(named)]
    rest = df[~df["target"].isin(named)]
    return {
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "n_meas": n_meas,
        "BO_FW_GH": sub.to_dict(orient="records"),
        "n_unnamed_tested": int(len(rest)),
        "n_unnamed_collapse": int(rest["collapse_lt_1"].sum()) if len(rest) else 0,
        "unnamed_p50_abs_p3_mmag": float(np.nanmedian(np.abs(rest["dmag_calib_p3_mmag"])))
        if len(rest)
        else float("nan"),
        "pass_named": bool(sub["collapse_lt_1"].all()) if len(sub) else False,
        "pass_unnamed": bool(len(rest) and int(rest["collapse_lt_1"].sum()) == int(len(rest))),
    }


def x6e() -> dict:
    t0 = time.perf_counter()
    p3 = json.loads((ERA03 / "platesolve" / SETUP / "photometry_plan.json").read_text(encoding="utf-8"))
    p4 = json.loads((ERA04 / "platesolve" / SETUP / "photometry_plan.json").read_text(encoding="utf-8"))
    vt4 = pd.read_csv(ERA04 / "platesolve" / SETUP / "variable_targets.csv", dtype={"catalog_id": str})
    vt4["catalog_id"] = vt4["catalog_id"].map(lambda x: _norm_cid(x) or str(x))
    bbox = p4.get("safe_bbox_px")
    rows = []
    for cid in EDGE_IDS:
        r = vt4[vt4["catalog_id"] == cid]
        rec = {"catalog_id": cid}
        if len(r):
            rec["x"] = float(r.iloc[0]["x"])
            rec["y"] = float(r.iloc[0]["y"])
            rec["name"] = str(r.iloc[0].get("name") or "")
            if bbox:
                rec["outside_x"] = not (float(bbox[0]) <= rec["x"] <= float(bbox[2]))
                rec["outside_y"] = not (float(bbox[1]) <= rec["y"] <= float(bbox[3]))
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "c63d_x6e_edge.csv", index=False)
    return {
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "era03_safe_bbox_px": p3.get("safe_bbox_px"),
        "era04_safe_bbox_px": p4.get("safe_bbox_px"),
        "era03_safe_bbox_r_out_px": p3.get("safe_bbox_r_out_px"),
        "era04_safe_bbox_r_out_px": p4.get("safe_bbox_r_out_px"),
        "naxis": [2082, 1397],
        "targets": rows,
        "rule": (
            "When photometry_plan.safe_bbox_px is set, select_active_targets "
            "drops stars outside the annulus-aware intersection (no chip enlarge). "
            "era03 had safe_bbox_px=null so y~1376-1394 stayed in via enlarge; "
            "era04 bbox y1=1349.25 = NAXIS2 - r_out. Named variables at the chip "
            "edge with annulus not fully on-chip are out_of_frame."
        ),
    }


def main() -> int:
    t_all = time.perf_counter()
    out = {
        "premise": (
            "era04 snapshot_20260826 photometry after CT-REF vs era03 freeze "
            "snapshot_20260820; pixels identical; aperture_r BO 5.999->5.499"
        ),
        "x6a": x6a(),
        "x6b_p1p2": p1_p2(),
        "x6e": x6e(),
    }
    print("running P3...", flush=True)
    out["x6b_p3"] = p3()
    out["elapsed_s"] = round(time.perf_counter() - t_all, 2)
    (OUT / "c63d_x6.json").write_text(json.dumps(out, indent=2, default=str), encoding="ascii")
    print(
        "mover", out["x6a"]["named_mover"],
        "P1", out["x6b_p1p2"]["P1"],
        "P2", {k: v["spearman_rho"] for k, v in out["x6b_p1p2"]["P2"].items()},
        "P3 named", out["x6b_p3"]["pass_named"],
        "P3 unnamed", out["x6b_p3"]["n_unnamed_collapse"], "/", out["x6b_p3"]["n_unnamed_tested"],
        "s", out["elapsed_s"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
