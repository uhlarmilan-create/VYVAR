# -*- coding: ascii -*-
"""EPSF-VAL-03: flux-scale accuracy vs growth-curve-tied large aperture.

Dev-only. Zero src_py imports. Live 516 read-only. Large-aperture
photometry measured by this harness only; never written back.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import astroalign as aa
import numpy as np
import pandas as pd
from astropy.io import fits
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "dev" / "xval_psfex") not in sys.path:
    sys.path.insert(0, str(REPO / "dev" / "xval_psfex"))

from epsf_core_02 import LIVE_CAL, map_aligned_to_cal  # noqa: E402
from epsf_core_03 import (  # noqa: E402
    LIVE_ALN,
    MS_PATH,
    TARGET_CID,
    g4_live_516,
    load_stems,
)
from frame_meta import load_qc_fwhm  # noqa: E402

OUT = REPO / "dev" / "results" / "context" / "session_20260916_epsf_val_03"
GAIA_DB = REPO / "GAIA_DR3" / "vyvar_gaia_dr3.db"
QC_LIVE = REPO / "Archive" / "Drafts" / "draft_000516" / "calibrated" / "lights" / "qc_metrics.csv"

# D-EPSF-XVAL-DOD-04 criterion 2 thresholds at HEAD 17cac43.
T2B = 5.0  # mmag/mag
T2S = 15.0  # mmag robust scatter 1.4826*MAD
FWHM_AUTH = 2.364  # SHAPE-01 native ePSF FWHM (px)
PLATE_SCALE = 9.774  # arcsec/px (WCS PC; A2-PREP)
R_ISO_MULT = 8.0
R_ISO_RELAX = 6.0
PEAK_P95_MAX = 40000.0
N_OK_MIN = 130
G_LO, G_HI = 8.5, 12.0
N_MIN_ISOLATED = 12
BLEND_SUSPECTS = ("1497368849430107904", "1496804834326599424")
R_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
ANN_IN, ANN_OUT = 6.0, 8.0  # x FWHM
N_BOOT = 2000

CRITERION_QUOTE = (
    "ACCURACY (curve-of-growth-tied large aperture). On isolated, "
    "unsaturated, constant bright stars, d(s) = "
    "median_epochs(m_psf_inst - m_L_inst), where m_L is a "
    "large-aperture (r_L ~ 4 x FWHM) instrumental magnitude tied by a "
    "measured growth curve. Fit d = a + b*(G - 10): |b| <= 5.0 "
    "mmag/mag and robust scatter (1.4826 * MAD) <= 15 mmag. The colour "
    "slope c of d vs BP-RP is RECORDED (expected ~0; a nonzero value "
    "would indicate PSF colour dependence), not judged. No catalogue "
    "transformation and no blended star enters the reference."
)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes", "t"))


def haversine_arcsec(ra1, dec1, ra2, dec2) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    a = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2
    return 2.0 * math.degrees(math.asin(min(1.0, math.sqrt(a)))) * 3600.0


def gaia_neighbors(ra: float, dec: float, radius_arcsec: float) -> pd.DataFrame:
    """Local Gaia DR3 box + haversine filter (sqlite; no src_py)."""
    pad = radius_arcsec / 3600.0 * 1.2
    cosd = max(0.2, abs(math.cos(math.radians(dec))))
    ra_min, ra_max = ra - pad / cosd, ra + pad / cosd
    dec_min, dec_max = dec - pad, dec + pad
    con = sqlite3.connect(f"file:{GAIA_DB.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT source_id, ra, dec, g_mag FROM gaia_dr3 "
            "WHERE ra>=? AND ra<=? AND dec>=? AND dec<=?",
            (ra_min, ra_max, dec_min, dec_max),
        ).fetchall()
    finally:
        con.close()
    out = []
    for sid, gra, gdec, gmag in rows:
        if gra is None or gdec is None or gmag is None:
            continue
        sep = haversine_arcsec(ra, dec, float(gra), float(gdec))
        if sep <= radius_arcsec + 1e-6:
            out.append(
                {
                    "source_id": str(int(sid)),
                    "ra": float(gra),
                    "dec": float(gdec),
                    "g_mag": float(gmag),
                    "sep_arcsec": sep,
                }
            )
    return pd.DataFrame(out)


def sigma_clip_median(x: np.ndarray, n_sigma: float = 3.0, n_iter: int = 3) -> float:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    for _ in range(n_iter):
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        scale = 1.4826 * mad if mad > 0 else float(np.std(v))
        if not (math.isfinite(scale) and scale > 0):
            return med
        keep = np.abs(v - med) <= n_sigma * scale
        if keep.sum() == v.size:
            return med
        v = v[keep]
        if v.size == 0:
            return med
    return float(np.median(v))


def aperture_fluxes(
    data: np.ndarray,
    x: float,
    y: float,
    fwhm: float,
    radii_mult: tuple[float, ...],
) -> dict[float, float]:
    """photutils exact CircularAperture; sky CircularAnnulus [6,8]*FWHM sigma-clipped median."""
    r_in = float(ANN_IN * fwhm)
    r_out = float(ANN_OUT * fwhm)
    ann = CircularAnnulus((x, y), r_in=r_in, r_out=r_out)
    mask = ann.to_mask(method="center").to_image(data.shape)
    if mask is None:
        sky = float("nan")
    else:
        pix = data[mask > 0]
        sky = sigma_clip_median(pix)
    out: dict[float, float] = {}
    for m in radii_mult:
        r_px = float(m * fwhm)
        ap = CircularAperture((x, y), r=r_px)
        tab = aperture_photometry(data, ap, method="exact")
        area = float(ap.area)
        flux = float(tab["aperture_sum"][0]) - sky * area
        out[m] = flux
    return out


def load_proc_stats(stems: list[str]) -> pd.DataFrame:
    rows = []
    for stem in stems:
        df = pd.read_csv(
            LIVE_ALN / f"proc_{stem}.csv",
            dtype={"catalog_id": str},
            usecols=lambda c: c
            in ("catalog_id", "x", "y", "psf_flux", "psf_chi2", "dao_flux", "vsx_known_variable", "is_saturated"),
        )
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        df["stem"] = stem
        rows.append(df)
    allp = pd.concat(rows, ignore_index=True)
    allp["psf"] = pd.to_numeric(allp["psf_flux"], errors="coerce")
    allp["dao"] = pd.to_numeric(allp["dao_flux"], errors="coerce")
    allp["chi2"] = pd.to_numeric(allp["psf_chi2"], errors="coerce")
    allp["ok_psf"] = np.isfinite(allp["psf"]) & (allp["psf"] > 0) & np.isfinite(allp["chi2"])
    allp["ok_dao"] = np.isfinite(allp["dao"]) & (allp["dao"] > 0)
    return allp


def select_isolated(allp: pd.DataFrame, ms: pd.DataFrame, r_iso_mult: float) -> tuple[pd.DataFrame, dict]:
    r_iso_px = r_iso_mult * FWHM_AUTH
    r_iso_as = r_iso_px * PLATE_SCALE
    r2_as = 2.0 * r_iso_as
    gstat = allp.groupby("catalog_id", as_index=False).agg(
        n_ok_psf=("ok_psf", "sum"),
        n_ok_dao=("ok_dao", "sum"),
        n_sat=("is_saturated", lambda s: int(_bool_series(s).sum()) if len(s) else 0),
    )
    work = gstat.merge(
        ms[["catalog_id", "G", "bp_rp", "ra_deg", "dec_deg", "vsx", "x_ms", "y_ms"]],
        on="catalog_id",
        how="left",
    )
    # Restrict expensive Gaia neighbour queries to G-window + blend suspects.
    work["_need"] = (
        (work["G"] >= G_LO) & (work["G"] <= G_HI)
    ) | work["catalog_id"].isin(BLEND_SUSPECTS)
    work = work[work["_need"]].copy()
    rows = []
    for _, row in work.iterrows():
        cid = str(row["catalog_id"])
        g = float(row["G"])
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])
        base = {
            "catalog_id": cid,
            "G": g,
            "bp_rp": float(row["bp_rp"]),
            "ra_deg": ra,
            "dec_deg": dec,
            "n_ok_psf": int(row["n_ok_psf"]),
            "n_ok_dao": int(row["n_ok_dao"]),
            "n_sat": int(row["n_sat"]),
            "vsx": bool(row["vsx"]),
            "is_target": cid == TARGET_CID,
            "is_blend_suspect": cid in BLEND_SUSPECTS,
            "r_iso_mult": r_iso_mult,
            "r_iso_px": r_iso_px,
            "r_iso_arcsec": r_iso_as,
            "n_neigh_Gplus5_within_Riso": None,
            "n_neigh_Gplus2_within_2Riso": None,
            "min_sep_arcsec_any": None,
            "peak_p95_adu": float("nan"),
            "pass_g": bool(math.isfinite(g) and G_LO <= g <= G_HI),
            "pass_constant": False,
            "pass_n_ok": False,
            "pass_isolated": False,
            "pass_unsaturated": False,
            "pass_selection": False,
        }
        base["pass_constant"] = (not base["vsx"]) and (not base["is_target"]) and (base["n_sat"] == 0)
        base["pass_n_ok"] = int(row["n_ok_psf"]) >= N_OK_MIN
        need_iso = base["pass_g"] or base["is_blend_suspect"]
        if need_iso and math.isfinite(ra) and math.isfinite(dec):
            neigh = gaia_neighbors(ra, dec, r2_as + 5.0)
            if len(neigh):
                neigh = neigh[neigh["source_id"] != cid].copy()
                base["min_sep_arcsec_any"] = float(neigh["sep_arcsec"].min()) if len(neigh) else None
                n5 = neigh[(neigh["sep_arcsec"] <= r_iso_as) & (neigh["g_mag"] <= g + 5.0)]
                n2 = neigh[(neigh["sep_arcsec"] <= r2_as) & (neigh["g_mag"] <= g + 2.0)]
                base["n_neigh_Gplus5_within_Riso"] = int(len(n5))
                base["n_neigh_Gplus2_within_2Riso"] = int(len(n2))
                base["pass_isolated"] = int(len(n5)) == 0 and int(len(n2)) == 0
            else:
                base["n_neigh_Gplus5_within_Riso"] = 0
                base["n_neigh_Gplus2_within_2Riso"] = 0
                base["min_sep_arcsec_any"] = None
                base["pass_isolated"] = True
        rows.append(base)
    tab = pd.DataFrame(rows)
    # provisional pass without unsaturated (filled later)
    tab["pass_selection_pre_peak"] = (
        tab["pass_g"]
        & tab["pass_constant"]
        & tab["pass_n_ok"]
        & tab["pass_isolated"]
        & (~tab["is_blend_suspect"])
    )
    meta = {
        "r_iso_mult": r_iso_mult,
        "r_iso_px": r_iso_px,
        "r_iso_arcsec": r_iso_as,
        "fwhm_authority_px": FWHM_AUTH,
        "plate_scale_arcsec_px": PLATE_SCALE,
        "gaia_query": (
            f"sqlite gaia_dr3 box then haversine; DB={GAIA_DB.name}; "
            f"R_iso={r_iso_as:.3f}\"; 2*R_iso={2*r_iso_as:.3f}\"; "
            "cuts: no G_n<=G+5 within R_iso; no G_n<=G+2 within 2*R_iso"
        ),
        "vsx": "masterstars vsx_known_variable (catalog_match vsx_match_max_sep_arcsec=5.0)",
        "n_pre_peak": int(tab["pass_selection_pre_peak"].sum()),
    }
    return tab, meta


def measure_peaks(cids: list[str], stems: list[str], allp: pd.DataFrame) -> dict[str, float]:
    """CORE-02 method: astroalign inverse map, 7x7 peak on calibrated grid; return p95 per cid."""
    peaks: dict[str, list[float]] = {c: [] for c in cids}
    t0 = time.perf_counter()
    for i, stem in enumerate(stems):
        cal_path = LIVE_CAL / f"{stem}.fits"
        aln_path = LIVE_ALN / f"{stem}.fits"
        cal = np.asarray(fits.getdata(cal_path), dtype=np.float64)
        aln = np.asarray(fits.getdata(aln_path), dtype=np.float64)
        transform = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                transform = aa.find_transform(aln, cal, max_control_points=50)[0]
        except Exception:  # noqa: BLE001
            transform = None
        sub = allp[allp["stem"] == stem]
        for cid in cids:
            hit = sub[sub["catalog_id"] == cid]
            if hit.empty:
                continue
            ax = float(pd.to_numeric(hit.iloc[0]["x"], errors="coerce"))
            ay = float(pd.to_numeric(hit.iloc[0]["y"], errors="coerce"))
            if not (math.isfinite(ax) and math.isfinite(ay)):
                continue
            mapped = map_aligned_to_cal(cal, aln, ax, ay, transform)
            pk = float(mapped["peak_adu_cal"])
            if math.isfinite(pk):
                peaks[cid].append(pk)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"[val03] peak frame {i+1}/{len(stems)} elapsed={time.perf_counter()-t0:.1f}s")
    out = {}
    for cid, vals in peaks.items():
        out[cid] = float(np.percentile(vals, 95)) if vals else float("nan")
    return out


def theilsen_boot(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT) -> dict:
    ok = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[ok], y[ok]
    n = int(xx.size)
    if n < 5:
        return {
            "n": n,
            "a": float("nan"),
            "b": float("nan"),
            "b_boot_std": float("nan"),
            "robust_scatter_mmag": float("nan"),
        }
    b, a, _, _ = stats.theilslopes(yy, xx)
    resid = yy - (a + b * xx)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    bs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            bb, _, _, _ = stats.theilslopes(yy[idx], xx[idx])
            bs.append(float(bb))
        except Exception:  # noqa: BLE001
            continue
    return {
        "n": n,
        "a": float(a),
        "b": float(b),
        "b_boot_std": float(np.std(bs, ddof=1)) if len(bs) > 2 else float("nan"),
        "robust_scatter_mmag": 1.4826 * mad,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stems = load_stems()
    print(f"[val03] n_stems={len(stems)}")
    qc_map = load_qc_fwhm(QC_LIVE)
    print(f"[val03] qc_fwhm keys={len(qc_map)} source={QC_LIVE}")

    ms = pd.read_csv(MS_PATH, dtype={"catalog_id": str}, low_memory=False)
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    ms["G"] = pd.to_numeric(ms["phot_g_mean_mag"], errors="coerce")
    ms["bp_rp"] = pd.to_numeric(ms["bp_rp"], errors="coerce")
    ms["ra_deg"] = pd.to_numeric(ms["ra_deg"], errors="coerce")
    ms["dec_deg"] = pd.to_numeric(ms["dec_deg"], errors="coerce")
    ms["vsx"] = _bool_series(ms["vsx_known_variable"]) if "vsx_known_variable" in ms.columns else False
    ms["x_ms"] = pd.to_numeric(ms["x"], errors="coerce")
    ms["y_ms"] = pd.to_numeric(ms["y"], errors="coerce")

    allp = load_proc_stats(stems)
    print("[val03] isolation selection")
    tab, sel_meta = select_isolated(allp, ms, R_ISO_MULT)
    relaxed = False
    if sel_meta["n_pre_peak"] < N_MIN_ISOLATED:
        print(f"[val03] only {sel_meta['n_pre_peak']} pre-peak at R_iso={R_ISO_MULT}; relax to {R_ISO_RELAX}")
        tab, sel_meta = select_isolated(allp, ms, R_ISO_RELAX)
        relaxed = True
        sel_meta["relaxed"] = True
    else:
        sel_meta["relaxed"] = False

    # peaks for pre-pass + blend suspects
    peak_cids = sorted(
        set(tab.loc[tab["pass_selection_pre_peak"] | tab["is_blend_suspect"], "catalog_id"].astype(str))
    )
    if not peak_cids:
        peak_cids = sorted(
            tab.loc[tab["pass_g"] & tab["pass_constant"] & tab["pass_n_ok"] & tab["pass_isolated"], "catalog_id"].astype(str)
        )
    print(f"[val03] peak census n_cids={len(peak_cids)}")
    peaks = measure_peaks(peak_cids, stems, allp)
    tab["peak_p95_adu"] = tab["catalog_id"].map(peaks)
    tab["pass_unsaturated"] = tab["peak_p95_adu"].apply(
        lambda v: bool(math.isfinite(float(v)) and float(v) < PEAK_P95_MAX)
    )
    tab["pass_selection"] = (
        tab["pass_selection_pre_peak"] & tab["pass_unsaturated"] & (~tab["is_blend_suspect"])
    )
    # blend suspects: never pass, but keep metrics
    tab.loc[tab["is_blend_suspect"], "pass_selection"] = False

    n_pass = int(tab["pass_selection"].sum())
    print(f"[val03] isolated pass={n_pass} relaxed={relaxed}")
    if n_pass < N_MIN_ISOLATED and not relaxed:
        tab, sel_meta = select_isolated(allp, ms, R_ISO_RELAX)
        relaxed = True
        sel_meta["relaxed"] = True
        peak_cids = sorted(
            set(tab.loc[tab["pass_selection_pre_peak"] | tab["is_blend_suspect"], "catalog_id"].astype(str))
        )
        peaks = measure_peaks(peak_cids, stems, allp)
        tab["peak_p95_adu"] = tab["catalog_id"].map(peaks)
        tab["pass_unsaturated"] = tab["peak_p95_adu"].apply(
            lambda v: bool(math.isfinite(float(v)) and float(v) < PEAK_P95_MAX)
        )
        tab["pass_selection"] = (
            tab["pass_selection_pre_peak"] & tab["pass_unsaturated"] & (~tab["is_blend_suspect"])
        )
        n_pass = int(tab["pass_selection"].sum())
        print(f"[val03] after relax pass={n_pass}")

    cand_cols = [
        "catalog_id",
        "G",
        "bp_rp",
        "ra_deg",
        "dec_deg",
        "n_ok_psf",
        "n_ok_dao",
        "n_sat",
        "vsx",
        "is_target",
        "is_blend_suspect",
        "r_iso_mult",
        "r_iso_px",
        "r_iso_arcsec",
        "n_neigh_Gplus5_within_Riso",
        "n_neigh_Gplus2_within_2Riso",
        "min_sep_arcsec_any",
        "peak_p95_adu",
        "pass_g",
        "pass_constant",
        "pass_n_ok",
        "pass_isolated",
        "pass_unsaturated",
        "pass_selection",
    ]
    # keep candidates of interest: pass or blend or G-range with nok
    keep = tab["pass_selection"] | tab["is_blend_suspect"] | (tab["pass_g"] & tab["pass_n_ok"])
    cand_out = tab.loc[keep, cand_cols].sort_values(["pass_selection", "G"], ascending=[False, True])
    cand_out.to_csv(OUT / "isolated_candidates.csv", index=False)

    if n_pass < N_MIN_ISOLATED:
        summary = {
            "stop": True,
            "reason": f"fewer than {N_MIN_ISOLATED} isolated stars after R_iso relax",
            "n_pass": n_pass,
            "selection": sel_meta,
            "criterion_quoted": CRITERION_QUOTE,
            "g4": g4_live_516(),
        }
        (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
        print("[val03] STOP", summary["reason"])
        return 2

    selected = tab.loc[tab["pass_selection"], "catalog_id"].astype(str).tolist()
    print(f"[val03] growth curve n_stars={len(selected)}")

    # --- growth curve photometry ---
    cog_rows = []
    lc_rows = []
    t0 = time.perf_counter()
    for i, stem in enumerate(stems):
        from frame_meta import light_key

        lk = light_key(stem)
        fwhm = qc_map.get(lk)
        if fwhm is None or not (math.isfinite(fwhm) and fwhm > 0):
            print(f"[val03] WARN missing fwhm for {stem}")
            continue
        data = np.asarray(fits.getdata(LIVE_ALN / f"{stem}.fits"), dtype=np.float64)
        sub = allp[allp["stem"] == stem]
        epoch_flux: dict[str, dict[float, float]] = {}
        for cid in selected:
            hit = sub[sub["catalog_id"] == cid]
            if hit.empty:
                continue
            x = float(pd.to_numeric(hit.iloc[0]["x"], errors="coerce"))
            y = float(pd.to_numeric(hit.iloc[0]["y"], errors="coerce"))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            try:
                fl = aperture_fluxes(data, x, y, float(fwhm), R_GRID)
            except Exception as exc:  # noqa: BLE001
                print(f"[val03] ap fail {stem} {cid}: {exc}")
                continue
            epoch_flux[cid] = fl
            psf = float(pd.to_numeric(hit.iloc[0]["psf"], errors="coerce"))
            dao = float(pd.to_numeric(hit.iloc[0]["dao"], errors="coerce"))
            lc_rows.append(
                {
                    "stem": stem,
                    "catalog_id": cid,
                    "fwhm_px_qc": float(fwhm),
                    "psf_flux": psf,
                    "dao_flux": dao,
                    **{f"F_{m:g}xFWHM": fl[m] for m in R_GRID},
                }
            )
        # growth curve median f(r)=F(r)/F(5)
        for m in R_GRID:
            ratios = []
            for cid, fl in epoch_flux.items():
                f5 = fl.get(5.0)
                fr = fl.get(m)
                if (
                    f5 is not None
                    and fr is not None
                    and math.isfinite(f5)
                    and f5 > 0
                    and math.isfinite(fr)
                    and fr > 0
                ):
                    ratios.append(fr / f5)
            if ratios:
                arr = np.asarray(ratios, dtype=np.float64)
                cog_rows.append(
                    {
                        "epoch": stem,
                        "r_mult": m,
                        "r_px": m * float(fwhm),
                        "f": float(np.median(arr)),
                        "n_stars": int(arr.size),
                        "MAD": float(np.median(np.abs(arr - np.median(arr)))),
                        "fwhm_px_qc": float(fwhm),
                    }
                )
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[val03] cog frame {i+1}/{len(stems)} elapsed={time.perf_counter()-t0:.1f}s")

    cog = pd.DataFrame(cog_rows)
    cog.to_csv(OUT / "cog_table.csv", index=False)
    lc = pd.DataFrame(lc_rows)
    lc.to_csv(OUT / "large_aperture_lc.csv", index=False)

    # choose r_L: smallest r where median increment f(r+0.5)-f(r) < 1 mmag on >=90% epochs
    # 1 mmag in flux ratio: df = 1 - 10^(-0.001/2.5) ~ 0.000921
    df_lim = 1.0 - 10 ** (-0.001 / 2.5)
    r_choices = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    r_L = None
    plateau_residual_mmag = float("nan")
    r_L_note = ""
    for r in r_choices:
        r_next = r + 0.5
        # per epoch: need f(r) and f(r_next)
        ok_epochs = 0
        n_epochs = 0
        increments = []
        for stem, g in cog.groupby("epoch"):
            fr = g[g["r_mult"] == r]
            fn = g[g["r_mult"] == r_next]
            if fr.empty or fn.empty:
                continue
            n_epochs += 1
            inc = float(fn["f"].iloc[0] - fr["f"].iloc[0])
            increments.append(inc)
            if inc < df_lim:
                ok_epochs += 1
        frac = ok_epochs / n_epochs if n_epochs else 0.0
        if frac >= 0.90:
            r_L = r
            # plateau residual: median remaining (1 - f(r_L)) in mmag
            fvals = cog.loc[cog["r_mult"] == r_L, "f"].to_numpy(dtype=np.float64)
            rem = 1.0 - fvals
            rem = rem[np.isfinite(rem)]
            # mmag of missing flux relative to plateau
            plateau_residual_mmag = float(np.median(-2.5 * np.log10(np.clip(1.0 - rem, 1e-12, 1.0)) * 0 + rem))
            # Better: residual flux fraction to mmag: -2.5log10(f) vs 0 at f=1 -> approx 1.0857*rem*1000
            plateau_residual_mmag = float(np.median(rem) * (2.5 / math.log(10)) * 1000.0)
            r_L_note = f"smallest r with df<1mmag on {ok_epochs}/{n_epochs} epochs ({frac:.2%})"
            break
    if r_L is None:
        r_L = 4.0
        fvals = cog.loc[cog["r_mult"] == r_L, "f"].to_numpy(dtype=np.float64)
        rem = 1.0 - fvals
        rem = rem[np.isfinite(rem)]
        plateau_residual_mmag = float(np.median(rem) * (2.5 / math.log(10)) * 1000.0) if rem.size else float("nan")
        # residual slope: median increment at 4.0->4.5
        incs = []
        for stem, g in cog.groupby("epoch"):
            fr = g[g["r_mult"] == 4.0]
            fn = g[g["r_mult"] == 4.5]
            if fr.empty or fn.empty:
                continue
            incs.append(float(fn["f"].iloc[0] - fr["f"].iloc[0]))
        slope_mmag = float(np.median(incs) * (2.5 / math.log(10)) * 1000.0) if incs else float("nan")
        r_L_note = f"no radius met 90% criterion; used 4.0 x FWHM; residual slope 4.0->4.5 = {slope_mmag:.3f} mmag"

    print(f"[val03] r_L={r_L} x FWHM; plateau_resid_mmag={plateau_residual_mmag:.3f}; {r_L_note}")

    # build m_L, m_psf, m_ap per star/epoch
    col_L = f"F_{r_L:g}xFWHM"
    if col_L not in lc.columns:
        col_L = f"F_{float(r_L):g}xFWHM"
    # ensure column
    possible = [c for c in lc.columns if c.startswith("F_") and c.endswith("xFWHM")]
    # map
    def _flux_col(mult: float) -> str:
        for c in possible:
            # F_4xFWHM or F_4.0xFWHM
            body = c[2 : -len("xFWHM")]
            if abs(float(body) - mult) < 1e-9:
                return c
        raise KeyError(mult)

    col_L = _flux_col(float(r_L))
    acc_psf_rows = []
    acc_ap_rows = []
    per_epoch_scatters = []
    for cid in selected:
        sub = lc[lc["catalog_id"] == cid].copy()
        fl = pd.to_numeric(sub[col_L], errors="coerce")
        psf = pd.to_numeric(sub["psf_flux"], errors="coerce")
        dao = pd.to_numeric(sub["dao_flux"], errors="coerce")
        ok = (fl > 0) & np.isfinite(fl) & (psf > 0) & np.isfinite(psf)
        ok_ap = (fl > 0) & np.isfinite(fl) & (dao > 0) & np.isfinite(dao)
        if ok.sum() < 8:
            continue
        m_L = -2.5 * np.log10(fl[ok].to_numpy(dtype=np.float64))
        m_psf = -2.5 * np.log10(psf[ok].to_numpy(dtype=np.float64))
        d_ep = (m_psf - m_L) * 1000.0
        per_epoch_scatters.append(float(np.std(d_ep, ddof=1)) if d_ep.size > 2 else float("nan"))
        g = float(tab.loc[tab["catalog_id"] == cid, "G"].iloc[0])
        bprp = float(tab.loc[tab["catalog_id"] == cid, "bp_rp"].iloc[0])
        acc_psf_rows.append(
            {
                "catalog_id": cid,
                "G": g,
                "bp_rp": bprp,
                "n_epochs": int(ok.sum()),
                "d_median_mmag": float(np.median(d_ep)),
                "d_epoch_std_mmag": float(np.std(d_ep, ddof=1)) if d_ep.size > 2 else float("nan"),
            }
        )
        if ok_ap.sum() >= 8:
            m_La = -2.5 * np.log10(fl[ok_ap].to_numpy(dtype=np.float64))
            m_ap = -2.5 * np.log10(dao[ok_ap].to_numpy(dtype=np.float64))
            d_ap = (m_ap - m_La) * 1000.0
            acc_ap_rows.append(
                {
                    "catalog_id": cid,
                    "G": g,
                    "bp_rp": bprp,
                    "n_epochs": int(ok_ap.sum()),
                    "d_median_mmag": float(np.median(d_ap)),
                }
            )

    acc_psf = pd.DataFrame(acc_psf_rows)
    acc_ap = pd.DataFrame(acc_ap_rows)
    acc_psf.to_csv(OUT / "accuracy_cog_psf.csv", index=False)
    acc_ap.to_csv(OUT / "accuracy_cog_ap.csv", index=False)

    xg = acc_psf["G"].to_numpy(dtype=np.float64) - 10.0
    y = acc_psf["d_median_mmag"].to_numpy(dtype=np.float64)
    fit_g = theilsen_boot(xg, y)
    xb = acc_psf["bp_rp"].to_numpy(dtype=np.float64) - 1.0
    fit_c = theilsen_boot(xb, y)
    fit_c_rec = {
        "n": fit_c["n"],
        "a": fit_c["a"],
        "c": fit_c["b"],
        "c_boot_std": fit_c["b_boot_std"],
        "robust_scatter_mmag": fit_c["robust_scatter_mmag"],
    }
    fit_ap = {"n": 0, "a": float("nan"), "b_ap": float("nan"), "b_ap_boot_std": float("nan"), "robust_scatter_mmag": float("nan")}
    if len(acc_ap):
        fa = theilsen_boot(acc_ap["G"].to_numpy(dtype=np.float64) - 10.0, acc_ap["d_median_mmag"].to_numpy(dtype=np.float64))
        fit_ap = {
            "n": fa["n"],
            "a": fa["a"],
            "b_ap": fa["b"],
            "b_ap_boot_std": fa["b_boot_std"],
            "robust_scatter_mmag": fa["robust_scatter_mmag"],
        }

    fits_df = pd.DataFrame(
        [
            {
                "path": "psf_vs_m_L",
                "n": fit_g["n"],
                "a": fit_g["a"],
                "b": fit_g["b"],
                "b_boot_std": fit_g["b_boot_std"],
                "robust_scatter_mmag": fit_g["robust_scatter_mmag"],
                "criterion": True,
            },
            {
                "path": "psf_vs_bprp_RECORD",
                "n": fit_c_rec["n"],
                "a": fit_c_rec["a"],
                "c": fit_c_rec["c"],
                "c_boot_std": fit_c_rec["c_boot_std"],
                "robust_scatter_mmag": fit_c_rec["robust_scatter_mmag"],
                "criterion": False,
            },
            {
                "path": "aperture_vs_m_L_D5-1",
                "n": fit_ap["n"],
                "a": fit_ap["a"],
                "b_ap": fit_ap["b_ap"],
                "b_ap_boot_std": fit_ap["b_ap_boot_std"],
                "robust_scatter_mmag": fit_ap["robust_scatter_mmag"],
                "criterion": False,
            },
        ]
    )
    fits_df.to_csv(OUT / "accuracy_fits.csv", index=False)

    per_epoch_med = float(np.nanmedian(per_epoch_scatters)) if per_epoch_scatters else float("nan")
    print("[val03] fit_g", fit_g)
    print("[val03] fit_c", fit_c_rec)
    print("[val03] fit_ap", fit_ap)

    # readings
    b = abs(float(fit_g["b"]))
    scat = float(fit_g["robust_scatter_mmag"])
    fired = []
    if math.isfinite(b) and math.isfinite(scat) and b <= T2B and scat <= T2S:
        fired.append(
            f"R-X1 PASS: |b|={b:.3f} <= {T2B} AND robust scatter={scat:.3f} <= {T2S} "
            f"(n={fit_g['n']}; b_boot_std={fit_g['b_boot_std']:.3f})."
        )
    else:
        plateau_note = (
            f"plateau residual={plateau_residual_mmag:.3f} mmag "
            f"{'could' if (math.isfinite(plateau_residual_mmag) and plateau_residual_mmag >= 0.5 * max(scat, 1e-6)) else 'cannot'} "
            f"account for the scatter alone."
        )
        fired.append(
            f"R-X1 FAIL: |b|={b:.3f} (lim {T2B}), robust scatter={scat:.3f} (lim {T2S}) "
            f"(n={fit_g['n']}; b_boot_std={fit_g['b_boot_std']:.3f}). {plateau_note}"
        )
    fired.append(
        f"R-X2 RECORD: c={fit_c_rec['c']:.3f} mmag/mag (boot_std={fit_c_rec['c_boot_std']:.3f}); "
        f"per-epoch PSF-vs-m_L median std={per_epoch_med:.3f} mmag; "
        f"b_ap={fit_ap['b_ap']:.3f} mmag/mag (D5-1; n={fit_ap['n']})."
    )
    if any(x.startswith("R-X1 PASS") for x in fired):
        fired.append(
            "R-X3: R-X1 PASS; criterion 2 of DOD-04 is met on 516; "
            "EPSF-XVAL-01 closure waits only on criterion 3 at the 520 re-cut."
        )
    else:
        fired.append("R-X3: FAIL on R-X1; sequencing is Milan's.")
    print("[val03] readings", fired)

    g4 = g4_live_516()
    summary = {
        "criterion_quoted_from_HEAD": {
            "decision": "D-EPSF-XVAL-DOD-04",
            "commit": "17cac43",
            "text": CRITERION_QUOTE,
            "T2b": T2B,
            "T2s": T2S,
        },
        "radius_grid": {
            "multiples_of_FWHM": list(R_GRID),
            "fwhm_authority_isolation_px": FWHM_AUTH,
            "fwhm_photometry": "per-frame qc_metrics.csv fwhm_px (live calibrated lights)",
            "qc_path": str(QC_LIVE),
            "annulus": f"[{ANN_IN},{ANN_OUT}] x FWHM; sigma-clipped median sky",
            "production_sky_estimator_name": "sky_median_mask (comparison only; harness uses sigma-clipped median)",
            "photutils": "CircularAperture/CircularAnnulus method=exact",
            "plate_scale_arcsec_px": PLATE_SCALE,
        },
        "selection": {**sel_meta, "n_pass": n_pass, "relaxed": relaxed},
        "r_L": {"r_L_x_FWHM": r_L, "plateau_residual_mmag": plateau_residual_mmag, "note": r_L_note},
        "accuracy_psf": fit_g,
        "accuracy_colour_RECORD": fit_c_rec,
        "accuracy_aperture_D5_1": fit_ap,
        "per_epoch_agreement_median_std_mmag": per_epoch_med,
        "readings": fired,
        "g4": g4,
    }
    (OUT / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print("[val03] g4", g4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
