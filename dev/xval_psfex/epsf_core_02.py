# -*- coding: ascii -*-
"""EPSF-CORE-02: peak-ADU census (H-PEAK) and harness delta hunt.

Dev-only. src_py must not import this module. Live 516/Archive read-only.
Linux a2/ is read-only and gitignored.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import time
import warnings
from pathlib import Path

import astroalign as aa
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src_py") not in sys.path:
    sys.path.insert(0, str(REPO / "src_py"))
if str(REPO / "dev" / "xval_psfex") not in sys.path:
    sys.path.insert(0, str(REPO / "dev" / "xval_psfex"))

from epsf_shape_01 import PsfexModel  # noqa: E402

import psf_photometry as _pp  # noqa: E402

SESSION_A2 = REPO / "dev" / "results" / "context" / "session_20260907_epsfxval_a2"
SESSION_SHAPE = REPO / "dev" / "results" / "context" / "session_20260914_epsf_shape_01"
SESSION_CORE1 = REPO / "dev" / "results" / "context" / "session_20260914_epsf_core_01"
A2_OUT = SESSION_A2 / "a2" / "out"
A2_COMPARE = SESSION_A2 / "a2_compare"
VYREF = SESSION_A2 / "vyvar_reference"
OUT = REPO / "dev" / "results" / "context" / "session_20260914_epsf_core_02"
LIVE_PS = REPO / "Archive" / "Drafts" / "draft_000516" / "platesolve" / "NoFilter_60_2"
LIVE_ALN = (
    REPO / "Archive" / "Drafts" / "draft_000516" / "detrended_aligned" / "lights" / "NoFilter_60_2"
)
LIVE_CAL = REPO / "Archive" / "Drafts" / "draft_000516" / "calibrated" / "lights" / "NoFilter_60_2"
EPSF_FITS = LIVE_PS / "masterstar_epsf.fits"
MS_PATH = LIVE_PS / "masterstars_full_match.csv"
PROC_FROZEN = VYREF / "proc_psf_flux.csv"
PIPE_META = LIVE_PS / "photometry" / "pipeline_meta.json"
GPT_JSON = LIVE_PS / "photometry" / "gain_photon_transfer.json"
SHAPE_DEG2 = SESSION_SHAPE / "shape_metrics_deg2.csv"
PHASE_CORR = SESSION_CORE1 / "phase_corr.csv"

TARGET_CID = "1498613634033133184"
CHECK_CID = "1497613731286514432"
ENS_IDS = [
    "1497771992240531712",
    "1499200223486564608",
    "1497974027502858240",
    "1497368849430107904",
]
STARS = [TARGET_CID, CHECK_CID] + ENS_IDS
ROLES = {
    TARGET_CID: "target",
    CHECK_CID: "check",
    ENS_IDS[0]: "ens1",
    ENS_IDS[1]: "ens2",
    ENS_IDS[2]: "ens3",
    ENS_IDS[3]: "ens4",
}
TOGGLE_STARS = [TARGET_CID, CHECK_CID, ENS_IDS[0], ENS_IDS[1], ENS_IDS[2]]
G4_EXPECT = {"csv": "bfa24039", "fits": "13e77cf8", "epsf": "172f9540"}
G4_PATHS = {
    "csv": MS_PATH,
    "fits": LIVE_PS / "MASTERSTAR.fits",
    "epsf": EPSF_FITS,
}
SAT_60000 = 60000.0
SAT_KNEE_PROXY = 0.80 * 65535.0  # 52428; D1-2 unmeasured
PEAK_BOX_HALF = 3
B1_GATE_MMAG = 1.0
OSAMP = 2
EPSF_N = 35
G_PT = 0.6370667331227862
RN_DB = 15.2
_ORIG_REFINE = _pp._residual_annulus_sky_per_px
_ORIG_GAIN = _pp._psf_resolve_gain_read_noise


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def g4_live_516() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for key, path in G4_PATHS.items():
        digest = _sha256_file(path)
        out[key] = {
            "prefix": digest[:8],
            "verdict": "PASS" if digest.startswith(G4_EXPECT[key]) else "FAIL",
        }
    return out


def box_peak_max(arr: np.ndarray, x: float, y: float, half: int = PEAK_BOX_HALF) -> float:
    a = np.asarray(arr)
    if a.ndim != 2 or not (math.isfinite(x) and math.isfinite(y)):
        return float("nan")
    h, w = a.shape
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    if y0 >= y1 or x0 >= x1:
        return float("nan")
    return float(np.nanmax(a[y0:y1, x0:x1]))


def _flux_to_inst_mag(flux: np.ndarray) -> np.ndarray:
    f = np.asarray(flux, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def rebuild_delta(target_flux: np.ndarray, comp_flux: dict[str, np.ndarray]) -> np.ndarray:
    ft = np.asarray(target_flux, dtype=np.float64)
    n = len(ft)
    delta = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not (math.isfinite(float(ft[i])) and float(ft[i]) > 0):
            continue
        csum = 0.0
        missing = False
        for cid in ENS_IDS:
            arr = comp_flux.get(cid)
            if arr is None or i >= len(arr):
                missing = True
                break
            fv = float(arr[i])
            if not (math.isfinite(fv) and fv > 0):
                missing = True
                break
            csum += fv
        if missing or csum <= 0:
            continue
        delta[i] = -2.5 * math.log10(float(ft[i]) / csum)
    return delta


def rms_after_median(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 8:
        return float("nan")
    d = d - float(np.median(d))
    return float(np.sqrt(np.mean(d * d)))


def spearman_theil(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(xx) & np.isfinite(yy)
    xx, yy = xx[ok], yy[ok]
    n = int(xx.size)
    if n < 8:
        return {
            "n": float(n),
            "rho": float("nan"),
            "p": float("nan"),
            "slope": float("nan"),
            "r2_rank": float("nan"),
        }
    rho, p = stats.spearmanr(xx, yy)
    slope = float("nan")
    try:
        slope = float(stats.theilslopes(yy, xx)[0])
    except Exception:  # noqa: BLE001
        slope = float("nan")
    rho_f = float(rho)
    return {
        "n": float(n),
        "rho": rho_f,
        "p": float(p),
        "slope": slope,
        "r2_rank": float(rho_f * rho_f) if math.isfinite(rho_f) else float("nan"),
    }


def resample_to_osamp(arr: np.ndarray, src_scale: float, n: int = EPSF_N, dest_scale: float = 0.5) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    h, w = a.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    oc = (n - 1) / 2.0
    iy, ix = np.mgrid[0:n, 0:n]
    ys = cy + (iy.astype(np.float64) - oc) * float(dest_scale) / float(src_scale)
    xs = cx + (ix.astype(np.float64) - oc) * float(dest_scale) / float(src_scale)
    out = np.zeros((n, n), dtype=np.float64)
    ok = (ys >= 0) & (xs >= 0) & (ys <= h - 1) & (xs <= w - 1)
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(h - 1, y0 + 1)
    x1 = np.minimum(w - 1, x0 + 1)
    wy = ys - y0
    wx = xs - x0
    y0c = np.clip(y0, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    y1c = np.clip(y1, 0, h - 1)
    x1c = np.clip(x1, 0, w - 1)
    v00 = a[y0c, x0c]
    v01 = a[y0c, x1c]
    v10 = a[y1c, x0c]
    v11 = a[y1c, x1c]
    out[ok] = (
        (1 - wy[ok]) * ((1 - wx[ok]) * v00[ok] + wx[ok] * v01[ok])
        + wy[ok] * ((1 - wx[ok]) * v10[ok] + wx[ok] * v11[ok])
    )
    return out


def normalize_osamp(arr: np.ndarray, osamp: int = OSAMP) -> np.ndarray:
    z = np.asarray(arr, dtype=np.float64)
    s = float(np.nansum(z))
    if not (math.isfinite(s) and s > 0):
        return z
    return z * (float(osamp) ** 2) / s


def ncc_map_xy(cal: np.ndarray, aln: np.ndarray, ax: float, ay: float) -> tuple[float, float, float]:
    """Template-match a 15x15 aligned cutout onto calibrated; return (x,y,ncc)."""
    th = 7
    search = 45
    h, w = aln.shape
    axi, ayi = int(round(ax)), int(round(ay))
    t0, t1 = max(0, ayi - th), min(h, ayi + th + 1)
    s0, s1 = max(0, axi - th), min(w, axi + th + 1)
    tmpl = np.nan_to_num(aln[t0:t1, s0:s1], nan=0.0)
    y0 = max(0, ayi - search)
    y1 = min(cal.shape[0], ayi + search + 1)
    x0 = max(0, axi - search)
    x1 = min(cal.shape[1], axi + search + 1)
    patch = np.nan_to_num(cal[y0:y1, x0:x1], nan=0.0)
    thh, tww = tmpl.shape
    if patch.shape[0] < thh or patch.shape[1] < tww or tmpl.size < 16:
        return float("nan"), float("nan"), float("nan")
    t = tmpl - float(np.mean(tmpl))
    tss = float(np.sqrt(np.sum(t * t)))
    if tss <= 0:
        return float("nan"), float("nan"), float("nan")
    best = -2.0
    by = bx = 0
    ymax = patch.shape[0] - thh + 1
    xmax = patch.shape[1] - tww + 1
    for yy in range(ymax):
        for xx in range(xmax):
            p = patch[yy : yy + thh, xx : xx + tww]
            p0 = p - float(np.mean(p))
            pss = float(np.sqrt(np.sum(p0 * p0)))
            if pss <= 0:
                continue
            ncc = float(np.sum(t * p0) / (tss * pss))
            if ncc > best:
                best = ncc
                by = yy
                bx = xx
    cy = y0 + by + thh / 2.0 - 0.5
    cx = x0 + bx + tww / 2.0 - 0.5
    return float(cx), float(cy), float(best)


def map_aligned_to_cal(cal: np.ndarray, aln: np.ndarray, ax: float, ay: float, transform) -> dict:
    method = "aligned_fallback"
    cx = ax
    cy = ay
    score = float("nan")
    if transform is not None:
        try:
            mapped = transform((float(ax), float(ay)))
            cx = float(np.asarray(mapped).reshape(-1, 2)[0, 0])
            cy = float(np.asarray(mapped).reshape(-1, 2)[0, 1])
            method = "astroalign_inverse"
        except Exception:  # noqa: BLE001
            transform = None
    if transform is None:
        cx, cy, score = ncc_map_xy(cal, aln, ax, ay)
        method = "ncc_template"
    peak = box_peak_max(cal, cx, cy, PEAK_BOX_HALF)
    return {
        "x_cal": cx,
        "y_cal": cy,
        "peak_adu_cal": peak,
        "method": method,
        "map_score": score,
    }


def load_stems() -> list[str]:
    frozen = pd.read_csv(PROC_FROZEN, comment="#", dtype={"catalog_id": str})
    stems = (
        frozen["source_file"]
        .astype(str)
        .str.replace("proc_", "", regex=False)
        .str.replace(".csv", "", regex=False)
        .unique()
        .tolist()
    )
    return sorted(stems)


def gain_facts() -> dict:
    gpt = json.loads(GPT_JSON.read_text(encoding="utf-8")) if GPT_JSON.is_file() else {}
    auth = gpt.get("authority", {}) if isinstance(gpt, dict) else {}
    pm = json.loads(PIPE_META.read_text(encoding="utf-8"))
    dyn = pm.get("dynamic_params", {}) if isinstance(pm, dict) else {}
    rf = pm.get("resolved_facts", {}) if isinstance(pm, dict) else {}
    hdr = fits.getheader(LIVE_ALN / "BO_CVn_Light_001.fits")
    g_hdr = hdr.get("GAIN", None)
    g_fit, rn_fit = _ORIG_GAIN(hdr)
    g_pt = float(auth.get("g_pt") or auth.get("value_e_per_adu_container") or G_PT)
    rn_auth = float((rf.get("read_noise") or {}).get("value") or dyn.get("read_noise") or RN_DB)
    return {
        "aligned_header_GAIN": g_hdr,
        "psf_fit_gain": float(g_fit),
        "psf_fit_rn": float(rn_fit),
        "psf_fit_cite": "psf_photometry.py:2268 value or 1.0; GAIN=0.0 -> (1.0, 10.0)",
        "aperture_g_pt": g_pt,
        "aperture_g_pt_source": str(auth.get("source", "g_pt")),
        "aperture_rn_db": rn_auth,
        "aperture_rn_source": str((rf.get("read_noise") or {}).get("source", "db")),
        "correct_gain_this_rig": g_pt,
        "correct_rn_this_rig": rn_auth,
        "defect_GAIN_FALSY_01": True,
        "defect_FIXPOS_NOOP_01": True,
    }


def part_a(stems: list[str]) -> tuple[pd.DataFrame, dict, float]:
    t0 = time.perf_counter()
    frozen = pd.read_csv(PROC_FROZEN, comment="#", dtype={"catalog_id": str})
    frozen["catalog_id"] = frozen["catalog_id"].astype(str).str.strip()
    m2t = pd.read_csv(A2_COMPARE / "m2_epochs_target.csv")
    m2c = pd.read_csv(A2_COMPARE / "m2_epochs_check.csv")
    resid = {
        TARGET_CID: dict(zip(m2t["stem"].astype(str), pd.to_numeric(m2t["resid_after_median"], errors="coerce"))),
        CHECK_CID: dict(zip(m2c["stem"].astype(str), pd.to_numeric(m2c["resid_after_median"], errors="coerce"))),
    }
    rows = []
    n_aa = n_ncc = n_fb = 0
    for i, stem in enumerate(stems):
        cal_path = LIVE_CAL / f"{stem}.fits"
        aln_path = LIVE_ALN / f"{stem}.fits"
        proc_path = LIVE_ALN / f"proc_{stem}.csv"
        cal = np.asarray(fits.getdata(cal_path), dtype=np.float64)
        aln = np.asarray(fits.getdata(aln_path), dtype=np.float64)
        proc = pd.read_csv(proc_path, dtype={"catalog_id": str})
        proc["catalog_id"] = proc["catalog_id"].astype(str).str.strip()
        transform = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                transform = aa.find_transform(aln, cal, max_control_points=50)[0]
        except Exception:  # noqa: BLE001
            transform = None
        for cid in STARS:
            hit = proc[proc["catalog_id"] == cid]
            if hit.empty:
                continue
            ax = float(hit.iloc[0]["x"])
            ay = float(hit.iloc[0]["y"])
            mapped = map_aligned_to_cal(cal, aln, ax, ay, transform)
            if mapped["method"] == "astroalign_inverse":
                n_aa += 1
            elif mapped["method"] == "ncc_template":
                n_ncc += 1
            else:
                n_fb += 1
            chi2 = float(pd.to_numeric(hit.iloc[0].get("psf_chi2"), errors="coerce"))
            flux = float(pd.to_numeric(hit.iloc[0].get("psf_flux"), errors="coerce"))
            sat_raw = float(pd.to_numeric(hit.iloc[0].get("peak_max_adu_raw"), errors="coerce")) if "peak_max_adu_raw" in hit.columns else float("nan")
            peak_aln = box_peak_max(aln, ax, ay, PEAK_BOX_HALF)
            rkey = resid.get(cid, {}).get(stem, float("nan"))
            rows.append(
                {
                    "stem": stem,
                    "catalog_id": cid,
                    "role": ROLES[cid],
                    "grid_peak_authority": "calibrated",
                    "x_aln": ax,
                    "y_aln": ay,
                    "x_cal": mapped["x_cal"],
                    "y_cal": mapped["y_cal"],
                    "map_method": mapped["method"],
                    "map_score": mapped["map_score"],
                    "peak_adu_cal": mapped["peak_adu_cal"],
                    "peak_adu_aln_lower_bound": peak_aln,
                    "peak_adu_satdiag_raw": sat_raw,
                    "psf_flux": flux,
                    "psf_chi2": chi2,
                    "resid_after_median": float(rkey) if rkey is not None else float("nan"),
                }
            )
        if i % 20 == 0:
            print(f"[core02] A frame {i+1}/{len(stems)} {stem}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "peak_census.csv", index=False)
    census = []
    for cid in STARS:
        s = df[df["catalog_id"] == cid]
        pk = pd.to_numeric(s["peak_adu_cal"], errors="coerce")
        census.append(
            {
                "catalog_id": cid,
                "role": ROLES[cid],
                "n": int(len(s)),
                "median": float(pk.median()),
                "p95": float(pk.quantile(0.95)),
                "max": float(pk.max()),
                "frac_gt_60000": float((pk > SAT_60000).mean()),
                "n_gt_60000": int((pk > SAT_60000).sum()),
                "frac_gt_52428": float((pk > SAT_KNEE_PROXY).mean()),
                "n_gt_52428": int((pk > SAT_KNEE_PROXY).sum()),
                "median_aln_lower_bound": float(pd.to_numeric(s["peak_adu_aln_lower_bound"], errors="coerce").median()),
            }
        )
    a1 = spearman_theil(
        df.loc[df["role"] == "check", "peak_adu_cal"],
        df.loc[df["role"] == "check", "resid_after_median"] * 1000.0,
    )
    a2 = spearman_theil(
        df.loc[df["role"] == "target", "peak_adu_cal"],
        df.loc[df["role"] == "target", "resid_after_median"] * 1000.0,
    )
    a3 = spearman_theil(
        df.loc[df["role"] == "check", "peak_adu_cal"],
        df.loc[df["role"] == "check", "psf_chi2"],
    )
    elapsed = time.perf_counter() - t0
    headline = {
        "grid": "calibrated (pre-resample) lights; peak = max in 7x7 (half=3)",
        "position_method": (
            "per-frame astroalign.find_transform(aligned, calibrated) mapping proc integer "
            "(x,y) onto the calibrated grid (inverse of VYALGM=astroalign; no persisted matrix). "
            "Fallback: 15x15 aligned-cutout NCC in a 45 px search. Uncertainty: similarity-transform "
            "residual typically <1 px plus 7x7 box (peak pixel exact once seed is inside the box)."
        ),
        "sat_diag_note": (
            "proc peak_max_adu_raw places aligned x,y on the calibrated array (sat_diag.py:585-608) "
            "and missed the check on some frames (Light_001 sat_raw=2816 vs mapped 33825)."
        ),
        "thresholds": {"saturate_limit_adu": SAT_60000, "d1_2_proxy_0p80_65535": SAT_KNEE_PROXY},
        "n_positions_astroalign": n_aa,
        "n_positions_ncc": n_ncc,
        "n_positions_fallback": n_fb,
        "per_star": census,
        "A1_check_resid_vs_peak": a1,
        "A2_target_resid_vs_peak": a2,
        "A3_check_chi2_vs_peak": a3,
        "elapsed_s": elapsed,
        "population": "6 M2 stars x 134 epochs; resid_after_median from m2_epochs_*.csv",
    }
    return df, headline, elapsed


def part_b(peaks: pd.DataFrame) -> tuple[pd.DataFrame, dict, float]:
    t0 = time.perf_counter()
    shape = pd.read_csv(SHAPE_DEG2, dtype={"catalog_id": str})
    shape["catalog_id"] = shape["catalog_id"].astype(str).str.strip()
    phase = pd.read_csv(PHASE_CORR, dtype={"catalog_id": str})
    phase["catalog_id"] = phase["catalog_id"].astype(str).str.strip()
    rows = []
    predictors = {
        "peak_adu_cal": "peak_adu_cal",
        "fwhm_psfex": "fwhm_psfex",
        "r_phase": "r_phase",
        "psf_chi2": "psf_chi2",
        "qc_fwhm_px": "qc_fwhm_px",
    }
    for cid, role in ((TARGET_CID, "target"), (CHECK_CID, "check")):
        pk = peaks[peaks["catalog_id"] == cid][["stem", "peak_adu_cal", "resid_after_median", "psf_chi2"]].copy()
        sh = shape[shape["catalog_id"] == cid][["stem", "fwhm_psfex", "qc_fwhm_px"]].copy()
        ph = phase[phase["catalog_id"] == cid][["stem", "r_phase"]].copy()
        merged = pk.merge(sh, on="stem", how="inner").merge(ph, on="stem", how="inner")
        resid = pd.to_numeric(merged["resid_after_median"], errors="coerce") * 1000.0
        best_name = ""
        best_r2 = -1.0
        rec = {"catalog_id": cid, "role": role, "n": int(len(merged))}
        for name, col in predictors.items():
            st = spearman_theil(pd.to_numeric(merged[col], errors="coerce"), resid)
            rec[f"{name}_rho"] = st["rho"]
            rec[f"{name}_p"] = st["p"]
            rec[f"{name}_r2_rank"] = st["r2_rank"]
            rec[f"{name}_theilsen"] = st["slope"]
            r2 = st["r2_rank"]
            if math.isfinite(r2) and r2 > best_r2:
                best_r2 = r2
                best_name = name
        rec["best_regressor"] = best_name
        rec["best_r2_rank"] = best_r2
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "regressor_race.csv", index=False)
    elapsed = time.perf_counter() - t0
    headline = {
        "population": "134 identical-ensemble epochs; rank R^2 = Spearman rho^2; Theil-Sen slope recorded",
        "table": rows,
        "elapsed_s": elapsed,
    }
    return out, headline, elapsed


def _err_map(data32: np.ndarray, hdr) -> np.ndarray:
    gain, rn = _pp._psf_resolve_gain_read_noise(hdr)
    return np.sqrt(np.abs(data32) / max(gain, 1e-6) + (rn / max(gain, 1e-6)) ** 2).astype(np.float32)


def _sky_off(*_a, **_k):
    return float("nan"), "disabled"


def _gain_gpt(_hdr):
    return float(G_PT), float(RN_DB)


def _fit_frame(data32, hdr, stars, refs, *, use_iterative=True, max_fit_iters=3, pass_error=True):
    kw = {
        "cutout_size": 17,
        "ref_fluxes": refs,
        "apply_aperture_correction": False,
        "psf_ac_policy": "p4_none",
        "use_iterative": use_iterative,
        "max_fit_iters": max_fit_iters,
    }
    if pass_error:
        kw["error"] = _err_map(data32, hdr)
    return _pp.psf_photometry_stars(data32, hdr, stars, EPSF_FITS, **kw)


def _restore_hooks():
    _pp._residual_annulus_sky_per_px = _ORIG_REFINE
    _pp._psf_resolve_gain_read_noise = _ORIG_GAIN


def part_c(stems: list[str], live_proc: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame | None, float]:
    t0 = time.perf_counter()
    idx = [0, len(stems) // 4, len(stems) // 2, (3 * len(stems)) // 4, len(stems) - 1]
    probe = [stems[i] for i in idx]
    configs = [
        {"name": "baseline_core01", "refine": True, "dao": True, "gpt": False, "iterative": True, "maxiters": 3},
        {"name": "i_no_refine", "refine": False, "dao": True, "gpt": False, "iterative": True, "maxiters": 3},
        {"name": "ii_clipped_sum", "refine": True, "dao": False, "gpt": False, "iterative": True, "maxiters": 3},
        {"name": "iii_gain_gpt_rn15", "refine": True, "dao": True, "gpt": True, "iterative": True, "maxiters": 3},
        {"name": "iv_iterative_off", "refine": True, "dao": True, "gpt": False, "iterative": False, "maxiters": 3},
        {"name": "iv_maxiters_1", "refine": True, "dao": True, "gpt": False, "iterative": True, "maxiters": 1},
    ]
    frozen = live_proc
    flux_store: dict[str, dict[str, list[float]]] = {c["name"]: {cid: [] for cid in TOGGLE_STARS} for c in configs}
    frozen_store = {cid: [] for cid in TOGGLE_STARS}
    for stem in probe:
        fits_path = LIVE_ALN / f"{stem}.fits"
        data32 = np.asarray(fits.getdata(fits_path), dtype=np.float32)
        hdr = fits.getheader(fits_path)
        proc = frozen[frozen["source_file"] == f"proc_{stem}.csv"]
        pos = []
        refs = []
        for cid in TOGGLE_STARS:
            hit = proc[proc["catalog_id"] == cid]
            pos.append({"x": float(hit.iloc[0]["x"]), "y": float(hit.iloc[0]["y"]), "catalog_id": cid, "name": cid})
            refs.append(float(pd.to_numeric(hit.iloc[0].get("dao_flux"), errors="coerce")))
            frozen_store[cid].append(float(pd.to_numeric(hit.iloc[0]["psf_flux"], errors="coerce")))
        stars = pd.DataFrame(pos)
        ref_arr = np.asarray(refs, dtype=np.float64)
        for cfg in configs:
            _restore_hooks()
            if not cfg["refine"]:
                _pp._residual_annulus_sky_per_px = _sky_off
            if cfg["gpt"]:
                _pp._psf_resolve_gain_read_noise = _gain_gpt
            try:
                out = _fit_frame(
                    data32,
                    hdr,
                    stars,
                    ref_arr if cfg["dao"] else None,
                    use_iterative=cfg["iterative"],
                    max_fit_iters=cfg["maxiters"],
                )
            finally:
                _restore_hooks()
            rec = {str(r["catalog_id"]): r for _, r in out.iterrows()}
            for cid in TOGGLE_STARS:
                r = rec.get(cid)
                flux_store[cfg["name"]][cid].append(
                    float(pd.to_numeric(r.get("psf_flux"), errors="coerce")) if r is not None else float("nan")
                )
        print(f"[core02] C toggle {stem}")

    def _stack(store: dict[str, list[float]]) -> np.ndarray:
        return _flux_to_inst_mag(np.concatenate([np.asarray(store[c], dtype=np.float64) for c in TOGGLE_STARS]))

    fr_m = _stack(frozen_store)
    base_m = _stack(flux_store["baseline_core01"])
    toggle_rows = []
    for cfg in configs:
        mag = _stack(flux_store[cfg["name"]])
        vs_fr = rms_after_median(mag, fr_m) * 1000.0
        vs_base = rms_after_median(mag, base_m) * 1000.0
        n = int(np.isfinite(mag - fr_m).sum())
        toggle_rows.append(
            {
                "config": cfg["name"],
                "n_rows": n,
                "rms_vs_frozen_mmag": vs_fr,
                "rms_vs_baseline_mmag": vs_base,
                "population": "5 stars x 5 frames; inst-mag after median",
            }
        )
    tog = pd.DataFrame(toggle_rows)
    tog.to_csv(OUT / "harness_toggles.csv", index=False)
    best = min(toggle_rows, key=lambda r: r["rms_vs_frozen_mmag"] if math.isfinite(r["rms_vs_frozen_mmag"]) else 1e9)
    print(f"[core02] C toggles best={best['config']} rms_vs_frozen={best['rms_vs_frozen_mmag']:.4f} mmag")

    use_cfg = configs[0]
    for cfg in configs:
        if cfg["name"] == best["config"]:
            use_cfg = cfg
            break
    # Full 804 rebuild uses the production-matching baseline; if a toggle
    # clearly wins vs frozen, rebuild with that config (it is the missing delta).
    b1 = {cid: [] for cid in STARS}
    frozen_f = {cid: [] for cid in STARS}
    for i, stem in enumerate(stems):
        fits_path = LIVE_ALN / f"{stem}.fits"
        data32 = np.asarray(fits.getdata(fits_path), dtype=np.float32)
        hdr = fits.getheader(fits_path)
        proc = frozen[frozen["source_file"] == f"proc_{stem}.csv"]
        pos = []
        refs = []
        for cid in STARS:
            hit = proc[proc["catalog_id"] == cid]
            if hit.empty:
                pos.append({"x": 80.0, "y": 80.0, "catalog_id": cid, "name": cid})
                refs.append(float("nan"))
                frozen_f[cid].append(float("nan"))
                continue
            pos.append({"x": float(hit.iloc[0]["x"]), "y": float(hit.iloc[0]["y"]), "catalog_id": cid, "name": cid})
            refs.append(float(pd.to_numeric(hit.iloc[0].get("dao_flux"), errors="coerce")))
            frozen_f[cid].append(float(pd.to_numeric(hit.iloc[0]["psf_flux"], errors="coerce")))
        stars = pd.DataFrame(pos)
        _restore_hooks()
        if not use_cfg["refine"]:
            _pp._residual_annulus_sky_per_px = _sky_off
        if use_cfg["gpt"]:
            _pp._psf_resolve_gain_read_noise = _gain_gpt
        try:
            out = _fit_frame(
                data32,
                hdr,
                stars,
                np.asarray(refs, dtype=np.float64) if use_cfg["dao"] else None,
                use_iterative=use_cfg["iterative"],
                max_fit_iters=use_cfg["maxiters"],
            )
        finally:
            _restore_hooks()
        rec = {str(r["catalog_id"]): r for _, r in out.iterrows()}
        for cid in STARS:
            r = rec.get(cid)
            b1[cid].append(
                float(pd.to_numeric(r.get("psf_flux"), errors="coerce")) if r is not None else float("nan")
            )
        if i % 20 == 0:
            print(f"[core02] C B1 full {i+1}/{len(stems)} {stem}")

    b1_m = _flux_to_inst_mag(np.concatenate([np.asarray(b1[c]) for c in STARS]))
    fr_full = _flux_to_inst_mag(np.concatenate([np.asarray(frozen_f[c]) for c in STARS]))
    b1_vs_frozen = rms_after_median(b1_m, fr_full) * 1000.0
    n_b1 = int(np.isfinite(b1_m - fr_full).sum())
    gate_ok = bool(math.isfinite(b1_vs_frozen) and b1_vs_frozen <= B1_GATE_MMAG)
    print(f"[core02] C B1 full RMS {b1_vs_frozen:.4f} mmag n={n_b1} cfg={use_cfg['name']} gate={gate_ok}")

    lc = None
    b2_stats: dict = {
        "b2_ran": False,
        "rms_b2_vs_psfex_target_mmag": float("nan"),
        "rms_b2_vs_psfex_check_mmag": float("nan"),
        "rms_b1_vs_b2_target_mmag": float("nan"),
        "rms_b1_vs_b2_check_mmag": float("nan"),
        "rms_b1_vs_psfex_target_mmag": float("nan"),
        "rms_b1_vs_psfex_check_mmag": float("nan"),
    }
    if not gate_ok:
        print(
            f"[core02] STOP C: B1 {b1_vs_frozen:.4f} mmag > {B1_GATE_MMAG}; "
            f"best toggle {best['config']}; B2 not run"
        )
    else:
        matches = pd.read_csv(A2_COMPARE / "match_rows_deg2.csv", dtype={"catalog_id": str})
        matches["catalog_id"] = matches["catalog_id"].astype(str).str.strip()
        work = A2_OUT / "work"
        meta_src = json.loads((LIVE_PS / "masterstar_epsf_meta.json").read_text(encoding="utf-8"))
        b2 = {cid: [] for cid in STARS}
        psfex_f = {cid: [] for cid in STARS}
        tmp = Path(tempfile.mkdtemp(prefix="epsf_core02_b2_"))
        for i, stem in enumerate(stems):
            fits_path = LIVE_ALN / f"{stem}.fits"
            data32 = np.asarray(fits.getdata(fits_path), dtype=np.float32)
            hdr = fits.getheader(fits_path)
            proc = frozen[frozen["source_file"] == f"proc_{stem}.csv"]
            psf_path = work / stem / f"{stem}.psf"
            model = PsfexModel(psf_path) if psf_path.is_file() else None
            for cid in STARS:
                hit = proc[proc["catalog_id"] == cid]
                mh = matches[(matches["stem"] == stem) & (matches["catalog_id"] == cid)]
                if mh.empty or not bool(mh.iloc[0]["matched"]) or bool(mh.iloc[0].get("flux_nonfinite", False)):
                    psfex_f[cid].append(float("nan"))
                else:
                    psfex_f[cid].append(float(mh.iloc[0]["FLUX_PSF"]))
                if model is None or hit.empty or mh.empty or not bool(mh.iloc[0]["matched"]):
                    b2[cid].append(float("nan"))
                    continue
                x_s = float(mh.iloc[0]["XPSF_IMAGE"])
                y_s = float(mh.iloc[0]["YPSF_IMAGE"])
                rec = model.reconstruct(x_s, y_s)
                arr = normalize_osamp(resample_to_osamp(rec, src_scale=model.psf_samp), OSAMP)
                tdir = tmp / f"{stem}_{cid}"
                tdir.mkdir(parents=True, exist_ok=True)
                tfit = tdir / "masterstar_epsf.fits"
                fits.writeto(tfit, np.asarray(arr, dtype=np.float32), overwrite=True)
                (tfit.parent / "masterstar_epsf_meta.json").write_text(
                    json.dumps(
                        {
                            "fwhm_px": float(meta_src.get("fwhm_px", 3.3014)),
                            "cutout_size": 17,
                            "oversampling": 2,
                            "spatial_order": 0,
                            "epsf_sum_native": 1.0,
                        }
                    ),
                    encoding="utf-8",
                )
                one = pd.DataFrame(
                    [{"x": float(hit.iloc[0]["x"]), "y": float(hit.iloc[0]["y"]), "catalog_id": cid, "name": cid}]
                )
                ref_one = np.asarray(
                    [float(pd.to_numeric(hit.iloc[0].get("dao_flux"), errors="coerce"))], dtype=np.float64
                )
                _restore_hooks()
                if not use_cfg["refine"]:
                    _pp._residual_annulus_sky_per_px = _sky_off
                if use_cfg["gpt"]:
                    _pp._psf_resolve_gain_read_noise = _gain_gpt
                try:
                    out2 = _fit_frame(
                        data32,
                        hdr,
                        one,
                        ref_one if use_cfg["dao"] else None,
                        use_iterative=use_cfg["iterative"],
                        max_fit_iters=use_cfg["maxiters"],
                    )
                finally:
                    _restore_hooks()
                if out2.empty:
                    b2[cid].append(float("nan"))
                else:
                    b2[cid].append(float(pd.to_numeric(out2.iloc[0]["psf_flux"], errors="coerce")))
            if i % 20 == 0:
                print(f"[core02] C B2 {i+1}/{len(stems)} {stem}")

        def _series(d: dict[str, list], cid: str) -> np.ndarray:
            return np.asarray(d[cid], dtype=np.float64)

        d_b1_t = rebuild_delta(_series(b1, TARGET_CID), {c: _series(b1, c) for c in ENS_IDS})
        d_b2_t = rebuild_delta(_series(b2, TARGET_CID), {c: _series(b2, c) for c in ENS_IDS})
        d_px_t = rebuild_delta(_series(psfex_f, TARGET_CID), {c: _series(psfex_f, c) for c in ENS_IDS})
        d_b1_c = rebuild_delta(_series(b1, CHECK_CID), {c: _series(b1, c) for c in ENS_IDS})
        d_b2_c = rebuild_delta(_series(b2, CHECK_CID), {c: _series(b2, c) for c in ENS_IDS})
        d_px_c = rebuild_delta(_series(psfex_f, CHECK_CID), {c: _series(psfex_f, c) for c in ENS_IDS})
        rows_lc = []
        for i, stem in enumerate(stems):
            rows_lc.append(
                {
                    "stem": stem,
                    "b1_target": d_b1_t[i],
                    "b2_target": d_b2_t[i],
                    "psfex_target": d_px_t[i],
                    "b1_check": d_b1_c[i],
                    "b2_check": d_b2_c[i],
                    "psfex_check": d_px_c[i],
                }
            )
        lc = pd.DataFrame(rows_lc)
        lc.to_csv(OUT / "modelswap_lc.csv", index=False)
        b2_stats = {
            "b2_ran": True,
            "rms_b1_vs_psfex_target_mmag": rms_after_median(d_b1_t, d_px_t) * 1000.0,
            "rms_b2_vs_psfex_target_mmag": rms_after_median(d_b2_t, d_px_t) * 1000.0,
            "rms_b1_vs_b2_target_mmag": rms_after_median(d_b1_t, d_b2_t) * 1000.0,
            "rms_b1_vs_psfex_check_mmag": rms_after_median(d_b1_c, d_px_c) * 1000.0,
            "rms_b2_vs_psfex_check_mmag": rms_after_median(d_b2_c, d_px_c) * 1000.0,
            "rms_b1_vs_b2_check_mmag": rms_after_median(d_b1_c, d_b2_c) * 1000.0,
            "psfex_wrap": (
                "reconstruct at XPSF/YPSF; bilinear to 35x35 dest_scale=0.5 (osamp=2); "
                "unit native sum (sum=osamp^2); ImagePSF via temp FITS + meta fwhm_px so fit_shape stays 9x9"
            ),
        }

    elapsed = time.perf_counter() - t0
    headline = {
        "probe_stems": probe,
        "probe_stars": TOGGLE_STARS,
        "toggles": toggle_rows,
        "best_toggle": best["config"],
        "full_rebuild_config": use_cfg["name"],
        "b1_vs_frozen_instmag_rms_mmag": b1_vs_frozen,
        "b1_n": n_b1,
        "b1_gate_mmag": B1_GATE_MMAG,
        "b1_gate_passed": gate_ok,
        "b2_void": (not gate_ok),
        "elapsed_s": elapsed,
        "population_full": "6 M2 stars x 134 frames; instrumental mag after median",
        **b2_stats,
    }
    return tog, headline, lc, elapsed


def readings(a_head: dict, b_head: dict, c_head: dict) -> list[str]:
    fired = []
    a1 = a_head["A1_check_resid_vs_peak"]
    chk = next(x for x in a_head["per_star"] if x["role"] == "check")
    peak_in_range = (chk["p95"] >= SAT_KNEE_PROXY) or (chk["p95"] >= SAT_60000) or (chk["frac_gt_52428"] > 0)
    if float(a1["r2_rank"]) >= 0.2 and peak_in_range:
        fired.append(
            f"R-P1: H-PEAK supported; A1 rank R^2={a1['r2_rank']:.3f} on check AND "
            f"p95 peak={chk['p95']:.0f} ADU (calibrated grid) in clipped/non-linear range."
        )
    race = {r["role"]: r for r in b_head["table"]}
    chk_b = race["check"]
    if float(a1["r2_rank"]) < 0.2 and chk_b.get("best_regressor") == "fwhm_psfex" and float(chk_b.get("best_r2_rank", 0)) >= 0.2:
        fired.append(
            f"R-P2: A1 fails but fwhm_psfex-at-star wins Part B race with "
            f"R^2={chk_b['best_r2_rank']:.3f}."
        )
    if c_head.get("b2_ran"):
        b2t = float(c_head.get("rms_b2_vs_psfex_target_mmag", float("nan")))
        b2c = float(c_head.get("rms_b2_vs_psfex_check_mmag", float("nan")))
        b1t = float(c_head.get("rms_b1_vs_psfex_target_mmag", float("nan")))
        b1c = float(c_head.get("rms_b1_vs_psfex_check_mmag", float("nan")))
        b2_ok = (math.isfinite(b2t) and b2t <= 3.0) and (math.isfinite(b2c) and b2c <= 3.0)
        b1_bad = (math.isfinite(b1t) and b1t >= 10.0) or (math.isfinite(b1c) and b1c >= 10.0)
        if b2_ok and b1_bad:
            fired.append(
                "R-P3: Part C gate passed and B2-vs-PSFEx <= 3 mmag while B1-vs-PSFEx stays >= 10 mmag "
                "-> MODEL content is the driver."
            )
    if not fired:
        fired.append("R-P0: none of R-P1/R-P2/R-P3; full decomposition table, no attribution claim.")
    return fired


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    stems = load_stems()
    print(f"[core02] n_stems={len(stems)}")
    facts = gain_facts()
    print("[core02] gain", facts)
    live_rows = []
    for path in sorted(LIVE_ALN.glob("proc_BO_CVn_Light_*.csv")):
        df = pd.read_csv(
            path,
            dtype={"catalog_id": str},
            usecols=lambda c: c
            in {"catalog_id", "x", "y", "psf_flux", "psf_chi2", "psf_fit_ok", "dao_flux", "peak_max_adu_raw"},
        )
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        df = df[df["catalog_id"].isin(STARS)].copy()
        df["source_file"] = path.name
        live_rows.append(df)
    live = pd.concat(live_rows, ignore_index=True)

    peaks, a_head, _ta = part_a(stems)
    print("[core02] A", {k: a_head[k] for k in ("A1_check_resid_vs_peak", "A2_target_resid_vs_peak", "A3_check_chi2_vs_peak", "elapsed_s")})
    _bdf, b_head, _tb = part_b(peaks)
    print("[core02] B", b_head["table"])
    skip_c = "--skip-c" in sys.argv
    if skip_c:
        c_head = {"b2_ran": False, "b1_gate_passed": False, "toggles": [], "elapsed_s": 0.0}
        print("[core02] --skip-c")
    else:
        _tog, c_head, _lc, _tc = part_c(stems, live)
        print("[core02] C", {k: c_head[k] for k in ("best_toggle", "b1_vs_frozen_instmag_rms_mmag", "b1_gate_passed", "b2_ran", "elapsed_s")})
    fired = readings(a_head, b_head, c_head)
    print("[core02] readings", fired)
    g4 = g4_live_516()
    summary = {
        "architect_error_23": (
            "CORE-01 Part A injected the live ePSF with the same ImagePSF evaluation used by the fit, "
            "so interpolation error cancels by construction; the noise-off 1.3e-7 mmag was guaranteed. "
            "Root class: harness validated against itself."
        ),
        "defects": ["GAIN-FALSY-01", "FIXPOS-NOOP-01"],
        "gain": facts,
        "part_a": a_head,
        "part_b": b_head,
        "part_c": c_head,
        "readings": fired,
        "g4": g4,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[core02] g4", g4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
