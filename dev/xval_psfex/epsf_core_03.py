# -*- coding: ascii -*-
"""EPSF-CORE-03: model-vs-truth phase bias, call-site replication, model swap.

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
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import stats
from scipy.ndimage import shift as nd_shift
from scipy.ndimage import zoom as nd_zoom

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
OUT = REPO / "dev" / "results" / "context" / "session_20260914_epsf_core_03"
LIVE_PS = REPO / "Archive" / "Drafts" / "draft_000516" / "platesolve" / "NoFilter_60_2"
LIVE_ALN = (
    REPO / "Archive" / "Drafts" / "draft_000516" / "detrended_aligned" / "lights" / "NoFilter_60_2"
)
EPSF_FITS = LIVE_PS / "masterstar_epsf.fits"
EPSF_META = LIVE_PS / "masterstar_epsf_meta.json"
MS_PATH = LIVE_PS / "masterstars_full_match.csv"
QC_PATH = VYREF / "qc_metrics.csv"
PROC_FROZEN = VYREF / "proc_psf_flux.csv"
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
G4_EXPECT = {"csv": "bfa24039", "fits": "13e77cf8", "epsf": "172f9540"}
G4_PATHS = {
    "csv": MS_PATH,
    "fits": LIVE_PS / "MASTERSTAR.fits",
    "epsf": EPSF_FITS,
}
B1_GATE_MMAG = 1.0
OSAMP = 2
EPSF_N = 35
FINE = 8
STAMP_NATIVE = 25
FWHM_EPSF = 2.364
MOFFAT_BETA = 2.5
PHASES = tuple(float(x) for x in np.linspace(0.0, 1.0, 9))
PHASE_CELLS = [(dx, dy) for dx in PHASES for dy in PHASES]
N_REAL_ON = 30
N_REAL_OFF = 1
FRAME076 = "BO_CVn_Light_076"
OBS_TARGET_RMS_MMAG = 10.48


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
    """Production ePSF convention: sum(arr) = osamp^2 so native sum = 1 (psf_photometry.py:649-661)."""
    z = np.asarray(arr, dtype=np.float64)
    s = float(np.nansum(z))
    if not (math.isfinite(s) and s > 0):
        return z
    return z * (float(osamp) ** 2) / s


def _block_sum(fine: np.ndarray, factor: int) -> np.ndarray:
    h, w = fine.shape
    h2, w2 = h // factor, w // factor
    cut = fine[: h2 * factor, : w2 * factor]
    return cut.reshape(h2, factor, w2, factor).sum(axis=(1, 3))


def render_moffat_native(dx: float, dy: float, flux: float) -> np.ndarray:
    """Analytic Moffat on FINE grid, then FINE x FINE block-sum to native. Not ImagePSF."""
    beta = float(MOFFAT_BETA)
    fwhm = float(FWHM_EPSF)
    alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    n = STAMP_NATIVE
    nf = n * FINE
    yy, xx = np.mgrid[0:nf, 0:nf]
    x_nat = (xx + 0.5) / float(FINE)
    y_nat = (yy + 0.5) / float(FINE)
    cx = (n - 1) / 2.0 + float(dx)
    cy = (n - 1) / 2.0 + float(dy)
    r2 = (x_nat - cx) ** 2 + (y_nat - cy) ** 2
    fine = (1.0 + r2 / (alpha * alpha)) ** (-beta)
    native = _block_sum(fine, FINE)
    s = float(np.nansum(native))
    if math.isfinite(s) and s > 0:
        native = native * (float(flux) / s)
    return native


def render_psfex_native(rec: np.ndarray, psf_samp: float, dx: float, dy: float, flux: float) -> np.ndarray:
    """PSFEx stamp at PSF_SAMP: upsample to 1/FINE native, shift in that domain, block-sum.

    Never uses ImagePSF. rec pixel scale = psf_samp native-px per array-px.
    """
    a = np.asarray(rec, dtype=np.float64)
    zoom = float(psf_samp) * float(FINE)
    fine = nd_zoom(a, zoom=zoom, order=3, mode="constant", cval=0.0)
    # Shift so the array centre lands at stamp centre + (dx, dy) native.
    # After zoom, 1 fine pixel = 1/FINE native px.
    n = STAMP_NATIVE
    nf = n * FINE
    if fine.shape[0] < nf or fine.shape[1] < nf:
        pad_y = max(0, nf - fine.shape[0])
        pad_x = max(0, nf - fine.shape[1])
        fine = np.pad(fine, ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)))
    cy = (fine.shape[0] - 1) / 2.0
    cx = (fine.shape[1] - 1) / 2.0
    dest_c = (nf - 1) / 2.0
    # Want PSF centre at dest_c + dy*FINE (fine px). Current centre at cy.
    shy = (dest_c + float(dy) * FINE) - cy
    shx = (dest_c + float(dx) * FINE) - cx
    shifted = nd_shift(fine, shift=(shy, shx), order=3, mode="constant", cval=0.0)
    y0 = max(0, int(round((shifted.shape[0] - nf) / 2.0)))
    x0 = max(0, int(round((shifted.shape[1] - nf) / 2.0)))
    crop = shifted[y0 : y0 + nf, x0 : x0 + nf]
    if crop.shape != (nf, nf):
        out = np.zeros((nf, nf), dtype=np.float64)
        out[: crop.shape[0], : crop.shape[1]] = crop
        crop = out
    native = _block_sum(crop, FINE)
    s = float(np.nansum(native))
    if math.isfinite(s) and s > 0:
        native = native * (float(flux) / s)
    return native


def _paste_stamp(frame: np.ndarray, stamp: np.ndarray, x_int: int, y_int: int) -> None:
    """Place stamp so its centre pixel sits on (x_int, y_int). Phase lives inside the stamp."""
    h, w = frame.shape
    sh, sw = stamp.shape
    half_y = sh // 2
    half_x = sw // 2
    x1 = int(x_int) - half_x
    y1 = int(y_int) - half_y
    x2 = x1 + sw
    y2 = y1 + sh
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return
    frame[y1:y2, x1:x2] += stamp


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


def load_live_proc() -> pd.DataFrame:
    rows = []
    for path in sorted(LIVE_ALN.glob("proc_BO_CVn_Light_*.csv")):
        df = pd.read_csv(path)
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        df["source_file"] = path.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def call_site_facts() -> dict:
    meta = json.loads(EPSF_META.read_text(encoding="utf-8"))
    hdr = fits.getheader(LIVE_ALN / "BO_CVn_Light_001.fits")
    g, rn = _pp._psf_resolve_gain_read_noise(hdr)
    proc = pd.read_csv(LIVE_ALN / "proc_BO_CVn_Light_001.csv")
    n_psf = int(pd.to_numeric(proc["psf_flux"], errors="coerce").gt(0).sum())
    return {
        "production_call": "pipeline_catalog.py:355-364 via _fill_psf_catalog_columns; also frame_export.py:792",
        "cutout_size": "OMITTED -> None -> meta cutout_size=17 (masterstar_epsf_meta.json)",
        "cutout_size_meta": int(meta.get("cutout_size", 17)),
        "error": (
            "full-frame Poisson map float32: sqrt(|data|/gain + (rn/gain)^2) from st.gain/st.read_noise "
            "after `value or 1.0` (pipeline_catalog.py:299-306). Then _psf_fit_error_cutout_full_ccd "
            "rebuilds a MODEL-BASED map (psf_photometry.py:3117-3127); err_full_cut does not replace it."
        ),
        "use_iterative": "OMITTED -> default True (psf_photometry.py:2731)",
        "max_fit_iters": "OMITTED -> default 3 (psf_photometry.py:2732)",
        "ref_fluxes": "dao_flux from _fit_df, dtype=float, same length as star_positions (pipeline_catalog.py:347-353)",
        "apply_aperture_correction": False,
        "psf_ac_policy": "p4_none",
        "grouper_enabled": "OMITTED -> AppConfig psf_grouper_enabled; 516 pipeline_meta false; neighbor_catalog None so inactive",
        "neighbor_catalog": None,
        "nn_dist_fwhm_map": "OMITTED -> {}",
        "nn_delta_mag_map": "OMITTED -> None",
        "quality_fallback_enabled": "OMITTED -> AppConfig default True; 516 pipeline_meta true",
        "star_positions_columns": ["catalog_id", "x", "y", "name"],
        "star_positions_note": (
            f"Light_001 proc: catalog_id int64, x/y float64; {n_psf} rows with psf_flux>0 "
            "(targeted fit via _epsf_fit_catalog_ids, pipeline_catalog.py:3637-3652). "
            "M2 x,y are integer-valued floats."
        ),
        "n_psf_flux_gt0_light001": n_psf,
        "frame_data": "np.asarray(data, dtype=np.float32) of the in-memory aligned array",
        "psf_fit_gain_rn": [float(g), float(rn)],
        "core02_delta": "CORE-02 baseline passed cutout_size=17 explicit and only the 6 M2 stars",
    }


def _err_map(data32: np.ndarray, hdr) -> np.ndarray:
    gain, rn = _pp._psf_resolve_gain_read_noise(hdr)
    return np.sqrt(np.abs(data32) / max(gain, 1e-6) + (rn / max(gain, 1e-6)) ** 2).astype(np.float32)


def fit_kwargs_core02():
    return {"cutout_size": 17, "pass_error": True, "use_iterative": True, "max_fit_iters": 3}


def fit_kwargs_production():
    return {"cutout_size": None, "pass_error": True, "use_iterative": True, "max_fit_iters": 3}


def _fit(data32, hdr, stars, refs, epsf_path: Path, kw: dict):
    call = {
        "ref_fluxes": refs,
        "apply_aperture_correction": False,
        "psf_ac_policy": "p4_none",
        "use_iterative": kw.get("use_iterative", True),
        "max_fit_iters": int(kw.get("max_fit_iters", 3)),
    }
    cs = kw.get("cutout_size", None)
    if cs is not None:
        call["cutout_size"] = int(cs)
    if kw.get("pass_error", True):
        call["error"] = _err_map(data32, hdr)
    qf = kw.get("quality_fallback_enabled", None)
    if qf is not None:
        call["quality_fallback_enabled"] = bool(qf)
    ge = kw.get("grouper_enabled", None)
    if ge is not None:
        call["grouper_enabled"] = bool(ge)
    return _pp.psf_photometry_stars(data32, hdr, stars, epsf_path, **call)


def _m2_pos_refs(proc: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict[str, float]]:
    pos = []
    refs = []
    frozen = {}
    for cid in STARS:
        hit = proc[proc["catalog_id"].astype(str).str.strip() == cid]
        if hit.empty:
            pos.append({"x": 80.0, "y": 80.0, "catalog_id": cid, "name": cid})
            refs.append(float("nan"))
            frozen[cid] = float("nan")
            continue
        pos.append(
            {
                "x": float(hit.iloc[0]["x"]),
                "y": float(hit.iloc[0]["y"]),
                "catalog_id": cid,
                "name": cid,
            }
        )
        refs.append(float(pd.to_numeric(hit.iloc[0].get("dao_flux"), errors="coerce")))
        frozen[cid] = float(pd.to_numeric(hit.iloc[0]["psf_flux"], errors="coerce"))
    return pd.DataFrame(pos), np.asarray(refs, dtype=np.float64), frozen


def _all_psf_pos_refs(proc: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    work = proc.copy()
    work["catalog_id"] = work["catalog_id"].astype(str).str.strip()
    flux = pd.to_numeric(work["psf_flux"], errors="coerce")
    work = work.loc[flux.gt(0)].copy()
    pos = work[["catalog_id", "x", "y"]].copy()
    pos["name"] = pos["catalog_id"].astype(str)
    refs = pd.to_numeric(work["dao_flux"], errors="coerce").to_numpy(dtype=np.float64)
    return pos, refs


def part_b(stems: list[str], live: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, float]:
    t0 = time.perf_counter()
    idx = [0, len(stems) // 4, len(stems) // 2, (3 * len(stems)) // 4, len(stems) - 1]
    probe = [stems[i] for i in idx]
    configs = [
        ("core02_baseline", fit_kwargs_core02(), "m2"),
        ("cutout_size_None", fit_kwargs_production(), "m2"),
        ("error_omitted", {**fit_kwargs_core02(), "pass_error": False}, "m2"),
        ("quality_fallback_False", {**fit_kwargs_core02(), "quality_fallback_enabled": False}, "m2"),
        ("grouper_False_explicit", {**fit_kwargs_core02(), "grouper_enabled": False}, "m2"),
        ("fit_all_psf_rows", fit_kwargs_production(), "all"),
        ("production_all_together", fit_kwargs_production(), "m2"),
    ]
    frozen_store = {cid: [] for cid in STARS}
    flux_store = {name: {cid: [] for cid in STARS} for name, _, _ in configs}
    for stem in probe:
        fits_path = LIVE_ALN / f"{stem}.fits"
        data32 = np.asarray(fits.getdata(fits_path), dtype=np.float32)
        hdr = fits.getheader(fits_path)
        proc = live[live["source_file"] == f"proc_{stem}.csv"]
        stars_m2, refs_m2, fr = _m2_pos_refs(proc)
        for cid in STARS:
            frozen_store[cid].append(fr[cid])
        stars_all, refs_all = _all_psf_pos_refs(proc)
        for name, kw, mode in configs:
            stars = stars_all if mode == "all" else stars_m2
            refs = refs_all if mode == "all" else refs_m2
            out = _fit(data32, hdr, stars, refs, EPSF_FITS, kw)
            rec = {str(r["catalog_id"]).strip(): r for _, r in out.iterrows()}
            for cid in STARS:
                r = rec.get(cid)
                flux_store[name][cid].append(
                    float(pd.to_numeric(r.get("psf_flux"), errors="coerce")) if r is not None else float("nan")
                )
        print(f"[core03] B toggle {stem} n_all={len(stars_all)}")

    def _stack(store: dict[str, list]) -> np.ndarray:
        return _flux_to_inst_mag(np.concatenate([np.asarray(store[c], dtype=np.float64) for c in STARS]))

    fr_m = _stack(frozen_store)
    rows = []
    for name, _kw, mode in configs:
        mag = _stack(flux_store[name])
        rows.append(
            {
                "config": name,
                "star_set": mode,
                "n_rows": int(np.isfinite(mag - fr_m).sum()),
                "rms_vs_frozen_mmag": rms_after_median(mag, fr_m) * 1000.0,
                "population": "6 M2 stars x 5 frames; inst-mag after median",
            }
        )
    tog = pd.DataFrame(rows)
    tog.to_csv(OUT / "callsite_toggles.csv", index=False)
    best = min(rows, key=lambda r: r["rms_vs_frozen_mmag"] if math.isfinite(r["rms_vs_frozen_mmag"]) else 1e9)
    print(f"[core03] B probe best={best['config']} {best['rms_vs_frozen_mmag']:.4f} mmag")

    use_kw = fit_kwargs_production() if best["config"] in ("cutout_size_None", "production_all_together", "fit_all_psf_rows") else fit_kwargs_core02()
    if best["config"] == "error_omitted":
        use_kw = {**fit_kwargs_core02(), "pass_error": False}
    if best["config"] == "quality_fallback_False":
        use_kw = {**fit_kwargs_core02(), "quality_fallback_enabled": False}
    use_all = best["config"] == "fit_all_psf_rows"

    b1 = {cid: [] for cid in STARS}
    frozen_f = {cid: [] for cid in STARS}
    for i, stem in enumerate(stems):
        fits_path = LIVE_ALN / f"{stem}.fits"
        data32 = np.asarray(fits.getdata(fits_path), dtype=np.float32)
        hdr = fits.getheader(fits_path)
        proc = live[live["source_file"] == f"proc_{stem}.csv"]
        stars_m2, refs_m2, fr = _m2_pos_refs(proc)
        for cid in STARS:
            frozen_f[cid].append(fr[cid])
        if use_all:
            stars, refs = _all_psf_pos_refs(proc)
        else:
            stars, refs = stars_m2, refs_m2
        out = _fit(data32, hdr, stars, refs, EPSF_FITS, use_kw)
        rec = {str(r["catalog_id"]).strip(): r for _, r in out.iterrows()}
        for cid in STARS:
            r = rec.get(cid)
            b1[cid].append(
                float(pd.to_numeric(r.get("psf_flux"), errors="coerce")) if r is not None else float("nan")
            )
        if i % 20 == 0:
            print(f"[core03] B1 full {i+1}/{len(stems)} {stem}")

    b1_m = _flux_to_inst_mag(np.concatenate([np.asarray(b1[c]) for c in STARS]))
    fr_full = _flux_to_inst_mag(np.concatenate([np.asarray(frozen_f[c]) for c in STARS]))
    b1_vs_frozen = rms_after_median(b1_m, fr_full) * 1000.0
    n_b1 = int(np.isfinite(b1_m - fr_full).sum())
    gate_ok = bool(math.isfinite(b1_vs_frozen) and b1_vs_frozen <= B1_GATE_MMAG)
    elapsed = time.perf_counter() - t0
    headline = {
        "probe_stems": probe,
        "toggles": rows,
        "best_toggle": best["config"],
        "full_rebuild_config": best["config"],
        "full_rebuild_kwargs": {k: (None if k == "cutout_size" and use_kw.get("cutout_size") is None else use_kw[k]) for k in use_kw},
        "b1_vs_frozen_instmag_rms_mmag": b1_vs_frozen,
        "b1_n": n_b1,
        "b1_gate_mmag": B1_GATE_MMAG,
        "b1_gate_passed": gate_ok,
        "floor_mmag": float(b1_vs_frozen),
        "elapsed_s": elapsed,
        "population_full": "6 M2 stars x 134 frames; instrumental mag after median",
        "largest_remaining": (
            "none (gate passed)"
            if gate_ok
            else f"baseline-class remainder {b1_vs_frozen:.3f} mmag after call-site toggles; best probe {best['config']}"
        ),
    }
    return tog, headline, b1, elapsed


def _synth_truth_and_fit(
    *,
    stamps: dict[tuple[float, float], np.ndarray],
    hdr,
    flux: float,
    sky: float,
    gain: float,
    rn: float,
    noise: bool,
    n_real: int,
    fit_kw: dict,
    rng: np.random.Generator,
) -> list[dict]:
    spacing = 40
    ncols = len(PHASE_CELLS)
    nrows = n_real
    pad = 48
    width = pad * 2 + spacing * (ncols - 1) + 1
    height = pad * 2 + spacing * (nrows - 1) + 1
    frame = np.full((height, width), float(sky), dtype=np.float64)
    pos_rows = []
    truth = []
    for iy in range(nrows):
        for ix, (dx, dy) in enumerate(PHASE_CELLS):
            x_int = pad + ix * spacing
            y_int = pad + iy * spacing
            if x_int % 2:
                x_int += 1
            if y_int % 2:
                y_int += 1
            x_true = float(x_int) + float(dx)
            y_true = float(y_int) + float(dy)
            _paste_stamp(frame, stamps[(dx, dy)], x_int, y_int)
            cid = f"p{ix:02d}r{iy:02d}"
            # Proc peaks are integer detection pixels (CORE-01 F1), not round(true).
            x_init = float(x_int)
            y_init = float(y_int)
            pos_rows.append({"x": x_init, "y": y_init, "catalog_id": cid, "name": cid})
            truth.append((cid, dx, dy, x_true, y_true))
    if noise:
        e_mean = np.maximum(frame * gain, 0.0)
        electrons = rng.poisson(e_mean).astype(np.float64) + rng.normal(0.0, rn, size=frame.shape)
        frame = electrons / max(gain, 1e-6)
    stars = pd.DataFrame(pos_rows)
    data32 = np.asarray(frame, dtype=np.float32)
    refs = np.full(len(stars), float(flux), dtype=np.float64)
    out = _fit(data32, hdr, stars, refs, EPSF_FITS, fit_kw)
    rec = {str(r["catalog_id"]): r for _, r in out.iterrows()}
    rows = []
    for cid, dx, dy, _xt, _yt in truth:
        r = rec.get(cid)
        rec_flux = float(pd.to_numeric(r.get("psf_flux"), errors="coerce")) if r is not None else float("nan")
        bias = (
            -2.5 * math.log10(rec_flux / flux) * 1000.0
            if (math.isfinite(rec_flux) and rec_flux > 0 and flux > 0)
            else float("nan")
        )
        rows.append({"dx": dx, "dy": dy, "bias_mmag": bias, "flux_rec": rec_flux, "noise": noise})
    return rows


def _summarize_phase(rows: list[dict], n_real: int) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    g = df.groupby(["dx", "dy", "noise"], as_index=False).agg(
        n=("bias_mmag", "count"),
        bias_median_mmag=("bias_mmag", "median"),
        bias_std_mmag=("bias_mmag", "std"),
    )
    g["n_real_requested"] = n_real
    return g


def _phase_headline(cell: pd.DataFrame, phase_spread_px: float) -> dict:
    off = cell[(cell["noise"] == False) & (cell["n"] >= 1)]  # noqa: E712
    on = cell[(cell["noise"] == True) & (cell["n"] >= 1)]  # noqa: E712
    def _ptp(sub: pd.DataFrame) -> float:
        v = pd.to_numeric(sub["bias_median_mmag"], errors="coerce")
        v = v[np.isfinite(v)]
        if len(v) < 2:
            return float("nan")
        return float(v.max() - v.min())

    def _slope_0p1(sub: pd.DataFrame) -> dict[str, float]:
        """Slope over a 0.1 px window covering the live target fracX 0.587-0.682.

        Cells dx=0.5 and dx=0.625 at dy nearest 0.5. Init is the integer peak
        (proc convention), so this window is a continuous 0.125 px phase step.
        Also report full-grid Theil-Sen at the same dy for context.
        """
        dy_tgt = min(PHASES, key=lambda d: abs(d - 0.5))
        s = sub[np.isclose(sub["dy"], dy_tgt)].sort_values("dx")
        st = spearman_theil(s["dx"], s["bias_median_mmag"])
        full = float(st["slope"]) * 0.1 if math.isfinite(st["slope"]) else float("nan")
        a = s[np.isclose(s["dx"], 0.5)]
        b = s[np.isclose(s["dx"], 0.625)]
        loc = float("nan")
        if len(a) and len(b):
            dB = float(b.iloc[0]["bias_median_mmag"]) - float(a.iloc[0]["bias_median_mmag"])
            loc = dB / 0.125 * 0.1
        return {"window": loc, "theilsen_full": full, "dy": float(dy_tgt)}

    sl = _slope_0p1(off)
    slope = sl["window"]
    pred = abs(slope) * (phase_spread_px / 0.1) if math.isfinite(slope) else float("nan")
    return {
        "n_phase_cells": 81,
        "phases": list(PHASES),
        "noise_off_ptp_mmag": _ptp(off),
        "noise_on_ptp_mmag": _ptp(on),
        "slope_mmag_per_0p1px_noise_off": slope,
        "slope_theilsen_fullgrid_mmag_per_0p1px": sl["theilsen_full"],
        "slope_window_dx": [0.5, 0.625],
        "slope_window_dy": sl["dy"],
        "init": "integer peak (x_int), replicating proc peaks; true = x_int + (dx, dy)",
        "observed_phase_spread_px": phase_spread_px,
        "predicted_target_rms_mmag": pred,
        "observed_target_rms_mmag": OBS_TARGET_RMS_MMAG,
        "n_realizations_noise_on": N_REAL_ON,
        "n_realizations_noise_off": N_REAL_OFF,
    }


def part_a(fit_kw: dict) -> tuple[dict, float]:
    t0 = time.perf_counter()
    frozen = pd.read_csv(PROC_FROZEN, comment="#", dtype={"catalog_id": str})
    frozen["catalog_id"] = frozen["catalog_id"].astype(str).str.strip()
    flux = float(
        pd.to_numeric(frozen.loc[frozen["catalog_id"] == TARGET_CID, "psf_flux"], errors="coerce").median()
    )
    qc = pd.read_csv(QC_PATH, comment="#")
    sky = float(pd.to_numeric(qc["bg_median"], errors="coerce").median())
    hdr = fits.getheader(LIVE_ALN / f"{FRAME076}.fits")
    gain, rn = _pp._psf_resolve_gain_read_noise(hdr)
    matches = pd.read_csv(A2_COMPARE / "match_rows_deg2.csv", dtype={"catalog_id": str})
    matches["catalog_id"] = matches["catalog_id"].astype(str).str.strip()
    mh = matches[(matches["stem"] == FRAME076) & (matches["catalog_id"] == TARGET_CID)]
    x_s = float(mh.iloc[0]["XPSF_IMAGE"])
    y_s = float(mh.iloc[0]["YPSF_IMAGE"])
    psf_path = A2_OUT / "work" / FRAME076 / f"{FRAME076}.psf"
    model = PsfexModel(psf_path)
    rec = model.reconstruct(x_s, y_s)
    print(
        f"[core03] A T1 PSFEx {FRAME076} PSF_SAMP={model.psf_samp} rec={rec.shape} "
        f"flux={flux:.4g} sky={sky:.3f} gain,rn={gain},{rn}"
    )
    stamps_t1 = {
        (dx, dy): render_psfex_native(rec, model.psf_samp, dx, dy, flux) for dx, dy in PHASE_CELLS
    }
    stamps_t2 = {(dx, dy): render_moffat_native(dx, dy, flux) for dx, dy in PHASE_CELLS}
    rng = np.random.default_rng(20260914)
    rows_t1 = []
    rows_t1.extend(
        _synth_truth_and_fit(
            stamps=stamps_t1, hdr=hdr, flux=flux, sky=sky, gain=gain, rn=rn,
            noise=False, n_real=N_REAL_OFF, fit_kw=fit_kw, rng=rng,
        )
    )
    print("[core03] A T1 noise-off done")
    rows_t1.extend(
        _synth_truth_and_fit(
            stamps=stamps_t1, hdr=hdr, flux=flux, sky=sky, gain=gain, rn=rn,
            noise=True, n_real=N_REAL_ON, fit_kw=fit_kw, rng=rng,
        )
    )
    print("[core03] A T1 noise-on done")
    rows_t2 = []
    rows_t2.extend(
        _synth_truth_and_fit(
            stamps=stamps_t2, hdr=hdr, flux=flux, sky=sky, gain=gain, rn=rn,
            noise=False, n_real=N_REAL_OFF, fit_kw=fit_kw, rng=rng,
        )
    )
    print("[core03] A T2 noise-off done")
    rows_t2.extend(
        _synth_truth_and_fit(
            stamps=stamps_t2, hdr=hdr, flux=flux, sky=sky, gain=gain, rn=rn,
            noise=True, n_real=N_REAL_ON, fit_kw=fit_kw, rng=rng,
        )
    )
    print("[core03] A T2 noise-on done")
    c1 = _summarize_phase(rows_t1, N_REAL_ON)
    c2 = _summarize_phase(rows_t2, N_REAL_ON)
    c1.to_csv(OUT / "phase_bias_T1.csv", index=False)
    c2.to_csv(OUT / "phase_bias_T2.csv", index=False)
    phase = pd.read_csv(PHASE_CORR, dtype={"catalog_id": str})
    tgt = phase[phase["catalog_id"].astype(str).str.strip() == TARGET_CID]
    fx = pd.to_numeric(tgt["frac_x"], errors="coerce")
    fy = pd.to_numeric(tgt["frac_y"], errors="coerce")
    spread = float(max(fx.max() - fx.min(), fy.max() - fy.min()))
    h1 = _phase_headline(c1, spread)
    h2 = _phase_headline(c2, spread)
    h1["truth"] = (
        f"PSFEx deg2 {FRAME076} at target XPSF/YPSF; reconstruct; zoom to 1/{FINE} native px; "
        f"nd_shift in oversampled domain; {FINE}x{FINE} block-sum; scale stamp sum=flux. "
        f"PSF_SAMP={model.psf_samp}. Never ImagePSF."
    )
    h2["truth"] = (
        f"analytic Moffat beta={MOFFAT_BETA} FWHM={FWHM_EPSF} px on 1/{FINE} native grid, "
        f"block-sum {FINE}x{FINE}; scale stamp sum=flux. Never ImagePSF."
    )
    h1["flux"] = flux
    h1["sky"] = sky
    h2["flux"] = flux
    h2["sky"] = sky
    elapsed = time.perf_counter() - t0
    return {"T1": h1, "T2": h2, "elapsed_s": elapsed, "fit_kw": fit_kw}, elapsed


def _wrap_psfex(model: PsfexModel, x_s: float, y_s: float, tmp: Path, tag: str, meta_src: dict) -> Path:
    rec = model.reconstruct(x_s, y_s)
    arr = normalize_osamp(resample_to_osamp(rec, src_scale=model.psf_samp), OSAMP)
    tdir = tmp / tag
    tdir.mkdir(parents=True, exist_ok=True)
    tfit = tdir / "masterstar_epsf.fits"
    fits.writeto(tfit, np.asarray(arr, dtype=np.float32), overwrite=True)
    (tdir / "masterstar_epsf_meta.json").write_text(
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
    return tfit


def part_c(
    stems: list[str],
    live: pd.DataFrame,
    fit_kw: dict,
    b1_pre: dict[str, list] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, float]:
    t0 = time.perf_counter()
    matches = pd.read_csv(A2_COMPARE / "match_rows_deg2.csv", dtype={"catalog_id": str})
    matches["catalog_id"] = matches["catalog_id"].astype(str).str.strip()
    meta_src = json.loads(EPSF_META.read_text(encoding="utf-8"))
    work = A2_OUT / "work"
    model076 = PsfexModel(work / FRAME076 / f"{FRAME076}.psf")
    b1 = {cid: [] for cid in STARS} if b1_pre is None else {c: list(b1_pre[c]) for c in STARS}
    b2 = {cid: [] for cid in STARS}
    b3 = {cid: [] for cid in STARS}
    psfex_f = {cid: [] for cid in STARS}
    tmp = Path(tempfile.mkdtemp(prefix="epsf_core03_"))
    rerun_b1 = b1_pre is None
    for i, stem in enumerate(stems):
        fits_path = LIVE_ALN / f"{stem}.fits"
        data32 = np.asarray(fits.getdata(fits_path), dtype=np.float32)
        hdr = fits.getheader(fits_path)
        proc = live[live["source_file"] == f"proc_{stem}.csv"]
        stars_m2, refs_m2, _fr = _m2_pos_refs(proc)
        if rerun_b1:
            out1 = _fit(data32, hdr, stars_m2, refs_m2, EPSF_FITS, fit_kw)
            rec1 = {str(r["catalog_id"]).strip(): r for _, r in out1.iterrows()}
            for cid in STARS:
                r = rec1.get(cid)
                b1[cid].append(
                    float(pd.to_numeric(r.get("psf_flux"), errors="coerce")) if r is not None else float("nan")
                )
        psf_path = work / stem / f"{stem}.psf"
        model = PsfexModel(psf_path) if psf_path.is_file() else None
        for j, cid in enumerate(STARS):
            mh = matches[(matches["stem"] == stem) & (matches["catalog_id"] == cid)]
            if mh.empty or not bool(mh.iloc[0]["matched"]) or bool(mh.iloc[0].get("flux_nonfinite", False)):
                psfex_f[cid].append(float("nan"))
                b2[cid].append(float("nan"))
                b3[cid].append(float("nan"))
                continue
            psfex_f[cid].append(float(mh.iloc[0]["FLUX_PSF"]))
            x_s = float(mh.iloc[0]["XPSF_IMAGE"])
            y_s = float(mh.iloc[0]["YPSF_IMAGE"])
            one = stars_m2[stars_m2["catalog_id"] == cid].copy()
            ref_one = np.asarray([refs_m2[j]], dtype=np.float64)
            if model is None or one.empty:
                b2[cid].append(float("nan"))
            else:
                t2 = _wrap_psfex(model, x_s, y_s, tmp, f"{stem}_{cid}_b2", meta_src)
                out2 = _fit(data32, hdr, one, ref_one, t2, fit_kw)
                b2[cid].append(
                    float(pd.to_numeric(out2.iloc[0]["psf_flux"], errors="coerce")) if not out2.empty else float("nan")
                )
            t3 = _wrap_psfex(model076, x_s, y_s, tmp, f"{stem}_{cid}_b3", meta_src)
            out3 = _fit(data32, hdr, one, ref_one, t3, fit_kw)
            b3[cid].append(
                float(pd.to_numeric(out3.iloc[0]["psf_flux"], errors="coerce")) if not out3.empty else float("nan")
            )
        if i % 10 == 0:
            print(f"[core03] C swap {i+1}/{len(stems)} {stem}")

    def _series(d: dict[str, list], cid: str) -> np.ndarray:
        return np.asarray(d[cid], dtype=np.float64)

    d_b1_t = rebuild_delta(_series(b1, TARGET_CID), {c: _series(b1, c) for c in ENS_IDS})
    d_b2_t = rebuild_delta(_series(b2, TARGET_CID), {c: _series(b2, c) for c in ENS_IDS})
    d_b3_t = rebuild_delta(_series(b3, TARGET_CID), {c: _series(b3, c) for c in ENS_IDS})
    d_px_t = rebuild_delta(_series(psfex_f, TARGET_CID), {c: _series(psfex_f, c) for c in ENS_IDS})
    d_b1_c = rebuild_delta(_series(b1, CHECK_CID), {c: _series(b1, c) for c in ENS_IDS})
    d_b2_c = rebuild_delta(_series(b2, CHECK_CID), {c: _series(b2, c) for c in ENS_IDS})
    d_b3_c = rebuild_delta(_series(b3, CHECK_CID), {c: _series(b3, c) for c in ENS_IDS})
    d_px_c = rebuild_delta(_series(psfex_f, CHECK_CID), {c: _series(psfex_f, c) for c in ENS_IDS})
    rows = []
    for i, stem in enumerate(stems):
        rows.append(
            {
                "stem": stem,
                "b1_target": d_b1_t[i],
                "b2_target": d_b2_t[i],
                "b3_target": d_b3_t[i],
                "psfex_target": d_px_t[i],
                "b1_check": d_b1_c[i],
                "b2_check": d_b2_c[i],
                "b3_check": d_b3_c[i],
                "psfex_check": d_px_c[i],
            }
        )
    lc = pd.DataFrame(rows)
    lc.to_csv(OUT / "modelswap_lc.csv", index=False)

    def _rms(a, b):
        return rms_after_median(a, b) * 1000.0

    stats_c = {
        "rms_b1_vs_psfex_target_mmag": _rms(d_b1_t, d_px_t),
        "rms_b2_vs_psfex_target_mmag": _rms(d_b2_t, d_px_t),
        "rms_b3_vs_psfex_target_mmag": _rms(d_b3_t, d_px_t),
        "rms_b1_vs_b2_target_mmag": _rms(d_b1_t, d_b2_t),
        "rms_b1_vs_b3_target_mmag": _rms(d_b1_t, d_b3_t),
        "rms_b2_vs_b3_target_mmag": _rms(d_b2_t, d_b3_t),
        "rms_b1_vs_psfex_check_mmag": _rms(d_b1_c, d_px_c),
        "rms_b2_vs_psfex_check_mmag": _rms(d_b2_c, d_px_c),
        "rms_b3_vs_psfex_check_mmag": _rms(d_b3_c, d_px_c),
        "rms_b1_vs_b2_check_mmag": _rms(d_b1_c, d_b2_c),
        "rms_b1_vs_b3_check_mmag": _rms(d_b1_c, d_b3_c),
        "rms_b2_vs_b3_check_mmag": _rms(d_b2_c, d_b3_c),
        "population_lc": "134 identical-ensemble epochs; pinned 4-star AIJ flux-sum; median-removed RMS",
        "psfex_wrap": (
            "reconstruct at XPSF/YPSF; bilinear to 35x35 dest_scale=0.5 (osamp=2); "
            "normalize sum=osamp^2 (psf_photometry.py:649-661); ImagePSF via temp FITS + meta fwhm_px"
        ),
        "b3_model": f"static PSFEx deg2 of {FRAME076} evaluated at each star XPSF/YPSF each epoch",
    }
    # Race on B2 residuals (after median on the B2-vs-PSFEx-cat diff LC, i.e. B2 resid)
    b2_resid_t = d_b2_t - d_px_t
    b2_resid_c = d_b2_c - d_px_c
    b2_resid_t = b2_resid_t - np.nanmedian(b2_resid_t)
    b2_resid_c = b2_resid_c - np.nanmedian(b2_resid_c)
    shape = pd.read_csv(SHAPE_DEG2, dtype={"catalog_id": str})
    shape["catalog_id"] = shape["catalog_id"].astype(str).str.strip()
    phase = pd.read_csv(PHASE_CORR, dtype={"catalog_id": str})
    phase["catalog_id"] = phase["catalog_id"].astype(str).str.strip()
    race_rows = []
    for cid, role, resid in (
        (TARGET_CID, "target", b2_resid_t),
        (CHECK_CID, "check", b2_resid_c),
    ):
        sh = shape[shape["catalog_id"] == cid][["stem", "fwhm_psfex", "qc_fwhm_px"]].copy()
        ph = phase[phase["catalog_id"] == cid][["stem", "r_phase", "psf_chi2"]].copy()
        merged = sh.merge(ph, on="stem", how="inner")
        order = {s: i for i, s in enumerate(stems)}
        merged["_i"] = merged["stem"].map(order)
        merged = merged.dropna(subset=["_i"]).sort_values("_i")
        y = np.asarray(resid[: len(stems)], dtype=np.float64) * 1000.0
        y = np.asarray([y[int(i)] if int(i) < len(y) else float("nan") for i in merged["_i"]])
        rec = {"catalog_id": cid, "role": role, "n": int(len(merged)), "resid_source": "B2_minus_PSFEx_cat"}
        preds = {
            "fwhm_psfex": merged["fwhm_psfex"],
            "r_phase": merged["r_phase"],
            "psf_chi2": merged["psf_chi2"],
            "qc_fwhm_px": merged["qc_fwhm_px"],
        }
        best_name, best_r2 = "", -1.0
        for name, col in preds.items():
            stt = spearman_theil(pd.to_numeric(col, errors="coerce"), y)
            rec[f"{name}_r2_rank"] = stt["r2_rank"]
            rec[f"{name}_rho"] = stt["rho"]
            rec[f"{name}_p"] = stt["p"]
            if math.isfinite(stt["r2_rank"]) and stt["r2_rank"] > best_r2:
                best_r2 = stt["r2_rank"]
                best_name = name
        rec["best_regressor"] = best_name
        rec["best_r2_rank"] = best_r2
        race_rows.append(rec)
    race = pd.DataFrame(race_rows)
    race.to_csv(OUT / "modelswap_race.csv", index=False)
    stats_c["b2_race"] = race_rows
    elapsed = time.perf_counter() - t0
    stats_c["elapsed_s"] = elapsed
    return lc, race, stats_c, elapsed


def readings(a_head: dict, b_head: dict, c_head: dict) -> list[str]:
    fired = []
    floor = float(b_head.get("floor_mmag") or 1.396)
    thr = max(3.0, 2.0 * floor)
    b1t = float(c_head.get("rms_b1_vs_psfex_target_mmag", float("nan")))
    b1c = float(c_head.get("rms_b1_vs_psfex_check_mmag", float("nan")))
    b2t = float(c_head.get("rms_b2_vs_psfex_target_mmag", float("nan")))
    b2c = float(c_head.get("rms_b2_vs_psfex_check_mmag", float("nan")))
    b3t = float(c_head.get("rms_b3_vs_psfex_target_mmag", float("nan")))
    b3c = float(c_head.get("rms_b3_vs_psfex_check_mmag", float("nan")))
    b1_bad = (math.isfinite(b1t) and b1t >= 10.0) or (math.isfinite(b1c) and b1c >= 10.0)
    b2_ok = (math.isfinite(b2t) and b2t <= thr) and (math.isfinite(b2c) and b2c <= thr)
    b3_ok = (math.isfinite(b3t) and b3t <= thr) and (math.isfinite(b3c) and b3c <= thr)
    if b2_ok and b1_bad and (not b3_ok):
        fired.append(
            f"R-Q1: MODEL per-frame; RMS(B2 vs PSFEx cat) <= {thr:.2f} mmag on both stars "
            "while B1 stays >= 10; B3 does not match B2."
        )
    if b2_ok and b3_ok and b1_bad:
        fired.append(
            "R-Q2: MODEL static shape; B3 achieves what B2 achieves; a static but better-built ePSF suffices."
        )
    b2_bad = (math.isfinite(b2t) and b2t >= 10.0) or (math.isfinite(b2c) and b2c >= 10.0)
    b3_bad = (math.isfinite(b3t) and b3t >= 10.0) or (math.isfinite(b3c) and b3c >= 10.0)
    if b2_bad and b3_bad:
        fired.append(
            "R-Q3: MACHINERY/SKY; B2 and B3 stay >= 10 mmag on the check or the target."
        )
    pred = float(a_head.get("T1", {}).get("predicted_target_rms_mmag", float("nan")))
    if not math.isfinite(pred):
        pred = float(a_head.get("T2", {}).get("predicted_target_rms_mmag", float("nan")))
    t1p = float(a_head.get("T1", {}).get("predicted_target_rms_mmag", float("nan")))
    t2p = float(a_head.get("T2", {}).get("predicted_target_rms_mmag", float("nan")))
    pred_use = max(v for v in (t1p, t2p) if math.isfinite(v)) if any(math.isfinite(v) for v in (t1p, t2p)) else float("nan")
    if math.isfinite(pred_use) and pred_use >= 5.0:
        fired.append(
            f"R-Q4: PHASE; Part A slope x observed phase spread = {pred_use:.2f} mmag (>=5) of the target RMS."
        )
    if not fired:
        fired.append("R-Q0: none of R-Q1..Q4; full table, no attribution claim.")
    return fired


def _write_summary(facts: dict, a_head: dict, b_head: dict, c_head: dict) -> list[str]:
    fired = readings(a_head, b_head, c_head)
    g4 = g4_live_516()
    summary = {
        "decision_D_EPSF_SWAP_DIFF_01": (
            "model swap is a DIFFERENTIAL measurement inside the harness; "
            "harness-vs-production floor is the resolution limit (CORE-02 1.396 mmag / achieved B1). "
            "CORE-01 1.0 mmag gate is NOT relaxed."
        ),
        "architect_error_24": (
            "CORE-02 H-PEAK ~59.8 kADU was a Gaussian idealization presented with too much confidence "
            "and grid-inconsistent; measured aligned-grid peak ~34k (calibrated p95 45260). "
            "Root class: prediction not calibrated against a measured quantity before use."
        ),
        "carried": ["architect_error_23", "GAIN-FALSY-01", "FIXPOS-NOOP-01"],
        "call_site": facts,
        "part_a": a_head,
        "part_b": b_head,
        "part_c": c_head,
        "readings": fired,
        "g4": g4,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[core03] readings", fired)
    print("[core03] g4", g4)
    return fired


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--a-only", action="store_true", help="re-run Part A; keep existing B/C")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    facts = call_site_facts()
    print("[core03] call-site", facts["production_call"], "n_psf", facts["n_psf_flux_gt0_light001"])
    kw = fit_kwargs_core02()
    if args.a_only:
        prev = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        a_head, _ta = part_a(kw)
        print("[core03] A T1", a_head["T1"])
        print("[core03] A T2", a_head["T2"])
        _write_summary(facts, a_head, prev["part_b"], prev["part_c"])
        return 0
    stems = load_stems()
    print(f"[core03] n_stems={len(stems)}")
    live = load_live_proc()
    live["catalog_id"] = live["catalog_id"].astype(str).str.strip()
    _tog, b_head, b1_fluxes, _tb = part_b(stems, live)
    print("[core03] B", {k: b_head[k] for k in ("best_toggle", "b1_vs_frozen_instmag_rms_mmag", "b1_gate_passed", "floor_mmag", "elapsed_s")})
    fit_kw = b_head.get("full_rebuild_kwargs") or fit_kwargs_core02()
    kw = {
        "cutout_size": fit_kw.get("cutout_size"),
        "pass_error": bool(fit_kw.get("pass_error", True)),
        "use_iterative": bool(fit_kw.get("use_iterative", True)),
        "max_fit_iters": int(fit_kw.get("max_fit_iters", 3)),
    }
    if "quality_fallback_enabled" in fit_kw:
        kw["quality_fallback_enabled"] = fit_kw["quality_fallback_enabled"]
    if "grouper_enabled" in fit_kw:
        kw["grouper_enabled"] = fit_kw["grouper_enabled"]
    a_head, _ta = part_a(kw)
    print("[core03] A T1", a_head["T1"])
    print("[core03] A T2", a_head["T2"])
    _lc, _race, c_head, _tc = part_c(stems, live, kw, b1_fluxes)
    c_head["floor_mmag"] = b_head["floor_mmag"]
    print("[core03] C", {k: c_head[k] for k in c_head if k.startswith("rms_") or k in ("elapsed_s",)})
    _write_summary(facts, a_head, b_head, c_head)
    return 0


if __name__ == "__main__":
    sys.exit(main())
