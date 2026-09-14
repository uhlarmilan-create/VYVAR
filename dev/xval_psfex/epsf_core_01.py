# -*- coding: ascii -*-
"""EPSF-CORE-01: fit-machinery measurement for the R-SH3 escalation.

Dev-only. src_py must not import this module. Calls the production
entry psf_photometry_stars (psf_photometry.py:2723). Linux a2/ is
read-only and gitignored.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

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
from photutils.psf import ImagePSF  # noqa: E402

import psf_photometry as _pp  # noqa: E402

SESSION_A2 = REPO / "dev" / "results" / "context" / "session_20260907_epsfxval_a2"
A2_OUT = SESSION_A2 / "a2" / "out"
A2_COMPARE = SESSION_A2 / "a2_compare"
VYREF = SESSION_A2 / "vyvar_reference"
OUT = REPO / "dev" / "results" / "context" / "session_20260914_epsf_core_01"
LIVE_PS = REPO / "Archive" / "Drafts" / "draft_000516" / "platesolve" / "NoFilter_60_2"
LIVE_LIGHTS = (
    REPO / "Archive" / "Drafts" / "draft_000516" / "detrended_aligned" / "lights" / "NoFilter_60_2"
)
SNAP_LIGHTS = (
    REPO
    / "Archive"
    / "Drafts"
    / "draft_000516_snapshot_era04_20260826"
    / "detrended_aligned"
    / "lights"
    / "NoFilter_60_2"
)
EPSF_FITS = LIVE_PS / "masterstar_epsf.fits"
EPSF_META = LIVE_PS / "masterstar_epsf_meta.json"
MS_PATH = LIVE_PS / "masterstars_full_match.csv"
QC_PATH = VYREF / "qc_metrics.csv"
PROC_FROZEN = VYREF / "proc_psf_flux.csv"
PIPE_META = LIVE_PS / "photometry" / "pipeline_meta.json"
SIDECAR = LIVE_PS / "photometry" / "lightcurves" / "lightcurve_1498613634033133184_psf.csv"

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
PHASES = (0.0, 0.125, 0.25, 0.375, 0.5)
PHASE_CELLS = [(dx, dy) for dx in PHASES for dy in PHASES]
N_REAL = 50
OSAMP = 2
EPSF_N = 35
B1_STOP_MMAG = 1.0

_FIX_POS = False
_ORIG_APPLY = _pp._apply_psf_fixed_position


def _apply_maybe(phot, *, fix: bool = False) -> None:
    do = bool(fix or _FIX_POS)
    if not do:
        return
    for obj in (getattr(phot, "psf_model", None), getattr(phot, "psf", None)):
        if obj is None:
            continue
        try:
            obj.x_0.fixed = True
            obj.y_0.fixed = True
            return
        except Exception:  # noqa: BLE001
            continue
    _ORIG_APPLY(phot, fix=True)


_pp._apply_psf_fixed_position = _apply_maybe


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


def _is_true(v: object) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def _flux_to_inst_mag(flux: np.ndarray) -> np.ndarray:
    f = np.asarray(flux, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def rebuild_delta(target_flux: np.ndarray, comp_flux: dict[str, np.ndarray]) -> np.ndarray:
    """AIJ tot_C_cnts: delta = -2.5 log10(F_t / sum F_c). Same as run_xval_a1.py:449."""
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


def resample_to_osamp(arr: np.ndarray, src_scale: float, n: int = EPSF_N, dest_scale: float = 0.5) -> np.ndarray:
    """PSFEx stamp (src_scale native-px / array-px) -> ImagePSF osamp=2 grid."""
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


def _fix_pos_in_config_py() -> bool:
    cfg_py = (REPO / "src_py" / "config.py").read_text(encoding="utf-8")
    return "psf_fix_position_enabled" in cfg_py


def confirm_f1_f4() -> dict:
    meta = json.loads(EPSF_META.read_text(encoding="utf-8"))
    pm = json.loads(PIPE_META.read_text(encoding="utf-8"))
    cfg = {}
    stack = [pm]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, dict):
            continue
        if "psf_chi2_threshold" in cur:
            cfg = cur
            break
        stack.extend(v for v in cur.values() if isinstance(v, dict))
    raw_pm = PIPE_META.read_text(encoding="utf-8")
    fit_shape_meta = meta.get("fit_shape")
    fwhm = float(meta.get("fwhm_px", 0.0))
    cutout = int(meta.get("cutout_size", 17))
    fs = _pp._fit_shape_for_cutout(cutout, fwhm_px=fwhm)
    sidecar_txt = SIDECAR.read_text(encoding="utf-8") if SIDECAR.is_file() else ""
    zp_eff = ""
    for line in sidecar_txt.splitlines():
        if line.startswith("# psf_zp_membership_effective="):
            zp_eff = line.split("=", 1)[1].strip()
    live0 = pd.read_csv(
        LIVE_LIGHTS / "proc_BO_CVn_Light_001.csv",
        nrows=1,
    )
    return {
        "spatial_order": int(meta.get("spatial_order", -1)),
        "oversampling": int(meta.get("oversampling", -1)),
        "cutout_size": cutout,
        "fwhm_px_meta": fwhm,
        "fit_shape_meta": list(fit_shape_meta) if fit_shape_meta is not None else None,
        "fit_shape_computed": list(fs),
        "psf_fix_position_in_pipeline_meta": "psf_fix_position" in raw_pm,
        "psf_fix_position_in_config_py": _fix_pos_in_config_py(),
        "psf_fix_position_used": False,
        "psf_chi2_threshold_persisted": float(cfg.get("psf_chi2_threshold", float("nan"))),
        "psf_grouper_enabled_persisted": bool(cfg.get("psf_grouper_enabled", True)),
        "psf_quality_fallback_persisted": bool(cfg.get("psf_quality_fallback_enabled", False)),
        "live_proc_has_x_fit": "x_fit" in live0.columns,
        "sidecar_zp_membership_effective": zp_eff,
        "gain_note": (
            "aligned Light_001 GAIN=0.0; _psf_resolve_gain_read_noise "
            "uses (value or 1.0) so gain=1.0, rn=10.0 "
            "(psf_photometry.py:2268-2270). Not g_pt 0.637067."
        ),
    }


def fitok_census() -> pd.DataFrame:
    frozen = pd.read_csv(PROC_FROZEN, comment="#", dtype={"catalog_id": str})
    frozen["catalog_id"] = frozen["catalog_id"].astype(str).str.strip()
    live_rows = []
    for path in sorted(LIVE_LIGHTS.glob("proc_BO_CVn_Light_*.csv")):
        df = pd.read_csv(
            path,
            dtype={"catalog_id": str},
            usecols=lambda c: c
            in {"catalog_id", "x", "y", "psf_flux", "psf_chi2", "psf_fit_ok", "dao_flux"},
        )
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        df = df[df["catalog_id"].isin(STARS)].copy()
        df["source_file"] = path.name
        live_rows.append(df)
    live = pd.concat(live_rows, ignore_index=True)
    rows = []
    for src_name, df in (("frozen", frozen), ("live", live)):
        for cid in STARS:
            s = df[df["catalog_id"] == cid]
            ok = s["psf_fit_ok"].map(_is_true)
            rows.append(
                {
                    "source": src_name,
                    "catalog_id": cid,
                    "role": ROLES[cid],
                    "n": int(len(s)),
                    "n_ok": int(ok.sum()),
                    "chi2_median": float(pd.to_numeric(s["psf_chi2"], errors="coerce").median()),
                    "flux_median": float(pd.to_numeric(s["psf_flux"], errors="coerce").median()),
                    "x_all_integer": bool(
                        np.allclose(
                            pd.to_numeric(s["x"], errors="coerce").to_numpy(),
                            np.round(pd.to_numeric(s["x"], errors="coerce").to_numpy()),
                        )
                    )
                    if "x" in s.columns
                    else False,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fitok_census.csv", index=False)
    return out, frozen, live


def _paste_star(frame: np.ndarray, model: ImagePSF, flux: float, x: float, y: float, stamp: int = 25) -> None:
    h, w = frame.shape
    half = stamp // 2
    xi = int(round(x))
    yi = int(round(y))
    x1 = xi - half
    y1 = yi - half
    x2 = x1 + stamp
    y2 = y1 + stamp
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return
    pred = _pp._psf_model_prediction_cutout(model, (stamp, stamp), flux, x - x1, y - y1)
    frame[y1:y2, x1:x2] += pred


def _synth_and_fit(
    *,
    model: ImagePSF,
    epsf_path: Path,
    hdr: fits.Header,
    flux: float,
    sky: float,
    gain: float,
    rn: float,
    noise: bool,
    fix_pos: bool,
    n_real: int,
    rng: np.random.Generator,
) -> list[dict]:
    global _FIX_POS
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
            _paste_star(frame, model, flux, x_true, y_true)
            cid = f"p{ix:02d}r{iy:02d}"
            pos_rows.append({"x": float(x_int), "y": float(y_int), "catalog_id": cid, "name": cid})
            truth.append((cid, dx, dy, x_true, y_true, x_int, y_int))
    if noise:
        e_mean = np.maximum(frame * gain, 0.0)
        electrons = rng.poisson(e_mean).astype(np.float64) + rng.normal(0.0, rn, size=frame.shape)
        frame = electrons / max(gain, 1e-6)
    stars = pd.DataFrame(pos_rows)
    _FIX_POS = bool(fix_pos)
    try:
        out = _pp.psf_photometry_stars(
            frame,
            hdr,
            stars,
            epsf_path,
            cutout_size=17,
            ref_fluxes=np.full(len(stars), float(flux)),
            apply_aperture_correction=False,
            psf_ac_policy="p4_none",
            use_iterative=True,
        )
    finally:
        _FIX_POS = False
    rec = {str(r["catalog_id"]): r for _, r in out.iterrows()}
    rows = []
    for cid, dx, dy, x_true, y_true, x_int, y_int in truth:
        r = rec.get(cid)
        if r is None:
            rec_flux = float("nan")
            chi2 = float("nan")
            fit_ok = False
        else:
            rec_flux = float(pd.to_numeric(r.get("psf_flux"), errors="coerce"))
            chi2 = float(pd.to_numeric(r.get("psf_chi2"), errors="coerce"))
            fit_ok = bool(_is_true(r.get("psf_fit_ok")))
        bias = (
            -2.5 * math.log10(rec_flux / flux) * 1000.0
            if (math.isfinite(rec_flux) and rec_flux > 0 and flux > 0)
            else float("nan")
        )
        rows.append(
            {
                "dx": dx,
                "dy": dy,
                "flux_inj": flux,
                "sky": sky,
                "noise": noise,
                "fix_pos": fix_pos,
                "n_real_cell": n_real,
                "catalog_id": cid,
                "flux_rec": rec_flux,
                "bias_mmag": bias,
                "chi2": chi2,
                "fit_ok": fit_ok,
            }
        )
    return rows


def part_a(meta_f: dict, frozen: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    flux_map = {
        "check": float(frozen.loc[frozen["catalog_id"] == CHECK_CID, "psf_flux"].median()),
        "target": float(frozen.loc[frozen["catalog_id"] == TARGET_CID, "psf_flux"].median()),
        "ens3": float(frozen.loc[frozen["catalog_id"] == ENS_IDS[2], "psf_flux"].median()),
    }
    qc = pd.read_csv(QC_PATH, comment="#")
    stems = set(
        pd.read_csv(PROC_FROZEN, comment="#")["fits"].astype(str).str.replace(".fits", "", regex=False)
    )
    qc["stem"] = qc["src"].astype(str).str.extract(r"(BO_CVn_Light_\d+)")
    sub = qc[qc["stem"].isin(stems)]
    sky_lo = float(sub["bg_median"].min())
    sky_hi = float(sub["bg_median"].max())
    hdr = fits.getheader(LIVE_LIGHTS / "BO_CVn_Light_001.fits")
    gain, rn = _pp._psf_resolve_gain_read_noise(hdr)
    print(f"[core] Part A gain={gain} rn={rn} sky={sky_lo}/{sky_hi} flux={flux_map}")
    epsf = np.asarray(fits.getdata(EPSF_FITS), dtype=np.float64)
    model = ImagePSF(epsf, oversampling=int(meta_f["oversampling"]))
    rng = np.random.default_rng(51614)
    raw = []
    for flux_name, flux in flux_map.items():
        for sky in (sky_lo, sky_hi):
            for noise in (False, True):
                n_use = 1 if not noise else N_REAL
                print(f"[core] A {flux_name} sky={sky:.1f} noise={noise} n={n_use}")
                rows = _synth_and_fit(
                    model=model,
                    epsf_path=EPSF_FITS,
                    hdr=hdr,
                    flux=flux,
                    sky=sky,
                    gain=gain,
                    rn=rn,
                    noise=noise,
                    fix_pos=False,
                    n_real=n_use,
                    rng=rng,
                )
                for r in rows:
                    r["flux_name"] = flux_name
                raw.extend(rows)
    # fixed-position, target flux, both skies, noise on + off
    for sky in (sky_lo, sky_hi):
        for noise in (False, True):
            n_use = 1 if not noise else N_REAL
            print(f"[core] A fixed target sky={sky:.1f} noise={noise} n={n_use}")
            rows = _synth_and_fit(
                model=model,
                epsf_path=EPSF_FITS,
                hdr=hdr,
                flux=flux_map["target"],
                sky=sky,
                gain=gain,
                rn=rn,
                noise=noise,
                fix_pos=True,
                n_real=n_use,
                rng=rng,
            )
            for r in rows:
                r["flux_name"] = "target"
            raw.extend(rows)
    raw_df = pd.DataFrame(raw)
    cells = (
        raw_df.groupby(["flux_name", "sky", "noise", "fix_pos", "dx", "dy"], as_index=False)
        .agg(
            n=("bias_mmag", lambda s: int(np.isfinite(s).sum())),
            bias_median_mmag=("bias_mmag", "median"),
            bias_std_mmag=("bias_mmag", "std"),
        )
    )
    cells.to_csv(OUT / "injection_bias.csv", index=False)
    real = cells[(~cells["fix_pos"]) & (cells["noise"])]
    worst = float(real["bias_median_mmag"].abs().max()) if len(real) else float("nan")
    dither = {}
    for (fn, sky), g in real.groupby(["flux_name", "sky"]):
        vals = g["bias_median_mmag"].to_numpy(dtype=np.float64)
        dither[f"{fn}|{sky:.3f}"] = float(np.sqrt(np.mean(vals * vals))) if vals.size else float("nan")
    headline = {
        "n_realizations_noise_on": N_REAL,
        "n_realizations_noise_off": 1,
        "phases": list(PHASES),
        "n_phase_cells": len(PHASE_CELLS),
        "flux_map": flux_map,
        "sky_lo": sky_lo,
        "sky_hi": sky_hi,
        "gain": float(gain),
        "rn": float(rn),
        "worst_cell_abs_bias_mmag_noise_on_free": worst,
        "dither_rms_mmag_by_flux_sky": dither,
        "n_cells_noise_on_free": int(len(real)),
    }
    return cells, headline


def _write_temp_epsf(arr: np.ndarray, meta_src: dict, tmp: Path, name: str) -> Path:
    fits.writeto(tmp / f"{name}.fits", np.asarray(arr, dtype=np.float32), overwrite=True)
    meta = {
        "fwhm_px": float(meta_src["fwhm_px_meta"]),
        "cutout_size": int(meta_src["cutout_size"]),
        "oversampling": int(meta_src["oversampling"]),
        "spatial_order": 0,
        "epsf_sum_native": 1.0,
    }
    (tmp / f"{name}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # production looks for masterstar_epsf_meta.json next to the FITS basename parent
    # load path: ep.parent / _MASTERSTAR_EPSF_META_NAME
    return tmp / f"{name}.fits"


def _ensure_meta_name(fits_path: Path, meta_src: dict) -> None:
    dest = fits_path.parent / "masterstar_epsf_meta.json"
    dest.write_text(
        json.dumps(
            {
                "fwhm_px": float(meta_src["fwhm_px_meta"]),
                "cutout_size": int(meta_src["cutout_size"]),
                "oversampling": int(meta_src["oversampling"]),
                "spatial_order": 0,
                "epsf_sum_native": 1.0,
            }
        ),
        encoding="utf-8",
    )


def part_b(meta_f: dict, live: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    matches = pd.read_csv(A2_COMPARE / "match_rows_deg2.csv", dtype={"catalog_id": str})
    matches["catalog_id"] = matches["catalog_id"].astype(str).str.strip()
    matches = matches[matches["catalog_id"].isin(STARS)].copy()
    stems = sorted(live["source_file"].str.replace("proc_", "", regex=False).str.replace(".csv", "", regex=False).unique())
    # Rule 0.1: snapshot Light_001 hash != live; frozen proc is live-derived.
    live_h = _sha256_file(LIVE_LIGHTS / "BO_CVn_Light_001.fits")[:12]
    snap_h = _sha256_file(SNAP_LIGHTS / "BO_CVn_Light_001.fits")[:12]
    print(f"[core] B lights live={live_h} snap={snap_h} (B1 uses live)")
    b1 = {cid: [] for cid in STARS}
    b2 = {cid: [] for cid in STARS}
    frozen_f = {cid: [] for cid in STARS}
    psfex_f = {cid: [] for cid in STARS}
    work = A2_OUT / "work"
    tmp = Path(tempfile.mkdtemp(prefix="epsf_core_b2_"))
    try:
        for i, stem in enumerate(stems):
            fits_path = LIVE_LIGHTS / f"{stem}.fits"
            data = np.asarray(fits.getdata(fits_path), dtype=np.float64)
            hdr = fits.getheader(fits_path)
            proc = live[live["source_file"] == f"proc_{stem}.csv"]
            pos = []
            refs = []
            for cid in STARS:
                hit = proc[proc["catalog_id"] == cid]
                if hit.empty:
                    pos.append({"x": 80.0, "y": 80.0, "catalog_id": cid, "name": cid})
                    refs.append(float("nan"))
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
            stars = pd.DataFrame(pos)
            data32 = np.asarray(data, dtype=np.float32)
            gain_s, rn_s = _pp._psf_resolve_gain_read_noise(hdr)
            err_map = np.sqrt(np.abs(data32) / max(gain_s, 1e-6) + (rn_s / max(gain_s, 1e-6)) ** 2).astype(
                np.float32
            )
            out1 = _pp.psf_photometry_stars(
                data32,
                hdr,
                stars,
                EPSF_FITS,
                error=err_map,
                ref_fluxes=np.asarray(refs, dtype=np.float64),
                apply_aperture_correction=False,
                psf_ac_policy="p4_none",
            )
            rec1 = {str(r["catalog_id"]): r for _, r in out1.iterrows()}
            psf_path = work / stem / f"{stem}.psf"
            model = PsfexModel(psf_path) if psf_path.is_file() else None
            for cid in STARS:
                r1 = rec1.get(cid)
                b1[cid].append(
                    float(pd.to_numeric(r1.get("psf_flux"), errors="coerce"))
                    if r1 is not None
                    else float("nan")
                )
                ph = proc[proc["catalog_id"] == cid]
                frozen_f[cid].append(
                    float(pd.to_numeric(ph.iloc[0]["psf_flux"], errors="coerce")) if not ph.empty else float("nan")
                )
                mh = matches[(matches["stem"] == stem) & (matches["catalog_id"] == cid)]
                if mh.empty or not bool(mh.iloc[0]["matched"]) or bool(mh.iloc[0].get("flux_nonfinite", False)):
                    psfex_f[cid].append(float("nan"))
                else:
                    psfex_f[cid].append(float(mh.iloc[0]["FLUX_PSF"]))
            if i % 10 == 0:
                print(f"[core] B1 frame {i+1}/{len(stems)} {stem}")

        b1_m = _flux_to_inst_mag(np.concatenate([np.asarray(b1[c]) for c in STARS]))
        fr_m = _flux_to_inst_mag(np.concatenate([np.asarray(frozen_f[c]) for c in STARS]))
        b1_vs_frozen = rms_after_median(b1_m, fr_m) * 1000.0
        n_b1 = int(np.isfinite(b1_m - fr_m).sum())
        print(f"[core] B1 vs frozen inst-mag RMS {b1_vs_frozen:.4f} mmag (n={n_b1})")
        b1_ok = bool(math.isfinite(b1_vs_frozen) and b1_vs_frozen <= B1_STOP_MMAG)
        if not b1_ok:
            print(
                f"[core] STOP B1: RMS={b1_vs_frozen} mmag > {B1_STOP_MMAG}; "
                "B2 not run; Part B/C model-swap readings void"
            )
            for cid in STARS:
                b2[cid] = [float("nan")] * len(stems)
        else:
            for i, stem in enumerate(stems):
                fits_path = LIVE_LIGHTS / f"{stem}.fits"
                data32 = np.asarray(fits.getdata(fits_path), dtype=np.float32)
                hdr = fits.getheader(fits_path)
                proc = live[live["source_file"] == f"proc_{stem}.csv"]
                pos = []
                refs = []
                for cid in STARS:
                    hit = proc[proc["catalog_id"] == cid]
                    if hit.empty:
                        pos.append({"x": 80.0, "y": 80.0, "catalog_id": cid, "name": cid})
                        refs.append(float("nan"))
                    else:
                        pos.append(
                            {
                                "x": float(hit.iloc[0]["x"]),
                                "y": float(hit.iloc[0]["y"]),
                                "catalog_id": cid,
                                "name": cid,
                            }
                        )
                        refs.append(float(pd.to_numeric(hit.iloc[0].get("dao_flux"), errors="coerce")))
                stars = pd.DataFrame(pos)
                psf_path = work / stem / f"{stem}.psf"
                model = PsfexModel(psf_path) if psf_path.is_file() else None
                for cid in STARS:
                    mh = matches[(matches["stem"] == stem) & (matches["catalog_id"] == cid)]
                    if (
                        model is None
                        or mh.empty
                        or not bool(mh.iloc[0]["matched"])
                        or bool(mh.iloc[0].get("flux_nonfinite", False))
                    ):
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
                    _ensure_meta_name(tfit, meta_f)
                    one = stars[stars["catalog_id"] == cid].copy()
                    ref_one = np.asarray([refs[STARS.index(cid)]], dtype=np.float64)
                    out2 = _pp.psf_photometry_stars(
                        data32,
                        hdr,
                        one,
                        tfit,
                        ref_fluxes=ref_one,
                        apply_aperture_correction=False,
                        psf_ac_policy="p4_none",
                    )
                    if out2.empty:
                        b2[cid].append(float("nan"))
                    else:
                        b2[cid].append(float(pd.to_numeric(out2.iloc[0]["psf_flux"], errors="coerce")))
                if i % 10 == 0:
                    print(f"[core] B2 frame {i+1}/{len(stems)} {stem}")
    finally:
        pass

    def _series(d: dict[str, list], cid: str) -> np.ndarray:
        return np.asarray(d[cid], dtype=np.float64)

    d_b1_t = rebuild_delta(_series(b1, TARGET_CID), {c: _series(b1, c) for c in ENS_IDS})
    d_b2_t = rebuild_delta(_series(b2, TARGET_CID), {c: _series(b2, c) for c in ENS_IDS})
    d_px_t = rebuild_delta(_series(psfex_f, TARGET_CID), {c: _series(psfex_f, c) for c in ENS_IDS})
    d_b1_c = rebuild_delta(_series(b1, CHECK_CID), {c: _series(b1, c) for c in ENS_IDS})
    d_b2_c = rebuild_delta(_series(b2, CHECK_CID), {c: _series(b2, c) for c in ENS_IDS})
    d_px_c = rebuild_delta(_series(psfex_f, CHECK_CID), {c: _series(psfex_f, c) for c in ENS_IDS})

    rows = []
    for i, stem in enumerate(stems):
        rows.append(
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
    lc = pd.DataFrame(rows)
    lc.to_csv(OUT / "modelswap_lc.csv", index=False)
    stats_b = {
        "n_epochs": int(len(stems)),
        "lights_source": "live draft_000516 aligned (frozen proc source); snapshot Light_001 hash differs",
        "live_light001_sha12": live_h,
        "snap_light001_sha12": snap_h,
        "b1_vs_frozen_instmag_rms_mmag": b1_vs_frozen,
        "b1_reproduced": bool(b1_ok),
        "b2_ran": bool(b1_ok),
        "b_readings_void": (not bool(b1_ok)),
        "b1_vs_frozen_population": "6 M2 stars x 134 live frames; instrumental mag after median",
        "rms_b1_vs_psfex_target_mmag": rms_after_median(d_b1_t, d_px_t) * 1000.0,
        "rms_b2_vs_psfex_target_mmag": rms_after_median(d_b2_t, d_px_t) * 1000.0,
        "rms_b1_vs_b2_target_mmag": rms_after_median(d_b1_t, d_b2_t) * 1000.0,
        "rms_b1_vs_psfex_check_mmag": rms_after_median(d_b1_c, d_px_c) * 1000.0,
        "rms_b2_vs_psfex_check_mmag": rms_after_median(d_b2_c, d_px_c) * 1000.0,
        "rms_b1_vs_b2_check_mmag": rms_after_median(d_b1_c, d_b2_c) * 1000.0,
        "psfex_wrap": "reconstruct at XPSF/YPSF; bilinear to 35x35 dest_scale=0.5 (osamp=2); unit native sum (sum=osamp^2); ImagePSF via temp FITS + meta fwhm_px so fit_shape stays 9x9",
        "population_lc": "134 identical-ensemble epochs; pinned 4-star AIJ flux-sum; median-removed RMS",
    }
    return lc, stats_b


def part_c(frozen: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    m2t = pd.read_csv(A2_COMPARE / "m2_epochs_target.csv")
    m2c = pd.read_csv(A2_COMPARE / "m2_epochs_check.csv")
    matches = pd.read_csv(A2_COMPARE / "match_rows_deg2.csv", dtype={"catalog_id": str})
    matches["catalog_id"] = matches["catalog_id"].astype(str).str.strip()
    rows = []
    for cid, m2, role in (
        (TARGET_CID, m2t, "target"),
        (CHECK_CID, m2c, "check"),
    ):
        sub = matches[(matches["catalog_id"] == cid) & (matches["matched"])]
        merged = m2.merge(sub[["stem", "XPSF_IMAGE", "YPSF_IMAGE"]], on="stem", how="inner")
        fx = np.asarray(merged["XPSF_IMAGE"], dtype=np.float64)
        fy = np.asarray(merged["YPSF_IMAGE"], dtype=np.float64)
        frac_x = fx - np.floor(fx)
        frac_y = fy - np.floor(fy)
        r_phase = np.hypot(frac_x - 0.5, frac_y - 0.5)
        resid = pd.to_numeric(merged["resid_after_median"], errors="coerce").to_numpy() * 1000.0
        chi = []
        for stem in merged["stem"].astype(str):
            hit = frozen[
                (frozen["catalog_id"] == cid) & (frozen["source_file"] == f"proc_{stem}.csv")
            ]
            chi.append(
                float(pd.to_numeric(hit.iloc[0]["psf_chi2"], errors="coerce")) if not hit.empty else float("nan")
            )
        chi_a = np.asarray(chi, dtype=np.float64)
        for i in range(len(merged)):
            rows.append(
                {
                    "stem": str(merged.iloc[i]["stem"]),
                    "catalog_id": cid,
                    "role": role,
                    "frac_x": float(frac_x[i]),
                    "frac_y": float(frac_y[i]),
                    "r_phase": float(r_phase[i]),
                    "resid_mmag": float(resid[i]),
                    "psf_chi2": float(chi_a[i]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phase_corr.csv", index=False)
    corr = {}
    for role in ("target", "check"):
        s = df[df["role"] == role]
        c1 = stats.spearmanr(s["r_phase"], s["resid_mmag"])
        c2 = stats.spearmanr(s["psf_chi2"], s["resid_mmag"])
        corr[role] = {
            "n": int(len(s)),
            "spearman_rphase_vs_resid": float(c1.statistic),
            "p_rphase": float(c1.pvalue),
            "spearman_chi2_vs_resid": float(c2.statistic),
            "p_chi2": float(c2.pvalue),
        }
    return df, corr


def readings(f5: pd.DataFrame, a_head: dict, b_head: dict, consumers: dict) -> list[str]:
    fired = []
    chk = f5[(f5["source"] == "live") & (f5["role"] == "check")].iloc[0]
    silent = bool(consumers.get("not_ok_consumed_in_psf_lc"))
    if int(chk["n_ok"]) == 0 and int(chk["n"]) == 134 and silent:
        fired.append(
            "R-C0: F5 confirmed AND no PSF-LC consumer filters on psf_fit_ok "
            "(fit_ok_for_zp admits finite flux+chi2); flag-consumed-silently."
        )
    worst = float(a_head.get("worst_cell_abs_bias_mmag_noise_on_free", float("nan")))
    real_cells = a_head.get("n_cells_noise_on_free", 0)
    a_clean = math.isfinite(worst) and worst < 1.0
    a_defect = math.isfinite(worst) and worst >= 3.0
    if a_defect:
        fired.append(
            f"R-C1: Part A worst-cell |bias|={worst:.3f} mmag (>=3) on {real_cells} "
            "noise-on free-position cells -> MACHINERY defect."
        )
    b2_t = float(b_head.get("rms_b2_vs_psfex_target_mmag", float("nan")))
    b2_c = float(b_head.get("rms_b2_vs_psfex_check_mmag", float("nan")))
    b1_t = float(b_head.get("rms_b1_vs_psfex_target_mmag", float("nan")))
    b1_c = float(b_head.get("rms_b1_vs_psfex_check_mmag", float("nan")))
    b2_ok = (math.isfinite(b2_t) and b2_t <= 3.0) and (math.isfinite(b2_c) and b2_c <= 3.0)
    b1_bad = (math.isfinite(b1_t) and b1_t >= 10.0) or (math.isfinite(b1_c) and b1_c >= 10.0)
    if b_head.get("b_readings_void"):
        b2_ok = False
    if a_clean and b2_ok and b1_bad:
        fired.append(
            "R-C2: Part A clean (<1 mmag) AND B2-vs-PSFEx <= 3 mmag while "
            "B1-vs-PSFEx stays >= 10 mmag -> MODEL-DATA coupling."
        )
    core_fired = any(x.startswith("R-C1") or x.startswith("R-C2") for x in fired)
    if not core_fired:
        fired.append(
            "R-C3: neither R-C1 nor R-C2 fires cleanly; full decomposition "
            "reported, no attribution claim."
        )
    return fired


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    facts = confirm_f1_f4()
    print("[core] F1-F4", facts)
    census, frozen, live = fitok_census()
    print("[core] F5\n", census.to_string(index=False))
    consumers = {
        "a2_compare_filters_fit_ok": False,
        "a2_compare_cite": "a2_compare.py:508 uses psf_flux only",
        "psf_internal_lc_mode_516": facts["sidecar_zp_membership_effective"],
        "psf_internal_lc_cite": (
            "psf_internal_lc.py:124-134 psf_fit_ok_for_zp_mask = fit_ok "
            "OR (finite flux>0 AND finite chi2); :490-509 applies that mask "
            "before ensemble_normalize. 516 sidecar fit_ok_for_zp, rig validated."
        ),
        "photometry_lightcurve_adaptive_filters": True,
        "photometry_lightcurve_cite": (
            "photometry_lightcurve.py:2393 psf_usable requires psf_fit_ok; "
            "this is the aperture-vs-psf science-method picker, not the internal PSF LC."
        ),
        "not_ok_consumed_in_psf_lc": facts["sidecar_zp_membership_effective"] == "fit_ok_for_zp",
    }
    a_cells, a_head = part_a(facts, frozen)
    print("[core] A headline", a_head)
    a_only = "--a-only" in sys.argv
    if a_only and (OUT / "summary.json").is_file():
        prev = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        b_head = prev.get("part_b", {})
        c_head = prev.get("part_c", {})
        print("[core] --a-only reused prior B/C")
    else:
        b_lc, b_head = part_b(facts, live)
        print("[core] B headline", b_head)
        _c_df, c_head = part_c(frozen)
        print("[core] C", c_head)
    fired = readings(census, a_head, b_head, consumers)
    print("[core] readings", fired)
    g4 = g4_live_516()
    summary = {
        "f1_f4": facts,
        "f5": census.to_dict(orient="records"),
        "consumers": consumers,
        "part_a": a_head,
        "part_b": b_head,
        "part_c": c_head,
        "readings": fired,
        "g4": g4,
        "n_a_cells": int(len(a_cells)),
        "n_b_epochs": int(b_head.get("n_epochs") or 0),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[core] g4", g4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
