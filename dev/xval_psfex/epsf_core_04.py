# -*- coding: ascii -*-
"""EPSF-CORE-04: reference arbitration, osamp phase probe, machinery knobs.

Dev-only. src_py must not import this module. Live 516/Archive read-only.
Rebuilt ePSF FITS live under the session dir only.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src_py") not in sys.path:
    sys.path.insert(0, str(REPO / "src_py"))
if str(REPO / "dev" / "xval_psfex") not in sys.path:
    sys.path.insert(0, str(REPO / "dev" / "xval_psfex"))

from epsf_core_03 import (  # noqa: E402
    A2_COMPARE,
    A2_OUT,
    CHECK_CID,
    ENS_IDS,
    EPSF_FITS,
    FRAME076,
    LIVE_ALN,
    LIVE_PS,
    PHASE_CELLS,
    PHASES,
    PROC_FROZEN,
    QC_PATH,
    STARS,
    TARGET_CID,
    _err_map,
    _m2_pos_refs,
    _paste_stamp,
    fit_kwargs_core02,
    g4_live_516,
    load_live_proc,
    load_stems,
    rebuild_delta,
    render_psfex_native,
)
from epsf_shape_01 import PsfexModel  # noqa: E402

import psf_photometry as _pp  # noqa: E402

OUT = REPO / "dev" / "results" / "context" / "session_20260915_epsf_core_04"
CORE03 = REPO / "dev" / "results" / "context" / "session_20260914_epsf_core_03"
MS_CSV = LIVE_PS / "masterstars_full_match.csv"
MS_FITS = LIVE_PS / "MASTERSTAR.fits"
DB_PATH = REPO / "vyvar.sqlite3"
G_PT = 0.637067
RN_PT = 15.2
FLOOR_MMAG = 1.396
A2_TARGET = 10.48
A2_CHECK = 21.41
LIVE_WIN_T = {"x": (0.587, 0.682), "y": (0.402, 0.504)}
LIVE_WIN_C = {"x": (0.539, 0.669), "y": (0.829, 0.935)}
WINDOW_CELLS = [(dx, dy) for dx in (0.5, 0.625, 0.75) for dy in (0.375, 0.5)]
N_WIN_ON = 30


def rms_med(d: np.ndarray) -> float:
    """RMS_med: sqrt(mean((d - median(d))^2)) over finite samples."""
    x = np.asarray(d, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float("nan")
    return float(np.sqrt(np.mean((x - float(np.median(x))) ** 2)))


def rms_med_diff(a: np.ndarray, b: np.ndarray) -> float:
    return rms_med(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))


def mag_from_flux(flux: np.ndarray) -> np.ndarray:
    f = np.asarray(flux, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def _stem_of(source_file: str) -> str:
    s = str(source_file).replace("proc_", "").replace(".csv", "")
    return s


def part_a(stems: list[str], live: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Zero new photometry: dao_flux aperture vs frozen psf_flux vs PSFEx catalog."""
    matches = pd.read_csv(A2_COMPARE / "match_rows_deg2.csv", dtype={"catalog_id": str})
    matches["catalog_id"] = matches["catalog_id"].astype(str).str.strip()
    frozen = pd.read_csv(PROC_FROZEN, comment="#", dtype={"catalog_id": str})
    frozen["catalog_id"] = frozen["catalog_id"].astype(str).str.strip()
    live = live.copy()
    live["catalog_id"] = live["catalog_id"].astype(str).str.strip()
    live["_stem"] = live["source_file"].map(_stem_of)

    lost: list[dict] = []
    ap: dict[str, list] = {c: [] for c in STARS}
    psf: dict[str, list] = {c: [] for c in STARS}
    px: dict[str, list] = {c: [] for c in STARS}
    ap_r: list[float] = []
    for stem in stems:
        proc = live[live["_stem"] == stem]
        fr = frozen[frozen["source_file"] == f"proc_{stem}.csv"]
        for cid in STARS:
            hit = proc[proc["catalog_id"] == cid]
            if hit.empty:
                lost.append({"stem": stem, "catalog_id": cid, "missing": "dao_flux_row"})
                ap[cid].append(float("nan"))
            else:
                ap[cid].append(float(pd.to_numeric(hit.iloc[0]["dao_flux"], errors="coerce")))
                ap_r.append(float(pd.to_numeric(hit.iloc[0].get("aperture_r_px"), errors="coerce")))
            fh = fr[fr["catalog_id"] == cid]
            psf[cid].append(
                float(pd.to_numeric(fh.iloc[0]["psf_flux"], errors="coerce")) if not fh.empty else float("nan")
            )
            mh = matches[(matches["stem"] == stem) & (matches["catalog_id"] == cid)]
            if mh.empty or not bool(mh.iloc[0]["matched"]) or bool(mh.iloc[0].get("flux_nonfinite", False)):
                px[cid].append(float("nan"))
            else:
                px[cid].append(float(mh.iloc[0]["FLUX_PSF"]))

    def _arr(d: dict[str, list], cid: str) -> np.ndarray:
        return np.asarray(d[cid], dtype=np.float64)

    d_ap_t = rebuild_delta(_arr(ap, TARGET_CID), {c: _arr(ap, c) for c in ENS_IDS})
    d_ap_c = rebuild_delta(_arr(ap, CHECK_CID), {c: _arr(ap, c) for c in ENS_IDS})
    d_psf_t = rebuild_delta(_arr(psf, TARGET_CID), {c: _arr(psf, c) for c in ENS_IDS})
    d_psf_c = rebuild_delta(_arr(psf, CHECK_CID), {c: _arr(psf, c) for c in ENS_IDS})
    d_px_t = rebuild_delta(_arr(px, TARGET_CID), {c: _arr(px, c) for c in ENS_IDS})
    d_px_c = rebuild_delta(_arr(px, CHECK_CID), {c: _arr(px, c) for c in ENS_IDS})

    rows = []
    for i, stem in enumerate(stems):
        rows.append(
            {
                "stem": stem,
                "ap_target": d_ap_t[i],
                "ap_check": d_ap_c[i],
                "psf_target": d_psf_t[i],
                "psf_check": d_psf_c[i],
                "psfex_target": d_px_t[i],
                "psfex_check": d_px_c[i],
            }
        )
    lc = pd.DataFrame(rows)
    lc.to_csv(OUT / "arbitration_lc.csv", index=False)

    n_ok = int(np.isfinite(d_ap_t).sum())
    r_ap = float(np.nanmedian(np.asarray(ap_r, dtype=np.float64))) if ap_r else float("nan")
    summ_rows = [
        {
            "id": "A1",
            "comparison": "PSFEx cat vs aperture",
            "target_rms_med_mmag": rms_med_diff(d_px_t, d_ap_t) * 1000.0,
            "check_rms_med_mmag": rms_med_diff(d_px_c, d_ap_c) * 1000.0,
            "n": n_ok,
        },
        {
            "id": "A2",
            "comparison": "VYVAR psf_flux vs aperture",
            "target_rms_med_mmag": rms_med_diff(d_psf_t, d_ap_t) * 1000.0,
            "check_rms_med_mmag": rms_med_diff(d_psf_c, d_ap_c) * 1000.0,
            "n": n_ok,
        },
        {
            "id": "A3",
            "comparison": "VYVAR psf_flux vs PSFEx cat (restate A2-COMPARE)",
            "target_rms_med_mmag": rms_med_diff(d_psf_t, d_px_t) * 1000.0,
            "check_rms_med_mmag": rms_med_diff(d_psf_c, d_px_c) * 1000.0,
            "n": n_ok,
            "a2_compare_target_mmag": A2_TARGET,
            "a2_compare_check_mmag": A2_CHECK,
        },
        {
            "id": "A4",
            "comparison": "aperture check-star RMS_med (arbiter noise floor)",
            "target_rms_med_mmag": float("nan"),
            "check_rms_med_mmag": rms_med(d_ap_c) * 1000.0,
            "n": int(np.isfinite(d_ap_c).sum()),
        },
    ]
    sdf = pd.DataFrame(summ_rows)
    sdf.to_csv(OUT / "arbitration_summary.csv", index=False)
    head = {
        "aperture_product": (
            "live proc CSV column dao_flux "
            f"(Archive/.../proc_*.csv); aperture_r_px median={r_ap:.4g} "
            "(snr_table); fwhm_px_for_aperture=3.3014 per-draft gaussian override; "
            "annulus APERTURE-01d 2.7/5.2 FWHM. Rebuilt pinned 4-star AIJ flux-sum "
            "diff LC (same ensemble as A2-COMPARE). Production LC mag_calib uses a "
            "larger comp pool and is NOT used here."
        ),
        "n_epochs": len(stems),
        "n_finite_ap_target": n_ok,
        "epochs_lost": lost,
        "population": "134 identical-ensemble epochs; RMS_med in mmag",
        "rows": summ_rows,
    }
    return lc, head


def _bias_grid(cell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    off = cell[(cell["noise"] == False) & (cell["n"] >= 1)].copy()  # noqa: E712
    xs = np.array(sorted(off["dx"].unique()), dtype=np.float64)
    ys = np.array(sorted(off["dy"].unique()), dtype=np.float64)
    z = np.full((ys.size, xs.size), np.nan, dtype=np.float64)
    for _, r in off.iterrows():
        ix = int(np.argmin(np.abs(xs - float(r["dx"]))))
        iy = int(np.argmin(np.abs(ys - float(r["dy"]))))
        z[iy, ix] = float(r["bias_median_mmag"])
    return xs, ys, z


def live_window_stats(cell: pd.DataFrame, win: dict) -> dict:
    off = cell[(cell["noise"] == False) & (cell["n"] >= 1)].copy()  # noqa: E712
    nan_out = {
        "ptp_full_mmag": float("nan"),
        "slope_mmag_per_0p1px": float("nan"),
        "phase_rms_live_mmag": float("nan"),
        "ptp_live_mmag": float("nan"),
        "mean_bias_full_mmag": float("nan"),
        "mean_bias_live_mmag": float("nan"),
        "n_live_samples": 0,
        "n_finite_off": 0,
    }
    if off.empty:
        return nan_out
    xs, ys, z = _bias_grid(cell)
    if xs.size < 2 or ys.size < 2 or not np.isfinite(z).any():
        v = pd.to_numeric(off["bias_median_mmag"], errors="coerce")
        v = v[np.isfinite(v)]
        nan_out["n_finite_off"] = int(v.size)
        nan_out["mean_bias_full_mmag"] = float(np.mean(v)) if len(v) else float("nan")
        nan_out["ptp_full_mmag"] = float(v.max() - v.min()) if len(v) else float("nan")
        nan_out["note"] = "bias grid not interpolable (all-NaN or degenerate)"
        return nan_out
    interp = RegularGridInterpolator((ys, xs), z, bounds_error=False, fill_value=None)
    gx = np.linspace(win["x"][0], win["x"][1], 21)
    gy = np.linspace(win["y"][0], win["y"][1], 21)
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    samp = interp(np.column_stack([yy.ravel(), xx.ravel()]))
    samp = np.asarray(samp, dtype=np.float64)
    samp = samp[np.isfinite(samp)]
    ptp = float(samp.max() - samp.min()) if samp.size else float("nan")
    phase_rms = float(np.sqrt(np.var(samp))) if samp.size else float("nan")
    off = cell[(cell["noise"] == False)]  # noqa: E712
    mean_full = float(np.nanmean(pd.to_numeric(off["bias_median_mmag"], errors="coerce")))
    dy_tgt = min(PHASES, key=lambda d: abs(d - 0.5))
    s = off[np.isclose(off["dy"], dy_tgt)].sort_values("dx")
    a = s[np.isclose(s["dx"], 0.5)]
    b = s[np.isclose(s["dx"], 0.625)]
    slope = float("nan")
    if len(a) and len(b):
        slope = (float(b.iloc[0]["bias_median_mmag"]) - float(a.iloc[0]["bias_median_mmag"])) / 0.125 * 0.1
    v = pd.to_numeric(off["bias_median_mmag"], errors="coerce")
    v = v[np.isfinite(v)]
    ptp_full = float(v.max() - v.min()) if len(v) else float("nan")
    return {
        "ptp_full_mmag": ptp_full,
        "slope_mmag_per_0p1px": slope,
        "phase_rms_live_mmag": phase_rms,
        "ptp_live_mmag": ptp,
        "mean_bias_full_mmag": mean_full,
        "mean_bias_live_mmag": float(np.mean(samp)) if samp.size else float("nan"),
        "n_live_samples": int(samp.size),
    }


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
    return _pp.psf_photometry_stars(data32, hdr, stars, epsf_path, **call)


def rebuild_epsf(osamp: int) -> Path:
    """Production builder offline into the session dir. Never writes live masterstar_epsf.fits."""
    dest = OUT / f"epsf_osamp{osamp}"
    dest.mkdir(parents=True, exist_ok=True)
    existing = dest / "masterstar_epsf.fits"
    if existing.is_file():
        return existing
    tmp_db = Path(REPO / "tmp" / f"core04_osamp{osamp}.sqlite3")
    tmp_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DB_PATH, tmp_db)
    from database import VyvarDatabase

    db = VyvarDatabase(tmp_db)
    try:
        path = _pp.build_epsf_model(
            MS_FITS,
            MS_CSV,
            db,
            516,
            oversampling=int(osamp),
            sandbox_output_dir=dest,
        )
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            tmp_db.unlink(missing_ok=True)
            Path(str(tmp_db) + "-wal").unlink(missing_ok=True)
            Path(str(tmp_db) + "-shm").unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    return Path(path)


def _t1_stamps_and_hdr(flux: float):
    hdr = fits.getheader(LIVE_ALN / f"{FRAME076}.fits")
    matches = pd.read_csv(A2_COMPARE / "match_rows_deg2.csv", dtype={"catalog_id": str})
    matches["catalog_id"] = matches["catalog_id"].astype(str).str.strip()
    mh = matches[(matches["stem"] == FRAME076) & (matches["catalog_id"] == TARGET_CID)]
    x_s = float(mh.iloc[0]["XPSF_IMAGE"])
    y_s = float(mh.iloc[0]["YPSF_IMAGE"])
    model = PsfexModel(A2_OUT / "work" / FRAME076 / f"{FRAME076}.psf")
    rec = model.reconstruct(x_s, y_s)
    stamps = {(dx, dy): render_psfex_native(rec, model.psf_samp, dx, dy, flux) for dx, dy in PHASE_CELLS}
    return stamps, hdr, model.psf_samp


def _pack_and_fit(
    *,
    cells: list[tuple[float, float]],
    stamps: dict,
    hdr,
    flux: float,
    sky: float,
    gain: float,
    rn: float,
    noise: bool,
    n_real: int,
    fit_kw: dict,
    epsf_path: Path,
    rng: np.random.Generator,
) -> list[dict]:
    spacing = 40
    pad = 48
    ncols = len(cells)
    nrows = n_real
    width = pad * 2 + spacing * (ncols - 1) + 1
    height = pad * 2 + spacing * (nrows - 1) + 1
    frame = np.full((height, width), float(sky), dtype=np.float64)
    pos_rows = []
    truth = []
    for iy in range(nrows):
        for ix, (dx, dy) in enumerate(cells):
            x_int = pad + ix * spacing
            y_int = pad + iy * spacing
            if x_int % 2:
                x_int += 1
            if y_int % 2:
                y_int += 1
            _paste_stamp(frame, stamps[(dx, dy)], x_int, y_int)
            cid = f"p{ix:02d}r{iy:02d}"
            pos_rows.append({"x": float(x_int), "y": float(y_int), "catalog_id": cid, "name": cid})
            truth.append((cid, dx, dy))
    if noise:
        e_mean = np.maximum(frame * gain, 0.0)
        electrons = rng.poisson(e_mean).astype(np.float64) + rng.normal(0.0, rn, size=frame.shape)
        frame = electrons / max(gain, 1e-6)
    stars = pd.DataFrame(pos_rows)
    data32 = np.asarray(frame, dtype=np.float32)
    refs = np.full(len(stars), float(flux), dtype=np.float64)
    out = _fit(data32, hdr, stars, refs, epsf_path, fit_kw)
    rec = {str(r["catalog_id"]): r for _, r in out.iterrows()}
    rows = []
    for cid, dx, dy in truth:
        r = rec.get(cid)
        rec_flux = float(pd.to_numeric(r.get("psf_flux"), errors="coerce")) if r is not None else float("nan")
        bias = (
            -2.5 * math.log10(rec_flux / flux) * 1000.0
            if (math.isfinite(rec_flux) and rec_flux > 0 and flux > 0)
            else float("nan")
        )
        rows.append({"dx": dx, "dy": dy, "bias_mmag": bias, "flux_rec": rec_flux, "noise": noise})
    return rows


def _run_t1_phase(epsf_path: Path, stamps, hdr, flux: float, sky: float, gain: float, rn: float, fit_kw: dict) -> pd.DataFrame:
    rng = np.random.default_rng(20260915)
    rows = []
    rows.extend(
        _pack_and_fit(
            cells=list(PHASE_CELLS), stamps=stamps, hdr=hdr, flux=flux, sky=sky,
            gain=gain, rn=rn, noise=False, n_real=1, fit_kw=fit_kw, epsf_path=epsf_path, rng=rng,
        )
    )
    print("[core04] T1 noise-off done", epsf_path.parent.name)
    rows.extend(
        _pack_and_fit(
            cells=list(WINDOW_CELLS), stamps=stamps, hdr=hdr, flux=flux, sky=sky,
            gain=gain, rn=rn, noise=True, n_real=N_WIN_ON, fit_kw=fit_kw, epsf_path=epsf_path, rng=rng,
        )
    )
    print("[core04] T1 noise-on window done", epsf_path.parent.name)
    df = pd.DataFrame(rows)
    return df.groupby(["dx", "dy", "noise"], as_index=False).agg(
        n=("bias_mmag", "count"),
        bias_median_mmag=("bias_mmag", "median"),
        bias_std_mmag=("bias_mmag", "std"),
    )


def part_b() -> dict:
    frozen = pd.read_csv(PROC_FROZEN, comment="#", dtype={"catalog_id": str})
    frozen["catalog_id"] = frozen["catalog_id"].astype(str).str.strip()
    flux = float(
        pd.to_numeric(frozen.loc[frozen["catalog_id"] == TARGET_CID, "psf_flux"], errors="coerce").median()
    )
    qc = pd.read_csv(QC_PATH, comment="#")
    sky = float(pd.to_numeric(qc["bg_median"], errors="coerce").median())
    hdr0 = fits.getheader(LIVE_ALN / f"{FRAME076}.fits")
    gain, rn = _pp._psf_resolve_gain_read_noise(hdr0)
    fit_kw = fit_kwargs_core02()
    stamps, hdr, psf_samp = _t1_stamps_and_hdr(flux)
    print(f"[core04] T1 stamps PSF_SAMP={psf_samp} flux={flux:.4g} sky={sky:.3f}")

    t2 = pd.read_csv(CORE03 / "phase_bias_T1.csv")
    stats2 = live_window_stats(t2, LIVE_WIN_T)
    stats2.update({"osamp": 2, "source": "CORE-03 delivered phase_bias_T1.csv", "builder": "live 516 ePSF"})
    rows = [stats2]
    check_s = live_window_stats(t2, LIVE_WIN_C)
    check_s.update({"osamp": 2, "star": "check_window_on_T1_target_truth"})

    for osamp in (3, 4):
        t0 = time.perf_counter()
        try:
            ep = rebuild_epsf(osamp)
            print(f"[core04] rebuilt osamp={osamp} -> {ep} in {time.perf_counter()-t0:.1f}s")
        except Exception as exc:  # noqa: BLE001
            print(f"[core04] osamp={osamp} BUILD FAIL: {exc}")
            rows.append({"osamp": osamp, "error": str(exc)})
            continue
        g = _run_t1_phase(ep, stamps, hdr, flux, sky, gain, rn, fit_kw)
        g.to_csv(OUT / f"phase_bias_osamp{osamp}.csv", index=False)
        st = live_window_stats(g, LIVE_WIN_T)
        st.update(
            {
                "osamp": osamp,
                "source": f"phase_bias_osamp{osamp}.csv",
                "builder": "psf_photometry.build_epsf_model sandbox_output_dir",
                "elapsed_s": time.perf_counter() - t0,
            }
        )
        rows.append(st)
        print(f"[core04] osamp={osamp}", st)

    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT / "phase_window_summary.csv", index=False)
    return {
        "rows": rows,
        "truth": (
            "T1 PSFEx deg2 Light_076 at target; zoom 1/8 native; nd_shift; 8x8 block-sum; "
            "never ImagePSF. Fit via psf_photometry_stars, CORE-02 kwargs, integer-peak init."
        ),
        "builder_cite": "psf_photometry.build_epsf_model (psf_photometry.py:1357) via _epsf_build_imagepsf_from_stars (572)",
        "check_window_on_osamp2": check_s,
        "flux": flux,
        "sky": sky,
    }


@contextmanager
def apply_knob(name: str):
    orig: dict = {}
    try:
        if name in ("K1", "K5"):
            orig["gain"] = _pp._psf_resolve_gain_read_noise
            _pp._psf_resolve_gain_read_noise = lambda _hdr: (float(G_PT), float(RN_PT))
        if name in ("K2", "K5"):
            orig["fit"] = _pp._fit_shape_for_cutout
            _pp._fit_shape_for_cutout = lambda cutout_size, fwhm_px=None: (13, 13)
        if name == "K3":
            orig["err"] = _pp._psf_fit_error_cutout_full_ccd

            def _uniform(cut_shape, **_kw):
                return np.ones(cut_shape, dtype=np.float64)

            _pp._psf_fit_error_cutout_full_ccd = _uniform
        if name == "K4":
            orig["ann"] = _pp._psf_annulus_radii_px

            def _wide(fwhm_px, *, inner_fwhm=None, outer_fwhm=None):
                return orig["ann"](
                    fwhm_px,
                    inner_fwhm=4.0 if inner_fwhm is None else inner_fwhm,
                    outer_fwhm=8.0 if outer_fwhm is None else outer_fwhm,
                )

            _pp._psf_annulus_radii_px = _wide
        yield
    finally:
        if "gain" in orig:
            _pp._psf_resolve_gain_read_noise = orig["gain"]
        if "fit" in orig:
            _pp._fit_shape_for_cutout = orig["fit"]
        if "err" in orig:
            _pp._psf_fit_error_cutout_full_ccd = orig["err"]
        if "ann" in orig:
            _pp._psf_annulus_radii_px = orig["ann"]


def part_c(stems: list[str], live: pd.DataFrame, a_lc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    matches = pd.read_csv(A2_COMPARE / "match_rows_deg2.csv", dtype={"catalog_id": str})
    matches["catalog_id"] = matches["catalog_id"].astype(str).str.strip()
    knobs = ["baseline", "K1", "K2", "K3", "K4", "K5"]
    store: dict[str, dict[str, list]] = {k: {c: [] for c in STARS} for k in knobs}
    px = {c: [] for c in STARS}
    fit_kw = fit_kwargs_core02()
    t0 = time.perf_counter()
    for i, stem in enumerate(stems):
        fits_path = LIVE_ALN / f"{stem}.fits"
        data32 = np.asarray(fits.getdata(fits_path), dtype=np.float32)
        hdr = fits.getheader(fits_path)
        proc = live[live["source_file"] == f"proc_{stem}.csv"]
        stars, refs, _fr = _m2_pos_refs(proc)
        for cid in STARS:
            mh = matches[(matches["stem"] == stem) & (matches["catalog_id"] == cid)]
            if mh.empty or not bool(mh.iloc[0]["matched"]) or bool(mh.iloc[0].get("flux_nonfinite", False)):
                px[cid].append(float("nan"))
            else:
                px[cid].append(float(mh.iloc[0]["FLUX_PSF"]))
        for kn in knobs:
            with apply_knob(kn):
                out = _fit(data32, hdr, stars, refs, EPSF_FITS, fit_kw)
            rec = {str(r["catalog_id"]).strip(): r for _, r in out.iterrows()}
            for cid in STARS:
                r = rec.get(cid)
                store[kn][cid].append(
                    float(pd.to_numeric(r.get("psf_flux"), errors="coerce")) if r is not None else float("nan")
                )
        if i % 10 == 0:
            print(f"[core04] C knobs {i+1}/{len(stems)} {stem}")

    def _d(fluxes: dict[str, list], cid: str) -> np.ndarray:
        return rebuild_delta(np.asarray(fluxes[cid], dtype=np.float64), {c: np.asarray(fluxes[c], dtype=np.float64) for c in ENS_IDS})

    d_px_t = _d(px, TARGET_CID)
    d_px_c = _d(px, CHECK_CID)
    d_ap_t = a_lc["ap_target"].to_numpy(dtype=np.float64)
    d_ap_c = a_lc["ap_check"].to_numpy(dtype=np.float64)

    cites = {
        "K1": "patch _psf_resolve_gain_read_noise -> (0.637067, 15.2); model-based err map uses these (psf_photometry.py:2262, 3117)",
        "K2": "patch _fit_shape_for_cutout -> (13,13); production is ceil(2*fwhm_px+1) odd, meta fwhm 3.3014 -> 9x9 (psf_photometry.py:337-351, 2819)",
        "K3": "patch _psf_fit_error_cutout_full_ccd -> ones (uniform weights)",
        "K4": "patch _psf_annulus_radii_px inner/outer 4.0/8.0 FWHM (production residual annulus defaults 2.7/5.2 via AppConfig; psf_photometry.py:2065-2083, 2184)",
        "K5": "K1+K2 together",
        "baseline": "CORE-02 kwargs, live ePSF, no patch",
    }
    out_rows = []
    for kn in knobs:
        dt = _d(store[kn], TARGET_CID)
        dc = _d(store[kn], CHECK_CID)
        out_rows.append(
            {
                "knob": kn,
                "cite": cites[kn],
                "target_vs_psfex_mmag": rms_med_diff(dt, d_px_t) * 1000.0,
                "check_vs_psfex_mmag": rms_med_diff(dc, d_px_c) * 1000.0,
                "target_vs_aperture_mmag": rms_med_diff(dt, d_ap_t) * 1000.0,
                "check_vs_aperture_mmag": rms_med_diff(dc, d_ap_c) * 1000.0,
                "floor_mmag": FLOOR_MMAG,
                "n": int(np.isfinite(dt).sum()),
            }
        )
    kdf = pd.DataFrame(out_rows)
    kdf.to_csv(OUT / "knobs.csv", index=False)
    return kdf, {"rows": out_rows, "elapsed_s": time.perf_counter() - t0, "cites": cites}


def readings(a_head: dict, b_head: dict, c_head: dict) -> list[str]:
    fired = []
    arows = {r["id"]: r for r in a_head.get("rows") or []}
    a1t = float(arows.get("A1", {}).get("target_rms_med_mmag", float("nan")))
    a1c = float(arows.get("A1", {}).get("check_rms_med_mmag", float("nan")))
    if math.isfinite(a1t) and math.isfinite(a1c) and a1t <= 3.0 and a1c <= 3.0:
        fired.append("R-R1: A1 <= 3.0 mmag on BOTH stars; PSFEx is a valid reference.")
    elif (math.isfinite(a1t) and a1t >= 10.0) or (math.isfinite(a1c) and a1c >= 10.0):
        fired.append("R-R2: A1 >= 10.0 mmag on EITHER star; no PSF method reaches 3 mmag against the arbiter.")
    elif math.isfinite(a1t) and math.isfinite(a1c) and 3.0 < min(a1t, a1c) and max(a1t, a1c) < 10.0:
        fired.append("R-R3: 3.0 < A1 < 10.0; intermediate; report the triangle, no verdict.")
    elif math.isfinite(a1t) and math.isfinite(a1c) and (3.0 < a1t < 10.0 or 3.0 < a1c < 10.0) and not (
        (a1t >= 10.0 or a1c >= 10.0) or (a1t <= 3.0 and a1c <= 3.0)
    ):
        fired.append("R-R3: 3.0 < A1 < 10.0 (mixed stars); intermediate; report the triangle, no verdict.")

    os4 = None
    for r in b_head.get("rows") or []:
        if int(r.get("osamp") or 0) == 4 and "error" not in r:
            os4 = r
    if os4 is not None:
        pr = float(os4.get("phase_rms_live_mmag", float("nan")))
        sl = abs(float(os4.get("slope_mmag_per_0p1px", float("nan"))))
        if math.isfinite(pr) and math.isfinite(sl) and pr <= 1.0 and sl <= 2.0:
            fired.append("R-P1: osamp=4 live-window phase RMS <= 1.0 mmag AND slope <= 2.0 mmag/0.1px.")
        else:
            fired.append("R-P2: osamp=4 does not collapse the phase component; dithered build required.")
    else:
        fired.append("R-P2: osamp=4 build/probe unavailable; phase component not shown to be sampling-only.")

    brows = {r["knob"]: r for r in c_head.get("rows") or []}
    base = brows.get("baseline") or {}
    k1_hit = False
    for kn in ("K1", "K2", "K3", "K4", "K5"):
        row = brows.get(kn) or {}
        for star in ("check", "target"):
            b = float(base.get(f"{star}_vs_aperture_mmag", float("nan")))
            v = float(row.get(f"{star}_vs_aperture_mmag", float("nan")))
            if math.isfinite(b) and math.isfinite(v) and (b - v) >= 3.0:
                fired.append(f"R-K1: {kn} lowers B1 vs aperture by >= 3.0 mmag on the {star}.")
                k1_hit = True
    if not k1_hit and brows:
        fired.append("R-K0: no knob moves B1 vs aperture by >= 3.0 mmag.")
    return fired


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-only", action="store_true")
    ap.add_argument("--b-only", action="store_true")
    ap.add_argument("--c-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stems = load_stems()
    live = load_live_proc()
    live["catalog_id"] = live["catalog_id"].astype(str).str.strip()
    prev = {}
    sp = OUT / "summary.json"
    if sp.is_file():
        prev = json.loads(sp.read_text(encoding="utf-8"))

    a_head = prev.get("part_a")
    b_head = prev.get("part_b")
    c_head = prev.get("part_c")
    a_lc = None
    if (OUT / "arbitration_lc.csv").is_file():
        a_lc = pd.read_csv(OUT / "arbitration_lc.csv")

    run_a = args.a_only or not (args.b_only or args.c_only)
    run_b = args.b_only or not (args.a_only or args.c_only)
    run_c = args.c_only or not (args.a_only or args.b_only)
    if args.a_only:
        run_b = run_c = False
    if args.b_only:
        run_a = run_c = False
    if args.c_only:
        run_a = run_b = False

    if run_a or a_head is None:
        _lc, a_head = part_a(stems, live)
        a_lc = _lc
        print("[core04] A", a_head["rows"])
    if run_b:
        b_head = part_b()
        print("[core04] B", b_head.get("rows"))
    if run_c:
        if a_lc is None:
            a_lc, a_head = part_a(stems, live)
        _kdf, c_head = part_c(stems, live, a_lc)
        print("[core04] C", c_head["rows"])

    fired = readings(a_head or {}, b_head or {}, c_head or {})
    g4 = g4_live_516()
    summary = {
        "architect_error_25": (
            "CORE-03 R-Q4 'slope x observed phase spread' is a peak-to-peak quantity, not RMS. "
            "'8.56 of 10.48' over-states the phase share. Correct: ptp over the 0.095 px live "
            "window ~8 mmag; RMS (uniform phase) = ptp/sqrt(12) ~2.3 mmag; empirical share from "
            "CORE-01 target phase rank R^2 0.221 ~4.9 mmag in quadrature. Phase is a co-driver, "
            "not 82% of the target residual. R-Q3 stands unchanged."
        ),
        "core03_summary_fix": "0528518 regenerated session_20260914_epsf_core_03/summary.json from delivered CSVs.",
        "rms_definition": "RMS_med(d)=sqrt(mean((d-median(d))^2)) over the 134 identical-ensemble epochs, mmag.",
        "phase_rms_definition": (
            "bilinear interpolate noise-off bias surface over the live window, sample uniformly, sqrt(var); ptp alongside."
        ),
        "part_a": a_head,
        "part_b": b_head,
        "part_c": c_head,
        "readings": fired,
        "g4": g4,
    }
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

    sp.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print("[core04] readings", fired)
    print("[core04] g4", g4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
