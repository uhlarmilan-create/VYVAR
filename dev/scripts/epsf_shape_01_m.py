#!/usr/bin/env python3
"""EPSF-SHAPE-01-M sandbox measurements. Production models/catalogs/LCs/exports are read-only."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from epsf_frame_accounting import list_epsf_science_light_fits  # noqa: E402
from epsf_science_set import build_epsf_science_set  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photutils.psf import ImagePSF  # noqa: E402
from psf_photometry import (  # noqa: E402
    _psf_model_prediction_cutout,
    _resolve_psf_fit_sky,
    build_epsf_model,
    psf_photometry_stars,
)

DRAFT = REPO / "Archive" / "Drafts" / "draft_000516"
PS = DRAFT / "platesolve" / "NoFilter_60_2"
FRAMES = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"
PHOT = PS / "photometry"
LC_DIR = PHOT / "lightcurves"
OUT = REPO / "dev" / "results" / "session_20260824_epsf_shape_01_m"
PROD_EPSF = PS / "masterstar_epsf.fits"
PROD_META = PS / "masterstar_epsf_meta.json"
BO_CVN = "1498613634033133184"
CUTOUT = 17
N_FRAMES_SUBSET = 20
N_MAG_BINS = 5
OBS_DROOP = 0.671


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(raw: Any) -> str:
    try:
        return str(normalize_gaia_source_id(raw)).strip()
    except Exception:  # noqa: BLE001
        s = str(raw or "").strip()
        return "" if s.lower() in ("", "nan", "none") else s


def production_watch_files() -> list[Path]:
    files = [PROD_EPSF, PROD_META]
    files.extend(sorted(LC_DIR.glob("lightcurve_*.csv"))[:8])
    files.extend(sorted((PHOT / "lightcurves_reports" / "aavso").glob("*.txt"))[:4])
    files.extend(sorted((PHOT / "lightcurves_reports" / "varastro").glob("*.txt"))[:4])
    files.extend(sorted(FRAMES.glob("proc_*.csv"))[:4])
    return [p for p in files if p.is_file()]


def snapshot_hashes(label: str) -> dict[str, str]:
    out = {str(p.relative_to(REPO)).replace("\\", "/"): _sha(p) for p in production_watch_files()}
    (OUT / f"hashes_{label}.json").write_text(json.dumps(out, indent=2), encoding="ascii")
    return out


def pick_frame_subset(fits_list: list[Path], n: int = N_FRAMES_SUBSET) -> list[Path]:
    if len(fits_list) <= n:
        return list(fits_list)
    idx = np.linspace(0, len(fits_list) - 1, n, dtype=int)
    return [fits_list[int(i)] for i in idx]


def load_prod_psf() -> tuple[Any, dict[str, Any]]:
    meta = json.loads(PROD_META.read_text(encoding="utf-8"))
    arr = np.asarray(fits.getdata(PROD_EPSF), dtype=np.float64)
    osamp = int(meta.get("oversampling", 2) or 2)
    return ImagePSF(arr, oversampling=osamp), meta


def m1_residual_stacks(frame_subset: list[Path], science_ids: set[str]) -> dict[str, Any]:
    psf, meta = load_prod_psf()
    cutout = int(meta.get("cutout_size", CUTOUT))
    half = cutout // 2
    fwhm = float(meta.get("fwhm_px") or 3.3)
    mag_rows: list[dict[str, Any]] = []
    residuals: list[np.ndarray] = []
    mags: list[float] = []
    radii_acc: list[np.ndarray] = []
    frac_acc: list[np.ndarray] = []

    yy, xx = np.mgrid[0:cutout, 0:cutout]
    r_grid = np.hypot(xx - half, yy - half)

    for fp in frame_subset:
        proc = FRAMES / f"proc_{fp.stem}.csv"
        if not proc.is_file():
            continue
        data = np.asarray(fits.getdata(fp), dtype=np.float64)
        df = pd.read_csv(proc, low_memory=False, dtype={"catalog_id": str})
        df["catalog_id"] = df["catalog_id"].map(_norm)
        sub = df.loc[df["catalog_id"].isin(science_ids)]
        h, w = data.shape
        for _, row in sub.iterrows():
            x = float(pd.to_numeric(row.get("x"), errors="coerce"))
            y = float(pd.to_numeric(row.get("y"), errors="coerce"))
            flux = float(pd.to_numeric(row.get("psf_flux"), errors="coerce"))
            mag = float(pd.to_numeric(row.get("phot_g_mean_mag", row.get("mag")), errors="coerce"))
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(flux) and flux > 0):
                continue
            xi, yi = int(round(x)), int(round(y))
            if xi < half or yi < half or xi >= w - half or yi >= h - half:
                continue
            x1, y1 = xi - half, yi - half
            cut = np.asarray(data[y1 : y1 + cutout, x1 : x1 + cutout], dtype=np.float64)
            if cut.shape != (cutout, cutout):
                continue
            sky, _meth = _resolve_psf_fit_sky(data, cut, x, y, fwhm_px=fwhm)
            cut_sub = cut - float(sky)
            xc, yc = x - x1, y - y1
            model = _psf_model_prediction_cutout(psf, cut.shape, flux, xc, yc)
            peak = float(np.nanmax(model))
            if not (math.isfinite(peak) and peak > 0):
                continue
            resid = (cut_sub - model) / peak
            residuals.append(resid)
            mags.append(mag if math.isfinite(mag) else float("nan"))
            mag_rows.append(
                {
                    "frame": fp.name,
                    "catalog_id": row["catalog_id"],
                    "mag": mag,
                    "psf_flux": flux,
                    "psf_chi2": float(pd.to_numeric(row.get("psf_chi2"), errors="coerce")),
                }
            )
            radii_acc.append(r_grid.ravel())
            frac_acc.append(resid.ravel())

    mag_df = pd.DataFrame(mag_rows)
    mag_df.to_csv(OUT / "m1_star_frame_rows.csv", index=False)
    finite_mag = np.asarray(mags, dtype=float)
    okm = np.isfinite(finite_mag)
    if int(okm.sum()) < N_MAG_BINS:
        raise RuntimeError(f"M1: not enough finite mags ({int(okm.sum())})")
    qs = np.quantile(finite_mag[okm], np.linspace(0, 1, N_MAG_BINS + 1))
    # bright -> faint: bin 0 is brightest (lowest mag)
    r_edges = np.arange(0.0, half + 0.51, 0.5)
    r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])
    radial_rows: list[dict[str, Any]] = []
    faint_profile = None
    for b in range(N_MAG_BINS):
        if b == 0:
            mask = okm & (finite_mag <= qs[1])
        elif b == N_MAG_BINS - 1:
            mask = okm & (finite_mag > qs[b])
        else:
            mask = okm & (finite_mag > qs[b]) & (finite_mag <= qs[b + 1])
        idx = np.where(mask)[0]
        stack = np.nanmedian(np.stack([residuals[i] for i in idx], axis=0), axis=0) if len(idx) else np.full((cutout, cutout), np.nan)
        png_path = OUT / f"m1_resid_bin{b}.png"
        fits_path = OUT / f"m1_resid_bin{b}.fits"
        fits.PrimaryHDU(np.asarray(stack, dtype=np.float32)).writeto(fits_path, overwrite=True)
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(stack, origin="lower", cmap="RdBu_r", vmin=-0.2, vmax=0.2)
            ax.set_title(f"bin{b} n={len(idx)} mag {qs[b]:.2f}-{qs[b+1]:.2f}")
            fig.colorbar(im, ax=ax, fraction=0.046)
            fig.tight_layout()
            fig.savefig(png_path, dpi=120)
            plt.close(fig)
        except Exception:  # noqa: BLE001
            png_path = Path("")
        rr = r_grid.ravel()
        vv = np.asarray(stack, dtype=float).ravel()
        prof = []
        for lo, hi in zip(r_edges[:-1], r_edges[1:]):
            sel = (rr >= lo) & (rr < hi) & np.isfinite(vv)
            prof.append(float(np.nanmedian(vv[sel])) if np.any(sel) else float("nan"))
        if b == N_MAG_BINS - 1:
            faint_profile = np.asarray(prof, dtype=float)
        for rc, pv in zip(r_cent, prof):
            radial_rows.append(
                {
                    "bin": b,
                    "bin_mag_lo": float(qs[b]),
                    "bin_mag_hi": float(qs[b + 1]),
                    "n_cutouts": int(len(idx)),
                    "radius_px": float(rc),
                    "frac_resid": float(pv) if math.isfinite(pv) else float("nan"),
                }
            )
        np.save(OUT / f"m1_stack_bin{b}.npy", stack)

    rad = pd.DataFrame(radial_rows)
    rad.to_csv(OUT / "m1_radial_frac_resid.csv", index=False)
    depart = []
    if faint_profile is not None:
        for b in range(N_MAG_BINS - 1):
            sub = rad.loc[rad["bin"] == b]
            for _, row in sub.iterrows():
                i = int(np.argmin(np.abs(r_cent - float(row["radius_px"]))))
                base = float(faint_profile[i]) if i < len(faint_profile) else float("nan")
                delta = float(row["frac_resid"]) - base if math.isfinite(base) else float("nan")
                if math.isfinite(delta) and abs(delta) >= 0.05:
                    depart.append(
                        {
                            "bin": int(b),
                            "radius_px": float(row["radius_px"]),
                            "delta_vs_faint": float(delta),
                        }
                    )
    pd.DataFrame(depart).to_csv(OUT / "m1_depart_vs_faint.csv", index=False)
    return {
        "n_cutouts": int(len(residuals)),
        "n_frames": len(frame_subset),
        "mag_edges": [float(x) for x in qs],
        "n_depart_ge_0p05": int(len(depart)),
        "core_mean_abs_bright": _bin_core_abs(rad, 0, 2.0),
        "wing_mean_abs_bright": _bin_wing_abs(rad, 0, 5.0),
        "core_mean_abs_faint": _bin_core_abs(rad, N_MAG_BINS - 1, 2.0),
        "wing_mean_abs_faint": _bin_wing_abs(rad, N_MAG_BINS - 1, 5.0),
    }


def _bin_core_abs(rad: pd.DataFrame, b: int, rmax: float) -> float:
    sub = rad.loc[(rad["bin"] == b) & (rad["radius_px"] <= rmax)]
    v = pd.to_numeric(sub["frac_resid"], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(np.abs(v))) if v.size else float("nan")


def _bin_wing_abs(rad: pd.DataFrame, b: int, rmin: float) -> float:
    sub = rad.loc[(rad["bin"] == b) & (rad["radius_px"] >= rmin)]
    v = pd.to_numeric(sub["frac_resid"], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(np.abs(v))) if v.size else float("nan")


def m2_curve_of_growth(frame_subset: list[Path]) -> dict[str, Any]:
    meta = json.loads(PROD_META.read_text(encoding="utf-8"))
    arr = np.asarray(fits.getdata(PROD_EPSF), dtype=np.float64)
    osamp = int(meta.get("oversampling", 2) or 2)
    cy, cx = np.array(arr.shape) / 2.0 - 0.5
    yy, xx = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
    r_os = np.hypot(xx - cx, yy - cy)
    r_native = r_os / float(osamp)
    tot = float(np.nansum(arr))
    rmax = float(CUTOUT) / 2.0
    radii = np.arange(0.5, rmax + 0.01, 0.5)
    model_ee = []
    for r in radii:
        ee = float(np.nansum(arr[r_native <= r]) / tot) if tot > 0 else float("nan")
        model_ee.append(ee)
    edge_ee = model_ee[-1] if model_ee else float("nan")
    # Model is normalized inside the cutout, so edge EE ~ 1 by construction.
    # Truncation deficit vs an analytic Gaussian with the measured FWHM:
    fwhm = float(meta.get("fwhm_px") or 3.3)
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    gauss_ee_edge = 1.0 - math.exp(-0.5 * (rmax / sigma) ** 2)
    trunc_gauss = 1.0 - gauss_ee_edge

    emp_rows = []
    for fp in frame_subset[:8]:
        proc = FRAMES / f"proc_{fp.stem}.csv"
        if not proc.is_file():
            continue
        data = np.asarray(fits.getdata(fp), dtype=np.float64)
        df = pd.read_csv(proc, low_memory=False, dtype={"catalog_id": str})
        df["catalog_id"] = df["catalog_id"].map(_norm)
        mag = pd.to_numeric(df.get("phot_g_mean_mag", df.get("mag")), errors="coerce")
        bright = df.loc[mag.nsmallest(12).index]
        h, w = data.shape
        for _, row in bright.iterrows():
            x = float(pd.to_numeric(row.get("x"), errors="coerce"))
            y = float(pd.to_numeric(row.get("y"), errors="coerce"))
            dao = float(pd.to_numeric(row.get("dao_flux"), errors="coerce"))
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            yyf, xxf = np.mgrid[0:h, 0:w]
            rr = np.hypot(xxf - x, yyf - y)
            sky_ann = (rr >= 12) & (rr <= 16)
            sky = float(np.nanmedian(data[sky_ann])) if np.any(sky_ann) else 0.0
            sub = data - sky
            ap_r = np.arange(1.0, 16.5, 1.0)
            flux_r = [float(np.nansum(sub[rr <= r])) for r in ap_r]
            if not flux_r or not (math.isfinite(flux_r[-1]) and abs(flux_r[-1]) > 0):
                continue
            for r, f in zip(ap_r, flux_r):
                emp_rows.append(
                    {
                        "frame": fp.name,
                        "catalog_id": row["catalog_id"],
                        "radius_px": float(r),
                        "ee": float(f / flux_r[-1]),
                        "dao_flux": dao,
                    }
                )
    emp = pd.DataFrame(emp_rows)
    emp.to_csv(OUT / "m2_empirical_growth.csv", index=False)
    model_df = pd.DataFrame({"radius_px": radii, "model_ee": model_ee})
    model_df.to_csv(OUT / "m2_model_growth.csv", index=False)
    emp_at_edge = float("nan")
    if not emp.empty:
        near = emp.loc[np.abs(emp["radius_px"] - 8.5) <= 0.6]
        if not near.empty:
            emp_at_edge = float(near["ee"].median())
    expected_ratio_from_trunc = float(edge_ee)  # ~1 for model self-normalization
    # Honest: model EE at edge cannot diagnose missing wings; Gaussian analytic trunc does.
    return {
        "model_ee_at_edge": float(edge_ee) if math.isfinite(edge_ee) else float("nan"),
        "gaussian_ee_at_8p5px": float(gauss_ee_edge),
        "gaussian_truncation_deficit": float(trunc_gauss),
        "empirical_ee_at_8p5_median": emp_at_edge,
        "observed_bo_cvn_psf_dao": OBS_DROOP,
        "trunc_explains_half_of_droop": bool(trunc_gauss >= 0.5 * (1.0 - OBS_DROOP)),
        "note": (
            "Production ePSF is flux-normalized inside the 17 px cutout, so model EE at the "
            "edge is ~1 by construction and cannot by itself explain a 0.671 PSF/DAO ratio. "
            "Analytic Gaussian truncation at FWHM=3.3 px / r=8.5 px is tiny."
        ),
    }


def _fit_science_on_subset(
    *,
    model_path: Path,
    frame_subset: list[Path],
    science_ids: set[str],
    two_pass: bool = False,
    label: str,
) -> pd.DataFrame:
    out_csv = OUT / f"fits_{label}.csv"
    rows: list[dict[str, Any]] = []
    done_frames: set[str] = set()
    if out_csv.is_file():
        prev = pd.read_csv(out_csv, dtype={"catalog_id": str})
        rows = prev.to_dict("records")
        done_frames = {str(x) for x in prev["frame"].astype(str).unique()}
        print(f"  resume {label} {len(done_frames)}/{len(frame_subset)} frames", flush=True)
    for i_fp, fp in enumerate(frame_subset):
        if fp.name in done_frames:
            continue
        print(f"  fit {label} {i_fp+1}/{len(frame_subset)} {fp.name}", flush=True)
        proc = FRAMES / f"proc_{fp.stem}.csv"
        if not proc.is_file():
            continue
        data = np.asarray(fits.getdata(fp), dtype=np.float64)
        hdr = fits.getheader(fp)
        df = pd.read_csv(proc, low_memory=False, dtype={"catalog_id": str})
        df["catalog_id"] = df["catalog_id"].map(_norm)
        pos = df.loc[df["catalog_id"].isin(science_ids), ["x", "y", "catalog_id"]].copy()
        if "name" not in pos.columns:
            pos["name"] = pos["catalog_id"]
        dao = {
            _norm(r["catalog_id"]): float(pd.to_numeric(r.get("dao_flux"), errors="coerce"))
            for _, r in df.iterrows()
        }
        ref = np.array([dao.get(str(c), float("nan")) for c in pos["catalog_id"]], dtype=float)
        fit1 = psf_photometry_stars(
            data,
            hdr,
            pos,
            model_path,
            apply_aperture_correction=False,
            ref_fluxes=ref,
            use_iterative=False,
        )
        fit_use = fit1
        if two_pass:
            ref2 = pd.to_numeric(fit1.get("psf_flux"), errors="coerce").to_numpy(dtype=float)
            fit_use = psf_photometry_stars(
                data,
                hdr,
                pos,
                model_path,
                apply_aperture_correction=False,
                ref_fluxes=ref2,
                use_iterative=False,
            )
        merged = pos.merge(fit_use, on="catalog_id", how="left", suffixes=("", "_fit"))
        merged = merged.merge(
            df[["catalog_id", "dao_flux", "phot_g_mean_mag", "mag", "psf_chi2", "psf_flux"]],
            on="catalog_id",
            how="left",
            suffixes=("", "_prod"),
        )
        for _, r in merged.iterrows():
            pf = float(pd.to_numeric(r.get("psf_flux"), errors="coerce"))
            dao_f = float(pd.to_numeric(r.get("dao_flux"), errors="coerce"))
            ratio = pf / dao_f if math.isfinite(pf) and math.isfinite(dao_f) and dao_f > 0 else float("nan")
            rows.append(
                {
                    "label": label,
                    "frame": fp.name,
                    "catalog_id": r.get("catalog_id"),
                    "mag": float(pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce")),
                    "psf_flux": pf,
                    "dao_flux": dao_f,
                    "psf_dao_ratio": ratio,
                    "psf_chi2": float(pd.to_numeric(r.get("psf_chi2"), errors="coerce")),
                    "psf_fit_ok": bool(r.get("psf_fit_ok")),
                    "two_pass": bool(two_pass),
                }
            )
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def m3_sensitivity(frame_subset: list[Path], science_ids: set[str], db: VyvarDatabase) -> dict[str, Any]:
    sha_before = _sha(PROD_EPSF)
    combos = [
        (1, "quadratic"),
        (1, "quartic"),
        (2, "quadratic"),
        (2, "quartic"),
        (4, "quadratic"),
        (4, "quartic"),
    ]
    summary = []
    ms_fits = PS / "MASTERSTAR.fits"
    ms_csv = PS / "masterstars_full_match.csv"
    for osamp, kern in combos:
        tag = f"os{osamp}_{kern}"
        print("M3 combo", tag, flush=True)
        sdir = OUT / "models" / tag
        sdir.mkdir(parents=True, exist_ok=True)
        existing = sdir / "masterstar_epsf.fits"
        try:
            if existing.is_file():
                model_path = existing
            else:
                model_path = build_epsf_model(
                    ms_fits,
                    ms_csv,
                    db,
                    516,
                    oversampling=int(osamp),
                    sandbox_output_dir=sdir,
                    smoothing_kernel=str(kern),
                    meta_extra={"shape01m_tag": tag},
                )
        except Exception as exc:  # noqa: BLE001
            print("M3 build fail", tag, type(exc).__name__, flush=True)
            summary.append(
                {
                    "tag": tag,
                    "oversampling": osamp,
                    "smoothing_kernel": kern,
                    "n_stars_used": None,
                    "model_sha256": "",
                    "sandbox": True,
                    "build_ok": False,
                    "build_error": f"{type(exc).__name__}: {exc}",
                    "ratio_median": float("nan"),
                    "ratio_corr_mag": float("nan"),
                    "chi2_median_bright": float("nan"),
                    "chi2_median_faint": float("nan"),
                    "n_fail_last_iter": None,
                    "determinism_note": "EPSFBuilder maxiters=15; sandbox_output_dir isolates writes",
                }
            )
            continue
        meta = json.loads((sdir / "masterstar_epsf_meta.json").read_text(encoding="utf-8"))
        fit_csv = OUT / f"fits_{tag}.csv"
        n_need = len(frame_subset)
        if fit_csv.is_file() and int(pd.read_csv(fit_csv)["frame"].nunique()) >= n_need:
            fits_df = pd.read_csv(fit_csv, dtype={"catalog_id": str})
            print("M3 fit resume", tag, "nrows", len(fits_df), flush=True)
        else:
            fits_df = _fit_science_on_subset(
                model_path=Path(model_path),
                frame_subset=frame_subset,
                science_ids=science_ids,
                label=tag,
            )
        print("M3 fit done", tag, "nrows", len(fits_df), flush=True)
        bright = fits_df.loc[fits_df["mag"] <= fits_df["mag"].quantile(0.2)]
        faint = fits_df.loc[fits_df["mag"] >= fits_df["mag"].quantile(0.8)]
        curve = meta.get("iteration_failure_curve") or []
        summary.append(
            {
                "tag": tag,
                "oversampling": osamp,
                "smoothing_kernel": kern,
                "n_stars_used": meta.get("n_stars_used"),
                "model_sha256": _sha(Path(model_path)),
                "sandbox": True,
                "build_ok": True,
                "build_error": "",
                "ratio_median": float(pd.to_numeric(fits_df["psf_dao_ratio"], errors="coerce").median()),
                "ratio_corr_mag": float(
                    pd.to_numeric(fits_df["psf_dao_ratio"], errors="coerce").corr(
                        pd.to_numeric(fits_df["mag"], errors="coerce")
                    )
                )
                if len(fits_df) > 5
                else float("nan"),
                "chi2_median_bright": float(pd.to_numeric(bright["psf_chi2"], errors="coerce").median())
                if not bright.empty
                else float("nan"),
                "chi2_median_faint": float(pd.to_numeric(faint["psf_chi2"], errors="coerce").median())
                if not faint.empty
                else float("nan"),
                "n_fail_last_iter": int(curve[-1].get("n_fail", 0)) if curve else None,
                "determinism_note": "EPSFBuilder maxiters=15, no global RNG; sandbox_output_dir isolates writes",
            }
        )
        (sdir / "iteration_failure_curve.json").write_text(json.dumps(curve, indent=2), encoding="ascii")
    sha_after = _sha(PROD_EPSF)
    pd.DataFrame(summary).to_csv(OUT / "m3_sensitivity.csv", index=False)
    if sha_before != sha_after:
        raise RuntimeError("INV: production ePSF SHA moved during M3 sandbox builds")
    return {"models": summary, "prod_epsf_sha_unchanged": True, "prod_epsf_sha256": sha_before}


def m4_two_pass(frame_subset: list[Path], science_ids: set[str]) -> dict[str, Any]:
    # Brightest 30 science-set stars by catalog mag on first frame
    proc0 = FRAMES / f"proc_{frame_subset[0].stem}.csv"
    df0 = pd.read_csv(proc0, low_memory=False, dtype={"catalog_id": str})
    df0["catalog_id"] = df0["catalog_id"].map(_norm)
    sub = df0.loc[df0["catalog_id"].isin(science_ids)].copy()
    sub["_mag"] = pd.to_numeric(sub.get("phot_g_mean_mag", sub.get("mag")), errors="coerce")
    bright_ids = set(sub.nsmallest(30, "_mag")["catalog_id"].tolist())
    one = _fit_science_on_subset(
        model_path=PROD_EPSF,
        frame_subset=frame_subset,
        science_ids=bright_ids,
        two_pass=False,
        label="m4_onepass",
    )
    two = _fit_science_on_subset(
        model_path=PROD_EPSF,
        frame_subset=frame_subset,
        science_ids=bright_ids,
        two_pass=True,
        label="m4_twopass",
    )
    j = one.merge(two, on=["frame", "catalog_id"], suffixes=("_1", "_2"))
    j["d_chi2"] = j["psf_chi2_2"] - j["psf_chi2_1"]
    j["d_flux_frac"] = (j["psf_flux_2"] - j["psf_flux_1"]) / j["psf_flux_1"]
    j.to_csv(OUT / "m4_two_pass_compare.csv", index=False)
    return {
        "n_stars": int(len(bright_ids)),
        "n_rows": int(len(j)),
        "chi2_median_one": float(pd.to_numeric(j["psf_chi2_1"], errors="coerce").median()),
        "chi2_median_two": float(pd.to_numeric(j["psf_chi2_2"], errors="coerce").median()),
        "d_chi2_median": float(pd.to_numeric(j["d_chi2"], errors="coerce").median()),
        "d_flux_frac_median": float(pd.to_numeric(j["d_flux_frac"], errors="coerce").median()),
        "d_flux_frac_p95_abs": float(np.nanpercentile(np.abs(pd.to_numeric(j["d_flux_frac"], errors="coerce")), 95)),
    }


def m5_rank(m1: dict, m2: dict, m3: dict, m4: dict) -> dict[str, Any]:
    # Discriminator is WHERE bright bins depart from the faint radial baseline,
    # not |core_bright| vs |core_faint| (faint core |resid| can be larger).
    n_depart = int(m1.get("n_depart_ge_0p05") or 0)
    depart_path = OUT / "m1_depart_vs_faint.csv"
    n_core_dep = 0
    n_wing_dep = 0
    if depart_path.is_file() and n_depart:
        dep = pd.read_csv(depart_path)
        rr = pd.to_numeric(dep["radius_px"], errors="coerce")
        n_core_dep = int((rr <= 2.5).sum())
        n_wing_dep = int((rr >= 5.0).sum())
    shape = "core" if n_core_dep >= n_wing_dep else "wing"
    trunc = float(m2.get("gaussian_truncation_deficit") or 0)
    droop = 1.0 - OBS_DROOP
    h3_frac = min(1.0, trunc / droop) if droop > 0 else 0.0
    chi2_drop = float(m4.get("chi2_median_one") or 0) - float(m4.get("chi2_median_two") or 0)
    flux_move = abs(float(m4.get("d_flux_frac_median") or 0))
    # Rank by contribution to (a) ratio droop (b) bright chi2
    rows = [
        {
            "id": "H1",
            "name": "oversampling=2 insufficient at FWHM 3.3 px",
            "ratio_droop": "measured_in_m3",
            "bright_chi2": "measured_in_m3",
            "rig_specific": True,
            "rig_note": "wide 9.77 arcsec/px undersampled; Anderson & King 2000 core sampling",
        },
        {
            "id": "H2",
            "name": "quadratic smoothing kernel core bias",
            "ratio_droop": "measured_in_m3",
            "bright_chi2": "measured_in_m3",
            "rig_specific": False,
            "rig_note": "DAOPHOT II quadratic smooth is generic; effect size may grow when undersampled",
        },
        {
            "id": "H3",
            "name": "wing truncation cutout 17 px",
            "ratio_droop": h3_frac,
            "bright_chi2": "wing" if shape == "wing" else "low",
            "rig_specific": True,
            "rig_note": "undersampled wide PSF has compact core; truncation is small vs 0.671 droop",
        },
        {
            "id": "H4",
            "name": "one-pass F_model init bias (FD-A chi2)",
            "ratio_droop": flux_move,
            "bright_chi2": chi2_drop,
            "rig_specific": False,
            "rig_note": "chi2 statistic artifact if flux shift is negligible; generic FD-A",
        },
    ]
    return {
        "primary_residual_shape": shape,
        "h3_trunc_fraction_of_droop": h3_frac,
        "h4_chi2_drop": chi2_drop,
        "h4_flux_frac_shift": flux_move,
        "hypotheses": rows,
        "m3": m3,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)
    hashes_before = snapshot_hashes("before")
    # positive control: sandbox file hash must change when we write it
    ctrl = OUT / "positive_control.txt"
    ctrl.write_text("before\n", encoding="ascii")
    sha_ctrl_before = _sha(ctrl)
    ctrl.write_text("after\n", encoding="ascii")
    sha_ctrl_after = _sha(ctrl)
    if sha_ctrl_before == sha_ctrl_after:
        raise RuntimeError("hash guard positive control failed (identical hashes)")

    sci = build_epsf_science_set(PS)
    science_ids = set(sci.catalog_ids)
    fits_list = list_epsf_science_light_fits(FRAMES)
    subset = pick_frame_subset(fits_list, N_FRAMES_SUBSET)
    (OUT / "frame_subset.txt").write_text("\n".join(p.name for p in subset) + "\n", encoding="ascii")

    print(f"M1 science_set={sci.n_total} frames_subset={len(subset)}")
    m1_path = OUT / "m1_summary.json"
    if m1_path.is_file():
        m1 = json.loads(m1_path.read_text(encoding="utf-8"))
        print("M1 resume", m1.get("n_cutouts"))
    else:
        m1 = m1_residual_stacks(subset, science_ids)
        m1_path.write_text(json.dumps(m1, indent=2), encoding="ascii")
        print("M1 done", m1.get("n_cutouts"))

    m2_path = OUT / "m2_summary.json"
    if m2_path.is_file():
        m2 = json.loads(m2_path.read_text(encoding="utf-8"))
        print("M2 resume", m2.get("gaussian_truncation_deficit"))
    else:
        m2 = m2_curve_of_growth(subset)
        m2_path.write_text(json.dumps(m2, indent=2), encoding="ascii")
        print("M2 done", m2.get("gaussian_truncation_deficit"))

    m3_path = OUT / "m3_summary.json"
    if m3_path.is_file():
        m3 = json.loads(m3_path.read_text(encoding="utf-8"))
        print("M3 resume", len(m3.get("models") or []))
    else:
        cfg = AppConfig(project_root=REPO)
        db = VyvarDatabase(cfg.database_path)
        try:
            m3 = m3_sensitivity(subset, science_ids, db)
        finally:
            db.close()
        m3_path.write_text(json.dumps(m3, indent=2), encoding="ascii")
        print("M3 done")

    m4_path = OUT / "m4_summary.json"
    if m4_path.is_file():
        m4 = json.loads(m4_path.read_text(encoding="utf-8"))
        print("M4 resume", m4)
    else:
        m4 = m4_two_pass(subset, science_ids)
        m4_path.write_text(json.dumps(m4, indent=2), encoding="ascii")
        print("M4 done", m4)

    m5 = m5_rank(m1, m2, m3, m4)
    (OUT / "m5_synthesis.json").write_text(json.dumps(m5, indent=2), encoding="ascii")

    hashes_after = snapshot_hashes("after")
    if hashes_before != hashes_after:
        diff = [k for k in hashes_before if hashes_before.get(k) != hashes_after.get(k)]
        raise RuntimeError(f"production files changed: {diff[:8]}")
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    summary = {
        "elapsed_s": elapsed,
        "prod_epsf_sha256": _sha(PROD_EPSF),
        "n_stars_used": json.loads(PROD_META.read_text(encoding="utf-8")).get("n_stars_used"),
        "positive_control_changed": sha_ctrl_before != sha_ctrl_after,
        "production_hashes_identical": True,
        "m1": m1,
        "m2": m2,
        "m4": m4,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    print("SHAPE-01-M done in", elapsed, "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
