#!/usr/bin/env python3
"""EPSF-SHAPE-01-F sandbox: F1 builder diagnosis, F2/F2b fitter+norm, F3 merge check.

Production ePSF / aperture LCs / AAVSO / VarAstro are read-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import traceback
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
from epsf_psf_merge import assert_inv_psf_additive_01, merge_psf_into_sidecar  # noqa: E402
from epsf_science_set import build_epsf_science_set  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from photutils.psf import ImagePSF  # noqa: E402
from psf_photometry import (  # noqa: E402
    InstrumentedEPSFBuilder,
    build_epsf_model,
    psf_photometry_stars,
)

DRAFT = REPO / "Archive" / "Drafts" / "draft_000516"
PS = DRAFT / "platesolve" / "NoFilter_60_2"
FRAMES = DRAFT / "detrended_aligned" / "lights" / "NoFilter_60_2"
PHOT = PS / "photometry"
LC_DIR = PHOT / "lightcurves"
OUT = REPO / "dev" / "results" / "session_20260824_epsf_shape_01_f"
PROD_EPSF = PS / "masterstar_epsf.fits"
PROD_META = PS / "masterstar_epsf_meta.json"
BO_CVN = "1498613634033133184"
N_FRAMES_SUBSET = 20
N_BRIGHT = 30
OBS_DROOP = 0.671
INPUT_FWHM = 3.301


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
    files.extend(sorted(p for p in LC_DIR.glob("lightcurve_*.csv") if "_psf" not in p.name)[:6])
    files.extend(sorted((PHOT / "lightcurves_reports" / "aavso").glob("*.txt"))[:4])
    files.extend(sorted((PHOT / "lightcurves_reports" / "varastro").glob("*.txt"))[:4])
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


def rms_median_check(values: np.ndarray) -> dict[str, float | bool]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": float("nan"), "rms": float("nan"), "rms_vs_abs_median": False}
    med = float(np.median(v))
    rms = float(np.sqrt(np.mean(v**2)))
    return {
        "n": int(v.size),
        "median": med,
        "rms": rms,
        "rms_vs_abs_median": bool(abs(rms - abs(med)) <= 0.25 * max(abs(med), abs(rms), 1e-12)),
    }


class DiagEPSFBuilder(InstrumentedEPSFBuilder):
    """Per-star status + recentering deltas + first exception (INV-PSF-FRAME-01)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.star_iter_rows: list[dict[str, Any]] = []
        self.recenter_rows: list[dict[str, Any]] = []
        self.first_exceptions: list[dict[str, str]] = []

    def _record_exc(self, where: str, exc: BaseException) -> None:
        if self.first_exceptions:
            return
        self.first_exceptions.append(
            {
                "where": where,
                "exception_class": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    def _recenter_epsf(self, epsf, **kwargs):  # noqa: ANN001
        data = np.asarray(getattr(epsf, "data", []), dtype=float)
        cy, cx = np.array(data.shape) / 2.0 - 0.5
        try:
            out = super()._recenter_epsf(epsf, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self._record_exc("_recenter_epsf", exc)
            raise
        arr = np.asarray(out, dtype=float)
        yy, xx = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
        w = np.clip(arr, 0, None)
        sw = float(np.nansum(w))
        if sw > 0:
            cx2 = float(np.nansum(xx * w) / sw)
            cy2 = float(np.nansum(yy * w) / sw)
        else:
            cx2, cy2 = float("nan"), float("nan")
        self.recenter_rows.append(
            {
                "n": len(self.recenter_rows) + 1,
                "dx_px_os": cx2 - cx,
                "dy_px_os": cy2 - cy,
                "shift_os": float(math.hypot(cx2 - cx, cy2 - cy))
                if math.isfinite(cx2)
                else float("nan"),
            }
        )
        return out

    def _process_iteration(self, stars, epsf, iter_num):  # noqa: ANN001
        try:
            result = super()._process_iteration(stars, epsf, iter_num)
        except Exception as exc:  # noqa: BLE001
            self._record_exc("_process_iteration", exc)
            raise
        all_stars = list(getattr(stars, "all_stars", stars))
        for i, star in enumerate(all_stars):
            st = int(getattr(star, "_fit_error_status", 0) or 0)
            ctr = getattr(star, "cutout_center", (float("nan"), float("nan")))
            orig = getattr(star, "_center_original", (float("nan"), float("nan")))
            try:
                dx = float(ctr[0]) - float(orig[0])
                dy = float(ctr[1]) - float(orig[1])
            except Exception:  # noqa: BLE001
                dx, dy = float("nan"), float("nan")
            self.star_iter_rows.append(
                {
                    "iteration": int(iter_num),
                    "star_index": i,
                    "status": st,
                    "excluded": bool(getattr(star, "_excluded_from_fit", False)),
                    "cx": float(ctr[0]) if ctr is not None else float("nan"),
                    "cy": float(ctr[1]) if ctr is not None else float("nan"),
                    "dcenter_px": float(math.hypot(dx, dy)) if math.isfinite(dx) else float("nan"),
                }
            )
        return result


def f2b_norm_audit() -> dict[str, Any]:
    meta = json.loads(PROD_META.read_text(encoding="utf-8"))
    arr = np.asarray(fits.getdata(PROD_EPSF), dtype=np.float64)
    osamp = int(meta.get("oversampling", 2) or 2)
    cy, cx = np.array(arr.shape) / 2.0 - 0.5
    yy, xx = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
    r_native = np.hypot(xx - cx, yy - cy) / float(osamp)
    tot = float(np.nansum(arr))
    abs_tot = float(np.nansum(np.abs(arr)))
    r_edges = np.arange(0.0, 9.01, 0.5)
    rows = []
    for lo, hi in zip(r_edges[:-1], r_edges[1:]):
        sel = (r_native >= lo) & (r_native < hi)
        rows.append(
            {
                "r_lo": float(lo),
                "r_hi": float(hi),
                "sum": float(np.nansum(arr[sel])),
                "sum_abs": float(np.nansum(np.abs(arr[sel]))),
                "n_pix": int(sel.sum()),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "f2b_radial_norm.csv", index=False)
    outer = r_native > 8.5
    inner = r_native <= 8.5
    neg = arr < 0
    pedestal = float(np.nanmedian(arr[outer])) if np.any(outer) else float("nan")
    ee_circ = float(np.nansum(arr[inner]) / tot) if tot != 0 else float("nan")
    frac_outer = float(np.nansum(arr[outer]) / tot) if tot != 0 else float("nan")
    frac_neg = float(np.nansum(arr[neg]) / tot) if tot != 0 else float("nan")
    frac_neg_abs = float(np.nansum(np.abs(arr[neg])) / abs_tot) if abs_tot > 0 else float("nan")
    # Geometry-only: fraction of square pixels outside r=8.5
    geom_outer = float(np.mean(outer))
    geometry_explains_ee = bool(abs((1.0 - ee_circ) - geom_outer) < 0.05) and abs(frac_neg) < 0.02
    out = {
        "shape": list(arr.shape),
        "sum": tot,
        "peak": float(np.nanmax(arr)),
        "min": float(np.nanmin(arr)),
        "ee_circular_8p5": ee_circ,
        "frac_norm_r_gt_8p5": frac_outer,
        "frac_norm_negative": frac_neg,
        "frac_abs_in_negative": frac_neg_abs,
        "outer_annulus_median": pedestal,
        "geom_frac_pixels_r_gt_8p5": geom_outer,
        "ee_is_geometry_not_pedestal": geometry_explains_ee,
        "peak_over_abs_min": float(np.nanmax(arr) / abs(np.nanmin(arr)))
        if np.nanmin(arr) < 0
        else float("nan"),
    }
    (OUT / "f2b_norm_audit.json").write_text(json.dumps(out, indent=2), encoding="ascii")
    # Pedestal-subtracted + renormalized sandbox copy
    arr2 = arr - pedestal
    s2 = float(np.nansum(arr2))
    os2 = int(osamp)
    if s2 != 0:
        arr2 = arr2 * (os2**2 / s2)
    sandbox_psf = OUT / "models" / "prod_pedestal_sub" / "masterstar_epsf.fits"
    sandbox_psf.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.asarray(arr2, dtype=np.float32)).writeto(sandbox_psf, overwrite=True)
    meta2 = dict(meta)
    meta2["sandbox_output"] = True
    meta2["shape01f_tag"] = "prod_pedestal_sub"
    meta2["pedestal_subtracted"] = pedestal
    (sandbox_psf.parent / "masterstar_epsf_meta.json").write_text(
        json.dumps(meta2, indent=2), encoding="ascii"
    )
    out["pedestal_sandbox"] = str(sandbox_psf.relative_to(REPO)).replace("\\", "/")
    return out


def f1_builds(db: VyvarDatabase) -> dict[str, Any]:
    import psf_photometry as pp

    ms_fits = PS / "MASTERSTAR.fits"
    ms_csv = PS / "masterstars_full_match.csv"
    attempts = [
        {"tag": "os4_quad_maxiters15", "osamp": 4, "kern": "quadratic", "bkw": {"maxiters": 15}},
        {"tag": "os4_quad_maxiters6", "osamp": 4, "kern": "quadratic", "bkw": {"maxiters": 6}},
        {
            "tag": "os4_quad_recenter1",
            "osamp": 4,
            "kern": "quadratic",
            "bkw": {"maxiters": 15, "recentering_maxiters": 1},
        },
        {
            "tag": "os4_quad_box3",
            "osamp": 4,
            "kern": "quadratic",
            "bkw": {"maxiters": 15, "recentering_boxsize": (3, 3)},
        },
    ]
    summary = []
    orig = pp.InstrumentedEPSFBuilder
    for att in attempts:
        tag = att["tag"]
        print("F1", tag, flush=True)
        sdir = OUT / "models" / tag
        sdir.mkdir(parents=True, exist_ok=True)
        last_builder: DiagEPSFBuilder | None = None

        class _Hook(DiagEPSFBuilder):
            def __init__(self, *a: Any, **k: Any) -> None:
                super().__init__(*a, **k)
                nonlocal last_builder
                last_builder = self

        pp.InstrumentedEPSFBuilder = _Hook
        rec: dict[str, Any] = {
            "tag": tag,
            "oversampling": att["osamp"],
            "smoothing_kernel": att["kern"],
            "builder_kwargs": {k: str(v) for k, v in att["bkw"].items()},
        }
        try:
            path = build_epsf_model(
                ms_fits,
                ms_csv,
                db,
                516,
                oversampling=int(att["osamp"]),
                sandbox_output_dir=sdir,
                smoothing_kernel=str(att["kern"]),
                builder_kwargs=dict(att["bkw"]),
                meta_extra={"shape01f_tag": tag},
            )
            meta = json.loads((sdir / "masterstar_epsf_meta.json").read_text(encoding="utf-8"))
            curve = meta.get("iteration_failure_curve") or []
            rec.update(
                {
                    "build_ok": True,
                    "build_error": "",
                    "n_stars_used": meta.get("n_stars_used"),
                    "epsf_fwhm_native_px": (meta.get("epsf_qc") or {}).get("epsf_fwhm_native_px"),
                    "n_fail_last_iter": int(curve[-1].get("n_fail", 0)) if curve else None,
                    "n_status_3_last": int(curve[-1].get("n_status_3", 0)) if curve else None,
                    "n_iters": len(curve),
                    "model_sha256": _sha(Path(path)),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rec.update(
                {
                    "build_ok": False,
                    "build_error": f"{type(exc).__name__}: {exc}",
                    "n_stars_used": None,
                    "epsf_fwhm_native_px": None,
                    "n_fail_last_iter": None,
                    "n_status_3_last": None,
                    "n_iters": None,
                    "model_sha256": "",
                }
            )
        finally:
            pp.InstrumentedEPSFBuilder = orig
        if last_builder is not None:
            pd.DataFrame(last_builder.star_iter_rows).to_csv(sdir / "star_iter_status.csv", index=False)
            pd.DataFrame(last_builder.recenter_rows).to_csv(sdir / "recenter_deltas.csv", index=False)
            (sdir / "first_exceptions.json").write_text(
                json.dumps(last_builder.first_exceptions, indent=2), encoding="ascii"
            )
            rec["n_star_iter_rows"] = len(last_builder.star_iter_rows)
            rec["n_recenter_rows"] = len(last_builder.recenter_rows)
            rec["first_exception"] = last_builder.first_exceptions[0] if last_builder.first_exceptions else None
        rec["converged"] = bool(
            rec.get("build_ok")
            and rec.get("n_fail_last_iter") is not None
            and int(rec["n_fail_last_iter"]) <= 2
            and rec.get("epsf_fwhm_native_px") not in (None, "null")
            and 2.5 <= float(rec.get("epsf_fwhm_native_px") or 0) <= 4.0
        )
        summary.append(rec)
    pd.DataFrame(summary).to_csv(OUT / "f1_attempts.csv", index=False)
    return {"attempts": summary, "prod_epsf_sha256": _sha(PROD_EPSF)}


def _bright_ids(frame: Path, science_ids: set[str], n: int = N_BRIGHT) -> set[str]:
    proc = FRAMES / f"proc_{frame.stem}.csv"
    df = pd.read_csv(proc, low_memory=False, dtype={"catalog_id": str})
    df["catalog_id"] = df["catalog_id"].map(_norm)
    sub = df.loc[df["catalog_id"].isin(science_ids)].copy()
    sub["_mag"] = pd.to_numeric(sub.get("phot_g_mean_mag", sub.get("mag")), errors="coerce")
    return set(sub.nsmallest(n, "_mag")["catalog_id"].tolist())


def _fit_subset(
    *,
    model_path: Path,
    frame_subset: list[Path],
    star_ids: set[str],
    label: str,
    use_iterative: bool,
) -> pd.DataFrame:
    out_csv = OUT / f"fits_{label}.csv"
    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if out_csv.is_file():
        prev = pd.read_csv(out_csv, dtype={"catalog_id": str})
        rows = prev.to_dict("records")
        done = {str(x) for x in prev["frame"].astype(str).unique()}
    for i_fp, fp in enumerate(frame_subset):
        if fp.name in done:
            continue
        print(f"  fit {label} {i_fp+1}/{len(frame_subset)} {fp.name}", flush=True)
        proc = FRAMES / f"proc_{fp.stem}.csv"
        data = np.asarray(fits.getdata(fp), dtype=np.float64)
        hdr = fits.getheader(fp)
        df = pd.read_csv(proc, low_memory=False, dtype={"catalog_id": str})
        df["catalog_id"] = df["catalog_id"].map(_norm)
        pos = df.loc[df["catalog_id"].isin(star_ids), ["x", "y", "catalog_id"]].copy()
        pos["name"] = pos["catalog_id"]
        dao = {
            _norm(r["catalog_id"]): float(pd.to_numeric(r.get("dao_flux"), errors="coerce"))
            for _, r in df.iterrows()
        }
        ap = {
            _norm(r["catalog_id"]): float(pd.to_numeric(r.get("flux"), errors="coerce"))
            for _, r in df.iterrows()
        }
        mag = {
            _norm(r["catalog_id"]): float(pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce"))
            for _, r in df.iterrows()
        }
        ref = np.array([dao.get(str(c), float("nan")) for c in pos["catalog_id"]], dtype=float)
        fit = psf_photometry_stars(
            data,
            hdr,
            pos,
            model_path,
            apply_aperture_correction=False,
            ref_fluxes=ref,
            use_iterative=use_iterative,
        )
        keep = [c for c in fit.columns if c not in ("x", "y", "name")]
        merged = pos.merge(fit[keep], on="catalog_id", how="left")
        for _, r in merged.iterrows():
            cid = str(r.get("catalog_id"))
            pf = float(pd.to_numeric(r.get("psf_flux"), errors="coerce"))
            dao_f = dao.get(cid, float("nan"))
            ap_f = ap.get(cid, float("nan"))
            rows.append(
                {
                    "label": label,
                    "frame": fp.name,
                    "catalog_id": cid,
                    "mag": mag.get(cid, float("nan")),
                    "psf_flux": pf,
                    "dao_flux": dao_f,
                    "aperture_flux": ap_f,
                    "psf_dao_ratio": pf / dao_f if dao_f and dao_f > 0 and math.isfinite(pf) else float("nan"),
                    "psf_ap_ratio": pf / ap_f if ap_f and ap_f > 0 and math.isfinite(pf) else float("nan"),
                    "psf_chi2": float(pd.to_numeric(r.get("psf_chi2"), errors="coerce")),
                    "psf_fit_ok": bool(r.get("psf_fit_ok")),
                    "psf_group_n": float(pd.to_numeric(r.get("psf_group_n"), errors="coerce")),
                    "n_iter_flag": int(use_iterative),
                    "x_fit": float(pd.to_numeric(r.get("x_fit"), errors="coerce")),
                    "y_fit": float(pd.to_numeric(r.get("y_fit"), errors="coerce")),
                }
            )
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


def f2_fitter_split(frame_subset: list[Path], science_ids: set[str]) -> dict[str, Any]:
    bright = _bright_ids(frame_subset[0], science_ids)
    (OUT / "f2_bright_ids.txt").write_text("\n".join(sorted(bright)) + "\n", encoding="ascii")
    one = _fit_subset(
        model_path=PROD_EPSF,
        frame_subset=frame_subset,
        star_ids=bright,
        label="f2_single",
        use_iterative=False,
    )
    ite = _fit_subset(
        model_path=PROD_EPSF,
        frame_subset=frame_subset,
        star_ids=bright,
        label="f2_iter",
        use_iterative=True,
    )
    j = one.merge(ite, on=["frame", "catalog_id"], suffixes=("_single", "_iter"))
    j["iter_dao"] = j["psf_dao_ratio_iter"]
    j["single_dao"] = j["psf_dao_ratio_single"]
    j["iter_single"] = j["psf_flux_iter"] / j["psf_flux_single"]
    j["dchi2"] = j["psf_chi2_iter"] - j["psf_chi2_single"]
    j.to_csv(OUT / "f2_three_way.csv", index=False)
    checks = {
        "iter_dao": rms_median_check(j["iter_dao"].to_numpy()),
        "single_dao": rms_median_check(j["single_dao"].to_numpy()),
        "iter_single": rms_median_check(j["iter_single"].to_numpy()),
        "dchi2": rms_median_check(j["dchi2"].to_numpy()),
    }
    mag = pd.to_numeric(j["mag_iter"], errors="coerce")
    grp = pd.to_numeric(j["psf_group_n_iter"], errors="coerce")
    corr_mag = float(pd.to_numeric(j["iter_single"], errors="coerce").corr(mag)) if len(j) > 5 else float("nan")
    corr_grp = float(pd.to_numeric(j["iter_single"], errors="coerce").corr(grp)) if len(j) > 5 else float("nan")
    by_frame = j.groupby("frame")["iter_single"].median()
    frame_spread = float(by_frame.std()) if len(by_frame) > 1 else float("nan")
    out = {
        "n_stars": int(len(bright)),
        "n_rows": int(len(j)),
        "checks": checks,
        "iter_dao_median": checks["iter_dao"]["median"],
        "single_dao_median": checks["single_dao"]["median"],
        "iter_single_median": checks["iter_single"]["median"],
        "dchi2_median": checks["dchi2"]["median"],
        "corr_iter_single_vs_mag": corr_mag,
        "corr_iter_single_vs_group_n": corr_grp,
        "iter_single_frame_median_std": frame_spread,
        "chi2_median_iter": float(pd.to_numeric(j["psf_chi2_iter"], errors="coerce").median()),
        "chi2_median_single": float(pd.to_numeric(j["psf_chi2_single"], errors="coerce").median()),
    }
    (OUT / "f2_summary.json").write_text(json.dumps(out, indent=2), encoding="ascii")
    return out


def f2b_refit(frame_subset: list[Path], science_ids: set[str], f2: dict[str, Any]) -> dict[str, Any]:
    model = OUT / "models" / "prod_pedestal_sub" / "masterstar_epsf.fits"
    if not model.is_file():
        return {"skipped": True}
    bright = _bright_ids(frame_subset[0], science_ids)
    one = _fit_subset(
        model_path=model,
        frame_subset=frame_subset,
        star_ids=bright,
        label="f2b_ped_single",
        use_iterative=False,
    )
    ite = _fit_subset(
        model_path=model,
        frame_subset=frame_subset,
        star_ids=bright,
        label="f2b_ped_iter",
        use_iterative=True,
    )
    out = {
        "single_dao_median": float(pd.to_numeric(one["psf_dao_ratio"], errors="coerce").median()),
        "iter_dao_median": float(pd.to_numeric(ite["psf_dao_ratio"], errors="coerce").median()),
        "chi2_single": float(pd.to_numeric(one["psf_chi2"], errors="coerce").median()),
        "chi2_iter": float(pd.to_numeric(ite["psf_chi2"], errors="coerce").median()),
        "baseline_iter_dao": f2.get("iter_dao_median"),
        "baseline_single_dao": f2.get("single_dao_median"),
        "baseline_chi2_iter": f2.get("chi2_median_iter"),
        "baseline_chi2_single": f2.get("chi2_median_single"),
    }
    (OUT / "f2b_refit_summary.json").write_text(json.dumps(out, indent=2), encoding="ascii")
    return out


def f3_sandbox_remerge() -> dict[str, Any]:
    src_fits = FRAMES / "BO_CVn_Light_001.fits"
    src_proc = FRAMES / "proc_BO_CVn_Light_001.csv"
    dest = OUT / "f3_remerge"
    dest.mkdir(parents=True, exist_ok=True)
    dst_fits = dest / src_fits.name
    dst_proc = dest / src_proc.name
    shutil.copy2(src_fits, dst_fits)
    shutil.copy2(src_proc, dst_proc)
    before = pd.read_csv(dst_proc, low_memory=False)
    st = {
        "_run_epsf": True,
        "epsf_model_path": str(PROD_EPSF),
        "epsf_frame_name": src_fits.name,
        "gain": 0.63707,
        "read_noise": 3.0,
    }
    merge_psf_into_sidecar(
        fits_path=dst_fits,
        sidecar_path=dst_proc,
        st=st,
        target_ids={BO_CVN},
    )
    after = pd.read_csv(dst_proc, low_memory=False)
    assert_inv_psf_additive_01(before, after, frame_name=src_fits.name)
    rec = after.loc[after["catalog_id"].astype(str).map(_norm) == BO_CVN]
    return {
        "has_x_fit": "x_fit" in after.columns,
        "has_y_fit": "y_fit" in after.columns,
        "has_psf_group_n": "psf_group_n" in after.columns,
        "bo_x_fit": float(pd.to_numeric(rec["x_fit"], errors="coerce").iloc[0]) if not rec.empty and "x_fit" in rec else float("nan"),
        "additive_ok": True,
        "sidecar": str(dst_proc.relative_to(REPO)).replace("\\", "/"),
    }


def f1_photometry_if_converged(
    f1: dict[str, Any],
    frame_subset: list[Path],
    science_ids: set[str],
) -> dict[str, Any] | None:
    ok = [a for a in f1.get("attempts") or [] if a.get("converged")]
    if not ok:
        return None
    best = ok[0]
    tag = best["tag"]
    model = OUT / "models" / tag / "masterstar_epsf.fits"
    print("F1 converged photometry", tag, flush=True)
    df = _fit_subset(
        model_path=model,
        frame_subset=frame_subset,
        star_ids=science_ids,
        label=f"f1_{tag}",
        use_iterative=False,
    )
    mag = pd.to_numeric(df["mag"], errors="coerce")
    bright = df.loc[mag <= mag.quantile(0.2)]
    faint = df.loc[mag >= mag.quantile(0.8)]
    return {
        "tag": tag,
        "epsf_fwhm_native_px": best.get("epsf_fwhm_native_px"),
        "input_fwhm_px": INPUT_FWHM,
        "ratio_median": float(pd.to_numeric(df["psf_dao_ratio"], errors="coerce").median()),
        "chi2_bright": float(pd.to_numeric(bright["psf_chi2"], errors="coerce").median()),
        "chi2_faint": float(pd.to_numeric(faint["psf_chi2"], errors="coerce").median()),
        "n_rows": int(len(df)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc)
    hashes_before = snapshot_hashes("before")
    ctrl = OUT / "positive_control.txt"
    ctrl.write_text("before\n", encoding="ascii")
    sha0 = _sha(ctrl)
    ctrl.write_text("after\n", encoding="ascii")
    sha1 = _sha(ctrl)
    if sha0 == sha1:
        raise RuntimeError("positive control failed")

    sci = build_epsf_science_set(PS)
    science_ids = set(sci.catalog_ids)
    subset = pick_frame_subset(list_epsf_science_light_fits(FRAMES), N_FRAMES_SUBSET)
    (OUT / "frame_subset.txt").write_text("\n".join(p.name for p in subset) + "\n", encoding="ascii")

    print("F2b audit", flush=True)
    f2b = f2b_norm_audit()

    print("F3 sandbox remerge", flush=True)
    f3 = f3_sandbox_remerge()
    (OUT / "f3_remerge.json").write_text(json.dumps(f3, indent=2), encoding="ascii")

    print("F2 fitter split", flush=True)
    f2_path = OUT / "f2_summary.json"
    if f2_path.is_file():
        f2 = json.loads(f2_path.read_text(encoding="utf-8"))
        print("F2 resume")
    else:
        f2 = f2_fitter_split(subset, science_ids)

    print("F2b pedestal refit", flush=True)
    f2b_ref = f2b_refit(subset, science_ids, f2)

    cfg = AppConfig(project_root=REPO)
    db = VyvarDatabase(cfg.database_path)
    try:
        f1_path = OUT / "f1_summary.json"
        if f1_path.is_file():
            f1 = json.loads(f1_path.read_text(encoding="utf-8"))
            print("F1 resume")
        else:
            f1 = f1_builds(db)
            f1_path.write_text(json.dumps(f1, indent=2), encoding="ascii")
    finally:
        db.close()

    f1_phot = f1_photometry_if_converged(f1, subset, science_ids)
    if f1_phot:
        (OUT / "f1_photometry.json").write_text(json.dumps(f1_phot, indent=2), encoding="ascii")

    hashes_after = snapshot_hashes("after")
    if hashes_before != hashes_after:
        diff = [k for k in hashes_before if hashes_before.get(k) != hashes_after.get(k)]
        raise RuntimeError(f"production files changed: {diff[:8]}")

    f4_unlocked = bool(
        any(a.get("converged") for a in (f1.get("attempts") or []))
        and abs(float(f2b.get("frac_norm_r_gt_8p5") or 0)) < 0.1
    )
    summary = {
        "elapsed_s": (datetime.now(timezone.utc) - t0).total_seconds(),
        "prod_epsf_sha256": _sha(PROD_EPSF),
        "positive_control_changed": True,
        "production_hashes_identical": True,
        "f1": f1,
        "f1_photometry": f1_phot,
        "f2": f2,
        "f2b": f2b,
        "f2b_refit": f2b_ref,
        "f3": f3,
        "f4_unlocked": f4_unlocked,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="ascii")
    print("SHAPE-01-F done in", summary["elapsed_s"], "s f4_unlocked", f4_unlocked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
