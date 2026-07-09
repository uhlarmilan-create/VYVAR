#!/usr/bin/env python3
"""Reduced chi-squared/dof harness for sigma-budget diagnostics (sandbox)."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from photometry_core import _sky_pp_for_photometric_error
from sigma_budget import (
    SIGMA_VARIANT_HOWELL_ONLY,
    SIGMA_VARIANT_HOWELL_SCINT_FULL,
    SIGMA_VARIANT_HOWELL_SCINT_FRESID,
    SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR,
    combine_sigma_mag_quadrature,
    relative_flux_err_to_mag_sigma,
    resolve_rig_scintillation_params,
    total_sigma,
)

CHI2_DOF_LO = 0.8
CHI2_DOF_HI = 1.2
_MAG_ERR_SCALE = 2.5 / math.log(10.0)


@dataclass
class Chi2StarResult:
    catalog_id: str
    variant: str
    chi2: float
    dof: int
    chi2_dof: float
    n_frames: int
    baseline_hours: float
    mag_g: float | None
    chi2_dof_ci_lo: float | None
    chi2_dof_ci_hi: float | None
    f_resid: float | None = None
    sigma_floor_mag: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JointFitResult:
    f_resid: float
    sigma_floor_mag: float
    median_chi2_dof: float
    chi2_dof_iqr: float
    f_resid_ci_lo: float | None
    f_resid_ci_hi: float | None
    sigma_floor_ci_lo: float | None
    sigma_floor_ci_hi: float | None
    f_resid_pinned_edge: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reduced_chi2_constant(
    mags: np.ndarray,
    sigmas: np.ndarray,
    *,
    mag_ref: float | None = None,
) -> tuple[float, int, float, float]:
    m = np.asarray(mags, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(s) & (s > 0)
    m = m[ok]
    s = s[ok]
    n = int(m.size)
    if n < 3:
        return float("nan"), max(0, n - 1), float("nan"), float("nan")
    if mag_ref is None or not math.isfinite(float(mag_ref)):
        w = 1.0 / (s * s)
        wsum = float(np.sum(w))
        mag_ref = float(np.sum(w * m) / wsum) if math.isfinite(wsum) and wsum > 0 else float(np.median(m))
    else:
        mag_ref = float(mag_ref)
    resid = m - mag_ref
    chi2 = float(np.sum((resid / s) ** 2))
    dof = n - 1
    chi2_dof = chi2 / dof if dof > 0 else float("nan")
    return chi2, dof, chi2_dof, mag_ref


def bootstrap_chi2_dof_ci(
    mags: np.ndarray,
    sigmas: np.ndarray,
    *,
    n_boot: int = 400,
    seed: int = 0,
    alpha: float = 0.16,
) -> tuple[float | None, float | None]:
    m = np.asarray(mags, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(s) & (s > 0)
    m = m[ok]
    s = s[ok]
    n = int(m.size)
    if n < 5 or n_boot <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        _, _, c2d, _ = reduced_chi2_constant(m[idx], s[idx])
        if math.isfinite(c2d):
            vals.append(c2d)
    if len(vals) < 10:
        return None, None
    arr = np.sort(np.asarray(vals, dtype=float))
    return float(np.quantile(arr, alpha)), float(np.quantile(arr, 1.0 - alpha))


def baseline_hours_from_bjd(bjd: np.ndarray) -> float:
    b = np.asarray(bjd, dtype=np.float64)
    b = b[np.isfinite(b)]
    if b.size < 2:
        return 0.0
    return float((float(np.max(b)) - float(np.min(b))) * 24.0)


def evaluate_lc_chi2_variants(
    mags: np.ndarray,
    sigma_variants: dict[str, np.ndarray],
    *,
    catalog_id: str,
    mag_g: float | None,
    bjd: np.ndarray,
    f_resid_map: dict[str, float] | None = None,
    sigma_floor_map: dict[str, float] | None = None,
) -> list[Chi2StarResult]:
    out: list[Chi2StarResult] = []
    bh = baseline_hours_from_bjd(bjd)
    for variant, sig in sigma_variants.items():
        chi2, dof, chi2_dof, _ = reduced_chi2_constant(mags, sig)
        lo, hi = bootstrap_chi2_dof_ci(mags, sig)
        out.append(
            Chi2StarResult(
                catalog_id=str(catalog_id),
                variant=variant,
                chi2=chi2,
                dof=dof,
                chi2_dof=chi2_dof,
                n_frames=int(np.sum(np.isfinite(mags) & np.isfinite(sig) & (sig > 0))),
                baseline_hours=bh,
                mag_g=mag_g,
                chi2_dof_ci_lo=lo,
                chi2_dof_ci_hi=hi,
                f_resid=(f_resid_map or {}).get(variant),
                sigma_floor_mag=(sigma_floor_map or {}).get(variant),
            )
        )
    return out


def _ensemble_median_chi2(
    calibrator_results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    f_resid: float,
    sigma_floor_mag: float = 0.0,
) -> float | None:
    c2ds: list[float] = []
    for mags, sh, ss in calibrator_results:
        m = np.asarray(mags, dtype=np.float64)
        shv = np.asarray(sh, dtype=np.float64)
        ssv = np.asarray(ss, dtype=np.float64)
        ok = np.isfinite(m) & np.isfinite(shv) & np.isfinite(ssv)
        if int(ok.sum()) < 3:
            continue
        sig = np.array(
            [
                combine_sigma_mag_quadrature(shv[i], f_resid * ssv[i], sigma_floor_mag=sigma_floor_mag)
                for i in np.where(ok)[0]
            ],
            dtype=np.float64,
        )
        _, _, c2d, _ = reduced_chi2_constant(m[ok], sig)
        if math.isfinite(c2d):
            c2ds.append(c2d)
    return float(np.median(c2ds)) if c2ds else None


def _ensemble_chi2_spread(
    calibrator_results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    f_resid: float,
    sigma_floor_mag: float = 0.0,
) -> float:
    c2ds: list[float] = []
    for mags, sh, ss in calibrator_results:
        m = np.asarray(mags, dtype=np.float64)
        shv = np.asarray(sh, dtype=np.float64)
        ssv = np.asarray(ss, dtype=np.float64)
        ok = np.isfinite(m) & np.isfinite(shv) & np.isfinite(ssv)
        if int(ok.sum()) < 3:
            continue
        sig = np.array(
            [
                combine_sigma_mag_quadrature(shv[i], f_resid * ssv[i], sigma_floor_mag=sigma_floor_mag)
                for i in np.where(ok)[0]
            ],
            dtype=np.float64,
        )
        _, _, c2d, _ = reduced_chi2_constant(m[ok], sig)
        if math.isfinite(c2d):
            c2ds.append(c2d)
    return (
        float(np.subtract(*np.percentile(c2ds, [75, 25])))
        if len(c2ds) >= 2
        else float("nan")
    )


def _fit_f_floor_grid(
    calibrator_results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    fit_floor: bool,
) -> tuple[float, float, float, float]:
    f_grid = np.linspace(0.0, 1.0, 51)
    floor_grid = np.linspace(0.0, 0.02, 41) if fit_floor else np.array([0.0])
    best_f, best_floor = 0.0, 0.0
    best_dist = float("inf")
    best_median = float("nan")
    for f in f_grid:
        for fl in floor_grid:
            med = _ensemble_median_chi2(calibrator_results, f_resid=float(f), sigma_floor_mag=float(fl))
            if med is None:
                continue
            dist = abs(med - 1.0)
            if dist < best_dist:
                best_dist = dist
                best_f = float(f)
                best_floor = float(fl)
                best_median = med
    spread = _ensemble_chi2_spread(
        calibrator_results, f_resid=best_f, sigma_floor_mag=best_floor,
    )
    return best_f, best_floor, best_median, spread


def fit_f_resid_ensemble(
    calibrator_results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, float, float]:
    f, _, med, spread = _fit_f_floor_grid(calibrator_results, fit_floor=False)
    return f, med, spread


def fit_f_resid_sigma_floor_ensemble(
    calibrator_results: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    n_boot: int = 400,
    seed: int = 0,
    alpha: float = 0.16,
) -> JointFitResult:
    f, floor, med, spread = _fit_f_floor_grid(calibrator_results, fit_floor=True)
    pinned: str | None = None
    if abs(f) < 1e-9:
        pinned = "lower"
    elif abs(f - 1.0) < 1e-9:
        pinned = "upper"
    f_boot: list[float] = []
    fl_boot: list[float] = []
    n_cal = len(calibrator_results)
    if n_cal >= 3 and n_boot > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = rng.integers(0, n_cal, size=n_cal)
            sample = [calibrator_results[int(i)] for i in idx]
            bf, bfl, _, _ = _fit_f_floor_grid(sample, fit_floor=True)
            f_boot.append(bf)
            fl_boot.append(bfl)
    f_lo = f_hi = fl_lo = fl_hi = None
    if len(f_boot) >= 10:
        f_arr = np.sort(np.asarray(f_boot, dtype=float))
        fl_arr = np.sort(np.asarray(fl_boot, dtype=float))
        f_lo = float(np.quantile(f_arr, alpha))
        f_hi = float(np.quantile(f_arr, 1.0 - alpha))
        fl_lo = float(np.quantile(fl_arr, alpha))
        fl_hi = float(np.quantile(fl_arr, 1.0 - alpha))
    return JointFitResult(
        f_resid=f,
        sigma_floor_mag=floor,
        median_chi2_dof=med,
        chi2_dof_iqr=spread,
        f_resid_ci_lo=f_lo,
        f_resid_ci_hi=f_hi,
        sigma_floor_ci_lo=fl_lo,
        sigma_floor_ci_hi=fl_hi,
        f_resid_pinned_edge=pinned,
    )


def plot_chi2_vs_g(
    results: list[Chi2StarResult],
    out_path: Path,
    *,
    title: str = "",
) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {
        SIGMA_VARIANT_HOWELL_ONLY: "#4C78A8",
        SIGMA_VARIANT_HOWELL_SCINT_FULL: "#E45756",
        SIGMA_VARIANT_HOWELL_SCINT_FRESID: "#72B7B2",
        SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR: "#54A24B",
    }
    for v in sorted({r.variant for r in results}):
        sub = [r for r in results if r.variant == v and r.mag_g is not None and math.isfinite(r.chi2_dof)]
        if not sub:
            continue
        ax.scatter(
            [float(r.mag_g) for r in sub],
            [float(r.chi2_dof) for r in sub],
            label=v,
            alpha=0.85,
            color=colors.get(v),
        )
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.axhspan(CHI2_DOF_LO, CHI2_DOF_HI, color="gray", alpha=0.12)
    ax.set_xlabel("Gaia G")
    ax.set_ylabel("reduced chi2/dof")
    ax.set_title(title or "chi2/dof vs G")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


def write_summary_json(payload: dict[str, Any], out_path: Path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(out_path)


def load_proc_row_for_source(proc_dir: Path, source_file: str, catalog_id: str) -> pd.Series | None:
    from gaia_catalog_id import normalize_gaia_source_id

    path = proc_dir / str(source_file).strip()
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, low_memory=False, dtype={"catalog_id": str})
    except Exception:  # noqa: BLE001
        return None
    cid = str(normalize_gaia_source_id(catalog_id) or "").strip()
    id_col = "catalog_id" if "catalog_id" in df.columns else "name"
    df["_nid"] = df[id_col].map(lambda x: str(normalize_gaia_source_id(x) or "").strip())
    sub = df.loc[df["_nid"] == cid]
    return None if sub.empty else sub.iloc[0]


def sigma_arrays_from_lc_and_proc(
    lc_df: pd.DataFrame,
    proc_dir: Path,
    catalog_id: str,
    *,
    rig_params: Any,
    f_resid: float = 0.0,
    sigma_floor_mag: float = 0.0,
    gain: float = 1.0,
    read_noise: float = 10.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    mags = pd.to_numeric(lc_df.get("delta_mag"), errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(mags).any() and "mag_calib" in lc_df.columns:
        mags = pd.to_numeric(lc_df["mag_calib"], errors="coerce").to_numpy(dtype=np.float64)
    n = len(lc_df)
    sh = np.full(n, np.nan)
    ss = np.full(n, np.nan)
    variants = {
        SIGMA_VARIANT_HOWELL_ONLY: np.full(n, np.nan),
        SIGMA_VARIANT_HOWELL_SCINT_FULL: np.full(n, np.nan),
        SIGMA_VARIANT_HOWELL_SCINT_FRESID: np.full(n, np.nan),
        SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR: np.full(n, np.nan),
    }
    for i, sf in enumerate(lc_df.get("source_file", pd.Series([""] * n)).astype(str).tolist()):
        row = load_proc_row_for_source(proc_dir, sf, catalog_id)
        am = float(pd.to_numeric(lc_df.iloc[i].get("airmass"), errors="coerce"))
        if not math.isfinite(am) or am < 1.0:
            am = 1.0
        if row is None:
            err = float(pd.to_numeric(lc_df.iloc[i].get("err"), errors="coerce"))
            if math.isfinite(err) and err > 0:
                variants[SIGMA_VARIANT_HOWELL_ONLY][i] = _MAG_ERR_SCALE * err
            continue
        flux = float(pd.to_numeric(row.get("dao_flux"), errors="coerce"))
        if not math.isfinite(flux) or flux <= 0:
            continue
        sky = float(_sky_pp_for_photometric_error(row))
        area = float(pd.to_numeric(row.get("aperture_area_px"), errors="coerce"))
        if not math.isfinite(area) or area <= 0:
            r = float(pd.to_numeric(row.get("aperture_r_px"), errors="coerce"))
            area = math.pi * r * r if math.isfinite(r) and r > 0 else float("nan")
        sig_h, _, _ = total_sigma(
            flux, sky, area, gain=gain, read_noise=read_noise,
            telescope_diameter_m=rig_params.telescope_diameter_m, airmass=am,
            exposure_s=rig_params.exposure_s, altitude_m=rig_params.altitude_m,
            c_y=rig_params.c_y, variant=SIGMA_VARIANT_HOWELL_ONLY,
        )
        sig_t_full, _, sig_s = total_sigma(
            flux, sky, area, gain=gain, read_noise=read_noise,
            telescope_diameter_m=rig_params.telescope_diameter_m, airmass=am,
            exposure_s=rig_params.exposure_s, altitude_m=rig_params.altitude_m,
            c_y=rig_params.c_y, variant=SIGMA_VARIANT_HOWELL_SCINT_FULL,
        )
        sig_t_fr, _, _ = total_sigma(
            flux, sky, area, gain=gain, read_noise=read_noise,
            telescope_diameter_m=rig_params.telescope_diameter_m, airmass=am,
            exposure_s=rig_params.exposure_s, altitude_m=rig_params.altitude_m,
            c_y=rig_params.c_y, f_resid=f_resid, variant=SIGMA_VARIANT_HOWELL_SCINT_FRESID,
        )
        sh[i] = relative_flux_err_to_mag_sigma(sig_h)
        ss[i] = relative_flux_err_to_mag_sigma(sig_s)
        variants[SIGMA_VARIANT_HOWELL_ONLY][i] = sh[i]
        variants[SIGMA_VARIANT_HOWELL_SCINT_FULL][i] = relative_flux_err_to_mag_sigma(sig_t_full)
        variants[SIGMA_VARIANT_HOWELL_SCINT_FRESID][i] = relative_flux_err_to_mag_sigma(sig_t_fr)
        variants[SIGMA_VARIANT_HOWELL_SCINT_FRESID_FLOOR][i] = combine_sigma_mag_quadrature(
            sh[i], f_resid * ss[i], sigma_floor_mag=sigma_floor_mag,
        )
    return mags, variants, sh, ss


def saturation_margin_distribution(
    lc_df: pd.DataFrame,
    proc_dir: Path,
    catalog_id: str,
) -> dict[str, Any]:
    """Per-frame peak ADU fill fraction vs pipeline saturate_limit_adu_85pct."""
    fills: list[float] = []
    peaks: list[float] = []
    limits: list[float] = []
    for sf in lc_df.get("source_file", pd.Series(dtype=str)).astype(str).tolist():
        row = load_proc_row_for_source(proc_dir, sf, catalog_id)
        if row is None:
            continue
        peak = float(pd.to_numeric(row.get("peak_max_adu"), errors="coerce"))
        limit = float(pd.to_numeric(row.get("saturate_limit_adu_85pct"), errors="coerce"))
        if not math.isfinite(limit) or limit <= 0:
            limit = float(pd.to_numeric(row.get("saturate_limit_adu"), errors="coerce"))
        if math.isfinite(peak) and math.isfinite(limit) and limit > 0:
            fills.append(peak / limit)
            peaks.append(peak)
            limits.append(limit)
    if not fills:
        return {"n_frames": 0, "fill_p50": None, "fill_p95": None, "fill_max": None}
    arr = np.asarray(fills, dtype=float)
    return {
        "n_frames": int(len(fills)),
        "fill_p50": float(np.quantile(arr, 0.5)),
        "fill_p95": float(np.quantile(arr, 0.95)),
        "fill_max": float(np.max(arr)),
        "peak_max_adu_max": float(np.max(peaks)),
        "saturate_limit_adu_85pct_median": float(np.median(limits)),
        "likely_saturated_frames": int(np.sum(arr >= 1.0)),
    }


def main() -> None:
    import argparse
    import sys

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    ap = argparse.ArgumentParser(description="Chi2/dof sigma-budget gate on a target LC CSV")
    ap.add_argument("--lc", type=Path, required=True, help="Target lightcurve CSV")
    ap.add_argument("--proc-dir", type=Path, required=True, help="Directory of proc_*.csv frames")
    ap.add_argument("--catalog-id", required=True, help="Gaia source id for sigma lookup")
    ap.add_argument("--draft-id", type=int, default=None)
    ap.add_argument("--setup", default="")
    ap.add_argument("--f-resid", type=float, default=0.0)
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/sigma_budget"))
    args = ap.parse_args()

    lc_df = pd.read_csv(args.lc, low_memory=False)
    meta_path = args.proc_dir.parent / "pipeline_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    rig = resolve_rig_scintillation_params(
        draft_id=args.draft_id, setup=args.setup or args.proc_dir.parent.name,
        pipeline_meta=meta,
    )
    mags, variants, _, _ = sigma_arrays_from_lc_and_proc(
        lc_df, args.proc_dir, args.catalog_id, rig_params=rig, f_resid=args.f_resid,
    )
    bjd = pd.to_numeric(lc_df.get("bjd"), errors="coerce").to_numpy(dtype=np.float64)
    results = evaluate_lc_chi2_variants(
        mags, variants, catalog_id=args.catalog_id, mag_g=None, bjd=bjd,
        f_resid_map={SIGMA_VARIANT_HOWELL_SCINT_FRESID: args.f_resid},
    )
    out_dir = Path(args.out_dir)
    payload = {
        "catalog_id": args.catalog_id,
        "rig": rig.to_dict(),
        "results": [r.to_dict() for r in results],
    }
    json_path = write_summary_json(payload, out_dir / f"chi2_gate_{args.catalog_id}.json")
    plot_path = plot_chi2_vs_g(
        results, out_dir / f"chi2_vs_g_{args.catalog_id}.png",
        title=f"chi2/dof variants ({args.catalog_id})",
    )
    print(json_path)
    print(plot_path)


if __name__ == "__main__":
    main()
