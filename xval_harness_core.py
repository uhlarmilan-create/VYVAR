"""Shared helpers for the offline cross-validation harness (``xval_run.py``).

Confidence thresholds and LOO differential utilities — not used by the production pipeline.
"""
from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Validated confidence thresholds (draft_000365 whole-night harness).
_R_SEP_CONFIRMED_HI = 1.40
_R_SEP_LOW = 0.70
_R_DAO_INDEP_LO = 0.71
_R_DAO_INDEP_HI = 1.56
_MIN_COMPS = 3


def sclip_std(x: np.ndarray, sig: float = 3.0, it: int = 5) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    for _ in range(it):
        if x.size < 5:
            break
        m, s = float(np.median(x)), float(np.std(x))
        k = np.abs(x - m) <= sig * s
        if k.all():
            break
        x = x[k]
    return float(np.std(x)) if x.size else float("nan")


def find_frames(aligned_root: Path) -> tuple[Path | None, list[Path]]:
    pats = ("*.fit", "*.fits", "*.fts", "*.FIT", "*.FITS", "*.FTS")
    files = sorted({Path(x) for p in pats for x in glob.glob(str(aligned_root / "**" / p), recursive=True)})
    master = next((f for f in files if "MASTER" in f.name.upper()), None)
    lights = [f for f in files if "MASTER" not in f.name.upper()]
    return master, lights


def estimate_fwhm(data: np.ndarray, xs: np.ndarray, ys: np.ndarray, box: int = 11) -> float:
    half = box // 2
    ny, nx = data.shape
    out: list[float] = []
    for x, y in zip(xs, ys, strict=True):
        xi, yi = int(round(x)), int(round(y))
        if xi - half < 0 or yi - half < 0 or xi + half >= nx or yi + half >= ny:
            continue
        s = data[yi - half : yi + half + 1, xi - half : xi + half + 1].astype(float)
        s = s - np.median(s)
        s[s < 0] = 0
        tot = s.sum()
        if tot <= 0:
            continue
        yy, xx = np.mgrid[0 : s.shape[0], 0 : s.shape[1]]
        cx, cy = (s * xx).sum() / tot, (s * yy).sum() / tot
        sx2 = (s * (xx - cx) ** 2).sum() / tot
        sy2 = (s * (yy - cy) ** 2).sum() / tot
        if sx2 > 0 and sy2 > 0:
            out.append(2.3548 * math.sqrt(math.sqrt(sx2 * sy2)))
    return float(np.median(out)) if out else float("nan")


def diff_series(w: pd.DataFrame, sid: str, comps: list[str]) -> np.ndarray:
    """Unweighted LOO differential mag (same as ``xval_run.py``)."""
    comps = [c for c in comps if c in w.columns]
    if sid not in w.columns or not comps:
        return np.array([])
    f = w[sid].values.astype(float)
    stack = w[comps].values.astype(float)
    good = np.isfinite(stack) & (stack > 0)
    es = np.nansum(np.where(good, stack, np.nan), axis=1)
    val = (good.sum(axis=1) == len(comps)) & np.isfinite(f) & (f > 0) & (es > 0)
    md = np.full(len(f), np.nan)
    md[val] = -2.5 * np.log10(f[val] / es[val])
    out = md - np.nanmedian(md)
    return out


def comp_loo_median(w: pd.DataFrame, comps: list[str]) -> float:
    comps = [c for c in comps if c in w.columns]
    if len(comps) < 3:
        return float("nan")
    vals = [
        sclip_std(diff_series(w, c, [x for x in comps if x != c]))
        for c in comps
    ]
    return float(np.nanmedian(vals))


def assign_sep_confidence(
    vyvar_lc_rms: float,
    target_rms_sep: float,
    target_rms_dao: float | None = None,
    *,
    n_comp: int = 8,
) -> str:
    """Map SEP vs VYVAR RMS metrics to validated confidence label (harness only)."""
    if not math.isfinite(vyvar_lc_rms):
        return "no_vyvar_rms"
    if not math.isfinite(target_rms_sep):
        if (
            target_rms_dao is not None
            and math.isfinite(target_rms_dao)
            and vyvar_lc_rms > 0
        ):
            r_d = target_rms_dao / vyvar_lc_rms
            if _R_DAO_INDEP_LO < r_d <= _R_DAO_INDEP_HI:
                return "vyvar_ok_indep_failed"
        return "no_independent"

    r_sep = target_rms_sep / vyvar_lc_rms
    r_dao = (
        target_rms_dao / vyvar_lc_rms
        if target_rms_dao is not None and math.isfinite(target_rms_dao)
        else float("nan")
    )

    if r_sep <= _R_SEP_LOW:
        if math.isfinite(r_dao) and r_dao > _R_DAO_INDEP_HI:
            return "review"
        if math.isfinite(r_dao) and r_dao > _R_DAO_INDEP_LO:
            return "vyvar_ok_indep_failed"
        if n_comp > 7.5:
            return "review"
        return "no_independent"

    if r_sep <= _R_SEP_CONFIRMED_HI:
        return "confirmed"
    return "vyvar_ok_indep_failed"


def query_gaia(ra0: float, dec0: float, radius_deg: float, gmax: float) -> pd.DataFrame:
    """Gaia DR3 cone query (same SQL as ``xval_run.py``)."""
    from astroquery.gaia import Gaia  # noqa: PLC0415

    Gaia.ROW_LIMIT = 300000
    q = (
        f"SELECT source_id,ra,dec,phot_g_mean_mag,bp_rp FROM gaiadr3.gaia_source "
        f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra0},{dec0},{radius_deg})) "
        f"AND phot_g_mean_mag IS NOT NULL AND phot_g_mean_mag < {gmax}"
    )
    df = Gaia.launch_job_async(q).get_results().to_pandas()
    df.columns = [str(c).lower() for c in df.columns]
    pick = lambda cs: next((c for c in cs if c in df.columns), None)
    cm = {
        "source_id": pick(["source_id", "dr3_source_id"]),
        "ra": pick(["ra"]),
        "dec": pick(["dec", "de"]),
        "phot_g_mean_mag": pick(["phot_g_mean_mag"]),
        "bp_rp": pick(["bp_rp"]),
    }
    out = df[[cm[k] for k in cm]].copy()
    out.columns = list(cm)
    out["source_id"] = out["source_id"].astype("int64").astype(str)
    return out
