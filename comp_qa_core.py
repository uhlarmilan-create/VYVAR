"""Comp-star LOO QA (Sokolovsky indices + magnitude locus) — shared core for pipeline and CLI."""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gaia_catalog_id import norm_id_or_empty as _normalize_id
from proc_frame_store import list_proc_csvs

LOGGER = logging.getLogger(__name__)

_MAD_SCALE = 0.6745
_IQR_NORM = 1.349
_MAG_BIN = 0.5
_INVNV_FLOOR = 1.0
_SPIKE_HARD = 3.0
_CAL_MAG_COLS = ("mag_calib", "comp_mag_calib", "lc_median_mag", "vyvar_calibrated_mag")


def mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return mad / _MAD_SCALE if mad > 0 else float("nan")


def robust_thr(vals: list[float], k: float = 4.0) -> float:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("inf")
    sig = mad_sigma(arr)
    if not math.isfinite(sig) or sig <= 0:
        return float("inf")
    return float(np.median(arr) + k * sig)


def flux_to_mag(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    out = np.full_like(f, np.nan)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def inst_mag_from_flux(flux: np.ndarray) -> float:
    f = np.asarray(flux, dtype=float)
    f = f[np.isfinite(f) & (f > 0)]
    if f.size == 0:
        return float("nan")
    return float(-2.5 * np.log10(float(np.median(f))))


def comp_axis_mag(flux: np.ndarray, row: pd.Series | None = None) -> float:
    """VYVAR calibrated comp magnitude when present, else median instrumental from flux."""
    if row is not None:
        for col in _CAL_MAG_COLS:
            if col not in row.index:
                continue
            v = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(v) and math.isfinite(float(v)):
                return float(v)
    return inst_mag_from_flux(flux)


def loo_diff_series(
    mag: dict[str, np.ndarray],
    focus_id: str,
    comp_ids: list[str],
) -> np.ndarray:
    others = [c for c in comp_ids if c != focus_id]
    if not others:
        return np.full(0, np.nan)
    n = len(next(iter(mag.values())))
    m_focus = mag.get(focus_id)
    if m_focus is None or len(m_focus) != n:
        return np.full(n, np.nan)
    stack = np.vstack([mag[c] for c in others if c in mag])
    flux = np.nansum(10.0 ** (-0.4 * stack), axis=0)
    ens = np.full(n, np.nan)
    ok = np.isfinite(flux) & (flux > 0)
    ens[ok] = -2.5 * np.log10(flux[ok])
    diff = m_focus - ens
    use = np.isfinite(diff) & np.isfinite(m_focus) & np.isfinite(ens)
    out = np.full(n, np.nan)
    out[use] = diff[use] - np.nanmedian(diff[use])
    return out


def sokolovsky_indices(m: np.ndarray) -> dict[str, float]:
    m = np.asarray(m, dtype=float)
    m = m[np.isfinite(m)]
    n = int(m.size)
    if n < 3:
        return {
            "sigma_iqr": float("nan"),
            "inv_nv": float("nan"),
            "spike": float("nan"),
            "n": n,
        }
    q25, q75 = np.percentile(m, [25, 75])
    sigma_iqr = float((q75 - q25) / _IQR_NORM)
    mbar = float(np.mean(m))
    s2 = float(np.sum((m - mbar) ** 2) / (n - 1))
    dm = np.diff(m)
    d2 = float(np.sum(dm**2) / (n - 1)) if dm.size else float("nan")
    inv_nv = float(s2 / d2) if math.isfinite(d2) and d2 > 0 else float("nan")
    std_m = float(np.std(m, ddof=1)) if n > 1 else float("nan")
    spike = (
        float(std_m / sigma_iqr)
        if math.isfinite(sigma_iqr) and sigma_iqr > 0 and math.isfinite(std_m)
        else float("nan")
    )
    return {"sigma_iqr": sigma_iqr, "inv_nv": inv_nv, "spike": spike, "n": n}


def build_locus(
    mags: np.ndarray,
    sigmas: np.ndarray,
    bin_width: float = _MAG_BIN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ok = np.isfinite(mags) & np.isfinite(sigmas)
    mags = mags[ok]
    sigmas = sigmas[ok]
    if mags.size < 3:
        return np.array([]), np.array([]), np.array([])

    lo = float(np.floor(np.min(mags) / bin_width) * bin_width)
    hi = float(np.ceil(np.max(mags) / bin_width) * bin_width)
    edges = np.arange(lo, hi + bin_width, bin_width)
    if edges.size < 2:
        edges = np.array([lo, lo + bin_width])

    centers, locs, spreads = [], [], []
    for i in range(len(edges) - 1):
        m = (mags >= edges[i]) & (mags < edges[i + 1])
        if i == len(edges) - 2:
            m = (mags >= edges[i]) & (mags <= edges[i + 1])
        if int(m.sum()) < 2:
            continue
        s = sigmas[m]
        med = float(np.median(s))
        mad = float(np.median(np.abs(s - med)))
        spread = mad / _MAD_SCALE if mad > 0 else float("nan")
        centers.append(float((edges[i] + edges[i + 1]) / 2.0))
        locs.append(med)
        spreads.append(spread if math.isfinite(spread) and spread > 0 else med * 0.1 + 1e-4)

    if not centers:
        gmed = float(np.median(sigmas))
        gm = float(np.median(np.abs(sigmas - gmed)) / _MAD_SCALE)
        return np.array([float(np.median(mags))]), np.array([gmed]), np.array([max(gm, 1e-4)])

    return np.asarray(centers), np.asarray(locs), np.asarray(spreads)


def locus_at(
    mag: float,
    centers: np.ndarray,
    locs: np.ndarray,
    spreads: np.ndarray,
) -> tuple[float, float]:
    if not math.isfinite(mag) or centers.size == 0:
        return float("nan"), float("nan")
    loc = float(np.interp(mag, centers, locs, left=locs[0], right=locs[-1]))
    spr = float(np.interp(mag, centers, spreads, left=spreads[0], right=spreads[-1]))
    return loc, spr


def flag_reasons(
    sigma_iqr: float,
    inv_nv: float,
    spike: float,
    inst_mag: float,
    locus_centers: np.ndarray,
    locus_med: np.ndarray,
    locus_spread: np.ndarray,
    thr_inv_nv: float,
    mad_k: float = 4.0,
) -> list[str]:
    flags: list[str] = []
    loc, spr = locus_at(inst_mag, locus_centers, locus_med, locus_spread)
    if math.isfinite(sigma_iqr) and math.isfinite(loc) and math.isfinite(spr):
        if sigma_iqr > loc + mad_k * spr:
            flags.append("amplitude")
    if math.isfinite(inv_nv) and math.isfinite(thr_inv_nv):
        if inv_nv > thr_inv_nv and inv_nv > _INVNV_FLOOR:
            flags.append("invNV")
    if math.isfinite(spike) and spike > _SPIKE_HARD:
        flags.append("spike")
    return flags


def worst_flagged_score(
    metrics: dict[str, float],
    flags: list[str],
    inst_mag: float,
    locus_centers: np.ndarray,
    locus_med: np.ndarray,
    locus_spread: np.ndarray,
    thr_inv_nv: float,
    mad_k: float = 4.0,
) -> float:
    score = 0.0
    loc, spr = locus_at(inst_mag, locus_centers, locus_med, locus_spread)
    if "amplitude" in flags and math.isfinite(spr) and spr > 0:
        thr_a = loc + mad_k * spr
        if math.isfinite(metrics["sigma_iqr"]) and thr_a > 0:
            score = max(score, metrics["sigma_iqr"] / thr_a)
    if "invNV" in flags and math.isfinite(thr_inv_nv) and thr_inv_nv > 0:
        score = max(score, metrics["inv_nv"] / thr_inv_nv)
    if "spike" in flags:
        score = max(score, metrics["spike"] / _SPIKE_HARD)
    return score


def load_proc_pivot(proc_dir: Path, ids: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = list_proc_csvs(proc_dir)
    rows = []
    times = []
    for fp in files:
        df = pd.read_csv(fp, dtype={"catalog_id": str}, usecols=lambda c: c in {
            "catalog_id", "dao_flux", "flux", "bjd_tdb_mid", "jd_mid", "hjd_mid", "source_file",
        } or c == "catalog_id")
        if "catalog_id" not in df.columns:
            continue
        df["catalog_id"] = df["catalog_id"].astype(str).str.strip()
        sub = df[df["catalog_id"].isin(ids)]
        if sub.empty:
            continue
        flux_col = "dao_flux" if "dao_flux" in sub.columns else "flux"
        tcol = next((c for c in ("bjd_tdb_mid", "jd_mid", "hjd_mid") if c in sub.columns), None)
        frame = sub["source_file"].iloc[0] if "source_file" in sub.columns else os.path.basename(fp)
        tval = float(pd.to_numeric(sub[tcol], errors="coerce").median()) if tcol else float("nan")
        times.append((frame, tval))
        for _, r in sub.iterrows():
            rows.append((frame, r["catalog_id"], float(pd.to_numeric(r[flux_col], errors="coerce"))))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    flux_df = pd.DataFrame(rows, columns=["frame", "catalog_id", "flux"])
    flux_w = flux_df.pivot_table(index="frame", columns="catalog_id", values="flux", aggfunc="first")
    time_df = pd.DataFrame(times, columns=["frame", "time"]).drop_duplicates("frame").set_index("frame")
    flux_w = flux_w.sort_index(key=lambda idx: time_df.reindex(idx)["time"].values)
    return flux_w, time_df


def compute_comp_qa(
    *,
    photometry_dir: Path,
    proc_dir: Path,
    mad_k: float = 4.0,
    min_comps: int = 3,
    max_comps: int = 8,
    _target_processing_order: list[str] | None = None,
) -> dict[str, Any]:
    """Run draft-wide comp QA; read-only w.r.t. photometry products.

    Returns dict with keys: per_target, per_comp_rows, stats, comp_csv_path.
    """
    phot = Path(photometry_dir).expanduser()
    proc_dir = Path(proc_dir).expanduser()
    comp_path = phot / "comparison_stars_per_target.csv"
    if not comp_path.is_file():
        raise FileNotFoundError(f"missing {comp_path}")

    comps = pd.read_csv(comp_path, dtype={"catalog_id": str, "target_catalog_id": str})
    comps["catalog_id"] = comps["catalog_id"].astype(str).str.strip()
    comps["target_catalog_id"] = comps["target_catalog_id"].astype(str).str.strip()
    comps["_catalog_id_n"] = comps["catalog_id"].map(_normalize_id)
    comps["_target_n"] = comps["target_catalog_id"].map(_normalize_id)

    target_data: dict[str, dict] = {}
    for tid, grp in comps.groupby("_target_n"):
        if not tid:
            continue
        comp_ids = sorted(grp["_catalog_id_n"].unique().tolist())
        comp_ids = [c for c in comp_ids if c]
        if len(comp_ids) < min_comps:
            continue
        all_ids = set(comp_ids) | {tid}
        flux_w, _time_df = load_proc_pivot(proc_dir, all_ids)
        if flux_w.empty or tid not in flux_w.columns:
            continue
        mag = {cid: flux_to_mag(flux_w[cid].values.astype(float)) for cid in flux_w.columns}
        flux_raw = {cid: flux_w[cid].values.astype(float) for cid in flux_w.columns}
        comp_rows = {row["_catalog_id_n"]: row for _, row in grp.iterrows()}
        pool = [c for c in comp_ids if c in mag]
        if len(pool) < min_comps:
            continue
        target_data[tid] = {
            "grp": grp,
            "pool": pool,
            "mag": mag,
            "flux_raw": flux_raw,
            "comp_rows": comp_rows,
        }

    pass1: list[tuple[str, str, float, float]] = []
    for tid, td in target_data.items():
        for cid in td["pool"]:
            m = loo_diff_series(td["mag"], cid, td["pool"])
            idx = sokolovsky_indices(m)
            row = td["comp_rows"].get(cid)
            imag = comp_axis_mag(td["flux_raw"].get(cid, np.array([])), row)
            pass1.append((tid, cid, imag, idx["sigma_iqr"]))

    if pass1:
        mags1 = np.array([p[2] for p in pass1])
        sig1 = np.array([p[3] for p in pass1])
        lc_c, lc_m, lc_s = build_locus(mags1, sig1)
    else:
        lc_c = lc_m = lc_s = np.array([])

    per_comp_rows: list[dict[str, Any]] = []
    per_target: dict[str, dict[str, Any]] = {}
    n_flag_total = 0
    n_flag_amp = n_flag_inv = n_flag_spike = 0
    n_flag_amp_inv = 0
    dropped_global: set[tuple[str, str]] = set()

    if _target_processing_order is not None:
        _extra = [t for t in target_data if t not in _target_processing_order]
        _target_items = [
            (t, target_data[t])
            for t in list(_target_processing_order) + _extra
            if t in target_data
        ]
    else:
        _target_items = list(target_data.items())

    for tid, td in _target_items:
        grp = td["grp"]
        surviving = list(td["pool"])

        while len(surviving) >= min_comps:
            metrics = {}
            for cid in surviving:
                m = loo_diff_series(td["mag"], cid, surviving)
                metrics[cid] = sokolovsky_indices(m)
                row = td["comp_rows"].get(cid)
                metrics[cid]["inst_mag"] = comp_axis_mag(td["flux_raw"].get(cid, np.array([])), row)

            thr_inv = robust_thr([metrics[c]["inv_nv"] for c in surviving], mad_k)
            flagged: dict[str, list[str]] = {}
            for cid in surviving:
                fr = flag_reasons(
                    metrics[cid]["sigma_iqr"],
                    metrics[cid]["inv_nv"],
                    metrics[cid]["spike"],
                    metrics[cid]["inst_mag"],
                    lc_c,
                    lc_m,
                    lc_s,
                    thr_inv,
                    mad_k,
                )
                if fr:
                    flagged[cid] = fr
            if not flagged:
                break
            worst_id = max(
                flagged,
                key=lambda cid: worst_flagged_score(
                    metrics[cid],
                    flagged[cid],
                    metrics[cid]["inst_mag"],
                    lc_c,
                    lc_m,
                    lc_s,
                    thr_inv,
                    mad_k,
                ),
            )
            dropped_global.add((tid, worst_id))
            surviving = [c for c in surviving if c != worst_id]

        surv_final = [c for c in td["pool"] if (tid, c) not in dropped_global]
        if len(surv_final) < min_comps:
            surv_final = list(td["pool"])

        thr_inv_f = robust_thr(
            [
                sokolovsky_indices(loo_diff_series(td["mag"], c, surv_final))["inv_nv"]
                for c in surv_final
            ],
            mad_k,
        )

        n_flag_t = 0
        comp_payload: dict[str, dict[str, Any]] = {}
        for cid in td["pool"]:
            peers = surv_final
            m = loo_diff_series(td["mag"], cid, peers)
            idx = sokolovsky_indices(m)
            row = td["comp_rows"].get(cid)
            imag = comp_axis_mag(td["flux_raw"].get(cid, np.array([])), row)
            loc, spr = locus_at(imag, lc_c, lc_m, lc_s)
            flags = flag_reasons(
                idx["sigma_iqr"],
                idx["inv_nv"],
                idx["spike"],
                imag,
                lc_c,
                lc_m,
                lc_s,
                thr_inv_f,
                mad_k,
            )
            flagged = len(flags) > 0
            reason = "+".join(flags) if flagged else ""
            if flagged:
                n_flag_total += 1
                n_flag_t += 1
                if flags == ["amplitude"]:
                    n_flag_amp += 1
                elif flags == ["invNV"]:
                    n_flag_inv += 1
                elif flags == ["spike"]:
                    n_flag_spike += 1
                elif set(flags) == {"amplitude", "invNV"}:
                    n_flag_amp_inv += 1

            comp_payload[cid] = {
                "sigma_iqr": idx["sigma_iqr"],
                "inv_nv": idx["inv_nv"],
                "spike": idx["spike"],
                "qa_flag": flagged,
                "reason": reason,
                "inst_mag": imag,
                "locus_sigma_iqr": loc,
                "locus_spread": spr,
                "n_frames": idx["n"],
            }
            row_df = grp[grp["_catalog_id_n"] == cid]
            r0 = row_df.iloc[0] if not row_df.empty else {}
            per_comp_rows.append({
                "target_catalog_id": tid,
                "target_vsx_name": str(r0.get("target_vsx_name", "")),
                "catalog_id": cid,
                "inst_mag": imag,
                "sigma_iqr": idx["sigma_iqr"],
                "inv_nv": idx["inv_nv"],
                "spike": idx["spike"],
                "locus_sigma_iqr": loc,
                "locus_spread": spr,
                "n_frames": idx["n"],
                "thr_inv_nv": thr_inv_f,
                "FLAG": flagged,
                "flag_reason": reason,
            })

        n_clean = len(td["pool"]) - n_flag_t
        per_target[tid] = {
            "target_catalog_id": tid,
            "n_comps": len(td["pool"]),
            "n_flagged": n_flag_t,
            "n_clean": n_clean,
            "comps": comp_payload,
        }

    tgt_df = pd.DataFrame(
        [
            {
                "target_catalog_id": v["target_catalog_id"],
                "n_comps": v["n_comps"],
                "n_flagged": v["n_flagged"],
                "n_clean": v["n_clean"],
            }
            for v in per_target.values()
        ]
    )
    mn = max(1, int(min_comps))
    mx = max(mn, int(max_comps))
    strong = min(mn + 2, mx)
    n_ge_strong = int((tgt_df["n_clean"] >= strong).sum()) if not tgt_df.empty else 0
    n_thin = (
        int(((tgt_df["n_clean"] >= mn) & (tgt_df["n_clean"] < strong)).sum())
        if not tgt_df.empty
        else 0
    )
    n_lt_min = int((tgt_df["n_clean"] < mn).sum()) if not tgt_df.empty else 0

    return {
        "per_target": per_target,
        "per_comp_rows": per_comp_rows,
        "stats": {
            "n_flagged": n_flag_total,
            "n_flag_amp": n_flag_amp,
            "n_flag_inv": n_flag_inv,
            "n_flag_spike": n_flag_spike,
            "n_flag_amp_inv": n_flag_amp_inv,
            "n_clean_ge_strong": n_ge_strong,
            "n_clean_thin": n_thin,
            "n_clean_lt_min": n_lt_min,
            "min_comps": mn,
            "strong_comps": strong,
        },
        "comp_csv_path": comp_path,
    }


def write_comp_qa_artifacts(
    result: dict[str, Any],
    *,
    photometry_dir: Path,
    lc_dir: Path | None = None,
    update_summary: bool = True,
) -> list[Path]:
    """Write ``lightcurves/comp_qa_{target}.json`` and optional ``n_clean`` on summary."""
    phot = Path(photometry_dir)
    lc = Path(lc_dir) if lc_dir is not None else phot / "lightcurves"
    lc.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for tid, tinfo in result.get("per_target", {}).items():
        out_path = lc / f"comp_qa_{tid}.json"
        payload = {
            "target_catalog_id": tid,
            "n_clean": int(tinfo["n_clean"]),
            "n_comps": int(tinfo["n_comps"]),
            "n_flagged": int(tinfo["n_flagged"]),
            "comps": {
                cid: {
                    "sigma_iqr": c["sigma_iqr"],
                    "inv_nv": c["inv_nv"],
                    "spike": c["spike"],
                    "qa_flag": bool(c["qa_flag"]),
                    "reason": c["reason"],
                }
                for cid, c in tinfo.get("comps", {}).items()
            },
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_path)

    if update_summary:
        summ_path = phot / "photometry_summary.csv"
        if summ_path.is_file():
            df = pd.read_csv(summ_path, dtype={"catalog_id": str}, low_memory=False)
            if "catalog_id" in df.columns:
                n_clean_map = {
                    _normalize_id(tid): int(tinfo["n_clean"])
                    for tid, tinfo in result.get("per_target", {}).items()
                }
                df["n_clean"] = df["catalog_id"].map(lambda x: n_clean_map.get(_normalize_id(x), np.nan))
                df.to_csv(summ_path, index=False)
                written.append(summ_path)

    return written


def run_comp_qa_for_photometry_dir(
    *,
    photometry_dir: Path,
    proc_dir: Path,
    lc_dir: Path | None = None,
    mad_k: float = 4.0,
    min_comps: int = 3,
    max_comps: int = 8,
    update_summary: bool = True,
) -> dict[str, Any]:
    """Pipeline entry: compute comp QA and write draft-tree artifacts."""
    result = compute_comp_qa(
        photometry_dir=photometry_dir,
        proc_dir=proc_dir,
        mad_k=mad_k,
        min_comps=min_comps,
        max_comps=max_comps,
    )
    paths = write_comp_qa_artifacts(
        result,
        photometry_dir=photometry_dir,
        lc_dir=lc_dir,
        update_summary=update_summary,
    )
    st = result.get("stats", {})
    LOGGER.info(
        "[COMP_QA] flagged=%s (amp=%s invNV=%s spike=%s amp+invNV=%s) "
        "n_clean ge%d=%s thin=%s lt%d=%s → %d JSON files",
        st.get("n_flagged"),
        st.get("n_flag_amp"),
        st.get("n_flag_inv"),
        st.get("n_flag_spike"),
        st.get("n_flag_amp_inv"),
        st.get("strong_comps"),
        st.get("n_clean_ge_strong"),
        st.get("n_clean_thin"),
        st.get("min_comps"),
        st.get("n_clean_lt_min"),
        len(paths),
    )
    result["written_paths"] = [str(p) for p in paths]
    return result
