# -*- coding: ascii -*-
"""FRAME-QC-PARITY-02 Part B: n_stars distribution + frame-29 residuals.

MEASUREMENT ONLY. Live 516/517 read-only. No production writes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\ASTRO\python\VYVAR")
OUT = ROOT / "dev" / "results" / "context" / "session_20260907_frameqc2"
SETUP = "NoFilter_60_2"
LIVE_SHA_PREFIX = {
    "csv": "bfa24039",
    "fits": "13e77cf8",
    "epsf": "172f9540",
}
BO_CVN_CID = "1498613634033133184"
K_VALUES = (1.5, 2.5, 3.5, 5.0)
SIGMA_MAD_SCALE = 1.4826


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def g4_live_516() -> dict:
    ps = ROOT / "Archive" / "Drafts" / "draft_000516" / "platesolve" / SETUP
    out = {
        "csv": sha256_file(ps / "masterstars_full_match.csv"),
        "fits": sha256_file(ps / "MASTERSTAR.fits"),
        "epsf": sha256_file(ps / "masterstar_epsf.fits"),
    }
    out["verdict"] = (
        out["csv"].startswith(LIVE_SHA_PREFIX["csv"])
        and out["fits"].startswith(LIVE_SHA_PREFIX["fits"])
        and out["epsf"].startswith(LIVE_SHA_PREFIX["epsf"])
    )
    return out


def _frame_id(path_s: str) -> str:
    m = re.search(r"Light_(\d+)", str(path_s).replace("\\", "/"))
    return f"Light_{m.group(1)}" if m else Path(str(path_s)).stem


def _load_qc(draft_id: int) -> pd.DataFrame:
    p = (
        ROOT
        / "Archive"
        / "Drafts"
        / f"draft_{draft_id:06d}"
        / "calibrated"
        / "lights"
        / "qc_metrics.csv"
    )
    df = pd.read_csv(p, low_memory=False)
    df["n_stars"] = pd.to_numeric(df["n_stars_detected"], errors="coerce")
    df["frame_id"] = df["src"].map(_frame_id)
    df["residual_flatness_p99"] = pd.to_numeric(
        df["residual_flatness_p99_adu"], errors="coerce"
    )
    df["draft_id"] = int(draft_id)
    return df


def _stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            "n": 0,
            "median": None,
            "mad": None,
            "sigma_mad": None,
            "min": None,
            "max": None,
        }
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return {
        "n": int(a.size),
        "median": med,
        "mad": mad,
        "sigma_mad": float(mad * SIGMA_MAD_SCALE),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def _bounds_table(df: pd.DataFrame, subset: str) -> list[dict]:
    a = df["n_stars"].to_numpy(dtype=float)
    st = _stats(a)
    rows = []
    med = st["median"]
    sig = st["sigma_mad"]
    for k in K_VALUES:
        if med is None or sig is None or not math.isfinite(sig):
            rows.append(
                {
                    "subset": subset,
                    "k": k,
                    "n_low": None,
                    "n_high": None,
                    "low_frames": [],
                    "high_frames": [],
                }
            )
            continue
        lo = med - float(k) * float(sig)
        hi = med + float(k) * float(sig)
        low = df[df["n_stars"] < lo]
        high = df[df["n_stars"] > hi]
        rows.append(
            {
                "subset": subset,
                "k": float(k),
                "lo": lo,
                "hi": hi,
                "n_low": int(len(low)),
                "n_high": int(len(high)),
                "low_frames": sorted(low["frame_id"].tolist()),
                "high_frames": sorted(high["frame_id"].tolist()),
            }
        )
    return rows


def n_stars_block(draft_id: int) -> dict:
    df = _load_qc(draft_id)
    ok = df[df["status"].astype(str) == "ok"].copy()
    all_rows = df.copy()
    frame29 = df[df["frame_id"] == "Light_029"]
    out = {
        "draft_id": draft_id,
        "n_all": int(len(all_rows)),
        "n_ok": int(len(ok)),
        "stats_all": _stats(all_rows["n_stars"].to_numpy()),
        "stats_ok": _stats(ok["n_stars"].to_numpy()),
        "bounds_all": _bounds_table(all_rows, "all"),
        "bounds_ok": _bounds_table(ok, "ok"),
        "frame29": None
        if frame29.empty
        else {
            "n_stars": float(frame29["n_stars"].iloc[0]),
            "status": str(frame29["status"].iloc[0]),
            "residual_flatness_p99": float(frame29["residual_flatness_p99"].iloc[0]),
        },
    }
    return out, df, ok


def _robust_z(value: float, others: np.ndarray) -> float:
    o = np.asarray(others, dtype=float)
    o = o[np.isfinite(o)]
    if o.size < 2:
        return float("nan")
    med = float(np.median(o))
    mad = float(np.median(np.abs(o - med)))
    sig = mad * SIGMA_MAD_SCALE
    if not math.isfinite(sig) or sig == 0.0:
        return float("nan")
    return float((value - med) / sig)


def _percentile_rank(value: float, pop: np.ndarray) -> float:
    p = np.asarray(pop, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0 or not math.isfinite(value):
        return float("nan")
    return float(100.0 * (np.sum(p <= value) / float(p.size)))


def residual_block() -> dict:
    lc_path = (
        ROOT
        / "Archive"
        / "Drafts"
        / "draft_000516"
        / "platesolve"
        / SETUP
        / "photometry"
        / "lightcurves"
        / f"lightcurve_{BO_CVN_CID}.csv"
    )
    chk_path = lc_path.with_name(f"check_kmag_{BO_CVN_CID}.csv")
    lc = pd.read_csv(lc_path, low_memory=False)
    chk = pd.read_csv(chk_path, low_memory=False)
    lc["frame_id"] = lc["source_file"].map(_frame_id)
    chk["frame_id"] = chk["source_file"].map(_frame_id)
    mag = pd.to_numeric(chk["kmag"], errors="coerce")
    finite = mag.notna()
    chk = chk.loc[finite].copy()
    mag = mag.loc[finite]
    mu = float(mag.median())
    resid = mag - mu
    chk["resid"] = resid
    # 134 normal epochs: LC flag==normal if present, else all sidecar rows.
    if "flag" in lc.columns:
        normal_frames = set(
            lc.loc[lc["flag"].astype(str) == "normal", "frame_id"].tolist()
        )
        use = chk[chk["frame_id"].isin(normal_frames)].copy()
    else:
        use = chk.copy()
    n = int(len(use))
    row29 = use[use["frame_id"] == "Light_029"]
    if row29.empty:
        z29 = None
        r29 = None
        pct29 = None
    else:
        r29 = float(row29["resid"].iloc[0])
        others = use.loc[use["frame_id"] != "Light_029", "resid"].to_numpy()
        z29 = _robust_z(r29, others)
        pct29 = _percentile_rank(r29, use["resid"].to_numpy())
    high_z = []
    resid_arr = use["resid"].to_numpy()
    for _, r in use.iterrows():
        others = use.loc[use["frame_id"] != r["frame_id"], "resid"].to_numpy()
        z = _robust_z(float(r["resid"]), others)
        if math.isfinite(z) and abs(z) >= 3.0:
            high_z.append(
                {
                    "frame_id": str(r["frame_id"]),
                    "resid": float(r["resid"]),
                    "z": float(z),
                }
            )
    qc = _load_qc(516)
    ok = qc[qc["status"].astype(str) == "ok"]
    p99 = ok["residual_flatness_p99"].to_numpy(dtype=float)
    p99_st = _stats(p99)
    f29 = ok[ok["frame_id"] == "Light_029"]
    p99_29 = float(f29["residual_flatness_p99"].iloc[0]) if not f29.empty else float("nan")
    others_p99 = ok.loc[ok["frame_id"] != "Light_029", "residual_flatness_p99"].to_numpy()
    z_p99 = _robust_z(p99_29, others_p99)
    return {
        "target": "BO CVn",
        "target_cid": BO_CVN_CID,
        "check_cid": str(chk["check_catalog_id"].iloc[0]),
        "n_epochs_used": n,
        "kmag_median": mu,
        "frame29": {
            "residual": r29,
            "robust_z_vs_other_133": z29,
            "percentile_rank_signed": pct29,
        },
        "other_frames_abs_z_ge_3": high_z,
        "residual_flatness_p99_ok": p99_st,
        "residual_flatness_p99_frame29": p99_29,
        "residual_flatness_p99_frame29_z": z_p99,
        "sidecar_has_per_comp_epoch_residuals": False,
        "lc_path": str(lc_path),
        "check_path": str(chk_path),
        "reading": None,
    }


def _reading(z: float | None) -> str:
    if z is None or not math.isfinite(float(z)):
        return "R-B3 (between / unreadable z)"
    az = abs(float(z))
    if az < 2.0:
        return "R-B1 (contained)"
    if az >= 3.0:
        return "R-B2 (damage)"
    return "R-B3 (between)"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    g4 = g4_live_516()
    blocks = {}
    nstar_rows = []
    bound_rows = []
    for did in (516, 517):
        blk, df, ok = n_stars_block(did)
        blocks[str(did)] = blk
        for subset, subdf in (("all", df), ("ok", ok)):
            st = _stats(subdf["n_stars"].to_numpy())
            nstar_rows.append(
                {
                    "draft_id": did,
                    "subset": subset,
                    **st,
                    "frame29_n_stars": blk["frame29"]["n_stars"]
                    if blk["frame29"]
                    else None,
                    "frame29_status": blk["frame29"]["status"]
                    if blk["frame29"]
                    else None,
                }
            )
            for b in blk[f"bounds_{subset}"]:
                bound_rows.append(
                    {
                        "draft_id": did,
                        "subset": subset,
                        "k": b["k"],
                        "n_low": b["n_low"],
                        "n_high": b["n_high"],
                        "low_frames": "|".join(b.get("low_frames") or []),
                        "high_frames": "|".join(b.get("high_frames") or []),
                    }
                )
    resid = residual_block()
    resid["reading"] = _reading(
        (resid.get("frame29") or {}).get("robust_z_vs_other_133")
    )
    pd.DataFrame(nstar_rows).to_csv(OUT / "n_stars_stats.csv", index=False)
    pd.DataFrame(bound_rows).to_csv(OUT / "n_stars_k_bounds.csv", index=False)
    resid_rows = [
        {
            "star": "check",
            "catalog_id": resid["check_cid"],
            "frame_id": "Light_029",
            "residual": resid["frame29"]["residual"],
            "robust_z": resid["frame29"]["robust_z_vs_other_133"],
            "percentile_rank_signed": resid["frame29"]["percentile_rank_signed"],
        }
    ]
    for h in resid["other_frames_abs_z_ge_3"]:
        resid_rows.append(
            {
                "star": "check",
                "catalog_id": resid["check_cid"],
                "frame_id": h["frame_id"],
                "residual": h["resid"],
                "robust_z": h["z"],
                "percentile_rank_signed": None,
            }
        )
    pd.DataFrame(resid_rows).to_csv(OUT / "check_residual_frame29.csv", index=False)
    summary = {
        "g4_live_516": g4,
        "n_stars": {k: {kk: vv for kk, vv in v.items() if kk != "bounds_all" and kk != "bounds_ok"} | {
            "bounds_all": v["bounds_all"],
            "bounds_ok": v["bounds_ok"],
        } for k, v in blocks.items()},
        "residuals": resid,
        "stop": (
            "STOP for Milan: decide (i) diagnostic vs drop, (ii) k, "
            "(iii) recut timing. Nothing wired."
        ),
    }
    # Compact n_stars without huge nested dumps for readability.
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="ascii"
    )
    print(json.dumps({
        "g4": g4["verdict"],
        "516_ok_n_stars_median": blocks["516"]["stats_ok"]["median"],
        "516_ok_n_stars_max": blocks["516"]["stats_ok"]["max"],
        "frame29_n_stars": blocks["516"]["frame29"],
        "z29": resid["frame29"]["robust_z_vs_other_133"],
        "reading": resid["reading"],
        "n_high_z": len(resid["other_frames_abs_z_ge_3"]),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
