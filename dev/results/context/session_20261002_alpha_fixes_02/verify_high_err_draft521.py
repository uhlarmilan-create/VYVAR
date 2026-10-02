# -*- coding: ascii -*-
"""LC-FLAG-ERR-01 verify on draft 521: reflag LCs; check HAT-148; plot red points."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))

from lc_outlier import (  # noqa: E402
    FLAG_HIGH_ERR,
    EvidenceCache,
    assign_lc_flags,
    export_keep_mask,
    frame_qc_mask_from_night_table,
    load_frame_metrics_from_manifest,
)

OUT = ROOT / "dev" / "results" / "context" / "session_20261002_alpha_fixes_02"
DRAFT = ROOT / "Archive" / "Drafts" / "draft_000521"
SETUP = "NoFilter_60_2"
FRAMES = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOTO = DRAFT / "platesolve" / SETUP / "photometry"
LC_DIR = PHOTO / "lightcurves"
HAT_CID = "1402911282957526912"
# Second LC for screenshot: V1023 Her (known outlier story)
V1023_CID = "1403049512185012992"


def _frame_num(source_file: str) -> str:
    stem = Path(str(source_file)).stem
    if stem.startswith("proc_"):
        stem = stem[5:]
    # Light_052 -> 052
    parts = stem.replace("-", "_").split("_")
    for p in parts:
        if p.isdigit() and len(p) >= 3:
            return p[-3:]
    return stem


def reflag_one(path: Path, frame_qc: dict, frames_dir: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    before = Counter(df["flag"].astype(str).str.strip().str.lower()) if "flag" in df.columns else {}
    mag = pd.to_numeric(df.get("mag_calib", df.get("mag")), errors="coerce").to_numpy(dtype=float)
    err = pd.to_numeric(df.get("err"), errors="coerce").to_numpy(dtype=float)
    bjd = pd.to_numeric(df.get("bjd"), errors="coerce").to_numpy(dtype=float)
    src = df["source_file"].astype(str).tolist() if "source_file" in df.columns else None
    sat = (
        (df["flag"].astype(str).str.lower() == "saturated").to_numpy()
        if "flag" in df.columns
        else None
    )
    ep = pd.to_numeric(df["err_photon"], errors="coerce").to_numpy(dtype=float) if "err_photon" in df.columns else None
    es = pd.to_numeric(df["err_sem_rel"], errors="coerce").to_numpy(dtype=float) if "err_sem_rel" in df.columns else None
    esc = pd.to_numeric(df["err_scint_rel"], errors="coerce").to_numpy(dtype=float) if "err_scint_rel" in df.columns else None
    esys = (
        pd.to_numeric(df["err_sigma_sys_rel"], errors="coerce").to_numpy(dtype=float)
        if "err_sigma_sys_rel" in df.columns
        else None
    )
    cid = path.stem.replace("lightcurve_", "")
    cache = EvidenceCache(frames_dir=frames_dir, catalog_id=cid, n_sigma=5.0)

    def _ev(i: int):
        if src is None:
            return None
        return cache.evidence_at(i, src)

    res = assign_lc_flags(
        mag,
        err,
        bjd,
        sat_flags=sat,
        source_files=src,
        frame_qc_reasons=frame_qc,
        evidence_for_index=_ev,
        high_err_n_sigma=5.0,
        err_photon=ep,
        err_sem_rel=es,
        err_scint_rel=esc,
        err_sigma_sys_rel=esys,
        enabled=True,
    )
    # Never change photometry columns - only flag / flag_reason.
    photo_cols = [
        c
        for c in df.columns
        if c
        not in (
            "flag",
            "flag_reason",
            "outlier",
            "is_outlier",
        )
    ]
    before_hash = {c: pd.util.hash_pandas_object(df[c], index=False).sum() for c in photo_cols}
    df = df.copy()
    df["flag"] = res.flags
    df["flag_reason"] = res.reasons
    after_hash = {c: pd.util.hash_pandas_object(df[c], index=False).sum() for c in photo_cols}
    photo_mismatch = [c for c in photo_cols if before_hash[c] != after_hash[c]]
    df.to_csv(path, index=False)
    after = Counter(res.flags)
    keep = export_keep_mask(res.flags)
    return {
        "catalog_id": cid,
        "before": dict(before),
        "after": dict(after),
        "n_high_err": int(res.n_high_err),
        "n_export_drop": int((~keep).sum()),
        "photo_mismatch_cols": photo_mismatch,
    }


def plot_lc_red(cid: str, title: str, out_png: Path) -> None:
    path = LC_DIR / f"lightcurve_{cid}.csv"
    df = pd.read_csv(path, low_memory=False)
    bjd = pd.to_numeric(df["bjd"], errors="coerce")
    mag = pd.to_numeric(df["mag_calib"], errors="coerce")
    err = pd.to_numeric(df["err"], errors="coerce")
    fl = df["flag"].astype(str).str.strip().str.lower()
    ok = bjd.notna() & mag.notna()
    x0 = float(bjd[ok].median()) if ok.any() else 0.0
    x = bjd - x0
    fig, ax = plt.subplots(figsize=(9, 4.5))
    m_n = ok & (fl == "normal")
    m_f = ok & (fl != "normal")
    ax.errorbar(
        x[m_n],
        mag[m_n],
        yerr=err[m_n],
        fmt="o",
        ms=4,
        color="#2563eb",
        ecolor="#93c5fd",
        elinewidth=0.6,
        label=f"normal (n={int(m_n.sum())})",
    )
    if m_f.any():
        ax.errorbar(
            x[m_f],
            mag[m_f],
            yerr=err[m_f],
            fmt="o",
            ms=6,
            color="#dc2626",
            ecolor="#fca5a5",
            elinewidth=0.8,
            label=f"flagged (n={int(m_f.sum())})",
        )
        for i in df.index[m_f]:
            ax.annotate(
                str(df.at[i, "flag"]),
                (float(x[i]), float(mag[i])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="#991b1b",
            )
    ax.invert_yaxis()
    ax.set_xlabel(f"BJD - {x0:.5f}")
    ax.set_ylabel("mag_calib")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    # Class counts caption
    counts = fl.value_counts().to_dict()
    ax.text(
        0.01,
        0.02,
        "flags: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Before totals (disk as-is)
    before_total: Counter = Counter()
    paths = sorted(LC_DIR.glob("lightcurve_*.csv"))
    for p in paths:
        df = pd.read_csv(p, usecols=lambda c: c == "flag", low_memory=False)
        if "flag" in df.columns:
            before_total.update(df["flag"].astype(str).str.strip().str.lower())

    # Frame QC from draft manifest if present
    frame_qc: dict = {}
    try:
        metrics = load_frame_metrics_from_manifest(DRAFT)
        if metrics is not None and not metrics.empty:
            frame_qc = frame_qc_mask_from_night_table(metrics, n_sigma=5.0)
    except Exception as exc:  # noqa: BLE001
        print("frame_qc skip:", exc)

    after_total: Counter = Counter()
    per_target = []
    photo_bad = 0
    for p in paths:
        r = reflag_one(p, frame_qc, FRAMES)
        after_total.update(r["after"])
        per_target.append(r)
        if r["photo_mismatch_cols"]:
            photo_bad += 1
            print("PHOTO CHANGED", r["catalog_id"], r["photo_mismatch_cols"])

    # HAT-148 check
    hat = pd.read_csv(LC_DIR / f"lightcurve_{HAT_CID}.csv", low_memory=False)
    hat_rows = []
    for _, row in hat.iterrows():
        fn = _frame_num(str(row.get("source_file", "")))
        if fn in ("052", "103"):
            hat_rows.append(
                {
                    "frame": fn,
                    "flag": str(row.get("flag")),
                    "flag_reason": str(row.get("flag_reason", "")),
                    "err": float(pd.to_numeric(row.get("err"), errors="coerce")),
                    "err_photon": float(pd.to_numeric(row.get("err_photon"), errors="coerce")),
                    "mag_calib": float(pd.to_numeric(row.get("mag_calib"), errors="coerce")),
                }
            )

    plot_lc_red(
        HAT_CID,
        f"draft 521 HAT-148-0001021 ({HAT_CID}) - red = non-normal",
        OUT / "lc_red_HAT148.png",
    )
    plot_lc_red(
        V1023_CID,
        f"draft 521 V1023 Her ({V1023_CID}) - red = non-normal",
        OUT / "lc_red_V1023.png",
    )

    n_high = sum(1 for r in per_target if r["n_high_err"] > 0)
    summary = {
        "task": "LC-FLAG-ERR-01",
        "draft": 521,
        "n_targets": len(paths),
        "class_totals_before": dict(before_total),
        "class_totals_after": dict(after_total),
        "n_targets_with_high_err": n_high,
        "n_photo_mismatch_targets": photo_bad,
        "hat148": {
            "catalog_id": HAT_CID,
            "frames_052_103": hat_rows,
            "flag_counts": dict(Counter(hat["flag"].astype(str).str.lower())),
        },
        "plots": [
            str(OUT / "lc_red_HAT148.png"),
            str(OUT / "lc_red_V1023.png"),
        ],
    }
    (OUT / "verify_high_err_draft521.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="ascii"
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
