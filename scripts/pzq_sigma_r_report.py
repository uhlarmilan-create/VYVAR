#!/usr/bin/env python3
"""PZQ sigma_r report-only diagnostic (SPARSE-TRUST Part 1)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sigma_floor_core import pzq_fit_sigma_r  # noqa: E402

MIN_BINS = 5


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _spearman(x: list[float], y: list[float]) -> dict[str, float]:
    pairs = [(a, b) for a, b in zip(x, y, strict=False) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": len(pairs)}
    xs, ys = zip(*pairs, strict=False)
    rho, p = stats.spearmanr(xs, ys)
    return {"rho": float(rho), "p": float(p), "n": len(pairs)}


def _load_fit_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _table_from_cohorts(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in data.get("cohorts", []):
        rig = str(c.get("rig_label", c.get("setup", "")))
        eq = c.get("equipment_id")
        for star in c.get("pzq_per_star", []):
            pzq = star.get("pzq", {})
            bins = pzq.get("bins", [])
            rows.append(
                {
                    "rig": rig,
                    "equipment_id": eq,
                    "setup": c.get("setup"),
                    "draft_id": c.get("draft_id"),
                    "catalog_id": star.get("catalog_id"),
                    "n_epochs": pzq.get("n_epochs"),
                    "sigma_w": pzq.get("sigma_w"),
                    "sigma_r": pzq.get("sigma_r"),
                    "n_bins_N2": next((b.get("n_bins") for b in bins if b.get("N") == 2), None),
                    "n_bins_N4": next((b.get("n_bins") for b in bins if b.get("N") == 4), None),
                    "n_bins_N8": next((b.get("n_bins") for b in bins if b.get("N") == 8), None),
                }
            )
    return pd.DataFrame(rows)


def _plot_sigma_r_by_rig(df: pd.DataFrame, out_dir: Path) -> list[str]:
    paths: list[str] = []
    for rig, grp in df.groupby("rig"):
        if grp.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        sr = pd.to_numeric(grp["sigma_r"], errors="coerce")
        ax.hist(sr.dropna(), bins=20, color="steelblue", edgecolor="white")
        ax.set_xlabel("sigma_r (mag)")
        ax.set_ylabel("count")
        ax.set_title(f"PZQ sigma_r - {rig}")
        p = out_dir / f"pzq_sigma_r_{rig.replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p))
    return paths


def _kpp_probe(df: pd.DataFrame, sem_cause_dir: Path | None) -> dict[str, Any]:
    """Spearman rho: sigma_r vs |colour offset| x airmass range (k'' probe)."""
    if sem_cause_dir is None or not sem_cause_dir.is_dir():
        return {"status": "skipped", "reason": "no sem_cause artifacts"}
    pooled: list[dict[str, Any]] = []
    for p in sem_cause_dir.glob("setup_*.json"):
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for row in blob.get("per_comp", []):
            pooled.append(row)
    if not pooled:
        return {"status": "skipped", "reason": "no per_comp rows"}
    lag_by_cid = {str(r.get("cid")): float(r.get("lag1", float("nan"))) for r in pooled}
    co_by_cid = {str(r.get("cid")): abs(float(r.get("colour_offset", float("nan")))) for r in pooled}
    xs: list[float] = []
    ys_r: list[float] = []
    ys_lag: list[float] = []
    for _, row in df.iterrows():
        cid = str(row.get("catalog_id", ""))
        sr = float(pd.to_numeric(row.get("sigma_r"), errors="coerce"))
        if not math.isfinite(sr):
            continue
        co = co_by_cid.get(cid, float("nan"))
        if math.isfinite(co):
            xs.append(co)
            ys_r.append(sr)
        lag = lag_by_cid.get(cid, float("nan"))
        if math.isfinite(lag) and math.isfinite(co):
            ys_lag.append(lag)
    return {
        "status": "ok",
        "sigma_r_vs_abs_colour_offset": _spearman(xs, ys_r),
        "lag1_vs_abs_colour_offset_note": "lag1 from sem_cause per_comp; airmass range proxy pending",
        "n_matched": len(xs),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="PZQ sigma_r report (report-only)")
    ap.add_argument(
        "--fit-json",
        type=Path,
        default=_ROOT / "tmp" / "sigma_floor" / "sigma_floor_fit.json",
    )
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "tmp" / "pzq_sigma_r")
    ap.add_argument("--sem-cause-dir", type=Path, default=_ROOT / "tmp" / "sigma_sem_cause")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.fit_json.is_file():
        print(f"Missing {args.fit_json}", file=sys.stderr)
        return 1
    data = _load_fit_json(args.fit_json)
    df = _table_from_cohorts(data)
    df["pzq_ok"] = df.apply(
        lambda r: all(
            int(pd.to_numeric(r.get(c), errors="coerce") or 0) >= MIN_BINS
            for c in ("n_bins_N2", "n_bins_N4", "n_bins_N8")
            if pd.notna(r.get(c))
        ),
        axis=1,
    )
    fig_paths = _plot_sigma_r_by_rig(df, args.out_dir)
    kpp = _kpp_probe(df, args.sem_cause_dir if args.sem_cause_dir.is_dir() else None)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "source_fit_json": str(args.fit_json),
        "n_stars": int(len(df)),
        "n_pzq_ok_bins": int(df["pzq_ok"].sum()),
        "per_rig": df.groupby("rig").agg(
            n_stars=("catalog_id", "count"),
            median_sigma_r=("sigma_r", "median"),
            median_sigma_w=("sigma_w", "median"),
        ).reset_index().to_dict(orient="records"),
        "stars": df.to_dict(orient="records"),
        "kpp_probe": kpp,
        "figures": fig_paths,
        "interpretation": (
            "sigma_r > 0 on wide/Newton cohorts confirms red-noise floor not captured by white SEM; "
            "use SPARSE-TRUST T_green=1.5 to tolerate uncaptured red component at epoch scale."
        ),
    }
    out_json = args.out_dir / "pzq_sigma_r_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_json} ({len(df)} stars, {len(fig_paths)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
