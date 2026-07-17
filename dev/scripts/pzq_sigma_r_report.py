#!/usr/bin/env python3
"""PZQ sigma_r report-only diagnostic (SPARSE-TRUST Part 1 complete)."""

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

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sigma_floor_core import pzq_fit_sigma_r  # noqa: E402

MIN_BINS = 5
BOOTSTRAP_DRAWS = 500


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


def _pzq_uncertainty(mags: np.ndarray, *, bin_sizes: tuple[int, ...] = (2, 4, 8)) -> dict[str, float]:
    """Bootstrap SE on sigma_w and sigma_r from PZQ regression."""
    m = np.asarray(mags, dtype=np.float64)
    ok = np.isfinite(m)
    m = m[ok]
    if m.size < max(bin_sizes) * 2:
        return {"sigma_w_se": float("nan"), "sigma_r_se": float("nan")}
    fit = pzq_fit_sigma_r(m, bin_sizes=bin_sizes)
    sw = float(fit.get("sigma_w", float("nan")))
    sr = float(fit.get("sigma_r", float("nan")))
    rng = np.random.default_rng(424)
    sws: list[float] = []
    srs: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        idx = rng.integers(0, m.size, size=m.size)
        f = pzq_fit_sigma_r(m[idx], bin_sizes=bin_sizes)
        w = float(f.get("sigma_w", float("nan")))
        r = float(f.get("sigma_r", float("nan")))
        if math.isfinite(w):
            sws.append(w)
        if math.isfinite(r):
            srs.append(r)
    sw_se = float(np.std(sws, ddof=1)) if len(sws) >= 10 else float("nan")
    sr_se = float(np.std(srs, ddof=1)) if len(srs) >= 10 else float("nan")
    return {"sigma_w": sw, "sigma_r": sr, "sigma_w_se": sw_se, "sigma_r_se": sr_se}


def _bootstrap_median_ci(vals: list[float], *, n_boot: int = BOOTSTRAP_DRAWS, seed: int = 426) -> dict[str, float]:
    arr = [float(v) for v in vals if math.isfinite(float(v))]
    if len(arr) < 2:
        return {"median": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": len(arr)}
    rng = np.random.default_rng(seed)
    meds: list[float] = []
    a = np.asarray(arr, dtype=np.float64)
    for _ in range(n_boot):
        samp = a[rng.integers(0, a.size, size=a.size)]
        meds.append(float(np.median(samp)))
    lo, hi = np.quantile(meds, [0.16, 0.84])
    return {"median": float(np.median(a)), "ci_lo": float(lo), "ci_hi": float(hi), "n": len(arr)}


def _load_fit_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bins_table(pzq: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for b in pzq.get("bins", []):
        n = int(b.get("N", 0))
        out[f"N{n}"] = {
            "sigma_N": b.get("sigma_N"),
            "sigma_white_expect": b.get("sigma_white_expect"),
            "n_bins": b.get("n_bins"),
            "fit_ok": int(b.get("n_bins", 0) or 0) >= MIN_BINS,
        }
    return out


def _table_from_cohorts(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for c in data.get("cohorts", []):
        rig = str(c.get("rig_label", c.get("setup", "")))
        for star in c.get("pzq_per_star", []):
            pzq = star.get("pzq", {})
            bins = _bins_table(pzq)
            fit_bins = [k for k, v in bins.items() if v.get("fit_ok")]
            rows.append(
                {
                    "rig": rig,
                    "equipment_id": c.get("equipment_id"),
                    "setup": c.get("setup"),
                    "draft_id": c.get("draft_id"),
                    "catalog_id": star.get("catalog_id"),
                    "n_epochs": pzq.get("n_epochs"),
                    "sigma_w": pzq.get("sigma_w"),
                    "sigma_r": pzq.get("sigma_r"),
                    "bins": bins,
                    "fit_bins_used": fit_bins,
                    "pzq_ok": len(fit_bins) >= 2,
                }
            )
    return pd.DataFrame(rows)


def _airmass_range_for_star(draft_id: int, setup: str, catalog_id: str) -> float:
    """Airmass range from anchor LC or proc CSVs for calibrators."""
    meta = _proc_meta_map(draft_id, setup)
    cid = str(catalog_id).strip()
    if cid in meta:
        am = float(meta[cid].get("airmass_range", float("nan")))
        if math.isfinite(am):
            return am
    if draft_id == 424:
        base = _ROOT / "Archive" / "Drafts" / "draft_000424_snapshot_sigma_floor_20260713"
        phot = base / "platesolve" / setup / "photometry" / "lightcurves"
    else:
        base = _ROOT / "Archive" / "Drafts" / f"draft_{draft_id:06d}"
        phot = base / "platesolve" / setup / "photometry" / "lightcurves"
    lc = phot / f"lightcurve_{catalog_id}.csv"
    if not lc.is_file():
        return float("nan")
    try:
        df = pd.read_csv(lc, usecols=["airmass"], low_memory=False)
        am = pd.to_numeric(df["airmass"], errors="coerce")
        if am.notna().sum() < 2:
            return float("nan")
        return float(am.max() - am.min())
    except Exception:  # noqa: BLE001
        return float("nan")


def _colour_offset_map(sem_cause_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not sem_cause_dir.is_dir():
        return out
    for p in sem_cause_dir.glob("setup_*.json"):
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for star in blob.get("stars", []):
            for row in star.get("d1", {}).get("per_comp", []):
                cid = str(row.get("cid", ""))
                co = float(row.get("colour_offset", float("nan")))
                if cid and math.isfinite(co):
                    out[cid] = abs(co)
    return out


def _lag1_map(sem_cause_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not sem_cause_dir.is_dir():
        return out
    for p in sem_cause_dir.glob("setup_*.json"):
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for star in blob.get("stars", []):
            for row in star.get("d1", {}).get("per_comp", []):
                cid = str(row.get("cid", ""))
                lag = float(row.get("lag1", float("nan")))
                if cid and math.isfinite(lag):
                    out[cid] = lag
    return out


def _proc_meta_map(draft_id: int, setup: str) -> dict[str, dict[str, float]]:
    """bp_rp and airmass range from detrended proc CSVs (calibrators lack LC sidecars)."""
    proc = (
        _ROOT
        / "Archive"
        / "Drafts"
        / f"draft_{draft_id:06d}"
        / "detrended_aligned"
        / "lights"
        / setup
    )
    if not proc.is_dir():
        return {}
    bp_rp: dict[str, float] = {}
    am_vals: dict[str, list[float]] = {}
    for p in proc.glob("proc_*.csv"):
        try:
            df = pd.read_csv(p, usecols=["catalog_id", "bp_rp", "airmass"], low_memory=False)
        except (ValueError, OSError):
            continue
        cid_col = df["catalog_id"].astype(str).str.strip()
        br = pd.to_numeric(df["bp_rp"], errors="coerce")
        am = pd.to_numeric(df["airmass"], errors="coerce")
        for i in range(len(df)):
            cid = cid_col.iloc[i]
            if not cid:
                continue
            if cid not in bp_rp:
                v = float(br.iloc[i])
                if math.isfinite(v):
                    bp_rp[cid] = v
            av = float(am.iloc[i])
            if math.isfinite(av):
                am_vals.setdefault(cid, []).append(av)
    out: dict[str, dict[str, float]] = {}
    for cid, v in bp_rp.items():
        ams = am_vals.get(cid, [])
        am_range = float(max(ams) - min(ams)) if len(ams) >= 2 else float("nan")
        out[cid] = {"bp_rp": v, "airmass_range": am_range}
    return out


def _colour_offset_wide424(star_ids: list[str]) -> dict[str, float]:
    """|BP-RP offset from ensemble median| for wide calibrators."""
    meta = _proc_meta_map(424, "NoFilter_60_2")
    if not meta:
        return _colour_offset_wide424_active_targets(star_ids)
    br_vals = [float(v["bp_rp"]) for v in meta.values() if math.isfinite(float(v["bp_rp"]))]
    med = float(np.median(br_vals)) if br_vals else float("nan")
    out: dict[str, float] = {}
    for cid in star_ids:
        m = meta.get(str(cid).strip())
        if not m:
            continue
        v = float(m["bp_rp"])
        if math.isfinite(v) and math.isfinite(med):
            out[str(cid)] = abs(v - med)
    return out


def _colour_offset_wide424_active_targets(star_ids: list[str]) -> dict[str, float]:
    paths = [
        _ROOT / "Archive" / "Drafts" / "draft_000424" / "platesolve" / "NoFilter_60_2" / "photometry" / "active_targets.csv",
        _ROOT / "Archive" / "Drafts" / "draft_000424_snapshot_sigma_floor_20260713" / "platesolve" / "NoFilter_60_2" / "photometry" / "active_targets.csv",
    ]
    frames: list[pd.DataFrame] = []
    for comp_path in paths:
        if comp_path.is_file():
            try:
                frames.append(pd.read_csv(comp_path, dtype={"catalog_id": str}, low_memory=False))
            except Exception:  # noqa: BLE001
                continue
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    if "bp_rp" not in df.columns:
        return {}
    br = pd.to_numeric(df["bp_rp"], errors="coerce")
    med = float(br.median()) if br.notna().any() else float("nan")
    out: dict[str, float] = {}
    for cid in star_ids:
        sub = df[df["catalog_id"].astype(str).str.strip() == str(cid).strip()]
        if sub.empty:
            continue
        v = float(pd.to_numeric(sub.iloc[0].get("bp_rp"), errors="coerce"))
        if math.isfinite(v) and math.isfinite(med):
            out[str(cid)] = abs(v - med)
    return out


def _kpp_probe_per_rig(df: pd.DataFrame, sem_cause_dir: Path) -> dict[str, Any]:
    co_map = _colour_offset_map(sem_cause_dir)
    lag_map = _lag1_map(sem_cause_dir)
    wide_ids = df.loc[df["rig"].astype(str).str.contains("wide", case=False, na=False), "catalog_id"].astype(str).tolist()
    co_map.update(_colour_offset_wide424(wide_ids))
    per_rig: dict[str, Any] = {}
    for rig, grp in df.groupby("rig"):
        xs: list[float] = []
        ys_r: list[float] = []
        ys_lag: list[float] = []
        for _, row in grp.iterrows():
            cid = str(row.get("catalog_id", ""))
            sr = float(pd.to_numeric(row.get("sigma_r"), errors="coerce"))
            if not math.isfinite(sr):
                continue
            co = co_map.get(cid, float("nan"))
            am = _airmass_range_for_star(
                int(row.get("draft_id", 0) or 0),
                str(row.get("setup", "")),
                cid,
            )
            if math.isfinite(co) and math.isfinite(am):
                xs.append(co * am)
                ys_r.append(sr)
            lag = lag_map.get(cid, float("nan"))
            if math.isfinite(lag) and math.isfinite(co) and math.isfinite(am):
                ys_lag.append(lag)
                if len(ys_lag) == len(xs):
                    pass
        lag_xs = xs[: len(ys_lag)] if ys_lag else []
        per_rig[str(rig)] = {
            "sigma_r_vs_abs_bprp_x_airmass_range": _spearman(xs, ys_r),
            "lag1_vs_abs_bprp_x_airmass_range": _spearman(lag_xs, ys_lag) if len(lag_xs) >= 3 else {"rho": float("nan"), "p": float("nan"), "n": len(lag_xs)},
        }
    return per_rig


def _plot_sigma_N_vs_N(df: pd.DataFrame, out_dir: Path) -> list[str]:
    paths: list[str] = []
    for rig, grp in df.groupby("rig"):
        fig, ax = plt.subplots(figsize=(8, 5))
        med_sigma: dict[int, list[float]] = {2: [], 4: [], 8: []}
        med_white: dict[int, list[float]] = {2: [], 4: [], 8: []}
        for _, row in grp.iterrows():
            bins = row.get("bins", {})
            if not isinstance(bins, dict):
                continue
            for n in (2, 4, 8):
                key = f"N{n}"
                b = bins.get(key, {})
                sn = float(pd.to_numeric(b.get("sigma_N", float("nan")), errors="coerce"))
                sw = float(pd.to_numeric(b.get("sigma_white_expect", float("nan")), errors="coerce"))
                if math.isfinite(sn):
                    ax.plot(n, sn, "o", color="steelblue", alpha=0.35, markersize=4)
                    med_sigma[n].append(sn)
                if math.isfinite(sw):
                    med_white[n].append(sw)
        for n in (2, 4, 8):
            if med_sigma[n]:
                ax.plot(n, float(np.median(med_sigma[n])), "s", color="navy", markersize=8, label="median measured" if n == 2 else "")
            if med_white[n]:
                ax.plot(n, float(np.median(med_white[n])), "^", color="darkorange", markersize=7, label="median white expect" if n == 2 else "")
        ax.set_xlabel("bin size N")
        ax.set_ylabel("sigma_N (mag)")
        ax.set_title(f"PZQ sigma_N vs N - {rig}")
        ax.legend(loc="best")
        p = out_dir / f"pzq_sigma_N_vs_N_{rig.replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(str(p))
    return paths


def _power_statement(df: pd.DataFrame) -> dict[str, Any]:
    wide = df[df["rig"].astype(str).str.contains("wide", case=False, na=False)]
    newton = df[df["rig"].astype(str).str.contains("Newton", case=False, na=False)]
    newton_ok = newton[newton["pzq_ok"] == True]  # noqa: E712
    return {
        "primary_rig": "wide_Carl-Zeiss (draft_424, N~139 epochs)",
        "newton_indicative_only": True,
        "newton_points_total": int(len(newton)),
        "newton_points_surviving_5bin_rule": int(len(newton_ok)),
        "newton_survivor_ids": newton_ok["catalog_id"].astype(str).tolist(),
        "statement": (
            "Wide rig is the primary sigma_r result (>=5 bins per N on 12 stars, N=139)."
            if len(newton_ok) == 0
            else f"Wide rig is primary; Newton g/i indicative with {len(newton_ok)}/{len(newton)} stars passing >=5-bin rule."
        ),
    }


def _rig_medians_bootstrap(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rig, grp in df.groupby("rig"):
        sw = pd.to_numeric(grp["sigma_w"], errors="coerce").tolist()
        sr = pd.to_numeric(grp["sigma_r"], errors="coerce").tolist()
        rows.append(
            {
                "rig": str(rig),
                "sigma_w_median_ci": _bootstrap_median_ci(sw),
                "sigma_r_median_ci": _bootstrap_median_ci(sr),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="PZQ sigma_r report (report-only)")
    ap.add_argument("--fit-json", type=Path, default=_ROOT / "tmp" / "sigma_floor" / "sigma_floor_fit.json")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "tmp" / "pzq_sigma_r")
    ap.add_argument("--sem-cause-dir", type=Path, default=_ROOT / "tmp" / "sigma_sem_cause")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.fit_json.is_file():
        print(f"Missing {args.fit_json}", file=sys.stderr)
        return 1
    data = _load_fit_json(args.fit_json)
    df = _table_from_cohorts(data)
    kpp = _kpp_probe_per_rig(df, args.sem_cause_dir)
    power = _power_statement(df)
    rig_medians = _rig_medians_bootstrap(df)
    fig_sigma_n = _plot_sigma_N_vs_N(df, args.out_dir)
    # k'' ROADMAP verdict
    wide_rho = kpp.get("wide_Carl-Zeiss", {}).get("sigma_r_vs_abs_bprp_x_airmass_range", {}).get("rho", float("nan"))
    if math.isfinite(wide_rho) and abs(wide_rho) >= 0.35:
        kpp_verdict = f"k'' priority UP (wide rho={wide_rho:.3f} vs colour x airmass range)."
    elif math.isfinite(wide_rho) and abs(wide_rho) < 0.15:
        kpp_verdict = f"k'' priority DOWN (wide rho={wide_rho:.3f}; weak colour-airmass correlation with sigma_r)."
    else:
        kpp_verdict = f"k'' priority UNCHANGED (wide rho={wide_rho:.3f}; inconclusive)."
    stars_out = []
    for _, row in df.iterrows():
        stars_out.append(
            {
                "rig": row["rig"],
                "catalog_id": row["catalog_id"],
                "n_epochs": row["n_epochs"],
                "sigma_w": row["sigma_w"],
                "sigma_r": row["sigma_r"],
                "bins": row["bins"],
                "fit_bins_used": row["fit_bins_used"],
                "pzq_ok": bool(row["pzq_ok"]),
            }
        )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "source_fit_json": str(args.fit_json),
        "min_bins_rule": MIN_BINS,
        "n_stars": int(len(df)),
        "per_star": stars_out,
        "per_rig_medians_bootstrap": rig_medians,
        "power_statement": power,
        "kpp_probe_per_rig": kpp,
        "kpp_roadmap_verdict": kpp_verdict,
        "figures": fig_sigma_n,
    }
    out_json = args.out_dir / "pzq_sigma_r_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_lines = [
        "# PZQ sigma_r summary",
        "",
        f"Stars: {len(df)} | git: {_git_head()}",
        "",
        "## Power statement",
        power["statement"],
        "",
        "## k'' probe verdict",
        kpp_verdict,
        "",
        "## Per-rig medians (bootstrap CI)",
    ]
    for r in rig_medians:
        sw = r["sigma_w_median_ci"]
        sr = r["sigma_r_median_ci"]
        md_lines.append(
            f"- {r['rig']}: sigma_w={sw['median']:.5f} [{sw['ci_lo']:.5f},{sw['ci_hi']:.5f}] mag; "
            f"sigma_r={sr['median']:.5f} [{sr['ci_lo']:.5f},{sr['ci_hi']:.5f}] mag (n={sw['n']} stars)"
        )
    (args.out_dir / "pzq_sigma_r_summary.md").write_text("\n".join(md_lines) + "\n", encoding="ascii")
    print(f"Wrote {out_json} ({len(df)} stars, {len(fig_sigma_n)} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
