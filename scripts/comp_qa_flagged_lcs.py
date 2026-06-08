#!/usr/bin/env python3
"""Plot LOO differential mags for comp_qa FLAG rows (diagnostic PDF only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from proc_frame_store import list_proc_csvs

ROOT = Path(__file__).resolve().parents[1]


def norm_id(x) -> str:
    s = str(x or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def load_proc_flux(proc_dir: Path, ids: set[str]) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    times = []
    for fp in list_proc_csvs(proc_dir):
        df = pd.read_csv(fp, dtype={"catalog_id": str}, low_memory=False)
        if "catalog_id" not in df.columns:
            continue
        df["catalog_id"] = df["catalog_id"].map(norm_id)
        tcol = next((c for c in ("bjd_tdb_mid", "jd_mid", "hjd_mid") if c in df.columns), None)
        frame = (
            str(df["source_file"].iloc[0])
            if "source_file" in df.columns and len(df)
            else Path(fp).name
        )
        tval = float(pd.to_numeric(df[tcol], errors="coerce").median()) if tcol else float("nan")
        times.append((frame, tval))
        flux_col = "dao_flux" if "dao_flux" in df.columns else "flux"
        sub = df[df["catalog_id"].isin(ids)][["catalog_id", flux_col]]
        for _, r in sub.iterrows():
            rows.append((frame, tval, r["catalog_id"], float(pd.to_numeric(r[flux_col], errors="coerce"))))
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float)
    raw = pd.DataFrame(rows, columns=["frame", "time", "catalog_id", "flux"])
    wide = raw.pivot_table(index="frame", columns="catalog_id", values="flux", aggfunc="first")
    tdf = pd.DataFrame(times, columns=["frame", "time"]).drop_duplicates("frame").set_index("frame")
    wide = wide.reindex(sorted(wide.index, key=lambda f: float(tdf.loc[f, "time"]) if f in tdf.index else 0.0))
    tser = tdf.reindex(wide.index)["time"]
    return wide, tser


def loo_dmag(wide: pd.DataFrame, focus: str, comps: list[str]) -> tuple[np.ndarray, np.ndarray]:
    others = [c for c in comps if c != focus and c in wide.columns]
    if focus not in wide.columns or len(others) < 3:
        return np.array([]), np.array([])
    f = pd.to_numeric(wide[focus], errors="coerce").to_numpy(dtype=float)
    stack = np.vstack(
        [pd.to_numeric(wide[c], errors="coerce").to_numpy(dtype=float) for c in others]
    )
    good_other = np.isfinite(stack) & (stack > 0)
    n_ok = good_other.sum(axis=0)
    ens_flux = np.nansum(np.where(good_other, stack, np.nan), axis=0)
    ok = (
        (n_ok >= 3)
        & np.isfinite(f)
        & (f > 0)
        & np.isfinite(ens_flux)
        & (ens_flux > 0)
    )
    if int(ok.sum()) < 3:
        return np.array([]), np.array([])
    m = -2.5 * np.log10(f[ok] / ens_flux[ok])
    m = m - float(np.median(m))
    return np.arange(len(wide))[ok], m


def vyvar_quality(lc_dir: Path, target_id: str, comp_id: str) -> tuple[str, str]:
    p = lc_dir / f"comp_quality_{target_id}.json"
    if not p.is_file():
        return "—", ""
    raw = json.loads(p.read_text(encoding="utf-8"))
    ent = raw.get(comp_id, raw.get(norm_id(comp_id)))
    if isinstance(ent, str):
        return ent, ""
    if isinstance(ent, dict):
        return str(ent.get("quality", "—")), str(ent.get("note", ""))[:60]
    return "—", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-csv", type=Path, default=ROOT / "xval_out" / "comp_qa_per_comp.csv")
    ap.add_argument("--photometry-dir", type=Path, required=True)
    ap.add_argument("--proc-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "xval_out" / "comp_qa_flagged_lcs.pdf")
    ap.add_argument("--panels-per-page", type=int, default=6)
    args = ap.parse_args()

    qa = pd.read_csv(args.qa_csv, dtype=str)
    qa["FLAG"] = qa["FLAG"].astype(str).str.lower().isin(("true", "1", "yes"))
    flagged = qa[qa["FLAG"]].copy()
    if flagged.empty:
        print("No FLAG rows in", args.qa_csv)
        return 1

    lc_dir = args.photometry_dir / "lightcurves"
    comps_df = pd.read_csv(
        args.photometry_dir / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comps_df["catalog_id"] = comps_df["catalog_id"].map(norm_id)
    comps_df["target_catalog_id"] = comps_df["target_catalog_id"].map(norm_id)

    cache: dict[str, tuple[pd.DataFrame, pd.Series, list[str]]] = {}

    def target_data(tid: str):
        if tid not in cache:
            cl = sorted(comps_df.loc[comps_df["target_catalog_id"] == tid, "catalog_id"].unique().tolist())
            ids = set(cl) | {tid}
            wide, tser = load_proc_flux(args.proc_dir, ids)
            cache[tid] = (wide, tser, cl)
        return cache[tid]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    npp = max(1, int(args.panels_per_page))
    with PdfPages(args.out) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
        axes_flat = axes.flatten()
        ip = 0
        for _, row in flagged.iterrows():
            tid = norm_id(row["target_catalog_id"])
            cid = norm_id(row["catalog_id"])
            wide, tser, clist = target_data(tid)
            idx, dmag = loo_dmag(wide, cid, clist)
            if dmag.size < 3:
                continue
            hours = (
                (tser.iloc[idx].to_numpy(dtype=float) - float(tser.iloc[0])) * 24.0
                if len(tser) > 0
                else np.arange(dmag.size, dtype=float)
            )
            si = float(pd.to_numeric(row.get("sigma_iqr"), errors="coerce"))
            inv = float(pd.to_numeric(row.get("inv_nv"), errors="coerce"))
            spk = float(pd.to_numeric(row.get("spike"), errors="coerce"))
            vq, vnote = vyvar_quality(lc_dir, tid, cid)
            tname = str(row.get("target_vsx_name", tid))
            short = cid[-6:]
            fr = str(row.get("flag_reason", ""))
            title = (
                f"{tname}  comp …{short}  QA:{fr}  "
                f"σ_IQR={si:.4f} invNV={inv:.2f} spike={spk:.2f}  "
                f"VYVAR:{vq}"
            )
            if vnote:
                title += f" ({vnote})"
            ax = axes_flat[ip % npp]
            ax.plot(hours, dmag, ".-", ms=3, lw=0.8, color="#378ADD")
            ax.axhline(0, color="#888", lw=0.6)
            ax.invert_yaxis()
            ax.set_xlabel("hours from start")
            ax.set_ylabel("LOO Δmag")
            kept = str(vq).strip().lower() != "excluded"
            ax.set_title(title, fontsize=7, color="#c0392b" if kept else "#555555")
            ip += 1
            if ip % npp == 0:
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
                axes_flat = axes.flatten()
        if ip % npp != 0:
            for j in range(ip % npp, npp):
                axes_flat[j].axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print("Wrote", args.out, "panels:", int(ip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
