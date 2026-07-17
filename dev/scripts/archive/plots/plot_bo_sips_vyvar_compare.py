#!/usr/bin/env python3
"""BO CVn: SIPS V−C vs VYVAR delta_mag / mag_inst / mag_calib (same time axis).

Usage:
  python scripts/plot_bo_sips_vyvar_compare.py
  python scripts/plot_bo_sips_vyvar_compare.py --draft draft_000311
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000311")
SETUP = "NoFilter_60_2"
BO_CID = "1498613634033133184"
JD0 = 2461154.0

SIPS_FILE = "SIPS_BO_CVn_3.7px_2026-04-23.txt"


def demean(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return y - np.nanmedian(y)


def rms(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    return float(np.std(y)) if y.size > 1 else float("nan")


def load_sips(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t_list: list[float] = []
    v_list: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            t_list.append(float(parts[0]) - JD0)
            v_list.append(float(parts[1]))
        except ValueError:
            continue
    return np.asarray(t_list), np.asarray(v_list)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, default=DRAFT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    draft = args.draft.resolve()
    phot = draft / "platesolve" / SETUP / "photometry"
    lc_path = phot / "lightcurves" / f"lightcurve_{BO_CID}.csv"
    sips_path = draft / "detrended_aligned" / "lights" / SIPS_FILE
    out_dir = args.out or (phot / "_lc_compare_bo_sips")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not lc_path.is_file():
        raise SystemExit(f"Missing LC: {lc_path}")
    if not sips_path.is_file():
        raise SystemExit(f"Missing SIPS: {sips_path}")

    lc = pd.read_csv(lc_path).sort_values("bjd")
    t = lc["bjd"].to_numpy(dtype=float) - JD0
    am = lc["airmass"].to_numpy(dtype=float)
    inst = demean(lc["mag_inst"].to_numpy(dtype=float))
    cal = demean(lc["mag_calib"].to_numpy(dtype=float))
    dm = demean(lc["delta_mag"].to_numpy(dtype=float))

    sips_t, sips_vc = load_sips(sips_path)
    sips_dm = demean(sips_vc)

    i_am_min = int(np.nanargmin(am))
    sl2 = slice(i_am_min, None)

    def corr_am(y: np.ndarray, sl: slice = slice(None)) -> float:
        a = am[sl]
        v = y[sl]
        m = np.isfinite(a) & np.isfinite(v)
        if m.sum() < 3:
            return float("nan")
        return float(np.corrcoef(a[m], v[m])[0, 1])

    ap_px = float("nan")
    for cand in (
        phot / "photometry_summary_before_aperture_7.0px.csv",
        phot / "photometry_summary.csv",
    ):
        if not cand.is_file():
            continue
        summ = pd.read_csv(cand, dtype={"catalog_id": str})
        row = summ[summ["catalog_id"] == BO_CID]
        if not row.empty and pd.notna(row["aperture_px"].iloc[0]):
            ap_px = float(row["aperture_px"].iloc[0])
            if "before_aperture" in cand.name:
                break
    if not math.isfinite(ap_px):
        ap_px = float(np.nanmedian(lc["aperture_r_px"])) if "aperture_r_px" in lc.columns else float("nan")
    ap_note = f"VYVAR SNR apertura BO ≈ {ap_px:.2f} px"

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(
        f"BO CVn — SIPS vs VYVAR (draft {draft.name}, {SETUP})\n"
        f"{ap_note} | červené pásmo = 2. polovica noci (airmass rastie)",
        fontsize=11,
    )

    panels = [
        ("SIPS V−C (demean)", sips_dm, sips_t, "#9467bd"),
        ("VYVAR delta_mag — AIJ-like rel. (demean)", dm, t, "#2ca02c"),
        ("VYVAR mag_inst — bez ZP (demean)", inst, t, "#d62728"),
        ("VYVAR mag_calib — ZP, bez AM detrend (demean)", cal, t, "#ff7f0e"),
    ]

    stats: list[str] = []
    for ax, (title, y, x, color) in zip(axes, panels, strict=True):
        m = np.isfinite(x) & np.isfinite(y)
        ax.plot(x[m], y[m], ".", ms=5, color=color, alpha=0.85, label=title)
        ax.axvspan(t[i_am_min], np.nanmax(t), color="red", alpha=0.08, zorder=0)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, alpha=0.35)
        r_all = rms(y[m]) * 1000.0
        r_2 = rms(y[sl2][np.isfinite(y[sl2])]) * 1000.0
        c2 = corr_am(y, sl2)
        stats.append(f"{title[:28]:28s}  RMS={r_all:5.0f} ppt  2nd½={r_2:5.0f} ppt  r(am)₂={c2:+.2f}")
        if title.startswith("VYVAR mag_calib"):
            ax2 = ax.twinx()
            ax2.plot(t, am, color="#4682b4", alpha=0.45, lw=1.5, label="airmass")
            ax2.set_ylabel("airmass", fontsize=8, color="#4682b4")
            ax2.tick_params(axis="y", labelsize=7)

    axes[-1].set_xlabel(f"Time — BJD − {JD0:.0f} (same night)")
    fig.text(
        0.02,
        0.01,
        "\n".join(stats)
        + f"\nZP frame jitter σ(mag_calib−mag_inst)={rms((lc['mag_calib']-lc['mag_inst']).to_numpy())*1000:.0f} ppt",
        fontsize=7,
        family="monospace",
        va="bottom",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    out_png = out_dir / "BO_CVn_SIPS_vs_VYVAR_ZP_AM.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
