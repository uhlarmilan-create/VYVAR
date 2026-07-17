#!/usr/bin/env python3
"""Plot LC variants (ZP / detrend / demean) + AIJ overlay for BO/FW CVn.

Usage:
  python scripts/plot_lc_variants_draft311.py
  python scripts/plot_lc_variants_draft311.py --out photometry/_lc_compare

Writes PNG per star with stacked panels and a summary RMS table.
"""
from __future__ import annotations

import argparse
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

STARS = [
    ("FW CVn", "1497343732462852864", "FW_CVn_aij.tbl"),
    ("BO CVn", "1498613634033133184", "BO_CVn_aij.tbl"),
]


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) > 1 else float("nan")


def demean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmedian(x)
    return x - m


def ppt(mag_rms: float) -> float:
    return mag_rms * 1000.0 if np.isfinite(mag_rms) else float("nan")


def load_aij(aij_path: Path) -> pd.DataFrame:
    aij = pd.read_csv(aij_path, sep="\t").sort_values("BJD_TDB")
    f = pd.to_numeric(aij["rel_flux_T1"], errors="coerce").to_numpy()
    mag = -2.5 * np.log10(np.clip(f, 1e-12, None))
    aij = aij.copy()
    aij["mag_aij"] = mag
    aij["mag_aij_dm"] = demean(mag)
    aij["rel_flux_norm"] = f / np.nanmedian(f)
    aij["rel_flux_dm"] = demean(f)
    # time axis like AIJ plot: JD - 2461154
    aij["t_plot"] = pd.to_numeric(aij["JD_UTC"], errors="coerce") - 2461154.0
    return aij


def build_vyvar_variants(lc: pd.DataFrame) -> dict[str, np.ndarray]:
    """LC columns from save_lightcurve_csv."""
    lc = lc.sort_values("bjd").reset_index(drop=True)
    inst = lc["mag_inst"].to_numpy(dtype=float)
    raw = lc["mag_calib_raw"].to_numpy(dtype=float)
    cal = lc["mag_calib"].to_numpy(dtype=float)
    dm = lc["delta_mag"].to_numpy(dtype=float)
    am = lc["airmass"].to_numpy(dtype=float)
    t = lc["bjd"].to_numpy(dtype=float) - 2461154.0

    # Bez ensemble ZP: instrumental (per-frame inst mag, no catalog ZP sum)
    no_zp = inst.copy()
    # Kalibrované bez airmass detrendu (po ZP, pred AM fit — uložené v mag_calib_raw)
    zp_only = raw.copy()
    # Plný mag_calib (ZP + pokus o AM detrend)
    full = cal.copy()
    # Odstrániť per-frame ZP offset (späť k „relatívnej“ variácii okolo medianu inst+ZP)
    cal_minus_dm = cal - dm

    return {
        "t": t,
        "airmass": am,
        "mag_inst (bez ZP)": inst,
        "mag_inst demean": demean(inst),
        "mag_calib_raw (ZP, bez AM detrend)": zp_only,
        "mag_calib_raw demean": demean(zp_only),
        "mag_calib (ZP+AM pipeline)": full,
        "mag_calib demean": demean(full),
        "mag_calib - delta_mag": cal_minus_dm,
        "mag_calib - delta_mag demean": demean(cal_minus_dm),
    }


def plot_star(
    name: str,
    cid: str,
    aij_file: str,
    lc_path: Path,
    out_dir: Path,
    summary_row: pd.Series | None,
) -> Path:
    lc = pd.read_csv(lc_path)
    variants = build_vyvar_variants(lc)
    aij = load_aij(DRAFT / "detrended_aligned" / "lights" / aij_file)

    n_panels = 5
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(f"{name} — draft_311 NoFilter_60_2 (Phase 2A, forced ap=7 px)", fontsize=12)

    t = variants["t"]
    am = variants["airmass"]

    panels = [
        ("AIJ rel_flux_T1 (normalized)", aij["rel_flux_norm"].to_numpy(), aij["t_plot"].to_numpy(), "#1f77b4"),
        ("AIJ mag from rel_flux (demean)", aij["mag_aij_dm"].to_numpy(), aij["t_plot"].to_numpy(), "#1f77b4"),
        ("VYVAR mag_inst — bez ZP", variants["mag_inst demean"], t, "#d62728"),
        ("VYVAR mag_calib_raw demean — ZP, bez AM", variants["mag_calib_raw demean"], t, "#ff7f0e"),
        ("VYVAR mag_calib demean — plná pipeline", variants["mag_calib demean"], t, "#2ca02c"),
    ]

    rms_lines: list[str] = []
    for ax, (title, y, x, color) in zip(axes, panels, strict=True):
        m = np.isfinite(y) & np.isfinite(x)
        ax.plot(x[m], y[m], ".", ms=4, color=color, alpha=0.85)
        ax.set_ylabel(title, fontsize=8)
        ax.grid(True, alpha=0.3)
        rms_lines.append(f"{title}: RMS={ppt(rms(y[m])):.1f} ppt")
        ax2 = ax.twinx()
        ax2.plot(t, am, color="#87ceeb", alpha=0.35, lw=1.2)
        ax2.set_ylabel("airmass", fontsize=7, color="#4682b4")
        ax2.tick_params(axis="y", labelsize=7)

    axes[-1].set_xlabel("Geocentric JD (UTC) − 2461154")
    if summary_row is not None:
        rms_lines.append(
            f"summary lc_rms={float(summary_row.get('lc_rms', float('nan'))):.4f}  "
            f"ap={float(summary_row.get('aperture_px', float('nan'))):.3f}  "
            f"am_detrended={summary_row.get('am_detrended')}"
        )
    fig.text(
        0.02,
        0.01,
        "\n".join(rms_lines),
        fontsize=7,
        family="monospace",
        va="bottom",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out_path = out_dir / f"lc_variants_{cid[-8:]}_{name.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_overlay(name: str, cid: str, aij_file: str, lc_path: Path, out_dir: Path) -> Path:
    """Single panel: AIJ vs VYVAR demeaned mags."""
    lc = pd.read_csv(lc_path).sort_values("bjd")
    aij = load_aij(DRAFT / "detrended_aligned" / "lights" / aij_file)
    t = lc["bjd"].to_numpy() - 2461154.0
    vy = demean(lc["mag_calib"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(aij["t_plot"], aij["mag_aij_dm"], ".", ms=4, label=f"AIJ (RMS {ppt(rms(aij['mag_aij_dm'])):.1f} ppt)", alpha=0.8)
    ax.plot(t, vy, ".", ms=4, label=f"VYVAR mag_calib demean (RMS {ppt(rms(vy)):.1f} ppt)", alpha=0.8)
    ax.set_xlabel("JD − 2461154")
    ax.set_ylabel("demeaned mag")
    ax.set_title(f"{name} — AIJ vs VYVAR (ap=7 px Phase 2A)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"lc_overlay_{cid[-8:]}_{name.replace(' ', '_')}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, default=DRAFT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    phot = args.draft / "platesolve" / SETUP / "photometry"
    out_dir = args.out or (phot / "_lc_compare_ap7")
    out_dir.mkdir(parents=True, exist_ok=True)
    lc_dir = phot / "lightcurves"

    summary = pd.read_csv(phot / "photometry_summary.csv", dtype={"catalog_id": str})

    print("Output:", out_dir)
    for name, cid, aijf in STARS:
        lc_path = lc_dir / f"lightcurve_{cid}.csv"
        if not lc_path.is_file():
            print("Missing", lc_path)
            continue
        row = summary[summary["catalog_id"] == cid]
        srow = row.iloc[0] if not row.empty else None
        p1 = plot_star(name, cid, aijf, lc_path, out_dir, srow)
        p2 = plot_overlay(name, cid, aijf, lc_path, out_dir)
        print(f"  {name}: {p1.name}, {p2.name}")

        lc = pd.read_csv(lc_path).sort_values("bjd")
        v = build_vyvar_variants(lc)
        print(f"    RMS ppt: inst={ppt(rms(v['mag_inst (bez ZP)'])):.1f}  "
              f"raw_dm={ppt(rms(v['mag_calib_raw demean'])):.1f}  "
              f"cal_dm={ppt(rms(v['mag_calib demean'])):.1f}  "
              f"no_dm_dm={ppt(rms(v['mag_calib - delta_mag demean'])):.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
