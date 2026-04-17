"""
Diagnostika: porovnaj flux TIC frame-po-frame VYVAR vs AIJ.

Spustenie: python diagnose_flux.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

sys.path.insert(0, r"C:\ASTRO\python\VYVAR")

DRAFT_ID = 198
ARCHIVE = r"C:\ASTRO\python\VYVAR\Archive"
SETUP = "NoFilter_120_2"

# AIJ Source-Sky_T1 z Table.tbl (prvých 10 framov pre porovnanie)
AIJ_T1 = [
    93345,
    91866,
    92432,
    92697,
    93103,
    91999,
    92616,
    93188,
    91966,
    92177,
]

draft_dir = Path(ARCHIVE) / "Drafts" / f"draft_{DRAFT_ID:06d}"
csv_dir = draft_dir / "detrended_aligned" / "lights" / SETUP
lc_dir = draft_dir / "platesolve" / SETUP / "photometry" / "lightcurves"

TIC_344 = "486430957815961344"
TIC_280 = "486430957815961280"


def norm(x: object) -> str:
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x).strip()


def main() -> None:
    csvs = sorted(csv_dir.glob("proc_*.csv"))
    print(f"Per-frame CSV súborov: {len(csvs)}")
    print()

    print(
        f"{'Frame':<6} {'TIC_344':>12} {'TIC_280':>12} {'AIJ_T1':>10} "
        f"{'diff_344%':>10} {'diff_280%':>10} {'found_by':<12}"
    )
    print("-" * 75)

    for i, csv_path in enumerate(csvs[:15]):
        df = pd.read_csv(csv_path, low_memory=False)
        df["_cid"] = df["catalog_id"].apply(norm)

        r344 = df[df["_cid"] == TIC_344]
        r280 = df[df["_cid"] == TIC_280]

        flux_344 = float(r344.iloc[0]["dao_flux"]) if not r344.empty else float("nan")
        flux_280 = float(r280.iloc[0]["dao_flux"]) if not r280.empty else float("nan")
        aij = float(AIJ_T1[i]) if i < len(AIJ_T1) else float("nan")

        diff_344 = (
            (flux_344 - aij) / aij * 100
            if math.isfinite(flux_344) and math.isfinite(aij) and aij > 0
            else float("nan")
        )
        diff_280 = (
            (flux_280 - aij) / aij * 100
            if math.isfinite(flux_280) and math.isfinite(aij) and aij > 0
            else float("nan")
        )

        found = "344" if not r344.empty else ("280" if not r280.empty else "NONE")

        print(
            f"{i + 1:<6} {flux_344:>12.0f} {flux_280:>12.0f} {aij:>10.0f} "
            f"{diff_344:>+9.1f}% {diff_280:>+9.1f}% {found:<12}"
        )

    print()

    # Teraz skontroluj lightcurve CSV pre TIC
    print("=== Lightcurve CSV analýza ===")
    lc_path_344 = lc_dir / f"lightcurve_{TIC_344}.csv"
    lc_path_280 = lc_dir / f"lightcurve_{TIC_280}.csv"

    for path, label in [(lc_path_344, "...344"), (lc_path_280, "...280")]:
        if path.exists():
            lc = pd.read_csv(path, low_memory=False)
            normal = lc[lc["flag"] == "normal"]
            mc = normal["mag_calib"].dropna().to_numpy(dtype=float)
            mc_raw = normal["mag_calib_raw"].dropna().to_numpy(dtype=float)
            am = normal["airmass"].dropna().to_numpy(dtype=float)

            print(f"\n  {label} ({len(lc)} framov, {len(normal)} normal):")
            print(f"    aperture_r_px unikátne: {lc['aperture_r_px'].unique()}")
            if len(mc_raw) > 0:
                print(
                    f"    mag_calib_raw: {mc_raw.min():.4f} – {mc_raw.max():.4f}  "
                    f"Δ={mc_raw.max() - mc_raw.min():.4f}"
                )
            if len(mc) > 0:
                print(
                    f"    mag_calib:     {mc.min():.4f} – {mc.max():.4f}  "
                    f"Δ={mc.max() - mc.min():.4f}"
                )

            # Airmass slope
            if len(am) == len(mc_raw) and len(am) > 3:
                mask = np.isfinite(am) & np.isfinite(mc_raw)
                slope_raw = float(np.polyfit(am[mask], mc_raw[mask], 1)[0])
                slope_cal = float(np.polyfit(am[mask], mc[mask], 1)[0])
                ok_cal = "✅ OK" if abs(slope_cal) <= 0.3 else "❌ TREND OSTÁVA"
                print(f"    airmass slope raw: {slope_raw:.4f} mag/am")
                print(f"    airmass slope cal: {slope_cal:.4f} mag/am  ({ok_cal})")

            # Tvar
            if len(mc) > 10:
                min_idx = int(np.argmin(mc))
                mid = len(mc) // 2
                print(f"    Minimum: frame {min_idx}/{len(mc)}")
                print(f"    Prvá pol: {mc[:mid].mean():.4f}  Druhá pol: {mc[mid:].mean():.4f}")
        else:
            print(f"\n  {label}: lightcurve neexistuje")

    print()
    print("=== Porovnaj delta_mag per-frame ===")
    if lc_path_344.exists():
        lc = pd.read_csv(lc_path_344, low_memory=False)
        if "delta_mag" in lc.columns:
            dm = lc["delta_mag"].dropna().to_numpy(dtype=float)
            print(f"  delta_mag: min={dm.min():.4f}  max={dm.max():.4f}  Δ={dm.max() - dm.min():.4f}")
            print("  (ensemble korekcia aplikovaná?)")
        else:
            print("  delta_mag stĺpec chýba")
    else:
        print("  lightcurve_344 neexistuje — preskočené")


if __name__ == "__main__":
    main()
