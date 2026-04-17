"""
Diagnostika ensemble normalizácie pre TIC v draft_198.
Porovná mag_inst(TIC) vs mag_inst(comps) frame-po-frame.

Spustenie: python diagnose_ensemble.py
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

draft_dir = Path(ARCHIVE) / "Drafts" / f"draft_{DRAFT_ID:06d}"
csv_dir = draft_dir / "detrended_aligned" / "lights" / SETUP
lc_path = (
    draft_dir
    / "platesolve"
    / SETUP
    / "photometry"
    / "lightcurves"
    / "lightcurve_486430957815961344.csv"
)

TIC_280 = "486430957815961280"
COMP_IDS = [
    "474335196039769344",  # C01
    "486558672963607424",  # C02
    "474332378541197184",  # C03
]


def norm(x: object) -> str:
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x).strip()


def flux_to_mag(f: float) -> float:
    return -2.5 * math.log10(f) if f and f > 0 else float("nan")


def main() -> None:
    csvs = sorted(csv_dir.glob("proc_*.csv"))

    print("=== Frame-po-frame: mag_inst TIC vs Comps vs Ensemble ===")
    print(
        f"{'Fr':>3} {'mag_T':>8} {'mag_C1':>8} {'mag_C2':>8} "
        f"{'mag_C3':>8} {'ens':>8} {'delta':>8} {'am':>6}"
    )
    print("-" * 68)

    mag_t_all: list[float] = []
    ens_all: list[float] = []
    am_all: list[float] = []
    delta_all: list[float] = []

    for i, csv_p in enumerate(csvs[:82]):
        df = pd.read_csv(csv_p, low_memory=False)
        df["_cid"] = df["catalog_id"].apply(norm)

        # TIC flux (...280 — ...344 v per-frame CSV nie je)
        r_t = df[df["_cid"] == TIC_280]
        flux_t = float(r_t.iloc[0]["dao_flux"]) if not r_t.empty else float("nan")
        mag_t = flux_to_mag(flux_t)

        comp_fluxes: list[float] = []
        comp_mags: list[float] = []
        for cid in COMP_IDS:
            r = df[df["_cid"] == cid]
            if not r.empty:
                f = float(r.iloc[0]["dao_flux"])
                comp_fluxes.append(f)
                comp_mags.append(flux_to_mag(f))
            else:
                comp_fluxes.append(float("nan"))
                comp_mags.append(float("nan"))

        am_col = df.get("airmass", df.get("AIRMASS", pd.Series(dtype=float)))
        am = float(am_col.iloc[0]) if len(am_col) > 0 else float("nan")

        valid_f = [f for f in comp_fluxes if math.isfinite(f) and f > 0]
        ens_flux = float(np.mean(valid_f)) if valid_f else float("nan")
        ens_mag = flux_to_mag(ens_flux)

        delta = mag_t - ens_mag if math.isfinite(mag_t) and math.isfinite(ens_mag) else float("nan")

        mag_t_all.append(mag_t)
        ens_all.append(ens_mag)
        am_all.append(am)
        delta_all.append(delta)

        cm = [f"{m:8.4f}" if math.isfinite(m) else "     nan" for m in comp_mags]
        ens_s = f"{ens_mag:8.4f}" if math.isfinite(ens_mag) else "     nan"
        delta_s = f"{delta:8.4f}" if math.isfinite(delta) else "     nan"
        mag_t_s = f"{mag_t:8.4f}" if math.isfinite(mag_t) else "     nan"
        am_s = f"{am:6.4f}" if math.isfinite(am) else "   nan"

        print(f"{i + 1:>3} {mag_t_s} {'  '.join(cm)} {ens_s} {delta_s} {am_s}")

    print()

    # Zarovnané polia (rovnaký frame pre všetky veličiny)
    mag_t_arr_list: list[float] = []
    ens_arr_list: list[float] = []
    delta_arr_list: list[float] = []
    am_arr_list: list[float] = []
    for mt, ens, d, am in zip(mag_t_all, ens_all, delta_all, am_all):
        if all(math.isfinite(x) for x in (mt, ens, d, am)):
            mag_t_arr_list.append(mt)
            ens_arr_list.append(ens)
            delta_arr_list.append(d)
            am_arr_list.append(am)

    mag_t_arr = np.array(mag_t_arr_list, dtype=float)
    ens_arr = np.array(ens_arr_list, dtype=float)
    delta_arr = np.array(delta_arr_list, dtype=float)
    am_arr = np.array(am_arr_list, dtype=float)

    n = len(mag_t_arr)
    if n > 3:
        slope_t = float(np.polyfit(am_arr, mag_t_arr, 1)[0])
        slope_ens = float(np.polyfit(am_arr, ens_arr, 1)[0])
        slope_d = float(np.polyfit(am_arr, delta_arr, 1)[0])

        print("=== Airmass slopes (zarovnané frame-y) ===")
        print(f"  mag_inst(TIC)   vs airmass: {slope_t:+.4f} mag/am")
        print(f"  mag_inst(ens)   vs airmass: {slope_ens:+.4f} mag/am")
        print(f"  delta_mag       vs airmass: {slope_d:+.4f} mag/am")
        print()
        print(f"  TIC pokles:  {slope_t:+.4f} mag/am")
        print(f"  Ens pokles:  {slope_ens:+.4f} mag/am")
        diff = slope_t - slope_ens
        print(f"  Rozdiel:     {diff:+.4f} mag/am")
        if abs(diff) > 0.3:
            print("  ❌ TIC a Ensemble majú rozdielny airmass response!")
            print("     Ensemble NEZRUŠUJE airmass trend TIC.")
            print("     → možný vignetting / iná extinkcia na okraji vs comps")
        else:
            print("  ✅ TIC a Ensemble majú podobný airmass response")

    print()

    print("=== Porovnanie s lightcurve_344.csv (uložená fázou 2A) ===")
    if lc_path.exists():
        lc = pd.read_csv(lc_path, low_memory=False)
        normal = lc[lc["flag"] == "normal"]
        print(f"  n_normal: {len(normal)}")
        if "delta_mag" in lc.columns:
            dm_all = lc["delta_mag"].dropna()
            print(
                f"  delta_mag LC (všetky riadky): min={dm_all.min():.4f}  max={dm_all.max():.4f}"
            )
            dm_ok = normal["delta_mag"].dropna()
            if len(dm_ok) > 0:
                print(
                    f"  delta_mag LC (iba normal): min={dm_ok.min():.4f}  max={dm_ok.max():.4f}"
                )
            if len(delta_arr) > 0:
                print(
                    f"  Naša delta (mean flux→mag, C01–C03 z CSV): min={delta_arr.min():.4f}  max={delta_arr.max():.4f}"
                )
                print(
                    "  Pozn.: Fáza 2A používa mag_inst z read_flux + váhovaný ensemble "
                    "(1/rms²), nie aritmetický mean dao_flux z raw CSV. "
                    "LC je pre catalog 344; mag_T vyššie je z ...280 (dao_flux v CSV pri susedovi)."
                )
                if len(dm_ok) > 0:
                    match = abs(float(dm_ok.min()) - float(delta_arr.min())) < 0.1
                    print(
                        f"  {'✅ Zhodné (min vs normal)' if match else '❌ ROZDIEL (očakávané — iná definícia ensemble)'}"
                    )
            else:
                print("  Naša delta_mag: žiadne kompletné frame-y")
        else:
            print("  delta_mag stĺpec chýba")
    else:
        print(f"  Chýba: {lc_path}")


if __name__ == "__main__":
    main()
