"""
Spusti Fázu 0+1 + Fázu 2A pre draft_000196 a zobraz výsledok pre TIC 392229331.

Spustenie: python run_draft196_phase2a.py
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\ASTRO\python\VYVAR")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

# ── Konfigurácia ─────────────────────────────────────────────────────────────
ARCHIVE_ROOT = r"C:\ASTRO\python\VYVAR\Archive"
DRAFT_ID = 196
TIC_VSX = "392229331"
# ─────────────────────────────────────────────────────────────────────────────


def p2p(arr: np.ndarray | list[float]) -> float:
    a = np.array(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(np.diff(a)) / math.sqrt(2)) if len(a) > 2 else float("nan")


def norm(x: object) -> str:
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return str(x).strip()


def _fwhm_from_masterstar(ms_fits: Path) -> float:
    try:
        from astropy.io import fits as astrofits

        with astrofits.open(ms_fits, memmap=False) as hdul:
            for key in ("VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN", "VY_FWHM"):
                v = hdul[0].header.get(key)
                if v is not None:
                    fv = float(v)
                    if 0.5 < fv < 30.0:
                        return round(fv, 3)
    except Exception:  # noqa: BLE001
        pass
    return 3.7


def main() -> None:
    from config import AppConfig
    from photometry_core import run_phase0_and_phase1, run_phase2a

    cfg = AppConfig()

    draft_dir = Path(ARCHIVE_ROOT) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ps_dir = draft_dir / "platesolve"
    ms_fits = ps_dir / "MASTERSTAR.fits"
    ms_csv = ps_dir / "masterstars_full_match.csv"

    obs_group_dir: Path | None = None
    for subdir in sorted(ps_dir.iterdir()):
        if subdir.is_dir() and (subdir / "per_frame_catalog_index.csv").exists():
            obs_group_dir = subdir
            break

    if obs_group_dir is None:
        print("❌ platesolve podadresár s per_frame_catalog_index.csv nenájdený")
        sys.exit(1)

    vt_csv = obs_group_dir / "variable_targets.csv"
    output_dir = obs_group_dir / "photometry"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_frame_dir: Path | None = None
    aligned_root = draft_dir / "detrended_aligned" / "lights"
    if aligned_root.exists():
        for subdir in sorted(aligned_root.iterdir()):
            if subdir.is_dir():
                per_frame_dir = subdir
                break

    if per_frame_dir is None:
        print("❌ Per-frame CSV adresár nenájdený")
        sys.exit(1)

    if not vt_csv.exists():
        print(f"❌ Chýba variable_targets: {vt_csv}")
        sys.exit(1)
    if not ms_csv.exists():
        print(f"❌ Chýba masterstars: {ms_csv}")
        sys.exit(1)

    print(f"Draft:           {DRAFT_ID}")
    print(f"Obs group:       {obs_group_dir.name}")
    print(f"Per-frame dir:   {per_frame_dir}")
    print(f"Output dir:      {output_dir}")
    print()

    # ── FÁZA 0+1 ────────────────────────────────────────────────────────────
    print("=== Spúšťam Fázu 0+1 ===")
    per_frame_csvs = sorted(per_frame_dir.glob("proc_*.csv"))
    fwhm_hint = _fwhm_from_masterstar(ms_fits)
    print(f"  MASTERSTAR:    {ms_fits.name}")
    print(f"  FWHM hint:     {fwhm_hint:.3f}px")
    print(f"  Per-frame CSV: {len(per_frame_csvs)} súborov")
    print("  Spúšťam (môže trvať 1-2 min)...")

    try:
        run_phase0_and_phase1(
            variable_targets_csv=vt_csv,
            masterstars_csv=ms_csv,
            per_frame_csv_dir=per_frame_dir,
            output_dir=output_dir,
            fwhm_px=fwhm_hint,
        )
        print("  ✅ Fáza 0+1 hotová")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ Fáza 0+1 zlyhala: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print()

    # ── FÁZA 2A ─────────────────────────────────────────────────────────────
    print("=== Spúšťam Fázu 2A ===")
    active_csv = output_dir / "active_targets.csv"
    comp_csv = output_dir / "comparison_stars_per_target.csv"
    if not active_csv.exists() or not comp_csv.exists():
        print(f"❌ Chýba active_targets alebo comparison_stars: {output_dir}")
        sys.exit(1)

    print(f"  FWHM hint:     {fwhm_hint:.3f}px")
    print("  Spúšťam (môže trvať 2-5 min)...")

    try:
        run_phase2a(
            masterstar_fits_path=ms_fits,
            active_targets_csv=active_csv,
            comparison_stars_csv=comp_csv,
            per_frame_csv_dir=per_frame_dir,
            detrended_aligned_dir=per_frame_dir,
            output_dir=output_dir,
            fwhm_px=fwhm_hint,
            cfg=cfg,
        )
        print("  ✅ Fáza 2A hotová")
    except Exception as e:  # noqa: BLE001
        print(f"  ❌ Fáza 2A zlyhala: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print()

    # ── VÝSLEDKY ───────────────────────────────────────────────────────────
    print("=== Výsledky pre TIC 392229331 ===")

    summary_path = output_dir / "photometry_summary.csv"
    if not summary_path.exists():
        print("❌ photometry_summary.csv neexistuje")
        sys.exit(1)

    summary = pd.read_csv(summary_path, low_memory=False)

    tic_row = None
    for col in ("vsx_name", "catalog_id"):
        if col in summary.columns:
            mask = summary[col].astype(str).str.contains(TIC_VSX, na=False)
            if mask.any():
                tic_row = summary[mask].iloc[0]
                break

    if tic_row is None:
        print(f"  TIC {TIC_VSX} nenájdený v summary")
        c = summary.get("vsx_name", summary.get("catalog_id", pd.Series(dtype=object)))
        print(f"  Dostupné (prvých 10): {c.head(10).tolist()}")
    else:
        print(f'  vsx_name:    {tic_row.get("vsx_name", "?")}')
        print(f'  catalog_id:  {norm(tic_row.get("catalog_id", ""))}')
        print(f'  aperture_px: {float(tic_row.get("aperture_px", 0)):.3f}px')
        print(f'  lc_rms:      {float(tic_row.get("lc_rms", 0)):.4f}')
        print(f'  n_good_comp: {tic_row.get("n_good_comp", "?")}')

        lc_dir = output_dir / "lightcurves"
        cid = norm(tic_row.get("catalog_id", ""))
        lc_path = lc_dir / f"lightcurve_{cid}.csv"

        if lc_path.exists():
            lc = pd.read_csv(lc_path, low_memory=False)
            normal = lc[lc["flag"] == "normal"]
            mc = normal["mag_calib"].dropna().to_numpy(dtype=float)
            mc_raw = normal["mag_calib_raw"].dropna().to_numpy(dtype=float)

            print()
            print(f"  Lightcurve ({len(lc)} framov, {len(normal)} normal):")
            if len(mc_raw) > 0:
                print(
                    f"  mag_calib_raw: {mc_raw.min():.4f} – {mc_raw.max():.4f}  "
                    f"Δ={mc_raw.max() - mc_raw.min():.4f}"
                )
            if len(mc) > 0:
                print(
                    f"  mag_calib:     {mc.min():.4f} – {mc.max():.4f}  "
                    f"Δ={mc.max() - mc.min():.4f}"
                )
                print(f"  p2p:           {p2p(mc) * 1000:.1f} ppt")

                mid = len(mc) // 2
                min_idx = int(np.argmin(mc))
                trend = float(np.polyfit(range(len(mc)), mc, 1)[0])
                print()
                print(f"  Lineárny trend: {trend * 1000:.2f} mmag/frame")
                print(f"  Minimum na frame {min_idx}/{len(mc)}")
                print(f"  Prvá pol. mean: {mc[:mid].mean():.4f}")
                print(f"  Druhá pol. mean: {mc[mid:].mean():.4f}")
                if min_idx > 5 and min_idx < len(mc) - 5:
                    print("  ✅ U-tvar (tranzit/zatmenie viditeľný)")
                else:
                    print("  ❌ Monotónna krivka — tranzit nie je viditeľný")
        else:
            print(f"  Lightcurve súbor nenájdený: {lc_path}")

    comp_path = output_dir / "comparison_stars_per_target.csv"
    if comp_path.exists() and tic_row is not None:
        comp = pd.read_csv(comp_path, low_memory=False)
        comp["_tcid"] = comp["target_catalog_id"].apply(norm)
        tic_cid_str = norm(tic_row.get("catalog_id", ""))
        tic_comps = comp[comp["_tcid"] == tic_cid_str]
        print()
        print(f"  Comp hviezdy: {len(tic_comps)}")
        for i, (_, r) in enumerate(tic_comps.iterrows(), 1):
            tier = r.get("comp_tier", "—")
            rms = float(r.get("comp_rms", 0) or 0)
            mag = float(r.get("mag", 0) or 0)
            dist_deg = float(r.get("dist_deg", 0) or 0)
            dist = dist_deg * 3600.0
            print(f'    C{i:02d} mag={mag:.3f} rms={rms:.4f} tier={tier} dist={dist:.0f}"')

    print()
    print("=== Hotovo ===")


if __name__ == "__main__":
    main()
