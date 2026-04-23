#!/usr/bin/env python3
"""Zožeň PSF svetelnú krivku pre MN Boo (alebo iný catalog_id) z draft *_psf.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description="PSF light curve z psf_results *_psf.csv")
    p.add_argument("--draft", type=int, default=246)
    p.add_argument("--obs-group", default="NoFilter_60_2")
    p.add_argument(
        "--catalog-id",
        type=int,
        default=1591057651117374976,
        help="Gaia catalog_id (MN Boo default)",
    )
    p.add_argument(
        "--archive",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Archive",
    )
    args = p.parse_args()

    psf_dir = (
        args.archive
        / "Drafts"
        / f"draft_{args.draft:06d}"
        / "epsf_data"
        / "psf_results"
    )
    files = sorted(psf_dir.glob("*_psf.csv"))
    if not files:
        raise SystemExit(f"Žiadne *_psf.csv v {psf_dir}")

    rows: list[dict] = []
    cid = int(args.catalog_id)
    for fp in files:
        df = pd.read_csv(fp, low_memory=False)
        if "catalog_id" not in df.columns:
            continue
        sub = df[df["catalog_id"] == cid]
        if sub.empty:
            continue
        r = sub.iloc[0]
        rows.append(
            {
                "bjd_tdb_mid": float(r.get("bjd_tdb_mid", np.nan)),
                "psf_flux": float(r.get("psf_flux", np.nan)),
                "psf_flux_err": float(r.get("psf_flux_err", np.nan)),
                "psf_chi2": float(r.get("psf_chi2", np.nan)),
                "psf_fit_ok": bool(r.get("psf_fit_ok", False)),
                "frame_stem": str(r.get("frame_stem", fp.stem)),
            }
        )

    out = pd.DataFrame(rows).sort_values("bjd_tdb_mid").reset_index(drop=True)
    if out.empty:
        raise SystemExit(f"Žiadne riadky pre catalog_id={cid}")

    out_csv = psf_dir / f"lightcurve_psf_{cid}.csv"
    out.to_csv(out_csv, index=False)

    bjd = out["bjd_tdb_mid"].to_numpy(dtype=float)
    flx = out["psf_flux"].to_numpy(dtype=float)
    err = out["psf_flux_err"].to_numpy(dtype=float)
    ok = np.isfinite(flx) & (flx > 0)
    rel_mag = np.full_like(flx, np.nan, dtype=float)
    med = float(np.nanmedian(flx[ok]))
    if med > 0:
        rel_mag[ok] = -2.5 * np.log10(flx[ok] / med)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    m = np.isfinite(err) & (err > 0) & ok
    ax0.errorbar(bjd, flx, yerr=np.where(m, err, np.nan), fmt=".", capsize=0, alpha=0.7, color="C0")
    ax0.set_ylabel("PSF flux (ADU)")
    ax0.set_title(f"MN Boo — PSF light curve (N={len(out)})")
    ax0.grid(True, alpha=0.3)

    ax1.scatter(bjd, rel_mag, s=12, alpha=0.7, c="C1")
    ax1.set_ylabel("Δmag vs median(PSF flux)")
    ax1.set_xlabel("BJD_TDB (mid)")
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    fig.tight_layout()
    out_png = psf_dir / f"lightcurve_psf_{cid}.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"CSV:  {out_csv}")
    print(f"PNG:  {out_png}")
    print(f"Body: {len(out)} (framov)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
