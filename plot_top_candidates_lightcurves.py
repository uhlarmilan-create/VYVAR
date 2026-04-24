from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _norm_cid(x: object) -> int | None:
    s = str(x or "").strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    try:
        return int(Decimal(s))
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        try:
            return int(s)
        except Exception:
            return None


def main() -> None:
    per_frame_dir = Path(
        r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000249\detrended_aligned\lights\NoFilter_60_2"
    )
    out_png = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000249\top_candidates_lightcurves.png")

    targets = {
        1497974233661289000: "Nový kandidát 1 (mag=12.47)",
        1485338783474459600: "Nový kandidát 2 (mag=13.45)",
        1497236186481686000: "SS CVn RRAB/BL (validácia)",
    }

    csvs = sorted(per_frame_dir.glob("proc_*.csv"))
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for ax, (cid, name) in zip(axes, targets.items()):
        bjd_list: list[float] = []
        mag_list: list[float] = []

        for csv in csvs:
            try:
                df = pd.read_csv(csv, usecols=["catalog_id", "dao_flux", "bjd_tdb_mid"], low_memory=False)
            except Exception:
                continue

            df["_cid"] = [_norm_cid(v) for v in df["catalog_id"].tolist()]
            row = df[df["_cid"] == int(cid)]
            if row.empty:
                continue

            try:
                flux = float(row.iloc[0]["dao_flux"])
                bjd = float(row.iloc[0]["bjd_tdb_mid"])
            except Exception:
                continue

            if np.isfinite(flux) and flux > 0 and np.isfinite(bjd):
                mag = -2.5 * float(np.log10(flux))
                bjd_list.append(bjd)
                mag_list.append(mag)

        if bjd_list:
            bjd_arr = np.array(bjd_list, dtype=float)
            mag_arr = np.array(mag_list, dtype=float)
            idx = np.argsort(bjd_arr)
            bjd_arr = bjd_arr[idx]
            mag_arr = mag_arr[idx]
            bjd_rel = bjd_arr - bjd_arr[0]

            ax.scatter(bjd_rel, mag_arr, s=8, alpha=0.7, color="steelblue")
            ax.invert_yaxis()
            ax.set_title(f"{name}  |  N={len(bjd_list)}  RMS={np.std(mag_arr):.3f} mag")
            ax.set_xlabel("BJD - BJD0")
            ax.set_ylabel("mag_inst")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.1, 0.5, "No data found", transform=ax.transAxes)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"Uložené: {out_png}")


if __name__ == "__main__":
    main()

