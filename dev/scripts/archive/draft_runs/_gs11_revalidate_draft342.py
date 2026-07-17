"""Re-run Phase 2A on draft_342 with GS11 enabled; print correction sign table."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import AppConfig, config_json_path  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402

DRAFT = _ROOT / "Archive/Drafts/draft_000342"
SETUP = "NoFilter_60_2"
PS = DRAFT / "platesolve" / SETUP
PHOT = PS / "photometry"
TARGETS = (
    "1497070744341492864",
    "1500380858456833536",
    "1499084499887740160",
    "1499064399440851968",
)


def main() -> int:
    path = config_json_path(_ROOT)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["gs11_dilution_enabled"] = True
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cfg = AppConfig()
    assert cfg.gs11_dilution_enabled

    run_phase2a(
        masterstar_fits_path=PS / "MASTERSTAR.fits",
        active_targets_csv=PHOT / "active_targets.csv",
        comparison_stars_csv=PHOT / "comparison_stars_per_target.csv",
        per_frame_csv_dir=DRAFT / "detrended_aligned" / "lights" / SETUP,
        detrended_aligned_dir=DRAFT / "detrended_aligned" / "lights" / SETUP,
        output_dir=PHOT,
        fwhm_px=float(_load_fwhm(PS / "MASTERSTAR.fits")),
        cfg=cfg,
        draft_id=342,
    )

    import pandas as pd

    summ = pd.read_csv(PHOT / "photometry_summary.csv", dtype={"catalog_id": str})
    print("| target_catalog_id | mag_before | mag_after | delta_mmag | n_neighbors | D |")
    for cid in TARGETS:
        r = summ[summ["catalog_id"].astype(str).str.strip() == cid]
        if r.empty:
            print(f"| {cid} | — | — | — | — | — |")
            continue
        row = r.iloc[0]
        mb = float(row.get("mag_median_pre_gs11", float("nan")))
        ma = float(row.get("mag_median_post_gs11", float("nan")))
        dm = float(row.get("dilution_delta_mag", 0.0)) * 1000.0
        nn = int(row.get("n_neighbors_aperture", 0))
        d = float(row.get("dilution_factor", 1.0))
        print(f"| {cid} | {mb:.4f} | {ma:.4f} | {dm:.1f} | {nn} | {d:.4f} |")
        if math.isfinite(mb) and math.isfinite(ma) and dm > 0:
            assert ma > mb, f"{cid}: mag_after must exceed mag_before"

    meta = json.loads((PHOT / "pipeline_meta.json").read_text(encoding="utf-8"))
    print("\ngs11_summary:", json.dumps(meta.get("gs11_summary", {}), indent=2))

    lc_dir = PHOT / "lightcurves"
    lc_files = sorted(lc_dir.glob("*.csv"))
    if lc_files:
        lc = pd.read_csv(lc_files[0], nrows=3)
        print(f"\nLC sample ({lc_files[0].name}) columns:", list(lc.columns))
        if "dilution_factor" in lc.columns:
            print(lc[["bjd", "mag_calib", "dilution_factor"]].head(3).to_string(index=False))

    data["gs11_dilution_enabled"] = False
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
