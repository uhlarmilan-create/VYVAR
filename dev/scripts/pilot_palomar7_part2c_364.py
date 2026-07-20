#!/usr/bin/env python3
"""Part 2c: export PSF catalogs + Part B RMS table for draft 364 Luminance_60_2."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from pipeline import export_per_frame_catalogs  # noqa: E402

DRAFT_ID = 364
SETUP = "Luminance_60_2"
CONFIG_PATH = _ROOT / "config.json"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    orig = bool(json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get("psf_photometry_enabled", False))
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    data["psf_photometry_enabled"] = True
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    cfg = AppConfig()
    cfg.psf_photometry_enabled = True
    db = VyvarDatabase(cfg.database_path)
    draft = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ps = draft / "platesolve" / SETUP
    aligned = draft / "detrended_aligned" / "lights" / SETUP
    row = db.fetch_obs_draft_by_id(DRAFT_ID) or {}
    eq_id = int(row.get("EQUIPMENT_ID") or row.get("ID_EQUIPMENTS") or 3)

    try:
        per = export_per_frame_catalogs(
            frames_root=aligned,
            platesolve_dir=ps,
            max_catalog_rows=15000,
            catalog_match_max_sep_arcsec=3.0,
            dao_threshold_sigma=3.5,
            dao_fwhm_px=2.5,
            masterstars_csv=ps / "masterstars_full_match.csv",
            masterstar_fits=ps / "MASTERSTAR.fits",
            use_master_fast_path=True,
            catalog_local_gaia_only=True,
            app_config=cfg,
            draft_id=DRAFT_ID,
            equipment_id=eq_id,
        )
        print(f"export written={per.get('written')}", flush=True)

        # Part B via diagnostic module
        diag_path = _ROOT / "scripts" / "diagnose_psf_elongation_362.py"
        spec = importlib.util.spec_from_file_location("diag", diag_path)
        diag = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diag)
        part_b_df, note, _ = diag._part_b_mag_bins(aligned, ps / "masterstars_full_match.csv")
        if part_b_df.empty:
            print(f"Part B skipped: {note}", flush=True)
        else:
            print("\nPart B - aperture vs PSF RMS by mag (Luminance_60_2):", flush=True)
            print("mag_bin | N | median RMS_aperture | median RMS_psf | median psf/aper", flush=True)
            for _, r in part_b_df.iterrows():
                print(
                    f"{r['mag_bin']} | {int(r['N'])} | {r['median_rms_aperture']:.5f} | "
                    f"{r['median_rms_psf']:.5f} | {r['median_ratio_psf_aper']:.3f}",
                    flush=True,
                )
            out_csv = draft / "diagnostics" / "psf_elongation_364" / "d364_part_b_luminance_60_2.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            part_b_df.to_csv(out_csv, index=False)
            print(f"\nWrote {out_csv}", flush=True)

        # count psf_ok
        import pandas as pd
        ok_total = 0
        for p in aligned.glob("proc_*.csv"):
            d = pd.read_csv(p, usecols=["psf_fit_ok"], low_memory=False)
            ok_total += int(d["psf_fit_ok"].fillna(False).astype(bool).sum())
        print(f"Total psf_fit_ok rows across frames: {ok_total}", flush=True)
    finally:
        data["psf_photometry_enabled"] = orig
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"restored psf_photometry_enabled={orig}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
