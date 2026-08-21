"""ERR-518-02: re-run enhance on draft 518 aligned FITS to refresh proc CSV sigma columns."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src_py"))

from config import AppConfig
from photometry_core import (
    ERR_BKG_SOURCE_COL,
    SIGMA_BKG_AP_COL,
    _phase2a_empirical_sigma_bkg_ap,
    enhance_catalog_dataframe_aperture_bpm,
)

DRAFT = REPO / "Archive" / "Drafts" / "draft_000518"
PROC_DIR = DRAFT / "detrended_aligned" / "lights" / "V_60_2"
DROP = [
    SIGMA_BKG_AP_COL,
    ERR_BKG_SOURCE_COL,
    "aperture_r_px",
    "aperture_factor_applied",
    "fwhm_px_for_aperture",
    "fwhm_px_scope",
    "snr_aperture_mode",
    "sky_annulus_r_out_px",
    "sky_adu_per_px_annulus",
    "flux_small",
    "flux_large",
    "noise_floor_adu",
    "fwhm_estimate_px",
]


def main() -> None:
    cfg = AppConfig()
    tot: Counter[str] = Counter()
    first_sigma: float | None = None
    preflight_ok = True
    proc_files = sorted(PROC_DIR.glob("proc_*.csv"))
    for proc_path in proc_files:
        stem = proc_path.name.replace("proc_", "").replace(".csv", "")
        fits_path = PROC_DIR / f"{stem}.fits"
        if not fits_path.is_file():
            continue
        df = pd.read_csv(proc_path, low_memory=False)
        base = df.drop(columns=[c for c in DROP if c in df.columns], errors="ignore")
        with fits.open(fits_path, memmap=False) as hd:
            hdr = hd[0].header.copy()
            data = hd[0].data.astype("float32")
        out = enhance_catalog_dataframe_aperture_bpm(
            base,
            data,
            hdr,
            aperture_enabled=True,
            aperture_fwhm_factor=float(cfg.aperture_fwhm_factor),
            annulus_inner_fwhm=float(cfg.annulus_inner_fwhm),
            annulus_outer_fwhm=float(cfg.annulus_outer_fwhm),
            nonlinearity_peak_percentile=float(cfg.nonlinearity_peak_percentile),
            nonlinearity_fwhm_ratio=float(cfg.nonlinearity_fwhm_ratio),
            master_dark_path=None,
            snr_aperture_table=None,
            gaussian_fwhm_px_override=2.2919,
            err_background_mode=cfg.err_background_mode,
            err_empty_apertures_n=int(cfg.err_empty_apertures_n),
            err_empty_apertures_min=int(cfg.err_empty_apertures_min),
        )
        out.to_csv(proc_path, index=False)
        tot.update(out[ERR_BKG_SOURCE_COL].astype(str))
        if first_sigma is None:
            sig = pd.to_numeric(out[SIGMA_BKG_AP_COL], errors="coerce").dropna()
            if not sig.empty:
                first_sigma = float(sig.iloc[0])
        row = out[out["catalog_id"].astype(str) == "1624628764771224960"]
        if not row.empty:
            try:
                _phase2a_empirical_sigma_bkg_ap(
                    row.iloc[0],
                    err_background_mode=cfg.err_background_mode,
                    source_file=proc_path.name,
                    catalog_id="1624628764771224960",
                )
            except Exception:
                preflight_ok = False

    summary = {
        "n_files": len(proc_files),
        "err_bkg_source": dict(tot),
        "first_sigma_bkg_ap": first_sigma,
        "inv_err_mode_01_preflight_sample": preflight_ok,
    }
    out_dir = REPO / "dev" / "results" / "context" / "session_20260821_err518_02"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reexport_counts.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
