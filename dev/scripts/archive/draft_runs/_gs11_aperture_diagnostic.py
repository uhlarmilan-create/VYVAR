"""Print GS11 aperture calculation for one target (draft_342)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from photometry_core import (  # noqa: E402
    _resolve_plate_scale_arcsec_per_px,
    _read_plate_scale_from_fits_path,
)

cid = "1500271044733081088"
ms = _ROOT / "Archive/Drafts/draft_000342/platesolve/NoFilter_60_2/MASTERSTAR.fits"
cfg = AppConfig()

import pandas as pd

summ = pd.read_csv(
    _ROOT / "Archive/Drafts/draft_000342/platesolve/NoFilter_60_2/photometry/photometry_summary.csv",
    dtype={"catalog_id": str},
)
ap_px = float(summ[summ["catalog_id"] == cid].iloc[0]["aperture_px"])
fits_ps = _read_plate_scale_from_fits_path(ms)
resolved_ps = _resolve_plate_scale_arcsec_per_px(cfg, ms, default=1.3)
ap_cfg = float(cfg.gs11_dilution_aperture_arcsec or 0.0)
ap_arcsec = ap_cfg if ap_cfg > 0 else ap_px * resolved_ps

print(f"target {cid}")
print(f"  cfg.gs11_dilution_aperture_arcsec = {ap_cfg} (0.0 = auto)")
print(f"  apertures_px[target] (radius px)     = {ap_px}")
print(f"  FITS SECPIX1 raw                     = {fits_ps}")
print(f"  state.plate_scale_arcsec (resolved)  = {resolved_ps}")
print(f"  _ap_arcsec = {ap_arcsec:.4f}")
