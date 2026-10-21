"""Unit tests for diagnostic sigma budget (Howell + Osborn scintillation)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from photometry_core import (
    SIGMA_BKG_AP_COL,
)
from sigma_budget import RigScintillationParams, scintillation_sigma


def test_scintillation_osborn_paper_scale_sanity():
    """Larger aperture -> lower scintillation variance (D^-4/3)."""
    v200 = scintillation_sigma(telescope_diameter_m=0.2, airmass=1.5, exposure_s=60.0, altitude_m=275.0) ** 2
    v300 = scintillation_sigma(telescope_diameter_m=0.3, airmass=1.5, exposure_s=60.0, altitude_m=275.0) ** 2
    assert v200 > v300
    ratio = v200 / v300
    assert abs(ratio - (0.3 / 0.2) ** (4.0 / 3.0)) < 0.05


def _rig_params() -> RigScintillationParams:
    return RigScintillationParams(
        draft_id=426,
        setup="g_60_4",
        telescope_diameter_m=0.2,
        altitude_m=275.0,
        exposure_s=60.0,
        c_y=1.5,
        source_notes=["test"],
    )


def _write_proc_row(
    proc_dir: Path,
    *,
    source_file: str,
    catalog_id: str,
    sigma_bkg_ap: float | None,
    err_bkg_source: str | None = None,
) -> None:
    row = {
        "catalog_id": catalog_id,
        "dao_flux": 12000.0,
        "sky_adu_per_px_annulus": 80.0,
        "aperture_area_px": 50.0,
        "aperture_r_px": 4.0,
        SIGMA_BKG_AP_COL: sigma_bkg_ap if sigma_bkg_ap is not None else float("nan"),
    }
    if err_bkg_source is not None:
        row["err_bkg_source"] = err_bkg_source
    pd.DataFrame([row]).to_csv(proc_dir / source_file, index=False)


