# -*- coding: ascii -*-
"""SAT-DIAG unit tests (INV-SAT-01 regression gate)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))

from sat_diag import (  # noqa: E402
    N_PILEUP_MIN,
    PILEUP_RATIO,
    SAT_PEAK_SOURCE_PLACED,
    apply_raw_peaks_to_proc_df,
    commit_sat_diag_provenance,
    derive_ceiling_from_paths,
    measure_raw_peaks_frame,
    resolve_sat_limit,
    PileupResult,
    SatDiagContext,
)


def _make_uint16_fits(path: Path, data: np.ndarray) -> None:
    hdu = fits.PrimaryHDU(data.astype(np.uint16))
    hdu.header["BITPIX"] = 16
    hdu.header["BZERO"] = 32768
    hdu.header["BSCALE"] = 1
    hdu.writeto(path, overwrite=True)


def test_pileup_detection_at_65535(tmp_path: Path) -> None:
    d = np.full((32, 32), 5000, dtype=np.uint16)
    d.ravel()[: N_PILEUP_MIN + 10] = 65535
    d.ravel()[N_PILEUP_MIN + 10] = 65532
    fp = tmp_path / "f.fits"
    _make_uint16_fits(fp, d)
    r = derive_ceiling_from_paths([fp])
    assert r.pileup_detected
    assert r.v_ceiling == 65535.0


def test_no_pileup_when_below_threshold(tmp_path: Path) -> None:
    d = np.full((32, 32), 5000, dtype=np.uint16)
    d[0, 0] = 65535
    fp = tmp_path / "f.fits"
    _make_uint16_fits(fp, d)
    r = derive_ceiling_from_paths([fp])
    assert not r.pileup_detected


def test_conflict_derived_refutes_equipment(tmp_path: Path) -> None:
    d = np.full((32, 32), 5000, dtype=np.uint16)
    d.ravel()[: N_PILEUP_MIN + 10] = 65535
    fp = tmp_path / "f.fits"
    _make_uint16_fits(fp, d)
    pileup = derive_ceiling_from_paths([fp])
    hdr = fits.Header()
    hdr["BITPIX"] = 16
    hdr["BZERO"] = 32768
    ctx = resolve_sat_limit(hdr=hdr, pileup=pileup, equipment_adu=16384.0)
    assert ctx.sat_source == "CONFLICT_DERIVED"
    assert ctx.sat_adu == 65535.0
    assert ctx.refuted_source == "EQUIPMENT"


def test_derived_no_pileup_bitpix(tmp_path: Path) -> None:
    d = np.full((32, 32), 5000, dtype=np.uint16)
    fp = tmp_path / "f.fits"
    _make_uint16_fits(fp, d)
    pileup = derive_ceiling_from_paths([fp])
    hdr = fits.Header()
    hdr["BITPIX"] = 16
    hdr["BZERO"] = 32768
    ctx = resolve_sat_limit(hdr=hdr, pileup=pileup, equipment_adu=None)
    assert ctx.sat_source == "DERIVED_NO_PILEUP"
    assert ctx.sat_adu == 65535.0


def test_placed_aperture_faint_near_bright() -> None:
    """Faint star at placed position must not hijack to bright neighbour."""
    from astropy.wcs import WCS

    arr = np.full((128, 128), 2000.0, dtype=np.float64)
    arr[64, 80] = 55000.0
    arr[64, 64] = 6000.0
    arr[63:66, 63:66] = 6200.0
    arr[32, 32] = 20000.0
    arr[31:34, 31:34] = 21000.0
    w = WCS(naxis=2)
    w.wcs.crpix = [64.0, 64.0]
    w.wcs.crval = [180.0, 45.0]
    w.wcs.cd = [[1.0, 0.0], [0.0, 1.0]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = w.to_header()
    hdr["BITPIX"] = 16
    drift_ra, drift_dec = w.all_pix2world(32, 32, 0)
    ra = np.array([180.0, float(drift_ra)])
    de = np.array([45.0, float(drift_dec)])
    ax = np.array([64.0, 32.0])
    ay = np.array([64.0, 32.0])
    ap = np.array([6000.0, 20000.0])
    peaks, px, py, _drift = measure_raw_peaks_frame(
        arr,
        hdr,
        ra_deg=ra,
        dec_deg=de,
        aligned_x=ax,
        aligned_y=ay,
        aligned_hdr=hdr,
        aligned_peak=ap,
        drift_ref_ra=float(drift_ra),
        drift_ref_dec=float(drift_dec),
        drift_ref_catalog_id="drift_ref",
        catalog_ids=["faint", "drift_ref"],
    )
    assert math.isfinite(float(peaks[0]))
    assert float(peaks[0]) < 10000.0
    assert abs(float(px[0]) - 64.0) <= 2.5
    assert abs(float(py[0]) - 64.0) <= 1.5


def test_apply_raw_peaks_uses_placed_source() -> None:
    import pandas as pd
    from astropy.wcs import WCS

    arr = np.full((64, 64), 2500.0, dtype=np.float64)
    arr[32, 32] = 12000.0
    w = WCS(naxis=2)
    w.wcs.crpix = [32.0, 32.0]
    w.wcs.crval = [180.0, 45.0]
    w.wcs.cd = [[1.0, 0.0], [0.0, 1.0]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    hdr = w.to_header()
    hdr["BITPIX"] = 16
    df = pd.DataFrame(
        {
            "ra_deg": [180.0],
            "dec_deg": [45.0],
            "x": [32.0],
            "y": [32.0],
            "catalog_id": ["test"],
            "peak_max_adu_aligned": [11000.0],
        }
    )
    ctx = SatDiagContext(sat_adu=65535.0, lin_adu=52428.0, sat_source="DERIVED", lin_source="DEFAULT_FRAC")
    apply_raw_peaks_to_proc_df(df, arr, hdr, ctx, aligned_hdr=hdr)
    assert str(df["sat_peak_source"].iloc[0]) == SAT_PEAK_SOURCE_PLACED
    assert "peak_loc_ok" not in df.columns
    assert math.isfinite(float(df["peak_max_adu"].iloc[0]))


def test_commit_sat_diag_provenance_sets_raw_peaks_flag(tmp_path: Path) -> None:
    ctx = SatDiagContext(sat_adu=65535.0, lin_adu=52428.0, sat_source="DERIVED", lin_source="DEFAULT_FRAC")
    commit_sat_diag_provenance(ctx, tmp_path, placed_aperture_used=False)
    raw = (tmp_path / "sat_diag.json").read_text(encoding="utf-8")
    assert '"raw_peaks_used": false' in raw

    commit_sat_diag_provenance(ctx, tmp_path, placed_aperture_used=True)
    raw2 = (tmp_path / "sat_diag.json").read_text(encoding="utf-8")
    assert '"raw_peaks_used": true' in raw2
    assert SAT_PEAK_SOURCE_PLACED in raw2


def test_refuse_float_input(tmp_path: Path) -> None:
    d = np.ones((8, 8), dtype=np.float32)
    hdu = fits.PrimaryHDU(d)
    hdu.header["BITPIX"] = -32
    fp = tmp_path / "float.fits"
    hdu.writeto(fp, overwrite=True)
    r = derive_ceiling_from_paths([fp])
    assert r.refused
    assert r.refuse_reason == "REFUSE_NON_RAW"
