"""G2-F002: catalog_only placement must accept sibling-recovered MASTERSTAR WCS."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from photometry_core import (
    _masterstar_wcs_usable_for_placement,
    select_active_targets,
)


def _make_tan_wcs(ra: float, dec: float, scale_arcsec: float = 2.0) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [256.0, 256.0]
    w.wcs.crval = [float(ra), float(dec)]
    w.wcs.cd = np.array(
        [
            [-scale_arcsec / 3600.0, 0.0],
            [0.0, scale_arcsec / 3600.0],
        ]
    )
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _write_masterstar_fits(
    path: Path,
    wcs: WCS,
    *,
    vy_psolv: int | None = None,
    vy_sibl: str | None = None,
    size: int = 512,
) -> None:
    data = np.zeros((size, size), dtype=np.float32)
    hdr = wcs.to_header()
    if vy_psolv is not None:
        hdr["VY_PSOLV"] = vy_psolv
    if vy_sibl is not None:
        hdr["VY_SIBL"] = vy_sibl
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def _placement_fixtures(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """One unmatched VSX at field center; one distant masterstar match (not this target)."""
    ms_fits = tmp_path / "MASTERSTAR.fits"
    vt_csv = tmp_path / "variable_targets.csv"
    ms_csv = tmp_path / "masterstars_full_match.csv"

    ra_c, dec_c = 150.0, 45.0
    wcs = _make_tan_wcs(ra_c, dec_c)
    _write_masterstar_fits(ms_fits, wcs)

    vt = pd.DataFrame(
        [
            {
                "name": "VSX_UNMATCHED",
                "vsx_name": "VSX_UNMATCHED",
                "vsx_type": "EA",
                "vsx_period": "",
                "priority": 1,
                "ra_deg": ra_c,
                "dec_deg": dec_c,
                "x": 256.0,
                "y": 256.0,
                "catalog_id": "9999999999999999999",
                "gaia_match_quality": "good",
                "gaia_match_arcsec": 0.5,
                "mag": 12.5,
                "vsx_mag_max": 12.5,
            }
        ]
    )
    vt.to_csv(vt_csv, index=False)

    ms = pd.DataFrame(
        [
            {
                "name": "8888888888888888888",
                "catalog_id": "8888888888888888888",
                "ra_deg": 140.0,
                "dec_deg": 35.0,
                "x": 100.0,
                "y": 100.0,
                "mag": 11.0,
                "zone": "linear",
                "is_usable": True,
                "is_saturated": False,
                "is_noisy": False,
                "snr50_ok": True,
            }
        ]
    )
    ms.to_csv(ms_csv, index=False)
    return ms_fits, vt_csv, ms_csv, tmp_path


def test_masterstar_wcs_usable_predicate() -> None:
    wcs = _make_tan_wcs(10.0, 20.0)
    hdr_solved = wcs.to_header()
    hdr_solved["VY_PSOLV"] = 1
    assert _masterstar_wcs_usable_for_placement(hdr_solved, wcs)

    hdr_sibl = wcs.to_header()
    hdr_sibl["VY_SIBL"] = "g_60_4"
    assert _masterstar_wcs_usable_for_placement(hdr_sibl, wcs)

    hdr_stale = wcs.to_header()
    assert not _masterstar_wcs_usable_for_placement(hdr_stale, wcs)

    wcs_lin = WCS(naxis=2)
    wcs_lin.wcs.ctype = ["LINEAR", "LINEAR"]
    assert not _masterstar_wcs_usable_for_placement(wcs_lin.to_header(), wcs_lin)


def test_sibling_recovered_wcs_places_catalog_only_targets(tmp_path: Path) -> None:
    ms_fits, vt_csv, ms_csv, _ = _placement_fixtures(tmp_path)
    _write_masterstar_fits(ms_fits, _make_tan_wcs(150.0, 45.0), vy_sibl="g_60_4")

    out = select_active_targets(
        vt_csv,
        ms_csv,
        frame_w_px=512,
        frame_h_px=512,
        edge_margin_px=50,
        safe_bbox=(50.0, 50.0, 450.0, 450.0),
        masterstar_fits_path=ms_fits,
        plate_scale_arcsec_px=2.0,
    )
    co = out[out["zone_flag"] == "catalog_only"]
    assert len(co) == 1
    row = co.iloc[0]
    assert row["catalog_id"] == "9999999999999999999"
    assert np.isfinite(float(row["x"])) and np.isfinite(float(row["y"]))
    assert abs(float(row["x"]) - 256.0) < 2.0
    assert abs(float(row["y"]) - 256.0) < 2.0


def test_independently_solved_wcs_places_catalog_only_as_before(tmp_path: Path) -> None:
    ms_fits, vt_csv, ms_csv, _ = _placement_fixtures(tmp_path)
    _write_masterstar_fits(ms_fits, _make_tan_wcs(150.0, 45.0), vy_psolv=1)

    out = select_active_targets(
        vt_csv,
        ms_csv,
        frame_w_px=512,
        frame_h_px=512,
        edge_margin_px=50,
        safe_bbox=(50.0, 50.0, 450.0, 450.0),
        masterstar_fits_path=ms_fits,
        plate_scale_arcsec_px=2.0,
    )
    co = out[out["zone_flag"] == "catalog_only"]
    assert len(co) == 1
    assert co.iloc[0]["catalog_id"] == "9999999999999999999"
    assert np.isfinite(float(co.iloc[0]["x"]))


def test_stale_wcs_skips_catalog_only_placement(tmp_path: Path) -> None:
    ms_fits, vt_csv, ms_csv, _ = _placement_fixtures(tmp_path)
    _write_masterstar_fits(ms_fits, _make_tan_wcs(150.0, 45.0))

    out = select_active_targets(
        vt_csv,
        ms_csv,
        frame_w_px=512,
        frame_h_px=512,
        edge_margin_px=50,
        safe_bbox=(50.0, 50.0, 450.0, 450.0),
        masterstar_fits_path=ms_fits,
        plate_scale_arcsec_px=2.0,
    )
    assert (out["zone_flag"] == "catalog_only").sum() == 0
