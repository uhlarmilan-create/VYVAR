"""ePSF candidate selection from masterstars_full_match.csv only (MS-SOURCES-RETIRE C1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from database import VyvarDatabase
from psf_photometry import _epsf_prepare_stars


def _write_min_masterstar(path: Path, *, n: int = 512) -> None:
    data = np.zeros((n, n), dtype=np.float32)
    hdr = fits.Header()
    hdr["NAXIS1"] = n
    hdr["NAXIS2"] = n
    hdr["VY_FWHM"] = 3.5
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def _write_min_csv(path: Path, *, with_xy: bool = True, drop_col: str | None = None) -> None:
    rows = []
    for i in range(40):
        row = {
            "catalog_id": f"GaiaDR3_{1000000000000000000 + i}",
            "catalog_known_variable": False,
            "likely_saturated": False,
            "photometry_ok": True,
            "is_saturated": False,
            "is_noisy": False,
            "is_usable": True,
            "ra_deg": 180.0 + i * 0.001,
            "dec_deg": 45.0,
            "mag": 12.0,
            "zone": "linear",
            "source_state": "DETECTED_P1",
        }
        if with_xy:
            row["x"] = 50.0 + i * 10.0
            row["y"] = 50.0 + i * 8.0
        rows.append(row)
    df = pd.DataFrame(rows)
    if drop_col and drop_col in df.columns:
        df = df.drop(columns=[drop_col])
    df.to_csv(path, index=False)


def _synthetic_field(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    """600x600 MASTERSTAR with a TAN WCS: 64 isolated stars on a 70 px grid + 6 faint stars with a
    4 mag brighter neighbour 6.5 px away. Returns (MASTERSTAR, csv, ids of the blended faint stars)."""
    from astropy.wcs import WCS
    from photutils.psf import CircularGaussianPRF

    n, fwhm = 600, 2.5
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [180.0, 45.0]
    w.wcs.crpix = [300.0, 300.0]
    w.wcs.cdelt = [-2.0 / 3600.0, 2.0 / 3600.0]
    rng = np.random.default_rng(4)
    rows, img = [], np.zeros((n, n))
    yy, xx = np.mgrid[0:n, 0:n]

    def add(cid: str, x: float, y: float, mag: float) -> None:
        flux = 10 ** (-0.4 * (mag - 22.0))
        nonlocal img
        img += CircularGaussianPRF(flux=flux, x_0=x, y_0=y, fwhm=fwhm)(xx, yy)
        ra, dec = w.all_pix2world([[x, y]], 0)[0]
        rows.append({"catalog_id": cid, "ra_deg": ra, "dec_deg": dec, "mag": mag, "x": x, "y": y})

    k = 0
    for gy in range(8):
        for gx in range(8):
            add(f"{1000000000000000000 + k}", 55.0 + 70 * gx + rng.uniform(-0.5, 0.5), 55.0 + 70 * gy + rng.uniform(-0.5, 0.5),
                11.0 + 2.0 * rng.random())
            k += 1
    blended = []
    for j in range(6):
        x0, y0 = 90.0 + 70 * j, 90.0
        add(f"{2000000000000000000 + j}", x0, y0, 12.5)
        add(f"{3000000000000000000 + j}", x0 + 6.5, y0, 8.5)
        blended.append(f"{2000000000000000000 + j}")
    img = rng.poisson(img + 300.0).astype(np.float32)
    hdr = w.to_header()
    hdr["VY_FWHM"] = fwhm
    ms = tmp_path / "MASTERSTAR.fits"
    fits.PrimaryHDU(img, header=hdr).writeto(ms, overwrite=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["x", "y"]).to_csv(tmp_path / "field_catalog_cone.csv", index=False)
    peak = 10 ** (-0.4 * (df["mag"] - 22.0)) / (1.1331 * fwhm**2)
    df = df.assign(
        catalog_known_variable=False, likely_saturated=peak > 0.6 * 65535, photometry_ok=True,
        is_saturated=peak > 65535, is_noisy=False, is_usable=True, zone="linear", source_state="DETECTED_P1",
        peak_max_adu=peak + 300.0, saturate_limit_adu=65535.0,
    )
    csv = tmp_path / "masterstars_full_match.csv"
    df.to_csv(csv, index=False)
    (tmp_path / "photometry").mkdir()
    (tmp_path / "photometry" / "pipeline_meta.json").write_text('{"frame_depth": {"g_lim_frame": 15.0}}')
    return ms, csv, blended


def test_build_stars_isolated_and_model_covers_the_wings(tmp_path: Path) -> None:
    """Build stars: Gaia-isolated (3xFWHM, delta mag 2.5); the cutout / model is 12xFWHM so a bright
    neighbour's wing is modelled in the joint fit (EPSF-BLEND-2026-10)."""
    ms, csv, blended = _synthetic_field(tmp_path)
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    try:
        prep = _epsf_prepare_stars(ms, csv, db, draft_id=1, min_stars=10)
    finally:
        db.close()
    kept = set(prep["psf_stars_df"]["catalog_id"].astype(str))
    assert prep["cutout_size"] == 31  # 12 x FWHM 2.5, odd
    assert not kept & set(blended)
    assert len(kept) >= 40


def test_epsf_prepare_missing_csv(tmp_path: Path) -> None:
    ms = tmp_path / "MASTERSTAR.fits"
    _write_min_masterstar(ms)
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    try:
        with pytest.raises(FileNotFoundError, match="masterstars_full_match.csv not found"):
            _epsf_prepare_stars(ms, tmp_path / "masterstars_full_match.csv", db, draft_id=1, min_stars=10)
    finally:
        db.close()


def test_epsf_prepare_missing_required_column(tmp_path: Path) -> None:
    ms = tmp_path / "MASTERSTAR.fits"
    csv = tmp_path / "masterstars_full_match.csv"
    _write_min_masterstar(ms)
    _write_min_csv(csv, drop_col="x")
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            _epsf_prepare_stars(ms, csv, db, draft_id=1, min_stars=10)
    finally:
        db.close()


def test_psf_photometry_has_no_master_sources_reads() -> None:
    text = Path(__file__).resolve().parents[2] / "src_py" / "psf_photometry.py"
    src = text.read_text(encoding="utf-8")
    assert "FROM MASTER_SOURCES" not in src
    assert "SELECT COUNT(*) AS n FROM MASTER_SOURCES" not in src
