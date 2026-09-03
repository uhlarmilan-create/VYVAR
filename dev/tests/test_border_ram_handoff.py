# -*- coding: ascii -*-
"""Border filter safe-bbox from RAM-handoff aligned frames."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

_SRC_PY = Path(__file__).resolve().parents[2] / "src_py"
if str(_SRC_PY) not in sys.path:
    sys.path.insert(0, str(_SRC_PY))

from pipeline import write_photometry_plan_files  # noqa: E402
from photometry_core import common_field_intersection_bbox_px_from_arrays  # noqa: E402


def _make_masterstar(ps: Path, *, size: int = 512) -> tuple[Path, Path]:
    ps.mkdir(parents=True, exist_ok=True)
    ms_csv = ps / "masterstars_full_match.csv"
    ms_csv.write_text(
        "catalog_id,ra_deg,dec_deg,mag,x,y\n"
        f"100,180.0,45.0,12.0,{size // 2}.0,{size // 2}.0\n",
        encoding="ascii",
    )
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = size
    hdr["NAXIS2"] = size
    hdr["VY_FWHM"] = 3.5
    w = WCS(naxis=2)
    w.wcs.crpix = [float(size // 2), float(size // 2)]
    w.wcs.crval = [180.0, 45.0]
    w.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
    hdr.update(w.to_header())
    ms_fits = ps / "MASTERSTAR.fits"
    fits.writeto(ms_fits, np.zeros((size, size), dtype=np.float32), header=hdr, overwrite=True)
    return ms_fits, ms_csv


def test_intersection_bbox_from_ram_arrays() -> None:
    a1 = np.full((64, 64), np.nan, dtype=np.float32)
    a1[8:56, 8:56] = 100.0
    a2 = np.full((64, 64), np.nan, dtype=np.float32)
    a2[12:52, 12:52] = 100.0
    bb = common_field_intersection_bbox_px_from_arrays(frame_arrays=[a1, a2], finite_stride=4)
    assert bb is not None
    x0, y0, x1, y1 = bb
    assert x0 >= 12.0
    assert y0 >= 12.0
    assert x1 <= 51.0
    assert y1 <= 51.0


def test_require_safe_bbox_from_ram_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ps = tmp_path / "Drafts" / "draft_000503" / "platesolve" / "NoFilter_60_2"
    ms_fits, ms_csv = _make_masterstar(ps)

    class _Cfg:
        vsx_local_db_path = ""
        aperture_fwhm_factor = 1.35
        annulus_inner_fwhm = 2.7
        annulus_outer_fwhm = 5.2
        gaia_db_path = ""
        exoplanet_local_db_path = ""

    monkeypatch.setattr("config.AppConfig", lambda *a, **k: _Cfg())

    a1 = np.full((512, 512), np.nan, dtype=np.float32)
    a1[64:448, 64:448] = 100.0
    a2 = np.full((512, 512), np.nan, dtype=np.float32)
    a2[96:416, 96:416] = 100.0
    ram = [("proc_001.fits", fits.Header(), a1), ("proc_002.fits", fits.Header(), a2)]

    out = write_photometry_plan_files(
        platesolve_dir=ps,
        masterstar_fits=ms_fits,
        masterstars_csv=ms_csv,
        aligned_ram_frames=ram,
        require_safe_bbox=True,
    )
    plan = json.loads((ps / "photometry_plan.json").read_text(encoding="utf-8"))
    assert plan.get("safe_bbox_px") is not None
    assert out.get("comparison_stars_csv")


def test_require_safe_bbox_raises_without_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ps = tmp_path / "Drafts" / "draft_000001" / "platesolve" / "NoFilter_60_2"
    ms_fits, ms_csv = _make_masterstar(ps)

    class _Cfg:
        vsx_local_db_path = ""
        aperture_fwhm_factor = 1.35
        annulus_inner_fwhm = 2.7
        annulus_outer_fwhm = 5.2
        gaia_db_path = ""
        exoplanet_local_db_path = ""

    monkeypatch.setattr("config.AppConfig", lambda *a, **k: _Cfg())

    with pytest.raises(RuntimeError, match="Post-alignment border filter"):
        write_photometry_plan_files(
            platesolve_dir=ps,
            masterstar_fits=ms_fits,
            masterstars_csv=ms_csv,
            require_safe_bbox=True,
        )
