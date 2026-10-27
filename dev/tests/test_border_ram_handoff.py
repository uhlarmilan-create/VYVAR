# -*- coding: ascii -*-
"""Border filter safe-bbox: overlap of the aligned frames' data footprints (BORDER-FOOTPRINT-2026-10)."""

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
from photometry_core import (  # noqa: E402
    common_field_intersection_bbox_px,
    common_field_intersection_bbox_px_from_headers,
)


def _make_vsx_db(path: Path, rows: list[tuple[int, float, float]]) -> None:
    import sqlite3

    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE vsx_data (oid INTEGER PRIMARY KEY, ra_deg REAL, dec_deg REAL)")
    con.execute("CREATE TABLE exoplanet_data (obj_id TEXT PRIMARY KEY, ra_deg REAL, dec_deg REAL)")
    con.executemany("INSERT INTO vsx_data (oid, ra_deg, dec_deg) VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()


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
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # a real MASTERSTAR WCS is celestial
    w.wcs.crpix = [float(size // 2), float(size // 2)]
    w.wcs.crval = [180.0, 45.0]
    w.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
    hdr.update(w.to_header())
    ms_fits = ps / "MASTERSTAR.fits"
    fits.writeto(ms_fits, np.zeros((size, size), dtype=np.float32), header=hdr, overwrite=True)
    return ms_fits, ms_csv


def _aligned_header(dx: float, dy: float, *, size: int = 512, rot_deg: float = 0.0) -> fits.Header:
    """Header of an aligned frame whose frame->reference transform shifts by (dx, dy)."""
    import math

    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    h = fits.Header()
    h["NAXIS1"], h["NAXIS2"] = size, size
    for k, v in enumerate((c, -s, dx, s, c, dy, 0.0, 0.0, 1.0)):
        h[f"VY_AM{k}"] = v
    return h


def test_overlap_is_the_intersection_of_the_shifted_footprints() -> None:
    hs = [(_aligned_header(0, 0, size=64), (64, 64)), (_aligned_header(-10, 6, size=64), (64, 64))]
    x0, y0, x1, y1 = common_field_intersection_bbox_px_from_headers(headers=hs)
    assert (x0, y0) == (1.0, 7.0)  # frame 2 starts 6 px down; 1 px inside the edges
    assert (x1, y1) == (52.0, 62.0)  # frame 2 ends 10 px left


def test_constant_fill_does_not_hide_the_shift(tmp_path: Path) -> None:
    """MN Boo 538: the aligner fills uncovered pixels with a constant; the overlap must come from the
    transform, not from the pixel values, and the frames keep their raw names (no proc_ prefix)."""
    paths = []
    for name, (dx, dy) in (("MN_Boo_Light_116.fits", (-80.4, -9.8)), ("MN_Boo_Light_143.fits", (0.0, 0.0))):
        fp = tmp_path / name
        fits.writeto(fp, np.full((512, 512), 1240.0, dtype=np.float32), header=_aligned_header(dx, dy))
        paths.append(fp)
    x0, y0, x1, y1 = common_field_intersection_bbox_px(frame_paths=paths)
    assert x1 <= 512 - 80.4 and y1 <= 512 - 9.8
    assert x0 == 1.0 and y0 == 1.0


def test_rotated_frame_footprint_stays_inside() -> None:
    hs = [(_aligned_header(0, 0), (512, 512)), (_aligned_header(0, 0, rot_deg=1.5), (512, 512))]
    x0, y0, x1, y1 = common_field_intersection_bbox_px_from_headers(headers=hs)
    assert y0 > 13.0 and x1 < 498.0  # the rotated corners cut the overlap
    from frame_footprint import aligned_footprint

    inside = aligned_footprint(hs[1][0], (512, 512))
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        assert inside[int(np.ceil(y)) if y == y0 else int(y), int(np.ceil(x)) if x == x0 else int(x)]


def test_missing_transform_raises() -> None:
    with pytest.raises(ValueError, match="transform"):
        common_field_intersection_bbox_px_from_headers(headers=[(fits.Header(), (64, 64))] * 2)


def test_require_safe_bbox_from_ram_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ps = tmp_path / "Drafts" / "draft_000503" / "platesolve" / "NoFilter_60_2"
    ms_fits, ms_csv = _make_masterstar(ps)
    # Fail-loud VSX export (FAIL-LOUD-2026-10) needs a real objects DB; one star outside the field.
    vsx = tmp_path / "vsx.db"
    _make_vsx_db(vsx, [(1, 0.0, 0.0)])

    class _Cfg:
        objects_db_path = str(vsx)
        aperture_fwhm_factor = 1.35
        annulus_inner_fwhm = 2.7
        annulus_outer_fwhm = 5.2
        gaia_db_path = ""

    monkeypatch.setattr("config.AppConfig", lambda *a, **k: _Cfg())

    a = np.full((512, 512), 100.0, dtype=np.float32)
    ram = [("f001.fits", _aligned_header(0, 0), a), ("f002.fits", _aligned_header(-40, 30), a)]

    out = write_photometry_plan_files(
        platesolve_dir=ps,
        masterstar_fits=ms_fits,
        masterstars_csv=ms_csv,
        aligned_ram_frames=ram,
        require_safe_bbox=True,
    )
    plan = json.loads((ps / "photometry_plan.json").read_text(encoding="utf-8"))
    sb = plan.get("safe_bbox_px")
    assert sb is not None and sb[1] > 30.0 and sb[2] < 512 - 40.0
    assert out.get("comparison_stars_csv")


def test_require_safe_bbox_raises_without_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ps = tmp_path / "Drafts" / "draft_000001" / "platesolve" / "NoFilter_60_2"
    ms_fits, ms_csv = _make_masterstar(ps)

    class _Cfg:
        objects_db_path = ""
        aperture_fwhm_factor = 1.35
        annulus_inner_fwhm = 2.7
        annulus_outer_fwhm = 5.2
        gaia_db_path = ""

    monkeypatch.setattr("config.AppConfig", lambda *a, **k: _Cfg())

    with pytest.raises(RuntimeError, match="Post-alignment border filter"):
        write_photometry_plan_files(
            platesolve_dir=ps,
            masterstar_fits=ms_fits,
            masterstars_csv=ms_csv,
            require_safe_bbox=True,
        )
