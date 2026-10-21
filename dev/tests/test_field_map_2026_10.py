"""FIELD-MAP-2026-10 (Milan 2026-10-21): JPEG field maps with an RA/Dec grid, scale bar and N/E arrows
that follow the field size (other telescope / camera)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src_py"
sys.path.insert(0, str(SRC))

from field_map import (  # noqa: E402
    FieldMapCanvas,
    TargetFieldMap,
    field_map_path,
    grid_steps,
    save_field_map,
    scale_bar_arcmin,
)


def _header(scale_arcsec: float, nx: int = 400, ny: int = 300, dec: float = 50.0):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [241.3, dec]
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    return w.to_header()


def test_grid_and_scale_follow_the_field() -> None:
    wide = grid_steps(340.0, 50.0)  # 5.7 deg field (Samyang-like)
    narrow = grid_steps(33.0, 50.0)  # 0.55 deg field (long focus)
    tiny = grid_steps(5.0, 50.0)
    assert wide[0] > narrow[0] > tiny[0] and wide[1] > narrow[1] > tiny[1]
    for fov in (5.0, 33.0, 340.0):
        d, _ = grid_steps(fov, 50.0)
        assert 2 <= fov / d <= 12
    assert grid_steps(60.0, 85.0)[1] > grid_steps(60.0, 0.0)[1]  # RA steps widen near the pole
    assert scale_bar_arcmin(340.0) == 60.0 and scale_bar_arcmin(33.0) == 5.0


def test_jpeg_with_wcs_and_without(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(1000.0, 20.0, size=(300, 400))
    t = pd.Series({"x": 200.0, "y": 150.0, "vsx_name": "T", "catalog_id": "1"})
    c = pd.DataFrame({"x": [100.0, 9999.0], "y": [100.0, 50.0]})  # one outside: not drawn, view unchanged
    for hdr, name in ((_header(2.0), "wcs.jpg"), (None, "nowcs.jpg")):
        fm = TargetFieldMap(data, hdr)
        out = fm.save(tmp_path / name, t, c)
        fm.close()
        assert out.read_bytes()[:3] == b"\xff\xd8\xff"  # JPEG
    canvas = FieldMapCanvas(data, _header(2.0))
    assert "field 0.22 x 0.17 deg" in canvas.subtitle
    assert canvas.ax.get_xlim() == (-0.5, 399.5)
    canvas.close()
    at = pd.DataFrame({"x": [200.0], "y": [150.0], "catalog_id": ["1"], "vsx_name": ["T"]})
    p = save_field_map(tmp_path / "field_map.jpg", data, _header(2.0), at, pd.DataFrame({"catalog_id": ["2"], "x": [50.0], "y": [60.0]}))
    assert p is not None and field_map_path(tmp_path) == tmp_path / "field_map.jpg"


def test_old_png_maps_are_still_found(tmp_path: Path) -> None:
    (tmp_path / "field_map_7.png").write_bytes(b"x")
    assert field_map_path(tmp_path, "7") == tmp_path / "field_map_7.png"
    (tmp_path / "field_map_7.jpg").write_bytes(b"x")
    assert field_map_path(tmp_path, "7") == tmp_path / "field_map_7.jpg"
