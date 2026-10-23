"""OUTPUT-REVIEW-2026-10 step 7 (Milan 2026-10-21): a new run replaces the products of the previous one."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src_py"
sys.path.insert(0, str(SRC))

from run_outputs import clear_export_dirs, clear_aperture_products, clear_report_cache  # noqa: E402


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x", encoding="ascii")
    return p


def test_aperture_start_removes_previous_products(tmp_path: Path) -> None:
    lc = tmp_path / "lightcurves"
    old = [_touch(lc / n) for n in ("lightcurve_1.csv", "lightcurve_1_psf.csv", "comp_quality_1.json",
                                     "check_kmag_1.csv", "field_map_1.png", "varcand_1.csv", "trust_1.json",
                                     "cutout_1.png", "lightcurve_1.png")]
    exports = [_touch(tmp_path / "lightcurves_reports" / s / "X_20260101.txt") for s in ("aavso", "varastro", "epsf")]
    keep = [_touch(tmp_path / "photometry_summary.csv"), _touch(tmp_path / "TESS" / "1" / "a.png")]
    out = clear_aperture_products(tmp_path)
    assert out == {"lightcurves": len(old), "exports": 3}
    assert not any(p.exists() for p in old + exports)
    assert all(p.exists() for p in keep)


def test_export_and_report_cache(tmp_path: Path) -> None:
    _touch(tmp_path / "lightcurves_reports" / "aavso" / "a.txt")
    epsf = _touch(tmp_path / "lightcurves_reports" / "epsf" / "b_epsf.txt")
    assert clear_export_dirs(tmp_path, ("aavso", "varastro")) == 1 and epsf.exists()
    _touch(tmp_path / "_report_cache" / "m3_cut_1.png")
    _touch(tmp_path / "_report_cache" / "pdf_embed" / "h_w900_q72.jpg")
    assert clear_report_cache(tmp_path) == 2
    assert not (tmp_path / "_report_cache" / "pdf_embed").exists()


def test_export_writes_measured_targets_only() -> None:
    s = (SRC / "export_reports.py").read_text(encoding="utf-8")
    assert "_nfr > 0" in s and 'clear_export_dirs(phot, ("aavso", "varastro"))' in s


def test_field_map_background_drawn_once_same_pixels(tmp_path: Path) -> None:
    from PIL import Image

    from field_map import TargetFieldMap

    data = np.random.default_rng(1).normal(1000.0, 30.0, size=(120, 160))
    t1 = pd.Series({"x": 40.0, "y": 50.0, "vsx_name": "A", "catalog_id": "1"})
    t2 = pd.Series({"x": 100.0, "y": 70.0, "vsx_name": "B", "catalog_id": "2"})
    c = pd.DataFrame({"x": [60.0, 90.0], "y": [60.0, 30.0]})
    fm = TargetFieldMap(data)
    fm.save(tmp_path / "a1.jpg", t1, c)
    fm.save(tmp_path / "b.jpg", t2, c.iloc[:1])
    fm.save(tmp_path / "a2.jpg", t1, c)  # markers of b removed: same picture as the first a
    fm.close()
    a1 = np.asarray(Image.open(tmp_path / "a1.jpg"))
    a2 = np.asarray(Image.open(tmp_path / "a2.jpg"))
    assert a1.shape == a2.shape and (a1 == a2).all()
