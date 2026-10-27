"""FAIL-LOUD-2026-10 (code review 2, T05): required steps raise; optional diagnostics log WARNING.

Each test pins one place where an error used to be swallowed (status ok with a missing result).
"""

from __future__ import annotations

import ast
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from osc_extract import effective_gain_rn  # noqa: E402
from photometry_core import merge_photometry_pipeline_meta  # noqa: E402


# --- G11-07: pipeline_meta.json -------------------------------------------------------------------------


def test_pipeline_meta_corrupt_file_is_kept_and_raises(tmp_path: Path) -> None:
    meta = tmp_path / "pipeline_meta.json"
    meta.write_text("{not json", encoding="ascii")
    with pytest.raises(RuntimeError, match="unreadable"):
        merge_photometry_pipeline_meta(tmp_path, {"k": 1})
    assert (tmp_path / "pipeline_meta.json.corrupt").read_text(encoding="ascii") == "{not json"
    assert not meta.exists()


def test_pipeline_meta_numpy_values_and_atomic_write(tmp_path: Path) -> None:
    merge_photometry_pipeline_meta(tmp_path, {"a": np.float32(1.5), "b": np.int64(3), "c": np.arange(2)})
    merge_photometry_pipeline_meta(tmp_path, {"d": "x"})
    out = json.loads((tmp_path / "pipeline_meta.json").read_text(encoding="utf-8"))
    assert out == {"a": 1.5, "b": 3, "c": [0, 1], "d": "x"}
    assert not (tmp_path / "pipeline_meta.json.tmp").exists()


# --- G01-40: OSC effective gain / read noise -------------------------------------------------------------


def test_effective_gain_rn_unknown_stays_nan() -> None:
    g, rn = effective_gain_rn(float("nan"), float("nan"), "G", 1)
    assert math.isnan(g) and math.isnan(rn)
    g, rn = effective_gain_rn(1.0, float("nan"), "oneRGGB", 2)
    assert g == pytest.approx(16.0) and math.isnan(rn)


def test_effective_gain_rn_invalid_known_value_raises() -> None:
    with pytest.raises(ValueError, match="gain"):
        effective_gain_rn(0.0, 3.0, "R", 1)
    with pytest.raises(ValueError, match="read_noise"):
        effective_gain_rn(1.0, -1.0, "R", 1)


# --- G03-06: VSX export is a required step -----------------------------------------------------------------


def _masterstar(ps: Path, *, celestial: bool) -> tuple[Path, Path]:
    ps.mkdir(parents=True, exist_ok=True)
    ms_csv = ps / "masterstars_full_match.csv"
    ms_csv.write_text("catalog_id,ra_deg,dec_deg,mag,x,y\n100,180.0,45.0,12.0,32.0,32.0\n", encoding="ascii")
    hdr = fits.Header()
    hdr["NAXIS"], hdr["NAXIS1"], hdr["NAXIS2"] = 2, 64, 64
    hdr["VY_FWHM"] = 3.5
    w = WCS(naxis=2)
    if celestial:
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [32.0, 32.0]
    w.wcs.crval = [180.0, 45.0]
    w.wcs.cd = [[0.0001, 0.0], [0.0, 0.0001]]
    hdr.update(w.to_header())
    ms_fits = ps / "MASTERSTAR.fits"
    fits.writeto(ms_fits, np.zeros((64, 64), dtype=np.float32), header=hdr, overwrite=True)
    return ms_fits, ms_csv


def test_vsx_export_error_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The old code logged any non-VSXCatalogError and continued: the run ended 'ok' with no targets."""
    import sqlite3

    from pipeline_astrometry import write_photometry_plan_files

    ps = tmp_path / "Drafts" / "draft_000001" / "platesolve" / "NoFilter_60_2"
    ms_fits, ms_csv = _masterstar(ps, celestial=False)
    vsx = tmp_path / "vsx.db"
    con = sqlite3.connect(str(vsx))
    con.execute("CREATE TABLE vsx_data (oid INTEGER PRIMARY KEY, ra_deg REAL, dec_deg REAL)")
    con.execute("CREATE TABLE exoplanet_data (obj_id TEXT PRIMARY KEY, ra_deg REAL, dec_deg REAL)")
    con.execute("INSERT INTO vsx_data VALUES (1, 180.0, 45.0)")
    con.commit()
    con.close()

    class _Cfg:
        objects_db_path = str(vsx)
        aperture_fwhm_factor = 1.35
        annulus_inner_fwhm = 2.7
        annulus_outer_fwhm = 5.2
        gaia_db_path = ""

    monkeypatch.setattr("config.AppConfig", lambda *a, **k: _Cfg())
    with pytest.raises(RuntimeError, match="VSX export"):
        write_photometry_plan_files(platesolve_dir=ps, masterstar_fits=ms_fits, masterstars_csv=ms_csv)


# --- G10-09: no process-wide warnings filter at import -----------------------------------------------------


def test_catalog_crossmatch_has_no_module_level_warnings_filter() -> None:
    tree = ast.parse((ROOT / "src_py" / "catalog_crossmatch.py").read_text(encoding="utf-8"))
    for node in tree.body:
        for sub in ast.walk(node) if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) else ():
            if isinstance(sub, ast.Attribute) and sub.attr in ("simplefilter", "filterwarnings"):
                pytest.fail(f"module-level warnings.{sub.attr} at line {sub.lineno}")


# --- G10-16: a SIMBAD failure for one star is not cached as 'no SIMBAD entry' ------------------------------


def test_simbad_per_star_failure_not_cached(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import types

    import hrd_enrich

    class _Simbad:
        def __init__(self) -> None:
            pass

        def add_votable_fields(self, *a: object) -> None:
            pass

        def query_object(self, ident: str) -> None:
            raise ConnectionError("server down")

    mod = types.ModuleType("astroquery.simbad")
    mod.Simbad = _Simbad  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "astroquery.simbad", mod)
    with caplog.at_level(logging.WARNING):
        hits = hrd_enrich._fetch_simbad(["1234567890123456789"], timeout_s=5.0)
    assert "1234567890123456789" not in hits
    assert any("not cached" in r.getMessage() for r in caplog.records)
