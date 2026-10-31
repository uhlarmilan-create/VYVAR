"""Step 5 batch A (tag bug): real bugs found in the step 5 triage (2026-10-30)."""

from __future__ import annotations

import glob
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2] / "src_py"


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


# --------------------------------------------------------------------------------------------------------------
# G02-05 / G13-31: one FITS suffix rule
# --------------------------------------------------------------------------------------------------------------


def _old_iter_fits_paths_recursive(root: Path) -> list[Path]:
    """HEAD 08e985a copy (reference for the .fits-only equality case)."""
    from fits_suffixes import path_suffix_is_fits

    root = Path(root)
    if not root.exists():
        return []
    root_s = os.path.normpath(str(root.resolve()))
    pats = [os.path.join(root_s, "**", "*" + e) for e in (".fits", ".fit", ".fts", ".FITS", ".FIT", ".FTS")]
    seen: set[str] = set()
    out: list[Path] = []
    for pat in pats:
        for p in glob.glob(pat, recursive=True):
            fp = Path(p)
            if not fp.is_file() or not path_suffix_is_fits(fp):
                continue
            key = str(fp.resolve()).casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(fp.resolve())
    return sorted(out)


def test_list_fits_files_all_suffixes(tmp_path: Path) -> None:
    from fits_suffixes import list_fits_files

    for n in ("a.fit", "b.fts", "c.FITS", "d.Fits", "e.fits", "notes.txt", ".hidden.fits", "f.fits.gz"):
        _touch(tmp_path / n)
    _touch(tmp_path / "sub" / "g.FIT")
    _touch(tmp_path / ".git" / "h.fits")
    names = [p.name for p in list_fits_files(tmp_path)]
    assert names == ["a.fit", "b.fts", "c.FITS", "d.Fits", "e.fits"]
    rnames = [p.relative_to(tmp_path).as_posix() for p in list_fits_files(tmp_path, recursive=True)]
    assert rnames == ["a.fit", "b.fts", "c.FITS", "d.Fits", "e.fits", "sub/g.FIT"]
    assert list_fits_files(tmp_path / "missing") == []


@pytest.mark.skipif(os.path.normcase("A") == "a", reason="case-insensitive OS")
def test_iter_fits_recursive_keeps_case_distinct_names(tmp_path: Path) -> None:
    """G13-31: the old finder missed y.Fits and dropped X.fits next to x.fits on Linux."""
    from utils import iter_fits_paths_recursive

    for n in ("x.fits", "X.fits", "y.Fits", "z.fit"):
        _touch(tmp_path / "lights" / n)
    got = sorted(p.name for p in iter_fits_paths_recursive(tmp_path))
    assert got == ["X.fits", "x.fits", "y.Fits", "z.fit"]
    old = sorted(p.name for p in _old_iter_fits_paths_recursive(tmp_path))
    assert old != got  # the old code was wrong here


def test_iter_fits_recursive_equal_on_fits_only_tree(tmp_path: Path) -> None:
    from utils import iter_fits_paths_recursive

    for i in range(7):
        _touch(tmp_path / "V_60s_1x1" / f"light_{i:03d}.fits")
        _touch(tmp_path / "B_60s_1x1" / "deeper" / f"light_{i:03d}.fits")
    _touch(tmp_path / "qc_metrics.csv")
    assert iter_fits_paths_recursive(tmp_path) == _old_iter_fits_paths_recursive(tmp_path)


@pytest.mark.parametrize("suffix", [".fit", ".fts", ".FITS"])
def test_sat_diag_sample_light_paths_finds_non_fits_suffix(tmp_path: Path, suffix: str) -> None:
    from sat_diag import sample_light_paths

    for i in range(3):
        _touch(tmp_path / "Raw" / "lights" / f"l{i}{suffix}")
    got = sample_light_paths(tmp_path)
    assert [p.name for p in got] == [f"l{i}{suffix}" for i in range(3)]


def test_draft_provenance_has_fits_any_suffix(tmp_path: Path) -> None:
    from draft_provenance import _has_fits

    _touch(tmp_path / "lights" / "a.FTS")
    assert _has_fits(tmp_path / "lights")
    assert not _has_fits(tmp_path / "nothing")


def test_lc_outlier_frame_key_for_fit_suffix() -> None:
    from lc_outlier import frame_key_from_source

    assert frame_key_from_source("light_001.fit") == "light_001"
    assert frame_key_from_source("light_001.FTS") == "light_001"
    assert frame_key_from_source("light_001.fits") == "light_001"
    assert frame_key_from_source("proc_light_001.csv") == "light_001"


def test_no_fits_only_globs_in_src_py() -> None:
    # a frame listing glob; named VYVAR products (MASTERSTAR*.fits) are written by VYVAR with .fits
    pat = re.compile(r"""glob\(\s*f?["'](\*\*/)?\*\.(fits|fit|fts|FITS|FIT|FTS)["']""")
    tup = re.compile(r"""\(\s*["']\.fits["']\s*,\s*["']\.fit["']""")
    bad: list[str] = []
    for p in sorted(_SRC.glob("*.py")):
        if p.name == "fits_suffixes.py":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line) or tup.search(line):
                bad.append(f"{p.name}:{i}: {line.strip()}")
    assert not bad, "\n".join(bad)


# --------------------------------------------------------------------------------------------------------------
# G02-39 / G09-25: one bool coercion
# --------------------------------------------------------------------------------------------------------------

_TRUE_VALUES = [True, np.True_, 1, 2, -1, 1.0, np.float64(1.0), np.int64(1), np.float32(0.5), float("inf"),
                "1", "1.0", "true", "True", " TRUE ", "yes", "Yes", "y", "t", "on", "2"]
_FALSE_VALUES = [False, np.False_, 0, 0.0, np.float64(0.0), np.int64(0), "0", "0.0", "false", "False", "no",
                 "n", "f", "off"]
_UNKNOWN_VALUES = [None, float("nan"), np.nan, "", "  ", "nan", "maybe"]


@pytest.mark.parametrize("v", _TRUE_VALUES)
def test_coerce_bool_true(v) -> None:
    from stats_core import coerce_bool, parse_bool

    assert parse_bool(v) is True
    assert coerce_bool(v) is True and coerce_bool(v, default=False) is True


@pytest.mark.parametrize("v", _FALSE_VALUES)
def test_coerce_bool_false(v) -> None:
    from stats_core import coerce_bool, parse_bool

    assert parse_bool(v) is False
    assert coerce_bool(v, default=True) is False


@pytest.mark.parametrize("v", _UNKNOWN_VALUES)
def test_coerce_bool_unknown(v) -> None:
    from stats_core import coerce_bool, parse_bool

    assert parse_bool(v) is None
    assert coerce_bool(v) is False and coerce_bool(v, default=True) is True


def test_coerce_bool_pandas_na() -> None:
    import pandas as pd

    from stats_core import parse_bool

    assert parse_bool(pd.NA) is None


def test_old_stats_core_coerce_bool_read_float_one_as_false() -> None:
    """The HEAD copy (psf_internal_lc / epsf_science_set) read 1.0 and '1.0' as False."""

    def old(raw) -> bool:
        import math

        if isinstance(raw, bool):
            return bool(raw)
        if raw is None or (isinstance(raw, float) and not math.isfinite(raw)):
            return False
        return str(raw).strip().lower() in ("1", "true", "t", "yes", "y")

    from stats_core import coerce_bool

    assert old(1.0) is False and coerce_bool(1.0) is True
    assert old("1.0") is False and coerce_bool("1.0") is True


def _series_cases():
    import pandas as pd

    rng = np.random.default_rng(5)
    b = rng.random(500) < 0.3
    nan_mask = rng.random(500) < 0.1
    obj = pd.Series(b, dtype=object)
    obj[nan_mask] = np.nan
    flt = pd.Series(b.astype(float))
    flt[nan_mask] = np.nan
    nullable = pd.Series(b, dtype="boolean")
    nullable[nan_mask] = pd.NA
    return {"bool": pd.Series(b), "object_bool_nan": obj, "float_nan": flt, "int": pd.Series(b.astype(int)),
            "nullable": nullable}


@pytest.mark.parametrize("name", ["bool", "object_bool_nan", "float_nan", "int", "nullable"])
@pytest.mark.parametrize("default", [False, True])
def test_coerce_bool_series_equals_fillna_astype_bool(name: str, default: bool) -> None:
    """Byte-identity: on bool / numeric / bool-with-NaN columns the new series coercion equals the old
    ``fillna(default).astype(bool)``."""
    from stats_core import coerce_bool_series

    s = _series_cases()[name]
    old = s.fillna(default).astype(bool)
    new = coerce_bool_series(s, default=default)
    assert new.dtype == bool
    assert new.index.equals(s.index)
    assert new.tolist() == old.tolist()


def test_coerce_bool_series_text_column() -> None:
    import pandas as pd

    from stats_core import coerce_bool_series

    s = pd.Series(["True", "False", "1.0", "0.0", "", "yes", "no"], dtype=object)
    assert coerce_bool_series(s).tolist() == [True, False, True, False, False, True, False]
    assert s.fillna(False).astype(bool).tolist()[1] is True  # the old reading of the text "False"



# --------------------------------------------------------------------------------------------------------------
# G01-14: HFR QC sign flip - moot after the step-5 merge: s5_sci G01-16 deleted the Layer A HFR QC
# (_mean_hfr_bright_stars_dao); FWHM is measured once at calibration QC (test_step5_sci_2026_10.py).
# --------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------
# G12-19: one processed_field_root
# --------------------------------------------------------------------------------------------------------------


def _precal_field(tmp_path: Path) -> Path:
    import json

    root = tmp_path / "Archive" / "Drafts" / "BO_CVn_20261001_001"
    (root / "non_calibrated" / "lights" / "r").mkdir(parents=True)
    (root / "platesolve" / "r" / "photometry").mkdir(parents=True)
    (root / "draft_manifest.json").write_text(json.dumps({"draft_id": 7}), encoding="ascii")
    (root / "sat_diag.json").write_text(json.dumps({"sat_adu": 60000.0}), encoding="ascii")
    return root


def test_processed_field_root_pre_calibrated(tmp_path: Path) -> None:
    from processed_field import processed_field_root

    root = _precal_field(tmp_path)
    for sub in ("platesolve/r/photometry", "platesolve/r", "non_calibrated/lights/r", "."):
        assert processed_field_root(root / sub) == root.resolve()


def test_invariants_field_files_found_on_pre_calibrated_field(tmp_path: Path) -> None:
    """The old _draft_root_from_photometry_dir needed calibrated/ (or sat_diag.json on the path): a
    pre-calibrated field without sat_diag.json gave None, so cal_stage.json was never read."""
    import json

    from invariants_runtime import _load_sat_diag_block_from_disk, _sample_cal_stage_from_disk

    root = _precal_field(tmp_path)
    phot = root / "platesolve" / "r" / "photometry"
    assert _load_sat_diag_block_from_disk(phot) == {"sat_adu": 60000.0}
    (root / "sat_diag.json").unlink()
    (root / "cal_stage.json").write_text(json.dumps({"stage": "PRECAL"}), encoding="ascii")
    assert _sample_cal_stage_from_disk(phot) == {"stage": "PRECAL"}


def test_processed_field_root_legacy_draft_folder(tmp_path: Path) -> None:
    from aperture_photometry import _draft_dir_from_aperture_paths
    from processed_field import processed_field_root

    legacy = tmp_path / "Drafts" / "draft_000538"
    phot = legacy / "platesolve" / "V_60_1" / "photometry"
    phot.mkdir(parents=True)
    snap = tmp_path / "Drafts" / "draft_000516_snapshot_era04"
    (snap / "platesolve").mkdir(parents=True)
    assert processed_field_root(phot) == legacy.resolve()
    assert processed_field_root(snap / "platesolve") == snap.resolve()
    assert _draft_dir_from_aperture_paths(phot, phot.parent / "MASTERSTAR.fits") == legacy.resolve()
    assert processed_field_root(tmp_path / "elsewhere") is None


# --------------------------------------------------------------------------------------------------------------
# G13-44: one default-row rule (Milan: an inactive default falls back to the lowest-ID active row + WARNING)
# --------------------------------------------------------------------------------------------------------------


def _loc_db(tmp_path: Path, rows: list[tuple[int, int, object]]) -> Path:
    import sqlite3

    p = tmp_path / "loc.sqlite3"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE LOCATION (ID INTEGER PRIMARY KEY, PLACENAME TEXT, LATITUDE REAL, LONGITUDE REAL, "
                 "ALTITUDE REAL, IS_DEFAULT INTEGER, ACTIVE TEXT)")
    for rid, is_def, act in rows:
        conn.execute("INSERT INTO LOCATION VALUES (?, ?, 50, 14, 300, ?, ?)", (rid, f"L{rid}", is_def, act))
    conn.commit()
    conn.close()
    return p


@pytest.mark.parametrize("inactive", ["NO", "0", "0.0", "false", 0])
def test_default_row_inactive_falls_back_to_lowest_active(tmp_path: Path, inactive, caplog) -> None:
    import logging
    import sqlite3

    from database import default_row_id, get_default_observer_location_id

    p = _loc_db(tmp_path, [(1, 1, inactive), (2, 0, "YES"), (3, 0, "YES")])
    with caplog.at_level(logging.WARNING):
        assert get_default_observer_location_id(p) == 2  # old: 0 (or 1 for '0.0')
    assert any("inactive" in r.message for r in caplog.records)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    try:
        assert default_row_id(conn, "LOCATION") == 2  # old get_default_id: 1 (ignored ACTIVE)
    finally:
        conn.close()


def test_default_row_active_marked_and_unmarked(tmp_path: Path) -> None:
    from database import get_default_observer_location_id

    assert get_default_observer_location_id(_loc_db(tmp_path, [(1, 0, "YES"), (4, 1, None)])) == 4
    (tmp_path / "loc.sqlite3").unlink()
    assert get_default_observer_location_id(_loc_db(tmp_path, [(1, 0, "YES"), (2, 0, "YES")])) == 0


def test_seed_re_marks_an_inactive_default(tmp_path: Path) -> None:
    from database import VyvarDatabase

    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    for t in ("EQUIPMENTS", "TELESCOPE", "LOCATION"):
        db.conn.execute(f"DELETE FROM {t};")
    db.conn.commit()
    a = db.save_reference_object("TELESCOPE", None, {"TELESCOPENAME": "A"})
    b = db.save_reference_object("TELESCOPE", None, {"TELESCOPENAME": "B"})
    db.conn.execute("UPDATE TELESCOPE SET ACTIVE = 'NO' WHERE ID = ?", (a,))
    db.conn.commit()
    assert db.get_default_id("TELESCOPE") == b
    db._seed_table_defaults()
    marked = [r[0] for r in db.conn.execute("SELECT ID FROM TELESCOPE WHERE IS_DEFAULT = 1")]
    assert marked == [b]


def test_normalize_active_uses_one_vocabulary() -> None:
    from database import VyvarDatabase

    f = VyvarDatabase.normalize_active_db_value
    assert [f(v) for v in (None, float("nan"), True, 1, "YES", "y", "", "1.0", "weird")] == [1] * 9
    assert [f(v) for v in (False, 0, 0.0, "NO", "n", "FALSE", "0", "0.0", "F", "off")] == [0] * 10


# --------------------------------------------------------------------------------------------------------------
# G10-37: git hash of the VYVAR source tree, not of the process CWD
# --------------------------------------------------------------------------------------------------------------


def test_hrd_enrich_git_head_ignores_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from photometry_provenance import git_head_hash

    try:
        expected = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_SRC.parent, text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    monkeypatch.chdir(tmp_path)  # not a repository (the old helper returned "unknown" here)
    assert git_head_hash(short=True) == expected
    import hrd_enrich

    assert hrd_enrich._enrich_summary(skip_reason=None, attempts=0)["git_head"] == expected
    assert not hasattr(hrd_enrich, "_git_head_short")


# --------------------------------------------------------------------------------------------------------------
# G05-22: one catalogue keep-filter on catalog_id
# --------------------------------------------------------------------------------------------------------------


def test_catalog_keep_filter_ignores_missing_source_type() -> None:
    """A masterstars CSV without source_type: after the left join the DAO rows carry NaN source_type and
    forced rows GAIA_MATCHED; the old second filter dropped every DAO-measured star."""
    import pandas as pd

    import pipeline_catalog as pc

    df = pd.DataFrame({
        "catalog_id": ["101", "102", "", None, "nan", "103"],
        "source_type": [np.nan, np.nan, np.nan, np.nan, np.nan, "GAIA_MATCHED"],
        "flux": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    out = pc._proc_drop_unmatched_dao_rows(df)
    assert out["catalog_id"].tolist() == ["101", "102", "103"]
    assert not hasattr(pc, "_proc_catalog_keep_matched_rows_only")
    # HEAD 08e985a second filter (reference): catalog_id AND source_type == GAIA_MATCHED
    st_ok = df["source_type"].fillna("").astype(str).str.strip().str.upper().eq("GAIA_MATCHED")
    assert df.loc[st_ok & df["catalog_id"].isin(["101", "102", "103"]), "catalog_id"].tolist() == ["103"]


# --------------------------------------------------------------------------------------------------------------
# G05-49: ra_intervals_deg (Milan: a reversed box raises; the wrap is passed explicitly)
# --------------------------------------------------------------------------------------------------------------


def _ra_covered(intervals, ra: float) -> bool:
    return any(a <= ra <= b for a, b in intervals)


def test_ra_intervals_reversed_box_raises_unless_wrap() -> None:
    from catalog_db import ra_intervals_deg

    with pytest.raises(ValueError):
        ra_intervals_deg(350.0, 10.0)  # old: [] -> silently no rows
    assert sorted(ra_intervals_deg(350.0, 10.0, wrap=True)) == [(0.0, 10.0), (350.0, 360.0)]
    with pytest.raises(ValueError):
        ra_intervals_deg(float("nan"), 10.0)


@pytest.mark.parametrize("box,expected", [
    ((-20.0, -10.0), [(340.0, 350.0)]),          # old: [(340, 360)]
    ((370.0, 380.0), [(10.0, 20.0)]),            # old: [(0, 20)]
    ((-370.0, -355.0), [(0.0, 5.0), (350.0, 360.0)]),
])
def test_ra_intervals_box_outside_0_360(box, expected) -> None:
    from catalog_db import ra_intervals_deg

    assert sorted(ra_intervals_deg(*box)) == expected


def test_ra_intervals_unchanged_on_normal_boxes() -> None:
    """Byte-identity: boxes that touch [0, 360) give the exact HEAD intervals (same values, same order)."""
    from catalog_db import ra_intervals_deg

    def old(rm: float, rM: float):
        if rM - rm >= 360.0:
            return [(0.0, 360.0)]
        if rm >= 0.0 and rM <= 360.0 and rm <= rM:
            return [(rm, rM)]
        out = []
        if rm < 0.0:
            out.append((360.0 + rm, 360.0))
            rm = 0.0
        if rM > 360.0:
            out.append((0.0, rM - 360.0))
            rM = 360.0
        if rm < rM:
            out.append((rm, rM))
        return out

    rng = np.random.default_rng(49)
    for _ in range(5000):
        c = float(rng.uniform(0.0, 360.0))
        h = float(rng.uniform(0.0, 200.0))
        assert ra_intervals_deg(c - h, c + h) == old(c - h, c + h)
    for c in (0.0, 1e-9, 359.999999, 360.0):
        for h in (0.0, 0.3, 5.0):
            assert ra_intervals_deg(c - h, c + h) == old(c - h, c + h)


def test_ra_intervals_property_any_box() -> None:
    from catalog_db import ra_intervals_deg

    rng = np.random.default_rng(7)
    for _ in range(2000):
        lo = float(rng.uniform(-800.0, 800.0))
        w = float(rng.uniform(0.0, 359.0))
        iv = ra_intervals_deg(lo, lo + w)
        assert all(0.0 <= a <= b <= 360.0 for a, b in iv)
        assert abs(sum(b - a for a, b in iv) - w) < 1e-6
        for t in rng.uniform(0.0, w, 5):
            assert _ra_covered(iv, (lo + float(t)) % 360.0)


# --------------------------------------------------------------------------------------------------------------
# G03-21: one WCS key strip list
# --------------------------------------------------------------------------------------------------------------


def _wcs_test_header():
    from astropy.io import fits

    h = fits.Header()
    for k, v in [("OBJECT", "BO CVn"), ("DATE-OBS", "2026-10-01T20:00:00"), ("MJD-OBS", 61314.8),
                 ("MJDREF", 0.0), ("DATEREF", "1858-11-17"), ("B_V", 0.65), ("A_MAG", 12.3), ("BP_RP", 0.8),
                 ("WCSAXES", 2), ("CTYPE1", "RA---TAN-SIP"), ("CTYPE2", "DEC--TAN-SIP"), ("CRVAL1", 10.0),
                 ("CRVAL2", 20.0), ("CRPIX1", 5.0), ("CRPIX2", 5.0), ("CD1_1", 1e-4), ("CD2_2", 1e-4),
                 ("PC3_3", 1.0), ("CROTA2", 12.0), ("WCSNAME", "ASTAP"), ("A_ORDER", 2), ("A_2_0", 1e-6),
                 ("B_ORDER", 2), ("B_0_2", 1e-6), ("AP_ORDER", 2), ("BP_1_1", 1e-7), ("A_DMAX", 0.1),
                 ("PV1_1", 1.0), ("PV2_10", 0.0), ("LONPOLE", 180.0), ("LATPOLE", 20.0), ("RADESYS", "ICRS"),
                 ("EQUINOX", 2000.0), ("PCOUNT", 0), ("CTYPE1A", "RA---TAN"), ("GAIN", 1.0)]:
        h[k] = v
    return h


KEEP = {"OBJECT", "DATE-OBS", "MJD-OBS", "MJDREF", "DATEREF", "B_V", "A_MAG", "BP_RP", "PCOUNT", "GAIN"}


def test_wcs_strip_one_list_utils() -> None:
    from utils import strip_celestial_wcs_keys

    h = _wcs_test_header()
    strip_celestial_wcs_keys(h)
    assert set(h.keys()) == KEEP


def test_wcs_strip_one_list_external_platesolve() -> None:
    from pipeline_astrometry import _strip_external_platesolve_header

    h = _wcs_test_header()
    h["VY_PSOLV"] = 1
    _strip_external_platesolve_header(h)
    assert set(h.keys()) == KEEP


def test_wcs_copy_predicate_excludes_structural_and_user_keys() -> None:
    from utils import header_key_is_celestial_wcs, header_key_is_wcs

    assert header_key_is_celestial_wcs("A_2_0") and header_key_is_celestial_wcs("CD1_2")
    for k in ("B_V", "A_MAG", "BP_RP", "PCOUNT", "MJDREF", "DATEREF", "WCSAXES", "CROTA2"):
        assert not header_key_is_celestial_wcs(k), k
    assert header_key_is_wcs("WCSAXES") and header_key_is_wcs("CROTA2") and not header_key_is_wcs("B_V")


def test_astrosolve_write_solution_keeps_user_and_time_keys(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from astropy.io import fits
    from astropy.wcs import WCS

    import astrosolve

    p = tmp_path / "ms.fits"
    fits.PrimaryHDU(np.zeros((10, 10), np.float32), header=_wcs_test_header()).writeto(p)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [11.0, 21.0]
    w.wcs.crpix = [5.0, 5.0]
    w.wcs.cd = [[2e-4, 0.0], [0.0, 2e-4]]
    res = SimpleNamespace(wcs=w, log_odds=50.0, n_match=40, rms_px=0.2, scale_arcsec=0.72)
    astrosolve._write_solution(p, res, "hinted")
    h = fits.getheader(p)
    assert h["B_V"] == 0.65 and h["A_MAG"] == 12.3  # the old strip_wcs deleted them
    assert h["DATEREF"] == "1858-11-17" and h["MJD-OBS"] == 61314.8
    assert "CROTA2" not in h and "A_ORDER" not in h and "WCSNAME" not in h and "CTYPE1A" not in h
    assert h["CRVAL1"] == 11.0


# --------------------------------------------------------------------------------------------------------------
# G03-17: one frame-bbox box, full RA ring when the pole is inside the frame
# --------------------------------------------------------------------------------------------------------------


def _tan_wcs(ra0: float, dec0: float, scale_arcsec: float, rot_deg: float, nx: int, ny: int):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [nx / 2.0, ny / 2.0]
    s = scale_arcsec / 3600.0
    c, sn = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    w.wcs.cd = [[-s * c, s * sn], [s * sn, s * c]]
    return w


def _old_frame_bbox(wcs, w: float, h: float, m: float):
    xs = np.asarray([-m, w * 0.5, w + m, -m, w + m, -m, w * 0.5, w + m], dtype=np.float64)
    ys = np.asarray([-m, -m, -m, h * 0.5, h * 0.5, h + m, h + m, h + m], dtype=np.float64)
    world = wcs.all_pix2world(xs, ys, 0)
    ras, decs = np.asarray(world[0]), np.asarray(world[1])
    ra0 = float(np.median(ras))
    dra = ((ras - ra0 + 180.0) % 360.0) - 180.0
    pad = 1.0 / 3600.0
    return (ra0 + float(np.min(dra)) - pad, ra0 + float(np.max(dra)) + pad, float(np.min(decs)) - pad,
            float(np.max(decs)) + pad)


@pytest.mark.parametrize("dec0,rot", [(30.0, 0.0), (60.0, 15.0), (82.0, 0.0), (-45.0, 40.0), (0.2, 90.0)])
def test_frame_bbox_unchanged_without_pole(dec0: float, rot: float) -> None:
    from vsx_gaia_crossmatch import frame_bbox_radec_limits

    w = _tan_wcs(355.0, dec0, 3.0, rot, 4144, 2822)
    assert frame_bbox_radec_limits(w, 4144, 2822, margin_px=50.0) == _old_frame_bbox(w, 4144, 2822, 50.0)



@pytest.mark.parametrize("dec0", [89.5, -89.5, 89.9])
def test_frame_bbox_pole_inside_covers_every_pixel(dec0: float) -> None:
    from catalog_db import ra_intervals_deg
    from vsx_gaia_crossmatch import frame_bbox_radec_limits

    nx, ny = 4144, 2822
    w = _tan_wcs(10.0, dec0, 3.0, 20.0, nx, ny)
    ra_min, ra_max, de_min, de_max = frame_bbox_radec_limits(w, nx, ny, margin_px=0.0)
    assert (ra_min, ra_max) == (0.0, 360.0)
    yy, xx = np.mgrid[0:ny:16, 0:nx:16]
    ra, de = w.all_pix2world(xx.ravel(), yy.ravel(), 0)
    iv = ra_intervals_deg(ra_min, ra_max)
    inside = (de >= de_min) & (de <= de_max) & np.array([any(a <= r <= b for a, b in iv) for r in ra % 360.0])
    assert inside.all()
    # the old 8-sample box lost a large part of the chip here
    o = _old_frame_bbox(w, nx, ny, 0.0)
    old_iv = ra_intervals_deg(o[0], o[1])
    old_in = (de >= o[2]) & (de <= o[3]) & np.array([any(a <= r <= b for a, b in old_iv) for r in ra % 360.0])
    assert old_in.mean() < 0.95


# --------------------------------------------------------------------------------------------------------------
# PSF-LOCAL: the hard-coded 50000 ADU saturation level (triage T16b) -> measured level (decision B5)
# --------------------------------------------------------------------------------------------------------------


def _psf_local_run(monkeypatch, tmp_path: Path, *, sat85: float | None):
    from types import SimpleNamespace

    import pandas as pd

    import aperture_pertarget
    from aperture_photometry import _rap_frame_qc_and_psf_scales

    got: dict = {}

    def _fake(ms_path, txy, refs, fwhm, *, target_mag, exclude):
        got.update(refs=refs, exclude=set(exclude))
        return {}, {}

    monkeypatch.setattr(aperture_pertarget, "psf_scale_for_targets", _fake)
    xs = np.arange(200.0, 1900.0, 200.0)
    rows = []
    for i, x in enumerate(xs):
        rows.append({"catalog_id": f"R{i}", "x": x, "y": 500.0, "mag": 12.0, "peak_max_adu": 51000.0 if i == 0 else 20000.0})
    rows.append({"catalog_id": "T", "x": 1000.0, "y": 1500.0, "mag": 11.0, "peak_max_adu": 55000.0})
    rows.append({"catalog_id": "EDGE", "x": 1950.0, "y": 1950.0, "mag": 13.0, "peak_max_adu": 1000.0})
    ms = pd.DataFrame(rows)
    if sat85 is not None:
        ms["saturate_limit_adu_85pct"] = sat85
    state = SimpleNamespace(per_frame_sat_meta=None, _comp_index={}, comp_df=None, masterstars_df=ms,
                            aperture_policy={"fwhm_night_median_px": 3.0}, fwhm_px=3.0,
                            _cfg=SimpleNamespace(comp_peak_sat_frac=0.80), _ms_first_pos=None)
    # s5_perf2 G06-40: the target rows come from the state's masterstars row index
    from types import MethodType

    from aperture_photometry import _AperturePhotometryState

    state.masterstar_row = MethodType(_AperturePhotometryState.masterstar_row, state)
    state.masterstar_pos = MethodType(_AperturePhotometryState.masterstar_pos, state)
    at_df = pd.DataFrame({"catalog_id": ["T"]})
    _rap_frame_qc_and_psf_scales(_p2=lambda *_a: None, at_df=at_df, masterstar_fits_path=tmp_path / "MS.fits",
                                 output_dir=tmp_path, state=state)
    return got


def test_psf_local_uses_measured_saturation(monkeypatch, tmp_path: Path) -> None:
    # SAT 65000: references < 0.80 x SAT = 52000 ADU, target < 0.90 x SAT = 58500 ADU
    got = _psf_local_run(monkeypatch, tmp_path, sat85=0.90 * 65000.0)
    assert (200.0, 500.0, 12.0) in got["refs"]  # 51000 ADU: was dropped by the fixed 50000
    assert "T" not in got["exclude"]  # 55000 ADU target: was excluded by the fixed 50000
    got = _psf_local_run(monkeypatch, tmp_path, sat85=0.90 * 60000.0)  # SAT 60000: 48000 / 54000
    assert (200.0, 500.0, 12.0) not in got["refs"] and "T" in got["exclude"]


def test_psf_local_without_saturation_level_logs_and_skips(monkeypatch, tmp_path: Path, caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        got = _psf_local_run(monkeypatch, tmp_path, sat85=None)
    assert any("no measured saturation level" in r.message for r in caplog.records)
    assert (200.0, 500.0, 12.0) in got["refs"] and "T" not in got["exclude"]


# --------------------------------------------------------------------------------------------------------------
# G05-28: no debug print() in production code (the night-run CLI messages to stderr stay)
# --------------------------------------------------------------------------------------------------------------


def test_no_print_in_production_code() -> None:
    import ast

    bad: list[str] = []
    for p in sorted(_SRC.glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                to_stderr = any(k.arg == "file" for k in node.keywords)
                if not (p.name == "night_run.py" and to_stderr):
                    bad.append(f"{p.name}:{node.lineno}")
    assert not bad, bad
