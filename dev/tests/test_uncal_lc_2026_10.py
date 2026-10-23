"""UNCAL-LC-2026-10: a light curve without a band zero point stays a light curve."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

import photometry_core  # noqa: F401  (import order: avoids circular import)
from check_star_kmag import compute_check_ensemble_mag_calib
from config import AppConfig

SRC = Path(__file__).resolve().parents[2] / "src_py"


def _lcs(n: int = 60, seed: int = 1):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.05, n)  # common transparency
    comps = {f"c{i}": 12.0 + i * 0.3 + z + rng.normal(0, 0.005, n) for i in range(4)}
    check = 13.0 + z + rng.normal(0, 0.005, n)
    return comps, check


def _check(cat: dict, allow_relative: bool):
    comps, check = _lcs()
    lc = dict(comps)
    lc["9"] = check
    ids = list(comps)
    return compute_check_ensemble_mag_calib(
        "9", ids, lc, cat, {c: {"quality": "good"} for c in ids},
        cfg=AppConfig(), allow_relative=allow_relative,
    )


def test_check_relative_only_when_allowed() -> None:
    nan_cat = {f"c{i}": float("nan") for i in range(4)}
    assert _check(nan_cat, allow_relative=False) is None  # calibrated target: no level guess
    r = _check(nan_cat, allow_relative=True)
    assert r is not None and np.isfinite(r.kmag).sum() == 60
    # scatter of the relative check equals the calibrated one (zero point is a constant)
    cal = _check({f"c{i}": 12.0 + i * 0.3 for i in range(4)}, allow_relative=False)
    assert abs(np.nanstd(r.kmag) - np.nanstd(cal.kmag)) < 1e-12
    # level = check minus ensemble level (13.0 - ~12.45), not the catalogue level 13.0;
    # it is relative and is never exported (target band_system NONE)
    assert abs(np.nanmedian(r.kmag) - 0.55) < 0.05
    assert abs(np.nanmedian(cal.kmag) - 13.0) < 0.05


def test_lc_uncalibrated_helper(tmp_path) -> None:
    from gaia_bands import lc_uncalibrated

    (tmp_path / "comp_quality_1.json").write_text('{"zero_point": {"band": "NONE"}}')
    (tmp_path / "comp_quality_2.json").write_text('{"zero_point": {"band": "g"}, "band_system": "g"}')
    assert lc_uncalibrated(tmp_path, "1") is True
    assert lc_uncalibrated(tmp_path, "2") is False
    assert lc_uncalibrated(tmp_path, "3") is False  # nothing known -> not relabelled
    assert lc_uncalibrated(tmp_path, "3", "NONE") is True  # summary fallback


def test_flags_from_differential_series() -> None:
    """The real flagging path (apply_reporting_postprocess -> assign_lc_flags): the
    differential series gives measured points, the all-NaN calibrated series gives no_data."""
    from aperture_photometry import apply_reporting_postprocess

    rng = np.random.default_rng(5)
    n = 40
    delta = 0.5 + 0.02 * np.sin(np.arange(n) / 5) + rng.normal(0, 0.003, n)
    nan = np.full(n, np.nan)
    kw = dict(target_name="t", sat_flags=np.zeros(n, bool),
              target_frames=pd.DataFrame({"source_file": [f"f{i}" for i in range(n)]}),
              ct_ok=False, err=np.full(n, 0.003), bjd=np.arange(n) * 0.01,
              source_files=[f"f{i}" for i in range(n)])
    _, _, fl_rel, _ = apply_reporting_postprocess(delta, delta, **kw)
    _, _, fl_nan, _ = apply_reporting_postprocess(nan, nan, **kw)
    assert fl_rel.count("normal") >= n - 2
    assert set(fl_nan) == {"no_data"}


def test_aperture_wiring() -> None:
    s = (SRC / "aperture_photometry_target.py").read_text(encoding="utf-8")
    assert '_calibrated = _band_system != "NONE"' in s
    i_pp = s.index("apply_reporting_postprocess(")
    assert "mag_calib if _calibrated else _meas" in s[i_pp:i_pp + 200]
    i_nan = s.index("mag_calib, mag_calib_ct = _nan_lc.copy()", i_pp)
    assert i_nan > i_pp
    assert "_lc_rms_full = float(np.std(finite_meas))" in s
    assert '"lc_median_mag": float(np.median(finite_calib))' in s  # level only when calibrated
    tree = ast.parse(s)
    kws = [k for n in ast.walk(tree) if isinstance(n, ast.Call)
           and getattr(n.func, "id", "") == "compute_check_ensemble_mag_calib" for k in n.keywords]
    assert any(k.arg == "allow_relative" and ast.unparse(k.value) == "not _calibrated" for k in kws)


def test_export_refuses_band_none() -> None:
    s = (SRC / "export_reports.py").read_text(encoding="utf-8")
    assert "on_band = bool(band_system) and band_system == fb.zp_band" in s
