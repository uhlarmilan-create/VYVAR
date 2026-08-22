"""ePSF build edge-star guard (EPSF-VALID-02 S6 Addendum 1)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from database import VyvarDatabase
from psf_photometry import (
    _EPSF_BUILD_GUARD_MAX_DROP_FRAC,
    _EPSF_BUILD_GUARD_REASON,
    _epsf_guard_pick_drop,
    build_epsf_model,
)


def _write_min_masterstar(path: Path, *, n: int = 512) -> None:
    data = np.zeros((n, n), dtype=np.float32)
    hdr = fits.Header()
    hdr["NAXIS1"] = n
    hdr["NAXIS2"] = n
    hdr["VY_FWHM"] = 3.5
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def _write_min_csv(path: Path) -> None:
    rows = []
    for i in range(40):
        rows.append(
            {
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
                "x": 50.0 + i * 10.0,
                "y": 50.0 + i * 8.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _fake_built() -> dict:
    arr = np.ones((17, 17), dtype=np.float32)
    arr /= float(arr.sum())
    return {
        "arr": arr,
        "qc": {"epsf_nan_fraction": 0.0},
        "norm_factor": 1.0,
        "smoothing": "quadratic",
        "fit_shape": (9, 9),
        "epsf_sum_native": 1.0,
        "iteration_failure_curve": [],
    }


def test_epsf_guard_pick_drop_deterministic() -> None:
    df = pd.DataFrame(
        [
            {"catalog_id": "b", "x": 100.0, "y": 100.0},
            {"catalog_id": "a", "x": 10.0, "y": 200.0},
        ]
    )
    drop = _epsf_guard_pick_drop(df, image_shape=(512, 512))
    assert drop["catalog_id"] == "a"
    assert drop["reason"] == _EPSF_BUILD_GUARD_REASON
    assert drop["dist_edge_px"] == pytest.approx(10.0)


def test_build_epsf_guard_drops_once_and_records_meta(tmp_path: Path) -> None:
    ms = tmp_path / "MASTERSTAR.fits"
    csv = tmp_path / "masterstars_full_match.csv"
    _write_min_masterstar(ms)
    _write_min_csv(csv)
    phot = tmp_path / "photometry"
    phot.mkdir()
    ids = [f"GaiaDR3_{1000000000000000000 + i}" for i in range(40)]
    pd.DataFrame({"catalog_id": ids[:10], "zone_flag": "linear"}).to_csv(
        phot / "active_targets.csv", index=False
    )
    pd.DataFrame({"catalog_id": ids[10:]}).to_csv(
        phot / "comparison_stars_per_target.csv", index=False
    )
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")
    calls = {"n": 0}

    def _build_side_effect(*args, **kwargs):  # noqa: ANN002, ANN003
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("All elements of input data must be finite")
        return _fake_built()

    try:
        with patch("psf_photometry._epsf_build_imagepsf_from_stars", side_effect=_build_side_effect):
            out = build_epsf_model(ms, csv, db, draft_id=1, min_stars=10, sandbox_output_dir=tmp_path)
        meta = __import__("json").loads((tmp_path / "masterstar_epsf_meta.json").read_text(encoding="utf-8"))
        guard = meta["build_guard"]
        assert guard["n_dropped"] == 1
        assert len(guard["dropped"]) == 1
        assert "catalog_id" in guard["dropped"][0]
        assert guard["dropped"][0]["reason"] == _EPSF_BUILD_GUARD_REASON
        assert Path(out).is_file()
    finally:
        db.close()


def test_build_epsf_guard_fails_loud_over_ten_percent(tmp_path: Path) -> None:
    ms = tmp_path / "MASTERSTAR.fits"
    csv = tmp_path / "masterstars_full_match.csv"
    _write_min_masterstar(ms)
    _write_min_csv(csv)
    phot = tmp_path / "photometry"
    phot.mkdir()
    ids = [f"GaiaDR3_{1000000000000000000 + i}" for i in range(40)]
    pd.DataFrame({"catalog_id": ids[:10], "zone_flag": "linear"}).to_csv(
        phot / "active_targets.csv", index=False
    )
    pd.DataFrame({"catalog_id": ids[10:]}).to_csv(
        phot / "comparison_stars_per_target.csv", index=False
    )
    db = VyvarDatabase(tmp_path / "vyvar.sqlite3")

    def _always_fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise ValueError("All elements of input data must be finite")

    try:
        with patch("psf_photometry._epsf_build_imagepsf_from_stars", side_effect=_always_fail):
            with pytest.raises(RuntimeError, match=f">{int(_EPSF_BUILD_GUARD_MAX_DROP_FRAC * 100)}%"):
                build_epsf_model(ms, csv, db, draft_id=1, min_stars=10, sandbox_output_dir=tmp_path)
    finally:
        db.close()
