from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tests.photometry_sha import compare_photometry_science_meaningful


def _write_lc(path: Path, *, errs: list[float], mags: list[float]) -> None:
    df = pd.DataFrame(
        {
            "source_file": [f"f{i:03d}.csv" for i in range(len(errs))],
            "err": errs,
            "mag_calib": mags,
            "bjd": [2450000.0 + i for i in range(len(errs))],
        }
    )
    path.write_text(df.to_csv(index=False), encoding="utf-8")


def _mk_tree(root: Path, setup: str, tid: str, *, errs: list[float], mags: list[float]) -> None:
    lc_dir = root / "platesolve" / setup / "photometry" / "lightcurves"
    lc_dir.mkdir(parents=True, exist_ok=True)
    _write_lc(lc_dir / f"lightcurve_{tid}.csv", errs=errs, mags=mags)
    # comparator expects comparison CSV to exist, but comp diffs are not the target here
    comp = root / "platesolve" / setup / "photometry" / "comparison_stars_per_target.csv"
    comp.parent.mkdir(parents=True, exist_ok=True)
    comp.write_text("target_catalog_id,catalog_id,_dist_deg\n", encoding="utf-8")


def test_err_designed_envelope_fails_on_1p6x(tmp_path: Path) -> None:
    setup = "NoFilter_60_2"
    tid = "T1"
    mags = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    err_b = [0.01] * len(mags)
    err_a = [0.016] * len(mags)
    ra = tmp_path / "a"
    rb = tmp_path / "b"
    _mk_tree(ra, setup, tid, errs=err_a, mags=mags)
    _mk_tree(rb, setup, tid, errs=err_b, mags=mags)

    rep = compare_photometry_science_meaningful(
        ra,
        rb,
        setups=(setup,),
        err_designed=True,
        err_accept={"mode": "envelope", "min_ratio": 0.96, "max_ratio": 1.05},
    )
    assert rep["summary"]["err_check"]["enabled"] is True
    assert rep["summary"]["err_check"]["ok"] is False
    assert rep["summary"]["benign"] is False


def test_err_designed_exact_pred_passes(tmp_path: Path) -> None:
    setup = "NoFilter_60_2"
    tid = "T1"
    mags = [10.0, 11.0, 12.0]
    err_b = [0.01, 0.02, 0.03]
    err_a = [0.011, 0.019, 0.031]
    ra = tmp_path / "a"
    rb = tmp_path / "b"
    _mk_tree(ra, setup, tid, errs=err_a, mags=mags)
    _mk_tree(rb, setup, tid, errs=err_b, mags=mags)

    # predicted errs for A keyed by source_file
    pred = {
        "setups": {
            setup: {
                tid: {
                    "f000.csv": err_a[0],
                    "f001.csv": err_a[1],
                    "f002.csv": err_a[2],
                }
            }
        }
    }
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps(pred), encoding="utf-8")

    rep = compare_photometry_science_meaningful(
        ra,
        rb,
        setups=(setup,),
        err_designed=True,
        err_accept={"mode": "exact_pred", "predicted_err_json": str(pred_path), "abs_tol": 2e-6},
    )
    assert rep["summary"]["err_check"]["enabled"] is True
    assert rep["summary"]["err_check"]["ok"] is True
    assert rep["summary"]["benign"] is True

