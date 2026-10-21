"""MASTERSTAR-GAIA-01 fire-proof invariants (INV-DET-FALSEFILL-01, INV-SEED-FALSEFILL-01, INV-MS-*)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]
CTX = ROOT / "dev" / "results" / "context" / "session_20260819_msgaia01"
MS_DIR = ROOT / "Archive" / "Drafts" / "draft_000516" / "platesolve" / "NoFilter_60_2"


@pytest.fixture(scope="module")
def empty_main() -> pd.DataFrame:
    p = CTX / "empty_positions_main.csv"
    if not p.is_file():
        pytest.skip("run tmp/masterstar_gaia_01_part_a.py first")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def ms_data0():
    with fits.open(MS_DIR / "MASTERSTAR.fits", memmap=False) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        fwhm = float(hdul[0].header.get("VY_FWHM", 5.3))
    from plain_stats import plain_mean_med_std

    _, med, _ = plain_mean_med_std(data, sigma=3.0, maxiters=3)
    data0 = np.nan_to_num((data - med).astype(np.float32), nan=0.0)
    h, w = data0.shape
    return data0, w, h, fwhm


def test_inv_det_falsefill_01(empty_main, ms_data0) -> None:
    from config import AppConfig
    from masterstar_gaia_accounting import Pass2AcceptParams, dao_pass2_try_at_position

    data0, w, h, fwhm = ms_data0
    cfg = AppConfig()
    params = Pass2AcceptParams(
        sigma=float(cfg.masterstar_dao_pass2_sigma),
        center_tol_px=float(cfg.masterstar_dao_pass2_center_tol_px),
        fwhm_px=fwhm,
    )
    accept = 0
    for _, row in empty_main.iterrows():
        if dao_pass2_try_at_position(data0, float(row["x"]), float(row["y"]), wpx=w, h=h, params=params)[
            "accepted"
        ]:
            accept += 1
    rate = accept / len(empty_main)
    assert rate <= 0.01, f"pass2 empty-sky false-accept {rate:.4f} > 1%"


def test_inv_seed_falsefill_01(empty_main, ms_data0) -> None:
    from config import AppConfig
    from masterstar_gaia_accounting import (
        ForcedSeedAcceptParams,
        forced_seed_accept,
        forced_seed_measure_at_position,
    )

    data0, w, h, fwhm = ms_data0
    cfg = AppConfig()
    params = ForcedSeedAcceptParams(
        centroid_max_px=float(cfg.masterstar_forced_seed_centroid_max_px),
        snr_min=float(cfg.masterstar_forced_seed_snr_min),
    )
    accept = 0
    for _, row in empty_main.iterrows():
        meas = forced_seed_measure_at_position(
            data0, float(row["x"]), float(row["y"]), fwhm_px=fwhm, params=params
        )
        if forced_seed_accept(meas, params=params)[0]:
            accept += 1
    rate = accept / len(empty_main)
    assert rate <= 0.01, f"seed empty-sky false-accept {rate:.4f} > 1%"


def test_inv_ms_census_01_fail_loud(tmp_path: Path) -> None:
    from invariants_runtime import InvariantViolation
    from masterstar_gaia_accounting import (
        SOURCE_DETECTED_P1,
        verify_gaia_census_complete,
        verify_gaia_census,
    )

    census = pd.DataFrame(
        {"catalog_id": ["1", "2"], "source_state": [SOURCE_DETECTED_P1, SOURCE_DETECTED_P1]}
    )
    rec = verify_gaia_census(census, n_on_chip=2)
    assert rec["ok"] is True and rec["n"] == 2
    ok, _ = verify_gaia_census_complete(census, 2)
    assert ok
    assert verify_gaia_census(pd.DataFrame(), n_on_chip=0)["n"] == 0
    with pytest.raises(InvariantViolation):
        verify_gaia_census(census, n_on_chip=99)
    assert not list(tmp_path.iterdir())  # OUTPUT-REVIEW-2026-10: no census file


def test_seed_rows_excluded_from_comp_pool_when_gate_off() -> None:
    from pipeline import select_comparison_stars_spatial_grid

    rows = []
    for i in range(12):
        rows.append(
            {
                "catalog_id": str(1000 + i),
                "catalog": "Gaia",
                "x": 100.0 + 40.0 * (i % 4),
                "y": 100.0 + 40.0 * (i // 4),
                "ra_deg": 200.0,
                "dec_deg": 40.0,
                "mag": 12.0 + 0.1 * i,
                "is_usable": True,
                "photometry_ok": True,
                "source_state": "DETECTED_P1",
                "forced_photometry": False,
            }
        )
    rows.append(
        {
            "catalog_id": "999999",
            "catalog": "Gaia",
            "x": 120.0,
            "y": 120.0,
            "ra_deg": 200.0,
            "dec_deg": 40.0,
            "mag": 11.0,
            "is_usable": True,
            "photometry_ok": True,
            "source_state": "FORCED_SEED",
            "forced_photometry": True,
        }
    )
    df = pd.DataFrame(rows)
    comp, _meta = select_comparison_stars_spatial_grid(
        df, width_px=400.0, height_px=400.0, n_comp=0, require_non_variable=False
    )
    ids = set(comp["catalog_id"].astype(str)) if len(comp) else set()
    assert "999999" not in ids


def test_part_a_report_exists() -> None:
    p = CTX / "part_a_false_accept.json"
    if not p.is_file():
        pytest.skip("Part A audit not run")
    rep = json.loads(p.read_text(encoding="ascii"))
    e1 = rep.get("E1") or rep.get("E1_tightened")
    assert e1 in ("PASS", "DEVIATE")
    assert int(rep.get("n_main", 0)) >= 2000
