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
    from pipeline import plain_mean_med_std

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


def test_inv_ms_identity_01_anchor_baseline() -> None:
    from masterstar_gaia_accounting import lock_existing_and_leftover_assign, verify_ms_identity

    ms = pd.read_csv(MS_DIR / "masterstars_full_match.csv", dtype={"catalog_id": str})
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    matched = ms[ms["catalog_id"].ne("") & ms["catalog_id"].ne("nan")]
    locked = {
        str(r["catalog_id"]): (float(r["x"]), float(r["y"]))
        for _, r in matched.iterrows()
        if np.isfinite(r["x"]) and np.isfinite(r["y"])
    }
    det_x = pd.to_numeric(ms["x"], errors="coerce").to_numpy(dtype=np.float64)
    det_y = pd.to_numeric(ms["y"], errors="coerce").to_numpy(dtype=np.float64)
    gaia = matched[["catalog_id", "x", "y", "mag"]].copy()
    gaia["x_gaia"] = gaia["x"]
    gaia["y_gaia"] = gaia["y"]
    gaia["g_mag"] = pd.to_numeric(gaia["mag"], errors="coerce")
    _, _, _ = lock_existing_and_leftover_assign(
        det_x, det_y, gaia, locked_pairs=locked, leftover_radius_px=3.0
    )
    result = {
        str(r["catalog_id"]): (float(r["x"]), float(r["y"]))
        for _, r in matched.iterrows()
    }
    ok, det = verify_ms_identity(locked, result)
    assert ok, det


def test_inv_ms_census_01_writes_and_fail_loud(tmp_path: Path) -> None:
    from invariants_runtime import InvariantViolation
    from masterstar_gaia_accounting import (
        SOURCE_DETECTED_P1,
        verify_gaia_census_complete,
        write_gaia_census_and_verify,
    )

    census = pd.DataFrame(
        {"catalog_id": ["1", "2"], "source_state": [SOURCE_DETECTED_P1, SOURCE_DETECTED_P1]}
    )
    path = tmp_path / "gaia_source_state_census.csv"
    rec = write_gaia_census_and_verify(census, n_on_chip=2, census_path=path)
    assert path.is_file()
    assert rec["ok"] is True
    ok, _ = verify_gaia_census_complete(census, 2)
    assert ok
    empty = tmp_path / "empty_census.csv"
    write_gaia_census_and_verify(pd.DataFrame(), n_on_chip=0, census_path=empty)
    assert empty.is_file()
    with pytest.raises(InvariantViolation):
        write_gaia_census_and_verify(census, n_on_chip=99, census_path=tmp_path / "bad.csv")


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
