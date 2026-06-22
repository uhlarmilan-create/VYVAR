"""Phase 2A proc-gap: variable targets = direct catalog_id hits only."""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from photometry_core import read_flux_from_csv, _build_csv_lookup


def _proc_csv(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "proc_frame.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _base_row(
    catalog_id: str,
    x: float,
    y: float,
    dao_flux: float,
    *,
    bjd: float = 2460898.41,
) -> dict:
    return {
        "catalog_id": catalog_id,
        "name": catalog_id,
        "x": x,
        "y": y,
        "dao_flux": dao_flux,
        "bjd_tdb_mid": bjd,
        "hjd_mid": bjd - 0.001,
        "jd_mid": bjd,
        "aperture_r_px": 3.0,
        "noise_floor_adu": 1.0,
        "peak_max_adu": 1000.0,
    }


@pytest.fixture
def bright_neighbor_flux() -> float:
    return 63096.0


@pytest.fixture
def faint_neighbor_flux() -> float:
    return 1932.94


def test_bright_neighbor_rejected(tmp_path, bright_neighbor_flux):
    target_cid = "458459863048332032"
    neighbor_cid = "458459863049757568"
    tx, ty = 1109.51, 1513.75
    proc = _proc_csv(
        tmp_path,
        [_base_row(neighbor_cid, tx + 4.0, ty + 5.0, bright_neighbor_flux)],
    )
    out = read_flux_from_csv(
        proc,
        [target_cid],
        {target_cid: 3.0},
        star_xy={target_cid: (tx, ty)},
        xy_tol_px=18.0,
        variable_target_catalog_ids=frozenset({target_cid}),
    )
    row = out.iloc[0]
    assert row["flag"] == "no_data"
    assert not math.isfinite(float(row["mag_inst"]))


def test_mag_similar_different_cid_neighbor_rejected(tmp_path, faint_neighbor_flux):
    """R-outlier geometry: 2.47 px, mag-similar wrong star -> NaN."""
    target_cid = "458470858164631936"
    neighbor_cid = "458470858164631040"
    tx, ty = 1463.82379487, 1507.36407859
    proc = _proc_csv(
        tmp_path,
        [_base_row(neighbor_cid, tx + 0.12, ty + 2.47, faint_neighbor_flux)],
    )
    out = read_flux_from_csv(
        proc,
        [target_cid],
        {target_cid: 2.768},
        star_xy={target_cid: (tx, ty)},
        xy_tol_px=18.0,
        variable_target_catalog_ids=frozenset({target_cid}),
    )
    assert not math.isfinite(float(out.iloc[0]["mag_inst"]))
    assert out.iloc[0]["flag"] == "no_data"


def test_direct_catalog_id_hit_unchanged(tmp_path):
    target_cid = "458308955066917120"
    flux = 2148.69
    proc = _proc_csv(tmp_path, [_base_row(target_cid, 100.0, 200.0, flux)])
    kwargs = {
        "star_xy": {target_cid: (100.0, 200.0)},
        "xy_tol_px": 18.0,
        "variable_target_catalog_ids": frozenset({target_cid}),
    }
    a = read_flux_from_csv(proc, [target_cid], {target_cid: 3.0}, **kwargs)
    b = read_flux_from_csv(proc, [target_cid], {target_cid: 3.0})
    assert a.iloc[0]["mag_inst"] == b.iloc[0]["mag_inst"]
    assert a.iloc[0]["err"] == b.iloc[0]["err"]


def test_comp_star_xy_fallback_legacy_guard_preserved(tmp_path, bright_neighbor_flux):
    comp_cid = "458377674554774016"
    tx, ty = 200.0, 300.0
    proc = _proc_csv(
        tmp_path,
        [_base_row("bright_neighbor", tx + 1.0, ty + 1.0, bright_neighbor_flux)],
    )
    out_legacy = read_flux_from_csv(
        proc,
        [comp_cid],
        {comp_cid: 3.0},
        star_xy={comp_cid: (tx, ty)},
        xy_tol_px=18.0,
    )
    out_new = read_flux_from_csv(
        proc,
        [comp_cid],
        {comp_cid: 3.0},
        star_xy={comp_cid: (tx, ty)},
        xy_tol_px=18.0,
        variable_target_catalog_ids=frozenset({"other_target"}),
    )
    assert float(out_legacy.iloc[0]["mag_inst"]) == float(out_new.iloc[0]["mag_inst"])
    assert out_legacy.iloc[0]["flag"] == out_new.iloc[0]["flag"]


def test_blowup_geometry_draft_419(tmp_path, bright_neighbor_flux):
    target_cid = "458459863048332032"
    tx, ty = 1109.5116432526306, 1513.7486912117267
    proc = _proc_csv(
        tmp_path,
        [
            _base_row(
                "458459863049757568",
                1105.775718,
                1519.051219,
                bright_neighbor_flux,
            ),
        ],
    )
    out = read_flux_from_csv(
        proc,
        [target_cid],
        {target_cid: 2.768},
        star_xy={target_cid: (tx, ty)},
        xy_tol_px=18.0,
        variable_target_catalog_ids=frozenset({target_cid}),
    )
    assert not math.isfinite(float(out.iloc[0]["mag_inst"]))
