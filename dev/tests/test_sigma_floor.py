"""Tests for PROD-SIGMA-FLOOR: c4, err combine, floor resolution."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mag_constants import MAG_ERR_SCALE
from sigma_floor_core import combine_production_err_mag, combine_production_err_rel, resolve_sigma_sys_mag


def test_combine_production_err_rel_domain() -> None:
    phot = 0.01
    sem_mag = 0.005
    floor_mag = 0.003
    rel = combine_production_err_rel(phot, sem_mag, sigma_sys_mag=floor_mag)
    sem_rel = sem_mag / MAG_ERR_SCALE
    floor_rel = floor_mag / MAG_ERR_SCALE
    expected = math.sqrt(phot * phot + sem_rel * sem_rel + floor_rel * floor_rel)
    assert abs(rel - expected) < 1e-12


def test_combine_production_err_mag_roundtrip() -> None:
    phot = 0.02
    sem = 0.004
    floor = 0.006
    mag = combine_production_err_mag(phot, sem, sigma_sys_mag=floor)
    rel = combine_production_err_rel(phot, sem, sigma_sys_mag=floor)
    assert abs(mag - rel * MAG_ERR_SCALE) < 1e-12


def test_resolve_sigma_sys_default_zero() -> None:
    class _Cfg:
        database_path = ""

    assert resolve_sigma_sys_mag(99, _Cfg()) == 0.0


def test_resolve_sigma_sys_from_equipments(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "v.sqlite3"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE EQUIPMENTS (ID INTEGER PRIMARY KEY, CAMERANAME TEXT, SIGMA_SYS_MAG REAL)")
    con.execute("INSERT INTO EQUIPMENTS VALUES (4, 'C5A-150M', 0.005)")
    con.commit()
    con.close()

    class _Cfg:
        database_path = str(db)

    assert resolve_sigma_sys_mag(4, _Cfg()) == pytest.approx(0.005)
    assert resolve_sigma_sys_mag(1, _Cfg()) == 0.0


def test_floor_add_after_sem_in_quadrature() -> None:
    from photometry_core import _combine_err_with_ensemble_scatter_keyed

    err = np.array([0.01, 0.012])
    scatter = {"a.csv": 0.004, "b.csv": 0.003}
    out, unmatched = _combine_err_with_ensemble_scatter_keyed(
        err, ["a.csv", "b.csv"], scatter, sigma_sys_mag=0.005,
    )
    assert not unmatched.any()
    for i, sf in enumerate(["a.csv", "b.csv"]):
        expected = combine_production_err_rel(
            float(err[i]), scatter[sf], sigma_sys_mag=0.005,
        )
        assert out[i] == pytest.approx(expected, rel=1e-9)
