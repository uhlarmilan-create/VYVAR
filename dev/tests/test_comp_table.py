# -*- coding: ascii -*-
"""OUTPUT-REVIEW-2026-10 step 5: one comparison table with comparison-star selection and used roles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from comp_table import (
    COMP_TABLE_COLUMNS,
    admitted,
    load_comp_table,
    comp_selection_table,
    reserve,
    used,
    with_used_sets,
    write_comp_table,
)


def _rows(tid: str, cids: list[str], **extra) -> pd.DataFrame:
    return pd.DataFrame({"target_catalog_id": tid, "catalog_id": cids, "x": 1.0, "y": 2.0,
                         "comp_rms": [0.01 * (i + 1) for i in range(len(cids))], "name": cids,
                         "phot_g_mean_mag": 11.0, "mag": 11.0, **extra})


def test_table_roles_order_and_columns(tmp_path: Path) -> None:
    adm = pd.concat([_rows("200", ["21", "22", "23"]), _rows("100", ["11", "12"])], ignore_index=True)
    res = pd.concat([_rows("200", ["25", "24"], reserve_rank=[1, 0]), _rows("100", ["13"], reserve_rank=[0])],
                    ignore_index=True)
    p = write_comp_table(comp_selection_table(adm, res), tmp_path / "comparison_stars_per_target.csv")
    assert list(pd.read_csv(p, nrows=0).columns) == list(COMP_TABLE_COLUMNS)  # no MASTERSTAR copy, no name
    t = load_comp_table(tmp_path)
    a = admitted(t)
    assert list(zip(a["target_catalog_id"], a["catalog_id"])) == [("200", "21"), ("200", "22"), ("200", "23"),
                                                                    ("100", "11"), ("100", "12")]
    assert list(reserve(t)["catalog_id"]) == ["24", "25", "13"]  # by rank within the target
    assert used(t).empty
    t2 = with_used_sets(t, {"200": [("22", "primary"), ("24", "replacement"), ("21", "kept_suspect")],
                            "100": [("11", "primary"), ("12", "primary")]})
    write_comp_table(t2, p)
    u = used(load_comp_table(p))
    assert list(u["catalog_id"]) == ["22", "24", "21", "11", "12"]
    assert list(u["used_role"]) == ["primary", "replacement", "kept_suspect", "primary", "primary"]
    assert len(admitted(load_comp_table(p))) == 5  # a re-run starts from the comparison-star selection rows


def test_legacy_three_files_give_the_same_view(tmp_path: Path) -> None:
    p1 = _rows("100", ["11", "12", "13"])
    p1.to_csv(tmp_path / "comparison_stars_per_target_phase1.csv", index=False)
    _rows("100", ["14", "15"], reserve_rank=[0, 1]).to_csv(tmp_path / "comparison_reserve_per_target.csv",
                                                            index=False)
    main = pd.concat([_rows("100", ["11", "13"]).assign(comp_role="primary"),
                      _rows("100", ["14"]).assign(comp_role="replacement")], ignore_index=True)
    main.to_csv(tmp_path / "comparison_stars_per_target.csv", index=False)
    t = load_comp_table(tmp_path)
    assert list(admitted(t)["catalog_id"]) == ["11", "12", "13"]
    assert list(reserve(t)["catalog_id"]) == ["14", "15"]
    assert list(used(t)["catalog_id"]) == ["11", "13", "14"]
    assert list(used(t)["used_role"]) == ["primary", "primary", "replacement"]


def test_legacy_single_file_is_used_as_is(tmp_path: Path) -> None:
    _rows("100", ["11", "12"]).to_csv(tmp_path / "comparison_stars_per_target.csv", index=False)
    t = load_comp_table(tmp_path)
    assert list(admitted(t)["catalog_id"]) == ["11", "12"] and list(used(t)["catalog_id"]) == ["11", "12"]


def test_pool_file_is_refused(tmp_path: Path) -> None:
    pd.DataFrame({"comp_id": ["COMP_0001"], "role": ["comparison"], "catalog_id": ["1"]}).to_csv(
        tmp_path / "comparison_stars_per_target.csv", index=False)
    with pytest.raises(ValueError, match="comparison pool file"):
        load_comp_table(tmp_path)
