# -*- coding: ascii -*-
"""OUTPUT-REVIEW-2026-10: the candidates are rows of variability_field.csv (no variability_candidates.csv)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from variability_field import load_variability_candidates, write_candidate_catalogs


def _field(phot: Path, with_catalogs: bool) -> None:
    f = pd.DataFrame({"catalog_id": ["11", "12", "13", "14"], "z_s": [3.0, 9.0, 1.0, 5.0],
                      "candidate": [True, True, False, True], "label": ["V3", "V1", "", "V2"]})
    if with_catalogs:
        f["catalogs"] = ["c11", "", "", "c14"]
    f.to_csv(phot / "variability_field.csv", index=False)


def test_candidates_from_field_in_label_order(tmp_path: Path) -> None:
    _field(tmp_path, True)
    c = load_variability_candidates(tmp_path)
    assert list(c["label"]) == ["V1", "V2", "V3"] and list(c["catalog_id"]) == ["12", "14", "11"]
    assert list(c.columns)[0] == "label" and list(c["catalogs"]) == ["", "c14", "c11"]
    write_candidate_catalogs(tmp_path, {"12": "VSX: XX Cnv"})
    c2 = load_variability_candidates(tmp_path)
    assert list(c2["catalogs"]) == ["VSX: XX Cnv", "c14", "c11"]
    assert not (tmp_path / "variability_candidates.csv").exists()


def test_legacy_candidates_file_is_read(tmp_path: Path) -> None:
    _field(tmp_path, False)
    pd.DataFrame({"label": ["V1"], "catalog_id": ["12"], "catalogs": ["old"]}).to_csv(
        tmp_path / "variability_candidates.csv", index=False)
    c = load_variability_candidates(tmp_path)
    assert list(c["catalog_id"]) == ["12"] and list(c["catalogs"]) == ["old"]


def test_no_files_gives_empty_frame(tmp_path: Path) -> None:
    c = load_variability_candidates(tmp_path)
    assert c.empty and {"label", "catalog_id", "catalogs"} <= set(c.columns)
