"""GAIA-ID-STRICT-2026-10 (code review 2, T07): Gaia ids are text, never fabricated, identity = catalog_id.

Pre-fail at 5b06fda: normalize_gaia_source_id(4.611686018427388e+18) -> "4611686018427388000" (a fabricated id);
a proc CSV with one blank id read by the aperture grid worker turned every id into a float; repair_catalog_ids
gave VSX targets without a Gaia match the nearest Gaia star; the proc row index keyed on ``name``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src_py") not in sys.path:
    sys.path.insert(0, str(ROOT / "src_py"))

import pipeline  # noqa: E402,F401

from gaia_catalog_id import VYVAR_CSV_DTYPE, masterstar_row_gaia_key, normalize_gaia_source_id, read_vyvar_csv  # noqa: E402

GID = "4611686018427387905"  # > 2**53: a float cannot hold it


def test_normaliser_never_fabricates_an_id():
    assert normalize_gaia_source_id(GID) == GID
    assert normalize_gaia_source_id(int(GID)) == GID
    assert normalize_gaia_source_id(np.int64(GID)) == GID
    assert normalize_gaia_source_id(f' "{GID}" ') == GID
    for bad in (float(GID), "4.611686018427388e+18", f"{float(GID)!r}", True, b"123", "\uff11\uff12", -5, "-5",
                {"source_id": int(GID)}, [GID], "DET_0001", "Gaia DR3 123", np.float32(1.5)):
        assert normalize_gaia_source_id(bad) == "", bad
    assert normalize_gaia_source_id(123.0) == "123" and normalize_gaia_source_id("123.0") == "123"  # exact, < 2**53


def test_one_blank_id_keeps_19_digit_ids_as_text(tmp_path):
    from proc_frame_store import read_proc_csv

    p = tmp_path / "proc_f001.csv"
    pd.DataFrame({"catalog_id": [GID, ""], "x": [1.0, 2.0], "y": [1.0, 2.0]}).to_csv(p, index=False)
    for df in (read_proc_csv(p), read_vyvar_csv(p), read_proc_csv(p, usecols=lambda c: c in {"catalog_id", "x"})):
        assert df["catalog_id"].iloc[0] == GID


def test_identity_is_catalog_id_only():
    assert masterstar_row_gaia_key(pd.Series({"catalog_id": "", "name": GID})) == ""
    assert masterstar_row_gaia_key(pd.Series({"catalog_id": GID, "name": "DET_0001"})) == GID
    assert {"catalog_id", "target_catalog_id", "comp_catalog_id", "check_catalog_id", "source_id"} <= set(VYVAR_CSV_DTYPE)


def test_repair_by_nearest_gaia_star_is_gone():
    assert not (ROOT / "src_py" / "repair_catalog_ids.py").exists()
    for p in (ROOT / "src_py").glob("*.py"):
        assert "repair_catalog_ids" not in p.read_text(encoding="ascii"), p.name


def test_grid_worker_reads_proc_with_the_one_reader():
    src = (ROOT / "src_py" / "aperture_pertarget.py").read_text(encoding="ascii")
    assert "pd.read_csv(\n            proc" not in src and "read_proc_csv(proc" in src


# --- name is a label, never run through the id normaliser (regression found in the T07 A/B on BO CVn:
# active_targets.csv lost every VSX name, "FY CVn" -> "") ---------------------------------------------------


def test_proc_store_keeps_name_label(tmp_path):
    from proc_frame_store import ProcFrameStore

    (tmp_path / "proc_001.csv").write_text(
        "catalog_id,name,x,y\n1496278752372040832,FY CVn,10.0,20.0\n1496293286541396480,,11.0,21.0\n",
        encoding="ascii",
    )
    store = ProcFrameStore.build(tmp_path, extra_cols=["catalog_id", "name", "x", "y"])
    df = next(iter(store.values()))
    assert list(df["catalog_id"]) == ["1496278752372040832", "1496293286541396480"]
    assert df["name"].iloc[0] == "FY CVn"


def test_no_id_normaliser_on_name_column():
    """Source scan: no loop normalises ("catalog_id", "name") together and no normaliser is applied to name."""
    import re

    pat_loop = re.compile(r'for \w+ in \("catalog_id", "name"\):\n(?:.*\n){0,3}.*(normaliz|_norm)', re.MULTILINE)
    pat_call = re.compile(r'(normaliz\w*|_norm\w*)\([^)\n]*(\[|get\()"name"')
    bad = []
    for p in sorted((ROOT / "src_py").glob("*.py")):
        s = p.read_text(encoding="utf-8")
        bad += [f"{p.name}: loop" for _ in pat_loop.finditer(s)]
        bad += [f"{p.name}: {m.group(0)}" for m in pat_call.finditer(s)]
    assert bad == []
