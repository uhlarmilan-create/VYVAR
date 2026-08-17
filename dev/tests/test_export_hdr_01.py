from __future__ import annotations

from pathlib import Path

import pandas as pd

from check_star_kmag import check_catalog_id_from_sidecar, check_kmag_sidecar_path
from export_reports import (
    _format_varastro_comp_table,
    find_truncated_gaia_ids,
    format_aavso_notes_ensemble,
)


def test_find_truncated_gaia_ids_flags_prefix_only() -> None:
    full = "1500748301498613248"
    text = "meth=aperture|GaiaDR3:150074830149861324"
    assert find_truncated_gaia_ids(text, [full]) == [full]


def test_find_truncated_gaia_ids_allows_no_id_and_full_id() -> None:
    full = "1500748301498613248"
    assert find_truncated_gaia_ids("meth=aperture|n_comp=5 GaiaDR3 ensemble", [full]) == []
    assert find_truncated_gaia_ids(f"full:{full}", [full]) == []


def test_format_aavso_notes_ensemble_uses_count_not_ids() -> None:
    notes = format_aavso_notes_ensemble(n_comp=4, lc_method="adaptive")
    assert notes == "meth=adaptive|n_comp=4 GaiaDR3 ensemble"
    assert "1500" not in notes


def test_check_catalog_id_from_sidecar_reads_sidecar_first(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir()
    side = check_kmag_sidecar_path(lc_dir, "1498613634033133184")
    pd.DataFrame(
        {
            "check_catalog_id": ["1497613731286514432"],
            "source_file": ["f1.fits"],
            "kmag": [12.345678],
        }
    ).to_csv(side, index=False)
    assert check_catalog_id_from_sidecar(lc_dir, "1498613634033133184") == "1497613731286514432"


def test_varastro_comp_table_prints_pre_and_post_weight_columns() -> None:
    comp = pd.DataFrame(
        {
            "catalog_id": ["1500748301498613248"],
            "mag": [8.0],
            "bp_rp": [0.4],
            "delta_bprp_abs": [0.1],
            "p2p_rms": [0.008],
            "w_rel": [0.321],
            "comp_tier": [1],
        }
    )
    text = _format_varastro_comp_table(comp, post_weight_rel_map={"1500748301498613248": 1.0})
    assert "w_pre  w_post" in text
    assert "0.321" in text
    assert "1.000" in text
