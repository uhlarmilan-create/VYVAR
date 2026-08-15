"""Unit test: rewrite_comparison_stars_weights_csv makes weights target-dependent."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from comp_weights import rewrite_comparison_stars_weights_csv


def test_rewrite_comp_weights_differ_by_target(tmp_path: Path) -> None:
    rows = []
    for tid, tb in [("T1", 0.5), ("T2", 2.0)]:
        for cid, bpr, rms in [("C1", 0.5, 0.01), ("C2", 1.5, 0.01), ("C3", 2.0, 0.02)]:
            rows.append(
                {
                    "target_catalog_id": tid,
                    "target_bp_rp": tb,
                    "catalog_id": cid,
                    "bp_rp": bpr,
                    "comp_rms": rms,
                    "comp_weight": 1.0 / (rms**2),  # identical formula across targets (old bug)
                }
            )
    p = tmp_path / "comps.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    stats = rewrite_comparison_stars_weights_csv(p)
    assert stats["ok"] == 1
    df = pd.read_csv(p)
    w1 = df[df["target_catalog_id"] == "T1"].set_index("catalog_id")["comp_weight"]
    w2 = df[df["target_catalog_id"] == "T2"].set_index("catalog_id")["comp_weight"]
    # Same star, different target colour -> different weight when colour term active
    assert abs(float(w1["C2"]) - float(w2["C2"])) > 1e-6
    assert "sigma_eff_mag" in df.columns
    assert int(stats["N_eff_unique_rounded"]) >= 2
