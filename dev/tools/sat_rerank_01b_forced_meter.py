"""SAT-RERANK-01B 4.3: forced-check MAD on the Part 3 product. Does not overwrite 01B."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dev" / "tools"))

from d515_accept_01b_same_meter import (  # noqa: E402
    BO,
    CHK_BO,
    CHK_FW,
    DRAFT515,
    FW,
    comps_of,
    run_forced_check,
)
import pandas as pd

OUT = ROOT / "dev" / "results" / "SAT_RERANK_01B_forced_meter.json"
DA9_BO = 7.0498
DA9_FW = 10.6836


def main() -> int:
    phot = DRAFT515 / "platesolve" / "NoFilter_60_2" / "photometry"
    meta = json.loads((phot / "pipeline_meta.json").read_text(encoding="utf-8"))
    sha = str(meta.get("photometry_sha") or meta.get("git_hash") or "part3")
    comp = pd.read_csv(
        phot / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    fw_ens = set(comps_of(comp, FW))
    if CHK_FW in fw_ens:
        print("STOP: FW check sits inside the FW ensemble", flush=True)
        OUT.write_text(
            json.dumps({"stop": True, "reason": "FW check consumed by FW ensemble", "fw_ens": sorted(fw_ens)}, indent=2),
            encoding="ascii",
        )
        return 1

    print("Forced BO check...", flush=True)
    bo = run_forced_check(
        draft=DRAFT515,
        draft_id=515,
        target_id=BO,
        check_id=CHK_BO,
        label="01B_BO",
    )
    print("Forced FW check...", flush=True)
    fw = run_forced_check(
        draft=DRAFT515,
        draft_id=515,
        target_id=FW,
        check_id=CHK_FW,
        label="01B_FW",
    )
    out = {
        "archive_git_or_meta": sha,
        "BO": bo,
        "FW": fw,
        "da9cce4_pointer": {"BO_mmag": DA9_BO, "FW_mmag": DA9_FW, "n_epochs": 134},
        "delta_vs_da9_BO_mmag": None
        if bo.get("check_scatter_mad_mmag") is None
        else float(bo["check_scatter_mad_mmag"]) - DA9_BO,
        "delta_vs_da9_FW_mmag": None
        if fw.get("check_scatter_mad_mmag") is None
        else float(fw["check_scatter_mad_mmag"]) - DA9_FW,
    }
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="ascii")
    print("BO MAD", bo.get("check_scatter_mad_mmag"), "vs", DA9_BO, flush=True)
    print("FW MAD", fw.get("check_scatter_mad_mmag"), "vs", DA9_FW, flush=True)
    print("WROTE", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
