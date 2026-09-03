# -*- coding: ascii -*-
"""Identity CSV round-trip vs INV-EXPORT-READ-ONLY-01 hash. Read-only on freeze."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src_py"))

from gaia_catalog_id import read_vyvar_csv  # noqa: E402
from epsf_psf_merge import hash_non_psf_columns, non_psf_columns  # noqa: E402
from pipeline import _vyvar_df_to_csv  # noqa: E402

SNAP = REPO / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
SETUP = "NoFilter_60_2"
OUT = Path(__file__).resolve().parent
TMP = REPO / "tmp" / "epsf_chain_roundtrip"


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    lights = SNAP / "detrended_aligned" / "lights" / SETUP
    fails = []
    n = 0
    for src in sorted(lights.glob("proc_*.csv")):
        n += 1
        before = read_vyvar_csv(src, low_memory=False)
        pre = hash_non_psf_columns(before)
        dst = TMP / src.name
        _vyvar_df_to_csv(before, dst)
        after = read_vyvar_csv(dst, low_memory=False)
        post = hash_non_psf_columns(after)
        if pre != post:
            bcols = set(non_psf_columns(before.columns))
            acols = set(non_psf_columns(after.columns))
            fails.append(
                {
                    "file": src.name,
                    "col_set_equal": bcols == acols,
                    "only_before": sorted(bcols - acols)[:10],
                    "only_after": sorted(acols - bcols)[:10],
                }
            )
            print("FAIL", src.name)
        if n % 20 == 0:
            print("checked", n, "fails", len(fails), flush=True)
    rec = {"n": n, "n_fail": len(fails), "fails": fails}
    (OUT / "roundtrip.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="ascii")
    print("done", rec["n"], "fails", rec["n_fail"])


if __name__ == "__main__":
    main()
