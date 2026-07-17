from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Ensure repo root is importable when running from scripts/
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from crossmatch_runner import auto_crossmatch_candidates  # noqa: E402
from tess_runner import auto_tess_verify_candidates  # noqa: E402


def main() -> None:
    cfg = AppConfig()
    DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000279")

    phot_dirs = list((DRAFT / "platesolve").glob("*/photometry"))
    assert phot_dirs, "photometry dir nenajdeny"
    PHOT = phot_dirs[0]
    print(f"Photometry: {PHOT}")

    candidates_csv = None
    for name in ["variability_candidates.csv", "candidates.csv", "rms_candidates.csv", "suspected_variables.csv"]:
        p = PHOT / name
        if p.exists():
            candidates_csv = p
            break

    assert candidates_csv, "candidates CSV nenajdeny"
    df = pd.read_csv(candidates_csv, dtype={"catalog_id": str, "name": str}, low_memory=False)
    print(f"Kandidati: {len(df)} riadkov, CSV: {candidates_csv.name}")
    print(f"Stlpce: {list(df.columns)}")

    # Crossmatch
    print("\n--- CROSSMATCH ---")
    # Keep this test fast: crossmatch only the first few rows.
    df_head = df.head(3).copy()
    test_cross_csv = PHOT / "_test_crossmatch_candidates.csv"
    df_head.to_csv(test_cross_csv, index=False)
    auto_crossmatch_candidates(candidates_csv=test_cross_csv, output_dir=PHOT, cfg=cfg)

    df2 = pd.read_csv(test_cross_csv, dtype={"catalog_id": str, "name": str}, low_memory=False)
    katalogy_col = [c for c in df2.columns if "katal" in c.lower()]
    if katalogy_col:
        col = katalogy_col[0]
        s = df2[col].astype(str)
        filled = df2[col].notna() & (s.str.strip() != "") & (s != "—") & (s.str.lower() != "nan")
        print(f"Stlpce '{col}': {int(filled.sum())}/{len(df2)} vyplnenych")
        first = df2.loc[filled].iloc[0] if bool(filled.any()) else None
        if first is not None:
            print(f"Priklad ({first['catalog_id']}):")
            print(str(first[col])[:300])
    else:
        print("Stlpec 'katalogy' nebol najdeny (zly nazov?)")

    # TESS (first candidate only)
    print("\n--- TESS (prvy kandidat) ---")
    df_test = df2.head(1).copy()
    test_csv = PHOT / "_test_candidates.csv"
    df_test.to_csv(test_csv, index=False)
    auto_tess_verify_candidates(candidates_csv=test_csv, output_dir=PHOT, cfg=cfg)

    cid = str(df_test.iloc[0]["catalog_id"])
    tess_dir = PHOT / "_tess" / cid
    result_json = tess_dir / "result.json"
    result_txt = tess_dir / "result.txt"

    print(f"_tess/{cid}/result.json: {'OK' if result_json.exists() else 'CHYBA'}")
    print(f"_tess/{cid}/result.txt:  {'OK' if result_txt.exists() else 'CHYBA'}")

    if result_json.exists():
        data = json.loads(result_json.read_text(encoding="utf-8"))
        print(f"Sektory found/ok: {data.get('total_sectors_found')}/{data.get('total_sectors_ok')}")
        print(f"P_consensus: {data.get('period_consensus')}")

    if result_txt.exists():
        print("\nresult.txt (prvych 20 riadkov):")
        lines = result_txt.read_text(encoding="utf-8").splitlines()
        for line in lines[:20]:
            print(f"  {line}")

    # Cleanup
    try:
        test_csv.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass
    try:
        test_cross_csv.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()

