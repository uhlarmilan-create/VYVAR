# -*- coding: ascii -*-
import sqlite3
from pathlib import Path

import pandas as pd

IDS = [
    "1498842882207281152",
    "1499842372636900992",
    "1500410236033012352",
    "1496795041799526400",
    "1497245497969274240",
    "1498425548825498112",
    "1497227287309482624",
]
ms = pd.read_csv(
    Path("Archive/Drafts/draft_000516/platesolve/NoFilter_60_2/masterstars_full_match.csv"),
    dtype={"catalog_id": str},
)
print("vsx_name" in ms.columns, [c for c in ms.columns if "vsx" in c.lower() or c == "name"])
sub = ms[ms["catalog_id"].isin(IDS)]
print(sub[["catalog_id", "name", "phot_g_mean_mag", "bp_rp", "ra_deg", "dec_deg"]].to_string(index=False))
db = Path(r"C:\ASTRO\python\VYVAR\VSX\vyvar_vsx_local_v2.db")
print("vsx_db", db.is_file())
con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
print(con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
print(con.execute("PRAGMA table_info(vsx_data)").fetchall())
