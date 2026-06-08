import sqlite3
import pandas as pd
from pathlib import Path

for dbpath in [
    Path(r"C:\ASTRO\python\VYVAR\vyvar.db"),
    Path(r"C:\ASTRO\python\VYVAR\vyvar.sqlite3"),
]:
    if not dbpath.is_file():
        print("missing", dbpath)
        continue
    print("=== DB", dbpath, "===")
    conn = sqlite3.connect(dbpath)
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
    )
    dao_like = tables[
        tables["name"].str.contains("DAO|dao|STAR|star|DETECT", case=False, na=False)
    ]
    print("DAO-related tables:", dao_like["name"].tolist() or "(none)")
    q = """
    SELECT COUNT(*) as dao_count, MIN(mag) as mag_min, MAX(mag) as mag_max
    FROM DAO_STARS WHERE draft_id = 343
    """
    try:
        print(pd.read_sql(q, conn))
    except Exception as e:
        print("Query error:", e)
    conn.close()
