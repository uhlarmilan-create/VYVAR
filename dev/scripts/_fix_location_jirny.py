import sqlite3
from pathlib import Path

db = Path(__file__).resolve().parents[1] / "vyvar.sqlite3"
conn = sqlite3.connect(db)
conn.execute("UPDATE LOCATION SET PLACENAME = ? WHERE ID = ?", ("Jirny", 2))
conn.commit()
for row in conn.execute("SELECT * FROM LOCATION"):
    print(row)
conn.close()
