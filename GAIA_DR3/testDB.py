import sqlite3

db_path = r"C:\ASTRO\python\VYVAR\GAIA_DR3\vyvar_gaia_dr3v1.db" # UPRAV PODĽA SEBA
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Skúsime nájsť čokoľvek v okolí tvojho stredu
query = "SELECT count(*) FROM gaia_dr3 WHERE ra BETWEEN 50 AND 60 AND dec BETWEEN 60 AND 62"
cursor.execute(query)
count = cursor.fetchone()[0]

print(f"Počet hviezd v testovacom poli: {count}")
conn.close()