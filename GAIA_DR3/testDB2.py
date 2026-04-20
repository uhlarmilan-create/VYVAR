import sqlite3
conn = sqlite3.connect(r"C:\ASTRO\python\VYVAR\GAIA_DR3\vyvar_gaia_dr3.db")
import pandas as pd
print(pd.read_sql("SELECT COUNT(*) as celkom, COUNT(g_mag) as ma_g_mag, MIN(g_mag), MAX(g_mag), MIN(dec), MAX(dec) FROM gaia_dr3", conn))
conn.close()