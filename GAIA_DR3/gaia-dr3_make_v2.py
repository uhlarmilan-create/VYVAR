import pandas as pd
from astroquery.gaia import Gaia
import sqlite3
import os
import time

# --- KONFIGURÁCIA ---
DB_NAME = "vyvar_gaia_dr3_turbo.db"
MAG_LIMIT = 15.5      # Kompromis (16 je 3x viac dát, 15 je bleskové)
DEC_MIN = -20.0
DEC_MAX = 90.0
STEP_DEC = 0.5        # Sťahujeme po 0.5° pásoch (veľmi efektívne pre server)

def build_turbo_db():
    conn = sqlite3.connect(DB_NAME)
    
    # Vytvorenie tabuľky
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gaia_dr3 (
            source_id INTEGER PRIMARY KEY,
            ra REAL, dec REAL,
            g_mag REAL, bp_mag REAL, rp_mag REAL, bp_rp REAL,
            parallax REAL, parallax_error REAL,
            pmra REAL, pmdec REAL
        )
    """)

    # Zistíme, kde sme skončili (pre prípad restartu)
    res = conn.execute("SELECT MAX(dec) FROM gaia_dr3").fetchone()
    start_dec = res[0] if res[0] is not None else DEC_MIN
    if start_dec < DEC_MIN: start_dec = DEC_MIN

    print(f"🚀 Štartujeme TURBO sťahovanie od Dec = {start_dec:.2f}°")
    t_start = time.time()

    current_dec = start_dec
    while current_dec < DEC_MAX:
        next_dec = min(current_dec + STEP_DEC, DEC_MAX)
        
        # ADQL dopyt optimalizovaný na pásový sken
        query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag as g_mag, 
               phot_bp_mean_mag as bp_mag, phot_rp_mean_mag as rp_mag,
               bp_rp, parallax, parallax_error, pmra, pmdec
        FROM gaiadr3.gaia_source
        WHERE dec >= {current_dec} AND dec < {next_dec}
        AND phot_g_mean_mag <= {MAG_LIMIT}
        """
        
        success = False
        for attempt in range(3):
            try:
                # Používame asynchrónny job pre stabilitu
                job = Gaia.launch_job_async(query)
                df = job.get_results().to_pandas()
                
                if not df.empty:
                    df.to_sql("gaia_dr3", conn, if_exists="append", index=False)
                    conn.commit()
                
                success = True
                break
            except Exception as e:
                print(f"⚠️ Chyba na Dec {current_dec:.2f}, pokus {attempt+1}: {e}")
                time.sleep(15)
        
        if success:
            elapsed = time.time() - t_start
            print(f"✅ Dec {current_dec: >5.2f}° -> {next_dec: >5.2f}° | "
                  f"Pridaných: {len(df): >6,d} | Čas: {elapsed/60:.1f} min")
            current_dec = next_dec
        else:
            print(f"❌ Kritické zlyhanie na Dec {current_dec}. Skúšam ďalší pás.")
            current_dec = next_dec

    print("⚡ Vytváram indexy (pre rýchle hľadanie vo VYVAR-e)...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_coords ON gaia_dr3 (ra, dec)")
    conn.execute("VACUUM")
    conn.close()
    print(f"✨ HOTOVO! Celkový čas: {(time.time()-t_start)/60:.1f} minút")

if __name__ == "__main__":
    build_turbo_db()