import pandas as pd
from astroquery.gaia import Gaia
import sqlite3
import time

# --- KONFIGURÁCIA ---
DB_NAME = "vyvar_gaia_dr3.db"
MAG_LIMIT = 16.0
DEC_MIN = -20.0
DEC_MAX = 90.0
STEP = 1.0  # 1-stupňové pásy sú pre server "malina"

def create_local_gaia():
    conn = sqlite3.connect(DB_NAME)
    
    # Zistíme, kde sme skončili (Resume)
    try:
        last_dec = pd.read_sql("SELECT MAX(dec) FROM gaia_dr3", conn).iloc[0,0]
        if last_dec is None: last_dec = DEC_MIN
        print(f"🔄 Nadväzujem na deklinácii: {last_dec:.2f}°")
    except:
        last_dec = DEC_MIN
        print("🆕 Začínam novú databázu.")

    current_dec = last_dec
    
    while current_dec < DEC_MAX:
        next_dec = min(current_dec + STEP, DEC_MAX)
        print(f"📡 Sťahujem pás: Dec {current_dec:.1f}° až {next_dec:.1f}°...")
        
        query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag as g_mag, phot_bp_mean_mag as bp_mag, 
               phot_rp_mean_mag as rp_mag, (phot_bp_mean_mag - phot_rp_mean_mag) as bp_rp,
               phot_variable_flag as var_flag, non_single_star,
               (phot_g_mean_flux_error / phot_g_mean_flux) as g_flux_error_rel
        FROM gaiadr3.gaia_source
        WHERE phot_g_mean_mag <= {MAG_LIMIT}
        AND dec >= {current_dec} AND dec < {next_dec}
        """
        
        success = False
        for attempt in range(5):
            try:
                # Navýšime limit riadkov pre istotu
                Gaia.ROW_LIMIT = 200000 
                job = Gaia.launch_job_async(query)
                df = job.get_results().to_pandas()
                
                if not df.empty:
                    df.to_sql("gaia_dr3", conn, if_exists="append", index=False)
                    print(f"   ✅ OK: +{len(df)} hviezd.")
                
                success = True
                break
            except Exception as e:
                print(f"   ⚠️ Chyba ({e}). Pokus {attempt+1}. Čakám...")
                time.sleep(10)
        
        if success:
            current_dec = next_dec
        else:
            print("❌ Server neodpovedá, skúste to o chvíľu.")
            break

    print("\n⚡ Indexovanie (toto urobí Plate-solve bleskovým)...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dec_ra ON gaia_dr3 (dec, ra)")
    conn.execute("VACUUM")
    conn.close()
    print("✨ HOTOVO!")

if __name__ == "__main__":
    create_local_gaia()