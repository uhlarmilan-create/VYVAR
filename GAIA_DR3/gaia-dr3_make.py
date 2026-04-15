import pandas as pd
from astroquery.gaia import Gaia
import sqlite3
import os
import time

# --- KONFIGURÁCIA ---
DB_NAME = "vyvar_gaia_dr3.db"
MAG_LIMIT = 16.0
DEC_MIN = -20.0
DEC_MAX = 90.0
LEVEL = 5          # HEALPix úroveň
BATCH_SIZE = 100   # Počet buniek v jednom dopyte (optimalizované pre rýchlosť)

def get_healpix_indices(dec_min, dec_max):
    """Získa zoznam unikátnych HEALPix indexov pre daný rozsah deklinácie."""
    query = f"""
    SELECT DISTINCT source_id / 34359738368 AS hpx
    FROM gaiadr3.gaia_source
    WHERE dec BETWEEN {dec_min} AND {dec_max}
    AND phot_g_mean_mag <= 10
    """
    print(f"🔍 Prehľadávam mapu oblohy pre Dec {dec_min}° až {dec_max}°...")
    job = Gaia.launch_job_async(query)
    res = job.get_results()
    return sorted(res['hpx'].tolist())

def get_already_downloaded(conn):
    """Zistí, ktoré bunky už máme v databáze (pre funkciu Resume)."""
    try:
        query = "SELECT DISTINCT (source_id / 34359738368) as hpx FROM gaia_dr3"
        df = pd.read_sql(query, conn)
        return set(df['hpx'].tolist())
    except:
        return set()

def create_local_gaia():
    # Vytvoríme/Otvoríme DB (nevymazávame ju, aby fungovalo Resume)
    conn = sqlite3.connect(DB_NAME)
    
    # 1. Získame cieľové indexy
    all_hpx = get_healpix_indices(DEC_MIN, DEC_MAX)
    
    # 2. Zistíme, čo už máme stiahnuté
    downloaded_hpx = get_already_downloaded(conn)
    to_download = [h for h in all_hpx if h not in downloaded_hpx]
    
    total_total = len(all_hpx)
    print(f"📊 Celkový rozsah: {total_total} buniek.")
    print(f"✅ Už hotovo: {len(downloaded_hpx)} buniek.")
    print(f"🚀 Zostáva stiahnuť: {len(to_download)} buniek.")

    if not to_download:
        print("✨ Databáza sa zdá byť kompletná pre tento rozsah.")
    else:
        # 3. Sťahujeme v dávkach (Batches)
        for i in range(0, len(to_download), BATCH_SIZE):
            batch = to_download[i : i + BATCH_SIZE]
            hpx_string = ", ".join(map(str, batch))
            
            print(f"📡 Dávka {i//BATCH_SIZE + 1}/{(len(to_download)//BATCH_SIZE)+1}...")
            
            query = f"""
            SELECT source_id, ra, dec, phot_g_mean_mag as g_mag, phot_bp_mean_mag as bp_mag, 
                   phot_rp_mean_mag as rp_mag, (phot_bp_mean_mag - phot_rp_mean_mag) as bp_rp,
                   phot_variable_flag as var_flag, non_single_star,
                   (phot_g_mean_flux_error / phot_g_mean_flux) as g_flux_error_rel
            FROM gaiadr3.gaia_source
            WHERE phot_g_mean_mag <= {MAG_LIMIT}
            AND (source_id / 34359738368) IN ({hpx_string})
            """
            
            success = False
            for attempt in range(5): # Viac pokusov pre stabilitu
                try:
                    job = Gaia.launch_job_async(query)
                    results = job.get_results()
                    if len(results) > 0:
                        df = results.to_pandas()
                        df.to_sql("gaia_dr3", conn, if_exists="append", index=False)
                        print(f"   ✅ OK: +{len(df)} hviezd pridaných.")
                    else:
                        print("   ℹ️ Dávka bola prázdna (žiadne hviezdy do 16 mag v tejto oblasti).")
                    
                    success = True
                    break
                except Exception as e:
                    wait_time = (attempt + 1) * 10
                    print(f"   ⚠️ Pokus {attempt+1} zlyhal ({e}). Čakám {wait_time}s...")
                    time.sleep(wait_time)
            
            if not success:
                print(f"   ❌ KRITICKÁ CHYBA: Dávka buniek {hpx_string[:20]}... zlyhala.")

    # 4. Záverečná optimalizácia (Indexy)
    print("\n⚡ Kontrola a vytváranie SQL indexov (toto môže chvíľu trvať)...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ra ON gaia_dr3 (ra)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dec ON gaia_dr3 (dec)")
    
    # VACUUM skomprimuje DB a uvoľní miesto
    print("🧹 Optimalizujem veľkosť súboru (VACUUM)...")
    conn.execute("VACUUM")
    
    conn.close()
    
    size_gb = os.path.getsize(DB_NAME) / (1024**3)
    print("✨ HOTOVO! Lokálna Gaia DB je pripravená.")
    print(f"📂 Finálna veľkosť: {size_gb:.2f} GB")

if __name__ == "__main__":
    create_local_gaia()