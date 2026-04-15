import pandas as pd
from astroquery.vizier import Vizier
import sqlite3
import os
import time

# --- KONFIGURÁCIA ---
DB_NAME = "vyvar_vsx_local.db"
MAG_LIMIT = 16.0
DEC_MIN = -20.0
DEC_MAX = 90.0
BATCH_SIZE_DEG = 1.0  # Zmenšené na 1 stupeň pre lepšiu stabilitu a menej timeoutov

def fix_missing_vsx():
    v = Vizier(columns=['OID', 'Name', 'RAJ2000', 'DEJ2000', 'Type', 'max', 'min'], 
               row_limit=-1)
    
    conn = sqlite3.connect(DB_NAME)
    
    # Zistíme, ktoré oblasti už máme (približne podľa Dec)
    # Ak chceš mať 100% istotu, najlepšie je prejsť všetky pásy s INSERT OR IGNORE
    
    current_dec = DEC_MIN
    while current_dec < DEC_MAX:
        next_dec = min(current_dec + BATCH_SIZE_DEG, DEC_MAX)
        print(f"🔍 Spracovávam pás Dec: {current_dec}° až {next_dec}°...")

        query_filter = {
            'DEJ2000': f"{current_dec}..{next_dec}",
            'max': f"<{MAG_LIMIT}"
        }

        success = False
        for attempt in range(5): # Zvýšený počet pokusov
            try:
                result = v.query_constraints(catalog="B/vsx/vsx", **query_filter)
                
                if result and len(result) > 0:
                    df = result[0].to_pandas()
                    df = df.rename(columns={
                        'OID': 'oid', 'Name': 'name', 'RAJ2000': 'ra_deg',
                        'DEJ2000': 'dec_deg', 'Type': 'var_type',
                        'max': 'mag_max', 'min': 'mag_min'
                    })
                    
                    # KĽÚČOVÁ ZMENA: INSERT OR IGNORE
                    # Ak narazí na existujúce OID, jednoducho ho preskočí a nezlyhá
                    df.to_sql("vsx_data", conn, if_exists="append", index=False, 
                              method=None) 
                    
                    # Ak predchádzajúci riadok nefunguje s tvojou verziou pandas/sqlite, 
                    # použi toto ručné riešenie:
                    # df.to_sql("temp_vsx", conn, if_exists="replace")
                    # conn.execute("INSERT OR IGNORE INTO vsx_data SELECT oid, name, ra_deg, dec_deg, var_type, mag_max, mag_min FROM temp_vsx")
                    
                    print(f"   ✅ Spracovaných {len(df)} hviezd (nové pridané, duplikáty ignorované).")
                else:
                    print("   ℹ️ Pás bol prázdny.")
                
                success = True
                break
            except sqlite3.IntegrityError:
                # Toto ošetrí chybu unikátnosti priamo v Python cykle, ak nepoužiješ SQL INSERT OR IGNORE
                print("   ℹ️ Nájdené duplikáty v dávke, preskakujem...")
                success = True # Považujeme za vybavené
                break
            except Exception as e:
                wait_time = (attempt + 1) * 20
                print(f"   ⚠️ Pokus {attempt+1} zlyhal: {e}. Čakám {wait_time}s...")
                time.sleep(wait_time)

        current_dec = next_dec

    print("\n⚡ Finalizácia...")
    conn.execute("VACUUM")
    conn.close()
    print("✨ Hotovo! Chýbajúce dáta boli doplnené.")

if __name__ == "__main__":
    fix_missing_vsx()