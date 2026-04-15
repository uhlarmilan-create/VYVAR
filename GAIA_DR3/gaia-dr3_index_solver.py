import sqlite3
import numpy as np
import pandas as pd
from scipy.spatial import KDTree, distance
import itertools
import pickle
import os

def build_blind_index(db_path, output_pkl=r"C:\ASTRO\python\VYVAR\GAIA_DR3\gaia_triangles.pkl"):
    # Skontrolujeme, či DB existuje
    if not os.path.exists(db_path):
        print(f"❌ Chyba: Databáza {db_path} nebola nájdená.")
        return

    conn = sqlite3.connect(db_path)
    # Vyberáme hviezdy do 12 mag pre stabilitu vzorov
    query = "SELECT ra, dec, g_mag FROM gaia_dr3 WHERE g_mag <= 12.0"
    df = pd.read_sql(query, conn)
    conn.close()

    # Pre výpočet vzdialeností na sfére (približný pre malé polia)
    # ra_adj kompenzuje zužovanie poludníkov smerom k pólu
    df['ra_adj'] = df['ra'] * np.cos(np.radians(df['dec']))
    coords_for_search = df[['ra_adj', 'dec']].to_numpy() # Na hľadanie susedov
    coords_real = df[['ra', 'dec']].to_numpy()      # Skutočné RA/Dec

    tree = KDTree(coords_for_search)
    
    hashes = []
    metadata = []
    seen_combos = set() # Zabráni duplicite trojuholníkov

    print(f"🔍 Vytváram index z {len(df)} hviezd...")

    for i in range(len(coords_for_search)):
        # k=6 (hviezda + 5 najbližších susedov)
        _, indices = tree.query(coords_for_search[i], k=6)
        
        # itertools.combinations zabezpečí unikátne trojice v rámci okolia
        for combo in itertools.combinations(sorted(indices), 3):
            if combo in seen_combos:
                continue
            seen_combos.add(combo)
            
            p = coords_for_search[list(combo)]
            # Euklidovská vzdialenosť na premietnutej ploche (približná)
            d = [distance.euclidean(p[0], p[1]), 
                 distance.euclidean(p[1], p[2]), 
                 distance.euclidean(p[0], p[2])]
            d.sort()
            
            # Ochrana pred nulou a príliš malými trojuholníkmi
            if d[2] > 1e-6:
                hashes.append([d[0]/d[2], d[1]/d[2]])
                # Uložíme reálny stred (zo skutočných RA/Dec)
                metadata.append(np.mean(coords_real[list(combo)], axis=0))

    # Konverzia na numpy pre KDTree
    hashes_arr = np.array(hashes, dtype=np.float32)
    metadata_arr = np.array(metadata, dtype=np.float32)

    index_data = {
        'tree': KDTree(hashes_arr),
        'metadata': metadata_arr
    }
    
    # Uloženie
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(index_data, f)
    
    print(f"✅ Blind Index úspešne uložený.")
    print(f"📊 Počet unikátnych trojuholníkov: {len(hashes)}")

# Spustenie (uprav cestu k DB podľa reality)
if __name__ == "__main__":
    db_file = r"C:\ASTRO\python\VYVAR\GAIA_DR3\vyvar_gaia_dr3.db"
    build_blind_index(db_file)