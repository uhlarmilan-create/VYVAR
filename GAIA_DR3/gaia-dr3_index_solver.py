"""
VYVAR — Blind Solver Index Builder (High Performance)
====================================================
Generuje gaia_triangles.pkl z lokálnej Gaia DR3 SQLite databázy.
Optimalizované pre hlboké katalógy (mag 13.5+) a nízku spotrebu RAM.
"""

import sqlite3
import sys
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import itertools
import pickle
import os
import time
import math
from tqdm import tqdm

# ─── KONFIGURÁCIA ─────────────────────────────────────────────────────────────

DB_PATH    = r"C:\ASTRO\python\VYVAR\GAIA_DR3\vyvar_gaia_dr3.db"
OUTPUT_PKL = r"C:\ASTRO\python\VYVAR\GAIA_DR3\gaia_triangles.pkl"

MAG_LIMIT       = 14.0  # Hĺbka katalógu
K_NEIGHBORS     = 8     # 5 susedov = 10 kombinácií na hviezdu (ideálny pomer rýchlosť/presnosť)
MIN_RATIO       = 0.15  # Filtruje "ihlicové" trojuholníky
# Euklidovská tolerancia v normalizovanom 3D priestore (r1,r2 v [0,1], log_L3_norm v [0,1])
TOLERANCE_UPPER = 0.002

# ─── POMOCNÉ FUNKCIE ──────────────────────────────────────────────────────────

def triangle_hash(p0, p1, p2):
    """3D hash: tvar (L1/L3, L2/L3) + log10 fyzickej veľkosti L3 v arcsekundách.

    p0/p1/p2 sú body v 3D kartézskych súradniciach na jednotkovej sfére.
    Euklidovská vzdialenosť medzi bodmi je pre malé uhly ~ θ (v radiánoch),
    preto L3 prepočítame na arcsekundy cez radians → degrees → arcsec.
    """
    d01 = math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)
    d12 = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    d02 = math.sqrt((p0[0] - p2[0]) ** 2 + (p0[1] - p2[1]) ** 2 + (p0[2] - p2[2]) ** 2)
    sides = sorted([d01, d12, d02])
    L1, L2, L3 = sides
    if L3 < 1e-8:
        return None
    r1, r2 = L1 / L3, L2 / L3
    if r1 < MIN_RATIO:
        return None
    # d ≈ θ (rad) pre malé uhly (na jednotkovej sfére)
    L3_arcsec = L3 * (180.0 / math.pi) * 3600.0
    if L3_arcsec < 0.1:
        return None
    log_L3 = math.log10(L3_arcsec)
    return (float(r1), float(r2), float(log_L3))

# ─── HLAVNÁ FUNKCIA ───────────────────────────────────────────────────────────

def build_blind_index():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print(f"{'='*60}\n VYVAR Blind Solver — Index Builder (Optimized)\n{'='*60}")

    if not os.path.exists(DB_PATH):
        print(f"❌ Databáza nenájdená: {DB_PATH}")
        return

    # 1. Načítanie dát
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT ra, dec FROM gaia_dr3 WHERE g_mag <= {MAG_LIMIT}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    n_stars = len(df)
    print(f"✅ Načítaných {n_stars:,} hviezd za {time.time()-t0:.1f}s")

    # 2. Projekcia súradníc pre KDTree susedov
    # 3D kartézske súradnice na jednotkovej sfére eliminujú RA wrap-around (0°/360°).
    ra_rad = np.radians(df["ra"].values)
    dec_rad = np.radians(df["dec"].values)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    coords_search = np.column_stack([x, y, z]).astype(np.float64)
    coords_real = df[["ra", "dec"]].values.astype(np.float32)

    # 3. KDTree pre susedov
    print("🌳 Staviam KDTree susedov...")
    neighbor_tree = KDTree(coords_search)

    # 4. Vektorizované hľadanie susedov (Extrémne rýchle)
    print(f"🔍 Hľadám {K_NEIGHBORS} najbližších susedov pre každú hviezdu...")
    _, all_indices = neighbor_tree.query(coords_search, k=K_NEIGHBORS)

    # 5. Generovanie hashov
    hashes_list = []
    metadata_list = []
    
    # Predvypočítané kombinácie indexov (napr. 0,1,2; 0,1,3...)
    combos = list(itertools.combinations(range(K_NEIGHBORS), 3))
    
    print(f"🔺 Generujem trojuholníky...")
    # NEPOUŽÍVAME seen_combos set -> šetríme gigabajty RAM a CPU čas
    for i in tqdm(range(n_stars)):
        neighbor_idx = all_indices[i]
        
        for c in combos:
            # i_tri obsahuje indexy 3 hviezd v hlavnom poli
            i_tri = neighbor_idx[list(c)]
            
            # Aby sme minimalizovali duplicity bez set(), 
            # spracujeme trojuholník len vtedy, ak 'i' je najmenší z jeho indexov
            if i != np.min(i_tri):
                continue
                
            p = coords_search[i_tri]
            h = triangle_hash(p[0], p[1], p[2])

            if h:
                hashes_list.append(h)
                # 8 hodnôt: [ra_centroid, dec_centroid, ra_A, dec_A, ra_B, dec_B, ra_C, dec_C]
                cr = coords_real[i_tri]
                metadata_list.append(
                    [
                        float(cr[:, 0].mean()),
                        float(cr[:, 1].mean()),
                        float(cr[0, 0]),
                        float(cr[0, 1]),
                        float(cr[1, 0]),
                        float(cr[1, 1]),
                        float(cr[2, 0]),
                        float(cr[2, 1]),
                    ]
                )

    # 6. Finálna štruktúra: 3D hash (r1, r2, log_L3_norm) + metadata (8): ra_c,dec_c, …
    print(f"🔢 Konvertujem {len(hashes_list):,} trojuholníkov...")
    hashes_arr = np.array(hashes_list, dtype=np.float32)
    metadata_arr = np.array(metadata_list, dtype=np.float32)

    log_L3_min = float(hashes_arr[:, 2].min())
    log_L3_max = float(hashes_arr[:, 2].max())
    log_L3_range = max(log_L3_max - log_L3_min, 1e-6)
    hashes_arr[:, 2] = (hashes_arr[:, 2] - log_L3_min) / log_L3_range

    print("🌳 Staviam finálny 3D Hash-Tree (normalizovaný log L3)...")
    hash_tree = KDTree(hashes_arr)

    # 7. Uloženie
    index_data = {
        "tree": hash_tree,
        "metadata": metadata_arr,
        "mag_limit": MAG_LIMIT,
        "tolerance": TOLERANCE_UPPER,
        "hash_dim": 3,
        "log_L3_min": log_L3_min,
        "log_L3_max": log_L3_max,
    }

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_time = time.time() - t0
    print(f"{'='*60}\n✅ HOTOVO! Súbor: {os.path.basename(OUTPUT_PKL)}\n"
          f"⏱️ Čas: {total_time:.1f}s | Trojuholníky: {len(hashes_arr):,}\n{'='*60}")

if __name__ == "__main__":
    build_blind_index()