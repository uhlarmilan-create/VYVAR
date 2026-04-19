"""
VYVAR — Blind Solver Index Builder
===================================
Generuje gaia_triangles.pkl z lokálnej Gaia DR3 SQLite databázy.

Spustenie:
    python build_gaia_blind_index.py

Výstup:
    C:\\ASTRO\\python\\VYVAR\\GAIA_DR3\\gaia_triangles.pkl

Požiadavky:
    pip install numpy scipy pandas tqdm
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import itertools
import pickle
import os
import time
import math

# ─── KONFIGURÁCIA ─────────────────────────────────────────────────────────────

DB_PATH    = r"C:\ASTRO\python\VYVAR\GAIA_DR3\vyvar_gaia_dr3.db"
OUTPUT_PKL = r"C:\ASTRO\python\VYVAR\GAIA_DR3\gaia_triangles.pkl"

MAG_LIMIT       = 12.0   # Hviezdy jasnejšie ako táto hodnota idú do indexu
K_NEIGHBORS     = 7      # Počet susedov (vrátane samotnej hviezdy) → C(6,3)=20 trojuholníkov
MIN_RATIO       = 0.15   # Minimálny pomer L1/L3 (filtruje príliš štíhle trojuholníky)
TOLERANCE_UPPER = 0.005  # Horná tolerancia pri vyhľadávaní (použiješ v solveri)

# ─── POMOCNÉ FUNKCIE ──────────────────────────────────────────────────────────

def angular_distance_approx(p1, p2):
    """
    Aproximatívna uhlová vzdialenosť v stupňoch (ra_adj, dec) priestore.
    Presná pre malé uhly (< 20°).
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def triangle_hash(p0, p1, p2):
    """
    Vypočíta trojuholníkový hash (L1/L3, L2/L3) invariantný voči
    rotácii, mierke a posunu.

    Vstup: tri body v (ra_adj, dec) priestore.
    Výstup: (r1, r2) tuple alebo None ak je trojuholník degenerovaný.
    """
    d01 = angular_distance_approx(p0, p1)
    d12 = angular_distance_approx(p1, p2)
    d02 = angular_distance_approx(p0, p2)

    sides = sorted([d01, d12, d02])  # [L1, L2, L3], L1 ≤ L2 ≤ L3

    L1, L2, L3 = sides
    if L3 < 1e-7:
        return None  # Degenerovaný (hviezdy na rovnakom mieste)

    r1 = L1 / L3
    r2 = L2 / L3

    if r1 < MIN_RATIO:
        return None  # Príliš štíhly trojuholník → nestabilný hash

    return (r1, r2)


# ─── HLAVNÁ FUNKCIA ───────────────────────────────────────────────────────────

def build_blind_index(db_path: str, output_pkl: str):
    print("=" * 60)
    print("  VYVAR Blind Solver — Index Builder")
    print("=" * 60)

    # 1. Načítanie hviezd z DB
    if not os.path.exists(db_path):
        print(f"❌  Databáza nenájdená: {db_path}")
        return

    print(f"\n📂  Načítavam hviezdy (g_mag ≤ {MAG_LIMIT}) z:\n    {db_path}")
    t0 = time.time()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        f"SELECT ra, dec, g_mag FROM gaia_dr3 WHERE g_mag <= {MAG_LIMIT} ORDER BY g_mag",
        conn,
    )
    conn.close()

    n_stars = len(df)
    print(f"✅  Načítaných hviezd: {n_stars:,}  ({time.time()-t0:.1f}s)")

    if n_stars < 10:
        print("❌  Príliš málo hviezd. Skontroluj DB a MAG_LIMIT.")
        return

    # 2. Príprava súradníc
    # ra_adj kompenzuje zužovanie poludníkov smerom k pólu
    df["ra_adj"] = df["ra"] * np.cos(np.radians(df["dec"]))

    coords_search = df[["ra_adj", "dec"]].to_numpy(dtype=np.float64)   # Pre KDTree susedov
    coords_real   = df[["ra",     "dec"]].to_numpy(dtype=np.float32)   # Pre metadata (ukladáme)

    # 3. Stavba KDTree pre hľadanie susedov
    print(f"\n🌳  Staviam KDTree pre {n_stars:,} hviezd...")
    t1 = time.time()
    neighbor_tree = KDTree(coords_search)
    print(f"✅  KDTree hotový ({time.time()-t1:.1f}s)")

    # 4. Generovanie trojuholníkov
    print(f"\n🔺  Generujem trojuholníky (k={K_NEIGHBORS-1} susedov)...")
    print(f"    Odhadovaný počet trojuholníkov: ~{n_stars * 20 // 3:,}")

    hashes_list   = []
    metadata_list = []
    seen_combos   = set()

    t2 = time.time()
    report_every = max(1, n_stars // 20)  # Výpis každých 5%

    # Pokus o tqdm, ak nie je dostupný, ide bez neho
    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_stars), desc="Hviezdy", unit="⭐")
    except ImportError:
        iterator = range(n_stars)

    for i in iterator:
        _, indices = neighbor_tree.query(coords_search[i], k=K_NEIGHBORS)

        for combo in itertools.combinations(sorted(indices.tolist()), 3):
            if combo in seen_combos:
                continue
            seen_combos.add(combo)

            p = coords_search[list(combo)]
            h = triangle_hash(p[0], p[1], p[2])
            if h is None:
                continue

            hashes_list.append(h)
            # Stred trojuholníka v skutočných RA/Dec súradniciach
            metadata_list.append(coords_real[list(combo)].mean(axis=0))

        # Progress bez tqdm
        if not hasattr(iterator, '__tqdm__') and (i % report_every == 0):
            pct = 100 * i / n_stars
            elapsed = time.time() - t2
            eta = (elapsed / max(i, 1)) * (n_stars - i)
            print(f"    {pct:5.1f}%  trojuholníkov: {len(hashes_list):,}  ETA: {eta:.0f}s")

    elapsed_gen = time.time() - t2
    n_triangles = len(hashes_list)
    print(f"\n✅  Trojuholníkov vygenerovaných: {n_triangles:,}  ({elapsed_gen:.1f}s)")

    if n_triangles == 0:
        print("❌  Žiadne trojuholníky. Skontroluj parametre.")
        return

    # 5. Konverzia na numpy arrays
    print("\n🔢  Konvertujem na numpy arrays...")
    hashes_arr   = np.array(hashes_list,   dtype=np.float32)
    metadata_arr = np.array(metadata_list, dtype=np.float32)

    mem_mb = (hashes_arr.nbytes + metadata_arr.nbytes) / 1024**2
    print(f"    Pamäť arrays: {mem_mb:.1f} MB")

    # 6. Stavba finálneho KDTree pre hľadanie
    print("🌳  Staviam finálny KDTree na hashoch...")
    t3 = time.time()
    hash_tree = KDTree(hashes_arr)
    print(f"✅  Hash KDTree hotový ({time.time()-t3:.1f}s)")

    # 7. Uloženie
    index_data = {
        "tree":          hash_tree,
        "metadata":      metadata_arr,   # shape (N, 2): [ra, dec] stredu trojuholníka
        "mag_limit":     MAG_LIMIT,
        "k_neighbors":   K_NEIGHBORS,
        "min_ratio":     MIN_RATIO,
        "tolerance":     TOLERANCE_UPPER,
        "n_stars":       n_stars,
        "n_triangles":   n_triangles,
    }

    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    print(f"\n💾  Ukladám do:\n    {output_pkl}")
    t4 = time.time()
    with open(output_pkl, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_pkl) / 1024**2
    total   = time.time() - t0

    print(f"✅  Hotovo!")
    print()
    print("=" * 60)
    print(f"  📊  Hviezd v indexe:      {n_stars:>10,}")
    print(f"  🔺  Trojuholníkov:         {n_triangles:>10,}")
    print(f"  📦  Veľkosť súboru:        {size_mb:>9.1f} MB")
    print(f"  ⏱️   Celkový čas:           {total:>9.1f} s")
    print("=" * 60)
    print()
    print("ℹ️   Ďalší krok: použij tento index v solve_completely_blind()")
    print(f"    Tolerancia pri vyhľadávaní: distance_upper_bound={TOLERANCE_UPPER}")


# ─── VSTUPNÝ BOD ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_blind_index(DB_PATH, OUTPUT_PKL)