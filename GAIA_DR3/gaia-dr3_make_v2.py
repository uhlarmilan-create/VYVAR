"""
create_gaia_dr3_db.py
─────────────────────────────────────────────────────────────────────────────
Stiahne Gaia DR3 katalóg do lokálnej SQLite databázy.

SCHÉMA (tabuľka: gaia_dr3)
─────────────────────────────────────────────────────────────────────────────
Identifikácia:
  source_id             INTEGER   Gaia DR3 unikátne ID
  ra                    REAL      Rektascenzia (deg, ICRS)
  dec                   REAL      Deklinácia (deg, ICRS)

Fotometria:
  g_mag                 REAL      Gaia G magnitúda
  bp_mag                REAL      Gaia BP magnitúda
  rp_mag                REAL      Gaia RP magnitúda
  bp_rp                 REAL      BP-RP farebný index (bp_mag - rp_mag)
  g_flux_error_rel      REAL      Relatívna chyba G fluxu (quality flag)

Parallaxa / vzdialenosť:
  parallax              REAL      Parallaxa [mas] (d_pc = 1000/parallax)
  parallax_error        REAL      Chyba parallaxy [mas]
  parallax_over_error   REAL      SNR parallaxy (>5 = spoľahlivé)

Astrofyzikálne parametre (GSP-Phot — z fotometrie):
  teff_gspphot          REAL      Efektívna teplota [K]  (pre HRD x-os)
  logg_gspphot          REAL      Povrchová gravitácia log g  (rozlíšenie obrov/trpaslíkov)
  mh_gspphot            REAL      Metalicita [M/H]
  distance_gspphot      REAL      Vzdialenosť z GSP-Phot [pc] (alternatíva k parallax)

Premennosť / kvalita:
  var_flag              TEXT      'VARIABLE' / 'NOT_AVAILABLE' / 'CONSTANT'
  non_single_star       INTEGER   1 = NSS (dvojhviezda, astrometrická anomália)

SQL indexy:
  idx_ra, idx_dec       → rýchle cone search
  idx_source_id         → lookup podľa ID
  idx_parallax_snr      → filtrovanie spoľahlivých paraláx pre HRD
  idx_teff              → filtrovanie podľa teploty

POZNÁMKY:
  - Stĺpce sú striktne pomenované — nevzniknú konflikty pri náhrade starej DB
  - Stará DB (vyvar_gaia_dr3.db) má stĺpce: source_id, ra, dec, g_mag, bp_mag,
    rp_mag, bp_rp, var_flag, non_single_star, g_flux_error_rel
    → nová DB tieto stĺpce zachováva + pridáva nové
  - Resume funkcia: ak DB už existuje, preskočí stiahnuté HEALPix bunky
─────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import sqlite3

import pandas as pd
from astroquery.gaia import Gaia

# ── KONFIGURÁCIA ─────────────────────────────────────────────────────────────
DB_NAME     = "vyvar_gaia_dr3_v2.db"   # Nová DB — stará zostane nedotknutá
MAG_LIMIT   = 16.0                      # G magnitúda limit
DEC_MIN     = -20.0                     # Min deklinácia (deg)
DEC_MAX     = 90.0                      # Max deklinácia (deg)
BATCH_SIZE  = 100                       # Počet HEALPix buniek na dávku
MAX_RETRIES = 5                         # Počet pokusov pri chybe
# ─────────────────────────────────────────────────────────────────────────────

# Gaia archive nastavenia
Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1  # Bez limitu riadkov


def get_healpix_indices(dec_min: float, dec_max: float) -> list[int]:
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
    return sorted(res["hpx"].tolist())


def get_already_downloaded(conn: sqlite3.Connection) -> set:
    """Zistí ktoré HEALPix bunky sú už v DB (Resume funkcia)."""
    try:
        df = pd.read_sql(
            "SELECT DISTINCT (source_id / 34359738368) AS hpx FROM gaia_dr3", conn
        )
        return set(df["hpx"].tolist())
    except Exception:
        return set()


def create_table_if_not_exists(conn: sqlite3.Connection) -> None:
    """Vytvorí tabuľku gaia_dr3 ak neexistuje."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gaia_dr3 (
            -- Identifikácia
            source_id           INTEGER,
            ra                  REAL,
            dec                 REAL,

            -- Fotometria
            g_mag               REAL,
            bp_mag              REAL,
            rp_mag              REAL,
            bp_rp               REAL,
            g_flux_error_rel    REAL,

            -- Parallaxa
            parallax            REAL,
            parallax_error      REAL,
            parallax_over_error REAL,

            -- Astrofyzikálne parametre (GSP-Phot)
            teff_gspphot        REAL,
            logg_gspphot        REAL,
            mh_gspphot          REAL,
            distance_gspphot    REAL,

            -- Premennosť / kvalita
            var_flag            TEXT,
            non_single_star     INTEGER
        )
    """)
    conn.commit()


def build_batch_query(hpx_string: str) -> str:
    """Zostaví ADQL dopyt pre jednu dávku HEALPix buniek."""
    return f"""
    SELECT
        source_id,
        ra,
        dec,

        phot_g_mean_mag                                         AS g_mag,
        phot_bp_mean_mag                                        AS bp_mag,
        phot_rp_mean_mag                                        AS rp_mag,
        (phot_bp_mean_mag - phot_rp_mean_mag)                   AS bp_rp,
        (phot_g_mean_flux_error / phot_g_mean_flux)             AS g_flux_error_rel,

        parallax,
        parallax_error,
        parallax_over_error,

        teff_gspphot,
        logg_gspphot,
        mh_gspphot,
        distance_gspphot,

        phot_variable_flag                                      AS var_flag,
        non_single_star

    FROM gaiadr3.gaia_source
    WHERE phot_g_mean_mag <= {MAG_LIMIT}
    AND (source_id / 34359738368) IN ({hpx_string})
    """


def create_indexes(conn: sqlite3.Connection) -> None:
    """Vytvorí SQL indexy pre rýchle vyhľadávanie."""
    indexes = [
        ("idx_ra",            "CREATE INDEX IF NOT EXISTS idx_ra ON gaia_dr3 (ra)"),
        ("idx_dec",           "CREATE INDEX IF NOT EXISTS idx_dec ON gaia_dr3 (dec)"),
        ("idx_source_id",     "CREATE INDEX IF NOT EXISTS idx_source_id ON gaia_dr3 (source_id)"),
        ("idx_parallax_snr",  "CREATE INDEX IF NOT EXISTS idx_parallax_snr ON gaia_dr3 (parallax_over_error)"),
        ("idx_teff",          "CREATE INDEX IF NOT EXISTS idx_teff ON gaia_dr3 (teff_gspphot)"),
    ]
    for name, sql in indexes:
        print(f"  Index: {name}...")
        conn.execute(sql)
    conn.commit()


def create_local_gaia() -> None:
    print("=" * 60)
    print("  VYVAR — Gaia DR3 lokálna databáza (v2)")
    print("=" * 60)
    print(f"  Výstup:    {DB_NAME}")
    print(f"  Mag limit: G ≤ {MAG_LIMIT}")
    print(f"  Dec rozsah: {DEC_MIN}° — {DEC_MAX}°")
    print(f"  Batch size: {BATCH_SIZE} HEALPix buniek")
    print()

    # Otvor / vytvor DB
    conn = sqlite3.connect(DB_NAME)
    create_table_if_not_exists(conn)

    # Zisti čo treba stiahnuť
    all_hpx        = get_healpix_indices(DEC_MIN, DEC_MAX)
    downloaded_hpx = get_already_downloaded(conn)
    to_download    = [h for h in all_hpx if h not in downloaded_hpx]

    print(f"📊 HEALPix buniek celkovo : {len(all_hpx)}")
    print(f"✅ Už stiahnutých         : {len(downloaded_hpx)}")
    print(f"🚀 Zostáva stiahnuť       : {len(to_download)}")
    print()

    if not to_download:
        print("✨ Databáza je kompletná pre tento rozsah.")
    else:
        total_batches = (len(to_download) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx, i in enumerate(range(0, len(to_download), BATCH_SIZE)):
            batch       = to_download[i: i + BATCH_SIZE]
            hpx_string  = ", ".join(map(str, batch))
            batch_num   = batch_idx + 1

            print(f"📡 Dávka {batch_num}/{total_batches} "
                  f"({len(batch)} buniek)...", end=" ", flush=True)

            success = False
            for attempt in range(MAX_RETRIES):
                try:
                    query   = build_batch_query(hpx_string)
                    job     = Gaia.launch_job_async(query)
                    results = job.get_results()

                    if len(results) > 0:
                        df = results.to_pandas()

                        # Zabezpeč správne názvy stĺpcov
                        rename_map = {
                            "phot_g_mean_mag": "g_mag",
                            "phot_bp_mean_mag": "bp_mag",
                            "phot_rp_mean_mag": "rp_mag",
                            "phot_variable_flag": "var_flag",
                        }
                        df.rename(columns=rename_map, inplace=True)

                        # Ulož do DB
                        df.to_sql("gaia_dr3", conn, if_exists="append", index=False)
                        print(f"✓ +{len(df):,} hviezd")
                    else:
                        print("✓ prázdna dávka")

                    conn.commit()
                    success = True
                    break

                except Exception as e:
                    wait = (attempt + 1) * 10
                    print(f"\n  ⚠️  Pokus {attempt + 1}/{MAX_RETRIES} zlyhal: {e}")
                    print(f"  Čakám {wait}s...", end=" ", flush=True)
                    time.sleep(wait)

            if not success:
                print(f"\n  ❌ CHYBA: Dávka {batch_num} zlyhala po {MAX_RETRIES} pokusoch.")
                print("     Databáza je uložená — môžeš reštartovať skript (Resume).")

    # Indexy
    print()
    print("⚡ Vytváram SQL indexy...")
    create_indexes(conn)

    # VACUUM
    print("🧹 VACUUM — komprimovanie DB...")
    conn.execute("VACUUM")
    conn.close()

    # Záverečná štatistika
    size_gb = os.path.getsize(DB_NAME) / (1024 ** 3)
    print()
    print("=" * 60)
    print("  ✨ HOTOVO!")
    print(f"  Súbor:  {DB_NAME}")
    print(f"  Veľkosť: {size_gb:.2f} GB")
    print()
    print("  Nové stĺpce oproti starej DB:")
    print("    parallax, parallax_error, parallax_over_error")
    print("    teff_gspphot, logg_gspphot, mh_gspphot, distance_gspphot")
    print()
    print("  Pre HRD diagram použi:")
    print("    x-os: bp_rp")
    print("    y-os: g_mag - 5*log10(1000/parallax) + 5  [absolútna mag]")
    print("    filter: parallax_over_error > 5  [spoľahlivé vzdialenosti]")
    print("    farba:  teff_gspphot  [efektívna teplota]")
    print("=" * 60)


if __name__ == "__main__":
    create_local_gaia()