from __future__ import annotations

import argparse
import sqlite3
import time

import pandas as pd

# --- KONFIGURÁCIA (default; prepísateľné cez CLI) ---
DB_NAME = "vyvar_vsx_local.db"
MAG_LIMIT = 18.0          # rez podľa VSX `max` (jas v maxime); zdvihni pre slabšie hviezdy
DEC_MIN = -90.0
DEC_MAX = 90.0
BATCH_SIZE_DEG = 1.0      # 1° pásy = stabilnejšie, menej timeoutov z VizieR

# Stĺpce, ktoré VYVAR z `vsx_data` číta (database.query_local_vsx / validate_vsx_local_db_schema):
#   TVRDO povinné : oid, ra_deg, dec_deg
#   používané     : name, var_type, mag_max, mag_min
#   voliteľné     : period  (VYVAR ho vezme ak existuje)
SCHEMA = """
CREATE TABLE IF NOT EXISTS vsx_data (
    oid       INTEGER PRIMARY KEY,   -- VSX OID; PK zaisťuje reálne INSERT OR IGNORE
    name      TEXT,
    ra_deg    REAL,
    dec_deg   REAL,
    var_type  TEXT,
    period    REAL,                  -- dni; NaN ak VSX periódu neuvádza
    mag_max   REAL,
    mag_min   REAL
);
"""

_INSERT = (
    "INSERT OR IGNORE INTO vsx_data "
    "(oid, name, ra_deg, dec_deg, var_type, period, mag_max, mag_min) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)
_COLS = ["oid", "name", "ra_deg", "dec_deg", "var_type", "period", "mag_max", "mag_min"]


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(SCHEMA)
    # Spatial indexy — VYVAR ich síce vytvorí pri prvom query, ale tu je to lacné a čistejšie.
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vsx_ra ON vsx_data (ra_deg);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vsx_dec ON vsx_data (dec_deg);")
    conn.commit()


def _count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM vsx_data;").fetchone()[0])


def build_vsx(
    db_name: str,
    *,
    dec_min: float,
    dec_max: float,
    mag_limit: float,
    batch_deg: float,
) -> None:
    """Stiahne VSX (VizieR B/vsx/vsx) po Dec pásoch do `vsx_data`.

    Bezpečne re-runnovateľné a inkrementálne: vďaka PK na `oid` a `INSERT OR IGNORE`
    sa duplikáty preskočia. Na dobudovanie slabších hviezd stačí znovu spustiť so
    zdvihnutým ``--mag-limit`` — pribudnú len nové OID.
    """
    from astroquery.vizier import Vizier  # type: ignore[import-not-found]

    v = Vizier(
        columns=["OID", "Name", "RAJ2000", "DEJ2000", "Type", "Period", "max", "min"],
        row_limit=-1,
    )

    conn = sqlite3.connect(db_name)
    try:
        _ensure_schema(conn)
        start_count = _count(conn)

        current_dec = float(dec_min)
        while current_dec < dec_max:
            next_dec = min(current_dec + batch_deg, dec_max)
            print(f"🔍 Spracovávam pás Dec: {current_dec}° až {next_dec}°...")

            query_filter = {
                "DEJ2000": f"{current_dec}..{next_dec}",
                "max": f"<{mag_limit}",
            }

            for attempt in range(5):
                try:
                    result = v.query_constraints(catalog="B/vsx/vsx", **query_filter)

                    if result and len(result) > 0:
                        df = result[0].to_pandas().rename(
                            columns={
                                "OID": "oid",
                                "Name": "name",
                                "RAJ2000": "ra_deg",
                                "DEJ2000": "dec_deg",
                                "Type": "var_type",
                                "Period": "period",
                                "max": "mag_max",
                                "min": "mag_min",
                            }
                        )

                        # Doplň prípadne chýbajúce stĺpce (rôzne verzie VizieR vrátia rôzne sady).
                        for c in _COLS:
                            if c not in df.columns:
                                df[c] = None

                        # Numerická koercia + zahodenie riadkov bez OID/RA/Dec (nepoužiteľné).
                        for c in ("oid", "ra_deg", "dec_deg", "period", "mag_max", "mag_min"):
                            df[c] = pd.to_numeric(df[c], errors="coerce")
                        df = df.dropna(subset=["oid", "ra_deg", "dec_deg"]).copy()
                        df["oid"] = df["oid"].astype("int64")
                        df["name"] = df["name"].astype("string")
                        df["var_type"] = df["var_type"].astype("string")

                        rows = [
                            (
                                int(r.oid),
                                None if pd.isna(r.name) else str(r.name),
                                float(r.ra_deg),
                                float(r.dec_deg),
                                None if pd.isna(r.var_type) else str(r.var_type),
                                None if pd.isna(r.period) else float(r.period),
                                None if pd.isna(r.mag_max) else float(r.mag_max),
                                None if pd.isna(r.mag_min) else float(r.mag_min),
                            )
                            for r in df[_COLS].itertuples(index=False)
                        ]

                        before = _count(conn)
                        conn.executemany(_INSERT, rows)
                        conn.commit()
                        added = _count(conn) - before
                        print(
                            f"   ✅ {len(rows)} hviezd v páse "
                            f"({added} nových, {len(rows) - added} duplikátov ignorovaných)."
                        )
                    else:
                        print("   ℹ️ Pás bol prázdny.")
                    break

                except Exception as e:  # noqa: BLE001
                    wait_time = (attempt + 1) * 20
                    print(f"   ⚠️ Pokus {attempt + 1} zlyhal: {e}. Čakám {wait_time}s...")
                    time.sleep(wait_time)

            current_dec = next_dec

        print("\n⚡ Finalizácia (VACUUM)...")
        conn.execute("VACUUM")
        end_count = _count(conn)
        print(
            f"✨ Hotovo! vsx_data: {end_count} riadkov "
            f"(+{end_count - start_count} pridaných v tomto behu)."
        )
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VYVAR VSX catalog builder — čisto VSX (VizieR B/vsx/vsx). "
        "Žiadne APASS/Tycho / B-V tabuľky. Re-runnovateľné a inkrementálne."
    )
    parser.add_argument("--db", default=DB_NAME, help=f"Cesta k SQLite DB (default: {DB_NAME})")
    parser.add_argument("--dec-min", type=float, default=DEC_MIN)
    parser.add_argument("--dec-max", type=float, default=DEC_MAX)
    parser.add_argument(
        "--mag-limit",
        type=float,
        default=MAG_LIMIT,
        help=f"Rez podľa VSX `max` (default {MAG_LIMIT}). Zdvihni pre slabšie hviezdy.",
    )
    parser.add_argument("--batch-deg", type=float, default=BATCH_SIZE_DEG)
    args = parser.parse_args()

    db_name = str(args.db).strip() or DB_NAME

    print("=== VSX (Gaia+VSX only) ===")
    print(f"DB={db_name}  Dec=[{args.dec_min}, {args.dec_max}]  mag<{args.mag_limit}")
    build_vsx(
        db_name,
        dec_min=float(args.dec_min),
        dec_max=float(args.dec_max),
        mag_limit=float(args.mag_limit),
        batch_deg=float(args.batch_deg),
    )


if __name__ == "__main__":
    main()