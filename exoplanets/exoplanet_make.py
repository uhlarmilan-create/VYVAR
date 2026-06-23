from __future__ import annotations

import argparse
import io
import sqlite3
import time
import urllib.parse
import urllib.request

import pandas as pd

# --- KONFIGURACIA (default; prepisatelne cez CLI) ---
DB_NAME = "vyvar_exoplanet_local.db"
MAG_LIMIT = None          # rez podla TESS/V magnitudy hostitela; None = bez rezu (cely katalog)
TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
HTTP_TIMEOUT_S = 300      # NEA TAP byva pomalsie pri *; tabulky su male, ale necháme rezervu

# Zdrojove NEA TAP tabulky:
#   pscomppars = potvrdene planety, JEDEN riadok na planetu (composite parametre)
#   toi        = TESS Project Candidates (kandidati PC/CP/KP/APC + FP/FA, jeden riadok na TOI)
#
# DESIRED_COLS = co chceme z kazdej tabulky (NEA nazov -> nasa unifikovana schema).
# Skript NEHADA: pred stiahnutim overi proti zivej TAP_SCHEMA, ze stlpce existuju.
# Povinne (cross-match nepouzitelny bez nich): ra, dec + identifikator.
# Volitelne (ak NEA stlpec chyba, doplni sa NULL): period, mag, host, disposition.

# Stlpce, ktore VYVAR z `exoplanet_data` cita pri cross-matchi:
#   TVRDO povinne : obj_id, ra_deg, dec_deg
#   pouzivane     : name, host_name, cat_source, disposition
#   volitelne     : period, mag, mag_band
SCHEMA = """
CREATE TABLE IF NOT EXISTS exoplanet_data (
    obj_id      TEXT PRIMARY KEY,   -- pl_name (CONFIRMED) / 'TOI-1234.01' (TOI); PK => INSERT OR IGNORE idempotencia
    name        TEXT,               -- oznacenie objektu
    host_name   TEXT,               -- hostitelska hviezda (kontext cross-matchu)
    ra_deg      REAL,
    dec_deg     REAL,
    cat_source  TEXT,               -- 'CONFIRMED' | 'TOI'
    disposition TEXT,               -- 'CONFIRMED' | TFOPWG: PC/CP/KP/APC/FP/FA
    period      REAL,               -- orbitalna perioda [dni]; NaN ak NEA neuvadza
    mag         REAL,               -- TESS mag (preferovane), V fallback
    mag_band    TEXT                -- 'TESS' | 'V' | NULL
);
"""

_INSERT = (
    "INSERT OR IGNORE INTO exoplanet_data "
    "(obj_id, name, host_name, ra_deg, dec_deg, cat_source, disposition, period, mag, mag_band) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_COLS = [
    "obj_id", "name", "host_name", "ra_deg", "dec_deg",
    "cat_source", "disposition", "period", "mag", "mag_band",
]

# NEA stlpce, ktore selectujeme z kazdej tabulky. Mag riesime "coalesce" nizsie.
#   required = bez nich nevieme cross-matchovat -> ak chyba, ABORT (fail-loud).
#   optional = ak chyba v zivej schema, len upozornime a doplnime NULL.
_SRC = {
    "pscomppars": {
        "required": ["pl_name", "ra", "dec"],
        "optional": ["hostname", "pl_orbper", "sy_tmag", "sy_vmag"],
    },
    "toi": {
        "required": ["toi", "ra", "dec"],
        "optional": ["tid", "pl_orbper", "st_tmag", "tfopwg_disp"],
    },
}


def _tap_csv(adql: str) -> str:
    url = TAP_BASE + "?" + urllib.parse.urlencode({"query": adql, "format": "csv"})
    req = urllib.request.Request(url, headers={"User-Agent": "VYVAR-exoplanet-builder/1.0"})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _schema_columns(table: str) -> set[str]:
    """Vrati skutocnu mnozinu nazvov stlpcov tabulky zo zivej TAP_SCHEMA."""
    adql = f"select column_name from TAP_SCHEMA.columns where table_name = '{table}'"
    txt = _tap_csv(adql)
    df = pd.read_csv(io.StringIO(txt))
    col = "column_name" if "column_name" in df.columns else df.columns[0]
    return {str(c).strip() for c in df[col].tolist()}


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(SCHEMA)
    # Spatial indexy pre cross-match (VYVAR by ich vytvoril pri prvom query; tu je to lacne a cistejsie).
    conn.execute("CREATE INDEX IF NOT EXISTS idx_exo_ra ON exoplanet_data (ra_deg);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_exo_dec ON exoplanet_data (dec_deg);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_exo_src ON exoplanet_data (cat_source);")
    conn.commit()


def _count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM exoplanet_data;").fetchone()[0])


def _verify_and_select(table: str) -> tuple[list[str], list[str]]:
    """Overi pozadovane stlpce proti zivej schema; vrati (select_cols, missing_optional).

    Fail-loud, ak chyba POVINNY stlpec (vypise dostupne stlpce, aby sa dal opravit mapping).
    """
    avail = _schema_columns(table)
    req = _SRC[table]["required"]
    opt = _SRC[table]["optional"]
    missing_req = [c for c in req if c not in avail]
    if missing_req:
        sample = ", ".join(sorted(avail)[:40])
        raise SystemExit(
            f"[ABORT] tabulka '{table}': chybaju POVINNE stlpce {missing_req}. "
            f"Schema sa zrejme zmenila. Dostupne stlpce (vzorka): {sample} ..."
        )
    missing_opt = [c for c in opt if c not in avail]
    if missing_opt:
        print(f"   WARNING tabulka '{table}': chybaju volitelne stlpce {missing_opt} -> doplnam NULL.")
    select_cols = [c for c in (req + opt) if c in avail]
    return select_cols, missing_opt


def _fetch_table(table: str, select_cols: list[str], mag_limit: float | None) -> pd.DataFrame:
    cols = ",".join(select_cols)
    where = ""
    if mag_limit is not None:
        magcol = "sy_tmag" if table == "pscomppars" else "st_tmag"
        if magcol in select_cols:
            where = f" where {magcol} < {float(mag_limit)}"
    adql = f"select {cols} from {table}{where}"
    txt = _tap_csv(adql)
    return pd.read_csv(io.StringIO(txt))


def _normalize_confirmed(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["obj_id"] = df["pl_name"].astype("string")
    out["name"] = df["pl_name"].astype("string")
    out["host_name"] = df["hostname"].astype("string") if "hostname" in df else pd.NA
    out["ra_deg"] = pd.to_numeric(df["ra"], errors="coerce")
    out["dec_deg"] = pd.to_numeric(df["dec"], errors="coerce")
    out["cat_source"] = "CONFIRMED"
    out["disposition"] = "CONFIRMED"
    out["period"] = pd.to_numeric(df["pl_orbper"], errors="coerce") if "pl_orbper" in df else pd.NA
    # mag: preferuj TESS (sy_tmag), inak V (sy_vmag)
    tmag = pd.to_numeric(df["sy_tmag"], errors="coerce") if "sy_tmag" in df else pd.Series([pd.NA] * len(df))
    vmag = pd.to_numeric(df["sy_vmag"], errors="coerce") if "sy_vmag" in df else pd.Series([pd.NA] * len(df))
    out["mag"] = tmag.where(tmag.notna(), vmag)
    out["mag_band"] = ["TESS" if pd.notna(t) else ("V" if pd.notna(v) else None) for t, v in zip(tmag, vmag)]
    return out


def _fmt_toi(x: object) -> str | None:
    try:
        return f"TOI-{float(x):.2f}"
    except (TypeError, ValueError):
        return None


def _normalize_toi(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["obj_id"] = [_fmt_toi(x) for x in df["toi"]]
    out["name"] = out["obj_id"]
    if "tid" in df:
        out["host_name"] = ["TIC " + str(int(t)) if pd.notna(t) else None for t in pd.to_numeric(df["tid"], errors="coerce")]
    else:
        out["host_name"] = pd.NA
    out["ra_deg"] = pd.to_numeric(df["ra"], errors="coerce")
    out["dec_deg"] = pd.to_numeric(df["dec"], errors="coerce")
    out["cat_source"] = "TOI"
    out["disposition"] = df["tfopwg_disp"].astype("string") if "tfopwg_disp" in df else pd.NA
    out["period"] = pd.to_numeric(df["pl_orbper"], errors="coerce") if "pl_orbper" in df else pd.NA
    tmag = pd.to_numeric(df["st_tmag"], errors="coerce") if "st_tmag" in df else pd.Series([pd.NA] * len(df))
    out["mag"] = tmag
    out["mag_band"] = ["TESS" if pd.notna(t) else None for t in tmag]
    return out


def _insert(conn: sqlite3.Connection, out: pd.DataFrame, label: str) -> None:
    # Dopln chybajuce stlpce (poistka) + zahod riadky bez ID/RA/Dec (nepouzitelne pre cross-match).
    for c in _COLS:
        if c not in out.columns:
            out[c] = None
    out = out.dropna(subset=["obj_id", "ra_deg", "dec_deg"]).copy()

    rows = [
        (
            str(r.obj_id),
            None if pd.isna(r.name) else str(r.name),
            None if pd.isna(r.host_name) else str(r.host_name),
            float(r.ra_deg),
            float(r.dec_deg),
            str(r.cat_source),
            None if pd.isna(r.disposition) else str(r.disposition),
            None if pd.isna(r.period) else float(r.period),
            None if pd.isna(r.mag) else float(r.mag),
            None if pd.isna(r.mag_band) else str(r.mag_band),
        )
        for r in out[_COLS].itertuples(index=False)
    ]
    before = _count(conn)
    conn.executemany(_INSERT, rows)
    conn.commit()
    added = _count(conn) - before
    print(f"   OK {label}: {len(rows)} riadkov ({added} novych, {len(rows) - added} duplikatov ignorovanych).")


def build_exoplanet(db_name: str, *, mag_limit: float | None, confirmed_only: bool) -> None:
    """Stiahne NEA pscomppars (+ toi) cez TAP do `exoplanet_data`.

    Re-runnovatelne a inkrementalne: vdaka PK na `obj_id` a `INSERT OR IGNORE`
    sa duplikaty preskocia. Zdroj kandidatov (FP/FA) sa NEZAHADZUJE pri buildu,
    len sa olabeluje v `disposition` -> VYVAR filtruje pri dotaze.
    """
    conn = sqlite3.connect(db_name)
    try:
        _ensure_schema(conn)
        start_count = _count(conn)

        plan = ["pscomppars"] if confirmed_only else ["pscomppars", "toi"]
        for table in plan:
            print(f"Spracovavam tabulku '{table}' ...")
            select_cols, _ = _verify_and_select(table)
            for attempt in range(5):
                try:
                    df = _fetch_table(table, select_cols, mag_limit)
                    if df is None or len(df) == 0:
                        print(f"   INFO tabulka '{table}' vratila 0 riadkov.")
                        break
                    out = _normalize_confirmed(df) if table == "pscomppars" else _normalize_toi(df)
                    _insert(conn, out, table)
                    break
                except Exception as e:  # noqa: BLE001
                    wait_time = (attempt + 1) * 15
                    print(f"   WARNING pokus {attempt + 1} pre '{table}' zlyhal: {e}. Cakam {wait_time}s ...")
                    time.sleep(wait_time)

        print("\nFinalizacia (VACUUM) ...")
        conn.execute("VACUUM")
        end_count = _count(conn)
        # Mala statistika podla zdroja/dispozicie.
        by_src = conn.execute(
            "SELECT cat_source, COUNT(*) FROM exoplanet_data GROUP BY cat_source ORDER BY cat_source;"
        ).fetchall()
        print(f"Hotovo! exoplanet_data: {end_count} riadkov (+{end_count - start_count} v tomto behu).")
        for src, n in by_src:
            print(f"   {src}: {n}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VYVAR exoplanet catalog builder -- NASA Exoplanet Archive (TAP). "
        "Potvrdene (pscomppars) + TOI kandidati (toi) do jednej lokalnej SQLite. "
        "Re-runnovatelne a inkrementalne. Cisto na cross-match (RA/Dec/perioda/mag/disposition)."
    )
    parser.add_argument("--db", default=DB_NAME, help=f"Cesta k SQLite DB (default: {DB_NAME})")
    parser.add_argument(
        "--mag-limit", type=float, default=MAG_LIMIT,
        help="Rez podla TESS/V magnitudy hostitela (default: bez rezu = cely katalog).",
    )
    parser.add_argument(
        "--confirmed-only", action="store_true",
        help="Len potvrdene (pscomppars), bez TOI kandidatov.",
    )
    args = parser.parse_args()

    db_name = str(args.db).strip() or DB_NAME
    plan = "CONFIRMED only" if args.confirmed_only else "CONFIRMED + TOI"
    mag = "bez rezu" if args.mag_limit is None else f"mag<{args.mag_limit}"
    print("=== NASA Exoplanet Archive -> lokalna DB ===")
    print(f"DB={db_name}  zdroj={plan}  {mag}")
    build_exoplanet(db_name, mag_limit=args.mag_limit, confirmed_only=bool(args.confirmed_only))


if __name__ == "__main__":
    main()
