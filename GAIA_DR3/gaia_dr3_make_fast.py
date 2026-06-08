"""
gaia_dr3_make_fast.py
─────────────────────────────────────────────────────────────────────────────
Stiahne Gaia DR3 (gaiadr3.gaia_source) do lokálnej SQLite pre VYVAR.

Optimalizácie:
  • Polootvorené RA/Dec intervaly → žiadne duplicity na hraniciach pásov.
  • INTEGER PRIMARY KEY (source_id) + INSERT OR IGNORE → bezpečný resume.
  • Zápis cez executemany po dávkach (rýchlejšie ako pandas to_sql).
  • NULLIF pri delení fluksi (full tab); predvolene ``gaia_source_lite`` + NULL pre voliteľné stĺpce.
  • strip_progress; WAL; batch commit.

VYVAR: tabuľka ``gaia_dr3`` zodpovedá ``database.query_local_gaia`` (g_mag, bp_rp,
var_flag, non_single_star, g_flux_error_rel, …).

Spustenie:
  python GAIA_DR3/gaia_dr3_make_fast.py

Prostredie:
  SKIP_VACUUM=1         — preskočí záverečný VACUUM.
  GAIA_SOURCE_FULL=1   — plný ``gaiadr3.gaia_source`` (pomalšie, častejšie chyby 500).
  GAIA_RA_STEP / GAIA_DEC_STEP — veľkosť pásov v stupňoch (predvolene 7.5 × 5).
  GAIA_MAX_STRIPS=800 — ak by bolo viac pásov, kroky sa automaticky zväčšia (TAP overhead).
  GAIA_NO_STRIP_CLAMP=1 — vypne auto-zväčšenie (iba pre experiment).

  Pri zmene siete pásov vymaž ``strip_progress`` alebo celú DB.

Predvolene: ``gaiadr3.gaia_source_lite`` — ESA ju odporúča na ťažké ADQL (menej HTTP 500).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import os
import sys
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

# Spustenie z GAIA_DR3/ aj z koreňa projektu
_VYVAR_ROOT = Path(__file__).resolve().parent.parent
if str(_VYVAR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VYVAR_ROOT))

import pandas as pd
from astroquery.gaia import Gaia

from gaia_catalog_id import normalize_gaia_source_id


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if math.isfinite(v) and v > 0 else default
    except ValueError:
        return default


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def _strip_count(dec_min: float, dec_max: float, ra_step: float, dec_step: float) -> int:
    return len(generate_strips_with_flags(dec_min, dec_max, dec_step, ra_step))


def _widen_steps_to_strip_cap(
    dec_min: float,
    dec_max: float,
    ra_step: float,
    dec_step: float,
    max_strips: int,
) -> tuple[float, float, int]:
    """Zväčší RA/Dec krok, kým počet pásov <= max_strips (každý pás = samostatný TAP job)."""
    ra_s = float(ra_step)
    dec_s = float(dec_step)
    n = _strip_count(dec_min, dec_max, ra_s, dec_s)
    if n <= max_strips:
        return ra_s, dec_s, n
    for _ in range(80):
        f = math.sqrt(n / float(max_strips)) * 1.02
        ra_s = min(60.0, max(ra_s * f, ra_s + 0.5))
        dec_s = min(60.0, max(dec_s * f, dec_s + 0.5))
        n = _strip_count(dec_min, dec_max, ra_s, dec_s)
        if n <= max_strips:
            return ra_s, dec_s, n
    return ra_s, dec_s, n


# ── KONFIGURÁCIA ─────────────────────────────────────────────────────────────
DB_NAME = Path(__file__).resolve().parent / "vyvar_gaia_dr3_v2.db"

MAG_LIMIT = 16.0
DEC_MIN = -20.0
DEC_MAX = 90.0
# Predvolene: ~400–900 pásov; príliš malý krok = tisíce TAP jobov (hodiny overheadu).
DEC_STEP = _env_float("GAIA_DEC_STEP", 5.0)
RA_STEP = _env_float("GAIA_RA_STEP", 7.5)
MAX_STRIPS_CAP = _env_int("GAIA_MAX_STRIPS", 800)

MAX_RETRIES = 8
RETRY_BASE_S = 20
RETRY_MAX_WAIT_S = 240
COMMIT_EVERY = 3
INSERT_BATCH = 8000

# True = gaiadr3.gaia_source (teff/logg/…); False = gaia_source_lite (odporúčané).
USE_GAIA_SOURCE_FULL = _env_bool("GAIA_SOURCE_FULL")

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source_lite" if not USE_GAIA_SOURCE_FULL else "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1

_ROW_COLUMNS = (
    "source_id",
    "ra",
    "dec",
    "g_mag",
    "bp_mag",
    "rp_mag",
    "bp_rp",
    "g_flux_error_rel",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "distance_gspphot",
    "var_flag",
    "non_single_star",
)


def generate_strips_with_flags(
    dec_min: float,
    dec_max: float,
    dec_step: float,
    ra_step: float,
) -> list[tuple[tuple[float, float, float, float], bool, bool]]:
    """Vráti zoznam ((r0,r1,d0,d1), last_ra, last_dec) pre ADQL polootvorené intervaly."""
    items: list[tuple[tuple[float, float, float, float], bool, bool]] = []
    d = dec_min
    while d < dec_max - 1e-12:
        d0 = d
        d1 = min(d + dec_step, dec_max)
        last_dec = abs(d1 - dec_max) < 1e-9

        r = 0.0
        while r < 360.0 - 1e-12:
            r0 = r
            r1 = min(r + ra_step, 360.0)
            last_ra = r1 >= 360.0 - 1e-9
            items.append(((r0, r1, d0, d1), last_ra, last_dec))
            r += ra_step
            if last_ra:
                break
        d += dec_step
    return items


def strip_key(strip: tuple[float, float, float, float]) -> str:
    r0, r1, d0, d1 = strip
    return f"{r0:.4f}_{r1:.4f}_{d0:.4f}_{d1:.4f}"


def _ra_clause(r0: float, r1: float, last_ra: bool) -> str:
    if last_ra:
        return f"ra >= {r0} AND ra <= 360.0"
    return f"ra >= {r0} AND ra < {r1}"


def _dec_clause(d0: float, d1: float, last_dec: bool) -> str:
    if last_dec:
        return f"dec >= {d0} AND dec <= {d1}"
    return f"dec >= {d0} AND dec < {d1}"


def build_adql(
    r0: float,
    r1: float,
    d0: float,
    d1: float,
    *,
    last_ra: bool,
    last_dec: bool,
    mag_limit: float,
    use_full_source: bool,
) -> str:
    """``use_full_source=False`` → ``gaia_source_lite`` (rýchlejšie, menej 500 od TAP)."""
    ra_c = _ra_clause(r0, r1, last_ra)
    de_c = _dec_clause(d0, d1, last_dec)
    mag_f = float(mag_limit)
    if use_full_source:
        return f"""
        SELECT
            source_id,
            ra,
            dec,
            phot_g_mean_mag AS g_mag,
            phot_bp_mean_mag AS bp_mag,
            phot_rp_mean_mag AS rp_mag,
            bp_rp,
            (phot_g_mean_flux_error / NULLIF(phot_g_mean_flux, 0.0)) AS g_flux_error_rel,
            parallax,
            parallax_error,
            parallax_over_error,
            teff_gspphot,
            logg_gspphot,
            mh_gspphot,
            distance_gspphot,
            phot_variable_flag AS var_flag,
            non_single_star
        FROM gaiadr3.gaia_source
        WHERE ({ra_c})
          AND ({de_c})
          AND phot_g_mean_mag IS NOT NULL
          AND phot_g_mean_mag <= {mag_f}
        """
    # Lite: menšia tabuľka (ESA odporúča pre náročné dotazy). Chýbajúce stĺpce doplní
    # ``_normalize_tap_dataframe`` — VYVAR ich nepotrebuje povinne.
    return f"""
        SELECT
            source_id,
            ra,
            dec,
            phot_g_mean_mag AS g_mag,
            phot_bp_mean_mag AS bp_mag,
            phot_rp_mean_mag AS rp_mag,
            bp_rp,
            parallax,
            parallax_error,
            parallax_over_error,
            phot_variable_flag AS var_flag,
            non_single_star
        FROM gaiadr3.gaia_source_lite
        WHERE ({ra_c})
          AND ({de_c})
          AND phot_g_mean_mag IS NOT NULL
          AND phot_g_mean_mag <= {mag_f}
        """


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gaia_dr3 (
            source_id           INTEGER PRIMARY KEY,
            ra                  REAL,
            dec                 REAL,
            g_mag               REAL,
            bp_mag              REAL,
            rp_mag              REAL,
            bp_rp               REAL,
            g_flux_error_rel    REAL,
            parallax            REAL,
            parallax_error      REAL,
            parallax_over_error REAL,
            teff_gspphot        REAL,
            logg_gspphot        REAL,
            mh_gspphot          REAL,
            distance_gspphot    REAL,
            var_flag            TEXT,
            non_single_star     INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strip_progress (
            strip_key   TEXT PRIMARY KEY,
            ra_min      REAL,
            ra_max      REAL,
            dec_min     REAL,
            dec_max     REAL,
            n_stars     INTEGER,
            finished_at TEXT
        )
        """
    )
    conn.commit()


def get_done_strips(conn: sqlite3.Connection) -> set[str]:
    try:
        cur = conn.execute("SELECT strip_key FROM strip_progress")
        return {str(r[0]) for r in cur.fetchall()}
    except sqlite3.Error:
        return set()


def mark_strip_done(
    conn: sqlite3.Connection,
    strip: tuple[float, float, float, float],
    n_stars: int,
) -> None:
    r0, r1, d0, d1 = strip
    conn.execute(
        """
        INSERT OR REPLACE INTO strip_progress
            (strip_key, ra_min, ra_max, dec_min, dec_max, n_stars, finished_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            strip_key(strip),
            r0,
            r1,
            d0,
            d1,
            n_stars,
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    )


def _normalize_tap_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Stĺpce z ADQL → malé písmená, mapovanie na očakávané názvy."""
    if df.empty:
        return pd.DataFrame(columns=list(_ROW_COLUMNS))
    rename: dict[str, str] = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        rename[str(c)] = cl
    out = df.rename(columns=rename)
    missing = [c for c in _ROW_COLUMNS if c not in out.columns]
    for c in missing:
        if c == "var_flag":
            out[c] = pd.Series([None] * len(out), dtype=object)
        else:
            out[c] = float("nan")
    out = out[list(_ROW_COLUMNS)].copy()
    if "var_flag" in out.columns:
        def _vf(x: object) -> str | None:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return None
            try:
                if pd.isna(x):
                    return None
            except (TypeError, ValueError):
                pass
            s = str(x).strip()
            if not s or s.lower() == "nan":
                return None
            return s

        out["var_flag"] = out["var_flag"].map(_vf)
    return out


def _source_id_sql(v: object) -> int | None:
    s = normalize_gaia_source_id(v)
    if not s or not s.isdigit():
        return None
    try:
        return int(s)
    except (TypeError, ValueError, OverflowError):
        return None


def _df_to_rows(df: pd.DataFrame) -> list[tuple]:
    df = _normalize_tap_dataframe(df)
    out: list[tuple] = []
    for _, row in df.iterrows():
        tup: list = []
        for c in _ROW_COLUMNS:
            v = row[c]
            if c == "source_id":
                sid = _source_id_sql(v)
                tup.append(sid)
                continue
            if v is None:
                tup.append(None)
                continue
            try:
                if pd.isna(v):
                    tup.append(None)
                    continue
            except (TypeError, ValueError):
                pass
            if c == "var_flag":
                tup.append(None if v is None else str(v))
            elif c == "non_single_star":
                try:
                    tup.append(int(v))
                except (TypeError, ValueError):
                    tup.append(None)
            else:
                try:
                    fv = float(v)
                    tup.append(fv if math.isfinite(fv) else None)
                except (TypeError, ValueError):
                    tup.append(None)
        out.append(tuple(tup))
    return out


def insert_dataframe(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    rows = _df_to_rows(df)
    rows = [t for t in rows if t[0] is not None]
    if not rows:
        return 0
    cols = ", ".join(_ROW_COLUMNS)
    ph = ", ".join("?" * len(_ROW_COLUMNS))
    sql = f"INSERT OR IGNORE INTO gaia_dr3 ({cols}) VALUES ({ph})"
    n = 0
    for i in range(0, len(rows), INSERT_BATCH):
        chunk = rows[i : i + INSERT_BATCH]
        conn.executemany(sql, chunk)
        n += len(chunk)
    return n


def download_strip(
    strip: tuple[float, float, float, float],
    last_ra: bool,
    last_dec: bool,
    *,
    use_full_source: bool,
    max_retries: int = MAX_RETRIES,
) -> pd.DataFrame | None:
    r0, r1, d0, d1 = strip
    query = build_adql(
        r0,
        r1,
        d0,
        d1,
        last_ra=last_ra,
        last_dec=last_dec,
        mag_limit=MAG_LIMIT,
        use_full_source=use_full_source,
    )

    for attempt in range(max_retries):
        try:
            job = Gaia.launch_job_async(query)
            results = job.get_results()
            if len(results) == 0:
                return pd.DataFrame(columns=list(_ROW_COLUMNS))
            df = results.to_pandas()
            return _normalize_tap_dataframe(df)
        except Exception as e:  # noqa: BLE001
            wait = min(RETRY_MAX_WAIT_S, int(RETRY_BASE_S * (2**attempt)))
            print(f"\n    WARN pokus {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print(f"    Cakam {wait}s ...", flush=True)
                time.sleep(wait)
    return None


def create_indexes(conn: sqlite3.Connection) -> None:
    # PRIMARY KEY na source_id už vytvára index; (ra,dec) je kritické pre VYVAR.
    stmts = [
        "CREATE INDEX IF NOT EXISTS idx_ra_dec ON gaia_dr3 (ra, dec)",
        "CREATE INDEX IF NOT EXISTS idx_ra ON gaia_dr3 (ra)",
        "CREATE INDEX IF NOT EXISTS idx_dec ON gaia_dr3 (dec)",
        "CREATE INDEX IF NOT EXISTS idx_g_mag ON gaia_dr3 (g_mag)",
        "CREATE INDEX IF NOT EXISTS idx_parallax_snr ON gaia_dr3 (parallax_over_error)",
        "CREATE INDEX IF NOT EXISTS idx_teff ON gaia_dr3 (teff_gspphot)",
    ]
    for sql in stmts:
        conn.execute(sql)
    conn.commit()


def create_local_gaia() -> None:
    db_path = Path(DB_NAME)
    ra_eff = float(RA_STEP)
    dec_eff = float(DEC_STEP)
    if not _env_bool("GAIA_NO_STRIP_CLAMP"):
        ra0, dec0 = ra_eff, dec_eff
        ra_eff, dec_eff, n_est = _widen_steps_to_strip_cap(
            DEC_MIN, DEC_MAX, ra_eff, dec_eff, int(MAX_STRIPS_CAP)
        )
        if (ra_eff, dec_eff) != (ra0, dec0):
            print(
                f"  AUTO-CLAMP: prilis vela stripov pri krokoch RA={ra0} Dec={dec0} — "
                f"zvacsené na RA={ra_eff:.2f} Dec={dec_eff:.2f} (max ~{MAX_STRIPS_CAP} TAP jobov)."
            )
            print("    (GAIA_NO_STRIP_CLAMP=1 ak naozaj chces ultra-jemnu siet.)")
            print()

    items = generate_strips_with_flags(DEC_MIN, DEC_MAX, dec_eff, ra_eff)
    n_total = len(items)

    print("=" * 62)
    print("  VYVAR — Gaia DR3 lokálna DB (fast strip)")
    print("=" * 62)
    print(f"  Vystup:      {db_path}")
    print(f"  Mag limit:   G <= {MAG_LIMIT}")
    print(f"  Dec:         {DEC_MIN} .. {DEC_MAX}  (efekt. krok {dec_eff:.2f})")
    print(f"  RA:          0 .. 360  (efekt. krok {ra_eff:.2f})")
    print(f"  Tabulka:     {'gaiadr3.gaia_source (FULL)' if USE_GAIA_SOURCE_FULL else 'gaiadr3.gaia_source_lite (default)'}")
    print(f"  Stripov:     {n_total}  (cieľ max ~{MAX_STRIPS_CAP} kvoli TAP overheadu)")
    print()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    init_db(conn)

    done = get_done_strips(conn)
    todo = [(s, lr, ld) for (s, lr, ld) in items if strip_key(s) not in done]
    n_done = n_total - len(todo)
    print(f"  Hotovych stripov: {n_done}/{n_total}, zostava: {len(todo)}")
    print()

    if todo:
        total_inserted = 0
        failed: list[tuple[float, float, float, float]] = []
        t0 = time.time()
        recent_dt: list[float] = []
        for idx, (strip, last_ra, last_dec) in enumerate(todo, start=1):
            r0, r1, d0, d1 = strip
            t_strip = time.time()
            remaining = len(todo) - idx + 1
            if len(recent_dt) >= 3:
                avg = sum(recent_dt[-20:]) / min(len(recent_dt), 20)
                eta = avg * remaining
                if eta >= 3600:
                    eta_str = f"ETA ~{int(eta // 3600)}h{int((eta % 3600) // 60)}m"
                else:
                    eta_str = f"ETA ~{int(eta // 60)}m{int(eta % 60)}s"
            else:
                eta_str = "ETA …"

            print(
                f"[{n_done + idx:4d}/{n_total}] "
                f"RA {r0:6.2f}-{r1:6.2f} Dec {d0:+6.2f}-{d1:+6.2f} | {eta_str} | inserts ~{total_inserted:,} ",
                end="",
                flush=True,
            )

            df = download_strip(strip, last_ra, last_dec, use_full_source=USE_GAIA_SOURCE_FULL)
            if df is None:
                print(" ERR")
                failed.append(strip)
                continue

            try:
                n_new = insert_dataframe(conn, df)
            except Exception as exc:  # noqa: BLE001
                print(f" INSERT FAIL: {exc}")
                failed.append(strip)
                continue

            total_inserted += n_new
            mark_strip_done(conn, strip, len(df))
            recent_dt.append(time.time() - t_strip)
            print(f" ok tap_rows={len(df):,} sql_tuples={n_new:,}")

            if idx % COMMIT_EVERY == 0:
                conn.commit()

        conn.commit()
        print()
        print(f"  Celkom INSERT pokusov (batched): {total_inserted:,}")
        if failed:
            print(f"  Zlyhalo stripov: {len(failed)} — restart preskoci hotove (strip_progress).")

    print()
    print("  Vytvaram indexy …")
    create_indexes(conn)

    if os.environ.get("SKIP_VACUUM", "").strip().lower() in ("1", "true", "yes"):
        print("  SKIP_VACUUM — VACUUM preskoceny.")
    else:
        print("  VACUUM (moze trvat dlho) …")
        conn.execute("VACUUM")

    conn.close()
    sz_gb = db_path.stat().st_size / (1024**3)
    print()
    print("=" * 62)
    print("  Hotovo.")
    print(f"  Subor: {db_path} ({sz_gb:.2f} GB)")
    print("  V config.json nastav gaia_db_path na tuto cestu po kontrole.")
    print("=" * 62)


if __name__ == "__main__":
    create_local_gaia()
