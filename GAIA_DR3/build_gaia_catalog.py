"""
build_gaia_catalog.py - kanonicky script #1 (Gaia DR3 -> lokalna SQLite pre VYVAR)
-----------------------------------------------------------------------------
Stiahne Gaia DR3 do lokalnej SQLite (tabulka ``gaia_dr3``).

Optimalizacie:
  * Polootvorene RA/Dec intervaly -> ziadne duplicity na hraniciach pasov.
  * INTEGER PRIMARY KEY (source_id) + INSERT OR IGNORE -> bezpecny resume.
  * Zapis cez executemany po davkach (rychlejsie ako pandas to_sql).
  * NULLIF pri deleni fluksi (full tab); predvolene ``gaia_source_lite``.
  * strip_progress; WAL; batch commit.

Spustenie (priklady):
  python GAIA_DR3/build_gaia_catalog.py --help
  python GAIA_DR3/build_gaia_catalog.py --dec-min 89 --dec-max 90 --mag-limit 10 --skip-vacuum

Env fallback (ak nie je CLI): GAIA_MAG_LIMIT, GAIA_DEC_MIN, GAIA_DEC_MAX,
GAIA_OUT, GAIA_RA_STEP, GAIA_DEC_STEP, GAIA_MAX_STRIPS, GAIA_SOURCE_FULL,
SKIP_VACUUM, GAIA_NO_STRIP_CLAMP. Volitelny login: GAIA_USER + GAIA_PASS
(inak anonymne TAP).

Pri zmene siete pasov vymaz ``strip_progress`` alebo celu DB.
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_GAIA_DR3_DIR = Path(__file__).resolve().parent


def _find_vyvar_root(start: Path) -> tuple[Path, Path]:
    """Walk upward from *start* to locate ``gaia_catalog_id`` (py or compiled).

    Post REPO-REORG (2026-07) the VYVAR modules live under ``src_py/``. Return
    ``(repo_root, module_dir)`` where *module_dir* is the directory that actually
    holds ``gaia_catalog_id`` (``<root>/src_py`` on the current layout, or the
    root itself on the legacy flat layout).
    """
    here = start.resolve()

    def _has_gaia_catalog_id(candidate: Path) -> Path | None:
        src = candidate / "src_py"
        if (src / "gaia_catalog_id.py").is_file():
            return src
        if any(src.glob("gaia_catalog_id*.pyd")) or any(src.glob("gaia_catalog_id*.so")):
            return src
        if (candidate / "gaia_catalog_id.py").is_file():
            return candidate
        return None

    for candidate in (here, *here.parents):
        mod_dir = _has_gaia_catalog_id(candidate)
        if mod_dir is not None:
            return candidate, mod_dir
    raise SystemExit(
        "build_gaia_catalog.py needs gaia_catalog_id from the VYVAR install "
        "(src_py/gaia_catalog_id). Run from the install root or scripts/catalogs/ "
        "and use --out to write the DB under your data directory."
    )


_VYVAR_ROOT, _VYVAR_SRC = _find_vyvar_root(_GAIA_DR3_DIR)
for _p in (_VYVAR_SRC, _VYVAR_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
from astroquery.gaia import Gaia

from gaia_catalog_id import normalize_gaia_source_id

def _catalog_default(script_dir: Path, *rel_parts: str, legacy: Path) -> Path:
    import importlib.util

    for base in (
        script_dir,
        script_dir.parent / "scripts" / "catalogs",
        _VYVAR_ROOT / "scripts" / "catalogs",
    ):
        hp = base / "vyvar_catalog_paths.py"
        if not hp.is_file():
            continue
        spec = importlib.util.spec_from_file_location("vyvar_catalog_paths", hp)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.default_catalog_file(script_dir, *rel_parts)
    return legacy


DEFAULT_OUT = _catalog_default(
    _GAIA_DR3_DIR, "GAIA_DR3", "vyvar_gaia_dr3.db", legacy=_GAIA_DR3_DIR / "vyvar_gaia_dr3.db"
)

MAX_RETRIES = 8
RETRY_BASE_S = 20
RETRY_MAX_WAIT_S = 240
COMMIT_EVERY = 3
INSERT_BATCH = 8000

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


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = float(raw)
        return v if math.isfinite(v) else default
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


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return Path(raw).expanduser()


@dataclass(frozen=True)
class BuildConfig:
    db_path: Path
    mag_limit: float
    dec_min: float
    dec_max: float
    ra_step: float
    dec_step: float
    max_strips: int
    use_full_source: bool
    skip_vacuum: bool
    no_strip_clamp: bool


def parse_args(argv: list[str] | None = None) -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Download Gaia DR3 into local SQLite for VYVAR (strip TAP builder).",
    )
    parser.add_argument(
        "--mag-limit",
        type=float,
        default=_env_float("GAIA_MAG_LIMIT", 16.5),
        help="Upper G mag limit (default 16.5; env GAIA_MAG_LIMIT).",
    )
    parser.add_argument(
        "--dec-min",
        type=float,
        default=_env_float("GAIA_DEC_MIN", -90.0),
        help="Dec lower bound in degrees (default -90; env GAIA_DEC_MIN).",
    )
    parser.add_argument(
        "--dec-max",
        type=float,
        default=_env_float("GAIA_DEC_MAX", 90.0),
        help="Dec upper bound in degrees (default 90; env GAIA_DEC_MAX).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_env_path("GAIA_OUT", DEFAULT_OUT),
        help=f"Output SQLite path (default {DEFAULT_OUT.name}; env GAIA_OUT).",
    )
    parser.add_argument(
        "--ra-step",
        type=float,
        default=_env_float("GAIA_RA_STEP", 7.5),
        help="RA strip size in degrees (default 7.5; env GAIA_RA_STEP).",
    )
    parser.add_argument(
        "--dec-step",
        type=float,
        default=_env_float("GAIA_DEC_STEP", 5.0),
        help="Dec strip size in degrees (default 5.0; env GAIA_DEC_STEP).",
    )
    parser.add_argument(
        "--max-strips",
        type=int,
        default=_env_int("GAIA_MAX_STRIPS", 800),
        help="Max TAP jobs; grid auto-widens if exceeded (default 800; env GAIA_MAX_STRIPS).",
    )
    parser.add_argument(
        "--full-source",
        action="store_true",
        default=_env_bool("GAIA_SOURCE_FULL"),
        help="Use gaiadr3.gaia_source (teff/logg/...); default lite (env GAIA_SOURCE_FULL).",
    )
    parser.add_argument(
        "--skip-vacuum",
        action="store_true",
        default=_env_bool("SKIP_VACUUM"),
        help="Skip final VACUUM (env SKIP_VACUUM).",
    )
    args = parser.parse_args(argv)

    if not math.isfinite(args.mag_limit) or args.mag_limit <= 0:
        parser.error("--mag-limit must be a positive finite number")
    if not math.isfinite(args.dec_min) or not math.isfinite(args.dec_max):
        parser.error("--dec-min and --dec-max must be finite numbers")
    if args.dec_min >= args.dec_max:
        parser.error("--dec-min must be less than --dec-max")
    for name, val in (("ra-step", args.ra_step), ("dec-step", args.dec_step)):
        if not math.isfinite(val) or val <= 0:
            parser.error(f"--{name} must be a positive finite number")
    if args.max_strips <= 0:
        parser.error("--max-strips must be positive")

    return BuildConfig(
        db_path=args.out.resolve(),
        mag_limit=float(args.mag_limit),
        dec_min=float(args.dec_min),
        dec_max=float(args.dec_max),
        ra_step=float(args.ra_step),
        dec_step=float(args.dec_step),
        max_strips=int(args.max_strips),
        use_full_source=bool(args.full_source),
        skip_vacuum=bool(args.skip_vacuum),
        no_strip_clamp=_env_bool("GAIA_NO_STRIP_CLAMP"),
    )


def _configure_gaia(cfg: BuildConfig) -> None:
    Gaia.MAIN_GAIA_TABLE = (
        "gaiadr3.gaia_source" if cfg.use_full_source else "gaiadr3.gaia_source_lite"
    )
    Gaia.ROW_LIMIT = -1
    user = os.environ.get("GAIA_USER", "").strip()
    passwd = os.environ.get("GAIA_PASS", "").strip()
    if user and passwd:
        Gaia.login(user=user, password=passwd)
        print("  TAP login:     authenticated (GAIA_USER set)")
    else:
        print("  TAP login:     anonymous (default)")


def _strip_count(dec_min: float, dec_max: float, ra_step: float, dec_step: float) -> int:
    return len(generate_strips_with_flags(dec_min, dec_max, dec_step, ra_step))


def _widen_steps_to_strip_cap(
    dec_min: float,
    dec_max: float,
    ra_step: float,
    dec_step: float,
    max_strips: int,
) -> tuple[float, float, int]:
    """Zvacsi RA/Dec krok, kym pocet pasov <= max_strips (kazdy pas = samostatny TAP job)."""
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


def generate_strips_with_flags(
    dec_min: float,
    dec_max: float,
    dec_step: float,
    ra_step: float,
) -> list[tuple[tuple[float, float, float, float], bool, bool]]:
    """Vrati zoznam ((r0,r1,d0,d1), last_ra, last_dec) pre ADQL polootvorene intervaly."""
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
    """``use_full_source=False`` -> ``gaia_source_lite`` (rychlejsie, menej 500 od TAP)."""
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
    """Stlpce z ADQL -> male pismena, mapovanie na ocakavane nazvy."""
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
    mag_limit: float,
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
        mag_limit=mag_limit,
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


def _rough_eta_hint(n_strips: int, use_full_source: bool) -> str:
    sec_per = 45.0 if use_full_source else 25.0
    total_s = n_strips * sec_per
    if total_s < 120:
        return f"~{int(total_s)}s (TAP, approx)"
    if total_s < 3600:
        return f"~{int(total_s // 60)}m (TAP, approx)"
    return f"~{total_s / 3600:.1f}h (TAP, approx)"


def create_local_gaia(cfg: BuildConfig) -> None:
    _configure_gaia(cfg)

    db_path = cfg.db_path
    ra_eff = float(cfg.ra_step)
    dec_eff = float(cfg.dec_step)
    if not cfg.no_strip_clamp:
        ra0, dec0 = ra_eff, dec_eff
        ra_eff, dec_eff, n_est = _widen_steps_to_strip_cap(
            cfg.dec_min, cfg.dec_max, ra_eff, dec_eff, int(cfg.max_strips)
        )
        if (ra_eff, dec_eff) != (ra0, dec0):
            print(
                f"  AUTO-CLAMP: too many strips at RA={ra0} Dec={dec0} - "
                f"widened to RA={ra_eff:.2f} Dec={dec_eff:.2f} (max ~{cfg.max_strips} TAP jobs)."
            )
            print("    (GAIA_NO_STRIP_CLAMP=1 to disable auto-widen.)")
            print()

    items = generate_strips_with_flags(cfg.dec_min, cfg.dec_max, dec_eff, ra_eff)
    n_total = len(items)

    print("=" * 62)
    print("  VYVAR - Gaia DR3 local DB (build_gaia_catalog)")
    print("=" * 62)
    print(f"  Output:      {db_path}")
    print(f"  Mag limit:   G <= {cfg.mag_limit}")
    print(f"  Dec:         {cfg.dec_min} .. {cfg.dec_max}  (effective step {dec_eff:.2f})")
    print(f"  RA:          0 .. 360  (effective step {ra_eff:.2f})")
    table = "gaiadr3.gaia_source (FULL)" if cfg.use_full_source else "gaiadr3.gaia_source_lite (default)"
    print(f"  Table:       {table}")
    print(f"  Strips:      {n_total}  (target max ~{cfg.max_strips} for TAP overhead)")
    print(f"  TAP estimate:{_rough_eta_hint(n_total, cfg.use_full_source)}")
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
    print(f"  Done strips: {n_done}/{n_total}, remaining: {len(todo)}")
    print()

    if todo:
        total_inserted = 0
        failed: list[tuple[float, float, float, float]] = []
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
                eta_str = "ETA ..."

            print(
                f"[{n_done + idx:4d}/{n_total}] "
                f"RA {r0:6.2f}-{r1:6.2f} Dec {d0:+6.2f}-{d1:+6.2f} | {eta_str} | inserts ~{total_inserted:,} ",
                end="",
                flush=True,
            )

            df = download_strip(
                strip,
                last_ra,
                last_dec,
                mag_limit=cfg.mag_limit,
                use_full_source=cfg.use_full_source,
            )
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
        print(f"  Total INSERT attempts (batched): {total_inserted:,}")
        if failed:
            print(f"  Failed strips: {len(failed)} - restart skips done (strip_progress).")

    print()
    print("  Creating indexes ...")
    create_indexes(conn)

    if cfg.skip_vacuum:
        print("  --skip-vacuum - VACUUM skipped.")
    else:
        print("  VACUUM (may take a while) ...")
        conn.execute("VACUUM")

    conn.close()
    sz_gb = db_path.stat().st_size / (1024**3)
    print()
    print("=" * 62)
    print("  Done.")
    print(f"  File: {db_path} ({sz_gb:.2f} GB)")
    print("  Set gaia_db_path in config.json to this path after verification.")
    print("=" * 62)


def main(argv: list[str] | None = None) -> None:
    create_local_gaia(parse_args(argv))


if __name__ == "__main__":
    main()
