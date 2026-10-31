"""build_gaia_catalog.py - local Gaia DR3 SQLite for VYVAR (table ``gaia_dr3``).

Downloads Gaia DR3 from the ESA archive (TAP, astroquery) in RA/Dec strips:

  * half-open RA/Dec intervals: no duplicates on strip borders;
  * INTEGER PRIMARY KEY (source_id) + INSERT OR IGNORE: safe resume;
  * ``strip_progress`` records finished strips; an interrupted build continues where it stopped;
  * G05-03 (Milan 2026-10-31): a strip is done only when the rows received equal the TAP
    ``COUNT(*)`` of the same query (same WHERE, same G limit); otherwise it is split into four
    sub-boxes and fetched again, recursively. The anonymous archive caps a job at a row limit, so
    a dense strip came back cut. Each strip is stored atomically (one transaction);
  * ``gaia_coverage`` records the box and G limit the build covers (VYVAR tops up fainter
    stars online per field);
  * the indexes VYVAR queries use (``catalog_db.GAIA_INDEXES``) and the ``catalog_meta`` row
    (source, build date, limits, row count) are written at the end.

Output: the ``gaia_db_path`` of config.json (Settings -> Paths and catalogs), default
``<data>/catalogs/vyvar_gaia_dr3.db``.

Examples (from the VYVAR folder):
  python catalogs/build_gaia_catalog.py --help
  python catalogs/build_gaia_catalog.py --mag-limit 17.5
  python catalogs/build_gaia_catalog.py --check                  # strips not verified complete
  python catalogs/build_gaia_catalog.py --repair --strip 258.3495_270.0926_-34.0319_-26.0365
  python catalogs/build_gaia_catalog.py --dec-min 89 --dec-max 90 --mag-limit 10 --skip-vacuum --out tmp/smoke_gaia.db

Environment fallbacks (when the option is not given): GAIA_MAG_LIMIT, GAIA_DEC_MIN, GAIA_DEC_MAX,
GAIA_OUT, GAIA_RA_STEP, GAIA_DEC_STEP, GAIA_MAX_STRIPS, GAIA_SOURCE_FULL, SKIP_VACUUM,
GAIA_NO_STRIP_CLAMP. Optional archive login: GAIA_USER + GAIA_PASS (anonymous TAP otherwise).

After a change of the strip grid delete ``strip_progress`` or the whole DB.
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

_HERE = Path(__file__).resolve().parent


def _find_src(start: Path) -> tuple[Path, Path]:
    """(install root, src_py) of a checkout or of a release bundle (compiled modules)."""
    for candidate in (start, *start.parents):
        src = candidate / "src_py"
        if (src / "gaia_coverage.py").is_file() or any(src.glob("gaia_coverage*.pyd")) or any(src.glob("gaia_coverage*.so")):
            return candidate, src
    raise SystemExit("build_gaia_catalog.py needs src_py/gaia_coverage from the VYVAR install; run it from the install.")


_ROOT, _SRC = _find_src(_HERE)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402
from astroquery.gaia import Gaia  # noqa: E402

from gaia_coverage import (  # noqa: E402
    GAIA_SOURCE_LITE_SELECT,
    GAIA_SOURCE_SELECT,
    IncompleteDownload,
    SkyBox,
    ensure_strip_columns,
    fetch_region_complete,
    init_gaia_tables,
    insert_gaia_rows,
    record_coverage,
    region_adql,
    region_count_adql,
    strips_needing_repair,
)

from catalog_db import (  # noqa: E402
    GAIA_BUILD_MAG_LIMIT_DEFAULT,
    GAIA_DB_DEFAULT,
    GAIA_INDEXES,
    add_catalog_meta_rows,
    configured_catalog_path,
    write_catalog_meta,
)

BUILDER = "catalogs/build_gaia_catalog.py"
DEFAULT_OUT = configured_catalog_path("gaia_db_path", GAIA_DB_DEFAULT)

MAX_RETRIES = 8
RETRY_BASE_S = 20
RETRY_MAX_WAIT_S = 240

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
    check: bool = False
    repair: bool = False
    repair_keys: tuple[str, ...] = ()


def parse_args(argv: list[str] | None = None) -> BuildConfig:
    parser = argparse.ArgumentParser(
        description="Download Gaia DR3 into local SQLite for VYVAR (strip TAP builder).",
    )
    parser.add_argument(
        "--mag-limit",
        type=float,
        default=_env_float("GAIA_MAG_LIMIT", GAIA_BUILD_MAG_LIMIT_DEFAULT),
        help=f"Upper G mag limit (default {GAIA_BUILD_MAG_LIMIT_DEFAULT:g}; env GAIA_MAG_LIMIT).",
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
        help=f"Output SQLite path (default {DEFAULT_OUT}; env GAIA_OUT).",
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
    parser.add_argument(
        "--check",
        action="store_true",
        help="List the strips of --out whose completeness is not verified (G05-03) and exit.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-check (COUNT) and, when short, re-download the unverified strips of --out (G05-03); "
        "resumable, one strip per transaction. With --strip only those strips.",
    )
    parser.add_argument(
        "--strip",
        action="append",
        default=[],
        metavar="KEY",
        help="strip_key (ra0_ra1_dec0_dec1, as --check prints it) for --repair; repeatable.",
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
        check=bool(args.check),
        repair=bool(args.repair),
        repair_keys=tuple(str(k).strip() for k in args.strip if str(k).strip()),
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
    """Widen the RA/Dec steps until the strip count is <= max_strips (each strip is one TAP job)."""
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
    """List of ((r0, r1, d0, d1), last_ra, last_dec) for half-open ADQL intervals."""
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


def _source(use_full_source: bool) -> tuple[str, str]:
    """(TAP table, SELECT list); ``gaia_source_lite`` by default (faster, fewer HTTP 500 from TAP)."""
    return (("gaiadr3.gaia_source", GAIA_SOURCE_SELECT) if use_full_source
            else ("gaiadr3.gaia_source_lite", GAIA_SOURCE_LITE_SELECT))


def _strip_box(r0: float, r1: float, d0: float, d1: float, last_ra: bool, last_dec: bool) -> SkyBox:
    """Half-open strip box; the last RA / Dec strip includes its upper edge."""
    return SkyBox(r0, r1, d0, d1, ra_closed=last_ra, de_closed=last_dec)


def _mag_where(mag_limit: float) -> str:
    return f"phot_g_mean_mag IS NOT NULL\n          AND phot_g_mean_mag <= {float(mag_limit)}"


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
    """Row query of one strip; :func:`build_count_adql` is its COUNT(*) (same WHERE)."""
    table, select = _source(use_full_source)
    return region_adql([_strip_box(r0, r1, d0, d1, last_ra, last_dec)], select=select, table=table,
                       mag_where=_mag_where(mag_limit))


def build_count_adql(
    r0: float, r1: float, d0: float, d1: float, *, last_ra: bool, last_dec: bool, mag_limit: float,
    use_full_source: bool,
) -> str:
    table, _select = _source(use_full_source)
    return region_count_adql([_strip_box(r0, r1, d0, d1, last_ra, last_dec)], table=table,
                             mag_where=_mag_where(mag_limit))


def get_done_strips(conn: sqlite3.Connection) -> set[str]:
    """Strips that need no download: verified complete, or of an older build (never checked; see
    ``--check`` / ``--repair``). A strip the COUNT check found truncated (verified 0) is not done."""
    try:
        cur = conn.execute("SELECT strip_key FROM strip_progress WHERE verified IS NULL OR verified = 1")
        return {str(r[0]) for r in cur.fetchall()}
    except sqlite3.Error:
        return set()


def mark_strip_done(
    conn: sqlite3.Connection,
    strip: tuple[float, float, float, float],
    n_stars: int,
    mag_limit: float,
    *,
    n_expected: int | None,
    verified: bool = True,
) -> None:
    """Record a strip: ``verified`` True when its rows equal the TAP COUNT(*) ``n_expected``,
    False when the COUNT check found it truncated (never counted as done; ``n_expected`` may be
    unknown then)."""
    if verified and (n_expected is None or int(n_stars) != int(n_expected)):
        raise ValueError(f"strip {strip_key(strip)}: {n_stars} rows != COUNT(*) {n_expected}; not done")
    r0, r1, d0, d1 = strip
    conn.execute(
        """
        INSERT OR REPLACE INTO strip_progress
            (strip_key, ra_min, ra_max, dec_min, dec_max, n_stars, finished_at, mag_limit, n_expected, verified)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            strip_key(strip),
            r0,
            r1,
            d0,
            d1,
            n_stars,
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            float(mag_limit),
            None if n_expected is None else int(n_expected),
            1 if verified else 0,
        ),
    )


def built_depth_g(conn: sqlite3.Connection, keys: set[str]) -> float | None:
    """G to which every strip of the grid is complete: the smallest strip limit. A strip from a
    build before the limit was stored counts with the faintest star it holds. G05-03: a strip the
    COUNT check found truncated has no depth, so the grid has none (None)."""
    depth: float | None = None
    for key, r0, r1, d0, d1, ml, ver in conn.execute(
        "SELECT strip_key, ra_min, ra_max, dec_min, dec_max, mag_limit, verified FROM strip_progress"
    ).fetchall():
        if str(key) not in keys:
            continue
        if ver is not None and int(ver) == 0:
            return None
        if ml is None:
            row = conn.execute(
                "SELECT MAX(g_mag) FROM gaia_dr3 WHERE ra >= ? AND ra <= ? AND dec >= ? AND dec <= ?",
                (r0, r1, d0, d1),
            ).fetchone()
            if row is None or row[0] is None:
                continue
            ml = float(row[0])
        depth = float(ml) if depth is None else min(depth, float(ml))
    return depth


def _tap_fetch(adql: str, *, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    """One TAP job (rows or COUNT) with retries; raises after ``max_retries`` failures."""
    for attempt in range(max_retries):
        try:
            results = Gaia.launch_job_async(adql).get_results()
            return results.to_pandas() if len(results) else pd.DataFrame()
        except Exception as e:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            wait = min(RETRY_MAX_WAIT_S, int(RETRY_BASE_S * (2**attempt)))
            print(f"\n    WARN attempt {attempt + 1}/{max_retries}: {e}")
            print(f"    waiting {wait} s ...", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def download_strip_complete(
    conn: sqlite3.Connection,
    strip: tuple[float, float, float, float],
    last_ra: bool,
    last_dec: bool,
    *,
    mag_limit: float,
    use_full_source: bool,
    fetch=_tap_fetch,
) -> tuple[int, int, int]:
    """G05-03: download one strip completely (COUNT check, split in four when short) and insert
    its rows into ``conn`` (no commit). Returns (COUNT(*), rows received, new rows)."""
    table, select = _source(use_full_source)
    n_new = 0

    def _insert(df: pd.DataFrame) -> None:
        nonlocal n_new
        n_new += insert_gaia_rows(conn, df)

    n_exp, n_rows = fetch_region_complete(
        fetch, [_strip_box(*strip, last_ra, last_dec)], select=select, table=table,
        mag_where=_mag_where(mag_limit), on_rows=_insert,
    )
    return n_exp, n_rows, n_new


def _store_strip(
    conn: sqlite3.Connection,
    strip: tuple[float, float, float, float],
    last_ra: bool,
    last_dec: bool,
    *,
    mag_limit: float,
    use_full_source: bool,
    fetch,
    n_stars_if_truncated: int = 0,
) -> tuple[str, int, int]:
    """Download + record one strip in ONE transaction: ("ok", rows, new), ("truncated", 0, 0)
    (its new rows rolled back, recorded verified 0 with ``n_stars_if_truncated``) or
    ("error", 0, 0) (nothing recorded, the strip stays as it was)."""
    try:
        n_exp, n_rows, n_new = download_strip_complete(
            conn, strip, last_ra, last_dec, mag_limit=mag_limit, use_full_source=use_full_source, fetch=fetch)
        mark_strip_done(conn, strip, n_rows, mag_limit, n_expected=n_exp)
        add_catalog_meta_rows(conn, "gaia_dr3", n_new)
        conn.commit()
        return "ok", n_rows, n_new
    except IncompleteDownload as exc:
        conn.rollback()
        print(f" TRUNCATED: {exc}")
        mark_strip_done(conn, strip, int(n_stars_if_truncated), mag_limit, n_expected=None, verified=False)
        conn.commit()
        return "truncated", 0, 0
    except Exception as exc:  # noqa: BLE001 - archive down / insert failure: strip stays to do
        conn.rollback()
        print(f" ERR: {type(exc).__name__}: {exc}")
        return "error", 0, 0


def create_indexes(conn: sqlite3.Connection) -> None:
    """Only the indexes VYVAR queries use (an unused index costs GB and build time)."""
    for sql in GAIA_INDEXES.values():
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


def create_local_gaia(cfg: BuildConfig, *, fetch=_tap_fetch) -> None:
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
    init_gaia_tables(conn)
    ensure_strip_columns(conn)
    conn.commit()

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

            state, n_rows, n_new = _store_strip(
                conn, strip, last_ra, last_dec, mag_limit=cfg.mag_limit,
                use_full_source=cfg.use_full_source, fetch=fetch,
            )
            if state != "ok":
                failed.append(strip)
                continue
            total_inserted += n_new
            recent_dt.append(time.time() - t_strip)
            print(f" ok tap_rows={n_rows:,} (= COUNT) new={n_new:,}")

        print()
        print(f"  New stars inserted: {total_inserted:,}")
        if failed:
            print(f"  Failed strips: {len(failed)} - restart skips done (strip_progress).")

    left = [s for (s, _lr, _ld) in items if strip_key(s) not in get_done_strips(conn)]
    n_cov = int(conn.execute("SELECT COUNT(*) FROM gaia_coverage").fetchone()[0])
    if not left and (todo or n_cov == 0):
        # FRAME-DEPTH-2026-10: the main DB records what it covers (box + G limit); VYVAR fetches
        # deeper stars online for a field whose frame reaches past it. Until this record exists
        # (empty gaia_coverage table) VYVAR claims no depth for a DB still being built.
        conn.commit()
        g_cov = built_depth_g(conn, {strip_key(s) for (s, _lr, _ld) in items})
        if g_cov is not None:
            record_coverage(
                db_path,
                box=(0.0, 360.0, float(cfg.dec_min), float(cfg.dec_max)),
                g_max=float(g_cov),
                source="build_gaia_catalog",
            )
            print(f"  Coverage:    RA 0-360, Dec {cfg.dec_min}..{cfg.dec_max}, G <= {g_cov}")

    print()
    print("  Creating indexes ...")
    create_indexes(conn)

    table_name = "gaiadr3.gaia_source" if cfg.use_full_source else "gaiadr3.gaia_source_lite"
    built = conn.execute("SELECT MAX(finished_at) FROM strip_progress").fetchone()[0]
    g_cov = built_depth_g(conn, {strip_key(s) for (s, _lr, _ld) in items})
    rec = write_catalog_meta(
        conn, "gaia_dr3", source=f"ESA Gaia DR3 {table_name} (TAP)", builder=BUILDER, built_utc=built,
        params={"g_max": g_cov, "ra": [0.0, 360.0], "dec": [float(cfg.dec_min), float(cfg.dec_max)],
                "strips_left": len(left), **_completeness_params(conn)},
    )
    print(f"  catalog_meta: {rec['row_count']:,} stars, G <= {g_cov}, built {built}")

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
    if db_path.resolve() != DEFAULT_OUT.resolve():
        print("  VYVAR reads the path in Settings -> Paths and catalogs (gaia_db_path); set it to this file.")
    print("=" * 62)


def _completeness_params(conn: sqlite3.Connection) -> dict[str, object]:
    """catalog_meta params of G05-03: strips never COUNT-checked, strips known truncated."""
    bad = strips_needing_repair(conn)
    return {"strips_unverified": sum(1 for s in bad if s["state"] == "never_checked"),
            "truncated_strips": [s["strip_key"] for s in bad if s["state"] == "truncated"]}


def check_strips(db_path: Path) -> list[dict]:
    """``--check``: print and return the strips whose completeness is not verified."""
    from catalog_db import connect_ro  # noqa: PLC0415

    conn = connect_ro(db_path)
    try:
        bad = strips_needing_repair(conn)
    finally:
        conn.close()
    print(f"  {db_path}: {len(bad)} strip(s) not verified complete")
    for s in bad:
        tag = " AT ANONYMOUS TAP ROW CAP" if s["at_anonymous_cap"] else ""
        print(f"    {s['strip_key']:40s} {s['state']:14s} rows={s['n_stars']} expected={s['n_expected']}{tag}")
    return bad


def repair_strips(cfg: BuildConfig, *, fetch=_tap_fetch) -> dict[str, int]:
    """``--repair`` (G05-03): for every unverified strip (or the ``--strip`` keys) run the COUNT(*)
    of its query; equal to the stored rows -> marked verified (no download); otherwise downloaded
    again completely (split in four when short), its rows added (INSERT OR IGNORE) and recorded,
    one strip per transaction (resumable: a repaired strip is verified and skipped next time). The
    strip keeps its stored G limit; a strip of an older build without one uses --mag-limit."""
    _configure_gaia(cfg)
    conn = sqlite3.connect(str(cfg.db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    init_gaia_tables(conn)
    ensure_strip_columns(conn)
    conn.commit()
    todo = strips_needing_repair(conn)
    if cfg.repair_keys:
        known = {s["strip_key"] for s in todo}
        missing = [k for k in cfg.repair_keys if k not in known]
        if missing:
            conn.close()
            raise SystemExit(f"--strip not among the unverified strips of {cfg.db_path}: {missing}")
        todo = [s for s in todo if s["strip_key"] in set(cfg.repair_keys)]
    dec_top = conn.execute("SELECT MAX(dec_max) FROM strip_progress").fetchone()[0]
    table, _sel = _source(cfg.use_full_source)
    print(f"  Repair:      {len(todo)} strip(s) of {cfg.db_path} ({table})")
    out = {"verified": 0, "downloaded": 0, "truncated": 0, "error": 0, "new_rows": 0}
    for i, s in enumerate(todo, start=1):
        r0, r1, d0, d1 = s["box"]
        last_ra, last_dec = r1 >= 360.0 - 1e-9, dec_top is not None and abs(d1 - float(dec_top)) < 1e-9
        ml = float(s["mag_limit"]) if s["mag_limit"] is not None else float(cfg.mag_limit)
        print(f"[{i:4d}/{len(todo)}] {s['strip_key']} G <= {ml:g} stored rows={s['n_stars']} ", end="", flush=True)
        try:
            n_exp = int(fetch(build_count_adql(r0, r1, d0, d1, last_ra=last_ra, last_dec=last_dec,
                                               mag_limit=ml, use_full_source=cfg.use_full_source)).iloc[0, 0])
        except Exception as exc:  # noqa: BLE001
            print(f" ERR (COUNT): {type(exc).__name__}: {exc}")
            out["error"] += 1
            continue
        if s["n_stars"] is not None and int(s["n_stars"]) == n_exp:
            mark_strip_done(conn, s["box"], n_exp, ml, n_expected=n_exp)
            conn.commit()
            print(f" COUNT={n_exp:,} = stored rows: verified")
            out["verified"] += 1
            continue
        print(f" COUNT={n_exp:,}: downloading again", end="", flush=True)
        state, n_rows, n_new = _store_strip(conn, s["box"], last_ra, last_dec, mag_limit=ml,
                                            use_full_source=cfg.use_full_source, fetch=fetch,
                                            n_stars_if_truncated=int(s["n_stars"] or 0))
        if state == "ok":
            print(f" ok rows={n_rows:,} new={n_new:,}")
            out["downloaded"] += 1
            out["new_rows"] += n_new
        else:
            out[state] += 1
    params = _completeness_params(conn)
    conn.close()
    print(f"  Repair done: {out}; still unverified {params['strips_unverified']}, "
          f"truncated {len(params['truncated_strips'])}")
    return out


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    if cfg.check:
        check_strips(cfg.db_path)
    elif cfg.repair:
        repair_strips(cfg)
    else:
        create_local_gaia(cfg)


if __name__ == "__main__":
    main()
