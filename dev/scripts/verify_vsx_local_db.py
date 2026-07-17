#!/usr/bin/env python3
"""Read-only VSX local DB verification (orchestrator task 2026-06-03)."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import query_local_vsx, validate_vsx_local_db_schema  # noqa: E402


def _field_center_from_draft(draft_dir: Path) -> tuple[float, float, str]:
    """RA/Dec deg from platesolve MASTERSTAR WCS or photometry_summary median."""
    ps = draft_dir / "platesolve"
    ms_candidates = sorted(ps.glob("*/MASTERSTAR.fits")) if ps.is_dir() else []
    if ms_candidates:
        from astropy.io import fits  # noqa: PLC0415

        with fits.open(ms_candidates[0], memmap=False) as hdul:
            h = hdul[0].header
            ra = float(h.get("CRVAL1", float("nan")))
            dec = float(h.get("CRVAL2", float("nan")))
        if ra == ra and dec == dec:  # finite
            return ra, dec, str(ms_candidates[0])
    summ = sorted(ps.glob("*/photometry/photometry_summary.csv")) if ps.is_dir() else []
    if summ:
        df = pd.read_csv(summ[0], usecols=["ra_deg", "dec_deg"], low_memory=False)
        ra = float(pd.to_numeric(df["ra_deg"], errors="coerce").median())
        de = float(pd.to_numeric(df["dec_deg"], errors="coerce").median())
        if ra == ra and de == de:
            return ra, de, str(summ[0])
    raise FileNotFoundError(f"No WCS/photometry center for {draft_dir}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="VSX SQLite path (default: config.vsx_local_db_path)")
    ap.add_argument("--draft", type=int, help="Draft id for cone + integration (e.g. 366)")
    args = ap.parse_args()

    cfg = AppConfig()
    db_cfg = Path(cfg.vsx_local_db_path).expanduser().resolve()
    db = Path(args.db).expanduser().resolve() if args.db else db_cfg

    print("=== Step 1: configured path ===")
    print(f"config.vsx_local_db_path: {db_cfg}")
    print(f"config exists: {db_cfg.is_file()}")
    if db_cfg.is_file():
        print(f"config size bytes: {db_cfg.stat().st_size}")
    print(f"verify db path: {db}")
    print(f"verify db exists: {db.is_file()}")
    if not db.is_file():
        print("STOP: DB file missing")
        return 1
    print(f"verify size bytes: {db.stat().st_size}")

    print("\n=== Step 2: validate_vsx_local_db_schema ===")
    ok, code = validate_vsx_local_db_schema(db)
    print(f"result: ok={ok} code={code!r}")
    if not ok:
        return 1

    con = sqlite3.connect(str(db))
    try:
        print("\n=== Step 3: PRAGMA table_info('vsx_data') ===")
        info = con.execute("PRAGMA table_info('vsx_data')").fetchall()
        for r in info:
            print(r)
        names = [r[1] for r in info]
        need = ["oid", "name", "ra_deg", "dec_deg", "var_type", "period", "mag_max", "mag_min"]
        missing = [c for c in need if c not in names]
        oid_pk = any(r[1] == "oid" and int(r[5]) == 1 for r in info)
        print(f"required present: {not missing} missing={missing}")
        print(f"oid PRIMARY KEY (pk=1): {oid_pk}")

        print("\n=== Step 4: duplicate OIDs ===")
        dups = con.execute(
            "SELECT oid, COUNT(*) c FROM vsx_data GROUP BY oid HAVING c > 1 LIMIT 5;"
        ).fetchall()
        print(f"dup rows: {dups}")

        print("\n=== Step 5: indexes ===")
        idx = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='vsx_data';"
            ).fetchall()
        ]
        print(idx)

        print("\n=== Step 6: population ===")
        n = con.execute("SELECT COUNT(*) FROM vsx_data;").fetchone()[0]
        c_period, c_mag, c_type = con.execute(
            "SELECT COUNT(period), COUNT(mag_max), COUNT(var_type) FROM vsx_data;"
        ).fetchone()
        print(f"COUNT(*)={n}")
        print(f"non-null period={c_period} ({100*c_period/n:.2f}%)")
        print(f"non-null mag_max={c_mag} ({100*c_mag/n:.2f}%)")
        print(f"non-null var_type={c_type} ({100*c_type/n:.2f}%)")

        tables = [
            t[0]
            for t in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1;"
            ).fetchall()
        ]
        print(f"tables: {tables}")
    finally:
        con.close()

    if args.draft is None:
        return 0

    draft_dir = _ROOT / "Archive" / "Drafts" / f"draft_{int(args.draft):06d}"
    ra_c, dec_c, src = _field_center_from_draft(draft_dir)
    half = 0.25
    print(f"\n=== Step 7: cone query (draft {args.draft}) ===")
    print(f"center RA={ra_c:.6f} Dec={dec_c:.6f} from {src}")
    print(f"box ±{half}°")
    rows = query_local_vsx(
        db,
        ra_min=ra_c - half,
        ra_max=ra_c + half,
        dec_min=dec_c - half,
        dec_max=dec_c + half,
        max_rows=50_000,
    )
    print(f"rows returned: {len(rows)}")
    for ex in rows[:3]:
        print("example:", {k: ex.get(k) for k in ("oid", "name", "var_type", "period", "mag_max", "ra_deg", "dec_deg")})

    print("\n=== Step 8: mag-cut (cone, limit from config) ===")
    mag_lim = float(cfg.vsx_variable_targets_mag_limit)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no rows for mag-cut")
    else:
        mm = pd.to_numeric(df.get("mag_max"), errors="coerce")
        n_null = int(mm.isna().sum())
        n_le = int((mm <= mag_lim).sum())
        n_gt = int((mm > mag_lim).sum())
        kept = int((mm.isna() | (mm <= mag_lim)).sum())
        print(f"mag_limit={mag_lim}")
        print(f"null mag_max: {n_null}")
        print(f"mag_max <= limit: {n_le}")
        print(f"mag_max > limit (would drop): {n_gt}")
        print(f"kept (null OR <= limit): {kept} / {len(df)}")

    print(f"\n=== Step 9: integration draft {args.draft} ===")
    from catalog_crossmatch import check_candidate_in_catalogs  # noqa: PLC0415

    ms_csv = sorted((draft_dir / "platesolve").glob("*/masterstars.csv"))
    if not ms_csv:
        print("no masterstars.csv")
        return 0
    ms = pd.read_csv(ms_csv[0], dtype={"catalog_id": str}, low_memory=False)
    ms["ra_deg"] = pd.to_numeric(ms.get("ra_deg"), errors="coerce")
    ms["dec_deg"] = pd.to_numeric(ms.get("dec_deg"), errors="coerce")
    ms = ms[ms["ra_deg"].notna() & ms["dec_deg"].notna()].head(200)
    n_vsx = 0
    examples: list[str] = []
    for _, row in ms.iterrows():
        res = check_candidate_in_catalogs(
            float(row["ra_deg"]),
            float(row["dec_deg"]),
            float(row.get("mag", float("nan"))),
            radius_arcsec=10.0,
            vsx_local_db_path=str(db),
        )
        matches = (res.matches or {}).get("VSX", []) if hasattr(res, "matches") else []
        if matches:
            n_vsx += 1
            if len(examples) < 5:
                m0 = matches[0]
                examples.append(f"{m0.name} {m0.var_type}")
    print(f"VSX matches in first {len(ms)} masterstars (10 arcsec): {n_vsx}")
    print("examples:", examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
