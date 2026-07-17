"""Repair precision-loss Gaia catalog_id values in variable_targets.csv.

Use case:
  External tools sometimes save 19-digit Gaia IDs into CSV as float/scientific notation,
  which truncates precision (float64). If RA/DEC are present, we can recover the correct
  Gaia DR3 source_id by nearest-neighbor lookup in the local Gaia SQLite DB.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Callable

import pandas as pd


def _default_log(msg: str) -> None:
    print(msg)


_DET_PLACEHOLDER_RE = re.compile(r"^DET_\d+$", re.IGNORECASE)


def _pick_gaia_table(con: sqlite3.Connection) -> str:
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {str(r[0]) for r in rows if r and r[0]}
    for cand in ("gaia_dr3", "gaia_source", "gaia"):
        if cand in names:
            return cand
    raise RuntimeError(f"Gaia DB: neznáma tabuľka (nájdené: {sorted(names)[:20]})")


def _sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Small-angle separation in arcsec (good for tiny offsets)."""
    dra = (ra2 - ra1) * math.cos(math.radians(dec1))
    ddec = dec2 - dec1
    return math.sqrt(dra * dra + ddec * ddec) * 3600.0


def repair_csv_catalog_ids_from_gaia_db(
    *,
    csv_path: Path,
    gaia_db_path: Path,
    id_col: str = "catalog_id",
    backup: bool = True,
    max_sep_arcsec: float = 10.0,
    log_fn: Callable[[str], None] | None = None,
    skip_unmatched_placeholders: bool = False,
    box_deg: float = 0.001,
) -> dict[str, Any]:
    """Repair Gaia IDs in-place using Gaia DB RA/DEC nearest match.

    Strategy per row:
      1) If ID in `id_col` exists in Gaia DB -> keep.
      2) Else query nearest by RA/DEC in +/-0.001 deg box and replace if sep <= max_sep_arcsec.

    Returns summary dict with counts and per-row info (best-effort).
    """
    log = log_fn or _default_log
    vt_path = Path(csv_path)
    db_path = Path(gaia_db_path)
    if not vt_path.is_file():
        raise FileNotFoundError(vt_path)
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    df = pd.read_csv(vt_path, dtype={str(id_col): str}, low_memory=False)
    if df.empty or str(id_col) not in df.columns:
        return {
            "ok": True,
            "repaired": 0,
            "warnings": 0,
            "rows": int(len(df)),
            "table": "",
            "path": str(vt_path),
            "id_col": str(id_col),
        }

    # Normalize obvious whitespace/None; do not attempt to "fix" float-rounded scientific notation here.
    df[str(id_col)] = df[str(id_col)].fillna("").astype(str).str.strip()

    con = sqlite3.connect(str(db_path))
    try:
        table = _pick_gaia_table(con)

        def _exists_source_id(source_id: int) -> bool:
            r = con.execute(f"SELECT source_id FROM {table} WHERE source_id=? LIMIT 1", (int(source_id),)).fetchone()
            return bool(r and r[0] is not None)

        repaired = 0
        warnings = 0
        per_row: list[dict[str, Any]] = []
        checked = 0
        kept_placeholder = 0
        no_gaia_in_box = 0
        over_max_sep = 0
        invalid_coords = 0

        for i in range(len(df)):
            row = df.iloc[i]
            old_raw = str(row.get(str(id_col), "") or "").strip()
            vsx_name = str(row.get("vsx_name", "") or row.get("name", "") or "").strip()

            if skip_unmatched_placeholders and (
                not old_raw or old_raw.lower() in ("nan", "none") or _DET_PLACEHOLDER_RE.match(old_raw)
            ):
                kept_placeholder += 1
                continue

            ra = row.get("ra_deg", row.get("ra", float("nan")))
            dec = row.get("dec_deg", row.get("dec", float("nan")))
            try:
                ra_f = float(ra)
                dec_f = float(dec)
            except Exception:  # noqa: BLE001
                ra_f, dec_f = float("nan"), float("nan")

            # Step 1: check by ID (if parseable)
            ok_id = False
            old_int: int | None = None
            try:
                if old_raw and old_raw.lower() not in ("nan", "none"):
                    # int("1498...") OK; int("1.49e18") fails -> will go to RA/DEC
                    old_int = int(old_raw)
                    ok_id = _exists_source_id(old_int)
            except Exception:  # noqa: BLE001
                ok_id = False

            if ok_id:
                continue

            checked += 1

            if not (math.isfinite(ra_f) and math.isfinite(dec_f)):
                invalid_coords += 1
                continue

            box = float(box_deg)
            r = con.execute(
                f"""
                SELECT source_id, ra, dec
                FROM {table}
                WHERE ra  BETWEEN ? AND ?
                  AND dec BETWEEN ? AND ?
                ORDER BY
                  (ra-?)*(ra-?) + (dec-?)*(dec-?)
                LIMIT 1
                """,
                (ra_f - box, ra_f + box, dec_f - box, dec_f + box, ra_f, ra_f, dec_f, dec_f),
            ).fetchone()

            if not r or r[0] is None:
                no_gaia_in_box += 1
                continue

            new_id = int(r[0])
            ra2 = float(r[1]) if r[1] is not None else float("nan")
            dec2 = float(r[2]) if r[2] is not None else float("nan")
            sep = _sep_arcsec(ra_f, dec_f, ra2, dec2) if (math.isfinite(ra2) and math.isfinite(dec2)) else float("nan")

            if math.isfinite(sep) and sep > float(max_sep_arcsec):
                over_max_sep += 1
                warnings += 1
                log(
                    f"REPAIR WARNING: {old_raw} -> {new_id} ({vsx_name}, sep={sep:.2f}) presahuje max_sep_arcsec={max_sep_arcsec:.1f} -> NEOPRAVUJEM"
                )
                continue

            if str(new_id) != old_raw:
                df.at[i, str(id_col)] = str(new_id)
                repaired += 1
                log(f"REPAIR: {old_raw} -> {new_id} ({vsx_name}, sep={sep:.2f})")
                per_row.append({"row": int(i), "vsx_name": vsx_name, "old": old_raw, "new": str(new_id), "sep_arcsec": sep})

        log(
            "REPAIR summary: "
            f"checked={checked} repaired={repaired} kept_placeholder={kept_placeholder} "
            f"no_gaia_in_box={no_gaia_in_box} over_max_sep={over_max_sep}"
            + (f" invalid_coords={invalid_coords}" if invalid_coords else "")
        )

        if repaired and backup:
            bak = vt_path.with_suffix(vt_path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(vt_path, bak)
            df.to_csv(vt_path, index=False)

        return {
            "ok": True,
            "path": str(vt_path),
            "backup": str(vt_path.with_suffix(vt_path.suffix + ".bak")) if backup else "",
            "table": table,
            "id_col": str(id_col),
            "rows": int(len(df)),
            "repaired": int(repaired),
            "warnings": int(warnings),
            "checked": int(checked),
            "kept_placeholder": int(kept_placeholder),
            "no_gaia_in_box": int(no_gaia_in_box),
            "over_max_sep": int(over_max_sep),
            "invalid_coords": int(invalid_coords),
            "details": per_row,
        }
    finally:
        con.close()


def repair_catalog_ids_from_gaia_db(
    *,
    variable_targets_csv: Path,
    gaia_db_path: Path,
    backup: bool = True,
    max_sep_arcsec: float = 10.0,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper (historicky pre variable_targets.csv)."""
    return repair_csv_catalog_ids_from_gaia_db(
        csv_path=Path(variable_targets_csv),
        gaia_db_path=Path(gaia_db_path),
        id_col="catalog_id",
        backup=backup,
        max_sep_arcsec=max_sep_arcsec,
        log_fn=log_fn,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vt", required=True, help="Path to variable_targets.csv (or any CSV to repair)")
    ap.add_argument("--gaia-db", required=True, help="Path to Gaia SQLite DB (vyvar_gaia_dr3*.db)")
    ap.add_argument("--id-col", default="catalog_id", help="Column name to repair (default: catalog_id)")
    ap.add_argument("--max-sep-arcsec", type=float, default=10.0)
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    res = repair_csv_catalog_ids_from_gaia_db(
        csv_path=Path(args.vt),
        gaia_db_path=Path(args.gaia_db),
        id_col=str(args.id_col or "catalog_id"),
        backup=not bool(args.no_backup),
        max_sep_arcsec=float(args.max_sep_arcsec),
        log_fn=_default_log,
    )
    _default_log(f"DONE: repaired={res.get('repaired')} warnings={res.get('warnings')} rows={res.get('rows')} table={res.get('table')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

