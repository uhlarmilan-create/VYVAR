"""One-off (CAMERA-PROFILE-2026-10): camera model, readout mode and binning convention for the five
cameras of Milan's database, so the new resolver reproduces the gain / read noise the runs used.

usage (from the VYVAR folder):  python dev\\tools\\camera_profile_migrate_2026_10.py [--db vyvar.sqlite3] [--dry]

Before writing, a copy of the database goes to _backup_20261023/camera_profile/vyvar_HHMMSS.sqlite3.
Only rows whose CAMERANAME matches are changed; MODEL_KEY etc. are set only where still empty.
C5A-150M GAIN_ADU 1.0 (placeholder) -> 0.78 (bin 1; the header GAIN 12.48 e-/ADU at bin 4 = 0.78 x 16).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

V = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V / "src_py"))

PLAN: dict[str, dict[str, object]] = {
    "QHY294MM": {"MODEL_KEY": "qhy294m_pro", "READOUT_MODE": "11mp", "BIN_MODE": "sum", "DEFAULT_GAIN_SETTING": 0},
    "C3-26000": {"MODEL_KEY": "moravian_c3_26000"},
    "QHY 600M": {"MODEL_KEY": "qhy600m"},
    "C5A-150M": {"MODEL_KEY": "moravian_c5a_150m"},
    "ASI533MC Pro": {"MODEL_KEY": "zwo_asi533mc_pro"},
}
FIX_VALUES: dict[str, dict[str, tuple[object, object]]] = {
    "C5A-150M": {"GAIN_ADU": (1.0, 0.78)},  # (expected old value, new value)
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(V / "vyvar.sqlite3"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    db_path = Path(a.db)
    if not db_path.is_file():
        print(f"database not found: {db_path}")
        return 2
    from database import VyvarDatabase  # noqa: PLC0415  (creates the new columns / table)

    if not a.dry:
        bdir = V / "_backup_20261023" / "camera_profile"
        bdir.mkdir(parents=True, exist_ok=True)
        src = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        import datetime as _dt  # noqa: PLC0415

        bfile = bdir / f"{db_path.stem}_{_dt.datetime.now():%H%M%S}{db_path.suffix}"
        dst = sqlite3.connect(str(bfile))
        src.backup(dst)
        dst.close()
        src.close()
        print(f"backup: {bfile}")
    db = VyvarDatabase(db_path)
    con = db.conn
    rows = con.execute(
        "SELECT ID, CAMERANAME, MODEL_KEY, READOUT_MODE, BIN_MODE, DEFAULT_GAIN_SETTING, GAIN_ADU FROM EQUIPMENTS"
    ).fetchall()
    for r in rows:
        eid, name = int(r[0]), str(r[1] or "")
        cur = {"MODEL_KEY": r[2], "READOUT_MODE": r[3], "BIN_MODE": r[4], "DEFAULT_GAIN_SETTING": r[5], "GAIN_ADU": r[6]}
        upd = {k: v for k, v in PLAN.get(name, {}).items() if cur.get(k) in (None, "")}
        for k, (old, new) in FIX_VALUES.get(name, {}).items():
            if cur.get(k) is not None and abs(float(cur[k]) - float(old)) < 1e-9:
                upd[k] = new
        if not upd:
            print(f"  {eid} {name}: no change")
            continue
        print(f"  {eid} {name}: " + ", ".join(f"{k} {cur.get(k)!r} -> {v!r}" for k, v in upd.items()))
        if not a.dry:
            con.execute(f"UPDATE EQUIPMENTS SET {', '.join(f'{k} = ?' for k in upd)} WHERE ID = ?;", (*upd.values(), eid))
    if not a.dry:
        con.commit()
    from camera_profile import lookup, profile_from_connection  # noqa: PLC0415

    print("effective values (bin 1 / bin 2 / bin 4 at the default setting):")
    for r in rows:
        p = profile_from_connection(con, int(r[0]))
        vals = []
        for b in (1, 2, 4):
            g, n = lookup(p, "gain", None, b).value, lookup(p, "read_noise", None, b).value
            vals.append(f"bin{b} g={g if g is None else round(g, 4)} rn={n if n is None else round(n, 3)}")
        print(f"  {r[0]} {r[1]}: {p.bin_mode} | " + " | ".join(vals))
    print("dry run - nothing written" if a.dry else "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
