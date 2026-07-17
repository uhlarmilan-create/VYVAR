#!/usr/bin/env python3
"""Build a scoped Gaia DR3 field SQLite DB (cone query + local indexes).

Generalises ``palomar7_build_field_db.py`` for any field centre/radius.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def build_field_db(
    *,
    center_ra: float,
    center_dec: float,
    radius_deg: float,
    out_path: Path,
    mag_limit_initial: float = 20.0,
    mag_limit_cap: float = 19.5,
    result_path: Path | None = None,
) -> dict:
    sys.path.insert(0, str(_ROOT / "scripts"))
    import pilot_palomar7_deep_gaia_ab as pal7  # noqa: E402

    pal7.PAL_RA = float(center_ra)
    pal7.PAL_DEC = float(center_dec)
    pal7.CONE_RADIUS_DEG = float(radius_deg)
    pal7.FIELD_DB = Path(out_path)
    pal7.MAG_LIMIT_INITIAL = float(mag_limit_initial)
    pal7.MAG_LIMIT_CAP = float(mag_limit_cap)

    report: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "center_ra_deg": float(center_ra),
        "center_dec_deg": float(center_dec),
        "cone_radius_deg": float(radius_deg),
        "mag_limit_initial": float(mag_limit_initial),
        "mag_limit_cap": float(mag_limit_cap),
        "out_path": str(Path(out_path).resolve()),
    }
    cone_df, cone_meta = pal7.part_a2_astroquery_cone()
    report["cone_query"] = cone_meta
    report["cone_query"]["faintest_g"] = float(
        __import__("pandas").to_numeric(cone_df.get("g_mag"), errors="coerce").max()
    )
    report["field_db"] = pal7.part_a3_build_field_db(cone_df)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if result_path is not None:
        Path(result_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--center", nargs=2, type=float, metavar=("RA", "DEC"), required=True)
    ap.add_argument("--radius", type=float, required=True, help="Cone radius [deg]")
    ap.add_argument("--out", type=Path, required=True, help="Output SQLite path")
    ap.add_argument("--mag-limit", type=float, default=20.0, help="Initial G mag limit")
    ap.add_argument("--mag-cap", type=float, default=19.5, help="Fallback G mag cap")
    ap.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Optional JSON report path (default: <out_stem>_build.json beside out)",
    )
    args = ap.parse_args()
    result_path = args.result_json
    if result_path is None:
        result_path = args.out.with_name(args.out.stem + "_build.json")

    try:
        report = build_field_db(
            center_ra=float(args.center[0]),
            center_dec=float(args.center[1]),
            radius_deg=float(args.radius),
            out_path=args.out,
            mag_limit_initial=float(args.mag_limit),
            mag_limit_cap=float(args.mag_cap),
            result_path=result_path,
        )
    except Exception as exc:  # noqa: BLE001
        err = {
            "error": str(exc),
            "center": [float(args.center[0]), float(args.center[1])],
            "radius_deg": float(args.radius),
            "out": str(args.out),
        }
        result_path.write_text(json.dumps(err, indent=2), encoding="utf-8")
        print("FAILED:", exc)
        return 1

    print(json.dumps(report["field_db"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
