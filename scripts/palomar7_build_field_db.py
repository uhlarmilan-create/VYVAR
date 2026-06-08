#!/usr/bin/env python3
"""Rebuild Palomar 7 deep field Gaia DB (cone 0.45 deg, G<=20). Part A2+A3 only."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

FIELD_DB = _ROOT / "GAIA_DR3" / "vyvar_gaia_dr3_pal7_field.db"
RESULT_PATH = _ROOT / "palomar7_field_db_rebuild.json"
PAL_RA = 272.684
PAL_DEC = -7.208


def main() -> int:
    sys.path.insert(0, str(_ROOT / "scripts"))
    import build_field_db as bfd  # noqa: E402

    try:
        report = bfd.build_field_db(
            center_ra=PAL_RA,
            center_dec=PAL_DEC,
            radius_deg=0.45,
            out_path=FIELD_DB,
            result_path=RESULT_PATH,
        )
    except Exception as exc:  # noqa: BLE001
        print("FAILED:", exc)
        return 1
    print(__import__("json").dumps(report["field_db"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
