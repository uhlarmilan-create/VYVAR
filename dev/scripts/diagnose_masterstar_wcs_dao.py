#!/usr/bin/env python3
"""CLI wrapper pre ``masterstar_wcs_dao_diagnostic.run_masterstar_wcs_dao_diagnostic``.

Použitie:
  python scripts/diagnose_masterstar_wcs_dao.py --archive "C:/.../Drafts/draft_000029"
  python scripts/diagnose_masterstar_wcs_dao.py --fits .../MASTERSTAR.fits --csv .../masterstars_full_match.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Spustenie z ľubovoľného CWD: pridaj koreň projektu
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from masterstar_wcs_dao_diagnostic import (  # noqa: E402
    resolve_paths_from_archive,
    run_masterstar_wcs_dao_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="DAO vs Gaia→pixel reziduály pre MASTERSTAR.")
    parser.add_argument("--archive", type=str, default="", help="Koreň draftu (platesolve/…).")
    parser.add_argument("--fits", type=str, default="", help="Cesta k MASTERSTAR.fits.")
    parser.add_argument("--csv", type=str, default="", help="Cesta k masterstars CSV.")
    parser.add_argument("--cone", type=str, default="", help="Voliteľne field_catalog_cone.csv.")
    parser.add_argument("--worst", type=int, default=12, help="Koľko najhorších riadkov (0 = vypnúť).")
    args = parser.parse_args()

    fits_path: Path | None = None
    csv_path: Path | None = None
    cone_path: Path | None = None

    if args.archive.strip():
        fits_path, csv_path, cone_path = resolve_paths_from_archive(args.archive.strip())
    if args.fits.strip():
        fits_path = Path(args.fits.strip())
    if args.csv.strip():
        csv_path = Path(args.csv.strip())
    if args.cone.strip():
        cone_path = Path(args.cone.strip())

    if fits_path is None or csv_path is None:
        parser.error("Zadaj --archive alebo oboje --fits a --csv.")

    try:
        print(
            run_masterstar_wcs_dao_diagnostic(
                fits_path,
                csv_path,
                cone_path=cone_path,
                worst_n=int(args.worst),
            )
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
