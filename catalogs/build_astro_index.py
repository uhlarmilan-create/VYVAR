"""build_astro_index.py - blind plate-solve index from the local Gaia DR3 DB (PLATESOLVE-2026-10).

Thin entry for ``astro_index.build_index`` (src_py/astro_index.py): multi-scale 4-star pattern index
(astrometry.net style, Lang et al. 2010), ~7 GB for a full-sky DB. astrosolve uses it only when a
FITS header has no usable position; hinted solving works without it. Build the Gaia DB first
(catalogs/build_gaia_catalog.py).

Defaults follow config.json (Settings -> Paths and catalogs): the DB is ``gaia_db_path``, the
index folder ``astrometry_index_dir`` or, when that is empty, ``vyvar_astro_index`` next to the DB.

Examples (from the VYVAR folder):
  python catalogs/build_astro_index.py
  python catalogs/build_astro_index.py --db D:/data/catalogs/vyvar_gaia_dr3.db --out D:/data/catalogs/vyvar_astro_index
  python catalogs/build_astro_index.py --dec 80 90          # quick test build of a polar cap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _find_src(start: Path) -> tuple[Path, Path]:
    """(install root, module dir holding astro_index) - src_py of a checkout or of a release bundle."""
    for candidate in (start, *start.parents):
        src = candidate / "src_py"
        if (src / "astro_index.py").is_file() or any(src.glob("astro_index*.pyd")) or any(src.glob("astro_index*.so")):
            return candidate, src
    raise SystemExit("build_astro_index.py needs src_py/astro_index from the VYVAR install; run it from the install.")


_ROOT, _SRC = _find_src(_HERE)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _defaults() -> tuple[Path, Path]:
    from astrosolve import default_index_dir  # noqa: PLC0415
    from catalog_db import GAIA_DB_DEFAULT, configured_catalog_path  # noqa: PLC0415

    db = configured_catalog_path("gaia_db_path", GAIA_DB_DEFAULT)
    idx = configured_catalog_path("astrometry_index_dir", "")
    return db, Path(default_index_dir(str(db), str(idx) if str(idx) not in ("", ".") else None))


def main(argv: list[str] | None = None) -> int:
    db_default, out_default = _defaults()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(db_default), help="local Gaia DR3 SQLite (default: %(default)s)")
    ap.add_argument("--out", default=str(out_default), help="index folder (default: %(default)s)")
    ap.add_argument("--ra", nargs=2, type=float, default=None, help="RA range (test builds)")
    ap.add_argument("--dec", nargs=2, type=float, default=(-90.0, 90.0), help="Dec range (test builds)")
    a = ap.parse_args(argv)
    if not Path(a.db).is_file():
        print(f"ERROR: Gaia DB not found: {a.db} (build it with catalogs/build_gaia_catalog.py or pass --db)")
        return 2
    from astro_index import build_index  # noqa: PLC0415

    build_index(a.db, a.out, ra_range=tuple(a.ra) if a.ra else None, dec_range=tuple(a.dec))
    print(f"OK: index written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
