"""Vstupný bod pre generovanie gaia_triangles.pkl (2D hash + vertex metadata).

Logika je v ``gaia-dr3_index_solver.py`` (historický názov súboru).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    parent = Path(__file__).resolve().parent
    mod_path = parent / "blind_index_build.py"
    spec = importlib.util.spec_from_file_location("blind_index_build", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nepodarilo sa načítať modul: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import sys

    if len(sys.argv) <= 1:
        mod.build_and_save(
            db_path=str(parent / "vyvar_gaia_dr3.db"),
            output_pkl=str(parent / "gaia_triangles.pkl"),
            mag_limit=16.0,
            cell_deg=1.0,
            stars_per_cell=int(__import__("os").environ.get("BLIND_STARS_PER_CELL", "120")),
        )
        return
    raise SystemExit(mod.main())


if __name__ == "__main__":
    main()
