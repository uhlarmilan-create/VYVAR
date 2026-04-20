"""Vstupný bod pre generovanie gaia_triangles.pkl (2D hash + vertex metadata).

Logika je v ``gaia-dr3_index_solver.py`` (historický názov súboru).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    mod_path = Path(__file__).resolve().parent / "gaia-dr3_index_solver.py"
    spec = importlib.util.spec_from_file_location("gaia_dr3_index_solver", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nepodarilo sa načítať modul: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_blind_index()


if __name__ == "__main__":
    main()
