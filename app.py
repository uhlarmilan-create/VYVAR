"""Thin root shim so ``streamlit run app.py`` keeps working after the src_py/ move.

The real Streamlit application lives at ``src_py/app.py`` together with every other VYVAR
module. This shim puts ``src_py`` on ``sys.path`` (so the app's flat imports resolve) and
executes the real app as ``__main__`` in place, exactly as before the layout change.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src_py"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

runpy.run_path(str(_SRC / "app.py"), run_name="__main__")
