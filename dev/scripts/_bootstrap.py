"""Shared sys.path bootstrap for dev/scripts/*.py run as standalone processes.

After the repo reorg the VYVAR modules live under ``src_py/`` and the dev-side namespace
packages (tests, scripts, tools, validation) under ``dev/``. A script launched as
``python dev/scripts/foo.py`` only gets its own directory on ``sys.path``, so importing
this module (after putting that directory on the path) makes both flat imports
(``from config import ...``) and package imports (``from tests.photometry_sha import ...``)
resolve exactly as they did before the move.

Usage at the top of a script::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: E402,F401
    _ROOT = _bootstrap.REPO_ROOT
"""
from __future__ import annotations

import sys
from pathlib import Path

# dev/scripts/_bootstrap.py -> repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]

for _p in (REPO_ROOT / "src_py", REPO_ROOT / "dev" / "scripts", REPO_ROOT / "dev", REPO_ROOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
