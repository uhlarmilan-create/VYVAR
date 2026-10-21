"""Single source of truth for the VYVAR release version string.

Keep ``__version__`` identical to ``[project].version`` in ``pyproject.toml``.
Export headers and UI copy should import from here (not hard-code a second string).
"""

from __future__ import annotations

__version__ = "0.10.0a1"

# Human-facing label for AAVSO #SOFTWARE / VarAstro Software lines.
VYVAR_SOFTWARE_VERSION = f"VYVAR {__version__}"
