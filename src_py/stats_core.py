"""Shared scalar stats helpers (flux-to-mag, MAD variants live at call sites).

Bodies here are the bit-identical survivors of CONSOLIDATE-01C C-2 merges.
Do not add a second MAD flavour under the same name.
"""
from __future__ import annotations

import math


def _flux_to_mag(flux: float) -> float:
    """Instrumental magnitude from flux; non-finite or non-positive flux returns nan."""
    if not math.isfinite(flux) or flux <= 0:
        return float("nan")
    return -2.5 * math.log10(flux)
