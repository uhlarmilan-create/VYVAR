"""Family-local helpers shared by dao_gaia_stage_01 generations.

Not imported by the production pipeline (night_run / pipeline / app).
Do not merge helpers from this family into global homes (CONSOLIDATE-01C R4).
"""
from __future__ import annotations

import math

import numpy as np
from astropy.io import fits

CORNER_MARGIN_PX = 120.0


def _is_corner(x: float, y: float, wpx: int, h: int) -> bool:
    m = float(CORNER_MARGIN_PX)
    return x < m or y < m or x >= float(wpx) - m or y >= float(h) - m


def _peak_at(data0: np.ndarray, x: float, y: float, r: int = 3) -> float:
    h, w = data0.shape
    ix, iy = int(round(x)), int(round(y))
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)
    patch = data0[y0:y1, x0:x1]
    return float(np.max(patch)) if patch.size else float("nan")


def _saturation_limit(hdr: fits.Header) -> float:
    for key in ("SATURATE", "VY_SATURATE", "HISTCUTLO"):
        if key in hdr:
            try:
                v = float(hdr[key])
                if math.isfinite(v) and v > 0:
                    return v
            except (TypeError, ValueError):
                pass
    return 60000.0
