"""Family-local helpers shared by dao_gaia_stage_01 generations.

Not imported by the production pipeline (night_run / pipeline / app).
Do not merge helpers from this family into global homes (CONSOLIDATE-01C R4).
"""
from __future__ import annotations

CORNER_MARGIN_PX = 120.0


def _is_corner(x: float, y: float, wpx: int, h: int) -> bool:
    m = float(CORNER_MARGIN_PX)
    return x < m or y < m or x >= float(wpx) - m or y >= float(h) - m
