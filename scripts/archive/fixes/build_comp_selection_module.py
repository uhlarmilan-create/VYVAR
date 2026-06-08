"""One-shot builder: extract select_comparison_stars_per_target helpers."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "photometry_core.py"
lines = PC.read_text(encoding="utf-8").splitlines(keepends=True)


def slice_lines(a: int, b: int) -> str:
    """Inclusive 1-based line numbers."""
    return "".join(lines[a - 1 : b])


HEADER = '''"""Per-target comparison star selection (extracted from photometry_core).

CQ-3 refactor: helpers for ``select_comparison_stars_per_target``.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import AbstractSet, Any, Callable

import numpy as np
import pandas as pd

from config import AppConfig
from gaia_catalog_id import normalize_gaia_source_id
from infolog import log_event
from photometry_core import (
    _BPRP_VALID_MAX,
    _BPRP_VALID_MIN,
    _PHASE_USECOLS_PERFRAME,
    _angular_distance_deg,
    _bool_col,
    _enrich_comp_bv,
    _is_catalog_only,
    _normalize_gaia_id,
    _normalize_id_series,
    _normalize_id_value,
    _select_comps_tiered,
    _warn_zero_compstars_edge,
    bp_rp_to_bv,
    bv_to_bprp_linear,
    lookup_bv_from_local_db,
    teff_to_bv,
)

LOGGER = logging.getLogger(__name__)

'''

# Section extractions (1-based inclusive)
sections = {
    "resolve": (7599, 7831),  # through _individual_tier def end - actually need through 7831
    "adaptive": (7833, 7884),
    "spatial": (7906, 8143),  # through early return - but 8131-8143 is early return in orchestrator
    "bootstrap": (8202, 8252),
    "accumulate": (8254, 8406),
    "hard_filters": (8410, 8498),
    "contamination": (8500, 8583),
    "detrend_rms": (8585, 8677),
    "mad": (8727, 8765),
    "score": (8767, 8864),
    "tiers": (8866, 9145),  # through best_tier
    "assemble": (9147, 9352),
}

# Read exact slices for manual wrapping - script writes raw extracted blocks to inspect
out = ROOT / "comp_selection_per_target.py"
parts: list[str] = [HEADER]

# We'll build manually in the script output file using transformations
print("Slices:")
for k, (a, b) in sections.items():
    print(k, b - a + 1, "lines")
