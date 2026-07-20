"""One-shot: move generate_photometry_report nested block into _PhotometryReportBuilder."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "photometry_report.py"
OUT_BUILDER = ROOT / "photometry_report_builder.py"

# Names assigned at generate top-level (4 spaces) before nested defs / used as closure
STATE_NAMES = """
draft_dir obs_group _var_results _candidates_set _crossmatch_bullets _accepted_periods
_variability_ts _report_draft_lbl _tess_results _report_title platesolve_dir photometry_dir
lc_dir cache_dir summary_csv comp_csv at_csv_primary at_csv_alt active_targets_csv
summary_df comp_df at_df _candidates_norm _use_bprp_primary obs_date_human date_token
output_pdf C_TITLE C_GOOD C_MID C_BAD n_lc med_rms rms_lt_005 avg_good_comp best_rms
worst_rms avg_bp_rp setups fwhm_px aperture_px aavso_dir varastro_dir n_aavso n_varastro
comp_pool_cover_rows PAGE_W PAGE_H M_LEFT M_RIGHT M_TOP M_BOTTOM USE_W USE_H TITLE_H
METRICS_H SEP_H LC_W GAP_W FI_W NOTE_TXT _IMAGE_PDF_SETTINGS FONT_REG FONT_BOLD FONT_OBL
bullets_by_cid candidates
""".split()

RENAME_METHODS = {
    "_draw_cover_sheet": "_report_cover_page",
    "_draw_observation_summary_page": "_report_observation_summary",
    "_draw_qa_page": "_report_fits_qa",
    "_draw_field_astrophysics_pages": "_report_hrd_page",
    "_draw_field_map_full_page": "_report_field_map",
    "draw_star_page": "_report_per_star_page",
    "draw_compact_stars_page": "_report_per_star_compact_page",
    "_draw_variability_hockey_page": "_report_hockey_stick",
    "_draw_variability_candidates_csv_page": "_report_candidates_table",
    "_draw_tess_report_section": "_report_tess_section",
    "_draw_abbreviations_page": "_report_abbreviations",
    "_draw_summary_page": "_report_summary_table",
}

ORCH_RENAME = {
    "_draw_cover_sheet(c)": "self._report_cover_page(c)",
    "_draw_observation_summary_page(c)": "self._report_observation_summary(c)",
    "_draw_qa_page(c)": "self._report_fits_qa(c)",
    "_draw_summary_page(c)": "self._report_summary_table(c)",
    "_draw_field_astrophysics_pages(c)": "self._report_hrd_page(c)",
    "_draw_field_map_full_page(c)": "self._report_field_map(c)",
    "draw_compact_stars_page(c,": "self._report_per_star_compact_page(c,",
    "draw_star_page(c,": "self._report_per_star_page(c,",
    "_draw_variability_hockey_page(c)": "self._report_hockey_stick(c)",
    "_draw_variability_candidates_csv_page(c)": "self._report_candidates_table(c)",
    "_draw_tess_report_section(c)": "self._report_tess_section(c)",
    "_draw_abbreviations_page(c)": "self._report_abbreviations(c)",
}


def main() -> None:
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    gen_start = next(i for i, l in enumerate(lines) if l.startswith("def generate_photometry_report"))
    gen_end = next(
        i for i in range(gen_start + 1, len(lines)) if lines[i].startswith("def ") and not lines[i].startswith("    ")
    )

    header = "".join(lines[:gen_start])
    gen_def_line = lines[gen_start]
    body_lines = lines[gen_start + 1 : gen_end]
    tail = "".join(lines[gen_end:])

    # Split body: imports/setup until first nested def; methods; orchestration
    first_def_i = next(i for i, l in enumerate(body_lines) if re.match(r"    def \w+", l))
    orch_i = next(i for i, l in enumerate(body_lines) if "# Build PDF" in l)

    init_part = body_lines[:first_def_i]
    methods_part = body_lines[first_def_i:orch_i]
    orch_part = body_lines[orch_i:]

    def to_self_assign(line: str) -> str:
        m = re.match(r"^    ([A-Za-z_][\w]*) =", line)
        if m and m.group(1) in STATE_NAMES:
            return line.replace(f"    {m.group(1)} =", f"        self.{m.group(1)} =", 1)
        return line

    init_self = []
    for line in init_part:
        init_self.append(to_self_assign(line))

    methods_text = "".join(methods_part)
    for old, new in RENAME_METHODS.items():
        methods_text = methods_text.replace(f"def {old}(", f"def {new}(self, ")
    # nested calls to renamed methods
    for old, new in RENAME_METHODS.items():
        bare = old.lstrip("_")
        methods_text = re.sub(rf"(?<!\w){re.escape(old)}\(", f"self.{new.split('_report_')[-1] if '_report_' in new else new}(", methods_text)
    # Fix: call other methods with self.
    for name in RENAME_METHODS.values():
        short = name
        methods_text = methods_text.replace(f"({short}(", f"({short}(")  # noop

    # Prefix self. for state reads in methods (careful: not in def lines)
    for name in sorted(STATE_NAMES, key=len, reverse=True):
        if name in ("c", "self"):
            continue
        methods_text = re.sub(
            rf"(?<!self\.)(?<!\.)(?<!\w){re.escape(name)}(?!\w)",
            f"self.{name}",
            methods_text,
        )
    # Fix double self.self.
    methods_text = methods_text.replace("self.self.", "self.")
    # Fix def self.
    methods_text = re.sub(r"def (self\.)+_report_", "def _report_", methods_text)
    methods_text = re.sub(r"def (self\.)+_", "def _", methods_text)

    orch_text = "".join(orch_part)
    for old, new in ORCH_RENAME.items():
        orch_text = orch_text.replace(old, new)
    for name in sorted(STATE_NAMES, key=len, reverse=True):
        orch_text = re.sub(rf"(?<!self\.)(?<!\w){re.escape(name)}(?!\w)", f"self.{name}", orch_text)
    orch_text = orch_text.replace("self.self.", "self.")

    builder = f'''"""PDF report builder - extracted from photometry_report.generate_photometry_report."""
from __future__ import annotations

# Re-export everything the builder body needs from photometry_report module at runtime
# (circular import avoided: photometry_report imports this after module constants are defined)

from photometry_report import *  # noqa: F403,F401

'''
    # Circular import won't work. Embed full imports instead.
    builder = '''"""PDF report builder - extracted from photometry_report (phase 1 split)."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import textwrap
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from decimal import Decimal, InvalidOperation

from gaia_catalog_id import normalize_gaia_source_id

_GAIA_ID_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}


class _PhotometryReportBuilder:
    """Builds VYVAR photometry PDF pages (state + section renderers)."""

    def __init__(
        self,
        *,
        draft_dir: Path,
        obs_group: str,
        output_pdf: Path | None,
        var_results: dict[str, Any] | None,
        candidates: list[str] | None,
        crossmatch_bullets: dict[str, str] | None,
        accepted_periods: dict[str, float] | None,
        variability_timestamp: str | None,
        report_draft_label: str | None,
        tess_results: dict | None,
        report_title: str,
        font_reg: str,
        font_bold: str,
        font_obl: str,
        colors_mod: Any,
        cm_mod: Any,
        mm_mod: Any,
        landscape_mod: Any,
        A4_mod: Any,
    ) -> None:
        self.FONT_REG = font_reg
        self.FONT_BOLD = font_bold
        self.FONT_OBL = font_obl
        self.colors = colors_mod
        self.cm = cm_mod
        self.mm = mm_mod
        self.landscape = landscape_mod
        self.A4 = A4_mod
        self.candidates = candidates
'''
    # Append init with self assignments - need colors alias
    init_fixed = []
    for line in init_self:
        line2 = line.replace("colors.", "self.colors.")
        line2 = line2.replace(" cm", " self.cm")  # bad
        init_fixed.append(line2)
    # Too fragile - abort script approach for manual class

    print("Script needs manual completion; use inline class in photometry_report.py")
    print(f"gen lines: {gen_end - gen_start}, first_def offset: {first_def_i}, orch offset: {orch_i}")


if __name__ == "__main__":
    main()
