"""One-shot: prefix closure locals with self. in _PhotometryReportBuilder methods."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "photometry_report.py"

lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)

# __init__ fixes (0-based line indices)
if len(lines) > 284:
    lines[253] = "                self.fwhm_px = float(_load_fwhm(self.platesolve_dir / \"MASTERSTAR.fits\"))\n"
    lines[256] = (
        "        self.aperture_px = float(np.nanmedian(pd.to_numeric(self.summary_df.get(\"aperture_px\"), "
        "errors=\"coerce\"))) if self.n_lc else float(\"nan\")\n"
    )
    lines[260] = (
        "        self.n_aavso = len(list(self.aavso_dir.glob(\"*.txt\"))) if self.aavso_dir.is_dir() else 0\n"
    )
    lines[261] = (
        "        self.n_varastro = len(list(self.varastro_dir.glob(\"*.txt\"))) if self.varastro_dir.is_dir() else 0\n"
    )
    lines[284] = "        self.FI_W = self.USE_W - self.LC_W - self.GAP_W\n"

start = next(i for i, l in enumerate(lines) if l.startswith("    def _vsx_type_sort_rank"))
end = next(i for i, l in enumerate(lines) if l.startswith("def generate_photometry_report"))

TOKENS = [
    ("PAGE_W", "self.PAGE_W"),
    ("PAGE_H", "self.PAGE_H"),
    ("M_LEFT", "self.M_LEFT"),
    ("M_RIGHT", "self.M_RIGHT"),
    ("M_TOP", "self.M_TOP"),
    ("M_BOTTOM", "self.M_BOTTOM"),
    ("USE_W", "self.USE_W"),
    ("USE_H", "self.USE_H"),
    ("FONT_REG", "self.FONT_REG"),
    ("FONT_BOLD", "self.FONT_BOLD"),
    ("FONT_OBL", "self.FONT_OBL"),
    ("C_TITLE", "self.C_TITLE"),
    ("C_GOOD", "self.C_GOOD"),
    ("C_MID", "self.C_MID"),
    ("C_BAD", "self.C_BAD"),
    ("LC_W", "self.LC_W"),
    ("GAP_W", "self.GAP_W"),
    ("FI_W", "self.FI_W"),
]
NAMES = [
    "obs_date_human",
    "draft_dir",
    "obs_group",
    "_report_title",
    "_report_draft_lbl",
    "n_lc",
    "med_rms",
    "rms_lt_005",
    "avg_good_comp",
    "best_rms",
    "worst_rms",
    "avg_bp_rp",
    "setups",
    "fwhm_px",
    "aperture_px",
    "n_aavso",
    "n_varastro",
    "photometry_dir",
    "platesolve_dir",
    "summary_df",
    "comp_df",
    "cache_dir",
    "lc_dir",
    "comp_pool_cover_rows",
    "bullets_by_cid",
    "_candidates_set",
]

for i in range(start, end):
    s = lines[i]
    if s.strip().startswith("def "):
        continue
    for old, new in TOKENS:
        s = re.sub(rf"(?<!self\.){re.escape(old)}(?!\w)", new, s)
    for name in sorted(NAMES, key=len, reverse=True):
        s = re.sub(rf"(?<!self\.){re.escape(name)}(?!\w)", f"self.{name}", s)
    s = re.sub(r"(?<!self\.)(?<!\w)cm(?!\w)", "self.cm", s)
    s = re.sub(r"(?<!self\.)(?<!\w)mm(?!\w)", "self.mm", s)
    s = re.sub(r"(?<!self\.)colors\.", "self.colors.", s)
    s = s.replace("self.self.", "self.")
    lines[i] = s

for i, l in enumerate(lines):
    if "ms_fits = platesolve_dir" in l:
        lines[i] = l.replace("platesolve_dir", "self.platesolve_dir")

SRC.write_text("".join(lines), encoding="utf-8")
print(f"patched method lines {start}-{end}")
