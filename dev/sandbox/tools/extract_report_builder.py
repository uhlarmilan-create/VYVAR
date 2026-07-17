"""Extract generate_photometry_report nested block into _PhotometryReportBuilder class."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "photometry_report.py"

RENAME = {
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

ORCH_CALLS = [
    ("_draw_cover_sheet(c)", "self._report_cover_page(c)"),
    ("_draw_observation_summary_page(c)", "self._report_observation_summary(c)"),
    ("_draw_qa_page(c)", "self._report_fits_qa(c)"),
    ("_draw_summary_page(c)", "self._report_summary_table(c)"),
    ("_draw_field_astrophysics_pages(c)", "self._report_hrd_page(c)"),
    ("_draw_field_map_full_page(c)", "self._report_field_map(c)"),
    ("draw_compact_stars_page(c,", "self._report_per_star_compact_page(c,"),
    ("draw_star_page(c,", "self._report_per_star_page(c,"),
    ("_draw_variability_hockey_page(c)", "self._report_hockey_stick(c)"),
    ("_draw_variability_candidates_csv_page(c)", "self._report_candidates_table(c)"),
    ("_draw_tess_report_section(c)", "self._report_tess_section(c)"),
    ("_draw_abbreviations_page(c)", "self._report_abbreviations(c)"),
]


def is_method_def(line: str) -> bool:
    return bool(re.match(r"    def \w+", line))


def _is_init_line_at_4(line: str) -> bool:
    """Top-level setup inside generate (4 spaces, not a nested def, not method body at 8)."""
    if not line.startswith("    ") or line.startswith("        "):
        return False
    stripped = line[4:].lstrip()
    if not stripped or stripped.startswith("#"):
        return True
    if stripped.startswith("def "):
        return False
    # Multiline def signature closing (still part of the preceding def)
    if stripped.startswith(")") and "->" in stripped:
        return False
    return True


def split_chunks(body: list[str]) -> list[tuple[str, str | None, list[str]]]:
    chunks: list[tuple[str, str | None, list[str]]] = []
    i = 0
    while i < len(body):
        if is_method_def(body[i]):
            name = re.match(r"    def (\w+)", body[i]).group(1)
            j = i + 1
            while j < len(body):
                if is_method_def(body[j]):
                    break
                if _is_init_line_at_4(body[j]):
                    break
                j += 1
            chunks.append(("method", name, body[i:j]))
            i = j
        else:
            j = i + 1
            while j < len(body):
                if is_method_def(body[j]):
                    break
                if _is_init_line_at_4(body[j]) and j > i:
                    # only break on next init segment when we already collected something
                    break
                j += 1
            if any(ln.strip() for ln in body[i:j]):
                chunks.append(("init", None, body[i:j]))
            i = j
    return chunks


def indent_block(lines: list[str], spaces: int) -> list[str]:
    pad = " " * spaces
    return [pad + ln[4:] if ln.startswith("    ") else pad + ln for ln in lines]


def to_self_assign(line: str) -> str:
    m = re.match(r"^    ([A-Za-z_][\w]*) =", line)
    if m and not m.group(1).startswith("self"):
        return line.replace(f"    {m.group(1)} =", f"        self.{m.group(1)} =", 1)
    return line.replace("    ", "        ", 1) if line.startswith("    ") else line


def repl_init_calls(line: str) -> str:
    for old, new in RENAME.items():
        line = line.replace(f"{old}(", f"self.{new}(")
    for fn in (
        "_vsx_type_sort_rank",
        "_try_load_variability_from_csv",
        "_obs_date_str",
        "_build_comp_pool_cover_rows",
    ):
        line = line.replace(f"{fn}(", f"self.{fn}(")
    # map callback
    line = line.replace(".map(_vsx_type_sort_rank)", ".map(self._vsx_type_sort_rank)")
    return line


def repl_method_calls(text: str, method_names: set[str]) -> str:
    for old, new in RENAME.items():
        text = re.sub(rf"(?<!self\.)(?<!\w){re.escape(old)}\(", f"self.{new}(", text)
    for name in sorted(method_names - set(RENAME.keys()), key=len, reverse=True):
        text = re.sub(rf"(?<!self\.)(?<!\w){re.escape(name)}\(", f"self.{name}(", text)
    return text.replace("self.self.", "self.")


def convert_method(name: str, lines: list[str], method_names: set[str]) -> list[str]:
    new_name = RENAME.get(name, name)
    out: list[str] = []
    first = lines[0]
    if f"def {name}(" in first:
        first = first.replace(f"def {name}(", f"def {new_name}(self, ", 1)
    elif f"def {name}(" not in first:
        first = re.sub(rf"def {re.escape(name)}\(", f"def {new_name}(self, ", first, count=1)
    out.append(first)
    body = repl_method_calls("\n".join(lines[1:]), method_names)
    out.extend(body.splitlines())
    return out


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    gen_i = next(i for i, l in enumerate(lines) if l.startswith("def generate_photometry_report"))
    gen_end = len(lines)
    first_def = next(i for i in range(gen_i, gen_end) if is_method_def(lines[i]))
    orch_i = next(i for i in range(gen_i, gen_end) if "# Build PDF" in lines[i])

    # Setup before nested defs: from draft_dir= through bullets_by_cid
    pre_def = lines[gen_i + 45 : first_def]  # platesolve_dir … (draft_dir/params set in __init__ header)
    methods_block = lines[first_def:orch_i]
    orch_block = lines[orch_i:gen_end]

    chunks = split_chunks(methods_block)
    method_names = {c[1] for c in chunks if c[0] == "method" and c[1]}

    init_lines: list[str] = []
    for ln in pre_def:
        init_lines.append(repl_init_calls(to_self_assign(ln)))
    class_methods: list[str] = []
    for kind, name, chunk in chunks:
        if kind == "init":
            for ln in chunk:
                init_lines.append(repl_init_calls(to_self_assign(ln)))
        elif kind == "method" and name:
            class_methods.extend(convert_method(name, chunk, method_names))

    # After self.* assignments, init body should read attrs via self.
    init_body = "\n".join(init_lines)
    for attr in sorted(
        {
            "draft_dir", "obs_group", "platesolve_dir", "photometry_dir", "lc_dir", "cache_dir",
            "summary_csv", "comp_csv", "at_csv_primary", "at_csv_alt", "active_targets_csv",
            "summary_df", "comp_df", "at_df", "_candidates_set", "_candidates_norm",
            "output_pdf", "colors", "cm", "mm", "landscape", "A4",
        },
        key=len,
        reverse=True,
    ):
        init_body = re.sub(rf"(?<!self\.)(?<!\w){re.escape(attr)}(?!\w)", f"self.{attr}", init_body)
    init_body = init_body.replace("self.self.", "self.")
    init_body = init_body.replace(".map(self._vsx_type_sort_rank)", ".map(self._vsx_type_sort_rank)")
    init_lines = init_body.splitlines()

    orch_text = repl_method_calls("\n".join(orch_block), method_names)
    for a, b in ORCH_CALLS:
        orch_text = orch_text.replace(a, b)
    # orch state reads
    state_attrs = [
        "output_pdf", "summary_df", "lc_dir", "platesolve_dir", "photometry_dir", "cache_dir",
        "_candidates_set", "_crossmatch_bullets", "_var_results", "_candidates_norm", "candidates",
    ]
    for attr in sorted(state_attrs, key=len, reverse=True):
        orch_text = re.sub(rf"(?<!self\.)(?<!\w){re.escape(attr)}(?!\w)", f"self.{attr}", orch_text)
    orch_text = orch_text.replace("self.self.", "self.")

    header = lines[:gen_i]
    sig = lines[gen_i : gen_i + 14]

    builder: list[str] = [
        "",
        "class _PhotometryReportBuilder:",
        '    """Internal PDF builder; section renderers extracted from generate_photometry_report."""',
        "",
        "    def __init__(",
        "        self,",
        "        draft_dir: Path,",
        "        obs_group: str,",
        "        output_pdf: Path | None,",
        "        var_results: dict[str, Any] | None,",
        "        candidates: list[str] | None,",
        "        crossmatch_bullets: dict[str, str] | None,",
        "        accepted_periods: dict[str, float] | None,",
        "        variability_timestamp: str | None,",
        "        report_draft_label: str | None,",
        "        tess_results: dict | None,",
        "        report_title: str,",
        "        font_reg: str,",
        "        font_bold: str,",
        "        font_obl: str,",
        "        colors_mod: Any,",
        "        cm_mod: Any,",
        "        mm_mod: Any,",
        "        landscape_fn: Any,",
        "        a4_size: Any,",
        "        canvas_mod: Any,",
        "        image_reader_mod: Any,",
        "        table_mod: Any,",
        "        paragraph_mod: Any,",
        "        paragraph_style_mod: Any,",
        "        ta_left_mod: Any,",
        "    ) -> None:",
        "        from reportlab.lib.units import cm, mm",
        "",
        "        self._colors = colors_mod",
        "        self.cm = cm_mod",
        "        self.mm = mm_mod",
        "        self.landscape = landscape_fn",
        "        self.A4 = a4_size",
        "        self.canvas = canvas_mod",
        "        self.ImageReader = image_reader_mod",
        "        self.Table = table_mod",
        "        self.TableStyle = table_style_mod",
        "        self.Paragraph = paragraph_mod",
        "        self.ParagraphStyle = paragraph_style_mod",
        "        self.TA_LEFT = ta_left_mod",
        "        colors = self._colors",
        "        cm = self.cm",
        "        mm = self.mm",
        "        landscape = self.landscape",
        "        A4 = self.A4",
        "        canvas = self.canvas",
        "        ImageReader = self.ImageReader",
        "        Table = self.Table",
        "        TableStyle = self.TableStyle",
        "        Paragraph = self.Paragraph",
        "        ParagraphStyle = self.ParagraphStyle",
        "        TA_LEFT = self.TA_LEFT",
        "        self.FONT_REG = font_reg",
        "        self.FONT_BOLD = font_bold",
        "        self.FONT_OBL = font_obl",
        "        self.candidates = candidates",
        "        self._var_results = var_results",
        "        self._crossmatch_bullets = dict(crossmatch_bullets or {})",
        "        self._accepted_periods = dict(accepted_periods or {})",
        "        self._variability_ts = str(variability_timestamp or '').strip()",
        "        self._report_draft_lbl = str(report_draft_label or '').strip() or str(Path(draft_dir).name)",
        "        self._tess_results = dict(tess_results or {})",
        "        self._report_title = str(report_title or 'VYVAR \\u2014 Summary Measure Report')",
        "        self._candidates_set = {str(x).strip() for x in (candidates or []) if str(x).strip()}",
    ]
    builder.extend(init_lines)
    builder.append("")
    builder.extend(class_methods)
    builder.append("")
    builder.append("    def build_pdf(self) -> Path:")
    builder.append("        from reportlab.lib.pagesizes import landscape, A4")
    builder.append("        from reportlab.pdfgen import canvas")
    builder.append("")
    for ln in orch_text.splitlines():
        if not ln.strip():
            builder.append("")
        elif ln.startswith("    "):
            builder.append("    " + ln)  # 4 -> 8 spaces inside build_pdf
        else:
            builder.append("        " + ln)
    builder.append("        return self.output_pdf")
    builder.append("")
    # drop duplicate return if present
    out_joined = "\n".join(builder)
    out_joined = out_joined.replace(
        "        return self.output_pdf\n        return self.output_pdf\n",
        "        return self.output_pdf\n",
    )
    builder = out_joined.splitlines()

    new_gen = [
        sig[0],
        sig[1],
        sig[2],
        sig[3],
        sig[4],
        sig[5],
        sig[6],
        sig[7],
        sig[8],
        sig[9],
        sig[10],
        sig[11],
        sig[12],
        sig[13],
        '    """',
        "    Build a PDF photometry report for one observation night.",
        "",
        "    Returns the path to the written PDF, or None if reportlab is not installed.",
        '    """',
        "    try:",
        "        from reportlab.lib import colors",
        "        from reportlab.lib.pagesizes import A4, landscape",
        "        from reportlab.lib.units import cm, mm",
        "        from reportlab.lib.utils import ImageReader",
        "        from reportlab.pdfgen import canvas",
        "        from reportlab.lib.enums import TA_LEFT",
        "        from reportlab.lib.styles import ParagraphStyle",
        "        from reportlab.platypus import Paragraph, Table, TableStyle",
        "    except Exception as exc:  # noqa: BLE001",
        '        logging.warning("reportlab is not installed, skipping PDF (%s)", exc)',
        "        return None",
        "",
        "    FONT_REG, FONT_BOLD, FONT_OBL = _register_pdf_unicode_fonts()",
        "",
        "    builder = _PhotometryReportBuilder(",
        "        draft_dir=Path(draft_dir),",
        "        obs_group=str(obs_group),",
        "        output_pdf=output_pdf,",
        "        var_results=var_results,",
        "        candidates=candidates,",
        "        crossmatch_bullets=crossmatch_bullets,",
        "        accepted_periods=accepted_periods,",
        "        variability_timestamp=variability_timestamp,",
        "        report_draft_label=report_draft_label,",
        "        tess_results=tess_results,",
        "        report_title=str(report_title or 'VYVAR \\u2014 Summary Measure Report'),",
        "        font_reg=FONT_REG,",
        "        font_bold=FONT_BOLD,",
        "        font_obl=FONT_OBL,",
        "        colors_mod=colors,",
        "        cm_mod=cm,",
        "        mm_mod=mm,",
        "        landscape_fn=landscape,",
        "        a4_size=A4,",
        "        canvas_mod=canvas,",
        "        image_reader_mod=ImageReader,",
        "        table_mod=Table,",
        "        table_style_mod=TableStyle,",
        "        paragraph_mod=Paragraph,",
        "        paragraph_style_mod=ParagraphStyle,",
        "        ta_left_mod=TA_LEFT,",
        "    )",
        "    return builder.build_pdf()",
        "",
    ]

    out = header + builder + new_gen
    out_path = ROOT / "photometry_report.py.new"
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(out)} lines), methods={len(method_names)}, init_lines={len(init_lines)}")


if __name__ == "__main__":
    main()
