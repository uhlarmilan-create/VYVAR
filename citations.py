"""VYVAR run citation emitter — single source: CITATIONS.bib."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from config import AppConfig
except ImportError:  # pragma: no cover
    AppConfig = Any  # type: ignore[misc, assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_BIB = _PROJECT_ROOT / "CITATIONS.bib"

_ENTRY_START = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
_FIELD = re.compile(
    r"^\s*([a-zA-Z_][\w-]*)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"(?:[^\"\\]|\\.)*\"|[^,\n]+)\s*,?\s*$",
    re.MULTILINE,
)


def _strip_bib_value(raw: str) -> str:
    s = str(raw or "").strip().rstrip(",")
    if len(s) >= 2 and s[0] == "{" and s[-1] == "}" or len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return (
        s.replace(r"\&", "&")
        .replace(r"\%", "%")
        .replace(r"{\'a}", "á")
        .replace(r"{\'e}", "é")
        .replace(r"{\'i}", "í")
        .replace(r"{\'o}", "ó")
        .replace(r"{\'u}", "ú")
        .replace(r"{\'n}", "ń")
        .replace(r"{\'c}", "ć")
        .replace(r"{\'s}", "ś")
        .replace(r"{\'z}", "ź")
        .replace(r"{\'l}", "ł")
        .replace(r"{\'y}", "ý")
        .replace(r"{\'r}", "ř")
        .replace(r"{\'A}", "Á")
        .replace(r"{\'E}", "É")
        .replace(r"{\'I}", "Í")
        .replace(r"{\'O}", "Ó")
        .replace(r"{\'U}", "Ú")
        .replace(r"{\'N}", "Ń")
        .replace(r"{\'C}", "Ć")
        .replace(r"{\'S}", "Ś")
        .replace(r"{\'Z}", "Ź")
        .replace(r"{\'L}", "Ł")
        .replace(r"{\'Y}", "Ý")
        .replace(r"{\'R}", "Ř")
        .replace(r"\H{o}", "ő")
        .replace(r"\H{O}", "Ő")
        .replace(r"\H{u}", "ű")
        .replace(r"\H{U}", "Ű")
        .replace(r"{\'a}", "á")
        .replace("{", "")
        .replace("}", "")
    )


@lru_cache(maxsize=4)
def load_citations_bib(path: str | None = None) -> dict[str, dict[str, str]]:
    bib_path = Path(path) if path else _DEFAULT_BIB
    if not bib_path.is_file():
        logging.warning("[CITATIONS] Missing %s", bib_path)
        return {}
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    entries: dict[str, dict[str, str]] = {}
    for m in _ENTRY_START.finditer(text):
        key = m.group(1).strip()
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start : i - 1]
        fields: dict[str, str] = {}
        for fm in _FIELD.finditer(body):
            fk = fm.group(1).strip().lower()
            fv = _strip_bib_value(fm.group(2))
            fields[fk] = fv
        entries[key] = fields
    return entries


def citation_line(key: str, *, bib: dict[str, dict[str, str]] | None = None) -> str:
    db = bib if bib is not None else load_citations_bib()
    entry = db.get(key, {})
    export = str(entry.get("export", "") or "").strip()
    if export:
        return export
    author = str(entry.get("author", "Unknown")).split(" and ")[0]
    year = str(entry.get("year", "?"))
    journal = str(entry.get("journal", entry.get("title", "")))
    return f"{author} ({year}) — {journal}"


@dataclass
class RunCitationContext:
    """Flags for methods actually used in a photometry run."""

    use_vsx: bool = False
    use_psf: bool = False
    use_color_tiers: bool = False
    use_temporal_binning: bool = False
    use_pytics: bool = False
    use_sysrem: bool = False
    use_savgol: bool = False
    use_democratic: bool = False
    use_gs11: bool = False
    use_period_analysis: bool = False
    use_variability_envelope: bool = False
    use_lightkurve: bool = False
    use_comp_qa: bool = True
    use_trust: bool = True
    use_common_mode_stability_detrend: bool = False
    use_iterative_comp_clip: bool = False
    use_catalog_recovery_verify: bool = False


def _vsx_db_configured(cfg: AppConfig | None) -> bool:
    if cfg is None:
        return False
    p = str(getattr(cfg, "vsx_local_db_path", "") or "").strip()
    return bool(p) and Path(p).is_file()


def _targets_use_vsx_names(targets_df: Any) -> bool:
    if targets_df is None:
        return False
    try:
        if getattr(targets_df, "empty", True):
            return False
    except Exception:  # noqa: BLE001
        return False
    col = "vsx_name" if "vsx_name" in targets_df.columns else None
    if col is None:
        return False
    for val in targets_df[col].astype(str):
        name = str(val or "").strip()
        if not name:
            continue
        low = name.lower()
        if not low.startswith("gaia") and not low.startswith("ztf j") and "dr3" not in low[:12]:
            return True
    return False


def _lc_method_implies_psf(method: str | None) -> bool:
    m = str(method or "").strip().lower()
    return m in ("psf", "adaptive")


def build_run_citation_context(
    cfg: AppConfig | None = None,
    *,
    pipeline_meta: dict[str, Any] | None = None,
    targets_df: Any = None,
    lc_method: str | None = None,
    period_analysis: bool = False,
    variability_detection: bool = False,
    tess_used: bool = False,
) -> RunCitationContext:
    meta = pipeline_meta if isinstance(pipeline_meta, dict) else {}
    gs11_meta = meta.get("gs11_summary") if isinstance(meta.get("gs11_summary"), dict) else {}

    use_psf = _lc_method_implies_psf(lc_method)
    if cfg is not None:
        use_psf = use_psf or bool(getattr(cfg, "psf_photometry_enabled", False))
        use_psf = use_psf or bool(getattr(cfg, "psf_adaptive_enabled", False))

    use_vsx = _targets_use_vsx_names(targets_df) or _vsx_db_configured(cfg)
    use_color = True  # colour-term / citations use Gaia BP-RP only (Johnson B-V retired)
    use_bin = bool(getattr(cfg, "temporal_binning_enabled", True)) if cfg else True
    use_pytics = bool(getattr(cfg, "pytics_enabled", True)) if cfg else True
    use_sysrem = bool(getattr(cfg, "sysrem_enabled", False)) if cfg else False
    use_savgol = bool(getattr(cfg, "savgol_detrend_enabled", False)) if cfg else False
    use_dem = bool(getattr(cfg, "democratic_detrend_enabled", False)) if cfg else False
    use_gs11 = bool(getattr(cfg, "gs11_dilution_enabled", False)) if cfg else False
    use_gs11 = use_gs11 or bool(gs11_meta.get("enabled", False))
    use_tess = bool(tess_used) or bool(getattr(cfg, "tess_enabled", False)) if cfg else bool(tess_used)
    use_comp_qa = bool(getattr(cfg, "comp_qa_enabled", True)) if cfg else True
    use_trust = bool(getattr(cfg, "trust_flag_enabled", True)) if cfg else True
    use_cm_detrend = bool(meta.get("common_mode_stability_detrend", False)) if meta else False
    use_iter_clip = False
    if cfg:
        from config import resolve_comp_sparse_fallback_enabled  # noqa: PLC0415

        use_iter_clip = resolve_comp_sparse_fallback_enabled(cfg)
    use_iter_clip = use_iter_clip or bool(
        int(meta.get("comp_sparse_fallback_target_count", 0) or 0) > 0
        or meta.get("comp_sparse_fallback_used", meta.get("comp_iterative_clip_used", False))
    )
    use_catalog_recovery = bool(meta.get("masterstar_catalog_recovery_verified", False))
    if not use_catalog_recovery:
        _ast = meta.get("astrometry") if isinstance(meta.get("astrometry"), dict) else {}
        use_catalog_recovery = bool(_ast.get("catalog_recovery_verified", False))

    return RunCitationContext(
        use_vsx=use_vsx,
        use_psf=use_psf,
        use_color_tiers=use_color,
        use_temporal_binning=use_bin,
        use_pytics=use_pytics,
        use_sysrem=use_sysrem,
        use_savgol=use_savgol,
        use_democratic=use_dem,
        use_gs11=use_gs11,
        use_period_analysis=bool(period_analysis),
        use_variability_envelope=bool(variability_detection),
        use_lightkurve=use_tess,
        use_comp_qa=use_comp_qa,
        use_trust=use_trust,
        use_common_mode_stability_detrend=use_cm_detrend,
        use_iterative_comp_clip=use_iter_clip,
        use_catalog_recovery_verify=use_catalog_recovery,
    )


def load_pipeline_meta(photometry_dir: Path | str | None) -> dict[str, Any]:
    if photometry_dir is None:
        return {}
    p = Path(photometry_dir) / "pipeline_meta.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _sections_for_context(ctx: RunCitationContext) -> list[tuple[str, list[str]]]:
    bib = load_citations_bib()
    sections: list[tuple[str, list[str]]] = []

    core = [
        "broeg2005",
        "collins2017",
        "howell1989",
        "stetson1987",
        "henden_kaitchuck1982",
        "aavso_ccd_guide",
        "eastman2010",
    ]
    sections.append(("CORE", [citation_line(k, bib=bib) for k in core]))

    catalogs = ["gaia2023", "lindegren2021"]
    if ctx.use_vsx:
        catalogs.append("watson2006")
    sections.append(("CATALOGS & TIME", [citation_line(k, bib=bib) for k in catalogs]))

    software = ["photutils", "astropy2022", "astroquery2019", "numpy2020", "scipy2020"]
    if ctx.use_lightkurve:
        software.append("lightkurve2018")
    sections.append(("SOFTWARE", [citation_line(k, bib=bib) for k in software]))

    optional: list[str] = []
    if ctx.use_color_tiers:
        optional.append(citation_line("riello2021", bib=bib))
    if ctx.use_temporal_binning:
        optional.append(citation_line("pont2006", bib=bib))
    if ctx.use_pytics:
        optional.append(citation_line("marconi2026", bib=bib))
    if ctx.use_sysrem:
        optional.append(citation_line("tamuz2005", bib=bib))
    if ctx.use_savgol:
        optional.extend(
            [
                citation_line("savitzky1964", bib=bib),
                citation_line("aigrain2004", bib=bib),
            ]
        )
    if ctx.use_democratic:
        optional.append(citation_line("hippke2024", bib=bib))
    if ctx.use_common_mode_stability_detrend:
        optional.append(citation_line("honeycutt1992", bib=bib))
    if ctx.use_iterative_comp_clip:
        optional.extend(
            [
                citation_line("gilliland1988", bib=bib),
                citation_line("burdanov2014", bib=bib),
                citation_line("everett2001", bib=bib),
            ]
        )
    if optional:
        sections.append(("METHODS — this run", optional))

    if ctx.use_psf:
        sections.append(
            (
                "PSF PHOTOMETRY",
                [
                    citation_line("stetson1987", bib=bib),
                    citation_line("anderson2000", bib=bib),
                    citation_line("moffat1969", bib=bib),
                    citation_line("astier2013", bib=bib),
                    citation_line("lacroix2025", bib=bib),
                    citation_line("guy2010", bib=bib),
                    citation_line("mighell1999", bib=bib),
                ],
            )
        )

    if ctx.use_gs11:
        sections.append(
            (
                "FLUX DILUTION (GS11)",
                [
                    citation_line("seager2003", bib=bib),
                    citation_line("ciardi2015", bib=bib),
                ],
            )
        )

    if ctx.use_period_analysis:
        sections.append(
            (
                "PERIOD ANALYSIS",
                [
                    citation_line("lomb1976", bib=bib),
                    citation_line("scargle1982", bib=bib),
                    citation_line("vanderplas2018", bib=bib),
                    citation_line("kovacs2002", bib=bib),
                    citation_line("stellingwerf1978", bib=bib),
                ],
            )
        )

    if ctx.use_variability_envelope and not ctx.use_savgol:
        sections.append(
            (
                "VARIABILITY DETECTION",
                [citation_line("aigrain2004", bib=bib)],
            )
        )

    dq: list[str] = []
    if ctx.use_comp_qa or ctx.use_trust:
        dq.extend(
            [
                citation_line("sokolovsky2017", bib=bib),
                citation_line("vonneumann1941", bib=bib),
            ]
        )
    if dq:
        sections.append(("DATA-QUALITY GATE", dq))

    if ctx.use_catalog_recovery_verify:
        sections.append(
            (
                "ASTROMETRY VERIFICATION",
                [
                    citation_line("lang2010", bib=bib),
                    citation_line("gaia2023", bib=bib),
                ],
            )
        )

    return sections


def emit_export_citation_lines(
    ctx: RunCitationContext,
    *,
    comment_prefix: str = "#",
) -> list[str]:
    """AAVSO / VarAstro comment header block."""
    p = comment_prefix
    lines: list[str] = [
        f"{p}\n",
        f"{p} Pipeline: VYVAR — Automated Differential Photometry Pipeline\n",
        f"{p}\n",
        f"{p} ALGORITHMS & REFERENCES (from CITATIONS.bib):\n",
        f"{p}\n",
    ]
    for title, items in _sections_for_context(ctx):
        lines.append(f"{p} [{title}]\n")
        for item in items:
            lines.append(f"{p}   {item}\n")
        lines.append(f"{p}\n")
    return lines


def emit_pdf_methods_sections(ctx: RunCitationContext) -> list[tuple[str, list[str]]]:
    """PDF Methods section: (section_title, bullet_lines)."""
    out: list[tuple[str, list[str]]] = []
    for title, items in _sections_for_context(ctx):
        out.append((title.replace(" — ", " — ").title(), [f"  {it}" for it in items]))
    return out


def emit_varastro_method_summary_lines(ctx: RunCitationContext) -> list[str]:
    """Short per-run algorithm lines for VarAstro body (non-comment)."""
    bib = load_citations_bib()
    lines: list[str] = [
        "# PHOTOMETRY: Comp selection + 1/rms^2 zeropoint | "
        f"{citation_line('broeg2005', bib=bib).strip()}\n",
        "# PHOTOMETRY: Ensemble flux-sum combination | "
        f"{citation_line('collins2017', bib=bib).strip()}; "
        f"{citation_line('honeycutt1992', bib=bib).strip()}\n",
    ]
    if ctx.use_temporal_binning:
        lines.append(f"# ALG: Temporal binning — {citation_line('pont2006', bib=bib)}\n")
    if ctx.use_pytics:
        lines.append(f"# ALG: PyTICS — {citation_line('marconi2026', bib=bib)}\n")
    if ctx.use_sysrem:
        lines.append(f"# ALG: SysRem — {citation_line('tamuz2005', bib=bib)}\n")
    if ctx.use_savgol:
        lines.append(f"# ALG: Savitzky-Golay — {citation_line('savitzky1964', bib=bib)}\n")
    if ctx.use_democratic:
        lines.append(f"# ALG: Democratic Detrender — {citation_line('hippke2024', bib=bib)}\n")
    if ctx.use_psf:
        lines.append(f"# ALG: ePSF — {citation_line('anderson2000', bib=bib)}\n")
    return lines
