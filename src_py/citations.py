"""VYVAR run citation emitter - single source: CITATIONS.bib."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from invariants_runtime import load_pipeline_meta  # noqa: F401  # re-export

try:
    from config import AppConfig
except ImportError:  # pragma: no cover
    AppConfig = Any  # type: ignore[misc, assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src_py -> repo root (CITATIONS.bib)
_DEFAULT_BIB = _PROJECT_ROOT / "CITATIONS.bib"

_ENTRY_START = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
_FIELD = re.compile(
    r"^\s*([a-zA-Z_][\w-]*)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"(?:[^\"\\]|\\.)*\"|[^,\n]+)\s*,?\s*$",
    re.MULTILINE,
)


def plain_ascii_citation_text(text: str) -> str:
    """Render bib export notes to plain ASCII (no LaTeX leaks in headers)."""
    s = str(text or "")
    for src, dst in (
        (r"\ensuremath{\Delta}", "Delta"),
        (r"\ensuremath\Delta", "Delta"),
        (r"\Delta", "Delta"),
        (r"\_", "_"),
        (r"\&", "&"),
        (r"\%", "%"),
        (r"\-", "-"),
        ("\u2014", "-"),  # em dash
        ("\u2013", "-"),  # en dash
        ("\u00b4", "'"),
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
    ):
        s = s.replace(src, dst)
    return s.replace("\\", "")


def _strip_bib_value(raw: str) -> str:
    s = str(raw or "").strip().rstrip(",")
    if len(s) >= 2 and s[0] == "{" and s[-1] == "}" or len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1]
    return plain_ascii_citation_text(
        s.replace(r"{\'a}", "a")
        .replace(r"{\'e}", "e")
        .replace(r"{\'i}", "i")
        .replace(r"{\'o}", "o")
        .replace(r"{\'u}", "u")
        .replace(r"{\'n}", "n")
        .replace(r"{\'c}", "c")
        .replace(r"{\'s}", "s")
        .replace(r"{\'z}", "z")
        .replace(r"{\'l}", "l")
        .replace(r"{\'y}", "y")
        .replace(r"{\'r}", "r")
        .replace(r"{\'A}", "A")
        .replace(r"{\'E}", "E")
        .replace(r"{\'I}", "I")
        .replace(r"{\'O}", "O")
        .replace(r"{\'U}", "U")
        .replace(r"{\'N}", "N")
        .replace(r"{\'C}", "C")
        .replace(r"{\'S}", "S")
        .replace(r"{\'Z}", "Z")
        .replace(r"{\'L}", "L")
        .replace(r"{\'Y}", "Y")
        .replace(r"{\'R}", "R")
        .replace(r"\H{o}", "o")
        .replace(r"\H{O}", "O")
        .replace(r"\H{u}", "u")
        .replace(r"\H{U}", "U")
        .replace(r"{\'a}", "a")
        .replace("{", "")
        .replace("}", "")
    )


@lru_cache(maxsize=4)
def load_citations_bib(path: str | None = None) -> dict[str, dict[str, str]]:
    bib_path = Path(path) if path else _DEFAULT_BIB
    if not bib_path.is_file():
        # The exports and the PDF methods text cite from this file; without it they would carry
        # "Unknown (?)" lines, so a missing bibliography is an installation error.
        raise FileNotFoundError(f"[CITATIONS] bibliography missing: {bib_path}")
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
    if key not in db:
        raise KeyError(f"[CITATIONS] key {key!r} missing from CITATIONS.bib")
    entry = db[key]
    export = str(entry.get("export", "") or "").strip()
    if export:
        return plain_ascii_citation_text(export)
    author = str(entry.get("author", "Unknown")).split(" and ")[0]
    year = str(entry.get("year", "?"))
    journal = str(entry.get("journal", entry.get("title", "")))
    return plain_ascii_citation_text(f"{author} ({year}) - {journal}")


@dataclass
class RunCitationContext:
    """Flags for methods actually used in a photometry run.

    Single source for conditional citations AND the methods ON/OFF matrix.
    """

    use_vsx: bool = False
    use_psf: bool = False
    use_color_tiers: bool = False
    use_gs11: bool = False
    use_period_analysis: bool = False
    use_variability_envelope: bool = False
    use_lightkurve: bool = False
    use_transients: bool = False
    use_common_mode_stability_detrend: bool = False
    use_iterative_comp_clip: bool = False
    use_catalog_recovery_verify: bool = False
    use_k2_literature: bool = False
    use_hrd_extreme: bool = True
    use_ensemble_flux_sum: bool = True
    use_per_frame_sat: bool = False
    use_empirical_background: bool = True
    use_airmass: bool = False
    color_term_mode: str = "off"
    k2_mode: str = "off"
    k2_source_label: str = "none"
    k2_value: float | None = None
    k2_off_reason: str = ""
    osc_channel_export: bool = False
    osc_channel_binning: int | None = None
    osc_transform_citation: str = ""
    # EXPORT-2026-10: facts of the current pipeline shown in the methods matrix.
    zp_band_label: str = ""
    filter_transformable: bool = False
    n_comp_min: int = 3
    n_comp_max: int = 8
    k_grid: tuple[float, ...] = ()
    annulus_fwhm: tuple[float, float] = (2.7, 5.2)
    snr_min: float = 10.0


def _vsx_db_configured(cfg: AppConfig) -> bool:
    p = str(cfg.objects_db_path or "").strip()
    return bool(p) and Path(p).is_file()


def _targets_use_vsx_names(targets_df: Any) -> bool:
    if targets_df is None:
        return False
    try:
        if getattr(targets_df, "empty", True):
            return False
    except Exception:  # noqa: BLE001
        # EXC-0027: T4 -- optional enrichment skipped (if getattr(targets_df, 'empty', True): / return False / ex... (EXCEPT-BULK 2026-07-08)
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


def build_run_citation_context(
    cfg: AppConfig,
    *,
    pipeline_meta: dict[str, Any] | None = None,
    targets_df: Any = None,
    psf_used: bool = False,
    period_analysis: bool = False,
    variability_detection: bool = False,
    tess_used: bool = False,
    obs_group: str = "",
) -> RunCitationContext:
    meta = pipeline_meta if isinstance(pipeline_meta, dict) else {}
    gs11_meta = meta.get("gs11_summary") if isinstance(meta.get("gs11_summary"), dict) else {}

    use_psf = bool(psf_used)  # ePSF light curves in the PDF report (exports are aperture only)

    use_vsx = _targets_use_vsx_names(targets_df) or _vsx_db_configured(cfg)
    use_color = True  # colour-term / citations use Gaia BP-RP only (Johnson B-V retired)
    use_gs11 = bool(cfg.gs11_dilution_enabled)
    use_gs11 = use_gs11 or bool(gs11_meta.get("enabled", False))
    use_tess = bool(tess_used) or bool(cfg.tess_enabled)
    use_cm_detrend = bool(meta.get("common_mode_stability_detrend", False)) if meta else False
    # Sparse fallback / iterative comp clip removed (COMP-RULES-2026-10); older drafts
    # may still carry the pipeline_meta flag.
    use_iter_clip = bool(
        int(meta.get("comp_sparse_fallback_target_count", 0) or 0) > 0
        or meta.get("comp_sparse_fallback_used", meta.get("comp_iterative_clip_used", False))
    )
    use_catalog_recovery = bool(meta.get("masterstar_catalog_recovery_verified", False))
    if not use_catalog_recovery:
        _ast = meta.get("astrometry") if isinstance(meta.get("astrometry"), dict) else {}
        use_catalog_recovery = bool(_ast.get("catalog_recovery_verified", False))
    _k2m = str(cfg.k2_mode or "off").strip().lower()
    use_k2 = _k2m not in ("0", "false", "no", "off", "none")
    use_hrd = bool(cfg.hrd_online_enrich_enabled)
    use_pfs = True  # SAT-LIN-2026-10: always on
    use_emp_bkg = True  # CONSOLIDATE-01D: empirical is the only background-err policy
    _ct = str(cfg.apply_color_term or "off").strip().lower()
    if _ct in ("0", "false", "no"):
        _ct = "off"
    k2_src_label, k2_val = _resolve_k2_matrix_fields(cfg, meta, k2_mode=_k2m, use_k2=use_k2,
                                                     obs_group=str(obs_group or ""))
    # REPORT-AUDIT-2026-10: k2_mode on but no k'' value for this filter (e.g. unfiltered) -> the
    # light curves carry no k'' term; Methods says so instead of "ON"
    k2_off_reason = ""
    if use_k2 and k2_val is None:
        use_k2, k2_off_reason = False, "no literature k'' for this filter"

    from gaia_johnson import TRANSFORM_CITATION
    from osc_align import is_osc_export_eligible_obs_group, parse_osc_channel

    _og = str(obs_group or "").strip()
    _osc_ch = parse_osc_channel(_og)
    _osc_export = is_osc_export_eligible_obs_group(_og) if _osc_ch else False
    _osc_bin = int(getattr(cfg, "osc_channel_binning", 2) or 2) if (cfg and _osc_ch) else None

    from gaia_bands import filter_band_for_obs_group  # noqa: PLC0415

    _fb = filter_band_for_obs_group(_og)
    try:
        from photometry_defs import DEFAULT_APERTURE_F_GRID  # noqa: PLC0415

        _kg = tuple(float(x) for x in DEFAULT_APERTURE_F_GRID)
    except Exception:  # noqa: BLE001
        _kg = ()
    return RunCitationContext(
        zp_band_label=_fb.label if _og else "",
        filter_transformable=bool(_fb.transformable) if _og else False,
        n_comp_min=int(cfg.phase01_comparison_n_comp_min),
        n_comp_max=int(cfg.phase01_comparison_n_comp_max),
        k_grid=_kg,
        annulus_fwhm=(
            float(cfg.annulus_inner_fwhm),
            float(cfg.annulus_outer_fwhm),
        ),
        snr_min=float(cfg.snr_min),
        use_vsx=use_vsx,
        use_psf=use_psf,
        use_color_tiers=use_color,
        use_gs11=use_gs11,
        use_period_analysis=bool(period_analysis),
        use_variability_envelope=bool(variability_detection),
        use_lightkurve=use_tess,
        use_transients=bool(cfg.transient_search_enabled),
        use_common_mode_stability_detrend=use_cm_detrend,
        use_iterative_comp_clip=use_iter_clip,
        use_catalog_recovery_verify=use_catalog_recovery,
        use_k2_literature=use_k2,
        use_hrd_extreme=use_hrd,
        use_ensemble_flux_sum=True,
        use_per_frame_sat=use_pfs,
        use_empirical_background=use_emp_bkg,
        color_term_mode=_ct,
        k2_mode=_k2m if use_k2 else "off",
        k2_source_label=k2_src_label,
        k2_value=k2_val,
        k2_off_reason=k2_off_reason,
        osc_channel_export=_osc_export,
        osc_channel_binning=_osc_bin,
        osc_transform_citation=TRANSFORM_CITATION if _osc_export else "",
    )


def _resolve_k2_matrix_fields(
    cfg: AppConfig | None,
    meta: dict[str, Any],
    *,
    k2_mode: str,
    use_k2: bool,
    obs_group: str = "",
) -> tuple[str, float | None]:
    """Return (source_label literature|fit|none, optional k2 value) for the matrix."""
    if not use_k2:
        return "none", None
    k2_block = meta.get("k2") if isinstance(meta.get("k2"), dict) else {}
    src_raw = str(
        k2_block.get("k2_source")
        or k2_block.get("source")
        or meta.get("k2_source")
        or ""
    ).strip()
    val_raw = k2_block.get("k2_value", meta.get("k2_value", meta.get("k2_bprp")))
    if not src_raw and cfg is not None:
        try:
            from k2_extinction import resolve_k2_bprp_value  # noqa: PLC0415

            og = str(obs_group or meta.get("obs_group") or meta.get("setup") or "").strip()
            v, src_enum = resolve_k2_bprp_value(cfg, og)
            src_raw = str(getattr(src_enum, "value", src_enum) or "")
            val_raw = v
        except Exception:  # noqa: BLE001
            src_raw = k2_mode
    src_l = src_raw.lower()
    if "fit" in src_l:
        label = "fit"
    elif src_l in ("none", "off", ""):
        label = "literature" if k2_mode not in ("off", "none") else "none"
        if src_l in ("none", "off", "") and k2_mode == "literature":
            label = "literature"
    elif "lit" in src_l or src_l == "literature_default":
        label = "literature"
    else:
        label = src_l or ("literature" if k2_mode == "literature" else k2_mode)
    val: float | None = None
    try:
        if val_raw is not None and str(val_raw).strip() != "":
            fv = float(val_raw)
            if fv == fv:  # not NaN
                val = fv
    except (TypeError, ValueError):
        val = None
    return label, val


def _method_citation_keys(ctx: RunCitationContext) -> list[str]:
    """Citations of the methods this run actually used (exports and PDF CORE)."""
    keys = [
        "stetson1987",  # DAOFIND detection
        "howell1989",  # CCD equation, aperture photometry error
        "merline1995",
        "young1967",  # scintillation
        "osborn2015",
        "broeg2005",  # ensemble weights
        "honeycutt1992",  # inhomogeneous ensemble
        "moffat1969",  # PSF wing for neighbour light in comp apertures
        "trujillo2001",
        "gaia2023",  # comparison magnitudes
        "gaiadr3doc",  # Gaia -> band relations
        "riello2021",
        "eastman2010",  # BJD_TDB
    ]
    _ct = str(ctx.color_term_mode or "off").strip().lower()
    if ctx.filter_transformable and _ct not in ("off", "0", "false", "no", "none", ""):
        keys.append("schwarz1978")
    if ctx.use_k2_literature:
        keys.extend(["smith2002", "jordi2010"])
    if ctx.use_airmass:
        keys.append("kastenyoung1989")
    if ctx.use_gs11:
        keys.append("seager2003")
    return keys


def _sections_for_context(ctx: RunCitationContext) -> list[tuple[str, list[str]]]:
    bib = load_citations_bib()
    sections: list[tuple[str, list[str]]] = []

    sections.append(("CORE", [citation_line(k, bib=bib) for k in _method_citation_keys(ctx)]))

    catalogs = ["lindegren2021"]
    if ctx.use_vsx:
        catalogs.append("watson2006")
    sections.append(("CATALOGS", [citation_line(k, bib=bib) for k in catalogs]))

    software = ["photutils", "astropy2022", "astroquery2019", "numpy2020", "scipy2020"]
    if ctx.use_lightkurve:
        software.append("lightkurve2018")
    sections.append(("SOFTWARE", [citation_line(k, bib=bib) for k in software]))

    if ctx.use_psf:
        sections.append(
            (
                "PSF PHOTOMETRY",
                [
                    citation_line("stetson1987", bib=bib),
                    citation_line("anderson2000", bib=bib),
                    citation_line("anderson2006", bib=bib),
                    citation_line("andersonking2006", bib=bib),
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

    if ctx.use_variability_envelope:
        sections.append(
            (
                "VARIABILITY DETECTION",
                [citation_line("aigrain2004", bib=bib)],
            )
        )

    # VAR-2026-10: the field variability search runs in every aperture photometry run (GUI and night run).
    # The trust gate (comp count, check star, lc quality) has no literature method to cite;
    # the Sokolovsky LOO comp QA was removed (COMP-QA-2026-10).
    sections.append(
        (
            "VARIABILITY SEARCH",
            [
                citation_line("sokolovsky2017", bib=bib),
                citation_line("vonneumann1941", bib=bib),
                citation_line("azzalini1985", bib=bib),
            ],
        )
    )

    # TRANSIENT-2026-10: new sources, minor bodies (only with transient_search_enabled)
    if ctx.use_transients:
        sections.append(
            (
                "TRANSIENT SEARCH",
                [
                    citation_line("hodgkin2021", bib=bib),
                    citation_line("sokolovsky2014", bib=bib),
                    citation_line("smith2020atlas", bib=bib),
                    citation_line("bowell1989", bib=bib),
                    citation_line("mpcorb", bib=bib),
                    citation_line("openngc", bib=bib),
                ],
            )
        )

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

    if ctx.use_hrd_extreme:
        sections.append(
            (
                "FIELD ASTROPHYSICS (HRD)",
                [
                    citation_line("pecaut2013", bib=bib),
                    citation_line("andrae2023", bib=bib),
                    citation_line("delchambre2023", bib=bib),
                    citation_line("creevey2023", bib=bib),
                    citation_line("lindegren2021", bib=bib),
                    citation_line("bailerjones2021", bib=bib),
                ],
            )
        )

    return sections


def build_methods_matrix_lines(ctx: RunCitationContext) -> list[str]:
    """ASCII methods matrix of the CURRENT pipeline (EXPORT-2026-10).

    Every line states what the pipeline does in this run; nothing is listed that the
    code does not do.
    """
    _kg = ctx.k_grid or ()
    _kg_s = f"{min(_kg):.2f}..{max(_kg):.2f}" if _kg else "grid"
    ct_mode = str(ctx.color_term_mode or "off").strip().lower()
    ct_cfg_on = ct_mode not in ("off", "0", "false", "no", "none", "")
    if not ctx.filter_transformable:
        ct_state = "OFF (filter reported untransformed)"
    elif ct_cfg_on:
        ct_state = "ON (polynomial in BP-RP, order by BIC; 0 = none needed)"
    else:
        ct_state = "OFF (config)"
    if ctx.use_k2_literature:
        if ctx.k2_value is not None:
            k2_state = f"ON ({ctx.k2_source_label}, {float(ctx.k2_value):.3f} mag/airmass/BP-RP)"
        else:
            k2_state = f"ON ({ctx.k2_source_label})"
    else:
        k2_state = f"OFF ({ctx.k2_off_reason})" if ctx.k2_off_reason else "OFF"
    a_in, a_out = ctx.annulus_fwhm
    pairs = [
        ("aperture photometry", f"ON (r = k x FWHM, k per target from {_kg_s})"),
        ("local sky", f"ON (median of annulus {a_in:.1f}-{a_out:.1f} FWHM, no clipping)"),
        ("zero point", f"Gaia DR3 -> {ctx.zp_band_label or 'observed band'}"),
        (
            "comparison stars",
            f"{ctx.n_comp_min}-{ctx.n_comp_max} per target; rejected: >= 0.90 SAT, "
            "neighbour light > 1%, no valid band magnitude",
        ),
        ("ensemble weights", "ON (Broeg 2005 iterative)"),
        ("colour term", ct_state),
        ("second-order extinction k''", k2_state),
        ("target dilution correction", "ON" if ctx.use_gs11 else "OFF"),
        ("faint limit", f"SNR >= {ctx.snr_min:g} (epochs below are not exported)"),
        ("sigma clipping / detrending / binning", "OFF"),
        ("check star", "ON (independent of the ensemble)"),
        ("trust gate", "ON"),
        ("PSF photometry", "ON" if ctx.use_psf else "OFF"),
    ]
    if ctx.osc_channel_export:
        _bin = ctx.osc_channel_binning if ctx.osc_channel_binning is not None else "?"
        pairs.append(("OSC channel extraction", "ON"))
        pairs.append(("OSC channel binning", f"{_bin}x{_bin} average"))
        if ctx.osc_transform_citation:
            pairs.append(("OSC Gaia->Johnson comps", f"ON ({ctx.osc_transform_citation})"))
    return [f"{name}: {state}" for name, state in pairs]


def emit_export_citation_lines(
    ctx: RunCitationContext,
    *,
    comment_prefix: str = "#",
) -> list[str]:
    """Slim AAVSO / VarAstro comment header: matrix + ON-method citations only."""
    p = comment_prefix
    lines: list[str] = [
        f"{p}\n",
        f"{p} Pipeline: VYVAR - Automated Differential Photometry Pipeline\n",
        f"{p}\n",
        f"{p} METHODS MATRIX (this run):\n",
    ]
    for row in build_methods_matrix_lines(ctx):
        lines.append(f"{p}   {row}\n")
    lines.append(f"{p}\n")
    bib = load_citations_bib()
    lines.append(f"{p} [METHODS - this run]\n")
    for key in _method_citation_keys(ctx):
        lines.append(f"{p}   {plain_ascii_citation_text(citation_line(key, bib=bib))}\n")
    lines.append(f"{p}\n")
    lines.append(f"{p} Full algorithm description: SUMMARY MEASURE REPORT (PDF)\n")
    lines.append(f"{p}\n")
    return lines


def emit_pdf_methods_sections(ctx: RunCitationContext) -> list[tuple[str, list[str]]]:
    """PDF Methods section: matrix + full citation blocks."""
    out: list[tuple[str, list[str]]] = [
        (
            "Methods Matrix (This Run)",
            [f"  {row}" for row in build_methods_matrix_lines(ctx)],
        )
    ]
    for title, items in _sections_for_context(ctx):
        out.append(
            (
                title.replace(" - ", " - ").title(),
                [f"  {plain_ascii_citation_text(it)}" for it in items],
            )
        )
    return out


def emit_varastro_method_summary_lines(ctx: RunCitationContext) -> list[str]:
    """Short per-run algorithm lines for the VarAstro header."""
    bib = load_citations_bib()
    lines: list[str] = [
        "# PHOTOMETRY: aperture, ensemble of Gaia DR3 comparison stars | "
        f"{citation_line('broeg2005', bib=bib).strip()}; "
        f"{citation_line('honeycutt1992', bib=bib).strip()}\n",
    ]
    if ctx.use_psf:
        lines.append(f"# ALG: ePSF - {citation_line('anderson2000', bib=bib)}\n")
    return lines
