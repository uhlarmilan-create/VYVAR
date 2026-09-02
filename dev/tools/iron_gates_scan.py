"""Static iron-rule scanner for INV-NOCLIP / NOCOSMIC / PIXELS / MASTER (dev/tests).

Scope definitions live in docs/VYVAR_INVARIANTS.md. This module is check-only.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"

# INV-NOCLIP-01 / INV-NOCOSMIC-01 / INV-PIXELS-01 production science path.
PRODUCTION_SCOPE: frozenset[str] = frozenset(
    {
        "photometry_core.py",
        "pipeline.py",
        "comp_selection_per_target.py",
        "comp_frame_normalize.py",
        "comp_pool_rms.py",
        "comp_qa_core.py",
        "check_star_kmag.py",
        "calibration.py",
        "importer.py",
        "trust_flag_core.py",
        "method_lc_output.py",
        "export_reports.py",
        "sat_diag.py",
        "cal_diag.py",
        "cal_stage.py",
        "vyvar_alignment_frame.py",
        "psf_photometry.py",
        "psf_neighbor_sub.py",
        "plain_stats.py",
    }
)

# Deliberately outside production scope (diagnostics, offline harness, UI, optional analysis).
OUT_OF_SCOPE_MODULES: frozenset[str] = frozenset(
    {
        "xval_harness_core.py",
        "xval_run.py",
        "tess_verify.py",
        "validate_lc_crossval.py",
        "hrd_colorfield.py",
        "hrd_analysis.py",
        "variability_detector.py",
        "ui_settings.py",
        "ui_aperture_photometry.py",
    }
)

MASTER_SCOPE: frozenset[str] = frozenset({"importer.py", "calibration.py", "pipeline.py"})

NOCLIP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # NOTE (SNR-GATE-01 / F5): INV-NOCLIP-01 deliberately does NOT match detection floors of
    # the form ``median + k * sigma`` (DAOStarFinder threshold, prematch peak gate). Those are
    # significance cuts, not iterative kappa-sigma rejection of science samples. A broken
    # *noise estimator* feeding such a gate is a separate defect (scene std vs sky MAD).
    (
        "one_sided_annulus_sky_clip",
        re.compile(r"sky_pixels\s*\[\s*sky_pixels\s*<\s*sky_med", re.MULTILINE),
    ),
    (
        "astropy_sigma_clip_import",
        re.compile(r"\bfrom\s+astropy\.stats\s+import\s+.*sigma_clip", re.MULTILINE),
    ),
    (
        "sigma_clip_call",
        re.compile(r"\bsigma_clip\s*\(", re.MULTILINE),
    ),
    (
        "sigma_clipped_stats_call",
        re.compile(r"\bsigma_clipped_stats\s*\(", re.MULTILINE),
    ),
    (
        "SigmaClip_ctor",
        re.compile(r"\bSigmaClip\s*\(", re.MULTILINE),
    ),
    (
        "sclip_std_helper",
        re.compile(r"\bdef\s+sclip_std\s*\(", re.MULTILINE),
    ),
    (
        "iterative_ensemble_clip_active",
        re.compile(r"def\s+_iterative_ensemble_clip_cm_residual\s*\(", re.MULTILINE),
    ),
)

NOCOSMIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("lacosmic", re.compile(r"\blacosmic\b", re.IGNORECASE)),
    ("LaCosmic", re.compile(r"\bLaCosmic\b")),
    ("cosmic_ray_clean", re.compile(r"cosmic.?ray.?clean", re.IGNORECASE)),
    ("astroscrappy", re.compile(r"\bastroscrappy\b", re.IGNORECASE)),
)

PIXELS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "nanmedian_fill_before_phot",
        re.compile(
            r"np\.where\s*\(\s*np\.isfinite\s*\(\s*d\s*\)\s*,\s*d\s*,\s*fill\s*\)",
            re.MULTILINE,
        ),
    ),
)

MASTER_BAD_COMBINE: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ccdproc_combiner", re.compile(r"\bccdproc\.(?:Combiner|combine)", re.MULTILINE)),
    ("sigma_clip_on_stack", re.compile(r"combine.*sigma|sigma.*combine", re.IGNORECASE)),
)


@dataclass(frozen=True)
class Violation:
    rule_id: str
    module: str
    line: int
    kind: str
    snippet: str


def _line_snippet(text: str, line_no: int) -> str:
    lines = text.splitlines()
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].strip()[:120]
    return ""


def _scan_patterns(
    rule_id: str,
    rel: str,
    text: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> list[Violation]:
    out: list[Violation] = []
    for kind, pat in patterns:
        for m in pat.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            out.append(
                Violation(
                    rule_id=rule_id,
                    module=rel,
                    line=line,
                    kind=kind,
                    snippet=_line_snippet(text, line),
                )
            )
    return out


def _fn_uses_clip_sigma_actively(fn: ast.FunctionDef) -> bool:
    """True if clip_sigma participates in a comparison or arithmetic (not a discard)."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if "clip_sigma" in names:
                return True
        if isinstance(node, ast.BinOp):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            if "clip_sigma" in names:
                return True
        if isinstance(node, (ast.For, ast.While)):
            return True
    return False


def _ensemble_clip_fn_is_passthrough(text: str) -> bool:
    """INV-NOCLIP-01: stub if AST body never uses clip_sigma and has no loop."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    fn = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_iterative_ensemble_clip_cm_residual"),
        None,
    )
    if fn is None:
        return False
    return not _fn_uses_clip_sigma_actively(fn)


def scan_noclip(scope: frozenset[str] | None = None) -> list[Violation]:
    scope = scope or PRODUCTION_SCOPE
    hits: list[Violation] = []
    for rel in sorted(scope):
        path = SRC / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for v in _scan_patterns("INV-NOCLIP-01", rel, text, NOCLIP_PATTERNS):
            if v.kind == "iterative_ensemble_clip_active" and _ensemble_clip_fn_is_passthrough(text):
                continue
            hits.append(v)
    return hits


def scan_nocosmic(scope: frozenset[str] | None = None) -> list[Violation]:
    scope = scope or PRODUCTION_SCOPE
    hits: list[Violation] = []
    for rel in sorted(scope):
        path = SRC / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits.extend(_scan_patterns("INV-NOCOSMIC-01", rel, text, NOCOSMIC_PATTERNS))
    return hits


def scan_pixels(scope: frozenset[str] | None = None) -> list[Violation]:
    scope = scope or PRODUCTION_SCOPE
    hits: list[Violation] = []
    for rel in sorted(scope):
        path = SRC / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits.extend(_scan_patterns("INV-PIXELS-01", rel, text, PIXELS_PATTERNS))
    return hits


def scan_master(scope: frozenset[str] | None = None) -> list[Violation]:
    scope = scope or MASTER_SCOPE
    hits: list[Violation] = []
    for rel in sorted(scope):
        path = SRC / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits.extend(_scan_patterns("INV-MASTER-01", rel, text, MASTER_BAD_COMBINE))
        if rel == "importer.py" and "_combine_stack_mean" not in text:
            hits.append(
                Violation(
                    rule_id="INV-MASTER-01",
                    module=rel,
                    line=0,
                    kind="missing_plain_combine_helpers",
                    snippet="_combine_stack_mean not found",
                )
            )
    return hits


def check_comp_membership_ensemble_normalize() -> list[Violation]:
    """INV-COMP-MEMBERSHIP: good_ids selected outside per-frame loop."""
    path = SRC / "photometry_lightcurve.py"
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text, filename=str(path))
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ensemble_normalize"),
        None,
    )
    if fn is None:
        return [
            Violation(
                rule_id="INV-COMP-MEMBERSHIP",
                module="photometry_lightcurve.py",
                line=0,
                kind="missing_function",
                snippet="ensemble_normalize not found",
            )
        ]
    src = ast.get_source_segment(text, fn) or ""
    hits: list[Violation] = []
    if "for i in range(n_frames):" in src and "good_ids = selected" not in src:
        hits.append(
            Violation(
                rule_id="INV-COMP-MEMBERSHIP",
                module="photometry_lightcurve.py",
                line=fn.lineno,
                kind="good_ids_not_before_frame_loop",
                snippet="good_ids assignment missing before frame loop",
            )
        )
    if re.search(r"for\s+i\s+in\s+range\s*\(\s*n_frames\s*\).*?selected\s*=\s*\[", src, re.DOTALL):
        hits.append(
            Violation(
                rule_id="INV-COMP-MEMBERSHIP",
                module="photometry_lightcurve.py",
                line=fn.lineno,
                kind="per_frame_selection",
                snippet="selected recomputed inside frame loop",
            )
        )
    banned = re.search(r"ZP.*MAD.*clip|sigma.?clip.*zp|clip.*per.?frame.*comp", src, re.IGNORECASE)
    if banned and "ZP MAD clip removed" not in src:
        hits.append(
            Violation(
                rule_id="INV-COMP-MEMBERSHIP",
                module="photometry_lightcurve.py",
                line=fn.lineno,
                kind="per_frame_zp_clip",
                snippet="per-frame ZP clip pattern",
            )
        )
    return hits


def scan_source_text(text: str, module: str = "<fixture>") -> list[Violation]:
    """Scan arbitrary source (fire-proof fixtures)."""
    hits: list[Violation] = []
    hits.extend(_scan_patterns("INV-NOCLIP-01", module, text, NOCLIP_PATTERNS))
    return hits
