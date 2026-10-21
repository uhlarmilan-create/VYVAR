"""Iron-rule wired gates (INV-NOCLIP-01, INV-NOCOSMIC-01, INV-PIXELS-01, INV-MASTER-01, INV-COMP-MEMBERSHIP)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dev" / "tools"))

from iron_gates_scan import (  # noqa: E402
    check_comp_membership_ensemble_normalize,
    scan_master,
    scan_noclip,
    scan_nocosmic,
    scan_pixels,
    scan_source_text,
)


def test_inv_noclip01_fire_proof_detects_annulus_clip() -> None:
    fixture = '''
def _sky_pp_from_annulus_image(d, ann_img):
    sky_pixels = d[ann_img > 0]
    sky_med = float(np.median(sky_pixels))
    sky_std = float(np.std(sky_pixels))
    clipped = sky_pixels[sky_pixels < sky_med + 2.0 * sky_std]
    return float(np.median(clipped))
'''
    hits = scan_source_text(fixture)
    kinds = {h.kind for h in hits}
    assert "one_sided_annulus_sky_clip" in kinds


def test_inv_noclip01_production_scope_clean() -> None:
    hits = scan_noclip()
    assert hits == [], "INV-NOCLIP-01 violations:\n" + "\n".join(
        f"{v.module}:{v.line} {v.kind} {v.snippet}" for v in hits
    )


def test_inv_noclip01_ensemble_stub_excluded_by_ast_not_comment() -> None:
    stub = '''
def _iterative_ensemble_clip_cm_residual(flux_map, bjd_map, provisional_rms, *, clip_sigma, n_comp_min, max_iter=5, min_final=None):
    _ = (flux_map, bjd_map, clip_sigma, n_comp_min, max_iter)
    return provisional_rms, {"comp_pool_n_clipped": 0}
'''
    from iron_gates_scan import _ensemble_clip_fn_is_passthrough, _scan_patterns, NOCLIP_PATTERNS

    assert _ensemble_clip_fn_is_passthrough(stub) is True
    hits = _scan_patterns("INV-NOCLIP-01", "comp_selection_per_target.py", stub, NOCLIP_PATTERNS)
    kinds = {h.kind for h in hits}
    assert "iterative_ensemble_clip_active" in kinds


def test_inv_noclip01_ensemble_body_with_loop_is_not_passthrough() -> None:
    active = '''
def _iterative_ensemble_clip_cm_residual(flux_map, bjd_map, provisional_rms, *, clip_sigma, n_comp_min, max_iter=5, min_final=None):
    """Passthrough: no ensemble sigma-clip"""
    out = dict(provisional_rms)
    for cid, rms in list(out.items()):
        if rms > clip_sigma:
            del out[cid]
    return out, {"comp_pool_n_clipped": 1}
'''
    from iron_gates_scan import _ensemble_clip_fn_is_passthrough

    assert _ensemble_clip_fn_is_passthrough(active) is False


def test_inv_nocosmic01_production_scope_clean() -> None:
    hits = scan_nocosmic()
    assert hits == [], "INV-NOCOSMIC-01 violations:\n" + "\n".join(
        f"{v.module}:{v.line} {v.kind}" for v in hits
    )


def test_inv_pixels01_known_sites_only() -> None:
    """INV-PIXELS-01: records nanmedian fill sites; Milan adjudication pending."""
    hits = scan_pixels()
    allowed = {
        ("photometry_core.py", "nanmedian_fill_before_phot"),
        ("pipeline.py", "nanmedian_fill_before_phot"),
        ("psf_photometry.py", "nanmedian_fill_before_phot"),
    }
    for v in hits:
        assert (v.module, v.kind) in allowed


def test_inv_master01_plain_combine_only() -> None:
    hits = scan_master()
    assert hits == [], "INV-MASTER-01 violations:\n" + "\n".join(
        f"{v.module}:{v.line} {v.kind}" for v in hits
    )


def test_inv_comp_membership_ensemble_normalize() -> None:
    hits = check_comp_membership_ensemble_normalize()
    assert hits == [], "INV-COMP-MEMBERSHIP violations:\n" + "\n".join(
        f"{v.kind} {v.snippet}" for v in hits
    )
